import hashlib
import json

import numpy as np
import pytest
import torch

from fracture_world.cases import Case
from fracture_world.equilibrium import EquilibriumDynamicsModel
from fracture_world.stress_evaluation import evaluate_stress
from fracture_world.teacher2d import Material, simulate
from fracture_world.topology import rectangle, split_edges


@pytest.fixture
def frozen_inputs(tmp_path):
    pytest.importorskip("felupe")
    torch.set_num_threads(2)
    mesh = split_edges(rectangle(2, 2), [[1, 4]])
    material = Material(young=1.0)
    case = simulate(mesh, material, np.linspace(0, 0.02, 11),
                    amplitude=0.002, ramp_time=0.02, max_step=0.0005)
    source = tmp_path / "source.npz"
    case.save(source)
    manifest = tmp_path / "collection.json"
    manifest.write_text(json.dumps({
        "format": "cut-dynamics-a-v1", "status": "completed",
        "cases": [{"id": "held-out", "group": "held-out", "split": "test",
                   "path": source.name, "case_hash": str(case.arrays["case_hash"].item())}],
    }), encoding="utf-8")
    model = EquilibriumDynamicsModel("stress", hidden_size=8, layers=1)
    checkpoint = tmp_path / "best.pt"
    torch.save({
        "format": "cut-dynamics-a-v2", "model_config": model.configuration(),
        "state_dict": model.state_dict(), "trained": True, "dt": 0.002,
        "training_case_hashes": [], "training_groups": [],
    }, checkpoint)
    return manifest, source, checkpoint, case, material


def scenario(source, case, material, category="force_audit", duration=0.02):
    return {
        "id": f"{category}-held-out", "category": category,
        "mesh": case.mesh, "material": material, "load": case.metadata["pull"],
        "duration": duration, "time_step": 0.002, "source_case": "held-out",
        "source_case_hash": str(case.arrays["case_hash"].item()),
        "source_group": "held-out", "source_path": source,
        "group": "held-out", "topology": "source",
        "novelty": {"topology": "unchanged_source_topology", "load": "unchanged_source_load",
                    "horizon": "extended" if duration > 0.02 else "original"},
        "novelty_flags": {},
    }


def protect_weights(monkeypatch):
    def forbidden(*args, **kwargs):
        raise AssertionError("Frozen evaluation cannot create an optimizer or save weights")

    monkeypatch.setattr(torch.optim.Optimizer, "__init__", forbidden)
    monkeypatch.setattr(torch, "save", forbidden)


def digests(paths):
    return {path: hashlib.sha256(path.read_bytes()).hexdigest() for path in paths}


def test_force_comparison_runs_all_four_frozen_paths_and_preserves_inputs(frozen_inputs, tmp_path, monkeypatch):
    manifest, source, checkpoint, case, material = frozen_inputs
    plan = [scenario(source, case, material)]
    monkeypatch.setattr("fracture_world.stress_evaluation.evaluation_plan", lambda *args: plan)
    protected = digests([manifest, source, checkpoint])
    protect_weights(monkeypatch)
    output = tmp_path / "evaluation"
    result = evaluate_stress(checkpoint, manifest, output, groups=["force"], render_results=True)
    assert result["status"] == "completed"
    assert result["training_executed"] is False
    assert result["inputs_unchanged"] is True
    assert digests([manifest, source, checkpoint]) == protected
    report = json.loads((output / "evaluation.json").read_text(encoding="utf-8"))
    row = report["cases"][0]
    animation = output / row["render"]
    assert animation.read_bytes().startswith(b"GIF")
    assert animation.with_suffix(".png").read_bytes().startswith(b"\x89PNG")
    assert set(row["methods"]) == {
        "neural_verlet", "neural_verlet_half_dt", "teacher_verlet", "teacher_verlet_half_dt",
    }
    reference = Case.load(output / row["reference_trajectory"])
    for name, details in row["methods"].items():
        assert details["status"] == "completed"
        assert details["aligned_completed_steps"] == 10
        assert details["metric_status"] == "available"
        prediction = Case.load(output / details["trajectory"])
        assert np.array_equal(prediction.arrays["time"], reference.arrays["time"])
        assert prediction.arrays["boundary_impulse"].shape == (10, len(case.mesh.rest), 2)
        if name.startswith("neural"):
            assert details["teacher_calls_in_rollout"] == 0
    for key in ("on_reference_states", "on_neural_states"):
        assert row["instantaneous_force_diagnostics"][key]["status"] == "available"
    coarse = row["methods"]["teacher_verlet"]["free_motion"]["velocity_rmse"]
    fine = row["methods"]["teacher_verlet_half_dt"]["free_motion"]["velocity_rmse"]
    assert fine < coarse / 2


def test_one_second_evaluation_uses_500_autonomous_steps(frozen_inputs, tmp_path, monkeypatch):
    manifest, source, checkpoint, case, material = frozen_inputs
    plan = [scenario(source, case, material, "long_rollout", 1.0)]
    monkeypatch.setattr("fracture_world.stress_evaluation.evaluation_plan", lambda *args: plan)
    protect_weights(monkeypatch)
    output = tmp_path / "long-evaluation"
    result = evaluate_stress(checkpoint, manifest, output, groups=["long"])
    assert result["status"] == "completed"
    assert result["inputs_unchanged"] is True
    report = json.loads((output / "evaluation.json").read_text(encoding="utf-8"))
    row = report["cases"][0]
    for name in ("neural_verlet", "teacher_verlet", "neural_verlet_half_dt", "teacher_verlet_half_dt"):
        method = row["methods"][name]
        assert method["status"] == "completed"
        assert method["aligned_completed_steps"] == 500
        assert method["aligned_completed_horizon"] == 1.0
        assert method["time_checkpoints"][-1]["reached"] is True
        assert method["time_checkpoints"][-1]["time"] == 1.0
        assert method["completed_steps"] == (1000 if name.endswith("half_dt") else 500)
        saved = Case.load(output / method["trajectory"])
        assert len(saved.arrays["time"]) == 501
        assert np.array_equal(saved.arrays["position"][0], case.arrays["position"][0])


def test_evaluation_rejects_modified_source_initial_state(frozen_inputs, tmp_path, monkeypatch):
    manifest, source, checkpoint, case, material = frozen_inputs
    arrays = {key: value.copy() for key, value in case.arrays.items()}
    del arrays["case_hash"]
    arrays["position"][0, ~arrays["boundary_mask"]] += 0.001
    changed = Case(arrays, dict(case.metadata))
    changed.save(source)
    plan = [scenario(source, changed, material)]
    monkeypatch.setattr("fracture_world.stress_evaluation.evaluation_plan", lambda *args: plan)
    protect_weights(monkeypatch)
    with pytest.raises(ValueError, match="source rest state"):
        evaluate_stress(checkpoint, manifest, tmp_path / "invalid", groups=["force"])
    assert not (tmp_path / "invalid").exists()


def test_topology_case_limit_keeps_one_novel_case_and_its_control(frozen_inputs, tmp_path, monkeypatch):
    manifest, source, checkpoint, case, material = frozen_inputs
    control = scenario(source, case, material, "control")
    first = scenario(source, case, material, "topology")
    second = {**first, "id": "topology-second"}
    monkeypatch.setattr("fracture_world.stress_evaluation.evaluation_plan", lambda *args: [control, first, second])

    def unavailable(*args, **kwargs):
        raise RuntimeError("Injected unavailable numerical reference")

    monkeypatch.setattr("fracture_world.stress_evaluation.reference_solution", unavailable)
    protect_weights(monkeypatch)
    result = evaluate_stress(checkpoint, manifest, tmp_path / "limited", groups=["topology"], case_limit=1)
    report = json.loads((tmp_path / "limited/evaluation.json").read_text(encoding="utf-8"))
    assert result["scenarios"] == 2
    assert {row["id"] for row in report["plan"]["scenarios"]} == {control["id"], first["id"]}
    assert result["reference_failed"] == 2
    assert result["neural_completed"] == 0
    assert result["inputs_unchanged"] is True


def test_unverified_reference_stays_unknown_and_skips_rollouts(frozen_inputs, tmp_path, monkeypatch):
    manifest, source, checkpoint, case, material = frozen_inputs
    plan = [scenario(source, case, material)]
    monkeypatch.setattr("fracture_world.stress_evaluation.evaluation_plan", lambda *args: plan)
    monkeypatch.setattr(
        "fracture_world.stress_evaluation.reference_solution",
        lambda *args: (case, {"status": "completed", "converged": False}),
    )

    def forbidden(*args, **kwargs):
        raise AssertionError("Unverified references cannot be used to claim rollout accuracy")

    monkeypatch.setattr("fracture_world.stress_evaluation.rollout", forbidden)
    monkeypatch.setattr("fracture_world.stress_evaluation.teacher_force_rollout", forbidden)
    protect_weights(monkeypatch)
    result = evaluate_stress(checkpoint, manifest, tmp_path / "unknown", groups=["force"])
    report = json.loads((tmp_path / "unknown/evaluation.json").read_text(encoding="utf-8"))
    assert result["reference_unverified"] == 1
    assert result["neural_completed"] == 0
    assert result["neural_orientation_valid"] == 0
    assert report["cases"][0]["status"] == "reference_unverified"
    assert report["cases"][0]["methods"] == {}
    assert report["summary"]["force_audit"]["reference_unavailable"] == 1
    assert result["inputs_unchanged"] is True
