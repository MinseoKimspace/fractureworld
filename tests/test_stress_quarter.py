import hashlib
import json

import numpy as np
import pytest
import torch

from fracture_world.cases import Case
from fracture_world.equilibrium import EquilibriumDynamicsModel
from fracture_world.force_reference2d import teacher_force_rollout
from fracture_world.stress_evaluation import (
    case_at_time,
    evaluate_stress,
    orientation_metrics,
    prediction_metrics,
    resample_prediction,
)
from fracture_world.teacher2d import Material, Pull, simulate
from fracture_world.topology import rectangle, split_edges


def stationary_source():
    mesh = rectangle(2, 2)
    time = np.linspace(0.0, 0.02, 11)
    mask, position, velocity = Pull(0.0, 0.02).boundary(mesh, time)
    return Case({
        "rest": mesh.rest.copy(), "triangles": mesh.triangles.copy(),
        "original_vertex": mesh.original_vertex.copy(), "cut_edges": mesh.cut_edges.copy(),
        "time": time, "position": position.copy(), "velocity": velocity.copy(),
        "boundary_mask": mask, "boundary_position": position.copy(),
        "boundary_velocity": velocity.copy(),
        "boundary_impulse": np.zeros((len(time) - 1, *mesh.rest.shape)),
    }, {"material": Material(young=1.0).metadata(), "pull": {"amplitude": 0.0, "ramp_time": 0.02}})


def test_quarter_substeps_sum_four_impulses_and_average_four_forces():
    time = np.linspace(0.0, 0.004, 9)
    state = np.broadcast_to(time[:, None, None], (9, 2, 2)).copy()
    values = np.broadcast_to(np.arange(1, 9)[:, None, None], (8, 2, 2)).copy()
    result = resample_prediction({
        "time": time, "position": state, "velocity": 2 * state,
        "boundary_impulse": values, "internal_force": 2 * values,
    }, 4, time[::4])
    np.testing.assert_array_equal(result["position"], state[::4])
    np.testing.assert_array_equal(result["boundary_impulse"][:, 0, 0], [10, 26])
    np.testing.assert_array_equal(result["internal_force"][:, 0, 0], [5, 13])


def test_quarter_geometry_catches_an_invalid_native_step_hidden_by_alignment():
    source = stationary_source()
    reference = case_at_time(source, np.linspace(0.0, 0.004, 3))
    reference.arrays["boundary_impulse"] = np.zeros((2, *reference.mesh.rest.shape))
    time = np.linspace(0.0, 0.004, 9)
    position = np.broadcast_to(reference.mesh.rest, (len(time), *reference.mesh.rest.shape)).copy()
    position[3, 4, 1] = -0.5
    native = {
        "time": time, "position": position, "velocity": np.zeros_like(position),
        "boundary_impulse": np.zeros((8, *reference.mesh.rest.shape)),
    }
    geometry = orientation_metrics(reference.mesh, native)
    aligned = resample_prediction(native, 4, reference.arrays["time"])
    result = prediction_metrics(reference, aligned, {"status": "completed", "runtime_seconds": 0}, geometry)
    assert result["aligned_geometry"]["orientation_valid"] is True
    assert result["orientation_valid"] is False
    assert result["first_invalid_orientation_time"] == 0.0015


def test_quarter_teacher_force_converges_without_replaying_reference_forces():
    pytest.importorskip("felupe")
    mesh = split_edges(rectangle(2, 2), [[1, 4]])
    source = simulate(mesh, Material(young=1.0), np.linspace(0, 0.02, 11),
                      amplitude=0.002, ramp_time=0.02, rtol=1e-11, atol=1e-13, max_step=0.000125)
    errors = []
    for factor in (1, 2, 4):
        time = np.linspace(0, 0.02, 10 * factor + 1)
        prediction, status = teacher_force_rollout(source, time)
        assert status["completed_steps"] == 10 * factor
        aligned = resample_prediction(prediction, factor, source.arrays["time"])
        errors.append(float(np.sqrt(np.mean((aligned["velocity"] - source.arrays["velocity"]) ** 2))))
    assert errors[1] < errors[0] / 2.5
    assert errors[2] < errors[1] / 2.5


def test_frozen_quarter_evaluation_runs_2000_steps_and_preserves_inputs(tmp_path, monkeypatch):
    torch.set_num_threads(2)
    source_case = stationary_source()
    source = tmp_path / "source.npz"
    source_case.save(source)
    manifest = tmp_path / "collection.json"
    manifest.write_text(json.dumps({
        "format": "cut-dynamics-a-v1", "status": "completed", "cases": [{
            "id": "held-out", "group": "held-out", "split": "test", "path": source.name,
            "case_hash": str(source_case.arrays["case_hash"].item()),
        }],
    }), encoding="utf-8")
    model = EquilibriumDynamicsModel("stress", hidden_size=8, layers=1)
    checkpoint = tmp_path / "best.pt"
    torch.save({
        "format": "cut-dynamics-a-v2", "trained": True, "model_config": model.configuration(),
        "state_dict": model.state_dict(), "dt": 0.002, "training_case_hashes": [], "training_groups": [],
    }, checkpoint)
    protected = {p: hashlib.sha256(p.read_bytes()).hexdigest() for p in (source, manifest, checkpoint)}
    plan = [{
        "id": "long-held-out", "category": "long_rollout", "mesh": source_case.mesh,
        "material": Material(young=1.0), "load": source_case.metadata["pull"],
        "duration": 1.0, "time_step": 0.002, "source_case": "held-out",
        "source_case_hash": str(source_case.arrays["case_hash"].item()),
        "source_group": "held-out", "source_path": source, "group": "held-out", "topology": "source",
        "novelty": {"topology": "unchanged", "load": "unchanged", "horizon": "extended"},
        "novelty_flags": {"topology": False, "load": False, "horizon": True},
    }]

    def reference(mesh, material, time, load):
        case = case_at_time(source_case, time)
        case.arrays["boundary_impulse"] = np.zeros((len(time) - 1, *mesh.rest.shape))
        free = ~case.arrays["boundary_mask"]
        case.arrays["position"][1:, free] += 0.01
        case.arrays["velocity"][1:, free] += 0.02
        return case, {"status": "completed", "converged": True}

    oracle_times = []

    def oracle(case, time):
        oracle_times.append(len(time))
        inputs = case_at_time(source_case, time)
        return {
            "time": time.copy(), "position": inputs.arrays["position"].copy(),
            "velocity": inputs.arrays["velocity"].copy(),
            "boundary_impulse": np.zeros((len(time) - 1, *case.mesh.rest.shape)),
        }, {
            "status": "completed", "failure": None, "completed_steps": len(time) - 1,
            "requested_steps": len(time) - 1, "runtime_seconds": 0.0, "precision": "float64",
        }

    def force_diagnostic(*args, **kwargs):
        return {"status": "available", "sample_count": 1, "force_rmse": 0.0}

    def forbidden(*args, **kwargs):
        raise AssertionError("Frozen evaluation cannot train or save weights")

    monkeypatch.setattr("fracture_world.stress_evaluation.evaluation_plan", lambda *args: plan)
    monkeypatch.setattr("fracture_world.stress_evaluation.reference_solution", reference)
    monkeypatch.setattr("fracture_world.stress_evaluation.teacher_force_rollout", oracle)
    monkeypatch.setattr("fracture_world.stress_evaluation.instantaneous_force_error", force_diagnostic)
    monkeypatch.setattr(torch.optim.Optimizer, "__init__", forbidden)
    monkeypatch.setattr(torch, "save", forbidden)
    output = tmp_path / "quarter-evaluation"
    result = evaluate_stress(checkpoint, manifest, output, groups=["long"], substeps=(1, 2, 4))
    assert result["inputs_unchanged"] is True
    assert result["training_executed"] is False
    report = json.loads((output / "evaluation.json").read_text(encoding="utf-8"))
    row = report["cases"][0]
    assert report["substeps"] == [1, 2, 4]
    assert oracle_times == [501, 1001, 2001]
    assert set(row["methods"]) == {
        "neural_verlet", "neural_verlet_half_dt", "neural_verlet_quarter_dt",
        "teacher_verlet", "teacher_verlet_half_dt", "teacher_verlet_quarter_dt",
    }
    assert set(row["instantaneous_force_diagnostics_by_method"]) == {
        "neural_verlet", "neural_verlet_half_dt", "neural_verlet_quarter_dt",
    }
    assert row["instantaneous_force_diagnostics"]["on_reference_states"]["status"] == "available"
    assert set(row["force_provider_difference"]) == {"dt", "half_dt", "quarter_dt"}
    for name, details in row["methods"].items():
        assert details["status"] == "completed"
        assert details["aligned_completed_steps"] == 500
        expected = 2000 if name.endswith("quarter_dt") else 1000 if name.endswith("half_dt") else 500
        assert details["completed_steps"] == expected
        prediction = Case.load(output / details["trajectory"])
        assert len(prediction.arrays["time"]) == 501
        np.testing.assert_array_equal(prediction.arrays["position"], np.broadcast_to(source_case.mesh.rest, prediction.arrays["position"].shape))
        if name.startswith("neural"):
            assert details["teacher_calls_in_rollout"] == 0
            assert [point["requested_time"] for point in details["time_checkpoints"]] == [0.2, 0.5, 1.0]
            assert all(point["reached"] for point in details["time_checkpoints"])
    assert {p: hashlib.sha256(p.read_bytes()).hexdigest() for p in protected} == protected
