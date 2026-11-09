import hashlib
import json
import sys
from pathlib import Path

import numpy as np
import pytest
import torch

import fracture_world.benchmark2d as evaluation
from fracture_world.cases import Case
from fracture_world.diagnostics2d import diagnose
from fracture_world.neural import DynamicsModel
from fracture_world.topology import rectangle, split_edges


def experiment(tmp_path):
    torch.set_num_threads(2)
    mesh = split_edges(rectangle(2, 2), [[1, 4], [4, 7]])
    time = np.arange(5) * 0.125
    rows = []
    for index, speed in enumerate((0.25, 0.5)):
        velocity = np.broadcast_to([speed, 0.0], (len(time), len(mesh.rest), 2)).copy()
        position = mesh.rest[None] + time[:, None, None] * velocity
        case = Case({
            "rest": mesh.rest, "triangles": mesh.triangles,
            "original_vertex": mesh.original_vertex, "cut_edges": mesh.cut_edges,
            "time": time, "position": position, "velocity": velocity,
            "boundary_mask": np.zeros_like(mesh.rest, dtype=bool),
            "boundary_position": position.copy(), "boundary_velocity": velocity.copy(),
            "boundary_impulse": np.zeros((len(time) - 1, len(mesh.rest), 2)),
        }, {"material": {"young": 1.0, "poisson": 0.25, "density": 1.0, "thickness": 1.0},
            "cut_depth": 2})
        path = tmp_path / f"case-{index}.npz"
        case.save(path)
        rows.append({"id": f"free-{index}", "group": "x1-depth2", "split": "test", "path": path.name})
    data = tmp_path / "collection.json"
    data.write_text(json.dumps({"format": "cut-dynamics-a-v1", "cases": rows}), encoding="utf-8")
    paths = []
    for method in ("direct", "stress"):
        model = DynamicsModel(method, hidden_size=8, layers=1)
        path = tmp_path / f"{method}.pt"
        torch.save({"format": "cut-dynamics-a-v1", "trained": True,
                    "model_config": model.configuration(), "state_dict": model.state_dict(),
                    "dt": 0.125, "training_case_hashes": [], "training_groups": ["training"]}, path)
        paths.append(path)
    evaluation.benchmark(data, paths, tmp_path / "benchmark")
    return data, tmp_path / "benchmark" / "report.json", paths


def read(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def test_real_diagnostics_reuse_saved_rollouts_without_training_or_engine(tmp_path, monkeypatch):
    data, report, checkpoints = experiment(tmp_path)
    protected = checkpoints + list((tmp_path / "benchmark").rglob("*.npz"))
    hashes = {path: hashlib.sha256(path.read_bytes()).hexdigest() for path in protected}

    def forbidden(*args, **kwargs):
        raise AssertionError("Training, checkpoint writes and autonomous reruns are forbidden")

    monkeypatch.setattr(evaluation, "rollout", forbidden)
    monkeypatch.setattr(torch.optim.Optimizer, "__init__", forbidden)
    monkeypatch.setattr(torch, "save", forbidden)
    monkeypatch.setitem(sys.modules, "felupe", None)
    result = diagnose(report, data, tmp_path / "diagnostics")
    saved = read(result["report"])
    assert saved["status"] == "completed"
    assert saved["training_executed"] is False
    assert saved["autonomous_rollout_recomputed"] is False
    assert len(saved["zero_internal_force"]["cases"]) == 2
    for method in ("direct", "stress"):
        values = saved["methods"][method]
        assert values["rest_probe_count"] == 1
        assert values["summary"]["attempted_cases"] == 2
        assert values["summary"]["saved_rollouts_completed"] == 2
        assert values["summary"]["teacher_forced_completed"] == 2
        assert values["checkpoint_identity_status"] == "original_checkpoint_checksum_unavailable"
        assert values["by_depth"]["2"]["attempted_cases"] == 2
        for case in values["cases"]:
            assert case["autonomous"]["free_motion"]["position_rmse"] == pytest.approx(0, abs=1e-12)
            assert case["teacher_forced"]["free_motion"]["velocity_rmse"] == pytest.approx(0, abs=1e-12)
            assert case["autonomous"]["momentum_balance"]["component_count"] == 2
    for path in protected:
        assert hashlib.sha256(path.read_bytes()).hexdigest() == hashes[path]


def test_case_limit_is_explicit_and_keeps_scope_counts(tmp_path):
    data, report, _ = experiment(tmp_path)
    result = diagnose(report, data, tmp_path / "diagnostics", case_limit=1)
    saved = read(result["report"])
    assert saved["selected_cases"] == 1
    assert saved["source_report_cases"] == 2
    assert all(value["summary"]["attempted_cases"] == 1 for value in saved["methods"].values())


def test_missing_saved_trajectory_is_a_recorded_failure(tmp_path):
    data, report, _ = experiment(tmp_path)
    source = read(report)
    relative = source["methods"]["direct"]["cases"][0]["trajectory"]
    (report.parent / relative).unlink()
    output = tmp_path / "diagnostics"
    with pytest.raises(ValueError, match="not found"):
        diagnose(report, data, output)
    failed = read(output / "diagnostics.json")
    assert failed["status"] == "failed"
    assert "not found" in failed["failure"]


def test_wrong_source_inputs_are_rejected(tmp_path):
    data, report, _ = experiment(tmp_path)
    source = read(report)
    source["methods"]["direct"]["cases"][0]["source_case_hash"] = "wrong"
    report.write_text(json.dumps(source), encoding="utf-8")
    with pytest.raises(ValueError, match="inputs differ"):
        diagnose(report, data, tmp_path / "diagnostics")


def test_checkpoint_checksum_mismatch_is_rejected(tmp_path):
    data, report, _ = experiment(tmp_path)
    source = read(report)
    source["methods"]["direct"]["checkpoint_sha256"] = "different"
    report.write_text(json.dumps(source), encoding="utf-8")
    with pytest.raises(ValueError, match="Checkpoint changed"):
        diagnose(report, data, tmp_path / "diagnostics")


def test_failed_prefix_remains_in_diagnostic_denominators(tmp_path):
    data, report, _ = experiment(tmp_path)
    source = read(report)
    record = source["methods"]["stress"]["cases"][0]
    path = report.parent / record["trajectory"]
    saved = Case.load(path)
    arrays = dict(saved.arrays)
    arrays.pop("case_hash")
    for name in ("position", "velocity", "time", "boundary_position", "boundary_velocity"):
        arrays[name] = arrays[name][:3]
    arrays["boundary_impulse"] = arrays["boundary_impulse"][:2]
    Case(arrays, dict(saved.metadata, status="failed")).save(path)
    record.update(status="failed", failure="Synthetic later failure", completed_steps=2)
    report.write_text(json.dumps(source), encoding="utf-8")
    result = diagnose(report, data, tmp_path / "diagnostics")
    values = read(result["report"])["methods"]["stress"]
    assert values["summary"]["attempted_cases"] == 2
    assert values["summary"]["saved_rollouts_completed"] == 1
    assert values["summary"]["saved_rollouts_failed"] == 1
    assert values["summary"]["saved_trajectories_available"] == 2
    assert values["summary"]["teacher_forced_completed"] == 2
    assert values["cases"][0]["autonomous"]["status"] == "available"
    assert not values["cases"][0]["improvement_over_zero_internal_force"]


def test_untrained_checkpoint_is_not_diagnosed_as_learned_model(tmp_path):
    data, report, paths = experiment(tmp_path)
    payload = torch.load(paths[0], weights_only=True)
    payload["trained"] = False
    torch.save(payload, paths[0])
    with pytest.raises(ValueError, match="trained checkpoint"):
        diagnose(report, data, tmp_path / "diagnostics")
