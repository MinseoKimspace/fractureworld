import json
import sys

import numpy as np
import pytest
import torch

import fracture_world.benchmark2d as evaluation
from fracture_world.cases import Case
from fracture_world.neural import DynamicsModel
from fracture_world.topology import rectangle


def inputs(tmp_path, group="test-cut"):
    mesh = rectangle(2, 2)
    time = np.arange(6) * 0.01
    velocity = np.broadcast_to([0.1, 0.02], (len(time), len(mesh.rest), 2)).copy()
    position = mesh.rest[None] + time[:, None, None] * velocity
    arrays = {
        "rest": mesh.rest, "triangles": mesh.triangles, "original_vertex": mesh.original_vertex,
        "cut_edges": mesh.cut_edges, "time": time, "position": position,
        "velocity": velocity, "boundary_mask": np.zeros_like(mesh.rest, dtype=bool),
        "boundary_position": position.copy(), "boundary_velocity": velocity.copy(),
        "boundary_impulse": np.zeros((len(time) - 1, len(mesh.rest), 2)),
    }
    case = Case(arrays, {"material": {"young": 1.0, "poisson": 0.25, "density": 1.0, "thickness": 1.0}})
    case.save(tmp_path / "case.npz")
    manifest = {"format": "cut-dynamics-a-v1", "cases": [
        {"id": "free-flight", "group": group, "split": "test", "path": "case.npz"}
    ]}
    data = tmp_path / "collection.json"
    data.write_text(json.dumps(manifest), encoding="utf-8")
    return case, data


def checkpoint(tmp_path, method="stress", dt=0.01):
    model = DynamicsModel(method, hidden_size=8, layers=1)
    payload = {
        "format": "cut-dynamics-a-v1", "model_config": model.configuration(),
        "state_dict": model.state_dict(), "dt": dt,
        "training_case_hashes": [], "training_groups": ["training-cut"],
    }
    path = tmp_path / f"{method}.pt"
    torch.save(payload, path)
    return path, payload


def test_full_autonomous_comparison_does_not_import_physics_engine(tmp_path, monkeypatch):
    case, data = inputs(tmp_path)
    paths = [checkpoint(tmp_path, method)[0] for method in ("direct", "stress")]
    monkeypatch.setitem(sys.modules, "felupe", None)
    report = evaluation.benchmark(data, paths, tmp_path / "results")
    assert (tmp_path / "results" / "report.json").is_file()
    for method in ("direct", "stress"):
        summary = report["methods"][method]["summary"]
        assert summary["completed_cases"] == 1
        assert summary["failed_cases"] == 0
        assert summary["mean_position_rmse_completed"] < 1e-6
        result = report["methods"][method]["cases"][0]
        assert result["runtime_seconds"] > 0
        assert result["completed_steps"] == 5
        assert result["orientation_valid"]
        assert "overlaps/contact not checked" in result["geometry_scope"]
        saved = Case.load(tmp_path / "results" / result["trajectory"])
        assert saved.metadata["source_case_hash"] == str(case.arrays["case_hash"].item())
        assert saved.arrays["case_hash"].item() == case.arrays["case_hash"].item()
        assert saved.arrays["position"].shape == case.arrays["position"].shape


@pytest.mark.parametrize("problem", ["timestep", "group", "case_hash"])
def test_training_overlap_and_mismatched_time_are_rejected(tmp_path, problem):
    case, data = inputs(tmp_path, "training-cut" if problem == "group" else "test-cut")
    path, payload = checkpoint(tmp_path, dt=0.02 if problem == "timestep" else 0.01)
    if problem == "case_hash":
        payload["training_case_hashes"] = [str(case.arrays["case_hash"].item())]
        torch.save(payload, path)
    with pytest.raises(ValueError, match="timestep|overlaps"):
        evaluation.benchmark(data, [path], tmp_path / "results")


class FailingModel(torch.nn.Module):
    def __init__(self, immediately=False):
        super().__init__()
        self.immediately = immediately

    def forward(self, graph, position, velocity, boundary_position, boundary_velocity, dt):
        next_position = position + dt * velocity
        if self.immediately or position[0, 0] > 0.0005:
            next_position = torch.full_like(position, float("nan"))
        return next_position, velocity, torch.zeros_like(position)


@pytest.mark.parametrize("immediately", [False, True])
def test_failed_rollouts_remain_in_completion_denominator(tmp_path, monkeypatch, immediately):
    _, data = inputs(tmp_path)
    path, payload = checkpoint(tmp_path)
    monkeypatch.setattr(evaluation, "load_model", lambda *args: (FailingModel(immediately), payload))
    report = evaluation.benchmark(data, [path], tmp_path / "results")
    method = report["methods"]["stress"]
    assert method["summary"]["attempted_cases"] == 1
    assert method["summary"]["completed_cases"] == 0
    assert method["summary"]["failed_cases"] == 1
    assert method["summary"]["completion_rate"] == 0
    assert method["summary"]["mean_position_rmse_completed"] is None
    result = method["cases"][0]
    assert "Non-finite prediction" in result["failure"]
    if immediately:
        assert result["completed_steps"] == 0
        assert "trajectory" not in result
    else:
        assert result["completed_steps"] == 1
        saved = Case.load(tmp_path / "results" / result["trajectory"])
        assert len(saved.arrays["time"]) == 2
        assert saved.metadata["status"] == "failed"


def test_future_teacher_motion_does_not_change_predictions(tmp_path):
    case, data = inputs(tmp_path)
    path, _ = checkpoint(tmp_path)
    first = evaluation.benchmark(data, [path], tmp_path / "first")
    case.arrays["position"][1:] += 0.1
    case.arrays["velocity"][1:] *= 4
    case.save(tmp_path / "case.npz")
    second = evaluation.benchmark(data, [path], tmp_path / "second")
    before = Case.load(tmp_path / "first" / first["methods"]["stress"]["cases"][0]["trajectory"])
    after = Case.load(tmp_path / "second" / second["methods"]["stress"]["cases"][0]["trajectory"])
    np.testing.assert_array_equal(before.arrays["position"], after.arrays["position"])
    np.testing.assert_array_equal(before.arrays["velocity"], after.arrays["velocity"])


def test_duplicate_method_labels_do_not_overwrite_results(tmp_path):
    _, data = inputs(tmp_path)
    path, _ = checkpoint(tmp_path)
    with pytest.raises(ValueError, match="Repeated method"):
        evaluation.benchmark(data, [path, path], tmp_path / "results")


def test_large_finite_prediction_metrics_use_float64(tmp_path):
    case, _ = inputs(tmp_path)
    position = np.full_like(case.arrays["position"], 1e30, dtype=np.float32)
    velocity = np.full_like(position, 1e30)
    result = evaluation.trajectory_metrics(case, position, velocity)
    assert np.isfinite(result["position_rmse"])
    assert np.isfinite(result["velocity_rmse"])
    assert np.isfinite(result["kinetic_energy_change"])
