import hashlib
import json
import sys
from pathlib import Path

import numpy as np
import pytest
import torch

from fracture_world.benchmark2d import benchmark, rollout
from fracture_world.cases import Case
from fracture_world.diagnostic_probes import rest_probe, teacher_forced
from fracture_world.diagnostics2d import diagnose
from fracture_world.neural import DynamicsModel, create_model
from fracture_world.topology import rectangle, split_edges
from fracture_world.training2d import graph_for, load_model


def sample(tmp_path):
    mesh = split_edges(rectangle(2, 2), [[1, 4], [4, 7]])
    time = np.arange(5) * 0.001
    position = np.broadcast_to(mesh.rest, (len(time), *mesh.rest.shape)).copy()
    velocity = np.zeros_like(position)
    case = Case({
        "rest": mesh.rest, "triangles": mesh.triangles, "original_vertex": mesh.original_vertex,
        "cut_edges": mesh.cut_edges, "time": time, "position": position, "velocity": velocity,
        "boundary_mask": np.zeros_like(mesh.rest, dtype=bool),
        "boundary_position": position.copy(), "boundary_velocity": velocity.copy(),
        "boundary_impulse": np.zeros((len(time) - 1, len(mesh.rest), 2)),
    }, {"material": {"young": 100.0, "poisson": 0.25, "density": 1.0, "thickness": 1.0}})
    case.save(tmp_path / "case.npz")
    manifest = {"format": "cut-dynamics-a-v1", "cases": [
        {"id": "stationary-cut", "group": "x1-depth2", "split": "test", "path": "case.npz"}
    ]}
    data = tmp_path / "collection.json"
    data.write_text(json.dumps(manifest), encoding="utf-8")
    return case, data


def save_checkpoint(tmp_path, architecture, method):
    model = create_model({"architecture": architecture, "method": method, "hidden_size": 8, "layers": 1})
    path = tmp_path / f"{method}-{architecture}.pt"
    torch.save({
        "format": "cut-dynamics-a-v1", "trained": True,
        "model_config": model.configuration(), "state_dict": model.state_dict(),
        "dt": 0.001, "training_case_hashes": [], "training_groups": ["training-only"],
    }, path)
    return path


@pytest.mark.parametrize("method", ["direct", "stress"])
def test_legacy_checkpoint_outputs_remain_exactly_unchanged(tmp_path, method):
    torch.set_num_threads(2)
    torch.manual_seed(41)
    case, _ = sample(tmp_path)
    model = DynamicsModel(method, hidden_size=8, layers=1)
    with torch.no_grad():
        model.decoder[-1].weight.normal_(std=0.02)
        model.decoder[-1].bias.normal_(std=0.01)
    path = tmp_path / "old-best.pt"
    torch.save({"format": "cut-dynamics-a-v1", "model_config": model.configuration(),
                "state_dict": model.state_dict()}, path)
    original_hash = hashlib.sha256(path.read_bytes()).hexdigest()
    loaded, _ = load_model(path)
    graph = graph_for(case)
    position = torch.as_tensor(case.arrays["position"][0], dtype=torch.float32)
    velocity = torch.as_tensor(case.arrays["velocity"][0], dtype=torch.float32)
    expected_x, expected_v = [position], [velocity]
    with torch.inference_mode():
        for frame in range(1, len(case.arrays["time"])):
            bx = torch.as_tensor(case.arrays["boundary_position"][frame], dtype=torch.float32)
            bv = torch.as_tensor(case.arrays["boundary_velocity"][frame], dtype=torch.float32)
            position, velocity, _ = model(graph, position, velocity, bx, bv, 0.001)
            expected_x.append(position)
            expected_v.append(velocity)
    actual, _, failure = rollout(loaded, case, 0.001, "cpu")
    assert failure is None
    np.testing.assert_array_equal(actual["position"], torch.stack(expected_x).numpy().astype(np.float64))
    np.testing.assert_array_equal(actual["velocity"], torch.stack(expected_v).numpy().astype(np.float64))
    assert hashlib.sha256(path.read_bytes()).hexdigest() == original_hash


@pytest.mark.parametrize("method", ["direct", "stress"])
def test_equilibrium_probes_cache_autonomous_forces_only(tmp_path, monkeypatch, method):
    torch.set_num_threads(2)
    case, _ = sample(tmp_path)
    model = create_model({"architecture": "equilibrium", "method": method, "hidden_size": 8, "layers": 1})
    original = model.instantaneous_force
    calls = []

    def record(graph, position):
        calls.append(position.detach().clone())
        return original(graph, position)

    monkeypatch.setattr(model, "instantaneous_force", record)
    monkeypatch.setitem(sys.modules, "felupe", None)
    model.train()
    result = rest_probe(model, case)
    steps = len(case.arrays["time"]) - 1
    assert len(calls) == steps + 1
    assert result["initial_force_scope"] == "instantaneous_start_force"
    assert result["integrator"] == "velocity_verlet"
    assert result["max_free_position_drift"] == result["max_free_speed"] == 0
    assert model.training
    calls.clear()
    arrays, status = teacher_forced(model, case)
    assert len(calls) == 2 * steps
    assert status["status"] == "completed"
    assert status["integrator"] == "velocity_verlet"
    np.testing.assert_array_equal(arrays["position"], case.arrays["position"])


def test_old_and_new_heads_share_comparison_without_overwriting(tmp_path, monkeypatch):
    torch.set_num_threads(2)
    case, data = sample(tmp_path)
    checkpoints = [save_checkpoint(tmp_path, architecture, method)
                   for method in ("direct", "stress") for architecture in ("legacy", "equilibrium")]
    protected = checkpoints + [tmp_path / "case.npz", data]
    checksums = {path: hashlib.sha256(path.read_bytes()).hexdigest() for path in protected}
    monkeypatch.setitem(sys.modules, "felupe", None)

    def forbidden(*args, **kwargs):
        raise AssertionError("Benchmark and diagnostics must not train")

    monkeypatch.setattr(torch.optim.Optimizer, "__init__", forbidden)
    output = tmp_path / "comparison"
    report = benchmark(data, checkpoints, output)
    assert set(report["methods"]) == {"direct_legacy", "direct_equilibrium", "stress_legacy", "stress_equilibrium"}
    for label, result in report["methods"].items():
        assert result["summary"]["completed_cases"] == 1
        assert result["checkpoint_sha256"] == checksums[Path(result["checkpoint"])]
        saved = Case.load(output / result["cases"][0]["trajectory"])
        assert saved.metadata["method_label"] == label
        assert saved.metadata["architecture"] == result["architecture"]
        assert saved.metadata["checkpoint_sha256"] == result["checkpoint_sha256"]
        np.testing.assert_array_equal(saved.arrays["position"], case.arrays["position"])
    result = diagnose(output / "report.json", data, tmp_path / "diagnostics")
    diagnostic = json.loads(Path(result["report"]).read_text(encoding="utf-8"))
    assert set(diagnostic["methods"]) == set(report["methods"])
    assert all(value["checkpoint_identity_status"] == "verified_original_checksum"
               for value in diagnostic["methods"].values())
    for path in protected:
        assert hashlib.sha256(path.read_bytes()).hexdigest() == checksums[path]


def test_old_report_without_identity_fields_remains_diagnosable(tmp_path):
    _, data = sample(tmp_path)
    checkpoint = save_checkpoint(tmp_path, "legacy", "stress")
    output = tmp_path / "benchmark"
    report = benchmark(data, [checkpoint], output)
    for key in ("checkpoint_sha256", "model_method", "architecture", "integrator", "force_semantics"):
        report["methods"]["stress"].pop(key)
    (output / "report.json").write_text(json.dumps(report), encoding="utf-8")
    result = diagnose(output / "report.json", data, tmp_path / "diagnostics")
    diagnostic = json.loads(Path(result["report"]).read_text(encoding="utf-8"))
    assert diagnostic["methods"]["stress"]["checkpoint_identity_status"] == "original_checkpoint_checksum_unavailable"


def test_saved_trajectory_identity_mismatch_is_rejected(tmp_path):
    _, data = sample(tmp_path)
    checkpoint = save_checkpoint(tmp_path, "equilibrium", "stress")
    output = tmp_path / "benchmark"
    report = benchmark(data, [checkpoint], output)
    trajectory = output / report["methods"]["stress"]["cases"][0]["trajectory"]
    saved = Case.load(trajectory)
    saved.metadata["architecture"] = "legacy"
    saved.save(trajectory)
    with pytest.raises(ValueError, match="architecture differs"):
        diagnose(output / "report.json", data, tmp_path / "diagnostics")
