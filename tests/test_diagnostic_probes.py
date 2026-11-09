import builtins

import numpy as np
import pytest
import torch

from fracture_world.cases import Case
from fracture_world.diagnostic_probes import rest_probe, teacher_forced
from fracture_world.neural import DynamicsModel
from fracture_world.topology import rectangle, split_edges


def fixture_case():
    mesh = split_edges(rectangle(2, 2), [[1, 4], [4, 7]])
    time = np.linspace(0.0, 0.04, 5)
    velocity = np.broadcast_to(np.array([0.2, -0.1]), (len(time), len(mesh.rest), 2)).copy()
    position = mesh.rest[None] + time[:, None, None] * velocity
    mask = np.repeat(np.isclose(mesh.rest[:, 0:1], 0), 2, axis=1)
    return Case({
        "rest": mesh.rest, "triangles": mesh.triangles, "original_vertex": mesh.original_vertex,
        "cut_edges": mesh.cut_edges, "time": time, "position": position, "velocity": velocity,
        "boundary_mask": mask, "boundary_position": position.copy(), "boundary_velocity": velocity.copy(),
    }, {"material": {"young": 1.0, "poisson": 0.2, "density": 1.0, "thickness": 1.0}})


@pytest.mark.parametrize("method", ["direct", "stress"])
def test_zero_models_rest_and_teacher_forcing_without_engine(monkeypatch, method):
    torch.set_num_threads(2)
    original_import = builtins.__import__

    def guarded_import(name, *args, **kwargs):
        if name == "felupe" or name.startswith("felupe."):
            raise AssertionError("Diagnostic called a mechanics engine")
        return original_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", guarded_import)
    case = fixture_case()
    original = {key: value.copy() for key, value in case.arrays.items()}
    model = DynamicsModel(method, hidden_size=8, layers=1)
    model.train()
    weights = {key: value.clone() for key, value in model.state_dict().items()}
    rest = rest_probe(model, case)
    arrays, status = teacher_forced(model, case)
    assert rest["status"] == status["status"] == "completed"
    assert rest["max_free_position_drift"] == rest["max_free_speed"] == 0
    assert rest["initial_free_acceleration_max_norm"] == 0
    assert status["mode"] == "teacher_forced_one_step"
    assert status["autonomous"] is False
    assert status["completed_steps"] == 4
    np.testing.assert_allclose(arrays["position"], case.arrays["position"], atol=1e-7)
    np.testing.assert_allclose(arrays["velocity"], case.arrays["velocity"], atol=1e-7)
    assert model.training
    for key, value in weights.items():
        torch.testing.assert_close(value, model.state_dict()[key], atol=0, rtol=0)
    for key, value in original.items():
        np.testing.assert_array_equal(value, case.arrays[key])


def test_nonzero_learned_bias_has_nonzero_rest_drift():
    model = DynamicsModel("direct", hidden_size=8, layers=1)
    with torch.no_grad():
        model.decoder[-1].bias[:] = torch.tensor([0.2, 0.0])
    result = rest_probe(model, fixture_case())
    assert result["status"] == "completed"
    assert result["initial_free_acceleration_max_norm"] > 0
    assert result["max_free_position_drift"] > 0
    assert result["max_free_speed"] > 0


class RecordingModel(torch.nn.Module):
    def __init__(self, fail_at=None, failure="nan"):
        super().__init__()
        self.positions = []
        self.velocities = []
        self.fail_at = fail_at
        self.failure = failure

    def forward(self, graph, position, velocity, boundary_position, boundary_velocity, dt):
        self.positions.append(position.cpu().numpy().copy())
        self.velocities.append(velocity.cpu().numpy().copy())
        if len(self.positions) == self.fail_at:
            if self.failure == "runtime":
                raise RuntimeError("Test failure")
            return position * float("nan"), velocity, torch.zeros_like(position)
        force = torch.zeros_like(position)
        next_position = position + 0.123
        next_velocity = velocity + 0.234
        return (torch.where(graph.boundary_mask, boundary_position, next_position),
                torch.where(graph.boundary_mask, boundary_velocity, next_velocity), force)


def test_teacher_forcing_resets_to_each_teacher_state():
    case = fixture_case()
    model = RecordingModel()
    arrays, status = teacher_forced(model, case)
    assert status["status"] == "completed"
    for frame in range(4):
        np.testing.assert_allclose(model.positions[frame], case.arrays["position"][frame], atol=1e-7)
        np.testing.assert_allclose(model.velocities[frame], case.arrays["velocity"][frame], atol=1e-7)
    assert not np.allclose(arrays["position"][1], model.positions[1])
    np.testing.assert_array_equal(arrays["position"][0], case.arrays["position"][0])


@pytest.mark.parametrize("fail_at", [1, 3])
@pytest.mark.parametrize("failure", ["nan", "runtime"])
def test_teacher_forcing_reports_only_successful_prefix(fail_at, failure):
    case = fixture_case()
    arrays, status = teacher_forced(RecordingModel(fail_at, failure), case)
    assert status["status"] == "failed"
    assert status["completed_steps"] == fail_at - 1
    assert status["requested_steps"] == 4
    assert len(arrays["position"]) == fail_at
    assert arrays["internal_force"].shape == (fail_at - 1, len(case.mesh.rest), 2)
    assert arrays["boundary_impulse"].shape == arrays["internal_force"].shape
    assert np.isfinite(arrays["position"]).all()
    assert "frame" in status["failure"]


@pytest.mark.parametrize("fail_at", [1, 3])
def test_rest_failure_preserves_incomplete_status(fail_at):
    result = rest_probe(RecordingModel(fail_at), fixture_case())
    assert result["status"] == "failed"
    assert result["completed_steps"] == fail_at - 1
    assert result["failure"]
    if fail_at == 1:
        assert result["initial_free_acceleration_max_norm"] is None
    else:
        assert result["initial_free_acceleration_max_norm"] == 0


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA unavailable")
def test_probes_run_on_cuda_without_training():
    case = fixture_case()
    model = DynamicsModel("stress", hidden_size=8, layers=1).cuda()
    arrays, status = teacher_forced(model, case, "cuda")
    rest = rest_probe(model, case, "cuda")
    assert status["status"] == rest["status"] == "completed"
    assert np.isfinite(arrays["position"]).all()
    assert rest["max_free_position_drift"] == 0
