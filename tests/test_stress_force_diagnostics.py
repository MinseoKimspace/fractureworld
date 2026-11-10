import hashlib

import numpy as np
import pytest
import torch

from fracture_world.cases import Case
from fracture_world.stress_force_diagnostics import instantaneous_force_error, vector_rms
from fracture_world.teacher2d import Material
from fracture_world.topology import rectangle


class ScaleForce(torch.nn.Module):
    def __init__(self, scale=2.0):
        super().__init__()
        self.scale = torch.nn.Parameter(torch.tensor(scale, dtype=torch.float32))

    def instantaneous_force(self, graph, position):
        return position * self.scale


@pytest.fixture
def reference():
    mesh = rectangle(2, 2)
    time = np.linspace(0, 1, 11)
    position = mesh.rest[None] + time[:, None, None] * np.array([0.071234567891, 0.123456789123])
    return Case({
        "rest": mesh.rest.copy(), "triangles": mesh.triangles.copy(),
        "original_vertex": mesh.original_vertex.copy(), "cut_edges": mesh.cut_edges.copy(),
        "time": time, "position": position, "velocity": np.zeros_like(position),
        "boundary_mask": np.zeros_like(mesh.rest, dtype=bool),
        "boundary_position": position.copy(), "boundary_velocity": np.zeros_like(position),
    }, {"material": Material().metadata()})


def install_teacher(monkeypatch, evaluate):
    observed = []

    class Teacher:
        def __init__(self, *args):
            pass

        def __call__(self, position):
            observed.append(position.copy())
            return evaluate(position)

    monkeypatch.setattr("fracture_world.stress_force_diagnostics.InternalForce", Teacher)
    return observed


def test_exact_checkpoint_states_are_sampled_and_functions_share_rounded_input(reference, monkeypatch):
    observed = install_teacher(monkeypatch, lambda value: value * 2)
    model = ScaleForce().train()
    positions = reference.arrays["position"].copy()
    result = instantaneous_force_error(model, reference, Material(), {"position": positions}, "cpu", samples=2)
    assert result["status"] == "available"
    assert result["frames"] == [0, 2, 5, 10]
    assert result["sample_count"] == 4
    assert result["force_rmse"] == 0
    assert result["force_relative_rmse"] == 0
    assert result["free_acceleration_rmse"] == 0
    assert model.training is True
    for supplied, frame in zip(observed, result["frames"]):
        np.testing.assert_array_equal(supplied, positions[frame].astype(np.float32).astype(np.float64))
    assert [point["requested_time"] for point in result["time_checkpoints"]] == [0.2, 0.5, 1.0]
    assert all(point["reached"] and point["status"] == "available" for point in result["time_checkpoints"])
    assert result["curves"]["time"] == [0.0, 0.2, 0.5, 1.0]


def test_short_prefix_never_substitutes_previous_state_for_unreached_checkpoint(reference, monkeypatch):
    observed = install_teacher(monkeypatch, lambda value: value * 2)
    prediction = {"position": reference.arrays["position"][:2], "time": reference.arrays["time"][:2]}
    result = instantaneous_force_error(ScaleForce(), reference, Material(), prediction, "cpu", samples=2)
    assert len(observed) == 2
    for point in result["time_checkpoints"]:
        assert point["reached"] is False and point["status"] == "unreached"
        assert point["time"] is None and point["frame"] is None
        assert point["force_rmse"] is None and point["free_acceleration_rmse"] is None


def test_reached_time_missing_from_grid_stays_unknown(reference, monkeypatch):
    install_teacher(monkeypatch, lambda value: value * 2)
    indices = [0, 3, 6, 9, 10]
    prediction = {name: reference.arrays[name][indices] for name in ("position", "time")}
    result = instantaneous_force_error(ScaleForce(), reference, Material(), prediction, "cpu", samples=2)
    for point in result["time_checkpoints"][:2]:
        assert point["reached"] is True
        assert point["status"] == "time_not_on_grid"
        assert point["force_rmse"] is None
    assert result["time_checkpoints"][-1]["status"] == "available"


def test_zero_teacher_signal_has_no_fabricated_relative_error(reference, monkeypatch):
    install_teacher(monkeypatch, np.zeros_like)
    result = instantaneous_force_error(ScaleForce(0), reference, Material(), reference.arrays, "cpu")
    assert result["status"] == "available"
    assert result["force_rmse"] == 0 and result["teacher_force_rms"] == 0
    assert result["force_relative_rmse"] is None
    assert result["normalization_status"] == "zero_signal"
    assert all(point["force_relative_rmse"] is None for point in result["time_checkpoints"])


def test_sampled_force_curves_and_aggregate_match_manual_values(reference, monkeypatch):
    install_teacher(monkeypatch, lambda value: value * 2)
    result = instantaneous_force_error(ScaleForce(0), reference, Material(), reference.arrays, "cpu", samples=2)
    states = reference.arrays["position"][result["frames"]].astype(np.float32).astype(np.float64)
    expected = np.sqrt(np.mean(np.sum((2 * states) ** 2, axis=-1)))
    assert result["force_rmse"] == pytest.approx(expected)
    assert result["force_relative_rmse"] == pytest.approx(1)
    for index, frame in enumerate(result["frames"]):
        value = np.sqrt(np.mean(np.sum((2 * states[index]) ** 2, axis=-1)))
        assert result["curves"]["force_rmse"][index] == pytest.approx(value)


def test_invalid_sample_does_not_silently_disappear_from_aggregate(reference, monkeypatch):
    def evaluate(value):
        return np.full_like(value, np.nan) if value[0, 0] > 0.03 else value * 2

    install_teacher(monkeypatch, evaluate)
    result = instantaneous_force_error(ScaleForce(), reference, Material(), reference.arrays, "cpu", samples=2)
    assert result["status"] == "unknown_incomplete_force_evaluation"
    assert result["valid_sample_count"] < result["sample_count"]
    assert result["force_rmse"] is None
    assert result["time_checkpoints"][-1]["status"] == "unknown_force_evaluation"
    assert result["time_checkpoints"][0]["status"] == "available"


def test_force_evaluation_failure_preserves_mode_inputs_and_weights(reference, monkeypatch, tmp_path):
    install_teacher(monkeypatch, lambda value: value * 2)
    model = ScaleForce().eval()
    reference_path = tmp_path / "reference.npz"
    checkpoint = tmp_path / "best.pt"
    reference.save(reference_path)
    torch.save(model.state_dict(), checkpoint)
    protected = {path: hashlib.sha256(path.read_bytes()).hexdigest() for path in (reference_path, checkpoint)}
    arrays = {name: value.copy() for name, value in reference.arrays.items()}
    weights = {name: value.clone() for name, value in model.state_dict().items()}

    def forbidden(*args, **kwargs):
        raise AssertionError("Force diagnostics cannot train or write weights")

    monkeypatch.setattr(torch.optim.Optimizer, "__init__", forbidden)
    monkeypatch.setattr(torch, "save", forbidden)
    result = instantaneous_force_error(model, reference, Material(), reference.arrays, "cpu")
    assert result["status"] == "available" and model.training is False
    assert {path: hashlib.sha256(path.read_bytes()).hexdigest() for path in protected} == protected
    for name, value in arrays.items():
        np.testing.assert_array_equal(reference.arrays[name], value)
    for name, value in weights.items():
        torch.testing.assert_close(model.state_dict()[name], value, atol=0, rtol=0)
    assert len(list(tmp_path.iterdir())) == 2


def test_force_exception_keeps_checkpoint_unknown_and_restores_training_mode(reference, monkeypatch):
    install_teacher(monkeypatch, lambda value: value * 2)
    model = ScaleForce().train()

    def broken(*args, **kwargs):
        raise RuntimeError("Diagnostic force failure")

    monkeypatch.setattr(model, "instantaneous_force", broken)
    result = instantaneous_force_error(model, reference, Material(), reference.arrays, "cpu", samples=2)
    assert result["status"] == "unknown_incomplete_force_evaluation"
    assert result["valid_sample_count"] == 0 and result["force_rmse"] is None
    assert model.training is True
    assert all(point["status"] == "unknown_force_evaluation" for point in result["time_checkpoints"])


def test_absent_free_coordinates_have_unknown_acceleration_error(reference, monkeypatch):
    install_teacher(monkeypatch, lambda value: value * 2)
    reference.arrays["boundary_mask"][:] = True
    result = instantaneous_force_error(ScaleForce(), reference, Material(), reference.arrays, "cpu", samples=2)
    assert result["status"] == "available" and result["force_rmse"] == 0
    assert result["free_acceleration_rmse"] is None
    assert all(point["free_acceleration_rmse"] is None for point in result["time_checkpoints"])


def test_vector_rms_avoids_overflow_from_squaring_finite_values():
    values = np.full((2, 3, 2), 1e200)
    assert np.isfinite(vector_rms(values))
    assert vector_rms(values) == pytest.approx(np.sqrt(2) * 1e200)


@pytest.mark.parametrize("samples", [0, -1, True, 1.5])
def test_invalid_force_sample_count_is_rejected(reference, samples):
    with pytest.raises(ValueError, match="sample count"):
        instantaneous_force_error(ScaleForce(), reference, Material(), reference.arrays, "cpu", samples=samples)


@pytest.mark.parametrize("checkpoints", [(0,), (-1,), (np.nan,), (0.2, 0.2)])
def test_invalid_force_checkpoints_are_rejected(reference, checkpoints):
    with pytest.raises(ValueError, match="checkpoints"):
        instantaneous_force_error(ScaleForce(), reference, Material(), reference.arrays, "cpu", checkpoints=checkpoints)
