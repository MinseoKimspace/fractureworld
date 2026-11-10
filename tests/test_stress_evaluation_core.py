import numpy as np
import pytest
import torch

from fracture_world.benchmark2d import rollout
from fracture_world.cases import Case
from fracture_world.neural import create_model
from fracture_world.stress_evaluation import (
    case_at_time,
    evaluation_summary,
    evaluation_time,
    orientation_metrics,
    prediction_metrics,
    provider_difference,
    resample_prediction,
)
from fracture_world.topology import rectangle


def stationary_case(duration=0.2, dt=0.002):
    mesh = rectangle(2, 2)
    time = evaluation_time(duration, dt)
    position = np.broadcast_to(mesh.rest, (len(time), *mesh.rest.shape)).copy()
    mask = np.repeat((mesh.rest[:, 0:1] == 0) | (mesh.rest[:, 0:1] == 1), 2, axis=1)
    return Case({
        "rest": mesh.rest.copy(), "triangles": mesh.triangles.copy(),
        "original_vertex": mesh.original_vertex.copy(), "cut_edges": mesh.cut_edges.copy(),
        "time": time, "position": position, "velocity": np.zeros_like(position),
        "boundary_mask": mask, "boundary_position": position.copy(),
        "boundary_velocity": np.zeros_like(position),
        "boundary_impulse": np.zeros((len(time) - 1, *mesh.rest.shape)),
    }, {
        "material": {"young": 1.0, "poisson": 0.25, "density": 1.0, "thickness": 1.0},
        "pull": {"amplitude": 0.0, "ramp_time": 0.05},
    })


def test_half_step_alignment_sums_impulses_and_averages_forces():
    fine_time = np.linspace(0, 0.004, 5)
    state = np.broadcast_to(fine_time[:, None, None], (5, 2, 2)).copy()
    impulses = np.broadcast_to(np.arange(1, 5)[:, None, None], (4, 2, 2)).copy()
    result = resample_prediction({
        "time": fine_time, "position": state, "velocity": state + 0.1,
        "boundary_impulse": impulses, "internal_force": impulses * 2,
    }, 2, fine_time[::2])
    np.testing.assert_array_equal(result["position"], state[::2])
    np.testing.assert_array_equal(result["boundary_impulse"][:, 0, 0], [3, 7])
    np.testing.assert_array_equal(result["internal_force"][:, 0, 0], [3, 7])


def test_half_step_failure_retains_only_completed_coarse_intervals():
    state = np.zeros((4, 2, 2))
    result = resample_prediction({
        "time": np.arange(4) * 0.001, "position": state, "velocity": state,
        "boundary_impulse": np.ones((3, 2, 2)),
    }, 2, np.arange(4) * 0.002)
    assert len(result["position"]) == 2
    assert len(result["boundary_impulse"]) == 1
    np.testing.assert_array_equal(result["boundary_impulse"], np.full((1, 2, 2), 2))


def test_half_step_geometry_checks_intermediate_states_before_downsampling():
    reference = stationary_case(0.004)
    time = evaluation_time(0.004, 0.001)
    position = np.broadcast_to(reference.mesh.rest, (len(time), *reference.mesh.rest.shape)).copy()
    position[1, 4] += [0.0, -1.0]
    native = {
        "time": time, "position": position, "velocity": np.zeros_like(position),
        "boundary_impulse": np.zeros((len(time) - 1, *reference.mesh.rest.shape)),
    }
    geometry = orientation_metrics(reference.mesh, native)
    coarse = resample_prediction(native, 2, reference.arrays["time"])
    result = prediction_metrics(reference, coarse, {"status": "completed", "runtime_seconds": 0}, geometry)
    assert result["aligned_geometry"]["orientation_valid"] is True
    assert result["orientation_valid"] is False
    assert result["frames_with_inversion"] == 1
    assert result["first_invalid_orientation_time"] == 0.001


@pytest.mark.parametrize("factor", [0, -1, 1.5, True])
def test_invalid_resampling_is_rejected(factor):
    with pytest.raises(ValueError, match="Resampling factor"):
        resample_prediction({}, factor, [0, 0.1])


def test_neural_evaluation_does_not_read_future_reference_states():
    torch.set_num_threads(2)
    torch.manual_seed(5)
    source = stationary_case(0.008)
    source.arrays["position"][1:, ~source.arrays["boundary_mask"]] += 100.0
    source.arrays["velocity"][1:, ~source.arrays["boundary_mask"]] += 20.0
    inputs = case_at_time(source, source.arrays["time"])
    assert not np.array_equal(inputs.arrays["position"][1:], source.arrays["position"][1:])
    model = create_model({"architecture": "equilibrium", "method": "stress", "hidden_size": 8, "layers": 1})
    with torch.no_grad():
        model.decoder[-1].weight.normal_(std=0.1)
    first, _, failure = rollout(model, inputs, 0.002, "cpu")
    inputs.arrays["position"][1:, ~inputs.arrays["boundary_mask"]] = 1e4
    inputs.arrays["velocity"][1:, ~inputs.arrays["boundary_mask"]] = -1e4
    second, _, second_failure = rollout(model, inputs, 0.002, "cpu")
    assert failure is None and second_failure is None
    np.testing.assert_array_equal(first["position"], second["position"])
    np.testing.assert_array_equal(first["velocity"], second["velocity"])


def test_partial_rollout_time_checkpoints_do_not_claim_unreached_horizons():
    reference = stationary_case(1.0)
    prediction = {name: reference.arrays[name][:20].copy() for name in ("time", "position", "velocity")}
    prediction["boundary_impulse"] = reference.arrays["boundary_impulse"][:19].copy()
    result = prediction_metrics(reference, prediction, {"status": "failed", "failure": "test", "runtime_seconds": 0})
    assert result["aligned_completed_steps"] == 19
    assert all(not point["reached"] for point in result["time_checkpoints"])
    assert result["aligned_completed_horizon"] < 0.2


def test_provider_difference_uses_free_coordinates_only():
    reference = stationary_case(0.008)
    neural = {name: reference.arrays[name].copy() for name in ("position", "velocity")}
    oracle = {name: value.copy() for name, value in neural.items()}
    neural["position"][1:, reference.arrays["boundary_mask"]] += 100
    result = provider_difference(reference, neural, oracle)
    assert result["free_position_rmse"] == 0
    assert result["free_velocity_rmse"] == 0


def test_summary_counts_invalid_complete_trajectories_and_failed_attempts():
    complete = {
        "status": "completed", "metric_status": "available", "orientation_valid": False,
        "runtime_seconds": 1.0, "free_motion": {"position_rmse": 2.0, "velocity_rmse": 3.0},
    }
    failed = {"status": "failed", "runtime_seconds": 0.5}
    summary = evaluation_summary([
        {"category": "load", "status": "completed", "methods": {"neural_verlet": complete}},
        {"category": "load", "status": "completed", "methods": {"neural_verlet": failed}},
        {"category": "load", "status": "reference_unverified", "methods": {}},
    ])["load"]
    assert summary["scenarios"] == 3 and summary["reference_unavailable"] == 1
    method = summary["methods"]["neural_verlet"]
    assert method["attempted"] == 2 and method["completed"] == 1 and method["orientation_valid"] == 0
    assert method["mean_free_position_rmse"] == 2.0
    assert method["runtime_seconds"] == 1.5
