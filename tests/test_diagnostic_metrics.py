import numpy as np
import pytest

from fracture_world.cases import Case
from fracture_world.diagnostic_metrics import free_motion_metrics, momentum_balance, zero_internal_force
from fracture_world.topology import rectangle, split_edges


def make_case(split=False, fixed=False, moving=False):
    mesh = rectangle(2, 2)
    if split:
        mesh = split_edges(mesh, [[1, 4], [4, 7]])
    time = np.array([0.0, 0.1, 0.2])
    mask = np.zeros_like(mesh.rest, dtype=bool)
    if fixed:
        mask[np.isclose(mesh.rest[:, 0], 0) | np.isclose(mesh.rest[:, 0], 1)] = True
    velocity = np.zeros((len(time), *mesh.rest.shape))
    if moving:
        velocity[:] = [0.2, -0.1]
    position = mesh.rest[None] + time[:, None, None] * velocity
    return Case({
        "rest": mesh.rest, "triangles": mesh.triangles,
        "original_vertex": mesh.original_vertex, "cut_edges": mesh.cut_edges,
        "time": time, "position": position.copy(), "velocity": velocity.copy(),
        "boundary_mask": mask, "boundary_position": position.copy(), "boundary_velocity": velocity.copy(),
    }, {"material": {"young": 1.0, "poisson": 0.2, "density": 1.0, "thickness": 1.0}})


def test_free_metrics_ignore_constrained_coordinate_errors():
    case = make_case(moving=True)
    case.arrays["boundary_mask"][:, 0] = True
    prediction = {key: case.arrays[key].copy() for key in ("position", "velocity")}
    prediction["position"][1:, :, 0] += 1000
    prediction["velocity"][1:, :, 0] += 1000
    prediction["position"][1:, :, 1] += 0.01
    prediction["velocity"][1:, :, 1] += 0.02
    metrics = free_motion_metrics(case, prediction)
    assert metrics["position_rmse"] == pytest.approx(0.01)
    assert metrics["velocity_rmse"] == pytest.approx(0.02)
    assert metrics["velocity_reference_rms"] == pytest.approx(0.1)
    assert metrics["velocity_relative_rmse"] == pytest.approx(0.2)
    assert metrics["curves"]["time"] == [0.1, 0.2]


def test_zero_signal_has_unknown_relative_error_without_epsilon():
    case = make_case()
    prediction = {key: case.arrays[key].copy() for key in ("position", "velocity")}
    prediction["position"][1:] += 0.1
    metrics = free_motion_metrics(case, prediction)
    assert metrics["position_rmse"] == pytest.approx(np.sqrt(0.02))
    assert metrics["position_relative_rmse"] is None
    assert metrics["position_normalization_status"] == "zero_signal"
    assert metrics["velocity_relative_rmse"] is None


def test_no_free_coordinates_are_not_reported_as_zero_error():
    case = make_case()
    case.arrays["boundary_mask"][:] = True
    metrics = free_motion_metrics(case, case.arrays)
    assert metrics["free_coordinates"] == 0
    assert metrics["position_rmse"] is None
    assert metrics["position_normalization_status"] == "no_free_coordinates"


def test_prefix_metrics_are_causal_and_validate_timestamps():
    case = make_case(moving=True)
    prediction = {key: case.arrays[key][:2].copy() for key in ("position", "velocity", "time")}
    case.arrays["position"][2] += 100
    metrics = free_motion_metrics(case, prediction)
    assert metrics["frames_evaluated"] == 1
    assert metrics["position_rmse"] == 0
    prediction["time"][1] = 0.12
    with pytest.raises(ValueError, match="timestamps"):
        free_motion_metrics(case, prediction)


def test_zero_internal_force_is_ballistic_with_exact_grip_motion():
    case = make_case(fixed=True, moving=True)
    case.arrays["boundary_position"][1:, :, 0] += 0.03
    case.arrays["boundary_velocity"][1:, :, 0] = 0.4
    prediction = zero_internal_force(case)
    free = ~case.arrays["boundary_mask"]
    expected = case.arrays["position"][0] + 0.2 * case.arrays["velocity"][0]
    assert np.array_equal(prediction["position"][-1, free], expected[free])
    assert np.array_equal(prediction["position"][:, case.arrays["boundary_mask"]],
                          case.arrays["boundary_position"][:, case.arrays["boundary_mask"]])
    assert np.array_equal(prediction["velocity"][:, case.arrays["boundary_mask"]],
                          case.arrays["boundary_velocity"][:, case.arrays["boundary_mask"]])
    balance = momentum_balance(case, prediction)
    assert balance["global"]["cumulative_max_norm"] < 1e-14


def test_full_split_constant_velocity_has_two_balanced_components():
    case = make_case(split=True, moving=True)
    prediction = zero_internal_force(case)
    balance = momentum_balance(case, prediction)
    assert balance["status"] == "available"
    assert balance["component_count"] == 2
    assert balance["global"]["cumulative_max_norm"] == 0
    assert all(component["cumulative_max_norm"] == 0 for component in balance["components"])


def test_zero_force_baseline_does_not_read_future_teacher_motion():
    case = make_case(fixed=True, moving=True)
    first = zero_internal_force(case)
    case.arrays["position"][1:] += 123
    case.arrays["velocity"][1:] -= 321
    second = zero_internal_force(case)
    for key in ("position", "velocity", "boundary_impulse"):
        assert np.array_equal(first[key], second[key])


def test_component_error_can_cancel_in_global_balance():
    case = make_case(split=True, fixed=True)
    prediction = zero_internal_force(case)
    left = np.flatnonzero(np.isclose(case.mesh.rest[:, 0], 0))[0]
    right = np.flatnonzero(np.isclose(case.mesh.rest[:, 0], 1))[0]
    prediction["boundary_impulse"][0, left, 0] = 0.1
    prediction["boundary_impulse"][0, right, 0] = -0.1
    balance = momentum_balance(case, prediction)
    assert balance["global"]["cumulative_max_norm"] == 0
    assert all(component["cumulative_max_norm"] == pytest.approx(0.1)
               for component in balance["components"])


def test_missing_boundary_impulse_is_unknown():
    case = make_case(split=True)
    balance = momentum_balance(case, case.arrays)
    assert balance["component_count"] == 2
    assert balance["status"] == "unknown_boundary_impulse"
    assert balance["global"] is None


def test_momentum_uses_initial_momentum_instead_of_zero_assumption():
    case = make_case(moving=True)
    prediction = zero_internal_force(case)
    prediction["velocity"][1:] += [0.3, 0.0]
    prediction["boundary_impulse"][0, :, 0] = case.mass * 0.3
    balance = momentum_balance(case, prediction)
    assert balance["global"]["cumulative_max_norm"] < 1e-14


@pytest.mark.parametrize("key", ["position", "velocity"])
def test_nonfinite_predictions_are_rejected(key):
    case = make_case()
    prediction = {name: case.arrays[name].copy() for name in ("position", "velocity")}
    prediction[key][1, 0, 0] = np.inf
    with pytest.raises(ValueError, match="finite"):
        free_motion_metrics(case, prediction)
