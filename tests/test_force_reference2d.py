import hashlib

import numpy as np
import pytest
import torch

from fracture_world.audit2d import REFINEMENT_LIMITS
from fracture_world.cases import Case
from fracture_world.force_reference2d import (
    checked_time,
    reference_settings,
    reference_solution,
    teacher_force_rollout,
)
from fracture_world.teacher2d import Material, simulate
from fracture_world.topology import rectangle, split_edges


def small_case():
    pytest.importorskip("felupe")
    mesh = split_edges(rectangle(2, 2), [[1, 4]])
    return simulate(mesh, Material(young=10), np.linspace(0, 0.04, 21),
                    amplitude=0.005, ramp_time=0.02, max_step=0.0005)


def test_independent_reference_has_stricter_refinement_and_preserves_mesh():
    pytest.importorskip("felupe")
    mesh = split_edges(rectangle(2, 2), [[1, 4]])
    before = mesh.rest.copy(), mesh.triangles.copy()
    reference, audit = reference_solution(
        mesh, Material(young=10), np.linspace(0, 0.02, 11),
        {"amplitude": 0.005, "ramp_time": 0.02},
    )
    assert audit["status"] == "completed"
    assert audit["converged"]
    assert audit["limits"] == REFINEMENT_LIMITS
    assert all(audit["checks"].values())
    assert audit["reference_level"] == "refined"
    assert audit["coarse_settings"]["max_step"] == pytest.approx(0.0005)
    assert audit["refined_settings"]["max_step"] == pytest.approx(0.00025)
    assert audit["refinement"]["position_max_difference"] < 1e-9
    assert audit["refinement"]["velocity_max_difference"] < 1e-8
    assert reference.metadata["rtol"] == 1e-11
    assert np.array_equal(mesh.rest, before[0])
    assert np.array_equal(mesh.triangles, before[1])


def test_unconverged_reference_remains_numerically_completed_but_unverified(monkeypatch):
    pytest.importorskip("felupe")
    monkeypatch.setattr("fracture_world.force_reference2d.refinement_error", lambda first, second: {
        "position_max_difference": 1e-6,
        "velocity_max_difference": 0.0,
        "boundary_impulse_max_difference": 0.0,
    })
    _, audit = reference_solution(
        rectangle(2, 2), Material(young=1), np.linspace(0, 0.004, 3),
        {"amplitude": 0.001, "ramp_time": 0.004},
    )
    assert audit["status"] == "completed"
    assert audit["converged"] is False
    assert audit["checks"]["position_max_difference"] is False
    assert audit["limits"]["position_max_difference"] == 1e-7


def test_force_oracle_uses_own_positions_and_ignores_future_truth():
    case = small_case()
    expected, _ = teacher_force_rollout(case)
    arrays = {key: value.copy() for key, value in case.arrays.items()}
    free = ~arrays["boundary_mask"]
    arrays["position"][1:, free] += 7
    arrays["velocity"][1:, free] -= 3
    arrays["internal_force"][:] = 1e9
    corrupted = Case(arrays, dict(case.metadata))
    actual, details = teacher_force_rollout(corrupted)
    for key in ("position", "velocity", "boundary_impulse", "internal_force"):
        assert np.array_equal(actual[key], expected[key])
    assert details["force_evaluations"] == len(case.arrays["time"])
    assert "Own evolving predicted positions" in details["force_evaluation_scope"]


def test_force_oracle_reuses_source_initial_state_and_not_rest_reset():
    case = small_case()
    arrays = {key: value.copy() for key, value in case.arrays.items()}
    del arrays["case_hash"]
    free = ~arrays["boundary_mask"]
    arrays["position"][0, free] += 0.002
    arrays["velocity"][0, free] = 0.01
    changed = Case(arrays, dict(case.metadata))
    prediction, details = teacher_force_rollout(changed)
    assert details["status"] == "completed"
    assert np.array_equal(prediction["position"][0], changed.arrays["position"][0])
    assert np.array_equal(prediction["velocity"][0], changed.arrays["velocity"][0])
    assert not np.array_equal(prediction["position"][0], changed.mesh.rest)


def test_half_step_oracle_converges_and_records_boundary_impulses():
    case = small_case()
    coarse, coarse_status = teacher_force_rollout(case)
    fine, fine_status = teacher_force_rollout(case, np.linspace(0, 0.04, 41))
    free = ~case.arrays["boundary_mask"]
    for key in ("position", "velocity"):
        coarse_error = np.sqrt(np.mean((coarse[key][1:, free] - case.arrays[key][1:, free]) ** 2))
        fine_error = np.sqrt(np.mean((fine[key][2::2, free] - case.arrays[key][1:, free]) ** 2))
        assert fine_error < coarse_error / 2.5
    assert coarse_status["completed_steps"] == 20
    assert fine_status["completed_steps"] == 40
    momentum = np.sum(np.diff(coarse["velocity"], axis=0) * case.mass[None, :, None], axis=1)
    assert np.max(np.abs(momentum - coarse["boundary_impulse"].sum(1))) < 1e-12
    assert fine["boundary_impulse"].shape == (40, len(case.mesh.rest), 2)


def test_one_second_oracle_extends_prescribed_pull_from_initial_state():
    pytest.importorskip("felupe")
    case = simulate(rectangle(2, 2), Material(young=1), np.linspace(0, 0.02, 11),
                    amplitude=0.002, ramp_time=0.02, max_step=0.0005)
    time = np.linspace(0, 1.0, 101)
    arrays, status = teacher_force_rollout(case, time)
    assert status["status"] == "completed"
    assert status["completed_horizon"] == 1
    assert status["completed_steps"] == 100
    assert status["force_evaluations"] == 101
    assert np.array_equal(arrays["position"][0], case.arrays["position"][0])
    assert arrays["time"][-1] == 1
    assert np.isfinite(arrays["position"]).all()


def test_reference_oracle_preserves_source_file_and_writes_nothing(tmp_path, monkeypatch):
    case = small_case()
    path = tmp_path / "source.npz"
    case.save(path)
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    before = {key: value.copy() for key, value in case.arrays.items()}
    monkeypatch.chdir(tmp_path)

    def forbidden(*args, **kwargs):
        raise AssertionError("Force reference cannot train or save checkpoints")

    monkeypatch.setattr(torch.optim.Optimizer, "__init__", forbidden)
    monkeypatch.setattr(torch, "save", forbidden)
    _, status = teacher_force_rollout(case)
    assert status["status"] == "completed"
    assert hashlib.sha256(path.read_bytes()).hexdigest() == digest
    assert [item.name for item in tmp_path.iterdir()] == ["source.npz"]
    assert all(np.array_equal(case.arrays[key], value) for key, value in before.items())


def test_force_failure_preserves_finite_prefix(monkeypatch):
    case = small_case()

    class FailingForce:
        def __init__(self, mesh, material):
            self.evaluations = 0
            self.runtime_seconds = 0

        def __call__(self, position):
            self.evaluations += 1
            if self.evaluations == 3:
                raise FloatingPointError("Injected force failure")
            return np.zeros_like(position)

    monkeypatch.setattr("fracture_world.force_reference2d.InternalForce", FailingForce)
    arrays, status = teacher_force_rollout(case)
    assert status["status"] == "failed"
    assert status["completed_steps"] == 1
    assert "Injected force failure" in status["failure"]
    assert len(arrays["position"]) == 2
    assert arrays["boundary_impulse"].shape[0] == 1
    assert np.isfinite(arrays["position"]).all()


@pytest.mark.parametrize("settings", [
    {"rtol": 1e-8}, {"atol": 1e-9}, {"max_step": 0.1},
    {"refined_rtol": 1e-10}, {"refined_atol": 1e-12}, {"refined_max_step": 0.0005},
    {"rtol": 0}, {"atol": np.nan}, {"unknown": 1},
])
def test_invalid_or_loosened_reference_settings_are_rejected(settings):
    with pytest.raises(ValueError):
        reference_settings(np.linspace(0, 0.02, 11), settings)


@pytest.mark.parametrize("time", [[0], [0.1, 0.2], [0, 0], [0, np.nan]])
def test_invalid_reference_timestamps_are_rejected(time):
    with pytest.raises(ValueError, match="timestamps"):
        checked_time(time)
