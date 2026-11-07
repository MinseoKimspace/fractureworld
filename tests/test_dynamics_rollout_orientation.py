import json

import numpy as np
from numpy.testing import assert_allclose, assert_array_equal
import pytest
import torch

from fracture_world import dynamics_runtime as runtime
from fracture_world.dynamics_geometry import geometry_diagnostics
from fracture_world.dynamics_initial import load_initial
from fracture_world.dynamics_model import TemporalDynamicsModel
from test_dynamics_runtime import portable_rollout_inputs


def orientation_config(**changes):
    values = dict(fracture_samples=0, fracture_policy="marginal_search", mechanics_mode="proximal",
                  max_iterations=20000, max_contact_iterations=5)
    return runtime.RolloutConfig(**{**values, **changes})


def prediction(batch, positions, *, fracture_probability=0.):
    return dict(positions=positions, velocities=torch.zeros_like(batch["velocities"]),
                fracture_probability=torch.full_like(batch["broken"], fracture_probability, dtype=torch.float32),
                support_release_probability=torch.zeros_like(batch["masses"]))


def determinants(positions):
    corners = np.asarray(positions).reshape(-1, 4, 3)
    return np.linalg.det(corners[:, 1:] - corners[:, :1])


@pytest.mark.parametrize("failure", ["inverted", "degenerate"])
def test_finite_raw_endpoints_are_saved_before_offline_geometry_evaluation(tmp_path, monkeypatch, failure):
    checkpoint, initial_path, topology, fixed = portable_rollout_inputs(tmp_path, has_corrector=False)
    calls = []

    def predict(self, batch):
        calls.append(batch["positions"].clone())
        positions = batch["positions"].clone()
        if len(calls) == 1:
            positions[:, 0] += .125
        elif failure == "inverted":
            positions[0] = positions[1:4].sum(dim=0) - 2 * positions[0]
        else:
            positions[0] = positions[1]
        return prediction(batch, positions)

    def forbidden(*args, **kwargs):
        raise AssertionError("online inference invoked a mechanical evaluator")

    monkeypatch.setattr(TemporalDynamicsModel, "forward", predict)
    for name in ("geometry_diagnostics", "swept_geometry_diagnostics", "step_energy_diagnostics", "advance_elastic_state"):
        monkeypatch.setattr(runtime, name, forbidden)
    output = tmp_path / "raw"
    report = runtime.rollout_dynamics(checkpoint, initial_path, output, 2, "raw", orientation_config(), "cpu")
    assert report["status"] == "completed"
    assert report["completed_steps"] == len(calls) == 2
    assert report["checked_constraint_steps"] == 0
    assert report["all_committed_geometry_valid"] is None
    assert report["all_committed_energy_valid"] is None
    assert json.loads((output / "report.json").read_text())["events"] == report["events"]
    with np.load(output / "trajectory.npz", allow_pickle=False) as saved:
        assert_allclose(saved["times"], [0., .01, .02])
        assert_array_equal(saved["positions"][0], topology.rest_positions)
        assert_array_equal(saved["positions"][1], topology.rest_positions + [.125, 0., 0.])
        assert_array_equal(calls[1].numpy(), saved["positions"][1])
        assert_array_equal(saved["velocities"], np.zeros((3, 8, 3)))
        assert_array_equal(saved["broken"], np.zeros((3, 1), dtype=bool))
        assert_array_equal(saved["fixed"], np.stack([fixed, fixed, fixed]))
        assert determinants(saved["positions"][-1])[0] <= 0
        initial = load_initial(initial_path)
        offline = geometry_diagnostics(initial.topology.mesh, saved["positions"][-1], saved["broken"][-1], 1e-6)
        assert offline["valid"] is False
        assert offline["inverted_tetrahedra"] == 1
    assert len(json.loads((output / "replay.json").read_text())["frames"]) == 3


def test_raw_shared_corner_violation_is_stored_for_offline_evaluation(tmp_path, monkeypatch):
    checkpoint, initial_path, _, _ = portable_rollout_inputs(tmp_path, has_corrector=False)

    def predict(self, batch):
        positions = batch["positions"].clone()
        positions[1, 0] += .125
        return prediction(batch, positions)

    monkeypatch.setattr(TemporalDynamicsModel, "forward", predict)
    output = tmp_path / "raw"
    report = runtime.rollout_dynamics(checkpoint, initial_path, output, 2, "raw", orientation_config(), "cpu")
    assert report["status"] == "completed"
    assert report["completed_steps"] == 2
    assert report["all_committed_geometry_valid"] is None
    with np.load(output / "trajectory.npz", allow_pickle=False) as saved:
        assert (determinants(saved["positions"]) > 0).all()
        initial = load_initial(initial_path)
        offline = geometry_diagnostics(initial.topology.mesh, saved["positions"][-1], saved["broken"][-1], 1e-6)
        assert offline["valid"] is False
        assert offline["inverted_tetrahedra"] == 0


def test_raw_fracture_ranking_uses_the_prediction_before_offline_evaluation(tmp_path, monkeypatch):
    checkpoint, initial, _, _ = portable_rollout_inputs(tmp_path, has_corrector=False)
    monkeypatch.setattr(TemporalDynamicsModel, "forward", lambda self, batch:
                        prediction(batch, batch["positions"], fracture_probability=.9))
    report = runtime.rollout_dynamics(checkpoint, initial, tmp_path / "raw", 1, "raw", orientation_config(), "cpu")
    assert report["status"] == "completed"
    assert report["completed_steps"] == 1
    assert report["all_committed_geometry_valid"] is None
    event = report["events"][0]
    assert event["selected_candidate"] == 0
    assert event["candidates"][0]["broken_faces"] == 1
    assert event["candidates"][0]["prior"] < event["candidates"][1]["prior"]
