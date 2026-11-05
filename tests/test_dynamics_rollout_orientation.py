import json

import numpy as np
from numpy.testing import assert_allclose, assert_array_equal
import pytest
import torch

from fracture_world import dynamics_runtime as runtime
from fracture_world.dynamics_experiment import compare_dynamics
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
def test_invalid_raw_endpoint_saves_committed_prefix_without_energy_or_elastic_advance(tmp_path, monkeypatch, failure):
    checkpoint, initial, topology, fixed = portable_rollout_inputs(tmp_path, has_corrector=False)
    calls = dict(predictor=0, energy=0, advance=0)
    original_energy = runtime.step_energy_diagnostics
    original_advance = runtime.advance_elastic_state

    def predict(self, batch):
        calls["predictor"] += 1
        positions = batch["positions"].clone()
        if calls["predictor"] == 1:
            positions[:, 0] += .125
        elif failure == "inverted":
            positions[0] = positions[1:4].sum(dim=0) - 2 * positions[0]
        else:
            positions[0] = positions[1]
        return prediction(batch, positions)

    def energy(initial, positions, velocities, next_positions, *args, **kwargs):
        assert (determinants(next_positions) > 0).all()
        calls["energy"] += 1
        return original_energy(initial, positions, velocities, next_positions, *args, **kwargs)

    def advance(initial, positions, *args, **kwargs):
        assert (determinants(positions) > 0).all()
        calls["advance"] += 1
        return original_advance(initial, positions, *args, **kwargs)

    monkeypatch.setattr(TemporalDynamicsModel, "forward", predict)
    monkeypatch.setattr(runtime, "step_energy_diagnostics", energy)
    monkeypatch.setattr(runtime, "advance_elastic_state", advance)
    output = tmp_path / "raw"
    progress = []
    report = runtime.rollout_dynamics(checkpoint, initial, output, 4, "raw", orientation_config(), "cpu", progress.append)
    assert report["status"] == "timestep_rejected"
    assert report["completed_steps"] == report["checked_constraint_steps"] == 1
    assert calls == dict(predictor=2, energy=1, advance=1)
    assert report["all_committed_geometry_valid"] is True
    rejected = report["events"][-1]
    assert rejected["step"] == 2
    assert rejected["status"] == "timestep_rejected"
    assert rejected["failure_kind"] == "invalid_mechanics_state"
    candidate = rejected["candidates"][rejected["selected_candidate"]]
    assert candidate["correction"]["geometry"]["inverted_tetrahedra"] == 1
    assert candidate["correction"]["mechanics_domain"]["status"] == "fail"
    assert candidate["correction"]["energy"]["energy_satisfied"] is None
    assert json.loads((output / "report.json").read_text())["events"] == report["events"]
    events = [json.loads(line) for line in (output / "progress.jsonl").read_text().splitlines()]
    assert [event["status"] for event in events] == ["committed", "timestep_rejected"]
    assert progress[-1]["event"] == "rollout_step_failed"
    assert progress[-1]["status"] == "timestep_rejected"
    with np.load(output / "trajectory.npz", allow_pickle=False) as saved:
        assert_allclose(saved["times"], [0., .01])
        assert_array_equal(saved["positions"][0], topology.rest_positions)
        assert_array_equal(saved["positions"][1], topology.rest_positions + [.125, 0., 0.])
        assert_array_equal(saved["velocities"], np.zeros((2, 8, 3)))
        assert_array_equal(saved["broken"], np.zeros((2, 1), dtype=bool))
        assert_array_equal(saved["fixed"], np.stack([fixed, fixed]))
        assert (determinants(saved["positions"]) > 0).all()
    assert len(json.loads((output / "replay.json").read_text())["frames"]) == 2


def test_compare_runs_central_and_central_pcg_after_raw_orientation_rejection(tmp_path, monkeypatch):
    checkpoint, initial, _, _ = portable_rollout_inputs(tmp_path, has_corrector=False)

    def predict(self, batch):
        positions = batch["positions"].clone()
        positions[0] = positions[1:4].sum(dim=0) - 2 * positions[0]
        return prediction(batch, positions)

    monkeypatch.setattr(TemporalDynamicsModel, "forward", predict)
    output = tmp_path / "comparison"
    report = compare_dynamics(checkpoint, initial, output, 1,
                              orientation_config(mechanics_mode="exact", energy_balance=False), "cpu",
                              ["raw", "central", "central_pcg"])
    assert report["status"] == "comparison_finished"
    assert report["attempted_methods"] == ["raw", "central", "central_pcg"]
    assert report["unattempted_methods"] == []
    assert report["all_rollouts_completed"] is False
    assert report["conditions"]["raw"]["status"] == "timestep_rejected"
    assert report["conditions"]["raw"]["completed_steps"] == 0
    for method in ("central", "central_pcg"):
        assert report["conditions"][method]["status"] == "completed"
        assert report["conditions"][method]["completed_steps"] == 1
        assert report["conditions"][method]["predictor_calls"] == 1
        assert report["conditions"][method]["optimization_solves"] > 0
        assert (output / method / "report.json").is_file()
    assert report["central_central_pcg_agreement"]["both_rollouts_completed"] is True


def test_orientation_guard_does_not_hide_unrelated_value_errors(tmp_path, monkeypatch):
    checkpoint, initial, _, _ = portable_rollout_inputs(tmp_path, has_corrector=False)
    monkeypatch.setattr(TemporalDynamicsModel, "forward", lambda self, batch: prediction(batch, batch["positions"]))

    def broken_energy(*args, **kwargs):
        raise ValueError("unexpected energy evaluator defect")

    monkeypatch.setattr(runtime, "step_energy_diagnostics", broken_energy)
    with pytest.raises(ValueError, match="unexpected energy evaluator defect"):
        runtime.rollout_dynamics(checkpoint, initial, tmp_path / "raw", 1, "raw", orientation_config(), "cpu")


def test_raw_positive_orientation_geometry_violation_remains_diagnostic_only(tmp_path, monkeypatch):
    checkpoint, initial, _, _ = portable_rollout_inputs(tmp_path, has_corrector=False)

    def predict(self, batch):
        positions = batch["positions"].clone()
        positions[1, 0] += .125
        return prediction(batch, positions)

    monkeypatch.setattr(TemporalDynamicsModel, "forward", predict)
    output = tmp_path / "raw"
    report = runtime.rollout_dynamics(checkpoint, initial, output, 2, "raw", orientation_config(), "cpu")
    assert report["status"] == "completed"
    assert report["completed_steps"] == 2
    assert report["all_committed_geometry_valid"] is False
    for event in report["events"]:
        assert event["status"] == "committed"
        geometry = event["candidates"][event["selected_candidate"]]["correction"]["geometry"]
        assert geometry["valid"] is False
        assert geometry["inverted_tetrahedra"] == 0
    with np.load(output / "trajectory.npz", allow_pickle=False) as saved:
        assert (determinants(saved["positions"]) > 0).all()


def test_invalid_candidate_keeps_original_raw_fracture_ranking(tmp_path, monkeypatch):
    checkpoint, initial, _, _ = portable_rollout_inputs(tmp_path, has_corrector=False)
    diagnose = runtime.geometry_diagnostics

    def geometry(mesh, positions, broken, *args, **kwargs):
        report = diagnose(mesh, positions, broken, *args, **kwargs)
        return {**report, "valid": False, "inverted_tetrahedra": 1} if broken[0] else report

    monkeypatch.setattr(TemporalDynamicsModel, "forward", lambda self, batch:
                        prediction(batch, batch["positions"], fracture_probability=.9))
    monkeypatch.setattr(runtime, "geometry_diagnostics", geometry)
    report = runtime.rollout_dynamics(checkpoint, initial, tmp_path / "raw", 1, "raw", orientation_config(), "cpu")
    assert report["status"] == "timestep_rejected"
    assert report["completed_steps"] == 0
    event = report["events"][-1]
    assert event["selected_candidate"] == 0
    assert event["candidates"][0]["broken_faces"] == 1
    assert event["candidates"][0]["prior"] < event["candidates"][1]["prior"]
    assert event["candidates"][1]["status"] == "pruned_by_nonnegative_correction_cost_lower_bound"
