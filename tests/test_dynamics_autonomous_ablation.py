import json

import numpy as np
from numpy.testing import assert_allclose, assert_array_equal

from fracture_world import dynamics_runtime as runtime
from fracture_world.dynamics_model import NeuralDynamicsCorrector, TemporalDynamicsModel
from fracture_world.io import file_hash
from test_dynamics_runtime import portable_rollout_inputs


def freefall_config(**changes):
    settings = dict(fracture_samples=1, flow_steps=1, fracture_policy="shared_sample",
                    max_iterations=10000, max_contact_iterations=3)
    return runtime.RolloutConfig(**{**settings, **changes})


def test_raw_predictor_runs_200_steps_without_online_mechanical_evaluation(tmp_path, monkeypatch):
    checkpoint, initial, topology, _ = portable_rollout_inputs(tmp_path)
    checksum = file_hash(checkpoint)
    staged_states = []
    build_batch = runtime.state_batch

    def observe(topology, positions, velocities, broken, fixed, *args, **kwargs):
        staged_states.append(tuple(np.array(value, copy=True) for value in (positions, velocities, broken, fixed)))
        return build_batch(topology, positions, velocities, broken, fixed, *args, **kwargs)

    def forbidden(*args, **kwargs):
        raise AssertionError("raw inference invoked a mechanical evaluator or neural corrector")

    monkeypatch.setattr(runtime, "state_batch", observe)
    for name in ("_correct", "_problem", "step_contacts", "geometry_diagnostics", "swept_geometry_diagnostics",
                 "step_energy", "step_energy_diagnostics", "_mechanical_energy", "current_elastic_state",
                 "advance_elastic_state"):
        monkeypatch.setattr(runtime, name, forbidden)
    monkeypatch.setattr(NeuralDynamicsCorrector, "forward", forbidden)
    config = runtime.RolloutConfig(fracture_samples=0, fracture_policy="marginal_search")
    output = tmp_path / "raw"
    result = runtime.rollout_dynamics(checkpoint, initial, output, 200, "raw", config, "cpu")
    assert result["status"] == "completed"
    assert result["completed_steps"] == result["requested_steps"] == len(staged_states) == 200
    assert result["predictor_calls"] == 200
    assert result["neural_corrector_calls"] == result["optimization_solves"] == result["checked_constraint_steps"] == 0
    assert result["all_committed_geometry_valid"] is None
    assert result["all_committed_energy_valid"] is None
    assert result["all_committed_swept_geometry_valid"] is None
    assert file_hash(checkpoint) == checksum
    with np.load(output / "trajectory.npz", allow_pickle=False) as saved:
        assert saved["positions"].shape == (201, 8, 3)
        for step, observed in enumerate(staged_states):
            for key, value in zip(("positions", "velocities", "broken", "fixed"), observed, strict=True):
                assert_array_equal(value, saved[key][step])
        assert_allclose(saved["velocities"][-1], np.broadcast_to([0., 0., -2.], (8, 3)), atol=3.e-5)
        assert_allclose(saved["positions"][-1], topology.rest_positions + [0., 0., -2.01], atol=3.e-5)
        assert np.all(np.diff(saved["broken"].astype(int), axis=0) >= 0)
        assert not saved["fixed"].any()


def test_physics_baseline_runs_200_autonomous_steps_without_teacher_reset(tmp_path, monkeypatch):
    checkpoint, initial, topology, _ = portable_rollout_inputs(tmp_path, has_corrector=False)
    checksum = file_hash(checkpoint)

    def forbidden(*args, **kwargs):
        raise AssertionError("physics baseline invoked a learned predictor")

    monkeypatch.setattr(TemporalDynamicsModel, "forward", forbidden)
    monkeypatch.setattr(TemporalDynamicsModel, "forward_with_flow_context", forbidden)
    output = tmp_path / "physics"
    result = runtime.rollout_dynamics(checkpoint, initial, output, 200, "physics", freefall_config(), "cpu")
    assert result["status"] == "completed"
    assert result["completed_steps"] == result["requested_steps"] == 200
    assert result["predictor_calls"] == result["fracture_flow_calls"] == 0
    assert result["neural_corrector_calls"] == result["optimization_solves"] == 0
    assert result["teacher_future_used"] is False
    assert result["engine_calls"] == 0
    assert result["all_committed_geometry_valid"] is None
    assert file_hash(checkpoint) == checksum
    with np.load(output / "trajectory.npz", allow_pickle=False) as saved:
        assert saved["positions"].shape == (201, 8, 3)
        for step, time in enumerate(saved["times"]):
            assert_allclose(saved["velocities"][step], np.broadcast_to([0., 0., -time], (8, 3)), atol=3.e-5)
            assert_allclose(saved["positions"][step], topology.rest_positions + [0., 0., -.5 * time**2], atol=3.e-5)
        assert_array_equal(saved["broken"], np.zeros((201, 1), dtype=bool))
    replay = json.loads((output / "replay.json").read_text())
    assert len(replay["frames"]) == 201
    assert replay["metadata"]["method"] == "physics"


def test_physics_baseline_does_not_pad_an_interrupted_horizon(tmp_path):
    checkpoint, initial, _, _ = portable_rollout_inputs(tmp_path, has_corrector=False)
    committed = []

    def progress(event):
        if "step" in event:
            committed.append(event["step"])

    output = tmp_path / "physics"
    result = runtime.rollout_dynamics(checkpoint, initial, output, 200, "physics", freefall_config(), "cpu",
                                      progress=progress, cancel=lambda: len(committed) >= 2)
    assert result["status"] == "interrupted"
    assert result["requested_steps"] == 200
    assert result["completed_steps"] == 2
    assert committed == [1, 2]
    with np.load(output / "trajectory.npz", allow_pickle=False) as saved:
        assert len(saved["times"]) == 3
