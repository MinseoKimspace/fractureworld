import json

import numpy as np
from numpy.testing import assert_allclose, assert_array_equal
import pytest
import torch

from fracture_world import dynamics_runtime as runtime
from fracture_world.dynamics_model import TemporalDynamicsModel
from fracture_world.io import file_hash
from test_dynamics_runtime import portable_rollout_inputs


METHODS = ("no_motion_central", "no_motion_consensus")


def freefall_config(**changes):
    settings = dict(fracture_samples=1, flow_steps=1, fracture_policy="shared_sample",
                    max_iterations=10000, max_contact_iterations=3)
    return runtime.RolloutConfig(**{**settings, **changes})


@pytest.mark.parametrize("method", METHODS)
def test_physical_motion_ablation_runs_200_autonomous_steps_without_teacher_reset(tmp_path, monkeypatch, method):
    checkpoint, initial, topology, _ = portable_rollout_inputs(tmp_path, has_corrector=False)
    checksum = file_hash(checkpoint)
    batch_inputs = []
    build_batch = runtime.state_batch

    def observe(topology, positions, velocities, broken, fixed, *args, **kwargs):
        batch_inputs.append(tuple(np.array(value, copy=True) for value in (positions, velocities, broken, fixed)))
        return build_batch(topology, positions, velocities, broken, fixed, *args, **kwargs)

    monkeypatch.setattr(runtime, "state_batch", observe)
    output = tmp_path / method
    result = runtime.rollout_dynamics(checkpoint, initial, output, 200, method, freefall_config(), "cpu")
    assert result["status"] == "completed"
    assert result["completed_steps"] == result["requested_steps"] == 200
    assert result["predictor_calls"] == len(batch_inputs) == 200
    assert result["fracture_flow_calls"] == 200
    assert result["learned_motion_corrector_calls"] == 0
    assert result["neural_corrector_calls"] == 0
    assert result["solver_method"] == method.removeprefix("no_motion_")
    assert result["motion_source"] == "physical_prior"
    assert result["fully_neural_free_baseline"] is False
    assert result["teacher_future_used"] is False
    assert result["engine_calls"] == 0
    assert result["optimization_solves"] >= 200
    assert result["all_committed_geometry_valid"] is True
    assert file_hash(checkpoint) == checksum
    with np.load(output / "trajectory.npz", allow_pickle=False) as saved:
        assert saved["positions"].shape == (201, 8, 3)
        for step, observed in enumerate(batch_inputs):
            for key, value in zip(("positions", "velocities", "broken", "fixed"), observed, strict=True):
                assert_array_equal(value, saved[key][step])
        assert_allclose(saved["velocities"][-1], np.broadcast_to([0., 0., -2.], (8, 3)), atol=3.e-5)
        assert_allclose(saved["positions"][-1], topology.rest_positions + [0., 0., -2.01], atol=3.e-5)
        assert np.all(np.diff(saved["broken"].astype(int), axis=0) >= 0)
    replay = json.loads((output / "replay.json").read_text())
    assert len(replay["frames"]) == 201
    assert replay["metadata"]["method"] == method


@pytest.mark.parametrize("method", METHODS)
@pytest.mark.parametrize("poison_value", (1234., float("nan")))
def test_autonomous_ablation_discards_every_learned_motion_output_and_preserves_release(tmp_path, monkeypatch, method, poison_value):
    checkpoint, initial, topology, fixed = portable_rollout_inputs(tmp_path, learned_supports=True, has_corrector=False)
    predict = TemporalDynamicsModel.forward_with_flow_context
    proposals = []

    def poisoned(self, batch):
        proposal, context = predict(self, batch)
        for key in ("positions", "velocities", "position_increment", "velocity_increment",
                    "unconstrained_position_increment", "unconstrained_velocity_increment"):
            proposal[key] = torch.full_like(batch["positions"], poison_value)
        proposals.append(proposal)
        return proposal, context

    monkeypatch.setattr(TemporalDynamicsModel, "forward_with_flow_context", poisoned)
    monkeypatch.setattr(TemporalDynamicsModel, "sample_fracture",
                        lambda self, batch, steps, noise, *, context=None: torch.ones_like(noise))
    output = tmp_path / method
    result = runtime.rollout_dynamics(checkpoint, initial, output, 2, method, freefall_config(), "cpu")
    assert result["completed_steps"] == 2
    assert len(proposals) == 2
    assert result["learned_motion_corrector_calls"] == 0
    assert result["support_policy"] == "learned_material_corner_release"
    with np.load(output / "trajectory.npz", allow_pickle=False) as saved:
        assert_array_equal(saved["fixed"][0], fixed)
        assert not saved["fixed"][1:].any()
        assert saved["broken"][1:].all()
        assert_allclose(saved["velocities"][-1], np.broadcast_to([0., 0., -.02], (8, 3)), atol=2.e-6)
        assert_allclose(saved["positions"][-1], topology.rest_positions + [0., 0., -.0003], atol=2.e-6)


def test_first_step_fm_pool_is_shared_across_learned_and_physical_motion_arms(tmp_path):
    checkpoint, initial, _, _ = portable_rollout_inputs(tmp_path, has_corrector=False)
    pools = []
    for method in ("central", "consensus", *METHODS):
        result = runtime.rollout_dynamics(checkpoint, initial, tmp_path / method, 1, method, freefall_config(), "cpu")
        assert result["completed_steps"] == 1
        pools.append(result["events"][0]["candidate_pool"])
    assert all(pool == pools[0] for pool in pools[1:])


@pytest.mark.parametrize("method", METHODS)
def test_physical_motion_arms_apply_the_same_mechanics_energy_and_fracture_controls(tmp_path, method):
    checkpoint, initial, _, _ = portable_rollout_inputs(tmp_path, broken=True, has_corrector=False)
    config = freefall_config(mechanics_mode="proximal", fracture_initiation=True, energy_balance=True)
    result = runtime.rollout_dynamics(checkpoint, initial, tmp_path / method, 1, method, config, "cpu")
    assert result["completed_steps"] == 1
    assert result["fracture_initiation_enforced"] is True
    assert result["energy_balance_enforced"] is True
    assert result["mechanics_mode"] == "proximal"
    assert result["optimization_solves"] > 0
    assert result["all_committed_energy_valid"] is True


@pytest.mark.parametrize("method", METHODS)
def test_autonomous_motion_ablation_does_not_pad_an_interrupted_horizon(tmp_path, method):
    checkpoint, initial, _, _ = portable_rollout_inputs(tmp_path, has_corrector=False)
    committed = []

    def progress(event):
        if "step" in event:
            committed.append(event["step"])

    output = tmp_path / method
    result = runtime.rollout_dynamics(checkpoint, initial, output, 200, method, freefall_config(), "cpu",
                                      progress=progress, cancel=lambda: len(committed) >= 2)
    assert result["status"] == "interrupted"
    assert result["requested_steps"] == 200
    assert result["completed_steps"] == 2
    assert committed == [1, 2]
    with np.load(output / "trajectory.npz", allow_pickle=False) as saved:
        assert len(saved["times"]) == 3


@pytest.mark.parametrize("method", METHODS)
def test_autonomous_motion_ablation_rejects_unsolved_steps_without_faking_predictions(tmp_path, monkeypatch, method):
    checkpoint, initial, _, _ = portable_rollout_inputs(tmp_path, has_corrector=False)
    solvers = []

    def reject(*args, **kwargs):
        solvers.append(args[8])
        return None, dict(status="PrimalInfeasible", solves=[])

    monkeypatch.setattr(runtime, "_correct", reject)
    output = tmp_path / method
    result = runtime.rollout_dynamics(checkpoint, initial, output, 200, method, freefall_config(), "cpu")
    assert result["status"] == "timestep_rejected"
    assert result["completed_steps"] == 0
    assert solvers == [method.removeprefix("no_motion_")]
    assert result["all_committed_geometry_valid"] is None
    with np.load(output / "trajectory.npz", allow_pickle=False) as saved:
        assert len(saved["times"]) == 1
