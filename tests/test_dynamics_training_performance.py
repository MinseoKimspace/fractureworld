from copy import deepcopy
from dataclasses import replace
import json

import numpy as np
import torch

from fracture_world import dynamics_learning
from fracture_world.dynamics_data import load_dynamics_trajectory
from fracture_world.dynamics_learning import _TrainingMetrics, _TrainingTrajectory, _rollout_loss, transition_batch
from test_dynamics_data import recorded_trajectory
from test_dynamics_learning import training_config
from test_dynamics_model import small_model


def test_device_staging_reuses_static_tensors_and_matches_each_observed_transition(tmp_path):
    case, path = recorded_trajectory(tmp_path, supports=True)
    trajectory = load_dynamics_trajectory(case, path)
    prepared = _TrainingTrajectory(trajectory, "cpu")
    first = prepared.batch(0)
    for frame in range(len(trajectory)):
        batch = prepared.batch(frame)
        expected = transition_batch(trajectory, frame, "cpu")
        assert batch.keys() == expected.keys()
        for key, value in expected.items():
            if isinstance(value, torch.Tensor):
                torch.testing.assert_close(batch[key], value, atol=0, rtol=0)
            else:
                assert batch[key] == value
        for key in ("rest_positions", "masses", "forces", "material", "face_pairs"):
            assert batch[key] is first[key]
        target = trajectory.transition(frame)
        for actual, expected in zip(prepared.targets(frame), (
            target.target_positions, target.target_velocities, target.target_broken, target.target_fixed
        )):
            torch.testing.assert_close(actual, torch.as_tensor(np.array(expected, copy=True), dtype=actual.dtype))


def test_staging_future_arrays_does_not_expose_future_state_in_model_inputs(tmp_path):
    case, path = recorded_trajectory(tmp_path, supports=True)
    trajectory = load_dynamics_trajectory(case, path)
    positions, velocities = trajectory.positions.copy(), trajectory.velocities.copy()
    positions[1:] += 100
    velocities[1:] -= 100
    altered = replace(trajectory, positions=positions, velocities=velocities)
    first = _TrainingTrajectory(trajectory, "cpu").batch(0)
    changed = _TrainingTrajectory(altered, "cpu").batch(0)
    for key, value in first.items():
        if isinstance(value, torch.Tensor):
            torch.testing.assert_close(value, changed[key], atol=0, rtol=0)
        else:
            assert value == changed[key]
    assert not any(key.startswith("target") for key in first)


def test_staged_autoregressive_loss_and_parameter_gradients_match_per_window_transfers(tmp_path):
    case, path = recorded_trajectory(tmp_path, supports=True)
    trajectory = load_dynamics_trajectory(case, path)
    torch.manual_seed(181)
    original = small_model()
    optimized = deepcopy(original)
    config = training_config(rollout_steps=2)
    torch.manual_seed(191)
    expected, expected_terms = _rollout_loss(original, trajectory, 0, "cpu", config)
    expected.backward()
    torch.manual_seed(191)
    actual, actual_terms = _rollout_loss(optimized, trajectory, 0, "cpu", config,
                                       prepared=_TrainingTrajectory(trajectory, "cpu"))
    actual.backward()
    torch.testing.assert_close(actual, expected)
    for key in expected_terms:
        torch.testing.assert_close(actual_terms[key], expected_terms[key])
        assert not actual_terms[key].requires_grad
    for parameter, reference in zip(optimized.parameters(), original.parameters()):
        if reference.grad is None:
            assert parameter.grad is None
        else:
            torch.testing.assert_close(parameter.grad, reference.grad)


def test_metrics_do_not_retain_training_graphs_and_preserve_window_weighting():
    metrics = _TrainingMetrics("cpu")
    assert metrics.scores() is None
    for value in (1.0, 2.0, 6.0):
        loss = torch.tensor(value, requires_grad=True)
        metrics.update(loss, {name: loss.detach() * 2 for name in metrics.names[1:]})
    assert metrics.count == 3
    assert not metrics.sums.requires_grad
    assert not metrics.last.requires_grad
    assert metrics.scores() == dict(loss=3.0, motion=6.0, fracture=6.0, flow=6.0, support_release=6.0)


def test_training_logs_measured_window_throughput_and_split_scoped_eta(tmp_path, monkeypatch, capsys):
    recorded_trajectory(tmp_path / "data")
    ticks = iter(range(1000))
    monkeypatch.setattr(dynamics_learning, "perf_counter", lambda: float(next(ticks)))
    dynamics_learning.train_dynamics(tmp_path / "data", tmp_path / "trained",
                                    training_config(corrector_epochs=0, stream_data=True))
    records = [json.loads(line) for line in capsys.readouterr().out.splitlines()]
    progress = [record for record in records if record.get("event") == "training_progress"]
    assert progress
    assert all(0 < item["windows_completed"] <= item["windows_total"] == 3 for item in progress)
    assert all(item["seconds_per_window"] > 0 and item["interval_seconds_per_window"] > 0 for item in progress)
    assert all(item["eta_seconds"] >= 0 and item["timestamp"].endswith("+00:00") for item in progress)
    epoch = next(item for item in records if "losses" in item)
    assert epoch["epoch_seconds"] > 0
    assert epoch["split_timing"]["train"]["windows"] == 3
    assert epoch["split_timing"]["validation"]["windows"] == 0
