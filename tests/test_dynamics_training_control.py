from copy import deepcopy
from dataclasses import replace
import json
import math

import pytest
import torch

from fracture_world import dynamics_learning
from fracture_world.dynamics_learning import (
    DynamicsTrainingConfig,
    _TrainingTrajectory,
    _rollout_loss,
    _rollout_loss_batch,
    _scales,
    _validation_generator,
    load_dynamics_model,
    train_dynamics,
    train_neural_corrector,
    transition_batch,
)
from fracture_world.dynamics_model import NeuralDynamicsCorrector, TemporalDynamicsModel
from fracture_world.io import file_hash
from test_dynamics_data import recorded_trajectory


def small_training_config(**updates):
    values = dict(epochs=1, hidden=12, heads=3, graph_layers=2, device="cpu")
    return DynamicsTrainingConfig(**{**values, **updates})


@pytest.fixture
def held_out_data(tmp_path, monkeypatch):
    data = tmp_path / "data"
    recorded_trajectory(data)
    source = dynamics_learning.load_dynamics_dataset(data)[0]
    trajectories = [replace(source, case_id=f"case-{index}", condition_digest=f"{index:064x}")
                    for index in range(3)]
    monkeypatch.setattr(dynamics_learning, "load_dynamics_dataset", lambda *args, **kwargs: trajectories)
    return data, trajectories


@pytest.fixture
def predictor(held_out_data, tmp_path):
    data, trajectories = held_out_data
    path = train_dynamics(data, tmp_path / "predictor", small_training_config())
    return data, trajectories, path


def controlled_losses(monkeypatch, validation_losses):
    validation_calls = 0

    def loss(network, trajectory, frame, device, config, **kwargs):
        nonlocal validation_calls
        if network.training:
            value = (network.motion_head.bias[0] - 1).square()
        else:
            index = validation_calls // len(trajectory)
            value = network.motion_head.bias.new_tensor(validation_losses[index])
            validation_calls += 1
        terms = {name: value.detach() for name in ("motion", "fracture", "flow", "support_release")}
        return value, terms

    monkeypatch.setattr(dynamics_learning, "_rollout_loss", loss)


def test_corrector_keeps_epoch_zero_baseline_when_validation_worsens(predictor, tmp_path, monkeypatch):
    data, _, path = predictor
    original_hash = file_hash(path)
    controlled_losses(monkeypatch, [1.0, 2.0, 3.0])
    baseline_states = []
    save = dynamics_learning._atomic_checkpoint

    def capture_baseline(destination, checkpoint):
        if checkpoint.get("selection", {}).get("epoch") == 0 and not baseline_states:
            baseline_states.append(deepcopy(checkpoint["corrector"]))
        save(destination, checkpoint)

    monkeypatch.setattr(dynamics_learning, "_atomic_checkpoint", capture_baseline)
    output = train_neural_corrector(data, path, tmp_path / "corrector", epochs=2, device="cpu")
    best = torch.load(output, weights_only=True)
    last = torch.load(output.parent / "last.pt", weights_only=True)
    assert best["selection"] == dict(stage="neural_corrector", epoch=0, loss=1.0, split="validation")
    assert best["training_baseline"]["losses"]["loss"] == 1.0
    assert best["training_baseline"]["epoch"] == 0
    assert len(baseline_states) == 1
    for name, value in baseline_states[0].items():
        torch.testing.assert_close(best["corrector"][name], value, atol=0, rtol=0)
    assert not torch.equal(last["corrector"]["motion_head.bias"], best["corrector"]["motion_head.bias"])
    assert file_hash(path) == original_hash
    records = [json.loads(line) for line in (output.parent / "training.jsonl").read_text().splitlines()]
    assert [record["baseline_loss"] for record in records] == [1.0, 1.0]
    assert [record["best_loss"] for record in records] == [1.0, 1.0]


def test_scheduler_and_early_stopping_resume_with_identical_controller_state(predictor, tmp_path, monkeypatch):
    data, _, path = predictor
    options = dict(epochs=6, device="cpu", lr_patience=1, lr_factor=0.5,
                   early_stopping_patience=3, min_improvement=0.05)
    controlled_losses(monkeypatch, [10.0, 9.9, 9.8, 9.7])
    complete = train_neural_corrector(data, path, tmp_path / "complete", **options)
    expected = torch.load(complete.parent / "last.pt", weights_only=True)
    records = [json.loads(line) for line in (complete.parent / "training.jsonl").read_text().splitlines()]
    assert [record["epoch"] for record in records] == [1, 2, 3]
    assert records[1]["next_learning_rate"] == pytest.approx(records[0]["learning_rate"] * 0.5)
    assert records[-1]["stop_reason"] == "validation_plateau"
    assert expected["training_state"]["completed_epoch"] == 3
    assert expected["training_state"]["bad_epochs"] == 3
    assert expected["training_state"]["significant_best_loss"] == 10.0
    assert expected["selection"]["epoch"] == 3
    assert expected["training_state"]["scheduler"]["last_epoch"] == 4

    controlled_losses(monkeypatch, [10.0, 9.9, 9.8, 9.7])
    save = dynamics_learning._atomic_checkpoint

    def interrupt_epoch(destination, checkpoint):
        save(destination, checkpoint)
        if checkpoint.get("training_state", {}).get("completed_epoch") == 2:
            raise KeyboardInterrupt("completed scheduled epoch")

    monkeypatch.setattr(dynamics_learning, "_atomic_checkpoint", interrupt_epoch)
    directory = tmp_path / "resumed"
    with pytest.raises(KeyboardInterrupt, match="completed scheduled epoch"):
        train_neural_corrector(data, path, directory, **options)
    interrupted = torch.load(directory / "last.pt", weights_only=True)
    assert interrupted["training_state"]["scheduler"]["last_epoch"] == 3
    monkeypatch.setattr(dynamics_learning, "_atomic_checkpoint", save)
    resumed = train_neural_corrector(data, path, directory, epochs=6, device="cpu", resume=directory / "last.pt")
    actual = torch.load(resumed.parent / "last.pt", weights_only=True)
    for name in ("model", "corrector"):
        for key, value in expected[name].items():
            torch.testing.assert_close(actual[name][key], value, atol=0, rtol=0)
    for name in ("completed_epoch", "best_loss", "significant_best_loss", "bad_epochs", "scheduler", "stop_reason"):
        assert actual["training_state"][name] == expected["training_state"][name]
    expected_optimizer = expected["training_state"]["optimizer"]
    actual_optimizer = actual["training_state"]["optimizer"]
    assert actual_optimizer["param_groups"] == expected_optimizer["param_groups"]
    assert actual_optimizer["state"].keys() == expected_optimizer["state"].keys()
    for parameter, state in expected_optimizer["state"].items():
        for key, value in state.items():
            torch.testing.assert_close(actual_optimizer["state"][parameter][key], value, atol=0, rtol=0)
    expected_random = expected["training_state"]["random_state"]
    actual_random = actual["training_state"]["random_state"]
    torch.testing.assert_close(actual_random["torch"], expected_random["torch"], atol=0, rtol=0)
    assert actual_random["python"] == expected_random["python"]
    assert actual_random["numpy"] == expected_random["numpy"]
    assert actual["selection"] == expected["selection"]

    previous_records = (directory / "training.jsonl").read_text()
    train_neural_corrector(data, path, directory, epochs=10, device="cpu", resume=directory / "last.pt")
    assert (directory / "training.jsonl").read_text() == previous_records
    assert torch.load(directory / "last.pt", weights_only=True)["training_state"]["completed_epoch"] == 3


@pytest.mark.parametrize("updates", [
    dict(batch_size=0), dict(batch_size=-1), dict(batch_size=True), dict(batch_size=1.5),
    dict(lr_patience=-1), dict(lr_patience=True), dict(lr_patience=1.5),
    dict(early_stopping_patience=-1), dict(early_stopping_patience=True), dict(early_stopping_patience=1.5),
    dict(lr_factor=0.0), dict(lr_factor=1.0), dict(lr_factor=float("nan")),
    dict(min_learning_rate=0.0), dict(min_learning_rate=-1.0), dict(min_learning_rate=float("nan")),
    dict(min_improvement=-0.1), dict(min_improvement=1.0), dict(min_improvement=float("nan")),
    dict(lr_patience=1, min_learning_rate=0.1),
])
def test_invalid_training_control_is_rejected_before_loading_data(tmp_path, monkeypatch, updates):
    def reject_loading(*args, **kwargs):
        pytest.fail("invalid training controls reached dataset loading")

    monkeypatch.setattr(dynamics_learning, "load_dynamics_dataset", reject_loading)
    with pytest.raises(ValueError):
        train_dynamics(tmp_path / "missing", tmp_path / "unused", small_training_config(**updates))
    assert not (tmp_path / "unused").exists()


@pytest.mark.parametrize("updates", [dict(lr_patience=1), dict(early_stopping_patience=1)])
def test_validation_controls_require_held_out_validation(held_out_data, tmp_path, updates):
    data, _ = held_out_data
    with pytest.raises(ValueError, match="held-out validation"):
        train_dynamics(data, tmp_path / "unused", small_training_config(overfit=True, **updates))
    assert not (tmp_path / "unused").exists()


@pytest.mark.parametrize("corrector", [False, True])
def test_batched_rollout_matches_window_mean_loss_and_gradients_with_unequal_horizons(held_out_data, corrector):
    _, trajectories = held_out_data
    trajectory = trajectories[0]
    config = small_training_config(rollout_steps=2)
    torch.manual_seed(401)
    model_options = dict(hidden=12, heads=3, graph_layers=2, transformer_layers=1, **_scales([trajectory]))
    model_type = NeuralDynamicsCorrector if corrector else TemporalDynamicsModel
    serial = model_type(**model_options)
    batched = deepcopy(serial)
    fixed = TemporalDynamicsModel(**model_options).eval().requires_grad_(False) if corrector else None
    fixed_batched = deepcopy(fixed)
    frames = [0, len(trajectory) - 1]
    assert [min(config.rollout_steps, len(trajectory) - frame) for frame in frames] == [2, 1]
    serial_results = [_rollout_loss(serial, trajectory, frame, "cpu", config, fixed_predictor=fixed,
                                   flow_generator=_validation_generator(trajectory.case_id, frame, 17, "cpu"))
                      for frame in frames]
    expected = torch.stack([result[0] for result in serial_results]).mean()
    expected.backward()
    prepared = _TrainingTrajectory(trajectory, "cpu")
    original_base = {name: value.clone() for name, value in prepared.base.items() if isinstance(value, torch.Tensor)}
    original_frames = {name: value.clone() for name, value in prepared.frames.items()}
    actual, terms = _rollout_loss_batch(batched, trajectory, frames, "cpu", config, fixed_predictor=fixed_batched,
                                       prepared=prepared, flow_generators=[
                                           _validation_generator(trajectory.case_id, frame, 17, "cpu") for frame in frames])
    actual.backward()
    torch.testing.assert_close(actual, expected, atol=1.e-6, rtol=1.e-5)
    for name in terms:
        expected_term = torch.stack([result[1][name] for result in serial_results]).mean()
        torch.testing.assert_close(terms[name], expected_term, atol=1.e-6, rtol=1.e-5)
    for (name, parameter), (other_name, reference) in zip(batched.named_parameters(), serial.named_parameters()):
        assert name == other_name
        if reference.grad is None:
            assert parameter.grad is None
        else:
            torch.testing.assert_close(parameter.grad, reference.grad, atol=3.e-6, rtol=1.e-4)
    for name, value in original_base.items():
        torch.testing.assert_close(prepared.base[name], value, atol=0, rtol=0)
    for name, value in original_frames.items():
        torch.testing.assert_close(prepared.frames[name], value, atol=0, rtol=0)
    if fixed_batched is not None:
        assert all(parameter.grad is None for parameter in fixed_batched.parameters())


def test_batched_corrector_training_and_direct_resume_preserve_frozen_predictor(predictor, tmp_path, monkeypatch):
    data, trajectories, path = predictor
    original_hash = file_hash(path)
    original = torch.load(path, weights_only=True)
    complete = train_neural_corrector(data, path, tmp_path / "complete", epochs=2, device="cpu", batch_size=2)
    expected = torch.load(complete.parent / "last.pt", weights_only=True)
    records = [json.loads(line) for line in (complete.parent / "training.jsonl").read_text().splitlines()]
    training = next(item for item in trajectories if item.case_id in original["splits"]["train"])
    assert [record["split_timing"]["train"]["optimizer_steps"] for record in records] == [math.ceil(len(training) / 2)] * 2
    assert all(record["split_timing"]["train"]["windows"] == len(training) for record in records)
    assert expected["training"]["batch_size"] == 2
    for name, value in original["model"].items():
        torch.testing.assert_close(expected["model"][name], value, atol=0, rtol=0)
    assert file_hash(path) == original_hash

    save = dynamics_learning._atomic_checkpoint

    def interrupt_epoch(destination, checkpoint):
        save(destination, checkpoint)
        if checkpoint.get("training_state", {}).get("completed_epoch") == 1:
            raise KeyboardInterrupt("completed batched epoch")

    monkeypatch.setattr(dynamics_learning, "_atomic_checkpoint", interrupt_epoch)
    directory = tmp_path / "resumed"
    with pytest.raises(KeyboardInterrupt, match="completed batched epoch"):
        train_neural_corrector(data, path, directory, epochs=2, device="cpu", batch_size=2)
    monkeypatch.setattr(dynamics_learning, "_atomic_checkpoint", save)
    resumed = train_neural_corrector(data, path, directory, epochs=2, device="cpu", resume=directory / "last.pt")
    actual = torch.load(resumed.parent / "last.pt", weights_only=True)
    for name in ("model", "corrector"):
        for key, value in expected[name].items():
            torch.testing.assert_close(actual[name][key], value, atol=0, rtol=0)
    assert actual["training"]["batch_size"] == 2
    assert actual["predictor_checkpoint_sha256"] == original_hash
    assert file_hash(path) == original_hash

    changed = tmp_path / "different-predictor.pt"
    torch.save({**original, "different_serialization": True}, changed)
    with pytest.raises(ValueError, match="original frozen predictor"):
        train_neural_corrector(data, changed, tmp_path / "rejected", epochs=3, device="cpu", resume=directory / "last.pt")
    assert not (tmp_path / "rejected").exists()


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required for end-to-end GPU training")
def test_cuda_corrector_training_batches_rollouts_and_reloads_on_both_devices(predictor, tmp_path):
    data, trajectories, path = predictor
    original_hash = file_hash(path)
    original = torch.load(path, map_location="cpu", weights_only=True)
    output = train_neural_corrector(data, path, tmp_path / "cuda-corrector", epochs=1, device="cuda",
                                    batch_size=2, rollout_steps=2, lr_patience=1, early_stopping_patience=2)
    trained = torch.load(output, map_location="cpu", weights_only=True)
    assert file_hash(path) == original_hash
    for name, value in original["model"].items():
        torch.testing.assert_close(trained["model"][name], value, atol=0, rtol=0)
    assert trained["training_baseline"]["epoch"] == 0
    assert trained["training_baseline"]["split"] == "validation"
    assert math.isfinite(trained["training_baseline"]["losses"]["loss"])
    records = [json.loads(line) for line in (output.parent / "training.jsonl").read_text().splitlines()]
    assert len(records) == 1
    assert records[0]["device"].startswith("cuda")
    assert records[0]["gpu_peak_memory_bytes"] > 0
    assert records[0]["batch_size"] == records[0]["rollout_steps"] == 2
    assert records[0]["baseline_loss"] == trained["training_baseline"]["losses"]["loss"]
    assert records[0]["split_timing"]["train"]["optimizer_steps"] == 2
    last = torch.load(output.parent / "last.pt", map_location="cpu", weights_only=True)
    assert last["training_state"]["device_type"] == "cuda"
    assert last["training_state"]["scheduler"] is not None

    predictions = {}
    for target in ("cpu", "cuda"):
        model, corrector, checkpoint, device = load_dynamics_model(output, target)
        assert device.type == target
        assert corrector is not None
        assert next(model.parameters()).device.type == target
        assert next(corrector.parameters()).device.type == target
        assert checkpoint["predictor_checkpoint_sha256"] == original_hash
        batch = transition_batch(trajectories[0], 0, device)
        with torch.no_grad():
            corrected = corrector(batch, model(batch))
        predictions[target] = {name: corrected[name].cpu() for name in ("positions", "velocities", "fracture_logits")}
        assert all(torch.isfinite(value).all() for value in predictions[target].values())
    for name, value in predictions["cpu"].items():
        torch.testing.assert_close(predictions["cuda"][name], value, atol=3.e-5, rtol=1.e-4)
