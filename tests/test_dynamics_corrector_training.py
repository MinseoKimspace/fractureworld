from dataclasses import replace
import json

import pytest
import torch

from fracture_world import dynamics_learning
from fracture_world.dynamics_learning import (
    DynamicsTrainingConfig,
    load_dynamics_model,
    train_dynamics,
    train_neural_corrector,
    transition_batch,
)
from fracture_world.dynamics_model import NeuralDynamicsCorrector
from fracture_world.io import file_hash
from test_dynamics_data import recorded_trajectory


@pytest.fixture
def pretrained(tmp_path):
    data = tmp_path / "data"
    recorded_trajectory(data, supports=True)
    config = DynamicsTrainingConfig(epochs=1, hidden=12, heads=3, graph_layers=2, device="cpu",
                                    overfit=True, learn_support_release=True, flow_architecture="independent",
                                    motion_prior="backward_euler")
    path = train_dynamics(data, tmp_path / "predictor", config)
    return data, path


@pytest.mark.parametrize("legacy", [False, True])
def test_corrector_only_training_preserves_predictor_and_original_splits(pretrained, tmp_path, monkeypatch, legacy):
    data, path = pretrained
    if legacy:
        saved = torch.load(path, weights_only=True)
        for key in ("probability_policy", "training"):
            saved.pop(key)
        for key in ("flow_architecture", "motion_prior"):
            saved["model_config"].pop(key)
        torch.save(saved, path)
    original_hash = file_hash(path)
    before, _, original, _ = load_dynamics_model(path, "cpu")
    trajectory = dynamics_learning.load_dynamics_dataset(data)[0]
    batch = transition_batch(trajectory, 0, "cpu")
    with torch.no_grad():
        expected = before(batch)

    def reject_recomputation(*args, **kwargs):
        raise AssertionError("predictor normalization and dataset splits must be reused")

    loss = dynamics_learning._rollout_loss
    calls = []

    def inspect_corrector(network, *args, **kwargs):
        predictor = kwargs["fixed_predictor"]
        assert isinstance(network, NeuralDynamicsCorrector)
        assert predictor is not None and not predictor.training
        assert all(not parameter.requires_grad for parameter in predictor.parameters())
        calls.append(network.training)
        return loss(network, *args, **kwargs)

    monkeypatch.setattr(dynamics_learning, "_scales", reject_recomputation)
    monkeypatch.setattr(dynamics_learning, "split_dynamics_cases", reject_recomputation)
    monkeypatch.setattr(dynamics_learning, "_rollout_loss", inspect_corrector)
    output = train_neural_corrector(data, path, tmp_path / "corrector", epochs=1, rollout_steps=2,
                                    device="cpu", stream_data=True)
    after, corrector, trained, _ = load_dynamics_model(output, "cpu")
    assert calls and any(calls)
    assert file_hash(path) == original_hash
    assert corrector is not None
    assert trained["training_mode"] == "neural_corrector_only"
    assert trained["predictor_checkpoint_sha256"] == original_hash
    assert trained["splits"] == original["splits"]
    assert trained["condition_splits"] == original["condition_splits"]
    assert after.config == before.config
    for key, value in original["model"].items():
        torch.testing.assert_close(trained["model"][key], value, atol=0, rtol=0)
    with torch.no_grad():
        actual = after(batch)
        corrected = corrector(batch, actual)
    for key in ("positions", "velocities", "fracture_logits", "support_release_logits"):
        torch.testing.assert_close(actual[key], expected[key], atol=0, rtol=0)
        assert torch.isfinite(corrected[key]).all()
    records = [json.loads(line) for line in (output.parent / "training.jsonl").read_text().splitlines()]
    assert [record["stage"] for record in records] == ["neural_corrector"]


def test_corrector_only_resume_preserves_predictor_and_matches_uninterrupted_training(pretrained, tmp_path, monkeypatch):
    data, path = pretrained
    uninterrupted = train_neural_corrector(data, path, tmp_path / "uninterrupted", epochs=2, device="cpu")
    save = dynamics_learning._atomic_checkpoint

    def interrupt_epoch(path, checkpoint):
        save(path, checkpoint)
        if checkpoint.get("training_state", {}).get("completed_epoch") == 1:
            raise KeyboardInterrupt("completed corrector epoch")

    monkeypatch.setattr(dynamics_learning, "_atomic_checkpoint", interrupt_epoch)
    output = tmp_path / "resumed"
    with pytest.raises(KeyboardInterrupt, match="completed corrector epoch"):
        train_neural_corrector(data, path, output, epochs=2, device="cpu")
    last = torch.load(output / "last.pt", weights_only=True)
    monkeypatch.setattr(dynamics_learning, "_atomic_checkpoint", save)
    resumed = train_dynamics(data, output, DynamicsTrainingConfig(**last["training"]), resume=output / "last.pt")
    expected = torch.load(uninterrupted, weights_only=True)
    actual = torch.load(resumed, weights_only=True)
    for name in ("model", "corrector"):
        for key, value in expected[name].items():
            torch.testing.assert_close(actual[name][key], value, atol=0, rtol=0)


@pytest.mark.parametrize("change", ["missing_case", "condition", "archive", "overlap"])
def test_checkpoint_splits_reject_changed_or_overlapping_data(pretrained, change):
    data, path = pretrained
    checkpoint = torch.load(path, weights_only=True)
    trajectories = dynamics_learning.load_dynamics_dataset(data)
    if change == "missing_case":
        trajectories = []
    elif change == "condition":
        trajectories = [replace(trajectories[0], condition_digest="changed")]
    elif change == "archive":
        metadata = dict(trajectories[0].metadata)
        metadata["source_hashes"] = {"changed": "different digest"}
        trajectories = [replace(trajectories[0], metadata=metadata)]
    else:
        checkpoint["splits"]["test"] = list(checkpoint["splits"]["train"])
    with pytest.raises(ValueError):
        dynamics_learning._checkpoint_splits(trajectories, checkpoint)


def test_corrector_sources_allow_relocation_and_reject_changed_hashes():
    original = {"case": {"source_hashes": {"old/case.npz": "abc", "old/trajectory.npz": "def"},
                          "source_dataset": "old/dataset", "units": "SI"}}
    relocated = {"case": {"source_hashes": {"new/case.npz": "abc", "new/trajectory.npz": "def"},
                           "source_dataset": "new/dataset", "units": "SI"}}
    assert dynamics_learning._same_training_sources(original, relocated, allow_relocated=True)
    assert not dynamics_learning._same_training_sources(original, relocated)
    relocated["case"]["source_hashes"]["new/case.npz"] = "changed"
    assert not dynamics_learning._same_training_sources(original, relocated, allow_relocated=True)
    assert original["case"]["source_dataset"] == "old/dataset"
