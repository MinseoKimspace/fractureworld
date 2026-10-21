from dataclasses import replace
import json

import numpy as np
import pytest
import torch

from fracture_world import dynamics_learning
from fracture_world.dynamics_data import load_dynamics_trajectory, split_dynamics_cases
from fracture_world.dynamics_learning import (
    DynamicsTrainingConfig,
    _loss,
    _scales,
    load_dynamics_model,
    train_dynamics,
    transition_batch,
)
from fracture_world.dynamics_model import TemporalDynamicsModel
from test_dynamics_data import recorded_trajectory


def training_config(**updates):
    values = dict(epochs=1, corrector_epochs=1, hidden=12, heads=3, graph_layers=2, transformer_layers=1,
                  device="cpu", overfit=True, learn_support_release=True)
    return DynamicsTrainingConfig(**{**values, **updates})


def test_required_event_coverage_rejects_debug_data_with_structured_reasons(tmp_path):
    data = tmp_path / "data"
    recorded_trajectory(data, supports=True)
    with pytest.raises(ValueError, match="dataset is not ready"):
        train_dynamics(data, tmp_path / "trained", training_config(require_event_coverage=True))


def test_recorded_data_trains_predictor_corrector_and_reloads_release_supervision(tmp_path):
    data = tmp_path / "data"
    recorded_trajectory(data, supports=True)
    checkpoint_path = train_dynamics(data, tmp_path / "trained", training_config())
    model, corrector, checkpoint, device = load_dynamics_model(checkpoint_path, "cpu")
    assert device.type == "cpu"
    assert corrector is not None
    assert checkpoint["model_config"]["learn_support_release"]
    assert checkpoint["support_release_training_events"] == 2
    assert checkpoint["fracture_training_events"] == 3
    assert checkpoint["class_weights"]["fracture_positive"] == 1
    assert checkpoint["class_weights"]["support_release_positive"] == 3.5
    assert checkpoint["runtime_engine_calls"] == 0
    assert checkpoint["training"]["overfit"]
    assert checkpoint["condition_splits"]["validation"] == []
    records = [json.loads(line) for line in (tmp_path / "trained" / "training.jsonl").read_text().splitlines()]
    assert [record["stage"] for record in records] == ["predictor", "neural_corrector"]
    assert all(np.isfinite(record["losses"]["train"]["loss"]) for record in records)
    trajectory = dynamics_learning.load_dynamics_dataset(data)[0]
    batch = transition_batch(trajectory, 0, device)
    with torch.no_grad():
        proposal = model(batch)
        corrected = corrector(batch, proposal)
    assert corrected["positions"].shape == trajectory.positions[0].shape
    assert corrected["support_release_logits"].shape == trajectory.fixed[0].shape
    assert torch.isfinite(corrected["fracture_logits"]).all()


def test_combined_supervision_reaches_motion_fracture_flow_and_release_heads(tmp_path):
    case, path = recorded_trajectory(tmp_path, supports=True)
    trajectory = load_dynamics_trajectory(case, path)
    model = TemporalDynamicsModel(hidden=12, heads=3, graph_layers=2, learn_support_release=True, **_scales([trajectory]))
    transition = trajectory.transition(2)
    batch = transition_batch(trajectory, 2, "cpu")
    tensors = [torch.tensor(np.array(value, copy=True), dtype=dtype) for value, dtype in (
        (transition.target_positions, torch.float32), (transition.target_velocities, torch.float32),
        (transition.target_broken, torch.bool), (transition.target_fixed, torch.bool))]
    loss, terms = _loss(model, batch, *tensors, training_config(), support_positive_weight=3.5)
    loss.backward()
    assert np.isfinite(loss.item())
    for head in (model.motion_head, model.fracture_head, model.flow_head[-1], model.support_head):
        assert head.weight.grad is not None and head.weight.grad.abs().sum() > 0
    assert terms["support_release"] > 0


def test_transition_batch_contains_current_state_and_ignores_changed_future_targets(tmp_path):
    case, path = recorded_trajectory(tmp_path)
    trajectory = load_dynamics_trajectory(case, path)
    before = transition_batch(trajectory, 0, "cpu")
    altered_positions, altered_velocities = trajectory.positions.copy(), trajectory.velocities.copy()
    altered_positions[1:] += 1.e10
    altered_velocities[1:] -= 1.e12
    after = transition_batch(replace(trajectory, positions=altered_positions, velocities=altered_velocities), 0, "cpu")
    assert not any(key.startswith("target") for key in before)
    for key in before:
        if isinstance(before[key], torch.Tensor):
            torch.testing.assert_close(after[key], before[key])
        else:
            assert after[key] == before[key]


def test_normalization_and_class_weights_are_computed_from_training_split_only(tmp_path, monkeypatch):
    case, path = recorded_trajectory(tmp_path / "source", supports=True)
    source = load_dynamics_trajectory(case, path)
    cases = [replace(source, case_id=f"case-{index}", condition_digest=f"{index:064x}") for index in range(3)]
    split = split_dynamics_cases(cases, seed=17)
    training_id = split["train"][0].case_id
    expected = _scales(split["train"])
    cases = [item if item.case_id == training_id else replace(
        item, velocities=item.velocities * 1.e6, broken=np.zeros_like(item.broken), fixed=np.zeros_like(item.fixed))
             for item in cases]
    monkeypatch.setattr(dynamics_learning, "load_dynamics_dataset", lambda _: cases)
    checkpoint_path = train_dynamics(tmp_path / "unused", tmp_path / "trained",
                                    training_config(overfit=False, corrector_epochs=0))
    _, _, checkpoint, _ = load_dynamics_model(checkpoint_path, "cpu")
    assert checkpoint["splits"]["train"] == [training_id]
    assert checkpoint["class_weights"] == {"fracture_positive": 1.0, "support_release_positive": 3.5}
    assert checkpoint["support_release_training_events"] == 2
    assert checkpoint["fracture_training_events"] == 3
    for key, value in expected.items():
        np.testing.assert_allclose(checkpoint["model_config"][key], value)


def test_support_release_training_requires_explicit_option(tmp_path):
    recorded_trajectory(tmp_path / "data", supports=True)
    with pytest.raises(ValueError, match="support release"):
        train_dynamics(tmp_path / "data", tmp_path / "trained", training_config(learn_support_release=False))
