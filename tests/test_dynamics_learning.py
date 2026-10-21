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
    _rollout_loss,
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
    checkpoint_path = train_dynamics(data, tmp_path / "trained", training_config(rollout_steps=2))
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
    assert checkpoint["training"]["rollout_steps"] == 2
    assert checkpoint["training_window_steps"] == 2
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


class RecordedAutoregressiveModel(torch.nn.Module):
    def __init__(self, invalid=False):
        super().__init__()
        self.shift = torch.nn.Parameter(torch.tensor(0.01))
        self.config = {"velocity_scale": 1.0}
        self.inputs = []
        self.outputs = []
        self.invalid = invalid

    def forward(self, batch, proposal=None):
        self.inputs.append(batch)
        displacement = torch.stack((self.shift, self.shift * 0, self.shift * 0))
        positions = batch["positions"] + displacement
        velocities = batch["velocities"] + displacement
        if proposal is not None:
            positions = positions + proposal["positions"] - batch["positions"]
            velocities = velocities + proposal["velocities"] - batch["velocities"]
        if self.invalid == "inverted":
            positions = positions.reshape(-1, 4, 3)[:, [1, 0, 2, 3]].reshape(-1, 3)
        elif self.invalid == "nonfinite":
            positions = positions * float("nan")
        broken = batch["broken"].clone()
        broken[1] = True
        output = dict(positions=positions, velocities=velocities,
                      fracture_logits=self.shift.expand(len(broken)), broken=broken,
                      support_release_logits=self.shift.expand(len(positions)),
                      next_fixed=torch.zeros_like(batch["fixed"]))
        self.outputs.append(output)
        return output

    def fracture_velocity(self, batch, field, time):
        return torch.zeros_like(field) + self.shift * 0


def test_autoregressive_window_passes_predicted_states_and_preserves_motion_gradients(tmp_path):
    case, path = recorded_trajectory(tmp_path / "source", supports=True)
    trajectory = load_dynamics_trajectory(case, path)
    model = RecordedAutoregressiveModel()
    loss, terms = _rollout_loss(model, trajectory, 0, "cpu", training_config(rollout_steps=2))
    assert len(model.inputs) == 2
    assert model.inputs[1]["positions"] is model.outputs[0]["positions"]
    assert model.inputs[1]["velocities"] is model.outputs[0]["velocities"]
    torch.testing.assert_close(model.inputs[1]["broken"], torch.tensor([False, True, False]))
    assert not model.inputs[1]["fixed"].any()
    assert not torch.equal(model.inputs[1]["broken"], torch.from_numpy(trajectory.broken[1].copy()))
    assert not torch.equal(model.inputs[1]["fixed"], torch.from_numpy(trajectory.fixed[1].copy()))
    derivative = torch.autograd.grad(model.outputs[1]["positions"].sum(), model.outputs[0]["positions"], retain_graph=True)[0]
    torch.testing.assert_close(derivative, torch.ones_like(derivative))
    loss.backward()
    assert torch.isfinite(model.shift.grad) and model.shift.grad.abs() > 0
    assert all(np.isfinite(value) for value in terms.values())


def test_autoregressive_inputs_do_not_depend_on_changed_future_supervision(tmp_path):
    case, path = recorded_trajectory(tmp_path / "source", supports=True)
    trajectory = load_dynamics_trajectory(case, path)
    positions, velocities = trajectory.positions.copy(), trajectory.velocities.copy()
    positions[1:] += 100
    velocities[1:] -= 100
    altered = replace(trajectory, positions=positions, velocities=velocities,
                      broken=np.zeros_like(trajectory.broken), fixed=np.zeros_like(trajectory.fixed))
    altered_fixed = altered.fixed.copy()
    altered_fixed[0] = trajectory.fixed[0]
    altered = replace(altered, fixed=altered_fixed)
    baseline, changed = RecordedAutoregressiveModel(), RecordedAutoregressiveModel()
    first_loss, _ = _rollout_loss(baseline, trajectory, 0, "cpu", training_config(rollout_steps=2))
    second_loss, _ = _rollout_loss(changed, altered, 0, "cpu", training_config(rollout_steps=2))
    assert first_loss.item() != second_loss.item()
    for first, second in zip(baseline.inputs, changed.inputs):
        assert not any(key.startswith("target") for key in first)
        for key, value in first.items():
            if isinstance(value, torch.Tensor):
                torch.testing.assert_close(value, second[key])
            else:
                assert value == second[key]


@pytest.mark.parametrize("invalid,match", [("inverted", "inverted or degenerate"), ("nonfinite", "nonfinite")])
def test_invalid_autoregressive_states_fail_explicitly(tmp_path, invalid, match):
    case, path = recorded_trajectory(tmp_path / "source")
    trajectory = load_dynamics_trajectory(case, path)
    model = RecordedAutoregressiveModel(invalid)
    with pytest.raises(FloatingPointError, match=match):
        _rollout_loss(model, trajectory, 0, "cpu", training_config(rollout_steps=2))
    assert len(model.inputs) == 1


def test_autoregressive_optional_corrector_retains_gradient_through_frozen_predictor(tmp_path):
    case, path = recorded_trajectory(tmp_path / "source")
    trajectory = load_dynamics_trajectory(case, path)
    predictor = RecordedAutoregressiveModel()
    predictor.requires_grad_(False)
    corrector = RecordedAutoregressiveModel()
    loss, _ = _rollout_loss(corrector, trajectory, 0, "cpu", training_config(rollout_steps=2), fixed_predictor=predictor)
    assert predictor.inputs[1]["positions"] is corrector.outputs[0]["positions"]
    derivative = torch.autograd.grad(predictor.outputs[1]["positions"].sum(), corrector.outputs[0]["positions"], retain_graph=True)[0]
    torch.testing.assert_close(derivative, torch.ones_like(derivative))
    loss.backward()
    assert predictor.shift.grad is None
    assert torch.isfinite(corrector.shift.grad) and corrector.shift.grad.abs() > 0


def test_cli_rollout_window_matches_training_config():
    from fracture_world.cli import parser

    arguments = parser().parse_args(["dynamics-train", "--data", "data", "--output", "output", "--epochs", "1", "--rollout-steps", "3"])
    config = DynamicsTrainingConfig(**{name: getattr(arguments, name) for name in DynamicsTrainingConfig.__dataclass_fields__})
    assert config.rollout_steps == 3


@pytest.mark.parametrize("steps", [0, -1, True, 1.5])
def test_training_rejects_invalid_autoregressive_window_before_loading_data(tmp_path, steps):
    with pytest.raises(ValueError, match="rollout_steps"):
        train_dynamics(tmp_path / "missing", tmp_path / "unused", training_config(rollout_steps=steps))
