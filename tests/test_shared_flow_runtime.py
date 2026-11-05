import numpy as np
import pytest
import torch

from fracture_world.dynamics_model import TemporalDynamicsModel
from fracture_world.dynamics_runtime import RolloutConfig, fracture_candidates, rollout_dynamics
from test_dynamics_model import adjacent_face_batch, dynamics_batch, small_model
from test_dynamics_runtime import portable_rollout_inputs


@pytest.mark.parametrize("architecture", ["independent", "coupled"])
def test_reused_flow_context_preserves_sampled_fields_and_checkpoint_layout(architecture, monkeypatch):
    torch.manual_seed(823)
    model = TemporalDynamicsModel(hidden=12, heads=3, graph_layers=2, flow_architecture=architecture).eval()
    batch = adjacent_face_batch()
    initial_keys = tuple(model.state_dict())
    noise = [torch.tensor([-.4, .7]), torch.tensor([.6, -.2])]
    with torch.no_grad():
        expected_prediction = model(batch)
        expected_fields = [model.sample_fracture(batch, steps=2, noise=value) for value in noise]
        calls = []
        original_encode = model.encode

        def encode(values):
            calls.append(values)
            return original_encode(values)

        monkeypatch.setattr(model, "encode", encode)
        prediction, context = model.forward_with_flow_context(batch)
        fields = [model.sample_fracture(batch, steps=2, noise=value, context=context) for value in noise]
    assert len(calls) == 1
    assert tuple(model.state_dict()) == initial_keys
    for name in prediction:
        torch.testing.assert_close(prediction[name], expected_prediction[name], rtol=0, atol=0)
    for actual, expected in zip(fields, expected_fields):
        torch.testing.assert_close(actual, expected, rtol=0, atol=0)
    restored = TemporalDynamicsModel(**model.config)
    restored.load_state_dict(model.state_dict(), strict=True)


@pytest.mark.parametrize("method,samples", [("raw", 0), ("raw", 4), ("physics", 4)])
def test_rollout_encodes_once_per_current_state_and_shares_only_within_the_step(tmp_path, monkeypatch, method, samples):
    checkpoint, initial, _, _ = portable_rollout_inputs(tmp_path, has_corrector=False)
    encoded_states, sampled_contexts = [], []
    original_encode = TemporalDynamicsModel.encode
    original_sample = TemporalDynamicsModel.sample_fracture

    def encode(self, batch, *args, **kwargs):
        encoded_states.append(batch["positions"].clone())
        return original_encode(self, batch, *args, **kwargs)

    def sample(self, batch, steps, noise, *, context=None):
        assert context is not None
        sampled_contexts.append(context)
        return original_sample(self, batch, steps, noise, context=context)

    monkeypatch.setattr(TemporalDynamicsModel, "encode", encode)
    monkeypatch.setattr(TemporalDynamicsModel, "sample_fracture", sample)
    settings = RolloutConfig(fracture_samples=samples, flow_steps=1,
                             fracture_policy="marginal_search" if samples == 0 else "fm_search")
    report = rollout_dynamics(checkpoint, initial, tmp_path / "rollout", 3, method, settings, "cpu")
    assert report["status"] == "completed"
    expected_encodes = 0 if method == "physics" else 3
    assert len(encoded_states) == expected_encodes
    assert len(sampled_contexts) == (3 * samples if method != "physics" else 0)
    if method != "physics":
        assert not torch.equal(encoded_states[0], encoded_states[-1])
    if sampled_contexts:
        for step in range(3):
            contexts = sampled_contexts[step * samples:(step + 1) * samples]
            assert all(context is contexts[0] for context in contexts)
        assert len({id(sampled_contexts[step * samples]) for step in range(3)}) == 3


def test_sampling_without_internal_faces_requires_no_additional_encoder_calls(monkeypatch):
    batch = dynamics_batch()
    for key in ("rest_positions", "positions", "velocities", "masses", "fixed", "forces"):
        batch[key] = batch[key][:4]
    batch["face_pairs"] = torch.empty((0, 3, 2), dtype=torch.long)
    batch["broken"] = torch.empty(0, dtype=torch.bool)
    model = small_model().eval()
    calls = []
    original_encode = model.encode

    def encode(values):
        calls.append(values)
        return original_encode(values)

    monkeypatch.setattr(model, "encode", encode)
    with torch.no_grad():
        proposal = model(batch)
        candidates = fracture_candidates(model, batch, proposal, RolloutConfig(fracture_samples=4, flow_steps=1), 0)
    assert len(calls) == 1
    assert len(candidates) == 1
    np.testing.assert_array_equal(candidates[0], np.empty(0, dtype=bool))
