import pytest
import torch

from fracture_world.dynamics_model import TemporalDynamicsModel, physics_motion_prediction
from fracture_world.dynamics_runtime import _candidate_proposal
from test_dynamics_model import dynamics_batch


def physical_batch(dtype=torch.float64):
    batch = dynamics_batch()
    for name in ("rest_positions", "positions", "velocities", "masses", "forces", "material", "dt"):
        batch[name] = batch[name].to(dtype)
    batch["velocities"] = torch.arange(24, dtype=dtype).reshape(8, 3) / 7
    batch["masses"] = torch.arange(1, 9, dtype=dtype)
    batch["forces"] = torch.arange(24, dtype=dtype).reshape(8, 3) / 11 - 1.
    return batch


@pytest.mark.parametrize("motion_prior", ["backward_euler", "constant_acceleration"])
@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
def test_physical_motion_exactly_matches_checkpoint_backbone_with_zero_learned_motion(motion_prior, dtype):
    batch = physical_batch(dtype)
    model = TemporalDynamicsModel(hidden=12, heads=3, graph_layers=2, transformer_layers=1,
                                   motion_prior=motion_prior).to(dtype=dtype)
    checkpoint = {name: tensor.clone() for name, tensor in model.state_dict().items()}
    zero_corner_context = torch.zeros((8, 12), dtype=dtype)
    zero_face_context = torch.zeros((1, 12), dtype=dtype)
    reference = model._prediction(batch, zero_corner_context, zero_face_context)
    physical = physics_motion_prediction(batch, motion_prior)
    for name, actual in physical.items():
        torch.testing.assert_close(actual, reference[name], rtol=0., atol=0.)
    assert set(model.state_dict()) == set(checkpoint)
    for name, tensor in model.state_dict().items():
        torch.testing.assert_close(tensor, checkpoint[name], rtol=0., atol=0.)


@pytest.mark.parametrize("motion_prior", ["backward_euler", "constant_acceleration"])
def test_current_support_clamp_preserves_unconstrained_motion_for_shared_candidate_support_decisions(motion_prior):
    batch = physical_batch()
    physical = physics_motion_prediction(batch, motion_prior)
    held = batch["fixed"]
    torch.testing.assert_close(physical["positions"][held], batch["positions"][held], rtol=0., atol=0.)
    torch.testing.assert_close(physical["velocities"][held], torch.zeros_like(batch["velocities"][held]), rtol=0., atol=0.)
    expected_velocity_increment = batch["forces"] * batch["dt"] / batch["masses"][:, None]
    torch.testing.assert_close(physical["unconstrained_velocity_increment"], expected_velocity_increment, rtol=0., atol=0.)
    assert not {"fracture_probability", "fracture_logits", "support_release_probability", "support_release_logits", "next_fixed"}.intersection(physical)
    candidate = _candidate_proposal(physical, batch["positions"].numpy(), batch["velocities"].numpy(),
                                     torch.zeros_like(held).numpy())
    torch.testing.assert_close(candidate["velocities"], batch["velocities"] + expected_velocity_increment, rtol=0., atol=0.)
    torch.testing.assert_close(candidate["positions"], batch["positions"] + physical["unconstrained_position_increment"],
                               rtol=0., atol=0.)


def test_physical_motion_requires_no_graph_or_neural_network_evaluation(monkeypatch):
    def unexpected(*args, **kwargs):
        raise AssertionError("physical motion must not execute a graph encoder or neural head")

    monkeypatch.setattr(TemporalDynamicsModel, "forward", unexpected)
    monkeypatch.setattr(TemporalDynamicsModel, "encode", unexpected)
    monkeypatch.setattr("fracture_world.dynamics_model._build_dynamics_graph", unexpected)
    batch = physical_batch()
    backward = physics_motion_prediction(batch, "backward_euler")
    acceleration = physics_motion_prediction(batch, "constant_acceleration")
    active = ~batch["fixed"]
    expected_difference = 0.5 * batch["dt"].square() * batch["forces"] / batch["masses"][:, None]
    torch.testing.assert_close(backward["position_increment"][active] - acceleration["position_increment"][active],
                               expected_difference[active], rtol=1e-12, atol=1e-17)
    torch.testing.assert_close(backward["velocities"], acceleration["velocities"], rtol=0., atol=0.)


def test_physical_motion_rejects_unknown_prior_and_invalid_physical_state():
    batch = physical_batch()
    with pytest.raises(ValueError, match="physical motion prior"):
        physics_motion_prediction(batch, "trapezoidal")
    batch["masses"][0] = 0.
    with pytest.raises(ValueError, match="masses"):
        physics_motion_prediction(batch, "backward_euler")
