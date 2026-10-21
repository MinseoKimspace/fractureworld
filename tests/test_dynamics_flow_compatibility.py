import math

import pytest
import torch

from fracture_world.dynamics_learning import FORMAT, load_dynamics_model
from fracture_world.dynamics_model import NeuralDynamicsCorrector, TemporalDynamicsModel
from test_dynamics_model import dynamics_batch


def checkpoint_file(tmp_path, architecture, *, legacy=False, probability_policy=None):
    model = TemporalDynamicsModel(hidden=12, heads=3, graph_layers=2, flow_architecture=architecture,
                                  learn_support_release=True)
    corrector = NeuralDynamicsCorrector(**model.config)
    with torch.no_grad():
        model.fracture_head.weight.zero_()
        model.fracture_head.bias.fill_(math.log(4))
        model.support_head.bias.fill_(math.log(9))
    config = dict(model.config)
    if legacy:
        config.pop("flow_architecture")
        config.pop("flow_layers")
    document = dict(format=FORMAT, model_config=config, model=model.state_dict(), corrector=corrector.state_dict())
    if probability_policy is not None:
        document.update(probability_policy=probability_policy,
                        class_weights=dict(fracture_positive=4., support_release_positive=9.))
    path = tmp_path / "checkpoint.pt"
    torch.save(document, path)
    return path, model.eval(), corrector.eval()


def test_legacy_independent_weights_reload_without_silently_changing_flow(tmp_path):
    path, original, _ = checkpoint_file(tmp_path, "independent", legacy=True)
    loaded, _, checkpoint, _ = load_dynamics_model(path, "cpu")
    batch = dynamics_batch()
    field = torch.tensor([0.2])
    assert loaded.config["flow_architecture"] == "independent"
    assert loaded.fracture_logit_offset == 0
    torch.testing.assert_close(loaded.fracture_velocity(batch, field, 0.3), original.fracture_velocity(batch, field, 0.3))
    torch.testing.assert_close(loaded(batch)["fracture_probability"], original(batch)["fracture_probability"])


def test_coupled_checkpoint_round_trip_is_strict_and_preserves_joint_parameters(tmp_path):
    path, original, _ = checkpoint_file(tmp_path, "coupled")
    loaded, _, _, _ = load_dynamics_model(path, "cpu")
    batch = dynamics_batch()
    noise = torch.tensor([-0.2])
    assert loaded.config["flow_architecture"] == "coupled"
    torch.testing.assert_close(loaded.sample_fracture(batch, steps=2, noise=noise), original.sample_fracture(batch, steps=2, noise=noise))
    document = torch.load(path, weights_only=True)
    document["model"].pop(next(key for key in document["model"] if key.startswith("flow_encoder")))
    torch.save(document, path)
    with pytest.raises(RuntimeError, match="Missing key"):
        load_dynamics_model(path, "cpu")


def test_checkpoint_logit_correction_applies_once_to_predictor_and_residual_corrector(tmp_path):
    path, _, _ = checkpoint_file(tmp_path, "coupled", probability_policy="subtract_log_training_positive_weight")
    model, corrector, _, _ = load_dynamics_model(path, "cpu")
    batch = dynamics_batch()
    proposal = model(batch)
    corrected = corrector(batch, proposal)
    for output in (proposal, corrected):
        torch.testing.assert_close(output["fracture_probability"], torch.tensor([0.5]))
        torch.testing.assert_close(output["support_release_probability"][batch["fixed"]], torch.tensor([0.5]))
        torch.testing.assert_close(output["fracture_logits"], torch.tensor([math.log(4)]))
