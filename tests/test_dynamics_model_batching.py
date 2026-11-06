from copy import deepcopy
import gc
import weakref

import pytest
import torch

from fracture_world import dynamics_model
from fracture_world.dynamics_model import DynamicsGraphCache, NeuralDynamicsCorrector, _contact_pairs
from test_dynamics_model import adjacent_face_batch, dynamics_batch, small_model


def scenes(device="cpu"):
    first = dynamics_batch(broken=True)
    first["velocities"][:, 0] = 0.2
    first["masses"][0] = 1.3
    first["dt"] = 0.02
    first["floor_height"] = -0.25
    second = adjacent_face_batch()
    second["broken"][1] = True
    second["positions"][:, 0] += 0.03
    second["velocities"][:, 1] = -0.1
    third = dynamics_batch()
    for name in ("rest_positions", "positions", "velocities", "forces", "masses", "fixed"):
        third[name] = third[name][:4].clone()
    third["face_pairs"] = torch.empty((0, 3, 2), dtype=torch.long)
    third["broken"] = torch.empty(0, dtype=torch.bool)
    third["fixed"][:] = False
    return [{name: value.to(device) if isinstance(value, torch.Tensor) else value for name, value in batch.items()}
            for batch in (first, second, third)]


def trainable_heads(model):
    with torch.no_grad():
        for head in (model.motion_head, model.fracture_head, model.support_head):
            head.weight.normal_(std=0.03)
            head.bias.normal_(std=0.03)
    return model


def objective(outputs):
    return sum(output["positions"].square().mean() + output["velocities"].square().mean()
               + output["fracture_logits"].square().sum() + output["support_release_logits"].square().mean()
               for output in outputs)


def assert_outputs(actual, expected):
    for batched, independent in zip(actual, expected):
        assert batched.keys() == independent.keys()
        for name in batched:
            torch.testing.assert_close(batched[name], independent[name], atol=2e-6, rtol=2e-5)


@pytest.mark.parametrize("training", [True, False])
@pytest.mark.parametrize("corrector", [True, False])
def test_batched_predictions_match_independent_mixed_scenes(training, corrector):
    torch.manual_seed(317)
    batches = scenes()
    model = trainable_heads(small_model(NeuralDynamicsCorrector) if corrector else small_model())
    model.train(training)
    keys = tuple(model.state_dict())
    with torch.set_grad_enabled(training):
        if corrector:
            predictor = small_model().eval()
            with torch.no_grad():
                proposals = [predictor(batch) for batch in batches]
            expected = [model(batch, proposal) for batch, proposal in zip(batches, proposals)]
            actual = model.forward_batch(batches, proposals)
        else:
            expected = [model(batch) for batch in batches]
            actual = model.forward_batch(batches)
    assert_outputs(actual, expected)
    assert tuple(model.state_dict()) == keys


@pytest.mark.parametrize("corrector", [True, False])
@pytest.mark.parametrize("device", ["cpu", pytest.param("cuda", marks=pytest.mark.skipif(
    not torch.cuda.is_available(), reason="CUDA is unavailable"))])
def test_batched_input_and_parameter_gradients_match_independent_scenes(corrector, device):
    torch.manual_seed(331)
    batches = scenes(device)
    for batch in batches:
        batch["positions"].requires_grad_()
        batch["velocities"].requires_grad_()
    other_batches = deepcopy(batches)
    independent = trainable_heads(small_model(NeuralDynamicsCorrector) if corrector else small_model()).to(device)
    batched = deepcopy(independent)
    if corrector:
        predictor = small_model().to(device).eval().requires_grad_(False)
        other_predictor = deepcopy(predictor)
        expected = [independent(batch, predictor(batch)) for batch in batches]
        proposals = other_predictor.forward_batch(other_batches)
        actual = batched.forward_batch(other_batches, proposals)
    else:
        expected = [independent(batch) for batch in batches]
        actual = batched.forward_batch(other_batches)
    objective(expected).backward()
    objective(actual).backward()
    assert_outputs(actual, expected)
    for batch, other in zip(batches, other_batches):
        for name in ("positions", "velocities"):
            torch.testing.assert_close(other[name].grad, batch[name].grad, atol=2e-5, rtol=2e-4)
    for parameter, reference in zip(batched.parameters(), independent.parameters()):
        if reference.grad is None:
            assert parameter.grad is None
        else:
            torch.testing.assert_close(parameter.grad, reference.grad, atol=2e-5, rtol=2e-4)


def test_batched_scene_predictions_and_gradients_are_isolated():
    torch.manual_seed(337)
    batches = scenes()
    for batch in batches:
        batch["positions"].requires_grad_()
    model = trainable_heads(small_model())
    original = model.forward_batch(batches)
    gradients = torch.autograd.grad(objective(original[:1]), [batch["positions"] for batch in batches])
    assert gradients[0].abs().sum() > 0
    assert torch.count_nonzero(gradients[1]) == 0
    assert torch.count_nonzero(gradients[2]) == 0
    changed = [{**batch} for batch in batches]
    changed[1]["positions"] = changed[1]["positions"] + 2
    changed[1]["velocities"] = changed[1]["velocities"] + 3
    actual = model.forward_batch(changed)
    assert_outputs(actual[:1], original[:1])
    assert_outputs(actual[2:], original[2:])


def test_multiple_scenes_use_one_graph_and_transformer_forward():
    model = small_model()
    calls = []
    hooks = [layer.register_forward_hook(lambda *_: calls.append("graph")) for layer in model.graph_layers]
    hooks.append(model.transformer.register_forward_hook(lambda *_: calls.append("transformer")))
    try:
        model.forward_batch(scenes())
    finally:
        for hook in hooks:
            hook.remove()
    assert calls.count("graph") == len(model.graph_layers)
    assert calls.count("transformer") == 1


@pytest.mark.parametrize("device", ["cpu", pytest.param("cuda", marks=pytest.mark.skipif(
    not torch.cuda.is_available(), reason="CUDA is unavailable"))])
def test_batched_two_step_corrector_preserves_gradients_through_frozen_predictor(device):
    torch.manual_seed(347)
    predictor = small_model().to(device).eval().requires_grad_(False)
    corrector = trainable_heads(small_model(NeuralDynamicsCorrector)).to(device)
    batches = scenes(device)
    first = corrector.forward_batch(batches, predictor.forward_batch(batches))
    advanced = [{**batch, "positions": output["positions"], "velocities": output["velocities"],
                 "broken": output["broken"], "fixed": output["next_fixed"]}
                for batch, output in zip(batches, first)]
    proposals = predictor.forward_batch(advanced)
    gradients = torch.autograd.grad(sum(proposal["positions"].sum() for proposal in proposals),
                                    [output["positions"] for output in first], retain_graph=True)
    assert all(torch.isfinite(value).all() and value.abs().sum() > 0 for value in gradients)
    objective(corrector.forward_batch(advanced, proposals)).backward()
    assert all(parameter.grad is None for parameter in predictor.parameters())
    assert corrector.motion_head.weight.grad.abs().sum() > 0


def test_shared_predictor_corrector_batch_cache_reuses_contacts_for_each_scene(monkeypatch):
    torch.manual_seed(349)
    batches = scenes()
    for batch in batches:
        batch["positions"].requires_grad_()
    calls = []
    tree = dynamics_model.cKDTree

    def counted(positions):
        calls.append(positions.copy())
        return tree(positions)

    monkeypatch.setattr(dynamics_model, "cKDTree", counted)
    predictor = small_model().eval().requires_grad_(False)
    corrector = small_model(NeuralDynamicsCorrector)
    corrector._graph_cache = predictor._graph_cache
    proposals = predictor.forward_batch(batches)
    outputs = corrector.forward_batch(batches, proposals)
    assert len(calls) == len(batches)
    objective(outputs).backward()
    assert all(torch.isfinite(batch["positions"].grad).all() for batch in batches)


def test_contact_cache_invalidates_positions_topology_broken_and_radius(monkeypatch):
    batch = adjacent_face_batch()
    batch["positions"].requires_grad_()
    cache = DynamicsGraphCache()
    calls = []
    tree = dynamics_model.cKDTree

    def counted(positions):
        calls.append(positions.copy())
        return tree(positions)

    monkeypatch.setattr(dynamics_model, "cKDTree", counted)
    first = _contact_pairs(batch, 0.1, cache)
    assert _contact_pairs(batch, 0.1, cache) is first
    assert len(calls) == 1
    with torch.no_grad():
        batch["positions"][0, 0] += 0.2
    _contact_pairs(batch, 0.1, cache)
    batch["broken"][0] = True
    _contact_pairs(batch, 0.1, cache)
    batch["face_pairs"][1] = batch["face_pairs"][0]
    _contact_pairs(batch, 0.1, cache)
    batch["positions"] = batch["positions"].clone()
    _contact_pairs(batch, 0.1, cache)
    _contact_pairs(batch, 0.2, cache)
    assert len(calls) == 6
    torch.testing.assert_close(_contact_pairs(batch, 0.2, cache), _contact_pairs(batch, 0.2))


def test_batched_models_accept_empty_lists_and_reject_mismatched_proposals():
    assert small_model().forward_batch([]) == []
    corrector = small_model(NeuralDynamicsCorrector)
    assert corrector.forward_batch([], []) == []
    with pytest.raises(ValueError, match="one proposal"):
        corrector.forward_batch(scenes(), [])


def test_contact_cache_does_not_retain_autoregressive_input_graphs():
    batch = dynamics_batch()
    batch["positions"] = batch["positions"].requires_grad_() + 0.01
    reference = weakref.ref(batch["positions"])
    cache = DynamicsGraphCache()
    _contact_pairs(batch, 0.1, cache)
    del batch
    gc.collect()
    assert reference() is None


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is unavailable")
def test_cuda_eval_batch_predictions_match_independent_predictor_and_corrector():
    torch.manual_seed(353)
    batches = scenes("cuda")
    predictor = trainable_heads(small_model()).cuda().eval()
    corrector = trainable_heads(small_model(NeuralDynamicsCorrector)).cuda().eval()
    with torch.no_grad():
        expected_proposals = [predictor(batch) for batch in batches]
        actual_proposals = predictor.forward_batch(batches)
        expected = [corrector(batch, proposal) for batch, proposal in zip(batches, expected_proposals)]
        actual = corrector.forward_batch(batches, actual_proposals)
    assert_outputs(actual_proposals, expected_proposals)
    assert_outputs(actual, expected)
