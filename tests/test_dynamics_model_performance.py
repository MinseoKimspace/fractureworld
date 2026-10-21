from copy import deepcopy

import numpy as np
import pytest
import torch
from scipy.spatial import cKDTree

from fracture_world import dynamics_model
from fracture_world.dynamics_model import DynamicsGraphCache, TemporalDynamicsModel, _contact_pairs, build_dynamics_graph
from test_dynamics_model import adjacent_face_batch, dynamics_batch, small_model


def reference_contact_pairs(batch, radius):
    positions = batch["positions"].detach().cpu().numpy()
    parents = np.arange(len(positions))

    def root(index):
        while parents[index] != index:
            parents[index] = parents[parents[index]]
            index = parents[index]
        return index

    faces = batch["face_pairs"][~batch["broken"]].cpu().numpy()
    for left, right in faces.reshape(-1, 2):
        parents[root(int(left))] = root(int(right))
    groups = np.array([root(index) for index in range(len(positions))])
    neighbors = {tuple(sorted((int(face[0, 0] // 4), int(face[0, 1] // 4)))) for face in faces}
    pairs = cKDTree(positions).query_pairs(radius, output_type="ndarray")
    selected = [(left, right) for left, right in pairs if left // 4 != right // 4 and groups[left] != groups[right]
                and tuple(sorted((int(left // 4), int(right // 4)))) not in neighbors]
    return torch.tensor(selected, dtype=torch.long).reshape(-1, 2)


@pytest.mark.parametrize("radius", [0.001, 0.2, 4.0])
def test_vectorized_contact_candidates_match_original_connectivity_and_neighbor_exclusion(radius):
    batch = adjacent_face_batch()
    cache = DynamicsGraphCache()
    generator = torch.Generator().manual_seed(713)
    for mask in ([False, False], [True, False], [False, True], [True, True]):
        batch["broken"][:] = torch.tensor(mask)
        batch["positions"][:] = batch["rest_positions"] + 0.04 * torch.randn(batch["positions"].shape, generator=generator)
        torch.testing.assert_close(_contact_pairs(batch, radius, cache), reference_contact_pairs(batch, radius))


def test_shared_prediction_flow_context_reuses_one_encode_and_preserves_gradients(monkeypatch):
    torch.manual_seed(719)
    batch = adjacent_face_batch()
    batch["positions"] = (batch["positions"] + 0.01).requires_grad_()
    original = small_model()
    shared = deepcopy(original)
    field = torch.tensor([-0.4, 0.7])
    original_prediction = original(batch)
    original_velocity = original.fracture_velocity(batch, field, 0.3)
    original_loss = original_prediction["fracture_logits"].square().sum() + original_velocity.square().sum()
    original_loss.backward()
    original_position_gradient = batch["positions"].grad.clone()
    batch["positions"].grad = None
    calls = []
    encode = shared.encode

    def counted(values):
        calls.append(values)
        return encode(values)

    monkeypatch.setattr(shared, "encode", counted)
    prediction, context = shared.forward_with_flow_context(batch)
    velocity = shared.fracture_velocity(batch, field, 0.3, context=context)
    loss = prediction["fracture_logits"].square().sum() + velocity.square().sum()
    loss.backward()
    assert len(calls) == 1
    torch.testing.assert_close(velocity, original_velocity)
    for name, value in prediction.items():
        torch.testing.assert_close(value, original_prediction[name])
    torch.testing.assert_close(batch["positions"].grad, original_position_gradient)
    for (name, parameter), (other_name, other_parameter) in zip(original.named_parameters(), shared.named_parameters()):
        assert name == other_name
        if parameter.grad is None:
            assert other_parameter.grad is None
        else:
            torch.testing.assert_close(parameter.grad, other_parameter.grad)


def test_static_cache_reuses_geometry_and_invalidates_in_place_rest_topology_and_masses():
    model = small_model()
    batch = adjacent_face_batch()
    model(batch)
    first_geometry = model._graph_cache.entries["reference_geometry"][2]
    first_center = model._graph_cache.entries["rest_center"][2]
    model(batch)
    assert model._graph_cache.entries["reference_geometry"][2] is first_geometry
    assert model._graph_cache.entries["rest_center"][2] is first_center
    batch["rest_positions"][:, 0] *= 1.1
    batch["masses"][0] *= 2
    model(batch)
    assert model._graph_cache.entries["reference_geometry"][2] is not first_geometry
    assert model._graph_cache.entries["rest_center"][2] is not first_center
    _, context = model.forward_with_flow_context(batch)
    old_adjacency = context.edge_index
    batch["face_pairs"][1] = batch["face_pairs"][0]
    _, context = model.forward_with_flow_context(batch)
    assert context.edge_index is not old_adjacency
    torch.testing.assert_close(context.edge_index, dynamics_model.build_fracture_adjacency(batch["face_pairs"]))


def test_cached_graph_features_match_uncached_graph_after_dynamic_and_static_changes(monkeypatch):
    model = small_model()
    batch = adjacent_face_batch()
    batch["floor_height"] = -0.1
    captured = []
    build = dynamics_model._build_dynamics_graph

    def capture(*args, **kwargs):
        result = build(*args, **kwargs)
        captured.append(result)
        return result

    monkeypatch.setattr(dynamics_model, "_build_dynamics_graph", capture)
    for frame in range(3):
        batch["positions"][:, 0] += 0.01
        batch["velocities"][:, 1] -= 0.02
        batch["broken"][frame % 2] = frame > 0
        if frame == 2:
            batch["rest_positions"][:, 2] *= 0.9
        model(batch)
        cached = captured[-1]
        uncached = build_dynamics_graph(batch, contact_radius=model.config["contact_radius"],
                                        material_scale=model.config["material_scale"])
        for node in cached.node_types:
            torch.testing.assert_close(cached[node].x, uncached[node].x)
        for relation in cached.edge_types:
            torch.testing.assert_close(cached[relation].edge_index, uncached[relation].edge_index)


def test_differentiable_rest_geometry_is_not_reused_across_backward_calls():
    model = small_model()
    batch = dynamics_batch()
    batch["rest_positions"] = batch["rest_positions"].requires_grad_()
    for _ in range(2):
        model.zero_grad(set_to_none=True)
        batch["rest_positions"].grad = None
        model(batch)["fracture_logits"].square().sum().backward()
        assert torch.isfinite(batch["rest_positions"].grad).all()
    assert "reference_geometry" not in model._graph_cache.entries


def test_static_cache_does_not_change_checkpoint_layout_and_supports_inference_tensors():
    model = small_model()
    original_keys = tuple(model.state_dict())
    with torch.inference_mode():
        batch = dynamics_batch()
        prediction, context = model.forward_with_flow_context(batch)
        assert torch.isfinite(prediction["positions"]).all()
        assert torch.isfinite(model.fracture_velocity(batch, torch.zeros(1), 0.5, context=context)).all()
    assert tuple(model.state_dict()) == original_keys
    restored = TemporalDynamicsModel(**model.config)
    restored.load_state_dict(model.state_dict(), strict=True)


def test_inference_on_normal_inputs_does_not_cache_inference_tensors_for_later_training():
    model = small_model()
    batch = adjacent_face_batch()
    with torch.inference_mode():
        model.forward_with_flow_context(batch)
    assert not model._graph_cache.entries
    batch["positions"] = batch["positions"].requires_grad_()
    prediction, context = model.forward_with_flow_context(batch)
    velocity = model.fracture_velocity(batch, torch.zeros(2), 0.3, context=context)
    (prediction["fracture_logits"].square().sum() + velocity.square().sum()).backward()
    assert torch.isfinite(batch["positions"].grad).all()


@pytest.mark.parametrize("name", ["positions", "rest_positions", "masses", "forces", "material", "velocities"])
def test_cached_model_still_rejects_in_place_nonfinite_inputs(name):
    model = small_model()
    batch = dynamics_batch()
    model(batch)
    batch[name].reshape(-1)[0] = float("nan")
    with pytest.raises(ValueError, match="finite"):
        model(batch)
