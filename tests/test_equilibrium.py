from unittest.mock import patch

import numpy as np
import pytest
import torch
from torch_geometric.data import Batch

from fracture_world.equilibrium import EquilibriumDynamicsModel
from fracture_world.integration2d import advance, velocity_verlet
from fracture_world.neural import DynamicsModel, build_graph, create_model
from fracture_world.topology import lumped_mass, rectangle, split_edges


def graph_for(mesh=None):
    mesh = rectangle(2, 2) if mesh is None else mesh
    return build_graph(
        mesh, lumped_mass(mesh), np.zeros_like(mesh.rest, dtype=bool),
        material=(100.0, 0.25, 1.0), dtype=torch.float64,
    )


def nonzero_model(method="stress"):
    torch.manual_seed(41)
    model = EquilibriumDynamicsModel(method, hidden_size=8, layers=2).double()
    torch.nn.init.normal_(model.decoder[-1].weight, std=0.1)
    return model


@pytest.mark.parametrize("method", ["direct", "stress"])
def test_anchored_equilibrium_with_nonzero_weights_and_long_stationary_rollout(method):
    graph = graph_for()
    model = nonzero_model(method)
    torch.testing.assert_close(model.instantaneous_force(graph, graph.rest), graph.rest * 0, atol=0, rtol=0)
    position, velocity, cache = graph.rest, graph.rest * 0, None
    with torch.inference_mode():
        for _ in range(100):
            result = advance(model, graph, position, velocity, graph.rest, graph.rest * 0, 0.002, cache)
            position, velocity, cache = result.position, result.velocity, result.cache
    torch.testing.assert_close(position, graph.rest, atol=0, rtol=0)
    torch.testing.assert_close(velocity, graph.rest * 0, atol=0, rtol=0)


@pytest.mark.parametrize("method", ["direct", "stress"])
def test_deformed_force_is_rotation_covariant_and_position_differentiable(method):
    graph = graph_for()
    model = nonzero_model(method)
    position = (graph.rest @ graph.rest.new_tensor([[1.03, 0.0], [0.02, 0.98]])).requires_grad_()
    angle = 0.63
    rotation = graph.rest.new_tensor([[np.cos(angle), -np.sin(angle)], [np.sin(angle), np.cos(angle)]])
    force = model.instantaneous_force(graph, position)
    rotated = model.instantaneous_force(graph, position @ rotation.T + 0.4)
    assert force.norm() > 0
    torch.testing.assert_close(rotated, force @ rotation.T, atol=1e-10, rtol=1e-9)
    force.square().sum().backward()
    assert torch.isfinite(position.grad).all()
    assert any(parameter.grad is not None and parameter.grad.norm() > 0 for parameter in model.parameters())


@pytest.mark.parametrize("method", ["direct", "stress"])
def test_batch_and_individual_forces_and_steps_match(method):
    graphs = [graph_for(), graph_for(rectangle(3, 2))]
    model = nonzero_model(method)
    batch = Batch.from_data_list(graphs)
    deformation = batch.rest.new_tensor([[1.02, 0.03], [0.0, 0.99]])
    combined = model(batch, batch.rest @ deformation.T, batch.rest * 0, batch.rest, batch.rest * 0, 0.001)
    separate = [model(graph, graph.rest @ deformation.T, graph.rest * 0, graph.rest, graph.rest * 0, 0.001)
                for graph in graphs]
    for index in range(3):
        torch.testing.assert_close(combined[index], torch.cat([row[index] for row in separate]), atol=1e-12, rtol=1e-10)


@pytest.mark.parametrize("method", ["direct", "stress"])
def test_disconnected_graph_perturbation_does_not_change_other_component(method):
    graph = graph_for(split_edges(rectangle(2, 2), [[1, 4], [4, 7]]))
    model = nonzero_model(method)
    left = (graph.rest[graph.triangles].mean(1)[:, 0] < 0.5)
    nodes = torch.unique(graph.triangles[left])
    other = torch.unique(graph.triangles[~left])
    assert not set(nodes.tolist()) & set(other.tolist())
    position = graph.rest @ graph.rest.new_tensor([[1.03, 0.0], [0.02, 0.99]])
    force = model.instantaneous_force(graph, position)
    changed = position.clone()
    changed[nodes, 0] += 0.04 * changed[nodes, 1]
    changed_force = model.instantaneous_force(graph, changed)
    torch.testing.assert_close(force[other], changed_force[other], atol=0, rtol=0)
    assert (changed_force[nodes] - force[nodes]).norm() > 0


def test_stress_force_and_torque_cancel_on_each_separate_component():
    graph = graph_for(split_edges(rectangle(2, 2), [[1, 4], [4, 7]]))
    model = nonzero_model()
    position = graph.rest @ graph.rest.new_tensor([[1.03, 0.0], [0.02, 0.99]])
    force = model.instantaneous_force(graph, position)
    left = graph.rest[graph.triangles].mean(1)[:, 0] < 0.5
    for selected in (left, ~left):
        nodes = torch.unique(graph.triangles[selected])
        arm = position[nodes] - position[nodes].mean(0)
        torque = arm[:, 0] * force[nodes, 1] - arm[:, 1] * force[nodes, 0]
        torch.testing.assert_close(force[nodes].sum(0), force.new_zeros(2), atol=1e-12, rtol=0)
        torch.testing.assert_close(torque.sum(), force.new_zeros(()), atol=1e-12, rtol=0)


def test_force_cache_reuses_only_exact_unchanged_state_and_model():
    graph = graph_for()
    model = nonzero_model()
    position, velocity = graph.rest * 1.01, graph.rest * 0
    with torch.inference_mode(), patch.object(model, "instantaneous_force", wraps=model.instantaneous_force) as force:
        first = advance(model, graph, position, velocity, graph.rest, velocity, 0.002)
        assert force.call_count == 2
        second = advance(model, graph, first.position, first.velocity, graph.rest, velocity, 0.002, first.cache)
        assert force.call_count == 3
        uncached = advance(model, graph, first.position, first.velocity, graph.rest, velocity, 0.002)
        torch.testing.assert_close(second.position, uncached.position, atol=0, rtol=0)
        torch.testing.assert_close(second.velocity, uncached.velocity, atol=0, rtol=0)
        count = force.call_count
        first.position.add_(0.01)
        advance(model, graph, first.position, first.velocity, graph.rest, velocity, 0.002, first.cache)
        assert force.call_count == count + 2


@pytest.mark.parametrize("change", ["parameters", "graph", "force", "copy", "new_graph"])
def test_force_cache_invalidates_after_parameter_or_graph_changes(change):
    graph = graph_for()
    model = nonzero_model()
    velocity = graph.rest * 0
    with torch.inference_mode(), patch.object(model, "instantaneous_force", wraps=model.instantaneous_force) as force:
        first = advance(model, graph, graph.rest * 1.01, velocity, graph.rest, velocity, 0.002)
        position = first.position
        if change == "parameters":
            model.decoder[-1].weight.add_(0.1)
        elif change == "graph":
            graph.edge_cut.add_(1)
        elif change == "force":
            first.cache.force.add_(0.1)
        elif change == "copy":
            position = position.clone()
        else:
            graph = graph_for()
        previous = force.call_count
        cached = advance(model, graph, position, first.velocity, graph.rest, velocity, 0.002, first.cache)
        assert force.call_count == previous + 2
        fresh = advance(model, graph, position, first.velocity, graph.rest, velocity, 0.002)
        torch.testing.assert_close(cached.position, fresh.position, atol=0, rtol=0)
        torch.testing.assert_close(cached.velocity, fresh.velocity, atol=0, rtol=0)


def test_velocity_verlet_prescribed_boundary_and_impulse_balance():
    position = torch.tensor([[0.0, 0.0], [1.1, 0.0]], dtype=torch.float64)
    velocity = torch.zeros_like(position)
    mass = torch.tensor([2.0, 3.0], dtype=torch.float64)
    mask = torch.tensor([[True, True], [False, False]])
    boundary_position = position.clone()
    boundary_position[0, 0] = 0.01
    boundary_velocity = velocity.clone()
    boundary_velocity[0, 0] = 0.2

    def force(value):
        spring = 10 * (value[1] - value[0] - value.new_tensor([1.0, 0.0]))
        return torch.stack((spring, -spring))

    result = velocity_verlet(force, mass, position, velocity, 0.01, mask, boundary_position, boundary_velocity)
    torch.testing.assert_close(result.position[mask], boundary_position[mask])
    torch.testing.assert_close(result.velocity[mask], boundary_velocity[mask])
    impulse = mass[:, None] * (result.velocity - velocity) - 0.01 * result.mean_force
    torch.testing.assert_close(impulse[~mask], torch.zeros_like(impulse[~mask]), atol=1e-15, rtol=0)
    torch.testing.assert_close(
        (mass[:, None] * (result.velocity - velocity)).sum(0),
        torch.where(mask, impulse, 0).sum(0), atol=1e-15, rtol=0,
    )


@pytest.mark.parametrize("method", ["direct", "stress"])
def test_legacy_factory_and_step_preserve_original_predictions(method):
    graph = graph_for()
    original = DynamicsModel(method, hidden_size=8, layers=2).double()
    torch.nn.init.normal_(original.decoder[-1].weight, std=0.1)
    loaded = create_model(original.configuration()).double()
    loaded.load_state_dict(original.state_dict())
    args = graph, graph.rest * 1.01, graph.rest * 0, graph.rest, graph.rest * 0, 0.002
    expected = original(*args)
    result = advance(loaded, *args)
    for left, right in zip(expected, (result.position, result.velocity, result.mean_force)):
        torch.testing.assert_close(left, right, atol=0, rtol=0)


def test_invalid_architecture_is_rejected():
    with pytest.raises(ValueError, match="architecture"):
        create_model({"architecture": "unknown"})


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA unavailable")
@pytest.mark.parametrize("method", ["direct", "stress"])
def test_cuda_equilibrium_cached_rollout_and_backward(method):
    mesh = rectangle(2, 2)
    graph = build_graph(
        mesh, lumped_mass(mesh), np.zeros_like(mesh.rest, dtype=bool),
        material=(100.0, 0.25, 1.0), device="cuda",
    )
    model = nonzero_model(method).float().cuda()
    torch.testing.assert_close(model.instantaneous_force(graph, graph.rest), graph.rest * 0, atol=0, rtol=0)
    position = graph.rest @ graph.rest.new_tensor([[1.03, 0.0], [0.02, 0.99]])
    first = advance(model, graph, position, graph.rest * 0, graph.rest, graph.rest * 0, 0.002)
    second = advance(model, graph, first.position, first.velocity, graph.rest, graph.rest * 0, 0.002, first.cache)
    second.velocity.square().sum().backward()
    assert second.position.is_cuda and second.velocity.is_cuda
    assert any(parameter.grad is not None and parameter.grad.norm() > 0 for parameter in model.parameters())
    assert all(torch.isfinite(parameter.grad).all() for parameter in model.parameters() if parameter.grad is not None)
