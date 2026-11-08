import numpy as np
import pytest
import torch
from torch_geometric.data import Batch

from fracture_world.neural import DynamicsModel, build_graph
from fracture_world.topology import lumped_mass, rectangle, split_edges


def graph_for(mesh):
    return build_graph(mesh, lumped_mass(mesh), np.zeros_like(mesh.rest, dtype=bool), dtype=torch.float64)


@pytest.mark.parametrize("method", ["direct", "stress"])
def test_zero_model_reproduces_free_flight_without_engine(method):
    graph = graph_for(rectangle(2, 2))
    model = DynamicsModel(method, hidden_size=8, layers=1).double()
    for parameter in model.parameters():
        parameter.data.zero_()
    position = graph.rest + 0.17
    velocity = torch.full_like(position, 0.3)
    next_position, next_velocity, force = model(graph, position, velocity, position, velocity, 0.02)
    torch.testing.assert_close(next_position, position + 0.02 * velocity)
    torch.testing.assert_close(next_velocity, velocity)
    torch.testing.assert_close(force, torch.zeros_like(force))


@pytest.mark.parametrize("method", ["direct", "stress"])
def test_boundary_conditions_and_autograd(method):
    mesh = rectangle(2, 2)
    fixed = np.repeat(np.isclose(mesh.rest[:, 0:1], 0), 2, axis=1)
    graph = build_graph(mesh, lumped_mass(mesh), fixed, dtype=torch.float64)
    model = DynamicsModel(method, hidden_size=8, layers=2).double()
    target_position = graph.rest + 0.01
    target_velocity = torch.full_like(graph.rest, 0.1)
    position, velocity, force = model(
        graph, graph.rest, torch.zeros_like(graph.rest), target_position, target_velocity, 0.01
    )
    torch.testing.assert_close(position[graph.boundary_mask], target_position[graph.boundary_mask])
    torch.testing.assert_close(velocity[graph.boundary_mask], target_velocity[graph.boundary_mask])
    (position.square().sum() + velocity.square().sum()).backward()
    assert any(parameter.grad is not None and parameter.grad.abs().sum() > 0 for parameter in model.parameters())
    assert torch.isfinite(force).all()


def test_stress_assembly_preserves_net_internal_force():
    graph = graph_for(rectangle(3, 2))
    model = DynamicsModel("stress", hidden_size=8, layers=1).double()
    torch.nn.init.normal_(model.decoder[-1].weight, std=0.1)
    position = graph.rest + 0.03 * torch.randn_like(graph.rest)
    _, _, force = model(graph, position, torch.randn_like(position), position, position * 0, 0.01)
    assert force.abs().sum() > 0
    torch.testing.assert_close(force.sum(dim=0), torch.zeros(2, dtype=force.dtype), atol=1e-12, rtol=0)


@pytest.mark.parametrize("method", ["direct", "stress"])
def test_batch_has_identical_predictions_to_individual_graphs(method):
    torch.manual_seed(3)
    graphs = [graph_for(rectangle(2, 2)), graph_for(rectangle(3, 2))]
    batch = Batch.from_data_list(graphs)
    model = DynamicsModel(method, hidden_size=8, layers=2).double()
    torch.nn.init.normal_(model.decoder[-1].weight, std=0.1)
    combined = model(batch, batch.rest, torch.zeros_like(batch.rest), batch.rest, batch.rest * 0, 0.01)
    separate = [model(graph, graph.rest, graph.rest * 0, graph.rest, graph.rest * 0, 0.01) for graph in graphs]
    for index in range(3):
        torch.testing.assert_close(combined[index], torch.cat([result[index] for result in separate]))


def test_cut_seam_uses_separate_physical_dofs_and_boundary_descriptors():
    mesh = rectangle(2, 2)
    mesh = split_edges(mesh, [[1, 4], [4, 7]])
    graph = graph_for(mesh)
    assert graph.num_nodes > 9
    assert graph.edge_cut.any()
    source, target = graph.edge_index
    assert not torch.any(graph.original_vertex[source] == graph.original_vertex[target])
    assert graph.rest.shape == (graph.num_nodes, 2)
    assert graph.triangles.max() < graph.num_nodes


def test_invalid_mass_is_rejected():
    mesh = rectangle(1, 1)
    with pytest.raises(ValueError, match="Mass"):
        build_graph(mesh, np.zeros(len(mesh.rest)), np.zeros_like(mesh.rest, dtype=bool))


@pytest.mark.parametrize("method", ["direct", "stress"])
def test_default_initialization_has_zero_internal_force(method):
    graph = graph_for(rectangle(2, 2))
    model = DynamicsModel(method, hidden_size=8, layers=1).double()
    _, _, force = model(graph, graph.rest, graph.rest * 0, graph.rest, graph.rest * 0, 0.01)
    torch.testing.assert_close(force, torch.zeros_like(force), atol=0, rtol=0)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA unavailable")
@pytest.mark.parametrize("method", ["direct", "stress"])
def test_cuda_backward_uses_external_graph_layers(method):
    graph = graph_for(rectangle(2, 2)).to("cuda")
    model = DynamicsModel(method, hidden_size=8, layers=1).double().cuda()
    torch.nn.init.normal_(model.decoder[-1].weight, std=0.1)
    position, velocity, force = model(
        graph, graph.rest, graph.rest * 0, graph.rest, graph.rest * 0, 0.01
    )
    (position.square().mean() + velocity.square().mean()).backward()
    assert force.is_cuda
    assert all(torch.isfinite(parameter.grad).all() for parameter in model.parameters() if parameter.grad is not None)
