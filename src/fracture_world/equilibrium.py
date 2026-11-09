import torch
from torch import nn
from torch_geometric.nn import GINEConv

from .integration2d import StepResult, cached_force, make_cache, tracked_tensor, velocity_verlet
from .neural import multilayer


def deformation_gradient(graph, position):
    displacement = position - graph.rest
    identity = torch.eye(2, dtype=position.dtype, device=position.device)
    return identity + torch.einsum("eai,eaj->eij", displacement[graph.triangles], graph.shape_gradient)


def green_strain(deformation):
    identity = torch.eye(2, dtype=deformation.dtype, device=deformation.device)
    return 0.5 * (deformation.transpose(-1, -2) @ deformation - identity)


def symmetric_components(tensor):
    return torch.stack((tensor[:, 0, 0], tensor[:, 1, 1], tensor[:, 0, 1]), dim=-1)


def symmetric_tensor(components):
    first, second, shear = components.unbind(-1)
    return torch.stack((first, shear, shear, second), dim=-1).reshape(-1, 2, 2)


def nodal_average(graph, values):
    cells = graph.triangles.flatten()
    weights = graph.area.expand(-1, 3).flatten()
    weighted = (values[:, None] * graph.area[:, :, None]).expand(-1, 3, -1).reshape(-1, values.shape[-1])
    total = values.new_zeros((len(graph.rest), values.shape[-1])).index_add(0, cells, weighted)
    denominator = values.new_zeros(len(graph.rest)).index_add(0, cells, weights)
    return total / denominator[:, None]


def assemble_force(graph, first_piola):
    corner = -torch.einsum("eij,eaj->eai", first_piola, graph.shape_gradient)
    corner = corner * (graph.area * graph.thickness)[:, :, None]
    return first_piola.new_zeros(graph.rest.shape).index_add(
        0, graph.triangles.flatten(), corner.reshape(-1, 2),
    )


def centered(module, baseline, change):
    reference = module(baseline)
    return reference, module(baseline + change) - reference


class EquilibriumDynamicsModel(nn.Module):
    architecture = "equilibrium"
    integrator = "velocity_verlet"

    def __init__(self, method="stress", hidden_size=64, layers=3):
        super().__init__()
        if method not in {"direct", "stress"}:
            raise ValueError("Method must be direct or stress")
        if hidden_size < 4 or layers < 1:
            raise ValueError("Hidden size and message-passing depth must be positive")
        self.method = method
        self.hidden_size = int(hidden_size)
        self.layers = int(layers)
        self.node_encoder = multilayer(5, hidden_size, hidden_size)
        self.edge_encoder = multilayer(8, hidden_size, hidden_size)
        self.processor = nn.ModuleList(
            GINEConv(multilayer(hidden_size, hidden_size, hidden_size), train_eps=True)
            for _ in range(layers)
        )
        self.normalizers = nn.ModuleList(nn.LayerNorm(hidden_size) for _ in range(layers))
        self.decoder = nn.Sequential(
            nn.Linear(hidden_size + (3 if method == "stress" else 0), hidden_size),
            nn.SiLU(),
            nn.Linear(hidden_size, 3 if method == "stress" else 2, bias=False),
        )
        nn.init.zeros_(self.decoder[-1].weight)

    def configuration(self):
        return {
            "architecture": self.architecture, "method": self.method,
            "hidden_size": self.hidden_size, "layers": self.layers,
        }

    def encode(self, graph, strain):
        node_strain = nodal_average(graph, strain)
        thickness = nodal_average(graph, graph.thickness)
        normalized_mass = graph.mass[:, None] / (
            graph.material[:, 2:3] * graph.length_scale.square() * thickness
        )
        poisson = graph.material[:, 1:2]
        node_features = torch.cat((torch.zeros_like(node_strain), normalized_mass, poisson), dim=-1)
        node_change = torch.cat((node_strain, torch.zeros_like(normalized_mass), torch.zeros_like(poisson)), dim=-1)
        source, target = graph.edge_index
        relative_rest = (graph.rest[source] - graph.rest[target]) / graph.length_scale[source]
        static_edges = torch.cat((
            relative_rest, relative_rest.norm(dim=-1, keepdim=True),
            graph.edge_boundary, graph.edge_cut,
        ), dim=-1)
        strain_change = node_strain[source] - node_strain[target]
        edge_features = torch.cat((static_edges, torch.zeros_like(strain_change)), dim=-1)
        edge_change = torch.cat((torch.zeros_like(static_edges), strain_change), dim=-1)
        latent, latent_change = centered(self.node_encoder, node_features, node_change)
        edges, edges_change = centered(self.edge_encoder, edge_features, edge_change)
        for processor, normalizer in zip(self.processor, self.normalizers):
            message_input = latent[source] + edges
            messages = message_input.relu()
            messages_change = (message_input + latent_change[source] + edges_change).relu() - messages
            aggregate = processor.aggr_module(messages, index=target, dim_size=len(graph.rest), dim=0)
            aggregate_change = processor.aggr_module(messages_change, index=target, dim_size=len(graph.rest), dim=0)
            update, update_change = centered(
                processor.nn, (1 + processor.eps) * latent + aggregate,
                (1 + processor.eps) * latent_change + aggregate_change,
            )
            latent, latent_change = centered(normalizer, latent + update, latent_change + update_change)
        if self.method == "stress":
            latent = torch.cat((latent[graph.triangles].mean(dim=1), torch.zeros_like(strain)), dim=-1)
            latent_change = torch.cat((latent_change[graph.triangles].mean(dim=1), strain), dim=-1)
        return centered(self.decoder, latent, latent_change)[1]

    def instantaneous_force(self, graph, position):
        if position.shape != graph.rest.shape:
            raise ValueError("Position must match the graph node array")
        deformation = deformation_gradient(graph, position)
        strain = symmetric_components(green_strain(deformation))
        prediction = self.encode(graph, strain)
        if self.method == "direct":
            nodal_deformation = nodal_average(graph, deformation.flatten(1)).reshape(-1, 2, 2)
            force = torch.einsum("nij,nj->ni", nodal_deformation, prediction)
            scale = graph.mass[:, None] * graph.material[:, 0:1] / (
                graph.material[:, 2:3] * graph.length_scale
            )
            return force * scale
        second_piola = symmetric_tensor(prediction) * graph.material[graph.triangles[:, 0], 0, None, None]
        return assemble_force(graph, deformation @ second_piola)

    def step(self, graph, position, velocity, boundary_position_next, boundary_velocity_next, dt, cache=None):
        start = cached_force(cache, self, graph, position)
        result = velocity_verlet(
            lambda state: self.instantaneous_force(graph, state), graph.mass,
            position, velocity, dt, graph.boundary_mask,
            boundary_position_next, boundary_velocity_next, start,
        )
        position = tracked_tensor(result.position)
        force = tracked_tensor(result.end_force)
        return StepResult(
            position, result.velocity, result.mean_force,
            make_cache(self, graph, position, force), result.start_force,
        )

    def forward(self, graph, position, velocity, boundary_position_next, boundary_velocity_next, dt):
        result = self.step(graph, position, velocity, boundary_position_next, boundary_velocity_next, dt)
        return result.position, result.velocity, result.mean_force
