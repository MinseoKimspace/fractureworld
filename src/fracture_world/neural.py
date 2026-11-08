import numpy as np
import torch
from torch import nn
from torch_geometric.data import Data
from torch_geometric.nn import GINEConv

from .topology import boundary_edges, mesh_edges


class MeshGraph(Data):
    def __inc__(self, key, value, *args, **kwargs):
        if key in {"triangles", "boundary_edges"}:
            return self.num_nodes
        return super().__inc__(key, value, *args, **kwargs)


def build_graph(
    mesh,
    mass,
    boundary_mask,
    material=(1.0, 0.25, 1.0),
    thickness=1.0,
    dtype=torch.float32,
    device="cpu",
):
    mass = np.asarray(mass, dtype=np.float64)
    boundary_mask = np.asarray(boundary_mask, dtype=bool)
    material = np.asarray(material, dtype=np.float64)
    if mass.shape != (len(mesh.rest),) or not np.isfinite(mass).all() or np.any(mass <= 0):
        raise ValueError("Mass must contain a positive finite value for every node")
    if boundary_mask.shape != mesh.rest.shape:
        raise ValueError("Boundary mask must have shape (nodes, 2)")
    if material.shape != (3,) or not np.isfinite(material).all():
        raise ValueError("Material must contain Young's modulus, Poisson ratio and density")
    if material[0] <= 0 or material[2] <= 0 or not -1 < material[1] < 0.5:
        raise ValueError("Material parameters are outside their admissible range")
    if not np.isfinite(thickness) or thickness <= 0:
        raise ValueError("Thickness must be positive and finite")
    extent = np.ptp(mesh.rest, axis=0)
    length = float(np.max(extent))
    if length <= 0:
        raise ValueError("Mesh must have positive extent")
    triangle_points = mesh.rest[mesh.triangles]
    rest_matrix = np.stack(
        (triangle_points[:, 1] - triangle_points[:, 0], triangle_points[:, 2] - triangle_points[:, 0]),
        axis=-1,
    )
    inverse = np.linalg.inv(rest_matrix)
    gradient = np.concatenate((-inverse.sum(axis=1, keepdims=True), inverse), axis=1)
    area = np.linalg.det(rest_matrix) / 2
    undirected = mesh_edges(mesh)
    edge_index = np.concatenate((undirected, undirected[:, ::-1]), axis=0).T.copy()
    boundary = boundary_edges(mesh)
    boundary_keys = {tuple(sorted(edge)) for edge in boundary.tolist()}
    cut_keys = {tuple(sorted(edge)) for edge in mesh.cut_edges.tolist()}
    boundary_flags = np.asarray(
        [tuple(sorted(edge)) in boundary_keys for edge in edge_index.T.tolist()], dtype=float
    )[:, None]
    material_edges = mesh.original_vertex[edge_index.T]
    cut_flags = np.asarray(
        [tuple(sorted(edge)) in cut_keys for edge in material_edges.tolist()], dtype=float
    )[:, None]

    def floating(value):
        return torch.as_tensor(value, dtype=dtype, device=device)

    return MeshGraph(
        rest=floating(mesh.rest),
        triangles=torch.as_tensor(mesh.triangles, dtype=torch.long, device=device),
        original_vertex=torch.as_tensor(mesh.original_vertex, dtype=torch.long, device=device),
        cut_edges=torch.as_tensor(mesh.cut_edges, dtype=torch.long, device=device),
        mass=floating(mass),
        boundary_mask=torch.as_tensor(boundary_mask, dtype=torch.bool, device=device),
        edge_index=torch.as_tensor(edge_index, dtype=torch.long, device=device),
        boundary_edges=torch.as_tensor(boundary, dtype=torch.long, device=device),
        edge_boundary=floating(boundary_flags),
        edge_cut=floating(cut_flags),
        shape_gradient=floating(gradient),
        area=floating(area[:, None]),
        thickness=floating(np.full((len(area), 1), thickness)),
        material=floating(np.broadcast_to(material, (len(mesh.rest), 3)).copy()),
        length_scale=floating(np.full((len(mesh.rest), 1), length)),
        num_nodes=len(mesh.rest),
    )


def multilayer(input_size, hidden_size, output_size):
    return nn.Sequential(
        nn.Linear(input_size, hidden_size),
        nn.SiLU(),
        nn.Linear(hidden_size, output_size),
    )


class DynamicsModel(nn.Module):
    def __init__(self, method="stress", hidden_size=64, layers=3):
        super().__init__()
        if method not in {"direct", "stress"}:
            raise ValueError("Method must be direct or stress")
        if hidden_size < 4 or layers < 1:
            raise ValueError("Hidden size and message-passing depth must be positive")
        self.method = method
        self.hidden_size = int(hidden_size)
        self.layers = int(layers)
        self.node_encoder = multilayer(15, hidden_size, hidden_size)
        self.edge_encoder = multilayer(10, hidden_size, hidden_size)
        self.processor = nn.ModuleList(
            GINEConv(multilayer(hidden_size, hidden_size, hidden_size), train_eps=True)
            for _ in range(layers)
        )
        self.normalizers = nn.ModuleList(nn.LayerNorm(hidden_size) for _ in range(layers))
        self.decoder = multilayer(
            hidden_size + (4 if method == "stress" else 0),
            hidden_size,
            4 if method == "stress" else 2,
        )
        nn.init.zeros_(self.decoder[-1].weight)
        nn.init.zeros_(self.decoder[-1].bias)

    def configuration(self):
        return {"method": self.method, "hidden_size": self.hidden_size, "layers": self.layers}

    def forward(
        self,
        graph,
        position,
        velocity,
        boundary_position_next,
        boundary_velocity_next,
        dt,
    ):
        if position.shape != graph.rest.shape or velocity.shape != graph.rest.shape:
            raise ValueError("Position and velocity must match the graph node array")
        if boundary_position_next.shape != position.shape or boundary_velocity_next.shape != position.shape:
            raise ValueError("Boundary targets must match the graph node array")
        if not np.isfinite(dt) or dt <= 0:
            raise ValueError("Time step must be positive and finite")
        length = graph.length_scale
        young = graph.material[:, 0:1]
        poisson = graph.material[:, 1:2]
        density = graph.material[:, 2:3]
        mass = graph.mass[:, None]
        wave_speed = torch.sqrt(young / density)
        mask = graph.boundary_mask.to(position.dtype)
        node_features = torch.cat(
            (
                graph.rest / length,
                (position - graph.rest) / length,
                velocity / wave_speed,
                mass / (density * length.square()),
                mask,
                (boundary_position_next - position) / length * mask,
                boundary_velocity_next / wave_speed * mask,
                poisson,
                dt * wave_speed / length,
            ),
            dim=-1,
        )
        source, target = graph.edge_index
        relative_position = (position[source] - position[target]) / length[source]
        relative_rest = (graph.rest[source] - graph.rest[target]) / length[source]
        edge_features = torch.cat(
            (
                relative_position,
                relative_rest,
                relative_position.norm(dim=-1, keepdim=True),
                relative_rest.norm(dim=-1, keepdim=True),
                (velocity[source] - velocity[target]) / wave_speed[source],
                graph.edge_boundary,
                graph.edge_cut,
            ),
            dim=-1,
        )
        latent = self.node_encoder(node_features)
        edges = self.edge_encoder(edge_features)
        for processor, normalizer in zip(self.processor, self.normalizers):
            latent = normalizer(latent + processor(latent, graph.edge_index, edges))
        if self.method == "direct":
            acceleration = self.decoder(latent) * young / (density * length)
            force = mass * acceleration
        else:
            cells = graph.triangles
            deformation = torch.einsum("eai,eaj->eij", position[cells], graph.shape_gradient)
            element_features = torch.cat((latent[cells].mean(dim=1), deformation.flatten(1)), dim=-1)
            stress = self.decoder(element_features).reshape(-1, 2, 2)
            stress = stress * young[cells[:, 0], :, None]
            corner_force = -torch.einsum("eij,eaj->eai", stress, graph.shape_gradient)
            corner_force = corner_force * (graph.area * graph.thickness)[:, :, None]
            force = torch.zeros_like(position).index_add(0, cells.flatten(), corner_force.reshape(-1, 2))
        next_velocity = velocity + dt * force / mass
        next_position = position + 0.5 * dt * (velocity + next_velocity)
        next_velocity = torch.where(graph.boundary_mask, boundary_velocity_next, next_velocity)
        next_position = torch.where(graph.boundary_mask, boundary_position_next, next_position)
        return next_position, next_velocity, force
