from dataclasses import dataclass
import math

import numpy as np
import torch
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import connected_components
from scipy.spatial import cKDTree
from torch import nn
from torch.nn import functional as F
from torch_geometric.data import HeteroData
from torch_geometric.nn import GATv2Conv, HGTConv
from torchdiffeq import odeint


DYNAMICS_INPUT_DIMS = {"corner": 21, "cell": 17, "face": 15, "body": 8, "interaction": 14}
DYNAMICS_EDGE_TYPES = [
    ("corner", "belongs_to", "cell"),
    ("cell", "contains", "corner"),
    ("cell", "belongs_to", "body"),
    ("body", "contains", "cell"),
    ("corner", "bounds", "face"),
    ("face", "borders", "corner"),
    ("corner", "participates", "interaction"),
    ("interaction", "affects", "corner"),
]
DYNAMICS_METADATA = (list(DYNAMICS_INPUT_DIMS), DYNAMICS_EDGE_TYPES)


@dataclass(frozen=True)
class FractureFlowContext:
    features: torch.Tensor
    edge_index: torch.Tensor


class DynamicsGraphCache:
    def __init__(self):
        self.entries = {}

    def resolve(self, name, tensors, build):
        if torch.is_inference_mode_enabled() or any(value.requires_grad for value in tensors):
            return build()
        try:
            versions = tuple(value._version for value in tensors)
        except RuntimeError:
            return build()
        previous = self.entries.get(name)
        if previous is not None:
            inputs, old_versions, result = previous
            if old_versions == versions and all(old is current for old, current in zip(inputs, tensors)):
                return result
        result = build()
        self.entries[name] = (tuple(tensors), versions, result)
        return result


def _cached(cache, name, tensors, build):
    return build() if cache is None else cache.resolve(name, tensors, build)


def build_fracture_adjacency(face_pairs):
    face_cells = (face_pairs[:, 0] // 4).detach().cpu().numpy()
    incident_faces = {}
    for face_id, cells in enumerate(face_cells):
        for cell_id in set(cells.tolist()):
            incident_faces.setdefault(cell_id, []).append(face_id)
    edges = {(left, right) for faces in incident_faces.values() for left in faces for right in faces if left != right}
    return torch.as_tensor(sorted(edges), device=face_pairs.device, dtype=torch.long).reshape(-1, 2).T.contiguous()


class FractureFlowLayer(nn.Module):
    def __init__(self, hidden, heads):
        super().__init__()
        self.local_attention = GATv2Conv(hidden, hidden // heads, heads=heads, dropout=0)
        self.global_attention = nn.MultiheadAttention(hidden, heads, dropout=0, batch_first=True)
        self.local_norm = nn.LayerNorm(hidden)
        self.global_norm = nn.LayerNorm(hidden)
        self.output_norm = nn.LayerNorm(hidden)
        self.feedforward = nn.Sequential(nn.Linear(hidden, 2 * hidden), nn.GELU(), nn.Linear(2 * hidden, hidden))

    def forward(self, tokens, edge_index):
        tokens = self.local_norm(tokens + self.local_attention(tokens, edge_index))
        update, _ = self.global_attention(tokens[None], tokens[None], tokens[None], need_weights=False)
        tokens = self.global_norm(tokens + update[0])
        return self.output_norm(tokens + self.feedforward(tokens))


def _validated_batch(batch):
    required = {"rest_positions", "positions", "velocities", "masses", "face_pairs", "broken", "fixed", "forces", "material", "dt"}
    if not required.issubset(batch):
        raise ValueError(f"dynamics input is missing {sorted(required - set(batch))}")
    positions = batch["positions"]
    if not isinstance(positions, torch.Tensor) or positions.ndim != 2 or positions.shape[1] != 3 or len(positions) < 4 or len(positions) % 4:
        raise ValueError("positions must contain four material corners per tetrahedron")
    if not positions.is_floating_point():
        raise ValueError("positions must be floating point tensors")
    count = len(positions)
    for key in ("rest_positions", "velocities", "forces"):
        if batch[key].shape != positions.shape:
            raise ValueError(f"{key} must have shape (N, 3)")
    if batch["masses"].shape != (count,):
        raise ValueError("masses must be positive with shape (N,)")
    if batch["fixed"].shape != (count,) or batch["fixed"].dtype != torch.bool:
        raise ValueError("fixed must be a boolean vector with shape (N,)")
    pairs = batch["face_pairs"]
    if pairs.ndim != 3 or pairs.shape[1:] != (3, 2) or pairs.dtype != torch.long:
        raise ValueError("face_pairs must have integer shape (F, 3, 2)")
    if batch["broken"].shape != (len(pairs),) or batch["broken"].dtype != torch.bool:
        raise ValueError("broken must be a boolean vector with shape (F,)")
    if batch["material"].shape != (4,):
        raise ValueError("material must contain density, Young modulus, Poisson ratio and fracture threshold")
    material = batch["material"]
    for key in required - {"face_pairs", "broken", "fixed", "dt"}:
        value = batch[key]
        if value.device != positions.device or value.dtype != positions.dtype:
            raise ValueError("floating point dynamics inputs must be finite and share dtype and device")
    for key in ("face_pairs", "broken", "fixed"):
        if batch[key].device != positions.device:
            raise ValueError("dynamics inputs must share a device")
    dt = torch.as_tensor(batch["dt"], device=positions.device, dtype=positions.dtype)
    if dt.numel() != 1:
        raise ValueError("dt must be a finite positive scalar")
    finite_inputs = torch.stack([torch.isfinite(batch[key]).all()
                                  for key in sorted(required - {"face_pairs", "broken", "fixed", "dt"})]).all()
    invalid = torch.stack(((batch["masses"] <= 0).any(), ((pairs < 0) | (pairs >= count)).any(),
                           (material[[0, 1, 3]] <= 0).any() | (material[2] <= -1) | (material[2] >= 0.5),
                           ~finite_inputs, ~torch.isfinite(dt).all() | (dt <= 0).any())).detach().cpu().tolist()
    errors = ("masses must be positive with shape (N,)", "face_pairs contain an invalid material corner",
              "material properties lie outside their admissible range",
              "floating point dynamics inputs must be finite and share dtype and device",
              "dt must be a finite positive scalar")
    for failed, message in zip(invalid, errors):
        if failed:
            raise ValueError(message)
    return dt.reshape(())


def _contact_pairs(batch, radius, cache=None):
    positions = batch["positions"].detach().cpu().numpy()
    count = len(positions)
    face_pairs = _cached(cache, "cpu_face_pairs", (batch["face_pairs"],),
                         lambda: batch["face_pairs"].detach().cpu().numpy().copy())

    def connectivity():
        intact_faces = face_pairs[~batch["broken"].detach().cpu().numpy()]
        intact = intact_faces.reshape(-1, 2)
        adjacency = coo_matrix((np.ones(len(intact), dtype=np.int8), (intact[:, 0], intact[:, 1])),
                               shape=(count, count)).tocsr()
        groups = connected_components(adjacency, directed=False, return_labels=True)[1]
        cell_pairs = np.sort(intact_faces[:, 0] // 4, axis=1)
        neighbors = np.unique(cell_pairs[:, 0] * (count // 4) + cell_pairs[:, 1])
        return groups, neighbors

    groups, neighboring_cells = _cached(cache, "contact_connectivity",
                                         (batch["rest_positions"], batch["face_pairs"], batch["broken"]), connectivity)
    pairs = cKDTree(positions).query_pairs(radius, output_type="ndarray")
    if len(pairs):
        pairs = pairs[(pairs[:, 0] // 4 != pairs[:, 1] // 4) & (groups[pairs[:, 0]] != groups[pairs[:, 1]])]
        cell_pairs = pairs // 4
        cell_codes = cell_pairs[:, 0] * (count // 4) + cell_pairs[:, 1]
        pairs = pairs[~np.isin(cell_codes, neighboring_cells)]
    return torch.as_tensor(pairs.reshape(-1, 2), device=batch["positions"].device, dtype=torch.long)


def constant_acceleration_prediction(batch):
    dt = _validated_batch(batch)
    return _constant_acceleration_prediction(batch, dt)


def _constant_acceleration_prediction(batch, dt):
    acceleration = batch["forces"] / batch["masses"][:, None]
    positions = batch["positions"] + dt * batch["velocities"] + 0.5 * dt.square() * acceleration
    velocities = batch["velocities"] + dt * acceleration
    position_increment = dt * batch["velocities"] + 0.5 * dt.square() * acceleration
    velocity_increment = dt * acceleration
    return {"positions": torch.where(batch["fixed"][:, None], batch["positions"], positions),
            "velocities": torch.where(batch["fixed"][:, None], torch.zeros_like(velocities), velocities),
            "position_increment": torch.where(batch["fixed"][:, None], torch.zeros_like(position_increment), position_increment),
            "velocity_increment": torch.where(batch["fixed"][:, None], -batch["velocities"], velocity_increment)}


def build_dynamics_graph(batch, length_scale=1.0, velocity_scale=1.0, mass_scale=1.0,
                         material_scale=(1.0, 1.0, 1.0, 1.0), contact_radius=0.1, proposal=None, physical_proposal=None):
    dt = _validated_batch(batch)
    return _build_dynamics_graph(batch, dt, length_scale, velocity_scale, mass_scale, material_scale,
                                  contact_radius, proposal, physical_proposal)


def _build_dynamics_graph(batch, dt, length_scale, velocity_scale, mass_scale, material_scale,
                           contact_radius, proposal, physical_proposal, cache=None):
    scales = np.asarray([length_scale, velocity_scale, mass_scale, contact_radius, *material_scale], dtype=float)
    if scales.shape != (8,) or not np.isfinite(scales).all() or (scales <= 0).any():
        raise ValueError("normalization scales and contact_radius must be finite and positive")
    x, v, rest = batch["positions"], batch["velocities"], batch["rest_positions"]
    mass, force, fixed = batch["masses"], batch["forces"], batch["fixed"]
    count = len(x)
    cell_count = count // 4
    pairs = batch["face_pairs"]
    center = _cached(cache, "rest_center", (rest, mass), lambda: (rest * mass[:, None]).sum(dim=0) / mass.sum())
    force_delta = force * dt / (mass[:, None] * velocity_scale)
    proposal_features = torch.zeros((count, 6), device=x.device, dtype=x.dtype)
    proposed_release_probability = torch.zeros(count, device=x.device, dtype=x.dtype)
    proposed_probability = torch.zeros(len(pairs), device=x.device, dtype=x.dtype)
    if physical_proposal is not None:
        proposal_features = torch.cat(((physical_proposal["positions"] - x) / length_scale,
                                       (physical_proposal["velocities"] - v) / velocity_scale), dim=-1)
    if proposal is not None:
        for name in ("positions", "velocities"):
            if proposal[name].shape != x.shape or not torch.isfinite(proposal[name]).all():
                raise ValueError("proposal positions and velocities must be finite with shape (N, 3)")
        if proposal["fracture_logits"].shape != (len(pairs),) or not torch.isfinite(proposal["fracture_logits"]).all():
            raise ValueError("proposal fracture_logits must be finite with shape (F,)")
        proposal_features = torch.cat(((proposal["positions"] - x) / length_scale,
                                       (proposal["velocities"] - v) / velocity_scale), dim=-1)
        proposed_probability = proposal.get("fracture_probability", proposal["fracture_logits"].sigmoid())
        if proposed_probability.shape != (len(pairs),) or not torch.isfinite(proposed_probability).all() or ((proposed_probability < 0) | (proposed_probability > 1)).any():
            raise ValueError("proposal fracture_probability must contain one finite probability per face")
        release_logits = proposal.get("support_release_logits")
        if release_logits is not None:
            if release_logits.shape != (count,) or not torch.isfinite(release_logits).all():
                raise ValueError("proposal support_release_logits must be finite with shape (N,)")
        if "support_release_probability" in proposal or release_logits is not None:
            proposed_release_probability = proposal.get("support_release_probability", release_logits.sigmoid() if release_logits is not None else None)
            if proposed_release_probability.shape != (count,) or not torch.isfinite(proposed_release_probability).all() or ((proposed_release_probability < 0) | (proposed_release_probability > 1)).any():
                raise ValueError("proposal support_release_probability must contain one finite probability per corner")
    corner_features = torch.cat(((x - center) / length_scale, (rest - center) / length_scale,
                                 v / velocity_scale, force_delta, (mass / mass_scale).log()[:, None],
                                 fixed.to(x.dtype)[:, None], proposal_features, proposed_release_probability[:, None]), dim=-1)
    current_cells = x.reshape(cell_count, 4, 3)
    current_edges = (current_cells[:, 1:] - current_cells[:, :1]).transpose(1, 2)

    def reference_geometry():
        rest_cells = rest.reshape(cell_count, 4, 3)
        reference_edges = (rest_cells[:, 1:] - rest_cells[:, :1]).transpose(1, 2)
        determinants = torch.linalg.det(reference_edges)
        if (determinants.abs() <= torch.finfo(x.dtype).tiny).any():
            raise ValueError("material tetrahedra must be nondegenerate")
        return determinants, torch.linalg.inv(reference_edges)

    determinants, reference_inverse = _cached(cache, "reference_geometry", (rest,), reference_geometry)
    deformation = current_edges @ reference_inverse
    material = batch["material"] / torch.as_tensor(material_scale, device=x.device, dtype=x.dtype)
    material_features = torch.stack((material[0].log(), material[1].log(), batch["material"][2], material[3].log()))
    cell_features = torch.cat((deformation.reshape(cell_count, 9), v.reshape(cell_count, 4, 3).mean(dim=1) / velocity_scale,
                               (determinants.abs() / (6 * length_scale ** 3)).log()[:, None],
                               material_features[None].expand(cell_count, -1)), dim=-1)
    left, right = pairs[:, :, 0], pairs[:, :, 1]

    def reference_faces():
        triangles = rest[left]
        normals = torch.linalg.cross(triangles[:, 1] - triangles[:, 0], triangles[:, 2] - triangles[:, 0], dim=-1)
        double_area = torch.linalg.vector_norm(normals, dim=-1)
        return normals / double_area.clamp_min(torch.finfo(x.dtype).tiny)[:, None], double_area

    normals, double_area = _cached(cache, "reference_faces", (rest, pairs), reference_faces)
    face_features = torch.cat((((x[left].mean(dim=1) + x[right].mean(dim=1)) / 2 - center) / length_scale,
                               (x[right] - x[left]).mean(dim=1) / length_scale,
                               (v[right] - v[left]).mean(dim=1) / velocity_scale, normals,
                               (double_area / (2 * length_scale ** 2)).clamp_min(torch.finfo(x.dtype).tiny).log()[:, None],
                               batch["broken"].to(x.dtype)[:, None], proposed_probability[:, None]), dim=-1)
    extent = (x.max(dim=0).values - x.min(dim=0).values) / length_scale
    mean_velocity = (v * mass[:, None]).sum(dim=0) / mass.sum() / velocity_scale
    body_features = torch.cat((extent, mean_velocity, (mass.sum() / mass_scale).log()[None],
                               (dt * velocity_scale / length_scale)[None]))[None]
    contacts = _contact_pairs(batch, contact_radius, cache)
    contact_count = len(contacts)
    load_features = torch.cat(((x - center) / length_scale, torch.zeros_like(x), v / velocity_scale,
                               torch.zeros_like(mass[:, None]), force_delta, torch.zeros_like(mass[:, None])), dim=-1)
    a, b = contacts[:, 0], contacts[:, 1]
    contact_features = torch.cat((((x[a] + x[b]) / 2 - center) / length_scale,
                                  (x[b] - x[a]) / length_scale, (v[b] - v[a]) / velocity_scale,
                                  torch.linalg.vector_norm(x[b] - x[a], dim=-1)[:, None] / length_scale,
                                  torch.zeros((contact_count, 3), device=x.device, dtype=x.dtype),
                                  torch.ones((contact_count, 1), device=x.device, dtype=x.dtype)), dim=-1)
    floor_features = torch.empty((0, 14), device=x.device, dtype=x.dtype)
    if "floor_height" in batch:
        floor = float(batch["floor_height"])
        if not np.isfinite(floor):
            raise ValueError("floor_height must be finite")
        projected = x.clone()
        projected[:, 1] = floor
        normals = torch.zeros_like(x)
        normals[:, 1] = 1
        floor_features = torch.cat(((projected - center) / length_scale, normals, -v / velocity_scale,
                                    (x[:, 1:2] - floor) / length_scale, torch.zeros_like(x),
                                    torch.full_like(mass[:, None], 2.0)), dim=-1)
    graph = HeteroData()
    features = (corner_features, cell_features, face_features, body_features,
                torch.cat((load_features, contact_features, floor_features)))
    for name, value in zip(DYNAMICS_INPUT_DIMS, features):
        graph[name].x = value
    corner_ids = torch.arange(count, device=x.device)
    cell_ids = torch.arange(cell_count, device=x.device)
    corner_cell = torch.stack((corner_ids, corner_ids // 4))
    cell_body = torch.stack((cell_ids, torch.zeros_like(cell_ids)))
    intact_faces = torch.arange(len(pairs), device=x.device)[~batch["broken"]]
    corner_face = torch.stack((pairs[~batch["broken"]].reshape(-1), intact_faces.repeat_interleave(6)))
    load_edges = torch.stack((corner_ids, corner_ids))
    contact_edges = torch.stack((contacts.reshape(-1), torch.arange(count, count + contact_count, device=x.device).repeat_interleave(2)))
    floor_edges = torch.stack((corner_ids, corner_ids + count + contact_count)) if len(floor_features) else load_edges[:, :0]
    corner_interaction = torch.cat((load_edges, contact_edges, floor_edges), dim=1)
    relations = (corner_cell, corner_cell.flip(0), cell_body, cell_body.flip(0), corner_face, corner_face.flip(0),
                 corner_interaction, corner_interaction.flip(0))
    for relation, edges in zip(DYNAMICS_EDGE_TYPES, relations):
        graph[relation].edge_index = edges
    graph["interaction"].contact_pairs = contacts
    graph["interaction"].floor_corner_ids = corner_ids if len(floor_features) else corner_ids[:0]
    return graph


class TemporalDynamicsModel(nn.Module):
    def __init__(self, hidden=64, heads=4, graph_layers=4, transformer_layers=1,
                 length_scale=1.0, velocity_scale=1.0, mass_scale=1.0,
                 material_scale=(1.0, 1.0, 1.0, 1.0), contact_radius=0.1, learn_support_release=False,
                 flow_architecture="coupled", flow_layers=2, motion_prior="backward_euler"):
        super().__init__()
        if hidden < 1 or heads < 1 or hidden % heads or graph_layers < 2 or transformer_layers < 1:
            raise ValueError("invalid temporal model dimensions")
        scale_values = np.asarray([length_scale, velocity_scale, mass_scale, contact_radius, *material_scale], dtype=float)
        if scale_values.shape != (8,) or not np.isfinite(scale_values).all() or (scale_values <= 0).any():
            raise ValueError("normalization scales and contact_radius must be finite and positive")
        if not isinstance(learn_support_release, bool):
            raise ValueError("learn_support_release must be boolean")
        if flow_architecture not in {"coupled", "independent"}:
            raise ValueError("flow_architecture must be coupled or independent")
        if motion_prior not in {"backward_euler", "constant_acceleration"}:
            raise ValueError("unsupported physical motion prior")
        if not isinstance(flow_layers, int) or isinstance(flow_layers, bool) or flow_layers < 1:
            raise ValueError("flow_layers must be a positive integer")
        self.config = dict(hidden=hidden, heads=heads, graph_layers=graph_layers, transformer_layers=transformer_layers,
                           length_scale=length_scale, velocity_scale=velocity_scale, mass_scale=mass_scale,
                           material_scale=list(material_scale), contact_radius=contact_radius,
                           learn_support_release=learn_support_release, flow_architecture=flow_architecture,
                           flow_layers=flow_layers, motion_prior=motion_prior)
        self._graph_cache = DynamicsGraphCache()
        self.fracture_logit_offset = 0.0
        self.support_logit_offset = 0.0
        self.encoders = nn.ModuleDict({name: nn.Linear(width, hidden) for name, width in DYNAMICS_INPUT_DIMS.items()})
        self.graph_layers = nn.ModuleList([HGTConv(hidden, hidden, DYNAMICS_METADATA, heads=heads) for _ in range(graph_layers)])
        self.norms = nn.ModuleList([nn.ModuleDict({name: nn.LayerNorm(hidden) for name in DYNAMICS_INPUT_DIMS}) for _ in range(graph_layers)])
        layer = nn.TransformerEncoderLayer(hidden, heads, 4 * hidden, dropout=0, activation="gelu", batch_first=True)
        self.transformer = nn.TransformerEncoder(layer, transformer_layers, norm=nn.LayerNorm(hidden), enable_nested_tensor=False)
        self.corner_fusion = nn.Sequential(nn.Linear(2 * hidden, hidden), nn.GELU(), nn.LayerNorm(hidden))
        self.face_fusion = nn.Sequential(nn.Linear(3 * hidden, hidden), nn.GELU(), nn.LayerNorm(hidden))
        self.motion_head = nn.Linear(hidden, 6)
        self.fracture_head = nn.Linear(hidden, 1)
        self.support_head = nn.Linear(hidden, 1)
        self.flow_head = nn.Sequential(nn.Linear(hidden + 4, hidden), nn.GELU(), nn.Linear(hidden, 1))
        if flow_architecture == "coupled":
            self.flow_encoder = nn.Linear(hidden + 4, hidden)
            self.flow_layers = nn.ModuleList([FractureFlowLayer(hidden, heads) for _ in range(flow_layers)])
        nn.init.zeros_(self.motion_head.weight)
        nn.init.zeros_(self.motion_head.bias)
        nn.init.zeros_(self.support_head.weight)
        nn.init.zeros_(self.support_head.bias)

    def encode(self, batch, proposal=None):
        dt = _validated_batch(batch)
        physical = _constant_acceleration_prediction(batch, dt) if self.config["motion_prior"] == "constant_acceleration" else None
        graph = _build_dynamics_graph(batch, dt, **{key: self.config[key] for key in (
            "length_scale", "velocity_scale", "mass_scale", "material_scale", "contact_radius")}, proposal=proposal,
                                      physical_proposal=physical, cache=self._graph_cache)
        hidden = {name: F.gelu(self.encoders[name](features)) for name, features in graph.x_dict.items()}
        for layer, norms in zip(self.graph_layers, self.norms):
            updates = layer(hidden, graph.edge_index_dict)
            hidden = {name: F.gelu(norms[name](value + updates[name])) if updates.get(name) is not None else value
                      for name, value in hidden.items()}
        cell_context = self.transformer(torch.cat((hidden["cell"], hidden["body"]), dim=0)[None])[0, :-1]
        corner_context = self.corner_fusion(torch.cat((hidden["corner"], cell_context.repeat_interleave(4, dim=0)), dim=-1))
        pair_cells = batch["face_pairs"][:, 0] // 4
        face_context = self.face_fusion(torch.cat((hidden["face"], cell_context[pair_cells[:, 0]], cell_context[pair_cells[:, 1]]), dim=-1))
        return corner_context, face_context

    def _prediction(self, batch, corner_context, face_context, proposal=None):
        dt = torch.as_tensor(batch["dt"], device=batch["positions"].device, dtype=batch["positions"].dtype)
        motion = self.motion_head(corner_context)
        scale = self.config["velocity_scale"]
        release_logits = self.support_head(corner_context).squeeze(-1)
        if proposal is None:
            velocity_increment = batch["forces"] * dt / batch["masses"][:, None] + scale * motion[:, 3:]
            velocities = batch["velocities"] + velocity_increment
            if self.config["motion_prior"] == "constant_acceleration":
                position_increment = dt * batch["velocities"] + 0.5 * dt.square() * batch["forces"] / batch["masses"][:, None] + dt * scale * motion[:, :3]
            else:
                position_increment = dt * velocities + dt * scale * motion[:, :3]
            positions = batch["positions"] + position_increment
            logits = self.fracture_head(face_context).squeeze(-1)
        else:
            positions = proposal["positions"] + dt * scale * motion[:, :3]
            velocities = proposal["velocities"] + scale * motion[:, 3:]
            position_increment = proposal.get("position_increment", proposal["positions"] - batch["positions"]) + dt * scale * motion[:, :3]
            velocity_increment = proposal.get("velocity_increment", proposal["velocities"] - batch["velocities"]) + scale * motion[:, 3:]
            logits = proposal["fracture_logits"] + self.fracture_head(face_context).squeeze(-1)
            release_logits = release_logits + proposal.get("support_release_logits", torch.zeros_like(release_logits))
        release_probability = torch.where(batch["fixed"], (release_logits - self.support_logit_offset).sigmoid(), torch.zeros_like(release_logits))
        next_fixed = batch["fixed"] & (release_probability < 0.5) if self.config["learn_support_release"] else batch["fixed"].clone()
        positions = torch.where(next_fixed[:, None], batch["positions"], positions)
        velocities = torch.where(next_fixed[:, None], torch.zeros_like(velocities), velocities)
        position_increment = torch.where(next_fixed[:, None], torch.zeros_like(position_increment), position_increment)
        velocity_increment = torch.where(next_fixed[:, None], -batch["velocities"], velocity_increment)
        probability = torch.where(batch["broken"], torch.ones_like(logits), (logits - self.fracture_logit_offset).sigmoid())
        return dict(positions=positions, velocities=velocities, position_increment=position_increment,
                    velocity_increment=velocity_increment, fracture_logits=logits,
                    fracture_probability=probability, broken=batch["broken"] | (probability >= 0.5),
                    support_release_logits=release_logits, support_release_probability=release_probability,
                    next_fixed=next_fixed)

    def forward(self, batch):
        return self._prediction(batch, *self.encode(batch))

    def forward_with_flow_context(self, batch):
        corner_context, face_context = self.encode(batch)
        adjacency = _cached(self._graph_cache, "fracture_adjacency", (batch["face_pairs"],),
                            lambda: build_fracture_adjacency(batch["face_pairs"]))
        return self._prediction(batch, corner_context, face_context), FractureFlowContext(face_context, adjacency)

    def fracture_context(self, batch):
        return self.encode(batch)[1]

    def prepare_fracture_flow(self, batch):
        adjacency = _cached(self._graph_cache, "fracture_adjacency", (batch["face_pairs"],),
                            lambda: build_fracture_adjacency(batch["face_pairs"]))
        return FractureFlowContext(self.fracture_context(batch), adjacency)

    def fracture_velocity(self, batch, field, time, context=None):
        if field.shape != batch["broken"].shape:
            raise ValueError("fracture field must be finite with one value per internal face")
        if not field.is_floating_point() or field.dtype != batch["positions"].dtype or field.device != batch["positions"].device:
            raise ValueError("fracture field must share the floating point dtype and device of the current state")
        context = self.prepare_fracture_flow(batch) if context is None else context
        features = context.features if isinstance(context, FractureFlowContext) else context
        if features.shape != (len(field), self.config["hidden"]) or features.dtype != field.dtype or features.device != field.device:
            raise ValueError("fracture context must have one compatible hidden vector per face")
        time = torch.as_tensor(time, dtype=field.dtype, device=field.device)
        if time.numel() != 1:
            raise ValueError("fracture flow time must be scalar in [0, 1]")
        invalid_field, invalid_time = torch.stack((~torch.isfinite(field).all(),
                                                   ~torch.isfinite(time).all() | (time < 0).any() | (time > 1).any())).detach().cpu().tolist()
        if invalid_field:
            raise ValueError("fracture field must be finite with one value per internal face")
        if invalid_time:
            raise ValueError("fracture flow time must be scalar in [0, 1]")
        if not len(field):
            return field * 0
        time = time.reshape(()).expand(len(field))
        field = torch.where(batch["broken"], torch.ones_like(field), field)
        flow_values = torch.stack((field, time, torch.sin(math.pi * time), torch.cos(math.pi * time)), dim=-1)
        if self.config["flow_architecture"] == "coupled":
            edge_index = context.edge_index if isinstance(context, FractureFlowContext) else build_fracture_adjacency(batch["face_pairs"])
            features = self.flow_encoder(torch.cat((features, flow_values), dim=-1))
            for layer in self.flow_layers:
                features = layer(features, edge_index)
        return self.flow_head(torch.cat((features, flow_values), dim=-1)).squeeze(-1).masked_fill(batch["broken"], 0)

    @torch.no_grad()
    def sample_fracture(self, batch, steps=8, noise=None):
        if not isinstance(steps, int) or steps < 1:
            raise ValueError("fracture flow steps must be a positive integer")
        if noise is None:
            noise = torch.randn_like(batch["broken"], dtype=batch["positions"].dtype)
        if noise.shape != batch["broken"].shape or not torch.isfinite(noise).all():
            raise ValueError("fracture noise must be finite with one value per internal face")
        initial = torch.where(batch["broken"], torch.ones_like(noise), noise)
        if not len(initial):
            return initial
        context = self.prepare_fracture_flow(batch)

        def vector_field(time, field):
            velocity = self.fracture_velocity(batch, field, time, context=context)
            if not torch.isfinite(velocity).all():
                raise FloatingPointError("nonfinite fracture flow velocity")
            return velocity

        times = torch.tensor([0.0, 1.0], device=initial.device, dtype=initial.dtype)
        final = odeint(vector_field, initial, times, method="rk4", options={"step_size": 1.0 / steps})[-1]
        if not torch.isfinite(final).all():
            raise FloatingPointError("nonfinite sampled fracture field")
        return torch.where(batch["broken"], torch.ones_like(final), final)


class NeuralDynamicsCorrector(TemporalDynamicsModel):
    def __init__(self, **config):
        super().__init__(**config)
        nn.init.zeros_(self.fracture_head.weight)
        nn.init.zeros_(self.fracture_head.bias)

    def forward(self, batch, proposal):
        return self._prediction(batch, *self.encode(batch, proposal=proposal), proposal=proposal)
