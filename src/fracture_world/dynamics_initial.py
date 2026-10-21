from dataclasses import dataclass
import json
from pathlib import Path

import numpy as np

from .dynamics_data import MaterialTopology, _frozen, load_dynamics_trajectory, material_topology
from .mesh import VolumeMesh


SCHEMA = "fracture-world.dynamics-initial.v1"
ARRAYS = {"schema", "nodes", "tets", "positions", "velocities", "broken", "fixed", "time", "dt",
          "gravity", "material", "case_id", "condition_digest", "metadata_json"}
MATERIAL_KEYS = ("rest_density", "youngs_modulus", "poissons_ratio", "fracture_stress_threshold")


@dataclass(frozen=True)
class DynamicsInitial:
    topology: MaterialTopology
    positions: np.ndarray
    velocities: np.ndarray
    broken: np.ndarray
    fixed: np.ndarray
    time: float
    dt: float
    gravity: np.ndarray
    material: dict
    case_id: str
    condition_digest: str
    metadata: dict


def _checked_state(mesh, positions, velocities, broken, fixed, time, dt, gravity, material, case_id, condition, metadata):
    if set(material) != set(MATERIAL_KEYS) or not np.isfinite(list(material.values())).all():
        raise ValueError("initial material requires finite density, elastic constants and fracture threshold")
    if min(material[key] for key in MATERIAL_KEYS if key != "poissons_ratio") <= 0 or not -1 < material["poissons_ratio"] < 0.5:
        raise ValueError("initial material parameters are outside their physical range")
    topology = material_topology(mesh, material["rest_density"])
    positions, velocities, gravity = (np.asarray(value, dtype=float) for value in (positions, velocities, gravity))
    broken, fixed = np.asarray(broken), np.asarray(fixed)
    if positions.shape != topology.rest_positions.shape or velocities.shape != positions.shape or not np.isfinite(positions).all() or not np.isfinite(velocities).all():
        raise ValueError("initial positions and velocities must be finite material-corner three-vectors")
    if broken.shape != (len(topology.face_pairs),) or broken.dtype != np.dtype(bool):
        raise ValueError("initial fracture state must contain one Boolean per internal facet")
    if fixed.shape != (len(positions),) or fixed.dtype != np.dtype(bool):
        raise ValueError("initial support state must contain one Boolean per material corner")
    if gravity.shape != (3,) or not np.isfinite(gravity).all():
        raise ValueError("initial gravity must be a finite three-vector")
    if not np.isfinite([time, dt]).all() or time < 0 or dt <= 0:
        raise ValueError("initial time must be nonnegative and timestep must be positive")
    if not isinstance(case_id, str) or not case_id or not isinstance(condition, str) or len(condition) != 64 or any(value not in "0123456789abcdef" for value in condition):
        raise ValueError("initial loading case and SHA-256 condition digest are required")
    if not isinstance(metadata, dict):
        raise ValueError("initial metadata must be a JSON object")
    json.dumps(metadata, allow_nan=False)
    policy = metadata.get("support_policy")
    if policy not in {"prescribed_material_corners", "free", "learned_material_corner_release"}:
        raise ValueError("initial state requires an explicit supported support policy")
    constant = metadata.get("observed_source_support_constant")
    if not isinstance(constant, bool):
        raise ValueError("initial state requires explicit observed source support availability")
    if not constant and policy != "learned_material_corner_release":
        raise ValueError("support-changing source trajectories require a separate runtime support-release model")
    if policy == "learned_material_corner_release" and metadata.get("support_semantics") != "recorded_numerical_teacher_kinematic_flags":
        raise ValueError("learned support release must identify its numerical teacher flag semantics")
    if policy == "free" and fixed.any():
        raise ValueError("free support policy cannot contain fixed material corners")
    if (velocities[fixed] != 0).any():
        raise ValueError("currently fixed stationary supports require zero current velocity")
    corners = positions.reshape(-1, 4, 3)
    edges = corners[:, 1:] - corners[:, :1]
    scale = np.max(np.abs(edges), axis=(-2, -1))
    if (scale <= 0).any() or (np.linalg.det(edges / scale[:, None, None]) <= 0).any():
        raise ValueError("initial state contains an inverted or degenerate tetrahedron")
    intact = topology.face_pairs[~broken]
    for values in (positions, velocities, fixed):
        if not np.array_equal(values[intact[:, :, 0]], values[intact[:, :, 1]]):
            raise ValueError("initial intact facets must have identical shared material-corner states")
    for array in (mesh.nodes, mesh.tets, mesh.volumes, mesh.centers, mesh.faces.vertices,
                  mesh.faces.cells, mesh.faces.normals, mesh.faces.areas):
        array.setflags(write=False)
    return DynamicsInitial(topology, _frozen(positions), _frozen(velocities), _frozen(broken), _frozen(fixed),
                           float(time), float(dt), _frozen(gravity), dict(material), case_id, condition, dict(metadata))


def export_initial(case_path, trajectory_path, frame, output_file, *, support_policy):
    if support_policy not in {"prescribed", "free", "learned"}:
        raise ValueError("support_policy must be explicitly selected as prescribed, free or learned")
    trajectory = load_dynamics_trajectory(case_path, trajectory_path)
    if isinstance(frame, bool) or not isinstance(frame, (int, np.integer)) or not 0 <= frame < len(trajectory.times):
        raise ValueError("initial frame must select an explicit recorded state")
    constant = bool(np.all(trajectory.fixed == trajectory.fixed[0]))
    if not constant and support_policy != "learned":
        raise ValueError("support-changing source trajectories require a separate runtime support-release model")
    if support_policy == "free" and trajectory.fixed.any():
        raise ValueError("free support policy cannot be selected for fixed source supports")
    hashes = trajectory.metadata["source_hashes"]
    source_hashes = {"case": hashes[str(Path(case_path).resolve())],
                     "trajectory": hashes[str(Path(trajectory_path).resolve())],
                     "mesh": hashes[str(Path(case_path).resolve().parent.parent / "mesh.npz")]}
    policies = {"prescribed": "prescribed_material_corners", "free": "free", "learned": "learned_material_corner_release"}
    metadata = {"schema": SCHEMA, "units": "SI", "source_hashes": source_hashes,
                "source_frame": int(frame), "source": "observed_current_material_state",
                "support_policy": policies[support_policy],
                "support_semantics": "recorded_numerical_teacher_kinematic_flags",
                "observed_source_support_constant": constant, "native_runtime_required": False,
                "future_state_included": False, "dt_policy": "source_native_base_timestep",
                "parameter_status": trajectory.metadata.get("parameter_status")}
    initial = _checked_state(trajectory.topology.mesh, trajectory.positions[frame], trajectory.velocities[frame],
                             trajectory.broken[frame], trajectory.fixed[frame], trajectory.times[frame],
                             float(np.float32(trajectory.metadata["conditioning_context"]["dt"])), trajectory.gravity,
                             trajectory.material, trajectory.case_id, trajectory.condition_digest, metadata)
    save_initial(initial, output_file)
    return initial


def save_initial(initial, output_file):
    initial = _checked_state(initial.topology.mesh, initial.positions, initial.velocities, initial.broken, initial.fixed,
                             initial.time, initial.dt, initial.gravity, initial.material, initial.case_id,
                             initial.condition_digest, initial.metadata)
    output_file = Path(output_file)
    if output_file.suffix != ".npz":
        raise ValueError("initial state output must use the .npz extension")
    if output_file.exists():
        raise ValueError("initial state output already exists")
    output_file.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(output_file, schema=SCHEMA, nodes=initial.topology.mesh.nodes, tets=initial.topology.mesh.tets,
                        positions=initial.positions, velocities=initial.velocities, broken=initial.broken, fixed=initial.fixed,
                        time=initial.time, dt=initial.dt, gravity=initial.gravity,
                        material=np.asarray([initial.material[key] for key in MATERIAL_KEYS]), case_id=initial.case_id,
                        condition_digest=initial.condition_digest, metadata_json=json.dumps(initial.metadata, allow_nan=False))
    return output_file


def load_initial(path):
    with np.load(path, allow_pickle=False) as archive:
        if set(archive.files) != ARRAYS:
            raise ValueError("initial artifact must contain exactly one current state and supported metadata")
        arrays = {key: np.array(archive[key], copy=True) for key in archive.files}
    if arrays["schema"].shape != () or str(arrays["schema"].item()) != SCHEMA:
        raise ValueError("unsupported dynamics initial artifact schema")
    if arrays["material"].shape != (4,):
        raise ValueError("initial material vector must contain density, E, nu and fracture threshold")
    if arrays["time"].shape != () or arrays["dt"].shape != () or arrays["case_id"].shape != () or arrays["condition_digest"].shape != ():
        raise ValueError("initial clock and condition identifiers must be scalar values")
    mesh = VolumeMesh(arrays["nodes"], arrays["tets"])
    if not np.array_equal(mesh.tets, arrays["tets"]):
        raise ValueError("initial artifact must preserve positive original tetrahedron ordering")
    metadata = json.loads(str(arrays["metadata_json"].item()))
    if metadata.get("schema") != SCHEMA or metadata.get("units") != "SI" or metadata.get("future_state_included") is not False:
        raise ValueError("initial artifact requires SI metadata and current-state-only semantics")
    return _checked_state(mesh, arrays["positions"], arrays["velocities"], arrays["broken"], arrays["fixed"],
                          float(arrays["time"]), float(arrays["dt"]), arrays["gravity"],
                          dict(zip(MATERIAL_KEYS, arrays["material"].tolist(), strict=True)), str(arrays["case_id"].item()),
                          str(arrays["condition_digest"].item()), metadata)
