from dataclasses import dataclass, replace
from functools import lru_cache
import hashlib
import json
from pathlib import Path

import numpy as np

from .data import _mesh_digest, load_prepared_case
from .io import file_hash
from .mesh import VolumeMesh, rigid_pose


def _frozen(values, dtype=None):
    result = np.array(values, dtype=dtype, copy=True)
    result.setflags(write=False)
    return result


def _digest(value):
    encoded = json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()
    return hashlib.sha256(encoded).hexdigest()


@dataclass(frozen=True)
class MaterialTopology:
    mesh: VolumeMesh
    rest_positions: np.ndarray
    masses: np.ndarray
    tet_corner_ids: np.ndarray
    corner_tets: np.ndarray
    face_pairs: np.ndarray
    internal_face_ids: np.ndarray
    facet_areas: np.ndarray
    facet_normals: np.ndarray
    fingerprint: str

    @property
    def facet_pairs(self):
        return self.face_pairs.reshape(-1, 2)


def material_topology(mesh, density):
    if not isinstance(mesh, VolumeMesh):
        raise TypeError("material topology requires a VolumeMesh")
    if not np.isscalar(density) or not np.isfinite(density) or density <= 0:
        raise ValueError("rest density must be finite and positive")
    corner_ids = np.arange(len(mesh.tets) * 4).reshape(-1, 4)
    pairs = mesh.corner_face_pairs
    return MaterialTopology(
        mesh, _frozen(mesh.nodes[mesh.tets].reshape(-1, 3)),
        _frozen(np.repeat(mesh.volumes * density / 4, 4)), _frozen(corner_ids),
        _frozen(np.repeat(np.arange(len(mesh.tets)), 4)), _frozen(pairs),
        _frozen(mesh.internal_faces), _frozen(mesh.faces.areas[mesh.internal_faces]),
        _frozen(mesh.faces.normals[mesh.internal_faces]), _mesh_digest(mesh),
    )


@dataclass(frozen=True)
class DynamicsTransition:
    positions: np.ndarray
    velocities: np.ndarray
    broken: np.ndarray
    fixed: np.ndarray
    target_positions: np.ndarray
    target_velocities: np.ndarray
    target_broken: np.ndarray
    target_fixed: np.ndarray
    dt: float
    time: float
    frame: int


@dataclass(frozen=True)
class DynamicsTrajectory:
    topology: MaterialTopology
    positions: np.ndarray
    velocities: np.ndarray
    broken: np.ndarray
    fixed: np.ndarray
    times: np.ndarray
    gravity: np.ndarray
    material: dict
    initial_impulses: np.ndarray
    case_id: str
    condition_digest: str
    metadata: dict

    @property
    def shape_family(self):
        context = self.metadata.get("conditioning_context", {})
        return self.metadata.get("shape_family", context.get("shape_family", self.topology.fingerprint))

    def __len__(self):
        return len(self.times) - 1

    def transition(self, frame):
        if isinstance(frame, bool) or not isinstance(frame, (int, np.integer)) or not 0 <= frame < len(self):
            raise ValueError("transition frame must select an available consecutive recorded pair")
        return DynamicsTransition(
            self.positions[frame], self.velocities[frame], self.broken[frame], self.fixed[frame],
            self.positions[frame + 1], self.velocities[frame + 1], self.broken[frame + 1], self.fixed[frame + 1],
            float(self.times[frame + 1] - self.times[frame]), float(self.times[frame]), int(frame),
        )


def _context(case):
    metadata = case.metadata
    if metadata.get("source") != "FEMFX" or metadata.get("schema_version") != "fracture-world.femfx.v1" or metadata.get("units") != "SI" or metadata.get("supervision_kind") != "native_facet_fracture":
        raise ValueError("dynamics training requires native facet fracture trajectories in SI units")
    settings = metadata.get("native_settings", {})
    if settings.get("plasticity") is not False or settings.get("enable_fracture") is not True:
        raise ValueError("the first dynamics model requires elastic fracture trajectories with plasticity disabled")
    context = metadata.get("conditioning_context")
    if not isinstance(context, dict) or _digest(context) != metadata.get("conditioning_context_sha256"):
        raise ValueError("dynamics conditioning context hash mismatch")
    if context.get("mesh_si_fingerprint") != _mesh_digest(case.mesh):
        raise ValueError("dynamics conditioning mesh mismatch")
    material = context.get("material", {})
    required = {"rest_density", "youngs_modulus", "poissons_ratio", "fracture_stress_threshold"}
    if set(material) != required or not np.isfinite(list(material.values())).all():
        raise ValueError("dynamics material requires finite density, elastic constants and fracture threshold")
    if min(material[name] for name in required - {"poissons_ratio"}) <= 0 or not -1 < material["poissons_ratio"] < 0.5:
        raise ValueError("dynamics material parameters are outside their physical range")
    if metadata.get("density_kg_per_m3") != material["rest_density"]:
        raise ValueError("dynamics density metadata differs from its material")
    gravity = np.asarray(context.get("gravity"), dtype=float)
    if gravity.shape != (3,) or not np.isfinite(gravity).all():
        raise ValueError("dynamics gravity must be a finite three-vector")
    dt, steps = context.get("dt"), context.get("steps")
    if not isinstance(dt, (int, float)) or isinstance(dt, bool) or not np.isfinite(dt) or dt <= 0:
        raise ValueError("native timestep must be finite and positive")
    if not isinstance(steps, int) or isinstance(steps, bool) or steps < 1:
        raise ValueError("native step count must be a positive integer")
    fixed = np.asarray(context.get("fixed_vertices"))
    if fixed.ndim != 1 or (len(fixed) and not np.issubdtype(fixed.dtype, np.integer)) or (fixed < 0).any() or (fixed >= len(case.mesh.nodes)).any():
        raise ValueError("fixed material vertex indices are invalid")
    if len(np.unique(fixed)) != len(fixed):
        raise ValueError("fixed material vertex indices must be unique")
    if context.get("loading") not in {"instantaneous_vertex_impulses", "gravity_driven_drop"}:
        raise ValueError("only explicitly recorded initial impulses and gravity-driven drops are supported")
    if context.get("loading") == "gravity_driven_drop":
        floor = context.get("floor_height_m")
        bounds = context.get("scene_options", {}).get("collision_bounds", [])
        if fixed.size or gravity[1] >= 0 or gravity[0] != 0 or gravity[2] != 0:
            raise ValueError("drop references require free supports and downward Y gravity")
        if not isinstance(floor, (int, float)) or not np.isfinite(floor) or len(bounds) != 6 or bounds[2] != floor:
            raise ValueError("drop references require a recorded native floor plane")
    return context, material, gravity, fixed.astype(np.int64)


def _loading(case, topology, fixed):
    momentum = np.zeros_like(case.mesh.nodes)
    impulses = case.metadata.get("case", {}).get("vertex_impulses")
    loading = case.metadata["conditioning_context"]["loading"]
    if not isinstance(impulses, list) or (not impulses and loading != "gravity_driven_drop"):
        raise ValueError("dynamics case requires initial vertex impulses")
    if loading == "gravity_driven_drop" and (impulses or "initial_pose" not in case.metadata["case"]):
        raise ValueError("drop references require an explicit rigid initial pose and no impulses")
    for item in impulses:
        vertex, impulse = item.get("vertex"), np.asarray(item.get("impulse"), dtype=float)
        if isinstance(vertex, bool) or not isinstance(vertex, int) or not 0 <= vertex < len(momentum) or vertex in fixed:
            raise ValueError("initial impulse must select an unfixed material vertex")
        if impulse.shape != (3,) or not np.isfinite(impulse).all():
            raise ValueError("initial impulse must be a finite three-vector")
        momentum[vertex] += impulse
    nodal_mass = np.bincount(case.mesh.tets.ravel(), weights=topology.masses, minlength=len(momentum))
    if (nodal_mass <= 0).any():
        raise ValueError("every material vertex must have positive rest mass")
    rigid_velocity = np.asarray(case.metadata["case"].get("initial_linear_velocity", [0.0, 0.0, 0.0]), dtype=float)
    if rigid_velocity.shape != (3,) or not np.isfinite(rigid_velocity).all() or (np.any(rigid_velocity) and len(fixed)):
        raise ValueError("rigid initial velocity must be finite and preserve prescribed supports")
    velocity = momentum / nodal_mass[:, None] + rigid_velocity
    initial_impulses = topology.masses[:, None] * velocity[case.mesh.tets].reshape(-1, 3)
    canonical = {**case.metadata["conditioning_context"]}
    for key in ("dt", "steps", "threads", "native_revision", "binding_sha256", "shape_id", "shape_family", "geometry_source"):
        canonical.pop(key, None)
    active = np.flatnonzero(np.linalg.norm(momentum, axis=1) > 0)
    if not len(active) and impulses:
        raise ValueError("initial vertex impulses cancel exactly")
    pose = case.metadata["case"].get("initial_pose")
    if pose is not None:
        canonical["initial_pose"] = rigid_pose(case.mesh.nodes, pose).tolist()
    if np.any(rigid_velocity):
        canonical["initial_linear_velocity"] = rigid_velocity.tolist()
    canonical["initial_momentum"] = [[int(vertex), *momentum[vertex].tolist()] for vertex in active]
    return velocity, initial_impulses, _digest(canonical)


def _validate_history(arrays, case, topology, context, fixed, initial_velocity):
    required = {"times", "corner_positions", "corner_velocities", "cut_labels", "original_tet_ids",
                "native_node_offsets", "native_corner_vertex_ids", "native_nodal_positions", "native_nodal_velocities",
                "native_nodal_rest_positions", "native_nodal_masses", "native_nodal_original_ids",
                "native_nodal_kinematic", "native_nodal_flags"}
    if not required.issubset(arrays):
        raise ValueError("dynamics trajectory requires complete recorded nodal state including support flags")
    times, positions, velocities, cuts = (arrays[key] for key in ("times", "corner_positions", "corner_velocities", "cut_labels"))
    frames, cells, facets = times.size, len(case.mesh.tets), len(topology.face_pairs)
    if times.ndim != 1 or frames < 2 or not np.isfinite(times).all() or times[0] != 0 or (np.diff(times) <= 0).any():
        raise ValueError("trajectory times must start at zero and strictly increase")
    clock = np.r_[0.0, np.cumsum(np.full(context["steps"], np.float32(context["dt"])), dtype=np.float32).astype(float)]
    step_ids = np.searchsorted(clock, times)
    if (step_ids >= len(clock)).any() or not np.array_equal(clock[step_ids], times) or step_ids[-1] != context["steps"]:
        raise ValueError("trajectory times differ from the recorded native timestep clock")
    if case.metadata.get("final_time") != float(times[-1]):
        raise ValueError("trajectory final time differs from the source case")
    expected_shape = (frames, cells, 4, 3)
    if positions.shape != expected_shape or velocities.shape != expected_shape or not np.isfinite(positions).all() or not np.isfinite(velocities).all():
        raise ValueError("trajectory corner positions and velocities must have finite shape (frames, cells, 4, 3)")
    if not np.issubdtype(arrays["original_tet_ids"].dtype, np.integer) or not np.array_equal(arrays["original_tet_ids"], np.arange(cells)):
        raise ValueError("trajectory changes original tetrahedron ordering")
    if cuts.shape != (frames, facets) or not np.issubdtype(cuts.dtype, np.integer) or not np.isin(cuts, [0, 1]).all():
        raise ValueError("fracture history must contain one binary integer per internal facet")
    if cuts[0].any() or (np.diff(cuts.astype(np.int64), axis=0) < 0).any() or not np.array_equal(cuts[-1], case.cut_labels):
        raise ValueError("fracture history must start intact, remain irreversible and match final labels")
    initial_nodes = rigid_pose(case.mesh.nodes, case.metadata["case"].get("initial_pose"))
    if not np.array_equal(positions[0], initial_nodes.astype(np.float32)[case.mesh.tets]):
        raise ValueError("initial corner positions differ from the source material mesh")
    if not np.array_equal(velocities[0], initial_velocity.astype(np.float32)[case.mesh.tets]):
        raise ValueError("initial corner velocities differ from the recorded loading case")
    edges = positions[:, :, 1:] - positions[:, :, :1]
    scale = np.max(np.abs(edges), axis=(-2, -1))
    if (scale <= 0).any() or (np.linalg.det(edges / scale[..., None, None]) <= 0).any():
        raise ValueError("trajectory contains an inverted or degenerate tetrahedron")
    return times, positions.reshape(frames, -1, 3), velocities.reshape(frames, -1, 3), cuts.astype(bool)


def _support_history(arrays, case, topology, positions, velocities, broken, fixed):
    offsets, connectivity = arrays["native_node_offsets"], arrays["native_corner_vertex_ids"]
    frames = len(positions)
    if offsets.shape != (frames + 1,) or not np.issubdtype(offsets.dtype, np.integer) or offsets[0] != 0 or (np.diff(offsets) <= 0).any():
        raise ValueError("native node offsets must delimit each recorded frame")
    if connectivity.shape != (frames, len(case.mesh.tets), 4) or not np.issubdtype(connectivity.dtype, np.integer):
        raise ValueError("native corner connectivity has invalid dimensions")
    total = int(offsets[-1])
    fields = {key.removeprefix("native_nodal_"): values for key, values in arrays.items() if key.startswith("native_nodal_")}
    for key in ("positions", "velocities", "rest_positions", "masses", "original_ids", "kinematic", "flags"):
        expected = (total, 3) if key in {"positions", "velocities", "rest_positions"} else (total,)
        if fields[key].shape != expected or not np.isfinite(fields[key]).all():
            raise ValueError(f"native nodal {key} has invalid shape or nonfinite values")
    if fields["kinematic"].dtype != np.dtype(bool) or not np.issubdtype(fields["flags"].dtype, np.integer) or (fields["flags"] < 0).any():
        raise ValueError("native supports require Boolean kinematic flags and integer vertex flags")
    if not np.array_equal(fields["kinematic"], (fields["flags"] & 1) != 0):
        raise ValueError("native support flags disagree")
    original = fields["original_ids"]
    if not np.issubdtype(original.dtype, np.integer) or (original < 0).any() or (original >= len(case.mesh.nodes)).any():
        raise ValueError("native original vertex IDs are outside the material mesh")
    if not np.array_equal(fields["rest_positions"], case.mesh.nodes.astype(np.float32)[original]):
        raise ValueError("native nodal rest positions differ from original vertex IDs")
    if (fields["kinematic"] & ~np.isin(original, fixed)).any():
        raise ValueError("native trajectory introduces an unsupported fixed vertex")
    result = np.empty((frames, len(topology.rest_positions)), dtype=bool)
    rest = case.mesh.nodes.astype(np.float32).astype(float)[case.mesh.tets]
    native_corner_mass = np.repeat(float(np.float32(case.metadata["density_kg_per_m3"])) * np.linalg.det(rest[:, 1:] - rest[:, :1]) / 24, 4)
    for frame in range(frames):
        selected = slice(offsets[frame], offsets[frame + 1])
        ids = connectivity[frame].reshape(-1)
        count = int(offsets[frame + 1] - offsets[frame])
        if (ids < 0).any() or (ids >= count).any() or len(np.unique(ids)) != count:
            raise ValueError("native nodal connectivity must reference every recorded vertex")
        if not np.array_equal(original[selected][ids], case.mesh.tets.ravel()):
            raise ValueError("native connectivity changes stable material-corner identity")
        for name, corners in (("positions", positions), ("velocities", velocities)):
            if not np.array_equal(fields[name][selected][ids], corners[frame]):
                raise ValueError(f"native nodal {name} disagree with material-corner state")
        pairs = ids[topology.face_pairs]
        if not np.array_equal(np.all(pairs[:, :, 0] == pairs[:, :, 1], axis=1), ~broken[frame]):
            raise ValueError("native topology disagrees with fracture history")
        groups = case.mesh.corner_groups(broken[frame])
        memberships = np.unique(np.column_stack((groups, ids)), axis=0)
        if len(memberships) != len(np.unique(groups)) or len(memberships) != count:
            raise ValueError("native corner connectivity cannot be represented by the recorded face-cut state")
        for values in (positions, velocities):
            intact_pairs = topology.face_pairs[~broken[frame]]
            if not np.array_equal(values[frame, intact_pairs[:, :, 0]], values[frame, intact_pairs[:, :, 1]]):
                raise ValueError("intact faces have inconsistent material-corner states")
        expected_mass = np.bincount(ids, weights=native_corner_mass, minlength=count)
        if (fields["masses"][selected] <= 0).any() or not np.allclose(fields["masses"][selected], expected_mass, rtol=64 * np.finfo(np.float32).eps, atol=0):
            raise ValueError("native nodal masses disagree with material rest-volume mass")
        result[frame] = fields["kinematic"][selected][ids]
    if not np.array_equal(result[0], np.isin(case.mesh.tets.ravel(), fixed)):
        raise ValueError("initial support flags differ from the loading specification")
    if (np.diff(result.astype(np.int8), axis=0) > 0).any():
        raise ValueError("unsupported support activation appears in the recorded history")
    if "corner_kinematic" in arrays:
        explicit = arrays["corner_kinematic"]
        if explicit.dtype != np.dtype(bool) or explicit.shape != (frames, len(case.mesh.tets), 4) or not np.array_equal(explicit.reshape(result.shape), result):
            raise ValueError("explicit corner support history disagrees with native nodal flags")
    if "support_release_labels" in arrays:
        labels = arrays["support_release_labels"]
        expected = result[:-1] & ~result[1:]
        if labels.dtype != np.dtype(bool) or labels.shape != (frames - 1, len(case.mesh.tets), 4) or not np.array_equal(labels.reshape(expected.shape), expected):
            raise ValueError("support release labels disagree with native fixed-to-free transitions")
    return result


def load_dynamics_trajectory(case_path, trajectory_path=None):
    case_path = Path(case_path).resolve(strict=True)
    trajectory_path = Path(trajectory_path or case_path.parent.parent / "trajectories" / case_path.name).resolve(strict=True)
    if case_path.parent.name != "cases" or trajectory_path.parent.name != "trajectories" or case_path.parent.parent != trajectory_path.parent.parent or case_path.stem != trajectory_path.stem:
        raise ValueError("dynamics case and trajectory must have matching IDs in the same generated dataset")
    mesh_path = case_path.parent.parent / "mesh.npz"
    before = {str(path): file_hash(path) for path in (case_path, trajectory_path, mesh_path)}
    case = load_prepared_case(case_path)
    context, material, gravity, fixed = _context(case)
    if case.metadata.get("case", {}).get("id") != case_path.stem:
        raise ValueError("loading case ID differs from its filename")
    topology = material_topology(case.mesh, material["rest_density"])
    initial_velocity, impulses, condition = _loading(case, topology, fixed)
    with np.load(trajectory_path, allow_pickle=False) as archive:
        arrays = {key: np.array(archive[key], copy=True) for key in archive.files}
    if "metadata_json" not in arrays:
        raise ValueError("dynamics trajectory requires embedded source provenance")
    provenance = json.loads(str(arrays["metadata_json"].item()))
    expected = {"schema": "fracture-world.femfx-trajectory.v1", "source_case_sha256": before[str(case_path)],
                "mesh_sha256": topology.fingerprint, "conditioning_context_sha256": case.metadata["conditioning_context_sha256"],
                "native_revision": context.get("native_revision"), "native_topology_schema": "fracture-world.femfx-nodal-topology.v1"}
    if not isinstance(provenance, dict) or any(provenance.get(key) != value for key, value in expected.items()):
        raise ValueError("dynamics trajectory provenance differs from its loading case")
    times, positions, velocities, broken = _validate_history(arrays, case, topology, context, fixed, initial_velocity)
    support = _support_history(arrays, case, topology, positions, velocities, broken, fixed)
    for path, checksum in before.items():
        if file_hash(path) != checksum:
            raise RuntimeError("dynamics source changed during archive validation")
    released = int(np.count_nonzero(support[:-1] & ~support[1:]))
    metadata = {"source": "recorded_FEMFX_numerical_trajectory", "units": "SI", "source_hashes": before,
                "conditioning_context": context, "stable_indexing": "original_tetrahedron_then_local_corner",
                "fracture_semantics": "irreversible_original_internal_facet_cuts", "observed_support_releases": released,
                "support_release_requires_explicit_runtime_policy": bool(released), "native_runtime_required": False,
                "split_unit": "complete_loading_condition", "parameter_status": case.metadata.get("parameter_status")}
    return DynamicsTrajectory(topology, _frozen(positions), _frozen(velocities), _frozen(broken), _frozen(support),
                              _frozen(times), _frozen(gravity), dict(material), _frozen(impulses), case_path.stem, condition, metadata)


@lru_cache(maxsize=1)
def _streamed_case(path, source_hashes):
    item = load_dynamics_trajectory(path)
    if tuple(sorted(item.metadata["source_hashes"].items())) != source_hashes:
        raise RuntimeError("streamed source changed after dataset indexing")
    return item


def supervision_counts(item):
    if isinstance(item, StreamedDynamicsTrajectory):
        return item.event_counts
    return dict(fracture_positive=int(np.count_nonzero(~item.broken[:-1] & item.broken[1:])),
                fracture_negative=int(np.count_nonzero(~item.broken[:-1] & ~item.broken[1:])),
                support_positive=int(np.count_nonzero(item.fixed[:-1] & ~item.fixed[1:])),
                support_negative=int(np.count_nonzero(item.fixed[:-1] & item.fixed[1:])))


@dataclass(frozen=True)
class StreamedDynamicsTrajectory:
    case_path: Path
    topology: MaterialTopology
    case_id: str
    condition_digest: str
    metadata: dict
    transitions: int
    velocity_rms: float
    event_counts: dict
    times: np.ndarray

    @property
    def shape_family(self):
        context = self.metadata.get("conditioning_context", {})
        return self.metadata.get("shape_family", context.get("shape_family", self.topology.fingerprint))

    def __len__(self):
        return self.transitions

    def _load(self):
        return _streamed_case(self.case_path, tuple(sorted(self.metadata["source_hashes"].items())))

    def __getattr__(self, name):
        if name == "material":
            return self.metadata["conditioning_context"]["material"]
        if name == "gravity":
            return np.asarray(self.metadata["conditioning_context"]["gravity"], dtype=float)
        if name in {"positions", "velocities", "broken", "fixed", "initial_impulses"}:
            return getattr(self._load(), name)
        raise AttributeError(name)

    def transition(self, frame):
        return self._load().transition(frame)


def load_dynamics_dataset(directory, *, stream=False, progress=None):
    directory = Path(directory).resolve()
    if directory.is_file():
        specification = json.loads(directory.read_text(encoding="utf-8"))
        if not isinstance(specification, dict) or specification.get("schema") != "fracture-world.dynamics-collection.v1":
            raise ValueError("unsupported dynamics collection schema")
        entries = specification.get("datasets")
        if not isinstance(entries, list) or not entries:
            raise ValueError("a dynamics collection requires a nonempty datasets list")
        trajectories, seen = [], set()
        for entry in entries:
            if not isinstance(entry, dict) or set(entry) - {"path", "shape_family", "split"}:
                raise ValueError("collection entries require path, shape_family and optional split")
            if not isinstance(entry.get("path"), str) or not isinstance(entry.get("shape_family"), str) or not entry["shape_family"].strip():
                raise ValueError("collection paths and shape families must be nonempty strings")
            source = (directory.parent / entry["path"]).resolve()
            if source in seen or not source.is_dir():
                raise ValueError("collection entries must select distinct generated dataset directories")
            seen.add(source)
            if "split" in entry and entry["split"] not in {"train", "validation", "test"}:
                raise ValueError("collection split must select train, validation or test")
            for item in load_dynamics_dataset(source, stream=stream, progress=progress):
                context = item.metadata.get("conditioning_context", {})
                if "shape_family" in context and context["shape_family"] != entry["shape_family"]:
                    raise ValueError("collection shape family differs from the recorded geometry source")
                metadata = {**item.metadata, "shape_family": entry["shape_family"], "source_dataset": str(source)}
                if "split" in entry:
                    metadata["dataset_split"] = entry["split"]
                trajectories.append(replace(item, metadata=metadata))
        validate_dynamics_compatibility(trajectories, allow_mixed_meshes=True)
        return trajectories
    paths = sorted((directory / "cases").glob("*.npz"))
    if not paths:
        raise ValueError("no native dynamics cases were found")
    trajectories, topologies = [], {}
    for index, path in enumerate(paths, start=1):
        if progress is not None:
            progress(dict(phase="dataset_loading", status="case_started", dataset=str(directory),
                          case=path.stem, case_index=index, cases=len(paths)))
        item = load_dynamics_trajectory(path)
        topology_key = (item.topology.fingerprint, item.material["rest_density"])
        topology = topologies.setdefault(topology_key, item.topology)
        if stream:
            trajectories.append(StreamedDynamicsTrajectory(path.resolve(), topology, item.case_id, item.condition_digest,
                                                           item.metadata, len(item), float(np.sqrt(np.mean(item.velocities**2))),
                                                           supervision_counts(item), item.times))
        else:
            trajectories.append(replace(item, topology=topology))
    validate_dynamics_compatibility(trajectories)
    if progress is not None:
        progress(dict(phase="dataset_loading", status="completed", dataset=str(directory), cases=len(paths)))
    return trajectories


def validate_dynamics_compatibility(trajectories, *, allow_mixed_meshes=False):
    trajectories = list(trajectories)
    if not trajectories:
        raise ValueError("dynamics dataset must contain at least one loading case")
    reference = trajectories[0]
    if len({item.case_id for item in trajectories}) != len(trajectories):
        raise ValueError("dynamics dataset contains duplicate case IDs")
    if not allow_mixed_meshes and any(item.topology.fingerprint != reference.topology.fingerprint for item in trajectories):
        raise ValueError("the first dynamics model requires one compatible material mesh")
    families = {}
    for item in trajectories:
        if not isinstance(item.shape_family, str) or not item.shape_family.strip():
            raise ValueError("shape families must be nonempty strings")
        previous = families.setdefault(item.topology.fingerprint, item.shape_family)
        if previous != item.shape_family:
            raise ValueError("the same material mesh cannot belong to different shape families")
    return reference.topology


def split_dynamics_cases(trajectories, seed=17, overfit=False, event_stratified=False, split_unit="loading_condition"):
    trajectories = list(trajectories)
    validate_dynamics_compatibility(trajectories, allow_mixed_meshes=True)
    if split_unit not in {"loading_condition", "shape_family"}:
        raise ValueError("split_unit must select loading_condition or shape_family")
    if overfit:
        return {"train": sorted(trajectories, key=lambda item: item.case_id), "validation": [], "test": []}
    assignments = [item.metadata.get("dataset_split") for item in trajectories]
    if any(value is not None for value in assignments):
        if split_unit != "shape_family" or any(value not in {"train", "validation", "test"} for value in assignments):
            raise ValueError("explicit collection splits require shape_family splitting and an assignment for every dataset")
        family_splits = {}
        for item, assignment in zip(trajectories, assignments, strict=True):
            if family_splits.setdefault(item.shape_family, assignment) != assignment:
                raise ValueError("shape-family leakage across explicit dataset splits")
        if set(family_splits.values()) != {"train", "validation", "test"}:
            raise ValueError("explicit collections require nonempty train, validation and test shape families")
        return {name: sorted((item for item in trajectories if item.metadata["dataset_split"] == name), key=lambda item: item.case_id)
                for name in ("train", "validation", "test")}
    groups = {}
    for item in trajectories:
        key = item.shape_family if split_unit == "shape_family" else item.condition_digest
        groups.setdefault(key, []).append(item)
    if len(groups) < 3:
        unit = "shape families" if split_unit == "shape_family" else "loading conditions"
        raise ValueError(f"held-out dynamics training requires at least three distinct {unit}; overfit enables debugging")
    strata = {}
    for condition, rows in groups.items():
        counts = [supervision_counts(item) for item in rows] if event_stratified else []
        stratum = (
            any(item["fracture_positive"] for item in counts),
            any(item["support_positive"] for item in counts),
        ) if event_stratified else None
        strata.setdefault(stratum, []).append(condition)
    selections = dict(validation=[], test=[], train=[])
    for keys in strata.values():
        ordered = sorted(keys, key=lambda key: hashlib.sha256(f"{seed}:{key}".encode()).hexdigest())
        if len(ordered) == 1:
            selections["train"].extend(ordered)
        elif len(ordered) == 2:
            selections["train"].append(ordered[0])
            selections["validation"].append(ordered[1])
        else:
            count = max(1, round(len(ordered) * 0.15))
            selections["validation"].extend(ordered[:count])
            selections["test"].extend(ordered[count:count * 2])
            selections["train"].extend(ordered[count * 2:])
    return {name: sorted((item for key in keys for item in groups[key]), key=lambda item: item.case_id)
            for name, keys in selections.items()}
