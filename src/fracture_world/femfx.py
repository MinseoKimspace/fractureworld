import hashlib
import importlib.util
import json
from pathlib import Path
import re
import shutil
import subprocess
import sys

import numpy as np

from .data import _mesh_digest
from .io import file_hash, write_json
from .mesh import VolumeMesh, rigid_pose
from .partition import component_labels


REVISION = "95cf656731a2465ae1d400138170af2d6575b5bb"
REPOSITORY = "https://github.com/GPUOpen-Effects/FEMFX.git"
SCHEMA = "fracture-world.femfx.v1"
OBSERVATION_SCHEMA = "fracture-world.femfx-observation.v1"
OBSERVATION_VECTOR_FIELDS = ("pre_contact_velocities", "applied_delta_velocities", "stabilization_delta_positions")
OBSERVATION_CORNER_FIELDS = ("contact_observed", "constraint_solve_active")
OBSERVATION_TET_FIELDS = ("fracture_stress_eigenvalues", "fracture_stress_thresholds", "fracture_stress_observed")
OBSERVATION_FIELDS = (*OBSERVATION_VECTOR_FIELDS, *OBSERVATION_CORNER_FIELDS, *OBSERVATION_TET_FIELDS)
SOLVER_OBSERVATION_SCHEMA = "fracture-world.femfx-mpcg-observation.v1"
SOLVER_VECTOR_FIELDS = ("mpcg_solution", "mpcg_rhs", "mpcg_residual")
SOLVER_CORNER_FIELDS = ("mpcg_observed", "mpcg_kinematic", "mpcg_masses")
SOLVER_TET_FIELDS = ("mpcg_recursive_norm_squared", "mpcg_true_norm_squared", "mpcg_rhs_norm_squared",
                     "mpcg_tolerance_squared", "mpcg_epsilon", "mpcg_iterations", "mpcg_max_iterations",
                     "mpcg_rhs_positive", "mpcg_stopping_observed")
SOLVER_BOOLEAN_FIELDS = ("mpcg_observed", "mpcg_kinematic", "mpcg_rhs_positive", "mpcg_stopping_observed")
SOLVER_OBSERVATION_FIELDS = (*SOLVER_VECTOR_FIELDS, *SOLVER_CORNER_FIELDS, *SOLVER_TET_FIELDS)
ELASTIC_STATE_SCHEMA = "fracture-world.femfx-elastic-state.v1"
ELASTIC_STATE_FIELDS = ("cached_tet_rotations", "cached_tet_centers_of_mass", "mesh_tet_ids")


def native_directory():
    return Path.cwd() / ".cache" / "femfx-build"


def binding_digest():
    digest = hashlib.sha256()
    for path in sorted(Path(__file__).with_name("native").rglob("*")):
        if path.is_file():
            digest.update(path.name.encode())
            digest.update(path.read_bytes())
    return digest.hexdigest()


def build_native(source_dir=None, build_dir=None, jobs=None):
    if sys.platform != "win32":
        raise RuntimeError("the pinned FEMFX source requires Windows; generate references in Windows and train on the portable data in WSL")
    try:
        import pybind11
        import cmake
    except ImportError as error:
        raise RuntimeError("install native build dependencies with uv sync --extra native") from error
    source_dir = Path(source_dir or Path.cwd() / ".cache" / "FEMFX").resolve()
    build_dir = Path(build_dir or native_directory()).resolve()
    new_checkout = not source_dir.exists()
    if new_checkout:
        source_dir.parent.mkdir(parents=True, exist_ok=True)
        subprocess.run(["git", "clone", "--no-checkout", REPOSITORY, str(source_dir)], check=True)
    git = ["git", "-c", f"safe.directory={source_dir.as_posix()}", "-C", str(source_dir)]
    if new_checkout:
        subprocess.run([*git, "checkout", "--detach", REVISION], check=True)
    revision = subprocess.check_output([*git, "rev-parse", "HEAD"], text=True).strip()
    if revision != REVISION:
        raise ValueError(f"FEMFX source must be at {REVISION}; refusing a different numerical reference")
    modifications = subprocess.check_output([*git, "status", "--porcelain", "--untracked-files=no"], text=True)
    if modifications.strip():
        raise ValueError("FEMFX reference source has unrecorded changes")
    executable = str(Path(cmake.CMAKE_BIN_DIR) / "cmake.exe")
    binding_before = binding_digest()
    subprocess.run([
        executable, "-S", str(Path(__file__).with_name("native")), "-B", str(build_dir),
        f"-DFEMFX_SOURCE_DIR={source_dir}", f"-DPython_EXECUTABLE={sys.executable}",
        f"-Dpybind11_DIR={pybind11.get_cmake_dir()}",
    ], check=True)
    command = [executable, "--build", str(build_dir), "--config", "Release", "--parallel"]
    if jobs is not None:
        if not isinstance(jobs, int) or jobs < 1:
            raise ValueError("build jobs must be positive")
        command.append(str(jobs))
    subprocess.run(command, check=True)
    if binding_digest() != binding_before:
        raise RuntimeError("native binding changed during compilation; rebuild before recording a reference manifest")
    modules = list(build_dir.rglob("_femfx*.pyd"))
    if len(modules) != 1:
        raise RuntimeError("native build did not produce exactly one FEMFX Python module")
    module = modules[0]
    observation = json.loads((build_dir / "observed-source" / "manifest.json").read_text(encoding="utf-8"))
    if (observation.get("schema") != OBSERVATION_SCHEMA or
            observation.get("solver_observation_schema") != SOLVER_OBSERVATION_SCHEMA):
        raise RuntimeError("native build is missing its observation instrumentation manifest")
    manifest = {
        "schema": SCHEMA,
        "repository": REPOSITORY,
        "revision": revision,
        "binding_sha256": binding_digest(),
        "module_sha256": file_hash(module),
        "module": module.relative_to(build_dir).as_posix(),
        "python": sys.version,
        "platform": sys.platform,
        "reference_source_modified": True,
        "upstream_checkout_modified": False,
        "compiled_source_instrumented": True,
        "physics_arithmetic_modified": False,
        "observation": observation,
    }
    write_json(build_dir / "manifest.json", manifest)
    shutil.copyfile(source_dir / "LICENSE.txt", build_dir / "FEMFX-LICENSE.txt")
    return manifest


def _observation_arrays(state, tetrahedra, required=False):
    present = set(state).intersection(OBSERVATION_FIELDS)
    if not present and not required:
        return {}
    if present != set(OBSERVATION_FIELDS):
        raise RuntimeError("native observations require the complete capture fields")
    arrays = {}
    for name in OBSERVATION_FIELDS:
        values = np.asarray(state[name])
        expected = (tetrahedra, 4, 3) if name in OBSERVATION_VECTOR_FIELDS else (tetrahedra, 4) if name in OBSERVATION_CORNER_FIELDS else (tetrahedra,)
        boolean = name in OBSERVATION_CORNER_FIELDS or name == "fracture_stress_observed"
        valid_type = values.dtype == np.dtype(bool) if boolean else np.issubdtype(values.dtype, np.floating)
        if values.shape != expected or not valid_type or not np.isfinite(values).all():
            raise RuntimeError(f"native observation {name} has invalid dimensions, values or mask type")
        arrays[name] = np.array(values, copy=True)
    observed = arrays["fracture_stress_observed"]
    if (arrays["fracture_stress_eigenvalues"][observed] < 0).any() or (arrays["fracture_stress_thresholds"][observed] <= 0).any():
        raise RuntimeError("observed native fracture stresses require nonnegative eigenvalues and positive thresholds")
    if (arrays["constraint_solve_active"] & ~arrays["contact_observed"]).any():
        raise RuntimeError("active native constraint solves require observed applied corrections")
    if state.get("step") == 0 and (arrays["contact_observed"].any() or observed.any()):
        raise RuntimeError("initial native state cannot claim a preceding observed update")
    return arrays


def _solver_observation_arrays(state, tetrahedra, required=False, *, frames=None):
    present = set(state).intersection(SOLVER_OBSERVATION_FIELDS)
    if not present and not required:
        return {}
    if present != set(SOLVER_OBSERVATION_FIELDS):
        raise RuntimeError("native solver observations require the complete capture fields")
    prefix = () if frames is None else (frames,)
    arrays = {}
    for name in SOLVER_OBSERVATION_FIELDS:
        shape = (*prefix, tetrahedra, 4, 3) if name in SOLVER_VECTOR_FIELDS else (
            (*prefix, tetrahedra, 4) if name in SOLVER_CORNER_FIELDS else (*prefix, tetrahedra))
        values = np.asarray(state[name])
        valid_type = values.dtype == np.dtype(bool) if name in SOLVER_BOOLEAN_FIELDS else np.issubdtype(values.dtype, np.floating)
        if values.shape != shape or not valid_type or not np.isfinite(values).all():
            raise RuntimeError(f"native solver observation {name} has invalid dimensions, values or mask type")
        arrays[name] = np.array(values, copy=True)
    observed = arrays["mpcg_observed"]
    stopping = arrays["mpcg_stopping_observed"]
    if (arrays["mpcg_masses"][observed] <= 0).any() or (arrays["mpcg_kinematic"] & ~observed).any():
        raise RuntimeError("native solver observations require positive observed masses and available kinematic flags")
    if np.any(stopping & ~observed.all(axis=-1)) or np.any(arrays["mpcg_rhs_positive"] & ~stopping):
        raise RuntimeError("native solver stopping observations require complete observed tetrahedron corners")
    for name in ("mpcg_recursive_norm_squared", "mpcg_true_norm_squared", "mpcg_rhs_norm_squared", "mpcg_tolerance_squared"):
        if (arrays[name][stopping] < 0).any():
            raise RuntimeError("native solver squared norms must be nonnegative where observed")
    epsilon, iterations, maximum = (arrays[name][stopping] for name in ("mpcg_epsilon", "mpcg_iterations", "mpcg_max_iterations"))
    if ((epsilon <= 0).any() or (maximum < 1).any() or (iterations < 0).any() or (iterations > maximum).any()
            or np.any(iterations != np.floor(iterations)) or np.any(maximum != np.floor(maximum))):
        raise RuntimeError("native solver observations contain invalid stopping settings or iteration counts")
    if np.any(arrays["mpcg_rhs_positive"][stopping] != (arrays["mpcg_rhs_norm_squared"][stopping] > 0)):
        raise RuntimeError("native solver RHS availability disagrees with its observed squared norm")
    if np.any(stopping & ~arrays["mpcg_rhs_positive"] & (arrays["mpcg_iterations"] != 0)):
        raise RuntimeError("native solver zero-RHS early return cannot perform iterations")
    for name, values in arrays.items():
        if name in SOLVER_BOOLEAN_FIELDS or name in {"mpcg_iterations", "mpcg_max_iterations"}:
            continue
        mask = observed if name in (*SOLVER_VECTOR_FIELDS, "mpcg_masses") else stopping
        selected = values[mask]
        if not np.array_equal(selected, selected.astype(np.float32).astype(values.dtype)):
            raise RuntimeError("native solver observations must retain captured float32 arithmetic exactly")
    if state.get("step") == 0 and (observed.any() or stopping.any()):
        raise RuntimeError("initial native state cannot claim a preceding observed solver update")
    return arrays


def load_native(build_dir=None):
    build_dir = Path(build_dir or native_directory()).resolve()
    path = build_dir / "manifest.json"
    if not path.exists():
        raise RuntimeError("FEMFX is not built; run fracture-world femfx-build in Windows first")
    manifest = json.loads(path.read_text(encoding="utf-8"))
    if manifest["revision"] != REVISION or manifest["binding_sha256"] != binding_digest():
        raise RuntimeError("FEMFX binding changed; rebuild before generating numerical references")
    module_path = (build_dir / manifest["module"]).resolve()
    if not module_path.is_relative_to(build_dir) or file_hash(module_path) != manifest["module_sha256"]:
        raise ValueError("FEMFX native binary differs from its build manifest")
    if "_femfx" in sys.modules:
        module = sys.modules["_femfx"]
        if Path(module.__file__).resolve() != module_path:
            raise RuntimeError("another FEMFX native binary is already loaded in this process")
        return module, manifest
    specification = importlib.util.spec_from_file_location("_femfx", module_path)
    if specification is None or specification.loader is None:
        raise RuntimeError("FEMFX native binary cannot be loaded by this Python interpreter")
    module = importlib.util.module_from_spec(specification)
    specification.loader.exec_module(module)
    sys.modules["_femfx"] = module
    return module, manifest


def context_digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()).hexdigest()


def validate_specification(mesh, specification):
    if specification.get("schema") != SCHEMA:
        raise ValueError(f"case specification must use schema {SCHEMA}")
    material = specification["material"]
    required = ["rest_density", "youngs_modulus", "poissons_ratio", "fracture_stress_threshold"]
    if set(material) != set(required):
        raise ValueError(f"the first brittle-material pilot requires exactly these material keys: {required}")
    if not np.isfinite([material[name] for name in required]).all():
        raise ValueError("material values must be finite")
    if min(material["rest_density"], material["youngs_modulus"], material["fracture_stress_threshold"]) <= 0 or not -1 < material["poissons_ratio"] < 0.5:
        raise ValueError("invalid density, elastic constants or fracture stress threshold")
    if specification.get("parameter_status") not in {"uncalibrated_trial", "externally_calibrated"}:
        raise ValueError("parameter_status must distinguish trial parameters from externally calibrated parameters")
    if specification["parameter_status"] == "externally_calibrated" and not specification.get("calibration_source"):
        raise ValueError("calibrated parameters require a calibration_source")
    dt, steps = specification["dt"], specification["steps"]
    if not np.isfinite(dt) or dt <= 0 or not isinstance(steps, int) or isinstance(steps, bool) or steps < 1:
        raise ValueError("dt and integer steps must be positive")
    for name in ["threads", "record_stride"]:
        value = specification.get(name, 1)
        if not isinstance(value, int) or isinstance(value, bool) or value < 1:
            raise ValueError(f"{name} must be a positive integer")
    gravity = np.asarray(specification["gravity"], dtype=float)
    if gravity.shape != (3,) or not np.isfinite(gravity).all():
        raise ValueError("gravity must be a finite three-vector")
    fixed = np.asarray(specification["fixed_vertices"])
    if fixed.ndim != 1 or (fixed.size and not np.issubdtype(fixed.dtype, np.integer)):
        raise ValueError("fixed_vertices must contain integer vertex IDs")
    fixed = fixed.astype(np.int64)
    if (fixed < 0).any() or (fixed >= len(mesh.nodes)).any() or len(np.unique(fixed)) != len(fixed):
        raise ValueError("fixed vertex IDs must be unique and inside the mesh")
    cases = specification["cases"]
    if not isinstance(cases, list) or not cases:
        raise ValueError("at least one loading case is required")
    identifiers = []
    for case in cases:
        identifier = case["id"]
        if not isinstance(identifier, str) or not re.fullmatch(r"[a-zA-Z0-9][a-zA-Z0-9_-]*", identifier):
            raise ValueError("case IDs must contain only letters, numbers, underscores and hyphens")
        identifiers.append(identifier)
        impulses = case["vertex_impulses"]
        if not isinstance(impulses, list) or (not impulses and specification.get("loading") != "gravity_driven_drop"):
            raise ValueError("each loading case needs explicit vertex impulses")
        rigid_pose(mesh.nodes, case.get("initial_pose"))
        initial_velocity = np.asarray(case.get("initial_linear_velocity", [0.0, 0.0, 0.0]), dtype=float)
        if initial_velocity.shape != (3,) or not np.isfinite(initial_velocity).all():
            raise ValueError("initial_linear_velocity must be a finite three-vector")
        if np.any(initial_velocity) and fixed.size:
            raise ValueError("rigid initial velocity cannot move prescribed fixed vertices")
        for impulse in impulses:
            vertex = impulse["vertex"]
            vector = np.asarray(impulse["impulse"], dtype=float)
            if not isinstance(vertex, int) or isinstance(vertex, bool) or not 0 <= vertex < len(mesh.nodes):
                raise ValueError("impulse vertex is outside the reference mesh")
            if vertex in fixed:
                raise ValueError("cannot apply an instantaneous velocity impulse to a fixed vertex")
            if vector.shape != (3,) or not np.isfinite(vector).all() or np.linalg.norm(vector) == 0:
                raise ValueError("impulses must be finite, nonzero three-vectors in newton seconds")
    if len(set(identifiers)) != len(identifiers):
        raise ValueError("case IDs must be unique")
    if specification.get("loading", "instantaneous_vertex_impulses") not in {"instantaneous_vertex_impulses", "gravity_driven_drop"}:
        raise ValueError("unsupported loading mode")
    if specification.get("loading") == "gravity_driven_drop":
        floor = specification.get("floor_height_m")
        bounds = specification.get("scene_options", {}).get("collision_bounds", [])
        if fixed.size or gravity[1] >= 0 or gravity[0] != 0 or gravity[2] != 0:
            raise ValueError("free-fall datasets require no fixed vertices and downward Y gravity")
        if not isinstance(floor, (int, float)) or isinstance(floor, bool) or not np.isfinite(floor):
            raise ValueError("drop datasets require an explicit finite floor_height_m")
        if len(bounds) != 6 or bounds[2] != floor:
            raise ValueError("drop floor must match the native minimum-Y collision plane")
        for case in cases:
            if case["vertex_impulses"] or "initial_pose" not in case:
                raise ValueError("gravity-driven drops require an explicit initial pose and empty initial impulses")
            if rigid_pose(mesh.nodes, case["initial_pose"])[:, 1].min() < floor:
                raise ValueError("drop initial poses must not penetrate the floor")
    return material, fixed


def initial_conditions(mesh, material, fixed, case):
    masses = np.zeros(len(mesh.nodes))
    np.add.at(masses, mesh.tets.ravel(), np.repeat(mesh.volumes * material["rest_density"] / 4, 4))
    if (masses <= 0).any():
        raise ValueError("all original vertices must belong to a tetrahedron")
    momentum = np.zeros_like(mesh.nodes)
    for impulse in case["vertex_impulses"]:
        momentum[impulse["vertex"]] += impulse["impulse"]
    active = np.flatnonzero(np.linalg.norm(momentum, axis=1) > 0)
    if not len(active) and case["vertex_impulses"]:
        raise ValueError("vertex impulses cancel exactly; provide a nonzero loading case")
    magnitudes = np.linalg.norm(momentum[active], axis=1)
    impacts = np.column_stack((mesh.nodes[active], momentum[active] / magnitudes[:, None], magnitudes))
    kinematic = np.zeros(len(mesh.nodes), dtype=bool)
    kinematic[fixed] = True
    return momentum / masses[:, None] + np.asarray(case.get("initial_linear_velocity", [0.0, 0.0, 0.0])), kinematic, impacts


def native_face_order(mesh, state):
    native_faces = np.asarray(state["face_vertices"], dtype=np.int64)
    expected = np.sort(mesh.faces.vertices[mesh.internal_faces], axis=1)
    if native_faces.shape != expected.shape:
        raise ValueError("FEMFX reference did not expose every original internal face")
    lookup = {tuple(face): index for index, face in enumerate(np.sort(native_faces, axis=1))}
    if len(lookup) != len(expected) or any(tuple(face) not in lookup for face in expected):
        raise ValueError("FEMFX internal faces do not match the original material mesh")
    return np.asarray([lookup[tuple(face)] for face in expected], dtype=np.int64)


def validate_state(mesh, state, previous_cuts, order):
    count = len(mesh.tets)
    if not np.array_equal(state["original_tet_ids"], np.arange(count)):
        raise ValueError("FEMFX states must preserve original material element ordering")
    positions = np.asarray(state["corner_positions"])
    velocities = np.asarray(state["corner_velocities"])
    if positions.shape != (count, 4, 3) or velocities.shape != positions.shape:
        raise ValueError("FEMFX must return one position and velocity for each original material corner")
    if not np.isfinite(positions).all() or not np.isfinite(velocities).all():
        raise RuntimeError("FEMFX reference produced nonfinite material states")
    if (np.linalg.det(positions[:, 1:] - positions[:, :1]) <= 0).any():
        raise RuntimeError("FEMFX reference contains inverted or collapsed elements")
    cuts = np.asarray(state["cut_labels"])[order]
    if not np.isin(cuts, [0, 1]).all() or cuts.shape != previous_cuts.shape or (previous_cuts > cuts).any():
        raise RuntimeError("FEMFX fracture labels must be binary and irreversible")
    if state.get("warnings", 0):
        raise RuntimeError(f"FEMFX reported a warning; the trajectory is not accepted as a reference: {state['warnings']}")
    validate_piece_partition(mesh, state, cuts)
    return cuts.astype(np.int8)


def validate_piece_partition(mesh, state, cuts):
    labels = component_labels(len(mesh.tets), mesh.dual_edges, cuts)
    native = np.asarray(state["tet_piece_ids"])
    vertices = np.asarray(state["corner_vertex_ids"])
    if native.shape != (len(mesh.tets),) or vertices.shape != mesh.tets.shape:
        raise ValueError("native piece and vertex connectivity dimensions differ from material tetrahedra")
    owners = np.repeat(np.arange(len(mesh.tets)), 4)
    ownership = np.unique(np.column_stack((vertices.ravel(), native[owners])), axis=0)
    groups = mesh.corner_groups(cuts)
    pairs = np.unique(np.column_stack((groups, vertices.ravel())), axis=0)
    diagnostic = dict(face_components=len(np.unique(labels)), material_corner_groups=len(np.unique(groups)),
                      native_vertices=len(np.unique(vertices)), native_mesh_containers=len(np.unique(native)))
    if len(ownership) != len(np.unique(vertices)):
        raise RuntimeError("native pieces disagree with current vertex connectivity: " + json.dumps(diagnostic))
    if len(pairs) != len(np.unique(groups)):
        raise RuntimeError("native vertices disagree with intact-face welding: " + json.dumps(diagnostic))
    if len(pairs) != len(np.unique(vertices)):
        raise RuntimeError("unsupported vertex-only coupling between face-separated components: " + json.dumps(diagnostic))
    return labels


def generate_dataset(mesh_path, specification_path, output_dir, length_scale, build_dir=None, export_replay=True, skip_invalid=False):
    if not np.isfinite(length_scale) or length_scale <= 0:
        raise ValueError("length scale must be positive metres per source coordinate unit")
    mesh_source = VolumeMesh.load(mesh_path)
    mesh = VolumeMesh(mesh_source.nodes * length_scale, mesh_source.tets)
    specification = json.loads(Path(specification_path).read_text(encoding="utf-8"))
    material, fixed = validate_specification(mesh, specification)
    output_dir = Path(output_dir)
    if output_dir.exists() and any(output_dir.iterdir()):
        raise ValueError("numerical dataset output must be empty; use a new directory")
    native, build = load_native(build_dir)
    capabilities = native.capabilities()
    capture_observations = capabilities.get("constraint_delta_observation", False)
    if capture_observations and capabilities.get("observation_schema") != OBSERVATION_SCHEMA:
        raise RuntimeError("native observation capability has an unsupported schema")
    capture_solver = capabilities.get("elastic_solver_observation", False)
    if capture_solver and (not capture_observations or capabilities.get("solver_observation_schema") != SOLVER_OBSERVATION_SCHEMA):
        raise RuntimeError("native solver observation capability has an unsupported schema or missing update observations")
    capture_elastic_state = capabilities.get("current_elastic_state", False)
    context = {
        "mesh_si_fingerprint": _mesh_digest(mesh),
        "material": material,
        "fixed_vertices": sorted(fixed.tolist()),
        "dt": specification["dt"],
        "steps": specification["steps"],
        "gravity": specification["gravity"],
        "loading": specification.get("loading", "instantaneous_vertex_impulses"),
        "scene_options": specification.get("scene_options", {}),
        "threads": specification.get("threads", 1),
        "native_revision": REVISION,
        "binding_sha256": build["binding_sha256"],
    }
    for key in ("shape_id", "shape_family", "geometry_source", "floor_height_m"):
        if key in specification:
            context[key] = specification[key]
    context_hash = context_digest(context)
    output_dir.mkdir(parents=True, exist_ok=True)
    mesh.save(output_dir / "mesh.npz")
    write_json(output_dir / "generation.json", {
        "schema": SCHEMA, "specification": specification, "build": build,
        "source_mesh_sha256": file_hash(mesh_path), "metres_per_source_unit": length_scale,
        "conditioning_context": context, "conditioning_context_sha256": context_hash,
        "capabilities": capabilities,
    })
    results, rejected = [], []
    nodal_fields = ["nodal_rest_positions", "nodal_positions", "nodal_velocities", "nodal_masses",
                    "nodal_kinematic", "nodal_flags", "nodal_original_ids", "nodal_piece_ids"]
    recorded_fields = ["time", "corner_positions", "corner_velocities", "corner_kinematic", "tet_piece_ids", "warnings",
                       "corner_vertex_ids", *nodal_fields]
    if capture_elastic_state:
        recorded_fields.extend(ELASTIC_STATE_FIELDS)
    for case in specification["cases"]:
        velocities, kinematic, impacts = initial_conditions(mesh, material, fixed, case)
        scene_arguments = (mesh.nodes, mesh.tets, material, velocities, kinematic,
                           specification["dt"], specification["gravity"],
                           specification.get("threads", 1), specification.get("scene_options", {}))
        scene = None
        try:
            if "initial_pose" in case:
                scene = native.Scene(*scene_arguments, deformed_positions=rigid_pose(mesh.nodes, case["initial_pose"]))
            else:
                scene = native.Scene(*scene_arguments)
            initial = scene.state()
            observations = _observation_arrays(initial, len(mesh.tets), capture_observations)
            solver_observations = _solver_observation_arrays(initial, len(mesh.tets), capture_solver)
            observation_fields = (*observations, *solver_observations)
            if observation_fields and not capture_observations:
                raise RuntimeError("native observations require an advertised instrumented capture capability")
            initial.update(observations)
            initial.update(solver_observations)
            case_recorded_fields = [*recorded_fields, *observation_fields]
            order = native_face_order(mesh, initial)
            previous = np.zeros(len(mesh.internal_faces), dtype=np.int8)
            states = [{key: initial[key] for key in case_recorded_fields}]
            recorded_steps = [0]
            previous_support = np.asarray(initial["corner_kinematic"], dtype=bool)
            cuts = validate_state(mesh, initial, previous, order)
            cut_history = [cuts]
            for step in range(specification["steps"]):
                state = scene.step()
                state.update(_observation_arrays(state, len(mesh.tets), capture_observations))
                state.update(_solver_observation_arrays(state, len(mesh.tets), capture_solver))
                previous_cuts = cuts
                cuts = validate_state(mesh, state, previous_cuts, order)
                current_support = np.asarray(state["corner_kinematic"], dtype=bool)
                if current_support.shape != mesh.tets.shape or (current_support & ~previous_support).any():
                    raise RuntimeError("native support flags have invalid dimensions or activate a new prescribed support")
                support_changed = not np.array_equal(current_support, previous_support)
                if (step + 1) % specification.get("record_stride", 1) == 0 or step + 1 == specification["steps"] or not np.array_equal(cuts, previous_cuts) or support_changed:
                    states.append({key: state[key] for key in case_recorded_fields})
                    cut_history.append(cuts)
                    recorded_steps.append(step + 1)
                previous_support = current_support.copy()
                print(json.dumps({"case": case["id"], "step": step + 1, "time": state["time"],
                                  "cut_facets": int(cuts.sum()), "separate_components": len(np.unique(state["tet_piece_ids"]))}), flush=True)
            labels = validate_piece_partition(mesh, states[-1], cuts)
            metadata = {
                "source": "FEMFX", "schema_version": SCHEMA,
                "supervision_kind": "native_facet_fracture", "units": "SI",
                "mesh_sha256": _mesh_digest(mesh), "conditioning_context": context,
                "conditioning_context_sha256": context_hash,
                "density_kg_per_m3": material["rest_density"],
                "parameter_status": specification["parameter_status"],
                "calibration_source": specification.get("calibration_source"),
                "case": case, "reference_build": build, "native_capabilities": capabilities,
                "native_settings": scene.settings(),
                "native_observation_schema": OBSERVATION_SCHEMA if observation_fields else None,
                "native_solver_observation_schema": SOLVER_OBSERVATION_SCHEMA if solver_observations else None,
                "native_elastic_state_schema": ELASTIC_STATE_SCHEMA if capture_elastic_state else None,
                "native_observation_semantics": "preceding single native update only; unobserved masks remain unknown; values are read at native application sites, not inferred from endpoints" if observation_fields else "not recorded",
                "cumulative_cut_semantics": "native_new_exterior_faces",
                "fracture_time_semantics": "first observed after native update; event is localized only to the preceding timestep interval",
                "physics_reference": "FEMFX numerical model; not an experimental physical ground truth",
                "fracture_law": "native stress-threshold tetrahedral-face splitting, plasticity disabled",
                "final_time": states[-1]["time"],
            }
            cases = output_dir / "cases"
            cases.mkdir(exist_ok=True)
            np.savez_compressed(cases / f"{case['id']}.npz", impacts=impacts, cell_labels=labels, cut_labels=cuts,
                                internal_face_ids=mesh.internal_faces, known_cells=np.ones(len(mesh.tets), dtype=bool),
                                valid_facets=np.ones(len(cuts), dtype=bool), shape="femfx", case_id=case["id"],
                                metadata_json=json.dumps(metadata, allow_nan=False))
            trajectories = output_dir / "trajectories"
            trajectories.mkdir(exist_ok=True)
            native_topology = {f"native_{name}": np.concatenate([record[name] for record in states]) for name in nodal_fields}
            native_topology["native_node_offsets"] = np.r_[0, np.cumsum([len(record["nodal_masses"]) for record in states])]
            native_topology["native_corner_vertex_ids"] = np.stack([record["corner_vertex_ids"] for record in states])
            if capture_elastic_state:
                native_topology.update({f"native_{name}": np.stack([record[name] for record in states]) for name in ELASTIC_STATE_FIELDS})
            native_observations = {f"native_{name}": np.stack([record[name] for record in states]) for name in observation_fields}
            support_history = np.stack([record["corner_kinematic"] for record in states]).astype(bool)
            releases = support_history[:-1] & ~support_history[1:]
            np.savez_compressed(trajectories / f"{case['id']}.npz",
                                times=np.asarray([state["time"] for state in states]),
                                corner_positions=np.stack([state["corner_positions"] for state in states]),
                                corner_velocities=np.stack([state["corner_velocities"] for state in states]),
                                cut_labels=np.stack(cut_history),
                                corner_kinematic=support_history,
                                support_release_labels=releases,
                                recorded_step_ids=np.asarray(recorded_steps, dtype=np.int64),
                                native_piece_ids=np.stack([state["tet_piece_ids"] for state in states]),
                                initial_vertex_masses=initial["initial_vertex_masses"],
                                first_observed_cut_times=np.asarray(state["fracture_times"])[order],
                                original_tet_ids=np.arange(len(mesh.tets)),
                                metadata_json=json.dumps({
                                    "schema": "fracture-world.femfx-trajectory.v1",
                                    "source_case_sha256": file_hash(cases / f"{case['id']}.npz"),
                                    "mesh_sha256": _mesh_digest(mesh),
                                    "conditioning_context_sha256": context_hash,
                                    "native_revision": REVISION,
                                    "native_topology_schema": "fracture-world.femfx-nodal-topology.v1",
                                    "native_observation_schema": OBSERVATION_SCHEMA if observation_fields else None,
                                    "native_solver_observation_schema": SOLVER_OBSERVATION_SCHEMA if solver_observations else None,
                                    "native_elastic_state_schema": ELASTIC_STATE_SCHEMA if capture_elastic_state else None,
                                }, allow_nan=False), **native_topology, **native_observations)
            if export_replay:
                export_reference_replay(mesh, states, cut_history, metadata, output_dir / "replays" / f"{case['id']}.json")
            result = {"case": case["id"], "steps": specification["steps"], "cut_facets": int(cuts.sum()),
                      "fragments": len(np.unique(labels)), "partial_crack_facets": int(np.sum(cuts & (labels[mesh.dual_edges[:, 0]] == labels[mesh.dual_edges[:, 1]])))}
            released_steps = np.flatnonzero(np.any(releases, axis=(1, 2)))
            result.update(support_release_corner_events=int(releases.sum()),
                          released_original_support_vertices=int(len(np.unique(mesh.tets[np.any(releases, axis=0)]))),
                          first_support_release_step=int(recorded_steps[released_steps[0] + 1]) if len(released_steps) else None,
                          native_steps_after_first_support_release=int(specification["steps"] - recorded_steps[released_steps[0] + 1]) if len(released_steps) else None,
                          support_release_semantics="recorded native fixed-to-free transitions of original material corners; no synthetic flag changes")
            write_json(cases / f"{case['id']}.json", metadata)
            results.append(result)
        except (RuntimeError, ValueError) as error:
            if not skip_invalid:
                raise
            for folder, extension in (("cases", ".npz"), ("cases", ".json"), ("trajectories", ".npz"), ("replays", ".json")):
                (output_dir / folder / f"{case['id']}{extension}").unlink(missing_ok=True)
            rejected.append({"case": case["id"], "reason": str(error), "accepted_as_training_reference": False})
            print(json.dumps({"case": case["id"], "status": "rejected", "reason": str(error)}), flush=True)
        finally:
            if scene is not None:
                scene.close()
    summary = {"source": "FEMFX", "cases": results, "conditioning_context_sha256": context_hash,
               "parameter_status": specification["parameter_status"], "output": str(output_dir.resolve()),
               "requested_cases": len(specification["cases"]), "rejected_cases": rejected,
               "support_release_corner_events": sum(row["support_release_corner_events"] for row in results)}
    write_json(output_dir / "summary.json", summary)
    return summary


def reference_surface(mesh, positions, cuts):
    boundary = np.flatnonzero(mesh.faces.cells[:, 1] < 0)
    opened = mesh.internal_faces[np.asarray(cuts, dtype=bool)]
    face_ids = np.r_[boundary, opened, opened]
    owners = np.r_[mesh.faces.cells[boundary, 0], mesh.faces.cells[opened, 0], mesh.faces.cells[opened, 1]]
    triangles = mesh.faces.vertices[face_ids].copy()
    if len(opened):
        triangles[-len(opened):] = triangles[-len(opened):, ::-1]
    corners = np.argmax(mesh.tets[owners, :, None] == triangles[:, None, :], axis=1)
    vertices = np.asarray(positions)[owners[:, None], corners].reshape((-1, 3))
    return {"vertices": vertices.tolist(), "faces": np.arange(len(vertices)).reshape((-1, 3)).tolist()}


def export_reference_replay(mesh, states, cut_history, metadata, path):
    geometry = reference_surface(mesh, states[0]["corner_positions"], cut_history[0])
    frames = []
    for state, cuts in zip(states, cut_history):
        frames.append({
            "time": state["time"],
            "poses": [{"position": [0, 0, 0], "orientation": [0, 0, 0, 1]}],
            "geometries": [reference_surface(mesh, state["corner_positions"], cuts)],
            "metrics": {"cut_facets": int(cuts.sum()), "native_pieces": len(np.unique(state["tet_piece_ids"])),
                        "warnings": int(state["warnings"])},
        })
    write_json(path, {"title": "FEMFX numerical fracture reference", "source": "femfx_numerical_reference",
                      "metadata": {**metadata, "neural_prediction": False, "proximal_fracture_correction": False,
                                   "suggested_playback_speed": 0.005 if "floor_height_m" in metadata["conditioning_context"] else 0.001,
                                   **({"floor_height_m": metadata["conditioning_context"]["floor_height_m"], "up_axis": "y"}
                                      if "floor_height_m" in metadata["conditioning_context"] else {})},
                      "bodies": [{"id": 0, **geometry}], "frames": frames})


def create_trial(output_dir, steps=50):
    from itertools import permutations, product

    if isinstance(steps, bool) or not isinstance(steps, int) or steps < 1:
        raise ValueError("trial steps must be a positive integer")
    output_dir = Path(output_dir)
    if output_dir.exists() and any(output_dir.iterdir()):
        raise ValueError("trial input output must be empty")
    subdivisions = np.array([6, 2, 2])
    size = np.array([0.12, 0.04, 0.04])
    nodes = np.array(list(product(*(np.linspace(0, size[axis], subdivisions[axis] + 1) for axis in range(3)))))
    indices = np.arange(len(nodes)).reshape(tuple(subdivisions + 1))
    tets = []
    for lower in product(*(range(value) for value in subdivisions)):
        for order in permutations(range(3)):
            corner = np.array(lower)
            tetrahedron = [indices[tuple(corner)]]
            for axis in order:
                corner = corner.copy()
                corner[axis] += 1
                tetrahedron.append(indices[tuple(corner)])
            tets.append(tetrahedron)
    mesh = VolumeMesh(nodes, np.asarray(tets, dtype=np.int64))
    material = {"rest_density": 1000.0, "youngs_modulus": 1e6, "poissons_ratio": 0.25,
                "fracture_stress_threshold": 1e4}
    masses = np.zeros(len(nodes))
    np.add.at(masses, mesh.tets.ravel(), np.repeat(mesh.volumes * material["rest_density"] / 4, 4))
    fixed = np.flatnonzero(nodes[:, 0] == 0)
    loaded = np.flatnonzero(nodes[:, 0] == size[0])
    cases = []
    for index, velocity in enumerate([0.05, 0.5, 2.0, 4.0]):
        cases.append({"id": f"trial-{index + 1}", "vertex_impulses": [
            {"vertex": int(vertex), "impulse": [float(masses[vertex] * velocity), 0.0, 0.0]} for vertex in loaded
        ]})
    specification = {
        "schema": SCHEMA, "parameter_status": "uncalibrated_trial", "material": material,
        "dt": 0.0001, "steps": steps, "gravity": [0.0, 0.0, 0.0],
        "fixed_vertices": fixed.tolist(), "threads": 1, "cases": cases,
        "scene_options": {"friction_coefficient": 0.0, "rayleigh_mass_damping": 0.0,
                          "rayleigh_stiffness_damping": 0.0},
        "purpose": "integration test on an uncalibrated fixed-end block with instantaneous end-vertex impulses",
    }
    validate_specification(mesh, specification)
    output_dir.mkdir(parents=True, exist_ok=True)
    mesh.save(output_dir / "mesh.npz")
    write_json(output_dir / "cases.json", specification)
    return {"mesh": str((output_dir / "mesh.npz").resolve()),
            "specification": str((output_dir / "cases.json").resolve()),
            "length_scale": 1.0, "parameter_status": "uncalibrated_trial", "cells": len(mesh.tets)}
