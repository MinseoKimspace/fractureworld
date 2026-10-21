from itertools import product
import json
from pathlib import Path

import numpy as np

from .femfx import create_trial, validate_specification
from .io import file_hash, new_directory, write_json
from .mesh import VolumeMesh


SCHEMA = "fracture-world.dynamics-suite.v1"
REGIONS = {"entire_end", "upper_y"}


def _configuration(path):
    values = json.loads(Path(path).read_text(encoding="utf-8"))
    required = {"steps", "velocity_magnitudes", "directions", "loaded_regions"}
    if not isinstance(values, dict) or not required.issubset(values) or set(values) - required - {"schema"}:
        raise ValueError("dynamics suite requires steps, velocity_magnitudes, directions and loaded_regions")
    if values.get("schema", SCHEMA) != SCHEMA:
        raise ValueError("unsupported dynamics suite schema")
    steps = values["steps"]
    if isinstance(steps, bool) or not isinstance(steps, int) or steps < 1:
        raise ValueError("steps must be a positive integer")
    speeds = values["velocity_magnitudes"]
    if not isinstance(speeds, list) or not speeds or any(isinstance(value, bool) or not isinstance(value, (int, float)) for value in speeds):
        raise ValueError("velocity_magnitudes must be a nonempty list of finite positive speeds")
    speeds = np.asarray(speeds, dtype=float)
    if not np.isfinite(speeds).all() or (speeds <= 0).any() or len(np.unique(speeds)) != len(speeds):
        raise ValueError("velocity_magnitudes must contain unique finite positive speeds")
    raw_directions = values["directions"]
    if not isinstance(raw_directions, list) or not raw_directions:
        raise ValueError("directions must be a nonempty list of finite nonzero three-vectors")
    if any(not isinstance(vector, list) or len(vector) != 3 or
           any(isinstance(value, bool) or not isinstance(value, (int, float)) for value in vector) for vector in raw_directions):
        raise ValueError("directions must contain finite nonzero three-vectors")
    directions = np.asarray(raw_directions, dtype=float)
    norms = np.linalg.norm(directions, axis=1)
    if not np.isfinite(directions).all() or not np.isfinite(norms).all() or (norms <= 0).any():
        raise ValueError("directions must contain finite nonzero three-vectors")
    directions = directions / norms[:, None]
    if len(np.unique(np.round(directions, 12), axis=0)) != len(directions):
        raise ValueError("directions must be unique after normalization")
    regions = values["loaded_regions"]
    if not isinstance(regions, list) or not regions or any(not isinstance(value, str) or value not in REGIONS for value in regions):
        raise ValueError("loaded_regions must select entire_end or upper_y")
    if len(set(regions)) != len(regions):
        raise ValueError("loaded_regions must be unique")
    return values, steps, speeds, directions, regions


def _loaded_vertices(mesh, region, fixed):
    end = mesh.nodes[:, 0] == mesh.nodes[:, 0].max()
    if region == "upper_y":
        end &= mesh.nodes[:, 1] > (mesh.nodes[:, 1].min() + mesh.nodes[:, 1].max()) / 2
    vertices = np.flatnonzero(end)
    if not len(vertices) or np.isin(vertices, fixed).any():
        raise ValueError("every loaded region must contain unfixed vertices and no fixed vertices")
    return vertices


def create_dynamics_suite(output_dir, specification_path):
    values, steps, speeds, directions, regions = _configuration(specification_path)
    output_dir = new_directory(output_dir)
    trial = create_trial(output_dir, steps=steps)
    mesh = VolumeMesh.load(trial["mesh"])
    specification = json.loads(Path(trial["specification"]).read_text(encoding="utf-8"))
    material, fixed = validate_specification(mesh, specification)
    masses = np.zeros(len(mesh.nodes))
    np.add.at(masses, mesh.tets.ravel(), np.repeat(mesh.volumes * material["rest_density"] / 4, 4))
    if not np.isfinite(masses).all() or (masses <= 0).any():
        raise ValueError("every loaded material vertex must have positive finite rest mass")
    case_parameters = []
    cases = []
    for index, (speed, direction, region) in enumerate(product(speeds, directions, regions), start=1):
        vertices = _loaded_vertices(mesh, region, fixed)
        identifier = f"dynamics-{index:04d}"
        cases.append({"id": identifier, "vertex_impulses": [
            {"vertex": int(vertex), "impulse": (masses[vertex] * speed * direction).tolist()} for vertex in vertices
        ]})
        case_parameters.append({"id": identifier, "velocity_magnitude_m_per_s": float(speed),
                                "direction": direction.tolist(), "loaded_region": region,
                                "loaded_vertices": vertices.tolist()})
    specification.update(cases=cases, record_stride=1,
                         purpose="offline uncalibrated numerical-teacher sweep for motion, fracture and observed support release")
    validate_specification(mesh, specification)
    write_json(output_dir / "cases.json", specification)
    manifest = {
        "schema": SCHEMA, "configuration": values, "configuration_sha256": file_hash(specification_path),
        "mesh": "mesh.npz", "specification": "cases.json", "case_count": len(cases),
        "parameter_status": "uncalibrated_trial", "length_scale": 1.0,
        "support_semantics": "original native FEMFX kinematic flags; fracture-copy release is recorded by the numerical teacher",
        "additional_removable_support_flags": False, "synthetic_release_labels": False,
        "simulation_executed": False, "positive_release_examples_confirmed": False,
        "loaded_region_definitions": {"entire_end": "maximum-x face",
                                      "upper_y": "maximum-x face with y strictly above the mesh y midpoint"},
        "cases": case_parameters,
    }
    write_json(output_dir / "manifest.json", manifest)
    return {**trial, "case_count": len(cases), "manifest": str((output_dir / "manifest.json").resolve()),
            "native_runtime_required": False, "simulation_executed": False}
