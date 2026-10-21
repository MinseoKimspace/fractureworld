import json

import numpy as np

from fracture_world.data import _mesh_digest
from fracture_world.femfx import REVISION, SCHEMA, context_digest, initial_conditions
from fracture_world.mesh import VolumeMesh
from fracture_world.partition import component_labels


def native_fixture(directory, optional=False):
    directory.mkdir(parents=True, exist_ok=True)
    mesh = VolumeMesh([[0, 0, -1], [0, 0, 1], [1, 0, 0], [-0.5, 0.75, 0], [-0.5, -0.75, 0]],
                      [[0, 1, 2, 3], [0, 1, 3, 4], [0, 1, 4, 2]])
    mesh.save(directory / "mesh.npz")
    material = {"rest_density": 6.0, "youngs_modulus": 1000.0,
                "poissons_ratio": 0.25, "fracture_stress_threshold": 10.0}
    case = {"id": "sample", "vertex_impulses": [{"vertex": 2, "impulse": [0.01, 0.0, 0.0]}]}
    context = {"mesh_si_fingerprint": _mesh_digest(mesh), "material": material,
               "fixed_vertices": [], "dt": 0.125, "steps": 3, "gravity": [0, 0, 0],
               "loading": "instantaneous_vertex_impulses", "scene_options": {}, "threads": 1,
               "native_revision": REVISION, "binding_sha256": "0" * 64}
    velocities, _, impacts = initial_conditions(mesh, material, np.empty(0, dtype=int), case)
    times = np.array([0, 0.125, 0.25, 0.375])
    positions = np.asarray([np.asarray(mesh.nodes + time * velocities, dtype=np.float32)[mesh.tets] for time in times], dtype=np.float64)
    corner_velocities = np.broadcast_to(np.asarray(velocities[mesh.tets], dtype=np.float32), positions.shape).astype(np.float64).copy()
    cuts = np.array([[0, 0, 0], [1, 0, 0], [1, 1, 0], [1, 1, 1]], dtype=np.int8)
    pieces = np.array([[0, 0, 0], [0, 0, 0], [0, 0, 0], [0, 1, 2]], dtype=np.int64)
    metadata = {
        "source": "FEMFX", "schema_version": SCHEMA, "supervision_kind": "native_facet_fracture",
        "units": "SI", "mesh_sha256": _mesh_digest(mesh), "conditioning_context": context,
        "conditioning_context_sha256": context_digest(context), "density_kg_per_m3": 6.0,
        "parameter_status": "uncalibrated_trial", "calibration_source": None, "case": case,
        "native_settings": {"revision": REVISION, "dt": 0.125, "gravity": [0, 0, 0], "material": material,
                            "plasticity": False, "enable_fracture": True},
        "reference_build": {"revision": REVISION, "binding_sha256": "0" * 64}, "final_time": 0.375,
    }
    case_path = directory / "cases" / "sample.npz"
    case_path.parent.mkdir()
    np.savez_compressed(case_path, impacts=impacts,
                        cell_labels=component_labels(len(mesh.tets), mesh.dual_edges, cuts[-1]),
                        cut_labels=cuts[-1], internal_face_ids=mesh.internal_faces,
                        known_cells=np.ones(3, dtype=bool), valid_facets=np.ones(3, dtype=bool),
                        shape="femfx", case_id="sample", metadata_json=json.dumps(metadata))
    trajectory_path = directory / "trajectories" / "sample.npz"
    trajectory_path.parent.mkdir()
    arrays = {"times": times, "corner_positions": positions, "corner_velocities": corner_velocities,
              "cut_labels": cuts, "native_piece_ids": pieces, "original_tet_ids": np.arange(3)}
    if optional:
        masses = np.zeros(len(mesh.nodes))
        np.add.at(masses, mesh.tets.ravel(), np.repeat(mesh.volumes * material["rest_density"] / 4, 4))
        arrays.update(initial_vertex_masses=masses, first_observed_cut_times=times[1:])
    np.savez_compressed(trajectory_path, **arrays)
    return case_path, trajectory_path


def update_archive(path, transform):
    with np.load(path, allow_pickle=False) as archive:
        arrays = {name: np.array(archive[name], copy=True) for name in archive.files}
    transform(arrays)
    np.savez_compressed(path, **arrays)
