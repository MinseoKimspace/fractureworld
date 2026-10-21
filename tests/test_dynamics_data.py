from dataclasses import replace
import json
import subprocess
import sys

import numpy as np
from numpy.testing import assert_allclose, assert_array_equal
from scipy.spatial import ConvexHull
import pytest

from fracture_world.data import _mesh_digest
from fracture_world.dynamics_data import (
    load_dynamics_dataset,
    load_dynamics_trajectory,
    material_topology,
    split_dynamics_cases,
    validate_dynamics_compatibility,
)
from fracture_world.io import file_hash
from fracture_world.mesh import VolumeMesh
from recorded_cases import native_fixture, update_archive


def recorded_trajectory(directory, supports=False):
    case_path, trajectory_path = native_fixture(directory)
    with np.load(case_path, allow_pickle=False) as archive:
        metadata = json.loads(str(archive["metadata_json"].item()))
    if supports:
        context = metadata["conditioning_context"]
        context["fixed_vertices"] = [0]
        import hashlib

        metadata["conditioning_context_sha256"] = hashlib.sha256(
            json.dumps(context, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()
        ).hexdigest()
        update_archive(case_path, lambda arrays: arrays.update(metadata_json=np.asarray(json.dumps(metadata))))
    mesh = VolumeMesh.load(directory / "mesh.npz")
    topology = material_topology(mesh, metadata["density_kg_per_m3"])
    with np.load(trajectory_path, allow_pickle=False) as archive:
        arrays = {key: np.array(archive[key], copy=True) for key in archive.files}
    arrays["corner_positions"][-1] += np.arange(len(mesh.tets))[:, None, None] * np.asarray([0.001, 0, 0])
    arrays["corner_velocities"][-1] += np.arange(len(mesh.tets))[:, None, None] * np.asarray([0.2, 0.1, 0])
    offsets = [0]
    connectivity = []
    fields = {name: [] for name in ("rest_positions", "positions", "velocities", "masses", "original_ids", "kinematic", "flags")}
    for frame, cuts in enumerate(arrays["cut_labels"]):
        parent = np.arange(len(topology.rest_positions))

        def find(index):
            while parent[index] != index:
                index = parent[index]
            return index

        for first, second in topology.face_pairs[cuts == 0].reshape(-1, 2):
            parent[find(first)] = find(second)
        roots = np.asarray([find(index) for index in range(len(parent))])
        _, first, ids = np.unique(roots, return_index=True, return_inverse=True)
        original = mesh.tets.ravel()[first]
        kinematic = np.isin(original, [0]) if supports else np.zeros(len(first), dtype=bool)
        if supports and frame == len(arrays["cut_labels"]) - 1:
            copies = np.flatnonzero(original == 0)
            kinematic[copies[1:]] = False
        fields["rest_positions"].append(mesh.nodes.astype(np.float32)[original])
        fields["positions"].append(arrays["corner_positions"][frame].reshape(-1, 3)[first])
        fields["velocities"].append(arrays["corner_velocities"][frame].reshape(-1, 3)[first])
        fields["masses"].append(np.bincount(ids, weights=topology.masses, minlength=len(first)))
        fields["original_ids"].append(original)
        fields["kinematic"].append(kinematic)
        fields["flags"].append(kinematic.astype(np.int64))
        connectivity.append(ids.reshape(-1, 4))
        offsets.append(offsets[-1] + len(first))
    arrays.update({f"native_nodal_{key}": np.concatenate(values) for key, values in fields.items()})
    arrays["native_corner_vertex_ids"] = np.asarray(connectivity)
    arrays["native_node_offsets"] = np.asarray(offsets)
    arrays["metadata_json"] = np.asarray(json.dumps({
        "schema": "fracture-world.femfx-trajectory.v1", "source_case_sha256": file_hash(case_path),
        "mesh_sha256": _mesh_digest(mesh), "conditioning_context_sha256": metadata["conditioning_context_sha256"],
        "native_revision": metadata["conditioning_context"]["native_revision"],
        "native_topology_schema": "fracture-world.femfx-nodal-topology.v1",
    }))
    np.savez_compressed(trajectory_path, **arrays)
    return case_path, trajectory_path


def test_stable_material_corner_indexing_preserves_split_velocities_and_partial_cuts(tmp_path):
    case, path = recorded_trajectory(tmp_path)
    trajectory = load_dynamics_trajectory(case, path)
    assert len(trajectory) == 3
    assert trajectory.positions.shape == (4, 12, 3)
    assert trajectory.broken.shape == (4, 3)
    assert_array_equal(trajectory.broken[1], [True, False, False])
    original_ids = trajectory.topology.mesh.tets.ravel()
    copies = np.flatnonzero(original_ids == 0)
    assert len(np.unique(trajectory.velocities[-1, copies], axis=0)) == 3
    assert_allclose(trajectory.topology.masses.sum(), 6 * ConvexHull(trajectory.topology.mesh.nodes).volume)
    assert_allclose(trajectory.initial_impulses.sum(axis=0), [0.01, 0, 0])
    pairs = trajectory.topology.face_pairs
    assert_array_equal(original_ids[pairs[:, :, 0]], original_ids[pairs[:, :, 1]])
    assert not trajectory.metadata["native_runtime_required"]
    transition = trajectory.transition(2)
    assert transition.dt == 0.125
    assert transition.time == 0.25
    assert_array_equal(transition.target_positions, trajectory.positions[3])
    for values in (trajectory.positions, trajectory.velocities, trajectory.broken, trajectory.fixed,
                   trajectory.topology.masses, trajectory.topology.face_pairs, transition.positions):
        assert not values.flags.writeable


def test_recorded_support_release_is_supervision_and_not_silently_reconstructed(tmp_path):
    case, path = recorded_trajectory(tmp_path, supports=True)
    trajectory = load_dynamics_trajectory(case, path)
    transition = trajectory.transition(2)
    assert np.count_nonzero(transition.fixed) > np.count_nonzero(transition.target_fixed)
    assert trajectory.metadata["observed_support_releases"] == 2
    assert trajectory.metadata["support_release_requires_explicit_runtime_policy"]
    update_archive(path, lambda arrays: arrays.pop("native_nodal_kinematic"))
    with pytest.raises(ValueError, match="support flags"):
        load_dynamics_trajectory(case, path)


@pytest.mark.parametrize("corruption,match", [
    ("healing", "irreversible"), ("clock", "timestep clock"), ("velocity", "recorded loading"),
    ("masses", "rest-volume mass"), ("provenance", "provenance"), ("support", "support flags"),
    ("identity", "material-corner identity"),
])
def test_incompatible_or_corrupted_trajectory_is_rejected(tmp_path, corruption, match):
    case, path = recorded_trajectory(tmp_path)

    def modify(arrays):
        if corruption == "healing":
            arrays["cut_labels"][2] = 0
        elif corruption == "clock":
            arrays["times"][1] += 1e-6
        elif corruption == "velocity":
            arrays["corner_velocities"][0, 0, 0, 0] += 1
        elif corruption == "masses":
            arrays["native_nodal_masses"][-1] *= 2
        elif corruption == "provenance":
            provenance = json.loads(str(arrays["metadata_json"].item()))
            provenance["source_case_sha256"] = "0" * 64
            arrays["metadata_json"] = np.asarray(json.dumps(provenance))
        elif corruption == "support":
            arrays["native_nodal_flags"][0] = 1
        elif corruption == "identity":
            arrays["native_corner_vertex_ids"][-1, 0, [0, 1]] = arrays["native_corner_vertex_ids"][-1, 0, [1, 0]]

    update_archive(path, modify)
    with pytest.raises(ValueError, match=match):
        load_dynamics_trajectory(case, path)


def test_complete_conditions_stay_together_and_split_is_order_independent(tmp_path):
    case, path = recorded_trajectory(tmp_path)
    trajectory = load_dynamics_trajectory(case, path)
    cases = [replace(trajectory, case_id=f"condition-{index}", condition_digest=str(index)) for index in range(6)]
    cases.append(replace(cases[0], case_id="same-loading-replicate"))
    first = split_dynamics_cases(cases, seed=9)
    second = split_dynamics_cases(list(reversed(cases)), seed=9)
    assert {name: [item.case_id for item in items] for name, items in first.items()} == {
        name: [item.case_id for item in items] for name, items in second.items()
    }
    sets = {name: {item.condition_digest for item in rows} for name, rows in first.items()}
    assert not sets["train"] & sets["validation"]
    assert not sets["train"] & sets["test"]
    assert not sets["validation"] & sets["test"]
    for rows in first.values():
        assert all(len(item) == 3 for item in rows)
    with pytest.raises(ValueError, match="three distinct"):
        split_dynamics_cases([trajectory])
    assert split_dynamics_cases([trajectory], overfit=True)["train"] == [trajectory]


def test_loading_context_replicates_group_despite_case_name_and_time_horizon(tmp_path):
    first, trajectory = recorded_trajectory(tmp_path / "one")
    original = load_dynamics_trajectory(first, trajectory)
    second, other_trajectory = recorded_trajectory(tmp_path / "two")
    import hashlib

    def update_case(arrays):
        metadata = json.loads(str(arrays["metadata_json"].item()))
        context = metadata["conditioning_context"]
        context["dt"], context["steps"] = 0.0625, 6
        metadata["native_settings"]["dt"] = 0.0625
        metadata["conditioning_context_sha256"] = hashlib.sha256(
            json.dumps(context, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()
        ).hexdigest()
        metadata["case"]["id"] = "renamed"
        arrays["case_id"] = np.asarray("renamed")
        arrays["metadata_json"] = np.asarray(json.dumps(metadata))

    update_archive(second, update_case)
    renamed_case = second.with_name("renamed.npz")
    second.rename(renamed_case)

    def update_provenance(arrays):
        with np.load(renamed_case, allow_pickle=False) as archive:
            context_hash = json.loads(str(archive["metadata_json"].item()))["conditioning_context_sha256"]
        metadata = json.loads(str(arrays["metadata_json"].item()))
        metadata["source_case_sha256"] = file_hash(renamed_case)
        metadata["conditioning_context_sha256"] = context_hash
        arrays["metadata_json"] = np.asarray(json.dumps(metadata))

    update_archive(other_trajectory, update_provenance)
    renamed_trajectory = other_trajectory.with_name("renamed.npz")
    other_trajectory.rename(renamed_trajectory)
    replicated = load_dynamics_trajectory(renamed_case, renamed_trajectory)
    assert replicated.case_id != original.case_id
    assert replicated.metadata["conditioning_context"]["steps"] != original.metadata["conditioning_context"]["steps"]
    assert replicated.condition_digest == original.condition_digest


def test_dataset_rejects_different_material_topologies(tmp_path):
    case, path = recorded_trajectory(tmp_path)
    trajectory = load_dynamics_trajectory(case, path)
    other_mesh = VolumeMesh(trajectory.topology.mesh.nodes * 2, trajectory.topology.mesh.tets)
    other = replace(trajectory, case_id="other", topology=material_topology(other_mesh, 6))
    with pytest.raises(ValueError, match="compatible material mesh"):
        validate_dynamics_compatibility([trajectory, other])
    assert len(load_dynamics_dataset(tmp_path)) == 1


@pytest.mark.parametrize("frame", [-1, True, 3, 0.5])
def test_invalid_transition_selection_is_rejected(tmp_path, frame):
    case, path = recorded_trajectory(tmp_path)
    with pytest.raises(ValueError, match="transition frame"):
        load_dynamics_trajectory(case, path).transition(frame)


def test_portable_data_import_does_not_load_any_native_engine_module():
    code = "import sys; import fracture_world.dynamics_data; assert 'fracture_world.femfx' not in sys.modules; assert '_femfx' not in sys.modules"
    result = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True)
    assert result.returncode == 0, result.stderr
