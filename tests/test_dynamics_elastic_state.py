import hashlib
import json

import numpy as np
from numpy.testing import assert_array_equal
import pytest

from fracture_world.dynamics_data import (
    ELASTIC_STATE_KEYS, ELASTIC_STATE_SCHEMA, elastic_state_signature,
    load_dynamics_dataset, load_dynamics_trajectory,
)
from fracture_world.dynamics_initial import export_initial, load_initial
from fracture_world.io import file_hash
from recorded_cases import update_archive
from test_dynamics_data import recorded_trajectory


def recorded_elastic_state(directory):
    case, trajectory = recorded_trajectory(directory)

    def advertise(arrays):
        metadata = json.loads(str(arrays["metadata_json"].item()))
        metadata["native_elastic_state_schema"] = ELASTIC_STATE_SCHEMA
        arrays["metadata_json"] = np.asarray(json.dumps(metadata))

    update_archive(case, advertise)

    def capture(arrays):
        provenance = json.loads(str(arrays["metadata_json"].item()))
        provenance.update(native_elastic_state_schema=ELASTIC_STATE_SCHEMA, source_case_sha256=file_hash(case))
        arrays["metadata_json"] = np.asarray(json.dumps(provenance))
        frames, tetrahedra = arrays["native_piece_ids"].shape
        quarter_turns = np.array([np.eye(3), [[0., -1., 0.], [1., 0., 0.], [0., 0., 1.]],
                                 [[-1., 0., 0.], [0., -1., 0.], [0., 0., 1.]],
                                 [[0., 1., 0.], [-1., 0., 0.], [0., 0., 1.]]], dtype=np.float32)
        arrays["native_cached_tet_rotations"] = np.broadcast_to(quarter_turns[:frames, None], (frames, tetrahedra, 3, 3)).copy()
        arrays["native_cached_tet_centers_of_mass"] = arrays["corner_positions"].mean(axis=2).astype(np.float32)
        local = np.zeros((frames, tetrahedra), dtype=np.int64)
        for frame, pieces in enumerate(arrays["native_piece_ids"]):
            for piece in np.unique(pieces):
                selected = np.flatnonzero(pieces == piece)
                local[frame, selected] = np.arange(len(selected))
        arrays["native_mesh_tet_ids"] = local

    update_archive(trajectory, capture)
    return case, trajectory


def test_legacy_archives_and_initials_do_not_invent_cached_elastic_state(tmp_path):
    case, path = recorded_trajectory(tmp_path / "source")
    item = load_dynamics_trajectory(case, path)
    assert item.elastic_state is None
    assert item.transition(0).elastic_state is None
    assert load_dynamics_dataset(tmp_path / "source", stream=True)[0].elastic_state is None
    initial = export_initial(case, path, 1, tmp_path / "initial.npz", support_policy="free")
    assert "elastic_state" not in initial.metadata
    assert "native_elastic_state_schema" not in initial.metadata


def test_elastic_state_uses_readonly_current_frame_not_future_transition_target(tmp_path):
    case, path = recorded_elastic_state(tmp_path)
    item = load_dynamics_trajectory(case, path)
    assert item.metadata["native_elastic_state_schema"] == ELASTIC_STATE_SCHEMA
    assert set(item.elastic_state) == set(ELASTIC_STATE_KEYS)
    transition = item.transition(1)
    for name, history in item.elastic_state.items():
        assert not history.flags.writeable
        assert not transition.elastic_state[name].flags.writeable
        assert_array_equal(transition.elastic_state[name], history[1])
    assert not np.array_equal(transition.elastic_state["rotations"], item.elastic_state["rotations"][2])
    assert transition.observations is None


def test_elastic_nodal_mass_history_follows_current_native_vertex_mapping_after_splits(tmp_path):
    case, path = recorded_elastic_state(tmp_path)
    item = load_dynamics_trajectory(case, path)
    with np.load(path, allow_pickle=False) as archive:
        for frame in range(len(item.times)):
            offsets = archive["native_node_offsets"]
            native = archive["native_nodal_masses"][offsets[frame]:offsets[frame + 1]]
            expected = native[archive["native_corner_vertex_ids"][frame]]
            assert_array_equal(item.elastic_state["nodal_masses"][frame], expected)
    streamed = load_dynamics_dataset(tmp_path, stream=True)[0]
    assert_array_equal(streamed.elastic_state["rotations"], item.elastic_state["rotations"])
    assert_array_equal(streamed.transition(2).elastic_state["tetrahedron_local_ids"], item.elastic_state["tetrahedron_local_ids"][2])


def test_portable_initial_exports_only_selected_current_elastic_state(tmp_path):
    case, path = recorded_elastic_state(tmp_path / "source")
    item = load_dynamics_trajectory(case, path)
    output = tmp_path / "initial.npz"
    initial = export_initial(case, path, 2, output, support_policy="free")
    state = initial.metadata["elastic_state"]
    assert state["schema"] == ELASTIC_STATE_SCHEMA
    assert state["positions_hash"] == elastic_state_signature(item.positions[2], item.broken[2], item.fixed[2])
    for name in ELASTIC_STATE_KEYS:
        assert_array_equal(state[name], item.elastic_state[name][2])
    assert np.asarray(state["rotations"]).shape == (3, 3, 3)
    assert initial.metadata["future_state_included"] is False
    for source in (case, path, case.parent.parent / "mesh.npz"):
        source.rename(source.with_suffix(".backup"))
    restored = load_initial(output)
    assert restored.metadata["elastic_state"] == state


def test_elastic_state_signature_is_canonical_and_binds_geometry_topology_and_supports():
    positions = np.arange(24, dtype=np.float32).reshape(8, 3)
    broken = np.array([False, True])
    fixed = np.zeros(8, dtype=bool)
    expected = hashlib.sha256(np.ascontiguousarray(positions, dtype=np.float64).tobytes() + broken.tobytes() + fixed.tobytes()).hexdigest()
    assert elastic_state_signature(positions, broken, fixed) == expected
    assert elastic_state_signature(np.asfortranarray(positions.astype(float)), broken, fixed) == expected
    moved, changed_broken, changed_fixed = positions.copy(), broken.copy(), fixed.copy()
    moved[0, 0] += 1.
    changed_broken[0] = True
    changed_fixed[0] = True
    assert elastic_state_signature(moved, broken, fixed) != expected
    assert elastic_state_signature(positions, changed_broken, fixed) != expected
    assert elastic_state_signature(positions, broken, changed_fixed) != expected


@pytest.mark.parametrize("field", ("native_cached_tet_rotations", "native_cached_tet_centers_of_mass", "native_mesh_tet_ids"))
def test_advertised_elastic_state_rejects_missing_capture_fields(tmp_path, field):
    case, path = recorded_elastic_state(tmp_path)
    update_archive(path, lambda arrays: arrays.pop(field))
    with pytest.raises(ValueError, match="complete captured fields"):
        load_dynamics_trajectory(case, path)


@pytest.mark.parametrize("corruption", ("case_only", "provenance_only", "unknown_schema", "unadvertised"))
def test_elastic_state_requires_matching_supported_case_and_trajectory_provenance(tmp_path, corruption):
    case, path = recorded_elastic_state(tmp_path)

    def change_case(arrays):
        metadata = json.loads(str(arrays["metadata_json"].item()))
        if corruption in {"case_only", "unadvertised"}:
            metadata.pop("native_elastic_state_schema")
        elif corruption == "unknown_schema":
            metadata["native_elastic_state_schema"] = "unsupported"
        arrays["metadata_json"] = np.asarray(json.dumps(metadata))

    def change_history(arrays):
        metadata = json.loads(str(arrays["metadata_json"].item()))
        metadata["source_case_sha256"] = file_hash(case)
        if corruption in {"provenance_only", "unadvertised"}:
            metadata.pop("native_elastic_state_schema")
        elif corruption == "unknown_schema":
            metadata["native_elastic_state_schema"] = "unsupported"
        arrays["metadata_json"] = np.asarray(json.dumps(metadata))

    update_archive(case, change_case)
    update_archive(path, change_history)
    with pytest.raises(ValueError, match="elastic state.*schema"):
        load_dynamics_trajectory(case, path)


@pytest.mark.parametrize("corruption,match", (
    ("rotation_shape", "dimensions"), ("rotation_nonfinite", "dimensions"),
    ("rotation_reflection", "proper rotations"), ("rotation_precision", "float32"),
    ("negative_local_id", "nonnegative"), ("duplicate_local_id", "enumerate"),
    ("shared_piece_vertex", "share native vertices"),
))
def test_corrupt_current_elastic_capture_is_rejected(tmp_path, corruption, match):
    case, path = recorded_elastic_state(tmp_path)

    def change(arrays):
        if corruption == "rotation_shape":
            arrays["native_cached_tet_rotations"] = arrays["native_cached_tet_rotations"][..., :2]
        elif corruption == "rotation_nonfinite":
            arrays["native_cached_tet_rotations"][0, 0, 0, 0] = np.nan
        elif corruption == "rotation_reflection":
            arrays["native_cached_tet_rotations"][0, 0, 0, 0] = -1.
        elif corruption == "rotation_precision":
            values = arrays["native_cached_tet_rotations"].astype(float)
            values[0, 0, 0, 0] = np.nextafter(1., 2.)
            arrays["native_cached_tet_rotations"] = values
        elif corruption == "negative_local_id":
            arrays["native_mesh_tet_ids"][0, 0] = -1
        elif corruption == "duplicate_local_id":
            arrays["native_mesh_tet_ids"][0, 0] = 1
        else:
            arrays["native_piece_ids"][0] = np.arange(3)

    update_archive(path, change)
    with pytest.raises(ValueError, match=match):
        load_dynamics_trajectory(case, path)


@pytest.mark.parametrize("corruption,match", (("signature", "signature"), ("current_position", "signature"),
                                             ("missing_state", "schema"), ("future_cache", "dimensions")))
def test_portable_initial_rejects_stale_or_noncurrent_elastic_state(tmp_path, corruption, match):
    case, path = recorded_elastic_state(tmp_path / "source")
    output = tmp_path / "initial.npz"
    export_initial(case, path, 1, output, support_policy="free")

    def change(arrays):
        metadata = json.loads(str(arrays["metadata_json"].item()))
        if corruption == "signature":
            metadata["elastic_state"]["positions_hash"] = "0" * 64
        elif corruption == "current_position":
            arrays["positions"][0, 0] += .001
        elif corruption == "missing_state":
            metadata.pop("elastic_state")
        else:
            metadata["elastic_state"]["rotations"] = [metadata["elastic_state"]["rotations"]] * 2
        arrays["metadata_json"] = np.asarray(json.dumps(metadata))

    update_archive(output, change)
    with pytest.raises(ValueError, match=match):
        load_initial(output)
