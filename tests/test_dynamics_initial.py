import json
import subprocess
import sys

import numpy as np
from numpy.testing import assert_array_equal
import pytest

from fracture_world.dynamics_initial import ARRAYS, export_initial, load_initial
from test_dynamics_data import recorded_trajectory
from recorded_cases import update_archive


def constant_supports(path):
    def change(arrays):
        original = arrays["native_nodal_original_ids"]
        arrays["native_nodal_kinematic"] = original == 0
        arrays["native_nodal_flags"] = (original == 0).astype(np.int64)
        arrays["native_nodal_positions"][original == 0] = [0, 0, -1]
        arrays["native_nodal_velocities"][original == 0] = 0
        offsets = arrays["native_node_offsets"]
        for frame in range(len(offsets) - 1):
            ids = arrays["native_corner_vertex_ids"][frame]
            selected = slice(offsets[frame], offsets[frame + 1])
            arrays["corner_positions"][frame] = arrays["native_nodal_positions"][selected][ids]
            arrays["corner_velocities"][frame] = arrays["native_nodal_velocities"][selected][ids]

    update_archive(path, change)


def test_export_contains_one_selected_state_and_loads_without_source_archives(tmp_path):
    case, trajectory = recorded_trajectory(tmp_path / "source")
    output = tmp_path / "initial.npz"
    initial = export_initial(case, trajectory, 2, output, support_policy="free")
    assert initial.time == 0.25
    assert initial.dt == 0.125
    assert initial.metadata["future_state_included"] is False
    assert initial.metadata["support_policy"] == "free"
    assert initial.metadata["observed_source_support_constant"]
    with np.load(output, allow_pickle=False) as archive:
        assert set(archive.files) == ARRAYS
        assert archive["positions"].shape == (12, 3)
        assert archive["broken"].shape == (3,)
        assert not any(key.startswith("target") or key in {"times", "cut_labels", "corner_positions"} for key in archive.files)
    for path in (case, trajectory, case.parent.parent / "mesh.npz"):
        path.rename(path.with_suffix(".backup"))
    loaded = load_initial(output)
    assert_array_equal(loaded.positions, initial.positions)
    assert_array_equal(loaded.velocities, initial.velocities)
    assert_array_equal(loaded.broken, initial.broken)
    assert loaded.material == initial.material
    assert len(loaded.metadata["source_hashes"]) == 3
    assert not loaded.topology.mesh.nodes.flags.writeable


def test_fixed_support_policy_requires_an_explicit_choice_and_invariant_source_history(tmp_path):
    case, trajectory = recorded_trajectory(tmp_path / "source", supports=True)
    output = tmp_path / "initial.npz"
    with pytest.raises(TypeError, match="support_policy"):
        export_initial(case, trajectory, 0, output)
    with pytest.raises(ValueError, match="support-changing"):
        export_initial(case, trajectory, 0, output, support_policy="prescribed")
    constant_supports(trajectory)
    with pytest.raises(ValueError, match="fixed source supports"):
        export_initial(case, trajectory, 0, output, support_policy="free")
    state = export_initial(case, trajectory, 3, output, support_policy="prescribed")
    assert state.time == 0.375
    assert state.fixed.sum() == 3
    assert state.metadata["support_policy"] == "prescribed_material_corners"
    assert_array_equal(state.velocities[state.fixed], 0)


def test_learned_support_policy_exports_current_flags_without_future_release_labels(tmp_path):
    case, trajectory = recorded_trajectory(tmp_path / "source", supports=True)
    output = tmp_path / "learned.npz"
    state = export_initial(case, trajectory, 0, output, support_policy="learned")
    assert state.metadata["support_policy"] == "learned_material_corner_release"
    assert state.metadata["support_semantics"] == "recorded_numerical_teacher_kinematic_flags"
    assert state.metadata["observed_source_support_constant"] is False
    assert state.fixed.sum() == 3
    with np.load(output, allow_pickle=False) as archive:
        assert set(archive.files) == ARRAYS
        assert archive["fixed"].shape == (12,)
        assert not any(key.startswith("target") or "release" in key for key in archive.files)
        metadata = json.loads(str(archive["metadata_json"].item()))
        assert "observed_support_releases" not in metadata
        assert "target_fixed" not in metadata
    loaded = load_initial(output)
    assert_array_equal(loaded.fixed, state.fixed)
    assert loaded.metadata["support_policy"] == "learned_material_corner_release"


def test_learned_support_policy_requires_explicit_numerical_teacher_semantics(tmp_path):
    case, trajectory = recorded_trajectory(tmp_path / "source", supports=True)
    output = tmp_path / "learned.npz"
    export_initial(case, trajectory, 0, output, support_policy="learned")

    def change(arrays):
        metadata = json.loads(str(arrays["metadata_json"].item()))
        metadata["support_semantics"] = "externally_prescribed_immutable_anchors"
        arrays["metadata_json"] = np.asarray(json.dumps(metadata))

    update_archive(output, change)
    with pytest.raises(ValueError, match="numerical teacher"):
        load_initial(output)


@pytest.mark.parametrize("corruption,match", [
    ("future", "exactly one current state"), ("fixed", "free support policy"),
    ("healing_metadata", "current-state-only"), ("inversion", "inverted"), ("nonfinite", "finite"),
    ("support_policy", "support policy"),
])
def test_initial_artifact_validation_rejects_future_data_and_inconsistent_states(tmp_path, corruption, match):
    case, trajectory = recorded_trajectory(tmp_path / "source")
    path = tmp_path / "initial.npz"
    export_initial(case, trajectory, 0, path, support_policy="free")

    def change(arrays):
        if corruption == "future":
            arrays["target_positions"] = arrays["positions"].copy()
        elif corruption == "fixed":
            arrays["fixed"][0] = True
        elif corruption == "inversion":
            arrays["positions"][[0, 1]] = arrays["positions"][[1, 0]]
        elif corruption == "nonfinite":
            arrays["positions"][0, 0] = np.nan
        else:
            metadata = json.loads(str(arrays["metadata_json"].item()))
            if corruption == "healing_metadata":
                metadata["future_state_included"] = True
            else:
                metadata["support_policy"] = "automatic_release"
            arrays["metadata_json"] = np.asarray(json.dumps(metadata))

    update_archive(path, change)
    with pytest.raises(ValueError, match=match):
        load_initial(path)


def test_initial_runtime_import_never_loads_engine_wrappers_or_native_binary():
    command = "import sys; from fracture_world.dynamics_initial import load_initial; assert 'fracture_world.femfx' not in sys.modules; assert 'fracture_world.bullet' not in sys.modules; assert '_femfx' not in sys.modules"
    result = subprocess.run([sys.executable, "-c", command], capture_output=True, text=True)
    assert result.returncode == 0, result.stderr
