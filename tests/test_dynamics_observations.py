from dataclasses import replace
import json

import numpy as np
import pytest
import torch

from fracture_world.dynamics_data import load_dynamics_dataset, load_dynamics_trajectory
from fracture_world.dynamics_learning import transition_batch
from fracture_world.femfx import OBSERVATION_FIELDS, OBSERVATION_SCHEMA, context_digest
from fracture_world.femfx import (SOLVER_BOOLEAN_FIELDS, SOLVER_CORNER_FIELDS, SOLVER_OBSERVATION_FIELDS,
                                  SOLVER_OBSERVATION_SCHEMA, SOLVER_VECTOR_FIELDS)
from fracture_world.io import file_hash
from recorded_cases import update_archive
from test_dynamics_data import recorded_trajectory


def observed_trajectory(directory):
    case, trajectory = recorded_trajectory(directory)

    def observe_case(arrays):
        metadata = json.loads(str(arrays["metadata_json"].item()))
        metadata["native_observation_schema"] = OBSERVATION_SCHEMA
        arrays["metadata_json"] = np.asarray(json.dumps(metadata))

    update_archive(case, observe_case)

    def observe_trajectory(arrays):
        shape = arrays["corner_velocities"].shape
        frames, tetrahedra = shape[:2]
        amount = np.arange(frames, dtype=np.float32)[:, None, None, None]
        delta = np.broadcast_to(.01 * amount, shape).copy()
        observed = np.ones(shape[:3], dtype=bool)
        observed[0] = False
        active = np.zeros_like(observed)
        active[2, 0, 0] = True
        stress_observed = np.ones((frames, tetrahedra), dtype=bool)
        stress_observed[0] = False
        stress_observed[2, 2] = False
        arrays.update(native_pre_contact_velocities=arrays["corner_velocities"].astype(np.float32) - delta,
                      native_applied_delta_velocities=delta,
                      native_stabilization_delta_positions=np.broadcast_to(.001 * amount, shape).copy(),
                      native_contact_observed=observed, native_constraint_solve_active=active,
                      native_fracture_stress_eigenvalues=np.broadcast_to(np.arange(frames, dtype=np.float32)[:, None], (frames, tetrahedra)).copy(),
                      native_fracture_stress_thresholds=np.where(stress_observed, 10., 0.).astype(np.float32),
                      native_fracture_stress_observed=stress_observed)
        provenance = json.loads(str(arrays["metadata_json"].item()))
        provenance.update(source_case_sha256=file_hash(case), native_observation_schema=OBSERVATION_SCHEMA)
        arrays["metadata_json"] = np.asarray(json.dumps(provenance))

    update_archive(trajectory, observe_trajectory)
    return case, trajectory


def test_observations_survive_loading_and_transition_uses_the_endpoint_update(tmp_path):
    case, path = observed_trajectory(tmp_path)
    item = load_dynamics_trajectory(case, path)
    assert item.metadata["native_observation_schema"] == OBSERVATION_SCHEMA
    assert set(item.observations) == {*OBSERVATION_FIELDS, "single_step_coverage"}
    np.testing.assert_array_equal(item.observations["single_step_coverage"], [False, True, True, True])
    with np.load(path, allow_pickle=False) as archive:
        for name in OBSERVATION_FIELDS:
            np.testing.assert_array_equal(item.observations[name], archive[f"native_{name}"])
            assert not item.observations[name].flags.writeable
    transition = item.transition(1)
    assert transition.observations["single_step_coverage"] is True
    for name in OBSERVATION_FIELDS:
        np.testing.assert_array_equal(transition.observations[name], item.observations[name][2])
        assert not transition.observations[name].flags.writeable
    assert np.any(transition.observations["applied_delta_velocities"] != item.observations["applied_delta_velocities"][1])


@pytest.mark.parametrize("explicit_none", [False, True])
def test_legacy_trajectories_keep_observations_unknown_and_metadata_unchanged(tmp_path, explicit_none):
    case, path = recorded_trajectory(tmp_path)
    if explicit_none:
        def no_case_observation(arrays):
            metadata = json.loads(str(arrays["metadata_json"].item()))
            metadata["native_observation_schema"] = None
            arrays["metadata_json"] = np.asarray(json.dumps(metadata))

        update_archive(case, no_case_observation)

        def no_trajectory_observation(arrays):
            provenance = json.loads(str(arrays["metadata_json"].item()))
            provenance.update(source_case_sha256=file_hash(case), native_observation_schema=None)
            arrays["metadata_json"] = np.asarray(json.dumps(provenance))

        update_archive(path, no_trajectory_observation)
    item = load_dynamics_trajectory(case, path)
    assert item.observations is None
    assert item.transition(0).observations is None
    assert "native_observation_schema" not in item.metadata


def test_streamed_transitions_retain_observations_without_adding_network_inputs(tmp_path):
    case, path = observed_trajectory(tmp_path)
    item = load_dynamics_trajectory(case, path)
    streamed = load_dynamics_dataset(tmp_path, stream=True)[0]
    for name in OBSERVATION_FIELDS:
        np.testing.assert_array_equal(streamed.transition(1).observations[name], item.transition(1).observations[name])
        np.testing.assert_array_equal(streamed.observations[name], item.observations[name])
    observed_batch = transition_batch(item, 1, "cpu")
    legacy_batch = transition_batch(replace(item, observations=None), 1, "cpu")
    assert observed_batch.keys() == legacy_batch.keys()
    assert not set(OBSERVATION_FIELDS).intersection(observed_batch)
    assert "observations" not in observed_batch
    for name in observed_batch:
        if isinstance(observed_batch[name], torch.Tensor):
            torch.testing.assert_close(observed_batch[name], legacy_batch[name], rtol=0, atol=0)
        else:
            assert observed_batch[name] == legacy_batch[name]


def test_sparse_recordings_do_not_treat_last_native_step_observations_as_interval_totals(tmp_path):
    case, path = observed_trajectory(tmp_path)
    selected_frames = [0, 2, 3]

    def select_frames(arrays):
        offsets = arrays["native_node_offsets"]
        counts = np.diff(offsets)[selected_frames]
        for name in tuple(arrays):
            if name.startswith("native_nodal_"):
                arrays[name] = np.concatenate([arrays[name][offsets[frame]:offsets[frame + 1]] for frame in selected_frames])
        for name in ("times", "corner_positions", "corner_velocities", "cut_labels", "native_piece_ids",
                     "native_corner_vertex_ids", *(f"native_{field}" for field in OBSERVATION_FIELDS)):
            arrays[name] = arrays[name][selected_frames]
        arrays["native_node_offsets"] = np.r_[0, np.cumsum(counts)]

    update_archive(path, select_frames)
    item = load_dynamics_trajectory(case, path)
    np.testing.assert_array_equal(item.observations["single_step_coverage"], [False, False, True])
    first = item.transition(0)
    assert first.dt == .25
    assert first.observations["contact_observed"].any()
    assert first.observations["single_step_coverage"] is False
    np.testing.assert_allclose(first.observations["applied_delta_velocities"], .02)
    assert item.transition(1).observations["single_step_coverage"] is True


@pytest.mark.parametrize("corruption", [
    "missing_field", "missing_all_fields", "case_schema_missing", "provenance_schema_missing", "unsupported_schema",
    "vector_shape", "mask_dtype", "numeric_dtype", "nonfinite_unobserved", "negative_observed_stress",
    "zero_observed_threshold", "active_unobserved_constraint", "initial_contact_observed", "initial_stress_observed",
])
def test_malformed_or_unproven_observations_are_rejected(tmp_path, corruption):
    case, path = observed_trajectory(tmp_path)
    if corruption in {"case_schema_missing", "unsupported_schema"}:
        def change_case(arrays):
            metadata = json.loads(str(arrays["metadata_json"].item()))
            if corruption == "case_schema_missing":
                metadata.pop("native_observation_schema")
            else:
                metadata["native_observation_schema"] = "unsupported"
            arrays["metadata_json"] = np.asarray(json.dumps(metadata))

        update_archive(case, change_case)

    def change_trajectory(arrays):
        provenance = json.loads(str(arrays["metadata_json"].item()))
        provenance["source_case_sha256"] = file_hash(case)
        if corruption == "provenance_schema_missing":
            provenance.pop("native_observation_schema")
        elif corruption == "unsupported_schema":
            provenance["native_observation_schema"] = "unsupported"
        arrays["metadata_json"] = np.asarray(json.dumps(provenance))
        if corruption == "missing_field":
            arrays.pop("native_applied_delta_velocities")
        elif corruption == "missing_all_fields":
            for name in OBSERVATION_FIELDS:
                arrays.pop(f"native_{name}")
        elif corruption == "vector_shape":
            arrays["native_applied_delta_velocities"] = arrays["native_applied_delta_velocities"].reshape(4, -1, 3)
        elif corruption == "mask_dtype":
            arrays["native_contact_observed"] = arrays["native_contact_observed"].astype(np.int8)
        elif corruption == "numeric_dtype":
            arrays["native_fracture_stress_thresholds"] = arrays["native_fracture_stress_thresholds"].astype(np.int64)
        elif corruption == "nonfinite_unobserved":
            arrays["native_fracture_stress_eigenvalues"][0, 0] = np.nan
        elif corruption == "negative_observed_stress":
            arrays["native_fracture_stress_eigenvalues"][1, 0] = -1
        elif corruption == "zero_observed_threshold":
            arrays["native_fracture_stress_thresholds"][1, 0] = 0
        elif corruption == "active_unobserved_constraint":
            arrays["native_contact_observed"][2, 0, 0] = False
        elif corruption == "initial_contact_observed":
            arrays["native_contact_observed"][0, 0, 0] = True
        elif corruption == "initial_stress_observed":
            arrays["native_fracture_stress_observed"][0, 0] = True
            arrays["native_fracture_stress_thresholds"][0, 0] = 10

    update_archive(path, change_trajectory)
    with pytest.raises(ValueError, match="native observation|native observations|native constraint|native state|native fracture"):
        load_dynamics_trajectory(case, path)


def test_unobserved_finite_values_remain_unavailable_instead_of_becoming_physical_evidence(tmp_path):
    case, path = observed_trajectory(tmp_path)

    def unobserved(arrays):
        arrays["native_fracture_stress_eigenvalues"][2, 2] = -1
        arrays["native_fracture_stress_thresholds"][2, 2] = 0

    update_archive(path, unobserved)
    item = load_dynamics_trajectory(case, path)
    observation = item.transition(1).observations
    assert not observation["fracture_stress_observed"][2]
    assert observation["fracture_stress_eigenvalues"][2] == -1


def test_observation_switch_does_not_change_the_physical_condition_or_mutate_provenance(tmp_path):
    captured = observed_trajectory(tmp_path / "captured")
    uncaptured = recorded_trajectory(tmp_path / "uncaptured")
    legacy = recorded_trajectory(tmp_path / "legacy")
    for (case, path), enabled in ((captured, True), (uncaptured, False)):
        def update_case(arrays):
            metadata = json.loads(str(arrays["metadata_json"].item()))
            metadata["conditioning_context"]["scene_options"]["record_observations"] = enabled
            metadata["conditioning_context_sha256"] = context_digest(metadata["conditioning_context"])
            arrays["metadata_json"] = np.asarray(json.dumps(metadata))

        update_archive(case, update_case)
        with np.load(case, allow_pickle=False) as archive:
            metadata = json.loads(str(archive["metadata_json"].item()))

        def update_provenance(arrays):
            provenance = json.loads(str(arrays["metadata_json"].item()))
            provenance.update(source_case_sha256=file_hash(case),
                              conditioning_context_sha256=metadata["conditioning_context_sha256"])
            arrays["metadata_json"] = np.asarray(json.dumps(provenance))

        update_archive(path, update_provenance)
    observed = load_dynamics_trajectory(*captured)
    unobserved = load_dynamics_trajectory(*uncaptured)
    original = load_dynamics_trajectory(*legacy)
    assert observed.condition_digest == unobserved.condition_digest == original.condition_digest
    assert observed.metadata["conditioning_context"]["scene_options"]["record_observations"] is True
    assert unobserved.metadata["conditioning_context"]["scene_options"]["record_observations"] is False
    assert observed.metadata["source_hashes"] != unobserved.metadata["source_hashes"]
    assert file_hash(captured[0]) != file_hash(uncaptured[0])
    assert observed.observations is not None and unobserved.observations is None


def solver_observed_trajectory(directory):
    case, path = observed_trajectory(directory)

    def update_case(arrays):
        metadata = json.loads(str(arrays["metadata_json"].item()))
        metadata["native_solver_observation_schema"] = SOLVER_OBSERVATION_SCHEMA
        arrays["metadata_json"] = np.asarray(json.dumps(metadata))

    update_archive(case, update_case)

    def update_trajectory(arrays):
        frames, tetrahedra = arrays["corner_velocities"].shape[:2]
        for name in SOLVER_OBSERVATION_FIELDS:
            shape = (frames, tetrahedra, 4, 3) if name in SOLVER_VECTOR_FIELDS else (
                (frames, tetrahedra, 4) if name in SOLVER_CORNER_FIELDS else (frames, tetrahedra))
            values = np.zeros(shape, dtype=bool if name in SOLVER_BOOLEAN_FIELDS else float)
            if name in {"mpcg_observed", "mpcg_stopping_observed", "mpcg_rhs_positive", "mpcg_masses", "mpcg_rhs_norm_squared"}:
                values[1:] = 1
            elif name == "mpcg_epsilon":
                values[1:] = np.float32(1e-3)
            elif name == "mpcg_max_iterations":
                values[1:] = 100
            elif name == "mpcg_tolerance_squared":
                values[1:] = np.float32(np.float32(1e-3)**2)
            arrays[f"native_{name}"] = values
        provenance = json.loads(str(arrays["metadata_json"].item()))
        provenance.update(source_case_sha256=file_hash(case), native_solver_observation_schema=SOLVER_OBSERVATION_SCHEMA)
        arrays["metadata_json"] = np.asarray(json.dumps(provenance))

    update_archive(path, update_trajectory)
    return case, path


def test_actual_solver_observations_are_optional_readonly_and_never_network_inputs(tmp_path):
    case, path = solver_observed_trajectory(tmp_path)
    item = load_dynamics_trajectory(case, path)
    assert item.metadata["native_solver_observation_schema"] == SOLVER_OBSERVATION_SCHEMA
    streamed = load_dynamics_dataset(tmp_path, stream=True)[0]
    for name in SOLVER_OBSERVATION_FIELDS:
        np.testing.assert_array_equal(item.transition(1).observations[name], streamed.transition(1).observations[name])
        assert not item.observations[name].flags.writeable
    with_observations = transition_batch(item, 1, "cpu")
    without_observations = transition_batch(replace(item, observations=None), 1, "cpu")
    assert with_observations.keys() == without_observations.keys()
    assert not set(SOLVER_OBSERVATION_FIELDS).intersection(with_observations)
    for key, value in with_observations.items():
        if isinstance(value, torch.Tensor):
            torch.testing.assert_close(value, without_observations[key], rtol=0, atol=0)


@pytest.mark.parametrize("corruption", ["missing", "schema", "negative_norm", "iterations", "initial", "mass", "mask", "rhs_flag"])
def test_solver_observation_corruption_is_rejected_instead_of_becoming_evidence(tmp_path, corruption):
    case, path = solver_observed_trajectory(tmp_path)

    def corrupt(arrays):
        if corruption == "missing":
            arrays.pop("native_mpcg_solution")
        elif corruption == "schema":
            metadata = json.loads(str(arrays["metadata_json"].item()))
            metadata.pop("native_solver_observation_schema")
            arrays["metadata_json"] = np.asarray(json.dumps(metadata))
        elif corruption == "negative_norm":
            arrays["native_mpcg_true_norm_squared"][1, 0] = -1
        elif corruption == "iterations":
            arrays["native_mpcg_iterations"][1, 0] = 101
        elif corruption == "initial":
            arrays["native_mpcg_observed"][0] = True
            arrays["native_mpcg_masses"][0] = 1
        elif corruption == "mass":
            arrays["native_mpcg_masses"][1, 0, 0] = 0
        elif corruption == "mask":
            arrays["native_mpcg_observed"] = arrays["native_mpcg_observed"].astype(float)
        elif corruption == "rhs_flag":
            arrays["native_mpcg_rhs_positive"][1, 0] = False

    update_archive(path, corrupt)
    with pytest.raises(ValueError, match="native solver|initial native state"):
        load_dynamics_trajectory(case, path)
