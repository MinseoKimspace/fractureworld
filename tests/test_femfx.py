import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from fracture_world.data import load_prepared_case
from fracture_world.femfx import (
    context_digest,
    create_trial,
    generate_dataset,
    initial_conditions,
    native_directory,
    reference_surface,
    validate_specification,
)
from fracture_world.mesh import VolumeMesh
from test_dynamics_data import recorded_trajectory


def test_trial_impulses_preserve_the_explicit_loading_and_units(tmp_path):
    trial = create_trial(tmp_path)
    mesh = VolumeMesh.load(trial["mesh"])
    specification = json.loads(Path(trial["specification"]).read_text(encoding="utf-8"))
    material, fixed = validate_specification(mesh, specification)
    assert mesh.volumes.sum() == pytest.approx(0.12 * 0.04 * 0.04)
    assert trial["parameter_status"] == "uncalibrated_trial"
    velocities, mask, impacts = initial_conditions(mesh, material, fixed, specification["cases"][-1])
    mass = np.zeros(len(mesh.nodes))
    np.add.at(mass, mesh.tets.ravel(), np.repeat(mesh.volumes * material["rest_density"] / 4, 4))
    expected = np.sum([value["impulse"] for value in specification["cases"][-1]["vertex_impulses"]], axis=0)
    np.testing.assert_allclose((mass[:, None] * velocities).sum(axis=0), expected)
    np.testing.assert_allclose((impacts[:, 3:6] * impacts[:, 6:]).sum(axis=0), expected)
    assert not velocities[mask].any()
    assert mask.sum() == len(fixed)
    assert context_digest({"a": 1, "b": [2]}) == context_digest({"b": [2], "a": 1})
    specification["cases"][0]["vertex_impulses"][0]["vertex"] = int(fixed[0])
    with pytest.raises(ValueError, match="fixed vertex"):
        validate_specification(mesh, specification)


def test_reference_surface_uses_both_deformed_sides_of_a_crack():
    mesh = VolumeMesh([[0, 0, 0], [1, 0, 0], [0, 1, 0], [0, 0, 1], [1, 1, 1]],
                      [[0, 1, 2, 3], [1, 2, 3, 4]])
    corners = mesh.nodes[mesh.tets].copy()
    intact = reference_surface(mesh, corners, [0])
    corners[1] += [0.1, 0.1, 0.1]
    opened = reference_surface(mesh, corners, [1])
    assert len(opened["faces"]) == len(intact["faces"]) + 2
    left, right = np.asarray(opened["vertices"])[-6:].reshape(2, 3, 3)
    np.testing.assert_allclose(right.mean(axis=0) - left.mean(axis=0), [0.1, 0.1, 0.1])


@pytest.mark.parametrize("observed", [False, True])
def test_rejected_replay_export_cannot_leave_discoverable_training_archives(tmp_path, monkeypatch, observed):
    import fracture_world.femfx as femfx
    from fracture_world.dynamics_data import load_dynamics_dataset

    case, trajectory = recorded_trajectory(tmp_path / "source")
    mesh = VolumeMesh.load(case.parent.parent / "mesh.npz")
    with np.load(case, allow_pickle=False) as archive:
        metadata = json.loads(str(archive["metadata_json"].item()))
    with np.load(trajectory, allow_pickle=False) as archive:
        arrays = {key: np.array(archive[key], copy=True) for key in archive.files}
    material = metadata["conditioning_context"]["material"]
    masses = np.bincount(mesh.tets.ravel(), weights=np.repeat(mesh.volumes * material["rest_density"] / 4, 4))

    class RecordedScene:
        def __init__(self, *args, **kwargs):
            self.frame = 0

        def state(self):
            frame = self.frame
            selected = slice(*arrays["native_node_offsets"][frame:frame + 2])
            state = {"time": float(arrays["times"][frame]), "warnings": 0,
                     "original_tet_ids": np.arange(len(mesh.tets)),
                     "face_vertices": mesh.faces.vertices[mesh.internal_faces],
                     "cut_labels": arrays["cut_labels"][frame],
                     "corner_positions": arrays["corner_positions"][frame],
                     "corner_velocities": arrays["corner_velocities"][frame],
                     "corner_vertex_ids": arrays["native_corner_vertex_ids"][frame],
                     "tet_piece_ids": arrays["native_piece_ids"][frame],
                     "initial_vertex_masses": masses, "fracture_times": arrays["times"][1:]}
            for name in ("rest_positions", "positions", "velocities", "masses", "kinematic", "flags", "original_ids"):
                state[f"nodal_{name}"] = arrays[f"native_nodal_{name}"][selected]
            state["corner_kinematic"] = state["nodal_kinematic"][state["corner_vertex_ids"]]
            state["nodal_piece_ids"] = np.zeros(len(state["nodal_masses"]), dtype=int)
            for cell, piece in zip(state["corner_vertex_ids"], state["tet_piece_ids"], strict=True):
                state["nodal_piece_ids"][cell] = piece
            if observed:
                shape = (len(mesh.tets), 4)
                state.update(pre_contact_velocities=np.full((*shape, 3), frame * .1),
                             applied_delta_velocities=np.full((*shape, 3), frame * .2),
                             stabilization_delta_positions=np.full((*shape, 3), frame * .003),
                             contact_observed=np.full(shape, frame > 0, dtype=bool),
                             constraint_solve_active=np.full(shape, frame > 0, dtype=bool),
                             fracture_stress_eigenvalues=np.full(len(mesh.tets), frame * 2.),
                             fracture_stress_thresholds=np.full(len(mesh.tets), 10.),
                             fracture_stress_observed=np.full(len(mesh.tets), frame > 0, dtype=bool))
            return state

        def step(self):
            self.frame += 1
            return self.state()

        def settings(self):
            return metadata["native_settings"]

        def close(self):
            pass

    capabilities = dict(constraint_delta_observation=True, observation_schema=femfx.OBSERVATION_SCHEMA) if observed else {}
    native = SimpleNamespace(Scene=RecordedScene, capabilities=lambda: capabilities)
    monkeypatch.setattr(femfx, "load_native", lambda _: (native, {"binding_sha256": "0" * 64}))
    export = femfx.export_reference_replay

    def fail_first_replay(mesh, states, cuts, metadata, path):
        if metadata["case"]["id"] == "rejected":
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text("incomplete replay", encoding="utf-8")
            raise ValueError("simulated replay export failure")
        return export(mesh, states, cuts, metadata, path)

    monkeypatch.setattr(femfx, "export_reference_replay", fail_first_replay)
    specification = {"schema": femfx.SCHEMA, "parameter_status": "uncalibrated_trial", "material": material,
                     "fixed_vertices": [], "dt": .125, "steps": 3, "gravity": [0, 0, 0],
                     "cases": [{**metadata["case"], "id": identifier} for identifier in ("rejected", "accepted")]}
    specification_path = tmp_path / "specification.json"
    specification_path.write_text(json.dumps(specification), encoding="utf-8")
    output = tmp_path / "generated"
    report = generate_dataset(case.parent.parent / "mesh.npz", specification_path, output, 1., skip_invalid=True)
    assert [item["case"] for item in report["cases"]] == ["accepted"]
    assert [item["case"] for item in report["rejected_cases"]] == ["rejected"]
    assert not list(output.rglob("rejected.*"))
    assert [item.case_id for item in load_dynamics_dataset(output)] == ["accepted"]
    with np.load(output / "trajectories" / "accepted.npz", allow_pickle=False) as archive:
        if observed:
            assert not archive["native_contact_observed"][0].any()
            assert archive["native_contact_observed"][1:].all()
            assert archive["native_pre_contact_velocities"].shape == (4, len(mesh.tets), 4, 3)
            np.testing.assert_allclose(archive["native_pre_contact_velocities"][-1], .3)
            np.testing.assert_allclose(archive["native_applied_delta_velocities"][-1], .6)
            np.testing.assert_allclose(archive["native_stabilization_delta_positions"][-1], .009)
            np.testing.assert_allclose(archive["native_fracture_stress_eigenvalues"][-1], 6.)
        else:
            assert not set(archive.files).intersection(f"native_{name}" for name in femfx.OBSERVATION_FIELDS)


def test_native_capture_requires_complete_fields_and_preserves_unavailable_masks():
    from fracture_world.femfx import _observation_arrays

    assert _observation_arrays({}, 2) == {}
    with pytest.raises(RuntimeError, match="complete capture"):
        _observation_arrays({}, 2, required=True)
    state = dict(pre_contact_velocities=np.zeros((2, 4, 3)), applied_delta_velocities=np.zeros((2, 4, 3)),
                 stabilization_delta_positions=np.zeros((2, 4, 3)), contact_observed=np.zeros((2, 4), dtype=bool),
                 constraint_solve_active=np.zeros((2, 4), dtype=bool), fracture_stress_eigenvalues=np.zeros(2),
                 fracture_stress_thresholds=np.zeros(2), fracture_stress_observed=np.zeros(2, dtype=bool), step=0)
    observed = _observation_arrays(state, 2, required=True)
    assert not observed["contact_observed"].any()
    assert not observed["fracture_stress_observed"].any()
    state["contact_observed"][0, 0] = True
    with pytest.raises(RuntimeError, match="preceding observed update"):
        _observation_arrays(state, 2, required=True)


@pytest.mark.skipif(not (native_directory() / "manifest.json").exists(), reason="native Windows FEMFX reference is not built")
def test_free_native_step_matches_reconstructed_frozen_elastic_balance():
    from scipy.sparse.linalg import spsolve

    from fracture_world.dynamics_mechanics import FrozenElasticBalance
    from fracture_world.femfx import load_native

    native, _ = load_native()
    mesh = VolumeMesh([[0., 0., 0.], [1., 0., 0.], [0., 1., 0.], [0., 0., 1.]], [[0, 1, 2, 3]])
    material = dict(rest_density=1., youngs_modulus=10., poissons_ratio=.25, fracture_stress_threshold=100.)
    deformed = mesh.nodes * [1.2, 1., 1.]
    options = dict(enable_fracture=False, friction_coefficient=0., rayleigh_mass_damping=0., rayleigh_stiffness_damping=0.)
    scene = native.Scene(mesh.nodes, mesh.tets, material, np.zeros_like(deformed), np.zeros(4, dtype=bool),
                         .01, [0., 0., 0.], 1, options, deformed_positions=deformed)
    try:
        initial = scene.state()
        final = scene.step()
        balance = FrozenElasticBalance(mesh, material, initial["corner_positions"], initial["corner_velocities"],
                                       float(np.float32(.01)), np.zeros(3), np.zeros(0, dtype=bool), np.zeros(4, dtype=bool))
        expected = spsolve(balance.matrix, balance.rhs).reshape(-1, 3)
        native_velocity = final["pre_contact_velocities"].reshape(-1, 3)
        np.testing.assert_allclose(native_velocity, expected, rtol=128 * np.finfo(np.float32).eps,
                                   atol=np.finfo(np.float32).eps)
        diagnostics = balance.diagnostics(final["corner_positions"], final["pre_contact_velocities"])
        assert diagnostics["maximum_free_normalized_residual"] < scene.settings()["epsilon_cg"]
        assert final["contact_observed"].all()
        assert not final["constraint_solve_active"].any()
        assert not final["applied_delta_velocities"].any()
    finally:
        scene.close()


@pytest.mark.skipif(not (native_directory() / "manifest.json").exists(), reason="native Windows FEMFX reference is not built")
def test_native_observation_records_actual_solver_deltas_and_leaves_physics_unchanged():
    from fracture_world.femfx import OBSERVATION_SCHEMA, load_native

    native, manifest = load_native()
    assert native.capabilities()["observation_schema"] == OBSERVATION_SCHEMA
    assert manifest["compiled_source_instrumented"] is True
    assert manifest["reference_source_modified"] is True
    assert manifest["physics_arithmetic_modified"] is False
    assert manifest["observation"]["upstream_checkout_modified"] is False
    nodes = np.array([[0., .001, 0.], [.05, .001, 0.], [0., .051, 0.], [0., .001, .05]])
    tets = np.array([[0, 1, 2, 3]], dtype=np.int64)
    material = dict(rest_density=1000., youngs_modulus=10000., poissons_ratio=.25, fracture_stress_threshold=10000.)
    velocity = np.broadcast_to([0., -1., 0.], nodes.shape).copy()
    fixed = np.zeros(4, dtype=bool)
    options = dict(enable_fracture=False, friction_coefficient=0., rayleigh_mass_damping=0.,
                   rayleigh_stiffness_damping=0., collision_bounds=[-1e6, 1e6, 0., 1e6, -1e6, 1e6])
    histories = []
    active_corrections = 0
    for enabled in (True, False):
        scene = native.Scene(nodes, tets, material, velocity, fixed, .001, [0., -9.81, 0.], 1,
                             {**options, "record_observations": enabled})
        states = [scene.state()]
        assert not states[0]["contact_observed"].any()
        try:
            for _ in range(20):
                state = scene.step()
                if enabled:
                    mask = state["contact_observed"]
                    assert mask.all()
                    assert state["mpcg_observed"].all()
                    assert state["mpcg_stopping_observed"].all()
                    np.testing.assert_array_equal(state["mpcg_solution"], state["pre_contact_velocities"])
                    expected_velocity = (state["pre_contact_velocities"].astype(np.float32) +
                                         state["applied_delta_velocities"].astype(np.float32)).astype(float)
                    np.testing.assert_array_equal(state["corner_velocities"][mask], expected_velocity[mask])
                    expected_position = ((states[-1]["corner_positions"].astype(np.float32) +
                                          state["stabilization_delta_positions"].astype(np.float32)) +
                                         np.float32(.001) * state["corner_velocities"].astype(np.float32)).astype(float)
                    np.testing.assert_array_equal(state["corner_positions"][mask], expected_position[mask])
                    active_corrections += int(np.any(np.abs(state["applied_delta_velocities"]) > 1e-8))
                else:
                    assert not state["contact_observed"].any()
                    assert not state["mpcg_observed"].any()
                    assert not state["mpcg_stopping_observed"].any()
                states.append(state)
        finally:
            scene.close()
        histories.append(states)
    assert active_corrections > 0
    for observed, unobserved in zip(*histories, strict=True):
        np.testing.assert_array_equal(observed["corner_positions"], unobserved["corner_positions"])
        np.testing.assert_array_equal(observed["corner_velocities"], unobserved["corner_velocities"])
        np.testing.assert_array_equal(observed["cut_labels"], unobserved["cut_labels"])


@pytest.mark.skipif(not (native_directory() / "manifest.json").exists(), reason="native Windows FEMFX reference is not built")
@pytest.mark.parametrize("kinematic,gravity", [(False, -9.81), (True, -9.81), (False, 0.)])
def test_native_mpcg_observation_separates_actual_residual_and_recursive_stopping(kinematic, gravity):
    from fracture_world.femfx import load_native

    native, _ = load_native()
    capabilities = native.capabilities()
    assert capabilities["elastic_solver_observation"] is True
    assert capabilities["solver_observation_schema"] == "fracture-world.femfx-mpcg-observation.v1"
    nodes = np.array([[0., 0., 0.], [1., 0., 0.], [0., 1., 0.], [0., 0., 1.]])
    tets = np.array([[0, 1, 2, 3]], dtype=np.int64)
    material = dict(rest_density=1000., youngs_modulus=10000., poissons_ratio=.25, fracture_stress_threshold=10000.)
    fixed = np.array([kinematic, False, False, False])
    options = dict(enable_fracture=False, friction_coefficient=0., rayleigh_mass_damping=0.,
                   rayleigh_stiffness_damping=0., collision_bounds=[-1e6, 1e6, -1e6, 1e6, -1e6, 1e6])
    scene = native.Scene(nodes, tets, material, np.zeros_like(nodes), fixed, .001, [0., gravity, 0.], 1, options)
    try:
        assert not scene.state()["mpcg_observed"].any()
        state = scene.step()
        assert state["mpcg_observed"].all()
        assert state["mpcg_stopping_observed"].all()
        np.testing.assert_array_equal(state["mpcg_kinematic"][0], fixed)
        np.testing.assert_array_equal(state["mpcg_solution"], state["pre_contact_velocities"])
        assert state["mpcg_residual"].shape == (1, 4, 3)
        assert np.isfinite(state["mpcg_residual"]).all()
        np.testing.assert_array_equal(state["mpcg_masses"], state["corner_masses"])
        rhs = state["mpcg_rhs_norm_squared"][0]
        epsilon = np.float32(state["mpcg_epsilon"][0])
        expected_tolerance = float(np.float32(np.float32(epsilon * epsilon) * np.float32(rhs)))
        assert state["mpcg_tolerance_squared"][0] == expected_tolerance
        assert bool(state["mpcg_rhs_positive"][0]) == (rhs > 0.)
        iterations = state["mpcg_iterations"][0]
        maximum = state["mpcg_max_iterations"][0]
        assert 0 <= iterations <= maximum
        if rhs > 0. and iterations < maximum:
            assert state["mpcg_recursive_norm_squared"][0] <= expected_tolerance
        if rhs == 0.:
            assert iterations == 0
    finally:
        scene.close()


@pytest.mark.skipif(not (native_directory() / "manifest.json").exists(), reason="native Windows FEMFX reference is not built")
def test_observing_native_fracture_preserves_original_states_cuts_and_supports(tmp_path):
    from fracture_world.femfx import load_native

    native, _ = load_native()
    trial = create_trial(tmp_path / "input")
    mesh = VolumeMesh.load(trial["mesh"])
    specification = json.loads(Path(trial["specification"]).read_text())
    fixed = np.asarray(specification["fixed_vertices"], dtype=int)
    velocities, kinematic, _ = initial_conditions(mesh, specification["material"], fixed, specification["cases"][-1])
    histories = []
    for observed in (True, False):
        options = {**specification["scene_options"], "record_observations": observed}
        scene = native.Scene(mesh.nodes, mesh.tets, specification["material"], velocities, kinematic,
                             specification["dt"], specification["gravity"], 1, options)
        states = [scene.state()]
        try:
            states.extend(scene.step() for _ in range(specification["steps"]))
        finally:
            scene.close()
        histories.append(states)
    assert histories[0][-1]["cut_labels"].any()
    assert any(state["fracture_stress_observed"].any() for state in histories[0][1:])
    for observed, unobserved in zip(*histories, strict=True):
        for name in ("corner_positions", "corner_velocities", "cut_labels", "corner_kinematic"):
            np.testing.assert_array_equal(observed[name], unobserved[name])


@pytest.mark.skipif(not (native_directory() / "manifest.json").exists(), reason="native Windows FEMFX reference is not built")
def test_actual_femfx_generates_complete_irreversible_native_fracture_labels(tmp_path, capsys):
    trial = create_trial(tmp_path / "input")
    result = generate_dataset(trial["mesh"], trial["specification"], tmp_path / "reference", 1.0)
    assert len(result["cases"]) == 4
    assert sum(row["cut_facets"] for row in result["cases"]) > 0
    assert result["support_release_corner_events"] > 0
    for row in result["cases"]:
        path = tmp_path / "reference" / "cases" / f"{row['case']}.npz"
        case = load_prepared_case(path)
        assert case.valid_facets.all()
        assert case.known_cells.all()
        assert case.metadata["source"] == "FEMFX"
        with np.load(tmp_path / "reference" / "trajectories" / f"{row['case']}.npz") as trajectory:
            assert len(trajectory["times"]) == 51
            assert (np.diff(trajectory["cut_labels"].astype(int), axis=0) >= 0).all()
            np.testing.assert_array_equal(trajectory["cut_labels"][-1], case.cut_labels)
            support = trajectory["corner_kinematic"]
            np.testing.assert_array_equal(trajectory["support_release_labels"], support[:-1] & ~support[1:])
            np.testing.assert_array_equal(trajectory["recorded_step_ids"], np.arange(51))
            assert int(trajectory["support_release_labels"].sum()) == row["support_release_corner_events"]
            assert not trajectory["native_contact_observed"][0].any()
            assert not trajectory["native_fracture_stress_observed"][0].any()
            assert trajectory["native_fracture_stress_observed"][1:].any()
            observed_stress = trajectory["native_fracture_stress_observed"]
            assert (trajectory["native_fracture_stress_eigenvalues"][observed_stress] >= 0).all()
            assert (trajectory["native_fracture_stress_thresholds"][observed_stress] > 0).all()
        replay = json.loads((tmp_path / "reference" / "replays" / f"{row['case']}.json").read_text())
        assert replay["metadata"]["neural_prediction"] is False
        assert replay["metadata"]["proximal_fracture_correction"] is False
        assert replay["frames"][-1]["metrics"]["cut_facets"] == row["cut_facets"]
    assert '"cut_facets"' in capsys.readouterr().out
