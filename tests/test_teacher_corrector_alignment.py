from dataclasses import replace
import json
from types import SimpleNamespace

import numpy as np
from numpy.testing import assert_array_equal
import pytest

from fracture_world import dynamics_teacher_audit as audit
from fracture_world.dynamics_data import load_dynamics_dataset
from fracture_world.dynamics_matched import _initial_observation
from fracture_world.dynamics_runtime import RolloutConfig, _problem, step_contacts
from fracture_world.femfx import generate_dataset, load_native, native_directory
from fracture_world.mesh import VolumeMesh


@pytest.fixture(scope="module")
def native_floor_dataset(tmp_path_factory):
    if not (native_directory() / "manifest.json").exists():
        pytest.skip("native FEMFX reference is not built")
    native, _ = load_native()
    if not native.capabilities().get("current_elastic_state"):
        pytest.skip("native FEMFX current elastic state capture is unavailable")
    directory = tmp_path_factory.mktemp("teacher-corrector-alignment")
    mesh = VolumeMesh([[0., 0., 0.], [.05, 0., 0.], [0., .05, 0.], [0., 0., .05]], [[0, 1, 2, 3]])
    mesh_path = directory / "mesh.npz"
    mesh.save(mesh_path)
    cases = [dict(id=f"drop-{index + 1}", vertex_impulses=[],
                  initial_pose=dict(rotation_xyzw=[0., 0., 0., 1.], translation=[0., height, 0.]),
                  initial_linear_velocity=velocity)
             for index, (height, velocity) in enumerate(((.001, [0., -1., 0.]), (.003, [0., -1., 0.]),
                                                         (.001, [.2, -1.5, .1])))]
    specification = dict(schema="fracture-world.femfx.v1", parameter_status="uncalibrated_trial",
                         material=dict(rest_density=1000., youngs_modulus=10000., poissons_ratio=.25,
                                       fracture_stress_threshold=10000.),
                         fixed_vertices=[], dt=.001, steps=8, gravity=[0., -9.81, 0.], threads=1,
                         loading="gravity_driven_drop", floor_height_m=0., cases=cases,
                         scene_options=dict(rayleigh_mass_damping=0., rayleigh_stiffness_damping=0.,
                                            friction_coefficient=0., dist_contact_bias=0., dist_contact_threshold=1e-5,
                                            collision_bounds=[-1e6, 1e6, 0., 1e6, -1e6, 1e6]))
    specification_path = directory / "specification.json"
    specification_path.write_text(json.dumps(specification), encoding="utf-8")
    output = directory / "reference"
    generate_dataset(mesh_path, specification_path, output, 1., export_replay=False)
    return output


def source_config():
    return RolloutConfig(kinematic_rule="backward_euler", mechanics_mode="proximal", fracture_initiation=True)


def source_model():
    return SimpleNamespace(config=dict(velocity_scale=1., motion_prior="backward_euler"))


def test_native_teacher_passes_source_corrector_hard_contract_without_loading_network(native_floor_dataset, monkeypatch):
    def forbid_network(*args, **kwargs):
        raise AssertionError("teacher corrector audit must not load a neural checkpoint")

    monkeypatch.setattr(audit, "load_dynamics_model", forbid_network)
    report = audit.inspect_teacher_constraints(native_floor_dataset, config=source_config(), velocity_scale=1.)
    assert report["checkpoint_sha256"] is None
    assert report["totals"]["recorded_transitions"] == 24
    assert report["ideal_corrector_compatibility"]["status"] == "pass"
    assert report["totals"]["checks"]["linear_constraints"] == {"pass": 24, "fail": 0, "unknown": 0}
    raw_failures = []
    contact_steps = 0
    for case in report["cases"]:
        for row in case["transitions"]:
            linear = row["checks"]["linear_constraints"]
            assert linear["status"] == "pass"
            assert np.isfinite(linear["arithmetic_envelope_metres"])
            assert linear["arithmetic_envelope_metres"] > 0.
            assert linear["stabilization"]["geometry_tolerance_metres"] == source_config().geometry_tolerance
            assert linear["stabilization"]["teacher_future_used"] is False
            assert linear["violating_factors"] == {}
            contact_steps += int(linear["contact_constraints"] > 0)
            if linear["raw_analytical_status"] == "fail":
                raw_failures.append(linear["raw_analytical_factor_violations"])
    assert contact_steps > 0
    assert raw_failures
    assert any(any(name.startswith(("kinematic:", "contact:")) and value > 0 for name, value in factors.items())
               for factors in raw_failures)


def test_state_error_larger_than_publication_envelope_is_still_rejected(native_floor_dataset):
    item = load_dynamics_dataset(native_floor_dataset)[0]
    config, model = source_config(), source_model()
    dt = float(np.float32(item.metadata["conditioning_context"]["dt"]))
    baseline = audit._transition_checks(item, 0, dt, model, config, None, audit._physics(item.metadata["native_settings"]))
    assert baseline["linear_constraints"]["status"] == "pass"
    positions = item.positions.copy()
    positions[1, :, 0] += max(100. * baseline["linear_constraints"]["arithmetic_envelope_metres"], 10. * config.geometry_tolerance)
    corrupted = replace(item, positions=positions)
    checks = audit._transition_checks(corrupted, 0, dt, model, config, None, audit._physics(item.metadata["native_settings"]))
    assert checks["linear_constraints"]["status"] == "fail"
    assert any(name.startswith("kinematic:") for name in checks["linear_constraints"]["violating_factors"])


def test_matched_runtime_problem_uses_current_elastic_state_without_future_observations(native_floor_dataset):
    item = load_dynamics_dataset(native_floor_dataset)[0]
    frame, model, config = 1, source_model(), source_config()
    first = _initial_observation(item, frame, model)
    positions, velocities = item.positions.copy(), item.velocities.copy()
    positions[frame + 1:] += 100.
    velocities[frame + 1:] *= -100.
    cache = {name: values.copy() for name, values in item.elastic_state.items()}
    cache["rotations"][frame + 1:] = 0.
    changed = replace(item, positions=positions, velocities=velocities, elastic_state=cache, observations=None)
    second = _initial_observation(changed, frame, model)
    assert "observations" not in first.metadata
    assert "target_positions" not in first.metadata
    predicted_positions = first.positions + first.dt * first.velocities + first.dt**2 * first.gravity
    problems = []
    for initial in (first, second):
        contacts = step_contacts(initial, initial.positions, predicted_positions, initial.broken, config)
        problems.append(_problem(initial, initial.positions, initial.velocities, predicted_positions, initial.velocities,
                                 initial.broken, initial.fixed, contacts, model, config,
                                 previous_broken=initial.broken, previous_fixed=initial.fixed))
    assert_array_equal(problems[0].prior, problems[1].prior)
    assert_array_equal(problems[0].position_roundoff, problems[1].position_roundoff)
    assert_array_equal(problems[0].elastic_balance.matrix.toarray(), problems[1].elastic_balance.matrix.toarray())
    assert [factor.name for factor in problems[0].factors] == [factor.name for factor in problems[1].factors]
    for first_factor, second_factor in zip(problems[0].factors, problems[1].factors):
        assert_array_equal(first_factor.bound, second_factor.bound)


def test_sparse_native_contact_interval_is_unknown_instead_of_invented_substeps(native_floor_dataset, monkeypatch):
    item = load_dynamics_dataset(native_floor_dataset)[0]
    indices = np.array([0, 2, 3, 4, 5, 6, 7, 8])
    sparse = replace(item, positions=item.positions[indices], velocities=item.velocities[indices],
                     broken=item.broken[indices], fixed=item.fixed[indices], times=item.times[indices],
                     observations={name: values[indices] for name, values in item.observations.items()},
                     elastic_state={name: values[indices] for name, values in item.elastic_state.items()})
    monkeypatch.setattr(audit, "load_dynamics_dataset", lambda *args, **kwargs: [sparse])
    report = audit.inspect_teacher_constraints("unused", config=source_config(), velocity_scale=1.)
    first = report["cases"][0]["transitions"][0]
    assert first["consecutive_native_transition"] is False
    assert first["ideal_corrector_compatibility"]["status"] == "unknown"
    assert first["native_transition_consistency"]["status"] == "unknown"
    assert first["checks"]["linear_constraints"]["status"] == "unknown"
    assert first["checks"]["kinematics"]["status"] == "unknown"


def test_missing_contact_observations_do_not_certify_native_transition(native_floor_dataset, monkeypatch):
    item = load_dynamics_dataset(native_floor_dataset)[0]
    monkeypatch.setattr(audit, "load_dynamics_dataset", lambda *args, **kwargs: [replace(item, observations=None)])
    report = audit.inspect_teacher_constraints("unused", config=source_config(), velocity_scale=1.)
    contacts = [row for row in report["cases"][0]["transitions"] if row["checks"]["linear_constraints"]["contact_constraints"]]
    assert contacts
    for row in contacts:
        assert row["native_transition_consistency"]["status"] == "unknown"
        assert row["checks"]["native_elastic_solver"]["status"] == "unknown"
        assert row["checks"]["kinematics"]["status"] == "unknown"


def test_causally_advanced_cache_survives_current_initial_roundtrip(native_floor_dataset, tmp_path):
    from fracture_world.dynamics_arithmetic import float32_position_update
    from fracture_world.dynamics_initial import export_initial, load_initial, save_initial
    from fracture_world.dynamics_runtime import advance_elastic_state, current_elastic_state

    case = native_floor_dataset / "cases" / "drop-2.npz"
    trajectory = native_floor_dataset / "trajectories" / "drop-2.npz"
    initial = export_initial(case, trajectory, 0, tmp_path / "observed.npz", support_policy="free")
    next_velocities = initial.velocities + initial.dt * initial.gravity
    next_positions = float32_position_update(initial.positions, initial.dt, next_velocities)
    advanced = advance_elastic_state(initial, next_positions, next_velocities, initial.broken, initial.broken,
                                     initial.fixed, initial.fixed, start_state=initial.metadata["elastic_state"])
    current = replace(advanced, positions=next_positions, velocities=next_velocities, time=initial.time + initial.dt)
    output = tmp_path / "advanced.npz"
    save_initial(current, output)
    restored = load_initial(output)
    assert_array_equal(restored.positions, next_positions)
    assert_array_equal(restored.velocities, next_velocities)
    assert restored.metadata["elastic_state"] == current.metadata["elastic_state"]
    assert restored.metadata["elastic_state_origin"] == "causal current-state constitutive math"
    assert "origin" not in restored.metadata["elastic_state"]
    assert "native_topology_repacking_verified" not in restored.metadata["elastic_state"]
    assert current_elastic_state(restored, restored.positions, restored.broken, restored.fixed) is not None
    assert current_elastic_state(restored, initial.positions, initial.broken, initial.fixed) is None


def test_matched_methods_evaluate_identical_endpoint_with_one_common_energy_contract(native_floor_dataset, monkeypatch):
    import torch

    from fracture_world import dynamics_matched, dynamics_runtime
    from fracture_world.dynamics_arithmetic import float32_position_update

    item = load_dynamics_dataset(native_floor_dataset)[1]
    observation = _initial_observation(item, 0, source_model())
    next_velocities = observation.velocities + observation.dt * observation.gravity
    next_positions = float32_position_update(observation.positions, observation.dt, next_velocities)

    class FixedProposal(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.anchor = torch.nn.Parameter(torch.zeros((), dtype=torch.float64))
            self.config = source_model().config

        def forward(self, batch):
            return dict(positions=torch.as_tensor(next_positions), velocities=torch.as_tensor(next_velocities),
                        fracture_probability=torch.zeros(len(observation.broken), dtype=torch.float64),
                        support_release_probability=torch.zeros(len(observation.fixed), dtype=torch.float64))

    def same_endpoint(*args, **kwargs):
        diagnostic = dict(status="converged", solves=[], geometry=dict(valid=True), swept_geometry=dict(valid=True),
                          mechanics=dict(fracture_initiation=dict(status="pass")))
        solution = SimpleNamespace(positions=next_positions.copy(), velocities=next_velocities.copy(), converged=True,
                                   objective=0., diagnostics=dict(constraints_satisfied=True))
        return solution, diagnostic

    calls = []
    evaluate = dynamics_runtime.step_energy_diagnostics

    def capture(initial, positions, velocities, endpoint_positions, endpoint_velocities, *args, **kwargs):
        calls.append((endpoint_positions.copy(), endpoint_velocities.copy()))
        return evaluate(initial, positions, velocities, endpoint_positions, endpoint_velocities, *args, **kwargs)

    monkeypatch.setattr(dynamics_runtime, "_correct", same_endpoint)
    monkeypatch.setattr(dynamics_runtime, "step_energy_diagnostics", capture)
    config = replace(source_config(), fracture_samples=0, fracture_policy="marginal_search")
    report = dynamics_matched.compare_matched_step(observation, FixedProposal(), config)
    assert report["candidate_count"] == 1
    assert [record["method"] for record in report["candidates"]] == ["raw", "central", "consensus"]
    assert len(calls) == 3
    for positions, velocities in calls:
        assert_array_equal(positions, next_positions)
        assert_array_equal(velocities, next_velocities)
    energies = [record["diagnostic"]["energy"] for record in report["candidates"]]
    assert energies[0] == energies[1] == energies[2]
    assert energies[0]["evaluation_only"] is True
    assert energies[0]["evaluator_engine_calls"] == 0
    assert energies[0]["independent_nonlinear_balance"]["enforced_as_native_law"] is False
