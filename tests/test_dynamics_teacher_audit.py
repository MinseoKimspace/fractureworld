from dataclasses import replace
import json

import numpy as np
import pytest

from fracture_world import dynamics_teacher_audit as audit
from fracture_world.dynamics_data import DynamicsTrajectory, material_topology
from fracture_world.dynamics_runtime import RolloutConfig
from fracture_world.io import file_hash
from fracture_world.mesh import VolumeMesh
from test_dynamics_flow_compatibility import checkpoint_file


def analytical_trajectory(gravity=(0., 0., 0.)):
    mesh = VolumeMesh([[0., 0., 0.], [1., 0., 0.], [0., 1., 0.], [0., 0., 1.], [1., 1., 1.]],
                      [[0, 1, 2, 3], [4, 1, 2, 3]])
    topology = material_topology(mesh, 1.)
    times = np.arange(4) * .125
    gravity = np.array(gravity)
    offsets = .5 * (times * (times + .125))[:, None, None] * gravity
    positions = topology.rest_positions[None] + offsets
    velocities = np.broadcast_to(times[:, None, None] * gravity, positions.shape).copy()
    material = dict(rest_density=1., youngs_modulus=10., poissons_ratio=.25, fracture_stress_threshold=100.)
    context = dict(dt=.125, steps=3, gravity=gravity.tolist(), material=material)
    metadata = dict(conditioning_context=context, source_hashes={"teacher": "a" * 64},
                    native_settings=dict(rayleigh_mass_damping=0., rayleigh_stiffness_damping=0., friction_coefficient=0.))
    return DynamicsTrajectory(topology, positions, velocities, np.zeros((4, 1), dtype=bool),
                              np.zeros((4, 8), dtype=bool), times, gravity, material,
                              np.zeros((8, 3)), "analytic", "b" * 64, metadata)


def mechanical_checks(initial, transition, contacts, config):
    return dict(elastic_dynamics=dict(status="pass"), fracture_initiation=dict(status="pass"),
                kinematics=dict(status="pass"), native_elastic_solver=dict(status="pass"))


def inspect(monkeypatch, item, **kwargs):
    monkeypatch.setattr(audit, "load_dynamics_dataset", lambda *args, **options: [item])
    settings = dict(config=RolloutConfig(kinematic_rule="backward_euler"), velocity_scale=1.)
    settings.update(kwargs)
    return audit.inspect_teacher_constraints("unused", **settings)


def test_analytic_teacher_passes_exact_runtime_constraints_without_projection(monkeypatch, tmp_path):
    item = analytical_trajectory((0., -9.81, 0.))
    output = tmp_path / "audit.json"
    report = inspect(monkeypatch, item, output=output)
    assert report["status"] == "unknown"
    assert report["constraint_status"] == "pass"
    assert report["ideal_corrector_compatibility"]["status"] == "pass"
    assert report["native_transition_consistency"]["status"] == "unknown"
    assert report["totals"]["recorded_transitions"] == 3
    assert all(counts == {"pass": 3, "fail": 0, "unknown": 0}
               for name, counts in report["totals"]["checks"].items() if name != "native_elastic_solver")
    assert report["totals"]["checks"]["native_elastic_solver"] == {"pass": 0, "fail": 0, "unknown": 3}
    assert report["cases"][0]["source_hashes"] == item.metadata["source_hashes"]
    assert report["provenance"]["state_projection_used"] is False
    assert report["provenance"]["teacher_future_used_only_for_offline_audit"] is True
    assert report["provenance"]["native_engine_calls"] == 0
    assert json.loads(output.read_text()) == report


def test_missing_mechanics_does_not_hide_verified_geometric_constraint_checks(monkeypatch):
    report = inspect(monkeypatch, analytical_trajectory(), mechanics=lambda *args: {
        name: dict(status="unknown", reason="unavailable test evaluator")
        for name in ("elastic_dynamics", "fracture_initiation", "kinematics")})
    assert report["constraint_status"] == "unknown"
    assert report["status"] == "unknown"
    assert report["totals"]["checks"]["elastic_dynamics"] == {"pass": 0, "fail": 0, "unknown": 3}
    assert report["totals"]["checks"]["endpoint_geometry"] == {"pass": 3, "fail": 0, "unknown": 0}


def test_teacher_kinematic_violation_is_preserved_and_not_corrected(monkeypatch):
    item = analytical_trajectory()
    positions = item.positions.copy()
    positions[1:, :, 0] += .01
    report = inspect(monkeypatch, replace(item, positions=positions), mechanics=mechanical_checks)
    assert report["status"] == "fail"
    checks = report["cases"][0]["transitions"][0]["checks"]
    assert checks["linear_constraints"]["status"] == "fail"
    assert checks["linear_constraints"]["violating_factors"]
    assert checks["endpoint_geometry"]["status"] == "pass"
    assert report["totals"]["checks"]["linear_constraints"] == {"pass": 2, "fail": 1, "unknown": 0}
    assert report["ideal_corrector_compatibility"]["status"] == "fail"
    assert report["native_transition_consistency"]["status"] == "pass"
    assert report["cases"][0]["ideal_corrector_compatibility"]["status"] == "fail"
    assert report["cases"][0]["native_transition_consistency"]["status"] == "pass"


def test_sparse_teacher_intervals_remain_unknown_without_interpolation(monkeypatch):
    item = analytical_trajectory()
    indices = [0, 2, 3]
    item = replace(item, positions=item.positions[indices], velocities=item.velocities[indices],
                   broken=item.broken[indices], fixed=item.fixed[indices], times=item.times[indices])
    calls = []

    def mechanics(*args):
        calls.append(args)
        return mechanical_checks(*args)

    report = inspect(monkeypatch, item, mechanics=mechanics)
    rows = report["cases"][0]["transitions"]
    assert len(calls) == 1
    assert rows[0]["status"] == "unknown"
    assert rows[0]["initial_native_step"] == 0
    assert rows[0]["final_native_step"] == 2
    assert rows[1]["status"] == "pass"
    assert report["status"] == "unknown"
    assert report["totals"]["checks"]["linear_constraints"] == {"pass": 1, "fail": 0, "unknown": 1}


def test_missing_or_incompatible_teacher_physics_is_not_assumed_zero(monkeypatch):
    item = analytical_trajectory()
    missing = replace(item, metadata={**item.metadata, "native_settings": {}})
    report = inspect(monkeypatch, missing, mechanics=mechanical_checks)
    assert report["status"] == "unknown"
    assert report["totals"]["checks"]["teacher_physics"] == {"pass": 0, "fail": 0, "unknown": 3}
    settings = {**item.metadata["native_settings"], "rayleigh_mass_damping": .15}
    incompatible = replace(item, metadata={**item.metadata, "native_settings": settings})
    report = inspect(monkeypatch, incompatible, mechanics=mechanical_checks)
    assert report["status"] == "fail"
    assert report["totals"]["checks"]["teacher_physics"] == {"pass": 0, "fail": 3, "unknown": 0}


def test_inferred_contacts_do_not_certify_unrecorded_stabilization(monkeypatch):
    item = analytical_trajectory()
    metadata = {**item.metadata, "conditioning_context": {**item.metadata["conditioning_context"], "floor_height_m": 0.}}
    report = inspect(monkeypatch, replace(item, metadata=metadata), mechanics=mechanical_checks)
    checks = report["cases"][0]["transitions"][0]["checks"]
    assert checks["linear_constraints"]["numerical_status"] == "pass"
    assert checks["linear_constraints"]["contact_constraints"] > 0
    assert checks["linear_constraints"]["status"] == "unknown"
    assert report["constraint_status"] == "unknown"


def test_contact_uncertainty_does_not_mask_known_corrector_constraint_failure(monkeypatch):
    item = analytical_trajectory()
    positions = item.positions.copy()
    positions[1:, :, 0] += .01
    metadata = {**item.metadata, "conditioning_context": {**item.metadata["conditioning_context"], "floor_height_m": 0.}}
    report = inspect(monkeypatch, replace(item, metadata=metadata, positions=positions), mechanics=mechanical_checks)
    row = report["cases"][0]["transitions"][0]
    check = row["checks"]["linear_constraints"]
    assert check["contact_constraints"] > 0
    assert check["numerical_status"] == "fail"
    assert check["status"] == "fail"
    assert row["ideal_corrector_compatibility"]["status"] == "fail"
    assert row["native_transition_consistency"]["status"] == "pass"
    assert report["status"] == "fail"


@pytest.mark.parametrize("native_status", ("fail", "unknown"))
def test_native_solver_status_is_independent_of_corrector_compatibility(monkeypatch, native_status):
    def mechanics(*args):
        checks = mechanical_checks(*args)
        checks["native_elastic_solver"]["status"] = native_status
        return checks

    report = inspect(monkeypatch, analytical_trajectory(), mechanics=mechanics)
    assert report["constraint_status"] == "pass"
    assert report["ideal_corrector_compatibility"] == {"status": "pass", "required_checks": list(audit.CORRECTOR_CHECKS)}
    assert report["native_transition_consistency"] == {"status": native_status, "required_checks": list(audit.NATIVE_CHECKS)}
    assert report["status"] == native_status
    for row in report["cases"][0]["transitions"]:
        assert row["ideal_corrector_compatibility"]["status"] == "pass"
        assert row["native_transition_consistency"]["status"] == native_status


def test_reconstructed_elastic_failure_is_not_reported_as_native_solver_failure(monkeypatch):
    def mechanics(*args):
        checks = mechanical_checks(*args)
        checks["elastic_dynamics"] = dict(status="fail", law="independently reconstructed float64 residual")
        return checks

    report = inspect(monkeypatch, analytical_trajectory(), mechanics=mechanics)
    assert report["status"] == "fail"
    assert report["ideal_corrector_compatibility"]["status"] == "pass"
    assert report["native_transition_consistency"]["status"] == "pass"
    assert report["totals"]["checks"]["elastic_dynamics"] == {"pass": 0, "fail": 3, "unknown": 0}
    assert report["totals"]["checks"]["native_elastic_solver"] == {"pass": 3, "fail": 0, "unknown": 0}


@pytest.mark.parametrize("equation_status", ("fail", None))
def test_native_explicit_residual_summary_remains_separate_from_recursive_certificate(monkeypatch, equation_status):
    def mechanics(*args):
        checks = mechanical_checks(*args)
        if equation_status is not None:
            checks["native_elastic_solver"]["equation_status"] = equation_status
        return checks

    report = inspect(monkeypatch, analytical_trajectory(), mechanics=mechanics)
    expected = {"pass": 0, "fail": 3 if equation_status == "fail" else 0, "unknown": 3 if equation_status is None else 0}
    assert report["native_transition_consistency"]["status"] == "pass"
    assert report["totals"]["native_explicit_residual"] == expected
    assert report["cases"][0]["native_explicit_residual"] == expected
    assert report["totals"]["checks"]["native_elastic_solver"] == {"pass": 3, "fail": 0, "unknown": 0}


def test_disabled_energy_is_excluded_from_ideal_corrector_contract(monkeypatch):
    report = inspect(monkeypatch, analytical_trajectory(), mechanics=mechanical_checks,
                     config=RolloutConfig(kinematic_rule="backward_euler", energy_balance=False))
    assert "mechanical_energy" not in report["ideal_corrector_compatibility"]["required_checks"]
    assert report["ideal_corrector_compatibility"]["status"] == "pass"


def test_teacher_audit_requires_explicit_normalization_and_integration_rule():
    with pytest.raises(ValueError, match="checkpoint or an explicit positive velocity_scale"):
        audit.inspect_teacher_constraints("unused")
    with pytest.raises(ValueError, match="explicit kinematic_rule"):
        audit.inspect_teacher_constraints("unused", velocity_scale=1.)


def test_legacy_best_checkpoint_is_loaded_without_changing_weights(monkeypatch, tmp_path):
    checkpoint, original, _ = checkpoint_file(tmp_path, "independent", legacy=True)
    before = file_hash(checkpoint)
    monkeypatch.setattr(audit, "load_dynamics_dataset", lambda *args, **options: [analytical_trajectory()])
    report = audit.inspect_teacher_constraints("unused", checkpoint=checkpoint, mechanics=mechanical_checks)
    assert report["status"] == "pass"
    assert report["velocity_scale"] == original.config["velocity_scale"]
    assert report["resolved_kinematic_rule"] == original.config["motion_prior"]
    assert report["checkpoint_sha256"] == before
    assert file_hash(checkpoint) == before
