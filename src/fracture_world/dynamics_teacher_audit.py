from dataclasses import asdict, replace
from pathlib import Path
from types import SimpleNamespace

import numpy as np

from .dynamics_corrector import CorrectorTolerances, _diagnostics
from .dynamics_data import load_dynamics_dataset
from .dynamics_geometry import geometry_diagnostics, phase_swept_geometry_diagnostics
from .dynamics_initial import DynamicsInitial
from .dynamics_mechanics import mechanics_diagnostics
from .dynamics_learning import load_dynamics_model
from .dynamics_readiness import _native_clock
from .dynamics_runtime import RolloutConfig, _mechanical_energy, _problem, step_contacts, step_energy
from .dynamics_stabilization import solve_stabilization
from .io import file_hash, write_json


CHECKS = ("linear_constraints", "endpoint_geometry", "swept_geometry", "mechanical_energy",
          "teacher_physics", "elastic_dynamics", "fracture_initiation", "kinematics", "native_elastic_solver")
CORRECTOR_CHECKS = ("linear_constraints", "endpoint_geometry", "swept_geometry", "mechanical_energy", "teacher_physics")
NATIVE_CHECKS = ("kinematics", "native_elastic_solver")


def _status(checks):
    values = [check["status"] for check in checks]
    return "fail" if "fail" in values else "unknown" if "unknown" in values or not values else "pass"


def _summary(rows, names):
    return {name: {status: sum(row["checks"][name]["status"] == status for row in rows)
                   for status in ("pass", "fail", "unknown")} for name in names}


def _contract(reports, required):
    return dict(status=_status(reports), required_checks=list(required))


def _native_explicit_residual_summary(rows):
    values = [row["checks"].get("native_elastic_solver", {}).get("equation_status", "unknown") for row in rows]
    return {status: values.count(status) for status in ("pass", "fail", "unknown")}


def _physics(settings):
    names = ("rayleigh_mass_damping", "rayleigh_stiffness_damping", "friction_coefficient")
    values = {name: settings[name] for name in names if name in settings}
    incompatible = {name: value for name, value in values.items() if value != 0}
    return dict(status="fail" if incompatible else "pass" if len(values) == len(names) else "unknown",
                recorded=values, missing=[name for name in names if name not in values],
                incompatible=incompatible, required_model="undamped frictionless elastic fracture")


def _unknown_checks(reason, physics):
    return {name: dict(physics) if name == "teacher_physics" else dict(status="unknown", reason=reason)
            for name in CHECKS}


def required_checks(config):
    return [name for name in CHECKS if (name != "mechanical_energy" or config.energy_balance)
            and (name != "elastic_dynamics" or config.mechanics_mode != "proximal")]


def _transition_checks(item, frame, dt, model, config, mechanics, physics):
    transition = replace(item.transition(frame), dt=dt)
    metadata = {**item.metadata, **item.metadata["conditioning_context"], "teacher_physics": physics["recorded"]}
    if getattr(transition, "elastic_state", None) is not None:
        from .dynamics_data import elastic_state_signature

        metadata["elastic_state"] = {**transition.elastic_state,
                                     "positions_hash": elastic_state_signature(transition.positions, transition.broken, transition.fixed)}
    initial = DynamicsInitial(item.topology, transition.positions, transition.velocities, transition.broken,
                              transition.fixed, transition.time, dt, item.gravity, item.material,
                              item.case_id, item.condition_digest, metadata)
    integration_broken = transition.broken if config.mechanics_mode != "disabled" else transition.target_broken
    contacts = step_contacts(initial, transition.positions, transition.target_positions, integration_broken, config)
    floor = initial.metadata.get("floor_height_m")
    problem = _problem(initial, transition.positions, transition.velocities, transition.target_positions,
                       transition.target_velocities, transition.target_broken, transition.target_fixed, contacts,
                       model, config, previous_broken=transition.broken, previous_fixed=transition.fixed)
    stabilization = None
    stabilization_report = dict(status="pass", law="disabled legacy integration")
    if problem.elastic_balance is not None:
        stabilization, stabilization_report = solve_stabilization(problem.elastic_balance, contacts, transition.target_velocities,
                                                                  config.geometry_tolerance, problem.position_scale,
                                                                  CorrectorTolerances(config.absolute_tolerance, config.relative_tolerance))
        if stabilization is not None:
            problem = _problem(initial, transition.positions, transition.velocities, transition.target_positions,
                               transition.target_velocities, transition.target_broken, transition.target_fixed, contacts,
                               model, config, previous_broken=transition.broken, previous_fixed=transition.fixed,
                               stabilization_offset=stabilization)
    state = problem.encode_state(transition.target_positions.reshape(problem.positions.shape),
                                 transition.target_velocities.reshape(problem.velocities.shape))
    linear = _diagnostics(problem, state, CorrectorTolerances(config.absolute_tolerance, config.relative_tolerance))
    raw_problem = replace(problem, position_roundoff=None, stabilization_offset=None, native_contact_phase=False,
                          contact_tolerance=0., mechanics_mode="disabled", elastic_balance=None)
    raw_linear = _diagnostics(raw_problem, raw_problem.encode_state(transition.target_positions.reshape(problem.positions.shape),
                                                                  transition.target_velocities.reshape(problem.velocities.shape)),
                               CorrectorTolerances(config.absolute_tolerance, config.relative_tolerance))
    geometry = geometry_diagnostics(item.topology.mesh, transition.target_positions, transition.target_broken,
                                    config.geometry_tolerance, floor)
    swept = phase_swept_geometry_diagnostics(item.topology.mesh, transition.positions, transition.target_positions,
                                             transition.target_broken, config.geometry_tolerance, floor, stabilization)
    energy_model, budget = step_energy(initial, transition.positions, transition.velocities, transition.broken, transition.fixed,
                                       config, balance=problem.elastic_balance, stabilization=stabilization)
    energy = energy_model.balance(transition.positions, transition.velocities,
                                                transition.target_positions, transition.target_velocities,
                                                model.config["velocity_scale"], budget=budget, absolute_tolerance=config.absolute_tolerance,
                                                relative_tolerance=config.relative_tolerance)
    swept_failure = bool(swept["degenerate_or_inverted_swept_tetrahedra"] or
                         swept["maximum_swept_floor_penetration_metres"] > config.geometry_tolerance)
    checks = dict(
        linear_constraints=dict(status="fail" if not linear["constraints_satisfied"] else "unknown"
                                if stabilization_report["status"] != "pass" or (contacts and config.mechanics_mode == "disabled") else "pass",
                                numerical_status="pass" if linear["constraints_satisfied"] else "fail",
                                maximum_normalized_violation=linear["constraint_violation"],
                                normalized_tolerance=linear["constraint_tolerance"],
                                violating_factors={name: value for name, value in linear["factor_violations"].items()
                                                   if value > linear["constraint_tolerance"]},
                                checked_factors=len(problem.factors), contact_constraints=len(contacts),
                                precision_scope="source-aligned phase constraints with IEEE-derived position publication envelope and configured geometry tolerance",
                                raw_analytical_status="pass" if raw_linear["constraints_satisfied"] else "fail",
                                raw_analytical_factor_violations=raw_linear["factor_violations"],
                                raw_physical_momentum=raw_linear["momentum_islands"],
                                stabilization=stabilization_report,
                                arithmetic_envelope_metres=float(np.max(problem.position_roundoff)) if problem.position_roundoff is not None else 0.,
                                contact_stabilization="recorded deltas; direct ideal-state constraint compatibility remains separate"
                                if transition.observations is not None else "unrecorded; contact kinematics cannot be certified" if contacts else "not_inferred_at_recorded_endpoints"),
        endpoint_geometry=dict(status="pass" if geometry["valid"] else "fail", diagnostics=geometry),
        swept_geometry=dict(status="pass" if swept["valid"] else "fail" if swept_failure else "unknown", diagnostics=swept),
        mechanical_energy=dict(status="unknown" if stabilization_report["status"] != "pass" else "pass" if energy["energy_satisfied"] else "fail",
                               normalized_residual=energy["normalized_residual"],
                               normalized_tolerance=energy["normalized_tolerance"],
                               energy_change_j=energy["energy_change"], enforced=config.energy_balance,
                               law=energy.get("law", "nonlinear corotational total energy"),
                               certificate_scope=energy.get("certificate_scope", "legacy nonlinear energy admissibility"),
                               nonlinear_energy=energy_model.stabilization_diagnostics(transition.target_positions, transition.target_velocities)
                               if hasattr(energy_model, "stabilization_diagnostics") else energy_model.nonlinear_diagnostics(transition.target_positions, transition.target_velocities)
                               if hasattr(energy_model, "nonlinear_diagnostics") else None,
                               native_stabilization_work_status="unknown" if contacts else "not inferred at recorded endpoints"),
        teacher_physics=dict(physics),
        elastic_dynamics=dict(status="unknown", reason="mechanics residual evaluator not supplied"),
        fracture_initiation=dict(status="unknown", reason="fracture initiation evaluator not supplied"),
        kinematics=dict(status="unknown", reason="native kinematic residual evaluator not supplied"),
        native_elastic_solver=dict(status="unknown", reason="native elastic solver residual telemetry not supplied"),
    )
    if mechanics is not None:
        updates = mechanics(initial, transition, contacts, config)
        if not isinstance(updates, dict) or set(updates) - {"elastic_dynamics", "fracture_initiation", *NATIVE_CHECKS}:
            raise ValueError("mechanics evaluator must return named elastic, fracture, kinematic or native solver checks")
        for name, report in updates.items():
            if not isinstance(report, dict) or report.get("status") not in {"pass", "fail", "unknown"}:
                raise ValueError("mechanics check must report pass, fail or unknown")
            checks[name] = report
    return checks


def inspect_teacher_constraints(data, checkpoint=None, config=None, device="cpu", output=None, progress=None,
                                *, velocity_scale=None, mechanics=None):
    config = config or RolloutConfig()
    if not isinstance(config, RolloutConfig):
        raise TypeError("teacher audit config must be a RolloutConfig")
    if checkpoint is None:
        if (isinstance(velocity_scale, bool) or not isinstance(velocity_scale, (int, float)) or
                not np.isfinite(velocity_scale) or velocity_scale <= 0):
            raise ValueError("teacher audit requires a checkpoint or an explicit positive velocity_scale")
        if config.kinematic_rule == "auto":
            raise ValueError("teacher audit without a checkpoint requires an explicit kinematic_rule")
        model = SimpleNamespace(config=dict(velocity_scale=float(velocity_scale), motion_prior=config.kinematic_rule))
        checkpoint_hash = None
    else:
        if velocity_scale is not None:
            raise ValueError("checkpoint normalization and explicit velocity_scale cannot both be selected")
        model, _, _, _ = load_dynamics_model(checkpoint, device)
        checkpoint_hash = file_hash(checkpoint)
    if mechanics is None:
        mechanics = lambda initial, transition, contacts, settings: mechanics_diagnostics(
            initial, transition, contacts, settings, velocity_scale=model.config["velocity_scale"])
    if output is not None and Path(output).exists():
        raise ValueError("teacher audit output already exists")
    items = load_dynamics_dataset(data, stream=True, progress=progress)
    required = required_checks(config)
    constraints = [name for name in required if name not in {"elastic_dynamics", "fracture_initiation", "native_elastic_solver"}]
    ideal_required = [name for name in CORRECTOR_CHECKS if name in required]
    cases, all_rows = [], []
    for index, item in enumerate(items, start=1):
        clock = _native_clock(item)
        context = item.metadata.get("conditioning_context", {})
        physics = _physics(item.metadata.get("native_settings", {}))
        rows = []
        if progress is not None:
            progress(dict(event="teacher_constraint_audit", phase="case_started", case=item.case_id,
                          case_index=index, cases=len(items)))
        for frame in range(len(item)):
            consecutive = clock is not None and clock[frame + 1] - clock[frame] == 1
            if not consecutive:
                reason = "unknown native clock" if clock is None else "recorded interval skips native steps"
                checks = _unknown_checks(reason, physics)
            else:
                dt = float(np.float32(context["dt"]))
                try:
                    checks = _transition_checks(item, frame, dt, model, config, mechanics, physics)
                except (ValueError, np.linalg.LinAlgError, FloatingPointError) as error:
                    checks = _unknown_checks(f"{type(error).__name__}: {error}", physics)
            row = dict(frame=frame, initial_native_step=int(clock[frame]) if clock is not None else None,
                       final_native_step=int(clock[frame + 1]) if clock is not None else None,
                       recorded_interval_dt_s=float(item.times[frame + 1] - item.times[frame]),
                       consecutive_native_transition=bool(consecutive), checks=checks)
            row["constraint_status"] = _status(checks[name] for name in constraints)
            row["status"] = _status(checks[name] for name in required)
            row["ideal_corrector_compatibility"] = _contract((checks[name] for name in ideal_required), ideal_required)
            row["native_transition_consistency"] = _contract((checks[name] for name in NATIVE_CHECKS), NATIVE_CHECKS)
            rows.append(row)
            if progress is not None and (frame == 0 or (frame + 1) % 25 == 0 or frame + 1 == len(item)):
                progress(dict(event="teacher_constraint_audit", phase="transition", case=item.case_id,
                              frame=frame, transitions=len(item), status=row["status"], constraint_status=row["constraint_status"]))
        cases.append(dict(case_id=item.case_id, condition_digest=item.condition_digest, shape_family=item.shape_family,
                          mesh_fingerprint=item.topology.fingerprint, source_hashes=item.metadata.get("source_hashes", {}),
                          status=_status(dict(status=row["status"]) for row in rows),
                          constraint_status=_status(dict(status=row["constraint_status"]) for row in rows),
                          ideal_corrector_compatibility=_contract((row["ideal_corrector_compatibility"] for row in rows), ideal_required),
                          native_transition_consistency=_contract((row["native_transition_consistency"] for row in rows), NATIVE_CHECKS),
                          native_explicit_residual=_native_explicit_residual_summary(rows),
                          recorded_transitions=len(rows), checks=_summary(rows, CHECKS), transitions=rows))
        all_rows.extend(rows)
    report = dict(schema="fracture-world.teacher-constraint-audit.v1", status=_status(dict(status=row["status"]) for row in all_rows),
                  constraint_status=_status(dict(status=row["constraint_status"]) for row in all_rows),
                  ideal_corrector_compatibility=_contract((row["ideal_corrector_compatibility"] for row in all_rows), ideal_required),
                  native_transition_consistency=_contract((row["native_transition_consistency"] for row in all_rows), NATIVE_CHECKS),
                  cases=cases, totals=dict(cases=len(cases), recorded_transitions=len(all_rows), checks=_summary(all_rows, CHECKS),
                                          native_explicit_residual=_native_explicit_residual_summary(all_rows)),
                  configuration=asdict(config), velocity_scale=float(model.config["velocity_scale"]),
                  resolved_kinematic_rule=model.config.get("motion_prior", "backward_euler") if config.kinematic_rule == "auto" else config.kinematic_rule,
                  checkpoint_sha256=checkpoint_hash, required_checks=required,
                  provenance=dict(source="validated recorded FEMFX numerical trajectories", physical_ground_truth=False,
                                  inference_inputs=False, teacher_future_used_only_for_offline_audit=True,
                                  native_engine_calls=0, state_projection_used=False,
                                  stabilization_target_solve_used=config.mechanics_mode != "disabled", teacher_state_modified=False,
                                  contact_constraint_source="recorded current and next teacher geometry"),
                  sparse_interval_policy="unknown; no interpolation or invented native states",
                  interpretation="overall pass requires every check to be available and pass; unknown is never a pass. "
                                 "Corrector compatibility tests the configured phase, IEEE publication envelope and geometry constraints; raw analytical violations remain recorded. "
                                 "constraint_status retains the legacy constraint aggregation, including the separate kinematic check. "
                                 "Native transition consistency certifies only the observed position update and recorded recursive solver stopping rule; "
                                 "native_elastic_solver.equation_status separately checks the explicitly recomputed native residual, "
                                 "not the complete contact law or topology evolution. Reconstructed elastic residuals are a separate model comparison, "
                                 "not a native solver convergence test. In proximal mode elastic and aggregate discrete balance are soft residual objectives; "
                                 "only exact reference mode enforces zero elastic residual as a hard constraint.")
    if output is not None:
        write_json(output, report)
    return report
