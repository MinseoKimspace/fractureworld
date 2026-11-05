from dataclasses import asdict, dataclass, replace
import hashlib
import json
from pathlib import Path
from time import perf_counter

import numpy as np
import torch

from .dynamics_corrector import CorrectorTolerances, DynamicsProblem, _diagnostics, solve_central, solve_consensus
from .dynamics_arithmetic import position_publication_envelope
from .dynamics_energy import CorotationalEnergy, EnergyProjection, native_corotational_energy
from .dynamics_energy_solvers import solve_energy
from .dynamics_geometry import boundary_triangles, corner_groups, floor_contacts, geometry_diagnostics, linearized_contacts, phase_swept_geometry_diagnostics, render_geometry, swept_geometry_diagnostics
from .dynamics_frozen_energy import FrozenQuadraticEnergy, TransportEnergy
from .dynamics_stabilization import solve_stabilization
from .dynamics_model import constant_acceleration_prediction, support_release_mask
from .dynamics_mechanics import FrozenElasticBalance, elastic_kernel_arguments, fracture_initiation_diagnostics
from .dynamics_initial import load_initial
from .dynamics_learning import load_dynamics_model, state_batch
from .io import file_hash, new_directory, write_json
from .partition import component_labels


@dataclass(frozen=True)
class RolloutConfig:
    fracture_samples: int = 2
    fracture_policy: str = "fm_search"
    flow_steps: int = 8
    fracture_prior_weight: float = 1.0
    rho: float = 0.01
    absolute_tolerance: float = 1e-8
    relative_tolerance: float = 1e-7
    geometry_tolerance: float = 1e-6
    max_iterations: int | None = None
    max_contact_iterations: int | None = None
    kinematic_rule: str = "auto"
    energy_balance: bool = True
    seed: int = 17
    mechanics_mode: str = "disabled"
    mechanics_weight: float = 1.0
    fracture_initiation: bool = False

    def __post_init__(self):
        for name, minimum in (("fracture_samples", 0), ("flow_steps", 1)):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
                raise ValueError(f"{name} must be an integer >= {minimum}")
        if self.fracture_policy not in {"marginal_search", "shared_sample", "fm_search"}:
            raise ValueError("fracture_policy must select fm_search, marginal_search or shared_sample")
        if self.fracture_policy == "shared_sample" and self.fracture_samples != 1:
            raise ValueError("shared_sample requires exactly one fracture sample")
        if self.fracture_policy == "fm_search" and self.fracture_samples < 1:
            raise ValueError("fm_search requires at least one fracture sample")
        values = [self.fracture_prior_weight, self.rho, self.absolute_tolerance, self.geometry_tolerance]
        if not np.isfinite([*values, self.relative_tolerance]).all() or min(values) <= 0 or self.relative_tolerance < 0:
            raise ValueError("rollout weights and tolerances must be finite and positive")
        for value in (self.max_iterations, self.max_contact_iterations):
            if value is not None and (isinstance(value, bool) or not isinstance(value, int) or value < 1):
                raise ValueError("explicit iteration budgets must be positive or null")
        if self.kinematic_rule not in ("auto", "backward_euler", "trapezoidal", "constant_acceleration"):
            raise ValueError("unsupported kinematic rule")
        if not isinstance(self.energy_balance, bool):
            raise ValueError("energy_balance must be boolean")
        if self.mechanics_mode not in {"disabled", "proximal", "exact"}:
            raise ValueError("mechanics_mode must select disabled, proximal or exact")
        if self.mechanics_mode != "disabled" and self.kinematic_rule not in {"auto", "backward_euler"}:
            raise ValueError("FEMFX mechanics requires backward_euler kinematics")
        if not np.isfinite(self.mechanics_weight) or self.mechanics_weight <= 0:
            raise ValueError("mechanics_weight must be finite and positive")
        if not isinstance(self.fracture_initiation, bool):
            raise ValueError("fracture_initiation must be boolean")


def load_rollout_config(path=None):
    return RolloutConfig(**json.loads(Path(path).read_text(encoding="utf-8"))) if path else RolloutConfig()


def _array(value):
    return value.detach().cpu().numpy().astype(float, copy=True)


def _proposal_arrays(proposal, positions, velocities):
    next_positions = positions + _array(proposal["position_increment"]) if "position_increment" in proposal else _array(proposal["positions"])
    next_velocities = velocities + _array(proposal["velocity_increment"]) if "velocity_increment" in proposal else _array(proposal["velocities"])
    return next_positions, next_velocities


def _candidate_proposal(proposal, positions, velocities, fixed):
    template = proposal["positions"]
    current_positions = torch.as_tensor(positions, dtype=template.dtype, device=template.device)
    current_velocities = torch.as_tensor(velocities, dtype=template.dtype, device=template.device)
    position_increment = proposal.get("unconstrained_position_increment", proposal.get("position_increment", template - current_positions))
    velocity_increment = proposal.get("unconstrained_velocity_increment", proposal.get("velocity_increment", proposal["velocities"] - current_velocities))
    held = torch.as_tensor(fixed, dtype=torch.bool, device=template.device)[:, None]
    position_increment = torch.where(held, torch.zeros_like(position_increment), position_increment)
    velocity_increment = torch.where(held, -current_velocities, velocity_increment)
    return {**proposal, "positions": current_positions + position_increment, "velocities": current_velocities + velocity_increment,
            "position_increment": position_increment, "velocity_increment": velocity_increment}


def _validate_proposal(proposal, positions, velocities):
    fields = ("positions", "velocities", "position_increment", "velocity_increment", "fracture_probability",
              "support_release_probability", "fracture_logits", "support_release_logits",
              "unconstrained_position_increment", "unconstrained_velocity_increment")
    invalid = [key for key in fields if key in proposal and not torch.isfinite(proposal[key]).all()]
    if invalid:
        raise FloatingPointError(f"nonfinite prediction fields: {', '.join(invalid)}")
    next_positions, next_velocities = _proposal_arrays(proposal, positions, velocities)
    if not np.isfinite(next_positions).all() or not np.isfinite(next_velocities).all():
        raise FloatingPointError("nonfinite predicted material-corner state")


def velocity_publication_domain(bound):
    bound = np.asarray(bound, dtype=float)
    if not np.isfinite(bound).all() or (bound < 0).any() or (bound > float(np.finfo(np.float32).max)).any():
        raise ValueError("velocity publication domain requires finite nonnegative magnitudes")
    mantissa, exponent = np.frexp(bound)
    exponent = np.where(mantissa == .5, exponent - 1, exponent)
    with np.errstate(over="ignore"):
        upper = np.ldexp(np.ones_like(bound), exponent)
    return np.where(bound == 0, 0., np.minimum(upper, float(np.finfo(np.float32).max)))


def fracture_candidates(model, batch, proposal, config, step, *, context=None):
    old = batch["broken"].detach().cpu().numpy()
    candidates = [] if config.fracture_policy in {"shared_sample", "fm_search"} else [old | (_array(proposal["fracture_probability"]) >= 0.5), old.copy()]
    for index in range(config.fracture_samples):
        generator = torch.Generator(device=batch["positions"].device).manual_seed(config.seed + step * 1000003 + index)
        noise = torch.randn(len(old), generator=generator, device=batch["positions"].device)
        field = _array(model.sample_fracture(batch, config.flow_steps, noise, context=context))
        if not np.isfinite(field).all():
            raise FloatingPointError("nonfinite sampled fracture field")
        candidates.append(old | (field >= 0))
    unique = {}
    for mask in candidates:
        unique.setdefault(np.packbits(mask).tobytes(), mask)
    return list(unique.values())


def _fracture_cost(mask, probability, old):
    active = ~old
    if not active.any():
        return 0.0
    probability = np.clip(probability[active], np.finfo(float).eps, 1 - np.finfo(float).eps)
    return float(-np.mean(np.where(mask[active], np.log(probability), np.log1p(-probability))))


def _problem(initial, positions, velocities, proposed_positions, proposed_velocities, broken, fixed, contacts, model, config, *, previous_broken=None, previous_fixed=None,
             stabilization_offset=None, velocity_envelope=None):
    mesh = initial.topology.mesh
    full_broken = np.zeros(len(mesh.faces.vertices), dtype=bool)
    full_broken[mesh.internal_faces] = broken
    previous_full_broken = None
    if previous_broken is not None:
        previous_broken = np.asarray(previous_broken)
        if previous_broken.shape != (len(mesh.internal_faces),) or previous_broken.dtype != np.dtype(bool):
            raise ValueError("previous_broken must be a Boolean vector over internal material faces")
        previous_full_broken = np.zeros(len(mesh.faces.vertices), dtype=bool)
        previous_full_broken[mesh.internal_faces] = previous_broken
    shape = (len(mesh.tets), 4, 3)
    if previous_fixed is not None:
        previous_fixed = np.asarray(previous_fixed)
        if previous_fixed.shape != (4 * len(mesh.tets),) or previous_fixed.dtype != np.dtype(bool):
            raise ValueError("previous_fixed must be a Boolean vector over material corners")
        previous_fixed = previous_fixed.reshape(shape[:2])
    balance = None
    rule = model.config.get("motion_prior", "backward_euler") if config.kinematic_rule == "auto" else config.kinematic_rule
    if config.mechanics_mode != "disabled":
        integration_broken = initial.broken if previous_broken is None else previous_broken
        integration_fixed = initial.fixed if previous_fixed is None else previous_fixed.ravel()
        full_broken[mesh.internal_faces] = integration_broken
        fixed = integration_fixed
        rule = "backward_euler"
        balance = FrozenElasticBalance(mesh, initial.material, positions, velocities, initial.dt, initial.gravity,
                                        integration_broken, integration_fixed, tuple(contacts), model.config["velocity_scale"],
                                        config.absolute_tolerance, config.relative_tolerance, coefficient_model="native_float32",
                                        current_elastic_state=current_elastic_state(initial, positions, integration_broken, integration_fixed))
    masses = initial.topology.masses if balance is None else balance.corner_masses
    gravity = initial.gravity if balance is None else balance.gravity
    roundoff = None
    if balance is not None:
        stabilization_offset = np.zeros_like(positions) if stabilization_offset is None else np.asarray(stabilization_offset).reshape(positions.shape)
        velocity_envelope = velocity_publication_domain(np.maximum(np.maximum(np.abs(velocities), np.abs(proposed_velocities)), model.config["velocity_scale"])) if velocity_envelope is None else velocity_envelope
        roundoff = position_publication_envelope(positions, initial.dt, velocity_envelope, stabilization_offset)
    return DynamicsProblem(mesh, positions.reshape(shape), velocities.reshape(shape), proposed_positions.reshape(shape),
                           proposed_velocities.reshape(shape), initial.dt, initial.material["rest_density"],
                           full_broken, fixed.reshape(-1, 4),
                           (initial.dt * masses[:, None] * gravity).reshape(shape), contacts,
                           initial.dt * model.config["velocity_scale"], model.config["velocity_scale"],
                           kinematic_rule=rule, previous_broken_faces=previous_full_broken, previous_fixed=previous_fixed,
                           elastic_balance=balance, mechanics_mode=config.mechanics_mode, mechanics_weight=config.mechanics_weight,
                           stabilization_offset=None if stabilization_offset is None else stabilization_offset.reshape(shape),
                           position_roundoff=None if roundoff is None else roundoff.reshape(shape),
                           kinematic_velocity_envelope=None if balance is None else np.asarray(velocity_envelope).reshape(shape),
                           contact_tolerance=config.geometry_tolerance, native_contact_phase=balance is not None)


def state_mechanics_diagnostics(initial, positions, velocities, next_positions, next_velocities, previous_broken, broken,
                                previous_fixed, fixed, model, config, *, contacts=None, balance=None):
    mesh = initial.topology.mesh
    report = {}
    try:
        if config.mechanics_mode != "disabled":
            if contacts is None:
                contacts = step_contacts(initial, positions, next_positions, previous_broken, config)
            balance = balance or FrozenElasticBalance(mesh, initial.material, positions, velocities, initial.dt,
                                                      initial.gravity, previous_broken, previous_fixed, tuple(contacts),
                                                      model.config["velocity_scale"], config.absolute_tolerance, config.relative_tolerance,
                                                      coefficient_model="native_float32",
                                                      current_elastic_state=current_elastic_state(initial, positions, previous_broken, previous_fixed))
            report["elastic_dynamics"] = balance.diagnostics(next_positions, next_velocities)
            settings = initial.metadata.get("teacher_physics", {})
            recorded = all(name in settings and settings[name] == 0 for name in
                           ("rayleigh_mass_damping", "rayleigh_stiffness_damping", "friction_coefficient"))
            report["elastic_dynamics"]["teacher_settings_verified"] = recorded
            if not recorded and report["elastic_dynamics"]["status"] == "pass":
                report["elastic_dynamics"]["status"] = "unknown"
                report["elastic_dynamics"]["unknown_reasons"].append("teacher damping/friction not fully recorded as zero")
        if config.fracture_initiation:
            report["fracture_initiation"] = fracture_initiation_diagnostics(mesh, initial.material, next_positions,
                                                                           previous_broken, broken, config.absolute_tolerance,
                                                                           config.relative_tolerance,
                                                                           coefficient_model="native_float32" if config.mechanics_mode != "disabled" else "analytical")
    except (ValueError, np.linalg.LinAlgError, FloatingPointError) as error:
        for active, name in ((config.mechanics_mode != "disabled", "elastic_dynamics"),
                             (config.fracture_initiation, "fracture_initiation")):
            if active and name not in report:
                report[name] = dict(status="unknown", reason=f"{type(error).__name__}: {error}")
    return report


def step_contacts(initial, positions, proposed, broken, config):
    mesh = initial.topology.mesh
    contacts = linearized_contacts(mesh, positions, proposed, broken, config.geometry_tolerance)
    if "floor_height_m" not in initial.metadata:
        return contacts
    options = {}
    if config.mechanics_mode != "disabled":
        eligible = np.zeros(len(positions), dtype=bool)
        eligible[boundary_triangles(mesh, broken).ravel()] = True
        settings = initial.metadata.get("native_settings", {})
        options = dict(eligible_corners=eligible, activation_distance=settings.get("dist_contact_threshold"),
                       bias=settings.get("dist_contact_bias", 0.))
    return contacts + floor_contacts(positions, proposed, initial.metadata["floor_height_m"], config.geometry_tolerance, **options)


def current_elastic_state(initial, positions, broken, fixed):
    state = initial.metadata.get("elastic_state")
    if state is None:
        return None
    from .dynamics_data import elastic_state_signature

    return state if state.get("positions_hash") == elastic_state_signature(positions, broken, fixed) else None


def advance_elastic_state(initial, positions, velocities, previous_broken, broken, previous_fixed, fixed, start_state=None):
    from .dynamics_data import elastic_state_signature
    from .dynamics_femfx_math import native_frozen_operator

    mesh = initial.topology.mesh
    previous_arguments = elastic_kernel_arguments(mesh, previous_broken, start_state)
    previous_arguments.pop("current_rotations", None)
    previous_arguments.pop("current_centers_of_mass", None)
    before_cut = native_frozen_operator(mesh, initial.material, positions, velocities, initial.dt,
                                        initial.gravity, previous_broken, previous_fixed, **previous_arguments)
    unchanged = np.array_equal(previous_broken, broken)
    after_cut = native_frozen_operator(mesh, initial.material, positions, velocities, initial.dt,
                                       initial.gravity, broken, fixed, current_rotations=before_cut.rotations,
                                       **(previous_arguments if unchanged else {}))
    pieces = component_labels(len(mesh.tets), mesh.dual_edges, broken)
    node_pieces = pieces[after_cut.representatives // 4]
    original_vertices = mesh.tets.ravel()[after_cut.representatives]
    node_order = np.lexsort((original_vertices, node_pieces))
    node_ids = np.empty(len(node_order), dtype=np.int64)
    node_ids[node_order] = np.arange(len(node_order))
    local = np.empty(len(mesh.tets), dtype=np.int64)
    for piece in np.unique(pieces):
        cells = np.flatnonzero(pieces == piece)
        local[cells] = np.arange(len(cells))
    state = dict(schema="fracture-world.femfx-elastic-state.v1",
                 positions_hash=elastic_state_signature(positions, broken, fixed),
                 rotations=before_cut.rotations.tolist(), centers_of_mass=after_cut.tet_centers_of_mass.tolist(),
                 nodal_masses=after_cut.node_masses[after_cut.groups].reshape(-1, 4).tolist(),
                 corner_vertex_ids=node_ids[after_cut.groups].reshape(-1, 4).tolist(),
                 tetrahedron_local_ids=local.tolist(), piece_ids=pieces.tolist())
    if unchanged and start_state is not None:
        for name in ("corner_vertex_ids", "tetrahedron_local_ids", "piece_ids"):
            state[name] = np.asarray(start_state[name]).tolist()
    return replace(initial, metadata={**initial.metadata, "elastic_state": state,
                                      "native_elastic_state_schema": state["schema"],
                                      "elastic_state_origin": "causal current-state constitutive math",
                                      "native_topology_repacking_verified": False})


def step_energy(initial, positions, velocities, broken, fixed, config, *, balance=None, stabilization=None, nonlinear=None):
    nonlinear = nonlinear or _mechanical_energy(initial, config.mechanics_mode != "disabled")
    if config.mechanics_mode == "disabled":
        return nonlinear, nonlinear.value(positions, velocities)
    balance = balance or FrozenElasticBalance(initial.topology.mesh, initial.material, positions, velocities, initial.dt,
                                              initial.gravity, broken, fixed, coefficient_model="native_float32",
                                              current_elastic_state=current_elastic_state(initial, positions, broken, fixed))
    frozen = FrozenQuadraticEnergy(balance, nonlinear)
    energy = frozen if stabilization is None else TransportEnergy(frozen, stabilization)
    return energy, frozen.value(positions, velocities)


def step_energy_diagnostics(initial, positions, velocities, next_positions, next_velocities, broken, fixed, model, config, *, nonlinear=None):
    nonlinear = nonlinear or _mechanical_energy(initial, config.mechanics_mode != "disabled")
    stabilization = None
    phase = dict(status="pass", law="legacy direct endpoint energy")
    balance = None
    if config.mechanics_mode != "disabled":
        contacts = step_contacts(initial, positions, next_positions, broken, config)
        balance = FrozenElasticBalance(initial.topology.mesh, initial.material, positions, velocities, initial.dt,
                                       initial.gravity, broken, fixed, contacts, coefficient_model="native_float32",
                                       current_elastic_state=current_elastic_state(initial, positions, broken, fixed))
        stabilization, phase = solve_stabilization(balance, contacts, next_velocities, config.geometry_tolerance,
                                                   initial.dt * model.config["velocity_scale"],
                                                   CorrectorTolerances(config.absolute_tolerance, config.relative_tolerance))
    energy, budget = step_energy(initial, positions, velocities, broken, fixed, config, balance=balance,
                                 stabilization=stabilization, nonlinear=nonlinear)
    report = energy.balance(positions, velocities, next_positions, next_velocities, model.config["velocity_scale"],
                            budget=budget, absolute_tolerance=config.absolute_tolerance, relative_tolerance=config.relative_tolerance)
    report.update(status="unknown" if phase["status"] != "pass" else "pass" if report["energy_satisfied"] else "fail",
                  stabilization_phase=phase, evaluation_only=True, evaluator_engine_calls=0)
    if phase["status"] != "pass":
        report["energy_satisfied"] = False
    if hasattr(energy, "stabilization_diagnostics"):
        report["stabilization"] = energy.stabilization_diagnostics(next_positions, next_velocities)
    if config.mechanics_mode != "disabled":
        report["independent_nonlinear_balance"] = nonlinear.balance(
            positions, velocities, next_positions, next_velocities, model.config["velocity_scale"],
            absolute_tolerance=config.absolute_tolerance, relative_tolerance=config.relative_tolerance)
        report["independent_nonlinear_balance"]["enforced_as_native_law"] = False
    return report


def _mechanical_energy(initial, native_precision=False):
    material = initial.material
    factory = native_corotational_energy if native_precision else CorotationalEnergy
    return factory(initial.topology.mesh, material["rest_density"], material["youngs_modulus"],
                   material["poissons_ratio"], initial.gravity)


def _correct(initial, positions, velocities, proposal, broken, fixed, model, config, method, cancel=None, *, energy_model=None, progress=None, previous_broken=None, previous_fixed=None):
    mesh = initial.topology.mesh
    proposed_positions, proposed_velocities = _proposal_arrays(proposal, positions, velocities)
    rule = model.config.get("motion_prior", "backward_euler") if config.kinematic_rule == "auto" else config.kinematic_rule
    integration_broken = broken
    integration_fixed = fixed
    fracture_previous = initial.broken if previous_broken is None else previous_broken
    if config.mechanics_mode != "disabled":
        rule = "backward_euler"
        integration_broken = initial.broken if previous_broken is None else previous_broken
        integration_fixed = initial.fixed if previous_fixed is None else previous_fixed
    trial_positions = positions + initial.dt * velocities + (1.0 if rule == "backward_euler" else 0.5) * initial.dt**2 * initial.gravity
    groups = corner_groups(mesh, integration_broken)
    held = np.isin(groups, groups[integration_fixed])
    trial_positions[held] = positions[held]
    tolerance = CorrectorTolerances(config.absolute_tolerance, config.relative_tolerance)
    energy_model = energy_model or _mechanical_energy(initial, config.mechanics_mode != "disabled")
    stabilization_offset = None
    velocity_envelope = velocity_publication_domain(np.maximum(np.maximum(np.abs(velocities), np.abs(proposed_velocities)), model.config["velocity_scale"]))
    histories, previous_trials = [], []
    outer = 0
    while config.max_contact_iterations is None or outer < config.max_contact_iterations:
        if cancel is not None and cancel():
            return None, dict(status="interrupted", outer_iterations=outer, solves=histories)
        try:
            contacts = step_contacts(initial, positions, trial_positions, integration_broken, config)
        except ValueError as error:
            return None, dict(status="invalid_contact_geometry", reason=str(error), outer_iterations=outer)
        problem = _problem(initial, positions, velocities, proposed_positions, proposed_velocities, broken, fixed,
                           tuple(contacts), model, config, previous_broken=previous_broken, previous_fixed=previous_fixed,
                           stabilization_offset=stabilization_offset, velocity_envelope=velocity_envelope)
        if config.mechanics_mode != "disabled" and stabilization_offset is None:
            stabilization_offset, stabilization = solve_stabilization(problem.elastic_balance, contacts, proposed_velocities,
                                                                       config.geometry_tolerance, problem.position_scale, tolerance, cancel)
            if stabilization_offset is None:
                return None, dict(status=stabilization["status"], stabilization=stabilization, outer_iterations=outer)
            problem = _problem(initial, positions, velocities, proposed_positions, proposed_velocities, broken, fixed,
                               tuple(contacts), model, config, previous_broken=previous_broken, previous_fixed=previous_fixed,
                               stabilization_offset=stabilization_offset, velocity_envelope=velocity_envelope)
        energy, budget = step_energy(initial, positions, velocities, integration_broken, integration_fixed, config,
                                     balance=problem.elastic_balance, stabilization=stabilization_offset, nonlinear=energy_model)
        if progress:
            progress(dict(event="corrector_started", method=method, contact_outer_iteration=outer + 1,
                          contact_constraints=len(contacts), material_corners=problem.corner_count))
        if config.energy_balance:
            projection = EnergyProjection(problem, energy, budget, config.absolute_tolerance, config.relative_tolerance)
            result = solve_energy(problem, projection, method, tolerance, config.rho, config.max_iterations, cancel, progress=progress)
        else:
            result = solve_central(problem, tolerance, config.max_iterations, cancel=cancel) if method == "central" else solve_consensus(
                problem, tolerances=tolerance, rho=config.rho, max_iterations=config.max_iterations, cancel=cancel, progress=progress)
        outer += 1
        histories.append(dict(outer_iteration=outer, status=result.status, iterations=result.iterations,
                              objective=result.objective, diagnostics=result.diagnostics,
                              history=[asdict(item) for item in result.history]))
        if progress:
            progress(dict(event="corrector_solved", method=method, contact_outer_iteration=outer,
                          status=result.status, iterations=result.iterations, runtime_seconds=result.runtime_seconds))
        geometry = geometry_diagnostics(mesh, result.positions, broken, config.geometry_tolerance,
                                        initial.metadata.get("floor_height_m"))
        mechanics = state_mechanics_diagnostics(initial, positions, velocities, result.positions, result.velocities,
                                                fracture_previous, broken, integration_fixed, fixed, model, config,
                                                contacts=contacts, balance=problem.elastic_balance)
        if problem.elastic_balance is not None:
            baseline = problem.elastic_balance.diagnostics(proposed_positions, proposed_velocities)
            elastic = mechanics["elastic_dynamics"]
            before = baseline["maximum_free_normalized_residual"]
            after = elastic["maximum_free_normalized_residual"]
            elastic.update(mode=config.mechanics_mode, equality_enforced=config.mechanics_mode == "exact",
                           before_correction_maximum_free_normalized_residual=before,
                           maximum_residual_reduction=float(before - after) if before is not None and after is not None else None)
        result.diagnostics["mechanics"] = mechanics
        if not result.converged:
            return None, dict(status=result.status, outer_iterations=outer, solves=histories, geometry=geometry, mechanics=mechanics)
        if config.mechanics_mode != "disabled":
            final_velocity = result.velocities.reshape(-1, 3)
            if np.any(np.abs(final_velocity) > velocity_envelope):
                velocity_envelope = velocity_publication_domain(np.maximum(velocity_envelope, np.abs(final_velocity)))
                continue
            updated_offset, stabilization = solve_stabilization(problem.elastic_balance, contacts, final_velocity,
                                                                 config.geometry_tolerance, problem.position_scale, tolerance, cancel)
            if updated_offset is None:
                return None, dict(status=stabilization["status"], stabilization=stabilization, outer_iterations=outer, solves=histories)
            mismatch = float(np.max(np.abs(updated_offset - stabilization_offset))) / problem.position_scale
            stabilization["frozen_offset_consistency_residual"] = mismatch
            stabilization["previous_offset_change_normalized"] = mismatch
            result.diagnostics["stabilization"] = stabilization
            if mismatch > tolerance.absolute + tolerance.relative:
                verified_problem = _problem(initial, positions, velocities, proposed_positions, proposed_velocities, broken, fixed,
                                             contacts, model, config, previous_broken=previous_broken, previous_fixed=previous_fixed,
                                             stabilization_offset=updated_offset, velocity_envelope=velocity_envelope)
                verified_state = verified_problem.encode_state(result.positions, result.velocities)
                verified = _diagnostics(verified_problem, verified_state, tolerance)
                verified_energy, verified_budget = step_energy(initial, positions, velocities, integration_broken, integration_fixed, config,
                                                                balance=verified_problem.elastic_balance, stabilization=updated_offset, nonlinear=energy_model)
                energy_check = verified_energy.balance(positions, velocities, result.positions, result.velocities,
                                                       model.config["velocity_scale"], budget=verified_budget,
                                                       absolute_tolerance=config.absolute_tolerance, relative_tolerance=config.relative_tolerance)
                stabilization_offset = updated_offset
                if not verified["constraints_satisfied"] or (config.energy_balance and not energy_check["energy_satisfied"]):
                    continue
                problem, energy, budget = verified_problem, verified_energy, verified_budget
                for key in ("constraints_satisfied", "constraint_violation", "constraint_tolerance", "factor_violations"):
                    result.diagnostics[key] = verified[key]
                stabilization["updated_phase_recertified"] = True
                stabilization["frozen_offset_consistency_residual"] = 0.
                stabilization["recertified_constraint_violation"] = verified["constraint_violation"]
            result.diagnostics["energy"] = {**result.diagnostics.get("energy", {}), **energy.balance(positions, velocities, result.positions, result.velocities,
                                                          model.config["velocity_scale"], budget=budget,
                                                          absolute_tolerance=config.absolute_tolerance, relative_tolerance=config.relative_tolerance)}
            result.diagnostics["energy"]["stabilization"] = energy.stabilization_diagnostics(result.positions, result.velocities)
        initiation = mechanics.get("fracture_initiation", {})
        if config.fracture_initiation and initiation.get("status") != "pass":
            return None, dict(status="fracture_initiation_rejected" if initiation.get("status") == "fail" else "fracture_initiation_unknown",
                              outer_iterations=outer, solves=histories, geometry=geometry, mechanics=mechanics)
        if geometry["valid"]:
            swept = phase_swept_geometry_diagnostics(mesh, positions, result.positions, broken, config.geometry_tolerance,
                                                     initial.metadata.get("floor_height_m"), stabilization_offset)
            if not swept["valid"]:
                return None, dict(status="continuous_geometry_unresolved", outer_iterations=outer, solves=histories,
                                  geometry=geometry, swept_geometry=swept, mechanics=mechanics)
            return result, dict(status="converged", outer_iterations=outer, solves=histories, geometry=geometry,
                                swept_geometry=swept, mechanics=mechanics,
                                contact_initialization="physical free motion with prescribed supports")
        if geometry.get("inverted_tetrahedra", 0):
            return None, dict(status="inverted_tetrahedra", outer_iterations=outer, solves=histories, geometry=geometry, mechanics=mechanics)
        if np.allclose(trial_positions, result.positions.reshape(-1, 3), rtol=0, atol=config.geometry_tolerance):
            return None, dict(status="contact_linearization_stagnation", outer_iterations=outer, solves=histories, geometry=geometry, mechanics=mechanics)
        if any(np.allclose(previous, result.positions.reshape(-1, 3), rtol=0, atol=config.geometry_tolerance)
               for previous in previous_trials):
            return None, dict(status="contact_linearization_cycle", outer_iterations=outer, solves=histories, geometry=geometry, mechanics=mechanics)
        previous_trials = (previous_trials + [trial_positions.copy()])[-2:]
        trial_positions = result.positions.reshape(-1, 3)
    return None, dict(status="contact_iteration_budget", outer_iterations=outer, solves=histories)


def release_supports(mesh, broken, fixed, release_probability):
    return support_release_mask(torch.tensor(mesh.corner_face_pairs.copy(), dtype=torch.long),
                                torch.as_tensor(broken, dtype=torch.bool), torch.as_tensor(fixed, dtype=torch.bool),
                                torch.as_tensor(release_probability)).numpy()


def _replay(initial, positions, broken, fixed, times, metrics, metadata):
    mesh = initial.topology.mesh
    frames = []
    for time, state, cuts, supports, values in zip(times, positions, broken, fixed, metrics):
        frames.append(dict(time=float(time), poses=[dict(position=[0, 0, 0], orientation=[0, 0, 0, 1])],
                           geometries=[render_geometry(mesh, state, cuts, supports)], metrics=values))
    first = frames[0]["geometries"][0]
    physics_only = metadata.get("method") == "physics"
    return dict(title="Constant-acceleration physics baseline" if physics_only else "Engine-free neural fracture rollout",
                source="constant-acceleration physical prediction; no learned dynamics or fracture generation" if physics_only
                       else "trained neural dynamics with optional proximal consensus",
                bodies=[dict(id=0, **first, color="#d3ab75")], frames=frames,
                metadata={**metadata, **{key: initial.metadata[key] for key in ("floor_height_m", "up_axis") if key in initial.metadata},
                          "simulation_executed": True, "suggested_playback_speed": 0.005 if "floor_height_m" in initial.metadata else 0.001,
                          "geometry": "deformed tetrahedron exterior and both sides of opened original faces"})


@torch.no_grad()
def rollout_dynamics(checkpoint_path, initial_path, output_dir, steps, method="consensus", config=None, device="auto", progress=None,
                     *, cancel=None):
    if method not in ("physics", "raw", "neural", "central", "consensus") or isinstance(steps, bool) or not isinstance(steps, int) or steps < 1:
        raise ValueError("select physics/raw/neural/central/consensus and a positive number of steps")
    config = config or RolloutConfig()
    initial = load_initial(initial_path)
    teacher_physics = initial.metadata.get("teacher_physics", {})
    teacher_parameters = ("rayleigh_mass_damping", "rayleigh_stiffness_damping", "friction_coefficient")
    teacher_compatibility = ("unknown_settings_not_fully_recorded" if not all(key in teacher_physics for key in teacher_parameters)
                             else "undamped_frictionless_settings_recorded" if all(teacher_physics[key] == 0 for key in teacher_parameters)
                             else "recorded_damping_or_friction_requires_a_different_physics_model")
    model, neural_corrector, checkpoint, device = load_dynamics_model(checkpoint_path, device)
    if config.kinematic_rule == "auto":
        config = replace(config, kinematic_rule="backward_euler" if config.mechanics_mode != "disabled"
                         else model.config.get("motion_prior", "backward_euler"))
    if method in {"central", "consensus"}:
        unsupported = {key: teacher_physics[key] for key in teacher_parameters if teacher_physics.get(key, 0) != 0}
        if unsupported:
            raise ValueError(f"the constrained rollout requires undamped frictionless teacher data; regenerate with zero damping and friction: {unsupported}")
    mesh_seen = initial.topology.fingerprint in checkpoint["training_mesh_fingerprints"]
    heldout_shape = checkpoint.get("generalization_protocol") == "shape_family" and initial.topology.fingerprint in checkpoint.get("evaluation_mesh_fingerprints", [])
    if not mesh_seen and not heldout_shape:
        raise ValueError("this checkpoint was not trained for the selected material mesh")
    if method == "neural" and neural_corrector is None:
        raise ValueError("checkpoint contains no trained neural corrector; use --corrector-epochs during training")
    learned_supports = initial.metadata["support_policy"] == "learned_material_corner_release"
    if learned_supports and not model.config.get("learn_support_release", False):
        raise ValueError("learned support release requires a checkpoint trained with --learn-support-release")
    if device.type == "cpu":
        torch.set_num_threads(min(torch.get_num_threads(), 4))
    directory = new_directory(output_dir)
    positions, velocities, old = initial.positions.copy(), initial.velocities.copy(), initial.broken.copy()
    energy_model = _mechanical_energy(initial, config.mechanics_mode != "disabled")
    fixed = initial.fixed.copy()
    position_history, velocity_history, cut_history = [positions.copy()], [velocities.copy()], [old.copy()]
    fixed_history = [fixed.copy()]
    times = [initial.time]
    metrics = [dict(broken_faces=int(old.sum()), inverted_tetrahedra=0,
                    material_components=len(np.unique(component_labels(len(initial.topology.mesh.tets), initial.topology.mesh.dual_edges, old))))]
    events, status = [], "completed"
    started = perf_counter()
    predictor_calls, neural_calls, flow_calls, optimization_solves, iterations = 0, 0, 0, 0, 0
    try:
        for step in range(steps):
            if cancel is not None and cancel():
                raise KeyboardInterrupt
            batch = state_batch(initial.topology, positions, velocities, old, fixed,
                                initial.gravity, initial.material, initial.dt, device,
                                initial.metadata.get("floor_height_m"))
            if method == "physics":
                proposal = {**constant_acceleration_prediction(batch),
                            "fracture_probability": batch["broken"].to(batch["positions"].dtype),
                            "support_release_probability": torch.zeros_like(batch["masses"])}
                candidates = [old.copy()]
            else:
                if config.fracture_samples and len(batch["broken"]):
                    proposal, flow_context = model.forward_with_flow_context(batch)
                else:
                    proposal, flow_context = model(batch), None
                predictor_calls += 1
                _validate_proposal(proposal, positions, velocities)
                candidates = fracture_candidates(model, batch, proposal, config, step, context=flow_context)
                flow_calls += config.fracture_samples
            corrected = neural_corrector(batch, proposal) if method == "neural" else proposal
            neural_calls += int(method == "neural")
            _validate_proposal(corrected, positions, velocities)
            probability = _array(corrected["fracture_probability"])
            candidate_records, accepted = [], []
            for index, mask in enumerate(candidates):
                if cancel is not None and cancel():
                    raise KeyboardInterrupt
                prior = 0.0 if config.fracture_policy == "shared_sample" else config.fracture_prior_weight * _fracture_cost(mask, probability, old)
                candidate_hash = hashlib.sha256(mask.tobytes()).hexdigest()
                if accepted and prior > min(item[0] for item in accepted) + config.absolute_tolerance:
                    candidate_records.append(dict(candidate=index, candidate_hash=candidate_hash, prior=prior,
                                                  status="pruned_by_nonnegative_correction_cost_lower_bound"))
                    continue
                next_fixed = release_supports(initial.topology.mesh, mask, fixed,
                                              _array(corrected["support_release_probability"])) if learned_supports else fixed.copy()
                candidate_proposal = _candidate_proposal(corrected, positions, velocities,
                                                          fixed if config.mechanics_mode != "disabled" else next_fixed) if learned_supports else corrected
                if method in ("central", "consensus"):
                    result, diagnostic = _correct(initial, positions, velocities, candidate_proposal, mask, next_fixed, model, config, method, cancel,
                                                  energy_model=energy_model, progress=progress, previous_broken=old, previous_fixed=fixed)
                    optimization_solves += len(diagnostic.get("solves", []))
                    iterations += sum(item["iterations"] for item in diagnostic.get("solves", []))
                    if result is None:
                        if diagnostic["status"] == "interrupted":
                            raise KeyboardInterrupt
                        candidate_records.append(dict(candidate=index, candidate_hash=candidate_hash,
                                                      prior=prior, valid=False, correction=diagnostic))
                        continue
                    next_positions, next_velocities = result.positions.reshape(-1, 3), result.velocities.reshape(-1, 3)
                    objective = prior + result.objective
                else:
                    next_positions, next_velocities = _proposal_arrays(candidate_proposal, positions, velocities)
                    next_positions[next_fixed] = positions[next_fixed]
                    next_velocities[next_fixed] = 0
                    geometry = geometry_diagnostics(initial.topology.mesh, next_positions, mask, config.geometry_tolerance,
                                                    initial.metadata.get("floor_height_m"))
                    swept = swept_geometry_diagnostics(initial.topology.mesh, positions, next_positions, mask,
                                                       config.geometry_tolerance, initial.metadata.get("floor_height_m"))
                    diagnostic = dict(status="unconstrained_prediction", geometry=geometry, swept_geometry=swept)
                    diagnostic["mechanics"] = state_mechanics_diagnostics(initial, positions, velocities, next_positions,
                                                                         next_velocities, old, mask, fixed, next_fixed,
                                                                         model, config)
                    objective = prior
                balance = step_energy_diagnostics(initial, positions, velocities, next_positions, next_velocities,
                                                   old, fixed, model, config, nonlinear=energy_model)
                energy_valid = balance["energy_satisfied"]
                diagnostic["energy"] = balance
                record = dict(candidate=index, prior=prior, objective=objective, broken_faces=int(mask.sum()),
                              candidate_hash=candidate_hash,
                              valid=diagnostic["geometry"]["valid"], correction=diagnostic)
                candidate_records.append(record)
                accepted.append((objective, index, mask, next_positions, next_velocities, next_fixed, record))
            if not accepted:
                status = "timestep_rejected"
                events.append(dict(step=step + 1, time=times[-1], status=status, candidates=candidate_records))
                break
            old_broken, old_fixed = old.copy(), fixed.copy()
            start_elastic_state = current_elastic_state(initial, positions, old_broken, old_fixed)
            _, chosen, old, positions, velocities, fixed, record = min(accepted, key=lambda item: (item[0], item[1]))
            if config.mechanics_mode != "disabled":
                initial = advance_elastic_state(initial, positions, velocities, old_broken, old, old_fixed, fixed, start_elastic_state)
            if not np.isfinite(positions).all() or not np.isfinite(velocities).all():
                status = "nonfinite_prediction"
                break
            times.append(initial.time + (step + 1) * initial.dt)
            position_history.append(positions.copy())
            velocity_history.append(velocities.copy())
            cut_history.append(old.copy())
            fixed_history.append(fixed.copy())
            metrics.append(dict(broken_faces=int(old.sum()), objective=record["objective"],
                                material_components=len(np.unique(component_labels(len(initial.topology.mesh.tets), initial.topology.mesh.dual_edges, old))),
                                inverted_tetrahedra=record["correction"]["geometry"].get("inverted_tetrahedra", 0),
                                overlap_pairs=record["correction"]["geometry"].get("overlapping_tetrahedron_pairs", 0),
                                mechanical_energy_j=record["correction"]["energy"]["components"]["total"],
                                energy_violation=record["correction"]["energy"]["normalized_violation"]))
            event = dict(step=step + 1, time=times[-1], selected_candidate=chosen, status="committed",
                         geometry_valid=record["valid"], candidate_pool=[item["candidate_hash"] for item in candidate_records if "candidate_hash" in item],
                         energy_valid=record["correction"]["energy"]["energy_satisfied"],
                         swept_geometry_valid=record["correction"]["swept_geometry"]["valid"],
                         mechanics=record["correction"].get("mechanics", {}),
                         candidates=candidate_records)
            events.append(event)
            with (directory / "progress.jsonl").open("a", encoding="utf-8") as stream:
                stream.write(json.dumps(event, allow_nan=False) + "\n")
            if progress:
                progress(dict(method=method, step=step + 1, time=times[-1], broken_faces=int(old.sum()),
                              geometry_valid=record["valid"], corrector_iterations=iterations))
    except KeyboardInterrupt:
        status = "interrupted"
    except FloatingPointError as error:
        status = "nonfinite_prediction"
        events.append(dict(step=step + 1, time=times[-1], status=status, reason=str(error)))
    condition_splits = checkpoint["condition_splits"]
    split = next((name for name, conditions in condition_splits.items() if initial.condition_digest in conditions), "unseen_condition")
    committed = [event for event in events if event.get("status") == "committed"]
    report = dict(format="fracture-world.neural-rollout.v1", status=status, method=method,
                  requested_steps=steps, completed_steps=len(times) - 1, engine_calls=0,
                  checkpoint_sha256=file_hash(checkpoint_path), initial_sha256=file_hash(initial_path),
                  case_id=initial.case_id, condition_digest=initial.condition_digest, condition_split=split,
                  mesh_fingerprint=initial.topology.fingerprint,
                  mesh_seen_during_training=mesh_seen, held_out_shape_family=bool(heldout_shape and not mesh_seen),
                  shape_family=checkpoint.get("mesh_shape_families", {}).get(initial.topology.fingerprint, initial.topology.fingerprint),
                  generalization_protocol=checkpoint.get("generalization_protocol", "loading_condition"),
                  floor_height_m=initial.metadata.get("floor_height_m"),
                  overfit_debug=checkpoint["training"]["overfit"], configuration=asdict(config),
                  predictor_calls=predictor_calls, neural_corrector_calls=neural_calls,
                  learned_motion_corrector_calls=predictor_calls,
                  physical_predictor="constant_acceleration" if method == "physics" else model.config.get("motion_prior", "backward_euler"),
                  learned_motion_role="none" if method == "physics" else "state-dependent correction of the physical motion prior",
                  predictor_parameters=sum(parameter.numel() for parameter in model.parameters()),
                  neural_corrector_parameters=sum(parameter.numel() for parameter in neural_corrector.parameters()) if method == "neural" else 0,
                  fracture_flow_calls=flow_calls, optimization_solves=optimization_solves,
                  flow_architecture=model.config.get("flow_architecture", "independent"),
                  probability_policy=checkpoint.get("probability_policy", "legacy_weighted_logits"),
                  corrector_iterations=iterations, runtime_seconds=perf_counter() - started,
                  support_policy=initial.metadata["support_policy"],
                  support_release_training_events=checkpoint.get("support_release_training_events", 0),
                  fracture_training_events=checkpoint.get("fracture_training_events", 0),
                  fracture_model_has_positive_training_examples=checkpoint.get("fracture_training_events", 0) > 0,
                  learned_support_release_has_positive_training_examples=checkpoint.get("support_release_training_events", 0) > 0,
                  teacher_future_used=False, runtime_native_required=False,
                  teacher_physics_compatibility=teacher_compatibility,
                  all_committed_geometry_valid=all(event.get("geometry_valid", False) for event in committed) if committed else None,
                  all_committed_energy_valid=all(event.get("energy_valid", False) for event in committed) if committed else None,
                  all_committed_swept_geometry_valid=all(event.get("swept_geometry_valid", False) for event in committed) if committed else None,
                  checked_constraint_steps=len(committed),
                  mechanics_mode=config.mechanics_mode,
                  mechanics_weight=config.mechanics_weight,
                  fracture_initiation_enforced=config.fracture_initiation and method in {"central", "consensus"},
                  all_committed_elastic_equalities_valid=all(event.get("mechanics", {}).get("elastic_dynamics", {}).get("status") == "pass"
                                                           for event in committed) if committed and config.mechanics_mode != "disabled" else None,
                  elastic_equality_check_scope="residual against the pinned numerical equation; proximal mode minimizes rather than enforces zero residual",
                  mechanics_scope="frozen-rotation implicit teacher model; contact/support reactions remain partially unknown",
                  mechanical_energy_law="corotational elastic + kinetic + gravitational; static boundaries, no supplied fracture-energy credit",
                  energy_balance_enforced=bool(config.energy_balance and method in {"central", "consensus"}),
                  energy_velocity_scale=float(model.config["velocity_scale"]),
                  contact_law=("frictionless normal nonpenetration correction; restitution and contact impulse law not independently enforced"
                               if method in {"central", "consensus"} else "not enforced; endpoint and swept geometry are diagnostics only"),
                  constitutive_response=("none; constant acceleration with prescribed current supports" if method == "physics"
                                         else "learned predictor; no explicit elastic constitutive solve"),
                  fracture_choice=("initial fracture topology held fixed; no new fracture generation" if method == "physics"
                                   else "one shared flow sample; fixed fracture mask during each state correction" if config.fracture_policy == "shared_sample"
                                   else "FM samples only; independent-marginal surrogate plus feasible motion/energy correction cost; not joint flow likelihood" if config.fracture_policy == "fm_search"
                                   else "legacy candidate search with independent-marginal surrogate plus motion correction cost; not joint flow likelihood"),
                  events=events)
    np.savez_compressed(directory / "trajectory.npz", times=np.asarray(times), positions=np.asarray(position_history),
                        velocities=np.asarray(velocity_history), broken=np.asarray(cut_history),
                        fixed=np.asarray(fixed_history),
                        metadata_json=json.dumps({key: value for key, value in report.items() if key != "events"}, allow_nan=False))
    write_json(directory / "report.json", report)
    write_json(directory / "replay.json", _replay(initial, position_history, cut_history, fixed_history, times, metrics,
                                                  {key: value for key, value in report.items() if key != "events"}))
    return report
