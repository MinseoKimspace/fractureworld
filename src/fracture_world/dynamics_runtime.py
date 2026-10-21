from dataclasses import asdict, dataclass, replace
import hashlib
import json
from pathlib import Path
from time import perf_counter

import numpy as np
import torch

from .dynamics_corrector import CorrectorTolerances, DynamicsProblem, solve_central, solve_consensus
from .dynamics_energy import CorotationalEnergy, EnergyProjection
from .dynamics_energy_solvers import solve_energy
from .dynamics_geometry import corner_groups, floor_contacts, geometry_diagnostics, linearized_contacts, render_geometry, swept_geometry_diagnostics
from .dynamics_model import constant_acceleration_prediction
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

    def __post_init__(self):
        if self.fracture_samples < 0 or self.flow_steps < 1:
            raise ValueError("fracture_samples must be nonnegative and flow_steps positive")
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
            if value is not None and (not isinstance(value, int) or value < 1):
                raise ValueError("explicit iteration budgets must be positive or null")
        if self.kinematic_rule not in ("auto", "backward_euler", "trapezoidal", "constant_acceleration"):
            raise ValueError("unsupported kinematic rule")
        if not isinstance(self.energy_balance, bool):
            raise ValueError("energy_balance must be boolean")


def load_rollout_config(path=None):
    return RolloutConfig(**json.loads(Path(path).read_text(encoding="utf-8"))) if path else RolloutConfig()


def _array(value):
    return value.detach().cpu().numpy().astype(float, copy=True)


def _proposal_arrays(proposal, positions, velocities):
    next_positions = positions + _array(proposal["position_increment"]) if "position_increment" in proposal else _array(proposal["positions"])
    next_velocities = velocities + _array(proposal["velocity_increment"]) if "velocity_increment" in proposal else _array(proposal["velocities"])
    return next_positions, next_velocities


def _validate_proposal(proposal, positions, velocities):
    fields = ("positions", "velocities", "position_increment", "velocity_increment", "fracture_probability",
              "support_release_probability", "fracture_logits", "support_release_logits")
    invalid = [key for key in fields if key in proposal and not torch.isfinite(proposal[key]).all()]
    if invalid:
        raise FloatingPointError(f"nonfinite prediction fields: {', '.join(invalid)}")
    next_positions, next_velocities = _proposal_arrays(proposal, positions, velocities)
    if not np.isfinite(next_positions).all() or not np.isfinite(next_velocities).all():
        raise FloatingPointError("nonfinite predicted material-corner state")


def fracture_candidates(model, batch, proposal, config, step):
    old = batch["broken"].detach().cpu().numpy()
    candidates = [] if config.fracture_policy in {"shared_sample", "fm_search"} else [old | (_array(proposal["fracture_probability"]) >= 0.5), old.copy()]
    for index in range(config.fracture_samples):
        generator = torch.Generator(device=batch["positions"].device).manual_seed(config.seed + step * 1000003 + index)
        noise = torch.randn(len(old), generator=generator, device=batch["positions"].device)
        field = _array(model.sample_fracture(batch, config.flow_steps, noise))
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


def _problem(initial, positions, velocities, proposed_positions, proposed_velocities, broken, fixed, contacts, model, config):
    mesh = initial.topology.mesh
    full_broken = np.zeros(len(mesh.faces.vertices), dtype=bool)
    full_broken[mesh.internal_faces] = broken
    shape = (len(mesh.tets), 4, 3)
    return DynamicsProblem(mesh, positions.reshape(shape), velocities.reshape(shape), proposed_positions.reshape(shape),
                           proposed_velocities.reshape(shape), initial.dt, initial.material["rest_density"],
                           full_broken, fixed.reshape(-1, 4),
                           (initial.dt * initial.topology.masses[:, None] * initial.gravity).reshape(shape), contacts,
                           initial.dt * model.config["velocity_scale"], model.config["velocity_scale"],
                           kinematic_rule=model.config.get("motion_prior", "backward_euler") if config.kinematic_rule == "auto" else config.kinematic_rule)


def _mechanical_energy(initial):
    material = initial.material
    return CorotationalEnergy(initial.topology.mesh, material["rest_density"], material["youngs_modulus"],
                              material["poissons_ratio"], initial.gravity)


def _correct(initial, positions, velocities, proposal, broken, fixed, model, config, method, cancel=None, *, energy_model=None, progress=None):
    mesh = initial.topology.mesh
    proposed_positions, proposed_velocities = _proposal_arrays(proposal, positions, velocities)
    rule = model.config.get("motion_prior", "backward_euler") if config.kinematic_rule == "auto" else config.kinematic_rule
    trial_positions = positions + initial.dt * velocities + (1.0 if rule == "backward_euler" else 0.5) * initial.dt**2 * initial.gravity
    groups = corner_groups(mesh, broken)
    held = np.isin(groups, groups[fixed])
    trial_positions[held] = positions[held]
    tolerance = CorrectorTolerances(config.absolute_tolerance, config.relative_tolerance)
    energy_model = energy_model or _mechanical_energy(initial)
    budget = energy_model.value(positions, velocities)
    histories, previous_trials = [], []
    outer = 0
    while config.max_contact_iterations is None or outer < config.max_contact_iterations:
        if cancel is not None and cancel():
            return None, dict(status="interrupted", outer_iterations=outer, solves=histories)
        try:
            contacts = linearized_contacts(mesh, positions, trial_positions, broken, config.geometry_tolerance)
            if "floor_height_m" in initial.metadata:
                contacts += floor_contacts(positions, trial_positions, initial.metadata["floor_height_m"], config.geometry_tolerance)
        except ValueError as error:
            return None, dict(status="invalid_contact_geometry", reason=str(error), outer_iterations=outer)
        problem = _problem(initial, positions, velocities, proposed_positions, proposed_velocities, broken, fixed,
                           tuple(contacts), model, config)
        if progress:
            progress(dict(event="corrector_started", method=method, contact_outer_iteration=outer + 1,
                          contact_constraints=len(contacts), material_corners=problem.corner_count))
        if config.energy_balance:
            energy = EnergyProjection(problem, energy_model, budget, config.absolute_tolerance, config.relative_tolerance)
            result = solve_energy(problem, energy, method, tolerance, config.rho, config.max_iterations, cancel, progress=progress)
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
        if not result.converged:
            return None, dict(status=result.status, outer_iterations=outer, solves=histories, geometry=geometry)
        if geometry["valid"]:
            swept = swept_geometry_diagnostics(mesh, positions, result.positions, broken, config.geometry_tolerance,
                                               initial.metadata.get("floor_height_m"))
            if not swept["valid"]:
                return None, dict(status="continuous_geometry_unresolved", outer_iterations=outer, solves=histories,
                                  geometry=geometry, swept_geometry=swept)
            return result, dict(status="converged", outer_iterations=outer, solves=histories, geometry=geometry,
                                swept_geometry=swept, contact_initialization="physical free motion with prescribed supports")
        if geometry.get("inverted_tetrahedra", 0):
            return None, dict(status="inverted_tetrahedra", outer_iterations=outer, solves=histories, geometry=geometry)
        if np.allclose(trial_positions, result.positions.reshape(-1, 3), rtol=0, atol=config.geometry_tolerance):
            return None, dict(status="contact_linearization_stagnation", outer_iterations=outer, solves=histories, geometry=geometry)
        if any(np.allclose(previous, result.positions.reshape(-1, 3), rtol=0, atol=config.geometry_tolerance)
               for previous in previous_trials):
            return None, dict(status="contact_linearization_cycle", outer_iterations=outer, solves=histories, geometry=geometry)
        previous_trials = (previous_trials + [trial_positions.copy()])[-2:]
        trial_positions = result.positions.reshape(-1, 3)
    return None, dict(status="contact_iteration_budget", outer_iterations=outer, solves=histories)


def release_supports(mesh, broken, fixed, release_probability):
    proposed = fixed & (np.asarray(release_probability) < 0.5)
    groups = corner_groups(mesh, broken)
    retained = np.zeros(groups.max() + 1, dtype=bool)
    np.logical_or.at(retained, groups, proposed)
    return fixed & retained[groups]


def _replay(initial, positions, broken, fixed, times, metrics, metadata):
    mesh = initial.topology.mesh
    frames = []
    for time, state, cuts, supports, values in zip(times, positions, broken, fixed, metrics):
        frames.append(dict(time=float(time), poses=[dict(position=[0, 0, 0], orientation=[0, 0, 0, 1])],
                           geometries=[render_geometry(mesh, state, cuts, supports)], metrics=values))
    first = frames[0]["geometries"][0]
    return dict(title="Engine-free neural fracture rollout", source="trained neural dynamics with optional proximal consensus",
                bodies=[dict(id=0, **first, color="#d3ab75")], frames=frames,
                metadata={**metadata, **{key: initial.metadata[key] for key in ("floor_height_m", "up_axis") if key in initial.metadata},
                          "simulation_executed": True, "suggested_playback_speed": 0.005 if "floor_height_m" in initial.metadata else 0.001,
                          "geometry": "deformed tetrahedron exterior and both sides of opened original faces"})


@torch.no_grad()
def rollout_dynamics(checkpoint_path, initial_path, output_dir, steps, method="consensus", config=None, device="auto", progress=None,
                     *, cancel=None):
    if method not in ("physics", "raw", "neural", "central", "consensus") or not isinstance(steps, int) or steps < 1:
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
        config = replace(config, kinematic_rule=model.config.get("motion_prior", "backward_euler"))
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
    energy_model = _mechanical_energy(initial)
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
                proposal = model(batch)
                predictor_calls += 1
                _validate_proposal(proposal, positions, velocities)
                candidates = fracture_candidates(model, batch, proposal, config, step)
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
                if method in ("central", "consensus"):
                    result, diagnostic = _correct(initial, positions, velocities, proposal, mask, next_fixed, model, config, method, cancel,
                                                  energy_model=energy_model, progress=progress)
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
                    next_positions, next_velocities = _proposal_arrays(corrected, positions, velocities)
                    next_positions[next_fixed] = positions[next_fixed]
                    next_velocities[next_fixed] = 0
                    geometry = geometry_diagnostics(initial.topology.mesh, next_positions, mask, config.geometry_tolerance,
                                                    initial.metadata.get("floor_height_m"))
                    swept = swept_geometry_diagnostics(initial.topology.mesh, positions, next_positions, mask,
                                                       config.geometry_tolerance, initial.metadata.get("floor_height_m"))
                    diagnostic = dict(status="unconstrained_prediction", geometry=geometry, swept_geometry=swept)
                    objective = prior
                balance = energy_model.balance(positions, velocities, next_positions, next_velocities, model.config["velocity_scale"],
                                               absolute_tolerance=config.absolute_tolerance, relative_tolerance=config.relative_tolerance)
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
            _, chosen, old, positions, velocities, fixed, record = min(accepted, key=lambda item: (item[0], item[1]))
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
                                overlap_pairs=record["correction"]["geometry"].get("overlapping_nonincident_tetrahedron_pairs", 0),
                                mechanical_energy_j=record["correction"]["energy"]["components"]["total"],
                                energy_violation=record["correction"]["energy"]["normalized_violation"]))
            event = dict(step=step + 1, time=times[-1], selected_candidate=chosen, status="committed",
                         geometry_valid=record["valid"], candidate_pool=[item["candidate_hash"] for item in candidate_records if "candidate_hash" in item],
                         energy_valid=record["correction"]["energy"]["energy_satisfied"],
                         swept_geometry_valid=record["correction"]["swept_geometry"]["valid"],
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
                  mechanical_energy_law="corotational elastic + kinetic + gravitational; static boundaries, no supplied fracture-energy credit",
                  energy_balance_enforced=bool(config.energy_balance and method in {"central", "consensus"}),
                  energy_velocity_scale=float(model.config["velocity_scale"]),
                  contact_law="frictionless normal nonpenetration correction; restitution and contact impulse law not independently enforced",
                  constitutive_response="learned predictor; no explicit elastic constitutive solve",
                  fracture_choice=("one shared flow sample; fixed fracture mask during each state correction" if config.fracture_policy == "shared_sample"
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
