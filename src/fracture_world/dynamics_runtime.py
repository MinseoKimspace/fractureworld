from dataclasses import asdict, dataclass
import hashlib
import json
from pathlib import Path
from time import perf_counter

import numpy as np
import torch

from .dynamics_corrector import CorrectorTolerances, DynamicsProblem, solve_central, solve_consensus
from .dynamics_geometry import corner_groups, geometry_diagnostics, linearized_contacts, render_geometry
from .dynamics_initial import load_initial
from .dynamics_learning import load_dynamics_model, state_batch
from .io import file_hash, new_directory, write_json
from .partition import component_labels


@dataclass(frozen=True)
class RolloutConfig:
    fracture_samples: int = 2
    fracture_policy: str = "marginal_search"
    flow_steps: int = 8
    fracture_prior_weight: float = 1.0
    rho: float = 0.01
    absolute_tolerance: float = 1e-8
    relative_tolerance: float = 1e-7
    geometry_tolerance: float = 1e-6
    max_iterations: int | None = None
    max_contact_iterations: int | None = None
    kinematic_rule: str = "backward_euler"
    seed: int = 17

    def __post_init__(self):
        if self.fracture_samples < 0 or self.flow_steps < 1:
            raise ValueError("fracture_samples must be nonnegative and flow_steps positive")
        if self.fracture_policy not in {"marginal_search", "shared_sample"}:
            raise ValueError("fracture_policy must select marginal_search or shared_sample")
        if self.fracture_policy == "shared_sample" and self.fracture_samples != 1:
            raise ValueError("shared_sample requires exactly one fracture sample")
        values = [self.fracture_prior_weight, self.rho, self.absolute_tolerance, self.geometry_tolerance]
        if not np.isfinite([*values, self.relative_tolerance]).all() or min(values) <= 0 or self.relative_tolerance < 0:
            raise ValueError("rollout weights and tolerances must be finite and positive")
        for value in (self.max_iterations, self.max_contact_iterations):
            if value is not None and (not isinstance(value, int) or value < 1):
                raise ValueError("explicit iteration budgets must be positive or null")
        if self.kinematic_rule not in ("backward_euler", "trapezoidal"):
            raise ValueError("unsupported kinematic rule")


def load_rollout_config(path=None):
    return RolloutConfig(**json.loads(Path(path).read_text(encoding="utf-8"))) if path else RolloutConfig()


def _array(value):
    return value.detach().cpu().numpy().astype(float, copy=True)


def fracture_candidates(model, batch, proposal, config, step):
    old = batch["broken"].detach().cpu().numpy()
    candidates = [] if config.fracture_policy == "shared_sample" else [old | (_array(proposal["fracture_probability"]) >= 0.5), old.copy()]
    for index in range(config.fracture_samples):
        generator = torch.Generator(device=batch["positions"].device).manual_seed(config.seed + step * 1000003 + index)
        noise = torch.randn(len(old), generator=generator, device=batch["positions"].device)
        candidates.append(old | (_array(model.sample_fracture(batch, config.flow_steps, noise)) >= 0))
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
                           kinematic_rule=config.kinematic_rule)


def _correct(initial, positions, velocities, proposal, broken, fixed, model, config, method, cancel=None):
    mesh = initial.topology.mesh
    proposed_positions, proposed_velocities = _array(proposal["positions"]), _array(proposal["velocities"])
    trial_positions = proposed_positions.copy()
    tolerance = CorrectorTolerances(config.absolute_tolerance, config.relative_tolerance)
    histories, contact_keys, contacts = [], set(), []
    outer = 0
    while config.max_contact_iterations is None or outer < config.max_contact_iterations:
        if cancel is not None and cancel():
            return None, dict(status="interrupted", outer_iterations=outer, solves=histories)
        try:
            additions = linearized_contacts(mesh, positions, trial_positions, broken, config.geometry_tolerance)
        except ValueError as error:
            return None, dict(status="invalid_contact_geometry", reason=str(error), outer_iterations=outer)
        for contact in additions:
            key = (tuple(contact.corner_ids), tuple(np.round(contact.normal, 12)), tuple(contact.weights), round(contact.gap, 12))
            if key not in contact_keys:
                contacts.append(contact)
                contact_keys.add(key)
        problem = _problem(initial, positions, velocities, proposed_positions, proposed_velocities, broken, fixed,
                           tuple(contacts), model, config)
        result = solve_central(problem, tolerance, config.max_iterations, cancel=cancel) if method == "central" else solve_consensus(
            problem, tolerances=tolerance, rho=config.rho, max_iterations=config.max_iterations, cancel=cancel)
        outer += 1
        histories.append(dict(outer_iteration=outer, status=result.status, iterations=result.iterations,
                              objective=result.objective, diagnostics=result.diagnostics,
                              history=[asdict(item) for item in result.history]))
        geometry = geometry_diagnostics(mesh, result.positions, broken, config.geometry_tolerance)
        if not result.converged:
            return None, dict(status=result.status, outer_iterations=outer, solves=histories, geometry=geometry)
        if geometry["valid"]:
            return result, dict(status="converged", outer_iterations=outer, solves=histories, geometry=geometry)
        if geometry.get("inverted_tetrahedra", 0):
            return None, dict(status="inverted_tetrahedra", outer_iterations=outer, solves=histories, geometry=geometry)
        if np.allclose(trial_positions, result.positions.reshape(-1, 3), rtol=0, atol=config.geometry_tolerance):
            return None, dict(status="contact_linearization_stagnation", outer_iterations=outer, solves=histories, geometry=geometry)
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
                metadata={**metadata, "simulation_executed": True, "suggested_playback_speed": 0.001,
                          "geometry": "deformed tetrahedron exterior and both sides of opened original faces"})


@torch.no_grad()
def rollout_dynamics(checkpoint_path, initial_path, output_dir, steps, method="consensus", config=None, device="auto", progress=None,
                     *, cancel=None):
    if method not in ("raw", "neural", "central", "consensus") or not isinstance(steps, int) or steps < 1:
        raise ValueError("select raw/neural/central/consensus and a positive number of steps")
    config = config or RolloutConfig()
    initial = load_initial(initial_path)
    model, neural_corrector, checkpoint, device = load_dynamics_model(checkpoint_path, device)
    if initial.topology.fingerprint not in checkpoint["training_mesh_fingerprints"]:
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
                                initial.gravity, initial.material, initial.dt, device)
            proposal = model(batch)
            predictor_calls += 1
            candidates = fracture_candidates(model, batch, proposal, config, step)
            flow_calls += config.fracture_samples
            corrected = neural_corrector(batch, proposal) if method == "neural" else proposal
            neural_calls += int(method == "neural")
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
                    result, diagnostic = _correct(initial, positions, velocities, proposal, mask, next_fixed, model, config, method, cancel)
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
                    next_positions, next_velocities = _array(corrected["positions"]), _array(corrected["velocities"])
                    next_positions[next_fixed] = positions[next_fixed]
                    next_velocities[next_fixed] = 0
                    geometry = geometry_diagnostics(initial.topology.mesh, next_positions, mask, config.geometry_tolerance)
                    diagnostic = dict(status="unconstrained_prediction", geometry=geometry)
                    objective = prior
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
                                overlap_pairs=record["correction"]["geometry"].get("overlapping_nonincident_tetrahedron_pairs", 0)))
            event = dict(step=step + 1, time=times[-1], selected_candidate=chosen, status="committed",
                         geometry_valid=record["valid"], candidate_pool=[item["candidate_hash"] for item in candidate_records if "candidate_hash" in item],
                         candidates=candidate_records)
            events.append(event)
            with (directory / "progress.jsonl").open("a", encoding="utf-8") as stream:
                stream.write(json.dumps(event, allow_nan=False) + "\n")
            if progress:
                progress(dict(method=method, step=step + 1, time=times[-1], broken_faces=int(old.sum()),
                              geometry_valid=record["valid"], corrector_iterations=iterations))
    except KeyboardInterrupt:
        status = "interrupted"
    condition_splits = checkpoint["condition_splits"]
    split = next((name for name, conditions in condition_splits.items() if initial.condition_digest in conditions), "unseen_condition")
    report = dict(format="fracture-world.neural-rollout.v1", status=status, method=method,
                  requested_steps=steps, completed_steps=len(times) - 1, engine_calls=0,
                  checkpoint_sha256=file_hash(checkpoint_path), initial_sha256=file_hash(initial_path),
                  case_id=initial.case_id, condition_digest=initial.condition_digest, condition_split=split,
                  mesh_fingerprint=initial.topology.fingerprint,
                  overfit_debug=checkpoint["training"]["overfit"], configuration=asdict(config),
                  predictor_calls=predictor_calls, neural_corrector_calls=neural_calls,
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
                  all_committed_geometry_valid=all(event.get("geometry_valid", False) for event in events if event.get("status") == "committed"),
                  contact_law="frictionless normal nonpenetration correction; restitution and contact impulse law not independently enforced",
                  constitutive_response="learned predictor; no explicit elastic constitutive solve",
                  fracture_choice=("one shared flow sample; fixed fracture mask during each state correction" if config.fracture_policy == "shared_sample"
                                   else "candidate search with independent-marginal surrogate plus motion correction cost; not joint flow likelihood"),
                  events=events)
    np.savez_compressed(directory / "trajectory.npz", times=np.asarray(times), positions=np.asarray(position_history),
                        velocities=np.asarray(velocity_history), broken=np.asarray(cut_history),
                        fixed=np.asarray(fixed_history),
                        metadata_json=json.dumps({key: value for key, value in report.items() if key != "events"}, allow_nan=False))
    write_json(directory / "report.json", report)
    write_json(directory / "replay.json", _replay(initial, position_history, cut_history, fixed_history, times, metrics,
                                                  {key: value for key, value in report.items() if key != "events"}))
    return report
