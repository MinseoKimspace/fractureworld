from dataclasses import asdict, replace
import hashlib
from itertools import combinations
import json
from pathlib import Path
from time import perf_counter

import numpy as np
import torch

from . import dynamics_runtime as runtime
from .dynamics_data import _frozen, _teacher_physics, load_dynamics_dataset, load_dynamics_trajectory
from .dynamics_evaluation import _native_clock, _new_fracture_classification
from .dynamics_geometry import floor_contacts, geometry_diagnostics, linearized_contacts, swept_geometry_diagnostics
from .dynamics_initial import DynamicsInitial
from .dynamics_learning import load_dynamics_model, state_batch
from .io import file_hash, new_directory, write_json


METHODS = ("raw", "central", "consensus")


def _state_hash(arrays, context):
    digest = hashlib.sha256(json.dumps(context, sort_keys=True, allow_nan=False).encode())
    for value in arrays:
        array = np.ascontiguousarray(value)
        digest.update(str((array.dtype.str, array.shape)).encode())
        digest.update(array.tobytes())
    return digest.hexdigest()


def _proposal_copy(proposal):
    return {key: value.detach().clone() if isinstance(value, torch.Tensor) else value for key, value in proposal.items()}


def _frozen_objective(initial, positions, velocities, old, fixed, candidate, model, config):
    old_phase = config.mechanics_mode != "disabled"
    integration_mask = old if old_phase else candidate["mask"]
    integration_fixed = fixed if old_phase else candidate["fixed"]
    rule = "backward_euler" if old_phase else config.kinematic_rule
    trial = positions + initial.dt * velocities + (1. if rule == "backward_euler" else .5) * initial.dt**2 * initial.gravity
    groups = runtime.corner_groups(initial.topology.mesh, integration_mask)
    held = np.isin(groups, groups[integration_fixed])
    trial[held] = positions[held]
    contacts = runtime.step_contacts(initial, positions, trial, integration_mask, config)
    return runtime._problem(initial, positions, velocities, candidate["positions"], candidate["velocities"],
                            candidate["mask"], candidate["fixed"], contacts, model, config,
                            previous_broken=old, previous_fixed=fixed)


def _constraint_status(geometry, swept, energy, solution, diagnostic, config):
    checks = dict(geometry=geometry.get("valid"), swept_geometry=swept.get("valid"),
                  energy=None if energy.get("status") == "unknown" else energy.get("energy_satisfied"),
                  solver_constraints=solution.diagnostics.get("constraints_satisfied") if solution is not None else None)
    mechanics = diagnostic.get("mechanics", solution.diagnostics.get("mechanics", {}) if solution is not None else {})
    for active, name in ((getattr(config, "mechanics_mode", "disabled") == "exact", "elastic_dynamics"),
                         (getattr(config, "fracture_initiation", False), "fracture_initiation")):
        if active:
            status = mechanics.get(name, {}).get("status")
            checks[name] = True if status == "pass" else False if status == "fail" else None
    known = all(isinstance(value, (bool, np.bool_)) for value in checks.values())
    valid = False if any(isinstance(value, (bool, np.bool_)) and not value for value in checks.values()) else True if known else None
    return {key: bool(value) if isinstance(value, (bool, np.bool_)) else None for key, value in checks.items()}, valid


def _compact_diagnostics(diagnostic):
    report = {key: value for key, value in diagnostic.items() if key != "solves"}
    report["solves"] = [{key: value for key, value in item.items() if key != "history"}
                        for item in diagnostic.get("solves", [])]
    return report


def _accumulate_summary(methods, pairwise, row):
    count = row.get("candidate_count")
    if isinstance(count, int):
        for values in methods.values():
            values["requested_candidates"] += count
    for record in row.get("candidates", []):
        values = methods[record["method"]]
        values["attempted_candidates"] += 1
        values["completed_candidates"] += int(record["accepted"])
        values["failed_candidates"] += int(not record["accepted"] and record["status"] != "interrupted")
        values["interrupted_candidates"] += int(record["status"] == "interrupted")
        values["unknown_candidates"] += int(record["verified_constraints"] is None)
        values["verified_candidates"] += int(record["verified_constraints"] is True)
        values["invalid_candidates"] += int(record["verified_constraints"] is False)
    for method, selected in row.get("methods", {}).items():
        if selected["selection_complete"] and selected["selected_candidate"] is not None:
            methods[method]["completed_selections"] += 1
            methods[method]["verified_selections"] += int(selected["verified_constraints"] is True)
    for comparison in row.get("selected_state_comparisons", []):
        if comparison["comparison_available"]:
            values = pairwise[tuple(comparison["methods"])]
            values["common_selected_observations"] += 1
            for metric in ("positions_rms_metres", "velocities_rms_metres_per_second", "fracture_disagreements"):
                values[metric] += comparison[metric]


def _reference_metrics(initial, positions, velocities, mask, fixed, reference):
    for name, expected in (("positions", positions), ("velocities", velocities), ("broken", mask), ("fixed", fixed)):
        value = np.asarray(reference[name])
        if value.shape != expected.shape or not np.isfinite(value).all():
            raise ValueError("matched reference arrays must be finite and match their current-state indexing")
        if name in {"broken", "fixed"} and value.dtype != np.dtype(bool):
            raise ValueError("matched reference fracture and support arrays must be Boolean")
    mass = initial.topology.masses / initial.topology.masses.sum()
    return dict(position_mass_weighted_rmse_m=float(np.sqrt(np.sum(mass * np.sum((positions - reference["positions"])**2, axis=1)))),
                velocity_mass_weighted_rmse_m_per_s=float(np.sqrt(np.sum(mass * np.sum((velocities - reference["velocities"])**2, axis=1)))),
                new_fracture=_new_fracture_classification(mask[None], np.asarray(reference["broken"])[None], initial.broken),
                support_mismatched_corners=int(np.count_nonzero(fixed != reference["fixed"])))


@torch.no_grad()
def compare_matched_step(initial, model, config=None, step=0, *, reference=None, progress=None, cancel=None):
    config = config or runtime.RolloutConfig()
    if isinstance(step, bool) or not isinstance(step, (int, np.integer)) or step < 0:
        raise ValueError("matched step must be a nonnegative integer")
    model.eval()
    device = next(model.parameters()).device
    if device.type == "cpu":
        torch.set_num_threads(min(torch.get_num_threads(), 4))
    if config.kinematic_rule == "auto":
        config = replace(config, kinematic_rule="backward_euler" if config.mechanics_mode != "disabled"
                         else model.config.get("motion_prior", "backward_euler"))
    physics = _teacher_physics(initial.metadata.get("teacher_physics", {}))
    if any(value != 0 for value in physics.values()):
        raise ValueError("matched constrained comparisons require undamped frictionless teacher settings")
    positions, velocities = np.array(initial.positions, copy=True), np.array(initial.velocities, copy=True)
    old, fixed = np.array(initial.broken, copy=True), np.array(initial.fixed, copy=True)
    batch = state_batch(initial.topology, positions, velocities, old, fixed, initial.gravity, initial.material,
                        initial.dt, device, initial.metadata.get("floor_height_m"))
    started = perf_counter()
    if config.fracture_samples and len(old):
        proposal, context = model.forward_with_flow_context(batch)
    else:
        proposal, context = model(batch), None
    runtime._validate_proposal(proposal, positions, velocities)
    masks = runtime.fracture_candidates(model, batch, proposal, config, int(step), context=context)
    proposal_seconds = perf_counter() - started
    probability = runtime._array(proposal["fracture_probability"])
    learned_supports = initial.metadata.get("support_policy") == "learned_material_corner_release"
    if learned_supports and not model.config.get("learn_support_release", False):
        raise ValueError("matched learned support release requires compatible model weights")
    input_hash = _state_hash((positions, velocities, old, fixed, initial.gravity),
                             dict(mesh=initial.topology.fingerprint, material=initial.material, dt=initial.dt,
                                  time=initial.time, floor_height_m=initial.metadata.get("floor_height_m"),
                                  support_policy=initial.metadata.get("support_policy"),
                                  configuration=asdict(config), sample_step=int(step)))
    prepared = []
    for index, mask in enumerate(masks):
        next_fixed = runtime.release_supports(initial.topology.mesh, mask, fixed,
                                              runtime._array(proposal["support_release_probability"])) if learned_supports else fixed.copy()
        candidate = runtime._candidate_proposal(proposal, positions, velocities,
                                                 fixed if config.mechanics_mode != "disabled" else next_fixed)
        xhat, vhat = runtime._proposal_arrays(candidate, positions, velocities)
        xhat[next_fixed], vhat[next_fixed] = positions[next_fixed], 0
        candidate_hash = hashlib.sha256(mask.tobytes()).hexdigest()
        prior = 0.0 if config.fracture_policy == "shared_sample" else config.fracture_prior_weight * runtime._fracture_cost(mask, probability, old)
        prepared.append(dict(index=index, mask=_frozen(mask), fixed=_frozen(next_fixed),
                             proposal=_proposal_copy(candidate), positions=_frozen(xhat), velocities=_frozen(vhat),
                             candidate_hash=candidate_hash, prior=prior,
                             proposal_hash=_state_hash((xhat, vhat, mask, next_fixed), dict(input_hash=input_hash))))
    positions, velocities, old, fixed = (_frozen(value) for value in (positions, velocities, old, fixed))
    energy = runtime._mechanical_energy(initial, config.mechanics_mode != "disabled")
    records, states = [], {}
    interrupted = False
    for candidate in prepared:
        evaluation_problem = _frozen_objective(initial, positions, velocities, old, fixed, candidate, model, config)
        for method in METHODS:
            if cancel is not None and cancel():
                interrupted = True
                break
            start = perf_counter()
            solution = None
            if method == "raw":
                next_positions, next_velocities = candidate["positions"].copy(), candidate["velocities"].copy()
                geometry = geometry_diagnostics(initial.topology.mesh, next_positions, candidate["mask"], config.geometry_tolerance,
                                                initial.metadata.get("floor_height_m"))
                swept = swept_geometry_diagnostics(initial.topology.mesh, positions, next_positions, candidate["mask"],
                                                   config.geometry_tolerance, initial.metadata.get("floor_height_m"))
                diagnostic = dict(status="uncorrected_prediction", geometry=geometry, swept_geometry=swept)
                diagnostic["mechanics"] = runtime.state_mechanics_diagnostics(
                    initial, positions, velocities, next_positions, next_velocities, old, candidate["mask"], fixed,
                    candidate["fixed"], model, config)
                objective, accepted = 0.0, True
            else:
                try:
                    solution, diagnostic = runtime._correct(initial, positions, velocities, _proposal_copy(candidate["proposal"]),
                                                           candidate["mask"], candidate["fixed"], model, config, method, cancel,
                                                           energy_model=energy, progress=progress,
                                                           previous_broken=old, previous_fixed=fixed)
                except KeyboardInterrupt:
                    solution, diagnostic = None, dict(status="interrupted", solves=[])
                accepted = solution is not None and solution.converged
                objective = solution.objective if solution is not None else None
                if accepted:
                    next_positions = np.array(solution.positions, copy=True).reshape(positions.shape)
                    next_velocities = np.array(solution.velocities, copy=True).reshape(velocities.shape)
            record = dict(candidate=candidate["index"], method=method, input_hash=input_hash,
                          candidate_hash=candidate["candidate_hash"], proposal_hash=candidate["proposal_hash"],
                          prior=candidate["prior"], correction_objective=objective,
                          selection_objective=candidate["prior"] + objective if accepted else None,
                          accepted=bool(accepted), solver_converged=bool(accepted) if method != "raw" else None,
                          status=diagnostic["status"], runtime_seconds=perf_counter() - start,
                          iterations=sum(item["iterations"] for item in diagnostic.get("solves", [])),
                          diagnostic=_compact_diagnostics(diagnostic), verified_constraints=None)
            record["verification_scope"] = "hard constraints and necessary fracture initiation; proximal elastic residual is reported separately, not certified as zero"
            if accepted:
                state = evaluation_problem.encode_state(next_positions.reshape(evaluation_problem.positions.shape),
                                                         next_velocities.reshape(evaluation_problem.velocities.shape))
                record["frozen_problem_objective"] = candidate["prior"] + evaluation_problem.objective(state)
                record["frozen_objective_scope"] = "identical prediction, start-state coefficients and physical-free-motion contact partition for every method"
                balance = runtime.step_energy_diagnostics(initial, positions, velocities, next_positions, next_velocities,
                                                           old, fixed, model, config, nonlinear=energy)
                record["diagnostic"]["energy"] = balance
                record["constraint_checks"], record["verified_constraints"] = _constraint_status(
                    diagnostic.get("geometry", {}), diagnostic.get("swept_geometry", {}), balance, solution, diagnostic, config)
                states[(candidate["index"], method)] = (next_positions.copy(), next_velocities.copy())
            records.append(record)
            if record["status"] == "interrupted":
                interrupted = True
                break
        if interrupted:
            break
    selections = {}
    for method in METHODS:
        eligible = [record for record in records if record["method"] == method and record["accepted"]]
        selected = min(eligible, key=lambda record: (record["selection_objective"], record["candidate"])) if eligible else None
        selections[method] = dict(selected_candidate=selected["candidate"] if selected else None,
                                  selected_candidate_hash=selected["candidate_hash"] if selected else None,
                                  selection_complete=not interrupted,
                                  verified_constraints=selected["verified_constraints"] if selected else None,
                                  status="selected" if selected else "no_accepted_candidate")
    reference_values = reference() if callable(reference) else reference
    if reference_values is not None:
        for record in records:
            key = record["candidate"], record["method"]
            if key in states:
                candidate = prepared[record["candidate"]]
                record["evaluation"] = _reference_metrics(initial, *states[key], candidate["mask"], candidate["fixed"], reference_values)
        for method, selected in selections.items():
            if selected["selected_candidate"] is not None:
                selected["evaluation"] = next(record["evaluation"] for record in records
                                               if record["method"] == method and record["candidate"] == selected["selected_candidate"])
    agreement = []
    indexed = {(record["candidate"], record["method"]): record for record in records}
    for candidate in prepared:
        key = candidate["index"]
        central, consensus = indexed.get((key, "central")), indexed.get((key, "consensus"))
        available = bool(central and consensus and central["accepted"] and consensus["accepted"])
        agreement.append(dict(candidate=key, comparison_available=available,
                              objective_difference_consensus_minus_central=(consensus["selection_objective"] - central["selection_objective"]) if available else None,
                              positions_rms_metres=float(np.sqrt(np.mean((states[(key, "central")][0] - states[(key, "consensus")][0])**2))) if available else None,
                              velocities_rms_metres_per_second=float(np.sqrt(np.mean((states[(key, "central")][1] - states[(key, "consensus")][1])**2))) if available else None))
    selected_pairs = []
    for first, second in combinations(METHODS, 2):
        a, b = selections[first]["selected_candidate"], selections[second]["selected_candidate"]
        available = not interrupted and a is not None and b is not None
        selected_pairs.append(dict(methods=[first, second], comparison_available=available,
                                   first_candidate=a, second_candidate=b,
                                   positions_rms_metres=float(np.sqrt(np.mean((states[(a, first)][0] - states[(b, second)][0])**2))) if available else None,
                                   velocities_rms_metres_per_second=float(np.sqrt(np.mean((states[(a, first)][1] - states[(b, second)][1])**2))) if available else None,
                                   fracture_disagreements=int(np.count_nonzero(prepared[a]["mask"] != prepared[b]["mask"])) if available else None))
    return dict(format="fracture-world.matched-step.v1", status="interrupted" if interrupted else "completed",
                case_id=initial.case_id, time_s=initial.time, dt=initial.dt, input_hash=input_hash,
                candidate_pool=[item["candidate_hash"] for item in prepared], candidate_count=len(prepared),
                proposal_generation_seconds=proposal_seconds, predictor_calls=1, fracture_flow_calls=config.fracture_samples,
                requested_method_runs=len(prepared) * len(METHODS), attempted_method_runs=len(records),
                unattempted_method_runs=len(prepared) * len(METHODS) - len(records),
                methods=selections, candidates=records, central_consensus=agreement, selected_state_comparisons=selected_pairs,
                reference_used_for_proposals=False, reference_used_for_selection=False,
                reference_used_only_for_evaluation=reference_values is not None, engine_calls=0,
                protocol="one frozen current state, one predictor output, one shared candidate pool; all methods evaluate every candidate without pruning",
                completion_semantics="completed means all requested method evaluations were attempted, not solver convergence or physical validity",
                constraint_scope="recorded endpoint geometry, conservative swept geometry, energy and supplied solver constraints; unknown is not verified",
                nonlinear_comparison="same nominal objective and inputs; different local nonlinear algorithms; no global optimum or consensus superiority certificate")


def _initial_observation(item, frame, model):
    context = item.metadata["conditioning_context"]
    current_fixed = item.fixed[frame]
    policy = "free" if not current_fixed.any() else "learned_material_corner_release" if model.config.get("learn_support_release", False) else "prescribed_material_corners"
    metadata = dict(support_policy=policy, teacher_physics=_teacher_physics(item.metadata.get("native_settings", {})),
                    native_settings=dict(item.metadata.get("native_settings", {})))
    if getattr(item, "elastic_state", None) is not None:
        from .dynamics_data import elastic_state_signature

        metadata["elastic_state"] = {name: values[frame] for name, values in item.elastic_state.items()}
        metadata["elastic_state"]["positions_hash"] = elastic_state_signature(item.positions[frame], item.broken[frame], current_fixed)
    if "floor_height_m" in context:
        metadata.update(floor_height_m=context["floor_height_m"], up_axis="y")
    return DynamicsInitial(item.topology, _frozen(item.positions[frame]), _frozen(item.velocities[frame]),
                           _frozen(item.broken[frame]), _frozen(current_fixed), float(item.times[frame]),
                           float(np.float32(context["dt"])), _frozen(item.gravity), dict(item.material),
                           item.case_id, item.condition_digest, metadata)


def compare_matched_dynamics(checkpoint, data, output, config=None, device="auto", *, split="test", frames=None, progress=None, cancel=None):
    if split not in {"train", "validation", "test", "all"}:
        raise ValueError("matched comparison split must select train, validation, test or all")
    if frames is not None and (not isinstance(frames, (list, tuple)) or not frames or
                               any(isinstance(frame, bool) or not isinstance(frame, int) or frame < 0 for frame in frames) or len(set(frames)) != len(frames)):
        raise ValueError("matched frames must be unique nonnegative recorded frame indices")
    model, _, saved, device = load_dynamics_model(checkpoint, device)
    if split in {"validation", "test"} and saved["training"]["overfit"]:
        raise ValueError("held-out matched comparison cannot use an overfit-debug checkpoint")
    if device.type == "cpu":
        torch.set_num_threads(min(torch.get_num_threads(), 4))
    items = load_dynamics_dataset(data, stream=True)
    if split != "all":
        names, conditions = set(saved["splits"][split]), set(saved["condition_splits"][split])
        items = [item for item in items if item.case_id in names and item.condition_digest in conditions]
        if len(items) != len(names):
            raise ValueError("matched dataset must contain every recorded split case and loading condition")
    if not items:
        raise ValueError("matched comparison requires at least one selected case")
    directory = new_directory(output)
    cases, interrupted = [], False
    methods = {method: dict.fromkeys(("requested_candidates", "attempted_candidates", "completed_candidates", "failed_candidates",
                                      "interrupted_candidates", "unknown_candidates", "verified_candidates", "invalid_candidates",
                                      "completed_selections", "verified_selections"), 0) for method in METHODS}
    pairwise = {pair: dict.fromkeys(("common_selected_observations", "positions_rms_metres", "velocities_rms_metres_per_second",
                                     "fracture_disagreements"), 0) for pair in combinations(METHODS, 2)}
    for item in items:
        trajectory = load_dynamics_trajectory(item.case_path) if hasattr(item, "case_path") else item
        known = set(saved["training_mesh_fingerprints"])
        if saved.get("generalization_protocol") == "shape_family":
            known.update(saved.get("evaluation_mesh_fingerprints", []))
        if trajectory.topology.fingerprint not in known:
            raise ValueError("matched checkpoint is not compatible with this material mesh")
        selected_frames = list(range(len(trajectory))) if frames is None else sorted(frames)
        if any(frame >= len(trajectory) for frame in selected_frames):
            raise ValueError("matched frame must select a recorded current state with a following observation")
        native_clock = _native_clock(trajectory.metadata["conditioning_context"])
        native_steps = np.searchsorted(native_clock, trajectory.times)
        rows = []
        for frame in selected_frames:
            if cancel is not None and cancel():
                interrupted = True
                break
            if native_steps[frame + 1] - native_steps[frame] != 1:
                rows.append(dict(frame=frame, status="unknown_sparse_transition", input_native_step=int(native_steps[frame])))
                continue
            initial = _initial_observation(trajectory, frame, model)

            def reference():
                return {key: getattr(trajectory, key)[frame + 1] for key in ("positions", "velocities", "broken", "fixed")}

            try:
                row = compare_matched_step(initial, model, config, int(native_steps[frame]), reference=reference, progress=progress, cancel=cancel)
            except KeyboardInterrupt:
                row = dict(status="interrupted", case_id=trajectory.case_id, candidate_count=None, attempted_method_runs=None)
            row.update(frame=frame, input_native_step=int(native_steps[frame]))
            if row["status"] != "interrupted":
                from .dynamics_mechanics import mechanics_diagnostics
                from .dynamics_teacher_audit import (CORRECTOR_CHECKS, NATIVE_CHECKS, _contract, _physics,
                                                     _status, _transition_checks, required_checks)

                evaluator = lambda initial, transition, contacts, settings: mechanics_diagnostics(
                    initial, transition, contacts, settings, velocity_scale=model.config["velocity_scale"])
                try:
                    checks = _transition_checks(trajectory, frame, initial.dt, model, config or runtime.RolloutConfig(),
                                                 evaluator, _physics(trajectory.metadata.get("native_settings", {})))
                    ideal_required = [name for name in CORRECTOR_CHECKS if name != "mechanical_energy" or
                                      (config or runtime.RolloutConfig()).energy_balance]
                    row["teacher_constraint_audit"] = dict(
                        status=_status(checks[name] for name in required_checks(config or runtime.RolloutConfig())),
                        checks=checks, evaluated_after_selection=True,
                        ideal_corrector_compatibility=_contract((checks[name] for name in ideal_required), ideal_required),
                        native_transition_consistency=_contract((checks[name] for name in NATIVE_CHECKS), NATIVE_CHECKS))
                except (ValueError, np.linalg.LinAlgError, FloatingPointError) as error:
                    row["teacher_constraint_audit"] = dict(status="unknown", reason=f"{type(error).__name__}: {error}",
                                                           evaluated_after_selection=True)
            rows.append(row)
            if progress:
                progress(dict(event="matched_step_completed", case=trajectory.case_id, frame=frame, status=row["status"],
                              candidate_count=row["candidate_count"], attempted_method_runs=row["attempted_method_runs"]))
            if row["status"] == "interrupted":
                interrupted = True
                break
        case_file = f"case-{len(cases):05d}.json"
        for row in rows:
            _accumulate_summary(methods, pairwise, row)
        write_json(directory / case_file, dict(case_id=trajectory.case_id, frames=rows))
        cases.append(dict(case_id=trajectory.case_id, report=case_file, requested_frames=len(selected_frames),
                          attempted_frames=len(rows), completed_frames=sum(row["status"] == "completed" for row in rows),
                          unknown_frames=sum(row["status"].startswith("unknown") for row in rows),
                          unknown_candidate_pools=sum(not isinstance(row.get("candidate_count"), int) for row in rows),
                          all_methods_common_selected_observations=sum(
                              row["status"] == "completed" and len(row.get("methods", {})) == len(METHODS)
                              and all(item["selected_candidate"] is not None for item in row["methods"].values()) for row in rows)))
        if interrupted:
            break
    for values in methods.values():
        values["unattempted_candidates"] = values["requested_candidates"] - values["attempted_candidates"]
    comparisons = []
    for pair, values in pairwise.items():
        count = values["common_selected_observations"]
        comparisons.append(dict(methods=list(pair), common_selected_observations=count,
                                **{f"mean_{key}": float(values[key] / count) if count else None
                                   for key in ("positions_rms_metres", "velocities_rms_metres_per_second", "fracture_disagreements")}))
    report = dict(format="fracture-world.matched-comparison.v1", status="interrupted" if interrupted else "completed",
                  checkpoint_sha256=file_hash(checkpoint), split=split, requested_cases=len(items), attempted_cases=len(cases),
                  unattempted_cases=len(items) - len(cases), cases=cases, methods=list(METHODS),
                  requested_observations=sum(len(item) if frames is None else len(frames) for item in items),
                  attempted_observations=sum(case["attempted_frames"] for case in cases),
                  unknown_candidate_pool_observations=sum(case["unknown_candidate_pools"] for case in cases),
                  all_methods_common_selected_observations=sum(case["all_methods_common_selected_observations"] for case in cases),
                  conditions=methods, selected_state_comparisons=comparisons,
                  candidate_count_scope="requested counts cover generated candidate pools only; sparse or unattempted observations have unknown pools",
                  completion_semantics="completed candidates returned a state; unknown validation may overlap completed or failed counts; verified counts require every recorded check",
                  protocol="offline one-step replay of observed current states; never an autonomous rollout",
                  engine_calls=0, teacher_future_used_for_proposals=False, teacher_future_used_for_selection=False,
                  teacher_future_used_only_for_evaluation=True, configuration=asdict(config or runtime.RolloutConfig()))
    write_json(directory / "matched.json", report)
    return report
