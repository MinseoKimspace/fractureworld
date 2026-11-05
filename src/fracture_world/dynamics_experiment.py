from itertools import combinations
from pathlib import Path

import numpy as np

from .dynamics_evaluation import _native_clock, evaluate_dynamics
from .dynamics_data import load_dynamics_dataset
from .dynamics_initial import export_initial
from .dynamics_learning import load_dynamics_model
from .dynamics_runtime import RolloutConfig, rollout_dynamics
from .io import file_hash, new_directory, write_json


_ACCURACY_METRICS = (
    "position_mass_weighted_rmse_m", "velocity_mass_weighted_rmse_m_per_s",
    "linear_momentum_error_kg_m_per_s", "angular_momentum_about_origin_error_kg_m2_per_s",
    "kinetic_energy_absolute_error_j", "elastic_energy_absolute_error_j", "gravitational_energy_absolute_error_j",
    "total_mechanical_energy_absolute_error_j", "maximum_floor_penetration_m", "maximum_reference_floor_penetration_m",
    "component_count_absolute_error",
    *[f"mean_{prefix}{name}_energy_j" for prefix in ("", "reference_")
      for name in ("kinetic", "elastic", "gravitational", "total")],
    *[f"maximum_{prefix}{name}_mechanical_energy_increment{unit}" for prefix in ("", "reference_")
      for name, unit in (("positive", "_j"), ("relative_positive", ""))],
)

_AUTONOMOUS_PROPOSAL_CONTROL = (
    "same checkpoint, initial state, configuration, seed and Gaussian noise schedule; each method feeds back "
    "its own predictions; fracture and support candidates are regenerated from that method's current state "
    "and need not match after trajectories diverge"
)


def _motion_ablation_protocol(methods):
    pairs = [[solver, f"no_motion_{solver}"] for solver in ("central", "central_pcg", "consensus")]
    return dict(enabled=any(method.startswith("no_motion_") for method in methods),
                comparison_pairs=[pair for pair in pairs if set(pair).issubset(methods)],
                removed="learned motion residual only",
                retained="learned encoder, fracture flow, fracture probabilities and support-release predictions",
                fully_neural_free=False,
                state_feedback="each method uses its own committed state; no teacher forcing or cross-method state sharing",
                candidate_control="same learned generator and noise schedule; identical candidate pools are not enforced after divergence")


def _paired_accuracy(rows, methods, metrics):
    methods = tuple(methods)
    indexed = {method: {item["benchmark_case_id"]: item for item in rows[method]} for method in methods}
    all_cases = set().union(*(set(items) for items in indexed.values()))
    completed = [{case for case, item in items.items() if item["full_horizon_evaluated"]}
                 for items in indexed.values()]
    common = sorted(set.intersection(*completed)) if completed else []
    summaries = {}
    for metric in metrics:
        available = [case for case in common if all(
            item[case]["evaluation"].get(metric) is not None
            and np.isfinite(item[case]["evaluation"][metric]) for item in indexed.values())]
        values = {method: [items[case]["evaluation"][metric] for case in available]
                  for method, items in indexed.items()}
        summary = dict(cases=len(available), case_ids=available,
                       comparison_available=len(methods) > 1 and bool(available),
                       means={method: float(np.mean(data)) if data else None for method, data in values.items()})
        if len(methods) == 2:
            first, second = methods
            differences = np.asarray(values[second]) - np.asarray(values[first])
            summary["mean_difference_second_minus_first"] = float(np.mean(differences)) if len(differences) else None
        summaries[metric] = summary
    return dict(methods=list(methods), total_cases=len(all_cases), common_full_horizon_cases=len(common),
                common_full_horizon_case_ids=common, excluded_case_ids=sorted(all_cases.difference(common)),
                comparison_available=len(methods) > 1 and bool(common), metrics=summaries)


def _paired_solver_metrics(rows, methods):
    methods = tuple(methods)
    metrics = ("runtime_seconds", "corrector_iterations", "optimization_solves")
    measurements = {method: [{**item, "evaluation": {metric: item.get(metric) for metric in metrics}}
                             for item in rows[method]] for method in methods}
    return dict(all_methods=_paired_accuracy(measurements, methods, metrics),
                pairwise=[_paired_accuracy(measurements, pair, metrics) for pair in combinations(methods, 2)],
                interpretation="runtime and solver work conditional on shared completed horizons; runtime includes prediction and shared preparation per autonomous rollout",
                iteration_interpretation="iteration counts are backend-specific: central uses interior-point iterations, central_pcg counts active-set passes plus PCG iterations, consensus counts ADMM iterations; counts are not equivalent units of work",
                difference_semantics="second method minus first method on each shared case")


def _runtime_support_policy(checkpoint, current_fixed):
    if not np.asarray(current_fixed, dtype=bool).any():
        return "free"
    learned = checkpoint.get("model_config", {}).get(
        "learn_support_release", checkpoint.get("training", {}).get("learn_support_release", False))
    return "learned" if learned else "prescribed"


def _checked_constraint_summary(results):
    keys = ("all_committed_geometry_valid", "all_committed_energy_valid", "all_committed_swept_geometry_valid")
    known = [item for item in results if all(isinstance(item.get(key), (bool, np.bool_)) for key in keys)]
    verified = [item["status"] == "completed" and all(
        isinstance(item.get(key), (bool, np.bool_)) and bool(item[key]) for key in keys) for item in results]
    return dict(fully_checked_constraint_cases=len(known), constraint_status_unknown_cases=len(results) - len(known),
                completed_and_checked_constraints_valid_rate=float(np.mean(verified)) if verified else None)


def _rollout_agreement(central, consensus, completed):
    with np.load(central, allow_pickle=False) as a, np.load(consensus, allow_pickle=False) as b:
        count = min(len(a["times"]), len(b["times"]))
        future = slice(1, count)
        available = count > 1
        return dict(common_frames=count, common_prediction_frames=max(count - 1, 0), agreement_available=available,
                    both_rollouts_completed=bool(completed),
                    positions_rms_metres=float(np.sqrt(np.mean((a["positions"][future] - b["positions"][future])**2))) if available else None,
                    velocities_rms_metres_per_second=float(np.sqrt(np.mean((a["velocities"][future] - b["velocities"][future])**2))) if available else None,
                    fracture_disagreements=int(np.count_nonzero(a["broken"][future] != b["broken"][future])) if available else None,
                    same_objective=None, same_objective_formulation=True,
                    objective_scope="same formulation within a motion-source pair; autonomous current states and candidate pools may differ, so identical per-step objectives are not enforced",
                    interpretation="shared future predictions only; incomplete rollouts do not validate the full horizon; agreement is not evidence of proximal superiority"
                    if available else "no shared future prediction; initial-state equality provides no implementation validation")


def _comparison_methods(methods, learned_corrector):
    selected = list(methods or (["raw", "neural", "central", "consensus"] if learned_corrector is not None
                                else ["raw", "central", "consensus"]))
    allowed = {"physics", "raw", "neural", "central", "central_pcg", "consensus",
               "no_motion_central", "no_motion_central_pcg", "no_motion_consensus"}
    if len(set(selected)) != len(selected) or not set(selected).issubset(allowed):
        raise ValueError("comparison methods must be unique physics/raw/neural/central/central_pcg/consensus/no_motion_central/no_motion_central_pcg/no_motion_consensus values")
    if "neural" in selected and learned_corrector is None:
        raise ValueError("neural baseline requested but checkpoint has no trained corrector")
    return selected


def compare_dynamics(checkpoint, initial, output, steps, config=None, device="auto", methods=None,
                     reference_case=None, reference_trajectory=None, progress=None):
    _, learned_corrector, _, _ = load_dynamics_model(checkpoint, device)
    methods = _comparison_methods(methods, learned_corrector)
    if reference_trajectory and not reference_case:
        raise ValueError("reference_trajectory requires reference_case")
    config = config or RolloutConfig()
    directory = new_directory(output)
    conditions = {}
    for method in methods:
        report = rollout_dynamics(checkpoint, initial, directory / method, steps, method, config, device, progress)
        conditions[method] = {key: value for key, value in report.items() if key != "events"}
        if report["status"] == "interrupted":
            break
    interrupted = any(item["status"] == "interrupted" for item in conditions.values())
    if reference_case:
        for method in conditions:
            conditions[method]["evaluation"] = evaluate_dynamics(directory / method, reference_case, reference_trajectory,
                                                                 directory / method / "evaluation.json")
    agreement = None
    if {"central", "consensus"}.issubset(conditions):
        agreement = _rollout_agreement(directory / "central" / "trajectory.npz", directory / "consensus" / "trajectory.npz",
                                       all(conditions[method]["status"] == "completed" for method in ("central", "consensus")))
    no_motion_agreement = None
    if {"no_motion_central", "no_motion_consensus"}.issubset(conditions):
        no_motion_agreement = _rollout_agreement(
            directory / "no_motion_central" / "trajectory.npz", directory / "no_motion_consensus" / "trajectory.npz",
            all(conditions[method]["status"] == "completed" for method in ("no_motion_central", "no_motion_consensus")))
    pcg_agreements = {}
    for prefix in ("", "no_motion_"):
        first, second = f"{prefix}central", f"{prefix}central_pcg"
        pcg_agreements[f"{prefix}central_central_pcg_agreement"] = _rollout_agreement(
            directory / first / "trajectory.npz", directory / second / "trajectory.npz",
            all(conditions[method]["status"] == "completed" for method in (first, second))) if {first, second}.issubset(conditions) else None
    report = dict(format="fracture-world.neural-comparison.v1", status="interrupted" if interrupted else "comparison_finished", engine_calls=0,
                  conditions=conditions, central_consensus_agreement=agreement,
                  no_motion_central_consensus_agreement=no_motion_agreement,
                  **pcg_agreements,
                  motion_ablation=_motion_ablation_protocol(methods),
                  requested_methods=methods, attempted_methods=list(conditions),
                  unattempted_methods=[method for method in methods if method not in conditions],
                  proposal_control=_AUTONOMOUS_PROPOSAL_CONTROL,
                  protocol="autonomous rollout from one initial state; every committed prediction is the next step's input",
                  numerical_reference_used_for_inference=False,
                  neural_baseline_available=learned_corrector is not None,
                  all_rollouts_completed=not interrupted and len(conditions) == len(methods)
                  and all(item["status"] == "completed" for item in conditions.values()))
    write_json(directory / "comparison.json", report)
    return report


def benchmark_dynamics(checkpoint, data, output, steps, config=None, device="auto", methods=None, split="test", progress=None):
    if isinstance(steps, bool) or not isinstance(steps, int) or steps < 1:
        raise ValueError("benchmark steps must be a positive integer number of native timesteps")
    if split not in {"validation", "test"}:
        raise ValueError("benchmark split must select validation or test")
    _, learned_corrector, saved, _ = load_dynamics_model(checkpoint, device)
    methods = _comparison_methods(methods, learned_corrector)
    if saved["training"]["overfit"]:
        raise ValueError("held-out benchmarking cannot use an overfit-debug checkpoint")
    conditions = set(saved["condition_splits"][split])
    names = set(saved["splits"][split])
    cases = [item for item in load_dynamics_dataset(data, stream=True)
             if item.case_id in names and item.condition_digest in conditions]
    if len(cases) != len(names) or not cases:
        raise ValueError("benchmark data must contain every recorded held-out case and matching loading condition")
    directory = new_directory(output)
    records, rows = [], {method: [] for method in methods}
    interrupted = False
    for index, item in enumerate(cases, start=1):
        clock = _native_clock(item.metadata["conditioning_context"])
        if steps >= len(clock):
            raise ValueError("benchmark horizon exceeds the recorded teacher trajectory")
        recorded_steps = np.searchsorted(clock, item.times)
        if steps not in recorded_steps:
            raise ValueError("benchmark horizon must end at a recorded numerical teacher step")
        if progress:
            progress(dict(event="benchmark_case", case=item.case_id, index=index, cases=len(cases), split=split))
        case_path = next(Path(path) for path in item.metadata["source_hashes"] if Path(path).parent.name == "cases")
        trajectory_path = case_path.parent.parent / "trajectories" / case_path.name
        case_directory = directory / f"case-{index:05d}"
        case_directory.mkdir()
        initial = case_directory / "initial.npz"
        policy = _runtime_support_policy(saved, item.fixed[0])
        export_initial(case_path, trajectory_path, 0, initial, support_policy=policy)
        comparison = compare_dynamics(checkpoint, initial, case_directory / "comparison", steps, config, device, methods,
                                      case_path, trajectory_path, progress)
        target_has_new_fracture = bool(np.any(item.broken[(recorded_steps > 0) & (recorded_steps <= steps)] & ~item.broken[0]))
        records.append(dict(case=item.case_id, shape_family=item.shape_family, directory=case_directory.name,
                            support_policy=policy, status=comparison.get("status", "comparison_finished"),
                            attempted_methods=list(comparison["conditions"]),
                            unattempted_methods=[method for method in methods if method not in comparison["conditions"]],
                            all_rollouts_completed=comparison["all_rollouts_completed"]))
        for method, result in comparison["conditions"].items():
            result["benchmark_case_id"] = item.case_id
            result["requested_horizon_target_has_new_fracture"] = target_has_new_fracture
            result["full_horizon_evaluated"] = result["status"] == "completed" and result["evaluation"]["last_compared_native_step"] == steps
            rows.setdefault(method, []).append(result)
        if comparison.get("status") == "interrupted" or any(result["status"] == "interrupted" for result in comparison["conditions"].values()):
            interrupted = True
            break
    summaries = {}
    for method, results in rows.items():
        evaluations = [result["evaluation"] for result in results if result["full_horizon_evaluated"]]
        positive = [item["evaluation"]["new_fracture"]["f1"] if item["full_horizon_evaluated"] else 0.0
                    for item in results if item["requested_horizon_target_has_new_fracture"]]
        negative = [item["evaluation"]["new_fracture"]["predicted_broken_facets"] > 0
                    for item in results if not item["requested_horizon_target_has_new_fracture"] and item["full_horizon_evaluated"]]
        mean = lambda values: float(np.mean(values)) if values else None
        metrics = _ACCURACY_METRICS
        summaries[method] = dict(cases=len(results), requested_cases=len(cases), attempted_cases=len(results),
                                  unattempted_cases=len(cases) - len(results),
                                  interrupted_cases=sum(item["status"] == "interrupted" for item in results),
                                  failed_cases=sum(item["status"] not in {"completed", "interrupted"} for item in results),
                                  completed_rate=mean([item["status"] == "completed" for item in results]),
                                  completed_and_geometry_valid_rate=mean([item["status"] == "completed" and item.get("all_committed_geometry_valid") is True for item in results]),
                                  **_checked_constraint_summary(results),
                                  fracture_positive_cases=len(positive), fracture_positive_case_f1_failure_penalized=mean(positive),
                                  completed_fracture_negative_cases=len(negative), fracture_negative_case_false_positive_rate=mean(negative),
                                  full_horizon_accuracy_cases=len(evaluations),
                                  runtime_seconds_total=float(sum(item["runtime_seconds"] for item in results)),
                                  held_out_shape_family_cases=sum(item.get("held_out_shape_family", False) for item in results),
                                  metric_availability_cases={key: sum(item.get(key) is not None for item in evaluations) for key in metrics},
                                  **{key: mean([item[key] for item in evaluations if item.get(key) is not None]) for key in metrics})
    paired = dict(all_methods=_paired_accuracy(rows, rows, _ACCURACY_METRICS),
                  pairwise=[_paired_accuracy(rows, pair, _ACCURACY_METRICS) for pair in combinations(rows, 2)],
                  interpretation="accuracy conditional on shared completed horizons; every metric uses the same cases across its methods; completion rates and failed cases must also be reported",
                  difference_semantics="second method minus first method on each shared case; negative differences are lower values, not an automatic claim of better physics")
    attempted_runs = sum(len(results) for results in rows.values())
    report = dict(schema="fracture-world.dynamics-benchmark.v1", status="interrupted" if interrupted else "benchmark_finished",
                  cases=records, conditions=summaries,
                  requested_cases=len(cases), attempted_cases=len(records), unattempted_cases=len(cases) - len(records),
                  requested_case_ids=[item.case_id for item in cases],
                  unattempted_case_ids=[item.case_id for item in cases[len(records):]],
                  requested_methods=methods, requested_method_runs=len(cases) * len(methods),
                  attempted_method_runs=attempted_runs, unattempted_method_runs=len(cases) * len(methods) - attempted_runs,
                  paired_accuracy=paired,
                  paired_solver_metrics=_paired_solver_metrics(rows, methods),
                  motion_ablation=_motion_ablation_protocol(methods),
                  proposal_control=_AUTONOMOUS_PROPOSAL_CONTROL,
                  protocol="autonomous rollout from each held-out initial state; teacher future is evaluated only after inference",
                  split=split, generalization_protocol=saved.get("generalization_protocol", "loading_condition"),
                  inference_engine_calls=0, reference_used_only_for_evaluation=True,
                  positive_and_negative_fracture_scores_reported_separately=True,
                  horizon_semantics="steps counts native base timesteps; accuracy uses recorded teacher states through the requested endpoint",
                  fracture_score_semantics="new broken-facet states over future frames only; initially broken facets excluded",
                  support_policy_selection="checkpoint support-release capability and current fixed corners only; teacher support future not inspected",
                  checked_constraint_success_semantics="completion plus verified endpoint geometry, swept geometry and mechanical energy; missing diagnostics remain unknown and do not count as verified success",
                  incomplete_rollout_policy="positive-case F1 is zero; error means and negative-case false positives use complete horizons; completion rates include every case",
                  interruption_policy="stop remaining methods and cases; summaries use attempted runs only; unattempted runs are neither successes nor failures",
                  checkpoint_sha256=file_hash(checkpoint),
                  all_rollouts_completed=not interrupted and len(records) == len(cases)
                  and all(item["all_rollouts_completed"] for item in records))
    write_json(directory / "benchmark.json", report)
    return report
