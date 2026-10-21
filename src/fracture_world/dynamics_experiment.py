from pathlib import Path

import numpy as np

from .dynamics_evaluation import evaluate_dynamics
from .dynamics_data import load_dynamics_dataset
from .dynamics_initial import export_initial
from .dynamics_learning import load_dynamics_model
from .dynamics_runtime import RolloutConfig, rollout_dynamics
from .io import file_hash, new_directory, write_json


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
                    same_objective=True,
                    interpretation="shared future predictions only; incomplete rollouts do not validate the full horizon; agreement is not evidence of proximal superiority"
                    if available else "no shared future prediction; initial-state equality provides no implementation validation")


def compare_dynamics(checkpoint, initial, output, steps, config=None, device="auto", methods=None,
                     reference_case=None, reference_trajectory=None, progress=None):
    _, learned_corrector, _, _ = load_dynamics_model(checkpoint, device)
    methods = methods or (["raw", "neural", "central", "consensus"] if learned_corrector is not None
                          else ["raw", "central", "consensus"])
    if len(set(methods)) != len(methods) or not set(methods).issubset({"physics", "raw", "neural", "central", "consensus"}):
        raise ValueError("comparison methods must be unique physics/raw/neural/central/consensus values")
    if "neural" in methods and learned_corrector is None:
        raise ValueError("neural baseline requested but checkpoint has no trained corrector")
    if reference_trajectory and not reference_case:
        raise ValueError("reference_trajectory requires reference_case")
    config = config or RolloutConfig()
    directory = new_directory(output)
    conditions = {}
    for method in methods:
        report = rollout_dynamics(checkpoint, initial, directory / method, steps, method, config, device, progress)
        conditions[method] = {key: value for key, value in report.items() if key != "events"}
    if reference_case:
        for method in methods:
            conditions[method]["evaluation"] = evaluate_dynamics(directory / method, reference_case, reference_trajectory,
                                                                 directory / method / "evaluation.json")
    agreement = None
    if {"central", "consensus"}.issubset(conditions):
        agreement = _rollout_agreement(directory / "central" / "trajectory.npz", directory / "consensus" / "trajectory.npz",
                                       all(conditions[method]["status"] == "completed" for method in ("central", "consensus")))
    report = dict(format="fracture-world.neural-comparison.v1", status="comparison_finished", engine_calls=0,
                  conditions=conditions, central_consensus_agreement=agreement,
                  proposal_control="same predictor, seed and Gaussian samples; autonomous states may diverge after the first step",
                  numerical_reference_used_for_inference=False,
                  neural_baseline_available=learned_corrector is not None,
                  all_rollouts_completed=all(item["status"] == "completed" for item in conditions.values()))
    write_json(directory / "comparison.json", report)
    return report


def benchmark_dynamics(checkpoint, data, output, steps, config=None, device="auto", methods=None, split="test", progress=None):
    if split not in {"validation", "test"}:
        raise ValueError("benchmark split must select validation or test")
    _, _, saved, _ = load_dynamics_model(checkpoint, device)
    if saved["training"]["overfit"]:
        raise ValueError("held-out benchmarking cannot use an overfit-debug checkpoint")
    conditions = set(saved["condition_splits"][split])
    names = set(saved["splits"][split])
    cases = [item for item in load_dynamics_dataset(data, stream=True)
             if item.case_id in names and item.condition_digest in conditions]
    if len(cases) != len(names) or not cases:
        raise ValueError("benchmark data must contain every recorded held-out case and matching loading condition")
    directory = new_directory(output)
    records, rows = [], {}
    for index, item in enumerate(cases, start=1):
        if steps > len(item):
            raise ValueError("benchmark horizon exceeds the recorded teacher trajectory")
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
        target_has_new_fracture = bool(np.any(item.broken[1:steps + 1] & ~item.broken[0]))
        records.append(dict(case=item.case_id, shape_family=item.shape_family, directory=case_directory.name,
                            support_policy=policy, all_rollouts_completed=comparison["all_rollouts_completed"]))
        for method, result in comparison["conditions"].items():
            result["requested_horizon_target_has_new_fracture"] = target_has_new_fracture
            result["full_horizon_evaluated"] = result["status"] == "completed" and result["evaluation"]["compared_future_frames"] == steps
            rows.setdefault(method, []).append(result)
    summaries = {}
    for method, results in rows.items():
        evaluations = [result["evaluation"] for result in results if result["full_horizon_evaluated"]]
        positive = [item["evaluation"]["new_fracture"]["f1"] if item["full_horizon_evaluated"] else 0.0
                    for item in results if item["requested_horizon_target_has_new_fracture"]]
        negative = [item["evaluation"]["new_fracture"]["predicted_broken_facets"] > 0
                    for item in results if not item["requested_horizon_target_has_new_fracture"] and item["full_horizon_evaluated"]]
        mean = lambda values: float(np.mean(values)) if values else None
        metrics = ("position_mass_weighted_rmse_m", "velocity_mass_weighted_rmse_m_per_s",
                   "linear_momentum_error_kg_m_per_s", "angular_momentum_about_origin_error_kg_m2_per_s",
                   "kinetic_energy_absolute_error_j", "elastic_energy_absolute_error_j", "gravitational_energy_absolute_error_j",
                   "total_mechanical_energy_absolute_error_j", "maximum_floor_penetration_m", "maximum_reference_floor_penetration_m",
                   "component_count_absolute_error",
                   *[f"mean_{prefix}{name}_energy_j" for prefix in ("", "reference_")
                     for name in ("kinetic", "elastic", "gravitational", "total")],
                   *[f"maximum_{prefix}{name}_mechanical_energy_increment{unit}" for prefix in ("", "reference_")
                     for name, unit in (("positive", "_j"), ("relative_positive", ""))])
        summaries[method] = dict(cases=len(results), completed_rate=mean([item["status"] == "completed" for item in results]),
                                  completed_and_geometry_valid_rate=mean([item["status"] == "completed" and item.get("all_committed_geometry_valid") is True for item in results]),
                                  **_checked_constraint_summary(results),
                                  fracture_positive_cases=len(positive), fracture_positive_case_f1_failure_penalized=mean(positive),
                                  completed_fracture_negative_cases=len(negative), fracture_negative_case_false_positive_rate=mean(negative),
                                  full_horizon_accuracy_cases=len(evaluations),
                                  runtime_seconds_total=float(sum(item["runtime_seconds"] for item in results)),
                                  held_out_shape_family_cases=sum(item.get("held_out_shape_family", False) for item in results),
                                  metric_availability_cases={key: sum(item.get(key) is not None for item in evaluations) for key in metrics},
                                  **{key: mean([item[key] for item in evaluations if item.get(key) is not None]) for key in metrics})
    report = dict(schema="fracture-world.dynamics-benchmark.v1", cases=records, conditions=summaries,
                  split=split, generalization_protocol=saved.get("generalization_protocol", "loading_condition"),
                  inference_engine_calls=0, reference_used_only_for_evaluation=True,
                  positive_and_negative_fracture_scores_reported_separately=True,
                  fracture_score_semantics="new broken-facet states over future frames only; initially broken facets excluded",
                  support_policy_selection="checkpoint support-release capability and current fixed corners only; teacher support future not inspected",
                  checked_constraint_success_semantics="completion plus verified endpoint geometry, swept geometry and mechanical energy; missing diagnostics remain unknown and do not count as verified success",
                  incomplete_rollout_policy="positive-case F1 is zero; error means and negative-case false positives use complete horizons; completion rates include every case",
                  checkpoint_sha256=file_hash(checkpoint),
                  all_rollouts_completed=all(item["all_rollouts_completed"] for item in records))
    write_json(directory / "benchmark.json", report)
    return report
