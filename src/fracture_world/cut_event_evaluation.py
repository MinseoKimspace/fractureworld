import json
from pathlib import Path
from time import perf_counter

import numpy as np
import torch

from .audit2d import REFINEMENT_LIMITS
from .benchmark2d import checkpoint_hash, synchronize
from .cut_event import CutSequence, corner_weights, experiment_plan
from .cut_event_teacher import simulate_event, simulate_segment, transfer_audit
from .force_reference2d import material_for, reference_settings
from .metrics2d import weighted_rmse
from .training2d import emit, graph_for, load_model


def sequence_metadata(scenario, label, **extra):
    return {
        "experiment_hash": scenario["experiment_hash"], "method_label": label,
        "source_case_hash": scenario["inputs"]["source_case_hash"],
        "event_time": scenario["event_time"], "status": "completed", **extra,
    }


def reference_sequences(scenario):
    started = perf_counter()
    source, time = scenario["source"], scenario["time"]
    material, load = material_for(source), source.metadata["pull"]
    coarse_settings, fine_settings = reference_settings(time)
    levels = []
    for settings in (coarse_settings, fine_settings):
        before, after, audit = simulate_event(
            source.mesh, scenario["final_mesh"], material, time, scenario["event_time"], load, settings,
        )
        if not np.array_equal(before.arrays["position"][0], source.arrays["position"][0]) or not np.array_equal(
            before.arrays["velocity"][0], source.arrays["velocity"][0]
        ):
            raise ValueError("The dynamic-cut teacher requires the source rest initial state")
        uncut = simulate_segment(
            source.mesh, material, after.arrays["time"], load,
            before.arrays["position"][-1], before.arrays["velocity"][-1], **settings,
        )
        levels.append((
            CutSequence(before, after, sequence_metadata(scenario, "teacher_cut")),
            CutSequence(before, uncut, sequence_metadata(scenario, "teacher_no_cut")), audit,
        ))
    refinement = {}
    checks = {}
    for index, condition in enumerate(("cut", "no_cut")):
        differences = {
            f"{field}_max_difference": float(np.max(np.abs(
                levels[0][index].corners(field) - levels[1][index].corners(field)
            ))) for field in ("position", "velocity")
        }
        differences["boundary_impulse_max_difference"] = max(float(np.max(np.abs(
            first.arrays["boundary_impulse"] - second.arrays["boundary_impulse"]
        ), initial=0)) for first, second in (
            (levels[0][index].before, levels[1][index].before),
            (levels[0][index].after, levels[1][index].after),
        ))
        refinement[condition] = differences
        checks[condition] = {key: value <= REFINEMENT_LIMITS[key] for key, value in differences.items()}
    return {"cut": levels[1][0], "no_cut": levels[1][1]}, {
        "converged": all(all(value.values()) for value in checks.values()),
        "checks": checks, "refinement": refinement, "limits": dict(REFINEMENT_LIMITS),
        "event_transfer": levels[1][2], "coarse_settings": coarse_settings,
        "refined_settings": fine_settings, "runtime_seconds": perf_counter() - started,
        "scope": "Independent refined FElupe/DOP853 numerical references; no experimental physical truth",
    }


def sequence_metrics(reference, prediction):
    count = min(len(reference.time), len(prediction.time))
    if count < 2 or not np.array_equal(reference.time[:count], prediction.time[:count]):
        raise ValueError("Sequence comparison requires matching physical timestamps")
    mass = corner_weights(reference.before)
    boundary = reference.before.arrays["boundary_mask"][reference.before.mesh.triangles].reshape(-1, 2)
    free_nodes = np.any(~boundary, axis=1)
    result = {"compared_frames": count, "compared_horizon": float(reference.time[count - 1]),
        "rmse_scope": "Material-corner mass-weighted Euclidean errors; initial frame excluded"}
    post = reference.time[:count] > reference.metadata["event_time"]
    for field in ("position", "velocity"):
        error = (prediction.corners(field)[:count] - reference.corners(field)[:count]).reshape(count, -1, 2)
        result[f"{field}_rmse"] = weighted_rmse(error[1:], mass)
        result[f"final_{field}_rmse"] = weighted_rmse(error[-1:], mass)
        result[f"post_event_{field}_rmse"] = weighted_rmse(error[post], mass) if post.any() else None
        free_error = np.where(boundary[None], 0, error[1:])[:, free_nodes]
        result[f"free_{field}_rmse"] = weighted_rmse(free_error, mass[free_nodes])
    invalid = 0
    for segment in (prediction.before, prediction.after):
        position = segment.arrays["position"]
        points = position[:, segment.mesh.triangles]
        first, second = points[:, :, 1] - points[:, :, 0], points[:, :, 2] - points[:, :, 0]
        inverted = first[:, :, 0] * second[:, :, 1] - first[:, :, 1] * second[:, :, 0] <= 0
        invalid += int(np.any(inverted, axis=1).sum())
    result.update(orientation_valid=invalid == 0, inverted_segment_frames=invalid,
                  sampled_geometry_scope="Positive triangle orientation at output frames; self-contact and overlap are excluded")
    return result


def causal_metrics(references, predictions):
    result = {}
    baseline = predictions.get("no_cut")
    if baseline is None:
        return result
    weights = corner_weights(references["cut"].before)
    for label in ("cut_stale", "cut_updated"):
        current = predictions.get(label)
        if current is None:
            continue
        count = min(len(baseline.time), len(current.time), len(references["cut"].time))
        post = references["cut"].time[:count] > references["cut"].metadata["event_time"]
        values = {"compared_horizon": float(references["cut"].time[count - 1])}
        for field in ("position", "velocity"):
            target = references["cut"].corners(field)[:count] - references["no_cut"].corners(field)[:count]
            actual = current.corners(field)[:count] - baseline.corners(field)[:count]
            values[f"{field}_intervention_rmse"] = weighted_rmse(
                (actual - target)[post].reshape(int(post.sum()), -1, 2), weights,
            ) if post.any() else None
            values[f"teacher_{field}_intervention_rms"] = weighted_rmse(
                target[post].reshape(int(post.sum()), -1, 2), weights,
            ) if post.any() else None
        result[label] = values
    return result


def event_force_audit(model, reference, device):
    from .cut_event_runtime import context_force

    physical, context = graph_for(reference.after, device), graph_for(reference.before, device)
    position = torch.as_tensor(reference.after.arrays["position"][0], dtype=physical.rest.dtype, device=device)
    truth = reference.after.arrays["internal_force"][0]
    free = ~reference.after.arrays["boundary_mask"]
    mass = reference.after.mass
    results = {}
    with torch.inference_mode():
        for label, encoding in (("cut_updated", None), ("cut_stale", context)):
            actual = context_force(model, physical, encoding, position).cpu().numpy().astype(np.float64)
            if not np.isfinite(actual).all():
                raise FloatingPointError("The same-state learned force is non-finite")
            error = actual - truth
            norm = float(np.linalg.norm(truth[free]))
            results[label] = {
                "free_force_rmse": float(np.sqrt(np.mean(error[free] ** 2))),
                "free_force_relative_l2": float(np.linalg.norm(error[free]) / norm) if norm > 0 else None,
                "free_acceleration_rmse": float(np.sqrt(np.mean((error / mass[:, None])[free] ** 2))),
            }
    before = reference.before
    before_acceleration = before.arrays["internal_force"][-1] / before.mass[:, None]
    after_acceleration = truth / mass[:, None]
    jump = after_acceleration[reference.after.mesh.triangles] - before_acceleration[before.mesh.triangles]
    return {"status": "completed", "states": "Same refined teacher state at the event, after physical separation",
            "purpose": "Diagnostic only; never used to initialize or correct autonomous predictions",
            "teacher_internal_acceleration_jump_rms": weighted_rmse(jump.reshape(1, -1, 2), corner_weights(before)),
            "methods": results}


def report_summary(cases):
    summary = {"cases": len(cases), "completed_cases": sum(row["status"] == "completed" for row in cases),
               "methods": {}}
    for label in ("no_cut", "cut_stale", "cut_updated"):
        records = [row["methods"][label] for row in cases if label in row["methods"]]
        completed = [row for row in records if row["status"] == "completed"]
        summary["methods"][label] = {
            "requested": len(cases), "completed": len(completed), "attempted": len(records),
            "unattempted": len(cases) - len(records),
            "orientation_valid_completed": sum(row["orientation_valid"] for row in completed),
            "runtime_seconds_total": sum(row["runtime_seconds"] for row in records),
        }
        for field in ("position", "velocity"):
            values = [row[f"post_event_{field}_rmse"] for row in completed]
            summary["methods"][label][f"mean_post_event_{field}_rmse"] = float(np.mean(values)) if values else None
    return summary


def record_case(report, result, output, started):
    result["runtime_seconds"] = perf_counter() - started
    report["cases"].append(result)
    (output / "report.json").write_text(json.dumps(report, indent=2, allow_nan=False), encoding="utf-8")
    emit("cut_event_case_completed", case=result["case"], status=result["status"],
         runtime_seconds=result["runtime_seconds"])


def evaluate_cut_event(data, checkpoint, output, source_cases=None, event_time=0.03,
                       horizon=None, device="cpu", substeps=1, render_results=False,
                       allow_untrained=False):
    from .cut_event_runtime import event_rollout

    if str(device).startswith("cuda") and not torch.cuda.is_available():
        raise ValueError("CUDA was requested but is unavailable")
    output = Path(output)
    if output.exists() and (not output.is_dir() or any(output.iterdir())):
        raise ValueError("Cut-event output must be a new or empty directory")
    if isinstance(substeps, bool) or substeps not in (1, 2, 4):
        raise ValueError("Substeps must be 1, 2, or 4")
    model, details = load_model(checkpoint, device)
    if getattr(model, "architecture", None) != "equilibrium" or model.method != "stress":
        raise ValueError("Dynamic cut evaluation requires an equilibrium stress checkpoint")
    if details.get("trained") is False and not allow_untrained:
        raise ValueError("An untrained pipeline check requires --allow-untrained")
    required = {"training_groups", "training_case_hashes", "dt"}
    if not required.issubset(details):
        raise ValueError("The checkpoint is missing training provenance or its timestep")
    plan = experiment_plan(data, source_cases, event_time, horizon)
    dt = float(details.get("dt", 0))
    if not np.isfinite(dt) or dt <= 0 or any(not np.allclose(
        np.diff(row["time"]), dt, rtol=1e-9, atol=1e-12,
    ) for row in plan):
        raise ValueError("The checkpoint and source collection timesteps must match")
    trained_groups = {str(group) for group in details.get("training_groups", [])}
    trained_hashes = set(details.get("training_case_hashes", []))
    if any(row["inputs"]["source_group"] in trained_groups or
           row["inputs"]["source_case_hash"] in trained_hashes for row in plan):
        raise ValueError("The source cut configuration overlaps checkpoint training data")
    provenance = {name: name in details for name in ("training_groups", "training_case_hashes")}
    training_status = "trained" if details.get("trained") is True else "untrained" if details.get("trained") is False else "unknown"
    digest = checkpoint_hash(checkpoint)
    output.mkdir(parents=True, exist_ok=True)
    report = {
        "format": "cut-event-evaluation-v1", "checkpoint": str(Path(checkpoint).resolve()),
        "checkpoint_sha256": digest, "device": str(device), "substeps": substeps,
        "model_config": details["model_config"], "training_provenance": provenance,
        "training_executed": False, "training_status": training_status,
        "purpose": {"trained": "zero_shot_dynamic_cut_transfer", "untrained": "untrained_pipeline_check",
                    "unknown": "training_status_unknown"}[training_status],
        "scope": "Prescribed single cut finishing a pre-existing notch during tension; no contact or learned fracture criterion",
        "topology_ablation": "Frozen pre-cut encoder context versus updated encoder context; both cut variants use the actual separated physical mesh",
        "interpretation": "Homogeneous local StVK stress need not depend on graph context at identical strain; updated-context superiority is not assumed",
        "cases": [],
    }
    for scenario in plan:
        started = perf_counter()
        emit("cut_event_reference_started", case=scenario["id"], event_time=scenario["event_time"])
        result = {"case": scenario["id"], "inputs": scenario["inputs"],
                  "experiment_hash": scenario["experiment_hash"], "methods": {}}
        try:
            references, audit = reference_sequences(scenario)
        except (RuntimeError, ValueError, FloatingPointError, OverflowError) as error:
            result.update(status="reference_failed", reference={"status": "failed",
                          "failure": f"{type(error).__name__}: {error}", "converged": False})
            record_case(report, result, output, started)
            continue
        result["reference"] = audit
        destination = output / scenario["id"]
        destination.mkdir()
        for name, sequence in references.items():
            sequence.save(destination / f"teacher_{name}.npz")
        if audit["converged"]:
            try:
                result["same_state_event_force"] = event_force_audit(model, references["cut"], device)
            except (RuntimeError, ValueError, FloatingPointError, OverflowError) as error:
                result["same_state_event_force"] = {"status": "failed", "methods": {},
                                                    "failure": f"{type(error).__name__}: {error}"}
            emit("cut_event_rollout_started", case=scenario["id"], substeps=substeps)
            synchronize(device)
            runtime = event_rollout(model, scenario["source"], scenario["final_mesh"],
                                    scenario["time"], scenario["event_time"], device, substeps)
            predictions = {}
            result["shared_prefix"] = runtime["before_status"]
            result["shared_prefix_runtime_seconds"] = runtime["shared_prefix_runtime_seconds"]
            result["neural_total_runtime_seconds"] = runtime["runtime_seconds"]
            if runtime["before_status"]["status"] == "completed":
                result["neural_event_transfer"] = transfer_audit(
                    scenario["source"].mesh, scenario["final_mesh"], material_for(scenario["source"]),
                    runtime["before"].arrays["position"][-1], runtime["before"].arrays["velocity"][-1],
                )
            for label, after in runtime["afters"].items():
                status = dict(runtime["statuses"][label])
                status["prefix_and_post_runtime_seconds"] = runtime["shared_prefix_runtime_seconds"] + status["runtime_seconds"]
                if runtime["before"] is not None and after is not None:
                    sequence = CutSequence(runtime["before"], after, sequence_metadata(
                        scenario, label, status=status["status"], checkpoint_sha256=digest,
                        training_status=training_status,
                    ))
                    predictions[label] = sequence
                    target = references["no_cut" if label == "no_cut" else "cut"]
                    metrics = sequence_metrics(target, sequence)
                    status["sampled_orientation_valid"] = metrics.pop("orientation_valid")
                    status.update(metrics)
                    status["reference_condition"] = "no_cut" if label == "no_cut" else "cut"
                    path = destination / f"{label}.npz"
                    sequence.save(path)
                    status["trajectory"] = str(path.relative_to(output))
                result["methods"][label] = status
            result["causal_response"] = causal_metrics(references, predictions)
            if render_results and len(predictions) == 3:
                from .render2d import render

                paths = [destination / "teacher_cut.npz", destination / "cut_updated.npz",
                         destination / "cut_stale.npz", destination / "teacher_no_cut.npz", destination / "no_cut.npz"]
                render(paths, destination / "comparison.gif", fps=30)
                result["render"] = str((destination / "comparison.gif").relative_to(output))
            result["status"] = "completed" if all(
                value["status"] == "completed" for value in result["methods"].values()
            ) and len(result["methods"]) == 3 else "failed"
            if result.get("neural_event_transfer", {}).get("status") == "failed":
                result["status"] = "event_transfer_failed"
        else:
            result["status"] = "reference_unconverged"
        record_case(report, result, output, started)
    report["checkpoint_unchanged"] = checkpoint_hash(checkpoint) == digest
    report["status"] = "completed" if all(row["status"] == "completed" for row in report["cases"]) else "incomplete"
    report["summary"] = report_summary(report["cases"])
    (output / "report.json").write_text(json.dumps(report, indent=2, allow_nan=False), encoding="utf-8")
    return report
