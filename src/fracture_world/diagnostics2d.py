import hashlib
import json
import re
from pathlib import Path
from time import perf_counter

import numpy as np
import torch

from .benchmark2d import check_checkpoint, checkpoint_hash
from .cases import Case, collection
from .diagnostic_metrics import free_motion_metrics, momentum_balance, zero_internal_force
from .diagnostic_probes import rest_probe, teacher_forced
from .training2d import emit, load_model


def save_diagnostics(output, report):
    temporary = output / "diagnostics.tmp"
    temporary.write_text(json.dumps(report, indent=2, allow_nan=False), encoding="utf-8")
    temporary.replace(output / "diagnostics.json")


def geometry_key(case):
    digest = hashlib.sha256(json.dumps(case.metadata["material"], sort_keys=True).encode())
    for name in ("rest", "triangles", "original_vertex", "cut_edges", "boundary_mask", "time"):
        array = np.ascontiguousarray(case.arrays[name])
        digest.update(name.encode())
        digest.update(str(array.shape).encode())
        digest.update(array.dtype.str.encode())
        digest.update(array.tobytes())
    return digest.hexdigest()


def saved_prediction(reference, result, report_directory, method=None, identity=None):
    relative = result.get("trajectory")
    if not relative:
        if result.get("status") == "completed" or result.get("completed_steps", 0) >= 1:
            raise ValueError("Rollout with saved finite steps is missing its trajectory record")
        return None
    relative = Path(str(relative).replace("\\", "/"))
    if relative.is_absolute() or ".." in relative.parts:
        raise ValueError("Saved trajectory must remain inside the benchmark output")
    path = (report_directory / relative).resolve()
    if not path.is_relative_to(report_directory.resolve()):
        raise ValueError("Saved trajectory resolves outside the benchmark output")
    if not path.is_file():
        raise ValueError(f"Saved trajectory not found: {path}")
    saved = Case.load(path)
    if method is not None and saved.metadata.get("method_label") != method:
        raise ValueError("Saved trajectory method differs from the benchmark label")
    for name, value in (identity or {}).items():
        if saved.metadata.get(name) != value:
            raise ValueError(f"Saved trajectory {name} differs from the benchmark identity")
    if saved.metadata.get("status") != result["status"]:
        raise ValueError("Saved trajectory status differs from the benchmark record")
    expected_hash = str(reference.arrays["case_hash"].item())
    if saved.metadata.get("source_case_hash") != expected_hash:
        raise ValueError("Saved trajectory source hash differs from the reference")
    if saved.metadata["material"] != reference.metadata["material"]:
        raise ValueError("Saved trajectory material differs from the reference")
    for name in ("rest", "triangles", "original_vertex", "cut_edges", "boundary_mask"):
        if not np.array_equal(saved.arrays[name], reference.arrays[name]):
            raise ValueError(f"Saved trajectory {name} differs from the reference")
    frames = len(saved.arrays["time"])
    if frames > len(reference.arrays["time"]):
        raise ValueError("Saved trajectory exceeds the reference horizon")
    for name in ("time", "boundary_position", "boundary_velocity"):
        if not np.array_equal(saved.arrays[name], reference.arrays[name][:frames]):
            raise ValueError(f"Saved trajectory {name} differs from prescribed inputs")
    for name in ("position", "velocity"):
        if not np.array_equal(saved.arrays[name][0], reference.arrays[name][0]):
            raise ValueError("Saved trajectory initial state differs from the reference")
    if int(result["completed_steps"]) != frames - 1:
        raise ValueError("Saved trajectory length differs from its benchmark record")
    if result["status"] == "completed" and frames != len(reference.arrays["time"]):
        raise ValueError("Completed rollout does not cover the reference horizon")
    return saved.arrays


def improvement(error, baseline):
    if error is None or baseline is None:
        return {"fraction": None, "status": "unavailable"}
    if baseline == 0:
        return {"fraction": None, "status": "zero_baseline_error"}
    return {"fraction": float(1 - error / baseline), "status": "available"}


def mean_values(records, key):
    values = [record[key] for record in records if record.get(key) is not None]
    return {"mean": float(np.mean(values)) if values else None, "count": len(values)}


def summarize(results):
    autonomous = [row["autonomous"]["free_motion"] for row in results
                  if row["saved_rollout_status"] == "completed" and row["autonomous"]["status"] == "available"]
    local = [row["teacher_forced"]["free_motion"] for row in results
             if row["teacher_forced"]["status"] == "completed"]
    keys = ("position_rmse", "velocity_rmse", "position_relative_rmse", "velocity_relative_rmse")
    return {
        "attempted_cases": len(results),
        "saved_rollouts_completed": sum(row["saved_rollout_status"] == "completed" for row in results),
        "saved_rollouts_failed": sum(row["saved_rollout_status"] != "completed" for row in results),
        "saved_trajectories_available": sum(row["autonomous"]["status"] == "available" for row in results),
        "teacher_forced_completed": len(local),
        "teacher_forced_failed": len(results) - len(local),
        "autonomous_completed_means": {key: mean_values(autonomous, key) for key in keys},
        "teacher_forced_completed_means": {key: mean_values(local, key) for key in keys},
        "mean_scope": "Complete trajectories only; failed prefixes remain in per-case diagnostics and denominators",
    }


def summarize_rest(probes):
    values = list(probes.values())
    names = ("max_free_position_drift", "max_free_speed", "initial_free_acceleration_max_norm",
             "initial_free_force_max_norm", "initial_net_internal_force_norm", "initial_net_internal_torque")
    return {
        "unique_geometries": len(values),
        "completed": sum(probe["status"] == "completed" for probe in values),
        "failed": sum(probe["status"] != "completed" for probe in values),
        "maxima": {
            name: max((abs(probe[name]) for probe in values if probe.get(name) is not None), default=None)
            for name in names
        },
        "scope": "Largest measured values across unique stationary counterfactual probes; no acceptance threshold applied",
    }


def diagnose(report, data, output, device="cpu", case_limit=None):
    if case_limit is not None and case_limit < 1:
        raise ValueError("Case limit must be positive")
    if str(device).startswith("cuda") and not torch.cuda.is_available():
        raise ValueError("CUDA was requested but is unavailable")
    report_path = Path(report).resolve()
    source = json.loads(report_path.read_text(encoding="utf-8"))
    if source.get("format") != "cut-dynamics-a-v1" or not source.get("methods"):
        raise ValueError("Expected a stage A benchmark report with methods")
    split = source.get("split")
    if split not in {"train", "validation", "test"}:
        raise ValueError("Benchmark report has an invalid dataset split")
    output = Path(output)
    if output.exists() and any(output.iterdir()):
        raise ValueError("Diagnostic output must be a new or empty directory")
    source_methods = source["methods"]
    indexed = {}
    common_ids = None
    for method, value in source_methods.items():
        records = value.get("cases", [])
        mapping = {row["case"]: row for row in records}
        if len(mapping) != len(records) or not mapping:
            raise ValueError("Benchmark cases must be nonempty with unique IDs")
        if common_ids is None:
            common_ids = set(mapping)
        elif set(mapping) != common_ids:
            raise ValueError("Benchmark methods contain different case sets")
        indexed[method] = mapping
    available = collection(data, split)
    available_ids = {str(row.get("id", row["path"])) for row, _ in available}
    if not common_ids.issubset(available_ids):
        raise ValueError("Benchmark contains cases absent from the dataset split")
    selected = [(row, path) for row, path in available if str(row.get("id", row["path"])) in common_ids]
    selected = selected if case_limit is None else selected[:case_limit]
    rows = [row for row, _ in selected]
    references = [Case.load(path) for _, path in selected]
    for method, mapping in indexed.items():
        for row, case in zip(rows, references):
            case_id = str(row.get("id", row["path"]))
            recorded = mapping[case_id]
            if recorded.get("source_case_hash") != str(case.arrays["case_hash"].item()) or recorded.get("group") != row["group"]:
                raise ValueError(f"Benchmark inputs differ from dataset case: {case_id}")
            if recorded.get("status") not in {"completed", "failed"}:
                raise ValueError("Benchmark case has an unsupported completion status")
            if recorded.get("requested_steps") != len(case.arrays["time"]) - 1:
                raise ValueError("Benchmark requested horizon differs from the dataset")
    output.mkdir(parents=True, exist_ok=True)
    results = {
        "format": "cut-dynamics-diagnostics-v1", "status": "in_progress",
        "source_report": str(report_path), "data": str(Path(data).resolve()), "split": split,
        "device": str(device), "selected_cases": len(rows), "source_report_cases": len(common_ids),
        "training_executed": False, "autonomous_rollout_recomputed": False,
        "methods": {}, "zero_internal_force": {"cases": []},
        "scope": "Saved autonomous trajectories, teacher-forced one-step prediction, stationary counterfactual probes and analytic zero-internal-force baseline",
        "rest_probe_scope": "Same cut geometry and grip mask with stationary grips; a zero-load counterfactual, cached across loading combinations",
    }
    started = perf_counter()
    save_diagnostics(output, results)
    baseline_metrics = {}
    try:
        for row, case in zip(rows, references):
            case_id = str(row.get("id", row["path"]))
            baseline = zero_internal_force(case)
            free = free_motion_metrics(case, baseline)
            baseline_metrics[case_id] = free
            results["zero_internal_force"]["cases"].append({
                "case": case_id, "group": row["group"], "free_motion": free,
                "momentum_balance": momentum_balance(case, baseline),
            })
        baseline_keys = ("position_rmse", "velocity_rmse", "position_relative_rmse", "velocity_relative_rmse")
        results["zero_internal_force"]["summary"] = {
            key: mean_values(list(baseline_metrics.values()), key) for key in baseline_keys
        }
        for method, source_method in source_methods.items():
            checkpoint_path = Path(source_method["checkpoint"])
            if not checkpoint_path.is_file():
                raise ValueError(f"Checkpoint recorded in benchmark is unavailable: {checkpoint_path}")
            digest = checkpoint_hash(checkpoint_path)
            original_digest = source_method.get("checkpoint_sha256")
            if original_digest is not None and digest != original_digest:
                raise ValueError("Checkpoint changed since the benchmark was recorded")
            model, checkpoint = load_model(checkpoint_path, device)
            if checkpoint.get("trained") is not True:
                raise ValueError("Diagnostics require a trained checkpoint")
            model_method = checkpoint["model_config"]["method"]
            architecture = checkpoint["model_config"].get("architecture", "legacy")
            integrator = getattr(model, "integrator", "legacy_constant_acceleration")
            if model_method != source_method.get("model_method", method):
                raise ValueError("Checkpoint method differs from its benchmark label")
            if source_method.get("architecture", architecture) != architecture:
                raise ValueError("Checkpoint architecture differs from its benchmark identity")
            if source_method.get("integrator", integrator) != integrator:
                raise ValueError("Checkpoint integrator differs from its benchmark identity")
            check_checkpoint(checkpoint, rows, references, split)
            method_results = {
                "checkpoint": str(checkpoint_path.resolve()), "checkpoint_sha256": digest,
                "checkpoint_identity_status": "verified_original_checksum" if original_digest else "original_checkpoint_checksum_unavailable",
                "model_method": model_method, "architecture": architecture, "integrator": integrator,
                "parameter_count": sum(parameter.numel() for parameter in model.parameters()),
                "rest_probes": {}, "cases": [],
            }
            results["methods"][method] = method_results
            for index, (row, case) in enumerate(zip(rows, references)):
                case_id = str(row.get("id", row["path"]))
                recorded = indexed[method][case_id]
                identity = {name: source_method[name] for name in
                            ("model_method", "architecture", "integrator", "checkpoint_sha256") if name in source_method}
                prediction = saved_prediction(case, recorded, report_path.parent, method, identity)
                key = geometry_key(case)
                if key not in method_results["rest_probes"]:
                    method_results["rest_probes"][key] = dict(rest_probe(model, case, device), source_case=case_id, group=row["group"])
                local_prediction, local_status = teacher_forced(model, case, device)
                local_free = free_motion_metrics(case, local_prediction) if len(local_prediction["position"]) >= 2 else None
                local = dict(local_status, free_motion=local_free)
                autonomous = {"status": "unavailable", "reason": "No saved finite trajectory with at least two frames"}
                comparisons = {}
                if prediction is not None:
                    free = free_motion_metrics(case, prediction)
                    autonomous = {"status": "available", "free_motion": free,
                                  "momentum_balance": momentum_balance(case, prediction)}
                    if recorded["status"] == "completed":
                        comparisons = {
                            name: improvement(free[name], baseline_metrics[case_id][name])
                            for name in ("position_rmse", "velocity_rmse")
                        }
                depth = case.metadata.get("cut_depth")
                if depth is None:
                    match = re.search(r"-depth(\d+)$", str(row["group"]))
                    depth = int(match.group(1)) if match else None
                method_results["cases"].append({
                    "case": case_id, "group": row["group"], "depth": depth,
                    "source_case_hash": str(case.arrays["case_hash"].item()),
                    "saved_rollout_status": recorded["status"], "saved_rollout_failure": recorded.get("failure"),
                    "completed_steps": recorded["completed_steps"], "requested_steps": recorded["requested_steps"],
                    "autonomous": autonomous, "teacher_forced": local,
                    "improvement_over_zero_internal_force": comparisons, "rest_probe_key": key,
                })
                emit("diagnostic_case_completed", method=method, index=index + 1, cases=len(rows),
                     case=case_id, saved_rollout_status=recorded["status"], teacher_forced_status=local["status"])
                if (index + 1) % 10 == 0:
                    save_diagnostics(output, results)
            method_results["summary"] = summarize(method_results["cases"])
            depths = {row["depth"] for row in method_results["cases"]}
            method_results["by_depth"] = {
                str(depth): summarize([row for row in method_results["cases"] if row["depth"] == depth])
                for depth in sorted(depths, key=lambda value: -1 if value is None else value)
            }
            method_results["rest_probe_count"] = len(method_results["rest_probes"])
            method_results["rest_probe_summary"] = summarize_rest(method_results["rest_probes"])
            save_diagnostics(output, results)
        results["status"] = "completed"
    except Exception as error:
        results["status"] = "failed"
        results["failure"] = f"{type(error).__name__}: {error}"
        results["runtime_seconds"] = perf_counter() - started
        save_diagnostics(output, results)
        raise
    results["runtime_seconds"] = perf_counter() - started
    save_diagnostics(output, results)
    return {"status": results["status"], "report": str(output / "diagnostics.json"),
            "training_executed": False, "autonomous_rollout_recomputed": False,
            "zero_internal_force": results["zero_internal_force"]["summary"],
            "methods": {method: dict(value["summary"], rest_probe=value["rest_probe_summary"])
                        for method, value in results["methods"].items()}}
