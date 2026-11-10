import re
from collections import Counter, defaultdict

import numpy as np


FIELDS = ("position", "velocity")
CUT_METHODS = ("cut_stale", "cut_updated")


def error_value(value):
    if value is None or isinstance(value, bool):
        return None
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if np.isfinite(number) and number >= 0 else None


def qualification(case, labels):
    reference = case.get("reference", {})
    if reference.get("status") == "failed":
        return "reference_failed"
    if reference.get("converged") is not True:
        return "reference_unconverged" if reference.get("converged") is False else "reference_unknown"
    transfer = case.get("neural_event_transfer", {})
    if transfer.get("status") != "conserved":
        return "event_transfer_failed" if transfer.get("status") == "failed" else "event_transfer_unknown"
    for label in labels:
        method = case.get("methods", {}).get(label)
        if method is None:
            return "method_unavailable"
        if method.get("status") != "completed":
            return "method_failed" if method.get("status") == "failed" else "method_unknown"
        if method.get("orientation_valid") is not True:
            return "geometry_invalid" if method.get("orientation_valid") is False else "geometry_unknown"
    return "valid"


def relative_response(case, label, field):
    values = case.get("causal_response", {}).get(label, {})
    error = error_value(values.get(f"{field}_intervention_rmse"))
    signal = error_value(values.get(f"teacher_{field}_intervention_rms"))
    floor = error_value(values.get(f"teacher_{field}_intervention_refinement_rms"))
    result = {"intervention_rmse": error, "teacher_signal_rms": signal,
              "reference_refinement_rms": floor, "relative_error": None}
    duration = error_value(case.get("inputs", {}).get("duration"))
    horizons = [error_value(values.get("compared_horizon"))]
    horizons.extend(error_value(case.get("methods", {}).get(name, {}).get("compared_horizon"))
                    for name in ("no_cut", label))
    if qualification(case, ("no_cut", label)) != "valid" or error is None or signal is None:
        result["status"] = "unavailable"
    elif duration is None or any(value is None or value < duration for value in horizons):
        result["status"] = "unavailable_horizon"
    elif signal == 0:
        result["status"] = "zero_signal"
    elif floor is not None and signal <= floor:
        result["status"] = "unresolved_at_reference_precision"
    else:
        ratio = error / signal
        result["relative_error"] = float(ratio) if np.isfinite(ratio) else None
        result["status"] = ("resolved" if floor is not None else "reference_precision_unknown") if np.isfinite(ratio) else "unavailable"
    return result


def condition_rows(cases):
    rows = []
    for index, case in enumerate(cases):
        inputs, sweep = case.get("inputs", {}), case.get("sweep", {})
        group = str(inputs.get("source_group", inputs.get("source_case", "")))
        match = re.search(r"x(\d+)-depth(\d+)", group)
        load = inputs.get("load", {})
        duration, event_time = error_value(inputs.get("duration")), error_value(inputs.get("event_time"))
        row = {"case": str(case.get("case", f"condition-{index:04d}")), "source_group": group,
               "column": sweep.get("column", inputs.get("column", int(match.group(1)) if match else None)),
               "event_time": event_time, "event_step": sweep.get("event_step"), "duration": duration,
               "post_event_duration": duration - event_time if duration is not None and event_time is not None else None,
               "amplitude": error_value(load.get("amplitude")), "ramp_time": error_value(load.get("ramp_time")),
               "reference_converged": case.get("reference", {}).get("converged"),
               "pair_status": qualification(case, CUT_METHODS),
               "neural_total_runtime_seconds": error_value(case.get("neural_total_runtime_seconds"))}
        for label in ("no_cut", *CUT_METHODS):
            method = case.get("methods", {}).get(label, {})
            row[f"{label}_status"] = method.get("status", "unavailable")
            row[f"{label}_orientation_valid"] = method.get("orientation_valid")
            for metric in ("runtime_seconds", "prefix_and_post_runtime_seconds"):
                row[f"{label}_{metric}"] = error_value(method.get(metric))
            for field in FIELDS:
                row[f"{label}_{field}_rmse"] = error_value(method.get(f"post_event_{field}_rmse"))
                if label in CUT_METHODS:
                    for key, value in relative_response(case, label, field).items():
                        row[f"{label}_{field}_{'signal_status' if key == 'status' else key}"] = value
        for field in FIELDS:
            stale, updated = (row[f"{label}_{field}_rmse"] for label in CUT_METHODS)
            valid = row["pair_status"] == "valid" and stale is not None and updated is not None
            row[f"{field}_pair_status"] = row["pair_status"] if row["pair_status"] != "valid" else "valid" if valid else "metric_unavailable"
            row[f"{field}_outcome"] = ("win" if updated < stale else "loss" if updated > stale else "tie") if valid else None
            row[f"{field}_absolute_improvement"] = stale - updated if valid else None
            row[f"{field}_relative_improvement"] = (stale - updated) / stale if valid and stale > 0 else None
        if row["pair_status"] == "valid" and any(row[f"{field}_pair_status"] != "valid" for field in FIELDS):
            row["pair_status"] = "metric_unavailable"
        rows.append(row)
    if len({row["case"] for row in rows}) != len(rows):
        raise ValueError("Sweep condition IDs must be unique")
    return rows


def distribution(records):
    if not records:
        return {"count": 0, "mean": None, "median": None, "p95": None, "max": None, "largest_cases": []}
    values = np.asarray([value for _, value in records], dtype=np.float64)
    return {"count": len(values), "mean": float(np.mean(values)), "median": float(np.median(values)),
            "p95": float(np.percentile(values, 95)), "max": float(np.max(values)),
            "largest_cases": [{"case": identity, "value": value} for identity, value in
                              sorted(records, key=lambda value: (-value[1], value[0]))[:3]]}


def row_distribution(rows, key):
    return distribution([(row["case"], row[key]) for row in rows if row[key] is not None])


def summarize_rows(rows):
    statuses = Counter(row["pair_status"] for row in rows)
    summary = {"requested_cases": len(rows), "valid_paired_cases": statuses["valid"],
               "unpaired_cases": len(rows) - statuses["valid"],
               "failed_cases": sum(statuses[key] for key in ("method_failed", "reference_failed", "event_transfer_failed")),
               "invalid_geometry_cases": statuses["geometry_invalid"],
               "reference_unconverged_cases": statuses["reference_unconverged"],
               "unknown_or_unavailable_cases": sum(count for key, count in statuses.items() if key.endswith(("unknown", "unavailable"))),
               "pair_status_counts": dict(statuses), "metrics": {},
               "neural_total_runtime_seconds": row_distribution(rows, "neural_total_runtime_seconds")}
    for field in FIELDS:
        valid = [row for row in rows if row[f"{field}_pair_status"] == "valid"]
        outcomes = Counter(row[f"{field}_outcome"] for row in valid)
        metrics = {"valid_pairs": len(valid), **{name: outcomes[name] for name in ("win", "tie", "loss")}}
        for name, key in (("stale_rmse", f"cut_stale_{field}_rmse"), ("updated_rmse", f"cut_updated_{field}_rmse"),
                          ("absolute_improvement", f"{field}_absolute_improvement"), ("relative_improvement", f"{field}_relative_improvement")):
            metrics[name] = row_distribution(valid, key)
        metrics["causal_relative_error"] = {}
        for label in CUT_METHODS:
            prefix = f"{label}_{field}"
            resolved = [row for row in rows if row[f"{prefix}_signal_status"] == "resolved"]
            metrics["causal_relative_error"][label] = {
                "status_counts": dict(Counter(row[f"{prefix}_signal_status"] for row in rows)),
                "resolved_distribution": row_distribution(resolved, f"{prefix}_relative_error"),
                "available_ratio_distribution": row_distribution(rows, f"{prefix}_relative_error")}
        summary["metrics"][field] = metrics
    return summary


def summarize_sweep(cases):
    rows = condition_rows(cases)
    groups = {name: defaultdict(list) for name in ("column", "event_time", "post_event_duration", "load")}
    for row in rows:
        for name in groups:
            key = f"amplitude={row['amplitude']},ramp_time={row['ramp_time']}" if name == "load" else str(row[name])
            groups[name][key].append(row)
    return {"scope": "Descriptive condition grid; shared material, checkpoint and source families are not independent replicates; event times give unequal post-event durations",
            "comparison": "Frozen versus updated encoder context on the same separated physical mesh, not a pure edge ablation; no_cut uses its own uncut reference",
            "load_scope": "Existing load combinations; this sweep does not establish unseen-load generalization",
            "normalization": "Intervention error divided by teacher cut-minus-uncut motion, not object size",
            "reference_precision_scope": "Observed coarse-to-refined intervention difference, not a rigorous error bound",
            "runtime_scope": "Neural total counts the shared prefix once; per-method prefix-plus-post runtime is standalone cost",
            "overall": summarize_rows(rows), "conditions": rows,
            **{f"by_{name}": {key: summarize_rows(values) for key, values in sorted(partition.items())}
               for name, partition in groups.items()}}
