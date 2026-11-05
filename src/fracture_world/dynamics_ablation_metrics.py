from collections import Counter
from collections.abc import Mapping
from math import isfinite
from numbers import Integral, Real


MOTION_PAIRS = (("central", "no_motion_central"), ("consensus", "no_motion_consensus"))
RMSE_METRICS = ("position_mass_weighted_rmse_m", "velocity_mass_weighted_rmse_m_per_s")


def _rmse(evaluation):
    values = dict.fromkeys(RMSE_METRICS)
    if evaluation is None:
        return values, "missing"
    if not isinstance(evaluation, Mapping):
        return values, "invalid"
    missing, invalid = False, False
    for name in RMSE_METRICS:
        value = evaluation.get(name)
        if value is None:
            missing = True
        elif isinstance(value, bool) or not isinstance(value, Real):
            invalid = True
        else:
            try:
                number = float(value)
            except (OverflowError, ValueError):
                invalid = True
                continue
            if not isfinite(number) or number < 0:
                invalid = True
            else:
                values[name] = number
    return values, "invalid" if invalid else "missing" if missing else "available"


def _side(method, candidate, record, selection=None):
    evaluation = record.get("evaluation") if record is not None else None
    if selection is not None and "evaluation" in selection:
        evaluation = selection["evaluation"]
    values, metric_status = _rmse(evaluation)
    verified = record.get("verified_constraints") if record is not None else None
    return dict(method=method, candidate=candidate,
                candidate_hash=record.get("candidate_hash") if record is not None else None,
                record_present=record is not None,
                accepted=bool(record.get("accepted")) if record is not None else None,
                verified_constraints=verified if isinstance(verified, bool) else None,
                reference_metrics_status=metric_status, rmse=values)


def _comparison(solver, scope, learned, no_motion, selection_status=None):
    first_hash, second_hash = learned["candidate_hash"], no_motion["candidate_hash"]
    known_hashes = all(isinstance(value, str) and bool(value) for value in (first_hash, second_hash))
    same_candidate = first_hash == second_hash if known_hashes else None
    status = selection_status
    if status is None:
        if not learned["record_present"] or not no_motion["record_present"]:
            status = "missing_method_record"
        elif not learned["accepted"] or not no_motion["accepted"]:
            status = "method_not_accepted"
        elif scope == "fixed_candidate" and same_candidate is not True:
            status = "candidate_hash_mismatch" if same_candidate is False else "candidate_hash_unavailable"
        elif "invalid" in (learned["reference_metrics_status"], no_motion["reference_metrics_status"]):
            status = "invalid_reference_metrics"
        elif "missing" in (learned["reference_metrics_status"], no_motion["reference_metrics_status"]):
            status = "missing_reference_metrics"
        else:
            status = "available"
    available = status == "available"
    difference = {name: no_motion["rmse"][name] - learned["rmse"][name] if available else None
                  for name in RMSE_METRICS}
    return dict(solver=solver, scope=scope, comparison_available=available, status=status,
                learned_motion=learned, no_motion=no_motion, same_candidate=same_candidate,
                rmse_no_motion_minus_learned=difference,
                accuracy_scope="reference prediction error after selection; objective values with different motion anchors are not quality scores")


def motion_ablation_comparisons(records, selections, *, candidate_count=None):
    if not isinstance(selections, Mapping):
        raise TypeError("motion ablation selections must be a method mapping")
    if candidate_count is not None and (isinstance(candidate_count, bool) or not isinstance(candidate_count, Integral) or candidate_count < 0):
        raise ValueError("motion ablation candidate count must be a nonnegative integer")
    paired_methods = {method for pair in MOTION_PAIRS for method in pair}
    indexed = {}
    candidates = set()
    for record in records:
        if record.get("method") not in paired_methods:
            continue
        candidate = record.get("candidate")
        if isinstance(candidate, bool) or not isinstance(candidate, Integral) or candidate < 0:
            raise ValueError("motion ablation records require nonnegative integer candidate indices")
        candidate = int(candidate)
        if candidate_count is not None and candidate >= candidate_count:
            raise ValueError("motion ablation candidate index exceeds the shared pool")
        key = record["method"], candidate
        if key in indexed:
            raise ValueError("motion ablation records must be unique per method and candidate")
        indexed[key] = record
        candidates.add(candidate)
    candidates = sorted(candidates) if candidate_count is None else range(int(candidate_count))
    comparisons = []
    for learned_method, no_motion_method in MOTION_PAIRS:
        for candidate in candidates:
            learned = _side(learned_method, candidate, indexed.get((learned_method, candidate)))
            no_motion = _side(no_motion_method, candidate, indexed.get((no_motion_method, candidate)))
            comparisons.append(_comparison(learned_method, "fixed_candidate", learned, no_motion))
        selected = [selections.get(method) for method in (learned_method, no_motion_method)]
        status = None
        if not all(isinstance(value, Mapping) for value in selected):
            status = "missing_selection"
        elif not all(value.get("selection_complete") is True for value in selected):
            status = "incomplete_selection"
        elif any(value.get("selected_candidate") is None for value in selected):
            status = "no_selected_candidate"
        sides = []
        for method, selection in zip((learned_method, no_motion_method), selected):
            candidate = selection.get("selected_candidate") if isinstance(selection, Mapping) else None
            valid_index = not isinstance(candidate, bool) and isinstance(candidate, Integral) and candidate >= 0
            if candidate is not None and not valid_index:
                status = "invalid_selected_candidate"
            candidate = int(candidate) if valid_index else None
            record = indexed.get((method, candidate)) if candidate is not None else None
            sides.append(_side(method, candidate, record, selection if isinstance(selection, Mapping) else None))
        comparisons.append(_comparison(learned_method, "selected", *sides, selection_status=status))
    return comparisons


def _accumulator():
    counts = ("recorded_pairs", "accepted_pairs", "not_accepted_pairs", "unattempted_pairs",
              "reference_available_pairs", "reference_missing_pairs", "reference_invalid_pairs",
              "constraints_verified_pairs", "constraints_violated_pairs", "constraints_unknown_pairs")
    return dict(requested_pairs=0, available_pairs=0, status_counts=Counter(),
                learned_motion=dict.fromkeys(counts, 0), no_motion=dict.fromkeys(counts, 0),
                candidate_hash_agreement=dict(same=0, different=0, unknown=0),
                paired_mean_rmse={side: dict.fromkeys(RMSE_METRICS) for side in ("learned_motion", "no_motion")},
                mean_no_motion_minus_learned=dict.fromkeys(RMSE_METRICS))


def summarize_motion_ablation(rows, summary=None):
    if summary is None:
        summary = {solver: {scope: _accumulator() for scope in ("fixed_candidate", "selected")}
                   for solver, _ in MOTION_PAIRS}
    else:
        for solver, _ in MOTION_PAIRS:
            for scope in ("fixed_candidate", "selected"):
                summary[solver][scope]["status_counts"] = Counter(summary[solver][scope]["status_counts"])
    for row in rows:
        for comparison in row.get("motion_ablation_comparisons", []):
            solver, scope = comparison["solver"], comparison["scope"]
            if solver not in summary or scope not in summary[solver]:
                raise ValueError("motion ablation comparison has an unknown solver or scope")
            values = summary[solver][scope]
            values["requested_pairs"] += 1
            values["status_counts"][comparison["status"]] += 1
            same = comparison.get("same_candidate")
            values["candidate_hash_agreement"]["same" if same is True else "different" if same is False else "unknown"] += 1
            for name in ("learned_motion", "no_motion"):
                side, counts = comparison[name], values[name]
                counts["recorded_pairs"] += int(side["record_present"])
                counts["unattempted_pairs"] += int(not side["record_present"])
                counts["accepted_pairs"] += int(side["accepted"] is True)
                counts["not_accepted_pairs"] += int(side["accepted"] is False)
                counts[f"reference_{side['reference_metrics_status']}_pairs"] += 1
                verified = side["verified_constraints"]
                counts["constraints_verified_pairs" if verified is True else
                       "constraints_violated_pairs" if verified is False else "constraints_unknown_pairs"] += 1
            if comparison["comparison_available"]:
                parsed = {name: _rmse(comparison[name]["rmse"]) for name in ("learned_motion", "no_motion")}
                if any(status != "available" for _, status in parsed.values()):
                    raise ValueError("available motion comparison requires finite nonnegative paired reference errors")
                values["available_pairs"] += 1
                count = values["available_pairs"]
                for metric in RMSE_METRICS:
                    for name in ("learned_motion", "no_motion"):
                        previous = values["paired_mean_rmse"][name][metric]
                        current = parsed[name][0][metric]
                        values["paired_mean_rmse"][name][metric] = current if previous is None else previous * ((count - 1) / count) + current / count
                    previous = values["mean_no_motion_minus_learned"][metric]
                    delta = parsed["no_motion"][0][metric] - parsed["learned_motion"][0][metric]
                    values["mean_no_motion_minus_learned"][metric] = delta if previous is None else previous * ((count - 1) / count) + delta / count
    for scopes in summary.values():
        for values in scopes.values():
            values["unavailable_pairs"] = values["requested_pairs"] - values["available_pairs"]
            values["status_counts"] = dict(values["status_counts"])
            values["accuracy_conditioning"] = "paired available reference metrics only; acceptance and constraint counts are reported separately"
            values["difference_interpretation"] = "positive no_motion_minus_learned means learned motion has lower reference prediction error"
    return summary
