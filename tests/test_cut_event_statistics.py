import copy
import json

import pytest

from fracture_world.cut_event_statistics import condition_rows, summarize_sweep


def condition(identity="event-x4-depth7-load8-at-step-0015", column=4, event_time=0.03,
              stale=0.004, updated=0.003, signal=0.1, floor=0.0001):
    methods = {label: {"status": "completed", "orientation_valid": True, "compared_horizon": 0.2,
                       "post_event_position_rmse": value, "post_event_velocity_rmse": value * 2}
               for label, value in (("no_cut", 0.00001), ("cut_stale", stale), ("cut_updated", updated))}
    causal = {label: {f"{field}_intervention_rmse": methods[label][f"post_event_{field}_rmse"]
                     for field in ("position", "velocity")} for label in ("cut_stale", "cut_updated")}
    for values in causal.values():
        values["compared_horizon"] = 0.2
        for field in ("position", "velocity"):
            values[f"teacher_{field}_intervention_rms"] = signal
            if floor is not None:
                values[f"teacher_{field}_intervention_refinement_rms"] = floor
    return {"case": identity, "status": "completed", "reference": {"converged": True},
            "neural_event_transfer": {"status": "conserved"},
            "inputs": {"source_group": f"x{column}-depth7", "event_time": event_time, "duration": 0.2,
                       "load": {"amplitude": 0.0225, "ramp_time": 0.048}},
            "methods": methods, "causal_response": causal}


def test_relative_response_uses_intervention_motion_and_paired_improvement():
    result = summarize_sweep([condition()])
    row = result["conditions"][0]
    assert row["column"] == 4
    assert row["cut_updated_position_relative_error"] == pytest.approx(0.03)
    assert row["position_absolute_improvement"] == pytest.approx(0.001)
    assert row["position_relative_improvement"] == pytest.approx(0.25)
    assert result["overall"]["metrics"]["position"]["win"] == 1
    assert "no_cut" not in result["overall"]["metrics"]["position"]
    json.dumps(result, allow_nan=False)


def test_counts_distributions_and_factor_groups_are_descriptive():
    cases = [condition("a", 1, 0.016, stale=0.004, updated=0.003),
             condition("b", 4, 0.024, stale=0.003, updated=0.003),
             condition("c", 7, 0.032, stale=0.002, updated=0.004)]
    result = summarize_sweep(cases)
    metrics = result["overall"]["metrics"]["position"]
    assert [metrics[name] for name in ("win", "tie", "loss")] == [1, 1, 1]
    assert metrics["updated_rmse"]["mean"] == pytest.approx(0.01 / 3)
    assert metrics["updated_rmse"]["median"] == 0.003
    assert metrics["updated_rmse"]["p95"] == pytest.approx(0.0039)
    assert metrics["updated_rmse"]["largest_cases"][0]["case"] == "c"
    assert set(result["by_column"]) == {"1", "4", "7"}
    assert set(result["by_event_time"]) == {"0.016", "0.024", "0.032"}
    assert len(result["by_load"]) == 1


@pytest.mark.parametrize("change,status", [
    ("failed", "method_failed"), ("invalid", "geometry_invalid"),
    ("unknown_geometry", "geometry_unknown"), ("missing", "method_unavailable"),
    ("reference_failed", "reference_failed"), ("unconverged", "reference_unconverged"),
    ("unknown_reference", "reference_unknown"), ("transfer", "event_transfer_failed"),
    ("unknown_transfer", "event_transfer_unknown"),
    ("metric", "metric_unavailable"),
])
def test_failed_invalid_unknown_and_missing_conditions_are_not_successes(change, status):
    case = condition()
    if change == "failed":
        case["methods"]["cut_updated"]["status"] = "failed"
    elif change == "invalid":
        case["methods"]["cut_updated"]["orientation_valid"] = False
    elif change == "unknown_geometry":
        del case["methods"]["cut_updated"]["orientation_valid"]
    elif change == "missing":
        del case["methods"]["cut_updated"]
    elif change == "reference_failed":
        case["reference"] = {"status": "failed", "converged": False}
    elif change == "unconverged":
        case["reference"]["converged"] = False
    elif change == "unknown_reference":
        del case["reference"]["converged"]
    elif change == "transfer":
        case["neural_event_transfer"]["status"] = "failed"
    elif change == "unknown_transfer":
        del case["neural_event_transfer"]
    else:
        case["methods"]["cut_updated"]["post_event_position_rmse"] = float("nan")
    result = summarize_sweep([case])
    assert result["conditions"][0]["pair_status"] == status
    assert result["overall"]["valid_paired_cases"] == 0
    assert result["overall"]["unpaired_cases"] == 1
    assert result["overall"]["metrics"]["position"]["valid_pairs"] == 0
    if change != "metric":
        assert result["conditions"][0]["cut_updated_position_signal_status"] == "unavailable"
    json.dumps(result, allow_nan=False)


@pytest.mark.parametrize("signal,floor,status,relative", [
    (0.0, 0.0001, "zero_signal", None),
    (0.0001, 0.0001, "unresolved_at_reference_precision", None),
    (0.00001, 0.0001, "unresolved_at_reference_precision", None),
    (0.1, None, "reference_precision_unknown", 0.03),
    (0.1, 0.0, "resolved", 0.03),
])
def test_zero_weak_and_uncalibrated_signal_are_reported_without_arbitrary_threshold(signal, floor, status, relative):
    value = condition_rows([condition(signal=signal, floor=floor)])[0]
    assert value["cut_updated_position_signal_status"] == status
    if relative is None:
        assert value["cut_updated_position_relative_error"] is None
    else:
        assert value["cut_updated_position_relative_error"] == pytest.approx(relative)


def test_no_cut_failure_invalidates_causal_ratio_but_not_paired_cut_errors():
    case = condition()
    case["methods"]["no_cut"]["status"] = "failed"
    result = summarize_sweep([case])
    assert result["overall"]["valid_paired_cases"] == 1
    assert result["conditions"][0]["cut_updated_position_signal_status"] == "unavailable"


def test_exact_zero_stale_error_has_no_relative_improvement_and_is_not_forced_to_win():
    row = condition_rows([condition(stale=0, updated=0.003)])[0]
    assert row["position_outcome"] == "loss"
    assert row["position_relative_improvement"] is None


@pytest.mark.parametrize("target", ["causal", "no_cut", "cut_updated"])
def test_causal_metric_requires_every_compared_horizon_to_reach_duration(target):
    case = condition()
    if target == "causal":
        case["causal_response"]["cut_updated"]["compared_horizon"] = 0.18
    else:
        case["methods"][target]["compared_horizon"] = 0.18
    row = condition_rows([case])[0]
    assert row["cut_updated_position_signal_status"] == "unavailable_horizon"
    assert row["cut_updated_position_relative_error"] is None


def test_precision_unknown_ratios_are_separate_from_resolved_distribution():
    result = summarize_sweep([condition(floor=None)])
    metric = result["overall"]["metrics"]["position"]["causal_relative_error"]["cut_updated"]
    assert metric["resolved_distribution"]["count"] == 0
    assert metric["available_ratio_distribution"]["count"] == 1


def test_rows_are_flat_with_sweep_identity_post_duration_and_actual_runtime():
    case = condition()
    case["sweep"] = {"column": 7, "event_step": 15}
    case["neural_total_runtime_seconds"] = 9.0
    row = condition_rows([case])[0]
    assert row["column"] == 7
    assert row["post_event_duration"] == pytest.approx(0.17)
    assert row["neural_total_runtime_seconds"] == 9.0
    assert all(not isinstance(value, (list, dict)) for value in row.values())


def test_duplicate_ids_rejected_and_empty_sweep_has_no_invented_metrics():
    case = condition()
    with pytest.raises(ValueError, match="unique"):
        summarize_sweep([case, copy.deepcopy(case)])
    result = summarize_sweep([])
    assert result["overall"]["requested_cases"] == 0
    assert result["overall"]["metrics"]["position"]["updated_rmse"]["mean"] is None
