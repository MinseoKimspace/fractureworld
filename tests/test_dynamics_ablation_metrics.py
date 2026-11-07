from copy import deepcopy
import json

import pytest

from fracture_world.dynamics_ablation_metrics import motion_ablation_comparisons, summarize_motion_ablation


POSITION = "position_mass_weighted_rmse_m"
VELOCITY = "velocity_mass_weighted_rmse_m_per_s"
METHODS = ("central", "no_motion_central")


def evaluation(position, velocity):
    return {POSITION: position, VELOCITY: velocity}


def record(method, candidate=0, position=1., velocity=2., **extra):
    return dict(method=method, candidate=candidate, candidate_hash=f"crack-{candidate}", accepted=True,
                verified_constraints=True, evaluation=evaluation(position, velocity), **extra)


def selection(candidate=0, complete=True):
    return dict(selected_candidate=candidate, selection_complete=complete)


def inputs():
    records = [record(method, position=3. if method.startswith("no_motion") else 1.,
                      velocity=5. if method.startswith("no_motion") else 2.) for method in METHODS]
    return records, {method: selection() for method in METHODS}


def find(comparisons, solver="central", scope="fixed_candidate", candidate=0):
    return next(item for item in comparisons if item["solver"] == solver and item["scope"] == scope
                and (scope == "selected" or item["learned_motion"]["candidate"] == candidate))


def test_paired_errors_measure_motion_contribution_without_changing_inputs_or_selection():
    records, selections = inputs()
    original = deepcopy((records, selections))
    for item in records:
        item["selection_objective"] = -1e100 if item["method"].startswith("no_motion") else 1e100
    comparisons = motion_ablation_comparisons(records, selections)
    pair = find(comparisons)
    assert pair["comparison_available"] is True
    assert pair["rmse_no_motion_minus_learned"] == {POSITION: 2., VELOCITY: 3.}
    assert pair["same_candidate"] is True
    assert selections == original[1]
    assert [item["evaluation"] for item in records] == [item["evaluation"] for item in original[0]]


def test_selected_scope_reports_different_fracture_selection_separately_from_fixed_candidates():
    records, selections = inputs()
    records.append(record("no_motion_central", 1, position=4., velocity=6.))
    selections["no_motion_central"] = selection(1)
    comparisons = motion_ablation_comparisons(records, selections)
    selected = find(comparisons, scope="selected")
    assert selected["comparison_available"] is True
    assert selected["same_candidate"] is False
    assert selected["rmse_no_motion_minus_learned"][POSITION] == 3.
    missing = find(comparisons, candidate=1)
    assert missing["comparison_available"] is False
    assert missing["status"] == "missing_method_record"


def test_fixed_scope_rejects_inconsistent_candidate_hashes():
    records, selections = inputs()
    records[1]["candidate_hash"] = "different-fracture"
    fixed = find(motion_ablation_comparisons(records, selections))
    assert fixed["status"] == "candidate_hash_mismatch"
    assert fixed["rmse_no_motion_minus_learned"][POSITION] is None


def test_shared_pool_count_preserves_unattempted_denominators_even_when_both_records_are_missing():
    records, selections = inputs()
    comparisons = motion_ablation_comparisons(records, selections, candidate_count=3)
    summary = summarize_motion_ablation([{"motion_ablation_comparisons": comparisons}])
    fixed = summary["central"]["fixed_candidate"]
    assert fixed["requested_pairs"] == 3
    assert fixed["available_pairs"] == 1
    assert fixed["unavailable_pairs"] == 2
    assert fixed["learned_motion"]["unattempted_pairs"] == 2
    assert fixed["no_motion"]["unattempted_pairs"] == 2
    assert fixed["paired_mean_rmse"]["learned_motion"][POSITION] == 1.


@pytest.mark.parametrize("value", [float("nan"), float("inf"), -1., True, "1.0"])
def test_nonphysical_or_nonfinite_reference_metrics_never_enter_paired_accuracy(value):
    records, selections = inputs()
    records[1]["evaluation"][VELOCITY] = value
    comparisons = motion_ablation_comparisons(records, selections)
    assert find(comparisons)["status"] == "invalid_reference_metrics"
    summary = summarize_motion_ablation([{"motion_ablation_comparisons": comparisons}])
    assert summary["central"]["fixed_candidate"]["available_pairs"] == 0
    assert summary["central"]["fixed_candidate"]["paired_mean_rmse"]["learned_motion"][POSITION] is None


def test_reference_callback_is_never_invoked_by_postevaluation_metrics():
    records, selections = inputs()

    def prohibited():
        raise AssertionError("reference callback must already have been evaluated outside this module")

    records[1]["evaluation"] = prohibited
    pair = find(motion_ablation_comparisons(records, selections))
    assert pair["status"] == "invalid_reference_metrics"


def test_missing_reference_is_distinct_from_method_failure_and_incomplete_selection():
    records, selections = inputs()
    records[1].pop("evaluation")
    comparisons = motion_ablation_comparisons(records, selections)
    assert find(comparisons)["status"] == "missing_reference_metrics"
    records[1]["accepted"] = False
    assert find(motion_ablation_comparisons(records, selections))["status"] == "method_not_accepted"
    selections["central"]["selection_complete"] = False
    assert find(motion_ablation_comparisons(records, selections), scope="selected")["status"] == "incomplete_selection"


def test_constraint_unknown_is_counted_without_silently_filtering_prediction_error():
    records, selections = inputs()
    records[0]["verified_constraints"] = None
    records[1]["verified_constraints"] = False
    comparisons = motion_ablation_comparisons(records, selections)
    assert find(comparisons)["comparison_available"] is True
    fixed = summarize_motion_ablation([{"motion_ablation_comparisons": comparisons}])["central"]["fixed_candidate"]
    assert fixed["learned_motion"]["constraints_unknown_pairs"] == 1
    assert fixed["no_motion"]["constraints_violated_pairs"] == 1
    assert fixed["available_pairs"] == 1


def test_summary_accuracy_is_paired_and_conditional_without_zero_filling_failed_runs():
    records, selections = inputs()
    second_records, second_selections = inputs()
    second_records[0]["evaluation"] = evaluation(1000., 2000.)
    second_records[1]["accepted"] = False
    rows = [{"motion_ablation_comparisons": motion_ablation_comparisons(records, selections)},
            {"motion_ablation_comparisons": motion_ablation_comparisons(second_records, second_selections)}]
    summary = summarize_motion_ablation(iter(rows))
    fixed = summary["central"]["fixed_candidate"]
    assert fixed["requested_pairs"] == 2
    assert fixed["available_pairs"] == 1
    assert fixed["learned_motion"]["accepted_pairs"] == 2
    assert fixed["no_motion"]["not_accepted_pairs"] == 1
    assert fixed["paired_mean_rmse"]["learned_motion"][POSITION] == 1.
    assert fixed["mean_no_motion_minus_learned"][VELOCITY] == 3.
    assert summary == summarize_motion_ablation(rows)


def test_incremental_summary_matches_one_call_across_failures_missing_pairs_and_serialization():
    records, selections = inputs()
    failed_records, failed_selections = inputs()
    failed_records[1]["accepted"] = False
    failed_records[0]["evaluation"] = evaluation(1000., 2000.)
    rows = [{"motion_ablation_comparisons": motion_ablation_comparisons(records, selections, candidate_count=2)},
            {"motion_ablation_comparisons": motion_ablation_comparisons(failed_records, failed_selections)},
            {"motion_ablation_comparisons": motion_ablation_comparisons([], {}, candidate_count=3)}]
    expected = summarize_motion_ablation(rows)
    first = summarize_motion_ablation(rows[:1])
    restored = json.loads(json.dumps(first, allow_nan=False))
    actual = summarize_motion_ablation(iter(rows[1:]), restored)
    assert actual == expected
    assert actual["central"]["fixed_candidate"]["requested_pairs"] == 6
    assert actual["central"]["fixed_candidate"]["available_pairs"] == 1
    assert actual["central"]["fixed_candidate"]["paired_mean_rmse"]["learned_motion"][POSITION] == 1.


def test_summary_selected_hash_agreement_counts_selection_changes_and_unknowns():
    records, selections = inputs()
    records.append(record("no_motion_central", 1))
    selections["no_motion_central"] = selection(1)
    comparisons = motion_ablation_comparisons(records, selections)
    selected = summarize_motion_ablation([{"motion_ablation_comparisons": comparisons}])["central"]["selected"]
    assert selected["candidate_hash_agreement"] == dict(same=0, different=1, unknown=0)


def test_no_available_results_preserve_missing_accuracy_instead_of_reporting_zero():
    comparisons = motion_ablation_comparisons([], {}, candidate_count=2)
    summary = summarize_motion_ablation([{"motion_ablation_comparisons": comparisons}, {"status": "unknown_sparse_transition"}])
    for scopes in summary.values():
        assert scopes["fixed_candidate"]["requested_pairs"] == 2
        for values in scopes.values():
            assert values["available_pairs"] == 0
            assert all(value is None for value in values["mean_no_motion_minus_learned"].values())


def test_duplicate_method_candidate_records_are_rejected_instead_of_silently_overwriting():
    records, selections = inputs()
    records.append(deepcopy(records[0]))
    with pytest.raises(ValueError, match="unique"):
        motion_ablation_comparisons(records, selections)


@pytest.mark.parametrize("count", [-1, 1.5, True])
def test_invalid_pool_count_is_rejected(count):
    with pytest.raises(ValueError, match="candidate count"):
        motion_ablation_comparisons([], {}, candidate_count=count)
