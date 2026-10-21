from dataclasses import replace

import numpy as np
import pytest

from fracture_world.dynamics_data import DynamicsTrajectory, material_topology, split_dynamics_cases
from fracture_world.dynamics_readiness import inspect_dynamics_dataset
from fracture_world.mesh import VolumeMesh
import fracture_world.dynamics_readiness as readiness


def trajectory(case_id, fracture=False, release=False, event_frame=1, steps=3, condition=None, fixed=True):
    nodes = np.asarray([[0, 0, 0], [1, 0, 0], [0, 1, 0], [0, 0, 1], [0, 0, -1]], dtype=float)
    topology = material_topology(VolumeMesh(nodes, np.asarray([[0, 1, 2, 3], [0, 2, 1, 4]])), 1.0)
    positions = np.broadcast_to(topology.rest_positions, (steps + 1, 8, 3)).copy()
    broken = np.zeros((steps + 1, 1), dtype=bool)
    supports = np.zeros((steps + 1, 8), dtype=bool)
    if fixed:
        supports[:, topology.mesh.tets.ravel() == 0] = True
    if fracture:
        broken[event_frame:] = True
    if release:
        supports[event_frame:] = False
    return DynamicsTrajectory(topology, positions, np.zeros_like(positions), broken, supports,
                              np.arange(steps + 1) * 0.25, np.zeros(3), {}, np.zeros((8, 3)), case_id,
                              condition or case_id, dict(conditioning_context=dict(dt=0.25, steps=steps)))


def dataset_with_coverage(seed=17):
    items = [trajectory(f"case-{index}") for index in range(20)]
    result = []
    for rows in split_dynamics_cases(items, seed=seed).values():
        for index, item in enumerate(rows):
            result.append(trajectory(item.case_id, fracture=index == 0, release=index == 0))
    return result


def inspect(monkeypatch, items, **kwargs):
    monkeypatch.setattr(readiness, "load_dynamics_dataset", lambda directory: items)
    return inspect_dynamics_dataset("unused", **kwargs)


def test_held_out_event_coverage_and_original_support_identity(monkeypatch):
    report = inspect(monkeypatch, dataset_with_coverage())
    assert report["ready_for_training"]
    assert report["status"] == "ready"
    assert report["totals"]["newly_broken_original_facets"] == 3
    assert report["totals"]["released_material_corner_copies"] == 6
    assert report["totals"]["released_original_support_vertices"] == 3
    assert report["totals"]["original_vertex_release_transition_events"] == 3
    assert report["provenance"]["physical_ground_truth"] is False
    assert report["provenance"]["native_engine_calls"] == 0
    for split in report["splits"].values():
        assert split["fracture_positive_cases"] >= 1
        assert split["fracture_negative_cases"] >= 1
        assert split["support_release_positive_cases_with_post_event_window"] >= 1
        assert split["support_release_negative_cases"] >= 1
    for case in report["cases"]:
        if case["support_release_positive"]:
            assert case["support_release_window"]["native_steps_after_first_event"] == 2
            assert case["released_material_corner_copies"] == 2
            assert case["released_original_support_vertices"] == 1


def test_absent_events_are_not_reported_as_training_ready(monkeypatch):
    report = inspect(monkeypatch, [trajectory(f"case-{index}") for index in range(20)])
    assert not report["ready_for_training"]
    assert report["totals"]["fracture_positive_cases"] == 0
    assert report["totals"]["support_release_positive_cases"] == 0
    missing = {(reason["split"], reason["event"]) for reason in report["reasons"]
               if reason["code"] == "missing_event_class" and reason["missing"] == "positive"}
    assert missing == {(split, event) for split in ("train", "validation", "test")
                       for event in ("fracture", "support_release")}


def test_original_vertex_releasing_in_two_steps_is_one_independent_support(monkeypatch):
    item = trajectory("partial-release", fracture=True)
    flags = item.fixed.copy()
    copies = np.flatnonzero(item.topology.mesh.tets.ravel() == 0)
    flags[1:, copies[0]] = False
    flags[2:, copies[1]] = False
    report = inspect(monkeypatch, [replace(item, fixed=flags)], overfit=True)
    case = report["cases"][0]
    assert case["released_material_corner_copies"] == 2
    assert case["original_vertex_release_transition_events"] == 2
    assert case["released_original_support_vertices"] == 1
    assert case["support_release_window"]["native_steps_after_first_event"] == 2
    assert case["support_release_window"]["native_steps_after_last_event"] == 1


def test_final_frame_events_do_not_provide_post_event_rollout(monkeypatch):
    items = [trajectory(item.case_id, fracture=item.broken.any(), release=not item.fixed[-1].any(), event_frame=3)
             for item in dataset_with_coverage()]
    report = inspect(monkeypatch, items)
    assert not report["ready_for_training"]
    reasons = [reason for reason in report["reasons"] if reason["code"] == "missing_post_event_window"]
    assert len(reasons) == 6
    for row in report["cases"]:
        if row["fracture_positive"]:
            assert row["fracture_window"]["native_steps_after_first_event"] == 0


def test_unknown_clock_stays_unknown_and_sparse_recording_is_not_ready(monkeypatch):
    items = dataset_with_coverage()
    items[0] = replace(items[0], metadata={})
    sparse = items[1]
    items[1] = replace(sparse, times=sparse.times[[0, 2, 3]], positions=sparse.positions[[0, 2, 3]],
                       velocities=sparse.velocities[[0, 2, 3]], broken=sparse.broken[[0, 2, 3]],
                       fixed=sparse.fixed[[0, 2, 3]])
    report = inspect(monkeypatch, items)
    assert not report["ready_for_training"]
    assert report["cases"][0]["native_clock_status"] == "unknown"
    assert report["cases"][0]["recorded_every_native_step"] is None
    assert report["totals"]["cases_with_unknown_native_clock"] == 1
    assert report["totals"]["cases_missing_native_steps"] == 1


def test_complete_loading_replicates_cannot_leak_between_splits(monkeypatch):
    items = dataset_with_coverage()
    items.append(replace(items[0], case_id="replicate-with-another-name"))
    report = inspect(monkeypatch, items)
    assert report["leakage"]["case_ids_disjoint"]
    assert report["leakage"]["loading_conditions_disjoint"]
    sets = [set(split["case_ids"]) for split in report["splits"].values()]
    assert any({items[0].case_id, items[-1].case_id} <= selected for selected in sets)


def test_event_stratification_preserves_rare_positive_cases_in_each_split(monkeypatch):
    items = [trajectory(f"case-{index}", fracture=index < 3, release=index < 3) for index in range(24)]
    report = inspect(monkeypatch, items, event_stratified=True)
    assert report["ready_for_training"]
    assert report["split_strategy"] == "event_stratified_loading_conditions"
    assert report["provenance"]["event_labels_used_for_split_only"]
    for split in report["splits"].values():
        assert split["fracture_positive_cases"] == 1
        assert split["support_release_positive_cases"] == 1
        assert split["fracture_negative_cases"] >= 1
        assert split["support_release_negative_cases"] >= 1


def test_event_stratification_is_deterministic_and_keeps_mixed_replicates_together():
    items = [trajectory(f"case-{index}", fracture=index < 3, release=index < 3) for index in range(24)]
    items.append(trajectory("short-negative-replicate", condition="case-0"))
    first = split_dynamics_cases(items, seed=31, event_stratified=True)
    second = split_dynamics_cases(list(reversed(items)), seed=31, event_stratified=True)
    assert {name: [item.case_id for item in rows] for name, rows in first.items()} == {
        name: [item.case_id for item in rows] for name, rows in second.items()
    }
    selected = [set(item.case_id for item in rows) for rows in first.values()]
    assert any({"case-0", "short-negative-replicate"} <= cases for cases in selected)
    condition_sets = [set(item.condition_digest for item in rows) for rows in first.values()]
    assert not condition_sets[0] & condition_sets[1]
    assert not condition_sets[0] & condition_sets[2]
    assert not condition_sets[1] & condition_sets[2]


def test_event_strata_with_one_or_two_conditions_do_not_claim_complete_coverage(monkeypatch):
    items = [trajectory("one-release", fracture=True, release=True),
             trajectory("fracture-a", fracture=True), trajectory("fracture-b", fracture=True),
             *[trajectory(f"negative-{index}") for index in range(12)]]
    splits = split_dynamics_cases(items, event_stratified=True)
    assert "one-release" in {item.case_id for item in splits["train"]}
    fracture_train = sum(item.case_id.startswith("fracture-") for item in splits["train"])
    fracture_validation = sum(item.case_id.startswith("fracture-") for item in splits["validation"])
    assert fracture_train == fracture_validation == 1
    assert not any(item.case_id.startswith("fracture-") for item in splits["test"])
    report = inspect(monkeypatch, items, event_stratified=True)
    assert not report["ready_for_training"]
    assert any(reason["code"] == "missing_event_class" and reason["event"] == "support_release"
               and reason["missing"] == "positive" for reason in report["reasons"])


def test_insufficient_conditions_and_overfit_are_explicitly_not_ready(monkeypatch):
    items = [trajectory("only-case", fracture=True, release=True)]
    report = inspect(monkeypatch, items)
    assert not report["ready_for_training"]
    assert any(reason["code"] == "split_unavailable" for reason in report["reasons"])
    report = inspect(monkeypatch, items, overfit=True)
    assert not report["ready_for_training"]
    assert any(reason["code"] == "overfit_debug" for reason in report["reasons"])


def test_cases_without_supports_are_not_negative_release_examples(monkeypatch):
    items = dataset_with_coverage()
    for index, item in enumerate(items):
        if not item.broken.any():
            items[index] = trajectory(item.case_id, fixed=False)
    report = inspect(monkeypatch, items)
    assert not report["ready_for_training"]
    assert report["totals"]["support_release_negative_cases"] == 0
    assert report["totals"]["cases_without_support_supervision"] == 17


def test_leakage_guard_reports_invalid_splitter_output(monkeypatch):
    items = dataset_with_coverage()
    monkeypatch.setattr(readiness, "split_dynamics_cases", lambda *args, **kwargs: dict(train=items, validation=items[:2], test=items[-2:]))
    report = inspect(monkeypatch, items)
    assert not report["ready_for_training"]
    assert not report["leakage"]["case_ids_disjoint"]
    assert not report["leakage"]["loading_conditions_disjoint"]
    assert any(reason["code"] == "split_leakage" for reason in report["reasons"])


@pytest.mark.parametrize("minimum", [0, -1, True, 1.5])
def test_post_event_minimum_requires_positive_integer(monkeypatch, minimum):
    with pytest.raises(ValueError, match="positive integer"):
        inspect(monkeypatch, dataset_with_coverage(), min_post_event_steps=minimum)
