from itertools import combinations

import numpy as np

from .dynamics_data import load_dynamics_dataset, split_dynamics_cases


def _native_clock(trajectory):
    context = trajectory.metadata.get("conditioning_context", {})
    dt, steps = context.get("dt"), context.get("steps")
    if isinstance(dt, bool) or not isinstance(dt, (int, float)) or not np.isfinite(dt) or dt <= 0:
        return None
    if isinstance(steps, bool) or not isinstance(steps, int) or steps < 1:
        return None
    clock = np.r_[0.0, np.cumsum(np.full(steps, np.float32(dt)), dtype=np.float32).astype(float)]
    indices = np.searchsorted(clock, trajectory.times)
    if (indices >= len(clock)).any() or not np.array_equal(clock[indices], trajectory.times):
        return None
    if indices[0] != 0 or indices[-1] != steps:
        return None
    return indices


def _event_window(events, native_steps, minimum):
    frames = np.flatnonzero(np.any(events, axis=1)) + 1
    if not len(frames):
        return dict(event_transitions=0, first_event_native_step=None, last_event_native_step=None,
                    native_steps_after_first_event=None, native_steps_after_last_event=None,
                    has_post_event_window=False)
    if native_steps is None:
        return dict(event_transitions=len(frames), first_event_native_step=None, last_event_native_step=None,
                    native_steps_after_first_event=None, native_steps_after_last_event=None,
                    has_post_event_window=None)
    first, last, end = (int(native_steps[index]) for index in (frames[0], frames[-1], -1))
    return dict(event_transitions=len(frames), first_event_native_step=first, last_event_native_step=last,
                native_steps_after_first_event=end - first, native_steps_after_last_event=end - last,
                has_post_event_window=end - first >= minimum)


def _case_report(trajectory, minimum):
    fracture = ~trajectory.broken[:-1] & trajectory.broken[1:]
    release = trajectory.fixed[:-1] & ~trajectory.fixed[1:]
    original = trajectory.topology.mesh.tets.ravel()
    released_corners = np.any(release, axis=0)
    released_original_events = sum(len(np.unique(original[row])) for row in release)
    native_steps = _native_clock(trajectory)
    fracture_events, release_events = int(fracture.sum()), int(release.sum())
    support_opportunities = int(trajectory.fixed[:-1].sum())
    return dict(
        case_id=trajectory.case_id, condition_digest=trajectory.condition_digest,
        recorded_frames=len(trajectory.times), recorded_transitions=len(trajectory),
        native_steps=int(native_steps[-1]) if native_steps is not None else None,
        native_clock_status="known" if native_steps is not None else "unknown",
        recorded_every_native_step=bool(np.all(np.diff(native_steps) == 1)) if native_steps is not None else None,
        newly_broken_original_facets=fracture_events,
        intact_facet_transition_examples=int((~trajectory.broken[:-1] & ~trajectory.broken[1:]).sum()),
        released_material_corner_copies=release_events,
        released_original_support_vertices=int(len(np.unique(original[released_corners]))),
        original_vertex_release_transition_events=int(released_original_events),
        retained_support_corner_transition_examples=int((trajectory.fixed[:-1] & trajectory.fixed[1:]).sum()),
        initial_original_support_vertices=int(len(np.unique(original[trajectory.fixed[0]]))),
        fracture_positive=bool(fracture_events), fracture_negative=not bool(fracture_events),
        support_release_positive=bool(release_events),
        support_release_negative=bool(support_opportunities and not release_events),
        has_support_supervision=bool(support_opportunities),
        fracture_window=_event_window(fracture, native_steps, minimum),
        support_release_window=_event_window(release, native_steps, minimum),
    )


def _summary(rows):
    sums = ("recorded_transitions", "newly_broken_original_facets", "intact_facet_transition_examples",
            "released_material_corner_copies", "released_original_support_vertices",
            "original_vertex_release_transition_events", "retained_support_corner_transition_examples")
    result = dict(cases=len(rows), loading_conditions=len({row["condition_digest"] for row in rows}),
                  case_ids=[row["case_id"] for row in rows], **{key: sum(row[key] for row in rows) for key in sums})
    for event in ("fracture", "support_release"):
        for kind in ("positive", "negative"):
            result[f"{event}_{kind}_cases"] = sum(row[f"{event}_{kind}"] for row in rows)
        result[f"{event}_positive_cases_with_post_event_window"] = sum(
            row[f"{event}_positive"] and row[f"{event}_window"]["has_post_event_window"] is True for row in rows
        )
    result["cases_without_support_supervision"] = sum(not row["has_support_supervision"] for row in rows)
    result["cases_with_unknown_native_clock"] = sum(row["native_clock_status"] == "unknown" for row in rows)
    result["cases_missing_native_steps"] = sum(row["recorded_every_native_step"] is False for row in rows)
    return result


def inspect_dynamics_dataset(directory, seed=17, overfit=False, min_post_event_steps=1, event_stratified=False):
    if isinstance(seed, bool) or not isinstance(seed, int):
        raise ValueError("split seed must be an integer")
    if isinstance(min_post_event_steps, bool) or not isinstance(min_post_event_steps, int) or min_post_event_steps < 1:
        raise ValueError("minimum post-event native steps must be a positive integer")
    trajectories = load_dynamics_dataset(directory)
    cases = [_case_report(item, min_post_event_steps) for item in trajectories]
    by_id = {row["case_id"]: row for row in cases}
    reasons = []
    try:
        splits = split_dynamics_cases(trajectories, seed=seed, overfit=overfit, event_stratified=event_stratified)
    except ValueError as error:
        splits = dict(train=[], validation=[], test=[])
        reasons.append(dict(code="split_unavailable", detail=str(error)))
    summaries = {name: _summary([by_id[item.case_id] for item in rows]) for name, rows in splits.items()}
    case_overlap, condition_overlap = [], []
    for first, second in combinations(splits, 2):
        overlap = sorted({item.case_id for item in splits[first]} & {item.case_id for item in splits[second]})
        if overlap:
            case_overlap.append(dict(splits=[first, second], case_ids=overlap))
        overlap = sorted({item.condition_digest for item in splits[first]} & {item.condition_digest for item in splits[second]})
        if overlap:
            condition_overlap.append(dict(splits=[first, second], condition_digests=overlap))
    if case_overlap or condition_overlap:
        reasons.append(dict(code="split_leakage", detail="a complete case or loading condition occurs in multiple splits"))
    if overfit:
        reasons.append(dict(code="overfit_debug", detail="debug training has no held-out validation or test cases"))
    for name, summary in summaries.items():
        for event in ("fracture", "support_release"):
            for kind in ("positive", "negative"):
                if summary[f"{event}_{kind}_cases"] == 0:
                    reasons.append(dict(code="missing_event_class", split=name, event=event, missing=kind))
            if summary[f"{event}_positive_cases_with_post_event_window"] == 0:
                reasons.append(dict(code="missing_post_event_window", split=name, event=event,
                                    minimum_native_steps=min_post_event_steps))
        if summary["cases_with_unknown_native_clock"]:
            reasons.append(dict(code="unknown_native_clock", split=name,
                                cases=summary["cases_with_unknown_native_clock"]))
        if summary["cases_missing_native_steps"]:
            reasons.append(dict(code="missing_native_steps", split=name,
                                cases=summary["cases_missing_native_steps"]))
    return dict(
        schema="fracture-world.dynamics-dataset-readiness.v1",
        status="ready" if not reasons else "not_ready", ready_for_training=not reasons,
        readiness_scope="minimum event-class and held-out coverage for fracture plus learned support release",
        minimum_post_event_native_steps=min_post_event_steps,
        post_event_window_rule="at least one positive case per event and split continues after its first event",
        split_seed=seed, overfit_debug=overfit, event_stratified=event_stratified,
        split_strategy="overfit_debug" if overfit else "event_stratified_loading_conditions" if event_stratified else "hashed_loading_conditions",
        reasons=reasons, totals=_summary(cases), splits=summaries, cases=cases,
        leakage=dict(split_unit="complete_loading_condition", case_ids_disjoint=not case_overlap,
                     loading_conditions_disjoint=not condition_overlap,
                     case_overlaps=case_overlap, condition_overlaps=condition_overlap),
        provenance=dict(labels="recorded FEMFX numerical teacher trajectories", physical_ground_truth=False,
                        event_labels_used_for_split_only=bool(event_stratified and not overfit),
                        native_engine_calls=0, loading="all validated trajectories are loaded into host memory"),
        limitations=["minimum coverage does not establish dataset size, generalization or physical validity",
                     "original support vertex counts are summed per case; material-corner copies are not independent supports",
                     "release labels may affect only some material copies of an original support vertex"],
    )
