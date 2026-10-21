import json
from pathlib import Path

import numpy as np

from .dynamics_data import load_dynamics_trajectory
from .dynamics_energy import CorotationalEnergy
from .io import file_hash, write_json
from .partition import component_labels


def _classification(predicted, target):
    true_positive = int(np.count_nonzero(predicted & target))
    false_positive = int(np.count_nonzero(predicted & ~target))
    false_negative = int(np.count_nonzero(~predicted & target))
    precision = true_positive / (true_positive + false_positive) if true_positive + false_positive else 1.0
    recall = true_positive / (true_positive + false_negative) if true_positive + false_negative else 1.0
    f1 = 2 * true_positive / (2 * true_positive + false_positive + false_negative) if true_positive + false_positive + false_negative else 1.0
    return dict(precision=float(precision), recall=float(recall), f1=float(f1), true_positive=true_positive,
                false_positive=false_positive, false_negative=false_negative,
                target_broken_facets=int(target.sum()), predicted_broken_facets=int(predicted.sum()),
                positive_event_comparison=bool(target.any() or predicted.any()),
                empty_positive_convention="precision=recall=f1=1 when neither prediction nor teacher contains a broken facet")


def _new_fracture_classification(predicted, target, initial):
    predicted = np.asarray(predicted, dtype=bool)
    target = np.asarray(target, dtype=bool)
    initial = np.asarray(initial, dtype=bool)
    result = _classification(predicted & ~initial, target & ~initial)
    result.update(semantics="new broken-facet states after the initial observation; initially broken facets excluded",
                  initially_broken_facets_excluded=int(initial.sum()), compared_future_states=len(predicted))
    if not len(predicted):
        result.update(availability="no_future_frames", precision=None, recall=None, f1=None)
    else:
        result["availability"] = "compared_future_frames"
    return result


def _read_rollout(directory, teacher):
    directory = Path(directory)
    report = json.loads((directory / "report.json").read_text(encoding="utf-8"))
    if report.get("format") != "fracture-world.neural-rollout.v1" or report.get("engine_calls") != 0 or report.get("teacher_future_used") is not False:
        raise ValueError("evaluation requires an engine-free rollout with explicit no-future-input metadata")
    if report.get("mesh_fingerprint") != teacher.topology.fingerprint:
        raise ValueError("rollout material mesh differs from the numerical teacher")
    if report.get("case_id") != teacher.case_id or report.get("condition_digest") != teacher.condition_digest:
        raise ValueError("rollout loading condition differs from the numerical teacher")
    with np.load(directory / "trajectory.npz", allow_pickle=False) as archive:
        required = {"times", "positions", "velocities", "broken", "fixed", "metadata_json"}
        if not required.issubset(archive.files):
            raise ValueError("rollout must record positions, velocities, fracture and support history")
        arrays = {key: np.array(archive[key], copy=True) for key in required}
    embedded = json.loads(str(arrays["metadata_json"].item()))
    if embedded != {key: value for key, value in report.items() if key != "events"}:
        raise ValueError("embedded rollout metadata differs from report.json")
    times, positions, velocities, broken, fixed = (arrays[key] for key in ("times", "positions", "velocities", "broken", "fixed"))
    if times.ndim != 1 or not len(times) or not np.isfinite(times).all() or (np.diff(times) <= 0).any():
        raise ValueError("rollout times must be finite and strictly increasing")
    shape = (len(times), len(teacher.topology.rest_positions), 3)
    if positions.shape != shape or velocities.shape != shape or not np.isfinite(positions).all() or not np.isfinite(velocities).all():
        raise ValueError("rollout positions and velocities have invalid dimensions or nonfinite values")
    if broken.shape != (len(times), len(teacher.topology.face_pairs)) or broken.dtype != np.dtype(bool):
        raise ValueError("rollout fracture history must be Boolean with one value per material facet")
    if fixed.shape != shape[:2] or fixed.dtype != np.dtype(bool):
        raise ValueError("rollout support history must be Boolean with one value per material corner")
    if (np.diff(broken.astype(np.int8), axis=0) < 0).any():
        raise ValueError("rollout fracture history must remain irreversible")
    if (np.diff(fixed.astype(np.int8), axis=0) > 0).any():
        raise ValueError("rollout support history cannot activate new fixed supports")
    if report.get("completed_steps") != len(times) - 1:
        raise ValueError("rollout completed step count differs from recorded states")
    return report, arrays


def _time_alignment(teacher, times):
    context = teacher.metadata["conditioning_context"]
    dt = float(np.float32(context["dt"]))
    native_clock = np.r_[0.0, np.cumsum(np.full(context["steps"], np.float32(dt)), dtype=np.float32).astype(float)]
    epsilon = np.finfo(np.float32).eps
    initial_record = int(np.argmin(np.abs(teacher.times - times[0])))
    if abs(teacher.times[initial_record] - times[0]) > 8 * epsilon * max(dt, abs(float(times[0]))):
        raise ValueError("rollout initial time does not match a recorded teacher state")
    initial_step = int(np.searchsorted(native_clock, teacher.times[initial_record]))
    relative_steps = np.rint((times - times[0]) / dt).astype(np.int64)
    analytic_clock = times[0] + relative_steps * dt
    tolerance = 8 * epsilon * np.maximum(dt, np.abs(analytic_clock))
    if (np.abs(times - analytic_clock) > tolerance).any() or (relative_steps < 0).any() or len(np.unique(relative_steps)) != len(times):
        raise ValueError("rollout times are incompatible with the native base timestep clock")
    native_steps = initial_step + relative_steps
    teacher_steps = np.searchsorted(native_clock, teacher.times)
    recorded = {int(step): frame for frame, step in enumerate(teacher_steps)}
    pairs = [(index, recorded[int(step)]) for index, step in enumerate(native_steps) if int(step) in recorded]
    if not pairs or pairs[0] != (0, initial_record):
        raise ValueError("rollout and teacher have no matching initial clock state")
    return np.asarray(pairs, dtype=np.int64), native_steps, native_clock


def _fracture_timing(teacher, broken, native_steps, native_clock, start_frame):
    initial = teacher.broken[start_frame]
    in_horizon = native_steps < len(native_clock)
    predicted = broken[in_horizon]
    predicted_times = native_clock[native_steps[in_horizon]]
    if not len(predicted):
        return dict(availability="no_overlapping_prediction_horizon")
    final_time = predicted_times[-1]
    teacher_frames = np.flatnonzero((teacher.times >= teacher.times[start_frame]) & (teacher.times <= final_time))
    target = teacher.broken[teacher_frames]
    target_events = target[-1] & ~initial
    predicted_events = predicted[-1] & ~initial
    both = target_events & predicted_events
    errors = []
    for facet in np.flatnonzero(both):
        target_time = teacher.times[teacher_frames[np.argmax(target[:, facet])]]
        predicted_time = predicted_times[np.argmax(predicted[:, facet])]
        errors.append(float(predicted_time - target_time))
    return dict(availability="recorded_first_observed_events", semantics="first observed after a timestep; teacher event time lies in its preceding recorded interval",
                matched_events=int(both.sum()), missed_events=int(np.count_nonzero(target_events & ~predicted_events)),
                spurious_events=int(np.count_nonzero(predicted_events & ~target_events)),
                mean_absolute_observed_time_error_s=float(np.mean(np.abs(errors))) if errors else None,
                signed_observed_time_errors_s=errors)


def _energy_interval(energy, previous_positions, previous_velocities, positions, velocities, previous, velocity_scale=None):
    internal_budget = max(0.0, previous["kinetic"] + previous["elastic"])
    scale = velocity_scale if velocity_scale is not None else np.sqrt(internal_budget / energy.masses.sum())
    if scale > 0:
        balance = energy.balance(previous_positions, previous_velocities, positions, velocities, scale,
                                  reference_components=previous)
        increment = max(0.0, balance["energy_change"]["total"])
        return balance["components"], dict(positive_increment_j=float(increment), budget_scale_j=balance["energy_scale"],
                                           relative_positive_increment=float(increment / balance["energy_scale"]),
                                           energy_change_j=balance["energy_change"])
    components = energy.components(positions, velocities)
    displacement = (positions - previous_positions).reshape(energy.masses.shape + (3,))
    change = {key: components[key] - previous[key] for key in ("elastic", "kinetic")}
    change["gravitational"] = float(-np.sum(energy.masses[..., None] * displacement * energy.gravity))
    change["total"] = sum(change.values())
    return components, dict(positive_increment_j=max(0.0, change["total"]), budget_scale_j=0.0,
                            relative_positive_increment=None, energy_change_j=change)


def evaluate_dynamics(rollout_directory, case_path, trajectory_path=None, output_file=None):
    teacher = load_dynamics_trajectory(case_path, trajectory_path)
    report, arrays = _read_rollout(rollout_directory, teacher)
    pairs, native_steps, native_clock = _time_alignment(teacher, arrays["times"])
    initial_frame = int(pairs[0, 1])
    for key, source in (("positions", teacher.positions), ("velocities", teacher.velocities),
                        ("broken", teacher.broken), ("fixed", teacher.fixed)):
        if not np.array_equal(arrays[key][0], source[initial_frame]):
            raise ValueError(f"rollout initial {key} differs from its selected teacher observation")
    mass = teacher.topology.masses
    normalized_mass = mass / mass.sum()
    initial_broken = teacher.broken[initial_frame]
    energy = CorotationalEnergy(teacher.topology.mesh, teacher.material["rest_density"],
                                teacher.material["youngs_modulus"], teacher.material["poissons_ratio"], teacher.gravity)
    velocity_scale = report.get("energy_velocity_scale")
    if velocity_scale is not None and (isinstance(velocity_scale, bool) or not isinstance(velocity_scale, (int, float))
                                       or not np.isfinite(velocity_scale) or velocity_scale <= 0):
        raise ValueError("energy_velocity_scale must be finite and positive when recorded")
    frames = []
    floor = teacher.metadata.get("conditioning_context", {}).get("floor_height_m")
    for predicted_frame, target_frame in pairs:
        positions, velocities = arrays["positions"][predicted_frame], arrays["velocities"][predicted_frame]
        target_positions, target_velocities = teacher.positions[target_frame], teacher.velocities[target_frame]
        position_rmse = np.sqrt(np.sum(normalized_mass * np.sum((positions - target_positions) ** 2, axis=1)))
        velocity_rmse = np.sqrt(np.sum(normalized_mass * np.sum((velocities - target_velocities) ** 2, axis=1)))
        momentum = np.sum(mass[:, None] * velocities, axis=0)
        target_momentum = np.sum(mass[:, None] * target_velocities, axis=0)
        angular = np.sum(np.cross(positions, mass[:, None] * velocities), axis=0)
        target_angular = np.sum(np.cross(target_positions, mass[:, None] * target_velocities), axis=0)
        if frames:
            previous = frames[-1]
            old_frame, old_target = previous["rollout_frame"], previous["teacher_frame"]
            components, increment = _energy_interval(energy, arrays["positions"][old_frame], arrays["velocities"][old_frame],
                                                     positions, velocities, previous["energy_components_j"], velocity_scale)
            reference_components, reference_increment = _energy_interval(
                energy, teacher.positions[old_target], teacher.velocities[old_target], target_positions, target_velocities,
                previous["reference_energy_components_j"], velocity_scale)
        else:
            components, reference_components = energy.components(positions, velocities), energy.components(target_positions, target_velocities)
            increment = reference_increment = None
        predicted_cuts, target_cuts = arrays["broken"][predicted_frame], teacher.broken[target_frame]
        predicted_count = len(np.unique(component_labels(len(teacher.topology.mesh.tets), teacher.topology.mesh.dual_edges, predicted_cuts)))
        target_count = len(np.unique(component_labels(len(teacher.topology.mesh.tets), teacher.topology.mesh.dual_edges, target_cuts)))
        frames.append(dict(rollout_frame=int(predicted_frame), teacher_frame=int(target_frame),
                           time_s=float(teacher.times[target_frame]), rollout_time_s=float(arrays["times"][predicted_frame]),
                           position_mass_weighted_rmse_m=float(position_rmse), velocity_mass_weighted_rmse_m_per_s=float(velocity_rmse),
                           fracture=_classification(predicted_cuts, target_cuts),
                           new_fracture=_new_fracture_classification(predicted_cuts[None], target_cuts[None], initial_broken)
                           if predicted_frame else None,
                           predicted_components=int(predicted_count), teacher_components=int(target_count),
                           component_count_absolute_error=int(abs(predicted_count - target_count)),
                           linear_momentum_error_kg_m_per_s=float(np.linalg.norm(momentum - target_momentum)),
                           angular_momentum_about_origin_error_kg_m2_per_s=float(np.linalg.norm(angular - target_angular)),
                           energy_components_j=components, reference_energy_components_j=reference_components,
                           energy_increment=increment, reference_energy_increment=reference_increment,
                           **{f"{name}_energy_absolute_error_j": abs(components[key] - reference_components[key])
                              for name, key in (("kinetic", "kinetic"), ("elastic", "elastic"),
                                                ("gravitational", "gravitational"), ("total_mechanical", "total"))},
                           floor_penetration_m=max(0.0, float(floor - positions[:, 1].min())) if floor is not None else 0.0,
                           reference_floor_penetration_m=max(0.0, float(floor - target_positions[:, 1].min())) if floor is not None else 0.0,
                           support_mismatched_corners=int(np.count_nonzero(arrays["fixed"][predicted_frame] != teacher.fixed[target_frame]))))
    predicted_frames, target_frames = pairs.T
    aggregate_fracture = _classification(arrays["broken"][predicted_frames], teacher.broken[target_frames])
    future_predicted = arrays["broken"][predicted_frames[1:]]
    future_target = teacher.broken[target_frames[1:]]
    new_fracture = _new_fracture_classification(future_predicted, future_target, initial_broken)
    final_fracture = _classification(arrays["broken"][predicted_frames[-1]], teacher.broken[target_frames[-1]])
    final_fracture.update(semantics="cumulative fracture state at the last aligned frame, including initially broken facets",
                          availability="future_frame" if len(predicted_frames) > 1 else "observed_initial_only")
    final_new_fracture = _new_fracture_classification(future_predicted[-1:], future_target[-1:], initial_broken)
    future = frames[1:]
    mean = lambda key: float(np.mean([frame[key] for frame in future])) if future else None
    energy_metrics = {}
    for prefix, components_key, increment_key in (("", "energy_components_j", "energy_increment"),
                                                   ("reference_", "reference_energy_components_j", "reference_energy_increment")):
        for name in ("kinetic", "elastic", "gravitational", "total"):
            energy_metrics[f"mean_{prefix}{name}_energy_j"] = float(np.mean([frame[components_key][name] for frame in future])) if future else None
        energy_metrics[f"maximum_{prefix}positive_mechanical_energy_increment_j"] = max(
            (frame[increment_key]["positive_increment_j"] for frame in future), default=None)
        energy_metrics[f"maximum_{prefix}relative_positive_mechanical_energy_increment"] = max(
            (frame[increment_key]["relative_positive_increment"] for frame in future
             if frame[increment_key]["relative_positive_increment"] is not None), default=None)
        energy_metrics[f"{prefix}relative_energy_increment_intervals"] = sum(
            frame[increment_key]["relative_positive_increment"] is not None for frame in future)
    result = dict(format="fracture-world.dynamics-evaluation.v1", method=report["method"], rollout_status=report["status"],
                  reference="same FEMFX numerical teacher, not physical ground truth", evaluation_only_teacher_future=True,
                  inference_teacher_future_used=False, engine_calls=0, mesh_fingerprint=teacher.topology.fingerprint,
                  case_id=teacher.case_id, condition_digest=teacher.condition_digest,
                  condition_split=report.get("condition_split", "unknown"), overfit_debug=bool(report.get("overfit_debug", False)),
                  held_out_shape_family=bool(report.get("held_out_shape_family", False)),
                  mesh_seen_during_training=report.get("mesh_seen_during_training"),
                  generalization_protocol=report.get("generalization_protocol", "loading_condition"),
                  held_out_condition=report.get("condition_split") in {"validation", "test", "unseen_condition"} and not report.get("overfit_debug", False),
                  compared_frames=len(frames), compared_future_frames=len(future), unmatched_rollout_frames=len(arrays["times"]) - len(frames),
                  time_alignment="native integer step IDs; float32 accumulated teacher clock and analytic rollout clock",
                  initial_teacher_frame=initial_frame, position_mass_weighted_rmse_m=mean("position_mass_weighted_rmse_m"),
                  velocity_mass_weighted_rmse_m_per_s=mean("velocity_mass_weighted_rmse_m_per_s"),
                  linear_momentum_error_kg_m_per_s=mean("linear_momentum_error_kg_m_per_s"),
                  angular_momentum_about_origin_error_kg_m2_per_s=mean("angular_momentum_about_origin_error_kg_m2_per_s"),
                  kinetic_energy_absolute_error_j=mean("kinetic_energy_absolute_error_j"),
                  elastic_energy_absolute_error_j=mean("elastic_energy_absolute_error_j"),
                  gravitational_energy_absolute_error_j=mean("gravitational_energy_absolute_error_j"),
                  total_mechanical_energy_absolute_error_j=mean("total_mechanical_energy_absolute_error_j"),
                  mechanical_energy_law="same corotational elastic + kinetic + gravitational energy for prediction and FEMFX numerical reference",
                  mechanical_energy_assumptions="static boundaries; no supplied fracture-energy credit; positive increments are diagnostics, not physical ground truth",
                  energy_interval_semantics="successive aligned recorded states; unrecorded intermediate substeps are not evaluated",
                  relative_energy_scale=("max(previous kinetic + elastic energy, total mass * recorded velocity scale squared)"
                                         if velocity_scale is not None else
                                         "legacy report: previous kinetic + elastic energy; derived velocity scale sqrt(internal budget / mass); zero scale is unknown"),
                  energy_change_semantics="delta elastic + delta kinetic - mass-weighted gravity work from displacement; invariant to potential-energy origin",
                  energy_velocity_scale=velocity_scale,
                  initial_energy_components_j=frames[0]["energy_components_j"],
                  initial_reference_energy_components_j=frames[0]["reference_energy_components_j"],
                  final_energy_components_j=frames[-1]["energy_components_j"],
                  final_reference_energy_components_j=frames[-1]["reference_energy_components_j"],
                  **energy_metrics,
                  maximum_floor_penetration_m=max((frame["floor_penetration_m"] for frame in future), default=None),
                  maximum_reference_floor_penetration_m=max((frame["reference_floor_penetration_m"] for frame in future), default=None),
                  component_count_absolute_error=mean("component_count_absolute_error"),
                  support_mismatched_corners_total=int(sum(frame["support_mismatched_corners"] for frame in future)),
                  fracture=aggregate_fracture,
                  fracture_metric_semantics="legacy cumulative facet states over aligned frames including initial observation",
                  new_fracture=new_fracture, final_fracture=final_fracture, final_new_fracture=final_new_fracture,
                  fracture_timing=_fracture_timing(teacher, arrays["broken"], native_steps, native_clock, initial_frame),
                  source_hashes=dict(rollout_report=file_hash(Path(rollout_directory) / "report.json"),
                                     rollout_trajectory=file_hash(Path(rollout_directory) / "trajectory.npz"),
                                     teacher=teacher.metadata["source_hashes"]), frames=frames)
    if output_file is not None:
        output_file = Path(output_file)
        if output_file.exists():
            raise ValueError("evaluation output already exists")
        output_file.parent.mkdir(parents=True, exist_ok=True)
        write_json(output_file, result)
    return result
