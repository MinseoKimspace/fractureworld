import json
from pathlib import Path

import numpy as np

from .dynamics_data import load_dynamics_trajectory
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
    frames = []
    for predicted_frame, target_frame in pairs:
        positions, velocities = arrays["positions"][predicted_frame], arrays["velocities"][predicted_frame]
        target_positions, target_velocities = teacher.positions[target_frame], teacher.velocities[target_frame]
        position_rmse = np.sqrt(np.sum(normalized_mass * np.sum((positions - target_positions) ** 2, axis=1)))
        velocity_rmse = np.sqrt(np.sum(normalized_mass * np.sum((velocities - target_velocities) ** 2, axis=1)))
        momentum = np.sum(mass[:, None] * velocities, axis=0)
        target_momentum = np.sum(mass[:, None] * target_velocities, axis=0)
        angular = np.sum(np.cross(positions, mass[:, None] * velocities), axis=0)
        target_angular = np.sum(np.cross(target_positions, mass[:, None] * target_velocities), axis=0)
        kinetic = 0.5 * np.sum(mass * np.sum(velocities**2, axis=1))
        target_kinetic = 0.5 * np.sum(mass * np.sum(target_velocities**2, axis=1))
        predicted_cuts, target_cuts = arrays["broken"][predicted_frame], teacher.broken[target_frame]
        predicted_count = len(np.unique(component_labels(len(teacher.topology.mesh.tets), teacher.topology.mesh.dual_edges, predicted_cuts)))
        target_count = len(np.unique(component_labels(len(teacher.topology.mesh.tets), teacher.topology.mesh.dual_edges, target_cuts)))
        frames.append(dict(rollout_frame=int(predicted_frame), teacher_frame=int(target_frame),
                           time_s=float(teacher.times[target_frame]), rollout_time_s=float(arrays["times"][predicted_frame]),
                           position_mass_weighted_rmse_m=float(position_rmse), velocity_mass_weighted_rmse_m_per_s=float(velocity_rmse),
                           fracture=_classification(predicted_cuts, target_cuts),
                           predicted_components=int(predicted_count), teacher_components=int(target_count),
                           component_count_absolute_error=int(abs(predicted_count - target_count)),
                           linear_momentum_error_kg_m_per_s=float(np.linalg.norm(momentum - target_momentum)),
                           angular_momentum_about_origin_error_kg_m2_per_s=float(np.linalg.norm(angular - target_angular)),
                           kinetic_energy_absolute_error_j=float(abs(kinetic - target_kinetic)),
                           support_mismatched_corners=int(np.count_nonzero(arrays["fixed"][predicted_frame] != teacher.fixed[target_frame]))))
    predicted_frames, target_frames = pairs.T
    aggregate_fracture = _classification(arrays["broken"][predicted_frames], teacher.broken[target_frames])
    future = frames[1:]
    mean = lambda key: float(np.mean([frame[key] for frame in future])) if future else None
    result = dict(format="fracture-world.dynamics-evaluation.v1", method=report["method"], rollout_status=report["status"],
                  reference="same FEMFX numerical teacher, not physical ground truth", evaluation_only_teacher_future=True,
                  inference_teacher_future_used=False, engine_calls=0, mesh_fingerprint=teacher.topology.fingerprint,
                  case_id=teacher.case_id, condition_digest=teacher.condition_digest,
                  condition_split=report.get("condition_split", "unknown"), overfit_debug=bool(report.get("overfit_debug", False)),
                  held_out_condition=report.get("condition_split") in {"validation", "test", "unseen_condition"} and not report.get("overfit_debug", False),
                  compared_frames=len(frames), compared_future_frames=len(future), unmatched_rollout_frames=len(arrays["times"]) - len(frames),
                  time_alignment="native integer step IDs; float32 accumulated teacher clock and analytic rollout clock",
                  initial_teacher_frame=initial_frame, position_mass_weighted_rmse_m=mean("position_mass_weighted_rmse_m"),
                  velocity_mass_weighted_rmse_m_per_s=mean("velocity_mass_weighted_rmse_m_per_s"),
                  linear_momentum_error_kg_m_per_s=mean("linear_momentum_error_kg_m_per_s"),
                  component_count_absolute_error=mean("component_count_absolute_error"),
                  support_mismatched_corners_total=int(sum(frame["support_mismatched_corners"] for frame in future)),
                  fracture=aggregate_fracture, fracture_timing=_fracture_timing(teacher, arrays["broken"], native_steps, native_clock, initial_frame),
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
