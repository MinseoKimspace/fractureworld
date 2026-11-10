from time import perf_counter

import numpy as np
import torch

from .benchmark2d import synchronize
from .metrics2d import weighted_rmse
from .teacher2d import InternalForce
from .training2d import graph_for


METRICS = ("force_rmse", "teacher_force_rms", "force_relative_rmse", "free_acceleration_rmse")


def vector_rms(values):
    values = np.asarray(values, dtype=np.float64)
    scale = float(np.max(np.abs(values), initial=0))
    if scale == 0:
        return 0.0
    with np.errstate(over="ignore", invalid="ignore"):
        result = float(scale * np.sqrt(np.mean(np.sum((values / scale) ** 2, axis=-1))))
    return result if np.isfinite(result) else None


def force_metrics(predicted, truth, mass, free):
    with np.errstate(over="ignore", invalid="ignore", divide="ignore"):
        error = predicted - truth
    if not np.isfinite(error).all():
        return dict.fromkeys(METRICS) | {"normalization_status": "nonfinite_force_error"}
    signal, rmse = vector_rms(truth), vector_rms(error)
    active = np.any(free, axis=1)
    acceleration_rmse = None
    if active.any():
        with np.errstate(over="ignore", invalid="ignore", divide="ignore"):
            acceleration = np.where(free[None], error / mass[None, :, None], 0)
        if np.isfinite(acceleration).all():
            acceleration_rmse = weighted_rmse(acceleration[:, active], mass[active])
    with np.errstate(over="ignore", invalid="ignore", divide="ignore"):
        ratio = None if signal in (None, 0) or rmse is None else float(rmse / signal)
    ratio = ratio if ratio is not None and np.isfinite(ratio) else None
    normalization = "zero_signal" if signal == 0 else "available" if ratio is not None else "unrepresentable_ratio"
    return {
        "force_rmse": rmse, "teacher_force_rms": signal, "force_relative_rmse": ratio,
        "normalization_status": normalization, "free_acceleration_rmse": acceleration_rmse,
    }


def selected_frames(time, samples, checkpoints):
    frames = set(np.linspace(0, len(time) - 1, min(samples, len(time)), dtype=int).tolist())
    points = []
    for target in checkpoints:
        tolerance = 8 * np.spacing(max(1.0, abs(target)))
        reached = bool(time[-1] >= target - tolerance)
        matches = np.flatnonzero(np.abs(time - target) <= tolerance)
        frame = int(matches[0]) if reached and len(matches) else None
        if frame is not None:
            frames.add(frame)
        points.append({
            "requested_time": target, "time": None if frame is None else float(time[frame]),
            "frame": frame, "reached": reached,
            "status": "pending" if frame is not None else "unreached" if not reached else "time_not_on_grid",
        })
    return np.asarray(sorted(frames), dtype=np.int64), points


def instantaneous_force_error(model, reference, material, prediction, device, samples=16,
                              checkpoints=(0.2, 0.5, 1.0)):
    if isinstance(samples, bool) or int(samples) != samples or samples < 1:
        raise ValueError("Force sample count must be a positive integer")
    checkpoints = tuple(float(value) for value in checkpoints)
    if not np.isfinite(checkpoints).all() or any(value <= 0 for value in checkpoints) or len(set(checkpoints)) != len(checkpoints):
        raise ValueError("Force checkpoints must be unique positive finite times")
    positions = np.asarray(prediction["position"], dtype=np.float64)
    if positions.ndim != 3 or positions.shape[1:] != reference.mesh.rest.shape or not len(positions) or not np.isfinite(positions).all():
        raise ValueError("Force diagnostics require finite (frames, nodes, 2) positions")
    if "time" not in prediction and len(positions) > len(reference.arrays["time"]):
        raise ValueError("Native substep states require explicit timestamps")
    time = np.asarray(prediction.get("time", reference.arrays["time"][:len(positions)]), dtype=np.float64)
    if time.shape != (len(positions),) or not np.isfinite(time).all() or not np.all(np.diff(time) > 0):
        raise ValueError("Force diagnostic timestamps must be finite and strictly increasing")
    if time[0] != reference.arrays["time"][0] or time[-1] > reference.arrays["time"][-1] + 8 * np.spacing(max(1.0, abs(time[-1]))):
        raise ValueError("Force diagnostic timestamps must remain within the reference horizon")
    synchronize(device)
    started = perf_counter()
    graph = graph_for(reference, device)
    frames, points = selected_frames(time, int(samples), checkpoints)
    teacher = InternalForce(reference.mesh, material)
    mass, free = reference.mass, ~reference.arrays["boundary_mask"]
    predicted, truth, records = [], [], []
    training = model.training
    model.eval()
    try:
        with torch.inference_mode():
            for frame in frames:
                record = {"frame": int(frame), "time": float(time[frame])}
                try:
                    state = torch.as_tensor(positions[frame], dtype=torch.float32, device=device)
                    if not torch.isfinite(state).all():
                        raise FloatingPointError("State cannot be represented in float32")
                    actual = model.instantaneous_force(graph, state).cpu().numpy().astype(np.float64)
                    expected = np.asarray(teacher(state.cpu().numpy().astype(np.float64)), dtype=np.float64)
                    if actual.shape != reference.mesh.rest.shape or expected.shape != actual.shape:
                        raise ValueError("Force functions must return one vector per material node")
                    if not np.isfinite(actual).all() or not np.isfinite(expected).all():
                        raise FloatingPointError("Non-finite force evaluation")
                    metrics = force_metrics(actual[None], expected[None], mass, free)
                    record.update(metrics, status="available" if metrics["force_rmse"] is not None else "unknown_unrepresentable_error")
                    predicted.append(actual)
                    truth.append(expected)
                except (RuntimeError, ValueError, FloatingPointError, OverflowError) as error:
                    record.update(dict.fromkeys(METRICS), status="unknown_force_evaluation",
                                  normalization_status="unavailable", failure=f"{type(error).__name__}: {error}")
                records.append(record)
    finally:
        model.train(training)
    complete = all(row["status"] == "available" for row in records)
    if complete:
        metrics = force_metrics(np.stack(predicted), np.stack(truth), mass, free)
    else:
        metrics = dict.fromkeys(METRICS) | {"normalization_status": "unavailable"}
    indexed = {row["frame"]: row for row in records}
    for point in points:
        if point["frame"] is not None:
            point.update({key: value for key, value in indexed[point["frame"]].items() if key not in {"frame", "time"}})
        else:
            point.update(dict.fromkeys(METRICS), normalization_status="unavailable")
    synchronize(device)
    return {
        "status": "available" if complete and metrics["force_rmse"] is not None else "unknown_incomplete_force_evaluation",
        "sample_count": len(frames), "valid_sample_count": sum(row["status"] == "available" for row in records),
        "frames": frames.tolist(), **metrics,
        "curves": {key: [row.get(key) for row in records] for key in (
            "frame", "time", *METRICS, "normalization_status", "status",
        )},
        "time_checkpoints": points,
        "runtime_seconds": perf_counter() - started,
        "scope": "Both force functions evaluated at the same float32-rounded positions; teacher is diagnostic only",
        "sampling_scope": "Evenly spaced states plus exact requested checkpoint states; aggregate force RMSE is over sampled states and all nodes",
        "acceleration_scope": "Mass-weighted Euclidean error on free coordinates; absent free coordinates yield null",
    }
