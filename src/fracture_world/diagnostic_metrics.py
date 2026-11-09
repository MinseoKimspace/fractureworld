import numpy as np
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import connected_components

from .topology import mesh_edges


def _trajectory(case, prediction, key):
    value = np.asarray(prediction[key], dtype=np.float64)
    if value.ndim != 3 or value.shape[1:] != case.mesh.rest.shape:
        raise ValueError(f"Prediction {key} must have shape (frames, nodes, 2)")
    if not 2 <= len(value) <= len(case.arrays["time"]) or not np.isfinite(value).all():
        raise ValueError(f"Prediction {key} requires at least two finite frames within the reference horizon")
    if "time" in prediction:
        time = np.asarray(prediction["time"])
        if time.shape != (len(value),) or not np.array_equal(time, case.arrays["time"][:len(value)]):
            raise ValueError("Prediction timestamps do not match the reference prefix")
    return value


def _rms(values):
    values = np.asarray(values, dtype=np.float64)
    scale = float(np.max(np.abs(values), initial=0))
    return 0.0 if scale == 0 else float(scale * np.sqrt(np.mean((values / scale) ** 2)))


def _frame_rms(value, mass, free):
    value = np.where(free[None], value, 0.0)
    denominator = mass[np.any(free, axis=1)].sum()
    scale = np.max(np.abs(value), axis=(1, 2))
    normalized = np.divide(value, scale[:, None, None], out=np.zeros_like(value), where=scale[:, None, None] != 0)
    squared = np.sum(normalized ** 2 * mass[None, :, None], axis=(1, 2)) / denominator
    return scale * np.sqrt(squared)


def _relative(error, signal):
    if signal == 0:
        return None, "zero_signal"
    with np.errstate(over="ignore", divide="ignore", invalid="ignore"):
        value = float(np.divide(error, signal))
    return (value, "available") if np.isfinite(value) else (None, "unrepresentable_ratio")


def free_motion_metrics(case, prediction):
    position = _trajectory(case, prediction, "position")
    velocity = _trajectory(case, prediction, "velocity")
    if len(position) != len(velocity):
        raise ValueError("Prediction position and velocity frame counts differ")
    frames = len(position)
    free = ~case.arrays["boundary_mask"]
    mass = case.mass
    active_mass = float(mass[np.any(free, axis=1)].sum())
    result = {
        "frames_evaluated": frames - 1,
        "free_coordinates": int(free.sum()),
        "free_node_mass": active_mass,
        "normalization_scope": "Teacher displacement from material rest coordinates and teacher velocity, on free coordinates only",
        "rmse_convention": "Mass-weighted Euclidean free-coordinate error; denominator is mass of nodes with at least one free coordinate; initial frame excluded",
        "curves": {"time": case.arrays["time"][1:frames].tolist()},
    }
    for name, value, truth, signal in (
        ("position", position[1:], case.arrays["position"][1:frames], case.arrays["position"][1:frames] - case.mesh.rest[None]),
        ("velocity", velocity[1:], case.arrays["velocity"][1:frames], case.arrays["velocity"][1:frames]),
    ):
        if active_mass == 0:
            errors = signals = [None] * (frames - 1)
            ratios = [None] * (frames - 1)
            error_rms = signal_rms = relative = None
            status = "no_free_coordinates"
        else:
            errors = _frame_rms(value - truth, mass, free)
            signals = _frame_rms(signal, mass, free)
            error_rms, signal_rms = _rms(errors), _rms(signals)
            relative, status = _relative(error_rms, signal_rms)
            ratios = [_relative(error, current_signal)[0] for error, current_signal in zip(errors, signals)]
            errors, signals = errors.tolist(), signals.tolist()
        result[f"{name}_rmse"] = error_rms
        result[f"{name}_reference_rms"] = signal_rms
        result[f"{name}_relative_rmse"] = relative
        result[f"{name}_normalization_status"] = status
        result["curves"][f"{name}_rmse"] = errors
        result["curves"][f"{name}_reference_rms"] = signals
        result["curves"][f"{name}_relative_rmse"] = ratios
    return result


def zero_internal_force(case):
    a = case.arrays
    position = np.empty_like(a["position"], dtype=np.float64)
    velocity = np.empty_like(a["velocity"], dtype=np.float64)
    reaction = np.zeros((len(position) - 1, *case.mesh.rest.shape), dtype=np.float64)
    position[0], velocity[0] = a["position"][0], a["velocity"][0]
    mask = a["boundary_mask"]
    mass = case.mass[:, None]
    for frame, dt in enumerate(np.diff(a["time"]), start=1):
        velocity[frame] = velocity[frame - 1]
        position[frame] = position[frame - 1] + dt * velocity[frame - 1]
        velocity[frame, mask] = a["boundary_velocity"][frame, mask]
        position[frame, mask] = a["boundary_position"][frame, mask]
        reaction[frame - 1, mask] = (
            mass * (velocity[frame] - velocity[frame - 1])
        )[mask]
    return {"position": position, "velocity": velocity,
            "boundary_impulse": reaction, "time": a["time"].copy()}


def _balance_summary(momentum, external, time):
    change = momentum[1:] - momentum[0]
    cumulative = np.cumsum(external, axis=0)
    residual = change - cumulative
    interval_residual = np.diff(momentum, axis=0) - external
    residual_norm = np.linalg.norm(residual, axis=1)
    interval_norm = np.linalg.norm(interval_residual, axis=1)
    change_max = float(np.max(np.linalg.norm(change, axis=1), initial=0))
    external_max = float(np.max(np.linalg.norm(cumulative, axis=1), initial=0))
    maximum = float(np.max(residual_norm, initial=0))
    relative, status = _relative(maximum, max(change_max, external_max))
    return {
        "cumulative_max_norm": maximum,
        "interval_max_norm": float(np.max(interval_norm, initial=0)),
        "momentum_change_max_norm": change_max,
        "external_impulse_max_norm": external_max,
        "relative_cumulative_max": relative,
        "normalization_status": status,
        "curves": {
            "time": time[1:].tolist(),
            "momentum_change": change.tolist(),
            "cumulative_external_impulse": cumulative.tolist(),
            "cumulative_residual": residual.tolist(),
            "cumulative_residual_norm": residual_norm.tolist(),
            "interval_residual_norm": interval_norm.tolist(),
        },
    }


def momentum_balance(case, prediction):
    velocity = _trajectory(case, prediction, "velocity")
    edges = mesh_edges(case.mesh)
    graph = coo_matrix((np.ones(len(edges)), (edges[:, 0], edges[:, 1])),
                       shape=(len(case.mesh.rest), len(case.mesh.rest))).tocsr()
    count, labels = connected_components(graph, directed=False)
    result = {
        "status": "unknown_boundary_impulse", "component_count": int(count),
        "global": None, "components": [],
        "scope": "Momentum change minus recorded boundary impulse; no unrecorded external forces included; no pass tolerance applied",
    }
    if "boundary_impulse" not in prediction:
        return result
    external = np.asarray(prediction["boundary_impulse"], dtype=np.float64)
    if external.shape != (len(velocity) - 1, *case.mesh.rest.shape) or not np.isfinite(external).all():
        raise ValueError("Boundary impulses must be finite and match the predicted intervals")
    mass = case.mass
    time = case.arrays["time"][:len(velocity)]
    momentum = np.sum(velocity * mass[None, :, None], axis=1)
    result["status"] = "available"
    result["global"] = _balance_summary(momentum, external.sum(axis=1), time)
    for component in range(count):
        nodes = labels == component
        values = _balance_summary(
            np.sum(velocity[:, nodes] * mass[None, nodes, None], axis=1),
            external[:, nodes].sum(axis=1), time,
        )
        result["components"].append({
            "component_id": component, "node_count": int(nodes.sum()),
            "mass": float(mass[nodes].sum()), **values,
        })
    return result
