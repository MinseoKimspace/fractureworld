from contextlib import contextmanager
from time import perf_counter

import numpy as np
import torch

from .benchmark2d import synchronize
from .training2d import graph_for


@contextmanager
def evaluation(model):
    training = model.training
    model.eval()
    try:
        with torch.inference_mode():
            yield
    finally:
        model.train(training)


def setup(model, case, device):
    device = torch.device(device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise ValueError("CUDA was requested but is unavailable")
    parameter = next(model.parameters(), None)
    dtype = parameter.dtype if parameter is not None else torch.float32
    if parameter is not None and (
        parameter.device.type != device.type
        or device.index is not None and parameter.device.index != device.index
    ):
        raise ValueError("Model and diagnostic device must match")
    graph = graph_for(case, device)
    graph.apply(lambda value: value.to(dtype=dtype) if value.is_floating_point() else value)
    return graph, device, dtype


def finite_prediction(model, graph, position, velocity, boundary_position, boundary_velocity, dt):
    prediction = model(graph, position, velocity, boundary_position, boundary_velocity, dt)
    if len(prediction) != 3 or any(
        not isinstance(value, torch.Tensor) or value.shape != position.shape for value in prediction
    ):
        raise ValueError("Model must return position, velocity and force arrays matching the graph")
    if not all(torch.isfinite(value).all().item() for value in prediction):
        raise FloatingPointError("Non-finite model prediction")
    return prediction


def teacher_forced(model, case, device="cpu"):
    synchronize(device)
    started = perf_counter()
    graph, device, dtype = setup(model, case, device)
    values = {
        name: torch.as_tensor(case.arrays[name], dtype=dtype, device=device).clone()
        for name in ("position", "velocity", "boundary_position", "boundary_velocity")
    }
    positions, velocities = [values["position"][0]], [values["velocity"][0]]
    forces, impulses = [], []
    failure = None
    with evaluation(model):
        for frame, dt in enumerate(np.diff(case.arrays["time"]), start=1):
            source_position = values["position"][frame - 1]
            source_velocity = values["velocity"][frame - 1]
            try:
                position, velocity, force = finite_prediction(
                    model, graph, source_position, source_velocity,
                    values["boundary_position"][frame], values["boundary_velocity"][frame], float(dt),
                )
            except (RuntimeError, ValueError, FloatingPointError) as error:
                failure = f"One-step prediction failed at frame {frame}: {error}"
                break
            impulse = graph.mass[:, None] * (velocity - source_velocity) - float(dt) * force
            positions.append(position)
            velocities.append(velocity)
            forces.append(force)
            impulses.append(torch.where(graph.boundary_mask, impulse, torch.zeros_like(impulse)))
        empty = np.empty((0, len(graph.rest), 2), dtype=np.float64)
        arrays = {
            "position": torch.stack(positions).cpu().numpy().astype(np.float64),
            "velocity": torch.stack(velocities).cpu().numpy().astype(np.float64),
            "internal_force": torch.stack(forces).cpu().numpy().astype(np.float64) if forces else empty.copy(),
            "boundary_impulse": torch.stack(impulses).cpu().numpy().astype(np.float64) if impulses else empty.copy(),
        }
    arrays["position"][0] = case.arrays["position"][0]
    arrays["velocity"][0] = case.arrays["velocity"][0]
    synchronize(device)
    completed = len(positions) - 1
    requested = len(case.arrays["time"]) - 1
    return arrays, {
        "mode": "teacher_forced_one_step", "autonomous": False,
        "status": "completed" if completed == requested and failure is None else "failed",
        "failure": failure, "completed_steps": completed, "requested_steps": requested,
        "completed_horizon": float(case.arrays["time"][completed] - case.arrays["time"][0]),
        "requested_horizon": float(case.arrays["time"][-1] - case.arrays["time"][0]),
        "runtime_seconds": perf_counter() - started,
        "runtime_scope": "Graph construction, teacher-state preparation, independent next-state predictions and output transfer",
        "scope": "Each prediction starts from the corresponding teacher state; outputs are not an autonomous trajectory",
    }


def maximum_norm(value):
    return float(torch.linalg.vector_norm(value.double(), dim=-1).max().item())


def rest_probe(model, case, device="cpu"):
    synchronize(device)
    started = perf_counter()
    graph, device, dtype = setup(model, case, device)
    position = graph.rest.clone()
    velocity = torch.zeros_like(position)
    boundary_position = graph.rest.clone()
    boundary_velocity = torch.zeros_like(position)
    free = ~graph.boundary_mask
    failure = None
    completed = 0
    initial = {
        "initial_free_acceleration_max_norm": None,
        "initial_free_force_max_norm": None,
        "initial_internal_force_max_norm": None,
        "initial_net_internal_force_norm": None,
        "initial_net_internal_torque": None,
    }
    max_drift, max_speed = 0.0, 0.0
    with evaluation(model):
        for frame, dt in enumerate(np.diff(case.arrays["time"]), start=1):
            try:
                next_position, next_velocity, force = finite_prediction(
                    model, graph, position, velocity, boundary_position, boundary_velocity, float(dt),
                )
            except (RuntimeError, ValueError, FloatingPointError) as error:
                failure = f"Rest prediction failed at frame {frame}: {error}"
                break
            if frame == 1:
                initial["initial_free_acceleration_max_norm"] = maximum_norm(force / graph.mass[:, None] * free)
                initial["initial_free_force_max_norm"] = maximum_norm(force * free)
                initial["initial_internal_force_max_norm"] = maximum_norm(force)
                initial["initial_net_internal_force_norm"] = float(force.double().sum(0).norm().item())
                center = (graph.rest.double() * graph.mass.double()[:, None]).sum(0) / graph.mass.double().sum()
                arm = graph.rest.double() - center
                initial["initial_net_internal_torque"] = float(
                    (arm[:, 0] * force.double()[:, 1] - arm[:, 1] * force.double()[:, 0]).sum().item()
                )
            max_drift = max(max_drift, maximum_norm((next_position - graph.rest) * free))
            max_speed = max(max_speed, maximum_norm(next_velocity * free))
            position, velocity = next_position, next_velocity
            completed += 1
    synchronize(device)
    requested = len(case.arrays["time"]) - 1
    return {
        "mode": "stationary_no_load", **initial,
        "status": "completed" if completed == requested and failure is None else "failed",
        "failure": failure, "completed_steps": completed, "requested_steps": requested,
        "completed_horizon": float(case.arrays["time"][completed] - case.arrays["time"][0]),
        "requested_horizon": float(case.arrays["time"][-1] - case.arrays["time"][0]),
        "free_coordinate_count": int(free.sum().item()),
        "max_free_position_drift": max_drift, "max_free_speed": max_speed,
        "runtime_seconds": perf_counter() - started,
        "scope": "Same cut geometry and grip mask; reference positions, zero initial velocity and stationary grips; no tolerance-based pass decision",
    }
