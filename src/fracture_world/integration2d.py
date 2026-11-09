from dataclasses import dataclass

import numpy as np
import torch


@dataclass
class VerletResult:
    position: torch.Tensor
    velocity: torch.Tensor
    mean_force: torch.Tensor
    start_force: torch.Tensor
    end_force: torch.Tensor


@dataclass
class ForceCache:
    position: torch.Tensor
    force: torch.Tensor
    graph: object
    model: object
    position_version: int
    force_version: int
    graph_versions: tuple
    parameter_versions: tuple


@dataclass
class StepResult:
    position: torch.Tensor
    velocity: torch.Tensor
    mean_force: torch.Tensor
    cache: ForceCache | None
    start_force: torch.Tensor


def tensor_version(value):
    try:
        return value._version
    except RuntimeError:
        return None


def graph_versions(graph):
    return tuple(
        (key, id(value), tensor_version(value))
        for key, value in sorted(graph.items()) if isinstance(value, torch.Tensor)
    )


def parameter_versions(model):
    return tuple((id(value), tensor_version(value)) for value in model.parameters())


def cached_force(cache, model, graph, position):
    if cache is None:
        return None
    if not isinstance(cache, ForceCache):
        raise ValueError("Force cache must originate from an equilibrium integration step")
    if (
        cache.model is model and cache.graph is graph and cache.position is position
        and cache.position_version == tensor_version(position)
        and cache.force_version == tensor_version(cache.force)
        and cache.graph_versions == graph_versions(graph)
        and cache.parameter_versions == parameter_versions(model)
        and cache.force.dtype == position.dtype and cache.force.device == position.device
    ):
        return cache.force
    return None


def make_cache(model, graph, position, force):
    versions = graph_versions(graph)
    if tensor_version(position) is None or tensor_version(force) is None or any(
        version is None for _, _, version in versions
    ):
        return None
    return ForceCache(
        position, force, graph, model, tensor_version(position), tensor_version(force),
        versions, parameter_versions(model),
    )


def tracked_tensor(value):
    if tensor_version(value) is not None:
        return value
    with torch.inference_mode(False):
        return value.clone()


def velocity_verlet(force_fn, mass, position, velocity, dt, boundary_mask=None,
                    boundary_position_next=None, boundary_velocity_next=None, force_start=None):
    if not np.isfinite(dt) or dt <= 0:
        raise ValueError("Time step must be positive and finite")
    if position.ndim != 2 or position.shape[-1] != 2 or velocity.shape != position.shape:
        raise ValueError("Position and velocity must have matching (nodes, 2) shapes")
    if mass.shape != position.shape[:1] or not bool(torch.isfinite(mass).all()) or bool((mass <= 0).any()):
        raise ValueError("Mass must contain a positive finite value per node")
    if boundary_mask is not None:
        if boundary_mask.shape != position.shape or boundary_mask.dtype != torch.bool:
            raise ValueError("Boundary mask must contain boolean coordinate flags")
        if boundary_position_next is None or boundary_velocity_next is None:
            raise ValueError("Both next boundary states are required")
        if boundary_position_next.shape != position.shape or boundary_velocity_next.shape != position.shape:
            raise ValueError("Boundary states must match the position shape")
    elif boundary_position_next is not None or boundary_velocity_next is not None:
        raise ValueError("Boundary states require a boundary mask")
    start = force_fn(position) if force_start is None else force_start
    if start.shape != position.shape:
        raise ValueError("Internal force must match the position shape")
    half_velocity = velocity + 0.5 * dt * start / mass[:, None]
    next_position = position + dt * half_velocity
    if boundary_mask is not None:
        next_position = torch.where(boundary_mask, boundary_position_next, next_position)
    end = force_fn(next_position)
    if end.shape != position.shape:
        raise ValueError("Internal force must match the position shape")
    next_velocity = half_velocity + 0.5 * dt * end / mass[:, None]
    if boundary_mask is not None:
        next_velocity = torch.where(boundary_mask, boundary_velocity_next, next_velocity)
    return VerletResult(next_position, next_velocity, 0.5 * (start + end), start, end)


def advance(model, graph, position, velocity, boundary_position_next, boundary_velocity_next, dt, cache=None):
    if getattr(model, "architecture", "legacy") == "equilibrium":
        return model.step(
            graph, position, velocity, boundary_position_next, boundary_velocity_next, dt, cache=cache,
        )
    next_position, next_velocity, force = model(
        graph, position, velocity, boundary_position_next, boundary_velocity_next, dt,
    )
    return StepResult(next_position, next_velocity, force, None, force)
