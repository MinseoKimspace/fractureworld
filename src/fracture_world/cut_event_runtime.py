from time import perf_counter

import numpy as np
import torch

from .cases import Case
from .equilibrium import (
    assemble_force, deformation_gradient, green_strain, symmetric_components,
    symmetric_tensor,
)
from .integration2d import velocity_verlet
from .neural import build_graph
from .teacher2d import Pull
from .topology import lift_state, lumped_mass


METHODS = ("no_cut", "cut_stale", "cut_updated")


def require_stress_model(model):
    if getattr(model, "architecture", None) != "equilibrium" or getattr(model, "method", None) != "stress":
        raise ValueError("Dynamic-cut evaluation requires an equilibrium stress checkpoint")


def validate_context(physical_graph, context_graph):
    if context_graph is None:
        return
    if physical_graph.triangles.shape != context_graph.triangles.shape:
        raise ValueError("Frozen context must retain the same ordered material elements")
    for name in ("rest", "original_vertex", "material"):
        physical = getattr(physical_graph, name)[physical_graph.triangles]
        context = getattr(context_graph, name)[context_graph.triangles]
        if not torch.equal(physical, context):
            raise ValueError("Frozen context must retain the same ordered material corners and properties")
    for name in ("shape_gradient", "area", "thickness"):
        if not torch.equal(getattr(physical_graph, name), getattr(context_graph, name)):
            raise ValueError("Frozen context cannot change physical material elements")


def encoded_force(model, physical_graph, context_graph, position):
    if context_graph is None:
        return model.instantaneous_force(physical_graph, position)
    if position.shape != physical_graph.rest.shape:
        raise ValueError("Position must match the physical graph node array")
    deformation = deformation_gradient(physical_graph, position)
    strain = symmetric_components(green_strain(deformation))
    prediction = model.encode(context_graph, strain)
    second_piola = symmetric_tensor(prediction) * physical_graph.material[
        physical_graph.triangles[:, 0], 0, None, None
    ]
    return assemble_force(physical_graph, deformation @ second_piola)


def context_force(model, physical_graph, context_graph, position):
    require_stress_model(model)
    validate_context(physical_graph, context_graph)
    return encoded_force(model, physical_graph, context_graph, position)


def synchronize(device):
    if torch.device(device).type == "cuda":
        torch.cuda.synchronize(device)


def event_grid(time, event_time):
    time = np.asarray(time, dtype=np.float64)
    if (
        time.ndim != 1 or len(time) < 3 or not np.isfinite(time).all()
        or time[0] != 0 or not np.all(np.diff(time) > 0)
    ):
        raise ValueError("Event timestamps must start at zero and strictly increase")
    if not np.isfinite(event_time):
        raise ValueError("Event time must be finite")
    indices = np.flatnonzero(np.isclose(time, event_time, rtol=1e-12, atol=1e-14))
    if len(indices) != 1 or not 0 < indices[0] < len(time) - 1:
        raise ValueError("Event time must match one interior output timestamp")
    return time, int(indices[0])


def orientation_valid(mesh, position):
    points = np.asarray(position)[mesh.triangles]
    first = points[:, 1] - points[:, 0]
    second = points[:, 2] - points[:, 0]
    return bool(np.all(first[:, 0] * second[:, 1] - first[:, 1] * second[:, 0] > 0))


def make_graph(mesh, material, pull, time, dtype, device):
    mask, _, _ = pull.boundary(mesh, [time])
    return build_graph(
        mesh, lumped_mass(mesh, material["density"], material["thickness"]), mask,
        np.array([material[key] for key in ("young", "poisson", "density")]),
        thickness=material["thickness"], dtype=dtype, device=device,
    )


def segment_rollout(model, mesh, graph, context, material, pull, time, initial_position,
                    initial_velocity, label, device, substeps, force_start=None):
    synchronize(device)
    started = perf_counter()
    dtype = graph.rest.dtype
    mask, boundary_position, boundary_velocity = pull.boundary(mesh, time)
    position = torch.as_tensor(initial_position.copy(), dtype=dtype, device=device)
    velocity = torch.as_tensor(initial_velocity.copy(), dtype=dtype, device=device)
    positions = [initial_position.copy()]
    velocities = [initial_velocity.copy()]
    impulses, forces = [], []
    failure = None
    evaluations = 0
    current_force = force_start
    invalid_substep = False

    def evaluate(state):
        nonlocal evaluations
        evaluations += 1
        value = encoded_force(model, graph, context, state)
        if value.shape != state.shape or not bool(torch.isfinite(value).all()):
            raise FloatingPointError("Non-finite or invalid learned internal force")
        return value

    try:
        if not orientation_valid(mesh, initial_position):
            raise ValueError("Initial configuration contains an inverted triangle")
        if current_force is None:
            current_force = evaluate(position)
        forces.append(current_force.detach().cpu().numpy().astype(np.float64))
        for frame in range(1, len(time)):
            impulse = torch.zeros_like(position)
            dt = float(time[frame] - time[frame - 1]) / substeps
            for substep in range(substeps):
                current_time = time[frame] if substep == substeps - 1 else time[frame - 1] + (substep + 1) * dt
                _, targets, speeds = pull.boundary(mesh, [current_time])
                result = velocity_verlet(
                    evaluate, graph.mass, position, velocity, dt, graph.boundary_mask,
                    torch.as_tensor(targets[0], dtype=dtype, device=device),
                    torch.as_tensor(speeds[0], dtype=dtype, device=device), current_force,
                )
                if not all(bool(torch.isfinite(value).all()) for value in (
                    result.position, result.velocity, result.mean_force, result.end_force,
                )):
                    raise FloatingPointError(f"Non-finite prediction at output frame {frame}")
                points = result.position[graph.triangles]
                first = points[:, 1] - points[:, 0]
                second = points[:, 2] - points[:, 0]
                signed_area = first[:, 0] * second[:, 1] - first[:, 1] * second[:, 0]
                if not bool((signed_area > 0).all()):
                    invalid_substep = True
                    raise ValueError(f"Inverted configuration at output frame {frame}, substep {substep + 1}")
                reaction = graph.mass[:, None] * (result.velocity - velocity) - dt * result.mean_force
                impulse += torch.where(graph.boundary_mask, reaction, 0)
                position, velocity, current_force = result.position, result.velocity, result.end_force
            current_position = position.detach().cpu().numpy().astype(np.float64)
            current_velocity = velocity.detach().cpu().numpy().astype(np.float64)
            positions.append(current_position)
            velocities.append(current_velocity)
            impulses.append(impulse.detach().cpu().numpy().astype(np.float64))
            forces.append(current_force.detach().cpu().numpy().astype(np.float64))
            if not orientation_valid(mesh, current_position):
                raise ValueError(f"Inverted configuration at output frame {frame}")
    except (RuntimeError, ValueError, FloatingPointError, OverflowError) as error:
        failure = f"{type(error).__name__}: {error}"

    synchronize(device)
    runtime = perf_counter() - started
    frames = len(positions)
    completed = failure is None and frames == len(time)
    valid = not invalid_substep and all(orientation_valid(mesh, value) for value in positions)
    status = {
        "status": "completed" if completed else "failed", "failure": failure,
        "requested_steps": len(time) - 1, "completed_steps": frames - 1,
        "requested_horizon": float(time[-1] - time[0]),
        "completed_horizon": float(time[frames - 1] - time[0]),
        "orientation_valid": valid, "runtime_seconds": runtime,
        "invalid_substep_detected": invalid_substep,
        "geometry_scope": "Positive signed area at every executed Verlet substep and saved output; contact and overlap are not evaluated",
        "force_evaluations": evaluations,
        "integrator": "velocity_verlet", "substeps": substeps,
        "runtime_scope": "Segment tensor preparation, learned forces, integration and output transfer",
    }
    if frames < 2:
        return None, status, None
    context_mode = "frozen_pre_cut_encoder_context" if context is not None else "current_physical_topology"
    arrays = {
        "rest": mesh.rest.copy(), "triangles": mesh.triangles.copy(),
        "original_vertex": mesh.original_vertex.copy(), "cut_edges": mesh.cut_edges.copy(),
        "time": time[:frames].copy(), "position": np.stack(positions), "velocity": np.stack(velocities),
        "boundary_mask": mask, "boundary_position": boundary_position[:frames].copy(),
        "boundary_velocity": boundary_velocity[:frames].copy(),
        "boundary_impulse": np.stack(impulses), "internal_force": np.stack(forces),
    }
    metadata = {
        "material": dict(material), "pull": {"amplitude": pull.amplitude, "ramp_time": pull.ramp_time},
        "method_label": label, "architecture": "equilibrium", "model_method": "stress",
        "runtime_seconds": runtime, "status": status["status"],
        "integrator": "velocity_verlet", "substeps": substeps,
        "force_semantics": "instantaneous_position_force", "context_mode": context_mode,
        "external_physics_engine": False, "numerical_corrector": False,
        "state_source": "own_autonomous_prediction", "contact": "excluded",
        "fracture_evolution": "prescribed_single_cut_event", "units": "dimensionless_trial",
    }
    return Case(arrays, metadata), status, current_force


def event_rollout(model, initial_case, final_mesh, time, event_time, device="cpu", substeps=1):
    require_stress_model(model)
    if isinstance(substeps, bool) or not isinstance(substeps, (int, np.integer)) or substeps < 1:
        raise ValueError("Substeps must be a positive integer")
    time, event_index = event_grid(time, event_time)
    event_time = float(time[event_index])
    mesh = initial_case.mesh
    if initial_case.arrays["time"][0] != 0:
        raise ValueError("The source initial state must correspond to time zero")
    initial_position = initial_case.arrays["position"][0].copy()
    initial_velocity = initial_case.arrays["velocity"][0].copy()
    lift_state(mesh, final_mesh, initial_position, initial_velocity)
    if len(final_mesh.rest) <= len(mesh.rest) or np.array_equal(final_mesh.triangles, mesh.triangles):
        raise ValueError("Dynamic-cut evaluation requires a newly separated physical mesh")
    if not {tuple(edge) for edge in mesh.cut_edges.tolist()}.issubset(
        {tuple(edge) for edge in final_mesh.cut_edges.tolist()}
    ):
        raise ValueError("The event must retain all previously separated material edges")
    material = dict(initial_case.metadata["material"])
    pull = Pull(**initial_case.metadata["pull"])
    mask, targets, speeds = pull.boundary(mesh, [time[0]])
    if not np.array_equal(mask, initial_case.arrays["boundary_mask"]):
        raise ValueError("Source boundary mask differs from the prescribed pull law")
    for values, expected in ((initial_position, targets[0]), (initial_velocity, speeds[0])):
        if not np.allclose(values[mask], expected[mask], rtol=1e-6, atol=1e-10):
            raise ValueError("Source initial boundary state differs from the prescribed pull law")
    device = torch.device(device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise ValueError("CUDA was requested but is unavailable")
    parameter = next(model.parameters(), None)
    dtype = torch.float32 if parameter is None else parameter.dtype
    if parameter is not None and (
        parameter.device.type != device.type
        or (device.index is not None and parameter.device.index != device.index)
    ):
        raise ValueError("Model parameters must be loaded on the requested device")
    synchronize(device)
    started = perf_counter()
    graph = make_graph(mesh, material, pull, 0.0, dtype, device)
    updated = make_graph(final_mesh, material, pull, event_time, dtype, device)
    validate_context(updated, graph)
    was_training = model.training
    model.eval()
    afters, statuses = {}, {}
    try:
        with torch.inference_mode():
            before, before_status, before_force = segment_rollout(
                model, mesh, graph, None, material, pull, time[:event_index + 1],
                initial_position, initial_velocity, "neural_before_cut", device, int(substeps),
            )
            if before_status["status"] == "completed":
                previous_position = before.arrays["position"][-1]
                previous_velocity = before.arrays["velocity"][-1]
                cut_position, cut_velocity = lift_state(mesh, final_mesh, previous_position, previous_velocity)
                for label in METHODS:
                    physical_mesh, physical_graph = (mesh, graph) if label == "no_cut" else (final_mesh, updated)
                    context = graph if label == "cut_stale" else None
                    position, velocity = (previous_position, previous_velocity) if label == "no_cut" else (cut_position, cut_velocity)
                    case, details, _ = segment_rollout(
                        model, physical_mesh, physical_graph, context, material, pull,
                        time[event_index:], position, velocity, label, device, int(substeps),
                        before_force if label == "no_cut" else None,
                    )
                    afters[label], statuses[label] = case, details
            else:
                for label in METHODS:
                    afters[label] = None
                    statuses[label] = {
                        "status": "failed", "failure": "Shared pre-event rollout did not complete",
                        "requested_steps": len(time) - event_index - 1, "completed_steps": 0,
                        "requested_horizon": float(time[-1] - event_time), "completed_horizon": 0.0,
                        "orientation_valid": None, "runtime_seconds": 0.0, "force_evaluations": 0,
                    }
    finally:
        model.train(was_training)
    synchronize(device)
    return {
        "before": before, "afters": afters, "before_status": before_status, "statuses": statuses,
        "event_time": event_time, "event_index": event_index, "substeps": int(substeps),
        "shared_prefix_runtime_seconds": before_status["runtime_seconds"],
        "runtime_seconds": perf_counter() - started,
        "runtime_scope": "Graph construction, shared prefix and all three post-event forks; weight loading excluded",
        "post_event_state_source": "own_predicted_pre_event_state",
        "stale_context_scope": "Pre-cut encoder aggregation, nodal features, boundary/cut flags and message edges; new physical elements and masses are retained",
    }
