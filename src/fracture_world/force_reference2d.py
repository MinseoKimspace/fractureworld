from time import perf_counter

import numpy as np
import torch

from .audit2d import REFINEMENT_LIMITS
from .integration2d import velocity_verlet
from .teacher2d import InternalForce, Material, Pull, refinement_error, simulate


def reference_settings(time, settings=None):
    step = float(np.min(np.diff(time)))
    values = {
        "rtol": 1e-10, "atol": 1e-12, "max_step": step / 4,
        "refined_rtol": 1e-11, "refined_atol": 1e-13, "refined_max_step": step / 8,
    }
    supplied = dict(settings or {})
    if set(supplied) - set(values):
        raise ValueError("Unknown independent-reference setting")
    values.update(supplied)
    if not np.isfinite(list(values.values())).all() or min(values.values()) <= 0:
        raise ValueError("Independent-reference settings must be positive and finite")
    if values["rtol"] > 1e-10 or values["atol"] > 1e-12 or values["max_step"] > step / 4:
        raise ValueError("Independent-reference settings cannot weaken the default accuracy")
    coarse = {key: values[key] for key in ("rtol", "atol", "max_step")}
    refined = {key: values["refined_" + key] for key in coarse}
    if any(refined[key] >= coarse[key] for key in coarse):
        raise ValueError("Reference refinement must tighten both tolerances and the maximum step")
    return coarse, refined


def checked_time(time):
    time = np.asarray(time, dtype=np.float64)
    if (
        time.ndim != 1 or len(time) < 2 or not np.isfinite(time).all()
        or time[0] != 0 or not np.all(np.diff(time) > 0)
    ):
        raise ValueError("Evaluation timestamps must start at zero and strictly increase")
    return time


def reference_solution(mesh, material, time, load, settings=None):
    started = perf_counter()
    time = checked_time(time)
    load = dict(load)
    Pull(**load)
    coarse_settings, fine_settings = reference_settings(time, settings)
    coarse = simulate(mesh, material, time, **load, **coarse_settings)
    reference = simulate(mesh, material, time, **load, **fine_settings)
    refinement = refinement_error(coarse, reference)
    checks = {
        key: bool(np.isfinite(refinement[key]) and refinement[key] <= limit)
        for key, limit in REFINEMENT_LIMITS.items()
    }
    return reference, {
        "status": "completed",
        "converged": all(checks.values()),
        "limits": dict(REFINEMENT_LIMITS),
        "checks": checks,
        "solver": "scipy.integrate.solve_ivp:DOP853",
        "force_provider": "FElupe StVenantKirchhoff plane strain",
        "precision": "float64",
        "reference_level": "refined",
        "coarse_settings": coarse_settings,
        "refined_settings": fine_settings,
        "refinement": refinement,
        "coarse_force_evaluations": coarse.metadata["force_evaluations"],
        "refined_force_evaluations": reference.metadata["force_evaluations"],
        "coarse_runtime_seconds": coarse.metadata["runtime_seconds"],
        "refined_runtime_seconds": reference.metadata["runtime_seconds"],
        "runtime_seconds": perf_counter() - started,
        "scope": "Independent numerical time integration and refinement using the same material law; not an experimental physical truth",
    }


def material_for(case):
    material = case.metadata["material"]
    if material.get("model", "StVenantKirchhoff") != "StVenantKirchhoff":
        raise ValueError("Teacher force requires the StVenantKirchhoff material law")
    if material.get("kinematics", "plane_strain") != "plane_strain":
        raise ValueError("Teacher force requires plane-strain kinematics")
    return Material(**{key: material[key] for key in ("young", "poisson", "density", "thickness")})


def teacher_force_rollout(case, time=None):
    started = perf_counter()
    time = checked_time(case.arrays["time"] if time is None else time)
    if case.arrays["time"][0] != 0:
        raise ValueError("The source initial state must correspond to time zero")
    pull = Pull(**case.metadata["pull"])
    mesh = case.mesh
    mask, boundary_position, boundary_velocity = pull.boundary(mesh, time)
    if not np.array_equal(mask, case.arrays["boundary_mask"]):
        raise ValueError("Source boundary mask differs from the prescribed pull law")
    for name, target in (("position", boundary_position), ("velocity", boundary_velocity)):
        if not np.array_equal(case.arrays[name][0, mask], target[0, mask]):
            raise ValueError("Source initial boundary state differs from the prescribed pull law")
    force = InternalForce(mesh, material_for(case))
    mass = torch.as_tensor(case.mass, dtype=torch.float64)
    boundary_mask = torch.as_tensor(mask)
    position = torch.as_tensor(case.arrays["position"][0].copy(), dtype=torch.float64)
    velocity = torch.as_tensor(case.arrays["velocity"][0].copy(), dtype=torch.float64)
    positions, velocities = [position.numpy().copy()], [velocity.numpy().copy()]
    impulses, means = [], []
    failure = None

    def evaluate(current):
        return torch.as_tensor(force(current.detach().cpu().numpy()), dtype=torch.float64)

    with torch.inference_mode():
        try:
            current_force = evaluate(position)
            if not torch.isfinite(current_force).all():
                raise FloatingPointError("Non-finite initial internal force")
            for frame, dt in enumerate(np.diff(time), start=1):
                result = velocity_verlet(
                    evaluate, mass, position, velocity, float(dt), boundary_mask,
                    torch.as_tensor(boundary_position[frame]),
                    torch.as_tensor(boundary_velocity[frame]), current_force,
                )
                if not all(torch.isfinite(value).all() for value in (
                    result.position, result.velocity, result.mean_force, result.end_force,
                )):
                    raise FloatingPointError(f"Non-finite teacher-force prediction at frame {frame}")
                impulse = mass[:, None] * (result.velocity - velocity) - float(dt) * result.mean_force
                impulses.append(torch.where(boundary_mask, impulse, 0).cpu().numpy().copy())
                means.append(result.mean_force.cpu().numpy().copy())
                position, velocity, current_force = result.position, result.velocity, result.end_force
                positions.append(position.cpu().numpy().copy())
                velocities.append(velocity.cpu().numpy().copy())
        except (RuntimeError, ValueError, FloatingPointError, OverflowError) as error:
            failure = f"{type(error).__name__}: {error}"
    position = np.stack(positions)
    empty = np.empty((0, len(mesh.rest), 2), dtype=np.float64)
    arrays = {
        "time": time[:len(position)].copy(),
        "position": position, "velocity": np.stack(velocities),
        "boundary_impulse": np.stack(impulses) if impulses else empty.copy(),
        "internal_force": np.stack(means) if means else empty.copy(),
    }
    points = position[:, mesh.triangles]
    first, second = points[:, :, 1] - points[:, :, 0], points[:, :, 2] - points[:, :, 0]
    inverted = first[:, :, 0] * second[:, :, 1] - first[:, :, 1] * second[:, :, 0] <= 0
    completed = len(position) - 1
    return arrays, {
        "status": "completed" if completed == len(time) - 1 and failure is None else "failed",
        "failure": failure,
        "completed_steps": completed,
        "requested_steps": len(time) - 1,
        "completed_horizon": float(time[completed] - time[0]),
        "requested_horizon": float(time[-1] - time[0]),
        "integrator": "velocity_verlet",
        "force_provider": "FElupe StVenantKirchhoff plane strain",
        "force_evaluation_scope": "Own evolving predicted positions; stored teacher force samples are never replayed",
        "force_semantics": "Instantaneous position force; returned internal_force is the interval endpoint mean",
        "precision": "float64",
        "device": "cpu",
        "force_evaluations": force.evaluations,
        "force_runtime_seconds": force.runtime_seconds,
        "orientation_valid": not bool(inverted.any()),
        "frames_with_inversion": int(np.any(inverted, axis=1).sum()),
        "runtime_seconds": perf_counter() - started,
        "scope": "Offline force-oracle comparison; prescribed initial cuts and tensile grips; no contact or fracture evolution",
    }
