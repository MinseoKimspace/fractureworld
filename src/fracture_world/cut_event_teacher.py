from dataclasses import asdict
from time import perf_counter

import numpy as np
from scipy.integrate import solve_ivp

from .cases import Case
from .teacher2d import InternalForce, Pull
from .topology import lift_state, lumped_mass, triangle_areas


def elastic_energy(mesh, material, position):
    position = np.asarray(position, dtype=np.float64)
    if position.shape != mesh.rest.shape or not np.isfinite(position).all():
        raise ValueError("Energy evaluation requires finite positions matching the mesh")
    rest = mesh.rest[mesh.triangles]
    displacement = (position - mesh.rest)[mesh.triangles]
    reference = np.stack((rest[:, 1] - rest[:, 0], rest[:, 2] - rest[:, 0]), axis=-1)
    difference = np.stack((displacement[:, 1] - displacement[:, 0],
                           displacement[:, 2] - displacement[:, 0]), axis=-1)
    deformation = np.eye(2) + difference @ np.linalg.inv(reference)
    strain = (deformation.swapaxes(-1, -2) @ deformation - np.eye(2)) / 2
    shear = material.young / (2 * (1 + material.poisson))
    lame = material.young * material.poisson / ((1 + material.poisson) * (1 - 2 * material.poisson))
    density = shear * np.sum(strain ** 2, axis=(1, 2))
    density += lame * np.trace(strain, axis1=1, axis2=2) ** 2 / 2
    return float(np.sum(density * triangle_areas(mesh)) * material.thickness)


def transfer_audit(initial_mesh, final_mesh, material, position, velocity):
    if not np.array_equal(initial_mesh.rest[initial_mesh.triangles],
                          final_mesh.rest[final_mesh.triangles]):
        raise ValueError("A cut must preserve material rest positions and element ordering")
    position = np.asarray(position, dtype=np.float64)
    velocity = np.asarray(velocity, dtype=np.float64)
    if not np.isfinite(position).all() or not np.isfinite(velocity).all():
        raise ValueError("A cut requires finite physical states")
    lifted_position, lifted_velocity = lift_state(initial_mesh, final_mesh, position, velocity)
    before_mass = lumped_mass(initial_mesh, material.density, material.thickness)
    after_mass = lumped_mass(final_mesh, material.density, material.thickness)
    before_linear = before_mass[:, None] * velocity
    after_linear = after_mass[:, None] * lifted_velocity
    before_angular = position[:, 0] * before_linear[:, 1] - position[:, 1] * before_linear[:, 0]
    after_angular = lifted_position[:, 0] * after_linear[:, 1] - lifted_position[:, 1] * after_linear[:, 0]
    before_kinetic = float(np.sum(before_mass[:, None] * velocity ** 2) / 2)
    after_kinetic = float(np.sum(after_mass[:, None] * lifted_velocity ** 2) / 2)
    before_elastic = elastic_energy(initial_mesh, material, position)
    after_elastic = elastic_energy(final_mesh, material, lifted_position)
    differences = {
        "mass_change": float(after_mass.sum() - before_mass.sum()),
        "linear_momentum_change": (after_linear.sum(0) - before_linear.sum(0)).tolist(),
        "angular_momentum_change": float(after_angular.sum() - before_angular.sum()),
        "kinetic_energy_change": after_kinetic - before_kinetic,
        "elastic_energy_change": after_elastic - before_elastic,
        "material_corner_position_max_change": float(np.max(np.abs(
            lifted_position[final_mesh.triangles] - position[initial_mesh.triangles]))),
        "material_corner_velocity_max_change": float(np.max(np.abs(
            lifted_velocity[final_mesh.triangles] - velocity[initial_mesh.triangles]))),
    }
    scales = {
        "mass_change": float(before_mass.sum() + after_mass.sum()),
        "linear_momentum_change": float(np.abs(before_linear).sum() + np.abs(after_linear).sum()),
        "angular_momentum_change": float(np.abs(before_angular).sum() + np.abs(after_angular).sum()),
        "kinetic_energy_change": before_kinetic + after_kinetic,
        "elastic_energy_change": before_elastic + after_elastic,
    }
    limits = {key: float(64 * np.finfo(float).eps * scale) for key, scale in scales.items()}
    checks = {key: bool(np.max(np.abs(value)) <= limits.get(key, 0.0))
              for key, value in differences.items()}
    return {
        "status": "conserved" if all(checks.values()) else "failed",
        "checks": checks, "floating_point_limits": limits,
        "mass_before": float(before_mass.sum()), "mass_after": float(after_mass.sum()),
        "kinetic_energy_before": before_kinetic, "kinetic_energy_after": after_kinetic,
        "elastic_energy_before": before_elastic, "elastic_energy_after": after_elastic,
        "prescribed_impulse": [0.0, 0.0], "prescribed_event_work": 0.0,
        "scope": "Prescribed zero-impulse cut; no fracture-energy or contact law",
        **differences,
    }


def simulate_segment(mesh, material, time, load, initial_position=None, initial_velocity=None,
                     rtol=1e-10, atol=1e-12, max_step=None):
    started = perf_counter()
    time = np.asarray(time, dtype=np.float64)
    if (time.ndim != 1 or len(time) < 2 or not np.isfinite(time).all()
            or time[0] < 0 or not np.all(np.diff(time) > 0)):
        raise ValueError("Segment timestamps must be nonnegative and strictly increasing")
    if not np.isfinite([rtol, atol]).all() or min(rtol, atol) <= 0:
        raise ValueError("Integration tolerances must be positive and finite")
    if max_step is None:
        max_step = float(np.min(np.diff(time)))
    if not np.isfinite(max_step) or max_step <= 0:
        raise ValueError("Maximum solver step must be positive and finite")
    if (initial_position is None) != (initial_velocity is None):
        raise ValueError("Restart requires both initial position and velocity")
    if initial_position is None:
        if time[0] != 0:
            raise ValueError("A nonzero-time segment requires an explicit initial state")
        initial_position, initial_velocity = mesh.rest.copy(), np.zeros_like(mesh.rest)
    initial_position = np.asarray(initial_position, dtype=np.float64)
    initial_velocity = np.asarray(initial_velocity, dtype=np.float64)
    if any(value.shape != mesh.rest.shape or not np.isfinite(value).all()
           for value in (initial_position, initial_velocity)):
        raise ValueError("Initial position and velocity must be finite and match the mesh")
    pull = Pull(**load)
    mask, boundary_position, boundary_velocity = pull.boundary(mesh, time)
    for state, target in ((initial_position, boundary_position[0]),
                          (initial_velocity, boundary_velocity[0])):
        if not np.allclose(state[mask], target[mask], rtol=1e-12, atol=1e-13):
            raise ValueError("Initial boundary state must match the global-time pull law")
    free = ~mask
    free_count = int(free.sum())
    mass = lumped_mass(mesh, material.density, material.thickness)
    mass_coordinates = np.repeat(mass[:, None], 2, axis=1)
    force = InternalForce(mesh, material)
    initial = np.concatenate((initial_position[free], initial_velocity[free], np.zeros(int(mask.sum()))))

    def right_hand_side(current_time, state):
        _, position, velocity = pull.boundary(mesh, [current_time])
        position, velocity = position[0], velocity[0]
        position[free] = state[:free_count]
        velocity[free] = state[free_count:2 * free_count]
        internal = force(position)
        return np.concatenate((velocity[free], (internal / mass_coordinates)[free], internal[mask]))

    solution = solve_ivp(right_hand_side, (float(time[0]), float(time[-1])), initial,
                         method="DOP853", t_eval=time, rtol=rtol, atol=atol, max_step=max_step)
    if not solution.success or solution.y.shape[1] != len(time):
        raise RuntimeError(f"Numerical teacher did not complete: {solution.message}")
    position, velocity = boundary_position.copy(), boundary_velocity.copy()
    position[:, free] = solution.y[:free_count].T
    velocity[:, free] = solution.y[free_count:2 * free_count].T
    position[0], velocity[0] = initial_position.copy(), initial_velocity.copy()
    points = position[:, mesh.triangles]
    first, second = points[:, :, 1] - points[:, :, 0], points[:, :, 2] - points[:, :, 0]
    signed = first[:, :, 0] * second[:, :, 1] - first[:, :, 1] * second[:, :, 0]
    if not np.isfinite(position).all() or not np.isfinite(velocity).all() or np.any(signed <= 0):
        raise RuntimeError("Numerical teacher produced a non-finite or inverted configuration")
    integral = solution.y[2 * free_count:].T
    boundary_impulse = np.zeros((len(time) - 1, *mesh.rest.shape))
    boundary_impulse[:, mask] = (
        (np.diff(boundary_velocity, axis=0) * mass_coordinates[None])[:, mask]
        - np.diff(integral, axis=0)
    )
    internal_force = np.stack([force(current) for current in position])
    return Case({
        "rest": mesh.rest.copy(), "triangles": mesh.triangles.copy(),
        "original_vertex": mesh.original_vertex.copy(), "cut_edges": mesh.cut_edges.copy(),
        "time": time.copy(), "position": position, "velocity": velocity,
        "boundary_mask": mask, "boundary_position": boundary_position,
        "boundary_velocity": boundary_velocity, "boundary_impulse": boundary_impulse,
        "internal_force": internal_force,
    }, {
        "material": material.metadata(), "method_label": "teacher",
        "source_url": "https://github.com/adtzlr/felupe", "source_revision": force.version,
        "runtime_seconds": perf_counter() - started, "pull": asdict(pull),
        "solver": "scipy.integrate.solve_ivp:DOP853", "rtol": rtol, "atol": atol,
        "max_step": max_step, "solver_evaluations": int(solution.nfev),
        "force_evaluations": force.evaluations, "force_runtime_seconds": force.runtime_seconds,
        "initial_time": float(time[0]), "boundary_time_origin": "global",
        "prescribed_cut": "fixed_within_segment", "contact": "excluded",
        "fracture_evolution": "externally_prescribed_event_only", "units": "dimensionless_trial",
    })


def simulate_event(initial_mesh, final_mesh, material, time, event_time, load, settings=None):
    time = np.asarray(time, dtype=np.float64)
    if (time.ndim != 1 or len(time) < 3 or not np.isfinite(time).all()
            or time[0] != 0 or not np.all(np.diff(time) > 0)):
        raise ValueError("Event timestamps must start at zero and strictly increase")
    if not np.isfinite(event_time):
        raise ValueError("Event time must be finite")
    matches = np.flatnonzero(time == event_time)
    if len(matches) != 1 or not 0 < matches[0] < len(time) - 1:
        raise ValueError("Event time must equal one interior output timestamp")
    index = int(matches[0])
    parameters = dict(settings or {})
    if set(parameters) - {"rtol", "atol", "max_step"}:
        raise ValueError("Unknown numerical teacher setting")
    initial_cuts = {tuple(edge) for edge in initial_mesh.cut_edges.tolist()}
    final_cuts = {tuple(edge) for edge in final_mesh.cut_edges.tolist()}
    if not initial_cuts < final_cuts:
        raise ValueError("A cut event must add cuts without removing previous cuts")
    lift_state(initial_mesh, final_mesh, initial_mesh.rest, np.zeros_like(initial_mesh.rest))
    before = simulate_segment(initial_mesh, material, time[:index + 1], load, **parameters)
    position, velocity = before.arrays["position"][-1], before.arrays["velocity"][-1]
    audit = transfer_audit(initial_mesh, final_mesh, material, position, velocity)
    if audit["status"] != "conserved":
        raise RuntimeError("Cut state transfer did not conserve its physical state")
    lifted_position, lifted_velocity = lift_state(initial_mesh, final_mesh, position, velocity)
    after = simulate_segment(final_mesh, material, time[index:], load,
                             lifted_position, lifted_velocity, **parameters)
    audit.update({"time": float(time[index]), "frame": index,
                  "added_cut_edges": sorted(final_cuts - initial_cuts),
                  "nodes_before": len(initial_mesh.rest), "nodes_after": len(final_mesh.rest)})
    return before, after, audit
