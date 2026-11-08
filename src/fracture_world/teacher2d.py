from dataclasses import asdict, dataclass
from time import perf_counter

import numpy as np
from scipy.integrate import solve_ivp

from .cases import Case
from .topology import lumped_mass


@dataclass(frozen=True)
class Material:
    young: float = 100.0
    poisson: float = 0.25
    density: float = 1.0
    thickness: float = 1.0

    def __post_init__(self):
        values = [self.young, self.poisson, self.density, self.thickness]
        if not np.isfinite(values).all():
            raise ValueError("Material parameters must be finite")
        if min(self.young, self.density, self.thickness) <= 0 or not -1 < self.poisson < 0.5:
            raise ValueError("Material parameters are outside their admissible range")

    def metadata(self):
        return dict(asdict(self), model="StVenantKirchhoff", kinematics="plane_strain", calibrated=False)


@dataclass(frozen=True)
class Pull:
    amplitude: float = 0.02
    ramp_time: float = 0.05

    def __post_init__(self):
        if not np.isfinite([self.amplitude, self.ramp_time]).all() or self.amplitude < 0 or self.ramp_time <= 0:
            raise ValueError("Pull amplitude must be nonnegative and ramp time positive")

    def evaluate(self, time):
        time = np.asarray(time, dtype=np.float64)
        fraction = np.clip(time / self.ramp_time, 0, 1)
        position = self.amplitude * fraction ** 3 * (10 - 15 * fraction + 6 * fraction ** 2)
        moving = (time > 0) & (time < self.ramp_time)
        velocity = np.where(moving, self.amplitude * 30 * fraction ** 2 * (fraction - 1) ** 2 / self.ramp_time, 0)
        return position, velocity

    def boundary(self, mesh, time):
        time = np.atleast_1d(np.asarray(time, dtype=np.float64))
        left = mesh.rest[:, 0] == mesh.rest[:, 0].min()
        right = mesh.rest[:, 0] == mesh.rest[:, 0].max()
        sign = right.astype(float) - left.astype(float)
        mask = np.repeat((left | right)[:, None], 2, axis=1)
        displacement, speed = self.evaluate(time)
        position = np.broadcast_to(mesh.rest, (len(time), *mesh.rest.shape)).copy()
        velocity = np.zeros_like(position)
        position[:, :, 0] += displacement[:, None] * sign[None] / 2
        velocity[:, :, 0] = speed[:, None] * sign[None] / 2
        return mask, position, velocity


class InternalForce:
    def __init__(self, mesh, material):
        import felupe as fem

        native_mesh = fem.Mesh(mesh.rest.copy(), mesh.triangles.copy(), cell_type="triangle")
        region = fem.RegionTriangle(native_mesh)
        self.field = fem.FieldContainer([fem.FieldPlaneStrain(region, values=np.zeros_like(mesh.rest))])
        shear = material.young / (2 * (1 + material.poisson))
        lame = material.young * material.poisson / ((1 + material.poisson) * (1 - 2 * material.poisson))
        constitutive = fem.MaterialStrain(fem.linear_elastic, framework="total-lagrange", **{"μ": shear, "λ": lame})
        self.body = fem.SolidBody(constitutive, self.field)
        self.rest = mesh.rest
        self.thickness = material.thickness
        self.version = fem.__version__
        self.evaluations = 0
        self.runtime_seconds = 0.0

    def __call__(self, position):
        started = perf_counter()
        position = np.asarray(position, dtype=np.float64)
        if position.shape != self.rest.shape or not np.isfinite(position).all():
            raise ValueError("Force evaluation requires finite positions matching the mesh")
        self.field[0].values[:] = position - self.rest
        gradient = self.body.assemble.vector(field=self.field).toarray().reshape(self.rest.shape)
        self.evaluations += 1
        self.runtime_seconds += perf_counter() - started
        return -self.thickness * gradient


def simulate(mesh, material, time, amplitude=0.02, ramp_time=0.05,
             rtol=1e-8, atol=1e-10, max_step=None):
    started = perf_counter()
    time = np.asarray(time, dtype=np.float64)
    if time.ndim != 1 or len(time) < 2 or not np.isfinite(time).all() or time[0] != 0 or not np.all(np.diff(time) > 0):
        raise ValueError("Teacher timestamps must start at zero and strictly increase")
    if not np.isfinite([rtol, atol]).all() or min(rtol, atol) <= 0:
        raise ValueError("Integration tolerances must be positive and finite")
    if max_step is None:
        max_step = float(np.min(np.diff(time)))
    if not np.isfinite(max_step) or max_step <= 0:
        raise ValueError("Maximum solver step must be positive and finite")
    pull = Pull(amplitude, ramp_time)
    mask, boundary_position, boundary_velocity = pull.boundary(mesh, time)
    free = ~mask
    free_count = int(free.sum())
    boundary_count = int(mask.sum())
    mass = lumped_mass(mesh, material.density, material.thickness)
    mass_coordinates = np.repeat(mass[:, None], 2, axis=1)
    force = InternalForce(mesh, material)
    initial = np.concatenate((mesh.rest[free], np.zeros(free_count + boundary_count)))

    def right_hand_side(current_time, state):
        _, position, velocity = pull.boundary(mesh, [current_time])
        position, velocity = position[0], velocity[0]
        position[free] = state[:free_count]
        velocity[free] = state[free_count:2 * free_count]
        internal = force(position)
        return np.concatenate((velocity[free], (internal / mass_coordinates)[free], internal[mask]))

    solution = solve_ivp(right_hand_side, (0.0, float(time[-1])), initial,
                         method="DOP853", t_eval=time, rtol=rtol, atol=atol, max_step=max_step)
    if not solution.success or solution.y.shape[1] != len(time):
        raise RuntimeError(f"Numerical teacher did not complete: {solution.message}")
    position, velocity = boundary_position.copy(), boundary_velocity.copy()
    position[:, free] = solution.y[:free_count].T
    velocity[:, free] = solution.y[free_count:2 * free_count].T
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
    defect = np.diff(position, axis=0) - np.diff(time)[:, None, None] * (velocity[:-1] + velocity[1:]) / 2
    free_defect = defect[:, free]
    return Case({
        "rest": mesh.rest.copy(), "triangles": mesh.triangles.copy(),
        "original_vertex": mesh.original_vertex.copy(), "cut_edges": mesh.cut_edges.copy(),
        "time": time, "position": position, "velocity": velocity,
        "boundary_mask": mask, "boundary_position": boundary_position,
        "boundary_velocity": boundary_velocity, "boundary_impulse": boundary_impulse,
        "internal_force": internal_force,
    }, {
        "material": material.metadata(), "method_label": "teacher",
        "source_url": "https://github.com/adtzlr/felupe", "source_revision": force.version,
        "runtime_seconds": perf_counter() - started,
        "pull": asdict(pull), "solver": "scipy.integrate.solve_ivp:DOP853",
        "rtol": rtol, "atol": atol, "max_step": max_step,
        "solver_evaluations": int(solution.nfev), "prescribed_cut": "initial_mesh",
        "force_evaluations": force.evaluations, "force_runtime_seconds": force.runtime_seconds,
        "trapezoidal_position_defect_max": float(np.max(np.abs(free_defect), initial=0)),
        "trapezoidal_position_defect_rms": float(np.sqrt(np.mean(free_defect ** 2))) if free_defect.size else 0.0,
        "contact": "excluded", "fracture_evolution": "excluded", "units": "dimensionless_trial",
    })


def refinement_error(case, refined):
    if str(case.arrays["case_hash"].item()) != str(refined.arrays["case_hash"].item()):
        raise ValueError("Refinement comparison requires identical prescribed inputs")
    return {
        f"{key}_max_difference": float(np.max(np.abs(case.arrays[key] - refined.arrays[key])))
        for key in ("position", "velocity", "boundary_impulse")
    }
