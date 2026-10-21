from dataclasses import dataclass, field
from time import perf_counter
from typing import Callable

import numpy as np
from scipy import sparse
from scipy.linalg import pinvh

from .mesh import VolumeMesh


def _finite_array(value, shape, name):
    result = np.array(value, dtype=np.float64, copy=True)
    if result.shape != shape or not np.isfinite(result).all():
        raise ValueError(f"{name} must be finite with shape {shape}")
    return result


@dataclass(frozen=True)
class LinearizedContact:
    corner_ids: np.ndarray
    weights: np.ndarray
    normal: np.ndarray
    gap: float
    environment_velocity: float = 0.0
    environment: bool = False

    def __post_init__(self):
        indices = np.asarray(self.corner_ids)
        if indices.ndim != 1 or not len(indices) or not np.issubdtype(indices.dtype, np.integer) or (indices < 0).any():
            raise ValueError("contact corner_ids must be a nonempty nonnegative integer vector")
        if len(np.unique(indices)) != len(indices):
            raise ValueError("contact corner_ids must be unique")
        weights = _finite_array(self.weights, indices.shape, "contact weights")
        normal = _finite_array(self.normal, (3,), "contact normal")
        if not np.isclose(np.linalg.norm(normal), 1.0, rtol=1e-10, atol=1e-12):
            raise ValueError("contact normal must have unit length")
        if not np.isfinite([self.gap, self.environment_velocity]).all():
            raise ValueError("contact gap and environment velocity must be finite")
        expected = 1.0 if self.environment else 0.0
        if not np.isclose(weights.sum(), expected, rtol=1e-10, atol=1e-12):
            raise ValueError("internal contact weights must sum to zero; environment weights must sum to one")
        if not self.environment and self.environment_velocity != 0:
            raise ValueError("environment velocity requires an environment contact")
        object.__setattr__(self, "corner_ids", indices.astype(np.int64, copy=True))
        object.__setattr__(self, "weights", weights)
        object.__setattr__(self, "normal", normal)

    @classmethod
    def pair(cls, corner_a, corner_b, normal, gap, environment_velocity=0.0):
        if corner_b == -1:
            return cls(np.array([corner_a]), np.array([1.0]), normal, gap, environment_velocity, True)
        return cls(np.array([corner_a, corner_b]), np.array([1.0, -1.0]), normal, gap, environment_velocity, False)


@dataclass(frozen=True)
class CorrectorTolerances:
    absolute: float = 1e-8
    relative: float = 1e-7

    def __post_init__(self):
        if not np.isfinite([self.absolute, self.relative]).all() or self.absolute <= 0 or self.relative < 0:
            raise ValueError("corrector tolerances must be finite, with absolute > 0 and relative >= 0")


@dataclass
class _Factor:
    name: str
    indices: np.ndarray
    matrix: sparse.csr_matrix
    bound: np.ndarray
    inequality: bool = False
    inverse_gram: np.ndarray = field(init=False, repr=False)

    def __post_init__(self):
        self.inverse_gram = pinvh((self.matrix @ self.matrix.T).toarray())

    def project(self, value):
        residual = self.bound - self.matrix @ value
        if self.inequality:
            residual = np.maximum(residual, 0)
        return value + np.asarray(self.matrix.T @ (self.inverse_gram @ residual)).ravel()

    def violation(self, value):
        residual = np.asarray(self.matrix @ value).ravel() - self.bound
        return np.maximum(-residual, 0) if self.inequality else np.abs(residual)


@dataclass
class DynamicsProblem:
    mesh: VolumeMesh
    positions: np.ndarray
    velocities: np.ndarray
    predicted_positions: np.ndarray
    predicted_velocities: np.ndarray
    dt: float
    density: float = 1.0
    broken_faces: np.ndarray | None = None
    fixed: np.ndarray | None = None
    external_impulse: np.ndarray | None = None
    contacts: tuple[LinearizedContact, ...] = ()
    position_scale: float | None = None
    velocity_scale: float | None = None
    position_weight: float = 1.0
    velocity_weight: float = 1.0
    kinematic_rule: str = "backward_euler"
    corner_masses: np.ndarray = field(init=False)
    prior: np.ndarray = field(init=False, repr=False)
    prior_weights: np.ndarray = field(init=False, repr=False)
    factors: tuple[_Factor, ...] = field(init=False, repr=False)
    metadata: dict = field(init=False)

    def __post_init__(self):
        if not isinstance(self.mesh, VolumeMesh):
            raise TypeError("mesh must be a VolumeMesh")
        if not np.isfinite([self.dt, self.density, self.position_weight, self.velocity_weight]).all():
            raise ValueError("time step, density, and prior weights must be finite")
        if min(self.dt, self.density, self.position_weight, self.velocity_weight) <= 0:
            raise ValueError("time step, density, and prior weights must be positive")
        if self.kinematic_rule not in ("backward_euler", "trapezoidal"):
            raise ValueError("kinematic_rule must be backward_euler or trapezoidal")
        shape = (len(self.mesh.tets), 4, 3)
        for name in ("positions", "velocities", "predicted_positions", "predicted_velocities"):
            setattr(self, name, _finite_array(getattr(self, name), shape, name))
        for name, length in (("broken_faces", len(self.mesh.faces.vertices)), ("fixed", 4 * len(self.mesh.tets))):
            raw = getattr(self, name)
            target_shape = (length,) if name == "broken_faces" else shape[:2]
            if raw is None:
                value = np.zeros(target_shape, dtype=bool)
            else:
                value = np.asarray(raw)
                if value.shape != target_shape or value.dtype != np.dtype(bool):
                    raise ValueError(f"{name} must be a boolean array with shape {target_shape}")
                value = value.copy()
            setattr(self, name, value)
        if self.broken_faces[self.mesh.faces.cells[:, 1] < 0].any():
            raise ValueError("only internal material faces can be broken")
        self.external_impulse = np.zeros(shape) if self.external_impulse is None else _finite_array(self.external_impulse, shape, "external_impulse")
        self.contacts = tuple(self.contacts)
        count = 4 * len(self.mesh.tets)
        if not all(isinstance(contact, LinearizedContact) for contact in self.contacts):
            raise TypeError("contacts must contain LinearizedContact objects")
        if any((contact.corner_ids >= count).any() for contact in self.contacts):
            raise ValueError("contact corner index is outside the material-corner array")
        extent = float(np.linalg.norm(np.ptp(self.mesh.nodes, axis=0)))
        self.position_scale = extent if self.position_scale is None else float(self.position_scale)
        self.velocity_scale = self.position_scale / self.dt if self.velocity_scale is None else float(self.velocity_scale)
        if not np.isfinite([self.position_scale, self.velocity_scale]).all() or min(self.position_scale, self.velocity_scale) <= 0:
            raise ValueError("position and velocity scales must be finite and positive")
        self.corner_masses = np.broadcast_to(self.mesh.volumes[:, None] * self.density / 4, shape[:2]).copy()
        fractions = np.repeat((self.corner_masses / self.corner_masses.sum()).ravel(), 3)
        self.prior_weights = np.r_[self.position_weight * fractions, self.velocity_weight * fractions]
        self.prior = np.r_[((self.predicted_positions - self.positions) / self.position_scale).ravel(), (self.predicted_velocities / self.velocity_scale).ravel()]
        self.factors = self._build_factors()
        closed = not self.fixed.any() and not any(contact.environment for contact in self.contacts)
        self.metadata = {
            "engine_calls": 0,
            "state_representation": "one Lagrangian corner per material tetrahedron vertex",
            "topology": "fixed during each convex corrector solve",
            "kinematic_rule": self.kinematic_rule,
            "contact_geometry": "supplied frozen linearized normal constraints",
            "contact_law": "nonpenetration only; no verified impulse or restitution law",
            "momentum_constraint": "exact global linear and angular momentum with supplied point impulses" if closed else "external reaction inferred; boundary momentum not independently verified",
            "boundary_momentum_verified": closed,
            "angular_momentum_constraint": closed,
            "external_torque_quadrature": "supplied point impulses act at current material-corner positions",
            "position_scale": self.position_scale,
            "velocity_scale": self.velocity_scale,
            "prior_objective": "0.5 * sum(corner_mass / total_mass * (position_weight * normalized_position_correction^2 + velocity_weight * normalized_velocity_correction^2))",
            "intact_faces": int(np.count_nonzero(~self.broken_faces[self.mesh.internal_faces])),
            "broken_faces": int(self.broken_faces.sum()),
            "factor_names": [factor.name for factor in self.factors],
        }

    @property
    def corner_count(self):
        return 4 * len(self.mesh.tets)

    def _factor(self, name, rows, columns, values, bound, inequality=False):
        columns = np.asarray(columns, dtype=np.int64)
        indices, local = np.unique(columns, return_inverse=True)
        matrix = sparse.csr_matrix((values, (rows, local)), shape=(len(bound), len(indices)))
        return _Factor(name, indices, matrix, np.asarray(bound, dtype=float), inequality)

    def _build_factors(self):
        factors = []
        count = self.corner_count
        beta = self.dt * self.velocity_scale / self.position_scale
        if self.kinematic_rule == "trapezoidal":
            beta /= 2
        for corner in range(count):
            columns = np.r_[3 * corner + np.arange(3), 3 * count + 3 * corner + np.arange(3)]
            bound = np.zeros(3) if self.kinematic_rule == "backward_euler" else self.dt * self.velocities.reshape(-1, 3)[corner] / (2 * self.position_scale)
            factors.append(self._factor(f"kinematic:{corner}", np.tile(np.arange(3), 2), columns, np.r_[np.ones(3), -beta * np.ones(3)], bound))
        flat_positions = self.positions.reshape(-1, 3)
        for face in self.mesh.internal_faces:
            if self.broken_faces[face]:
                continue
            cell_a, cell_b = self.mesh.faces.cells[face]
            columns, rows, values, bound = [], [], [], []
            for vertex in self.mesh.faces.vertices[face]:
                corner_a = 4 * cell_a + int(np.flatnonzero(self.mesh.tets[cell_a] == vertex)[0])
                corner_b = 4 * cell_b + int(np.flatnonzero(self.mesh.tets[cell_b] == vertex)[0])
                for offset in (0, 3 * count):
                    for axis in range(3):
                        row = len(bound)
                        rows.extend([row, row])
                        columns.extend([offset + 3 * corner_a + axis, offset + 3 * corner_b + axis])
                        values.extend([1.0, -1.0])
                        bound.append((flat_positions[corner_b, axis] - flat_positions[corner_a, axis]) / self.position_scale if offset == 0 else 0.0)
            factors.append(self._factor(f"intact_face:{face}", rows, columns, values, bound))
        for corner in np.flatnonzero(self.fixed.ravel()):
            columns = np.r_[3 * corner + np.arange(3), 3 * count + 3 * corner + np.arange(3)]
            factors.append(self._factor(f"fixed:{corner}", np.arange(6), columns, np.ones(6), np.zeros(6)))
        for index, contact in enumerate(self.contacts):
            columns = (3 * contact.corner_ids[:, None] + np.arange(3)).ravel()
            values = (contact.weights[:, None] * contact.normal).ravel()
            bound = [(-contact.gap + self.dt * contact.environment_velocity) / self.position_scale]
            factors.append(self._factor(f"contact:{index}", np.zeros(len(columns), dtype=int), columns, values, bound, True))
        if not self.fixed.any() and not any(contact.environment for contact in self.contacts):
            mass = self.corner_masses.ravel()
            rows = np.tile(np.arange(3), count)
            columns = 3 * count + np.arange(3 * count)
            values = np.repeat(mass / mass.sum(), 3)
            target = (np.sum(self.corner_masses[..., None] * self.velocities, axis=(0, 1)) + self.external_impulse.sum(axis=(0, 1))) / (mass.sum() * self.velocity_scale)
            factors.append(self._factor("linear_momentum", rows, columns, values, target))
            origin = np.average(flat_positions, axis=0, weights=mass)
            lever = flat_positions - origin
            angular_length = float(np.linalg.norm(np.ptp(flat_positions, axis=0)))
            if angular_length <= 0:
                raise ValueError("current material corners must have nonzero spatial extent")
            moment_arm = lever.copy()
            if self.kinematic_rule == "trapezoidal":
                moment_arm += self.dt * self.velocities.reshape(-1, 3) / 2
            rows, columns, values = [], [], []
            for corner, arm in enumerate(moment_arm / angular_length):
                cross = np.array([[0, -arm[2], arm[1]], [arm[2], 0, -arm[0]], [-arm[1], arm[0], 0]])
                rows.extend(np.repeat(np.arange(3), 3))
                columns.extend(np.tile(3 * count + 3 * corner + np.arange(3), 3))
                values.extend((mass[corner] / mass.sum() * cross).ravel())
            old_angular = np.sum(np.cross(lever, mass[:, None] * self.velocities.reshape(-1, 3)), axis=0)
            angular_impulse = np.sum(np.cross(lever, self.external_impulse.reshape(-1, 3)), axis=0)
            target = (old_angular + angular_impulse) / (mass.sum() * angular_length * self.velocity_scale)
            factors.append(self._factor("angular_momentum", rows, columns, values, target))
        return tuple(factors)

    def unpack(self, state):
        state = _finite_array(state, self.prior.shape, "corrector state")
        cut = 3 * self.corner_count
        return self.positions + self.position_scale * state[:cut].reshape(self.positions.shape), self.velocity_scale * state[cut:].reshape(self.velocities.shape)

    def objective(self, state):
        difference = np.asarray(state) - self.prior
        return float(0.5 * np.dot(self.prior_weights, difference**2))


@dataclass(frozen=True)
class CorrectorIteration:
    iteration: int
    objective: float
    primal_residual: float
    dual_residual: float
    primal_tolerance: float
    dual_tolerance: float
    constraint_violation: float
    rho: float


@dataclass(frozen=True)
class CorrectorResult:
    method: str
    positions: np.ndarray
    velocities: np.ndarray
    objective: float
    converged: bool
    status: str
    iterations: int
    diagnostics: dict
    history: list[CorrectorIteration]
    runtime_seconds: float


def _limit(value):
    if value is not None and (not isinstance(value, (int, np.integer)) or value < 0):
        raise ValueError("max_iterations must be a nonnegative integer or None")


def _diagnostics(problem, state, tolerance):
    positions, velocities = problem.unpack(state)
    violations = {factor.name: float(np.max(factor.violation(state[factor.indices]), initial=0)) for factor in problem.factors}
    actual_impulse = np.sum(problem.corner_masses[..., None] * (velocities - problem.velocities), axis=(0, 1))
    supplied_impulse = problem.external_impulse.sum(axis=(0, 1))
    residual = actual_impulse - supplied_impulse
    mass = problem.corner_masses.ravel()
    origin = np.average(problem.positions.reshape(-1, 3), axis=0, weights=mass)
    old_lever = problem.positions.reshape(-1, 3) - origin
    old_angular = np.sum(np.cross(old_lever, mass[:, None] * problem.velocities.reshape(-1, 3)), axis=0)
    new_angular = np.sum(np.cross(positions.reshape(-1, 3) - origin, mass[:, None] * velocities.reshape(-1, 3)), axis=0)
    supplied_angular = np.sum(np.cross(old_lever, problem.external_impulse.reshape(-1, 3)), axis=0)
    angular_residual = new_angular - old_angular - supplied_angular
    angular_length = float(np.linalg.norm(np.ptp(problem.positions.reshape(-1, 3), axis=0)))
    bound = tolerance.absolute + tolerance.relative * max(float(np.max(np.abs(state), initial=0)), 1.0)
    violation = max(violations.values(), default=0.0)
    contacts = [float(contact.gap + np.dot(contact.normal, np.sum(contact.weights[:, None] * (positions - problem.positions).reshape(-1, 3)[contact.corner_ids], axis=0)) - problem.dt * contact.environment_velocity) for contact in problem.contacts]
    return {
        "constraint_violation": violation,
        "constraint_tolerance": float(bound),
        "constraints_satisfied": bool(violation <= bound),
        "factor_violations": violations,
        "linearized_contact_gaps": contacts,
        "linear_momentum_change": actual_impulse.tolist(),
        "supplied_external_impulse": supplied_impulse.tolist(),
        "momentum_residual_or_inferred_boundary_impulse": residual.tolist(),
        "angular_momentum_change": (new_angular - old_angular).tolist(),
        "supplied_external_angular_impulse": supplied_angular.tolist(),
        "angular_residual_or_inferred_boundary_angular_impulse": angular_residual.tolist(),
        "angular_momentum_origin": origin.tolist(),
        "closed_system_momentum_constraint": problem.metadata["boundary_momentum_verified"],
        "boundary_momentum_verified": bool(problem.metadata["boundary_momentum_verified"] and np.max(np.abs(residual)) / (problem.corner_masses.sum() * problem.velocity_scale) <= bound),
        "angular_momentum_verified": bool(problem.metadata["angular_momentum_constraint"] and np.max(np.abs(angular_residual)) / (problem.corner_masses.sum() * angular_length * problem.velocity_scale) <= bound),
        "contact_impulse_law_verified": False,
        "engine_calls": 0,
    }


def _result(problem, method, state, tolerance, converged, status, iterations, history, start):
    positions, velocities = problem.unpack(state)
    return CorrectorResult(method, positions, velocities, problem.objective(state), converged, status, iterations, _diagnostics(problem, state, tolerance), history, perf_counter() - start)


def solve_central(problem, tolerances=None, max_iterations=None, cancel=None):
    import clarabel

    start = perf_counter()
    tolerances = tolerances or CorrectorTolerances()
    _limit(max_iterations)
    if cancel is not None and cancel():
        return _result(problem, "central", problem.prior, tolerances, False, "interrupted", 0, [], start)
    if max_iterations == 0:
        return _result(problem, "central", problem.prior, tolerances, False, "maximum_iterations", 0, [], start)
    matrices, bounds, cones = [], [], []
    for factor in problem.factors:
        coo = factor.matrix.tocoo()
        sign = -1.0 if factor.inequality else 1.0
        matrices.append(sparse.csc_matrix((sign * coo.data, (coo.row, factor.indices[coo.col])), shape=(len(factor.bound), len(problem.prior))))
        bounds.append(sign * factor.bound)
        cones.append(clarabel.NonnegativeConeT(len(factor.bound)) if factor.inequality else clarabel.ZeroConeT(len(factor.bound)))
    settings = clarabel.DefaultSettings()
    settings.verbose = False
    settings.tol_feas = max(1e-12, min(tolerances.absolute, tolerances.relative or tolerances.absolute) * 0.01)
    settings.tol_gap_abs = settings.tol_feas
    settings.tol_gap_rel = settings.tol_feas
    if max_iterations is not None:
        settings.max_iter = int(max_iterations)
    iterations = 0
    while True:
        solver = clarabel.DefaultSolver(sparse.diags(problem.prior_weights, format="csc"), -problem.prior_weights * problem.prior, sparse.vstack(matrices, format="csc"), np.concatenate(bounds), cones, settings)
        if cancel is not None:
            solver.set_termination_callback(lambda info: bool(cancel()))
        solution = solver.solve()
        iterations += int(solution.iterations)
        if max_iterations is not None or str(solution.status) != "MaxIterations":
            break
        settings.max_iter *= 2
    state = np.asarray(solution.x, dtype=float) if solution.x is not None else problem.prior.copy()
    if state.shape != problem.prior.shape or not np.isfinite(state).all():
        return _result(problem, "central", problem.prior, tolerances, False, "invalid_solver_output", iterations, [], start)
    diagnostics = _diagnostics(problem, state, tolerances)
    converged = str(solution.status) in ("Solved", "AlmostSolved") and diagnostics["constraints_satisfied"]
    status = "interrupted" if cancel is not None and cancel() else "converged" if converged else str(solution.status)
    converged = converged and status != "interrupted"
    return _result(problem, "central", state, tolerances, converged, status, iterations, [], start)


def solve_consensus(problem, rho=None, tolerances=None, max_iterations=None, cancel: Callable[[], bool] | None = None, adaptive_rho=True):
    start = perf_counter()
    tolerances = tolerances or CorrectorTolerances()
    _limit(max_iterations)
    rho = float(np.median(problem.prior_weights)) if rho is None else float(rho)
    if not np.isfinite(rho) or rho <= 0:
        raise ValueError("rho must be finite and positive")
    state = problem.prior.copy()
    indices = np.concatenate([factor.indices for factor in problem.factors])
    matrix = sparse.block_diag([factor.matrix for factor in problem.factors], format="csr")
    transpose = matrix.T.tocsr()
    inverse_gram = sparse.block_diag([factor.inverse_gram for factor in problem.factors], format="csr")
    bound = np.concatenate([factor.bound for factor in problem.factors])
    inequality = np.concatenate([np.full(len(factor.bound), factor.inequality, dtype=bool) for factor in problem.factors])
    local = state[indices].copy()
    dual = np.zeros_like(local)
    counts = np.bincount(indices, minlength=len(state))
    local_dimension = len(indices)
    history = []
    iteration = 0
    converged = False
    status = "maximum_iterations"
    previous = []
    try:
        while max_iterations is None or iteration < max_iterations:
            if cancel is not None and cancel():
                status = "interrupted"
                break
            old = state.copy()
            local = state[indices] - dual
            projection_residual = bound - matrix @ local
            projection_residual[inequality] = np.maximum(projection_residual[inequality], 0)
            local += transpose @ (inverse_gram @ projection_residual)
            totals = np.bincount(indices, weights=local + dual, minlength=len(state))
            state = (problem.prior_weights * problem.prior + rho * totals) / (problem.prior_weights + rho * counts)
            if not np.isfinite(state).all():
                state = old
                status = "nonfinite_iterate"
                break
            selected = state[indices]
            residual = local - selected
            dual += residual
            primal_squared = float(residual @ residual)
            local_squared = float(local @ local)
            selected_squared = float(selected @ selected)
            dual_squared = float(dual @ dual)
            iteration += 1
            primal = float(np.sqrt(primal_squared))
            dual_residual = float(rho * np.sqrt(np.dot(counts, (state - old)**2)))
            primal_tolerance = float(np.sqrt(local_dimension) * tolerances.absolute + tolerances.relative * max(np.sqrt(local_squared), np.sqrt(selected_squared)))
            dual_tolerance = float(np.sqrt(local_dimension) * tolerances.absolute + tolerances.relative * rho * np.sqrt(dual_squared))
            constraint_residual = matrix @ selected - bound
            constraint_residual[inequality] = np.minimum(constraint_residual[inequality], 0)
            violation = float(np.max(np.abs(constraint_residual), initial=0))
            constraint_tolerance = tolerances.absolute + tolerances.relative * max(float(np.max(np.abs(state), initial=0)), 1.0)
            history.append(CorrectorIteration(iteration, problem.objective(state), primal, dual_residual, primal_tolerance, dual_tolerance, violation, rho))
            if primal <= primal_tolerance and dual_residual <= dual_tolerance and violation <= constraint_tolerance:
                converged, status = True, "converged"
                break
            if adaptive_rho and iteration % 25 == 0:
                multiplier = 2.0 if primal > 10 * dual_residual else 0.5 if dual_residual > 10 * primal else 1.0
                rho *= multiplier
                dual /= multiplier
            snapshot = (state.copy(), dual.copy(), rho)
            if any(snapshot[2] == prior[2] and np.array_equal(snapshot[0], prior[0]) and np.array_equal(snapshot[1], prior[1]) for prior in previous):
                status = "numerical_cycle"
                break
            previous = (previous + [snapshot])[-2:]
    except KeyboardInterrupt:
        status = "interrupted"
    return _result(problem, "consensus", state, tolerances, converged, status, iteration, history, start)
