from dataclasses import dataclass, field
from time import perf_counter

import numpy as np
from scipy.optimize import brentq, minimize
from scipy.sparse.linalg import LinearOperator, cg

from .mesh import VolumeMesh


@dataclass
class CorotationalEnergy:
    mesh: VolumeMesh
    density: float
    youngs_modulus: float
    poissons_ratio: float
    gravity: np.ndarray
    masses: np.ndarray = field(init=False, repr=False)
    inverse_rest: np.ndarray = field(init=False, repr=False)
    shear_modulus: float = field(init=False)
    lame_parameter: float = field(init=False)
    _cached_positions: np.ndarray | None = field(default=None, init=False, repr=False)
    _cached_deformation: tuple | None = field(default=None, init=False, repr=False)
    _cached_eigensystem: tuple | None = field(default=None, init=False, repr=False)

    def __post_init__(self):
        if not isinstance(self.mesh, VolumeMesh):
            raise TypeError("energy mesh must be a VolumeMesh")
        values = [self.density, self.youngs_modulus, self.poissons_ratio]
        if not np.isfinite(values).all() or min(self.density, self.youngs_modulus) <= 0 or not -1 < self.poissons_ratio < 0.5:
            raise ValueError("energy material parameters are outside their physical range")
        self.gravity = np.array(self.gravity, dtype=float, copy=True)
        if self.gravity.shape != (3,) or not np.isfinite(self.gravity).all():
            raise ValueError("energy gravity must be a finite three-vector")
        rest = self.mesh.nodes[self.mesh.tets]
        self.inverse_rest = np.linalg.inv((rest[:, 1:] - rest[:, :1]).transpose(0, 2, 1))
        self.masses = np.broadcast_to(self.mesh.volumes[:, None] * self.density / 4, rest.shape[:2]).copy()
        self.shear_modulus = self.youngs_modulus / (2 * (1 + self.poissons_ratio))
        self.lame_parameter = self.youngs_modulus * self.poissons_ratio / ((1 + self.poissons_ratio) * (1 - 2 * self.poissons_ratio))

    def _states(self, positions, velocities):
        shape = (len(self.mesh.tets), 4, 3)
        positions, velocities = np.asarray(positions, dtype=float), np.asarray(velocities, dtype=float)
        if positions.shape not in (shape, (4 * len(self.mesh.tets), 3)) or velocities.shape != positions.shape:
            raise ValueError("energy states require equal material-corner position and velocity arrays")
        if not np.isfinite(positions).all() or not np.isfinite(velocities).all():
            raise ValueError("energy states must be finite")
        return positions.reshape(shape), velocities.reshape(shape)

    def _deformation(self, positions):
        if self._cached_positions is not None and np.array_equal(positions, self._cached_positions):
            return self._cached_deformation
        deformation = (positions[:, 1:] - positions[:, :1]).transpose(0, 2, 1) @ self.inverse_rest
        left, singular, right = np.linalg.svd(deformation)
        orientation = np.linalg.det(left @ right)
        signs = np.ones_like(singular)
        signs[:, -1] = np.where(orientation < 0, -1.0, 1.0)
        rotation = (left * signs[:, None, :]) @ right
        stretch_offset = signs * singular - 1
        self._cached_positions = np.array(positions, dtype=float, copy=True)
        self._cached_deformation = (deformation, rotation, stretch_offset)
        self._cached_eigensystem = None
        return self._cached_deformation

    def components(self, positions, velocities):
        positions, velocities = self._states(positions, velocities)
        _, _, stretch = self._deformation(positions)
        elastic = np.sum(self.mesh.volumes * (self.shear_modulus * np.sum(stretch**2, axis=1) + 0.5 * self.lame_parameter * np.sum(stretch, axis=1)**2))
        kinetic = 0.5 * np.sum(self.masses[..., None] * velocities**2)
        gravitational = -np.sum(self.masses[..., None] * positions * self.gravity)
        return {"elastic": float(elastic), "kinetic": float(kinetic), "gravitational": float(gravitational),
                "total": float(elastic + kinetic + gravitational)}

    def value(self, positions, velocities):
        return self.components(positions, velocities)["total"]

    def balance(self, previous_positions, previous_velocities, positions, velocities, velocity_scale, *, budget=None,
                energy_scale=None, absolute_tolerance=1e-8, relative_tolerance=1e-7, reference_components=None):
        previous_positions, previous_velocities = self._states(previous_positions, previous_velocities)
        positions, velocities = self._states(positions, velocities)
        values = [velocity_scale, absolute_tolerance, relative_tolerance]
        if not np.isfinite(values).all() or min(velocity_scale, absolute_tolerance) <= 0 or relative_tolerance < 0:
            raise ValueError("energy balance requires positive finite scales and nonnegative tolerances")
        reference = self.components(previous_positions, previous_velocities) if reference_components is None else reference_components
        components = self.components(positions, velocities)
        budget = reference["total"] if budget is None else float(budget)
        if energy_scale is None:
            energy_scale = max(reference["elastic"] + reference["kinetic"], float(self.masses.sum() * velocity_scale**2), np.finfo(float).tiny)
        if not np.isfinite([budget, energy_scale]).all() or energy_scale <= 0:
            raise ValueError("energy budget and balance scale must be finite with positive scale")
        change = dict(elastic=components["elastic"] - reference["elastic"], kinetic=components["kinetic"] - reference["kinetic"],
                      gravitational=float(-np.sum(self.masses[..., None] * (positions - previous_positions) * self.gravity)))
        change["total"] = float(sum(change.values()))
        allowed_work = budget - reference["total"]
        residual = (change["total"] - allowed_work) / energy_scale
        tolerance = absolute_tolerance + relative_tolerance
        return dict(components=components, budget=float(budget), energy_scale=float(energy_scale), energy_change=change,
                    allowed_external_work=float(allowed_work), normalized_violation=float(max(residual, 0.0)),
                    normalized_residual=float(residual), normalized_tolerance=float(tolerance), energy_satisfied=bool(residual <= tolerance))

    def gradient(self, positions, velocities):
        original_shape = np.shape(positions)
        positions, velocities = self._states(positions, velocities)
        deformation, rotation, stretch = self._deformation(positions)
        stress = 2 * self.shear_modulus * (deformation - rotation) + self.lame_parameter * np.sum(stretch, axis=1)[:, None, None] * rotation
        edge_gradient = self.mesh.volumes[:, None, None] * (stress @ self.inverse_rest.transpose(0, 2, 1))
        position_gradient = np.empty_like(positions)
        position_gradient[:, 1:] = edge_gradient.transpose(0, 2, 1)
        position_gradient[:, 0] = -position_gradient[:, 1:].sum(axis=1)
        position_gradient -= self.masses[..., None] * self.gravity
        velocity_gradient = self.masses[..., None] * velocities
        return position_gradient.reshape(original_shape), velocity_gradient.reshape(original_shape)

    def hessian_vector(self, positions, velocities, position_direction, velocity_direction):
        original_shape = np.shape(positions)
        positions, velocities = self._states(positions, velocities)
        position_direction, velocity_direction = self._states(position_direction, velocity_direction)
        deformation, rotation, stretch = self._deformation(positions)
        differential = (position_direction[:, 1:] - position_direction[:, :1]).transpose(0, 2, 1) @ self.inverse_rest
        if self._cached_eigensystem is None:
            symmetric_stretch = rotation.transpose(0, 2, 1) @ deformation
            eigenvalues, eigenvectors = np.linalg.eigh(symmetric_stretch)
            denominator = eigenvalues[:, :, None] + eigenvalues[:, None, :]
            if (np.abs(denominator) <= np.finfo(float).eps).any():
                raise ValueError("polar Hessian is undefined at a singular or orientation-changing deformation")
            self._cached_eigensystem = (eigenvectors, denominator)
        eigenvectors, denominator = self._cached_eigensystem
        basis_differential = eigenvectors.transpose(0, 2, 1) @ rotation.transpose(0, 2, 1) @ differential @ eigenvectors
        spin = (basis_differential - basis_differential.transpose(0, 2, 1)) / denominator
        rotation_differential = rotation @ eigenvectors @ spin @ eigenvectors.transpose(0, 2, 1)
        trace_differential = np.sum(rotation * differential, axis=(1, 2))
        stress_differential = (2 * self.shear_modulus * (differential - rotation_differential) +
                               self.lame_parameter * (trace_differential[:, None, None] * rotation +
                                                      np.sum(stretch, axis=1)[:, None, None] * rotation_differential))
        edge_gradient = self.mesh.volumes[:, None, None] * (stress_differential @ self.inverse_rest.transpose(0, 2, 1))
        position_product = np.empty_like(positions)
        position_product[:, 1:] = edge_gradient.transpose(0, 2, 1)
        position_product[:, 0] = -position_product[:, 1:].sum(axis=1)
        velocity_product = self.masses[..., None] * velocity_direction
        return position_product.reshape(original_shape), velocity_product.reshape(original_shape)


@dataclass(frozen=True)
class EnergyProjectionResult:
    state: np.ndarray
    converged: bool
    status: str
    diagnostics: dict


@dataclass
class EnergyProjection:
    problem: object
    energy: CorotationalEnergy
    budget: float
    absolute_tolerance: float = 1e-8
    relative_tolerance: float = 1e-7
    energy_scale: float | None = None
    stationarity_tolerance: float = 1e-10
    reference_components: dict = field(init=False, repr=False)

    def __post_init__(self):
        values = [self.budget, self.absolute_tolerance, self.relative_tolerance, self.stationarity_tolerance]
        if not np.isfinite(values).all() or min(self.absolute_tolerance, self.stationarity_tolerance) <= 0 or self.relative_tolerance < 0:
            raise ValueError("energy budget and tolerances must be finite and tolerances positive")
        self.reference_components = self.energy.components(self.problem.positions, self.problem.velocities)
        if self.energy_scale is None:
            self.energy_scale = max(self.reference_components["elastic"] + self.reference_components["kinetic"],
                                    float(self.energy.masses.sum() * self.problem.velocity_scale**2), np.finfo(float).tiny)
        if not np.isfinite(self.energy_scale) or self.energy_scale <= 0:
            raise ValueError("energy scale must be finite and positive")
        self.energy_scale = float(self.energy_scale)

    @property
    def tolerance(self):
        return float(self.absolute_tolerance + self.relative_tolerance)

    def state_value(self, state):
        positions, velocities = self.problem.unpack(state)
        return self.energy.balance(self.problem.positions, self.problem.velocities, positions, velocities, self.problem.velocity_scale,
                                   budget=self.budget, energy_scale=self.energy_scale, absolute_tolerance=self.absolute_tolerance,
                                   relative_tolerance=self.relative_tolerance, reference_components=self.reference_components)["normalized_residual"]

    def state_gradient(self, state):
        positions, velocities = self.problem.unpack(state)
        position_gradient, velocity_gradient = self.energy.gradient(positions, velocities)
        return np.r_[self.problem.position_scale * position_gradient.ravel(), self.problem.velocity_scale * velocity_gradient.ravel()] / self.energy_scale

    def state_hessian_vector(self, state, direction):
        positions, velocities = self.problem.unpack(state)
        cut = 3 * self.problem.corner_count
        position_direction = self.problem.position_scale * np.asarray(direction[:cut]).reshape(positions.shape)
        velocity_direction = self.problem.velocity_scale * np.asarray(direction[cut:]).reshape(velocities.shape)
        position_product, velocity_product = self.energy.hessian_vector(positions, velocities, position_direction, velocity_direction)
        return np.r_[self.problem.position_scale * position_product.ravel(), self.problem.velocity_scale * velocity_product.ravel()] / self.energy_scale

    def diagnostics(self, state):
        positions, velocities = self.problem.unpack(state)
        report = self.energy.balance(self.problem.positions, self.problem.velocities, positions, velocities, self.problem.velocity_scale,
                                     budget=self.budget, energy_scale=self.energy_scale, absolute_tolerance=self.absolute_tolerance,
                                     relative_tolerance=self.relative_tolerance, reference_components=self.reference_components)
        report.update(law="corotational elastic + kinetic + gravitational potential",
                      fracture_energy="not calibrated; no artificial energy is credited for fracture", engine_calls=0)
        return report

    def project(self, state, cancel=None):
        start = perf_counter()
        state = np.array(state, dtype=float, copy=True)
        if state.shape != self.problem.prior.shape or not np.isfinite(state).all():
            raise ValueError("energy projection state must match the finite normalized corrector state")
        calls, iterations, evaluations = 0, 0, 0
        cache = {}
        terminations = []

        def result(value, converged, status, **extra):
            report = self.diagnostics(value)
            report.update(method="local nonconvex L-BFGS-B Lagrangian with scalar multiplier search",
                          global_projection_verified=False, optimizer_calls=calls, optimizer_iterations=iterations,
                          energy_evaluations=evaluations, optimizer_terminations=terminations,
                          runtime_seconds=perf_counter() - start)
            report.update(extra)
            return EnergyProjectionResult(value.copy(), converged, status, report)

        if cancel is not None and cancel():
            return result(state, False, "interrupted")
        if not self.energy.gravity.any() and self.budget < 0:
            return result(state, False, "negative_energy_budget_infeasible", certificate="stable elastic and kinetic energies are nonnegative without gravity")
        if self.state_value(state) <= 0:
            return result(state, True, "already_feasible", multiplier=0.0, stationarity_residual=0.0)
        if not self.energy.gravity.any() and self.budget == 0:
            positions, _ = self.problem.unpack(state)
            rest = self.energy.mesh.nodes[self.energy.mesh.tets]
            source = rest - rest.mean(axis=1, keepdims=True)
            target_center = positions.mean(axis=1, keepdims=True)
            covariance = source.transpose(0, 2, 1) @ (positions - target_center)
            left, _, right = np.linalg.svd(covariance)
            signs = np.ones((len(rest), 3))
            signs[:, -1] = np.where(np.linalg.det(left @ right) < 0, -1.0, 1.0)
            rotation = (left * signs[:, None, :]) @ right
            projected_positions = source @ rotation + target_center
            projected = np.r_[((projected_positions - self.problem.positions) / self.problem.position_scale).ravel(),
                              np.zeros(3 * self.problem.corner_count)]
            verified = self.state_value(projected) <= self.tolerance
            return result(projected, verified, "exact_zero_energy_projection" if verified else "energy_feasibility_unknown",
                          method="per-tetrahedron proper Kabsch rigid projection with zero velocity",
                          global_projection_verified=verified, multiplier=None, stationarity_residual=0.0)
        target = 0.5 * self.tolerance
        if self.state_value(state) <= target:
            return result(state, True, "already_feasible", multiplier=0.0, stationarity_residual=0.0,
                          proximal_set_normalized_slack=target)

        def solve(multiplier):
            nonlocal calls, iterations, evaluations
            if multiplier in cache:
                return cache[multiplier]
            if cancel is not None and cancel():
                raise InterruptedError("energy projection interrupted")
            initial = min(cache.items(), key=lambda item: abs(item[0] - multiplier))[1][0] if cache else state

            def objective(value):
                nonlocal evaluations
                evaluations += 1
                displacement = value - state
                return 0.5 * float(displacement @ displacement) + multiplier * self.state_value(value), displacement + multiplier * self.state_gradient(value)

            solution = minimize(objective, initial, method="L-BFGS-B", jac=True,
                                options={"gtol": self.stationarity_tolerance * 0.01, "ftol": 0.0})
            calls += 1
            iterations += int(solution.nit)
            if not np.isfinite(solution.x).all() or not np.isfinite(solution.jac).all():
                raise RuntimeError("nonfinite energy optimizer output")
            stationarity = float(np.max(np.abs(solution.jac), initial=0))
            threshold = self.stationarity_tolerance * max(float(np.max(np.abs(solution.x), initial=0)), 1.0)
            terminations.append(dict(library_success=bool(solution.success), message=str(solution.message),
                                     stationarity_residual=stationarity, stationarity_tolerance=threshold))
            if stationarity > threshold:
                refined = minimize(objective, solution.x, method="Newton-CG", jac=True,
                                   hessp=lambda value, direction: direction + multiplier * self.state_hessian_vector(value, direction),
                                   options={"xtol": 1e-12})
                calls += 1
                iterations += int(refined.nit)
                if not np.isfinite(refined.x).all() or not np.isfinite(refined.jac).all():
                    raise RuntimeError("nonfinite energy optimizer refinement output")
                refined_stationarity = float(np.max(np.abs(refined.jac), initial=0))
                refined_threshold = self.stationarity_tolerance * max(float(np.max(np.abs(refined.x), initial=0)), 1.0)
                terminations.append(dict(library_success=bool(refined.success), message=str(refined.message),
                                         stationarity_residual=refined_stationarity, stationarity_tolerance=refined_threshold))
                if np.isfinite(refined.x).all() and np.isfinite(refined.jac).all() and refined_stationarity < stationarity:
                    solution, stationarity = refined, refined_stationarity
                    threshold = refined_threshold
            while stationarity > threshold:
                operator = LinearOperator((len(state), len(state)),
                                          matvec=lambda direction: direction + multiplier * self.state_hessian_vector(solution.x, direction))
                step, cg_status = cg(operator, -solution.jac, rtol=1e-10, atol=0.0)
                calls += 1
                candidate = solution.x + step
                if cg_status != 0 or not np.isfinite(candidate).all() or np.array_equal(candidate, solution.x):
                    terminations.append(dict(method="linear-memory Newton refinement", library_success=False,
                                             reason="linear solve failure or unresolved floating point step", linear_status=int(cg_status)))
                    break
                candidate_value, candidate_gradient = objective(candidate)
                if not np.isfinite(candidate_value) or not np.isfinite(candidate_gradient).all():
                    raise RuntimeError("nonfinite energy Newton refinement output")
                candidate_stationarity = float(np.max(np.abs(candidate_gradient), initial=0))
                candidate_energy = self.diagnostics(candidate)
                component_scale = (abs(candidate_energy["components"]["elastic"]) + abs(candidate_energy["components"]["kinetic"]) +
                                   abs(self.reference_components["elastic"]) + abs(self.reference_components["kinetic"]) +
                                   abs(candidate_energy["energy_change"]["gravitational"]) + abs(candidate_energy["allowed_external_work"]))
                objective_scale = 0.5 * float(np.sum((candidate - state)**2)) + abs(multiplier) * component_scale / self.energy_scale
                roundoff = 128 * np.finfo(float).eps * max(abs(float(solution.fun)), objective_scale, 1.0)
                if candidate_value > solution.fun + roundoff or stationarity - candidate_stationarity <= np.finfo(float).eps * max(stationarity, 1.0):
                    terminations.append(dict(method="linear-memory Newton refinement", library_success=False,
                                             reason="unresolved objective or stationarity improvement", objective_change=float(candidate_value - solution.fun),
                                             objective_roundoff=float(roundoff), stationarity_residual=candidate_stationarity))
                    break
                solution.x, solution.fun, solution.jac = candidate, candidate_value, candidate_gradient
                stationarity = candidate_stationarity
                threshold = self.stationarity_tolerance * max(float(np.max(np.abs(candidate), initial=0)), 1.0)
                terminations.append(dict(method="linear-memory Newton refinement", library_success=True,
                                         stationarity_residual=stationarity, stationarity_tolerance=threshold))
            if stationarity > threshold:
                raise RuntimeError(f"energy local optimizer unknown: {solution.message}; stationarity={stationarity:g}")
            value = self.state_value(solution.x) - target
            cache[multiplier] = (solution.x.copy(), value, stationarity)
            return cache[multiplier]

        try:
            gradient = self.state_gradient(state)
            squared = float(gradient @ gradient)
            if squared <= np.finfo(float).tiny:
                return result(state, False, "energy_stationary_infeasible")
            upper = max((self.state_value(state) - target) / squared, np.finfo(float).eps)
            previous = None
            while True:
                projected, residual, stationarity = solve(upper)
                if residual <= 0:
                    break
                if previous is not None and np.array_equal(projected, previous):
                    return result(projected, False, "energy_multiplier_stagnation", multiplier=upper, stationarity_residual=stationarity)
                previous = projected.copy()
                upper *= 2
                if not np.isfinite(upper):
                    return result(projected, False, "energy_multiplier_overflow", stationarity_residual=stationarity)

            def residual(multiplier):
                if multiplier == 0:
                    return self.state_value(state) - target
                value, error, _ = solve(multiplier)
                gradient_norm = float(np.linalg.norm(self.state_gradient(value)))
                root_distance = abs(error) / max(gradient_norm, np.finfo(float).tiny)
                distance_tolerance = np.sqrt(len(state)) * self.stationarity_tolerance * max(float(np.max(np.abs(value), initial=0)), 1.0)
                return 0.0 if root_distance <= distance_tolerance else error

            multiplier = brentq(residual, 0.0, upper, xtol=max(np.finfo(float).eps, self.stationarity_tolerance * 0.01 * max(upper, 1.0)),
                                rtol=4 * np.finfo(float).eps)
            projected, _, stationarity = solve(multiplier)
            root_distance = abs(self.state_value(projected) - target) / max(float(np.linalg.norm(self.state_gradient(projected))), np.finfo(float).tiny)
            distance_tolerance = np.sqrt(len(state)) * self.stationarity_tolerance * max(float(np.max(np.abs(projected), initial=0)), 1.0)
            verified = self.state_value(projected) <= self.tolerance and root_distance <= distance_tolerance
            return result(projected, verified, "local_projection" if verified else "energy_feasibility_unknown",
                          multiplier=float(multiplier), stationarity_residual=stationarity,
                          root_normal_distance=root_distance, root_normal_distance_tolerance=distance_tolerance,
                          root_distance_norm="L2 normal-distance estimate; tolerance is sqrt(state dimension) times the per-coordinate accuracy",
                          proximal_set_normalized_slack=target)
        except InterruptedError:
            return result(state, False, "interrupted")
        except (RuntimeError, ValueError, np.linalg.LinAlgError, FloatingPointError) as error:
            return result(state, False, "energy_projection_unknown", solver_message=str(error))
