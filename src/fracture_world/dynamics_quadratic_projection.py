from dataclasses import dataclass
import hashlib
from time import perf_counter

import numpy as np
from scipy import sparse
from scipy.sparse.linalg import splu

from .dynamics_energy import EnergyProjectionResult
from .dynamics_frozen_energy import FrozenQuadraticEnergy, TransportEnergy


@dataclass(frozen=True)
class QuadraticEnergyModel:
    hessian: sparse.csc_matrix
    linear: np.ndarray
    constant: float


def _coefficient_digest(arrays):
    digest = hashlib.sha256()
    for value in arrays:
        array = np.ascontiguousarray(value)
        digest.update(str((array.dtype.str, array.shape)).encode())
        digest.update(array.tobytes())
    return digest.digest()


def quadratic_energy_coefficients(projection):
    energy = projection.energy.energy if isinstance(projection.energy, TransportEnergy) else projection.energy
    if not isinstance(energy, FrozenQuadraticEnergy):
        return None
    problem = projection.problem
    dimension = len(problem.prior)
    corners = energy.masses.size
    state_indices = vars(problem).get("state_indices", np.arange(6 * corners))
    if len(state_indices) != 6 * corners or np.max(state_indices, initial=-1) >= dimension:
        raise ValueError("quadratic energy requires the full or shared affine material-corner coordinates")
    stiffness = energy.stiffness.tocsr()
    hessian_key = (dimension, problem.position_scale, problem.velocity_scale, projection.energy_scale,
                   len(energy.node_masses), _coefficient_digest((state_indices, energy.corner_weights,
                       energy.groups, energy.masses, stiffness.data, stiffness.indices, stiffness.indptr,
                       np.asarray(stiffness.shape))))
    stabilization = projection.energy.stabilization if isinstance(projection.energy, TransportEnergy) else np.empty(0)
    key = (id(problem), id(projection.energy), hessian_key, projection.budget, energy.reference_elastic_energy,
           tuple(sorted(projection.reference_components.items())),
           _coefficient_digest((energy.linear_term, energy.reference_positions, energy.gravity, stabilization,
                                problem.positions, problem.velocities)))
    cached = getattr(projection, "_quadratic_coefficients", None)
    projection._quadratic_coefficients_reused = cached is not None and cached[0] == key
    if cached is None or cached[0] != key:
        shared = getattr(energy, "_quadratic_state_hessian", None)
        projection._quadratic_hessian_reused = shared is not None and shared[0] == hessian_key
        if shared is None or shared[0] != hessian_key:
            rows = np.arange(3 * corners)
            position_map = sparse.csr_matrix((np.full(3 * corners, problem.position_scale),
                                              (rows, state_indices[:3 * corners])), shape=(3 * corners, dimension))
            velocity_map = sparse.csr_matrix((np.full(3 * corners, problem.velocity_scale),
                                              (rows, state_indices[3 * corners:])), shape=(3 * corners, dimension))
            average = sparse.csr_matrix((energy.corner_weights, (energy.groups, np.arange(corners))),
                                         shape=(len(energy.node_masses), corners))
            nodal_map = sparse.kron(average, sparse.eye(3, format="csr"), format="csr") @ position_map
            mass = sparse.diags(np.repeat(energy.masses.ravel(), 3), format="csr")
            hessian = (nodal_map.T @ stiffness @ nodal_map + velocity_map.T @ mass @ velocity_map) / projection.energy_scale
            hessian = (0.5 * (hessian + hessian.T)).tocsc()
            hessian.eliminate_zeros()
            shared = (hessian_key, hessian)
            energy._quadratic_state_hessian = shared
        hessian = shared[1]
        linear = projection.state_gradient(np.zeros(dimension))
        if not np.isfinite(hessian.data).all() or not np.isfinite(linear).all():
            raise ValueError("quadratic energy coefficients must be finite")
        cached = (key, hessian, linear, float(projection.state_value(np.zeros(dimension))))
        projection._quadratic_coefficients = cached
    else:
        projection._quadratic_hessian_reused = True
    return QuadraticEnergyModel(cached[1], cached[2], cached[3])


class _PositiveDefinitenessBoundary(Exception):
    pass


def _positive_factor(matrix, stationarity_tolerance):
    if not np.isfinite(matrix.data).all():
        raise _PositiveDefinitenessBoundary("nonfinite Lagrangian matrix")
    try:
        factor = splu(matrix, permc_spec="MMD_AT_PLUS_A", diag_pivot_thresh=0., options={"SymmetricMode": True})
    except RuntimeError as error:
        raise _PositiveDefinitenessBoundary(str(error)) from error
    pivots = factor.U.diagonal()
    if not np.array_equal(factor.perm_r, factor.perm_c) or not np.isfinite(pivots).all() or (pivots <= 0).any():
        raise _PositiveDefinitenessBoundary("symmetric factorization requires equal permutations and positive pivots")
    difference = factor.U - sparse.diags(pivots) @ factor.L.T
    error = float(np.max(np.abs(difference.data), initial=0.))
    scale = max(float(np.max(np.abs(factor.U.data), initial=0.)), 1.)
    if error > stationarity_tolerance * scale:
        raise _PositiveDefinitenessBoundary("Lagrangian factorization symmetry cannot be certified numerically")
    return factor, dict(equal_row_column_permutations=True, positive_pivots=True,
                         minimum_pivot=float(pivots.min()), maximum_factor_symmetry_error=error,
                         certificate="symmetric-mode unpivoted sparse LU with positive diagonal pivots")


def project_quadratic_energy(projection, state, cancel=None):
    started = perf_counter()
    model = quadratic_energy_coefficients(projection)
    if model is None:
        return None
    state = np.array(state, dtype=float, copy=True)
    if state.shape != projection.problem.prior.shape or not np.isfinite(state).all():
        raise ValueError("quadratic projection requires a finite normalized corrector state")
    target = 0.5 * projection.tolerance
    factorizations, iterations, evaluations, refinements = 0, 0, 0, 0
    identity = sparse.eye(len(state), format="csc")
    last_certificate = None

    def result(value, converged, status, **extra):
        report = projection.diagnostics(value)
        report.update(method="sparse quadratic Lagrangian secular projection",
                      global_projection_verified=bool(converged),
                      global_optimality_scope="numerical positive-definite Lagrangian and original-energy KKT certificate",
                      source_hessian_assumed_psd=False, eigenvalues_clipped=False,
                      optimizer_calls=factorizations, optimizer_iterations=iterations,
                      linear_refinement_steps=refinements, energy_evaluations=evaluations,
                      nonlinear_optimization_variables=0, linear_system_variables=len(state),
                      proximal_state_variables=len(state), proximal_set_normalized_slack=target,
                      quadratic_coefficients_reused=projection._quadratic_coefficients_reused,
                      quadratic_hessian_reused=projection._quadratic_hessian_reused,
                      positive_definiteness=last_certificate, runtime_seconds=perf_counter() - started)
        report.update(extra)
        return EnergyProjectionResult(np.array(value, copy=True), converged, status, report)

    if cancel is not None and cancel():
        return result(state, False, "interrupted")
    initial_energy = projection.state_value(state)
    evaluations += 1
    if not np.isfinite(initial_energy):
        raise FloatingPointError("nonfinite quadratic initial energy")
    if initial_energy <= target:
        return result(state, True, "already_feasible", multiplier=0., stationarity_residual=0., complementarity_residual=0.)
    initial_gradient = projection.state_gradient(state)
    if not np.isfinite(initial_gradient).all():
        raise FloatingPointError("nonfinite quadratic initial gradient")
    gradient_squared = float(initial_gradient @ initial_gradient)
    if not np.isfinite(gradient_squared) or gradient_squared == 0.:
        return result(state, False, "quadratic_stationary_hard_case_unknown",
                      reason="projection may require a non-positive-definite hard-case branch")

    def evaluate(multiplier):
        nonlocal factorizations, evaluations, refinements, last_certificate
        if cancel is not None and cancel():
            raise InterruptedError
        if not np.isfinite(multiplier):
            raise ArithmeticError("nonfinite quadratic multiplier")
        matrix = identity + multiplier * model.hessian
        factorizations += 1
        factor, certificate = _positive_factor(matrix, projection.stationarity_tolerance)
        last_certificate = certificate
        candidate = factor.solve(state - multiplier * model.linear)
        previous_residual = np.inf
        while True:
            if cancel is not None and cancel():
                raise InterruptedError
            if not np.isfinite(candidate).all():
                raise ArithmeticError("nonfinite quadratic Lagrangian solution")
            gradient = projection.state_gradient(candidate)
            residual = candidate - state + multiplier * gradient
            stationarity = float(np.max(np.abs(residual), initial=0.))
            threshold = projection.stationarity_tolerance * max(float(np.max(np.abs(candidate), initial=0.)), 1.)
            if stationarity <= threshold:
                break
            if stationarity >= previous_residual:
                raise ArithmeticError("quadratic Lagrangian refinement cannot certify stationarity")
            previous_residual = stationarity
            refined = candidate - factor.solve(residual)
            if np.array_equal(refined, candidate):
                raise ArithmeticError("quadratic Lagrangian refinement reached floating-point resolution")
            candidate = refined
            refinements += 1
        energy = float(projection.state_value(candidate))
        evaluations += 1
        derivative = -float(gradient @ factor.solve(gradient))
        if not np.isfinite([energy, derivative]).all() or derivative > 0.:
            raise ArithmeticError("quadratic secular energy is not a finite decreasing function")
        return dict(multiplier=multiplier, state=candidate, energy=energy, residual=energy - target,
                    derivative=derivative, gradient=gradient, stationarity=stationarity,
                    stationarity_tolerance=threshold, certificate=certificate)

    def certified(point):
        energy_error = abs(point["residual"])
        gradient_norm = float(np.linalg.norm(point["gradient"]))
        distance = energy_error / max(gradient_norm, np.finfo(float).tiny)
        distance_tolerance = np.sqrt(len(state)) * point["stationarity_tolerance"]
        complementarity = abs(point["multiplier"] * point["residual"])
        objective = 0.5 * float(np.sum((point["state"] - state)**2))
        gap_tolerance = projection.stationarity_tolerance * max(objective, 1.)
        return (point["energy"] <= projection.tolerance and point["residual"] <= 0. and
                distance <= distance_tolerance and complementarity <= gap_tolerance), dict(
                    multiplier=float(point["multiplier"]), stationarity_residual=point["stationarity"],
                    stationarity_tolerance=point["stationarity_tolerance"], complementarity_residual=float(complementarity),
                    complementarity_tolerance=float(gap_tolerance), duality_gap_upper_bound=float(complementarity),
                    root_normal_distance=float(distance), root_normal_distance_tolerance=float(distance_tolerance),
                    positive_definiteness=point["certificate"])

    multiplier = max((initial_energy - target) / gradient_squared, np.finfo(float).eps)
    if not np.isfinite(multiplier):
        raise FloatingPointError("nonfinite quadratic initial multiplier")
    lower = dict(multiplier=0., state=state, energy=initial_energy, residual=initial_energy - target,
                 derivative=-gradient_squared, gradient=initial_gradient,
                 stationarity=0., stationarity_tolerance=projection.stationarity_tolerance)
    upper = None
    boundary = None
    try:
        while upper is None:
            if cancel is not None and cancel():
                raise InterruptedError
            try:
                point = evaluate(multiplier)
            except _PositiveDefinitenessBoundary:
                boundary = multiplier
                next_multiplier = lower["multiplier"] + 0.5 * (boundary - lower["multiplier"])
                if next_multiplier == lower["multiplier"] or next_multiplier == boundary:
                    return result(lower["state"], False, "quadratic_pd_domain_exhausted_unknown",
                                  reason="feasibility was not certified before the positive-definite multiplier boundary")
                multiplier = next_multiplier
                continue
            iterations += 1
            if point["residual"] <= 0.:
                upper = point
                break
            lower = point
            next_multiplier = (multiplier * 2 if boundary is None else
                               multiplier + 0.5 * (boundary - multiplier))
            if not np.isfinite(next_multiplier) or next_multiplier == multiplier:
                return result(point["state"], False, "quadratic_multiplier_resolution_unknown")
            multiplier = next_multiplier

        while True:
            valid, certificate = certified(upper)
            if valid:
                return result(upper["state"], True, "quadratic_projection", **certificate)
            if cancel is not None and cancel():
                raise InterruptedError
            anchor = upper if abs(upper["residual"]) < abs(lower["residual"]) else lower
            candidate = anchor["multiplier"] - anchor["residual"] / anchor["derivative"] if anchor["derivative"] < 0 else np.nan
            if not np.isfinite(candidate) or not lower["multiplier"] < candidate < upper["multiplier"]:
                candidate = lower["multiplier"] + 0.5 * (upper["multiplier"] - lower["multiplier"])
            if candidate in (lower["multiplier"], upper["multiplier"]):
                return result(upper["state"], False, "quadratic_root_resolution_unknown", **certificate)
            point = evaluate(candidate)
            iterations += 1
            if point["residual"] <= 0.:
                upper = point
            else:
                lower = point
    except InterruptedError:
        return result(state if upper is None else upper["state"], False, "interrupted")
    except (_PositiveDefinitenessBoundary, ArithmeticError, ValueError) as error:
        return result(state if upper is None else upper["state"], False, "quadratic_projection_unknown", reason=str(error))
