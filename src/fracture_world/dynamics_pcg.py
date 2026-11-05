from hashlib import blake2b
from heapq import heapify, heappop, heappush
from time import perf_counter

import numpy as np
from scipy import sparse
from scipy.optimize import linprog
from scipy.sparse.linalg import LinearOperator, cg, splu, spsolve_triangular

from .dynamics_corrector import (
    CorrectorTolerances, _constraint_system, _diagnostics, _feasibility, _limit,
    _prior_is_minimum, _quadratic_objective, _result, _shared_problem,
)


class _StopSolve(Exception):
    pass


class _SparseReduction:
    def __init__(self, matrix, bounds):
        self.matrix = matrix.tocsr()
        rows, right, pivots, selected = [], [], [], []
        pivot_order = {}
        self.discarded_row_residual = 0.0
        epsilon = np.finfo(float).eps
        for index in range(matrix.shape[0]):
            start, end = self.matrix.indptr[index:index + 2]
            row = dict(zip(self.matrix.indices[start:end], self.matrix.data[start:end]))
            magnitude = max((abs(value) for value in row.values()), default=0.0)
            if not magnitude:
                continue
            row = {column: value / magnitude for column, value in row.items()}
            bound = float(bounds[index] / magnitude)
            growth = 1.0
            pending = [pivot_order[column] for column in row if column in pivot_order]
            heapify(pending)
            while pending:
                previous_index = heappop(pending)
                pivot = pivots[previous_index]
                previous = rows[previous_index]
                previous_bound = right[previous_index]
                coefficient = row.pop(pivot, 0.0)
                if coefficient:
                    growth += abs(coefficient)
                    bound -= coefficient * previous_bound
                    for column, value in previous.items():
                        if column == pivot:
                            continue
                        old = row.get(column, 0.0)
                        updated = old - coefficient * value
                        if updated:
                            row[column] = updated
                            if not old and column in pivot_order:
                                heappush(pending, pivot_order[column])
                        else:
                            row.pop(column, None)
            magnitude = max((abs(value) for value in row.values()), default=0.0)
            if magnitude <= 32 * epsilon * growth:
                self.discarded_row_residual = max(self.discarded_row_residual, magnitude)
                continue
            pivot = max(row, key=lambda column: (abs(row[column]), -column))
            divisor = row[pivot]
            rows.append({column: value / divisor for column, value in row.items()})
            right.append(bound / divisor)
            pivot_order[pivot] = len(pivots)
            pivots.append(pivot)
            selected.append(index)
        self.selected = np.asarray(selected, dtype=int)
        self.pivots = np.asarray(pivots, dtype=int)
        mask = np.ones(matrix.shape[1], dtype=bool)
        mask[self.pivots] = False
        self.free = np.flatnonzero(mask)
        row_ids, columns, values = [], [], []
        for index, row in enumerate(rows):
            for column, value in row.items():
                row_ids.append(index)
                columns.append(column)
                values.append(value)
        self.echelon = sparse.csr_matrix((values, (row_ids, columns)), shape=(len(rows), matrix.shape[1]))
        self.bound = np.asarray(right)
        self.upper = self.echelon[:, self.pivots].tocsr()
        self.upper_transpose = self.upper.T.tocsr()
        self.free_matrix = self.echelon[:, self.free].tocsr()
        self.dual_factor = (splu(self.matrix[self.selected][:, self.pivots].T.tocsc())
                            if len(rows) else None)

    def particular(self, state):
        result = np.zeros(self.matrix.shape[1])
        if len(self.pivots):
            result[self.pivots] = spsolve_triangular(
                self.upper, self.bound - self.echelon @ state, lower=False, unit_diagonal=True)
        return result

    def expand(self, value):
        result = np.zeros(self.matrix.shape[1])
        result[self.free] = value
        if len(self.pivots):
            result[self.pivots] = spsolve_triangular(
                self.upper, -(self.free_matrix @ value), lower=False, unit_diagonal=True)
        return result

    def transpose(self, value):
        result = np.asarray(value)[self.free].copy()
        if len(self.pivots):
            selected = spsolve_triangular(
                self.upper_transpose, np.asarray(value)[self.pivots], lower=True, unit_diagonal=True)
            result -= self.free_matrix.T @ selected
        return result

    def multipliers(self, gradient):
        result = np.zeros(self.matrix.shape[0])
        if self.dual_factor is not None:
            result[self.selected] = self.dual_factor.solve(-np.asarray(gradient)[self.pivots])
        return result


def _optimality(problem, state, hessian, linear, equalities, equality_bounds,
                inequalities, inequality_bounds, equality_dual, inequality_dual, tolerances):
    inequality_dual = np.maximum(inequality_dual, 0.0)
    gradient = np.asarray(hessian @ state).ravel() + linear
    residual = gradient + equalities.T @ equality_dual + inequalities.T @ inequality_dual
    weights = problem.prior_weights
    weighted_residual = float(np.linalg.norm(residual / np.sqrt(weights)))
    stationarity_tolerance = tolerances.absolute + tolerances.relative * max(
        float(np.linalg.norm(np.sqrt(weights) * state)),
        float(np.linalg.norm(linear / np.sqrt(weights))), 1.0)
    equality_residual = np.asarray(equalities @ state).ravel() - equality_bounds
    slack = inequality_bounds - np.asarray(inequalities @ state).ravel()
    complementarity = float(np.max(np.abs(inequality_dual * slack), initial=0.0))
    gap_bound = (0.5 * weighted_residual**2 + abs(float(equality_dual @ equality_residual))
                 + abs(float(inequality_dual @ slack)))
    objective_tolerance = tolerances.absolute + tolerances.relative * max(abs(problem.objective(state)), 1.0)
    primal = _diagnostics(problem, state, tolerances)
    finite = bool(np.isfinite([weighted_residual, stationarity_tolerance, gap_bound, objective_tolerance,
                               complementarity]).all() and np.isfinite(equality_dual).all()
                  and np.isfinite(inequality_dual).all())
    satisfied = bool(finite and primal["constraints_satisfied"] and weighted_residual <= stationarity_tolerance
                     and gap_bound <= objective_tolerance and complementarity <= objective_tolerance)
    return dict(satisfied=satisfied, weighted_stationarity_residual=weighted_residual,
                stationarity_tolerance=float(stationarity_tolerance), objective_gap_bound=float(gap_bound),
                objective_gap_tolerance=float(objective_tolerance), complementarity_residual=complementarity,
                dual_nonnegative=True, finite=finite, primal_satisfied=bool(primal["constraints_satisfied"]),
                equality_multiplier_norm=float(np.linalg.norm(equality_dual)),
                inequality_multiplier_norm=float(np.linalg.norm(inequality_dual)),
                minimum_inequality_multiplier=float(np.min(inequality_dual, initial=0.0)),
                nonzero_inequality_multipliers=int(np.count_nonzero(inequality_dual)))


def _dual_certificate(problem, state, hessian, linear, equalities, equality_bounds,
                      inequalities, inequality_bounds, tolerances):
    primal_tolerance = tolerances.absolute + tolerances.relative * max(float(np.max(np.abs(state), initial=0)), 1.0)
    active = np.flatnonzero(inequality_bounds - inequalities @ state <= primal_tolerance)
    matrix = sparse.vstack((equalities, inequalities[active]), format="csc")
    if not matrix.shape[0]:
        return None
    scale = 1.0 / np.sqrt(problem.prior_weights)
    gradient = np.asarray(hessian @ state).ravel() + linear
    result = linprog(np.zeros(matrix.shape[0]), A_eq=sparse.diags(scale) @ matrix.T,
                     b_eq=-scale * gradient,
                     bounds=[(None, None)] * equalities.shape[0] + [(0.0, None)] * len(active),
                     method="highs", options={"primal_feasibility_tolerance": max(1e-10, min(tolerances.absolute, 1e-7))})
    if result.status != 0 or result.x is None or not np.isfinite(result.x).all():
        return None
    inequality_dual = np.zeros(inequalities.shape[0])
    inequality_dual[active] = result.x[equalities.shape[0]:]
    certificate = _optimality(problem, state, hessian, linear, equalities, equality_bounds,
                              inequalities, inequality_bounds, result.x[:equalities.shape[0]], inequality_dual, tolerances)
    return certificate, result.x[:equalities.shape[0]], inequality_dual


def _publication_polish(problem, state, scale, matrix, working, hessian, linear,
                        equalities, equality_bounds, inequalities, inequality_bounds,
                        equality_dual, inequality_dual, tolerances):
    report = dict(attempted=False, accepted=False, status="no_binding_inequalities", retries=0,
                  maximum_inward_guard=0.0, maximum_state_change=0.0)
    if not working:
        return state, None, report
    published = problem.encode_state(*problem.unpack(state))
    ulps = np.maximum(np.abs(np.spacing(state)), np.abs(published - state))
    terms = (np.diff(inequalities.tocsr().indptr) + 1) * np.finfo(float).eps
    gamma = terms / (1.0 - terms)
    guards = (np.abs(np.spacing(inequality_bounds)) + np.asarray(abs(inequalities) @ ulps).ravel()
              + gamma * np.asarray(abs(inequalities) @ abs(state)).ravel())
    slack = inequality_bounds - inequalities @ state
    binding = np.asarray(working)[slack[working] <= guards[working]]
    if not len(binding):
        return state, None, report
    guarded_bounds = inequality_bounds.copy()
    guarded_bounds[binding] = np.minimum(
        inequality_bounds[binding] - guards[binding], np.nextafter(inequality_bounds[binding], -np.inf))
    report.update(attempted=True, retries=1,
                  maximum_inward_guard=float(np.max(inequality_bounds[binding] - guarded_bounds[binding])))
    basis = _SparseReduction(matrix, np.r_[equality_bounds, guarded_bounds[working]])
    candidate = scale * (state / scale + basis.particular(state / scale))
    if np.any(inequalities @ candidate > inequality_bounds):
        report["status"] = "inward_candidate_violates_original_inequality"
        return state, None, report
    if np.array_equal(candidate, state):
        report["status"] = "no_representable_state_change"
        return state, None, report
    certificate = _optimality(problem, candidate, hessian, linear, equalities, equality_bounds,
                              inequalities, inequality_bounds, equality_dual, inequality_dual, tolerances)
    if not certificate["satisfied"]:
        report["status"] = "original_optimality_revalidation_failed"
        return state, None, report
    report.update(accepted=True, status="original_constraints_and_optimality_revalidated",
                  maximum_state_change=float(np.max(np.abs(candidate - state), initial=0.0)))
    return candidate, certificate, report


def _pcg_direction(basis, hessian, gradient, particular, target, stats, max_iterations, cancel, progress, start):
    if not len(basis.free):
        return particular
    right = -basis.transpose(gradient + hessian @ particular)
    threshold = target * max(float(np.linalg.norm(right)), 1.0)
    if float(np.linalg.norm(right)) <= threshold:
        return particular

    def product(value):
        result = basis.transpose(hessian @ basis.expand(value))
        if not np.isfinite(result).all() or (np.any(value) and float(value @ result) <= 0):
            raise _StopSolve("pcg_nonpositive_curvature")
        return result

    operator = LinearOperator((len(right), len(right)), matvec=product, dtype=float)
    diagonal = np.asarray(hessian.diagonal())[basis.free] + np.asarray(basis.free_matrix.power(2).sum(axis=0)).ravel()
    preconditioner = LinearOperator(operator.shape, matvec=lambda value: value / diagonal, dtype=float)
    value = np.zeros(len(right))
    visited = set()

    def callback(iterate):
        if cancel is not None and cancel():
            raise _StopSolve("interrupted")
        stats["iterations"] += 1
        stats["pcg_iterations"] += 1
        if progress and (stats["pcg_iterations"] == 1 or stats["pcg_iterations"] % 25 == 0):
            progress(dict(event="corrector_iteration", method="central_pcg", stage="pcg",
                          iteration=stats["iterations"], pcg_iterations=stats["pcg_iterations"],
                          active_set_iterations=stats["active_set_iterations"], runtime_seconds=perf_counter() - start))

    while True:
        if cancel is not None and cancel():
            raise _StopSolve("interrupted")
        remaining = None if max_iterations is None else max_iterations - stats["iterations"]
        if remaining is not None and remaining <= 0:
            raise _StopSolve("maximum_iterations")
        chunk = max(20, len(right)) if remaining is None else remaining
        value, info = cg(operator, right, x0=value, rtol=target, atol=target, maxiter=chunk,
                         M=preconditioner, callback=callback)
        residual = float(np.linalg.norm(right - operator @ value))
        stats["last_pcg_residual"] = residual
        stats["last_pcg_tolerance"] = threshold
        if not np.isfinite(value).all() or not np.isfinite(residual) or info < 0:
            raise _StopSolve("pcg_numerical_failure")
        if residual <= threshold:
            return particular + basis.expand(value)
        if remaining is not None:
            raise _StopSolve("maximum_iterations")
        signature = blake2b(value.tobytes(), digest_size=24).digest()
        if signature in visited:
            raise _StopSolve("pcg_stagnation")
        visited.add(signature)
        stats["pcg_restarts"] += 1


def solve_central_pcg(problem, tolerances=None, max_iterations=None, cancel=None, progress=None):
    start = perf_counter()
    problem = _shared_problem(problem)
    tolerances = tolerances or CorrectorTolerances()
    _limit(max_iterations)
    stats = dict(iterations=0, pcg_iterations=0, active_set_iterations=0, pcg_restarts=0,
                 constraint_factorizations=0, dual_certificate_solves=0,
                 algorithm="feasible active set with sparse implicit null-space PCG",
                 optimality_certificate=None)
    feasibility = None
    state = problem.prior.copy()

    def finish(converged, status):
        if cancel is not None and cancel():
            converged, status = False, "interrupted"
        result = _result(problem, "central_pcg", state, tolerances, converged, status,
                         stats["iterations"], [], start, feasibility)
        result.diagnostics["pcg"] = dict(stats)
        return result

    if cancel is not None and cancel():
        return finish(False, "interrupted")
    if max_iterations == 0:
        return finish(False, "maximum_iterations")
    feasibility = _feasibility(problem, tolerances)
    if feasibility["status"] != "feasible":
        return finish(False, "PrimalInfeasible" if feasibility["status"] == "infeasible" else "feasibility_unknown")
    (equalities, equality_bounds), (inequalities, inequality_bounds) = _constraint_system(problem)
    upper, linear = _quadratic_objective(problem)
    hessian = (upper + upper.T - sparse.diags(upper.diagonal())).tocsc()
    if feasibility["method"] != "direct feasible prior certificate":
        try:
            recovered = linprog(np.zeros(len(state)), A_ub=inequalities if len(inequality_bounds) else None,
                                b_ub=inequality_bounds if len(inequality_bounds) else None,
                                A_eq=equalities if len(equality_bounds) else None,
                                b_eq=equality_bounds if len(equality_bounds) else None, bounds=(None, None), method="highs",
                                options={"primal_feasibility_tolerance": max(1e-10, min(tolerances.absolute, 1e-7))})
        except (RuntimeError, ValueError) as error:
            stats["numerical_error"] = f"{type(error).__name__}: {error}"
            return finish(False, "feasible_start_unknown")
        stats["feasible_start_iterations"] = int(getattr(recovered, "nit", 0) or 0)
        if recovered.status != 0 or recovered.x is None or not np.isfinite(recovered.x).all():
            return finish(False, "feasible_start_unknown")
        state = np.asarray(recovered.x)
        if not _diagnostics(problem, state, tolerances)["constraints_satisfied"]:
            return finish(False, "feasible_start_validation_failed")
    if cancel is not None and cancel():
        return finish(False, "interrupted")
    if _prior_is_minimum(problem) and np.array_equal(state, problem.prior):
        certificate = _optimality(problem, state, hessian, linear, equalities, equality_bounds,
                                  inequalities, inequality_bounds, np.zeros(len(equality_bounds)),
                                  np.zeros(len(inequality_bounds)), tolerances)
        stats["optimality_certificate"] = certificate
        if certificate["satisfied"]:
            return finish(True, "converged")
    scale = 1.0 / np.sqrt(hessian.diagonal())
    scaling = sparse.diags(scale)
    scaled_hessian = (scaling @ hessian @ scaling).tocsc()
    scaled_equalities = (equalities @ scaling).tocsr()
    scaled_inequalities = (inequalities @ scaling).tocsr()
    scaled_state = state / scale
    working = []
    visited = set()
    target = 0.01 * min(tolerances.absolute, tolerances.relative or tolerances.absolute)
    try:
        while True:
            if cancel is not None and cancel():
                raise _StopSolve("interrupted")
            if max_iterations is not None and stats["iterations"] >= max_iterations:
                raise _StopSolve("maximum_iterations")
            signature = (tuple(working), blake2b(scaled_state.tobytes(), digest_size=24).digest())
            if signature in visited:
                raise _StopSolve("active_set_cycle")
            visited.add(signature)
            stats["iterations"] += 1
            stats["active_set_iterations"] += 1
            matrix = sparse.vstack((scaled_equalities, scaled_inequalities[working]), format="csr")
            bounds = np.r_[equality_bounds, inequality_bounds[working]]
            basis = _SparseReduction(matrix, bounds)
            stats["constraint_factorizations"] += 1
            stats["working_set_size"] = len(working)
            stats["independent_constraint_rows"] = len(basis.selected)
            stats["discarded_row_residual"] = basis.discarded_row_residual
            selected_working = {int(index - len(equality_bounds)) for index in basis.selected if index >= len(equality_bounds)}
            dependent = [index for position, index in enumerate(working) if position not in selected_working]
            if dependent:
                working = [index for index in working if index not in dependent]
                continue
            gradient = scale * (hessian @ state + linear)
            particular = basis.particular(scaled_state)
            direction = _pcg_direction(basis, scaled_hessian, gradient, particular, target, stats,
                                       max_iterations, cancel, progress, start)
            derivative = np.asarray(scaled_inequalities @ direction).ravel()
            slack = inequality_bounds - inequalities @ state
            inactive = np.ones(len(inequality_bounds), dtype=bool)
            inactive[working] = False
            rounding = 64 * np.finfo(float).eps * np.asarray(abs(scaled_inequalities) @ abs(direction)).ravel()
            blocking = np.flatnonzero(inactive & (derivative > rounding) & (derivative - slack > target))
            fractions = np.maximum(slack[blocking], 0.0) / derivative[blocking]
            blocker = None
            step = 1.0
            if len(blocking):
                minimum = float(np.min(fractions))
                if minimum < 1.0:
                    step = minimum
                    blocker = int(blocking[np.flatnonzero(fractions == minimum)[0]])
            scaled_state = scaled_state + step * direction
            state = scale * scaled_state
            if not np.isfinite(state).all():
                raise _StopSolve("pcg_numerical_failure")
            if progress:
                progress(dict(event="corrector_iteration", method="central_pcg", stage="active_set",
                              iteration=stats["iterations"], objective=problem.objective(state),
                              active_constraints=len(working), step_length=float(step), runtime_seconds=perf_counter() - start))
            if blocker is not None:
                working.append(blocker)
                working.sort()
                continue
            dual = basis.multipliers(scale * (hessian @ state + linear))
            inequality_dual = np.zeros(len(inequality_bounds))
            inequality_dual[working] = dual[len(equality_bounds):]
            certificate = _optimality(problem, state, hessian, linear, equalities, equality_bounds,
                                      inequalities, inequality_bounds, dual[:len(equality_bounds)], inequality_dual, tolerances)
            stats["optimality_certificate"] = certificate
            if certificate["satisfied"]:
                state, polished_certificate, report = _publication_polish(
                    problem, state, scale, matrix, working, hessian, linear,
                    equalities, equality_bounds, inequalities, inequality_bounds,
                    dual[:len(equality_bounds)], inequality_dual, tolerances)
                stats["publication_polish"] = report
                stats["constraint_factorizations"] += int(report["attempted"])
                if polished_certificate is not None:
                    stats["optimality_certificate"] = polished_certificate
                return finish(True, "converged")
            if np.any(dual[len(equality_bounds):] < 0):
                stats["dual_certificate_solves"] += 1
                alternative = _dual_certificate(problem, state, hessian, linear, equalities, equality_bounds,
                                                inequalities, inequality_bounds, tolerances)
                if alternative is not None and alternative[0]["satisfied"]:
                    state, polished_certificate, report = _publication_polish(
                        problem, state, scale, matrix, working, hessian, linear,
                        equalities, equality_bounds, inequalities, inequality_bounds,
                        alternative[1], alternative[2], tolerances)
                    stats["publication_polish"] = report
                    stats["constraint_factorizations"] += int(report["attempted"])
                    stats["optimality_certificate"] = alternative[0] if polished_certificate is None else polished_certificate
                    return finish(True, "converged")
                release = min(index for position, index in enumerate(working) if dual[len(equality_bounds) + position] < 0)
                working.remove(release)
                continue
            raise _StopSolve("pcg_optimality_validation_failed")
    except _StopSolve as error:
        return finish(False, str(error))
    except (RuntimeError, ValueError, FloatingPointError, np.linalg.LinAlgError) as error:
        stats["numerical_error"] = f"{type(error).__name__}: {error}"
        return finish(False, "pcg_numerical_failure")
