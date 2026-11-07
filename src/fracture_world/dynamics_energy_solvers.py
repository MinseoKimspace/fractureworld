from copy import copy
from dataclasses import replace
from time import perf_counter

import numpy as np

from .dynamics_corrector import CorrectorTolerances, _diagnostics, _feasibility, _limit, _prior_is_minimum, _result, _shared_problem, solve_central


def _phase_progress(progress, method, phase, status, start, **values):
    if progress:
        progress(dict(event="energy_solver_phase", method=method, phase=phase, status=status,
                      runtime_seconds=perf_counter() - start, **values))


def _energy_result(problem, energy, method, state, tolerances, converged, status, iterations, history, start, feasibility, **extra):
    energy_report = energy.diagnostics(state)
    linear_report = _diagnostics(problem, state, tolerances)
    converged = bool(converged and energy_report["energy_satisfied"] and linear_report["constraints_satisfied"])
    if not converged and status == "converged":
        status = "energy_feasibility_unknown"
    result = _result(problem, method, state, tolerances, converged, status, iterations, history, start, feasibility)
    result.diagnostics["energy"] = {**energy_report, **extra, "global_optimality_verified": False}
    return result


def _central(problem, energy, tolerances, max_iterations, cancel, progress, *, start=None):
    start = perf_counter() if start is None else start
    _phase_progress(progress, "central", "linear_feasibility", "started", start)
    feasibility = _feasibility(problem, tolerances)
    _phase_progress(progress, "central", "linear_feasibility", feasibility["status"], start,
                    phase_runtime_seconds=feasibility["runtime_seconds"])
    state = problem.prior.copy()
    iterations, outer = 0, 0
    cuts, reports = [], []
    previous = None
    previous_violation, stagnant = None, 0
    status, converged = "maximum_iterations", False

    if feasibility["status"] != "feasible":
        status = "PrimalInfeasible" if feasibility["status"] == "infeasible" else "feasibility_unknown"
    else:
        while max_iterations is None or iterations < max_iterations:
            if cancel is not None and cancel():
                status = "interrupted"
                break
            local = copy(problem)
            local.factors = (*problem.factors, *cuts)
            remaining = None if max_iterations is None else max_iterations - iterations
            _phase_progress(progress, "central", "linear_subproblem", "started", start, outer_iteration=outer + 1)
            solution = solve_central(local, tolerances, remaining, cancel)
            _phase_progress(progress, "central", "linear_subproblem", solution.status, start,
                            outer_iteration=outer + 1, phase_runtime_seconds=solution.runtime_seconds,
                            iterations=solution.iterations)
            outer += 1
            iterations += solution.iterations
            reports.append({"outer_iteration": outer, "status": solution.status, "iterations": solution.iterations,
                            "feasibility": solution.diagnostics["feasibility"], "objective": solution.objective,
                            "runtime_seconds": solution.runtime_seconds})
            if not solution.converged:
                status = "local_energy_model_infeasible_unknown" if solution.status == "PrimalInfeasible" and cuts else solution.status
                break
            candidate = problem.encode_state(solution.positions, solution.velocities)
            violation = energy.state_value(candidate)
            reports[-1]["normalized_energy_violation"] = float(violation)
            state = candidate
            if progress:
                progress(dict(event="energy_corrector_progress", method="central", iteration=iterations,
                              outer_iteration=outer, objective=solution.objective,
                              constraint_violation=solution.diagnostics["constraint_violation"],
                              energy_violation=float(max(violation, 0.0)),
                              runtime_seconds=perf_counter() - start))
            if violation <= energy.tolerance:
                converged, status = True, "converged"
                break
            if previous is not None and np.array_equal(candidate, previous):
                status = "energy_linearization_stagnation"
                break
            if previous is not None:
                numerical_tolerance = max(64 * np.finfo(float).eps * max(float(np.max(np.abs(candidate), initial=0)), 1.0), tolerances.absolute * 0.01)
                unchanged = np.max(np.abs(candidate - previous), initial=0) <= numerical_tolerance
                nonimproving = previous_violation - violation <= energy.tolerance
                stagnant = stagnant + 1 if unchanged and nonimproving else 0
                if stagnant >= 2:
                    status = "energy_linearization_stagnation"
                    break
            gradient = energy.state_gradient(candidate)
            if not np.isfinite(gradient).all() or np.linalg.norm(gradient) <= np.finfo(float).tiny:
                status = "energy_stationary_infeasible_unknown"
                break
            bound = violation - 0.5 * energy.tolerance - float(gradient @ candidate)
            cuts.append(problem._factor(f"energy_tangent:{outer}", np.zeros(len(candidate), dtype=int),
                                        np.arange(len(candidate)), -gradient, [bound], True))
            previous = candidate.copy()
            previous_violation = violation
    return _energy_result(problem, energy, "central", state, tolerances, converged, status, iterations, [], start, feasibility,
                          algorithm="Clarabel sequential energy tangent cuts with exact final energy validation",
                          approximation="local nonconvex approximation; accumulated cuts can exclude other feasible branches",
                          outer_iterations=outer, linear_subproblems=reports,
                          initial_feasibility_seconds=feasibility["runtime_seconds"],
                          linear_subproblem_seconds=sum(item["runtime_seconds"] for item in reports))


def solve_energy(problem, energy_projection, method, tolerances=None, max_iterations=None, cancel=None, *, progress=None):
    start = perf_counter()
    if method != "central":
        raise ValueError("energy solver method must be central")
    _limit(max_iterations)
    tolerances = tolerances or CorrectorTolerances()
    problem = _shared_problem(problem)
    if energy_projection.problem is not problem:
        energy_projection = replace(energy_projection, problem=problem)
    coordinate_seconds = perf_counter() - start
    _phase_progress(progress, method, "coordinate_setup", "completed", start,
                    phase_runtime_seconds=coordinate_seconds,
                    full_state_variables=problem.metadata.get("full_state_variables", len(problem.prior)),
                    solver_state_variables=len(problem.prior),
                    original_factor_count=problem.metadata.get("original_factor_count", len(problem.factors)),
                    solver_factor_count=len(problem.factors))
    if cancel is not None and cancel() or max_iterations == 0:
        status = "interrupted" if cancel is not None and cancel() else "maximum_iterations"
        return _energy_result(problem, energy_projection, method, problem.prior, tolerances, False, status, 0, [], start, None,
                              coordinate_reduction_seconds=coordinate_seconds)
    if (_prior_is_minimum(problem) and _diagnostics(problem, problem.prior, tolerances)["constraints_satisfied"] and
            energy_projection.state_value(problem.prior) <= energy_projection.tolerance):
        feasibility = _feasibility(problem, tolerances)
        return _energy_result(problem, energy_projection, method, problem.prior, tolerances, True, "converged", 0, [], start,
                              feasibility, algorithm="unconstrained coordinate minimum with directly verified feasibility",
                              zero_objective_certificate=problem.objective(problem.prior) == 0.0,
                              coordinate_minimum_certificate=True, coordinate_reduction_seconds=coordinate_seconds)
    result = _central(problem, energy_projection, tolerances, max_iterations, cancel, progress, start=start)
    result.diagnostics["energy"]["coordinate_reduction_seconds"] = coordinate_seconds
    return replace(result, runtime_seconds=perf_counter() - start)
