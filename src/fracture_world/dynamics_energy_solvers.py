from copy import copy
from time import perf_counter

import numpy as np
from scipy import sparse

from .dynamics_corrector import CorrectorIteration, CorrectorTolerances, _diagnostics, _feasibility, _limit, _result, solve_central, solve_consensus


def _energy_result(problem, energy, method, state, tolerances, converged, status, iterations, history, start, feasibility, **extra):
    energy_report = energy.diagnostics(state)
    linear_report = _diagnostics(problem, state, tolerances)
    converged = bool(converged and energy_report["energy_satisfied"] and linear_report["constraints_satisfied"])
    if not converged and status == "converged":
        status = "energy_feasibility_unknown"
    result = _result(problem, method, state, tolerances, converged, status, iterations, history, start, feasibility)
    result.diagnostics["energy"] = {**energy_report, **extra, "global_optimality_verified": False}
    return result


def _central(problem, energy, tolerances, max_iterations, cancel, progress):
    start = perf_counter()
    feasibility = _feasibility(problem, tolerances)
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
            solution = solve_central(local, tolerances, remaining, cancel)
            outer += 1
            iterations += solution.iterations
            reports.append({"outer_iteration": outer, "status": solution.status, "iterations": solution.iterations,
                            "feasibility": solution.diagnostics["feasibility"], "objective": solution.objective})
            if not solution.converged:
                status = "local_energy_model_infeasible_unknown" if solution.status == "PrimalInfeasible" and cuts else solution.status
                break
            candidate = np.r_[((solution.positions - problem.positions) / problem.position_scale).ravel(),
                              (solution.velocities / problem.velocity_scale).ravel()]
            violation = energy.state_value(candidate)
            reports[-1]["normalized_energy_violation"] = float(violation)
            state = candidate
            if progress:
                progress(dict(event="energy_corrector_progress", method="central", iteration=iterations,
                              outer_iteration=outer, objective=solution.objective, primal_residual=None,
                              dual_residual=None, constraint_violation=solution.diagnostics["constraint_violation"],
                              energy_violation=float(max(violation, 0.0)), rho=None,
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
                          outer_iterations=outer, linear_subproblems=reports)


def _consensus(problem, energy, tolerances, rho, max_iterations, cancel, progress):
    start = perf_counter()
    workspace = {}

    def linear_progress(record):
        if progress:
            progress(dict(event="energy_corrector_progress", method="consensus", phase="linear_consensus_relaxation",
                          iteration=record["iteration"], objective=record["objective"], primal_residual=record["primal"],
                          dual_residual=record["dual"], primal_tolerance=record["primal_tolerance"], dual_tolerance=record["dual_tolerance"],
                          constraint_violation=record["constraint_violation"], constraint_tolerance=record["constraint_tolerance"],
                          energy_violation=None, energy_tolerance=energy.tolerance, rho=record["rho"], runtime_seconds=perf_counter() - start))

    relaxation = solve_consensus(problem, rho=rho, tolerances=tolerances, max_iterations=max_iterations,
                                 cancel=cancel, progress=linear_progress, _workspace=workspace)
    feasibility = relaxation.diagnostics["feasibility"]
    state = workspace["state"].copy()
    history = list(relaxation.history)
    phases = [dict(name="linear_consensus_relaxation", iterations=relaxation.iterations, status=relaxation.status,
                   runtime_seconds=relaxation.runtime_seconds, final_rho=float(workspace["rho"]))]
    if not relaxation.converged:
        return _energy_result(problem, energy, "consensus", state, tolerances, False, relaxation.status,
                              relaxation.iterations, history, start, feasibility, phases=phases, energy_proximal_calls=0)
    if cancel is not None and cancel():
        return _energy_result(problem, energy, "consensus", state, tolerances, False, "interrupted",
                              relaxation.iterations, history, start, feasibility, phases=phases, energy_proximal_calls=0)
    if energy.state_value(state) <= energy.tolerance:
        return _energy_result(problem, energy, "consensus", state, tolerances, True, "converged",
                              relaxation.iterations, history, start, feasibility, phases=phases, energy_proximal_calls=0,
                              algorithm="linear consensus relaxation followed by exact mechanical-energy validation",
                              inactive_energy_certificate=True,
                              certificate="the converged linear relaxation satisfies the additional nonlinear energy constraint at the configured tolerances")
    linear_indices = np.concatenate([factor.indices for factor in problem.factors])
    indices = np.r_[linear_indices, np.arange(len(state))]
    matrix = sparse.block_diag([factor.matrix for factor in problem.factors], format="csr")
    inverse_gram = sparse.block_diag([factor.inverse_gram for factor in problem.factors], format="csr")
    bound = np.concatenate([factor.bound for factor in problem.factors])
    inequality = np.concatenate([np.full(len(factor.bound), factor.inequality, dtype=bool) for factor in problem.factors])
    counts = np.bincount(indices, minlength=len(state))
    if workspace["factor_names"] != tuple(factor.name for factor in problem.factors) or workspace["scaled_dual"].shape != linear_indices.shape:
        raise ValueError("linear consensus warm start does not match the nonlinear factor order")
    rho = float(workspace["rho"])
    warm_start_rho = rho
    warm_start_dual_norm = float(np.linalg.norm(workspace["scaled_dual"]))
    dual = np.r_[workspace["scaled_dual"], np.zeros(len(state))]
    projection_count, projection_iterations, projection_seconds = 0, 0, 0.0
    last_projection = None
    previous = []
    stagnant = 0
    previous_primal, previous_violation = np.inf, np.inf
    iteration, converged, status = relaxation.iterations, False, "maximum_iterations"
    nonlinear_started = perf_counter()
    while max_iterations is None or iteration < max_iterations:
        if cancel is not None and cancel():
            status = "interrupted"
            break
        old = state.copy()
        local = state[indices] - dual
        residual = bound - matrix @ local[:len(linear_indices)]
        residual[inequality] = np.maximum(residual[inequality], 0)
        local[:len(linear_indices)] += matrix.T @ (inverse_gram @ residual)
        projected = energy.project(local[len(linear_indices):], cancel)
        projection_count += 1
        projection_iterations += projected.diagnostics.get("optimizer_iterations", 0)
        projection_seconds += projected.diagnostics.get("runtime_seconds", 0.0)
        last_projection = projected.diagnostics
        if not projected.converged:
            status = projected.status
            break
        local[len(linear_indices):] = projected.state
        totals = np.bincount(indices, weights=local + dual, minlength=len(state))
        state = (problem.prior_weights * problem.prior + rho * totals) / (problem.prior_weights + rho * counts)
        if not np.isfinite(state).all():
            state, status = old, "nonfinite_iterate"
            break
        selected = state[indices]
        residual = local - selected
        dual += residual
        iteration += 1
        primal = float(np.linalg.norm(residual))
        dual_residual = float(rho * np.sqrt(np.dot(counts, (state - old)**2)))
        primal_tolerance = float(np.sqrt(len(indices)) * tolerances.absolute + tolerances.relative * max(np.linalg.norm(local), np.linalg.norm(selected)))
        dual_tolerance = float(np.sqrt(len(indices)) * tolerances.absolute + tolerances.relative * rho * np.linalg.norm(dual))
        constraint_residual = matrix @ state[linear_indices] - bound
        constraint_residual[inequality] = np.minimum(constraint_residual[inequality], 0)
        linear_violation = float(np.max(np.abs(constraint_residual), initial=0))
        linear_tolerance = tolerances.absolute + tolerances.relative * max(float(np.max(np.abs(state), initial=0)), 1.0)
        energy_violation = max(energy.state_value(state), 0.0)
        history.append(CorrectorIteration(iteration, problem.objective(state), primal, dual_residual, primal_tolerance,
                                          dual_tolerance, max(linear_violation, energy_violation), rho))
        if progress and (iteration == relaxation.iterations + 1 or iteration % 25 == 0):
            progress(dict(event="energy_corrector_progress", method="consensus", phase="nonlinear_energy_consensus", iteration=iteration,
                          objective=history[-1].objective, primal_residual=primal, dual_residual=dual_residual,
                          primal_tolerance=primal_tolerance, dual_tolerance=dual_tolerance,
                          constraint_violation=linear_violation, constraint_tolerance=linear_tolerance,
                          energy_violation=float(energy_violation), energy_tolerance=energy.tolerance, rho=float(rho),
                          runtime_seconds=perf_counter() - start))
        if (primal <= primal_tolerance and dual_residual <= dual_tolerance and
                linear_violation <= linear_tolerance and energy_violation <= energy.tolerance):
            converged, status = True, "converged"
            break
        numerical_tolerance = max(64 * np.finfo(float).eps * max(float(np.max(np.abs(state), initial=0)), 1.0), tolerances.absolute * 0.01)
        violation = max(linear_violation, energy_violation)
        unchanged = np.max(np.abs(state - old), initial=0) <= numerical_tolerance
        nonimproving = previous_violation - violation <= max(numerical_tolerance, energy.tolerance) and previous_primal - primal <= primal_tolerance
        multiplier = 1.0
        if iteration % 25 == 0:
            multiplier = 2.0 if primal > 10 * dual_residual else 0.5 if dual_residual > 10 * primal else 1.0
            rho *= multiplier
            dual /= multiplier
        stagnant = stagnant + 1 if unchanged and nonimproving and multiplier == 1.0 else 0
        if stagnant >= 2:
            status = "nonlinear_consensus_stagnation"
            break
        previous_primal, previous_violation = primal, violation
        snapshot = (state.copy(), dual.copy(), rho)
        if any(snapshot[2] == prior[2] and np.array_equal(snapshot[0], prior[0]) and np.array_equal(snapshot[1], prior[1]) for prior in previous):
            status = "numerical_cycle"
            break
        previous = (previous + [snapshot])[-2:]
    phases.append(dict(name="nonlinear_energy_consensus", iterations=iteration - relaxation.iterations, status=status,
                       runtime_seconds=perf_counter() - nonlinear_started, initial_rho=warm_start_rho,
                       initial_linear_scaled_dual_norm=warm_start_dual_norm, initial_energy_scaled_dual_norm=0.0))
    return _energy_result(problem, energy, "consensus", state, tolerances, converged, status, iteration, history, start, feasibility,
                          algorithm="consensus ADMM with local linear projections and a global nonlinear mechanical-energy proximal block",
                          approximation="nonconvex local energy projection; no global convergence guarantee",
                          energy_proximal_calls=projection_count, energy_optimizer_iterations=projection_iterations,
                          energy_projection_seconds=projection_seconds, last_energy_projection=last_projection,
                          phases=phases, warm_started_from_linear_consensus=True,
                          stagnation_criterion="two consecutive state steps within normalized solver tolerance without a resolvable primal or feasibility improvement; conservatively unknown")


def solve_energy(problem, energy_projection, method, tolerances=None, rho=None, max_iterations=None, cancel=None, *, progress=None):
    if method not in {"central", "consensus"}:
        raise ValueError("energy solver method must be central or consensus")
    _limit(max_iterations)
    tolerances = tolerances or CorrectorTolerances()
    rho = float(np.median(problem.prior_weights)) if rho is None else float(rho)
    if not np.isfinite(rho) or rho <= 0:
        raise ValueError("rho must be finite and positive")
    if cancel is not None and cancel() or max_iterations == 0:
        start = perf_counter()
        status = "interrupted" if cancel is not None and cancel() else "maximum_iterations"
        return _energy_result(problem, energy_projection, method, problem.prior, tolerances, False, status, 0, [], start, None)
    if (_diagnostics(problem, problem.prior, tolerances)["constraints_satisfied"] and
            energy_projection.state_value(problem.prior) <= energy_projection.tolerance):
        start = perf_counter()
        feasibility = _feasibility(problem, tolerances)
        return _energy_result(problem, energy_projection, method, problem.prior, tolerances, True, "converged", 0, [], start,
                              feasibility, algorithm="zero correction cost with directly verified feasible prior",
                              zero_objective_certificate=True)
    if method == "central":
        return _central(problem, energy_projection, tolerances, max_iterations, cancel, progress)
    return _consensus(problem, energy_projection, tolerances, rho, max_iterations, cancel, progress)
