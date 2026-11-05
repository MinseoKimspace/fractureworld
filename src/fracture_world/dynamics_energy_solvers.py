from copy import copy
from dataclasses import replace
from time import perf_counter

import numpy as np
from scipy import sparse

from .dynamics_corrector import CorrectorIteration, CorrectorTolerances, _diagnostics, _factor_inverse_gram, _feasibility, _limit, _prior_is_minimum, _result, _shared_problem, solve_central, solve_consensus
from .dynamics_pcg import solve_central_pcg
from .dynamics_solver_workspace import consensus_setup, energy_key, inverse_gram, save_consensus


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


def _central(problem, energy, tolerances, max_iterations, cancel, progress, *, start=None, method="central"):
    start = perf_counter() if start is None else start
    _phase_progress(progress, method, "linear_feasibility", "started", start)
    feasibility = _feasibility(problem, tolerances)
    _phase_progress(progress, method, "linear_feasibility", feasibility["status"], start,
                    phase_runtime_seconds=feasibility["runtime_seconds"])
    state = problem.prior.copy()
    iterations, outer = 0, 0
    cuts, reports = [], []
    previous = None
    previous_violation, stagnant = None, 0
    status, converged = "maximum_iterations", False

    def linear_progress(event):
        if progress:
            progress({**event, "subproblem_runtime_seconds": event["runtime_seconds"],
                      "runtime_seconds": perf_counter() - start, "outer_iteration": outer + 1})

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
            _phase_progress(progress, method, "linear_subproblem", "started", start, outer_iteration=outer + 1)
            solution = (solve_central_pcg(local, tolerances, remaining, cancel, progress=linear_progress if progress else None)
                        if method == "central_pcg" else solve_central(local, tolerances, remaining, cancel))
            _phase_progress(progress, method, "linear_subproblem", solution.status, start,
                            outer_iteration=outer + 1, phase_runtime_seconds=solution.runtime_seconds,
                            iterations=solution.iterations)
            outer += 1
            iterations += solution.iterations
            reports.append({"outer_iteration": outer, "status": solution.status, "iterations": solution.iterations,
                            "feasibility": solution.diagnostics["feasibility"], "objective": solution.objective,
                            "runtime_seconds": solution.runtime_seconds})
            if method == "central_pcg":
                reports[-1]["pcg"] = solution.diagnostics.get("pcg")
            if not solution.converged:
                status = "local_energy_model_infeasible_unknown" if solution.status == "PrimalInfeasible" and cuts else solution.status
                break
            candidate = problem.encode_state(solution.positions, solution.velocities)
            violation = energy.state_value(candidate)
            reports[-1]["normalized_energy_violation"] = float(violation)
            state = candidate
            if progress:
                progress(dict(event="energy_corrector_progress", method=method, iteration=iterations,
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
    backend = "PCG active-set" if method == "central_pcg" else "Clarabel"
    return _energy_result(problem, energy, method, state, tolerances, converged, status, iterations, [], start, feasibility,
                          algorithm=f"{backend} sequential energy tangent cuts with exact final energy validation",
                          approximation="local nonconvex approximation; accumulated cuts can exclude other feasible branches",
                          outer_iterations=outer, linear_subproblems=reports,
                          initial_feasibility_seconds=feasibility["runtime_seconds"],
                          linear_subproblem_seconds=sum(item["runtime_seconds"] for item in reports))


def _consensus(problem, energy, tolerances, rho, max_iterations, cancel, progress, *, start=None, workspace=None):
    start = perf_counter() if start is None else start
    persistent = workspace if workspace is not None else {}
    linear_workspace = persistent.setdefault("linear", {})
    nonlinear_workspace = persistent.setdefault("nonlinear", {})
    resumed, model_key = False, None
    if "state" in nonlinear_workspace:
        state, continued_rho, continued_dual, operators, reuse = consensus_setup(problem, nonlinear_workspace, rho)
        model_key = energy_key(energy)
        energy_dual = nonlinear_workspace.get("energy_scaled_dual")
        resumed = (reuse["state_reused"] and model_key is not None and nonlinear_workspace.get("energy_key") == model_key
                   and isinstance(energy_dual, np.ndarray) and energy_dual.shape == state.shape and np.isfinite(energy_dual).all())

    def linear_progress(record):
        if progress:
            progress(dict(event="energy_corrector_progress", method="consensus", phase="linear_consensus_relaxation",
                          iteration=record["iteration"], objective=record["objective"], primal_residual=record["primal"],
                          dual_residual=record["dual"], primal_tolerance=record["primal_tolerance"], dual_tolerance=record["dual_tolerance"],
                          constraint_violation=record["constraint_violation"], constraint_tolerance=record["constraint_tolerance"],
                          energy_violation=None, energy_tolerance=energy.tolerance, rho=record["rho"], runtime_seconds=perf_counter() - start))

    if resumed:
        rho = continued_rho
        verified = _diagnostics(problem, state, tolerances)
        feasibility = (dict(method="validated continuation state", status="feasible", iterations=0, runtime_seconds=0.,
                            engine_calls=0, maximum_violation=verified["constraint_violation"])
                       if verified["constraints_satisfied"] else _feasibility(problem, tolerances))
        if feasibility["status"] != "feasible":
            status = "PrimalInfeasible" if feasibility["status"] == "infeasible" else "feasibility_unknown"
            return _energy_result(problem, energy, "consensus", state, tolerances, False, status, 0, [], start, feasibility,
                                  continuation=reuse, energy_proximal_calls=0)
        history, relaxation_iterations = [], 0
        linear_dual = continued_dual
        phases = [dict(name="continued_energy_consensus", iterations=0, status="initialized", runtime_seconds=0.,
                       final_rho=rho, **reuse)]
        _phase_progress(progress, "consensus", "continued_energy_consensus", "initialized", start, **reuse)
    else:
        _phase_progress(progress, "consensus", "linear_consensus_relaxation", "started", start)
        relaxation = solve_consensus(problem, rho=rho, tolerances=tolerances, max_iterations=max_iterations,
                                     cancel=cancel, progress=linear_progress, _workspace=linear_workspace)
        feasibility = relaxation.diagnostics["feasibility"]
        _phase_progress(progress, "consensus", "linear_consensus_relaxation", relaxation.status, start,
                        phase_runtime_seconds=relaxation.runtime_seconds,
                        feasibility_seconds=feasibility["runtime_seconds"], iterations=relaxation.iterations)
        state = linear_workspace["state"].copy()
        history = list(relaxation.history)
        relaxation_iterations = relaxation.iterations
        phases = [dict(name="linear_consensus_relaxation", iterations=relaxation.iterations, status=relaxation.status,
                       runtime_seconds=relaxation.runtime_seconds, feasibility_seconds=feasibility["runtime_seconds"],
                       final_rho=float(linear_workspace["rho"]))]
        if not relaxation.converged:
            return _energy_result(problem, energy, "consensus", state, tolerances, False, relaxation.status,
                                  relaxation.iterations, history, start, feasibility, phases=phases, energy_proximal_calls=0)
        if cancel is not None and cancel():
            return _energy_result(problem, energy, "consensus", state, tolerances, False, "interrupted",
                                  relaxation.iterations, history, start, feasibility, phases=phases, energy_proximal_calls=0)
        if energy.state_value(state) <= energy.tolerance:
            nonlinear_workspace.clear()
            return _energy_result(problem, energy, "consensus", state, tolerances, True, "converged",
                                  relaxation.iterations, history, start, feasibility, phases=phases, energy_proximal_calls=0,
                                  algorithm="linear consensus relaxation followed by exact mechanical-energy validation",
                                  inactive_energy_certificate=True,
                                  certificate="the converged linear relaxation satisfies the additional nonlinear energy constraint at the configured tolerances")
        rho = float(linear_workspace["rho"])
        linear_dual = linear_workspace["scaled_dual"].copy()
        operators = linear_workspace["operators"]
        reuse = relaxation.diagnostics.get("continuation", dict(state_reused=False, reused_factor_duals=0,
                                                                 reset_factor_duals=len(problem.factors), operators_reused=False))
        energy_dual = np.zeros(len(state))
    if model_key is None:
        model_key = energy_key(energy)
    linear_indices = operators["indices"]
    indices = np.r_[linear_indices, np.arange(len(state))]
    matrix = operators["matrix"]
    gram = inverse_gram(problem, operators, rho, _factor_inverse_gram)
    bound = np.concatenate([factor.bound for factor in problem.factors])
    inequality, soft_rows = operators["inequality"], operators["soft_rows"]
    counts = np.bincount(indices, minlength=len(state))
    warm_start_rho = rho
    warm_start_dual_norm = float(np.linalg.norm(linear_dual))
    energy_dual_norm = float(np.linalg.norm(energy_dual))
    dual = np.r_[linear_dual, energy_dual]
    projection_count, projection_iterations, projection_seconds = 0, 0, 0.0
    last_projection = None
    previous = []
    stagnant = 0
    previous_primal, previous_violation = np.inf, np.inf
    iteration, converged, status = relaxation_iterations, False, "maximum_iterations"
    nonlinear_started = perf_counter()
    while max_iterations is None or iteration < max_iterations:
        if cancel is not None and cancel():
            status = "interrupted"
            break
        old = state.copy()
        previous_dual = dual.copy()
        local = state[indices] - dual
        residual = bound - matrix @ local[:len(linear_indices)]
        residual[inequality] = np.maximum(residual[inequality], 0)
        local[:len(linear_indices)] += matrix.T @ (gram @ residual)
        projection_visible = projection_count == 0 or (iteration + 1) % 25 == 0
        if projection_visible:
            _phase_progress(progress, "consensus", "energy_projection", "started", start,
                            iteration=iteration + 1, projection=projection_count + 1)
        projection_started = perf_counter()
        projected = energy.project(local[len(linear_indices):], cancel)
        projection_elapsed = perf_counter() - projection_started
        projection_count += 1
        projection_iterations += projected.diagnostics.get("optimizer_iterations", 0)
        projection_seconds += projection_elapsed
        last_projection = projected.diagnostics
        if projection_visible or projection_elapsed >= 5:
            _phase_progress(progress, "consensus", "energy_projection", projected.status, start,
                            iteration=iteration + 1, projection=projection_count,
                            phase_runtime_seconds=projection_elapsed,
                            optimizer_iterations=projected.diagnostics.get("optimizer_iterations", 0),
                            energy_evaluations=projected.diagnostics.get("energy_evaluations", 0))
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
        constraint_residual[soft_rows] = 0
        linear_violation = float(np.max(np.abs(constraint_residual), initial=0))
        linear_tolerance = tolerances.absolute + tolerances.relative * max(float(np.max(np.abs(state), initial=0)), 1.0)
        energy_violation = max(energy.state_value(state), 0.0)
        history.append(CorrectorIteration(iteration, problem.objective(state), primal, dual_residual, primal_tolerance,
                                          dual_tolerance, max(linear_violation, energy_violation), rho))
        if progress and (iteration == relaxation_iterations + 1 or iteration % 25 == 0):
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
        dual_unchanged = np.max(np.abs(dual - previous_dual), initial=0) <= numerical_tolerance
        nonimproving = previous_violation - violation <= max(numerical_tolerance, energy.tolerance) and previous_primal - primal <= primal_tolerance
        multiplier = 1.0
        if iteration % 25 == 0:
            multiplier = 2.0 if primal > 10 * dual_residual else 0.5 if dual_residual > 10 * primal else 1.0
            rho *= multiplier
            dual /= multiplier
            if multiplier != 1.0:
                gram = inverse_gram(problem, operators, rho, _factor_inverse_gram)
        certified_quadratic = (projected.diagnostics.get("method") == "sparse quadratic Lagrangian secular projection" and
                               projected.diagnostics.get("global_projection_verified", False))
        local_nonconvex_projection = not certified_quadratic
        stationary = unchanged and (dual_unchanged or local_nonconvex_projection)
        stagnant = stagnant + 1 if stationary and nonimproving and multiplier == 1.0 else 0
        if stagnant >= 2:
            status = "nonlinear_consensus_stagnation"
            break
        previous_primal, previous_violation = primal, violation
        snapshot = (state.copy(), dual.copy(), rho)
        if any(snapshot[2] == prior[2] and np.array_equal(snapshot[0], prior[0]) and np.array_equal(snapshot[1], prior[1]) for prior in previous):
            status = "numerical_cycle"
            break
        previous = (previous + [snapshot])[-2:]
    phases.append(dict(name="nonlinear_energy_consensus", iterations=iteration - relaxation_iterations, status=status,
                       runtime_seconds=perf_counter() - nonlinear_started, initial_rho=warm_start_rho,
                       initial_linear_scaled_dual_norm=warm_start_dual_norm, initial_energy_scaled_dual_norm=energy_dual_norm))
    save_consensus(problem, nonlinear_workspace, state, dual[:len(linear_indices)], rho, operators)
    nonlinear_workspace.update(energy_scaled_dual=dual[len(linear_indices):].copy(), energy_key=model_key, phase="nonlinear")
    return _energy_result(problem, energy, "consensus", state, tolerances, converged, status, iteration, history, start, feasibility,
                          algorithm="consensus ADMM with local linear projections and a global nonlinear mechanical-energy proximal block",
                          approximation="nonconvex local energy projection; no global convergence guarantee",
                          energy_proximal_calls=projection_count, energy_optimizer_iterations=projection_iterations,
                          energy_projection_seconds=projection_seconds, last_energy_projection=last_projection,
                          phases=phases, warm_started_from_linear_consensus=not resumed, continued_from_previous_solve=bool(resumed),
                          continuation=reuse,
                          stagnation_criterion="certified quadratic projections require primal and scaled-dual fixed points; uncertified local nonconvex projection stagnation remains unknown")


def solve_energy(problem, energy_projection, method, tolerances=None, rho=None, max_iterations=None, cancel=None, *, progress=None, workspace=None):
    start = perf_counter()
    if method not in {"central", "central_pcg", "consensus"}:
        raise ValueError("energy solver method must be central, central_pcg or consensus")
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
    rho = float(np.median(problem.prior_weights)) if rho is None else float(rho)
    if not np.isfinite(rho) or rho <= 0:
        raise ValueError("rho must be finite and positive")
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
    if method in {"central", "central_pcg"}:
        result = _central(problem, energy_projection, tolerances, max_iterations, cancel, progress, start=start, method=method)
    else:
        result = _consensus(problem, energy_projection, tolerances, rho, max_iterations, cancel, progress, start=start, workspace=workspace)
    result.diagnostics["energy"]["coordinate_reduction_seconds"] = coordinate_seconds
    return replace(result, runtime_seconds=perf_counter() - start)
