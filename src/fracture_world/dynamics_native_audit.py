import numpy as np

from .femfx import REVISION, SOLVER_OBSERVATION_FIELDS, _solver_observation_arrays


def _system_statistics(values, tetrahedra):
    tet = int(tetrahedra[0])
    recursive, explicit, rhs, tolerance, epsilon, iterations, maximum = (
        float(values[name][tet]) for name in ("mpcg_recursive_norm_squared", "mpcg_true_norm_squared",
                                              "mpcg_rhs_norm_squared", "mpcg_tolerance_squared", "mpcg_epsilon",
                                              "mpcg_iterations", "mpcg_max_iterations"))
    if rhs == 0:
        termination = "zero_rhs_early_return"
    elif recursive <= tolerance:
        termination = "recursive_tolerance"
    else:
        termination = "iteration_limit" if iterations == maximum else "inconsistent_early_exit"
    return dict(original_tetrahedron=tet, original_tetrahedra=tetrahedra.tolist(),
                iterations=int(iterations), maximum_iterations=int(maximum), termination=termination,
                recursive_norm_squared=recursive, explicit_norm_squared=explicit,
                adjusted_rhs_norm_squared=rhs, stopping_tolerance_squared=tolerance, epsilon_cg=epsilon,
                recursive_relative_norm=float(np.sqrt(recursive / rhs)) if rhs > 0 else None,
                explicit_relative_norm=float(np.sqrt(explicit / rhs)) if rhs > 0 else None,
                repeated_piece_statistic=True)


def native_solver_diagnostics(observations, balance, velocity_scale):
    report = dict(status="unknown", certificate_scope="native MPCG stopping rule and pre-constraint solution publication only",
                  equation_status="unknown", engine_calls=0, inference_input=False,
                  native_revision=REVISION,
                  source=f"https://github.com/GPUOpen-Effects/FEMFX/blob/{REVISION}/amd_femfx/src/SparseSolvers/FEMFXMpcgSolver.cpp",
                  norm="r.T @ P_inverse @ r over unique free native vertices; relative norm=sqrt(delta/rhs_delta)",
                  stopping_rule="recursive_norm_squared <= epsilon_cg**2 * adjusted_rhs_norm_squared",
                  explicit_residual_is_stopping_rule=False,
                  physical_contact_law_status="unknown")
    if observations is None or observations.get("single_step_coverage") is not True:
        return {**report, "reason": "complete single-step native solver observations unavailable"}
    if not set(SOLVER_OBSERVATION_FIELDS).intersection(observations):
        return {**report, "reason": "actual native linear-system observations were not recorded"}
    values = _solver_observation_arrays(observations, len(balance.mesh.tets), required=True)
    observed = values["mpcg_observed"].reshape(-1)
    stopping = values["mpcg_stopping_observed"]
    free = observed & ~values["mpcg_kinematic"].reshape(-1)
    recursive = values["mpcg_recursive_norm_squared"]
    explicit = values["mpcg_true_norm_squared"]
    rhs = values["mpcg_rhs_norm_squared"]
    tolerance = values["mpcg_tolerance_squared"]
    expected_tolerance = ((values["mpcg_epsilon"].astype(np.float32)**2) * rhs.astype(np.float32)).astype(float)
    tolerance_identity = bool(np.array_equal(tolerance[stopping], expected_tolerance[stopping]))
    recursive_pass = recursive <= tolerance
    explicit_pass = explicit <= tolerance
    iterative = stopping & values["mpcg_rhs_positive"]
    zero_rhs = stopping & ~values["mpcg_rhs_positive"]
    zero_rhs_exact = zero_rhs & (recursive == 0) & (explicit == 0)
    complete = bool(observed.all() and stopping.all())
    failed = bool(np.any(iterative & ~recursive_pass) or np.any(zero_rhs & ~zero_rhs_exact)) or not tolerance_identity
    explicit_failed = bool(np.any(stopping & ~explicit_pass))
    publication_observed = np.asarray(observations["contact_observed"]).reshape(-1) & observed
    solution = values["mpcg_solution"].reshape(-1, 3)
    published = np.asarray(observations["pre_contact_velocities"]).reshape(-1, 3)
    publication_error = float(np.max(np.abs(solution[publication_observed] - published[publication_observed]))) if publication_observed.any() else None
    publication_status = "fail" if publication_error is not None and publication_error != 0 else (
        "pass" if publication_observed.all() else "unknown")
    failed |= publication_status == "fail"
    observed_tets = np.flatnonzero(stopping)
    statistic_fields = ("mpcg_iterations", "mpcg_max_iterations", "mpcg_recursive_norm_squared", "mpcg_true_norm_squared",
                        "mpcg_rhs_norm_squared", "mpcg_tolerance_squared", "mpcg_epsilon")
    statistics = np.column_stack([values[name][stopping] for name in statistic_fields])
    _, representatives, groups = np.unique(statistics, axis=0, return_index=True, return_inverse=True)
    grouped_tets = [observed_tets[groups == group] for group in np.argsort(representatives)]
    report.update(status="fail" if failed else "pass" if complete and publication_status == "pass" else "unknown",
                  equation_status="fail" if explicit_failed else "pass" if complete else "unknown",
                  recursive_stopping_status="fail" if np.any(iterative & ~recursive_pass) else "pass" if iterative.all() else "unknown",
                  zero_rhs_exact_solution_tetrahedra=np.flatnonzero(zero_rhs_exact).tolist(),
                  zero_rhs_nonzero_residual_tetrahedra=np.flatnonzero(zero_rhs & ~zero_rhs_exact).tolist(),
                  tolerance_arithmetic_status="fail" if not tolerance_identity else "pass" if stopping.any() else "unknown",
                  solution_publication=dict(status=publication_status, maximum_velocity_difference_m_per_s=publication_error,
                                            arithmetic="exact equality of captured native float32 values"),
                  observed_tetrahedra=int(stopping.sum()), observed_corners=int(observed.sum()), free_corners=int(free.sum()),
                  incomplete_observation=not complete,
                  failed_stopping_tetrahedra=np.flatnonzero(iterative & ~recursive_pass).tolist(),
                  failed_explicit_residual_tetrahedra=np.flatnonzero(stopping & ~explicit_pass).tolist(),
                  statistic_grouping="identical repeated statistics grouped by value; distinct native pieces may share a group",
                  systems=[_system_statistics(values, tets) for tets in grouped_tets])
    if free.any() and observed.all():
        mass = values["mpcg_masses"].reshape(-1)
        native_residual = values["mpcg_residual"].reshape(-1, 3)
        nodal_solution = solution[balance.representatives].ravel()
        reconstructed = (balance.rhs - balance.matrix @ nodal_solution).reshape(-1, 3)[balance.groups]
        native_normalized = native_residual[free] / (mass[free, None] * velocity_scale)
        reconstructed_normalized = reconstructed[free] / (balance.node_masses[balance.groups[free], None] * velocity_scale)
        report["operator_reconstruction_comparison"] = dict(
            status="diagnostic_only", no_empirical_acceptance_tolerance=True,
            maximum_native_mass_normalized_residual=float(np.max(np.abs(native_normalized))),
            maximum_reconstructed_mass_normalized_residual=float(np.max(np.abs(reconstructed_normalized))),
            maximum_normalized_residual_difference=float(np.max(np.abs(native_normalized - reconstructed_normalized))),
            maximum_relative_nodal_mass_difference=float(np.max(np.abs(mass[free] / balance.node_masses[balance.groups[free]] - 1))),
            native_residual_scope="captured float32 b-Ax before constraint solve",
            coefficient_model=balance.coefficient_model,
            reconstruction_scope="source-ordered float32 coefficient assembly with float64 residual evaluation; native topology preparation order remains separately scoped"
            if balance.coefficient_model == "native_float32" else
            "independent float64 undamped corotational assembly; equivalence requires compatible recorded teacher settings")
    else:
        report["operator_reconstruction_comparison"] = dict(status="unknown", reason="complete native solution dependencies unavailable or no free row")
    return report
