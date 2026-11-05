from copy import deepcopy

import numpy as np
import pytest

from fracture_world.dynamics_native_audit import native_solver_diagnostics
from test_dynamics_mechanics import balance_for, observed_rest_transition


def native_solver_case():
    initial, transition = observed_rest_transition()
    balance = balance_for(initial.topology.mesh)
    observations = deepcopy(transition.observations)
    observations.update(
        mpcg_solution=np.zeros((1, 4, 3), dtype=np.float32),
        mpcg_rhs=np.zeros((1, 4, 3), dtype=np.float32),
        mpcg_residual=np.zeros((1, 4, 3), dtype=np.float32),
        mpcg_observed=np.ones((1, 4), dtype=bool),
        mpcg_kinematic=np.zeros((1, 4), dtype=bool),
        mpcg_masses=balance.node_masses[balance.groups].reshape(1, 4).astype(np.float32),
        mpcg_recursive_norm_squared=np.zeros(1, dtype=np.float32),
        mpcg_true_norm_squared=np.zeros(1, dtype=np.float32),
        mpcg_rhs_norm_squared=np.zeros(1, dtype=np.float32),
        mpcg_tolerance_squared=np.zeros(1, dtype=np.float32),
        mpcg_epsilon=np.full(1, 1e-7, dtype=np.float32),
        mpcg_iterations=np.zeros(1, dtype=np.float32),
        mpcg_max_iterations=np.full(1, 100., dtype=np.float32),
        mpcg_rhs_positive=np.zeros(1, dtype=bool),
        mpcg_stopping_observed=np.ones(1, dtype=bool),
    )
    return observations, balance


def set_norms(observations, *, rhs=1., recursive=0., explicit=0.):
    observations["mpcg_rhs_norm_squared"][:] = rhs
    observations["mpcg_recursive_norm_squared"][:] = recursive
    observations["mpcg_true_norm_squared"][:] = explicit
    observations["mpcg_rhs_positive"][:] = rhs > 0
    observations["mpcg_tolerance_squared"][:] = (
        observations["mpcg_epsilon"]**2 * observations["mpcg_rhs_norm_squared"])


def test_complete_native_solver_certificate_preserves_independent_reconstruction_scope():
    observations, balance = native_solver_case()
    report = native_solver_diagnostics(observations, balance, 1.)
    assert report["status"] == report["equation_status"] == "pass"
    assert report["recursive_stopping_status"] == "unknown"
    assert report["zero_rhs_exact_solution_tetrahedra"] == [0]
    assert report["tolerance_arithmetic_status"] == "pass"
    assert report["solution_publication"]["status"] == "pass"
    assert report["systems"][0]["termination"] == "zero_rhs_early_return"
    assert report["physical_contact_law_status"] == "unknown"
    assert report["explicit_residual_is_stopping_rule"] is False
    assert report["engine_calls"] == 0
    comparison = report["operator_reconstruction_comparison"]
    assert comparison["status"] == "diagnostic_only"
    assert comparison["no_empirical_acceptance_tolerance"] is True


def test_converged_positive_rhs_native_iteration_passes_its_own_norm():
    observations, balance = native_solver_case()
    set_norms(observations, recursive=1e-16, explicit=1e-16)
    observations["mpcg_iterations"][:] = 3.
    report = native_solver_diagnostics(observations, balance, 1.)
    assert report["status"] == report["equation_status"] == "pass"
    assert report["recursive_stopping_status"] == "pass"
    assert report["systems"][0]["termination"] == "recursive_tolerance"


@pytest.mark.parametrize("missing", (None, "solver_fields", "single_step"))
def test_missing_or_sparse_native_solver_capture_is_unknown(missing):
    observations, balance = native_solver_case()
    if missing is None:
        observations = None
    elif missing == "solver_fields":
        observations = {name: value for name, value in observations.items() if not name.startswith("mpcg_")}
    else:
        observations["single_step_coverage"] = False
    report = native_solver_diagnostics(observations, balance, 1.)
    assert report["status"] == "unknown"
    assert report["equation_status"] == "unknown"
    assert report["reason"]


def test_explicit_residual_failure_is_not_hidden_by_recursive_convergence():
    observations, balance = native_solver_case()
    set_norms(observations, recursive=1e-16, explicit=1e-10)
    report = native_solver_diagnostics(observations, balance, 1.)
    assert report["status"] == "pass"
    assert report["recursive_stopping_status"] == "pass"
    assert report["equation_status"] == "fail"
    assert report["failed_stopping_tetrahedra"] == []
    assert report["failed_explicit_residual_tetrahedra"] == [0]
    assert report["systems"][0]["termination"] == "recursive_tolerance"


def test_exhausted_iterations_with_unconverged_native_residual_fails():
    observations, balance = native_solver_case()
    set_norms(observations, recursive=1e-8, explicit=1e-8)
    observations["mpcg_iterations"][:] = observations["mpcg_max_iterations"]
    report = native_solver_diagnostics(observations, balance, 1.)
    assert report["status"] == report["equation_status"] == "fail"
    assert report["recursive_stopping_status"] == "fail"
    assert report["failed_stopping_tetrahedra"] == [0]
    assert report["systems"][0]["termination"] == "iteration_limit"


@pytest.mark.parametrize("recursive", (0., 1e-12))
def test_zero_rhs_does_not_certify_nonzero_native_residual(recursive):
    observations, balance = native_solver_case()
    set_norms(observations, rhs=0., recursive=recursive, explicit=1e-12)
    report = native_solver_diagnostics(observations, balance, 1.)
    assert report["status"] == "fail"
    assert report["recursive_stopping_status"] == "unknown"
    assert report["equation_status"] == "fail"
    assert report["zero_rhs_nonzero_residual_tetrahedra"] == [0]
    assert report["systems"][0]["recursive_relative_norm"] is None
    assert report["systems"][0]["explicit_relative_norm"] is None


@pytest.mark.parametrize("partial", (False, True))
def test_corrupted_native_solution_publication_fails_even_with_partial_capture(partial):
    observations, balance = native_solver_case()
    observations["pre_contact_velocities"][0, 0, 0] = .25
    if partial:
        observations["mpcg_observed"][0, 1:] = False
        observations["mpcg_stopping_observed"][:] = False
        observations["contact_observed"][0, 1:] = False
    report = native_solver_diagnostics(observations, balance, 1.)
    assert report["status"] == "fail"
    assert report["solution_publication"]["status"] == "fail"
    assert report["solution_publication"]["maximum_velocity_difference_m_per_s"] == .25
    assert report["incomplete_observation"] is partial


@pytest.mark.parametrize("mask", ("corners", "stopping", "publication"))
def test_partial_native_capture_never_certifies_complete_transition(mask):
    observations, balance = native_solver_case()
    if mask == "corners":
        observations["mpcg_observed"][0, 1:] = False
        observations["mpcg_stopping_observed"][:] = False
    elif mask == "stopping":
        observations["mpcg_stopping_observed"][:] = False
    else:
        observations["contact_observed"][0, 1:] = False
    report = native_solver_diagnostics(observations, balance, 1.)
    assert report["status"] == "unknown"
    if mask in {"corners", "stopping"}:
        assert report["equation_status"] == "unknown"
        assert report["recursive_stopping_status"] == "unknown"
        assert report["tolerance_arithmetic_status"] == "unknown"
    if mask in {"corners", "publication"}:
        assert report["solution_publication"]["status"] == "unknown"


def test_captured_tolerance_must_match_native_float32_arithmetic_exactly():
    observations, balance = native_solver_case()
    set_norms(observations, recursive=0., explicit=0.)
    observations["mpcg_tolerance_squared"] = np.nextafter(
        observations["mpcg_tolerance_squared"], np.float32(np.inf))
    report = native_solver_diagnostics(observations, balance, 1.)
    assert report["recursive_stopping_status"] == "pass"
    assert report["equation_status"] == "pass"
    assert report["tolerance_arithmetic_status"] == "fail"
    assert report["status"] == "fail"


def test_unconverged_early_exit_is_not_mislabeled_as_exhausted_iterations():
    observations, balance = native_solver_case()
    set_norms(observations, recursive=1e-8, explicit=1e-8)
    observations["mpcg_iterations"][:] = 1.
    report = native_solver_diagnostics(observations, balance, 1.)
    assert report["status"] == "fail"
    assert report["systems"][0]["termination"] == "inconsistent_early_exit"


@pytest.mark.parametrize("field", ("mpcg_solution", "mpcg_residual", "mpcg_epsilon", "mpcg_true_norm_squared", "mpcg_masses"))
def test_observed_native_solver_values_must_preserve_float32_capture(field):
    observations, balance = native_solver_case()
    values = observations[field].astype(float)
    values.flat[0] = np.nextafter(values.flat[0], np.inf)
    observations[field] = values
    with pytest.raises(RuntimeError, match="float32"):
        native_solver_diagnostics(observations, balance, 1.)


def test_zero_rhs_early_return_cannot_claim_completed_iterations():
    observations, balance = native_solver_case()
    observations["mpcg_iterations"][:] = 1.
    with pytest.raises(RuntimeError, match="zero-RHS"):
        native_solver_diagnostics(observations, balance, 1.)
