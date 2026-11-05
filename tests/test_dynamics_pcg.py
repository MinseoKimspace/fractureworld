import numpy as np
import pytest
from numpy.testing import assert_allclose
from scipy import sparse

from fracture_world.dynamics_corrector import (
    CorrectorTolerances, DynamicsProblem, LinearizedContact, _Factor, solve_central,
)
from fracture_world.dynamics_pcg import _SparseReduction, solve_central_pcg
from fracture_world.mesh import VolumeMesh


def _problem(**kwargs):
    mesh = VolumeMesh(np.array([[0, 0, 0], [1, 0, 0], [0, 1, 0], [0, 0, 1], [0, 0, -1]], dtype=float),
                      np.array([[0, 1, 2, 3], [0, 2, 1, 4]]))
    positions = mesh.nodes[mesh.tets]
    prediction = np.array([[[0.3, 0.2, -0.1]] * 4, [[-0.1, -0.2, 0.3]] * 4])
    values = dict(mesh=mesh, positions=positions, velocities=np.zeros_like(positions),
                  predicted_positions=positions + 0.05 * prediction, predicted_velocities=prediction,
                  dt=0.05, position_scale=1.0, velocity_scale=1.0)
    values.update(kwargs)
    return DynamicsProblem(**values)


def _compare(problem, atol=2e-7):
    tolerance = CorrectorTolerances(1e-10, 1e-9)
    expected = solve_central(problem, tolerance)
    actual = solve_central_pcg(problem, tolerance)
    assert expected.converged, expected.status
    assert actual.converged, (actual.status, actual.diagnostics["pcg"])
    assert actual.method == "central_pcg"
    assert actual.diagnostics["constraints_satisfied"]
    assert actual.diagnostics["pcg"]["optimality_certificate"]["satisfied"]
    assert_allclose(actual.objective, expected.objective, atol=1e-9, rtol=1e-8)
    assert_allclose(actual.positions, expected.positions, atol=atol, rtol=atol)
    assert_allclose(actual.velocities, expected.velocities, atol=atol, rtol=atol)
    return actual


@pytest.mark.parametrize("gap", [-0.01, 1.0])
def test_active_and_inactive_contacts_match_clarabel(gap):
    result = _compare(_problem(contacts=(LinearizedContact.pair(0, -1, [0, 0, 1], gap),)))
    assert result.diagnostics["pcg"]["pcg_iterations"] > 0
    assert result.diagnostics["linearized_contact_gaps"][0] >= -1e-9


def test_fixed_support_and_redundant_equalities_match_clarabel():
    fixed = np.zeros((2, 4), dtype=bool)
    fixed[:, 0] = True
    problem = _problem(fixed=fixed)
    equality = next(factor for factor in problem.factors if not factor.inequality)
    duplicate = _Factor("dependent_equality", equality.indices, 3 * equality.matrix, 3 * equality.bound)
    problem.factors = (*problem.factors, duplicate)
    result = _compare(problem)
    assert_allclose(result.positions[fixed], problem.positions[fixed], atol=1e-10)
    assert_allclose(result.velocities[fixed], 0, atol=1e-10)


def test_soft_residual_hessian_is_symmetric_and_matches_clarabel():
    problem = _problem()
    rng = np.random.default_rng(64)
    matrix = sparse.csr_matrix(rng.normal(size=(6, len(problem.prior))))
    soft = _Factor("soft", np.arange(len(problem.prior)), matrix, rng.normal(size=6), penalty=4.0)
    problem.factors = (*problem.factors, soft)
    result = _compare(problem)
    assert result.diagnostics["pcg"]["pcg_iterations"] > 1


def test_unconstrained_soft_problem_uses_pcg_without_clarabel_fallback(monkeypatch):
    problem = _problem()
    problem.broken_faces[problem.mesh.internal_faces] = True
    size = len(problem.prior)
    rng = np.random.default_rng(83)
    matrix = rng.normal(size=(5, size))
    bound = rng.normal(size=5)
    penalty = 3.0
    problem.factors = (_Factor("soft", np.arange(size), sparse.csr_matrix(matrix), bound, penalty=penalty),)
    expected = np.linalg.solve(np.diag(problem.prior_weights) + penalty * matrix.T @ matrix,
                               problem.prior_weights * problem.prior + penalty * matrix.T @ bound)

    def fail(*args, **kwargs):
        raise AssertionError("Clarabel fallback was called")

    monkeypatch.setattr("clarabel.DefaultSolver", fail)
    result = solve_central_pcg(problem, CorrectorTolerances(1e-10, 1e-9))
    assert result.converged, (result.status, result.diagnostics["pcg"])
    assert result.diagnostics["pcg"]["pcg_iterations"] > 0
    assert_allclose(problem.encode_state(result.positions, result.velocities), expected, atol=1e-9, rtol=1e-9)


def test_dependent_opposing_active_inequalities_have_nonnegative_certificate():
    problem = _problem()
    size = len(problem.prior)
    row = sparse.csr_matrix(([1.0], ([0], [0])), shape=(1, size))
    problem.prior = np.zeros(size)
    problem.prior[0] = -1.0
    problem.factors = (
        _Factor("lower", np.arange(size), row, np.zeros(1), True),
        _Factor("upper", np.arange(size), -row, np.zeros(1), True),
        _Factor("duplicate_lower", np.arange(size), 3 * row, np.zeros(1), True),
    )
    result = _compare(problem)
    assert result.diagnostics["pcg"]["optimality_certificate"]["minimum_inequality_multiplier"] >= 0


def test_ill_conditioned_positive_weights_keep_original_tolerances():
    problem = _problem()
    problem.prior_weights = np.geomspace(1e-7, 1e3, len(problem.prior))
    result = _compare(problem, atol=1e-5)
    certificate = result.diagnostics["pcg"]["optimality_certificate"]
    assert certificate["weighted_stationarity_residual"] <= certificate["stationarity_tolerance"]
    assert certificate["objective_gap_bound"] <= certificate["objective_gap_tolerance"]


def test_random_known_kkt_solution_with_duplicate_blockers():
    problem = _problem()
    problem.broken_faces[problem.mesh.internal_faces] = True
    size = len(problem.prior)
    rng = np.random.default_rng(2)
    expected = rng.normal(size=size)
    equality = rng.normal(size=(2, size))
    inequality = rng.normal(size=(16, size))
    slack = np.r_[np.zeros(5), rng.uniform(0.1, 2, 11), 0, 0]
    inequality = np.vstack((inequality, inequality[0], 2 * inequality[1]))
    weight = 1e-8
    dual = np.r_[rng.uniform(0.2, 1, 5), np.zeros(13)] * weight
    equality_dual = rng.normal(size=2) * weight
    problem.prior_weights = np.full(size, weight)
    problem.prior = expected + (equality.T @ equality_dual + inequality.T @ dual) / weight
    problem.factors = (
        _Factor("equality", np.arange(size), sparse.csr_matrix(equality), equality @ expected),
        _Factor("contacts", np.arange(size), sparse.csr_matrix(-inequality), -(inequality @ expected + slack), True),
    )
    result = solve_central_pcg(problem, CorrectorTolerances(1e-10, 1e-9))
    assert result.converged, (result.status, result.diagnostics["pcg"])
    actual = problem.encode_state(result.positions, result.velocities)
    assert_allclose(actual, expected, atol=1e-7, rtol=1e-7)


def test_sparse_implicit_nullspace_and_transpose_are_consistent():
    rng = np.random.default_rng(17)
    matrix = rng.normal(size=(5, 13))
    matrix = np.vstack((matrix, matrix[0] + 2 * matrix[3], 7 * matrix[2]))
    basis = _SparseReduction(sparse.csr_matrix(matrix), np.zeros(len(matrix)))
    free = rng.normal(size=len(basis.free))
    full = rng.normal(size=matrix.shape[1])
    assert len(basis.selected) == 5
    assert_allclose(matrix @ basis.expand(free), 0, atol=1e-12)
    assert_allclose(full @ basis.expand(free), basis.transpose(full) @ free, atol=1e-12)


def test_infeasible_contact_preserves_failure_and_original_certificate():
    problem = _problem(fixed=np.ones((2, 4), dtype=bool),
                       contacts=(LinearizedContact.pair(0, -1, [0, 0, 1], -0.01),))
    result = solve_central_pcg(problem)
    assert not result.converged
    assert result.status == "PrimalInfeasible"
    assert result.iterations == 0
    assert result.diagnostics["feasibility"]["status"] == "infeasible"


@pytest.mark.parametrize("budget", [0, 1])
def test_iteration_budget_never_claims_success(budget):
    result = solve_central_pcg(_problem(), max_iterations=budget)
    assert not result.converged
    assert result.status == "maximum_iterations"
    assert result.iterations == budget


def test_cancellation_before_and_during_pcg_is_not_success():
    assert solve_central_pcg(_problem(), cancel=lambda: True).status == "interrupted"
    interrupted = False

    def progress(event):
        nonlocal interrupted
        if event["stage"] == "pcg":
            interrupted = True

    result = solve_central_pcg(_problem(), cancel=lambda: interrupted, progress=progress)
    assert interrupted
    assert result.status == "interrupted"
    assert not result.converged


def test_actual_runtime_repairs_penetration_without_rounding_past_geometry_boundary():
    from test_dynamics_phase_runtime import phase_fixture
    from fracture_world.dynamics_runtime import _correct

    initial, model, proposal, config = phase_fixture(penetrated=True)
    result, report = _correct(initial, initial.positions, initial.velocities, proposal,
                              initial.broken, initial.fixed, model, config, "central_pcg",
                              previous_broken=initial.broken, previous_fixed=initial.fixed)
    assert result is not None, report
    assert result.converged and report["status"] == "converged"
    assert report["geometry"]["valid"]
    assert report["geometry"]["maximum_floor_penetration_metres"] <= config.geometry_tolerance
    assert report["swept_geometry"]["valid"]
    assert result.diagnostics["stabilization"]["status"] == "pass"
    assert result.diagnostics["energy"]["energy_satisfied"]
    subproblems = result.diagnostics["energy"]["linear_subproblems"]
    assert all(item["pcg"]["optimality_certificate"]["satisfied"] for item in subproblems)
    assert any(item["pcg"].get("publication_polish", {}).get("accepted") for item in subproblems)
