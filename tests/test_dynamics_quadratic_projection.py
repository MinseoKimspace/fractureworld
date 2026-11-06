import numpy as np
import pytest
from numpy.testing import assert_allclose
from scipy import sparse
from scipy.optimize import minimize

from fracture_world.dynamics_corrector import DynamicsProblem, _shared_problem
from fracture_world.dynamics_energy import CorotationalEnergy, EnergyProjection
from fracture_world.dynamics_frozen_energy import FrozenQuadraticEnergy, TransportEnergy
from fracture_world.dynamics_quadratic_projection import project_quadratic_energy, quadratic_energy_coefficients
from test_dynamics_frozen_energy import frozen_fixture


def quadratic_fixture(shared=False, transport=False):
    balance, energy, positions, velocities = frozen_fixture(True, True, (.2, -.3, .1))
    source = DynamicsProblem(balance.mesh, positions, velocities, positions.copy(), velocities.copy(), balance.dt,
                              density=balance.material["rest_density"], position_scale=.2, velocity_scale=1.7)
    problem = _shared_problem(source) if shared else source
    if transport:
        energy = TransportEnergy(energy, np.broadcast_to([.01, -.02, .03], positions.shape))
    projection = EnergyProjection(problem, energy, energy.initial_components(positions, velocities)["total"]
                                   if transport else energy.value(positions, velocities))
    return projection


@pytest.mark.parametrize("shared", [False, True])
@pytest.mark.parametrize("transport", [False, True])
def test_sparse_quadratic_coefficients_match_original_energy_derivatives_and_coordinate_offsets(shared, transport):
    projection = quadratic_fixture(shared, transport)
    model = quadratic_energy_coefficients(projection)
    rng = np.random.default_rng(149)
    state = projection.problem.prior + rng.normal(0., .04, projection.problem.prior.shape)
    direction = rng.normal(size=state.shape)
    expected = .5 * state @ (model.hessian @ state) + model.linear @ state + model.constant
    assert_allclose(expected, projection.state_value(state), atol=3e-15, rtol=2e-14)
    assert_allclose(model.hessian @ state + model.linear, projection.state_gradient(state), atol=3e-15, rtol=2e-14)
    assert_allclose(model.hessian @ direction, projection.state_hessian_vector(state, direction), atol=2e-14, rtol=2e-14)
    projection.budget += .1
    changed = quadratic_energy_coefficients(projection)
    assert_allclose(changed.constant - model.constant, -.1 / projection.energy_scale, atol=1e-15)


@pytest.mark.parametrize("shared", [False, True])
@pytest.mark.parametrize("transport", [False, True])
def test_quadratic_projection_has_original_energy_kkt_and_independent_lagrangian_minimum(shared, transport):
    projection = quadratic_fixture(shared, transport)
    rng = np.random.default_rng(151)
    state = projection.problem.prior + rng.normal(0., .2, projection.problem.prior.shape)
    state[len(state) // 2:] += .8
    result = project_quadratic_energy(projection, state)
    assert result.converged, result.diagnostics
    assert result.status == "quadratic_projection"
    assert projection.state_value(result.state) <= .5 * projection.tolerance
    assert result.diagnostics["global_projection_verified"] is True
    assert result.diagnostics["source_hessian_assumed_psd"] is False
    assert result.diagnostics["eigenvalues_clipped"] is False
    multiplier = result.diagnostics["multiplier"]
    stationarity = result.state - state + multiplier * projection.state_gradient(result.state)
    assert np.max(np.abs(stationarity)) <= result.diagnostics["stationarity_tolerance"]
    assert result.diagnostics["complementarity_residual"] <= result.diagnostics["complementarity_tolerance"]
    assert result.diagnostics["positive_definiteness"]["equal_row_column_permutations"]
    assert result.diagnostics["positive_definiteness"]["positive_pivots"]

    def lagrangian(value):
        difference = value - state
        return (.5 * float(difference @ difference) + multiplier * projection.state_value(value),
                difference + multiplier * projection.state_gradient(value))

    reference = minimize(lagrangian, state, jac=True, method="BFGS", options={"gtol": 1e-12})
    assert_allclose(result.state, reference.x, atol=2e-8, rtol=2e-8)
    assert_allclose(lagrangian(result.state)[0], lagrangian(reference.x)[0], atol=2e-13)


def indefinite_projection(negative_curvature):
    balance, _, positions, velocities = frozen_fixture()
    velocities = np.zeros_like(velocities)
    balance.stiffness = sparse.diags(np.r_[-negative_curvature, np.zeros(positions.size - 1)], format="csr")
    balance.elastic_gradient[:] = 0.
    energy = FrozenQuadraticEnergy(balance)
    problem = DynamicsProblem(balance.mesh, positions, velocities, positions.copy(), velocities.copy(), balance.dt,
                               density=balance.material["rest_density"], position_scale=1., velocity_scale=1.)
    projection = EnergyProjection(problem, energy, 0., energy_scale=1.)
    state = problem.prior.copy()
    state[positions.size] = 1.
    return projection, state


def test_small_negative_source_mode_does_not_create_false_zero_budget_infeasibility():
    projection, state = indefinite_projection(1e-8)
    model = quadratic_energy_coefficients(projection)
    assert model.hessian.diagonal().min() < 0.
    result = project_quadratic_energy(projection, state)
    assert result.converged, result.diagnostics
    assert projection.state_value(result.state) <= .5 * projection.tolerance
    assert result.diagnostics["eigenvalues_clipped"] is False


def test_indefinite_projection_inside_positive_definite_lagrangian_domain_is_certified():
    projection, state = indefinite_projection(3.)
    state[0] = .1
    result = project_quadratic_energy(projection, state)
    assert result.converged, result.diagnostics
    assert result.diagnostics["global_projection_verified"]
    assert result.diagnostics["multiplier"] < 1 / 3.
    assert projection.state_value(result.state) <= .5 * projection.tolerance


def test_uncertified_nonconvex_hard_case_returns_unknown_without_changing_negative_mode():
    projection, state = indefinite_projection(10.)
    original = projection.energy.stiffness.copy()
    result = project_quadratic_energy(projection, state)
    assert not result.converged
    assert result.status.endswith("unknown")
    assert result.diagnostics["global_projection_verified"] is False
    assert_allclose(projection.energy.stiffness.toarray(), original.toarray(), atol=0.)
    assert result.diagnostics["eigenvalues_clipped"] is False


def test_stationary_indefinite_hard_case_is_unknown_even_when_a_feasible_branch_exists():
    projection, state = indefinite_projection(10.)
    projection.budget = -.1
    state[:] = 0.
    result = project_quadratic_energy(projection, state)
    assert not result.converged
    assert result.status == "quadratic_stationary_hard_case_unknown"
    feasible = state.copy()
    feasible[0] = 1.
    assert projection.state_value(feasible) < 0.


def test_nonfinite_quadratic_energy_cannot_enter_multiplier_bracket():
    projection, state = indefinite_projection(1e-120)
    projection.energy.stiffness = sparse.diags(
        np.r_[-1e-120, 1e-120, np.zeros(projection.energy.stiffness.shape[0] - 2)], format="csr")
    state[:] = 0.
    state[0] = state[1] = 1e221
    checks = 0

    def cancel():
        nonlocal checks
        checks += 1
        return checks > 1

    with np.errstate(over="ignore", invalid="ignore"):
        model = quadratic_energy_coefficients(projection)
        assert np.isfinite(model.hessian.data).all() and np.isfinite(model.linear).all() and np.isfinite(model.constant)
        assert np.isnan(projection.state_value(state))
        gradient = projection.state_gradient(state)
        assert np.isfinite(gradient @ gradient)
        with pytest.raises(FloatingPointError, match="nonfinite quadratic initial energy"):
            project_quadratic_energy(projection, state, cancel)
    assert checks == 1


@pytest.mark.parametrize("value", [np.nan, np.inf])
def test_nonfinite_quadratic_initial_gradient_is_rejected(monkeypatch, value):
    projection, state = indefinite_projection(1e-8)
    original = projection.state_gradient
    monkeypatch.setattr(projection, "state_gradient", lambda candidate:
                        np.full_like(candidate, value) if np.array_equal(candidate, state) else original(candidate))
    with pytest.raises(FloatingPointError, match="nonfinite quadratic initial gradient"):
        project_quadratic_energy(projection, state)


def test_nonfinite_quadratic_initial_multiplier_is_rejected():
    projection, state = indefinite_projection(1e-8)
    projection.budget = -1e100
    state[:] = 0.
    state[len(state) // 2] = 1e-150
    with np.errstate(over="ignore"):
        with pytest.raises(FloatingPointError, match="nonfinite quadratic initial multiplier"):
            project_quadratic_energy(projection, state)


def test_quadratic_projection_identity_cancellation_and_failure_are_explicit(monkeypatch):
    projection = quadratic_fixture()
    feasible = projection.problem.prior.copy()
    result = project_quadratic_energy(projection, feasible)
    assert result.converged and result.status == "already_feasible"
    assert_allclose(result.state, feasible, atol=0.)
    interrupted = project_quadratic_energy(projection, feasible, cancel=lambda: True)
    assert not interrupted.converged and interrupted.status == "interrupted"
    invalid = feasible.copy()
    invalid[0] = np.nan
    with pytest.raises(ValueError):
        project_quadratic_energy(projection, invalid)

    def failure(*args):
        raise ArithmeticError("deliberate factorization failure")

    monkeypatch.setattr("fracture_world.dynamics_quadratic_projection._positive_factor", failure)
    infeasible = feasible.copy()
    infeasible[len(infeasible) // 2:] += 1.
    failed = project_quadratic_energy(projection, infeasible)
    assert not failed.converged and failed.status == "quadratic_projection_unknown"
    assert "deliberate factorization failure" in failed.diagnostics["reason"]


def test_nonlinear_energy_is_not_silently_replaced_by_a_quadratic_model():
    projection = quadratic_fixture()
    energy = projection.energy
    nonlinear = CorotationalEnergy(energy.mesh, energy.density, energy.youngs_modulus, energy.poissons_ratio, energy.gravity)
    projection = EnergyProjection(projection.problem, nonlinear, 0.)
    assert project_quadratic_energy(projection, projection.problem.prior) is None
