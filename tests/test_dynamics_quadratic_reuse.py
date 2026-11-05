from dataclasses import replace

import numpy as np
from numpy.testing import assert_allclose
import pytest

from fracture_world.dynamics_energy import EnergyProjection
from fracture_world.dynamics_frozen_energy import TransportEnergy
from fracture_world.dynamics_quadratic_projection import project_quadratic_energy, quadratic_energy_coefficients
from test_dynamics_quadratic_projection import quadratic_fixture


@pytest.mark.parametrize("shared", [False, True])
def test_changed_stabilization_reuses_only_invariant_hessian(shared):
    first = quadratic_fixture(shared=shared, transport=True)
    base = first.energy.energy
    original = quadratic_energy_coefficients(first)
    displacement = first.energy.stabilization + np.array([.003, -.002, .001])
    second = EnergyProjection(first.problem, TransportEnergy(base, displacement), first.budget,
                              energy_scale=first.energy_scale)
    changed = quadratic_energy_coefficients(second)
    assert changed.hessian is original.hessian
    assert not np.array_equal(changed.linear, original.linear)
    rng = np.random.default_rng(2201)
    state = second.problem.prior + rng.normal(0., .02, second.problem.prior.shape)
    assert_allclose(.5 * state @ (changed.hessian @ state) + changed.linear @ state + changed.constant,
                    second.state_value(state), atol=5e-15, rtol=2e-13)
    assert_allclose(changed.hessian @ state + changed.linear, second.state_gradient(state), atol=5e-15)


def test_changed_energy_scale_rebuilds_normalized_hessian():
    first = quadratic_fixture(transport=True)
    original = quadratic_energy_coefficients(first)
    second = replace(first, energy_scale=2 * first.energy_scale)
    changed = quadratic_energy_coefficients(second)
    assert changed.hessian is not original.hessian
    assert_allclose(changed.hessian.toarray(), .5 * original.hessian.toarray(), atol=2e-15)
    assert_allclose(changed.linear, .5 * original.linear, atol=2e-15)
    assert_allclose(changed.constant, .5 * original.constant, atol=2e-15)


def test_mutated_frozen_coefficients_do_not_reuse_stale_projection_model():
    projection = quadratic_fixture(transport=True)
    original = quadratic_energy_coefficients(projection)
    energy = projection.energy.energy
    energy.stiffness = energy.stiffness * 1.2
    energy.linear_term = energy.linear_term + np.linspace(.001, .003, len(energy.linear_term))
    changed = quadratic_energy_coefficients(projection)
    assert changed.hessian is not original.hessian
    state = projection.problem.prior + .03
    assert_allclose(changed.hessian @ state + changed.linear, projection.state_gradient(state), atol=5e-15)
    assert_allclose(.5 * state @ (changed.hessian @ state) + changed.linear @ state + changed.constant,
                    projection.state_value(state), atol=5e-15)


def test_changed_transport_offset_invalidates_affine_coefficients_without_rebuilding_hessian():
    projection = quadratic_fixture(transport=True)
    original = quadratic_energy_coefficients(projection)
    projection.energy.stabilization += .004
    changed = quadratic_energy_coefficients(projection)
    assert changed.hessian is original.hessian
    assert not np.array_equal(original.linear, changed.linear)
    state = projection.problem.prior + .01
    assert_allclose(changed.hessian @ state + changed.linear, projection.state_gradient(state), atol=5e-15)
    assert_allclose(.5 * state @ (changed.hessian @ state) + changed.linear @ state + changed.constant,
                    projection.state_value(state), atol=5e-15)


def test_repeated_cached_projection_preserves_original_energy_kkt_certificate():
    projection = quadratic_fixture(shared=True, transport=True)
    state = projection.problem.prior + np.random.default_rng(2203).normal(0., .2, projection.problem.prior.shape)
    state[len(state) // 2:] += .8
    first = project_quadratic_energy(projection, state)
    repeated = project_quadratic_energy(projection, state)
    assert first.converged and repeated.converged
    assert repeated.diagnostics["quadratic_coefficients_reused"]
    assert repeated.diagnostics["quadratic_hessian_reused"]
    assert repeated.diagnostics["optimizer_calls"] <= first.diagnostics["optimizer_calls"]
    assert_allclose(first.state, repeated.state, atol=2e-9, rtol=0.)
    for result in (first, repeated):
        assert result.diagnostics["global_projection_verified"]
        assert projection.state_value(result.state) <= .5 * projection.tolerance
        stationarity = result.state - state + result.diagnostics["multiplier"] * projection.state_gradient(result.state)
        assert np.max(np.abs(stationarity)) <= result.diagnostics["stationarity_tolerance"]
