from copy import copy

import numpy as np
from numpy.testing import assert_allclose
import pytest

from fracture_world import dynamics_corrector
from fracture_world.dynamics_corrector import CorrectorTolerances, DynamicsProblem, LinearizedContact, _shared_problem, solve_central, solve_consensus
from fracture_world.mesh import VolumeMesh


def shared_problem(drift=False):
    mesh = VolumeMesh([[0., 0., 0.], [1., 0., 0.], [0., 1., 0.], [0., 0., 1.], [1., 1., 1.]],
                      [[0, 1, 2, 3], [4, 1, 2, 3]])
    positions = mesh.nodes[mesh.tets].copy()
    if drift:
        positions[1, :, 0] += 1e-11
    random = np.random.default_rng(71)
    predicted_velocity = random.normal(scale=.1, size=positions.shape)
    predicted_position = positions + .02 * random.normal(scale=.1, size=positions.shape)
    return DynamicsProblem(mesh, positions, np.zeros_like(positions), predicted_position, predicted_velocity,
                           .02, position_scale=.02, velocity_scale=1., kinematic_rule="constant_acceleration")


@pytest.mark.parametrize("drift", [False, True])
def test_shared_coordinates_preserve_weighted_objective_and_all_constraint_violations(drift):
    original = shared_problem(drift)
    reduced = _shared_problem(original)
    random = np.random.default_rng(83)
    assert len(reduced.prior) < len(original.prior)
    assert reduced.objective_offset > 0
    assert_allclose(reduced.prior_weights.sum(), original.prior_weights.sum(), atol=1e-15)
    for state in (reduced.prior, random.normal(size=reduced.prior.shape)):
        expanded = state[reduced.state_indices] + reduced.offset
        assert_allclose(reduced.objective(state), original.objective(expanded), rtol=1e-13, atol=1e-14)
        factors = {factor.name: factor for factor in reduced.factors}
        for factor in original.factors:
            expected = np.max(factor.violation(expanded[factor.indices]), initial=0)
            name = reduced.metadata["factor_aliases"].get(factor.name, factor.name)
            actual = 0 if name == "exactly eliminated" else np.max(
                factors[name].violation(state[factors[name].indices]), initial=0)
            assert_allclose(actual, expected, rtol=1e-12, atol=1e-12)


def test_shared_prior_uses_mass_weighted_average_without_losing_prediction_disagreement_cost():
    original = shared_problem()
    reduced = _shared_problem(original)
    positions, velocities = reduced.unpack(reduced.prior)
    masses = original.corner_masses.ravel()
    for group in range(reduced.group_count):
        corners = np.flatnonzero(reduced.groups == group)
        expected_position = np.average(original.predicted_positions.reshape(-1, 3)[corners], axis=0, weights=masses[corners])
        expected_velocity = np.average(original.predicted_velocities.reshape(-1, 3)[corners], axis=0, weights=masses[corners])
        assert_allclose(positions.reshape(-1, 3)[corners], np.broadcast_to(expected_position, (len(corners), 3)), atol=1e-15)
        assert_allclose(velocities.reshape(-1, 3)[corners], np.broadcast_to(expected_velocity, (len(corners), 3)), atol=1e-15)
    assert_allclose(reduced.objective(reduced.prior), original.objective(original.encode_state(positions, velocities)), atol=1e-15)
    assert reduced.objective(reduced.prior) > 0


@pytest.mark.parametrize("solver", [solve_central, solve_consensus])
def test_shared_internal_contact_cannot_hide_an_impossible_positive_gap(solver):
    original = shared_problem()
    first, second = original.mesh.corner_face_pairs[0, 0]
    contact = LinearizedContact.pair(first, second, [1., 0., 0.], -.001)
    problem = DynamicsProblem(original.mesh, original.positions, original.velocities,
                              original.predicted_positions, original.predicted_velocities,
                              original.dt, contacts=(contact,), position_scale=.02, velocity_scale=1.)
    reduced = _shared_problem(problem)
    factor = next(factor for factor in reduced.factors if factor.name == "contact:0")
    assert factor.matrix.nnz == 0
    assert factor.bound[0] > 0
    result = solver(problem, max_iterations=2000)
    assert result.status == "PrimalInfeasible"
    assert not result.converged


def test_reduced_central_matches_independent_full_coordinate_quadratic_solution(monkeypatch):
    problem = shared_problem()
    tolerances = CorrectorTolerances(1e-10, 1e-9)
    reduced = solve_central(problem, tolerances)
    assert reduced.converged, reduced.status
    with monkeypatch.context() as context:
        context.setattr(dynamics_corrector, "_shared_problem", lambda value: value)
        full = solve_central(problem, tolerances)
    assert full.converged, full.status
    assert_allclose(reduced.positions, full.positions, atol=1e-9)
    assert_allclose(reduced.velocities, full.velocities, atol=1e-9)
    assert_allclose(reduced.objective, full.objective, rtol=1e-10, atol=1e-12)
    assert reduced.diagnostics["shared_coordinates"]["solver_state_variables"] < len(problem.prior)


def test_broken_faces_do_not_merge_independent_material_corners():
    original = shared_problem()
    broken = np.zeros(len(original.mesh.faces.vertices), dtype=bool)
    broken[original.mesh.internal_faces] = True
    problem = DynamicsProblem(original.mesh, original.positions, original.velocities,
                              original.predicted_positions, original.predicted_velocities, original.dt,
                              broken_faces=broken)
    assert _shared_problem(problem) is problem


def test_extra_reduced_constraints_are_not_skipped_by_original_state_validation():
    base = shared_problem()
    original = DynamicsProblem(base.mesh, base.positions, base.velocities, base.positions, base.velocities,
                               base.dt, position_scale=.02, velocity_scale=1.)
    reduced = copy(_shared_problem(original))
    cut = reduced._factor("additional_energy_cut", [0], [0], [1.], [.1], True)
    reduced.factors = (*reduced.factors, cut)
    result = solve_central(reduced)
    assert result.converged, result.status
    state = reduced.encode_state(result.positions, result.velocities)
    assert state[0] >= .1 - 1e-7
    assert result.diagnostics["factor_violations"]["additional_energy_cut"] <= result.diagnostics["constraint_tolerance"]
    assert result.objective > 0
