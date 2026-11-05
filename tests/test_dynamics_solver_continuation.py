from dataclasses import replace

import numpy as np
from numpy.testing import assert_allclose
import pytest

from fracture_world.dynamics_corrector import CorrectorTolerances, DynamicsProblem, LinearizedContact, _shared_problem, solve_central, solve_consensus
from fracture_world.dynamics_energy import EnergyProjection
from fracture_world.dynamics_energy_solvers import solve_energy
from fracture_world.dynamics_frozen_energy import TransportEnergy
from fracture_world.dynamics_solver_workspace import _digest, consensus_setup, coordinate_key, save_consensus, translate_consensus
from test_dynamics_corrector import mesh_pair, problem_for
from test_dynamics_frozen_energy import frozen_fixture


def active_quadratic_problem(offset=0.):
    balance, energy, positions, velocities = frozen_fixture(shared=True, deformed=True)
    proposed_velocity = velocities + 2. * (positions - positions.mean(axis=(0, 1), keepdims=True))
    problem = DynamicsProblem(balance.mesh, positions, velocities, positions + balance.dt * proposed_velocity,
                              proposed_velocity, balance.dt, density=energy.density, position_scale=.2, velocity_scale=1.7)
    wrapper = TransportEnergy(energy, np.full_like(positions, offset))
    projection = EnergyProjection(problem, wrapper, wrapper.initial_components(positions, velocities)["total"] + .2)
    return problem, projection


def test_linear_continuation_reuses_solution_duals_and_operators_without_relaxing_tolerances():
    problem = problem_for(mesh_pair())
    workspace = {}
    cold = solve_consensus(problem, rho=.01, _workspace=workspace)
    assert cold.converged
    previous_rho = workspace["rho"]
    warm = solve_consensus(problem, rho=.01, _workspace=workspace)
    assert warm.converged
    assert warm.iterations < cold.iterations
    assert warm.diagnostics["continuation"]["state_reused"]
    assert warm.diagnostics["continuation"]["operators_reused"]
    assert warm.diagnostics["continuation"]["reset_factor_duals"] == 0
    assert warm.history[0].rho == previous_rho
    assert_allclose(warm.positions, cold.positions, atol=2.e-7, rtol=0.)
    assert_allclose(warm.velocities, cold.velocities, atol=2.e-7, rtol=0.)
    assert_allclose(warm.objective, cold.objective, atol=1.e-8, rtol=0.)


def test_changed_linear_bounds_are_resolved_and_do_not_reuse_a_success_certificate():
    problem = problem_for(mesh_pair())
    tolerance = CorrectorTolerances(1.e-10, 1.e-9)
    workspace = {}
    assert solve_consensus(problem, rho=.01, tolerances=tolerance, _workspace=workspace).converged
    impulse = np.zeros_like(problem.positions)
    impulse[..., 0] = .03
    changed = replace(problem, external_impulse=impulse)
    warm = solve_consensus(changed, rho=.01, tolerances=tolerance, _workspace=workspace)
    central = solve_central(changed, tolerance)
    assert warm.converged and central.converged
    assert warm.iterations > 0
    assert warm.diagnostics["constraints_satisfied"]
    assert_allclose(warm.positions, central.positions, atol=3.e-7, rtol=0.)
    assert_allclose(warm.velocities, central.velocities, atol=3.e-7, rtol=0.)
    assert_allclose(warm.objective, central.objective, atol=1.e-7, rtol=0.)


def test_changed_contact_matrix_resets_only_affected_dual_blocks():
    source = problem_for(mesh_pair(), contacts=(LinearizedContact.pair(0, -1, [0., 1., 0.], 0.),))
    workspace = {}
    assert solve_consensus(source, _workspace=workspace).converged
    changed = _shared_problem(replace(source, contacts=(LinearizedContact.pair(0, -1, [1., 0., 0.], 0.),)))
    _, _, dual, operators, info = consensus_setup(changed, workspace, .01)
    assert info["state_reused"]
    assert info["reset_factor_duals"] > 0
    start = 0
    for factor in changed.factors:
        size = len(factor.indices)
        if factor.name == "contact:0":
            assert_allclose(dual[start:start + size], 0., atol=0.)
        start += size
    assert len(dual) == len(operators["indices"])


def test_topology_coordinate_change_discards_the_old_state_and_duals():
    source = problem_for(mesh_pair())
    workspace = {}
    assert solve_consensus(source, _workspace=workspace).converged
    broken = source.broken_faces.copy()
    broken[source.mesh.internal_faces] = True
    changed = _shared_problem(replace(source, broken_faces=broken, previous_broken_faces=source.broken_faces))
    state, _, dual, _, info = consensus_setup(changed, workspace, .01)
    assert not info["state_reused"]
    assert info["reused_factor_duals"] == 0
    assert_allclose(state, changed.prior, atol=0.)
    assert_allclose(dual, 0., atol=0.)


@pytest.mark.parametrize("stop", ["cancel", "zero_budget"])
def test_a_saved_linear_state_does_not_override_cancellation_or_zero_budget(stop):
    problem = problem_for(mesh_pair())
    workspace = {}
    assert solve_consensus(problem, _workspace=workspace).converged
    options = dict(cancel=lambda: True) if stop == "cancel" else dict(max_iterations=0)
    result = solve_consensus(problem, _workspace=workspace, **options)
    assert not result.converged
    assert result.iterations == 0
    assert result.status == ("interrupted" if stop == "cancel" else "maximum_iterations")


def test_active_quadratic_energy_resumes_the_full_admm_state_without_repeating_linear_relaxation():
    problem, projection = active_quadratic_problem()
    workspace = {}
    cold = solve_energy(problem, projection, "consensus", rho=.01, workspace=workspace)
    assert cold.converged, cold.status
    assert cold.diagnostics["energy"].get("energy_proximal_calls", 0) > 0
    warm_problem, warm_projection = active_quadratic_problem()
    warm = solve_energy(warm_problem, warm_projection, "consensus", rho=.01, workspace=workspace)
    assert warm.converged, warm.status
    assert warm.iterations < cold.iterations
    assert warm.diagnostics["energy"]["continued_from_previous_solve"]
    assert warm.diagnostics["energy"]["phases"][0]["name"] == "continued_energy_consensus"
    assert warm.diagnostics["energy"]["energy_satisfied"]
    assert warm.diagnostics["constraints_satisfied"]
    assert_allclose(warm.positions, cold.positions, atol=5.e-7, rtol=0.)
    assert_allclose(warm.velocities, cold.velocities, atol=5.e-7, rtol=0.)
    assert_allclose(warm.objective, cold.objective, atol=2.e-7, rtol=0.)


def test_changed_transport_energy_is_revalidated_after_warm_start():
    problem, projection = active_quadratic_problem()
    workspace = {}
    assert solve_energy(problem, projection, "consensus", rho=.01, workspace=workspace).converged
    changed, changed_projection = active_quadratic_problem(offset=1.e-4)
    warm = solve_energy(changed, changed_projection, "consensus", rho=.01, workspace=workspace)
    cold = solve_energy(changed, changed_projection, "consensus", rho=.01)
    assert warm.converged and cold.converged
    assert warm.diagnostics["energy"]["continued_from_previous_solve"]
    assert warm.diagnostics["energy"]["energy_satisfied"]
    assert_allclose(warm.positions, cold.positions, atol=1.e-6, rtol=0.)
    assert_allclose(warm.velocities, cold.velocities, atol=1.e-6, rtol=0.)
    assert_allclose(warm.objective, cold.objective, atol=5.e-7, rtol=0.)


@pytest.mark.parametrize("stop", ["cancel", "zero_budget"])
def test_saved_energy_continuation_never_bypasses_cancellation_or_zero_budget(stop):
    problem, projection = active_quadratic_problem()
    workspace = {}
    assert solve_energy(problem, projection, "consensus", workspace=workspace).converged
    options = dict(cancel=lambda: True) if stop == "cancel" else dict(max_iterations=0)
    result = solve_energy(problem, projection, "consensus", workspace=workspace, **options)
    assert not result.converged and result.iterations == 0
    assert result.status == ("interrupted" if stop == "cancel" else "maximum_iterations")


def test_stabilization_translation_preserves_transport_energy_and_saved_velocities():
    problem, projection = active_quadratic_problem()
    shared = _shared_problem(problem)
    state, rho, dual, operators, _ = consensus_setup(shared, None, .01)
    state = state + np.random.default_rng(81).normal(0., .001, state.shape)
    workspace = dict(linear={}, nonlinear={})
    for name in workspace:
        save_consensus(shared, workspace[name], state, dual, rho, operators)
    workspace["nonlinear"]["energy_scaled_dual"] = np.ones_like(state)
    old_projection = EnergyProjection(shared, projection.energy, projection.budget)
    delta = np.broadcast_to([1.e-4, -2.e-4, 3.e-4], problem.positions.shape).copy()
    changed = replace(problem, stabilization_offset=delta)
    changed_energy = TransportEnergy(projection.energy.energy, delta)
    changed_projection = EnergyProjection(_shared_problem(changed), changed_energy, projection.budget)
    report = translate_consensus(workspace, problem, changed, delta)
    assert report["translated_states"] == 2
    assert report["status"] == "translated"
    for name in ("linear", "nonlinear"):
        translated = workspace[name]["state"]
        positions, velocities = shared.unpack(state)
        next_positions, next_velocities = changed_projection.problem.unpack(translated)
        assert_allclose(next_positions - positions, delta, atol=2.e-16, rtol=0.)
        assert_allclose(next_velocities, velocities, atol=0., rtol=0.)
        assert_allclose(changed_projection.state_value(translated), old_projection.state_value(state), atol=2.e-15, rtol=0.)
    assert_allclose(workspace["nonlinear"]["energy_scaled_dual"], 1., atol=0., rtol=0.)


def test_stabilization_translation_refuses_incompatible_material_coordinates():
    source = problem_for(mesh_pair())
    workspace = dict(linear={})
    shared = _shared_problem(source)
    state, rho, dual, operators, _ = consensus_setup(shared, None, .01)
    save_consensus(shared, workspace["linear"], state, dual, rho, operators)
    broken = source.broken_faces.copy()
    broken[source.mesh.internal_faces] = True
    changed = replace(source, broken_faces=broken, previous_broken_faces=source.broken_faces)
    result = translate_consensus(workspace, source, changed, np.zeros_like(source.positions))
    assert result["status"] == "incompatible_coordinates"
    assert result["translated_states"] == 0
    assert_allclose(workspace["linear"]["state"], state, atol=0., rtol=0.)


def test_coordinate_signature_matches_exact_shared_affine_map_without_reducing_factors(monkeypatch):
    source = problem_for(mesh_pair())
    shared = _shared_problem(source)
    expected = _digest((len(shared.prior), shared.position_scale, shared.velocity_scale,
                        np.asarray(shared.positions), np.asarray(shared.velocities), shared.state_indices,
                        shared.offset, shared.reference_positions))

    def forbidden(*args, **kwargs):
        raise AssertionError("coordinate signatures must not rebuild reduced constraint factors")

    monkeypatch.setattr(type(shared), "_reduced_factors", forbidden)
    assert coordinate_key(source) == coordinate_key(shared) == expected
