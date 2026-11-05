import numpy as np
import pytest
from numpy.testing import assert_allclose

from fracture_world.dynamics_corrector import CorrectorTolerances, DynamicsProblem
from fracture_world.dynamics_energy import EnergyProjection
from fracture_world.dynamics_energy_solvers import solve_energy
from test_dynamics_energy import energy_fixture
from test_dynamics_frozen_energy import frozen_fixture


def expansion_problem():
    mesh, energy, positions, velocities = energy_fixture()
    velocity = .2 * (positions - positions.mean(axis=1, keepdims=True))
    problem = DynamicsProblem(mesh, positions, velocities, positions + .05 * velocity, velocity, .05,
                              density=energy.density, position_scale=1., velocity_scale=1.)
    return problem, EnergyProjection(problem, energy, 0.)


def test_pcg_energy_cuts_preserve_original_energy_and_central_objective():
    problem, projection = expansion_problem()
    projection.budget = .001
    tolerance = CorrectorTolerances(1e-9, 1e-8)
    central = solve_energy(problem, projection, "central", tolerance)
    events = []
    pcg = solve_energy(problem, projection, "central_pcg", tolerance, progress=events.append)
    assert central.converged, central.status
    assert pcg.converged, (pcg.status, pcg.diagnostics)
    assert pcg.method == "central_pcg"
    assert pcg.diagnostics["constraints_satisfied"]
    assert pcg.diagnostics["energy"]["energy_satisfied"]
    assert not pcg.diagnostics["energy"]["global_optimality_verified"]
    assert_allclose(pcg.objective, central.objective, atol=1e-8, rtol=1e-6)
    assert_allclose(pcg.positions, central.positions, atol=2e-5)
    assert_allclose(pcg.velocities, central.velocities, atol=2e-5)
    subproblems = pcg.diagnostics["energy"]["linear_subproblems"]
    assert len(subproblems) > 1
    assert all(item["pcg"] is not None for item in subproblems)
    assert all(event["method"] == "central_pcg" for event in events)
    assert (np.diff([event["runtime_seconds"] for event in events]) >= 0).all()
    assert any(event["phase"] == "linear_subproblem" for event in events if "phase" in event)


def test_zero_budget_energy_boundary_is_validated_without_relaxing_tolerance():
    problem, projection = expansion_problem()
    tolerance = projection.tolerance
    result = solve_energy(problem, projection, "central_pcg")
    assert result.converged, (result.status, result.diagnostics)
    state = problem.encode_state(result.positions, result.velocities)
    assert projection.state_value(state) <= tolerance
    assert projection.tolerance == tolerance
    assert result.diagnostics["energy"]["energy_satisfied"]
    assert result.diagnostics["constraints_satisfied"]


@pytest.mark.parametrize("deformed", [False, True])
def test_pcg_frozen_mechanics_keeps_soft_residual_and_shared_coordinates(deformed):
    balance, energy, positions, velocities = frozen_fixture(True, deformed, (.2, -.3, .1))
    predicted_velocity = velocities + .05 * (positions - positions.mean(axis=(0, 1), keepdims=True))
    problem = DynamicsProblem(balance.mesh, positions, velocities, positions + balance.dt * predicted_velocity,
                              predicted_velocity, balance.dt, density=3., position_scale=.2, velocity_scale=1.7,
                              external_impulse=balance.dt * balance.corner_masses.reshape(positions.shape[:2])[..., None] * balance.gravity,
                              elastic_balance=balance, mechanics_mode="proximal")
    projection = EnergyProjection(problem, energy, energy.value(positions, velocities))
    central = solve_energy(problem, projection, "central")
    pcg = solve_energy(problem, projection, "central_pcg")
    assert central.converged, central.status
    assert pcg.converged, (pcg.status, pcg.diagnostics)
    assert pcg.diagnostics["constraints_satisfied"]
    assert pcg.diagnostics["energy"]["energy_satisfied"]
    assert pcg.diagnostics["shared_coordinates"]["solver_state_variables"] < len(problem.prior)
    assert_allclose(pcg.objective, central.objective, atol=1e-8, rtol=1e-5)
    assert_allclose(pcg.positions, central.positions, atol=1e-6)
    assert_allclose(pcg.velocities, central.velocities, atol=1e-6)


@pytest.mark.parametrize("options,status", [({"max_iterations": 0}, "maximum_iterations"),
                                            ({"cancel": lambda: True}, "interrupted")])
def test_pcg_energy_does_not_accept_unfinished_solutions(options, status):
    problem, projection = expansion_problem()
    result = solve_energy(problem, projection, "central_pcg", **options)
    assert not result.converged
    assert result.status == status
    assert result.iterations == 0


def test_pcg_energy_infeasible_support_contact_remains_infeasible():
    from fracture_world.dynamics_corrector import LinearizedContact

    mesh, energy, positions, velocities = energy_fixture()
    problem = DynamicsProblem(mesh, positions, velocities, positions.copy(), velocities.copy(), .05,
                              density=energy.density, fixed=np.ones(positions.shape[:2], dtype=bool),
                              contacts=(LinearizedContact.pair(0, -1, [0., 0., 1.], -.01),),
                              position_scale=1., velocity_scale=1.)
    result = solve_energy(problem, EnergyProjection(problem, energy, 0.), "central_pcg")
    assert not result.converged
    assert result.status == "PrimalInfeasible"
