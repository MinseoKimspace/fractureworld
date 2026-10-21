import numpy as np
import pytest
from numpy.testing import assert_allclose
from scipy.spatial.transform import Rotation

from fracture_world.dynamics_corrector import DynamicsProblem, LinearizedContact, solve_consensus
from fracture_world.dynamics_energy import CorotationalEnergy, EnergyProjection
from fracture_world.dynamics_energy_solvers import solve_energy
from fracture_world.mesh import VolumeMesh


def energy_fixture(gravity=(0.0, 0.0, 0.0), poissons_ratio=0.25):
    mesh = VolumeMesh(np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]]), np.array([[0, 1, 2, 3]]))
    energy = CorotationalEnergy(mesh, 2.0, 100.0, poissons_ratio, np.array(gravity))
    positions = mesh.nodes[mesh.tets]
    velocities = np.zeros_like(positions)
    return mesh, energy, positions, velocities


@pytest.mark.parametrize("poissons_ratio", [0.25, -0.2])
def test_energy_gradient_agrees_with_finite_differences(poissons_ratio):
    _, energy, positions, velocities = energy_fixture((0.2, -9.81, 0.5), poissons_ratio)
    deformation = np.array([[1.1, 0.12, 0.04], [0.05, 0.9, -0.1], [0.1, 0.02, 1.2]])
    positions = positions @ deformation.T + np.array([0.2, 0.3, -0.1])
    velocities = np.random.default_rng(17).normal(size=velocities.shape)
    position_gradient, velocity_gradient = energy.gradient(positions, velocities)
    numerical = np.empty(positions.size + velocities.size)
    values = np.r_[positions.ravel(), velocities.ravel()]
    for index in range(len(values)):
        above, below = values.copy(), values.copy()
        above[index] += 1e-6
        below[index] -= 1e-6
        numerical[index] = (energy.value(above[:positions.size].reshape(positions.shape), above[positions.size:].reshape(velocities.shape)) -
                            energy.value(below[:positions.size].reshape(positions.shape), below[positions.size:].reshape(velocities.shape))) / 2e-6
    assert_allclose(np.r_[position_gradient.ravel(), velocity_gradient.ravel()], numerical, atol=2e-7, rtol=2e-6)


def test_rigid_rotation_has_zero_elastic_energy_and_gradient():
    _, energy, positions, velocities = energy_fixture()
    positions = Rotation.from_rotvec([0.6, -0.3, 0.8]).apply(positions.reshape(-1, 3)).reshape(positions.shape) + [2, 3, -1]
    components = energy.components(positions, velocities)
    assert abs(components["elastic"]) < 1e-25
    assert_allclose(energy.gradient(positions, velocities)[0], 0, atol=1e-12)


def test_ballistic_motion_preserves_kinetic_plus_gravity_potential():
    _, energy, positions, velocities = energy_fixture((0.0, -9.81, 0.0))
    velocities[:] = [0.3, 0.2, -0.1]
    dt = 0.1
    later_positions = positions + dt * velocities + 0.5 * dt**2 * energy.gravity
    later_velocity = velocities + dt * energy.gravity
    assert_allclose(energy.value(later_positions, later_velocity), energy.value(positions, velocities), atol=1e-12)


def projection_fixture(velocity=(1.0, 0.0, 0.0)):
    mesh, energy, positions, velocities = energy_fixture()
    predicted_velocity = np.broadcast_to(velocity, positions.shape).copy()
    problem = DynamicsProblem(mesh, positions, velocities, positions.copy(), predicted_velocity, 0.01,
                              density=energy.density, position_scale=1.0, velocity_scale=1.0)
    projection = EnergyProjection(problem, energy, energy.value(positions, velocities))
    return problem, projection


def test_zero_budget_rejects_and_corrects_spontaneous_kinetic_energy():
    problem, projection = projection_fixture()
    before = projection.diagnostics(problem.prior)
    assert not before["energy_satisfied"]
    assert before["components"]["kinetic"] > 0
    result = projection.project(problem.prior)
    assert result.converged, result.diagnostics
    assert result.diagnostics["energy_satisfied"]
    assert result.diagnostics["global_projection_verified"] is True
    assert result.status == "exact_zero_energy_projection"
    assert result.diagnostics["engine_calls"] == 0
    assert result.diagnostics["components"]["total"] < before["components"]["total"] * 1e-5


def test_normalized_energy_gradient_matches_state_finite_differences():
    problem, projection = projection_fixture()
    state = problem.prior.copy()
    state[:12] = np.random.default_rng(9).normal(0, 0.02, 12)
    gradient = projection.state_gradient(state)
    numerical = np.empty_like(state)
    for index in range(len(state)):
        above, below = state.copy(), state.copy()
        above[index] += 1e-6
        below[index] -= 1e-6
        numerical[index] = (projection.state_value(above) - projection.state_value(below)) / 2e-6
    assert_allclose(gradient, numerical, atol=1e-7, rtol=2e-6)


def test_energy_hessian_vector_matches_gradient_finite_differences():
    problem, projection = projection_fixture()
    rng = np.random.default_rng(9)
    state = problem.prior + rng.normal(0, 0.02, problem.prior.shape)
    direction = rng.normal(size=state.shape)
    actual = projection.state_hessian_vector(state, direction)
    expected = (projection.state_gradient(state + 1e-6 * direction) - projection.state_gradient(state - 1e-6 * direction)) / 2e-6
    assert_allclose(actual, expected, atol=1e-6, rtol=2e-6)


def test_feasible_state_projection_is_identity_and_cancellation_is_explicit():
    problem, projection = projection_fixture((0.0, 0.0, 0.0))
    result = projection.project(problem.prior)
    assert result.converged
    assert result.status == "already_feasible"
    assert_allclose(result.state, problem.prior, atol=0)
    interrupted = projection.project(problem.prior, cancel=lambda: True)
    assert not interrupted.converged
    assert interrupted.status == "interrupted"


def test_projection_solver_failure_is_unknown(monkeypatch):
    problem, projection = projection_fixture()
    projection.budget = 0.01

    def failure(*args, **kwargs):
        raise RuntimeError("deliberate numerical failure")

    monkeypatch.setattr("fracture_world.dynamics_energy.minimize", failure)
    result = projection.project(problem.prior)
    assert not result.converged
    assert result.status == "energy_projection_unknown"
    assert "deliberate numerical failure" in result.diagnostics["solver_message"]


def test_proper_rotation_polar_factor_does_not_make_reflection_stress_free():
    _, energy, positions, velocities = energy_fixture()
    reflected = positions.copy()
    reflected[..., 0] *= -1
    assert energy.components(reflected, velocities)["elastic"] > 1


@pytest.mark.parametrize("change", [{"density": 0.0}, {"youngs_modulus": -1.0}, {"poissons_ratio": 0.5}, {"gravity": [0.0, np.nan, 0.0]}])
def test_invalid_material_energy_is_rejected(change):
    mesh, _, _, _ = energy_fixture()
    values = dict(mesh=mesh, density=1.0, youngs_modulus=100.0, poissons_ratio=0.25, gravity=np.zeros(3))
    values.update(change)
    with pytest.raises(ValueError):
        CorotationalEnergy(**values)


@pytest.mark.parametrize("method", ["central", "consensus"])
def test_energy_constrained_solver_blocks_force_free_expansion(method):
    mesh, energy, positions, velocities = energy_fixture()
    center = positions.mean(axis=1, keepdims=True)
    predicted_velocity = 0.2 * (positions - center)
    dt = 0.05
    problem = DynamicsProblem(mesh, positions, velocities, positions + dt * predicted_velocity, predicted_velocity, dt,
                              density=energy.density, position_scale=1.0, velocity_scale=1.0)
    constraint = EnergyProjection(problem, energy, energy.value(positions, velocities))
    result = solve_energy(problem, constraint, method)
    assert result.converged, (result.status, result.diagnostics)
    assert result.diagnostics["constraints_satisfied"]
    assert result.diagnostics["energy"]["energy_satisfied"]
    assert result.diagnostics["energy"]["components"]["total"] < energy.value(problem.predicted_positions, problem.predicted_velocities) * 1e-3
    assert result.diagnostics["energy"]["global_optimality_verified"] is False


@pytest.mark.parametrize("method", ["central", "consensus"])
def test_energy_constrained_ballistic_state_is_not_distorted(method):
    mesh, energy, positions, velocities = energy_fixture((0.0, -9.81, 0.0))
    velocities[:] = [0.5, 0.3, -0.2]
    dt = 0.02
    next_velocity = velocities + dt * energy.gravity
    next_position = positions + dt * velocities + 0.5 * dt**2 * energy.gravity
    problem = DynamicsProblem(mesh, positions, velocities, next_position, next_velocity, dt, density=energy.density,
                              external_impulse=dt * energy.masses[..., None] * energy.gravity,
                              position_scale=1.0, velocity_scale=1.0, kinematic_rule="constant_acceleration")
    constraint = EnergyProjection(problem, energy, energy.value(positions, velocities))
    result = solve_energy(problem, constraint, method)
    assert result.converged, result.status
    assert_allclose(result.positions, next_position, atol=1e-8)
    assert_allclose(result.velocities, next_velocity, atol=1e-8)
    assert result.diagnostics["energy"]["energy_satisfied"]


@pytest.mark.parametrize("method", ["central", "consensus"])
def test_elastic_release_can_convert_strain_energy_into_motion(method):
    mesh, energy, positions, velocities = energy_fixture()
    rest = positions.copy()
    positions = rest.mean(axis=1, keepdims=True) + 1.1 * (rest - rest.mean(axis=1, keepdims=True))
    center = positions.mean(axis=1, keepdims=True)
    next_velocity = -0.1 * (positions - center)
    dt = 0.02
    next_position = positions + dt * next_velocity
    problem = DynamicsProblem(mesh, positions, velocities, next_position, next_velocity, dt, density=energy.density,
                              position_scale=1.0, velocity_scale=1.0)
    constraint = EnergyProjection(problem, energy, energy.value(positions, velocities))
    result = solve_energy(problem, constraint, method)
    assert result.converged, result.status
    assert result.diagnostics["energy"]["energy_satisfied"]
    assert result.diagnostics["energy"]["components"]["kinetic"] > 0
    assert result.diagnostics["energy"]["components"]["elastic"] < energy.components(positions, velocities)["elastic"]


def test_energy_local_optimizer_failure_is_not_counted_as_consensus_success(monkeypatch):
    mesh, energy, positions, velocities = energy_fixture()
    velocity = positions - positions.mean(axis=1, keepdims=True)
    problem = DynamicsProblem(mesh, positions, velocities, positions + 0.05 * velocity, velocity, 0.05,
                              density=energy.density, position_scale=1.0, velocity_scale=1.0)
    constraint = EnergyProjection(problem, energy, 0.01)

    def failure(*args, **kwargs):
        raise RuntimeError("forced local failure")

    monkeypatch.setattr("fracture_world.dynamics_energy.minimize", failure)
    result = solve_energy(problem, constraint, "consensus")
    assert not result.converged
    assert result.status == "energy_projection_unknown"


def test_positive_budget_uses_verified_nonlinear_local_projection():
    problem, constraint = projection_fixture()
    constraint.budget = 0.01
    result = constraint.project(problem.prior)
    assert result.converged, result.diagnostics
    assert result.diagnostics["energy_satisfied"]
    assert result.diagnostics["global_projection_verified"] is False
    assert result.diagnostics["optimizer_calls"] > 0
    assert constraint.stationarity_tolerance == 1e-10
    assert result.diagnostics["root_normal_distance"] <= result.diagnostics["root_normal_distance_tolerance"]
    assert result.diagnostics["proximal_set_normalized_slack"] == 0.5 * constraint.tolerance


def test_identity_and_projection_use_the_same_relaxed_energy_set():
    problem, constraint = projection_fixture()
    constraint.budget = 0.01
    state = problem.prior.copy()
    target_energy = constraint.budget + 0.75 * constraint.tolerance * constraint.energy_scale
    velocity = np.zeros_like(problem.velocities)
    velocity[..., 0] = np.sqrt(2 * target_energy / constraint.energy.masses.sum())
    state[3 * problem.corner_count:] = (velocity / problem.velocity_scale).ravel()
    assert 0.5 * constraint.tolerance < constraint.state_value(state) < constraint.tolerance
    result = constraint.project(state)
    assert result.converged, result.diagnostics
    assert result.status == "local_projection"
    assert result.diagnostics["energy_satisfied"]
    assert result.diagnostics["root_normal_distance"] <= result.diagnostics["root_normal_distance_tolerance"]


@pytest.mark.parametrize("method", ["central", "consensus"])
def test_nonlinear_energy_conflict_terminates_as_unknown_without_iteration_cap(method):
    mesh, energy, positions, velocities = energy_fixture()
    contact = LinearizedContact.pair(0, -1, [0, 1, 0], -0.01)
    problem = DynamicsProblem(mesh, positions, velocities, positions.copy(), velocities.copy(), 0.05,
                              density=energy.density, contacts=(contact,), position_scale=1.0, velocity_scale=1.0)
    constraint = EnergyProjection(problem, energy, 0.0)
    result = solve_energy(problem, constraint, method, max_iterations=None)
    assert not result.converged
    assert result.status in {"nonlinear_consensus_stagnation", "local_energy_model_infeasible_unknown", "energy_linearization_stagnation"}
    assert result.diagnostics["feasibility"]["status"] == "feasible"
    assert result.diagnostics["energy"]["global_optimality_verified"] is False


def test_negative_gravity_free_energy_budget_has_a_nonnegativity_certificate():
    problem, constraint = projection_fixture()
    constraint.budget = -1.0
    result = constraint.project(problem.prior)
    assert not result.converged
    assert result.status == "negative_energy_budget_infeasible"
    assert "nonnegative" in result.diagnostics["certificate"]


@pytest.mark.parametrize("method", ["central", "consensus"])
def test_energy_solver_reports_progress_without_changing_results(method):
    mesh, energy, positions, velocities = energy_fixture()
    velocity = 0.2 * (positions - positions.mean(axis=1, keepdims=True))
    problem = DynamicsProblem(mesh, positions, velocities, positions + 0.05 * velocity, velocity, 0.05,
                              density=energy.density, position_scale=1.0, velocity_scale=1.0)
    constraint = EnergyProjection(problem, energy, 0.0)
    reports = []
    monitored = solve_energy(problem, constraint, method, progress=reports.append)
    plain = solve_energy(problem, constraint, method)
    assert monitored.converged and plain.converged
    assert reports
    assert_allclose(monitored.positions, plain.positions, atol=0)
    assert_allclose(monitored.velocities, plain.velocities, atol=0)
    assert all(item["event"] == "energy_corrector_progress" and item["method"] == method for item in reports)
    assert all(item["runtime_seconds"] >= 0 for item in reports)
    assert all({"objective", "constraint_violation", "energy_violation", "primal_residual", "dual_residual", "rho"}.issubset(item) for item in reports)
    if method == "consensus":
        assert reports[0]["iteration"] == 1
        assert all(item["iteration"] % 25 == 0 for item in reports[1:])


def test_energy_acceptance_is_invariant_to_large_gravitational_origin_shift():
    mesh, energy, rest, velocities = energy_fixture((0.0, -9.81, 0.0))
    reports = []
    for height in (0.0, 1000000.0):
        positions = rest + [0.0, height, 0.0]
        predicted_velocity = np.broadcast_to([1.0, 0.0, 0.0], positions.shape).copy()
        problem = DynamicsProblem(mesh, positions, velocities, positions.copy(), predicted_velocity, 0.01,
                                  density=energy.density, position_scale=1.0, velocity_scale=1.0)
        projection = EnergyProjection(problem, energy, energy.value(positions, velocities))
        report = projection.diagnostics(problem.prior)
        shared = energy.balance(positions, velocities, positions, predicted_velocity, 1.0)
        assert not report["energy_satisfied"]
        assert_allclose(report["normalized_residual"], projection.state_value(problem.prior), atol=0)
        assert_allclose(report["normalized_residual"], shared["normalized_residual"], atol=0)
        reports.append(report)
    assert_allclose(reports[0]["energy_scale"], reports[1]["energy_scale"], atol=0)
    assert_allclose(reports[0]["normalized_tolerance"], reports[1]["normalized_tolerance"], atol=0)
    assert_allclose(reports[0]["normalized_residual"], reports[1]["normalized_residual"], atol=0)
    assert_allclose(reports[0]["energy_change"]["total"], 1 / 6, atol=1e-12)


def test_stable_ballistic_energy_change_does_not_subtract_large_potentials():
    mesh, energy, rest, velocities = energy_fixture((0.0, -9.81, 0.0))
    velocities[:] = [0.5, 0.3, -0.2]
    dt = 0.02
    residuals = []
    for height in (0.0, 1000000.0):
        positions = rest + [0.0, height, 0.0]
        predicted_velocity = velocities + dt * energy.gravity
        predicted_position = positions + dt * velocities + 0.5 * dt**2 * energy.gravity
        problem = DynamicsProblem(mesh, positions, velocities, predicted_position, predicted_velocity, dt,
                                  density=energy.density, position_scale=1.0, velocity_scale=1.0)
        constraint = EnergyProjection(problem, energy, energy.value(positions, velocities))
        residuals.append(constraint.state_value(problem.prior))
        assert constraint.diagnostics(problem.prior)["energy_satisfied"]
    assert_allclose(residuals, 0.0, atol=1e-8)


def test_energy_feasible_linear_relaxation_needs_no_nonlinear_projection(monkeypatch):
    mesh, energy, positions, velocities = energy_fixture()
    velocities[:] = [1.0, 0.0, 0.0]
    proposal_velocity = velocities + [0.1, 0.0, 0.0]
    problem = DynamicsProblem(mesh, positions, velocities, positions + 0.01 * proposal_velocity, proposal_velocity, 0.01,
                              density=energy.density, position_scale=1.0, velocity_scale=1.0)
    constraint = EnergyProjection(problem, energy, energy.value(positions, velocities))

    def forbidden(*args, **kwargs):
        raise AssertionError("inactive energy must not invoke the nonlinear proximal block")

    monkeypatch.setattr(constraint, "project", forbidden)
    result = solve_energy(problem, constraint, "consensus")
    assert result.converged, result.diagnostics
    assert result.diagnostics["energy"]["inactive_energy_certificate"]
    assert result.diagnostics["energy"]["energy_proximal_calls"] == 0
    assert result.iterations > 0
    assert len(result.history) == result.iterations
    assert len(result.diagnostics["energy"]["phases"]) == 1


def test_active_energy_preserves_linear_duals_and_counts_both_consensus_phases():
    mesh, energy, positions, velocities = energy_fixture()
    proposal_velocity = 0.2 * (positions - positions.mean(axis=1, keepdims=True)) + [0.1, 0.0, 0.0]
    problem = DynamicsProblem(mesh, positions, velocities, positions + 0.05 * proposal_velocity, proposal_velocity, 0.05,
                              density=energy.density, position_scale=1.0, velocity_scale=1.0)
    constraint = EnergyProjection(problem, energy, 0.0)
    result = solve_energy(problem, constraint, "consensus")
    assert result.converged, (result.status, result.diagnostics)
    phases = result.diagnostics["energy"]["phases"]
    assert [item["name"] for item in phases] == ["linear_consensus_relaxation", "nonlinear_energy_consensus"]
    assert all(item["iterations"] > 0 for item in phases)
    assert result.iterations == sum(item["iterations"] for item in phases)
    assert result.iterations == len(result.history)
    assert [item.iteration for item in result.history] == list(range(1, result.iterations + 1))
    assert phases[1]["initial_rho"] == phases[0]["final_rho"]
    assert phases[1]["initial_linear_scaled_dual_norm"] > 0
    assert phases[1]["initial_energy_scaled_dual_norm"] == 0
    assert result.diagnostics["energy"]["energy_satisfied"]
    assert result.runtime_seconds >= sum(item["runtime_seconds"] for item in phases)


def test_linear_relaxation_uses_the_same_total_iteration_budget():
    mesh, energy, positions, velocities = energy_fixture()
    proposal_velocity = 0.2 * (positions - positions.mean(axis=1, keepdims=True)) + [0.1, 0.0, 0.0]
    problem = DynamicsProblem(mesh, positions, velocities, positions + 0.05 * proposal_velocity, proposal_velocity, 0.05,
                              density=energy.density, position_scale=1.0, velocity_scale=1.0)
    constraint = EnergyProjection(problem, energy, 0.0)
    result = solve_energy(problem, constraint, "consensus", max_iterations=1)
    assert not result.converged
    assert result.status == "maximum_iterations"
    assert result.iterations == 1
    assert result.diagnostics["energy"]["energy_proximal_calls"] == 0


def test_active_energy_has_no_extra_iteration_after_linear_budget_is_exhausted():
    mesh, energy, positions, velocities = energy_fixture()
    proposal_velocity = 0.2 * (positions - positions.mean(axis=1, keepdims=True)) + [0.1, 0.0, 0.0]
    problem = DynamicsProblem(mesh, positions, velocities, positions + 0.05 * proposal_velocity, proposal_velocity, 0.05,
                              density=energy.density, position_scale=1.0, velocity_scale=1.0)
    relaxation = solve_consensus(problem)
    assert relaxation.converged and relaxation.iterations > 0
    constraint = EnergyProjection(problem, energy, 0.0)
    result = solve_energy(problem, constraint, "consensus", max_iterations=relaxation.iterations)
    assert not result.converged
    assert result.status == "maximum_iterations"
    assert result.iterations == relaxation.iterations
    assert result.diagnostics["energy"]["energy_proximal_calls"] == 0
    assert result.diagnostics["energy"]["phases"][1]["iterations"] == 0
