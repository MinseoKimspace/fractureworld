import numpy as np
import pytest
from numpy.testing import assert_allclose

from fracture_world.dynamics_corrector import (
    CorrectorTolerances,
    DynamicsProblem,
    _diagnostics,
    solve_central,
)
from fracture_world.dynamics_energy import CorotationalEnergy, EnergyProjection
from fracture_world.dynamics_energy_solvers import solve_energy
from fracture_world.dynamics_geometry import geometry_diagnostics
from fracture_world.mesh import VolumeMesh


def connected_mesh():
    return VolumeMesh(
        np.array([[0., 0., 0.], [1., 0., 0.], [0., 1., 0.], [0., 0., 1.], [0., 0., -1.]]),
        np.array([[0, 1, 2, 3], [0, 2, 1, 4]]),
    )


def solve_linear(problem, method):
    tolerance = CorrectorTolerances(1e-8, 1e-7)
    return solve_central(problem, tolerance, max_iterations=20000)


def predicted_state(positions, velocities, gravity, dt, frame):
    coordinates = np.arange(positions.size).reshape(positions.shape)
    velocity_residual = .003 * np.sin(.71 * coordinates + .23 * frame)
    position_residual = .002 * np.cos(.41 * coordinates + .37 * frame)
    predicted_velocity = velocities + dt * gravity + velocity_residual
    predicted_position = positions + dt * velocities + .5 * dt**2 * gravity + dt * position_residual
    return predicted_position, predicted_velocity


def make_problem(mesh, positions, velocities, prediction, gravity, dt, broken, previous_broken):
    mass = np.broadcast_to(mesh.volumes[:, None] / 4, positions.shape[:2])
    return DynamicsProblem(
        mesh, positions, velocities, *prediction, dt,
        broken_faces=broken, previous_broken_faces=previous_broken,
        external_impulse=mass[..., None] * dt * gravity,
        position_scale=dt, velocity_scale=1., kinematic_rule="constant_acceleration",
    )


def assert_exact_connections(mesh, positions, velocities, broken):
    pairs = mesh.corner_face_pairs[~broken[mesh.internal_faces]].reshape(-1, 2)
    for values in (positions, velocities):
        flat = values.reshape(-1, 3)
        assert np.array_equal(flat[pairs[:, 0]], flat[pairs[:, 1]])


@pytest.mark.parametrize("method", ["central"])
def test_perturbed_predictions_remain_feasible_across_two_hundred_shared_state_steps(method):
    mesh = connected_mesh()
    positions = mesh.nodes[mesh.tets] + [0., 2., 0.]
    velocities = np.broadcast_to([.2, -.1, .05], positions.shape).copy()
    gravity = np.array([0., -9.81, 0.])
    broken = np.zeros(len(mesh.faces.vertices), dtype=bool)
    dt = 1e-4
    for frame in range(200):
        prediction = predicted_state(positions, velocities, gravity, dt, frame)
        problem = make_problem(mesh, positions, velocities, prediction, gravity, dt, broken, broken)
        result = solve_linear(problem, method)
        assert result.converged, (frame, result.status, result.diagnostics)
        assert result.diagnostics["constraints_satisfied"]
        assert result.diagnostics["free_momentum_islands_verified"]
        assert result.objective > 0
        assert_exact_connections(mesh, result.positions, result.velocities, broken)
        assert_allclose(result.positions, positions + dt * result.velocities - .5 * dt**2 * gravity,
                        atol=2e-10, rtol=0)
        previous_momentum = np.sum(problem.corner_masses[..., None] * velocities, axis=(0, 1))
        momentum = np.sum(problem.corner_masses[..., None] * result.velocities, axis=(0, 1))
        assert_allclose(momentum, previous_momentum + problem.corner_masses.sum() * dt * gravity,
                        atol=5e-8, rtol=0)
        positions, velocities = result.positions, result.velocities


@pytest.mark.parametrize("method", ["central"])
def test_shared_state_rollout_opens_fracture_without_retying_later_steps(method):
    mesh = connected_mesh()
    positions = mesh.nodes[mesh.tets] + [0., 2., 0.]
    velocities = np.zeros_like(positions)
    gravity = np.array([0., -9.81, 0.])
    broken = np.zeros(len(mesh.faces.vertices), dtype=bool)
    dt = 1e-4
    for frame in range(200):
        previous_broken = broken.copy()
        if frame < 10:
            prediction = predicted_state(positions, velocities, gravity, dt, frame)
        else:
            predicted_velocity = velocities + dt * gravity
            if frame == 10:
                broken[mesh.internal_faces] = True
                predicted_velocity += np.array([[0., 0., .1], [0., 0., -.1]])[:, None]
            prediction = (positions + dt * predicted_velocity - .5 * dt**2 * gravity, predicted_velocity)
        problem = make_problem(mesh, positions, velocities, prediction, gravity, dt, broken, previous_broken)
        result = solve_linear(problem, method)
        assert result.converged, (frame, result.status, result.diagnostics)
        assert_exact_connections(mesh, result.positions, result.velocities, broken)
        if frame >= 10:
            pairs = mesh.corner_face_pairs.reshape(-1, 2)
            flat = result.positions.reshape(-1, 3)
            assert np.min(np.linalg.norm(flat[pairs[:, 0]] - flat[pairs[:, 1]], axis=1)) > 1e-6
            assert not any(name.startswith("intact_face") for name in result.diagnostics["factor_violations"])
            assert result.diagnostics["free_momentum_island_count"] == (1 if frame == 10 else 2)
        positions, velocities = result.positions, result.velocities


@pytest.mark.parametrize("method", ["central"])
def test_preexisting_tolerance_sized_corner_drift_is_not_silently_certified_as_feasible(method):
    mesh = connected_mesh()
    positions = mesh.nodes[mesh.tets]
    velocities = np.zeros_like(positions)
    dt = 1e-4
    broken = np.zeros(len(mesh.faces.vertices), dtype=bool)
    prediction = predicted_state(positions, velocities, np.zeros(3), dt, 0)
    previous = make_problem(mesh, positions, velocities, prediction, np.zeros(3), dt, broken, broken)
    accepted_state = np.zeros_like(previous.prior)
    accepted_state[12:24:3] = 1e-7
    accepted_state[36:48:3] = 1e-7
    drifted_position, drifted_velocity = previous.unpack(accepted_state)
    assert _diagnostics(previous, accepted_state, CorrectorTolerances())["constraints_satisfied"]
    assert geometry_diagnostics(mesh, drifted_position, broken[mesh.internal_faces], 1e-6)["valid"]
    prediction = predicted_state(drifted_position, drifted_velocity, np.zeros(3), dt, 1)
    problem = make_problem(mesh, drifted_position, drifted_velocity, prediction, np.zeros(3), dt, broken, broken)
    result = solve_linear(problem, method)
    assert not result.converged
    assert result.status == "PrimalInfeasible"


@pytest.mark.parametrize("method", ["central"])
def test_active_energy_correction_preserves_exact_shared_corners(method):
    mesh = connected_mesh()
    positions = mesh.nodes[mesh.tets]
    velocities = np.zeros_like(positions)
    gravity = np.zeros(3)
    dt = 1e-3
    broken = np.zeros(len(mesh.faces.vertices), dtype=bool)
    prediction = predicted_state(positions, velocities, gravity, dt, 3)
    problem = make_problem(mesh, positions, velocities, prediction, gravity, dt, broken, broken)
    energy = CorotationalEnergy(mesh, 1., 10., .25, gravity)
    projection = EnergyProjection(problem, energy, energy.value(positions, velocities))
    assert not projection.diagnostics(problem.prior)["energy_satisfied"]
    result = solve_energy(problem, projection, method, CorrectorTolerances(), max_iterations=20000)
    assert result.converged, (result.status, result.diagnostics)
    assert result.diagnostics["energy"]["energy_satisfied"]
    assert_exact_connections(mesh, result.positions, result.velocities, broken)
    assert result.diagnostics["energy"]["outer_iterations"] > 1
    next_prediction = predicted_state(result.positions, result.velocities, gravity, dt, 4)
    next_problem = make_problem(mesh, result.positions, result.velocities, next_prediction, gravity, dt, broken, broken)
    next_result = solve_linear(next_problem, method)
    assert next_result.converged, (next_result.status, next_result.diagnostics)
    assert_exact_connections(mesh, next_result.positions, next_result.velocities, broken)
