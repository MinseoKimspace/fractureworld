import numpy as np
import pytest
from numpy.testing import assert_allclose

from fracture_world.dynamics_corrector import CorrectorTolerances, DynamicsProblem, LinearizedContact, solve_central, solve_consensus
from fracture_world.mesh import VolumeMesh


def mesh_pair():
    return VolumeMesh(np.array([[0, 0, 0], [1, 0, 0], [0, 1, 0], [0, 0, 1], [0, 0, -1]], dtype=float), np.array([[0, 1, 2, 3], [0, 2, 1, 4]]))


def problem_for(mesh, **kwargs):
    positions = mesh.nodes[mesh.tets]
    velocities = np.zeros_like(positions)
    predicted_velocity = np.array([[[0.3, 0.2, -0.1]] * 4, [[-0.1, -0.2, 0.3]] * 4])[:len(mesh.tets)]
    values = {"mesh": mesh, "positions": positions, "velocities": velocities, "predicted_positions": positions + 0.05 * predicted_velocity, "predicted_velocities": predicted_velocity, "dt": 0.05, "position_scale": 1.0, "velocity_scale": 1.0}
    values.update(kwargs)
    return DynamicsProblem(**values)


@pytest.mark.parametrize("solver", [solve_central, solve_consensus])
def test_user_cancellation_is_reported_without_a_successful_commit(solver):
    result = solver(problem_for(mesh_pair()), cancel=lambda: True)
    assert result.status == "interrupted"
    assert not result.converged


def check_solvers(problem):
    tolerance = CorrectorTolerances(1e-10, 1e-9)
    central = solve_central(problem, tolerance)
    consensus = solve_consensus(problem, tolerances=tolerance, max_iterations=20000)
    assert central.converged, central.status
    assert consensus.converged, consensus.status
    assert central.diagnostics["constraints_satisfied"]
    assert consensus.diagnostics["constraints_satisfied"]
    assert_allclose(consensus.positions, central.positions, atol=1e-7, rtol=1e-7)
    assert_allclose(consensus.velocities, central.velocities, atol=1e-7, rtol=1e-7)
    assert_allclose(consensus.objective, central.objective, atol=1e-8, rtol=1e-7)
    assert consensus.history[-1].primal_residual <= consensus.history[-1].primal_tolerance
    assert consensus.history[-1].dual_residual <= consensus.history[-1].dual_tolerance
    return central, consensus


def test_intact_face_agrees_and_closed_system_conserves_momentum():
    problem = problem_for(mesh_pair())
    _, result = check_solvers(problem)
    face = problem.mesh.internal_faces[0]
    a, b = problem.mesh.faces.cells[face]
    for vertex in problem.mesh.faces.vertices[face]:
        ia = int(np.flatnonzero(problem.mesh.tets[a] == vertex)[0])
        ib = int(np.flatnonzero(problem.mesh.tets[b] == vertex)[0])
        assert_allclose(result.positions[a, ia], result.positions[b, ib], atol=1e-8)
        assert_allclose(result.velocities[a, ia], result.velocities[b, ib], atol=1e-8)
    assert_allclose(result.positions, problem.positions + problem.dt * result.velocities, atol=1e-8)
    assert_allclose(result.diagnostics["momentum_residual_or_inferred_boundary_impulse"], 0, atol=1e-8)
    assert_allclose(result.diagnostics["angular_residual_or_inferred_boundary_angular_impulse"], 0, atol=1e-8)
    assert result.diagnostics["boundary_momentum_verified"]
    assert result.diagnostics["angular_momentum_verified"]
    assert result.diagnostics["engine_calls"] == 0


def test_broken_face_is_not_retied():
    mesh = mesh_pair()
    broken = np.zeros(len(mesh.faces.vertices), dtype=bool)
    broken[mesh.internal_faces] = True
    problem = problem_for(mesh, broken_faces=broken)
    _, result = check_solvers(problem)
    face = mesh.internal_faces[0]
    a, b = mesh.faces.cells[face]
    vertex = mesh.faces.vertices[face, 0]
    ia = int(np.flatnonzero(mesh.tets[a] == vertex)[0])
    ib = int(np.flatnonzero(mesh.tets[b] == vertex)[0])
    assert np.linalg.norm(result.positions[a, ia] - result.positions[b, ib]) > 1e-3
    assert not any(name.startswith("intact_face") for name in problem.metadata["factor_names"])


def test_supplied_external_impulse_is_in_closed_system_balance():
    mesh = mesh_pair()
    impulse = np.zeros((2, 4, 3))
    impulse[..., 0] = 0.01
    problem = problem_for(mesh, external_impulse=impulse)
    _, result = check_solvers(problem)
    change = np.sum(problem.corner_masses[..., None] * result.velocities, axis=(0, 1))
    assert_allclose(change, impulse.sum(axis=(0, 1)), atol=1e-8)
    assert_allclose(result.diagnostics["angular_momentum_change"], result.diagnostics["supplied_external_angular_impulse"], atol=1e-8)


def test_fixed_support_reaction_is_reported_as_inferred():
    mesh = mesh_pair()
    fixed = np.zeros((2, 4), dtype=bool)
    fixed[:, 0] = True
    problem = problem_for(mesh, fixed=fixed)
    _, result = check_solvers(problem)
    assert_allclose(result.positions[fixed], problem.positions[fixed], atol=1e-8)
    assert_allclose(result.velocities[fixed], 0, atol=1e-8)
    assert not result.diagnostics["boundary_momentum_verified"]
    assert not result.diagnostics["angular_momentum_verified"]
    assert np.linalg.norm(result.diagnostics["momentum_residual_or_inferred_boundary_impulse"]) > 1e-4
    assert "inferred" in problem.metadata["momentum_constraint"]


def test_frozen_contact_reduces_penetration_without_claiming_impulse_law():
    mesh = mesh_pair()
    contact = LinearizedContact.pair(0, -1, [0, 0, 1], -0.01)
    problem = problem_for(mesh, contacts=(contact,))
    _, result = check_solvers(problem)
    assert result.positions.reshape(-1, 3)[0, 2] >= 0.01 - 1e-8
    assert result.diagnostics["linearized_contact_gaps"][0] >= -1e-8
    assert not result.diagnostics["boundary_momentum_verified"]
    assert not result.diagnostics["contact_impulse_law_verified"]


def test_weighted_vertex_face_contact_is_supported():
    mesh = mesh_pair()
    broken = np.zeros(len(mesh.faces.vertices), dtype=bool)
    broken[mesh.internal_faces] = True
    contact = LinearizedContact(np.array([0, 4, 5, 6]), np.array([1.0, -0.2, -0.3, -0.5]), [0, 0, 1], -0.02)
    problem = problem_for(mesh, broken_faces=broken, contacts=(contact,))
    _, result = check_solvers(problem)
    displacement = (result.positions - problem.positions).reshape(-1, 3)
    assert contact.gap + np.sum(contact.weights * displacement[contact.corner_ids, 2]) >= -1e-8
    assert result.diagnostics["boundary_momentum_verified"]


def test_trapezoidal_kinematics_is_explicit():
    mesh = mesh_pair()
    velocity = np.full((2, 4, 3), 0.2)
    problem = problem_for(mesh, velocities=velocity, kinematic_rule="trapezoidal")
    _, result = check_solvers(problem)
    assert_allclose(result.positions - problem.positions, problem.dt * (velocity + result.velocities) / 2, atol=1e-8)
    assert problem.metadata["kinematic_rule"] == "trapezoidal"
    assert_allclose(result.diagnostics["angular_residual_or_inferred_boundary_angular_impulse"], 0, atol=1e-8)


@pytest.mark.parametrize("rule", ["backward_euler", "trapezoidal"])
def test_closed_rotating_state_preserves_linear_and_angular_momentum(rule):
    mesh = mesh_pair()
    positions = mesh.nodes[mesh.tets]
    velocity = np.cross(np.array([0.3, -0.4, 0.8]), positions) + np.array([0.1, 0.2, -0.1])
    problem = problem_for(mesh, velocities=velocity, kinematic_rule=rule)
    _, result = check_solvers(problem)
    assert_allclose(result.diagnostics["momentum_residual_or_inferred_boundary_impulse"], 0, atol=1e-8)
    assert_allclose(result.diagnostics["angular_residual_or_inferred_boundary_angular_impulse"], 0, atol=1e-8)


def test_zero_budget_and_cancellation_do_not_claim_convergence():
    problem = problem_for(mesh_pair())
    for result in (solve_central(problem, max_iterations=0), solve_consensus(problem, max_iterations=0), solve_consensus(problem, cancel=lambda: True)):
        assert not result.converged
        assert result.iterations == 0
    result = solve_consensus(problem, max_iterations=1)
    assert not result.converged
    assert result.status == "maximum_iterations"
    assert result.iterations == 1


def test_infeasible_fixed_contact_is_not_reported_as_converged():
    mesh = mesh_pair()
    fixed = np.ones((2, 4), dtype=bool)
    problem = problem_for(mesh, fixed=fixed, contacts=(LinearizedContact.pair(0, -1, [0, 0, 1], -0.01),))
    result = solve_central(problem)
    assert not result.converged
    assert "Infeasible" in result.status
    result = solve_consensus(problem, max_iterations=50)
    assert not result.converged
    assert not result.diagnostics["constraints_satisfied"]


@pytest.mark.parametrize("change", [{"dt": 0}, {"position_weight": 0}, {"velocity_scale": -1}, {"kinematic_rule": "unknown"}, {"fixed": np.zeros(8, dtype=bool)}])
def test_invalid_problem_is_rejected(change):
    with pytest.raises(ValueError):
        problem_for(mesh_pair(), **change)


def test_invalid_contact_weights_are_rejected():
    with pytest.raises(ValueError, match="sum"):
        LinearizedContact([0, 1], [1, 1], [0, 0, 1], 0)
