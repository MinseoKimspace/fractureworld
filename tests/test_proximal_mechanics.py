from types import SimpleNamespace

import numpy as np
from numpy.testing import assert_allclose
import pytest
from scipy.sparse.linalg import spsolve

from fracture_world.dynamics_corrector import CorrectorTolerances, DynamicsProblem, _shared_problem, solve_central, solve_consensus
from fracture_world.dynamics_data import material_topology
from fracture_world.dynamics_energy import CorotationalEnergy, EnergyProjection
from fracture_world.dynamics_energy_solvers import solve_energy
from fracture_world.dynamics_initial import DynamicsInitial
from fracture_world.dynamics_mechanics import FrozenElasticBalance
from fracture_world.dynamics_runtime import RolloutConfig, _correct
from test_dynamics_mechanics import material, six_tetrahedron_mesh, tetrahedron_mesh


def mechanical_problem(mode="proximal", weight=1., shared=False):
    mesh = six_tetrahedron_mesh() if shared else tetrahedron_mesh()
    positions = mesh.nodes.astype(np.float32).astype(float)[mesh.tets].reshape(-1, 3) * [1.2, 1., 1.]
    velocities = np.zeros_like(positions)
    broken = np.zeros(len(mesh.internal_faces), dtype=bool)
    balance = FrozenElasticBalance(mesh, material(), positions, velocities, .01, np.zeros(3), broken,
                                    np.zeros(len(positions), dtype=bool), velocity_scale=1.7)
    problem = DynamicsProblem(mesh, positions.reshape(-1, 4, 3), velocities.reshape(-1, 4, 3),
                              positions.reshape(-1, 4, 3), velocities.reshape(-1, 4, 3), .01,
                              density=material()["rest_density"], position_scale=.017, velocity_scale=1.7,
                              kinematic_rule="backward_euler", elastic_balance=balance,
                              mechanics_mode=mode, mechanics_weight=weight)
    return problem, balance


@pytest.mark.parametrize("method", ["central", "consensus"])
def test_proximal_elastic_penalty_moves_a_feasible_but_mechanically_wrong_prediction(method):
    problem, balance = mechanical_problem()
    tolerance = CorrectorTolerances(1e-10, 1e-9)
    solver = solve_central if method == "central" else solve_consensus
    result = solver(problem, tolerances=tolerance, max_iterations=20000)
    assert result.converged, (result.status, result.diagnostics)
    mass = np.repeat(balance.node_masses / balance.node_masses.sum(), 3)
    normalized = balance.matrix.toarray() / np.repeat(balance.node_masses, 3)[:, None]
    target = balance.rhs / np.repeat(balance.node_masses, 3) / balance.velocity_scale
    weights = np.diag(mass)
    expected = np.linalg.solve(2 * weights + normalized.T @ weights @ normalized,
                               normalized.T @ weights @ target) * balance.velocity_scale
    assert_allclose(result.velocities.ravel(), expected, rtol=2e-6, atol=2e-8)
    assert_allclose(result.positions, problem.positions + problem.dt * result.velocities, atol=2e-10)
    before = balance.diagnostics(problem.positions, problem.velocities)["maximum_free_normalized_residual"]
    after = balance.diagnostics(result.positions, result.velocities)["maximum_free_normalized_residual"]
    assert 0 < after < before
    assert np.linalg.norm(result.velocities) > 0
    components = result.diagnostics["objective_components"]
    assert components["prediction_fidelity"] > 0
    assert components["elastic_residual_penalty"] > 0
    assert_allclose(sum(components.values()), result.objective, atol=1e-12)


@pytest.mark.parametrize("method", ["central", "consensus"])
def test_exact_reference_agrees_with_independent_sparse_implicit_solution(method):
    problem, balance = mechanical_problem("exact")
    solver = solve_central if method == "central" else solve_consensus
    result = solver(problem, tolerances=CorrectorTolerances(1e-10, 1e-9), max_iterations=20000)
    assert result.converged, (result.status, result.diagnostics)
    expected = spsolve(balance.matrix, balance.rhs).reshape(-1, 3)[balance.groups]
    assert_allclose(result.velocities.reshape(-1, 3), expected, atol=2e-8)
    assert balance.diagnostics(result.positions, result.velocities)["free_status"] == "pass"


def test_shared_reduction_preserves_soft_penalties_duplicates_and_affine_constants():
    problem, _ = mechanical_problem(shared=True)
    duplicate = next(factor for factor in problem.factors if factor.penalty is not None)
    extra = problem._factor("constant_soft", [0, 0], [0, 4 * 3], [1., -1.], [.2])
    extra.penalty = 2.
    problem.factors = (*problem.factors, duplicate, extra)
    reduced = _shared_problem(problem)
    assert sum(factor.penalty is not None for factor in reduced.factors) == sum(factor.penalty is not None for factor in problem.factors)
    for state in (reduced.prior, np.random.default_rng(401).normal(size=reduced.prior.shape)):
        full = state[reduced.state_indices] + reduced.offset
        assert_allclose(reduced.objective(state), problem.objective(full), atol=1e-12, rtol=1e-12)


@pytest.mark.parametrize("method", ["central", "consensus"])
def test_energy_path_does_not_skip_a_nonzero_soft_residual_at_a_feasible_prior(method):
    problem, balance = mechanical_problem()
    energy = CorotationalEnergy(problem.mesh, material()["rest_density"], material()["youngs_modulus"],
                                 material()["poissons_ratio"], np.zeros(3))
    projection = EnergyProjection(problem, energy, energy.value(problem.positions, problem.velocities))
    result = solve_energy(problem, projection, method, max_iterations=20000)
    assert result.converged, (result.status, result.diagnostics)
    assert np.linalg.norm(result.velocities) > 0
    assert result.diagnostics["energy"]["energy_satisfied"]
    assert balance.diagnostics(result.positions, result.velocities)["maximum_free_normalized_residual"] < balance.diagnostics(problem.positions, problem.velocities)["maximum_free_normalized_residual"]


def initial_for(mesh, positions):
    topology = material_topology(mesh, material()["rest_density"])
    fixed = np.zeros(len(positions), dtype=bool)
    metadata = dict(support_policy="free", teacher_physics=dict(rayleigh_mass_damping=0., rayleigh_stiffness_damping=0., friction_coefficient=0.))
    return DynamicsInitial(topology, positions, np.zeros_like(positions), np.zeros(len(mesh.internal_faces), dtype=bool),
                           fixed, 0., .01, np.zeros(3), material(), "mechanical", "a" * 64, metadata)


def test_unloaded_new_fracture_is_rejected_by_the_actual_corrector_path():
    mesh = six_tetrahedron_mesh()
    positions = mesh.nodes.astype(np.float32).astype(float)[mesh.tets].reshape(-1, 3)
    initial = initial_for(mesh, positions)
    broken = initial.broken.copy()
    broken[0] = True
    model = SimpleNamespace(config=dict(velocity_scale=1., motion_prior="constant_acceleration"))
    import torch

    proposal = dict(position_increment=torch.zeros_like(torch.tensor(positions)), velocity_increment=torch.zeros_like(torch.tensor(positions)))
    config = RolloutConfig(mechanics_mode="proximal", fracture_initiation=True, kinematic_rule="backward_euler",
                           energy_balance=False, max_iterations=20000)
    result, report = _correct(initial, positions, initial.velocities, proposal, broken, initial.fixed, model, config,
                              "central", previous_broken=initial.broken, previous_fixed=initial.fixed)
    assert result is None
    assert report["status"] == "fracture_initiation_rejected"
    assert report["mechanics"]["fracture_initiation"]["unsupported_new_face_ids"] == [0]


def test_native_rounded_rest_state_has_no_spurious_elastic_force():
    mesh = six_tetrahedron_mesh()
    positions = mesh.nodes.astype(np.float32).astype(float)[mesh.tets].reshape(-1, 3)
    balance = FrozenElasticBalance(mesh, material(), positions, np.zeros_like(positions), .01, np.zeros(3),
                                    np.zeros(len(mesh.internal_faces), dtype=bool), np.zeros(len(positions), dtype=bool))
    assert_allclose(balance.elastic_gradient, 0., atol=1e-14)
    assert balance.diagnostics(positions, np.zeros_like(positions))["status"] == "pass"
