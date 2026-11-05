from types import SimpleNamespace
from dataclasses import replace

import numpy as np
from numpy.testing import assert_allclose
import pytest
from scipy.sparse.linalg import spsolve
from scipy.spatial.transform import Rotation

from fracture_world.dynamics_corrector import DynamicsProblem, LinearizedContact
from fracture_world.dynamics_data import DynamicsTransition, material_topology
from fracture_world.dynamics_energy import CorotationalEnergy
from fracture_world.dynamics_mechanics import FrozenElasticBalance, fracture_initiation_diagnostics, mechanics_diagnostics
from fracture_world.mesh import VolumeMesh


def material(threshold=.5):
    return dict(rest_density=2., youngs_modulus=10., poissons_ratio=.25, fracture_stress_threshold=threshold)


def tetrahedron_mesh():
    return VolumeMesh([[0., 0., 0.], [1., 0., 0.], [0., 1., 0.], [0., 0., 1.]], [[0, 1, 2, 3]])


def observed_rest_transition():
    mesh = tetrahedron_mesh()
    positions = mesh.nodes[mesh.tets].reshape(-1, 3)
    velocity = np.zeros_like(positions)
    old = np.zeros(0, dtype=bool)
    fixed = np.zeros(4, dtype=bool)
    initial = SimpleNamespace(topology=material_topology(mesh, material()["rest_density"]), material=material(), gravity=np.zeros(3),
                              metadata=dict(teacher_physics=dict(rayleigh_mass_damping=0., rayleigh_stiffness_damping=0., friction_coefficient=0.)))
    observations = dict(pre_contact_velocities=np.zeros((1, 4, 3)), applied_delta_velocities=np.zeros((1, 4, 3)),
                        stabilization_delta_positions=np.zeros((1, 4, 3)), contact_observed=np.ones((1, 4), dtype=bool),
                        constraint_solve_active=np.zeros((1, 4), dtype=bool), fracture_stress_eigenvalues=np.zeros(1),
                        fracture_stress_thresholds=np.ones(1), fracture_stress_observed=np.ones(1, dtype=bool),
                        single_step_coverage=True)
    transition = DynamicsTransition(positions, velocity, old, fixed, positions.copy(), velocity.copy(), old.copy(), fixed.copy(),
                                     .01, 0., 0, observations)
    return initial, transition


@pytest.mark.parametrize("partial", [False, True])
def test_observed_native_velocity_failure_is_not_hidden_by_outer_status_or_partial_availability(partial):
    initial, transition = observed_rest_transition()
    transition.observations["applied_delta_velocities"][0, 0, 0] = 1.
    if partial:
        transition.observations["contact_observed"][0, 1:] = False
    report = mechanics_diagnostics(initial, transition, (), dict(velocity_scale=1.))
    assert report["elastic_dynamics"]["native_velocity_update"]["status"] == "fail"
    assert report["elastic_dynamics"]["status"] == "fail"


def test_unobserved_position_deltas_do_not_create_a_verified_failure():
    initial, transition = observed_rest_transition()
    transition.observations["contact_observed"][0, 1:] = False
    transition.observations["stabilization_delta_positions"][0, 1:, 0] = 1.
    report = mechanics_diagnostics(initial, transition, (), dict(velocity_scale=1.))
    assert report["kinematics"]["status"] == "unknown"
    assert report["kinematics"]["maximum_normalized_residual"] == 0
    assert report["kinematics"]["all_corner_unverified_maximum_normalized_residual"] > 0


def test_sparse_native_observation_never_certifies_a_whole_transition():
    initial, transition = observed_rest_transition()
    transition.observations["single_step_coverage"] = False
    report = mechanics_diagnostics(initial, replace(transition, dt=.02), (), dict(velocity_scale=1.))
    assert all(check["status"] == "unknown" for check in report.values())


def six_tetrahedron_mesh():
    angles = np.arange(6) * np.pi / 3
    nodes = np.vstack(([0., 0., 0.], [0., 0., 1.], np.column_stack((np.cos(angles), np.sin(angles), np.zeros(6)))))
    return VolumeMesh(nodes, [[0, 1, 2 + index, 2 + (index + 1) % 6] for index in range(6)])


def balance_for(mesh, positions=None, velocities=None, contacts=(), fixed=None):
    positions = mesh.nodes[mesh.tets].reshape(-1, 3) if positions is None else positions
    velocities = np.zeros_like(positions) if velocities is None else velocities
    fixed = np.zeros(len(positions), dtype=bool) if fixed is None else fixed
    return FrozenElasticBalance(mesh, material(), positions, velocities, .01, np.zeros(3),
                                np.zeros(len(mesh.internal_faces), dtype=bool), fixed, contacts=contacts)


def exact_free_step(balance):
    velocity = spsolve(balance.matrix, balance.rhs).reshape(-1, 3)[balance.groups]
    return balance.positions + balance.dt * velocity, velocity


def test_frozen_stiffness_matches_independent_energy_curvature_and_is_positive_semidefinite():
    mesh = tetrahedron_mesh()
    rotation = Rotation.from_rotvec([.3, -.4, .2]).as_matrix()
    deformation = rotation @ np.array([[1.2, .1, .02], [.1, .9, .03], [.02, .03, 1.1]])
    positions = mesh.nodes[mesh.tets].reshape(-1, 3) @ deformation.T
    balance = balance_for(mesh, positions)
    stiffness = balance.stiffness.toarray()
    assert_allclose(stiffness, stiffness.T, atol=1e-14)
    assert np.linalg.eigvalsh(stiffness).min() >= -1e-12
    assert_allclose(stiffness @ np.tile([.2, -.3, .1], 4), 0., atol=1e-14)
    inverse = np.linalg.inv((mesh.nodes[mesh.tets[0, 1:]] - mesh.nodes[mesh.tets[0, 0]]).T)
    direction = np.random.default_rng(61).normal(size=positions.shape)
    shear = material()["youngs_modulus"] / (2 * (1 + material()["poissons_ratio"]))
    lame = material()["youngs_modulus"] * material()["poissons_ratio"] / ((1 + material()["poissons_ratio"]) * (1 - 2 * material()["poissons_ratio"]))

    def energy(values):
        current_deformation = (values[1:] - values[:1]).T @ inverse
        strain = .5 * (rotation.T @ current_deformation + current_deformation.T @ rotation) - np.eye(3)
        return mesh.volumes[0] * (shear * np.sum(strain**2) + .5 * lame * np.trace(strain)**2)

    epsilon = 1e-4
    curvature = (energy(positions + epsilon * direction) - 2 * energy(positions) + energy(positions - epsilon * direction)) / epsilon**2
    assert_allclose(direction.ravel() @ stiffness @ direction.ravel(), curvature, rtol=2e-8, atol=2e-8)
    elastic = CorotationalEnergy(mesh, 2., 10., .25, np.zeros(3))
    expected = elastic.gradient(positions, np.zeros_like(positions))[0]
    assert_allclose(balance.elastic_gradient[balance.groups], expected, atol=1e-13)


def test_frozen_balance_uses_old_shared_nodal_mass_and_exact_linear_implicit_step():
    mesh = six_tetrahedron_mesh()
    positions = mesh.nodes[mesh.tets].reshape(-1, 3) * [1.2, .95, 1.]
    balance = balance_for(mesh, positions)
    assert len(balance.node_masses) == len(mesh.nodes)
    assert_allclose(balance.node_masses.sum(), mesh.volumes.sum() * material()["rest_density"])
    proposed_positions, proposed_velocities = exact_free_step(balance)
    report = balance.diagnostics(proposed_positions, proposed_velocities)
    assert report["status"] == report["free_status"] == "pass"
    assert report["unknown_nodes"] == 0
    assert report["maximum_free_normalized_residual"] < 1e-12
    assert report["old_node_endpoint_continuity_satisfied"]
    assert np.linalg.norm(proposed_velocities) > 0
    unchanged = balance.diagnostics(positions, np.zeros_like(positions))
    assert unchanged["status"] == unchanged["free_status"] == "fail"
    assert unchanged["maximum_free_normalized_residual"] > 1e-3


def test_balance_factors_match_original_corrector_velocity_coordinates_and_scaling():
    mesh = six_tetrahedron_mesh()
    positions = mesh.nodes[mesh.tets] * [1.2, 1., 1.]
    balance = balance_for(mesh, positions.reshape(-1, 3))
    next_positions, next_velocities = exact_free_step(balance)
    problem = DynamicsProblem(mesh, positions, np.zeros_like(positions), next_positions.reshape(positions.shape),
                              next_velocities.reshape(positions.shape), .01, density=material()["rest_density"],
                              position_scale=.03, velocity_scale=3., kinematic_rule="backward_euler")
    state = problem.encode_state(next_positions.reshape(positions.shape), next_velocities.reshape(positions.shape))
    factors = balance.factors(problem)
    assert len(factors) == len(balance.free_nodes)
    assert max(np.max(factor.violation(state[factor.indices])) for factor in factors) < 1e-12
    invalid = problem.encode_state(positions, np.zeros_like(positions))
    assert max(np.max(factor.violation(invalid[factor.indices])) for factor in factors) > 1e-3


def test_unloaded_fracture_is_rejected_even_when_piece_size_allows_native_splitting():
    mesh = six_tetrahedron_mesh()
    previous = np.zeros(len(mesh.internal_faces), dtype=bool)
    candidate = previous.copy()
    candidate[0] = True
    report = fracture_initiation_diagnostics(mesh, material(), mesh.nodes[mesh.tets], previous, candidate)
    assert report["status"] == "fail"
    assert report["unsupported_new_face_ids"] == [0]
    assert report["faces"][0]["minimum_piece_size_satisfied"]
    assert not report["faces"][0]["stress_support_satisfied"]
    assert report["full_topology_status"] == "unknown"
    assert fracture_initiation_diagnostics(mesh, material(), mesh.nodes[mesh.tets], previous, previous)["status"] == "pass"


def test_native_fracture_criterion_includes_compressive_stress():
    mesh = six_tetrahedron_mesh()
    positions = mesh.nodes[mesh.tets] * [.8, 1., 1.]
    previous = np.zeros(len(mesh.internal_faces), dtype=bool)
    candidate = previous.copy()
    candidate[0] = True
    report = fracture_initiation_diagnostics(mesh, material(.1), positions, previous, candidate)
    assert report["status"] == "pass"
    assert report["maximum_cell_stress_ratio"] > 20
    assert report["certificate_scope"] == "necessary initiation conditions only"
    assert report["full_topology_status"] == "unknown"


def test_fracture_criterion_preserves_pinned_native_shear_component_order():
    mesh = six_tetrahedron_mesh()
    strain = np.array([[.1, .2, .3], [.2, .4, .5], [.3, .5, .6]])
    deformation = np.eye(3) + strain
    assert np.linalg.eigvalsh(deformation).min() > 0
    parameters = material(1.)
    shear, lame = 4., 4.
    standard = 2 * shear * strain + lame * np.trace(strain) * np.eye(3)
    pinned = np.array([[standard[0, 0], standard[0, 2], standard[1, 2]],
                       [standard[0, 2], standard[1, 1], standard[0, 1]],
                       [standard[1, 2], standard[0, 1], standard[2, 2]]])
    previous = np.zeros(len(mesh.internal_faces), dtype=bool)
    report = fracture_initiation_diagnostics(mesh, parameters, mesh.nodes[mesh.tets] @ deformation.T, previous, previous)
    assert_allclose(report["maximum_cell_stress_ratio"], np.max(np.abs(np.linalg.eigvalsh(pinned))), atol=1e-12)
    assert abs(report["maximum_cell_stress_ratio"] - np.max(np.abs(np.linalg.eigvalsh(standard)))) > .01


def test_new_face_can_be_supported_by_nonadjacent_overstressed_tet_in_its_vertex_star():
    mesh = six_tetrahedron_mesh()
    positions = mesh.nodes.copy()
    positions[5, :2] *= 1.4
    previous = np.zeros(len(mesh.internal_faces), dtype=bool)
    face = int(np.flatnonzero(np.all(mesh.dual_edges == [0, 5], axis=1))[0])
    candidate = previous.copy()
    candidate[face] = True
    report = fracture_initiation_diagnostics(mesh, material(.5), positions[mesh.tets], previous, candidate)
    assert report["status"] == "pass"
    witnesses = report["faces"][0]["overstressed_witness_tetrahedra"]
    assert witnesses
    assert not set(witnesses).intersection(mesh.dual_edges[face].tolist())


def test_small_mesh_stress_exceedance_does_not_certify_native_fracture():
    mesh = VolumeMesh([[0., 0., 0.], [1., 0., 0.], [0., 1., 0.], [0., 0., 1.], [1., 1., 1.]],
                      [[0, 1, 2, 3], [4, 1, 2, 3]])
    report = fracture_initiation_diagnostics(mesh, material(.01), mesh.nodes[mesh.tets] * [.8, 1., 1.],
                                             np.array([False]), np.array([True]))
    assert report["status"] == "fail"
    assert report["faces"][0]["stress_support_satisfied"]
    assert not report["faces"][0]["minimum_piece_size_satisfied"]


@pytest.mark.parametrize("boundary", ["contact", "support", "all_supported"])
def test_boundary_reactions_remain_unknown_instead_of_being_fitted_to_force_residual(boundary):
    mesh = tetrahedron_mesh()
    fixed = np.zeros(4, dtype=bool)
    contacts = ()
    if boundary == "contact":
        contacts = (LinearizedContact.pair(0, -1, np.array([0., 1., 0.]), 0.),)
    else:
        fixed[:1 if boundary == "support" else 4] = True
    balance = balance_for(mesh, contacts=contacts, fixed=fixed)
    report = balance.diagnostics(balance.positions, balance.velocities)
    assert report["status"] == "unknown"
    assert report["free_status"] == ("unknown" if boundary == "all_supported" else "pass")
    assert report["unknown_nodes"] == (4 if boundary == "all_supported" else 1)
    assert len(balance.factors(DynamicsProblem(mesh, balance.positions.reshape(1, 4, 3), balance.velocities.reshape(1, 4, 3),
                                              balance.positions.reshape(1, 4, 3), balance.velocities.reshape(1, 4, 3), .01))) == report["verified_free_nodes"]


def teacher_transition(contacts=(), missing_settings=False):
    mesh = six_tetrahedron_mesh()
    positions = mesh.nodes[mesh.tets].reshape(-1, 3) * [.8, 1., 1.]
    balance = balance_for(mesh, positions, contacts=contacts)
    next_positions, next_velocities = exact_free_step(balance)
    candidate = balance.broken.copy()
    candidate[0] = True
    transition = DynamicsTransition(positions, balance.velocities, balance.broken, balance.fixed,
                                    next_positions, next_velocities, candidate, balance.fixed, .01, 0., 0)
    settings = {} if missing_settings else dict(rayleigh_mass_damping=0., rayleigh_stiffness_damping=0., friction_coefficient=0.)
    initial = SimpleNamespace(topology=material_topology(mesh, 2.), material=material(), gravity=np.zeros(3),
                              metadata=dict(native_settings=settings))
    return initial, transition


def test_teacher_wrapper_uses_prefracture_balance_and_copied_endpoint_states():
    initial, transition = teacher_transition()
    report = mechanics_diagnostics(initial, transition, (), dict(velocity_scale=1.))
    assert report["elastic_dynamics"]["status"] == "pass"
    assert report["elastic_dynamics"]["old_node_endpoint_continuity_satisfied"]
    assert report["kinematics"]["status"] == "pass"
    assert report["fracture_initiation"]["status"] == "pass"
    assert report["fracture_initiation"]["full_topology_status"] == "unknown"


def test_teacher_wrapper_keeps_missing_settings_and_unrecorded_stabilization_unknown():
    initial, transition = teacher_transition(missing_settings=True)
    report = mechanics_diagnostics(initial, transition, (), dict(velocity_scale=1.))
    assert report["elastic_dynamics"]["status"] == "unknown"
    assert report["elastic_dynamics"]["free_status"] == "pass"
    contact = LinearizedContact.pair(0, -1, np.array([0., 1., 0.]), 0.)
    report = mechanics_diagnostics(initial, transition, (contact,), dict(velocity_scale=1.))
    assert report["kinematics"]["status"] == "unknown"
    with pytest.raises(ValueError, match="explicit velocity scale"):
        mechanics_diagnostics(initial, transition, (), {})
