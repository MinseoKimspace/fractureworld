import numpy as np
import pytest
from numpy.testing import assert_allclose
from scipy import sparse
from scipy.sparse.linalg import spsolve

from fracture_world.dynamics_energy import native_corotational_energy
from fracture_world.dynamics_frozen_energy import FrozenQuadraticEnergy, TransportEnergy
from fracture_world.dynamics_mechanics import FrozenElasticBalance
from fracture_world.femfx import native_directory
from fracture_world.mesh import VolumeMesh


def frozen_fixture(shared=False, deformed=False, gravity=(0., 0., 0.)):
    nodes = np.array([[0., 0., 0.], [1., 0., 0.], [0., 1., 0.], [0., 0., 1.], [0., 0., -2.]])
    mesh = VolumeMesh(nodes if shared else nodes[:4],
                      np.array([[0, 1, 2, 3], [0, 2, 1, 4]]) if shared else np.array([[0, 1, 2, 3]]))
    material = dict(rest_density=3., youngs_modulus=100., poissons_ratio=.25, fracture_stress_threshold=10000.)
    positions = mesh.nodes[mesh.tets].copy()
    if deformed:
        positions = positions @ np.array([[1.1, .1, .02], [0., .95, .05], [.03, 0., 1.04]]).T
    velocities = np.broadcast_to([.2, -.1, .3], positions.shape).copy()
    balance = FrozenElasticBalance(mesh, material, positions, velocities, .015, np.asarray(gravity),
                                   np.zeros(len(mesh.internal_faces), dtype=bool), np.zeros(4 * len(mesh.tets), dtype=bool))
    nonlinear = native_corotational_energy(mesh, 3., 100., .25, gravity)
    return balance, FrozenQuadraticEnergy(balance, nonlinear), positions, velocities


def test_instantaneous_rigid_rotation_exposes_nonlinear_energy_defect_without_frozen_violation():
    balance, energy, positions, _ = frozen_fixture()
    velocities = np.cross([0., 0., 10.], positions - positions.mean(axis=1, keepdims=True))
    balance = FrozenElasticBalance(balance.mesh, balance.material, positions, velocities, .01, np.zeros(3),
                                   np.zeros(0, dtype=bool), np.zeros(4, dtype=bool))
    energy = FrozenQuadraticEnergy(balance, energy.nonlinear_energy)
    next_velocity = spsolve(balance.matrix, balance.rhs).reshape(positions.shape)
    next_position = positions + balance.dt * next_velocity
    report = energy.balance(positions, velocities, next_position, next_velocity, 10.)
    assert report["energy_satisfied"]
    assert abs(report["energy_change"]["total"]) < 1e-12
    independent = energy.nonlinear_diagnostics(next_position, next_velocity)
    assert independent["elastic_linearization_defect"] > 1e-4
    assert independent["defect_credited_as_allowed_work"] is False
    assert independent["status"] == "diagnostic_only"
    assert energy.supports_zero_energy_shortcut is False
    assert energy.supports_nonnegative_energy_shortcut is False


def test_exact_frozen_backward_euler_satisfies_discrete_work_identity_on_shared_nodes():
    balance, energy, positions, velocities = frozen_fixture(True, True, (.2, -9.81, -.4))
    next_nodal_velocity = spsolve(balance.matrix, balance.rhs).reshape(-1, 3)
    next_velocity = next_nodal_velocity[balance.groups].reshape(positions.shape)
    next_position = positions + balance.dt * next_velocity
    previous_nodal_velocity = velocities.reshape(-1, 3)[balance.representatives]
    delta = next_nodal_velocity - previous_nodal_velocity
    inertial_loss = .5 * np.sum(balance.node_masses[:, None] * delta**2)
    elastic_loss = .5 * balance.dt**2 * float(next_nodal_velocity.ravel() @
                                           (balance.stiffness @ next_nodal_velocity.ravel()))
    report = energy.balance(positions, velocities, next_position, next_velocity, 1.)
    assert_allclose(report["energy_change"]["total"], -inertial_loss - elastic_loss, atol=2e-14, rtol=1e-11)
    assert report["energy_satisfied"]
    assert report["allowed_external_work"] == 0.
    assert report["engine_calls"] == 0


def test_frozen_corner_gradient_matches_finite_differences_with_unequal_shared_masses():
    _, energy, positions, velocities = frozen_fixture(True, True, (.2, -9.81, -.4))
    rng = np.random.default_rng(113)
    positions = positions + rng.normal(0., .02, positions.shape)
    velocities = velocities + rng.normal(0., .1, velocities.shape)
    gradient = np.concatenate([value.ravel() for value in energy.gradient(positions, velocities)])
    state = np.r_[positions.ravel(), velocities.ravel()]
    numerical = np.empty_like(state)
    for index in range(len(state)):
        above, below = state.copy(), state.copy()
        above[index] += 1e-6
        below[index] -= 1e-6
        numerical[index] = (energy.value(above[:positions.size].reshape(positions.shape),
                                        above[positions.size:].reshape(velocities.shape)) -
                            energy.value(below[:positions.size].reshape(positions.shape),
                                        below[positions.size:].reshape(velocities.shape))) / 2e-6
    assert_allclose(gradient, numerical, atol=1e-8, rtol=2e-7)
    shared = np.flatnonzero(energy.groups == energy.groups[0])
    assert len(shared) == 2
    assert_allclose(energy.corner_weights[shared], [1 / 3, 2 / 3])


def test_frozen_hessian_matches_gradient_finite_differences_and_flat_state_interface():
    _, energy, positions, velocities = frozen_fixture(True, True, (.2, -9.81, -.4))
    rng = np.random.default_rng(127)
    position_direction = rng.normal(size=positions.shape)
    velocity_direction = rng.normal(size=velocities.shape)
    actual = energy.hessian_vector(positions, velocities, position_direction, velocity_direction)
    above = energy.gradient(positions + 1e-6 * position_direction, velocities + 1e-6 * velocity_direction)
    below = energy.gradient(positions - 1e-6 * position_direction, velocities - 1e-6 * velocity_direction)
    for result, plus, minus in zip(actual, above, below, strict=True):
        assert_allclose(result, (plus - minus) / 2e-6, atol=2e-8, rtol=2e-8)
    flat = energy.hessian_vector(positions.reshape(-1, 3), velocities.reshape(-1, 3),
                                 position_direction.reshape(-1, 3), velocity_direction.reshape(-1, 3))
    for structured, flattened in zip(actual, flat, strict=True):
        assert_allclose(structured.reshape(-1, 3), flattened, atol=0.)


def test_stabilization_work_is_separate_exact_quadratic_position_work():
    _, energy, positions, velocities = frozen_fixture(True, True, (.2, -9.81, -.4))
    rng = np.random.default_rng(131)
    positions = positions + rng.normal(0., .02, positions.shape)
    stabilization = rng.normal(0., .005, positions.shape)
    work = energy.stabilization_work(positions, stabilization)
    assert_allclose(work["total"], energy.value(positions, velocities) - energy.value(positions - stabilization, velocities),
                    atol=2e-15, rtol=1e-12)
    assert work["scope"] == "numerical position stabilization work; not physical external work"


def test_frozen_energy_does_not_clip_source_stiffness_or_claim_nonnegative_potential():
    balance, _, positions, velocities = frozen_fixture()
    balance.stiffness = sparse.diags(np.r_[-1., np.zeros(positions.size - 1)], format="csr")
    energy = FrozenQuadraticEnergy(balance)
    displaced = positions.copy()
    displaced[0, 0, 0] += 1
    assert energy.components(displaced, np.zeros_like(velocities))["elastic"] < 0.
    assert energy.supports_nonnegative_energy_shortcut is False
    assert energy.nonlinear_diagnostics(displaced, velocities)["status"] == "unknown"


def test_corner_reduction_normalizes_material_weights_before_distributing_actual_source_mass():
    balance, _, positions, velocities = frozen_fixture(True, True)
    balance.node_masses *= 1 + np.finfo(np.float32).eps
    energy = FrozenQuadraticEnergy(balance)
    weights = np.bincount(energy.groups, weights=energy.corner_weights)
    masses = np.bincount(energy.groups, weights=energy.masses.ravel())
    assert_allclose(weights, 1., atol=0., rtol=2e-16)
    assert_allclose(masses, balance.node_masses, atol=0., rtol=2e-16)
    assert_allclose(energy._nodal_average(np.ones_like(positions)), 1., atol=0., rtol=2e-16)
    assert energy.gradient(positions, velocities)[1].shape == positions.shape


def test_transport_energy_enforces_unshifted_mechanical_transport_and_reports_stabilization_work_separately():
    balance, energy, positions, velocities = frozen_fixture(True, True, (0., -9.81, 0.))
    next_velocity = spsolve(balance.matrix, balance.rhs).reshape(-1, 3)[balance.groups].reshape(positions.shape)
    transported = positions + balance.dt * next_velocity
    stabilization = np.broadcast_to([0., .03, 0.], positions.shape).copy()
    old_reference = energy.reference_positions.copy()
    wrapper = TransportEnergy(energy, stabilization)
    endpoint = transported + stabilization
    report = wrapper.balance(positions, velocities, endpoint, next_velocity, 1.,
                             budget=energy.value(positions, velocities),
                             reference_components=wrapper.components(positions, velocities))
    expected = energy.balance(positions, velocities, transported, next_velocity, 1.)
    assert report["energy_satisfied"]
    assert report["allowed_external_work"] == 0.
    assert_allclose(report["normalized_residual"], expected["normalized_residual"], atol=1e-15)
    assert_allclose(wrapper.initial_components(positions, velocities)["total"], energy.value(positions, velocities), atol=0.)
    independent = wrapper.stabilization_diagnostics(endpoint, next_velocity)
    assert independent["numerical_stabilization_work"]["total"] > 0.
    assert independent["work_credited_as_allowed_work"] is False
    assert_allclose(independent["endpoint_components"]["total"] - independent["transport_components"]["total"],
                    independent["numerical_stabilization_work"]["total"], atol=1e-14)
    np.testing.assert_array_equal(energy.reference_positions, old_reference)


def test_transport_energy_gradient_and_hessian_differentiate_the_fixed_position_map():
    _, energy, positions, velocities = frozen_fixture(True, True, (.2, -9.81, -.4))
    rng = np.random.default_rng(137)
    stabilization = rng.normal(0., .01, positions.shape)
    wrapper = TransportEnergy(energy, stabilization)
    positions = positions + rng.normal(0., .02, positions.shape)
    direction = rng.normal(size=positions.shape)
    velocity_direction = rng.normal(size=velocities.shape)
    actual = wrapper.hessian_vector(positions, velocities, direction, velocity_direction)
    above = wrapper.gradient(positions + 1e-6 * direction, velocities + 1e-6 * velocity_direction)
    below = wrapper.gradient(positions - 1e-6 * direction, velocities - 1e-6 * velocity_direction)
    for result, plus, minus in zip(actual, above, below, strict=True):
        assert_allclose(result, (plus - minus) / 2e-6, atol=2e-8, rtol=2e-8)
    gradient = wrapper.gradient(positions, velocities)
    directional = np.sum(gradient[0] * direction) + np.sum(gradient[1] * velocity_direction)
    numerical = (wrapper.value(positions + 1e-6 * direction, velocities + 1e-6 * velocity_direction) -
                 wrapper.value(positions - 1e-6 * direction, velocities - 1e-6 * velocity_direction)) / 2e-6
    assert_allclose(directional, numerical, atol=1e-8, rtol=2e-7)
    flattened = wrapper.gradient(positions.reshape(-1, 3), velocities.reshape(-1, 3))
    assert flattened[0].shape == (positions.size // 3, 3)
    assert wrapper.supports_zero_energy_shortcut is False


@pytest.mark.skipif(not (native_directory() / "manifest.json").exists(), reason="native Windows FEMFX reference is not built")
def test_free_native_teacher_satisfies_frozen_work_bound_without_nonlinear_energy_allowance():
    from fracture_world.femfx import load_native

    native, _ = load_native()
    balance, _, positions, velocities = frozen_fixture(False, False, (0., -9.81, 0.))
    material = balance.material
    options = dict(enable_fracture=False, rayleigh_mass_damping=0., rayleigh_stiffness_damping=0.,
                   friction_coefficient=0., collision_bounds=[-1e6, 1e6, -1e6, 1e6, -1e6, 1e6])
    scene = native.Scene(balance.mesh.nodes, balance.mesh.tets, material, velocities[0],
                         np.zeros(4, dtype=bool), balance.dt, [0., -9.81, 0.], 1, options)
    try:
        initial = scene.state()
        final = scene.step()
        observed_balance = FrozenElasticBalance(balance.mesh, material, initial["corner_positions"],
                                                initial["corner_velocities"], float(np.float32(balance.dt)),
                                                np.array([0., -9.81, 0.]), np.zeros(0, dtype=bool),
                                                np.zeros(4, dtype=bool))
        energy = FrozenQuadraticEnergy(observed_balance)
        report = energy.balance(initial["corner_positions"], initial["corner_velocities"],
                                final["corner_positions"], final["corner_velocities"], 1.)
        assert report["energy_satisfied"], report
        assert report["allowed_external_work"] == 0.
        assert final["mpcg_stopping_observed"].all()
    finally:
        scene.close()
