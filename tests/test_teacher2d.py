import numpy as np
import pytest

from fracture_world.teacher2d import InternalForce, Material, Pull, refinement_error, simulate
from fracture_world.topology import Mesh, rectangle, split_edges, triangle_areas


def energy(mesh, material, position):
    reference = mesh.rest[mesh.triangles]
    current = position[mesh.triangles]
    reference_matrix = np.stack((reference[:, 1] - reference[:, 0], reference[:, 2] - reference[:, 0]), axis=-1)
    current_matrix = np.stack((current[:, 1] - current[:, 0], current[:, 2] - current[:, 0]), axis=-1)
    deformation = current_matrix @ np.linalg.inv(reference_matrix)
    strain = (np.swapaxes(deformation, -1, -2) @ deformation - np.eye(2)) / 2
    shear = material.young / (2 * (1 + material.poisson))
    lame = material.young * material.poisson / ((1 + material.poisson) * (1 - 2 * material.poisson))
    density = shear * np.sum(strain ** 2, axis=(1, 2)) + lame * np.trace(strain, axis1=1, axis2=2) ** 2 / 2
    return np.sum(density * triangle_areas(mesh)) * material.thickness


def test_external_force_matches_independent_energy_gradient():
    mesh = rectangle(2, 2)
    material = Material(young=9.0, poisson=0.2, thickness=0.7)
    position = mesh.rest @ np.array([[1.04, 0.03], [0.01, 0.98]]).T
    force = InternalForce(mesh, material)(position)
    epsilon = 1e-6
    derivative = np.empty_like(position)
    for node in range(len(position)):
        for coordinate in range(2):
            positive, negative = position.copy(), position.copy()
            positive[node, coordinate] += epsilon
            negative[node, coordinate] -= epsilon
            derivative[node, coordinate] = (energy(mesh, material, positive) - energy(mesh, material, negative)) / (2 * epsilon)
    assert np.allclose(force, -derivative, rtol=1e-7, atol=1e-9)
    assert np.linalg.norm(force.sum(axis=0)) < 1e-12


def test_rigid_motion_has_zero_force_and_evaluation_has_no_history():
    mesh = rectangle(2, 2)
    force = InternalForce(mesh, Material())
    angle = 0.7
    rotation = np.array([[np.cos(angle), -np.sin(angle)], [np.sin(angle), np.cos(angle)]])
    position = mesh.rest @ rotation.T + [0.3, -0.4]
    assert np.max(np.abs(force(position))) < 1e-12
    deformed = mesh.rest * [1.04, 1.0]
    first = force(deformed)
    force(mesh.rest)
    assert np.array_equal(first, force(deformed))


def test_force_scales_with_thickness():
    mesh = rectangle(2, 2)
    position = mesh.rest * [1.02, 1.0]
    thin = InternalForce(mesh, Material(thickness=0.5))(position)
    thick = InternalForce(mesh, Material(thickness=2.0))(position)
    assert np.allclose(thick, 4 * thin, atol=1e-12)


def test_full_cut_prevents_cross_component_force():
    mesh = rectangle(2, 2)
    cut = split_edges(mesh, [[1, 4], [4, 7]])
    centroids = cut.rest[cut.triangles].mean(axis=1)
    left = np.unique(cut.triangles[centroids[:, 0] < 0.5])
    right = np.unique(cut.triangles[centroids[:, 0] > 0.5])
    position = cut.rest.copy()
    position[left] += [-0.1, 0.0]
    force = InternalForce(cut, Material())(position)
    assert np.max(np.abs(force[right])) < 1e-12
    assert np.max(np.abs(force[left])) < 1e-12


def test_pull_matches_velocity_and_holds_after_ramp():
    pull = Pull(0.03, 0.1)
    time = 0.037
    epsilon = 1e-7
    finite_difference = (pull.evaluate(time + epsilon)[0] - pull.evaluate(time - epsilon)[0]) / (2 * epsilon)
    assert pull.evaluate(time)[1] == pytest.approx(finite_difference, rel=1e-8)
    assert pull.evaluate(0.2) == (0.03, 0.0)
    assert pull.evaluate(0.0) == (0.0, 0.0)


def test_teacher_boundary_impulse_and_time_refinement():
    mesh = rectangle(2, 2)
    cut = split_edges(mesh, [[1, 4]])
    time = np.linspace(0, 0.04, 9)
    material = Material(young=10.0)
    first = simulate(cut, material, time, amplitude=0.005, ramp_time=0.02,
                     rtol=1e-9, atol=1e-11, max_step=0.0025)
    second = simulate(cut, material, time, amplitude=0.005, ramp_time=0.02,
                      rtol=1e-10, atol=1e-12, max_step=0.00125)
    assert np.max(np.abs(first.arrays["position"] - second.arrays["position"])) < 1e-8
    assert np.max(np.abs(first.arrays["velocity"] - second.arrays["velocity"])) < 1e-7
    mass = first.mass
    momentum_change = np.sum(np.diff(first.arrays["velocity"], axis=0) * mass[None, :, None], axis=1)
    impulse = first.arrays["boundary_impulse"].sum(axis=1)
    assert np.max(np.abs(momentum_change - impulse)) < 1e-9
    assert first.metadata["material"]["calibrated"] is False
    assert first.metadata["contact"] == "excluded"
    assert np.any(first.arrays["position"][-1] != cut.rest)
    assert first.metadata["trapezoidal_position_defect_max"] > 0
    assert first.metadata["force_evaluations"] > first.metadata["solver_evaluations"]
    assert first.metadata["force_runtime_seconds"] > 0
    audit = refinement_error(first, second)
    assert audit["position_max_difference"] < 1e-8
    assert audit["velocity_max_difference"] < 1e-7


def test_rest_teacher_is_exactly_stationary():
    mesh = rectangle(2, 2)
    case = simulate(mesh, Material(), np.linspace(0, 0.01, 3), amplitude=0)
    assert np.allclose(case.arrays["position"], mesh.rest[None], atol=1e-14)
    assert np.allclose(case.arrays["velocity"], 0, atol=1e-14)
    assert np.allclose(case.arrays["boundary_impulse"], 0, atol=1e-14)


def test_asymmetric_cut_reaction_matches_nonzero_momentum_change():
    mesh = split_edges(rectangle(3, 2), [[1, 5]])
    case = simulate(mesh, Material(young=10), np.linspace(0, 0.04, 9),
                    amplitude=0.005, ramp_time=0.02, max_step=0.00125)
    momentum = np.sum(case.arrays["velocity"] * case.mass[None, :, None], axis=1)
    supplied = np.cumsum(case.arrays["boundary_impulse"].sum(axis=1), axis=0)
    assert np.max(np.abs(momentum)) > 1e-6
    assert np.max(np.abs(momentum[1:] - momentum[0] - supplied)) < 1e-9


@pytest.mark.parametrize("values", [{"young": 0}, {"poisson": 0.5}, {"density": -1}, {"thickness": np.nan}])
def test_invalid_material_is_rejected(values):
    with pytest.raises(ValueError):
        Material(**values)


def test_invalid_teacher_time_is_rejected():
    with pytest.raises(ValueError, match="timestamps"):
        simulate(rectangle(2, 2), Material(), np.array([0.1, 0.2]))
