import numpy as np
import pytest
from numpy.testing import assert_allclose
from scipy.sparse.linalg import spsolve
from types import SimpleNamespace

from fracture_world.dynamics_corrector import CorrectorTolerances, LinearizedContact
from fracture_world.dynamics_mechanics import FrozenElasticBalance
from fracture_world.dynamics_stabilization import contact_jacobian, solve_stabilization
from fracture_world.mesh import VolumeMesh


def stabilization_fixture(disconnected=False, fixed=()):
    nodes = np.array([[0., 0., 0.], [1., 0., 0.], [0., 1., 0.], [0., 0., 1.]])
    if disconnected:
        nodes = np.vstack((nodes, nodes + [3., 0., 0.]))
    tets = np.array([[0, 1, 2, 3], [4, 5, 6, 7]]) if disconnected else np.array([[0, 1, 2, 3]])
    mesh = VolumeMesh(nodes, tets)
    positions = mesh.nodes[mesh.tets].reshape(-1, 3)
    velocities = np.zeros_like(positions)
    supports = np.zeros(len(positions), dtype=bool)
    supports[list(fixed)] = True
    material = dict(rest_density=2., youngs_modulus=10., poissons_ratio=.25, fracture_stress_threshold=10000.)
    balance = FrozenElasticBalance(mesh, material, positions, velocities, .01, np.zeros(3),
                                   np.zeros(len(mesh.internal_faces), dtype=bool), supports)
    return balance, velocities, CorrectorTolerances()


def test_deep_initial_penetration_produces_contact_response_without_changing_velocity():
    balance, velocity, tolerance = stabilization_fixture()
    contact = LinearizedContact.pair(0, -1, [0., 1., 0.], -.2)
    offset, report = solve_stabilization(balance, (contact,), velocity, 1e-6, 1., tolerance)
    assert report["status"] == "pass", report
    assert offset is not None and offset[0, 1] > .19
    assert contact.gap + offset[0, 1] >= -1e-6 - report["normalized_tolerance"]
    assert report["teacher_future_used"] is False
    assert report["kinetic_state_changed"] is False
    assert report["engine_calls"] == 0
    assert_allclose(velocity, 0., atol=0.)


def test_no_contact_and_already_separating_contact_return_zero_offset():
    balance, velocity, tolerance = stabilization_fixture()
    for contacts in ((), (LinearizedContact.pair(0, -1, [0., 1., 0.], .02),)):
        offset, report = solve_stabilization(balance, contacts, velocity, 1e-6, 1., tolerance)
        assert report["status"] == "pass"
        assert_allclose(offset, 0., atol=0.)
        assert report["objective"] == 0.


def test_stabilization_includes_final_velocity_and_moving_environment_in_endpoint_gap():
    balance, velocity, tolerance = stabilization_fixture()
    velocity[:, 1] = .3
    contact = LinearizedContact.pair(0, -1, [0., 1., 0.], -.01, environment_velocity=.1)
    offset, report = solve_stabilization(balance, (contact,), velocity, 1e-6, 1., tolerance)
    gap = contact.gap + balance.dt * (.3 - .1) + offset[0, 1]
    assert report["status"] == "pass", report
    assert gap >= -1e-6 - report["normalized_tolerance"]
    assert offset[0, 1] < .01


def test_fixed_support_remains_fixed_and_incompatible_fixed_contact_is_unknown():
    balance, velocity, tolerance = stabilization_fixture(fixed=(0,))
    offset, report = solve_stabilization(balance, (LinearizedContact.pair(1, -1, [0., 1., 0.], -.1),),
                                         velocity, 1e-6, 1., tolerance)
    assert report["status"] == "pass", report
    assert_allclose(offset[0], 0., atol=report["normalized_tolerance"])
    assert offset[1, 1] > .09
    impossible, report = solve_stabilization(balance, (LinearizedContact.pair(0, -1, [0., 1., 0.], -.1),),
                                             velocity, 1e-6, 1., tolerance)
    assert impossible is None
    assert report["status"] == "unknown"


def test_internal_contact_reaction_is_equal_and_opposite_and_preserves_center_of_mass():
    balance, velocity, tolerance = stabilization_fixture(disconnected=True)
    contact = LinearizedContact.pair(0, 4, [1., 0., 0.], -.1)
    offset, report = solve_stabilization(balance, (contact,), velocity, 1e-6, 1., tolerance)
    assert report["status"] == "pass", report
    nodal = offset[balance.representatives]
    reaction = (balance.matrix @ nodal.ravel()).reshape(-1, 3)
    assert_allclose(reaction[:4].sum(axis=0), -reaction[4:].sum(axis=0), atol=2e-10)
    assert_allclose(np.sum(balance.node_masses[:, None] * nodal, axis=0), 0., atol=2e-10)
    assert contact.gap + offset[0, 0] - offset[4, 0] >= -1e-6 - report["normalized_tolerance"]


def test_stabilization_certificate_checks_primal_dual_stationarity_and_complementarity():
    balance, velocity, tolerance = stabilization_fixture()
    contact = LinearizedContact.pair(0, -1, [0., 1., 0.], -.1)
    scale, geometry = .2, 1e-6
    offset, report = solve_stabilization(balance, (contact,), velocity, geometry, scale, tolerance)
    assert report["status"] == "pass", report
    jacobian = contact_jacobian(balance, (contact,))
    displacement = offset[balance.representatives].ravel() / scale
    metric = .5 * (balance.matrix + balance.matrix.T) / balance.node_masses.sum()
    response = metric @ displacement
    multiplier = response[1]
    lower = (-geometry - contact.gap) / scale
    slack = float((jacobian @ displacement)[0] - lower)
    threshold = report["normalized_tolerance"]
    assert multiplier >= -threshold
    assert slack >= -threshold
    assert_allclose(response, jacobian.T.toarray().ravel() * multiplier, atol=threshold)
    assert abs(multiplier * slack) <= threshold
    assert report["stationarity_residual"] <= threshold
    assert report["complementarity_residual"] <= threshold


def test_stabilization_cancellation_and_invalid_inputs_are_explicit():
    balance, velocity, tolerance = stabilization_fixture()
    contact = LinearizedContact.pair(0, -1, [0., 1., 0.], -.1)
    offset, report = solve_stabilization(balance, (contact,), velocity, 1e-6, 1., tolerance, cancel=lambda: True)
    assert offset is None and report["status"] == "interrupted"
    for geometry, scale in ((-1., 1.), (np.nan, 1.), (1e-6, 0.)):
        with pytest.raises(ValueError):
            solve_stabilization(balance, (contact,), velocity, geometry, scale, tolerance)
    with pytest.raises(ValueError):
        solve_stabilization(balance, (contact,), np.full_like(velocity, np.nan), 1e-6, 1., tolerance)
    with pytest.raises(ValueError):
        solve_stabilization(balance, (LinearizedContact.pair(10, -1, [0., 1., 0.], -.1),), velocity, 1e-6, 1., tolerance)


def test_feasible_stationary_but_noncomplementary_solver_output_is_not_certified(monkeypatch):
    import clarabel

    balance, velocity, tolerance = stabilization_fixture()
    contact = LinearizedContact.pair(0, -1, [0., 1., 0.], -.1)
    jacobian = contact_jacobian(balance, (contact,))
    metric = .5 * (balance.matrix + balance.matrix.T) / balance.node_masses.sum()
    response = spsolve(metric.tocsc(), jacobian.T.toarray().ravel())
    multiplier = 2 * (.1 - 1e-6) / float((jacobian @ response)[0])
    candidate = SimpleNamespace(status="Solved", x=multiplier * response, z=np.array([multiplier]))

    class NoncomplementarySolver:
        def __init__(self, *args):
            pass

        def solve(self):
            return candidate

    monkeypatch.setattr(clarabel, "DefaultSolver", NoncomplementarySolver)
    offset, report = solve_stabilization(balance, (contact,), velocity, 1e-6, 1., tolerance)
    assert offset is None
    assert report["status"] == "unknown"
    assert report["stationarity_residual"] <= report["normalized_tolerance"]
    assert report["maximum_normalized_violation"] <= report["normalized_tolerance"]
    assert report["complementarity_residual"] > report["normalized_tolerance"]
