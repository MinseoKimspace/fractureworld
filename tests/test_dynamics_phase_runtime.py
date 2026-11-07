from types import SimpleNamespace

import numpy as np
import pytest
import torch
from numpy.testing import assert_allclose

from fracture_world.dynamics_arithmetic import position_publication_envelope
from fracture_world.dynamics_corrector import CorrectorTolerances
from fracture_world.dynamics_data import material_topology
from fracture_world.dynamics_initial import DynamicsInitial
from fracture_world.dynamics_mechanics import FrozenElasticBalance
from fracture_world.dynamics_runtime import RolloutConfig, _correct, fracture_candidates, step_contacts
from fracture_world.dynamics_stabilization import solve_stabilization
from fracture_world.mesh import VolumeMesh


def phase_fixture(*, penetrated=False, shared=False):
    nodes = np.array([[0., 0., 0.], [1., 0., 0.], [0., 1., 0.], [0., 0., 1.], [1., 1., 1.]])
    mesh = VolumeMesh(nodes if shared else nodes[:4],
                      np.array([[0, 1, 2, 3], [4, 1, 2, 3]]) if shared else np.array([[0, 1, 2, 3]]))
    material = dict(rest_density=6., youngs_modulus=12., poissons_ratio=0., fracture_stress_threshold=1e6)
    topology = material_topology(mesh, material["rest_density"])
    positions = topology.rest_positions.copy()
    if penetrated:
        positions[:, 1] -= .125
    velocities = np.zeros_like(positions)
    broken = np.zeros(len(mesh.internal_faces), dtype=bool)
    fixed = np.zeros(len(positions), dtype=bool)
    settings = dict(rayleigh_mass_damping=0., rayleigh_stiffness_damping=0., friction_coefficient=0.)
    metadata = dict(teacher_physics=settings, native_settings={**settings, "dist_contact_bias": 0., "dist_contact_threshold": .003})
    if penetrated:
        metadata.update(floor_height_m=0., up_axis="y")
    initial = DynamicsInitial(topology, positions, velocities, broken, fixed, 0., .125, np.zeros(3), material,
                              "source-phase", "a" * 64, metadata)
    model = SimpleNamespace(config=dict(velocity_scale=1., motion_prior="constant_acceleration"))
    proposal = dict(positions=torch.as_tensor(positions.copy()), velocities=torch.as_tensor(velocities.copy()),
                    fracture_probability=torch.full((len(broken),), .75), support_release_probability=torch.zeros(len(positions)))
    config = RolloutConfig(fracture_policy="marginal_search", fracture_samples=0, mechanics_mode="proximal",
                           max_iterations=20000, max_contact_iterations=50)
    return initial, model, proposal, config


@pytest.mark.parametrize("method", ["central"])
def test_actual_corrector_repairs_initial_penetration_by_source_position_response_without_spurious_velocity(method):
    initial, model, proposal, config = phase_fixture(penetrated=True)
    result, report = _correct(initial, initial.positions, initial.velocities, proposal, initial.broken, initial.fixed,
                              model, config, method)
    assert result is not None, report
    assert result.converged and report["status"] == "converged"
    assert report["geometry"]["valid"]
    stabilization = result.diagnostics["stabilization"]
    assert stabilization["status"] == "pass"
    assert stabilization["maximum_stabilization_metres"] > .1
    assert stabilization["frozen_offset_consistency_residual"] <= config.absolute_tolerance + config.relative_tolerance
    energy = result.diagnostics["energy"]
    assert "algorithm" in energy
    assert "coordinate_reduction_seconds" in energy
    assert energy["energy_satisfied"]
    assert energy["position_stabilization_excluded"] is True
    assert "transport" in energy["law"]
    assert energy["allowed_external_work"] == 0.
    independent = energy["stabilization"]
    assert independent["numerical_stabilization_work"]["total"] > 0.
    assert independent["nonlinear_endpoint"]["status"] == "diagnostic_only"
    assert independent["work_credited_as_allowed_work"] is False
    minimum_corner_mass = float(initial.topology.masses.min())
    energy_velocity_bound = np.sqrt(2 * energy["normalized_tolerance"] * energy["energy_scale"] / minimum_corner_mass)
    assert float(np.max(np.abs(result.velocities))) <= energy_velocity_bound
    balance = FrozenElasticBalance(initial.topology.mesh, initial.material, initial.positions, initial.velocities,
                                   initial.dt, initial.gravity, initial.broken, initial.fixed, coefficient_model="native_float32")
    contacts = step_contacts(initial, initial.positions, initial.positions, initial.broken, config)
    expected, certificate = solve_stabilization(balance, contacts, result.velocities, config.geometry_tolerance,
                                                initial.dt * model.config["velocity_scale"], CorrectorTolerances())
    assert certificate["status"] == "pass"
    actual = result.positions.reshape(-1, 3) - initial.positions - initial.dt * result.velocities.reshape(-1, 3)
    velocity_envelope = np.maximum(np.abs(result.velocities.reshape(-1, 3)), model.config["velocity_scale"])
    roundoff = position_publication_envelope(initial.positions, initial.dt, velocity_envelope, expected)
    bound = float(np.max(roundoff)) + initial.dt * (config.absolute_tolerance + config.relative_tolerance)
    assert_allclose(actual, expected, atol=bound, rtol=0.)


@pytest.mark.parametrize("method", ["central"])
def test_no_contact_source_phase_has_no_stabilization_slack_for_an_old_checkpoint_shaped_proposal(method):
    initial, model, proposal, config = phase_fixture()
    proposal["positions"] = proposal["positions"] + torch.tensor([.02, 0., 0.])
    result, report = _correct(initial, initial.positions, initial.velocities, proposal, initial.broken, initial.fixed,
                              model, config, method)
    assert result is not None, report
    assert result.diagnostics["stabilization"]["objective"] == 0.
    assert result.diagnostics["stabilization"]["frozen_offset_consistency_residual"] == 0.
    energy = result.diagnostics["energy"]
    energy_velocity_bound = np.sqrt(2 * energy["normalized_tolerance"] * energy["energy_scale"] /
                                    float(initial.topology.masses.min()))
    assert float(np.max(np.abs(result.velocities))) <= energy_velocity_bound
    velocity = result.velocities.reshape(-1, 3)
    residual = result.positions.reshape(-1, 3) - initial.positions - initial.dt * velocity
    envelope = position_publication_envelope(initial.positions, initial.dt, np.maximum(np.abs(velocity), 1.),
                                              np.zeros_like(initial.positions))
    assert float(np.max(np.abs(residual))) <= float(np.max(envelope)) + initial.dt * (
        config.absolute_tolerance + config.relative_tolerance)
    assert result.diagnostics["energy"]["stabilization"]["numerical_stabilization_work"]["total"] == 0.


def test_current_fracture_pool_is_validated_by_the_central_source_phase_corrector():
    initial, model, proposal, config = phase_fixture(shared=True)
    batch = dict(broken=torch.as_tensor(initial.broken))
    candidates = fracture_candidates(model, batch, proposal, config, 0)
    assert len(candidates) == 2
    for candidate in candidates:
        result, report = _correct(initial, initial.positions, initial.velocities, proposal, candidate, initial.fixed,
                                  model, config, "central", previous_broken=initial.broken, previous_fixed=initial.fixed)
        assert result is not None, report
        assert result.diagnostics["constraints_satisfied"]
        assert result.diagnostics["energy"]["energy_satisfied"]
        assert report["geometry"]["valid"]
