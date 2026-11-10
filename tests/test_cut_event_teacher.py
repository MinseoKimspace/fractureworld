import numpy as np
import pytest

from fracture_world.cut_event_teacher import (
    elastic_energy,
    simulate_event,
    simulate_segment,
    transfer_audit,
)
from fracture_world.teacher2d import InternalForce, Material, Pull, simulate
from fracture_world.topology import Mesh, lift_state, rectangle, split_edges


def meshes():
    original = rectangle(3, 3)
    partial = split_edges(original, [[1, 5], [5, 9]])
    return partial, split_edges(partial, [[9, 13]])


def test_energy_matches_external_force_and_cut_transfer():
    before, after = meshes()
    material = Material(young=9, poisson=0.2, density=4, thickness=0.7)
    position = before.rest @ np.array([[1.04, 0.03], [0.01, 0.98]]).T + [0.3, -0.2]
    velocity = np.column_stack((0.4 + before.rest[:, 1], -0.2 - before.rest[:, 0]))
    force = InternalForce(before, material)(position)
    epsilon = 1e-6
    derivative = np.empty_like(position)
    for node in range(len(position)):
        for coordinate in range(2):
            positive, negative = position.copy(), position.copy()
            positive[node, coordinate] += epsilon
            negative[node, coordinate] -= epsilon
            derivative[node, coordinate] = (
                elastic_energy(before, material, positive) - elastic_energy(before, material, negative)
            ) / (2 * epsilon)
    np.testing.assert_allclose(force, -derivative, rtol=1e-7, atol=1e-9)
    audit = transfer_audit(before, after, material, position, velocity)
    assert audit["status"] == "conserved"
    assert all(audit["checks"].values())
    assert audit["kinetic_energy_before"] > 0
    assert audit["elastic_energy_before"] > 0
    assert audit["material_corner_position_max_change"] == 0
    assert audit["material_corner_velocity_max_change"] == 0
    assert audit["prescribed_impulse"] == [0, 0]
    assert audit["prescribed_event_work"] == 0


def test_event_uses_global_time_and_preserves_nonzero_motion():
    before_mesh, after_mesh = meshes()
    material = Material(young=10)
    time = np.linspace(0, 0.04, 9)
    load = {"amplitude": 0.005, "ramp_time": 0.03}
    before, after, audit = simulate_event(
        before_mesh, after_mesh, material, time, time[4], load,
        {"max_step": 0.00125},
    )
    expected_position, expected_velocity = lift_state(
        before_mesh, after_mesh, before.arrays["position"][-1], before.arrays["velocity"][-1],
    )
    np.testing.assert_array_equal(after.arrays["position"][0], expected_position)
    np.testing.assert_array_equal(after.arrays["velocity"][0], expected_velocity)
    assert np.max(np.abs(expected_velocity)) > 0.01
    assert np.max(np.abs(expected_position - after_mesh.rest)) > 0.001
    mask, boundary_position, boundary_velocity = Pull(**load).boundary(after_mesh, time[4:])
    np.testing.assert_array_equal(after.arrays["position"][:, mask], boundary_position[:, mask])
    np.testing.assert_array_equal(after.arrays["velocity"][:, mask], boundary_velocity[:, mask])
    assert after.metadata["initial_time"] == time[4]
    assert after.metadata["boundary_time_origin"] == "global"
    assert audit["status"] == "conserved"
    assert audit["added_cut_edges"] == [(9, 13)]
    assert audit["nodes_after"] > audit["nodes_before"]


def test_segment_restart_matches_uninterrupted_teacher_and_impulse():
    mesh, _ = meshes()
    material = Material(young=10)
    time = np.linspace(0, 0.04, 9)
    load = {"amplitude": 0.005, "ramp_time": 0.03}
    parameters = {"rtol": 1e-10, "atol": 1e-12, "max_step": 0.00125}
    full = simulate(mesh, material, time, **load, **parameters)
    before = simulate_segment(mesh, material, time[:5], load, **parameters)
    after = simulate_segment(mesh, material, time[4:], load,
                             before.arrays["position"][-1], before.arrays["velocity"][-1],
                             **parameters)
    for name in ("position", "velocity"):
        joined = np.concatenate((before.arrays[name], after.arrays[name][1:]))
        np.testing.assert_allclose(joined, full.arrays[name], rtol=1e-8, atol=1e-10)
    joined_impulse = np.concatenate((before.arrays["boundary_impulse"], after.arrays["boundary_impulse"]))
    np.testing.assert_allclose(joined_impulse, full.arrays["boundary_impulse"], rtol=1e-8, atol=1e-10)


def test_after_cut_follows_new_forces_and_boundary_impulse_balance():
    before_mesh, after_mesh = meshes()
    material = Material(young=10)
    time = np.linspace(0, 0.08, 17)
    event_time = time[4]
    load = {"amplitude": 0.005, "ramp_time": 0.03}
    before, after, _ = simulate_event(before_mesh, after_mesh, material, time, event_time,
                                     load, {"max_step": 0.00125})
    uncut = simulate_segment(before_mesh, material, time[4:], load,
                             before.arrays["position"][-1], before.arrays["velocity"][-1],
                             max_step=0.00125)
    acceleration_before = before.arrays["internal_force"][-1] / before.mass[:, None]
    acceleration_after = after.arrays["internal_force"][0] / after.mass[:, None]
    assert np.max(np.abs(acceleration_after[after_mesh.triangles]
                         - acceleration_before[before_mesh.triangles])) > 0.01
    assert np.max(np.abs(after.arrays["position"][-1, after_mesh.triangles]
                         - uncut.arrays["position"][-1, before_mesh.triangles])) > 1e-5
    for case in (before, after):
        momentum = np.sum(case.arrays["velocity"] * case.mass[None, :, None], axis=1)
        supplied = np.cumsum(case.arrays["boundary_impulse"].sum(axis=1), axis=0)
        np.testing.assert_allclose(momentum[1:] - momentum[0], supplied, atol=1e-10)


def test_stationary_event_does_not_inject_motion():
    before_mesh, after_mesh = meshes()
    before, after, audit = simulate_event(
        before_mesh, after_mesh, Material(), np.linspace(0, 0.01, 5), 0.005,
        {"amplitude": 0, "ramp_time": 0.01},
    )
    for case in (before, after):
        np.testing.assert_allclose(case.arrays["position"],
                                   np.broadcast_to(case.mesh.rest, case.arrays["position"].shape), atol=1e-14)
        np.testing.assert_allclose(case.arrays["velocity"], 0, atol=1e-14)
        np.testing.assert_allclose(case.arrays["boundary_impulse"], 0, atol=1e-14)
    assert audit["status"] == "conserved"
    assert audit["elastic_energy_before"] == audit["elastic_energy_after"] == 0


def test_event_reference_refinement_agrees_on_both_segments():
    before_mesh, after_mesh = meshes()
    time = np.linspace(0, 0.04, 9)
    args = (before_mesh, after_mesh, Material(young=10), time, time[4],
            {"amplitude": 0.005, "ramp_time": 0.03})
    coarse = simulate_event(*args, settings={"rtol": 1e-9, "atol": 1e-11, "max_step": 0.0025})
    fine = simulate_event(*args, settings={"rtol": 1e-10, "atol": 1e-12, "max_step": 0.00125})
    for first, second in zip(coarse[:2], fine[:2]):
        np.testing.assert_allclose(first.arrays["position"], second.arrays["position"], atol=1e-8)
        np.testing.assert_allclose(first.arrays["velocity"], second.arrays["velocity"], atol=1e-7)
        np.testing.assert_allclose(first.arrays["boundary_impulse"], second.arrays["boundary_impulse"], atol=1e-8)


@pytest.mark.parametrize("event_time", [0, 0.04, 0.021, np.nan])
def test_invalid_event_time_is_rejected(event_time):
    before, after = meshes()
    with pytest.raises(ValueError, match="Event time"):
        simulate_event(before, after, Material(), np.linspace(0, 0.04, 9), event_time,
                       {"amplitude": 0.005, "ramp_time": 0.03})


def test_event_cannot_remove_cuts_or_change_material_reference():
    before, after = meshes()
    time = np.linspace(0, 0.04, 9)
    load = {"amplitude": 0.005, "ramp_time": 0.03}
    for initial, final in ((after, before), (before, before)):
        with pytest.raises(ValueError, match="add cuts"):
            simulate_event(initial, final, Material(), time, time[4], load)
    changed = Mesh(after.rest * 1.1, after.triangles, after.original_vertex, after.cut_edges)
    with pytest.raises(ValueError, match="rest positions"):
        transfer_audit(before, changed, Material(), before.rest, np.zeros_like(before.rest))


def test_restart_requires_explicit_finite_state_and_global_boundary_agreement():
    before, _ = meshes()
    time = np.linspace(0.01, 0.03, 5)
    load = {"amplitude": 0.005, "ramp_time": 0.03}
    with pytest.raises(ValueError, match="explicit initial"):
        simulate_segment(before, Material(), time, load)
    with pytest.raises(ValueError, match="both initial"):
        simulate_segment(before, Material(), time, load, before.rest)
    with pytest.raises(ValueError, match="global-time"):
        simulate_segment(before, Material(), time, load, before.rest, np.zeros_like(before.rest))
    position, velocity = before.rest.copy(), np.zeros_like(before.rest)
    position[0, 0] = np.nan
    with pytest.raises(ValueError, match="finite"):
        simulate_segment(before, Material(), time, load, position, velocity)
