from itertools import permutations, product
from pathlib import Path

import numpy as np
import pytest
from scipy import sparse

from fracture_world.dynamics_femfx_math import fma32, native_corner_masses, native_fracture_stress, native_frozen_operator, shape_parameters
from fracture_world.mesh import VolumeMesh


def tetrahedron(translation=0.):
    nodes = .05 * np.array([[0, 0, 0], [1, 0, 0], [0, 1, 0], [0, 0, 1]], dtype=float) + translation
    return VolumeMesh(nodes, [[0, 1, 2, 3]])


def material():
    return dict(rest_density=1000., youngs_modulus=10000., poissons_ratio=.25, fracture_stress_threshold=1e12)


def block():
    divisions = (6, 2, 2)
    nodes = np.array(list(product(*(np.linspace(0, size, count + 1) for size, count in zip((.12, .04, .04), divisions)))))
    indices = np.arange(len(nodes)).reshape(tuple(count + 1 for count in divisions))
    cells = []
    for lower in product(*(range(count) for count in divisions)):
        for axes in permutations(range(3)):
            vertex = np.array(lower)
            cell = [indices[tuple(vertex)]]
            for axis in axes:
                vertex = vertex.copy()
                vertex[axis] += 1
                cell.append(indices[tuple(vertex)])
            cells.append(cell)
    return VolumeMesh(nodes, cells)


def operator(mesh, *, fixed=None, broken=None, positions=None, velocities=None, **kwargs):
    positions = mesh.nodes[mesh.tets] if positions is None else positions
    velocities = np.zeros_like(positions) if velocities is None else velocities
    fixed = np.zeros((len(mesh.tets), 4), dtype=bool) if fixed is None else fixed
    broken = np.zeros(len(mesh.internal_faces), dtype=bool) if broken is None else broken
    return native_frozen_operator(mesh, material(), positions, velocities, .001, [0., 0., 0.], broken, fixed, **kwargs)


def test_fused_float32_multiply_add_resolves_double_rounding_midpoint():
    left = np.float32(1.5)
    right = np.nextafter(np.float32(1.), np.float32(np.inf))
    product64 = np.float64(left) * np.float64(right)
    rounded = np.float32(product64)
    lower = np.nextafter(rounded, np.float32(-np.inf))
    tiny = np.float32(2.**-100)
    assert np.float32(product64 - np.float64(tiny)) == rounded
    assert fma32(left, right, -tiny) == lower
    assert fma32(left, right, tiny) == rounded


def test_fused_float32_multiply_add_handles_broadcasting_cancellation_and_subnormal_values():
    smallest = np.nextafter(np.float32(0.), np.float32(1.))
    result = fma32(np.array([1., 1.5, 1., smallest], dtype=np.float32), np.float32(2.),
                   np.array([-2., -2., 0., -smallest], dtype=np.float32))
    np.testing.assert_array_equal(result, np.array([0., 1., 2., smallest], dtype=np.float32))
    assert result.dtype == np.float32


def test_unit_shape_parameters_have_exact_barycentric_gradient_and_native_volume():
    rest = np.array([[[0., 0., 0.], [1., 0., 0.], [0., 1., 0.], [0., 0., 1.]]])
    barycentric, volumes = shape_parameters(rest)
    np.testing.assert_array_equal(barycentric[0], [[-1., -1., -1., 1.], [1., 0., 0., 0.],
                                                [0., 1., 0., 0.], [0., 0., 1., 0.]])
    np.testing.assert_array_equal(volumes, [np.float32(1. / 6.)])


def test_native_mass_definition_is_shared_by_frozen_operator_and_mass_helper():
    mesh = block()
    value = operator(mesh)
    np.testing.assert_array_equal(value.corner_masses, native_corner_masses(mesh, material()["rest_density"]))
    expected = np.zeros(len(value.node_masses), dtype=np.float32)
    np.add.at(expected, value.groups, value.corner_masses.astype(np.float32))
    np.testing.assert_array_equal(value.node_masses, expected)


def test_frozen_operator_is_sparse_current_state_only_and_reproducible():
    mesh = block()
    first, second = operator(mesh), operator(mesh)
    assert sparse.isspmatrix_csr(first.matrix)
    assert first.matrix.nnz <= 16 * 9 * len(mesh.tets)
    np.testing.assert_array_equal(first.rhs, second.rhs)
    np.testing.assert_array_equal(first.matrix.data, second.matrix.data)
    assert first.ordering["engine_calls"] == 0
    assert first.ordering["future_observation_inputs"] is False
    assert first.ordering["postfracture_native_order_verified"] is False


def test_fixed_rows_match_native_identity_and_remain_separate_from_reaction_forces():
    mesh = tetrahedron()
    fixed = np.array([True, False, False, False])
    value = operator(mesh, fixed=fixed)
    expected = np.zeros((3, 12))
    expected[:, :3] = np.eye(3)
    np.testing.assert_array_equal(value.matrix[:3].toarray(), expected)
    assert value.stiffness[:3].nnz > 0


def test_current_fracture_separates_operator_components_without_any_future_state():
    mesh = block()
    broken = np.ones(len(mesh.internal_faces), dtype=bool)
    value = operator(mesh, broken=broken)
    assert len(value.node_masses) == 4 * len(mesh.tets)
    assert value.centers_of_mass.shape == (len(mesh.tets), 3)
    assert value.matrix.nnz <= 16 * 9 * len(mesh.tets)


def test_current_cached_elastic_state_reconstructs_same_operator_without_future_observations():
    mesh = block()
    initial = operator(mesh)
    restored = operator(mesh, current_rotations=initial.rotations,
                        current_centers_of_mass=initial.tet_centers_of_mass,
                        current_nodal_masses=initial.node_masses)
    np.testing.assert_array_equal(initial.matrix.data, restored.matrix.data)
    np.testing.assert_array_equal(initial.rhs, restored.rhs)
    assert restored.ordering["cached_rotations_supplied"] is True
    assert restored.ordering["cached_centers_of_mass_supplied"] is True
    assert restored.ordering["current_nodal_masses_supplied"] is True


def test_current_nodal_mass_input_preserves_solver_mass_instead_of_reconstructing_its_roundoff():
    mesh = block()
    initial = operator(mesh)
    masses = np.nextafter(initial.node_masses.astype(np.float32), np.float32(np.inf))
    repeated = masses[initial.groups].reshape(-1, 4)
    restored = operator(mesh, current_nodal_masses=repeated)
    np.testing.assert_array_equal(restored.node_masses, masses)
    assert not np.array_equal(restored.node_masses, initial.node_masses)


@pytest.mark.parametrize("kwargs", [dict(current_rotations=np.zeros((1, 3, 3))),
                                    dict(current_centers_of_mass=np.zeros((4, 3))),
                                    dict(current_nodal_masses=[0., 1., 1., 1.])])
def test_invalid_current_cached_elastic_state_is_rejected(kwargs):
    with pytest.raises(ValueError):
        operator(tetrahedron(), **kwargs)


def test_native_fracture_stress_preserves_compressive_spectral_radius_and_prefracture_scope():
    mesh = tetrahedron()
    positions = mesh.nodes[mesh.tets] * [.8, 1., 1.]
    report = native_fracture_stress(mesh, {**material(), "fracture_stress_threshold": 1.}, positions,
                                    np.zeros(0, dtype=bool))
    assert report["stress_ratios"][0] > 1.
    assert np.min(np.linalg.eigvalsh(report["fracture_tensor"][0])) < -1.
    assert report["full_topology_certified"] is False
    assert report["engine_calls"] == 0
    assert "previous connectivity" in report["phase"]


def test_native_fracture_stress_rejects_nonpositive_threshold():
    mesh = tetrahedron()
    with pytest.raises(ValueError, match="threshold"):
        native_fracture_stress(mesh, {**material(), "fracture_stress_threshold": 0.}, mesh.nodes[mesh.tets],
                               np.zeros(0, dtype=bool))


@pytest.mark.parametrize("kwargs", [dict(vertex_order=[0, 0, 2, 3]), dict(tetrahedron_order=[1]),
                                    dict(vertex_order=[0., 1., 2., 3.])])
def test_invalid_current_numeric_ordering_is_rejected(kwargs):
    with pytest.raises(ValueError, match="order"):
        operator(tetrahedron(), **kwargs)


@pytest.mark.parametrize("corruption", ["nan_position", "float32_overflow", "wrong_supports", "wrong_fracture"])
def test_invalid_native_operator_states_are_rejected(corruption):
    mesh = tetrahedron()
    positions = mesh.nodes[mesh.tets].copy()
    fixed = np.zeros(4, dtype=bool)
    broken = np.zeros(0, dtype=bool)
    if corruption == "nan_position":
        positions[0, 0, 0] = np.nan
    elif corruption == "float32_overflow":
        positions[0, 0, 0] = 1e100
    elif corruption == "wrong_supports":
        fixed = np.zeros(4)
    else:
        broken = np.zeros(1, dtype=bool)
    with pytest.raises(ValueError):
        operator(mesh, positions=positions, fixed=fixed, broken=broken)


@pytest.mark.skipif(not Path(".cache/femfx-build/manifest.json").exists(), reason="native Windows FEMFX reference is not built")
@pytest.mark.parametrize("case", ["rest", "translated_rest", "translated_current", "stretched", "rotated", "gravity", "fixed_beam"])
def test_current_state_math_reconstructs_actual_native_rhs_and_masses(case):
    from fracture_world.femfx import load_native

    native, _ = load_native()
    if native.capabilities().get("solver_observation_schema") != "fracture-world.femfx-mpcg-observation.v1":
        pytest.skip("native reference needs MPCG observation rebuild")
    mesh = block() if case == "fixed_beam" else tetrahedron(10. if case == "translated_rest" else 0.)
    current = mesh.nodes.copy()
    if case == "translated_current":
        current += 10.
    if case in {"stretched", "rotated"}:
        current *= [1.2, .95, 1.]
    if case == "rotated":
        sine, cosine = np.sin(.7), np.cos(.7)
        current = current @ np.array([[cosine, -sine, 0.], [sine, cosine, 0.], [0., 0., 1.]]).T
    fixed = mesh.nodes[:, 0] == 0 if case == "fixed_beam" else np.zeros(len(mesh.nodes), dtype=bool)
    velocity = np.zeros_like(current)
    if case == "fixed_beam":
        velocity[mesh.nodes[:, 0] == .12, 0] = .5
    gravity = [0., -9.81, 0.] if case == "gravity" else [0., 0., 0.]
    constitutive = {**material(), "youngs_modulus": 1e6} if case == "fixed_beam" else material()
    dt = .0001 if case == "fixed_beam" else .001
    options = dict(enable_fracture=False, friction_coefficient=0., rayleigh_mass_damping=0.,
                   rayleigh_stiffness_damping=0., record_observations=True)
    scene = native.Scene(mesh.nodes, mesh.tets, constitutive, velocity, fixed, dt, gravity, 1, options,
                         deformed_positions=current)
    try:
        previous = scene.state()
        for _ in range(3):
            final = scene.step()
            value = native_frozen_operator(mesh, constitutive, previous["corner_positions"], previous["corner_velocities"],
                                           dt, gravity, np.zeros(len(mesh.internal_faces), dtype=bool), fixed[mesh.tets])
            representatives = value.representatives
            np.testing.assert_array_equal(value.rhs.reshape(-1, 3), final["mpcg_rhs"].reshape(-1, 3)[representatives])
            np.testing.assert_array_equal(value.node_masses, final["mpcg_masses"].ravel()[representatives])
            solution = final["mpcg_solution"].reshape(-1, 3)[representatives]
            residual = (value.matrix @ solution.ravel() - value.rhs).reshape(-1, 3) / value.node_masses[:, None]
            free = ~fixed[mesh.tets].ravel()[representatives]
            assert np.max(np.abs(residual[free])) <= 1.1e-7
            previous = final
    finally:
        scene.close()


@pytest.mark.skipif(not Path(".cache/femfx-build/manifest.json").exists(), reason="native Windows FEMFX reference is not built")
def test_source_native_endpoint_stress_matches_all_cells_through_actual_fracture():
    from fracture_world.femfx import load_native, native_face_order

    native, _ = load_native()
    if not native.capabilities().get("current_elastic_state"):
        pytest.skip("native reference needs current elastic-state snapshot rebuild")
    mesh = block()
    constitutive = {**material(), "youngs_modulus": 1e6, "fracture_stress_threshold": 1e4}
    velocity = np.zeros_like(mesh.nodes)
    velocity[mesh.nodes[:, 0] == .12, 0] = 4.
    fixed = mesh.nodes[:, 0] == 0
    options = dict(friction_coefficient=0., rayleigh_mass_damping=0., rayleigh_stiffness_damping=0., record_observations=True)
    scene = native.Scene(mesh.nodes, mesh.tets, constitutive, velocity, fixed, .0001, [0., 0., 0.], 1, options)
    observed_count = 0
    try:
        previous = scene.state()
        face_order = native_face_order(mesh, previous)
        for _ in range(50):
            current = scene.step()
            broken = np.asarray(previous["cut_labels"], dtype=bool)[face_order]
            groups = mesh.corner_groups(broken)
            _, representatives = np.unique(groups, return_index=True)
            vertex_order = np.argsort(previous["corner_vertex_ids"].ravel()[representatives], kind="stable")
            tet_order = np.lexsort((previous["mesh_tet_ids"], previous["tet_piece_ids"]))
            report = native_fracture_stress(mesh, constitutive, current["corner_positions"], broken,
                                            endpoint_rotations=current["cached_tet_rotations"],
                                            endpoint_centers_of_mass=current["cached_tet_centers_of_mass"],
                                            current_nodal_masses=previous["corner_masses"],
                                            vertex_order=vertex_order, tetrahedron_order=tet_order)
            observed = np.asarray(current["fracture_stress_observed"], dtype=bool)
            np.testing.assert_array_equal(report["maximum_stress_eigenvalue"][observed], current["fracture_stress_eigenvalues"][observed])
            observed_count += int(observed.sum())
            previous = current
        assert previous["cut_labels"].any()
        assert len(np.unique(previous["tet_piece_ids"])) > 1
        assert observed_count == 50 * len(mesh.tets)
    finally:
        scene.close()
