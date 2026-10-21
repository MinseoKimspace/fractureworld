import numpy as np
import pytest
from scipy.optimize import linprog
from scipy.spatial import ConvexHull

from fracture_world.dynamics_geometry import (
    boundary_triangles,
    corner_groups,
    face_corner_pairs,
    geometry_diagnostics,
    linearized_contacts,
    render_geometry,
    swept_geometry_diagnostics,
    tetrahedron_separation,
)
from fracture_world.mesh import VolumeMesh


def paired_tetrahedra():
    return VolumeMesh(np.array([[0., 0., 0.], [1., 0., 0.], [0., 1., 0.], [0., 0., 1.], [1., 1., 1.]]),
                      np.array([[0, 1, 2, 3], [4, 1, 2, 3]]))


def tetrahedron_ring():
    angles = np.arange(3) * 2 * np.pi / 3
    nodes = np.vstack(([[0., 0., 0.], [0., 0., 1.]], np.column_stack((np.cos(angles), np.sin(angles), np.zeros(3)))))
    return VolumeMesh(nodes, np.array([[0, 1, 2, 3], [0, 1, 3, 4], [0, 1, 4, 2]]))


def test_boundary_surface_is_outward_oriented_before_and_after_crack():
    mesh = paired_tetrahedra()
    corners = mesh.nodes[mesh.tets].reshape(-1, 3)
    for broken, expected_faces in ((np.array([False]), 6), (np.array([True]), 8)):
        triangles = boundary_triangles(mesh, broken)
        assert len(triangles) == expected_faces
        coordinates = corners[triangles]
        normals = np.cross(coordinates[:, 1] - coordinates[:, 0], coordinates[:, 2] - coordinates[:, 0])
        owner_centers = mesh.centers[triangles[:, 0] // 4]
        assert np.all(np.einsum("ij,ij->i", normals, owner_centers - coordinates.mean(axis=1)) < 0)


def test_render_metadata_matches_actual_components_crack_sides_and_supports():
    mesh = paired_tetrahedra()
    positions = mesh.nodes[mesh.tets].reshape(-1, 3)
    fixed = np.zeros(8, dtype=bool)
    fixed[0] = True
    for broken in (np.array([False]), np.array([True])):
        geometry = render_geometry(mesh, positions, broken, fixed)
        assert len(geometry["crack_faces"]) == len(geometry["faces"])
        assert sum(geometry["crack_faces"]) == 2 * int(broken.sum())
        assert len(set(geometry["vertex_components"])) == 1 + int(broken.sum())
        assert geometry["fixed_vertices"] == [0]
        np.testing.assert_array_equal(geometry["vertices"], positions)


def test_intact_corner_groups_separate_only_when_material_face_breaks():
    mesh = paired_tetrahedra()
    pairs = face_corner_pairs(mesh)[0]
    corners = mesh.nodes[mesh.tets].reshape(-1, 3)
    np.testing.assert_array_equal(corners[pairs[:, 0]], corners[pairs[:, 1]])
    intact = corner_groups(mesh, np.array([False]))
    broken = corner_groups(mesh, np.array([True]))
    assert np.all(intact[pairs[:, 0]] == intact[pairs[:, 1]])
    assert np.all(broken[pairs[:, 0]] != broken[pairs[:, 1]])


def test_tetrahedron_sat_rejects_aabb_false_positive_and_measures_real_overlap():
    tetrahedron = np.array([[0., 0., 0.], [1., 0., 0.], [0., 1., 0.], [0., 0., 1.]])
    separated = tetrahedron + 0.6
    gap, normal = tetrahedron_separation(tetrahedron, separated)
    assert gap > 0
    assert np.linalg.norm(normal) == pytest.approx(1)
    reverse_gap, reverse_normal = tetrahedron_separation(separated, tetrahedron)
    assert gap == pytest.approx(reverse_gap)
    np.testing.assert_allclose(normal, -reverse_normal)
    overlapping = tetrahedron + 0.1
    assert tetrahedron_separation(tetrahedron, overlapping)[0] < 0


def test_sat_agrees_with_independent_convex_halfspace_feasibility():
    rng = np.random.default_rng(48)
    base = np.array([[0., 0., 0.], [1., 0., 0.], [0., 1., 0.], [0., 0., 1.]])
    for _ in range(30):
        rotation, _ = np.linalg.qr(rng.normal(size=(3, 3)))
        second = base @ rotation + rng.uniform(-1, 1, size=3)
        equations = np.vstack((ConvexHull(base).equations, ConvexHull(second).equations))
        result = linprog([0., 0., 0., -1.], A_ub=np.column_stack((equations[:, :3], np.ones(8))),
                         b_ub=-equations[:, 3], bounds=[(None, None)] * 4, method="highs")
        assert result.success
        gap, _ = tetrahedron_separation(base, second)
        if abs(result.x[3]) > 1.e-8:
            assert (gap < 0) == (result.x[3] > 0)


def test_negative_old_face_plane_gap_is_allowed_when_fragments_are_separated():
    mesh = paired_tetrahedra()
    broken = np.array([True])
    rest = mesh.nodes[mesh.tets].reshape(-1, 3)
    normal = mesh.faces.normals[mesh.internal_faces[0]]
    tangent = np.cross(normal, [1., 0., 0.])
    tangent /= np.linalg.norm(tangent)
    moved = rest.copy()
    moved[4:] += 5 * tangent - normal
    diagnostics = geometry_diagnostics(mesh, moved, broken, 1.e-8)
    assert diagnostics["valid"]
    assert diagnostics["minimum_opened_face_normal_gap_metres"] < 0
    assert diagnostics["penetrating_opened_material_faces"] == 0
    assert not linearized_contacts(mesh, moved, moved, broken, 1.e-8)


def test_opened_face_overlap_is_detected_even_when_intact_corner_incidence_excludes_pair():
    mesh = tetrahedron_ring()
    face = int(np.flatnonzero(np.all(mesh.dual_edges == [0, 1], axis=1))[0])
    broken = np.zeros(len(mesh.internal_faces), dtype=bool)
    broken[face] = True
    rest = mesh.nodes[mesh.tets].reshape(-1, 3)
    moved = rest.copy()
    pairs = face_corner_pairs(mesh)[face]
    independent = pairs[np.linalg.norm(rest[pairs[:, 0], :2], axis=1) > 0.5][0]
    normal = mesh.faces.normals[mesh.internal_faces[face]]
    moved[independent[1]] -= 0.2 * normal
    diagnostics = geometry_diagnostics(mesh, moved, broken, 1.e-8)
    assert diagnostics["intact_face_position_mismatch_metres"] == 0
    assert diagnostics["overlapping_nonincident_tetrahedron_pairs"] == 0
    assert diagnostics["penetrating_opened_material_faces"] == 1
    assert diagnostics["maximum_opened_face_normal_penetration_metres"] > 0
    assert not diagnostics["valid"]
    contacts = linearized_contacts(mesh, rest, moved, broken, 1.e-8)
    assert contacts
    assert any(np.array_equal(contact.corner_ids, independent) for contact in contacts)


def test_geometry_diagnostics_rejects_inversion_and_intact_face_mismatch():
    mesh = paired_tetrahedra()
    broken = np.array([False])
    rest = mesh.nodes[mesh.tets].reshape(-1, 3)
    assert geometry_diagnostics(mesh, rest, broken, 1.e-8)["valid"]
    displaced = rest.copy()
    displaced[face_corner_pairs(mesh)[0, 0, 1]] += 0.01
    diagnostics = geometry_diagnostics(mesh, displaced, broken, 1.e-8)
    assert not diagnostics["valid"]
    assert diagnostics["intact_face_position_mismatch_metres"] > 0
    inverted = rest.copy()
    inverted[[0, 1]] = inverted[[1, 0]]
    assert geometry_diagnostics(mesh, inverted, broken, 1.e-8)["inverted_tetrahedra"] == 1


def test_collapsed_tetrahedra_return_invalid_geometry_instead_of_crashing_contact_detection():
    mesh = paired_tetrahedra()
    diagnostics = geometry_diagnostics(mesh, np.zeros((8, 3)), np.array([True]), 1.e-8)
    assert not diagnostics["valid"]
    assert diagnostics["inverted_tetrahedra"] == 2
    with pytest.raises(ValueError, match="match"):
        geometry_diagnostics(mesh, np.zeros((4, 3)), np.array([True]), 1.e-8)


def separate_tetrahedra():
    first = np.array([[0., 0., 0.], [1., 0., 0.], [0., 1., 0.], [0., 0., 1.]])
    return VolumeMesh(np.vstack((first, first + [3., 0., 0.])), np.arange(8).reshape(2, 4))


def test_rotated_trial_contact_uses_trial_axis_and_every_active_support_pair():
    mesh = separate_tetrahedra()
    current = mesh.nodes[mesh.tets]
    proposed = np.stack((current[0, [1, 2, 0, 3]], current[0, [1, 2, 0, 3]] + 0.1))
    displacement = (proposed - current).reshape(-1, 3)
    old_normal = tetrahedron_separation(*current)[1]
    old_first = np.argmax(current[0] @ old_normal)
    old_second = np.argmin(current[1] @ old_normal)
    assert np.dot(proposed[1, old_second] - proposed[0, old_first], old_normal) > 0
    separation, normal = tetrahedron_separation(*proposed)
    assert separation < -0.1
    contacts = linearized_contacts(mesh, current, proposed, np.empty(0, dtype=bool), 1.e-8)
    expected = {(first, 4 + second) for first in range(4) for second in range(4)
                if np.dot(proposed[1, second] - proposed[0, first], normal) <= 1.e-8}
    assert {tuple(contact.corner_ids) for contact in contacts} == expected
    evaluated = []
    for contact in contacts:
        np.testing.assert_allclose(contact.normal, normal)
        movement = np.einsum("i,ij->j", contact.weights, displacement[contact.corner_ids])
        evaluated.append(contact.gap + np.dot(movement, contact.normal))
    assert min(evaluated) == pytest.approx(separation)
    first_constraint = contacts[int(np.argmin(evaluated))]
    assert evaluated[int(np.argmin(evaluated))] < 0
    assert np.dot(np.einsum("i,ij->j", first_constraint.weights,
                            current.reshape(-1, 3)[first_constraint.corner_ids]), normal) == pytest.approx(first_constraint.gap)


def test_sweep_certifies_shared_translation_and_floor_touch_with_tolerance():
    mesh = separate_tetrahedra()
    current = mesh.nodes[mesh.tets]
    proposed = current + [0.2, 0.1, -0.4]
    result = swept_geometry_diagnostics(mesh, current, proposed, np.empty(0, dtype=bool), 1.e-8, floor_height=0.0)
    assert result["valid"]
    assert result["continuous_collision_unresolved_pair_count"] == 0
    assert result["maximum_swept_floor_penetration_metres"] == 0
    proposed[:, :, 1] -= 0.1 + 0.5e-8
    assert swept_geometry_diagnostics(mesh, current, proposed, np.empty(0, dtype=bool), 1.e-8, 0.0)["valid"]
    proposed[:, :, 1] -= 2.e-8
    assert not swept_geometry_diagnostics(mesh, current, proposed, np.empty(0, dtype=bool), 1.e-8, 0.0)["valid"]


def test_sweep_certifies_opening_crack_from_initial_touch():
    mesh = paired_tetrahedra()
    broken = np.array([True])
    current = mesh.nodes[mesh.tets]
    normal = tetrahedron_separation(*current)[1]
    proposed = current.copy()
    proposed[1] += 0.1 * normal
    result = swept_geometry_diagnostics(mesh, current, proposed, broken, 1.e-8)
    assert result["valid"]
    assert result["certified_nonpenetrating_pairs"] == 1


def test_sweep_rejects_pass_through_even_when_both_endpoint_geometries_are_valid():
    mesh = separate_tetrahedra()
    current = mesh.nodes[mesh.tets]
    proposed = current[::-1].copy()
    broken = np.empty(0, dtype=bool)
    assert geometry_diagnostics(mesh, current, broken, 1.e-8)["valid"]
    assert geometry_diagnostics(mesh, proposed, broken, 1.e-8)["valid"]
    result = swept_geometry_diagnostics(mesh, current, proposed, broken, 1.e-8)
    assert not result["valid"]
    assert result["continuous_collision_unresolved_pairs"] == [[0, 1]]


def test_sweep_rejects_interior_inversion_with_positive_endpoint_volumes():
    mesh = VolumeMesh([[0., 0., 0.], [1., 0., 0.], [0., 1., 0.], [0., 0., 1.]], [[0, 1, 2, 3]])
    current = mesh.nodes[mesh.tets]
    proposed = current @ np.diag([-1.0, -2.0, 1.0])
    assert np.linalg.det((proposed[:, 1:] - proposed[:, :1]))[0] > 0
    result = swept_geometry_diagnostics(mesh, current, proposed, np.empty(0, dtype=bool), 1.e-8)
    assert not result["valid"]
    assert result["degenerate_or_inverted_swept_tetrahedra"] == [0]
    assert result["minimum_normalized_swept_determinant"] < 0
