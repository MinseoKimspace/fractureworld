import numpy as np
import pytest
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import connected_components

from fracture_world.topology import (
    boundary_edges,
    lift_state,
    lumped_mass,
    mesh_edges,
    rectangle,
    split_edges,
    triangle_areas,
)


def seam(mesh, start=0.0, stop=1.0):
    edges = mesh_edges(mesh)
    points = mesh.rest[edges]
    selected = (
        np.isclose(points[:, :, 0], 0.5).all(axis=1)
        & (points[:, :, 1].min(axis=1) >= start)
        & (points[:, :, 1].max(axis=1) <= stop)
    )
    return mesh.original_vertex[edges[selected]]


def components(mesh):
    edges = mesh_edges(mesh)
    graph = coo_matrix(
        (np.ones(len(edges)), (edges[:, 0], edges[:, 1])),
        shape=(len(mesh.rest), len(mesh.rest)),
    )
    return connected_components(graph, directed=False)[0]


def test_partial_cut_preserves_crack_tip_and_bulk_connectivity():
    original = rectangle(4, 4)
    cut = split_edges(original, seam(original, stop=0.5))
    assert components(cut) == 1
    crack_tip = np.flatnonzero(np.all(np.isclose(cut.rest, [0.5, 0.5]), axis=1))
    assert len(crack_tip) == 1
    opened_vertex = np.flatnonzero(np.all(np.isclose(cut.rest, [0.5, 0.25]), axis=1))
    assert len(opened_vertex) == 2
    np.testing.assert_allclose(triangle_areas(cut), triangle_areas(original))


def test_complete_cut_forms_independently_movable_components():
    original = rectangle(4, 4)
    cut = split_edges(original, seam(original))
    assert components(cut) == 2
    positions = cut.rest.copy()
    right_elements = cut.rest[cut.triangles].mean(axis=1)[:, 0] > 0.5
    right_nodes = np.unique(cut.triangles[right_elements])
    left_nodes = np.unique(cut.triangles[~right_elements])
    assert not np.intersect1d(left_nodes, right_nodes).size
    positions[right_nodes, 0] += 0.1
    np.testing.assert_array_equal(positions[left_nodes], cut.rest[left_nodes])
    assert len(boundary_edges(cut)) == len(boundary_edges(original)) + 8


def test_split_preserves_mass_momentum_and_current_deformation():
    original = rectangle(4, 4, width=1, height=2)
    cuts = seam(original, stop=1.0)
    cut = split_edges(original, cuts)
    positions = original.rest @ np.array([[1.1, 0.2], [-0.1, 0.95]]) + [0.3, 0.7]
    velocities = np.column_stack((original.rest[:, 1], -original.rest[:, 0]))
    moved_positions, moved_velocities = lift_state(original, cut, positions, velocities)
    before = lumped_mass(original, density=7, thickness=0.3)
    after = lumped_mass(cut, density=7, thickness=0.3)
    np.testing.assert_allclose(after.sum(), before.sum(), rtol=1e-14)
    np.testing.assert_allclose(after @ moved_velocities, before @ velocities, rtol=1e-14)
    np.testing.assert_allclose(after @ moved_positions, before @ positions, rtol=1e-14)
    np.testing.assert_allclose(
        moved_positions[cut.triangles], positions[original.triangles]
    )
    np.testing.assert_allclose(
        np.sum(after[:, None] * moved_velocities**2),
        np.sum(before[:, None] * velocities**2),
    )


def test_successive_cut_accumulates_and_does_not_rejoin():
    original = rectangle(4, 4)
    partial = split_edges(original, seam(original, stop=0.5))
    complete = split_edges(partial, seam(original, start=0.5))
    assert components(complete) == 2
    assert split_edges(complete, seam(original)) is complete
    assert split_edges(complete, []) is complete
    np.testing.assert_array_equal(complete.cut_edges, seam(original))
    positions, velocities = lift_state(partial, complete, partial.rest, partial.rest * 0)
    np.testing.assert_array_equal(positions, complete.rest)
    np.testing.assert_array_equal(velocities, np.zeros_like(complete.rest))


def test_boundary_orientation_and_intact_shared_dofs():
    mesh = rectangle(2, 2)
    boundary = boundary_edges(mesh)
    midpoint = mesh.rest[boundary].mean(axis=1)
    direction = mesh.rest[boundary[:, 1]] - mesh.rest[boundary[:, 0]]
    outward = np.column_stack((direction[:, 1], -direction[:, 0]))
    assert np.all(np.einsum("ij,ij->i", outward, midpoint - [0.5, 0.5]) > 0)
    assert len(mesh.rest) == 9
    assert len(np.unique(mesh.triangles)) == 9


def test_invalid_boundary_cut_and_join_are_rejected():
    mesh = rectangle(4, 4)
    with pytest.raises(ValueError, match="interior"):
        split_edges(mesh, [[0, 1]])
    cut = split_edges(mesh, seam(mesh))
    with pytest.raises(ValueError, match="join"):
        lift_state(cut, mesh, cut.rest, np.zeros_like(cut.rest))


@pytest.mark.parametrize("density, thickness", [(0, 1), (1, -1), (np.nan, 1)])
def test_invalid_material_mass_is_rejected(density, thickness):
    with pytest.raises(ValueError):
        lumped_mass(rectangle(2, 2), density, thickness)
