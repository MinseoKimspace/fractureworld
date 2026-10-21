import numpy as np
import pytest
from numpy.testing import assert_allclose, assert_array_equal

from fracture_world.mesh import VolumeMesh


@pytest.fixture
def two_tets():
    nodes = np.array([[0, 0, 0], [1, 0, 0], [0, 1, 0], [0, 0, 1], [0, 0, -1]], dtype=float)
    return VolumeMesh(nodes, [[0, 1, 2, 3], [0, 2, 1, 4]])


def test_invalid_nodes_and_tetrahedra(two_tets):
    with pytest.raises(ValueError, match="finite"):
        VolumeMesh(np.full((4, 3), np.nan), [[0, 1, 2, 3]])
    with pytest.raises(ValueError, match="outside"):
        VolumeMesh(two_tets.nodes, [[0, 1, 2, 99]])
    with pytest.raises(ValueError, match="integers"):
        VolumeMesh(two_tets.nodes, [[0.0, 1.0, 2.0, 3.0]])
    with pytest.raises(ValueError, match="nondegenerate"):
        VolumeMesh(two_tets.nodes, [[0, 1, 2, 2]])
    with pytest.raises(ValueError, match="shape"):
        VolumeMesh(two_tets.nodes, [])
    nodes = np.concatenate((two_tets.nodes, [[0, 0, 2]]))
    with pytest.raises(ValueError, match="more than two"):
        VolumeMesh(nodes, [[0, 1, 2, 3], [0, 2, 1, 4], [0, 1, 2, 5]])
    with pytest.raises(ValueError, match="opposite sides"):
        VolumeMesh(nodes, [[0, 1, 2, 3], [0, 1, 2, 5]])


def test_npz_rejects_pickled_arrays(tmp_path):
    path = tmp_path / "unsafe.npz"
    np.savez(path, nodes=np.array([object()], dtype=object), tets=np.array([[0, 1, 2, 3]]))
    with pytest.raises(ValueError, match="allow_pickle=False"):
        VolumeMesh.load(path)


def test_shared_face_preserves_adjacency_area_and_outward_orientation(two_tets):
    assert len(two_tets.faces.vertices) == 7
    assert_array_equal(two_tets.dual_edges, [[0, 1]])
    shared = two_tets.internal_faces[0]
    assert_allclose(two_tets.faces.areas[shared], 0.5)
    assert_allclose(two_tets.faces.normals[shared], [0, 0, -1])
    face_centers = two_tets.nodes[two_tets.faces.vertices].mean(axis=1)
    inward = two_tets.centers[two_tets.faces.cells[:, 0]] - face_centers
    assert np.all(np.einsum("ij,ij->i", two_tets.faces.normals, inward) < 0)


def test_reversed_tetrahedra_recover_positive_orientation(two_tets):
    reversed_mesh = VolumeMesh(two_tets.nodes, two_tets.tets[:, [1, 0, 2, 3]])
    assert_allclose(reversed_mesh.volumes, two_tets.volumes)
    assert_array_equal(reversed_mesh.tets, two_tets.tets)


def test_npz_round_trip_preserves_material_topology(two_tets, tmp_path):
    path = tmp_path / "mesh.npz"
    two_tets.save(path)
    loaded = VolumeMesh.load(path)
    assert_array_equal(loaded.nodes, two_tets.nodes)
    assert_array_equal(loaded.tets, two_tets.tets)
    assert_array_equal(loaded.faces.cells, two_tets.faces.cells)
    assert_allclose(loaded.volumes, two_tets.volumes)
