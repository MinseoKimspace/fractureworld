from dataclasses import dataclass, field

import numpy as np
from numpy.typing import NDArray
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import connected_components


@dataclass(frozen=True)
class Mesh:
    rest: NDArray[np.float64]
    triangles: NDArray[np.int64]
    original_vertex: NDArray[np.int64]
    cut_edges: NDArray[np.int64] = field(
        default_factory=lambda: np.empty((0, 2), dtype=np.int64)
    )

    def __post_init__(self):
        rest = np.asarray(self.rest, dtype=np.float64)
        triangles = np.asarray(self.triangles, dtype=np.int64)
        original_vertex = np.asarray(self.original_vertex, dtype=np.int64)
        cuts = np.asarray(self.cut_edges, dtype=np.int64).reshape(-1, 2)
        if rest.ndim != 2 or rest.shape[1] != 2 or not np.isfinite(rest).all():
            raise ValueError("Rest coordinates must be a finite (nodes, 2) array")
        if triangles.ndim != 2 or triangles.shape[1] != 3 or not len(triangles):
            raise ValueError("Triangles must be a nonempty (elements, 3) array")
        if original_vertex.shape != (len(rest),) or np.any(original_vertex < 0):
            raise ValueError("Each node must have a nonnegative original vertex ID")
        if np.any(triangles < 0) or np.any(triangles >= len(rest)):
            raise ValueError("Triangle indices are outside the node array")
        if len(np.unique(triangles)) != len(rest):
            raise ValueError("Every physical node must belong to an element")
        points = rest[triangles]
        first, second = points[:, 1] - points[:, 0], points[:, 2] - points[:, 0]
        signed_area = first[:, 0] * second[:, 1] - first[:, 1] * second[:, 0]
        if np.any(signed_area <= 0):
            raise ValueError("Rest triangles must have positive orientation")
        for name, value in (
            ("rest", rest),
            ("triangles", triangles),
            ("original_vertex", original_vertex),
            ("cut_edges", np.unique(np.sort(cuts, axis=1), axis=0)),
        ):
            object.__setattr__(self, name, value)


def rectangle(nx: int, ny: int, width: float = 1.0, height: float = 1.0) -> Mesh:
    if nx < 1 or ny < 1 or int(nx) != nx or int(ny) != ny:
        raise ValueError("Mesh cell counts must be positive integers")
    if not np.isfinite([width, height]).all() or width <= 0 or height <= 0:
        raise ValueError("Rectangle dimensions must be positive and finite")
    x, y = np.meshgrid(np.linspace(0, width, nx + 1), np.linspace(0, height, ny + 1))
    rest = np.column_stack((x.ravel(), y.ravel()))
    lower_left = (
        np.arange(ny)[:, None] * (nx + 1) + np.arange(nx)[None, :]
    ).ravel()
    lower_right = lower_left + 1
    upper_left = lower_left + nx + 1
    upper_right = upper_left + 1
    triangles = np.stack(
        (
            np.column_stack((lower_left, lower_right, upper_right)),
            np.column_stack((lower_left, upper_right, upper_left)),
        ),
        axis=1,
    ).reshape(-1, 3)
    return Mesh(rest, triangles, np.arange(len(rest), dtype=np.int64))


def triangle_areas(mesh: Mesh) -> NDArray[np.float64]:
    points = mesh.rest[mesh.triangles]
    first, second = points[:, 1] - points[:, 0], points[:, 2] - points[:, 0]
    return (first[:, 0] * second[:, 1] - first[:, 1] * second[:, 0]) / 2


def lumped_mass(mesh: Mesh, density: float = 1.0, thickness: float = 1.0):
    if not np.isfinite([density, thickness]).all() or density <= 0 or thickness <= 0:
        raise ValueError("Density and thickness must be positive and finite")
    contributions = np.repeat(triangle_areas(mesh) * density * thickness / 3, 3)
    return np.bincount(mesh.triangles.ravel(), contributions, minlength=len(mesh.rest))


def mesh_edges(mesh: Mesh) -> NDArray[np.int64]:
    edges = mesh.triangles[:, [[0, 1], [1, 2], [2, 0]]].reshape(-1, 2)
    return np.unique(np.sort(edges, axis=1), axis=0)


def boundary_edges(mesh: Mesh) -> NDArray[np.int64]:
    edges = mesh.triangles[:, [[0, 1], [1, 2], [2, 0]]].reshape(-1, 2)
    _, inverse, counts = np.unique(
        np.sort(edges, axis=1), axis=0, return_inverse=True, return_counts=True
    )
    if np.any(counts > 2):
        raise ValueError("Nonmanifold physical mesh edges are unsupported")
    return edges[counts[inverse] == 1]


def split_edges(mesh: Mesh, cut_edges) -> Mesh:
    requested = np.asarray(cut_edges, dtype=np.int64)
    if requested.size == 0:
        return mesh
    if requested.ndim != 2 or requested.shape[1] != 2:
        raise ValueError("Cut edges must contain pairs of original vertex IDs")
    requested = np.unique(np.sort(requested, axis=1), axis=0)
    accumulated = np.unique(np.concatenate((mesh.cut_edges, requested)), axis=0)
    if np.array_equal(accumulated, mesh.cut_edges):
        return mesh
    material_corners = mesh.original_vertex[mesh.triangles]
    edge_owners = {}
    for element, corners in enumerate(material_corners):
        for first, second in ((0, 1), (1, 2), (2, 0)):
            edge = tuple(sorted((int(corners[first]), int(corners[second]))))
            edge_owners.setdefault(edge, []).append((element, first, second))
    broken = {tuple(edge) for edge in accumulated.tolist()}
    for edge in broken:
        if edge not in edge_owners or len(edge_owners[edge]) != 2:
            raise ValueError("A cut must follow an interior material mesh edge")
    rows, columns = [], []
    for edge, owners in edge_owners.items():
        if len(owners) > 2:
            raise ValueError("Nonmanifold material mesh edges are unsupported")
        if len(owners) != 2 or edge in broken:
            continue
        left, right = owners
        for vertex in edge:
            first = left[0] * 3 + int(np.flatnonzero(material_corners[left[0]] == vertex)[0])
            second = right[0] * 3 + int(np.flatnonzero(material_corners[right[0]] == vertex)[0])
            rows.extend((first, second))
            columns.extend((second, first))
    graph = coo_matrix(
        (np.ones(len(rows), dtype=np.int8), (rows, columns)),
        shape=(material_corners.size, material_corners.size),
    ).tocsr()
    count, labels = connected_components(graph, directed=False)
    component_to_node = np.full(count, -1, dtype=np.int64)
    parents = list(range(len(mesh.rest)))
    physical_corners = mesh.triangles.ravel()
    for node in range(len(mesh.rest)):
        components = np.unique(labels[physical_corners == node])
        component_to_node[components[0]] = node
        for component in components[1:]:
            component_to_node[component] = len(parents)
            parents.append(node)
    parents = np.asarray(parents, dtype=np.int64)
    return Mesh(
        mesh.rest[parents],
        component_to_node[labels].reshape(-1, 3),
        mesh.original_vertex[parents],
        accumulated,
    )


def lift_state(old: Mesh, new: Mesh, positions, velocities):
    positions = np.asarray(positions)
    velocities = np.asarray(velocities)
    if positions.shape != old.rest.shape or velocities.shape != old.rest.shape:
        raise ValueError("Position and velocity arrays must match the old node array")
    if old.triangles.shape != new.triangles.shape or not np.array_equal(
        old.original_vertex[old.triangles], new.original_vertex[new.triangles]
    ):
        raise ValueError("State transfer requires unchanged material elements and corners")
    old_nodes, new_nodes = old.triangles.ravel(), new.triangles.ravel()
    _, first = np.unique(new_nodes, return_index=True)
    parents = old_nodes[first]
    if not np.array_equal(parents[new_nodes], old_nodes):
        raise ValueError("State transfer cannot join previously separated physical nodes")
    return positions[parents].copy(), velocities[parents].copy()
