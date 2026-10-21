from itertools import combinations

import numpy as np
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import connected_components

from .dynamics_corrector import LinearizedContact
from .partition import component_labels


LOCAL_FACES = np.array([[1, 2, 3], [0, 3, 2], [0, 1, 3], [0, 2, 1]])
LOCAL_EDGES = np.array(list(combinations(range(4), 2)))


def face_corner_pairs(mesh):
    cells = mesh.faces.cells[mesh.internal_faces]
    vertices = mesh.faces.vertices[mesh.internal_faces]
    sides = []
    for side in range(2):
        local = np.argmax(mesh.tets[cells[:, side], :, None] == vertices[:, None, :], axis=1)
        sides.append(4 * cells[:, side, None] + local)
    return np.stack(sides, axis=-1)


def corner_groups(mesh, broken):
    broken = np.asarray(broken, dtype=bool)
    if broken.shape != (len(mesh.internal_faces),):
        raise ValueError("broken must contain one flag per internal material face")
    pairs = face_corner_pairs(mesh)[~broken].reshape(-1, 2)
    count = 4 * len(mesh.tets)
    graph = coo_matrix((np.ones(len(pairs)), (pairs[:, 0], pairs[:, 1])), shape=(count, count))
    return connected_components(graph, directed=False)[1]


def boundary_triangles(mesh, broken):
    broken = np.asarray(broken, dtype=bool)
    if broken.shape != (len(mesh.internal_faces),):
        raise ValueError("broken must contain one flag per internal material face")
    opened = np.zeros(len(mesh.faces.vertices), dtype=bool)
    opened[mesh.internal_faces] = broken
    triangles = []
    for face_id, (first, second) in enumerate(mesh.faces.cells):
        if second >= 0 and not opened[face_id]:
            continue
        vertices = mesh.faces.vertices[face_id]
        for cell, order in ((first, [0, 1, 2]), (second, [0, 2, 1])):
            if cell >= 0:
                local = np.argmax(mesh.tets[cell, :, None] == vertices[None, order], axis=0)
                triangles.append(4 * cell + local)
    return np.asarray(triangles, dtype=np.int64).reshape(-1, 3)


def render_geometry(mesh, positions, broken, fixed):
    positions = np.asarray(positions).reshape(-1, 3)
    fixed = np.asarray(fixed)
    if positions.shape != (4 * len(mesh.tets), 3) or not np.isfinite(positions).all():
        raise ValueError("render positions require finite material corners")
    if fixed.shape != (len(positions),) or fixed.dtype != np.dtype(bool):
        raise ValueError("render supports require one Boolean per material corner")
    faces = boundary_triangles(mesh, broken)
    original_ids = mesh.tets.ravel()
    opened = {tuple(sorted(vertices)) for vertices in mesh.faces.vertices[mesh.internal_faces[np.asarray(broken, dtype=bool)]]}
    cracks = [tuple(sorted(original_ids[triangle])) in opened for triangle in faces]
    labels = component_labels(len(mesh.tets), mesh.dual_edges, broken)
    return dict(vertices=positions.tolist(), faces=faces.tolist(), crack_faces=cracks,
                vertex_components=np.repeat(labels, 4).tolist(), fixed_vertices=np.flatnonzero(fixed).tolist())


def tetrahedron_separation(first, second):
    first, second = np.asarray(first, dtype=float), np.asarray(second, dtype=float)
    if first.shape != (4, 3) or second.shape != (4, 3) or not np.isfinite(first).all() or not np.isfinite(second).all():
        raise ValueError("contact query requires two finite tetrahedra")
    triangles = np.concatenate((first[LOCAL_FACES], second[LOCAL_FACES]))
    normals = np.cross(triangles[:, 1] - triangles[:, 0], triangles[:, 2] - triangles[:, 0])
    first_edges = first[LOCAL_EDGES[:, 1]] - first[LOCAL_EDGES[:, 0]]
    second_edges = second[LOCAL_EDGES[:, 1]] - second[LOCAL_EDGES[:, 0]]
    axes = np.concatenate((normals, np.cross(first_edges[:, None], second_edges[None]).reshape(-1, 3)))
    lengths = np.linalg.norm(axes, axis=1)
    scale = max(np.linalg.norm(np.ptp(first, axis=0)), np.linalg.norm(np.ptp(second, axis=0)))
    valid = lengths > 64 * np.finfo(float).eps * scale**2
    if not valid.any():
        raise ValueError("contact query contains degenerate tetrahedra")
    axes = axes[valid] / lengths[valid, None]
    first_projection, second_projection = first @ axes.T, second @ axes.T
    forward = second_projection.min(axis=0) - first_projection.max(axis=0)
    backward = first_projection.min(axis=0) - second_projection.max(axis=0)
    flipped = backward > forward
    gaps = np.maximum(forward, backward)
    index = int(np.argmax(gaps))
    normal = axes[index] * (-1 if flipped[index] else 1)
    return float(gaps[index]), normal


def _candidate_pairs(mesh, current, proposed, broken, tolerance):
    lower = np.minimum(current.min(axis=1), proposed.min(axis=1)) - tolerance
    upper = np.maximum(current.max(axis=1), proposed.max(axis=1)) + tolerance
    groups = corner_groups(mesh, broken).reshape(-1, 4)
    for first in range(len(current)):
        candidates = np.flatnonzero(np.all(upper[first] >= lower, axis=1) & np.all(upper >= lower[first], axis=1))
        for second in candidates[candidates > first]:
            if not np.intersect1d(groups[first], groups[second]).size:
                yield first, int(second)


def linearized_contacts(mesh, current, proposed, broken, tolerance):
    current = np.asarray(current, dtype=float).reshape(-1, 4, 3)
    proposed = np.asarray(proposed, dtype=float).reshape(current.shape)
    broken = np.asarray(broken, dtype=bool)
    if len(current) != len(mesh.tets) or broken.shape != (len(mesh.internal_faces),):
        raise ValueError("contact states and broken flags must match the material mesh")
    if not np.isfinite(current).all() or not np.isfinite(proposed).all() or not np.isfinite(tolerance) or tolerance <= 0:
        raise ValueError("contact states must be finite and tolerance positive")
    contacts = []
    pairs = face_corner_pairs(mesh)
    flat_current, flat_proposed = current.reshape(-1, 3), proposed.reshape(-1, 3)
    for face in np.flatnonzero(broken):
        first_cell, second_cell = mesh.dual_edges[face]
        current_separation, _ = tetrahedron_separation(current[first_cell], current[second_cell])
        proposed_separation, _ = tetrahedron_separation(proposed[first_cell], proposed[second_cell])
        if min(current_separation, proposed_separation) >= -tolerance:
            continue
        ids = pairs[face, :, 0]
        triangle = flat_current[ids]
        normal = np.cross(triangle[1] - triangle[0], triangle[2] - triangle[0])
        length = np.linalg.norm(normal)
        if length <= np.finfo(float).tiny:
            continue
        normal /= length
        if np.dot(normal, current[first_cell].mean(axis=0) - triangle.mean(axis=0)) > 0:
            normal = -normal
        for first, second in pairs[face]:
            gap = float(np.dot(flat_current[second] - flat_current[first], normal))
            next_gap = float(np.dot(flat_proposed[second] - flat_proposed[first], normal))
            if min(gap, next_gap) < tolerance:
                contacts.append(LinearizedContact(np.array([first, second]), np.array([-1.0, 1.0]), normal, gap))
    for first, second in _candidate_pairs(mesh, current, proposed, broken, tolerance):
        next_gap, _ = tetrahedron_separation(proposed[first], proposed[second])
        if next_gap > tolerance:
            continue
        _, normal = tetrahedron_separation(current[first], current[second])
        first_corner = int(np.argmax(current[first] @ normal))
        second_corner = int(np.argmin(current[second] @ normal))
        gap = float(np.dot(current[second, second_corner] - current[first, first_corner], normal))
        contacts.append(LinearizedContact(np.array([4 * first + first_corner, 4 * second + second_corner]),
                                          np.array([-1.0, 1.0]), normal, gap))
    return tuple(contacts)


def geometry_diagnostics(mesh, positions, broken, tolerance):
    positions = np.asarray(positions, dtype=float).reshape(-1, 4, 3)
    broken = np.asarray(broken, dtype=bool)
    if len(positions) != len(mesh.tets) or broken.shape != (len(mesh.internal_faces),):
        raise ValueError("geometry state and broken flags must match the material mesh")
    if not np.isfinite(tolerance) or tolerance <= 0:
        raise ValueError("geometry tolerance must be finite and positive")
    if not np.isfinite(positions).all():
        return {"valid": False, "nonfinite": True}
    determinant = np.linalg.det(positions[:, 1:] - positions[:, :1])
    overlap_count, maximum_depth = 0, 0.0
    for first, second in _candidate_pairs(mesh, positions, positions, broken, tolerance):
        if determinant[first] <= 0 or determinant[second] <= 0:
            continue
        gap, _ = tetrahedron_separation(positions[first], positions[second])
        if gap < -tolerance:
            overlap_count += 1
            maximum_depth = max(maximum_depth, -gap)
    pairs = face_corner_pairs(mesh)[~np.asarray(broken, dtype=bool)].reshape(-1, 2)
    flat = positions.reshape(-1, 3)
    continuity = float(np.max(np.linalg.norm(flat[pairs[:, 0]] - flat[pairs[:, 1]], axis=1), initial=0))
    opened_count, opened_depth, minimum_opened_gap = 0, 0.0, float("inf")
    all_pairs = face_corner_pairs(mesh)
    for face in np.flatnonzero(broken):
        first_cell, second_cell = mesh.dual_edges[face]
        if determinant[first_cell] <= 0 or determinant[second_cell] <= 0:
            continue
        ids = all_pairs[face, :, 0]
        triangle = flat[ids]
        normal = np.cross(triangle[1] - triangle[0], triangle[2] - triangle[0])
        length = np.linalg.norm(normal)
        if length <= np.finfo(float).tiny:
            continue
        normal /= length
        if np.dot(normal, positions[first_cell].mean(axis=0) - triangle.mean(axis=0)) > 0:
            normal = -normal
        gaps = (flat[all_pairs[face, :, 1]] - flat[all_pairs[face, :, 0]]) @ normal
        minimum_opened_gap = min(minimum_opened_gap, float(gaps.min()))
        separation, _ = tetrahedron_separation(positions[first_cell], positions[second_cell])
        if separation < -tolerance:
            opened_count += 1
            opened_depth = max(opened_depth, float(max(0, -gaps.min())))
    inverted = int(np.count_nonzero(determinant <= 0))
    return {"valid": inverted == 0 and overlap_count == 0 and opened_count == 0 and continuity <= tolerance,
            "inverted_tetrahedra": inverted, "overlapping_nonincident_tetrahedron_pairs": overlap_count,
            "maximum_separating_axis_penetration_metres": maximum_depth,
            "intact_face_position_mismatch_metres": continuity,
            "penetrating_opened_material_faces": opened_count,
            "maximum_opened_face_normal_penetration_metres": opened_depth,
            "minimum_opened_face_normal_gap_metres": minimum_opened_gap if np.isfinite(minimum_opened_gap) else 0.0,
            "contact_detection": "discrete tetrahedron SAT and opened material-face normal gaps",
            "continuous_collision_detection": False,
            "incident_tetrahedron_collision_exclusion": "pairs sharing an intact material corner"}
