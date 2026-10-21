from dataclasses import dataclass, field
from functools import cached_property
from pathlib import Path

import numpy as np
from scipy.spatial.transform import Rotation

from .partition import component_labels


FEMFX_MAX_TETRAHEDRON_ASPECT_RATIO = 25.0


def tetrahedron_aspect_ratios(nodes, tets):
    vertices = np.asarray(nodes, dtype=float)[np.asarray(tets)]
    vertices = vertices - vertices[:, :1]
    scale = np.max(np.abs(vertices), axis=(1, 2))
    with np.errstate(divide="ignore", invalid="ignore"):
        vertices = vertices / scale[:, None, None]
    edges = np.array([[0, 1], [0, 2], [0, 3], [1, 2], [1, 3], [2, 3]])
    lengths = np.linalg.norm(vertices[:, edges[:, 1]] - vertices[:, edges[:, 0]], axis=-1)
    face_ids = np.array([[1, 2, 3], [0, 3, 2], [0, 1, 3], [0, 2, 1]])
    faces = vertices[:, face_ids]
    double_areas = np.linalg.norm(np.cross(faces[:, :, 1] - faces[:, :, 0],
                                          faces[:, :, 2] - faces[:, :, 0]), axis=-1)
    six_volumes = np.abs(np.linalg.det(vertices[:, 1:] - vertices[:, :1]))
    with np.errstate(divide="ignore", invalid="ignore"):
        return lengths.max(axis=1) * double_areas.max(axis=1) / six_volumes


def rigid_pose(nodes, pose=None):
    nodes = np.asarray(nodes, dtype=float)
    if pose is None:
        return nodes.copy()
    if not isinstance(pose, dict) or set(pose) != {"rotation_xyzw", "translation"}:
        raise ValueError("initial_pose requires rotation_xyzw and translation")
    quaternion = np.asarray(pose["rotation_xyzw"], dtype=float)
    translation = np.asarray(pose["translation"], dtype=float)
    if quaternion.shape != (4,) or translation.shape != (3,) or not np.isfinite(np.r_[quaternion, translation]).all():
        raise ValueError("initial pose must contain finite quaternion and translation vectors")
    if not np.isclose(np.linalg.norm(quaternion), 1.0, rtol=1e-10, atol=1e-12):
        raise ValueError("initial rotation quaternion must have unit length")
    return Rotation.from_quat(quaternion).apply(nodes) + translation


@dataclass(frozen=True)
class FaceTable:
    vertices: np.ndarray
    cells: np.ndarray
    normals: np.ndarray
    areas: np.ndarray


@dataclass
class VolumeMesh:
    nodes: np.ndarray
    tets: np.ndarray
    volumes: np.ndarray = field(init=False)
    centers: np.ndarray = field(init=False)
    faces: FaceTable = field(init=False)

    def __post_init__(self):
        self.nodes = np.array(self.nodes, dtype=np.float64, copy=True)
        raw_tets = np.asarray(self.tets)
        if self.nodes.ndim != 2 or self.nodes.shape[1] != 3 or len(self.nodes) < 4:
            raise ValueError("nodes must have shape (N, 3), with N >= 4")
        if not np.isfinite(self.nodes).all():
            raise ValueError("nodes must be finite")
        if raw_tets.ndim != 2 or raw_tets.shape[1] != 4 or not len(raw_tets):
            raise ValueError("tets must have shape (M, 4), with M >= 1")
        if not np.issubdtype(raw_tets.dtype, np.integer):
            raise ValueError("tetrahedron indices must be integers")
        if (raw_tets < 0).any() or (raw_tets >= len(self.nodes)).any():
            raise ValueError("tetrahedron indices are outside the node array")
        self.tets = np.array(raw_tets, dtype=np.int64, copy=True)
        vertices = self.nodes[self.tets]
        offsets = vertices[:, 1:] - vertices[:, :1]
        scale = np.max(np.abs(offsets), axis=(1, 2))
        if (scale == 0).any() or not np.isfinite(offsets).all():
            raise ValueError("tetrahedra must have finite, nonzero extent")
        determinants = np.linalg.det(offsets / scale[:, None, None])
        if (np.abs(determinants) <= 64 * np.finfo(np.float64).eps).any():
            raise ValueError("tetrahedra must be nondegenerate")
        reversed_cells = determinants < 0
        self.tets[reversed_cells, :2] = self.tets[reversed_cells, 1::-1]
        self.volumes = np.abs(determinants) * scale**3 / 6
        if not np.isfinite(self.volumes).all() or (self.volumes <= 0).any():
            raise ValueError("tetrahedron volumes must be finite and positive")
        self.centers = self.nodes[self.tets].mean(axis=1)
        self.faces = self._build_faces()

    def _build_faces(self):
        local_faces = np.array([[1, 2, 3], [0, 3, 2], [0, 1, 3], [0, 2, 1]])
        oriented_faces = self.tets[:, local_faces].reshape(-1, 3)
        owners = np.repeat(np.arange(len(self.tets)), 4)
        _, first, inverse, counts = np.unique(
            np.sort(oriented_faces, axis=1),
            axis=0,
            return_index=True,
            return_inverse=True,
            return_counts=True,
        )
        if (counts > 2).any():
            raise ValueError("a triangle face has more than two incident tetrahedra")
        cells = np.column_stack((owners[first], np.full(len(first), -1)))
        internal = counts == 2
        ordered = np.argsort(inverse, kind="stable")
        starts = np.cumsum(counts) - counts
        cells[internal, 1] = owners[ordered[starts[internal] + 1]]
        face_vertices = oriented_faces[first]
        triangles = self.nodes[face_vertices]
        cross_products = np.cross(triangles[:, 1] - triangles[:, 0], triangles[:, 2] - triangles[:, 0])
        double_areas = np.linalg.norm(cross_products, axis=1)
        if not np.isfinite(double_areas).all() or (double_areas <= 0).any():
            raise ValueError("triangle face areas must be finite and positive")
        normals = cross_products / double_areas[:, None]
        across_face = self.centers[cells[internal, 1]] - triangles[internal].mean(axis=1)
        if (np.einsum("ij,ij->i", normals[internal], across_face) <= 0).any():
            raise ValueError("tetrahedra sharing a face must lie on opposite sides")
        return FaceTable(face_vertices, cells, normals, double_areas / 2)

    @property
    def internal_faces(self):
        return np.flatnonzero(self.faces.cells[:, 1] >= 0)

    @cached_property
    def tetrahedron_aspect_ratios(self):
        result = tetrahedron_aspect_ratios(self.nodes, self.tets)
        result.setflags(write=False)
        return result

    @property
    def dual_edges(self):
        return self.faces.cells[self.internal_faces].copy()

    @cached_property
    def corner_face_pairs(self):
        cells = self.dual_edges
        vertices = self.faces.vertices[self.internal_faces]
        sides = []
        for side in range(2):
            local = np.argmax(self.tets[cells[:, side], :, None] == vertices[:, None, :], axis=1)
            sides.append(4 * cells[:, side, None] + local)
        pairs = np.stack(sides, axis=-1)
        pairs.setflags(write=False)
        return pairs

    def corner_groups(self, broken):
        broken = np.asarray(broken, dtype=bool)
        if broken.shape != (len(self.internal_faces),):
            raise ValueError("broken must contain one flag per internal material face")
        pairs = self.corner_face_pairs[~broken].reshape(-1, 2)
        return component_labels(4 * len(self.tets), pairs, np.zeros(len(pairs), dtype=bool))

    def save(self, path: str | Path):
        np.savez_compressed(path, nodes=self.nodes, tets=self.tets)

    @classmethod
    def load(cls, path: str | Path):
        with np.load(path, allow_pickle=False) as data:
            return cls(data["nodes"], data["tets"])
