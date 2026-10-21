from dataclasses import dataclass
import hashlib
import json
from pathlib import Path

import numpy as np

from .mesh import VolumeMesh
from .partition import component_labels


@dataclass(frozen=True)
class PreparedCase:
    mesh: VolumeMesh
    impacts: np.ndarray
    cell_labels: np.ndarray
    cut_labels: np.ndarray
    metadata: dict
    path: Path
    known_cells: np.ndarray
    valid_facets: np.ndarray


def _mesh_digest(mesh):
    digest = hashlib.sha256()
    digest.update(np.ascontiguousarray(mesh.nodes, dtype="<f8").tobytes())
    digest.update(np.ascontiguousarray(mesh.tets, dtype="<i8").tobytes())
    return digest.hexdigest()


def load_prepared_case(path, mesh=None):
    path = Path(path)
    if mesh is None:
        mesh = VolumeMesh.load(path.parent.parent / "mesh.npz")
    with np.load(path, allow_pickle=False) as archive:
        impacts = np.array(archive["impacts"], copy=True)
        labels = np.array(archive["cell_labels"], copy=True)
        cuts = np.array(archive["cut_labels"], copy=True)
        internal = np.array(archive["internal_face_ids"], copy=True)
        known = np.array(archive["known_cells"], dtype=bool, copy=True)
        valid_facets = np.array(archive["valid_facets"], dtype=bool, copy=True)
        metadata = json.loads(str(archive["metadata_json"].item()))
    if _mesh_digest(mesh) != metadata["mesh_sha256"]:
        raise ValueError("prepared case mesh hash mismatch")
    if impacts.ndim != 2 or impacts.shape[1] != 7 or not np.isfinite(impacts).all():
        raise ValueError("prepared impacts must have finite shape (N, 7)")
    if labels.shape != (len(mesh.tets),) or not np.issubdtype(labels.dtype, np.integer) or (labels < -1).any():
        raise ValueError("prepared cell labels are invalid")
    if known.shape != labels.shape or not np.array_equal(known, labels >= 0):
        raise ValueError("prepared known-cell mask disagrees with labels")
    if metadata.get("supervision_kind") != "native_facet_fracture":
        raise ValueError("the dynamics pipeline requires native FEMFX fracture supervision")
    if not np.array_equal(internal, mesh.internal_faces):
        raise ValueError("prepared internal face ordering differs from mesh")
    edges = mesh.dual_edges
    expected_valid = known[edges].all(axis=1)
    if not np.array_equal(valid_facets, expected_valid):
        raise ValueError("prepared valid-facet mask disagrees with known cells")
    _validate_native_fracture_case(mesh, labels, cuts, known, valid_facets, metadata)
    return PreparedCase(mesh, impacts, labels, cuts, metadata, path, known, valid_facets)


def _validate_native_fracture_case(mesh, labels, cuts, known, valid_facets, metadata):
    if metadata.get("source") != "FEMFX":
        raise ValueError("native fracture supervision requires FEMFX source metadata")
    context = metadata.get("conditioning_context_sha256")
    if not isinstance(context, str) or len(context) != 64 or any(
        value not in "0123456789abcdefABCDEF" for value in context
    ):
        raise ValueError("native fracture supervision requires a conditioning context SHA-256")
    if not known.all() or not valid_facets.all():
        raise ValueError("native fracture supervision requires complete cell and facet labels")
    if cuts.shape != (len(mesh.dual_edges),) or not np.issubdtype(cuts.dtype, np.integer) or not np.isin(cuts, [0, 1]).all():
        raise ValueError("native fracture cut labels must be a binary integer facet array")
    components = component_labels(len(mesh.tets), mesh.dual_edges, cuts)
    memberships = np.unique(np.column_stack((labels, components)), axis=0)
    if len(memberships) != len(np.unique(labels)) or len(memberships) != len(np.unique(components)):
        raise ValueError("native fragment labels disagree with connectivity of uncut facets")
