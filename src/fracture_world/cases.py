import hashlib
import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from .topology import Mesh, lumped_mass


def input_hash(arrays, material):
    digest = hashlib.sha256(json.dumps(material, sort_keys=True).encode())
    for name in (
        "rest", "triangles", "original_vertex", "time", "boundary_mask",
        "boundary_position", "boundary_velocity", "cut_edges",
    ):
        value = np.ascontiguousarray(arrays[name])
        digest.update(name.encode())
        digest.update(str(value.shape).encode())
        digest.update(value.dtype.str.encode())
        digest.update(value.tobytes())
    for name in ("position", "velocity"):
        digest.update(np.ascontiguousarray(arrays[name][0]).tobytes())
    return digest.hexdigest()


@dataclass
class Case:
    arrays: dict
    metadata: dict

    def __post_init__(self):
        a = self.arrays
        a.setdefault("cut_edges", np.empty((0, 2), dtype=np.int64))
        mesh = self.mesh
        time = np.asarray(a["time"])
        if time.ndim != 1 or len(time) < 2 or not np.all(np.diff(time) > 0):
            raise ValueError("Time must be strictly increasing with at least two frames")
        shape = (len(time), len(mesh.rest), 2)
        for key in ("position", "velocity", "boundary_position", "boundary_velocity"):
            if np.shape(a[key]) != shape or not np.isfinite(a[key]).all():
                raise ValueError(f"Invalid {key} array")
        if np.shape(a["boundary_mask"]) != shape[1:]:
            raise ValueError("Boundary mask must contain one flag per coordinate")
        if a["boundary_mask"].dtype != np.bool_:
            raise ValueError("Boundary mask must be boolean")
        mask = a["boundary_mask"]
        for state, target in (("position", "boundary_position"), ("velocity", "boundary_velocity")):
            if not np.allclose(a[state][:, mask], a[target][:, mask], rtol=1e-6, atol=1e-10):
                raise ValueError(f"Prescribed boundary {state} is violated")
        if not np.isfinite(time).all():
            raise ValueError("Non-finite time")
        expected = input_hash(a, self.metadata["material"])
        if "case_hash" in a and str(a["case_hash"].item()) != expected:
            raise ValueError("Case input hash does not match stored inputs")
        a["case_hash"] = np.array(expected)

    @property
    def mesh(self):
        a = self.arrays
        return Mesh(a["rest"], a["triangles"], a["original_vertex"], a.get("cut_edges", []))

    @property
    def mass(self):
        material = self.metadata["material"]
        return lumped_mass(self.mesh, material["density"], material["thickness"])

    def save(self, path):
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(path, **self.arrays, metadata_json=json.dumps(self.metadata, sort_keys=True))

    @classmethod
    def load(cls, path):
        with np.load(path, allow_pickle=False) as archive:
            arrays = {key: archive[key] for key in archive.files if key != "metadata_json"}
            metadata = json.loads(str(archive["metadata_json"].item()))
        return cls(arrays, metadata)


def collection(path, split=None):
    path = Path(path)
    manifest = json.loads(path.read_text(encoding="utf-8"))
    if manifest.get("format") != "cut-dynamics-a-v1":
        raise ValueError("Expected a stage A dataset collection")
    if manifest.get("status", "completed") != "completed":
        raise ValueError("Dataset generation has not completed")
    groups = {}
    for item in manifest["cases"]:
        group = str(item["group"])
        previous = groups.setdefault(group, item["split"])
        if previous != item["split"]:
            raise ValueError("A cut configuration group leaks across dataset splits")
    rows = [row for row in manifest["cases"] if split is None or row["split"] == split]
    if not rows:
        raise ValueError(f"No cases for split {split!r}")
    return [(row, path.parent / row["path"]) for row in rows]
