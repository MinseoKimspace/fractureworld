import json

import numpy as np
import pytest

from fracture_world.cases import Case, collection
from fracture_world.topology import rectangle


def free_case():
    mesh = rectangle(2, 2)
    time = np.linspace(0, 0.1, 5)
    velocity = np.broadcast_to([0.2, -0.1], (5, len(mesh.rest), 2)).copy()
    position = mesh.rest[None] + time[:, None, None] * velocity
    return Case({
        "rest": mesh.rest, "triangles": mesh.triangles,
        "original_vertex": mesh.original_vertex, "time": time,
        "position": position, "velocity": velocity,
        "boundary_mask": np.zeros_like(mesh.rest, dtype=bool),
        "boundary_position": position.copy(), "boundary_velocity": velocity.copy(),
    }, {"material": {"young": 1.0, "poisson": 0.2, "density": 1.0, "thickness": 1.0},
        "method_label": "reference"})


def test_case_roundtrip_and_target_isolation(tmp_path):
    case = free_case()
    path = tmp_path / "case.npz"
    case.save(path)
    loaded = Case.load(path)
    assert np.array_equal(loaded.arrays["position"], case.arrays["position"])
    original = str(case.arrays["case_hash"])
    loaded.arrays["position"][1:, 0] += 0.01
    assert str(Case(loaded.arrays, loaded.metadata).arrays["case_hash"]) == original


def test_changed_initial_condition_rejects_old_hash():
    case = free_case()
    case.arrays["velocity"][0, 0, 0] += 0.1
    with pytest.raises(ValueError, match="hash"):
        Case(case.arrays, case.metadata)


def test_group_split_leakage_is_rejected(tmp_path):
    path = tmp_path / "collection.json"
    path.write_text(json.dumps({"format": "cut-dynamics-a-v1", "cases": [
        {"id": "a", "path": "a.npz", "split": "train", "group": "same-cut"},
        {"id": "b", "path": "b.npz", "split": "test", "group": "same-cut"},
    ]}), encoding="utf-8")
    with pytest.raises(ValueError, match="leaks"):
        collection(path, "test")


def test_boundary_mismatch_is_rejected():
    case = free_case()
    case.arrays["boundary_mask"][0] = True
    case.arrays["boundary_position"][:, 0] = 100
    with pytest.raises(ValueError, match="boundary"):
        Case(case.arrays, case.metadata)
