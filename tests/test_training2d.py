import json

import numpy as np
import pytest
import torch

from fracture_world.cases import Case
from fracture_world.render2d import render
from fracture_world.topology import rectangle, split_edges
from fracture_world.training2d import fit, load_model


@pytest.mark.parametrize("device", ["cpu", pytest.param("cuda", marks=pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA unavailable"))])
def test_small_batched_training_and_render(tmp_path, device):
    torch.set_num_threads(2)
    rows = []
    mesh = rectangle(2, 2)
    for index, split in enumerate(("train", "train", "validation")):
        if split == "validation":
            mesh = rectangle(2, 2, width=1.1)
        current = mesh if index == 0 else split_edges(mesh, [[1, 4]])
        times = np.linspace(0, 0.02, 5)
        positions = np.broadcast_to(current.rest, (len(times), len(current.rest), 2)).copy()
        velocities = np.zeros_like(positions)
        case = Case({
            "rest": current.rest, "triangles": current.triangles,
            "original_vertex": current.original_vertex, "cut_edges": current.cut_edges,
            "time": times, "position": positions, "velocity": velocities,
            "boundary_mask": np.zeros_like(current.rest, dtype=bool),
            "boundary_position": positions.copy(), "boundary_velocity": velocities.copy(),
        }, {"material": {"young": 1.0, "poisson": 0.2, "density": 1.0, "thickness": 1.0},
            "method_label": "analytic_rest"})
        path = tmp_path / f"case-{index}.npz"
        case.save(path)
        rows.append({"id": str(index), "split": split, "group": str(index), "path": path.name})
    manifest = tmp_path / "collection.json"
    manifest.write_text(json.dumps({"format": "cut-dynamics-a-v1", "cases": rows}), encoding="utf-8")
    result = fit(manifest, tmp_path / "training", "stress", epochs=1,
                 windows_per_epoch=4, batch_size=2, hidden=8, layers=1, device=device)
    model, checkpoint = load_model(result["checkpoint"])
    assert model.method == "stress"
    assert checkpoint["epoch"] == 1
    assert np.isfinite(result["best_validation_loss"])
    output = tmp_path / "render.png"
    render([tmp_path / "case-0.npz"], output)
    assert output.stat().st_size > 1000
