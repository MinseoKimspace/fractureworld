import numpy as np
import pytest
import torch

from fracture_world.cases import Case
from fracture_world.profile2d import profile_case
from fracture_world.topology import rectangle


def case_file(tmp_path):
    mesh = rectangle(2, 2)
    times = np.linspace(0, 0.02, 5)
    positions = np.broadcast_to(mesh.rest, (len(times), len(mesh.rest), 2)).copy()
    velocities = np.zeros_like(positions)
    case = Case({
        "rest": mesh.rest, "triangles": mesh.triangles,
        "original_vertex": mesh.original_vertex, "time": times,
        "position": positions, "velocity": velocities,
        "boundary_position": positions.copy(), "boundary_velocity": velocities.copy(),
        "boundary_mask": np.zeros_like(mesh.rest, dtype=bool),
    }, {"material": {"young": 1.0, "poisson": 0.2, "density": 1.0, "thickness": 1.0},
        "method_label": "analytic_rest"})
    path = tmp_path / "case.npz"
    case.save(path)
    return path


def test_profile_does_not_train_or_write_artifacts(tmp_path, monkeypatch):
    torch.set_num_threads(2)
    path = case_file(tmp_path)

    def forbidden(*args, **kwargs):
        raise AssertionError("Profiling must not construct optimizers or save checkpoints")

    monkeypatch.setattr(torch.optim.Optimizer, "__init__", forbidden)
    monkeypatch.setattr(torch, "save", forbidden)
    before = sorted(item.name for item in tmp_path.iterdir())
    result = profile_case(path, device="cpu", batch_sizes=(1, 2), hidden=8,
                          layers=1, rollout_steps=2, repeats=2)
    assert result["maximum_parameter_change"] == 0
    assert result["optimizer_steps"] == 0
    assert result["checkpoint_written"] is False
    assert result["parameter_count"] > 0
    assert [record["status"] for record in result["records"]] == ["completed", "completed"]
    assert all(record["windows_per_second"] > 0 for record in result["records"])
    assert all(record["cpu_peak_rss_bytes_sampled"] > 0 for record in result["records"])
    assert sorted(item.name for item in tmp_path.iterdir()) == before


@pytest.mark.parametrize("kwargs", [
    {"batch_sizes": ()}, {"batch_sizes": (1, 1)}, {"batch_sizes": (0,)},
    {"rollout_steps": 5}, {"repeats": 0}, {"device": "meta"},
])
def test_profile_rejects_invalid_requests(tmp_path, kwargs):
    path = case_file(tmp_path)
    parameters = {"device": "cpu", "batch_sizes": (1,), "hidden": 8,
                  "layers": 1, "rollout_steps": 2, "repeats": 1, **kwargs}
    with pytest.raises(ValueError):
        profile_case(path, **parameters)
