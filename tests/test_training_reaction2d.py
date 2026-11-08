import json

import numpy as np
import pytest
import torch

from fracture_world.benchmark2d import benchmark
from fracture_world.cases import Case
from fracture_world.neural import DynamicsModel
from fracture_world.topology import rectangle
from fracture_world.training2d import fit, load_model, reaction_configuration, reaction_impulse_loss


def case_fixture(impulse=1.0, width=1.0):
    mesh = rectangle(2, 2, width=width)
    time = np.arange(3) * 0.01
    positions = np.broadcast_to(mesh.rest, (len(time), len(mesh.rest), 2)).copy()
    mask = np.zeros_like(mesh.rest, dtype=bool)
    mask[0] = True
    arrays = {
        "rest": mesh.rest, "triangles": mesh.triangles, "original_vertex": mesh.original_vertex,
        "cut_edges": mesh.cut_edges, "time": time, "position": positions,
        "velocity": np.zeros_like(positions), "boundary_mask": mask,
        "boundary_position": positions.copy(), "boundary_velocity": np.zeros_like(positions),
    }
    if impulse is not None:
        arrays["boundary_impulse"] = np.zeros((len(time) - 1, len(mesh.rest), 2))
        arrays["boundary_impulse"][:, mask] = impulse
    return Case(arrays, {"material": {"young": 1.0, "poisson": 0.2, "density": 1.0, "thickness": 1.0}})


def test_reaction_normalization_uses_training_boundary_components_only():
    training = case_fixture(impulse=2.0)
    validation = case_fixture(impulse=1000.0, width=1.1)
    config = reaction_configuration([training], [validation])
    assert config["enabled"]
    assert config["scale"] == pytest.approx(2.0)
    assert config["scale_source"] == "training_boundary_impulse_rms"


def test_reaction_supervision_missing_everywhere_is_explicitly_disabled():
    config = reaction_configuration([case_fixture(None)], [case_fixture(None, width=1.1)])
    assert config == {"enabled": False, "scale": None, "scale_source": "unavailable"}


def test_mixed_reaction_availability_is_rejected():
    with pytest.raises(ValueError, match="every training and validation case"):
        reaction_configuration([case_fixture()], [case_fixture(None, width=1.1)])


def test_invalid_reaction_targets_are_rejected():
    case = case_fixture()
    case.arrays["boundary_impulse"][:, 1, 0] = 1.0
    with pytest.raises(ValueError, match="unconstrained"):
        reaction_configuration([case], [case_fixture(width=1.1)])


def test_zero_training_reaction_has_finite_explicit_scale():
    config = reaction_configuration([case_fixture(0)], [case_fixture(10, width=1.1)])
    assert config["scale"] == 1.0
    assert config["scale_source"] == "unit_scale_zero_training_signal"


def test_reaction_loss_identifies_boundary_force_without_free_motion_error():
    mass = torch.ones(2)
    mask = torch.tensor([[True, True], [False, False]])
    previous = torch.zeros(2, 2)
    next_velocity = previous.clone()
    force = torch.tensor([[1.0, 3.0], [0.0, 0.0]], requires_grad=True)
    target = torch.zeros_like(force)
    loss = reaction_impulse_loss(mass, mask, previous, next_velocity, force, target, 0.1, 2.0)
    assert torch.equal(next_velocity[~mask], previous[~mask])
    assert loss.item() == pytest.approx(0.0125)
    loss.backward()
    assert torch.all(force.grad[mask] > 0)
    assert torch.equal(force.grad[~mask], torch.zeros_like(force.grad[~mask]))


def test_reaction_impulse_uses_prescribed_acceleration_and_interval_force():
    mass = torch.tensor([2.0])
    mask = torch.ones((1, 2), dtype=torch.bool)
    previous = torch.tensor([[1.0, 2.0]])
    next_velocity = torch.tensor([[1.2, 1.8]])
    force = torch.tensor([[3.0, -4.0]])
    target = mass[:, None] * (next_velocity - previous) - 0.1 * force
    assert reaction_impulse_loss(mass, mask, previous, next_velocity, force, target, 0.1, 1.0) == 0


@pytest.mark.parametrize("method", ["direct", "stress"])
def test_tiny_training_records_same_reaction_objective_for_both_methods(tmp_path, method):
    torch.set_num_threads(2)
    training, validation = case_fixture(), case_fixture(width=1.1)
    training.save(tmp_path / "train.npz")
    validation.save(tmp_path / "validation.npz")
    manifest = tmp_path / "collection.json"
    manifest.write_text(json.dumps({"format": "cut-dynamics-a-v1", "cases": [
        {"id": "train", "path": "train.npz", "split": "train", "group": "train"},
        {"id": "validation", "path": "validation.npz", "split": "validation", "group": "validation"},
    ]}), encoding="utf-8")
    result = fit(manifest, tmp_path / method, method, epochs=1, windows_per_epoch=1,
                 batch_size=1, rollout_steps=1, hidden=8, layers=1)
    _, checkpoint = load_model(result["checkpoint"])
    assert result["training_loss_terms"] == ["position", "velocity", "boundary_impulse"]
    assert checkpoint["training_loss_terms"] == result["training_loss_terms"]
    assert checkpoint["reaction_supervision"]
    assert checkpoint["reaction_scale"] == pytest.approx(1.0)
    assert checkpoint["trained"] is True


def test_untrained_checkpoint_requires_explicit_pipeline_mode(tmp_path):
    case = case_fixture()
    case.save(tmp_path / "test.npz")
    manifest = tmp_path / "collection.json"
    manifest.write_text(json.dumps({"format": "cut-dynamics-a-v1", "cases": [
        {"id": "test", "path": "test.npz", "split": "test", "group": "test"},
    ]}), encoding="utf-8")
    model = DynamicsModel("direct", hidden_size=8, layers=1)
    checkpoint = tmp_path / "untrained.pt"
    torch.save({"format": "cut-dynamics-a-v1", "model_config": model.configuration(),
                "state_dict": model.state_dict(), "dt": 0.01, "trained": False,
                "training_case_hashes": [], "training_groups": []}, checkpoint)
    with pytest.raises(ValueError, match="allow_untrained"):
        benchmark(manifest, [checkpoint], tmp_path / "denied")
    result = benchmark(manifest, [checkpoint], tmp_path / "allowed", allow_untrained=True)
    assert result["purpose"] == "untrained_pipeline_check"
    assert result["methods"]["direct"]["training_status"] == "untrained"
