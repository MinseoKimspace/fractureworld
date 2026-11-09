import hashlib
import json

import numpy as np
import pytest
import torch

from fracture_world.cases import Case
from fracture_world.neural import DynamicsModel
from fracture_world.profile2d import profile_case
from fracture_world.stage_a import main
from fracture_world.topology import rectangle
from fracture_world.training2d import (
    fit,
    curriculum_patience,
    force_configuration,
    load_model,
    rollout_curriculum,
    rollout_for_epoch,
    validation_subset,
)


def force_case(force=0.0, width=1.0):
    mesh = rectangle(2, 2, width=width)
    time = np.arange(5) * 0.002
    position = np.broadcast_to(mesh.rest, (len(time), *mesh.rest.shape)).copy()
    return Case({
        "rest": mesh.rest, "triangles": mesh.triangles, "original_vertex": mesh.original_vertex,
        "cut_edges": mesh.cut_edges, "time": time, "position": position,
        "velocity": np.zeros_like(position), "boundary_mask": np.zeros_like(mesh.rest, dtype=bool),
        "boundary_position": position.copy(), "boundary_velocity": np.zeros_like(position),
        "internal_force": np.full_like(position, force),
    }, {"material": {"young": 1.0, "poisson": 0.2, "density": 1.0, "thickness": 1.0}})


def test_force_scale_uses_training_data_only():
    config = force_configuration([force_case(2.0)], [force_case(1000.0, width=1.1)])
    assert config["scale"] == pytest.approx(2.0)
    assert config["scale_source"] == "training_instantaneous_force_rms"
    assert config["target"] == "instantaneous_nodal_internal_force_at_window_start"


def test_zero_force_scale_is_explicit_and_not_an_epsilon():
    config = force_configuration([force_case()], [force_case(width=1.1)])
    assert config["scale"] == 1.0
    assert config["scale_source"] == "unnormalized_zero_training_force"


@pytest.mark.parametrize("fault", ["missing", "shape", "nonfinite"])
def test_force_supervision_rejects_invalid_labels(fault):
    case = force_case()
    if fault == "missing":
        del case.arrays["internal_force"]
    elif fault == "shape":
        case.arrays["internal_force"] = case.arrays["internal_force"][:-1]
    else:
        case.arrays["internal_force"][0, 0, 0] = np.nan
    with pytest.raises(ValueError, match="instantaneous internal_force"):
        force_configuration([case], [force_case(width=1.1)])


def test_rollout_schedule_reaches_requested_horizon():
    schedule = rollout_curriculum(16)
    assert schedule == [1, 2, 4, 8, 16]
    assert rollout_for_epoch(schedule, 1, 10) == 1
    assert rollout_for_epoch(schedule, 10, 10) == 16
    assert rollout_for_epoch(schedule, 1, 1) == 16
    assert rollout_curriculum(3) == [1, 2, 3]


def test_early_stopping_cannot_skip_final_curriculum_stage():
    schedule = [1, 2, 4]
    stale, previous = 0, None
    stages = []
    for epoch in range(1, 41):
        active = rollout_for_epoch(schedule, epoch, 40)
        stale, stopped = curriculum_patience(stale, epoch == 1, previous, active, schedule[-1], 6)
        stages.append(active)
        previous = active
        if stopped:
            break
    assert stages.count(1) > 6
    assert stages.count(2) > 6
    assert stages.count(4) == 6
    assert epoch == 36


def test_curriculum_transition_resets_patience_and_improvement_restarts_it():
    assert curriculum_patience(20, False, 2, 4, 4, 6) == (1, False)
    assert curriculum_patience(5, True, 4, 4, 4, 6) == (0, False)
    assert curriculum_patience(5, False, 4, 4, 4, 6) == (6, True)
    assert curriculum_patience(5, False, 2, 2, 4, 6) == (6, False)
    assert curriculum_patience(20, False, 4, 4, 4, None) == (21, False)


def test_explicit_curriculum_ramp_leaves_final_horizon_training_budget():
    stages = [rollout_for_epoch([1, 2, 4], epoch, 40, curriculum_epochs=12)
              for epoch in range(1, 41)]
    assert stages[0] == 1
    assert all(stage == 4 for stage in stages[11:])


@pytest.mark.parametrize("curriculum_epochs", [0, -1, 41, True, 0.5])
def test_invalid_curriculum_budget_is_rejected_before_data_loading(tmp_path, curriculum_epochs):
    with pytest.raises(ValueError, match="Curriculum epochs"):
        fit(tmp_path / "missing.json", tmp_path / "unused", "stress", epochs=40,
            architecture="equilibrium", curriculum_epochs=curriculum_epochs)
    assert not (tmp_path / "unused").exists()


@pytest.mark.parametrize("schedule", [[], [0, 2], [2, 1], [1, 1, 2], [1, 4], [True, 2]])
def test_rollout_schedule_rejects_invalid_lengths(schedule):
    with pytest.raises(ValueError, match="Rollout schedule"):
        rollout_curriculum(2, schedule)


def test_autonomous_validation_subset_prioritizes_distinct_topologies():
    rows = [{"group": group} for group in ("a", "a", "a", "b", "c", "c")]
    indices = validation_subset(rows, 10, 3, 17)
    assert {rows[index - 10]["group"] for index in indices} == {"a", "b", "c"}
    assert indices == validation_subset(rows, 10, 3, 17)
    assert len(set(validation_subset(rows, 10, 100, 17))) == len(rows)


@pytest.mark.parametrize("method", ["direct", "stress"])
def test_legacy_checkpoint_load_is_exact_and_read_only(tmp_path, method):
    torch.manual_seed(3)
    original = DynamicsModel(method, hidden_size=8, layers=1)
    path = tmp_path / "best.pt"
    torch.save({"format": "cut-dynamics-a-v1", "model_config": original.configuration(),
                "state_dict": original.state_dict()}, path)
    before = hashlib.sha256(path.read_bytes()).hexdigest()
    restored, payload = load_model(path)
    assert payload["format"] == "cut-dynamics-a-v1"
    assert restored.configuration() == original.configuration()
    assert all(torch.equal(value, restored.state_dict()[name]) for name, value in original.state_dict().items())
    assert hashlib.sha256(path.read_bytes()).hexdigest() == before


@pytest.mark.parametrize("method", ["direct", "stress"])
def test_equilibrium_training_smoke_uses_saved_force_labels_and_preserves_inputs(tmp_path, method):
    pytest.importorskip("felupe")
    from fracture_world.teacher2d import Material, simulate

    torch.set_num_threads(2)
    rows = []
    for split, width in (("train", 1.0), ("validation", 1.1)):
        path = tmp_path / f"{split}.npz"
        case = simulate(rectangle(2, 2, width=width), Material(young=1.0), np.arange(5) * 0.002,
                        amplitude=0.002, ramp_time=0.004)
        case.save(path)
        rows.append({"id": split, "split": split, "group": split, "path": path.name})
    manifest = tmp_path / "collection.json"
    manifest.write_text(json.dumps({"format": "cut-dynamics-a-v1", "cases": rows}), encoding="utf-8")
    files = [manifest, *(tmp_path / row["path"] for row in rows)]
    hashes = {path: hashlib.sha256(path.read_bytes()).hexdigest() for path in files}
    result = fit(manifest, tmp_path / method, method, architecture="equilibrium", epochs=1,
                 batch_size=1, windows_per_epoch=1, rollout_steps=2, hidden=8, layers=1)
    model, payload = load_model(result["checkpoint"])
    assert model.configuration()["architecture"] == "equilibrium"
    assert payload["format"] == "cut-dynamics-a-v2"
    assert payload["training_loss_terms"] == ["position", "velocity", "boundary_impulse", "instantaneous_internal_force"]
    assert payload["checkpoint_selection"] == "autonomous_validation"
    assert len(payload["autonomous_validation_case_hashes"]) == 1
    assert np.isfinite(payload["autonomous_validation_loss"])
    assert payload["active_rollout_steps"] == 2
    assert payload["force_supervision"]["enabled"]
    assert model.decoder[-1].weight.abs().max() > 0
    assert {path: hashlib.sha256(path.read_bytes()).hexdigest() for path in files} == hashes


def test_equilibrium_profile_only_backpropagates_without_training_or_artifacts(tmp_path, monkeypatch):
    path = tmp_path / "case.npz"
    force_case().save(path)
    before = hashlib.sha256(path.read_bytes()).hexdigest()

    def forbidden(*args, **kwargs):
        raise AssertionError("Profiling cannot train or write checkpoints")

    monkeypatch.setattr(torch.optim.Optimizer, "__init__", forbidden)
    monkeypatch.setattr(torch, "save", forbidden)
    result = profile_case(path, device="cpu", architecture="equilibrium", batch_sizes=(1,),
                          hidden=8, layers=1, rollout_steps=2, repeats=1)
    assert result["architecture"] == "equilibrium"
    assert result["instantaneous_force_supervision"]
    assert result["optimizer_steps"] == 0
    assert result["maximum_parameter_change"] == 0
    assert result["records"][0]["status"] == "completed"
    assert hashlib.sha256(path.read_bytes()).hexdigest() == before
    assert [item.name for item in tmp_path.iterdir()] == ["case.npz"]


@pytest.mark.parametrize("architecture", [None, "legacy"])
def test_training_cli_selects_architecture_without_starting_training(monkeypatch, capsys, architecture):
    received = {}

    def fake_fit(**kwargs):
        received.update(kwargs)
        return {"test_only": True}

    monkeypatch.setattr("fracture_world.training2d.fit", fake_fit)
    args = ["train", "--data", "unused.json", "--output", "unused", "--method", "stress"]
    if architecture is not None:
        args += ["--architecture", architecture]
    main(args)
    assert received["architecture"] == (architecture or "equilibrium")
    assert received["autonomous_validation_cases"] == 8
    assert json.loads(capsys.readouterr().out)["test_only"]


def test_equilibrium_validation_cli_does_not_require_output_directory(monkeypatch, capsys):
    received = {}

    def fake_validate(**kwargs):
        received.update(kwargs)
        return {"test_only": True}

    monkeypatch.setattr("fracture_world.equilibrium_validation.validate_equilibrium", fake_validate)
    main(["validate-equilibrium", "--device", "cpu", "--hidden", "8", "--layers", "1", "--no-teacher"])
    assert received == {"device": "cpu", "hidden": 8, "layers": 1, "include_teacher": False}
    assert json.loads(capsys.readouterr().out)["test_only"]
