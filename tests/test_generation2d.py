import json
from pathlib import Path

import pytest

from fracture_world.cases import Case, collection
from fracture_world.generation2d import configuration, cut_mesh, generate, smoke_configuration
from fracture_world.stage_a import main


def source_config():
    return configuration(Path(__file__).parents[1] / "configs" / "stage-a.json")


def test_real_small_collection_has_disjoint_cut_families(tmp_path):
    config = smoke_configuration(source_config())
    config["time"] = {"step": 0.005, "duration": 0.02}
    config["integration"]["max_step"] = 0.0025
    result = generate(config, tmp_path / "data")
    assert result["cases"] == 5
    seen = set()
    for split in ("train", "validation", "test"):
        rows = collection(result["collection"], split)
        groups = {row["group"] for row, _ in rows}
        assert not seen.intersection(groups)
        seen.update(groups)
        for row, path in rows:
            case = Case.load(path)
            assert case.metadata["split"] == split
            assert case.metadata["material"]["kinematics"] == "plane_strain"
            assert case.metadata["material"]["calibrated"] is False
            assert str(case.arrays["case_hash"].item()) == row["case_hash"]


def test_invalid_depth_split_is_rejected(tmp_path):
    config = source_config()
    config["cut_depths"]["test"] = config["cut_depths"]["train"]
    path = tmp_path / "bad.json"
    path.write_text(json.dumps(config), encoding="utf-8")
    with pytest.raises(ValueError, match="disjoint"):
        configuration(path)


def test_training_profile_only_applies_when_train_is_explicit(monkeypatch, tmp_path):
    captured = {}

    def fake_fit(**kwargs):
        captured.update(kwargs)
        return {"training_executed": False}

    monkeypatch.setattr("fracture_world.training2d.fit", fake_fit)
    config_path = Path(__file__).parents[1] / "configs" / "stage-a.json"
    main(["train", "--config", str(config_path), "--profile", "a6000-48gb",
          "--data", "unused.json", "--method", "stress", "--output", str(tmp_path),
          "--epochs", "3", "--device", "cuda"])
    assert captured["epochs"] == 3
    assert captured["batch_size"] == 256
    assert captured["hidden"] == 128
    assert captured["layers"] == 6
    assert captured["rollout_steps"] == 4
    assert captured["device"] == "cuda"
    assert captured["patience"] == 6


def test_cut_geometry_preserves_material_area():
    from fracture_world.topology import lumped_mass

    config = source_config()
    intact = cut_mesh(config, 0, 0)
    partial = cut_mesh(config, 2, 4)
    full = cut_mesh(config, 2, 8)
    assert lumped_mass(intact).sum() == pytest.approx(lumped_mass(partial).sum())
    assert lumped_mass(intact).sum() == pytest.approx(lumped_mass(full).sum())
    assert len(intact.rest) < len(partial.rest) < len(full.rest)


def test_incomplete_generation_is_not_training_data(tmp_path):
    path = tmp_path / "collection.json"
    path.write_text(json.dumps({"format": "cut-dynamics-a-v1", "status": "in_progress", "cases": []}), encoding="utf-8")
    with pytest.raises(ValueError, match="not completed"):
        collection(path, "train")
