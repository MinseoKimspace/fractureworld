import json
from pathlib import Path

import pytest

from fracture_world.cases import Case
from fracture_world.generation2d import case_path, configuration, generate, smoke_configuration
from fracture_world.teacher2d import Pull


def small_config():
    config = smoke_configuration(configuration(Path(__file__).parents[1] / "configs" / "stage-a.json"))
    config["time"] = {"step": 0.01, "duration": 0.02}
    return config


def simulated_case(mesh, material, time, amplitude, ramp_time, **kwargs):
    mask, position, velocity = Pull(amplitude, ramp_time).boundary(mesh, time)
    return Case({
        "rest": mesh.rest, "triangles": mesh.triangles,
        "original_vertex": mesh.original_vertex, "cut_edges": mesh.cut_edges,
        "time": time, "position": position.copy(), "velocity": velocity.copy(),
        "boundary_mask": mask, "boundary_position": position,
        "boundary_velocity": velocity,
    }, {"material": material.metadata(), "method_label": "analytic_io_fixture", "runtime_seconds": 0.0})


def read_manifest(output):
    return json.loads((output / "collection.json").read_text(encoding="utf-8"))


def write_manifest(output, manifest):
    (output / "collection.json").write_text(json.dumps(manifest), encoding="utf-8")


def complete(monkeypatch, output):
    monkeypatch.setattr("fracture_world.generation2d.simulate", simulated_case)
    config = small_config()
    generate(config, output)
    return config


def test_interrupted_generation_resumes_only_verified_cases(monkeypatch, tmp_path):
    config = small_config()
    output = tmp_path / "data"
    calls = []

    def interrupted(*args, **kwargs):
        calls.append(1)
        if len(calls) == 3:
            raise RuntimeError("Interrupted fixture")
        return simulated_case(*args, **kwargs)

    monkeypatch.setattr("fracture_world.generation2d.simulate", interrupted)
    with pytest.raises(RuntimeError, match="Interrupted"):
        generate(config, output)
    manifest = read_manifest(output)
    assert manifest["status"] == "in_progress"
    assert len(manifest["cases"]) == 2
    saved = {row["path"]: (output / row["path"]).read_bytes() for row in manifest["cases"]}
    resumed = []

    def remaining(*args, **kwargs):
        resumed.append(1)
        return simulated_case(*args, **kwargs)

    monkeypatch.setattr("fracture_world.generation2d.simulate", remaining)
    result = generate(config, output, resume=True)
    assert result["cases"] == 5
    assert result["skipped_cases"] == 2
    assert result["generated_cases"] == len(resumed) == 3
    assert read_manifest(output)["status"] == "completed"
    for name, content in saved.items():
        assert (output / name).read_bytes() == content


def test_completed_resume_skips_every_case_and_allows_training_tuning(monkeypatch, tmp_path):
    output = tmp_path / "data"
    config = complete(monkeypatch, output)
    config["training"]["batch_size"] = 999
    config["training_profiles"]["a6000-48gb"]["epochs"] = 1

    def forbidden(*args, **kwargs):
        raise AssertionError("Completed cases must not run the solver again")

    monkeypatch.setattr("fracture_world.generation2d.simulate", forbidden)
    result = generate(config, output, resume=True)
    assert result["generated_cases"] == 0
    assert result["skipped_cases"] == 5


def test_resume_accepts_new_output(monkeypatch, tmp_path):
    monkeypatch.setattr("fracture_world.generation2d.simulate", simulated_case)
    result = generate(small_config(), tmp_path / "new", resume=True)
    assert result["generated_cases"] == 5
    assert result["skipped_cases"] == 0


def test_physical_configuration_change_rejects_resume(monkeypatch, tmp_path):
    output = tmp_path / "data"
    config = complete(monkeypatch, output)
    config["cut_columns"] = [2]
    with pytest.raises(ValueError, match="configuration differs"):
        generate(config, output, resume=True)


@pytest.mark.parametrize("change", ["missing", "checksum", "duplicate", "path", "input_hash"])
def test_invalid_resume_data_is_rejected(monkeypatch, tmp_path, change):
    output = tmp_path / "data"
    config = complete(monkeypatch, output)
    manifest = read_manifest(output)
    row = manifest["cases"][0]
    path = output / row["path"]
    if change == "missing":
        path.unlink()
    elif change == "checksum":
        with path.open("ab") as stream:
            stream.write(b"changed")
    elif change == "duplicate":
        manifest["cases"].append(dict(row))
        write_manifest(output, manifest)
    elif change == "path":
        row["path"] = "../outside.npz"
        write_manifest(output, manifest)
    else:
        row["case_hash"] = "incorrect"
        write_manifest(output, manifest)
    with pytest.raises(ValueError):
        generate(config, output, resume=True)


def test_legacy_rows_without_file_checksum_still_validate_inputs(monkeypatch, tmp_path):
    output = tmp_path / "data"
    config = complete(monkeypatch, output)
    manifest = read_manifest(output)
    for row in manifest["cases"]:
        row.pop("file_sha256")
    write_manifest(output, manifest)
    assert generate(config, output, resume=True)["skipped_cases"] == 5


def test_resume_refuses_unrelated_nonempty_output(tmp_path):
    (tmp_path / "keep.txt").write_text("keep", encoding="utf-8")
    with pytest.raises(ValueError, match="requires collection"):
        generate(small_config(), tmp_path, resume=True)
    assert (tmp_path / "keep.txt").read_text(encoding="utf-8") == "keep"


def test_fresh_generation_refuses_existing_collection(monkeypatch, tmp_path):
    config = complete(monkeypatch, tmp_path)
    with pytest.raises(ValueError, match="new or empty"):
        generate(config, tmp_path)


def test_case_paths_reject_parent_traversal(tmp_path):
    with pytest.raises(ValueError, match="inside"):
        case_path(tmp_path, "../outside.npz")
