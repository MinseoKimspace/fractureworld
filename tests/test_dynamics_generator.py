import json
from pathlib import Path

import numpy as np
import pytest

from fracture_world.dynamics_generator import create_dynamics_suite
from fracture_world.femfx import initial_conditions, validate_specification
from fracture_world.io import write_json
from fracture_world.mesh import VolumeMesh


def configuration():
    return {"steps": 50, "velocity_magnitudes": [0.05, 0.5, 2.0, 4.0],
            "directions": [[1, 0, 0], [0, 1, 0], [0, 0, 1]],
            "loaded_regions": ["entire_end", "upper_y"]}


def test_suite_preserves_teacher_material_and_builds_distinct_mass_weighted_loads(tmp_path):
    config = tmp_path / "config.json"
    write_json(config, configuration())
    result = create_dynamics_suite(tmp_path / "suite", config)
    mesh = VolumeMesh.load(result["mesh"])
    specification = json.loads(Path(result["specification"]).read_text())
    manifest = json.loads(Path(result["manifest"]).read_text())
    material, fixed = validate_specification(mesh, specification)
    assert result["case_count"] == 24
    assert result["cells"] == 144
    assert specification["steps"] == 50
    assert specification["record_stride"] == 1
    assert material == {"rest_density": 1000.0, "youngs_modulus": 1e6, "poissons_ratio": 0.25,
                        "fracture_stress_threshold": 1e4}
    assert specification["scene_options"] == {"friction_coefficient": 0.0}
    np.testing.assert_array_equal(fixed, np.flatnonzero(mesh.nodes[:, 0] == 0))
    signatures = set()
    for case, parameters in zip(specification["cases"], manifest["cases"], strict=True):
        velocities, mask, _ = initial_conditions(mesh, material, fixed, case)
        selected = np.asarray(parameters["loaded_vertices"])
        expected = parameters["velocity_magnitude_m_per_s"] * np.asarray(parameters["direction"])
        np.testing.assert_allclose(velocities[selected], np.broadcast_to(expected, (len(selected), 3)))
        assert not velocities[mask].any()
        assert not np.isin(selected, fixed).any()
        assert (mesh.nodes[selected, 0] == mesh.nodes[:, 0].max()).all()
        if parameters["loaded_region"] == "upper_y":
            assert (mesh.nodes[selected, 1] > 0.02).all()
            assert len(selected) == 3
        else:
            assert len(selected) == 9
        signatures.add(velocities.tobytes())
    assert len(signatures) == 24
    assert manifest["parameter_status"] == "uncalibrated_trial"
    assert not manifest["synthetic_release_labels"]
    assert not manifest["additional_removable_support_flags"]
    assert not manifest["positive_release_examples_confirmed"]
    assert not result["simulation_executed"]


def test_suite_normalizes_directions_and_is_reproducible_without_native_runtime(tmp_path):
    config = tmp_path / "config.json"
    values = configuration()
    values["directions"] = [[3, 4, 0]]
    write_json(config, values)
    first = create_dynamics_suite(tmp_path / "first", config)
    second = create_dynamics_suite(tmp_path / "second", config)
    assert Path(first["specification"]).read_bytes() == Path(second["specification"]).read_bytes()
    manifest = json.loads(Path(first["manifest"]).read_text())
    np.testing.assert_allclose(manifest["cases"][0]["direction"], [0.6, 0.8, 0])
    assert first["case_count"] == 8
    with pytest.raises(ValueError, match="empty"):
        create_dynamics_suite(tmp_path / "first", config)


def test_thousand_case_preset_covers_distinct_loads_and_preserves_original_cases(tmp_path):
    configs = Path(__file__).resolve().parents[1] / "configs"
    preset = configs / "dynamics-teacher-1000.json"
    values = json.loads(preset.read_text(encoding="utf-8"))
    original = json.loads((configs / "dynamics-teacher.json").read_text(encoding="utf-8"))
    directions = np.asarray(values["directions"])
    assert len(values["velocity_magnitudes"]) == 10
    assert len(directions) == 50
    assert values["steps"] == original["steps"]
    assert values["loaded_regions"] == original["loaded_regions"]
    assert set(original["velocity_magnitudes"]).issubset(values["velocity_magnitudes"])
    assert min(values["velocity_magnitudes"]) == min(original["velocity_magnitudes"])
    assert max(values["velocity_magnitudes"]) == max(original["velocity_magnitudes"])
    np.testing.assert_allclose(np.linalg.norm(directions, axis=1), 1.0, atol=1e-11)
    assert len(np.unique(np.sign(directions[np.all(directions != 0, axis=1)]), axis=0)) == 8
    for axis in np.concatenate([np.eye(3), -np.eye(3)]):
        assert np.any(np.all(directions == axis, axis=1))
    assert np.linalg.norm(directions.mean(axis=0)) < 0.01
    result = create_dynamics_suite(tmp_path / "suite", preset)
    mesh = VolumeMesh.load(result["mesh"])
    specification = json.loads(Path(result["specification"]).read_text(encoding="utf-8"))
    material, fixed = validate_specification(mesh, specification)
    velocities = [initial_conditions(mesh, material, fixed, case)[0] for case in specification["cases"]]
    assert result["case_count"] == len(velocities) == 1000
    assert len({value.tobytes() for value in velocities}) == 1000
    for speed in original["velocity_magnitudes"]:
        for direction in original["directions"]:
            for region in original["loaded_regions"]:
                selected = mesh.nodes[:, 0] == mesh.nodes[:, 0].max()
                if region == "upper_y":
                    selected &= mesh.nodes[:, 1] > 0.02
                expected = np.zeros_like(mesh.nodes)
                expected[selected] = speed * np.asarray(direction)
                assert any(np.allclose(value, expected, rtol=0, atol=1e-14) for value in velocities)


@pytest.mark.parametrize("change", [
    {"steps": True}, {"steps": 0}, {"velocity_magnitudes": []},
    {"velocity_magnitudes": [1, 1]}, {"velocity_magnitudes": [-1]},
    {"velocity_magnitudes": [True]}, {"velocity_magnitudes": [float("inf")]},
    {"directions": [[0, 0, 0]]}, {"directions": [[1, 0]]},
    {"directions": [[1, 0, 0], [2, 0, 0]]}, {"directions": [[True, 0, 0]]},
    {"loaded_regions": ["fixed_face"]}, {"loaded_regions": ["entire_end", "entire_end"]},
    {"extra_material_law": True}, {"schema": "unknown"},
])
def test_invalid_sweeps_fail_before_creating_a_teacher_dataset(tmp_path, change):
    config = tmp_path / "config.json"
    config.write_text(json.dumps({**configuration(), **change}), encoding="utf-8")
    with pytest.raises(ValueError):
        create_dynamics_suite(tmp_path / "suite", config)
    assert not (tmp_path / "suite").exists()
