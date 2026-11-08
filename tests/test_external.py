import json

import numpy as np
import pytest

from fracture_world.external import compare_external


def trajectory():
    rest = np.array([[0.0, 0.0], [1.0, 0.0], [0.0, 1.0]])
    position = np.stack([rest, rest + [0.01, 0.0], rest + [0.02, 0.0]])
    return {
        "rest": rest,
        "triangles": np.array([[0, 1, 2]]),
        "original_vertex": np.arange(3),
        "time": np.array([0.0, 0.1, 0.2]),
        "position": position,
        "velocity": np.zeros_like(position),
        "boundary_position": position.copy(),
        "boundary_velocity": np.zeros_like(position),
        "boundary_mask": np.zeros((3, 2), dtype=bool),
        "case_hash": np.array("matching-case"),
        "metadata_json": np.array(json.dumps({
            "material": {"law": "stvk", "young": 100.0, "poisson": 0.2, "density": 2.0, "thickness": 0.5},
            "method_label": "Wind Lifter planar adaptation",
            "source_url": "https://github.com/Changy1506/WindLifter",
            "source_revision": "1234567890abcdef",
            "runtime_seconds": 0.5,
        })),
    }


def write_pair(tmp_path, change=None):
    reference = trajectory()
    external = trajectory()
    if change is not None:
        change(external)
    reference_path = tmp_path / "reference.npz"
    external_path = tmp_path / "external.npz"
    np.savez(reference_path, **reference)
    np.savez(external_path, **external)
    return reference_path, external_path


def test_identical_validated_inputs_compare_motion_errors(tmp_path):
    def change(external):
        external["position"][1:, :, 0] += 0.03
        external["velocity"][1:, :, 1] += 0.06

    result = compare_external(*write_pair(tmp_path, change))
    assert result["status"] == "compatible"
    assert result["position_rmse"] == pytest.approx(0.03)
    assert result["velocity_rmse"] == pytest.approx(0.06)
    assert result["rmse_convention"] == "Mass-weighted Euclidean point error, excluding initial frame"
    assert result["final_position_rmse"] == pytest.approx(0.03)
    assert result["completed_physical_horizon"] == pytest.approx(0.2)
    assert result["seconds_per_step"] == pytest.approx(0.25)
    assert result["provenance"]["source_revision"] == "1234567890abcdef"


@pytest.mark.parametrize("name", [
    "rest", "triangles", "original_vertex", "time", "boundary_position",
    "boundary_velocity", "boundary_mask", "case_hash",
])
def test_case_input_mismatch_is_rejected(tmp_path, name):
    def change(external):
        if name == "triangles":
            external[name] = np.array([[1, 2, 0]])
        elif name == "case_hash":
            external[name] = np.array("another-case")
        elif name == "boundary_mask":
            external[name][0, 0] = True
        else:
            external[name].flat[-1] += 1

    with pytest.raises(ValueError, match=name):
        compare_external(*write_pair(tmp_path, change))


@pytest.mark.parametrize("name", ["position", "velocity"])
def test_initial_state_mismatch_is_rejected(tmp_path, name):
    def change(external):
        external[name][0, 0, 0] += 0.01

    with pytest.raises(ValueError, match=f"initial {name}"):
        compare_external(*write_pair(tmp_path, change))


@pytest.mark.parametrize("field,value", [
    ("source_url", ""),
    ("source_revision", "main"),
    ("source_revision", ""),
    ("runtime_seconds", -1.0),
    ("runtime_seconds", float("nan")),
    ("runtime_seconds", True),
    ("method_label", "teacher"),
    ("material", {"law": "different"}),
])
def test_provenance_and_material_must_be_verifiable(tmp_path, field, value):
    def change(external):
        metadata = json.loads(str(external["metadata_json"]))
        metadata[field] = value
        external["metadata_json"] = np.array(json.dumps(metadata))

    with pytest.raises(ValueError):
        compare_external(*write_pair(tmp_path, change))


def test_truncated_trajectory_is_not_reported_as_full_success(tmp_path):
    def change(external):
        for name in ("time", "position", "velocity", "boundary_position", "boundary_velocity"):
            external[name] = external[name][:-1]

    with pytest.raises(ValueError, match="time"):
        compare_external(*write_pair(tmp_path, change))


def test_nonfinite_prediction_is_rejected(tmp_path):
    def change(external):
        external["position"][1, 0, 0] = np.nan

    with pytest.raises(ValueError, match="finite"):
        compare_external(*write_pair(tmp_path, change))


def test_three_dimensional_wind_output_requires_explicit_conversion(tmp_path):
    def change(external):
        external["position"] = np.pad(external["position"], ((0, 0), (0, 0), (0, 1)))

    with pytest.raises(ValueError, match="position must have shape"):
        compare_external(*write_pair(tmp_path, change))


def test_external_metrics_use_nodal_mass_and_exclude_initial_frame(tmp_path):
    rest = np.array([[0.0, 0.0], [1.0, 0.0], [0.0, 1.0], [2.0, 0.0]])
    reference = trajectory()
    reference["rest"] = rest
    reference["triangles"] = np.array([[0, 1, 2], [1, 3, 2]])
    reference["original_vertex"] = np.arange(4)
    reference["position"] = np.broadcast_to(rest, (3, 4, 2)).copy()
    reference["velocity"] = np.zeros((3, 4, 2))
    reference["boundary_position"] = reference["position"].copy()
    reference["boundary_velocity"] = reference["velocity"].copy()
    reference["boundary_mask"] = np.zeros((4, 2), dtype=bool)
    external = {name: value.copy() for name, value in reference.items()}
    external["position"][1:, 0, 0] += 0.03
    reference_path = tmp_path / "reference.npz"
    external_path = tmp_path / "external.npz"
    np.savez(reference_path, **reference)
    np.savez(external_path, **external)
    result = compare_external(reference_path, external_path)
    assert result["position_rmse"] == pytest.approx(0.03 / np.sqrt(6))
    assert result["final_position_rmse"] == pytest.approx(result["position_rmse"])
