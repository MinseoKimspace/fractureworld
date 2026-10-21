import json
from pathlib import Path

import numpy as np
import pytest

from fracture_world.data import load_prepared_case
from fracture_world.femfx import (
    context_digest,
    create_trial,
    generate_dataset,
    initial_conditions,
    native_directory,
    reference_surface,
    validate_specification,
)
from fracture_world.mesh import VolumeMesh


def test_trial_impulses_preserve_the_explicit_loading_and_units(tmp_path):
    trial = create_trial(tmp_path)
    mesh = VolumeMesh.load(trial["mesh"])
    specification = json.loads(Path(trial["specification"]).read_text(encoding="utf-8"))
    material, fixed = validate_specification(mesh, specification)
    assert mesh.volumes.sum() == pytest.approx(0.12 * 0.04 * 0.04)
    assert trial["parameter_status"] == "uncalibrated_trial"
    velocities, mask, impacts = initial_conditions(mesh, material, fixed, specification["cases"][-1])
    mass = np.zeros(len(mesh.nodes))
    np.add.at(mass, mesh.tets.ravel(), np.repeat(mesh.volumes * material["rest_density"] / 4, 4))
    expected = np.sum([value["impulse"] for value in specification["cases"][-1]["vertex_impulses"]], axis=0)
    np.testing.assert_allclose((mass[:, None] * velocities).sum(axis=0), expected)
    np.testing.assert_allclose((impacts[:, 3:6] * impacts[:, 6:]).sum(axis=0), expected)
    assert not velocities[mask].any()
    assert mask.sum() == len(fixed)
    assert context_digest({"a": 1, "b": [2]}) == context_digest({"b": [2], "a": 1})
    specification["cases"][0]["vertex_impulses"][0]["vertex"] = int(fixed[0])
    with pytest.raises(ValueError, match="fixed vertex"):
        validate_specification(mesh, specification)


def test_reference_surface_uses_both_deformed_sides_of_a_crack():
    mesh = VolumeMesh([[0, 0, 0], [1, 0, 0], [0, 1, 0], [0, 0, 1], [1, 1, 1]],
                      [[0, 1, 2, 3], [1, 2, 3, 4]])
    corners = mesh.nodes[mesh.tets].copy()
    intact = reference_surface(mesh, corners, [0])
    corners[1] += [0.1, 0.1, 0.1]
    opened = reference_surface(mesh, corners, [1])
    assert len(opened["faces"]) == len(intact["faces"]) + 2
    left, right = np.asarray(opened["vertices"])[-6:].reshape(2, 3, 3)
    np.testing.assert_allclose(right.mean(axis=0) - left.mean(axis=0), [0.1, 0.1, 0.1])


@pytest.mark.skipif(not (native_directory() / "manifest.json").exists(), reason="native Windows FEMFX reference is not built")
def test_actual_femfx_generates_complete_irreversible_native_fracture_labels(tmp_path, capsys):
    trial = create_trial(tmp_path / "input")
    result = generate_dataset(trial["mesh"], trial["specification"], tmp_path / "reference", 1.0)
    assert len(result["cases"]) == 4
    assert sum(row["cut_facets"] for row in result["cases"]) > 0
    assert result["support_release_corner_events"] > 0
    for row in result["cases"]:
        path = tmp_path / "reference" / "cases" / f"{row['case']}.npz"
        case = load_prepared_case(path)
        assert case.valid_facets.all()
        assert case.known_cells.all()
        assert case.metadata["source"] == "FEMFX"
        with np.load(tmp_path / "reference" / "trajectories" / f"{row['case']}.npz") as trajectory:
            assert len(trajectory["times"]) == 51
            assert (np.diff(trajectory["cut_labels"].astype(int), axis=0) >= 0).all()
            np.testing.assert_array_equal(trajectory["cut_labels"][-1], case.cut_labels)
            support = trajectory["corner_kinematic"]
            np.testing.assert_array_equal(trajectory["support_release_labels"], support[:-1] & ~support[1:])
            np.testing.assert_array_equal(trajectory["recorded_step_ids"], np.arange(51))
            assert int(trajectory["support_release_labels"].sum()) == row["support_release_corner_events"]
        replay = json.loads((tmp_path / "reference" / "replays" / f"{row['case']}.json").read_text())
        assert replay["metadata"]["neural_prediction"] is False
        assert replay["metadata"]["proximal_fracture_correction"] is False
        assert replay["frames"][-1]["metrics"]["cut_facets"] == row["cut_facets"]
    assert '"cut_facets"' in capsys.readouterr().out
