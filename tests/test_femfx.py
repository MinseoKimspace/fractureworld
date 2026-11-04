import json
from pathlib import Path
from types import SimpleNamespace

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
from test_dynamics_data import recorded_trajectory


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


def test_rejected_replay_export_cannot_leave_discoverable_training_archives(tmp_path, monkeypatch):
    import fracture_world.femfx as femfx
    from fracture_world.dynamics_data import load_dynamics_dataset

    case, trajectory = recorded_trajectory(tmp_path / "source")
    mesh = VolumeMesh.load(case.parent.parent / "mesh.npz")
    with np.load(case, allow_pickle=False) as archive:
        metadata = json.loads(str(archive["metadata_json"].item()))
    with np.load(trajectory, allow_pickle=False) as archive:
        arrays = {key: np.array(archive[key], copy=True) for key in archive.files}
    material = metadata["conditioning_context"]["material"]
    masses = np.bincount(mesh.tets.ravel(), weights=np.repeat(mesh.volumes * material["rest_density"] / 4, 4))

    class RecordedScene:
        def __init__(self, *args, **kwargs):
            self.frame = 0

        def state(self):
            frame = self.frame
            selected = slice(*arrays["native_node_offsets"][frame:frame + 2])
            state = {"time": float(arrays["times"][frame]), "warnings": 0,
                     "original_tet_ids": np.arange(len(mesh.tets)),
                     "face_vertices": mesh.faces.vertices[mesh.internal_faces],
                     "cut_labels": arrays["cut_labels"][frame],
                     "corner_positions": arrays["corner_positions"][frame],
                     "corner_velocities": arrays["corner_velocities"][frame],
                     "corner_vertex_ids": arrays["native_corner_vertex_ids"][frame],
                     "tet_piece_ids": arrays["native_piece_ids"][frame],
                     "initial_vertex_masses": masses, "fracture_times": arrays["times"][1:]}
            for name in ("rest_positions", "positions", "velocities", "masses", "kinematic", "flags", "original_ids"):
                state[f"nodal_{name}"] = arrays[f"native_nodal_{name}"][selected]
            state["corner_kinematic"] = state["nodal_kinematic"][state["corner_vertex_ids"]]
            state["nodal_piece_ids"] = np.zeros(len(state["nodal_masses"]), dtype=int)
            for cell, piece in zip(state["corner_vertex_ids"], state["tet_piece_ids"], strict=True):
                state["nodal_piece_ids"][cell] = piece
            return state

        def step(self):
            self.frame += 1
            return self.state()

        def settings(self):
            return metadata["native_settings"]

        def close(self):
            pass

    native = SimpleNamespace(Scene=RecordedScene, capabilities=lambda: {})
    monkeypatch.setattr(femfx, "load_native", lambda _: (native, {"binding_sha256": "0" * 64}))
    export = femfx.export_reference_replay

    def fail_first_replay(mesh, states, cuts, metadata, path):
        if metadata["case"]["id"] == "rejected":
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text("incomplete replay", encoding="utf-8")
            raise ValueError("simulated replay export failure")
        return export(mesh, states, cuts, metadata, path)

    monkeypatch.setattr(femfx, "export_reference_replay", fail_first_replay)
    specification = {"schema": femfx.SCHEMA, "parameter_status": "uncalibrated_trial", "material": material,
                     "fixed_vertices": [], "dt": .125, "steps": 3, "gravity": [0, 0, 0],
                     "cases": [{**metadata["case"], "id": identifier} for identifier in ("rejected", "accepted")]}
    specification_path = tmp_path / "specification.json"
    specification_path.write_text(json.dumps(specification), encoding="utf-8")
    output = tmp_path / "generated"
    report = generate_dataset(case.parent.parent / "mesh.npz", specification_path, output, 1., skip_invalid=True)
    assert [item["case"] for item in report["cases"]] == ["accepted"]
    assert [item["case"] for item in report["rejected_cases"]] == ["rejected"]
    assert not list(output.rglob("rejected.*"))
    assert [item.case_id for item in load_dynamics_dataset(output)] == ["accepted"]


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
