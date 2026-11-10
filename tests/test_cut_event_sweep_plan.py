import copy
import hashlib
import json
from pathlib import Path

import numpy as np
import pytest

from fracture_world.cases import Case
from fracture_world.cut_event_sweep_plan import sweep_plan
from fracture_world.generation2d import cut_mesh, file_hash
from fracture_world.teacher2d import Material, Pull


def configuration():
    return {
        "mesh": {"nx": 4, "ny": 4, "width": 1.0, "height": 1.0},
        "material": {"young": 100.0, "poisson": 0.25, "density": 1.0, "thickness": 1.0},
        "time": {"step": 0.002, "duration": 0.2}, "cut_columns": [1, 2, 3],
        "cut_depths": {"train": [1], "validation": [2], "test": [3, 4]},
        "loads": [{"amplitude": amplitude, "ramp_time": 0.048} for amplitude in (0.015, 0.0225, 0.03)],
    }


def dataset(tmp_path, config=None, columns=None, load_indices=None):
    config = configuration() if config is None else copy.deepcopy(config)
    columns = config["cut_columns"] if columns is None else columns
    load_indices = range(len(config["loads"])) if load_indices is None else load_indices
    depth = max(value for value in config["cut_depths"]["test"] if value < config["mesh"]["ny"])
    dt, duration = config["time"]["step"], config["time"]["duration"]
    time = np.linspace(0, duration, round(duration / dt) + 1)
    rows = []
    for column in columns:
        mesh = cut_mesh(config, column, depth)
        for index in load_indices:
            load = config["loads"][index]
            mask, positions, velocities = Pull(**load).boundary(mesh, time)
            source = Case({
                "rest": mesh.rest.copy(), "triangles": mesh.triangles.copy(),
                "original_vertex": mesh.original_vertex.copy(), "cut_edges": mesh.cut_edges.copy(),
                "time": time.copy(), "position": positions.copy(), "velocity": velocities.copy(),
                "boundary_mask": mask, "boundary_position": positions, "boundary_velocity": velocities,
            }, {"material": Material(**config["material"]).metadata(), "pull": load})
            identity = f"x{column}-depth{depth}-load{index}"
            path = tmp_path / f"{identity}.npz"
            source.save(path)
            rows.append({"id": identity, "group": f"x{column}-depth{depth}", "split": "test",
                         "column": column, "depth": depth, "load_index": index, "load": load,
                         "path": path.name, "case_hash": str(source.arrays["case_hash"].item()),
                         "file_sha256": file_hash(path)})
    data = tmp_path / "collection.json"
    data.write_text(json.dumps({"format": "cut-dynamics-a-v1", "configuration": config, "cases": rows}), encoding="utf-8")
    return data


def test_default_stage_a_sweep_covers_full_factorial_without_changing_inputs(tmp_path, monkeypatch):
    config_path = Path(__file__).resolve().parents[1] / "configs" / "stage-a.json"
    config = json.loads(config_path.read_text(encoding="utf-8"))
    data = dataset(tmp_path, config, columns=[1, 4, 7], load_indices=[2, 8, 14])
    before = {path.name: file_hash(path) for path in tmp_path.iterdir()}
    monkeypatch.setattr("fracture_world.teacher2d.simulate", lambda *args, **kwargs: pytest.fail("Planner ran a simulation"))
    scenarios, description = sweep_plan(data)
    assert len(scenarios) == description["requested_count"] == description["scenario_count"] == 27
    assert description["columns"] == [1, 4, 7]
    assert description["load_indices"] == [2, 8, 14]
    assert description["event_times"] == [0.016, 0.024, 0.032]
    assert description["factor_level_counts"] == {"columns": 3, "loads": 3, "event_times": 3}
    assert len(description["source_cases"]) == 9
    assert len({scenario["id"] for scenario in scenarios}) == 27
    assert len({scenario["experiment_hash"] for scenario in scenarios}) == 27
    assert {(scenario["sweep"]["column"], scenario["sweep"]["load_index"], scenario["sweep"]["event_step"])
            for scenario in scenarios} == {(column, load, step) for column in (1, 4, 7)
                                          for load in (2, 8, 14) for step in (8, 12, 16)}
    for scenario in scenarios:
        assert scenario["inputs"]["initial_cut_depth"] == 7
        assert scenario["inputs"]["final_cut_depth"] == 8
        assert scenario["id"].endswith(f"-at-step-{scenario['sweep']['event_step']:04d}")
        encoded = json.dumps(scenario["inputs"], sort_keys=True, allow_nan=False).encode()
        assert hashlib.sha256(encoded).hexdigest() == scenario["experiment_hash"]
    assert before == {path.name: file_hash(path) for path in tmp_path.iterdir()}
    assert json.loads(json.dumps(description, allow_nan=False)) == description


def test_nominal_load_selection_is_independent_of_load_array_order(tmp_path):
    config = configuration()
    config["loads"] = [{"amplitude": amplitude, "ramp_time": ramp} for amplitude in (0.015, 0.0225, 0.03)
                       for ramp in (0.04, 0.048, 0.06)]
    config["loads"] = [config["loads"][index] for index in (8, 1, 3, 5, 0, 7, 2, 6, 4)]
    data = dataset(tmp_path, config)
    _, description = sweep_plan(data)
    assert description["load_indices"] == [1, 8, 5]
    assert [load["amplitude"] for load in description["loads"]] == [0.015, 0.0225, 0.03]
    assert {load["ramp_time"] for load in description["loads"]} == {0.048}


def test_fewer_available_levels_are_reported_and_default_rounding_is_explicit(tmp_path):
    config = configuration()
    config["cut_columns"] = [2]
    config["loads"] = [{"amplitude": 0.015, "ramp_time": 0.04}]
    scenarios, description = sweep_plan(dataset(tmp_path, config))
    assert description["factor_level_counts"] == {"columns": 1, "loads": 1, "event_times": 3}
    assert description["requested_count"] == 3
    assert description["event_times"] == [0.014, 0.02, 0.026000000000000002]
    assert description["requested_event_times"] == pytest.approx([0.04 / 3, 0.02, 0.08 / 3])
    assert description["event_time_adjustments"] == pytest.approx([0.014 - 0.04 / 3, 0, 0.026 - 0.08 / 3])
    assert description["event_time_rounding_policy"] == "nearest_output_step_ties_to_even"
    for scenario in scenarios:
        factors = scenario["sweep"]
        assert factors["event_time_adjustment"] == factors["event_time"] - factors["requested_event_time"]


def test_explicit_levels_and_horizon_are_respected(tmp_path):
    data = dataset(tmp_path)
    scenarios, description = sweep_plan(data, columns=[3], load_indices=[2], event_times=[0.012, 0.026], horizon=0.3)
    assert len(scenarios) == 2
    assert description["event_time_selection"] == "explicit_grid_aligned"
    assert description["horizons"] == [0.3]
    assert all(scenario["time"][-1] == 0.3 for scenario in scenarios)
    assert all(scenario["sweep"]["column"] == 3 and scenario["sweep"]["load_index"] == 2 for scenario in scenarios)


def test_mixed_selected_ramps_share_a_single_valid_default_event_grid(tmp_path):
    config = configuration()
    config["loads"][1]["ramp_time"] = 0.036
    scenarios, description = sweep_plan(dataset(tmp_path, config), columns=[2], load_indices=[0, 1])
    assert description["event_time_reference_ramp"] == 0.036
    assert description["event_times"] == pytest.approx([0.012, 0.018, 0.024])
    assert len(scenarios) == 6
    assert {scenario["sweep"]["event_step"] for scenario in scenarios if scenario["sweep"]["load_index"] == 0} == {6, 9, 12}
    assert {scenario["sweep"]["event_step"] for scenario in scenarios if scenario["sweep"]["load_index"] == 1} == {6, 9, 12}


@pytest.mark.parametrize("kwargs", [
    {"columns": []}, {"columns": [1, 1]}, {"columns": [0]}, {"columns": [True]}, {"columns": [1.0]},
    {"load_indices": []}, {"load_indices": [1, 1]}, {"load_indices": [3]}, {"load_indices": [False]},
    {"event_times": []}, {"event_times": [0.012, 0.012]}, {"event_times": [0.013]},
    {"event_times": [0.0]}, {"event_times": [float("nan")]}, {"event_times": [float("inf")]},
    {"event_times": [True]}, {"event_times": [0.048]}, {"horizon": 0.02}, {"horizon": 0.201},
])
def test_invalid_requested_factor_levels_fail_instead_of_being_dropped(tmp_path, kwargs):
    with pytest.raises(ValueError):
        sweep_plan(dataset(tmp_path), **kwargs)


def test_default_event_levels_that_collapse_on_output_grid_are_rejected(tmp_path):
    config = configuration()
    config["loads"] = [{"amplitude": 0.015, "ramp_time": 0.004}]
    with pytest.raises(ValueError, match="collapse"):
        sweep_plan(dataset(tmp_path, config))


def test_missing_source_combination_and_corrupted_source_are_rejected(tmp_path):
    data = dataset(tmp_path, columns=[2])
    with pytest.raises(ValueError, match="absent from the held-out"):
        sweep_plan(data)
    manifest = json.loads(data.read_text(encoding="utf-8"))
    source = tmp_path / manifest["cases"][0]["path"]
    with source.open("ab") as stream:
        stream.write(b"modified")
    with pytest.raises(ValueError, match="checksum"):
        sweep_plan(data, columns=[2])


def test_source_future_states_do_not_change_experiment_inputs(tmp_path):
    config = configuration()
    config["cut_columns"] = [2]
    config["loads"] = config["loads"][:1]
    data = dataset(tmp_path, config)
    original, _ = sweep_plan(data)
    manifest = json.loads(data.read_text(encoding="utf-8"))
    row = manifest["cases"][0]
    path = tmp_path / row["path"]
    case = Case.load(path)
    free = ~case.arrays["boundary_mask"]
    case.arrays["position"][1:, free] += 0.123
    case.arrays["velocity"][1:, free] -= 0.456
    case.save(path)
    row["file_sha256"] = file_hash(path)
    data.write_text(json.dumps(manifest), encoding="utf-8")
    changed, _ = sweep_plan(data)
    assert [scenario["experiment_hash"] for scenario in changed] == [scenario["experiment_hash"] for scenario in original]
    assert [scenario["inputs"] for scenario in changed] == [scenario["inputs"] for scenario in original]
