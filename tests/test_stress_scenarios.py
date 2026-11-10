import hashlib
import json
from collections import Counter
from pathlib import Path

import numpy as np
import pytest
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import connected_components

from fracture_world.cases import Case
from fracture_world.generation2d import cut_mesh
from fracture_world.stress_scenarios import (
    DEFAULT_SOURCE_CASES,
    evaluation_plan,
    nominal_load,
    novel_loads,
    plan_summary,
    topology_meshes,
)
from fracture_world.teacher2d import Material, Pull
from fracture_world.topology import lumped_mass, mesh_edges, rectangle


@pytest.fixture
def dataset(tmp_path):
    config = json.loads(Path("configs/stage-a.json").read_text(encoding="utf-8"))
    selected = [
        ("x7-depth7-load12", 7, 7, 12),
        ("x3-depth8-load12", 3, 8, 12),
        ("x5-depth8-load12", 5, 8, 12),
        ("x4-depth7-load8", 4, 7, 8),
        ("x4-depth8-load8", 4, 8, 8),
    ]
    material = Material(**config["material"])
    time = np.linspace(0, 0.2, 101)
    rows = []
    for case_id, column, depth, load_index in selected:
        mesh = cut_mesh(config, column, depth)
        load = config["loads"][load_index]
        mask, position, velocity = Pull(**load).boundary(mesh, time)
        case = Case({
            "rest": mesh.rest, "triangles": mesh.triangles,
            "original_vertex": mesh.original_vertex, "cut_edges": mesh.cut_edges,
            "time": time, "position": position.copy(), "velocity": velocity.copy(),
            "boundary_mask": mask, "boundary_position": position.copy(),
            "boundary_velocity": velocity.copy(),
        }, {"material": material.metadata(), "pull": load})
        relative = f"cases/{case_id}.npz"
        case.save(tmp_path / relative)
        rows.append({
            "id": case_id, "group": f"x{column}-depth{depth}",
            "split": "test", "path": relative,
            "case_hash": str(case.arrays["case_hash"].item()),
            "file_sha256": hashlib.sha256((tmp_path / relative).read_bytes()).hexdigest(),
        })
    manifest = tmp_path / "collection.json"
    manifest.write_text(json.dumps({
        "format": "cut-dynamics-a-v1", "status": "completed",
        "configuration": config, "cases": rows,
    }), encoding="utf-8")
    return manifest


def components(mesh):
    edges = mesh_edges(mesh)
    adjacency = coo_matrix(
        (np.ones(len(edges)), (edges[:, 0], edges[:, 1])),
        shape=(len(mesh.rest), len(mesh.rest)),
    ).tocsr()
    return connected_components(adjacency, directed=False)[0]


def test_plan_is_small_read_only_and_preserves_source_inputs(dataset):
    paths = [dataset, *dataset.parent.glob("cases/*.npz")]
    before = {path: hashlib.sha256(path.read_bytes()).hexdigest() for path in paths}
    plan = evaluation_plan(dataset)
    assert len(plan) == len({row["id"] for row in plan}) == 21
    assert Counter(row["category"] for row in plan) == {
        "force_audit": 3, "long_rollout": 3, "topology": 3, "load": 10, "control": 2,
    }
    assert {row["source_case"] for row in plan[:6]} == set(DEFAULT_SOURCE_CASES)
    assert {path: hashlib.sha256(path.read_bytes()).hexdigest() for path in paths} == before
    assert sorted(path.relative_to(dataset.parent).as_posix() for path in dataset.parent.rglob("*") if path.is_file()) == sorted(
        path.relative_to(dataset.parent).as_posix() for path in paths
    )
    for row in plan:
        source = Case.load(row["source_path"])
        assert row["source_case_hash"] == str(source.arrays["case_hash"].item())
        assert row["time_step"] == 0.002
        assert row["material"].metadata() == source.metadata["material"]
        if row["category"] != "topology":
            np.testing.assert_array_equal(row["mesh"].triangles, source.mesh.triangles)
            np.testing.assert_array_equal(row["mesh"].rest, source.mesh.rest)
        if row["category"] in {"force_audit", "long_rollout", "topology"}:
            assert row["load"] == source.metadata["pull"]
        if row["category"] == "long_rollout":
            assert row["duration"] == 1.0
            assert row["novelty"]["horizon"] == "extended"
        else:
            assert row["duration"] == 0.2
            assert row["novelty"]["horizon"] == "original"


def test_nominal_controls_do_not_change_topology_load_or_horizon(dataset):
    controls = [row for row in evaluation_plan(dataset) if row["category"] == "control"]
    assert {row["source_case"] for row in controls} == {"x4-depth7-load8", "x4-depth8-load8"}
    for row in controls:
        source = Case.load(row["source_path"])
        assert row["novelty_flags"] == {"topology": False, "load": False, "horizon": False}
        assert row["load"] == source.metadata["pull"]
        assert row["duration"] == source.arrays["time"][-1]
        np.testing.assert_array_equal(row["mesh"].triangles, source.mesh.triangles)


def test_topology_variants_only_change_initial_material_connections(dataset):
    plan = evaluation_plan(dataset)
    expected = {
        "horizontal-full": 2,
        "diagonal-boundary-to-boundary": 2,
        "double-vertical-full": 3,
    }
    base = rectangle(8, 8)
    mass = lumped_mass(base).sum()
    for row in plan:
        if row["category"] != "topology":
            continue
        mesh = row["mesh"]
        assert components(mesh) == expected[row["topology"]]
        assert row["novelty"]["topology"] == "new_initial_cut_pattern"
        assert row["load"] == {"amplitude": 0.0225, "ramp_time": 0.048}
        assert lumped_mass(mesh).sum() == pytest.approx(mass)
        np.testing.assert_array_equal(mesh.original_vertex[mesh.triangles], base.triangles)


def test_novel_loads_change_one_dimension_on_unchanged_topology(dataset):
    rows = [row for row in evaluation_plan(dataset) if row["category"] == "load"]
    expected = {
        "interpolation-amplitude": {"amplitude": 0.01875, "ramp_time": 0.048},
        "interpolation-ramp": {"amplitude": 0.0225, "ramp_time": 0.05},
        "extrapolation-amplitude-high": {"amplitude": 0.0375, "ramp_time": 0.048},
        "extrapolation-ramp-slow": {"amplitude": 0.0225, "ramp_time": 0.08},
        "extrapolation-ramp-fast": {"amplitude": 0.0225, "ramp_time": 0.03},
    }
    counts = Counter()
    for row in rows:
        source = Case.load(row["source_path"])
        name = row["novelty"]["load"]
        assert row["load"] == pytest.approx(expected[name])
        assert sum(row["load"][key] != source.metadata["pull"][key] for key in row["load"]) == 1
        assert row["novelty"]["topology"] == "unchanged_source_topology"
        counts[name] += 1
    assert counts == {name: 2 for name in expected}


def test_plan_summary_is_json_serializable_without_arrays_or_writes(dataset):
    plan = evaluation_plan(dataset, source_cases=[DEFAULT_SOURCE_CASES[0]], horizon=0.4)
    summary = plan_summary(plan)
    assert summary["scenario_count"] == 17
    assert summary["categories"]["force_audit"] == 1
    assert summary["categories"]["long_rollout"] == 1
    encoded = json.dumps(summary, allow_nan=False)
    assert "new_initial_cut_pattern" in encoded
    assert "instantaneous" not in encoded


@pytest.mark.parametrize("horizon", [True, 0, -1, float("nan"), float("inf"), 0.2, 1.001])
def test_invalid_or_nonextended_horizons_are_rejected(dataset, horizon):
    with pytest.raises(ValueError):
        evaluation_plan(dataset, horizon=horizon)


@pytest.mark.parametrize("source_cases", [[], ["missing"], [DEFAULT_SOURCE_CASES[0]] * 2])
def test_invalid_source_selection_is_rejected(dataset, source_cases):
    with pytest.raises(ValueError, match="Source cases|held-out test split"):
        evaluation_plan(dataset, source_cases=source_cases)


def test_source_hash_mismatch_is_rejected(dataset):
    manifest = json.loads(dataset.read_text(encoding="utf-8"))
    manifest["cases"][0]["case_hash"] = "incorrect"
    dataset.write_text(json.dumps(manifest), encoding="utf-8")
    with pytest.raises(ValueError, match="hash differs"):
        evaluation_plan(dataset)


def test_source_material_mismatch_is_rejected_even_with_matching_file_input_hash(dataset):
    manifest = json.loads(dataset.read_text(encoding="utf-8"))
    entry = manifest["cases"][0]
    path = dataset.parent / entry["path"]
    original = Case.load(path)
    arrays = {key: value.copy() for key, value in original.arrays.items() if key != "case_hash"}
    metadata = dict(original.metadata)
    metadata["material"] = {**metadata["material"], "young": 200.0}
    changed = Case(arrays, metadata)
    changed.save(path)
    entry["case_hash"] = str(changed.arrays["case_hash"].item())
    entry["file_sha256"] = hashlib.sha256(path.read_bytes()).hexdigest()
    dataset.write_text(json.dumps(manifest), encoding="utf-8")
    with pytest.raises(ValueError, match="material differs"):
        evaluation_plan(dataset)


def test_future_truth_file_corruption_is_detected_even_when_input_hash_is_unchanged(dataset):
    manifest = json.loads(dataset.read_text(encoding="utf-8"))
    path = dataset.parent / manifest["cases"][0]["path"]
    original = Case.load(path)
    input_hash = str(original.arrays["case_hash"].item())
    free_node = np.flatnonzero(~original.arrays["boundary_mask"][:, 0])[0]
    original.arrays["position"][2, free_node, 0] += 1e-3
    original.save(path)
    assert str(Case.load(path).arrays["case_hash"].item()) == input_hash
    with pytest.raises(ValueError, match="file checksum differs"):
        evaluation_plan(dataset)


def test_nominal_load_uses_existing_lower_middle_combination():
    loads = [{"amplitude": amplitude, "ramp_time": ramp}
             for amplitude in (0.015, 0.0225, 0.03)
             for ramp in (0.04, 0.044, 0.048, 0.052, 0.056, 0.06)]
    assert nominal_load(loads) == {"amplitude": 0.0225, "ramp_time": 0.048}
    baseline, variants = novel_loads(loads)
    assert baseline in loads
    assert all(load not in loads for _, load in variants)


def test_interpolation_requires_distinct_load_parameters():
    with pytest.raises(ValueError, match="two distinct"):
        novel_loads([{"amplitude": 0.02, "ramp_time": 0.05}])


def test_topology_suite_supports_rectangular_grid_without_boundary_edge_cuts():
    meshes = topology_meshes({"nx": 4, "ny": 6, "width": 1.0, "height": 1.0})
    assert components(meshes["horizontal-full"]) == 2
    assert components(meshes["diagonal-boundary-to-boundary"]) == 2
    assert components(meshes["double-vertical-full"]) == 3
