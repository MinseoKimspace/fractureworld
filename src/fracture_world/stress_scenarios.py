import json
from collections import Counter
from pathlib import Path

import numpy as np

from .cases import Case, collection
from .generation2d import case_path, file_hash, validate_configuration
from .teacher2d import Material
from .topology import Mesh, rectangle, split_edges


DEFAULT_SOURCE_CASES = (
    "x7-depth7-load12",
    "x3-depth8-load12",
    "x5-depth8-load12",
)


def nominal_load(loads):
    amplitudes = sorted({float(load["amplitude"]) for load in loads})
    ramps = sorted({float(load["ramp_time"]) for load in loads})
    amplitude = amplitudes[(len(amplitudes) - 1) // 2]
    ramp = ramps[(len(ramps) - 1) // 2]
    candidates = [load for load in loads if float(load["amplitude"]) == amplitude]
    selected = min(candidates, key=lambda load: (abs(load["ramp_time"] - ramp), load["ramp_time"]))
    return {key: float(selected[key]) for key in ("amplitude", "ramp_time")}


def novel_loads(loads):
    baseline = nominal_load(loads)
    amplitudes = sorted({float(load["amplitude"]) for load in loads})
    ramps = sorted({float(load["ramp_time"]) for load in loads})
    if len(amplitudes) < 2 or len(ramps) < 2:
        raise ValueError("Load interpolation requires at least two distinct amplitudes and ramp times")
    amplitude = 0.5 * (amplitudes[0] + amplitudes[1])
    center = (len(ramps) - 2) // 2
    ramp = 0.5 * (ramps[center] + ramps[center + 1])
    values = (
        ("interpolation-amplitude", {**baseline, "amplitude": amplitude}),
        ("interpolation-ramp", {**baseline, "ramp_time": ramp}),
        ("extrapolation-amplitude-high", {**baseline, "amplitude": 1.25 * amplitudes[-1]}),
        ("extrapolation-ramp-slow", {**baseline, "ramp_time": 4 * ramps[-1] / 3}),
        ("extrapolation-ramp-fast", {**baseline, "ramp_time": 0.75 * ramps[0]}),
    )
    known = {(float(load["amplitude"]), float(load["ramp_time"])) for load in loads}
    for name, load in values:
        if (load["amplitude"], load["ramp_time"]) in known:
            raise ValueError(f"Requested novel loading is already present: {name}")
    return baseline, list(values)


def topology_meshes(mesh_config):
    nx, ny = mesh_config["nx"], mesh_config["ny"]
    if nx < 3 or ny < 2:
        raise ValueError("The topology suite requires at least three columns and two rows")
    base = rectangle(**mesh_config)
    row = ny // 2
    horizontal = row * (nx + 1) + np.arange(nx + 1)
    diagonal = np.arange(min(nx, ny) + 1) * (nx + 2)
    columns = (nx // 3, 2 * nx // 3)
    vertical = [np.arange(ny + 1) * (nx + 1) + column for column in columns]
    cuts = {
        "horizontal-full": np.column_stack((horizontal[:-1], horizontal[1:])),
        "diagonal-boundary-to-boundary": np.column_stack((diagonal[:-1], diagonal[1:])),
        "double-vertical-full": np.concatenate([
            np.column_stack((vertices[:-1], vertices[1:])) for vertices in vertical
        ]),
    }
    return {name: split_edges(base, edges) for name, edges in cuts.items()}


def copy_mesh(mesh):
    return Mesh(*(value.copy() for value in (
        mesh.rest, mesh.triangles, mesh.original_vertex, mesh.cut_edges,
    )))


def evaluation_plan(data, source_cases=None, horizon=1.0):
    data = Path(data).resolve()
    manifest = json.loads(data.read_text(encoding="utf-8-sig"))
    config = validate_configuration(manifest["configuration"])
    material = Material(**config["material"])
    sources = tuple(DEFAULT_SOURCE_CASES if source_cases is None else source_cases)
    if not sources or len(sources) != len(set(sources)):
        raise ValueError("Source cases must be nonempty and unique")
    if isinstance(horizon, bool) or not np.isfinite(horizon) or horizon <= 0:
        raise ValueError("Evaluation horizon must be positive and finite")
    entries = {str(row.get("id", row["path"])): row for row, _ in collection(data, "test")}
    cached = {}

    def load_source(case_id):
        if case_id not in entries:
            raise ValueError(f"Source case is absent from the held-out test split: {case_id}")
        if case_id not in cached:
            row = entries[case_id]
            path = case_path(data.parent, row["path"])
            if "file_sha256" in row and file_hash(path) != row["file_sha256"]:
                raise ValueError(f"Source file checksum differs from its collection record: {case_id}")
            case = Case.load(path)
            if row.get("case_hash") != str(case.arrays["case_hash"].item()):
                raise ValueError(f"Source case hash differs from its collection record: {case_id}")
            for name in ("young", "poisson", "density", "thickness"):
                if case.metadata["material"][name] != config["material"][name]:
                    raise ValueError(f"Source material differs from collection configuration: {case_id}")
            time = case.arrays["time"]
            dt = float(config["time"]["step"])
            if time[0] != 0 or not np.allclose(np.diff(time), dt, rtol=1e-9, atol=1e-12):
                raise ValueError(f"Source timestamps differ from the configured uniform timestep: {case_id}")
            cached[case_id] = row, path, case
        return cached[case_id]

    scenarios = []

    def append(case_id, scenario_id, category, mesh, load, duration, topology, load_novelty):
        row, path, case = load_source(case_id)
        original_duration = float(case.arrays["time"][-1])
        dt = float(config["time"]["step"])
        intervals = round(duration / dt)
        if intervals < 2 or not np.isclose(intervals * dt, duration, rtol=1e-12, atol=1e-14):
            raise ValueError("Scenario duration must be an integer multiple of the configured timestep")
        scenarios.append({
            "id": scenario_id,
            "category": category,
            "mesh": copy_mesh(mesh),
            "material": material,
            "load": dict(load),
            "duration": float(duration),
            "time_step": dt,
            "source_case": case_id,
            "source_case_hash": str(case.arrays["case_hash"].item()),
            "source_group": str(row["group"]),
            "source_path": path,
            "group": f"topology-{topology}" if category == "topology" else str(row["group"]),
            "topology": topology,
            "novelty": {
                "topology": "new_initial_cut_pattern" if category == "topology" else "unchanged_source_topology",
                "load": load_novelty,
                "horizon": "extended" if duration > original_duration else "original",
            },
            "novelty_flags": {
                "topology": category == "topology",
                "load": category == "load",
                "horizon": duration > original_duration,
            },
        })

    for case_id in sources:
        _, _, case = load_source(case_id)
        duration = float(case.arrays["time"][-1])
        if horizon <= duration:
            raise ValueError("Long-rollout horizon must exceed every selected source duration")
        load = {name: float(case.metadata["pull"][name]) for name in ("amplitude", "ramp_time")}
        append(case_id, f"force-{case_id}", "force_audit", case.mesh, load, duration, "source", "unchanged_source_load")
        append(case_id, f"long-{case_id}", "long_rollout", case.mesh, load, horizon, "source", "unchanged_source_load")

    baseline, variants = novel_loads(config["loads"])
    middle = config["mesh"]["nx"] // 2
    depths = (config["mesh"]["ny"] - 1, config["mesh"]["ny"])

    def baseline_source(depth):
        group = f"x{middle}-depth{depth}"
        for case_id, row in entries.items():
            if row["group"] != group:
                continue
            _, _, case = load_source(case_id)
            if all(float(case.metadata["pull"][key]) == value for key, value in baseline.items()):
                return case_id
        raise ValueError(f"The held-out collection lacks the nominal-load source: {group}/{baseline}")

    baselines = {depth: baseline_source(depth) for depth in depths}
    for case_id in baselines.values():
        _, _, case = load_source(case_id)
        append(case_id, f"control-{case_id}", "control", case.mesh, baseline,
               float(case.arrays["time"][-1]), "source", "unchanged_source_load")
    base_id = baselines[depths[0]]
    _, _, base_case = load_source(base_id)
    duration = float(base_case.arrays["time"][-1])
    for name, mesh in topology_meshes(config["mesh"]).items():
        append(base_id, f"topology-{name}", "topology", mesh, baseline, duration, name, "unchanged_source_load")
    for depth, case_id in baselines.items():
        _, _, case = load_source(case_id)
        duration = float(case.arrays["time"][-1])
        for name, load in variants:
            append(case_id, f"load-{name}-x{middle}-depth{depth}", "load", case.mesh,
                   load, duration, "source", name)
    return scenarios


def plan_summary(plan):
    return {
        "scenario_count": len(plan),
        "categories": dict(Counter(row["category"] for row in plan)),
        "scenarios": [{
            key: row[key] for key in (
                "id", "category", "load", "duration", "time_step", "source_case",
                "source_case_hash", "source_group", "group", "topology", "novelty", "novelty_flags",
            )
        } | {
            "source_path": str(row["source_path"]),
            "material": row["material"].metadata(),
            "nodes": len(row["mesh"].rest),
            "elements": len(row["mesh"].triangles),
            "cut_edges": row["mesh"].cut_edges.tolist(),
        } for row in plan],
    }
