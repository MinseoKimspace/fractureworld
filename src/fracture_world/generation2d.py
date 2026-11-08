import copy
import json
from pathlib import Path

import numpy as np

from .teacher2d import Material, simulate
from .topology import rectangle, split_edges


def configuration(path):
    config = json.loads(Path(path).read_text(encoding="utf-8"))
    rectangle(**config["mesh"])
    Material(**config["material"])
    nx, ny = config["mesh"]["nx"], config["mesh"]["ny"]
    columns = config["cut_columns"]
    if not columns or len(set(columns)) != len(columns):
        raise ValueError("Cut columns must be nonempty and unique")
    if any(int(column) != column or not 0 < column < nx for column in columns):
        raise ValueError("Cut columns must lie inside the plate")
    seen = set()
    for split in ("train", "validation", "test"):
        depths = config["cut_depths"][split]
        if not depths or any(int(depth) != depth or not 0 < depth <= ny for depth in depths):
            raise ValueError("Every split requires valid positive cut depths")
        if len(set(depths)) != len(depths) or seen.intersection(depths):
            raise ValueError("Cut depth families must be disjoint across splits")
        seen.update(depths)
    if not config["loads"]:
        raise ValueError("At least one tensile loading is required")
    for load in config["loads"]:
        if load["amplitude"] < 0 or load["ramp_time"] <= 0:
            raise ValueError("Invalid tensile loading")
    times(config)
    return config


def times(config):
    step, duration = config["time"]["step"], config["time"]["duration"]
    if not np.isfinite([step, duration]).all() or min(step, duration) <= 0:
        raise ValueError("Time step and duration must be positive and finite")
    intervals = round(duration / step)
    if intervals < 2 or not np.isclose(intervals * step, duration, rtol=1e-12, atol=1e-14):
        raise ValueError("Duration must be an integer multiple of the time step")
    return np.linspace(0.0, duration, intervals + 1)


def cut_mesh(config, column, depth):
    mesh = rectangle(**config["mesh"])
    if depth == 0:
        return mesh
    nx, ny = config["mesh"]["nx"], config["mesh"]["ny"]
    if not 0 < column < nx or not 0 < depth <= ny:
        raise ValueError("Cut must follow an interior column from the bottom boundary")
    vertices = np.arange(depth + 1, dtype=np.int64) * (nx + 1) + column
    return split_edges(mesh, np.column_stack((vertices[:-1], vertices[1:])))


def smoke_configuration(config):
    result = copy.deepcopy(config)
    result["mesh"].update(nx=4, ny=4)
    result["cut_columns"] = [1]
    result["cut_depths"] = {"train": [1], "validation": [2], "test": [3, 4]}
    result["loads"] = [result["loads"][0]]
    return result


def generate(config, output):
    output = Path(output)
    if output.exists() and any(output.iterdir()):
        raise ValueError("Generation output must be a new or empty directory")
    output.mkdir(parents=True, exist_ok=True)
    manifest = {
        "format": "cut-dynamics-a-v1", "status": "in_progress", "cases": [],
        "configuration": config, "split_scope": "Entire initial cut-depth families held out",
        "physics_scope": "Trial StVK plane strain; prescribed initial cuts and tension; no contact or fracture evolution",
    }
    plan = []
    for split, depths in config["cut_depths"].items():
        for column in config["cut_columns"]:
            for depth in depths:
                plan.append((split, column, depth))
    if config.get("include_intact_training", False):
        plan.append(("train", 0, 0))
    total = len(plan) * len(config["loads"])
    material = Material(**config["material"])
    time = times(config)
    for split, column, depth in plan:
        mesh = cut_mesh(config, column, depth)
        group = f"x{column}-depth{depth}"
        for load_index, load in enumerate(config["loads"]):
            case_id = f"{group}-load{load_index}"
            case = simulate(mesh, material, time, **load, **config["integration"])
            case.metadata.update(split=split, cut_group=group, cut_column=column,
                                 cut_depth=depth, configuration_units="dimensionless_trial")
            path = Path("cases") / f"{case_id}.npz"
            case.save(output / path)
            manifest["cases"].append({
                "id": case_id, "group": group, "split": split,
                "path": path.as_posix(), "case_hash": str(case.arrays["case_hash"].item()),
            })
            temporary = output / "collection.tmp"
            temporary.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
            temporary.replace(output / "collection.json")
            print(json.dumps({"event": "teacher_case_completed", "case": case_id,
                              "completed": len(manifest["cases"]), "total": total,
                              "split": split, "runtime_seconds": case.metadata["runtime_seconds"]}), flush=True)
    manifest["status"] = "completed"
    (output / "collection.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    return {"collection": str(output / "collection.json"), "cases": total,
            "splits": {split: sum(row["split"] == split for row in manifest["cases"])
                       for split in ("train", "validation", "test")}}
