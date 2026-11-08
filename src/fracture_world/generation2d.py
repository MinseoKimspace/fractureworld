import copy
import hashlib
import json
from pathlib import Path

import numpy as np

from .cases import Case, input_hash
from .teacher2d import Material, Pull, simulate
from .topology import rectangle, split_edges


def configuration(path):
    config = json.loads(Path(path).read_text(encoding="utf-8"))
    return validate_configuration(config)


def validate_configuration(config):
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
    combinations = set()
    for load in config["loads"]:
        values = (load["amplitude"], load["ramp_time"])
        if not np.isfinite(values).all() or values[0] < 0 or values[1] <= 0:
            raise ValueError("Invalid tensile loading")
        if values in combinations:
            raise ValueError("Tensile loading combinations must be unique")
        combinations.add(values)
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


def generation_plan(config):
    validate_configuration(config)
    geometries = []
    for split in ("train", "validation", "test"):
        depths = config["cut_depths"][split]
        for column in config["cut_columns"]:
            for depth in depths:
                geometries.append((split, column, depth))
    if config.get("include_intact_training", False):
        geometries.append(("train", 0, 0))
    plan = []
    for split, column, depth in geometries:
        group = f"x{column}-depth{depth}"
        for load_index, load in enumerate(config["loads"]):
            case_id = f"{group}-load{load_index}"
            plan.append({
                "id": case_id, "group": group, "split": split,
                "column": column, "depth": depth, "load_index": load_index,
                "load": dict(load), "path": f"cases/{case_id}.npz",
            })
    if len({item["id"] for item in plan}) != len(plan):
        raise ValueError("Generation plan contains repeated case IDs")
    return plan


def generation_summary(config):
    plan = generation_plan(config)
    time = times(config)
    return {
        "cases": len(plan),
        "splits": {split: sum(item["split"] == split for item in plan)
                   for split in ("train", "validation", "test")},
        "cut_groups": len({item["group"] for item in plan}),
        "cut_groups_by_split": {
            split: len({item["group"] for item in plan if item["split"] == split})
            for split in ("train", "validation", "test")
        },
        "load_combinations": len(config["loads"]),
        "states_per_case": len(time), "steps_per_case": len(time) - 1,
        "duration": float(time[-1] - time[0]),
    }


def generation_configuration(config):
    keys = ("mesh", "material", "time", "cut_columns", "cut_depths", "loads", "integration")
    recipe = {key: config[key] for key in keys}
    recipe["include_intact_training"] = config.get("include_intact_training", False)
    return json.dumps(recipe, sort_keys=True, allow_nan=False)


def save_manifest(output, manifest):
    temporary = output / "collection.tmp"
    temporary.write_text(json.dumps(manifest, indent=2, allow_nan=False), encoding="utf-8")
    temporary.replace(output / "collection.json")


def case_path(output, relative):
    relative = Path(relative)
    if relative.is_absolute() or ".." in relative.parts:
        raise ValueError("Case paths must stay inside the generation output")
    destination = (output / relative).resolve()
    if not destination.is_relative_to(output.resolve()):
        raise ValueError("Case paths must stay inside the generation output")
    return destination


def file_hash(path):
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def expected_case_hash(mesh, material, time, load):
    mask, boundary_position, boundary_velocity = Pull(**load).boundary(mesh, time)
    return input_hash({
        "rest": mesh.rest, "triangles": mesh.triangles,
        "original_vertex": mesh.original_vertex, "cut_edges": mesh.cut_edges,
        "time": time, "boundary_mask": mask,
        "boundary_position": boundary_position, "boundary_velocity": boundary_velocity,
        "position": boundary_position[:1], "velocity": boundary_velocity[:1],
    }, material.metadata())


def resume_manifest(config, output, plan, material, time):
    path = output / "collection.json"
    if not path.is_file():
        raise ValueError("Nonempty resume output requires collection.json")
    manifest = json.loads(path.read_text(encoding="utf-8"))
    if manifest.get("format") != "cut-dynamics-a-v1" or manifest.get("status") not in {"in_progress", "completed"}:
        raise ValueError("Resume manifest has an unsupported format or status")
    if generation_configuration(manifest["configuration"]) != generation_configuration(config):
        raise ValueError("Resume generation configuration differs from the requested physical inputs")
    planned = {item["id"]: item for item in plan}
    verified = set()
    meshes = {}
    for row in manifest["cases"]:
        case_id = row.get("id")
        if case_id not in planned or case_id in verified:
            raise ValueError("Resume manifest contains an unknown or duplicate case ID")
        item = planned[case_id]
        if any(row.get(key) != item[key] for key in ("group", "split", "path")):
            raise ValueError("Resume manifest case identity does not match the generation plan")
        path = case_path(output, row["path"])
        if not path.is_file():
            raise ValueError(f"Recorded completed case is missing: {case_id}")
        if "file_sha256" in row and file_hash(path) != row["file_sha256"]:
            raise ValueError(f"Recorded completed case checksum differs: {case_id}")
        case = Case.load(path)
        key = (item["column"], item["depth"])
        if key not in meshes:
            meshes[key] = cut_mesh(config, *key)
        expected = expected_case_hash(meshes[key], material, time, item["load"])
        if row.get("case_hash") != expected or str(case.arrays["case_hash"].item()) != expected:
            raise ValueError(f"Recorded completed case inputs differ: {case_id}")
        verified.add(case_id)
    if manifest["status"] == "completed" and len(verified) != len(plan):
        raise ValueError("Completed resume manifest is missing planned cases")
    return manifest, verified


def generate(config, output, resume=False):
    plan = generation_plan(config)
    output = Path(output)
    material = Material(**config["material"])
    time = times(config)
    nonempty = output.exists() and any(output.iterdir())
    if nonempty and not resume:
        raise ValueError("Generation output must be a new or empty directory; use resume for verified existing data")
    if nonempty:
        manifest, verified = resume_manifest(config, output, plan, material, time)
    else:
        output.mkdir(parents=True, exist_ok=True)
        manifest = {
            "format": "cut-dynamics-a-v1", "status": "in_progress", "cases": [],
            "configuration": copy.deepcopy(config), "split_scope": "Entire initial cut-depth families held out",
            "physics_scope": "Trial StVK plane strain; prescribed initial cuts and tension; no contact or fracture evolution",
        }
        verified = set()
    total, skipped = len(plan), len(verified)
    meshes = {}
    if skipped:
        print(json.dumps({"event": "teacher_resume_verified", "completed": skipped, "total": total}), flush=True)
    manifest["status"] = "in_progress"
    save_manifest(output, manifest)
    for item in plan:
        if item["id"] in verified:
            continue
        path = case_path(output, item["path"])
        if path.exists():
            raise ValueError(f"Unrecorded case file exists and will not be overwritten: {item['id']}")
        key = (item["column"], item["depth"])
        if key not in meshes:
            meshes[key] = cut_mesh(config, *key)
        mesh = meshes[key]
        case = simulate(mesh, material, time, **item["load"], **config["integration"])
        case.metadata.update(split=item["split"], cut_group=item["group"], cut_column=item["column"],
                             cut_depth=item["depth"], configuration_units="dimensionless_trial")
        expected = expected_case_hash(mesh, material, time, item["load"])
        if str(case.arrays["case_hash"].item()) != expected:
            raise ValueError("Generated teacher inputs do not match the requested case")
        case.save(path)
        manifest["cases"].append({
            key: item[key] for key in ("id", "group", "split", "path")
        } | {"case_hash": expected, "file_sha256": file_hash(path)})
        save_manifest(output, manifest)
        print(json.dumps({"event": "teacher_case_completed", "case": item["id"],
                          "completed": len(manifest["cases"]), "total": total,
                          "split": item["split"], "runtime_seconds": case.metadata["runtime_seconds"]}), flush=True)
    order = {item["id"]: index for index, item in enumerate(plan)}
    manifest["cases"].sort(key=lambda row: order[row["id"]])
    manifest["status"] = "completed"
    save_manifest(output, manifest)
    return {"collection": str(output / "collection.json"), "cases": total,
            "generated_cases": total - skipped, "skipped_cases": skipped,
            "splits": {split: sum(row["split"] == split for row in manifest["cases"])
                       for split in ("train", "validation", "test")}}
