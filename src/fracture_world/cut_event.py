import hashlib
import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from .cases import Case, collection
from .generation2d import case_path, cut_mesh, file_hash, validate_configuration
from .stress_scenarios import nominal_load
from .topology import lift_state, split_edges, triangle_areas


@dataclass
class CutSequence:
    before: Case
    after: Case
    metadata: dict

    def __post_init__(self):
        if self.before.arrays["time"][-1] != self.after.arrays["time"][0]:
            raise ValueError("Cut segments must meet at the same physical timestamp")
        if self.before.metadata["material"] != self.after.metadata["material"]:
            raise ValueError("A prescribed cut cannot change the material law")
        if self.before.metadata["pull"] != self.after.metadata["pull"]:
            raise ValueError("A prescribed cut cannot restart or change the tensile loading")
        position, velocity = lift_state(
            self.before.mesh, self.after.mesh,
            self.before.arrays["position"][-1], self.before.arrays["velocity"][-1],
        )
        for value, name in ((position, "position"), (velocity, "velocity")):
            if not np.array_equal(value, self.after.arrays[name][0]):
                raise ValueError(f"Cut sequence must preserve event {name}")
        self.metadata = dict(self.metadata, format="cut-event-sequence-v1")

    @property
    def time(self):
        return np.concatenate((self.before.arrays["time"][:-1], self.after.arrays["time"]))

    def frame(self, index):
        transition = len(self.before.arrays["time"]) - 1
        case, local = (self.before, index) if index < transition else (self.after, index - transition)
        return case, local

    def corners(self, field):
        return np.concatenate((
            self.before.arrays[field][:-1, self.before.mesh.triangles],
            self.after.arrays[field][:, self.after.mesh.triangles],
        ))

    def save(self, path):
        values = {}
        for prefix, case in (("before", self.before), ("after", self.after)):
            values.update({f"{prefix}_{key}": value for key, value in case.arrays.items()})
            values[f"{prefix}_metadata_json"] = json.dumps(case.metadata, sort_keys=True)
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(path, **values, metadata_json=json.dumps(self.metadata, sort_keys=True))

    @classmethod
    def load(cls, path):
        with np.load(path, allow_pickle=False) as archive:
            metadata = json.loads(str(archive["metadata_json"].item()))
            if metadata.get("format") != "cut-event-sequence-v1":
                raise ValueError("Expected a dynamic cut sequence")
            cases = []
            for prefix in ("before", "after"):
                arrays = {key[len(prefix) + 1:]: archive[key] for key in archive.files
                          if key.startswith(prefix + "_") and key != f"{prefix}_metadata_json"}
                details = json.loads(str(archive[f"{prefix}_metadata_json"].item()))
                cases.append(Case(arrays, details))
        return cls(*cases, metadata)


def load_trajectory(path):
    with np.load(path, allow_pickle=False) as archive:
        metadata = json.loads(str(archive["metadata_json"].item()))
    return CutSequence.load(path) if metadata.get("format") == "cut-event-sequence-v1" else Case.load(path)


def corner_weights(case):
    material = case.metadata["material"]
    return np.repeat(triangle_areas(case.mesh) * material["density"] * material["thickness"] / 3, 3)


def experiment_plan(data, source_cases=None, event_time=0.03, horizon=None):
    data = Path(data).resolve()
    manifest = json.loads(data.read_text(encoding="utf-8-sig"))
    config = validate_configuration(manifest["configuration"])
    rows = {str(row.get("id", row["path"])): row for row, _ in collection(data, "test")}
    if source_cases is None:
        load = nominal_load(config["loads"])
        load_index = next(index for index, value in enumerate(config["loads"]) if value == load)
        column = min(config["cut_columns"], key=lambda value: (abs(value - config["mesh"]["nx"] / 2), value))
        depths = [depth for depth in config["cut_depths"]["test"] if depth < config["mesh"]["ny"]]
        if not depths:
            raise ValueError("The held-out split must contain a partially cut source")
        source_cases = [f"x{column}-depth{max(depths)}-load{load_index}"]
    if not source_cases or len(set(source_cases)) != len(source_cases):
        raise ValueError("Source cases must be nonempty and unique")
    if isinstance(event_time, bool) or not np.isfinite(event_time) or event_time <= 0:
        raise ValueError("Cut event time must be positive and finite")
    if horizon is not None and (isinstance(horizon, bool) or not np.isfinite(horizon) or horizon <= event_time):
        raise ValueError("The evaluation horizon must exceed the cut event time")
    scenarios = []
    for identity in source_cases:
        if identity not in rows:
            raise ValueError(f"Source case is absent from the held-out split: {identity}")
        row = rows[identity]
        path = case_path(data.parent, row["path"])
        if row.get("file_sha256") is not None and file_hash(path) != row["file_sha256"]:
            raise ValueError(f"Source file checksum differs: {identity}")
        case = Case.load(path)
        source_hash = str(case.arrays["case_hash"].item())
        if row.get("case_hash") is not None and row["case_hash"] != source_hash:
            raise ValueError(f"Source input hash differs: {identity}")
        if "column" in row and "depth" in row:
            column, depth = int(row["column"]), int(row["depth"])
        else:
            import re

            match = re.fullmatch(r"x(\d+)-depth(\d+)", str(row["group"]))
            if match is None:
                raise ValueError("The cut event requires an initial vertical notch")
            column, depth = map(int, match.groups())
        if not 0 < depth < config["mesh"]["ny"]:
            raise ValueError("The source must be partially cut, with a remaining ligament")
        expected = cut_mesh(config, column, depth)
        for name in ("rest", "triangles", "original_vertex", "cut_edges"):
            if not np.array_equal(getattr(case.mesh, name), getattr(expected, name)):
                raise ValueError(f"Source mesh differs from its declared cut configuration: {name}")
        for name, value in config["material"].items():
            if case.metadata["material"][name] != value:
                raise ValueError("Source material differs from the collection configuration")
        vertices = np.arange(depth, config["mesh"]["ny"] + 1) * (config["mesh"]["nx"] + 1) + column
        added = np.column_stack((vertices[:-1], vertices[1:]))
        final_mesh = split_edges(case.mesh, added)
        dt = float(config["time"]["step"])
        duration = float(case.arrays["time"][-1] if horizon is None else horizon)
        intervals, event_index = round(duration / dt), round(event_time / dt)
        if intervals < 2 or not np.isclose(intervals * dt, duration, rtol=1e-12, atol=1e-14):
            raise ValueError("Horizon must be an integer multiple of the dataset timestep")
        time = np.linspace(0, duration, intervals + 1)
        if not 0 < event_index < intervals or not np.isclose(time[event_index], event_time, rtol=1e-12, atol=1e-14):
            raise ValueError("Cut event must coincide with an interior output timestamp")
        if case.arrays["time"][0] != 0 or not np.allclose(np.diff(case.arrays["time"]), dt, rtol=1e-9, atol=1e-12):
            raise ValueError("Source timestep differs from the collection configuration")
        load = {name: float(case.metadata["pull"][name]) for name in ("amplitude", "ramp_time")}
        if not event_time < load["ramp_time"]:
            raise ValueError("The first experiment cuts during the tensile loading ramp")
        inputs = {
            "source_case": identity, "source_case_hash": source_hash,
            "source_group": str(row["group"]), "event_time": float(time[event_index]),
            "duration": duration, "output_step": dt, "added_cut_edges": added.tolist(),
            "initial_cut_depth": depth, "final_cut_depth": config["mesh"]["ny"],
            "load": load, "material": case.metadata["material"],
        }
        fingerprint = hashlib.sha256(json.dumps(inputs, sort_keys=True, allow_nan=False).encode()).hexdigest()
        scenarios.append({"id": "event-" + identity, "source": case, "source_path": path,
                          "final_mesh": final_mesh, "time": time, "event_time": float(time[event_index]),
                          "experiment_hash": fingerprint, "inputs": inputs})
    return scenarios
