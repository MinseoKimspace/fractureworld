import json
from numbers import Integral, Real
from pathlib import Path

import numpy as np

from .cut_event import experiment_plan
from .generation2d import validate_configuration
from .stress_scenarios import nominal_load


def integer_levels(values, allowed, label):
    values = list(values)
    if not values or any(isinstance(value, bool) or not isinstance(value, Integral) for value in values):
        raise ValueError(f"{label} must be nonempty integer levels")
    values = [int(value) for value in values]
    if len(set(values)) != len(values) or not set(values).issubset(allowed):
        raise ValueError(f"{label} must be unique configured levels")
    return values


def representative_levels(values):
    values = sorted(set(values))
    return values if len(values) <= 3 else [values[0], values[(len(values) - 1) // 2], values[-1]]


def sweep_plan(data, columns=None, load_indices=None, event_times=None, horizon=None):
    data = Path(data).resolve()
    manifest = json.loads(data.read_text(encoding="utf-8-sig"))
    config = validate_configuration(manifest["configuration"])
    nx, ny = config["mesh"]["nx"], config["mesh"]["ny"]
    configured_columns = [int(value) for value in config["cut_columns"]]
    if columns is None:
        center = min(configured_columns, key=lambda value: (abs(value - nx / 2), value))
        columns = sorted({min(configured_columns), center, max(configured_columns)})
    columns = integer_levels(columns, configured_columns, "Columns")
    loads = config["loads"]
    if load_indices is None:
        ramp = nominal_load(loads)["ramp_time"]
        candidates = [(index, load) for index, load in enumerate(loads) if load["ramp_time"] == ramp]
        amplitudes = representative_levels(load["amplitude"] for _, load in candidates)
        load_indices = [next(index for index, load in candidates if load["amplitude"] == amplitude)
                        for amplitude in amplitudes]
        load_selection = "representative_existing_amplitudes_at_nominal_ramp"
    else:
        load_selection = "explicit_existing_load_indices"
    load_indices = integer_levels(load_indices, range(len(loads)), "Load indices")
    selected_loads = [{"load_index": index, **{key: float(loads[index][key])
                      for key in ("amplitude", "ramp_time")}} for index in load_indices]
    partial_depths = [int(depth) for depth in config["cut_depths"]["test"] if depth < ny]
    if not partial_depths:
        raise ValueError("The held-out split must contain a partially cut source")
    depth = max(partial_depths)
    reference_ramp = min(load["ramp_time"] for load in selected_loads)
    default_times = event_times is None
    requested_times = [fraction * reference_ramp for fraction in (1 / 3, 1 / 2, 2 / 3)] if default_times else list(event_times)
    if not requested_times or any(isinstance(value, bool) or not isinstance(value, Real)
                                  or not np.isfinite(value) or value <= 0 for value in requested_times):
        raise ValueError("Event times must be nonempty, positive and finite")
    requested_times = [float(value) for value in requested_times]
    if len(set(requested_times)) != len(requested_times):
        raise ValueError("Event time levels must be unique")
    dt = float(config["time"]["step"])
    event_steps = [round(value / dt) for value in requested_times]
    canonical_times = [float(step * dt) for step in event_steps]
    if len(set(event_steps)) != len(event_steps):
        raise ValueError("Distinct event times collapse to the same output timestamp")
    for requested, canonical in zip(requested_times, canonical_times):
        if not default_times and not np.isclose(requested, canonical, rtol=1e-12, atol=1e-14):
            raise ValueError("Explicit event times must coincide with output timestamps")
        if canonical <= 0 or any(canonical >= load["ramp_time"] for load in selected_loads):
            raise ValueError("Every event must occur during every selected tensile loading ramp")
    factors = [(column, load) for column in columns for load in selected_loads]
    source_cases = [f"x{column}-depth{depth}-load{load['load_index']}" for column, load in factors]
    scenarios = []
    for requested, canonical, step in zip(requested_times, canonical_times, event_steps):
        batch = experiment_plan(data, source_cases=source_cases, event_time=canonical, horizon=horizon)
        for scenario, (column, load) in zip(batch, factors):
            expected_load = {key: load[key] for key in ("amplitude", "ramp_time")}
            if scenario["inputs"]["load"] != expected_load:
                raise ValueError(f"Source loading differs from its configured sweep level: {scenario['id']}")
            if scenario["inputs"]["source_group"] != f"x{column}-depth{depth}":
                raise ValueError(f"Source cut group differs from its configured sweep level: {scenario['id']}")
            actual = scenario["event_time"]
            scenario["id"] += f"-at-step-{step:04d}"
            scenario["sweep"] = {"column": column, **load, "event_step": step,
                                 "requested_event_time": requested, "event_time": actual,
                                 "event_time_adjustment": actual - requested}
            scenarios.append(scenario)
    if len({scenario["id"] for scenario in scenarios}) != len(scenarios):
        raise ValueError("Sweep scenario IDs must be unique")
    description = {
        "format": "cut-event-sweep-plan-v1", "source_dataset": str(data),
        "columns": columns, "load_indices": load_indices, "loads": selected_loads,
        "load_selection": load_selection, "initial_cut_depth": depth, "final_cut_depth": int(ny),
        "requested_event_times": requested_times, "event_times": canonical_times, "event_steps": event_steps,
        "event_time_adjustments": [actual - requested for actual, requested in zip(canonical_times, requested_times)],
        "event_time_selection": "fractions_of_minimum_selected_ramp" if default_times else "explicit_grid_aligned",
        "event_time_reference_ramp": reference_ramp,
        "event_time_rounding_policy": "nearest_output_step_ties_to_even" if default_times else "grid_validation_only",
        "event_time_fractions": [1 / 3, 1 / 2, 2 / 3] if default_times else None,
        "requested_count": len(columns) * len(load_indices) * len(requested_times),
        "scenario_count": len(scenarios),
        "factor_level_counts": {"columns": len(columns), "loads": len(load_indices), "event_times": len(requested_times)},
        "source_cases": source_cases,
        "horizons": sorted({scenario["inputs"]["duration"] for scenario in scenarios}),
        "scope": {"material": "existing_collection_material", "load": "selected_existing_load_conditions",
                  "geometry": "held_out_partial_cut_completed_to_separation",
                  "generalization": "new_prescribed_temporal_cut_conditions", "learned_fracture_criterion": False},
    }
    return scenarios, description
