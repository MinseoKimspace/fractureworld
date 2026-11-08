import json
from itertools import combinations
from pathlib import Path
from time import perf_counter

import numpy as np
import torch

from .cases import Case, collection
from .metrics2d import RMSE_CONVENTION, weighted_rmse
from .training2d import emit, graph_for, load_model


def synchronize(device):
    if str(device).startswith("cuda"):
        torch.cuda.synchronize(device)


def trajectory_metrics(case, position, velocity):
    position = np.asarray(position, dtype=np.float64)
    velocity = np.asarray(velocity, dtype=np.float64)
    frames = len(position)
    arrays = case.arrays
    mass = case.mass
    error_position = position[1:] - arrays["position"][1:frames]
    error_velocity = velocity[1:] - arrays["velocity"][1:frames]
    points = position[:, case.mesh.triangles]
    first, second = points[:, :, 1] - points[:, :, 0], points[:, :, 2] - points[:, :, 0]
    inverted = first[:, :, 0] * second[:, :, 1] - first[:, :, 1] * second[:, :, 0] <= 0
    pairs = [pair for vertex in np.unique(case.mesh.original_vertex)
             for pair in combinations(np.flatnonzero(case.mesh.original_vertex == vertex), 2)]
    opening_error = None
    if pairs:
        left, right = np.asarray(pairs).T
        prediction = position[1:, right] - position[1:, left]
        truth = arrays["position"][1:frames, right] - arrays["position"][1:frames, left]
        opening_error = float(np.sqrt(np.mean(np.sum((prediction - truth) ** 2, axis=-1))))
    kinetic = 0.5 * np.sum(velocity ** 2 * mass[None, :, None], axis=(1, 2))
    teacher_kinetic = 0.5 * np.sum(arrays["velocity"][:frames] ** 2 * mass[None, :, None], axis=(1, 2))
    return {
        "position_rmse": weighted_rmse(error_position, mass),
        "velocity_rmse": weighted_rmse(error_velocity, mass),
        "final_position_rmse": weighted_rmse(error_position[-1:], mass),
        "final_velocity_rmse": weighted_rmse(error_velocity[-1:], mass),
        "cut_opening_vector_rmse": opening_error,
        "cut_vertex_pair_count": len(pairs),
        "inverted_element_frames": int(inverted.sum()),
        "frames_with_inversion": int(np.any(inverted, axis=1).sum()),
        "orientation_valid": not bool(inverted.any()),
        "geometry_scope": "Positive signed triangle area only; overlaps/contact not checked",
        "kinetic_energy_change": float(kinetic[-1] - kinetic[0]),
        "teacher_kinetic_energy_change": float(teacher_kinetic[-1] - teacher_kinetic[0]),
        "energy_scope": "Kinetic energy only; external work and elastic energy are not evaluated",
        "rmse_convention": RMSE_CONVENTION,
    }


def check_checkpoint(checkpoint, rows, cases, split):
    required = {"dt", "training_case_hashes", "training_groups", "model_config"}
    if not required.issubset(checkpoint):
        raise ValueError("Checkpoint is missing timestep or training provenance")
    dt = float(checkpoint["dt"])
    if not np.isfinite(dt) or dt <= 0:
        raise ValueError("Checkpoint timestep must be positive and finite")
    for row, case in zip(rows, cases):
        if not np.allclose(np.diff(case.arrays["time"]), dt, rtol=1e-9, atol=1e-12):
            raise ValueError("Evaluation timestep does not match the checkpoint")
        if split != "train" and (
            str(case.arrays["case_hash"].item()) in checkpoint["training_case_hashes"]
            or str(row["group"]) in {str(group) for group in checkpoint["training_groups"]}
        ):
            raise ValueError("Evaluation case or cut configuration overlaps checkpoint training data")
    return dt


def rollout(model, case, dt, device):
    synchronize(device)
    started = perf_counter()
    graph = graph_for(case, device)
    position = torch.as_tensor(case.arrays["position"][0], dtype=torch.float32, device=device)
    velocity = torch.as_tensor(case.arrays["velocity"][0], dtype=torch.float32, device=device)
    boundary_position = torch.as_tensor(case.arrays["boundary_position"], dtype=torch.float32, device=device)
    boundary_velocity = torch.as_tensor(case.arrays["boundary_velocity"], dtype=torch.float32, device=device)
    positions, velocities, reactions = [position], [velocity], []
    failure = None
    with torch.inference_mode():
        for frame in range(1, len(case.arrays["time"])):
            try:
                next_position, next_velocity, force = model(
                    graph, position, velocity, boundary_position[frame], boundary_velocity[frame], dt
                )
                if not all(torch.isfinite(value).all().item() for value in (next_position, next_velocity, force)):
                    failure = f"Non-finite prediction at frame {frame}"
                    break
                reaction = graph.mass[:, None] * (next_velocity - velocity) - dt * force
                reactions.append(torch.where(graph.boundary_mask, reaction, torch.zeros_like(reaction)))
                positions.append(next_position)
                velocities.append(next_velocity)
                position, velocity = next_position, next_velocity
            except RuntimeError as error:
                failure = f"Prediction failed at frame {frame}: {error}"
                break
        arrays = {
            "position": torch.stack(positions).cpu().numpy().astype(np.float64),
            "velocity": torch.stack(velocities).cpu().numpy().astype(np.float64),
            "boundary_impulse": torch.stack(reactions).cpu().numpy() if reactions else np.empty((0, len(position), 2)),
        }
        arrays["position"][0] = case.arrays["position"][0]
        arrays["velocity"][0] = case.arrays["velocity"][0]
    synchronize(device)
    return arrays, perf_counter() - started, failure


def benchmark(data, checkpoints, output, split="test", device="cpu", case_limit=None, allow_untrained=False):
    if not checkpoints or split not in {"train", "validation", "test"}:
        raise ValueError("Provide checkpoints and a train, validation or test split")
    if case_limit is not None and case_limit < 1:
        raise ValueError("Case limit must be positive")
    if str(device).startswith("cuda") and not torch.cuda.is_available():
        raise ValueError("CUDA was requested but is unavailable")
    output = Path(output)
    if output.exists() and any(output.iterdir()):
        raise ValueError("Benchmark output must be a new or empty directory")
    selected = collection(data, split)
    selected = selected if case_limit is None else selected[:case_limit]
    rows, paths = zip(*selected)
    cases = [Case.load(path) for path in paths]
    methods = {}
    prepared = []
    for checkpoint_path in checkpoints:
        synchronize(device)
        started = perf_counter()
        model, checkpoint = load_model(checkpoint_path, device)
        synchronize(device)
        loading_seconds = perf_counter() - started
        dt = check_checkpoint(checkpoint, rows, cases, split)
        training_status = (
            "trained" if checkpoint.get("trained") is True else
            "untrained" if checkpoint.get("trained") is False else "unknown"
        )
        if training_status == "untrained" and not allow_untrained:
            raise ValueError("Untrained checkpoint requires explicit allow_untrained=True for pipeline checks")
        method = checkpoint["model_config"]["method"]
        if method in methods:
            raise ValueError(f"Repeated method label: {method}")
        methods[method] = {
            "checkpoint": str(Path(checkpoint_path).resolve()),
            "weight_loading_seconds": loading_seconds,
            "parameter_count": sum(parameter.numel() for parameter in model.parameters()),
            "training_status": training_status,
            "cases": [],
        }
        prepared.append((method, model, dt))
    output.mkdir(parents=True, exist_ok=True)
    for method, model, dt in prepared:
        synchronize(device)
        started = perf_counter()
        first = cases[0]
        graph = graph_for(first, device)
        values = [torch.as_tensor(first.arrays[key][index], dtype=torch.float32, device=device)
                  for key, index in (("position", 0), ("velocity", 0),
                                     ("boundary_position", 1), ("boundary_velocity", 1))]
        with torch.inference_mode():
            model(graph, *values, dt)
        synchronize(device)
        methods[method]["warmup_seconds"] = perf_counter() - started
        for index, (row, case) in enumerate(zip(rows, cases)):
            prediction, runtime, failure = rollout(model, case, dt, device)
            frames = len(prediction["position"])
            completed = failure is None and frames == len(case.arrays["time"])
            result = {
                "case": str(row.get("id", row["path"])), "group": str(row["group"]),
                "source_case_hash": str(case.arrays["case_hash"].item()),
                "status": "completed" if completed else "failed", "failure": failure,
                "requested_steps": len(case.arrays["time"]) - 1, "completed_steps": frames - 1,
                "completed_horizon": float(case.arrays["time"][frames - 1] - case.arrays["time"][0]),
                "requested_horizon": float(case.arrays["time"][-1] - case.arrays["time"][0]),
                "runtime_seconds": runtime,
            }
            if frames >= 2:
                result.update(trajectory_metrics(case, prediction["position"], prediction["velocity"]))
                target_reaction = case.arrays.get("boundary_impulse")
                if target_reaction is not None and target_reaction.shape != (
                    len(case.arrays["time"]) - 1, len(case.mesh.rest), 2
                ):
                    raise ValueError("Teacher boundary impulse array has an invalid shape")
                result["boundary_impulse_status"] = "unknown" if target_reaction is None else "available"
                result["boundary_impulse_rmse"] = None if target_reaction is None else float(
                    np.sqrt(np.mean(np.sum((prediction["boundary_impulse"] - target_reaction[:frames - 1]) ** 2, axis=(1, 2))))
                )
                arrays = {key: case.arrays[key].copy() for key in (
                    "rest", "triangles", "original_vertex", "cut_edges", "time", "boundary_mask",
                    "boundary_position", "boundary_velocity"
                ) if key in case.arrays}
                for key in ("time", "boundary_position", "boundary_velocity"):
                    arrays[key] = arrays[key][:frames]
                arrays.update(prediction)
                metadata = dict(case.metadata, method_label=method, source_revision="local-stage-a-v1",
                                training_status=methods[method]["training_status"],
                                source_url="https://arxiv.org/abs/2010.03409",
                                source_scope="Architecture reference; independent stage A implementation",
                                runtime_seconds=runtime, status=result["status"],
                                source_case_hash=result["source_case_hash"])
                saved = output / method / f"case-{index:04d}.npz"
                Case(arrays, metadata).save(saved)
                result["trajectory"] = str(saved.relative_to(output))
            methods[method]["cases"].append(result)
            emit("benchmark_case_completed", method=method, index=index + 1, cases=len(cases),
                 status=result["status"], completed_steps=frames - 1, runtime_seconds=runtime)
        results = methods[method]["cases"]
        completed = [result for result in results if result["status"] == "completed"]
        methods[method]["summary"] = {
            "attempted_cases": len(results), "completed_cases": len(completed),
            "failed_cases": len(results) - len(completed), "completion_rate": len(completed) / len(results),
            "runtime_seconds_total": sum(result["runtime_seconds"] for result in results),
            "mean_position_rmse_completed": None if not completed else float(np.mean([result["position_rmse"] for result in completed])),
            "mean_velocity_rmse_completed": None if not completed else float(np.mean([result["velocity_rmse"] for result in completed])),
            "orientation_valid_completed": sum(result["orientation_valid"] for result in completed),
        }
    untrained = any(method["training_status"] == "untrained" for method in methods.values())
    report = {"format": "cut-dynamics-a-v1", "split": split, "device": str(device), "methods": methods,
              "purpose": "untrained_pipeline_check" if untrained else "evaluation",
              "runtime_scope": "Graph construction, tensor preparation, autonomous prediction and output transfer; weight loading and warmup reported separately"}
    (output / "report.json").write_text(json.dumps(report, indent=2, allow_nan=False), encoding="utf-8")
    return report
