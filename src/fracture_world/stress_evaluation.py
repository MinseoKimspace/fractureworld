import json
from pathlib import Path
from time import perf_counter

import numpy as np
import torch

from .benchmark2d import check_checkpoint, checkpoint_hash, rollout, synchronize, trajectory_metrics
from .cases import Case, collection
from .diagnostic_metrics import free_motion_metrics, momentum_balance
from .force_reference2d import reference_solution, teacher_force_rollout
from .generation2d import file_hash
from .metrics2d import weighted_rmse
from .stress_scenarios import evaluation_plan, plan_summary
from .teacher2d import InternalForce, Pull
from .training2d import emit, graph_for, load_model


GROUPS = {"force": "force_audit", "long": "long_rollout", "topology": "topology", "load": "load"}


def evaluation_time(duration, dt):
    count = round(duration / dt)
    if min(duration, dt) <= 0 or count < 1 or not np.isclose(count * dt, duration, rtol=1e-12, atol=1e-14):
        raise ValueError("Evaluation duration must be a positive integer multiple of the timestep")
    return np.linspace(0.0, duration, count + 1)


def resample_prediction(prediction, factor, reference_time):
    if isinstance(factor, bool) or factor < 1 or int(factor) != factor:
        raise ValueError("Resampling factor must be a positive integer")
    intervals = (len(prediction["position"]) - 1) // factor
    frames = intervals + 1
    if frames > len(reference_time):
        raise ValueError("Resampled prediction exceeds the reference horizon")
    result = {
        key: np.asarray(prediction[key])[:frames * factor:factor].copy()
        for key in ("position", "velocity")
    }
    result["time"] = np.asarray(reference_time[:frames]).copy()
    if "time" in prediction and not np.allclose(
        np.asarray(prediction["time"])[:frames * factor:factor], result["time"], rtol=1e-12, atol=1e-14,
    ):
        raise ValueError("Substep timestamps do not align with the reference")
    for key in ("boundary_impulse", "internal_force"):
        if key in prediction:
            values = np.asarray(prediction[key])[:intervals * factor]
            values = values.reshape(intervals, factor, *values.shape[1:])
            result[key] = values.sum(axis=1) if key == "boundary_impulse" else values.mean(axis=1)
    return result


def case_at_time(case, time):
    time = np.asarray(time, dtype=np.float64)
    mask, boundary_position, boundary_velocity = Pull(**case.metadata["pull"]).boundary(case.mesh, time)
    arrays = {name: case.arrays[name].copy() for name in ("rest", "triangles", "original_vertex", "cut_edges")}
    arrays.update(
        time=time.copy(), boundary_mask=mask, boundary_position=boundary_position,
        boundary_velocity=boundary_velocity, position=boundary_position.copy(), velocity=boundary_velocity.copy(),
    )
    arrays["position"][0] = case.arrays["position"][0]
    arrays["velocity"][0] = case.arrays["velocity"][0]
    return Case(arrays, dict(case.metadata, state_scope="Only initial state and prescribed boundary inputs are meaningful"))


def save_prediction(reference, prediction, destination, method, identity):
    frames = len(prediction["position"])
    if frames < 2:
        return None
    arrays = {name: reference.arrays[name].copy() for name in (
        "rest", "triangles", "original_vertex", "cut_edges", "boundary_mask",
    )}
    arrays.update({name: reference.arrays[name][:frames].copy() for name in (
        "time", "boundary_position", "boundary_velocity",
    )})
    arrays.update({key: value.copy() for key, value in prediction.items() if key != "time"})
    Case(arrays, dict(
        reference.metadata, **identity, method_label=method,
        source_case_hash=str(reference.arrays["case_hash"].item()),
        force_semantics="Interval endpoint-average force when present",
    )).save(destination)
    return destination


def orientation_metrics(mesh, prediction):
    points = prediction["position"][:, mesh.triangles]
    first, second = points[:, :, 1] - points[:, :, 0], points[:, :, 2] - points[:, :, 0]
    area = first[:, :, 0] * second[:, :, 1] - first[:, :, 1] * second[:, :, 0]
    invalid = np.any(area <= 0, axis=1)
    indices = np.flatnonzero(invalid)
    return {
        "orientation_valid": not bool(invalid.any()),
        "frames_with_inversion": int(invalid.sum()),
        "first_invalid_orientation_time": None if not len(indices) else float(prediction["time"][indices[0]]),
        "minimum_signed_area": float(area.min() / 2),
        "geometry_scope": "All native integration states, including intermediate half-dt states; positive area only",
    }


def prediction_metrics(reference, prediction, status, native_geometry=None):
    frames = len(prediction["position"])
    result = dict(status)
    result.update(
        aligned_completed_steps=frames - 1,
        aligned_requested_steps=len(reference.arrays["time"]) - 1,
        aligned_completed_horizon=float(reference.arrays["time"][frames - 1]),
    )
    if frames < 2:
        result["metric_status"] = "unavailable_no_completed_interval"
        if native_geometry is not None:
            result.update(native_geometry, native_geometry=native_geometry)
        return result
    result["metric_status"] = "available"
    result.update(trajectory_metrics(reference, prediction["position"], prediction["velocity"]))
    free = free_motion_metrics(reference, prediction)
    result["free_motion"] = free
    result["momentum_balance"] = momentum_balance(reference, prediction)
    points = prediction["position"][:, reference.mesh.triangles]
    first, second = points[:, :, 1] - points[:, :, 0], points[:, :, 2] - points[:, :, 0]
    area = first[:, :, 0] * second[:, :, 1] - first[:, :, 1] * second[:, :, 0]
    invalid = np.flatnonzero(np.any(area <= 0, axis=1))
    result["first_invalid_orientation_time"] = None if not len(invalid) else float(reference.arrays["time"][invalid[0]])
    result["minimum_signed_area"] = float(area.min() / 2)
    difference = prediction["boundary_impulse"] - reference.arrays["boundary_impulse"][:frames - 1]
    result["boundary_impulse_rmse"] = float(np.sqrt(np.mean(np.sum(difference ** 2, axis=(1, 2)))))
    result["time_checkpoints"] = []
    for target in sorted({0.2, 0.5, float(reference.arrays["time"][-1])}):
        indices = np.flatnonzero(reference.arrays["time"][1:frames] <= target + 1e-14)
        if target > reference.arrays["time"][-1] or not len(indices):
            continue
        end = int(indices[-1])
        result["time_checkpoints"].append({
            "time": float(reference.arrays["time"][end + 1]),
            "requested_time": target,
            "reached": bool(reference.arrays["time"][frames - 1] >= target - 1e-14),
            "free_position_error": free["curves"]["position_rmse"][end],
            "free_velocity_error": free["curves"]["velocity_rmse"][end],
        })
    if native_geometry is not None:
        result["aligned_geometry"] = {
            key: result[key] for key in native_geometry if key in result
        }
        result.update(native_geometry, native_geometry=native_geometry)
    return result


def instantaneous_force_error(model, reference, material, prediction, device, samples=16):
    synchronize(device)
    started = perf_counter()
    graph = graph_for(reference, device)
    frames = np.unique(np.linspace(0, len(prediction["position"]) - 1, min(samples, len(prediction["position"])) , dtype=int))
    force = InternalForce(reference.mesh, material)
    predicted, truth = [], []
    with torch.inference_mode():
        for frame in frames:
            position = np.asarray(prediction["position"][frame])
            state = torch.as_tensor(position, dtype=torch.float32, device=device)
            predicted.append(model.instantaneous_force(graph, state).cpu().numpy().astype(np.float64))
            truth.append(force(state.cpu().numpy().astype(np.float64)))
    predicted, truth = np.stack(predicted), np.stack(truth)
    error = predicted - truth
    finite = np.isfinite(predicted).all() and np.isfinite(truth).all()
    if not finite:
        return {"status": "unknown_nonfinite_force", "frames": frames.tolist()}
    mass = reference.mass
    free = ~reference.arrays["boundary_mask"]
    acceleration = np.where(free[None], error / mass[None, :, None], 0)
    signal = float(np.sqrt(np.mean(np.sum(truth ** 2, axis=-1))))
    rmse = float(np.sqrt(np.mean(np.sum(error ** 2, axis=-1))))
    synchronize(device)
    return {
        "status": "available", "sample_count": len(frames), "frames": frames.tolist(),
        "force_rmse": rmse, "teacher_force_rms": signal,
        "force_relative_rmse": None if signal == 0 else rmse / signal,
        "normalization_status": "zero_signal" if signal == 0 else "available",
        "free_acceleration_rmse": weighted_rmse(acceleration[:, np.any(free, axis=1)], mass[np.any(free, axis=1)])
        if np.any(free) else None,
        "runtime_seconds": perf_counter() - started,
        "scope": "Both force functions evaluated at the same float32-rounded positions; teacher is diagnostic only",
    }


def write_report(output, report):
    temporary = output / "evaluation.tmp"
    temporary.write_text(json.dumps(report, indent=2, allow_nan=False), encoding="utf-8")
    temporary.replace(output / "evaluation.json")


def provider_difference(reference, neural, oracle):
    frames = min(len(neural["position"]), len(oracle["position"]))
    free = ~reference.arrays["boundary_mask"]
    active = np.any(free, axis=1)
    if frames < 2 or not active.any():
        return {"status": "unavailable"}
    result = {"status": "available", "compared_steps": frames - 1}
    for name in ("position", "velocity"):
        difference = np.where(free[None], neural[name][1:frames] - oracle[name][1:frames], 0)
        result["free_" + name + "_rmse"] = weighted_rmse(difference[:, active], reference.mass[active])
    result["scope"] = "Same timestep and integrator; force provider and float32/float64 precision differ; errors are not an additive decomposition"
    return result


def evaluation_summary(cases):
    result = {}
    for category in dict.fromkeys(row["category"] for row in cases):
        rows = [row for row in cases if row["category"] == category]
        methods = {}
        for name in dict.fromkeys(name for row in rows for name in row.get("methods", {})):
            attempted = [row["methods"][name] for row in rows if name in row.get("methods", {})]
            complete = [row for row in attempted if row["status"] == "completed" and row.get("metric_status") == "available"]
            position_errors = [row["free_motion"]["position_rmse"] for row in complete if row["free_motion"]["position_rmse"] is not None]
            velocity_errors = [row["free_motion"]["velocity_rmse"] for row in complete if row["free_motion"]["velocity_rmse"] is not None]
            methods[name] = {
                "attempted": len(attempted), "completed": len(complete),
                "orientation_valid": sum(row.get("orientation_valid") is True for row in complete),
                "runtime_seconds": sum(row["runtime_seconds"] for row in attempted),
                "mean_free_position_rmse": float(np.mean(position_errors)) if position_errors else None,
                "mean_free_velocity_rmse": float(np.mean(velocity_errors)) if velocity_errors else None,
                "max_case_free_position_rmse": max(position_errors, default=None),
                "max_case_free_velocity_rmse": max(velocity_errors, default=None),
                "position_metric_count": len(position_errors), "velocity_metric_count": len(velocity_errors),
            }
        result[category] = {
            "scenarios": len(rows),
            "reference_unavailable": sum(row["status"] in {"reference_failed", "reference_unverified"} for row in rows),
            "methods": methods,
            "scope": "Completed finite trajectories contribute to errors even when orientation is invalid; failures remain in attempt counts",
        }
    return result


def evaluate_stress(checkpoint, data, output, device="cpu", source_cases=None, horizon=1.0,
                    case_limit=None, groups=None, render_results=False):
    if str(device).startswith("cuda") and not torch.cuda.is_available():
        raise ValueError("CUDA was requested but is unavailable")
    groups = list(GROUPS) if groups is None else list(groups)
    if not groups or set(groups) - set(GROUPS) or len(groups) != len(set(groups)):
        raise ValueError("Evaluation groups must be unique force, long, topology or load names")
    if case_limit is not None and (isinstance(case_limit, bool) or int(case_limit) != case_limit or case_limit < 1):
        raise ValueError("Case limit must be a positive integer")
    output, checkpoint, data = Path(output).resolve(), Path(checkpoint).resolve(), Path(data).resolve()
    if output.exists() and (not output.is_dir() or any(output.iterdir())):
        raise ValueError("Evaluation output must be a new or empty directory")
    started = perf_counter()
    model, payload = load_model(checkpoint, device)
    if payload.get("trained") is not True or getattr(model, "architecture", None) != "equilibrium" or model.method != "stress":
        raise ValueError("Evaluation requires a trained equilibrium stress checkpoint")
    plan = evaluation_plan(data, source_cases, horizon)
    controls = [row for row in plan if row["category"] == "control"]
    categories = {GROUPS[group] for group in groups}
    plan = [row for row in plan if row["category"] in categories]
    if case_limit is not None:
        plan = plan[:case_limit]
    control_sources = {row["source_case"] for row in plan if row["category"] in {"topology", "load"}}
    plan.extend(row for row in controls if row["source_case"] in control_sources)
    if not plan:
        raise ValueError("No evaluation scenarios selected")
    source_rows = {row["id"]: row for row, _ in collection(data, "test")}
    originals = {row["source_case"]: Case.load(row["source_path"]) for row in plan}
    for row in plan:
        source = originals[row["source_case"]]
        check_checkpoint(payload, [source_rows[row["source_case"]]], [source], "test")
        if not np.array_equal(source.arrays["position"][0], source.mesh.rest) or np.any(source.arrays["velocity"][0] != 0):
            raise ValueError("Independent reference currently requires the source rest state and zero initial velocity")
        mask, boundary_position, boundary_velocity = Pull(**source.metadata["pull"]).boundary(source.mesh, source.arrays["time"])
        if not all(np.array_equal(actual, expected) for actual, expected in (
            (source.arrays["boundary_mask"], mask),
            (source.arrays["boundary_position"], boundary_position),
            (source.arrays["boundary_velocity"], boundary_velocity),
        )):
            raise ValueError("Source boundary arrays differ from their declared tensile loading")
        if not np.isclose(row["time_step"], payload["dt"], rtol=1e-9, atol=1e-12):
            raise ValueError("Scenario timestep differs from checkpoint training")
    protected = {checkpoint, data, *(Path(row["source_path"]).resolve() for row in plan)}
    hashes = {path: file_hash(path) for path in protected}
    report = {
        "format": "cut-dynamics-stress-evaluation-v1", "status": "in_progress",
        "checkpoint": str(checkpoint), "checkpoint_sha256": checkpoint_hash(checkpoint),
        "data": str(data), "device": str(device), "training_executed": False,
        "model_configuration": model.configuration(), "trained_dt": float(payload["dt"]),
        "plan": plan_summary(plan), "cases": [], "inputs_unchanged": None,
        "scope": "Frozen-model evaluation; prescribed initial cuts and tensile loading; no contact or fracture evolution",
        "comparison_scope": "Oracle uses FElupe on CPU float64; neural rollout uses the requested device float32; timings are not a hardware-matched speed benchmark",
    }
    output.mkdir(parents=True, exist_ok=True)
    write_report(output, report)
    try:
        for index, scenario in enumerate(plan):
            row = {key: value for key, value in plan_summary([scenario])["scenarios"][0].items()}
            row.update(status="in_progress", methods={})
            report["cases"].append(row)
            emit("stress_evaluation_case_started", index=index + 1, cases=len(plan), case=scenario["id"], category=scenario["category"])
            time = evaluation_time(scenario["duration"], scenario["time_step"])
            try:
                emit("stress_evaluation_phase", case=scenario["id"], phase="independent_reference_started", device="cpu")
                reference, reference_audit = reference_solution(scenario["mesh"], scenario["material"], time, scenario["load"])
            except (RuntimeError, ValueError, FloatingPointError, OverflowError) as error:
                row.update(status="reference_failed", failure=f"{type(error).__name__}: {error}")
                write_report(output, report)
                emit("stress_evaluation_case_completed", case=scenario["id"], status=row["status"])
                continue
            row["reference_audit"] = reference_audit
            reference.metadata.update(
                method_label="reference_DOP853", source_case=scenario["source_case"],
                original_source_case_hash=scenario["source_case_hash"],
            )
            directory = output / scenario["id"]
            directory.mkdir()
            reference_path = directory / "reference.npz"
            reference.save(reference_path)
            row["reference_trajectory"] = str(reference_path.relative_to(output))
            row["evaluation_case_hash"] = str(reference.arrays["case_hash"].item())
            if reference_audit.get("converged") is not True:
                row["status"] = "reference_unverified"
                write_report(output, report)
                emit("stress_evaluation_case_completed", case=scenario["id"], status=row["status"])
                continue
            if scenario["category"] in {"force_audit", "long_rollout", "control"}:
                source = originals[scenario["source_case"]]
                frames = len(source.arrays["time"])
                if not np.allclose(reference.arrays["time"][:frames], source.arrays["time"], rtol=1e-12, atol=1e-14):
                    raise ValueError("Independent reference does not preserve the original source timestamps")
                row["source_reference_agreement"] = free_motion_metrics(source, {
                    "time": source.arrays["time"], "position": reference.arrays["position"][:frames],
                    "velocity": reference.arrays["velocity"][:frames],
                })
            saved_predictions = {}
            factors = (1, 2) if scenario["category"] in {"force_audit", "long_rollout"} else (1,)
            for provider in ("neural", "teacher"):
                for factor in factors:
                    name = f"{provider}_verlet" + ("_half_dt" if factor == 2 else "")
                    emit("stress_evaluation_phase", case=scenario["id"], phase=name + "_started")
                    current_time = evaluation_time(scenario["duration"], scenario["time_step"] / factor)
                    if provider == "neural":
                        inputs = case_at_time(reference, current_time)
                        prediction, runtime, failure = rollout(model, inputs, scenario["time_step"] / factor, device)
                        prediction["time"] = current_time[:len(prediction["position"])]
                        status = {
                            "status": "completed" if failure is None and len(prediction["position"]) == len(current_time) else "failed",
                            "failure": failure, "runtime_seconds": runtime, "device": str(device), "precision": "float32",
                            "completed_steps": len(prediction["position"]) - 1, "requested_steps": len(current_time) - 1,
                            "force_provider": "frozen_neural_model", "teacher_calls_in_rollout": 0,
                        }
                    else:
                        prediction, status = teacher_force_rollout(reference, current_time)
                    native_geometry = orientation_metrics(reference.mesh, prediction)
                    prediction = resample_prediction(prediction, factor, time)
                    result = prediction_metrics(reference, prediction, status, native_geometry)
                    result.update(actual_dt=scenario["time_step"] / factor, trained_dt=float(payload["dt"]),
                                  temporal_resampling="Substep impulses summed before coarse-grid comparison")
                    destination = save_prediction(reference, prediction, directory / f"{name}.npz", name, {
                        "status": result["status"], "checkpoint_sha256": hashes[checkpoint] if provider == "neural" else None,
                        "training_status": "trained" if provider == "neural" else "numerical_reference",
                    })
                    if destination is not None:
                        result["trajectory"] = str(destination.relative_to(output))
                    row["methods"][name] = result
                    saved_predictions[name] = prediction
                    write_report(output, report)
                    emit("stress_evaluation_phase", case=scenario["id"], phase=name + "_completed", status=result["status"], runtime_seconds=result["runtime_seconds"])
            row["force_provider_difference"] = {
                "dt": provider_difference(reference, saved_predictions["neural_verlet"], saved_predictions["teacher_verlet"]),
            }
            if 2 in factors:
                row["force_provider_difference"]["half_dt"] = provider_difference(
                    reference, saved_predictions["neural_verlet_half_dt"], saved_predictions["teacher_verlet_half_dt"],
                )
            if scenario["category"] == "force_audit":
                emit("stress_evaluation_phase", case=scenario["id"], phase="same_state_force_diagnostics")
                row["instantaneous_force_diagnostics"] = {
                    "on_reference_states": instantaneous_force_error(model, reference, scenario["material"], reference.arrays, device),
                    "on_neural_states": instantaneous_force_error(model, reference, scenario["material"], saved_predictions["neural_verlet"], device),
                }
            row["status"] = "completed"
            if render_results:
                from .render2d import render

                names = ("neural_verlet", "teacher_verlet")
                if all(row["methods"][name].get("trajectory") for name in names):
                    paths = [reference_path, *(output / row["methods"][name]["trajectory"] for name in names)]
                    try:
                        for suffix in ("gif", "png"):
                            render(paths, directory / f"comparison.{suffix}", fps=30)
                        row["render"] = str((directory / "comparison.gif").relative_to(output))
                        row["render_scope"] = "Common finite prefix; method completion and orientation reported separately"
                    except (RuntimeError, ValueError, OSError) as error:
                        row["render_failure"] = str(error)
            write_report(output, report)
            emit("stress_evaluation_case_completed", case=scenario["id"], status=row["status"])
        report["status"] = "completed"
    except BaseException as error:
        report["status"] = "interrupted" if isinstance(error, KeyboardInterrupt) else "failed"
        report["failure"] = f"{type(error).__name__}: {error}"
        raise
    finally:
        report["runtime_seconds"] = perf_counter() - started
        report["summary"] = evaluation_summary(report["cases"])
        changed = [str(path) for path, digest in hashes.items() if not path.is_file() or file_hash(path) != digest]
        report["inputs_unchanged"] = not changed
        if changed:
            report["changed_inputs"] = changed
            report["status"] = "failed"
        write_report(output, report)
        if changed:
            raise RuntimeError("Protected evaluation inputs changed during execution")
    return {
        "status": report["status"], "report": str(output / "evaluation.json"),
        "training_executed": False, "inputs_unchanged": report["inputs_unchanged"],
        "scenarios": len(plan), "reference_failed": sum(row["status"] == "reference_failed" for row in report["cases"]),
        "reference_unverified": sum(row["status"] == "reference_unverified" for row in report["cases"]),
        "neural_completed": sum(row.get("methods", {}).get("neural_verlet", {}).get("status") == "completed" for row in report["cases"]),
        "neural_orientation_valid": sum(row.get("methods", {}).get("neural_verlet", {}).get("orientation_valid") is True for row in report["cases"]),
        "summary": report["summary"],
    }
