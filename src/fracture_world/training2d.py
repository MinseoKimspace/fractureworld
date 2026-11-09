import json
import random
from pathlib import Path
from time import perf_counter

import numpy as np
import torch
from torch_geometric.data import Batch

from .cases import Case, collection
from .neural import build_graph, create_model


def emit(event, **fields):
    print(json.dumps({"event": event, **fields}, allow_nan=False), flush=True)


def graph_for(case, device="cpu"):
    material = case.metadata["material"]
    return build_graph(
        case.mesh, case.mass, case.arrays["boundary_mask"],
        material=np.array([material["young"], material["poisson"], material["density"]]),
        thickness=material["thickness"], device=device,
    )


def load_model(path, device="cpu"):
    checkpoint = torch.load(path, map_location=device, weights_only=True)
    if checkpoint.get("format") not in {"cut-dynamics-a-v1", "cut-dynamics-a-v2"}:
        raise ValueError("Checkpoint is not a stage A dynamics model")
    if checkpoint["format"] == "cut-dynamics-a-v2" and checkpoint["model_config"].get("architecture") != "equilibrium":
        raise ValueError("Version 2 checkpoints must declare the equilibrium architecture")
    model = create_model(checkpoint["model_config"]).to(device)
    model.load_state_dict(checkpoint["state_dict"])
    model.eval()
    return model, checkpoint


def reaction_configuration(training, validation):
    cases = training + validation
    available = ["boundary_impulse" in case.arrays for case in cases]
    if any(available) and not all(available):
        raise ValueError("Boundary impulse supervision must be available for every training and validation case, or absent for all")
    if not any(available):
        return {"enabled": False, "scale": None, "scale_source": "unavailable"}
    for case in cases:
        impulse = case.arrays["boundary_impulse"]
        expected = (len(case.arrays["time"]) - 1, len(case.mesh.rest), 2)
        if impulse.shape != expected or not np.isfinite(impulse).all():
            raise ValueError("Boundary impulse supervision has invalid shape or non-finite values")
        if np.any(impulse[:, ~case.arrays["boundary_mask"]] != 0):
            raise ValueError("Boundary impulse targets must vanish on unconstrained coordinates")
    squared_sum, count = 0.0, 0
    for case in training:
        values = np.asarray(case.arrays["boundary_impulse"][:, case.arrays["boundary_mask"]], dtype=np.float64)
        squared_sum += float(np.sum(values ** 2))
        count += values.size
    if count == 0:
        raise ValueError("Reaction supervision requires constrained coordinates in the training set")
    scale = float(np.sqrt(squared_sum / count))
    if not np.isfinite(scale):
        raise ValueError("Training boundary impulse scale is non-finite")
    return {
        "enabled": True,
        "scale": scale if scale > 0 else 1.0,
        "scale_source": "training_boundary_impulse_rms" if scale > 0 else "unit_scale_zero_training_signal",
    }


def reaction_impulse_loss(mass, boundary_mask, previous_velocity, next_velocity,
                          internal_force, target, dt, scale):
    if not np.isfinite([dt, scale]).all() or min(dt, scale) <= 0:
        raise ValueError("Reaction loss timestep and normalization scale must be positive and finite")
    if target.shape != previous_velocity.shape or boundary_mask.shape != target.shape:
        raise ValueError("Reaction targets and boundary mask must match the nodal velocity array")
    prediction = mass[:, None] * (next_velocity - previous_velocity) - dt * internal_force
    residual = ((prediction - target) / scale)[boundary_mask]
    return residual.square().mean() if residual.numel() else prediction.sum() * 0


def force_configuration(training, validation):
    for case in training + validation:
        values = case.arrays.get("internal_force")
        expected = case.arrays["position"].shape
        if values is None or values.shape != expected or not np.isfinite(values).all():
            raise ValueError("Equilibrium training requires finite instantaneous internal_force at every saved state")
    squared_sum = sum(float(np.sum(case.arrays["internal_force"] ** 2)) for case in training)
    count = sum(case.arrays["internal_force"].size for case in training)
    scale = float(np.sqrt(squared_sum / count))
    if not np.isfinite(scale):
        raise ValueError("Training instantaneous force scale is non-finite")
    return {
        "enabled": True, "scale": scale if scale > 0 else 1.0,
        "scale_source": "training_instantaneous_force_rms" if scale > 0 else "unnormalized_zero_training_force",
        "target": "instantaneous_nodal_internal_force_at_window_start",
    }


def rollout_curriculum(rollout_steps, schedule=None):
    if schedule is None:
        values = [step for step in (1, 2, 4, 8, 16, 32, 64) if step < rollout_steps]
        return [*values, rollout_steps]
    values = list(schedule)
    if not values or any(isinstance(value, bool) or int(value) != value or value < 1 for value in values):
        raise ValueError("Rollout schedule must contain positive integer lengths")
    if values != sorted(set(values)) or values[-1] != rollout_steps:
        raise ValueError("Rollout schedule must be increasing and end at rollout_steps")
    return values


def rollout_for_epoch(schedule, epoch, epochs, curriculum_epochs=None):
    ramp_epochs = epochs if curriculum_epochs is None else curriculum_epochs
    index = len(schedule) - 1 if ramp_epochs == 1 else round(
        (min(epoch, ramp_epochs) - 1) * (len(schedule) - 1) / (ramp_epochs - 1)
    )
    return schedule[index]


def curriculum_patience(stale_epochs, improved, previous_steps, active_steps, final_steps, patience):
    if improved or previous_steps != active_steps:
        stale_epochs = 0
    if not improved:
        stale_epochs += 1
    stopped = patience is not None and active_steps == final_steps and stale_epochs >= patience
    return stale_epochs, stopped


def motion_scale(training, name):
    values = [case.arrays[name] - case.mesh.rest if name == "position" else case.arrays[name]
              for case in training]
    scale = float(np.sqrt(sum(float(np.sum(value ** 2)) for value in values) / sum(value.size for value in values)))
    return scale if scale > 0 else 1.0


def validation_subset(rows, first_index, limit, seed):
    order = np.random.default_rng(seed).permutation(len(rows)).tolist()
    selected, seen = [], set()
    for index in order:
        group = str(rows[index]["group"])
        if group not in seen:
            selected.append(index)
            seen.add(group)
    selected.extend(index for index in order if index not in selected)
    return [first_index + index for index in selected[:limit]]


def autonomous_validation(model, cached, indices, dt, position_scale, velocity_scale):
    from .integration2d import advance

    scores = []
    for index in indices:
        graph, arrays = cached[index]
        position, velocity = arrays["position"][0], arrays["velocity"][0]
        free = ~graph.boundary_mask
        cache = None
        values = []
        for frame in range(1, len(arrays["position"])):
            result = advance(model, graph, position, velocity,
                             arrays["boundary_position"][frame], arrays["boundary_velocity"][frame], dt, cache)
            position, velocity, cache = result.position, result.velocity, result.cache
            if not torch.isfinite(position).all() or not torch.isfinite(velocity).all():
                raise RuntimeError("Non-finite autonomous validation rollout")
            position_error = ((position - arrays["position"][frame]) / position_scale)[free]
            velocity_error = ((velocity - arrays["velocity"][frame]) / velocity_scale)[free]
            if position_error.numel():
                values.append(position_error.square().mean() + velocity_error.square().mean())
        if not values:
            raise ValueError("Autonomous validation requires free coordinates")
        scores.append(float(torch.stack(values).mean()))
    return float(np.mean(scores))


def fit(data, output, method, epochs=20, batch_size=8, rollout_steps=2,
        windows_per_epoch=1024, device="cpu", seed=17, hidden=64, layers=3,
        learning_rate=0.0003, patience=None, architecture="legacy", rollout_schedule=None,
        autonomous_validation_cases=8, force_weight=1.0, curriculum_epochs=None):
    if min(epochs, batch_size, rollout_steps, windows_per_epoch, hidden, layers) < 1:
        raise ValueError("Training counts must be positive")
    if not np.isfinite(learning_rate) or learning_rate <= 0:
        raise ValueError("Learning rate must be positive")
    if architecture not in {"legacy", "equilibrium"}:
        raise ValueError("Architecture must be legacy or equilibrium")
    if not np.isfinite(force_weight) or force_weight <= 0:
        raise ValueError("Force supervision weight must be positive and finite")
    if isinstance(autonomous_validation_cases, bool) or int(autonomous_validation_cases) != autonomous_validation_cases or autonomous_validation_cases < 1:
        raise ValueError("Autonomous validation case count must be a positive integer")
    if patience is not None and (isinstance(patience, bool) or int(patience) != patience or patience < 1):
        raise ValueError("Early stopping patience must be a positive integer")
    if curriculum_epochs is not None and (
        isinstance(curriculum_epochs, bool) or int(curriculum_epochs) != curriculum_epochs
        or not 1 <= curriculum_epochs <= epochs
    ):
        raise ValueError("Curriculum epochs must be a positive integer no greater than epochs")
    if device.startswith("cuda") and not torch.cuda.is_available():
        raise ValueError("CUDA was requested but is unavailable")
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)
    output = Path(output)
    if output.exists() and any(output.iterdir()):
        raise ValueError("Training output must be a new or empty directory")
    output.mkdir(parents=True, exist_ok=True)
    training_entries = collection(data, "train")
    validation_entries = collection(data, "validation")
    training = [Case.load(path) for _, path in training_entries]
    validation = [Case.load(path) for _, path in validation_entries]
    training_hashes = {str(case.arrays["case_hash"].item()) for case in training}
    if any(str(case.arrays["case_hash"].item()) in training_hashes for case in validation):
        raise ValueError("An identical input case leaks from training into validation")
    cases = training + validation
    reactions = reaction_configuration(training, validation)
    forces = force_configuration(training, validation) if architecture == "equilibrium" else {"enabled": False}
    schedule = rollout_curriculum(rollout_steps, rollout_schedule) if architecture == "equilibrium" else [rollout_steps]
    loss_terms = ["position", "velocity"]
    if reactions["enabled"]:
        loss_terms.append("boundary_impulse")
    if forces["enabled"]:
        loss_terms.append("instantaneous_internal_force")
    for case in cases:
        if len(case.arrays["time"]) <= rollout_steps:
            raise ValueError("Rollout window exceeds a training trajectory")
    dt = float(np.diff(cases[0].arrays["time"])[0])
    if any(not np.allclose(np.diff(case.arrays["time"]), dt, rtol=1e-9, atol=1e-12) for case in cases):
        raise ValueError("Training requires a common uniform timestep")
    velocity_scale = max(float(np.sqrt(np.mean([
        np.mean(case.arrays["velocity"] ** 2) for case in training
    ]))), 1e-8)
    position_scale = motion_scale(training, "position")
    if architecture == "equilibrium":
        velocity_scale = motion_scale(training, "velocity")
    cached = []
    for case in cases:
        arrays = {
            key: torch.as_tensor(case.arrays[key], dtype=torch.float32, device=device)
            for key in ("position", "velocity", "boundary_position", "boundary_velocity")
        }
        if reactions["enabled"]:
            arrays["boundary_impulse"] = torch.as_tensor(case.arrays["boundary_impulse"], dtype=torch.float32, device=device)
        if forces["enabled"]:
            arrays["internal_force"] = torch.as_tensor(case.arrays["internal_force"], dtype=torch.float32, device=device)
        cached.append((graph_for(case, device), arrays))
    config = {"method": method, "hidden_size": hidden, "layers": layers}
    if architecture == "equilibrium":
        config["architecture"] = architecture
    model = create_model(config).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)
    best = float("inf")
    stale_epochs = 0
    previous_steps = None
    validation_windows = [(index, start) for index in range(len(training), len(cases))
                          for start in range(0, len(cases[index].arrays["time"]) - rollout_steps, rollout_steps)]
    validation_indices = validation_subset([row for row, _ in validation_entries], len(training),
                                           autonomous_validation_cases, seed)
    started = perf_counter()

    def batch_loss(windows, steps):
        from .integration2d import advance

        graph = Batch.from_data_list([cached[index][0] for index, _ in windows])
        positions = torch.cat([cached[index][1]["position"][start] for index, start in windows])
        velocities = torch.cat([cached[index][1]["velocity"][start] for index, start in windows])
        loss = torch.zeros((), device=device)
        force_loss = torch.zeros((), device=device)
        cache = None
        for offset in range(1, steps + 1):
            target_x, target_v, boundary_x, boundary_v = [
                torch.cat([cached[index][1][key][start + offset] for index, start in windows])
                for key in ("position", "velocity", "boundary_position", "boundary_velocity")
            ]
            previous_velocity = velocities
            result = advance(model, graph, positions, velocities, boundary_x, boundary_v, dt, cache)
            positions, velocities, internal_force, cache = result.position, result.velocity, result.mean_force, result.cache
            if forces["enabled"] and offset == 1:
                target_force = torch.cat([cached[index][1]["internal_force"][start] for index, start in windows])
                force_loss = ((result.start_force - target_force) / forces["scale"]).square().mean()
            free = ~graph.boundary_mask
            free_count = free.sum().clamp_min(1)
            loss = loss + ((((velocities - target_v) / velocity_scale) ** 2) * free).sum() / free_count
            loss = loss + ((((positions - target_x) / (velocity_scale * dt)) ** 2) * free).sum() / free_count
            if reactions["enabled"]:
                target_impulse = torch.cat([
                    cached[index][1]["boundary_impulse"][start + offset - 1]
                    for index, start in windows
                ])
                loss = loss + reaction_impulse_loss(
                    graph.mass, graph.boundary_mask, previous_velocity, velocities,
                    internal_force, target_impulse, dt, reactions["scale"],
                )
        return loss / steps + force_weight * force_loss

    emit("training_started", method=method, device=device, training_cases=len(training),
         validation_cases=len(validation), batch_size=batch_size, windows_per_epoch=windows_per_epoch,
         training_loss_terms=loss_terms, reaction_supervision=reactions["enabled"],
         reaction_scale=reactions["scale"], reaction_scale_source=reactions["scale_source"],
         architecture=architecture, rollout_schedule=schedule, force_supervision=forces,
         curriculum_epochs=epochs if curriculum_epochs is None else curriculum_epochs,
         early_stopping_scope="final_rollout_stage_only",
         checkpoint_selection="autonomous_validation" if architecture == "equilibrium" else "window_validation",
         autonomous_validation_cases=len(validation_indices) if architecture == "equilibrium" else 0)
    for epoch in range(1, epochs + 1):
        active_steps = rollout_for_epoch(schedule, epoch, epochs, curriculum_epochs)
        model.train()
        total = 0.0
        count = 0
        for begin in range(0, windows_per_epoch, batch_size):
            size = min(batch_size, windows_per_epoch - begin)
            windows = []
            for _ in range(size):
                index = int(rng.integers(len(training)))
                start = int(rng.integers(len(training[index].arrays["time"]) - active_steps))
                windows.append((index, start))
            optimizer.zero_grad(set_to_none=True)
            loss = batch_loss(windows, active_steps)
            if not torch.isfinite(loss):
                raise RuntimeError("Non-finite training loss")
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0, error_if_nonfinite=True)
            optimizer.step()
            total += float(loss.detach()) * size
            count += size
            if begin == 0 or count == windows_per_epoch or count % (batch_size * 20) == 0:
                emit("training_progress", epoch=epoch, windows=count, total=windows_per_epoch,
                     rollout_steps=active_steps, loss=float(loss.detach()), elapsed_seconds=perf_counter() - started)
        model.eval()
        validation_total = 0.0
        with torch.inference_mode():
            for begin in range(0, len(validation_windows), batch_size):
                windows = validation_windows[begin:begin + batch_size]
                value = float(batch_loss(windows, rollout_steps))
                if not np.isfinite(value):
                    raise RuntimeError("Non-finite validation loss")
                validation_total += value * len(windows)
            autonomous_loss = autonomous_validation(
                model, cached, validation_indices, dt, position_scale, velocity_scale,
            ) if architecture == "equilibrium" else None
        validation_loss = validation_total / len(validation_windows)
        selection_loss = autonomous_loss if autonomous_loss is not None else validation_loss
        payload = {
            "format": "cut-dynamics-a-v2" if architecture == "equilibrium" else "cut-dynamics-a-v1", "model_config": config, "trained": True,
            "state_dict": model.state_dict(), "epoch": epoch, "seed": seed,
            "validation_loss": validation_loss, "velocity_scale": velocity_scale,
            "training_case_hashes": [str(case.arrays["case_hash"].item()) for case in training],
            "training_groups": sorted({str(row["group"]) for row, _ in training_entries}),
            "rollout_steps": rollout_steps, "dt": dt,
            "training_loss_terms": loss_terms, "reaction_supervision": reactions["enabled"],
            "reaction_scale": reactions["scale"], "reaction_scale_source": reactions["scale_source"],
            "force_supervision": forces, "force_weight": force_weight,
            "position_scale": position_scale, "rollout_schedule": schedule, "active_rollout_steps": active_steps,
            "curriculum_epochs": epochs if curriculum_epochs is None else curriculum_epochs,
            "autonomous_validation_loss": autonomous_loss,
            "autonomous_validation_case_hashes": [str(cases[index].arrays["case_hash"].item()) for index in validation_indices] if architecture == "equilibrium" else [],
            "checkpoint_selection": "autonomous_validation" if architecture == "equilibrium" else "window_validation",
            "selection_loss": selection_loss,
        }
        torch.save(payload, output / "last.pt")
        improved = selection_loss < best
        if improved:
            best = selection_loss
            torch.save(payload, output / "best.pt")
        stale_epochs, stopped = curriculum_patience(
            stale_epochs, improved, previous_steps, active_steps, schedule[-1], patience,
        )
        previous_steps = active_steps
        emit("epoch_completed", epoch=epoch, train_loss=total / count,
             rollout_steps=active_steps, validation_loss=validation_loss,
             autonomous_validation_loss=autonomous_loss, selection_loss=selection_loss, best_validation_loss=best,
             patience_stale_epochs=stale_epochs, final_rollout_stage=active_steps == schedule[-1],
             elapsed_seconds=perf_counter() - started)
        if stopped:
            emit("training_stopped", reason="validation_early_stopping", epoch=epoch,
                 patience=patience, best_validation_loss=best)
            break
    return {"checkpoint": str(output / "best.pt"), "best_validation_loss": best,
            "training_loss_terms": loss_terms, "reaction_supervision": reactions["enabled"],
            "reaction_scale": reactions["scale"], "reaction_scale_source": reactions["scale_source"],
            "architecture": architecture, "force_supervision": forces, "rollout_schedule": schedule,
            "checkpoint_selection": "autonomous_validation" if architecture == "equilibrium" else "window_validation"}
