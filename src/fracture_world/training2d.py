import json
import random
from pathlib import Path
from time import perf_counter

import numpy as np
import torch
from torch_geometric.data import Batch

from .cases import Case, collection
from .neural import DynamicsModel, build_graph


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
    if checkpoint.get("format") != "cut-dynamics-a-v1":
        raise ValueError("Checkpoint is not a stage A dynamics model")
    model = DynamicsModel(**checkpoint["model_config"]).to(device)
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


def fit(data, output, method, epochs=20, batch_size=8, rollout_steps=2,
        windows_per_epoch=1024, device="cpu", seed=17, hidden=64, layers=3,
        learning_rate=0.0003, patience=None):
    if min(epochs, batch_size, rollout_steps, windows_per_epoch, hidden, layers) < 1:
        raise ValueError("Training counts must be positive")
    if learning_rate <= 0:
        raise ValueError("Learning rate must be positive")
    if patience is not None and (isinstance(patience, bool) or int(patience) != patience or patience < 1):
        raise ValueError("Early stopping patience must be a positive integer")
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
    training = [Case.load(path) for _, path in collection(data, "train")]
    validation = [Case.load(path) for _, path in collection(data, "validation")]
    training_hashes = {str(case.arrays["case_hash"].item()) for case in training}
    if any(str(case.arrays["case_hash"].item()) in training_hashes for case in validation):
        raise ValueError("An identical input case leaks from training into validation")
    cases = training + validation
    reactions = reaction_configuration(training, validation)
    loss_terms = ["position", "velocity"]
    if reactions["enabled"]:
        loss_terms.append("boundary_impulse")
    for case in cases:
        if len(case.arrays["time"]) <= rollout_steps:
            raise ValueError("Rollout window exceeds a training trajectory")
    dt = float(np.diff(cases[0].arrays["time"])[0])
    if any(not np.allclose(np.diff(case.arrays["time"]), dt, rtol=1e-9, atol=1e-12) for case in cases):
        raise ValueError("Training requires a common uniform timestep")
    velocity_scale = max(float(np.sqrt(np.mean([
        np.mean(case.arrays["velocity"] ** 2) for case in training
    ]))), 1e-8)
    cached = []
    for case in cases:
        arrays = {
            key: torch.as_tensor(case.arrays[key], dtype=torch.float32, device=device)
            for key in ("position", "velocity", "boundary_position", "boundary_velocity")
        }
        if reactions["enabled"]:
            arrays["boundary_impulse"] = torch.as_tensor(case.arrays["boundary_impulse"], dtype=torch.float32, device=device)
        cached.append((graph_for(case, device), arrays))
    config = {"method": method, "hidden_size": hidden, "layers": layers}
    model = DynamicsModel(**config).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)
    best = float("inf")
    stale_epochs = 0
    validation_windows = [(index, start) for index in range(len(training), len(cases))
                          for start in range(0, len(cases[index].arrays["time"]) - rollout_steps, rollout_steps)]
    started = perf_counter()

    def batch_loss(windows):
        graph = Batch.from_data_list([cached[index][0] for index, _ in windows])
        positions = torch.cat([cached[index][1]["position"][start] for index, start in windows])
        velocities = torch.cat([cached[index][1]["velocity"][start] for index, start in windows])
        loss = torch.zeros((), device=device)
        for offset in range(1, rollout_steps + 1):
            target_x, target_v, boundary_x, boundary_v = [
                torch.cat([cached[index][1][key][start + offset] for index, start in windows])
                for key in ("position", "velocity", "boundary_position", "boundary_velocity")
            ]
            previous_velocity = velocities
            positions, velocities, internal_force = model(graph, positions, velocities, boundary_x, boundary_v, dt)
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
        return loss / rollout_steps

    emit("training_started", method=method, device=device, training_cases=len(training),
         validation_cases=len(validation), batch_size=batch_size, windows_per_epoch=windows_per_epoch,
         training_loss_terms=loss_terms, reaction_supervision=reactions["enabled"],
         reaction_scale=reactions["scale"], reaction_scale_source=reactions["scale_source"])
    for epoch in range(1, epochs + 1):
        model.train()
        total = 0.0
        count = 0
        for begin in range(0, windows_per_epoch, batch_size):
            size = min(batch_size, windows_per_epoch - begin)
            windows = []
            for _ in range(size):
                index = int(rng.integers(len(training)))
                start = int(rng.integers(len(training[index].arrays["time"]) - rollout_steps))
                windows.append((index, start))
            optimizer.zero_grad(set_to_none=True)
            loss = batch_loss(windows)
            if not torch.isfinite(loss):
                raise RuntimeError("Non-finite training loss")
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0, error_if_nonfinite=True)
            optimizer.step()
            total += float(loss.detach()) * size
            count += size
            if begin == 0 or count == windows_per_epoch or count % (batch_size * 20) == 0:
                emit("training_progress", epoch=epoch, windows=count, total=windows_per_epoch,
                     loss=float(loss.detach()), elapsed_seconds=perf_counter() - started)
        model.eval()
        validation_total = 0.0
        with torch.inference_mode():
            for begin in range(0, len(validation_windows), batch_size):
                windows = validation_windows[begin:begin + batch_size]
                value = float(batch_loss(windows))
                if not np.isfinite(value):
                    raise RuntimeError("Non-finite validation loss")
                validation_total += value * len(windows)
        validation_loss = validation_total / len(validation_windows)
        payload = {
            "format": "cut-dynamics-a-v1", "model_config": config, "trained": True,
            "state_dict": model.state_dict(), "epoch": epoch, "seed": seed,
            "validation_loss": validation_loss, "velocity_scale": velocity_scale,
            "training_case_hashes": [str(case.arrays["case_hash"].item()) for case in training],
            "training_groups": sorted({str(row["group"]) for row, _ in collection(data, "train")}),
            "rollout_steps": rollout_steps, "dt": dt,
            "training_loss_terms": loss_terms, "reaction_supervision": reactions["enabled"],
            "reaction_scale": reactions["scale"], "reaction_scale_source": reactions["scale_source"],
        }
        torch.save(payload, output / "last.pt")
        if validation_loss < best:
            best = validation_loss
            stale_epochs = 0
            torch.save(payload, output / "best.pt")
        else:
            stale_epochs += 1
        emit("epoch_completed", epoch=epoch, train_loss=total / count,
             validation_loss=validation_loss, best_validation_loss=best,
             elapsed_seconds=perf_counter() - started)
        if patience is not None and stale_epochs >= patience:
            emit("training_stopped", reason="validation_early_stopping", epoch=epoch,
                 patience=patience, best_validation_loss=best)
            break
    return {"checkpoint": str(output / "best.pt"), "best_validation_loss": best,
            "training_loss_terms": loss_terms, "reaction_supervision": reactions["enabled"],
            "reaction_scale": reactions["scale"], "reaction_scale_source": reactions["scale_source"]}
