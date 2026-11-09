import threading
import json
from time import perf_counter

import numpy as np
import psutil
import torch
from torch_geometric.data import Batch

from .cases import Case
from .neural import create_model
from .training2d import force_configuration, graph_for, reaction_impulse_loss


def synchronize(device):
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def batch_inputs(case, batch_size, rollout_steps, device):
    graph = Batch.from_data_list([graph_for(case, device)] * batch_size)
    values = {}
    for name in ("position", "velocity", "boundary_position", "boundary_velocity"):
        trajectory = torch.as_tensor(
            case.arrays[name][:rollout_steps + 1], dtype=torch.float32, device=device
        )
        values[name] = trajectory.repeat(1, batch_size, 1)
    for name in ("internal_force", "boundary_impulse"):
        if name in case.arrays:
            trajectory = torch.as_tensor(case.arrays[name][:rollout_steps + (name == "internal_force")],
                                         dtype=torch.float32, device=device)
            values[name] = trajectory.repeat(1, batch_size, 1)
    return graph, values


def backward_window(model, graph, values, dt, rollout_steps, velocity_scale, force_scale=None, reaction_scale=None):
    from .integration2d import advance

    model.zero_grad(set_to_none=True)
    position = values["position"][0]
    velocity = values["velocity"][0]
    free = ~graph.boundary_mask
    loss = torch.zeros((), device=position.device)
    force_loss = torch.zeros_like(loss)
    cache = None
    for step in range(1, rollout_steps + 1):
        previous_velocity = velocity
        result = advance(model,
            graph, position, velocity,
            values["boundary_position"][step], values["boundary_velocity"][step], dt, cache,
        )
        position, velocity, cache = result.position, result.velocity, result.cache
        if step == 1 and force_scale is not None:
            force_loss = ((result.start_force - values["internal_force"][0]) / force_scale).square().mean()
        loss = loss + (((velocity - values["velocity"][step]) / velocity_scale)[free] ** 2).mean()
        loss = loss + (((position - values["position"][step]) / (velocity_scale * dt))[free] ** 2).mean()
        if reaction_scale is not None:
            loss = loss + reaction_impulse_loss(
                graph.mass, graph.boundary_mask, previous_velocity, velocity,
                result.mean_force, values["boundary_impulse"][step - 1], dt, reaction_scale,
            )
    loss = loss / rollout_steps + force_loss
    if not torch.isfinite(loss):
        raise RuntimeError("Non-finite profiling loss")
    loss.backward()
    return loss.detach()


def measured_repeats(model, graph, values, dt, rollout_steps, velocity_scale, repeats, device,
                     force_scale=None, reaction_scale=None):
    process = psutil.Process()
    peak_rss = [process.memory_info().rss]
    stopped = threading.Event()

    def sample_memory():
        while not stopped.wait(0.01):
            peak_rss[0] = max(peak_rss[0], process.memory_info().rss)

    monitor = threading.Thread(target=sample_memory, daemon=True)
    monitor.start()
    model.zero_grad(set_to_none=True)
    synchronize(device)
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
    started = perf_counter()
    try:
        for _ in range(repeats):
            loss = backward_window(model, graph, values, dt, rollout_steps, velocity_scale, force_scale, reaction_scale)
        synchronize(device)
        elapsed = perf_counter() - started
        memory = {
            "cpu_peak_rss_bytes_sampled": max(peak_rss[0], process.memory_info().rss),
            "gpu_peak_allocated_bytes": torch.cuda.max_memory_allocated(device) if device.type == "cuda" else None,
            "gpu_peak_reserved_bytes": torch.cuda.max_memory_reserved(device) if device.type == "cuda" else None,
        }
        return elapsed, float(loss), memory
    finally:
        stopped.set()
        monitor.join()


def profile_case(case_path, device="cuda", batch_sizes=(8, 16, 32), hidden=128,
                 layers=6, rollout_steps=4, repeats=2, method="stress", architecture="legacy"):
    device = torch.device(device)
    batch_sizes = tuple(batch_sizes)
    counts = (*batch_sizes, hidden, layers, rollout_steps, repeats)
    if not batch_sizes or any(isinstance(value, bool) or int(value) != value or value < 1 for value in counts):
        raise ValueError("Profile counts must be positive integers")
    if len(set(batch_sizes)) != len(batch_sizes):
        raise ValueError("Batch sizes must be unique")
    if device.type not in {"cpu", "cuda"}:
        raise ValueError("Profiling supports CPU or CUDA")
    if device.type == "cuda" and not torch.cuda.is_available():
        raise ValueError("CUDA was requested but is unavailable")
    case = Case.load(case_path)
    if len(case.arrays["time"]) <= rollout_steps:
        raise ValueError("Profiling window exceeds the available trajectory")
    dt = float(np.diff(case.arrays["time"])[0])
    if not np.allclose(np.diff(case.arrays["time"]), dt, rtol=1e-9, atol=1e-12):
        raise ValueError("Profiling requires a uniform timestep")
    if not np.any(~case.arrays["boundary_mask"]):
        raise ValueError("Profiling requires at least one free coordinate")
    velocity_scale = max(float(np.sqrt(np.mean(case.arrays["velocity"] ** 2))), 1e-8)
    if architecture not in {"legacy", "equilibrium"}:
        raise ValueError("Architecture must be legacy or equilibrium")
    config = {"method": method, "hidden_size": hidden, "layers": layers}
    if architecture == "equilibrium":
        config["architecture"] = architecture
    force_scale = force_configuration([case], [])['scale'] if architecture == "equilibrium" else None
    reaction_scale = None
    if "boundary_impulse" in case.arrays and np.any(case.arrays["boundary_mask"]):
        values = case.arrays["boundary_impulse"][:, case.arrays["boundary_mask"]]
        scale = float(np.sqrt(np.mean(values ** 2)))
        reaction_scale = scale if scale > 0 else 1.0
    model = create_model(config).to(device)
    model.train()
    initial = {name: parameter.detach().cpu().clone() for name, parameter in model.named_parameters()}
    records = []
    for batch_size in batch_sizes:
        graph = values = None
        try:
            print(json.dumps({"event": "profile_batch_started", "batch_size": batch_size}), flush=True)
            synchronize(device)
            prepared = perf_counter()
            graph, values = batch_inputs(case, batch_size, rollout_steps, device)
            synchronize(device)
            preparation_seconds = perf_counter() - prepared
            warmup = perf_counter()
            backward_window(model, graph, values, dt, rollout_steps, velocity_scale, force_scale, reaction_scale)
            synchronize(device)
            warmup_seconds = perf_counter() - warmup
            elapsed, loss, memory = measured_repeats(
                model, graph, values, dt, rollout_steps, velocity_scale, repeats, device,
                force_scale, reaction_scale,
            )
            records.append({
                "batch_size": batch_size, "status": "completed", "repeats": repeats,
                "preparation_seconds": preparation_seconds, "warmup_seconds": warmup_seconds,
                "measured_seconds": elapsed, "seconds_per_repeat": elapsed / repeats,
                "windows_per_second": batch_size * repeats / elapsed,
                "transitions_per_second": batch_size * rollout_steps * repeats / elapsed,
                "loss": loss, **memory,
            })
            print(json.dumps({"event": "profile_batch_completed", **records[-1]}), flush=True)
        except RuntimeError as error:
            if device.type != "cuda" or "out of memory" not in str(error).lower():
                raise
            records.append({"batch_size": batch_size, "status": "oom", "error": str(error)})
            print(json.dumps({"event": "profile_batch_oom", "batch_size": batch_size}), flush=True)
        finally:
            model.zero_grad(set_to_none=True)
            del graph, values
            if device.type == "cuda":
                torch.cuda.empty_cache()
    maximum_change = max(
        float((parameter.detach().cpu() - initial[name]).abs().max())
        for name, parameter in model.named_parameters()
    )
    completed = [record for record in records if record["status"] == "completed"]
    best = max(completed, key=lambda record: record["windows_per_second"]) if completed else None
    properties = torch.cuda.get_device_properties(device) if device.type == "cuda" else None
    return {
        "mode": "forward_backward_profile_only", "device": str(device),
        "device_name": properties.name if properties else "CPU",
        "device_total_memory_bytes": properties.total_memory if properties else None,
        "method": method, "architecture": architecture, "hidden": hidden, "layers": layers,
        "instantaneous_force_supervision": force_scale is not None,
        "reaction_supervision": reaction_scale is not None,
        "rollout_steps": rollout_steps, "parameter_count": sum(parameter.numel() for parameter in model.parameters()),
        "maximum_parameter_change": maximum_change, "optimizer_steps": 0,
        "checkpoint_written": False, "records": records,
        "best_measured_batch_size": best["batch_size"] if best else None,
        "scope": "Repeated copies of one case; excludes optimizer state, data loading, and heterogeneous-case variation",
        "cpu_memory_scope": "Process RSS sampled every 10 ms during measured repetitions",
        "timing_scope": "Fixed forward/backward repetitions; graph preparation and one warmup are reported separately",
    }
