from dataclasses import asdict, dataclass
from datetime import datetime, timezone
import json
from pathlib import Path
from time import perf_counter

import numpy as np
import torch
from torch.nn import functional as F

from .dynamics_data import load_dynamics_dataset, split_dynamics_cases, supervision_counts
from .dynamics_model import NeuralDynamicsCorrector, TemporalDynamicsModel
from .io import new_directory, write_json
from .learning import choose_device, seed_all


FORMAT = "fracture-world.temporal-dynamics.v1"
MATERIAL_KEYS = ("rest_density", "youngs_modulus", "poissons_ratio", "fracture_stress_threshold")


@dataclass(frozen=True)
class DynamicsTrainingConfig:
    epochs: int
    corrector_epochs: int = 0
    hidden: int = 48
    heads: int = 3
    graph_layers: int = 2
    transformer_layers: int = 1
    flow_architecture: str = "coupled"
    flow_layers: int = 2
    learning_rate: float = 0.0003
    fracture_weight: float = 1.0
    flow_weight: float = 1.0
    seed: int = 17
    device: str = "auto"
    overfit: bool = False
    learn_support_release: bool = False
    require_event_coverage: bool = False
    min_post_event_steps: int = 10
    event_stratified: bool = False
    split_unit: str = "loading_condition"
    motion_prior: str = "constant_acceleration"
    stream_data: bool = False
    rollout_steps: int = 1


def state_batch(topology, positions, velocities, broken, fixed, gravity, material, dt, device, floor_height=None):
    arrays = {"rest_positions": topology.rest_positions, "positions": positions, "velocities": velocities,
              "masses": topology.masses, "forces": topology.masses[:, None] * np.asarray(gravity),
              "material": [material[name] for name in MATERIAL_KEYS]}
    result = {key: torch.tensor(np.array(value, copy=True), dtype=torch.float32, device=device)
              for key, value in arrays.items()}
    result.update(face_pairs=torch.tensor(topology.face_pairs.copy(), dtype=torch.long, device=device),
                  broken=torch.tensor(np.array(broken, copy=True), dtype=torch.bool, device=device),
                  fixed=torch.tensor(np.array(fixed, copy=True), dtype=torch.bool, device=device), dt=float(dt))
    if floor_height is not None:
        result["floor_height"] = float(floor_height)
    return result


def transition_batch(trajectory, frame, device):
    item = trajectory.transition(frame)
    return state_batch(trajectory.topology, item.positions, item.velocities, item.broken, item.fixed,
                       trajectory.gravity, trajectory.material, item.dt, device,
                       trajectory.metadata.get("conditioning_context", {}).get("floor_height_m"))


class _TrainingTrajectory:
    def __init__(self, trajectory, device):
        self.base = transition_batch(trajectory, 0, device)
        self.frames = {key: torch.tensor(np.array(getattr(trajectory, key), copy=True), dtype=dtype, device=device)
                       for key, dtype in (("positions", torch.float32), ("velocities", torch.float32),
                                          ("broken", torch.bool), ("fixed", torch.bool))}
        self.timesteps = np.diff(trajectory.times)

    def batch(self, frame):
        return {**self.base, **{key: values[frame] for key, values in self.frames.items()},
                "dt": float(self.timesteps[frame])}

    def targets(self, frame):
        return [self.frames[key][frame + 1] for key in ("positions", "velocities", "broken", "fixed")]


class _TrainingMetrics:
    names = ("loss", "motion", "fracture", "flow", "support_release")

    def __init__(self, device):
        self.sums = torch.zeros(len(self.names), dtype=torch.float64, device=device)
        self.last = None
        self.count = 0

    def update(self, loss, terms):
        self.last = torch.stack([loss.detach(), *(terms[key] for key in self.names[1:])]).to(torch.float64)
        self.sums.add_(self.last)
        self.count += 1

    def scores(self):
        return dict(zip(self.names, (self.sums / self.count).cpu().tolist())) if self.count else None


def _scales(trajectories):
    length = max(float(np.linalg.norm(np.ptp(item.topology.mesh.nodes, axis=0))) for item in trajectories)
    velocity = max(item.velocity_rms if hasattr(item, "velocity_rms") else float(np.sqrt(np.mean(item.velocities**2)))
                   for item in trajectories)
    if velocity <= np.finfo(float).eps:
        velocity = length / max(float(np.ptp(item.times)) for item in trajectories)
    mass = float(np.mean(np.concatenate([item.topology.masses for item in trajectories])))
    material = np.mean([[item.material[key] for key in MATERIAL_KEYS] for item in trajectories], axis=0)
    material[2] = 1.0
    edges = np.concatenate([np.linalg.norm(item.topology.rest_positions[item.topology.tet_corner_ids[:, 1:]]
                                           - item.topology.rest_positions[item.topology.tet_corner_ids[:, :1]], axis=-1).ravel()
                            for item in trajectories])
    return dict(length_scale=length, velocity_scale=velocity, mass_scale=mass,
                material_scale=material.tolist(), contact_radius=float(np.median(edges)))


def _loss(model, batch, target_positions, target_velocities, target_broken, target_fixed, config, proposal=None,
          flow=True, fracture_positive_weight=1.0, support_positive_weight=1.0, return_output=False):
    context = None
    if flow and proposal is None and hasattr(model, "forward_with_flow_context"):
        output, context = model.forward_with_flow_context(batch)
    else:
        output = model(batch) if proposal is None else model(batch, proposal)
    mass = batch["masses"] / batch["masses"].sum()
    velocity = model.config["velocity_scale"]
    position_error = ((output["positions"] - target_positions) / (batch["dt"] * velocity)).square().sum(dim=1)
    velocity_error = ((output["velocities"] - target_velocities) / velocity).square().sum(dim=1)
    motion = (mass * (position_error + velocity_error)).sum()
    intact = ~batch["broken"]
    fracture = motion.new_zeros(())
    matching = motion.new_zeros(())
    if intact.any():
        target = target_broken.to(output["fracture_logits"].dtype)
        fracture = F.binary_cross_entropy_with_logits(output["fracture_logits"][intact], target[intact],
                                                       pos_weight=motion.new_tensor(fracture_positive_weight))
        if flow:
            noise = torch.randn_like(target)
            time = torch.rand((), device=target.device)
            destination = 2 * target - 1
            field = (1 - time) * noise + time * destination
            field = torch.where(batch["broken"], torch.ones_like(field), field)
            prediction = (model.fracture_velocity(batch, field, time, context=context) if context is not None
                          else model.fracture_velocity(batch, field, time))
            matching = (prediction[intact] - (destination - noise)[intact]).square().mean()
    support = motion.new_zeros(())
    if config.learn_support_release and batch["fixed"].any():
        release = (~target_fixed[batch["fixed"]]).to(motion.dtype)
        support = F.binary_cross_entropy_with_logits(output["support_release_logits"][batch["fixed"]], release,
                                                      pos_weight=motion.new_tensor(support_positive_weight))
    loss = motion + config.fracture_weight * fracture + config.flow_weight * matching + support
    terms = dict(motion=motion.detach(), fracture=fracture.detach(), flow=matching.detach(),
                 support_release=support.detach())
    return (loss, terms, output) if return_output else (loss, terms)


def _validate_autoregressive_prediction(output, case_id, frame):
    if not all(torch.isfinite(output[key]).all() for key in ("positions", "velocities")):
        raise FloatingPointError(f"nonfinite autoregressive training state in {case_id} at frame {frame}")
    corners = output["positions"].reshape(-1, 4, 3)
    edges = corners[:, 1:] - corners[:, :1]
    scale = edges.abs().amax(dim=(-2, -1))
    if (scale <= 0).any() or (torch.linalg.det(edges / scale[:, None, None]) <= 0).any():
        raise FloatingPointError(f"inverted or degenerate autoregressive training tetrahedron in {case_id} at frame {frame}")


def _rollout_loss(network, trajectory, frame, device, config, *, fixed_predictor=None,
                  fracture_positive_weight=1.0, support_positive_weight=1.0, prepared=None):
    batch = transition_batch(trajectory, frame, device) if prepared is None else prepared.batch(frame)
    horizon = min(config.rollout_steps, len(trajectory) - frame)
    losses, components = [], []
    for offset in range(horizon):
        if prepared is None:
            target = trajectory.transition(frame + offset)
            batch = {**batch, "dt": target.dt}
            target_values = [torch.tensor(np.array(value, copy=True), dtype=dtype, device=device) for value, dtype in (
                (target.target_positions, torch.float32), (target.target_velocities, torch.float32),
                (target.target_broken, torch.bool), (target.target_fixed, torch.bool))]
        else:
            batch = {**batch, "dt": float(prepared.timesteps[frame + offset])}
            target_values = prepared.targets(frame + offset)
        proposal = fixed_predictor(batch) if fixed_predictor is not None else None
        loss, terms, output = _loss(network, batch, *target_values, config, proposal,
                                    flow=fixed_predictor is None, fracture_positive_weight=fracture_positive_weight,
                                    support_positive_weight=support_positive_weight, return_output=True)
        if not torch.isfinite(loss):
            raise FloatingPointError(f"nonfinite temporal dynamics training loss in {trajectory.case_id} at frame {frame + offset}")
        if config.rollout_steps > 1:
            _validate_autoregressive_prediction(output, trajectory.case_id, frame + offset + 1)
        losses.append(loss)
        components.append(terms)
        batch = {**batch, "positions": output["positions"], "velocities": output["velocities"],
                 "broken": batch["broken"] | output["broken"], "fixed": batch["fixed"] & output["next_fixed"]}
    return torch.stack(losses).mean(), {key: torch.stack([item[key] for item in components]).mean(dtype=torch.float64)
                                      for key in components[0]}


def train_dynamics(data_dir, output_dir, config):
    if not isinstance(config, DynamicsTrainingConfig):
        raise TypeError("config must be DynamicsTrainingConfig")
    if config.epochs < 1 or config.corrector_epochs < 0 or config.learning_rate <= 0:
        raise ValueError("epochs and learning rate must be positive; corrector_epochs must be nonnegative")
    if isinstance(config.rollout_steps, bool) or not isinstance(config.rollout_steps, int) or config.rollout_steps < 1:
        raise ValueError("rollout_steps must be a positive integer")
    if not np.isfinite([config.learning_rate, config.fracture_weight, config.flow_weight]).all():
        raise ValueError("training weights must be finite")
    if config.fracture_weight <= 0 or config.flow_weight <= 0:
        raise ValueError("fracture and flow weights must be positive")
    trajectories = load_dynamics_dataset(data_dir, stream=True) if config.stream_data else load_dynamics_dataset(data_dir)
    for trajectory in trajectories:
        if not config.learn_support_release and not np.array_equal(trajectory.fixed, np.broadcast_to(trajectory.fixed[0], trajectory.fixed.shape)):
            raise ValueError("automatic support release is not modeled; use trajectories with prescribed constant material supports")
        base_dt = trajectory.metadata["conditioning_context"]["dt"]
        if not np.allclose(np.diff(trajectory.times), base_dt, rtol=64 * np.finfo(np.float32).eps, atol=0):
            raise ValueError("temporal dynamics training requires every native step; regenerate with record_stride=1")
    splits = split_dynamics_cases(trajectories, config.seed, config.overfit, event_stratified=config.event_stratified,
                                  split_unit=config.split_unit)
    training = splits["train"]
    if not training:
        raise ValueError("no training trajectories")
    coverage = None
    if config.require_event_coverage:
        from .dynamics_readiness import inspect_dynamics_dataset

        coverage = inspect_dynamics_dataset(data_dir, config.seed, config.overfit, config.min_post_event_steps,
                                           event_stratified=config.event_stratified, split_unit=config.split_unit,
                                           trajectories=trajectories)
        if not coverage["ready_for_training"]:
            raise ValueError("dataset is not ready for fracture/support-release training: " +
                             json.dumps(coverage["reasons"], allow_nan=False))
    counts = [supervision_counts(item) for item in training]
    fracture_positive = sum(item["fracture_positive"] for item in counts)
    fracture_negative = sum(item["fracture_negative"] for item in counts)
    support_positive = sum(item["support_positive"] for item in counts)
    support_negative = sum(item["support_negative"] for item in counts)
    fracture_positive_weight = max(1.0, fracture_negative / fracture_positive) if fracture_positive else 1.0
    support_positive_weight = max(1.0, support_negative / support_positive) if support_positive else 1.0
    seed_all(config.seed)
    device = choose_device(config.device)
    if device.type == "cpu":
        torch.set_num_threads(min(torch.get_num_threads(), 4))
    model_config = dict(hidden=config.hidden, heads=config.heads, graph_layers=config.graph_layers,
                        transformer_layers=config.transformer_layers, learn_support_release=config.learn_support_release,
                        flow_architecture=config.flow_architecture, flow_layers=config.flow_layers, motion_prior=config.motion_prior,
                        **_scales(training))
    model = TemporalDynamicsModel(**model_config).to(device)
    model.fracture_logit_offset = float(np.log(fracture_positive_weight))
    model.support_logit_offset = float(np.log(support_positive_weight))
    directory = new_directory(output_dir)
    manifest = {name: [item.case_id for item in rows] for name, rows in splits.items()}
    conditions = {name: sorted({item.condition_digest for item in rows}) for name, rows in splits.items()}
    shape_families = {name: sorted({item.shape_family for item in rows}) for name, rows in splits.items()}
    mesh_fingerprints = {name: sorted({item.topology.fingerprint for item in rows}) for name, rows in splits.items()}
    provenance = {item.case_id: item.metadata for item in trajectories}
    write_json(directory / "split.json", dict(cases=manifest, conditions=conditions,
                                              shape_families=shape_families, mesh_fingerprints=mesh_fingerprints,
                                              unit="complete shape family" if config.split_unit == "shape_family" else "complete loading condition",
                                              overfit_debug=config.overfit,
                                              event_stratified=config.event_stratified))
    if coverage is not None:
        write_json(directory / "coverage.json", coverage)
    checkpoint = {"format": FORMAT, "model_config": model_config, "training": asdict(config),
                  "splits": manifest, "condition_splits": conditions, "source": provenance,
                  "support_policy": "learned material-corner release" if config.learn_support_release else "prescribed material-corner supports",
                  "support_release_training_events": support_positive,
                  "fracture_training_events": fracture_positive,
                  "class_weights": dict(fracture_positive=fracture_positive_weight, support_release_positive=support_positive_weight),
                  "probability_policy": "subtract_log_training_positive_weight",
                  "runtime_engine_calls": 0, "constitutive_response": "learned from teacher trajectories",
                  "fracture_representation": "irreversible original tetrahedron-face separation",
                  "flow": "conditional Gaussian-to-signed-new-break-field flow matching; no OT",
                  "flow_architecture": config.flow_architecture,
                  "training_state_policy": "autoregressive predicted motion, irreversible classifier fracture state and support release inside each window",
                  "training_gradient_policy": "motion gradients retained through every predicted state in the window; discrete topology transitions are nondifferentiable",
                  "training_window_steps": config.rollout_steps,
                  "training_mesh_fingerprints": mesh_fingerprints["train"],
                  "evaluation_mesh_fingerprints": sorted(set(mesh_fingerprints["validation"] + mesh_fingerprints["test"])),
                  "shape_family_splits": shape_families, "mesh_fingerprint_splits": mesh_fingerprints,
                  "mesh_shape_families": {item.topology.fingerprint: item.shape_family for item in trajectories},
                  "generalization_protocol": config.split_unit}
    started = perf_counter()

    def train_stage(network, epochs, stage, fixed_predictor=None):
        optimizer = torch.optim.AdamW(network.parameters(), lr=config.learning_rate)
        if fixed_predictor is not None:
            fixed_predictor.requires_grad_(False)
            fixed_predictor.eval()
        best, best_state = float("inf"), None
        last_progress = perf_counter()
        for epoch in range(epochs):
            epoch_started = perf_counter()
            scores = {}
            timings = {}
            for name in ("train", "validation"):
                split_started = perf_counter()
                is_training = name == "train"
                network.train(is_training)
                metrics = _TrainingMetrics(device)
                window_count = sum(len(item) for item in splits[name])
                prepare_seconds = 0.0
                interval_started, interval_windows = split_started, 0
                for case_index, trajectory in enumerate(splits[name], start=1):
                    prepare_started = perf_counter()
                    prepared = _TrainingTrajectory(trajectory, device)
                    prepare_seconds += perf_counter() - prepare_started
                    order = np.random.permutation(len(trajectory)) if is_training else range(len(trajectory))
                    for transition_index, frame in enumerate(order, start=1):
                        with torch.set_grad_enabled(is_training):
                            loss, terms = _rollout_loss(network, trajectory, int(frame), device, config,
                                                        fixed_predictor=fixed_predictor,
                                                        fracture_positive_weight=fracture_positive_weight,
                                                        support_positive_weight=support_positive_weight, prepared=prepared)
                            if is_training:
                                optimizer.zero_grad(set_to_none=True)
                                loss.backward()
                                torch.nn.utils.clip_grad_norm_(network.parameters(), 1.0, error_if_nonfinite=True)
                                optimizer.step()
                        metrics.update(loss, terms)
                        if perf_counter() - last_progress >= 5:
                            current_loss = float(metrics.last[0])
                            now = perf_counter()
                            elapsed = now - split_started
                            seconds_per_window = elapsed / metrics.count
                            interval_count = metrics.count - interval_windows
                            print(json.dumps(dict(event="training_progress", stage=stage, epoch=epoch + 1,
                                                  timestamp=datetime.now(timezone.utc).isoformat(),
                                                  split=name, case=trajectory.case_id, case_index=case_index,
                                                  cases=len(splits[name]), transition=transition_index,
                                                  transitions=len(trajectory), rollout_steps=config.rollout_steps,
                                                  windows_completed=metrics.count, windows_total=window_count,
                                                  elapsed_seconds=elapsed, seconds_per_window=seconds_per_window,
                                                  interval_seconds_per_window=(now - interval_started) / interval_count,
                                                  eta_seconds=(window_count - metrics.count) * seconds_per_window,
                                                  data_prepare_seconds=prepare_seconds,
                                                  loss=current_loss, device=str(device)),
                                             allow_nan=False), flush=True)
                            last_progress = interval_started = perf_counter()
                            interval_windows = metrics.count
                    del prepared
                scores[name] = metrics.scores()
                timings[name] = dict(seconds=perf_counter() - split_started, windows=metrics.count,
                                     data_prepare_seconds=prepare_seconds)
                if metrics.count:
                    timings[name]["seconds_per_window"] = timings[name]["seconds"] / metrics.count
            criterion = (scores["validation"] or scores["train"])["loss"]
            record = dict(stage=stage, epoch=epoch + 1, losses=scores, seconds=perf_counter() - started,
                          epoch_seconds=perf_counter() - epoch_started, split_timing=timings,
                          device=str(device), overfit_debug=config.overfit, rollout_steps=config.rollout_steps)
            with (directory / "training.jsonl").open("a", encoding="utf-8") as stream:
                stream.write(json.dumps(record, allow_nan=False) + "\n")
            print(json.dumps(record, allow_nan=False), flush=True)
            if criterion < best:
                best = criterion
                best_state = {key: value.detach().cpu().clone() for key, value in network.state_dict().items()}
            if stage == "predictor":
                torch.save({**checkpoint, "model": network.state_dict(), "epoch": epoch + 1}, directory / "last.pt")
        network.load_state_dict(best_state)
        network.eval()
        return best_state

    checkpoint["model"] = train_stage(model, config.epochs, "predictor")
    torch.save(checkpoint, directory / "best.pt")
    if config.corrector_epochs:
        corrector = NeuralDynamicsCorrector(**model_config).to(device)
        corrector.fracture_logit_offset = model.fracture_logit_offset
        corrector.support_logit_offset = model.support_logit_offset
        checkpoint["corrector"] = train_stage(corrector, config.corrector_epochs, "neural_corrector", model)
        torch.save(checkpoint, directory / "best.pt")
    return directory / "best.pt"


def load_dynamics_model(path, device="auto"):
    device = choose_device(device)
    checkpoint = torch.load(Path(path), map_location=device, weights_only=True)
    if checkpoint.get("format") != FORMAT:
        raise ValueError("checkpoint is not a temporal fracture dynamics model")
    model_config = {"flow_architecture": "independent", **checkpoint["model_config"]}
    model = TemporalDynamicsModel(**model_config).to(device)
    model.load_state_dict(checkpoint["model"], strict=True)
    model.eval()
    corrector = None
    if "corrector" in checkpoint:
        corrector = NeuralDynamicsCorrector(**model_config).to(device)
        corrector.load_state_dict(checkpoint["corrector"], strict=True)
        corrector.eval()
    policy = checkpoint.get("probability_policy", "legacy_weighted_logits")
    if policy not in {"legacy_weighted_logits", "subtract_log_training_positive_weight"}:
        raise ValueError("unsupported checkpoint probability policy")
    if policy == "subtract_log_training_positive_weight":
        weights = checkpoint.get("class_weights", {})
        fracture_weight = weights.get("fracture_positive")
        support_weight = weights.get("support_release_positive")
        if any(not isinstance(value, (int, float)) or not np.isfinite(value) or value < 1
               for value in (fracture_weight, support_weight)):
            raise ValueError("checkpoint probability correction requires finite training positive weights >= 1")
        for network in (model, corrector):
            if network is not None:
                network.fracture_logit_offset = float(np.log(fracture_weight))
                network.support_logit_offset = float(np.log(support_weight))
    return model, corrector, checkpoint, device
