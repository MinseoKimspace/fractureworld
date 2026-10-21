from dataclasses import asdict, dataclass
import json
from pathlib import Path
from time import perf_counter

import numpy as np
import torch
from torch.nn import functional as F

from .dynamics_data import load_dynamics_dataset, split_dynamics_cases
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


def state_batch(topology, positions, velocities, broken, fixed, gravity, material, dt, device):
    arrays = {"rest_positions": topology.rest_positions, "positions": positions, "velocities": velocities,
              "masses": topology.masses, "forces": topology.masses[:, None] * np.asarray(gravity),
              "material": [material[name] for name in MATERIAL_KEYS]}
    result = {key: torch.tensor(np.array(value, copy=True), dtype=torch.float32, device=device)
              for key, value in arrays.items()}
    result.update(face_pairs=torch.tensor(topology.face_pairs.copy(), dtype=torch.long, device=device),
                  broken=torch.tensor(np.array(broken, copy=True), dtype=torch.bool, device=device),
                  fixed=torch.tensor(np.array(fixed, copy=True), dtype=torch.bool, device=device), dt=float(dt))
    return result


def transition_batch(trajectory, frame, device):
    item = trajectory.transition(frame)
    return state_batch(trajectory.topology, item.positions, item.velocities, item.broken, item.fixed,
                       trajectory.gravity, trajectory.material, item.dt, device)


def _scales(trajectories):
    length = max(float(np.linalg.norm(np.ptp(item.topology.mesh.nodes, axis=0))) for item in trajectories)
    velocity = max(float(np.sqrt(np.mean(item.velocities**2))) for item in trajectories)
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
          flow=True, fracture_positive_weight=1.0, support_positive_weight=1.0):
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
            prediction = model.fracture_velocity(batch, field, time)
            matching = (prediction[intact] - (destination - noise)[intact]).square().mean()
    support = motion.new_zeros(())
    if config.learn_support_release and batch["fixed"].any():
        release = (~target_fixed[batch["fixed"]]).to(motion.dtype)
        support = F.binary_cross_entropy_with_logits(output["support_release_logits"][batch["fixed"]], release,
                                                      pos_weight=motion.new_tensor(support_positive_weight))
    loss = motion + config.fracture_weight * fracture + config.flow_weight * matching + support
    return loss, dict(motion=float(motion.detach()), fracture=float(fracture.detach()), flow=float(matching.detach()),
                      support_release=float(support.detach()))


def train_dynamics(data_dir, output_dir, config):
    if not isinstance(config, DynamicsTrainingConfig):
        raise TypeError("config must be DynamicsTrainingConfig")
    if config.epochs < 1 or config.corrector_epochs < 0 or config.learning_rate <= 0:
        raise ValueError("epochs and learning rate must be positive; corrector_epochs must be nonnegative")
    if not np.isfinite([config.learning_rate, config.fracture_weight, config.flow_weight]).all():
        raise ValueError("training weights must be finite")
    if config.fracture_weight <= 0 or config.flow_weight <= 0:
        raise ValueError("fracture and flow weights must be positive")
    trajectories = load_dynamics_dataset(data_dir)
    for trajectory in trajectories:
        if not config.learn_support_release and not np.array_equal(trajectory.fixed, np.broadcast_to(trajectory.fixed[0], trajectory.fixed.shape)):
            raise ValueError("automatic support release is not modeled; use trajectories with prescribed constant material supports")
        base_dt = trajectory.metadata["conditioning_context"]["dt"]
        if not np.allclose(np.diff(trajectory.times), base_dt, rtol=64 * np.finfo(np.float32).eps, atol=0):
            raise ValueError("temporal dynamics training requires every native step; regenerate with record_stride=1")
    splits = split_dynamics_cases(trajectories, config.seed, config.overfit, event_stratified=config.event_stratified)
    training = splits["train"]
    if not training:
        raise ValueError("no training trajectories")
    coverage = None
    if config.require_event_coverage:
        from .dynamics_readiness import inspect_dynamics_dataset

        coverage = inspect_dynamics_dataset(data_dir, config.seed, config.overfit, config.min_post_event_steps,
                                           event_stratified=config.event_stratified)
        if not coverage["ready_for_training"]:
            raise ValueError("dataset is not ready for fracture/support-release training: " +
                             json.dumps(coverage["reasons"], allow_nan=False))
    fracture_positive = sum(int(np.count_nonzero(~item.broken[:-1] & item.broken[1:])) for item in training)
    fracture_negative = sum(int(np.count_nonzero(~item.broken[:-1] & ~item.broken[1:])) for item in training)
    support_positive = sum(int(np.count_nonzero(item.fixed[:-1] & ~item.fixed[1:])) for item in training)
    support_negative = sum(int(np.count_nonzero(item.fixed[:-1] & item.fixed[1:])) for item in training)
    fracture_positive_weight = max(1.0, fracture_negative / fracture_positive) if fracture_positive else 1.0
    support_positive_weight = max(1.0, support_negative / support_positive) if support_positive else 1.0
    seed_all(config.seed)
    device = choose_device(config.device)
    if device.type == "cpu":
        torch.set_num_threads(min(torch.get_num_threads(), 4))
    model_config = dict(hidden=config.hidden, heads=config.heads, graph_layers=config.graph_layers,
                        transformer_layers=config.transformer_layers, learn_support_release=config.learn_support_release,
                        flow_architecture=config.flow_architecture, flow_layers=config.flow_layers,
                        **_scales(training))
    model = TemporalDynamicsModel(**model_config).to(device)
    model.fracture_logit_offset = float(np.log(fracture_positive_weight))
    model.support_logit_offset = float(np.log(support_positive_weight))
    directory = new_directory(output_dir)
    manifest = {name: [item.case_id for item in rows] for name, rows in splits.items()}
    conditions = {name: sorted({item.condition_digest for item in rows}) for name, rows in splits.items()}
    provenance = {item.case_id: item.metadata for item in trajectories}
    write_json(directory / "split.json", dict(cases=manifest, conditions=conditions,
                                              unit="complete loading condition", overfit_debug=config.overfit,
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
                  "training_mesh_fingerprints": sorted({item.topology.fingerprint for item in training})}
    started = perf_counter()

    def train_stage(network, epochs, stage, fixed_predictor=None):
        optimizer = torch.optim.AdamW(network.parameters(), lr=config.learning_rate)
        best, best_state = float("inf"), None
        for epoch in range(epochs):
            scores = {}
            for name in ("train", "validation"):
                is_training = name == "train"
                network.train(is_training)
                total, components = [], []
                for trajectory in splits[name]:
                    order = np.random.permutation(len(trajectory)) if is_training else range(len(trajectory))
                    for frame in order:
                        batch = transition_batch(trajectory, int(frame), device)
                        item = trajectory.transition(int(frame))
                        target_positions = torch.tensor(item.target_positions.copy(), dtype=torch.float32, device=device)
                        target_velocities = torch.tensor(item.target_velocities.copy(), dtype=torch.float32, device=device)
                        target_broken = torch.tensor(item.target_broken.copy(), dtype=torch.bool, device=device)
                        target_fixed = torch.tensor(item.target_fixed.copy(), dtype=torch.bool, device=device)
                        proposal = None
                        if fixed_predictor is not None:
                            with torch.no_grad():
                                proposal = fixed_predictor(batch)
                        with torch.set_grad_enabled(is_training):
                            loss, terms = _loss(network, batch, target_positions, target_velocities, target_broken, target_fixed,
                                                config, proposal, flow=fixed_predictor is None,
                                                fracture_positive_weight=fracture_positive_weight,
                                                support_positive_weight=support_positive_weight)
                            if not torch.isfinite(loss):
                                raise FloatingPointError("nonfinite temporal dynamics training loss")
                            if is_training:
                                optimizer.zero_grad(set_to_none=True)
                                loss.backward()
                                torch.nn.utils.clip_grad_norm_(network.parameters(), 1.0, error_if_nonfinite=True)
                                optimizer.step()
                        total.append(float(loss.detach()))
                        components.append(terms)
                scores[name] = {"loss": float(np.mean(total)),
                                **{key: float(np.mean([item[key] for item in components])) for key in components[0]}} if total else None
            criterion = (scores["validation"] or scores["train"])["loss"]
            record = dict(stage=stage, epoch=epoch + 1, losses=scores, seconds=perf_counter() - started,
                          device=str(device), overfit_debug=config.overfit)
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
