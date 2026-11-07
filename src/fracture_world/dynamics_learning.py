from dataclasses import asdict, dataclass, fields
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import random
import tempfile
from time import perf_counter

import numpy as np
import torch
from torch.nn import functional as F

from . import dynamics_baseline
from .dynamics_data import load_dynamics_dataset, split_dynamics_cases, supervision_counts
from .dynamics_model import NeuralDynamicsCorrector, TemporalDynamicsModel
from .io import file_hash, new_directory, write_json
from .learning import choose_device, seed_all


FORMAT = "fracture-world.temporal-dynamics.v1"
MATERIAL_KEYS = ("rest_density", "youngs_modulus", "poissons_ratio", "fracture_stress_threshold")


def _atomic_checkpoint(path, checkpoint):
    path = Path(path)
    temporary = None
    try:
        with tempfile.NamedTemporaryFile(dir=path.parent, prefix=f".{path.name}.", suffix=".tmp", delete=False) as stream:
            temporary = Path(stream.name)
            torch.save(checkpoint, stream)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)


def _random_state():
    numpy_state = np.random.get_state()
    return dict(python=random.getstate(), numpy=[numpy_state[0], numpy_state[1].tolist(), *numpy_state[2:]],
                torch=torch.get_rng_state(), cuda=torch.cuda.get_rng_state_all() if torch.cuda.is_available() else [])


def _restore_random_state(state):
    random.setstate(state["python"])
    numpy_state = state["numpy"]
    np.random.set_state((numpy_state[0], np.asarray(numpy_state[1], dtype=np.uint32), *numpy_state[2:]))
    torch.set_rng_state(state["torch"].cpu())
    if state["cuda"]:
        if not torch.cuda.is_available() or len(state["cuda"]) != torch.cuda.device_count():
            raise ValueError("resuming this checkpoint requires the same number of CUDA devices")
        torch.cuda.set_rng_state_all([value.cpu() for value in state["cuda"]])


def _same_training_sources(recorded, current, *, allow_relocated=False):
    if not isinstance(recorded, dict) or recorded.keys() != current.keys():
        return False
    for case_id, metadata in current.items():
        previous = dict(recorded[case_id]) if isinstance(recorded[case_id], dict) else recorded[case_id]
        expected = dict(metadata)
        if isinstance(previous, dict) and "native_settings" not in previous:
            expected.pop("native_settings", None)
        if allow_relocated and isinstance(previous, dict):
            for values in (previous, expected):
                values.pop("source_dataset", None)
                hashes = values.get("source_hashes")
                if not isinstance(hashes, dict) or not hashes:
                    return False
                values["source_hashes"] = sorted(hashes.values())
        if previous != expected:
            return False
    return True


def _validation_generator(case_id, frame, seed, device):
    digest = hashlib.sha256(f"{seed}:{case_id}:{frame}".encode()).digest()
    return torch.Generator(device=device).manual_seed(int.from_bytes(digest[:8], "little") % (2**63 - 1))


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
    batch_size: int = 1
    lr_patience: int = 0
    lr_factor: float = 0.5
    min_learning_rate: float = 1.e-6
    early_stopping_patience: int = 0
    min_improvement: float = 0.0


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

    def update(self, loss, terms, count=1):
        self.last = torch.stack([loss.detach(), *(terms[key] for key in self.names[1:])]).to(torch.float64)
        self.sums.add_(self.last * count)
        self.count += count

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
          flow=True, fracture_positive_weight=1.0, support_positive_weight=1.0, return_output=False, flow_generator=None,
          output=None):
    context = None
    if output is None:
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
            noise = torch.randn(target.shape, dtype=target.dtype, device=target.device, generator=flow_generator)
            time = torch.rand((), device=target.device, generator=flow_generator)
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
                  fracture_positive_weight=1.0, support_positive_weight=1.0, prepared=None, flow_generator=None, flow=True):
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
                                    flow=flow and fixed_predictor is None, fracture_positive_weight=fracture_positive_weight,
                                    support_positive_weight=support_positive_weight, return_output=True, flow_generator=flow_generator)
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


def _rollout_loss_batch(network, trajectory, frames, device, config, *, fixed_predictor=None,
                        fracture_positive_weight=1.0, support_positive_weight=1.0, prepared=None, flow_generators=None, flow=True):
    prepared = _TrainingTrajectory(trajectory, device) if prepared is None else prepared
    batches = [prepared.batch(int(frame)) for frame in frames]
    horizons = [min(config.rollout_steps, len(trajectory) - int(frame)) for frame in frames]
    losses, components = [[] for _ in frames], [[] for _ in frames]
    for offset in range(max(horizons)):
        active = [index for index, horizon in enumerate(horizons) if offset < horizon]
        inputs = [{**batches[index], "dt": float(prepared.timesteps[int(frames[index]) + offset])} for index in active]
        proposals = fixed_predictor.forward_batch(inputs) if fixed_predictor is not None else [None] * len(inputs)
        outputs = network.forward_batch(inputs, proposals) if fixed_predictor is not None else network.forward_batch(inputs)
        for index, batch, proposal, output in zip(active, inputs, proposals, outputs):
            frame = int(frames[index]) + offset
            generator = None if flow_generators is None else flow_generators[index]
            loss, terms = _loss(network, batch, *prepared.targets(frame), config, proposal,
                                flow=flow and fixed_predictor is None, fracture_positive_weight=fracture_positive_weight,
                                support_positive_weight=support_positive_weight, flow_generator=generator, output=output)
            if not torch.isfinite(loss):
                raise FloatingPointError(f"nonfinite temporal dynamics training loss in {trajectory.case_id} at frame {frame}")
            if config.rollout_steps > 1:
                _validate_autoregressive_prediction(output, trajectory.case_id, frame + 1)
            losses[index].append(loss)
            components[index].append(terms)
            batches[index] = {**batch, "positions": output["positions"], "velocities": output["velocities"],
                              "broken": batch["broken"] | output["broken"], "fixed": batch["fixed"] & output["next_fixed"]}
    window_losses = torch.stack([torch.stack(values).mean() for values in losses])
    terms = {key: torch.stack([torch.stack([value[key] for value in window]).mean(dtype=torch.float64)
                              for window in components]).mean() for key in components[0][0]}
    return window_losses.mean(), terms


def _checkpoint_splits(trajectories, checkpoint):
    names = ("train", "validation", "test")
    manifest = checkpoint.get("splits")
    conditions = checkpoint.get("condition_splits")
    if not isinstance(manifest, dict) or not isinstance(conditions, dict):
        raise ValueError("corrector training requires recorded case and loading-condition splits")
    if any(not isinstance(manifest.get(name), list) or not isinstance(conditions.get(name), list) for name in names):
        raise ValueError("checkpoint must record train, validation and test splits")
    assigned = [case for name in names for case in manifest[name]]
    if any(not isinstance(case, str) for case in assigned) or len(set(assigned)) != len(assigned):
        raise ValueError("checkpoint case splits must contain unique case IDs")
    indexed = {item.case_id: item for item in trajectories}
    if len(indexed) != len(trajectories) or set(indexed) != set(assigned):
        raise ValueError("corrector data must match every recorded checkpoint case")
    splits = {name: [indexed[case] for case in manifest[name]] for name in names}
    observed = {name: {item.condition_digest for item in rows} for name, rows in splits.items()}
    if any(observed[name] != set(conditions[name]) for name in names):
        raise ValueError("corrector data loading conditions must match the checkpoint")
    for index, name in enumerate(names):
        for other in names[index + 1:]:
            if observed[name].intersection(observed[other]):
                raise ValueError("checkpoint loading-condition splits overlap")
            if checkpoint.get("generalization_protocol") == "shape_family":
                if {item.shape_family for item in splits[name]}.intersection(item.shape_family for item in splits[other]):
                    raise ValueError("checkpoint shape-family splits overlap")
    recorded = checkpoint.get("source")
    current = {item.case_id: item.metadata for item in trajectories}
    if recorded is not None and not _same_training_sources(recorded, current, allow_relocated=True):
        raise ValueError("corrector data archives or physical metadata differ from the checkpoint")
    return splits


def train_neural_corrector(data_dir, checkpoint_path, output_dir, *, epochs, device="auto", stream_data=False,
                           rollout_steps=None, learning_rate=None, batch_size=None, lr_patience=None, lr_factor=None,
                           min_learning_rate=None, early_stopping_patience=None, min_improvement=None, resume=None,
                           baseline_checkpoint=None, skip_baseline=None):
    if isinstance(epochs, bool) or not isinstance(epochs, int) or epochs < 1:
        raise ValueError("corrector epochs must be a positive integer")
    model, _, checkpoint, _ = load_dynamics_model(checkpoint_path, "cpu")
    saved = checkpoint.get("training", {})
    if resume is not None:
        resumed = torch.load(Path(resume), map_location="cpu", weights_only=True)
        if (resumed.get("training_mode") != "neural_corrector_only" or
                resumed.get("training_state", {}).get("stage") != "neural_corrector"):
            raise ValueError("corrector resume requires a corrector-only last.pt checkpoint")
        if resumed.get("predictor_checkpoint_sha256") != file_hash(checkpoint_path):
            raise ValueError("resuming requires the original frozen predictor checkpoint")
        saved = resumed["training"]
    options = {field.name: saved[field.name] for field in fields(DynamicsTrainingConfig) if field.name in saved}
    for name in ("hidden", "heads", "graph_layers", "transformer_layers", "flow_architecture", "flow_layers",
                 "motion_prior", "learn_support_release"):
        options[name] = model.config[name]
    options.setdefault("epochs", 1)
    options.update(corrector_epochs=epochs, device=device, stream_data=stream_data)
    options["split_unit"] = checkpoint.get("generalization_protocol", options.get("split_unit", "loading_condition"))
    overrides = dict(rollout_steps=rollout_steps, learning_rate=learning_rate, batch_size=batch_size,
                     lr_patience=lr_patience, lr_factor=lr_factor, min_learning_rate=min_learning_rate,
                     early_stopping_patience=early_stopping_patience, min_improvement=min_improvement)
    options.update({name: value for name, value in overrides.items() if value is not None})
    config = DynamicsTrainingConfig(**options)
    return train_dynamics(data_dir, output_dir, config, resume=resume,
                          predictor_checkpoint=checkpoint_path if resume is None else None,
                          baseline_checkpoint=baseline_checkpoint, skip_baseline=skip_baseline)


def train_dynamics(data_dir, output_dir, config, *, resume=None, predictor_checkpoint=None, baseline_checkpoint=None,
                   skip_baseline=None):
    if not isinstance(config, DynamicsTrainingConfig):
        raise TypeError("config must be DynamicsTrainingConfig")
    if config.epochs < 1 or config.corrector_epochs < 0 or config.learning_rate <= 0:
        raise ValueError("epochs and learning rate must be positive; corrector_epochs must be nonnegative")
    if isinstance(config.rollout_steps, bool) or not isinstance(config.rollout_steps, int) or config.rollout_steps < 1:
        raise ValueError("rollout_steps must be a positive integer")
    for name in ("batch_size", "lr_patience", "early_stopping_patience"):
        value = getattr(config, name)
        if isinstance(value, bool) or not isinstance(value, int) or value < (1 if name == "batch_size" else 0):
            raise ValueError(f"{name} must be a {'positive' if name == 'batch_size' else 'nonnegative'} integer")
    if (not np.isfinite([config.lr_factor, config.min_learning_rate, config.min_improvement]).all() or
            not 0 < config.lr_factor < 1 or config.min_learning_rate <= 0 or not 0 <= config.min_improvement < 1):
        raise ValueError("learning-rate factor, minimum learning rate and relative improvement must be finite and admissible")
    if config.lr_patience and config.min_learning_rate > config.learning_rate:
        raise ValueError("minimum learning rate cannot exceed the initial learning rate")
    if not np.isfinite([config.learning_rate, config.fracture_weight, config.flow_weight]).all():
        raise ValueError("training weights must be finite")
    if config.fracture_weight <= 0 or config.flow_weight <= 0:
        raise ValueError("fracture and flow weights must be positive")
    if predictor_checkpoint is not None and (resume is not None or config.corrector_epochs < 1):
        raise ValueError("a pretrained predictor requires a fresh corrector run with positive corrector epochs")
    if baseline_checkpoint is not None and (resume is not None or predictor_checkpoint is None):
        raise ValueError("a baseline checkpoint requires a fresh corrector-only run")
    if skip_baseline is not None and not isinstance(skip_baseline, bool):
        raise TypeError("skip_baseline must be boolean")
    if skip_baseline and (baseline_checkpoint is not None or config.corrector_epochs < 1):
        raise ValueError("skipping the baseline requires corrector training without a baseline checkpoint")
    resumed = torch.load(Path(resume), map_location="cpu", weights_only=True) if resume is not None else None
    if resumed is not None:
        if resumed.get("format") != FORMAT or "training_state" not in resumed:
            raise ValueError("resume requires a resumable last.pt checkpoint; legacy model weights remain valid for inference")
        previous = asdict(DynamicsTrainingConfig(**resumed["training"]))
        variable_options = {"epochs", "corrector_epochs", "device", "stream_data", "batch_size"}
        if any(previous.get(key) != value for key, value in asdict(config).items() if key not in variable_options):
            raise ValueError("resuming requires unchanged model, loss, split, seed and rollout settings")
        state = resumed["training_state"]
        if state["stage"] not in {"predictor", "neural_corrector"}:
            raise ValueError("unsupported resumable training stage")
        target_epochs = config.epochs if state["stage"] == "predictor" else config.corrector_epochs
        if target_epochs < state["completed_epoch"]:
            raise ValueError("target epoch count cannot precede the saved completed epoch")
        if state["stage"] == "neural_corrector" and config.epochs != previous["epochs"]:
            raise ValueError("predictor epochs cannot change after neural-corrector training has begun")
        previous_skip = resumed.get("baseline_policy") == "skipped"
        if skip_baseline is not None and skip_baseline != previous_skip:
            raise ValueError("resuming requires the recorded baseline policy")
        skip_baseline = previous_skip
    skip_baseline = bool(skip_baseline)
    setup_started = perf_counter()

    def setup_progress(update):
        print(json.dumps(dict(event="training_setup", timestamp=datetime.now(timezone.utc).isoformat(),
                              elapsed_seconds=perf_counter() - setup_started, **update), allow_nan=False), flush=True)

    setup_progress(dict(phase="initialization", status="started", data=str(Path(data_dir).resolve())))
    device = choose_device(config.device)
    if resumed is not None and resumed["training_state"]["device_type"] != device.type:
        raise ValueError("resuming requires the same training device type")
    if device.type == "cpu":
        torch.set_num_threads(min(torch.get_num_threads(), 4))
    frozen_checkpoint, model = None, None
    frozen_path = predictor_checkpoint
    if resumed is not None and resumed.get("training_mode") == "neural_corrector_only":
        frozen_path = resume
    if frozen_path is not None:
        model, _, frozen_checkpoint, _ = load_dynamics_model(frozen_path, device)
    options = dict(progress=setup_progress)
    if config.stream_data:
        options["stream"] = True
    trajectories = load_dynamics_dataset(data_dir, **options)
    setup_progress(dict(phase="training_inputs", status="started", cases=len(trajectories)))
    for trajectory in trajectories:
        if not config.learn_support_release and supervision_counts(trajectory)["support_positive"]:
            raise ValueError("automatic support release is not modeled; use trajectories with prescribed constant material supports")
        base_dt = trajectory.metadata["conditioning_context"]["dt"]
        if not np.allclose(np.diff(trajectory.times), base_dt, rtol=64 * np.finfo(np.float32).eps, atol=0):
            raise ValueError("temporal dynamics training requires every native step; regenerate with record_stride=1")
    setup_progress(dict(phase="training_inputs", status="completed", cases=len(trajectories)))
    splits = (_checkpoint_splits(trajectories, frozen_checkpoint) if frozen_checkpoint is not None else
              split_dynamics_cases(trajectories, config.seed, config.overfit, event_stratified=config.event_stratified,
                                   split_unit=config.split_unit))
    training = splits["train"]
    if not training:
        raise ValueError("no training trajectories")
    if (config.lr_patience or config.early_stopping_patience) and not splits["validation"]:
        raise ValueError("learning-rate scheduling and early stopping require a held-out validation split")
    coverage = None
    if config.require_event_coverage and frozen_checkpoint is None:
        from .dynamics_readiness import inspect_dynamics_dataset

        coverage = inspect_dynamics_dataset(data_dir, config.seed, config.overfit, config.min_post_event_steps,
                                           event_stratified=config.event_stratified, split_unit=config.split_unit,
                                           trajectories=trajectories, progress=setup_progress)
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
    if frozen_checkpoint is not None:
        weights = frozen_checkpoint.get("class_weights", {})
        fracture_positive_weight = float(weights.get("fracture_positive", fracture_positive_weight))
        support_positive_weight = float(weights.get("support_release_positive", support_positive_weight))
    seed_all(config.seed)
    setup_progress(dict(phase="normalization", status="started", cases=len(training)))
    model_config = (dict(model.config) if frozen_checkpoint is not None else
                    dict(hidden=config.hidden, heads=config.heads, graph_layers=config.graph_layers,
                         transformer_layers=config.transformer_layers, learn_support_release=config.learn_support_release,
                         flow_architecture=config.flow_architecture, flow_layers=config.flow_layers, motion_prior=config.motion_prior,
                         **_scales(training)))
    setup_progress(dict(phase="model_initialization", status="started", device=str(device)))
    if frozen_checkpoint is None:
        model = TemporalDynamicsModel(**model_config).to(device)
        model.fracture_logit_offset = float(np.log(fracture_positive_weight))
        model.support_logit_offset = float(np.log(support_positive_weight))
    manifest = {name: [item.case_id for item in rows] for name, rows in splits.items()}
    conditions = {name: sorted({item.condition_digest for item in rows}) for name, rows in splits.items()}
    shape_families = {name: sorted({item.shape_family for item in rows}) for name, rows in splits.items()}
    mesh_fingerprints = {name: sorted({item.topology.fingerprint for item in rows}) for name, rows in splits.items()}
    provenance = {item.case_id: item.metadata for item in trajectories}
    if resumed is not None:
        matching = {"model_config": model_config, "splits": manifest, "condition_splits": conditions,
                    "class_weights": dict(fracture_positive=fracture_positive_weight, support_release_positive=support_positive_weight)}
        if (any(resumed.get(key) != value for key, value in matching.items()) or
                not _same_training_sources(resumed.get("source"), provenance, allow_relocated=frozen_checkpoint is not None)):
            raise ValueError("resume checkpoint does not match the dataset, model normalization or supervision")
    directory = Path(output_dir).resolve()
    if resumed is None or directory != Path(resume).resolve().parent:
        directory = new_directory(directory)
    elif not directory.is_dir():
        raise ValueError("resume output must be a directory")
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
                  "validation_flow_policy": "fixed independent Gaussian noise and flow time per seed, case and window",
                  "checkpoint_policy": "atomic best-on-improvement and resumable completed-epoch state",
                  "training_mesh_fingerprints": mesh_fingerprints["train"],
                  "evaluation_mesh_fingerprints": sorted(set(mesh_fingerprints["validation"] + mesh_fingerprints["test"])),
                  "shape_family_splits": shape_families, "mesh_fingerprint_splits": mesh_fingerprints,
                  "mesh_shape_families": {item.topology.fingerprint: item.shape_family for item in trajectories},
                  "generalization_protocol": config.split_unit}
    if frozen_checkpoint is not None:
        checkpoint.update(model=frozen_checkpoint["model"], training_mode="neural_corrector_only",
                          predictor_training=frozen_checkpoint.get("predictor_training", frozen_checkpoint.get("training", {})),
                          predictor_checkpoint_sha256=(frozen_checkpoint["predictor_checkpoint_sha256"] if resumed is not None
                                                       else file_hash(predictor_checkpoint)),
                          probability_policy=frozen_checkpoint.get("probability_policy", "legacy_weighted_logits"))
    if resumed is not None and "batch_size_changes" in resumed:
        checkpoint["batch_size_changes"] = list(resumed["batch_size_changes"])
    if config.corrector_epochs:
        checkpoint["baseline_policy"] = "skipped" if skip_baseline else "evaluated"
    started = perf_counter()
    setup_progress(dict(phase="ready", status="completed", device=str(device),
                        splits={name: len(rows) for name, rows in splits.items()},
                        baseline_policy=checkpoint.get("baseline_policy")))

    def train_stage(network, epochs, stage, fixed_predictor=None):
        optimizer = torch.optim.AdamW(network.parameters(), lr=config.learning_rate)
        scheduler = (torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode="min", factor=config.lr_factor, patience=config.lr_patience,
            threshold=config.min_improvement, threshold_mode="rel", min_lr=config.min_learning_rate)
            if config.lr_patience else None)
        if fixed_predictor is not None:
            fixed_predictor.requires_grad_(False)
            fixed_predictor.eval()
            network._graph_cache = fixed_predictor._graph_cache
        best, best_state, first_epoch = float("inf"), None, 0
        significant_best, bad_epochs, stop_reason = float("inf"), 0, None
        selection = None
        key = "model" if stage == "predictor" else "corrector"

        def run_split(name, epoch, is_training, baseline=False):
            split_started = last_progress = perf_counter()
            active_network = fixed_predictor if baseline else network
            active_network.train(is_training)
            metrics = _TrainingMetrics(device)
            window_count = sum(len(item) for item in splits[name])
            prepare_seconds, optimizer_steps = 0.0, 0
            interval_started, interval_windows = split_started, 0
            for case_index, trajectory in enumerate(splits[name], start=1):
                prepare_started = perf_counter()
                prepared = _TrainingTrajectory(trajectory, device)
                prepare_seconds += perf_counter() - prepare_started
                order = np.random.permutation(len(trajectory)) if is_training else range(len(trajectory))
                for begin in range(0, len(order), config.batch_size):
                    frames = order[begin:begin + config.batch_size]
                    generators = None if is_training else [
                        _validation_generator(trajectory.case_id, int(frame), config.seed, device) for frame in frames]
                    with torch.set_grad_enabled(is_training):
                        if is_training:
                            optimizer.zero_grad(set_to_none=True)
                        options = dict(fixed_predictor=None if baseline else fixed_predictor, fracture_positive_weight=fracture_positive_weight,
                                       support_positive_weight=support_positive_weight, prepared=prepared)
                        if baseline:
                            options["flow"] = False
                        if len(frames) == 1:
                            loss, terms = _rollout_loss(active_network, trajectory, int(frames[0]), device, config,
                                                        flow_generator=None if generators is None else generators[0], **options)
                        else:
                            loss, terms = _rollout_loss_batch(active_network, trajectory, frames, device, config,
                                                              flow_generators=generators, **options)
                        if is_training:
                            loss.backward()
                            torch.nn.utils.clip_grad_norm_(network.parameters(), 1.0, error_if_nonfinite=True)
                            optimizer.step()
                            optimizer_steps += 1
                    metrics.update(loss, terms, len(frames))
                    if perf_counter() - last_progress >= 5:
                        current_loss = float(metrics.last[0])
                        now = perf_counter()
                        elapsed = now - split_started
                        seconds_per_window = elapsed / metrics.count
                        interval_count = metrics.count - interval_windows
                        print(json.dumps(dict(event="training_progress", stage=stage, epoch=epoch,
                                              timestamp=datetime.now(timezone.utc).isoformat(),
                                              split=name, case=trajectory.case_id, case_index=case_index,
                                              cases=len(splits[name]), transition=begin + len(frames),
                                              transitions=len(trajectory), rollout_steps=config.rollout_steps,
                                              batch_size=len(frames), optimizer_steps=optimizer_steps,
                                              learning_rate=optimizer.param_groups[0]["lr"],
                                              gpu_peak_memory_bytes=(torch.cuda.max_memory_allocated(device)
                                                                     if device.type == "cuda" else None),
                                              windows_completed=metrics.count, windows_total=window_count,
                                              elapsed_seconds=elapsed, seconds_per_window=seconds_per_window,
                                              interval_seconds_per_window=(now - interval_started) / interval_count,
                                              eta_seconds=(window_count - metrics.count) * seconds_per_window,
                                              data_prepare_seconds=prepare_seconds,
                                              loss=current_loss, device=str(device)), allow_nan=False), flush=True)
                        last_progress = interval_started = perf_counter()
                        interval_windows = metrics.count
                del prepared
            timing = dict(seconds=perf_counter() - split_started, windows=metrics.count,
                          optimizer_steps=optimizer_steps, data_prepare_seconds=prepare_seconds)
            if metrics.count:
                timing["seconds_per_window"] = timing["seconds"] / metrics.count
            return metrics.scores(), timing

        if resumed is not None and resumed["training_state"]["stage"] == stage:
            saved = resumed["training_state"]
            network.load_state_dict(resumed[key], strict=True)
            optimizer.load_state_dict(saved["optimizer"])
            best, best_state, first_epoch = saved["best_loss"], saved["best_state"], saved["completed_epoch"]
            significant_best, bad_epochs = saved.get("significant_best_loss", best), saved.get("bad_epochs", 0)
            stop_reason = saved.get("stop_reason")
            if scheduler is not None and saved.get("scheduler") is not None:
                scheduler.load_state_dict(saved["scheduler"])
            selection = resumed.get("selection")
            if "training_baseline" in resumed:
                checkpoint["training_baseline"] = resumed["training_baseline"]
            if "training_baseline_context" in resumed:
                checkpoint["training_baseline_context"] = resumed["training_baseline_context"]
            previous_batch_size = resumed["training"].get("batch_size", 1)
            if previous_batch_size != config.batch_size:
                change = dict(stage=stage, after_completed_epoch=first_epoch,
                              previous_batch_size=previous_batch_size, batch_size=config.batch_size)
                checkpoint.setdefault("batch_size_changes", []).append(change)
            _restore_random_state(saved["random_state"])
            _atomic_checkpoint(directory / "best.pt", {**checkpoint, key: best_state, "selection": selection})
            setup_progress(dict(phase="resume", status="completed", stage=stage, completed_epoch=first_epoch,
                                target_epochs=epochs, checkpoint=str(Path(resume).resolve()),
                                previous_batch_size=previous_batch_size, batch_size=config.batch_size))
        elif fixed_predictor is not None and not skip_baseline:
            baseline_split = "validation" if splits["validation"] else "train"
            windows = sum(len(item) for item in splits[baseline_split])
            context = dynamics_baseline.baseline_context(checkpoint, asdict(config), baseline_split, windows, device)
            cache_path = dynamics_baseline.baseline_cache_path(directory, context)
            baseline_source = "cache"
            if baseline_checkpoint is not None:
                baseline = dynamics_baseline.read_checkpoint_baseline(baseline_checkpoint, context)
                baseline_source = "checkpoint"
            else:
                baseline = dynamics_baseline.read_cached_baseline(cache_path, context)
            if baseline is None:
                scores, timing = run_split(baseline_split, 0, False, baseline=True)
                baseline = dict(stage=stage, epoch=0, split=baseline_split, losses=scores, timing=timing)
                baseline_source = "evaluated"
            cache_path.parent.mkdir(parents=True, exist_ok=True)
            _atomic_checkpoint(cache_path, dict(format=dynamics_baseline.FORMAT, context=context, baseline=baseline))
            best = significant_best = baseline["losses"]["loss"]
            best_state = {name: value.detach().cpu().clone() for name, value in network.state_dict().items()}
            checkpoint["training_baseline"] = baseline
            checkpoint["training_baseline_context"] = context
            selection = dict(stage=stage, epoch=0, loss=best, split=baseline_split)
            if scheduler is not None:
                scheduler.step(best)
            _atomic_checkpoint(directory / "best.pt", {**checkpoint, key: best_state, "selection": selection})
            print(json.dumps(dict(event="training_baseline", source=baseline_source, cache=str(cache_path), **baseline),
                             allow_nan=False), flush=True)
        elif fixed_predictor is not None:
            print(json.dumps(dict(event="training_baseline", source="skipped", stage=stage)), flush=True)
        if stop_reason == "validation_plateau":
            network.load_state_dict(best_state)
            network.eval()
            checkpoint["selection"] = selection
            return best_state
        for epoch in range(first_epoch, epochs):
            epoch_started = perf_counter()
            if device.type == "cuda":
                torch.cuda.reset_peak_memory_stats(device)
            scores, timings = {}, {}
            learning_rate = optimizer.param_groups[0]["lr"]
            for name in ("train", "validation"):
                scores[name], timings[name] = run_split(name, epoch + 1, name == "train")
            criterion = (scores["validation"] or scores["train"])["loss"]
            if criterion < best:
                best = criterion
                best_state = {key: value.detach().cpu().clone() for key, value in network.state_dict().items()}
                selection = dict(stage=stage, epoch=epoch + 1, loss=best,
                                 split="validation" if scores["validation"] else "train")
                _atomic_checkpoint(directory / "best.pt", {**checkpoint, key: best_state, "selection": selection})
            if criterion < significant_best * (1 - config.min_improvement):
                significant_best, bad_epochs = criterion, 0
            else:
                bad_epochs += 1
            if scheduler is not None:
                scheduler.step(criterion)
            stop_reason = ("validation_plateau" if config.early_stopping_patience and bad_epochs >= config.early_stopping_patience
                           else "epoch_budget" if epoch + 1 == epochs else None)
            training_state = dict(stage=stage, completed_epoch=epoch + 1, device_type=device.type,
                                  optimizer=optimizer.state_dict(), random_state=_random_state(),
                                  best_loss=best, best_state=best_state, significant_best_loss=significant_best,
                                  bad_epochs=bad_epochs, scheduler=None if scheduler is None else scheduler.state_dict(),
                                  stop_reason=stop_reason)
            _atomic_checkpoint(directory / "last.pt", {**checkpoint, key: network.state_dict(), "selection": selection,
                                                        "epoch": epoch + 1,
                                                        "training_state": training_state})
            record = dict(stage=stage, epoch=epoch + 1, losses=scores, seconds=perf_counter() - started,
                          epoch_seconds=perf_counter() - epoch_started, split_timing=timings, batch_size=config.batch_size,
                          learning_rate=learning_rate, next_learning_rate=optimizer.param_groups[0]["lr"],
                          best_loss=best, significant_best_loss=significant_best, bad_epochs=bad_epochs,
                          gpu_peak_memory_bytes=(torch.cuda.max_memory_allocated(device) if device.type == "cuda" else None),
                          stop_reason=stop_reason, baseline_loss=checkpoint.get("training_baseline", {}).get("losses", {}).get("loss"),
                          device=str(device), overfit_debug=config.overfit, rollout_steps=config.rollout_steps)
            with (directory / "training.jsonl").open("a", encoding="utf-8") as stream:
                stream.write(json.dumps(record, allow_nan=False) + "\n")
            print(json.dumps(record, allow_nan=False), flush=True)
            if stop_reason == "validation_plateau":
                break
        network.load_state_dict(best_state)
        network.eval()
        checkpoint["selection"] = selection
        return best_state

    if resumed is not None and resumed["training_state"]["stage"] == "neural_corrector":
        checkpoint["model"] = resumed["model"]
        model.load_state_dict(checkpoint["model"], strict=True)
        model.eval()
    elif frozen_checkpoint is None:
        checkpoint["model"] = train_stage(model, config.epochs, "predictor")
        _atomic_checkpoint(directory / "best.pt", checkpoint)
    if config.corrector_epochs:
        corrector = NeuralDynamicsCorrector(**model_config).to(device)
        corrector.fracture_logit_offset = model.fracture_logit_offset
        corrector.support_logit_offset = model.support_logit_offset
        checkpoint["corrector"] = train_stage(corrector, config.corrector_epochs, "neural_corrector", model)
        _atomic_checkpoint(directory / "best.pt", checkpoint)
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
