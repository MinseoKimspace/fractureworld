from copy import deepcopy
import hashlib
import importlib.metadata
import inspect
import json
from pathlib import Path
import pickle

import numpy as np
import torch

from .dynamics_model import TemporalDynamicsModel
from .io import file_hash


FORMAT = "fracture-world.corrector-baseline.v1"
LOSS_NAMES = ("loss", "motion", "fracture", "flow", "support_release")


def _digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()).hexdigest()


def _model_digest(state):
    digest = hashlib.sha256()
    for name, tensor in sorted(state.items()):
        value = tensor.detach().cpu().contiguous()
        digest.update(json.dumps([name, str(value.dtype), list(value.shape)]).encode())
        digest.update(value.reshape(-1).view(torch.uint8).numpy().tobytes())
    return digest.hexdigest()


def _source_digest(source):
    sources = {}
    for case_id, metadata in source.items():
        values = dict(metadata)
        values.pop("source_dataset", None)
        values["source_hashes"] = sorted(values["source_hashes"].values())
        sources[case_id] = values
    return _digest(sources)


def _model_config(saved):
    defaults = {name: parameter.default for name, parameter in inspect.signature(TemporalDynamicsModel).parameters.items()}
    values = {**defaults, "flow_architecture": "independent", **saved}
    values["material_scale"] = list(values["material_scale"])
    return values


def _input_context(checkpoint, training, split, windows, device_type):
    return dict(model_sha256=_model_digest(checkpoint["model"]), model_config=_model_config(checkpoint["model_config"]),
                source_sha256=_source_digest(checkpoint["source"]), splits=checkpoint["splits"],
                condition_splits=checkpoint["condition_splits"], class_weights=checkpoint["class_weights"],
                probability_policy=checkpoint.get("probability_policy", "legacy_weighted_logits"),
                split=split, windows=windows, device_type=device_type,
                rollout_steps=training.get("rollout_steps", 1), batch_size=training.get("batch_size", 1),
                fracture_weight=training.get("fracture_weight", 1.0), seed=training.get("seed", 17),
                learn_support_release=training.get("learn_support_release", False))


def baseline_context(checkpoint, training, split, windows, device):
    package = Path(__file__).parent
    execution = dict(implementation={name: file_hash(package / name) for name in (
        "dynamics_baseline.py", "dynamics_learning.py", "dynamics_model.py", "dynamics_data.py")},
        versions={name: importlib.metadata.version(name) for name in ("torch", "torch-geometric", "numpy", "scipy")},
        float32_matmul_precision=torch.get_float32_matmul_precision(),
        matmul_tf32=torch.backends.cuda.matmul.allow_tf32, cudnn_tf32=torch.backends.cudnn.allow_tf32,
        deterministic=torch.are_deterministic_algorithms_enabled())
    if device.type == "cuda":
        execution.update(device_name=torch.cuda.get_device_name(device), capability=list(torch.cuda.get_device_capability(device)))
    return dict(inputs=_input_context(checkpoint, training, split, windows, device.type), execution=execution)


def baseline_cache_path(output_dir, context):
    return Path(output_dir).parent / ".baseline-cache" / f"{_digest(context)}.pt"


def _valid_baseline(baseline, context):
    if not isinstance(baseline, dict):
        return False
    epoch = baseline.get("epoch")
    if (baseline.get("stage") != "neural_corrector" or not isinstance(epoch, int) or isinstance(epoch, bool) or epoch != 0 or
            baseline.get("split") != context["inputs"]["split"]):
        return False
    losses, timing = baseline.get("losses"), baseline.get("timing")
    if not isinstance(losses, dict) or not isinstance(timing, dict):
        return False
    windows = timing.get("windows")
    if not isinstance(windows, int) or isinstance(windows, bool) or windows != context["inputs"]["windows"]:
        return False
    values = [losses.get(name) for name in LOSS_NAMES]
    return (all(isinstance(value, (int, float)) and not isinstance(value, bool) and np.isfinite(value) and value >= 0
                for value in values) and losses["flow"] == 0)


def read_cached_baseline(path, context):
    try:
        cached = torch.load(path, map_location="cpu", weights_only=True)
        if not isinstance(cached, dict) or cached.get("format") != FORMAT or cached.get("context") != context:
            return None
        baseline = cached.get("baseline")
        return deepcopy(baseline) if _valid_baseline(baseline, context) else None
    except (OSError, ValueError, TypeError, EOFError, RuntimeError, pickle.UnpicklingError):
        return None


def read_checkpoint_baseline(path, context):
    checkpoint = torch.load(path, map_location="cpu", weights_only=True)
    if not isinstance(checkpoint, dict) or checkpoint.get("format") != "fracture-world.temporal-dynamics.v1":
        raise ValueError("baseline checkpoint must contain a temporal dynamics model")
    recorded = checkpoint.get("training_baseline_context")
    legacy = recorded is None
    if legacy:
        training = checkpoint.get("training", {})
        history = checkpoint.get("batch_size_changes", [])
        if not isinstance(training, dict) or not isinstance(history, list) or any(not isinstance(change, dict) for change in history):
            raise ValueError("baseline checkpoint contains invalid training provenance")
        training = dict(training)
        changes = [change for change in history if change.get("stage") == "neural_corrector"]
        if changes:
            training["batch_size"] = changes[0]["previous_batch_size"]
        device = checkpoint.get("training_state", {}).get("device_type", str(training.get("device", "auto")).split(":")[0])
        if device == "auto":
            raise ValueError("legacy baseline checkpoint lacks the actual device type; use its last.pt or evaluate the baseline once")
        try:
            inputs = _input_context(checkpoint, training, context["inputs"]["split"], context["inputs"]["windows"], device)
        except (KeyError, TypeError, ValueError) as error:
            raise ValueError("baseline checkpoint lacks the input provenance required for reuse") from error
        expected = context["inputs"]
    else:
        inputs, expected = recorded, context
    if not isinstance(inputs, dict):
        raise ValueError("baseline checkpoint contains an invalid evaluation context")
    mismatches = [name for name, value in expected.items() if inputs.get(name) != value]
    if mismatches:
        raise ValueError("baseline checkpoint does not match " + ", ".join(mismatches))
    baseline = checkpoint.get("training_baseline")
    if not _valid_baseline(baseline, context):
        raise ValueError("baseline checkpoint must contain a complete finite epoch-zero evaluation")
    baseline = deepcopy(baseline)
    previous_origin = baseline.get("origin")
    legacy_runtime = legacy or (isinstance(previous_origin, dict) and previous_origin.get("legacy_runtime_unrecorded", False))
    baseline["origin"] = dict(checkpoint=str(Path(path).resolve()), sha256=file_hash(path), legacy_runtime_unrecorded=legacy_runtime)
    if isinstance(previous_origin, dict):
        baseline["origin"]["previous"] = previous_origin
    return baseline
