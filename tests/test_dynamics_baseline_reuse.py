from copy import deepcopy
from dataclasses import replace
import json
import math
from pathlib import Path

import pytest
import torch

from fracture_world import dynamics_baseline, dynamics_learning
from fracture_world.dynamics_learning import (
    DynamicsTrainingConfig,
    _rollout_loss,
    _rollout_loss_batch,
    _scales,
    train_dynamics,
    train_neural_corrector,
)
from fracture_world.dynamics_model import NeuralDynamicsCorrector, TemporalDynamicsModel
from fracture_world.io import file_hash
from test_dynamics_data import recorded_trajectory


@pytest.fixture
def pretrained(tmp_path, monkeypatch):
    data = tmp_path / "data"
    recorded_trajectory(data)
    source = dynamics_learning.load_dynamics_dataset(data)[0]
    trajectories = [replace(source, case_id=f"case-{index}", condition_digest=f"{index:064x}")
                    for index in range(3)]
    monkeypatch.setattr(dynamics_learning, "load_dynamics_dataset", lambda *args, **kwargs: trajectories)
    config = DynamicsTrainingConfig(epochs=1, hidden=12, heads=3, graph_layers=2, device="cpu")
    path = train_dynamics(data, tmp_path / "predictor", config)
    return data, path


@pytest.fixture
def evaluated_baseline(pretrained, tmp_path):
    data, path = pretrained
    output = train_neural_corrector(data, path, tmp_path / "first", epochs=1, device="cpu", batch_size=2)
    checkpoint = torch.load(output, weights_only=True)
    context = checkpoint["training_baseline_context"]
    cache = dynamics_baseline.baseline_cache_path(output.parent, context)
    return data, path, output, checkpoint, context, cache


def watch_baseline_evaluations(monkeypatch, *, reject=False):
    calls = []
    serial = dynamics_learning._rollout_loss
    batched = dynamics_learning._rollout_loss_batch

    def tracked(function, *args, **kwargs):
        if kwargs.get("flow", True) is False and kwargs.get("fixed_predictor") is None:
            calls.append(args[2])
            if reject:
                raise AssertionError("reused baseline evaluated epoch zero again")
        return function(*args, **kwargs)

    monkeypatch.setattr(dynamics_learning, "_rollout_loss", lambda *args, **kwargs: tracked(serial, *args, **kwargs))
    monkeypatch.setattr(dynamics_learning, "_rollout_loss_batch", lambda *args, **kwargs: tracked(batched, *args, **kwargs))
    return calls


def test_fresh_corrector_reuses_cached_baseline_without_changing_weights_or_random_state(pretrained, tmp_path, monkeypatch, capsys):
    data, path = pretrained
    original_hash = file_hash(path)
    initial_states = {}
    save = dynamics_learning._atomic_checkpoint

    def capture_initial(destination, checkpoint):
        selection = checkpoint.get("selection", {})
        if Path(destination).name == "best.pt" and selection.get("epoch") == 0:
            initial_states.setdefault(Path(destination).parent.name, deepcopy(checkpoint["corrector"]))
        save(destination, checkpoint)

    monkeypatch.setattr(dynamics_learning, "_atomic_checkpoint", capture_initial)
    first = train_neural_corrector(data, path, tmp_path / "first", epochs=1, device="cpu", batch_size=2)
    original = torch.load(first.parent / "last.pt", weights_only=True)
    context = original["training_baseline_context"]
    cache = dynamics_baseline.baseline_cache_path(first.parent, context)
    assert cache.is_file()
    assert cache.parent == tmp_path / ".baseline-cache"
    assert torch.load(cache, weights_only=True).keys() == {"format", "context", "baseline"}
    first_events = [json.loads(line) for line in capsys.readouterr().out.splitlines()]
    assert next(event for event in first_events if event.get("event") == "training_baseline")["source"] == "evaluated"
    calls = watch_baseline_evaluations(monkeypatch, reject=True)
    second = train_neural_corrector(data, path, tmp_path / "second", epochs=1, device="cpu", batch_size=2)
    actual = torch.load(second.parent / "last.pt", weights_only=True)
    events = [json.loads(line) for line in capsys.readouterr().out.splitlines()]
    assert next(event for event in events if event.get("event") == "training_baseline")["source"] == "cache"
    assert calls == []
    assert initial_states.keys() == {"first", "second"}
    for name, value in initial_states["first"].items():
        torch.testing.assert_close(initial_states["second"][name], value, atol=0, rtol=0)
    for name in ("model", "corrector"):
        for key, value in original[name].items():
            torch.testing.assert_close(actual[name][key], value, atol=0, rtol=0)
    assert actual["training_baseline"] == original["training_baseline"]
    expected_random = original["training_state"]["random_state"]
    actual_random = actual["training_state"]["random_state"]
    assert actual_random["python"] == expected_random["python"]
    assert actual_random["numpy"] == expected_random["numpy"]
    torch.testing.assert_close(actual_random["torch"], expected_random["torch"], atol=0, rtol=0)
    for actual_cuda, expected_cuda in zip(actual_random["cuda"], expected_random["cuda"]):
        torch.testing.assert_close(actual_cuda, expected_cuda, atol=0, rtol=0)
    assert actual["predictor_checkpoint_sha256"] == original_hash
    assert file_hash(path) == original_hash


def test_baseline_fingerprint_rejects_changed_rollout_batch_loss_model_and_data(evaluated_baseline):
    _, _, output, checkpoint, context, cache = evaluated_baseline
    for change in ("rollout_steps", "batch_size", "fracture_weight", "model", "data"):
        altered = deepcopy(checkpoint)
        training = dict(altered["training"])
        if change in ("rollout_steps", "batch_size", "fracture_weight"):
            training[change] += 1
        elif change == "model":
            name = next(name for name, value in altered["model"].items() if value.is_floating_point())
            altered["model"][name] += 0.01
        else:
            source = next(iter(altered["source"].values()))
            name = next(iter(source["source_hashes"]))
            source["source_hashes"][name] = "changed archive digest"
        changed = dynamics_baseline.baseline_context(altered, training, "validation", context["inputs"]["windows"],
                                                    torch.device("cpu"))
        assert changed != context
        assert dynamics_baseline.baseline_cache_path(output.parent, changed) != cache
        assert dynamics_baseline.read_cached_baseline(cache, changed) is None
        with pytest.raises(ValueError, match="does not match"):
            dynamics_baseline.read_checkpoint_baseline(output, changed)


def test_legacy_checkpoint_baseline_imports_metrics_and_initializes_a_fresh_corrector(evaluated_baseline, tmp_path, monkeypatch, capsys):
    data, path, _, checkpoint, _, _ = evaluated_baseline
    legacy = deepcopy(checkpoint)
    legacy.pop("training_baseline_context")
    donor = tmp_path / "legacy-best.pt"
    torch.save(legacy, donor)
    calls = watch_baseline_evaluations(monkeypatch, reject=True)
    capsys.readouterr()
    output = train_neural_corrector(data, path, tmp_path / "imported", epochs=1, device="cpu", batch_size=2,
                                    baseline_checkpoint=donor)
    imported = torch.load(output, weights_only=True)
    baseline = imported["training_baseline"]
    assert baseline["losses"] == checkpoint["training_baseline"]["losses"]
    assert baseline["origin"] == dict(checkpoint=str(donor.resolve()), sha256=file_hash(donor), legacy_runtime_unrecorded=True)
    assert imported["training_baseline_context"]["inputs"] == checkpoint["training_baseline_context"]["inputs"]
    assert calls == []
    events = [json.loads(line) for line in capsys.readouterr().out.splitlines()]
    assert next(event for event in events if event.get("event") == "training_baseline")["source"] == "checkpoint"
    for name, value in checkpoint["corrector"].items():
        torch.testing.assert_close(imported["corrector"][name], value, atol=0, rtol=0)


def test_legacy_batch_history_preserves_the_batch_size_used_for_epoch_zero(evaluated_baseline, tmp_path):
    data, path, _, checkpoint, context, _ = evaluated_baseline
    legacy = deepcopy(checkpoint)
    legacy.pop("training_baseline_context")
    legacy["training"]["batch_size"] = 6
    legacy["batch_size_changes"] = [dict(stage="predictor", previous_batch_size=1, batch_size=4),
                                    dict(stage="neural_corrector", previous_batch_size=2, batch_size=6, after_completed_epoch=1)]
    donor = tmp_path / "resized-best.pt"
    torch.save(legacy, donor)
    reused = dynamics_baseline.read_checkpoint_baseline(donor, context)
    assert reused["losses"] == checkpoint["training_baseline"]["losses"]
    training = {**checkpoint["training"], "batch_size": 6}
    changed = dynamics_baseline.baseline_context(checkpoint, training, "validation", context["inputs"]["windows"],
                                                torch.device("cpu"))
    with pytest.raises(ValueError, match="batch_size"):
        dynamics_baseline.read_checkpoint_baseline(donor, changed)
    with pytest.raises(ValueError, match="batch_size"):
        train_neural_corrector(data, path, tmp_path / "different-batch", epochs=1, device="cpu", batch_size=6,
                                baseline_checkpoint=donor)


@pytest.mark.parametrize("corruption", ["malformed", "partial", "nonfinite", "context_tensor", "window_tensor"])
def test_invalid_cache_is_recomputed(evaluated_baseline, tmp_path, monkeypatch, corruption):
    data, path, _, checkpoint, context, cache = evaluated_baseline
    cached = torch.load(cache, weights_only=True)
    if corruption == "malformed":
        cache.write_bytes(b"interrupted checkpoint serialization")
    elif corruption == "partial":
        cached.pop("baseline")
        torch.save(cached, cache)
    elif corruption == "nonfinite":
        cached["baseline"]["losses"]["loss"] = float("nan")
        torch.save(cached, cache)
    else:
        if corruption == "context_tensor":
            cached["context"]["inputs"]["windows"] = torch.tensor([3, 4])
        else:
            cached["baseline"]["timing"]["windows"] = torch.tensor([3, 4])
        torch.save(cached, cache)
    assert dynamics_baseline.read_cached_baseline(cache, context) is None
    calls = watch_baseline_evaluations(monkeypatch)
    output = train_neural_corrector(data, path, tmp_path / "recomputed", epochs=1, device="cpu", batch_size=2)
    assert calls
    trained = torch.load(output, weights_only=True)
    assert trained["training_baseline"]["losses"] == checkpoint["training_baseline"]["losses"]
    restored = dynamics_baseline.read_cached_baseline(cache, context)
    assert restored is not None and math.isfinite(restored["losses"]["loss"])


@pytest.mark.parametrize("corruption", ["partial", "nonfinite", "context"])
def test_explicit_invalid_baseline_checkpoint_is_rejected(evaluated_baseline, tmp_path, corruption):
    data, path, _, checkpoint, _, _ = evaluated_baseline
    invalid = deepcopy(checkpoint)
    if corruption == "partial":
        invalid["training_baseline"]["losses"].pop("motion")
    elif corruption == "nonfinite":
        invalid["training_baseline"]["losses"]["loss"] = float("nan")
    else:
        invalid["training_baseline_context"]["inputs"]["batch_size"] = 6
    donor = tmp_path / "invalid-best.pt"
    torch.save(invalid, donor)
    with pytest.raises(ValueError, match="baseline checkpoint"):
        train_neural_corrector(data, path, tmp_path / "rejected", epochs=1, device="cpu", batch_size=2,
                                baseline_checkpoint=donor)
    assert not (tmp_path / "rejected" / "best.pt").exists()


def test_reimport_keeps_legacy_execution_provenance(evaluated_baseline, tmp_path):
    _, _, _, checkpoint, context, _ = evaluated_baseline
    legacy = deepcopy(checkpoint)
    legacy.pop("training_baseline_context")
    donor = tmp_path / "legacy.pt"
    torch.save(legacy, donor)
    imported = dynamics_baseline.read_checkpoint_baseline(donor, context)
    current = deepcopy(checkpoint)
    current["training_baseline"] = imported
    second_donor = tmp_path / "imported.pt"
    torch.save(current, second_donor)
    repeated = dynamics_baseline.read_checkpoint_baseline(second_donor, context)
    assert repeated["origin"]["legacy_runtime_unrecorded"] is True
    assert repeated["origin"]["previous"] == imported["origin"]


def test_legacy_auto_device_requires_a_recorded_runtime(evaluated_baseline, tmp_path):
    _, _, _, checkpoint, context, _ = evaluated_baseline
    legacy = deepcopy(checkpoint)
    legacy.pop("training_baseline_context")
    legacy["training"]["device"] = "auto"
    legacy.pop("training_state", None)
    donor = tmp_path / "unknown-device.pt"
    torch.save(legacy, donor)
    with pytest.raises(ValueError, match="actual device type"):
        dynamics_baseline.read_checkpoint_baseline(donor, context)


@pytest.mark.parametrize("motion_prior", ["backward_euler", "constant_acceleration"])
@pytest.mark.parametrize("supports", [False, True])
@pytest.mark.parametrize("batched", [False, True])
def test_predictor_only_baseline_matches_zero_corrector_for_both_priors_and_support_policies(tmp_path, motion_prior, supports, batched):
    recorded_trajectory(tmp_path / "data", supports=supports)
    trajectory = dynamics_learning.load_dynamics_dataset(tmp_path / "data")[0]
    torch.set_num_threads(min(torch.get_num_threads(), 4))
    torch.manual_seed(743)
    options = dict(hidden=12, heads=3, graph_layers=2, transformer_layers=1, motion_prior=motion_prior,
                   learn_support_release=supports, **_scales([trajectory]))
    predictor = TemporalDynamicsModel(**options).eval().requires_grad_(False)
    corrector = NeuralDynamicsCorrector(**options).eval()
    predictor.fracture_logit_offset = corrector.fracture_logit_offset = math.log(3)
    predictor.support_logit_offset = corrector.support_logit_offset = math.log(2)
    with torch.no_grad():
        predictor.motion_head.bias.fill_(0.001)
    config = DynamicsTrainingConfig(epochs=1, rollout_steps=2, learn_support_release=supports, motion_prior=motion_prior)
    state = torch.get_rng_state().clone()
    function = _rollout_loss_batch if batched else _rollout_loss
    frames = [0, len(trajectory) - 1] if batched else 0
    with torch.no_grad():
        expected, expected_terms = function(corrector, trajectory, frames, "cpu", config, fixed_predictor=predictor)
        actual, actual_terms = function(predictor, trajectory, frames, "cpu", config, flow=False)
    assert not actual.requires_grad and not expected.requires_grad
    torch.testing.assert_close(actual, expected, atol=2.e-6, rtol=1.e-5)
    for name, value in expected_terms.items():
        torch.testing.assert_close(actual_terms[name], value, atol=2.e-6, rtol=1.e-5)
    assert actual_terms["flow"] == 0
    torch.testing.assert_close(torch.get_rng_state(), state, atol=0, rtol=0)
