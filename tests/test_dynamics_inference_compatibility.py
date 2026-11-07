import json
from pathlib import Path

import pytest
import torch

from fracture_world import dynamics_learning
from fracture_world import dynamics_runtime
from fracture_world.dynamics_runtime import load_rollout_config, rollout_dynamics
from test_dynamics_runtime import portable_rollout_inputs


@pytest.mark.parametrize("policy", ["legacy_weighted_logits", "subtract_log_training_positive_weight"])
def test_predictor_loading_preserves_weights_and_ignores_optional_corrector(tmp_path, monkeypatch, policy):
    path, initial, _, _ = portable_rollout_inputs(tmp_path)
    saved = torch.load(path, map_location="cpu", weights_only=True)
    saved["probability_policy"] = policy
    saved["class_weights"] = dict(fracture_positive=3.0, support_release_positive=5.0)
    saved["corrector"] = {"unused_invalid_corrector": torch.ones(1)}
    torch.save(saved, path)

    def forbidden(*args, **kwargs):
        raise AssertionError("online inference must not construct a neural corrector")

    monkeypatch.setattr(dynamics_learning, "NeuralDynamicsCorrector", forbidden)
    model, checkpoint, device = dynamics_learning.load_dynamics_predictor(path, "cpu")
    assert device.type == "cpu"
    assert not model.training
    for name, value in model.state_dict().items():
        torch.testing.assert_close(value, saved["model"][name], rtol=0, atol=0)
    corrected = policy == "subtract_log_training_positive_weight"
    assert model.fracture_logit_offset == pytest.approx(float(torch.log(torch.tensor(3.0, dtype=torch.float64))) if corrected else 0)
    assert model.support_logit_offset == pytest.approx(float(torch.log(torch.tensor(5.0, dtype=torch.float64))) if corrected else 0)
    assert checkpoint["corrector"]["unused_invalid_corrector"].device.type == "cpu"
    report = rollout_dynamics(path, initial, tmp_path / "rollout", 2, device="cpu")
    assert report["completed_steps"] == 2
    assert report["neural_corrector_calls"] == 0
    assert report["optimization_solves"] == 0


def test_legacy_rollout_configuration_discards_removed_solver_parameter(tmp_path):
    path = tmp_path / "legacy.json"
    path.write_text(json.dumps(dict(rho=.01, fracture_samples=2, absolute_tolerance=1.e-10)))
    config = load_rollout_config(path)
    assert config.fracture_samples == 2
    assert config.fracture_policy == "marginal_search"
    assert config.absolute_tolerance == 1.e-10
    assert not hasattr(config, "rho")


@pytest.mark.parametrize("name", ["bunny-rollout", "neural-rollout", "neural-joint-rollout", "bunny-exact-reference"])
def test_bundled_rollout_configuration_loads(name):
    path = Path(__file__).parents[1] / "configs" / f"{name}.json"
    config = load_rollout_config(path)
    if name != "bunny-exact-reference":
        assert config.fracture_policy == "shared_sample"
        assert config.fracture_samples == 1


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA unavailable")
def test_cuda_rollout_reports_synchronized_stage_costs_without_optimization(tmp_path):
    checkpoint, initial, _, _ = portable_rollout_inputs(tmp_path, has_corrector=False)
    report = rollout_dynamics(checkpoint, initial, tmp_path / "cuda", 4, device="cuda")
    assert report["status"] == "completed"
    assert report["completed_steps"] == 4
    assert report["optimization_solves"] == 0
    timing = report["timing"]
    assert timing["inference_seconds"] == report["runtime_seconds"]
    assert timing["seconds_per_completed_step"] == report["runtime_seconds"] / 4
    assert all(value >= 0 for value in timing["stages_seconds"].values())
    assert 0 < sum(timing["stages_seconds"].values()) <= report["runtime_seconds"]


def test_fully_fractured_topology_skips_redundant_flow_generation(tmp_path, monkeypatch):
    checkpoint, initial, _, _ = portable_rollout_inputs(tmp_path, broken=True)

    def forbidden(*args, **kwargs):
        raise AssertionError("no intact facets remain to generate")

    monkeypatch.setattr(dynamics_learning.TemporalDynamicsModel, "sample_fracture", forbidden)
    monkeypatch.setattr(dynamics_learning.TemporalDynamicsModel, "forward_with_flow_context", forbidden)
    report = rollout_dynamics(checkpoint, initial, tmp_path / "fractured", 4, device="cpu")
    assert report["status"] == "completed"
    assert report["predictor_calls"] == 4
    assert report["fracture_flow_calls"] == 0


def test_staging_overflow_preserves_already_committed_prefix(tmp_path, monkeypatch):
    checkpoint, initial, _, _ = portable_rollout_inputs(tmp_path)
    original = dynamics_runtime.state_batch
    calls = []

    def stage(*args, **kwargs):
        batch = original(*args, **kwargs)
        calls.append(batch)
        if len(calls) == 2:
            batch["positions"][0, 0] = torch.inf
        return batch

    monkeypatch.setattr(dynamics_runtime, "state_batch", stage)
    directory = tmp_path / "overflow"
    report = rollout_dynamics(checkpoint, initial, directory, 3, device="cpu")
    assert report["status"] == "nonfinite_prediction"
    assert report["completed_steps"] == 1
    assert report["predictor_calls"] == 1
    assert report["events"][-1]["step"] == 2
    assert "numeric range" in report["events"][-1]["reason"]
    assert (directory / "trajectory.npz").is_file()
    assert (directory / "replay.json").is_file()
