import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from fracture_world import dynamics_experiment
from fracture_world.cli import parser
from fracture_world.dynamics_experiment import _comparison_methods, benchmark_dynamics, compare_dynamics


METHODS = ["raw", "physics"]


@pytest.mark.parametrize("method", METHODS[-2:])
@pytest.mark.parametrize("command", ["dynamics-rollout", "dynamics-compare", "dynamics-benchmark"])
def test_autonomous_cli_accepts_model_and_physics_methods(command, method):
    arguments = [command, "--checkpoint", "best.pt", "--output", "output", "--steps", "200"]
    if command == "dynamics-benchmark":
        arguments.extend(["--data", "collection.json", "--methods", method])
    else:
        arguments.extend(["--initial", "initial.npz"])
        arguments.extend(["--method" if command == "dynamics-rollout" else "--methods", method])
    parsed = parser().parse_args(arguments)
    assert parsed.steps == 200
    assert (parsed.method if command == "dynamics-rollout" else parsed.methods[0]) == method


def test_autonomous_comparison_defaults_to_the_model():
    assert _comparison_methods(None) == ["raw"]
    assert _comparison_methods(METHODS) == METHODS


def test_autonomous_ablation_rejects_duplicate_conditions():
    with pytest.raises(ValueError, match="unique"):
        _comparison_methods(["raw", "raw"])


def test_comparison_runs_both_full_horizons_before_accessing_reference(tmp_path, monkeypatch):
    checkpoint = tmp_path / "best.pt"
    checkpoint.write_bytes(b"existing-checkpoint")
    initial = tmp_path / "initial.npz"
    initial.write_bytes(b"immutable-initial")
    checkpoint_before, initial_before = checkpoint.read_bytes(), initial.read_bytes()
    calls, evaluated = [], []

    def rollout(received_checkpoint, received_initial, output, steps, method, config, device, progress):
        assert received_checkpoint == checkpoint and received_initial == initial
        assert steps == 200
        calls.append(method)
        output.mkdir()
        positions = np.full((201, 4, 3), METHODS.index(method), dtype=float)
        positions[0] = 0
        np.savez(output / "trajectory.npz", times=np.arange(201) * .01, positions=positions,
                 velocities=np.zeros((201, 4, 3)), broken=np.zeros((201, 1), dtype=bool))
        return dict(status="completed", completed_steps=200, events=[], engine_calls=0)

    def evaluate(output, case, trajectory, report):
        assert calls == METHODS
        evaluated.append(output.name)
        return dict(last_compared_native_step=200)

    monkeypatch.setattr(dynamics_experiment, "rollout_dynamics", rollout)
    monkeypatch.setattr(dynamics_experiment, "evaluate_dynamics", evaluate)
    report = compare_dynamics(checkpoint, initial, tmp_path / "comparison", 200, methods=METHODS,
                              reference_case=tmp_path / "teacher.npz")
    assert calls == evaluated == METHODS
    assert report["all_rollouts_completed"]
    assert "need not match" in report["proposal_control"]
    assert report["numerical_reference_used_for_inference"] is False
    assert checkpoint.read_bytes() == checkpoint_before
    assert initial.read_bytes() == initial_before
    assert json.loads((tmp_path / "comparison" / "comparison.json").read_text())["requested_methods"] == METHODS


@pytest.mark.parametrize("method", ["central", "no_motion_central", "consensus", "no_motion_consensus", "neural"])
@pytest.mark.parametrize("command", ["dynamics-rollout", "dynamics-compare", "dynamics-benchmark"])
def test_online_cli_rejects_correction_methods(command, method):
    arguments = [command, "--checkpoint", "best.pt", "--output", "output", "--steps", "1"]
    arguments.extend(["--data", "data"] if command == "dynamics-benchmark" else ["--initial", "initial.npz"])
    arguments.extend(["--method" if command == "dynamics-rollout" else "--methods", method])
    with pytest.raises(SystemExit):
        parser().parse_args(arguments)


class BenchmarkCase(SimpleNamespace):
    def __len__(self):
        return len(self.times) - 1


def test_benchmark_scores_model_and_physics_on_same_complete_cases_and_keeps_failures(tmp_path, monkeypatch):
    checkpoint = tmp_path / "best.pt"
    checkpoint.write_bytes(b"existing-checkpoint")
    source = tmp_path / "source" / "cases" / "bunny.npz"
    case = BenchmarkCase(case_id="bunny", condition_digest="held-out", shape_family="bunny",
                         fixed=np.zeros((201, 4), dtype=bool), broken=np.zeros((201, 1), dtype=bool),
                         times=np.arange(201) * .01,
                         metadata=dict(source_hashes={str(source): "recorded"}, conditioning_context=dict(dt=.01, steps=200)))
    saved = dict(training=dict(overfit=False), splits=dict(test=["bunny"]), condition_splits=dict(test=["held-out"]))
    monkeypatch.setattr(dynamics_experiment, "load_dynamics_predictor", lambda *args: (None, saved, "cpu"))
    monkeypatch.setattr(dynamics_experiment, "load_dynamics_dataset", lambda *args, **kwargs: [case])
    exported = []

    def export(received_source, trajectory, frame, initial, *, support_policy):
        assert received_source == source and frame == 0 and support_policy == "free"
        exported.append(initial)

    def compare(received_checkpoint, initial, output, steps, config, device, methods, *reference):
        assert steps == 200 and methods == METHODS and initial == exported[0]
        conditions = {}
        for index, method in enumerate(methods):
            failed = method == "physics"
            evaluation = dict(last_compared_native_step=20 if failed else 200,
                              new_fracture=dict(f1=1., predicted_broken_facets=0),
                              position_mass_weighted_rmse_m=float(index), velocity_mass_weighted_rmse_m_per_s=float(index))
            conditions[method] = dict(status="failed" if failed else "completed", evaluation=evaluation,
                                      all_committed_geometry_valid=True, all_committed_energy_valid=True,
                                      all_committed_swept_geometry_valid=True, runtime_seconds=1.)
        return dict(status="comparison_finished", conditions=conditions, all_rollouts_completed=False)

    monkeypatch.setattr(dynamics_experiment, "export_initial", export)
    monkeypatch.setattr(dynamics_experiment, "compare_dynamics", compare)
    report = benchmark_dynamics(checkpoint, tmp_path / "collection.json", tmp_path / "benchmark", 200, methods=METHODS)
    assert report["requested_method_runs"] == report["attempted_method_runs"] == 2
    assert report["motion_ablation"]["enabled"] is False
    assert report["reference_used_only_for_evaluation"]
    assert report["conditions"]["physics"]["failed_cases"] == 1
    assert report["conditions"]["physics"]["full_horizon_accuracy_cases"] == 0
    pairs = {tuple(item["methods"]): item for item in report["paired_accuracy"]["pairwise"]}
    assert pairs[("raw", "physics")]["common_full_horizon_cases"] == 0
    assert report["paired_accuracy"]["all_methods"]["common_full_horizon_cases"] == 0
