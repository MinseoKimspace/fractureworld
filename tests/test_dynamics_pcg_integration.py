import json

import numpy as np
import pytest

from fracture_world import dynamics_experiment, dynamics_matched, dynamics_runtime
from fracture_world.cli import parser
from fracture_world.dynamics_ablation_metrics import summarize_motion_ablation
from fracture_world.dynamics_benchmark_merge import merge_dynamics_benchmarks
from fracture_world.dynamics_corrector import solve_central
from fracture_world.dynamics_experiment import _comparison_methods, _paired_solver_metrics, compare_dynamics
from fracture_world.dynamics_initial import load_initial
from fracture_world.dynamics_learning import load_dynamics_model
from fracture_world.dynamics_runtime import RolloutConfig
from test_dynamics_benchmark_merge import make_benchmark, write_report
from test_dynamics_motion_ablation import motion_inputs
from test_dynamics_runtime import portable_rollout_inputs


@pytest.mark.parametrize("method", ["central_pcg", "no_motion_central_pcg"])
def test_cli_accepts_optional_pcg_methods_without_changing_defaults(method):
    rollout = ["dynamics-rollout", "--checkpoint", "best.pt", "--initial", "initial.npz",
               "--output", "rollout", "--steps", "1"]
    assert parser().parse_args(rollout).method == "consensus"
    assert parser().parse_args([*rollout, "--method", method]).method == method
    matched = ["dynamics-matched", "--checkpoint", "best.pt", "--data", "collection.json", "--output", "matched"]
    assert parser().parse_args(matched).methods is None
    assert parser().parse_args([*matched, "--methods", "central", method]).methods == ["central", method]
    assert _comparison_methods(None, None) == ["raw", "central", "consensus"]
    assert _comparison_methods(["central", method], None) == ["central", method]


def test_matched_pcg_uses_one_frozen_proposal_and_candidate_pool(tmp_path, monkeypatch):
    _, initial, model, config, calls = motion_inputs(tmp_path, monkeypatch)
    report = dynamics_matched.compare_matched_step(initial, model, config, methods=["raw", "central", "central_pcg"])
    assert tuple(report["methods"]) == ("raw", "central", "central_pcg")
    assert len(calls["forwards"]) == report["predictor_calls"] == 1
    assert report["attempted_method_runs"] == report["requested_method_runs"] == 3 * report["candidate_count"]
    assert len(calls["solves"]) == 2 * report["candidate_count"]
    for index in range(report["candidate_count"]):
        records = [row for row in report["candidates"] if row["candidate"] == index]
        assert len({row["input_hash"] for row in records}) == 1
        assert len({row["candidate_hash"] for row in records}) == 1
        assert len({row["proposal_hash"] for row in records}) == 1
        first, second = calls["solves"][2 * index:2 * index + 2]
        assert (first["method"], second["method"]) == ("central", "central_pcg")
        for field in ("positions", "velocities", "mask", "fixed", "proposal_positions", "proposal_velocities"):
            np.testing.assert_array_equal(first[field], second[field])
    assert len(report["central_central_pcg"]) == report["candidate_count"]
    assert all(row["comparison_available"] for row in report["central_central_pcg"])
    assert all(row["objective_difference_second_minus_first"] == 0. for row in report["central_central_pcg"])
    selected = next(row for row in report["selected_state_comparisons"] if row["methods"] == ["central", "central_pcg"])
    assert selected["comparison_available"] is True
    assert selected["positions_rms_metres"] == 0.


def test_matched_motion_ablation_expands_selected_pcg_solver_and_reports_pairs(tmp_path, monkeypatch):
    _, initial, model, config, calls = motion_inputs(tmp_path, monkeypatch)
    report = dynamics_matched.compare_matched_step(initial, model, config, methods=["central", "central_pcg"], motion_ablation=True)
    assert tuple(report["methods"]) == ("central", "central_pcg", "no_motion_central", "no_motion_central_pcg")
    assert len(calls["forwards"]) == 1
    assert len(calls["solves"]) == 4 * report["candidate_count"]
    assert all(row["comparison_available"] for row in report["no_motion_central_central_pcg"])
    pcg = [row for row in report["motion_ablation_comparisons"] if row["solver"] == "central_pcg"]
    assert len(pcg) == report["candidate_count"] + 1
    summary = summarize_motion_ablation([report])
    assert summary["central_pcg"]["fixed_candidate"]["requested_pairs"] == report["candidate_count"]
    for index in range(report["candidate_count"]):
        group = calls["solves"][4 * index:4 * index + 4]
        for item in group[1:]:
            np.testing.assert_array_equal(item["envelope"], group[0]["envelope"])
            np.testing.assert_array_equal(item["mask"], group[0]["mask"])
            np.testing.assert_array_equal(item["fixed"], group[0]["fixed"])


def test_explicit_no_motion_pcg_does_not_add_unrequested_arms(tmp_path, monkeypatch):
    _, initial, model, config, calls = motion_inputs(tmp_path, monkeypatch)
    report = dynamics_matched.compare_matched_step(initial, model, config, methods=["no_motion_central_pcg"])
    assert tuple(report["methods"]) == ("no_motion_central_pcg",)
    assert report["motion_ablation"] is True
    assert len(calls["solves"]) == report["candidate_count"]
    assert all(row["motion_source"] == "physical_prior" for row in report["candidates"])
    assert all(row["method"] == "central_pcg" for row in calls["solves"])


def test_matched_dataset_runner_keeps_explicit_mixed_motion_methods(tmp_path, monkeypatch):
    from fracture_world.dynamics_data import load_dynamics_trajectory
    from test_dynamics_data import recorded_trajectory

    checkpoint, _, _, _ = portable_rollout_inputs(tmp_path, has_corrector=False)
    model, _, saved, device = load_dynamics_model(checkpoint, "cpu")
    case, trajectory = recorded_trajectory(tmp_path / "source")
    item = load_dynamics_trajectory(case, trajectory)
    saved["training_mesh_fingerprints"] = [item.topology.fingerprint]
    monkeypatch.setattr(dynamics_matched, "load_dynamics_model", lambda *args: (model, None, saved, device))
    monkeypatch.setattr(dynamics_matched, "load_dynamics_dataset", lambda *args, **kwargs: [item])
    calls = []

    def correct(*args, **kwargs):
        calls.append(args[8])
        return None, dict(status="feasibility_unknown", solves=[])

    monkeypatch.setattr(dynamics_runtime, "_correct", correct)
    methods = ["central", "no_motion_central_pcg"]
    report = dynamics_matched.compare_matched_dynamics(
        checkpoint, tmp_path / "unused", tmp_path / "matched", RolloutConfig(fracture_samples=0, fracture_policy="marginal_search"),
        "cpu", split="all", frames=[0], methods=methods)
    assert report["methods"] == methods
    assert list(report["conditions"]) == methods
    assert calls == ["central", "central_pcg"] * report["conditions"]["central"]["requested_candidates"]
    assert report["motion_ablation"] is True


@pytest.mark.parametrize("methods", [[], ["central_pcg", "central_pcg"], ["physics"], ["neural"], "central_pcg"])
def test_matched_rejects_invalid_optional_methods(methods):
    with pytest.raises(ValueError, match="matched methods"):
        dynamics_matched._matched_methods(methods, False)


@pytest.mark.parametrize("energy_balance", [False, True])
def test_runtime_dispatches_pcg_to_the_selected_linear_or_energy_solver(tmp_path, monkeypatch, energy_balance):
    checkpoint, path, _, _ = portable_rollout_inputs(tmp_path, has_corrector=False)
    initial = load_initial(path)
    model, _, _, _ = load_dynamics_model(checkpoint, "cpu")
    calls = []
    energy_solver = dynamics_runtime.solve_energy

    def linear(problem, tolerances=None, max_iterations=None, cancel=None, progress=None):
        calls.append(("linear", tolerances.absolute, tolerances.relative, max_iterations))
        return solve_central(problem, tolerances, max_iterations, cancel=cancel)

    def energy(problem, projection, method, tolerances, rho, max_iterations, cancel, **kwargs):
        calls.append((method, tolerances.absolute, tolerances.relative, max_iterations))
        return energy_solver(problem, projection, "central", tolerances, rho, max_iterations, cancel, **kwargs)

    monkeypatch.setattr(dynamics_runtime, "solve_central_pcg", linear)
    monkeypatch.setattr(dynamics_runtime, "solve_energy", energy)
    config = RolloutConfig(energy_balance=energy_balance, mechanics_mode="disabled", fracture_initiation=False,
                           fracture_samples=0, fracture_policy="marginal_search", max_iterations=10000)
    report = dynamics_matched.compare_matched_step(initial, model, config, methods=["central_pcg"])
    assert calls
    assert all(call == ("central_pcg" if energy_balance else "linear", config.absolute_tolerance,
                        config.relative_tolerance, config.max_iterations) for call in calls)
    assert tuple(report["methods"]) == ("central_pcg",)


@pytest.mark.parametrize("method", ["central_pcg", "no_motion_central_pcg"])
def test_pcg_rollout_exports_method_and_motion_metadata(tmp_path, monkeypatch, method):
    checkpoint, initial, _, _ = portable_rollout_inputs(tmp_path, has_corrector=False)
    original = {path: path.read_bytes() for path in (checkpoint, initial)}
    monkeypatch.setattr(dynamics_runtime, "solve_central_pcg", lambda problem, tolerances, max_iterations, **kwargs:
                        solve_central(problem, tolerances, max_iterations, cancel=kwargs.get("cancel")))
    config = RolloutConfig(energy_balance=False, mechanics_mode="disabled", fracture_initiation=False,
                           fracture_samples=0, fracture_policy="marginal_search", max_iterations=10000)
    directory = tmp_path / method
    report = dynamics_runtime.rollout_dynamics(checkpoint, initial, directory, 1, method, config, "cpu")
    assert report["status"] == "completed"
    assert report["method"] == method
    assert report["solver_method"] == "central_pcg"
    assert report["motion_source"] == ("physical_prior" if method.startswith("no_motion_") else "learned")
    with np.load(directory / "trajectory.npz", allow_pickle=False) as archive:
        metadata = json.loads(str(archive["metadata_json"]))
        assert metadata["method"] == method
        assert metadata["solver_method"] == "central_pcg"
    replay = json.loads((directory / "replay.json").read_text())
    assert replay["metadata"]["method"] == method
    assert all(path.read_bytes() == content for path, content in original.items())


def test_paired_solver_metrics_use_method_order_and_shared_completed_cases():
    def row(case, completed, runtime, iterations):
        return dict(benchmark_case_id=case, full_horizon_evaluated=completed, runtime_seconds=runtime,
                    corrector_iterations=iterations, optimization_solves=2)

    rows = {"central_pcg": [row("a", True, 1., 4), row("b", False, 100., 100)],
            "central": [row("a", True, 3., 7), row("b", True, 10., 20)]}
    report = _paired_solver_metrics(rows, rows)
    pair = report["pairwise"][0]
    assert pair["methods"] == ["central_pcg", "central"]
    assert pair["common_full_horizon_case_ids"] == ["a"]
    assert pair["metrics"]["runtime_seconds"]["means"] == {"central_pcg": 1., "central": 3.}
    assert pair["metrics"]["runtime_seconds"]["mean_difference_second_minus_first"] == 2.
    assert pair["metrics"]["corrector_iterations"]["mean_difference_second_minus_first"] == 3.


def test_comparison_reports_pcg_agreement_for_both_motion_sources(tmp_path, monkeypatch):
    methods = ["central", "central_pcg", "no_motion_central", "no_motion_central_pcg"]
    monkeypatch.setattr(dynamics_experiment, "load_dynamics_model", lambda *args: (None, None, {}, "cpu"))

    def rollout(checkpoint, initial, output, steps, method, *args):
        output.mkdir()
        offset = .25 if method.endswith("pcg") else 0.
        positions = np.asarray([[[0., 0., 0.]], [[offset, 0., 0.]]])
        np.savez(output / "trajectory.npz", times=[0., 1.], positions=positions, velocities=positions,
                 broken=np.zeros((2, 1), dtype=bool))
        return dict(status="completed", events=[])

    monkeypatch.setattr(dynamics_experiment, "rollout_dynamics", rollout)
    report = compare_dynamics("best.pt", "initial.npz", tmp_path / "comparison", 1, methods=methods)
    for name in ("central_central_pcg_agreement", "no_motion_central_central_pcg_agreement"):
        assert report[name]["agreement_available"] is True
        assert report[name]["positions_rms_metres"] == pytest.approx(.25 / np.sqrt(3.))
    assert ["central_pcg", "no_motion_central_pcg"] in report["motion_ablation"]["comparison_pairs"]


def test_merge_recomputes_pcg_paired_runtime_without_changing_source_reports(tmp_path):
    central = make_benchmark(tmp_path / "central", ["central"])
    pcg = make_benchmark(tmp_path / "pcg", ["central"])

    def renamed(value):
        if isinstance(value, dict):
            return {"central_pcg" if key == "central" else key: renamed(item) for key, item in value.items()}
        if isinstance(value, list):
            return [renamed(item) for item in value]
        return "central_pcg" if value == "central" else value

    for directory, runtime, iterations in ((central, 3., 7), (pcg, 1., 4)):
        for path in directory.rglob("*.json"):
            data = json.loads(path.read_text())
            if directory == pcg:
                data = renamed(data)
            if "conditions" in data and "format" in data:
                for condition in data["conditions"].values():
                    condition.update(runtime_seconds=runtime, corrector_iterations=iterations, optimization_solves=2)
            write_report(path, data)
    before = {path: path.read_bytes() for directory in (central, pcg) for path in directory.rglob("*.json")}
    report = merge_dynamics_benchmarks([central, pcg], tmp_path / "merged.json")
    pair = report["paired_solver_metrics"]["pairwise"][0]
    assert pair["methods"] == ["central", "central_pcg"]
    assert pair["metrics"]["runtime_seconds"]["mean_difference_second_minus_first"] == -2.
    assert pair["metrics"]["corrector_iterations"]["mean_difference_second_minus_first"] == -3.
    assert all(path.read_bytes() == content for path, content in before.items())
