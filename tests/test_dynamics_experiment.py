import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from fracture_world import dynamics_experiment
from fracture_world.dynamics_experiment import _comparison_methods, _paired_accuracy, _runtime_support_policy, benchmark_dynamics, compare_dynamics


class MetadataWithoutSupportFuture(dict):
    def __getitem__(self, key):
        if key == "observed_support_releases":
            raise AssertionError("teacher support-release future cannot select a runtime policy")
        return super().__getitem__(key)


class BenchmarkCase(SimpleNamespace):
    def __len__(self):
        return len(self.broken) - 1


def test_online_comparison_defaults_to_the_direct_model():
    assert _comparison_methods(None) == ["raw"]
    assert _comparison_methods(["raw", "physics"]) == ["raw", "physics"]


@pytest.mark.parametrize("methods", [[], "raw", ["raw", "raw"], ["central"], ["no_motion_central"],
                                     ["consensus"], ["no_motion_consensus"], ["neural"]])
def test_online_comparison_rejects_unsupported_methods_before_loading_data(tmp_path, methods):
    with pytest.raises(ValueError, match="comparison methods"):
        compare_dynamics(tmp_path / "missing.pt", tmp_path / "missing.npz", tmp_path / "unused", 1, methods=methods)
    assert not (tmp_path / "unused").exists()


@pytest.mark.parametrize("checkpoint,expected", [
    ({"model_config": {"learn_support_release": True}}, "learned"),
    ({"training": {"learn_support_release": True}}, "learned"),
    ({"model_config": {"learn_support_release": False}, "training": {"learn_support_release": True}}, "prescribed"),
    ({}, "prescribed"),
])
def test_support_policy_uses_checkpoint_capability_and_current_flags(checkpoint, expected):
    assert _runtime_support_policy(checkpoint, np.array([True, False])) == expected
    assert _runtime_support_policy(checkpoint, np.array([False, False])) == "free"


@pytest.mark.parametrize("learned", [False, True])
@pytest.mark.parametrize("native_step_stride", [1, 5])
@pytest.mark.parametrize("recorded_horizon", [1, 2])
def test_benchmark_does_not_select_support_policy_from_teacher_events(tmp_path, monkeypatch, learned,
                                                                    native_step_stride, recorded_horizon):
    checkpoint = tmp_path / "checkpoint.pt"
    checkpoint.write_bytes(b"test-checkpoint")
    cases = []
    for index, name in enumerate(("fixed-negative", "fixed-positive", "free-negative")):
        fixed = np.full((3, 2), index != 2, dtype=bool)
        if index == 1:
            fixed[-1] = False
        broken = np.tile([True, False], (3, 1))
        if index == 1:
            broken[-1, 1] = True
        source = tmp_path / "source" / "cases" / f"{name}.npz"
        cases.append(BenchmarkCase(case_id=name, condition_digest=f"condition-{index}", shape_family="fixture",
                                   fixed=fixed, broken=broken, times=np.array([0., 1., 2.]) * native_step_stride,
                                   metadata=MetadataWithoutSupportFuture(source_hashes={str(source): "digest"},
                                                                        conditioning_context=dict(dt=1., steps=2 * native_step_stride))))
    saved = dict(training=dict(overfit=False, learn_support_release=learned),
                 model_config=dict(learn_support_release=learned),
                 condition_splits={"test": [item.condition_digest for item in cases]},
                 splits={"test": [item.case_id for item in cases]})
    monkeypatch.setattr(dynamics_experiment, "load_dynamics_predictor", lambda *args: (None, saved, "cpu"))
    monkeypatch.setattr(dynamics_experiment, "load_dynamics_dataset", lambda *args, **kwargs: cases)
    exported = {}

    def export(case, trajectory, frame, initial, *, support_policy):
        assert frame == 0
        assert trajectory == case.parent.parent / "trajectories" / case.name
        exported[Path(initial)] = (case.stem, support_policy)

    def compare(checkpoint, initial, output, steps, config, device, methods, reference_case, reference_trajectory, progress):
        name, policy = exported[Path(initial)]
        evaluation = dict(compared_future_frames=recorded_horizon, last_compared_native_step=steps,
                          fracture=dict(f1=0.999, predicted_broken_facets=100),
                          new_fracture=dict(f1=0.0 if name == "fixed-positive" else 1.0,
                                            predicted_broken_facets=int(name == "fixed-negative")))
        metrics = ("position_mass_weighted_rmse_m", "velocity_mass_weighted_rmse_m_per_s",
                   "linear_momentum_error_kg_m_per_s", "angular_momentum_about_origin_error_kg_m2_per_s",
                   "kinetic_energy_absolute_error_j", "maximum_floor_penetration_m", "component_count_absolute_error")
        evaluation.update({key: 0.0 for key in metrics})
        return dict(all_rollouts_completed=True,
                    conditions={"raw": dict(status="completed", evaluation=evaluation,
                                             all_committed_geometry_valid=True, runtime_seconds=1.0)})

    monkeypatch.setattr(dynamics_experiment, "export_initial", export)
    monkeypatch.setattr(dynamics_experiment, "compare_dynamics", compare)
    report = benchmark_dynamics(checkpoint, tmp_path / "unused-data", tmp_path / "benchmark",
                                recorded_horizon * native_step_stride, methods=["raw"])
    expected = "learned" if learned else "prescribed"
    assert {name: policy for name, policy in exported.values()} == {
        "fixed-negative": expected, "fixed-positive": expected, "free-negative": "free"}
    summary = report["conditions"]["raw"]
    assert summary["fracture_positive_cases"] == (1 if recorded_horizon == 2 else 0)
    assert summary["fracture_positive_case_f1_failure_penalized"] == (0 if recorded_horizon == 2 else None)
    assert summary["completed_fracture_negative_cases"] == (2 if recorded_horizon == 2 else 3)
    assert summary["fracture_negative_case_false_positive_rate"] == (0.5 if recorded_horizon == 2 else 1 / 3)
    assert summary["full_horizon_accuracy_cases"] == 3
    assert report["reference_used_only_for_evaluation"]
    assert "initially broken facets excluded" in report["fracture_score_semantics"]
    paired = report["paired_accuracy"]
    assert paired["all_methods"]["common_full_horizon_cases"] == 3
    assert paired["all_methods"]["common_full_horizon_case_ids"] == sorted(item.case_id for item in cases)
    assert paired["all_methods"]["comparison_available"] is False
    assert paired["pairwise"] == []


def benchmark_result(case, completed, **metrics):
    return dict(benchmark_case_id=case, full_horizon_evaluated=completed, evaluation=metrics)


def test_paired_accuracy_excludes_method_specific_failures_from_every_method_mean():
    rows = {
        "raw": [benchmark_result("a", True, error=10.), benchmark_result("b", True, error=1000.),
                benchmark_result("c", False, error=0.)],
        "central": [benchmark_result("a", True, error=9.), benchmark_result("b", True, error=500.),
                    benchmark_result("c", True, error=10000.)],
        "physics": [benchmark_result("a", True, error=8.), benchmark_result("b", False, error=0.),
                      benchmark_result("c", True, error=9000.)],
    }
    common = _paired_accuracy(rows, rows, ["error"])
    assert common["common_full_horizon_case_ids"] == ["a"]
    assert common["excluded_case_ids"] == ["b", "c"]
    assert common["metrics"]["error"]["means"] == {"raw": 10., "central": 9., "physics": 8.}
    paired = _paired_accuracy(rows, ["raw", "central"], ["error"])
    assert paired["total_cases"] == 3
    assert paired["common_full_horizon_cases"] == 2
    assert paired["common_full_horizon_case_ids"] == ["a", "b"]
    assert paired["metrics"]["error"]["means"] == {"raw": 505., "central": 254.5}
    assert paired["metrics"]["error"]["mean_difference_second_minus_first"] == -250.5


def test_paired_accuracy_uses_shared_available_cases_for_each_metric():
    rows = {
        "raw": [benchmark_result("a", True, position=1.), benchmark_result("b", True, position=None),
                benchmark_result("c", True, position=5.), benchmark_result("d", True, position=float("inf")),
                benchmark_result("e", True, position=4.)],
        "physics": [benchmark_result("a", True, position=3.), benchmark_result("b", True, position=2.),
                      benchmark_result("c", True, position=None), benchmark_result("d", True, position=4.),
                      benchmark_result("e", True, position=8.)],
    }
    report = _paired_accuracy(rows, rows, ["position", "velocity"])
    assert report["common_full_horizon_cases"] == 5
    assert report["metrics"]["position"]["cases"] == 2
    assert report["metrics"]["position"]["case_ids"] == ["a", "e"]
    assert report["metrics"]["position"]["means"] == {"raw": 2.5, "physics": 5.5}
    assert report["metrics"]["position"]["mean_difference_second_minus_first"] == 3.
    assert report["metrics"]["velocity"]["cases"] == 0
    assert report["metrics"]["velocity"]["comparison_available"] is False
    assert report["metrics"]["velocity"]["means"] == {"raw": None, "physics": None}
    assert report["metrics"]["velocity"]["mean_difference_second_minus_first"] is None


def test_paired_accuracy_does_not_report_accuracy_when_no_case_completes_in_both_methods():
    rows = {
        "raw": [benchmark_result("a", True, error=1.), benchmark_result("b", False, error=0.)],
        "physics": [benchmark_result("a", False, error=0.), benchmark_result("b", True, error=2.)],
    }
    report = _paired_accuracy(rows, rows, ["error"])
    assert report["common_full_horizon_cases"] == 0
    assert report["common_full_horizon_case_ids"] == []
    assert report["comparison_available"] is False
    assert report["metrics"]["error"]["means"] == {"raw": None, "physics": None}
    assert report["metrics"]["error"]["mean_difference_second_minus_first"] is None


@pytest.mark.parametrize("steps", [0, -1, True, 1.5])
def test_benchmark_rejects_invalid_native_step_counts_before_loading_model(tmp_path, steps):
    with pytest.raises(ValueError, match="positive integer number of native timesteps"):
        benchmark_dynamics(tmp_path / "missing.pt", tmp_path / "data", tmp_path / "output", steps)


@pytest.mark.parametrize("interrupted_method", ["raw", "physics"])
def test_comparison_stops_after_interrupted_rollout_and_saves_partial_report(tmp_path, monkeypatch, interrupted_method):
    methods = ["raw", "physics"]
    calls, evaluations = [], []
    monkeypatch.setattr(dynamics_experiment, "load_dynamics_predictor", lambda *args: (None, {}, "cpu"))

    def rollout(checkpoint, initial, output, steps, method, *args):
        calls.append(method)
        output.mkdir()
        np.savez(output / "trajectory.npz", times=[0., 1.], positions=np.zeros((2, 4, 3)),
                 velocities=np.zeros((2, 4, 3)), broken=np.zeros((2, 1), dtype=bool))
        report = dict(status="interrupted" if method == interrupted_method else "completed",
                      completed_steps=1 if method == interrupted_method else steps, events=[{"preserved": True}])
        (output / "report.json").write_text(json.dumps(report), encoding="utf-8")
        return report

    def evaluate(output, *args):
        evaluations.append(output.name)
        return dict(last_compared_native_step=1 if output.name == interrupted_method else 2)

    monkeypatch.setattr(dynamics_experiment, "rollout_dynamics", rollout)
    monkeypatch.setattr(dynamics_experiment, "evaluate_dynamics", evaluate)
    output = tmp_path / "comparison"
    report = compare_dynamics(tmp_path / "model.pt", tmp_path / "initial.npz", output, 2,
                              methods=methods, reference_case=tmp_path / "case.npz")
    attempted = methods[:methods.index(interrupted_method) + 1]
    assert calls == attempted
    assert evaluations == attempted
    assert report["status"] == "interrupted"
    assert report["requested_methods"] == methods
    assert report["attempted_methods"] == attempted
    assert report["unattempted_methods"] == methods[len(attempted):]
    assert not report["all_rollouts_completed"]
    assert list(report["conditions"]) == attempted
    assert report["conditions"][interrupted_method]["status"] == "interrupted"
    assert (output / interrupted_method / "trajectory.npz").is_file()
    assert json.loads((output / "comparison.json").read_text())["status"] == "interrupted"
    for method in methods[len(attempted):]:
        assert not (output / method).exists()


@pytest.mark.parametrize("interrupted_index", [0, 1])
def test_benchmark_stops_after_interrupted_case_without_scoring_unattempted_runs(tmp_path, monkeypatch, interrupted_index):
    checkpoint = tmp_path / "checkpoint.pt"
    checkpoint.write_bytes(b"test-checkpoint")
    cases = []
    for index in range(3):
        source = tmp_path / "source" / "cases" / f"case-{index}.npz"
        cases.append(BenchmarkCase(case_id=f"case-{index}", condition_digest=f"condition-{index}", shape_family="fixture",
                                   fixed=np.zeros((3, 2), dtype=bool), broken=np.array([[False], [True], [True]]),
                                   times=np.array([0., 1., 2.]),
                                   metadata=dict(source_hashes={str(source): "digest"}, conditioning_context=dict(dt=1., steps=2))))
    saved = dict(training=dict(overfit=False), condition_splits={"test": [item.condition_digest for item in cases]},
                 splits={"test": [item.case_id for item in cases]})
    monkeypatch.setattr(dynamics_experiment, "load_dynamics_predictor", lambda *args: (None, saved, "cpu"))
    monkeypatch.setattr(dynamics_experiment, "load_dynamics_dataset", lambda *args, **kwargs: cases)
    exported, calls = {}, []
    methods = ["raw", "physics"]

    def export(case, trajectory, frame, initial, **kwargs):
        exported[Path(initial)] = case.stem

    def compare(checkpoint, initial, output, steps, config, device, selected, *args):
        name = exported[Path(initial)]
        calls.append(name)
        interrupted = name == f"case-{interrupted_index}"
        conditions = {}
        for method in (selected[:2] if interrupted else selected):
            status = "interrupted" if interrupted and method == "physics" else "completed"
            evaluation = dict(last_compared_native_step=1 if status == "interrupted" else 2,
                              new_fracture=dict(f1=.75, predicted_broken_facets=1), position_mass_weighted_rmse_m=.1)
            conditions[method] = dict(status=status, evaluation=evaluation, runtime_seconds=1.)
        output.mkdir()
        report = dict(status="interrupted" if interrupted else "comparison_finished", conditions=conditions,
                      all_rollouts_completed=not interrupted)
        (output / "comparison.json").write_text(json.dumps(report), encoding="utf-8")
        return report

    monkeypatch.setattr(dynamics_experiment, "export_initial", export)
    monkeypatch.setattr(dynamics_experiment, "compare_dynamics", compare)
    output = tmp_path / "benchmark"
    report = benchmark_dynamics(checkpoint, tmp_path / "data", output, 2, methods=methods)
    assert calls == [f"case-{index}" for index in range(interrupted_index + 1)]
    assert report["status"] == "interrupted"
    assert not report["all_rollouts_completed"]
    assert report["requested_cases"] == 3
    assert report["attempted_cases"] == interrupted_index + 1
    assert report["unattempted_cases"] == 2 - interrupted_index
    assert report["unattempted_case_ids"] == [f"case-{index}" for index in range(interrupted_index + 1, 3)]
    assert report["requested_method_runs"] == 6
    assert report["attempted_method_runs"] == 2 * interrupted_index + 2
    assert report["unattempted_method_runs"] == 4 - 2 * interrupted_index
    assert len(report["cases"]) == interrupted_index + 1
    assert report["conditions"]["raw"]["cases"] == interrupted_index + 1
    assert report["conditions"]["raw"]["fracture_positive_case_f1_failure_penalized"] == .75
    assert report["conditions"]["physics"]["interrupted_cases"] == 1
    assert report["conditions"]["physics"]["failed_cases"] == 0
    assert report["conditions"]["physics"]["cases"] == interrupted_index + 1
    assert report["conditions"]["physics"]["unattempted_cases"] == 2 - interrupted_index
    assert report["paired_accuracy"]["all_methods"]["common_full_horizon_cases"] == interrupted_index
    assert json.loads((output / "benchmark.json").read_text())["status"] == "interrupted"


def test_paired_solver_metrics_use_method_order_and_shared_completed_cases():
    def row(case, completed, runtime, iterations):
        return dict(benchmark_case_id=case, full_horizon_evaluated=completed, runtime_seconds=runtime,
                    corrector_iterations=iterations, optimization_solves=2)

    rows = {"central": [row("a", True, 1., 4), row("b", False, 100., 100)],
            "physics": [row("a", True, 3., 7), row("b", True, 10., 20)]}
    report = dynamics_experiment._paired_solver_metrics(rows, rows)
    pair = report["pairwise"][0]
    assert pair["methods"] == ["central", "physics"]
    assert pair["common_full_horizon_case_ids"] == ["a"]
    assert pair["metrics"]["runtime_seconds"]["means"] == {"central": 1., "physics": 3.}
    assert pair["metrics"]["runtime_seconds"]["mean_difference_second_minus_first"] == 2.
    assert pair["metrics"]["corrector_iterations"]["mean_difference_second_minus_first"] == 3.


@pytest.mark.parametrize("case_ids", [[], ["case", "case"], [None], "case"])
def test_benchmark_rejects_invalid_case_selection_before_loading_model(tmp_path, case_ids):
    with pytest.raises(ValueError, match="case IDs"):
        benchmark_dynamics(tmp_path / "missing.pt", tmp_path / "data", tmp_path / "output", 1, case_ids=case_ids)
    assert not (tmp_path / "output").exists()


@pytest.mark.parametrize("case_ids", [["case-1"], ["case-2", "case-0"]])
def test_benchmark_subset_evaluates_only_selected_recorded_cases(tmp_path, monkeypatch, case_ids):
    checkpoint = tmp_path / "checkpoint.pt"
    checkpoint.write_bytes(b"existing-checkpoint")
    cases = []
    for index in range(3):
        source = tmp_path / "source" / "cases" / f"case-{index}.npz"
        cases.append(BenchmarkCase(case_id=f"case-{index}", condition_digest=f"condition-{index}", shape_family="fixture",
                                   fixed=np.zeros((2, 2), dtype=bool), broken=np.zeros((2, 1), dtype=bool),
                                   times=np.array([0., 1.]), metadata=dict(source_hashes={str(source): "digest"},
                                                                        conditioning_context=dict(dt=1., steps=1))))
    saved = dict(training=dict(overfit=False), splits=dict(test=[item.case_id for item in cases]),
                 condition_splits=dict(test=[item.condition_digest for item in cases]))
    monkeypatch.setattr(dynamics_experiment, "load_dynamics_predictor", lambda *args: (None, saved, "cpu"))
    monkeypatch.setattr(dynamics_experiment, "load_dynamics_dataset", lambda *args, **kwargs: cases)
    exported = []
    monkeypatch.setattr(dynamics_experiment, "export_initial", lambda case, *args, **kwargs: exported.append(case.stem))

    def compare(*args):
        return dict(status="comparison_finished", all_rollouts_completed=True,
                    conditions=dict(raw=dict(status="completed", runtime_seconds=1.,
                                             evaluation=dict(last_compared_native_step=1,
                                                             new_fracture=dict(f1=1., predicted_broken_facets=0)))))

    monkeypatch.setattr(dynamics_experiment, "compare_dynamics", compare)
    report = benchmark_dynamics(checkpoint, tmp_path / "data", tmp_path / "subset", 1, case_ids=case_ids)
    selected = [item.case_id for item in cases if item.case_id in case_ids]
    assert exported == report["requested_case_ids"] == selected
    assert report["available_split_cases"] == 3
    assert report["requested_cases"] == report["attempted_cases"] == len(selected)
    assert report["requested_method_runs"] == len(selected)
    assert report["requested_methods"] == ["raw"]
    with pytest.raises(ValueError, match="selected split"):
        benchmark_dynamics(checkpoint, tmp_path / "data", tmp_path / "rejected", 1, case_ids=["unknown"])
    assert not (tmp_path / "rejected").exists()
