import hashlib
import json

import pytest

from fracture_world.cli import parser, run
from fracture_world.dynamics_benchmark_merge import merge_dynamics_benchmarks


OLD_METHODS = ["raw", "central", "consensus"]
NEW_METHODS = ["no_motion_central", "no_motion_consensus"]


def write_report(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value), encoding="utf-8")


def make_benchmark(directory, methods, *, failing_method=None):
    summaries = {method: dict(cases=2, requested_cases=2, attempted_cases=2, unattempted_cases=0,
                              interrupted_cases=0, failed_cases=int(method == failing_method),
                              full_horizon_accuracy_cases=2 - int(method == failing_method),
                              completed_rate=.5 if method == failing_method else 1.,
                              preserved_source_metric=f"original-{method}") for method in methods}
    entries = []
    for index, case in enumerate(("bunny-a", "bunny-b")):
        relative = f"case-{index + 1:05d}"
        conditions = {}
        for method_index, method in enumerate(OLD_METHODS + NEW_METHODS):
            if method not in methods:
                continue
            failed = method == failing_method and index == 1
            status = "infeasible" if failed else "completed"
            steps = 17 if failed else 200
            evaluation = dict(format="fracture-world.dynamics-evaluation.v1", method=method, case_id=case,
                              rollout_status=status, condition_digest=f"condition-{index}", mesh_fingerprint="mesh",
                              evaluation_only_teacher_future=True, inference_teacher_future_used=False,
                              condition_split="test", generalization_protocol="loading_condition", overfit_debug=False,
                              last_compared_native_step=steps,
                              source_hashes=dict(rollout_report=f"report-{method}", rollout_trajectory=f"trajectory-{method}",
                                                 teacher={f"{case}.npz": "a" * 64}),
                              position_mass_weighted_rmse_m=1000. if failed else float(method_index + index),
                              velocity_mass_weighted_rmse_m_per_s=1000. if failed else float(2 * method_index + index))
            conditions[method] = dict(format="fracture-world.neural-rollout.v1", status=status, method=method,
                                      case_id=case, checkpoint_sha256="f" * 64, initial_sha256=str(index + 1) * 64,
                                      condition_split="test", generalization_protocol="loading_condition",
                                      teacher_future_used=False, runtime_native_required=False, engine_calls=0,
                                      overfit_debug=False, requested_steps=200, completed_steps=steps,
                                      condition_digest=f"condition-{index}", mesh_fingerprint="mesh",
                                      configuration=dict(seed=17, rho=.01, mechanics_mode="proximal"), evaluation=evaluation)
        complete = all(result["status"] == "completed" for result in conditions.values())
        comparison = dict(format="fracture-world.neural-comparison.v1", status="comparison_finished", engine_calls=0,
                          numerical_reference_used_for_inference=False, requested_methods=methods, attempted_methods=methods,
                          unattempted_methods=[], conditions=conditions, all_rollouts_completed=complete)
        write_report(directory / relative / "comparison" / "comparison.json", comparison)
        entries.append(dict(case=case, directory=relative, status="comparison_finished", attempted_methods=methods,
                            unattempted_methods=[], all_rollouts_completed=complete))
    benchmark = dict(schema="fracture-world.dynamics-benchmark.v1", status="benchmark_finished", inference_engine_calls=0,
                     reference_used_only_for_evaluation=True, requested_methods=methods, requested_case_ids=["bunny-a", "bunny-b"],
                     requested_cases=2, attempted_cases=2, unattempted_cases=0, requested_method_runs=2 * len(methods),
                     attempted_method_runs=2 * len(methods), unattempted_method_runs=0, unattempted_case_ids=[],
                     checkpoint_sha256="f" * 64, split="test", generalization_protocol="loading_condition",
                     cases=entries, conditions=summaries, all_rollouts_completed=failing_method is None)
    write_report(directory / "benchmark.json", benchmark)
    return directory


def mutate(path, change):
    report = json.loads(path.read_text())
    change(report)
    write_report(path, report)


def source_hashes(directories):
    return {str(path): hashlib.sha256(path.read_bytes()).hexdigest() for directory in directories
            for path in directory.rglob("*.json")}


def test_merge_reuses_saved_five_arm_results_and_preserves_sources_and_failure_counts(tmp_path):
    old = make_benchmark(tmp_path / "old", OLD_METHODS)
    new = make_benchmark(tmp_path / "new", NEW_METHODS, failing_method="no_motion_consensus")
    before = source_hashes([old, new])
    output = tmp_path / "merged.json"
    report = merge_dynamics_benchmarks([old, new], output)
    assert report["requested_methods"] == OLD_METHODS + NEW_METHODS
    assert report["requested_method_runs"] == report["attempted_method_runs"] == 10
    assert report["merge_simulation_calls"] == report["inference_engine_calls"] == 0
    assert report["runtime_source_revision_match"] == "unknown_not_recorded"
    assert not report["all_rollouts_completed"]
    assert report["conditions"]["no_motion_consensus"]["failed_cases"] == 1
    assert report["conditions"]["no_motion_consensus"]["full_horizon_accuracy_cases"] == 1
    assert report["conditions"]["raw"]["preserved_source_metric"] == "original-raw"
    paired = report["paired_accuracy"]
    assert paired["all_methods"]["common_full_horizon_case_ids"] == ["bunny-a"]
    pairs = {tuple(row["methods"]): row for row in paired["pairwise"]}
    central = pairs[("central", "no_motion_central")]
    assert central["common_full_horizon_case_ids"] == ["bunny-a", "bunny-b"]
    assert central["metrics"]["position_mass_weighted_rmse_m"]["mean_difference_second_minus_first"] == 2.
    consensus = pairs[("consensus", "no_motion_consensus")]
    assert consensus["common_full_horizon_case_ids"] == ["bunny-a"]
    assert consensus["metrics"]["position_mass_weighted_rmse_m"]["means"] == {"consensus": 2., "no_motion_consensus": 4.}
    assert before == source_hashes([old, new])
    assert json.loads(output.read_text())["source_benchmarks"][0]["benchmark"] == str(old / "benchmark.json")
    assert len(report["source_benchmarks"][1]["comparisons"]) == 2
    assert report["motion_ablation"]["comparison_pairs"] == [["central", "no_motion_central"], ["consensus", "no_motion_consensus"]]


@pytest.mark.parametrize("field,value", [("checkpoint_sha256", "e" * 64), ("split", "validation"),
                                        ("generalization_protocol", "shape_family"),
                                        ("requested_case_ids", ["bunny-b", "bunny-a"])])
def test_merge_rejects_incompatible_benchmark_contract(tmp_path, field, value):
    old = make_benchmark(tmp_path / "old", OLD_METHODS)
    new = make_benchmark(tmp_path / "new", NEW_METHODS)
    mutate(new / "benchmark.json", lambda report: report.update({field: value}))
    with pytest.raises(ValueError):
        merge_dynamics_benchmarks([old, new], tmp_path / "merged.json")
    assert not (tmp_path / "merged.json").exists()


@pytest.mark.parametrize("change", [
    lambda result: result["configuration"].update(seed=18),
    lambda result: result.update(initial_sha256="a" * 64),
    lambda result: result.update(requested_steps=201),
    lambda result: result["evaluation"]["source_hashes"]["teacher"].update({"bunny-a.npz": "b" * 64}),
    lambda result: result["evaluation"]["source_hashes"]["teacher"].update({"bunny-a.npz": "invalid-hash"}),
    lambda result: result.update(teacher_future_used=True),
    lambda result: result["evaluation"].update(inference_teacher_future_used=True),
    lambda result: result.pop("initial_sha256"),
    lambda result: result["evaluation"].pop("source_hashes"),
    lambda result: result.update(condition_digest="different-condition"),
    lambda result: result.update(mesh_fingerprint="different-mesh"),
])
def test_merge_rejects_incompatible_numerical_conditions_and_missing_provenance(tmp_path, change):
    old = make_benchmark(tmp_path / "old", OLD_METHODS)
    new = make_benchmark(tmp_path / "new", NEW_METHODS)
    mutate(new / "case-00001" / "comparison" / "comparison.json", lambda report: change(report["conditions"][NEW_METHODS[0]]))
    with pytest.raises(ValueError):
        merge_dynamics_benchmarks([old, new], tmp_path / "merged.json")


@pytest.mark.parametrize("change", [
    lambda report: report.update(status="interrupted"),
    lambda report: report.update(unattempted_method_runs=1),
    lambda report: report.update(attempted_cases=1),
    lambda report: report.update(unattempted_case_ids=["bunny-b"]),
    lambda report: report["conditions"][NEW_METHODS[0]].update(failed_cases=1),
    lambda report: report.update(unattempted_method_runs=False),
    lambda report: report.update(attempted_cases=2.),
])
def test_merge_rejects_incomplete_or_inconsistent_benchmarks(tmp_path, change):
    old = make_benchmark(tmp_path / "old", OLD_METHODS)
    new = make_benchmark(tmp_path / "new", NEW_METHODS)
    mutate(new / "benchmark.json", change)
    with pytest.raises(ValueError):
        merge_dynamics_benchmarks([old, new], tmp_path / "merged.json")


def test_merge_rejects_duplicate_methods_and_overwriting_output(tmp_path):
    old = make_benchmark(tmp_path / "old", OLD_METHODS)
    duplicate = make_benchmark(tmp_path / "duplicate", ["central"])
    with pytest.raises(ValueError, match="disjoint"):
        merge_dynamics_benchmarks([old, duplicate], tmp_path / "merged.json")
    new = make_benchmark(tmp_path / "new", NEW_METHODS)
    existing = tmp_path / "existing.json"
    existing.write_bytes(b"preserve")
    with pytest.raises(ValueError, match="new JSON file"):
        merge_dynamics_benchmarks([old, new], existing)
    assert existing.read_bytes() == b"preserve"


def test_merge_rejects_case_path_escape(tmp_path):
    old = make_benchmark(tmp_path / "old", OLD_METHODS)
    new = make_benchmark(tmp_path / "new", NEW_METHODS)
    mutate(new / "benchmark.json", lambda report: report["cases"][0].update(directory="../old/case-00001"))
    with pytest.raises(ValueError, match="within"):
        merge_dynamics_benchmarks([old, new], tmp_path / "merged.json")


def test_merge_cli_reads_reports_without_checkpoint_or_simulation_files(tmp_path, capsys):
    old = make_benchmark(tmp_path / "old", OLD_METHODS)
    new = make_benchmark(tmp_path / "new", NEW_METHODS)
    output = tmp_path / "merged.json"
    arguments = parser().parse_args(["dynamics-merge-benchmarks", "--inputs", str(old), str(new), "--output", str(output)])
    run(arguments)
    printed = json.loads(capsys.readouterr().out)
    assert printed["status"] == "merged"
    assert printed["merge_simulation_calls"] == 0
    assert printed["methods"] == OLD_METHODS + NEW_METHODS
    assert output.is_file()


def test_merge_reports_only_common_complete_cases_with_failures_in_multiple_sources(tmp_path):
    old = make_benchmark(tmp_path / "old", OLD_METHODS, failing_method="central")
    new = make_benchmark(tmp_path / "new", NEW_METHODS, failing_method="no_motion_consensus")
    report = merge_dynamics_benchmarks([old, new], tmp_path / "merged.json")
    assert report["paired_accuracy"]["all_methods"]["common_full_horizon_cases"] == 1
    assert report["conditions"]["central"]["completed_rate"] == .5
    assert report["conditions"]["no_motion_consensus"]["completed_rate"] == .5
