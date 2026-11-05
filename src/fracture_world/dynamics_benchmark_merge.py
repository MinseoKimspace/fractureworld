import copy
from itertools import combinations
import json
from pathlib import Path

from .dynamics_experiment import (_ACCURACY_METRICS, _AUTONOMOUS_PROPOSAL_CONTROL, _motion_ablation_protocol,
                                  _paired_accuracy)
from .io import file_hash


def _read_report(path):
    def invalid_constant(value):
        raise ValueError(f"nonfinite JSON value in {path}: {value}")

    with path.open(encoding="utf-8") as stream:
        value = json.load(stream, parse_constant=invalid_constant)
    if not isinstance(value, dict):
        raise ValueError(f"report must contain a JSON object: {path}")
    return value


def _required(report, name, expected_type):
    if not isinstance(report, dict):
        raise ValueError("report fields must belong to a JSON object")
    value = report.get(name)
    if not isinstance(value, expected_type) or isinstance(value, bool) and expected_type is int:
        raise ValueError(f"missing or invalid report field: {name}")
    return value


def _equal(value, expected, name):
    wrong_boolean = isinstance(expected, bool) and not isinstance(value, bool)
    wrong_integer = type(expected) is int and type(value) is not int
    if value != expected or wrong_boolean or wrong_integer:
        raise ValueError(f"incompatible or incomplete benchmark field: {name}")


def _sha256(value, name):
    if not isinstance(value, str) or len(value) != 64 or any(character not in "0123456789abcdef" for character in value):
        raise ValueError(f"{name} must be a SHA-256 digest")
    return value


def _safe_report(directory, relative):
    path = (directory / relative).resolve()
    if not path.is_relative_to(directory) or not path.is_file():
        raise ValueError("case report must be an existing file within its benchmark directory")
    return path


def _benchmark_contract(report):
    _equal(report.get("schema"), "fracture-world.dynamics-benchmark.v1", "schema")
    _equal(report.get("status"), "benchmark_finished", "status")
    _equal(report.get("inference_engine_calls"), 0, "inference_engine_calls")
    _equal(report.get("reference_used_only_for_evaluation"), True, "reference_used_only_for_evaluation")
    methods = _required(report, "requested_methods", list)
    cases = _required(report, "requested_case_ids", list)
    if not methods or any(not isinstance(value, str) or not value for value in methods) or len(set(methods)) != len(methods):
        raise ValueError("benchmark methods must be unique nonempty strings")
    if not cases or any(not isinstance(value, str) or not value for value in cases) or len(set(cases)) != len(cases):
        raise ValueError("benchmark case IDs must be unique nonempty strings")
    for name, expected in (("requested_cases", len(cases)), ("attempted_cases", len(cases)), ("unattempted_cases", 0),
                           ("requested_method_runs", len(cases) * len(methods)),
                           ("attempted_method_runs", len(cases) * len(methods)), ("unattempted_method_runs", 0),
                           ("unattempted_case_ids", [])):
        _equal(report.get(name), expected, name)
    checkpoint = _sha256(report.get("checkpoint_sha256"), "checkpoint_sha256")
    split = _required(report, "split", str)
    if split not in {"test", "validation"}:
        raise ValueError("merge requires a recorded held-out split")
    return dict(checkpoint_sha256=checkpoint, split=split,
                generalization_protocol=_required(report, "generalization_protocol", str), requested_case_ids=cases)


def _condition_contract(result, method, case, checkpoint, split, protocol):
    _equal(result.get("format"), "fracture-world.neural-rollout.v1", "rollout format")
    for name, expected in (("method", method), ("case_id", case), ("checkpoint_sha256", checkpoint),
                           ("condition_split", split), ("generalization_protocol", protocol),
                           ("teacher_future_used", False), ("runtime_native_required", False),
                           ("engine_calls", 0), ("overfit_debug", False)):
        _equal(result.get(name), expected, name)
    status = _required(result, "status", str)
    if status == "interrupted":
        raise ValueError("interrupted rollouts cannot be merged")
    steps = _required(result, "requested_steps", int)
    completed = _required(result, "completed_steps", int)
    if steps < 1 or not 0 <= completed <= steps or status == "completed" and completed != steps:
        raise ValueError("invalid rollout horizon or completed_steps")
    initial_hash = _sha256(result.get("initial_sha256"), "initial_sha256")
    condition = _required(result, "condition_digest", str)
    mesh = _required(result, "mesh_fingerprint", str)
    configuration = _required(result, "configuration", dict)
    if not condition or not mesh or not configuration:
        raise ValueError("condition, mesh and configuration must be recorded")
    evaluation = _required(result, "evaluation", dict)
    _equal(evaluation.get("format"), "fracture-world.dynamics-evaluation.v1", "evaluation format")
    for name, expected in (("method", method), ("case_id", case), ("rollout_status", status),
                           ("condition_digest", condition), ("mesh_fingerprint", mesh),
                           ("evaluation_only_teacher_future", True), ("inference_teacher_future_used", False),
                           ("condition_split", split), ("generalization_protocol", protocol), ("overfit_debug", False)):
        _equal(evaluation.get(name), expected, f"evaluation.{name}")
    last_step = _required(evaluation, "last_compared_native_step", int)
    if not 0 <= last_step <= completed or status == "completed" and last_step != steps:
        raise ValueError("evaluation horizon does not match the requested completed rollout")
    sources = _required(evaluation, "source_hashes", dict)
    teacher = _required(sources, "teacher", dict)
    if not teacher:
        raise ValueError("teacher source provenance must be recorded")
    for source, digest in teacher.items():
        if not isinstance(source, str) or not source:
            raise ValueError("teacher source paths must be nonempty strings")
        _sha256(digest, "teacher source hash")
    return dict(initial_sha256=initial_hash, requested_steps=steps, configuration=configuration,
                condition_digest=condition, mesh_fingerprint=mesh, teacher_source_hashes=teacher)


def merge_dynamics_benchmarks(inputs, output):
    if not isinstance(inputs, (list, tuple)) or len(inputs) < 2:
        raise ValueError("merging requires at least two benchmark directories")
    directories = [Path(path).resolve() for path in inputs]
    if len(set(directories)) != len(directories) or any(not path.is_dir() for path in directories):
        raise ValueError("input benchmark directories must exist and be distinct")
    output = Path(output).resolve()
    if output.exists() or output.suffix.lower() != ".json":
        raise ValueError("output must be a new JSON file")
    contract, configuration = None, None
    case_contracts, rows, summaries, provenance = {}, {}, {}, []
    all_completed = True
    for directory in directories:
        benchmark_path = _safe_report(directory, "benchmark.json")
        benchmark = _read_report(benchmark_path)
        current_contract = _benchmark_contract(benchmark)
        if contract is None:
            contract = current_contract
        else:
            for name, expected in contract.items():
                _equal(current_contract[name], expected, name)
        methods = benchmark["requested_methods"]
        if set(methods).intersection(rows):
            raise ValueError("input benchmarks must have disjoint method sets")
        source_summaries = _required(benchmark, "conditions", dict)
        _equal(set(source_summaries), set(methods), "conditions")
        entries = _required(benchmark, "cases", list)
        if any(not isinstance(entry, dict) for entry in entries):
            raise ValueError("case entries must contain JSON objects")
        _equal([entry.get("case") for entry in entries], contract["requested_case_ids"], "cases")
        rows.update({method: [] for method in methods})
        comparisons = []
        source_completed = True
        for entry in entries:
            case = entry["case"]
            for name, expected in (("status", "comparison_finished"), ("attempted_methods", methods), ("unattempted_methods", [])):
                _equal(entry.get(name), expected, f"case.{name}")
            relative = _required(entry, "directory", str)
            comparison_path = _safe_report(directory, Path(relative) / "comparison" / "comparison.json")
            comparison = _read_report(comparison_path)
            _equal(comparison.get("format"), "fracture-world.neural-comparison.v1", "comparison format")
            for name, expected in (("status", "comparison_finished"), ("engine_calls", 0),
                                   ("numerical_reference_used_for_inference", False), ("requested_methods", methods),
                                   ("attempted_methods", methods), ("unattempted_methods", [])):
                _equal(comparison.get(name), expected, f"comparison.{name}")
            conditions = _required(comparison, "conditions", dict)
            _equal(set(conditions), set(methods), "comparison conditions")
            completed_case = True
            for method in methods:
                result = _required(conditions, method, dict)
                current = _condition_contract(result, method, case, contract["checkpoint_sha256"],
                                              contract["split"], contract["generalization_protocol"])
                if case in case_contracts:
                    for name, expected in case_contracts[case].items():
                        _equal(current[name], expected, f"{case}.{name}")
                else:
                    case_contracts[case] = current
                if configuration is None:
                    configuration = current["configuration"]
                _equal(current["configuration"], configuration, "configuration")
                row = dict(result, benchmark_case_id=case,
                           full_horizon_evaluated=result["status"] == "completed")
                rows[method].append(row)
                completed_case = completed_case and row["full_horizon_evaluated"]
            _equal(comparison.get("all_rollouts_completed"), completed_case, "comparison.all_rollouts_completed")
            _equal(entry.get("all_rollouts_completed"), completed_case, "case.all_rollouts_completed")
            source_completed = source_completed and completed_case
            comparisons.append(dict(case_id=case, path=str(comparison_path), sha256=file_hash(comparison_path)))
        _equal(benchmark.get("all_rollouts_completed"), source_completed, "all_rollouts_completed")
        all_completed = all_completed and source_completed
        for method in methods:
            summary = _required(source_summaries, method, dict)
            for name, expected in (("cases", len(entries)), ("requested_cases", len(entries)),
                                   ("attempted_cases", len(entries)), ("unattempted_cases", 0), ("interrupted_cases", 0),
                                   ("failed_cases", sum(not row["full_horizon_evaluated"] for row in rows[method])),
                                   ("full_horizon_accuracy_cases", sum(row["full_horizon_evaluated"] for row in rows[method]))):
                _equal(summary.get(name), expected, f"{method}.{name}")
            summaries[method] = copy.deepcopy(summary)
        provenance.append(dict(directory=str(directory), benchmark=str(benchmark_path), sha256=file_hash(benchmark_path),
                               methods=methods, comparisons=comparisons))
    methods = list(rows)
    paired = dict(all_methods=_paired_accuracy(rows, methods, _ACCURACY_METRICS),
                  pairwise=[_paired_accuracy(rows, pair, _ACCURACY_METRICS) for pair in combinations(methods, 2)],
                  interpretation="accuracy conditional on common completed horizons; report source completion and failure counts alongside errors",
                  difference_semantics="second method minus first on each common complete case")
    report = dict(schema="fracture-world.merged-dynamics-benchmark.v1", status="merged", **contract,
                  requested_methods=methods, requested_cases=len(case_contracts),
                  requested_method_runs=len(case_contracts) * len(methods), attempted_method_runs=sum(map(len, rows.values())),
                  unattempted_method_runs=0, conditions=summaries, paired_accuracy=paired,
                  configuration=configuration, all_rollouts_completed=all_completed,
                  cases=[dict(case_id=case, **case_contracts[case]) for case in contract["requested_case_ids"]],
                  motion_ablation=_motion_ablation_protocol(methods), proposal_control=_AUTONOMOUS_PROPOSAL_CONTROL,
                  source_benchmarks=provenance, inference_engine_calls=0, merge_simulation_calls=0,
                  runtime_source_revision_match="unknown_not_recorded",
                  provenance_scope="checkpoint, initial-state bytes, recorded numerical configuration, horizon and teacher sources validated; source-code revision is not recorded in legacy reports",
                  reference_used_only_for_evaluation=True,
                  summary_policy="source condition summaries preserved; paired accuracy recomputed from immutable per-case comparison reports")
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("x", encoding="utf-8") as stream:
        json.dump(report, stream, indent=2, allow_nan=False)
    return report
