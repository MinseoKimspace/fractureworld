import json

import pytest

from fracture_world import cli, dynamics_experiment, dynamics_runtime


@pytest.mark.parametrize("command", ["dynamics-rollout", "dynamics-compare", "dynamics-benchmark"])
def test_interrupted_experiment_prints_final_report_and_exits_nonzero(command, tmp_path, monkeypatch, capsys):
    report = dict(status="interrupted", all_rollouts_completed=False, requested_methods=["raw"],
                  cases=[dict(case="first")], conditions={}, requested_cases=3, attempted_cases=1, unattempted_cases=2)
    monkeypatch.setattr(dynamics_runtime, "rollout_dynamics", lambda *args, **kwargs: report)
    monkeypatch.setattr(dynamics_experiment, "compare_dynamics", lambda *args, **kwargs: report)
    monkeypatch.setattr(dynamics_experiment, "benchmark_dynamics", lambda *args, **kwargs: report)
    arguments = [command, "--checkpoint", "model.pt", "--output", str(tmp_path), "--steps", "1"]
    arguments.extend(["--data", "data"] if command == "dynamics-benchmark" else ["--initial", "initial.npz"])
    with pytest.raises(SystemExit) as error:
        cli.run(cli.parser().parse_args(arguments))
    assert error.value.code == 130
    printed = json.loads(capsys.readouterr().out)
    assert printed["status"] == "interrupted"
    if command == "dynamics-benchmark":
        assert printed["attempted_cases"] == 1
        assert printed["unattempted_cases"] == 2


@pytest.mark.parametrize("command", ["dynamics-rollout", "dynamics-compare", "dynamics-benchmark"])
def test_online_cli_defaults_to_the_direct_model(command, tmp_path):
    arguments = [command, "--checkpoint", "model.pt", "--output", str(tmp_path), "--steps", "1"]
    arguments.extend(["--data", "data"] if command == "dynamics-benchmark" else ["--initial", "initial.npz"])
    parsed = cli.parser().parse_args(arguments)
    assert (parsed.method if command == "dynamics-rollout" else parsed.methods) == ("raw" if command == "dynamics-rollout" else ["raw"])


def test_benchmark_cli_forwards_selected_case_ids(tmp_path, monkeypatch, capsys):
    calls = []
    report = dict(status="benchmark_finished", all_rollouts_completed=True, cases=[], conditions={},
                  requested_cases=1, attempted_cases=1, unattempted_cases=0)

    def benchmark(*args, **kwargs):
        calls.append((args, kwargs))
        return report

    monkeypatch.setattr(dynamics_experiment, "benchmark_dynamics", benchmark)
    arguments = ["dynamics-benchmark", "--checkpoint", "model.pt", "--data", "data", "--output", str(tmp_path),
                 "--steps", "1", "--case-ids", "bunny-one"]
    cli.run(cli.parser().parse_args(arguments))
    assert calls[0][0][6] == ["raw"]
    assert calls[0][1] == dict(case_ids=["bunny-one"])
    assert json.loads(capsys.readouterr().out)["requested_cases"] == 1
