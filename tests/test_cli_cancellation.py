import json

import pytest

from fracture_world import cli, dynamics_experiment, dynamics_runtime


@pytest.mark.parametrize("command", ["dynamics-rollout", "dynamics-compare", "dynamics-benchmark"])
def test_interrupted_experiment_prints_final_report_and_exits_nonzero(command, tmp_path, monkeypatch, capsys):
    report = dict(status="interrupted", all_rollouts_completed=False, central_consensus_agreement=None,
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
