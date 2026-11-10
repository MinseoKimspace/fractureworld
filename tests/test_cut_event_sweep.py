import csv
import hashlib
import json

import pytest
import torch

from fracture_world.cut_event_sweep import render_worst_cases, run_sweep
from fracture_world.stage_a import main
from test_cut_event_evaluation import checkpoint, sample
from test_cut_event_statistics import condition


@pytest.fixture(autouse=True)
def threads():
    previous = torch.get_num_threads()
    torch.set_num_threads(2)
    yield
    torch.set_num_threads(previous)


def test_plan_only_needs_no_checkpoint_and_creates_no_output(tmp_path, monkeypatch):
    from fracture_world import cut_event_evaluation

    data, _, _ = sample(tmp_path)
    original = set(tmp_path.iterdir())
    monkeypatch.setattr(cut_event_evaluation, "evaluate_cut_event",
                        lambda *args, **kwargs: pytest.fail("Planning ran evaluation"))
    result = run_sweep(data, columns=[2], load_indices=[0], event_times=[0.006, 0.008],
                       output=tmp_path / "unused", plan_only=True)
    assert result["status"] == "planned"
    assert result["scenario_count"] == 2
    assert result["simulation_executed"] is False
    assert result["training_executed"] is False
    assert set(tmp_path.iterdir()) == original
    json.dumps(result, allow_nan=False)


@pytest.mark.parametrize("settings", [
    {}, {"checkpoint": "best.pt"}, {"output": "unused"},
    {"plan_only": True, "render_worst": -1}, {"plan_only": True, "render_worst": 1.5},
    {"plan_only": True, "render_worst": True}, {"plan_only": True, "substeps": 3},
])
def test_invalid_options_fail_before_planning_or_simulation(monkeypatch, settings):
    monkeypatch.setattr("fracture_world.cut_event_sweep.sweep_plan",
                        lambda *args, **kwargs: pytest.fail("Invalid options reached planning"))
    with pytest.raises(ValueError):
        run_sweep("absent-collection.json", **settings)


def test_real_small_sweep_loads_checkpoint_once_and_preserves_existing_inputs(tmp_path, monkeypatch):
    pytest.importorskip("felupe")
    from fracture_world import cut_event_evaluation
    from PIL import Image

    data, _, _ = sample(tmp_path)
    weights = checkpoint(tmp_path)
    originals = {path: hashlib.sha256(path.read_bytes()).hexdigest() for path in tmp_path.iterdir()}
    load_model = cut_event_evaluation.load_model
    calls = []

    def counted_load(*args, **kwargs):
        calls.append(args)
        return load_model(*args, **kwargs)

    monkeypatch.setattr(cut_event_evaluation, "load_model", counted_load)
    output = tmp_path / "sweep"
    result = run_sweep(data, weights, output, columns=[2], load_indices=[0], event_times=[0.006, 0.008],
                       substeps=4, render_worst=1, allow_untrained=True)
    saved = json.loads((output / "report.json").read_text(encoding="utf-8"))
    assert json.loads(json.dumps(result, allow_nan=False)) == saved
    assert result["status"] == "completed"
    assert result["training_executed"] is False
    assert result["training_status"] == "untrained"
    assert result["checkpoint_unchanged"] is True
    assert len(calls) == 1
    assert result["requested_cases"] == 2
    assert result["sweep_summary"]["overall"]["valid_paired_cases"] == 2
    assert len({case["case"] for case in result["cases"]}) == 2
    assert len({case["experiment_hash"] for case in result["cases"]}) == 2
    for case in result["cases"]:
        assert case["sweep"]["column"] == 2
        assert case["reference"]["converged"]
        assert case["neural_event_transfer"]["status"] == "conserved"
        assert case["reference"]["causal_refinement"]["position_intervention_refinement_rms"] >= 0
        assert case["causal_response"]["cut_updated"]["teacher_position_intervention_refinement_rms"] >= 0
    assert result["renders"]["completed"] == 1
    with Image.open(output / result["renders"]["cases"][0]["path"]) as animation:
        assert animation.n_frames == 7
    with (output / "conditions.csv").open(encoding="utf-8-sig", newline="") as handle:
        rows = list(csv.DictReader(handle))
    assert len(rows) == 2
    assert {row["pair_status"] for row in rows} == {"valid"}
    assert {row["event_step"] for row in rows} == {"3", "4"}
    assert all(not isinstance(value, (dict, list)) for row in result["sweep_summary"]["conditions"]
               for value in row.values())
    assert originals == {path: hashlib.sha256(path.read_bytes()).hexdigest() for path in originals}


def test_failed_conditions_remain_in_saved_csv_and_summary(tmp_path, monkeypatch):
    from fracture_world import cut_event_evaluation

    data, _, _ = sample(tmp_path)
    attempted = []

    def evaluate(*args, **kwargs):
        attempted.append(kwargs)
        output = args[2]
        output.mkdir()
        case = {"case": kwargs["scenario_plan"][0]["id"], "status": "reference_failed", "methods": {},
                "inputs": kwargs["scenario_plan"][0]["inputs"], "reference": {"status": "failed", "converged": False}}
        return {"format": "cut-event-evaluation-v1", "status": "incomplete", "cases": [case],
                "training_executed": False, "checkpoint_unchanged": True}

    monkeypatch.setattr(cut_event_evaluation, "evaluate_cut_event", evaluate)
    result = run_sweep(data, "best.pt", tmp_path / "results", columns=[2], load_indices=[0],
                       event_times=[0.006], render_worst=3)
    assert len(attempted) == 1
    assert result["status"] == "incomplete"
    assert result["sweep_summary"]["overall"]["failed_cases"] == 1
    assert result["sweep_summary"]["overall"]["valid_paired_cases"] == 0
    assert result["renders"]["attempted"] == 0
    with (tmp_path / "results" / "conditions.csv").open(encoding="utf-8-sig", newline="") as handle:
        assert list(csv.DictReader(handle))[0]["pair_status"] == "reference_failed"


def test_worst_renderer_ranks_valid_cases_and_reports_render_failures(tmp_path, monkeypatch):
    calls = []
    cases = [condition("small", updated=0.001), condition("large", updated=0.005),
             condition("failed", updated=0.1)]
    cases[-1]["methods"]["cut_updated"]["status"] = "failed"
    for case in cases:
        for label, values in case["methods"].items():
            values["trajectory"] = f"{case['case']}/{label}.npz"

    def render(paths, target, fps):
        calls.append((paths, target, fps))
        raise OSError("Deliberate rendering failure")

    monkeypatch.setattr("fracture_world.render2d.render", render)
    result = render_worst_cases({"cases": cases}, tmp_path, 1)
    assert result["eligible_cases"] == 2
    assert result["attempted"] == 1
    assert result["completed"] == 0
    assert result["cases"][0]["case"] == "large"
    assert result["cases"][0]["status"] == "failed"
    assert "Deliberate" in result["cases"][0]["failure"]
    assert len(calls[0][0]) == 5


def test_cli_plan_only_and_actual_summary_dispatch(tmp_path, monkeypatch, capsys):
    data, _, _ = sample(tmp_path)
    main(["cut-event-sweep", "--data", str(data), "--columns", "2", "--load-indices", "0",
          "--event-times", "0.006", "--plan-only"])
    planned = json.loads(capsys.readouterr().out)
    assert planned["scenario_count"] == 1
    assert planned["substeps"] == 4
    received = {}

    def run(**kwargs):
        received.update(kwargs)
        return {"status": "completed", "report": "result/report.json", "conditions_csv": "result/conditions.csv",
                "training_executed": False, "checkpoint_unchanged": True, "requested_cases": 27,
                "renders": {}, "sweep_runtime_seconds": 1.0, "sweep_summary": {"overall": {"valid_paired_cases": 27}}}

    monkeypatch.setattr("fracture_world.cut_event_sweep.run_sweep", run)
    main(["cut-event-sweep", "--data", str(data), "--checkpoint", "best.pt", "--output", "fresh-result",
          "--device", "cuda", "--render-worst", "2"])
    result = json.loads(capsys.readouterr().out)
    assert received["device"] == "cuda"
    assert received["render_worst"] == 2
    assert received["plan_only"] is False
    assert result["summary"]["valid_paired_cases"] == 27
    assert "cases" not in result
