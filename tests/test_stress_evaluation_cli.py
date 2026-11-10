import json
import sys
from types import ModuleType

import pytest

from fracture_world.stage_a import main


def install_evaluation(monkeypatch, received):
    module = ModuleType("fracture_world.stress_evaluation")

    def evaluate_stress(**kwargs):
        received.update(kwargs)
        return {"status": "completed", "training_executed": False}

    def forbidden(*args, **kwargs):
        raise AssertionError("Stress evaluation must not start training")

    module.evaluate_stress = evaluate_stress
    monkeypatch.setitem(sys.modules, module.__name__, module)
    monkeypatch.setattr("fracture_world.training2d.fit", forbidden)


def test_stress_evaluation_cli_defaults_do_not_train(monkeypatch, capsys):
    received = {}
    install_evaluation(monkeypatch, received)
    main(["stress-evaluate", "--checkpoint", "best.pt", "--data", "collection.json",
          "--output", "new-evaluation"])
    assert received == {
        "checkpoint": "best.pt", "data": "collection.json", "output": "new-evaluation",
        "device": "cpu", "source_cases": None, "horizon": 1.0, "case_limit": None,
        "groups": ["force", "long", "topology", "load"], "render_results": False,
    }
    assert json.loads(capsys.readouterr().out) == {
        "status": "completed", "training_executed": False,
    }


def test_stress_evaluation_cli_explicit_selection(monkeypatch, capsys):
    received = {}
    install_evaluation(monkeypatch, received)
    main(["stress-evaluate", "--checkpoint", "best.pt", "--data", "collection.json",
          "--output", "new-evaluation", "--device", "cuda", "--source-cases",
          "x7-depth7-load12", "x3-depth8-load12", "--horizon", "1.2", "--case-limit", "2",
          "--groups", "force", "long", "--render"])
    assert received["device"] == "cuda"
    assert received["source_cases"] == ["x7-depth7-load12", "x3-depth8-load12"]
    assert received["horizon"] == 1.2
    assert received["case_limit"] == 2
    assert received["groups"] == ["force", "long"]
    assert received["render_results"] is True
    assert json.loads(capsys.readouterr().out)["training_executed"] is False


def test_stress_evaluation_cli_rejects_unknown_group():
    with pytest.raises(SystemExit) as error:
        main(["stress-evaluate", "--checkpoint", "best.pt", "--data", "collection.json",
              "--output", "new-evaluation", "--groups", "contact"])
    assert error.value.code == 2
