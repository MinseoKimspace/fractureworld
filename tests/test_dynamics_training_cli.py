import json
from pathlib import Path

import pytest

from fracture_world import dynamics_learning
from fracture_world.cli import parser, run
from fracture_world.dynamics_learning import DynamicsTrainingConfig


TRAINING_DEFAULTS = dict(batch_size=1, lr_patience=0, lr_factor=0.5, min_learning_rate=0.000001,
                         early_stopping_patience=0, min_improvement=0.0)
TRAINING_OPTIONS = ["--batch-size", "8", "--lr-patience", "3", "--lr-factor", "0.25",
                    "--min-learning-rate", "0.000002", "--early-stopping-patience", "7",
                    "--min-improvement", "0.01"]
TRAINING_OVERRIDES = dict(batch_size=8, lr_patience=3, lr_factor=0.25, min_learning_rate=0.000002,
                          early_stopping_patience=7, min_improvement=0.01)


@pytest.mark.parametrize("options,expected", [([], TRAINING_DEFAULTS), (TRAINING_OPTIONS, TRAINING_OVERRIDES)])
def test_training_cli_passes_batch_scheduler_and_stopping_options(monkeypatch, capsys, options, expected):
    calls = []

    def train(data, output, config, *, resume):
        calls.append((data, output, config, resume))
        return output / "best.pt"

    monkeypatch.setattr(dynamics_learning, "train_dynamics", train)
    arguments = parser().parse_args(["dynamics-train", "--data", "data", "--output", "output",
                                    "--epochs", "5", *options])
    run(arguments)
    assert len(calls) == 1
    data, output, config, resume = calls[0]
    assert (data, output, resume) == (Path("data"), Path("output"), None)
    assert isinstance(config, DynamicsTrainingConfig)
    assert {name: getattr(config, name) for name in expected} == expected
    assert json.loads(capsys.readouterr().out) == {"checkpoint": str(Path("output") / "best.pt")}


@pytest.mark.parametrize("options,expected", [([], {}), (TRAINING_OPTIONS, TRAINING_OVERRIDES)])
@pytest.mark.parametrize("resume", [None, "output/last.pt"])
def test_corrector_cli_inherits_or_overrides_training_options(monkeypatch, capsys, options, expected, resume):
    calls = []

    def train(data, checkpoint, output, **kwargs):
        calls.append((data, checkpoint, output, kwargs))
        return output / "best.pt"

    monkeypatch.setattr(dynamics_learning, "train_neural_corrector", train)
    resume_options = [] if resume is None else ["--resume", resume]
    arguments = parser().parse_args(["dynamics-train-corrector", "--data", "data", "--checkpoint", "predictor.pt",
                                    "--output", "output", "--epochs", "5", *options, *resume_options])
    if not options:
        assert all(getattr(arguments, name) is None for name in TRAINING_DEFAULTS)
    run(arguments)
    keywords = dict(epochs=5, device="auto", stream_data=False, rollout_steps=None, learning_rate=None, **expected)
    if resume is not None:
        keywords["resume"] = Path(resume)
    assert calls == [(Path("data"), Path("predictor.pt"), Path("output"), keywords)]
    assert json.loads(capsys.readouterr().out) == {"checkpoint": str(Path("output") / "best.pt")}
