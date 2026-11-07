import json
from pathlib import Path

import pytest

from fracture_world import dynamics_learning
from fracture_world.cli import parser, run


@pytest.mark.parametrize("baseline", [None, "previous/best.pt"])
def test_corrector_cli_passes_baseline_checkpoint_only_when_provided(monkeypatch, capsys, baseline):
    calls = []

    def train(data, checkpoint, output, **options):
        calls.append((data, checkpoint, output, options))
        return output / "best.pt"

    monkeypatch.setattr(dynamics_learning, "train_neural_corrector", train)
    baseline_options = [] if baseline is None else ["--baseline-checkpoint", baseline]
    arguments = parser().parse_args(["dynamics-train-corrector", "--data", "data", "--checkpoint", "predictor.pt",
                                    "--output", "output", "--epochs", "5", *baseline_options])
    assert arguments.baseline_checkpoint == (None if baseline is None else Path(baseline))
    run(arguments)
    expected = dict(epochs=5, device="auto", stream_data=False, rollout_steps=None, learning_rate=None)
    if baseline is not None:
        expected["baseline_checkpoint"] = Path(baseline)
    assert calls == [(Path("data"), Path("predictor.pt"), Path("output"), expected)]
    assert json.loads(capsys.readouterr().out) == {"checkpoint": str(Path("output") / "best.pt")}
