import json
from pathlib import Path

import pytest

from fracture_world import dynamics_learning
from fracture_world.cli import parser, run


@pytest.mark.parametrize("options,skip_baseline", [([], None), (["--skip-baseline"], True)])
def test_corrector_cli_forwards_skip_baseline_only_when_requested(monkeypatch, capsys, options, skip_baseline):
    calls = []

    def train(data, checkpoint, output, **keywords):
        calls.append((data, checkpoint, output, keywords))
        return output / "best.pt"

    monkeypatch.setattr(dynamics_learning, "train_neural_corrector", train)
    arguments = parser().parse_args(["dynamics-train-corrector", "--data", "data", "--checkpoint", "predictor.pt",
                                    "--output", "output", "--epochs", "5", *options])
    assert arguments.skip_baseline is skip_baseline
    assert arguments.baseline_checkpoint is None
    run(arguments)
    expected = dict(epochs=5, device="auto", stream_data=False, rollout_steps=None, learning_rate=None)
    if skip_baseline is not None:
        expected["skip_baseline"] = True
    assert calls == [(Path("data"), Path("predictor.pt"), Path("output"), expected)]
    assert json.loads(capsys.readouterr().out) == {"checkpoint": str(Path("output") / "best.pt")}


def test_corrector_cli_rejects_skipping_and_importing_baseline_together(capsys):
    with pytest.raises(SystemExit) as error:
        parser().parse_args(["dynamics-train-corrector", "--data", "data", "--checkpoint", "predictor.pt",
                             "--output", "output", "--epochs", "5", "--skip-baseline",
                             "--baseline-checkpoint", "previous/best.pt"])
    assert error.value.code == 2
    assert "not allowed with argument" in capsys.readouterr().err
