from fracture_world.stage_a import main


def test_diagnose_dispatches_without_starting_training(monkeypatch, tmp_path):
    captured = {}

    def forbidden(*args, **kwargs):
        raise AssertionError("Diagnostics must not start training")

    def diagnose(**kwargs):
        captured.update(kwargs)
        return {"training_executed": False, "status": "completed"}

    monkeypatch.setattr("fracture_world.training2d.fit", forbidden)
    monkeypatch.setattr("fracture_world.diagnostics2d.diagnose", diagnose)
    main(["diagnose", "--data", "collection.json", "--report", "report.json",
          "--output", str(tmp_path), "--device", "cuda"])
    assert captured == {
        "data": "collection.json", "report": "report.json",
        "output": str(tmp_path), "device": "cuda", "case_limit": None,
    }
