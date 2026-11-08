import json
from pathlib import Path

import torch

from .audit2d import audit_teacher
from .benchmark2d import benchmark
from .generation2d import cut_mesh, generate, smoke_configuration, times
from .neural import DynamicsModel
from .render2d import render
from .teacher2d import Material


def validate(config, output, device="cpu"):
    output = Path(output)
    if output.exists() and any(output.iterdir()):
        raise ValueError("Validation output must be a new or empty directory")
    output.mkdir(parents=True, exist_ok=True)
    column = config["cut_columns"][0]
    depth = config["cut_depths"]["train"][-1]
    audit, reference = audit_teacher(
        cut_mesh(config, column, depth), Material(**config["material"]), times(config),
        **config["loads"][0], **config["integration"],
    )
    reference.save(output / "reference.npz")
    (output / "teacher-audit.json").write_text(json.dumps(audit, indent=2, allow_nan=False), encoding="utf-8")
    if not audit["passed"]:
        raise RuntimeError("Numerical reference audit failed; inspect teacher-audit.json")
    small_config = smoke_configuration(config)
    data = generate(small_config, output / "data")
    paths = []
    for method in ("direct", "stress"):
        model = DynamicsModel(method=method, hidden_size=32, layers=2)
        path = output / f"untrained-{method}.pt"
        torch.save({
            "format": "cut-dynamics-a-v1", "trained": False,
            "model_config": model.configuration(), "state_dict": model.state_dict(),
            "dt": config["time"]["step"], "training_case_hashes": [], "training_groups": [],
            "purpose": "Untrained pipeline validation; no optimizer or training executed",
        }, path)
        paths.append(path)
    results = benchmark(data["collection"], paths, output / "inference", device=device,
                        allow_untrained=True)
    manifest = json.loads(Path(data["collection"]).read_text(encoding="utf-8"))
    first = next(row for row in manifest["cases"] if row["split"] == "test")
    trajectories = [output / "data" / first["path"]]
    for method in ("direct", "stress"):
        trajectories.append(output / "inference" / results["methods"][method]["cases"][0]["trajectory"])
    render(trajectories, output / "comparison.gif", fps=30, frame_stride=2)
    render(trajectories, output / "comparison.png")
    completed = all(method["summary"]["failed_cases"] == 0 for method in results["methods"].values())
    report = {
        "passed": audit["passed"] and completed,
        "purpose": "Execution and numerical-reference validation only; no learned accuracy claim",
        "training_executed": False, "teacher_audit": audit,
        "generated_cases": data["cases"], "inference": {
            name: result["summary"] for name, result in results["methods"].items()
        }, "artifacts": str(output.resolve()),
    }
    (output / "validation.json").write_text(json.dumps(report, indent=2, allow_nan=False), encoding="utf-8")
    return report
