import csv
import json
from numbers import Integral
from pathlib import Path
from time import perf_counter

from .cut_event_statistics import condition_rows, summarize_sweep
from .cut_event_sweep_plan import sweep_plan


def render_worst_cases(report, output, count):
    from .render2d import render
    from .training2d import emit

    eligible = {row["case"] for row in condition_rows(report["cases"]) if row["pair_status"] == "valid"}
    ranked = sorted((case for case in report["cases"] if case["case"] in eligible and
                     case.get("methods", {}).get("no_cut", {}).get("status") == "completed"),
                    key=lambda case: (-case["methods"]["cut_updated"]["post_event_velocity_rmse"], case["case"]))
    renders = []
    for case in ranked[:count]:
        destination = output / case["case"]
        paths = [destination / "teacher_cut.npz", output / case["methods"]["cut_updated"]["trajectory"],
                 output / case["methods"]["cut_stale"]["trajectory"], destination / "teacher_no_cut.npz",
                 output / case["methods"]["no_cut"]["trajectory"]]
        target = destination / "comparison.gif"
        entry = {"case": case["case"], "ranking_metric": "cut_updated_post_event_velocity_rmse"}
        emit("cut_event_render_started", case=case["case"])
        try:
            render(paths, target, fps=30)
        except (RuntimeError, ValueError, OSError, FloatingPointError) as error:
            entry.update(status="failed", failure=f"{type(error).__name__}: {error}")
        else:
            entry.update(status="completed", path=str(target.relative_to(output)))
            case["render"] = entry["path"]
        renders.append(entry)
    return {"requested": count, "eligible_cases": len(ranked), "attempted": len(renders),
            "completed": sum(row["status"] == "completed" for row in renders), "cases": renders}


def run_sweep(data, checkpoint=None, output=None, columns=None, load_indices=None, event_times=None,
              horizon=None, substeps=4, device="cpu", plan_only=False, render_worst=3, allow_untrained=False):
    started = perf_counter()
    if isinstance(render_worst, bool) or not isinstance(render_worst, Integral) or render_worst < 0:
        raise ValueError("The number of worst-case renders must be a nonnegative integer")
    if isinstance(substeps, bool) or substeps not in (1, 2, 4):
        raise ValueError("Substeps must be 1, 2, or 4")
    if not plan_only and (checkpoint is None or output is None):
        raise ValueError("Evaluation requires --checkpoint and --output; use --plan-only to inspect conditions")
    plan, description = sweep_plan(data, columns, load_indices, event_times, horizon)
    if plan_only:
        return {**description, "status": "planned", "training_executed": False,
                "simulation_executed": False, "substeps": substeps,
                "conditions": [{"case": row["id"], "experiment_hash": row["experiment_hash"],
                                "inputs": row["inputs"], "sweep": row["sweep"]} for row in plan]}
    from .cut_event_evaluation import evaluate_cut_event

    output = Path(output)
    report = evaluate_cut_event(data, checkpoint, output, device=device, substeps=substeps,
                                allow_untrained=allow_untrained, scenario_plan=plan)
    report["base_evaluation_format"] = report["format"]
    report["format"] = "cut-event-sweep-evaluation-v1"
    report["sweep_plan"] = description
    report["sweep_summary"] = summarize_sweep(report["cases"])
    report["timing_scope"] = {
        "neural_total": "Actual per-condition neural evaluation, counting the shared prefix once",
        "method_path": "Shared prefix plus one post-event branch; do not sum these as actual suite runtime",
        "reference": "Separate FElupe/DOP853 numerical reference generation",
        "comparison": "Fixed final horizons; earlier cuts have longer post-event evaluation windows",
    }
    rows = condition_rows(report["cases"])
    with (output / "conditions.csv").open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(dict.fromkeys(key for row in rows for key in row)))
        writer.writeheader()
        writer.writerows(rows)
    report["conditions_csv"] = str((output / "conditions.csv").resolve())
    report["report"] = str((output / "report.json").resolve())
    report["renders"] = render_worst_cases(report, output, render_worst) if render_worst else {
        "requested": 0, "attempted": 0, "completed": 0, "cases": [],
    }
    report["sweep_runtime_seconds"] = perf_counter() - started
    (output / "report.json").write_text(json.dumps(report, indent=2, allow_nan=False), encoding="utf-8")
    return report
