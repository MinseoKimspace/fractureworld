import json

import numpy as np
from numpy.testing import assert_allclose
import pytest

from fracture_world.dynamics_data import load_dynamics_trajectory
from fracture_world.dynamics_evaluation import evaluate_dynamics
from test_dynamics_data import recorded_trajectory


def saved_rollout(directory, teacher, transform=None, report_update=None):
    directory.mkdir()
    report = dict(format="fracture-world.neural-rollout.v1", method="raw", status="completed", engine_calls=0,
                  teacher_future_used=False, mesh_fingerprint=teacher.topology.fingerprint,
                  case_id=teacher.case_id, condition_digest=teacher.condition_digest,
                  completed_steps=len(teacher), condition_split="test", overfit_debug=False, events=[])
    arrays = {key: getattr(teacher, key).copy() for key in ("times", "positions", "velocities", "broken", "fixed")}
    if transform:
        transform(arrays)
    if report_update:
        report.update(report_update)
    report["completed_steps"] = len(arrays["times"]) - 1
    arrays["metadata_json"] = np.asarray(json.dumps({key: value for key, value in report.items() if key != "events"}))
    (directory / "report.json").write_text(json.dumps(report), encoding="utf-8")
    np.savez_compressed(directory / "trajectory.npz", **arrays)
    return directory


def test_exact_numerical_teacher_scores_zero_without_modifying_inference_files(tmp_path):
    case, trajectory = recorded_trajectory(tmp_path / "source")
    teacher = load_dynamics_trajectory(case, trajectory)
    rollout = saved_rollout(tmp_path / "rollout", teacher)
    report_before = (rollout / "report.json").read_bytes()
    archive_before = (rollout / "trajectory.npz").read_bytes()
    result = evaluate_dynamics(rollout, case, trajectory)
    assert result["position_mass_weighted_rmse_m"] == 0
    assert result["velocity_mass_weighted_rmse_m_per_s"] == 0
    assert result["component_count_absolute_error"] == 0
    assert result["fracture"]["f1"] == 1
    assert result["fracture"]["positive_event_comparison"]
    assert result["fracture_timing"]["mean_absolute_observed_time_error_s"] == 0
    assert result["fracture_timing"]["matched_events"] == 3
    assert result["compared_future_frames"] == 3
    assert result["held_out_condition"]
    assert result["engine_calls"] == 0
    assert "not physical ground truth" in result["reference"]
    assert (rollout / "report.json").read_bytes() == report_before
    assert (rollout / "trajectory.npz").read_bytes() == archive_before


def test_mass_weighted_motion_metrics_and_missed_fracture_events_are_measured(tmp_path):
    case, trajectory = recorded_trajectory(tmp_path / "source")
    teacher = load_dynamics_trajectory(case, trajectory)

    def modify(arrays):
        arrays["positions"][1:, :, 0] += 0.02
        arrays["velocities"][1:, :, 1] += 0.3
        arrays["broken"][:] = False

    rollout = saved_rollout(tmp_path / "rollout", teacher, modify, {"overfit_debug": True})
    result = evaluate_dynamics(rollout, case)
    assert_allclose(result["position_mass_weighted_rmse_m"], 0.02, rtol=0, atol=1e-16)
    assert_allclose(result["velocity_mass_weighted_rmse_m_per_s"], 0.3, rtol=0, atol=1e-16)
    assert_allclose(result["linear_momentum_error_kg_m_per_s"], teacher.topology.masses.sum() * 0.3)
    assert result["fracture"]["recall"] == 0
    assert result["fracture_timing"]["missed_events"] == 3
    assert result["frames"][-1]["component_count_absolute_error"] == 2
    assert not result["held_out_condition"]


def test_support_release_mismatches_are_scored_without_becoming_runtime_inputs(tmp_path):
    case, trajectory = recorded_trajectory(tmp_path / "source", supports=True)
    teacher = load_dynamics_trajectory(case, trajectory)

    def modify(arrays):
        arrays["fixed"][:] = teacher.fixed[0]

    rollout = saved_rollout(tmp_path / "rollout", teacher, modify)
    result = evaluate_dynamics(rollout, case)
    assert result["support_mismatched_corners_total"] == 2
    assert result["inference_teacher_future_used"] is False
    assert result["evaluation_only_teacher_future"] is True


def test_float32_clock_roundoff_and_unmatched_future_horizon_are_reported(tmp_path):
    case, trajectory = recorded_trajectory(tmp_path / "source")
    teacher = load_dynamics_trajectory(case, trajectory)

    def modify(arrays):
        arrays["times"] = np.r_[arrays["times"], 0.5, 0.625]
        arrays["times"][1] = np.nextafter(np.float32(0.125), np.float32(1.0))
        for key in ("positions", "velocities", "broken", "fixed"):
            arrays[key] = np.concatenate([arrays[key], np.repeat(arrays[key][-1:], 2, axis=0)])

    rollout = saved_rollout(tmp_path / "rollout", teacher, modify)
    output = tmp_path / "evaluation.json"
    result = evaluate_dynamics(rollout, case, output_file=output)
    assert result["unmatched_rollout_frames"] == 2
    assert result["compared_future_frames"] == 3
    assert json.loads(output.read_text(encoding="utf-8"))["fracture"]["f1"] == 1
    with pytest.raises(ValueError, match="already exists"):
        evaluate_dynamics(rollout, case, output_file=output)


@pytest.mark.parametrize("corruption,match", [
    ("healing", "irreversible"), ("supports", "activate new fixed"), ("clock", "timestep clock"),
    ("shape", "invalid dimensions"), ("initial", "initial positions"), ("mesh", "material mesh"),
    ("condition", "loading condition"), ("future_input", "no-future-input"),
])
def test_invalid_rollouts_and_incompatible_teacher_associations_are_rejected(tmp_path, corruption, match):
    case, trajectory = recorded_trajectory(tmp_path / "source")
    teacher = load_dynamics_trajectory(case, trajectory)

    def modify(arrays):
        if corruption == "healing":
            arrays["broken"][-1, 0] = False
        elif corruption == "supports":
            arrays["fixed"][-1, 0] = True
        elif corruption == "clock":
            arrays["times"][1] += 0.01
        elif corruption == "shape":
            arrays["positions"] = arrays["positions"][:, :-1]
        elif corruption == "initial":
            arrays["positions"][0] += [0.001, 0, 0]

    updates = {}
    if corruption == "mesh":
        updates["mesh_fingerprint"] = "0" * 64
    elif corruption == "condition":
        updates["condition_digest"] = "0" * 64
    elif corruption == "future_input":
        updates["teacher_future_used"] = True
    rollout = saved_rollout(tmp_path / "rollout", teacher, modify, updates)
    with pytest.raises(ValueError, match=match):
        evaluate_dynamics(rollout, case)


def test_initial_only_rollout_does_not_claim_predictive_accuracy(tmp_path):
    case, trajectory = recorded_trajectory(tmp_path / "source")
    teacher = load_dynamics_trajectory(case, trajectory)

    def modify(arrays):
        for key in arrays:
            arrays[key] = arrays[key][:1]

    rollout = saved_rollout(tmp_path / "rollout", teacher, modify)
    result = evaluate_dynamics(rollout, case)
    assert result["compared_future_frames"] == 0
    assert result["position_mass_weighted_rmse_m"] is None
    assert result["velocity_mass_weighted_rmse_m_per_s"] is None
