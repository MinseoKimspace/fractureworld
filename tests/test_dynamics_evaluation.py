import json

import numpy as np
from numpy.testing import assert_allclose
import pytest

from fracture_world.dynamics_data import load_dynamics_trajectory
from fracture_world.dynamics_evaluation import _classification, _energy_interval, _new_fracture_classification, evaluate_dynamics
from fracture_world.dynamics_experiment import _checked_constraint_summary
from test_dynamics_data import recorded_trajectory
from test_dynamics_energy import energy_fixture


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
    assert result["new_fracture"]["f1"] == 1
    assert result["final_fracture"]["f1"] == 1
    assert result["final_new_fracture"]["f1"] == 1
    assert result["fracture"]["positive_event_comparison"]
    assert result["fracture_timing"]["mean_absolute_observed_time_error_s"] == 0
    assert result["fracture_timing"]["matched_events"] == 3
    assert result["compared_future_frames"] == 3
    assert result["held_out_condition"]
    assert result["engine_calls"] == 0
    assert result["kinetic_energy_absolute_error_j"] == 0
    assert result["elastic_energy_absolute_error_j"] == 0
    assert result["gravitational_energy_absolute_error_j"] == 0
    assert result["total_mechanical_energy_absolute_error_j"] == 0
    assert result["maximum_positive_mechanical_energy_increment_j"] == result["maximum_reference_positive_mechanical_energy_increment_j"]
    assert result["final_energy_components_j"] == result["final_reference_energy_components_j"]
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
    assert result["new_fracture"]["f1"] is None
    assert result["final_new_fracture"]["f1"] is None
    assert result["final_fracture"]["availability"] == "observed_initial_only"


def test_observed_fractures_cannot_hide_a_completely_missed_new_event():
    initial = np.r_[np.ones(99, dtype=bool), False]
    predicted = np.broadcast_to(initial, (3, 100))
    target = predicted.copy()
    target[-1, -1] = True
    assert _classification(predicted, target)["f1"] > 0.99
    result = _new_fracture_classification(predicted, target, initial)
    assert result["f1"] == 0
    assert result["true_positive"] == 0
    assert result["false_negative"] == 1
    assert result["initially_broken_facets_excluded"] == 99


def test_new_fracture_scores_future_only_for_a_later_observed_initial_state(tmp_path):
    case, trajectory = recorded_trajectory(tmp_path / "source")
    teacher = load_dynamics_trajectory(case, trajectory)

    def modify(arrays):
        for key in arrays:
            arrays[key] = arrays[key][1:]
        arrays["broken"][:] = arrays["broken"][0]

    rollout = saved_rollout(tmp_path / "rollout", teacher, modify)
    result = evaluate_dynamics(rollout, case, trajectory)
    assert result["initial_teacher_frame"] == 1
    assert result["new_fracture"]["compared_future_states"] == 2
    assert result["new_fracture"]["f1"] == 0
    assert result["new_fracture"]["target_broken_facets"] == 3
    assert result["new_fracture"]["initially_broken_facets_excluded"] == 1
    assert result["final_fracture"]["true_positive"] == 1
    assert result["final_new_fracture"]["true_positive"] == 0
    assert result["final_new_fracture"]["false_negative"] == 2
    assert result["frames"][0]["new_fracture"] is None


def test_preexisting_cuts_do_not_count_as_spurious_new_fractures():
    initial = np.r_[np.ones(99, dtype=bool), False]
    target = initial[None]
    predicted = target.copy()
    exact = _new_fracture_classification(predicted, target, initial)
    assert exact["predicted_broken_facets"] == 0
    assert exact["target_broken_facets"] == 0
    predicted[0, -1] = True
    spurious = _new_fracture_classification(predicted, target, initial)
    assert spurious["f1"] == 0
    assert spurious["false_positive"] == 1


def test_spontaneous_predicted_energy_is_measured_separately_from_reference_energy(tmp_path):
    case, trajectory = recorded_trajectory(tmp_path / "source")
    teacher = load_dynamics_trajectory(case, trajectory)

    def modify(arrays):
        arrays["velocities"][1:, ::2, 0] += 10
        arrays["velocities"][1:, 1::2, 0] -= 10

    rollout = saved_rollout(tmp_path / "rollout", teacher, modify, {"energy_velocity_scale": 1.0})
    result = evaluate_dynamics(rollout, case, trajectory)
    assert result["mean_kinetic_energy_j"] > result["mean_reference_kinetic_energy_j"]
    assert result["mean_total_energy_j"] > result["mean_reference_total_energy_j"]
    assert result["maximum_positive_mechanical_energy_increment_j"] > result["maximum_reference_positive_mechanical_energy_increment_j"]
    assert result["maximum_relative_positive_mechanical_energy_increment"] > 1
    assert result["total_mechanical_energy_absolute_error_j"] > 1
    assert result["elastic_energy_absolute_error_j"] == 0
    assert result["gravitational_energy_absolute_error_j"] == 0
    assert result["relative_energy_increment_intervals"] == 3
    assert result["reference_relative_energy_increment_intervals"] == 3
    assert "recorded velocity scale" in result["relative_energy_scale"]
    assert "not physical ground truth" in result["reference"]


def test_zero_energy_budget_has_explicit_unknown_relative_scale_without_a_recorded_velocity_scale():
    _, energy, positions, velocities = energy_fixture()
    previous = energy.components(positions, velocities)
    later_velocities = velocities.copy()
    later_velocities[..., 0] = np.sqrt(2 / energy.masses.sum())
    _, legacy = _energy_interval(energy, positions, velocities, positions, later_velocities, previous)
    assert legacy["positive_increment_j"] == pytest.approx(1)
    assert legacy["relative_positive_increment"] is None
    assert legacy["budget_scale_j"] == 0
    _, recorded = _energy_interval(energy, positions, velocities, positions, later_velocities, previous, 0.5)
    assert recorded["budget_scale_j"] == pytest.approx(energy.masses.sum() * 0.25)
    assert recorded["relative_positive_increment"] == pytest.approx(1 / recorded["budget_scale_j"])


def test_evaluation_energy_increment_scale_is_independent_of_potential_energy_origin():
    _, energy, rest, _ = energy_fixture((0, -9.81, 0))
    old_velocity = np.broadcast_to([0.5, 0, 0], rest.shape).copy()
    new_velocity = old_velocity + [1, 0, 0]
    records = []
    for elevation in (0, 1.e6):
        previous_positions = rest + [0, elevation, 0]
        current_positions = previous_positions + [0, -0.001, 0]
        previous = energy.components(previous_positions, old_velocity)
        _, increment = _energy_interval(energy, previous_positions, old_velocity, current_positions, new_velocity, previous, 1.0)
        records.append(increment)
    assert_allclose(records[0]["positive_increment_j"], records[1]["positive_increment_j"], rtol=1e-8, atol=1e-9)
    assert_allclose(records[0]["budget_scale_j"], records[1]["budget_scale_j"], rtol=1e-12)
    assert_allclose(records[0]["relative_positive_increment"], records[1]["relative_positive_increment"], rtol=1e-8, atol=1e-9)


def test_unknown_constraint_diagnostics_never_count_as_verified_success():
    valid = dict(status="completed", all_committed_geometry_valid=True, all_committed_energy_valid=True,
                 all_committed_swept_geometry_valid=True)
    unknown = dict(status="completed", all_committed_geometry_valid=True)
    failed_energy = {**valid, "all_committed_energy_valid": False}
    incomplete = {**valid, "status": "timestep_rejected"}
    result = _checked_constraint_summary([valid, unknown, failed_energy, incomplete])
    assert result["fully_checked_constraint_cases"] == 3
    assert result["constraint_status_unknown_cases"] == 1
    assert result["completed_and_checked_constraints_valid_rate"] == 0.25
    assert _checked_constraint_summary([unknown])["completed_and_checked_constraints_valid_rate"] == 0
    assert _checked_constraint_summary([])["completed_and_checked_constraints_valid_rate"] is None
