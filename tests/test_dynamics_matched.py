from dataclasses import replace
import json
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from fracture_world import dynamics_matched
from fracture_world.dynamics_initial import load_initial
from fracture_world.dynamics_learning import load_dynamics_model
from fracture_world.dynamics_matched import _constraint_status, compare_matched_dynamics, compare_matched_step
from fracture_world.dynamics_runtime import RolloutConfig
from test_dynamics_data import recorded_trajectory
from test_dynamics_runtime import portable_rollout_inputs


def matched_inputs(tmp_path):
    checkpoint, path, _, _ = portable_rollout_inputs(tmp_path, has_corrector=False)
    model, _, saved, _ = load_dynamics_model(checkpoint, "cpu")
    return checkpoint, load_initial(path), model, saved


def test_matched_methods_receive_identical_frozen_candidates_and_only_one_encoding(tmp_path, monkeypatch):
    _, initial, model, _ = matched_inputs(tmp_path)
    encode, correct = model.encode, dynamics_matched.runtime._correct
    encoded, corrected = [], []

    def capture_encode(batch, *args, **kwargs):
        encoded.append(batch["positions"].clone())
        return encode(batch, *args, **kwargs)

    def capture_correct(observation, positions, velocities, proposal, mask, fixed, network, config, method, *args, **kwargs):
        assert not positions.flags.writeable and not velocities.flags.writeable
        assert not mask.flags.writeable and not fixed.flags.writeable
        predicted = dynamics_matched.runtime._proposal_arrays(proposal, positions, velocities)
        corrected.append((method, mask.copy(), fixed.copy(), *(values.copy() for values in predicted)))
        return correct(observation, positions, velocities, proposal, mask, fixed, network, config, method, *args, **kwargs)

    monkeypatch.setattr(model, "encode", capture_encode)
    monkeypatch.setattr(dynamics_matched.runtime, "_correct", capture_correct)
    config = RolloutConfig(fracture_samples=4, fracture_policy="fm_search", flow_steps=1,
                           max_iterations=10000, max_contact_iterations=3)
    report = compare_matched_step(initial, model, config)
    assert report["status"] == "completed"
    assert len(encoded) == report["predictor_calls"] == 1
    assert report["attempted_method_runs"] == 3 * report["candidate_count"]
    assert report["unattempted_method_runs"] == 0
    assert len(corrected) == 2 * report["candidate_count"]
    for central, physical in zip(corrected[::2], corrected[1::2], strict=True):
        assert central[0] == physical[0] == "central"
        for first, second in zip(central[1:], physical[1:], strict=True):
            np.testing.assert_array_equal(first, second)
    for index in range(report["candidate_count"]):
        rows = [item for item in report["candidates"] if item["candidate"] == index]
        assert {item["method"] for item in rows} == {"raw", "central", "no_motion_central"}
        assert len({item["input_hash"] for item in rows}) == 1
        assert len({item["proposal_hash"] for item in rows}) == 1
        assert len({item["candidate_hash"] for item in rows}) == 1
    assert not report["reference_used_for_proposals"]
    assert not report["reference_used_for_selection"]


def test_teacher_future_is_requested_only_after_all_solver_runs_and_cannot_change_selection(tmp_path, monkeypatch):
    _, initial, model, _ = matched_inputs(tmp_path)
    config = RolloutConfig(fracture_samples=0, fracture_policy="marginal_search", max_iterations=10000, max_contact_iterations=3)
    completed = []
    correct = dynamics_matched.runtime._correct

    def capture(*args, **kwargs):
        result = correct(*args, **kwargs)
        completed.append(args[8])
        return result

    monkeypatch.setattr(dynamics_matched.runtime, "_correct", capture)

    def future(offset, cut):
        assert len(completed) == 4
        return dict(positions=initial.positions + offset, velocities=np.zeros_like(initial.velocities),
                    broken=np.full_like(initial.broken, cut), fixed=initial.fixed.copy())

    first = compare_matched_step(initial, model, config, reference=lambda: future(0., False))
    completed.clear()
    second = compare_matched_step(initial, model, config, reference=lambda: future(10., True))
    assert first["input_hash"] == second["input_hash"]
    assert first["candidate_pool"] == second["candidate_pool"]
    for method in ("raw", "central", "no_motion_central"):
        assert first["methods"][method]["selected_candidate"] == second["methods"][method]["selected_candidate"]
        assert first["methods"][method]["evaluation"] != second["methods"][method]["evaluation"]


def test_failed_and_unknown_candidates_are_not_selected_or_counted_as_verified(tmp_path, monkeypatch):
    _, initial, model, _ = matched_inputs(tmp_path)
    monkeypatch.setattr(dynamics_matched.runtime, "_correct", lambda *args, **kwargs:
                        (None, dict(status="feasibility_unknown", solves=[])))
    config = RolloutConfig(fracture_samples=0, fracture_policy="marginal_search")
    report = compare_matched_step(initial, model, config)
    for method in ("central", "no_motion_central"):
        summary = report["methods"][method]
        assert summary["selected_candidate"] is None
        assert summary["verified_constraints"] is None
        for record in (item for item in report["candidates"] if item["method"] == method):
            assert not record["accepted"]
            assert record["selection_objective"] is None
            assert record["verified_constraints"] is None
    assert report["methods"]["raw"]["verified_constraints"] is None
    assert all(not item["comparison_available"] for item in report["selected_state_comparisons"])


def test_prescribed_supports_override_a_release_capable_model_before_inputs_are_frozen(tmp_path, monkeypatch):
    checkpoint, path, _, fixed = portable_rollout_inputs(tmp_path, learned_supports=True, has_corrector=False)
    model, _, _, _ = load_dynamics_model(checkpoint, "cpu")
    observation = load_initial(path)
    initial = replace(observation, metadata={**observation.metadata, "support_policy": "prescribed_material_corners"})
    observed = []

    def check(observation, positions, velocities, proposal, mask, next_fixed, *args, **kwargs):
        predicted_positions, predicted_velocities = dynamics_matched.runtime._proposal_arrays(proposal, positions, velocities)
        np.testing.assert_array_equal(next_fixed, fixed)
        np.testing.assert_array_equal(predicted_positions[fixed], positions[fixed])
        np.testing.assert_array_equal(predicted_velocities[fixed], 0)
        observed.append(proposal)
        return None, dict(status="feasibility_unknown", solves=[])

    monkeypatch.setattr(dynamics_matched.runtime, "_correct", check)
    report = compare_matched_step(initial, model, RolloutConfig(fracture_samples=0, fracture_policy="marginal_search"))
    assert len(observed) == 2 * report["candidate_count"]


def test_unknown_required_mechanics_is_not_a_verified_pass():
    config = SimpleNamespace(mechanics_mode="exact", fracture_initiation=True)
    solution = SimpleNamespace(diagnostics=dict(constraints_satisfied=True))
    checks, valid = _constraint_status(dict(valid=True), dict(valid=True), dict(energy_satisfied=True), solution,
                                       dict(mechanics=dict(elastic_dynamics=dict(status="pass"),
                                                           fracture_initiation=dict(status="unknown"))), config)
    assert checks["elastic_dynamics"] is True
    assert checks["fracture_initiation"] is None
    assert valid is None
    _, valid = _constraint_status(dict(valid=False), dict(valid=True), dict(energy_satisfied=True), solution, {}, config)
    assert valid is False


def test_proximal_residual_is_not_misreported_as_an_enforced_equality():
    config = SimpleNamespace(mechanics_mode="proximal", fracture_initiation=True)
    solution = SimpleNamespace(diagnostics=dict(constraints_satisfied=True))
    diagnostic = dict(mechanics=dict(elastic_dynamics=dict(status="fail"), fracture_initiation=dict(status="pass")))
    checks, valid = _constraint_status(dict(valid=True), dict(valid=True), dict(energy_satisfied=True), solution,
                                       diagnostic, config)
    assert "elastic_dynamics" not in checks
    assert checks["fracture_initiation"] is True
    assert valid is True
    assert diagnostic["mechanics"]["elastic_dynamics"]["status"] == "fail"


def test_cancelled_matched_step_reports_unattempted_methods(tmp_path, monkeypatch):
    _, initial, model, _ = matched_inputs(tmp_path)
    monkeypatch.setattr(dynamics_matched.runtime, "_correct", lambda *args, **kwargs:
                        (None, dict(status="interrupted", solves=[])))
    report = compare_matched_step(initial, model, RolloutConfig(fracture_samples=0, fracture_policy="marginal_search"))
    assert report["status"] == "interrupted"
    assert report["attempted_method_runs"] == 2
    assert report["unattempted_method_runs"] == report["requested_method_runs"] - 2
    assert not any(item["selection_complete"] for item in report["methods"].values())


def test_keyboard_interrupt_inside_solver_preserves_partial_candidate_report(tmp_path, monkeypatch):
    _, initial, model, _ = matched_inputs(tmp_path)

    def interrupt(*args, **kwargs):
        raise KeyboardInterrupt

    monkeypatch.setattr(dynamics_matched.runtime, "_correct", interrupt)
    report = compare_matched_step(initial, model, RolloutConfig(fracture_samples=0, fracture_policy="marginal_search"))
    assert report["status"] == "interrupted"
    assert report["attempted_method_runs"] == 2
    assert report["candidates"][-1]["status"] == "interrupted"


def test_sparse_teacher_intervals_remain_unknown_without_running_predictor(tmp_path, monkeypatch):
    from fracture_world.dynamics_data import load_dynamics_trajectory

    checkpoint, _, model, saved = matched_inputs(tmp_path)
    case, path = recorded_trajectory(tmp_path / "source")
    original = load_dynamics_trajectory(case, path)
    indices = [0, 2, 3]
    sparse = replace(original, **{key: getattr(original, key)[indices] for key in
                                 ("positions", "velocities", "broken", "fixed", "times")})
    saved["training_mesh_fingerprints"] = [sparse.topology.fingerprint]
    monkeypatch.setattr(dynamics_matched, "load_dynamics_predictor", lambda *args: (model, saved, torch.device("cpu")))
    monkeypatch.setattr(dynamics_matched, "load_dynamics_dataset", lambda *args, **kwargs: [sparse])
    monkeypatch.setattr(model, "forward", lambda *args: pytest.fail("sparse interval cannot be a one-step model input"))
    output = tmp_path / "matched"
    report = compare_matched_dynamics(checkpoint, tmp_path / "unused", output, split="all", frames=[0])
    assert report["cases"][0]["unknown_frames"] == 1
    assert report["cases"][0]["completed_frames"] == 0
    assert report["unknown_candidate_pool_observations"] == 1
    for method in ("raw", "central", "no_motion_central"):
        assert report["conditions"][method]["attempted_candidates"] == 0
        assert report["conditions"][method]["verified_candidates"] == 0
    recorded = json.loads((output / report["cases"][0]["report"]).read_text())
    assert recorded["frames"][0]["status"] == "unknown_sparse_transition"


def test_matched_data_runner_accepts_legacy_inference_checkpoint_without_training_state(tmp_path, monkeypatch):
    from fracture_world.dynamics_data import load_dynamics_trajectory

    checkpoint, _, _, saved = matched_inputs(tmp_path)
    case, path = recorded_trajectory(tmp_path / "source")
    item = load_dynamics_trajectory(case, path)
    saved["training_mesh_fingerprints"] = [item.topology.fingerprint]
    saved.pop("training_state", None)
    torch.save(saved, checkpoint)
    monkeypatch.setattr(dynamics_matched.runtime, "_correct", lambda *args, **kwargs:
                        (None, dict(status="feasibility_unknown", solves=[])))
    output = tmp_path / "matched"
    report = compare_matched_dynamics(checkpoint, case.parent.parent, output,
                                      RolloutConfig(fracture_samples=0, fracture_policy="marginal_search"),
                                      "cpu", split="all", frames=[0])
    assert report["status"] == "completed"
    assert report["cases"][0]["completed_frames"] == 1
    assert report["teacher_future_used_only_for_evaluation"]
    for method in ("raw", "central", "no_motion_central"):
        assert report["conditions"][method]["requested_candidates"] == 2
        assert report["conditions"][method]["attempted_candidates"] == 2
    for method in ("central", "no_motion_central"):
        summary = report["conditions"][method]
        assert summary["completed_candidates"] == summary["verified_candidates"] == 0
        assert summary["failed_candidates"] == summary["unknown_candidates"] == 2
    assert all(item["common_selected_observations"] == 0 for item in report["selected_state_comparisons"])
    assert (output / "matched.json").is_file()
