from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from fracture_world import dynamics_experiment
from fracture_world.dynamics_experiment import _rollout_agreement, _runtime_support_policy, benchmark_dynamics


class MetadataWithoutSupportFuture(dict):
    def __getitem__(self, key):
        if key == "observed_support_releases":
            raise AssertionError("teacher support-release future cannot select a runtime policy")
        return super().__getitem__(key)


class BenchmarkCase(SimpleNamespace):
    def __len__(self):
        return len(self.broken) - 1


def test_equal_initial_states_do_not_imply_rollout_agreement(tmp_path):
    path = tmp_path / "initial-only.npz"
    np.savez(path, times=[0.0], positions=np.zeros((1, 4, 3)), velocities=np.zeros((1, 4, 3)),
             broken=np.zeros((1, 2), dtype=bool))
    report = _rollout_agreement(path, path, False)
    assert report["common_prediction_frames"] == 0
    assert report["agreement_available"] is False
    assert report["positions_rms_metres"] is None
    assert report["velocities_rms_metres_per_second"] is None
    assert report["fracture_disagreements"] is None
    assert report["both_rollouts_completed"] is False


def test_rollout_agreement_scores_only_future_predictions(tmp_path):
    central, consensus = tmp_path / "central.npz", tmp_path / "consensus.npz"
    np.savez(central, times=[0.0, 0.1], positions=np.zeros((2, 4, 3)), velocities=np.zeros((2, 4, 3)),
             broken=np.zeros((2, 2), dtype=bool))
    np.savez(consensus, times=[0.0, 0.1], positions=np.full((2, 4, 3), 2.0), velocities=np.ones((2, 4, 3)),
             broken=np.array([[True, True], [False, True]]))
    report = _rollout_agreement(central, consensus, True)
    assert report["agreement_available"] is True
    assert report["common_prediction_frames"] == 1
    assert report["positions_rms_metres"] == 2.0
    assert report["velocities_rms_metres_per_second"] == 1.0
    assert report["fracture_disagreements"] == 1
    assert report["both_rollouts_completed"] is True


@pytest.mark.parametrize("checkpoint,expected", [
    ({"model_config": {"learn_support_release": True}}, "learned"),
    ({"training": {"learn_support_release": True}}, "learned"),
    ({"model_config": {"learn_support_release": False}, "training": {"learn_support_release": True}}, "prescribed"),
    ({}, "prescribed"),
])
def test_support_policy_uses_checkpoint_capability_and_current_flags(checkpoint, expected):
    assert _runtime_support_policy(checkpoint, np.array([True, False])) == expected
    assert _runtime_support_policy(checkpoint, np.array([False, False])) == "free"


@pytest.mark.parametrize("learned", [False, True])
def test_benchmark_does_not_select_support_policy_from_teacher_events(tmp_path, monkeypatch, learned):
    checkpoint = tmp_path / "checkpoint.pt"
    checkpoint.write_bytes(b"test-checkpoint")
    cases = []
    for index, name in enumerate(("fixed-negative", "fixed-positive", "free-negative")):
        fixed = np.full((3, 2), index != 2, dtype=bool)
        if index == 1:
            fixed[-1] = False
        broken = np.tile([True, False], (3, 1))
        if index == 1:
            broken[-1, 1] = True
        source = tmp_path / "source" / "cases" / f"{name}.npz"
        cases.append(BenchmarkCase(case_id=name, condition_digest=f"condition-{index}", shape_family="fixture",
                                   fixed=fixed, broken=broken,
                                   metadata=MetadataWithoutSupportFuture(source_hashes={str(source): "digest"})))
    saved = dict(training=dict(overfit=False, learn_support_release=learned),
                 model_config=dict(learn_support_release=learned),
                 condition_splits={"test": [item.condition_digest for item in cases]},
                 splits={"test": [item.case_id for item in cases]})
    monkeypatch.setattr(dynamics_experiment, "load_dynamics_model", lambda *args: (None, None, saved, "cpu"))
    monkeypatch.setattr(dynamics_experiment, "load_dynamics_dataset", lambda *args, **kwargs: cases)
    exported = {}

    def export(case, trajectory, frame, initial, *, support_policy):
        assert frame == 0
        assert trajectory == case.parent.parent / "trajectories" / case.name
        exported[Path(initial)] = (case.stem, support_policy)

    def compare(checkpoint, initial, output, steps, config, device, methods, reference_case, reference_trajectory, progress):
        name, policy = exported[Path(initial)]
        evaluation = dict(compared_future_frames=steps, fracture=dict(f1=0.999, predicted_broken_facets=100),
                          new_fracture=dict(f1=0.0 if name == "fixed-positive" else 1.0,
                                            predicted_broken_facets=int(name == "fixed-negative")))
        metrics = ("position_mass_weighted_rmse_m", "velocity_mass_weighted_rmse_m_per_s",
                   "linear_momentum_error_kg_m_per_s", "angular_momentum_about_origin_error_kg_m2_per_s",
                   "kinetic_energy_absolute_error_j", "maximum_floor_penetration_m", "component_count_absolute_error")
        evaluation.update({key: 0.0 for key in metrics})
        return dict(all_rollouts_completed=True,
                    conditions={"raw": dict(status="completed", evaluation=evaluation,
                                             all_committed_geometry_valid=True, runtime_seconds=1.0)})

    monkeypatch.setattr(dynamics_experiment, "export_initial", export)
    monkeypatch.setattr(dynamics_experiment, "compare_dynamics", compare)
    report = benchmark_dynamics(checkpoint, tmp_path / "unused-data", tmp_path / "benchmark", 2, methods=["raw"])
    expected = "learned" if learned else "prescribed"
    assert {name: policy for name, policy in exported.values()} == {
        "fixed-negative": expected, "fixed-positive": expected, "free-negative": "free"}
    summary = report["conditions"]["raw"]
    assert summary["fracture_positive_cases"] == 1
    assert summary["fracture_positive_case_f1_failure_penalized"] == 0
    assert summary["completed_fracture_negative_cases"] == 2
    assert summary["fracture_negative_case_false_positive_rate"] == 0.5
    assert report["reference_used_only_for_evaluation"]
    assert "initially broken facets excluded" in report["fracture_score_semantics"]
