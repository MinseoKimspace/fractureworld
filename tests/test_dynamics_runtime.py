import json
import subprocess
import sys

import numpy as np
from numpy.testing import assert_allclose, assert_array_equal
import pytest
import torch

from fracture_world.dynamics_data import material_topology
from fracture_world.dynamics_geometry import face_corner_pairs
from fracture_world.dynamics_initial import SCHEMA, load_initial
from fracture_world.dynamics_learning import FORMAT
from fracture_world.dynamics_model import NeuralDynamicsCorrector, TemporalDynamicsModel
from fracture_world.dynamics_runtime import RolloutConfig, _problem, release_supports, rollout_dynamics
from fracture_world.mesh import VolumeMesh


def portable_rollout_inputs(tmp_path, *, broken=False, learned_supports=False, has_corrector=True):
    mesh = VolumeMesh(np.array([[0., 0., 0.], [1., 0., 0.], [0., 1., 0.], [0., 0., 1.], [1., 1., 1.]]),
                      np.array([[0, 1, 2, 3], [4, 1, 2, 3]]))
    topology = material_topology(mesh, 1)
    material = {"rest_density": 1., "youngs_modulus": 2., "poissons_ratio": 0.2, "fracture_stress_threshold": 3.}
    config = dict(hidden=12, heads=3, graph_layers=2, transformer_layers=1, contact_radius=0.05,
                  material_scale=(1, 2, 1, 3), learn_support_release=learned_supports)
    predictor = TemporalDynamicsModel(**config)
    with torch.no_grad():
        predictor.fracture_head.weight.zero_()
        predictor.fracture_head.bias.fill_(4)
        predictor.support_head.weight.zero_()
        predictor.support_head.bias.fill_(4)
    checkpoint = dict(format=FORMAT, model_config=predictor.config, model=predictor.state_dict(), training={"overfit": True},
                      condition_splits={"train": ["a" * 64], "validation": [], "test": []},
                      training_mesh_fingerprints=[topology.fingerprint], support_release_training_events=0)
    if has_corrector:
        checkpoint["corrector"] = NeuralDynamicsCorrector(**config).state_dict()
    checkpoint_path = tmp_path / "model.pt"
    torch.save(checkpoint, checkpoint_path)
    fixed = np.zeros(8, dtype=bool)
    if learned_supports:
        fixed[face_corner_pairs(mesh)[0, 0]] = True
    metadata = dict(schema=SCHEMA, units="SI", future_state_included=False,
                    support_policy="learned_material_corner_release" if learned_supports else "free",
                    support_semantics="recorded_numerical_teacher_kinematic_flags", observed_source_support_constant=True)
    initial_path = tmp_path / "initial.npz"
    np.savez_compressed(initial_path, schema=SCHEMA, nodes=mesh.nodes, tets=mesh.tets,
                        positions=topology.rest_positions, velocities=np.zeros((8, 3)), broken=np.array([broken]),
                        fixed=fixed, time=0., dt=0.01, gravity=[0., 0., -1.], material=list(material.values()),
                        case_id="portable", condition_digest="a" * 64, metadata_json=json.dumps(metadata))
    return checkpoint_path, initial_path, topology, fixed


@pytest.mark.parametrize("method", ["raw"])
def test_actual_engine_free_rollout_advances_all_methods_on_portable_current_state(tmp_path, method):
    checkpoint, initial, topology, _ = portable_rollout_inputs(tmp_path)
    directory = tmp_path / method
    config = RolloutConfig(fracture_samples=1, flow_steps=1, max_iterations=10000, max_contact_iterations=3,
                           absolute_tolerance=1.e-10, relative_tolerance=1.e-9, fracture_policy="marginal_search")
    report = rollout_dynamics(checkpoint, initial, directory, 2, method, config, "cpu")
    assert report["status"] == "completed"
    assert report["completed_steps"] == 2
    assert report["engine_calls"] == 0
    assert report["teacher_future_used"] is False
    assert report["runtime_native_required"] is False
    assert report["all_committed_geometry_valid"] is None
    assert report["checked_constraint_steps"] == 0
    assert report["optimization_solves"] == 0
    with np.load(directory / "trajectory.npz", allow_pickle=False) as archive:
        assert archive["positions"].shape == (3, 8, 3)
        assert_array_equal(archive["broken"][:, 0], [False, True, True])
        assert not archive["fixed"].any()
        assert np.all(np.diff(archive["broken"].astype(int), axis=0) >= 0)
        assert_allclose(archive["velocities"][-1], np.broadcast_to([0., 0., -0.02], (8, 3)), atol=2.e-6)
        assert_allclose(archive["positions"][-1], topology.rest_positions + [0., 0., -0.0003], atol=2.e-6)
        momentum = (topology.masses[:, None] * archive["velocities"][-1]).sum(axis=0)
        assert_allclose(momentum, topology.masses.sum() * np.array([0., 0., -0.02]), atol=2.e-8)
    replay = json.loads((directory / "replay.json").read_text())
    assert len(replay["frames"]) == 3


def test_repeated_rollouts_receive_the_same_seeded_first_step_fracture_candidate_pool(tmp_path):
    checkpoint, initial, _, _ = portable_rollout_inputs(tmp_path)
    config = RolloutConfig(fracture_samples=2, fracture_policy="fm_search", flow_steps=1,
                           max_iterations=10000, max_contact_iterations=3)
    pools = []
    for name in ("first", "second"):
        report = rollout_dynamics(checkpoint, initial, tmp_path / name, 1, "raw", config, "cpu")
        pools.append(report["events"][0]["candidate_pool"])
    assert all(pool == pools[0] for pool in pools[1:])


def test_old_fractures_are_irreversible_and_positive_release_evidence_is_reported_honestly(tmp_path):
    checkpoint, initial, _, fixed = portable_rollout_inputs(tmp_path, broken=True, learned_supports=True)
    report = rollout_dynamics(checkpoint, initial, tmp_path / "released", 1, "raw",
                              RolloutConfig(fracture_samples=0, fracture_policy="marginal_search"), "cpu")
    assert report["completed_steps"] == 1
    assert not report["learned_support_release_has_positive_training_examples"]
    with np.load(tmp_path / "released" / "trajectory.npz", allow_pickle=False) as archive:
        assert archive["broken"].all()
        assert_array_equal(archive["fixed"][0], fixed)
        assert not archive["fixed"][1].any()
        assert np.all(archive["velocities"][1, fixed, 2] < 0)


def test_intact_material_corner_group_retains_support_until_all_fixed_copies_request_release():
    mesh = VolumeMesh(np.array([[0., 0., 0.], [1., 0., 0.], [0., 1., 0.], [0., 0., 1.], [1., 1., 1.]]),
                      np.array([[0, 1, 2, 3], [4, 1, 2, 3]]))
    pair = face_corner_pairs(mesh)[0, 0]
    fixed = np.zeros(8, dtype=bool)
    fixed[pair] = True
    probability = np.ones(8)
    probability[pair[0]] = 0
    intact = release_supports(mesh, np.array([False]), fixed, probability)
    assert_array_equal(intact, fixed)
    separated = release_supports(mesh, np.array([True]), fixed, probability)
    assert separated[pair[0]] and not separated[pair[1]]
    assert not separated[~fixed].any()


@pytest.mark.parametrize("candidate_broken", [False, True])
def test_fm_candidate_supports_reapply_unclamped_motion_after_topology_selection(tmp_path, monkeypatch, candidate_broken):
    checkpoint, initial, topology, fixed = portable_rollout_inputs(tmp_path, learned_supports=True, has_corrector=False)
    pair = topology.face_pairs[0, 0]

    def predict(self, batch):
        corner_context = torch.zeros(8, 12)
        corner_context[pair[0], 0] = 4
        corner_context[pair[1], 0] = -4
        with torch.no_grad():
            self.support_head.weight.zero_()
            self.support_head.weight[0, 0] = 1
            self.support_head.bias.zero_()
            self.fracture_head.bias.fill_(-4 if candidate_broken else 4)
        return self._prediction(batch, corner_context, torch.zeros(1, 12)), self.prepare_fracture_flow(batch)

    monkeypatch.setattr(TemporalDynamicsModel, "forward_with_flow_context", predict)
    monkeypatch.setattr(TemporalDynamicsModel, "sample_fracture", lambda self, batch, steps, noise, *, context=None:
                        torch.full((1,), 1. if candidate_broken else -1.))
    output = tmp_path / "candidate"
    report = rollout_dynamics(checkpoint, initial, output, 1, "raw",
                              RolloutConfig(fracture_samples=1, fracture_policy="shared_sample", flow_steps=1), "cpu")
    assert report["status"] == "completed"
    with np.load(output / "trajectory.npz", allow_pickle=False) as archive:
        assert archive["broken"][1, 0] == candidate_broken
        if candidate_broken:
            assert not archive["fixed"][1, pair[0]]
            assert archive["fixed"][1, pair[1]]
            assert archive["velocities"][1, pair[0], 2] < 0
        else:
            assert_array_equal(archive["fixed"][1], fixed)
        assert_allclose(archive["velocities"][1, archive["fixed"][1]], 0)


@pytest.mark.parametrize("previous", [np.array(False), np.array([], dtype=bool), np.array([False, False]), np.array([0]), np.array([np.nan])])
def test_previous_fracture_topology_rejects_broadcast_or_nonboolean_inputs(tmp_path, previous):
    checkpoint, initial_path, _, _ = portable_rollout_inputs(tmp_path)
    initial = load_initial(initial_path)
    model = TemporalDynamicsModel(**torch.load(checkpoint, weights_only=True)["model_config"])
    with pytest.raises(ValueError, match="previous_broken must be a Boolean vector"):
        _problem(initial, initial.positions, initial.velocities, initial.positions, initial.velocities,
                 initial.broken, initial.fixed, (), model, RolloutConfig(), previous_broken=previous)


@pytest.mark.parametrize("previous", [np.array(False), np.array([], dtype=bool), np.zeros(8, dtype=int), np.zeros((2, 4), dtype=bool)])
def test_previous_support_state_rejects_broadcast_or_nonboolean_inputs(tmp_path, previous):
    checkpoint, initial_path, _, _ = portable_rollout_inputs(tmp_path)
    initial = load_initial(initial_path)
    model = TemporalDynamicsModel(**torch.load(checkpoint, weights_only=True)["model_config"])
    with pytest.raises(ValueError, match="previous_fixed must be a Boolean vector"):
        _problem(initial, initial.positions, initial.velocities, initial.positions, initial.velocities,
                 initial.broken, initial.fixed, (), model, RolloutConfig(), previous_fixed=previous)


def test_offline_comparison_passes_start_supports_when_a_candidate_releases_them(tmp_path, monkeypatch):
    from fracture_world import dynamics_runtime
    from fracture_world.dynamics_learning import load_dynamics_model
    from fracture_world.dynamics_matched import compare_matched_step

    checkpoint, initial_path, _, fixed = portable_rollout_inputs(tmp_path, learned_supports=True)
    original = dynamics_runtime._correct
    observed = []

    def recorded(*args, **kwargs):
        observed.append((args[5].copy(), kwargs["previous_fixed"].copy()))
        return original(*args, **kwargs)

    monkeypatch.setattr(dynamics_runtime, "_correct", recorded)
    model, _, _, _ = load_dynamics_model(checkpoint, "cpu")
    report = compare_matched_step(load_initial(initial_path), model,
                                  RolloutConfig(fracture_policy="marginal_search", fracture_samples=0))
    assert report["status"] == "completed"
    assert observed
    for next_fixed, previous_fixed in observed:
        assert not next_fixed.any()
        assert_array_equal(previous_fixed, fixed)


@pytest.mark.parametrize("method", ["neural", "central", "no_motion_central"])
def test_offline_correction_methods_are_rejected_before_rollout(tmp_path, method):
    checkpoint, initial, _, _ = portable_rollout_inputs(tmp_path, has_corrector=False)
    with pytest.raises(ValueError, match="offline correction"):
        rollout_dynamics(checkpoint, initial, tmp_path / method, 1, method, device="cpu")
    assert not (tmp_path / method).exists()


def test_runtime_imports_no_native_engine_wrappers():
    command = "import sys; from fracture_world.dynamics_runtime import rollout_dynamics; assert 'fracture_world.femfx' not in sys.modules; assert 'fracture_world.bullet' not in sys.modules; assert '_femfx' not in sys.modules"
    result = subprocess.run([sys.executable, "-c", command], capture_output=True, text=True)
    assert result.returncode == 0, result.stderr


def test_shared_flow_sample_is_preserved_across_repeated_inference_runs(tmp_path):
    checkpoint, initial, _, _ = portable_rollout_inputs(tmp_path)
    config = RolloutConfig(fracture_samples=1, fracture_policy="shared_sample", flow_steps=2,
                           max_iterations=20000, max_contact_iterations=3)
    records = []
    for name in ("first", "second"):
        result = rollout_dynamics(checkpoint, initial, tmp_path / name, 1, "raw", config, "cpu")
        assert result["completed_steps"] == 1
        assert result["fracture_flow_calls"] == 1
        event = result["events"][0]
        assert len(event["candidates"]) == 1
        assert event["candidates"][0]["prior"] == 0
        records.append(event["candidate_pool"])
    assert all(record == records[0] for record in records)


def test_cancelled_rollout_saves_initial_frame_without_committing_a_step(tmp_path):
    checkpoint, initial, _, _ = portable_rollout_inputs(tmp_path)
    report = rollout_dynamics(checkpoint, initial, tmp_path / "cancelled", 3, "raw", device="cpu", cancel=lambda: True)
    assert report["status"] == "interrupted"
    assert report["completed_steps"] == 0
    assert report["predictor_calls"] == 0
    assert report["checked_constraint_steps"] == 0
    assert report["all_committed_geometry_valid"] is None
    assert report["all_committed_energy_valid"] is None
    assert report["all_committed_swept_geometry_valid"] is None
    assert (tmp_path / "cancelled" / "trajectory.npz").is_file()


def test_replay_records_each_frame_supports_components_and_both_crack_sides(tmp_path):
    checkpoint, initial, _, fixed = portable_rollout_inputs(tmp_path, broken=True, learned_supports=True)
    rollout_dynamics(checkpoint, initial, tmp_path / "rendered", 1, "raw",
                     RolloutConfig(fracture_samples=0, fracture_policy="marginal_search"), "cpu")
    replay = json.loads((tmp_path / "rendered" / "replay.json").read_text())
    before = replay["frames"][0]["geometries"][0]
    after = replay["frames"][1]["geometries"][0]
    assert before["fixed_vertices"] == np.flatnonzero(fixed).tolist()
    assert after["fixed_vertices"] == []
    for geometry in (before, after):
        assert sum(geometry["crack_faces"]) == 2
        assert len(set(geometry["vertex_components"])) == 2



@pytest.mark.parametrize("samples", [0, 2])
def test_shared_flow_policy_rejects_ambiguous_candidate_count(samples):
    with pytest.raises(ValueError, match="exactly one"):
        RolloutConfig(fracture_samples=samples, fracture_policy="shared_sample")


def test_default_fracture_policy_uses_one_shared_flow_sample():
    assert RolloutConfig().fracture_policy == "shared_sample"
    assert RolloutConfig().fracture_samples == 1
    with pytest.raises(ValueError, match="at least one"):
        RolloutConfig(fracture_policy="fm_search", fracture_samples=0)
    assert RolloutConfig(fracture_policy="marginal_search", fracture_samples=0).fracture_policy == "marginal_search"


@pytest.mark.parametrize("settings", [
    {"fracture_samples": 1.5}, {"fracture_samples": True}, {"flow_steps": 1.5}, {"flow_steps": True},
    {"max_iterations": True}, {"max_contact_iterations": False},
])
def test_rollout_configuration_rejects_noninteger_counts_before_execution(settings):
    with pytest.raises(ValueError, match="integer|iteration budgets"):
        RolloutConfig(**settings)


def test_boolean_rollout_length_is_rejected_before_loading_files(tmp_path):
    with pytest.raises(ValueError, match="positive number of steps"):
        rollout_dynamics(tmp_path / "missing.pt", tmp_path / "missing.npz", tmp_path / "output", True)
    assert not (tmp_path / "output").exists()


def test_physics_only_replay_identifies_the_unlearned_baseline(tmp_path):
    checkpoint, initial, _, _ = portable_rollout_inputs(tmp_path)
    directory = tmp_path / "physics-replay"
    report = rollout_dynamics(checkpoint, initial, directory, 1, "physics", device="cpu")
    replay = json.loads((directory / "replay.json").read_text())
    assert report["predictor_calls"] == 0
    assert report["constitutive_response"] == "constant acceleration"
    assert report["fracture_choice"] == "initial topology"
    assert report["optimization_solves"] == 0
    assert report["all_committed_geometry_valid"] is None
    assert replay["title"] == "Constant-acceleration physics baseline"
    assert replay["source"] == "constant-acceleration physical prediction; no learned dynamics or fracture generation"
    assert replay["metadata"]["method"] == "physics"


@pytest.mark.parametrize("field", ["positions", "velocities", "position_increment", "velocity_increment",
                                   "fracture_probability", "support_release_probability",
                                   "unconstrained_position_increment", "unconstrained_velocity_increment"])
def test_nonfinite_predictions_save_failure_report_before_geometry_or_optimization(tmp_path, monkeypatch, field):
    checkpoint, initial, _, _ = portable_rollout_inputs(tmp_path)
    original = TemporalDynamicsModel.forward_with_flow_context

    def corrupted(self, *args, **kwargs):
        prediction = original(self, *args, **kwargs)
        output = prediction[0]
        source = output.get(field, output["positions"])
        output[field] = torch.full_like(source, float("nan"))
        return output, prediction[1]

    monkeypatch.setattr(TemporalDynamicsModel, "forward_with_flow_context", corrupted)
    directory = tmp_path / "failed"
    report = rollout_dynamics(checkpoint, initial, directory, 3, "raw", device="cpu")
    assert report["status"] == "nonfinite_prediction"
    assert report["completed_steps"] == 0
    assert report["optimization_solves"] == 0
    assert report["events"][-1]["status"] == "nonfinite_prediction"
    assert field in report["events"][-1]["reason"]
    assert report["all_committed_geometry_valid"] is None
    assert report["all_committed_energy_valid"] is None
    assert report["all_committed_swept_geometry_valid"] is None
    assert json.loads((directory / "report.json").read_text())["status"] == "nonfinite_prediction"
    with np.load(directory / "trajectory.npz", allow_pickle=False) as archive:
        assert len(archive["times"]) == 1
        assert np.isfinite(archive["positions"]).all()
        assert np.isfinite(archive["velocities"]).all()


@pytest.mark.parametrize("field", ["unconstrained_position_increment", "unconstrained_velocity_increment"])
def test_nonfinite_hidden_support_motion_is_rejected_before_fm_can_restore_it(tmp_path, monkeypatch, field):
    checkpoint, initial, _, fixed = portable_rollout_inputs(tmp_path, learned_supports=True, has_corrector=False)
    original = TemporalDynamicsModel.forward_with_flow_context

    def corrupted(self, batch):
        with torch.no_grad():
            self.support_head.bias.fill_(-4)
        output, context = original(self, batch)
        assert output["next_fixed"][batch["fixed"]].all()
        assert torch.isfinite(output["positions"]).all()
        assert torch.isfinite(output["velocities"]).all()
        output[field] = output[field].clone()
        output[field][torch.as_tensor(fixed), 0] = float("nan")
        return output, context

    monkeypatch.setattr(TemporalDynamicsModel, "forward_with_flow_context", corrupted)
    report = rollout_dynamics(checkpoint, initial, tmp_path / "hidden-nonfinite", 1, "raw", device="cpu")
    assert report["status"] == "nonfinite_prediction"
    assert report["completed_steps"] == 0
    assert report["fracture_flow_calls"] == 0
    assert field in report["events"][-1]["reason"]


def test_nonfinite_flow_sample_is_not_silently_thresholded_into_an_unbroken_mask(tmp_path, monkeypatch):
    checkpoint, initial, _, _ = portable_rollout_inputs(tmp_path)
    monkeypatch.setattr(TemporalDynamicsModel, "sample_fracture", lambda self, batch, steps, noise, *, context=None: torch.full_like(noise, float("nan")))
    report = rollout_dynamics(checkpoint, initial, tmp_path / "failed-flow", 1, "raw", device="cpu")
    assert report["status"] == "nonfinite_prediction"
    assert report["completed_steps"] == 0
    assert "sampled fracture field" in report["events"][-1]["reason"]


def test_nonfinite_prediction_saves_the_completed_rollout_prefix(tmp_path, monkeypatch):
    checkpoint, initial, _, _ = portable_rollout_inputs(tmp_path)
    original = TemporalDynamicsModel.forward_with_flow_context
    calls = 0

    def predict(self, batch):
        nonlocal calls
        calls += 1
        proposal, context = original(self, batch)
        if calls == 2:
            proposal["positions"] = torch.full_like(proposal["positions"], float("nan"))
        return proposal, context

    monkeypatch.setattr(TemporalDynamicsModel, "forward_with_flow_context", predict)
    config = RolloutConfig(fracture_samples=1, fracture_policy="shared_sample", flow_steps=1,
                           max_iterations=10000, max_contact_iterations=3)
    directory = tmp_path / "raw"
    report = rollout_dynamics(checkpoint, initial, directory, 3, "raw", config, "cpu")
    assert report["status"] == "nonfinite_prediction"
    assert report["completed_steps"] == 1
    assert "positions" in report["events"][-1]["reason"]
    assert json.loads((directory / "report.json").read_text())["completed_steps"] == 1
    with np.load(directory / "trajectory.npz", allow_pickle=False) as archive:
        assert len(archive["times"]) == 2
        assert np.isfinite(archive["positions"]).all()
        assert np.isfinite(archive["velocities"]).all()
