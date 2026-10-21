import json
import subprocess
import sys

import numpy as np
from numpy.testing import assert_allclose, assert_array_equal
import pytest
import torch

from fracture_world.dynamics_data import material_topology
from fracture_world.dynamics_geometry import face_corner_pairs
from fracture_world.dynamics_initial import SCHEMA
from fracture_world.dynamics_learning import FORMAT
from fracture_world.dynamics_model import NeuralDynamicsCorrector, TemporalDynamicsModel
from fracture_world.dynamics_runtime import RolloutConfig, release_supports, rollout_dynamics
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


@pytest.mark.parametrize("method", ["raw", "neural", "central", "consensus"])
def test_actual_engine_free_rollout_advances_all_methods_on_portable_current_state(tmp_path, method):
    checkpoint, initial, topology, _ = portable_rollout_inputs(tmp_path)
    directory = tmp_path / method
    config = RolloutConfig(fracture_samples=1, flow_steps=1, max_iterations=10000, max_contact_iterations=3,
                           absolute_tolerance=1.e-10, relative_tolerance=1.e-9)
    report = rollout_dynamics(checkpoint, initial, directory, 2, method, config, "cpu")
    assert report["status"] == "completed"
    assert report["completed_steps"] == 2
    assert report["engine_calls"] == 0
    assert report["teacher_future_used"] is False
    assert report["runtime_native_required"] is False
    assert report["all_committed_geometry_valid"]
    assert report["optimization_solves"] > 0 if method in ("central", "consensus") else report["optimization_solves"] == 0
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


def test_all_methods_receive_the_same_first_step_fracture_candidate_pool(tmp_path):
    checkpoint, initial, _, _ = portable_rollout_inputs(tmp_path)
    config = RolloutConfig(fracture_samples=2, flow_steps=1, max_iterations=10000, max_contact_iterations=3)
    pools = []
    for method in ("raw", "neural", "central", "consensus"):
        report = rollout_dynamics(checkpoint, initial, tmp_path / method, 1, method, config, "cpu")
        pools.append(report["events"][0]["candidate_pool"])
    assert all(pool == pools[0] for pool in pools[1:])


def test_old_fractures_are_irreversible_and_positive_release_evidence_is_reported_honestly(tmp_path):
    checkpoint, initial, _, fixed = portable_rollout_inputs(tmp_path, broken=True, learned_supports=True)
    report = rollout_dynamics(checkpoint, initial, tmp_path / "released", 1, "raw", RolloutConfig(fracture_samples=0), "cpu")
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


def test_missing_neural_corrector_is_rejected_before_rollout(tmp_path):
    checkpoint, initial, _, _ = portable_rollout_inputs(tmp_path, has_corrector=False)
    with pytest.raises(ValueError, match="no trained neural corrector"):
        rollout_dynamics(checkpoint, initial, tmp_path / "neural", 1, "neural", device="cpu")


def test_runtime_imports_no_native_engine_wrappers():
    command = "import sys; from fracture_world.dynamics_runtime import rollout_dynamics; assert 'fracture_world.femfx' not in sys.modules; assert 'fracture_world.bullet' not in sys.modules; assert '_femfx' not in sys.modules"
    result = subprocess.run([sys.executable, "-c", command], capture_output=True, text=True)
    assert result.returncode == 0, result.stderr


def test_shared_flow_sample_is_preserved_across_correction_methods(tmp_path):
    checkpoint, initial, _, _ = portable_rollout_inputs(tmp_path)
    config = RolloutConfig(fracture_samples=1, fracture_policy="shared_sample", flow_steps=2,
                           max_iterations=20000, max_contact_iterations=3)
    records = []
    for method in ("raw", "neural", "central", "consensus"):
        result = rollout_dynamics(checkpoint, initial, tmp_path / method, 1, method, config, "cpu")
        assert result["completed_steps"] == 1
        assert result["fracture_flow_calls"] == 1
        event = result["events"][0]
        assert len(event["candidates"]) == 1
        assert event["candidates"][0]["prior"] == 0
        records.append(event["candidate_pool"])
    assert all(record == records[0] for record in records)


def test_cancelled_rollout_saves_initial_frame_without_committing_a_step(tmp_path):
    checkpoint, initial, _, _ = portable_rollout_inputs(tmp_path)
    report = rollout_dynamics(checkpoint, initial, tmp_path / "cancelled", 3, "consensus", device="cpu", cancel=lambda: True)
    assert report["status"] == "interrupted"
    assert report["completed_steps"] == 0
    assert report["predictor_calls"] == 0
    assert (tmp_path / "cancelled" / "trajectory.npz").is_file()


def test_replay_records_each_frame_supports_components_and_both_crack_sides(tmp_path):
    checkpoint, initial, _, fixed = portable_rollout_inputs(tmp_path, broken=True, learned_supports=True)
    rollout_dynamics(checkpoint, initial, tmp_path / "rendered", 1, "raw", RolloutConfig(fracture_samples=0), "cpu")
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
