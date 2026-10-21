from dataclasses import replace
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
from numpy.testing import assert_allclose
import pytest
from scipy.spatial.transform import Rotation
import torch

from fracture_world import drop_generator
from fracture_world.drop_generator import ballistic_drop_state, create_drop_suite, prepare_volume
from fracture_world.dynamics_corrector import DynamicsProblem, solve_central, solve_consensus
from fracture_world.dynamics_data import load_dynamics_dataset, material_topology, split_dynamics_cases
from fracture_world.data import _mesh_digest
from fracture_world.dynamics_initial import export_initial
from fracture_world.dynamics_experiment import benchmark_dynamics
from fracture_world.dynamics_learning import DynamicsTrainingConfig, load_dynamics_model, train_dynamics
from fracture_world.femfx import context_digest, create_trial, generate_dataset, validate_piece_partition
from fracture_world.io import file_hash
from fracture_world.dynamics_geometry import floor_contacts, geometry_diagnostics
from fracture_world.dynamics_learning import state_batch
from fracture_world.dynamics_model import TemporalDynamicsModel, build_dynamics_graph, constant_acceleration_prediction
from fracture_world.dynamics_readiness import inspect_dynamics_dataset
from fracture_world.dynamics_runtime import RolloutConfig, rollout_dynamics
from fracture_world.femfx import initial_conditions, validate_specification
from fracture_world.mesh import VolumeMesh, rigid_pose, tetrahedron_aspect_ratios
from test_dynamics_data import recorded_trajectory
from test_dynamics_readiness import trajectory
from test_dynamics_runtime import portable_rollout_inputs


def mesh():
    return VolumeMesh([[0, 0, 0], [0.1, 0, 0], [0, 0.1, 0], [0, 0, 0.1]], [[0, 1, 2, 3]])


def test_ballistic_window_preserves_drop_energy_and_rigid_shape():
    volume = mesh()
    rotation = Rotation.from_euler("xyz", [30, 70, 10], degrees=True)
    pose, velocity, approach = ballistic_drop_state(volume, 0.3, rotation, 9.81, 0.002, 0.0)
    positions = rigid_pose(volume.nodes, pose)
    assert_allclose(positions[:, 1].min(), approach["remaining_clearance_m"], atol=1e-15)
    assert_allclose(0.5 * velocity[1]**2 + 9.81 * positions[:, 1].min(), 9.81 * 0.3, atol=1e-14)
    assert_allclose(positions[:, None] - positions[None], rotation.apply(
        (volume.nodes[:, None] - volume.nodes[None]).reshape(-1, 3)).reshape(4, 4, 3), atol=1e-15)
    assert approach["nominal_time_until_first_floor_contact_s"] == pytest.approx(0.002)


def test_drop_suite_builds_thousand_distinct_free_cases_without_native_execution(tmp_path, monkeypatch):
    def prepared(shape, output, settings, surface_path):
        volume = mesh()
        volume.save(output / "mesh.npz")
        return volume, {"source": {"shape_family": "stanford_bunny"}}

    monkeypatch.setattr(drop_generator, "prepare_volume", prepared)
    config = Path(__file__).resolve().parents[1] / "configs" / "bunny-drop.json"
    result = create_drop_suite(tmp_path / "input", config)
    specification = json.loads(Path(result["specification"]).read_text())
    volume = VolumeMesh.load(result["mesh"])
    material, fixed = validate_specification(volume, specification)
    assert len(specification["cases"]) == result["case_count"] == 1000
    assert not len(fixed)
    assert not result["simulation_executed"]
    assert specification["scene_options"]["rayleigh_mass_damping"] == 0
    assert specification["scene_options"]["rayleigh_stiffness_damping"] == 0
    assert len({json.dumps(case["initial_pose"], sort_keys=True) for case in specification["cases"]}) == 1000
    for case in specification["cases"]:
        velocity, flags, impacts = initial_conditions(volume, material, fixed, case)
        assert not flags.any()
        assert impacts.shape == (0, 7)
        assert_allclose(velocity, np.broadcast_to(case["initial_linear_velocity"], velocity.shape))
        assert rigid_pose(volume.nodes, case["initial_pose"])[:, 1].min() > specification["floor_height_m"]


def test_external_volume_mesher_produces_positive_solid_primitive(tmp_path):
    pytest.importorskip("tetgen")
    pytest.importorskip("pymeshfix")
    settings = dict(geometry_mode="solid", height_m=0.12, surface_triangles=80,
                    radius_edge_ratio=1.5, minimum_dihedral_degrees=10)
    volume, report = prepare_volume("box", tmp_path, settings)
    assert (volume.volumes > 0).all()
    assert_allclose(volume.volumes.sum(), 0.12**3, rtol=1e-6)
    assert report["source"]["shape_family"] == "primitive_box"
    quality = report["native_mesh_quality"]
    assert quality["passed"]
    assert quality["attempts"][-1]["maximum_aspect_ratio"] < quality["femfx_limit"]
    assert Path(tmp_path / "surface.ply").exists()


def test_native_tetrahedron_aspect_metric_is_scale_invariant():
    regular = np.array([[0, 0, 0], [1, 0, 0], [0.5, np.sqrt(3) / 2, 0],
                        [0.5, np.sqrt(3) / 6, np.sqrt(2 / 3)]])
    for scale in (1e-100, 1.0, 1e100):
        assert_allclose(tetrahedron_aspect_ratios(regular * scale, [[0, 1, 2, 3]]), np.sqrt(1.5))
    flattened = regular.copy()
    flattened[3, 2] *= 0.001
    assert tetrahedron_aspect_ratios(flattened, [[0, 1, 2, 3]])[0] > 25


def test_tetrahedralizer_refines_failed_native_quality_and_reports_stagnation(tmp_path, monkeypatch):
    tetgen = pytest.importorskip("tetgen")
    settings = dict(radius_edge_ratio=1.5, minimum_dihedral_degrees=10, target_tetrahedra=10)
    bad = np.array([[0, 0, 0], [1, 0, 0], [0, 1, 0], [0, 0, 0.01]])
    good = bad.copy()
    good[3, 2] = 0.5
    calls = []
    results = iter((bad, good))

    def tetrahedralize(**options):
        calls.append(options)
        return next(results), np.array([[0, 1, 2, 3]])

    monkeypatch.setattr(tetgen, "TetGen", lambda surface: SimpleNamespace(tetrahedralize=tetrahedralize))
    volume, quality = drop_generator._tetrahedralize(SimpleNamespace(volume=1.0), settings, tmp_path)
    assert quality["passed"] and len(quality["attempts"]) == 2
    assert calls[1]["maxvolume"] < calls[0]["maxvolume"]
    assert volume.tetrahedron_aspect_ratios.max() < 25
    assert not (tmp_path / "mesh-quality-failure.json").exists()
    results = iter((bad, bad))
    with pytest.raises(ValueError, match="quality refinement stagnated"):
        drop_generator._tetrahedralize(SimpleNamespace(volume=1.0), settings, tmp_path)
    failed = json.loads((tmp_path / "mesh-quality-failure.json").read_text())
    assert not failed["passed"] and len(failed["attempts"]) == 2


def test_physical_prior_matches_acceleration_and_is_exposed_to_learned_corrector():
    volume = mesh()
    topology = material_topology(volume, 1.0)
    gravity = np.array([0.0, -9.81, 0.0])
    velocities = np.broadcast_to([0.5, -2.0, -0.2], topology.rest_positions.shape)
    material = dict(rest_density=1.0, youngs_modulus=2.0, poissons_ratio=0.2, fracture_stress_threshold=3.0)
    batch = state_batch(topology, topology.rest_positions + [0, 1, 0], velocities,
                        np.zeros(0, dtype=bool), np.zeros(4, dtype=bool), gravity, material, 0.01, "cpu", 0.0)
    physical = constant_acceleration_prediction(batch)
    assert_allclose(physical["positions"], batch["positions"] + 0.01 * batch["velocities"] + 0.5 * 0.01**2 * gravity, atol=1e-7)
    assert_allclose(physical["velocities"], velocities + 0.01 * gravity, atol=2e-7)
    model = TemporalDynamicsModel(hidden=12, heads=3, graph_layers=2, motion_prior="constant_acceleration")
    model.eval()
    with torch.no_grad():
        predicted = model(batch)
    torch.testing.assert_close(predicted["positions"], physical["positions"])
    torch.testing.assert_close(predicted["velocities"], physical["velocities"])
    graph = build_dynamics_graph(batch, physical_proposal=physical)
    assert len(graph["interaction"].floor_corner_ids) == 4
    assert graph["interaction"].x.shape[1] == 14
    assert torch.all(graph["interaction"].x[-4:, -1] == 2)


def test_constant_acceleration_correctors_preserve_exact_free_motion():
    volume = mesh()
    current = volume.nodes[volume.tets] + [0, 1, 0]
    dt, gravity = 0.01, np.array([0.0, -9.81, 0.0])
    velocity = np.zeros_like(current)
    masses = np.broadcast_to(volume.volumes[:, None] / 4, (1, 4))
    problem = DynamicsProblem(volume, current, velocity, current + 0.5 * dt**2 * gravity,
                              velocity + dt * gravity, dt, external_impulse=masses[..., None] * dt * gravity,
                              kinematic_rule="constant_acceleration")
    for result in (solve_central(problem), solve_consensus(problem, max_iterations=1000)):
        assert result.converged
        assert_allclose(result.positions, problem.predicted_positions, atol=1e-8)
        assert_allclose(result.velocities, problem.predicted_velocities, atol=1e-8)


def test_floor_contacts_detect_approach_and_geometric_penetration():
    volume = mesh()
    current = volume.nodes[volume.tets] + [0, 0.01, 0]
    proposed = current - [0, 0.02, 0]
    contacts = floor_contacts(current, proposed, 0.0, 1e-8)
    assert contacts and all(contact.environment for contact in contacts)
    report = geometry_diagnostics(volume, proposed, np.zeros(0, dtype=bool), 1e-8, 0.0)
    assert not report["valid"]
    assert report["maximum_floor_penetration_metres"] == pytest.approx(0.01)


def test_frictionless_floor_cannot_create_tangential_momentum_or_suction():
    volume = mesh()
    current = volume.nodes[volume.tets]
    dt, gravity = 0.01, np.array([0.0, -9.81, 0.0])
    masses = np.broadcast_to(volume.volumes[:, None] / 4, (1, 4))
    velocity = np.broadcast_to([0.0, -1.0, 0.0], current.shape)
    predicted_velocity = np.broadcast_to([2.0, 0.0, 3.0], current.shape)
    proposed = current + dt * predicted_velocity
    contacts = floor_contacts(current, proposed, 0.0, 1e-8)
    problem = DynamicsProblem(volume, current, velocity, proposed, predicted_velocity, dt,
                              external_impulse=masses[..., None] * dt * gravity, contacts=contacts,
                              kinematic_rule="constant_acceleration")
    result = solve_central(problem)
    assert result.converged
    momentum = np.sum(masses[..., None] * result.velocities, axis=(0, 1))
    assert_allclose(momentum[[0, 2]], 0, atol=1e-8)
    assert momentum[1] >= np.sum(masses[..., None] * (velocity + dt * gravity), axis=(0, 1))[1] - 1e-8
    assert result.positions[:, :, 1].min() >= -1e-8


def test_vertex_only_coupling_is_distinguished_from_a_native_mapping_error():
    volume = VolumeMesh([[0, 0, 0], [1, 0, 0], [0, 1, 0], [0, 0, 1],
                         [-1, 0, 0], [0, -1, 0], [0, 0, -1]], [[0, 1, 2, 3], [0, 4, 5, 6]])
    state = dict(tet_piece_ids=np.array([0, 0]), corner_vertex_ids=volume.tets)
    with pytest.raises(RuntimeError, match="unsupported vertex-only coupling"):
        validate_piece_partition(volume, state, np.zeros(0, dtype=bool))
    state["tet_piece_ids"] = np.array([0, 1])
    with pytest.raises(RuntimeError, match="current vertex connectivity"):
        validate_piece_partition(volume, state, np.zeros(0, dtype=bool))
    state["tet_piece_ids"] = np.array([0, 0])
    state["corner_vertex_ids"] = volume.tets.copy()
    state["corner_vertex_ids"][1, np.flatnonzero(volume.tets[1] == 0)[0]] = 7
    labels = validate_piece_partition(volume, state, np.zeros(0, dtype=bool))
    assert len(np.unique(labels)) == 2


def test_native_setup_failures_are_recorded_by_skip_invalid(tmp_path, monkeypatch):
    def failed_scene(*args, **kwargs):
        raise RuntimeError("native mesh quality rejected")

    native = SimpleNamespace(Scene=failed_scene, capabilities=lambda: {})
    monkeypatch.setattr("fracture_world.femfx.load_native", lambda _: (native, {"binding_sha256": "0" * 64}))
    trial = create_trial(tmp_path / "input")
    report = generate_dataset(trial["mesh"], trial["specification"], tmp_path / "data", 1.0, skip_invalid=True)
    assert len(report["rejected_cases"]) == 4
    assert not report["cases"]
    assert all("mesh quality" in item["reason"] for item in report["rejected_cases"])


def test_physics_only_rollout_does_not_call_network_or_generate_cracks(tmp_path):
    checkpoint, initial, topology, _ = portable_rollout_inputs(tmp_path)
    report = rollout_dynamics(checkpoint, initial, tmp_path / "physical", 3, "physics", RolloutConfig(), "cpu")
    assert report["completed_steps"] == 3
    assert report["predictor_calls"] == report["fracture_flow_calls"] == 0
    with np.load(tmp_path / "physical" / "trajectory.npz") as archive:
        assert_allclose(archive["positions"][-1], topology.rest_positions + [0, 0, -0.5 * 0.03**2], atol=1e-7)
        assert not archive["broken"].any()


def test_only_explicit_held_out_shape_protocol_admits_untrained_mesh(tmp_path):
    path, initial, topology, _ = portable_rollout_inputs(tmp_path)
    checkpoint = torch.load(path, weights_only=True)
    checkpoint["training_mesh_fingerprints"] = ["other"]
    checkpoint["evaluation_mesh_fingerprints"] = [topology.fingerprint]
    torch.save(checkpoint, path)
    with pytest.raises(ValueError, match="not trained"):
        rollout_dynamics(path, initial, tmp_path / "blocked", 1, "physics", device="cpu")
    checkpoint["generalization_protocol"] = "shape_family"
    torch.save(checkpoint, path)
    report = rollout_dynamics(path, initial, tmp_path / "heldout", 1, "physics", device="cpu")
    assert report["held_out_shape_family"]
    assert not report["mesh_seen_during_training"]


def test_shape_families_keep_remeshes_and_conditions_together():
    items = []
    for family in range(4):
        for variant in range(2):
            item = trajectory(f"shape-{family}-case-{variant}", fixed=False)
            original = item.topology.mesh
            changed = VolumeMesh(original.nodes * [1 + family + variant / 10, 1, 1], original.tets)
            items.append(replace(item, topology=material_topology(changed, 1.0),
                                 metadata={**item.metadata, "shape_family": str(family)}))
    splits = split_dynamics_cases(items, split_unit="shape_family")
    families = [set(item.shape_family for item in rows) for rows in splits.values()]
    assert all(families)
    assert not families[0] & families[1]
    assert not families[0] & families[2]
    assert not families[1] & families[2]
    assert sum(len(group) for group in families) == 4
    assignments = {item.case_id: name for name, rows in splits.items() for item in rows}
    for family in range(4):
        assert assignments[f"shape-{family}-case-0"] == assignments[f"shape-{family}-case-1"]


def test_free_drop_readiness_does_not_require_support_release(monkeypatch):
    import fracture_world.dynamics_readiness as readiness

    items = [trajectory(f"case-{index}", fracture=index < 3, fixed=False) for index in range(24)]
    monkeypatch.setattr(readiness, "load_dynamics_dataset", lambda _: items)
    report = inspect_dynamics_dataset("unused", event_stratified=True)
    assert report["ready_for_training"]
    assert report["required_events"] == ["fracture"]
    assert report["support_release_status"] == "not_applicable_no_fixed_vertices"


def test_case_streaming_matches_validated_eager_data_with_one_case_cache(tmp_path):
    from fracture_world.dynamics_data import _streamed_case

    recorded_trajectory(tmp_path / "first")
    recorded_trajectory(tmp_path / "second")
    first = load_dynamics_dataset(tmp_path / "first", stream=True)[0]
    eager = load_dynamics_dataset(tmp_path / "first")[0]
    assert_allclose(first.positions, eager.positions)
    assert_allclose(first.transition(1).target_velocities, eager.transition(1).target_velocities)
    second = load_dynamics_dataset(tmp_path / "second", stream=True)[0]
    assert_allclose(second.positions, eager.positions)
    assert _streamed_case.cache_info().currsize == 1
    assert "positions" not in first.__dict__ and "velocities" not in first.__dict__


def affine_recorded_case(directory, name, stretch):
    original_case, original_trajectory = recorded_trajectory(directory)
    volume = VolumeMesh.load(directory / "mesh.npz")
    transformed = VolumeMesh(volume.nodes * stretch, volume.tets)
    transformed.save(directory / "mesh.npz")
    determinant = float(np.prod(stretch))
    with np.load(original_case, allow_pickle=False) as archive:
        case = {key: np.array(archive[key], copy=True) for key in archive.files}
    metadata = json.loads(str(case["metadata_json"].item()))
    metadata["case"]["id"] = name
    for impulse in metadata["case"]["vertex_impulses"]:
        impulse["impulse"] = (np.asarray(impulse["impulse"]) * stretch * determinant).tolist()
    context = metadata["conditioning_context"]
    context["mesh_si_fingerprint"] = _mesh_digest(transformed)
    context["shape_family"] = name
    metadata["mesh_sha256"] = _mesh_digest(transformed)
    metadata["conditioning_context_sha256"] = context_digest(context)
    case["case_id"] = np.asarray(name)
    case["metadata_json"] = np.asarray(json.dumps(metadata))
    case_path = original_case.with_name(name + ".npz")
    np.savez_compressed(case_path, **case)
    original_case.unlink()
    with np.load(original_trajectory, allow_pickle=False) as archive:
        arrays = {key: np.array(archive[key], copy=True) for key in archive.files}
    for key in ("corner_positions", "corner_velocities", "native_nodal_positions", "native_nodal_velocities", "native_nodal_rest_positions"):
        arrays[key] *= stretch
    arrays["native_nodal_masses"] *= determinant
    if "initial_vertex_masses" in arrays:
        arrays["initial_vertex_masses"] *= determinant
    provenance = json.loads(str(arrays["metadata_json"].item()))
    provenance.update(source_case_sha256=file_hash(case_path), mesh_sha256=_mesh_digest(transformed),
                      conditioning_context_sha256=metadata["conditioning_context_sha256"])
    arrays["metadata_json"] = np.asarray(json.dumps(provenance))
    trajectory_path = original_trajectory.with_name(name + ".npz")
    np.savez_compressed(trajectory_path, **arrays)
    original_trajectory.unlink()
    return case_path, trajectory_path


def test_multi_shape_collection_trains_and_runs_an_explicitly_held_out_shape(tmp_path, monkeypatch):
    entries, sources = [], {}
    for name, stretch, split in (("shape-a", [1, 1, 1], "train"), ("shape-b", [2, 1, 1], "validation"),
                                  ("shape-c", [1, 2, 1], "test")):
        sources[name] = affine_recorded_case(tmp_path / name, name, stretch)
        entries.append(dict(path=name, shape_family=name, split=split))
    manifest = tmp_path / "collection.json"
    manifest.write_text(json.dumps(dict(schema="fracture-world.dynamics-collection.v1", datasets=entries)))
    items = load_dynamics_dataset(manifest, stream=True)
    assert len(items) == 3
    splits = split_dynamics_cases(items, split_unit="shape_family")
    assert splits["test"][0].case_id == "shape-c"
    config = DynamicsTrainingConfig(epochs=1, hidden=12, heads=3, graph_layers=2,
                                    split_unit="shape_family", stream_data=True, device="cpu")
    checkpoint = train_dynamics(manifest, tmp_path / "training", config)
    _, _, saved, _ = load_dynamics_model(checkpoint, "cpu")
    assert saved["shape_family_splits"] == {"train": ["shape-a"], "validation": ["shape-b"], "test": ["shape-c"]}
    assert items[-1].topology.fingerprint not in saved["training_mesh_fingerprints"]
    initial_path = tmp_path / "heldout-initial.npz"
    export_initial(*sources["shape-c"], 0, initial_path, support_policy="free")
    report = rollout_dynamics(checkpoint, initial_path, tmp_path / "heldout-rollout", 1, "raw",
                              RolloutConfig(fracture_samples=1, flow_steps=1), "cpu")
    assert report["completed_steps"] == 1
    assert report["held_out_shape_family"]
    assert report["shape_family"] == "shape-c"
    benchmark = benchmark_dynamics(checkpoint, manifest, tmp_path / "benchmark", 1,
                                   RolloutConfig(fracture_samples=0, fracture_policy="marginal_search"), "cpu", ["physics"])
    assert len(benchmark["cases"]) == 1
    assert benchmark["conditions"]["physics"]["held_out_shape_family_cases"] == 1
    assert benchmark["reference_used_only_for_evaluation"]
    assert len(benchmark["checkpoint_sha256"]) == 64
    monkeypatch.setattr("fracture_world.dynamics_experiment.compare_dynamics", lambda *args, **kwargs: dict(
        all_rollouts_completed=False, conditions={"physics": dict(status="timestep_rejected", runtime_seconds=0.1,
            all_committed_geometry_valid=True, evaluation={"fracture": {"f1": 1.0}})}))
    failed = benchmark_dynamics(checkpoint, manifest, tmp_path / "failed-benchmark", 3,
                                RolloutConfig(fracture_samples=0, fracture_policy="marginal_search"), "cpu", ["physics"])
    summary = failed["conditions"]["physics"]
    assert summary["fracture_positive_case_f1_failure_penalized"] == 0
    assert summary["completed_rate"] == summary["completed_and_geometry_valid_rate"] == 0
    assert summary["full_horizon_accuracy_cases"] == 0
    entries[-1]["shape_family"] = "shape-a"
    manifest.write_text(json.dumps(dict(schema="fracture-world.dynamics-collection.v1", datasets=entries)))
    with pytest.raises(ValueError, match="shape family differs"):
        load_dynamics_dataset(manifest, stream=True)
