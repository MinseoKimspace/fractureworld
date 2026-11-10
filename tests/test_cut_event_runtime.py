import copy
import sys

import numpy as np
import pytest
import torch
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import connected_components

from fracture_world.cases import Case
from fracture_world.cut_event_runtime import context_force, event_rollout, make_graph
from fracture_world.equilibrium import EquilibriumDynamicsModel
from fracture_world.neural import DynamicsModel
from fracture_world.teacher2d import Material, Pull
from fracture_world.topology import lift_state, mesh_edges, rectangle, split_edges


@pytest.fixture(autouse=True)
def threads():
    previous = torch.get_num_threads()
    torch.set_num_threads(2)
    yield
    torch.set_num_threads(previous)


def sample():
    base = rectangle(4, 4)
    vertices = np.arange(5) * 5 + 2
    edges = np.column_stack((vertices[:-1], vertices[1:]))
    mesh = split_edges(base, edges[:-1])
    final_mesh = split_edges(mesh, edges)
    time = np.linspace(0, 0.06, 13)
    material = Material().metadata()
    pull = Pull(0.015, 0.04)
    mask, position, velocity = pull.boundary(mesh, time)
    case = Case({
        "rest": mesh.rest.copy(), "triangles": mesh.triangles.copy(),
        "original_vertex": mesh.original_vertex.copy(), "cut_edges": mesh.cut_edges.copy(),
        "time": time.copy(), "position": position.copy(), "velocity": velocity.copy(),
        "boundary_mask": mask, "boundary_position": position.copy(), "boundary_velocity": velocity.copy(),
    }, {"material": material, "pull": {"amplitude": pull.amplitude, "ramp_time": pull.ramp_time}})
    return case, final_mesh, time


def model():
    torch.manual_seed(41)
    result = EquilibriumDynamicsModel("stress", hidden_size=8, layers=2)
    with torch.no_grad():
        result.decoder[-1].weight.normal_(std=0.03)
    return result


def test_runtime_is_autonomous_and_uses_no_external_engine(monkeypatch):
    case, final_mesh, time = sample()
    first = model()
    snapshot = {key: value.clone() for key, value in first.state_dict().items()}
    original = {key: value.copy() for key, value in case.arrays.items()}
    altered = Case(copy.deepcopy(case.arrays), copy.deepcopy(case.metadata))
    free = ~altered.arrays["boundary_mask"]
    altered.arrays["position"][1:, free] += 100
    altered.arrays["velocity"][1:, free] -= 1000
    monkeypatch.setitem(sys.modules, "felupe", None)
    expected = event_rollout(first, case, final_mesh, time, 0.02)
    actual = event_rollout(first, altered, final_mesh, time, 0.02)
    assert expected["before_status"]["status"] == "completed"
    assert all(status["status"] == "completed" for status in expected["statuses"].values())
    for label in expected["afters"]:
        for key in ("position", "velocity", "internal_force", "boundary_impulse"):
            np.testing.assert_array_equal(expected["afters"][label].arrays[key], actual["afters"][label].arrays[key])
        assert expected["afters"][label].metadata["external_physics_engine"] is False
        assert expected["afters"][label].metadata["numerical_corrector"] is False
    for key, value in original.items():
        np.testing.assert_array_equal(case.arrays[key], value)
    for key, value in snapshot.items():
        torch.testing.assert_close(first.state_dict()[key], value, rtol=0, atol=0)
    assert first.training


def test_event_continuity_and_force_cache_invalidation():
    case, final_mesh, time = sample()
    result = event_rollout(model(), case, final_mesh, time, 0.02, substeps=2)
    before = result["before"]
    expected_x, expected_v = lift_state(case.mesh, final_mesh, before.arrays["position"][-1], before.arrays["velocity"][-1])
    for label in ("cut_stale", "cut_updated"):
        after = result["afters"][label]
        np.testing.assert_array_equal(after.arrays["position"][0], expected_x)
        np.testing.assert_array_equal(after.arrays["velocity"][0], expected_v)
        assert after.arrays["time"][0] == before.arrays["time"][-1] == 0.02
        assert result["statuses"][label]["force_evaluations"] == 17
    np.testing.assert_array_equal(result["afters"]["no_cut"].arrays["position"][0], before.arrays["position"][-1])
    assert result["before_status"]["force_evaluations"] == 9
    assert result["statuses"]["no_cut"]["force_evaluations"] == 16
    assert not np.array_equal(result["afters"]["cut_stale"].arrays["internal_force"][0], result["afters"]["cut_updated"].arrays["internal_force"][0])
    np.testing.assert_array_equal(result["afters"]["cut_stale"].arrays["triangles"], final_mesh.triangles)
    np.testing.assert_array_equal(result["afters"]["cut_updated"].arrays["triangles"], final_mesh.triangles)


def test_global_grip_time_does_not_restart_at_cut():
    case, final_mesh, time = sample()
    result = event_rollout(model(), case, final_mesh, time, 0.02)
    after = result["afters"]["cut_updated"]
    pull = Pull(**case.metadata["pull"])
    mask, expected_x, expected_v = pull.boundary(final_mesh, after.arrays["time"])
    np.testing.assert_array_equal(after.arrays["boundary_position"], expected_x)
    np.testing.assert_array_equal(after.arrays["boundary_velocity"], expected_v)
    np.testing.assert_allclose(after.arrays["position"][:, mask], expected_x[:, mask], rtol=1e-6, atol=1e-10)
    assert not np.array_equal(expected_x[0], final_mesh.rest)


@pytest.mark.parametrize("frozen", [False, True])
def test_new_physical_assembly_preserves_component_force_and_torque_balance(frozen):
    case, final_mesh, _ = sample()
    predictor = model().double()
    pull = Pull(**case.metadata["pull"])
    physical = make_graph(final_mesh, case.metadata["material"], pull, 0.02, torch.float64, "cpu")
    context = make_graph(case.mesh, case.metadata["material"], pull, 0.02, torch.float64, "cpu") if frozen else None
    position = physical.rest.clone()
    position[:, 0] += 0.025 * position[:, 0] * position[:, 1]
    position[:, 1] += 0.01 * position[:, 0].square()
    with torch.no_grad():
        force = context_force(predictor, physical, context, position).numpy()
        at_rest = context_force(predictor, physical, context, physical.rest)
    torch.testing.assert_close(at_rest, torch.zeros_like(at_rest), rtol=0, atol=0)
    edges = mesh_edges(final_mesh)
    graph = coo_matrix((np.ones(len(edges)), (edges[:, 0], edges[:, 1])), shape=(len(final_mesh.rest), len(final_mesh.rest)))
    count, labels = connected_components(graph, directed=False)
    assert count == 2
    for component in range(count):
        selected = labels == component
        np.testing.assert_allclose(force[selected].sum(axis=0), 0, rtol=0, atol=1e-13)
        points = position.numpy()[selected]
        torque = points[:, 0] * force[selected, 1] - points[:, 1] * force[selected, 0]
        np.testing.assert_allclose(torque.sum(), 0, rtol=0, atol=1e-13)


def test_full_cut_removes_physical_cross_interface_connections():
    case, final_mesh, time = sample()
    result = event_rollout(model(), case, final_mesh, time, 0.02)
    for label in ("cut_stale", "cut_updated"):
        mesh = result["afters"][label].mesh
        points = mesh.rest[mesh.triangles]
        assert not np.any((points[:, :, 0].min(axis=1) < 0.5) & (points[:, :, 0].max(axis=1) > 0.5))
        left = np.unique(mesh.triangles[points[:, :, 0].mean(axis=1) < 0.5])
        right = np.unique(mesh.triangles[points[:, :, 0].mean(axis=1) > 0.5])
        assert not np.intersect1d(left, right).size
    assert result["afters"]["cut_stale"].metadata["context_mode"] == "frozen_pre_cut_encoder_context"


def test_failures_are_not_marked_complete(monkeypatch):
    case, final_mesh, time = sample()
    predictor = model()
    original = predictor.instantaneous_force
    calls = 0

    def invalid_force(graph, position):
        nonlocal calls
        calls += 1
        return original(graph, position) if calls < 3 else torch.full_like(position, float("nan"))

    monkeypatch.setattr(predictor, "instantaneous_force", invalid_force)
    result = event_rollout(predictor, case, final_mesh, time, 0.02)
    assert result["before_status"]["status"] == "failed"
    assert result["before_status"]["completed_steps"] < result["before_status"]["requested_steps"]
    assert all(status["status"] == "failed" for status in result["statuses"].values())
    assert all(value is None for value in result["afters"].values())


def test_transient_substep_inversion_stops_before_output_and_remains_invalid(monkeypatch):
    import fracture_world.cut_event_runtime as runtime

    case, final_mesh, time = sample()
    original = runtime.velocity_verlet
    calls = 0

    def transient_inversion(*args, **kwargs):
        nonlocal calls
        calls += 1
        result = original(*args, **kwargs)
        if calls == 3:
            result.position = result.position.clone()
            result.position[:, 0] *= -1
        return result

    monkeypatch.setattr(runtime, "velocity_verlet", transient_inversion)
    result = event_rollout(model(), case, final_mesh, time, 0.02, substeps=2)
    assert calls == 3
    assert result["before_status"]["status"] == "failed"
    assert result["before_status"]["orientation_valid"] is False
    assert result["before_status"]["invalid_substep_detected"] is True
    assert "substep 1" in result["before_status"]["failure"]
    assert result["before_status"]["completed_steps"] == 1
    before = result["before"]
    assert before is not None
    assert len(before.arrays["time"]) == 2
    assert all(runtime.orientation_valid(before.mesh, position) for position in before.arrays["position"])
    assert all(status["status"] == "failed" for status in result["statuses"].values())


def test_existing_checkpoint_loads_without_architecture_or_weight_changes(tmp_path):
    case, final_mesh, time = sample()
    predictor = model()
    path = tmp_path / "best.pt"
    torch.save({"format": "cut-dynamics-a-v2", "model_config": predictor.configuration(), "state_dict": predictor.state_dict()}, path)
    original_bytes = path.read_bytes()
    from fracture_world.training2d import load_model

    loaded, _ = load_model(path)
    result = event_rollout(loaded, case, final_mesh, time, 0.02)
    assert result["before_status"]["status"] == "completed"
    assert path.read_bytes() == original_bytes


@pytest.mark.parametrize("predictor", [DynamicsModel("stress", 8, 1), EquilibriumDynamicsModel("direct", 8, 1)])
def test_unsupported_model_is_rejected(predictor):
    case, final_mesh, time = sample()
    with pytest.raises(ValueError, match="equilibrium stress"):
        event_rollout(predictor, case, final_mesh, time, 0.02)


@pytest.mark.parametrize("event_time", [0, 0.06, 0.021, float("nan")])
def test_invalid_event_time_is_rejected(event_time):
    case, final_mesh, time = sample()
    with pytest.raises(ValueError, match="Event time"):
        event_rollout(model(), case, final_mesh, time, event_time)


def test_mismatched_context_is_rejected():
    case, final_mesh, _ = sample()
    pull = Pull(**case.metadata["pull"])
    physical = make_graph(final_mesh, case.metadata["material"], pull, 0.02, torch.float32, "cpu")
    context = physical.clone()
    context.material[0, 0] *= 2
    with pytest.raises(ValueError, match="material corners"):
        context_force(model(), physical, context, physical.rest)
