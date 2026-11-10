import copy
import hashlib
import json
import sys
from types import ModuleType

import numpy as np
import pytest
import torch

from fracture_world.cases import Case
from fracture_world.cut_event import CutSequence, corner_weights, experiment_plan, load_trajectory
from fracture_world.cut_event_evaluation import causal_metrics, evaluate_cut_event, sequence_metrics
from fracture_world.cut_event_teacher import simulate_event
from fracture_world.equilibrium import EquilibriumDynamicsModel
from fracture_world.generation2d import cut_mesh, file_hash
from fracture_world.render2d import render, trajectory_frame, trajectory_time
from fracture_world.stage_a import main
from fracture_world.teacher2d import Material, Pull
from fracture_world.topology import lift_state


@pytest.fixture(autouse=True)
def threads():
    previous = torch.get_num_threads()
    torch.set_num_threads(2)
    yield
    torch.set_num_threads(previous)


def sample(tmp_path):
    config = {
        "mesh": {"nx": 4, "ny": 4, "width": 1.0, "height": 1.0},
        "material": {"young": 100.0, "poisson": 0.25, "density": 1.0, "thickness": 1.0},
        "time": {"step": 0.002, "duration": 0.012}, "cut_columns": [1, 2, 3],
        "cut_depths": {"train": [1], "validation": [2], "test": [3, 4]},
        "loads": [{"amplitude": 0.015, "ramp_time": 0.04}],
    }
    mesh = cut_mesh(config, 2, 3)
    time = np.linspace(0, 0.012, 7)
    pull = Pull(**config["loads"][0])
    mask, boundary_position, boundary_velocity = pull.boundary(mesh, time)
    source = Case({
        "rest": mesh.rest.copy(), "triangles": mesh.triangles.copy(),
        "original_vertex": mesh.original_vertex.copy(), "cut_edges": mesh.cut_edges.copy(),
        "time": time, "position": boundary_position.copy(), "velocity": boundary_velocity.copy(),
        "boundary_mask": mask, "boundary_position": boundary_position, "boundary_velocity": boundary_velocity,
    }, {"material": Material(**config["material"]).metadata(), "pull": config["loads"][0]})
    source.save(tmp_path / "source.npz")
    row = {"id": "x2-depth3-load0", "group": "x2-depth3", "split": "test", "path": "source.npz",
           "case_hash": str(source.arrays["case_hash"].item()), "file_sha256": file_hash(tmp_path / "source.npz")}
    manifest = {"format": "cut-dynamics-a-v1", "configuration": config, "cases": [row]}
    data = tmp_path / "collection.json"
    data.write_text(json.dumps(manifest), encoding="utf-8")
    return data, source, config


def checkpoint(tmp_path, method="stress", trained=False):
    model = EquilibriumDynamicsModel(method, hidden_size=8, layers=1)
    path = tmp_path / "best.pt"
    torch.save({"format": "cut-dynamics-a-v2", "trained": trained,
                "model_config": model.configuration(), "state_dict": model.state_dict(),
                "dt": 0.002, "training_case_hashes": [], "training_groups": ["x1-depth1"]}, path)
    return path


def sequence_parts(source, final, event_index=3):
    before_values = {key: value.copy() for key, value in source.arrays.items() if key != "case_hash"}
    for key in ("time", "position", "velocity", "boundary_position", "boundary_velocity"):
        before_values[key] = before_values[key][:event_index + 1]
    before = Case(before_values, copy.deepcopy(source.metadata))
    time = source.arrays["time"][event_index:]
    mask, targets, speeds = Pull(**source.metadata["pull"]).boundary(final, time)
    position, velocity = lift_state(before.mesh, final, before.arrays["position"][-1], before.arrays["velocity"][-1])
    positions = np.broadcast_to(position, targets.shape).copy()
    velocities = np.broadcast_to(velocity, targets.shape).copy()
    positions[:, mask] = targets[:, mask]
    velocities[:, mask] = speeds[:, mask]
    after = Case({"rest": final.rest, "triangles": final.triangles, "original_vertex": final.original_vertex,
                  "cut_edges": final.cut_edges, "time": time, "position": positions, "velocity": velocities,
                  "boundary_mask": mask, "boundary_position": targets, "boundary_velocity": speeds},
                 copy.deepcopy(source.metadata))
    return before, after


def test_plan_finishes_a_held_out_notch_without_mutating_source(tmp_path):
    data, source, _ = sample(tmp_path)
    digest = file_hash(tmp_path / "source.npz")
    plan = experiment_plan(data, event_time=0.006)
    assert len(plan) == 1
    scenario = plan[0]
    assert scenario["inputs"]["initial_cut_depth"] == 3
    assert scenario["inputs"]["final_cut_depth"] == 4
    assert len(scenario["final_mesh"].cut_edges) == 4
    assert len(scenario["final_mesh"].rest) > len(source.mesh.rest)
    assert file_hash(tmp_path / "source.npz") == digest
    assert scenario["experiment_hash"] != experiment_plan(data, event_time=0.008)[0]["experiment_hash"]


@pytest.mark.parametrize("event", [0.0, 0.003, 0.012, float("nan")])
def test_plan_rejects_ambiguous_or_noninterior_event(tmp_path, event):
    data, _, _ = sample(tmp_path)
    with pytest.raises(ValueError):
        experiment_plan(data, event_time=event)


def test_sequence_roundtrip_and_renderer_switch_at_the_event(tmp_path):
    data, source, _ = sample(tmp_path)
    scenario = experiment_plan(data, event_time=0.006)[0]
    before, after = sequence_parts(source, scenario["final_mesh"])
    metadata = {"method_label": "cut_updated", "experiment_hash": "same-inputs", "event_time": 0.006}
    sequence = CutSequence(before, after, metadata)
    saved = tmp_path / "sequence.npz"
    sequence.save(saved)
    loaded = load_trajectory(saved)
    np.testing.assert_array_equal(trajectory_time(loaded), source.arrays["time"])
    assert trajectory_frame(loaded, 2)[0].mesh.rest.shape == before.mesh.rest.shape
    assert trajectory_frame(loaded, 3)[0].mesh.rest.shape == after.mesh.rest.shape
    np.testing.assert_array_equal(loaded.corners("position"), sequence.corners("position"))
    assert corner_weights(before).sum() == pytest.approx(before.mass.sum())
    image = tmp_path / "comparison.gif"
    render([saved, saved], image, fps=10)
    from PIL import Image

    with Image.open(image) as result:
        assert result.n_frames == len(source.arrays["time"])


def test_sequence_rejects_an_event_velocity_reset(tmp_path):
    data, source, _ = sample(tmp_path)
    scenario = experiment_plan(data, event_time=0.006)[0]
    before, after = sequence_parts(source, scenario["final_mesh"])
    after.arrays["velocity"][0, ~after.arrays["boundary_mask"]] += 0.01
    with pytest.raises(ValueError, match="event velocity"):
        CutSequence(before, after, {})


def test_mass_weighted_corner_metrics_do_not_depend_on_node_duplication(tmp_path):
    data, source, _ = sample(tmp_path)
    scenario = experiment_plan(data, event_time=0.006)[0]
    before, after = sequence_parts(source, scenario["final_mesh"])
    metadata = {"experiment_hash": "test", "event_time": 0.006}
    reference = CutSequence(before, after, metadata)
    prediction = copy.deepcopy(reference)
    prediction.after.arrays["position"][1:] += [0.03, 0.04]
    result = sequence_metrics(reference, prediction)
    assert result["final_position_rmse"] == pytest.approx(0.05)
    assert result["post_event_position_rmse"] == pytest.approx(0.05)
    assert causal_metrics({"cut": reference, "no_cut": reference}, {
        "no_cut": reference, "cut_updated": reference,
    })["cut_updated"]["position_intervention_rmse"] == 0


def test_untrained_end_to_end_evaluation_preserves_inputs_and_uses_own_references(tmp_path):
    pytest.importorskip("felupe")
    data, _, _ = sample(tmp_path)
    weights = checkpoint(tmp_path)
    original = {path: hashlib.sha256(path.read_bytes()).hexdigest() for path in
                (weights, data, tmp_path / "source.npz")}
    output = tmp_path / "evaluation"
    report = evaluate_cut_event(data, weights, output, event_time=0.006, substeps=2,
                                allow_untrained=True, render_results=True)
    assert report["status"] == "completed"
    assert report["purpose"] == "untrained_pipeline_check"
    assert report["training_executed"] is False
    assert report["checkpoint_unchanged"] is True
    case = report["cases"][0]
    assert case["reference"]["converged"]
    assert case["reference"]["event_transfer"]["status"] == "conserved"
    assert case["neural_event_transfer"]["status"] == "conserved"
    assert case["methods"]["no_cut"]["reference_condition"] == "no_cut"
    for name in ("cut_updated", "cut_stale"):
        assert case["methods"][name]["reference_condition"] == "cut"
        trajectory = CutSequence.load(output / case["methods"][name]["trajectory"])
        np.testing.assert_array_equal(trajectory.after.arrays["position"][0],
                                      lift_state(trajectory.before.mesh, trajectory.after.mesh,
                                                 trajectory.before.arrays["position"][-1],
                                                 trajectory.before.arrays["velocity"][-1])[0])
    for path, digest in original.items():
        assert hashlib.sha256(path.read_bytes()).hexdigest() == digest
    assert (output / case["render"]).exists()
    json.loads((output / "report.json").read_text(encoding="utf-8"))
    with pytest.raises(ValueError, match="new or empty"):
        evaluate_cut_event(data, weights, output, event_time=0.006, allow_untrained=True)


def test_wrong_checkpoint_and_untrained_weights_are_rejected_before_teacher(tmp_path):
    data, _, _ = sample(tmp_path)
    with pytest.raises(ValueError, match="equilibrium stress"):
        evaluate_cut_event(data, checkpoint(tmp_path, "direct"), tmp_path / "result", event_time=0.006)
    with pytest.raises(ValueError, match="allow-untrained"):
        evaluate_cut_event(data, checkpoint(tmp_path), tmp_path / "result", event_time=0.006)


def test_checkpoint_without_training_provenance_is_not_called_held_out(tmp_path):
    data, _, _ = sample(tmp_path)
    weights = checkpoint(tmp_path)
    values = torch.load(weights, weights_only=True)
    del values["training_groups"]
    torch.save(values, weights)
    with pytest.raises(ValueError, match="training provenance"):
        evaluate_cut_event(data, weights, tmp_path / "result", event_time=0.006, allow_untrained=True)


def test_reference_failure_is_recorded_without_a_false_pass(tmp_path, monkeypatch):
    import fracture_world.cut_event_evaluation as evaluation

    data, _, _ = sample(tmp_path)
    weights = checkpoint(tmp_path)

    def fail(*args, **kwargs):
        raise RuntimeError("Reference integration did not complete")

    monkeypatch.setattr(evaluation, "reference_sequences", fail)
    report = evaluate_cut_event(data, weights, tmp_path / "result", event_time=0.006, allow_untrained=True)
    assert report["status"] == "incomplete"
    assert report["cases"][0]["status"] == "reference_failed"
    assert report["summary"]["methods"]["cut_updated"]["unattempted"] == 1
    assert report["summary"]["methods"]["cut_updated"]["mean_post_event_position_rmse"] is None


def test_diagnostic_failure_does_not_reset_or_prevent_autonomous_rollout(tmp_path, monkeypatch):
    pytest.importorskip("felupe")
    import fracture_world.cut_event_evaluation as evaluation

    data, _, _ = sample(tmp_path)
    weights = checkpoint(tmp_path)

    def fail(*args, **kwargs):
        raise FloatingPointError("Non-finite diagnostic force")

    monkeypatch.setattr(evaluation, "event_force_audit", fail)
    report = evaluate_cut_event(data, weights, tmp_path / "result", event_time=0.006, allow_untrained=True)
    assert report["cases"][0]["same_state_event_force"]["status"] == "failed"
    assert report["cases"][0]["methods"]["cut_updated"]["status"] == "completed"


class ConstitutiveOracle(EquilibriumDynamicsModel):
    def encode(self, graph, strain):
        poisson = graph.material[graph.triangles[:, 0], 1]
        shear = 1 / (2 * (1 + poisson))
        lame = poisson / ((1 + poisson) * (1 - 2 * poisson))
        trace = strain[:, 0] + strain[:, 1]
        return torch.stack((lame * trace + 2 * shear * strain[:, 0],
                            lame * trace + 2 * shear * strain[:, 1],
                            2 * shear * strain[:, 2]), dim=-1)


def test_force_oracle_event_rollout_converges_to_independent_felupe_teacher(tmp_path):
    pytest.importorskip("felupe")
    from fracture_world.cut_event_runtime import event_rollout

    data, source, config = sample(tmp_path)
    scenario = experiment_plan(data, event_time=0.006)[0]
    time = np.linspace(0, 0.08, 41)
    before, after, _ = simulate_event(
        source.mesh, scenario["final_mesh"], Material(**config["material"]), time, time[3],
        config["loads"][0], {"rtol": 1e-11, "atol": 1e-13, "max_step": 0.00025},
    )
    metadata = {"experiment_hash": "oracle-check", "event_time": time[3]}
    reference = CutSequence(before, after, metadata)
    model = ConstitutiveOracle("stress", hidden_size=8, layers=1).double()
    errors = []
    for substeps in (1, 2, 4):
        result = event_rollout(model, source, scenario["final_mesh"], time, time[3], substeps=substeps)
        assert result["statuses"]["cut_updated"]["status"] == "completed"
        sequence = CutSequence(result["before"], result["afters"]["cut_updated"], metadata)
        errors.append(sequence_metrics(reference, sequence)["post_event_velocity_rmse"])
        np.testing.assert_array_equal(result["afters"]["cut_updated"].arrays["position"],
                                      result["afters"]["cut_stale"].arrays["position"])
    assert errors[1] < errors[0] * 0.4
    assert errors[2] < errors[1] * 0.4


def test_cli_dispatches_without_training(monkeypatch, capsys):
    received = {}
    module = ModuleType("fracture_world.cut_event_evaluation")

    def evaluate(**kwargs):
        received.update(kwargs)
        return {"training_executed": False}

    module.evaluate_cut_event = evaluate
    monkeypatch.setitem(sys.modules, module.__name__, module)
    main(["cut-event-evaluate", "--data", "collection.json", "--checkpoint", "best.pt",
          "--output", "fresh-result", "--device", "cuda", "--substeps", "4", "--render"])
    assert received["event_time"] == 0.03
    assert received["substeps"] == 4
    assert received["render_results"] is True
    assert json.loads(capsys.readouterr().out)["training_executed"] is False
