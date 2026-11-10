from types import SimpleNamespace

import numpy as np
import pytest

import fracture_world.cut_event_evaluation as evaluation
from fracture_world.cases import Case
from fracture_world.cut_event import CutSequence
from fracture_world.topology import Mesh


@pytest.fixture
def synthetic_references(monkeypatch):
    mesh = Mesh(np.array([[0.0, 0.0], [2.0, 0.0], [0.0, 1.0], [2.0, 2.0]]),
                np.array([[0, 1, 2], [1, 3, 2]]), np.arange(4))
    time = np.arange(4) * 0.002
    material = {"young": 100.0, "poisson": 0.25, "density": 1.0, "thickness": 1.0}
    metadata = {"material": material, "pull": {"amplitude": 0.02, "ramp_time": 0.04}}
    pattern = np.array([[1.0, 0.0], [0.0, 2.0], [3.0, -1.0], [-2.0, 4.0]])

    def segment(times, positions, velocities):
        rest = np.broadcast_to(mesh.rest, (len(times), len(mesh.rest), 2)).copy()
        return Case({
            "rest": mesh.rest.copy(), "triangles": mesh.triangles.copy(),
            "original_vertex": mesh.original_vertex.copy(), "time": times.copy(),
            "position": rest + np.asarray(positions)[:, None, None] * pattern * 1e-9,
            "velocity": np.asarray(velocities)[:, None, None] * pattern * 1e-9,
            "boundary_mask": np.zeros_like(mesh.rest, dtype=bool),
            "boundary_position": rest, "boundary_velocity": np.zeros_like(rest),
            "boundary_impulse": np.zeros((len(times) - 1, len(mesh.rest), 2)),
        }, metadata.copy())

    source = segment(time, [0, 0, 0, 0], [0, 0, 0, 0])
    levels = [
        (segment(time[:2], [0, 0], [0, 0]),
         segment(time[1:], [0, 2, 5], [0, 1, 3]),
         segment(time[1:], [0, 0.5, 1], [0, 0.25, 0.5])),
        (segment(time[:2], [0, 0], [0, 0]),
         segment(time[1:], [0, 2.25, 4.5], [0, 1.5, 2.75]),
         segment(time[1:], [0, 0, 1.25], [0, 0, 1])),
    ]
    event_calls, uncut_calls = iter(levels), iter(levels)
    monkeypatch.setattr(evaluation, "simulate_event", lambda *args: (
        *next(event_calls)[:2], {"status": "conserved"},
    ))
    monkeypatch.setattr(evaluation, "simulate_segment", lambda *args, **kwargs: next(uncut_calls)[2])
    scenario = {"source": source, "final_mesh": mesh, "time": time, "event_time": time[1],
                "experiment_hash": "synthetic", "inputs": {"source_case_hash": "source"}}
    return scenario, levels


def paired_refinement(scenario, levels, field):
    metadata = {"event_time": scenario["event_time"]}
    interventions = []
    for before, cut, uncut in levels:
        interventions.append(CutSequence(before, cut, metadata).corners(field)
                             - CutSequence(before, uncut, metadata).corners(field))
    post = scenario["time"] > scenario["event_time"]
    difference = (interventions[1] - interventions[0])[post].reshape(int(post.sum()), -1, 2)
    masses = np.repeat([1 / 3, 2 / 3], 3)
    return np.sqrt(np.sum(difference ** 2 * masses[None, :, None]) / (post.sum() * masses.sum()))


def test_causal_refinement_uses_paired_post_event_differences_and_unequal_corner_masses(synthetic_references):
    scenario, levels = synthetic_references
    expected = {field: paired_refinement(scenario, levels, field) for field in ("position", "velocity")}
    references, audit = evaluation.reference_sequences(scenario)
    for field in ("position", "velocity"):
        assert audit["causal_refinement"][f"{field}_intervention_refinement_rms"] == pytest.approx(
            expected[field], rel=1e-12, abs=1e-22,
        )
        np.testing.assert_array_equal(references["cut"].after.arrays[field], levels[1][1].arrays[field])
    assert audit["converged"] is True
    assert audit["limits"] == {"position_max_difference": 1e-7, "velocity_max_difference": 1e-6,
                               "boundary_impulse_max_difference": 1e-7}
    assert audit["coarse_settings"] == {"rtol": 1e-10, "atol": 1e-12, "max_step": 0.0005}
    assert audit["refined_settings"] == {"rtol": 1e-11, "atol": 1e-13, "max_step": 0.00025}


@pytest.mark.parametrize("field,limit", [("position", 1e-7), ("velocity", 1e-6),
                                        ("boundary_impulse", 1e-7)])
def test_causal_diagnostic_does_not_weaken_reference_acceptance_limits(synthetic_references, field, limit):
    scenario, levels = synthetic_references
    levels[1][1].arrays[field][-1, 0, 0] += 2 * limit
    _, audit = evaluation.reference_sequences(scenario)
    key = f"{field}_max_difference"
    assert audit["limits"][key] == limit
    assert audit["refinement"]["cut"][key] > limit
    assert audit["checks"]["cut"][key] is False
    assert audit["converged"] is False


def test_common_reference_error_cancels_in_causal_floor_without_passing_absolute_checks(synthetic_references):
    scenario, levels = synthetic_references
    for field in ("position", "velocity"):
        for coarse, refined in zip(levels[0][1:], levels[1][1:]):
            refined.arrays[field][1:] = coarse.arrays[field][1:]
            refined.arrays[field][1:, 0, 0] += 1e-4
    _, audit = evaluation.reference_sequences(scenario)
    assert audit["causal_refinement"]["position_intervention_refinement_rms"] == pytest.approx(0, abs=1e-18)
    assert audit["causal_refinement"]["velocity_intervention_refinement_rms"] == pytest.approx(0, abs=1e-18)
    assert audit["converged"] is False
    assert audit["checks"]["cut"]["position_max_difference"] is False
    assert audit["checks"]["no_cut"]["velocity_max_difference"] is False


@pytest.mark.parametrize("scenarios,message", [
    ([], "nonempty"),
    ([{"id": "same", "experiment_hash": "a"}, {"id": "same", "experiment_hash": "b"}], "unique"),
    ([{"id": "a", "experiment_hash": "same"}, {"id": "b", "experiment_hash": "same"}], "duplicate physical"),
])
def test_explicit_scenario_plan_rejects_empty_or_duplicate_conditions_before_execution(
    tmp_path, monkeypatch, scenarios, message,
):
    model = SimpleNamespace(architecture="equilibrium", method="stress")
    details = {"trained": True, "dt": 0.002, "training_groups": [], "training_case_hashes": []}
    monkeypatch.setattr(evaluation, "load_model", lambda *args: (model, details))

    def unexpected(*args, **kwargs):
        pytest.fail("Invalid explicit plans must be rejected before data planning or reference execution")

    monkeypatch.setattr(evaluation, "experiment_plan", unexpected)
    monkeypatch.setattr(evaluation, "reference_sequences", unexpected)
    output = tmp_path / "evaluation"
    with pytest.raises(ValueError, match=message):
        evaluation.evaluate_cut_event(tmp_path / "collection.json", tmp_path / "best.pt", output,
                                      scenario_plan=scenarios)
    assert not output.exists()
