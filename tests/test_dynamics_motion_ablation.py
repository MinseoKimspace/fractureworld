from dataclasses import replace
from types import SimpleNamespace

import numpy as np
from numpy.testing import assert_allclose, assert_array_equal
import pytest
import torch

from fracture_world import dynamics_matched as matched
from fracture_world.dynamics_initial import load_initial
from fracture_world.dynamics_learning import load_dynamics_model, state_batch
from fracture_world.dynamics_runtime import RolloutConfig
from fracture_world.io import file_hash
from test_dynamics_runtime import portable_rollout_inputs


ABLATION_METHODS = ("raw", "central", "consensus", "no_motion_central", "no_motion_consensus")


def motion_inputs(directory, monkeypatch, motion_prior="constant_acceleration"):
    directory.mkdir(parents=True, exist_ok=True)
    checkpoint, initial_path, _, _ = portable_rollout_inputs(directory, learned_supports=True, has_corrector=False)
    model, _, _, _ = load_dynamics_model(checkpoint, "cpu")
    model.config["motion_prior"] = motion_prior
    with torch.no_grad():
        model.motion_head.weight.zero_()
        model.motion_head.bias.copy_(torch.tensor([.4, -.3, .2, .2, -.1, .3]))
    observed = load_initial(initial_path)
    initial = replace(observed, metadata={**observed.metadata, "teacher_physics": dict(
        rayleigh_mass_damping=0., rayleigh_stiffness_damping=0., friction_coefficient=0.)})
    config = RolloutConfig(fracture_samples=2, fracture_policy="fm_search", flow_steps=1,
                           mechanics_mode="proximal", kinematic_rule="backward_euler", energy_balance=False)
    calls = dict(samples=[], forwards=[], solves=[], evaluators=[])
    predict = model.forward_with_flow_context

    def prediction(batch):
        output, context = predict(batch)
        calls["forwards"].append({name: value.detach().clone() for name, value in output.items()})
        return output, context

    def sample(batch, steps, noise, context=None):
        index = len(calls["samples"])
        calls["samples"].append(dict(noise=noise.clone(), context=context))
        return torch.full_like(noise, 1. if index % 2 == 0 else -1.)

    def correct(observation, positions, velocities, proposal, mask, fixed, network, settings, method, *args, **kwargs):
        next_positions, next_velocities = matched.runtime._proposal_arrays(proposal, positions, velocities)
        calls["solves"].append(dict(method=method, positions=positions.copy(), velocities=velocities.copy(),
                                     proposal_positions=next_positions.copy(), proposal_velocities=next_velocities.copy(),
                                     mask=mask.copy(), fixed=fixed.copy(), proposal={name: value.detach().clone()
                                         if isinstance(value, torch.Tensor) else value for name, value in proposal.items()},
                                     envelope=kwargs.get("initial_velocity_envelope")))
        solution = SimpleNamespace(positions=next_positions, velocities=next_velocities, converged=True, objective=0.,
                                   diagnostics=dict(constraints_satisfied=True))
        diagnostic = dict(status="converged", solves=[], geometry=dict(valid=True), swept_geometry=dict(valid=True))
        return solution, diagnostic

    frozen = matched._frozen_objective

    def evaluation(*args, **kwargs):
        problem = frozen(*args, **kwargs)
        calls["evaluators"].append(problem)
        return problem

    monkeypatch.setattr(model, "forward_with_flow_context", prediction)
    monkeypatch.setattr(model, "sample_fracture", sample)
    monkeypatch.setattr(matched.runtime, "_correct", correct)
    monkeypatch.setattr(matched, "_frozen_objective", evaluation)
    return checkpoint, initial, model, config, calls


def expected_physical_motion(initial, model):
    batch = state_batch(initial.topology, initial.positions, initial.velocities, initial.broken, initial.fixed,
                        initial.gravity, initial.material, initial.dt, "cpu", initial.metadata.get("floor_height_m"))
    dt = torch.as_tensor(batch["dt"], dtype=batch["positions"].dtype)
    acceleration = batch["forces"] / batch["masses"][:, None]
    velocities = batch["velocities"] + dt * acceleration
    positions = batch["positions"] + dt * (velocities if model.config["motion_prior"] == "backward_euler"
                                           else batch["velocities"] + .5 * dt * acceleration)
    positions = torch.where(batch["fixed"][:, None], batch["positions"], positions)
    velocities = torch.where(batch["fixed"][:, None], torch.zeros_like(velocities), velocities)
    return positions.numpy(), velocities.numpy()


@pytest.mark.parametrize("motion_prior", ("constant_acceleration", "backward_euler"))
def test_motion_ablation_shares_candidates_supports_and_fm_calls_but_changes_only_motion(tmp_path, monkeypatch, motion_prior):
    checkpoint, initial, model, config, calls = motion_inputs(tmp_path, monkeypatch, motion_prior)
    checksum = file_hash(checkpoint)
    report = matched.compare_matched_step(initial, model, config, motion_ablation=True)
    assert file_hash(checkpoint) == checksum
    assert report["motion_ablation"] is True
    assert tuple(report["methods"]) == ABLATION_METHODS
    assert report["candidate_count"] == 2
    assert len(calls["forwards"]) == report["predictor_calls"] == 1
    assert len(calls["samples"]) == report["fracture_flow_calls"] == 2
    assert calls["samples"][0]["context"] is calls["samples"][1]["context"]
    assert len(calls["solves"]) == 8
    assert report["requested_method_runs"] == report["attempted_method_runs"] == 10
    assert report["unattempted_method_runs"] == 0
    expected_positions, expected_velocities = expected_physical_motion(initial, model)
    corrected = [row for row in report["candidates"] if row["solver_method"] != "raw"]
    for candidate in range(2):
        records = [row for row in report["candidates"] if row["candidate"] == candidate]
        assert tuple(row["method"] for row in records) == ABLATION_METHODS
        assert len({row["input_hash"] for row in records}) == 1
        assert len({row["candidate_hash"] for row in records}) == 1
        assert len({row["learned_proposal_hash"] for row in records}) == 1
        assert len({row["prior"] for row in records}) == 1
        assert len({row["proposal_hash"] for row in records if row["motion_source"] == "learned"}) == 1
        assert len({row["proposal_hash"] for row in records}) == 2
    assert report["candidates"][0]["prior"] < report["candidates"][5]["prior"]
    assert len({tuple(solve["mask"]) for solve in calls["solves"]}) == 2
    for record, solve in zip(corrected, calls["solves"], strict=True):
        assert record["solver_method"] == solve["method"]
        assert_array_equal(solve["positions"], initial.positions)
        assert_array_equal(solve["velocities"], initial.velocities)
        if record["motion_source"] == "physical_prior":
            assert_allclose(solve["proposal_positions"], expected_positions, rtol=0., atol=2e-8)
            assert_allclose(solve["proposal_velocities"], expected_velocities, rtol=0., atol=2e-8)
        else:
            assert np.max(np.abs(solve["proposal_velocities"] - expected_velocities)) > .1
    for candidate in range(2):
        group = calls["solves"][4 * candidate:4 * candidate + 4]
        for solve in group[1:]:
            assert_array_equal(solve["mask"], group[0]["mask"])
            assert_array_equal(solve["fixed"], group[0]["fixed"])
        assert all(solve["envelope"] is not None for solve in group)
        for solve in group[1:]:
            assert_array_equal(solve["envelope"], group[0]["envelope"])


def test_physical_motion_inputs_do_not_depend_on_learned_residual_outputs(tmp_path, monkeypatch):
    _, initial, model, config, calls = motion_inputs(tmp_path, monkeypatch)
    first = matched.compare_matched_step(initial, model, config, motion_ablation=True)
    first_solves = calls["solves"].copy()
    calls["solves"].clear()
    with torch.no_grad():
        model.motion_head.bias.mul_(-7.)
    second = matched.compare_matched_step(initial, model, config, motion_ablation=True)
    first_corrected = [row for row in first["candidates"] if row["solver_method"] != "raw"]
    second_corrected = [row for row in second["candidates"] if row["solver_method"] != "raw"]
    for first_record, second_record, first_solve, second_solve in zip(first_corrected, second_corrected, first_solves, calls["solves"], strict=True):
        assert first_record["candidate_hash"] == second_record["candidate_hash"]
        assert_array_equal(first_solve["fixed"], second_solve["fixed"])
        if first_record["motion_source"] == "physical_prior":
            assert first_record["proposal_hash"] == second_record["proposal_hash"]
            for key in ("proposal_positions", "proposal_velocities"):
                assert_array_equal(first_solve[key], second_solve[key])
            for key in ("position_increment", "velocity_increment", "unconstrained_position_increment", "unconstrained_velocity_increment"):
                torch.testing.assert_close(first_solve["proposal"][key], second_solve["proposal"][key], rtol=0., atol=0.)
        else:
            assert first_record["proposal_hash"] != second_record["proposal_hash"]


def test_teacher_future_is_evaluated_after_every_motion_arm_and_cannot_change_selection(tmp_path, monkeypatch):
    _, initial, model, config, calls = motion_inputs(tmp_path, monkeypatch)

    def future(offset):
        assert len(calls["solves"]) == 8
        return dict(positions=initial.positions + offset, velocities=initial.velocities + offset,
                    broken=initial.broken.copy(), fixed=initial.fixed.copy())

    first = matched.compare_matched_step(initial, model, config, motion_ablation=True, reference=lambda: future(0.))
    calls["solves"].clear()
    second = matched.compare_matched_step(initial, model, config, motion_ablation=True, reference=lambda: future(10.))
    assert first["candidate_pool"] == second["candidate_pool"]
    for method in ABLATION_METHODS:
        assert first["methods"][method]["selected_candidate"] == second["methods"][method]["selected_candidate"]
        assert first["methods"][method]["evaluation"] != second["methods"][method]["evaluation"]
    assert first["reference_used_for_proposals"] is False
    assert first["reference_used_for_selection"] is False


def test_default_matched_protocol_remains_three_methods_without_motion_ablation(tmp_path, monkeypatch):
    _, initial, model, config, calls = motion_inputs(tmp_path, monkeypatch)
    report = matched.compare_matched_step(initial, model, config)
    assert tuple(report["methods"]) == ("raw", "central", "consensus")
    assert report.get("motion_ablation", False) is False
    assert report["requested_method_runs"] == report["attempted_method_runs"] == 6
    assert len(calls["solves"]) == 4
    assert len(calls["forwards"]) == 1
    assert len(calls["samples"]) == 2


def test_each_motion_arm_reports_one_common_frozen_evaluation_objective(tmp_path, monkeypatch):
    _, initial, model, config, calls = motion_inputs(tmp_path, monkeypatch)
    report = matched.compare_matched_step(initial, model, config, motion_ablation=True)
    corrected = [row for row in report["candidates"] if row["solver_method"] != "raw"]
    assert len(calls["evaluators"]) == report["candidate_count"]
    for record, solve in zip(corrected, calls["solves"], strict=True):
        problem = calls["evaluators"][record["candidate"]]
        state = problem.encode_state(solve["proposal_positions"].reshape(problem.positions.shape),
                                      solve["proposal_velocities"].reshape(problem.velocities.shape))
        assert record["frozen_problem_objective"] == pytest.approx(record["prior"] + problem.objective(state))


def test_motion_ablation_cli_is_opt_in_on_the_matched_protocol_only():
    from fracture_world.cli import parser

    arguments = ["dynamics-matched", "--checkpoint", "best.pt", "--data", "collection.json", "--output", "matched"]
    assert parser().parse_args(arguments).motion_ablation is False
    assert parser().parse_args([*arguments, "--motion-ablation"]).motion_ablation is True
    with pytest.raises(SystemExit):
        parser().parse_args(["dynamics-benchmark", "--checkpoint", "best.pt", "--data", "collection.json",
                             "--output", "benchmark", "--steps", "1", "--motion-ablation"])
