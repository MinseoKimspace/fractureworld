from dataclasses import replace
from types import SimpleNamespace

import numpy as np
from numpy.testing import assert_allclose
import pytest

from fracture_world import dynamics_runtime as runtime
from fracture_world.dynamics_corrector import LinearizedContact
from test_dynamics_phase_runtime import phase_fixture


@pytest.mark.parametrize("method", ["central", "consensus"])
def test_runtime_reuses_current_contact_and_frozen_operators_without_changing_constraints(method):
    initial, model, proposal, config = phase_fixture(penetrated=True)
    config = replace(config, max_contact_iterations=None)
    events = []
    result, report = runtime._correct(initial, initial.positions, initial.velocities, proposal,
                                      initial.broken, initial.fixed, model, config, method, progress=events.append)
    assert result is not None, report
    performance = report["performance"]
    assert performance["reuse_enabled"]
    assert performance["elastic_balance_builds"] == 1
    assert performance["elastic_balance_reuses"] >= 1
    assert performance["frozen_energy_builds"] == 1
    assert performance["inner_iterations_total"] == sum(item["iterations"] for item in report["solves"])
    assert performance["inner_solver_seconds_total"] == sum(item["runtime_seconds"] for item in report["solves"])
    assert all(not item["history"] for item in report["solves"])
    assert result.diagnostics["constraints_satisfied"]
    assert result.diagnostics["energy"]["energy_satisfied"]
    assert report["geometry"]["valid"] and report["swept_geometry"]["valid"]
    assert events[-1]["event"] == "corrector_finished"
    assert events[-1]["status"] == "converged"
    assert config.absolute_tolerance == 1e-8 and config.relative_tolerance == 1e-7


def test_reusable_balance_requires_exact_current_state_and_contacts():
    initial, model, _, config = phase_fixture(penetrated=True)
    contacts = runtime.step_contacts(initial, initial.positions, initial.positions, initial.broken, config)
    args = (initial, initial.positions, initial.velocities, initial.positions, initial.velocities,
            initial.broken, initial.fixed, contacts, model, config)
    first = runtime._problem(*args)
    second = runtime._problem(*args, reusable_balance=first.elastic_balance)
    assert second.elastic_balance is first.elastic_balance
    changed_positions = initial.positions.copy()
    changed_positions[:, 0] += 1e-7
    with pytest.raises(ValueError, match="complete current state"):
        runtime._problem(initial, changed_positions, initial.velocities, initial.positions, initial.velocities,
                         initial.broken, initial.fixed, contacts, model, config, reusable_balance=first.elastic_balance)
    changed_contacts = list(contacts)
    contact = changed_contacts[0]
    changed_contacts[0] = LinearizedContact(contact.corner_ids, contact.weights, contact.normal,
                                          contact.gap + 1e-7, contact.environment_velocity, contact.environment)
    with pytest.raises(ValueError, match="complete current state"):
        runtime._problem(initial, initial.positions, initial.velocities, initial.positions, initial.velocities,
                         initial.broken, initial.fixed, changed_contacts, model, config,
                         reusable_balance=first.elastic_balance)
    initial.material["youngs_modulus"] *= 2
    with pytest.raises(ValueError, match="complete current state"):
        runtime._problem(*args, reusable_balance=first.elastic_balance)


def fake_coupled_retry(monkeypatch, *, progressing_dual=False):
    initial, model, proposal, config = phase_fixture()
    config = replace(config, max_contact_iterations=None)
    calls = dict(stabilization=0, solves=0)

    def stabilization(balance, contacts, velocities, *args):
        index = calls["stabilization"]
        calls["stabilization"] += 1
        offset = np.zeros_like(initial.positions)
        if index:
            offset[:, 0] = .01 if index % 2 else .02
        return offset, dict(status="pass")

    def solve(problem, projection, method, *args, workspace=None, **kwargs):
        calls["solves"] += 1
        if workspace is not None:
            workspace["linear"] = dict(state=np.zeros_like(problem.prior), rho=.1,
                                       scaled_dual=np.array([calls["solves"] if progressing_dual else 0.]))
        return SimpleNamespace(positions=initial.positions.copy(), velocities=initial.velocities.copy(),
                               converged=True, status="converged", iterations=1, runtime_seconds=.01,
                               objective=0., diagnostics={}, history=[])

    monkeypatch.setattr(runtime, "solve_stabilization", stabilization)
    monkeypatch.setattr(runtime, "solve_energy", solve)
    monkeypatch.setattr(runtime, "_diagnostics", lambda *args: dict(constraints_satisfied=False,
                        constraint_violation=.1, constraint_tolerance=1.1e-7, factor_violations={}))
    return initial, model, proposal, config, calls


def test_stabilization_retry_cycle_is_reported_unknown_without_an_iteration_cap(monkeypatch):
    initial, model, proposal, config, calls = fake_coupled_retry(monkeypatch)
    events = []
    result, report = runtime._correct(initial, initial.positions, initial.velocities, proposal,
                                      initial.broken, initial.fixed, model, config, "consensus", progress=events.append)
    assert result is None
    assert report["status"] == "outer_fixed_point_cycle_unknown"
    assert calls["solves"] == 3
    assert report["performance"]["contact_builds"] == 1
    assert report["performance"]["contact_cache_hits"] == 2
    assert report["performance"]["elastic_balance_builds"] == 1
    assert report["performance"]["pending_problem_reuses"] == 2
    retries = [event for event in events if event["event"] == "corrector_outer_retry"]
    assert len(retries) == 3
    assert retries[-1]["complete_outer_state_repeated"]
    assert all(event["reason"] == "stabilization_recheck" for event in retries)
    assert all(event["constraints_satisfied"] is False for event in retries)
    assert retries[-1]["cumulative_iterations"] == 3


def test_outer_cycle_detection_keeps_progressing_duals_distinct(monkeypatch):
    initial, model, proposal, config, calls = fake_coupled_retry(monkeypatch, progressing_dual=True)
    result, report = runtime._correct(initial, initial.positions, initial.velocities, proposal,
                                      initial.broken, initial.fixed, model, config, "consensus",
                                      cancel=lambda: calls["solves"] >= 5)
    assert result is None
    assert report["status"] == "interrupted"
    assert calls["solves"] == 5
    assert not any(item["complete_outer_state_repeated"] for item in report["outer_retries"])


def test_failed_stabilization_recheck_translates_only_the_saved_consensus_state(monkeypatch):
    initial, model, proposal, config, calls = fake_coupled_retry(monkeypatch)
    translations = []
    nonlinear_states = []
    original_solve = runtime.solve_energy

    def solve_with_nonlinear_state(problem, *args, workspace=None, **kwargs):
        result = original_solve(problem, *args, workspace=workspace, **kwargs)
        nonlinear = dict(state=np.full_like(problem.prior, 17.), scaled_dual=np.array([23.]),
                         energy_scaled_dual=np.full_like(problem.prior, 29.), rho=.1)
        workspace["nonlinear"] = nonlinear
        nonlinear_states.append(nonlinear)
        return result

    def translate(workspace, old_problem, new_problem, delta):
        translations.append(delta.copy())
        assert set(workspace) == {"linear"}
        assert workspace["linear"]["rho"] == .1
        assert_allclose(delta, new_problem.stabilization_offset.reshape(-1, 3)
                        - old_problem.stabilization_offset.reshape(-1, 3), rtol=0., atol=0.)
        return dict(translated_states=1, status="translated")

    monkeypatch.setattr(runtime, "solve_energy", solve_with_nonlinear_state)
    monkeypatch.setattr(runtime, "translate_consensus", translate)
    result, report = runtime._correct(initial, initial.positions, initial.velocities, proposal,
                                      initial.broken, initial.fixed, model, config, "consensus")
    assert result is None
    assert report["status"] == "outer_fixed_point_cycle_unknown"
    assert len(translations) == calls["solves"] == 3
    assert all(item["continuation_states_translated"] == 1 for item in report["outer_retries"])
    assert_allclose(initial.positions, proposal["positions"].numpy(), rtol=0., atol=0.)
    for saved in nonlinear_states:
        assert_allclose(saved["state"], 17., rtol=0., atol=0.)
        assert_allclose(saved["scaled_dual"], 23., rtol=0., atol=0.)
        assert_allclose(saved["energy_scaled_dual"], 29., rtol=0., atol=0.)


@pytest.mark.parametrize("energy_balance", [False, True])
def test_changed_contact_geometry_discards_prior_solver_continuation(monkeypatch, energy_balance):
    initial, model, proposal, config = phase_fixture()
    config = replace(config, mechanics_mode="disabled", energy_balance=energy_balance, max_contact_iterations=None)
    calls = dict(contacts=0, solves=0)
    seen = []

    def contacts(*args):
        calls["contacts"] += 1
        return (LinearizedContact.pair(0, -1, [0., 1., 0.], -.01 * calls["contacts"]),)

    def solve(problem, *args, workspace=None, _workspace=None, **kwargs):
        saved = workspace if energy_balance else _workspace
        seen.append(dict(saved))
        saved["marker"] = calls["solves"]
        calls["solves"] += 1
        return SimpleNamespace(positions=initial.positions + [.1, 0., 0.], velocities=initial.velocities.copy(),
                               converged=True, status="converged", iterations=1, runtime_seconds=.01,
                               objective=0., diagnostics={}, history=[])

    monkeypatch.setattr(runtime, "step_contacts", contacts)
    monkeypatch.setattr(runtime, "solve_energy", solve)
    monkeypatch.setattr(runtime, "solve_consensus", solve)
    monkeypatch.setattr(runtime, "geometry_diagnostics", lambda *args: dict(valid=calls["solves"] > 1))
    result, report = runtime._correct(initial, initial.positions, initial.velocities, proposal,
                                      initial.broken, initial.fixed, model, config, "consensus")
    assert result is not None, report
    assert calls["contacts"] == calls["solves"] == 2
    assert seen == [{}, {}]
    assert report["performance"]["contact_geometry_invalidations"] == 1
    assert report["performance"]["solver_workspace_resets"] == 1


def test_disabled_reuse_rebuilds_operators_and_explicit_history_remains_available():
    initial, model, proposal, config = phase_fixture(penetrated=True)
    result, report = runtime._correct(initial, initial.positions, initial.velocities, proposal,
                                      initial.broken, initial.fixed, model, config, "central",
                                      reuse_state=False, record_history=True)
    assert result is not None, report
    performance = report["performance"]
    assert performance["reuse_enabled"] is False
    assert performance["elastic_balance_builds"] > 1
    assert performance["elastic_balance_reuses"] == performance["frozen_energy_reuses"] == 0
    assert_allclose(result.diagnostics["energy"]["normalized_tolerance"], 1.1e-7, rtol=0., atol=1e-22)
