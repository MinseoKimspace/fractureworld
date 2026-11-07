from dataclasses import replace
from types import SimpleNamespace

import numpy as np
from numpy.testing import assert_allclose
import pytest

from fracture_world import dynamics_runtime as runtime
from fracture_world.dynamics_corrector import LinearizedContact
from test_dynamics_phase_runtime import phase_fixture


@pytest.mark.parametrize("method", ["central"])
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


def fake_coupled_retry(monkeypatch):
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

    def solve(problem, projection, method, *args, **kwargs):
        calls["solves"] += 1
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
                                      initial.broken, initial.fixed, model, config, "central", progress=events.append)
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
