from dataclasses import replace

import numpy as np
from numpy.testing import assert_allclose
import pytest
import torch

from fracture_world.dynamics_runtime import _correct
from test_dynamics_phase_runtime import phase_fixture


def stored_energy_fixture():
    initial, model, proposal, config = phase_fixture(shared=True)
    positions = initial.positions.copy()
    positions[:, 0] *= 1.2
    initial = replace(initial, positions=positions, dt=1.e-4,
                      material={**initial.material, 'youngs_modulus': 1.e6})
    proposal = {**proposal, 'positions': torch.as_tensor(positions.copy()),
                'velocities': torch.zeros_like(torch.as_tensor(initial.velocities))}
    return initial, model, proposal, replace(config, max_iterations=None, max_contact_iterations=None)


@pytest.mark.parametrize('method', ('central', 'consensus'))
@pytest.mark.parametrize('fixture', ('penetration', 'stored_energy'))
def test_corrector_continuation_preserves_original_precision_and_physical_checks(method, fixture):
    initial, model, proposal, config = (stored_energy_fixture() if fixture == 'stored_energy'
                                        else phase_fixture(penetrated=True))
    settings = (config.absolute_tolerance, config.relative_tolerance, config.geometry_tolerance)
    results = []
    for reuse in (False, True):
        result, report = _correct(initial, initial.positions, initial.velocities, proposal, initial.broken, initial.fixed,
                                  model, config, method, previous_broken=initial.broken,
                                  previous_fixed=initial.fixed, reuse_state=reuse)
        assert result is not None, report
        assert result.converged
        assert result.diagnostics['constraints_satisfied']
        assert result.diagnostics['energy']['energy_satisfied']
        assert report['geometry']['valid']
        assert report['swept_geometry']['valid']
        results.append((result, report))
    assert settings == (config.absolute_tolerance, config.relative_tolerance, config.geometry_tolerance)
    cold, warm = results[0][0], results[1][0]
    assert_allclose(warm.objective, cold.objective, atol=config.absolute_tolerance + config.relative_tolerance, rtol=config.relative_tolerance)
    scale = initial.dt * model.config['velocity_scale']
    normalized_position_error = np.sqrt(np.mean(((warm.positions - cold.positions) / scale)**2))
    normalized_velocity_error = np.sqrt(np.mean(((warm.velocities - cold.velocities) / model.config['velocity_scale'])**2))
    minimum_weight = float(initial.topology.masses.min() / initial.topology.masses.sum())
    objective_accuracy_bound = np.sqrt(2 * (config.absolute_tolerance + config.relative_tolerance) / minimum_weight)
    assert normalized_position_error <= objective_accuracy_bound
    assert normalized_velocity_error <= objective_accuracy_bound
    if fixture == 'stored_energy':
        cold_performance = results[0][1]['performance']
        warm_performance = results[1][1]['performance']
        assert warm_performance['contact_builds'] < cold_performance['contact_builds']
        assert warm_performance['elastic_balance_builds'] < cold_performance['elastic_balance_builds']
        assert warm_performance['frozen_energy_builds'] < cold_performance['frozen_energy_builds']
        if method == 'consensus':
            assert warm_performance['inner_iterations_total'] < cold_performance['inner_iterations_total']


@pytest.mark.parametrize('method', ('central', 'consensus'))
def test_cancelled_reused_corrector_never_publishes_a_successful_state(method):
    initial, model, proposal, config = phase_fixture(penetrated=True)
    result, report = _correct(initial, initial.positions, initial.velocities, proposal, initial.broken, initial.fixed,
                              model, config, method, cancel=lambda: True, reuse_state=True)
    assert result is None
    assert report['status'] == 'interrupted'


def test_matched_cold_control_changes_only_reuse_policy_and_preserves_frozen_inputs(tmp_path, monkeypatch):
    from fracture_world import dynamics_matched as matched
    from fracture_world.dynamics_runtime import RolloutConfig
    from test_dynamics_matched import matched_inputs

    _, initial, model, _ = matched_inputs(tmp_path)
    calls = []

    def capture(*args, **kwargs):
        calls.append((args[8], kwargs.get('reuse_state', 'default')))
        if kwargs.get('progress'):
            kwargs['progress'](dict(event='corrector_started', method=args[8]))
        return None, dict(status='feasibility_unknown', solves=[])

    monkeypatch.setattr(matched.runtime, '_correct', capture)
    config = RolloutConfig(fracture_samples=0, fracture_policy='marginal_search')
    progress = []
    warm = matched.compare_matched_step(initial, model, config, step=19, progress=progress.append)
    warm_calls = calls.copy()
    calls.clear()
    cold = matched.compare_matched_step(initial, model, config, step=19, cold_corrector=True)
    assert all(value == 'default' for _, value in warm_calls)
    assert all(value is False for _, value in calls)
    assert [method for method, _ in calls] == [method for method, _ in warm_calls]
    assert warm['input_hash'] == cold['input_hash']
    assert warm['candidate_pool'] == cold['candidate_pool']
    assert warm['corrector_reuse'] is True
    assert cold['corrector_reuse'] is False
    assert progress
    for event in progress:
        assert event['case'] == initial.case_id
        assert event['input_native_step'] == 19
        assert isinstance(event['candidate'], int)


def selected_case_inputs(tmp_path, monkeypatch):
    from fracture_world import dynamics_matched as matched
    from fracture_world.dynamics_data import load_dynamics_trajectory
    from test_dynamics_data import recorded_trajectory
    from test_dynamics_matched import matched_inputs

    checkpoint, _, model, saved = matched_inputs(tmp_path)
    case, trajectory = recorded_trajectory(tmp_path / 'source')
    item = load_dynamics_trajectory(case, trajectory)
    cases = [replace(item, case_id=f'case-{name}', condition_digest=name * 64) for name in ('a', 'b')]
    saved['training']['overfit'] = False
    saved['splits'] = dict(test=[case.case_id for case in cases])
    saved['condition_splits'] = dict(test=[case.condition_digest for case in cases])
    saved['training_mesh_fingerprints'] = [item.topology.fingerprint]
    monkeypatch.setattr(matched, 'load_dynamics_model', lambda *args: (model, None, saved, torch.device('cpu')))
    monkeypatch.setattr(matched, 'load_dynamics_dataset', lambda *args, **kwargs: cases)
    return checkpoint, cases


def test_matched_case_selection_preserves_saved_split_and_limits_only_requested_observations(tmp_path, monkeypatch):
    from fracture_world import dynamics_matched as matched

    checkpoint, _ = selected_case_inputs(tmp_path, monkeypatch)
    report = matched.compare_matched_dynamics(checkpoint, tmp_path / 'unused', tmp_path / 'selected',
        split='test', frames=[0], case_ids=['case-b'], cancel=lambda: True)
    assert report['available_split_cases'] == 2
    assert report['selected_case_ids'] == ['case-b']
    assert report['requested_cases'] == 1
    assert report['requested_observations'] == 1
    assert report['cases'][0]['case_id'] == 'case-b'


@pytest.mark.parametrize('case_ids', ([], ['case-a', 'case-a'], [''], [None], 'case-a'))
def test_matched_case_selection_rejects_invalid_ids_before_loading_or_writing(tmp_path, case_ids):
    from fracture_world.dynamics_matched import compare_matched_dynamics

    output = tmp_path / 'invalid'
    with pytest.raises(ValueError, match='unique nonempty'):
        compare_matched_dynamics('unused.pt', 'unused', output, case_ids=case_ids)
    assert not output.exists()


def test_matched_case_selection_rejects_unknown_ids_and_missing_saved_cases(tmp_path, monkeypatch):
    from fracture_world import dynamics_matched as matched

    checkpoint, cases = selected_case_inputs(tmp_path, monkeypatch)
    output = tmp_path / 'invalid'
    with pytest.raises(ValueError, match='selected split'):
        matched.compare_matched_dynamics(checkpoint, 'unused', output, case_ids=['unknown'])
    assert not output.exists()
    monkeypatch.setattr(matched, 'load_dynamics_dataset', lambda *args, **kwargs: cases[:1])
    with pytest.raises(ValueError, match='every recorded split'):
        matched.compare_matched_dynamics(checkpoint, 'unused', output, case_ids=['case-a'])
    assert not output.exists()


@pytest.mark.parametrize('cold_corrector', (1, None, 'false'))
def test_matched_cold_control_requires_a_boolean(tmp_path, cold_corrector):
    from fracture_world.dynamics_matched import compare_matched_dynamics

    with pytest.raises(ValueError, match='Boolean'):
        compare_matched_dynamics('unused.pt', 'unused', tmp_path / 'invalid', cold_corrector=cold_corrector)


def test_matched_cli_exposes_cold_control_and_explicit_case_subset():
    from fracture_world.cli import parser

    command = ['dynamics-matched', '--checkpoint', 'best.pt', '--data', 'collection.json', '--output', 'matched']
    defaults = parser().parse_args(command)
    assert defaults.cold_corrector is False and defaults.case_ids is None
    selected = parser().parse_args([*command, '--cold-corrector', '--case-ids', 'bunny-0003', '--frames', '19', '20'])
    assert selected.cold_corrector is True
    assert selected.case_ids == ['bunny-0003']
    assert selected.frames == [19, 20]
