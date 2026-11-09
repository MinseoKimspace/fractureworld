import numpy as np
import pytest

from fracture_world.equilibrium_validation import (
    oscillator_probe,
    teacher_connectivity_probe,
    teacher_integration_probe,
    validate_equilibrium,
)


def test_same_state_cut_changes_acceleration_without_changing_material_state():
    pytest.importorskip("felupe")
    result = teacher_connectivity_probe()
    assert result["same_material_corner_position_max_error"] == 0
    assert result["same_material_corner_velocity_max_error"] == 0
    assert result["mass_transfer_abs_error"] < 1e-14
    assert result["momentum_transfer_error_norm"] < 1e-14
    assert result["aggregated_force_max_error"] < 1e-12
    assert result["corner_acceleration_response_rms"] > 1
    assert result["unmoved_component_force_change_max"] < 1e-12
    assert result["uncut"]["component_count"] == 1
    assert result["cut"]["component_count"] == 2
    assert result["cut"]["max_net_force_norm"] < 1e-12
    assert result["cut"]["max_net_torque_abs"] < 1e-12


def test_verlet_does_not_exhibit_legacy_oscillator_amplification():
    result = oscillator_probe()
    old = result["methods"]["legacy_constant_acceleration"]
    new = result["methods"]["velocity_verlet"]
    assert old["final_energy_ratio"] > 6
    assert new["min_energy_ratio"] >= 0.99 - 1e-12
    assert new["max_energy_ratio"] <= 1 + 1e-12


@pytest.mark.parametrize("nx,ny,column,depth", [(2, 2, 1, 1), (8, 8, 4, 7), (8, 8, 4, 8)])
def test_teacher_force_verlet_agrees_with_independent_reference_and_refines(nx, ny, column, depth):
    pytest.importorskip("felupe")
    result = teacher_integration_probe(nx=nx, ny=ny, column=column, depth=depth)
    assert result["mesh"] == {"nx": nx, "ny": ny, "cut_column": column, "cut_depth": depth}
    assert result["teacher_refinement"]["position_max_difference"] < 1e-9
    assert result["teacher_refinement"]["velocity_max_difference"] < 1e-8
    methods = result["methods"]
    old = methods["legacy_constant_acceleration"]
    new = methods["velocity_verlet"]
    refined = methods["velocity_verlet_half_dt"]
    for key in ("free_position_rmse", "free_velocity_rmse"):
        assert np.isfinite(old[key]) and np.isfinite(new[key]) and np.isfinite(refined[key])
        assert new[key] < old[key]
        assert refined[key] < new[key] / 2.5


def test_structural_validation_uses_nonzero_untrained_heads_and_writes_nothing(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    result = validate_equilibrium(hidden=8, layers=1, include_teacher=False)
    assert result["training_executed"] is False
    assert list(tmp_path.iterdir()) == []
    for method, probe in result["neural_structure"].items():
        assert probe["deformed_force_max_abs"] > 1e-8
        assert probe["rest_force_max_abs"] < 1e-12
        assert probe["rigid_motion_force_max_abs"] < 1e-10
        assert probe["rotation_covariance_max_error"] < 1e-10
        assert probe["independent_component_rotation_max_error"] < 1e-10
        assert probe["independent_component_rigid_force_max_abs"] < 1e-10
        assert probe["modified_component_force_change_max"] > 1e-8
        assert probe["unmodified_component_force_change_max"] < 1e-12
        assert probe["same_state_cut_acceleration_response_rms"] > 1e-8
        assert "coordinate-frame covariance is not claimed" in probe["rotation_scope"]
        if method == "stress":
            assert probe["component_balance"]["max_net_force_norm"] < 1e-12
            assert probe["component_balance"]["max_net_torque_abs"] < 1e-12
