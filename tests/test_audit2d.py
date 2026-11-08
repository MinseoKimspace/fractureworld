import numpy as np

from fracture_world.audit2d import audit_teacher
from fracture_world.teacher2d import Material
from fracture_world.topology import rectangle, split_edges


def test_teacher_audit_checks_convergence_and_nonzero_momentum():
    mesh = split_edges(rectangle(3, 2), [[1, 5]])
    report, case = audit_teacher(mesh, Material(young=10), np.linspace(0, 0.04, 9),
                                amplitude=0.005, ramp_time=0.02, max_step=0.0025)
    assert report["passed"]
    assert report["converged"]
    assert report["momentum_change_peak"] > 1e-6
    assert report["momentum_balance_max_error"] < 1e-8
    assert report["cases_simulated"] == 3
    assert report["coarse_frames"] == 9
    assert report["refined_frames"] == 17
    assert all(value == 0 for value in report["repeat_noise"].values())
    assert report["trapezoidal_position_defect"]["coarse_max"] > report["trapezoidal_position_defect"]["refined_max"]
    assert report["trapezoidal_position_defect"]["coarse_max"] > 0
    assert report["thresholds"]["position_max_difference"] == 1e-7
    assert report["thresholds"]["velocity_max_difference"] == 1e-6
    assert report["thresholds"]["boundary_impulse_max_difference"] == 1e-7
    assert report["thresholds"]["momentum_balance_max_error"] == 1e-8
    assert report["refined_solver"]["rtol"] == report["baseline_solver"]["rtol"] / 10
    assert report["refined_solver"]["atol"] == report["baseline_solver"]["atol"] / 10
    assert report["refined_solver"]["max_step"] == report["baseline_solver"]["max_step"] / 2
    assert case.metadata["method_label"] == "teacher"
