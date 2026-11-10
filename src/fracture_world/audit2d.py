from time import perf_counter

import numpy as np

from .teacher2d import simulate


REFINEMENT_LIMITS = {
    "position_max_difference": 1e-7,
    "velocity_max_difference": 1e-6,
    "boundary_impulse_max_difference": 1e-7,
}


def maximum_difference(first, second):
    return float(np.max(np.abs(first - second), initial=0))


def audit_teacher(mesh, material, time, amplitude=0.02, ramp_time=0.05,
                  rtol=1e-8, atol=1e-10, max_step=None):
    started = perf_counter()
    time = np.asarray(time, dtype=np.float64)
    if time.ndim != 1 or len(time) < 2 or not np.all(np.diff(time) > 0):
        raise ValueError("Audit timestamps must strictly increase")
    if max_step is None:
        max_step = float(np.min(np.diff(time)))
    arguments = dict(amplitude=amplitude, ramp_time=ramp_time, rtol=rtol, atol=atol, max_step=max_step)
    baseline = simulate(mesh, material, time, **arguments)
    repeated = simulate(mesh, material, time, **arguments)
    refined_time = np.empty(2 * len(time) - 1)
    refined_time[::2] = time
    refined_time[1::2] = time[:-1] + np.diff(time) / 2
    refined = simulate(mesh, material, refined_time, amplitude=amplitude, ramp_time=ramp_time,
                       rtol=rtol / 10, atol=atol / 10, max_step=max_step / 2)
    repeat_noise = {
        f"{key}_max_difference": maximum_difference(baseline.arrays[key], repeated.arrays[key])
        for key in ("position", "velocity", "boundary_impulse")
    }
    refined_impulse = refined.arrays["boundary_impulse"].reshape(
        len(time) - 1, 2, *mesh.rest.shape
    ).sum(axis=1)
    refinement = {
        "position_max_difference": maximum_difference(baseline.arrays["position"], refined.arrays["position"][::2]),
        "velocity_max_difference": maximum_difference(baseline.arrays["velocity"], refined.arrays["velocity"][::2]),
        "boundary_impulse_max_difference": maximum_difference(baseline.arrays["boundary_impulse"], refined_impulse),
    }
    momentum = np.sum(baseline.arrays["velocity"] * baseline.mass[None, :, None], axis=1)
    impulse = np.cumsum(baseline.arrays["boundary_impulse"].sum(axis=1), axis=0)
    balance = maximum_difference(momentum[1:] - momentum[0], impulse)
    thresholds = {
        "repeatability_max_difference": 0.0,
        **REFINEMENT_LIMITS,
        "momentum_balance_max_error": 1e-8,
    }
    checks = {
        "repeatability": all(value <= thresholds["repeatability_max_difference"] for value in repeat_noise.values()),
        "time_refinement": all(value <= thresholds[key] for key, value in refinement.items()),
        "momentum_balance": balance <= thresholds["momentum_balance_max_error"],
    }
    defect = {
        "coarse_max": baseline.metadata["trapezoidal_position_defect_max"],
        "coarse_rms": baseline.metadata["trapezoidal_position_defect_rms"],
        "refined_max": refined.metadata["trapezoidal_position_defect_max"],
        "refined_rms": refined.metadata["trapezoidal_position_defect_rms"],
        "scope": "Sampling and fixed-model integrator mismatch; no pass threshold or correction applied",
    }
    report = {
        "passed": all(checks.values()), "converged": checks["time_refinement"],
        "checks": checks, "thresholds": thresholds, "repeat_noise": repeat_noise,
        "time_refinement": refinement,
        "momentum_balance_max_error": balance,
        "momentum_change_peak": float(np.max(np.abs(momentum - momentum[0]), initial=0)),
        "cumulative_boundary_impulse_peak": float(np.max(np.abs(impulse), initial=0)),
        "trapezoidal_position_defect": defect,
        "baseline_solver": {key: baseline.metadata[key] for key in (
            "solver", "rtol", "atol", "max_step", "solver_evaluations", "force_evaluations"
        )},
        "refined_solver": {key: refined.metadata[key] for key in (
            "solver", "rtol", "atol", "max_step", "solver_evaluations", "force_evaluations"
        )},
        "runtime_seconds": perf_counter() - started,
        "simulation_runtime_seconds": {
            "baseline": baseline.metadata["runtime_seconds"],
            "repeat": repeated.metadata["runtime_seconds"],
            "refined": refined.metadata["runtime_seconds"],
        },
        "cases_simulated": 3, "coarse_frames": len(time), "refined_frames": len(refined_time),
        "units": "dimensionless_trial", "material_calibrated": False,
        "scope": "Numerical teacher repeatability, time convergence and momentum balance; no experimental physical validation, contact or evolving fracture",
    }
    return report, baseline
