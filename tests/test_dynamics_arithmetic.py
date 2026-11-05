from fractions import Fraction

import numpy as np
import pytest
from numpy.testing import assert_array_equal

from fracture_world.dynamics_arithmetic import (
    FLOAT32_MAXIMUM, FLOAT32_MINIMUM_SUBNORMAL, float32_position_update,
    position_publication_diagnostics, position_publication_envelope,
)


def assert_exact_real_error_enclosed(positions, dt, velocities, stabilization, envelope):
    published = float32_position_update(positions, dt, velocities, stabilization)
    stabilization = np.broadcast_to(stabilization, positions.shape)
    for old, speed, shift, new, bound in zip(positions.flat, velocities.flat, stabilization.flat, published.flat, envelope.flat):
        error = abs(Fraction(float(new)) - Fraction(float(old)) - Fraction(float(shift)) - Fraction(float(dt)) * Fraction(float(speed)))
        assert error <= Fraction(float(bound))


def test_publication_preserves_separate_native_operation_order():
    positions = np.array([[1e8, 1., -1e8]])
    stabilization = np.array([[-1e8, 1e-8, 1e8]])
    velocities = np.ones_like(positions)
    published = float32_position_update(positions, 1., velocities, stabilization)
    assert_array_equal(published, [[1., 2., 1.]])
    reassociated = (positions.astype(np.float32) +
                    (stabilization.astype(np.float32) + velocities.astype(np.float32))).astype(float)
    assert not np.array_equal(published, reassociated)


@pytest.mark.parametrize("positions,dt,velocities,stabilization", (
    ([[.1, -.3, 10.]], .001, [[.17, -3.41, 0.]], [[0., 0., 0.]]),
    ([[1e8, 1., -1e8]], .3, [[.10000000001, -1e-8, -.777777777]], [[-1e8, 1e-8, 1e8]]),
    ([[1e-320, -1e-320, 0.]], 1e-45, [[1e20, -1e20, 1e-8]], [[0., 0., 0.]]),
    ([[FLOAT32_MAXIMUM, -FLOAT32_MAXIMUM, 0.]], 1., [[0., 0., FLOAT32_MAXIMUM]], [[0., 0., 0.]]),
    ([[1e-30, -1e-30, 1e-30]], 1e-15, [[1e-30, -1e-30, 1e-15]], [[1e-40, -1e-40, -1e-30]]),
))
def test_envelope_bounds_exact_real_error_including_all_input_casts(positions, dt, velocities, stabilization):
    positions, velocities, stabilization = (np.array(value, dtype=float) for value in (positions, velocities, stabilization))
    envelope = position_publication_envelope(positions, dt, np.abs(velocities), stabilization)
    assert np.isfinite(envelope).all()
    assert (envelope >= 0).all()
    assert_exact_real_error_enclosed(positions, dt, velocities, stabilization, envelope)


@pytest.mark.parametrize("native_velocity", (False, True))
def test_subnormal_tie_to_even_is_covered_without_relative_error_assumptions(native_velocity):
    positions = np.zeros((1, 3))
    velocities = np.full((1, 3), FLOAT32_MINIMUM_SUBNORMAL)
    published = float32_position_update(positions, .5, velocities)
    assert_array_equal(published, np.zeros_like(positions))
    envelope = position_publication_envelope(positions, .5, np.abs(velocities), velocity_is_float32=native_velocity)
    assert (envelope >= FLOAT32_MINIMUM_SUBNORMAL / 2).all()
    assert_exact_real_error_enclosed(positions, .5, velocities, 0., envelope)


def test_frozen_bound_covers_all_continuous_velocities_within_it():
    generator = np.random.default_rng(87331)
    positions = generator.normal(size=(50, 3)) * np.exp2(generator.integers(-130, 110, size=(50, 3)))
    stabilization = -positions + generator.normal(size=(50, 3)) * .01
    velocity_bound = np.exp2(generator.integers(-130, 110, size=(50, 3))).astype(float)
    envelope = position_publication_envelope(positions, .003, velocity_bound, stabilization)
    for _ in range(5):
        velocities = generator.uniform(-1., 1., size=positions.shape) * velocity_bound
        assert_exact_real_error_enclosed(positions, .003, velocities, stabilization, envelope)


def test_envelope_is_monotone_in_velocity_bound_and_native_velocity_information():
    positions = np.array([[.05, -.2, 1.2]])
    smaller = position_publication_envelope(positions, .001, 1.)
    larger = position_publication_envelope(positions, .001, 10.)
    native = position_publication_envelope(positions, .001, 10., velocity_is_float32=True)
    assert (larger >= smaller).all()
    assert (larger >= native).all()


def test_diagnostics_retains_raw_error_while_certifying_exact_publication():
    positions = np.array([[1., .5, -.2]])
    velocities = np.array([[.1, -.3, .7]])
    stabilization = np.array([[.0001, 0., -.00001]])
    published = float32_position_update(positions, .001, velocities, stabilization)
    report = position_publication_diagnostics(published, positions, .001, velocities, stabilization)
    assert report["status"] == "pass"
    assert report["exact_publication"] is True
    assert report["within_roundoff_envelope"] is True
    assert report["maximum_publication_residual_metres"] == 0.
    assert report["maximum_raw_analytical_residual_metres"] > 0.
    assert report["engine_calls"] == 0
    assert report["empirical_tolerance_used"] is False
    assert report["physical_contact_law_status"] == "unknown"


def test_exact_publication_and_roundoff_enclosure_are_distinct_checks():
    positions = np.array([[1., .5, -.2]])
    velocities = np.array([[.1, -.3, .7]])
    published = float32_position_update(positions, .001, velocities)
    changed = published.copy()
    changed[0, 0] = np.nextafter(changed[0, 0], np.inf)
    report = position_publication_diagnostics(changed, positions, .001, velocities)
    assert report["status"] == "fail"
    assert report["exact_publication"] is False
    assert report["within_roundoff_envelope"] is True
    changed[0, 0] += 1e-3
    report = position_publication_diagnostics(changed, positions, .001, velocities)
    assert report["within_roundoff_envelope"] is False


@pytest.mark.parametrize("dt", (0., -1., np.inf, np.nan, True))
def test_nonphysical_time_steps_are_rejected(dt):
    with pytest.raises(ValueError, match="time step"):
        float32_position_update(np.ones((1, 3)), dt, np.zeros((1, 3)))


@pytest.mark.parametrize("bound", (-1., np.inf, np.nan, FLOAT32_MAXIMUM * 2))
def test_invalid_velocity_bounds_are_rejected(bound):
    with pytest.raises(ValueError, match="velocity bound"):
        position_publication_envelope(np.zeros((1, 3)), .01, bound)


def test_bound_must_guarantee_finite_products_and_sums():
    with pytest.raises(ValueError, match="displacement"):
        position_publication_envelope(np.zeros((1, 3)), 2., FLOAT32_MAXIMUM)
    with pytest.raises(ValueError, match="position update"):
        position_publication_envelope(np.full((1, 3), FLOAT32_MAXIMUM), 1., FLOAT32_MAXIMUM)
    with pytest.raises(ValueError, match="finite float32"):
        float32_position_update(np.full((1, 3), FLOAT32_MAXIMUM), 1., np.full((1, 3), FLOAT32_MAXIMUM))


def test_publication_functions_leave_current_state_unchanged():
    positions = np.array([[.1, -.2, .3]])
    velocities = np.array([[1., -2., 3.]])
    stabilization = np.array([[.001, -.002, .003]])
    originals = [value.copy() for value in (positions, velocities, stabilization)]
    published = float32_position_update(positions, .001, velocities, stabilization)
    position_publication_envelope(positions, .001, np.abs(velocities), stabilization)
    position_publication_diagnostics(published, positions, .001, velocities, stabilization)
    for value, original in zip((positions, velocities, stabilization), originals):
        assert_array_equal(value, original)
