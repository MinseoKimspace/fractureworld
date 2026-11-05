import numpy as np


FLOAT32_UNIT_ROUNDOFF = float(np.finfo(np.float32).eps) / 2
FLOAT32_MINIMUM_SUBNORMAL = float(np.nextafter(np.float32(0), np.float32(1)))
FLOAT32_MAXIMUM = float(np.finfo(np.float32).max)
FLOAT32_OVERFLOW_THRESHOLD = FLOAT32_MAXIMUM + 2.**103


def _positions(value):
    result = np.asarray(value, dtype=float)
    if result.ndim < 1 or result.shape[-1] != 3 or not result.size or not np.isfinite(result).all():
        raise ValueError("positions must be a nonempty finite array ending in three coordinates")
    return result


def _coordinates(value, shape, name):
    try:
        result = np.broadcast_to(np.asarray(value, dtype=float), shape)
    except ValueError as error:
        raise ValueError(f"{name} must broadcast to the position coordinates") from error
    if not np.isfinite(result).all():
        raise ValueError(f"{name} must be finite")
    return result


def _time_step(value):
    if isinstance(value, (bool, np.bool_)) or np.ndim(value) != 0 or not np.isfinite(value) or value <= 0:
        raise ValueError("time step must be a finite positive scalar")
    return float(value)


def _float32(value, name):
    with np.errstate(over="ignore", invalid="ignore"):
        result = np.asarray(value, dtype=np.float32)
    if not np.isfinite(result).all():
        raise ValueError(f"{name} cannot be published as finite float32 values")
    return result


def _upper_product(first, second):
    return np.nextafter(np.multiply(first, second), np.inf)


def _upper_sum(*values):
    result = 0.
    for value in values:
        result = np.nextafter(np.add(result, value), np.inf)
    return result


def _publication_inputs(positions, dt, stabilization):
    positions = _positions(positions)
    dt = _time_step(dt)
    stabilization = _coordinates(0. if stabilization is None else stabilization, positions.shape, "position stabilization")
    native_positions = _float32(positions, "positions")
    native_stabilization = _float32(stabilization, "position stabilization")
    native_dt = _float32(dt, "time step")
    with np.errstate(over="ignore", invalid="ignore"):
        base = np.add(native_positions, native_stabilization, dtype=np.float32)
    if not np.isfinite(base).all():
        raise ValueError("stabilized positions cannot be published as finite float32 values")
    return positions, dt, stabilization, base.astype(float), native_dt


def float32_position_update(positions, dt, velocities, stabilization=None):
    positions, _, _, base, native_dt = _publication_inputs(positions, dt, stabilization)
    velocities = _coordinates(velocities, positions.shape, "velocities")
    native_velocity = _float32(velocities, "velocities")
    with np.errstate(over="ignore", invalid="ignore"):
        displacement = np.multiply(native_dt, native_velocity, dtype=np.float32)
        result = np.add(base.astype(np.float32), displacement, dtype=np.float32)
    if not np.isfinite(result).all():
        raise ValueError("position update cannot be published as finite float32 values")
    return result.astype(float)


def position_publication_envelope(positions, dt, velocity_bound, stabilization=None, *, velocity_is_float32=False):
    positions, dt, stabilization, base, native_dt = _publication_inputs(positions, dt, stabilization)
    bound = _coordinates(velocity_bound, positions.shape, "velocity bound")
    if (bound < 0).any() or (bound > FLOAT32_MAXIMUM).any():
        raise ValueError("velocity bound must be nonnegative and within the finite float32 range")
    if not isinstance(velocity_is_float32, (bool, np.bool_)):
        raise TypeError("velocity_is_float32 must be Boolean")
    unit, half_subnormal = FLOAT32_UNIT_ROUNDOFF, FLOAT32_MINIMUM_SUBNORMAL / 2
    ideal_base = positions + stabilization
    lower, upper = np.nextafter(ideal_base, -np.inf), np.nextafter(ideal_base, np.inf)
    exact_add = (positions == 0) | (stabilization == 0)
    lower, upper = np.where(exact_add, ideal_base, lower), np.where(exact_add, ideal_base, upper)
    base_error = np.nextafter(np.maximum(np.abs(base - lower), np.abs(base - upper)), np.inf)
    velocity_error = np.zeros_like(bound) if velocity_is_float32 else np.where(
        bound == 0, 0., _upper_sum(_upper_product(unit, bound), half_subnormal))
    native_velocity_bound = np.minimum(FLOAT32_MAXIMUM, _upper_sum(bound, velocity_error))
    native_time = float(native_dt)
    product_magnitude = _upper_product(abs(native_time), native_velocity_bound)
    if (product_magnitude >= FLOAT32_OVERFLOW_THRESHOLD).any():
        raise ValueError("velocity bound cannot guarantee a finite float32 displacement")
    product_error = _upper_sum(_upper_product(unit, product_magnitude), half_subnormal)
    native_displacement_bound = np.minimum(FLOAT32_MAXIMUM, _upper_sum(product_magnitude, product_error))
    addition_magnitude = _upper_sum(np.abs(base), native_displacement_bound)
    if (addition_magnitude >= FLOAT32_OVERFLOW_THRESHOLD).any():
        raise ValueError("velocity bound cannot guarantee a finite float32 position update")
    addition_error = _upper_sum(_upper_product(unit, addition_magnitude), half_subnormal)
    time_error = np.nextafter(abs(native_time - dt), np.inf)
    displacement_error = _upper_sum(product_error, _upper_product(abs(native_time), velocity_error),
                                    _upper_product(time_error, bound))
    return _upper_sum(base_error, displacement_error, addition_error)


def position_publication_diagnostics(next_positions, positions, dt, velocities, stabilization=None):
    positions = _positions(positions)
    next_positions = _coordinates(next_positions, positions.shape, "next positions")
    velocities = _coordinates(velocities, positions.shape, "velocities")
    stabilization = _coordinates(0. if stabilization is None else stabilization, positions.shape, "position stabilization")
    dt = _time_step(dt)
    published = float32_position_update(positions, dt, velocities, stabilization)
    envelope = position_publication_envelope(positions, dt, np.abs(velocities), stabilization)
    raw_residual = next_positions - positions - stabilization - dt * velocities
    publication_residual = next_positions - published
    exact = bool(np.array_equal(next_positions, published))
    enclosed = bool(np.all(np.abs(raw_residual) <= envelope))
    return dict(status="pass" if exact else "fail", arithmetic="float32 casts, position stabilization add, multiply, final add",
                rounding_model="IEEE 754 binary32 nearest ties to even, gradual underflow, separate multiply and add",
                exact_publication=exact, within_roundoff_envelope=enclosed,
                maximum_publication_residual_metres=float(np.max(np.abs(publication_residual))),
                maximum_raw_analytical_residual_metres=float(np.max(np.abs(raw_residual))),
                maximum_roundoff_envelope_metres=float(np.max(envelope)),
                raw_analytical_residual_metres=raw_residual.tolist(),
                publication_residual_metres=publication_residual.tolist(), roundoff_envelope_metres=envelope.tolist(),
                physical_contact_law_status="unknown", engine_calls=0, empirical_tolerance_used=False)
