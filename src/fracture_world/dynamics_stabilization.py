import numpy as np
from scipy import sparse

from .dynamics_corrector import LinearizedContact


def contact_jacobian(balance, contacts):
    rows, columns, values = [], [], []
    for row, contact in enumerate(contacts):
        if not isinstance(contact, LinearizedContact) or (contact.corner_ids >= len(balance.groups)).any():
            raise ValueError("stabilization contacts must reference valid material corners")
        for corner, weight in zip(contact.corner_ids, contact.weights, strict=True):
            rows.extend([row] * 3)
            columns.extend((3 * balance.groups[corner] + np.arange(3)).tolist())
            values.extend((weight * contact.normal).tolist())
    return sparse.csr_matrix((values, (rows, columns)), shape=(len(contacts), len(balance.rhs)))


def solve_stabilization(balance, contacts, velocities, geometry_tolerance, position_scale, tolerances, cancel=None):
    contacts = tuple(contacts)
    velocity = np.asarray(velocities, dtype=float).reshape(-1, 3)
    if velocity.shape != balance.positions.shape or not np.isfinite(velocity).all():
        raise ValueError("stabilization velocities must match finite material corners")
    if not np.isfinite([geometry_tolerance, position_scale]).all() or geometry_tolerance < 0 or position_scale <= 0:
        raise ValueError("stabilization requires nonnegative geometry tolerance and positive position scale")
    zero = np.zeros_like(velocity)
    report = dict(status="unknown", law="converged minimum frozen-A-distance contact position stabilization target",
                  engine_calls=0, teacher_future_used=False, contact_constraints=len(contacts),
                  geometry_tolerance_metres=float(geometry_tolerance),
                  position_update="x_next = x_old + stabilization + dt * final_velocity")
    if cancel is not None and cancel():
        return None, {**report, "status": "interrupted"}
    if not contacts:
        return zero, {**report, "status": "pass", "objective": 0., "maximum_normalized_violation": 0.}
    jacobian = contact_jacobian(balance, contacts)
    nodal_velocity = velocity[balance.representatives].ravel()
    gap = np.array([contact.gap for contact in contacts])
    environment = np.array([contact.environment_velocity for contact in contacts])
    endpoint_gap = gap + balance.dt * (jacobian @ nodal_velocity - environment)
    lower = (-geometry_tolerance - endpoint_gap) / position_scale
    if np.max(lower, initial=0) <= 0:
        return zero, {**report, "status": "pass", "objective": 0., "maximum_normalized_violation": 0.}
    import clarabel

    dimension = len(balance.rhs)
    mass = float(balance.node_masses.sum())
    metric = .5 * (balance.matrix + balance.matrix.T) / mass
    fixed_columns = (3 * balance.support_nodes[:, None] + np.arange(3)).ravel()
    fixed = sparse.csc_matrix((np.ones(len(fixed_columns)), (np.arange(len(fixed_columns)), fixed_columns)),
                              shape=(len(fixed_columns), dimension))
    matrix = sparse.vstack((fixed, -jacobian), format="csc")
    bound = np.r_[np.zeros(len(fixed_columns)), -lower]
    cones = ([clarabel.ZeroConeT(len(fixed_columns))] if len(fixed_columns) else []) + [clarabel.NonnegativeConeT(len(contacts))]
    settings = clarabel.DefaultSettings()
    settings.verbose = False
    settings.max_iter = np.iinfo(np.int32).max
    settings.tol_feas = tolerances.absolute
    settings.tol_gap_abs = tolerances.absolute
    settings.tol_gap_rel = max(tolerances.relative, np.finfo(float).eps)
    solver = clarabel.DefaultSolver(sparse.triu(metric, format="csc"), np.zeros(dimension), matrix, bound, cones, settings)
    if cancel is not None:
        solver.set_termination_callback(lambda info: bool(cancel()))
    solution = solver.solve()
    if str(solution.status) not in {"Solved", "AlmostSolved"}:
        return None, {**report, "status": "interrupted" if str(solution.status) == "CallbackTerminated" else "unknown",
                      "solver_status": str(solution.status)}
    displacement = np.asarray(solution.x)
    dual = np.asarray(solution.z)
    if displacement.shape != (dimension,) or dual.shape != bound.shape or not np.isfinite(displacement).all() or not np.isfinite(dual).all():
        return None, {**report, "reason": "nonfinite or malformed contact position solution"}
    tolerance = tolerances.absolute + tolerances.relative * max(float(np.max(np.abs(displacement), initial=0)), 1.)
    violation = max(float(np.max(lower - jacobian @ displacement, initial=0)),
                    float(np.max(np.abs(fixed @ displacement), initial=0)))
    stationarity = float(np.max(np.abs(metric @ displacement + matrix.T @ dual), initial=0))
    impulses = dual[len(fixed_columns):]
    complementarity = float(np.max(np.abs(impulses * (jacobian @ displacement - lower)), initial=0))
    valid = (violation <= tolerance and stationarity <= tolerance and complementarity <= tolerance and
             np.min(impulses, initial=0) >= -tolerance)
    report.update(status="pass" if valid else "unknown", solver_status=str(solution.status),
                  objective=float(.5 * displacement @ (metric @ displacement)),
                  maximum_normalized_violation=violation, normalized_tolerance=tolerance,
                  stationarity_residual=stationarity, complementarity_residual=complementarity,
                  maximum_stabilization_metres=float(np.max(np.abs(displacement), initial=0) * position_scale),
                  kinetic_state_changed=False)
    return (position_scale * displacement.reshape(-1, 3))[balance.groups] if valid else None, report
