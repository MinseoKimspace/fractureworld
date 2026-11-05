import hashlib

import numpy as np
from scipy import sparse


def _digest(values):
    digest = hashlib.sha256()
    for value in values:
        if isinstance(value, np.ndarray):
            array = np.ascontiguousarray(value)
            digest.update(str((array.shape, array.dtype.str)).encode())
            digest.update(array.tobytes())
        else:
            digest.update(repr(value).encode())
        digest.update(b"\0")
    return digest.digest()


def _coordinate_view(problem):
    if hasattr(problem, "state_indices"):
        return (len(problem.prior), problem.state_indices, problem.offset, problem.reference_positions,
                problem.groups, problem.representatives)
    groups = problem.mesh.corner_groups(problem.broken_faces[problem.mesh.internal_faces])
    _, representatives = np.unique(groups, return_index=True)
    if len(representatives) == problem.corner_count:
        return len(problem.prior), None, None, None, groups, representatives
    reference = problem.positions.reshape(-1, 3)[representatives]
    position_indices = (3 * groups[:, None] + np.arange(3)).ravel()
    state_indices = np.r_[position_indices, 3 * len(representatives) + position_indices]
    offset = np.r_[((reference[groups] - problem.positions.reshape(-1, 3)) / problem.position_scale).ravel(),
                   np.zeros(3 * problem.corner_count)]
    return 6 * len(representatives), state_indices, offset, reference, groups, representatives


def coordinate_key(problem):
    dimension, indices, offset, reference, _, _ = _coordinate_view(problem)
    values = [dimension, problem.position_scale, problem.velocity_scale,
              np.asarray(problem.positions), np.asarray(problem.velocities), indices, offset, reference]
    return _digest(values)


def factor_key(factor):
    matrix = factor.matrix.tocsr()
    return _digest((factor.name, factor.inequality, factor.penalty, factor.indices,
                    matrix.shape, matrix.indptr, matrix.indices, matrix.data))


def consensus_setup(problem, workspace, rho):
    keys = tuple(factor_key(factor) for factor in problem.factors)
    layout = coordinate_key(problem)
    saved = workspace if isinstance(workspace, dict) else {}
    value = saved.get("state")
    compatible = (saved.get("coordinate_key") == layout and isinstance(value, np.ndarray)
                  and value.shape == problem.prior.shape and np.isfinite(value).all())
    state = value.copy() if compatible else problem.prior.copy()
    saved_rho = saved.get("rho")
    if compatible and isinstance(saved_rho, (float, int)) and np.isfinite(saved_rho) and saved_rho > 0:
        rho = float(saved_rho)
    structure = saved.get("operators")
    cached = isinstance(structure, dict) and structure.get("keys") == keys
    if not cached:
        indices = np.concatenate([factor.indices for factor in problem.factors])
        matrix = sparse.block_diag([factor.matrix for factor in problem.factors], format="csr")
        structure = dict(keys=keys, indices=indices, matrix=matrix, transpose=matrix.T.tocsr(),
                         inequality=np.concatenate([np.full(len(factor.bound), factor.inequality, dtype=bool)
                                                    for factor in problem.factors]),
                         soft_rows=np.concatenate([np.full(len(factor.bound), factor.penalty is not None, dtype=bool)
                                                   for factor in problem.factors]))
    dual = np.zeros(len(structure["indices"]))
    reused = 0
    records = saved.get("factor_duals", {}) if compatible else {}
    offset = 0
    for factor, key in zip(problem.factors, keys, strict=True):
        size = len(factor.indices)
        previous = records.get(key)
        if isinstance(previous, np.ndarray) and previous.shape == (size,) and np.isfinite(previous).all():
            dual[offset:offset + size] = previous
            reused += 1
        offset += size
    info = dict(state_reused=bool(compatible), reused_factor_duals=reused,
                reset_factor_duals=len(keys) - reused, operators_reused=bool(cached))
    return state, rho, dual, structure, info


def inverse_gram(problem, operators, rho, build):
    cached = operators.get("inverse_gram")
    if cached is None or cached[0] != rho:
        cached = (rho, build(problem, rho))
        operators["inverse_gram"] = cached
    return cached[1]


def save_consensus(problem, workspace, state, dual, rho, operators):
    if workspace is None:
        return
    records, offset = {}, 0
    for factor, key in zip(problem.factors, operators["keys"], strict=True):
        size = len(factor.indices)
        records[key] = np.array(dual[offset:offset + size], copy=True)
        offset += size
    workspace.update(coordinate_key=coordinate_key(problem), state=np.array(state, copy=True),
                     scaled_dual=np.array(dual, copy=True), rho=float(rho),
                     factor_names=tuple(factor.name for factor in problem.factors), factor_duals=records,
                     operators=operators)


def energy_key(projection):
    energy = getattr(projection.energy, "energy", projection.energy)
    if not hasattr(energy, "stiffness") or not hasattr(energy, "masses"):
        return None
    matrix = energy.stiffness.tocsr()
    return _digest((type(projection.energy).__name__, coordinate_key(projection.problem),
                    projection.energy_scale, matrix.shape, matrix.indptr, matrix.indices, matrix.data,
                    np.asarray(energy.masses)))


def translate_consensus(workspace, old_problem, new_problem, stabilization_delta):
    old_key, new_key = coordinate_key(old_problem), coordinate_key(new_problem)
    if old_key != new_key:
        return dict(translated_states=0, status="incompatible_coordinates")
    delta = np.asarray(stabilization_delta, dtype=float).reshape(-1, 3)
    if delta.shape != np.asarray(new_problem.positions).reshape(-1, 3).shape or not np.isfinite(delta).all():
        raise ValueError("stabilization continuation requires a finite material-corner displacement")
    _, indices, _, _, groups, representatives = _coordinate_view(new_problem)
    if indices is not None:
        nodal = delta[representatives]
        if not np.array_equal(nodal[groups], delta):
            return dict(translated_states=0, status="incompatible_shared_displacements")
    else:
        nodal = delta
    shift = np.r_[nodal.ravel() / new_problem.position_scale, np.zeros(3 * len(nodal))]
    translated = 0
    for name in ("linear", "nonlinear"):
        saved = workspace.get(name)
        if not isinstance(saved, dict) or saved.get("coordinate_key") != old_key:
            continue
        state = saved.get("state")
        if isinstance(state, np.ndarray) and state.shape == shift.shape and np.isfinite(state).all():
            saved["state"] = state + shift
            saved["coordinate_key"] = new_key
            translated += 1
    return dict(translated_states=translated, status="translated", maximum_normalized_shift=float(np.max(np.abs(shift), initial=0.)))
