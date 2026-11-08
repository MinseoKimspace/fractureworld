import numpy as np


RMSE_CONVENTION = "Mass-weighted Euclidean point error, excluding initial frame"


def weighted_rmse(error, mass):
    error = np.asarray(error, dtype=np.float64)
    mass = np.asarray(mass, dtype=np.float64)
    if error.ndim != 3 or error.shape[1:] != (len(mass), 2) or len(error) == 0:
        raise ValueError("Error must contain nonempty (frames, nodes, 2) values")
    if not np.isfinite(error).all() or not np.isfinite(mass).all() or np.any(mass <= 0):
        raise ValueError("Metric inputs must be finite with positive nodal masses")
    scale = float(np.max(np.abs(error)))
    if scale == 0:
        return 0.0
    squared = (error / scale) ** 2 * mass[None, :, None]
    return float(scale * np.sqrt(np.sum(squared) / (len(error) * mass.sum())))
