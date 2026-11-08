import json
from pathlib import Path
from urllib.parse import urlparse

import numpy as np

from .metrics2d import RMSE_CONVENTION, weighted_rmse
from .topology import Mesh, lumped_mass


ARRAY_FIELDS = (
    "rest",
    "triangles",
    "original_vertex",
    "time",
    "position",
    "velocity",
    "boundary_position",
    "boundary_velocity",
    "boundary_mask",
)


def _text(value, name):
    array = np.asarray(value)
    if array.shape != () or array.dtype.kind not in "US":
        raise ValueError(f"{name} must be a scalar string")
    result = str(array.item())
    if not result.strip():
        raise ValueError(f"{name} must not be empty")
    return result


def _load_trajectory(path):
    with np.load(Path(path), allow_pickle=False) as archive:
        required = {*ARRAY_FIELDS, "case_hash", "metadata_json"}
        missing = required.difference(archive.files)
        if missing:
            raise ValueError(f"Missing trajectory fields: {', '.join(sorted(missing))}")
        arrays = {name: archive[name] for name in ARRAY_FIELDS}
        case_hash = _text(archive["case_hash"], "case_hash")
        try:
            metadata = json.loads(_text(archive["metadata_json"], "metadata_json"))
        except json.JSONDecodeError as error:
            raise ValueError("metadata_json must contain valid JSON") from error
    if not isinstance(metadata, dict):
        raise ValueError("metadata_json must contain an object")
    material = metadata.get("material")
    if not isinstance(material, dict) or not material:
        raise ValueError("Trajectory metadata must include the material configuration")
    try:
        json.dumps(material, sort_keys=True, allow_nan=False)
    except (TypeError, ValueError) as error:
        raise ValueError("Material configuration must contain finite JSON values") from error
    _validate_arrays(arrays)
    return arrays, case_hash, metadata


def _validate_arrays(arrays):
    rest, triangles, times = (arrays[name] for name in ("rest", "triangles", "time"))
    if rest.ndim != 2 or rest.shape[1] != 2 or not len(rest):
        raise ValueError("Rest coordinates must have shape (nodes, 2)")
    if triangles.ndim != 2 or triangles.shape[1] != 3 or not len(triangles):
        raise ValueError("Triangles must have shape (elements, 3)")
    if triangles.dtype.kind not in "iu" or np.any(triangles < 0) or np.any(triangles >= len(rest)):
        raise ValueError("Triangle indices must be valid integers")
    original = arrays["original_vertex"]
    if original.shape != (len(rest),) or original.dtype.kind not in "iu" or np.any(original < 0):
        raise ValueError("Original vertex IDs must be nonnegative integers for every node")
    if times.ndim != 1 or len(times) < 2 or not np.all(np.diff(times) > 0):
        raise ValueError("Time must contain at least two strictly increasing timestamps")
    expected = (len(times), len(rest), 2)
    for name in ("position", "velocity", "boundary_position", "boundary_velocity"):
        if arrays[name].shape != expected:
            raise ValueError(f"{name} must have shape {expected}")
    mask = arrays["boundary_mask"]
    if mask.shape != (len(rest), 2) or mask.dtype.kind != "b":
        raise ValueError("boundary_mask must be boolean with shape (nodes, 2)")
    for name, array in arrays.items():
        if array.dtype.kind not in "biuf" or not np.isfinite(array).all():
            raise ValueError(f"{name} must contain finite numeric values")
    points = rest[triangles]
    first, second = points[:, 1] - points[:, 0], points[:, 2] - points[:, 0]
    if np.any(first[:, 0] * second[:, 1] - first[:, 1] * second[:, 0] <= 0):
        raise ValueError("Reference triangles must have positive orientation")


def _provenance(metadata):
    label = metadata.get("method_label")
    if not isinstance(label, str) or not label.strip():
        raise ValueError("External metadata requires method_label")
    normalized = label.strip().lower().replace("-", "_").replace(" ", "_")
    if normalized in {"teacher", "reference", "ground_truth", "truth", "felupe", "femfx"}:
        raise ValueError("External method_label must identify the evaluated method")
    url = metadata.get("source_url")
    if not isinstance(url, str):
        raise ValueError("External metadata requires source_url")
    parsed = urlparse(url)
    if parsed.scheme not in {"http", "https"} or not parsed.netloc:
        raise ValueError("source_url must be an HTTP or HTTPS source URL")
    revision = metadata.get("source_revision")
    if not isinstance(revision, str) or not revision.strip():
        raise ValueError("External metadata requires source_revision")
    if revision.strip().lower() in {"main", "master", "head", "latest", "unknown"}:
        raise ValueError("source_revision must identify a fixed revision or release")
    runtime = metadata.get("runtime_seconds")
    if isinstance(runtime, bool) or not isinstance(runtime, (int, float)):
        raise ValueError("External metadata requires numeric runtime_seconds")
    if not np.isfinite(runtime) or runtime <= 0:
        raise ValueError("runtime_seconds must be positive and finite")
    return {
        "method_label": label,
        "source_url": url,
        "source_revision": revision,
        "runtime_seconds": float(runtime),
    }


def compare_external(reference_path, external_path):
    reference, reference_hash, reference_metadata = _load_trajectory(reference_path)
    external, external_hash, external_metadata = _load_trajectory(external_path)
    provenance = _provenance(external_metadata)
    if reference_hash != external_hash:
        raise ValueError("External trajectory has a different case_hash")
    if json.dumps(reference_metadata["material"], sort_keys=True) != json.dumps(
        external_metadata["material"], sort_keys=True
    ):
        raise ValueError("External trajectory has a different material configuration")
    for name in (
        "rest",
        "triangles",
        "original_vertex",
        "time",
        "boundary_position",
        "boundary_velocity",
        "boundary_mask",
    ):
        if not np.array_equal(reference[name], external[name]):
            raise ValueError(f"External trajectory has incompatible {name}")
    for name in ("position", "velocity"):
        if not np.array_equal(reference[name][0], external[name][0]):
            raise ValueError(f"External trajectory has incompatible initial {name}")
    material = reference_metadata["material"]
    if "density" not in material or "thickness" not in material:
        raise ValueError("Material requires density and thickness for mass-weighted comparison")
    mesh = Mesh(reference["rest"], reference["triangles"], reference["original_vertex"])
    mass = lumped_mass(mesh, material["density"], material["thickness"])
    position_error = external["position"].astype(np.float64)[1:] - reference["position"][1:]
    velocity_error = external["velocity"].astype(np.float64)[1:] - reference["velocity"][1:]
    position_squared = np.sum(position_error ** 2, axis=-1)
    velocity_squared = np.sum(velocity_error ** 2, axis=-1)
    duration = float(reference["time"][-1] - reference["time"][0])
    return {
        "status": "compatible",
        "case_hash": reference_hash,
        "frames": len(reference["time"]),
        "start_time": float(reference["time"][0]),
        "end_time": float(reference["time"][-1]),
        "completed_physical_horizon": duration,
        "position_rmse": weighted_rmse(position_error, mass),
        "velocity_rmse": weighted_rmse(velocity_error, mass),
        "final_position_rmse": weighted_rmse(position_error[-1:], mass),
        "final_velocity_rmse": weighted_rmse(velocity_error[-1:], mass),
        "rmse_convention": RMSE_CONVENTION,
        "maximum_position_error": float(np.sqrt(np.max(position_squared))),
        "maximum_velocity_error": float(np.sqrt(np.max(velocity_squared))),
        "seconds_per_step": provenance["runtime_seconds"] / (len(reference["time"]) - 1),
        "provenance": provenance,
    }
