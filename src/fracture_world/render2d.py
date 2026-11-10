from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.animation import FuncAnimation, PillowWriter
from matplotlib.collections import LineCollection, PolyCollection

from .cut_event import CutSequence, load_trajectory
from .topology import boundary_edges


def trajectory_time(case):
    return case.time if isinstance(case, CutSequence) else case.arrays["time"]


def trajectory_frame(case, frame):
    return case.frame(frame) if isinstance(case, CutSequence) else (case, frame)


def trajectory_hash(case):
    if isinstance(case, CutSequence):
        return case.metadata["experiment_hash"]
    return case.metadata.get("source_case_hash", str(case.arrays["case_hash"].item()))


def render(paths, output, fps=30, frame_stride=1):
    if not paths or fps <= 0 or frame_stride < 1:
        raise ValueError("FPS and frame stride must be positive")
    cases = [load_trajectory(path) for path in paths]
    reference = cases[0]
    reference_hash = trajectory_hash(reference)
    for case in cases[1:]:
        source_hash = trajectory_hash(case)
        if source_hash != reference_hash:
            raise ValueError("Rendering comparison requires identical case inputs")
        count = min(len(trajectory_time(case)), len(trajectory_time(reference)))
        if not np.array_equal(trajectory_time(case)[:count], trajectory_time(reference)[:count]):
            raise ValueError("Rendering comparison requires matching physical timestamps")
    all_positions = np.concatenate([
        segment.arrays["position"].reshape(-1, 2) for case in cases
        for segment in ((case.before, case.after) if isinstance(case, CutSequence) else (case,))
    ])
    lower, upper = all_positions.min(0), all_positions.max(0)
    margin = max(float(np.max(upper - lower)) * 0.06, 1e-6)
    figure, axes = plt.subplots(1, len(cases), figsize=(5 * len(cases), 5), squeeze=False)
    artists = []
    for axis, case in zip(axes[0], cases):
        axis.set_aspect("equal")
        axis.set_xlim(lower[0] - margin, upper[0] + margin)
        axis.set_ylim(lower[1] - margin, upper[1] + margin)
        axis.set_xlabel("x")
        axis.set_ylabel("y")
        polygons = PolyCollection([], facecolor="steelblue", edgecolor="slategray", linewidth=0.35)
        boundary = LineCollection([], colors="darkorange", linewidths=1.6)
        axis.add_collection(polygons)
        axis.add_collection(boundary)
        artists.append((axis, polygons, boundary))

    def update(frame):
        for case, (axis, polygons, boundary) in zip(cases, artists):
            segment, local = trajectory_frame(case, frame)
            position = segment.arrays["position"][local]
            polygons.set_verts(position[segment.mesh.triangles])
            boundary.set_segments(position[boundary_edges(segment.mesh)])
            status = case.metadata.get("status", "completed")
            training = " (untrained)" if case.metadata.get("training_status") == "untrained" else ""
            event = ""
            if isinstance(case, CutSequence):
                event = " | post-event" if trajectory_time(case)[frame] >= case.metadata["event_time"] else " | pre-event"
            axis.set_title(f"{case.metadata['method_label']}{training}\n{status} | t={trajectory_time(case)[frame]:.5f}{event}")
        return []

    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    frames = np.arange(0, min(len(trajectory_time(case)) for case in cases), frame_stride)
    if output.suffix.lower() == ".gif":
        animation = FuncAnimation(figure, update, frames=frames, interval=1000 / fps, blit=False)
        animation.save(output, writer=PillowWriter(fps=fps))
    elif output.suffix.lower() in (".png", ".svg"):
        update(int(frames[-1]))
        figure.savefig(output, dpi=160, bbox_inches="tight")
    else:
        plt.close(figure)
        raise ValueError("Output must be GIF, PNG, or SVG")
    plt.close(figure)
