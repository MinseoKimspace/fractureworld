from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.animation import FuncAnimation, PillowWriter
from matplotlib.collections import LineCollection, PolyCollection

from .cases import Case
from .topology import boundary_edges


def render(paths, output, fps=30, frame_stride=1):
    if fps <= 0 or frame_stride < 1:
        raise ValueError("FPS and frame stride must be positive")
    cases = [Case.load(path) for path in paths]
    reference = cases[0]
    reference_hash = reference.metadata.get("source_case_hash", str(reference.arrays["case_hash"].item()))
    for case in cases[1:]:
        source_hash = case.metadata.get("source_case_hash", str(case.arrays["case_hash"].item()))
        if source_hash != reference_hash:
            raise ValueError("Rendering comparison requires identical case inputs")
        count = min(len(case.arrays["time"]), len(reference.arrays["time"]))
        if not np.array_equal(case.arrays["time"][:count], reference.arrays["time"][:count]):
            raise ValueError("Rendering comparison requires matching physical timestamps")
    all_positions = np.concatenate([case.arrays["position"].reshape(-1, 2) for case in cases])
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
        artists.append((axis, polygons, boundary, boundary_edges(case.mesh)))

    def update(frame):
        for case, (axis, polygons, boundary, edges) in zip(cases, artists):
            position = case.arrays["position"][frame]
            polygons.set_verts(position[case.mesh.triangles])
            boundary.set_segments(position[edges])
            status = case.metadata.get("status", "completed")
            training = " (untrained)" if case.metadata.get("training_status") == "untrained" else ""
            axis.set_title(f"{case.metadata['method_label']}{training} | {status} | t={case.arrays['time'][frame]:.5f}")
        return []

    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    frames = np.arange(0, min(len(case.arrays["time"]) for case in cases), frame_stride)
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
