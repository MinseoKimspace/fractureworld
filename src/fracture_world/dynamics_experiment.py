from pathlib import Path

import numpy as np

from .dynamics_evaluation import evaluate_dynamics
from .dynamics_learning import load_dynamics_model
from .dynamics_runtime import RolloutConfig, rollout_dynamics
from .io import new_directory, write_json


def compare_dynamics(checkpoint, initial, output, steps, config=None, device="auto", methods=None,
                     reference_case=None, reference_trajectory=None, progress=None):
    _, learned_corrector, _, _ = load_dynamics_model(checkpoint, device)
    methods = methods or (["raw", "neural", "central", "consensus"] if learned_corrector is not None
                          else ["raw", "central", "consensus"])
    if len(set(methods)) != len(methods) or not set(methods).issubset({"raw", "neural", "central", "consensus"}):
        raise ValueError("comparison methods must be unique raw/neural/central/consensus values")
    if "neural" in methods and learned_corrector is None:
        raise ValueError("neural baseline requested but checkpoint has no trained corrector")
    if reference_trajectory and not reference_case:
        raise ValueError("reference_trajectory requires reference_case")
    config = config or RolloutConfig()
    directory = new_directory(output)
    conditions = {}
    for method in methods:
        report = rollout_dynamics(checkpoint, initial, directory / method, steps, method, config, device, progress)
        conditions[method] = {key: value for key, value in report.items() if key != "events"}
    if reference_case:
        for method in methods:
            conditions[method]["evaluation"] = evaluate_dynamics(directory / method, reference_case, reference_trajectory,
                                                                 directory / method / "evaluation.json")
    agreement = None
    if {"central", "consensus"}.issubset(conditions):
        with np.load(directory / "central" / "trajectory.npz", allow_pickle=False) as a:
            with np.load(directory / "consensus" / "trajectory.npz", allow_pickle=False) as b:
                count = min(len(a["times"]), len(b["times"]))
                agreement = dict(common_frames=count, positions_rms_metres=float(np.sqrt(np.mean((a["positions"][:count] - b["positions"][:count])**2))),
                                 velocities_rms_metres_per_second=float(np.sqrt(np.mean((a["velocities"][:count] - b["velocities"][:count])**2))),
                                 fracture_disagreements=int(np.count_nonzero(a["broken"][:count] != b["broken"][:count])),
                                 same_objective=True,
                                 interpretation="agreement validates implementations; it is not evidence of proximal superiority")
    report = dict(format="fracture-world.neural-comparison.v1", status="comparison_finished", engine_calls=0,
                  conditions=conditions, central_consensus_agreement=agreement,
                  proposal_control="same predictor, seed and Gaussian samples; autonomous states may diverge after the first step",
                  numerical_reference_used_for_inference=False,
                  neural_baseline_available=learned_corrector is not None,
                  all_rollouts_completed=all(item["status"] == "completed" for item in conditions.values()))
    write_json(directory / "comparison.json", report)
    return report
