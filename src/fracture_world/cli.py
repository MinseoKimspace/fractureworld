import argparse
import importlib.metadata
import json
from pathlib import Path
import sys


def parser():
    root = argparse.ArgumentParser(prog="fracture-world")
    commands = root.add_subparsers(dest="command", required=True)
    view = commands.add_parser("view", help="Serve a Three.js replay of Python-computed poses")
    view.add_argument("replay", type=Path)
    view.add_argument("--host", default="127.0.0.1")
    view.add_argument("--port", type=int, default=8765)
    view.add_argument("--open", action="store_true")
    commands.add_parser("doctor", help="Check installed libraries and CUDA availability")
    commands.add_parser("sources", help="Print the primary code and data sources")
    native = commands.add_parser("femfx-build", help="Build the pinned Windows CPU FEMFX reference")
    native.add_argument("--source", type=Path)
    native.add_argument("--build", type=Path)
    native.add_argument("--jobs", type=int)
    generation = commands.add_parser("femfx-generate", help="Generate native facet fracture labels and material-corner trajectories")
    generation.add_argument("--mesh", type=Path, required=True)
    generation.add_argument("--specification", type=Path, required=True)
    generation.add_argument("--output", type=Path, required=True)
    generation.add_argument("--length-scale", type=float, required=True, help="Metres per input mesh coordinate unit")
    generation.add_argument("--build", type=Path)
    generation.add_argument("--no-replay", action="store_true", help="Save numerical training data without a large browser replay")
    generation.add_argument("--skip-invalid", action="store_true", help="Record rejected numerical cases and continue; rejected cases are not training references")
    suite = commands.add_parser("dynamics-suite", help="Create explicit native loading sweeps without fabricating fracture or support labels")
    suite.add_argument("--specification", type=Path, required=True)
    suite.add_argument("--output", type=Path, required=True)
    drops = commands.add_parser("drop-suite", help="Prepare solid geometry and explicit free-fall impact-window cases")
    drops.add_argument("--shape", required=True)
    drops.add_argument("--surface", type=Path)
    drops.add_argument("--specification", type=Path, required=True)
    drops.add_argument("--output", type=Path, required=True)
    readiness = commands.add_parser("dynamics-inspect", help="Check actual fracture/release event coverage in disjoint training, validation and test cases")
    readiness.add_argument("--data", type=Path, required=True)
    readiness.add_argument("--seed", type=int, default=17)
    readiness.add_argument("--min-post-event-steps", type=int, default=1)
    readiness.add_argument("--event-stratified", action="store_true")
    readiness.add_argument("--split-unit", choices=["loading_condition", "shape_family"], default="loading_condition")
    readiness.add_argument("--stream-data", action="store_true")
    readiness.add_argument("--output", type=Path)
    dynamics_training = commands.add_parser("dynamics-train", help="Train engine-free motion, irreversible fracture flow and optional support release")
    dynamics_training.add_argument("--data", type=Path, required=True)
    dynamics_training.add_argument("--output", type=Path, required=True)
    dynamics_training.add_argument("--epochs", type=int, required=True)
    dynamics_training.add_argument("--corrector-epochs", type=int, default=0)
    dynamics_training.add_argument("--rollout-steps", type=int, default=1,
                                   help="Autoregressive training window length; only its initial state is observed")
    dynamics_training.add_argument("--hidden", type=int, default=48)
    dynamics_training.add_argument("--heads", type=int, default=3)
    dynamics_training.add_argument("--graph-layers", type=int, default=2)
    dynamics_training.add_argument("--transformer-layers", type=int, default=1)
    dynamics_training.add_argument("--flow-architecture", choices=["coupled", "independent"], default="coupled")
    dynamics_training.add_argument("--flow-layers", type=int, default=2)
    dynamics_training.add_argument("--learning-rate", type=float, default=0.0003)
    dynamics_training.add_argument("--fracture-weight", type=float, default=1.0)
    dynamics_training.add_argument("--flow-weight", type=float, default=1.0)
    dynamics_training.add_argument("--device", default="auto")
    dynamics_training.add_argument("--seed", type=int, default=17)
    dynamics_training.add_argument("--overfit", action="store_true")
    dynamics_training.add_argument("--learn-support-release", action="store_true")
    dynamics_training.add_argument("--require-event-coverage", action="store_true")
    dynamics_training.add_argument("--min-post-event-steps", type=int, default=10)
    dynamics_training.add_argument("--event-stratified", action="store_true")
    dynamics_training.add_argument("--split-unit", choices=["loading_condition", "shape_family"], default="loading_condition")
    dynamics_training.add_argument("--motion-prior", choices=["backward_euler", "constant_acceleration"], default="constant_acceleration")
    dynamics_training.add_argument("--stream-data", action="store_true")
    initial = commands.add_parser("dynamics-initial", help="Export only one current state, without teacher future arrays")
    initial.add_argument("--case", type=Path, required=True)
    initial.add_argument("--trajectory", type=Path, required=True)
    initial.add_argument("--frame", type=int, default=0)
    initial.add_argument("--support-policy", choices=["prescribed", "free", "learned"], required=True)
    initial.add_argument("--output", type=Path, required=True)
    for name in ("dynamics-rollout", "dynamics-compare"):
        command = commands.add_parser(name, help="Run learned fracture dynamics without a runtime physics engine")
        command.add_argument("--checkpoint", type=Path, required=True)
        command.add_argument("--initial", type=Path, required=True)
        command.add_argument("--output", type=Path, required=True)
        command.add_argument("--steps", type=int, required=True)
        command.add_argument("--specification", type=Path)
        command.add_argument("--device", default="auto")
        if name == "dynamics-rollout":
            command.add_argument("--method", choices=["physics", "raw", "neural", "central", "consensus"], default="consensus")
        else:
            command.add_argument("--methods", nargs="+", choices=["physics", "raw", "neural", "central", "consensus"])
            command.add_argument("--reference-case", type=Path)
            command.add_argument("--reference-trajectory", type=Path)
    dynamics_evaluation = commands.add_parser("dynamics-evaluate", help="Evaluate frozen neural rollouts against held-out numerical trajectories")
    dynamics_evaluation.add_argument("--rollout", type=Path, required=True)
    dynamics_evaluation.add_argument("--case", type=Path, required=True)
    dynamics_evaluation.add_argument("--trajectory", type=Path)
    dynamics_evaluation.add_argument("--output", type=Path)
    benchmark = commands.add_parser("dynamics-benchmark", help="Evaluate all held-out cases with separate positive-fracture metrics")
    benchmark.add_argument("--checkpoint", type=Path, required=True)
    benchmark.add_argument("--data", type=Path, required=True)
    benchmark.add_argument("--output", type=Path, required=True)
    benchmark.add_argument("--steps", type=int, required=True)
    benchmark.add_argument("--specification", type=Path)
    benchmark.add_argument("--device", default="auto")
    benchmark.add_argument("--split", choices=["validation", "test"], default="test")
    benchmark.add_argument("--methods", nargs="+", choices=["physics", "raw", "neural", "central", "consensus"])
    return root


def output(value):
    print(json.dumps(value, indent=2, allow_nan=False), flush=True)



def run(args):
    if args.command == "view":
        from .viewer import serve

        serve(args.replay, args.host, args.port, args.open)
    elif args.command == "doctor":
        import torch

        names = ["numpy", "scipy", "torch", "torch-geometric", "torchdiffeq", "clarabel"]
        versions = {}
        for name in names:
            try:
                versions[name] = importlib.metadata.version(name)
            except importlib.metadata.PackageNotFoundError:
                versions[name] = None
        output({"python": sys.version, "libraries": versions, "cuda_available": torch.cuda.is_available(),
                "cuda_device": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None})
    elif args.command == "sources":
        from .sources import SOURCES

        output(SOURCES)
    elif args.command == "femfx-build":
        from .femfx import build_native

        output(build_native(args.source, args.build, args.jobs))
    elif args.command == "femfx-generate":
        from .femfx import generate_dataset

        output(generate_dataset(args.mesh, args.specification, args.output, args.length_scale, args.build, not args.no_replay, args.skip_invalid))
    elif args.command == "dynamics-suite":
        from .dynamics_generator import create_dynamics_suite

        output(create_dynamics_suite(args.output, args.specification))
    elif args.command == "drop-suite":
        from .drop_generator import create_drop_suite

        output(create_drop_suite(args.output, args.specification, args.shape, args.surface))
    elif args.command == "dynamics-inspect":
        from .dynamics_readiness import inspect_dynamics_dataset
        from .io import write_json

        report = inspect_dynamics_dataset(args.data, args.seed, min_post_event_steps=args.min_post_event_steps,
                                          event_stratified=args.event_stratified, split_unit=args.split_unit,
                                          stream_data=args.stream_data)
        if args.output:
            if args.output.exists():
                raise ValueError("inspection output already exists")
            write_json(args.output, report)
        output(report)
    elif args.command == "dynamics-train":
        from .dynamics_learning import DynamicsTrainingConfig, train_dynamics

        config = DynamicsTrainingConfig(**{name: getattr(args, name) for name in DynamicsTrainingConfig.__dataclass_fields__})
        output({"checkpoint": str(train_dynamics(args.data, args.output, config))})
    elif args.command == "dynamics-initial":
        from .dynamics_initial import export_initial

        state = export_initial(args.case, args.trajectory, args.frame, args.output, support_policy=args.support_policy)
        output({"initial": str(args.output.resolve()), "time": state.time, "engine_calls": 0,
                "support_policy": state.metadata["support_policy"], "future_arrays_exported": False})
    elif args.command in ("dynamics-rollout", "dynamics-compare"):
        from .dynamics_runtime import load_rollout_config, rollout_dynamics

        config = load_rollout_config(args.specification)
        progress = lambda event: print(json.dumps(event, allow_nan=False), flush=True)
        if args.command == "dynamics-rollout":
            report = rollout_dynamics(args.checkpoint, args.initial, args.output, args.steps, args.method, config, args.device, progress)
            output({key: value for key, value in report.items() if key != "events"})
        else:
            from .dynamics_experiment import compare_dynamics

            report = compare_dynamics(args.checkpoint, args.initial, args.output, args.steps, config, args.device,
                                      args.methods, args.reference_case, args.reference_trajectory, progress)
            output({"status": report["status"], "all_rollouts_completed": report["all_rollouts_completed"],
                    "engine_calls": 0, "report": str((args.output / "comparison.json").resolve()),
                    "central_consensus_agreement": report["central_consensus_agreement"]})
    elif args.command == "dynamics-evaluate":
        from .dynamics_evaluation import evaluate_dynamics

        output(evaluate_dynamics(args.rollout, args.case, args.trajectory, args.output))
    elif args.command == "dynamics-benchmark":
        from .dynamics_experiment import benchmark_dynamics
        from .dynamics_runtime import load_rollout_config

        progress = lambda event: print(json.dumps(event, allow_nan=False), flush=True)
        report = benchmark_dynamics(args.checkpoint, args.data, args.output, args.steps,
                                    load_rollout_config(args.specification), args.device, args.methods, args.split, progress)
        output(dict(report=str((args.output / "benchmark.json").resolve()), cases=len(report["cases"]),
                    all_rollouts_completed=report["all_rollouts_completed"], conditions=report["conditions"]))



def main():
    arguments = parser().parse_args()
    try:
        run(arguments)
    except (ValueError, RuntimeError, FileNotFoundError) as error:
        print(f"error: {error}", file=sys.stderr)
        raise SystemExit(1) from error
