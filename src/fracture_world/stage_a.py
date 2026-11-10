import argparse
import json


def main(argv=None):
    parser = argparse.ArgumentParser(prog="python -m fracture_world.stage_a")
    commands = parser.add_subparsers(dest="command", required=True)
    plan = commands.add_parser("plan")
    plan.add_argument("--config", default="configs/stage-a.json")
    generate = commands.add_parser("generate")
    generate.add_argument("--config", default="configs/stage-a.json")
    generate.add_argument("--output", required=True)
    generate.add_argument("--resume", action="store_true")
    validate = commands.add_parser("validate")
    validate.add_argument("--config", default="configs/stage-a.json")
    validate.add_argument("--output", required=True)
    validate.add_argument("--device", default="cpu")
    validate.add_argument("--threads", type=int, default=4)
    equilibrium = commands.add_parser("validate-equilibrium")
    equilibrium.add_argument("--device", default="cpu")
    equilibrium.add_argument("--hidden", type=int, default=16)
    equilibrium.add_argument("--layers", type=int, default=2)
    equilibrium.add_argument("--no-teacher", dest="include_teacher", action="store_false")
    equilibrium.add_argument("--threads", type=int, default=4)
    train = commands.add_parser("train")
    train.add_argument("--config", default="configs/stage-a.json")
    train.add_argument("--profile", choices=("pilot", "a6000-48gb"), default="pilot")
    train.add_argument("--data", required=True)
    train.add_argument("--output", required=True)
    train.add_argument("--method", choices=("direct", "stress"), required=True)
    train.add_argument("--architecture", choices=("legacy", "equilibrium"), default="equilibrium")
    train.add_argument("--epochs", type=int)
    train.add_argument("--batch-size", type=int)
    train.add_argument("--rollout-steps", type=int)
    train.add_argument("--rollout-schedule", type=int, nargs="+")
    train.add_argument("--curriculum-epochs", type=int)
    train.add_argument("--autonomous-validation-cases", type=int, default=8)
    train.add_argument("--force-weight", type=float, default=1.0)
    train.add_argument("--windows-per-epoch", type=int)
    train.add_argument("--device", default="cpu")
    train.add_argument("--seed", type=int)
    train.add_argument("--hidden", type=int)
    train.add_argument("--layers", type=int)
    train.add_argument("--learning-rate", type=float)
    train.add_argument("--patience", type=int)
    train.add_argument("--threads", type=int, default=4)
    benchmark = commands.add_parser("benchmark")
    benchmark.add_argument("--data", required=True)
    benchmark.add_argument("--checkpoints", nargs="+", required=True)
    benchmark.add_argument("--output", required=True)
    benchmark.add_argument("--split", choices=("train", "validation", "test"), default="test")
    benchmark.add_argument("--device", default="cpu")
    benchmark.add_argument("--case-limit", type=int)
    benchmark.add_argument("--threads", type=int, default=4)
    diagnose = commands.add_parser("diagnose")
    diagnose.add_argument("--data", required=True)
    diagnose.add_argument("--report", required=True)
    diagnose.add_argument("--output", required=True)
    diagnose.add_argument("--device", default="cpu")
    diagnose.add_argument("--case-limit", type=int)
    diagnose.add_argument("--threads", type=int, default=4)
    stress = commands.add_parser("stress-evaluate")
    stress.add_argument("--checkpoint", required=True)
    stress.add_argument("--data", required=True)
    stress.add_argument("--output", required=True)
    stress.add_argument("--device", default="cpu")
    stress.add_argument("--source-cases", nargs="+")
    stress.add_argument("--horizon", type=float, default=1.0)
    stress.add_argument("--substeps", type=int, nargs="+", choices=(1, 2, 4), default=[1, 2])
    stress.add_argument("--case-limit", type=int)
    stress.add_argument("--groups", nargs="+", choices=("force", "long", "topology", "load"),
                        default=["force", "long", "topology", "load"])
    stress.add_argument("--render", dest="render_results", action="store_true")
    stress.add_argument("--threads", type=int, default=4)
    event = commands.add_parser("cut-event-evaluate")
    event.add_argument("--checkpoint", required=True)
    event.add_argument("--data", required=True)
    event.add_argument("--output", required=True)
    event.add_argument("--source-cases", nargs="+")
    event.add_argument("--event-time", type=float, default=0.03)
    event.add_argument("--horizon", type=float)
    event.add_argument("--substeps", type=int, choices=(1, 2, 4), default=1)
    event.add_argument("--device", default="cpu")
    event.add_argument("--render", dest="render_results", action="store_true")
    event.add_argument("--allow-untrained", action="store_true")
    event.add_argument("--threads", type=int, default=4)
    sweep = commands.add_parser("cut-event-sweep")
    sweep.add_argument("--data", required=True)
    sweep.add_argument("--checkpoint")
    sweep.add_argument("--output")
    sweep.add_argument("--columns", type=int, nargs="+")
    sweep.add_argument("--load-indices", type=int, nargs="+")
    sweep.add_argument("--event-times", type=float, nargs="+")
    sweep.add_argument("--horizon", type=float)
    sweep.add_argument("--substeps", type=int, choices=(1, 2, 4), default=4)
    sweep.add_argument("--device", default="cpu")
    sweep.add_argument("--plan-only", action="store_true")
    sweep.add_argument("--render-worst", type=int, default=3)
    sweep.add_argument("--allow-untrained", action="store_true")
    sweep.add_argument("--threads", type=int, default=4)
    profile = commands.add_parser("profile")
    profile.add_argument("--case", required=True)
    profile.add_argument("--architecture", choices=("legacy", "equilibrium"), default="equilibrium")
    profile.add_argument("--method", choices=("direct", "stress"), default="stress")
    profile.add_argument("--device", default="cuda")
    profile.add_argument("--batch-sizes", type=int, nargs="+", default=[32, 64, 128, 256])
    profile.add_argument("--hidden", type=int, default=128)
    profile.add_argument("--layers", type=int, default=6)
    profile.add_argument("--rollout-steps", type=int, default=4)
    profile.add_argument("--repeats", type=int, default=2)
    profile.add_argument("--threads", type=int, default=4)
    render = commands.add_parser("render")
    render.add_argument("--trajectories", nargs="+", required=True)
    render.add_argument("--output", required=True)
    render.add_argument("--fps", type=int, default=30)
    render.add_argument("--frame-stride", type=int, default=1)
    external = commands.add_parser("external-compare")
    external.add_argument("--reference", required=True)
    external.add_argument("--trajectory", required=True)
    commands.add_parser("sources")
    args = vars(parser.parse_args(argv))
    command = args.pop("command")
    threads = args.pop("threads", None)
    if threads is not None:
        if threads < 1:
            parser.error("--threads must be positive")
        import torch

        torch.set_num_threads(threads)
    if command in ("plan", "generate", "validate", "train"):
        from .generation2d import configuration

        config = configuration(args.pop("config"))
    if command == "plan":
        from .generation2d import generation_summary

        result = generation_summary(config)
    elif command == "generate":
        from .generation2d import generate

        result = generate(config, **args)
    elif command == "validate":
        from .validation2d import validate

        result = validate(config, **args)
    elif command == "validate-equilibrium":
        from .equilibrium_validation import validate_equilibrium

        result = validate_equilibrium(**args)
    elif command == "train":
        from .training2d import fit

        profile = args.pop("profile")
        settings = dict(config["training"] if profile == "pilot" else config["training_profiles"][profile])
        settings.update({name: value for name, value in args.items() if value is not None})
        result = fit(**settings)
    elif command == "benchmark":
        from .benchmark2d import benchmark

        result = benchmark(**args)
    elif command == "diagnose":
        from .diagnostics2d import diagnose

        result = diagnose(**args)
    elif command == "stress-evaluate":
        from .stress_evaluation import evaluate_stress

        result = evaluate_stress(**args)
    elif command == "cut-event-evaluate":
        from .cut_event_evaluation import evaluate_cut_event

        result = evaluate_cut_event(**args)
    elif command == "cut-event-sweep":
        from .cut_event_sweep import run_sweep

        result = run_sweep(**args)
        if not args["plan_only"]:
            result = {key: result[key] for key in (
                "status", "report", "conditions_csv", "training_executed", "checkpoint_unchanged",
                "requested_cases", "renders", "sweep_runtime_seconds",
            )} | {"summary": result["sweep_summary"]["overall"]}
    elif command == "render":
        from .render2d import render

        render(args.pop("trajectories"), **args)
        result = {"render": args["output"]}
    elif command == "external-compare":
        from .external import compare_external

        result = compare_external(args["reference"], args["trajectory"])
    elif command == "profile":
        from .profile2d import profile_case

        result = profile_case(args.pop("case"), **args)
    else:
        from .attribution import SOURCES

        result = SOURCES
    print(json.dumps(result, indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
