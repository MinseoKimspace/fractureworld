import argparse
import json


def main(argv=None):
    parser = argparse.ArgumentParser(prog="python -m fracture_world.stage_a")
    commands = parser.add_subparsers(dest="command", required=True)
    generate = commands.add_parser("generate")
    generate.add_argument("--config", default="configs/stage-a.json")
    generate.add_argument("--output", required=True)
    validate = commands.add_parser("validate")
    validate.add_argument("--config", default="configs/stage-a.json")
    validate.add_argument("--output", required=True)
    validate.add_argument("--device", default="cpu")
    validate.add_argument("--threads", type=int, default=4)
    train = commands.add_parser("train")
    train.add_argument("--config", default="configs/stage-a.json")
    train.add_argument("--profile", choices=("pilot", "a6000-48gb"), default="pilot")
    train.add_argument("--data", required=True)
    train.add_argument("--output", required=True)
    train.add_argument("--method", choices=("direct", "stress"), required=True)
    train.add_argument("--epochs", type=int)
    train.add_argument("--batch-size", type=int)
    train.add_argument("--rollout-steps", type=int)
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
    profile = commands.add_parser("profile")
    profile.add_argument("--case", required=True)
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
    if command in ("generate", "validate", "train"):
        from .generation2d import configuration

        config = configuration(args.pop("config"))
    if command == "generate":
        from .generation2d import generate

        result = generate(config, **args)
    elif command == "validate":
        from .validation2d import validate

        result = validate(config, **args)
    elif command == "train":
        from .training2d import fit

        profile = args.pop("profile")
        settings = dict(config["training"] if profile == "pilot" else config["training_profiles"][profile])
        settings.update({name: value for name, value in args.items() if value is not None})
        result = fit(**settings)
    elif command == "benchmark":
        from .benchmark2d import benchmark

        result = benchmark(**args)
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
