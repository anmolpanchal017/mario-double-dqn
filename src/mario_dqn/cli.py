from __future__ import annotations

import argparse
from pathlib import Path

from .config import load_config

def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Mario Double DQN command line interface.")
    subparsers = parser.add_subparsers(dest="command", required=True)

    train_parser = subparsers.add_parser("train", help="Train the Double DQN agent.")
    train_parser.add_argument("--config", default="configs/smoke.toml", help="Path to a TOML config.")
    train_parser.add_argument(
        "--resume-checkpoint",
        default=None,
        help="Resume training from an existing checkpoint, usually checkpoints/latest.pt.",
    )

    eval_parser = subparsers.add_parser("eval", help="Evaluate a checkpoint.")
    eval_parser.add_argument("--config", default="configs/smoke.toml", help="Path to a TOML config.")
    eval_parser.add_argument("--checkpoint", required=True, help="Path to a .pt checkpoint.")
    eval_parser.add_argument("--episodes", type=int, default=1, help="Number of evaluation episodes.")
    eval_parser.add_argument("--render", action="store_true", help="Render the emulator window.")
    eval_parser.add_argument("--fps", type=float, default=0.0, help="Playback FPS when rendering.")
    eval_parser.add_argument(
        "--window-scale",
        type=int,
        default=1,
        help="Scale the rendered emulator window, for example 3 for a larger view.",
    )

    plot_parser = subparsers.add_parser("plot", help="Generate plots from a run directory.")
    plot_parser.add_argument("--run-dir", required=True, help="Run directory containing metrics.csv.")

    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)

    if args.command == "train":
        config = load_config(args.config)
        from .train import train

        run_dir = train(config, args.config, args.resume_checkpoint)
        print(f"Training complete. Run directory: {run_dir}")
        return 0

    if args.command == "eval":
        config = load_config(args.config)
        from .evaluate import evaluate

        rewards = evaluate(
            config,
            args.checkpoint,
            args.episodes,
            args.render,
            args.fps,
            args.window_scale,
        )
        mean_reward = sum(rewards) / len(rewards)
        print(f"Mean evaluation reward: {mean_reward:.1f}")
        return 0

    if args.command == "plot":
        from .plots import generate_plots

        outputs = generate_plots(Path(args.run_dir))
        for output in outputs:
            print(f"Wrote {output}")
        return 0

    parser.error(f"Unknown command: {args.command}")
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
