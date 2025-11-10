#!/usr/bin/env python3
"""Grid experiment runner.

Reads configs/exp_grid.yaml and runs experiments for all combinations of
model × task × mode. For baseline mode, calls run_experiment. For ACE mode,
calls run_ace_epoch.

This script is Mac-safe and can be used for running large experiment grids.
"""

import argparse
import subprocess
import sys
from pathlib import Path

import yaml


def load_grid_config(config_path: Path) -> dict:
    """Load experiment grid configuration from YAML file."""
    with open(config_path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def sanitize_model_id(model_id: str) -> str:
    """Sanitize model ID for use in filenames."""
    return model_id.replace("/", "_").replace("-", "_")


def main():
    """Run grid experiments."""
    parser = argparse.ArgumentParser(
        description="Run grid experiments from configs/exp_grid.yaml",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--config",
        type=str,
        default="configs/exp_grid.yaml",
        help="Path to grid config YAML file (default: configs/exp_grid.yaml)",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print commands without executing them",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Limit number of examples per experiment (optional)",
    )
    parser.add_argument(
        "--model-id",
        type=str,
        default=None,
        help="Filter to only run this model ID (optional)",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="results/grid",
        help="Output directory for results (default: results/grid)",
    )

    args = parser.parse_args()

    # Load grid config
    config_path = Path(args.config)
    if not config_path.exists():
        print(f"Error: Config file not found: {config_path}")
        return 1

    try:
        config = load_grid_config(config_path)
    except Exception as e:
        print(f"Error: Failed to load config: {e}")
        return 1

    models = config.get("models", [])
    tasks = config.get("tasks", [])
    modes = config.get("modes", [])

    if not models or not tasks or not modes:
        print("Error: Config must specify models, tasks, and modes")
        return 1

    # Filter models if --model-id specified
    if args.model_id:
        if args.model_id not in models:
            print(f"Warning: Model '{args.model_id}' not in config, but proceeding anyway")
        models = [args.model_id]

    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    repo_root = Path(__file__).parent.parent

    # Generate all combinations
    total_experiments = len(models) * len(tasks) * len(modes)
    print(f"Grid configuration:")
    print(f"  Models: {models}")
    print(f"  Tasks: {tasks}")
    print(f"  Modes: {modes}")
    print(f"  Total experiments: {total_experiments}")
    if args.dry_run:
        print(f"  Mode: DRY RUN (commands will be printed, not executed)")
    print()

    experiment_num = 0
    for model_id in models:
        for task_name in tasks:
            for mode in modes:
                experiment_num += 1
                print(f"[{experiment_num}/{total_experiments}] Model: {model_id}, Task: {task_name}, Mode: {mode}")

                if mode == "baseline":
                    # Use run_experiment for baseline
                    output_path = output_dir / f"{task_name}_{sanitize_model_id(model_id)}_baseline.csv"
                    cmd = [
                        sys.executable,
                        "-m",
                        "scripts.run_experiment",
                        "--model-id",
                        model_id,
                        "--task-name",
                        task_name,
                        "--mode",
                        "baseline",
                        "--output-path",
                        str(output_path),
                    ]
                    if args.limit:
                        cmd.extend(["--limit", str(args.limit)])

                elif mode == "ace":
                    # Use run_ace_epoch for ACE (with epochs=1 for single ACE run)
                    # Note: This runs epoch 0 (baseline) + epoch 1 (ACE)
                    ace_output_dir = output_dir / f"{task_name}_{sanitize_model_id(model_id)}_ace"
                    cmd = [
                        sys.executable,
                        "-m",
                        "scripts.run_ace_epoch",
                        "--model-id",
                        model_id,
                        "--task-name",
                        task_name,
                        "--epochs",
                        "2",  # epoch 0 = baseline, epoch 1 = ACE
                        "--output-dir",
                        str(ace_output_dir),
                    ]
                    if args.limit:
                        cmd.extend(["--limit", str(args.limit)])

                else:
                    print(f"  Warning: Unknown mode '{mode}', skipping")
                    continue

                if args.dry_run:
                    print(f"  Would run: {' '.join(cmd)}")
                else:
                    try:
                        result = subprocess.run(
                            cmd,
                            cwd=repo_root,
                            check=True,
                            capture_output=False,
                        )
                        print(f"  ✅ Completed")
                    except subprocess.CalledProcessError as e:
                        print(f"  ❌ Failed with exit code {e.returncode}")
                        # Continue with next experiment
                        continue

                print()

    print("=" * 60)
    print(f"Grid experiment complete!")
    print(f"Results saved to: {output_dir}")
    print("=" * 60)

    return 0


if __name__ == "__main__":
    sys.exit(main())

