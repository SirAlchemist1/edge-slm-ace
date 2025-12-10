#!/usr/bin/env python3
"""Grid experiment runner for systematic model × task × mode × device evaluation.

This script reads a YAML configuration file and runs experiments for all
combinations of models, tasks, modes, and devices.

Usage:
    # Dry run (print commands without executing)
    python -m scripts.run_eval_grid --config configs/experiment_grid.yaml --dry-run

    # Run all experiments
    python -m scripts.run_eval_grid --config configs/experiment_grid.yaml

    # Run with limit per experiment
    python -m scripts.run_eval_grid --config configs/experiment_grid.yaml --limit 10
"""

import argparse
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import torch

try:
    import yaml
except ImportError:
    print("Error: PyYAML is required. Install with: pip install pyyaml")
    sys.exit(1)


def load_grid_config(config_path: Path) -> Dict[str, Any]:
    """Load experiment grid configuration from YAML file."""
    with open(config_path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def check_device_availability(device: str) -> Tuple[bool, str]:
    """
    Check if a device is available.
    
    Args:
        device: Device name ("cpu", "cuda", "mps").
        
    Returns:
        Tuple of (is_available, fallback_device).
    """
    if device == "cpu":
        return True, "cpu"
    
    if device == "cuda":
        if torch.cuda.is_available():
            return True, "cuda"
        else:
            return False, "cpu"
    
    if device == "mps":
        if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            return True, "mps"
        else:
            return False, "cpu"
    
    return False, "cpu"


def sanitize_for_path(s: str) -> str:
    """Sanitize a string for use in file paths."""
    return s.replace("/", "_").replace("-", "_").replace(" ", "_")


def build_output_dir(
    results_root: str,
    model_name: str,
    task_name: str,
    mode_name: str,
    device: str,
) -> Path:
    """Build the output directory path for an experiment."""
    return Path(results_root) / sanitize_for_path(model_name) / task_name / mode_name / device


def build_experiment_command(
    model_config: Dict[str, Any],
    task_config: Dict[str, Any],
    mode_config: Dict[str, Any],
    device: str,
    output_dir: Path,
    defaults: Dict[str, Any],
    limit: Optional[int],
) -> List[str]:
    """Build the command to run a single experiment."""
    cmd = [
        sys.executable, "-m", "scripts.run_experiment",
        "--model-id", model_config["hf_id"],
        "--task-name", task_config["task_name"],
        "--mode", mode_config["mode"],
        "--output-path", str(output_dir / "results.csv"),
        "--metrics-path", str(output_dir / "metrics.json"),
        "--predictions-path", str(output_dir / "predictions.jsonl"),
        "--device", device,
    ]
    
    # Add ACE-specific parameters
    if mode_config["mode"] == "ace":
        ace_mode = mode_config.get("ace_mode", "ace_full")
        cmd.extend(["--ace-mode", ace_mode])
        
        # Playbook path
        playbook_dir = defaults.get("playbook_dir", "playbooks")
        playbook_filename = f"{task_config['task_name']}_{sanitize_for_path(model_config['name'])}.jsonl"
        playbook_path = output_dir / "playbook.jsonl"
        cmd.extend(["--playbook-path", str(playbook_path)])
        
        # Token budget (for working memory mode)
        if "token_budget" in mode_config:
            cmd.extend(["--token-budget", str(mode_config["token_budget"])])
        
        # Top-k
        if "top_k" in mode_config:
            cmd.extend(["--top-k", str(mode_config["top_k"])])
        
        # Pruning settings
        if "prune_every_n" in mode_config:
            cmd.extend(["--prune-every-n", str(mode_config["prune_every_n"])])
        if "max_entries_per_domain" in mode_config:
            cmd.extend(["--max-entries-per-domain", str(mode_config["max_entries_per_domain"])])
    
    # Add limit if specified
    effective_limit = limit or defaults.get("limit")
    if effective_limit:
        cmd.extend(["--limit", str(effective_limit)])
    
    # Add generation parameters from defaults
    if "max_new_tokens" in defaults:
        cmd.extend(["--max-new-tokens", str(defaults["max_new_tokens"])])
    if "temperature" in defaults:
        cmd.extend(["--temperature", str(defaults["temperature"])])
    if "top_p" in defaults:
        cmd.extend(["--top-p", str(defaults["top_p"])])
    
    # Run name
    run_name = f"{model_config['name']}_{task_config['name']}_{mode_config['name']}_{device}"
    cmd.extend(["--run-name", run_name])
    
    return cmd


def run_experiment(
    cmd: List[str],
    output_dir: Path,
    dry_run: bool = False,
    verbose: bool = True,
) -> Tuple[bool, Optional[str]]:
    """
    Run a single experiment.
    
    Args:
        cmd: Command to run.
        output_dir: Output directory for the experiment.
        dry_run: If True, just print the command without running.
        verbose: If True, print progress information.
        
    Returns:
        Tuple of (success, error_message).
    """
    if dry_run:
        print(f"  [DRY RUN] Would run: {' '.join(cmd)}")
        return True, None
    
    # Create output directory
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Run the experiment
    try:
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=3600,  # 1 hour timeout
        )
        
        if result.returncode == 0:
            return True, None
        else:
            error_msg = result.stderr or result.stdout or f"Exit code: {result.returncode}"
            return False, error_msg
            
    except subprocess.TimeoutExpired:
        return False, "Experiment timed out after 1 hour"
    except Exception as e:
        return False, str(e)


def main() -> int:
    """Main entry point for the grid runner."""
    parser = argparse.ArgumentParser(
        description="Run experiments for all combinations in a grid configuration.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    
    parser.add_argument(
        "--config",
        type=str,
        required=True,
        help="Path to experiment grid YAML configuration",
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
        help="Override limit per experiment (for quick testing)",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Print detailed progress information",
    )
    parser.add_argument(
        "--skip-unavailable-devices",
        action="store_true",
        default=True,
        help="Skip experiments for unavailable devices (default: True)",
    )
    
    args = parser.parse_args()
    
    # Load configuration
    config_path = Path(args.config)
    if not config_path.exists():
        print(f"Error: Config file not found: {config_path}")
        return 1
    
    try:
        config = load_grid_config(config_path)
    except Exception as e:
        print(f"Error: Failed to load config: {e}")
        return 1
    
    # Extract configuration sections
    models = config.get("models", [])
    tasks = config.get("tasks", [])
    modes = config.get("modes", [])
    devices = config.get("devices", ["cpu"])
    defaults = config.get("defaults", {})
    
    if not models:
        print("Error: No models specified in config")
        return 1
    if not tasks:
        print("Error: No tasks specified in config")
        return 1
    if not modes:
        print("Error: No modes specified in config")
        return 1
    
    # Calculate total experiments
    total_experiments = len(models) * len(tasks) * len(modes) * len(devices)
    
    print(f"=" * 60)
    print(f"Experiment Grid Runner")
    print(f"=" * 60)
    print(f"Config: {config_path}")
    print(f"Models: {len(models)}")
    print(f"Tasks: {len(tasks)}")
    print(f"Modes: {len(modes)}")
    print(f"Devices: {len(devices)}")
    print(f"Total combinations: {total_experiments}")
    if args.dry_run:
        print(f"Mode: DRY RUN")
    print(f"=" * 60)
    
    # Track results
    results = {
        "success": [],
        "failed": [],
        "skipped": [],
    }
    
    start_time = time.time()
    experiment_num = 0
    
    # Iterate over all combinations
    for model_config in models:
        model_name = model_config.get("name", model_config.get("hf_id", "unknown"))
        
        for task_config in tasks:
            task_name = task_config.get("name", task_config.get("task_name", "unknown"))
            
            for mode_config in modes:
                mode_name = mode_config.get("name", mode_config.get("mode", "unknown"))
                
                for device in devices:
                    experiment_num += 1
                    experiment_id = f"{model_name}/{task_name}/{mode_name}/{device}"
                    
                    print(f"\n[{experiment_num}/{total_experiments}] {experiment_id}")
                    
                    # Check device availability
                    device_available, fallback_device = check_device_availability(device)
                    
                    if not device_available:
                        if args.skip_unavailable_devices:
                            print(f"  ⚠ Device '{device}' not available, skipping")
                            results["skipped"].append({
                                "experiment": experiment_id,
                                "reason": f"Device '{device}' not available",
                            })
                            continue
                        else:
                            print(f"  ⚠ Device '{device}' not available, using '{fallback_device}'")
                            device = fallback_device
                    
                    # Build output directory
                    results_root = defaults.get("results_root", "results")
                    output_dir = build_output_dir(
                        results_root, model_name, task_name, mode_name, device
                    )
                    
                    # Build command
                    cmd = build_experiment_command(
                        model_config=model_config,
                        task_config=task_config,
                        mode_config=mode_config,
                        device=device,
                        output_dir=output_dir,
                        defaults=defaults,
                        limit=args.limit,
                    )
                    
                    if args.verbose or args.dry_run:
                        print(f"  Output: {output_dir}")
                    
                    # Run experiment
                    success, error_msg = run_experiment(
                        cmd=cmd,
                        output_dir=output_dir,
                        dry_run=args.dry_run,
                        verbose=args.verbose,
                    )
                    
                    if success:
                        if not args.dry_run:
                            print(f"  ✓ Success")
                        results["success"].append({
                            "experiment": experiment_id,
                            "output_dir": str(output_dir),
                        })
                    else:
                        print(f"  ✗ Failed: {error_msg[:200] if error_msg else 'Unknown error'}")
                        results["failed"].append({
                            "experiment": experiment_id,
                            "error": error_msg,
                        })
    
    # Print summary
    elapsed_time = time.time() - start_time
    
    print(f"\n" + "=" * 60)
    print(f"Grid Run Complete")
    print(f"=" * 60)
    print(f"Total time: {elapsed_time:.1f}s")
    print(f"Successful: {len(results['success'])}")
    print(f"Failed: {len(results['failed'])}")
    print(f"Skipped: {len(results['skipped'])}")
    
    if results["failed"]:
        print(f"\nFailed experiments:")
        for failure in results["failed"]:
            print(f"  - {failure['experiment']}")
    
    if results["skipped"]:
        print(f"\nSkipped experiments:")
        for skipped in results["skipped"]:
            print(f"  - {skipped['experiment']}: {skipped['reason']}")
    
    print(f"=" * 60)
    
    # Return non-zero if any experiments failed
    return 1 if results["failed"] else 0


if __name__ == "__main__":
    sys.exit(main())
