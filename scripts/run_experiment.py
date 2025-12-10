#!/usr/bin/env python3
"""CLI entrypoint for running baseline or ACE-style experiments.

This script provides a unified interface for running experiments with:
- Multiple evaluation modes (baseline, ACE full, ACE working memory)
- Device selection (CPU, CUDA, MPS)
- Structured output (metrics JSON, predictions JSONL, CSV)

Example usage:
    # Baseline evaluation
    python -m scripts.run_experiment \
        --model-id sshleifer/tiny-gpt2 \
        --task-name tatqa_tiny \
        --mode baseline \
        --output-path results/baseline.csv \
        --metrics-path results/metrics.json \
        --predictions-path results/predictions.jsonl

    # ACE working memory mode
    python -m scripts.run_experiment \
        --model-id sshleifer/tiny-gpt2 \
        --task-name tatqa_tiny \
        --mode ace \
        --ace-mode ace_working_memory \
        --token-budget 500 \
        --playbook-path playbooks/tatqa.jsonl \
        --output-path results/ace.csv
"""

import argparse
import json
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

import pandas as pd

from edge_slm_ace.utils.config import get_model_config, get_task_config, ModelConfig
from edge_slm_ace.models.model_manager import load_model_and_tokenizer
from edge_slm_ace.memory.playbook import Playbook
from edge_slm_ace.core.runner import run_dataset_baseline, run_dataset_ace, run_dataset_self_refine
from edge_slm_ace.utils.device_utils import get_device, resolve_device_override


def load_dataset(path: Path) -> List[Dict]:
    """
    Load a dataset from a JSON file.
    
    Expected format: list of dicts with keys: id, question, answer, (optional) context, domain.
    """
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def save_metrics(metrics: Dict[str, Any], path: Path) -> None:
    """Save metrics to a JSON file."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2, default=str)


def save_predictions(results: List[Dict], path: Path) -> None:
    """Save per-example predictions to a JSONL file."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for result in results:
            f.write(json.dumps(result, default=str) + "\n")


def save_run_metadata(metadata: Dict[str, Any], path: Path) -> None:
    """Save run metadata to a JSON file."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2, default=str)


def main() -> int:
    """
    Main CLI entrypoint for running experiments.
    
    Returns:
        Exit code (0 for success, non-zero for error).
    """
    parser = argparse.ArgumentParser(
        description="Run baseline or ACE-style evaluation on a dataset.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Run baseline evaluation
  python -m scripts.run_experiment \\
      --model-id sshleifer/tiny-gpt2 \\
      --task-name tatqa_tiny \\
      --mode baseline \\
      --output-path results/baseline.csv

  # Run ACE with working memory
  python -m scripts.run_experiment \\
      --model-id sshleifer/tiny-gpt2 \\
      --task-name tatqa_tiny \\
      --mode ace \\
      --ace-mode ace_working_memory \\
      --token-budget 500 \\
      --playbook-path playbooks/tatqa.jsonl \\
      --output-path results/ace.csv
        """
    )
    
    # Required arguments
    parser.add_argument(
        "--model-id",
        type=str,
        required=True,
        help="HuggingFace model ID or key from config (e.g., 'sshleifer/tiny-gpt2', 'phi3-mini')",
    )
    parser.add_argument(
        "--mode",
        type=str,
        choices=["baseline", "ace", "self_refine"],
        required=True,
        help="Run mode: 'baseline', 'ace', or 'self_refine'",
    )
    parser.add_argument(
        "--output-path",
        type=str,
        required=True,
        help="Path to save results CSV",
    )
    
    # Dataset/task specification (one of these required)
    task_group = parser.add_mutually_exclusive_group()
    task_group.add_argument(
        "--task-name",
        type=str,
        default=None,
        help="Task name from registry (e.g., 'tatqa_tiny', 'medqa_tiny', 'iot_tiny')",
    )
    task_group.add_argument(
        "--dataset-path",
        type=str,
        default=None,
        help="Path to JSON dataset file (use with --domain)",
    )
    
    parser.add_argument(
        "--domain",
        type=str,
        default=None,
        help="Domain name (required if using --dataset-path instead of --task-name)",
    )
    
    # Output paths
    parser.add_argument(
        "--metrics-path",
        type=str,
        default=None,
        help="Path to save metrics JSON (optional)",
    )
    parser.add_argument(
        "--predictions-path",
        type=str,
        default=None,
        help="Path to save per-example predictions JSONL (optional)",
    )
    parser.add_argument(
        "--run-name",
        type=str,
        default=None,
        help="Run name for logging and metadata (optional)",
    )
    
    # ACE-specific parameters
    parser.add_argument(
        "--playbook-path",
        type=str,
        default=None,
        help="Path to playbook JSONL file (required for ACE mode)",
    )
    parser.add_argument(
        "--ace-mode",
        type=str,
        choices=["ace_full", "ace_working_memory"],
        default="ace_full",
        help="ACE mode: 'ace_full' (top-k entries) or 'ace_working_memory' (token-budgeted). Default: ace_full",
    )
    parser.add_argument(
        "--token-budget",
        type=int,
        default=500,
        help="Token budget for working memory mode (default: 500)",
    )
    parser.add_argument(
        "--top-k",
        type=int,
        default=5,
        help="Number of top playbook entries to use (default: 5)",
    )
    parser.add_argument(
        "--prune-every-n",
        type=int,
        default=10,
        help="Prune playbook every N examples (default: 10)",
    )
    parser.add_argument(
        "--max-entries-per-domain",
        type=int,
        default=32,
        help="Maximum playbook entries per domain after pruning (default: 32)",
    )
    
    # Generation parameters
    parser.add_argument(
        "--max-new-tokens",
        type=int,
        default=None,
        help="Override max_new_tokens from config",
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=None,
        help="Override temperature from config",
    )
    parser.add_argument(
        "--top-p",
        type=float,
        default=None,
        help="Override top_p from config",
    )
    
    # Device and limits
    parser.add_argument(
        "--device",
        type=str,
        default=None,
        choices=["cpu", "cuda", "mps"],
        help="Device override. If not set, auto-detects (cuda -> mps -> cpu)",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Limit number of examples to process (for quick testing)",
    )
    
    # Other options
    parser.add_argument(
        "--auto-plots",
        action="store_true",
        help="Automatically regenerate plots after evaluation",
    )
    parser.add_argument(
        "--quiet",
        action="store_true",
        help="Suppress progress output",
    )
    
    args = parser.parse_args()
    
    # Track timing
    run_start_time = datetime.now()
    wall_start = time.time()
    
    try:
        # Validate arguments
        if args.task_name is None and args.dataset_path is None:
            parser.error("Either --task-name or --dataset-path must be provided")
        
        if args.dataset_path is not None and args.domain is None:
            parser.error("--domain is required when using --dataset-path")
        
        if args.mode == "ace" and args.playbook_path is None:
            parser.error("--playbook-path is required for ACE mode")
        
        # Resolve dataset path and domain
        if args.task_name:
            try:
                task_config = get_task_config(args.task_name)
                dataset_path_str = task_config["path"]
                domain = task_config["domain"]
                task_name = args.task_name
                
                # Resolve relative paths
                if not Path(dataset_path_str).is_absolute():
                    repo_root = Path(__file__).parent.parent
                    dataset_path_str = str(repo_root / dataset_path_str)
            except KeyError as e:
                print(f"Error: {e}")
                return 1
        else:
            dataset_path_str = args.dataset_path
            domain = args.domain
            task_name = Path(args.dataset_path).stem
        
        # Load model config
        try:
            config = get_model_config(args.model_id)
        except Exception as e:
            print(f"Error: Failed to load model config for '{args.model_id}': {e}")
            return 1
        
        # Override config parameters
        if args.max_new_tokens is not None:
            config.max_new_tokens = args.max_new_tokens
        if args.temperature is not None:
            config.temperature = args.temperature
        if args.top_p is not None:
            config.top_p = args.top_p
        
        # Resolve device
        device_requested = args.device or "auto"
        if args.device:
            device, forced = resolve_device_override(args.device, model_id=config.model_id)
            device_override_str = str(device.type) if forced != "forced" else "cpu"
        else:
            device = get_device()
            device_override_str = None
        device_used = str(device.type)
        
        if not args.quiet:
            print(f"Using device: {device} (requested: {device_requested})")
        
        # Load model and tokenizer
        if not args.quiet:
            print(f"Loading model: {config.model_id}")
        
        try:
            model, tokenizer = load_model_and_tokenizer(
                config.model_id,
                device=device,
                device_override=device_override_str,
            )
            if not args.quiet:
                print("Model loaded successfully.")
        except Exception as e:
            print(f"Error: Failed to load model '{config.model_id}': {e}")
            return 1
        
        # Load dataset
        dataset_path = Path(dataset_path_str)
        if not dataset_path.exists():
            print(f"Error: Dataset not found: {dataset_path}")
            return 1
        
        try:
            dataset = load_dataset(dataset_path)
        except Exception as e:
            print(f"Error: Failed to load dataset: {e}")
            return 1
        
        original_size = len(dataset)
        if args.limit is not None:
            dataset = dataset[:args.limit]
        
        if not args.quiet:
            print(f"Loaded {len(dataset)} examples from {dataset_path}" + 
                  (f" (limited from {original_size})" if args.limit else ""))
        
        # Run evaluation
        if args.mode == "baseline":
            if not args.quiet:
                print("Running baseline evaluation...")
            results, summary = run_dataset_baseline(
                model=model,
                tokenizer=tokenizer,
                dataset=dataset,
                domain=domain,
                config=config,
                model_id=config.model_id,
                task_name=task_name,
                mode=args.mode,
            )
            playbook_stats = None
            
        elif args.mode == "self_refine":
            if not args.quiet:
                print("Running self-refinement evaluation...")
            results, summary = run_dataset_self_refine(
                model=model,
                tokenizer=tokenizer,
                dataset=dataset,
                domain=domain,
                config=config,
                model_id=config.model_id,
                task_name=task_name,
                mode=args.mode,
            )
            playbook_stats = None
            
        else:  # ACE mode
            if not args.quiet:
                print(f"Running ACE evaluation (mode: {args.ace_mode})...")
            
            playbook_path = Path(args.playbook_path)
            
            # Load or create playbook
            if playbook_path.exists():
                if not args.quiet:
                    print(f"Loading playbook from {playbook_path}")
                playbook = Playbook.load(playbook_path, token_budget=args.token_budget)
            else:
                if not args.quiet:
                    print(f"Creating new playbook at {playbook_path}")
                playbook = Playbook(token_budget=args.token_budget)
            
            initial_playbook_size = len(playbook.entries)
            
            results, summary = run_dataset_ace(
                model=model,
                tokenizer=tokenizer,
                dataset=dataset,
                domain=domain,
                config=config,
                playbook=playbook,
                playbook_path=playbook_path,
                model_id=config.model_id,
                task_name=task_name,
                mode=args.mode,
                ace_mode=args.ace_mode,
                token_budget=args.token_budget,
                top_k=args.top_k,
                prune_every_n=args.prune_every_n,
                max_entries_per_domain=args.max_entries_per_domain,
            )
            
            playbook_stats = {
                "initial_size": initial_playbook_size,
                "final_size": len(playbook.entries),
                "entries_added": len(playbook.entries) - initial_playbook_size,
                "domain_stats": playbook.get_stats(domain),
            }
            
            if not args.quiet:
                print(f"Playbook: {initial_playbook_size} → {len(playbook.entries)} entries")
        
        # Calculate wall time
        wall_time_seconds = time.time() - wall_start
        
        # Save results CSV
        output_path = Path(args.output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        df = pd.DataFrame(results)
        df.to_csv(output_path, index=False)
        if not args.quiet:
            print(f"Results saved to {output_path}")
        
        # Build comprehensive metrics
        metrics = {
            "run_name": args.run_name or f"{task_name}_{args.mode}",
            "timestamp": run_start_time.isoformat(),
            "wall_time_seconds": wall_time_seconds,
            "model_id": config.model_id,
            "task_name": task_name,
            "domain": domain,
            "mode": args.mode,
            "ace_mode": args.ace_mode if args.mode == "ace" else None,
            "device_requested": device_requested,
            "device_used": device_used,
            "num_examples": len(dataset),
            "limit_applied": args.limit,
            **summary,
        }
        
        if playbook_stats:
            metrics["playbook"] = playbook_stats
        
        # Save metrics JSON if requested
        if args.metrics_path:
            save_metrics(metrics, Path(args.metrics_path))
            if not args.quiet:
                print(f"Metrics saved to {args.metrics_path}")
        
        # Save predictions JSONL if requested
        if args.predictions_path:
            save_predictions(results, Path(args.predictions_path))
            if not args.quiet:
                print(f"Predictions saved to {args.predictions_path}")
        
        # Print summary
        if not args.quiet:
            print("\n" + "=" * 50)
            print("Summary:")
            print("=" * 50)
            print(f"Task: {task_name} ({domain})")
            print(f"Mode: {args.mode}" + (f" ({args.ace_mode})" if args.mode == "ace" else ""))
            print(f"Examples: {len(dataset)}")
            print(f"Accuracy: {summary.get('accuracy', 0):.4f}")
            print(f"Avg Latency: {summary.get('avg_latency_ms', 0):.1f}ms")
            print(f"Wall Time: {wall_time_seconds:.1f}s")
            if playbook_stats:
                print(f"Playbook: {playbook_stats['initial_size']} → {playbook_stats['final_size']} entries")
            print("=" * 50)
        
        # Optionally regenerate plots
        if args.auto_plots:
            try:
                import sys
                repo_root = Path(__file__).parent.parent
                sys.path.insert(0, str(repo_root))
                from tinyace_plots import main as regenerate_plots
                
                if not args.quiet:
                    print("\nRegenerating plots...")
                regenerate_plots(results_dir="results", output_dir="tinyace_plots")
                if not args.quiet:
                    print("Plots regenerated successfully.")
            except Exception as e:
                if not args.quiet:
                    print(f"Warning: Plot regeneration failed: {e}")
        
        return 0
        
    except KeyboardInterrupt:
        print("\n\nInterrupted by user")
        return 130
    except Exception as e:
        print(f"\nUnexpected error: {e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    import sys
    sys.exit(main())
