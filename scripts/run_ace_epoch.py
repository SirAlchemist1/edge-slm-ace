#!/usr/bin/env python3
"""Driver script to run ACE evolution over multiple epochs.

Epoch 0: Baseline (no ACE)
Epochs 1+: ACE mode with playbook evolution

This script is Mac-safe and can be used for testing ACE evolution.
"""

import argparse
import sys
from pathlib import Path

import pandas as pd

from slm_ace.config import get_model_config, get_task_config
from slm_ace.model_manager import load_model_and_tokenizer
from slm_ace.playbook import Playbook
from slm_ace.runner import run_dataset_baseline, run_dataset_ace
from slm_ace.utils import get_device


def load_dataset(path: Path) -> list[dict]:
    """Load a dataset from a JSON file."""
    import json
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def sanitize_model_id(model_id: str) -> str:
    """Sanitize model ID for use in filenames."""
    return model_id.replace("/", "_")


def main():
    """Run ACE evolution over multiple epochs."""
    parser = argparse.ArgumentParser(
        description="Run ACE evolution over multiple epochs (baseline + ACE epochs).",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--model-id",
        type=str,
        required=True,
        help="Model ID or config key (e.g., 'phi3-mini', 'sshleifer/tiny-gpt2')",
    )
    parser.add_argument(
        "--task-name",
        type=str,
        required=True,
        help="Task name from registry (e.g., 'tatqa_tiny', 'medqa_tiny', 'iot_tiny')",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=3,
        help="Total number of epochs (epoch 0 = baseline, epochs 1+ = ACE) (default: 3)",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Limit number of examples per epoch (optional)",
    )
    parser.add_argument(
        "--ace-mode",
        type=str,
        choices=["ace_full", "ace_working_memory"],
        default="ace_full",
        help="ACE mode: 'ace_full' (unbounded playbook) or 'ace_working_memory' (token-budgeted) (default: ace_full)",
    )
    parser.add_argument(
        "--token-budget",
        type=int,
        default=500,
        help="Token budget for working memory mode (default: 500)",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="results/ace",
        help="Output directory for CSVs (default: results/ace)",
    )
    parser.add_argument(
        "--device",
        type=str,
        default=None,
        choices=["cpu", "cuda", "mps"],
        help="Optional device override. If not set, auto-detects (cuda -> mps -> cpu).",
    )
    parser.add_argument(
        "--auto-plots",
        action="store_true",
        help="Automatically regenerate plots after all epochs complete (requires tinyace_plots.py)",
    )
    
    args = parser.parse_args()
    
    # Validate epochs
    if args.epochs < 1:
        parser.error("--epochs must be at least 1")
    
    # Get task config
    try:
        task_config = get_task_config(args.task_name)
        dataset_path_str = task_config["path"]
        domain = task_config["domain"]
    except KeyError as e:
        print(f"Error: {e}")
        return 1
    
    # Resolve dataset path
    repo_root = Path(__file__).parent.parent
    dataset_path = repo_root / dataset_path_str
    
    if not dataset_path.exists():
        print(f"Error: Dataset not found: {dataset_path}")
        return 1
    
    # Load model config first (needed for device resolution)
    try:
        config = get_model_config(args.model_id)
    except Exception as e:
        print(f"Error: Failed to load model config: {e}")
        return 1
    
    # Resolve device (with override support)
    # Note: load_model_and_tokenizer will handle tiny-gpt2 CPU override internally
    if args.device:
        from slm_ace.utils import resolve_device_override
        device, _ = resolve_device_override(args.device, model_id=config.model_id)
    else:
        device = get_device()
    
    # Print summary
    mode_str = f"ACE ({args.ace_mode})" if args.epochs > 1 else "Baseline"
    print(f"Model: {args.model_id} | Task: {args.task_name} | Mode: {mode_str} | Device override: {args.device or 'auto'}")
    print(f"Using device: {device}\n")
    
    # Load model and tokenizer (reused across epochs)
    print(f"Loading model: {config.model_id}")
    try:
        model, tokenizer = load_model_and_tokenizer(
            config.model_id,
            device=device,
            device_override=args.device,
        )
        print("Model loaded successfully.\n")
    except Exception as e:
        print(f"Error: Failed to load model: {e}")
        return 1
    
    # Load dataset
    try:
        dataset = load_dataset(dataset_path)
    except Exception as e:
        print(f"Error: Failed to load dataset: {e}")
        return 1
    
    # Apply limit if specified
    if args.limit is not None:
        dataset = dataset[:args.limit]
        print(f"Limited to {len(dataset)} examples\n")
    
    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    sanitized_model_id = sanitize_model_id(config.model_id)
    
    # Playbook path (shared across ACE epochs)
    playbook_path = output_dir / f"{args.task_name}_playbook.jsonl"
    
    # Run epochs
    epoch_summaries = []
    
    for epoch in range(args.epochs):
        print("=" * 60)
        if epoch == 0:
            print(f"Epoch {epoch}: Baseline")
        else:
            print(f"Epoch {epoch}: ACE")
        print("=" * 60)
        
        try:
            if epoch == 0:
                # Baseline epoch
                results, summary = run_dataset_baseline(
                    model=model,
                    tokenizer=tokenizer,
                    dataset=dataset,
                    domain=domain,
                    config=config,
                    model_id=config.model_id,
                    task_name=args.task_name,
                    mode="baseline",
                )
                
                output_path = output_dir / f"{args.task_name}_{sanitized_model_id}_epoch0_baseline.csv"
                
            else:
                # ACE epochs
                # Load or create playbook
                if playbook_path.exists():
                    playbook = Playbook.load(playbook_path)
                else:
                    playbook = Playbook()
                
                results, summary = run_dataset_ace(
                    model=model,
                    tokenizer=tokenizer,
                    dataset=dataset,
                    domain=domain,
                    config=config,
                    playbook=playbook,
                    playbook_path=playbook_path,
                    model_id=config.model_id,
                    task_name=args.task_name,
                    mode="ace",
                    ace_mode=args.ace_mode,
                    token_budget=args.token_budget,
                )
                
                output_path = output_dir / f"{args.task_name}_{sanitized_model_id}_epoch{epoch}.csv"
                print(f"Playbook size: {len(playbook.entries)} entries")
            
            # Save CSV
            df = pd.DataFrame(results)
            # Add epoch column for tracking
            df["epoch"] = epoch
            df.to_csv(output_path, index=False)
            print(f"Results saved to {output_path}")
            
            # Store summary
            epoch_summaries.append({
                "epoch": epoch,
                "mode": "baseline" if epoch == 0 else "ace",
                "accuracy": summary["accuracy"],
                "avg_latency_ms": summary["avg_latency_ms"],
                "num_examples": summary["num_examples"],
            })
            
            print(f"Accuracy: {summary['accuracy']:.3f}")
            print(f"Avg latency: {summary['avg_latency_ms']:.2f} ms\n")
            
        except Exception as e:
            print(f"Error during epoch {epoch}: {e}")
            import traceback
            traceback.print_exc()
            continue
    
    # Print final summary table
    print("\n" + "=" * 60)
    print("Epoch Summary")
    print("=" * 60)
    print(f"{'Epoch':<8} {'Mode':<12} {'Accuracy':<12} {'Avg Latency (ms)':<18} {'Examples':<10}")
    print("-" * 60)
    for s in epoch_summaries:
        print(f"{s['epoch']:<8} {s['mode']:<12} {s['accuracy']:<12.3f} {s['avg_latency_ms']:<18.2f} {s['num_examples']:<10}")
    print("=" * 60)
    
    # Optionally regenerate plots
    if args.auto_plots:
        try:
            import sys
            from pathlib import Path
            # Import tinyace_plots module
            repo_root = Path(__file__).parent.parent
            sys.path.insert(0, str(repo_root))
            from tinyace_plots import main as regenerate_plots
            
            print("\n" + "=" * 60)
            print("Regenerating plots...")
            print("=" * 60)
            regenerate_plots(results_dir="results", output_dir="tinyace_plots")
            print("Plots regenerated successfully.")
        except ImportError as e:
            print(f"\nWarning: Could not import tinyace_plots: {e}")
            print("Skipping plot regeneration. Run manually with: python tinyace_plots.py")
        except Exception as e:
            print(f"\nWarning: Plot regeneration failed: {e}")
            print("You can regenerate plots manually with: python tinyace_plots.py")
    
    return 0


if __name__ == "__main__":
    sys.exit(main())


# Example commands:
#
# Baseline + ACE full mode (tiny model, Mac-safe):
# python -m scripts.run_ace_epoch \
#   --model-id sshleifer/tiny-gpt2 \
#   --task-name tatqa_tiny \
#   --epochs 2 \
#   --ace-mode ace_full \
#   --limit 5 \
#   --output-dir results/ace_tiny
#
# Baseline + ACE working memory mode:
# python -m scripts.run_ace_epoch \
#   --model-id sshleifer/tiny-gpt2 \
#   --task-name tatqa_tiny \
#   --epochs 2 \
#   --ace-mode ace_working_memory \
#   --token-budget 500 \
#   --limit 5 \
#   --output-dir results/ace_tiny
#
# Full run (GPU/supercomputer):
# python -m scripts.run_ace_epoch \
#   --model-id phi3-mini \
#   --task-name medqa_tiny \
#   --epochs 5 \
#   --ace-mode ace_working_memory \
#   --device cuda \
#   --output-dir results/ace_phi3

