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
        "--output-dir",
        type=str,
        default="results/ace",
        help="Output directory for CSVs (default: results/ace)",
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
    
    # Get device
    device = get_device()
    print(f"Using device: {device}")
    
    # Load model config
    try:
        config = get_model_config(args.model_id)
    except Exception as e:
        print(f"Error: Failed to load model config: {e}")
        return 1
    
    # Load model and tokenizer (reused across epochs)
    print(f"Loading model: {config.model_id}")
    try:
        model, tokenizer = load_model_and_tokenizer(config.model_id, device=device)
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
    
    return 0


if __name__ == "__main__":
    sys.exit(main())

