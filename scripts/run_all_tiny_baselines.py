#!/usr/bin/env python3
"""Driver script to run baseline evaluation on all tiny tasks with tiny-gpt2.

This script is Mac-safe and intended for quick smoke tests.

IMPORTANT: [tiny-gpt2] This script is intended for CPU/MPS smoke tests, not CUDA.
Due to PyTorch security restrictions, sshleifer/tiny-gpt2 uses old pickled weights
that cannot be loaded on CUDA with recent Torch versions. For GPU tests, use
phi3-mini or other real models instead.
"""

import argparse
import sys
from pathlib import Path

import pandas as pd

from slm_ace.config import TASK_CONFIGS, get_model_config, get_task_config
from slm_ace.model_manager import load_model_and_tokenizer
from slm_ace.runner import run_dataset_baseline
from slm_ace.utils import get_device


def load_dataset(path: Path) -> list[dict]:
    """Load a dataset from a JSON file."""
    import json
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def main():
    """Run baseline evaluation on all tiny tasks."""
    parser = argparse.ArgumentParser(
        description="Run baseline evaluation on all tiny tasks with tiny-gpt2.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=3,
        help="Limit number of examples per task (default: 3)",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="results/tiny",
        help="Output directory for CSVs (default: results/tiny)",
    )
    parser.add_argument(
        "--model-id",
        type=str,
        default="sshleifer/tiny-gpt2",
        help="Model ID to use (default: sshleifer/tiny-gpt2)",
    )
    
    args = parser.parse_args()
    
    # Warn if using tiny-gpt2 with CUDA
    if "tiny-gpt2" in args.model_id.lower():
        import warnings
        warnings.warn(
            "[tiny-gpt2] This script is intended for CPU/MPS smoke tests, not CUDA. "
            "For GPU tests, use phi3-mini or other real models.",
            UserWarning
        )
    
    # Get device
    device = get_device()
    print(f"Using device: {device}")
    print(f"Model: {args.model_id}")
    print(f"Limit: {args.limit} examples per task")
    print(f"Output directory: {args.output_dir}\n")
    
    # Load model once (reused for all tasks)
    try:
        config = get_model_config(args.model_id)
        print(f"Loading model: {config.model_id}")
        model, tokenizer = load_model_and_tokenizer(config.model_id, device=device)
        print("Model loaded successfully.\n")
    except Exception as e:
        print(f"Error: Failed to load model: {e}")
        return 1
    
    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Run baseline for each task
    tasks = ["tatqa_tiny", "medqa_tiny", "iot_tiny"]
    summaries = []
    
    for task_name in tasks:
        print("=" * 60)
        print(f"Task: {task_name}")
        print("=" * 60)
        
        try:
            # Get task config
            task_config = get_task_config(task_name)
            dataset_path_str = task_config["path"]
            domain = task_config["domain"]
            
            # Resolve dataset path
            repo_root = Path(__file__).parent.parent
            dataset_path = repo_root / dataset_path_str
            
            if not dataset_path.exists():
                print(f"Error: Dataset not found: {dataset_path}")
                continue
            
            # Load dataset
            dataset = load_dataset(dataset_path)
            
            # Apply limit
            if args.limit is not None:
                dataset = dataset[:args.limit]
            
            print(f"Loaded {len(dataset)} examples")
            
            # Run baseline
            results, summary = run_dataset_baseline(
                model=model,
                tokenizer=tokenizer,
                dataset=dataset,
                domain=domain,
                config=config,
                model_id=config.model_id,
                task_name=task_name,
                mode="baseline",
            )
            
            # Save CSV
            sanitized_model_id = config.model_id.replace("/", "_")
            output_path = output_dir / f"{task_name}_{sanitized_model_id}_baseline.csv"
            df = pd.DataFrame(results)
            df.to_csv(output_path, index=False)
            print(f"Results saved to {output_path}")
            
            # Store summary
            summaries.append({
                "task_name": task_name,
                "accuracy": summary["accuracy"],
                "avg_latency_ms": summary["avg_latency_ms"],
                "num_examples": summary["num_examples"],
            })
            
            print(f"Accuracy: {summary['accuracy']:.3f}")
            print(f"Avg latency: {summary['avg_latency_ms']:.2f} ms\n")
            
        except Exception as e:
            print(f"Error processing {task_name}: {e}")
            import traceback
            traceback.print_exc()
            continue
    
    # Print final summary
    print("\n" + "=" * 60)
    print("Summary of All Tasks")
    print("=" * 60)
    print(f"{'Task':<15} {'Accuracy':<12} {'Avg Latency (ms)':<18} {'Examples':<10}")
    print("-" * 60)
    for s in summaries:
        print(f"{s['task_name']:<15} {s['accuracy']:<12.3f} {s['avg_latency_ms']:<18.2f} {s['num_examples']:<10}")
    print("=" * 60)
    
    return 0


if __name__ == "__main__":
    sys.exit(main())

