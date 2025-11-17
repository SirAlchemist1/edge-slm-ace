#!/usr/bin/env python3
"""GPU smoke test script using Phi-3 Mini.

This script runs a quick baseline test with phi3-mini on CUDA to verify
GPU setup is working correctly. Use this instead of tiny-gpt2 for GPU tests.
"""

import argparse
import sys
from pathlib import Path

from slm_ace.config import get_model_config, get_task_config
from slm_ace.model_manager import load_model_and_tokenizer
from slm_ace.runner import run_dataset_baseline
from slm_ace.utils import resolve_device_override


def load_dataset(path: Path) -> list[dict]:
    """Load a dataset from a JSON file."""
    import json
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def main():
    """Run GPU smoke test with Phi-3 Mini."""
    parser = argparse.ArgumentParser(
        description="GPU smoke test with Phi-3 Mini (use instead of tiny-gpt2 for GPU tests).",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--task-name",
        type=str,
        default="tatqa_tiny",
        help="Task name from registry (default: tatqa_tiny)",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=2,
        help="Limit number of examples (default: 2)",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda",
        choices=["cpu", "cuda", "mps"],
        help="Device to use (default: cuda)",
    )
    parser.add_argument(
        "--model-id",
        type=str,
        default="phi3-mini",
        help="Model ID to use (default: phi3-mini)",
    )
    
    args = parser.parse_args()
    
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
    
    # Load model config
    try:
        config = get_model_config(args.model_id)
    except Exception as e:
        print(f"Error: Failed to load model config: {e}")
        return 1
    
    # Resolve device
    device, _ = resolve_device_override(args.device, model_id=config.model_id)
    
    print("=" * 60)
    print("GPU Smoke Test with Phi-3 Mini")
    print("=" * 60)
    print(f"Model: {config.model_id}")
    print(f"Task: {args.task_name}")
    print(f"Device: {device} (requested: {args.device})")
    print(f"Limit: {args.limit} examples")
    print("=" * 60)
    print()
    
    # Load model and tokenizer
    print(f"Loading model: {config.model_id}")
    try:
        model, tokenizer = load_model_and_tokenizer(
            config.model_id,
            device=device,
            device_override=args.device,
        )
        print(f"✓ Model loaded successfully on device: {device}")
    except Exception as e:
        print(f"✗ Error: Failed to load model: {e}")
        return 1
    
    # Load dataset
    try:
        dataset = load_dataset(dataset_path)
    except Exception as e:
        print(f"Error: Failed to load dataset: {e}")
        return 1
    
    # Apply limit
    if args.limit is not None:
        dataset = dataset[:args.limit]
    
    print(f"Loaded {len(dataset)} examples")
    print()
    
    # Run baseline
    print("Running baseline evaluation...")
    try:
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
        
        print()
        print("=" * 60)
        print("GPU smoke test with Phi-3 Mini succeeded!")
        print("=" * 60)
        print(f"Device: {device}")
        print(f"Accuracy: {summary['accuracy']:.3f}")
        print(f"Avg latency: {summary['avg_latency_ms']:.2f} ms")
        print(f"Examples processed: {summary['num_examples']}")
        print("=" * 60)
        
        return 0
        
    except Exception as e:
        print(f"✗ Error during evaluation: {e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    sys.exit(main())

