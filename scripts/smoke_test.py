#!/usr/bin/env python3
"""Smoke test script for tiny-gpt2 CPU-only verification.

This script verifies that tiny-gpt2 runs correctly on CPU, even when CUDA is requested.
This is expected behavior due to PyTorch >=2.6 security restrictions.
"""

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
    """Run smoke test with tiny-gpt2."""
    print("=" * 60)
    print("tiny-gpt2 CPU-only Smoke Test")
    print("=" * 60)
    print("This test verifies that tiny-gpt2 runs on CPU even when CUDA is requested.")
    print("This is expected behavior due to PyTorch >=2.6 security restrictions.")
    print("=" * 60)
    print()
    
    model_id = "sshleifer/tiny-gpt2"
    task_name = "tatqa_tiny"
    
    # Get task config
    try:
        task_config = get_task_config(task_name)
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
        config = get_model_config(model_id)
    except Exception as e:
        print(f"Error: Failed to load model config: {e}")
        return 1
    
    # Test: Request CUDA but should get CPU
    print("Testing device resolution (requesting CUDA, expecting CPU override)...")
    device, forced = resolve_device_override("cuda", model_id=config.model_id)
    if device.type != "cpu":
        print(f"ERROR: Expected CPU device, got {device}")
        return 1
    if forced != "forced":
        print(f"ERROR: Expected forced=True, got {forced}")
        return 1
    print(f"✓ Device correctly forced to CPU: {device}")
    print()
    
    # Load model and tokenizer
    print(f"Loading model: {config.model_id}")
    try:
        model, tokenizer = load_model_and_tokenizer(
            config.model_id,
            device=device,
            device_override="cuda",  # Request CUDA, should be forced to CPU
        )
        print(f"✓ Model loaded successfully on device: {device}")
    except Exception as e:
        print(f"✗ Error: Failed to load model: {e}")
        import traceback
        traceback.print_exc()
        return 1
    
    # Load dataset
    try:
        dataset = load_dataset(dataset_path)
    except Exception as e:
        print(f"Error: Failed to load dataset: {e}")
        return 1
    
    # Limit to 1 example for quick test
    dataset = dataset[:1]
    
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
            task_name=task_name,
            mode="baseline",
        )
        
        print()
        print("=" * 60)
        print("✓ tiny-gpt2 CPU-only smoke test completed")
        print("=" * 60)
        print(f"Device used: {device}")
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

