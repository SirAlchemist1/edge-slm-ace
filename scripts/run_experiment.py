#!/usr/bin/env python3
"""CLI entrypoint for running baseline or ACE-style experiments."""

import argparse
import json
from pathlib import Path

import pandas as pd

from slm_ace.config import get_model_config, get_task_config, ModelConfig
from slm_ace.model_manager import load_model_and_tokenizer
from slm_ace.playbook import Playbook
from slm_ace.runner import run_dataset_baseline, run_dataset_ace, run_dataset_self_refine
from slm_ace.utils import get_device


def load_dataset(path: Path) -> list[dict]:
    """
    Load a dataset from a JSON file.
    
    Expected format: list of dicts with keys: id, question, answer, (optional) context, domain.
    """
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def main():
    """
    Main CLI entrypoint for running experiments.
    
    This script handles:
    - Model loading (with device detection)
    - Dataset loading
    - Running baseline or ACE evaluation
    - Saving results to CSV
    
    Example usage:
        python -m scripts.run_experiment \\
            --model-id sshleifer/tiny-gpt2 \\
            --dataset-path data/tasks/tatqa_tiny.json \\
            --domain finance \\
            --mode baseline \\
            --output-path results/tatqa_baseline.csv
    """
    parser = argparse.ArgumentParser(
        description="Run baseline or ACE-style evaluation on a dataset.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    
    parser.add_argument(
        "--model-id",
        type=str,
        required=True,
        help="HuggingFace model ID or key from config (e.g., 'phi3-mini', 'mistral-7b')",
    )
    parser.add_argument(
        "--dataset-path",
        type=str,
        default=None,
        help="Path to JSON dataset file (required if --task-name not provided)",
    )
    parser.add_argument(
        "--task-name",
        type=str,
        default=None,
        help="Task name from registry (e.g., 'tatqa_tiny', 'medqa_tiny', 'iot_tiny'). If provided, overrides --dataset-path and --domain",
    )
    parser.add_argument(
        "--domain",
        type=str,
        default=None,
        help="Domain name (e.g., 'finance', 'medical', 'iot'). Required if --task-name not provided",
    )
    parser.add_argument(
        "--mode",
        type=str,
        choices=["baseline", "ace", "self_refine"],
        required=True,
        help="Run mode: 'baseline', 'ace', or 'self_refine'",
    )
    parser.add_argument(
        "--playbook-path",
        type=str,
        default=None,
        help="Path to playbook JSONL file (required for ACE mode)",
    )
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
    parser.add_argument(
        "--output-path",
        type=str,
        required=True,
        help="Path to save results CSV",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Limit number of examples to process (useful for quick testing)",
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
        help="Automatically regenerate plots after evaluation completes (requires tinyace_plots.py)",
    )
    
    args = parser.parse_args()
    
    try:
        # Resolve dataset path and domain from task-name or explicit args
        if args.task_name:
            try:
                task_config = get_task_config(args.task_name)
                dataset_path_str = task_config["path"]
                domain = task_config["domain"]
                task_name = args.task_name
                # Resolve relative paths relative to repo root
                if not Path(dataset_path_str).is_absolute():
                    repo_root = Path(__file__).parent.parent
                    dataset_path_str = str(repo_root / dataset_path_str)
            except KeyError as e:
                print(f"Error: {e}")
                return 1
        else:
            if args.dataset_path is None:
                parser.error("Either --task-name or --dataset-path must be provided")
            if args.domain is None:
                parser.error("Either --task-name or --domain must be provided")
            dataset_path_str = args.dataset_path
            domain = args.domain
            task_name = None  # Will be inferred from dataset path if possible
        
        # Validate ACE mode requirements
        if args.mode == "ace" and args.playbook_path is None:
            parser.error("--playbook-path is required for ACE mode")
        
        # Load model config
        try:
            config = get_model_config(args.model_id)
        except Exception as e:
            print(f"Error: Failed to load model config for '{args.model_id}': {e}")
            print("Hint: Use a pre-configured key (e.g., 'tiny-gpt2', 'phi3-mini') or a valid HuggingFace model ID")
            return 1
        
        # Override config if provided
        if args.max_new_tokens is not None:
            config.max_new_tokens = args.max_new_tokens
        if args.temperature is not None:
            config.temperature = args.temperature
        if args.top_p is not None:
            config.top_p = args.top_p
        
        # Resolve device (with override support)
        # Note: load_model_and_tokenizer will handle tiny-gpt2 CPU override internally
        if args.device:
            from slm_ace.utils import resolve_device_override
            device, forced = resolve_device_override(args.device, model_id=config.model_id)
            # Pass the resolved device string, not the original override
            device_override_str = str(device.type) if not forced else "cpu"
        else:
            device = get_device()
            device_override_str = None
        print(f"Using device: {device} (override: {args.device or 'auto'})")
        
        # Load model and tokenizer
        print(f"Loading model: {config.model_id}")
        try:
            model, tokenizer = load_model_and_tokenizer(
                config.model_id,
                device=device,
                device_override=device_override_str,
            )
            print("Model loaded successfully.")
        except Exception as e:
            print(f"Error: Failed to load model '{config.model_id}': {e}")
            print("Hint: Make sure the model ID is correct and you have internet access for downloading")
            return 1
        
        # Load dataset
        dataset_path = Path(dataset_path_str)
        if not dataset_path.exists():
            print(f"Error: Dataset not found: {dataset_path}")
            print(f"Hint: Check that the path is correct and the file exists")
            return 1
        
        try:
            dataset = load_dataset(dataset_path)
        except json.JSONDecodeError as e:
            print(f"Error: Invalid JSON in dataset file: {e}")
            return 1
        except Exception as e:
            print(f"Error: Failed to load dataset: {e}")
            return 1
        
        if len(dataset) == 0:
            print("Warning: Dataset is empty")
        
        # Apply limit if specified (for quick testing)
        if args.limit is not None:
            dataset = dataset[:args.limit]
            print(f"Limited to {len(dataset)} examples (--limit {args.limit})")
        
        print(f"Loaded {len(dataset)} examples from {dataset_path}")
        
        # Run evaluation
        if args.mode == "baseline":
            print("Running baseline evaluation...")
            try:
                results, summary = run_dataset_baseline(
                    model=model,
                    tokenizer=tokenizer,
                    dataset=dataset,
                    domain=domain,
                    config=config,
                    model_id=config.model_id,
                    task_name=task_name or dataset_path.stem,
                    mode=args.mode,
                )
            except Exception as e:
                print(f"Error during baseline evaluation: {e}")
                import traceback
                traceback.print_exc()
                return 1
        elif args.mode == "self_refine":
            print("Running self-refinement evaluation...")
            try:
                results, summary = run_dataset_self_refine(
                    model=model,
                    tokenizer=tokenizer,
                    dataset=dataset,
                    domain=domain,
                    config=config,
                    model_id=config.model_id,
                    task_name=task_name or dataset_path.stem,
                    mode=args.mode,
                )
            except Exception as e:
                print(f"Error during self-refinement evaluation: {e}")
                import traceback
                traceback.print_exc()
                return 1
        else:  # ACE mode
            print("Running ACE-style evaluation...")
            playbook_path = Path(args.playbook_path)
            
            # Load or create playbook
            try:
                if playbook_path.exists():
                    print(f"Loading playbook from {playbook_path}")
                    playbook = Playbook.load(playbook_path)
                else:
                    print(f"Creating new playbook at {playbook_path}")
                    playbook = Playbook()
            except Exception as e:
                print(f"Error: Failed to load/create playbook: {e}")
                return 1
            
            try:
                results, summary = run_dataset_ace(
                    model=model,
                    tokenizer=tokenizer,
                    dataset=dataset,
                    domain=domain,
                    config=config,
                    playbook=playbook,
                    playbook_path=playbook_path,
                    model_id=config.model_id,
                    task_name=task_name or dataset_path.stem,
                    mode=args.mode,
                )
                print(f"Playbook saved with {len(playbook.entries)} entries")
            except Exception as e:
                print(f"Error during ACE evaluation: {e}")
                import traceback
                traceback.print_exc()
                return 1
        
        # Save results to CSV
        output_path = Path(args.output_path)
        try:
            output_path.parent.mkdir(parents=True, exist_ok=True)
            df = pd.DataFrame(results)
            df.to_csv(output_path, index=False)
            print(f"Results saved to {output_path}")
        except Exception as e:
            print(f"Error: Failed to save results: {e}")
            return 1
        
        # Print summary
        print("\n" + "=" * 50)
        print("Summary:")
        print("=" * 50)
        for key, value in summary.items():
            print(f"{key}: {value}")
        print("=" * 50)
        
        # Optionally regenerate plots
        if args.auto_plots:
            try:
                import sys
                from pathlib import Path
                # Import tinyace_plots module
                repo_root = Path(__file__).parent.parent
                sys.path.insert(0, str(repo_root))
                from tinyace_plots import main as regenerate_plots
                
                print("\n" + "=" * 50)
                print("Regenerating plots...")
                print("=" * 50)
                regenerate_plots(results_dir="results", output_dir="tinyace_plots")
                print("Plots regenerated successfully.")
            except ImportError as e:
                print(f"\nWarning: Could not import tinyace_plots: {e}")
                print("Skipping plot regeneration. Run manually with: python tinyace_plots.py")
            except Exception as e:
                print(f"\nWarning: Plot regeneration failed: {e}")
                print("You can regenerate plots manually with: python tinyace_plots.py")
        
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

