#!/usr/bin/env python3
"""
Qwen2.5 Rival Experiments Runner

This script runs controlled experiments comparing Qwen2.5 models against existing
baselines (TinyLlama-1.1B, Phi-3-mini-3.8B, Mistral-7B) with fair comparison guarantees.

Key features:
- Fixed SciQ slice (n=50, deterministic order)
- Same prompt template across all models
- Same decoding settings (greedy, max_new_tokens=256)
- Same WM token budgets (256/512)
- Stability over time diagnostic (accuracy vs example index)
- Paper-ready outputs (JSON, CSV, figures, LaTeX)

Usage:
    # Run all experiments
    python -m scripts.run_qwen_rivals

    # Run specific model class only
    python -m scripts.run_qwen_rivals --model-class small
    python -m scripts.run_qwen_rivals --model-class medium
    python -m scripts.run_qwen_rivals --model-class large

    # Dry run (show what would be run)
    python -m scripts.run_qwen_rivals --dry-run

    # Run with custom config
    python -m scripts.run_qwen_rivals --config configs/qwen_rivals_contract.yaml
"""

import argparse
import csv
import json
import os
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

try:
    import yaml
except ImportError:
    print("Error: PyYAML is required. Install with: pip install pyyaml")
    sys.exit(1)

try:
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    import seaborn as sns
    PLOTTING_AVAILABLE = True
except ImportError:
    PLOTTING_AVAILABLE = False
    print("Warning: matplotlib/seaborn not available. Figures will not be generated.")

from edge_slm_ace.utils.config import get_model_config, get_task_config, ModelConfig
from edge_slm_ace.models.model_manager import load_model_and_tokenizer, count_tokens
from edge_slm_ace.memory.playbook import Playbook, ScoringParams
from edge_slm_ace.core.runner import run_dataset_baseline, run_dataset_ace
from edge_slm_ace.utils.device_utils import get_device, resolve_device_override
from edge_slm_ace.utils.metrics import PeakMemoryTracker, SemanticEvaluator


# Default contract configuration
DEFAULT_CONFIG = {
    "experiment_contract": {
        "dataset": {
            "task_name": "sciq_test",
            "domain": "science",
            "slice_size": 50,
            "slice_start": 0,
            "deterministic_order": True,
            "seed": 42,
        },
        "generation": {
            "max_new_tokens": 256,
            "temperature": 0.0,
            "top_p": 1.0,
            "seed": 42,
        },
        "reflection": {
            "reflect_on_incorrect": True,
            "reflect_on_correct_every_n": 5,
            "prune_every_n": 10,
            "max_entries_per_domain": 32,
        },
        "working_memory_budgets": [256, 512],
    },
    "models": {
        "small": [
            {"name": "tinyllama-1.1b", "hf_id": "TinyLlama/TinyLlama-1.1B-Chat-v1.0", "params": "1.1B", "family": "llama"},
            {"name": "qwen-2.5-1.5b", "hf_id": "Qwen/Qwen2.5-1.5B-Instruct", "params": "1.5B", "family": "qwen"},
        ],
        "medium": [
            {"name": "phi-3-mini", "hf_id": "microsoft/Phi-3-mini-4k-instruct", "params": "3.8B", "family": "phi"},
            {"name": "qwen-2.5-3b", "hf_id": "Qwen/Qwen2.5-3B-Instruct", "params": "3B", "family": "qwen"},
        ],
        "large": [
            {"name": "mistral-7b", "hf_id": "mistralai/Mistral-7B-Instruct-v0.3", "params": "7B", "family": "mistral"},
            {"name": "qwen-2.5-7b", "hf_id": "Qwen/Qwen2.5-7B-Instruct", "params": "7B", "family": "qwen"},
        ],
    },
    "modes": [
        {"name": "baseline", "mode": "baseline"},
        {"name": "tinyace_wm_256", "mode": "ace", "ace_mode": "ace_working_memory", "working_memory_token_budget": 256},
        {"name": "tinyace_wm_512", "mode": "ace", "ace_mode": "ace_working_memory", "working_memory_token_budget": 512},
    ],
    "output": {
        "results_dir": "results/qwen_rivals",
        "paper_snippets_dir": "paper_snippets",
        "figures_dir": "results/qwen_rivals/figures",
    },
}


def load_config(config_path: Optional[str] = None) -> Dict[str, Any]:
    """Load experiment configuration from YAML file or use defaults."""
    if config_path and Path(config_path).exists():
        with open(config_path, "r", encoding="utf-8") as f:
            config = yaml.safe_load(f)
        return config
    return DEFAULT_CONFIG


def load_dataset_slice(task_name: str, slice_size: int, slice_start: int = 0) -> List[Dict]:
    """
    Load a fixed slice of the dataset for fair comparison.
    
    Args:
        task_name: Task name from registry.
        slice_size: Number of examples to use.
        slice_start: Starting index.
        
    Returns:
        List of dataset examples.
    """
    task_config = get_task_config(task_name)
    dataset_path_str = task_config["path"]
    
    # Resolve relative paths
    if not Path(dataset_path_str).is_absolute():
        repo_root = Path(__file__).parent.parent
        dataset_path_str = str(repo_root / dataset_path_str)
    
    dataset_path = Path(dataset_path_str)
    
    with open(dataset_path, "r", encoding="utf-8") as f:
        if dataset_path.suffix.lower() == ".jsonl":
            dataset = [json.loads(line.strip()) for line in f if line.strip()]
        else:
            dataset = json.load(f)
    
    # Extract fixed slice
    end_idx = min(slice_start + slice_size, len(dataset))
    dataset = dataset[slice_start:end_idx]
    
    print(f"Loaded {len(dataset)} examples from {dataset_path} (slice [{slice_start}:{end_idx}])")
    return dataset


def run_single_experiment(
    model_config: Dict[str, Any],
    mode_config: Dict[str, Any],
    dataset: List[Dict],
    contract: Dict[str, Any],
    device: str,
    output_dir: Path,
) -> Tuple[List[Dict], Dict[str, Any]]:
    """
    Run a single experiment and return results with stability tracking.
    
    Returns:
        Tuple of (per_example_results, summary_metrics)
    """
    model_id = model_config["hf_id"]
    model_name = model_config["name"]
    mode_name = mode_config["name"]
    
    print(f"\n{'='*60}")
    print(f"Running: {model_name} / {mode_name}")
    print(f"{'='*60}")
    
    # Get generation parameters from contract
    gen_params = contract["generation"]
    
    # Create model config with contract parameters
    config = ModelConfig(
        model_id=model_id,
        max_new_tokens=gen_params["max_new_tokens"],
        temperature=gen_params["temperature"],
        top_p=gen_params["top_p"],
    )
    
    # Resolve device
    device_obj, _ = resolve_device_override(device, model_id=model_id)
    
    # Track memory and time
    memory_tracker = PeakMemoryTracker()
    start_time = time.time()
    
    with memory_tracker:
        # Load model
        print(f"Loading model: {model_id}")
        model, tokenizer = load_model_and_tokenizer(model_id, device_override=device)
        
        # Log tokenization comparison (for tokenizer efficiency analysis)
        sample_prompt = "Context: A sample context.\n\nQuestion: What is the answer?\n\nAnswer:"
        prompt_tokens = count_tokens(tokenizer, sample_prompt)
        print(f"Tokenizer check: '{sample_prompt[:50]}...' = {prompt_tokens} tokens")
        
        # Run evaluation based on mode
        task_name = contract["dataset"]["task_name"]
        domain = contract["dataset"]["domain"]
        
        if mode_config["mode"] == "baseline":
            results, summary = run_dataset_baseline(
                model=model,
                tokenizer=tokenizer,
                dataset=dataset,
                domain=domain,
                config=config,
                model_id=model_id,
                task_name=task_name,
                mode="baseline",
            )
        else:  # ACE mode
            # Setup playbook
            token_budget = mode_config.get("working_memory_token_budget", 256)
            playbook_path = output_dir / f"playbook_{model_name}_{mode_name}.jsonl"
            playbook = Playbook(token_budget=token_budget)
            
            ace_mode = mode_config.get("ace_mode", "ace_working_memory")
            
            results, summary = run_dataset_ace(
                model=model,
                tokenizer=tokenizer,
                dataset=dataset,
                domain=domain,
                config=config,
                playbook=playbook,
                playbook_path=playbook_path,
                model_id=model_id,
                task_name=task_name,
                mode="ace",
                ace_mode=ace_mode,
                token_budget=token_budget,
                top_k=5,
                reflect_on_correct_every_n=contract["reflection"]["reflect_on_correct_every_n"],
                prune_every_n=contract["reflection"]["prune_every_n"],
                max_entries_per_domain=contract["reflection"]["max_entries_per_domain"],
            )
    
    wall_time = time.time() - start_time
    
    # Add model metadata to each result for stability tracking
    for idx, result in enumerate(results):
        result["model_name"] = model_name
        result["model_family"] = model_config.get("family", "unknown")
        result["model_params"] = model_config.get("params", "unknown")
        result["mode_name"] = mode_name
        result["example_idx"] = idx + 1  # 1-indexed for clarity
        result["wm_tokens"] = mode_config.get("working_memory_token_budget", 0)
    
    # Enhance summary
    summary["model_name"] = model_name
    summary["model_id"] = model_id
    summary["model_family"] = model_config.get("family", "unknown")
    summary["model_params"] = model_config.get("params", "unknown")
    summary["mode_name"] = mode_name
    summary["wm_tokens"] = mode_config.get("working_memory_token_budget", 0)
    summary["wall_time_seconds"] = wall_time
    summary["peak_memory_mb"] = memory_tracker.peak_memory_mb
    summary["peak_gpu_memory_mb"] = memory_tracker.peak_gpu_memory_mb
    
    # Compute mean tokens
    if results:
        summary["avg_prompt_tokens"] = np.mean([r.get("prompt_tokens", 0) for r in results])
        summary["avg_output_tokens"] = np.mean([r.get("output_tokens", 0) for r in results])
        summary["avg_total_tokens"] = summary["avg_prompt_tokens"] + summary["avg_output_tokens"]
    
    print(f"Completed: OMA={summary.get('oma_accuracy', summary.get('accuracy', 0)):.3f}, "
          f"Latency={summary.get('avg_latency_ms', 0):.0f}ms, "
          f"Wall time={wall_time:.1f}s")
    
    return results, summary


def compute_stability_metrics(results: List[Dict]) -> pd.DataFrame:
    """
    Compute stability over time metrics (accuracy vs example index).
    
    This addresses the professor's point #3: separate memory drift from reasoning failure.
    
    Args:
        results: List of per-example results from all experiments.
        
    Returns:
        DataFrame with columns: model, mode, idx, correct, cumulative_accuracy
    """
    stability_rows = []
    
    # Group by model and mode
    df = pd.DataFrame(results)
    
    for (model_name, mode_name), group in df.groupby(["model_name", "mode_name"]):
        group = group.sort_values("example_idx").reset_index(drop=True)
        
        cumulative_correct = 0
        for idx, row in group.iterrows():
            # Use oma_correct if available, else is_correct
            correct = row.get("oma_correct", row.get("is_correct", 0))
            if pd.isna(correct):
                correct = 0
            correct = int(correct)
            
            cumulative_correct += correct
            cumulative_accuracy = cumulative_correct / (idx + 1)
            
            stability_rows.append({
                "model": model_name,
                "config": mode_name,
                "idx": int(row["example_idx"]),
                "correct": correct,
                "cumulative_accuracy": cumulative_accuracy,
            })
    
    return pd.DataFrame(stability_rows)


def generate_sweetspot_bar_chart(
    summaries: List[Dict],
    output_path: Path,
    title: str = "Baseline vs TinyACE WM-512 Across Model Scales"
):
    """
    Generate bar chart comparing baseline vs TinyACE WM-512 across model scales.
    
    This is the main comparison figure for the paper.
    """
    if not PLOTTING_AVAILABLE:
        print("Skipping bar chart: matplotlib not available")
        return
    
    df = pd.DataFrame(summaries)
    
    # Filter to baseline and WM-512 only
    df = df[df["mode_name"].isin(["baseline", "tinyace_wm_512"])]
    
    # Get OMA accuracy
    df["oma"] = df.apply(lambda r: r.get("oma_accuracy", r.get("accuracy", 0)), axis=1)
    
    # Create figure
    fig, ax = plt.subplots(figsize=(12, 6))
    
    # Prepare data for grouped bar chart
    models = df["model_name"].unique()
    x = np.arange(len(models))
    width = 0.35
    
    baseline_oma = []
    tinyace_oma = []
    
    for model in models:
        model_df = df[df["model_name"] == model]
        baseline_row = model_df[model_df["mode_name"] == "baseline"]
        tinyace_row = model_df[model_df["mode_name"] == "tinyace_wm_512"]
        
        baseline_oma.append(baseline_row["oma"].values[0] if len(baseline_row) > 0 else 0)
        tinyace_oma.append(tinyace_row["oma"].values[0] if len(tinyace_row) > 0 else 0)
    
    # Create bars
    bars1 = ax.bar(x - width/2, baseline_oma, width, label='Baseline', color='#1f77b4', alpha=0.8)
    bars2 = ax.bar(x + width/2, tinyace_oma, width, label='TinyACE WM-512', color='#ff7f0e', alpha=0.8)
    
    # Add value labels on bars
    def add_labels(bars):
        for bar in bars:
            height = bar.get_height()
            ax.annotate(f'{height:.2f}',
                       xy=(bar.get_x() + bar.get_width() / 2, height),
                       xytext=(0, 3),
                       textcoords="offset points",
                       ha='center', va='bottom', fontsize=9)
    
    add_labels(bars1)
    add_labels(bars2)
    
    # Labels and formatting
    ax.set_xlabel('Model', fontsize=12)
    ax.set_ylabel('OMA (Option-Mapped Accuracy)', fontsize=12)
    ax.set_title(title, fontsize=14, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(models, rotation=45, ha='right')
    ax.legend(loc='upper right')
    ax.set_ylim([0, 1.05])
    ax.grid(True, alpha=0.3, axis='y')
    
    plt.tight_layout()
    
    # Save in multiple formats
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path.with_suffix('.pdf'), format='pdf', bbox_inches='tight')
    plt.savefig(output_path.with_suffix('.png'), format='png', bbox_inches='tight', dpi=300)
    plt.close()
    
    print(f"Saved sweetspot bar chart to {output_path}")


def generate_stability_curves(
    stability_df: pd.DataFrame,
    output_path: Path,
    title: str = "Accuracy vs Example Index (Stability Analysis)"
):
    """
    Generate stability curves showing accuracy vs example index.
    
    This addresses the professor's point about separating memory drift from reasoning failure.
    """
    if not PLOTTING_AVAILABLE:
        print("Skipping stability curves: matplotlib not available")
        return
    
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    
    # Get unique models grouped by size class
    models = stability_df["model"].unique()
    
    # Define size classes
    small_models = [m for m in models if "1.1b" in m.lower() or "1.5b" in m.lower()]
    medium_models = [m for m in models if "3b" in m.lower() or "phi" in m.lower()]
    large_models = [m for m in models if "7b" in m.lower()]
    
    model_groups = [
        ("Small (~1-2B)", small_models),
        ("Medium (~3-4B)", medium_models),
        ("Large (~7B)", large_models),
    ]
    
    colors = plt.cm.Set2(np.linspace(0, 1, 10))
    
    for ax_idx, (group_name, group_models) in enumerate(model_groups):
        ax = axes[ax_idx]
        
        color_idx = 0
        for model in group_models:
            model_df = stability_df[stability_df["model"] == model]
            
            for config in model_df["config"].unique():
                config_df = model_df[model_df["config"] == config]
                config_df = config_df.sort_values("idx")
                
                label = f"{model} ({config})"
                linestyle = '-' if config == "baseline" else ('--' if "256" in config else ':')
                
                ax.plot(config_df["idx"], config_df["cumulative_accuracy"],
                       label=label, color=colors[color_idx % len(colors)],
                       linestyle=linestyle, linewidth=2, marker='o', markersize=3)
                
                color_idx += 1
        
        ax.set_xlabel('Example Index', fontsize=11)
        ax.set_ylabel('Cumulative Accuracy', fontsize=11)
        ax.set_title(f'{group_name} Models', fontsize=12, fontweight='bold')
        ax.legend(loc='lower right', fontsize=8)
        ax.set_ylim([0, 1.05])
        ax.grid(True, alpha=0.3)
    
    plt.suptitle(title, fontsize=14, fontweight='bold', y=1.02)
    plt.tight_layout()
    
    # Save in multiple formats
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path.with_suffix('.pdf'), format='pdf', bbox_inches='tight')
    plt.savefig(output_path.with_suffix('.png'), format='png', bbox_inches='tight', dpi=300)
    plt.close()
    
    print(f"Saved stability curves to {output_path}")


def generate_latex_table(summaries: List[Dict], output_path: Path):
    """
    Generate LaTeX table snippet for the paper.
    
    Creates a table with: Model, Params, Config, OMA, SemSim, Latency(s), WM Tokens
    """
    df = pd.DataFrame(summaries)
    
    # Sort by model params then model name
    param_order = {"1.1B": 0, "1.5B": 1, "3B": 2, "3.8B": 3, "7B": 4}
    df["param_order"] = df["model_params"].map(lambda x: param_order.get(x, 99))
    df = df.sort_values(["param_order", "model_name", "mode_name"])
    
    latex_lines = [
        r"\begin{table}[h]",
        r"\centering",
        r"\caption{Qwen2.5 Rival Comparison on SciQ (n=50). OMA = Option-Mapped Accuracy, SemSim = Semantic Similarity.}",
        r"\label{tab:qwen_rivals}",
        r"\begin{tabular}{llcccc}",
        r"\toprule",
        r"Model & Config & OMA & SemSim & Latency (s) & WM Tokens \\",
        r"\midrule",
    ]
    
    prev_model = None
    for _, row in df.iterrows():
        model = row["model_name"]
        config = row["mode_name"].replace("_", " ").replace("tinyace ", "TinyACE ")
        oma = row.get("oma_accuracy", row.get("accuracy", 0))
        semsim = row.get("avg_semantic_similarity", 0) or 0
        latency = row.get("avg_latency_sec", row.get("avg_latency_ms", 0) / 1000)
        wm_tokens = row.get("wm_tokens", 0)
        
        # Add horizontal line between model groups
        if prev_model and model != prev_model:
            latex_lines.append(r"\midrule")
        prev_model = model
        
        latex_lines.append(
            f"{model} & {config} & {oma:.3f} & {semsim:.3f} & {latency:.2f} & {wm_tokens} \\\\"
        )
    
    latex_lines.extend([
        r"\bottomrule",
        r"\end{tabular}",
        r"\end{table}",
    ])
    
    latex_content = "\n".join(latex_lines)
    
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        f.write(latex_content)
    
    print(f"Saved LaTeX table to {output_path}")
    return latex_content


def generate_latex_summary_text(summaries: List[Dict], output_path: Path):
    """
    Generate summary text paragraph for the paper.
    """
    df = pd.DataFrame(summaries)
    
    # Find key comparisons
    text_lines = [
        "% Qwen2.5 Rival Experiment Summary",
        "% Auto-generated by run_qwen_rivals.py",
        "",
        "\\paragraph{Qwen2.5 Comparison Results}",
        "",
    ]
    
    # Small model comparison
    small_df = df[df["model_params"].isin(["1.1B", "1.5B"])]
    if not small_df.empty:
        for model in small_df["model_name"].unique():
            model_df = small_df[small_df["model_name"] == model]
            baseline = model_df[model_df["mode_name"] == "baseline"]
            wm512 = model_df[model_df["mode_name"] == "tinyace_wm_512"]
            if len(baseline) > 0 and len(wm512) > 0:
                baseline_oma = baseline["oma_accuracy"].values[0] if "oma_accuracy" in baseline.columns else baseline["accuracy"].values[0]
                wm512_oma = wm512["oma_accuracy"].values[0] if "oma_accuracy" in wm512.columns else wm512["accuracy"].values[0]
                diff = wm512_oma - baseline_oma
                direction = "improvement" if diff > 0 else "degradation"
                text_lines.append(
                    f"For {model} (small class), TinyACE WM-512 shows a {abs(diff)*100:.1f}\\% {direction} "
                    f"(baseline: {baseline_oma*100:.1f}\\%, TinyACE: {wm512_oma*100:.1f}\\%)."
                )
    
    # Medium model comparison
    medium_df = df[df["model_params"].isin(["3B", "3.8B"])]
    if not medium_df.empty:
        for model in medium_df["model_name"].unique():
            model_df = medium_df[medium_df["model_name"] == model]
            baseline = model_df[model_df["mode_name"] == "baseline"]
            wm512 = model_df[model_df["mode_name"] == "tinyace_wm_512"]
            if len(baseline) > 0 and len(wm512) > 0:
                baseline_oma = baseline["oma_accuracy"].values[0] if "oma_accuracy" in baseline.columns else baseline["accuracy"].values[0]
                wm512_oma = wm512["oma_accuracy"].values[0] if "oma_accuracy" in wm512.columns else wm512["accuracy"].values[0]
                diff = wm512_oma - baseline_oma
                direction = "improvement" if diff > 0 else "degradation"
                text_lines.append(
                    f"For {model} (medium class), TinyACE WM-512 shows a {abs(diff)*100:.1f}\\% {direction} "
                    f"(baseline: {baseline_oma*100:.1f}\\%, TinyACE: {wm512_oma*100:.1f}\\%)."
                )
    
    # Large model comparison
    large_df = df[df["model_params"] == "7B"]
    if not large_df.empty:
        for model in large_df["model_name"].unique():
            model_df = large_df[large_df["model_name"] == model]
            baseline = model_df[model_df["mode_name"] == "baseline"]
            wm512 = model_df[model_df["mode_name"] == "tinyace_wm_512"]
            if len(baseline) > 0 and len(wm512) > 0:
                baseline_oma = baseline["oma_accuracy"].values[0] if "oma_accuracy" in baseline.columns else baseline["accuracy"].values[0]
                wm512_oma = wm512["oma_accuracy"].values[0] if "oma_accuracy" in wm512.columns else wm512["accuracy"].values[0]
                diff = wm512_oma - baseline_oma
                direction = "improvement" if diff > 0 else "degradation"
                text_lines.append(
                    f"For {model} (large class), TinyACE WM-512 shows a {abs(diff)*100:.1f}\\% {direction} "
                    f"(baseline: {baseline_oma*100:.1f}\\%, TinyACE: {wm512_oma*100:.1f}\\%)."
                )
    
    text_content = "\n".join(text_lines)
    
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        f.write(text_content)
    
    print(f"Saved LaTeX summary text to {output_path}")
    return text_content


def main():
    """Main entry point for Qwen rivals experiments."""
    parser = argparse.ArgumentParser(
        description="Run Qwen2.5 rival experiments with fair comparison guarantees.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    
    parser.add_argument(
        "--config",
        type=str,
        default="configs/qwen_rivals_contract.yaml",
        help="Path to experiment configuration YAML (default: configs/qwen_rivals_contract.yaml)",
    )
    parser.add_argument(
        "--model-class",
        type=str,
        choices=["small", "medium", "large", "all"],
        default="all",
        help="Which model class to run (default: all)",
    )
    parser.add_argument(
        "--models",
        type=str,
        nargs="+",
        default=None,
        help="Specific model names to run (overrides --model-class)",
    )
    parser.add_argument(
        "--modes",
        type=str,
        nargs="+",
        default=None,
        help="Specific modes to run (default: all modes)",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda",
        choices=["cuda", "mps", "cpu"],
        help="Device to use (default: cuda)",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print experiment plan without running",
    )
    parser.add_argument(
        "--skip-plots",
        action="store_true",
        help="Skip generating plots (useful if matplotlib unavailable)",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=None,
        help="Override output directory",
    )
    
    args = parser.parse_args()
    
    # Load configuration
    config = load_config(args.config)
    contract = config.get("experiment_contract", DEFAULT_CONFIG["experiment_contract"])
    
    # Determine output directories
    output_config = config.get("output", DEFAULT_CONFIG["output"])
    results_dir = Path(args.output_dir or output_config.get("results_dir", "results/qwen_rivals"))
    paper_snippets_dir = Path(output_config.get("paper_snippets_dir", "paper_snippets"))
    figures_dir = Path(output_config.get("figures_dir", "results/qwen_rivals/figures"))
    
    # Create directories
    results_dir.mkdir(parents=True, exist_ok=True)
    paper_snippets_dir.mkdir(parents=True, exist_ok=True)
    figures_dir.mkdir(parents=True, exist_ok=True)
    
    # Determine which models to run
    all_models = []
    model_classes = config.get("models", DEFAULT_CONFIG["models"])
    
    if args.models:
        # Specific models requested
        for model_class in model_classes.values():
            for model in model_class:
                if model["name"] in args.models:
                    all_models.append(model)
    elif args.model_class == "all":
        # All models
        for model_class in model_classes.values():
            all_models.extend(model_class)
    else:
        # Specific class
        all_models = model_classes.get(args.model_class, [])
    
    if not all_models:
        print("Error: No models selected. Check --model-class or --models arguments.")
        return 1
    
    # Determine which modes to run
    all_modes = config.get("modes", DEFAULT_CONFIG["modes"])
    if args.modes:
        all_modes = [m for m in all_modes if m["name"] in args.modes]
    
    if not all_modes:
        print("Error: No modes selected. Check --modes argument.")
        return 1
    
    # Print experiment plan
    print("\n" + "=" * 70)
    print("Qwen2.5 Rival Experiments")
    print("=" * 70)
    print(f"Config: {args.config}")
    print(f"Device: {args.device}")
    print(f"Output: {results_dir}")
    print(f"\nModels ({len(all_models)}):")
    for m in all_models:
        print(f"  - {m['name']} ({m.get('params', 'unknown')}) [{m.get('family', 'unknown')}]")
    print(f"\nModes ({len(all_modes)}):")
    for m in all_modes:
        print(f"  - {m['name']}")
    print(f"\nDataset: {contract['dataset']['task_name']} (n={contract['dataset']['slice_size']})")
    print(f"Total experiments: {len(all_models) * len(all_modes)}")
    print("=" * 70)
    
    if args.dry_run:
        print("\n[DRY RUN] Would run the above experiments.")
        return 0
    
    # Load dataset slice
    dataset = load_dataset_slice(
        task_name=contract["dataset"]["task_name"],
        slice_size=contract["dataset"]["slice_size"],
        slice_start=contract["dataset"]["slice_start"],
    )
    
    # Run all experiments
    all_results = []
    all_summaries = []
    
    start_time = time.time()
    
    for model_config in all_models:
        for mode_config in all_modes:
            try:
                results, summary = run_single_experiment(
                    model_config=model_config,
                    mode_config=mode_config,
                    dataset=dataset,
                    contract=contract,
                    device=args.device,
                    output_dir=results_dir,
                )
                
                all_results.extend(results)
                all_summaries.append(summary)
                
                # Save intermediate results
                model_name = model_config["name"]
                mode_name = mode_config["name"]
                
                # Save per-experiment results
                exp_results_path = results_dir / f"{model_name}_{mode_name}_results.jsonl"
                with open(exp_results_path, "w", encoding="utf-8") as f:
                    for r in results:
                        f.write(json.dumps(r, default=str) + "\n")
                
            except Exception as e:
                print(f"Error running {model_config['name']}/{mode_config['name']}: {e}")
                import traceback
                traceback.print_exc()
                continue
    
    total_time = time.time() - start_time
    
    # Save consolidated results
    print("\n" + "=" * 70)
    print("Generating outputs...")
    print("=" * 70)
    
    # 1. Save results JSON
    results_json_path = results_dir / "results_models_qwen.json"
    with open(results_json_path, "w", encoding="utf-8") as f:
        json.dump(all_summaries, f, indent=2, default=str)
    print(f"Saved results JSON to {results_json_path}")
    
    # 2. Compute and save stability metrics
    stability_df = compute_stability_metrics(all_results)
    stability_csv_path = results_dir / "results_stability_qwen.csv"
    stability_df.to_csv(stability_csv_path, index=False)
    print(f"Saved stability CSV to {stability_csv_path}")
    
    # 3. Generate figures
    if not args.skip_plots and PLOTTING_AVAILABLE:
        # Sweetspot bar chart
        generate_sweetspot_bar_chart(
            all_summaries,
            figures_dir / "sweetspot_qwen_bar",
            title="Baseline vs TinyACE WM-512: Including Qwen2.5 Rivals"
        )
        
        # Stability curves
        generate_stability_curves(
            stability_df,
            figures_dir / "stability_curves_qwen",
            title="Accuracy vs Example Index (Stability Analysis)"
        )
    
    # 4. Generate LaTeX snippets
    latex_table = generate_latex_table(all_summaries, paper_snippets_dir / "qwen_rivals_table.tex")
    latex_summary = generate_latex_summary_text(all_summaries, paper_snippets_dir / "qwen_rivals_summary.tex")
    
    # Print final summary
    print("\n" + "=" * 70)
    print("Experiment Complete!")
    print("=" * 70)
    print(f"Total time: {total_time:.1f}s ({total_time/60:.1f} min)")
    print(f"Experiments run: {len(all_summaries)}")
    print(f"\nOutputs:")
    print(f"  - Results JSON: {results_json_path}")
    print(f"  - Stability CSV: {stability_csv_path}")
    print(f"  - Figures: {figures_dir}/")
    print(f"  - LaTeX snippets: {paper_snippets_dir}/")
    print("=" * 70)
    
    # Print LaTeX table for easy copy-paste
    print("\n" + "=" * 70)
    print("LaTeX Table (copy to Overleaf):")
    print("=" * 70)
    print(latex_table)
    
    return 0


if __name__ == "__main__":
    sys.exit(main())
