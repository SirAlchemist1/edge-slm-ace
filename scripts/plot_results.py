#!/usr/bin/env python3
"""Generate plots from aggregated experiment results.

This script reads a summary CSV file and generates visualization plots
comparing accuracy across different modes, models, and tasks.

Usage:
    python -m scripts.plot_results
    python -m scripts.plot_results --summary-csv results/summary.csv --output-dir results/plots/
"""

import argparse
import csv
import json
import sys
from pathlib import Path
from typing import Dict, List, Optional

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# =============================================================================
# Mode label normalization for publication-ready plots
# =============================================================================
MODE_LABELS = {
    "baseline": "Baseline",
    "ace": "ACE",
    "ace_full": "ACE Full",
    "ace_working_memory": "ACE WM",
    "tinyace_wm_256": "TinyACE-256",
    "tinyace_wm_512": "TinyACE-512",
    "tinyace_ablate_no_vagueness": "Ablate: No Vagueness",
    "tinyace_ablate_no_recency": "Ablate: No Recency",
    "tinyace_ablate_no_failure": "Ablate: No Failure",
    "tinyace_fifo": "FIFO Eviction",
    "self_refine": "Self-Refine",
}

# Mode ordering for consistent plot legends
MODE_ORDER = [
    "baseline", "ace_full", "ace_working_memory",
    "tinyace_wm_256", "tinyace_wm_512",
    "tinyace_ablate_no_vagueness", "tinyace_ablate_no_recency",
    "tinyace_ablate_no_failure", "tinyace_fifo", "self_refine",
]


def normalize_mode_label(mode: str) -> str:
    """Convert internal mode name to publication-ready label."""
    return MODE_LABELS.get(mode, mode.replace("_", " ").title())


def get_effective_mode(row: pd.Series) -> str:
    """Get effective mode from row (prefer ace_mode over mode for ACE runs)."""
    if "effective_mode" in row and pd.notna(row["effective_mode"]):
        return row["effective_mode"]
    if "ace_mode" in row and pd.notna(row["ace_mode"]):
        return row["ace_mode"]
    return row.get("mode", "unknown")


def plot_accuracy_by_mode(
    df: pd.DataFrame,
    output_path: Path,
    figsize: tuple = (10, 6),
) -> None:
    """
    Plot accuracy by mode (baseline vs ace_full vs ace_working_memory) for each task.
    
    Args:
        df: DataFrame with columns: task_name, mode, accuracy
        output_path: Path to save the plot.
        figsize: Figure size tuple.
    """
    if "accuracy" not in df.columns:
        print("Warning: Missing 'accuracy' column for accuracy_by_mode plot", file=sys.stderr)
        return
    
    # Filter to rows with valid accuracy
    plot_df = df[df["accuracy"].notna()].copy()
    
    if len(plot_df) == 0:
        print("Warning: No valid accuracy data for accuracy_by_mode plot", file=sys.stderr)
        return
    
    # Use effective_mode if available, otherwise derive from mode/ace_mode
    if "effective_mode" in plot_df.columns:
        plot_df["plot_mode"] = plot_df["effective_mode"]
    elif "ace_mode" in plot_df.columns:
        plot_df["plot_mode"] = plot_df.apply(
            lambda r: r["ace_mode"] if pd.notna(r.get("ace_mode")) else r.get("mode", "unknown"),
            axis=1
        )
    else:
        plot_df["plot_mode"] = plot_df.get("mode", "unknown")
    
    # Group by task and mode
    if "task_name" in plot_df.columns:
        # Multiple tasks: create subplots
        tasks = sorted(plot_df["task_name"].unique())
        n_tasks = len(tasks)
        
        if n_tasks == 0:
            return
        
        fig, axes = plt.subplots(1, n_tasks, figsize=(5 * n_tasks, 6), sharey=True)
        if n_tasks == 1:
            axes = [axes]
        
        for idx, task in enumerate(tasks):
            task_df = plot_df[plot_df["task_name"] == task]
            
            # Group by mode and compute mean accuracy
            mode_accuracy = task_df.groupby("plot_mode")["accuracy"].mean()
            
            # Sort modes by predefined order
            sorted_modes = [m for m in MODE_ORDER if m in mode_accuracy.index]
            sorted_modes += [m for m in mode_accuracy.index if m not in sorted_modes]
            mode_accuracy = mode_accuracy.reindex(sorted_modes)
            
            ax = axes[idx]
            # Use normalized labels for x-axis
            x_labels = [normalize_mode_label(m) for m in mode_accuracy.index]
            bars = ax.bar(x_labels, mode_accuracy.values, alpha=0.7, color=plt.cm.Set2(range(len(x_labels))))
            ax.set_title(f"Task: {task}", fontsize=12)
            ax.set_xlabel("Mode", fontsize=10)
            ax.set_ylabel("Accuracy", fontsize=10)
            ax.set_ylim(0, max(1.0, mode_accuracy.max() * 1.1) if mode_accuracy.max() > 0 else 1.0)
            ax.grid(axis="y", alpha=0.3)
            ax.tick_params(axis='x', rotation=45)
            
            # Add value labels on bars
            for bar in bars:
                height = bar.get_height()
                ax.text(
                    bar.get_x() + bar.get_width() / 2.0,
                    height,
                    f"{height:.3f}",
                    ha="center",
                    va="bottom",
                    fontsize=9,
                )
        
        plt.tight_layout()
    else:
        # Single task or no task column: single plot
        mode_accuracy = plot_df.groupby("plot_mode")["accuracy"].mean()
        
        # Sort modes by predefined order
        sorted_modes = [m for m in MODE_ORDER if m in mode_accuracy.index]
        sorted_modes += [m for m in mode_accuracy.index if m not in sorted_modes]
        mode_accuracy = mode_accuracy.reindex(sorted_modes)
        
        fig, ax = plt.subplots(figsize=figsize)
        x_labels = [normalize_mode_label(m) for m in mode_accuracy.index]
        bars = ax.bar(x_labels, mode_accuracy.values, alpha=0.7, color=plt.cm.Set2(range(len(x_labels))))
        ax.set_title("Accuracy by Mode", fontsize=14)
        ax.set_xlabel("Mode", fontsize=12)
        ax.set_ylabel("Accuracy", fontsize=12)
        ax.set_ylim(0, max(1.0, mode_accuracy.max() * 1.1) if mode_accuracy.max() > 0 else 1.0)
        ax.grid(axis="y", alpha=0.3)
        plt.xticks(rotation=45, ha="right")
        
        # Add value labels
        for bar in bars:
            height = bar.get_height()
            ax.text(
                bar.get_x() + bar.get_width() / 2.0,
                height,
                f"{height:.3f}",
                ha="center",
                va="bottom",
                fontsize=10,
            )
    
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved plot: {output_path}")


def plot_accuracy_by_model_and_mode(
    df: pd.DataFrame,
    output_path: Path,
    figsize: tuple = (12, 6),
) -> None:
    """
    Plot accuracy by model and mode (to compare SLMs).
    
    Args:
        df: DataFrame with columns: model_id, mode, accuracy
        output_path: Path to save the plot.
        figsize: Figure size tuple.
    """
    if "model_id" not in df.columns or "mode" not in df.columns or "accuracy" not in df.columns:
        print("Warning: Missing required columns (model_id, mode, accuracy) for accuracy_by_model_and_mode plot", file=sys.stderr)
        return
    
    # Filter to rows with valid accuracy
    plot_df = df[df["accuracy"].notna()].copy()
    
    if len(plot_df) == 0:
        print("Warning: No valid accuracy data for accuracy_by_model_and_mode plot", file=sys.stderr)
        return
    
    models = sorted(plot_df["model_id"].unique())
    modes = sorted(plot_df["mode"].unique())
    
    if len(models) == 0 or len(modes) == 0:
        return
    
    # Create grouped bar chart
    fig, ax = plt.subplots(figsize=figsize)
    
    # Prepare data: group by model and mode
    x_pos = range(len(models))
    width = 0.8 / len(modes)  # Width of bars
    
    colors = plt.cm.Set3(range(len(modes)))  # Different color for each mode
    
    for mode_idx, mode in enumerate(modes):
        mode_values = []
        for model in models:
            model_mode_df = plot_df[(plot_df["model_id"] == model) & (plot_df["mode"] == mode)]
            if len(model_mode_df) > 0:
                # Average accuracy across all tasks/devices for this model+mode
                avg_accuracy = model_mode_df["accuracy"].mean()
                mode_values.append(avg_accuracy)
            else:
                mode_values.append(0.0)
        
        # Position bars
        x_offset = (mode_idx - len(modes) / 2 + 0.5) * width
        bars = ax.bar(
            [x + x_offset for x in x_pos],
            mode_values,
            width,
            label=mode,
            alpha=0.7,
            color=colors[mode_idx],
        )
        
        # Add value labels
        for bar, val in zip(bars, mode_values):
            if val > 0:
                ax.text(
                    bar.get_x() + bar.get_width() / 2.0,
                    val,
                    f"{val:.3f}",
                    ha="center",
                    va="bottom",
                    fontsize=8,
                )
    
    ax.set_xlabel("Model", fontsize=12)
    ax.set_ylabel("Accuracy", fontsize=12)
    ax.set_title("Accuracy by Model and Mode", fontsize=14)
    ax.set_xticks(x_pos)
    ax.set_xticklabels(models, rotation=45, ha="right")
    ax.legend(title="Mode", fontsize=10)
    ax.set_ylim(0, max(1.0, max([plot_df[plot_df["model_id"] == m]["accuracy"].max() for m in models]) * 1.1))
    ax.grid(axis="y", alpha=0.3)
    
    plt.tight_layout()
    
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved plot: {output_path}")


def main() -> int:
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Generate plots from aggregated experiment results.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Generate all plots from default summary CSV
  python -m scripts.plot_results

  # Generate plots from custom summary CSV
  python -m scripts.plot_results --summary-csv results/summary.csv --output-dir results/plots/

  # Generate only specific plots
  python -m scripts.plot_results --plots accuracy latency memory
        """,
    )
    
    parser.add_argument(
        "--summary-csv",
        type=str,
        default="results/summary.csv",
        help="Path to summary CSV file (default: results/summary.csv)",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="results/plots",
        help="Directory to save plots (default: results/plots/)",
    )
    parser.add_argument(
        "--results-root",
        type=str,
        default="results",
        help="Root directory for finding playbook_log.csv files (default: results/)",
    )
    parser.add_argument(
        "--plots",
        type=str,
        nargs="*",
        default=None,
        help="Specific plots to generate (accuracy, latency, memory, semantic, playbook, evictions, combined). Default: all.",
    )
    parser.add_argument(
        "--dpi",
        type=int,
        default=150,
        help="DPI for saved plots (default: 150)",
    )
    
    args = parser.parse_args()
    
    summary_csv = Path(args.summary_csv)
    if not summary_csv.exists():
        print(f"Error: Summary CSV not found: {summary_csv}", file=sys.stderr)
        print(f"Hint: Run `python -m scripts.aggregate_results` first to generate the summary CSV.", file=sys.stderr)
        return 1
    
    # Load summary CSV
    try:
        df = pd.read_csv(summary_csv)
    except Exception as e:
        print(f"Error: Failed to load summary CSV: {e}", file=sys.stderr)
        return 1
    
    if len(df) == 0:
        print("Error: Summary CSV is empty", file=sys.stderr)
        return 1
    
    output_dir = Path(args.output_dir)
    results_root = Path(args.results_root)
    
    # Determine which plots to generate
    all_plots = ["accuracy", "model_mode", "memory", "semantic", "latency", "playbook", "evictions", "combined"]
    plots_to_generate = args.plots if args.plots else all_plots
    
    # Generate plots
    print(f"Generating plots from {summary_csv}...")
    print(f"Output directory: {output_dir}")
    print(f"Available data: {len(df)} rows")
    print()
    
    # Plot 1: Accuracy by mode
    if "accuracy" in plots_to_generate:
        plot_accuracy_by_mode(df, output_dir / "accuracy_by_mode.png")
    
    # Plot 2: Accuracy by model and mode
    if "model_mode" in plots_to_generate:
        plot_accuracy_by_model_and_mode(df, output_dir / "accuracy_by_model_and_mode.png")
    
    # Plot 3: Accuracy vs Peak RAM (Edge Feasibility)
    if "memory" in plots_to_generate:
        plot_accuracy_vs_peak_memory(df, output_dir / "accuracy_vs_peak_memory.png")
    
    # Plot 4: Semantic Similarity by Mode (Quality Story)
    if "semantic" in plots_to_generate:
        plot_semantic_similarity_by_mode(df, output_dir / "semantic_similarity_by_mode.png")
    
    # Plot 5: Latency by mode
    if "latency" in plots_to_generate:
        plot_latency_by_mode(df, output_dir / "latency_by_mode.png")
    
    # Plot 6: Playbook growth (reads playbook_log.csv files)
    if "playbook" in plots_to_generate:
        plot_playbook_growth(df, output_dir / "playbook_growth.png", results_root=results_root)
    
    # Plot 7: Evictions by mode (reads playbook_log.csv files)
    if "evictions" in plots_to_generate:
        plot_evictions_by_mode(df, output_dir / "evictions_by_mode.png", results_root=results_root)
    
    # Plot 8: Combined figure for paper
    if "combined" in plots_to_generate:
        plot_combined_accuracy_memory_latency(df, output_dir / "combined_metrics.png")
    
    print(f"\nPlots saved to {output_dir}/")
    
    return 0


def plot_accuracy_vs_peak_memory(
    df: pd.DataFrame,
    output_path: Path,
    figsize: tuple = (10, 6),
) -> None:
    """
    Plot accuracy vs peak memory usage by mode (edge feasibility story).
    
    Args:
        df: DataFrame with columns: mode, accuracy, peak_memory_mb
        output_path: Path to save the plot.
        figsize: Figure size tuple.
    """
    if "mode" not in df.columns or "accuracy" not in df.columns or "peak_memory_mb" not in df.columns:
        print("Warning: Missing required columns (mode, accuracy, peak_memory_mb) for accuracy_vs_peak_memory plot", file=sys.stderr)
        return
    
    # Filter to rows with valid data
    plot_df = df[(df["accuracy"].notna()) & (df["peak_memory_mb"].notna())].copy()
    
    if len(plot_df) == 0:
        print("Warning: No valid data for accuracy_vs_peak_memory plot", file=sys.stderr)
        return
    
    fig, ax = plt.subplots(figsize=figsize)
    
    # Plot scatter points colored by mode
    modes = sorted(plot_df["mode"].unique())
    colors = plt.cm.Set1(range(len(modes)))
    
    for mode, color in zip(modes, colors):
        mode_df = plot_df[plot_df["mode"] == mode]
        ax.scatter(
            mode_df["peak_memory_mb"],
            mode_df["accuracy"],
            label=mode,
            alpha=0.7,
            s=100,
            color=color,
        )
    
    ax.set_xlabel("Peak RAM (MB)", fontsize=12)
    ax.set_ylabel("Accuracy", fontsize=12)
    ax.set_title("Accuracy vs Peak RAM by Mode (Edge Feasibility)", fontsize=14)
    ax.legend(title="Mode", fontsize=10)
    ax.grid(alpha=0.3)
    
    plt.tight_layout()
    
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved plot: {output_path}")


def plot_semantic_similarity_by_mode(
    df: pd.DataFrame,
    output_path: Path,
    figsize: tuple = (10, 6),
) -> None:
    """
    Plot average semantic similarity by mode (quality story).
    
    Args:
        df: DataFrame with columns: mode, avg_semantic_similarity
        output_path: Path to save the plot.
        figsize: Figure size tuple.
    """
    if "mode" not in df.columns or "avg_semantic_similarity" not in df.columns:
        print("Warning: Missing required columns (mode, avg_semantic_similarity) for semantic_similarity_by_mode plot", file=sys.stderr)
        return
    
    # Filter to rows with valid semantic similarity
    plot_df = df[df["avg_semantic_similarity"].notna()].copy()
    
    if len(plot_df) == 0:
        print("Warning: No valid semantic similarity data for semantic_similarity_by_mode plot", file=sys.stderr)
        return
    
    # Group by mode and compute mean
    mode_means = plot_df.groupby("mode")["avg_semantic_similarity"].mean().sort_index()
    
    fig, ax = plt.subplots(figsize=figsize)
    
    bars = ax.bar(mode_means.index, mode_means.values, alpha=0.7, color=plt.cm.Set2(range(len(mode_means))))
    
    ax.set_ylabel("Avg Semantic Similarity", fontsize=12)
    ax.set_xlabel("Mode", fontsize=12)
    ax.set_title("Semantic Quality by Mode", fontsize=14)
    ax.set_ylim(0, max(1.0, mode_means.max() * 1.1))
    ax.grid(axis="y", alpha=0.3)
    
    # Add value labels on bars
    for bar in bars:
        height = bar.get_height()
        ax.text(
            bar.get_x() + bar.get_width() / 2.0,
            height,
            f"{height:.3f}",
            ha="center",
            va="bottom",
            fontsize=10,
        )
    
    plt.tight_layout()
    
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved plot: {output_path}")


def plot_latency_by_mode(
    df: pd.DataFrame,
    output_path: Path,
    figsize: tuple = (10, 6),
) -> None:
    """
    Plot average latency by mode.
    
    Args:
        df: DataFrame with columns: mode, avg_latency_ms (or avg_latency_sec)
        output_path: Path to save the plot.
        figsize: Figure size tuple.
    """
    # Check for latency column (prefer ms, fall back to sec)
    latency_col = None
    latency_label = "Avg Latency (ms)"
    if "avg_latency_ms" in df.columns and df["avg_latency_ms"].notna().any():
        latency_col = "avg_latency_ms"
    elif "avg_latency_sec" in df.columns and df["avg_latency_sec"].notna().any():
        latency_col = "avg_latency_sec"
        latency_label = "Avg Latency (sec)"
    
    if latency_col is None:
        print("Warning: No latency data available for latency_by_mode plot", file=sys.stderr)
        return
    
    # Filter to rows with valid latency
    plot_df = df[df[latency_col].notna()].copy()
    
    if len(plot_df) == 0:
        print("Warning: No valid latency data for latency_by_mode plot", file=sys.stderr)
        return
    
    # Use effective_mode if available
    if "effective_mode" in plot_df.columns:
        plot_df["plot_mode"] = plot_df["effective_mode"]
    elif "ace_mode" in plot_df.columns:
        plot_df["plot_mode"] = plot_df.apply(
            lambda r: r["ace_mode"] if pd.notna(r.get("ace_mode")) else r.get("mode", "unknown"),
            axis=1
        )
    else:
        plot_df["plot_mode"] = plot_df.get("mode", "unknown")
    
    # Group by mode and compute mean latency
    mode_latency = plot_df.groupby("plot_mode")[latency_col].mean()
    
    # Sort modes by predefined order
    sorted_modes = [m for m in MODE_ORDER if m in mode_latency.index]
    sorted_modes += [m for m in mode_latency.index if m not in sorted_modes]
    mode_latency = mode_latency.reindex(sorted_modes)
    
    fig, ax = plt.subplots(figsize=figsize)
    x_labels = [normalize_mode_label(m) for m in mode_latency.index]
    bars = ax.bar(x_labels, mode_latency.values, alpha=0.7, color=plt.cm.Set3(range(len(x_labels))))
    
    ax.set_ylabel(latency_label, fontsize=12)
    ax.set_xlabel("Mode", fontsize=12)
    ax.set_title("Average Latency by Mode", fontsize=14)
    ax.grid(axis="y", alpha=0.3)
    plt.xticks(rotation=45, ha="right")
    
    # Add value labels on bars
    for bar in bars:
        height = bar.get_height()
        ax.text(
            bar.get_x() + bar.get_width() / 2.0,
            height,
            f"{height:.1f}",
            ha="center",
            va="bottom",
            fontsize=10,
        )
    
    plt.tight_layout()
    
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved plot: {output_path}")


def find_playbook_logs(results_root: Path) -> List[Dict]:
    """
    Find all playbook_log.csv files and extract their data.
    
    Args:
        results_root: Root directory to search.
        
    Returns:
        List of dicts with playbook log data including mode info.
    """
    all_logs = []
    
    if not results_root.exists():
        return all_logs
    
    for log_file in results_root.rglob("playbook_log.csv"):
        try:
            # Extract mode from path (e.g., results/model/task/mode/device/)
            parts = log_file.parts
            mode = None
            for i, part in enumerate(parts):
                if part in MODE_ORDER or "ace" in part.lower() or "tinyace" in part.lower():
                    mode = part
                    break
            
            if mode is None:
                # Try to get from parent directory structure
                parent_parts = log_file.parent.parts
                if len(parent_parts) >= 2:
                    mode = parent_parts[-2]  # mode is typically 2nd to last
            
            # Read the CSV
            with open(log_file, "r", encoding="utf-8") as f:
                reader = csv.DictReader(f)
                for row in reader:
                    entry = {
                        "mode": mode or "unknown",
                        "step_index": int(row.get("step_index", 0)),
                        "num_entries": int(row.get("num_entries", 0)),
                        "total_tokens": int(row.get("total_tokens", 0)),
                        "num_evictions": int(row.get("num_evictions", 0)),
                    }
                    all_logs.append(entry)
                    
        except Exception as e:
            print(f"Warning: Failed to read {log_file}: {e}", file=sys.stderr)
            continue
    
    return all_logs


def plot_playbook_growth(
    df: pd.DataFrame,
    output_path: Path,
    figsize: tuple = (10, 6),
    results_root: Path = None,
) -> None:
    """
    Plot playbook growth over steps (num_entries vs step_index).
    
    Reads playbook_log.csv files from result directories.
    
    Args:
        df: DataFrame with aggregated metrics (used for fallback if no logs found).
        output_path: Path to save the plot.
        figsize: Figure size tuple.
        results_root: Root directory to search for playbook_log.csv files.
    """
    if results_root is None:
        results_root = Path("results")
    
    # Find and load all playbook logs
    logs = find_playbook_logs(results_root)
    
    if not logs:
        print("Info: No playbook_log.csv files found. Skipping playbook growth plot.", file=sys.stderr)
        return
    
    # Convert to DataFrame
    log_df = pd.DataFrame(logs)
    
    if len(log_df) == 0:
        print("Warning: No valid playbook log data for playbook_growth plot", file=sys.stderr)
        return
    
    fig, ax = plt.subplots(figsize=figsize)
    
    # Plot each mode
    modes = log_df["mode"].unique()
    colors = plt.cm.Set2(np.linspace(0, 1, len(modes)))
    
    for mode, color in zip(sorted(modes), colors):
        mode_log = log_df[log_df["mode"] == mode]
        # Aggregate by step_index (in case of multiple runs)
        step_data = mode_log.groupby("step_index")["num_entries"].mean()
        ax.plot(
            step_data.index,
            step_data.values,
            label=normalize_mode_label(mode),
            color=color,
            marker="o",
            markersize=4,
            linewidth=2,
        )
    
    ax.set_xlabel("Step Index", fontsize=12)
    ax.set_ylabel("Number of Playbook Entries", fontsize=12)
    ax.set_title("Playbook Growth Over Steps", fontsize=14)
    ax.legend(title="Mode", fontsize=10)
    ax.grid(alpha=0.3)
    
    plt.tight_layout()
    
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved plot: {output_path}")


def plot_evictions_by_mode(
    df: pd.DataFrame,
    output_path: Path,
    figsize: tuple = (10, 6),
    results_root: Path = None,
) -> None:
    """
    Plot total evictions by mode (bar chart).
    
    Reads playbook_log.csv files and aggregates eviction counts.
    
    Args:
        df: DataFrame with aggregated metrics.
        output_path: Path to save the plot.
        figsize: Figure size tuple.
        results_root: Root directory to search for playbook_log.csv files.
    """
    if results_root is None:
        results_root = Path("results")
    
    # Find and load all playbook logs
    logs = find_playbook_logs(results_root)
    
    if not logs:
        print("Info: No playbook_log.csv files found. Skipping evictions plot.", file=sys.stderr)
        return
    
    # Convert to DataFrame
    log_df = pd.DataFrame(logs)
    
    if len(log_df) == 0 or "num_evictions" not in log_df.columns:
        print("Warning: No eviction data available for evictions_by_mode plot", file=sys.stderr)
        return
    
    # Sum evictions by mode
    evictions_by_mode = log_df.groupby("mode")["num_evictions"].sum()
    
    if evictions_by_mode.sum() == 0:
        print("Info: No evictions recorded. Skipping evictions plot.", file=sys.stderr)
        return
    
    # Sort modes by predefined order
    sorted_modes = [m for m in MODE_ORDER if m in evictions_by_mode.index]
    sorted_modes += [m for m in evictions_by_mode.index if m not in sorted_modes]
    evictions_by_mode = evictions_by_mode.reindex(sorted_modes)
    
    fig, ax = plt.subplots(figsize=figsize)
    x_labels = [normalize_mode_label(m) for m in evictions_by_mode.index]
    bars = ax.bar(x_labels, evictions_by_mode.values, alpha=0.7, color=plt.cm.Reds(np.linspace(0.3, 0.8, len(x_labels))))
    
    ax.set_ylabel("Total Evictions", fontsize=12)
    ax.set_xlabel("Mode", fontsize=12)
    ax.set_title("Playbook Evictions by Mode", fontsize=14)
    ax.grid(axis="y", alpha=0.3)
    plt.xticks(rotation=45, ha="right")
    
    # Add value labels on bars
    for bar in bars:
        height = bar.get_height()
        ax.text(
            bar.get_x() + bar.get_width() / 2.0,
            height,
            f"{int(height)}",
            ha="center",
            va="bottom",
            fontsize=10,
        )
    
    plt.tight_layout()
    
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved plot: {output_path}")


def plot_combined_accuracy_memory_latency(
    df: pd.DataFrame,
    output_path: Path,
    figsize: tuple = (12, 5),
) -> None:
    """
    Create a combined figure with accuracy, memory, and latency subplots.
    Good for paper figures showing comprehensive mode comparison.
    
    Args:
        df: DataFrame with aggregated metrics.
        output_path: Path to save the plot.
        figsize: Figure size tuple.
    """
    # Check required columns
    has_accuracy = "accuracy" in df.columns and df["accuracy"].notna().any()
    has_memory = "peak_memory_mb" in df.columns and df["peak_memory_mb"].notna().any()
    has_latency = ("avg_latency_ms" in df.columns and df["avg_latency_ms"].notna().any()) or \
                  ("avg_latency_sec" in df.columns and df["avg_latency_sec"].notna().any())
    
    if not has_accuracy:
        print("Warning: No accuracy data for combined plot", file=sys.stderr)
        return
    
    # Use effective_mode if available
    plot_df = df.copy()
    if "effective_mode" in plot_df.columns:
        plot_df["plot_mode"] = plot_df["effective_mode"]
    elif "ace_mode" in plot_df.columns:
        plot_df["plot_mode"] = plot_df.apply(
            lambda r: r["ace_mode"] if pd.notna(r.get("ace_mode")) else r.get("mode", "unknown"),
            axis=1
        )
    else:
        plot_df["plot_mode"] = plot_df.get("mode", "unknown")
    
    # Filter to rows with valid data
    plot_df = plot_df[plot_df["accuracy"].notna()].copy()
    
    # Determine number of subplots
    n_plots = 1 + (1 if has_memory else 0) + (1 if has_latency else 0)
    fig, axes = plt.subplots(1, n_plots, figsize=(4 * n_plots, 5))
    if n_plots == 1:
        axes = [axes]
    
    # Group by mode
    mode_stats = plot_df.groupby("plot_mode").agg({
        "accuracy": "mean",
        **({("peak_memory_mb" if has_memory else "accuracy"): "mean"}),
        **({"avg_latency_ms": "mean"} if "avg_latency_ms" in plot_df.columns else {}),
    })
    
    # Sort modes by predefined order
    sorted_modes = [m for m in MODE_ORDER if m in mode_stats.index]
    sorted_modes += [m for m in mode_stats.index if m not in sorted_modes]
    mode_stats = mode_stats.reindex(sorted_modes)
    x_labels = [normalize_mode_label(m) for m in mode_stats.index]
    
    ax_idx = 0
    colors = plt.cm.Set2(np.linspace(0, 1, len(x_labels)))
    
    # Plot 1: Accuracy
    ax = axes[ax_idx]
    bars = ax.bar(x_labels, mode_stats["accuracy"].values, alpha=0.7, color=colors)
    ax.set_ylabel("Accuracy", fontsize=11)
    ax.set_title("(a) Accuracy", fontsize=12)
    ax.set_ylim(0, 1.0)
    ax.grid(axis="y", alpha=0.3)
    ax.tick_params(axis='x', rotation=45)
    for bar in bars:
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2.0, height, f"{height:.3f}", ha="center", va="bottom", fontsize=8)
    ax_idx += 1
    
    # Plot 2: Memory (if available)
    if has_memory:
        ax = axes[ax_idx]
        mem_values = plot_df.groupby("plot_mode")["peak_memory_mb"].mean().reindex(sorted_modes)
        bars = ax.bar(x_labels, mem_values.values, alpha=0.7, color=colors)
        ax.set_ylabel("Peak Memory (MB)", fontsize=11)
        ax.set_title("(b) Peak Memory", fontsize=12)
        ax.grid(axis="y", alpha=0.3)
        ax.tick_params(axis='x', rotation=45)
        for bar in bars:
            height = bar.get_height()
            if pd.notna(height):
                ax.text(bar.get_x() + bar.get_width()/2.0, height, f"{height:.0f}", ha="center", va="bottom", fontsize=8)
        ax_idx += 1
    
    # Plot 3: Latency (if available)
    if has_latency:
        ax = axes[ax_idx]
        latency_col = "avg_latency_ms" if "avg_latency_ms" in plot_df.columns else "avg_latency_sec"
        latency_values = plot_df.groupby("plot_mode")[latency_col].mean().reindex(sorted_modes)
        bars = ax.bar(x_labels, latency_values.values, alpha=0.7, color=colors)
        ax.set_ylabel("Avg Latency (ms)" if latency_col == "avg_latency_ms" else "Avg Latency (s)", fontsize=11)
        ax.set_title("(c) Latency", fontsize=12)
        ax.grid(axis="y", alpha=0.3)
        ax.tick_params(axis='x', rotation=45)
        for bar in bars:
            height = bar.get_height()
            if pd.notna(height):
                ax.text(bar.get_x() + bar.get_width()/2.0, height, f"{height:.0f}", ha="center", va="bottom", fontsize=8)
    
    plt.tight_layout()
    
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved plot: {output_path}")


if __name__ == "__main__":
    sys.exit(main())
