#!/usr/bin/env python3
"""Generate plots from aggregated experiment results.

This script reads a summary CSV file and generates visualization plots
comparing accuracy across different modes, models, and tasks.

Usage:
    python -m scripts.plot_results
    python -m scripts.plot_results --summary-csv results/summary.csv --output-dir results/plots/
"""

import argparse
import sys
from pathlib import Path
from typing import List, Optional

import matplotlib.pyplot as plt
import pandas as pd


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
    if "mode" not in df.columns or "accuracy" not in df.columns:
        print("Warning: Missing required columns (mode, accuracy) for accuracy_by_mode plot", file=sys.stderr)
        return
    
    # Filter to rows with valid accuracy
    plot_df = df[df["accuracy"].notna()].copy()
    
    if len(plot_df) == 0:
        print("Warning: No valid accuracy data for accuracy_by_mode plot", file=sys.stderr)
        return
    
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
            mode_accuracy = task_df.groupby("mode")["accuracy"].mean().sort_index()
            
            ax = axes[idx]
            bars = ax.bar(mode_accuracy.index, mode_accuracy.values, alpha=0.7)
            ax.set_title(f"Task: {task}", fontsize=12)
            ax.set_xlabel("Mode", fontsize=10)
            ax.set_ylabel("Accuracy", fontsize=10)
            ax.set_ylim(0, max(1.0, mode_accuracy.max() * 1.1))
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
                    fontsize=9,
                )
        
        plt.tight_layout()
    else:
        # Single task or no task column: single plot
        mode_accuracy = plot_df.groupby("mode")["accuracy"].mean().sort_index()
        
        fig, ax = plt.subplots(figsize=figsize)
        bars = ax.bar(mode_accuracy.index, mode_accuracy.values, alpha=0.7)
        ax.set_title("Accuracy by Mode", fontsize=14)
        ax.set_xlabel("Mode", fontsize=12)
        ax.set_ylabel("Accuracy", fontsize=12)
        ax.set_ylim(0, max(1.0, mode_accuracy.max() * 1.1))
        ax.grid(axis="y", alpha=0.3)
        
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
    
    args = parser.parse_args()
    
    summary_csv = Path(args.summary_csv)
    if not summary_csv.exists():
        print(f"Error: Summary CSV not found: {summary_csv}", file=sys.stderr)
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
    
    # Generate plots
    print(f"Generating plots from {summary_csv}...")
    
    # Plot 1: Accuracy by mode
    plot_accuracy_by_mode(df, output_dir / "accuracy_by_mode.png")
    
    # Plot 2: Accuracy by model and mode
    plot_accuracy_by_model_and_mode(df, output_dir / "accuracy_by_model_and_mode.png")
    
    print(f"\nPlots saved to {output_dir}/")
    
    return 0


if __name__ == "__main__":
    sys.exit(main())
