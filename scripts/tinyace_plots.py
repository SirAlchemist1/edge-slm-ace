#!/usr/bin/env python3
"""
TinyACE Plotting Pipeline

This module generates all paper figures from standardized evaluation results.

Expected CSV Schema:
--------------------
Each CSV under results/ should have at least:
- "qid" or "sample_id"      : question/sample identifier
- "task" or "task_name"     : task name (e.g., medqa, tatqa_tiny, gsm8k)
- "model" or "model_id"     : model name (e.g., phi3mini, llama1b)
- "mode"                    : one of ["zero_shot", "baseline", "ace_full", "ace_working_memory", "tinyace", "self_refine"]
- "is_correct" or "correct" : 0/1 or boolean
- "context_tokens"          : integer, total tokens fed into model (optional)
- "latency_ms"              : float, end-to-end response time (optional)

Usage:
------
    # Generate all plots from results/
    python -m scripts.tinyace_plots

    # Custom paths
    python -m scripts.tinyace_plots --results_dir results --output_dir tinyace_plots

LaTeX Figure Mapping:
---------------------
- fig1_memory_cliff.pdf      -> Figure~\ref{fig:motivation} (Memory Cliff / Learning Curve)
- fig3_token_efficiency.pdf  -> Token overhead comparison
- fig6_ablation.pdf          -> Ablation study results
- fig7_device_comparison.pdf -> Device/latency comparison (optional)

"""

import argparse
import logging
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import matplotlib
matplotlib.use('Agg')  # Non-interactive backend
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

# Set style
sns.set_style("whitegrid")
plt.rcParams['figure.dpi'] = 300
plt.rcParams['savefig.dpi'] = 300
plt.rcParams['pdf.fonttype'] = 42  # TrueType fonts for LaTeX
plt.rcParams['ps.fonttype'] = 42

logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
logger = logging.getLogger(__name__)


def normalize_column_names(df: pd.DataFrame) -> pd.DataFrame:
    """
    Normalize column names to canonical format.
    
    Maps:
    - sample_id -> qid
    - task_name -> task
    - model_id -> model
    - correct -> is_correct
    """
    df = df.copy()
    
    # Column mapping
    column_map = {
        'sample_id': 'qid',
        'task_name': 'task',
        'model_id': 'model',
        'correct': 'is_correct',
    }
    
    for old_col, new_col in column_map.items():
        if old_col in df.columns and new_col not in df.columns:
            df[new_col] = df[old_col]
        elif old_col in df.columns:
            # Both exist, prefer the canonical name
            if df[old_col].equals(df[new_col]):
                df = df.drop(columns=[old_col])
            else:
                logger.warning(f"Both {old_col} and {new_col} exist, keeping {new_col}")
    
    return df


def extract_model_name(model_id: str) -> str:
    """
    Extract short model name from full model_id.
    
    Examples:
    - "microsoft/Phi-3-mini-4k-instruct" -> "phi3mini"
    - "sshleifer/tiny-gpt2" -> "tinygpt2"
    - "meta-llama/Llama-3.2-1B-Instruct" -> "llama1b"
    """
    model_id_lower = model_id.lower()
    
    # Common patterns
    if 'phi-3' in model_id_lower or 'phi3' in model_id_lower:
        if 'mini' in model_id_lower:
            return 'phi3mini'
        return 'phi3'
    elif 'llama' in model_id_lower:
        if '1b' in model_id_lower or '1b' in model_id:
            return 'llama1b'
        elif '3.2' in model_id_lower:
            return 'llama32'
        return 'llama'
    elif 'tiny-gpt2' in model_id_lower or 'tinygpt2' in model_id_lower:
        return 'tinygpt2'
    elif 'mistral' in model_id_lower:
        return 'mistral'
    
    # Fallback: use last part of path
    parts = model_id.split('/')
    if len(parts) > 1:
        return parts[-1].lower().replace('-', '').replace('_', '')[:10]
    return model_id.lower().replace('-', '').replace('_', '')[:10]


def extract_task_name(task_name: str) -> str:
    """
    Normalize task name.
    
    Examples:
    - "tatqa_tiny" -> "tatqa"
    - "medqa_tiny" -> "medqa"
    - "iot_tiny" -> "iot"
    """
    # Remove common suffixes
    task = task_name.replace('_tiny', '').replace('_small', '').lower()
    return task


def normalize_mode(mode: str) -> str:
    """
    Normalize mode names to canonical format.
    
    Maps:
    - "baseline" -> "zero_shot"
    - "ace_full" -> "ace_full"
    - "ace_working_memory" -> "tinyace" (or keep as is)
    """
    mode_lower = mode.lower()
    
    if mode_lower in ['baseline', 'zero_shot']:
        return 'zero_shot'
    elif mode_lower == 'ace_full':
        return 'ace_full'
    elif mode_lower in ['ace_working_memory', 'tinyace']:
        return 'tinyace'
    elif mode_lower == 'self_refine':
        return 'self_refine'
    
    return mode_lower


def load_results(results_dir: str = "results") -> pd.DataFrame:
    """
    Load all CSV files from results directory recursively.
    
    Args:
        results_dir: Path to results directory.
        
    Returns:
        Combined DataFrame with normalized columns.
    """
    results_dir = Path(results_dir)
    if not results_dir.exists():
        logger.error(f"Results directory not found: {results_dir}")
        return pd.DataFrame()
    
    all_dfs = []
    csv_files = list(results_dir.rglob("*.csv"))
    
    if not csv_files:
        logger.warning(f"No CSV files found in {results_dir}")
        return pd.DataFrame()
    
    logger.info(f"Found {len(csv_files)} CSV files")
    
    for csv_path in csv_files:
        try:
            df = pd.read_csv(csv_path)
            
            # Skip summary CSVs
            if 'summary' in csv_path.name.lower():
                continue
            
            # Normalize columns
            df = normalize_column_names(df)
            
            # Extract normalized fields
            if 'model' not in df.columns and 'model_id' in df.columns:
                df['model'] = df['model_id'].apply(extract_model_name)
            elif 'model_id' in df.columns:
                df['model'] = df['model_id'].apply(extract_model_name)
            
            if 'task' not in df.columns and 'task_name' in df.columns:
                df['task'] = df['task_name'].apply(extract_task_name)
            elif 'task_name' in df.columns:
                df['task'] = df['task_name'].apply(extract_task_name)
            
            if 'mode' in df.columns:
                df['mode'] = df['mode'].apply(normalize_mode)
            
            # Ensure is_correct is numeric
            if 'is_correct' in df.columns:
                df['is_correct'] = pd.to_numeric(df['is_correct'], errors='coerce').fillna(0)
            
            # Add source file for debugging
            df['source_file'] = str(csv_path.relative_to(results_dir))
            
            all_dfs.append(df)
            logger.debug(f"Loaded {len(df)} rows from {csv_path.name}")
            
        except Exception as e:
            logger.warning(f"Failed to load {csv_path}: {e}")
            continue
    
    if not all_dfs:
        logger.error("No valid CSV files loaded")
        return pd.DataFrame()
    
    # Combine all DataFrames
    combined_df = pd.concat(all_dfs, ignore_index=True)
    
    # Ensure required columns exist
    required_cols = ['qid', 'task', 'model', 'mode', 'is_correct']
    missing_cols = [col for col in required_cols if col not in combined_df.columns]
    
    if missing_cols:
        logger.warning(f"Missing columns: {missing_cols}. Attempting to infer...")
        
        # Try to infer qid from sample_id or index
        if 'qid' not in combined_df.columns:
            if 'sample_id' in combined_df.columns:
                combined_df['qid'] = combined_df['sample_id']
            else:
                combined_df['qid'] = combined_df.index.astype(str)
    
    logger.info(f"Loaded {len(combined_df)} total rows from {len(all_dfs)} files")
    return combined_df


def compute_summary(df: pd.DataFrame) -> pd.DataFrame:
    """
    Compute summary statistics for each (task, model, mode) combination.
    
    Args:
        df: Combined results DataFrame.
        
    Returns:
        Summary DataFrame with columns:
        - task, model, mode
        - accuracy (mean of is_correct)
        - avg_context_tokens (mean when available)
        - avg_latency_ms (mean when available)
        - num_samples
    """
    if df.empty:
        return pd.DataFrame()
    
    summary_rows = []
    
    for (task, model, mode), group_df in df.groupby(['task', 'model', 'mode']):
        row = {
            'task': task,
            'model': model,
            'mode': mode,
            'accuracy': group_df['is_correct'].mean() if 'is_correct' in group_df.columns else np.nan,
            'num_samples': len(group_df),
        }
        
        if 'context_tokens' in group_df.columns:
            row['avg_context_tokens'] = group_df['context_tokens'].mean()
        else:
            row['avg_context_tokens'] = np.nan
        
        if 'latency_ms' in group_df.columns:
            row['avg_latency_ms'] = group_df['latency_ms'].mean()
        else:
            row['avg_latency_ms'] = np.nan
        
        summary_rows.append(row)
    
    summary_df = pd.DataFrame(summary_rows)
    return summary_df.sort_values(['task', 'model', 'mode'])


def plot_memory_cliff(df: pd.DataFrame, output_path: Path, task: str = None, model: str = None):
    """
    Plot memory cliff / learning curve showing accuracy vs questions processed.
    
    Creates separate curves for zero_shot, ace_full, and tinyace modes.
    
    Args:
        df: Combined results DataFrame.
        output_path: Path to save figure.
        task: Task name to filter (if None, uses first available).
        model: Model name to filter (if None, uses first available).
    """
    if df.empty:
        logger.warning("No data for memory cliff plot")
        return
    
    # Filter data
    plot_df = df.copy()
    
    if task:
        plot_df = plot_df[plot_df['task'] == task]
    if model:
        plot_df = plot_df[plot_df['model'] == model]
    
    if plot_df.empty:
        logger.warning(f"No data for task={task}, model={model}")
        return
    
    # Use first available task/model if not specified
    if task is None:
        task = plot_df['task'].iloc[0]
        plot_df = plot_df[plot_df['task'] == task]
    if model is None:
        model = plot_df['model'].iloc[0]
        plot_df = plot_df[plot_df['model'] == model]
    
    # Filter to relevant modes
    relevant_modes = ['zero_shot', 'ace_full', 'tinyace']
    plot_df = plot_df[plot_df['mode'].isin(relevant_modes)]
    
    if plot_df.empty:
        logger.warning(f"No data for modes {relevant_modes}")
        return
    
    # Convert qid to numeric for sorting
    plot_df['qid_numeric'] = pd.to_numeric(plot_df['qid'], errors='coerce')
    plot_df = plot_df.sort_values('qid_numeric').reset_index(drop=True)
    
    # Create bins (sliding windows)
    num_bins = 10
    bin_size = max(1, len(plot_df) // num_bins)
    
    fig, ax = plt.subplots(figsize=(8, 6))
    
    for mode in relevant_modes:
        mode_df = plot_df[plot_df['mode'] == mode].copy()
        if mode_df.empty:
            continue
        
        mode_df = mode_df.sort_values('qid_numeric').reset_index(drop=True)
        
        # Compute rolling accuracy
        questions_processed = []
        accuracies = []
        
        for i in range(bin_size, len(mode_df) + 1, max(1, bin_size // 2)):
            window = mode_df.iloc[:i]
            acc = window['is_correct'].mean()
            questions_processed.append(i)
            accuracies.append(acc)
        
        if questions_processed:
            ax.plot(questions_processed, accuracies, marker='o', label=mode.replace('_', ' ').title(), linewidth=2)
    
    ax.set_xlabel('Questions Processed', fontsize=12)
    ax.set_ylabel('Accuracy', fontsize=12)
    ax.set_title(f'Memory Cliff: {task.upper()} - {model.upper()}', fontsize=14, fontweight='bold')
    ax.legend(loc='best')
    ax.grid(True, alpha=0.3)
    ax.set_ylim([0, 1.05])
    
    plt.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, format='pdf', bbox_inches='tight')
    plt.savefig(str(output_path).replace('.pdf', '.png'), format='png', bbox_inches='tight', dpi=300)
    plt.close()
    
    logger.info(f"Saved memory cliff plot to {output_path}")


def plot_token_efficiency(df: pd.DataFrame, output_path: Path):
    """
    Plot token efficiency comparison (accuracy vs context tokens).
    
    Args:
        df: Combined results DataFrame.
        output_path: Path to save figure.
    """
    if df.empty:
        logger.warning("No data for token efficiency plot")
        return
    
    # Compute summary
    summary = compute_summary(df)
    
    if summary.empty or 'avg_context_tokens' not in summary.columns:
        logger.warning("No context_tokens data available for token efficiency plot")
        return
    
    # Filter to relevant modes
    relevant_modes = ['zero_shot', 'ace_full', 'tinyace']
    plot_df = summary[summary['mode'].isin(relevant_modes)].copy()
    
    if plot_df.empty:
        logger.warning(f"No data for modes {relevant_modes}")
        return
    
    # Remove rows with missing token data
    plot_df = plot_df.dropna(subset=['avg_context_tokens'])
    
    if plot_df.empty:
        logger.warning("No valid token data after filtering")
        return
    
    fig, ax = plt.subplots(figsize=(10, 6))
    
    # Group by mode and compute averages
    mode_stats = plot_df.groupby('mode').agg({
        'accuracy': 'mean',
        'avg_context_tokens': 'mean'
    }).reset_index()
    
    # Create bar plot
    x_pos = np.arange(len(mode_stats))
    bars = ax.bar(x_pos, mode_stats['avg_context_tokens'], alpha=0.7, color=['#1f77b4', '#ff7f0e', '#2ca02c'])
    
    # Add accuracy as text on bars
    for i, (idx, row) in enumerate(mode_stats.iterrows()):
        ax.text(i, row['avg_context_tokens'] + max(mode_stats['avg_context_tokens']) * 0.02,
                f"Acc: {row['accuracy']:.2f}", ha='center', va='bottom', fontsize=10)
    
    ax.set_xlabel('Mode', fontsize=12)
    ax.set_ylabel('Average Context Tokens', fontsize=12)
    ax.set_title('Token Efficiency Comparison', fontsize=14, fontweight='bold')
    ax.set_xticks(x_pos)
    ax.set_xticklabels([m.replace('_', ' ').title() for m in mode_stats['mode']])
    ax.grid(True, alpha=0.3, axis='y')
    
    plt.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, format='pdf', bbox_inches='tight')
    plt.savefig(str(output_path).replace('.pdf', '.png'), format='png', bbox_inches='tight', dpi=300)
    plt.close()
    
    logger.info(f"Saved token efficiency plot to {output_path}")


def plot_ablation(df: pd.DataFrame, output_path: Path):
    """
    Plot ablation study results.
    
    Looks for modes like:
    - tinyace_full
    - tinyace_no_recency
    - tinyace_no_vagueness
    - tinyace_random_eviction
    
    Args:
        df: Combined results DataFrame.
        output_path: Path to save figure.
    """
    if df.empty:
        logger.warning("No data for ablation plot")
        return
    
    # Find ablation modes
    ablation_modes = [m for m in df['mode'].unique() if 'tinyace' in m.lower() and ('no_' in m.lower() or 'random' in m.lower() or 'full' in m.lower())]
    
    if not ablation_modes:
        logger.warning("No ablation modes found. Expected modes like 'tinyace_full', 'tinyace_no_recency', etc.")
        return
    
    # Compute summary
    summary = compute_summary(df)
    ablation_summary = summary[summary['mode'].isin(ablation_modes)].copy()
    
    if ablation_summary.empty:
        logger.warning("No ablation data in summary")
        return
    
    # Find baseline (tinyace_full)
    baseline_mode = None
    for mode in ablation_modes:
        if 'full' in mode.lower():
            baseline_mode = mode
            break
    
    if baseline_mode is None:
        logger.warning("No baseline mode (tinyace_full) found")
        return
    
    baseline_acc = ablation_summary[ablation_summary['mode'] == baseline_mode]['accuracy'].mean()
    
    # Compute accuracy drops relative to baseline
    ablation_summary['accuracy_drop'] = baseline_acc - ablation_summary['accuracy']
    
    fig, ax = plt.subplots(figsize=(10, 6))
    
    # Sort by accuracy drop
    ablation_summary = ablation_summary.sort_values('accuracy_drop', ascending=False)
    
    x_pos = np.arange(len(ablation_summary))
    colors = ['red' if drop > 0 else 'green' for drop in ablation_summary['accuracy_drop']]
    bars = ax.bar(x_pos, ablation_summary['accuracy_drop'], alpha=0.7, color=colors)
    
    ax.axhline(y=0, color='black', linestyle='--', linewidth=1)
    ax.set_xlabel('Ablation Variant', fontsize=12)
    ax.set_ylabel('Accuracy Drop (vs tinyace_full)', fontsize=12)
    ax.set_title('Ablation Study Results', fontsize=14, fontweight='bold')
    ax.set_xticks(x_pos)
    ax.set_xticklabels([m.replace('tinyace_', '').replace('_', ' ').title() for m in ablation_summary['mode']], rotation=45, ha='right')
    ax.grid(True, alpha=0.3, axis='y')
    
    plt.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, format='pdf', bbox_inches='tight')
    plt.savefig(str(output_path).replace('.pdf', '.png'), format='png', bbox_inches='tight', dpi=300)
    plt.close()
    
    logger.info(f"Saved ablation plot to {output_path}")


def plot_device_comparison(df: pd.DataFrame, output_path: Path):
    """
    Plot device comparison (latency vs accuracy per device).
    
    This is optional - if device info is not available, logs a message.
    
    Args:
        df: Combined results DataFrame.
        output_path: Path to save figure.
    """
    # Check if device info exists
    if 'device' not in df.columns and 'hardware' not in df.columns:
        logger.info("Device comparison plot skipped: no device/hardware column found")
        return
    
    device_col = 'device' if 'device' in df.columns else 'hardware'
    
    if df.empty or df[device_col].isna().all():
        logger.info("Device comparison plot skipped: no device data available")
        return
    
    # Compute summary by device
    summary = compute_summary(df)
    
    if 'device' not in summary.columns:
        logger.info("Device comparison plot skipped: device info not in summary")
        return
    
    fig, ax = plt.subplots(figsize=(10, 6))
    
    devices = summary['device'].unique()
    colors = plt.cm.Set3(np.linspace(0, 1, len(devices)))
    
    for i, device in enumerate(devices):
        device_df = summary[summary['device'] == device]
        ax.scatter(device_df['avg_latency_ms'], device_df['accuracy'],
                  label=device, s=100, alpha=0.7, color=colors[i])
    
    ax.set_xlabel('Average Latency (ms)', fontsize=12)
    ax.set_ylabel('Accuracy', fontsize=12)
    ax.set_title('Device Comparison: Latency vs Accuracy', fontsize=14, fontweight='bold')
    ax.legend(loc='best')
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, format='pdf', bbox_inches='tight')
    plt.savefig(str(output_path).replace('.pdf', '.png'), format='png', bbox_inches='tight', dpi=300)
    plt.close()
    
    logger.info(f"Saved device comparison plot to {output_path}")


def main(results_dir: str = "results", output_dir: str = "tinyace_plots"):
    """
    Main entry point: load results, compute summary, generate all plots.
    
    Args:
        results_dir: Directory containing result CSVs.
        output_dir: Directory to save plots and summary.
    """
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    logger.info(f"Loading results from {results_dir}")
    df = load_results(results_dir)
    
    if df.empty:
        logger.error("No results loaded. Check that CSV files exist in results directory.")
        return
    
    logger.info(f"Computing summary statistics...")
    summary = compute_summary(df)
    
    # Save summary CSV
    summary_path = output_path / "summary.csv"
    summary.to_csv(summary_path, index=False)
    logger.info(f"Saved summary to {summary_path}")
    
    # Generate all plots
    logger.info("Generating plots...")
    
    # Figure 1: Memory Cliff
    plot_memory_cliff(df, output_path / "fig1_memory_cliff.pdf")
    
    # Figure 3: Token Efficiency
    plot_token_efficiency(df, output_path / "fig3_token_efficiency.pdf")
    
    # Figure 6: Ablation
    plot_ablation(df, output_path / "fig6_ablation.pdf")
    
    # Figure 7: Device Comparison (optional)
    plot_device_comparison(df, output_path / "fig7_device_comparison.pdf")
    
    logger.info(f"All plots saved to {output_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Generate TinyACE paper figures from evaluation results",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__
    )
    parser.add_argument(
        "--results_dir",
        type=str,
        default="results",
        help="Directory containing result CSV files (default: results)"
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="tinyace_plots",
        help="Directory to save plots and summary (default: tinyace_plots)"
    )
    
    args = parser.parse_args()
    main(results_dir=args.results_dir, output_dir=args.output_dir)

