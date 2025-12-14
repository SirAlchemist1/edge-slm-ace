#!/usr/bin/env python3
"""Aggregate experiment results from metrics.json files.

This script recursively walks a results directory, finds all metrics.json files,
and aggregates them into a single summary CSV and JSON file.

Usage:
    python -m scripts.aggregate_results
    python -m scripts.aggregate_results --results-root results/ --output-csv results/summary.csv
"""

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

import pandas as pd


def find_metrics_files(results_root: Path) -> List[Path]:
    """
    Recursively find all metrics.json files in the results directory.
    
    Args:
        results_root: Root directory to search.
        
    Returns:
        List of paths to metrics.json files.
    """
    metrics_files = []
    if not results_root.exists():
        return metrics_files
    
    for metrics_file in results_root.rglob("metrics.json"):
        metrics_files.append(metrics_file)
    
    return sorted(metrics_files)


def extract_device_from_path(path: Path) -> Optional[str]:
    """
    Try to extract device name from path.
    
    Paths are typically: results/{model}/{task}/{mode}/{device}/metrics.json
    
    Args:
        path: Path to metrics.json file.
        
    Returns:
        Device name if found, None otherwise.
    """
    parts = path.parts
    # Look for common device names in path
    for part in parts:
        if part in ["cpu", "cuda", "mps"]:
            return part
    return None


def derive_run_name_from_path(path: Path) -> str:
    """
    Derive a run name from the path if run_name is not in metrics.
    
    Args:
        path: Path to metrics.json file.
        
    Returns:
        Derived run name.
    """
    # Path is typically: results/{model}/{task}/{mode}/{device}/metrics.json
    parts = path.parts
    # Get the last few meaningful parts before "metrics.json"
    relevant_parts = []
    for part in reversed(parts[:-1]):  # Exclude "metrics.json"
        if part not in ["results", "metrics.json"]:
            relevant_parts.insert(0, part)
        if len(relevant_parts) >= 4:  # model/task/mode/device
            break
    
    return "_".join(relevant_parts) if relevant_parts else "unknown"


def load_and_extract_metrics(metrics_path: Path) -> Optional[Dict[str, Any]]:
    """
    Load a metrics.json file and extract relevant fields.
    
    Args:
        metrics_path: Path to metrics.json file.
        
    Returns:
        Dictionary with extracted metrics, or None if loading failed.
    """
    try:
        with open(metrics_path, "r", encoding="utf-8") as f:
            metrics = json.load(f)
    except Exception as e:
        print(f"Warning: Failed to load {metrics_path}: {e}", file=sys.stderr)
        return None
    
    # Extract core fields
    row = {
        "run_name": metrics.get("run_name") or derive_run_name_from_path(metrics_path),
        "model_id": metrics.get("model_id", "unknown"),
        "task_name": metrics.get("task_name", "unknown"),
        "domain": metrics.get("domain"),
        "mode": metrics.get("mode", "unknown"),
        "ace_mode": metrics.get("ace_mode"),
        "device_requested": metrics.get("device_requested"),
        "device_used": metrics.get("device_used") or extract_device_from_path(metrics_path),
        "accuracy": metrics.get("accuracy"),
        "avg_latency_ms": metrics.get("avg_latency_ms"),
        "avg_latency_sec": metrics.get("avg_latency_sec"),
        "median_latency_sec": metrics.get("median_latency_sec"),
        "num_examples": metrics.get("num_examples"),
        "limit_applied": metrics.get("limit_applied"),
        "wall_time_seconds": metrics.get("wall_time_seconds"),
        "timestamp": metrics.get("timestamp"),
        # Edge & Semantic metrics
        "peak_memory_mb": metrics.get("peak_memory_mb"),
        "peak_gpu_memory_mb": metrics.get("peak_gpu_memory_mb"),
        "avg_semantic_similarity": metrics.get("avg_semantic_similarity"),
        # Playbook metrics (check both top-level and nested playbook object)
        "final_playbook_num_entries": metrics.get("final_playbook_num_entries") or metrics.get("playbook_size"),
        "final_playbook_total_tokens": metrics.get("final_playbook_total_tokens"),
        # MCQ metrics (SciQ tasks only - will be NaN for non-SciQ tasks)
        "oma_accuracy": metrics.get("oma_accuracy"),
        "avg_gom": metrics.get("avg_gom"),
        "acr_rate": metrics.get("acr_rate"),  # Legacy format only
    }
    
    # Derive effective mode label for plotting (combine mode + ace_mode)
    # This creates cleaner labels: "baseline", "ace_full", "ace_working_memory", etc.
    effective_mode = row["ace_mode"] if row["ace_mode"] else row["mode"]
    row["effective_mode"] = effective_mode
    
    # Extract token metrics (support both naming conventions)
    if "mean_prompt_token" in metrics:
        row["mean_prompt_tokens"] = metrics["mean_prompt_token"]
    elif "mean_prompt_tokens" in metrics:
        row["mean_prompt_tokens"] = metrics["mean_prompt_tokens"]
    
    if "mean_output_token" in metrics:
        row["mean_output_tokens"] = metrics["mean_output_token"]
    elif "mean_output_tokens" in metrics:
        row["mean_output_tokens"] = metrics["mean_output_tokens"]
    
    # Extract playbook stats if present
    playbook = metrics.get("playbook")
    if playbook:
        row["playbook_initial_size"] = playbook.get("initial_size", 0)
        row["playbook_final_size"] = playbook.get("final_size", 0)
        row["playbook_entries_added"] = playbook.get("entries_added", 0)
        
        # Extract domain stats if available
        domain_stats = playbook.get("domain_stats", {})
        if domain_stats:
            row["playbook_total_tokens"] = domain_stats.get("total_tokens", 0)
            row["playbook_avg_score"] = domain_stats.get("avg_score")
            row["playbook_avg_success_rate"] = domain_stats.get("avg_success_rate")
    else:
        row["playbook_initial_size"] = None
        row["playbook_final_size"] = None
        row["playbook_entries_added"] = None
    
    # Add path for reference (relative to results root if "results" is in path, else absolute)
    try:
        if "results" in metrics_path.parts:
            results_idx = metrics_path.parts.index("results")
            row["metrics_path"] = str(Path(*metrics_path.parts[results_idx:]))
        else:
            row["metrics_path"] = str(metrics_path)
    except (ValueError, IndexError):
        row["metrics_path"] = str(metrics_path)
    
    return row


def aggregate_metrics(results_root: Path) -> pd.DataFrame:
    """
    Aggregate all metrics.json files into a DataFrame.
    
    Args:
        results_root: Root directory to search for metrics.json files.
        
    Returns:
        DataFrame with aggregated metrics.
    """
    metrics_files = find_metrics_files(results_root)
    
    if len(metrics_files) == 0:
        print(f"Warning: No metrics.json files found in {results_root}", file=sys.stderr)
        return pd.DataFrame()
    
    print(f"Found {len(metrics_files)} metrics.json files")
    
    rows = []
    for metrics_file in metrics_files:
        row = load_and_extract_metrics(metrics_file)
        if row:
            rows.append(row)
    
    if len(rows) == 0:
        print("Warning: No valid metrics could be loaded", file=sys.stderr)
        return pd.DataFrame()
    
    df = pd.DataFrame(rows)
    
    # Sort by model, task, mode, device
    sort_cols = ["model_id", "task_name", "mode"]
    if "ace_mode" in df.columns:
        sort_cols.append("ace_mode")
    if "device_used" in df.columns:
        sort_cols.append("device_used")
    
    available_sort_cols = [col for col in sort_cols if col in df.columns]
    if available_sort_cols:
        df = df.sort_values(available_sort_cols)
    
    return df


def print_summary_table(df: pd.DataFrame) -> None:
    """Print a summary table to stdout."""
    if len(df) == 0:
        print("No data to display")
        return
    
    # Select columns to display (use effective_mode for cleaner output)
    display_cols = ["model_id", "task_name", "effective_mode", "accuracy", "avg_latency_ms", "peak_memory_mb"]
    if "device_used" in df.columns:
        display_cols.append("device_used")
    
    # Filter to available columns
    available_cols = [col for col in display_cols if col in df.columns]
    display_df = df[available_cols].copy()
    
    # Format accuracy as percentage
    if "accuracy" in display_df.columns:
        display_df["accuracy"] = display_df["accuracy"].apply(
            lambda x: f"{x:.3f}" if pd.notna(x) else "N/A"
        )
    
    # Format latency
    if "avg_latency_ms" in display_df.columns:
        display_df["avg_latency_ms"] = display_df["avg_latency_ms"].apply(
            lambda x: f"{x:.1f}" if pd.notna(x) else "N/A"
        )
    
    # Format memory
    if "peak_memory_mb" in display_df.columns:
        display_df["peak_memory_mb"] = display_df["peak_memory_mb"].apply(
            lambda x: f"{x:.1f}" if pd.notna(x) else "N/A"
        )
    
    print("\n" + "=" * 100)
    print("Summary Table")
    print("=" * 100)
    print(display_df.to_string(index=False))
    print("=" * 100 + "\n")


def main() -> int:
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Aggregate experiment results from metrics.json files.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    
    parser.add_argument(
        "--results-root",
        type=str,
        default="results",
        help="Root directory to search for metrics.json files (default: results/)",
    )
    parser.add_argument(
        "--output-csv",
        type=str,
        default="results/summary.csv",
        help="Path to save summary CSV (default: results/summary.csv)",
    )
    parser.add_argument(
        "--output-json",
        type=str,
        default="results/summary.json",
        help="Path to save summary JSON (default: results/summary.json)",
    )
    parser.add_argument(
        "--no-table",
        action="store_true",
        help="Don't print summary table to stdout",
    )
    
    args = parser.parse_args()
    
    results_root = Path(args.results_root)
    if not results_root.exists():
        print(f"Error: Results root directory does not exist: {results_root}", file=sys.stderr)
        return 1
    
    # Aggregate metrics
    df = aggregate_metrics(results_root)
    
    if len(df) == 0:
        print("Error: No metrics found to aggregate", file=sys.stderr)
        return 1
    
    # Save CSV
    output_csv = Path(args.output_csv)
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_csv, index=False)
    print(f"Summary CSV saved to {output_csv} ({len(df)} rows)")
    
    # Save JSON
    output_json = Path(args.output_json)
    output_json.parent.mkdir(parents=True, exist_ok=True)
    # Convert DataFrame to list of dicts for JSON
    df_json = df.replace({pd.NA: None}).to_dict(orient="records")
    with open(output_json, "w", encoding="utf-8") as f:
        json.dump(df_json, f, indent=2, default=str)
    print(f"Summary JSON saved to {output_json}")
    
    # Print summary table
    if not args.no_table:
        print_summary_table(df)
    
    return 0


if __name__ == "__main__":
    sys.exit(main())
