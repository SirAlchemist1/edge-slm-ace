#!/usr/bin/env python3
"""Summarize experiment results from CSV files.

Reads all CSV files in a directory and produces aggregate statistics.
"""

import argparse
import sys
from pathlib import Path
import pandas as pd


def main():
    """Summarize results from CSV files."""
    parser = argparse.ArgumentParser(
        description="Summarize experiment results from CSV files.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--input-dir",
        type=str,
        required=True,
        help="Directory containing CSV files to summarize",
    )
    parser.add_argument(
        "--output-path",
        type=str,
        default=None,
        help="Path to save summary CSV (default: <input-dir>/summary.csv)",
    )
    parser.add_argument(
        "--semantic-threshold",
        type=float,
        default=0.8,
        help="Threshold for semantic pass (default: 0.8)",
    )

    args = parser.parse_args()

    input_dir = Path(args.input_dir)
    if not input_dir.exists():
        print(f"Error: Input directory not found: {input_dir}")
        return 1

    # Find all CSV files
    csv_files = list(input_dir.glob("*.csv"))
    if len(csv_files) == 0:
        print(f"Error: No CSV files found in {input_dir}")
        return 1

    print(f"Found {len(csv_files)} CSV files in {input_dir}\n")

    # Load and combine all CSVs
    all_data = []
    for csv_file in csv_files:
        try:
            df = pd.read_csv(csv_file)
            all_data.append(df)
        except Exception as e:
            print(f"Warning: Failed to load {csv_file}: {e}")
            continue

    if len(all_data) == 0:
        print("Error: No valid CSV files could be loaded")
        return 1

    combined_df = pd.concat(all_data, ignore_index=True)

    # Determine grouping columns
    group_cols = ["model_id", "task_name", "mode"]

    # Add epoch if present
    if "epoch" in combined_df.columns:
        group_cols.append("epoch")

    # Group and compute statistics
    summary_rows = []
    for group_key, group_df in combined_df.groupby(group_cols):
        if isinstance(group_key, tuple):
            group_dict = dict(zip(group_cols, group_key))
        else:
            group_dict = {group_cols[0]: group_key}

        # Base metrics (present in your runs)
        accuracy = group_df["correct"].mean() if "correct" in group_df.columns else float("nan")
        avg_latency_ms = group_df["latency_ms"].mean() if "latency_ms" in group_df.columns else float("nan")
        num_samples = len(group_df)

        # Optional metrics
        if "reflection_latency_ms" in group_df.columns:
            group_dict["avg_reflection_latency_ms"] = group_df["reflection_latency_ms"].mean()

        # ---- NEW: semantic + token aggregates ----
        if "semantic_score" in group_df.columns:
            sem_mean = group_df["semantic_score"].mean()
            sem_acc = (group_df["semantic_score"] >= args.semantic_threshold).mean()
            group_dict["mean_semantic_score"] = sem_mean
            group_dict["mean_semantic_accuracy"] = sem_acc  # pass@threshold

        if "prompt_tokens" in group_df.columns:
            group_dict["mean_prompt_tokens"] = group_df["prompt_tokens"].mean()

        if "output_tokens" in group_df.columns:
            group_dict["mean_output_tokens"] = group_df["output_tokens"].mean()

        group_dict.update({
            "accuracy": accuracy,
            "avg_latency_ms": avg_latency_ms,
            "num_samples": num_samples,
        })

        summary_rows.append(group_dict)

    # Create summary DataFrame
    summary_df = pd.DataFrame(summary_rows)

    # Sort by model_id, task_name, mode, epoch (if present)
    sort_cols = ["model_id", "task_name", "mode"]
    if "epoch" in summary_df.columns:
        sort_cols.append("epoch")
    summary_df = summary_df.sort_values(sort_cols)

    # Save summary CSV
    if args.output_path:
        output_path = Path(args.output_path)
    else:
        output_path = input_dir / "summary.csv"

    output_path.parent.mkdir(parents=True, exist_ok=True)
    summary_df.to_csv(output_path, index=False)
    print(f"Summary saved to {output_path}\n")

    # Print Markdown table
    print("=" * 80)
    print("Summary Table (Markdown format)")
    print("=" * 80)
    print()

    # Header (conditionally add new columns if present)
    header_cols = ["Model", "Task", "Mode"]
    if "epoch" in summary_df.columns:
        header_cols.append("Epoch")
    header_cols.extend(["Accuracy", "Latency (ms)", "Samples"])

    if "mean_semantic_score" in summary_df.columns:
        header_cols.append("Mean Semantic")
    if "mean_semantic_accuracy" in summary_df.columns:
        header_cols.append(f"Semantic@{args.semantic_threshold:g}")
    if "mean_prompt_tokens" in summary_df.columns:
        header_cols.append("Mean Prompt Toks")
    if "mean_output_tokens" in summary_df.columns:
        header_cols.append("Mean Output Toks")
    if "avg_reflection_latency_ms" in summary_df.columns:
        header_cols.append("Reflection Latency (ms)")

    print("| " + " | ".join(header_cols) + " |")
    print("|" + "|".join(["---"] * len(header_cols)) + "|")

    for _, row in summary_df.iterrows():
        row_values = [
            str(row.get("model_id", "")),
            str(row.get("task_name", "")),
            str(row.get("mode", "")),
        ]
        if "epoch" in summary_df.columns:
            # epoch may be float in CSV; print nicely
            epoch_val = row.get("epoch", "")
            row_values.append(str(int(epoch_val)) if pd.notna(epoch_val) else "")

        row_values.extend([
            f"{row.get('accuracy', 0):.3f}" if pd.notna(row.get("accuracy")) else "",
            f"{row.get('avg_latency_ms', 0):.2f}" if pd.notna(row.get("avg_latency_ms")) else "",
            str(int(row.get("num_samples", 0))),
        ])

        if "mean_semantic_score" in summary_df.columns:
            val = row.get("mean_semantic_score")
            row_values.append(f"{val:.3f}" if pd.notna(val) else "")
        if "mean_semantic_accuracy" in summary_df.columns:
            val = row.get("mean_semantic_accuracy")
            row_values.append(f"{val:.3f}" if pd.notna(val) else "")
        if "mean_prompt_tokens" in summary_df.columns:
            val = row.get("mean_prompt_tokens")
            row_values.append(f"{val:.1f}" if pd.notna(val) else "")
        if "mean_output_tokens" in summary_df.columns:
            val = row.get("mean_output_tokens")
            row_values.append(f"{val:.1f}" if pd.notna(val) else "")
        if "avg_reflection_latency_ms" in summary_df.columns:
            val = row.get("avg_reflection_latency_ms")
            row_values.append(f"{val:.2f}" if pd.notna(val) else "")

        print("| " + " | ".join(row_values) + " |")

    print()
    print("=" * 80)

    return 0


if __name__ == "__main__":
    sys.exit(main())
