#!/usr/bin/env python3
"""Summarize experiment results from CSV files.

Reads all CSV files in a directory and produces aggregate statistics.

Required CSV columns:
- model_id: Model identifier
- task_name: Task identifier
- mode: Evaluation mode (baseline, ace_full, ace_working_memory, self_refine)
- correct: Binary correctness (1 or 0)
- pred: Model prediction
- gold: Ground truth answer
- latency_ms: Generation latency in milliseconds

Optional CSV columns:
- epoch: Epoch number (for multi-epoch experiments)
- prompt_tokens: Number of tokens in the prompt
- context_tokens: Number of tokens in context (playbook for ACE modes)
- output_tokens: Number of tokens in the output
- reflection_latency_ms: Reflection latency (for ACE modes)

Summary metrics computed:
- accuracy_exact: Mean of 'correct' column (exact match accuracy)
- accuracy_semantic: Semantic similarity accuracy (if sentence-transformers available)
- avg_latency_ms: Mean latency in milliseconds
- avg_prompt_tokens: Average prompt tokens (if column present)
- avg_context_tokens: Average context tokens (if column present)
- avg_output_tokens: Average output tokens (if column present)
- num_samples: Number of samples in the group
"""

import argparse
import sys
from pathlib import Path

import pandas as pd

from slm_ace.metrics import compute_semantic_accuracy


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
        
        # Compute metrics
        accuracy_exact = group_df["correct"].mean()
        avg_latency_ms = group_df["latency_ms"].mean()
        num_samples = len(group_df)
        
        # Compute semantic accuracy if possible
        try:
            accuracy_semantic = compute_semantic_accuracy(
                preds=group_df["pred"].tolist(),
                labels=group_df["gold"].tolist(),
            )
            if accuracy_semantic is not None:
                group_dict["accuracy_semantic"] = accuracy_semantic
        except Exception:
            # If semantic accuracy fails, skip it
            pass
        
        # Token metrics
        if "prompt_tokens" in group_df.columns:
            avg_prompt_tokens = group_df["prompt_tokens"].mean()
            group_dict["avg_prompt_tokens"] = avg_prompt_tokens
        
        if "context_tokens" in group_df.columns:
            avg_context_tokens = group_df["context_tokens"].mean()
            group_dict["avg_context_tokens"] = avg_context_tokens
        
        if "output_tokens" in group_df.columns:
            avg_output_tokens = group_df["output_tokens"].mean()
            group_dict["avg_output_tokens"] = avg_output_tokens
        
        # Add optional metrics if present
        if "reflection_latency_ms" in group_df.columns:
            avg_reflection_latency = group_df["reflection_latency_ms"].mean()
            group_dict["avg_reflection_latency_ms"] = avg_reflection_latency
        
        group_dict.update({
            "accuracy_exact": accuracy_exact,
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
    
    # Print header
    header_cols = ["Model", "Task", "Mode"]
    if "epoch" in summary_df.columns:
        header_cols.append("Epoch")
    header_cols.extend(["Exact Acc", "Semantic Acc"])
    if "avg_prompt_tokens" in summary_df.columns:
        header_cols.append("Prompt Tokens")
    if "avg_context_tokens" in summary_df.columns:
        header_cols.append("Context Tokens")
    if "avg_output_tokens" in summary_df.columns:
        header_cols.append("Output Tokens")
    header_cols.extend(["Latency (ms)", "Samples"])
    if "avg_reflection_latency_ms" in summary_df.columns:
        header_cols.append("Reflection Latency (ms)")
    
    print("| " + " | ".join(header_cols) + " |")
    print("|" + "|".join(["---"] * len(header_cols)) + "|")
    
    # Print rows
    for _, row in summary_df.iterrows():
        row_values = [
            str(row.get("model_id", "")),
            str(row.get("task_name", "")),
            str(row.get("mode", "")),
        ]
        if "epoch" in summary_df.columns:
            row_values.append(str(int(row.get("epoch", 0))))
        
        # Accuracy metrics
        row_values.append(f"{row.get('accuracy_exact', 0):.3f}")
        if "accuracy_semantic" in summary_df.columns:
            row_values.append(f"{row.get('accuracy_semantic', 0):.3f}")
        else:
            row_values.append("N/A")
        
        # Token metrics
        if "avg_prompt_tokens" in summary_df.columns:
            row_values.append(f"{row.get('avg_prompt_tokens', 0):.1f}")
        if "avg_context_tokens" in summary_df.columns:
            row_values.append(f"{row.get('avg_context_tokens', 0):.1f}")
        if "avg_output_tokens" in summary_df.columns:
            row_values.append(f"{row.get('avg_output_tokens', 0):.1f}")
        
        row_values.extend([
            f"{row.get('avg_latency_ms', 0):.2f}",
            str(int(row.get("num_samples", 0))),
        ])
        if "avg_reflection_latency_ms" in summary_df.columns:
            row_values.append(f"{row.get('avg_reflection_latency_ms', 0):.2f}")
        
        print("| " + " | ".join(row_values) + " |")
    
    print()
    print("=" * 80)
    
    return 0


# Example commands:
#
# Summarize results from ACE epoch directory:
# python -m scripts.summarize_results \
#   --input-dir results/ace_tiny \
#   --output-path results/ace_tiny_summary.csv
#
# Summarize all results in a directory:
# python -m scripts.summarize_results \
#   --input-dir results/ \
#   --output-path results/summary.csv


if __name__ == "__main__":
    sys.exit(main())

