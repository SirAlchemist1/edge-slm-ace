# TinyACE Plotting Pipeline Guide

## Overview

The `tinyace_plots.py` module automatically generates all paper figures from standardized evaluation results. It reads CSV files from the `results/` directory and generates publication-ready PDF and PNG figures.

## Quick Start

### Generate All Plots

```bash
# Generate all plots from results/
python tinyace_plots.py

# Custom paths
python tinyace_plots.py --results_dir results --output_dir tinyace_plots
```

### Auto-Generate Plots After Evaluation

```bash
# Run evaluation and auto-generate plots
python -m scripts.run_experiment \
  --model-id phi3-mini \
  --task-name tatqa_tiny \
  --mode baseline \
  --output-path results/tatqa_phi3_baseline.csv \
  --auto-plots

# Run ACE epochs and auto-generate plots
python -m scripts.run_ace_epoch \
  --model-id phi3-mini \
  --task-name tatqa_tiny \
  --epochs 3 \
  --ace-mode ace_working_memory \
  --device cuda \
  --auto-plots
```

## Expected CSV Format

Each CSV file under `results/` should have at least these columns:

### Required Columns:
- `qid` or `sample_id`: Question/sample identifier
- `task` or `task_name`: Task name (e.g., `medqa`, `tatqa_tiny`, `gsm8k`)
- `model` or `model_id`: Model identifier (e.g., `phi3mini`, `llama1b`)
- `mode`: One of `["zero_shot", "baseline", "ace_full", "ace_working_memory", "tinyace", "self_refine"]`
- `is_correct` or `correct`: 0/1 or boolean indicating correctness

### Optional Columns (for enhanced plots):
- `context_tokens`: Integer, total tokens fed into model
- `latency_ms`: Float, end-to-end response time in milliseconds

### Legacy Columns (backward compatible):
The plotting script automatically maps:
- `sample_id` → `qid`
- `task_name` → `task`
- `model_id` → `model` (with normalization)
- `correct` → `is_correct`

## Generated Figures

### Figure 1: Memory Cliff (`fig1_memory_cliff.pdf`)
**LaTeX Reference:** `Figure~\ref{fig:motivation}`

Shows learning curve: accuracy vs questions processed for different modes (zero_shot, ace_full, tinyace).

**Requirements:**
- Multiple samples with sequential `qid` values
- At least two modes: `zero_shot` and one of `ace_full` or `tinyace`

### Figure 3: Token Efficiency (`fig3_token_efficiency.pdf`)
**LaTeX Reference:** Token overhead comparison figure

Compares average context tokens across modes (zero_shot, ace_full, tinyace).

**Requirements:**
- `context_tokens` column in CSV files
- Multiple modes available

### Figure 6: Ablation Study (`fig6_ablation.pdf`)
**LaTeX Reference:** Ablation results figure

Shows accuracy drops for ablation variants relative to `tinyace_full`.

**Requirements:**
- Modes like `tinyace_full`, `tinyace_no_recency`, `tinyace_no_vagueness`, etc.
- If not available, function logs a warning and skips

### Figure 7: Device Comparison (`fig7_device_comparison.pdf`)
**LaTeX Reference:** Device/latency comparison (optional)

Compares latency vs accuracy across different devices.

**Requirements:**
- `device` or `hardware` column in CSV files
- `latency_ms` column
- If not available, function logs a message and skips

## Output Structure

```
tinyace_plots/
├── summary.csv              # Aggregated statistics per (task, model, mode)
├── fig1_memory_cliff.pdf    # Memory cliff plot
├── fig1_memory_cliff.png    # PNG version
├── fig3_token_efficiency.pdf
├── fig3_token_efficiency.png
├── fig6_ablation.pdf
├── fig6_ablation.png
├── fig7_device_comparison.pdf (if device data available)
└── fig7_device_comparison.png
```

## CSV File Organization

The plotting script recursively searches `results/` for CSV files. Recommended structure:

```
results/
├── {task}/
│   ├── {model}_{mode}.csv
│   └── {model}_{mode}_epoch{N}.csv
├── {task}_baseline.csv
└── ...
```

Examples:
- `results/medqa/phi3mini_zero_shot.csv`
- `results/medqa/phi3mini_ace_full.csv`
- `results/medqa/phi3mini_tinyace.csv`
- `results/gsm8k/llama1b_tinyace.csv`

## Mode Normalization

The plotting script normalizes mode names:
- `baseline` → `zero_shot`
- `ace_full` → `ace_full`
- `ace_working_memory` → `tinyace`
- `self_refine` → `self_refine`

## Model Name Normalization

Model IDs are automatically normalized:
- `microsoft/Phi-3-mini-4k-instruct` → `phi3mini`
- `sshleifer/tiny-gpt2` → `tinygpt2`
- `meta-llama/Llama-3.2-1B-Instruct` → `llama1b`

## Troubleshooting

### "No CSV files found"
- Check that CSV files exist in `results/` directory
- Verify file extensions are `.csv` (case-sensitive)

### "No data for memory cliff plot"
- Ensure you have multiple samples with sequential `qid` values
- Check that at least two modes are present (zero_shot and ace_full/tinyace)

### "No context_tokens data available"
- Token efficiency plot requires `context_tokens` column
- Check that evaluation scripts are writing this column
- Plot will be skipped if data unavailable

### "No ablation modes found"
- Ablation plot requires modes like `tinyace_full`, `tinyace_no_recency`, etc.
- If not running ablation studies, this is expected
- Plot will be skipped with a warning

## Integration with LaTeX

In your LaTeX paper (`tinyace_paper.tex`), include figures like:

```latex
\begin{figure}[h]
    \centering
    \includegraphics[width=0.95\columnwidth]{tinyace_plots/fig1_memory_cliff.pdf}
    \caption{Memory Cliff: Learning curve showing accuracy improvement over questions processed.}
    \label{fig:motivation}
\end{figure}

\begin{figure}[h]
    \centering
    \includegraphics[width=0.95\columnwidth]{tinyace_plots/fig3_token_efficiency.pdf}
    \caption{Token Efficiency: Average context tokens across different modes.}
    \label{fig:token_efficiency}
\end{figure}
```

## Dependencies

Required Python packages (already in `requirements.txt`):
- `matplotlib>=3.7.0`
- `seaborn>=0.12.0`
- `pandas>=2.0.0`
- `numpy>=1.24.0`

Install with:
```bash
pip install -r requirements.txt
```

## Advanced Usage

### Custom Task/Model Selection

The plotting functions accept optional `task` and `model` parameters. You can modify `tinyace_plots.py` to filter specific combinations:

```python
# In plot_memory_cliff()
plot_memory_cliff(df, output_path, task="medqa", model="phi3mini")
```

### Adding New Plots

To add a new figure:

1. Create a new plotting function (e.g., `plot_new_figure()`)
2. Add it to `main()` function
3. Update LaTeX mapping in docstring
4. Follow the same pattern as existing plots

## Best Practices

1. **Standardize CSV format**: Ensure all evaluation scripts write canonical columns
2. **Run plots after experiments**: Use `--auto-plots` flag for convenience
3. **Check summary.csv**: Review aggregated statistics before generating plots
4. **Version control plots**: Commit generated PDFs to track figure evolution
5. **Document mode names**: Keep mode naming consistent across experiments

