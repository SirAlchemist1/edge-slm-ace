# SciQ MCQ-Aware Evaluation Guide

This document describes the MCQ-aware evaluation metrics for SciQ-style multiple choice question tasks.

## Overview

SciQ tasks have multiple choice options (A/B/C/D) where the model's free-form answer needs to be mapped to one of the options. Standard exact-match accuracy may underestimate model performance when the answer is semantically correct but phrased differently.

## New Metrics

### Option-Mapped Accuracy (OMA)

Maps model predictions to the closest option via semantic similarity (using MiniLM embeddings) and checks if it matches the gold option.

- **Per-example column**: `oma_correct` (0 or 1)
- **Run-level metric**: `oma_accuracy` (mean of oma_correct)

### Gold Option Margin (GOM)

Measures the margin between semantic similarity to the gold option versus average similarity to distractor options. Higher values indicate more confident correct predictions.

- **Per-example column**: `gom` (float, can be negative)
- **Run-level metric**: `avg_gom` (mean of gom)

### Answerable Choice Rate (ACR)

Measures format adherence - how often the model outputs a clear choice marker (A/B/C/D) in its response.

- **Per-example column**: `acr_hit` (0 or 1)
- **Run-level metric**: `acr_rate` (mean of acr_hit)

## Quick Sanity Check

Run a quick baseline evaluation on SciQ:

```bash
# Run Phi-3 baseline on SciQ with limit=5
python -m scripts.run_experiment \
    --model-id microsoft/Phi-3-mini-4k-instruct \
    --task-name sciq_test \
    --mode baseline \
    --output-path results/sciq_sanity/baseline/results.csv \
    --metrics-path results/sciq_sanity/baseline/metrics.json \
    --limit 5 \
    --device cpu

# Or use tiny-gpt2 for faster testing (lower quality results)
python -m scripts.run_experiment \
    --model-id sshleifer/tiny-gpt2 \
    --task-name sciq_test \
    --mode baseline \
    --output-path results/sciq_sanity/baseline/results.csv \
    --metrics-path results/sciq_sanity/baseline/metrics.json \
    --limit 5 \
    --device cpu
```

## Verification Checklist

After running the sanity check, verify:

- [ ] **results.csv has MCQ columns**:
  ```bash
  head -1 results/sciq_sanity/baseline/results.csv | tr ',' '\n' | grep -E 'gold_option|pred_option|oma_correct|gom|acr_hit'
  ```
  Expected output should show: `gold_option`, `pred_option`, `oma_correct`, `gom`, `acr_hit`

- [ ] **gold_option is populated** (should be "A" for all examples in sciq_tiny):
  ```bash
  cut -d',' -f$(head -1 results/sciq_sanity/baseline/results.csv | tr ',' '\n' | nl | grep gold_option | awk '{print $1}') results/sciq_sanity/baseline/results.csv | tail -n +2
  ```

- [ ] **oma_accuracy is computed** (check metrics.json):
  ```bash
  grep -E '"oma_accuracy"|"avg_gom"|"acr_rate"' results/sciq_sanity/baseline/metrics.json
  ```

- [ ] **Metrics are not always zero** (unless model is truly random):
  The `oma_accuracy` should be > 0 for models that produce reasonable outputs.

## Dataset Format

SciQ datasets should follow this JSON format:

```json
[
  {
    "id": "sciq_1",
    "domain": "science",
    "question": "What is the chemical formula for water?",
    "correct_answer": "H2O",
    "distractor1": "CO2",
    "distractor2": "NaCl",
    "distractor3": "O2",
    "support": "Water consists of two hydrogen atoms bonded to one oxygen atom.",
    "answer": "H2O"
  }
]
```

Required fields for MCQ evaluation:
- `correct_answer`: The correct answer text
- `distractor1`, `distractor2`, `distractor3`: Three incorrect option texts
- `answer`: Ground truth (same as `correct_answer`, for backward compatibility)

Optional:
- `support`: Context/supporting text (used as context in prompts)

## Running Full Evaluation

```bash
# Run all modes on SciQ
python -m scripts.run_experiment \
    --model-id microsoft/Phi-3-mini-4k-instruct \
    --task-name sciq_tiny \
    --mode baseline \
    --output-path results/phi3_sciq/baseline/results.csv \
    --metrics-path results/phi3_sciq/baseline/metrics.json

# Aggregate results
python -m scripts.aggregate_results

# Generate plots (includes OMA/GOM/ACR plots for SciQ data)
python -m scripts.plot_results
```

## New Plots

When SciQ data is present in `results/summary.csv`, three new plots are generated:

1. `oma_accuracy_by_mode.png` - Option-Mapped Accuracy comparison
2. `avg_gom_by_mode.png` - Gold Option Margin comparison
3. `acr_rate_by_mode.png` - Format adherence comparison

## Tests

Run the MCQ evaluation tests:

```bash
pytest tests/test_mcq_eval.py -v
```

## Implementation Notes

- MCQ metrics are only computed when `task_name` contains "sciq" (case-insensitive)
- The evaluation uses the existing SemanticEvaluator singleton (MiniLM-L6-v2)
- For non-SciQ tasks, MCQ columns are absent (not null) in results
- Aggregation handles missing MCQ metrics gracefully (NaN in summary)
