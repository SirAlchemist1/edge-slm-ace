# Plotting Pipeline Setup Summary

## ✅ Completed Tasks

### 1. Created `tinyace_plots.py` ✅
- **Location:** Project root (`/Users/jarvis/TINY ACE/tinyace_plots.py`)
- **Features:**
  - Recursively loads all CSVs from `results/` directory
  - Normalizes column names (handles both canonical and legacy formats)
  - Computes summary statistics
  - Generates 4 core figures:
    - `fig1_memory_cliff.pdf` - Memory cliff / learning curve
    - `fig3_token_efficiency.pdf` - Token efficiency comparison
    - `fig6_ablation.pdf` - Ablation study results
    - `fig7_device_comparison.pdf` - Device comparison (optional)
  - Robust error handling (skips missing data gracefully)
  - Saves both PDF and PNG versions

### 2. Updated Evaluation Scripts ✅
- **`slm_ace/runner.py`:**
  - Updated all result dictionaries to include canonical columns:
    - `qid` (from `sample_id`)
    - `task` (from `task_name`)
    - `model` (from `model_id`)
    - `is_correct` (from `correct`)
  - Maintains backward compatibility with legacy columns
  - All three runner functions updated: `run_dataset_baseline()`, `run_dataset_self_refine()`, `run_dataset_ace()`

- **`scripts/run_experiment.py`:**
  - Added `--auto-plots` flag
  - Automatically calls `tinyace_plots.py` after evaluation completes
  - Graceful error handling if plotting module unavailable

- **`scripts/run_ace_epoch.py`:**
  - Added `--auto-plots` flag
  - Automatically calls `tinyace_plots.py` after all epochs complete
  - Graceful error handling

### 3. Updated Dependencies ✅
- **`requirements.txt`:**
  - Added `matplotlib>=3.7.0`
  - Added `seaborn>=0.12.0`

### 4. Documentation ✅
- **`PLOTTING_GUIDE.md`:**
  - Comprehensive guide covering:
    - Quick start instructions
    - CSV format requirements
    - Figure descriptions
    - LaTeX integration
    - Troubleshooting
    - Advanced usage

## 📋 CSV Format Standardization

### Canonical Format (for plotting):
```python
{
    "qid": "question_1",              # Required
    "task": "medqa",                  # Required
    "model": "phi3mini",              # Required (normalized)
    "mode": "zero_shot",              # Required
    "is_correct": 1,                  # Required (0 or 1)
    "context_tokens": 512,            # Optional (for token plots)
    "latency_ms": 123.45,            # Optional (for latency plots)
}
```

### Legacy Format (backward compatible):
The plotting script automatically maps:
- `sample_id` → `qid`
- `task_name` → `task`
- `model_id` → `model` (with normalization)
- `correct` → `is_correct`

## 🎯 Usage Examples

### Manual Plot Generation:
```bash
# Generate all plots
python tinyace_plots.py

# Custom directories
python tinyace_plots.py --results_dir results --output_dir tinyace_plots
```

### Auto-Generate After Evaluation:
```bash
# Single experiment with auto-plots
python -m scripts.run_experiment \
  --model-id phi3-mini \
  --task-name tatqa_tiny \
  --mode baseline \
  --output-path results/tatqa_phi3_baseline.csv \
  --auto-plots

# ACE epochs with auto-plots
python -m scripts.run_ace_epoch \
  --model-id phi3-mini \
  --task-name tatqa_tiny \
  --epochs 3 \
  --ace-mode ace_working_memory \
  --device cuda \
  --auto-plots
```

## 📊 Generated Outputs

All figures are saved to `tinyace_plots/` directory:

```
tinyace_plots/
├── summary.csv                    # Aggregated stats
├── fig1_memory_cliff.pdf          # Memory cliff plot
├── fig1_memory_cliff.png          # PNG version
├── fig3_token_efficiency.pdf      # Token efficiency
├── fig3_token_efficiency.png
├── fig6_ablation.pdf              # Ablation study
├── fig6_ablation.png
├── fig7_device_comparison.pdf     # Device comparison (if available)
└── fig7_device_comparison.png
```

## 🔧 LaTeX Integration

Figures are ready to use in LaTeX:

```latex
\includegraphics[width=0.95\columnwidth]{tinyace_plots/fig1_memory_cliff.pdf}
```

## ⚠️ Notes

1. **Dependencies:** Install plotting dependencies with:
   ```bash
   pip install matplotlib seaborn
   ```
   Or install all requirements:
   ```bash
   pip install -r requirements.txt
   ```

2. **Mode Normalization:**
   - `baseline` → `zero_shot`
   - `ace_working_memory` → `tinyace`
   - `ace_full` → `ace_full`

3. **Model Normalization:**
   - `microsoft/Phi-3-mini-4k-instruct` → `phi3mini`
   - `meta-llama/Llama-3.2-1B-Instruct` → `llama1b`
   - `sshleifer/tiny-gpt2` → `tinygpt2`

4. **Robustness:**
   - Missing columns are handled gracefully
   - Plots skip if insufficient data
   - Clear warning messages logged

## 🚀 Next Steps

1. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

2. **Test plotting:**
   ```bash
   python tinyace_plots.py
   ```

3. **Run evaluation with auto-plots:**
   ```bash
   python -m scripts.run_experiment --model-id phi3-mini --task-name tatqa_tiny --mode baseline --output-path results/test.csv --auto-plots
   ```

4. **Review generated figures** in `tinyace_plots/` directory

5. **Integrate into LaTeX paper** using the figure paths

## 📝 Files Modified/Created

### Created:
- `tinyace_plots.py` - Main plotting module
- `PLOTTING_GUIDE.md` - Comprehensive guide
- `PLOTTING_SETUP_SUMMARY.md` - This file

### Modified:
- `slm_ace/runner.py` - Added canonical CSV columns
- `scripts/run_experiment.py` - Added `--auto-plots` flag
- `scripts/run_ace_epoch.py` - Added `--auto-plots` flag
- `requirements.txt` - Added matplotlib and seaborn

## ✅ Status

**All tasks completed!** The plotting pipeline is ready to use. Install dependencies and run your first plot generation.

