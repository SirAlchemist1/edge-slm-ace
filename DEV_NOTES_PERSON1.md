# DEV_NOTES_PERSON1.md

Development notes for Person 1 (Infrastructure & Core Framework)

## Quick Start (Mac M3 Pro)

### 1. Set up virtual environment

```bash
cd /path/to/TINY\ ACE
python3 -m venv .venv
source .venv/bin/activate
pip install -U pip
pip install -r requirements.txt
```

**Note on PyTorch for Mac M3:**
- If you encounter MPS (Metal Performance Shaders) issues, you may need to install PyTorch with CPU support first:
  ```bash
  pip install torch --index-url https://download.pytorch.org/whl/cpu
  pip install transformers pandas
  ```
- For MPS support, ensure you have PyTorch 2.0+ with MPS backend enabled.

### 2. Smoke test with tiny model

Run a baseline evaluation to verify everything works (using task registry):

```bash
python -m scripts.run_experiment \
  --model-id sshleifer/tiny-gpt2 \
  --task-name tatqa_tiny \
  --mode baseline \
  --output-path results/tatqa_tiny_baseline.csv
```

Or with explicit dataset path (legacy mode):

```bash
python -m scripts.run_experiment \
  --model-id sshleifer/tiny-gpt2 \
  --dataset-path data/tasks/tatqa_tiny.json \
  --domain finance \
  --mode baseline \
  --output-path results/tatqa_tiny_baseline.csv
```

**Expected output:**
- Model loads successfully (may download on first run)
- Processes 3 examples from tatqa_tiny.json
- Creates `results/tatqa_tiny_baseline.csv` with consistent schema:
  - Core columns: `model_id`, `task_name`, `domain`, `mode`, `sample_id`, `gold`, `pred`, `latency_ms`
  - Additional columns: `question`, `context`, `correct`
- Prints summary with accuracy and average latency

**Note on Mac M3 Performance:**
- Tiny GPT-2: ✅ Fast, good for smoke tests
- Phi-3 Mini (3.8B): ⚠️ May be slow on Mac M3, but should work for small tests
- Llama 3.2 1B: ⚠️ Should work but may be slow
- **Recommendation**: Use Mac M3 for debugging and tiny model tests. For production runs with Phi-3 Mini or larger models, use GPU laptop (Shatvik) or supercomputer (Archit).

### 3. Test ACE mode

```bash
python -m scripts.run_experiment \
  --model-id sshleifer/tiny-gpt2 \
  --task-name tatqa_tiny \
  --mode ace \
  --playbook-path playbooks/tatqa_playbook.jsonl \
  --output-path results/tatqa_tiny_ace.csv
```

This will:
- Create/load a playbook at `playbooks/tatqa_playbook.jsonl`
- Run ACE pipeline (Generator → Reflector → Curator)
- Save updated playbook and results

**Note**: ACE mode currently uses the full ACE pipeline. TODO(Sathwik): ACE logic improvements are in `ace_roles.py`.

## Project Structure Overview

### Core Modules (`slm_ace/`)

- **`config.py`**: Model registry, task registry, and device detection
  - `ModelConfig` dataclass for model settings
  - `get_model_config()` to resolve model IDs
  - Pre-configured models: `phi3-mini`, `llama-3.2-1b`, `mistral-7b`, `llama-3-8b`, `tiny-gpt2`
  - `TASK_CONFIGS` registry mapping task names to dataset paths and domains
  - `get_task_config()` to resolve task configurations

- **`model_manager.py`**: Model loading and text generation
  - `load_model_and_tokenizer()`: Handles MPS/CUDA/CPU automatically
  - `generate()`: Simple text generation wrapper
  - Includes error handling and MPS fallback

- **`playbook.py`**: ACE memory system
  - `PlaybookEntry`: Single strategy/lesson entry
  - `Playbook`: Collection with load/save, pruning, top-k retrieval
  - JSONL format for persistence

- **`ace_roles.py`**: Prompt builders for ACE roles
  - `build_generator_prompt()`: Creates prompts with playbook context
  - `build_reflector_prompt()`: Creates prompts for lesson generation
  - `parse_reflector_output_to_lessons()`: Extracts lessons from model output
  - `choose_lessons_for_playbook()`: Filters and deduplicates lessons
  - **TODO markers for Person 2 (Shatvik)** throughout

- **`runner.py`**: Main evaluation pipelines
  - `run_dataset_baseline()`: Simple one-shot evaluation
  - `run_dataset_ace()`: Full ACE pipeline with playbook updates
  - **TODO markers for Person 2 and Person 3** in docstrings

- **`metrics.py`**: Evaluation metrics
  - `compute_accuracy()`: Simple exact match (case-insensitive)
  - `compute_average_latency()`: Average latency calculation
  - **TODO markers for Person 3 (Archit)** for advanced metrics

- **`utils.py`**: Utility functions
  - `get_device()`: Auto-detects MPS/CUDA/CPU
  - `time_function()`: Timing decorator

### Scripts (`scripts/`)

- **`run_experiment.py`**: Main CLI entrypoint
  - Handles argument parsing, model loading, dataset loading
  - Supports `--task-name` for convenient task selection (or `--dataset-path` + `--domain` for explicit paths)
  - Runs baseline or ACE evaluation
  - Saves results to CSV with consistent schema
  - Includes comprehensive error handling

### Data (`data/tasks/`)

- **`tatqa_tiny.json`**: 3 finance QA examples (for testing)
- **`medqa_tiny.json`**: 3 medical QA examples (for testing)
- **`iot_tiny.json`**: 5 IoT/anomaly detection examples (for testing)

### Output Directories

- **`results/`**: CSV files with per-example results
- **`playbooks/`**: JSONL files with ACE playbooks (persisted across runs)

## Where Person 2 (Shatvik) Should Work

**Primary file: `slm_ace/ace_roles.py`**

Shatvik should focus on:
1. **Generator prompts** (`build_generator_prompt`):
   - Improve prompt structure and wording
   - Add few-shot examples if needed
   - Domain-specific engineering

2. **Reflector prompts** (`build_reflector_prompt`):
   - Better lesson generation quality
   - Examples of good vs bad lessons
   - Domain-specific reflection guidelines

3. **Lesson parsing** (`parse_reflector_output_to_lessons`):
   - Handle various output formats
   - Extract structured information

4. **Lesson filtering** (`choose_lessons_for_playbook`):
   - Better deduplication logic
   - Quality filtering (remove generic advice)

**Secondary file: `slm_ace/runner.py`**
- The ACE pipeline logic in `run_dataset_ace()` is already wired up
- Shatvik can adjust reflection frequency, pruning strategy, etc.

## Where Person 3 (Archit) Should Work

**Primary file: `slm_ace/metrics.py`**

Archit should extend:
1. **Answer comparison** (`compute_accuracy`):
   - Robust normalization (punctuation, whitespace)
   - Numeric format handling
   - Semantic similarity for partial credit
   - Per-domain metrics (EM for finance, F1 for medical, etc.)

2. **Latency metrics**:
   - Percentiles (p50, p95, p99)
   - Per-step breakdown (generation vs reflection)

3. **Additional metrics**:
   - Energy consumption tracking
   - Memory usage
   - Playbook evolution metrics

**Secondary files:**
- **`slm_ace/runner.py`**: Add logging hooks for metrics
- **`scripts/run_experiment.py`**: Add CLI flags for metric options

## Testing

Run tests:

```bash
pytest tests/
```

Current tests:
- `test_playbook.py`: Playbook add/save/load/prune functionality
- `test_model_manager.py`: Model loading and generation (uses tiny-gpt2)

## Common Issues & Solutions

### MPS (Metal Performance Shaders) Issues
- **Symptom**: Errors about MPS device or operations
- **Solution**: Code automatically falls back to CPU. If you want to force CPU:
  ```python
  # In slm_ace/utils.py, temporarily return torch.device("cpu")
  ```

### Model Download Issues
- **Symptom**: Timeout or connection errors when loading models
- **Solution**: 
  - Check internet connection
  - Set `HF_HOME` environment variable for cache location
  - For large models, consider downloading manually

### Import Errors
- **Symptom**: `ModuleNotFoundError` for `slm_ace`
- **Solution**: 
  - Make sure you're in the repo root
  - Install package: `pip install -e .`
  - Or add to PYTHONPATH: `export PYTHONPATH="${PYTHONPATH}:$(pwd)"`

### Dataset Format Issues
- **Symptom**: JSON decode errors or missing keys
- **Solution**: Ensure dataset JSON has required keys: `id`, `question`, `answer`, (optional) `context`, `domain`

## Available Models and Tasks

### Pre-configured Models

- **`tiny-gpt2`**: `sshleifer/tiny-gpt2` - Tiny test model (fast, no download needed)
- **`phi3-mini`**: `microsoft/Phi-3-mini-4k-instruct` - 3.8B params, 4K context
- **`llama-3.2-1b`**: `meta-llama/Llama-3.2-1B-Instruct` - 1B params, 8K context
- **`mistral-7b`**: `mistralai/Mistral-7B-Instruct-v0.3` - 7B params (for GPU/supercomputer)
- **`llama-3-8b`**: `meta-llama/Meta-Llama-3-8B-Instruct` - 8B params (for supercomputer)

### Pre-configured Tasks

- **`tatqa_tiny`**: Finance QA (3 examples) - `data/tasks/tatqa_tiny.json`, domain: `finance`
- **`medqa_tiny`**: Medical QA (3 examples) - `data/tasks/medqa_tiny.json`, domain: `medical`
- **`iot_tiny`**: IoT/Anomaly Detection (5 examples) - `data/tasks/iot_tiny.json`, domain: `iot`

To add more tasks, edit `TASK_CONFIGS` in `slm_ace/config.py`.

## Current Tested Tasks and Models

### Smoke Tests (Mac M3 Pro)

✅ **Verified working:**
- `tiny-gpt2` + `tatqa_tiny` + baseline + `--limit 3` → ✅ Passed
- `tiny-gpt2` + `medqa_tiny` + baseline + `--limit 3` → ✅ Passed
- `tiny-gpt2` + `iot_tiny` + baseline + `--limit 3` → ✅ Passed

All three tasks produce CSVs with consistent schema:
- Core columns: `model_id`, `task_name`, `domain`, `mode`, `sample_id`, `gold`, `pred`, `latency_ms`
- Debug columns: `question`, `context`, `correct`

## Recommended Commands for Mac M3 Pro (Person 1)

### Tiny GPT-2 Sanity Checks

Quick smoke tests to verify infrastructure:

```bash
# Finance domain
python -m scripts.run_experiment --model-id sshleifer/tiny-gpt2 --task-name tatqa_tiny --mode baseline --output-path results/tatqa_tiny_baseline.csv --limit 3

# Medical domain
python -m scripts.run_experiment --model-id sshleifer/tiny-gpt2 --task-name medqa_tiny --mode baseline --output-path results/medqa_tiny_baseline.csv --limit 3

# IoT domain
python -m scripts.run_experiment --model-id sshleifer/tiny-gpt2 --task-name iot_tiny --mode baseline --output-path results/iot_tiny_baseline.csv --limit 3
```

### Optional: Heavier Models on Mac (Small Subsets Only)

For testing real models on Mac M3, use very small limits:

```bash
# Phi-3 Mini with limit (may be slow)
python -m scripts.run_experiment --model-id phi3-mini --task-name tatqa_tiny --mode baseline --output-path results/tatqa_phi3_baseline_l3.csv --limit 3
```

**Note**: These may be slow. If too slow, skip and use GPU laptop/supercomputer for production runs.

## Mac vs GPU vs Supercomputer

**Mac M3 Pro:**
- ✅ Good for: Development, debugging, tiny smoke tests
- ✅ Use: `tiny-gpt2` for all smoke tests
- ⚠️ Use: Real models (Phi-3, Llama 3.2 1B) only with very small limits (`--limit 3`) for quick checks
- ❌ Not recommended: Full benchmarks or ACE experiments (too slow)

**GPU Laptop (Shatvik) / Supercomputer (Archit):**
- ✅ Use for: All real benchmarks
- ✅ Use for: All ACE experiments
- ✅ Use for: Production runs with Phi-3 Mini, Llama 3.2 1B, or larger models
- ✅ Archit will be primary runner for evaluation metrics

## Example Commands

### Baseline with task registry (recommended)
```bash
python -m scripts.run_experiment \
  --model-id tiny-gpt2 \
  --task-name tatqa_tiny \
  --mode baseline \
  --output-path results/tatqa_tiny_baseline.csv
```

### Baseline with real model (quick test with limit)

Test Phi-3 Mini with limited examples to avoid long runs:

```bash
python -m scripts.run_experiment \
  --model-id phi3-mini \
  --task-name tatqa_tiny \
  --mode baseline \
  --output-path results/tatqa_phi3_baseline_l3.csv \
  --limit 3
```

**What to check:**
- Model loads without OOM errors
- Latency is acceptable (even if slow, that's okay)
- Outputs look reasonable (may not be perfect, but should be coherent)

**Note**: If Phi-3 Mini is too slow/heavy on Mac M3, that's fine. Note it in your testing and use GPU laptop/supercomputer for production runs.

## New Experiment Helpers

Three new driver scripts are available for running experiments:

### 1. Run All Tiny Baselines

Run baseline evaluation on all three tiny tasks with tiny-gpt2 (Mac-safe):

```bash
python -m scripts.run_all_tiny_baselines \
  --limit 3 \
  --output-dir results/tiny
```

**What it does:**
- Loops over `tatqa_tiny`, `medqa_tiny`, `iot_tiny`
- Runs baseline mode for each with `tiny-gpt2`
- Saves CSVs to `results/tiny/`
- Prints summary table at the end

**Options:**
- `--limit`: Number of examples per task (default: 3)
- `--output-dir`: Output directory (default: `results/tiny`)
- `--model-id`: Model to use (default: `sshleifer/tiny-gpt2`)

### 2. Run ACE Evolution (Multi-Epoch)

Run ACE evolution over multiple epochs (epoch 0 = baseline, epochs 1+ = ACE):

```bash
python -m scripts.run_ace_epoch \
  --model-id sshleifer/tiny-gpt2 \
  --task-name tatqa_tiny \
  --epochs 3 \
  --limit 3 \
  --output-dir results/ace_tiny
```

**What it does:**
- Epoch 0: Runs baseline (no ACE)
- Epochs 1+: Runs ACE mode with playbook evolution
- Saves separate CSV for each epoch
- Prints epoch-by-epoch summary table

**Options:**
- `--model-id`: Model ID (required)
- `--task-name`: Task name (required, e.g., `tatqa_tiny`, `medqa_tiny`, `iot_tiny`)
- `--epochs`: Total epochs (default: 3)
- `--limit`: Limit examples per epoch (optional)
- `--output-dir`: Output directory (default: `results/ace`)

**Note**: This script is Mac-safe for tiny models. For heavy models (phi3-mini, mistral, llama-3), use GPU laptop/supercomputer.

### 3. Summarize Results

Aggregate statistics from multiple CSV files:

```bash
python -m scripts.summarize_results \
  --input-dir results/ace_tiny \
  --output-path results/ace_tiny_summary.csv
```

**What it does:**
- Reads all `*.csv` files in input directory
- Groups by `model_id`, `task_name`, `mode` (and `epoch` if present)
- Computes: accuracy, avg_latency_ms, num_samples
- Saves summary CSV and prints Markdown table

**Options:**
- `--input-dir`: Directory with CSV files (required)
- `--output-path`: Output CSV path (default: `<input-dir>/summary.csv`)

**Note**: All three scripts are Mac-safe (no GPU-specific imports). Heavy models should be run by Archit on GPU/supercomputer.

### 4. Run Grid Experiments

Run experiments for all combinations of model × task × mode from a config file:

```bash
# Dry-run to see what would be executed
python -m scripts.run_grid --dry-run

# Run with tiny-gpt2 only (Mac-safe, for testing)
python -m scripts.run_grid --limit 3 --model-id sshleifer/tiny-gpt2

# Run full grid (use on GPU laptop/supercomputer)
python -m scripts.run_grid
```

**What it does:**
- Reads `configs/exp_grid.yaml` for model/task/mode combinations
- For `baseline` mode: Calls `run_experiment`
- For `ace` mode: Calls `run_ace_epoch` (epoch 0 = baseline, epoch 1 = ACE)
- Saves results to `results/grid/`

**Options:**
- `--config`: Path to grid config YAML (default: `configs/exp_grid.yaml`)
- `--dry-run`: Print commands without executing
- `--limit`: Limit examples per experiment (optional)
- `--model-id`: Filter to only run this model (optional)
- `--output-dir`: Output directory (default: `results/grid`)

**Config file format** (`configs/exp_grid.yaml`):
```yaml
models:
  - phi3-mini
  - llama-3.2-1b
tasks:
  - tatqa_tiny
  - medqa_tiny
  - iot_tiny
modes:
  - baseline
  - ace
```

**Note**: Grid runner is Mac-safe. For production runs with real models, use GPU laptop/supercomputer.

### ACE mode (single run)
```bash
python -m scripts.run_experiment \
  --model-id tiny-gpt2 \
  --task-name tatqa_tiny \
  --mode ace \
  --playbook-path playbooks/tatqa_playbook.jsonl \
  --output-path results/tatqa_tiny_ace.csv
```

### Legacy mode (explicit paths)
```bash
python -m scripts.run_experiment \
  --model-id sshleifer/tiny-gpt2 \
  --dataset-path data/tasks/medqa_tiny.json \
  --domain medical \
  --mode baseline \
  --output-path results/medqa_baseline.csv
```

## Result CSV Schema

All result CSVs follow a consistent schema:

**Core columns** (always present):
- `model_id`: HuggingFace model ID used
- `task_name`: Task name (or inferred from dataset path)
- `domain`: Domain name (finance, medical, etc.)
- `mode`: Evaluation mode (`baseline` or `ace`)
- `sample_id`: Example ID from dataset
- `gold`: Ground truth answer
- `pred`: Model prediction
- `latency_ms`: Generation latency in milliseconds

**Additional columns** (for debugging):
- `question`: Original question
- `context`: Context if provided
- `correct`: 1 if exact match, 0 otherwise

**ACE-specific columns** (only in ACE mode):
- `reflection_latency_ms`: Time spent on reflection
- `reflected`: Whether reflection occurred for this example

## Next Steps for Person 1

1. ✅ **Done**: Core infrastructure is in place
2. ✅ **Done**: Error handling and device detection working
3. ✅ **Done**: Task registry and CLI convenience (`--task-name`)
4. ✅ **Done**: Consistent CSV schema
5. ✅ **Done**: Real models wired (Phi-3 Mini, Llama 3.2 1B)
6. **Optional**: Add more test datasets for different domains
7. **Optional**: Add logging configuration (e.g., Python logging module)
8. **Optional**: Add configuration file support (YAML/TOML) for experiment settings

## Notes for Collaboration

- **No weight fine-tuning**: All adaptation is via context/playbook evolution
- **Hardware compatibility**: Code works on Mac M3 (MPS), Linux GPU (CUDA), and CPU
- **Model-agnostic**: Works with any HuggingFace model ID (just pass via `--model-id`)
- **Playbook persistence**: Playbooks are saved after each ACE run and can be reused

