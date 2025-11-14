# TEAM_NOTES.md

Team handoff notes for the SLM-ACE project.

## Current Status (Person 1 - Suryodaya)

✅ **Infrastructure Complete:**
- Core framework in place (`slm_ace/` package)
- Model loading with MPS/CUDA/CPU auto-detection
- Task registry (`--task-name` CLI flag)
- Consistent CSV schema for results
- Baseline and ACE pipeline skeletons
- Error handling and device fallback

✅ **Verified on Mac M3 Pro:**
- Infra stable on Mac M3 Pro
- Tasks in registry: `tatqa_tiny` (finance), `medqa_tiny` (medical), `iot_tiny` (iot)
- Baseline pipeline verified with `tiny-gpt2` on all three tasks
- CSV schema stable across tasks
- ACE + metrics work delegated to Sathwik and Archit

⚠️ **Mac M3 Limitations:**
- Phi-3 Mini and larger models may be slow on Mac M3
- Recommended: Use Mac for tiny models + debugging
- Production runs: Use GPU laptop (Shatvik) or supercomputer (Archit)

---

## For Sathwik (Person 2 — ACE / Prompts)

### ACE Modes

The system now supports two ACE modes:

1. **`ace_full`**: Unbounded playbook mode (default)
   - Uses top-k entries (default k=5) from playbook
   - No token budget limit
   - Good for exploring full playbook evolution

2. **`ace_working_memory`**: Token-budgeted playbook mode
   - Uses scored entries that fit within a token budget (default: 500 tokens)
   - Entries are scored by: correctness ratio + recency decay - genericity penalty
   - Scoring logic lives in `slm_ace/playbook.py` (PlaybookEntry.score() method)
   - Token budget selection in `slm_ace/playbook.py` (get_top_entries_for_budget())

### Testing ACE Modes

Quick test with tiny model (Mac-safe):
```bash
# ACE full mode
python -m scripts.run_ace_epoch \
  --model-id sshleifer/tiny-gpt2 \
  --task-name tatqa_tiny \
  --epochs 2 \
  --ace-mode ace_full \
  --limit 5 \
  --output-dir results/ace_tiny

# ACE working memory mode
python -m scripts.run_ace_epoch \
  --model-id sshleifer/tiny-gpt2 \
  --task-name tatqa_tiny \
  --epochs 2 \
  --ace-mode ace_working_memory \
  --token-budget 500 \
  --limit 5 \
  --output-dir results/ace_tiny
```

**On your GPU laptop**: Always pass `--device cuda` once your CUDA PyTorch is working:
```bash
python -m scripts.run_ace_epoch \
  --model-id phi3-mini \
  --task-name medqa_tiny \
  --epochs 3 \
  --ace-mode ace_working_memory \
  --device cuda \
  --output-dir results/ace_phi3
```

## For Sathwik (Person 2 — ACE / Prompts)

### Your Primary Files

**`slm_ace/ace_roles.py`** — This is your main workspace:

1. **`build_generator_prompt()`** (lines ~8-59)
   - Currently has basic prompt structure
   - TODO(Sathwik) markers indicate where to improve
   - **Goal**: Create prompts that effectively incorporate playbook strategies
   - **Test**: Generator prompts should lead to better answers when playbook has good entries

2. **`build_reflector_prompt()`** (lines ~62-123)
   - Basic reflection prompt exists
   - **Goal**: Generate specific, actionable lessons (not generic advice)
   - **Test**: Reflector should produce lessons like "For TAT-QA, always compute total revenue before tax" not "think carefully"

3. **`parse_reflector_output_to_lessons()`** (lines ~126-145)
   - Basic parser (splits on bullets)
   - **Goal**: Handle various output formats robustly
   - **Test**: Should extract clean lessons from model output

4. **`choose_lessons_for_playbook()`** (lines ~148-207)
   - Basic filtering and deduplication
   - **Goal**: Filter out generic/duplicate lessons before adding to playbook
   - **Test**: Only high-quality, unique lessons should enter playbook

**`slm_ace/runner.py`** — Secondary adjustments:

- `run_dataset_ace()` (lines ~110-288) is already wired up
- You can adjust:
  - `reflect_on_correct_every_n` parameter (currently 5)
  - `prune_every_n` parameter (currently 10)
  - Reflection triggers (currently: always on incorrect, periodically on correct)

**Important**: You will implement and iterate ACE logic in:
- `slm_ace/ace_roles.py` (main file for prompts)
- `slm_ace/playbook.py` (data structures — don't change API)

You do NOT need to touch `model_manager.py` or `runner.py` function signatures.

### Testing Your Changes

**Quick test (tiny model, Mac-friendly):**
```bash
python -m scripts.run_experiment \
  --model-id sshleifer/tiny-gpt2 \
  --task-name tatqa_tiny \
  --mode ace \
  --playbook-path playbooks/tatqa_playbook.jsonl \
  --output-path results/tatqa_tiny_ace_test.csv \
  --limit 3
```

**ACE epoch evolution test (recommended for development):**
```bash
# Test ACE evolution over multiple epochs (tiny model, Mac-friendly)
python -m scripts.run_ace_epoch \
  --model-id sshleifer/tiny-gpt2 \
  --task-name tatqa_tiny \
  --epochs 3 \
  --limit 3 \
  --output-dir results/ace_tiny
```

**Real test (Phi-3 Mini, on GPU laptop):**
```bash
# Test on any of the three tasks: tatqa_tiny, medqa_tiny, iot_tiny
python -m scripts.run_experiment \
  --model-id phi3-mini \
  --task-name tatqa_tiny \
  --mode ace \
  --playbook-path playbooks/tatqa_playbook.jsonl \
  --output-path results/tatqa_phi3_ace.csv

# Or use ACE epoch driver for multi-epoch evolution:
python -m scripts.run_ace_epoch \
  --model-id phi3-mini \
  --task-name tatqa_tiny \
  --epochs 5 \
  --output-dir results/ace_phi3
```

**What to check:**
- Playbook entries are specific and domain-relevant (not generic)
- ACE mode accuracy improves over baseline after several examples
- Playbook grows sensibly (not too fast, not too slow)
- Reflection produces useful lessons

### Success Criteria

✅ **ACE mode shows improvement:**
- After 10-20 examples, ACE accuracy > baseline accuracy
- Playbook contains 5-15 useful, specific strategies
- Latency increase is acceptable (<2x baseline)

✅ **Playbook quality:**
- Lessons are specific (e.g., "For finance QA, extract numbers before arithmetic")
- No generic fluff (e.g., "think carefully", "be thorough")
- Deduplication works (similar lessons merge)

---

## For Archit (Person 3 — Metrics / Evaluation)

### Metrics and Summary Script

The metrics system now includes:

1. **Exact accuracy**: Case-insensitive exact match (already implemented)
2. **Semantic accuracy**: Sentence embedding similarity (if sentence-transformers available)
   - Uses `all-MiniLM-L6-v2` model for embeddings
   - Falls back to exact match if sentence-transformers not available
   - Threshold: 0.7 (configurable)

3. **Token metrics**: 
   - `prompt_tokens`: Full input prompt tokens
   - `context_tokens`: Playbook context tokens (for ACE modes)
   - `output_tokens`: Generated output tokens

4. **Summary script**: `scripts/summarize_results.py`
   - Groups by model_id, task_name, mode (and epoch if present)
   - Computes: accuracy_exact, accuracy_semantic, avg_latency_ms, avg_prompt_tokens, avg_context_tokens, avg_output_tokens
   - Outputs CSV summary and Markdown table

### Testing Metrics

Generate test data and summarize:
```bash
# Run experiments
python -m scripts.run_experiment \
  --model-id sshleifer/tiny-gpt2 \
  --task-name tatqa_tiny \
  --mode baseline \
  --output-path results/test_baseline.csv \
  --limit 5

# Summarize results
python -m scripts.summarize_results \
  --input-dir results/ \
  --output-path results/summary.csv
```

**Note**: You can run small tests locally with `--limit 5`, full runs later on GPU/supercomputer.

**On your GPU laptop**: Always pass `--device cuda` once your CUDA PyTorch is working. If CUDA is misconfigured, you can temporarily run with `--device cpu` for debugging:

```bash
# GPU run (once CUDA is set up)
python -m scripts.run_experiment \
  --model-id phi3-mini \
  --task-name medqa_tiny \
  --mode baseline \
  --device cuda \
  --output-path results/medqa_phi3_baseline.csv

# CPU fallback (for debugging)
python -m scripts.run_experiment \
  --model-id phi3-mini \
  --task-name medqa_tiny \
  --mode baseline \
  --device cpu \
  --limit 5 \
  --output-path results/medqa_phi3_cpu_test.csv
```

## For Archit (Person 3 — Metrics / Evaluation)

### Your Primary Files

**`slm_ace/metrics.py`** — Main metrics implementation:

1. **`compute_accuracy()`** (lines ~8-33)
   - Currently: Simple case-insensitive exact match
   - **Goal**: Robust answer comparison
   - **TODO(Archit)** markers indicate extensions needed:
     - Numeric format normalization ("100" vs "100.0" vs "100.00")
     - Punctuation/whitespace normalization
     - Semantic similarity for partial credit
     - Per-domain metrics (EM for finance, F1 for medical)

2. **`compute_average_latency()`** (lines ~36-44)
   - Currently: Simple average
   - **Goal**: Add percentiles (p50, p95, p99)
   - Consider per-step breakdown (generation vs reflection latency)

**Summary script: `scripts/summarize_results.py`** (already created by Person 1)

This script:
- Reads all CSV files in a directory
- Groups by `model_id`, `task_name`, `mode` (and `epoch` if present)
- Computes aggregate statistics (accuracy, latency, sample counts)
- Outputs CSV summary and Markdown table

**You will implement and extend metrics in:**
- `slm_ace/metrics.py` (main file for answer comparison and latency metrics)

**You will run larger models and more samples on:**
- GPU laptop or supercomputer

**Use these scripts for experiments:**
- `scripts.run_experiment` — Single baseline or ACE run
- `scripts.run_ace_epoch` — Multi-epoch ACE evolution (epoch 0 = baseline, epochs 1+ = ACE)
- `scripts.summarize_results` — Aggregate CSVs into summary tables
- `scripts.run_grid` — Run grid experiments (model × task × mode) from `configs/exp_grid.yaml`

You should track per-epoch improvements and possibly add CI/bootstrap scripts later.

### Testing Your Changes

**Generate test data:**
```bash
# Baseline run
python -m scripts.run_experiment \
  --model-id phi3-mini \
  --task-name tatqa_tiny \
  --mode baseline \
  --output-path results/tatqa_phi3_baseline.csv

# ACE run
python -m scripts.run_experiment \
  --model-id phi3-mini \
  --task-name tatqa_tiny \
  --mode ace \
  --playbook-path playbooks/tatqa_playbook.jsonl \
  --output-path results/tatqa_phi3_ace.csv
```

**Then analyze:**
```bash
python scripts/summarize_results.py \
  --baseline results/tatqa_phi3_baseline.csv \
  --ace results/tatqa_phi3_ace.csv \
  --output results/comparison.md
```

### Success Criteria

✅ **Robust metrics:**
- Handles numeric answers correctly (100 = 100.0 = 100.00)
- Handles text answers with minor variations
- Per-domain metrics work (EM for finance, F1 for medical if needed)

✅ **Summary tables:**
- Clear comparison: baseline vs ACE accuracy
- Latency breakdown (generation vs reflection)
- Per-model performance table
- Ready for paper Methods/Results sections

---

## Common Workflow

### 1. Person 1 (Suryodaya) runs smoke test
```bash
python -m scripts.run_experiment \
  --model-id sshleifer/tiny-gpt2 \
  --task-name tatqa_tiny \
  --mode baseline \
  --output-path results/smoke_test.csv
```

### 2. Sathwik develops ACE prompts
- Edit `slm_ace/ace_roles.py`
- Test with tiny model first
- Then test with Phi-3 Mini on GPU laptop

### 3. Archit develops metrics
- Edit `slm_ace/metrics.py`
- Create `scripts/summarize_results.py`
- Analyze results from Sathwik's runs

### 4. Integration test
- All three run full pipeline
- Compare results
- Document findings

---

## File Ownership

**Person 1 (Suryodaya) — Infrastructure:**
- `slm_ace/config.py` ✅
- `slm_ace/model_manager.py` ✅
- `slm_ace/utils.py` ✅
- `slm_ace/runner.py` (pipeline, but Sathwik adjusts ACE logic)
- `scripts/run_experiment.py` ✅
- `scripts/run_all_tiny_baselines.py` ✅
- `scripts/run_ace_epoch.py` ✅
- `scripts/summarize_results.py` ✅
- `scripts/run_grid.py` ✅
- `configs/exp_grid.yaml` ✅
- `DEV_NOTES_PERSON1.md` ✅

**Person 2 (Sathwik) — ACE Logic:**
- `slm_ace/ace_roles.py` 🔧 (your main file)
- `slm_ace/runner.py` (adjust ACE parameters)

**Person 3 (Archit) — Metrics:**
- `slm_ace/metrics.py` 🔧 (your main file)
- `scripts/summarize_results.py` 🔧 (create this)

**Shared:**
- `slm_ace/playbook.py` (data structures — don't change API)
- `data/tasks/*.json` (add more datasets as needed)
- `tests/` (add tests for your changes)

---

## Quick Reference Commands

### Smoke test (Mac M3)
```bash
python -m scripts.run_experiment \
  --model-id sshleifer/tiny-gpt2 \
  --task-name tatqa_tiny \
  --mode baseline \
  --output-path results/smoke_test.csv
```

### Real baseline (GPU laptop / supercomputer)
```bash
python -m scripts.run_experiment \
  --model-id phi3-mini \
  --task-name tatqa_tiny \
  --mode baseline \
  --output-path results/tatqa_phi3_baseline.csv
```

### Real ACE (GPU laptop / supercomputer)
```bash
python -m scripts.run_experiment \
  --model-id phi3-mini \
  --task-name tatqa_tiny \
  --mode ace \
  --playbook-path playbooks/tatqa_playbook.jsonl \
  --output-path results/tatqa_phi3_ace.csv
```

### Quick test (limit examples)
```bash
python -m scripts.run_experiment \
  --model-id phi3-mini \
  --task-name tatqa_tiny \
  --mode ace \
  --playbook-path playbooks/tatqa_playbook.jsonl \
  --output-path results/test.csv \
  --limit 3
```

---

## Questions?

- **Infrastructure issues**: Ask Suryodaya
- **ACE/prompt questions**: Ask Sathwik
- **Metrics/evaluation questions**: Ask Archit
- **General project questions**: Group discussion

