# DEV_ARCHITECTURE_OVERVIEW.md

> **Last Updated: 2024-12-09 (Phases 2-4 Complete)**
> This document describes the current architecture based on actual code inspection.

---

## 1. High-Level Overview

This repository implements an **ACE (Agentic Context Engineering)** evaluation framework for Small Language Models (SLMs). Instead of fine-tuning, the framework uses a self-improving "playbook" memory that accumulates domain-specific strategies over time.

### Core Concept

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                           ACE EVALUATION LOOP                                │
│                                                                              │
│  ┌──────────┐    ┌───────────┐    ┌──────────┐    ┌──────────┐              │
│  │ Playbook │───►│ Generator │───►│ Evaluate │───►│ Reflector│              │
│  │ (Memory) │    │ (Answer)  │    │ (Score)  │    │ (Lessons)│              │
│  └──────────┘    └───────────┘    └──────────┘    └────┬─────┘              │
│       ▲                                                │                     │
│       │                ┌──────────┐                    │                     │
│       └────────────────│ Curator  │◄───────────────────┘                     │
│                        │ (Filter) │                                          │
│                        └──────────┘                                          │
└─────────────────────────────────────────────────────────────────────────────┘
```

---

## 2. Project Structure

```
edge-slm-ace/
├── src/
│   └── edge_slm_ace/          # Main package (pip install -e .)
│       ├── __init__.py
│       ├── core/               # ACE logic
│       │   ├── __init__.py
│       │   ├── runner.py       # Main evaluation loops (baseline, ACE, self-refine)
│       │   └── ace_roles.py    # Prompt builders, output parsers
│       ├── memory/             # Playbook system
│       │   ├── __init__.py
│       │   └── playbook.py     # Retention scoring, token-budget eviction
│       ├── models/             # Model management
│       │   ├── __init__.py
│       │   └── model_manager.py  # HuggingFace loading, generation
│       └── utils/              # Utilities
│           ├── __init__.py
│           ├── config.py       # Model/task configs
│           ├── device_utils.py # Device detection
│           └── metrics.py      # Evaluation metrics
├── scripts/                    # CLI entry points
│   ├── run_experiment.py       # Single experiment runner
│   ├── run_eval_grid.py        # Grid runner (model × task × mode)
│   └── ...
├── tests/                      # Unit tests (42 tests)
├── configs/
│   └── experiment_grid.yaml    # Grid configuration
├── data/tasks/                 # Datasets
├── results/                    # Outputs
├── setup.py                    # Package installation
└── pyproject.toml
```

---

## 3. Retention Scoring Formula

The playbook uses a formal retention scoring formula:

```
S(l_i, t) = α · (N_succ / (N_used + ε))      # Success ratio
          - β · (N_fail / (N_used + ε))      # Failure penalty
          + γ · exp(-λ · (t - t_last))       # Recency bonus
          - δ · V(l_i)                        # Vagueness penalty
```

**Hyperparameters (default values):**
| Parameter | Default | Description |
|-----------|---------|-------------|
| α (alpha) | 1.0 | Weight for success ratio |
| β (beta) | 0.5 | Weight for failure penalty |
| γ (gamma) | 0.3 | Weight for recency bonus |
| δ (delta) | 0.4 | Weight for vagueness penalty |
| λ (lambda) | 0.05 | Recency decay rate |
| ε (epsilon) | 1.0 | Smoothing constant |

**Vagueness Detection:**
- Short lessons (< 5 words) → high vagueness
- Generic phrases ("think carefully", "pay attention") → high vagueness
- Formulas, numbers, specific procedures → low vagueness

---

## 4. Key Components

### 4.1 PlaybookEntry (`memory/playbook.py`)

```python
@dataclass
class PlaybookEntry:
    id: str
    domain: str
    text: str
    success_count: int = 0      # Times used with correct answer
    failure_count: int = 0      # Times used with incorrect answer
    created_at: float = 0.0
    last_used_at: int = 0       # Step when last used
    token_count: int = 0        # Estimated tokens
    vagueness_score: float = 0.0  # 0=specific, 1=vague
```

### 4.2 Playbook Class

**Key Methods:**
- `get_top_k(domain, k, current_step)` → Top-K entries by score
- `get_top_entries_for_budget(domain, token_budget, current_step)` → Token-budgeted selection
- `add_entry(domain, text, step, enforce_budget)` → Add with optional eviction
- `record_feedback(entry_id, helpful)` → Update success/failure counts
- `prune(max_entries_per_domain, current_step)` → Remove low-score entries

**Token-Budget Eviction:**
When `token_budget` is set and a new entry would exceed the budget, the playbook automatically evicts lowest-scoring entries to make room.

### 4.3 ACE Runner (`core/runner.py`)

**Three Modes:**
1. **baseline**: No playbook, vanilla prompting
2. **ace_full**: ACE loop with top-k retrieval
3. **ace_working_memory**: ACE loop with token-budgeted retrieval

**ACE Loop:**
```python
for step, example in enumerate(dataset):
    # 1. Retrieve lessons (top-k or token-budgeted)
    used_entries = playbook.get_top_k(...) or playbook.get_top_entries_for_budget(...)
    
    # 2. Build prompt with lessons
    prompt = build_generator_prompt(domain, playbook, question, context, ace_mode, ...)
    
    # 3. Generate answer
    raw_answer = generate(model, tokenizer, prompt, ...)
    answer, reasoning = parse_generator_output(raw_answer)
    
    # 4. Evaluate correctness
    correct = (answer.lower() == ground_truth.lower())
    
    # 5. Record feedback for USED entries only
    for entry_id in used_entry_ids:
        playbook.mark_entry_used(entry_id, step)
        playbook.record_feedback(entry_id, helpful=correct)
    
    # 6. Reflection (if incorrect or periodic)
    if not correct or (step % reflect_every_n == 0):
        lessons = generate_and_parse_reflection(...)
        for lesson in lessons:
            playbook.add_entry(domain, lesson, step)  # NO feedback at creation
    
    # 7. Periodic pruning
    if step % prune_every_n == 0:
        playbook.prune(max_entries_per_domain)
```

**Critical Design Decision:**
- New lessons are added **without** feedback
- Feedback is only recorded when a lesson is **actually used** in a subsequent prompt
- This prevents biasing lessons based on the example they were derived from

### 4.4 Output Parsing (`core/ace_roles.py`)

**`parse_generator_output(text)`:**
- Robust parser for model outputs
- Handles: Reasoning/Answer sections, unstructured text, numeric answers
- Falls back gracefully if format is unexpected

**`parse_reflector_output_to_lessons(text)`:**
- Extracts bullet-pointed lessons from reflection output

**`choose_lessons_for_playbook(domain, lessons, existing_playbook)`:**
- Filters: too short, generic, duplicates

---

## 5. CLI Usage

### Single Experiment

```bash
# Baseline
python -m scripts.run_experiment \
    --model-id sshleifer/tiny-gpt2 \
    --task-name tatqa_tiny \
    --mode baseline \
    --output-path results/baseline.csv \
    --metrics-path results/metrics.json \
    --predictions-path results/predictions.jsonl

# ACE Working Memory
python -m scripts.run_experiment \
    --model-id sshleifer/tiny-gpt2 \
    --task-name tatqa_tiny \
    --mode ace \
    --ace-mode ace_working_memory \
    --token-budget 500 \
    --playbook-path playbooks/tatqa.jsonl \
    --output-path results/ace.csv
```

### Grid Experiments

```bash
# Dry run (print commands)
python -m scripts.run_eval_grid --config configs/experiment_grid.yaml --dry-run

# Run all combinations
python -m scripts.run_eval_grid --config configs/experiment_grid.yaml

# With limit per experiment
python -m scripts.run_eval_grid --config configs/experiment_grid.yaml --limit 10
```

---

## 6. Configuration

### `configs/experiment_grid.yaml`

```yaml
models:
  - name: tiny-gpt2
    hf_id: sshleifer/tiny-gpt2
    max_context: 1024

tasks:
  - name: tatqa_tiny
    task_name: tatqa_tiny
    domain: finance

modes:
  - name: baseline
    mode: baseline
  - name: ace_full
    mode: ace
    ace_mode: ace_full
    top_k: 5
  - name: ace_working_memory
    mode: ace
    ace_mode: ace_working_memory
    token_budget: 500

devices:
  - cpu

defaults:
  results_root: results
  max_new_tokens: 256
  temperature: 0.7
```

---

## 7. Output Formats

### Metrics JSON
```json
{
  "run_name": "smoke_baseline",
  "timestamp": "2024-12-09T...",
  "wall_time_seconds": 2.5,
  "model_id": "sshleifer/tiny-gpt2",
  "task_name": "tatqa_tiny",
  "mode": "baseline",
  "accuracy": 0.0,
  "avg_latency_ms": 784.8,
  "playbook": {
    "initial_size": 0,
    "final_size": 3,
    "entries_added": 3
  }
}
```

### Predictions JSONL
```jsonl
{"qid": "tatqa_1", "task": "tatqa_tiny", "model": "...", "mode": "baseline", "gold": "100000", "pred": "...", "is_correct": 0, ...}
{"qid": "tatqa_2", ...}
```

### Playbook JSONL
```jsonl
{"id": "1", "domain": "finance", "text": "For percentage: divide by 100...", "success_count": 3, "failure_count": 1, ...}
```

---

## 8. Tests

```bash
# Run all tests (42 tests)
pytest tests/ -v

# Run specific test file
pytest tests/test_playbook.py -v
pytest tests/test_ace_roles.py -v
```

**Test Coverage:**
- `test_playbook.py`: Scoring formula, eviction, feedback, pruning
- `test_ace_roles.py`: Output parsing robustness, lesson filtering
- `test_model_manager.py`: Model loading, generation

---

## 9. Key Improvements (Phase 2-4)

1. **Formal Retention Scoring**: Implemented the exact formula with configurable hyperparameters
2. **Token-Budget Eviction**: Automatic eviction of low-score entries when over budget
3. **Vagueness Detection**: Heuristic-based scoring to penalize generic lessons
4. **Robust Parsing**: Multi-strategy parsing for model outputs
5. **Enhanced CLI**: Metrics JSON, predictions JSONL, run metadata
6. **Grid Runner**: Systematic model × task × mode × device sweeps
7. **Comprehensive Tests**: 42 unit tests covering all core functionality

---

## 10. Results Aggregation and Plotting

After running experiments, you can aggregate results and generate plots:

### Aggregating Results

```bash
# Aggregate all metrics.json files from results/
python -m scripts.aggregate_results

# Custom paths
python -m scripts.aggregate_results \
    --results-root results/ \
    --output-csv results/summary.csv \
    --output-json results/summary.json
```

This creates:
- `results/summary.csv` - Aggregated metrics in CSV format
- `results/summary.json` - Same data in JSON format

The script:
- Recursively finds all `metrics.json` files
- Extracts: model_id, task_name, mode, ace_mode, device, accuracy, latency, playbook stats
- Prints a summary table to stdout

### Generating Plots

```bash
# Generate plots from summary CSV
python -m scripts.plot_results

# Custom paths
python -m scripts.plot_results \
    --summary-csv results/summary.csv \
    --output-dir results/plots/
```

This creates:
- `results/plots/accuracy_by_mode.png` - Accuracy comparison by mode (baseline vs ACE modes) per task
- `results/plots/accuracy_by_model_and_mode.png` - Accuracy comparison across models and modes

### Complete Workflow

```bash
# 1. Run grid experiments
python -m scripts.run_eval_grid --config configs/experiment_grid.yaml

# 2. Aggregate results
python -m scripts.aggregate_results

# 3. Generate plots
python -m scripts.plot_results
```

---

*Document auto-generated from codebase analysis.*
