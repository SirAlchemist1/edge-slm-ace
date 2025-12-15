# TinyACE Architecture Documentation

**Last Updated:** December 2024  
**Version:** 1.0

---

## Table of Contents

1. [System Overview](#system-overview)
2. [Core Components](#core-components)
3. [ACE Loop Implementation](#ace-loop-implementation)
4. [Retention Scoring System](#retention-scoring-system)
5. [Modes of Operation](#modes-of-operation)
6. [Ablation Studies](#ablation-studies)

---

## System Overview

TinyACE (Agentic Context Engineering) is a framework that enables Small Language Models (SLMs) to self-improve without fine-tuning by maintaining a dynamic **Playbook Memory** of domain-specific strategies.

### Key Design Principles

1. **No Model Fine-tuning**: Model weights remain frozen; only prompt context evolves
2. **Domain-Specific Learning**: Playbook accumulates strategies specific to each domain
3. **Strategic Forgetting**: Token-budgeted memory forces prioritization of high-value lessons
4. **Feedback-Driven**: Lessons are evaluated based on actual usage, not creation context

---

## Core Components

### 1. Playbook Memory (`src/edge_slm_ace/memory/playbook.py`)

The playbook stores domain-specific lessons (strategies) that guide the model's reasoning.

**Key Features:**
- Retention scoring for entry prioritization
- Token-budgeted eviction for working memory mode
- Success/failure tracking per entry
- Vagueness detection to filter generic lessons

**Data Structure:**
```python
@dataclass
class PlaybookEntry:
    id: str
    domain: str
    text: str
    success_count: int = 0
    failure_count: int = 0
    last_used_at: int = 0
    token_count: int = 0
    vagueness_score: float = 0.0
```

### 2. ACE Roles (`src/edge_slm_ace/core/ace_roles.py`)

Three roles orchestrate the ACE loop:

**Generator:**
- Builds prompts with playbook context
- Generates answers with step-by-step reasoning
- Parses output to extract reasoning and answer

**Reflector:**
- Analyzes incorrect answers (or periodically correct ones)
- Generates specific, actionable lessons
- Filters out generic/vague advice

**Curator:**
- Filters and deduplicates lessons
- Adds lessons to playbook without initial feedback
- Ensures only specific, actionable rules are stored

### 3. Runner (`src/edge_slm_ace/core/runner.py`)

Orchestrates the evaluation pipeline:

- **Baseline Mode**: Vanilla prompting without playbook
- **ACE Full Mode**: Top-k retrieval from unbounded playbook
- **ACE Working Memory Mode**: Token-budgeted retrieval with eviction

---

## ACE Loop Implementation

The ACE loop processes each example through these steps:

```python
for step, example in enumerate(dataset):
    # 1. RETRIEVE: Get relevant lessons from playbook
    if ace_mode == "ace_working_memory":
        used_entries = playbook.get_top_entries_for_budget(
            domain, token_budget, current_step
        )
    else:
        used_entries = playbook.get_top_k(domain, k=top_k, current_step)
    
    # 2. GENERATE: Build prompt with lessons → generate answer
    prompt = build_generator_prompt(domain, playbook, question, context, ...)
    answer, reasoning = parse_generator_output(generate(model, prompt))
    
    # 3. EVALUATE: Check correctness
    correct = (answer.lower() == ground_truth.lower())
    
    # 4. REFLECT: Generate lessons if incorrect or periodically
    if not correct or (step % reflect_every_n == 0):
        lessons = generate_and_parse_reflection(...)
        for lesson in filtered_lessons:
            playbook.add_entry(domain, lesson, step)
    
    # 5. FEEDBACK: Record success/failure for USED entries only
    for entry_id in used_entry_ids:
        playbook.mark_entry_used(entry_id, step)
        playbook.record_feedback(entry_id, helpful=correct)
    
    # 6. PRUNE: Periodically remove low-scoring entries
    if step % prune_every_n == 0:
        playbook.prune(max_entries_per_domain)
```

### Critical Design Decision: Feedback on Use, Not Creation

New lessons are added **without** feedback. Feedback is only recorded when a lesson is **actually used** in a subsequent prompt. This prevents biasing lessons based on the example they were derived from.

---

## Retention Scoring System

### Formal Scoring Equation

```
S(l_i, t) = α·(N_succ/(N_used+ε)) - β·(N_fail/(N_used+ε)) 
          + γ·exp(-λ·(t-t_last)) - δ·V(l_i)
```

**Hyperparameters:**
- `α = 1.0`: Success ratio weight
- `β = 0.5`: Failure ratio penalty weight
- `γ = 0.3`: Recency bonus weight
- `δ = 0.4`: Vagueness penalty weight
- `λ = 0.05`: Recency decay rate
- `ε = 1.0`: Smoothing constant

### Component Breakdown

1. **Success Term**: `α · (N_succ / (N_used + ε))`
   - Rewards entries that lead to correct answers
   - Higher success rate → higher score

2. **Failure Term**: `-β · (N_fail / (N_used + ε))`
   - Penalizes entries that lead to incorrect answers
   - Higher failure rate → lower score
   - **Critical**: Ablation shows OMA drops 4% without this term

3. **Recency Term**: `γ · exp(-λ · (t - t_last))`
   - Bonus for recently used entries
   - Decays exponentially with time since last use
   - Trade-off: Helps accuracy but may reduce semantic quality

4. **Vagueness Term**: `-δ · V(l_i)`
   - Penalizes vague/generic lessons
   - `V(l_i)` computed from:
     - Generic phrases ("think carefully", "pay attention")
     - Short text (< 5 words)
     - Lack of specificity (no numbers, formulas, domain terms)
   - Prevents playbook bloat

### Implementation

```python
def score(self, current_step: int, params: ScoringParams) -> float:
    # Success ratio term
    success_term = params.alpha * (self.success_count / (n_used + params.epsilon))
    
    # Failure ratio term (can be disabled)
    if params.disable_failure_penalty:
        failure_term = 0.0
    else:
        failure_term = params.beta * (self.failure_count / (n_used + params.epsilon))
    
    # Recency term (can be disabled)
    if params.disable_recency_decay:
        recency_term = 0.0
    else:
        age = max(0, current_step - self.last_used_at)
        recency_term = params.gamma * math.exp(-params.lambda_decay * age)
    
    # Vagueness penalty (can be disabled)
    if params.disable_vagueness_penalty:
        vagueness_term = 0.0
    else:
        vagueness_term = params.delta * self.vagueness_score
    
    return success_term - failure_term + recency_term - vagueness_term
```

---

## Modes of Operation

### 1. Baseline Mode

- **No playbook**: Vanilla prompting
- **Prompt**: `Question: {question}\nAnswer:`
- **Use Case**: Baseline comparison, strong models that don't need context

### 2. ACE Full Mode

- **Retrieval**: Top-k entries by retention score
- **Playbook**: Unbounded (pruned periodically)
- **Use Case**: Full context available, no token constraints
- **Configuration**: `ace_mode: ace_full`, `top_k: 5`

### 3. ACE Working Memory Mode

- **Retrieval**: Token-budgeted entries (256 or 512 tokens)
- **Eviction**: Lowest-scoring entries evicted when over budget
- **Use Case**: Edge devices, limited context windows
- **Configuration**: `ace_mode: ace_working_memory`, `token_budget: 256`

---

## Ablation Studies

The codebase supports ablation studies through scoring parameter flags:

### Available Ablations

1. **No Failure Penalty** (`disable_failure_penalty=True`)
   - Sets `β=0`, removes failure term
   - **Result**: OMA drops 4% (critical component)

2. **No Recency Decay** (`disable_recency_decay=True`)
   - Sets `γ=0`, removes recency term
   - **Result**: Lower OMA but best semantic similarity (trade-off)

3. **No Vagueness Penalty** (`disable_vagueness_penalty=True`)
   - Sets `δ=0`, removes vagueness term
   - **Result**: Playbook grows larger, slight OMA drop

4. **FIFO Eviction** (`fifo_memory=True`)
   - Bypasses scoring entirely, uses insertion order
   - **Result**: Best OMA (78%), simple and effective

### Configuration

Ablations are configured in `configs/experiment_grid.yaml`:

```yaml
modes:
  - name: tinyace_ablate_no_failure
    disable_failure_penalty: true
  - name: tinyace_ablate_no_recency
    disable_recency_decay: true
  - name: tinyace_ablate_no_vagueness
    disable_vagueness_penalty: true
  - name: tinyace_fifo
    fifo_memory: true
```

---

## File Structure

```
src/edge_slm_ace/
├── core/
│   ├── ace_roles.py      # Generator, Reflector, Curator prompts
│   └── runner.py          # Main evaluation loop
├── memory/
│   └── playbook.py       # Playbook storage and scoring
├── models/
│   └── model_manager.py  # HuggingFace model loading
└── utils/
    ├── config.py         # Model/task configurations
    ├── metrics.py        # Evaluation metrics
    └── mcq_eval.py       # MCQ-specific evaluation
```

---

## Key Design Insights

1. **Feedback on Use**: Lessons evaluated on future examples, not creation context
2. **Strategic Forgetting**: Token budgets force prioritization
3. **Component Importance**: Failure tracking is critical; FIFO can match complex scoring
4. **Model-Dependent**: ACE helps medium models (Phi-3) more than strong (Mistral) or weak (TinyLlama)

---

## References

- See `docs/RESULTS.md` for detailed experimental results
- See `README.md` for quick start guide
- See `configs/experiment_grid.yaml` for configuration options
