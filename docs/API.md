# TinyACE API Reference

## Core Modules

### `edge_slm_ace.core.runner`

Main evaluation loop functions.

#### `run_dataset_baseline()`

Run baseline evaluation without playbook.

```python
from edge_slm_ace.core.runner import run_dataset_baseline

results, summary = run_dataset_baseline(
    model=model,
    tokenizer=tokenizer,
    dataset=dataset,
    domain="science",
    config=ModelConfig(...),
    model_id="microsoft/Phi-3-mini-4k-instruct",
    task_name="sciq_test",
    mode="baseline"
)
```

#### `run_dataset_ace()`

Run ACE-style adaptive evaluation.

```python
from edge_slm_ace.core.runner import run_dataset_ace
from edge_slm_ace.memory.playbook import Playbook

playbook = Playbook(token_budget=256)
results, summary = run_dataset_ace(
    model=model,
    tokenizer=tokenizer,
    dataset=dataset,
    domain="science",
    config=ModelConfig(...),
    playbook=playbook,
    playbook_path=Path("playbook.jsonl"),
    model_id="microsoft/Phi-3-mini-4k-instruct",
    task_name="sciq_test",
    mode="ace",
    ace_mode="ace_working_memory",
    token_budget=256,
    top_k=5,
    reflect_on_correct_every_n=5,
    prune_every_n=10,
    max_entries_per_domain=32
)
```

### `edge_slm_ace.memory.playbook`

Playbook memory system.

#### `Playbook`

Main playbook class for storing and managing domain-specific strategies.

```python
from edge_slm_ace.memory.playbook import Playbook, ScoringParams

# Create playbook with token budget
playbook = Playbook(token_budget=256)

# Create with custom scoring parameters
scoring_params = ScoringParams(
    alpha=1.0,
    beta=0.5,
    gamma=0.3,
    delta=0.4,
    disable_failure_penalty=False,
    disable_recency_decay=False,
    disable_vagueness_penalty=False,
    fifo_memory=False
)
playbook = Playbook(token_budget=256, scoring_params=scoring_params)

# Add entry
entry = playbook.add_entry(
    domain="science",
    text="For physics questions, use F=ma formula",
    step=1
)

# Get top entries
top_entries = playbook.get_top_k(domain="science", k=5, current_step=10)

# Get token-budgeted entries
budget_entries = playbook.get_top_entries_for_budget(
    domain="science",
    token_budget=256,
    current_step=10
)

# Record feedback
playbook.record_feedback(entry_id="1", helpful=True)
playbook.mark_entry_used(entry_id="1", step=10)

# Prune playbook
removed = playbook.prune(max_entries_per_domain=32, current_step=10)
```

#### `PlaybookEntry`

Individual playbook entry.

```python
from edge_slm_ace.memory.playbook import PlaybookEntry

entry = PlaybookEntry(
    id="1",
    domain="science",
    text="Use F=ma for force calculations",
    success_count=5,
    failure_count=1,
    last_used_at=10,
    token_count=10,
    vagueness_score=0.1
)

# Compute score
score = entry.score(current_step=15, params=scoring_params)
```

### `edge_slm_ace.core.ace_roles`

ACE role prompt builders.

#### `build_generator_prompt()`

Build prompt for Generator role.

```python
from edge_slm_ace.core.ace_roles import build_generator_prompt

prompt = build_generator_prompt(
    domain="science",
    playbook=playbook,
    question="What is the force required to accelerate a 10kg object at 2 m/s²?",
    context=None,
    ace_mode="ace_working_memory",
    token_budget=256,
    top_k=5,
    current_step=10
)
```

#### `build_reflector_prompt()`

Build prompt for Reflector role.

```python
from edge_slm_ace.core.ace_roles import build_reflector_prompt

prompt = build_reflector_prompt(
    domain="science",
    question="...",
    context=None,
    model_answer="...",
    ground_truth="...",
    reasoning="..."
)
```

### `edge_slm_ace.models.model_manager`

Model loading and generation.

#### `load_model_and_tokenizer()`

Load HuggingFace model and tokenizer.

```python
from edge_slm_ace.models.model_manager import load_model_and_tokenizer

model, tokenizer = load_model_and_tokenizer(
    model_id="microsoft/Phi-3-mini-4k-instruct",
    device="cuda"
)
```

#### `generate()`

Generate text from model.

```python
from edge_slm_ace.models.model_manager import generate

answer = generate(
    model=model,
    tokenizer=tokenizer,
    prompt="Question: ...",
    max_new_tokens=256,
    temperature=0.0,
    top_p=1.0
)
```

### `edge_slm_ace.utils.metrics`

Evaluation metrics.

#### `semantic_answer_score()`

Compute semantic similarity between answers.

```python
from edge_slm_ace.utils.metrics import semantic_answer_score

score = semantic_answer_score(
    predicted="The force is 20 Newtons",
    gold="20 N"
)
```

#### `compute_accuracy()`

Compute exact match accuracy.

```python
from edge_slm_ace.utils.metrics import compute_accuracy

accuracy = compute_accuracy(
    predictions=["A", "B", "C"],
    labels=["A", "B", "D"]
)
```

## Configuration

### `ModelConfig`

Model configuration dataclass.

```python
from edge_slm_ace.utils.config import ModelConfig

config = ModelConfig(
    model_id="microsoft/Phi-3-mini-4k-instruct",
    max_new_tokens=256,
    temperature=0.0,
    top_p=1.0
)
```

### `ScoringParams`

Scoring parameters for retention scoring.

```python
from edge_slm_ace.memory.playbook import ScoringParams

params = ScoringParams(
    alpha=1.0,           # Success ratio weight
    beta=0.5,            # Failure ratio penalty
    gamma=0.3,           # Recency bonus weight
    delta=0.4,           # Vagueness penalty weight
    lambda_decay=0.05,   # Recency decay rate
    epsilon=1.0,         # Smoothing constant
    disable_vagueness_penalty=False,
    disable_recency_decay=False,
    disable_failure_penalty=False,
    fifo_memory=False
)
```

## Examples

See `scripts/run_experiment.py` for complete usage examples.
