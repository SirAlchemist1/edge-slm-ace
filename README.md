# edge-slm-ace

**Domain-Specific Benchmarking of Small Language Models for Edge Devices with Agentic Context Engineering (ACE)**

[![CI](https://github.com/SirAlchemist1/edge-slm-ace/actions/workflows/ci.yaml/badge.svg)](https://github.com/SirAlchemist1/edge-slm-ace/actions/workflows/ci.yaml)
[![License](https://img.shields.io/badge/License-Apache_2.0-blue.svg)](https://opensource.org/licenses/Apache-2.0)

## Overview

This project evaluates Small Language Models (SLMs) on domain-specific tasks for edge devices using **Agentic Context Engineering (ACE)** rather than traditional fine-tuning. We investigate whether dynamic, self-improving context strategies can match or exceed parameter updates for resource-constrained environments.

### Key Innovation

- **No Fine-Tuning**: Adapt frozen SLMs using ACE-style playbook memory
- **Edge-Friendly**: Optimize for Mac M-series, consumer GPUs, and compute-constrained environments
- **Domain-Specific**: Benchmark on TAT-QA (finance), MedQA (medical), and other specialized tasks
- **Self-Learning**: Implement SEAL/SPICE-inspired algorithms for automatic prompt evolution

## Team

- **Person 1 (Suryodaya)**: Infrastructure & Core Framework (Mac M3 Pro)
- **Person 2 (Sathwik)**: Self-Learning Logic, ACE/SEAL/SPICE Prompts & Algorithms  
- **Person 3 (Archit)**: Evaluation, Metrics, Large-Scale Runs (GPU + Supercomputer)

## Architecture

```
slm_ace/
├── model_manager.py    # Load HuggingFace SLMs
├── playbook.py         # ACE-style playbook memory (JSONL)
├── ace_roles.py        # Role-based context strategies
├── runner.py           # Baseline vs ACE experiment runner
├── metrics.py          # Domain-specific evaluation metrics
└── config.py           # Experiment configuration

scripts/
├── run_experiment.py      # Main entry point
├── run_ace_epoch.py       # ACE self-learning loop
└── run_eval_matrix.py     # Multi-model, multi-task evaluation
```

## Quickstart

### Setup

```bash
# Clone the repository
git clone https://github.com/SirAlchemist1/edge-slm-ace.git
cd edge-slm-ace

# Install dependencies
pip install -r requirements.txt

# Install package in editable mode
pip install -e .
```

### Standard Experiment Run

**⚠️ IMPORTANT: Use this exact command sequence for all initial experiments to ensure comparability.**

```bash
# Step 1: Run the evaluation grid (all examples, no limit)
python -m scripts.run_eval_grid \
  --config configs/experiment_grid.yaml

# Step 2: Aggregate results
python -m scripts.aggregate_results

# Step 3: Generate plots
python -m scripts.plot_results
```

This will:
- Run all combinations: 1 model × 3 tasks × 3 modes × 1 device = 9 experiments
- Use all available examples in each dataset (no limit)
- Generate `results/summary.csv` and `results/summary.json`
- Create plots in `results/plots/`

**Note:** Only deviate from this standard run if explicitly changing the experimental setup. This ensures all results are comparable.

# Create virtual environment
python3.10 -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

### Run Baseline Experiment

```bash
python scripts/run_experiment.py \
  --model-id "sshleifer/tiny-gpt2" \
  --dataset-path "data/tasks/tatqa_tiny.json" \
  --domain finance \
  --mode baseline \
  --output-path "results/tatqa_baseline_tiny.csv"
```

### Run ACE Mode

```bash
python scripts/run_experiment.py \
  --model-id "sshleifer/tiny-gpt2" \
  --dataset-path "data/tasks/tatqa_tiny.json" \
  --domain finance \
  --mode ace \
  --playbook-path "playbooks/tatqa_playbook.jsonl" \
  --output-path "results/tatqa_ace_tiny.csv"
```

## Development

### Branch Strategy

- `main`: Protected, stable branch
- `infra/*`: Infrastructure & core framework (Person 1)
- `adapt/*`: ACE/SEAL/SPICE adaptation logic (Person 2)
- `eval/*`: Evaluation & metrics (Person 3)

### Pull Request Workflow

1. Create feature branch: `git checkout -b infra/model-loader`
2. Make changes and commit
3. Push: `git push origin infra/model-loader`
4. Open PR to `main` (requires 1 approval)
5. CI must pass (pytest, formatting checks)
6. Code owners are auto-assigned based on `.github/CODEOWNERS`

### Testing

```bash
# Run all tests
pytest

# With coverage
pytest --cov=slm_ace --cov-report=html

# Specific test file
pytest tests/test_model_manager.py -v
```

### Code Formatting

```bash
# Format code
black .

# Check formatting
black --check .

# Lint
flake8 .
```

## Project Structure (To Be Created)

```
edge-slm-ace/
├── .github/
│   ├── workflows/
│   │   └── ci.yaml              ✓ Created
│   └── CODEOWNERS               ✓ Created
├── slm_ace/                     ⚠ To create
│   ├── __init__.py
│   ├── config.py
│   ├── model_manager.py
│   ├── playbook.py
│   ├── ace_roles.py
│   ├── runner.py
│   ├── metrics.py
│   └── utils.py
├── scripts/                     ⚠ To create
│   ├── run_experiment.py
│   ├── run_ace_epoch.py
│   └── run_eval_matrix.py
├── data/                        ⚠ To create
│   └── tasks/
│       ├── tatqa_tiny.json
│       └── medqa_tiny.json
├── playbooks/                   ⚠ To create
│   ├── tatqa_playbook.jsonl
│   └── medqa_playbook.jsonl
├── tests/                       ⚠ To create
│   ├── test_model_manager.py
│   ├── test_playbook.py
│   └── test_runner_smoke.py
├── notebooks/                   ⚠ To create
│   └── analysis_template.ipynb
├── results/                     ⚠ To create (add to .gitignore)
├── .gitignore                   ✓ Created
├── LICENSE                      ✓ Created
├── README.md                    ✓ Created
├── requirements.txt             ✓ Created
└── pyproject.toml               ✓ Created
```


## License

Apache License 2.0 - See [LICENSE](LICENSE) for details.

## Contributing

See code ownership in [CODEOWNERS](.github/CODEOWNERS). All PRs require review from designated code owners.
