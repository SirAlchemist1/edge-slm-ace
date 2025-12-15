# Repository Structure

**Last Updated:** December 2024

## Directory Tree

```
TINY ACE/
├── .github/                      # GitHub configuration
│   ├── CODEOWNERS               # Code ownership
│   └── workflows/               # CI/CD workflows
│       └── ci.yaml              # Continuous integration
│
├── configs/                      # Configuration files
│   └── experiment_grid.yaml     # Experiment configuration
│
├── data/                        # Datasets
│   └── tasks/                   # Task datasets
│       ├── sciq_test.json       # SciQ test set
│       ├── sciq_tiny.json       # SciQ tiny set
│       ├── medqa_tiny.json      # MedQA tiny set
│       └── ...
│
├── docs/                        # Documentation
│   ├── ARCHITECTURE.md          # System architecture
│   ├── RESULTS.md               # Experimental results
│   ├── API.md                   # API reference
│   ├── QUICK_START.md           # Quick start guide
│   ├── INSTALLATION.md          # Installation guide
│   ├── README.md                # Documentation index
│   ├── STRUCTURE.md             # Structure guide
│   ├── PAPER_TRACKING.md        # Paper progress
│   ├── guides/                  # User guides
│   │   ├── EVAL_SCIQ.md
│   │   └── EVAL_SCIQ_MCQ.md
│   └── dev/                     # Development notes
│       ├── DEV_NOTES_PERSON1.md
│       ├── TEAM_NOTES.md
│       └── TEAM_MESSAGES.md
│
├── scripts/                      # CLI scripts
│   ├── run_experiment.py        # Single experiment runner
│   ├── run_eval_grid.py         # Grid experiment runner
│   ├── run_ace_epoch.py         # ACE epoch runner
│   ├── summarize_results.py    # Results aggregation
│   ├── plot_results.py          # Visualization
│   ├── aggregate_results.py    # Results aggregation
│   ├── tinyace_plots.py         # Plotting pipeline
│   └── smoke_test.py            # Smoke tests
│
├── src/edge_slm_ace/            # Core package
│   ├── core/                    # Main logic
│   │   ├── ace_roles.py        # Generator, Reflector, Curator
│   │   └── runner.py           # Evaluation loop
│   ├── memory/                  # Playbook system
│   │   └── playbook.py         # Retention scoring & eviction
│   ├── models/                  # Model management
│   │   └── model_manager.py    # HuggingFace wrapper
│   └── utils/                   # Utilities
│       ├── config.py           # Configurations
│       ├── metrics.py          # Evaluation metrics
│       ├── mcq_eval.py         # MCQ evaluation
│       └── device_utils.py    # Device detection
│
├── tests/                       # Test suite
│   ├── test_ace_roles.py
│   ├── test_playbook.py
│   ├── test_model_manager.py
│   └── ...
│
├── README.md                    # Main README
├── CONTRIBUTING.md              # Contribution guidelines
├── CHANGELOG.md                 # Version history
├── PLOTTING_GUIDE.md            # Plotting instructions
├── LICENSE                      # Apache 2.0 license
├── requirements.txt             # Python dependencies
├── setup.py                     # Package setup
└── pyproject.toml               # Project metadata
```

## File Descriptions

### Root Files

| File | Purpose |
|------|---------|
| `README.md` | Main project documentation and quick start |
| `CONTRIBUTING.md` | Contribution guidelines |
| `CHANGELOG.md` | Version history and changes |
| `PLOTTING_GUIDE.md` | Visualization instructions |
| `LICENSE` | Apache 2.0 license |
| `requirements.txt` | Python dependencies |
| `setup.py` | Package installation script |
| `pyproject.toml` | Project metadata and configuration |

### Core Directories

#### `src/edge_slm_ace/`
Core package containing all implementation code.

#### `scripts/`
Command-line tools for running experiments and analysis.

#### `configs/`
YAML configuration files for experiment grids.

#### `docs/`
Comprehensive documentation including architecture, results, and guides.

#### `tests/`
Pytest test suite for validation.

#### `data/`
Task datasets in JSON/JSONL format.

## Git Ignored

- `results_*/` - Experimental results
- `.venv/`, `venv/` - Virtual environments
- `.pytest_cache/` - Test cache
- `.DS_Store` - macOS system files
- `*.csv`, `*.jsonl` - Result files
- `assets/`, `notebooks/` - User-created directories

## Navigation Guide

- **Getting Started**: Read `README.md` → `docs/QUICK_START.md`
- **Understanding System**: Read `docs/ARCHITECTURE.md`
- **Viewing Results**: Read `docs/RESULTS.md`
- **Running Experiments**: Use `scripts/run_eval_grid.py`
- **Configuration**: Edit `configs/experiment_grid.yaml`
- **API Reference**: See `docs/API.md`
- **Contributing**: Read `CONTRIBUTING.md`
