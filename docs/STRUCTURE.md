# Repository Structure

**Last Updated:** December 2024

## Directory Organization

```
TINY ACE/
├── docs/                          # 📚 Documentation
│   ├── ARCHITECTURE.md           # System architecture & design
│   ├── RESULTS.md                # Experimental results & analysis
│   ├── PAPER_TRACKING.md         # Paper writing progress
│   ├── RESTRUCTURING_SUMMARY.md  # Repository restructuring notes
│   ├── README.md                 # Documentation index
│   ├── guides/                   # User guides
│   │   ├── EVAL_SCIQ.md         # SciQ evaluation guide
│   │   ├── EVAL_SCIQ_MCQ.md     # SciQ MCQ evaluation guide
│   │   └── *.old                # Archived guides
│   ├── dev/                      # Development notes
│   │   ├── DEV_NOTES_PERSON1.md # Development notes
│   │   ├── TEAM_NOTES.md        # Team collaboration notes
│   │   └── TEAM_MESSAGES.md     # Team communication logs
│   └── *.old                     # Archived documentation
│
├── src/edge_slm_ace/              # 📦 Core package
│   ├── core/                     # Main logic
│   │   ├── ace_roles.py         # Generator, Reflector, Curator
│   │   └── runner.py            # Evaluation loop
│   ├── memory/                   # Playbook system
│   │   └── playbook.py          # Retention scoring & eviction
│   ├── models/                   # Model management
│   │   └── model_manager.py     # HuggingFace wrapper
│   └── utils/                    # Utilities
│       ├── config.py            # Configurations
│       ├── metrics.py           # Evaluation metrics
│       └── mcq_eval.py          # MCQ evaluation
│
├── scripts/                       # 🔧 CLI Scripts
│   ├── run_experiment.py        # Single experiment runner
│   ├── run_eval_grid.py         # Grid experiment runner
│   ├── run_ace_epoch.py         # ACE epoch runner
│   ├── summarize_results.py     # Results aggregation
│   ├── plot_results.py          # Visualization
│   ├── aggregate_results.py     # Results aggregation
│   ├── run_all_tiny_baselines.py # Baseline runner
│   ├── smoke_test.py            # Smoke tests
│   ├── smoke_gpu_phi3.py        # GPU smoke test
│   └── tinyace_plots.py         # Plotting pipeline
│
├── configs/                       # ⚙️ Configuration
│   └── experiment_grid.yaml     # Experiment configuration
│
├── data/                          # 📊 Datasets
│   └── tasks/                   # Task datasets
│
├── tests/                         # 🧪 Test Suite
│   ├── test_ace_roles.py
│   ├── test_playbook.py
│   ├── test_model_manager.py
│   └── ...
│
├── results_*/                     # 📈 Results (GitIgnored)
│   ├── results_models/          # Model comparison results
│   └── results_ablation/        # Ablation study results
│
├── README.md                      # 📖 Main README
├── PLOTTING_GUIDE.md             # 📊 Plotting guide
├── LICENSE                        # 📄 License (Apache 2.0)
├── requirements.txt               # 📋 Dependencies
├── setup.py                       # 🐍 Python setup
└── pyproject.toml                 # 📦 Project metadata
```

## File Categories

### Core Files (Root)
- **README.md** - Main project documentation
- **LICENSE** - Apache 2.0 license
- **requirements.txt** - Python dependencies
- **setup.py** - Package installation
- **pyproject.toml** - Project metadata
- **PLOTTING_GUIDE.md** - Plotting instructions

### Documentation (`docs/`)
- **Core Docs**: Architecture, Results, Paper Tracking
- **Guides** (`docs/guides/`): Evaluation guides
- **Dev Notes** (`docs/dev/`): Development and team notes
- **Archived** (`*.old`): Historical documentation

### Scripts (`scripts/`)
- **Experiment Runners**: `run_experiment.py`, `run_eval_grid.py`
- **Analysis**: `summarize_results.py`, `plot_results.py`
- **Utilities**: `smoke_test.py`, `tinyace_plots.py`

### Configuration (`configs/`)
- **experiment_grid.yaml** - Main experiment configuration

### Source Code (`src/edge_slm_ace/`)
- **core/** - ACE loop implementation
- **memory/** - Playbook system
- **models/** - Model loading
- **utils/** - Utilities and metrics

## Key Files Reference

| File | Purpose | Location |
|------|---------|----------|
| Main README | Project overview & quick start | `README.md` |
| Architecture | System design details | `docs/ARCHITECTURE.md` |
| Results | Experimental findings | `docs/RESULTS.md` |
| Experiment Config | Grid configuration | `configs/experiment_grid.yaml` |
| Plotting Guide | Visualization instructions | `PLOTTING_GUIDE.md` |
| Core Runner | ACE loop implementation | `src/edge_slm_ace/core/runner.py` |
| Playbook | Memory system | `src/edge_slm_ace/memory/playbook.py` |

## Git Ignored

- `results_*/` - Experimental results
- `*.csv`, `*.jsonl` - Result files
- `.DS_Store` - macOS system files
- `__pycache__/` - Python cache
- `.venv/` - Virtual environments

## Navigation Tips

1. **Getting Started**: Read `README.md`
2. **Understanding System**: Read `docs/ARCHITECTURE.md`
3. **Viewing Results**: Read `docs/RESULTS.md`
4. **Running Experiments**: Use `scripts/run_eval_grid.py`
5. **Generating Plots**: Use `scripts/tinyace_plots.py`
6. **Configuration**: Edit `configs/experiment_grid.yaml`
