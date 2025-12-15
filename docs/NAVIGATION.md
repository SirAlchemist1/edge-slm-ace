# Navigation Guide

Quick reference for finding information in the TinyACE repository.

## 🚀 Getting Started

**New to TinyACE?**
1. Start with [README.md](../README.md) - Overview and quick start
2. Read [QUICK_START.md](QUICK_START.md) - 5-minute guide
3. Follow [INSTALLATION.md](INSTALLATION.md) - Setup instructions

## 📚 Understanding the System

**Want to understand how it works?**
1. Read [ARCHITECTURE.md](ARCHITECTURE.md) - Complete system design
2. Check [API.md](API.md) - API reference for implementation details
3. Review code in `src/edge_slm_ace/` - Source code

## 📊 Viewing Results

**Interested in experimental findings?**
1. Read [RESULTS.md](RESULTS.md) - Comprehensive results analysis
2. Check `results_models/` - Model comparison results
3. Check `results_ablation/` - Ablation study results

## 🔧 Running Experiments

**Want to run experiments?**
1. Configure `configs/experiment_grid.yaml`
2. Use `scripts/run_eval_grid.py` - Grid runner
3. Use `scripts/run_experiment.py` - Single experiment
4. See [PLOTTING_GUIDE.md](../PLOTTING_GUIDE.md) - Visualization

## 📖 Documentation by Topic

### Architecture & Design
- [ARCHITECTURE.md](ARCHITECTURE.md) - System architecture
- [API.md](API.md) - API reference
- [STRUCTURE.md](STRUCTURE.md) - Repository structure

### Results & Analysis
- [RESULTS.md](RESULTS.md) - Experimental results
- `results_models/` - Model comparison data
- `results_ablation/` - Ablation study data

### Guides & Tutorials
- [QUICK_START.md](QUICK_START.md) - Quick start
- [INSTALLATION.md](INSTALLATION.md) - Installation
- [PLOTTING_GUIDE.md](../PLOTTING_GUIDE.md) - Plotting
- [guides/EVAL_SCIQ.md](guides/EVAL_SCIQ.md) - SciQ evaluation

### Development
- [CONTRIBUTING.md](../CONTRIBUTING.md) - Contribution guidelines
- [dev/DEV_NOTES_PERSON1.md](dev/DEV_NOTES_PERSON1.md) - Dev notes
- [dev/TEAM_NOTES.md](dev/TEAM_NOTES.md) - Team notes

### Project Management
- [PAPER_TRACKING.md](PAPER_TRACKING.md) - Paper progress
- [CHANGELOG.md](../CHANGELOG.md) - Version history

## 🔍 Finding Specific Information

| What You Need | Where to Look |
|---------------|---------------|
| Installation | [INSTALLATION.md](INSTALLATION.md) |
| Quick start | [QUICK_START.md](QUICK_START.md) |
| How it works | [ARCHITECTURE.md](ARCHITECTURE.md) |
| Results | [RESULTS.md](RESULTS.md) |
| API reference | [API.md](API.md) |
| Configuration | `configs/experiment_grid.yaml` |
| Running experiments | `scripts/run_eval_grid.py` |
| Plotting | [PLOTTING_GUIDE.md](../PLOTTING_GUIDE.md) |
| Contributing | [CONTRIBUTING.md](../CONTRIBUTING.md) |

## 📁 Key Directories

- `src/edge_slm_ace/` - Source code
- `scripts/` - CLI tools
- `configs/` - Configuration files
- `docs/` - Documentation
- `data/` - Datasets
- `tests/` - Test suite

## 🎯 Common Tasks

**Run your first experiment:**
```bash
python -m scripts.run_experiment --model-id microsoft/Phi-3-mini-4k-instruct --task-name sciq_test --mode baseline
```

**Understand the scoring formula:**
→ See [ARCHITECTURE.md](ARCHITECTURE.md) → Retention Scoring System

**View experimental results:**
→ See [RESULTS.md](RESULTS.md) → Model Comparison Results

**Configure experiments:**
→ Edit `configs/experiment_grid.yaml`

**Generate plots:**
→ See [PLOTTING_GUIDE.md](../PLOTTING_GUIDE.md)
