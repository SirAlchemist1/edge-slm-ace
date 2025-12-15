# Quick Start Guide

**Get started with TinyACE in 5 minutes**

## Prerequisites

- Python 3.10 or higher
- CUDA-capable GPU (optional, for GPU acceleration)
- 8GB+ RAM recommended

## Installation

```bash
# Clone repository
git clone https://github.com/SirAlchemist1/edge-slm-ace.git
cd edge-slm-ace

# Create and activate virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
pip install -e .
```

## Quick Test

Run a smoke test to verify installation:

```bash
# CPU test (uses tiny-gpt2)
python -m scripts.smoke_test

# GPU test (uses Phi-3 Mini)
python -m scripts.smoke_gpu_phi3 --task-name tatqa_tiny --device cuda --limit 2
```

## Run Your First Experiment

### Baseline Evaluation

```bash
python -m scripts.run_experiment \
  --model-id microsoft/Phi-3-mini-4k-instruct \
  --task-name sciq_test \
  --mode baseline \
  --device cuda \
  --limit 10
```

### ACE Working Memory Mode

```bash
python -m scripts.run_experiment \
  --model-id microsoft/Phi-3-mini-4k-instruct \
  --task-name sciq_test \
  --mode ace \
  --ace-mode ace_working_memory \
  --token-budget 256 \
  --device cuda \
  --limit 10
```

## View Results

Results are saved to:
- `results/{model}/{task}/{mode}/{device}/results.csv` - Per-example results
- `results/{model}/{task}/{mode}/{device}/metrics.json` - Aggregate metrics

## Next Steps

- Read [ARCHITECTURE.md](ARCHITECTURE.md) to understand the system
- Check [RESULTS.md](RESULTS.md) for experimental findings
- See [PLOTTING_GUIDE.md](../PLOTTING_GUIDE.md) for visualization
- Review [configs/experiment_grid.yaml](../configs/experiment_grid.yaml) for configuration options
