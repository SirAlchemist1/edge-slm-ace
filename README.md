# 📘 TinyACE Evaluation Framework

**Domain-Specific Benchmarking of Small Language Models on Edge Devices via Agentic Context Engineering.**

[![License](https://img.shields.io/badge/License-Apache_2.0-blue.svg)](https://opensource.org/licenses/Apache-2.0)
[![Python](https://img.shields.io/badge/python-3.10+-green.svg)](https://www.python.org/)
[![Paper](https://img.shields.io/badge/Paper-Technical_Report-red)](tinyace.pdf)

---

## 📖 Abstract

Small Language Models (SLMs) like Phi-3 and TinyLlama are efficient for edge deployment but often lack the reasoning depth of larger models. **TinyACE** (Agentic Context Engineering) is a framework that enables SLMs to "self-improve" without fine-tuning by maintaining a dynamic **Playbook Memory** of past lessons.

This repository contains the complete experimental pipeline to:
1.  **Run** SLMs on domain tasks (MedQA, GSM8K, Finance).
2.  **Execute** the ACE loop (Attempt $\to$ Reflect $\to$ Curate $\to$ Memorize).
3.  **Benchmark** performance across CPU, CUDA, and MPS (Apple Silicon).

---

## 🏗️ System Architecture

The core of TinyACE is a feedback loop that freezes model weights but evolves the prompt context.

```mermaid
flowchart LR
    subgraph Memory System
    P[(Playbook)]
    end

    subgraph ACE Loop
    Q(Question) --> Ret{Retrieval}
    P --> Ret
    Ret -->|Context| G[Generator]
    G -->|Reasoning + Answer| Eval{Evaluator}
    
    Eval -->|Correct| Update[Update Stats]
    Eval -->|Incorrect| Ref[Reflector]
    
    Ref -->|New Lesson| Cur[Curator]
    Cur -->|Filter & Prune| P
    Update --> P
    end
````

### Key Mechanisms

  * **Formal Retention Scoring:** Lessons are scored and evicted based on:
    $$ S(l) = \alpha \cdot \text{Success} - \beta \cdot \text{Failure} + \gamma \cdot e^{-\lambda t} - \delta \cdot \text{Vagueness} $$
  * **Token-Budgeted Memory:** The context window is strictly capped (e.g., 1500 tokens), forcing the system to prioritize high-value lessons.
  * **Edge Optimization:** Native support for `mps` (Mac) and quantized execution.

-----

## 📂 Repository Structure

```text
edge-slm-ace/
├── src/edge_slm_ace/
│   ├── core/            # Main logic (runner.py, ace_roles.py)
│   ├── memory/          # Playbook & Retention scoring
│   ├── models/          # HuggingFace wrapper (CPU/GPU/MPS)
│   └── utils/           # Metrics & Logging
├── configs/             # Experiment configurations (YAML)
├── scripts/             # CLI entry points
│   ├── run_experiment.py
│   ├── run_eval_grid.py
│   └── summarize_results.py
├── data/                # Dataset subsets
├── results/             # Logs & Artifacts (GitIgnored)
└── tests/               # Pytest suite
```

-----

## 🚀 Quick Start

### 1\. Installation

```bash
git clone [https://github.com/SirAlchemist1/edge-slm-ace.git](https://github.com/SirAlchemist1/edge-slm-ace.git)
cd edge-slm-ace

# Create virtual environment
python -m venv venv
source venv/bin/activate

# Install package in editable mode
pip install -r requirements.txt
pip install -e .
```

### 2\. Run a "Smoke Test"

Verify the pipeline works on your hardware (CPU/GPU/MPS).

```bash
# Run a single baseline experiment on a tiny dataset
python -m scripts.run_experiment \
  --model-id sshleifer/tiny-gpt2 \
  --task-name tatqa_tiny \
  --mode baseline
```

### 3\. Run the Full Evaluation Grid

To reproduce paper results (sweeping models × tasks × modes):

```bash
python -m scripts.run_eval_grid --config configs/experiment_grid.yaml
```

-----

## 📊 Analyzing Results

After the grid run completes, aggregate the distributed metrics:

```bash
python -m scripts.summarize_results
```

This generates `results/summary.csv`:

| Model | Task | Mode | Accuracy | Avg Tokens |
|-------|------|------|----------|------------|
| Phi-3 | GSM8K | Baseline | 48.5% | 120 |
| Phi-3 | GSM8K | ACE-WM | **56.8%** | 1456 |

-----

##  Configuration

Modify `configs/experiment_grid.yaml` to control the sweep:

```yaml
models:
  - name: phi-3-mini
    hf_id: microsoft/Phi-3-mini-4k-instruct

modes:
  - name: baseline
  - name: ace_working_memory
    ace_mode: ace_working_memory
    token_budget: 1500  # Max context tokens for memory
    top_k: 5

devices:
  - mps   # Use Apple Silicon
  - cuda  # Use NVIDIA GPU
```

-----

## Citation

If you use this codebase, please cite:

```bibtex
@software{tinyace2024,
  title={TinyACE: Lightweight Self-Improving Small Language Models for Edge Devices},
  author={Shahi, Suryodaya and Collaborators},
  year={2024},
  url={[https://github.com/SirAlchemist1/edge-slm-ace](https://github.com/SirAlchemist1/edge-slm-ace)}
}
```

## License

Apache 2.0

```
```
