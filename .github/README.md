# TinyACE: Bounded Working-Memory Context Engineering

<div align="center">

**Lightweight Agentic Context Engineering (ACE) for Small Language Models on Edge Devices**

[![License](https://img.shields.io/badge/License-Apache_2.0-blue.svg)](https://opensource.org/licenses/Apache-2.0)
[![Python](https://img.shields.io/badge/python-3.10+-green.svg)](https://www.python.org/)
[![Paper](https://img.shields.io/badge/Paper-PDF-red)](TinyAce%20Paper.pdf)

</div>

---

## 📖 Abstract

**TinyACE** is a lightweight adaptation of Agentic Context Engineering (ACE) designed specifically for Small Language Models (SLMs) with limited context windows and strict latency constraints.

While large models benefit from extensive history, naive context accumulation in SLMs leads to "context saturation" and prompt drift. TinyACE introduces a **Bounded Working-Memory Playbook** with strategic forgetting. Instead of fine-tuning, the system evolves the prompt context through a feedback loop (Generate → Reflect → Prune → Memorize), enabling domain-specific self-improvement on edge hardware.

### Key Findings
* **The Capacity Sweet Spot:** We identify that mid-sized edge models (**Phi-3 Mini, 3.8B**) benefit most from TinyACE, gaining **+4% accuracy** on the SciQ benchmark.
* **Bounded Efficiency:** Very small models (1.1B) collapse under the cognitive load, while large models (7B) saturate the benchmark, making TinyACE uniquely suited for the 3B-4B parameter range.
* **Simplicity Wins:** Our ablation studies reveal that a simple **FIFO eviction policy** often outperforms complex utility scoring for SLMs.

---

## 🏗️ Architecture

TinyACE separates the Generator (SLM) from its memory to manage context within a strict token budget (e.g., 512 tokens).

```mermaid
flowchart LR
    Q[Question] --> G[Generator<br/>SLM]
    WM[(Working<br/>Memory)] -->|Context| G
    G -->|Answer| Eval{Evaluator}
    Eval -- Pass --> End((Output))
    Eval -- Fail --> Ref[Reflector]
    Ref -->|New Lesson| P{Pruner}
    P -- Prune if > T_max --> WM
````

### Retention Scoring Policy

When using the utility-based scoring mode (as opposed to FIFO), lessons are retained based on the following utility function:

$$S(e) = \alpha \frac{N_{succ}}{N_{used}+\epsilon} - \beta \frac{N_{fail}}{N_{used}+\epsilon} + \gamma e^{-\lambda(t-t_{last})} - \delta \mathcal{V}(e)$$

Where:

  * **Success/Failure ($\alpha, \beta$):** Rewards useful strategies and penalizes those leading to hallucinations.
  * **Recency ($\gamma$):** A decay factor to keep context "fresh."
  * **Vagueness ($\delta$):** Penalizes generic strategies to prevent context bloat.

-----

## 📊 Experimental Results

Evaluated on the **SciQ Test Split** ($n=50$ sequential slice) using an NVIDIA A100.

### 1\. The Capacity Sweet Spot

Performance comparison across model scales with a 512-token working memory budget:

| Model | Params | Baseline OMA | TinyACE OMA | Δ Accuracy | Status |
|:---|:---:|:---:|:---:|:---:|:---|
| **TinyLlama** | 1.1B | 72% | 46% | 🔻 -26% | **Collapse** |
| **Phi-3 Mini** | 3.8B | 74% | **78%** | 🟢 **+4%** | **Sweet Spot** |
| **Mistral** | 7B | **96%** | 94% | 🔻 -2% | **Saturation** |

> **Note:** "OMA" (Option-Mapped Accuracy) measures the model's ability to select the correct multiple-choice option via embedding similarity.

### 2\. Ablation Study (Phi-3 Mini)

Analyzing the impact of memory policies (256 token budget):

| Configuration | Accuracy (OMA) | Latency (s) | Insight |
|:---|:---:|:---:|:---|
| **Baseline** | 74% | 6.43s | Fast, moderate accuracy. |
| **TinyACE (FIFO)** | **78%** | 8.99s | **Best Performance.** Simple eviction works best. |
| **No Failure Track** | 70% | 8.88s | Critical component; removing it hurts performance. |
| **No Recency** | 72% | 8.83s | Higher semantic similarity, but lower accuracy. |

-----

## 🚀 Quick Start

### Installation

```bash
git clone [https://github.com/SirAlchemist1/edge-slm-ace.git](https://github.com/SirAlchemist1/edge-slm-ace.git)
cd edge-slm-ace

# Create virtual environment
python -m venv venv
source venv/bin/activate  # or venv\Scripts\activate on Windows

# Install dependencies
pip install -r requirements.txt
pip install -e .
```

### Usage

**1. Run a Baseline Evaluation**

```bash
python -m scripts.run_experiment \
  --model-id microsoft/Phi-3-mini-4k-instruct \
  --task-name sciq_test \
  --mode baseline \
  --device cuda
```

**2. Run TinyACE with Working Memory**

```bash
python -m scripts.run_experiment \
  --model-id microsoft/Phi-3-mini-4k-instruct \
  --task-name sciq_test \
  --mode ace \
  --ace-mode ace_working_memory \
  --token-budget 512 \
  --device cuda
```

**3. Run the Full Evaluation Grid**
To reproduce the paper results:

```bash
python -m scripts.run_eval_grid --config configs/experiment_grid.yaml
```

-----

## 📂 Repository Structure

```
TINY ACE/
├── src/edge_slm_ace/        # Core package
│   ├── core/                # ACE loop (Generator, Reflector, Curator)
│   ├── memory/              # Playbook & Eviction Policies (FIFO/Scoring)
│   └── models/              # HuggingFace Model Wrappers
├── scripts/                 # CLI Tools
│   ├── run_experiment.py    # Single run entry point
│   ├── run_eval_grid.py     # Batch experiment runner
│   └── tinyace_plots.py     # Generate paper figures
├── configs/                 # YAML configurations
├── docs/                    # Documentation & Analysis
└── data/                    # SciQ and other datasets
```

-----

## 📝 Citation

If you use TinyACE or this codebase in your research, please cite the workshop paper:

```bibtex
@article{shahi2024tinyace,
  title={TINYACE: Bounded Working-Memory Context Engineering for Small Language Models},
  author={Shahi, Suryodaya B. and Naik, Sathwik H. and Harsh, Archit},
  journal={University of Maryland, College Park},
  year={2024}
}
```

## 📄 License

This project is licensed under the Apache License 2.0 - see the [LICENSE](https://www.google.com/search?q=LICENSE) file for details.

```
```
