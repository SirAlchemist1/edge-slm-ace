# TINYACE: Domain-Specific Benchmarking of Small Language Models for Edge Devices via Agentic Context Engineering (ACE)

**A research-oriented framework for evaluating the efficacy of self-improving, context-based adaptation for Small Language Models (SLMs) on edge-compute environments.** *This repository accompanies the experimental pipeline described in the forthcoming technical report.*

---

## Abstract

Small Language Models (SLMs) are attractive for edge deployment due to their low compute requirements, but they typically underperform on domain-specific reasoning tasks without fine-tuning. This repository investigates whether **Agentic Context Engineering (ACE)**—a mechanism where a model dynamically accumulates *lessons* in a structured *playbook memory*—can substitute for parameter updates in low-compute settings.

We introduce **TinyACE**, a standardized evaluation pipeline that:
1.  Benchmarks SLMs on domain-specific tasks (finance, medical QA, etc.).
2.  Implements an ACE memory loop with formal retention scoring, token-budget constraints, and vagueness penalties.
3.  Automates large experiment sweeps (models × tasks × modes × devices).
4.  Produces aggregated metrics and visualizations suitable for publication.

This framework allows us to rigorously test whether structured external memory can meaningfully improve SLM reasoning without modifying the model’s parameters, offering insights into the viability of **edge-deployable self-improving AI systems**.

---

## Key Contributions

### 1. A Formalized ACE Working-Memory Mechanism
We implement a retention scoring function derived from cognitive memory models:
$$
S(l_i, t) = \alpha \frac{N_{\text{succ}}}{N_{\text{used}}+\epsilon} - \beta \frac{N_{\text{fail}}}{N_{\text{used}}+\epsilon} + \gamma e^{-\lambda (t - t_{\text{last}})} - \delta\, \text{Vagueness}(l_i)
$$
* **Automatic Pruning:** Evicts low-scoring lessons when the context token budget is exceeded.
* **Vagueness Detection:** Penalizes generic advice to keep the context window high-entropy.

### 2. A Unified Experimental Pipeline
* **Modes:** Baseline vs. ACE Full vs. ACE Working-Memory.
* **Grid Search:** Automated sweeping across models (TinyGPT2, Phi-3), tasks, and hardware (CPU, CUDA, MPS).
* **Reproducibility:** Single configuration file drives all experiments.

### 3. Publication-Ready Results Workflow
* Aggregation of distributed `metrics.json` files into summary CSVs.
* Auto-generated plots for accuracy comparisons.
* Standardized report generation.

### 4. Edge-Oriented Evaluation Philosophy
* **No Fine-Tuning:** All adaptation happens via context.
* **Resource Constraints:** Designed for consumer-grade hardware (Laptops, embedded devices).

---

## Method Overview

### ACE Loop (Formal Framework)
For each example (step $t$):
1.  **Retrieve** relevant lessons from playbook (Top-k or Token-Budgeted).
2.  **Generator** produces reasoning trace and final answer.
3.  **Evaluator** scores correctness against ground truth.
4.  **Reflector** (if incorrect) produces new candidate lessons.
5.  **Curator** filters lessons based on quality and redundancy.
6.  **Playbook** updates retention scores and evicts if necessary.

### System Diagram
```text
┌───────────────┐   lessons   ┌──────────────┐
│   PLAYBOOK    │────────────►│   GENERATOR  │
└──────┬────────┘             └──────┬───────┘
       │                             │
       │                     answer + reasoning
       │                             ▼
       │                      ┌──────────────┐
       │                      │   EVALUATOR  │
       │                      └──────┬───────┘
       │                             │ correctness
       │                             ▼
       │                      ┌──────────────┐
       │                      │  REFLECTOR   │
       │                      └──────┬───────┘
       │                             │ lessons
       ▼                             ▼
┌───────────────┐   filtered  ┌──────────────┐
│    MEMORY     │◄────────────│   CURATOR    │
└───────────────┘             └──────────────┘
````

-----

## Evaluation Tasks

We evaluate on domain-specific reasoning datasets. Currently, we utilize "tiny" subsets for rapid prototyping and pipeline validation:

| Task | Domain | Skill Tested |
| :--- | :--- | :--- |
| **TAT-QA** | Finance | Numeric reasoning, table QA |
| **MedQA** | Medical | Clinical recall, MCQ reasoning |
| **GSM8K** | Math | Grade-school chain-of-thought |

-----

## Repository Structure

```text
edge-slm-ace/
├── src/edge_slm_ace/
│   ├── core/runner.py           # Baseline vs ACE evaluation logic
│   ├── core/ace_roles.py        # Prompt builders + output parsing
│   ├── memory/playbook.py       # Retention scoring + token budget eviction
│   ├── models/model_manager.py  # HuggingFace loading + generation wrapper
│   └── utils/...                # Config, metrics, device utils
├── scripts/
│   ├── run_experiment.py        # Single experiment CLI
│   ├── run_eval_grid.py         # Full grid sweep runner
│   ├── summarize_results.py     # Aggregation script
│   └── ...
├── configs/experiment_grid.yaml # Declarative sweep configuration
├── results/                     # Experiment outputs (gitignored)
└── tests/                       # Unit tests (40+ tests covering all modules)
```

-----

## Installation

```bash
git clone [https://github.com/SirAlchemist1/edge-slm-ace](https://github.com/SirAlchemist1/edge-slm-ace)
cd edge-slm-ace

python3.10 -m venv venv
source venv/bin/activate

pip install -r requirements.txt
pip install -e .
```

-----

## Standard Experiment Run (Reproducibility)

To reproduce the results reported in the technical report, follow this sequence:

### 1\. Run the Evaluation Grid

Executes all model $\times$ task $\times$ mode combinations defined in `configs/experiment_grid.yaml`.

```bash
python -m scripts.run_eval_grid --config configs/experiment_grid.yaml
```

### 2\. Aggregate Metrics

Compiles all `metrics.json` files into a single CSV.

```bash
python -m scripts.summarize_results
```

### 3\. Analyze Results

Inspect the summary table:

```bash
cat results/summary.csv
```

*(Optional plotting scripts can be run here if available)*

-----

## Example Commands

### Run a Single Baseline Experiment

```bash
python -m scripts.run_experiment \
  --model-id sshleifer/tiny-gpt2 \
  --task-name tatqa_tiny \
  --mode baseline
```

### Run an ACE Working Memory Experiment

```bash
python -m scripts.run_experiment \
  --model-id sshleifer/tiny-gpt2 \
  --task-name tatqa_tiny \
  --mode ace \
  --ace-mode ace_working_memory \
  --token-budget 500
```

-----

## Experimental Modes

| Mode | Description |
| :--- | :--- |
| **baseline** | Standard zero-shot prompting. No memory. |
| **ace\_full** | Unbounded memory. Retrieves top-k lessons regardless of context length. |
| **ace\_working\_memory** | Bounded memory. Enforces token budget via retention scoring eviction. |

-----

## Limitations & Future Work

**Current Limitations:**

  * Datasets are currently small analysis slices to verify pipeline stability.
  * Evaluation focuses on structural efficacy rather than absolute SOTA accuracy.
  * Models are evaluated without fine-tuning.

**Future Directions:**

  * Expand to full validation/test splits of GSM8K and MedQA.
  * Implement vector-based retrieval (RAG) for larger playbooks.
  * Compare against QLoRA fine-tuned baselines.
  * Integrate reward models for automated reflection quality scoring.

-----

## Citation

If you use this framework in your research, please cite:

```bibtex
@software{tinyace2024,
  title={TinyACE: Agentic Context Engineering for Small Language Models on Edge Devices},
  author={Shahi, Suryodaya and Collaborators},
  year={2024},
  url={[https://github.com/SirAlchemist1/edge-slm-ace](https://github.com/SirAlchemist1/edge-slm-ace)},
}
```

## License

Apache 2.0 — see [LICENSE](https://www.google.com/search?q=LICENSE) file for details.

```
```
