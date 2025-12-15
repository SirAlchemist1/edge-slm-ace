# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [0.1.0] - 2024-12-15

### Added
- Initial release of TinyACE framework
- Playbook memory system with retention scoring
- Three evaluation modes: baseline, ACE full, ACE working memory
- Token-budgeted memory management (256/512 token budgets)
- Ablation study support (no_failure, no_recency, no_vagueness, FIFO)
- Comprehensive evaluation metrics (OMA accuracy, semantic similarity, latency)
- Multi-device support (CPU, CUDA, MPS)
- Experimental results on Mistral 7B, Phi-3 Mini, TinyLlama 1.1B
- Complete documentation suite

### Features
- **Core System**
  - ACE loop implementation (Generate → Reflect → Curate → Memorize)
  - Retention scoring formula with four components
  - Strategic forgetting via token-budgeted eviction
  - Feedback-on-use mechanism

- **Evaluation**
  - Baseline mode (vanilla prompting)
  - ACE Full mode (top-k retrieval)
  - ACE Working Memory mode (token-budgeted retrieval)
  - MCQ-aware evaluation (OMA, GOM metrics)

- **Ablation Studies**
  - No failure penalty
  - No recency decay
  - No vagueness penalty
  - FIFO eviction

### Documentation
- Comprehensive architecture documentation
- Experimental results analysis
- Quick start guide
- Installation guide
- Plotting guide
- API documentation

### Results
- Phi-3 Mini: +4% OMA improvement with FIFO eviction
- Mistral 7B: Baseline preferred (96% OMA)
- TinyLlama 1.1B: Baseline preferred (72% OMA)
- Key finding: FIFO eviction outperforms complex scoring

---

## [Unreleased]

### Planned
- Additional model support
- More evaluation tasks
- Performance optimizations
- Extended ablation studies
