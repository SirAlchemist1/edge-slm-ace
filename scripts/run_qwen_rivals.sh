#!/bin/bash
# =============================================================================
# Qwen2.5 Rival Experiments Runner Shell Script
# =============================================================================
# This script runs controlled experiments comparing Qwen2.5 models against 
# existing baselines (TinyLlama-1.1B, Phi-3-mini-3.8B, Mistral-7B).
#
# Usage:
#   ./scripts/run_qwen_rivals.sh [options]
#
# Options:
#   --dry-run           Print experiment plan without running
#   --small             Run only small models (~1-2B)
#   --medium            Run only medium models (~3-4B)
#   --large             Run only large models (~7B)
#   --device DEVICE     Device to use (cuda, mps, cpu). Default: cuda
#   --help              Show this help message
#
# Examples:
#   # Run all experiments on GPU
#   ./scripts/run_qwen_rivals.sh
#
#   # Dry run to preview
#   ./scripts/run_qwen_rivals.sh --dry-run
#
#   # Run only medium-sized models
#   ./scripts/run_qwen_rivals.sh --medium
#
#   # Run on CPU (slower but no GPU required)
#   ./scripts/run_qwen_rivals.sh --device cpu
# =============================================================================

set -e

# Change to repository root
cd "$(dirname "$0")/.."

# Default values
MODEL_CLASS="all"
DEVICE="cuda"
DRY_RUN=""
EXTRA_ARGS=""

# Parse arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --dry-run)
            DRY_RUN="--dry-run"
            shift
            ;;
        --small)
            MODEL_CLASS="small"
            shift
            ;;
        --medium)
            MODEL_CLASS="medium"
            shift
            ;;
        --large)
            MODEL_CLASS="large"
            shift
            ;;
        --device)
            DEVICE="$2"
            shift 2
            ;;
        --help|-h)
            head -35 "$0" | tail -30
            exit 0
            ;;
        *)
            EXTRA_ARGS="$EXTRA_ARGS $1"
            shift
            ;;
    esac
done

echo "=============================================================="
echo "TinyACE - Qwen2.5 Rival Experiments"
echo "=============================================================="
echo "Model class: $MODEL_CLASS"
echo "Device: $DEVICE"
if [ -n "$DRY_RUN" ]; then
    echo "Mode: DRY RUN"
fi
echo "=============================================================="
echo ""

# Run the Python script
python -m scripts.run_qwen_rivals \
    --model-class "$MODEL_CLASS" \
    --device "$DEVICE" \
    $DRY_RUN \
    $EXTRA_ARGS

echo ""
echo "=============================================================="
echo "Qwen2.5 rival experiments complete!"
echo ""
echo "Results saved to:"
echo "  - results/qwen_rivals/results_models_qwen.json"
echo "  - results/qwen_rivals/results_stability_qwen.csv"
echo "  - results/qwen_rivals/figures/"
echo "  - paper_snippets/"
echo "=============================================================="
