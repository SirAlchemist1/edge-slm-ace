# SciQ MCQ Option-Aware Evaluation

This document describes the option-aware evaluation for SciQ MCQ tasks using the new format (`options` list + `gold_option_idx`).

## Quick Sanity Check

Run a baseline evaluation on `sciq_mcq_test`:

```bash
python -m scripts.run_experiment \
    --model-id microsoft/Phi-3-mini-4k-instruct \
    --task-name sciq_mcq_test \
    --mode baseline \
    --output-path results/sciq_mcq_sanity/results.csv \
    --metrics-path results/sciq_mcq_sanity/metrics.json \
    --limit 5 \
    --device cpu
```

## Pass Criteria

After running the sanity check, verify:

1. **gold_option_idx is not None**: Check that `gold_option_idx` column exists and is populated (should be 0-3)
   ```bash
   python -c "import pandas as pd; df = pd.read_csv('results/sciq_mcq_sanity/results.csv'); print('gold_option_idx values:', df['gold_option_idx'].unique() if 'gold_option_idx' in df.columns else 'MISSING')"
   ```

2. **Results CSV is populated**: Check that results.csv has the new columns
   ```bash
   python -c "import pandas as pd; df = pd.read_csv('results/sciq_mcq_sanity/results.csv'); print('Columns:', [c for c in df.columns if 'option' in c.lower() or 'oma' in c.lower() or 'gom' in c.lower()])"
   ```

3. **oma_correct is not all zero**: At least some predictions should map correctly
   ```bash
   python -c "import pandas as pd; df = pd.read_csv('results/sciq_mcq_sanity/results.csv'); print('oma_correct values:', df['oma_correct'].value_counts() if 'oma_correct' in df.columns else 'MISSING')"
   ```

4. **Metrics JSON includes oma_accuracy and avg_gom**:
   ```bash
   python -c "import json; m = json.load(open('results/sciq_mcq_sanity/metrics.json')); print('oma_accuracy:', m.get('oma_accuracy')); print('avg_gom:', m.get('avg_gom'))"
   ```

## Dataset Format

The new format uses `options` (list) and `gold_option_idx` (int):

```json
{
  "id": "sciq_mcq_1",
  "question": "What is H2O?",
  "options": ["water", "oxygen", "hydrogen", "carbon"],
  "gold_option_idx": 0,
  "answer": "water",
  "support": "H2O is a chemical compound..."
}
```

The loader also supports legacy format (`correct_answer` + `distractor1/2/3`) for backward compatibility.

## New Metrics

- **chosen_option_idx**: Index (0-3) of option with highest semantic similarity to prediction
- **oma_correct**: 1 if chosen_option_idx == gold_option_idx, else 0
- **gom**: similarity(pred, gold_option) - mean(similarity(pred, distractors))
- **gold_option_idx**: The correct option index (0-3)

Run-level metrics:
- **oma_accuracy**: Mean of oma_correct
- **avg_gom**: Mean of gom

## Backward Compatibility

- Existing BLEU and semantic similarity metrics are unchanged
- Legacy format (correct_answer + distractors) still supported
- Non-MCQ tasks work exactly as before (no new columns added)
