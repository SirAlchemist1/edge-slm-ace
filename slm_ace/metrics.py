"""Evaluation metrics for model performance."""

from typing import List, Optional, Tuple
import re
import math
from collections import Counter
from difflib import SequenceMatcher


# ------------------------------
# Basic helpers
# ------------------------------

def _norm_text(s: str) -> str:
    """Lowercase + strip + collapse internal whitespace."""
    return re.sub(r"\s+", " ", str(s).strip().lower())


_NUMBER_RE = re.compile(
    r"""
    ^\s*
    (?P<currency>[$])?              # optional leading currency
    (?P<paren_open>\()?             # optional opening parenthesis for negatives
    \s*
    (?P<sign>[+-])?                 # explicit sign
    \s*
    (?P<int>\d{1,3}(?:,\d{3})*|\d+) # integer part w/ optional thousands sep
    (?P<frac>\.\d+)?                # fractional part
    \s*
    (?P<paren_close>\))?            # optional closing parenthesis
    \s*
    (?P<unit>%|kg|g|lb|lbs)?        # very light unit handling
    \s*$
    """,
    re.VERBOSE,
)

_UNIT_TO_BASE = {
    # mass -> base is grams
    "g": ("g", 1.0),
    "kg": ("g", 1000.0),
    "lb": ("g", 453.59237),
    "lbs": ("g", 453.59237),
    # percent handled separately
    "%": ("%", 1.0),
    # currency -> treat as unit "$" but same scale
    "$": ("$", 1.0),
}


def _parse_number_with_unit(s: str) -> Optional[Tuple[float, str]]:
    """
    Parse strings like '$1,234.50', '12.3%', '(1,000)', '5kg', '500 g', '2 lbs'.
    Returns (value_in_base_units, unit_key) where unit_key in {'g','%','$', ''}.
    If unparsable, returns None.
    """
    s0 = str(s).strip()
    s1 = s0.replace(" ", "").lower()
    m = _NUMBER_RE.match(s1)
    if not m:
        return None

    sign = -1.0 if (m.group("paren_open") and m.group("paren_close")) else 1.0
    if m.group("sign") == "-":
        sign *= -1.0

    raw = (m.group("int") or "") + (m.group("frac") or "")
    raw = raw.replace(",", "")
    try:
        num = float(raw)
    except ValueError:
        return None
    num *= sign

    unit = m.group("unit") or ""
    currency = m.group("currency")
    if currency:
        unit = "$"  # prefer currency if present

    # Map to base units when known
    if unit in _UNIT_TO_BASE:
        base_unit, scale = _UNIT_TO_BASE[unit]
        return num * scale, base_unit

    return num, ""


def _token_f1(a: str, b: str) -> float:
    """Token-level F1 over whitespace tokens."""
    ta, tb = _norm_text(a).split(), _norm_text(b).split()
    if not ta and not tb:
        return 1.0
    if not ta or not tb:
        return 0.0
    ca, cb = Counter(ta), Counter(tb)
    inter = sum(min(ca[t], cb.get(t, 0)) for t in ca)
    prec = inter / max(1, len(ta))
    rec = inter / max(1, len(tb))
    return 0.0 if (prec + rec) == 0 else (2 * prec * rec) / (prec + rec)


# ------------------------------
# Numeric & unit-aware similarity
# ------------------------------

def compare_numbers_with_units(pred: str, gold: str, tol: float = 1e-3) -> float:
    """
    Compare numeric answers with light unit handling.
    Returns a similarity in [0, 1].

    Rules:
      - Exact numeric equality within `tol` -> 1.0
      - If both parse but differ, return a score that decays with relative error:
            score = max(0, 1 - min(1, rel_error))
        where rel_error = |p - g| / max(|g|, 1)
      - Units handled: $, %, kg/g, lb/lbs. Others fall back to "no numeric match".
      - '%' is compatible with unitless by trying both (x vs x/100).
      - Currency treated as unit '$' but same scale (format differences ignored).
    """
    pa = _parse_number_with_unit(pred)
    ga = _parse_number_with_unit(gold)
    if not pa or not ga:
        # Try a percent ↔ unitless bridge if one side is a bare percent-like string
        # e.g., "50%" vs "0.5"
        # If either parse failed, attempt a secondary parse that assumes bare percent
        return 0.0

    p_val, p_unit = pa
    g_val, g_unit = ga

    # If units match exactly (including '', '$', '%', 'g')
    def rel_err(x, y):
        denom = max(1.0, abs(y))
        return abs(x - y) / denom

    # Same unit
    if p_unit == g_unit:
        if math.isclose(p_val, g_val, rel_tol=tol, abs_tol=tol):
            return 1.0
        return max(0.0, 1.0 - min(1.0, rel_err(p_val, g_val)))

    # Percent vs unitless: allow comparing p% with g (p/100 vs g) both ways
    if (p_unit == "%" and g_unit == "") or (p_unit == "" and g_unit == "%"):
        # Normalize both interpretations and take best
        if p_unit == "%":
            # compare p/100 vs g
            cand1 = max(0.0, 1.0 - min(1.0, rel_err(p_val / 100.0, g_val)))
            # compare p vs g*100
            cand2 = max(0.0, 1.0 - min(1.0, rel_err(p_val, g_val * 100.0)))
        else:
            cand1 = max(0.0, 1.0 - min(1.0, rel_err(p_val, g_val / 100.0)))
            cand2 = max(0.0, 1.0 - min(1.0, rel_err(p_val * 100.0, g_val)))
        return max(cand1, cand2)

    # Mass units both mapped to 'g' already; any other cross-unit mismatch: give up
    # (Extend here if you want additional unit families.)
    return 0.0


# ------------------------------
# Semantic similarity scorer
# ------------------------------

def semantic_answer_score(pred: str, label: str) -> float:
    """
    Return a similarity score in [0, 1] between `pred` and `label`.

    Combines:
      1) Case-insensitive exact match (fast path)
      2) Numeric/unit-aware comparison (e.g., '5kg' vs '5000 g', '50%' vs '0.5')
      3) Token-level F1
      4) Character-level similarity (SequenceMatcher.ratio)

    Final score is the max of:
      - numeric/unit score
      - the average of (token_f1, sequence_ratio)

    Examples:
        semantic_answer_score("5kg", "5.0 kg")          -> 1.0
        semantic_answer_score("$1,200", "1200")         -> 1.0
        semantic_answer_score("50%", "0.5")             -> ~1.0
        semantic_answer_score("twelve", "12")           -> low (not numeric)
        semantic_answer_score("abc corp", "ABC CORP.")  -> ~1.0
    """
    a = _norm_text(pred)
    b = _norm_text(label)

    # 1) exact (case-insensitive) match
    if a == b:
        return 1.0

    # 2) numeric/unit-aware comparison
    num_score = compare_numbers_with_units(pred, label)

    # 3) token-level F1
    f1 = _token_f1(pred, label)

    # 4) character-level similarity
    seq = SequenceMatcher(None, a, b).ratio()

    # combine non-numeric signals
    text_score = (f1 + seq) / 2.0

    # final: take the best signal
    return max(num_score, text_score)

from typing import List, Dict
import numpy as np

def compute_token_metrics(tokenizer, prompts: List[str], outputs: List[str]) -> Dict[str, float]:
    """
    Compute token usage metrics for a batch of samples.

    Args:
        tokenizer: The same tokenizer used in model_manager (e.g., AutoTokenizer).
        prompts:  List of input prompts sent to the model.
        outputs:  List of generated model responses (decoded text).

    Returns:
        A dict containing per-sample token counts and aggregate means:
            {
                "prompt_tokens": [...],
                "output_tokens": [...],
                "mean_prompt_tokens": float,
                "mean_output_tokens": float
            }
    """
    assert len(prompts) == len(outputs), "Prompts and outputs must have the same length."

    prompt_tokens = [len(tokenizer.encode(p, add_special_tokens=False)) for p in prompts]
    output_tokens = [len(tokenizer.encode(o, add_special_tokens=False)) for o in outputs]

    metrics = {
        "prompt_tokens": prompt_tokens,
        "output_tokens": output_tokens,
        "mean_prompt_tokens": float(np.mean(prompt_tokens) if prompt_tokens else 0),
        "mean_output_tokens": float(np.mean(output_tokens) if output_tokens else 0),
    }
    return metrics

# ------------------------------
# Existing metrics (kept as-is)
# ------------------------------

def compute_accuracy(preds: List[str], labels: List[str]) -> float:
    """
    Compute accuracy using case-insensitive exact match.
    
    Args:
        preds: List of predicted answers.
        labels: List of ground truth answers.
        
    Returns:
        Accuracy as a float between 0 and 1.
    """
    assert len(preds) == len(labels), "Predictions and labels must have same length"
    
    correct = 0
    for pred, label in zip(preds, labels):
        # Simple case-insensitive comparison
        if pred.strip().lower() == label.strip().lower():
            correct += 1
    return correct / len(preds) if len(preds) > 0 else 0.0


def compute_average_latency(latencies_ms: List[float]) -> float:
    """
    Compute average latency in milliseconds.
    
    Args:
        latencies_ms: List of latencies in milliseconds.
        
    Returns:
        Average latency in milliseconds.
    """
    if not latencies_ms:
        return 0.0
    return sum(latencies_ms) / len(latencies_ms)


# ------------------------------
# Convenience: semantic accuracy
# ------------------------------

def compute_semantic_accuracy(
    preds: List[str],
    labels: List[str],
    threshold: float = 0.8,
) -> float:
    """
    Share of (pred, label) with semantic score >= threshold.
    Default threshold 0.8 is strict but tolerant to formatting.
    """
    assert len(preds) == len(labels), "Predictions and labels must have same length"
    if not preds:
        return 0.0
    hits = sum(1 for p, g in zip(preds, labels) if semantic_answer_score(p, g) >= threshold)
    return hits / len(preds)
