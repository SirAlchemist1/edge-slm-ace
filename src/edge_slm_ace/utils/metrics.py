"""Evaluation metrics for model performance."""

from typing import List, Optional, Tuple, Dict
import re
import math
from collections import Counter
from difflib import SequenceMatcher
import numpy as np

# Try to import sentence transformers for semantic similarity (optional)
try:
    from sentence_transformers import SentenceTransformer
    _SEMANTIC_MODEL_AVAILABLE = True
    _SEMANTIC_WARNING_PRINTED = False
except ImportError:
    SentenceTransformer = None
    _SEMANTIC_MODEL_AVAILABLE = False
    _SEMANTIC_WARNING_PRINTED = False

# Lazy load semantic model
_semantic_model_cache = None

# Try to import sacrebleu for BLEU scores (optional)
try:
    import sacrebleu
    _BLEU_AVAILABLE = True
except ImportError:
    sacrebleu = None
    _BLEU_AVAILABLE = False


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
    return 0.0


# ------------------------------
# Semantic similarity scorer (Archit's version)
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


# ------------------------------
# BLEU score (Archit's addition)
# ------------------------------

def compute_bleu_score(prediction: str, reference: str) -> float:
    """
    Compute BLEU score between a prediction and reference using sacrebleu.

    Args:
        prediction: The predicted/generated text.
        reference: The ground truth/reference text.

    Returns:
        BLEU score as a float between 0 and 100.
    """
    if not _BLEU_AVAILABLE:
        raise ImportError("sacrebleu is required for BLEU scores. Install with: pip install sacrebleu")
    
    # sacrebleu expects references as a list of lists
    # Each reference needs to be a list (for multiple references per prediction)
    # We only have one reference per prediction
    bleu = sacrebleu.sentence_bleu(prediction, [reference])
    return bleu.score


# ------------------------------
# Token metrics (Archit's addition)
# ------------------------------

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
# Basic accuracy metrics
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


def compute_average_tokens(token_counts: List[int]) -> float:
    """
    Compute average token count.
    
    Args:
        token_counts: List of token counts.
        
    Returns:
        Average token count.
    """
    if not token_counts:
        return 0.0
    return sum(token_counts) / len(token_counts)


# ------------------------------
# Semantic accuracy (two implementations)
# ------------------------------

def _get_semantic_model():
    """Lazy load semantic model for similarity computation (sentence-transformers)."""
    global _semantic_model_cache, _SEMANTIC_WARNING_PRINTED
    
    if not _SEMANTIC_MODEL_AVAILABLE:
        if not _SEMANTIC_WARNING_PRINTED:
            print("[metrics] sentence-transformers not installed; semantic accuracy will use exact match fallback.")
            _SEMANTIC_WARNING_PRINTED = True
        return None
    
    if _semantic_model_cache is None:
        try:
            # Use a small, fast model for semantic similarity
            _semantic_model_cache = SentenceTransformer('all-MiniLM-L6-v2')
        except Exception as e:
            # If loading fails, disable semantic matching
            if not _SEMANTIC_WARNING_PRINTED:
                print(f"[metrics] Failed to load sentence-transformers model: {e}; semantic accuracy will use exact match fallback.")
                _SEMANTIC_WARNING_PRINTED = True
            return None
    
    return _semantic_model_cache


def compute_semantic_accuracy(
    preds: List[str],
    labels: List[str],
    threshold: float = 0.8,
    use_sentence_transformers: bool = False,
) -> Optional[float]:
    """
    Compute semantic accuracy using similarity scoring.
    
    Two modes:
    1. Default (use_sentence_transformers=False): Uses semantic_answer_score (Archit's version)
       - Fast, rule-based similarity (numeric/unit-aware, token F1, character similarity)
       - No external dependencies required
       - Returns float (never None)
    
    2. use_sentence_transformers=True: Uses sentence-transformers embeddings
       - More sophisticated semantic similarity via embeddings
       - Requires sentence-transformers package
       - Returns float or None if unavailable
    
    Args:
        preds: List of predicted answers.
        labels: List of ground truth answers.
        threshold: Similarity threshold for considering answers correct (default: 0.8).
        use_sentence_transformers: If True, use sentence-transformers; else use rule-based scoring.
        
    Returns:
        Semantic accuracy as a float between 0 and 1.
        Returns None only if use_sentence_transformers=True and model unavailable.
    """
    assert len(preds) == len(labels), "Predictions and labels must have same length"
    
    if len(preds) == 0:
        return 0.0
    
    if use_sentence_transformers:
        # Use sentence-transformers approach (from HEAD)
        model = _get_semantic_model()
        
        if model is None:
            return None
        
        try:
            # Compute embeddings
            pred_embeddings = model.encode(preds, convert_to_numpy=True)
            label_embeddings = model.encode(labels, convert_to_numpy=True)
            
            # Compute cosine similarity
            from sklearn.metrics.pairwise import cosine_similarity
            similarities = cosine_similarity(pred_embeddings, label_embeddings)
            
            # Extract diagonal (pred[i] vs label[i])
            diagonal_similarities = np.diag(similarities)
            
            # Count correct (similarity >= threshold)
            correct = np.sum(diagonal_similarities >= threshold)
            
            return float(correct / len(preds))
        except Exception as e:
            print(f"[metrics] Error computing semantic accuracy with sentence-transformers: {e}; falling back to rule-based")
            # Fall through to rule-based method
    else:
        # Use rule-based semantic_answer_score (Archit's version)
        hits = sum(1 for p, g in zip(preds, labels) if semantic_answer_score(p, g) >= threshold)
        return hits / len(preds)
    
    # Fallback to rule-based if sentence-transformers failed
    hits = sum(1 for p, g in zip(preds, labels) if semantic_answer_score(p, g) >= threshold)
    return hits / len(preds)
