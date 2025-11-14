"""Evaluation metrics for model performance."""

from typing import List, Optional
import numpy as np

# Try to import sentence transformers for semantic similarity
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
    
    # TODO(Archit): Add more robust normalization and evaluation
    # - Strip punctuation and whitespace normalization
    # - Handle numeric formats (e.g., "100" vs "100.0" vs "100.00")
    # - Handle date formats and units (e.g., "5kg" vs "5 kg")
    # - Consider semantic similarity for partial credit (e.g., using sentence transformers)
    # - Per-domain metrics (e.g., finance: exact match, medical: F1 score)
    # - Confidence scores if model provides them
    
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


def _get_semantic_model():
    """Lazy load semantic model for similarity computation."""
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
    threshold: float = 0.7,
) -> Optional[float]:
    """
    Compute semantic accuracy using sentence embedding similarity.
    
    Uses sentence transformers if available, otherwise returns None (caller should fall back to exact match).
    
    Args:
        preds: List of predicted answers.
        labels: List of ground truth answers.
        threshold: Similarity threshold for considering answers correct (default: 0.7).
        
    Returns:
        Semantic accuracy as a float between 0 and 1, or None if semantic matching is unavailable.
    """
    assert len(preds) == len(labels), "Predictions and labels must have same length"
    
    if len(preds) == 0:
        return 0.0
    
    model = _get_semantic_model()
    
    if model is None:
        # Return None if semantic model not available (caller should handle fallback)
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
        # If anything fails, return None (caller should handle fallback)
        print(f"[metrics] Error computing semantic accuracy: {e}; falling back to exact match")
        return None


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

