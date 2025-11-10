"""Evaluation metrics for model performance."""

from typing import List


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

