"""MCQ-aware evaluation metrics for SciQ-style multiple choice questions.

This module provides evaluation metrics specifically designed for multiple-choice
questions with option texts (like SciQ). These metrics are complementary to the
standard exact-match accuracy and provide deeper insight into model performance
on MCQ tasks.

New Metrics:
    - Option-Mapped Accuracy (OMA): Maps model predictions to closest option via
      semantic similarity and checks if it matches the gold option.
    - Gold Option Margin (GOM): Measures the margin between similarity to gold
      option versus average similarity to distractors.
    - Answerable Choice Rate (ACR): Measures how often the model outputs a clear
      choice marker (A/B/C/D) in its response.

Usage:
    from edge_slm_ace.utils.mcq_eval import MCQEvaluator, is_sciq_task
    
    if is_sciq_task(task_name):
        evaluator = MCQEvaluator.get_instance()
        metrics = evaluator.evaluate_mcq(
            prediction="The answer is B because...",
            options={"A": "oxygen", "B": "carbon dioxide", "C": "nitrogen", "D": "helium"},
            gold_option="B",
        )
"""

import re
from typing import Dict, List, Optional, Tuple
import numpy as np


def is_sciq_task(task_name: str) -> bool:
    """
    Check if a task is SciQ-style (MCQ with options).
    
    Gated by task name to ensure backward compatibility with other datasets.
    
    Args:
        task_name: The task name (e.g., "sciq_tiny", "sciq_test").
        
    Returns:
        True if the task is SciQ-style, False otherwise.
    """
    if task_name is None:
        return False
    return "sciq" in task_name.lower()


def has_mcq_options(example: Dict) -> bool:
    """
    Check if an example has MCQ options (SciQ format).
    
    Supports two formats:
    1. Legacy: correct_answer + distractor1/2/3
    2. New: options (list) + gold_option_idx (int)
    
    Args:
        example: A dataset example dict.
        
    Returns:
        True if the example has MCQ options in either format.
    """
    # Check for new format: options list + gold_option_idx
    if "options" in example and isinstance(example.get("options"), list) and len(example.get("options", [])) == 4:
        if "gold_option_idx" in example:
            return True
    
    # Check for legacy format: correct_answer + distractors
    return (
        "correct_answer" in example and
        "distractor1" in example and
        "distractor2" in example and
        "distractor3" in example
    )


def extract_mcq_options(example: Dict) -> Tuple[Dict[str, str], str, str]:
    """
    Extract MCQ options from a SciQ-format example (legacy format).
    
    Converts SciQ format (correct_answer + 3 distractors) to standard
    A/B/C/D option format. The correct answer is placed at a consistent
    position (A) for deterministic behavior.
    
    Args:
        example: A SciQ-format example dict with keys:
            - correct_answer: The correct answer text
            - distractor1, distractor2, distractor3: Incorrect option texts
            
    Returns:
        Tuple of (options_dict, gold_option, gold_text):
            - options_dict: {"A": text, "B": text, "C": text, "D": text}
            - gold_option: The letter of the correct option (always "A")
            - gold_text: The text of the correct answer
    """
    correct = example.get("correct_answer", "")
    d1 = example.get("distractor1", "")
    d2 = example.get("distractor2", "")
    d3 = example.get("distractor3", "")
    
    # Place correct answer at A for consistency
    # Note: This is deterministic to ensure reproducible evaluation.
    # In real MCQ scenarios, you might randomize option order.
    options = {
        "A": correct,
        "B": d1,
        "C": d2,
        "D": d3,
    }
    
    return options, "A", correct


def extract_mcq_options_with_indices(example: Dict) -> Tuple[List[str], int]:
    """
    Extract MCQ options from an example in the new format (options list + gold_option_idx).
    
    Supports two formats:
    1. New format: options (list of 4 strings) + gold_option_idx (int 0-3)
    2. Legacy format: correct_answer + distractor1/2/3 (converted to new format)
    
    Args:
        example: A dataset example dict with either:
            - New format: "options" (list) and "gold_option_idx" (int)
            - Legacy format: "correct_answer" and "distractor1/2/3"
            
    Returns:
        Tuple of (options_list, gold_option_idx):
            - options_list: List of 4 option strings ["option0", "option1", "option2", "option3"]
            - gold_option_idx: Integer index (0-3) of the correct option
    """
    # Check for new format first
    if "options" in example and isinstance(example.get("options"), list):
        options = example["options"]
        if len(options) == 4 and "gold_option_idx" in example:
            gold_idx = example["gold_option_idx"]
            if 0 <= gold_idx < 4:
                return options, gold_idx
    
    # Fall back to legacy format: correct_answer + distractors
    if "correct_answer" in example and "distractor1" in example:
        correct = example.get("correct_answer", "")
        d1 = example.get("distractor1", "")
        d2 = example.get("distractor2", "")
        d3 = example.get("distractor3", "")
        
        # Convert to list format: correct answer at index 0
        options = [correct, d1, d2, d3]
        return options, 0
    
    # No options found
    raise ValueError("Example does not contain MCQ options in supported formats")


# ACR detection pattern: matches A/B/C/D as standalone choice markers
# Patterns matched:
#   - "Answer: B" or "answer: C"
#   - "The answer is A" or "answer is D"
#   - "Option B" or "option C"
#   - Just "B" or "C" at word boundary near end of text
#   - "(A)" or "(B)" style markers
_ACR_PATTERNS = [
    # "Answer: X" or "answer: X"
    re.compile(r'\banswer\s*[:=]\s*\(?([ABCD])\)?', re.IGNORECASE),
    # "The answer is X"
    re.compile(r'\bthe\s+answer\s+is\s+\(?([ABCD])\)?', re.IGNORECASE),
    # "answer is X"
    re.compile(r'\banswer\s+is\s+\(?([ABCD])\)?', re.IGNORECASE),
    # "Option X" or "option X"
    re.compile(r'\boption\s+\(?([ABCD])\)?', re.IGNORECASE),
    # "I choose X" or "I select X"
    re.compile(r'\b(?:choose|select)\s+\(?([ABCD])\)?', re.IGNORECASE),
    # "(X)" anywhere
    re.compile(r'\(([ABCD])\)', re.IGNORECASE),
    # Standalone letter at end of text (within last 20 chars)
    re.compile(r'\b([ABCD])\b\s*[.!?\s]*$', re.IGNORECASE),
]


def detect_choice_marker(text: str) -> Optional[str]:
    """
    Detect if the model output contains a clear choice marker (A/B/C/D).
    
    This is used for the Answerable Choice Rate (ACR) metric, which measures
    format adherence in MCQ responses.
    
    Args:
        text: The model's output text.
        
    Returns:
        The detected choice letter (A/B/C/D) if found, None otherwise.
    """
    if not text:
        return None
    
    text = text.strip()
    
    # Try each pattern in priority order
    for pattern in _ACR_PATTERNS:
        match = pattern.search(text)
        if match:
            return match.group(1).upper()
    
    return None


class MCQEvaluator:
    """
    Singleton evaluator for MCQ-aware metrics using semantic embeddings.
    
    Reuses the SemanticEvaluator's model to avoid loading embeddings twice.
    
    Usage:
        evaluator = MCQEvaluator.get_instance()
        result = evaluator.evaluate_mcq(prediction, options, gold_option)
    """
    
    _instance = None
    _semantic_model = None
    
    def __init__(self):
        """Private constructor - use get_instance() instead."""
        self._load_model()
    
    @classmethod
    def get_instance(cls) -> "MCQEvaluator":
        """Get singleton instance of MCQEvaluator."""
        if cls._instance is None:
            cls._instance = cls()
        return cls._instance
    
    def _load_model(self):
        """Load the semantic model (reuse from SemanticEvaluator if available)."""
        # Try to reuse existing model from SemanticEvaluator
        try:
            from edge_slm_ace.utils.metrics import SemanticEvaluator
            se = SemanticEvaluator.get_instance()
            # Access the class-level model
            if SemanticEvaluator._model is not None:
                MCQEvaluator._semantic_model = SemanticEvaluator._model
                return
        except Exception:
            pass
        
        # Fallback: load our own model
        try:
            from sentence_transformers import SentenceTransformer
            MCQEvaluator._semantic_model = SentenceTransformer(
                "sentence-transformers/all-MiniLM-L6-v2"
            )
        except ImportError:
            raise ImportError(
                "sentence-transformers is required for MCQEvaluator. "
                "Install with: pip install sentence-transformers"
            )
    
    def compute_similarities(
        self,
        prediction: str,
        option_texts: List[str],
    ) -> np.ndarray:
        """
        Compute cosine similarities between prediction and each option.
        
        Args:
            prediction: The model's prediction text.
            option_texts: List of option texts to compare against.
            
        Returns:
            Array of similarity scores (one per option).
        """
        if MCQEvaluator._semantic_model is None:
            self._load_model()
        
        if not prediction or not option_texts:
            return np.zeros(len(option_texts))
        
        try:
            # Encode all texts at once for efficiency
            all_texts = [prediction] + list(option_texts)
            embeddings = MCQEvaluator._semantic_model.encode(
                all_texts,
                convert_to_numpy=True,
                normalize_embeddings=True,
            )
            
            pred_emb = embeddings[0]
            option_embs = embeddings[1:]
            
            # Compute cosine similarities (already normalized)
            similarities = np.dot(option_embs, pred_emb)
            return np.clip(similarities, 0.0, 1.0)
            
        except Exception as e:
            print(f"[MCQEvaluator] Warning: Similarity computation failed: {e}")
            return np.zeros(len(option_texts))
    
    def evaluate_mcq(
        self,
        prediction: str,
        options: Dict[str, str],
        gold_option: str,
    ) -> Dict[str, any]:
        """
        Evaluate a single MCQ prediction.
        
        Computes:
            - pred_option: The option letter closest to prediction (by semantic similarity)
            - gold_option: The correct option letter
            - oma_correct: 1 if pred_option == gold_option, else 0
            - gom: Gold Option Margin (similarity to gold - avg similarity to others)
            - acr_hit: 1 if prediction contains clear choice marker, else 0
            - detected_marker: The detected choice marker (A/B/C/D) or None
            - similarities: Dict of option letter -> similarity score
        
        Args:
            prediction: The model's prediction text.
            options: Dict mapping option letters to option texts.
                     Example: {"A": "oxygen", "B": "carbon", "C": "nitrogen", "D": "helium"}
            gold_option: The correct option letter (e.g., "A").
            
        Returns:
            Dict with all computed metrics.
        """
        # Ensure options are in standard order
        option_letters = ["A", "B", "C", "D"]
        option_texts = [options.get(letter, "") for letter in option_letters]
        
        # Compute similarities
        sims = self.compute_similarities(prediction, option_texts)
        sim_dict = {letter: float(sims[i]) for i, letter in enumerate(option_letters)}
        
        # Determine predicted option (argmax of similarities)
        pred_idx = int(np.argmax(sims))
        pred_option = option_letters[pred_idx]
        
        # OMA: Option-Mapped Accuracy
        oma_correct = 1 if pred_option == gold_option else 0
        
        # GOM: Gold Option Margin
        gold_idx = option_letters.index(gold_option) if gold_option in option_letters else 0
        s_gold = sims[gold_idx]
        other_sims = [sims[i] for i in range(len(sims)) if i != gold_idx]
        s_others = np.mean(other_sims) if other_sims else 0.0
        gom = float(s_gold - s_others)
        
        # ACR: Answerable Choice Rate
        detected_marker = detect_choice_marker(prediction)
        acr_hit = 1 if detected_marker is not None else 0
        
        return {
            "pred_option": pred_option,
            "gold_option": gold_option,
            "oma_correct": oma_correct,
            "gom": gom,
            "acr_hit": acr_hit,
            "detected_marker": detected_marker,
            "similarities": sim_dict,
        }


def build_prompt_with_choices(
    question: str,
    context: Optional[str] = None,
    options: Optional[List[str]] = None,
) -> str:
    """
    Build a prompt with MCQ choices included.
    
    If options are provided, includes a "Choices (A)-(D)" block and asks
    the model to answer with the exact choice text or letter.
    
    Args:
        question: The question text.
        context: Optional context/support text.
        options: Optional list of 4 option strings.
        
    Returns:
        Formatted prompt string.
    """
    prompt_parts = []
    
    if context:
        prompt_parts.append(f"Context: {context}")
    
    prompt_parts.append(f"Question: {question}")
    
    if options and len(options) == 4:
        prompt_parts.append("")
        prompt_parts.append("Choices:")
        prompt_parts.append(f"(A) {options[0]}")
        prompt_parts.append(f"(B) {options[1]}")
        prompt_parts.append(f"(C) {options[2]}")
        prompt_parts.append(f"(D) {options[3]}")
        prompt_parts.append("")
        prompt_parts.append("Answer with the exact choice text or the letter (A, B, C, or D):")
    else:
        prompt_parts.append("")
        prompt_parts.append("Answer:")
    
    return "\n".join(prompt_parts)


def evaluate_mcq_with_indices(
    prediction: str,
    options: List[str],
    gold_option_idx: int,
    evaluator: Optional["MCQEvaluator"] = None,
) -> Dict[str, any]:
    """
    Evaluate MCQ prediction using option indices (0-3).
    
    Computes:
        - chosen_option_idx: Index (0-3) of option with highest semantic similarity to prediction
        - oma_correct: 1 if chosen_option_idx == gold_option_idx, else 0
        - gom: similarity(pred, gold_option) - mean(similarity(pred, distractors))
    
    Args:
        prediction: The model's prediction text.
        options: List of 4 option strings.
        gold_option_idx: Integer index (0-3) of the correct option.
        evaluator: Optional MCQEvaluator instance. If None, creates one.
        
    Returns:
        Dict with:
            - chosen_option_idx: int (0-3)
            - oma_correct: int (0 or 1)
            - gom: float
    """
    if evaluator is None:
        evaluator = MCQEvaluator.get_instance()
    
    if len(options) != 4:
        raise ValueError(f"Expected 4 options, got {len(options)}")
    
    if not (0 <= gold_option_idx < 4):
        raise ValueError(f"gold_option_idx must be 0-3, got {gold_option_idx}")
    
    # Compute similarities between prediction and each option
    similarities = evaluator.compute_similarities(prediction, options)
    
    # Find option with highest similarity
    chosen_option_idx = int(np.argmax(similarities))
    
    # OMA: Option-Mapped Accuracy
    oma_correct = 1 if chosen_option_idx == gold_option_idx else 0
    
    # GOM: Gold Option Margin
    gold_sim = similarities[gold_option_idx]
    distractor_indices = [i for i in range(4) if i != gold_option_idx]
    distractor_sims = [similarities[i] for i in distractor_indices]
    mean_distractor_sim = np.mean(distractor_sims) if distractor_sims else 0.0
    gom = float(gold_sim - mean_distractor_sim)
    
    return {
        "chosen_option_idx": chosen_option_idx,
        "oma_correct": oma_correct,
        "gom": gom,
    }


def compute_mcq_aggregate_metrics(
    results: List[Dict],
) -> Dict[str, float]:
    """
    Compute aggregate MCQ metrics from per-example results.
    
    Args:
        results: List of result dicts, each containing:
            - oma_correct (int): 1 if correct, 0 otherwise
            - gom (float): Gold Option Margin
            - acr_hit (int): 1 if choice marker detected, 0 otherwise
            
    Returns:
        Dict with aggregate metrics:
            - oma_accuracy: Mean of oma_correct
            - avg_gom: Mean of gom
            - acr_rate: Mean of acr_hit
    """
    if not results:
        return {
            "oma_accuracy": None,
            "avg_gom": None,
            "acr_rate": None,
        }
    
    # Filter to results that have MCQ metrics
    mcq_results = [r for r in results if "oma_correct" in r and r["oma_correct"] is not None]
    
    if not mcq_results:
        return {
            "oma_accuracy": None,
            "avg_gom": None,
            "acr_rate": None,
        }
    
    oma_vals = [r["oma_correct"] for r in mcq_results]
    gom_vals = [r["gom"] for r in mcq_results if "gom" in r and r["gom"] is not None]
    acr_vals = [r["acr_hit"] for r in mcq_results if "acr_hit" in r and r["acr_hit"] is not None]
    
    return {
        "oma_accuracy": sum(oma_vals) / len(oma_vals) if oma_vals else None,
        "avg_gom": sum(gom_vals) / len(gom_vals) if gom_vals else None,
        "acr_rate": sum(acr_vals) / len(acr_vals) if acr_vals else None,
    }
