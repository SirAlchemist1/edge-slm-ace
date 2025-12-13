"""Tests for MCQ-aware evaluation metrics (OMA/GOM/ACR).

These tests verify the SciQ-specific evaluation metrics work correctly
without requiring network access (embeddings are mocked).
"""

import pytest
import numpy as np
from unittest.mock import MagicMock, patch


class TestDetectChoiceMarker:
    """Tests for the ACR choice marker detection."""
    
    def test_detect_answer_colon_format(self):
        """Test detection of 'Answer: X' format."""
        from edge_slm_ace.utils.mcq_eval import detect_choice_marker
        
        assert detect_choice_marker("Answer: B") == "B"
        assert detect_choice_marker("answer: C") == "C"
        assert detect_choice_marker("The answer: A") == "A"
        assert detect_choice_marker("answer:D") == "D"
    
    def test_detect_answer_is_format(self):
        """Test detection of 'The answer is X' format."""
        from edge_slm_ace.utils.mcq_eval import detect_choice_marker
        
        assert detect_choice_marker("The answer is B") == "B"
        assert detect_choice_marker("The answer is C because...") == "C"
        assert detect_choice_marker("I think the answer is A") == "A"
    
    def test_detect_option_format(self):
        """Test detection of 'Option X' format."""
        from edge_slm_ace.utils.mcq_eval import detect_choice_marker
        
        assert detect_choice_marker("Option B is correct") == "B"
        assert detect_choice_marker("I choose option C") == "C"
    
    def test_detect_parenthesis_format(self):
        """Test detection of '(X)' format."""
        from edge_slm_ace.utils.mcq_eval import detect_choice_marker
        
        assert detect_choice_marker("The correct choice is (B)") == "B"
        assert detect_choice_marker("(A) is the answer") == "A"
    
    def test_detect_standalone_at_end(self):
        """Test detection of standalone letter at end."""
        from edge_slm_ace.utils.mcq_eval import detect_choice_marker
        
        assert detect_choice_marker("Based on the context, B") == "B"
        assert detect_choice_marker("The solution is D.") == "D"
    
    def test_no_detection_for_unrelated_text(self):
        """Test that unrelated text doesn't trigger false positives."""
        from edge_slm_ace.utils.mcq_eval import detect_choice_marker
        
        # Should not detect choice markers in regular text
        assert detect_choice_marker("This is about the ABC company") is None
        assert detect_choice_marker("The quick brown fox") is None
        assert detect_choice_marker("Carbon dioxide is produced") is None
        assert detect_choice_marker("") is None
        assert detect_choice_marker("123") is None
    
    def test_case_insensitivity(self):
        """Test that detection is case-insensitive."""
        from edge_slm_ace.utils.mcq_eval import detect_choice_marker
        
        assert detect_choice_marker("ANSWER: b") == "B"
        assert detect_choice_marker("The ANSWER IS c") == "C"


class TestIsSciQTask:
    """Tests for SciQ task detection."""
    
    def test_sciq_task_names(self):
        """Test that SciQ task names are correctly identified."""
        from edge_slm_ace.utils.mcq_eval import is_sciq_task
        
        assert is_sciq_task("sciq_tiny") is True
        assert is_sciq_task("sciq_test") is True
        assert is_sciq_task("SciQ_Train") is True
        assert is_sciq_task("my_sciq_dataset") is True
    
    def test_non_sciq_task_names(self):
        """Test that non-SciQ task names are correctly rejected."""
        from edge_slm_ace.utils.mcq_eval import is_sciq_task
        
        assert is_sciq_task("medqa_tiny") is False
        assert is_sciq_task("tatqa_tiny") is False
        assert is_sciq_task("iot_tiny") is False
        assert is_sciq_task(None) is False


class TestHasMCQOptions:
    """Tests for MCQ options detection."""
    
    def test_sciq_format_example(self):
        """Test that SciQ format examples are correctly identified."""
        from edge_slm_ace.utils.mcq_eval import has_mcq_options
        
        example = {
            "question": "What is H2O?",
            "correct_answer": "water",
            "distractor1": "oxygen",
            "distractor2": "hydrogen",
            "distractor3": "carbon",
            "support": "H2O is the chemical formula for water.",
        }
        assert has_mcq_options(example) is True
    
    def test_non_mcq_example(self):
        """Test that non-MCQ examples are correctly rejected."""
        from edge_slm_ace.utils.mcq_eval import has_mcq_options
        
        example = {
            "question": "What is 2+2?",
            "answer": "4",
        }
        assert has_mcq_options(example) is False


class TestExtractMCQOptions:
    """Tests for MCQ option extraction."""
    
    def test_option_extraction(self):
        """Test that options are correctly extracted and labeled."""
        from edge_slm_ace.utils.mcq_eval import extract_mcq_options
        
        example = {
            "correct_answer": "water",
            "distractor1": "oxygen",
            "distractor2": "hydrogen",
            "distractor3": "carbon",
        }
        
        options, gold_option, gold_text = extract_mcq_options(example)
        
        # Correct answer should be at position A
        assert gold_option == "A"
        assert gold_text == "water"
        assert options["A"] == "water"
        assert options["B"] == "oxygen"
        assert options["C"] == "hydrogen"
        assert options["D"] == "carbon"


class TestMCQEvaluator:
    """Tests for the MCQ evaluator with mocked embeddings."""
    
    def test_evaluate_mcq_correct_prediction(self):
        """Test MCQ evaluation when prediction matches option C semantically."""
        from edge_slm_ace.utils.mcq_eval import MCQEvaluator
        
        # Reset singleton for testing
        MCQEvaluator._instance = None
        MCQEvaluator._semantic_model = None
        
        # Create evaluator with mocked _load_model
        with patch.object(MCQEvaluator, '_load_model'):
            evaluator = MCQEvaluator.get_instance()
        
        # Mock compute_similarities to return predetermined values
        # Similarities for options [A, B, C, D] - C should be highest
        def mock_compute_sims(prediction, option_texts):
            return np.array([0.2, 0.3, 0.9, 0.1])  # C (index 2) has highest sim
        
        evaluator.compute_similarities = MagicMock(side_effect=mock_compute_sims)
        
        options = {
            "A": "oxygen",
            "B": "carbon dioxide",
            "C": "nitrogen",
            "D": "helium",
        }
        
        result = evaluator.evaluate_mcq(
            prediction="The gas is nitrogen",
            options=options,
            gold_option="C",
        )
        
        # Prediction should map to C (highest similarity)
        assert result["pred_option"] == "C"
        assert result["gold_option"] == "C"
        assert result["oma_correct"] == 1
        assert result["gom"] > 0  # Positive margin since pred matched gold
        
        # Clean up singleton
        MCQEvaluator._instance = None
        MCQEvaluator._semantic_model = None
    
    def test_evaluate_mcq_incorrect_prediction(self):
        """Test MCQ evaluation when prediction matches wrong option."""
        from edge_slm_ace.utils.mcq_eval import MCQEvaluator
        
        # Reset singleton for testing
        MCQEvaluator._instance = None
        MCQEvaluator._semantic_model = None
        
        # Create evaluator with mocked _load_model
        with patch.object(MCQEvaluator, '_load_model'):
            evaluator = MCQEvaluator.get_instance()
        
        # Mock compute_similarities - B should be highest but gold is A
        def mock_compute_sims(prediction, option_texts):
            return np.array([0.3, 0.95, 0.2, 0.1])  # B (index 1) has highest sim
        
        evaluator.compute_similarities = MagicMock(side_effect=mock_compute_sims)
        
        options = {
            "A": "water",
            "B": "ice",
            "C": "steam",
            "D": "vapor",
        }
        
        result = evaluator.evaluate_mcq(
            prediction="I think it's frozen water",
            options=options,
            gold_option="A",  # Gold is A, but prediction maps to B
        )
        
        assert result["pred_option"] == "B"
        assert result["gold_option"] == "A"
        assert result["oma_correct"] == 0
        # GOM should be negative since gold wasn't selected
        
        # Clean up singleton
        MCQEvaluator._instance = None
        MCQEvaluator._semantic_model = None
    
    def test_acr_detection_in_evaluate(self):
        """Test that ACR is correctly detected during evaluation."""
        from edge_slm_ace.utils.mcq_eval import MCQEvaluator
        
        # Reset singleton for testing
        MCQEvaluator._instance = None
        MCQEvaluator._semantic_model = None
        
        # Create evaluator with mocked _load_model
        with patch.object(MCQEvaluator, '_load_model'):
            evaluator = MCQEvaluator.get_instance()
        
        # Mock compute_similarities with uniform values
        def mock_compute_sims(prediction, option_texts):
            return np.array([0.25, 0.25, 0.25, 0.25])
        
        evaluator.compute_similarities = MagicMock(side_effect=mock_compute_sims)
        
        options = {"A": "a", "B": "b", "C": "c", "D": "d"}
        
        # With choice marker
        result = evaluator.evaluate_mcq(
            prediction="The answer is B",
            options=options,
            gold_option="A",
        )
        assert result["acr_hit"] == 1
        assert result["detected_marker"] == "B"
        
        # Without choice marker
        result = evaluator.evaluate_mcq(
            prediction="I think the solution involves carbon",
            options=options,
            gold_option="A",
        )
        assert result["acr_hit"] == 0
        assert result["detected_marker"] is None
        
        # Clean up singleton
        MCQEvaluator._instance = None
        MCQEvaluator._semantic_model = None


class TestComputeMCQAggregateMetrics:
    """Tests for aggregate MCQ metrics computation."""
    
    def test_aggregate_with_valid_results(self):
        """Test aggregation with valid MCQ results."""
        from edge_slm_ace.utils.mcq_eval import compute_mcq_aggregate_metrics
        
        results = [
            {"oma_correct": 1, "gom": 0.5, "acr_hit": 1},
            {"oma_correct": 0, "gom": -0.2, "acr_hit": 1},
            {"oma_correct": 1, "gom": 0.3, "acr_hit": 0},
            {"oma_correct": 1, "gom": 0.4, "acr_hit": 1},
        ]
        
        agg = compute_mcq_aggregate_metrics(results)
        
        assert agg["oma_accuracy"] == 0.75  # 3/4
        assert abs(agg["avg_gom"] - 0.25) < 0.01  # (0.5 - 0.2 + 0.3 + 0.4) / 4
        assert agg["acr_rate"] == 0.75  # 3/4
    
    def test_aggregate_with_empty_results(self):
        """Test aggregation with empty results."""
        from edge_slm_ace.utils.mcq_eval import compute_mcq_aggregate_metrics
        
        agg = compute_mcq_aggregate_metrics([])
        
        assert agg["oma_accuracy"] is None
        assert agg["avg_gom"] is None
        assert agg["acr_rate"] is None
    
    def test_aggregate_with_mixed_results(self):
        """Test aggregation with mixed (some non-MCQ) results."""
        from edge_slm_ace.utils.mcq_eval import compute_mcq_aggregate_metrics
        
        results = [
            {"oma_correct": 1, "gom": 0.5, "acr_hit": 1},
            {"other_field": "value"},  # Non-MCQ result
            {"oma_correct": None, "gom": None, "acr_hit": None},  # Null MCQ result
            {"oma_correct": 0, "gom": -0.1, "acr_hit": 0},
        ]
        
        agg = compute_mcq_aggregate_metrics(results)
        
        # Should only consider valid MCQ results
        assert agg["oma_accuracy"] == 0.5  # 1/2
        assert agg["avg_gom"] == 0.2  # (0.5 - 0.1) / 2
        assert agg["acr_rate"] == 0.5  # 1/2
