"""Utility functions: metrics, device detection, configuration."""

from edge_slm_ace.utils.metrics import (
    compute_accuracy,
    compute_average_latency,
    semantic_answer_score,
    compute_bleu_score,
    compute_semantic_accuracy,
)
from edge_slm_ace.utils.device_utils import get_device, resolve_device_override
from edge_slm_ace.utils.config import (
    ModelConfig,
    get_model_config,
    get_task_config,
    TASK_CONFIGS,
    MODEL_CONFIGS,
    ACE_MODE_FULL,
    ACE_MODE_WORKING,
)
from edge_slm_ace.utils.mcq_eval import (
    is_sciq_task,
    has_mcq_options,
    extract_mcq_options,
    detect_choice_marker,
    MCQEvaluator,
    compute_mcq_aggregate_metrics,
)

__all__ = [
    "compute_accuracy",
    "compute_average_latency",
    "semantic_answer_score",
    "compute_bleu_score",
    "compute_semantic_accuracy",
    "get_device",
    "resolve_device_override",
    "ModelConfig",
    "get_model_config",
    "get_task_config",
    "TASK_CONFIGS",
    "MODEL_CONFIGS",
    "ACE_MODE_FULL",
    "ACE_MODE_WORKING",
    # MCQ evaluation
    "is_sciq_task",
    "has_mcq_options",
    "extract_mcq_options",
    "detect_choice_marker",
    "MCQEvaluator",
    "compute_mcq_aggregate_metrics",
]
