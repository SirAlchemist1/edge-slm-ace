"""Configuration utilities for models and device settings."""

from dataclasses import dataclass
from typing import Dict, Optional

import torch


# ACE mode constants
ACE_MODE_FULL = "ace_full"
ACE_MODE_WORKING = "ace_working_memory"


@dataclass
class ModelConfig:
    """Configuration for a language model."""
    model_id: str
    max_new_tokens: int = 256
    temperature: float = 0.7
    top_p: float = 0.95
    
    def __post_init__(self):
        """Validate configuration values."""
        assert 0.0 <= self.temperature <= 2.0, "Temperature must be in [0, 2]"
        assert 0.0 <= self.top_p <= 1.0, "top_p must be in [0, 1]"
        assert self.max_new_tokens > 0, "max_new_tokens must be positive"


# Default model configurations mapped to HuggingFace model IDs
MODEL_CONFIGS: Dict[str, ModelConfig] = {
    "phi3-mini": ModelConfig(
        model_id="microsoft/Phi-3-mini-4k-instruct",
        max_new_tokens=256,
        temperature=0.0,  # Greedy decoding for reproducible evaluation
        top_p=1.0,
    ),
    "llama-3.2-1b": ModelConfig(
        model_id="meta-llama/Llama-3.2-1B-Instruct",
        max_new_tokens=256,
        temperature=0.7,
        top_p=0.95,
    ),
    "mistral-7b": ModelConfig(
        model_id="mistralai/Mistral-7B-Instruct-v0.3",
        max_new_tokens=512,
        temperature=0.7,
        top_p=0.95,
    ),
    "llama-3-8b": ModelConfig(
        model_id="meta-llama/Meta-Llama-3-8B-Instruct",
        max_new_tokens=512,
        temperature=0.7,
        top_p=0.95,
    ),
    # Tiny model for testing
    "tiny-gpt2": ModelConfig(
        model_id="sshleifer/tiny-gpt2",
        max_new_tokens=50,
        temperature=0.7,
        top_p=0.95,
    ),
    # Pre-fine-tuned small model for comparison (fine-tuned upper bound)
    # This represents a small model that has been fine-tuned on domain-specific QA data
    # For medical QA, we use a small model fine-tuned on medical question answering
    # Note: This is our "fine-tuned small upper bound" for comparison with ACE methods
    "medqa_finetuned_small": ModelConfig(
        model_id="microsoft/DialoGPT-small",  # Small conversational model as placeholder for fine-tuned QA model
        max_new_tokens=256,
        temperature=0.3,  # Lower temperature for more focused answers
        top_p=0.9,
    ),
    # Qwen models - use greedy decoding to avoid CUDA numerical instability
    # These are parameter-matched rivals to TinyLlama (1.1B), Phi-3 (3.8B), and Mistral (7B)
    "qwen-2.5-1.5b": ModelConfig(
        model_id="Qwen/Qwen2.5-1.5B-Instruct",
        max_new_tokens=256,
        temperature=0.0,  # Greedy decoding for reproducibility
        top_p=1.0,
    ),
    "qwen-2.5-3b": ModelConfig(
        model_id="Qwen/Qwen2.5-3B-Instruct",
        max_new_tokens=256,
        temperature=0.0,  # Greedy decoding for reproducibility
        top_p=1.0,
    ),
    "qwen-2.5-7b": ModelConfig(
        model_id="Qwen/Qwen2.5-7B-Instruct",
        max_new_tokens=256,
        temperature=0.0,  # Greedy decoding for reproducibility
        top_p=1.0,
    ),
    # TinyLlama - match settings for fair comparison
    "tinyllama-1.1b": ModelConfig(
        model_id="TinyLlama/TinyLlama-1.1B-Chat-v1.0",
        max_new_tokens=256,
        temperature=0.0,  # Greedy decoding for reproducibility
        top_p=1.0,
    ),
}


def get_model_config(model_id_or_key: str) -> ModelConfig:
    """
    Get a model configuration by key or model ID.

    Args:
        model_id_or_key: Either a key from MODEL_CONFIGS or a HuggingFace model ID.

    Returns:
        ModelConfig: The configuration for the model.
    """
    if model_id_or_key in MODEL_CONFIGS:
        # Return a copy so modifications don't affect the default
        config = MODEL_CONFIGS[model_id_or_key]
        return ModelConfig(
            model_id=config.model_id,
            max_new_tokens=config.max_new_tokens,
            temperature=config.temperature,
            top_p=config.top_p,
        )
    else:
        # Check if it matches any model_id in the configs
        for key, config in MODEL_CONFIGS.items():
            if config.model_id == model_id_or_key:
                # Found a match, return a copy
                return ModelConfig(
                    model_id=config.model_id,
                    max_new_tokens=config.max_new_tokens,
                    temperature=config.temperature,
                    top_p=config.top_p,
                )

        # Not found anywhere - create default config with the provided model ID
        # Use temperature=0.0 for greedy decoding to avoid CUDA numerical issues
        return ModelConfig(model_id=model_id_or_key, temperature=0.0, top_p=1.0)


# Task registry: maps task names to dataset paths and domains
# Available tasks:
#   - tatqa_tiny: Finance QA (3 examples)
#   - medqa_tiny: Medical QA (3 examples)
#   - iot_tiny: IoT/Anomaly Detection (5 examples)
#   - sciq_tiny: Science MCQ (5 examples) - for MCQ-aware evaluation
#   - sciq_test: Science MCQ (5 examples) - alias for testing
#   - medqa_train: Medical QA training set (full MedQA dataset)
#   - math_train: Math word problems training set
#   - sciq_train: Science QA training set (11,679 examples)
#   - sciq_val: Science QA validation set (1,000 examples)
TASK_CONFIGS: Dict[str, Dict[str, str]] = {
    # Tiny datasets for smoke tests
    "tatqa_tiny": {
        "path": "data/tasks/tatqa_tiny.json",
        "domain": "finance",
    },
    "medqa_tiny": {
        "path": "data/tasks/medqa_tiny.json",
        "domain": "medical",
    },
    "iot_tiny": {
        "path": "data/tasks/iot_tiny.json",
        "domain": "iot",
    },
    "sciq_tiny": {
        "path": "data/tasks/sciq_tiny.json",
        "domain": "science",
    },
    # HPC full datasets
    "medqa_train": {
        "path": "data/tasks/train_med.jsonl",
        "domain": "medical",
    },
    "math_train": {
        "path": "data/tasks/test_math.jsonl",
        "domain": "math",
    },
    "sciq_train": {
        "path": "data/tasks/sciq_train.jsonl",
        "domain": "science",
    },
    "sciq_val": {
        "path": "data/tasks/sciq_val.json",
        "domain": "science",
    },
    "sciq_test": {
        "path": "data/tasks/sciq_test.json",
        "domain": "science",
    },
    "sciq_mcq_test": {
        "path": "data/tasks/sciq_mcq_test.jsonl",
        "domain": "science",
    },
}


def get_task_config(task_name: str) -> Dict[str, str]:
    """
    Get task configuration by name.
    
    Args:
        task_name: Task name from TASK_CONFIGS.
        
    Returns:
        Dict with 'path' and 'domain' keys.
        
    Raises:
        KeyError: If task_name is not found.
    """
    if task_name not in TASK_CONFIGS:
        raise KeyError(
            f"Task '{task_name}' not found. Available tasks: {list(TASK_CONFIGS.keys())}"
        )
    return TASK_CONFIGS[task_name]

