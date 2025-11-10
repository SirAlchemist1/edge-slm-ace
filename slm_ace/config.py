"""Configuration utilities for models and device settings."""

from dataclasses import dataclass
from typing import Dict, Optional

import torch


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
        temperature=0.7,
        top_p=0.95,
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
        # Assume it's a direct model ID
        return ModelConfig(model_id=model_id_or_key)


# Task registry: maps task names to dataset paths and domains
TASK_CONFIGS: Dict[str, Dict[str, str]] = {
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

