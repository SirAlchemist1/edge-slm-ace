"""Tests for model manager functionality."""

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from edge_slm_ace.models.model_manager import load_model_and_tokenizer, generate
from edge_slm_ace.utils.device_utils import get_device


def test_load_model_and_tokenizer():
    """Test loading a tiny model."""
    # Use tiny-gpt2 for fast testing
    model_id = "sshleifer/tiny-gpt2"
    
    device = get_device()
    model, tokenizer = load_model_and_tokenizer(model_id, device=device)
    
    assert model is not None
    assert tokenizer is not None
    assert tokenizer.pad_token is not None  # Should be set
    
    # Check model is on correct device
    # Note: tiny-gpt2 is always forced to CPU on Torch >= 2.6 due to security restrictions
    model_device = next(model.parameters()).device
    expected_device = "cpu"  # tiny-gpt2 is forced to CPU
    assert model_device.type == expected_device


def test_generate():
    """Test text generation."""
    model_id = "sshleifer/tiny-gpt2"
    
    device = get_device()
    model, tokenizer = load_model_and_tokenizer(model_id, device=device)
    
    prompt = "Hello"
    output = generate(
        model,
        tokenizer,
        prompt,
        max_new_tokens=10,
        temperature=0.7,
        top_p=0.95,
    )
    
    assert isinstance(output, str)
    assert len(output) > 0  # Should produce some output

