"""Tests for model manager functionality."""

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from slm_ace.model_manager import load_model_and_tokenizer, generate
from slm_ace.utils import get_device


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
    model_device = next(model.parameters()).device
    assert model_device.type == device.type


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

