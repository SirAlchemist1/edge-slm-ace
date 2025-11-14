"""Model loading and generation utilities."""

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from transformers.generation.utils import DynamicCache
if not hasattr(DynamicCache, "seen_tokens"):
    @property
    def seen_tokens(self):
        return 0
    DynamicCache.seen_tokens = seen_tokens
from typing import Optional


def load_model_and_tokenizer(
    model_id: str,
    device: Optional[torch.device] = None,
) -> tuple[AutoModelForCausalLM, AutoTokenizer]:
    """
    Load a HuggingFace model and tokenizer for inference.
    
    Args:
        model_id: HuggingFace model identifier (e.g., "microsoft/Phi-3-mini-4k-instruct").
        device: Target device. If None, auto-detects based on availability.
        
    Returns:
        Tuple of (model, tokenizer).
    """
    if device is None:
        from slm_ace.utils import get_device
        device = get_device()
    
    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
    
    # Set pad_token if not present (some models don't have one)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    # Load model
    # Use device_map="auto" for CUDA multi-GPU setups
    # For single device (CPU/MPS), move explicitly
    if device.type == "cuda":
        model = AutoModelForCausalLM.from_pretrained(
            model_id,
            torch_dtype=torch.float16,  # Use half precision for GPU efficiency
            device_map="auto",
            trust_remote_code=True,
        )
    else:
        # For CPU or MPS, load in float32 and move to device
        # Note: Some tiny models may not support MPS well; fallback to CPU if needed
        try:
            model = AutoModelForCausalLM.from_pretrained(
                model_id,
                torch_dtype=torch.float32,
                trust_remote_code=True,
            )
            model = model.to(device)
            # Test MPS compatibility with a dummy forward pass
            if device.type == "mps":
                try:
                    dummy_input = tokenizer("test", return_tensors="pt")
                    dummy_input = {k: v.to(device) for k, v in dummy_input.items()}
                    with torch.no_grad():
                        _ = model(**dummy_input)
                except Exception:
                    # MPS not working, fallback to CPU
                    print(f"Warning: MPS device not compatible with {model_id}, falling back to CPU")
                    device = torch.device("cpu")
                    model = model.to(device)
        except Exception as e:
            raise RuntimeError(f"Failed to load model {model_id}: {e}") from e
    
    model.eval()  # Set to evaluation mode
    
    return model, tokenizer


def generate(
    model: AutoModelForCausalLM,
    tokenizer: AutoTokenizer,
    prompt: str,
    max_new_tokens: int = 256,
    temperature: float = 0.7,
    top_p: float = 0.95,
) -> str:
    """
    Generate text from a prompt using the model.
    
    Args:
        model: The loaded language model.
        tokenizer: The tokenizer for the model.
        prompt: Input text prompt.
        max_new_tokens: Maximum number of tokens to generate.
        temperature: Sampling temperature (higher = more random).
        top_p: Nucleus sampling parameter.
        
    Returns:
        Generated text (decoded and stripped of special tokens).
    """
    # Tokenize input
    try:
        inputs = tokenizer(prompt, return_tensors="pt", padding=True, truncation=True)
    except Exception as e:
        raise ValueError(f"Failed to tokenize prompt: {e}") from e
    
    # Move inputs to the same device as model
    device = next(model.parameters()).device
    inputs = {k: v.to(device) for k, v in inputs.items()}
    
    # Generate
    try:
        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                temperature=temperature,
                top_p=top_p,
                do_sample=temperature > 0.0,
                pad_token_id=tokenizer.pad_token_id,
                eos_token_id=tokenizer.eos_token_id,
            )
    except Exception as e:
        raise RuntimeError(f"Generation failed: {e}") from e
    
    # Decode output (skip the input tokens)
    try:
        generated_ids = outputs[0][inputs["input_ids"].shape[1]:]
        generated_text = tokenizer.decode(generated_ids, skip_special_tokens=True)
    except Exception as e:
        raise ValueError(f"Failed to decode output: {e}") from e
    
    return generated_text.strip()

