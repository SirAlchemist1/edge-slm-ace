"""Model loading and generation utilities."""

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from typing import Optional


# Models that require trust_remote_code=True (add as needed)
_MODELS_REQUIRING_TRUST_REMOTE_CODE = {
    # Add model IDs here if they truly require trust_remote_code
    # Example: "some/custom-model-that-needs-it"
}


def load_model_and_tokenizer(
    model_id: str,
    device: Optional[torch.device] = None,
    device_override: Optional[str] = None,
) -> tuple[AutoModelForCausalLM, AutoTokenizer]:
    """
    Load a HuggingFace model and tokenizer for inference.
    
    Args:
        model_id: HuggingFace model identifier (e.g., "microsoft/Phi-3-mini-4k-instruct").
        device: Target device (deprecated, use device_override instead). If None, uses device_override or auto-detects.
        device_override: Optional device override string ("cuda", "mps", "cpu", or None for auto-detect).
        
    Returns:
        Tuple of (model, tokenizer).
    """
    # Resolve device: prefer device_override, then device, then auto-detect
    # resolve_device_override will handle tiny-gpt2 forcing automatically
    is_tiny_gpt2 = "tiny-gpt2" in model_id.lower()
    
    from slm_ace.utils import resolve_device_override, get_device
    if device_override is not None:
        device, forced = resolve_device_override(device_override, model_id=model_id)
    elif device is not None:
        # Device was explicitly provided as torch.device, but we still need to check for tiny-gpt2 forcing
        # Convert to string for resolve_device_override, or just check directly
        device, forced = resolve_device_override(None, model_id=model_id)
    else:
        # Auto-detect
        device, forced = resolve_device_override(None, model_id=model_id)
    
    # Print warning if tiny-gpt2 was forced to CPU
    if forced == "forced" and device.type == "cpu":
        print("[model_manager] tiny-gpt2 detected → forcing CPU (CUDA load blocked by torch >=2.6)")
    
    # Determine if trust_remote_code is needed
    # tiny-gpt2: always use safetensors=True and trust_remote_code=False for safety
    # Other models: try False first (safer), only use True if explicitly required or if loading fails
    if is_tiny_gpt2:
        trust_remote_code = False
        use_safetensors = True
    else:
        trust_remote_code = model_id in _MODELS_REQUIRING_TRUST_REMOTE_CODE
        use_safetensors = True  # Prefer safetensors for all models
    
    # Load tokenizer
    # Standard HF models don't need trust_remote_code, but some custom models do
    # For tiny-gpt2, always use trust_remote_code=False
    try:
        tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=trust_remote_code)
    except Exception:
        # If loading fails and not tiny-gpt2, try with trust_remote_code=True (some models require it)
        if not is_tiny_gpt2 and not trust_remote_code:
            print(f"Warning: Failed to load tokenizer for {model_id} without trust_remote_code, retrying with trust_remote_code=True")
            trust_remote_code = True
            tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
        else:
            raise
    
    # Set pad_token if not present (some models don't have one)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    # Load model
    # Use device_map="auto" for CUDA multi-GPU setups
    # For single device (CPU/MPS), move explicitly
    dtype = torch.float16 if device.type == "cuda" else torch.float32
    
    if device.type == "cuda":
        try:
            # Try with safetensors=True first (safer)
            model = AutoModelForCausalLM.from_pretrained(
                model_id,
                torch_dtype=dtype,
                device_map="auto",
                trust_remote_code=trust_remote_code,
                use_safetensors=use_safetensors,
            )
        except Exception as e:
            # If loading fails and not tiny-gpt2, try without safetensors or with trust_remote_code=True
            if not is_tiny_gpt2:
                if use_safetensors:
                    try:
                        print(f"Warning: Failed to load {model_id} with safetensors=True, retrying with safetensors=False")
                        model = AutoModelForCausalLM.from_pretrained(
                            model_id,
                            torch_dtype=dtype,
                            device_map="auto",
                            trust_remote_code=trust_remote_code,
                            use_safetensors=False,
                        )
                    except Exception:
                        if not trust_remote_code:
                            print(f"Warning: Failed to load model {model_id} without trust_remote_code, retrying with trust_remote_code=True")
                            model = AutoModelForCausalLM.from_pretrained(
                                model_id,
                                torch_dtype=dtype,
                                device_map="auto",
                                trust_remote_code=True,
                                use_safetensors=False,
                            )
                        else:
                            raise
                elif not trust_remote_code:
                    print(f"Warning: Failed to load model {model_id} without trust_remote_code, retrying with trust_remote_code=True")
                    model = AutoModelForCausalLM.from_pretrained(
                        model_id,
                        torch_dtype=dtype,
                        device_map="auto",
                        trust_remote_code=True,
                    )
                else:
                    raise
            else:
                # tiny-gpt2: try alternative loading strategies
                error_msg = str(e)
                if "torch.load" in error_msg or "CVE-2025-32434" in error_msg:
                    # Model uses old pickled weights - try with explicit safetensors=False and see if we can work around it
                    # Note: This may still fail on Torch >= 2.6, but we'll try
                    try:
                        print(f"[tiny-gpt2] Attempting to load with legacy weights (may fail on Torch >= 2.6)...")
                        model = AutoModelForCausalLM.from_pretrained(
                            model_id,
                            torch_dtype=dtype,
                            device_map="auto",
                            trust_remote_code=False,
                            use_safetensors=False,
                        )
                    except Exception as e2:
                        # Final fallback: provide helpful error message
                        torch_version = torch.__version__
                        error_msg = (
                            f"Failed to load tiny-gpt2 model {model_id}.\n"
                            f"Error: {e2}\n\n"
                            f"This model uses old pickled weights that cannot be loaded with Torch >= 2.6 due to security restrictions (CVE-2025-32434).\n"
                            f"Current Torch version: {torch_version}\n\n"
                            f"Solutions:\n"
                            f"  1. Use a different model for smoke tests:\n"
                            f"     python -m scripts.smoke_gpu_phi3 --task-name tatqa_tiny --device cuda --limit 2\n"
                            f"  2. For CPU smoke tests, ensure you're using Torch < 2.6 or a CPU-only build\n"
                            f"  3. Use phi3-mini or llama-3.2-1b for GPU tests instead"
                        )
                        raise RuntimeError(error_msg) from e2
                else:
                    raise RuntimeError(f"Failed to load tiny-gpt2 model {model_id}: {e}") from e
    else:
        # For CPU or MPS, load in float32 and move to device
        # Note: Some tiny models may not support MPS well; fallback to CPU if needed
        try:
            try:
                # Try with safetensors=True first (safer)
                model = AutoModelForCausalLM.from_pretrained(
                    model_id,
                    torch_dtype=dtype,
                    trust_remote_code=trust_remote_code,
                    use_safetensors=use_safetensors,
                )
            except Exception as e:
                # If loading fails and not tiny-gpt2, try without safetensors or with trust_remote_code=True
                if not is_tiny_gpt2:
                    if use_safetensors:
                        try:
                            print(f"Warning: Failed to load {model_id} with safetensors=True, retrying with safetensors=False")
                            model = AutoModelForCausalLM.from_pretrained(
                                model_id,
                                torch_dtype=dtype,
                                trust_remote_code=trust_remote_code,
                                use_safetensors=False,
                            )
                        except Exception:
                            if not trust_remote_code:
                                print(f"Warning: Failed to load model {model_id} without trust_remote_code, retrying with trust_remote_code=True")
                                model = AutoModelForCausalLM.from_pretrained(
                                    model_id,
                                    torch_dtype=dtype,
                                    trust_remote_code=True,
                                    use_safetensors=False,
                                )
                            else:
                                raise
                    elif not trust_remote_code:
                        print(f"Warning: Failed to load model {model_id} without trust_remote_code, retrying with trust_remote_code=True")
                        model = AutoModelForCausalLM.from_pretrained(
                            model_id,
                            torch_dtype=dtype,
                            trust_remote_code=True,
                        )
                    else:
                        raise
                else:
                    # tiny-gpt2: try alternative loading strategies
                    error_msg = str(e)
                    if "torch.load" in error_msg or "CVE-2025-32434" in error_msg:
                        # Model uses old pickled weights - try with explicit safetensors=False
                        try:
                            print(f"[tiny-gpt2] Attempting to load with legacy weights (may fail on Torch >= 2.6)...")
                            model = AutoModelForCausalLM.from_pretrained(
                                model_id,
                                torch_dtype=dtype,
                                trust_remote_code=False,
                                use_safetensors=False,
                            )
                        except Exception as e2:
                            # Final fallback: provide helpful error message
                            torch_version = torch.__version__
                            error_msg = (
                                f"Failed to load tiny-gpt2 model {model_id}.\n"
                                f"Error: {e2}\n\n"
                                f"This model uses old pickled weights that cannot be loaded with Torch >= 2.6 due to security restrictions (CVE-2025-32434).\n"
                                f"Current Torch version: {torch_version}\n\n"
                                f"Solutions:\n"
                                f"  1. Use a different model for smoke tests:\n"
                                f"     python -m scripts.smoke_gpu_phi3 --task-name tatqa_tiny --device cuda --limit 2\n"
                                f"  2. For CPU smoke tests, ensure you're using Torch < 2.6 or a CPU-only build\n"
                                f"  3. Use phi3-mini or llama-3.2-1b for GPU tests instead"
                            )
                            raise RuntimeError(error_msg) from e2
                    else:
                        raise RuntimeError(f"Failed to load tiny-gpt2 model {model_id}: {e}") from e
            
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
    
    print(f"[model_manager] Loaded {model_id} on device={device} (dtype={dtype})")
    
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


def count_tokens(tokenizer: AutoTokenizer, text: str) -> int:
    """
    Count the number of tokens in a text string.
    
    Args:
        tokenizer: The tokenizer to use.
        text: Text to count tokens for.
        
    Returns:
        Number of tokens.
    """
    try:
        tokens = tokenizer.encode(text, add_special_tokens=False)
        return len(tokens)
    except Exception:
        # Fallback: approximate via word count
        return int(len(text.split()) * 1.3)

