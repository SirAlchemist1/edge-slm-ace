"""Utility functions for device detection and timing."""

import platform
import time
from typing import Optional

import torch


def get_device() -> torch.device:
    """
    Detect the best available device for model inference.
    
    Priority:
    - macOS: mps (if available) > cpu
    - Linux/Windows: cuda (if available) > cpu
    
    Returns:
        torch.device: The best available device.
    """
    if torch.cuda.is_available():
        return torch.device("cuda")
    
    # Check for MPS (Metal Performance Shaders) on macOS
    if platform.system() == "Darwin" and hasattr(torch.backends, "mps"):
        if torch.backends.mps.is_available():
            return torch.device("mps")
    
    return torch.device("cpu")


def resolve_device_override(
    device_override: Optional[str],
    model_id: Optional[str] = None,
) -> tuple[torch.device, Optional[str]]:
    """
    Resolve device override with safe fallback.
    
    Args:
        device_override: One of {"cuda", "cpu", "mps", None}.
            - If "cuda": use cuda:0 if torch.cuda.is_available(), else fall back to cpu.
            - If "mps": use mps if torch.backends.mps.is_available(), else fall back to cpu.
            - If "cpu": always cpu.
            - If None: auto-detect in priority order: cuda -> mps -> cpu.
        model_id: Optional model ID to check for special handling (e.g., tiny-gpt2 is CPU-only).
    
    Returns:
        Tuple of (torch.device, Optional[str]): 
        - The resolved device (always valid, never raises)
        - Forced reason string if device was forced (e.g., "forced" for tiny-gpt2), None otherwise
    """
    # Special handling: tiny-gpt2 is always CPU-only (Torch >=2.6 blocks CUDA load)
    if model_id and ("tiny-gpt2" in model_id.lower()):
        if device_override and device_override.lower() == "cuda":
            print(f"[tiny-gpt2] CUDA requested → forcing CPU (this model cannot run on CUDA on Torch >= 2.6)")
        return torch.device("cpu"), "forced"
    
    if device_override is None:
        return get_device(), None
    
    device_override = device_override.lower().strip()
    
    if device_override == "cuda":
        if torch.cuda.is_available():
            return torch.device("cuda"), None
        else:
            print("Warning: CUDA requested but not available, falling back to CPU")
            return torch.device("cpu"), None
    
    elif device_override == "mps":
        if platform.system() == "Darwin" and hasattr(torch.backends, "mps"):
            if torch.backends.mps.is_available():
                return torch.device("mps"), None
        print("Warning: MPS requested but not available, falling back to CPU")
        return torch.device("cpu"), None
    
    elif device_override == "cpu":
        return torch.device("cpu"), None
    
    else:
        print(f"Warning: Unknown device override '{device_override}', using auto-detection")
        return get_device(), None


def time_function(func):
    """Simple decorator to time function execution."""
    def wrapper(*args, **kwargs):
        start = time.time()
        result = func(*args, **kwargs)
        elapsed_ms = (time.time() - start) * 1000
        return result, elapsed_ms
    return wrapper

