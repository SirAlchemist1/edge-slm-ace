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


def resolve_device_override(device_override: Optional[str]) -> torch.device:
    """
    Resolve device override with safe fallback.
    
    Args:
        device_override: One of {"cuda", "cpu", "mps", None}.
            - If "cuda": use cuda:0 if torch.cuda.is_available(), else fall back to cpu.
            - If "mps": use mps if torch.backends.mps.is_available(), else fall back to cpu.
            - If "cpu": always cpu.
            - If None: auto-detect in priority order: cuda -> mps -> cpu.
    
    Returns:
        torch.device: The resolved device (always valid, never raises).
    """
    if device_override is None:
        return get_device()
    
    device_override = device_override.lower().strip()
    
    if device_override == "cuda":
        if torch.cuda.is_available():
            return torch.device("cuda")
        else:
            print("Warning: CUDA requested but not available, falling back to CPU")
            return torch.device("cpu")
    
    elif device_override == "mps":
        if platform.system() == "Darwin" and hasattr(torch.backends, "mps"):
            if torch.backends.mps.is_available():
                return torch.device("mps")
        print("Warning: MPS requested but not available, falling back to CPU")
        return torch.device("cpu")
    
    elif device_override == "cpu":
        return torch.device("cpu")
    
    else:
        print(f"Warning: Unknown device override '{device_override}', using auto-detection")
        return get_device()


def time_function(func):
    """Simple decorator to time function execution."""
    def wrapper(*args, **kwargs):
        start = time.time()
        result = func(*args, **kwargs)
        elapsed_ms = (time.time() - start) * 1000
        return result, elapsed_ms
    return wrapper

