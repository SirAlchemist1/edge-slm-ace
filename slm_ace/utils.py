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


def time_function(func):
    """Simple decorator to time function execution."""
    def wrapper(*args, **kwargs):
        start = time.time()
        result = func(*args, **kwargs)
        elapsed_ms = (time.time() - start) * 1000
        return result, elapsed_ms
    return wrapper

