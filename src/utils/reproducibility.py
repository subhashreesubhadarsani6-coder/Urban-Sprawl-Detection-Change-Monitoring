"""Reproducibility and Determinism Utilities.

Provides comprehensive seed setting across Python, NumPy, PyTorch, and CUDA,
plus hardware and environment profiling helpers.
"""

from typing import Dict, Any
import os
import random
import platform
import numpy as np
import torch

def set_seed(seed: int = 42, deterministic: bool = True):
    """Seed Python, NumPy, and PyTorch for reproducible runs."""
    random.seed(seed)
    np.random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)

    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)

    if deterministic:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def seed_worker(worker_id: int):
    """Worker initialization function for PyTorch DataLoader."""
    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed)


def get_environment_info() -> Dict[str, Any]:
    """Capture runtime hardware, OS, and library versions."""
    info = {
        "os": platform.platform(),
        "python_version": platform.python_version(),
        "pytorch_version": torch.__version__,
        "cuda_available": torch.cuda.is_available(),
    }
    if torch.cuda.is_available():
        info.update({
            "device_count": torch.cuda.device_count(),
            "device_name": torch.cuda.get_device_name(0),
            "vram_gb": round(torch.cuda.get_device_properties(0).total_memory / (1024**3), 2),
            "cuda_version": torch.version.cuda
        })
    return info
