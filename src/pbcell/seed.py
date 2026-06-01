"""Utilities for reproducible training and evaluation."""

from __future__ import annotations

import os
import random

import numpy as np
import torch


def seed_all(seed: int, deterministic: bool = True) -> int:
    """Seed Python, NumPy, and PyTorch RNGs with a single integer."""
    seed_int = int(seed)
    os.environ["PYTHONHASHSEED"] = str(seed_int)

    random.seed(seed_int)
    np.random.seed(seed_int)

    torch.manual_seed(seed_int)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed_int)

    if deterministic:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    else:
        torch.backends.cudnn.deterministic = False

    use_det = getattr(torch, "use_deterministic_algorithms", None)
    if callable(use_det):
        try:
            use_det(deterministic, warn_only=True)
        except TypeError:
            use_det(deterministic)

    return seed_int
