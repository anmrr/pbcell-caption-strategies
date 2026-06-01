"""Evaluation metrics for the diffusion model.

Evaluation modules for diffusion model assessment
- fid:       Distributional image quality (FID with InceptionV3).
- oracle:    Class fidelity (does a real-trained classifier recognize synthetic samples?).
"""

from .fid import compute_fid_for_dirs
from .oracle import evaluate_synthetic

__all__ = ["compute_fid_for_dirs", "evaluate_synthetic"]
