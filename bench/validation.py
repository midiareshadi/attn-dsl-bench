"""Correctness validation for attention kernels.

Every measured kernel must produce numerically equivalent output to a
reference implementation. We use the same metrics as HazyResearch's
attn_fwd_baselines.py for consistency across the literature.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch


@dataclass
class ValidationResult:
    """Numerical comparison between two attention outputs."""
    max_abs_error: float
    rel_l2_error: float
    cos_similarity: float
    error_count: int
    numel: int
    error_rate: float
    passed: bool


DEFAULT_ABS_TOL = 0.001
DEFAULT_REL_TOL = 0.05
DEFAULT_MAX_ERROR_RATE = 0.005


def validate_outputs(
    reference: torch.Tensor,
    candidate: torch.Tensor,
    abs_tol: float = DEFAULT_ABS_TOL,
    rel_tol: float = DEFAULT_REL_TOL,
    max_error_rate: float = DEFAULT_MAX_ERROR_RATE,
) -> ValidationResult:
    """Compare two attention outputs and report robustness metrics."""
    if reference.shape != candidate.shape:
        raise ValueError(
            f"Shape mismatch: reference {reference.shape} vs candidate {candidate.shape}"
        )

    ref = reference.float()
    pred = candidate.float()
    diff = (ref - pred).abs()

    denom = ref.abs().clamp_min(1e-6)
    error_mask = diff > (abs_tol + rel_tol * denom)
    error_count = int(error_mask.sum().item())
    numel = ref.numel()
    error_rate = error_count / numel if numel > 0 else 0.0

    max_abs_error = float(diff.max().item())

    ref_norm = ref.pow(2).sum().sqrt()
    if ref_norm.item() > 0:
        rel_l2_error = float((diff.pow(2).sum().sqrt() / ref_norm).item())
    else:
        rel_l2_error = float("nan")

    cos_similarity = float(
        torch.nn.functional.cosine_similarity(
            ref.flatten(), pred.flatten(), dim=0
        ).item()
    )

    return ValidationResult(
        max_abs_error=max_abs_error,
        rel_l2_error=rel_l2_error,
        cos_similarity=cos_similarity,
        error_count=error_count,
        numel=numel,
        error_rate=error_rate,
        passed=error_rate <= max_error_rate,
    )
