"""AITER kernel wrapper.

Wraps aiter.flash_attn_func with the harness-canonical layout (B, H, N, D)
to match the PyTorch SDPA wrapper. Internally transposes to AITER's
expected (B, N, H, D) layout before the call, and transposes the output
back so validation against any (B, H, N, D)-layout reference works
without extra logic in the runner.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Tuple

import torch


@dataclass
class AttentionShape:
    """Standard attention input shape parameters."""
    batch: int
    num_heads: int
    num_kv_heads: int
    seq_len: int
    head_dim: int
    causal: bool


def make_inputs(
    shape: AttentionShape,
    dtype: torch.dtype,
    device: str = "cuda",
    seed: int = 0,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Allocate (Q, K, V) in the harness-canonical (B, H, N, D) layout.

    The wrapper handles the transpose to AITER's (B, N, H, D) layout
    inside the callable.
    """
    g = torch.Generator(device=device).manual_seed(seed)
    q = torch.randn(
        shape.batch, shape.num_heads, shape.seq_len, shape.head_dim,
        dtype=dtype, device=device, generator=g,
    )
    k = torch.randn(
        shape.batch, shape.num_kv_heads, shape.seq_len, shape.head_dim,
        dtype=dtype, device=device, generator=g,
    )
    v = torch.randn(
        shape.batch, shape.num_kv_heads, shape.seq_len, shape.head_dim,
        dtype=dtype, device=device, generator=g,
    )
    return q, k, v


def build_callable(
    shape: AttentionShape,
    dtype: torch.dtype,
    device: str = "cuda",
    seed: int = 0,
) -> Tuple[Callable[[], torch.Tensor], torch.Tensor, torch.Tensor, torch.Tensor]:
    """Construct the closure that the timing harness will invoke.

    Inputs are allocated in (B, H, N, D); the closure transposes to
    AITER's (B, N, H, D), runs flash_attn_func, then transposes the
    output back to (B, H, N, D) so it's comparable against PyTorch
    SDPA outputs.
    """
    import aiter  # imported lazily so this file parses on non-AMD hosts

    q_bhnd, k_bhnd, v_bhnd = make_inputs(shape, dtype, device=device, seed=seed)

    # Pre-transpose once and capture in the closure. We're timing the
    # AITER call only, not the transposes — which is fair because the
    # PyTorch SDPA wrapper also isn't paying for any layout conversion.
    q_bnhd = q_bhnd.transpose(1, 2).contiguous()
    k_bnhd = k_bhnd.transpose(1, 2).contiguous()
    v_bnhd = v_bhnd.transpose(1, 2).contiguous()

    def fn() -> torch.Tensor:
        out, _ = aiter.flash_attn_func(
            q_bnhd, k_bnhd, v_bnhd,
            causal=shape.causal,
            return_lse=True,
            deterministic=True,
        )
        # Transpose back to harness-canonical (B, H, N, D)
        return out.transpose(1, 2).contiguous()

    return fn, q_bhnd, k_bhnd, v_bhnd


def run_once(
    shape: AttentionShape,
    dtype: torch.dtype,
    device: str = "cuda",
    seed: int = 0,
) -> torch.Tensor:
    """Convenience: build inputs, call AITER, return the output tensor."""
    fn, _, _, _ = build_callable(shape, dtype, device=device, seed=seed)
    return fn()


IMPLEMENTATION_NAME = "aiter"
