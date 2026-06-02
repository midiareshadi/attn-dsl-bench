"""PyTorch SDPA kernel wrapper.

Wraps torch.nn.functional.scaled_dot_product_attention with the
common harness interface: a function that builds inputs and returns
(callable, expected_output_shape).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Tuple

import torch
from torch.nn.functional import scaled_dot_product_attention


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
    """Allocate (Q, K, V) tensors for an attention call.

    Layout: (B, H, N, D) for Q, (B, H_kv, N, D) for K and V — the
    layout PyTorch's SDPA expects (heads-second).
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
    """Construct the closure that the timing harness will invoke."""
    q, k, v = make_inputs(shape, dtype, device=device, seed=seed)

    def fn() -> torch.Tensor:
        return scaled_dot_product_attention(
            q, k, v,
            attn_mask=None,
            dropout_p=0.0,
            is_causal=shape.causal,
            scale=None,
            enable_gqa=(shape.num_kv_heads != shape.num_heads),
        )

    return fn, q, k, v


def run_once(
    shape: AttentionShape,
    dtype: torch.dtype,
    device: str = "cuda",
    seed: int = 0,
) -> torch.Tensor:
    """Convenience: build inputs, call SDPA, return the output tensor."""
    fn, _, _, _ = build_callable(shape, dtype, device=device, seed=seed)
    return fn()


IMPLEMENTATION_NAME = "pytorch_sdpa"
