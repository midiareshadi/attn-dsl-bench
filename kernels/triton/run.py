"""Triton kernel wrapper.

Wraps HazyResearch's triton_baseline_v02.py (the AMD Triton team's
FlashAttention v2 implementation) with the harness-canonical interface.

The Triton baseline is a 83KB self-contained script with:
- An autograd.Function `_attention` at line 1084
- The alias `attention = _attention.apply` at line 1259
- A `MetaData` class for configuring the call
- Native bhsd layout — same as our harness canonical, no transposes

We import the script by file path because it isn't on sys.path and
doesn't ship as a Python package. The HipKittens repo is expected to
be cloned at /workspace/HipKittens (the standard bootstrap location).

Configurable via the TRITON_BASELINE_PATH environment variable for
flexibility in non-default setups.
"""

from __future__ import annotations

import importlib.util
import os
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


_DEFAULT_TRITON_PATH = "/workspace/HipKittens/analysis/baselines/attn/triton_baseline_v02.py"
_triton_module = None


def _load_triton_module():
    """Lazily load triton_baseline_v02 as a module.

    Cached so we pay the import cost (which includes Triton autotune
    setup) only once per Python process.
    """
    global _triton_module
    if _triton_module is not None:
        return _triton_module

    path = os.environ.get("TRITON_BASELINE_PATH", _DEFAULT_TRITON_PATH)
    if not os.path.isfile(path):
        raise FileNotFoundError(
            f"Triton baseline not found at {path}. Set TRITON_BASELINE_PATH "
            f"or clone HipKittens at /workspace/HipKittens."
        )

    spec = importlib.util.spec_from_file_location("triton_baseline_v02", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    _triton_module = module
    return module


def make_inputs(
    shape: AttentionShape,
    dtype: torch.dtype,
    device: str = "cuda",
    seed: int = 0,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Allocate (Q, K, V) tensors in the harness-canonical (B, H, N, D) layout.

    This matches the Triton baseline's 'bhsd' layout natively — no
    transpose needed inside the closure.
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

    The Triton baseline expects pre-allocated input and output tensors
    plus a MetaData object. We build all of these once outside the
    closure so the timing measures only the kernel launch + execution.
    """
    triton_mod = _load_triton_module()
    attention = triton_mod.attention            # _attention.apply
    MetaData = triton_mod.MetaData              # configuration object

    q, k, v = make_inputs(shape, dtype, device=device, seed=seed)
    o = torch.empty_like(q)

    metadata = MetaData(sm_scale=shape.head_dim ** -0.5)
    metadata.max_seqlens_q = shape.seq_len
    metadata.max_seqlens_k = shape.seq_len
    metadata.layout = "bhsd"
    if shape.causal:
        metadata.need_causal()

    def fn() -> torch.Tensor:
        # The autograd.Function returns (output, softmax_lse, exp_scores).
        # We only care about the output for timing/validation purposes.
        out, _, _ = attention(q, k, v, o, metadata)
        return out

    return fn, q, k, v


def run_once(
    shape: AttentionShape,
    dtype: torch.dtype,
    device: str = "cuda",
    seed: int = 0,
) -> torch.Tensor:
    """Convenience: build inputs, call Triton FA, return the output tensor."""
    fn, _, _, _ = build_callable(shape, dtype, device=device, seed=seed)
    return fn()


IMPLEMENTATION_NAME = "triton"
