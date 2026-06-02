"""TileLang kernel wrapper.

Wraps the FlashAttention forward example from tile-ai/tilelang
(examples/flash_attention/example_mha_fwd_bhsd.py).

The TileLang example:
- Hardcodes FP16 dtype in the kernel definition (T.float16).
- Uses (B, H, N, D) bhsd layout — same as our harness canonical.
- Is parameterized by shape: flashattn(batch, heads, seq_q, seq_kv,
  dim, is_causal, block_M, block_N, num_stages, threads) returns a
  compiled kernel object callable as kernel(Q, K, V) -> Output.

Implication: this wrapper runs at FP16 even when the harness asks
for BF16. The CSV will faithfully record dtype=float16 for TileLang
runs so this is comparable-but-not-apples-to-apples relative to the
BF16 runs of AITER, Triton, SDPA. A BF16 TileLang example can be
added later if needed.

Path: /workspace/tilelang/examples/flash_attention/example_mha_fwd_bhsd.py
(overridable via TILELANG_EXAMPLE_PATH env var).
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


_DEFAULT_TILELANG_EXAMPLE = (
    "/workspace/tilelang/examples/flash_attention/example_mha_fwd_bhsd.py"
)
_tilelang_module = None


def _load_tilelang_module():
    """Lazily load the TileLang FA example as a module.

    Cached per-process. The first load also triggers TileLang's own
    TVM-based import chain, which is slow (~10s) and emits a few
    harmless field-duplication warnings.
    """
    global _tilelang_module
    if _tilelang_module is not None:
        return _tilelang_module

    path = os.environ.get("TILELANG_EXAMPLE_PATH", _DEFAULT_TILELANG_EXAMPLE)
    if not os.path.isfile(path):
        raise FileNotFoundError(
            f"TileLang example not found at {path}. Set TILELANG_EXAMPLE_PATH "
            f"or clone tile-ai/tilelang at /workspace/tilelang."
        )

    spec = importlib.util.spec_from_file_location("tilelang_example_fa", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    _tilelang_module = module
    return module


def make_inputs(
    shape: AttentionShape,
    dtype: torch.dtype,
    device: str = "cuda",
    seed: int = 0,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Allocate (Q, K, V) in the harness-canonical (B, H, N, D) layout.

    Note: TileLang's example expects FP16. Callers should pass
    torch.float16 to match the kernel's compiled dtype.
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

    Compiles the TileLang kernel for our specific shape once, then
    returns a closure that calls the compiled kernel on the prepared
    inputs.
    """
    if shape.num_heads != shape.num_kv_heads:
        raise NotImplementedError(
            "example_mha_fwd_bhsd.py only supports MHA (num_heads == num_kv_heads). "
            "Use a GQA-specific TileLang example for grouped-query attention."
        )

    if dtype != torch.float16:
        raise ValueError(
            "The TileLang example_mha_fwd_bhsd kernel hardcodes T.float16; "
            "pass torch.float16 inputs. To benchmark BF16 with TileLang, "
            "build a BF16-variant of the example."
        )

    tilelang_mod = _load_tilelang_module()
    flashattn = tilelang_mod.flashattn

    q, k, v = make_inputs(shape, dtype, device=device, seed=seed)

    # Compile the kernel for this specific shape. The example uses
    # block_M=128, block_N=128, num_stages=2, threads=256 in its
    # run_regression_perf — we use the same.
    kernel = flashattn(
        batch=shape.batch,
        heads=shape.num_heads,
        seq_q=shape.seq_len,
        seq_kv=shape.seq_len,
        dim=shape.head_dim,
        is_causal=shape.causal,
        block_M=128,
        block_N=128,
        num_stages=2,
        threads=256,
    )

    def fn() -> torch.Tensor:
        # kernel(Q, K, V) returns Output thanks to @tilelang.jit(out_idx=[3]).
        return kernel(q, k, v)

    return fn, q, k, v


def run_once(
    shape: AttentionShape,
    dtype: torch.dtype,
    device: str = "cuda",
    seed: int = 0,
) -> torch.Tensor:
    """Convenience: build inputs, call TileLang FA, return the output."""
    fn, _, _, _ = build_callable(shape, dtype, device=device, seed=seed)
    return fn()


IMPLEMENTATION_NAME = "tilelang"
