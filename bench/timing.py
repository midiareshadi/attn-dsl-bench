"""Timing utilities for GPU kernel benchmarks.

All timing uses torch.cuda.Event for accurate GPU-side measurement.
We measure the kernel call only — not data allocation or transfer.
"""

from __future__ import annotations

import statistics
from dataclasses import dataclass
from typing import Callable

import torch


@dataclass
class TimingResult:
    """Statistics from a timed measurement."""
    avg_ms: float
    median_ms: float
    p25_ms: float
    p75_ms: float
    min_ms: float
    max_ms: float
    num_iters: int


def time_kernel(
    fn: Callable[[], object],
    num_warmup: int,
    num_iters: int,
) -> TimingResult:
    """Time a no-argument callable using torch.cuda.Event.

    The callable is invoked num_warmup times (discarded), then
    num_iters times (measured). Each measurement synchronizes the
    GPU before and after the call.

    Returns
    -------
    TimingResult
        Statistics over the num_iters measured runs.
    """
    if num_iters < 1:
        raise ValueError("num_iters must be >= 1")

    # Warmup
    for _ in range(num_warmup):
        fn()
    torch.cuda.synchronize()

    # Measurement
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    timings_ms: list[float] = []

    for _ in range(num_iters):
        torch.cuda.synchronize()
        start.record()
        fn()
        end.record()
        torch.cuda.synchronize()
        timings_ms.append(start.elapsed_time(end))

    timings_ms.sort()
    n = len(timings_ms)
    return TimingResult(
        avg_ms=statistics.mean(timings_ms),
        median_ms=statistics.median(timings_ms),
        p25_ms=timings_ms[n // 4],
        p75_ms=timings_ms[(3 * n) // 4],
        min_ms=timings_ms[0],
        max_ms=timings_ms[-1],
        num_iters=n,
    )


def attention_flops(
    batch: int,
    seq_len: int,
    num_heads: int,
    head_dim: int,
    causal: bool,
) -> int:
    """FLOPs for a forward attention pass.

    Standard formula: 4 * B * N^2 * H * D, halved for causal masking.
    Matches the formula used in HazyResearch's attn_fwd_baselines.py.
    """
    flop = 4 * batch * seq_len * seq_len * num_heads * head_dim
    if causal:
        flop //= 2
    return flop


def tflops(total_flops: int, time_ms: float) -> float:
    """Convert FLOPs + milliseconds to achieved TFLOPS."""
    if time_ms <= 0:
        return 0.0
    flops_per_sec = total_flops / (time_ms / 1000.0)
    return flops_per_sec / 1e12
