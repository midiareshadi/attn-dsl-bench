"""Generate figures from the sweep v1 timing + counter datasets.

Reads results/raw/sweep_v1.csv (throughput) and results/raw/counters_v1.csv
(hardware counters), joins them on (implementation, seq_len, head_dim),
and emits three figures to results/figures/ as both PNG and SVG:

1. throughput_vs_seqlen  — TFLOPS vs sequence length, faceted by head_dim
2. mfma_util_vs_seqlen   — MfmaUtil vs sequence length, faceted by head_dim
3. throughput_vs_mfma    — scatter of TFLOPS against MfmaUtil (the thesis)

Reproducible: anyone can regenerate the figures from the committed CSVs.

Usage
-----
    python3 analysis/plot_sweep.py
"""

from __future__ import annotations

import os

import matplotlib
matplotlib.use("Agg")  # headless; no display needed
import matplotlib.pyplot as plt
import pandas as pd


RAW_DIR = "results/raw"
FIG_DIR = "results/figures"

# Consistent identity per implementation across all figures.
IMPL_ORDER = ["aiter", "triton", "pytorch_sdpa"]
IMPL_LABEL = {
    "aiter": "AITER",
    "triton": "Triton (HK baseline)",
    "pytorch_sdpa": "PyTorch SDPA (AOTriton)",
}
IMPL_COLOR = {
    "aiter": "#1f77b4",        # blue
    "triton": "#d62728",       # red
    "pytorch_sdpa": "#2ca02c", # green
}
HEAD_DIM_MARKER = {64: "o", 128: "s"}

plt.rcParams.update({
    "figure.dpi": 120,
    "font.size": 11,
    "axes.grid": True,
    "grid.alpha": 0.3,
    "axes.spines.top": False,
    "axes.spines.right": False,
})


def load_data() -> pd.DataFrame:
    """Load and join the timing and counter CSVs on shape keys."""
    timing = pd.read_csv(os.path.join(RAW_DIR, "sweep_v1.csv"))
    counters = pd.read_csv(os.path.join(RAW_DIR, "counters_v1.csv"))

    keys = ["implementation", "seq_len", "head_dim"]
    timing_cols = keys + ["achieved_tflops", "causal", "dtype"]
    counter_cols = keys + ["mfma_util", "occupancy_percent", "lds_util"]

    merged = pd.merge(
        timing[timing_cols],
        counters[counter_cols],
        on=keys,
        how="inner",
    )
    return merged


def _save(fig, name: str) -> None:
    for ext in ("png", "svg"):
        path = os.path.join(FIG_DIR, f"{name}.{ext}")
        fig.savefig(path, bbox_inches="tight")
        print(f"  wrote {path}")


def plot_metric_vs_seqlen(df: pd.DataFrame, metric: str, ylabel: str,
                          title: str, name: str) -> None:
    """One panel per head_dim; one line per implementation."""
    head_dims = sorted(df["head_dim"].unique())
    fig, axes = plt.subplots(1, len(head_dims), figsize=(11, 4.2), sharey=True)
    if len(head_dims) == 1:
        axes = [axes]

    for ax, hd in zip(axes, head_dims):
        sub = df[df["head_dim"] == hd]
        for impl in IMPL_ORDER:
            s = sub[sub["implementation"] == impl].sort_values("seq_len")
            if s.empty:
                continue
            ax.plot(
                s["seq_len"], s[metric],
                marker=HEAD_DIM_MARKER.get(hd, "o"),
                color=IMPL_COLOR[impl],
                label=IMPL_LABEL[impl],
                linewidth=2, markersize=6,
            )
        ax.set_xscale("log", base=2)
        ax.set_xticks(sorted(sub["seq_len"].unique()))
        ax.get_xaxis().set_major_formatter(matplotlib.ticker.ScalarFormatter())
        ax.set_xlabel("Sequence length")
        ax.set_title(f"head_dim = {hd}")
    axes[0].set_ylabel(ylabel)
    axes[0].legend(frameon=False, fontsize=9, loc="upper left")
    fig.suptitle(title, fontsize=13, y=1.02)
    _save(fig, name)
    plt.close(fig)


def plot_throughput_vs_mfma(df: pd.DataFrame) -> None:
    """Scatter: throughput against MfmaUtil. The thesis figure."""
    fig, ax = plt.subplots(figsize=(7, 5.5))
    for impl in IMPL_ORDER:
        s = df[df["implementation"] == impl]
        for hd in sorted(s["head_dim"].unique()):
            ss = s[s["head_dim"] == hd]
            ax.scatter(
                ss["mfma_util"], ss["achieved_tflops"],
                color=IMPL_COLOR[impl],
                marker=HEAD_DIM_MARKER.get(hd, "o"),
                s=70, alpha=0.85,
                edgecolors="white", linewidths=0.5,
                label=f"{IMPL_LABEL[impl]}, d={hd}",
            )
    ax.set_xlabel("MFMA utilization (%)")
    ax.set_ylabel("Achieved throughput (TFLOPS)")
    ax.set_title("Throughput is explained by matrix-core utilization\n"
                 "MI300X, B=16, H=16, causal", fontsize=12)
    ax.legend(frameon=False, fontsize=8.5, loc="upper left")
    _save(fig, "throughput_vs_mfma")
    plt.close(fig)


def main() -> None:
    os.makedirs(FIG_DIR, exist_ok=True)
    df = load_data()
    print(f"Joined dataset: {len(df)} rows")
    print(df[["implementation", "seq_len", "head_dim",
              "achieved_tflops", "mfma_util"]].to_string(index=False))
    print()

    print("Figure 1: throughput vs sequence length")
    plot_metric_vs_seqlen(
        df, "achieved_tflops", "Achieved TFLOPS",
        "Attention forward throughput on MI300X (B=16, H=16, causal, BF16)",
        "throughput_vs_seqlen",
    )
    print("Figure 2: MfmaUtil vs sequence length")
    plot_metric_vs_seqlen(
        df, "mfma_util", "MFMA utilization (%)",
        "Matrix-core utilization on MI300X (B=16, H=16, causal)",
        "mfma_util_vs_seqlen",
    )
    print("Figure 3: throughput vs MfmaUtil scatter")
    plot_throughput_vs_mfma(df)
    print("\nDone.")


if __name__ == "__main__":
    import matplotlib.ticker  # noqa: needed for ScalarFormatter
    main()
