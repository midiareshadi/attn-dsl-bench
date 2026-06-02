"""Hardware-counter profiling runner using rocprofv3.

Separate from bench/runner.py (which does pure GPU-event timing). This
module wraps each kernel invocation in rocprofv3 to collect hardware
performance counters, writing one row per (implementation, shape) to a
counters CSV that joins to the timing sweep on the same keys.

Design notes
------------
- rocprofv3 collects counters for ALL GPU kernels in the wrapped
  process. We isolate the attention kernel as the one with the highest
  MfmaFlopsBF16 (flash attention is matrix-core dominated; helper
  kernels like RNG/elementwise have zero MFMA). This is more robust
  than per-implementation kernel-name regexes.
- JIT/compilation is warmed up BEFORE the kernel runs we profile, so
  AITER's ~80s build and Triton's autotune are not captured.
- Counters are deterministic across iterations, so we profile only a
  few iterations (default 3), not the 30 used for timing.
- The four derived metrics (MfmaUtil, OccupancyPercent, MfmaFlopsBF16,
  LdsUtil) collect in a single rocprofv3 pass on this hardware.

Usage
-----
    python3 -m bench.profile_runner --config configs/sweep_v1_aiter.yaml \\
                                     --output results/raw/counters_v1.csv
"""

from __future__ import annotations

import argparse
import csv
import glob
import os
import subprocess
import sys
import tempfile
from collections import defaultdict
from pathlib import Path
from typing import Any

import yaml

from bench.environment import capture as capture_env


# The derived metrics we collect. All confirmed to collect in one pass
# on MI300X / rocprofv3 1.1.0.
COUNTERS = ["MfmaUtil", "OccupancyPercent", "MfmaFlopsBF16", "LdsUtil"]

# Number of profiled iterations (counters are deterministic; few suffice).
PROFILE_ITERS = 3

# Number of warmup iterations (run before profiling, triggers JIT/autotune).
WARMUP_ITERS = 5


CSV_COLUMNS = [
    # Provenance
    "timestamp_utc", "session_id", "hostname",
    "rocm_version", "hip_version", "pytorch_version", "gpu_name",
    # Configuration (join keys with the timing sweep)
    "implementation", "kernel",
    "batch", "num_heads", "num_kv_heads",
    "seq_len", "head_dim", "causal", "dtype",
    # Profiling metadata
    "profile_iters", "kernel_name", "num_dispatches",
    # Hardware counters (averaged across dispatches)
    "mfma_util", "occupancy_percent", "mfma_flops_bf16", "lds_util",
    # Notes
    "notes",
]


# Build the temp script that runs one kernel cell under profiling.
_PROFILE_SCRIPT_TEMPLATE = '''
import sys
sys.path.insert(0, "{repo_root}")
import torch
from kernels.{impl}.run import AttentionShape, build_callable

shape = AttentionShape(
    batch={batch}, num_heads={num_heads}, num_kv_heads={num_kv_heads},
    seq_len={seq_len}, head_dim={head_dim}, causal={causal},
)
dtype = {dtype_expr}
fn, q, k, v = build_callable(shape, dtype, device="cuda", seed=0)

# Warmup OUTSIDE the profiled region (triggers JIT / autotune).
for _ in range({warmup_iters}):
    out = fn()
torch.cuda.synchronize()

# Profiled region: these are the dispatches rocprofv3 captures.
for _ in range({profile_iters}):
    out = fn()
torch.cuda.synchronize()
print("profiled", "{impl}", tuple(out.shape))
'''


DTYPE_EXPR = {
    "bfloat16": "torch.bfloat16",
    "bf16": "torch.bfloat16",
    "float16": "torch.float16",
    "fp16": "torch.float16",
    "float32": "torch.float32",
    "fp32": "torch.float32",
}


def load_config(path: str) -> dict:
    with open(path) as f:
        return yaml.safe_load(f)


def _shapes_from_config(config: dict) -> list[dict]:
    if "shapes" in config:
        return config["shapes"]
    elif "shape" in config:
        return [config["shape"]]
    raise ValueError("Config must have 'shape' or 'shapes'")


def profile_one(config: dict, shape_cfg: dict, repo_root: str) -> dict:
    """Run one cell under rocprofv3, parse counters, return a row dict."""
    impl = config["implementation"]
    dtype_str = config["dtype"].lower()
    dtype_expr = DTYPE_EXPR[dtype_str]

    script = _PROFILE_SCRIPT_TEMPLATE.format(
        repo_root=repo_root,
        impl=impl,
        batch=int(shape_cfg["batch"]),
        num_heads=int(shape_cfg["num_heads"]),
        num_kv_heads=int(shape_cfg["num_kv_heads"]),
        seq_len=int(shape_cfg["seq_len"]),
        head_dim=int(shape_cfg["head_dim"]),
        causal=bool(shape_cfg["causal"]),
        dtype_expr=dtype_expr,
        warmup_iters=WARMUP_ITERS,
        profile_iters=PROFILE_ITERS,
    )

    with tempfile.TemporaryDirectory() as tmpdir:
        script_path = os.path.join(tmpdir, "cell.py")
        out_dir = os.path.join(tmpdir, "rocprof_out")
        with open(script_path, "w") as f:
            f.write(script)

        cmd = [
            "rocprofv3",
            "--pmc", *COUNTERS,
            "--output-format", "csv",
            "-d", out_dir,
            "--", sys.executable, script_path,
        ]
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=600)

        csvs = glob.glob(os.path.join(out_dir, "*", "*counter_collection.csv"))
        if not csvs:
            raise RuntimeError(
                f"No counter CSV produced. stderr tail:\\n{result.stderr[-500:]}"
            )
        counter_csv = csvs[0]

        rows = list(csv.DictReader(open(counter_csv)))

    # Group by kernel, collect each counter's values.
    by_kernel: dict[str, dict] = defaultdict(
        lambda: {"dispatches": set(), "counters": defaultdict(list)}
    )
    for r in rows:
        kname = r["Kernel_Name"]
        by_kernel[kname]["dispatches"].add(r["Dispatch_Id"])
        by_kernel[kname]["counters"][r["Counter_Name"]].append(
            float(r["Counter_Value"])
        )

    # Attention kernel = highest total MfmaFlopsBF16.
    def total_mfma(data):
        return sum(data["counters"].get("MfmaFlopsBF16", [0]))

    attn_name, attn = max(by_kernel.items(), key=lambda kv: total_mfma(kv[1]))

    def avg(counter):
        vals = attn["counters"].get(counter, [])
        return sum(vals) / len(vals) if vals else float("nan")

    env = capture_env()
    row: dict[str, Any] = {}
    row.update({
        "timestamp_utc": env.timestamp_utc,
        "session_id": env.session_id,
        "hostname": env.hostname,
        "rocm_version": env.rocm_version,
        "hip_version": env.hip_version,
        "pytorch_version": env.pytorch_version,
        "gpu_name": env.gpu_name,
        "implementation": impl,
        "kernel": config.get("kernel", ""),
        "batch": int(shape_cfg["batch"]),
        "num_heads": int(shape_cfg["num_heads"]),
        "num_kv_heads": int(shape_cfg["num_kv_heads"]),
        "seq_len": int(shape_cfg["seq_len"]),
        "head_dim": int(shape_cfg["head_dim"]),
        "causal": bool(shape_cfg["causal"]),
        "dtype": dtype_str,
        "profile_iters": PROFILE_ITERS,
        "kernel_name": attn_name[:200],
        "num_dispatches": len(attn["dispatches"]),
        "mfma_util": avg("MfmaUtil"),
        "occupancy_percent": avg("OccupancyPercent"),
        "mfma_flops_bf16": avg("MfmaFlopsBF16"),
        "lds_util": avg("LdsUtil"),
        "notes": config.get("notes", ""),
    })
    return row


def append_row(csv_path: str, row: dict) -> None:
    path = Path(csv_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    file_exists = path.exists()
    with open(path, "a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=CSV_COLUMNS, extrasaction="ignore")
        if not file_exists:
            writer.writeheader()
        writer.writerow(row)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Profile attention cells with rocprofv3")
    parser.add_argument("--config", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--repo-root", default=os.getcwd(),
                        help="Repo root for the profiled script's sys.path")
    args = parser.parse_args(argv)

    config = load_config(args.config)
    shapes = _shapes_from_config(config)

    rows_written = 0
    for i, shape_cfg in enumerate(shapes):
        try:
            row = profile_one(config, shape_cfg, args.repo_root)
        except Exception as e:
            print(f"[cell {i+1}/{len(shapes)}] FAILED for shape {shape_cfg}: {e}")
            continue
        append_row(args.output, row)
        rows_written += 1
        print(f"[cell {i+1}/{len(shapes)}] {row['implementation']} @ "
              f"N={row['seq_len']} D={row['head_dim']}  "
              f"MfmaUtil={row['mfma_util']:.1f}%  "
              f"Occ={row['occupancy_percent']:.1f}%  "
              f"LdsUtil={row['lds_util']:.1f}%  "
              f"({row['num_dispatches']} dispatches)")

    print(f"Wrote {rows_written}/{len(shapes)} rows to {args.output}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
