"""Benchmark runner: read a YAML config, run one or more cells, append CSV rows.

Two config formats supported:
- Single-cell: top-level `shape: {...}`
- Sweep:        top-level `shapes: [{...}, {...}, ...]`

Usage
-----
    python3 -m bench.runner --config configs/sweep.yaml \\
                            --output results/raw/sweep.csv

Each cell appends one row to the output CSV (creating the file with a
header if it doesn't exist). Cells that crash log a failure and the
sweep continues with the next cell.
"""

from __future__ import annotations

import argparse
import csv
import importlib
import os
import sys
from dataclasses import asdict
from pathlib import Path
from typing import Any

import torch
import yaml

from bench.environment import capture as capture_env
from bench.timing import attention_flops, tflops, time_kernel
from bench.validation import validate_outputs


DTYPE_MAP = {
    "bfloat16": torch.bfloat16,
    "bf16": torch.bfloat16,
    "float16": torch.float16,
    "fp16": torch.float16,
    "float32": torch.float32,
    "fp32": torch.float32,
}


CSV_COLUMNS = [
    # Provenance
    "timestamp_utc", "session_id", "hostname",
    "rocm_version", "hip_version",
    "pytorch_version", "pytorch_hip_version",
    "python_version", "gpu_name",
    # Configuration
    "implementation", "kernel",
    "batch", "num_heads", "num_kv_heads",
    "seq_len", "head_dim", "causal", "dtype",
    "num_warmup", "num_iters",
    # Timing
    "avg_time_ms", "median_time_ms",
    "p25_time_ms", "p75_time_ms",
    "min_time_ms", "max_time_ms",
    "achieved_tflops",
    # Validation
    "validated_against", "max_abs_error",
    "rel_l2_error", "cos_similarity",
    "error_count", "numel", "error_rate", "validation_passed",
    # Notes
    "notes",
]


def load_config(path: str) -> dict:
    """Load a YAML config file into a dict."""
    with open(path) as f:
        return yaml.safe_load(f)


def import_wrapper(impl_name: str):
    """Import the kernel wrapper module by implementation name."""
    module_path = f"kernels.{impl_name}.run"
    return importlib.import_module(module_path)


def append_row(csv_path: str, row: dict) -> None:
    """Append a row to the CSV, writing the header if file is new."""
    path = Path(csv_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    file_exists = path.exists()
    with open(path, "a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=CSV_COLUMNS, extrasaction="ignore")
        if not file_exists:
            writer.writeheader()
        writer.writerow(row)


def run_one(config: dict) -> dict:
    """Execute one benchmark cell. Returns a row dict ready for CSV."""
    impl_name = config["implementation"]
    kernel_name = config["kernel"]
    shape_cfg = config["shape"]
    dtype_str = config["dtype"]
    measurement_cfg = config["measurement"]

    dtype = DTYPE_MAP[dtype_str.lower()]
    num_warmup = int(measurement_cfg["num_warmup"])
    num_iters = int(measurement_cfg["num_iters"])
    validate_against = measurement_cfg.get("validate_against")

    env = capture_env()

    wrapper = import_wrapper(impl_name)
    shape = wrapper.AttentionShape(
        batch=int(shape_cfg["batch"]),
        num_heads=int(shape_cfg["num_heads"]),
        num_kv_heads=int(shape_cfg["num_kv_heads"]),
        seq_len=int(shape_cfg["seq_len"]),
        head_dim=int(shape_cfg["head_dim"]),
        causal=bool(shape_cfg["causal"]),
    )
    fn, q, k, v = wrapper.build_callable(shape, dtype, device="cuda", seed=0)

    validation_result = None
    if validate_against:
        ref_wrapper = import_wrapper(validate_against)
        ref_fn, _, _, _ = ref_wrapper.build_callable(
            shape, dtype, device="cuda", seed=0
        )
        with torch.no_grad():
            ref_out = ref_fn()
            cand_out = fn()
        torch.cuda.synchronize()
        validation_result = validate_outputs(ref_out, cand_out)

    timing = time_kernel(fn, num_warmup=num_warmup, num_iters=num_iters)

    total_flops = attention_flops(
        batch=shape.batch,
        seq_len=shape.seq_len,
        num_heads=shape.num_heads,
        head_dim=shape.head_dim,
        causal=shape.causal,
    )
    achieved = tflops(total_flops, timing.avg_ms)

    row: dict[str, Any] = {}
    row.update(env.as_row())
    row.update({
        "implementation": wrapper.IMPLEMENTATION_NAME,
        "kernel": kernel_name,
        "batch": shape.batch,
        "num_heads": shape.num_heads,
        "num_kv_heads": shape.num_kv_heads,
        "seq_len": shape.seq_len,
        "head_dim": shape.head_dim,
        "causal": shape.causal,
        "dtype": dtype_str,
        "num_warmup": num_warmup,
        "num_iters": num_iters,
        "avg_time_ms": timing.avg_ms,
        "median_time_ms": timing.median_ms,
        "p25_time_ms": timing.p25_ms,
        "p75_time_ms": timing.p75_ms,
        "min_time_ms": timing.min_ms,
        "max_time_ms": timing.max_ms,
        "achieved_tflops": achieved,
        "validated_against": validate_against or "",
        "notes": config.get("notes", ""),
    })
    if validation_result is not None:
        row.update({
            "max_abs_error": validation_result.max_abs_error,
            "rel_l2_error": validation_result.rel_l2_error,
            "cos_similarity": validation_result.cos_similarity,
            "error_count": validation_result.error_count,
            "numel": validation_result.numel,
            "error_rate": validation_result.error_rate,
            "validation_passed": validation_result.passed,
        })
    return row


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Run one or more attention benchmark cells")
    parser.add_argument("--config", required=True, help="Path to YAML config")
    parser.add_argument("--output", required=True, help="Path to output CSV (appended)")
    parser.add_argument("--print-row", action="store_true",
                        help="Also print each row to stdout")
    args = parser.parse_args(argv)

    config = load_config(args.config)

    if "shapes" in config:
        shapes = config["shapes"]
        if not isinstance(shapes, list):
            raise ValueError("'shapes' must be a list of shape dicts")
    elif "shape" in config:
        shapes = [config["shape"]]
    else:
        raise ValueError("Config must have either 'shape' or 'shapes'")

    rows_written = 0
    for i, shape_cfg in enumerate(shapes):
        cell_config = dict(config)
        cell_config["shape"] = shape_cfg
        cell_config.pop("shapes", None)

        try:
            row = run_one(cell_config)
        except Exception as e:
            print(f"[cell {i+1}/{len(shapes)}] FAILED for shape {shape_cfg}: {e}")
            continue

        append_row(args.output, row)
        rows_written += 1

        if args.print_row:
            print(f"--- cell {i+1}/{len(shapes)} ---")
            for k in CSV_COLUMNS:
                print(f"  {k}: {row.get(k, '')}")
        else:
            print(f"[cell {i+1}/{len(shapes)}] {row['implementation']} @ "
                  f"B={row['batch']} N={row['seq_len']} D={row['head_dim']} "
                  f"-> {row['achieved_tflops']:.2f} TFLOPS  "
                  f"(validation={row['validation_passed']})")

    print(f"Wrote {rows_written}/{len(shapes)} rows to {args.output}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
