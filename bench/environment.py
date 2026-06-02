"""Capture per-session environment metadata for benchmark provenance.

Every CSV row stamps these fields so we can group/filter by environment
during analysis (e.g., separate measurements taken under different
PyTorch builds or ROCm versions).
"""

from __future__ import annotations

import os
import platform
import socket
import subprocess
from dataclasses import dataclass, asdict
from datetime import datetime, timezone
from typing import Optional


@dataclass
class EnvironmentSnapshot:
    """Snapshot of session/environment metadata.

    All fields are JSON/CSV-safe strings or ints.
    """
    timestamp_utc: str
    hostname: str
    session_id: str
    rocm_version: str
    hip_version: str
    pytorch_version: str
    pytorch_hip_version: str
    python_version: str
    gpu_name: str

    def as_row(self) -> dict:
        """Return a flat dict suitable for adding to a CSV row."""
        return asdict(self)


def _read_file(path: str) -> str:
    """Return the first line of a file, or empty string if missing."""
    try:
        with open(path) as f:
            return f.read().strip().splitlines()[0]
    except (FileNotFoundError, IndexError):
        return ""


def _run(cmd: list[str]) -> str:
    """Run a command and return its stdout stripped, or empty on failure."""
    try:
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=5,
        )
        return result.stdout.strip()
    except (FileNotFoundError, subprocess.TimeoutExpired):
        return ""


def _detect_session_id() -> str:
    """Best-effort detection of session id from the container hostname."""
    host = socket.gethostname()
    return host


def _detect_gpu_name() -> str:
    """Detect GPU model. Falls back to gracefully empty if not available."""
    try:
        import torch
        if torch.cuda.is_available():
            return torch.cuda.get_device_name(0)
    except Exception:
        pass
    return ""


def _detect_pytorch() -> tuple[str, str]:
    """Return (pytorch_version, pytorch_hip_version) or empty strings."""
    try:
        import torch
        version = torch.__version__
        hip = getattr(torch.version, "hip", None) or ""
        return version, hip
    except ImportError:
        return "", ""


def capture() -> EnvironmentSnapshot:
    """Capture environment metadata for the current process."""
    pt_version, pt_hip = _detect_pytorch()

    return EnvironmentSnapshot(
        timestamp_utc=datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        hostname=socket.gethostname(),
        session_id=_detect_session_id(),
        rocm_version=_read_file("/opt/rocm/.info/version"),
        hip_version=_run(["hipconfig", "--version"]),
        pytorch_version=pt_version,
        pytorch_hip_version=pt_hip,
        python_version=platform.python_version(),
        gpu_name=_detect_gpu_name(),
    )
