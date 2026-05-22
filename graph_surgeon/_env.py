"""Runtime environment tweaks (onnxruntime noise suppression on headless/WSL hosts)."""

from __future__ import annotations

import os
import sys
from contextlib import contextmanager
from typing import Any


def configure_runtime_quiet() -> None:
    """Set ORT log env before any onnxruntime import (best-effort)."""
    os.environ.setdefault("ORT_LOGGING_LEVEL", "3")
    os.environ.setdefault("ORT_LOG_LEVEL", "3")


@contextmanager
def silence_stderr():
    """Redirect stderr to devnull for the duration of the context."""
    devnull = os.open(os.devnull, os.O_WRONLY)
    old_stderr = os.dup(2)
    try:
        os.dup2(devnull, 2)
        yield
    finally:
        os.dup2(old_stderr, 2)
        os.close(devnull)
        os.close(old_stderr)


def import_onnxruntime() -> Any:
    """
    Import onnxruntime without printing GPU device-discovery warnings.

    ORT emits warnings at import time that ignore ORT_LOGGING_LEVEL on some builds.
    """
    configure_runtime_quiet()
    with silence_stderr():
        import onnxruntime as ort  # noqa: PLC0415

    try:
        ort.set_default_logger_severity(3)
    except Exception:
        pass
    return ort
