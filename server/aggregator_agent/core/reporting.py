# core/reporting.py (server-side copy — mirrors installer/runtime/core/reporting.py)
"""
Structured terminal observability helper for the federated pipeline.

This module is purely additive: it formats and prints values that the
pipeline stages already compute. It never performs ML/FL/DP/crypto logic,
never changes control flow, and never invents values — every call site
passes in a number/string that already exists in the calling code.

IMPORTANT (server side): aggregator.py's stdout is a machine-readable
contract — the Rust orchestrator parses it as a single JSON line via
serde_json::from_slice(). aggregator.py calls set_default_stream(sys.stderr)
at import time so every call in this module defaults to stderr, leaving
stdout untouched. Rust already captures and logs subprocess stderr on
failure, so nothing here is silently lost.
"""

import sys
import time
from datetime import datetime, timezone

_WIDTH = 64

_default_stream = sys.stdout


def set_default_stream(stream):
    global _default_stream
    _default_stream = stream


def _resolve(file):
    return _default_stream if file is None else file


def _ts() -> str:
    return datetime.now(timezone.utc).strftime("%H:%M:%S")


def header(title: str, file=None):
    file = _resolve(file)
    line = "=" * _WIDTH
    print(line, file=file)
    print(title.center(_WIDTH), file=file)
    print(line, file=file)


def subheader(title: str, file=None):
    file = _resolve(file)
    line = "-" * _WIDTH
    print(line, file=file)
    print(title, file=file)
    print(line, file=file)


def kv(label: str, value, indent: int = 0, width: int = 22, file=None):
    file = _resolve(file)
    pad = " " * indent
    print(f"{pad}{label:<{width}}: {value}", file=file)


def line(msg: str = "", file=None):
    print(msg, file=_resolve(file))


def step(msg: str, file=None):
    print(f"[{_ts()}] {msg}", file=_resolve(file))


def ok(msg: str, file=None):
    print(f"[OK] {msg}", file=_resolve(file))


def warn(msg: str, file=None):
    print(f"[WARN] {msg}", file=_resolve(file))


def fail(msg: str, file=None):
    print(f"[FAIL] {msg}", file=_resolve(file))


def rule(file=None):
    print("-" * _WIDTH, file=_resolve(file))


def tensor_summary(name: str, t, indent: int = 4, sample_n: int = 5, file=None):
    """
    Print shape/dtype/count/norm/min/max/mean/std + a small deterministic
    sample for a tensor — never the full tensor.
    """
    file = _resolve(file)
    pad = " " * indent
    try:
        flat = t.detach().float().flatten()
        n = flat.numel()
        if n == 0:
            print(f"{pad}{name}: shape={tuple(t.shape)} dtype={t.dtype} numel=0 (empty)", file=file)
            return
        sample = [round(float(x), 4) for x in flat[:sample_n].tolist()]
        print(
            f"{pad}{name}: shape={tuple(t.shape)} dtype={t.dtype} numel={n} "
            f"norm={flat.norm().item():.6f} min={flat.min().item():.6f} "
            f"max={flat.max().item():.6f} mean={flat.mean().item():.6f} "
            f"std={flat.std().item():.6f} sample={sample}",
            file=file,
        )
    except Exception as e:
        print(f"{pad}{name}: <summary unavailable: {e}>", file=file)


class Timer:
    """Wall-clock timer for reporting stage durations. Read-only side effect (time.time())."""

    def __init__(self):
        self.t0 = time.time()

    def elapsed(self) -> float:
        return time.time() - self.t0
