#!/usr/bin/env python3
"""
scripts/ollama_narration_utils.py

Shared utilities extracted from scripts/privacy_explanation_agent.py (Phase D)
during Phase B's Step B2 refactor. Both are pure, domain-agnostic - no
privacy-specific or clinical-specific assumptions - so they're shared as-is
rather than duplicated between privacy_explanation_agent.py and
clinical_narrative_agent.py:

  - call_ollama(): a thin HTTP client for local Ollama generation. Localhost
    only, no external network calls.
  - check_consistency() / _flatten_numbers(): the numeric hallucination
    guard - every number an LLM narrative states must trace back to a
    number literally present in the facts dict it was given.

Neither function computes or knows anything about privacy telemetry, DP
receipts, XAI attribution, or clinical data - they operate on a generic
`facts: dict` and a `narrative: str`, whatever the caller's domain is.
"""

from __future__ import annotations

import re
import time

import requests


def call_ollama(prompt: str, model: str, url: str, timeout: int) -> tuple[str | None, str | None, float]:
    """Returns (narrative, error, elapsed_seconds)."""
    t0 = time.time()
    try:
        requests.get(f"{url}/api/tags", timeout=5)
    except requests.exceptions.RequestException as e:
        return None, f"Ollama unreachable at {url}: {e}", time.time() - t0

    print(f"[ollama] generating narrative via local Ollama (model={model}, "
          f"this can take ~30-90s)...", flush=True)
    try:
        resp = requests.post(
            f"{url}/api/generate",
            json={"model": model, "prompt": prompt, "stream": False},
            timeout=(10, timeout),
        )
    except requests.exceptions.RequestException as e:
        return None, f"Ollama request failed: {e}", time.time() - t0

    elapsed = time.time() - t0
    if resp.status_code != 200:
        return None, f"Ollama returned HTTP {resp.status_code}: {resp.text[:300]}", elapsed

    body = resp.json()
    narrative = body.get("response", "").strip()
    if not narrative:
        return None, f"Ollama returned an empty response: {body}", elapsed
    return narrative, None, elapsed


_NUMBER_RE = re.compile(r"-?\d+\.\d+(?:[eE][-+]?\d+)?|-?\d{2,}")


def _flatten_numbers(obj) -> list[float]:
    out = []
    if isinstance(obj, bool):
        return out
    if isinstance(obj, (int, float)):
        out.append(float(obj))
    elif isinstance(obj, dict):
        for v in obj.values():
            out.extend(_flatten_numbers(v))
    elif isinstance(obj, list):
        for v in obj:
            out.extend(_flatten_numbers(v))
    elif isinstance(obj, str):
        # numeric strings stored as strings (e.g. session/record ids are NOT
        # numeric facts, and timestamps embed numbers we don't want as
        # "facts" to match against) - skip strings entirely, only real JSON
        # numbers count.
        pass
    return out


def check_consistency(narrative: str, facts: dict) -> dict:
    """Hallucination guard: every number the LLM wrote must trace back to a
    number in the facts dict (exact match, or matches after rounding — LLMs
    routinely shorten '5.302585092994046' to '5.3' or '5.30'). Single/double
    digit bare integers (0-99 without a decimal point) are excluded from
    flagging: they are overwhelmingly generic prose ("a single device", "one
    round") rather than fabricated measurements, and would otherwise dominate
    the warning list with noise."""
    fact_numbers = _flatten_numbers(facts)

    candidates = []
    for m in _NUMBER_RE.finditer(narrative):
        try:
            candidates.append(float(m.group()))
        except ValueError:
            continue

    checked = []
    untraceable = []
    for c in candidates:
        if "." not in f"{c}" and abs(c) < 100:
            continue  # generic small integer, not treated as a claimed measurement
        checked.append(c)
        traceable = False
        for f in fact_numbers:
            if c == f:
                traceable = True
                break
            for digits in range(0, 7):
                if round(f, digits) == c:
                    traceable = True
                    break
            if traceable:
                break
            if f != 0 and abs(c - f) / abs(f) < 1e-3:
                traceable = True
                break
        if not traceable:
            untraceable.append(c)

    return {
        "passed": len(untraceable) == 0,
        "num_candidates_extracted": len(candidates),
        "num_checked_as_claims": len(checked),
        "num_fact_numbers_available": len(fact_numbers),
        "untraceable_numbers": untraceable,
    }
