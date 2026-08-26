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


def _traceable_to_facts(value: float, fact_numbers: list[float]) -> bool:
    for f in fact_numbers:
        if value == f:
            return True
        for digits in range(0, 7):
            if round(f, digits) == value:
                return True
        if f != 0 and abs(value - f) / abs(f) < 1e-3:
            return True
    return False


_PERCENT_CONTEXT_RE = re.compile(r"\s*(%|percent(age)?\b)", re.IGNORECASE)


def _is_percentage_context(narrative: str, end_pos: int) -> bool:
    """True if a '%' or the word 'percent'/'percentage' immediately follows
    the matched number (skipping intervening whitespace) - i.e. the
    narrative is presenting the number AS a percentage, not just a decimal
    that happens to be close to one."""
    tail = narrative[end_pos:end_pos + 20]
    return bool(_PERCENT_CONTEXT_RE.match(tail))


def _is_identifier_substring(narrative: str, start: int, end: int) -> bool:
    """True if the number sits inside a longer alphanumeric token (a
    session/device ID, hash, etc.) rather than standing on its own as a
    claimed measurement.

    Boundary rule (Step B4), applied to the single character immediately
    before `start` and immediately after `end` in the raw narrative text -
    deliberately local, not "does the whole containing word have a letter
    in it anywhere" (that would also suppress real hyphenated number
    references like "round-1"):

      1. If that adjacent character is a letter (a-z/A-Z) - exclude. This is
         the direct case: hex-ish tokens like "client-aef1978aef7f" or
         "b487b6aaef2c9e48" mix letters and digits with no separator, so a
         digit run's immediate neighbour is a letter.
      2. If that adjacent character is a hyphen, look one character further
         in the same direction. If THAT character is alphanumeric (not
         whitespace, not start/end of string, not other punctuation) then
         the hyphen is itself "within a token" - e.g. "client-1978" - and we
         exclude too. If it's whitespace or another non-alphanumeric
         character (e.g. a sentence dash "— 5.3", a list bullet "- 5.3"),
         the hyphen is ordinary punctuation, not part of an identifier, and
         the number is NOT excluded on this basis.

    Known, accepted limitation of rule 2: a fabricated decimal directly
    hyphen-joined to a word (e.g. "round-42.5", no space) would also be
    excluded, since "word-number" and "identifier-number" are locally
    indistinguishable by adjacent characters alone. This is judged
    acceptable because (a) that is not a natural English construction none
    of this system's real generated narratives have produced it, and (b)
    small bare integers directly after a hyphen (e.g. "round-1") are
    already filtered by the pre-existing small-integer rule regardless.
    """
    def _touches_letter_or_token_hyphen(idx: int, step: int) -> bool:
        if idx < 0 or idx >= len(narrative):
            return False
        ch = narrative[idx]
        if ch.isalpha():
            return True
        if ch == "-":
            other = idx + step
            if 0 <= other < len(narrative) and narrative[other].isalnum():
                return True
        return False

    before_ok = _touches_letter_or_token_hyphen(start - 1, -1)
    after_ok = _touches_letter_or_token_hyphen(end, 1)
    return before_ok or after_ok


def check_consistency(narrative: str, facts: dict) -> dict:
    """Hallucination guard: every number the LLM wrote must trace back to a
    number in the facts dict (exact match, or matches after rounding — LLMs
    routinely shorten '5.302585092994046' to '5.3' or '5.30'). Single/double
    digit bare integers (0-99 without a decimal point) are excluded from
    flagging: they are overwhelmingly generic prose ("a single device", "one
    round") rather than fabricated measurements, and would otherwise dominate
    the warning list with noise.

    Step B4 additions, both narrow and both logged so they're visible rather
    than silent:
      - A number presented with a '%'/"percent" immediately after it is ALSO
        checked against fact_numbers scaled by 100 (i.e. the LLM restating
        0.5676 as "56.76%") - counted in num_traceable_as_percentage.
      - A number sitting inside a longer alphanumeric token (session ID,
        device ID, hash) is excluded from checking entirely, not just
        forgiven - it was never a claimed measurement - counted in
        num_excluded_as_identifier_substring.
    Neither loosens what counts as a genuine fabrication: a number with no
    percentage context and no identifier context still must match a fact
    number (exact, rounded, or within 0.1% relative tolerance) or it is
    flagged, exactly as before.
    """
    fact_numbers = _flatten_numbers(facts)

    checked = []
    untraceable = []
    num_traceable_as_percentage = 0
    num_excluded_as_identifier_substring = 0
    num_candidates = 0

    for m in _NUMBER_RE.finditer(narrative):
        try:
            c = float(m.group())
        except ValueError:
            continue
        num_candidates += 1

        if "." not in f"{c}" and abs(c) < 100:
            continue  # generic small integer, not treated as a claimed measurement

        if _is_identifier_substring(narrative, m.start(), m.end()):
            num_excluded_as_identifier_substring += 1
            continue

        checked.append(c)
        traceable = _traceable_to_facts(c, fact_numbers)
        if not traceable and _is_percentage_context(narrative, m.end()):
            if _traceable_to_facts(c / 100.0, fact_numbers):
                traceable = True
                num_traceable_as_percentage += 1
        if not traceable:
            untraceable.append(c)

    return {
        "passed": len(untraceable) == 0,
        "num_candidates_extracted": num_candidates,
        "num_checked_as_claims": len(checked),
        "num_fact_numbers_available": len(fact_numbers),
        "num_traceable_as_percentage": num_traceable_as_percentage,
        "num_excluded_as_identifier_substring": num_excluded_as_identifier_substring,
        "untraceable_numbers": untraceable,
    }
