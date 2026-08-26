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

# Step B8 item 3: a file:line source citation embedded in a facts string
# (e.g. "server.rs:1497-1500") - excluded from string-number extraction
# below because these are code locations, not measurements, and several of
# them (92, 104, 528, 542, 629, 660, 1497, 1500 - the exact citations in
# this project's STATIC_SYSTEM_FACTS) would otherwise become "known facts"
# every single Phase D run, purely as a side effect of being quoted in a
# caveat string.
_FILE_LINE_CITATION_RE = re.compile(r"\b[\w./\\-]+\.\w{1,4}:\d+(?:-\d+)?\b")


def _flatten_numbers(obj) -> list[float]:
    """Step B8 item 3: also extracts numbers embedded in string VALUES
    (e.g. "trimmed_mean, trim_ratio=0.1" -> 0.1), not just real JSON number
    fields. Previously a fact stated only inside descriptive prose (like
    that trim_ratio) was invisible to this function and so could never be
    traced against, even when the LLM restated it correctly.

    Risk, stated plainly rather than left implicit: this widens what counts
    as a "known fact" to include any number that happens to appear anywhere
    in any string in the facts dict - including incidental numbers that
    are not measurements at all (source line numbers, hex-ish fragments,
    dates). A fabricated claim that coincidentally matches one of those
    incidental numbers would now be forgiven when it shouldn't be. The
    concrete, non-hypothetical instance of this risk in THIS system is file:
    line citations in STATIC_SYSTEM_FACTS ("server.rs:1497-1500" and
    similar) - without a guard, ~8 source-line integers between 92 and 1500
    would become "traceable" every run purely as a byproduct of being
    quoted in a caveat. _FILE_LINE_CITATION_RE excludes exactly that
    pattern and nothing broader. This does not eliminate the general risk
    (a coincidental match to some OTHER incidental string-embedded number is
    still conceivable), but it removes the specific, guaranteed-to-recur
    case observed in this codebase. Judged an acceptable, narrow widening on
    that basis - see check_consistency()'s docstring for the fabrication
    re-test performed after this change."""
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
        citation_spans = [(m.start(), m.end()) for m in _FILE_LINE_CITATION_RE.finditer(obj)]
        for m in _NUMBER_RE.finditer(obj):
            if any(cs <= m.start() and m.end() <= ce for cs, ce in citation_spans):
                continue  # file:line source citation, not a fact
            try:
                out.append(float(m.group()))
            except ValueError:
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


_CLOCK_RE = re.compile(r"\b\d{1,2}:\d{2}(?::\d{2}(?:\.\d+)?)?\b")
_ISO_DATETIME_RE = re.compile(r"\b\d{4}-\d{2}-\d{2}(?:[T ]\d{2}:\d{2}:\d{2}(?:\.\d+)?Z?)?\b")


def _is_timestamp_fragment(narrative: str, start: int, end: int) -> bool:
    """True if the number is a substring of a recognisable clock-time
    (H:MM, H:MM:SS, H:MM:SS.ffffff) or ISO-8601 datetime
    (YYYY-MM-DD, optionally with a T/space-separated HH:MM:SS.ffffffZ) span
    in the narrative text.

    Local, pattern-based like the identifier rule (Step B4) - NOT "is there
    a colon somewhere nearby". A candidate span must fall entirely INSIDE a
    span the clock/ISO regex actually matched, so e.g. "48" only counts as a
    timestamp fragment when it is literally the MM or SS component of a
    matched "2:48:09.825000"-shaped string, not merely near one.

    Deliberately narrow scope, matching what was asked: this catches clock
    fragments and ISO datetimes, NOT free-form written dates like "August
    26, 2026" - "26" and "2026" in that construction have no colon or ISO
    date-punctuation touching them, so neither regex matches, and they are
    NOT excluded by this function. That is a separate, still-open category
    (see check_consistency()'s docstring).
    """
    window_start = max(0, start - 30)
    window_end = min(len(narrative), end + 30)
    for pattern in (_CLOCK_RE, _ISO_DATETIME_RE):
        for m in pattern.finditer(narrative, window_start, window_end):
            if m.start() <= start and end <= m.end():
                return True
    return False


_MONTH_NAMES = (
    r"January|February|March|April|May|June|July|August|September|October|November|December|"
    r"Jan|Feb|Mar|Apr|Jun|Jul|Aug|Sept|Sep|Oct|Nov|Dec"
)
_WRITTEN_DATE_RE = re.compile(
    rf"\b(?:{_MONTH_NAMES})\.?\s+\d{{1,2}}(?:st|nd|rd|th)?,?\s*\d{{4}}\b"    # "August 26, 2026" / "Aug 26th 2026"
    rf"|\b\d{{1,2}}(?:st|nd|rd|th)?\s+(?:{_MONTH_NAMES})\.?,?\s*\d{{4}}\b"   # "26 August 2026"
    rf"|\b(?:{_MONTH_NAMES})\.?\s+\d{{4}}\b",                                # "August 2026" (no day)
    re.IGNORECASE,
)


def _is_written_date_fragment(narrative: str, start: int, end: int) -> bool:
    """True if the number is a substring of a recognisable written date
    ("August 26, 2026", "26 August 2026", "August 2026") - anchored to an
    actual month name (full or standard abbreviation) directly adjacent to
    the day/year numbers, not "any number near any capitalised word".

    Same containment approach as the identifier/timestamp rules: the number
    must fall entirely inside a span _WRITTEN_DATE_RE actually matched. "In
    May, 26 patients were enrolled" does NOT match - the comma right after
    the month breaks the required "month <whitespace> day" adjacency, so
    "26" there is correctly left as an ordinary number to check, not
    excluded as a date fragment.
    """
    window_start = max(0, start - 30)
    window_end = min(len(narrative), end + 30)
    for m in _WRITTEN_DATE_RE.finditer(narrative, window_start, window_end):
        if m.start() <= start and end <= m.end():
            return True
    return False


def _parse_number_match(narrative: str, m: re.Match) -> tuple[float, int, int]:
    """Step B8 item 2: corrects _NUMBER_RE's bare-int alternative allowing an
    optional leading '-', which misreads a numeric-RANGE hyphen (e.g. the
    "1497-1500" in a file:line citation "server.rs:1497-1500") as a unary
    minus, producing a spurious -1500.

    Detection: the character immediately before the matched '-' is itself a
    digit. No genuine negative-number usage in English prose produces this -
    a real negative is always preceded by whitespace, punctuation, "is/of/
    was", an opening parenthesis, or the start of the string (e.g. "value:
    -3", "temperature is -3", "(-3)", "-3 degrees"); nothing in ordinary
    prose writes a digit immediately followed by a minus sign immediately
    followed by another digit except a range. Returns the corrected
    (value, start, end) - start is advanced past the stripped '-' so
    downstream adjacency checks (identifier/timestamp/date) see the number's
    true span, not the discarded sign character.
    """
    text = m.group()
    start, end = m.start(), m.end()
    if text.startswith("-") and start > 0 and narrative[start - 1].isdigit():
        text = text[1:]
        start += 1
    return float(text), start, end


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

    Step B6 addition, same treatment: a number that is a substring of a
    recognisable clock-time or ISO-8601 datetime span in the narrative text
    (e.g. the "48" and "09.825000" inside "2:48:09.825000") is excluded -
    counted in num_excluded_as_timestamp. Free-form written dates ("August
    26, 2026") are NOT covered by this - see _is_timestamp_fragment()'s
    docstring - that is a separate, still-open false-positive category,
    reported but deliberately not addressed here (Step B6 scoped this
    narrowly to clock/ISO patterns).

    Step B8 additions:
      - Written dates ("August 26, 2026") are excluded, same local
        containment approach as timestamps - counted in
        num_excluded_as_written_date. "In May, 26 patients..." is NOT
        excluded (no month-day adjacency) - see _is_written_date_fragment().
      - The range-hyphen misparse (a file:line citation's "1497-1500"
        producing a spurious -1500) is corrected at match level - see
        _parse_number_match(). Verified against a genuine negative-number
        narrative (see the Step B8 verification report) to confirm real
        negatives still parse correctly.
      - _flatten_numbers() now also extracts numbers from string values
        (e.g. "trim_ratio=0.1"), with a narrow file:line-citation exclusion
        - see _flatten_numbers()'s own docstring for the risk this
        introduces and why it's judged acceptable.
      - The small-bare-integer skip below now checks the RAW MATCHED TEXT
        (m.group(), pre-sign-correction) for a decimal point, not the
        stringified float - the previous version was a no-op (a Python
        float always stringifies with "."), so this exclusion has never
        actually fired before Step B8. See the Step B8 verification report
        for how many numbers this newly excludes in real narratives.

    None of these loosen what counts as a genuine fabrication: a number with
    no percentage context, no identifier context, no timestamp context, and
    no written-date context still must match a fact number (exact, rounded,
    or within 0.1% relative tolerance) or it is flagged, exactly as before.
    Re-verified after every addition in this file via a deliberate
    fabrication test (fabricated F1 + fabricated percentage) - see the Step
    B4/B6/B8 verification reports.

    Known, separately-reported, NOT fixed here: large integers the LLM
    reformats with a misplaced thousands separator (e.g. 1506992 written as
    "150,6992") are a real generation defect, not a false positive - see
    docs/IMPLEMENTATION_NOTES.md "Step B6" for the investigation and the
    recommended FACTS-level fix (implemented separately in
    privacy_explanation_agent.py's gather_facts(), not here).
    """
    fact_numbers = _flatten_numbers(facts)

    checked = []
    untraceable = []
    num_traceable_as_percentage = 0
    num_excluded_as_identifier_substring = 0
    num_excluded_as_timestamp = 0
    num_excluded_as_written_date = 0
    num_excluded_as_small_integer = 0
    num_candidates = 0

    for raw_m in _NUMBER_RE.finditer(narrative):
        num_candidates += 1
        try:
            c, start, end = _parse_number_match(narrative, raw_m)
        except ValueError:
            continue
        raw_text = narrative[start:end]

        if "." not in raw_text and abs(c) < 100:
            num_excluded_as_small_integer += 1
            continue  # generic small integer, not treated as a claimed measurement

        if _is_timestamp_fragment(narrative, start, end):
            num_excluded_as_timestamp += 1
            continue

        if _is_written_date_fragment(narrative, start, end):
            num_excluded_as_written_date += 1
            continue

        if _is_identifier_substring(narrative, start, end):
            num_excluded_as_identifier_substring += 1
            continue

        checked.append(c)
        traceable = _traceable_to_facts(c, fact_numbers)
        if not traceable and _is_percentage_context(narrative, end):
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
        "num_excluded_as_written_date": num_excluded_as_written_date,
        "num_excluded_as_small_integer": num_excluded_as_small_integer,
        "num_excluded_as_identifier_substring": num_excluded_as_identifier_substring,
        "num_excluded_as_timestamp": num_excluded_as_timestamp,
        "untraceable_numbers": untraceable,
    }
