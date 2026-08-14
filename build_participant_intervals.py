#!/usr/bin/env python3
"""
build_participant_intervals.py  (Phase 16 - Stage S1: participant speech index)

Builds the participant-only speech interval index that Stage S2 will use to gate
COVAREP/FORMANT frames down to the participant's own voice.

WHY THIS STAGE EXISTS
---------------------
The AVEC-2017 documentation states the audio "might contain small amounts of
bleed-over of virtual interviewer; use transcript files to alleviate this issue
when processing". Ellie's synthesized speech is a contaminant, not signal, and
she holds the floor for roughly three quarters of a session. Gating the audio to
participant-only intervals therefore requires per-turn timestamps.

WHY ./data AND NOT ./data_norm
------------------------------
Only the RAW staged transcripts under ./data carry timestamps:

    ./data/<pid>_P/<pid>_TRANSCRIPT.csv       start_time, stop_time, speaker, value   (TAB)
    ./data_norm/<pid>_P/<pid>_TRANSCRIPT.csv  speaker, value                          (COMMA)

normalize_transcripts.py deliberately drops start_time/stop_time when producing
the text-only training corpus. Reading data_norm/ here would silently yield an
index with no timestamps at all.

SCOPE BOUNDARY - THIS SCRIPT DOES NOT CLAMP TO THE AUDIO STREAM
---------------------------------------------------------------
The intervals produced here are pure transcript geometry. They are NOT yet
guaranteed to lie inside the COVAREP stream: Stage S0 measured that clamping to
the audio span removes 0.020 s of participant speech across the entire corpus
(all of it from participant 381). Clamping needs the per-participant COVAREP row
count, which lives in the archives, so it belongs to S2. Consumers must clamp.
Run with --audit to get an advisory report of who would be clipped; it annotates,
it never modifies the intervals.

MERGE SEMANTICS
---------------
Intervals are merged when they overlap OR touch (next.start <= current.end),
because both cases map to the same set of COVAREP frames and leaving them split
would double-count the shared frames in the duration total.

Inputs (read-only):
    <data-dir>/<pid>_P/<pid>_TRANSCRIPT.csv     staged transcripts (TAB-delimited)
    <audit>                                     optional S0 report, for cross-checks

Output:
    <out>          JSON interval index (default ./dataset_build/participant_intervals.json)
    stdout         human-readable log
    exit code      0 = all checks passed, 1 = validation failures, 2 = fatal setup error

Usage:
    python build_participant_intervals.py
    python build_participant_intervals.py --pids 300,402 --verbose
    python build_participant_intervals.py --no-audit-check

NOTE: run with the project venv (.venv/Scripts/python.exe); the system
interpreter has no pandas.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import statistics
import sys
import time
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

try:
    import tqdm as _tqdm_mod
    from tqdm import tqdm as _tqdm
except ImportError:  # pragma: no cover - environment dependent
    _tqdm_mod = None
    _tqdm = None

try:
    import pandas as pd
except ImportError:  # pragma: no cover - environment dependent
    pd = None


# --------------------------------------------------------------------------
# Constants. Values marked "measured" were established by Stage S0 over all 188
# readable archives; they are assertions here, not guesses.
# --------------------------------------------------------------------------
DEFAULT_DATA_DIR = Path("./data")
DEFAULT_OUT = Path("./dataset_build/participant_intervals.json")
DEFAULT_AUDIT = Path("./dataset_build/archive_audit.json")

TRANSCRIPT_COLUMNS = ["start_time", "stop_time", "speaker", "value"]
PARTICIPANT_LABEL = "participant"
KNOWN_SPEAKERS = {"ellie", PARTICIPANT_LABEL}

EXPECTED_PARTICIPANTS = 188        # measured: the frozen corpus size
EXPECTED_PARTICIPANT_TURNS = 32102  # measured: total participant turns
MIN_SPEECH_SECONDS = 1.0           # a participant below this is unusable
# Measured floor is 62.23 s (pid 385). Anything under 30 s would be so far
# outside the observed distribution that it signals a parsing fault, not a
# quiet interviewee.
LOW_SPEECH_WARN_S = 30.0

# Timestamps in the source carry 3 decimals. 6 decimals preserves them exactly
# while removing float-accumulation noise from summed durations, so two runs on
# two machines serialise byte-identically.
ROUND_DP = 6
EPS = 1e-9  # tolerance for float comparisons in the validators

SEV_FAIL = "FAIL"
SEV_WARN = "WARN"
SEV_INFO = "INFO"


# --------------------------------------------------------------------------
# Reporting (same idiom as audit_daic_archives.py: ASCII only - the console
# here is cp1252 and non-ASCII output raises UnicodeEncodeError mid-run)
# --------------------------------------------------------------------------
class Reporter:
    def __init__(self, verbose: bool = False) -> None:
        self.findings: List[Dict[str, Any]] = []
        self.verbose = verbose

    def add(self, severity: str, check: str, message: str,
            pid: Optional[str] = None) -> None:
        self.findings.append({"severity": severity, "check": check,
                              "pid": pid, "message": message})
        if severity == SEV_FAIL or (severity == SEV_WARN and self.verbose):
            where = f" [{pid}]" if pid else ""
            print(f"[{severity}]{where} {check}: {message}")

    def fail(self, check: str, message: str, pid: Optional[str] = None) -> None:
        self.add(SEV_FAIL, check, message, pid)

    def warn(self, check: str, message: str, pid: Optional[str] = None) -> None:
        self.add(SEV_WARN, check, message, pid)

    def info(self, check: str, message: str, pid: Optional[str] = None) -> None:
        self.add(SEV_INFO, check, message, pid)

    def count(self, severity: str) -> int:
        return sum(1 for f in self.findings if f["severity"] == severity)

    def digest(self) -> str:
        """Digest of findings only - excludes timings and absolute paths so the
        same dataset yields the same digest on any machine."""
        canonical = sorted((f["severity"], f["check"], f["pid"] or "", f["message"])
                           for f in self.findings)
        return hashlib.sha256(
            json.dumps(canonical, sort_keys=True, ensure_ascii=True).encode("utf-8")
        ).hexdigest()


def progress(items: Iterable, total: int, desc: str, enabled: bool) -> Iterable:
    if enabled and _tqdm is not None:
        # ascii=True: cp1252 cannot render tqdm's default block glyphs.
        return _tqdm(items, total=total, desc=desc, unit="pid", ascii=True, ncols=78)
    return items


# --------------------------------------------------------------------------
# Core interval construction
# --------------------------------------------------------------------------
def merge_intervals(raw: List[Tuple[float, float]]) -> List[List[float]]:
    """Merge overlapping or touching intervals.

    Input need not be sorted; the sort here is what makes the output canonical
    and therefore deterministic. Merging on 'start <= current_end' folds both
    genuine overlaps (speech-overlap markers in the transcript) and exactly
    touching turns, since both resolve to the same COVAREP frame set.
    """
    merged: List[List[float]] = []
    for s, e in sorted(raw):
        if merged and s <= merged[-1][1]:
            merged[-1][1] = max(merged[-1][1], e)
        else:
            merged.append([s, e])
    return merged


def validate_intervals(pid: str, intervals: List[List[float]],
                       rep: Reporter) -> bool:
    """Post-conditions on the emitted intervals. Runs on the ROUNDED values.

    Rounding is monotonic, so it cannot turn a disjoint pair into an overlapping
    one - but it can turn a tiny gap into a touch, so the ordering check permits
    touching (start >= previous end) while forbidding overlap.
    """
    ok = True
    prev_end: Optional[float] = None
    for idx, (s, e) in enumerate(intervals):
        if not (math.isfinite(s) and math.isfinite(e)):
            rep.fail("V4.nonfinite", f"interval {idx} has non-finite bound", pid)
            ok = False
            continue
        if e < s - EPS:
            rep.fail("V5.negative", f"interval {idx} ends before it starts", pid)
            ok = False
        if s < -EPS:
            rep.fail("V6.negative_time", f"interval {idx} starts at {s}", pid)
            ok = False
        if prev_end is not None:
            if s < prev_end - EPS:
                rep.fail("V3.overlap",
                         f"interval {idx} starts {s} before previous end {prev_end}",
                         pid)
                ok = False
            elif s < prev_end:
                # inside EPS: ordering holds, flagged for visibility only
                rep.warn("V3.touch", f"interval {idx} touches previous within EPS", pid)
        prev_end = e
    return ok


def build_one(pid: str, tpath: Path, rep: Reporter) -> Optional[Dict[str, Any]]:
    """Parse one transcript and produce its interval record.

    Returns None on any condition that makes the participant unusable; the
    caller records the failure and continues, so one bad transcript never aborts
    the run.
    """
    try:
        df = pd.read_csv(tpath, sep="\t")
    except Exception as exc:  # noqa: BLE001 - report and continue
        rep.fail("V1.parse", f"unreadable: {type(exc).__name__}: {exc}", pid)
        return None

    cols = [str(c).strip() for c in df.columns]
    if cols != TRANSCRIPT_COLUMNS:
        # A comma-parsed tab file collapses to ONE column whose name contains
        # all four field names - the exact failure normalize_transcripts.py was
        # written to document. Fail loudly rather than produce empty intervals.
        rep.fail("V1.columns", f"columns {cols} != {TRANSCRIPT_COLUMNS}", pid)
        return None
    df.columns = cols

    if df.empty:
        rep.fail("V1.empty", "transcript has no rows", pid)
        return None

    # -- timestamps: numeric, present and finite on EVERY row -----------------
    # Checked across all speakers, because transcript duration (the coverage
    # denominator) is taken over the whole file, not just participant turns.
    times = df[["start_time", "stop_time"]].apply(pd.to_numeric, errors="coerce")
    bad = int(times.isna().any(axis=1).sum())
    if bad:
        rep.fail("V2.timestamps", f"{bad} row(s) with missing/non-numeric timestamps", pid)
        return None
    if not bool(times.map(math.isfinite).all().all()):
        rep.fail("V2.nonfinite", "non-finite timestamp present", pid)
        return None

    df["start_time"] = times["start_time"]
    df["stop_time"] = times["stop_time"]

    # -- speaker vocabulary ---------------------------------------------------
    speaker = df["speaker"].astype(str).str.strip().str.lower()
    unknown = sorted(set(speaker) - KNOWN_SPEAKERS)
    if unknown:
        # Not fatal: an unknown label is simply not the participant, so it is
        # excluded from gating. Recorded because it would mean the corpus
        # changed shape since S0 verified the vocabulary.
        rep.warn("V7.speaker", f"unknown speaker label(s): {unknown}", pid)

    part = df[speaker == PARTICIPANT_LABEL]
    if part.empty:
        rep.fail("V8.no_participant",
                 "no Participant turns - audio gating would select nothing", pid)
        return None

    # -- per-turn sanity BEFORE merging --------------------------------------
    durations = part["stop_time"] - part["start_time"]
    n_negative = int((durations < -EPS).sum())
    if n_negative:
        rep.fail("V5.turn_negative",
                 f"{n_negative} participant turn(s) end before they start", pid)
        return None
    n_zero = int((durations.abs() <= EPS).sum())
    if n_zero:
        rep.warn("V5.turn_zero", f"{n_zero} zero-length participant turn(s)", pid)

    raw_pairs = list(zip(part["start_time"].tolist(), part["stop_time"].tolist()))
    raw_speech = float(durations.sum())

    merged = merge_intervals(raw_pairs)
    merged_speech = sum(e - s for s, e in merged)

    # Rounded LAST, so all arithmetic happens at full precision and only the
    # serialised representation is quantised.
    intervals = [[round(s, ROUND_DP), round(e, ROUND_DP)] for s, e in merged]
    if not validate_intervals(pid, intervals, rep):
        return None

    t_min_start = float(df["start_time"].min())
    t_max_stop = float(df["stop_time"].max())
    if t_max_stop <= 0:
        rep.fail("V2.duration", f"transcript max stop_time is {t_max_stop}", pid)
        return None

    # Coverage denominator is max(stop_time), NOT (max - min): COVAREP is
    # indexed from t=0, so the window S2 will sample is [0, max_stop]. Using the
    # span would understate the fraction of the audio stream that is silent or
    # Ellie, which is exactly what this ratio is meant to expose.
    coverage = merged_speech / t_max_stop

    if merged_speech < MIN_SPEECH_SECONDS:
        rep.fail("V9.speech", f"only {merged_speech:.3f}s of participant speech", pid)
        return None
    if merged_speech < LOW_SPEECH_WARN_S:
        rep.warn("V9.low_speech",
                 f"{merged_speech:.2f}s of speech is far below the measured "
                 f"corpus floor of 62.23s", pid)

    # Merging must never invent or destroy time: the merged total can only be
    # less than or equal to the raw total, and the difference is exactly the
    # double-counted overlap.
    overlap = raw_speech - merged_speech
    if overlap < -EPS:
        rep.fail("V10.merge", "merged duration exceeds raw duration", pid)
        return None
    if merged_speech > t_max_stop + EPS:
        rep.fail("V10.bounds",
                 f"speech {merged_speech:.3f}s exceeds transcript duration "
                 f"{t_max_stop:.3f}s", pid)
        return None

    return {
        "participant_id": pid,
        "intervals": intervals,
        "n_intervals": len(intervals),
        "n_participant_turns": int(len(part)),
        "participant_speech_seconds": round(merged_speech, ROUND_DP),
        "raw_speech_seconds": round(raw_speech, ROUND_DP),
        "merged_overlap_seconds": round(overlap, ROUND_DP),
        "transcript_min_start": round(t_min_start, ROUND_DP),
        "transcript_max_stop": round(t_max_stop, ROUND_DP),
        "transcript_duration_seconds": round(t_max_stop, ROUND_DP),
        "participant_coverage_ratio": round(coverage, ROUND_DP),
        "participant_max_stop": round(float(part["stop_time"].max()), ROUND_DP),
        "n_transcript_rows": int(len(df)),
        "has_ellie": bool("ellie" in set(speaker)),
    }


# --------------------------------------------------------------------------
# Cross-checks against the Stage S0 audit report
# --------------------------------------------------------------------------
def audit_cross_check(records: Dict[str, Dict[str, Any]], audit_path: Path,
                      rep: Reporter) -> Dict[str, Any]:
    """Compare against S0 and preview which participants S2 will have to clip.

    This never alters an interval. It exists so that a mismatch between the two
    stages surfaces here rather than as a confusing feature-extraction bug later.
    """
    out: Dict[str, Any] = {"checked": False}
    try:
        with open(audit_path, "r", encoding="utf-8") as fh:
            audit = json.load(fh)
    except (OSError, json.JSONDecodeError) as exc:
        rep.warn("X1.audit", f"cannot read audit report: {type(exc).__name__}: {exc}")
        return out

    per = audit.get("per_participant", {})
    audit_pids = set(per)
    mine = set(records)
    missing = sorted(audit_pids - mine, key=int)
    extra = sorted(mine - audit_pids, key=int)
    if missing:
        rep.fail("X1.participants", f"in S0 audit but not indexed here: {missing}")
    if extra:
        rep.fail("X1.participants", f"indexed here but absent from S0 audit: {extra}")

    # Speech durations must agree with S0's independent computation.
    disagree: List[str] = []
    clip_preview: Dict[str, float] = {}
    for pid, rec in records.items():
        entry = per.get(pid) or {}
        tr = entry.get("transcript") or {}
        deep = entry.get("deep") or {}
        s0_speech = tr.get("speech_seconds")
        if s0_speech is not None and abs(s0_speech - rec["participant_speech_seconds"]) > 1e-3:
            disagree.append(pid)
        cov_s = deep.get("covarep_seconds")
        if cov_s is not None:
            clipped = sum(e - max(s, cov_s) for s, e in rec["intervals"] if e > cov_s)
            if clipped > 0:
                clip_preview[pid] = round(clipped, ROUND_DP)
    if disagree:
        rep.fail("X2.speech_mismatch",
                 f"speech duration disagrees with S0 for: {disagree}")
    for pid, secs in sorted(clip_preview.items(), key=lambda kv: int(kv[0])):
        rep.warn("X3.clip_preview",
                 f"S2 clamping to the audio stream will discard {secs:.3f}s "
                 f"of participant speech", pid)

    out.update({
        "checked": True,
        "audit_digest": audit.get("findings_digest"),
        "audit_participants": len(audit_pids),
        "speech_mismatches": disagree,
        "clip_preview_seconds": clip_preview,
        "clip_preview_total_seconds": round(sum(clip_preview.values()), ROUND_DP),
    })
    return out


# --------------------------------------------------------------------------
# Helpers
# --------------------------------------------------------------------------
def describe(values: List[float]) -> Dict[str, float]:
    if not values:
        return {}
    return {
        "min": round(min(values), 4),
        "median": round(statistics.median(values), 4),
        "mean": round(statistics.fmean(values), 4),
        "max": round(max(values), 4),
        "n": len(values),
    }


def find_transcript(folder: Path) -> Optional[Path]:
    """Case-insensitive lookup, matching build_daic_records.find_transcript_csv.

    Windows and Linux disagree on TRANSCRIPT.csv vs Transcript.csv; matching on
    a lowercased suffix keeps this stage portable to Colab.
    """
    if not folder.is_dir():
        return None
    for child in sorted(folder.iterdir()):
        if child.is_file() and child.name.lower().endswith("_transcript.csv"):
            return child
    return None


def environment() -> Dict[str, Any]:
    return {
        "python": sys.version.split()[0],
        "platform": sys.platform,
        "pandas": getattr(pd, "__version__", None) if pd is not None else None,
        "tqdm": getattr(_tqdm_mod, "__version__", None) if _tqdm_mod else None,
    }


# --------------------------------------------------------------------------
# Main
# --------------------------------------------------------------------------
def main() -> int:
    ap = argparse.ArgumentParser(
        description="Build the participant-only speech interval index (Stage S1)")
    ap.add_argument("--data-dir", type=Path, default=DEFAULT_DATA_DIR,
                    help=f"staged RAW transcripts, tab-delimited (default: {DEFAULT_DATA_DIR})")
    ap.add_argument("--out", type=Path, default=DEFAULT_OUT,
                    help=f"output JSON index (default: {DEFAULT_OUT})")
    ap.add_argument("--audit", type=Path, default=DEFAULT_AUDIT,
                    help=f"Stage S0 report for cross-checks (default: {DEFAULT_AUDIT})")
    ap.add_argument("--no-audit-check", action="store_true",
                    help="skip the S0 cross-check")
    ap.add_argument("--pids", default="",
                    help="comma-separated participant IDs (targeted rebuilds)")
    ap.add_argument("--limit", type=int, default=0,
                    help="only the first N participants (0 = all); smoke tests")
    ap.add_argument("--no-progress", action="store_true", help="disable progress bar")
    ap.add_argument("--verbose", action="store_true", help="print WARN findings too")
    args = ap.parse_args()

    t0 = time.time()
    rep = Reporter(verbose=args.verbose)

    # ---- fatal setup checks (exit 2) --------------------------------------
    if pd is None:
        print("[FATAL] pandas is required. Run with .venv/Scripts/python.exe")
        return 2
    if not args.data_dir.is_dir():
        print(f"[FATAL] staged transcript root not found: {args.data_dir}")
        return 2

    data_dir = args.data_dir.resolve()
    out_path = args.out.resolve()
    # The transcripts are an input of this stage AND of the frozen text pipeline.
    # Writing the index underneath them would make this stage non-read-only with
    # respect to its own source tree.
    if data_dir == out_path.parent or data_dir in out_path.parents:
        print(f"[FATAL] refusing to write the index inside the transcript tree: {out_path}")
        return 2

    print("=" * 78)
    print("PARTICIPANT SPEECH INTERVAL INDEX (Stage S1) - READ-ONLY")
    print("=" * 78)
    env = environment()
    print(f"[INFO] data-dir : {data_dir}")
    print(f"[INFO] output   : {out_path}")
    print(f"[INFO] python   : {env['python']}  pandas={env['pandas']}  tqdm={env['tqdm']}")
    print(f"[INFO] source   : RAW tab-delimited transcripts (data_norm/ has no timestamps)")
    print("-" * 78)

    # ---- discovery ---------------------------------------------------------
    folders = sorted((p for p in data_dir.iterdir()
                      if p.is_dir() and p.name.endswith("_P")),
                     key=lambda p: int(p.name.split("_")[0])
                     if p.name.split("_")[0].isdigit() else 0)
    pids = [f.name.split("_")[0] for f in folders]
    if not pids:
        print(f"[FATAL] no <pid>_P folders under {data_dir}")
        return 2

    subset = bool(args.pids or args.limit)
    if args.pids:
        wanted = {p.strip() for p in args.pids.split(",") if p.strip()}
        unknown = sorted(wanted - set(pids))
        if unknown:
            print(f"[FATAL] --pids: no staged transcript for {unknown}")
            return 2
        folders = [f for f in folders if f.name.split("_")[0] in wanted]
    if args.limit:
        folders = folders[:args.limit]
    print(f"[STEP 1/3] indexing {len(folders)} participant transcript(s)")

    # ---- build -------------------------------------------------------------
    records: Dict[str, Dict[str, Any]] = {}
    failed: List[str] = []
    for folder in progress(folders, len(folders), "intervals", not args.no_progress):
        pid = folder.name.split("_")[0]
        tpath = find_transcript(folder)
        if tpath is None:
            rep.fail("V0.missing", f"no *_transcript.csv under {folder.name}", pid)
            failed.append(pid)
            continue
        rec = build_one(pid, tpath, rep)
        if rec is None:
            failed.append(pid)
        else:
            records[pid] = rec

    print(f"[ OK ] indexed={len(records)}  failed={len(failed)} {failed}")

    # ---- corpus-level invariants (whole-dataset runs only) -----------------
    print("[STEP 2/3] corpus-level validation")
    if not subset:
        if len(records) != EXPECTED_PARTICIPANTS:
            rep.fail("V11.count",
                     f"indexed {len(records)} participants, expected "
                     f"{EXPECTED_PARTICIPANTS}")
        turns = sum(r["n_participant_turns"] for r in records.values())
        if turns != EXPECTED_PARTICIPANT_TURNS:
            rep.warn("V11.turns",
                     f"total participant turns {turns} != measured "
                     f"{EXPECTED_PARTICIPANT_TURNS}")
    else:
        rep.info("V11.subset", "corpus-level count checks skipped (subset run)")

    # Measured invariant: participant turns never genuinely overlap, so merging
    # coalesces touching turns without changing total duration. A non-zero
    # overlap is legitimate data, but it would be new, so it is surfaced.
    overlapped = {p: r["merged_overlap_seconds"] for p, r in records.items()
                  if r["merged_overlap_seconds"] > EPS}
    if overlapped:
        for p, secs in sorted(overlapped.items(), key=lambda kv: int(kv[0])):
            rep.warn("V12.overlap",
                     f"{secs:.3f}s of overlapping participant speech merged", p)
    coalesced = sum(1 for r in records.values()
                    if r["n_intervals"] < r["n_participant_turns"])
    print(f"[ OK ] participants with merged overlap: {len(overlapped)}   "
          f"with coalesced touching turns: {coalesced}")

    # ---- S0 cross-check ----------------------------------------------------
    print("[STEP 3/3] Stage S0 cross-check")
    cross: Dict[str, Any] = {"checked": False}
    if args.no_audit_check or subset:
        reason = "--no-audit-check" if args.no_audit_check else "subset run"
        print(f"[INFO] skipped ({reason})")
    elif not args.audit.is_file():
        rep.warn("X1.audit", f"S0 report not found: {args.audit}")
    else:
        cross = audit_cross_check(records, args.audit, rep)
        if cross.get("checked"):
            print(f"[ OK ] S0 participants={cross['audit_participants']}  "
                  f"speech mismatches={cross['speech_mismatches']}  "
                  f"clip preview={cross['clip_preview_seconds']}")

    # ---- serialise ---------------------------------------------------------
    speech = [r["participant_speech_seconds"] for r in records.values()]
    coverage = [r["participant_coverage_ratio"] for r in records.values()]
    n_iv = [float(r["n_intervals"]) for r in records.values()]

    index = {
        "schema": "daic_participant_intervals/1",
        "generated_by": "build_participant_intervals.py",
        "environment": env,
        "inputs": {
            "data_dir": str(data_dir),
            "audit": str(args.audit.resolve()) if args.audit.is_file() else None,
            "pids": args.pids or None,
            "limit": args.limit,
        },
        "conventions": {
            "source": "RAW tab-delimited transcripts under data/ (data_norm/ drops timestamps)",
            "speaker_filter": "speaker.strip().lower() == 'participant'",
            "merge_rule": "merge when next.start <= current.end (overlap OR touch)",
            "interval_units": "seconds on the transcript clock, [start, stop)",
            "coverage_denominator": "max(stop_time) over ALL speakers; COVAREP is indexed from t=0",
            "rounding_decimals": ROUND_DP,
            "clamping": ("NOT applied - intervals are transcript geometry only. "
                         "S2 must clamp to the COVAREP span (S0 measured the "
                         "corpus-wide cost at 0.020s)."),
        },
        "totals": {
            "participants": len(records),
            "failed": sorted(failed, key=int) if failed else [],
            "participant_turns": sum(r["n_participant_turns"] for r in records.values()),
            "intervals": sum(r["n_intervals"] for r in records.values()),
            "speech_seconds": round(sum(speech), ROUND_DP),
            "fail": rep.count(SEV_FAIL),
            "warn": rep.count(SEV_WARN),
        },
        "summary_statistics": {
            "participant_speech_seconds": describe(speech),
            "participant_coverage_ratio": describe(coverage),
            "intervals_per_participant": describe(n_iv),
        },
        "audit_cross_check": cross,
        "participants": {pid: records[pid] for pid in sorted(records, key=int)},
        "findings": rep.findings,
        "findings_digest": rep.digest(),
        "elapsed_seconds": round(time.time() - t0, 1),
    }

    try:
        out_path.parent.mkdir(parents=True, exist_ok=True)
        with open(out_path, "w", encoding="utf-8") as fh:
            json.dump(index, fh, indent=2, sort_keys=True, ensure_ascii=True)
    except OSError as exc:
        print(f"[FATAL] cannot write index to {out_path}: {exc}")
        return 2

    n_fail, n_warn = rep.count(SEV_FAIL), rep.count(SEV_WARN)
    print("-" * 78)
    print("=== INTERVAL INDEX SUMMARY ===")
    print(f"participants indexed : {len(records)}")
    print(f"participant turns    : {index['totals']['participant_turns']}")
    print(f"merged intervals     : {index['totals']['intervals']}")
    print(f"total speech         : {index['totals']['speech_seconds']:.1f}s")
    print(f"speech per participant: {describe(speech)}")
    print(f"coverage ratio       : {describe(coverage)}")
    print(f"FAIL findings        : {n_fail}")
    print(f"WARN findings        : {n_warn}")
    print(f"findings digest      : {rep.digest()}")
    print(f"elapsed              : {index['elapsed_seconds']:.1f}s")
    print(f"index                : {out_path}")

    if n_fail:
        print("\n[FAIL] interval index is NOT usable - do not proceed to S2.")
        by_check: Dict[str, int] = {}
        for f in rep.findings:
            if f["severity"] == SEV_FAIL:
                by_check[f["check"]] = by_check.get(f["check"], 0) + 1
        for check, n in sorted(by_check.items()):
            print(f"    {check}: {n}")
        return 1

    print("\n[PASS] participant interval index built and validated.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
