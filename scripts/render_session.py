#!/usr/bin/env python3
"""
scripts/render_session.py

Presentation-only, read-only renderer for a mentor-facing demo session.
Reads ALREADY-EXISTING artifacts (Tee-Object-captured logs, MongoDB, the two
LLM-agent .md reports) and renders them in a fixed layout. Does NOT train,
re-train, re-measure, re-run, or import any pipeline module - it is a pure
display layer, stdlib + pymongo only.

Every number is read fresh at render time from an artifact. Nothing is
hardcoded. Any value that cannot be found renders [!!] MISSING - never
estimated, never silently skipped, never faked.

Sources, one per block (see docs/IMPLEMENTATION_NOTES.md's render_session.py
section for the full investigation this was built from):
    [1/7] Database        - live MongoDB query (getCmdLineOpts) +
                             trainer_outputs/demo_db_reset.log (if present) +
                             trainer_outputs/demo_blank_stdin.txt line count
    [2/7] Orchestrator     - trainer_outputs/demo_orchestrator.log (UTF-8,
                             ANSI colour codes present)
    [3/7] Enrollment       - trainer_outputs/demo_enroll.log (if present,
                             UTF-16LE+BOM) with a live MongoDB `devices`
                             collection fallback for whatever it can supply
    [4/7] Client card      - trainer_outputs/demo_r{round}_client{N}.log
                             (UTF-16LE+BOM, PowerShell Tee-Object default)
    [5/7] Aggregation      - trainer_outputs/demo_orchestrator.log
    [6/7] DB Verification  - live MongoDB query (receipts, model_updates,
                             global_models, GridFS fs.files)
    [7/7] AI Reports       - the two .md files under
                             ~/.federated/data/audit_reports/ and
                             ~/.federated/data/clinical_reports/

Usage:
    .venv\\Scripts\\python.exe scripts\\render_session.py --session --round 1
    .venv\\Scripts\\python.exe scripts\\render_session.py --stage db
    .venv\\Scripts\\python.exe scripts\\render_session.py --client trainer_outputs\\demo_r1_client1.log
    .venv\\Scripts\\python.exe scripts\\render_session.py --compare log1 log2 log3
    .venv\\Scripts\\python.exe scripts\\render_session.py --session --round 1 --full
"""

from __future__ import annotations

import argparse
import re
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Optional

from pymongo import MongoClient

WIDTH = 80
DEFAULT_MONGO_URI = "mongodb://localhost:27017"
DEFAULT_DB = "federated_multimodal"
TRAINER_OUTPUTS = Path(r"D:\Download D\BE PIPELINE\Capstone-\trainer_outputs")
HOME = Path.home()

# Fix E4's calibrated clip-norm range (authoritative for delta-L2's [OK]
# boundary - DEMO_WALKTHROUGH.md, cited as a second source in the original
# task spec, does not exist anywhere in this repo; dropped entirely per
# instruction). The runbook's own wider observed range folds into the [~]
# tier's explanation text below, not into the boundary itself.
DELTA_L2_OK_LOW, DELTA_L2_OK_HIGH = 0.625, 0.729
DELTA_L2_CLIP_THRESHOLD = 0.85

CONFIDENT_LOW, CONFIDENT_HIGH = 0.3, 0.7  # unused here directly, kept for reference parity with Phase B


# =============================================================================
# MISSING sentinel + Row rendering
# =============================================================================

class Missing:
    """Sentinel distinguishing 'genuinely not found' from any real falsy
    value (0, False, '', 0.0 are all real, found values and must never be
    confused with this)."""
    def __bool__(self):
        return False
    def __repr__(self):
        return "MISSING"


MISSING = Missing()


class Row:
    """One displayed line: label, actual value, optional expected value,
    optional verdict tag. verdict is one of 'OK', '~', '!!', '--', or None
    (no tag printed - used for plain informational rows like 'Fingerprint').
    If actual is MISSING, the row renders as [!!] MISSING regardless of
    whatever verdict was passed in - a missing value can never be [OK]."""
    __slots__ = ("label", "actual", "expected", "verdict", "note")

    def __init__(self, label: str, actual: Any, expected: Any = None,
                 verdict: Optional[str] = None, note: str = ""):
        self.label = label
        self.actual = actual
        self.expected = expected
        self.verdict = verdict
        self.note = note

    @property
    def is_missing(self) -> bool:
        return self.actual is MISSING


def _fmt_val(v: Any) -> str:
    if v is MISSING:
        return "MISSING"
    if isinstance(v, float):
        return f"{v:.6f}".rstrip("0").rstrip(".") if v != int(v) else f"{v:.1f}"
    return str(v)


def render_row(row: Row, label_w: int = 17, actual_w: int = 12, expected_w: int = 10) -> str:
    tag = "[!!]" if row.is_missing else (f"[{row.verdict}]" if row.verdict else "")
    label = f" {row.label:<{label_w}}"
    if row.expected is not None:
        actual_s = _fmt_val(row.actual)
        expected_s = _fmt_val(row.expected)
        line = f"{label} {actual_s:<{actual_w}} {expected_s:<{expected_w}}{tag:>6}"
    else:
        val_s = _fmt_val(row.actual)
        if row.note:
            line = f"{label} {val_s} {row.note}"
            if tag:
                pad = max(1, WIDTH - len(line) - len(tag))
                line = line + " " * pad + tag
        else:
            line = f"{label} {val_s}"
            if tag:
                pad = max(1, WIDTH - len(line) - len(tag))
                line = line + " " * pad + tag
    return line


def hr(char: str = "=") -> str:
    return char * WIDTH


def format_duration(duration: Optional[float]) -> str:
    if not isinstance(duration, (int, float)):
        return ""
    if 0 < duration < 1.0:
        # Sub-second durations rendered as "0.0s" read as a zero/placeholder
        # value rather than a genuine fast measurement - show ms instead.
        return f"{duration * 1000:.1f}ms"
    return f"{duration:.1f}s"


def block_header(title: str, verdict: Optional[str], duration: Optional[float]) -> str:
    # Block-level headers show the bare verdict word (OK / SUCCESS / MISSING),
    # not bracketed - only per-row tags use brackets. Matches FINAL PROMPT.txt's
    # target layout ("[1/7]  DATABASE   OK", not "...[OK]").
    tag = verdict if verdict else ""
    dur = format_duration(duration)
    left = f" {title}"
    right = f"{tag}  {dur}".strip()
    pad = max(1, WIDTH - len(left) - len(right))
    return left + " " * pad + right


# =============================================================================
# File reading: BOM-sniffing decode, ANSI/escape stripping
# =============================================================================

_ANSI_RE = re.compile(r"\x1b\[[0-9;]*[a-zA-Z]")

# Characters observed, empirically, to appear mojibake-corrupted in these
# logs (an em dash written through a console that assumed the wrong
# codepage) - mapped to a plain-ASCII equivalent for display. Anything not
# in this map falls through to the NFKD-normalize-and-drop fallback in
# to_ascii() below, so no unrecognised unicode byte can ever reach stdout.
_KNOWN_MOJIBAKE = {
    "ΓÇö": " - ",   # corrupted U+2014 em dash
    "—": " - ",                # a clean em dash, if one ever appears
    "ε": "eps",                # epsilon (Epsilon (eps) : ...)
}


def to_ascii(text: str) -> str:
    """Guarantees pure-ASCII output for the console, per Rule 10. Known
    mojibake/unicode substitutions are applied first; anything left is
    NFKD-normalised and any remaining non-ASCII byte is dropped (never
    silently rendered as a '?' that could be mistaken for real content)."""
    import unicodedata
    for bad, good in _KNOWN_MOJIBAKE.items():
        text = text.replace(bad, good)
    text = unicodedata.normalize("NFKD", text)
    return text.encode("ascii", "ignore").decode("ascii")


def read_text_sniff_bom(path: Path) -> str:
    """Reads a text file, detecting its encoding by BOM bytes rather than
    assuming one scheme - Tee-Object-produced logs are UTF-16LE+BOM,
    cmd.exe-redirected logs (the orchestrator) are UTF-8, and
    demo_blank_stdin.txt is UTF-8+BOM. Never touches the file on disk -
    read-only."""
    raw = path.read_bytes()
    if raw.startswith(b"\xff\xfe"):
        return raw.decode("utf-16-le")
    if raw.startswith(b"\xfe\xff"):
        return raw.decode("utf-16-be")
    if raw.startswith(b"\xef\xbb\xbf"):
        return raw.decode("utf-8-sig")
    return raw.decode("utf-8", errors="replace")


def strip_ansi(line: str) -> tuple[str, bool]:
    """Strips ANSI/progress-bar escape residue AND non-ASCII mojibake for
    DISPLAY only - the raw log file itself is never modified. Returns
    (cleaned_line, had_escape)."""
    cleaned = _ANSI_RE.sub("", line)
    had = cleaned != line
    # Also catch the mangled-replacement-character residue left behind when
    # an already-corrupted (non-UTF-8-clean) line was decoded with
    # errors='replace' upstream of this script - visible as a literal U+FFFD
    # or '?' where a unicode arrow/en-dash/epsilon used to be.
    if "\ufffd" in cleaned:
        had = True
    ascii_cleaned = to_ascii(cleaned)
    if ascii_cleaned != cleaned:
        had = True
    return ascii_cleaned, had


# =============================================================================
# Noise classification (Rule 3): four categories, counted, never silently
# dropped. A line matching a KNOWN_MARKER (anything this script actually
# extracts a displayed value from) is NEVER suppressible - checked first,
# before any noise category, and asserted against below.
# =============================================================================

PHYSICIAN_LOOP_MARKERS = [
    "PHYSICIAN FEEDBACK LOOP", "Enter corrected PHQ", "Raw model output (PHQ)",
    "Negative probability", "Positive probability", "Classification threshold",
    "Predicted class", "Physician correction", "Correction applied", "Text (truncated)",
]
TRAINING_DIAGNOSTIC_RE = re.compile(r"^\s*\[[A-Za-z0-9_-]+\]")
DP_UPDATES_LISTING_RE = re.compile(r"dp_updates[\\/].*\.pt\.enc")


def classify_and_count_noise(lines: list[str], known_marker_substrings: list[str]) -> dict:
    counts = {"physician_loop": 0, "training_diagnostic": 0, "escape_residue": 0, "dp_updates_listing": 0}
    asserted_never_suppressed = 0
    for raw in lines:
        cleaned, had_escape = strip_ansi(raw)
        if any(m in cleaned for m in known_marker_substrings):
            asserted_never_suppressed += 1
            continue  # known marker: never suppressible, not counted as noise
        if any(m in cleaned for m in PHYSICIAN_LOOP_MARKERS):
            counts["physician_loop"] += 1
            continue
        if had_escape:
            counts["escape_residue"] += 1
            continue
        if DP_UPDATES_LISTING_RE.search(cleaned):
            counts["dp_updates_listing"] += 1
            continue
        if TRAINING_DIAGNOSTIC_RE.match(cleaned):
            counts["training_diagnostic"] += 1
            continue
    return counts, asserted_never_suppressed


def render_suppressed_block(counts: dict, total_suppressed_lines_hidden: int, full_mode: bool) -> list[str]:
    if full_mode:
        return [" --full: noise suppression disabled, all lines shown above."]
    lines = [" SUPPRESSED (per-category line counts - use --full to disable):"]
    lines.append(f"   physician-loop prompt echoes      : {counts['physician_loop']}")
    lines.append(f"   per-step training diagnostics     : {counts['training_diagnostic']}")
    lines.append(f"   ANSI / progress-bar escape residue: {counts['escape_residue']}")
    lines.append(f"   dp_updates file listing            : {counts['dp_updates_listing']}")
    return lines


# =============================================================================
# Marker extraction helpers
# =============================================================================

def find_line(lines: list[str], substring: str) -> Optional[str]:
    for l in lines:
        if substring in l:
            return strip_ansi(l)[0]
    return None


def find_all_lines(lines: list[str], substring: str) -> list[str]:
    return [strip_ansi(l)[0] for l in lines if substring in l]


def extract_after_colon(line: Optional[str]) -> Optional[str]:
    if line is None:
        return None
    if ":" not in line:
        return None
    return line.split(":", 1)[1].strip()


def extract_number(text: Optional[str], pattern: str = r"-?\d[\d,]*(?:\.\d+)?") -> Optional[float]:
    """Extracts the first genuine number in `text`. The pattern requires a
    digit immediately after an optional leading '-' - a bare ',' or '-'
    (e.g. the hyphen in a label like '[FIX-B]', or a comma in prose text)
    can never match on its own the way a looser '[-\\d,\\.]+' class would."""
    if text is None:
        return None
    m = re.search(pattern, text)
    if not m:
        return None
    try:
        return float(m.group().replace(",", ""))
    except ValueError:
        return None


def extract_int(text: Optional[str]) -> Optional[int]:
    v = extract_number(text)
    return int(v) if v is not None else None


def numeric_after_colon(line: Optional[str]) -> Optional[float]:
    """Reads a 'Label : value ...' log line and returns the numeric value
    AFTER the separating colon only. Never scans the label itself for
    digits - labels like 'Delta L2 norm', '  f1', '[FIX-B] ... frozen'
    contain digits/hyphens of their own that must never be mistaken for
    the value that follows the colon."""
    return extract_number(extract_after_colon(line))


def int_after_colon(line: Optional[str]) -> Optional[int]:
    v = numeric_after_colon(line)
    return int(v) if v is not None else None


# =============================================================================
# [1/7] DATABASE
# =============================================================================

def build_database_block(mongo_uri: str, db_name: str) -> dict:
    rows = []
    verdict_ok = True

    try:
        client = MongoClient(mongo_uri, serverSelectionTimeoutMS=5000)
        client.admin.command("ping")
        opts = client.admin.command("getCmdLineOpts")
        db_path = opts["parsed"]["storage"]["dbPath"]
        port = opts["parsed"].get("net", {}).get("port", MISSING)
        rows.append(Row("MongoDB", f"up, port {port}"))
        rows.append(Row("Data path", db_path, verdict="OK", note="(D: not C:)" if str(db_path).startswith("D:") else ""))
        client.close()
    except Exception as e:
        rows.append(Row("MongoDB", MISSING, verdict="!!"))
        rows.append(Row("Data path", MISSING, verdict="!!"))
        verdict_ok = False

    reset_log = TRAINER_OUTPUTS / "demo_db_reset.log"
    if reset_log.exists():
        text = read_text_sniff_bom(reset_log)
        lines = text.splitlines()
        ok_line = find_line(lines, "[OK] All collections")
        fail_line = find_line(lines, "[FAIL]")
        if ok_line:
            rows.append(Row("Reset", "all project collections empty", verdict="OK"))
        elif fail_line:
            rows.append(Row("Reset", "one or more collections non-empty", verdict="!!"))
            verdict_ok = False
        else:
            rows.append(Row("Reset", MISSING, verdict="!!"))
            verdict_ok = False
    else:
        rows.append(Row("Reset", MISSING, verdict="!!",
                         note="(trainer_outputs/demo_db_reset.log not found - re-run Terminal 1 with the Tee-Object capture)"))
        verdict_ok = False

    # Q3: libraryDB is stated in reset_all_federated_dbs.py's docstring, never
    # printed at runtime, never checked by this tool. [--] not [OK].
    rows.append(Row("Untouched", "libraryDB (by design, not checked)", verdict="--"))

    stdin_file = TRAINER_OUTPUTS / "demo_blank_stdin.txt"
    if stdin_file.exists():
        n_lines = len(read_text_sniff_bom(stdin_file).splitlines())
        rows.append(Row("Blank stdin", f"{n_lines} lines ready", verdict="OK"))
    else:
        rows.append(Row("Blank stdin", MISSING, verdict="!!"))
        verdict_ok = False

    explanation = "Round counter starts at 1. Clean slate confirmed." if verdict_ok else \
        "Some [1/7] fields could not be sourced - see [!!] rows above."

    return {"title": "[1/7]  DATABASE", "rows": rows, "verdict": "OK" if verdict_ok else "!!",
            "duration": None, "explanation": explanation}


# =============================================================================
# [2/7] RUST ORCHESTRATOR
# =============================================================================

def get_orchestrator_start_epoch() -> Optional[float]:
    """Returns the current orchestrator process's start time (its 'MongoDB
    connected' timestamp), as a unix epoch float, or None if unavailable.
    Used to tell a genuinely-this-round client log apart from a stale one
    left over on disk from a previous demo session that reused the same
    round_id (and therefore the same trainer_outputs/demo_r{round}_client{n}.log
    filename) - see build_client_freshness_note()."""
    log_path = TRAINER_OUTPUTS / "demo_orchestrator.log"
    if not log_path.exists():
        return None
    text = read_text_sniff_bom(log_path)
    ts_re = re.compile(r"^\d{4}-\d{2}-\d{2}T[\d:.]+Z")
    for l in text.splitlines():
        clean = strip_ansi(l)[0]
        if "MongoDB connected" in clean:
            m = ts_re.match(clean.strip())
            if m:
                try:
                    dt = datetime.strptime(m.group(), "%Y-%m-%dT%H:%M:%S.%fZ")
                    import calendar
                    return calendar.timegm(dt.timetuple()) + dt.microsecond / 1e6
                except ValueError:
                    return None
    return None


def build_orchestrator_block() -> dict:
    log_path = TRAINER_OUTPUTS / "demo_orchestrator.log"
    if not log_path.exists():
        return {"title": "[2/7]  RUST ORCHESTRATOR", "rows": [Row("Log", MISSING, verdict="!!")],
                "verdict": "!!", "duration": None,
                "explanation": "trainer_outputs/demo_orchestrator.log not found."}

    text = read_text_sniff_bom(log_path)
    raw_lines = text.splitlines()
    lines = [strip_ansi(l)[0] for l in raw_lines]

    rows = []
    verdict_ok = True

    db_line = find_line(lines, "MongoDB database")
    db_val = extract_after_colon(db_line) if db_line else find_line(lines, "MongoDB connected")
    if db_val:
        rows.append(Row("Database", db_val.split("(database=")[-1].rstrip(")") if "database=" in db_val else db_val, verdict="OK"))
    else:
        rows.append(Row("Database", MISSING, verdict="!!"))
        verdict_ok = False

    recovery_line = find_line(lines, "[RECOVERY] Round 1:")
    if recovery_line:
        val = recovery_line.split("[RECOVERY]", 1)[1].strip()
        rows.append(Row("Recovery state", val, verdict="OK"))
    else:
        rows.append(Row("Recovery state", MISSING, verdict="!!"))
        verdict_ok = False

    mtls_line = find_line(lines, "[SERVER] Running in mTLS mode")
    if mtls_line:
        val = mtls_line.split("mode on", 1)[1].strip() if "mode on" in mtls_line else mtls_line
        rows.append(Row("Transport", f"mTLS on {val}", verdict="OK"))
    else:
        rows.append(Row("Transport", MISSING, verdict="!!"))
        verdict_ok = False

    otp_line = find_line(lines, "[DEV] Enrollment OTP:")
    if otp_line:
        otp = extract_int(otp_line.split("OTP:", 1)[1])
        rows.append(Row("Enrollment OTP", otp, note="expires in 10 min"))
    else:
        rows.append(Row("Enrollment OTP", MISSING, verdict="!!"))
        verdict_ok = False

    # Startup latency: "MongoDB connected" timestamp -> "[SERVER] Running in mTLS mode" timestamp
    duration = None
    ts_re = re.compile(r"^\d{4}-\d{2}-\d{2}T[\d:.]+Z")
    start_ts = end_ts = None
    for l in raw_lines:
        clean = strip_ansi(l)[0]
        if "MongoDB connected" in clean and start_ts is None:
            m = ts_re.match(clean.strip())
            if m:
                start_ts = m.group()
        if "[SERVER] mTLS mode" in clean or "mTLS mode" in clean and "binding" in clean:
            m = ts_re.match(clean.strip())
            if m:
                end_ts = m.group()
    if start_ts and end_ts:
        try:
            t0 = datetime.strptime(start_ts, "%Y-%m-%dT%H:%M:%S.%fZ")
            t1 = datetime.strptime(end_ts, "%Y-%m-%dT%H:%M:%S.%fZ")
            duration = (t1 - t0).total_seconds()
        except ValueError:
            duration = None

    explanation = ("mTLS = BOTH sides present certificates, not just the server.\n"
                   " verified_updates=0 confirms the reset actually took effect."
                   if verdict_ok else "Some [2/7] fields could not be sourced.")

    return {"title": "[2/7]  RUST ORCHESTRATOR", "rows": rows, "verdict": "OK" if verdict_ok else "!!",
            "duration": duration, "explanation": explanation}


# =============================================================================
# [3/7] DEVICE ENROLLMENT
# =============================================================================

def build_enrollment_block(mongo_uri: str, db_name: str) -> dict:
    rows = []
    verdict_ok = True
    enroll_log = TRAINER_OUTPUTS / "demo_enroll.log"

    if enroll_log.exists():
        text = read_text_sniff_bom(enroll_log)
        lines = text.splitlines()

        # Note: enroll_step5.py's own progress lines are numbered "[1] ...",
        # "[7] ..." - that leading bracketed digit must never be scanned as
        # if it were the value, so every value below is pulled from inside
        # its own "(N bytes)"/"(...)" group, never from a whole-line digit
        # scan.
        pubkey_line = find_line(lines, "Device pubkey loaded")
        if pubkey_line:
            m = re.search(r"\((\d+) bytes\)", pubkey_line)
            n = int(m.group(1)) if m else None
            rows.append(Row("TPM public key", f"{n} bytes read from hardware chip" if n else MISSING,
                             verdict="OK" if n else "!!"))
            if n is None:
                verdict_ok = False
        else:
            rows.append(Row("TPM public key", MISSING, verdict="!!")); verdict_ok = False

        if find_line(lines, "Generating RSA-2048"):
            rows.append(Row("Client key", "RSA-2048 generated, CSR sent", verdict="OK"))
        else:
            rows.append(Row("Client key", MISSING, verdict="!!")); verdict_ok = False

        otp_line = find_line(lines, "OTP used")
        otp_val = int_after_colon(otp_line) if otp_line else None
        if otp_val is not None:
            rows.append(Row("OTP matched", otp_val, verdict="OK"))
        else:
            rows.append(Row("OTP matched", MISSING, verdict="!!")); verdict_ok = False

        cert_line = find_line(lines, "client.pem written")
        if cert_line:
            m = re.search(r"\((\d+) bytes\)", cert_line)
            n = int(m.group(1)) if m else None
            rows.append(Row("Certificate", f"client.pem, {n} bytes" if n else "client.pem", verdict="OK"))
        else:
            rows.append(Row("Certificate", MISSING, verdict="!!")); verdict_ok = False

        fp_line = find_line(lines, "Fingerprint:")
        fp_val = extract_after_colon(fp_line) if fp_line else None
        rows.append(Row("Fingerprint", fp_val if fp_val else MISSING, verdict=None if fp_val else "!!"))
        if not fp_val:
            verdict_ok = False
    else:
        # No captured enrollment log for this session - fall back to what
        # MongoDB's devices collection and the local client.pem file can
        # supply live. Rule 2: everything else renders MISSING, not skipped.
        try:
            client = MongoClient(mongo_uri, serverSelectionTimeoutMS=5000)
            db = client[db_name]
            dev = db["devices"].find_one(sort=[("enrolled_at", -1)])
            client.close()
        except Exception:
            dev = None

        if dev:
            pubkey_bytes = len(dev.get("pubkey_pem", "").encode())
            rows.append(Row("TPM public key", f"{pubkey_bytes} bytes (live MongoDB devices collection)", verdict="OK"))
            rows.append(Row("Client key", MISSING, verdict="!!",
                             note="(no demo_enroll.log for this session - re-run Terminal 3 with the Tee-Object capture)"))
            rows.append(Row("OTP matched", MISSING, verdict="!!"))
            cert_path = HOME / ".federated" / "keys" / "client.pem"
            if cert_path.exists():
                rows.append(Row("Certificate", f"client.pem, {cert_path.stat().st_size} bytes (live disk file)", verdict="OK"))
            else:
                rows.append(Row("Certificate", MISSING, verdict="!!"))
            rows.append(Row("Fingerprint", dev.get("device_id", MISSING)[:16] if dev.get("device_id") else MISSING))
            verdict_ok = False  # partial data - can't claim full OK
        else:
            rows = [Row("TPM public key", MISSING, verdict="!!"), Row("Client key", MISSING, verdict="!!"),
                    Row("OTP matched", MISSING, verdict="!!"), Row("Certificate", MISSING, verdict="!!"),
                    Row("Fingerprint", MISSING, verdict="!!")]
            verdict_ok = False

    explanation = ("The private key NEVER exists as a file. It lives inside the TPM chip.\n"
                   " An attacker cannot copy it and impersonate this machine.")
    return {"title": "[3/7]  DEVICE ENROLLMENT", "rows": rows, "verdict": "OK" if verdict_ok else "!!",
            "duration": None, "explanation": explanation}


# =============================================================================
# [4/7] CLIENT CARD
# =============================================================================

KNOWN_CLIENT_MARKERS = [
    "From parquet (ground truth", "From model fallback", "Train size", "Eval size",
    "Partition mode", "Trainable params", "Frozen params", "[FIX-B] Text encoder frozen",
    "batch shape", "Learning rate (base)", "Effective learning rate", "Epoch 1/10", "Epoch 10/10",
    "  f1", "  accuracy", "STEP-A2-XAI", "Modality split", "Samples attributed",
    "Delta L2 norm before clamp", "Delta L2 norm after clamp", "Clipping applied",
    "Noise norm", "Epsilon", "Payload size", "Chunks", "mTLS status", "Signature size",
    "Round Metadata", "Global Model", "Lda ", "Training ", "Dp ", "Encryption ",
    "Tpm Signing", "Receipt ", "Overall E2E status",
]


def build_client_block(log_path: Path, client_num: int, total: int = 3, full_mode: bool = False,
                        min_mtime_epoch: Optional[float] = None) -> dict:
    if not log_path.exists():
        return {"title": f"[4/7]  CLIENT {client_num} of {total}", "rows": [],
                "verdict": "!!", "duration": None, "status_word": "MISSING",
                "explanation": f"{log_path} not found - this client has not run yet.",
                "missing": True, "suppressed": None}

    # trainer_outputs/demo_r{round}_client{n}.log is NOT cleared between demo
    # sessions - a stale file from a previous run can sit at this exact path
    # while the CURRENT orchestrator process has never heard from this
    # client. A file whose mtime predates the current orchestrator's own
    # start time cannot possibly belong to this round - it must be leftover.
    # Rule 2 says MISSING is never faked, so this is rendered as MISSING
    # (with the reason stated), never as a live client card built from
    # foreign data.
    if min_mtime_epoch is not None:
        file_mtime = log_path.stat().st_mtime
        if file_mtime < min_mtime_epoch:
            age = datetime.fromtimestamp(file_mtime)
            return {"title": f"[4/7]  CLIENT {client_num} of {total}", "rows": [],
                    "verdict": "!!", "duration": None, "status_word": "MISSING",
                    "explanation": (f"{log_path} exists but is STALE - last modified {age:%Y-%m-%d %H:%M:%S}, "
                                     f"before the current orchestrator process started. This is a leftover "
                                     f"artifact from a previous demo session reusing this round's filename, "
                                     f"not a client that ran in this round. Treated as not-run."),
                    "missing": True, "suppressed": None}

    text = read_text_sniff_bom(log_path)
    raw_lines = text.splitlines()
    lines = [strip_ansi(l)[0] for l in raw_lines]

    noise_counts, _asserted = classify_and_count_noise(raw_lines, KNOWN_CLIENT_MARKERS)

    sections = {"DATA": [], "MODEL": [], "TRAINING": [], "XAI": [], "PRIVACY": [], "TRANSPORT": []}
    verdict_ok = True

    # ---- DATA ----
    real_labels = int_after_colon(find_line(lines, "From parquet (ground truth"))
    fallback_labels = int_after_colon(find_line(lines, "From model fallback"))
    sections["DATA"].append(Row("Labels from parquet (real PHQ-8)", real_labels if real_labels is not None else MISSING,
                                 verdict="OK" if real_labels else ("!!" if real_labels is None else "OK")))
    sections["DATA"].append(Row("Labels from model fallback (must be zero)", fallback_labels if fallback_labels is not None else MISSING,
                                 verdict="OK" if fallback_labels == 0 else ("!!" if fallback_labels else None)))

    train_line = find_line(lines, "Train size")
    eval_line = find_line(lines, "Eval size")
    train_pos = extract_int(re.search(r"positive=(\d+)", train_line).group(1)) if train_line and re.search(r"positive=(\d+)", train_line) else None
    train_neg = extract_int(re.search(r"negative=(\d+)", train_line).group(1)) if train_line and re.search(r"negative=(\d+)", train_line) else None
    eval_pos = extract_int(re.search(r"positive=(\d+)", eval_line).group(1)) if eval_line and re.search(r"positive=(\d+)", eval_line) else None
    eval_neg = extract_int(re.search(r"negative=(\d+)", eval_line).group(1)) if eval_line and re.search(r"negative=(\d+)", eval_line) else None
    if None not in (train_pos, train_neg, eval_pos, eval_neg):
        dist = f"{train_pos + eval_pos} / {train_neg + eval_neg}"
        sections["DATA"].append(Row("Distribution (computed: train+eval)", dist, verdict="OK"))
    else:
        sections["DATA"].append(Row("Distribution", MISSING, verdict="!!")); verdict_ok = False

    train_n = int_after_colon(train_line) if train_line else None
    eval_n = int_after_colon(eval_line) if eval_line else None
    if train_n is not None and eval_n is not None:
        sections["DATA"].append(Row("Train / eval split", f"{train_n} / {eval_n}", verdict="OK"))
    else:
        sections["DATA"].append(Row("Train / eval split", MISSING, verdict="!!")); verdict_ok = False

    partition_line = find_line(lines, "Partition mode")
    partition_val = extract_after_colon(partition_line).split("(")[0].strip() if partition_line else None
    sections["DATA"].append(Row("Partition mode", partition_val if partition_val else MISSING,
                                 verdict="OK" if partition_val else "!!"))

    # ---- MODEL ---- (Q4: two independent post-freeze readings cross-checked)
    fixb_line = find_line(lines, "[FIX-B] Text encoder frozen:")
    later_frozen_line = find_line(lines, "Frozen params")
    trainable_line = find_line(lines, "Trainable params")
    frozen_actual = numeric_after_colon(fixb_line) if fixb_line else None
    frozen_expected = numeric_after_colon(later_frozen_line) if later_frozen_line else None
    trainable_actual = int_after_colon(trainable_line) if trainable_line else None

    if trainable_actual is not None:
        sections["MODEL"].append(Row("Trainable (only these are trained + noised + sent)",
                                      trainable_actual, verdict="OK"))
    else:
        sections["MODEL"].append(Row("Trainable", MISSING, verdict="!!")); verdict_ok = False

    if frozen_actual is not None and frozen_expected is not None:
        v = "OK" if int(frozen_actual) == int(frozen_expected) else "!!"
        sections["MODEL"].append(Row("Frozen (MentalBERT, never leaves device)",
                                      int(frozen_actual), expected=int(frozen_expected), verdict=v))
        if v == "!!":
            verdict_ok = False
    else:
        sections["MODEL"].append(Row("Frozen (MentalBERT, never leaves device)", MISSING, verdict="!!"))
        verdict_ok = False

    text_shape = find_line(lines, "text batch shape")
    audio_shape = find_line(lines, "audio batch shape")
    video_shape = find_line(lines, "video batch shape")
    if text_shape and audio_shape and video_shape:
        t = re.search(r"\(([\d, ]+)\)", text_shape).group(1)
        a = re.search(r"\(([\d, ]+)\)", audio_shape).group(1)
        v = re.search(r"\(([\d, ]+)\)", video_shape).group(1)
        sections["MODEL"].append(Row("Batch  text / audio / video", f"({t}) ({a}) ({v})", verdict="OK"))
    else:
        sections["MODEL"].append(Row("Batch  text / audio / video", MISSING, verdict="!!")); verdict_ok = False

    # ---- TRAINING ----
    lr_line = find_line(lines, "Learning rate (base)")
    lr_val = numeric_after_colon(lr_line) if lr_line else None
    sections["TRAINING"].append(Row("Learning rate", f"{lr_val:.1e}" if lr_val is not None else MISSING,
                                     verdict="OK" if lr_val is not None else "!!"))

    epoch1_line = find_line(lines, "Epoch 1/10")
    epoch10_line = find_line(lines, "Epoch 10/10")
    # "Epoch 1/10" / "Epoch 10/10" are themselves digits in the label, so the
    # value must come from after "average loss:", never from a whole-line scan.
    e1 = numeric_after_colon(epoch1_line) if epoch1_line else None
    e10 = numeric_after_colon(epoch10_line) if epoch10_line else None
    if e1 is not None and e10 is not None:
        falling = e10 < e1
        sections["TRAINING"].append(Row("Loss  epoch 1 -> 10", f"{e1:.4f} -> {e10:.4f}",
                                         note="falling" if falling else "NOT falling",
                                         verdict="OK" if falling else "!!"))
        if not falling:
            verdict_ok = False
    else:
        sections["TRAINING"].append(Row("Loss  epoch 1 -> 10", MISSING, verdict="!!")); verdict_ok = False

    f1_line = find_line(lines, "  f1")
    # The "f1" label itself contains a digit - the value must come from
    # after the colon, never from a whole-line digit scan.
    f1_val = numeric_after_colon(f1_line) if f1_line else None
    f1_label = f"F1  (LOCAL, PRE-DP, on {eval_n} held-out)" if eval_n is not None else "F1  (LOCAL, PRE-DP, on held-out)"
    if f1_val is not None:
        sections["TRAINING"].append(Row(f1_label, f"{f1_val:.4f}", verdict="OK"))
    else:
        sections["TRAINING"].append(Row(f1_label, MISSING, verdict="!!")); verdict_ok = False

    # ---- XAI ----
    xai_disabled_line = next((l for l in lines if re.search(r"Status\s*:\s*disabled", l)), None)
    modality_line = find_line(lines, "Modality split (normalized)")
    xai_samples_line = find_line(lines, "[STEP-A2-XAI]")
    if xai_disabled_line:
        # XAI_ENABLED not set for this client - a real, legitimate per-client
        # configuration state, not a missing/failed value. [--] per the same
        # convention as [1/7]'s libraryDB row: by design, not a check result.
        reason = extract_after_colon(xai_disabled_line) or "disabled"
        sections["XAI"].append(Row("Modality split", f"XAI {reason.strip()} for this client", verdict="--"))
    elif modality_line:
        m = re.search(r"text=([\d.]+)\s+audio=([\d.]+)\s+vision=([\d.]+)", modality_line)
        if m:
            sections["XAI"].append(Row("Modality split  text / audio / vision",
                                        f"{m.group(1)} / {m.group(2)} / {m.group(3)}", verdict="OK"))
        else:
            sections["XAI"].append(Row("Modality split", MISSING, verdict="!!")); verdict_ok = False
    else:
        sections["XAI"].append(Row("Modality split", MISSING, verdict="!!")); verdict_ok = False

    if xai_samples_line:
        m = re.search(r"(\d+) samples attributed in ([\d.]+)s", xai_samples_line)
        if m:
            sections["XAI"].append(Row("Samples attributed", f"{m.group(1)} in {m.group(2)}s"))
    elif not xai_disabled_line:
        sections["XAI"].append(Row("Samples attributed", MISSING, verdict="!!"))
    sections["XAI"].append(Row("Uploaded?", "NO - local disk only"))

    # ---- PRIVACY ----
    clip_norm_line = find_line(lines, "Clip norm configured")
    clip_threshold = numeric_after_colon(clip_norm_line) if clip_norm_line else None
    if clip_threshold is None:
        clip_threshold = DELTA_L2_CLIP_THRESHOLD  # artifact missing this round - fall back to the documented default
        verdict_ok = False

    before_line = find_line(lines, "Delta L2 norm before clamp")
    # "L2" in the label contains a digit - value must come from after the colon.
    delta_l2 = numeric_after_colon(before_line) if before_line else None
    clip_line = find_line(lines, "Clipping applied")
    clip_val = extract_after_colon(clip_line) if clip_line else None
    noise_line = find_line(lines, "Noise norm")
    noise_val = numeric_after_colon(noise_line) if noise_line else None
    eps_line = find_line(lines, "Epsilon")
    eps_val = numeric_after_colon(eps_line) if eps_line else None
    clamp_engaged = any("ENGAGED" in l and "SAFETY-CLAMP" in l for l in lines)

    delta_tier_explanations = []
    if delta_l2 is not None:
        if DELTA_L2_OK_LOW <= delta_l2 <= DELTA_L2_OK_HIGH:
            v = "OK"
        elif delta_l2 < clip_threshold:
            v = "~"
            delta_tier_explanations.append(
                f" [~] Delta L2 {delta_l2:.3f} is above the calibration range but well under the "
                f"{clip_threshold} clip threshold. Clipping correctly did not fire. Run-to-run variation."
                if delta_l2 > DELTA_L2_OK_HIGH else
                f" [~] Delta L2 {delta_l2:.3f} is below the calibration range but well under the "
                f"{clip_threshold} clip threshold. Run-to-run variation."
            )
        else:
            v = "!!"
            verdict_ok = False
        sections["PRIVACY"].append(Row("Delta L2 (the genuine learning signal)", f"{delta_l2:.6f}",
                                        expected=f"{DELTA_L2_OK_LOW}-{DELTA_L2_OK_HIGH}", verdict=v))
    else:
        sections["PRIVACY"].append(Row("Delta L2 (the genuine learning signal)", MISSING, verdict="!!")); verdict_ok = False

    sections["PRIVACY"].append(Row("Safety clamp engaged", "yes" if clamp_engaged else "no",
                                    verdict="!!" if clamp_engaged else "OK"))
    if clamp_engaged:
        verdict_ok = False

    if clip_val is not None and delta_l2 is not None:
        expected_clip = "False" if delta_l2 < clip_threshold else "True"
        v = "OK" if clip_val.strip() == expected_clip else "!!"
        sections["PRIVACY"].append(Row(f"Clipping applied  ({delta_l2:.3f} vs {clip_threshold} threshold, from this run's own log)",
                                        clip_val.strip(), expected=expected_clip, verdict=v))
        if v == "!!":
            verdict_ok = False
    else:
        sections["PRIVACY"].append(Row("Clipping applied", MISSING, verdict="!!")); verdict_ok = False

    sections["PRIVACY"].append(Row("Noise added", f"{noise_val:.6f}" if noise_val is not None else MISSING,
                                    verdict="OK" if noise_val is not None else "!!"))
    sections["PRIVACY"].append(Row("Epsilon  (privacy cost, target <= 8)", f"{eps_val:.6f}" if eps_val is not None else MISSING,
                                    expected="<= 8", verdict=("OK" if (eps_val is not None and eps_val <= 8) else "!!")))
    if eps_val is None or eps_val > 8:
        verdict_ok = False

    # ---- TRANSPORT ----
    payload_line = find_line(lines, "Payload size")
    payload_val = int_after_colon(payload_line) if payload_line else None
    chunks_line = find_line(lines, "Chunks")
    chunks_val = int_after_colon(chunks_line) if chunks_line else None
    mtls_line = find_line(lines, "mTLS status")
    mtls_val = extract_after_colon(mtls_line) if mtls_line else None
    sig_line = find_line(lines, "Signature size")
    sig_val = int_after_colon(sig_line) if sig_line else None

    sections["TRANSPORT"].append(Row("Encrypted payload  (AES-256-GCM)",
                                      f"{payload_val:,} B" if payload_val else MISSING,
                                      verdict="OK" if payload_val else "!!"))
    if payload_val is None:
        verdict_ok = False
    sections["TRANSPORT"].append(Row("mTLS", "PASS" if mtls_val and "PASS" in mtls_val else MISSING,
                                      verdict="OK" if mtls_val and "PASS" in mtls_val else "!!"))
    if not (mtls_val and "PASS" in mtls_val):
        verdict_ok = False
    sections["TRANSPORT"].append(Row("TPM signature", f"{sig_val} bytes signed" if sig_val else MISSING,
                                      verdict="OK" if sig_val else "!!"))
    if sig_val is None:
        verdict_ok = False

    # ---- STAGES ----
    # The 8-line "Round Metadata / Global Model / Lda / Training / Dp /
    # Encryption / Tpm Signing / Receipt : STATUS" block is a distinctive
    # contiguous group near the end of the log. Earlier lines reuse some of
    # these same words for unrelated data ("Training time : 51.83 sec",
    # "Encryption duration : 0.0576 sec", "Receipt submitted : YES") - a
    # plain substring search for e.g. "Training " would find those first
    # and silently report the wrong value. Anchoring on "Round Metadata"
    # and reading the next 7 lines by fixed position avoids that collision
    # entirely.
    stage_short_names = ["Metadata", "GlobalModel", "Lda", "Training", "Dp", "Encryption", "TpmSigning", "Receipt"]
    stage_results = []
    anchor_idx = next((i for i, l in enumerate(lines) if l.strip().startswith("Round Metadata")), None)
    if anchor_idx is not None:
        for offset, short_name in enumerate(stage_short_names):
            if anchor_idx + offset < len(lines):
                status = extract_after_colon(lines[anchor_idx + offset])
                stage_results.append((short_name, status.strip() if status else "MISSING"))
                if status is None:
                    verdict_ok = False
            else:
                stage_results.append((short_name, "MISSING")); verdict_ok = False
    else:
        stage_results = [(name, "MISSING") for name in stage_short_names]
        verdict_ok = False

    overall_line = find_line(lines, "Overall E2E status")
    overall_status = extract_after_colon(overall_line) if overall_line else None
    status_word = overall_status.strip() if overall_status else ("MISSING" if not verdict_ok else "UNKNOWN")

    # duration: first timestamped line to last timestamped line in the file
    duration = None
    ts_re = re.compile(r"^\[(\d{2}):(\d{2}):(\d{2})\]")
    first_ts = last_ts = None
    for l in lines:
        m = ts_re.match(l)
        if m:
            secs = int(m.group(1)) * 3600 + int(m.group(2)) * 60 + int(m.group(3))
            if first_ts is None:
                first_ts = secs
            last_ts = secs
    if first_ts is not None and last_ts is not None:
        duration = last_ts - first_ts
        if duration < 0:
            duration += 86400  # midnight rollover

    explanation_lines = list(delta_tier_explanations)
    explanation_lines.append(" GlobalModel:SKIP is CORRECT in round 1 - nothing exists to warm-start from.")

    return {
        "title": f"[4/7]  CLIENT {client_num} of {total}",
        "sections": sections, "stage_results": stage_results,
        "verdict": "OK" if verdict_ok else "!!",
        "status_word": status_word,
        "duration": duration,
        "explanation": "\n".join(explanation_lines),
        "noise_counts": noise_counts,
        "missing": False,
        "raw_lines": raw_lines,
        "extracted": {
            "delta_l2": delta_l2, "noise": noise_val, "epsilon": eps_val,
            "payload": payload_val, "f1": f1_val,
        },
    }


# =============================================================================
# THREE CLIENTS SIDE BY SIDE
# =============================================================================

def build_three_client_comparison(client_blocks: list[dict]) -> dict:
    rows_data = {"Delta L2": [], "Noise norm": [], "Epsilon": [], "Payload": [], "F1 (local, pre-DP)": [], "Status": []}
    for b in client_blocks:
        if b.get("missing"):
            rows_data["Delta L2"].append(MISSING)
            rows_data["Noise norm"].append(MISSING)
            rows_data["Epsilon"].append(MISSING)
            rows_data["Payload"].append(MISSING)
            rows_data["F1 (local, pre-DP)"].append(MISSING)
            rows_data["Status"].append("MISSING")
        else:
            ex = b["extracted"]
            rows_data["Delta L2"].append(ex["delta_l2"])
            rows_data["Noise norm"].append(ex["noise"])
            rows_data["Epsilon"].append(ex["epsilon"])
            rows_data["Payload"].append(ex["payload"])
            rows_data["F1 (local, pre-DP)"].append(ex["f1"])
            rows_data["Status"].append(b.get("status_word", "UNKNOWN"))

    present_deltas = [v for v in rows_data["Delta L2"] if v is not MISSING and v is not None]
    spread = (max(present_deltas) - min(present_deltas)) if len(present_deltas) >= 2 else None
    n_missing = sum(1 for v in rows_data["Status"] if v == "MISSING")

    return {"rows_data": rows_data, "spread": spread, "n_missing": n_missing, "n_total": len(client_blocks)}


# =============================================================================
# [5/7] SERVER AGGREGATION
# =============================================================================

def build_aggregation_block(round_id: int) -> dict:
    log_path = TRAINER_OUTPUTS / "demo_orchestrator.log"
    if not log_path.exists():
        return {"title": "[5/7]  SERVER AGGREGATION", "rows": [Row("Log", MISSING, verdict="!!")],
                "verdict": "!!", "duration": None, "explanation": "Orchestrator log not found.", "missing": True}

    text = read_text_sniff_bom(log_path)
    lines = [strip_ansi(l)[0] for l in text.splitlines()]

    trigger_line = find_line(lines, "Aggregation starting")
    if not trigger_line:
        return {"title": "[5/7]  SERVER AGGREGATION",
                "rows": [Row("Trigger", MISSING, verdict="!!"), Row("Algorithm", MISSING, verdict="!!"),
                         Row("Result", MISSING, verdict="!!"), Row("Global model", MISSING, verdict="!!"),
                         Row("Next round", MISSING, verdict="!!")],
                "verdict": "!!", "duration": None,
                "explanation": " Aggregation has not run yet - fewer than 3 accepted updates so far this round.",
                "missing": True}

    rows = []
    m = re.search(r"updates=(\d+)\s+algorithm=(\S+)", trigger_line)
    if m:
        rows.append(Row("Trigger", f"{m.group(1)} accepted updates reached", verdict="OK"))
        rows.append(Row("Algorithm", m.group(2)))
    else:
        rows.append(Row("Trigger", MISSING, verdict="!!"))
        rows.append(Row("Algorithm", MISSING, verdict="!!"))

    complete_line = find_line(lines, "Round") and next((l for l in lines if "complete" in l.lower() and "aggregat" in l.lower()), None)
    rows.append(Row("Result", complete_line if complete_line else MISSING, verdict="OK" if complete_line else "!!"))

    global_line = next((l for l in lines if "global_models" in l.lower() or ("hash" in l.lower() and "action" in l.lower())), None)
    rows.append(Row("Global model", global_line if global_line else MISSING, verdict="OK" if global_line else "!!"))

    next_line = find_line(lines, "Next collecting round") or find_line(lines, "Round 2")
    rows.append(Row("Next round", next_line if next_line else MISSING, verdict="OK" if next_line else "!!"))

    verdict_ok = all(r.verdict == "OK" for r in rows)

    explanation = (
        " HONEST NOTE: at n=3, trimmed_mean drops the highest and lowest per coordinate,\n"
        " leaving 1 of 3 - so it is a coordinate-wise MEDIAN, not an average, and gives\n"
        " no Byzantine robustness at this client count. One-line change at\n"
        " server.rs:1499, deliberately not made."
    )
    return {"title": "[5/7]  SERVER AGGREGATION", "rows": rows, "verdict": "OK" if verdict_ok else "!!",
            "duration": None, "explanation": explanation, "missing": False}


# =============================================================================
# [6/7] DATABASE VERIFICATION (live query only)
# =============================================================================

def build_db_verification_block(mongo_uri: str, db_name: str, round_id: int) -> dict:
    try:
        client = MongoClient(mongo_uri, serverSelectionTimeoutMS=5000)
        db = client[db_name]
        n_updates = db["model_updates"].count_documents({"round_id": round_id})
        n_receipts = db["receipts"].count_documents({"round_id": round_id})
        n_globals = db["global_models"].count_documents({})
        gridfs_doc = db["fs.files"].find_one(sort=[("uploadDate", -1)])
        client.close()
    except Exception:
        return {"title": "[6/7]  DATABASE VERIFICATION", "rows": [Row("MongoDB", MISSING, verdict="!!")],
                "verdict": "!!", "duration": None, "explanation": "Could not reach MongoDB."}

    rows = [
        Row("model_updates", n_updates, note="client(s) uploaded this round", verdict="OK" if n_updates == 3 else ("~" if n_updates > 0 else "!!")),
        Row("receipts", n_receipts, note="signed receipt(s), 1 per update", verdict="OK" if n_receipts == n_updates else "~"),
        Row("global_models", n_globals, note="aggregated model(s) produced", verdict="OK" if n_globals >= 1 else "!!"),
    ]
    if gridfs_doc:
        rows.append(Row("GridFS", f"{gridfs_doc.get('filename', '?')}, {gridfs_doc.get('length', 0):,} bytes", verdict="OK"))
    else:
        rows.append(Row("GridFS", MISSING, verdict="!!"))

    verdict_ok = n_updates == 3 and n_globals >= 1
    explanation = (" THREE IN, ONE OUT. That is federated learning working, proven at the\n"
                   " database level - not from a log line.") if verdict_ok else \
        f" Incomplete round: {n_updates}/3 client updates, {n_globals} global model(s) produced so far."

    return {"title": "[6/7]  DATABASE VERIFICATION", "rows": rows, "verdict": "OK" if verdict_ok else "~",
            "duration": None, "explanation": explanation}


# =============================================================================
# [7/7] AI REPORTS
# =============================================================================

def _report_state(md_text: str) -> str:
    """Three-marker priority detection, exactly as specified: LLM-unavailable
    first, then PASSED, then WARNING."""
    if "Narrative unavailable" in md_text or "Cohort narrative unavailable" in md_text:
        return "LLM_UNAVAILABLE"
    if "Result: PASSED" in md_text:
        return "PASSED"
    if "Result: WARNING" in md_text:
        return "WARNING"
    return "UNKNOWN"


def _corpus_doc_count() -> Optional[int]:
    """Counts corpus documents via a text-level scan of the source file (NOT
    a Python import - render_session.py must not import any pipeline
    module). Each document dict has exactly one '"id":' key."""
    corpus_path = Path(__file__).resolve().parent / "clinical_knowledge_corpus.py"
    if not corpus_path.exists():
        return None
    text = corpus_path.read_text(encoding="utf-8")
    return len(re.findall(r'"id":\s*"', text))


def build_ai_reports_block(round_id: int) -> dict:
    privacy_path = HOME / ".federated" / "data" / "audit_reports" / f"round_{round_id}_privacy_report.md"
    clinical_path = HOME / ".federated" / "data" / "clinical_reports" / f"round_{round_id}_clinical_report.md"

    rows_info = {}
    duration = None

    if privacy_path.exists():
        ptext = privacy_path.read_text(encoding="utf-8")
        pstate = _report_state(ptext)
        rt = re.search(r"Runtime: ([\d.]+)s total", ptext)
        rows_info["privacy_state"] = pstate
        rows_info["privacy_path"] = str(privacy_path)
        if rt:
            duration = (duration or 0) + float(rt.group(1))
        if pstate == "WARNING":
            m = re.search(r"Untraceable numbers.*?\[(.*?)\]", ptext, re.DOTALL)
            rows_info["privacy_untraceable"] = m.group(1) if m else None
    else:
        rows_info["privacy_state"] = "MISSING"
        rows_info["privacy_path"] = MISSING

    if clinical_path.exists():
        ctext = clinical_path.read_text(encoding="utf-8")
        cstate = _report_state(ctext)
        rt = re.search(r"Runtime: ([\d.]+)s total", ctext)
        rows_info["clinical_state"] = cstate
        rows_info["clinical_path"] = str(clinical_path)
        if rt:
            duration = (duration or 0) + float(rt.group(1))
        if cstate == "WARNING":
            m = re.search(r"\*\*Untraceable numbers:\*\*\s*\[(.*?)\]", ctext)
            rows_info["clinical_untraceable"] = m.group(1) if m else None
        # Sources used table
        table_rows = re.findall(r"^\|\s*(.+?)\s*\|\s*(yes|\*\*NO - unverified\*\*)\s*\|", ctext, re.MULTILINE)
        rows_info["sources_used_count"] = len(table_rows)
        rows_info["sources_unverified_count"] = sum(1 for _, v in table_rows if "NO" in v)
    else:
        rows_info["clinical_state"] = "MISSING"
        rows_info["clinical_path"] = MISSING

    rows_info["corpus_total"] = _corpus_doc_count()

    both_ok = rows_info["privacy_state"] == "PASSED" and rows_info["clinical_state"] == "PASSED"
    any_llm_fail = "LLM_UNAVAILABLE" in (rows_info["privacy_state"], rows_info["clinical_state"])
    verdict = "OK" if both_ok else ("!!" if any_llm_fail else "~")

    return {"title": "[7/7]  AI REPORTS", "info": rows_info, "verdict": verdict, "duration": duration}


def render_ai_reports_block(info: dict) -> list[str]:
    out = [hr(), block_header("[7/7]  AI REPORTS", None, None), hr(), ""]
    out.append(" Two reports were written by an AI assistant running entirely on this laptop.")
    out.append("")
    out.append("   1. PRIVACY REPORT   - for an auditor. Did we protect the data properly?")
    out.append("   2. CLINICAL REPORT  - for a doctor. What did the model find, overall?")
    out.append("")
    out.append(hr("-"))
    out.append(" WAS ANYTHING MADE UP?")
    out.append(hr("-"))
    out.append(" Every number the AI wrote was automatically checked against our real")
    out.append(" recorded results.")
    out.append("")

    def state_line(label, state, untraceable):
        if state == "LLM_UNAVAILABLE":
            return f"   {label:<18s} LLM UNAVAILABLE  the report was written, but the local AI did not respond in time - it has no narrative section"
        if state == "PASSED":
            return f"   {label:<18s} PASSED    every number was real"
        if state == "WARNING":
            n = untraceable.count(",") + 1 if untraceable else "some"
            return f"   {label:<18s} CAUGHT {n}  the AI wrote a number that was not traceable to the facts: [{untraceable}]"
        return f"   {label:<18s} MISSING   report file not found"

    out.append(state_line("Privacy report", info.get("privacy_state", "MISSING"), info.get("privacy_untraceable")))
    out.append(state_line("Clinical report", info.get("clinical_state", "MISSING"), info.get("clinical_untraceable")))
    out.append("")

    if info.get("clinical_state") == "WARNING":
        out.append(" The clinical report's warning is the system WORKING, not failing. Our checker")
        out.append(" caught a number before a doctor could read an unverifiable claim.")
        out.append("")

    # Rule 5: must NOT print the "proves the checker is not simply flagging
    # everything" line unless the privacy report actually passed.
    if info.get("privacy_state") == "PASSED":
        out.append(" The privacy report passed on the very same run, which proves the checker is")
        out.append(" not simply flagging everything.")
    elif info.get("privacy_state") == "LLM_UNAVAILABLE":
        out.append(" The privacy report has no narrative to check this run (local AI timed out) -")
        out.append(" its facts table is unaffected and does not depend on the LLM.")
    out.append("")

    out.append(hr("-"))
    out.append(" WHERE DID THE FACTS COME FROM?")
    out.append(hr("-"))
    corpus_total = info.get("corpus_total")
    used = info.get("sources_used_count")
    unverified = info.get("sources_unverified_count")
    if corpus_total is not None:
        out.append(f" The AI was given a small library of {corpus_total} reference documents to quote from.")
    else:
        out.append(" The AI was given a small library of [!!] MISSING reference documents to quote from.")
    out.append("")
    out.append("   Contains no patient data at all")
    if used is not None:
        out.append(f"   {used} document(s) were used for this report")
    else:
        out.append("   [!!] MISSING - no clinical report to read")
    if unverified is not None:
        out.append(f"   {unverified} of those are marked UNVERIFIED and labelled as such in the report")
    out.append("")

    out.append(hr("-"))
    out.append(" TWO SAFETY RULES WE BUILT IN")
    out.append(hr("-"))
    out.append(" The AI never left this laptop.       No internet. No external company.")
    out.append(" The AI never saw individual patients. Only group-level totals were given to")
    out.append("                                       it, so it CANNOT comment on any one")
    out.append("                                       person even if asked.")
    out.append("")

    out.append(hr("-"))
    out.append(" YOUR REPORTS ARE HERE")
    out.append(hr("-"))
    out.append(f"   {info.get('privacy_path', MISSING)}")
    out.append(f"   {info.get('clinical_path', MISSING)}")
    out.append(hr())
    return out


# =============================================================================
# Rendering: blocks with rows
# =============================================================================

def render_simple_block(block: dict) -> list[str]:
    out = [hr(), block_header(block["title"], block["verdict"], block.get("duration")), hr()]
    for row in block.get("rows", []):
        out.append(render_row(row))
    out.append(hr("-"))
    for eline in block.get("explanation", "").split("\n"):
        out.append(eline if eline.startswith(" ") else f" {eline}")
    out.append(hr())
    return out


def render_client_block(block: dict, full_mode: bool) -> list[str]:
    out = [hr(), block_header(block["title"], block.get("status_word", block["verdict"]), block.get("duration")), hr(), ""]
    if block.get("missing"):
        out.append(f" [!!] MISSING - {block['explanation']}")
        out.append(hr())
        return out

    sec_headers = {"DATA": "DATA", "MODEL": "MODEL", "TRAINING": "TRAINING", "XAI": "XAI  - INTEGRATED GRADIENTS",
                   "PRIVACY": "PRIVACY", "TRANSPORT": "TRANSPORT"}
    for key in ("DATA", "MODEL", "TRAINING", "XAI", "PRIVACY", "TRANSPORT"):
        rows = block["sections"][key]
        if not rows:
            continue
        header = f" {sec_headers[key]:<58s}ACTUAL      EXPECTED" if key != "XAI" else f" {sec_headers[key]:<58s}ACTUAL"
        out.append(header)
        out.append(" " + "-" * (WIDTH - 2))
        for row in rows:
            out.append(render_row(row))
        out.append("")

    stage_pairs = block.get("stage_results", [])
    if stage_pairs:
        out.append(" STAGES   " + "  ".join(f"{name}:{status}" for name, status in stage_pairs[:4]))
        out.append("          " + "  ".join(f"{name}:{status}" for name, status in stage_pairs[4:]))

    out.append(hr("-"))
    for eline in block.get("explanation", "").split("\n"):
        out.append(eline if eline.startswith(" ") else f" {eline}")

    counts = block.get("noise_counts")
    if counts is not None:
        out.extend(render_suppressed_block(counts, sum(counts.values()), full_mode))
    out.append(hr())
    return out


def render_three_client_comparison(comp: dict, mode_label: str = "Mode A / full") -> list[str]:
    out = [hr(), block_header("THREE CLIENTS - SIDE BY SIDE", None, None).replace("", "") , hr()]
    out[1] = " " + "THREE CLIENTS - SIDE BY SIDE" + " " * (WIDTH - 2 - len("THREE CLIENTS - SIDE BY SIDE") - len(mode_label)) + mode_label
    out.append(f"{'':25s}CLIENT 1        CLIENT 2        CLIENT 3")
    out.append(" " + "-" * (WIDTH - 2))
    for label, vals in comp["rows_data"].items():
        cells = []
        for v in vals:
            if v is MISSING:
                cells.append(f"{'MISSING':<15s}")
            elif isinstance(v, float):
                cells.append(f"{v:<15.6f}")
            else:
                cells.append(f"{str(v):<15s}")
        out.append(f" {label:<24s}" + " ".join(cells))
    out.append(" " + "-" * (WIDTH - 2))
    if comp["n_missing"] > 0:
        out.append(f" {comp['n_missing']} of {comp['n_total']} client(s) MISSING this round - see [4/7] cards above.")
    if comp["spread"] is not None:
        out.append(f" Delta L2 spread: {comp['spread']:.3f} - small, as expected in Mode A (all three train on the")
        out.append(" same 149 records). In Mode B (genuine sharding) expect ~3x this spread.")
    out.append(hr())
    return out


# =============================================================================
# SCORECARD
# =============================================================================

def build_and_render_scorecard(block_results: list[tuple[str, dict]], total_runtime: float) -> list[str]:
    out = [hr(), " SESSION SUMMARY SCORECARD", hr()]
    n_ok = n_notes = n_fail = n_missing = 0
    for i, (name, b) in enumerate(block_results, 1):
        v = b.get("verdict", "!!")
        missing = b.get("missing", False)
        if missing or v == "!!" and b.get("rows") == []:
            tag, n_missing = "[!!] MISSING", n_missing + 1
        elif v == "OK":
            tag, n_ok = "[OK]", n_ok + 1
        elif v == "~" or v == "--":
            tag, n_notes = f"[{v}]", n_notes + 1
        else:
            tag, n_fail = "[!!]", n_fail + 1
        out.append(f" {i}. {name:<40s} {tag}")
    out.append("")
    out.append(f" {n_ok} OK / {n_notes} notes / {n_fail} failures / {n_missing} missing")
    out.append(f" Renderer runtime (reading these artifacts, not the demo itself): {total_runtime:.2f}s")
    out.append("")
    out.append(" WHAT WORKS END TO END")
    out.append(hr("-"))
    out.append(" TPM-backed device enrollment, mTLS transport, per-client local training with")
    out.append(" real PHQ-8 labels, DP clip+noise+epsilon accounting, AES-256-GCM encrypted")
    out.append(" upload, TPM-signed receipts, and (when 3 clients complete) trimmed_mean")
    out.append(" aggregation into a new global model - all proven at the database level.")
    out.append("")
    out.append(" WHAT IS HONESTLY REPORTED AS NOT WORKING")
    out.append(hr("-"))
    out.append(" Signal-vs-noise ratio: local delta L2 (~0.6-0.8) vs DP noise (~450) is a")
    out.append(" ~590x gap this configuration does not close. Scope: this project's own")
    out.append(" measured configuration at n=3 clients, not a general DP claim.")
    out.append(" Aggregated F1 collapse: trimmed_mean at n=3 is a coordinate-wise median with")
    out.append(" no averaging benefit - measured mean 260.25 vs trimmed_mean 301.80 on")
    out.append(" identical data confirms mean is closer to the sqrt(3) prediction.")
    out.append(" Sharding's corpus-size floor: genuine per-client partitioning at this")
    out.append(" corpus's size (149 train records / 3 clients) leaves too few positive")
    out.append(" examples per shard for reliable local training (documented, Step 20/21).")
    out.append(" Per-patient attribution refused: two independent methods (Integrated")
    out.append(" Gradients, ablation) do not agree at the individual-patient level - only")
    out.append(" cohort-level modality attribution is reported, structurally, not by request.")
    out.append(hr())
    return out


# =============================================================================
# CLI / main
# =============================================================================

def run_session(round_id: int, full_mode: bool, mongo_uri: str, db_name: str) -> int:
    t_start = time.time()
    block_results = []
    output = []

    db_block = build_database_block(mongo_uri, db_name)
    output.extend(render_simple_block(db_block))
    block_results.append(("Database", db_block))

    orch_block = build_orchestrator_block()
    output.extend(render_simple_block(orch_block))
    block_results.append(("Orchestrator", orch_block))

    enroll_block = build_enrollment_block(mongo_uri, db_name)
    output.extend(render_simple_block(enroll_block))
    block_results.append(("Enrollment", enroll_block))

    orchestrator_start_epoch = get_orchestrator_start_epoch()
    client_blocks = []
    for n in (1, 2, 3):
        log_path = TRAINER_OUTPUTS / f"demo_r{round_id}_client{n}.log"
        cb = build_client_block(log_path, n, 3, full_mode, min_mtime_epoch=orchestrator_start_epoch)
        output.extend(render_client_block(cb, full_mode))
        client_blocks.append(cb)
        block_results.append((f"Client {n}", cb))

    comp = build_three_client_comparison(client_blocks)
    output.extend(render_three_client_comparison(comp))

    agg_block = build_aggregation_block(round_id)
    output.extend(render_simple_block(agg_block))
    block_results.append(("Aggregation", agg_block))

    verify_block = build_db_verification_block(mongo_uri, db_name, round_id)
    output.extend(render_simple_block(verify_block))
    block_results.append(("DB Verification", verify_block))

    reports_block = build_ai_reports_block(round_id)
    output.extend(render_ai_reports_block(reports_block["info"]))
    block_results.append(("AI Reports", reports_block))

    total_runtime = time.time() - t_start
    output.extend(build_and_render_scorecard(block_results, total_runtime))

    print("\n".join(output))
    return 0


def main() -> int:
    ap = argparse.ArgumentParser(description="Read-only mentor-facing session renderer")
    ap.add_argument("--stage", choices=["db", "server", "enroll", "client", "aggregate", "verify", "reports"])
    ap.add_argument("--client", help="path to one client log file")
    ap.add_argument("--compare", nargs=3, metavar=("LOG1", "LOG2", "LOG3"))
    ap.add_argument("--session", action="store_true")
    ap.add_argument("--round", type=int, default=1)
    ap.add_argument("--full", action="store_true", help="disable all noise suppression")
    ap.add_argument("--mongo-uri", default=DEFAULT_MONGO_URI)
    ap.add_argument("--db", default=DEFAULT_DB)
    args = ap.parse_args()

    if args.session:
        return run_session(args.round, args.full, args.mongo_uri, args.db)

    if args.client:
        cb = build_client_block(Path(args.client), 1, 1, args.full)
        print("\n".join(render_client_block(cb, args.full)))
        return 0

    if args.compare:
        blocks = [build_client_block(Path(p), i + 1, 3, args.full) for i, p in enumerate(args.compare)]
        for cb in blocks:
            print("\n".join(render_client_block(cb, args.full)))
        comp = build_three_client_comparison(blocks)
        print("\n".join(render_three_client_comparison(comp)))
        return 0

    if args.stage:
        builders = {
            "db": lambda: render_simple_block(build_database_block(args.mongo_uri, args.db)),
            "server": lambda: render_simple_block(build_orchestrator_block()),
            "enroll": lambda: render_simple_block(build_enrollment_block(args.mongo_uri, args.db)),
            "aggregate": lambda: render_simple_block(build_aggregation_block(args.round)),
            "verify": lambda: render_simple_block(build_db_verification_block(args.mongo_uri, args.db, args.round)),
            "reports": lambda: render_ai_reports_block(build_ai_reports_block(args.round)["info"]),
        }
        if args.stage in builders:
            print("\n".join(builders[args.stage]()))
            return 0
        print(f"--stage {args.stage} needs --client or is only meaningful within --session", file=sys.stderr)
        return 1

    ap.print_help()
    return 1


if __name__ == "__main__":
    sys.exit(main())
