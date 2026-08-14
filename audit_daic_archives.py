#!/usr/bin/env python3
"""
audit_daic_archives.py  (Phase 16 - Stage S0: multimodal dataset preflight)

READ-ONLY structural audit of the DAIC-WOZ archives and the staged transcripts.
This is the gate that must pass BEFORE any multimodal feature extraction runs.

It answers, reproducibly, the questions the builder depends on:

  * Which archives exist, which are readable, which participant IDs they cover.
  * Do all archives carry the 5 feature files the builder will consume?
  * Are the column widths uniform (AUs 24, gaze 16, pose 10, COVAREP 74, FORMANT 5)?
  * Do the CLNF files carry headers and do COVAREP/FORMANT NOT carry headers?
  * Are the duplicate "<pid>_P_2.zip" archives identical to their "<pid>_P.zip" twins?
  * Do the staged transcripts parse, and do they still match the archive bytes?
  * Does every participant actually speak (non-empty participant-speech intervals)?
  * DEEP: CLNF tracker success rate, COVAREP/FORMANT row counts, AUs/gaze/pose
    frame alignment, and the gating-safety invariant
        max(transcript stop_time) <= covarep_rows / 100
    which is what makes participant-only audio gating safe.

It NEVER writes to the dataset, NEVER extracts an archive to disk, and streams
every member directly out of the ZIP. The only thing it writes is its own JSON
report.

Inputs (read-only):
    <dataset>/<pid>_P.zip                        (participant archives)
    <dataset>/<pid>_P_2.zip                      (duplicate re-downloads, if any)
    <data-dir>/<pid>_P/<pid>_TRANSCRIPT.csv      (staged transcripts)
    <parquet>                                    (frozen text-only records, optional)

Output:
    <out>            JSON report (default ./dataset_build/archive_audit.json)
    stdout           human-readable summary
    exit code        0 = all checks passed, 1 = violations found, 2 = fatal setup error

Usage:
    python audit_daic_archives.py                     # full audit (deep, slow)
    python audit_daic_archives.py --quick             # structural only, no decompression
    python audit_daic_archives.py --limit 5 --quick   # fast smoke test
    python audit_daic_archives.py --out report.json

NOTE: run with the project venv (.venv/Scripts/python.exe) - the system
interpreter has neither pandas nor tqdm.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import statistics
import sys
import time
import zipfile
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

# tqdm is optional on purpose: the audit must run on a bare interpreter too.
# The module is kept alongside the class: tqdm.tqdm carries no __version__, so
# reporting the environment needs the module object.
try:
    import tqdm as _tqdm_mod
    from tqdm import tqdm as _tqdm
except ImportError:  # pragma: no cover - environment dependent
    _tqdm_mod = None
    _tqdm = None

# pandas is required only for the transcript checks; degrade loudly, not silently.
try:
    import pandas as pd
except ImportError:  # pragma: no cover - environment dependent
    pd = None


# --------------------------------------------------------------------------
# Frozen expectations. These are ASSERTIONS, not observations: the audit is a
# gate. Every value below was measured across all 188 readable archives before
# this script was written. A deviation means the dataset is not what the
# multimodal builder was planned against.
# --------------------------------------------------------------------------
DEFAULT_DATASET = Path("D:/datasets/DAIC-WOZ")
DEFAULT_DATA_DIR = Path("./data")
DEFAULT_PARQUET = Path("./daic_records.parquet")
DEFAULT_OUT = Path("./dataset_build/archive_audit.json")

# Members the multimodal builder will actually consume. Missing -> FAIL.
REQUIRED_MEMBERS: Dict[str, int] = {
    "CLNF_AUs.txt": 24,
    "CLNF_gaze.txt": 16,
    "CLNF_pose.txt": 10,
    "COVAREP.csv": 74,
    "FORMANT.csv": 5,
}
# Members expected to exist but not consumed by the builder. Missing -> WARN.
EXPECTED_EXTRA_MEMBERS = ("TRANSCRIPT.csv", "AUDIO.wav",
                          "CLNF_features.txt", "CLNF_features3D.txt")

# CLNF files carry a header line whose first field is "frame".
CLNF_MEMBERS = ("CLNF_AUs.txt", "CLNF_gaze.txt", "CLNF_pose.txt")
# COVAREP/FORMANT carry NO header - first row must be numeric.
HEADERLESS_MEMBERS = ("COVAREP.csv", "FORMANT.csv")

TRANSCRIPT_COLUMNS = ["start_time", "stop_time", "speaker", "value"]
KNOWN_SPEAKERS = {"ellie", "participant"}

COVAREP_HZ = 100.0                 # COVAREP has no timestamp column; t = row / 100
FORMANT_ROW_DELTA_MAX = 1          # |formant_rows - covarep_rows| must be <= 1
TRACKER_SUCCESS_FLOOR = 0.50       # below this a participant is unusable
GATING_SLACK_S = 0.5               # tolerance on participant-vs-COVAREP bound
CLIP_FAIL_S = 1.0                  # participant speech lost to clamping: FAIL above this
# Audio and video streams do not always span the same wall-clock window. Small
# skews are ubiquitous and harmless; large ones mean a modality only covers part
# of the session and must be surfaced (e.g. 420: video ends ~5.7 min before the
# audio; 402: video cut ~2 min early, which the AVEC documentation states).
SKEW_WARN_S = 60.0

EXPECTED_PARTICIPANT_ARCHIVES = 189   # 300..492 minus the documented exclusions
EXPECTED_READABLE = 188               # 440_P.zip is a truncated download
KNOWN_UNREADABLE = {"440"}            # documented, expected, not a surprise
KNOWN_NO_ELLIE = {"451", "458", "480"}  # AVEC doc: participant-only transcripts

ARCHIVE_RE = re.compile(r"^(\d+)_P\.zip$")
DUPLICATE_RE = re.compile(r"^(\d+)_P_2\.zip$")
NUMERIC_RE = re.compile(r"^-?(\d+\.?\d*|\.\d+)([eE][-+]?\d+)?$")

READ_BLOCK = 1 << 20  # 1 MiB streaming block for row counting

SEV_FAIL = "FAIL"
SEV_WARN = "WARN"
SEV_INFO = "INFO"


# --------------------------------------------------------------------------
# Reporting
# --------------------------------------------------------------------------
class Reporter:
    """Collects findings and prints them in the repository's [TAG] idiom.

    Output is ASCII-only: the Windows console here is cp1252 and non-ASCII
    characters raise UnicodeEncodeError mid-run.
    """

    def __init__(self, verbose: bool = False) -> None:
        self.findings: List[Dict[str, Any]] = []
        self.verbose = verbose

    def add(self, severity: str, check: str, message: str,
            pid: Optional[str] = None, **extra: Any) -> None:
        rec = {"severity": severity, "check": check, "pid": pid, "message": message}
        if extra:
            rec["detail"] = extra
        self.findings.append(rec)
        if severity == SEV_FAIL or (severity == SEV_WARN and self.verbose):
            where = f" [{pid}]" if pid else ""
            print(f"[{severity}]{where} {check}: {message}")

    def fail(self, check: str, message: str, pid: Optional[str] = None, **extra: Any) -> None:
        self.add(SEV_FAIL, check, message, pid, **extra)

    def warn(self, check: str, message: str, pid: Optional[str] = None, **extra: Any) -> None:
        self.add(SEV_WARN, check, message, pid, **extra)

    def info(self, check: str, message: str, pid: Optional[str] = None, **extra: Any) -> None:
        self.add(SEV_INFO, check, message, pid, **extra)

    def count(self, severity: str) -> int:
        return sum(1 for f in self.findings if f["severity"] == severity)

    def digest(self) -> str:
        """Deterministic digest of the findings.

        Timings and absolute paths are deliberately excluded so two runs over
        the same dataset produce the same digest.
        """
        canonical = sorted(
            (f["severity"], f["check"], f["pid"] or "", f["message"])
            for f in self.findings
        )
        blob = json.dumps(canonical, sort_keys=True, ensure_ascii=True)
        return hashlib.sha256(blob.encode("utf-8")).hexdigest()


def progress(items: Iterable, total: int, desc: str, enabled: bool) -> Iterable:
    """tqdm when available and wanted, otherwise a quiet periodic printer."""
    if enabled and _tqdm is not None:
        # ascii=True: the cp1252 console cannot render tqdm's default block glyphs.
        return _tqdm(items, total=total, desc=desc, unit="arch", ascii=True, ncols=78)
    if not enabled:
        return items

    def _plain():
        step = max(1, total // 20)
        for i, item in enumerate(items, 1):
            if i % step == 0 or i == total:
                pct = 100.0 * i / total if total else 100.0
                print(f"  {desc}: {i}/{total} ({pct:.0f}%)", flush=True)
            yield item

    return _plain()


# --------------------------------------------------------------------------
# Low-level ZIP streaming helpers - nothing here ever writes to disk
# --------------------------------------------------------------------------
def member_map(zf: zipfile.ZipFile) -> Dict[str, str]:
    """basename -> full archive path, for non-directory members."""
    out: Dict[str, str] = {}
    for name in zf.namelist():
        if name.endswith("/"):
            continue
        out[os.path.basename(name)] = name
    return out


def archive_fingerprint(zf: zipfile.ZipFile) -> str:
    """Deterministic identity of an archive from its central directory alone.

    Uses (name, uncompressed size, CRC-32) per member. This never decompresses
    anything, so it is cheap enough to run on every archive, yet it detects any
    content change. Chosen over hashing the .zip bytes because recompression
    would change the file hash while the contents stayed identical.
    """
    parts = sorted(
        f"{i.filename}:{i.file_size}:{i.CRC:08x}"
        for i in zf.infolist() if not i.filename.endswith("/")
    )
    return hashlib.sha256("|".join(parts).encode("utf-8")).hexdigest()


def first_line(zf: zipfile.ZipFile, member: str) -> str:
    with zf.open(member) as fh:
        raw = fh.readline()
    return raw.decode("utf-8", "replace").strip()


def count_rows_fast(zf: zipfile.ZipFile, member: str) -> int:
    """Exact line count via block reads.

    Used for COVAREP (37-51 MB per participant) where per-line iteration is the
    dominant cost of the whole audit. Handles a missing trailing newline.
    """
    total = 0
    last_byte = b""
    with zf.open(member) as fh:
        while True:
            block = fh.read(READ_BLOCK)
            if not block:
                break
            total += block.count(b"\n")
            last_byte = block[-1:]
    if last_byte and last_byte != b"\n":
        total += 1  # final line had no terminator
    return total


def scan_clnf(zf: zipfile.ZipFile, member: str,
              success_col: Optional[int] = None) -> Dict[str, Any]:
    """Single streaming pass over a CLNF file.

    Returns row count, first/last frame index, last timestamp and - when
    success_col is given - the tracker success rate. One pass, no buffering of
    the whole file.
    """
    rows = 0
    first_frame: Optional[int] = None
    last_frame: Optional[int] = None
    last_ts: Optional[float] = None
    ok = 0
    malformed = 0

    with zf.open(member) as fh:
        fh.readline()  # header
        for line in fh:
            parts = line.split(b",")
            if len(parts) < 4:
                malformed += 1
                continue
            rows += 1
            try:
                frame = int(parts[0])
                if first_frame is None:
                    first_frame = frame
                last_frame = frame
                last_ts = float(parts[1])
            except ValueError:
                malformed += 1
                continue
            if success_col is not None and parts[success_col].strip() == b"1":
                ok += 1

    return {
        "rows": rows,
        "first_frame": first_frame,
        "last_frame": last_frame,
        "last_timestamp": last_ts,
        "success_rate": (ok / rows) if (success_col is not None and rows) else None,
        "malformed": malformed,
    }


# --------------------------------------------------------------------------
# Discovery
# --------------------------------------------------------------------------
def discover_archives(dataset: Path, rep: Reporter) -> Tuple[Dict[str, Path], Dict[str, Path]]:
    """Split the dataset directory into participant archives and duplicates.

    Matching is by explicit regex rather than glob: '*_P.zip' happens to exclude
    '<pid>_P_2.zip' on this platform, but relying on that is a silent dependency
    on glob semantics. The regex makes the intent explicit and testable.
    """
    primary: Dict[str, Path] = {}
    duplicate: Dict[str, Path] = {}
    for path in sorted(dataset.iterdir()):
        if not path.is_file():
            continue
        m = ARCHIVE_RE.match(path.name)
        if m:
            primary[m.group(1)] = path
            continue
        d = DUPLICATE_RE.match(path.name)
        if d:
            duplicate[d.group(1)] = path

    rep.info("A1.discover", f"{len(primary)} participant archives, "
                            f"{len(duplicate)} duplicate archives")
    if len(primary) != EXPECTED_PARTICIPANT_ARCHIVES:
        rep.warn("A1.count", f"expected {EXPECTED_PARTICIPANT_ARCHIVES} participant "
                             f"archives, found {len(primary)}")
    return primary, duplicate


# --------------------------------------------------------------------------
# Structural checks (cheap: central directory + first line only)
# --------------------------------------------------------------------------
def audit_structure(pid: str, path: Path, rep: Reporter) -> Optional[Dict[str, Any]]:
    """Open one archive and validate its inventory, headers and column widths.

    Returns None when the archive cannot be opened (recorded as FAIL/INFO), so
    the caller can keep going. One bad archive must never abort the audit.
    """
    try:
        zf = zipfile.ZipFile(path, "r")   # read-only; never 'a' or 'w'
    except zipfile.BadZipFile:
        if pid in KNOWN_UNREADABLE:
            rep.info("A2.unreadable", "corrupt archive (documented, expected)", pid)
        else:
            rep.fail("A2.unreadable", "corrupt or truncated archive", pid)
        return None
    except OSError as exc:
        rep.fail("A2.unreadable", f"cannot open: {type(exc).__name__}: {exc}", pid)
        return None

    with zf:
        members = member_map(zf)
        info: Dict[str, Any] = {
            "pid": pid,
            "archive": path.name,
            "size_bytes": path.stat().st_size,
            "fingerprint": archive_fingerprint(zf),
            "member_count": len(members),
            "members": sorted(members),
            "columns": {},
        }

        # -- A3: required members present -----------------------------------
        for kind in REQUIRED_MEMBERS:
            if f"{pid}_{kind}" not in members:
                rep.fail("A3.required_member", f"missing {pid}_{kind}", pid)
        for kind in EXPECTED_EXTRA_MEMBERS:
            if f"{pid}_{kind}" not in members:
                rep.warn("A3.expected_member", f"missing {pid}_{kind}", pid)

        # HOG ships as .bin for 197 archives and .txt for one. The builder does
        # not consume it, so this is recorded, never enforced.
        hog = [m for m in members if "CLNF_hog" in m]
        if hog:
            info["hog_member"] = hog[0]

        unexpected = [
            m for m in members
            if not any(m == f"{pid}_{k}" for k in REQUIRED_MEMBERS)
            and not any(m == f"{pid}_{k}" for k in EXPECTED_EXTRA_MEMBERS)
            and "CLNF_hog" not in m
        ]
        if unexpected:
            # e.g. the macOS resource fork '._487_TRANSCRIPT.csv'. Harmless.
            rep.info("A3.unexpected_member", f"extra members: {unexpected}", pid)
            info["unexpected_members"] = unexpected

        # -- A4/A5: column widths and header presence -----------------------
        for kind, expected_cols in REQUIRED_MEMBERS.items():
            key = f"{pid}_{kind}"
            if key not in members:
                continue
            try:
                line = first_line(zf, members[key])
            except (zipfile.BadZipFile, OSError, EOFError) as exc:
                rep.fail("A4.read", f"{kind}: unreadable ({type(exc).__name__})", pid)
                continue

            fields = line.split(",")
            info["columns"][kind] = len(fields)
            if len(fields) != expected_cols:
                rep.fail("A4.columns",
                         f"{kind}: {len(fields)} columns, expected {expected_cols}",
                         pid)

            head = fields[0].strip()
            if kind in CLNF_MEMBERS and head.lower() != "frame":
                rep.fail("A5.header",
                         f"{kind}: expected header starting 'frame', got {head!r}", pid)
            if kind in HEADERLESS_MEMBERS and not NUMERIC_RE.match(head):
                rep.fail("A5.header",
                         f"{kind}: expected headerless numeric first row, got {head!r}",
                         pid)

    return info


def audit_duplicates(primary: Dict[str, Path], duplicate: Dict[str, Path],
                     audited: set, readable: set, rep: Reporter) -> Dict[str, Any]:
    """Confirm every '<pid>_P_2.zip' is content-identical to its '<pid>_P.zip'.

    If they ever diverge, the builder's choice of archive would silently matter
    and the dataset would be ambiguous. Comparison is by (name, size, CRC) from
    the central directory - no decompression.

    'primary' must be the COMPLETE archive map, not the audited subset:
    orphan-detection is a statement about the dataset, and scoping it to a
    --limit subset (or to the readable subset) would invent orphans that do not
    exist. Only the comparison itself is scoped to what was actually audited.
    """
    result = {"checked": 0, "identical": 0, "divergent": [],
              "orphans": [], "skipped": []}
    for pid in sorted(duplicate):
        if pid not in primary:
            result["orphans"].append(pid)
            rep.warn("A6.orphan_duplicate",
                     "duplicate archive with no primary twin", pid)
            continue
        if pid not in audited:
            result["skipped"].append(pid)   # outside --limit; not a finding
            continue
        if pid not in readable:
            rep.warn("A6.duplicate_unreadable",
                     "primary twin is unreadable - cannot compare", pid)
            continue
        try:
            with zipfile.ZipFile(primary[pid], "r") as za, \
                 zipfile.ZipFile(duplicate[pid], "r") as zb:
                fa, fb = archive_fingerprint(za), archive_fingerprint(zb)
        except (zipfile.BadZipFile, OSError) as exc:
            rep.warn("A6.duplicate_unreadable",
                     f"cannot compare: {type(exc).__name__}", pid)
            continue
        result["checked"] += 1
        if fa == fb:
            result["identical"] += 1
        else:
            result["divergent"].append(pid)
            rep.fail("A6.divergent_duplicate",
                     "duplicate archive differs from primary - dataset is ambiguous",
                     pid)
    return result


# --------------------------------------------------------------------------
# Transcript checks (staged copies under ./data)
# --------------------------------------------------------------------------
def find_transcript(folder: Path) -> Optional[Path]:
    """Case-insensitive lookup, matching build_daic_records.find_transcript_csv."""
    if not folder.is_dir():
        return None
    for child in sorted(folder.iterdir()):
        if child.is_file() and child.name.lower().endswith("_transcript.csv"):
            return child
    return None


def audit_transcript(pid: str, data_dir: Path, archive: Optional[Path],
                     rep: Reporter) -> Optional[Dict[str, Any]]:
    """Validate one staged transcript and its participant-speech coverage.

    Also compares the staged bytes against the archive member: ./data is a
    staged copy, and drift between it and the archive would silently corrupt
    audio gating without any other check noticing.
    """
    if pd is None:
        return None

    folder = data_dir / f"{pid}_P"
    tpath = find_transcript(folder)
    if tpath is None:
        rep.fail("A7.missing", f"no staged transcript under {folder.name}", pid)
        return None

    try:
        df = pd.read_csv(tpath, sep="\t")
    except Exception as exc:  # noqa: BLE001 - report and continue
        rep.fail("A7.parse", f"unreadable: {type(exc).__name__}: {exc}", pid)
        return None

    cols = [c.strip() for c in df.columns]
    if cols != TRANSCRIPT_COLUMNS:
        rep.fail("A7.columns", f"columns {cols} != {TRANSCRIPT_COLUMNS}", pid)
        return None
    df.columns = cols

    # -- A7b: staged bytes still match the archive member --------------------
    byte_match: Optional[bool] = None
    if archive is not None:
        try:
            with zipfile.ZipFile(archive, "r") as zf:
                members = member_map(zf)
                key = f"{pid}_TRANSCRIPT.csv"
                if key in members:
                    with zf.open(members[key]) as fh:
                        byte_match = (fh.read() == tpath.read_bytes())
        except (zipfile.BadZipFile, OSError):
            byte_match = None
        if byte_match is False:
            rep.fail("A7b.drift",
                     "staged transcript differs from archive member", pid)

    speaker = df["speaker"].astype(str).str.strip()
    unknown = sorted(set(speaker.str.lower()) - KNOWN_SPEAKERS)
    if unknown:
        rep.fail("A8.speaker", f"unknown speaker labels: {unknown}", pid)

    has_ellie = "ellie" in set(speaker.str.lower())
    if not has_ellie and pid not in KNOWN_NO_ELLIE:
        rep.warn("A8.no_ellie", "transcript has no Ellie turns (undocumented)", pid)

    part = df[speaker.str.lower() == "participant"]
    if part.empty:
        rep.fail("A9.no_participant",
                 "no Participant turns - audio gating would yield nothing", pid)
        return None

    try:
        starts = part["start_time"].astype(float)
        stops = part["stop_time"].astype(float)
    except (TypeError, ValueError) as exc:
        rep.fail("A9.timestamps", f"non-numeric timestamps: {exc}", pid)
        return None

    if bool((stops < starts).any()):
        rep.fail("A9.interval", "stop_time before start_time in a Participant turn", pid)

    # Merge overlapping/adjacent intervals: the AVEC transcription manual uses
    # overlapping timestamps to mark speech overlap, so the raw sum would
    # double-count those frames.
    raw = sorted(zip(starts.tolist(), stops.tolist()))
    merged: List[List[float]] = []
    for s, e in raw:
        if merged and s <= merged[-1][1]:
            merged[-1][1] = max(merged[-1][1], e)
        else:
            merged.append([s, e])
    speech_s = sum(e - s for s, e in merged)

    return {
        "pid": pid,
        "rows": int(len(df)),
        "participant_turns": int(len(part)),
        "merged_intervals": len(merged),
        "speech_seconds": round(speech_s, 3),
        "raw_speech_seconds": round(float((stops - starts).sum()), 3),
        # transcript_max_stop spans BOTH speakers and is informational only.
        # participant_max_stop is the quantity audio gating actually depends on:
        # Ellie's turns are never sampled, so an Ellie turn running past the end
        # of the audio stream is irrelevant to the builder.
        "transcript_max_stop": round(float(df["stop_time"].astype(float).max()), 3),
        "participant_max_stop": round(float(stops.max()), 3),
        "has_ellie": bool(has_ellie),
        "matches_archive_bytes": byte_match,
        # Kept in memory for the deep pass, stripped before serialisation:
        # 188 x ~150 intervals would bloat the report for no benefit.
        "_merged": merged,
    }


# --------------------------------------------------------------------------
# Deep checks (full decompression - the expensive pass)
# --------------------------------------------------------------------------
def audit_deep(pid: str, path: Path, transcript: Optional[Dict[str, Any]],
               rep: Reporter) -> Optional[Dict[str, Any]]:
    """Row counts, tracker success, frame alignment and the gating invariant."""
    try:
        zf = zipfile.ZipFile(path, "r")
    except (zipfile.BadZipFile, OSError):
        return None  # already reported by audit_structure

    with zf:
        members = member_map(zf)

        def need(kind: str) -> Optional[str]:
            return members.get(f"{pid}_{kind}")

        cov_m, fmt_m = need("COVAREP.csv"), need("FORMANT.csv")
        au_m, gaze_m, pose_m = need("CLNF_AUs.txt"), need("CLNF_gaze.txt"), need("CLNF_pose.txt")
        if not all([cov_m, fmt_m, au_m, gaze_m, pose_m]):
            return None  # missing members already reported as FAIL

        try:
            cov_rows = count_rows_fast(zf, cov_m)
            fmt_rows = count_rows_fast(zf, fmt_m)
            au = scan_clnf(zf, au_m, success_col=3)
            gaze = scan_clnf(zf, gaze_m)
            pose = scan_clnf(zf, pose_m)
        except (zipfile.BadZipFile, OSError, EOFError) as exc:
            rep.fail("A10.read", f"decompression failed: {type(exc).__name__}: {exc}", pid)
            return None

        for name, scan in (("AUs", au), ("gaze", gaze), ("pose", pose)):
            if scan["malformed"]:
                rep.warn("A13.malformed",
                         f"{name}: {scan['malformed']} malformed line(s)", pid)

        # -- A10: tracker success -------------------------------------------
        success = au["success_rate"]
        if success is None:
            rep.fail("A10.success", "could not compute tracker success rate", pid)
        elif success < TRACKER_SUCCESS_FLOOR:
            rep.fail("A10.success",
                     f"tracker success {success:.3f} below floor {TRACKER_SUCCESS_FLOOR}",
                     pid)

        # -- A11: FORMANT/COVAREP row agreement ------------------------------
        delta = fmt_rows - cov_rows
        if abs(delta) > FORMANT_ROW_DELTA_MAX:
            rep.fail("A11.rows",
                     f"FORMANT-COVAREP row delta {delta} exceeds "
                     f"+/-{FORMANT_ROW_DELTA_MAX}", pid)

        # -- A13: AUs/gaze/pose frame alignment ------------------------------
        if not (au["rows"] == gaze["rows"] == pose["rows"]):
            rep.fail("A13.alignment",
                     f"row counts differ: AUs={au['rows']} gaze={gaze['rows']} "
                     f"pose={pose['rows']}", pid)
        if not (au["first_frame"] == gaze["first_frame"] == pose["first_frame"]):
            rep.fail("A13.alignment", "first frame index differs across CLNF files", pid)
        if not (au["last_frame"] == gaze["last_frame"] == pose["last_frame"]):
            rep.fail("A13.alignment", "last frame index differs across CLNF files", pid)

        # -- A12: the gating-safety invariant --------------------------------
        # COVAREP carries no timestamp column, so the builder must map row i to
        # t = i / 100. Participant-only gating is only sound if the transcript
        # clock stays inside that span.
        # The bound is on PARTICIPANT intervals, not on the transcript as a
        # whole. Measuring the whole transcript conflates Ellie's closing turn
        # with a real overrun: participant 402's transcript ends at 972.25s
        # against a 955.50s audio stream, yet its last participant turn ends at
        # 954.64s, so nothing the builder samples is affected.
        cov_duration = cov_rows / COVAREP_HZ
        gating_ok: Optional[bool] = None
        clipped_s: Optional[float] = None
        if transcript is not None:
            pmax = transcript["participant_max_stop"]
            gating_ok = pmax <= cov_duration + GATING_SLACK_S

            # Exact participant speech that clamping to the audio stream removes.
            clipped_s = 0.0
            for s, e in transcript.get("_merged", []):
                if e > cov_duration:
                    clipped_s += e - max(s, cov_duration)
            clipped_s = round(clipped_s, 3)

            if clipped_s > CLIP_FAIL_S:
                rep.fail("A12.gating",
                         f"clamping to the audio stream would discard "
                         f"{clipped_s:.2f}s of participant speech "
                         f"(participant ends {pmax:.2f}s, COVAREP spans "
                         f"{cov_duration:.2f}s)", pid)
            elif clipped_s > 0.0:
                rep.warn("A12.gating",
                         f"clamping discards {clipped_s:.3f}s of participant "
                         f"speech (below the {CLIP_FAIL_S}s failure threshold)", pid)
            elif not gating_ok:
                rep.warn("A12.gating",
                         f"last participant turn ends {pmax:.2f}s, past the "
                         f"{cov_duration:.2f}s audio stream, but no speech is lost",
                         pid)

        # Audio and video do not always span the same window. This is a real
        # property of the corpus, not a defect: it means a modality covers only
        # part of the session for that participant. Surfaced, never blocking.
        av_skew = None
        if au["last_timestamp"] is not None:
            av_skew = round(cov_duration - au["last_timestamp"], 3)
            if abs(av_skew) > SKEW_WARN_S:
                shorter = "video" if av_skew > 0 else "audio"
                rep.warn("A15.stream_skew",
                         f"{shorter} stream is {abs(av_skew):.1f}s shorter than the "
                         f"other (audio {cov_duration:.1f}s vs video "
                         f"{au['last_timestamp']:.1f}s)", pid)

        return {
            "pid": pid,
            "covarep_rows": cov_rows,
            "formant_rows": fmt_rows,
            "formant_delta": delta,
            "covarep_seconds": round(cov_duration, 3),
            "clnf_rows": au["rows"],
            "clnf_first_frame": au["first_frame"],
            "clnf_last_frame": au["last_frame"],
            "clnf_last_timestamp": au["last_timestamp"],
            "tracker_success": round(success, 5) if success is not None else None,
            "av_skew_seconds": av_skew,
            "gating_ok": gating_ok,
            "clipped_participant_seconds": clipped_s,
        }


# --------------------------------------------------------------------------
# Summary helpers
# --------------------------------------------------------------------------
def describe(values: List[float]) -> Dict[str, float]:
    if not values:
        return {}
    return {
        "min": round(min(values), 4),
        "median": round(statistics.median(values), 4),
        "max": round(max(values), 4),
        "n": len(values),
    }


def environment() -> Dict[str, Any]:
    def ver(mod: Any) -> Optional[str]:
        return getattr(mod, "__version__", None) if mod is not None else None
    return {
        "python": sys.version.split()[0],
        "platform": sys.platform,
        "pandas": ver(pd),
        "tqdm": ver(_tqdm_mod),
    }


# --------------------------------------------------------------------------
# Main
# --------------------------------------------------------------------------
def main() -> int:
    ap = argparse.ArgumentParser(
        description="Read-only structural audit of the DAIC-WOZ archives (Stage S0)")
    ap.add_argument("--dataset", type=Path, default=DEFAULT_DATASET,
                    help=f"folder holding <pid>_P.zip (default: {DEFAULT_DATASET})")
    ap.add_argument("--data-dir", type=Path, default=DEFAULT_DATA_DIR,
                    help=f"staged transcript root (default: {DEFAULT_DATA_DIR})")
    ap.add_argument("--parquet", type=Path, default=DEFAULT_PARQUET,
                    help="frozen text-only records, for the bijection check")
    ap.add_argument("--out", type=Path, default=DEFAULT_OUT,
                    help=f"JSON report path (default: {DEFAULT_OUT})")
    ap.add_argument("--quick", action="store_true",
                    help="skip the deep pass (no decompression); structure only")
    ap.add_argument("--limit", type=int, default=0,
                    help="audit only the first N archives (0 = all); smoke tests")
    ap.add_argument("--pids", default="",
                    help="comma-separated participant IDs to audit (e.g. 300,440,451); "
                         "for targeted re-checks after a failure")
    ap.add_argument("--no-progress", action="store_true", help="disable progress bars")
    ap.add_argument("--no-parquet-check", action="store_true",
                    help="skip the archive/parquet participant bijection check")
    ap.add_argument("--verbose", action="store_true", help="print WARN findings too")
    args = ap.parse_args()

    t_start = time.time()
    rep = Reporter(verbose=args.verbose)

    # ---- fatal setup checks (exit 2) --------------------------------------
    if not args.dataset.is_dir():
        print(f"[FATAL] dataset folder not found: {args.dataset}")
        return 2
    out_path = args.out.resolve()
    try:
        dataset_root = args.dataset.resolve()
        if dataset_root == out_path.parent or dataset_root in out_path.parents:
            print(f"[FATAL] refusing to write the report inside the dataset: {out_path}")
            return 2
    except OSError as exc:
        print(f"[FATAL] cannot resolve paths: {exc}")
        return 2
    if pd is None:
        print("[FATAL] pandas is required for the transcript checks. "
              "Run with .venv/Scripts/python.exe")
        return 2

    print("=" * 78)
    print("DAIC-WOZ ARCHIVE AUDIT (Stage S0) - READ-ONLY")
    print("=" * 78)
    print(f"[INFO] dataset   : {dataset_root}")
    print(f"[INFO] data-dir  : {args.data_dir.resolve()}")
    print(f"[INFO] report    : {out_path}")
    print(f"[INFO] mode      : {'QUICK (structure only)' if args.quick else 'FULL (deep pass)'}")
    env = environment()
    print(f"[INFO] python    : {env['python']}  pandas={env['pandas']}  tqdm={env['tqdm']}")
    if _tqdm is None and not args.no_progress:
        print("[WARN] tqdm not installed - falling back to plain progress output")
    print("-" * 78)

    # ---- discovery ---------------------------------------------------------
    primary, duplicate = discover_archives(dataset_root, rep)
    if not primary:
        print(f"[FATAL] no <pid>_P.zip archives found in {dataset_root}")
        return 2

    pids = sorted(primary, key=lambda p: int(p))
    if args.pids:
        wanted = [p.strip() for p in args.pids.split(",") if p.strip()]
        unknown = [p for p in wanted if p not in primary]
        if unknown:
            print(f"[FATAL] --pids: no archive for {unknown}")
            return 2
        pids = [p for p in pids if p in set(wanted)]
        print(f"[INFO] --pids: auditing {len(pids)} archive(s): {pids}")
    if args.limit:
        pids = pids[:args.limit]
        print(f"[INFO] --limit {args.limit}: auditing {len(pids)} archive(s)")

    # ---- structural pass ---------------------------------------------------
    print(f"[STEP 1/4] structural audit of {len(pids)} archive(s)")
    structures: Dict[str, Dict[str, Any]] = {}
    unreadable: List[str] = []
    for pid in progress(pids, len(pids), "structure", not args.no_progress):
        info = audit_structure(pid, primary[pid], rep)
        if info is None:
            unreadable.append(pid)
        else:
            structures[pid] = info
    print(f"[ OK ] readable={len(structures)}  unreadable={len(unreadable)} {unreadable}")

    # Corpus-level expectations are only meaningful over the whole dataset.
    subset = bool(args.limit or args.pids)
    if not subset and len(structures) != EXPECTED_READABLE:
        rep.warn("A2.readable_count",
                 f"expected {EXPECTED_READABLE} readable archives, "
                 f"found {len(structures)}")
    unexpected_bad = sorted(set(unreadable) - KNOWN_UNREADABLE)
    if unexpected_bad:
        rep.fail("A2.unexpected_unreadable",
                 f"undocumented unreadable archives: {unexpected_bad}")

    # ---- duplicates --------------------------------------------------------
    print("[STEP 2/4] duplicate-archive comparison")
    dup = audit_duplicates(primary, duplicate, set(pids), set(structures), rep)
    print(f"[ OK ] duplicates checked={dup['checked']} identical={dup['identical']} "
          f"divergent={dup['divergent']} skipped={len(dup['skipped'])}")

    # ---- transcripts -------------------------------------------------------
    print(f"[STEP 3/4] staged transcript audit ({args.data_dir})")
    transcripts: Dict[str, Dict[str, Any]] = {}
    if not args.data_dir.is_dir():
        rep.fail("A7.data_dir", f"staged transcript root not found: {args.data_dir}")
    else:
        for pid in progress(sorted(structures, key=lambda p: int(p)),
                            len(structures), "transcripts", not args.no_progress):
            t = audit_transcript(pid, args.data_dir, primary.get(pid), rep)
            if t is not None:
                transcripts[pid] = t
    speech = [t["speech_seconds"] for t in transcripts.values()]
    drift = [p for p, t in transcripts.items() if t["matches_archive_bytes"] is False]
    print(f"[ OK ] transcripts={len(transcripts)}  "
          f"speech_s={describe(speech)}  byte_drift={drift}")

    # ---- deep pass ---------------------------------------------------------
    deep: Dict[str, Dict[str, Any]] = {}
    if args.quick:
        print("[STEP 4/4] deep pass SKIPPED (--quick)")
        rep.warn("A0.quick",
                 "deep pass skipped: row counts, tracker success, frame alignment "
                 "and the gating invariant were NOT verified")
    else:
        print(f"[STEP 4/4] deep pass over {len(structures)} archive(s) "
              f"(full decompression; expect tens of minutes)")
        for pid in progress(sorted(structures, key=lambda p: int(p)),
                            len(structures), "deep", not args.no_progress):
            d = audit_deep(pid, primary[pid], transcripts.get(pid), rep)
            if d is not None:
                deep[pid] = d
        succ = [d["tracker_success"] for d in deep.values() if d["tracker_success"] is not None]
        skew = [abs(d["av_skew_seconds"]) for d in deep.values() if d["av_skew_seconds"] is not None]
        clip = [d["clipped_participant_seconds"] for d in deep.values()
                if d.get("clipped_participant_seconds") is not None]
        print(f"[ OK ] deep={len(deep)}  tracker_success={describe(succ)}")
        print(f"[ OK ] |audio-video skew| seconds={describe(skew)}")
        print(f"[ OK ] participant speech lost to clamping: "
              f"total={sum(clip):.3f}s  max={max(clip) if clip else 0.0:.3f}s")

    # ---- bijection against the frozen parquet ------------------------------
    bijection: Dict[str, Any] = {"checked": False}
    if not args.no_parquet_check and not subset:
        if not args.parquet.is_file():
            rep.warn("A14.parquet", f"frozen parquet not found: {args.parquet}")
        else:
            try:
                frozen = pd.read_parquet(args.parquet)
                frozen_pids = {str(x).strip() for x in frozen["participant_id"]}
                archive_pids = set(structures)
                missing = sorted(frozen_pids - archive_pids, key=int)
                extra = sorted(archive_pids - frozen_pids, key=int)
                bijection = {
                    "checked": True,
                    "parquet_participants": len(frozen_pids),
                    "readable_archives": len(archive_pids),
                    "in_parquet_no_archive": missing,
                    "archive_not_in_parquet": extra,
                }
                if missing:
                    rep.fail("A14.bijection",
                             f"parquet participants with no readable archive: {missing}")
                if extra:
                    rep.fail("A14.bijection",
                             f"readable archives absent from the parquet: {extra}")
                print(f"[ OK ] bijection: parquet={len(frozen_pids)} "
                      f"archives={len(archive_pids)} missing={missing} extra={extra}")
            except Exception as exc:  # noqa: BLE001 - report and continue
                rep.warn("A14.parquet", f"could not read parquet: "
                                        f"{type(exc).__name__}: {exc}")

    # ---- report ------------------------------------------------------------
    elapsed = time.time() - t_start
    n_fail, n_warn = rep.count(SEV_FAIL), rep.count(SEV_WARN)

    report = {
        "schema": "daic_archive_audit/1",
        "mode": "quick" if args.quick else "full",
        "environment": env,
        "inputs": {
            "dataset": str(dataset_root),
            "data_dir": str(args.data_dir.resolve()),
            "parquet": str(args.parquet.resolve()) if args.parquet.is_file() else None,
            "limit": args.limit,
        },
        "expectations": {
            "required_members": REQUIRED_MEMBERS,
            "covarep_hz": COVAREP_HZ,
            "formant_row_delta_max": FORMANT_ROW_DELTA_MAX,
            "tracker_success_floor": TRACKER_SUCCESS_FLOOR,
            "gating_slack_s": GATING_SLACK_S,
        },
        "totals": {
            "participant_archives": len(primary),
            "duplicate_archives": len(duplicate),
            "audited": len(pids),
            "readable": len(structures),
            "unreadable": sorted(unreadable, key=int),
            "transcripts": len(transcripts),
            "deep": len(deep),
            "fail": n_fail,
            "warn": n_warn,
        },
        "duplicates": dup,
        "bijection": bijection,
        "summary_statistics": {
            "speech_seconds": describe(speech),
            "tracker_success": describe([d["tracker_success"] for d in deep.values()
                                         if d["tracker_success"] is not None]),
            "clnf_rows": describe([float(d["clnf_rows"]) for d in deep.values()]),
            "covarep_rows": describe([float(d["covarep_rows"]) for d in deep.values()]),
            "av_skew_seconds": describe([abs(d["av_skew_seconds"]) for d in deep.values()
                                         if d["av_skew_seconds"] is not None]),
            "clipped_participant_seconds": describe(
                [d["clipped_participant_seconds"] for d in deep.values()
                 if d.get("clipped_participant_seconds") is not None]),
        },
        "per_participant": {
            pid: {
                "structure": structures.get(pid),
                # Underscore-prefixed keys are working state (merged intervals),
                # deliberately not serialised.
                "transcript": {k: v for k, v in (transcripts.get(pid) or {}).items()
                               if not k.startswith("_")} or None,
                "deep": deep.get(pid),
            } for pid in sorted(structures, key=int)
        },
        "findings": rep.findings,
        "findings_digest": rep.digest(),
        "elapsed_seconds": round(elapsed, 1),
    }

    try:
        out_path.parent.mkdir(parents=True, exist_ok=True)
        with open(out_path, "w", encoding="utf-8") as fh:
            json.dump(report, fh, indent=2, sort_keys=True, ensure_ascii=True)
    except OSError as exc:
        print(f"[FATAL] cannot write report to {out_path}: {exc}")
        return 2

    print("-" * 78)
    print("=== AUDIT SUMMARY ===")
    print(f"archives audited   : {len(pids)}")
    print(f"readable           : {len(structures)}   unreadable: {sorted(unreadable, key=int)}")
    print(f"transcripts parsed : {len(transcripts)}")
    print(f"deep-scanned       : {len(deep)}")
    print(f"FAIL findings      : {n_fail}")
    print(f"WARN findings      : {n_warn}")
    print(f"findings digest    : {rep.digest()}")
    print(f"elapsed            : {elapsed:.1f}s")
    print(f"report             : {out_path}")

    if n_fail:
        print("\n[FAIL] audit found violations - DO NOT proceed to feature extraction.")
        by_check: Dict[str, int] = {}
        for f in rep.findings:
            if f["severity"] == SEV_FAIL:
                by_check[f["check"]] = by_check.get(f["check"], 0) + 1
        for check, n in sorted(by_check.items()):
            print(f"    {check}: {n}")
        return 1

    if args.quick:
        print("\n[PASS] structural checks passed. Deep checks were SKIPPED (--quick); "
              "a full run is required before feature extraction.")
    else:
        print("\n[PASS] all archive, transcript and deep checks passed.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
