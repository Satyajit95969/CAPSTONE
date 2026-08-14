#!/usr/bin/env python3
"""
build_daic_multimodal_features.py  (Phase 16 - Stage S2: feature extraction)

Streams the DAIC-WOZ archives and produces, for every participant, the frozen
154-dimensional audio vector and 84-dimensional vision vector that Stage S3 will
attach to the multimodal parquet.

NOTHING IS EXTRACTED TO DISK. Every member is read directly out of its ZIP.
HOG IS NEVER OPENED - it is 348-489 MB per participant and is not part of the
agreed design.

AUDIO  (COVAREP + FORMANT -> 154 dims)
--------------------------------------
COVAREP ships 74 headerless columns. The AVEC documentation names 73; the 74th
was resolved empirically (see COVAREP_LAYOUT below). Two columns are gates or
metadata rather than features and are excluded, leaving 72:

    index 1 = VUV      voicing flag, used as a gate
    index 9 = Rd_conf  a confidence, not a measurement

Gating order, applied exactly as agreed:
    1. truncate COVAREP and FORMANT to their minimum shared length
       (Stage S0 measured the row-count delta as -1, 0 or +1)
    2. participant-interval gate: row i covers [i/100, (i+1)/100); it is kept
       when its start time falls inside a merged participant interval
    3. scrub gate: drop all-zero rows (the documentation states scrubbed
       entries are zeroed in the feature files)
    4. voicing gate: VUV == 1, applied to the glottal/prosodic block ONLY.
       The documentation states F0, NAQ, QOQ, H1H2, PSP, MDQ, peakSlope and Rd
       "should not be utilized" when VUV = 0. Spectral coefficients
       (MCEP/HMPDM/HMPDD) keep every speech frame.
    5. mean and std per retained column

VISION  (CLNF AUs + gaze + pose -> 84 dims)
-------------------------------------------
    AU_r  14 x {mean, std, p95} = 42
    AU_c   6 x {mean}           =  6   (std of a Bernoulli is sqrt(p(1-p)),
                                        a deterministic function of the mean,
                                        so it carries no extra information)
    gaze  12 x {mean, std}      = 24
    pose   6 x {mean, std}      = 12

The three CLNF files are joined on their `frame` column - never on row position -
and gated on success == 1. Vision is NOT speaker-gated: the camera films only
the participant, so listening behaviour during Ellie's turns is genuine signal.

DEFENSIVE PARSING
-----------------
Stage S0 found two non-finite representations in the corpus:
    -1.#IND   legacy MSVC NaN, 2460 tokens in CLNF_pose.txt for participants
              367/396/432. pandas RAISES ValueError on these with dtype=float.
    -Inf      3 values in COVAREP column 7 (peakSlope) for participant 371.
Both currently fall outside the gates (all -1.#IND rows have success == 0; the
-Inf rows sit at t=0.02-0.04s, before any speech). That is luck, not a guarantee,
so every file is parsed with an explicit na_values list and every gated matrix is
asserted finite before any statistic is taken. A non-finite value that survives
the gates is a hard failure, never a silent NaN.

Inputs (read-only):
    <intervals>                       Stage S1 index (participant speech intervals)
    <dataset>/<pid>_P.zip             participant archives, streamed
    <audit>                           Stage S0 report, optional cross-check

Output:
    <out>       JSON feature file (default ./dataset_build/multimodal_features.json)
    exit code   0 = all checks passed, 1 = validation failure, 2 = fatal setup error

Usage:
    python build_daic_multimodal_features.py
    python build_daic_multimodal_features.py --pids 300,371,432 --verbose
    python build_daic_multimodal_features.py --limit 5

NOTE: run with the project venv (.venv/Scripts/python.exe).
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import statistics
import sys
import time
import zipfile
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

import numpy as np

try:
    import tqdm as _tqdm_mod
    from tqdm import tqdm as _tqdm
except ImportError:  # pragma: no cover
    _tqdm_mod = None
    _tqdm = None

try:
    import pandas as pd
except ImportError:  # pragma: no cover
    pd = None


# --------------------------------------------------------------------------
# Paths
# --------------------------------------------------------------------------
DEFAULT_DATASET = Path("D:/datasets/DAIC-WOZ")
DEFAULT_INTERVALS = Path("./dataset_build/participant_intervals.json")
DEFAULT_AUDIT = Path("./dataset_build/archive_audit.json")
DEFAULT_OUT = Path("./dataset_build/multimodal_features.json")

# --------------------------------------------------------------------------
# Frozen dimensions - these are the agreed contract, asserted not computed
# --------------------------------------------------------------------------
AUDIO_DIM = 154
VISION_DIM = 84
EXPECTED_PARTICIPANTS = 188

COVAREP_HZ = 100.0
COVAREP_COLS = 74
FORMANT_COLS = 5

# Column identities. Indices 0-9 follow the AVEC documentation's ordered list.
# Index 10 is undocumented and measured all-zero in 18/18 sampled participants;
# it is retained as a feature so the 72-column count matches the frozen
# audio_dim, and is reported as degenerate rather than silently dropped.
COVAREP_LAYOUT: Dict[int, str] = {
    0: "F0", 1: "VUV", 2: "NAQ", 3: "QOQ", 4: "H1H2", 5: "PSP",
    6: "MDQ", 7: "peakSlope", 8: "Rd", 9: "Rd_conf", 10: "undocumented",
}
for _i in range(11, 36):
    COVAREP_LAYOUT[_i] = f"MCEP_{_i - 11}"
for _i in range(36, 61):
    COVAREP_LAYOUT[_i] = f"HMPDM_{_i - 36}"
for _i in range(61, 74):
    COVAREP_LAYOUT[_i] = f"HMPDD_{_i - 61}"

COVAREP_VUV_IDX = 1        # the voicing gate
COVAREP_RDCONF_IDX = 9     # a confidence, not a measurement
COVAREP_EXCLUDED = (COVAREP_VUV_IDX, COVAREP_RDCONF_IDX)
COVAREP_FEATURE_IDX: Tuple[int, ...] = tuple(
    i for i in range(COVAREP_COLS) if i not in COVAREP_EXCLUDED)  # 72 columns

# Columns the documentation says are invalid when VUV == 0.
COVAREP_VOICED_ONLY = (0, 2, 3, 4, 5, 6, 7, 8)  # F0 NAQ QOQ H1H2 PSP MDQ peakSlope Rd

# The agreed rule voices-gates "the leading glottal block ONLY". FORMANT lives in
# a separate file and the rule does not name it, so it is gated on speech but not
# on voicing. Exposed as a constant because the agreed wording is ambiguous here
# and this is the literal reading - flagged for review, not silently decided.
FORMANT_VOICED_GATED = False

# Frozen CLNF headers, verified byte-for-byte across all 188 archives by S0.
AUS_HEADER = ["frame", "timestamp", "confidence", "success",
              "AU01_r", "AU02_r", "AU04_r", "AU05_r", "AU06_r", "AU09_r",
              "AU10_r", "AU12_r", "AU14_r", "AU15_r", "AU17_r", "AU20_r",
              "AU25_r", "AU26_r",
              "AU04_c", "AU12_c", "AU15_c", "AU23_c", "AU28_c", "AU45_c"]
GAZE_HEADER = ["frame", "timestamp", "confidence", "success",
               "x_0", "y_0", "z_0", "x_1", "y_1", "z_1",
               "x_h0", "y_h0", "z_h0", "x_h1", "y_h1", "z_h1"]
POSE_HEADER = ["frame", "timestamp", "confidence", "success",
               "Tx", "Ty", "Tz", "Rx", "Ry", "Rz"]

AU_R_COLS = [c for c in AUS_HEADER if c.endswith("_r")]   # 14
AU_C_COLS = [c for c in AUS_HEADER if c.endswith("_c")]   # 6
GAZE_COLS = GAZE_HEADER[4:]                                # 12
POSE_COLS = POSE_HEADER[4:]                                # 6

P95 = 95.0
SUCCESS_COL = "success"

# Every spelling of a non-finite value observed or plausible in this corpus.
# pandas RAISES on '-1.#IND' with dtype=float unless it is listed here.
NA_VALUES = ["-1.#IND", "1.#IND", "-1.#INF", "1.#INF",
             "-1.#QNAN", "1.#QNAN", "-1.#SNAN", "1.#SNAN",
             "NaN", "nan", "NAN", "Inf", "inf", "-Inf", "-inf",
             "Infinity", "-Infinity", ""]

MIN_GATED_AUDIO_FRAMES = 100    # 1 second at 100 Hz
MIN_GATED_VISION_FRAMES = 30    # 1 second at 30 fps

SEV_FAIL, SEV_WARN, SEV_INFO = "FAIL", "WARN", "INFO"


# --------------------------------------------------------------------------
# Feature name construction - the frozen ordering contract
# --------------------------------------------------------------------------
def audio_feature_names() -> List[str]:
    """154 names, in emission order.

    COVAREP feature columns ascending by ORIGINAL index (so a name always maps
    back to its source column unambiguously), mean then std adjacent, followed
    by FORMANT 1..5 likewise.
    """
    names: List[str] = []
    for i in COVAREP_FEATURE_IDX:
        base = f"covarep_{i:02d}_{COVAREP_LAYOUT[i]}"
        names.append(f"{base}_mean")
        names.append(f"{base}_std")
    for i in range(FORMANT_COLS):
        names.append(f"formant_{i + 1}_mean")
        names.append(f"formant_{i + 1}_std")
    return names


def vision_feature_names() -> List[str]:
    """84 names, in emission order: AU_r, AU_c, gaze, pose."""
    names: List[str] = []
    for c in AU_R_COLS:
        names += [f"{c}_mean", f"{c}_std", f"{c}_p95"]
    for c in AU_C_COLS:
        names.append(f"{c}_mean")
    for c in GAZE_COLS:
        names += [f"gaze_{c}_mean", f"gaze_{c}_std"]
    for c in POSE_COLS:
        names += [f"pose_{c}_mean", f"pose_{c}_std"]
    return names


AUDIO_NAMES = audio_feature_names()
VISION_NAMES = vision_feature_names()
assert len(AUDIO_NAMES) == AUDIO_DIM, f"audio names {len(AUDIO_NAMES)} != {AUDIO_DIM}"
assert len(VISION_NAMES) == VISION_DIM, f"vision names {len(VISION_NAMES)} != {VISION_DIM}"


# --------------------------------------------------------------------------
# Reporting (ASCII only: the console is cp1252)
# --------------------------------------------------------------------------
class Reporter:
    def __init__(self, verbose: bool = False) -> None:
        self.findings: List[Dict[str, Any]] = []
        self.verbose = verbose

    def add(self, sev: str, check: str, msg: str, pid: Optional[str] = None) -> None:
        self.findings.append({"severity": sev, "check": check, "pid": pid, "message": msg})
        if sev == SEV_FAIL or (sev == SEV_WARN and self.verbose):
            print(f"[{sev}]{f' [{pid}]' if pid else ''} {check}: {msg}")

    def fail(self, c: str, m: str, pid: Optional[str] = None) -> None:
        self.add(SEV_FAIL, c, m, pid)

    def warn(self, c: str, m: str, pid: Optional[str] = None) -> None:
        self.add(SEV_WARN, c, m, pid)

    def info(self, c: str, m: str, pid: Optional[str] = None) -> None:
        self.add(SEV_INFO, c, m, pid)

    def count(self, sev: str) -> int:
        return sum(1 for f in self.findings if f["severity"] == sev)

    def digest(self) -> str:
        canon = sorted((f["severity"], f["check"], f["pid"] or "", f["message"])
                       for f in self.findings)
        return hashlib.sha256(
            json.dumps(canon, sort_keys=True, ensure_ascii=True).encode("utf-8")).hexdigest()


def progress(items: Iterable, total: int, desc: str, enabled: bool) -> Iterable:
    if enabled and _tqdm is not None:
        return _tqdm(items, total=total, desc=desc, unit="pid", ascii=True, ncols=78)
    return items


# --------------------------------------------------------------------------
# Streaming readers
# --------------------------------------------------------------------------
def read_member(zf: zipfile.ZipFile, member: str, header: Optional[int]) -> np.ndarray:
    """Parse one archive member into a float64 matrix, streamed, never extracted.

    float64 rather than float32: std over ~90k rows of MCEP values loses
    meaningful precision in float32, and the memory cost is bounded (the largest
    participant is ~155k x 74 = 91 MB, transient).
    """
    with zf.open(member) as fh:
        # skipinitialspace is REQUIRED, not cosmetic: the CLNF files are
        # comma-SPACE delimited, so a field arrives as ' -1.#IND' and would not
        # match the na_values entry '-1.#IND'. Without this, pandas raises
        # ValueError on participants 367/396/432.
        df = pd.read_csv(fh, header=header, na_values=NA_VALUES,
                         keep_default_na=True, engine="c", dtype=np.float64,
                         skipinitialspace=True)
    return df.to_numpy(dtype=np.float64, copy=False)


def read_clnf(zf: zipfile.ZipFile, member: str, expected: List[str],
              pid: str, kind: str, rep: Reporter) -> Optional[np.ndarray]:
    """Read a CLNF file and verify its header matches the frozen contract."""
    with zf.open(member) as fh:
        df = pd.read_csv(fh, na_values=NA_VALUES, keep_default_na=True,
                         engine="c", dtype=np.float64, skipinitialspace=True)
    got = [str(c).strip() for c in df.columns]
    if got != expected:
        rep.fail("F2.header", f"{kind}: header {got[:6]}... != frozen contract", pid)
        return None
    df.columns = got
    return df.to_numpy(dtype=np.float64, copy=False)


def interval_mask(n_rows: int, intervals: List[List[float]], hz: float) -> np.ndarray:
    """Boolean mask over feature rows for the merged participant intervals.

    Row i spans [i/hz, (i+1)/hz) and is kept when its START falls inside an
    interval, i.e. i in [ceil(s*hz), ceil(e*hz)). This is the same half-open
    [start, stop) convention Stage S1 used, so the two stages agree by
    construction. Slices are clamped to the array, which is what performs the
    clamping S0 identified as necessary (0.020s of speech, participant 381).
    """
    mask = np.zeros(n_rows, dtype=bool)
    for s, e in intervals:
        i0 = max(0, int(np.ceil(s * hz)))
        i1 = min(n_rows, int(np.ceil(e * hz)))
        if i1 > i0:
            mask[i0:i1] = True
    return mask


# --------------------------------------------------------------------------
# Audio pipeline
# --------------------------------------------------------------------------
def build_audio(pid: str, zf: zipfile.ZipFile, members: Dict[str, str],
                intervals: List[List[float]], rep: Reporter
                ) -> Optional[Tuple[List[float], Dict[str, Any]]]:
    cov = read_member(zf, members[f"{pid}_COVAREP.csv"], header=None)
    fmt = read_member(zf, members[f"{pid}_FORMANT.csv"], header=None)

    if cov.shape[1] != COVAREP_COLS:
        rep.fail("F1.covarep_cols", f"{cov.shape[1]} columns != {COVAREP_COLS}", pid)
        return None
    if fmt.shape[1] != FORMANT_COLS:
        rep.fail("F1.formant_cols", f"{fmt.shape[1]} columns != {FORMANT_COLS}", pid)
        return None

    n_cov_raw, n_fmt_raw = len(cov), len(fmt)
    # (1) truncate to the minimum shared length: S0 measured the delta as -1/0/+1
    n = min(n_cov_raw, n_fmt_raw)
    cov, fmt = cov[:n], fmt[:n]

    # (2) participant-interval gate
    speech = interval_mask(n, intervals, COVAREP_HZ)
    n_speech = int(speech.sum())
    if n_speech == 0:
        rep.fail("F3.no_speech_frames", "participant gate selected no frames", pid)
        return None

    # (3) scrub gate: the documentation states scrubbed entries are zeroed.
    # Judged on COVAREP only - FORMANT is truncated to the same index space, so
    # a scrubbed COVAREP row is a scrubbed instant regardless of FORMANT.
    nonzero = np.any(np.nan_to_num(cov, nan=0.0, posinf=0.0, neginf=0.0) != 0.0, axis=1)
    keep = speech & nonzero
    n_scrub_removed = int(n_speech - keep.sum())
    if not keep.any():
        rep.fail("F3.all_scrubbed", "every gated frame is scrubbed (all-zero)", pid)
        return None

    # (4) voicing gate for the glottal block only
    vuv = cov[:, COVAREP_VUV_IDX]
    voiced = keep & (vuv == 1.0)
    n_voiced = int(voiced.sum())
    if n_voiced == 0:
        rep.fail("F3.no_voiced_frames", "no voiced frames inside participant speech", pid)
        return None
    if int(keep.sum()) < MIN_GATED_AUDIO_FRAMES:
        rep.warn("F3.few_frames",
                 f"only {int(keep.sum())} gated audio frames", pid)

    voiced_set = set(COVAREP_VOICED_ONLY)
    values: List[float] = []
    for i in COVAREP_FEATURE_IDX:
        m = voiced if i in voiced_set else keep
        col = cov[m, i]
        # Requirement: assert finiteness AFTER gating; never silently drop.
        if not np.all(np.isfinite(col)):
            n_bad = int((~np.isfinite(col)).sum())
            rep.fail("F4.nonfinite_audio",
                     f"{n_bad} non-finite value(s) survived gating in COVAREP "
                     f"column {i} ({COVAREP_LAYOUT[i]})", pid)
            return None
        values.append(float(col.mean()))
        values.append(float(col.std()))  # population std (ddof=0)

    fmt_mask = voiced if FORMANT_VOICED_GATED else keep
    for i in range(FORMANT_COLS):
        col = fmt[fmt_mask, i]
        if not np.all(np.isfinite(col)):
            n_bad = int((~np.isfinite(col)).sum())
            rep.fail("F4.nonfinite_audio",
                     f"{n_bad} non-finite value(s) survived gating in FORMANT "
                     f"column {i}", pid)
            return None
        values.append(float(col.mean()))
        values.append(float(col.std()))

    coverage = {
        "covarep_rows_raw": n_cov_raw,
        "formant_rows_raw": n_fmt_raw,
        "rows_after_truncation": n,
        "frames_in_speech": n_speech,
        "frames_scrubbed_removed": n_scrub_removed,
        "frames_gated": int(keep.sum()),
        "frames_voiced": n_voiced,
        "speech_frame_ratio": round(n_speech / n, 6) if n else 0.0,
        "gated_frame_ratio": round(float(keep.sum()) / n, 6) if n else 0.0,
        "voiced_ratio_within_gate": round(n_voiced / float(keep.sum()), 6) if keep.any() else 0.0,
        "formant_voiced_gated": FORMANT_VOICED_GATED,
    }
    return values, coverage


# --------------------------------------------------------------------------
# Vision pipeline
# --------------------------------------------------------------------------
def build_vision(pid: str, zf: zipfile.ZipFile, members: Dict[str, str],
                 rep: Reporter) -> Optional[Tuple[List[float], Dict[str, Any]]]:
    aus = read_clnf(zf, members[f"{pid}_CLNF_AUs.txt"], AUS_HEADER, pid, "AUs", rep)
    gaze = read_clnf(zf, members[f"{pid}_CLNF_gaze.txt"], GAZE_HEADER, pid, "gaze", rep)
    pose = read_clnf(zf, members[f"{pid}_CLNF_pose.txt"], POSE_HEADER, pid, "pose", rep)
    if aus is None or gaze is None or pose is None:
        return None

    # Join on the frame column, never on row position. S0 verified the three
    # files align for all 188, but joining defensively costs nothing and makes
    # a future misalignment a visible failure rather than a silent shear.
    fa, fg, fp = aus[:, 0], gaze[:, 0], pose[:, 0]
    if not (len(fa) == len(fg) == len(fp) and
            np.array_equal(fa, fg) and np.array_equal(fa, fp)):
        common, ia, ig = np.intersect1d(fa, fg, return_indices=True)
        common2, iac, ip = np.intersect1d(common, fp, return_indices=True)
        if len(common2) == 0:
            rep.fail("F5.no_common_frames", "CLNF files share no frame indices", pid)
            return None
        rep.warn("F5.frame_realign",
                 f"CLNF files misaligned; using {len(common2)} common frames", pid)
        aus, gaze, pose = aus[ia[iac]], gaze[ig[iac]], pose[ip]
    n_raw = len(aus)

    succ_idx = AUS_HEADER.index(SUCCESS_COL)
    success = aus[:, succ_idx] == 1.0
    n_ok = int(success.sum())
    if n_ok == 0:
        rep.fail("F6.no_tracked_frames", "no frames with success == 1", pid)
        return None
    if n_ok < MIN_GATED_VISION_FRAMES:
        rep.warn("F6.few_frames", f"only {n_ok} tracked frames", pid)

    a, g, p = aus[success], gaze[success], pose[success]

    def finite_or_fail(mat: np.ndarray, cols: List[str], header: List[str],
                       kind: str) -> Optional[np.ndarray]:
        idx = [header.index(c) for c in cols]
        sub = mat[:, idx]
        if not np.all(np.isfinite(sub)):
            bad = int((~np.isfinite(sub)).sum())
            rep.fail("F4.nonfinite_vision",
                     f"{bad} non-finite value(s) survived the success gate in {kind}",
                     pid)
            return None
        return sub

    au_r = finite_or_fail(a, AU_R_COLS, AUS_HEADER, "AU_r")
    au_c = finite_or_fail(a, AU_C_COLS, AUS_HEADER, "AU_c")
    gz = finite_or_fail(g, GAZE_COLS, GAZE_HEADER, "gaze")
    ps = finite_or_fail(p, POSE_COLS, POSE_HEADER, "pose")
    if au_r is None or au_c is None or gz is None or ps is None:
        return None

    values: List[float] = []
    # AU_r: mean, std, p95 (linear interpolation - numpy's deterministic default)
    p95 = np.percentile(au_r, P95, axis=0, method="linear")
    for j in range(au_r.shape[1]):
        values.append(float(au_r[:, j].mean()))
        values.append(float(au_r[:, j].std()))
        values.append(float(p95[j]))
    # AU_c: mean only - activation rate; std is a function of the mean
    for j in range(au_c.shape[1]):
        values.append(float(au_c[:, j].mean()))
    for mat in (gz, ps):
        for j in range(mat.shape[1]):
            values.append(float(mat[:, j].mean()))
            values.append(float(mat[:, j].std()))

    coverage = {
        "clnf_rows_raw": n_raw,
        "frames_tracked": n_ok,
        "tracker_success_ratio": round(n_ok / n_raw, 6) if n_raw else 0.0,
    }
    return values, coverage


# --------------------------------------------------------------------------
# Per-participant driver
# --------------------------------------------------------------------------
REQUIRED_MEMBERS = ("COVAREP.csv", "FORMANT.csv",
                    "CLNF_AUs.txt", "CLNF_gaze.txt", "CLNF_pose.txt")


def build_one(pid: str, archive: Path, intervals: List[List[float]],
              rep: Reporter) -> Optional[Dict[str, Any]]:
    try:
        zf = zipfile.ZipFile(archive, "r")   # read-only, always
    except (zipfile.BadZipFile, OSError) as exc:
        rep.fail("F0.archive", f"cannot open archive: {type(exc).__name__}", pid)
        return None

    with zf:
        members = {os.path.basename(n): n for n in zf.namelist() if not n.endswith("/")}
        missing = [k for k in REQUIRED_MEMBERS if f"{pid}_{k}" not in members]
        if missing:
            rep.fail("F0.members", f"missing {missing}", pid)
            return None
        # Explicit guarantee: HOG is never referenced anywhere in this stage.
        assert not any("hog" in m.lower() for m in
                       (f"{pid}_{k}" for k in REQUIRED_MEMBERS))

        try:
            audio = build_audio(pid, zf, members, intervals, rep)
            vision = build_vision(pid, zf, members, rep)
        except (ValueError, MemoryError, zipfile.BadZipFile, OSError) as exc:
            rep.fail("F7.extract", f"{type(exc).__name__}: {exc}", pid)
            return None

    if audio is None or vision is None:
        return None
    a_vec, a_cov = audio
    v_vec, v_cov = vision

    if len(a_vec) != AUDIO_DIM:
        rep.fail("F8.audio_dim", f"produced {len(a_vec)} dims != {AUDIO_DIM}", pid)
        return None
    if len(v_vec) != VISION_DIM:
        rep.fail("F8.vision_dim", f"produced {len(v_vec)} dims != {VISION_DIM}", pid)
        return None
    for name, vec in (("audio", a_vec), ("vision", v_vec)):
        arr = np.asarray(vec, dtype=np.float64)
        if not np.all(np.isfinite(arr)):
            rep.fail("F8.nonfinite_output",
                     f"{name} vector contains non-finite values after aggregation", pid)
            return None

    return {
        "participant_id": pid,
        "audio": a_vec,
        "vision": v_vec,
        "coverage": {"audio": a_cov, "vision": v_cov},
    }


# --------------------------------------------------------------------------
# Helpers
# --------------------------------------------------------------------------
def describe(vals: List[float]) -> Dict[str, float]:
    if not vals:
        return {}
    return {"min": round(min(vals), 6), "median": round(statistics.median(vals), 6),
            "mean": round(statistics.fmean(vals), 6), "max": round(max(vals), 6),
            "n": len(vals)}


def environment() -> Dict[str, Any]:
    return {"python": sys.version.split()[0], "platform": sys.platform,
            "numpy": np.__version__,
            "pandas": getattr(pd, "__version__", None) if pd is not None else None,
            "tqdm": getattr(_tqdm_mod, "__version__", None) if _tqdm_mod else None}


# --------------------------------------------------------------------------
# Main
# --------------------------------------------------------------------------
def main() -> int:
    ap = argparse.ArgumentParser(
        description="Extract the frozen multimodal feature vectors (Stage S2)")
    ap.add_argument("--dataset", type=Path, default=DEFAULT_DATASET)
    ap.add_argument("--intervals", type=Path, default=DEFAULT_INTERVALS)
    ap.add_argument("--audit", type=Path, default=DEFAULT_AUDIT)
    ap.add_argument("--out", type=Path, default=DEFAULT_OUT)
    ap.add_argument("--pids", default="", help="comma-separated participant IDs")
    ap.add_argument("--limit", type=int, default=0, help="first N participants")
    ap.add_argument("--no-progress", action="store_true")
    ap.add_argument("--verbose", action="store_true")
    args = ap.parse_args()

    t0 = time.time()
    rep = Reporter(verbose=args.verbose)

    if pd is None:
        print("[FATAL] pandas is required. Run with .venv/Scripts/python.exe")
        return 2
    if not args.dataset.is_dir():
        print(f"[FATAL] dataset folder not found: {args.dataset}")
        return 2
    if not args.intervals.is_file():
        print(f"[FATAL] Stage S1 interval index not found: {args.intervals}")
        return 2

    dataset = args.dataset.resolve()
    out_path = args.out.resolve()
    if dataset == out_path.parent or dataset in out_path.parents:
        print(f"[FATAL] refusing to write output inside the dataset: {out_path}")
        return 2

    try:
        with open(args.intervals, "r", encoding="utf-8") as fh:
            s1 = json.load(fh)
    except (OSError, json.JSONDecodeError) as exc:
        print(f"[FATAL] cannot read interval index: {type(exc).__name__}: {exc}")
        return 2
    s1_participants = s1.get("participants") or {}
    if not s1_participants:
        print("[FATAL] interval index contains no participants")
        return 2

    print("=" * 78)
    print("DAIC MULTIMODAL FEATURE EXTRACTION (Stage S2) - READ-ONLY, STREAMED")
    print("=" * 78)
    env = environment()
    print(f"[INFO] dataset   : {dataset}")
    print(f"[INFO] intervals : {args.intervals.resolve()}  "
          f"({len(s1_participants)} participants)")
    print(f"[INFO] output    : {out_path}")
    print(f"[INFO] python    : {env['python']}  numpy={env['numpy']}  "
          f"pandas={env['pandas']}")
    print(f"[INFO] dims      : audio={AUDIO_DIM}  vision={VISION_DIM}")
    print(f"[INFO] covarep   : {len(COVAREP_FEATURE_IDX)} feature columns "
          f"(excluded {COVAREP_EXCLUDED} = VUV, Rd_conf)")
    print(f"[INFO] formant voiced-gated: {FORMANT_VOICED_GATED}")
    print("-" * 78)

    pids = sorted(s1_participants, key=int)
    subset = bool(args.pids or args.limit)
    if args.pids:
        wanted = {p.strip() for p in args.pids.split(",") if p.strip()}
        unknown = sorted(wanted - set(pids))
        if unknown:
            print(f"[FATAL] --pids: not in the interval index: {unknown}")
            return 2
        pids = [p for p in pids if p in wanted]
    if args.limit:
        pids = pids[:args.limit]
    print(f"[STEP 1/3] extracting features for {len(pids)} participant(s)")

    records: Dict[str, Dict[str, Any]] = {}
    failed: List[str] = []
    for pid in progress(pids, len(pids), "features", not args.no_progress):
        archive = dataset / f"{pid}_P.zip"
        if not archive.is_file():
            rep.fail("F0.archive", f"archive not found: {archive.name}", pid)
            failed.append(pid)
            continue
        iv = s1_participants[pid]["intervals"]
        rec = build_one(pid, archive, iv, rep)
        if rec is None:
            failed.append(pid)
        else:
            records[pid] = rec
    print(f"[ OK ] extracted={len(records)}  failed={len(failed)} {failed}")

    # ---- validation --------------------------------------------------------
    print("[STEP 2/3] validation")
    if not subset and len(records) != EXPECTED_PARTICIPANTS:
        rep.fail("V1.count",
                 f"{len(records)} participants != expected {EXPECTED_PARTICIPANTS}")
    if not subset:
        missing = sorted(set(s1_participants) - set(records), key=int)
        if missing:
            rep.fail("V1.missing", f"no features for: {missing}")

    for pid in sorted(records, key=int):
        r = records[pid]
        if len(r["audio"]) != AUDIO_DIM:
            rep.fail("V2.audio_dim", f"{len(r['audio'])} != {AUDIO_DIM}", pid)
        if len(r["vision"]) != VISION_DIM:
            rep.fail("V2.vision_dim", f"{len(r['vision'])} != {VISION_DIM}", pid)
        for name in ("audio", "vision"):
            arr = np.asarray(r[name], dtype=np.float64)
            if np.isnan(arr).any():
                rep.fail(f"V3.nan_{name}", "NaN in final vector", pid)
            if np.isinf(arr).any():
                rep.fail(f"V3.inf_{name}", "Inf in final vector", pid)

    # Ordering must be deterministic and ascending by integer participant id.
    emitted = list(sorted(records, key=int))
    if emitted != sorted(emitted, key=int):
        rep.fail("V4.ordering", "participant ordering is not deterministic")

    if len(AUDIO_NAMES) != AUDIO_DIM or len(VISION_NAMES) != VISION_DIM:
        rep.fail("V5.names", "feature name list length does not match the frozen dims")
    if len(set(AUDIO_NAMES)) != len(AUDIO_NAMES):
        rep.fail("V5.names", "duplicate audio feature names")
    if len(set(VISION_NAMES)) != len(VISION_NAMES):
        rep.fail("V5.names", "duplicate vision feature names")

    # Degenerate (zero-variance) dimensions are legitimate but must be visible:
    # COVAREP column 10 is all-zero corpus-wide, so its mean/std are constant 0.
    if records:
        amat = np.array([records[p]["audio"] for p in emitted], dtype=np.float64)
        vmat = np.array([records[p]["vision"] for p in emitted], dtype=np.float64)
        a_const = [AUDIO_NAMES[i] for i in np.where(amat.std(axis=0) == 0.0)[0]]
        v_const = [VISION_NAMES[i] for i in np.where(vmat.std(axis=0) == 0.0)[0]]
        for n in a_const:
            rep.warn("V6.constant_audio", f"dimension is constant across the corpus: {n}")
        for n in v_const:
            rep.warn("V6.constant_vision", f"dimension is constant across the corpus: {n}")
    else:
        a_const = v_const = []
    print(f"[ OK ] constant dims: audio={len(a_const)} vision={len(v_const)}")

    # ---- S0 cross-check ----------------------------------------------------
    print("[STEP 3/3] Stage S0 cross-check")
    cross: Dict[str, Any] = {"checked": False}
    if subset or not args.audit.is_file():
        print("[INFO] skipped (subset run or no S0 report)")
    else:
        try:
            with open(args.audit, "r", encoding="utf-8") as fh:
                audit = json.load(fh)
            per = audit.get("per_participant", {})
            mism = []
            for pid, r in records.items():
                d = (per.get(pid) or {}).get("deep") or {}
                s0_succ = d.get("tracker_success")
                mine = r["coverage"]["vision"]["tracker_success_ratio"]
                if s0_succ is not None and abs(s0_succ - mine) > 1e-3:
                    mism.append(pid)
            if mism:
                rep.fail("X1.tracker_mismatch",
                         f"tracker success disagrees with S0 for: {mism}")
            cross = {"checked": True, "audit_digest": audit.get("findings_digest"),
                     "tracker_mismatches": mism}
            print(f"[ OK ] tracker-success agreement with S0: "
                  f"{len(records) - len(mism)}/{len(records)}")
        except (OSError, json.JSONDecodeError) as exc:
            rep.warn("X1.audit", f"cannot read S0 report: {type(exc).__name__}: {exc}")

    # ---- serialise ---------------------------------------------------------
    out = {
        "schema": "daic_multimodal_features/1",
        "generated_by": "build_daic_multimodal_features.py",
        "environment": env,
        "inputs": {
            "dataset": str(dataset),
            "intervals": str(args.intervals.resolve()),
            "intervals_digest": s1.get("findings_digest"),
            "pids": args.pids or None,
            "limit": args.limit,
        },
        "spec": {
            "audio_dim": AUDIO_DIM,
            "vision_dim": VISION_DIM,
            "audio": {
                "source": "COVAREP + FORMANT",
                "names": AUDIO_NAMES,
                "covarep_columns_used": list(COVAREP_FEATURE_IDX),
                "covarep_excluded": {str(COVAREP_VUV_IDX): "VUV (voicing gate)",
                                     str(COVAREP_RDCONF_IDX): "Rd_conf (confidence)"},
                "covarep_layout": {str(k): v for k, v in COVAREP_LAYOUT.items()},
                "voiced_only_columns": list(COVAREP_VOICED_ONLY),
                "formant_voiced_gated": FORMANT_VOICED_GATED,
                "gates": ["truncate to min(COVAREP, FORMANT)",
                          "participant intervals, row i -> [i/100,(i+1)/100)",
                          "drop all-zero (scrubbed) rows",
                          "VUV == 1 for the glottal block only"],
                "statistics": ["mean", "std (population, ddof=0)"],
            },
            "vision": {
                "source": "CLNF_AUs + CLNF_gaze + CLNF_pose",
                "names": VISION_NAMES,
                "gates": ["join on frame index", "success == 1"],
                "statistics": {"AU_r": ["mean", "std", "p95"], "AU_c": ["mean"],
                               "gaze": ["mean", "std"], "pose": ["mean", "std"]},
                "speaker_gated": False,
            },
            "na_values": NA_VALUES,
        },
        "totals": {
            "participants": len(records),
            "failed": sorted(failed, key=int),
            "fail": rep.count(SEV_FAIL),
            "warn": rep.count(SEV_WARN),
            "constant_audio_dims": a_const,
            "constant_vision_dims": v_const,
        },
        "summary_statistics": {
            "gated_frame_ratio": describe([records[p]["coverage"]["audio"]["gated_frame_ratio"]
                                           for p in emitted]),
            "voiced_ratio_within_gate": describe(
                [records[p]["coverage"]["audio"]["voiced_ratio_within_gate"] for p in emitted]),
            "tracker_success_ratio": describe(
                [records[p]["coverage"]["vision"]["tracker_success_ratio"] for p in emitted]),
            "frames_gated": describe([float(records[p]["coverage"]["audio"]["frames_gated"])
                                      for p in emitted]),
            "frames_tracked": describe([float(records[p]["coverage"]["vision"]["frames_tracked"])
                                        for p in emitted]),
        },
        "audit_cross_check": cross,
        "participants": {pid: records[pid] for pid in emitted},
        "findings": rep.findings,
        "findings_digest": rep.digest(),
        "elapsed_seconds": round(time.time() - t0, 1),
    }

    try:
        out_path.parent.mkdir(parents=True, exist_ok=True)
        with open(out_path, "w", encoding="utf-8") as fh:
            json.dump(out, fh, indent=2, sort_keys=True, ensure_ascii=True)
    except OSError as exc:
        print(f"[FATAL] cannot write features to {out_path}: {exc}")
        return 2

    n_fail, n_warn = rep.count(SEV_FAIL), rep.count(SEV_WARN)
    size_kb = out_path.stat().st_size / 1024
    print("-" * 78)
    print("=== FEATURE EXTRACTION SUMMARY ===")
    print(f"participants        : {len(records)}")
    print(f"audio dim / vision  : {AUDIO_DIM} / {VISION_DIM}")
    print(f"gated frame ratio   : {out['summary_statistics']['gated_frame_ratio']}")
    print(f"voiced within gate  : {out['summary_statistics']['voiced_ratio_within_gate']}")
    print(f"tracker success     : {out['summary_statistics']['tracker_success_ratio']}")
    print(f"FAIL findings       : {n_fail}")
    print(f"WARN findings       : {n_warn}")
    print(f"findings digest     : {rep.digest()}")
    print(f"elapsed             : {out['elapsed_seconds']:.1f}s")
    print(f"output              : {out_path}  ({size_kb:.1f} KB)")

    if n_fail:
        print("\n[FAIL] feature extraction is NOT usable - do not proceed to S3.")
        by: Dict[str, int] = {}
        for f in rep.findings:
            if f["severity"] == SEV_FAIL:
                by[f["check"]] = by.get(f["check"], 0) + 1
        for c, k in sorted(by.items()):
            print(f"    {c}: {k}")
        return 1

    print("\n[PASS] multimodal feature vectors extracted and validated.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
