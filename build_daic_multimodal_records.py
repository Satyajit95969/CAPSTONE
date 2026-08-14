#!/usr/bin/env python3
"""
build_daic_multimodal_records.py  (Phase 16 - Stage S3: multimodal parquet)

Joins the FROZEN text-only records with the FROZEN Stage S2 feature vectors and
writes the multimodal parquet that Experiment 7 will consume.

This stage performs NO feature extraction and recomputes NO statistics. It is a
join plus a serialisation, and nothing else. Both inputs are treated as
immutable.

THE ONE-FACTOR GUARANTEE
------------------------
Experiment 7 changes exactly one factor: input modality. That claim is only
credible if the text-only content of the corpus is provably untouched, so this
stage does not rebuild `participant_id`, `text` or `phq_score` from source - it
carries the ORIGINAL Arrow arrays across into the new table with
`Table.append_column`. Byte-identity is therefore structural, not something the
validator has to hope for; the validator then proves it anyway with an
Arrow-level equality check against the frozen file.

The frozen parquet is NEVER opened for writing. An explicit guard refuses to run
if the output path resolves to it.

TRAINER CONTRACT
----------------
`trainer_mentalbert_daic.py` (SHA 65b1902e...) reads exactly these keys:

    features.audio.wav2vec2   -> audio vector   (MultiModalDataset._extract_audio_vec)
    features.video.densenet   -> vision vector  (MultiModalDataset._extract_video_vec)

The names are historical and inaccurate - the payloads are COVAREP/FORMANT and
CLNF statistics, not wav2vec2 or DenseNet embeddings. They are reused verbatim
because renaming them would require editing the trainer, breaking the frozen SHA
and with it the comparability chain that Baseline-CV and Experiments 3-6 all
depend on. The true provenance travels alongside in `features.<mod>.spec`, which
the trainer ignores (both extractors read one key and return).

`features` is written as a JSON STRING. The trainer's read_parquet_records()
decodes string-valued `features` (trainer lines 89-96), which avoids nested
struct round-tripping entirely.

Inputs (read-only):
    daic_records.parquet                        frozen text-only records
    dataset_build/multimodal_features.json      frozen Stage S2 vectors

Outputs:
    dataset_build/daic_records_multimodal.parquet
    dataset_build/multimodal_records_report.json

Exit codes:
    0 = built and validated, 1 = validation failure, 2 = fatal setup error

Usage:
    python build_daic_multimodal_records.py
    python build_daic_multimodal_records.py --no-trainer-check

NOTE: run with the project venv (.venv/Scripts/python.exe).
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import math
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import pyarrow as pa
import pyarrow.parquet as pq

try:
    import pandas as pd
except ImportError:  # pragma: no cover
    pd = None


# --------------------------------------------------------------------------
# Paths and frozen contract
# --------------------------------------------------------------------------
DEFAULT_PARQUET = Path("./daic_records.parquet")
DEFAULT_FEATURES = Path("./dataset_build/multimodal_features.json")
DEFAULT_OUT = Path("./dataset_build/daic_records_multimodal.parquet")
DEFAULT_REPORT = Path("./dataset_build/multimodal_records_report.json")
TRAINER_PATH = Path("./trainer_mentalbert_daic.py")

# Measured SHA of the frozen text-only corpus. Recorded BEFORE this stage ever
# ran, so pinning it here is a genuine tripwire rather than a rubber stamp.
FROZEN_PARQUET_SHA = "9a241851760727779fcaa4d50041b8bdc4406fe6b66ddbbd7105f4ce4fc55f00"
FROZEN_TRAINER_SHA = "65b1902e2ba8dd996c6960db1a6595d84fd48500f4d8707cb79688093d7a230b"

FROZEN_COLUMNS = ["participant_id", "text", "phq_score"]
NEW_COLUMNS = ["features", "audio_coverage", "video_coverage"]
FINAL_COLUMNS = FROZEN_COLUMNS + NEW_COLUMNS

EXPECTED_ROWS = 188
AUDIO_DIM = 154
VISION_DIM = 84

# The trainer's key names. Do not rename - see TRAINER CONTRACT above.
AUDIO_KEY = "wav2vec2"
VISION_KEY = "densenet"

COMPRESSION = "snappy"   # matches the frozen parquet's codec

SEV_FAIL, SEV_WARN, SEV_INFO = "FAIL", "WARN", "INFO"


class Reporter:
    """ASCII-only reporting, matching the S0/S1/S2 idiom (console is cp1252)."""

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


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for block in iter(lambda: fh.read(1 << 16), b""):
            h.update(block)
    return h.hexdigest()


def load_trainer():
    """Import the frozen trainer WITHOUT executing main().

    Same mechanism verify_daic_records.py uses: no model or tokenizer is
    constructed, so nothing is downloaded.
    """
    spec = importlib.util.spec_from_file_location("trainer_mentalbert_daic", TRAINER_PATH)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


# --------------------------------------------------------------------------
# Feature payload construction
# --------------------------------------------------------------------------
def build_features_json(rec: Dict[str, Any], spec: Dict[str, Any]) -> str:
    """Serialise ONE record's features into the agreed JSON string.

    The vectors are passed through exactly as Stage S2 emitted them - no
    rounding, no reordering, no recomputation. sort_keys makes the byte output
    deterministic for a given input.
    """
    audio_spec = spec.get("audio", {})
    video_spec = spec.get("video", {})
    payload = {
        "audio": {
            AUDIO_KEY: rec["audio"],
            "spec": {
                "source": audio_spec.get("source"),
                "names": audio_spec.get("names"),
                "gates": audio_spec.get("gates"),
            },
        },
        "video": {
            VISION_KEY: rec["vision"],
            "spec": {
                "source": video_spec.get("source"),
                "names": video_spec.get("names"),
                "gates": video_spec.get("gates"),
            },
        },
    }
    return json.dumps(payload, sort_keys=True, ensure_ascii=True)


# --------------------------------------------------------------------------
# Main
# --------------------------------------------------------------------------
def main() -> int:
    ap = argparse.ArgumentParser(
        description="Join frozen text records with frozen S2 features (Stage S3)")
    ap.add_argument("--parquet", type=Path, default=DEFAULT_PARQUET,
                    help="frozen text-only records (immutable input)")
    ap.add_argument("--features", type=Path, default=DEFAULT_FEATURES,
                    help="Stage S2 feature file (immutable input)")
    ap.add_argument("--out", type=Path, default=DEFAULT_OUT)
    ap.add_argument("--report", type=Path, default=DEFAULT_REPORT)
    ap.add_argument("--no-trainer-check", action="store_true",
                    help="skip importing the frozen trainer for the contract check")
    ap.add_argument("--allow-sha-drift", action="store_true",
                    help="proceed even if the frozen parquet SHA has changed "
                         "(for a legitimately re-staged corpus)")
    ap.add_argument("--verbose", action="store_true")
    args = ap.parse_args()

    t0 = time.time()
    rep = Reporter(verbose=args.verbose)

    # ---- fatal setup checks (exit 2) --------------------------------------
    if pd is None:
        print("[FATAL] pandas is required. Run with .venv/Scripts/python.exe")
        return 2
    if not args.parquet.is_file():
        print(f"[FATAL] frozen parquet not found: {args.parquet}")
        return 2
    if not args.features.is_file():
        print(f"[FATAL] Stage S2 feature file not found: {args.features}")
        return 2

    frozen_path = args.parquet.resolve()
    out_path = args.out.resolve()
    report_path = args.report.resolve()

    # Requirement 9: the frozen parquet must NEVER be overwritten.
    if out_path == frozen_path:
        print(f"[FATAL] refusing to overwrite the frozen parquet: {frozen_path}")
        return 2
    if report_path == frozen_path:
        print(f"[FATAL] refusing to overwrite the frozen parquet with the report")
        return 2

    print("=" * 78)
    print("DAIC MULTIMODAL RECORDS (Stage S3) - JOIN ONLY, NO RECOMPUTATION")
    print("=" * 78)
    print(f"[INFO] frozen parquet : {frozen_path}")
    print(f"[INFO] features       : {args.features.resolve()}")
    print(f"[INFO] output         : {out_path}")
    print("-" * 78)

    # ---- load and verify the frozen inputs --------------------------------
    print("[STEP 1/6] loading frozen inputs")
    in_sha = sha256_file(frozen_path)
    if in_sha != FROZEN_PARQUET_SHA:
        msg = (f"frozen parquet SHA {in_sha} != pinned {FROZEN_PARQUET_SHA}; "
               "the text-only corpus is not the one this stage was built against")
        if args.allow_sha_drift:
            rep.warn("I1.parquet_sha", msg)
        else:
            rep.fail("I1.parquet_sha", msg)
            print("[FATAL] refusing to build on an unrecognised text corpus "
                  "(use --allow-sha-drift only if the corpus was legitimately re-staged)")
            return 1
    print(f"[ OK ] frozen parquet SHA verified: {in_sha[:32]}...")

    frozen_table = pq.read_table(frozen_path)
    if frozen_table.column_names != FROZEN_COLUMNS:
        rep.fail("I2.columns",
                 f"frozen parquet columns {frozen_table.column_names} != {FROZEN_COLUMNS}")
        print("[FATAL] frozen parquet does not have the expected schema")
        return 1

    try:
        with open(args.features, "r", encoding="utf-8") as fh:
            s2 = json.load(fh)
    except (OSError, json.JSONDecodeError) as exc:
        print(f"[FATAL] cannot read feature file: {type(exc).__name__}: {exc}")
        return 2
    s2_participants: Dict[str, Any] = s2.get("participants") or {}
    s2_spec: Dict[str, Any] = s2.get("spec") or {}
    if not s2_participants:
        print("[FATAL] feature file contains no participants")
        return 2
    if s2_spec.get("audio_dim") != AUDIO_DIM or s2_spec.get("vision_dim") != VISION_DIM:
        rep.fail("I3.dims",
                 f"feature file declares audio_dim={s2_spec.get('audio_dim')} "
                 f"vision_dim={s2_spec.get('vision_dim')}, expected {AUDIO_DIM}/{VISION_DIM}")
        print("[FATAL] feature file dimensions do not match the frozen contract")
        return 1
    print(f"[ OK ] S2 features loaded: {len(s2_participants)} participants, "
          f"dims {AUDIO_DIM}/{VISION_DIM}")

    # ---- join validation (abort on ANY mismatch) --------------------------
    print("[STEP 2/6] join validation")
    parquet_pids: List[str] = [str(v) for v in
                               frozen_table.column("participant_id").to_pylist()]
    n_rows = len(parquet_pids)

    if n_rows != EXPECTED_ROWS:
        rep.fail("J1.rows", f"frozen parquet has {n_rows} rows, expected {EXPECTED_ROWS}")
    if len(set(parquet_pids)) != n_rows:
        dupes = sorted({p for p in parquet_pids if parquet_pids.count(p) > 1})
        rep.fail("J2.duplicate_parquet", f"duplicate participant_id in parquet: {dupes}")

    feature_pids = [str(p) for p in s2_participants]
    if len(set(feature_pids)) != len(feature_pids):
        rep.fail("J2.duplicate_features", "duplicate participant_id in the feature file")

    missing = [p for p in parquet_pids if p not in s2_participants]
    extra = [p for p in feature_pids if p not in set(parquet_pids)]
    if missing:
        rep.fail("J3.missing", f"parquet participants with no features: {missing}")
    if extra:
        rep.fail("J3.extra", f"features with no parquet row: {extra}")

    if rep.count(SEV_FAIL):
        print("[FATAL] join preconditions not satisfied - refusing to build")
        return 1
    print(f"[ OK ] one-to-one join over {n_rows} participants, no duplicates")

    # ---- build the new columns, IN PARQUET ROW ORDER ----------------------
    # Row order follows the frozen parquet exactly. The features are looked up
    # per row rather than the parquet being reordered to match the features -
    # the frozen artifact defines the ordering, never the new one.
    print("[STEP 3/6] building feature payloads")
    features_col: List[str] = []
    audio_cov: List[float] = []
    video_cov: List[float] = []
    for pid in parquet_pids:
        rec = s2_participants[pid]
        if len(rec["audio"]) != AUDIO_DIM:
            rep.fail("B1.audio_dim", f"{len(rec['audio'])} != {AUDIO_DIM}", pid)
        if len(rec["vision"]) != VISION_DIM:
            rep.fail("B1.vision_dim", f"{len(rec['vision'])} != {VISION_DIM}", pid)
        cov = rec.get("coverage") or {}
        a_cov = (cov.get("audio") or {}).get("gated_frame_ratio")
        v_cov = (cov.get("vision") or {}).get("tracker_success_ratio")
        if a_cov is None or v_cov is None:
            rep.fail("B2.coverage", "missing coverage statistics", pid)
            a_cov = a_cov if a_cov is not None else float("nan")
            v_cov = v_cov if v_cov is not None else float("nan")
        features_col.append(build_features_json(rec, s2_spec))
        audio_cov.append(float(a_cov))
        video_cov.append(float(v_cov))

    if rep.count(SEV_FAIL):
        print("[FATAL] feature payload construction failed - refusing to write")
        return 1
    print(f"[ OK ] {len(features_col)} payloads built "
          f"(mean JSON length {sum(map(len, features_col)) // len(features_col)} bytes)")

    # ---- assemble and write -----------------------------------------------
    # append_column carries the ORIGINAL Arrow arrays through untouched; the
    # frozen columns are never rebuilt, reparsed or round-tripped via pandas.
    print("[STEP 4/6] writing parquet")
    table = frozen_table
    table = table.append_column("features", pa.array(features_col, pa.string()))
    table = table.append_column("audio_coverage", pa.array(audio_cov, pa.float64()))
    table = table.append_column("video_coverage", pa.array(video_cov, pa.float64()))

    if table.column_names != FINAL_COLUMNS:
        rep.fail("W1.schema", f"assembled columns {table.column_names} != {FINAL_COLUMNS}")
        return 1

    try:
        out_path.parent.mkdir(parents=True, exist_ok=True)
        pq.write_table(table, out_path, compression=COMPRESSION)
    except OSError as exc:
        print(f"[FATAL] cannot write parquet to {out_path}: {exc}")
        return 2
    out_sha = sha256_file(out_path)
    out_kb = out_path.stat().st_size / 1024
    print(f"[ OK ] wrote {out_path.name}  ({out_kb:.1f} KB)  sha={out_sha[:32]}...")

    # ---- validate the artifact ON DISK, by re-reading it ------------------
    # Validating the in-memory table would only prove the assembly logic. The
    # thing Experiment 7 will consume is the file, so the file is what is checked.
    print("[STEP 5/6] validating the written artifact")
    check = pq.read_table(out_path)

    if check.num_rows != EXPECTED_ROWS:
        rep.fail("V1.rows", f"output has {check.num_rows} rows, expected {EXPECTED_ROWS}")
    if check.column_names != FINAL_COLUMNS:
        rep.fail("V2.columns", f"output columns {check.column_names} != {FINAL_COLUMNS}")

    # Byte-identity: Arrow-level equality against the frozen file, per column.
    for col in FROZEN_COLUMNS:
        if not check.column(col).equals(frozen_table.column(col)):
            rep.fail("V3.byte_identity", f"column '{col}' differs from the frozen parquet")
        if check.schema.field(col).type != frozen_table.schema.field(col).type:
            rep.fail("V3.dtype", f"column '{col}' dtype changed")

    out_pids = [str(v) for v in check.column("participant_id").to_pylist()]
    if out_pids != parquet_pids:
        rep.fail("V4.ordering", "row ordering differs from the frozen parquet")
    if len(set(out_pids)) != len(out_pids):
        rep.fail("V4.duplicates", "duplicate participant_id in the output")

    feats_out = check.column("features").to_pylist()
    ac_out = check.column("audio_coverage").to_pylist()
    vc_out = check.column("video_coverage").to_pylist()
    for i, (pid, blob) in enumerate(zip(out_pids, feats_out)):
        try:
            parsed = json.loads(blob)
        except json.JSONDecodeError as exc:
            rep.fail("V5.json", f"features JSON does not parse: {exc}", pid)
            continue
        a = parsed.get("audio", {}).get(AUDIO_KEY)
        v = parsed.get("video", {}).get(VISION_KEY)
        if not isinstance(a, list) or len(a) != AUDIO_DIM:
            rep.fail("V6.audio_dim",
                     f"features.audio.{AUDIO_KEY} length "
                     f"{len(a) if isinstance(a, list) else 'n/a'} != {AUDIO_DIM}", pid)
        if not isinstance(v, list) or len(v) != VISION_DIM:
            rep.fail("V6.vision_dim",
                     f"features.video.{VISION_KEY} length "
                     f"{len(v) if isinstance(v, list) else 'n/a'} != {VISION_DIM}", pid)
        # The vectors must be exactly what S2 produced - a join must not perturb.
        src = s2_participants[pid]
        if isinstance(a, list) and a != src["audio"]:
            rep.fail("V7.audio_roundtrip", "audio vector changed in the round trip", pid)
        if isinstance(v, list) and v != src["vision"]:
            rep.fail("V7.vision_roundtrip", "vision vector changed in the round trip", pid)
        if not (isinstance(ac_out[i], float) and math.isfinite(ac_out[i])):
            rep.fail("V8.coverage", "audio_coverage missing or non-finite", pid)
        if not (isinstance(vc_out[i], float) and math.isfinite(vc_out[i])):
            rep.fail("V8.coverage", "video_coverage missing or non-finite", pid)

    print(f"[ OK ] {check.num_rows} rows, {len(FINAL_COLUMNS)} columns, "
          f"frozen columns byte-identical, vectors round-trip exactly")

    # ---- trainer contract --------------------------------------------------
    print("[STEP 6/6] trainer contract")
    trainer_check: Dict[str, Any] = {"checked": False}
    if args.no_trainer_check:
        print("[INFO] skipped (--no-trainer-check)")
    elif not TRAINER_PATH.is_file():
        rep.warn("T1.trainer", f"trainer not found: {TRAINER_PATH}")
    else:
        tsha = sha256_file(TRAINER_PATH)
        if tsha != FROZEN_TRAINER_SHA:
            rep.fail("T1.trainer_sha",
                     f"trainer SHA {tsha} != frozen {FROZEN_TRAINER_SHA}")
        try:
            T = load_trainer()
            records = T.read_parquet_records(str(out_path))
            # Replay the trainer's own extraction and dim inference.
            ds_audio = [T.MultiModalDataset._extract_audio_vec(None, r) for r in records]
            ds_video = [T.MultiModalDataset._extract_video_vec(None, r) for r in records]
            n_a = sum(1 for x in ds_audio if x is not None)
            n_v = sum(1 for x in ds_video if x is not None)
            dims_a = {tuple(x.shape) for x in ds_audio if x is not None}
            dims_v = {tuple(x.shape) for x in ds_video if x is not None}
            if len(records) != EXPECTED_ROWS:
                rep.fail("T2.records", f"trainer loaded {len(records)} records")
            if n_a != EXPECTED_ROWS or n_v != EXPECTED_ROWS:
                rep.fail("T3.extract",
                         f"trainer extracted audio for {n_a}/{EXPECTED_ROWS} and "
                         f"vision for {n_v}/{EXPECTED_ROWS} records")
            if dims_a != {(AUDIO_DIM,)}:
                rep.fail("T4.audio_shape", f"trainer audio shapes {dims_a}")
            if dims_v != {(VISION_DIM,)}:
                rep.fail("T4.vision_shape", f"trainer vision shapes {dims_v}")
            trainer_check = {
                "checked": True, "trainer_sha": tsha,
                "records_loaded": len(records),
                "audio_extracted": n_a, "vision_extracted": n_v,
                "audio_shapes": sorted(str(d) for d in dims_a),
                "vision_shapes": sorted(str(d) for d in dims_v),
            }
            print(f"[ OK ] trainer read {len(records)} records; extracted "
                  f"audio {n_a}/{EXPECTED_ROWS} {sorted(dims_a)}, "
                  f"vision {n_v}/{EXPECTED_ROWS} {sorted(dims_v)}")
        except Exception as exc:  # noqa: BLE001 - report, do not crash the build
            rep.warn("T5.trainer_import",
                     f"could not run the trainer check: {type(exc).__name__}: {exc}")

    # ---- report ------------------------------------------------------------
    elapsed = time.time() - t0
    report = {
        "schema": "daic_multimodal_records_report/1",
        "generated_by": "build_daic_multimodal_records.py",
        "inputs": {
            "parquet": str(frozen_path),
            "parquet_sha256": in_sha,
            "parquet_sha256_pinned": FROZEN_PARQUET_SHA,
            "parquet_sha_matches_pin": in_sha == FROZEN_PARQUET_SHA,
            "features": str(args.features.resolve()),
            "features_sha256": sha256_file(args.features),
            "features_digest": s2.get("findings_digest"),
        },
        "output": {
            "parquet": str(out_path),
            "parquet_sha256": out_sha,
            "size_bytes": out_path.stat().st_size,
            "compression": COMPRESSION,
        },
        "row_count": check.num_rows,
        "feature_dimensions": {"audio_dim": AUDIO_DIM, "vision_dim": VISION_DIM},
        "schema_summary": {
            "columns": [{"name": f.name, "type": str(f.type)} for f in check.schema],
            "frozen_columns": FROZEN_COLUMNS,
            "appended_columns": NEW_COLUMNS,
            "features_keys": {"audio": f"features.audio.{AUDIO_KEY}",
                              "video": f"features.video.{VISION_KEY}"},
            "features_encoding": "JSON string (trainer decodes at lines 89-96)",
        },
        "coverage_summary": {
            "audio_coverage": {"min": min(ac_out), "max": max(ac_out),
                               "mean": sum(ac_out) / len(ac_out)},
            "video_coverage": {"min": min(vc_out), "max": max(vc_out),
                               "mean": sum(vc_out) / len(vc_out)},
        },
        "trainer_contract": trainer_check,
        "totals": {"fail": rep.count(SEV_FAIL), "warn": rep.count(SEV_WARN)},
        "findings": rep.findings,
        "findings_digest": rep.digest(),
        "elapsed_seconds": round(elapsed, 2),
    }
    try:
        report_path.parent.mkdir(parents=True, exist_ok=True)
        with open(report_path, "w", encoding="utf-8") as fh:
            json.dump(report, fh, indent=2, sort_keys=True, ensure_ascii=True)
    except OSError as exc:
        print(f"[FATAL] cannot write report to {report_path}: {exc}")
        return 2

    n_fail, n_warn = rep.count(SEV_FAIL), rep.count(SEV_WARN)
    print("-" * 78)
    print("=== MULTIMODAL RECORDS SUMMARY ===")
    print(f"rows                : {check.num_rows}")
    print(f"columns             : {check.column_names}")
    print(f"audio / vision dim  : {AUDIO_DIM} / {VISION_DIM}")
    print(f"input  parquet sha  : {in_sha}")
    print(f"output parquet sha  : {out_sha}")
    print(f"output size         : {out_kb:.1f} KB")
    print(f"FAIL findings       : {n_fail}")
    print(f"WARN findings       : {n_warn}")
    print(f"findings digest     : {rep.digest()}")
    print(f"elapsed             : {elapsed:.2f}s")
    print(f"report              : {report_path}")

    if n_fail:
        print("\n[FAIL] multimodal parquet is NOT usable.")
        by: Dict[str, int] = {}
        for f in rep.findings:
            if f["severity"] == SEV_FAIL:
                by[f["check"]] = by.get(f["check"], 0) + 1
        for c, k in sorted(by.items()):
            print(f"    {c}: {k}")
        return 1

    print("\n[PASS] multimodal parquet built and validated; "
          "frozen columns byte-identical.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
