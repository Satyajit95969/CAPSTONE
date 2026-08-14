#!/usr/bin/env python
"""
Build the Experiment 3 Colab upload bundle.

Collects ONLY the files required to rerun Phase 13 / Experiment 3 on a fresh
Colab runtime, preserving the repository's directory structure, and writes
exp3_upload.zip at the repository root.

WHY THIS EXISTS INSTEAD OF Compress-Archive
-------------------------------------------
Windows PowerShell 5.1's Compress-Archive stores entry names with BACKSLASH
separators, violating ZIP APPNOTE 4.4.17.1 (which mandates '/'). Windows
extractors tolerate it; Linux does not - it treats the backslash as a literal
filename character and produces flat files called
"trainer_outputs\\baseline_cv\\fold_manifest.json". That failure has already
cost this project one Colab session. This writer sets every arcname explicitly
with forward slashes and then re-reads the RAW central directory to prove no
backslash survived.

INTEGRITY
---------
Every required file is checked for existence BEFORE any zipping begins, so a
missing input fails immediately with a readable list rather than producing a
half-formed archive. The five frozen data inputs are additionally SHA-256
verified against the values pre-registered in PHASE_13_EXP3_DESIGN.md
Appendix A - a corrupted or substituted input is caught here rather than after
40 minutes of GPU time.

The three Experiment 3 scripts are reported informationally, not hard-gated:
verify_exp3.py was deliberately patched after pre-registration (the J2
artifact-count fix), so its digest legitimately differs from Appendix A and
carries a recorded erratum.

Usage (from the repository root):
    python make_exp3_upload_bundle.py
"""
from __future__ import annotations

import hashlib
import os
import re
import sys
import zipfile
from typing import Dict, List, Tuple

OUT_ZIP = "exp3_upload.zip"
RUNS_DIR = "trainer_outputs/baseline_cv/runs"
EXPECTED_RUNS = 30                      # 5 per-repeat metrics + 25 per-fold predictions
BACKSLASH = chr(92)

# Fixed inputs, in the order they will be reported.
REQUIRED_FILES: List[str] = [
    "trainer_mentalbert_daic.py",
    "daic_records.parquet",
    "exp3_convergence/run_exp3.py",
    "exp3_convergence/aggregate_exp3.py",
    "exp3_convergence/verify_exp3.py",
    "trainer_outputs/baseline_cv/baseline_cv_summary.json",
    "trainer_outputs/baseline_cv/fold_manifest.json",
    "trainer_outputs/baseline_cv/trivial_control_arm.csv",
]

# Hard-gated: the frozen data the experiment consumes. A mismatch means the
# bundle would carry non-frozen inputs and the run would not be comparable.
FROZEN_SHA: Dict[str, str] = {
    "trainer_mentalbert_daic.py":
        "65b1902e2ba8dd996c6960db1a6595d84fd48500f4d8707cb79688093d7a230b",
    "daic_records.parquet":
        "9a241851760727779fcaa4d50041b8bdc4406fe6b66ddbbd7105f4ce4fc55f00",
    "trainer_outputs/baseline_cv/fold_manifest.json":
        "b9a7a91f1bdd43c78d3cd1ee0c36e4ce3fb8be4b0971dcff3ff62ee5af8fca2f",
    "trainer_outputs/baseline_cv/trivial_control_arm.csv":
        "2ecbfee3225910034a959fe949c7ad027cfa7d41e9bef8040374578ff8c5d572",
    "trainer_outputs/baseline_cv/baseline_cv_summary.json":
        "f065f5e1ff7f8e5ed4d122a8031c9cc12425802e756697d5512a915674871aac",
}

# Reported, not gated (see module docstring).
SCRIPT_FILES: List[str] = [
    "exp3_convergence/run_exp3.py",
    "exp3_convergence/aggregate_exp3.py",
    "exp3_convergence/verify_exp3.py",
]


def sha256(path: str) -> str:
    """SHA-256 of a file, streamed so large inputs do not load into memory."""
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for block in iter(lambda: fh.read(8192), b""):
            h.update(block)
    return h.hexdigest()


def abort(message: str) -> "NoReturn":  # noqa: F821
    """Stop before writing anything. A half-formed bundle is worse than none."""
    print(f"\n[ABORT] {message}", file=sys.stderr)
    raise SystemExit(1)


def collect() -> List[Tuple[str, str]]:
    """Return (source_path, arcname) pairs, verifying every input first.

    arcname always uses forward slashes and mirrors the repository layout, so
    extraction on Colab reproduces the structure the Experiment 3 scripts expect.
    """
    if not os.path.isdir("exp3_convergence"):
        abort("exp3_convergence/ not found - run this from the repository root.")
    if not os.path.isdir(RUNS_DIR):
        abort(f"{RUNS_DIR} not found - the frozen Baseline-CV runs are required.")

    missing = [p for p in REQUIRED_FILES if not os.path.isfile(p)]
    if missing:
        abort("missing required file(s):\n        " + "\n        ".join(missing))

    run_files = sorted(f for f in os.listdir(RUNS_DIR)
                       if os.path.isfile(os.path.join(RUNS_DIR, f)))
    if len(run_files) != EXPECTED_RUNS:
        abort(f"{RUNS_DIR} holds {len(run_files)} files, expected {EXPECTED_RUNS} "
              "(5 rep*_metrics.csv + 25 rep*_fold*_preds.csv). The frozen "
              "Baseline-CV runs are incomplete.")

    bad_sha = []
    for path, want in FROZEN_SHA.items():
        got = sha256(path)
        if got != want:
            bad_sha.append(f"{path}\n            expected {want}\n            got      {got}")
    if bad_sha:
        abort("frozen input SHA mismatch - the bundle would carry non-frozen data:\n"
              "        " + "\n        ".join(bad_sha))

    pairs = [(p, p) for p in REQUIRED_FILES]
    pairs += [(os.path.join(RUNS_DIR, f).replace(os.sep, "/"), f"{RUNS_DIR}/{f}")
              for f in run_files]
    return pairs


def raw_entry_names(zip_path: str) -> List[str]:
    """Read entry names straight from the ZIP central directory.

    Python's zipfile normalises os.sep to '/' when READING on Windows, which
    would mask exactly the defect this check exists to catch. Parsing the raw
    PK\\x01\\x02 records bypasses that normalisation.
    """
    raw = open(zip_path, "rb").read()
    names = []
    for m in re.finditer(b"PK\x01\x02", raw):
        off = m.start()
        n = int.from_bytes(raw[off + 28:off + 30], "little")
        names.append(raw[off + 46:off + 46 + n].decode("utf-8", "replace"))
    return names


def main() -> None:
    pairs = collect()

    for _src, arc in pairs:
        if BACKSLASH in arc:
            abort(f"arcname contains a backslash: {arc}")

    if os.path.exists(OUT_ZIP):
        os.remove(OUT_ZIP)
    with zipfile.ZipFile(OUT_ZIP, "w", zipfile.ZIP_DEFLATED) as z:
        for src, arc in pairs:
            z.write(src, arcname=arc)

    names = raw_entry_names(OUT_ZIP)
    bad = [n for n in names if BACKSLASH in n]
    if bad:
        abort(f"backslash separators survived into the archive: {bad[:5]}")
    if len(names) != len(pairs):
        abort(f"archive holds {len(names)} entries, expected {len(pairs)}")
    with zipfile.ZipFile(OUT_ZIP) as z:
        if z.testzip() is not None:
            abort("archive failed its own integrity test")

    print("=" * 70)
    print("EXPERIMENT 3 UPLOAD BUNDLE")
    print("=" * 70)
    print(f"fixed inputs        : {len(REQUIRED_FILES)}")
    print(f"baseline runs/      : {len(pairs) - len(REQUIRED_FILES)}")
    print(f"files included      : {len(pairs)}")
    print(f"output              : {OUT_ZIP}")
    print(f"size                : {os.path.getsize(OUT_ZIP):,} bytes")
    print(f"sha256              : {sha256(OUT_ZIP)}")
    print(f"backslash entries   : 0")
    print()
    print("frozen inputs verified against PHASE_13_EXP3_DESIGN.md Appendix A:")
    for path in FROZEN_SHA:
        print(f"   OK  {path}")
    print()
    print("experiment scripts (reported, not gated):")
    for path in SCRIPT_FILES:
        print(f"   {sha256(path)}  {path}")
    print()
    print("Upload exp3_upload.zip to Colab and extract to /content/be_project.")


if __name__ == "__main__":
    main()
