#!/usr/bin/env python
"""
Build the Experiment 5 Colab upload bundle.

Collects ONLY the files required to run Phase 14 / Experiment 5 on a fresh Colab
runtime, preserving the repository's directory structure, and writes
exp5_upload.zip at the repository root.

WHY THIS EXISTS INSTEAD OF Compress-Archive
-------------------------------------------
Windows PowerShell 5.1's Compress-Archive stores entry names with BACKSLASH
separators, violating ZIP APPNOTE 4.4.17.1 (which mandates '/'). Linux treats the
backslash as a literal filename character, producing flat files named
"trainer_outputs\\baseline_cv\\fold_manifest.json". That failure already cost this
project one Colab session. This writer sets every arcname explicitly with forward
slashes and re-reads the RAW central directory to prove no backslash survived.

WHAT THE BUNDLE CARRIES, AND WHY
--------------------------------
  trainer_mentalbert_daic.py            frozen trainer - model/dataset/inference/AdamW
  daic_records.parquet                  frozen dataset (also read by verify_exp5)
  exp5_imbalance_objective/*.py         the three pre-registered scripts
  baseline_cv/fold_manifest.json        the frozen folds
  baseline_cv/trivial_control_arm.csv   SHA-gated by all three scripts
  baseline_cv/baseline_cv_summary.json  the W0 equivalence gate's reference
  baseline_cv/runs/ (30 files)          aggregation self-test + verify L3
  exp4_decision_rule/exp4_summary.json  Criterion 3 / 6 comparator
  exp3_convergence/exp3_summary.json    SHA-gated context comparator

`baseline_cv_summary.md` is deliberately NOT shipped: no Experiment 5 script
reads it, and the verifier's containment check derives its expectation from the
SHA-gated inputs rather than from a raw file count. That is the fix applied after
the Exp 3 `J2` false failure.

INTEGRITY
---------
Every required file is checked for existence BEFORE any zipping begins, and the
seven frozen inputs are SHA-256 verified against the digests pre-registered in
PHASE_14_EXP5_DESIGN.md Appendix A.1. The three Experiment 5 scripts are verified
against Appendix A.2 - unlike the Exp 3 bundler, these ARE hard-gated, because
Experiment 5's pre-registration is complete and any drift would make a result
inadmissible.

Usage (from the repository root):
    python make_exp5_upload_bundle.py
"""
from __future__ import annotations

import hashlib
import os
import re
import sys
import zipfile
from typing import Dict, List, Tuple

OUT_ZIP = "exp5_upload.zip"
RUNS_DIR = "trainer_outputs/baseline_cv/runs"
EXPECTED_RUNS = 30                      # 5 per-repeat metrics + 25 per-fold predictions
BACKSLASH = chr(92)

REQUIRED_FILES: List[str] = [
    "trainer_mentalbert_daic.py",
    "daic_records.parquet",
    "exp5_imbalance_objective/run_exp5.py",
    "exp5_imbalance_objective/aggregate_exp5.py",
    "exp5_imbalance_objective/verify_exp5.py",
    "trainer_outputs/baseline_cv/fold_manifest.json",
    "trainer_outputs/baseline_cv/trivial_control_arm.csv",
    "trainer_outputs/baseline_cv/baseline_cv_summary.json",
    "trainer_outputs/exp4_decision_rule/exp4_summary.json",
    "trainer_outputs/exp3_convergence/exp3_summary.json",
]

# Appendix A.1 - frozen inputs and comparators.
FROZEN_INPUT_SHA: Dict[str, str] = {
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
    "trainer_outputs/exp4_decision_rule/exp4_summary.json":
        "5defdae2a20d0abc164611e8cbe6b8034e3f65c593d8c9b3e38266a1800aa6f2",
    "trainer_outputs/exp3_convergence/exp3_summary.json":
        "9dd135b6b79af06a85e64a5aa8c1896d19df8059dd89c053e50e1cffe20af2f9",
}

# Appendix A.2 - implementation code. Hard-gated: the pre-registration is
# complete, so any drift would make a result inadmissible.
FROZEN_SCRIPT_SHA: Dict[str, str] = {
    "exp5_imbalance_objective/run_exp5.py":
        "4234508133c74c413f5a6498de0f3993e55bf0e579cadc3f4d0380c01a5cbb7f",
    "exp5_imbalance_objective/aggregate_exp5.py":
        "66bbac11345445740b0f31cf7ae148b9bd8dea651b5541a31e6def68e701ca57",
    "exp5_imbalance_objective/verify_exp5.py":
        "57843f65a6b83c9015301fb37f34b2c8652f8b800f8efbcd495b86b017d2b6b8",
}


def sha256(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for block in iter(lambda: fh.read(8192), b""):
            h.update(block)
    return h.hexdigest()


def abort(message: str) -> "NoReturn":  # noqa: F821
    print(f"\n[ABORT] {message}", file=sys.stderr)
    raise SystemExit(1)


def collect() -> List[Tuple[str, str]]:
    """Return (source_path, arcname) pairs, verifying every input first."""
    if not os.path.isdir("exp5_imbalance_objective"):
        abort("exp5_imbalance_objective/ not found - run this from the repository root.")
    if not os.path.isdir(RUNS_DIR):
        abort(f"{RUNS_DIR} not found - the frozen Baseline-CV runs are required.")

    missing = [p for p in REQUIRED_FILES if not os.path.isfile(p)]
    if missing:
        abort("missing required file(s):\n        " + "\n        ".join(missing))

    run_files = sorted(f for f in os.listdir(RUNS_DIR)
                       if os.path.isfile(os.path.join(RUNS_DIR, f)))
    if len(run_files) != EXPECTED_RUNS:
        abort(f"{RUNS_DIR} holds {len(run_files)} files, expected {EXPECTED_RUNS} "
              "(5 rep*_metrics.csv + 25 rep*_fold*_preds.csv).")

    bad = []
    for path, want in {**FROZEN_INPUT_SHA, **FROZEN_SCRIPT_SHA}.items():
        got = sha256(path)
        if got != want:
            bad.append(f"{path}\n            expected {want}\n            got      {got}")
    if bad:
        abort("SHA mismatch against PHASE_14_EXP5_DESIGN.md Appendix A - the bundle "
              "would carry code or data that was never pre-registered:\n        "
              + "\n        ".join(bad))

    pairs = [(p, p) for p in REQUIRED_FILES]
    pairs += [(os.path.join(RUNS_DIR, f).replace(os.sep, "/"), f"{RUNS_DIR}/{f}")
              for f in run_files]
    return pairs


def raw_entry_names(zip_path: str) -> List[str]:
    """Read entry names straight from the ZIP central directory.

    Python's zipfile normalises os.sep to '/' when READING on Windows, which
    would mask exactly the defect this check exists to catch.
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
    print("EXPERIMENT 5 UPLOAD BUNDLE")
    print("=" * 70)
    print(f"fixed inputs        : {len(REQUIRED_FILES)}")
    print(f"baseline runs/      : {len(pairs) - len(REQUIRED_FILES)}")
    print(f"files included      : {len(pairs)}")
    print(f"output              : {OUT_ZIP}")
    print(f"size                : {os.path.getsize(OUT_ZIP):,} bytes")
    print(f"sha256              : {sha256(OUT_ZIP)}")
    print("backslash entries   : 0")
    print()
    print("frozen inputs verified against Appendix A.1:")
    for path in FROZEN_INPUT_SHA:
        print(f"   OK  {path}")
    print()
    print("experiment scripts verified against Appendix A.2:")
    for path in FROZEN_SCRIPT_SHA:
        print(f"   OK  {path}")
    print()
    print("Upload exp5_upload.zip to Colab and extract to /content/be_project.")


if __name__ == "__main__":
    main()
