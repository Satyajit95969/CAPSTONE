#!/usr/bin/env python3
"""
make_exp8_upload_bundle.py - package the Experiment 8 RESULT artifacts.

Authoritative specification: PHASE_17_EXP8_DESIGN.md (FROZEN), SS 21.

Unlike the Exp 7 bundle (which shipped code and frozen inputs TO Colab),
Experiment 8 is a local mechanism-level experiment: it needs no GPU and no
remote execution (design SS 15). This bundle therefore packages the FINAL RESULT
ARTIFACTS for archival and review.

INCLUDED - only Exp 8 result artifacts plus the frozen specification that
governs them:
    trainer_outputs/exp8_dp_mechanism/exp8_metrics.csv
    trainer_outputs/exp8_dp_mechanism/exp8_summary.json
    trainer_outputs/exp8_dp_mechanism/exp8_delta.json
    trainer_outputs/exp8_dp_mechanism/exp8_receipt.json          (if present)
    trainer_outputs/exp8_dp_mechanism/d0_reference_check.json
    trainer_outputs/exp8_dp_mechanism/exp8_verification.json
    PHASE_17_EXP8_DESIGN.md

EXCLUDED, deliberately:
    * mutable project files, source trees, datasets, checkpoints
    * secure_store/ and receipts/ working directories written during the run
      (they contain key-derived material and are not results)
    * anything not produced by Exp 8

The bundle refuses to build if any required result artifact is missing, so a
partial or aborted run can never be packaged as a complete result.

Usage:
    python make_exp8_upload_bundle.py
    python make_exp8_upload_bundle.py --out C:/Users/DELL/Downloads/exp8_results.zip
"""

from __future__ import annotations

import argparse
import hashlib
import os
import sys
import zipfile
from typing import List

REPO = os.path.dirname(os.path.abspath(__file__))
EXP_DIR = os.path.join("trainer_outputs", "exp8_dp_mechanism")
DEFAULT_OUT = os.path.join(os.path.expanduser("~"), "Downloads", "exp8_results.zip")

REQUIRED: List[str] = [
    os.path.join(EXP_DIR, "exp8_metrics.csv"),
    os.path.join(EXP_DIR, "exp8_summary.json"),
    os.path.join(EXP_DIR, "exp8_delta.json"),
    # design SS 21 artifact, produced by run_exp8.py. REQUIRED, not optional:
    # P4 reproducibility is derived from it, so a bundle without it would carry
    # an unverifiable verdict.
    os.path.join(EXP_DIR, "exp8_receipt.json"),
    os.path.join(EXP_DIR, "d0_reference_check.json"),
    os.path.join(EXP_DIR, "exp8_verification.json"),
    "PHASE_17_EXP8_DESIGN.md",
]
OPTIONAL: List[str] = []


def sha256_file(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for b in iter(lambda: fh.read(8192), b""):
            h.update(b)
    return h.hexdigest()


def main() -> int:
    ap = argparse.ArgumentParser(description="Build the Experiment 8 result bundle")
    ap.add_argument("--out", default=DEFAULT_OUT)
    a = ap.parse_args()
    os.chdir(REPO)

    print("=" * 74)
    print("EXPERIMENT 8 RESULT BUNDLE")
    print("=" * 74)

    missing = [p for p in REQUIRED if not os.path.isfile(p)]
    if missing:
        print("[FATAL] missing required Exp 8 result artifacts:", file=sys.stderr)
        for m in missing:
            print(f"    {m}", file=sys.stderr)
        print("        (run run_exp8.py, aggregate_exp8.py and verify_exp8.py first)",
              file=sys.stderr)
        return 2
    files = REQUIRED + [p for p in OPTIONAL if os.path.isfile(p)]
    print(f"[OK] {len(files)} artifact(s) present")

    out = os.path.abspath(a.out)
    os.makedirs(os.path.dirname(out), exist_ok=True)
    manifest = "\n".join(f"{sha256_file(p)}  {p}" for p in files) + "\n"
    with zipfile.ZipFile(out, "w", zipfile.ZIP_DEFLATED) as z:
        for p in files:
            z.write(p, p)
        z.writestr("EXP8_SHA_MANIFEST.txt", manifest)

    print(f"[OK] wrote {out}  ({os.path.getsize(out)/1024:.1f} KB, "
          f"{len(files)+1} entries)")
    print(f"[OK] bundle sha256: {sha256_file(out)}")
    print("\n=== EXP8_SHA_MANIFEST.txt ===")
    print(manifest)
    return 0


if __name__ == "__main__":
    sys.exit(main())
