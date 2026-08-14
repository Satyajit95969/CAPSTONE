#!/usr/bin/env python3
"""
make_exp7_upload_bundle.py - package Experiment 7 for the Colab GPU run.

Experiment 7 MUST run on GPU: Baseline-CV and Experiments 3-6 are all Colab GPU
results, and run_exp7.py aborts on CPU because device would be a second factor -
which would invalidate the A0 equivalence gate. Measured locally, one fold-run
takes ~312 s on this CPU (100 fold-runs ~= 8.7 hours) against ~18 s/fold-run on
the Colab GPU that produced Exp 6 (~30 minutes).

Every file below is SHA-verified before zipping. The bundle deliberately mirrors
make_exp6_upload_bundle.py: the same class of missing-file failure (a runner
looking for trainer_outputs/baseline_cv/fold_manifest.json that was never
uploaded) has cost this project a full Colab session before.

Usage:
    python make_exp7_upload_bundle.py
    python make_exp7_upload_bundle.py --out C:/Users/DELL/Downloads/exp7_upload.zip
"""

from __future__ import annotations

import argparse
import hashlib
import os
import sys
import zipfile
from typing import Dict, List, Optional

REPO = os.path.dirname(os.path.abspath(__file__))
DEFAULT_OUT = os.path.join(os.path.expanduser("~"), "Downloads", "exp7_upload.zip")

REQUIRED_FILES: List[str] = [
    # frozen trainer - model / dataset / collate / inference / ablation
    "trainer_mentalbert_daic.py",
    # both artifacts: the multimodal one is trained on, the text-only one is the
    # byte-identity reference run_exp7.py checks before training
    "daic_records.parquet",
    "dataset_build/daic_records_multimodal.parquet",
    # the three pre-registered Exp 7 scripts
    "exp7_modality/run_exp7.py",
    "exp7_modality/run_exp7_ablation.py",
    "exp7_modality/aggregate_exp7.py",
    # frozen folds and the A0 gate's reference
    "trainer_outputs/baseline_cv/fold_manifest.json",
    "trainer_outputs/baseline_cv/trivial_control_arm.csv",
    "trainer_outputs/baseline_cv/baseline_cv_summary.json",
    # SHA-gated context comparators
    "trainer_outputs/exp3_convergence/exp3_summary.json",
    "trainer_outputs/exp4_decision_rule/exp4_summary.json",
    "trainer_outputs/exp5_imbalance_objective/exp5_summary.json",
    "trainer_outputs/exp6_loss_rebalance/exp6_summary.json",
    # provenance for the multimodal artifact
    "dataset_build/multimodal_records_report.json",
    "dataset_build/multimodal_verification.json",
]

EXPECTED_SHA: Dict[str, str] = {
    "trainer_mentalbert_daic.py":
        "65b1902e2ba8dd996c6960db1a6595d84fd48500f4d8707cb79688093d7a230b",
    "daic_records.parquet":
        "9a241851760727779fcaa4d50041b8bdc4406fe6b66ddbbd7105f4ce4fc55f00",
    "dataset_build/daic_records_multimodal.parquet":
        "1ac9f53e6102ec0dbaab84dcfdfa3f2e70f2b4a1867a841ea2b7e3ba24c67a95",
    "trainer_outputs/baseline_cv/fold_manifest.json":
        "b9a7a91f1bdd43c78d3cd1ee0c36e4ce3fb8be4b0971dcff3ff62ee5af8fca2f",
    "trainer_outputs/baseline_cv/trivial_control_arm.csv":
        "2ecbfee3225910034a959fe949c7ad027cfa7d41e9bef8040374578ff8c5d572",
    "trainer_outputs/baseline_cv/baseline_cv_summary.json":
        "f065f5e1ff7f8e5ed4d122a8031c9cc12425802e756697d5512a915674871aac",
    "trainer_outputs/exp3_convergence/exp3_summary.json":
        "9dd135b6b79af06a85e64a5aa8c1896d19df8059dd89c053e50e1cffe20af2f9",
    "trainer_outputs/exp4_decision_rule/exp4_summary.json":
        "5defdae2a20d0abc164611e8cbe6b8034e3f65c593d8c9b3e38266a1800aa6f2",
    "trainer_outputs/exp5_imbalance_objective/exp5_summary.json":
        "70ad13ffc4db633419595761c0396fc20dd1b62400ee74238be1f06b73ed48d3",
    "trainer_outputs/exp6_loss_rebalance/exp6_summary.json":
        "bea7478594f6af98943ddda2e997e8bc8780abf57c9c2ee720f9426cd2acc94c",
}

RUN_INSTRUCTIONS = """\
EXPERIMENT 7 - COLAB GPU RUN ORDER

  0. Runtime > Change runtime type > GPU. Confirm:
       import torch; assert torch.cuda.is_available()

  1. Unzip at the repo root so relative paths resolve.

  2. A0 FIRST - it is a HARD GATE:
       python exp7_modality/run_exp7.py --arms A0

  3. Aggregate to evaluate the gate. This ABORTS if A0 did not
     reproduce Baseline-CV:
       python exp7_modality/aggregate_exp7.py

     If it aborts, STOP. Do not run A1/A2/A3; report the diagnostic.

  4. Only if the gate passed:
       python exp7_modality/run_exp7.py --arms A1,A2,A3

  5. Ablation (frozen trainer via its own CLI - never imported, never patched):
       python exp7_modality/run_exp7_ablation.py

  6. Final aggregation:
       python exp7_modality/aggregate_exp7.py

  Expected GPU wall time: ~30 min for the 100 fold-runs (Exp 6 measured
  18.0 s/fold-run), plus ~2 min for the two ablation runs.

  Bring back: trainer_outputs/exp7_modality/ in full.
"""


def sha256(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for b in iter(lambda: fh.read(8192), b""):
            h.update(b)
    return h.hexdigest()


def main() -> int:
    ap = argparse.ArgumentParser(description="Build the Experiment 7 Colab bundle")
    ap.add_argument("--out", default=DEFAULT_OUT)
    ap.add_argument("--skip-sha", action="store_true",
                    help="package anyway if a SHA has legitimately moved")
    a = ap.parse_args()
    os.chdir(REPO)

    print("=" * 74)
    print("EXPERIMENT 7 UPLOAD BUNDLE")
    print("=" * 74)

    # Existence BEFORE any zipping - the Exp 6 lesson.
    missing = [p for p in REQUIRED_FILES if not os.path.isfile(p)]
    if missing:
        print("[FATAL] missing required files:", file=sys.stderr)
        for m in missing:
            print(f"    {m}", file=sys.stderr)
        return 2
    print(f"[OK] all {len(REQUIRED_FILES)} required files present")

    bad = []
    for path, want in EXPECTED_SHA.items():
        got = sha256(path)
        if got != want:
            bad.append((path, want, got))
    if bad:
        print("[FATAL] SHA mismatch on frozen artifacts:", file=sys.stderr)
        for p, w, g in bad:
            print(f"    {p}\n      expected {w}\n      got      {g}", file=sys.stderr)
        if not a.skip_sha:
            return 1
        print("[WARN] --skip-sha: packaging anyway")
    else:
        print(f"[OK] all {len(EXPECTED_SHA)} SHA-pinned artifacts verified")

    out = os.path.abspath(a.out)
    os.makedirs(os.path.dirname(out), exist_ok=True)
    with zipfile.ZipFile(out, "w", zipfile.ZIP_DEFLATED) as z:
        for p in REQUIRED_FILES:
            z.write(p, p)
        z.writestr("EXP7_RUN_INSTRUCTIONS.txt", RUN_INSTRUCTIONS)
        z.writestr("EXP7_SHA_MANIFEST.txt",
                   "\n".join(f"{sha256(p)}  {p}" for p in REQUIRED_FILES) + "\n")

    size_mb = os.path.getsize(out) / (1024 * 1024)
    print(f"[OK] wrote {out}  ({size_mb:.2f} MB, {len(REQUIRED_FILES) + 2} entries)")
    print("\n" + RUN_INSTRUCTIONS)
    return 0


if __name__ == "__main__":
    sys.exit(main())
