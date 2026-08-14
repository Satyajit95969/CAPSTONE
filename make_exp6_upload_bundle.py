#!/usr/bin/env python
"""
Build the Experiment 6 Colab upload bundle.

Collects ONLY the files required to run Phase 15 / Experiment 6 on a fresh Colab
runtime, preserving the repository's directory structure exactly as
run_exp6.py / aggregate_exp6.py / verify_exp6.py expect to find it.

WHY THIS EXISTS INSTEAD OF Compress-Archive
-------------------------------------------
Windows PowerShell 5.1's Compress-Archive stores entry names with BACKSLASH
separators, violating ZIP APPNOTE 4.4.17.1 (which mandates '/'). Linux treats the
backslash as a literal filename character, producing flat files named
"trainer_outputs\\baseline_cv\\fold_manifest.json". That failure already cost this
project one Colab session. This writer sets every arcname explicitly with forward
slashes and re-reads the RAW central directory to prove no backslash survived.

WHAT THE BUNDLE CARRIES, AND WHY EACH FILE IS NEEDED
-----------------------------------------------------
  trainer_mentalbert_daic.py            frozen trainer - model/dataset/collate/
                                        inference/AdamW, imported by run_exp6
  daic_records.parquet                  frozen dataset; also read by verify_exp6
                                        to re-derive per-fold training means
  exp6_loss_rebalance/*.py              the three pre-registered scripts
  baseline_cv/fold_manifest.json        the frozen folds
  baseline_cv/trivial_control_arm.csv   SHA-gated by run_exp6
  baseline_cv/baseline_cv_summary.json  the L0 equivalence gate's reference
  baseline_cv/runs/ (30 files)          aggregation self-test + verify_exp6 P3
  exp4_decision_rule/exp4_summary.json  SHA-gated context comparator
  exp3_convergence/exp3_summary.json    SHA-gated context comparator
  exp5_imbalance_objective/
      exp5_summary.json                 C3's binding comparator, SHA-gated in A.1
      exp5_metrics.csv                  PER-FOLD Exp 5 values, required by
                                        aggregate_exp6's paired comparison and by
                                        verify_exp6 check I0/I2

NO GENERATED ARTIFACT IS SHIPPED. `trainer_outputs/exp6_loss_rebalance/` is
created by the run on Colab and must not exist beforehand; this bundler refuses
to package anything from it. `baseline_cv_summary.md` is also deliberately NOT
shipped: no Experiment 6 script reads it, and verify_exp6's containment check
derives its expectation from the SHA-gated inputs rather than from a raw file
count. That is the fix applied after the Exp 3 `J2` false failure.

INTEGRITY
---------
Every required file is checked for existence BEFORE any zipping begins. The eight
frozen inputs are SHA-256 verified against the digests pre-registered in
PHASE_15_EXP6_DESIGN.md Appendix A.1, and the three Experiment 6 scripts against
Appendix A.2 - both hard-gated, because Experiment 6's pre-registration is
complete and any drift would make a result inadmissible.

`exp5_metrics.csv` is additionally pinned here. Appendix A.1 gates
`exp5_summary.json` rather than the CSV, so this is a BUNDLER-level integrity
check, not a pre-registration gate; aggregate_exp6.py independently re-derives the
CSV's aggregate and compares it to the gated summary at runtime.

Usage (from the repository root):
    python make_exp6_upload_bundle.py
    python make_exp6_upload_bundle.py --out C:/Users/DELL/Downloads/exp6_upload.zip
"""
from __future__ import annotations

import argparse
import hashlib
import os
import re
import sys
import zipfile
from typing import Dict, List, Tuple

DEFAULT_OUT = os.path.join(os.path.expanduser("~"), "Downloads", "exp6_results.zip")
RUNS_DIR = "trainer_outputs/baseline_cv/runs"
EXPECTED_RUNS = 30                      # 5 per-repeat metrics + 25 per-fold predictions
EXP6_OUTPUT_DIR = "trainer_outputs/exp6_loss_rebalance"
BACKSLASH = chr(92)

REQUIRED_FILES: List[str] = [
    "trainer_mentalbert_daic.py",
    "daic_records.parquet",
    "exp6_loss_rebalance/run_exp6.py",
    "exp6_loss_rebalance/aggregate_exp6.py",
    "exp6_loss_rebalance/verify_exp6.py",
    "trainer_outputs/baseline_cv/fold_manifest.json",
    "trainer_outputs/baseline_cv/trivial_control_arm.csv",
    "trainer_outputs/baseline_cv/baseline_cv_summary.json",
    "trainer_outputs/exp4_decision_rule/exp4_summary.json",
    "trainer_outputs/exp3_convergence/exp3_summary.json",
    "trainer_outputs/exp5_imbalance_objective/exp5_summary.json",
    "trainer_outputs/exp5_imbalance_objective/exp5_metrics.csv",
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
    "trainer_outputs/exp5_imbalance_objective/exp5_summary.json":
        "70ad13ffc4db633419595761c0396fc20dd1b62400ee74238be1f06b73ed48d3",
}

# Appendix A.2 - implementation code. Hard-gated: the pre-registration is
# complete, so any drift would make a result inadmissible.
FROZEN_SCRIPT_SHA: Dict[str, str] = {
    "exp6_loss_rebalance/run_exp6.py":
        "5933bc29eff056c9f976c67cde51edb2b311461b3368bd276dc2024cbdd5c5f2",
    "exp6_loss_rebalance/aggregate_exp6.py":
        "546ad22f89ef3581aeccbbf930286a2d7dc06cc7f0d009cc9559160675194799",
    "exp6_loss_rebalance/verify_exp6.py":
        "fc7c0f742cbd062728c17cc4a3cd53e226330d2cee3c640862d26694aa74d61d",
}

# Bundler-level pin, NOT a pre-registration gate (see module docstring).
FROZEN_SUPPORT_SHA: Dict[str, str] = {
    "trainer_outputs/exp5_imbalance_objective/exp5_metrics.csv":
        "a1692ac4cceee22f8a88638ecbaf007af706c39186427e519998aec7f60bdc84",
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
    if not os.path.isdir("exp6_loss_rebalance"):
        abort("exp6_loss_rebalance/ not found - run this from the repository root.")
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
    for path, want in {**FROZEN_INPUT_SHA, **FROZEN_SCRIPT_SHA,
                       **FROZEN_SUPPORT_SHA}.items():
        got = sha256(path)
        if got != want:
            bad.append(f"{path}\n            expected {want}\n            got      {got}")
    if bad:
        abort("SHA mismatch against PHASE_15_EXP6_DESIGN.md Appendix A - the bundle "
              "would carry code or data that was never pre-registered:\n        "
              + "\n        ".join(bad))

    pairs = [(p, p) for p in REQUIRED_FILES]
    pairs += [(os.path.join(RUNS_DIR, f).replace(os.sep, "/"), f"{RUNS_DIR}/{f}")
              for f in run_files]

    # No generated Experiment 6 artifact may be shipped. The output directory is
    # created by the run itself; shipping a stale copy would let a resumed run
    # skip fold-runs it never actually performed.
    strays = [a for _s, a in pairs if a.startswith(EXP6_OUTPUT_DIR)]
    if strays:
        abort(f"bundle would contain generated Exp 6 artifacts: {strays[:5]}")
    if os.path.isdir(EXP6_OUTPUT_DIR) and os.listdir(EXP6_OUTPUT_DIR):
        print(f"[WARN] {EXP6_OUTPUT_DIR} exists locally and is non-empty. Nothing from "
              "it is being packaged, but Experiment 6 may already have been run.",
              file=sys.stderr)
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
    ap = argparse.ArgumentParser(
        description="Build the Phase 15 / Experiment 6 Colab upload bundle.")
    ap.add_argument("--out", default=DEFAULT_OUT,
                    help=f"output zip path (default: {DEFAULT_OUT})")
    a = ap.parse_args()
    out_zip = os.path.abspath(a.out)

    pairs = collect()
    for _src, arc in pairs:
        if BACKSLASH in arc:
            abort(f"arcname contains a backslash: {arc}")

    parent = os.path.dirname(out_zip)
    if parent and not os.path.isdir(parent):
        abort(f"output directory does not exist: {parent}")
    if os.path.exists(out_zip):
        os.remove(out_zip)
    with zipfile.ZipFile(out_zip, "w", zipfile.ZIP_DEFLATED) as z:
        for src, arc in pairs:
            z.write(src, arcname=arc)

    names = raw_entry_names(out_zip)
    bad = [n for n in names if BACKSLASH in n]
    if bad:
        abort(f"backslash separators survived into the archive: {bad[:5]}")
    if len(names) != len(pairs):
        abort(f"archive holds {len(names)} entries, expected {len(pairs)}")
    with zipfile.ZipFile(out_zip) as z:
        if z.testzip() is not None:
            abort("archive failed its own integrity test")
        # every entry must round-trip to the bytes on disk
        for src, arc in pairs:
            if hashlib.sha256(z.read(arc)).hexdigest() != sha256(src):
                abort(f"archived content differs from disk: {arc}")

    dirs: Dict[str, int] = {}
    for _s, arc in pairs:
        dirs[os.path.dirname(arc) or "<root>"] = dirs.get(os.path.dirname(arc) or "<root>", 0) + 1

    print("=" * 74)
    print("EXPERIMENT 6 UPLOAD BUNDLE")
    print("=" * 74)
    print(f"fixed inputs        : {len(REQUIRED_FILES)}")
    print(f"baseline runs/      : {len(pairs) - len(REQUIRED_FILES)}")
    print(f"files included      : {len(pairs)}")
    print(f"output              : {out_zip}")
    print(f"size                : {os.path.getsize(out_zip):,} bytes")
    print(f"sha256              : {sha256(out_zip)}")
    print("backslash entries   : 0")
    print("generated artifacts : 0 (none shipped)")
    print()
    print("directory layout:")
    for d in sorted(dirs):
        print(f"   {d + '/':52s} {dirs[d]:>3} file(s)")
    print()
    print("frozen inputs verified against Appendix A.1:")
    for path in FROZEN_INPUT_SHA:
        print(f"   OK  {path}")
    print()
    print("experiment scripts verified against Appendix A.2:")
    for path in FROZEN_SCRIPT_SHA:
        print(f"   OK  {path}")
    print()
    print("bundler-level pin (not a pre-registration gate):")
    for path in FROZEN_SUPPORT_SHA:
        print(f"   OK  {path}")
    print()
    print(f"Upload {os.path.basename(out_zip)} to Colab and extract to /content/be_project.")


if __name__ == "__main__":
    main()
