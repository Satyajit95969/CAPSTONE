#!/usr/bin/env python3
"""
make_exp10_upload_bundle.py - package Experiment 10 for Colab, or its results.

Authoritative specification: PHASE_19_EXP10_DESIGN.md (FROZEN), SS 19, SS 20, SS 24.

TWO MODES:

  (default)   UPLOAD bundle - the code and frozen inputs the GPU run needs.
              Deliberately EXCLUDES every model checkpoint and delta, every
              Exp 7 / Exp 8 / Exp 9 result artifact, and the text-only Exp 9
              transport machinery. Experiment 10 has no DP, no masking and no
              payload (design SS 5), so none of that belongs in the bundle.

  --results   RESULT bundle - the Exp 10 output artifacts plus the frozen
              specification that governs them. Refuses to build if a required
              result artifact is missing, so a partial or aborted run can never
              be packaged as a complete result.

Every shipped file is SHA-verified before zipping, forbidden paths are asserted
absent, and a STATIC AST-BASED import-closure check proves every project-local
module reachable from the shipped code is actually shipped. That check exists
because the first Exp 9 bundle silently omitted the eight-file dp_agent closure
and failed at import on Colab.

Usage:
    python make_exp10_upload_bundle.py
    python make_exp10_upload_bundle.py --results
    python make_exp10_upload_bundle.py --out C:/Users/DELL/Downloads/exp10_upload.zip
"""

from __future__ import annotations

import argparse
import ast
import hashlib
import os
import sys
import zipfile
from typing import Dict, List

REPO = os.path.dirname(os.path.abspath(__file__))
DEFAULT_UPLOAD = os.path.join(os.path.expanduser("~"), "Downloads",
                              "exp10_upload.zip")
DEFAULT_RESULTS = os.path.join(os.path.expanduser("~"), "Downloads",
                               "exp10_results.zip")
EXP_DIR = "trainer_outputs/multimodal_exp"

# --------------------------------------------------------------------------
# Colab needs exactly this. Note what is ABSENT and why: Experiment 10 imports
# no dp_agent, so the centralised_receipts / centralized_secure_store /
# installer.security closure that Experiment 9 required is not needed here. The
# import-closure check below proves that rather than assuming it.
# --------------------------------------------------------------------------
UPLOAD_FILES: List[str] = [
    "PHASE_19_EXP10_DESIGN.md",
    "trainer_mentalbert_daic.py",
    "exp10_multimodal_conditioning/exp10_common.py",
    "exp10_multimodal_conditioning/run_exp10.py",
    # design SS 8 - the multimodal artifact is what M1/M2 train on; the text
    # parquet ships because SS 12.4 requires the byte-identity assertion.
    "dataset_build/daic_records_multimodal.parquet",
    "daic_records.parquet",
    "trainer_outputs/baseline_cv/fold_manifest.json",
    "trainer_outputs/baseline_cv/baseline_cv_summary.json",
    # RUNTIME frozen-input dependencies. run_exp10.py:308 calls
    # C.snapshot_frozen() with NO subset, so its pre-run gate covers EVERY key in
    # C.FROZEN_SHA and hard-aborts on any that is missing. These four are frozen
    # INPUTS (a Baseline-CV control artifact and three prior design/closure
    # documents) - not Exp 10 results - and the runner cannot start without them.
    #
    # The AST import-closure check below covers PYTHON imports only; it cannot
    # see a data dependency read at runtime. That gap is why the first bundle
    # passed its own audit and still aborted Cell 9.
    "trainer_outputs/baseline_cv/trivial_control_arm.csv",
    "PHASE_16_EXP7_DESIGN.md",
    "FINAL_EXP7_CLOSURE.md",
    "PHASE_18_EXP9_DESIGN.md",
]

RESULT_FILES: List[str] = [
    f"{EXP_DIR}/exp10_metrics.csv",
    f"{EXP_DIR}/exp10_conditioning_manifest.json",
    f"{EXP_DIR}/exp10_receipt.json",
    f"{EXP_DIR}/exp10_summary.json",
    f"{EXP_DIR}/exp10_acceptance.csv",
    f"{EXP_DIR}/exp10_verification.json",
    "PHASE_19_EXP10_DESIGN.md",
]

EXPECTED_SHA: Dict[str, str] = {
    "PHASE_19_EXP10_DESIGN.md":
        "71fdb1b3c6b3c4bc402e688769a8aaae447283dfbe5ed62dcc3076f9c7b59a20",
    "trainer_mentalbert_daic.py":
        "65b1902e2ba8dd996c6960db1a6595d84fd48500f4d8707cb79688093d7a230b",
    "dataset_build/daic_records_multimodal.parquet":
        "1ac9f53e6102ec0dbaab84dcfdfa3f2e70f2b4a1867a841ea2b7e3ba24c67a95",
    "daic_records.parquet":
        "9a241851760727779fcaa4d50041b8bdc4406fe6b66ddbbd7105f4ce4fc55f00",
    "trainer_outputs/baseline_cv/fold_manifest.json":
        "b9a7a91f1bdd43c78d3cd1ee0c36e4ce3fb8be4b0971dcff3ff62ee5af8fca2f",
    "trainer_outputs/baseline_cv/baseline_cv_summary.json":
        "f065f5e1ff7f8e5ed4d122a8031c9cc12425802e756697d5512a915674871aac",
    # the four runtime frozen-input dependencies of run_exp10.py's pre-run gate
    "trainer_outputs/baseline_cv/trivial_control_arm.csv":
        "2ecbfee3225910034a959fe949c7ad027cfa7d41e9bef8040374578ff8c5d572",
    "PHASE_16_EXP7_DESIGN.md":
        "098b7bb11c57012c3e8c56a790423a1741fc7e65654b34f0a87e1f4c59fb6be1",
    "FINAL_EXP7_CLOSURE.md":
        "928eb7b0d6c5eab7cd3c68a631d97518053774d784ab33905a682d8bcc0d36eb",
    "PHASE_18_EXP9_DESIGN.md":
        "cf361986e4f058259e21fefd548b335ad9fb91758696d57fdf6c762254bf288a",
}

# design SS 12.6, SS 20, SS 24 - asserted absent at build time so an edit to
# UPLOAD_FILES can never reintroduce them.
FORBIDDEN_IN_UPLOAD: List[str] = [
    "trainer_outputs/exp7_modality/ablation_multimodal/model.pt",
    "trainer_outputs/exp7_modality/ablation_multimodal/delta.pt",
    "trainer_outputs/exp7_modality/ablation_text_only/model.pt",
    "trainer_outputs/exp7_modality/ablation_text_only/delta.pt",
    "trainer_outputs/mentalbert_delta.pt",
    "trainer_outputs/mentalbert_privacy_subset.pt",
    "trainer_outputs/local_probe_base.pt",
]
FORBIDDEN_PREFIXES: List[str] = [
    "trainer_outputs/exp7_modality/",
    "trainer_outputs/exp8_dp_mechanism/",
    "trainer_outputs/exp9_compact_update/",
    "exp9_compact_update/",
    "exp8_dp_mechanism/",
    "dp_agent/",
    f"{EXP_DIR}/",          # results never ship in the UPLOAD bundle
]
FORBIDDEN_SUFFIXES: List[str] = [".pt", ".bin", ".safetensors", ".enc", ".ckpt"]

RUN_INSTRUCTIONS = """\
EXPERIMENT 10 - COLAB GPU RUN ORDER (PHASE_19_EXP10_DESIGN.md)

  0. Runtime > Change runtime type > GPU (T4 is sufficient). Confirm:
       import torch; assert torch.cuda.is_available()

  1. Unzip at the repo root so relative paths resolve.

  2. Pin transformers (mandatory - the frozen trainer imports AdamW from
     transformers, removed in 4.46+, and older versions change the BERT
     state_dict key count):
       !pip install -q transformers==4.44.0
     then Runtime > Restart session.

  3. Authenticate with Hugging Face. mental/mental-bert-base-uncased is GATED
     and returns HTTP 401 anonymously:
       from huggingface_hub import notebook_login; notebook_login()

  4. Run the experiment - 3 arms x 5 folds x 5 repeats = 75 fold-runs:
       python exp10_multimodal_conditioning/run_exp10.py

     It gates the frozen SHAs before and after, refuses to run on CPU, asserts
     participant_id/text/phq_score byte-identity before training, and hard-aborts
     if the fold-local negative control fails.

     Expected GPU wall time: ~27-32 min on a T4 (75 x 18.0 s measured in Exp 7,
     plus inference).

  5. A [G0 advisory] line prints at the end. It is ADVISORY ONLY - the
     authoritative G0 equivalence gate is aggregate_exp10.py on the Terminal.
     If it says OUTSIDE, still bring the results back; do NOT re-run, re-seed
     or adjust anything (design SS 29).

  6. Bring back: trainer_outputs/multimodal_exp/ in full.

DO NOT RUN ON COLAB: aggregate_exp10.py, verify_exp10.py.
Those belong to the Terminal half and gate on artifacts not in this bundle.
"""


def sha256(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for b in iter(lambda: fh.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


def repo_key(path: str) -> str:
    """Canonical repo-relative forward-slash key (platform-neutral)."""
    return path.replace("\\", "/")


# --------------------------------------------------------------------------
# Import-closure validation. STATIC ONLY: every shipped .py is parsed with ast
# and never imported or executed, so this cannot train, condition, or touch a
# dataset.
# --------------------------------------------------------------------------
def _resolve_local_module(mod: str, search_roots: List[str]) -> List[str]:
    """Repo-relative paths a dotted module resolves to, or [] if not project-local.

    Mirrors how the runners set sys.path: the repo root plus the importing file's
    own directory. Parent-package __init__.py files are included because Python
    executes them on import - the mechanism that hid the missing
    installer/security/__init__.py closure in the first Exp 9 bundle.
    """
    parts = mod.split(".")
    for root in search_roots:
        found: List[str] = []
        for i in range(1, len(parts)):
            init = os.path.join(root, *parts[:i], "__init__.py")
            if os.path.isfile(init):
                found.append(repo_key(os.path.relpath(init, REPO)))
        mod_py = os.path.join(root, *parts) + ".py"
        pkg_init = os.path.join(root, *parts, "__init__.py")
        if os.path.isfile(mod_py):
            found.append(repo_key(os.path.relpath(mod_py, REPO)))
        elif os.path.isfile(pkg_init):
            found.append(repo_key(os.path.relpath(pkg_init, REPO)))
        if found:
            return found
    return []


def _imported_modules(path: str) -> List[str]:
    """Dotted module names imported by one file, absolute and relative."""
    with open(path, "r", encoding="utf-8") as fh:
        tree = ast.parse(fh.read(), filename=path)
    pkg = repo_key(os.path.dirname(os.path.relpath(path, REPO))).replace("/", ".")
    mods: List[str] = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            mods.extend(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom):
            if node.level:
                base = pkg.split(".")
                base = base[:len(base) - (node.level - 1)] if node.level > 1 else base
                mods.append(".".join([b for b in base if b]
                                     + ([node.module] if node.module else [])))
            elif node.module:
                mods.append(node.module)
    return mods


def check_import_closure(files: List[str]) -> List[str]:
    """Project-local modules reachable from the bundle but NOT shipped."""
    shipped = {repo_key(p) for p in files}
    queue = [p for p in files if p.endswith(".py")]
    seen = set(queue)
    missing: List[str] = []
    while queue:
        cur = queue.pop()
        roots = [REPO, os.path.join(REPO, os.path.dirname(cur))]
        for mod in _imported_modules(cur):
            for dep in _resolve_local_module(mod, roots):
                if dep not in shipped and dep not in missing:
                    missing.append(dep)
                if dep not in seen and os.path.isfile(os.path.join(REPO, dep)):
                    seen.add(dep)
                    queue.append(dep)
    return sorted(missing)


def main() -> int:
    ap = argparse.ArgumentParser(description="Build an Experiment 10 bundle")
    ap.add_argument("--results", action="store_true",
                    help="package the Exp 10 RESULT artifacts instead of the "
                         "Colab upload bundle")
    ap.add_argument("--out", default=None)
    ap.add_argument("--skip-sha", action="store_true",
                    help="package anyway if a SHA has legitimately moved")
    a = ap.parse_args()
    os.chdir(REPO)

    results_mode = a.results
    files = RESULT_FILES if results_mode else UPLOAD_FILES
    out = os.path.abspath(a.out or (DEFAULT_RESULTS if results_mode
                                    else DEFAULT_UPLOAD))
    title = "RESULT BUNDLE" if results_mode else "UPLOAD BUNDLE"

    print("=" * 74)
    print(f"EXPERIMENT 10 {title}")
    print("=" * 74)

    missing = [p for p in files if not os.path.isfile(p)]
    if missing:
        print("[FATAL] missing required file(s):", file=sys.stderr)
        for m in missing:
            print(f"    {m}", file=sys.stderr)
        if results_mode:
            print("        (run run_exp10.py, aggregate_exp10.py and "
                  "verify_exp10.py first)", file=sys.stderr)
        return 2
    print(f"[OK] all {len(files)} required files present")

    bad = []
    for path, want in EXPECTED_SHA.items():
        if path in files:
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
        n = len([p for p in EXPECTED_SHA if p in files])
        print(f"[OK] all {n} SHA-pinned artifact(s) verified")

    if not results_mode:
        leaked = [p for p in files
                  if repo_key(p) in FORBIDDEN_IN_UPLOAD
                  or any(repo_key(p).startswith(x) for x in FORBIDDEN_PREFIXES)
                  or any(repo_key(p).endswith(s) for s in FORBIDDEN_SUFFIXES)]
        if leaked:
            print("[FATAL] forbidden path(s) present in the upload bundle:",
                  file=sys.stderr)
            for p in leaked:
                print(f"    {p}", file=sys.stderr)
            return 3
        print(f"[OK] {len(FORBIDDEN_IN_UPLOAD)} forbidden path(s), "
              f"{len(FORBIDDEN_PREFIXES)} prefix(es) and "
              f"{len(FORBIDDEN_SUFFIXES)} suffix(es) confirmed absent")

        missing_deps = check_import_closure(files)
        if missing_deps:
            print("[FATAL] project-local import(s) reachable from the bundle but "
                  "NOT shipped:", file=sys.stderr)
            for p in missing_deps:
                print(f"    {p}", file=sys.stderr)
            print("        (the bundle would fail at import on Colab)",
                  file=sys.stderr)
            return 4
        n_py = len([p for p in files if p.endswith(".py")])
        print(f"[OK] import closure complete: {n_py} shipped .py file(s), "
              "0 unshipped project-local dependencies")

    os.makedirs(os.path.dirname(out), exist_ok=True)
    manifest = "\n".join(f"{sha256(p)}  {repo_key(p)}" for p in files) + "\n"
    with zipfile.ZipFile(out, "w", zipfile.ZIP_DEFLATED) as z:
        for p in files:
            z.write(p, repo_key(p))         # repo-relative paths preserved
        if not results_mode:
            z.writestr("EXP10_RUN_INSTRUCTIONS.txt", RUN_INSTRUCTIONS)
        z.writestr("EXP10_SHA_MANIFEST.txt", manifest)

    print(f"[OK] wrote {out}  ({os.path.getsize(out) / (1024*1024):.2f} MB)")
    print(f"[OK] bundle sha256: {sha256(out)}")
    print("\n=== EXP10_SHA_MANIFEST.txt ===")
    print(manifest)
    if not results_mode:
        print(RUN_INSTRUCTIONS)
    return 0


if __name__ == "__main__":
    sys.exit(main())
