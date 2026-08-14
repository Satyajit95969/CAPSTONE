#!/usr/bin/env python3
"""
make_exp9_upload_bundle.py - package Experiment 9 for Colab, or its results.

Authoritative specification: PHASE_18_EXP9_DESIGN.md (FROZEN), SS 27, SS 30, SS 31.

TWO MODES, because design SS 27 splits Exp 9 across two environments:

  (default)   UPLOAD bundle - the code and frozen inputs the GPU half needs.
              Deliberately EXCLUDES trainer_outputs/mentalbert_delta.pt (438 MB):
              that object belongs to the Terminal half (design SS 21.1), and the
              whole point of the split is that it never crosses environments.

  --results   RESULT bundle - the Exp 9 output artifacts plus the frozen
              specification that governs them, for archival and review. Refuses
              to build if a required result artifact is missing, so a partial or
              aborted run can never be packaged as a complete result.

Every file is SHA-verified before zipping. The same class of missing-file failure
(a runner looking for trainer_outputs/baseline_cv/fold_manifest.json that was
never uploaded) has cost this project a full Colab session before.

Usage:
    python make_exp9_upload_bundle.py
    python make_exp9_upload_bundle.py --results
    python make_exp9_upload_bundle.py --out C:/Users/DELL/Downloads/exp9_upload.zip
"""

from __future__ import annotations

import argparse
import hashlib
import os
import sys
import zipfile
from typing import Dict, List

REPO = os.path.dirname(os.path.abspath(__file__))
DEFAULT_UPLOAD = os.path.join(os.path.expanduser("~"), "Downloads", "exp9_upload.zip")
DEFAULT_RESULTS = os.path.join(os.path.expanduser("~"), "Downloads", "exp9_results.zip")
EXP_DIR = os.path.join("trainer_outputs", "exp9_compact_update")

# --------------------------------------------------------------------------
# Colab needs exactly this: the task runner, the shared frozen constants, the
# frozen dp_agent that exp9_common imports, the frozen trainer, the dataset the
# fold manifest names, the manifest, and the K0 gate's reference.
# --------------------------------------------------------------------------
UPLOAD_FILES: List[str] = [
    "PHASE_18_EXP9_DESIGN.md",
    "trainer_mentalbert_daic.py",
    "dp_agent/__init__.py",
    "dp_agent/dp_agent.py",
    # dp_agent/dp_agent.py imports these THREE at MODULE level (lines 7, 8, 10)
    # and calls integrity_guard() at line 11, so importing exp9_common - which
    # takes _rdp_to_dp from it - pulls the whole chain in. Without them
    # run_exp9_task.py dies at import, before the GPU check and before any fold.
    #
    # centralized_secure_store.py MUST be the REPOSITORY-ROOT copy (3,959 B).
    # The 7,636 B variants under installer/runtime/core/ and
    # server/aggregator_agent/core/ are a DIFFERENT SecureStore; shipping one of
    # those would substitute a different AES-GCM envelope from the one the
    # Terminal half already measured for G2. EXPECTED_SHA pins the right one.
    "centralised_receipts.py",
    "centralized_secure_store.py",
    # installer/security/__init__.py is NOT empty - it does
    #   from .anti_debug import anti_debug
    #   from .tpm_attestation import tpm_attestation
    # so importing installer.security.integrity executes those two modules too.
    # None of the four has a module-level call; they only define functions, and
    # nothing on the Exp 9 path invokes anti_debug() or tpm_attestation().
    # self_destruct is imported inside integrity.py:153/:190 and in anti_debug.
    "installer/__init__.py",
    "installer/security/__init__.py",
    "installer/security/integrity.py",
    "installer/security/anti_debug.py",
    "installer/security/tpm_attestation.py",
    "installer/security/self_destruct.py",
    "exp9_compact_update/exp9_common.py",
    "exp9_compact_update/run_exp9_task.py",
    # design SS 20: the manifest is authoritative and names daic_records.parquet.
    # The multimodal parquet is deliberately NOT shipped - substituting it would
    # change a second factor and break the SS 22 K0 gate.
    "daic_records.parquet",
    "trainer_outputs/baseline_cv/fold_manifest.json",
    "trainer_outputs/baseline_cv/baseline_cv_summary.json",
]

RESULT_FILES: List[str] = [
    os.path.join(EXP_DIR, "exp9_mask_manifest.json"),
    os.path.join(EXP_DIR, "exp9_dp_metrics.csv"),
    os.path.join(EXP_DIR, "exp9_payload.json"),
    os.path.join(EXP_DIR, "exp9_receipt.json"),
    os.path.join(EXP_DIR, "exp9_task_metrics.csv"),
    os.path.join(EXP_DIR, "exp9_task_receipt.json"),
    os.path.join(EXP_DIR, "exp9_summary.json"),
    os.path.join(EXP_DIR, "exp9_acceptance.csv"),
    os.path.join(EXP_DIR, "exp9_verification.json"),
    "PHASE_18_EXP9_DESIGN.md",
]

EXPECTED_SHA: Dict[str, str] = {
    "PHASE_18_EXP9_DESIGN.md":
        "cf361986e4f058259e21fefd548b335ad9fb91758696d57fdf6c762254bf288a",
    "trainer_mentalbert_daic.py":
        "65b1902e2ba8dd996c6960db1a6595d84fd48500f4d8707cb79688093d7a230b",
    "dp_agent/dp_agent.py":
        "758642fff57695cb970af88789c3b6a17c77f2b01d16303ff96230fce5798732",
    "daic_records.parquet":
        "9a241851760727779fcaa4d50041b8bdc4406fe6b66ddbbd7105f4ce4fc55f00",
    "trainer_outputs/baseline_cv/fold_manifest.json":
        "b9a7a91f1bdd43c78d3cd1ee0c36e4ce3fb8be4b0971dcff3ff62ee5af8fca2f",
    "trainer_outputs/baseline_cv/baseline_cv_summary.json":
        "f065f5e1ff7f8e5ed4d122a8031c9cc12425802e756697d5512a915674871aac",
    # Import-closure dependencies of the frozen dp_agent. Pinned so the builder
    # cannot ship a divergent copy - in particular the root SecureStore, whose
    # 7,636 B namesakes elsewhere in the tree are a different implementation.
    "centralised_receipts.py":
        "987a799b2ad1d3185d86bab0bdb5b18735e5dfc05a318a09b9f51cfe1803b510",
    "centralized_secure_store.py":
        "be4fed1d3037fbe20366a05acacecacee793aa6b343db1d421bafaeef0036df3",
    "installer/__init__.py":
        "e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855",
    "installer/security/__init__.py":
        "b92e06d9dac493e2a3ab1377f523f97a8d6a996cbcb0d7c5b548c1610f99accd",
    "installer/security/integrity.py":
        "5d2aa0ee5a8e32953a8722a66f55b49904be778909ecbee76010d244b65e1940",
    "installer/security/anti_debug.py":
        "072d556142e025b74ce0ce3926cf65884f1578583c5a2f0c36792b8baf1f557b",
    "installer/security/tpm_attestation.py":
        "723edeb93970af32f5ead73e89896a986c6574449f3f9a0b3ac042728cfe9baa",
    "installer/security/self_destruct.py":
        "5cc391f7bf68b70f808e6094d772e9e970758a6c24b0eb009cf86dd0ff8a2009",
}

# Paths that must NEVER enter the upload bundle, asserted at build time so an
# exclusion can never be lost by an edit to UPLOAD_FILES.
FORBIDDEN_IN_UPLOAD: List[str] = [
    "trainer_outputs/mentalbert_delta.pt",          # Terminal-half object, 438 MB
    "dataset_build/daic_records_multimodal.parquet",  # design SS 20: no substitution
    "trainer_outputs/local_probe_base.pt",           # Exp 8 target
    "exp9_compact_update/run_exp9.py",               # Terminal-only
    "exp9_compact_update/aggregate_exp9.py",         # Terminal-only
    "exp9_compact_update/verify_exp9.py",            # Terminal-only
    "installer/runtime/core/centralized_secure_store.py",   # divergent copy
    "server/aggregator_agent/core/centralized_secure_store.py",
]
FORBIDDEN_PREFIXES: List[str] = [
    "trainer_outputs/exp9_compact_update/",   # results, keys/, receipts/, secure_store/
]


# --------------------------------------------------------------------------
# Import-closure validation. STATIC ONLY: every shipped .py is parsed with ast
# and never imported or executed, so this cannot run experiment code, train,
# draw noise or encrypt anything.
# --------------------------------------------------------------------------
def _resolve_local_module(mod: str, search_roots: List[str]) -> List[str]:
    """Repo-relative paths a dotted module resolves to, or [] if not project-local.

    Mirrors how the runners set sys.path: the repo root plus the importing
    file's own directory. Parent-package __init__.py files are included because
    Python executes them on import - which is exactly how the non-empty
    installer/security/__init__.py was missed the first time.
    """
    parts = mod.split(".")
    for root in search_roots:
        found: List[str] = []
        for i in range(1, len(parts)):
            init = os.path.join(root, *parts[:i], "__init__.py")
            if os.path.isfile(init):
                found.append(os.path.relpath(init, REPO).replace("\\", "/"))
        mod_py = os.path.join(root, *parts) + ".py"
        pkg_init = os.path.join(root, *parts, "__init__.py")
        if os.path.isfile(mod_py):
            found.append(os.path.relpath(mod_py, REPO).replace("\\", "/"))
        elif os.path.isfile(pkg_init):
            found.append(os.path.relpath(pkg_init, REPO).replace("\\", "/"))
        if found:
            return found
    return []


def _imported_modules(path: str) -> List[str]:
    """Dotted module names imported by one file, absolute and relative."""
    import ast

    with open(path, "r", encoding="utf-8") as fh:
        tree = ast.parse(fh.read(), filename=path)
    pkg = os.path.dirname(os.path.relpath(path, REPO)).replace("\\", "/").replace("/", ".")
    mods: List[str] = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            mods.extend(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom):
            if node.level:                      # relative: from .x import y
                base = pkg.split(".")
                base = base[:len(base) - (node.level - 1)] if node.level > 1 else base
                mods.append(".".join([b for b in base if b] + ([node.module] if node.module else [])))
            elif node.module:
                mods.append(node.module)
    return mods


def check_import_closure(files: List[str]) -> List[str]:
    """Return project-local modules reachable from the bundle but NOT shipped.

    Breadth-first over the shipped .py files. A module that does not resolve to
    a file inside this repository is third-party or stdlib and is ignored.
    """
    shipped = {p.replace("\\", "/") for p in files}
    queue = [p for p in files if p.endswith(".py")]
    seen: set = set(queue)
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

RUN_INSTRUCTIONS = """\
EXPERIMENT 9 - COLAB GPU RUN ORDER (design SS 27, right-hand column)

  0. Runtime > Change runtime type > GPU. Confirm:
       import torch; assert torch.cuda.is_available()

  1. Unzip at the repo root so relative paths resolve.

  2. Run the task half - 25 fold trainings, then 4 arms x 25 ROC-AUC
     evaluations with the mask applied in-flight:

       python exp9_compact_update/run_exp9_task.py

     It gates the frozen SHAs before and after, refuses to run on CPU,
     and refuses any dataset other than the one the frozen fold manifest
     names. Full deltas are NEVER persisted (design SS 21.2).

     Expected GPU wall time: ~15-20 min (25 trainings at Exp 7's measured
     18 s/fold-run is about 8 min, plus 100 inference passes).

  3. A [K0 advisory] line prints at the end. It is ADVISORY ONLY - the
     authoritative K0 equivalence gate is aggregate_exp9.py on the Terminal.
     If the advisory says OUTSIDE, still bring the results back; do not
     re-run, re-seed or adjust anything (design SS 32).

  4. Bring back: trainer_outputs/exp9_compact_update/ in full.

DO NOT RUN ON COLAB: run_exp9.py, aggregate_exp9.py, verify_exp9.py.
Those are the Terminal half - they need trainer_outputs/mentalbert_delta.pt,
which is deliberately not in this bundle (design SS 21.1, SS 27).
"""


def sha256(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for b in iter(lambda: fh.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


def main() -> int:
    ap = argparse.ArgumentParser(description="Build an Experiment 9 bundle")
    ap.add_argument("--results", action="store_true",
                    help="package the Exp 9 RESULT artifacts instead of the "
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
    print(f"EXPERIMENT 9 {title}")
    print("=" * 74)

    missing = [p for p in files if not os.path.isfile(p)]
    if missing:
        print("[FATAL] missing required file(s):", file=sys.stderr)
        for m in missing:
            print(f"    {m}", file=sys.stderr)
        if results_mode:
            print("        (run run_exp9.py, run_exp9_task.py, aggregate_exp9.py "
                  "and verify_exp9.py first)", file=sys.stderr)
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
        # Exclusions are asserted, not assumed. An edit to UPLOAD_FILES cannot
        # silently reintroduce the frozen delta, the multimodal parquet, a
        # Terminal-only script, a divergent SecureStore, or a result artifact.
        leaked = [p for p in files if p.replace("\\", "/") in FORBIDDEN_IN_UPLOAD
                  or any(p.replace("\\", "/").startswith(x) for x in FORBIDDEN_PREFIXES)]
        if leaked:
            print("[FATAL] forbidden path(s) present in the upload bundle:",
                  file=sys.stderr)
            for p in leaked:
                print(f"    {p}", file=sys.stderr)
            return 3
        print(f"[OK] {len(FORBIDDEN_IN_UPLOAD)} forbidden path(s) + "
              f"{len(FORBIDDEN_PREFIXES)} prefix(es) confirmed absent")

        # STATIC import-closure check - ast only, nothing is imported or run.
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
    manifest = "\n".join(f"{sha256(p)}  {p}" for p in files) + "\n"
    with zipfile.ZipFile(out, "w", zipfile.ZIP_DEFLATED) as z:
        for p in files:
            z.write(p, p)
        if not results_mode:
            z.writestr("EXP9_RUN_INSTRUCTIONS.txt", RUN_INSTRUCTIONS)
        z.writestr("EXP9_SHA_MANIFEST.txt", manifest)

    size_mb = os.path.getsize(out) / (1024 * 1024)
    print(f"[OK] wrote {out}  ({size_mb:.2f} MB)")
    print(f"[OK] bundle sha256: {sha256(out)}")
    if results_mode:
        print("\n=== EXP9_SHA_MANIFEST.txt ===")
        print(manifest)
    else:
        print("\n" + RUN_INSTRUCTIONS)
    return 0


if __name__ == "__main__":
    sys.exit(main())
