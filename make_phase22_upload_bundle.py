#!/usr/bin/env python3
"""
make_phase22_upload_bundle.py - build the Phase 22 Colab GPU bundle.

FAIL CLOSED: every declared file must exist, or the bundle is not written.

LESSON FROM EXPERIMENT 9 (encoded here deliberately): an AST import-closure check
sees Python imports but NOT runtime data dependencies. Exp 9's Colab run aborted
because four frozen artifacts were omitted. This manifest therefore lists BOTH
code and data explicitly, including the transitive chain that `dp_agent` needs:

    dp_agent/dp_agent.py
      -> centralised_receipts.py
      -> centralized_secure_store.py
      -> installer/security/integrity.py  (+ installer package __init__ files)

The MentalBERT snapshot is included by default (--include-model) because
`mental/mental-bert-base-uncased` is a GATED Hugging Face repository: shipping the
pinned revision avoids requiring an HF token inside Colab and guarantees the exact
weights. Use --no-model to build a small bundle and authenticate in Colab instead.

Usage:
    python make_phase22_upload_bundle.py
    python make_phase22_upload_bundle.py --no-model
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import zipfile

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
REPO = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(REPO, "phase22_end_to_end"))

import phase22_common as C  # noqa: E402

# ---- code -----------------------------------------------------------------
CODE = [
    "phase22_end_to_end/phase22_common.py",
    "phase22_end_to_end/validate_phase22.py",
    "phase22_end_to_end/run_phase22.py",
    "phase22_end_to_end/aggregate_phase22.py",
    "phase22_end_to_end/verify_phase22.py",
    "trainer_mentalbert_daic.py",
    "dp_agent/dp_agent.py",
    "dp_agent/__init__.py",
    "centralised_receipts.py",
    "centralized_secure_store.py",
    "installer/__init__.py",
    # installer/security/__init__.py eagerly imports BOTH anti_debug and
    # tpm_attestation, so importing installer.security.integrity pulls all three.
    # self_destruct is imported lazily by integrity.py:153 (inside integrity_guard,
    # which dp_agent.py:12 CALLS at import) and by anti_debug.py at four Linux-path
    # sites. All are pre-existing, git-tracked and UNMODIFIED - packaging only.
    "installer/security/__init__.py",
    "installer/security/anti_debug.py",
    "installer/security/tpm_attestation.py",
    "installer/security/self_destruct.py",
    "installer/security/integrity.py",
    "server/aggregator_agent/aggregator.py",
    # aggregator.py:319 lazily imports this inside _decrypt_from_store. Phase 22 does
    # not call that path, but it is in the AST transitive closure and costs ~8 KB.
    "server/aggregator_agent/core/centralized_secure_store.py",
    "server/aggregator_agent/core/centralised_receipts.py",
]

# NOTE on external packages: aggregator.py also imports pymongo / bson / gridfs, but
# ONLY inside its GridFS decryption methods, which Phase 22 never calls. They are
# therefore NOT required in Colab and are deliberately not listed as dependencies.

# ---- data / frozen artifacts (the Exp 9 omission class) --------------------
DATA = [
    "PHASE_22_DESIGN.md",
    "dataset_build/daic_records_multimodal.parquet",
    "daic_records.parquet",
    "trainer_outputs/baseline_cv/fold_manifest.json",
    "trainer_outputs/baseline_cv/baseline_cv_summary.json",
]


def sha256(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def main() -> int:
    ap = argparse.ArgumentParser(description="Phase 22 Colab bundle builder")
    ap.add_argument("--out", default="phase22_upload.zip")
    ap.add_argument("--no-model", dest="include_model", action="store_false",
                    help="exclude the MentalBERT snapshot (requires HF auth in Colab)")
    ap.set_defaults(include_model=True)
    a = ap.parse_args()
    os.chdir(REPO)

    files = list(CODE) + list(DATA)

    model_files = []
    if a.include_model:
        snap = C.hf_snapshot_dir()
        if not os.path.isdir(snap):
            print(f"[ABORT] pinned MentalBERT snapshot not found: {snap}",
                  file=sys.stderr)
            return 1
        bad = [f for f, w in C.BERT_ARTIFACT_SHA.items()
               if sha256(os.path.join(snap, f)) != w]
        if bad:
            print(f"[ABORT] MentalBERT artifact SHA mismatch: {bad}", file=sys.stderr)
            return 1
        model_files = [(os.path.join(snap, f), f"mentalbert_snapshot/{f}")
                       for f in C.BERT_ARTIFACT_SHA]

    # ---- fail closed on any missing declared file -------------------------
    missing = [f for f in files if not os.path.isfile(f)]
    if missing:
        print("[ABORT] declared files missing - bundle NOT written:", file=sys.stderr)
        for f in missing:
            print(f"          {f}", file=sys.stderr)
        return 1

    # ---- frozen SHA gate ---------------------------------------------------
    drift = [p for p, w in C.FROZEN_SHA.items()
             if w is not None and sha256(p) != w]
    if drift:
        print(f"[ABORT] frozen input SHA drift: {drift}", file=sys.stderr)
        return 1

    manifest = {
        "bundle": "phase22_upload/1",
        "experiment": C.EXPERIMENT,
        "design": {"path": C.DESIGN_PATH, "sha256": C.design_sha(),
                   "frozen_expected": C.DESIGN_SHA},
        "includes_mentalbert_snapshot": a.include_model,
        "mentalbert_revision": C.BERT_REVISION,
        "code": {f: sha256(f) for f in CODE},
        "data": {f: sha256(f) for f in DATA},
        "model": {arc: sha256(src) for src, arc in model_files},
        "scope_caveat": C.SCOPE_CAVEAT,
    }

    with zipfile.ZipFile(a.out, "w", zipfile.ZIP_DEFLATED) as z:
        for f in files:
            z.write(f, f)
        for src, arc in model_files:
            z.write(src, arc)
        z.writestr("phase22_manifest.json",
                   json.dumps(manifest, indent=2, sort_keys=True))

    size = os.path.getsize(a.out)
    print("=" * 74)
    print(f"[OK] {a.out}  ({size/1e6:.1f} MB)")
    print(f"     code   : {len(CODE)} files")
    print(f"     data   : {len(DATA)} files")
    print(f"     model  : {len(model_files)} files "
          f"({'included' if a.include_model else 'EXCLUDED - HF auth needed'})")
    print(f"     bundle sha256: {sha256(a.out)}")
    print(f"     design sha256: {C.design_sha()}")
    print("=" * 74)
    return 0


if __name__ == "__main__":
    sys.exit(main())
