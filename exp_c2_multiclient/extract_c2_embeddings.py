#!/usr/bin/env python3
"""
extract_c2_embeddings.py - Phase 20 / C-2 representation extraction (Module A).

Authoritative specification: PHASE_20_C2_DESIGN.md (FROZEN), SS 6, SS 6.2.

Produces the ONE C-2 representation artifact: 188 x 768 CLS embeddings for every
DAIC participant, at the PINNED MentalBERT revision.

WHY THIS EXISTS AS A NEW MODULE (design SS 6.2)
-----------------------------------------------
No 188-participant 768-d artifact exists in the repository. The only DAIC
embedding pipeline (format_daic_to_lda.py) is capped at MAX_PARTICIPANTS = 20 and
uses max_length = 256, which is incompatible with the frozen family's 128. That
file and its artifact are READ-NEVER, WRITE-NEVER here (design SS 6.1).

Equivalence with the frozen family is by construction: MultiModalModel.__init__
builds `self.bert = AutoModel.from_pretrained(bert_name)` - a pretrained-only
encoder - and forward() takes `last_hidden_state[:, 0, :]`
(trainer_mentalbert_daic.py:256). This module loads the SAME encoder at the
pinned revision and takes the SAME slice. Padding does not affect CLS (index 0,
pads masked), so padding="max_length" matches the frozen dataset exactly.

THE REVISION IS LOADED FROM THE LOCAL SNAPSHOT DIRECTORY, not by name+revision,
so the exact pinned weights are used with no possibility of a silent refetch.

NO TASK METRIC is computed here or anywhere in C-2 (design SS 14).

Usage:
    python exp_c2_multiclient/extract_c2_embeddings.py
    python exp_c2_multiclient/extract_c2_embeddings.py --checkpoint-only
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from typing import Any, Dict, List

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import c2_common as C  # noqa: E402


def integrity_checkpoint(npz_path: str, parquet_path: str) -> Dict[str, Any]:
    """The embedding integrity checkpoint. FAIL CLOSED - aborts on any failure.

    Runs BEFORE any client training. Every check is a hard abort: an embedding
    artifact that fails any of these must never reach the training loop.
    """
    import pandas as pd

    print("=" * 78)
    print("C-2 EMBEDDING INTEGRITY CHECKPOINT - FAIL CLOSED")
    print("=" * 78)

    if not os.path.isfile(npz_path):
        C.abort(f"embedding artifact not found: {npz_path}")
    z = np.load(npz_path, allow_pickle=False)
    ids = z["participant_id"]
    emb = z["embedding"]
    phq = z["phq_score"]

    df = pd.read_parquet(parquet_path)
    src_ids = [int(x) for x in df["participant_id"]]
    src_phq = {int(r["participant_id"]): float(r["phq_score"])
               for _, r in df.iterrows()}

    checks: List[Dict[str, Any]] = []

    def ck(name: str, cond: bool, detail: str = "") -> None:
        checks.append({"check": name, "status": "PASS" if cond else "FAIL",
                       "detail": detail})
        if not cond:
            print(f"  [FAIL] {name}\n         {detail}")

    ck("exactly 188 participants", len(ids) == C.N_PARTICIPANTS,
       f"n={len(ids)}")
    ck("participant IDs unique", len(set(int(i) for i in ids)) == len(ids),
       f"unique={len(set(int(i) for i in ids))} of {len(ids)}")
    ck("exactly one embedding per participant", emb.shape[0] == len(ids),
       f"rows={emb.shape[0]} ids={len(ids)}")
    ck("embedding dimension = 768 for every participant",
       emb.ndim == 2 and emb.shape[1] == C.EMBED_DIM,
       f"shape={tuple(emb.shape)}")
    ck("no NaN / Inf in embeddings", bool(np.isfinite(emb).all()),
       f"finite={bool(np.isfinite(emb).all())} "
       f"n_nonfinite={int((~np.isfinite(emb)).sum())}")
    ck("PHQ values present and finite",
       len(phq) == len(ids) and bool(np.isfinite(phq).all()),
       f"n={len(phq)} finite={bool(np.isfinite(phq).all())}")
    ck("participant IDs match daic_records.parquet exactly",
       [int(i) for i in ids] == src_ids,
       f"order_identical={[int(i) for i in ids] == src_ids}")
    ck("PHQ values unchanged from source",
       all(float(phq[i]) == src_phq[int(ids[i])] for i in range(len(ids))),
       "elementwise equality against the frozen parquet")

    n_fail = sum(1 for c in checks if c["status"] == "FAIL")
    for c in checks:
        if c["status"] == "PASS":
            print(f"  [ok  ] {c['check']}")
            if c["detail"]:
                print(f"         {c['detail']}")
    print("-" * 78)
    print(f"embedding checkpoint: {len(checks) - n_fail} PASS, {n_fail} FAIL")
    if n_fail:
        C.abort("embedding integrity checkpoint FAILED - no client may be trained")
    print("EMBEDDING CHECKPOINT: PASS")
    return {"checks": checks, "n_pass": len(checks) - n_fail, "n_fail": n_fail,
            "overall": "PASS"}


def main() -> int:
    ap = argparse.ArgumentParser(
        description="Phase 20 / C-2 representation extraction (Module A)")
    ap.add_argument("--out-dir", default=os.path.join(C.REPO, C.OUT_DIRNAME))
    ap.add_argument("--parquet", default=C.DATA_PARQUET)
    ap.add_argument("--checkpoint-only", action="store_true",
                    help="run only the integrity checkpoint on an existing artifact")
    a = ap.parse_args()
    os.chdir(C.REPO)

    npz_path = os.path.join(a.out_dir, "c2_embeddings.npz")
    if a.checkpoint_only:
        integrity_checkpoint(npz_path, a.parquet)
        return 0

    t0 = time.time()
    print("=" * 78)
    print("C-2 REPRESENTATION EXTRACTION (design SS 6)")
    print("no task metric is computed anywhere in C-2 (design SS 14)")
    print("=" * 78)

    # ---- frozen-input gate, in the execution path -------------------------
    bad = [p for p, w in C.FROZEN_SHA.items()
           if w is not None and C.sha256_file(p) != w]
    if bad:
        C.abort(f"frozen input SHA mismatch before extraction: {bad}")
    print(f"[OK] frozen inputs verified ({len([w for w in C.FROZEN_SHA.values() if w])})")

    # ---- the PINNED revision, loaded from the local snapshot directory ----
    snap = C.hf_snapshot_dir()
    if not os.path.isdir(snap):
        C.abort(f"pinned MentalBERT snapshot not found: {snap}")
    art_bad = []
    for fname, want in C.BERT_ARTIFACT_SHA.items():
        got = C.sha256_file(os.path.join(snap, fname))
        if got != want:
            art_bad.append(f"{fname}: got {got} want {want}")
    if art_bad:
        C.abort("MentalBERT artifact SHA mismatch: " + "; ".join(art_bad))
    print(f"[OK] MentalBERT revision {C.BERT_REVISION} verified "
          f"({len(C.BERT_ARTIFACT_SHA)} artifacts)")

    import torch
    from transformers import AutoModel, AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(snap, local_files_only=True)
    model = AutoModel.from_pretrained(snap, local_files_only=True)
    model.eval()
    if int(model.config.hidden_size) != C.EMBED_DIM:
        C.abort(f"hidden_size {model.config.hidden_size} != {C.EMBED_DIM}")
    print(f"[OK] loaded from snapshot dir (local_files_only=True); "
          f"hidden_size={model.config.hidden_size}")

    import pandas as pd
    df = pd.read_parquet(a.parquet)
    if len(df) != C.N_PARTICIPANTS:
        C.abort(f"parquet holds {len(df)} rows, expected {C.N_PARTICIPANTS}")

    ids: List[int] = []
    phqs: List[float] = []
    embs: List[np.ndarray] = []
    print(f"[INFO] extracting {len(df)} x {C.EMBED_DIM} "
          f"(max_length={C.MAX_LENGTH}, truncation={C.TRUNCATION}, "
          f"padding={C.PADDING}, pooling={C.POOLING})")

    with torch.no_grad():
        for n, (_, r) in enumerate(df.iterrows(), 1):
            enc = tokenizer(str(r["text"]), return_tensors="pt",
                            truncation=C.TRUNCATION, padding=C.PADDING,
                            max_length=C.MAX_LENGTH)
            out = model(**enc)
            cls = out.last_hidden_state[:, 0, :].squeeze(0).cpu().numpy()
            if cls.shape != (C.EMBED_DIM,):
                C.abort(f"participant {r['participant_id']}: CLS shape "
                        f"{cls.shape} != ({C.EMBED_DIM},)")
            if not np.isfinite(cls).all():
                C.abort(f"participant {r['participant_id']}: non-finite CLS")
            ids.append(int(r["participant_id"]))
            phqs.append(float(r["phq_score"]))
            embs.append(cls.astype(np.float64))
            if n % 25 == 0 or n == len(df):
                print(f"  [{n:>3}/{len(df)}] {time.time() - t0:6.1f}s")

    E = np.stack(embs)
    os.makedirs(a.out_dir, exist_ok=True)
    np.savez(npz_path,
             participant_id=np.array(ids, dtype=np.int64),
             embedding=E,
             phq_score=np.array(phqs, dtype=np.float64))
    elapsed = round(time.time() - t0, 2)
    print(f"[OK] wrote {npz_path}  shape={tuple(E.shape)}  ({elapsed:.1f}s)")

    # ---- integrity checkpoint BEFORE any training can begin ---------------
    cp = integrity_checkpoint(npz_path, a.parquet)

    receipt = {
        "schema": "c2_model_receipt/1",
        "experiment": C.EXPERIMENT,
        "design": {"path": C.DESIGN_PATH, "sha256": C.design_sha()},
        "generated_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "model": {"name": C.BERT_MODEL, "revision": C.BERT_REVISION,
                  "loaded_from": "local snapshot directory (local_files_only=True)",
                  "snapshot_dir": snap,
                  "artifact_sha256": {f: C.sha256_file(os.path.join(snap, f))
                                      for f in C.BERT_ARTIFACT_SHA}},
        "representation": {"max_length": C.MAX_LENGTH, "truncation": C.TRUNCATION,
                           "padding": C.PADDING, "pooling": C.POOLING,
                           "embed_dim": C.EMBED_DIM,
                           "dtype": "float64",
                           "deterministic": "model.eval() + torch.no_grad(); "
                                            "no dropout, no RNG consumed"},
        "prohibited_sources_not_used": list(C.PROHIBITED_REPRESENTATION_SOURCES),
        "source_parquet": {"path": C.DATA_PARQUET,
                           "sha256": C.sha256_file(C.DATA_PARQUET)},
        "artifact": {"path": C.repo_key(os.path.relpath(npz_path, C.REPO)),
                     "sha256": C.sha256_file(npz_path),
                     "n_participants": int(E.shape[0]),
                     "embed_dim": int(E.shape[1]),
                     "bytes": os.path.getsize(npz_path)},
        "integrity_checkpoint": cp,
        "environment": {"python": sys.version.split()[0],
                        "torch": torch.__version__,
                        "platform": sys.platform},
        "runtime_seconds": elapsed,
        "task_metric_computed": False,
    }
    rpath = os.path.join(a.out_dir, "c2_model_receipt.json")
    with open(rpath, "w", encoding="utf-8") as fh:
        json.dump(receipt, fh, indent=2, sort_keys=True, default=float)
    print(f"[OK] receipt -> {rpath}")
    print(f"[OK] embedding artifact sha256: {receipt['artifact']['sha256']}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
