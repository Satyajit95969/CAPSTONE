#!/usr/bin/env python3
"""
validate_phase22.py - Phase 22 pre-execution validation. FAIL CLOSED.

Runs BEFORE any training. Verifies every frozen input, constant and structural
assumption in PHASE_22_DESIGN.md actually holds. Read-only apart from --report.

Exit 0 only if EVERY check passes.

CUDA is a HARD requirement (design SS 16 / Colab execution): the frozen recipe
must run on GPU. There is no silent CPU fallback. Use --allow-cpu ONLY to audit
the non-GPU checks on a CPU box; it still exits non-zero.
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

import phase22_common as C  # noqa: E402

CHECKS: List[Dict[str, Any]] = []


def rec(name: str, ok: bool, detail: str = "") -> None:
    CHECKS.append({"check": name, "status": "PASS" if ok else "FAIL",
                   "detail": detail})
    print(f"  [{'ok  ' if ok else 'FAIL'}] {name}")
    if detail:
        print(f"         {detail}")


def main() -> int:
    ap = argparse.ArgumentParser(description="Phase 22 pre-execution validation")
    ap.add_argument("--report", default=None)
    ap.add_argument("--allow-cpu", action="store_true",
                    help="audit non-GPU checks on a CPU box; still exits non-zero")
    ap.add_argument("--skip-model", action="store_true",
                    help="skip the MentalBERT instantiation check (slow load)")
    a = ap.parse_args()
    os.chdir(C.REPO)
    t0 = time.time()

    print("=" * 78)
    print("PHASE 22 PRE-EXECUTION VALIDATION - FAIL CLOSED")
    print("ENGINEERING DEMONSTRATION - not Row 10 / C-1 (design SS 1, SS 23)")
    print("=" * 78)

    # -- 1 design frozen ----------------------------------------------------
    ds = C.design_sha()
    rec("1. Phase 22 design SHA matches the frozen value",
        ds == C.DESIGN_SHA, f"{ds}")

    # -- 2 frozen inputs ----------------------------------------------------
    bad = [f"{p}: {C.sha256_file(p)} != {w}" for p, w in C.FROZEN_SHA.items()
           if w is not None and C.sha256_file(p) != w]
    rec("2. every frozen input SHA matches", not bad,
        f"{len([w for w in C.FROZEN_SHA.values() if w])} pinned artifacts"
        if not bad else "; ".join(bad))

    # -- 3 dataset integrity ------------------------------------------------
    import pandas as pd
    df = pd.read_parquet(C.DATA_PARQUET)
    ids = [int(x) for x in df["participant_id"]]
    rec("3. dataset holds 188 unique participants",
        len(df) == C.N_PARTICIPANTS and len(set(ids)) == C.N_PARTICIPANTS,
        f"rows={len(df)} unique={len(set(ids))}")

    # -- 4 modality dimensions ----------------------------------------------
    A = np.array([json.loads(f)["audio"]["wav2vec2"] for f in df["features"]],
                 dtype=np.float64)
    V = np.array([json.loads(f)["video"]["densenet"] for f in df["features"]],
                 dtype=np.float64)
    rec("4. modality dimensions are 768 / 154 / 84",
        A.shape == (C.N_PARTICIPANTS, C.AUDIO_DIM)
        and V.shape == (C.N_PARTICIPANTS, C.VISION_DIM),
        f"text={C.TEXT_DIM} audio={A.shape} vision={V.shape}")

    # -- 5 no missing modalities / values -----------------------------------
    texts = df["text"].astype(str)
    rec("5. no missing modalities or values",
        bool(np.isfinite(A).all()) and bool(np.isfinite(V).all())
        and int(df.isna().sum().sum()) == 0
        and int((texts.str.strip() == "").sum()) == 0
        and int((np.abs(A).sum(1) == 0).sum()) == 0
        and int((np.abs(V).sum(1) == 0).sum()) == 0,
        "all finite; 0 nulls; 0 empty texts; 0 all-zero audio/vision rows")

    # -- 6 participant alignment with the frozen base parquet ---------------
    base = pd.read_parquet(C.BASE_PARQUET)
    bids = [int(x) for x in base["participant_id"]]
    bphq = {int(r["participant_id"]): float(r["phq_score"])
            for _, r in base.iterrows()}
    phq = {int(r["participant_id"]): float(r["phq_score"]) for _, r in df.iterrows()}
    rec("6. participant IDs and PHQ align exactly with frozen daic_records.parquet",
        ids == bids and all(abs(bphq[i] - phq[i]) < 1e-12 for i in ids),
        "188 IDs identical in order; PHQ elementwise equal")

    # -- 7 label prevalence --------------------------------------------------
    pos = sum(1 for v in phq.values() if C.binarize(v))
    rec("7. PHQ>10 binarisation yields 45 pos / 143 neg",
        pos == C.N_POSITIVE and (len(phq) - pos) == C.N_NEGATIVE,
        f"pos={pos} neg={len(phq)-pos}")

    # -- 8 fold manifest integrity ------------------------------------------
    with open(C.FOLD_MANIFEST, "r", encoding="utf-8") as fh:
        folds = C.load_folds(json.load(fh))
    sizes = [(len(f["train_ids"]), len(f["test_ids"])) for f in folds]
    rec("8. fold manifest gives the frozen 150/38 x3 + 151/37 x2 splits",
        sizes == [(150, 38), (150, 38), (150, 38), (151, 37), (151, 37)],
        f"{sizes}")

    # -- 9 train/test separation --------------------------------------------
    allp = set(ids)
    sep_ok = all(not (set(f["train_ids"]) & set(f["test_ids"]))
                 and set(f["train_ids"]) | set(f["test_ids"]) == allp
                 for f in folds)
    tests = [set(f["test_ids"]) for f in folds]
    disj = all(not (tests[i] & tests[j])
               for i in range(5) for j in range(i + 1, 5))
    rec("9. train/test separation holds and test folds partition the population",
        sep_ok and disj and set().union(*tests) == allp,
        f"per-fold train∩test=∅: {sep_ok}; test folds disjoint: {disj}; coverage 188/188")

    # -- 10..12 client partition --------------------------------------------
    part_ok, disj_ok, exh_ok = True, True, True
    details: List[str] = []
    for f in folds:
        parts = C.a2_partition_training(f["train_ids"], phq)
        st = C.check_partition(parts, set(f["train_ids"]), set(f["test_ids"]))
        exp_sizes = C.client_sizes(len(f["train_ids"]))
        if st["sizes"] != exp_sizes or len(parts) != C.N_CLIENTS:
            part_ok = False
        if not st["disjoint"] or st["duplicates"]:
            disj_ok = False
        if not st["coverage_ok"] or not st["no_leakage"]:
            exh_ok = False
        details.append(f"fold{f['fold']}:{st['sizes']}")
    rec("10. 5-client partition has the correct deterministic per-fold sizes",
        part_ok, " ".join(details) + "  (150→30x5, 151→31+30x4)")
    rec("11. clients are pairwise disjoint with zero duplicates", disj_ok,
        "no participant appears on two clients in any fold")
    rec("12. clients are exhaustive over TRAIN and leak no TEST participant",
        exh_ok, "union(clients) == train_ids and clients ∩ test_ids = ∅ (design SS 7.2)")

    # -- 13 model architecture ----------------------------------------------
    if a.skip_model:
        rec("13. model architecture 109,763,494 params / 215 keys", False,
            "SKIPPED via --skip-model (fail-closed: skipped is not passed)")
    else:
        snap = C.hf_snapshot_dir()
        if not os.path.isdir(snap):
            rec("13. model architecture 109,763,494 params / 215 keys", False,
                f"pinned MentalBERT snapshot missing: {snap}")
        else:
            artbad = [f for f, w in C.BERT_ARTIFACT_SHA.items()
                      if C.sha256_file(os.path.join(snap, f)) != w]
            if artbad:
                rec("13. model architecture 109,763,494 params / 215 keys", False,
                    f"MentalBERT artifact SHA mismatch: {artbad}")
            else:
                from trainer_mentalbert_daic import MultiModalModel
                m = MultiModalModel(snap, audio_dim=C.AUDIO_DIM,
                                    vision_dim=C.VISION_DIM, device="cpu")
                sd = m.state_dict()
                n = int(sum(v.numel() for v in sd.values()))
                rec("13. model architecture 109,763,494 params / 215 keys",
                    n == C.MODEL_N_PARAMS and len(sd) == C.MODEL_N_KEYS,
                    f"{len(sd)} keys, {n:,} params; fusion_in={C.FUSION_DIM}; "
                    f"revision {C.BERT_REVISION}")
                del m, sd

    # -- 14 training recipe --------------------------------------------------
    import inspect
    from trainer_mentalbert_daic import fine_tune_supervised
    sig = inspect.signature(fine_tune_supervised).parameters
    rec("14. frozen training recipe matches the design",
        sig["epochs"].default == 1 and sig["batch_size"].default == C.BATCH_SIZE
        and abs(sig["lr"].default - 2e-5) < 1e-12
        and C.LR == 2e-5 and C.EPOCHS == 3 and C.GRAD_CLIP == 1.0,
        f"design: AdamW lr={C.LR} epochs={C.EPOCHS} batch={C.BATCH_SIZE} "
        f"clip={C.GRAD_CLIP} dropout={C.DROPOUT} loss={C.LOSS}")

    # -- 15 delta convention -------------------------------------------------
    import torch
    from trainer_mentalbert_daic import compute_state_delta
    b = {"w": torch.zeros(3)}
    aa = {"w": torch.tensor([1.0, 2.0, 3.0])}
    rec("15. frozen compute_state_delta computes w_after - w_before",
        torch.equal(compute_state_delta(b, aa)["w"],
                    torch.tensor([1.0, 2.0, 3.0])),
        "verified against the real frozen function")

    # -- 16 DP configuration -------------------------------------------------
    eps = C.epsilon()
    rec("16. DP configuration frozen; epsilon recomputed live",
        abs(eps - C.EPS_PER_UPDATE) < 1e-12 and C.CLIP_NORM == 1.0
        and C.NOISE_MULTIPLIER == 1.0 and C.DELTA_DP == 1e-05,
        f"gaussian C={C.CLIP_NORM} nm={C.NOISE_MULTIPLIER} "
        f"eps={eps!r} delta={C.DELTA_DP} T=1")

    # -- 17 aggregation configuration ---------------------------------------
    import importlib.util as iu
    spec = iu.spec_from_file_location("p22agg",
                                      os.path.join(C.REPO, C.AGGREGATOR_PATH))
    am = iu.module_from_spec(spec)
    spec.loader.exec_module(am)
    agg = am.AggregatorAgent(mode=C.AGG_MODE, trim_ratio=C.TRIM_RATIO)
    probe = torch.tensor([[0.0], [1.0], [2.0], [3.0], [100.0]])
    got = float(agg._aggregate_tensor(probe).item())
    lo, up, kept = C.trim_window(C.N_CLIENTS)
    rec("17. frozen aggregator trims 1 low + 1 high, keeping 3 of 5",
        abs(got - 2.0) < 1e-9 and kept == 3 and agg.mode == C.AGG_MODE
        and agg.trim_ratio == C.TRIM_RATIO,
        f"[0,1,2,3,100] -> {got} (mean of 1,2,3); window [{lo}:{up}]")

    # -- 18 seed hierarchy ---------------------------------------------------
    seeds = {(f, c): C.client_seed(f, c) for f in C.FOLDS
             for c in range(1, C.N_CLIENTS + 1)}
    rec("18. seed hierarchy yields 25 unique client seeds",
        len(set(seeds.values())) == 25,
        f"fold_seed=(1000+1)*100+fold; client_seed=fold_seed*10+i; "
        f"{min(seeds.values())}..{max(seeds.values())}")

    # -- 19 GPU / CUDA -------------------------------------------------------
    cuda = torch.cuda.is_available()
    gpu = torch.cuda.get_device_name(0) if cuda else "NONE"
    vram = (torch.cuda.get_device_properties(0).total_memory / 1e9) if cuda else 0.0
    rec("19. CUDA is available (no silent CPU fallback)", cuda,
        f"torch={torch.__version__} cuda={torch.version.cuda} gpu={gpu} "
        f"vram={vram:.1f} GB" if cuda else
        f"torch={torch.__version__} CUDA NOT AVAILABLE - Phase 22 must run on GPU")

    # -- 20 output isolation -------------------------------------------------
    # The validator's own report lives here, so a second run would otherwise fail
    # this check and 20/20 could never be reached. The check exists to guarantee no
    # EXPERIMENT artifact will be overwritten, so validation reports are excluded.
    od = os.path.join(C.REPO, C.OUT_DIRNAME)
    existing = os.listdir(od) if os.path.isdir(od) else []
    artifacts = [f for f in existing if not f.startswith("validate_phase22_report")]
    rec("20. output directory holds no experiment artifacts", not artifacts,
        f"{C.OUT_DIRNAME} — {len(existing)} file(s), "
        f"{len(artifacts)} experiment artifact(s)"
        + (f": {artifacts}" if artifacts else " (validation reports ignored)"))

    n_pass = sum(1 for c in CHECKS if c["status"] == "PASS")
    n_fail = len(CHECKS) - n_pass
    print("-" * 78)
    print(f"RESULT: {n_pass} PASS, {n_fail} FAIL of {len(CHECKS)}  "
          f"({time.time() - t0:.1f}s)")
    verdict = "PASS" if n_fail == 0 else "FAIL"
    print(f"VALIDATION: {verdict}")
    if n_fail and a.allow_cpu:
        gpu_only = all(c["check"].startswith("19.")
                       for c in CHECKS if c["status"] == "FAIL")
        if gpu_only:
            print("NOTE: the ONLY failure is the CUDA check. All scientific and "
                  "structural checks passed. Re-run on the Colab GPU runtime.")
    print("=" * 78)

    if a.report:
        os.makedirs(od, exist_ok=True)
        rp = a.report if os.path.isabs(a.report) else os.path.join(od, a.report)
        with open(rp, "w", encoding="utf-8") as fh:
            json.dump({"schema": "phase22_validation/1", "experiment": C.EXPERIMENT,
                       "design": {"path": C.DESIGN_PATH, "sha256": ds},
                       "generated_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ",
                                                      time.gmtime()),
                       "environment": {"torch": torch.__version__,
                                       "cuda_available": bool(cuda),
                                       "cuda_version": torch.version.cuda,
                                       "gpu": gpu, "vram_gb": round(vram, 2),
                                       "python": sys.version.split()[0]},
                       "checks": CHECKS, "n_pass": n_pass, "n_fail": n_fail,
                       "verdict": verdict,
                       "scope_caveat": C.SCOPE_CAVEAT}, fh, indent=2,
                      sort_keys=True)
        print(f"[report] {rp}")

    return 0 if verdict == "PASS" else 1


if __name__ == "__main__":
    sys.exit(main())
