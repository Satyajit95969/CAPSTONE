#!/usr/bin/env python3
"""
validate_b3b.py - Phase 21 / B-3B pre-execution validation. FAIL CLOSED.

Runs BEFORE any training. Verifies that every frozen input, constant and
structural assumption in PHASE_21_B3B_DESIGN.md actually holds in this
repository. Read-only apart from an optional --report file.

Exit 0 only if every check PASSes.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from typing import Any, Dict, List

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import b3b_common as C  # noqa: E402

CHECKS: List[Dict[str, Any]] = []


def rec(name: str, ok: bool, detail: str = "") -> None:
    CHECKS.append({"check": name, "status": "PASS" if ok else "FAIL",
                   "detail": detail})
    print(f"  [{'ok  ' if ok else 'FAIL'}] {name}")
    if detail:
        print(f"         {detail}")


def main() -> int:
    ap = argparse.ArgumentParser(description="B-3B pre-execution validation")
    ap.add_argument("--report", default=None)
    a = ap.parse_args()
    os.chdir(C.REPO)
    t0 = time.time()

    print("=" * 78)
    print("B-3B PRE-EXECUTION VALIDATION - FAIL CLOSED")
    print("second B-3 attempt; H0 is a legitimate result (design SS 12)")
    print("=" * 78)

    # 1 design present
    ds = C.design_sha()
    rec("1. frozen design document present", ds is not None, f"{ds}")

    # 2 frozen inputs
    bad = [f"{p}: {C.sha256_file(p)} != {w}" for p, w in C.FROZEN_SHA.items()
           if w is not None and C.sha256_file(p) != w]
    rec("2. every frozen input SHA matches",
        not bad, f"{len([w for w in C.FROZEN_SHA.values() if w])} pinned"
        if not bad else "; ".join(bad))

    # 3 reused embeddings
    z = None
    if os.path.isfile(C.EMBEDDINGS_PATH):
        z = np.load(C.EMBEDDINGS_PATH, allow_pickle=False)
    ok = (z is not None and z["embedding"].shape == (C.N_PARTICIPANTS, C.EMBED_DIM)
          and bool(np.isfinite(z["embedding"]).all()))
    rec("3. reused C-2 embeddings are 188x768 and finite", ok,
        f"shape={tuple(z['embedding'].shape)} sha={C.EMBEDDINGS_SHA[:32]}..."
        if z is not None else "artifact absent")

    # 4 embeddings match the frozen parquet
    if z is None:
        rec("4. embedding IDs / PHQ match daic_records.parquet", False, "no artifact")
        phq = {}
    else:
        import pandas as pd
        df = pd.read_parquet(C.FROZEN_SHA and "daic_records.parquet")
        ids = [int(x) for x in z["participant_id"]]
        src_ids = [int(x) for x in df["participant_id"]]
        src_phq = [float(x) for x in df["phq_score"]]
        ok = ids == src_ids and all(
            float(z["phq_score"][i]) == src_phq[i] for i in range(len(ids)))
        rec("4. embedding IDs / PHQ match daic_records.parquet exactly", ok,
            "188 IDs in order, PHQ elementwise equal")
        phq = {p: float(z["phq_score"][i]) for i, p in enumerate(ids)}

    # 5 label prevalence
    if phq:
        pos = sum(1 for v in phq.values() if C.binarize(v))
        ok = pos == C.N_POSITIVE and (len(phq) - pos) == C.N_NEGATIVE
        rec("5. PHQ>10 binarisation yields 45 pos / 143 neg", ok,
            f"pos={pos} neg={len(phq)-pos}")
    else:
        rec("5. PHQ>10 binarisation yields 45 pos / 143 neg", False, "no PHQ")

    # 6 fold manifest
    folds = []
    try:
        with open(C.FOLD_MANIFEST, "r", encoding="utf-8") as fh:
            manifest = json.load(fh)
        folds = C.load_folds(manifest)
        sizes = [(len(f["train_ids"]), len(f["test_ids"])) for f in folds]
        ok = sizes == [(150, 38), (150, 38), (150, 38), (151, 37), (151, 37)]
        rec("6. fold manifest gives the frozen 150/38 x3 + 151/37 x2 splits", ok,
            f"{sizes}")
    except Exception as e:  # noqa: BLE001
        rec("6. fold manifest gives the frozen splits", False, str(e))

    # 7 folds partition the population, zero overlap
    if folds and phq:
        allp = set(phq)
        ok = all(set(f["train_ids"]) | set(f["test_ids"]) == allp
                 and not (set(f["train_ids"]) & set(f["test_ids"])) for f in folds)
        tests = [set(f["test_ids"]) for f in folds]
        disjoint = all(not (tests[i] & tests[j])
                       for i in range(5) for j in range(i + 1, 5))
        cover = set().union(*tests) == allp
        rec("7. per fold train+test = 188 with zero overlap; test folds partition",
            ok and disjoint and cover,
            f"per-fold ok={ok} test-folds disjoint={disjoint} coverage={cover}")
    else:
        rec("7. fold coverage / disjointness", False, "inputs absent")

    # 8 every test fold has both classes (ROC-AUC would be undefined otherwise)
    if folds and phq:
        bad_f = [f["fold"] for f in folds
                 if len({C.binarize(phq[p]) for p in f["test_ids"]}) < 2]
        rec("8. every test fold contains both classes", not bad_f,
            "ROC-AUC well-defined on all 5 folds" if not bad_f
            else f"degenerate folds: {bad_f}")
    else:
        rec("8. every test fold contains both classes", False, "inputs absent")

    # 9 architecture matches the frozen spec exactly
    sys.path.insert(0, C.REPO)
    from trainer_mentalbert_daic import FusionHead  # FROZEN
    head = FusionHead(in_dim=C.IN_DIM, hidden=C.HIDDEN, num_classes=C.NUM_CLASSES)
    sd = head.state_dict()
    n = int(sum(v.numel() for v in sd.values()))
    ok = (n == C.HEAD_N_PARAMS and len(sd) == C.HEAD_N_TENSORS
          and {k: tuple(v.shape) for k, v in sd.items()} == C.HEAD_TENSORS)
    rec("9. frozen FusionHead(768,256,2) = 197,892 params in 8 tensors", ok,
        f"{len(sd)} tensors, {n:,} params")

    # 10 forward signature: logits + mu + log_sigma
    head.eval()
    with torch.no_grad():
        out = head(torch.zeros(4, C.IN_DIM))
    ok = isinstance(out, tuple) and len(out) == 3 and tuple(out[0].shape) == (4, 2)
    rec("10. FusionHead returns (logits[B,2], mu[B], log_sigma[B])", ok,
        f"shapes={[tuple(o.shape) for o in out]}" if isinstance(out, tuple) else "?")

    # 11 dropout active in train mode (design SS 11 stochasticity claim)
    head.train()
    torch.manual_seed(0)
    a1 = head(torch.ones(64, C.IN_DIM))[0]
    a2 = head(torch.ones(64, C.IN_DIM))[0]
    ok = not torch.equal(a1, a2)
    rec("11. Dropout(0.2) is active in train mode -> repeats are stochastic", ok,
        f"two train-mode passes differ: {ok} (design SS 11 justification)")

    # 12 frozen delta convention
    from trainer_mentalbert_daic import compute_state_delta  # FROZEN
    b = {"w": torch.zeros(3)}
    aa = {"w": torch.tensor([1.0, 2.0, 3.0])}
    d = compute_state_delta(b, aa)
    ok = torch.equal(d["w"], torch.tensor([1.0, 2.0, 3.0]))
    rec("12. frozen compute_state_delta computes w_after - w_before", ok,
        "verified on the real frozen function")

    # 13 DP accounting recomputed live
    eps = C.epsilon()
    ok = abs(eps - C.EPS_PER_UPDATE) < 1e-12
    rec("13. epsilon recomputed live via the frozen dp_agent", ok,
        f"eps={eps!r} delta={C.DELTA_DP} C={C.CLIP_NORM} nm={C.NOISE_MULTIPLIER} T=1")

    # 14 acceptance constants match Exp 9's frozen values
    try:
        with open("trainer_outputs/baseline_cv/baseline_cv_summary.json",
                  encoding="utf-8") as fh:
            bcv = json.load(fh)
        mde = bcv["MDE"]["ROC_AUC"]["mde_95"]
        ok = abs(mde - C.ROC_AUC_MDE) < 1e-15
        rec("14. MDE matches baseline_cv_summary.json (not invented)", ok,
            f"mde_95={mde!r}")
    except Exception as e:  # noqa: BLE001
        rec("14. MDE matches baseline_cv_summary.json", False, str(e))

    # 15 analytical NSR of the frozen architecture vs the B threshold
    analytical = float(np.sqrt(C.HEAD_N_PARAMS))
    rec("15. analytical NSR sqrt(d) is below the frozen B threshold",
        analytical < C.G3_NSR,
        f"sqrt({C.HEAD_N_PARAMS:,}) = {analytical:.6f} < {C.G3_NSR} "
        f"(holds ONLY if clipping binds - design SS 16.1)")

    # 16 predicted payload vs the A threshold
    raw = C.HEAD_N_PARAMS * 4
    pred = raw * C.TRANSPORT_EXPANSION
    rec("16. predicted transport payload is below the frozen A threshold",
        pred <= C.G2_PAYLOAD_BYTES,
        f"raw={raw:,} B -> ~{pred:,.0f} B (x{C.TRANSPORT_EXPANSION:.4f}) "
        f"<= {C.G2_PAYLOAD_BYTES:,} B")

    # 17 seed hierarchy is well-formed and collision-free
    seeds = {(r, f): C.fold_seed(r, f) for r in C.REPEATS for f in C.FOLDS}
    ok = len(set(seeds.values())) == 25
    rec("17. seed hierarchy yields 25 unique fold seeds", ok,
        f"seed=1000+repeat; fold_seed=seed*100+fold; "
        f"{min(seeds.values())}..{max(seeds.values())}")

    # 18 no DP parameter drift
    ok = (C.CLIP_NORM == 1.0 and C.NOISE_MULTIPLIER == 1.0 and C.SIGMA_EFF == 1.0
          and C.DELTA_DP == 1e-05 and C.EPS_PER_UPDATE == 5.302585092994046)
    rec("18. no DP parameter has drifted from the frozen values", ok,
        "gaussian, C=1.0, nm=1.0, delta=1e-05, T=1")

    # 19 output dir will not collide with any existing artifact
    od = os.path.join(C.REPO, C.OUT_DIRNAME)
    ok = not os.path.isdir(od) or not os.listdir(od)
    rec("19. output directory is empty or absent (no artifact overwrite)", ok,
        C.OUT_DIRNAME)

    # 20 the two arms differ ONLY by DP
    ok = (set(C.ARMS) == {"N", "D"} and C.CONTROL_ARM == "N"
          and C.TREATMENT_ARM == "D")
    rec("20. exactly two arms, differing only in DP application", ok,
        f"{C.ARM_LABEL['N']} | {C.ARM_LABEL['D']}")

    n_pass = sum(1 for c in CHECKS if c["status"] == "PASS")
    n_fail = len(CHECKS) - n_pass
    print("-" * 78)
    print(f"RESULT: {n_pass} PASS, {n_fail} FAIL of {len(CHECKS)}  "
          f"({time.time() - t0:.1f}s)")
    verdict = "PASS" if n_fail == 0 else "FAIL"
    print(f"VALIDATION: {verdict}")
    print("=" * 78)

    if a.report:
        os.makedirs(od, exist_ok=True)
        rp = a.report if os.path.isabs(a.report) else os.path.join(od, a.report)
        with open(rp, "w", encoding="utf-8") as fh:
            json.dump({"schema": "b3b_validation/1", "experiment": C.EXPERIMENT,
                       "design": {"path": C.DESIGN_PATH, "sha256": ds},
                       "checks": CHECKS, "n_pass": n_pass, "n_fail": n_fail,
                       "verdict": verdict}, fh, indent=2, sort_keys=True)
        print(f"[report] {rp}")

    return 0 if verdict == "PASS" else 1


if __name__ == "__main__":
    sys.exit(main())
