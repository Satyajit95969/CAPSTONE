#!/usr/bin/env python3
"""
verify_b3b.py - Phase 21 / B-3B independent fail-closed verification.

  PASS    - independently recomputed and matched
  FAIL    - recomputed and did NOT match
  PENDING - the evidence needed to judge is absent

Missing evidence is NEVER PASS. Exit non-zero unless every check is PASS.

This verifier re-derives the result rather than trusting b3b_summary.json: it
retrains ALL 25 folds from scratch with an independent implementation, recomputes
both arms' ROC-AUC, recomputes A, B and C against the frozen thresholds, and only
then compares against the recorded artifacts.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import sys
import time
from typing import Any, Dict, List, Optional

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import b3b_common as C  # noqa: E402
from trainer_mentalbert_daic import FusionHead, compute_state_delta  # noqa: E402
from transformers import AdamW  # noqa: E402

CHECKS: List[Dict[str, Any]] = []


def record(name: str, status: str, detail: str = "") -> None:
    CHECKS.append({"check": name, "status": status, "detail": detail})
    tag = {"PASS": "ok  ", "FAIL": "FAIL", "PENDING": "PEND"}[status]
    print(f"  [{tag}] {name}")
    if detail:
        print(f"         {detail}")


def cmp_check(name: str, got: Any, want: Any, detail: str = "") -> None:
    if got is None:
        record(name, "PENDING", detail or "evidence absent")
    elif got == want:
        record(name, "PASS", detail or f"{got!r}")
    else:
        record(name, "FAIL", detail or f"got {got!r}, expected {want!r}")


def load_json(p: str) -> Optional[Dict[str, Any]]:
    if not os.path.isfile(p):
        return None
    with open(p, "r", encoding="utf-8") as fh:
        return json.load(fh)


def retrain(X, y, phq, seed):
    """Independent re-implementation of the frozen recipe (design SS 6)."""
    torch.manual_seed(seed)
    h = FusionHead(in_dim=C.IN_DIM, hidden=C.HIDDEN, num_classes=C.NUM_CLASSES)
    wb = {k: v.detach().cpu().clone() for k, v in h.state_dict().items()}
    ds = TensorDataset(torch.tensor(X, dtype=torch.float32),
                       torch.tensor(y, dtype=torch.long),
                       torch.tensor(phq, dtype=torch.float32))
    dl = DataLoader(ds, batch_size=C.BATCH_SIZE, shuffle=C.SHUFFLE)
    opt = AdamW(h.parameters(), lr=C.LR)
    ce, mse = nn.CrossEntropyLoss(), nn.MSELoss()
    for _ in range(C.EPOCHS):
        h.train()
        for xb, yb, pb in dl:
            opt.zero_grad()
            lo, mu, _ = h(xb)
            (ce(lo, yb) + C.LAMBDA_REG * mse(mu, pb)).backward()
            torch.nn.utils.clip_grad_norm_(h.parameters(), C.GRAD_CLIP)
            opt.step()
    wa = {k: v.detach().cpu().clone() for k, v in h.state_dict().items()}
    return wb, wa


def score(state, X, y):
    h = FusionHead(in_dim=C.IN_DIM, hidden=C.HIDDEN, num_classes=C.NUM_CLASSES)
    h.load_state_dict(state, strict=True)
    h.eval()
    with torch.no_grad():
        lo, _, _ = h(torch.tensor(X, dtype=torch.float32))
        p = torch.softmax(lo, dim=1)[:, 1].cpu().numpy()
    return C.roc_auc(p, y)


def main() -> int:
    ap = argparse.ArgumentParser(description="B-3B independent verification")
    ap.add_argument("--out-dir", default=os.path.join(C.REPO, C.OUT_DIRNAME))
    ap.add_argument("--report", default=None)
    a = ap.parse_args()
    os.chdir(C.REPO)
    od = a.out_dir
    t0 = time.time()

    print("=" * 78)
    print("B-3B INDEPENDENT VERIFICATION - FAIL CLOSED")
    print("=" * 78)

    summary = load_json(os.path.join(od, "b3b_summary.json"))
    receipt = load_json(os.path.join(od, "b3b_receipt.json"))
    aucj = load_json(os.path.join(od, "b3b_roc_auc.json"))
    deltaj = load_json(os.path.join(od, "b3b_deltas.json"))
    payj = load_json(os.path.join(od, "b3b_payload.json"))
    embr = load_json(os.path.join(od, "b3b_embeddings_receipt.json"))

    # 1 design pinned into artifacts
    live = C.design_sha()
    pinned = {n: (o or {}).get("design", {}).get("sha256")
              for n, o in (("summary", summary), ("receipt", receipt),
                           ("roc_auc", aucj), ("deltas", deltaj), ("payload", payj))}
    seen = {n: v for n, v in pinned.items() if v is not None}
    if not seen:
        record("1. design unchanged since the run", "PENDING", "no artifact pins it")
    else:
        mism = {n: v for n, v in seen.items() if v != live}
        record("1. design unchanged since the run (SHA pinned in artifacts)",
               "PASS" if not mism else "FAIL",
               f"live {live} == SHA in all {len(seen)} artifacts" if not mism
               else f"disagreeing: {mism}")

    # 2 frozen inputs
    mism = [f"{p}" for p, w in C.FROZEN_SHA.items()
            if w is not None and C.sha256_file(p) != w]
    record("2. every frozen input SHA unchanged", "PASS" if not mism else "FAIL",
           f"{len([w for w in C.FROZEN_SHA.values() if w])} pinned artifacts"
           if not mism else f"changed: {mism}")

    # 3 C-2 artifacts untouched
    ok = C.sha256_file(C.EMBEDDINGS_PATH) == C.EMBEDDINGS_SHA
    record("3. reused C-2 embedding artifact is byte-identical (C-2 not re-run)",
           "PASS" if ok else "FAIL", f"sha {C.EMBEDDINGS_SHA}")

    # 4 architecture
    h = FusionHead(in_dim=C.IN_DIM, hidden=C.HIDDEN, num_classes=C.NUM_CLASSES)
    sd = h.state_dict()
    n = int(sum(v.numel() for v in sd.values()))
    record("4. frozen FusionHead(768,256,2) = 197,892 params in 8 tensors",
           "PASS" if (n == C.HEAD_N_PARAMS and len(sd) == C.HEAD_N_TENSORS) else "FAIL",
           f"{len(sd)} tensors, {n:,} params")

    # 5 thresholds not drifted from their frozen sources
    bcv = load_json("trainer_outputs/baseline_cv/baseline_cv_summary.json")
    mde_src = (bcv or {}).get("MDE", {}).get("ROC_AUC", {}).get("mde_95")
    cmp_check("5. MDE equals baseline_cv_summary.json (not invented)",
              mde_src, C.ROC_AUC_MDE, f"mde_95={mde_src!r}")

    # 6 epsilon live
    eps = C.epsilon()
    cmp_check("6. epsilon recomputed live via the frozen dp_agent",
              eps, C.EPS_PER_UPDATE,
              f"eps={eps!r} delta={C.DELTA_DP} C={C.CLIP_NORM} nm={C.NOISE_MULTIPLIER}")

    # 7 INDEPENDENT RE-DERIVATION of all 25 runs, both arms
    z = np.load(C.EMBEDDINGS_PATH, allow_pickle=False)
    ids = [int(x) for x in z["participant_id"]]
    E = z["embedding"].astype(np.float64)
    PHQ = z["phq_score"].astype(np.float64)
    Y = np.array([C.binarize(v) for v in PHQ], dtype=np.int64)
    idx = {p: i for i, p in enumerate(ids)}
    with open(C.FOLD_MANIFEST, encoding="utf-8") as fh:
        folds = C.load_folds(json.load(fh))

    rec_auc = {ar: {f: {} for f in C.FOLDS} for ar in C.ARMS}
    rec_nsr, rec_clipped, rec_pre = [], 0, []
    for r in C.REPEATS:
        for fs in folds:
            f = fs["fold"]
            seed = C.fold_seed(r, f)
            tr = [idx[p] for p in fs["train_ids"]]
            te = [idx[p] for p in fs["test_ids"]]
            wb, wa = retrain(E[tr], Y[tr], PHQ[tr], seed)
            d = compute_state_delta(wb, wa)
            flat = np.concatenate([d[k].numpy().astype(np.float64).ravel()
                                   for k in C.HEAD_TENSORS])
            rec_auc["N"][f][r] = score(wa, E[te], Y[te])

            v = torch.tensor(flat, dtype=torch.float32)
            l2 = float(torch.norm(v, p=2))
            rec_pre.append(l2)
            if l2 > C.CLIP_NORM:
                rec_clipped += 1
            clipped = v * min(1.0, C.CLIP_NORM / (l2 + 1e-12))
            g = torch.Generator().manual_seed(seed)
            noise = torch.normal(0.0, C.SIGMA_EFF * C.CLIP_NORM,
                                 size=clipped.shape, generator=g)
            rec_nsr.append(float(torch.norm(noise, p=2)) /
                           float(torch.norm(clipped, p=2)))
            noisy = (clipped + noise).numpy().astype(np.float64)
            off, st = 0, {}
            for k, shp in C.HEAD_TENSORS.items():
                m = int(np.prod(shp))
                st[k] = wb[k] + torch.tensor(noisy[off:off + m],
                                             dtype=torch.float32).reshape(shp)
                off += m
            rec_auc["D"][f][r] = score(st, E[te], Y[te])

    if aucj is None:
        record("7. all 25 runs reproduce from an independent retrain", "PENDING",
               "b3b_roc_auc.json absent")
        worst = None
    else:
        rj = aucj["by_arm_fold_repeat"]
        worst = max(abs(rec_auc[ar][f][r] - rj[ar][str(f)][str(r)])
                    for ar in C.ARMS for f in C.FOLDS for r in C.REPEATS)
        record("7. all 25 runs x 2 arms reproduce from an INDEPENDENT retrain",
               "PASS" if worst == 0.0 else "FAIL",
               f"50 recorded ROC-AUC values; max |diff| = {worst:.3e}")

    # 8 fold means recomputed
    rec_fold = {ar: [float(np.mean([rec_auc[ar][f][r] for r in C.REPEATS]))
                     for f in C.FOLDS] for ar in C.ARMS}
    rec_d_mean = float(np.mean(rec_fold["D"]))
    rec_n_mean = float(np.mean(rec_fold["N"]))
    stored_d = (summary or {}).get("arms", {}).get("D", {}).get("mean")
    ok = stored_d is not None and abs(stored_d - rec_d_mean) < 1e-12
    record("8. fold-level means recomputed (repeats averaged within fold)",
           "PASS" if ok else ("PENDING" if stored_d is None else "FAIL"),
           f"D mean recomputed={rec_d_mean:.9f} recorded={stored_d}")

    # 9 criterion A
    raw = C.HEAD_N_PARAMS * 4
    rec_a_val = raw * C.TRANSPORT_EXPANSION
    rec_a = bool(rec_a_val <= C.G2_PAYLOAD_BYTES)
    stored_a = (summary or {}).get("criteria", {}).get("A_payload", {}).get("pass")
    cmp_check("9. criterion A recomputed independently", stored_a, rec_a,
              f"{rec_a_val:,.0f} B <= {C.G2_PAYLOAD_BYTES:,} B -> {rec_a}")

    # 10 criterion B
    rec_b_val = float(np.mean(rec_nsr))
    rec_b = bool(rec_b_val < C.G3_NSR)
    stored_b = (summary or {}).get("criteria", {}).get("B_nsr", {}).get("pass")
    cmp_check("10. criterion B recomputed independently", stored_b, rec_b,
              f"mean NSR {rec_b_val:.6f} < {C.G3_NSR} -> {rec_b}")

    # 11 criterion C
    rec_c1 = C.in_ci(rec_d_mean)
    rec_degr = float(np.mean([rec_fold["N"][i] - rec_fold["D"][i]
                              for i in range(C.N_FOLDS)]))
    rec_c2 = bool(rec_degr <= C.ROC_AUC_MDE)
    rec_c = bool(rec_c1 and rec_c2)
    stored_c = (summary or {}).get("criteria", {}).get("C_task", {}).get("pass")
    cmp_check("11. criterion C recomputed independently", stored_c, rec_c,
              f"C1 ci({rec_d_mean:.9f})={rec_c1}  C2 degr({rec_degr:+.9f} "
              f"<= {C.ROC_AUC_MDE})={rec_c2} -> C={rec_c}")

    # 12 overall verdict
    rec_verdict = "SUCCESS" if (rec_a and rec_b and rec_c) else "H0"
    cmp_check("12. overall verdict equals the independent recomputation",
              (summary or {}).get("verdict"), rec_verdict,
              f"recomputed={rec_verdict}")

    # 13 clipping statistics
    stored_clip = (deltaj or {}).get("clipping", {}).get("n_clipped")
    cmp_check("13. clipping count recomputed independently", stored_clip,
              rec_clipped,
              f"{rec_clipped}/25 clipped; pre-clip norms "
              f"{min(rec_pre):.4f}-{max(rec_pre):.4f} vs C={C.CLIP_NORM}")

    # 14 repeats genuinely varied (design SS 11 claim)
    sds = [float(np.std([rec_auc[ar][f][r] for r in C.REPEATS], ddof=1))
           for ar in C.ARMS for f in C.FOLDS]
    varied = bool(np.mean(sds) > 0.0)
    claimed = (summary or {}).get("arms", {}).get("D", {}).get(
        "repeat_structure", {}).get("repeats_genuinely_varied")
    cmp_check("14. repeats genuinely varied (not pseudo-replication)",
              claimed, varied,
              f"mean within-fold SD across repeats = {np.mean(sds):.6f} > 0")

    # 15 seeds
    seeds = {C.fold_seed(r, f) for r in C.REPEATS for f in C.FOLDS}
    record("15. seed hierarchy yields 25 unique fold seeds",
           "PASS" if len(seeds) == 25 else "FAIL",
           f"seed=1000+repeat, fold_seed=seed*100+fold; "
           f"{min(seeds)}..{max(seeds)}")

    # 16 completeness
    cpath = os.path.join(od, "b3b_runs.csv")
    nrows = (sum(1 for _ in csv.DictReader(open(cpath, encoding="utf-8")))
             if os.path.isfile(cpath) else None)
    cmp_check("16. all 25 runs recorded", nrows, C.N_REPEATS * C.N_FOLDS,
              f"{nrows} rows" if nrows else "CSV absent")

    # 17 no threshold drift
    ok = (C.G2_PAYLOAD_BYTES == 1_579_963 and C.G3_NSR == 544.341809
          and C.ROC_AUC_CI == (0.5755177227457472, 0.6911434704671701)
          and C.ROC_AUC_MDE == 0.05781287386071144)
    record("17. no acceptance threshold was moved, added or invented",
           "PASS" if ok else "FAIL",
           "A/B/C thresholds identical to their frozen Exp 9 / Baseline-CV sources")

    # 18 roadmap claim
    r10 = (summary or {}).get("roadmap", {}).get("row_10_unblocked")
    cmp_check("18. summary does NOT claim Row 10 is unblocked", r10, False,
              "Row 10 requires B-2 AND B-3; B-2 (Exp 8) returned H0")

    # 19 embeddings receipt
    cmp_check("19. embedding reuse receipt records a verified SHA",
              (embr or {}).get("verified"), True,
              "C-2 artifact reused read-only and SHA-verified")

    # 20 artifact set
    want = ["b3b_embeddings_receipt.json", "b3b_runs.csv", "b3b_deltas.json",
            "b3b_roc_auc.json", "b3b_payload.json", "b3b_receipt.json",
            "b3b_summary.json", "b3b_acceptance.csv"]
    missing = [f for f in want if not os.path.isfile(os.path.join(od, f))]
    record("20. all 8 declared artifacts present (design SS 18)",
           "PASS" if not missing else "FAIL",
           f"{len(want)} artifacts" if not missing else f"missing: {missing}")

    n_pass = sum(1 for c in CHECKS if c["status"] == "PASS")
    n_fail = sum(1 for c in CHECKS if c["status"] == "FAIL")
    n_pend = sum(1 for c in CHECKS if c["status"] == "PENDING")
    print("=" * 78)
    print(f"RESULT: {n_pass} PASS, {n_fail} FAIL, {n_pend} PENDING of {len(CHECKS)}"
          f"   ({time.time() - t0:.1f}s)")
    verdict = "PASS" if (n_fail == 0 and n_pend == 0) else "FAIL"
    print(f"VERIFICATION: {verdict}")
    print("=" * 78)

    if a.report:
        rp = a.report if os.path.isabs(a.report) else os.path.join(od, a.report)
        with open(rp, "w", encoding="utf-8") as fh:
            json.dump({"schema": "b3b_verification/1", "experiment": C.EXPERIMENT,
                       "design": {"path": C.DESIGN_PATH, "sha256": live},
                       "generated_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ",
                                                      time.gmtime()),
                       "checks": CHECKS, "n_pass": n_pass, "n_fail": n_fail,
                       "n_pending": n_pend, "verdict": verdict,
                       "independent_recomputation": {
                           "roc_auc_max_abs_diff": worst,
                           "D_mean": rec_d_mean, "N_mean": rec_n_mean,
                           "A_pass": rec_a, "B_pass": rec_b, "C_pass": rec_c,
                           "experiment_verdict": rec_verdict}},
                      fh, indent=2, sort_keys=True, default=float)
        print(f"[report] {rp}")

    return 0 if verdict == "PASS" else 1


if __name__ == "__main__":
    sys.exit(main())
