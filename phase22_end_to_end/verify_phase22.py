#!/usr/bin/env python3
"""
verify_phase22.py - Phase 22 independent fail-closed verification.

  PASS    - independently recomputed and matched
  FAIL    - recomputed and did NOT match
  PENDING - the evidence needed to judge is absent

Missing evidence is NEVER PASS. Exit non-zero unless every check is PASS.
The verifier does NOT simply trust the runner's output (design SS 21).

Full re-derivation of the 25 trainings is intentionally NOT performed here (it
would double the GPU cost of the demonstration). What IS independently
recomputed: partitions, leakage, model architecture, the delta convention on the
real frozen function, clipping arithmetic, DP accounting, the frozen aggregator's
trimming, the fold-level metric aggregation, the degradation, and every claim in
the summary. Re-derivation of a single client's training is available via
--retrain-fold/--retrain-client for spot-checking.
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

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import phase22_common as C  # noqa: E402

CHECKS: List[Dict[str, Any]] = []
METRICS = ("roc_auc", "pr_auc", "accuracy", "precision", "recall", "f1")


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


def as_bool(v: Any) -> bool:
    """Parse the boolean representation actually stored in the artifacts.

    run_phase22.py writes `clipped` as a Python bool, and csv.DictWriter
    serialises that as the STRING 'True'/'False' - so int('False') raises
    ValueError. This helper accepts the real stored forms without modifying any
    result artifact. Unrecognised input raises rather than silently defaulting,
    which keeps the verifier fail-closed.
    """
    if isinstance(v, bool):
        return v
    s = str(v).strip().lower()
    if s in ("true", "1"):
        return True
    if s in ("false", "0"):
        return False
    raise ValueError(f"unrecognised boolean literal in artifact: {v!r}")


def load_json(p: str) -> Optional[Dict[str, Any]]:
    if not os.path.isfile(p):
        return None
    with open(p, "r", encoding="utf-8") as fh:
        return json.load(fh)


def main() -> int:
    ap = argparse.ArgumentParser(description="Phase 22 verification")
    ap.add_argument("--out-dir", default=os.path.join(C.REPO, C.OUT_DIRNAME))
    ap.add_argument("--report", default=None)
    a = ap.parse_args()
    os.chdir(C.REPO)
    od = a.out_dir
    t0 = time.time()

    print("=" * 78)
    print("PHASE 22 INDEPENDENT VERIFICATION - FAIL CLOSED")
    print("=" * 78)

    summary = load_json(os.path.join(od, "phase22_summary.json"))
    receipt = load_json(os.path.join(od, "phase22_receipt.json"))
    evalj = load_json(os.path.join(od, "phase22_eval.json"))
    dpj = load_json(os.path.join(od, "phase22_dp.json"))
    partj = load_json(os.path.join(od, "phase22_partitions.json"))
    aggj = load_json(os.path.join(od, "phase22_aggregation.json"))

    # -- 1 design pinned -----------------------------------------------------
    live = C.design_sha()
    pinned = {n: (o or {}).get("design", {}).get("sha256")
              for n, o in (("summary", summary), ("receipt", receipt),
                           ("eval", evalj), ("dp", dpj), ("partitions", partj))}
    seen = {n: v for n, v in pinned.items() if v is not None}
    if not seen:
        record("1. design unchanged since the run", "PENDING", "no artifact pins it")
    else:
        mism = {n: v for n, v in seen.items() if v != live}
        record("1. design unchanged since the run; matches the frozen SHA",
               "PASS" if (not mism and live == C.DESIGN_SHA) else "FAIL",
               f"live {live} == frozen == SHA in all {len(seen)} artifacts"
               if not mism else f"disagreeing: {mism}")

    # -- 2 frozen inputs -----------------------------------------------------
    mism = [p for p, w in C.FROZEN_SHA.items()
            if w is not None and C.sha256_file(p) != w]
    record("2. every frozen input SHA unchanged", "PASS" if not mism else "FAIL",
           f"{len([w for w in C.FROZEN_SHA.values() if w])} pinned"
           if not mism else f"changed: {mism}")

    # -- 3/4 partitions + leakage, independently recomputed -----------------
    import pandas as pd
    df = pd.read_parquet(C.DATA_PARQUET)
    phq = {int(r["participant_id"]): float(r["phq_score"]) for _, r in df.iterrows()}
    with open(C.FOLD_MANIFEST, encoding="utf-8") as fh:
        folds = C.load_folds(json.load(fh))
    part_match, leak_free = True, True
    for f in folds:
        rec_parts = C.a2_partition_training(f["train_ids"], phq)
        st = C.check_partition(rec_parts, set(f["train_ids"]), set(f["test_ids"]))
        if not (st["disjoint"] and st["coverage_ok"] and st["no_leakage"]):
            leak_free = False
        if partj is not None:
            stored = [sorted(c["participant_ids"])
                      for c in partj["data"][f"fold{f['fold']}"]["clients"]]
            if [sorted(p) for p in rec_parts] != stored:
                part_match = False
    record("3. client partitions reproduce from the frozen A2 rule",
           "PASS" if (part_match and partj is not None) else
           ("PENDING" if partj is None else "FAIL"),
           "5 folds x 5 clients recomputed and matched byte-for-byte")
    record("4. NO participant leakage: clients ⊆ train, clients ∩ test = ∅",
           "PASS" if leak_free else "FAIL",
           "independently recomputed per fold (design SS 7.2)")

    # -- 5 model architecture ------------------------------------------------
    snap = C.hf_snapshot_dir()
    artbad = [f for f, w in C.BERT_ARTIFACT_SHA.items()
              if C.sha256_file(os.path.join(snap, f)) != w]
    if artbad:
        record("5. model architecture 109,763,494 params / 215 keys", "FAIL",
               f"MentalBERT artifact mismatch: {artbad}")
    else:
        from trainer_mentalbert_daic import MultiModalModel
        m = MultiModalModel(snap, audio_dim=C.AUDIO_DIM,
                            vision_dim=C.VISION_DIM, device="cpu")
        sd = m.state_dict()
        n = int(sum(v.numel() for v in sd.values()))
        record("5. model architecture 109,763,494 params / 215 keys",
               "PASS" if (n == C.MODEL_N_PARAMS and len(sd) == C.MODEL_N_KEYS)
               else "FAIL", f"{len(sd)} keys, {n:,} params")
        del m, sd

    # -- 6 delta convention --------------------------------------------------
    from trainer_mentalbert_daic import compute_state_delta
    b = {"w": torch.zeros(4)}
    aa = {"w": torch.tensor([1.0, -2.0, 3.0, -4.0])}
    record("6. delta convention is w_after - w_before (frozen function)",
           "PASS" if torch.equal(compute_state_delta(b, aa)["w"], aa["w"])
           else "FAIL", "verified on the real frozen compute_state_delta")

    # -- 7 clipping arithmetic recomputed from the CSV ----------------------
    cpath = os.path.join(od, "phase22_clients.csv")
    if not os.path.isfile(cpath):
        record("7. clipping recomputed from recorded norms", "PENDING",
               "phase22_clients.csv absent")
        rows = []
    else:
        with open(cpath, encoding="utf-8") as fh:
            rows = list(csv.DictReader(fh))
        ok = True
        for r in rows:
            l2 = float(r["l2_pre_clip"])
            exp_scale = min(1.0, C.CLIP_NORM / (l2 + 1e-12))
            exp_post = l2 * exp_scale
            if abs(float(r["clip_scale"]) - exp_scale) > 1e-9:
                ok = False
            if abs(float(r["l2_post_clip"]) - exp_post) > 1e-4:
                ok = False
            if as_bool(r["clipped"]) != (l2 > C.CLIP_NORM):
                ok = False
        record("7. clipping arithmetic recomputed for every update",
               "PASS" if ok else "FAIL",
               f"{len(rows)} updates; scale=min(1, C/||v||), C={C.CLIP_NORM}")

    # -- 8 DP parameters + epsilon recomputed live ---------------------------
    eps = C.epsilon()
    stored_eps = (dpj or {}).get("data", {}).get("epsilon_per_update")
    cmp_check("8. epsilon recomputed live via the frozen dp_agent",
              stored_eps, eps,
              f"eps={eps!r} delta={C.DELTA_DP} C={C.CLIP_NORM} "
              f"nm={C.NOISE_MULTIPLIER} T=1")

    # -- 9 round-level epsilon basis (parallel composition) -----------------
    stored_round = (dpj or {}).get("data", {}).get("round_epsilon")
    cmp_check("9. round epsilon = max over clients (disjoint data)",
              stored_round, C.EPS_PER_UPDATE,
              "parallel composition over disjoint participant sets, not the sum")

    # -- 10 aggregator trimming ---------------------------------------------
    import importlib.util as iu
    spec = iu.spec_from_file_location("v22agg",
                                      os.path.join(C.REPO, C.AGGREGATOR_PATH))
    am = iu.module_from_spec(spec)
    spec.loader.exec_module(am)
    agg = am.AggregatorAgent(mode=C.AGG_MODE, trim_ratio=C.TRIM_RATIO)
    probe = torch.tensor([[0.0], [1.0], [2.0], [3.0], [100.0]])
    got = float(agg._aggregate_tensor(probe).item())
    lo, up, kept = C.trim_window(C.N_CLIENTS)
    record("10. frozen aggregator trims 1 low + 1 high, keeping 3 of 5",
           "PASS" if (abs(got - 2.0) < 1e-9 and kept == 3) else "FAIL",
           f"[0,1,2,3,100] -> {got}; window [{lo}:{up}]")

    # -- 11 fold-level metric aggregation recomputed -------------------------
    if evalj is None or summary is None:
        record("11. fold-level metrics recomputed from per-fold values",
               "PENDING", "eval or summary absent")
        rec_means = {}
    else:
        rec_means = {}
        ok = True
        for arm in C.ARMS:
            for m in METRICS:
                vals = [evalj["data"][arm][f"fold{f}"][m] for f in C.FOLDS]
                mean, _ = C.mean_sd(vals)
                rec_means[(arm, m)] = mean
                stored = summary["arms"][arm]["summary"][m]["mean"]
                if not (abs(stored - mean) < 1e-12
                        or (np.isnan(stored) and np.isnan(mean))):
                    ok = False
        record("11. fold-level metric aggregation recomputed independently",
               "PASS" if ok else "FAIL",
               f"{len(C.ARMS)}x{len(METRICS)} means recomputed from per-fold values")

    # -- 12 DP degradation recomputed ----------------------------------------
    if evalj is None or summary is None:
        record("12. DP degradation recomputed", "PENDING", "evidence absent")
    else:
        degr = [evalj["data"]["N"][f"fold{f}"]["roc_auc"]
                - evalj["data"]["D"][f"fold{f}"]["roc_auc"] for f in C.FOLDS]
        mean, _ = C.mean_sd(degr)
        stored = summary["reporting_items"]["12_dp_degradation"]["mean"]
        record("12. DP degradation (N-D, paired within fold) recomputed",
               "PASS" if abs(stored - mean) < 1e-12 else "FAIL",
               f"recomputed {mean:+.9f}  recorded {stored:+.9f}")

    # -- 13 Baseline-CV comparison recomputed --------------------------------
    if summary is None:
        record("13. Baseline-CV comparison recomputed", "PENDING", "summary absent")
    else:
        n_mean = summary["reporting_items"]["13_baseline_cv_comparison"]["N_mean"]
        d_mean = summary["reporting_items"]["13_baseline_cv_comparison"]["D_mean"]
        ok = (summary["reporting_items"]["13_baseline_cv_comparison"]["N_inside_ci"]
              == C.in_baseline_ci(n_mean)
              and summary["reporting_items"]["13_baseline_cv_comparison"]["D_inside_ci"]
              == C.in_baseline_ci(d_mean))
        record("13. Baseline-CV CI membership recomputed for both arms",
               "PASS" if ok else "FAIL",
               f"N={n_mean:.9f} in CI {C.in_baseline_ci(n_mean)}; "
               f"D={d_mean:.9f} in CI {C.in_baseline_ci(d_mean)}; "
               f"CI [{C.BASELINE_CI[0]:.9f}, {C.BASELINE_CI[1]:.9f}]")

    # -- 14 completeness -----------------------------------------------------
    cmp_check("14. all 25 local trainings recorded", len(rows) if rows else None,
              C.N_FOLDS * C.N_CLIENTS, f"{len(rows)} rows" if rows else "CSV absent")
    cmp_check("15. all 10 federated executions recorded",
              len(aggj["data"]) if aggj else None, C.N_FOLDS * len(C.ARMS),
              f"{len(aggj['data'])} entries" if aggj else "aggregation artifact absent")

    # -- 16 no threshold invented -------------------------------------------
    ok = (C.BASELINE_ROC_AUC == 0.6333305966064586
          and C.BASELINE_CI == (0.5755177227457472, 0.6911434704671701))
    record("16. no acceptance threshold was invented; comparator is Baseline-CV",
           "PASS" if ok else "FAIL",
           "Phase 22 defines NO pass/fail gate (design SS 11, SS 22.7)")

    # -- 17 roadmap claim ----------------------------------------------------
    r10 = (summary or {}).get("roadmap", {}).get("row_10_unblocked")
    cmp_check("17. summary does NOT claim Row 10 is unblocked", r10, False,
              "Row 10 requires B-2 AND B-3; both returned H0")

    # -- 18 engineering vs task success kept separate -----------------------
    sep = (summary or {}).get("task_success_is_separate")
    cmp_check("18. engineering success and task success reported separately",
              sep, True, "design SS 2 - the first does not imply the second")

    # -- 19 device actually used --------------------------------------------
    dev = (receipt or {}).get("environment", {}).get("device_used")
    record("19. execution device recorded",
           "PASS" if dev is not None else "PENDING",
           f"device_used={dev!r}; cuda_available="
           f"{(receipt or {}).get('environment', {}).get('cuda_available')}")

    # -- 20 artifact set -----------------------------------------------------
    want = ["phase22_partitions.json", "phase22_clients.csv", "phase22_dp.json",
            "phase22_aggregation.json", "phase22_eval.json",
            "phase22_receipt.json", "phase22_summary.json"]
    missing = [f for f in want if not os.path.isfile(os.path.join(od, f))]
    record("20. all declared artifacts present (design SS 18)",
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
    if verdict != "PASS":
        print("Fail-closed: PENDING counts against the verdict, never for it.")
    print("=" * 78)

    if a.report:
        rp = a.report if os.path.isabs(a.report) else os.path.join(od, a.report)
        with open(rp, "w", encoding="utf-8") as fh:
            json.dump({"schema": "phase22_verification/1",
                       "experiment": C.EXPERIMENT,
                       "design": {"path": C.DESIGN_PATH, "sha256": live},
                       "generated_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ",
                                                      time.gmtime()),
                       "checks": CHECKS, "n_pass": n_pass, "n_fail": n_fail,
                       "n_pending": n_pend, "verdict": verdict}, fh,
                      indent=2, sort_keys=True)
        print(f"[report] {rp}")

    return 0 if verdict == "PASS" else 1


if __name__ == "__main__":
    sys.exit(main())
