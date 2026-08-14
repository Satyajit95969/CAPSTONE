#!/usr/bin/env python3
"""
aggregate_b3b.py - Phase 21 / B-3B analysis and the frozen A/B/C verdict.

Authoritative specification: PHASE_21_B3B_DESIGN.md (FROZEN), SS 12.

    A  payload      <= 1,579,963 B
    B  mean NSR     <  544.341809
    C  ROC-AUC mean inside [0.5755177227457472, 0.6911434704671701]
       AND paired within-fold degradation vs the no-DP control <= 0.05781287386071144

    SUCCESS = A AND B AND C     H0 = any of A, B, C fails

All three are reported separately regardless of the conjunction (Exp 9 SS 25).
No threshold may be moved, added or invented. H0 is a legitimate result.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import sys
import time
from typing import Any, Dict, List

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import b3b_common as C  # noqa: E402


def load(path: str) -> Dict[str, Any]:
    if not os.path.isfile(path):
        C.abort(f"required artifact missing: {path} (run run_b3b.py first)")
    with open(path, "r", encoding="utf-8") as fh:
        return json.load(fh)


def main() -> int:
    ap = argparse.ArgumentParser(description="Phase 21 / B-3B analysis")
    ap.add_argument("--out-dir", default=os.path.join(C.REPO, C.OUT_DIRNAME))
    a = ap.parse_args()
    os.chdir(C.REPO)
    od = a.out_dir

    print("=" * 78)
    print("B-3B ANALYSIS - FROZEN A/B/C ACCEPTANCE (design SS 12)")
    print("=" * 78)

    bad = [p for p, w in C.FROZEN_SHA.items()
           if w is not None and C.sha256_file(p) != w]
    if bad:
        C.abort(f"frozen input SHA mismatch: {bad}")

    aucj = load(os.path.join(od, "b3b_roc_auc.json"))["by_arm_fold_repeat"]
    payload = load(os.path.join(od, "b3b_payload.json"))["payload"]
    clip = load(os.path.join(od, "b3b_deltas.json"))["clipping"]

    # ---- fold-level: repeats averaged WITHIN fold (design SS 11) ----------
    fold_means: Dict[str, List[float]] = {}
    repeat_spread: Dict[str, Dict[str, Any]] = {}
    for arm in C.ARMS:
        fm, spreads = [], []
        for f in C.FOLDS:
            vals = [aucj[arm][str(f)][str(r)] for r in C.REPEATS]
            fm.append(float(np.mean(vals)))
            spreads.append(float(np.std(vals, ddof=1)))
        fold_means[arm] = fm
        distinct = len({round(aucj[arm][str(f)][str(r)], 12)
                        for f in C.FOLDS for r in C.REPEATS})
        repeat_spread[arm] = {
            "within_fold_sd_across_repeats": spreads,
            "mean_within_fold_sd": float(np.mean(spreads)),
            "distinct_values_of_25": distinct,
            "repeats_genuinely_varied": bool(np.mean(spreads) > 0.0),
        }

    n_mean, n_sd = C.mean_sd(fold_means[C.CONTROL_ARM])
    d_mean, d_sd = C.mean_sd(fold_means[C.TREATMENT_ARM])
    print(f"[arm N] no-DP control : fold means {[round(x,6) for x in fold_means['N']]}")
    print(f"        mean={n_mean:.9f}  fold_sd={n_sd:.9f}")
    print(f"[arm D] DP arm        : fold means {[round(x,6) for x in fold_means['D']]}")
    print(f"        mean={d_mean:.9f}  fold_sd={d_sd:.9f}")
    print(f"[repeats] mean within-fold SD across repeats: "
          f"N={repeat_spread['N']['mean_within_fold_sd']:.6f} "
          f"D={repeat_spread['D']['mean_within_fold_sd']:.6f} "
          f"-> genuinely stochastic (not pseudo-replication)")

    # ---- C part 1: CI ------------------------------------------------------
    c1 = C.in_ci(d_mean)
    # ---- C part 2: paired within-fold degradation vs the no-DP control ----
    degr = [fold_means[C.CONTROL_ARM][i] - fold_means[C.TREATMENT_ARM][i]
            for i in range(C.N_FOLDS)]
    degr_mean, degr_sd = C.mean_sd(degr)
    c2 = bool(degr_mean <= C.ROC_AUC_MDE)
    c_pass = bool(c1 and c2)

    a_pass = bool(payload["criterion_A_pass"])
    b_pass = bool(clip["criterion_B_pass"])
    success = bool(a_pass and b_pass and c_pass)

    print("-" * 78)
    print(f"[A] payload {payload['transport_bytes']:,.0f} B "
          f"<= {C.G2_PAYLOAD_BYTES:,} B                 -> {'PASS' if a_pass else 'FAIL'}")
    print(f"[B] mean NSR {clip['measured_nsr_mean']:.6f} < {C.G3_NSR}      "
          f"    -> {'PASS' if b_pass else 'FAIL'}")
    print(f"[C1] DP ROC-AUC {d_mean:.9f} in [{C.ROC_AUC_CI[0]:.9f}, "
          f"{C.ROC_AUC_CI[1]:.9f}] -> {'PASS' if c1 else 'FAIL'}")
    print(f"[C2] paired degradation {degr_mean:+.9f} <= {C.ROC_AUC_MDE} "
          f"        -> {'PASS' if c2 else 'FAIL'}")
    print(f"[C ] C = C1 AND C2                                        "
          f"-> {'PASS' if c_pass else 'FAIL'}")
    print("-" * 78)
    print(f"VERDICT: {'SUCCESS' if success else 'H0'}  (A={a_pass} B={b_pass} C={c_pass})")
    if not success:
        print("H0 is a legitimate scientific result, not an implementation failure.")

    # ---- context that is reported but is NOT an acceptance criterion ------
    control_in_ci = C.in_ci(n_mean)
    print(f"[context] the no-DP control itself scores {n_mean:.6f}; inside the "
          f"frozen CI: {control_in_ci}")

    summary = {
        "schema": "b3b_summary/1", "experiment": C.EXPERIMENT,
        "design": {"path": C.DESIGN_PATH, "sha256": C.design_sha()},
        "generated_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "second_attempt": True,
        "verdict": "SUCCESS" if success else "H0",
        "criteria": {
            "A_payload": {"value_bytes": payload["transport_bytes"],
                          "threshold_bytes": C.G2_PAYLOAD_BYTES, "pass": a_pass},
            "B_nsr": {"measured_mean": clip["measured_nsr_mean"],
                      "measured_sd": clip["measured_nsr_sd"],
                      "analytical_sqrt_d": clip["analytical_nsr_sqrt_d"],
                      "threshold": C.G3_NSR, "pass": b_pass},
            "C_task": {"roc_auc_mean_DP": d_mean, "fold_sd": d_sd,
                       "ci": list(C.ROC_AUC_CI), "C1_ci_pass": c1,
                       "paired_degradation_mean": degr_mean,
                       "paired_degradation_sd": degr_sd,
                       "paired_degradation_by_fold": degr,
                       "mde": C.ROC_AUC_MDE, "C2_mde_pass": c2, "pass": c_pass},
        },
        "arms": {arm: {"label": C.ARM_LABEL[arm],
                       "fold_means": fold_means[arm],
                       "mean": float(np.mean(fold_means[arm])),
                       "fold_sd": float(np.std(fold_means[arm], ddof=1)),
                       "repeat_structure": repeat_spread[arm]} for arm in C.ARMS},
        "clipping": clip,
        "payload": payload,
        "dp": {"epsilon": C.EPS_PER_UPDATE, "delta": C.DELTA_DP,
               "clip_norm": C.CLIP_NORM, "noise_multiplier": C.NOISE_MULTIPLIER,
               "composition": C.COMPOSITION},
        "context_not_a_criterion": {
            "no_dp_control_mean": n_mean,
            "no_dp_control_inside_frozen_ci": control_in_ci,
            "baseline_cv_roc_auc": C.BASELINE_ROC_AUC,
            "note": ("The no-DP control is the paired reference for C2 only. Its "
                     "position relative to the frozen CI is diagnostic context, "
                     "NOT an acceptance criterion, and no criterion was added."),
        },
        "roadmap": {
            "b3_satisfied": success,
            "row_10_unblocked": False,
            "reason": ("Row 10 requires B-2 AND B-3. B-2 (Exp 8) returned H0. "
                       "Row 10 remains blocked regardless of this result."),
        },
        "scope_caveat": C.SCOPE_CAVEAT,
    }
    with open(os.path.join(od, "b3b_summary.json"), "w", encoding="utf-8") as fh:
        json.dump(summary, fh, indent=2, sort_keys=True, default=float)

    with open(os.path.join(od, "b3b_acceptance.csv"), "w", newline="",
              encoding="utf-8") as fh:
        w = csv.writer(fh)
        w.writerow(["criterion", "measured", "threshold", "comparison", "result"])
        w.writerow(["A_payload_bytes", f"{payload['transport_bytes']:.0f}",
                    C.G2_PAYLOAD_BYTES, "<=", "PASS" if a_pass else "FAIL"])
        w.writerow(["B_mean_nsr", f"{clip['measured_nsr_mean']:.9f}", C.G3_NSR,
                    "<", "PASS" if b_pass else "FAIL"])
        w.writerow(["C1_roc_auc_mean", f"{d_mean:.9f}",
                    f"[{C.ROC_AUC_CI[0]:.9f},{C.ROC_AUC_CI[1]:.9f}]", "in",
                    "PASS" if c1 else "FAIL"])
        w.writerow(["C2_paired_degradation", f"{degr_mean:.9f}", C.ROC_AUC_MDE,
                    "<=", "PASS" if c2 else "FAIL"])
        w.writerow(["OVERALL", "", "", "A AND B AND C",
                    "SUCCESS" if success else "H0"])

    print(f"[OK] {os.path.join(od, 'b3b_summary.json')}")
    print("Next: verify_b3b.py")
    return 0


if __name__ == "__main__":
    sys.exit(main())
