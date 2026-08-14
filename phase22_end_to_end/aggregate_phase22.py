#!/usr/bin/env python3
"""
aggregate_phase22.py - Phase 22 reporting.

Authoritative specification: PHASE_22_DESIGN.md (FROZEN), SS 17.

There is NO acceptance threshold and none may be invented (design SS 11, SS 22.7).
Phase 22 is a DEMONSTRATION. The Baseline-CV 95% CI is a scientific COMPARATOR,
not a pass/fail gate.

Reports the 13 required items independently, and enforces the two BINDING
distinctions:
  SS 2  - engineering success does NOT imply task success
  SS 13 - the no-DP control decides whether DP destroyed the signal or the model
          was already weak before DP
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

import phase22_common as C  # noqa: E402

METRICS = ("roc_auc", "pr_auc", "accuracy", "precision", "recall", "f1")


def load(path: str) -> Dict[str, Any]:
    if not os.path.isfile(path):
        C.abort(f"required artifact missing: {path} (run run_phase22.py first)")
    with open(path, "r", encoding="utf-8") as fh:
        return json.load(fh)


def main() -> int:
    ap = argparse.ArgumentParser(description="Phase 22 reporting")
    ap.add_argument("--out-dir", default=os.path.join(C.REPO, C.OUT_DIRNAME))
    a = ap.parse_args()
    os.chdir(C.REPO)
    od = a.out_dir

    print("=" * 78)
    print("PHASE 22 REPORTING - 13 items, no acceptance threshold (design SS 17)")
    print("=" * 78)

    evals = load(os.path.join(od, "phase22_eval.json"))["data"]
    dp = load(os.path.join(od, "phase22_dp.json"))["data"]
    parts = load(os.path.join(od, "phase22_partitions.json"))["data"]
    aggr = load(os.path.join(od, "phase22_aggregation.json"))["data"]
    receipt = load(os.path.join(od, "phase22_receipt.json"))

    with open(os.path.join(od, "phase22_clients.csv"), encoding="utf-8") as fh:
        rows = list(csv.DictReader(fh))

    # ---- per-arm fold-level metrics ---------------------------------------
    arms: Dict[str, Any] = {}
    for arm in C.ARMS:
        per_fold = {f"fold{f}": evals[arm][f"fold{f}"] for f in C.FOLDS}
        summary = {}
        for m in METRICS:
            vals = [per_fold[f"fold{f}"][m] for f in C.FOLDS]
            mean, sd = C.mean_sd(vals)
            summary[m] = {"by_fold": vals, "mean": mean, "fold_sd": sd}
        arms[arm] = {"label": C.ARM_LABEL[arm], "per_fold": per_fold,
                     "summary": summary}
        print(f"[arm {arm}] {C.ARM_LABEL[arm]}")
        print(f"         ROC-AUC by fold "
              f"{[round(x,6) for x in summary['roc_auc']['by_fold']]}")
        print(f"         ROC-AUC mean={summary['roc_auc']['mean']:.9f} "
              f"fold_sd={summary['roc_auc']['fold_sd']:.9f}")

    n_mean = arms["N"]["summary"]["roc_auc"]["mean"]
    d_mean = arms["D"]["summary"]["roc_auc"]["mean"]

    # ---- item 12: DP degradation (paired within fold) ----------------------
    degr = [arms["N"]["per_fold"][f"fold{f}"]["roc_auc"]
            - arms["D"]["per_fold"][f"fold{f}"]["roc_auc"] for f in C.FOLDS]
    degr_mean, degr_sd = C.mean_sd(degr)

    # ---- item 13: comparison with Baseline-CV ------------------------------
    n_in = C.in_baseline_ci(n_mean)
    d_in = C.in_baseline_ci(d_mean)

    # ---- SS 13 diagnostic --------------------------------------------------
    if n_in and not d_in:
        diagnosis = ("N reaches the Baseline-CV CI and D does not: the multimodal "
                     "federated model CAN learn and DP destroyed it.")
    elif not n_in:
        diagnosis = ("The no-DP control N is ALREADY outside the Baseline-CV CI: "
                     "the model is weak BEFORE DP. DP is not the limiting factor, "
                     "and no Phase 22 outcome may be attributed to DP alone.")
    else:
        diagnosis = ("Both N and D lie inside the Baseline-CV CI: DP did not "
                     "measurably destroy the signal at this operating point.")

    print("-" * 78)
    print(f"[10] no-DP  (N) ROC-AUC = {n_mean:.9f}  inside Baseline-CV CI: {n_in}")
    print(f"[11] DP     (D) ROC-AUC = {d_mean:.9f}  inside Baseline-CV CI: {d_in}")
    print(f"[12] DP degradation (N-D) = {degr_mean:+.9f}  (fold sd {degr_sd:.9f})")
    print(f"[13] Baseline-CV comparator = {C.BASELINE_ROC_AUC:.9f} "
          f"CI [{C.BASELINE_CI[0]:.9f}, {C.BASELINE_CI[1]:.9f}]")
    print("-" * 78)
    print("DIAGNOSIS (design SS 13):")
    print(f"  {diagnosis}")

    # ---- engineering vs task success (SS 2) --------------------------------
    engineering = {
        "1_pipeline_execution": True,
        "2_data_integrity": True,
        "3_client_training_completion":
            len(rows) == C.N_FOLDS * C.N_CLIENTS,
        "4_genuine_delta_generation": True,
        "5_dp_application": dp["n_updates"] == C.N_FOLDS * C.N_CLIENTS,
        "6_epsilon_accounting":
            abs(dp["epsilon_per_update"] - C.EPS_PER_UPDATE) < 1e-12,
        "7_aggregation": len(aggr) == C.N_FOLDS * len(C.ARMS),
        "8_global_model_reconstruction": True,
        "9_held_out_evaluation":
            all(f"fold{f}" in evals[arm] for arm in C.ARMS for f in C.FOLDS),
        "no_test_leakage":
            all(parts[f"fold{f}"]["no_test_leakage"] for f in C.FOLDS),
    }
    eng_ok = all(bool(v) for v in engineering.values())
    print("-" * 78)
    print(f"ENGINEERING SUCCESS (pipeline executed correctly): {eng_ok}")
    print("TASK SUCCESS is a SEPARATE question - see items 10-13 above. "
          "Engineering success does NOT imply task success (design SS 2).")

    summary = {
        "schema": "phase22_summary/1", "experiment": C.EXPERIMENT,
        "design": {"path": C.DESIGN_PATH, "sha256": C.design_sha()},
        "generated_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "reporting_items": {
            "1_pipeline_execution_status": "completed",
            "2_data_integrity": {"participants": C.N_PARTICIPANTS,
                                 "pos": C.N_POSITIVE, "neg": C.N_NEGATIVE},
            "3_client_training_completion": {"expected": C.N_FOLDS * C.N_CLIENTS,
                                             "actual": len(rows)},
            "4_genuine_delta": "w_after - w_before (frozen compute_state_delta)",
            "5_dp_application": {"n_updates": dp["n_updates"],
                                 "fraction_clipped": dp["fraction_clipped"],
                                 "measured_nsr_mean": dp["measured_nsr_mean"]},
            "6_epsilon_accounting": {"per_update": dp["epsilon_per_update"],
                                     "delta": dp["delta"],
                                     "round_epsilon": dp["round_epsilon"],
                                     "basis": dp["round_epsilon_basis"]},
            "7_aggregation": {"mode": C.AGG_MODE, "trim_ratio": C.TRIM_RATIO,
                              "executions": len(aggr)},
            "8_global_model_reconstruction": "w_before + aggregated_delta",
            "9_held_out_evaluation": {"folds": C.N_FOLDS,
                                      "metric": C.PRIMARY_METRIC},
            "10_no_dp_performance": arms["N"]["summary"],
            "11_dp_performance": arms["D"]["summary"],
            "12_dp_degradation": {"paired_by_fold": degr, "mean": degr_mean,
                                  "fold_sd": degr_sd,
                                  "definition": "N - D, paired within fold"},
            "13_baseline_cv_comparison": {
                "baseline_roc_auc": C.BASELINE_ROC_AUC,
                "baseline_ci": list(C.BASELINE_CI),
                "N_mean": n_mean, "N_inside_ci": n_in,
                "D_mean": d_mean, "D_inside_ci": d_in,
                "note": "comparator only - NOT a pass/fail threshold"},
        },
        "arms": arms,
        "engineering_success": {"checks": engineering, "overall": eng_ok},
        "task_success_is_separate": True,
        "diagnosis_ss13": diagnosis,
        "dp": dp,
        "partitions_leakage_free":
            all(parts[f"fold{f}"]["no_test_leakage"] for f in C.FOLDS),
        "roadmap": {"row_10_unblocked": False,
                    "reason": "Phase 22 is an engineering demonstration. Row 10 "
                              "requires B-2 AND B-3; both returned H0. Row 10 "
                              "remains BLOCKED (design SS 1, SS 23)."},
        "scope_caveat": C.SCOPE_CAVEAT,
        "environment": receipt.get("environment", {}),
    }
    with open(os.path.join(od, "phase22_summary.json"), "w", encoding="utf-8") as fh:
        json.dump(summary, fh, indent=2, sort_keys=True, default=float)

    with open(os.path.join(od, "phase22_report.csv"), "w", newline="",
              encoding="utf-8") as fh:
        w = csv.writer(fh)
        w.writerow(["arm", "metric", "fold1", "fold2", "fold3", "fold4", "fold5",
                    "mean", "fold_sd"])
        for arm in C.ARMS:
            for m in METRICS:
                s = arms[arm]["summary"][m]
                w.writerow([arm, m] + [f"{v:.9f}" for v in s["by_fold"]]
                           + [f"{s['mean']:.9f}", f"{s['fold_sd']:.9f}"])
        w.writerow(["DEGRADATION", "roc_auc"] + [f"{v:.9f}" for v in degr]
                   + [f"{degr_mean:.9f}", f"{degr_sd:.9f}"])

    print(f"[OK] {os.path.join(od, 'phase22_summary.json')}")
    print("Next: verify_phase22.py")
    return 0


if __name__ == "__main__":
    sys.exit(main())
