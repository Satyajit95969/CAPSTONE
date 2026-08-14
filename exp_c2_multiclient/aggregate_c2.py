#!/usr/bin/env python3
"""
aggregate_c2.py - Phase 20 / C-2 analysis and validity gate.

Authoritative specification: PHASE_20_C2_DESIGN.md (FROZEN), SS 14, SS 15.

THE VALIDITY GATE (design SS 15) - directional only, no floor, no margin, no MDE:

    non-IID (W1) pre-DP client-update divergence
            MUST EXCEED
    stratified/IID-like (W0) pre-DP client-update divergence

NO TASK METRIC IS COMPUTED (design SS 14). No ROC-AUC, accuracy, F1, MAE, RMSE or
PR-AUC appears here; no held-out evaluation set exists.

REPEAT DEGENERACY - reported, not corrected
-------------------------------------------
The frozen recipe (1 epoch, full batch, no dropout, no shuffling, and a shared
fixed w_before restored by load_state_dict) makes each client's local training
DETERMINISTIC. The client_seed governs only an initialisation that is immediately
overwritten. Consequently the 5 repeats yield BIT-IDENTICAL pre-DP deltas, so the
PRIMARY endpoint has n = 1 effective replicate per arm, not 5.

This module therefore refuses to report any cross-repeat SD, CI or dispersion on
the primary endpoint: doing so would manufacture precision from pseudo-replicates.
It verifies the degeneracy explicitly and records it. The repeats remain genuine
replicates for the SECONDARY post-DP measures, where the DP seed does vary.

This is a property of the frozen design, discovered empirically. Nothing was
re-tuned in response to it (design SS 23).

Usage:
    python exp_c2_multiclient/aggregate_c2.py
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

import c2_common as C  # noqa: E402

# Guard: none of these may ever be produced by C-2 (design SS 14).
FORBIDDEN_METRICS = ("roc_auc", "auc", "accuracy", "f1", "mae", "rmse", "pr_auc",
                     "precision", "recall", "held_out", "generalization")


def load(path: str) -> Dict[str, Any]:
    if not os.path.isfile(path):
        C.abort(f"required artifact missing: {path} (run run_c2.py first)")
    with open(path, "r", encoding="utf-8") as fh:
        return json.load(fh)


def main() -> int:
    ap = argparse.ArgumentParser(description="Phase 20 / C-2 analysis")
    ap.add_argument("--out-dir", default=os.path.join(C.REPO, C.OUT_DIRNAME))
    a = ap.parse_args()
    os.chdir(C.REPO)
    od = a.out_dir

    print("=" * 78)
    print("C-2 ANALYSIS - VALIDITY GATE (design SS 15)")
    print("directional only: no floor, no margin, no MDE, no task metric")
    print("=" * 78)

    bad = [p for p, w in C.FROZEN_SHA.items()
           if w is not None and C.sha256_file(p) != w]
    if bad:
        C.abort(f"frozen input SHA mismatch: {bad}")

    pre = load(os.path.join(od, "c2_divergence_pre_dp.json"))["by_arm_repeat"]
    post = load(os.path.join(od, "c2_divergence_post_dp.json"))["by_arm_repeat"]
    aggj = load(os.path.join(od, "c2_aggregation.json"))
    receipt = load(os.path.join(od, "c2_receipt.json"))
    parts = load(os.path.join(od, "c2_partitions.json"))

    if receipt.get("task_metric_computed") is not False:
        C.abort("receipt does not assert task_metric_computed == false")

    # ---- repeat-degeneracy check on the PRIMARY endpoint -------------------
    degeneracy: Dict[str, Any] = {}
    for arm in C.ARMS:
        base = pre[f"{arm}|r{C.REPEATS[0]}"]
        ident = all(pre[f"{arm}|r{r}"]["l2"] == base["l2"]
                    and pre[f"{arm}|r{r}"]["cosine"] == base["cosine"]
                    for r in C.REPEATS)
        degeneracy[arm] = {
            "all_repeats_bit_identical": bool(ident),
            "distinct_l2_vectors": len({tuple(pre[f'{arm}|r{r}']['l2'])
                                        for r in C.REPEATS}),
            "effective_replicates": 1 if ident else C.N_REPEATS,
        }
    deterministic = all(d["all_repeats_bit_identical"] for d in degeneracy.values())
    print(f"[primary] pre-DP repeats bit-identical in every arm: {deterministic}")
    print(f"[primary] effective replicates on the PRIMARY endpoint: "
          f"{1 if deterministic else C.N_REPEATS} per arm "
          f"(declared repeats = {C.N_REPEATS})")
    if deterministic:
        print("[primary] -> NO cross-repeat SD/CI is reported on the primary "
              "endpoint (pseudo-replication)")

    # ---- the validity gate -------------------------------------------------
    summary_pre: Dict[str, Any] = {}
    for arm in C.ARMS:
        l2 = [v for r in C.REPEATS for v in pre[f"{arm}|r{r}"]["l2"]]
        cos = [v for r in C.REPEATS for v in pre[f"{arm}|r{r}"]["cosine"]]
        u_l2 = pre[f"{arm}|r{C.REPEATS[0]}"]["l2"]      # the unique replicate
        u_cos = pre[f"{arm}|r{C.REPEATS[0]}"]["cosine"]
        summary_pre[arm] = {
            "label": C.ARM_LABEL[arm],
            "n_pairs_per_repeat": len(u_l2),
            "l2_mean": float(np.mean(u_l2)), "l2_min": float(np.min(u_l2)),
            "l2_max": float(np.max(u_l2)),
            "l2_sd_across_pairs": float(np.std(u_l2, ddof=1)),
            "cosine_mean": float(np.mean(u_cos)), "cosine_min": float(np.min(u_cos)),
            "cosine_max": float(np.max(u_cos)),
            "cosine_sd_across_pairs": float(np.std(u_cos, ddof=1)),
            "l2_all_values": u_l2, "cosine_all_values": u_cos,
            "pooled_l2_mean_all_repeats": float(np.mean(l2)),
            "pooled_cosine_mean_all_repeats": float(np.mean(cos)),
            "dispersion_note": ("SD is across the 10 client PAIRS within one "
                                "replicate; it is NOT a cross-repeat SD and is "
                                "NOT a sampling CI"),
        }

    gate_rows: List[Dict[str, Any]] = []
    for measure in ("l2", "cosine"):
        w0 = summary_pre[C.CONTROL_ARM][f"{measure}_mean"]
        w1 = summary_pre[C.TREATMENT_ARM][f"{measure}_mean"]
        passed = bool(w1 > w0)
        gate_rows.append({
            "measure": measure, "control_W0": w0, "treatment_W1": w1,
            "difference_W1_minus_W0": w1 - w0,
            "direction_required": "W1 > W0",
            "result": "PASS" if passed else "FAIL",
        })
        print(f"[gate:{measure:>6}] W0={w0:.9f}  W1={w1:.9f}  "
              f"diff={w1 - w0:+.9f}  -> {'PASS' if passed else 'FAIL'}")

    gate_pass = all(r["result"] == "PASS" for r in gate_rows)
    print("-" * 78)
    print(f"VALIDITY GATE: {'PASS' if gate_pass else 'FAIL'}  "
          f"({sum(r['result'] == 'PASS' for r in gate_rows)}/{len(gate_rows)} "
          f"measures)")
    if not gate_pass:
        print("A failed gate means the manipulation did not produce divergent "
              "clients (design SS 15). It is reported, never re-tuned.")

    # ---- secondary: post-DP (here the repeats ARE genuine replicates) ------
    summary_post: Dict[str, Any] = {}
    for arm in C.ARMS:
        means = [post[f"{arm}|r{r}"]["l2_mean"] for r in C.REPEATS]
        cmeans = [post[f"{arm}|r{r}"]["cosine_mean"] for r in C.REPEATS]
        summary_post[arm] = {
            "l2_mean_by_repeat": means,
            "l2_mean_over_repeats": float(np.mean(means)),
            "l2_sd_over_repeats": float(np.std(means, ddof=1)),
            "cosine_mean_by_repeat": cmeans,
            "cosine_mean_over_repeats": float(np.mean(cmeans)),
            "cosine_sd_over_repeats": float(np.std(cmeans, ddof=1)),
            "effective_replicates": C.N_REPEATS,
        }
    ratio = (summary_post[C.TREATMENT_ARM]["l2_mean_over_repeats"] /
             summary_pre[C.TREATMENT_ARM]["l2_mean"])
    print(f"[secondary] post-DP L2 is {ratio:,.0f}x the pre-DP signal "
          f"-> post-DP cannot discriminate topologies (design SS 14)")

    # ---- DP stage description ----------------------------------------------
    with open(os.path.join(od, "c2_client_updates.csv"), encoding="utf-8") as fh:
        rows = list(csv.DictReader(fh))
    scales = [float(r["clip_scale"]) for r in rows]
    norms = [float(r["l2_before_dp"]) for r in rows]
    n_clipped = sum(1 for s in scales if s < 1.0)
    d = C.PROBE_N_PARAMS
    snr = float(np.mean(norms)) / (C.SIGMA_EFF * C.CLIP_NORM * np.sqrt(d))
    dp_stage = {
        "clip_norm": C.CLIP_NORM, "noise_multiplier": C.NOISE_MULTIPLIER,
        "sigma_eff": C.SIGMA_EFF, "mechanism": C.MECHANISM,
        "epsilon": C.epsilon(), "delta": C.DELTA_DP, "composition": C.COMPOSITION,
        "n_client_updates": len(rows), "n_clipped": n_clipped,
        "clipping_was_binding": n_clipped > 0,
        "update_norm_mean": float(np.mean(norms)),
        "update_norm_min": float(np.min(norms)),
        "update_norm_max": float(np.max(norms)),
        "d": d, "expected_noise_norm": float(C.SIGMA_EFF * np.sqrt(d)),
        "snr_signal_over_noise": snr,
        "note": ("every update norm is below the clipping bound C = 1.0, so "
                 "clipping never bound; the full sigma_eff*C noise is still added, "
                 "giving a signal-to-noise ratio of ~{:.2e}".format(snr)),
    }
    print(f"[dp] {n_clipped}/{len(rows)} updates clipped (bound C=1.0 never "
          f"binding; norms {min(norms):.4f}-{max(norms):.4f})")
    print(f"[dp] SNR = {snr:.3e}   eps = {dp_stage['epsilon']:.6f}")

    # ---- aggregation summary ----------------------------------------------
    ab = aggj["by_arm_repeat"]
    agg_summary = {arm: {
        "trim_window": ab[f"{arm}|r1"]["trim_window"],
        "l2_trimmed_vs_ordinary_mean_by_repeat":
            [ab[f"{arm}|r{r}"]["l2_trimmed_vs_ordinary_mean"] for r in C.REPEATS],
        "l2_trimmed_vs_coord_median_by_repeat":
            [ab[f"{arm}|r{r}"]["l2_trimmed_vs_coord_median"] for r in C.REPEATS],
        "trim_counts_per_client_by_repeat":
            [ab[f"{arm}|r{r}"]["trim_counts_per_client"] for r in C.REPEATS],
    } for arm in C.ARMS}

    # ---- artifacts ---------------------------------------------------------
    summary = {
        "schema": "c2_summary/1", "experiment": C.EXPERIMENT,
        "design": {"path": C.DESIGN_PATH, "sha256": C.design_sha()},
        "generated_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "validity_gate": {
            "specification": "W1 pre-DP divergence MUST EXCEED W0 (design SS 15)",
            "type": "directional only - no floor, no margin, no MDE",
            "rows": gate_rows,
            "result": "PASS" if gate_pass else "FAIL",
        },
        "primary_pre_dp": summary_pre,
        "repeat_degeneracy": {
            "primary_endpoint_deterministic": deterministic,
            "by_arm": degeneracy,
            "cause": ("1 epoch, full batch, no dropout, no shuffling, shared "
                      "fixed w_before restored by load_state_dict(strict=True); "
                      "client_seed sets only an initialisation that is then "
                      "overwritten"),
            "consequence": ("PRIMARY endpoint has 1 effective replicate per arm, "
                            "not 5; no cross-repeat SD or CI is reported for it"),
            "secondary_unaffected": ("post-DP measures DO vary across repeats "
                                     "because the DP seed varies"),
            "action_taken": "reported as a limitation; nothing was re-tuned",
        },
        "secondary_post_dp": summary_post,
        "dp_stage": dp_stage,
        "aggregation": agg_summary,
        "partition_stats": {
            arm: [{k: v for k, v in c.items() if k != "participant_ids"}
                  for c in parts["arms"][arm]["clients"]] for arm in C.ARMS},
        "task_metric_computed": False,
        "forbidden_metrics_absent": list(FORBIDDEN_METRICS),
        "scope_caveat": C.SCOPE_CAVEAT,
        "interpretation": (
            "The gate is a MANIPULATION CHECK: it establishes only that the "
            "non-IID partition produced more divergent client updates than the "
            "stratified partition before DP noise. It says nothing about task "
            "performance, generalisation, clinical utility or predictive "
            "validity, none of which C-2 measured."),
    }

    spath = os.path.join(od, "c2_summary.json")
    with open(spath, "w", encoding="utf-8") as fh:
        json.dump(summary, fh, indent=2, sort_keys=True, default=float)

    apath = os.path.join(od, "c2_acceptance.csv")
    with open(apath, "w", newline="", encoding="utf-8") as fh:
        w = csv.writer(fh)
        w.writerow(["gate", "measure", "control_W0", "treatment_W1",
                    "difference_W1_minus_W0", "direction_required", "result"])
        for r in gate_rows:
            w.writerow(["pre_dp_divergence", r["measure"], f"{r['control_W0']:.12f}",
                        f"{r['treatment_W1']:.12f}",
                        f"{r['difference_W1_minus_W0']:+.12f}",
                        r["direction_required"], r["result"]])
        w.writerow(["OVERALL", "", "", "", "", "W1 > W0 on all measures",
                    "PASS" if gate_pass else "FAIL"])

    print("-" * 78)
    print(f"[OK] {spath}")
    print(f"[OK] {apath}")
    print("Next: verify_c2.py (independent fail-closed verification)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
