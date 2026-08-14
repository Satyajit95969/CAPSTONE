#!/usr/bin/env python3
"""
aggregate_exp9.py - Phase 18 / Experiment 9 aggregation and acceptance.

Authoritative specification: PHASE_18_EXP9_DESIGN.md (FROZEN). This module
implements ONLY the frozen reporting rules (design SS 30) and the frozen
acceptance logic (design SS 15, SS 17, SS 22, SS 23, SS 24, SS 25). No criterion,
threshold or constant may be altered, including if K2 lands marginally on either
side of G2 (design SS 32.2).

FROZEN PER-ARM ACCEPTANCE - all three required (design SS 25):
    A   complete transport payload <= 1,579,963 B          (SS 17)
    B   measured mean SNR          <  544.341809           (SS 15)
    C   ROC-AUC fold-level mean inside [0.5755177227457472,
                                        0.6911434704671701] (SS 23)
        AND paired per-fold degradation vs K0 <= 0.05781287386071144 (SS 24)

A, B and C are computed and reported SEPARATELY FOR EVERY ARM regardless of the
conjunction (design SS 25): a partial pattern such as "K3 collapsed payload and
SNR but lost the ranking signal" is precisely the boundary measurement Exp 9
exists to produce. They are never collapsed into a single hidden number.

EXPERIMENT-LEVEL RESULT: the set of passing arms; the SMALLEST PASSING k answers
the design SS 4 question. If no arm passes, that is H0 - a legitimate scientific
result and never an implementation failure (design SS 25, SS 32.7).

THE K0 GATE IS MANDATORY (design SS 22). If K0's fold-level ROC-AUC mean falls
outside the frozen Baseline-CV CI, the task interpretation HARD ABORTS and no
compressed-arm task acceptance may be reported. It is never asserted, never
hard-coded, and Exp 7's A0 precedent does not discharge it.

Usage:
    python exp9_compact_update/aggregate_exp9.py
    python exp9_compact_update/aggregate_exp9.py --emit-schema   # schema, no result
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import sys
from typing import Any, Dict, List, Optional

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import exp9_common as C  # noqa: E402

PENDING = "PENDING"


def read_csv_rows(path: str) -> List[Dict[str, str]]:
    with open(path, "r", newline="", encoding="utf-8") as fh:
        return list(csv.DictReader(fh))


# ==========================================================================
# design SS 30.2 - per-arm SNR
# ==========================================================================
def snr_table(rows: List[Dict[str, str]]) -> Dict[str, Any]:
    out: Dict[str, Any] = {}
    for arm in C.ARMS:
        vals = [float(r["snr"]) for r in rows if r["arm"] == arm]
        if not vals:
            out[arm] = {"status": PENDING,
                        "reason": "no DP draws recorded for this arm"}
            continue
        if len(vals) != C.N_REPETITIONS:
            C.abort(f"{arm}: {len(vals)} SNR draws, expected {C.N_REPETITIONS} "
                    "(design SS 26). Refusing to aggregate a partial arm.")
        seeds = sorted(int(r["seed"]) for r in rows if r["arm"] == arm)
        if seeds != sorted(C.SEEDS):
            C.abort(f"{arm}: seeds {seeds} != frozen {list(C.SEEDS)} (design SS 26)")
        an = C.analytical_snr(C.K_VALUES[arm])
        m = C.mean(vals)
        out[arm] = {
            "status": "MEASURED",
            "k": C.K_VALUES[arm],
            "per_seed": {str(int(r["seed"])): float(r["snr"])
                         for r in rows if r["arm"] == arm},
            "mean": m, "sd": C.sd(vals), "min": min(vals), "max": max(vals),
            "analytical": an,                      # design SS 13, never an input
            "deviation_abs": m - an,
            "deviation_rel": (m - an) / an if an else float("nan"),
            "g3_threshold": C.G3_SNR_MAX,
            "criterion_B_pass": bool(m < C.G3_SNR_MAX),
        }
    return out


# ==========================================================================
# design SS 30.5 - privacy
# ==========================================================================
def privacy_table(rows: List[Dict[str, str]]) -> Dict[str, Any]:
    eps_declared = C.epsilon()
    out: Dict[str, Any] = {"recomputed_epsilon": eps_declared,
                           "declared_epsilon": C.EPS_PER_UPDATE_MAX,
                           "delta": C.DELTA_DP, "composition": C.COMPOSITION,
                           "eps_max_ceiling_only": C.EPS_MAX_CEILING, "arms": {}}
    if abs(eps_declared - C.EPS_PER_UPDATE_MAX) > 1e-12:
        C.abort(f"recomputed eps {eps_declared!r} != declared "
                f"{C.EPS_PER_UPDATE_MAX} (design SS 29: HARD ABORT)")
    for arm in C.ARMS:
        vals = sorted({float(r["epsilon"]) for r in rows if r["arm"] == arm})
        if not vals:
            out["arms"][arm] = {"status": PENDING}
            continue
        if len(vals) != 1 or abs(vals[0] - C.EPS_PER_UPDATE_MAX) > 1e-12:
            C.abort(f"{arm}: recorded eps {vals} != {C.EPS_PER_UPDATE_MAX} "
                    "(design SS 29: HARD ABORT)")
        cum = vals[0] * C.N_UPDATES_PER_ROUND
        if cum > C.EPS_CUMULATIVE_MAX + 1e-9:
            C.abort(f"{arm}: cumulative eps {cum} > {C.EPS_CUMULATIVE_MAX}")
        out["arms"][arm] = {"status": "MEASURED", "epsilon": vals[0],
                            "cumulative_epsilon": cum,
                            "cumulative_budget": C.EPS_CUMULATIVE_MAX,
                            "delta": C.DELTA_DP, "composition": C.COMPOSITION}
    return out


# ==========================================================================
# design SS 30.1 - per-arm payload (criterion A)
# ==========================================================================
def payload_table(payload: Optional[Dict[str, Any]]) -> Dict[str, Any]:
    out: Dict[str, Any] = {}
    arms = (payload or {}).get("arms", {})
    for arm in C.ARMS:
        p = arms.get(arm)
        if not p:
            out[arm] = {"status": PENDING,
                        "reason": "no transport payload measurement for this arm"}
            continue
        total = int(p["total_uploaded_bytes"])
        out[arm] = {
            "status": "MEASURED",
            "k": C.K_VALUES[arm],
            "raw_4k_bytes": int(p["raw_4k_bytes"]),      # diagnostic only (SS 16)
            "serialised_bytes": int(p["serialised_bytes"]),
            "encrypted_bytes": int(p["encrypted_bytes"]),
            "n_chunks": int(p["n_chunks"]),
            "total_uploaded_bytes": total,               # <- the G2 quantity
            "payload_hash": p["payload_hash"],
            "g2_threshold_bytes": C.G2_PAYLOAD_MAX_BYTES,
            "margin_bytes": C.G2_PAYLOAD_MAX_BYTES - total,
            "criterion_A_pass": bool(total <= C.G2_PAYLOAD_MAX_BYTES),
        }
    return out


# ==========================================================================
# design SS 22, SS 23, SS 24, SS 30.3 - task
# ==========================================================================
def fold_means(rows: List[Dict[str, str]], arm: str) -> Dict[int, float]:
    """Average the repeats WITHIN each fold (design SS 19)."""
    buckets: Dict[int, List[float]] = {}
    for r in rows:
        if r["arm"] != arm:
            continue
        v = float(r[C.TASK_METRIC])
        if math.isnan(v):
            C.abort(f"{arm} repeat {r['repeat']} fold {r['fold']}: "
                    f"{C.TASK_METRIC} is NaN; a degenerate fold cannot be averaged")
        buckets.setdefault(int(r["fold"]), []).append(v)
    return {f: C.mean(v) for f, v in sorted(buckets.items())}


def task_table(rows: List[Dict[str, str]]) -> Dict[str, Any]:
    if not rows:
        return {arm: {"status": PENDING,
                      "reason": "task half not executed (run_exp9_task.py on GPU)"}
                for arm in C.ARMS}

    per_arm_folds: Dict[str, Dict[int, float]] = {}
    out: Dict[str, Any] = {}
    for arm in C.ARMS:
        arm_rows = [r for r in rows if r["arm"] == arm]
        if not arm_rows:
            out[arm] = {"status": PENDING, "reason": "no fold-runs for this arm"}
            continue
        if len(arm_rows) != C.N_FOLDS * C.N_REPEATS:
            C.abort(f"{arm}: {len(arm_rows)} fold-runs, expected "
                    f"{C.N_FOLDS * C.N_REPEATS} (design SS 19). Refusing to "
                    "aggregate a partial arm.")
        fm = fold_means(arm_rows, arm)
        if sorted(fm) != list(range(C.N_FOLDS)) and len(fm) != C.N_FOLDS:
            C.abort(f"{arm}: folds {sorted(fm)} != {C.N_FOLDS} distinct folds")
        per_arm_folds[arm] = fm
        stats = C.fold_level_ci(list(fm.values()))
        lo, hi = C.ROC_AUC_CI
        out[arm] = {"status": "MEASURED", "k": C.K_VALUES[arm],
                    "n_fold_runs": len(arm_rows), "fold_means": fm,
                    **stats,
                    "baseline_ci": [lo, hi],
                    "inside_baseline_ci": bool(lo <= stats["mean"] <= hi)}

    # ---- design SS 22: the MANDATORY K0 equivalence gate --------------------
    k0 = out.get(C.CONTROL_ARM, {})
    if k0.get("status") != "MEASURED":
        for arm in C.ARMS:
            if out.get(arm, {}).get("status") == "MEASURED":
                out[arm]["criterion_C_pass"] = PENDING
                out[arm]["criterion_C_reason"] = (
                    "K0 not measured; SS 22 forbids reporting compressed-arm task "
                    "acceptance without the control")
        out["_k0_gate"] = {"status": PENDING,
                           "reason": "K0 arm not present in the task metrics"}
        return out

    lo, hi = C.ROC_AUC_CI
    gate_pass = bool(lo <= k0["mean"] <= hi)
    out["_k0_gate"] = {
        "status": "MEASURED",
        "k0_fold_level_mean": k0["mean"],
        "baseline_reference_mean": C.BASELINE_ROC_AUC_MEAN,
        "baseline_ci": [lo, hi],
        "passed": gate_pass,
        "precedent_note": ("Exp 7's A0 re-ran this recipe and reproduced "
                           "Baseline-CV, but that is PRECEDENT, not proof: the gate "
                           "is evaluated on this run's own K0 output (design SS 22)"),
    }
    if not gate_pass:
        # design SS 22 / SS 29: HARD ABORT of the task interpretation. The
        # measurement is retained and reported; the INTERPRETATION is voided.
        for arm in C.ARMS:
            if out.get(arm, {}).get("status") == "MEASURED":
                out[arm]["criterion_C_pass"] = "VOID"
                out[arm]["criterion_C_reason"] = (
                    "K0 failed the SS 22 equivalence gate; no compressed-arm task "
                    "acceptance may be reported")
        return out

    # ---- design SS 23 + SS 24: criterion C ----------------------------------
    k0_folds = per_arm_folds[C.CONTROL_ARM]
    for arm in C.ARMS:
        if out.get(arm, {}).get("status") != "MEASURED":
            continue
        fm = per_arm_folds[arm]
        # Paired WITHIN fold: fold i's arm against fold i's K0 (design SS 24).
        # Adverse direction = arm BELOW K0, so degradation = K0 - arm.
        deg = {f: k0_folds[f] - fm[f] for f in sorted(fm)}
        dstats = C.fold_level_ci(list(deg.values()))
        mde_pass = bool(dstats["mean"] <= C.ROC_AUC_MDE)
        ci_pass = bool(out[arm]["inside_baseline_ci"])
        out[arm].update({
            "paired_degradation_per_fold": deg,
            "paired_degradation_mean": dstats["mean"],
            "paired_degradation_sd": dstats["sd"],
            "paired_degradation_ci95": [dstats["ci95_lo"], dstats["ci95_hi"]],
            "mde_threshold": C.ROC_AUC_MDE,
            "criterion_C1_ci_pass": ci_pass,          # SS 23, reported separately
            "criterion_C2_mde_pass": mde_pass,        # SS 24, reported separately
            "criterion_C_pass": bool(ci_pass and mde_pass),
        })
    return out


# ==========================================================================
# design SS 25 - the A/B/C matrix and the conjunction
# ==========================================================================
def acceptance(payload: Dict[str, Any], snr: Dict[str, Any],
               task: Dict[str, Any]) -> Dict[str, Any]:
    matrix: Dict[str, Any] = {}
    for arm in C.ARMS:
        A = payload.get(arm, {}).get("criterion_A_pass", PENDING)
        B = snr.get(arm, {}).get("criterion_B_pass", PENDING)
        Cc = task.get(arm, {}).get("criterion_C_pass", PENDING)
        if A is True and B is True and Cc is True:
            overall = "PASS"
        elif PENDING in (A, B, Cc):
            overall = PENDING          # never PASS on missing evidence
        elif Cc == "VOID":
            overall = "VOID"
        else:
            overall = "FAIL"
        matrix[arm] = {"k": C.K_VALUES[arm], "A_payload": A, "B_snr": B,
                       "C_task": Cc, "overall": overall}

    passing = [a for a in C.ARMS if matrix[a]["overall"] == "PASS"]
    pending = [a for a in C.ARMS if matrix[a]["overall"] == PENDING]
    smallest = min(passing, key=lambda a: C.K_VALUES[a]) if passing else None
    if pending:
        verdict = (f"INCOMPLETE - {len(pending)} arm(s) PENDING: {pending}. "
                   "No experiment-level verdict may be issued on partial evidence.")
    elif passing:
        verdict = (f"PASS - passing arms {passing}; smallest passing k = "
                   f"{C.K_VALUES[smallest]} ({smallest}), which answers the "
                   "design SS 4 question")
    elif any(matrix[a]["overall"] == "VOID" for a in C.ARMS):
        verdict = ("VOID - K0 failed the SS 22 equivalence gate; the task "
                   "interpretation is aborted, not reinterpreted")
    else:
        verdict = ("H0 - no arm satisfies A and B and C. This is a LEGITIMATE "
                   "SCIENTIFIC RESULT, not an implementation failure (design "
                   "SS 25, SS 32.7). It establishes only that the "
                   "publicly-specified family defined by O fails - NOT that no "
                   "compact DAIC-derived object survives DP (design SS 4, SS 34).")
    return {"matrix": matrix, "passing_arms": passing, "pending_arms": pending,
            "smallest_passing_arm": smallest,
            "smallest_passing_k": C.K_VALUES[smallest] if smallest else None,
            "verdict": verdict}


# ==========================================================================
# Schema (design SS 12 of the implementation brief): pending fields, never
# invented results.
# ==========================================================================
def schema() -> Dict[str, Any]:
    return {
        "schema": "exp9_summary/1",
        "experiment": C.EXPERIMENT,
        "design": {"path": C.DESIGN_PATH, "sha256": C.DESIGN_SHA},
        "mechanism": C.MECHANISM_NAME,
        "magnitude_based": False,
        "thresholds": {"G2_payload_bytes": C.G2_PAYLOAD_MAX_BYTES,
                       "G3_snr": C.G3_SNR_MAX,
                       "roc_auc_ci": list(C.ROC_AUC_CI),
                       "roc_auc_mde": C.ROC_AUC_MDE,
                       "baseline_roc_auc_mean": C.BASELINE_ROC_AUC_MEAN},
        "k_values": C.K_VALUES,
        "payload": {a: PENDING for a in C.ARMS},
        "snr": {a: PENDING for a in C.ARMS},
        "privacy": PENDING,
        "task": {a: PENDING for a in C.ARMS},
        "k0_gate": PENDING,
        "acceptance": PENDING,
        "note": ("Every field marked PENDING exists only after execution. This "
                 "file is a SCHEMA, not a result, and contains no measured value."),
    }


# ==========================================================================
# Main
# ==========================================================================
def main() -> int:
    ap = argparse.ArgumentParser(
        description="Phase 18 / Experiment 9 aggregation and acceptance")
    ap.add_argument("--out-dir", default=os.path.join(C.REPO, C.OUT_DIRNAME))
    ap.add_argument("--emit-schema", action="store_true",
                    help="write the artifact schema with explicit PENDING fields "
                         "and exit; produces no experimental result")
    a = ap.parse_args()
    os.chdir(C.REPO)

    if a.emit_schema:
        os.makedirs(a.out_dir, exist_ok=True)
        p = os.path.join(a.out_dir, "exp9_schema.json")
        with open(p, "w", encoding="utf-8") as fh:
            json.dump(schema(), fh, indent=2, sort_keys=True)
        print(f"[OK] schema (no results) -> {p}")
        return 0

    print("=" * 78)
    print("EXPERIMENT 9 - AGGREGATION AND ACCEPTANCE")
    print(f"mechanism: {C.MECHANISM_NAME}  --  NOT top-k")
    print("=" * 78)

    # ---- design SS 31: the frozen gate applies to the verdict path too ------
    C.gate_frozen_before(C.snapshot_frozen())
    print(f"[OK] {len(C.FROZEN_SHA)} frozen inputs verified")

    dp_path = os.path.join(a.out_dir, "exp9_dp_metrics.csv")
    pay_path = os.path.join(a.out_dir, "exp9_payload.json")
    task_path = os.path.join(a.out_dir, "exp9_task_metrics.csv")

    if not os.path.isfile(dp_path):
        C.abort(f"missing {dp_path}. Run run_exp9.py first. Refusing to emit a "
                "verdict without the DP/SNR half.")
    dp_rows = read_csv_rows(dp_path)
    payload_raw = (json.load(open(pay_path, "r", encoding="utf-8"))
                   if os.path.isfile(pay_path) else None)
    task_rows = read_csv_rows(task_path) if os.path.isfile(task_path) else []
    if not task_rows:
        print("[INFO] task half not present - criterion C reported PENDING "
              "(never PASS on missing evidence)")

    snr = snr_table(dp_rows)
    priv = privacy_table(dp_rows)
    pay = payload_table(payload_raw)
    task = task_table(task_rows)
    k0_gate = task.pop("_k0_gate", {"status": PENDING})
    acc = acceptance(pay, snr, task)

    # ---------------- design SS 30.1 ----------------
    print("\n[TABLE 1] per-arm complete transport payload (criterion A)")
    print(f"{'arm':<5}{'k':>13}{'raw 4k':>14}{'serialised':>13}{'encrypted':>13}"
          f"{'chunks':>8}{'uploaded':>13}{'vs G2':>8}")
    for arm in C.ARMS:
        p = pay[arm]
        if p["status"] == PENDING:
            print(f"{arm:<5}{C.K_VALUES[arm]:>13,}{'PENDING':>69}")
            continue
        print(f"{arm:<5}{p['k']:>13,}{p['raw_4k_bytes']:>14,}"
              f"{p['serialised_bytes']:>13,}{p['encrypted_bytes']:>13,}"
              f"{p['n_chunks']:>8}{p['total_uploaded_bytes']:>13,}"
              f"{'PASS' if p['criterion_A_pass'] else 'FAIL':>8}")
    print(f"      G2 threshold = {C.G2_PAYLOAD_MAX_BYTES:,} B (frozen, design SS 17)")

    # ---------------- design SS 30.2 ----------------
    print("\n[TABLE 2] per-arm SNR (criterion B)")
    print(f"{'arm':<5}{'mean':>14}{'sd':>12}{'min':>14}{'max':>14}"
          f"{'analytical':>14}{'dev':>12}{'vs G3':>8}")
    for arm in C.ARMS:
        s = snr[arm]
        if s["status"] == PENDING:
            print(f"{arm:<5}{'PENDING':>88}")
            continue
        print(f"{arm:<5}{s['mean']:>14.6f}{s['sd']:>12.6f}{s['min']:>14.6f}"
              f"{s['max']:>14.6f}{s['analytical']:>14.6f}{s['deviation_abs']:>12.6f}"
              f"{'PASS' if s['criterion_B_pass'] else 'FAIL':>8}")
    print(f"      G3 threshold = {C.G3_SNR_MAX} (frozen, design SS 15); "
          "analytical is NEVER an acceptance input (SS 13)")

    # ---------------- design SS 30.3 ----------------
    print("\n[TABLE 3] per-arm task signal (criterion C)")
    print(f"{'arm':<5}{'runs':>6}{'fold mean':>13}{'ci95_lo':>11}{'ci95_hi':>11}"
          f"{'in CI':>7}{'paired deg':>12}{'<=MDE':>7}{'C':>6}")
    for arm in C.ARMS:
        t = task[arm]
        if t.get("status") != "MEASURED":
            print(f"{arm:<5}{'PENDING - task half not executed':>73}")
            continue
        deg = t.get("paired_degradation_mean")
        print(f"{arm:<5}{t['n_fold_runs']:>6}{t['mean']:>13.6f}"
              f"{t['ci95_lo']:>11.6f}{t['ci95_hi']:>11.6f}"
              f"{'yes' if t['inside_baseline_ci'] else 'NO':>7}"
              f"{(f'{deg:+.6f}' if deg is not None else 'n/a'):>12}"
              f"{('yes' if t.get('criterion_C2_mde_pass') else 'NO') if deg is not None else 'n/a':>7}"
              f"{str(t.get('criterion_C_pass')):>6}")
    print(f"      SS 23 CI = [{C.ROC_AUC_CI[0]}, {C.ROC_AUC_CI[1]}]  "
          f"SS 24 MDE = {C.ROC_AUC_MDE}")

    print("\n[K0 EQUIVALENCE GATE - design SS 22, MANDATORY]")
    if k0_gate.get("status") != "MEASURED":
        print("   PENDING - the gate is evaluated only on this run's own K0 output; "
              "it is never asserted")
    else:
        print(f"   K0 fold-level ROC-AUC mean = {k0_gate['k0_fold_level_mean']:.6f}")
        print(f"   Baseline-CV reference mean = {C.BASELINE_ROC_AUC_MEAN:.6f}  "
              f"CI [{C.ROC_AUC_CI[0]:.6f}, {C.ROC_AUC_CI[1]:.6f}]")
        print(f"   GATE = {'PASS' if k0_gate['passed'] else 'FAIL -> HARD ABORT of the task interpretation'}")

    # ---------------- design SS 30.4 / SS 30.5 ----------------
    print("\n[TABLE 4] A/B/C matrix and conjunction (design SS 25)")
    print(f"{'arm':<5}{'k':>13}{'A payload':>12}{'B snr':>10}{'C task':>10}{'overall':>10}")
    for arm in C.ARMS:
        m = acc["matrix"][arm]
        print(f"{arm:<5}{m['k']:>13,}{str(m['A_payload']):>12}"
              f"{str(m['B_snr']):>10}{str(m['C_task']):>10}{m['overall']:>10}")

    print("\n[TABLE 5] privacy")
    print(f"   eps/update = {priv['recomputed_epsilon']:.15f} (recomputed by the "
          f"frozen dp_agent) - IDENTICAL for every arm; eps does not depend on k")
    print(f"   delta = {C.DELTA_DP}   composition = {C.COMPOSITION}")
    print(f"   cumulative <= {C.EPS_CUMULATIVE_MAX} over "
          f"{C.N_UPDATES_PER_ROUND} updates")

    print("\n[VERDICT] " + acc["verdict"])

    summary = {
        "schema": "exp9_summary/1",
        "experiment": C.EXPERIMENT,
        "design": {"path": C.DESIGN_PATH, "sha256": C.DESIGN_SHA},
        "mechanism": C.MECHANISM_NAME, "magnitude_based": False,
        "thresholds": {"G2_payload_bytes": C.G2_PAYLOAD_MAX_BYTES,
                       "G3_snr": C.G3_SNR_MAX,
                       "roc_auc_ci": list(C.ROC_AUC_CI),
                       "roc_auc_mde": C.ROC_AUC_MDE,
                       "baseline_roc_auc_mean": C.BASELINE_ROC_AUC_MEAN,
                       "t_crit": C.T_CRIT, "df": C.DF},
        "k_values": C.K_VALUES,
        "payload": pay, "snr": snr, "privacy": priv, "task": task,
        "k0_gate": k0_gate, "acceptance": acc,
        "scope_caveat": ("Exp 9 answers the PUBLICLY-SPECIFIED question of design "
                         "SS 4, which is strictly weaker than B-3's. A negative "
                         "result establishes only that the family defined by O "
                         "fails, not that no compact DAIC-derived object survives "
                         "DP (design SS 4, SS 34)."),
        "utility_caveat": ("Exp 9 makes no predictive-utility claim beyond "
                           "preservation of the frozen Baseline-CV ranking signal "
                           "(design SS 33.4, SS 34)."),
    }
    spath = os.path.join(a.out_dir, "exp9_summary.json")
    with open(spath, "w", encoding="utf-8") as fh:
        json.dump(summary, fh, indent=2, sort_keys=True, default=float)

    apath = os.path.join(a.out_dir, "exp9_acceptance.csv")
    with open(apath, "w", newline="", encoding="utf-8") as fh:
        w = csv.writer(fh)
        w.writerow(["arm", "k", "A_payload", "B_snr", "C_task", "overall"])
        for arm in C.ARMS:
            m = acc["matrix"][arm]
            w.writerow([arm, m["k"], m["A_payload"], m["B_snr"], m["C_task"],
                        m["overall"]])

    print(f"\n[DONE] -> {spath}")
    print(f"[DONE] -> {apath}")
    print("Next: verify_exp9.py (fail-closed verification)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
