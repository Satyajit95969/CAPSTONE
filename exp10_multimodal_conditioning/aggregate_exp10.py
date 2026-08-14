#!/usr/bin/env python3
"""
aggregate_exp10.py - Phase 19 / Experiment 10 aggregation and acceptance.

Authoritative specification: PHASE_19_EXP10_DESIGN.md (FROZEN). This module
implements ONLY the frozen acceptance logic of design SS 14, SS 15, SS 16, SS 17
and SS 18. No criterion, threshold or constant may be altered.

FROZEN ACCEPTANCE (design SS 18) - every constant is PRE-EXISTING:

    G0   M0 fold-level ROC-AUC mean inside [0.5755177227457472,
                                            0.6911434704671701]   MANDATORY GATE
    P1a  mean paired (M2 - M1) ROC-AUC  >=  0.05781287386071144
    P1b  95 % CI of that paired difference EXCLUDES 0
    P2   M2 fold-level ROC-AUC 95 % CI EXCLUDES 0.5               (chance)
    P3   mean paired (M0 - M2) ROC-AUC  <=  0.05781287386071144   (adverse dir.)
    P4   mean paired (MAE M2 - MAE M0)  <=  0.40660852779350554   (adverse dir.)
    S1   mean paired (M2 - M1) PR-AUC   reported, NOT in the conjunction

SIGN CONVENTIONS (design SS 18) - stated at every call site below:
    P1  delta = ROC_AUC(M2, fold) - ROC_AUC(M1, fold)   positive = M2 BETTER
    P3  delta = ROC_AUC(M0, fold) - ROC_AUC(M2, fold)   positive = M2 WORSE
    P4  delta = MAE(M2, fold)     - MAE(M0, fold)       positive = M2 WORSE
                                                        (MAE is an error metric)

If G0 fails the interpretation is VOID: M1 vs M2 must not be interpreted and no
conditioning verdict may be reported (design SS 14). H0 is a legitimate result
and is never an implementation failure (design SS 18.7).

Usage:
    python exp10_multimodal_conditioning/aggregate_exp10.py
    python exp10_multimodal_conditioning/aggregate_exp10.py --emit-schema
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import sys
from typing import Any, Dict, List

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import exp10_common as C  # noqa: E402

PENDING = "PENDING"
VOID = "VOID"


def read_csv_rows(path: str) -> List[Dict[str, str]]:
    with open(path, "r", newline="", encoding="utf-8") as fh:
        return list(csv.DictReader(fh))


# ==========================================================================
# Per-arm fold-level statistics - design SS 11, SS 15, SS 16
# ==========================================================================
def arm_table(rows: List[Dict[str, str]]) -> Dict[str, Any]:
    out: Dict[str, Any] = {}
    for arm in C.ARMS:
        arm_rows = [r for r in rows if r["arm"] == arm]
        if not arm_rows:
            out[arm] = {"status": PENDING, "reason": "no fold-runs for this arm"}
            continue
        expected = C.N_FOLDS * C.N_REPEATS
        if len(arm_rows) != expected:
            C.abort(f"{arm}: {len(arm_rows)} fold-runs, expected {expected} "
                    "(design SS 11). Refusing to aggregate a partial arm.")
        seeds = sorted({int(r["seed"]) for r in arm_rows})
        if seeds != sorted(C.BASE_SEED + r for r in C.REPEATS):
            C.abort(f"{arm}: seeds {seeds} != frozen "
                    f"{sorted(C.BASE_SEED + r for r in C.REPEATS)} (design SS 11)")

        entry: Dict[str, Any] = {"status": "MEASURED", "n_fold_runs": len(arm_rows),
                                 "conditioned": C.ARM_SPEC[arm]["conditioned"],
                                 "n_params": C.ARM_SPEC[arm]["n_params"],
                                 "fusion_in": C.ARM_SPEC[arm]["fusion_in"]}
        primary = C.fold_means(arm_rows, arm, C.PRIMARY_METRIC)
        if sorted(primary) != list(C.FOLD_IDS):
            C.abort(f"{arm}: folds {sorted(primary)} != frozen {list(C.FOLD_IDS)}")
        st = C.fold_level_ci(list(primary.values()))
        lo, hi = C.ROC_AUC_CI
        entry.update({
            "fold_means": {str(f): v for f, v in primary.items()},
            **st,
            "baseline_ci": [lo, hi],
            "inside_baseline_ci": bool(lo <= st["mean"] <= hi),
            # design SS 18.3 - 0.5 is the chance level, a mathematical constant.
            "excludes_chance": bool(st["ci95_lo"] > C.CHANCE_ROC_AUC
                                    or st["ci95_hi"] < C.CHANCE_ROC_AUC),
        })
        for metric in (C.SECONDARY_METRIC, C.GUARD_METRIC, C.DIAGNOSTIC_METRIC):
            fm = C.fold_means(arm_rows, arm, metric)
            entry[metric] = {"fold_means": {str(f): v for f, v in fm.items()},
                             **C.fold_level_ci(list(fm.values()))}
        # design SS 16 - recorded, never acceptance inputs.
        entry["non_acceptance_metrics"] = {
            m: C.mean([float(r[m]) for r in arm_rows])
            for m in C.NON_ACCEPTANCE_METRICS if m in arm_rows[0]}
        out[arm] = entry
    return out


# ==========================================================================
# G0 - the mandatory equivalence gate (design SS 14)
# ==========================================================================
def g0_gate(arms: Dict[str, Any]) -> Dict[str, Any]:
    m0 = arms.get(C.CONTROL_ARM, {})
    if m0.get("status") != "MEASURED":
        return {"status": PENDING,
                "reason": f"{C.CONTROL_ARM} not present in the metrics"}
    lo, hi = C.ROC_AUC_CI
    passed = bool(lo <= m0["mean"] <= hi)
    return {
        "status": "MEASURED",
        "m0_fold_level_mean": m0["mean"],
        "baseline_reference_mean": C.BASELINE_ROC_AUC_MEAN,
        "baseline_ci": [lo, hi],
        "passed": passed,
        "precedent_note": ("Exp 7's A0 reproduced Baseline-CV on all twelve "
                           "primary metrics and Exp 9's K0 passed this gate at "
                           "0.6332539682539682, but that is PRECEDENT, not proof: "
                           "the gate is evaluated on this run's own M0 output "
                           "(design SS 14)."),
    }


# ==========================================================================
# P1 - P4 and S1 (design SS 18)
# ==========================================================================
def contrasts(rows: List[Dict[str, str]], arms: Dict[str, Any],
              gate_passed: bool) -> Dict[str, Any]:
    need = [a for a in C.ARMS if arms.get(a, {}).get("status") != "MEASURED"]
    if need:
        return {"status": PENDING, "reason": f"arms not measured: {need}"}

    def fm(arm: str, metric: str) -> Dict[int, float]:
        return C.fold_means([r for r in rows if r["arm"] == arm], arm, metric)

    roc = {a: fm(a, C.PRIMARY_METRIC) for a in C.ARMS}
    pr = {a: fm(a, C.SECONDARY_METRIC) for a in C.ARMS}
    mae = {a: fm(a, C.GUARD_METRIC) for a in C.ARMS}

    # P1: positive = M2 BETTER than M1 (design SS 18.2)
    p1 = C.paired_contrast(roc[C.TREATMENT_ARM], roc[C.RAW_ARM])
    p1a = bool(p1["mean"] >= C.ROC_AUC_MDE)
    p1b = bool(p1["ci95_lo"] > 0.0 or p1["ci95_hi"] < 0.0)
    # P3: positive = M2 WORSE than M0, i.e. adverse (design SS 18.4)
    p3 = C.paired_contrast(roc[C.CONTROL_ARM], roc[C.TREATMENT_ARM])
    p3_pass = bool(p3["mean"] <= C.ROC_AUC_MDE)
    # P4: positive = M2 has HIGHER error than M0, i.e. adverse (design SS 18.5)
    p4 = C.paired_contrast(mae[C.TREATMENT_ARM], mae[C.CONTROL_ARM])
    p4_pass = bool(p4["mean"] <= C.MAE_MDE)
    # S1: reported only (design SS 18.6)
    s1 = C.paired_contrast(pr[C.TREATMENT_ARM], pr[C.RAW_ARM])

    out = {
        "status": "MEASURED",
        "P1_M2_minus_M1_ROC_AUC": {
            **p1, "direction": "positive = M2 better than M1",
            "mde": C.ROC_AUC_MDE,
            "P1a_meets_mde": p1a, "P1b_ci_excludes_zero": p1b,
            "P1_pass": bool(p1a and p1b)},
        "P3_M0_minus_M2_ROC_AUC": {
            **p3, "direction": "positive = M2 worse than M0 (adverse)",
            "mde": C.ROC_AUC_MDE, "P3_pass": p3_pass},
        "P4_MAE_M2_minus_M0": {
            **p4, "direction": "positive = M2 higher error than M0 (adverse)",
            "mde": C.MAE_MDE, "P4_pass": p4_pass},
        "S1_M2_minus_M1_PR_AUC": {
            **s1, "direction": "positive = M2 better than M1",
            "mde": C.PR_AUC_MDE,
            "note": "reported, NOT part of the conjunction (design SS 18.6)"},
    }
    if not gate_passed:
        # design SS 14 - the measurements are retained; the INTERPRETATION is void.
        for key in ("P1_M2_minus_M1_ROC_AUC", "P3_M0_minus_M2_ROC_AUC",
                    "P4_MAE_M2_minus_M0"):
            out[key]["interpretation"] = VOID
            out[key]["void_reason"] = ("M0 failed the SS 14 equivalence gate; no "
                                       "conditioning verdict may be reported")
    return out


# ==========================================================================
# Conjunction and verdict (design SS 18.7)
# ==========================================================================
def acceptance(gate: Dict[str, Any], arms: Dict[str, Any],
               con: Dict[str, Any]) -> Dict[str, Any]:
    def pend(x: Any) -> bool:
        return x is PENDING or x == PENDING

    g0 = gate.get("passed") if gate.get("status") == "MEASURED" else PENDING
    if con.get("status") != "MEASURED":
        p1 = p2 = p3 = p4 = PENDING
    elif g0 is not True:
        p1 = p2 = p3 = p4 = (VOID if g0 is False else PENDING)
    else:
        p1 = con["P1_M2_minus_M1_ROC_AUC"]["P1_pass"]
        p2 = arms[C.TREATMENT_ARM]["excludes_chance"]
        p3 = con["P3_M0_minus_M2_ROC_AUC"]["P3_pass"]
        p4 = con["P4_MAE_M2_minus_M0"]["P4_pass"]

    if g0 is False:
        conj, verdict = VOID, (
            "VOID - M0 failed the SS 14 equivalence gate. The harness did not "
            "reproduce Baseline-CV, so M1 vs M2 must not be interpreted and no "
            "conditioning success or failure may be claimed. The measurement is "
            "reported; the interpretation is not.")
    elif any(pend(x) for x in (g0, p1, p2, p3, p4)):
        conj, verdict = PENDING, (
            "INCOMPLETE - required evidence is missing. No experiment-level "
            "verdict may be issued on partial evidence.")
    elif p1 and p2 and p3 and p4:
        conj, verdict = "SUPPORTED", (
            "SUPPORTED - G0, P1, P2, P3 and P4 all hold. The feature-scale "
            "hypothesis is supported FOR THIS ARCHITECTURE (design SS 30). This "
            "is not a claim of demonstrated predictive utility, and not a claim "
            "about other fusion designs, feature extractors or datasets.")
    elif p1 and not p2:
        conj, verdict = "PARTIAL", (
            "PARTIAL - conditioning improved ranking (P1) without establishing "
            "an above-chance signal (P2 failed). Reported as the boundary "
            "measurement it is (design SS 18.7).")
    else:
        conj, verdict = "H0", (
            "H0 - this conditioning strategy does not rescue this fusion "
            "architecture. This is a LEGITIMATE SCIENTIFIC RESULT, not an "
            "implementation failure (design SS 18.7). It establishes ONLY that "
            "THIS conditioning strategy fails for THIS architecture - NOT that "
            "audio and vision modalities carry no predictive information in "
            "general (design SS 30).")

    return {"G0": g0, "P1": p1, "P2": p2, "P3": p3, "P4": p4,
            "conjunction": conj, "verdict": verdict}


def schema() -> Dict[str, Any]:
    return {
        "schema": "exp10_summary/1",
        "design": {"path": C.DESIGN_PATH, "sha256": C.DESIGN_SHA},
        "thresholds": THRESHOLDS,
        "arms": {a: PENDING for a in C.ARMS},
        "g0_gate": PENDING, "contrasts": PENDING, "acceptance": PENDING,
        "scope_caveat": C.SCOPE_CAVEAT,
        "mde_transfer_assumption": C.MDE_TRANSFER_ASSUMPTION,
        "note": ("Every field marked PENDING exists only after execution. This "
                 "file is a SCHEMA, not a result, and contains no measured value."),
    }


THRESHOLDS: Dict[str, Any] = {
    "baseline_roc_auc_mean": C.BASELINE_ROC_AUC_MEAN,
    "roc_auc_ci": list(C.ROC_AUC_CI),
    "roc_auc_mde": C.ROC_AUC_MDE,
    "pr_auc_mde": C.PR_AUC_MDE,
    "mae_mde": C.MAE_MDE,
    "chance_roc_auc": C.CHANCE_ROC_AUC,
    "t_crit": C.T_CRIT, "df": C.DF,
}


def write_acceptance_csv(path: str, gate: Dict[str, Any], arms: Dict[str, Any],
                         con: Dict[str, Any], acc: Dict[str, Any]) -> None:
    """design SS 21.7 - frozen row set and column order."""
    def g(d: Dict[str, Any], *keys: str) -> Any:
        for k in keys:
            if not isinstance(d, dict):
                return PENDING
            d = d.get(k, PENDING)
        return d
    m2 = arms.get(C.TREATMENT_ARM, {})
    rows = [
        ["G0", "M0 fold-level ROC-AUC mean", g(gate, "m0_fold_level_mean"),
         f"[{C.ROC_AUC_CI[0]}, {C.ROC_AUC_CI[1]}]", "inside", acc["G0"]],
        ["P1a", "mean paired M2-M1 ROC-AUC",
         g(con, "P1_M2_minus_M1_ROC_AUC", "mean"), C.ROC_AUC_MDE, ">=",
         g(con, "P1_M2_minus_M1_ROC_AUC", "P1a_meets_mde")],
        ["P1b", "95% CI of paired M2-M1",
         f'[{g(con, "P1_M2_minus_M1_ROC_AUC", "ci95_lo")}, '
         f'{g(con, "P1_M2_minus_M1_ROC_AUC", "ci95_hi")}]', 0, "excludes",
         g(con, "P1_M2_minus_M1_ROC_AUC", "P1b_ci_excludes_zero")],
        ["P2", "M2 fold-level ROC-AUC CI",
         f'[{m2.get("ci95_lo", PENDING)}, {m2.get("ci95_hi", PENDING)}]',
         C.CHANCE_ROC_AUC, "excludes", acc["P2"]],
        ["P3", "mean paired M0-M2 ROC-AUC",
         g(con, "P3_M0_minus_M2_ROC_AUC", "mean"), C.ROC_AUC_MDE, "<=",
         g(con, "P3_M0_minus_M2_ROC_AUC", "P3_pass")],
        ["P4", "mean paired MAE M2-M0",
         g(con, "P4_MAE_M2_minus_M0", "mean"), C.MAE_MDE, "<=",
         g(con, "P4_MAE_M2_minus_M0", "P4_pass")],
        ["S1", "mean paired M2-M1 PR-AUC",
         g(con, "S1_M2_minus_M1_PR_AUC", "mean"), C.PR_AUC_MDE, "reported", ""],
    ]
    with open(path, "w", newline="", encoding="utf-8") as fh:
        w = csv.writer(fh)
        w.writerow(["criterion", "quantity", "value", "threshold", "direction",
                    "pass"])
        w.writerows(rows)


def main() -> int:
    ap = argparse.ArgumentParser(
        description="Phase 19 / Experiment 10 aggregation and acceptance")
    ap.add_argument("--out-dir", default=os.path.join(C.REPO, C.OUT_DIRNAME))
    ap.add_argument("--emit-schema", action="store_true",
                    help="write the artifact schema with explicit PENDING fields "
                         "and exit; produces no experimental result")
    a = ap.parse_args()
    os.chdir(C.REPO)

    if a.emit_schema:
        os.makedirs(a.out_dir, exist_ok=True)
        p = os.path.join(a.out_dir, "exp10_schema.json")
        with open(p, "w", encoding="utf-8") as fh:
            json.dump(schema(), fh, indent=2, sort_keys=True)
        print(f"[OK] schema (no results) -> {p}")
        return 0

    print("=" * 78)
    print("EXPERIMENT 10 - AGGREGATION AND ACCEPTANCE")
    print("single factor: feature conditioning - no DP, no masking, no payload")
    print("=" * 78)

    C.gate_frozen_before(C.snapshot_frozen())
    print(f"[OK] {len(C.FROZEN_SHA)} frozen inputs verified")

    mpath = os.path.join(a.out_dir, "exp10_metrics.csv")
    if not os.path.isfile(mpath):
        C.abort(f"missing {mpath}. Run run_exp10.py first. Refusing to emit a "
                "verdict without the task metrics.")
    rows = read_csv_rows(mpath)

    arms = arm_table(rows)
    gate = g0_gate(arms)
    con = contrasts(rows, arms, bool(gate.get("passed") is True))
    acc = acceptance(gate, arms, con)

    # ---------------- reporting ----------------
    print("\n[TABLE 1] per-arm fold-level ROC-AUC")
    print(f"{'arm':<5}{'runs':>6}{'mean':>12}{'sd':>10}{'ci95_lo':>11}"
          f"{'ci95_hi':>11}{'in base CI':>12}{'>chance':>9}")
    for arm in C.ARMS:
        e = arms[arm]
        if e.get("status") != "MEASURED":
            print(f"{arm:<5}{'PENDING':>66}")
            continue
        print(f"{arm:<5}{e['n_fold_runs']:>6}{e['mean']:>12.6f}{e['sd']:>10.6f}"
              f"{e['ci95_lo']:>11.6f}{e['ci95_hi']:>11.6f}"
              f"{('yes' if e['inside_baseline_ci'] else 'no'):>12}"
              f"{('yes' if e['excludes_chance'] else 'NO'):>9}")

    print("\n[TABLE 2] secondary / guard / diagnostic (fold-level means)")
    print(f"{'arm':<5}{'PR_AUC':>12}{'MAE':>12}{'pred_var':>12}")
    for arm in C.ARMS:
        e = arms[arm]
        if e.get("status") != "MEASURED":
            print(f"{arm:<5}{'PENDING':>36}")
            continue
        print(f"{arm:<5}{e[C.SECONDARY_METRIC]['mean']:>12.6f}"
              f"{e[C.GUARD_METRIC]['mean']:>12.6f}"
              f"{e[C.DIAGNOSTIC_METRIC]['mean']:>12.6f}")

    print("\n[G0 EQUIVALENCE GATE - design SS 14, MANDATORY]")
    if gate.get("status") != "MEASURED":
        print("   PENDING - evaluated only on this run's own M0 output")
    else:
        print(f"   M0 fold-level ROC-AUC mean = {gate['m0_fold_level_mean']:.6f}")
        print(f"   Baseline-CV reference mean = {C.BASELINE_ROC_AUC_MEAN:.6f}  "
              f"CI [{C.ROC_AUC_CI[0]:.6f}, {C.ROC_AUC_CI[1]:.6f}]")
        print(f"   GATE = {'PASS' if gate['passed'] else 'FAIL -> INTERPRETATION VOID'}")

    print("\n[TABLE 3] contrasts (design SS 18)")
    if con.get("status") != "MEASURED":
        print("   PENDING")
    else:
        for key, label in (("P1_M2_minus_M1_ROC_AUC", "P1  M2-M1 ROC-AUC"),
                           ("P3_M0_minus_M2_ROC_AUC", "P3  M0-M2 ROC-AUC"),
                           ("P4_MAE_M2_minus_M0", "P4  MAE M2-M0"),
                           ("S1_M2_minus_M1_PR_AUC", "S1  M2-M1 PR-AUC")):
            d = con[key]
            print(f"   {label:<20} mean={d['mean']:+.6f}  "
                  f"CI [{d['ci95_lo']:+.6f}, {d['ci95_hi']:+.6f}]  "
                  f"mde={d['mde']}  ({d['direction']})")

    print("\n[TABLE 4] acceptance matrix (design SS 18.7)")
    for k in ("G0", "P1", "P2", "P3", "P4"):
        print(f"   {k}: {acc[k]}")
    print(f"\n[VERDICT] {acc['verdict']}")

    summary = {
        "schema": "exp10_summary/1",
        "experiment": C.EXPERIMENT,
        "design": {"path": C.DESIGN_PATH, "sha256": C.DESIGN_SHA},
        "scientific_question": C.SCIENTIFIC_QUESTION,
        "thresholds": THRESHOLDS,
        "arms": arms, "g0_gate": gate, "contrasts": con, "acceptance": acc,
        "scope_caveat": C.SCOPE_CAVEAT,
        "mde_transfer_assumption": C.MDE_TRANSFER_ASSUMPTION,
        "dp": {"applied": False}, "masking": {"applied": False},
        "payload": {"measured": False}, "federation": {"applied": False},
    }
    spath = os.path.join(a.out_dir, "exp10_summary.json")
    with open(spath, "w", encoding="utf-8") as fh:
        json.dump(summary, fh, indent=2, sort_keys=True, default=float)
    apath = os.path.join(a.out_dir, "exp10_acceptance.csv")
    write_acceptance_csv(apath, gate, arms, con, acc)

    print(f"\n[DONE] -> {spath}")
    print(f"[DONE] -> {apath}")
    print("Next: verify_exp10.py (fail-closed verification)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
