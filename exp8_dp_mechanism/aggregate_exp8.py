#!/usr/bin/env python3
"""
aggregate_exp8.py - Phase 17 / Experiment 8 aggregation and acceptance.

Authoritative specification: PHASE_17_EXP8_DESIGN.md (FROZEN). This module
implements ONLY the frozen statistical procedure of design SS 20 and the frozen
acceptance logic of design SS 17. No criterion, threshold or constant may be
altered.

FROZEN STATISTICAL PROCEDURE (design SS 20)
    Delta_i  = SNR_D0,i - SNR_D1,i          i = 1..5
    mean     = (1/5) sum Delta_i
    s        = sqrt( sum (Delta_i - mean)^2 / (5-1) )        [ddof = 1]
    SE       = s / sqrt(5)
    CI95     = [ mean - t_crit*SE , mean + t_crit*SE ]       t_crit = 2.7764451051977987
                                                             df = 5 - 1 = 4

FROZEN ACCEPTANCE - T3 + k_min = 1.01 (design SS 17.3). P2 requires ALL THREE:
    A   mean(Delta) > 0
    B   the paired 95% CI excludes 0
    C   mean(SNR_D0)/mean(SNR_D1) >= 1.01

47.91 is NEVER used to judge P2 (design SS 17.2); it is carried as a diagnostic
only. H0 is a VALID scientific outcome and is never reported as an
implementation failure (design SS 18, SS 25.7).

Usage:
    python exp8_dp_mechanism/aggregate_exp8.py
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
import sys
from typing import Any, Dict, List, Optional

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO)

# ==========================================================================
# FROZEN CONSTANTS - design SS 12, SS 15.1, SS 16, SS 17.3, SS 20
# ==========================================================================
N_REPETITIONS = 5
DF = N_REPETITIONS - 1                       # = 4  (design SS 20)
T_CRIT = 2.7764451051977987                  # design SS 20; == scipy t.ppf(.975, 4)
K_MIN = 1.01                                 # design SS 17.3 condition C
BASE_SEED = 1000
SEEDS = tuple(BASE_SEED + i for i in range(1, N_REPETITIONS + 1))

EPS_PER_UPDATE_MAX = 5.302585092994046       # design SS 12
EPS_CUMULATIVE_MAX = 15.9078
N_UPDATES_PER_ROUND = 3
DELTA = 1e-05
CONTROL_ARM, CANDIDATE_ARM = "D0", "D1"

# design SS 9, SS 10 - P0 hard-gate constants (F5). Identical to the values the
# runner and verifier enforce; duplicated here so the VERDICT path itself gates.
EXPECTED_D = 295_681
D0_BAND_LO, D0_BAND_HI = 540.0, 548.0


def abort(message: str) -> "NoReturn":  # noqa: F821
    print(f"\n[ABORT] {message}", file=sys.stderr)
    raise SystemExit(1)


def sha256_file(path: str) -> Optional[str]:
    if not os.path.isfile(path):
        return None
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for b in iter(lambda: fh.read(8192), b""):
            h.update(b)
    return h.hexdigest()


# ==========================================================================
# Frozen paired statistics - design SS 20
# ==========================================================================
def paired_ci(deltas: List[float]) -> Dict[str, float]:
    """Paired mean and 95% CI, exactly as written in design SS 20.

    Mirrors the project's documented paired-difference convention
    (aggregate_exp7.py: paired differences -> fold_level_ci, df = k-1). With
    k = r = 5 the df is 4 and t_crit is the project's existing constant.
    """
    k = len(deltas)
    if k < 2:
        abort(f"paired CI requires >= 2 draws, got {k}")
    mean = sum(deltas) / k
    var = sum((x - mean) ** 2 for x in deltas) / (k - 1)      # ddof = 1
    s = math.sqrt(var)
    se = s / math.sqrt(k)
    half = T_CRIT * se
    return {"mean": mean, "sd": s, "se": se,
            "ci95_lo": mean - half, "ci95_hi": mean + half,
            "k": k, "df": k - 1, "t_crit": T_CRIT}


def describe(vals: List[float]) -> Dict[str, float]:
    n = len(vals)
    mean = sum(vals) / n
    var = sum((x - mean) ** 2 for x in vals) / (n - 1) if n > 1 else 0.0
    return {"mean": mean, "sd": math.sqrt(var),
            "min": min(vals), "max": max(vals), "n": n}


# ==========================================================================
# Frozen acceptance - design SS 17
# ==========================================================================
def evaluate_p2(snr_d0: List[float], snr_d1: List[float]) -> Dict[str, Any]:
    """T3 + k_min = 1.01 (design SS 17.3). ALL THREE of A, B, C required."""
    deltas = [a - b for a, b in zip(snr_d0, snr_d1)]
    ci = paired_ci(deltas)
    m0 = sum(snr_d0) / len(snr_d0)
    m1 = sum(snr_d1) / len(snr_d1)
    ratio = m0 / m1 if m1 != 0 else math.inf

    cond_a = ci["mean"] > 0.0
    cond_b = (ci["ci95_lo"] > 0.0) or (ci["ci95_hi"] < 0.0)   # excludes zero
    cond_c = ratio >= K_MIN

    return {
        "per_seed_delta": deltas,
        "paired": ci,
        "mean_snr_D0": m0, "mean_snr_D1": m1,
        "relative_improvement_factor": ratio,
        "k_min": K_MIN,
        "condition_A_mean_positive": bool(cond_a),
        "condition_B_ci_excludes_zero": bool(cond_b),
        "condition_C_ratio_ge_kmin": bool(cond_c),
        "P2_met": bool(cond_a and cond_b and cond_c),
    }


def main() -> int:
    ap = argparse.ArgumentParser(description="Aggregate Experiment 8")
    ap.add_argument("--exp-dir", default=os.path.join(
        REPO, "trainer_outputs", "exp8_dp_mechanism"))
    a = ap.parse_args()
    os.chdir(REPO)

    mpath = os.path.join(a.exp_dir, "exp8_metrics.csv")
    if not os.path.isfile(mpath):
        abort(f"missing {mpath}; run run_exp8.py first")
    with open(mpath, newline="", encoding="utf-8") as fh:
        rows = list(csv.DictReader(fh))

    print("=" * 74)
    print("EXPERIMENT 8 AGGREGATION")
    print("=" * 74)

    by_arm: Dict[str, Dict[int, Dict[str, Any]]] = {CONTROL_ARM: {}, CANDIDATE_ARM: {}}
    for r in rows:
        arm = r["arm"]
        if arm not in by_arm:
            abort(f"unexpected arm {arm!r}; only {CONTROL_ARM}/{CANDIDATE_ARM} allowed")
        by_arm[arm][int(r["seed"])] = r

    # ---- completeness and pairing (design SS 15, SS 16) ---------------------
    for arm in (CONTROL_ARM, CANDIDATE_ARM):
        got = sorted(by_arm[arm])
        if got != list(SEEDS):
            abort(f"{arm}: seeds {got} != frozen {list(SEEDS)}")
    print(f"[OK] both arms complete at the frozen seeds {list(SEEDS)} (paired)")

    # ---- privacy-budget preservation, P1 (design SS 12, SS 17) --------------
    p1_detail = []
    p1_ok = True
    for arm in (CONTROL_ARM, CANDIDATE_ARM):
        eps = [float(by_arm[arm][s]["epsilon"]) for s in SEEDS]
        e = max(eps)
        cum = e * N_UPDATES_PER_ROUND
        ok = (e <= EPS_PER_UPDATE_MAX + 1e-12) and (cum <= EPS_CUMULATIVE_MAX + 1e-9)
        p1_ok &= ok
        p1_detail.append({"arm": arm, "epsilon_max": e, "cumulative": cum,
                          "within_budget": bool(ok)})
    print(f"[{'OK' if p1_ok else 'FAIL'}] P1 privacy budget: " +
          ", ".join(f"{d['arm']} eps={d['epsilon_max']:.12f}" for d in p1_detail))

    # ---- P3 cryptographic verification -------------------------------------
    p3_ok = all(str(r["crypto_ok"]).strip().lower() in ("true", "1") for r in rows)
    print(f"[{'OK' if p3_ok else 'FAIL'}] P3 cryptographic verification")

    # ---- per-arm descriptive stats (design SS 20 reporting) -----------------
    snr = {arm: [float(by_arm[arm][s]["snr"]) for s in SEEDS]
           for arm in (CONTROL_ARM, CANDIDATE_ARM)}
    analytical = {arm: float(by_arm[arm][SEEDS[0]]["analytical_snr"])
                  for arm in (CONTROL_ARM, CANDIDATE_ARM)}
    per_arm = {}
    for arm in (CONTROL_ARM, CANDIDATE_ARM):
        d = describe(snr[arm])
        d["analytical_snr"] = analytical[arm]
        d["empirical_minus_analytical"] = d["mean"] - analytical[arm]
        d["relative_deviation"] = ((d["mean"] - analytical[arm]) / analytical[arm]
                                   if analytical[arm] else float("nan"))
        per_arm[arm] = d

    # ---- P2 (design SS 17.3) ------------------------------------------------
    p2 = evaluate_p2(snr[CONTROL_ARM], snr[CANDIDATE_ARM])

    # ---- P0 HARD GATE (design SS 17 P0) - F5 --------------------------------
    # The COMPLETE P0 requirement is enforced here, in the verdict path:
    # published D0 L2-after band, exact eps, identical pre-DP object, and the
    # D0 reference record. A separate verifier invocation is NOT relied upon.
    gate_path = os.path.join(a.exp_dir, "d0_reference_check.json")
    gate = json.load(open(gate_path, encoding="utf-8")) if os.path.isfile(gate_path) else None
    p0: Dict[str, Any] = {}

    p0["reference_record_present"] = gate is not None

    d0_snr = [float(by_arm[CONTROL_ARM][s]["snr"]) for s in SEEDS]
    p0["d0_band"] = {"lo": D0_BAND_LO, "hi": D0_BAND_HI,
                     "min": min(d0_snr), "max": max(d0_snr),
                     "within": all(D0_BAND_LO <= x <= D0_BAND_HI for x in d0_snr)}

    d0_eps = [float(by_arm[CONTROL_ARM][s]["epsilon"]) for s in SEEDS]
    p0["d0_epsilon_exact"] = all(abs(e - EPS_PER_UPDATE_MAX) < 1e-12 for e in d0_eps)

    l2b = {round(float(r["l2_before"]), 10) for r in rows}
    p0["identical_pre_dp_object"] = len(l2b) == 1

    p0["d0_target_d"] = (gate or {}).get("d")
    p0["d0_target_d_ok"] = p0["d0_target_d"] == EXPECTED_D
    equiv = float(((gate or {}).get("allocation") or {}).get("delta_sigma_sq", float("nan")))
    p0["privacy_equivalence"] = abs(equiv - 1.0) < 1e-9

    p0_ok = bool(p0["reference_record_present"] and p0["d0_band"]["within"]
                 and p0["d0_epsilon_exact"] and p0["identical_pre_dp_object"]
                 and p0["d0_target_d_ok"] and p0["privacy_equivalence"])
    p0["passed"] = p0_ok
    print(f"[{'OK' if p0_ok else 'FAIL'}] P0 hard gate: band={p0['d0_band']['within']} "
          f"eps_exact={p0['d0_epsilon_exact']} identical_object="
          f"{p0['identical_pre_dp_object']} d={p0['d0_target_d_ok']} "
          f"equiv={p0['privacy_equivalence']}")

    # ---- P4 REPRODUCIBILITY (design SS 17 P4) - F3 --------------------------
    # Derived from the receipt the RUN produced. Never hardcoded, never assumed.
    rcp_path = os.path.join(a.exp_dir, "exp8_receipt.json")
    rcp = json.load(open(rcp_path, encoding="utf-8")) if os.path.isfile(rcp_path) else None
    p4: Dict[str, Any] = {"receipt_present": rcp is not None}
    if rcp is None:
        p4.update({"draws_identical": False, "frozen_unchanged_pre_post": False,
                   "reason": "exp8_receipt.json absent - reproducibility unverified"})
    else:
        repro = rcp.get("reproducibility") or {}
        fz = rcp.get("frozen_input_sha") or {}
        p4["draws_identical"] = bool(repro.get("checked")
                                     and repro.get("all_draws_identical"))
        p4["n_draws_rechecked"] = repro.get("n_draws_rechecked")
        p4["frozen_unchanged_pre_post"] = bool(fz.get("unchanged_pre_post")
                                               and fz.get("pinned_all_match"))
    p4_ok = bool(p4["receipt_present"] and p4.get("draws_identical")
                 and p4.get("frozen_unchanged_pre_post"))
    p4["passed"] = p4_ok
    print(f"[{'OK' if p4_ok else 'FAIL'}] P4 reproducibility: "
          f"draws_identical={p4.get('draws_identical')} "
          f"frozen_unchanged={p4.get('frozen_unchanged_pre_post')}")

    # A PASS verdict is impossible unless EVERY criterion, P0 first, has passed.
    verdict_pass = bool(p0_ok and p1_ok and p2["P2_met"] and p3_ok and p4_ok)
    verdict = ("PASS - material improvement demonstrated at unchanged privacy budget"
               if verdict_pass else
               "H0 / FAIL FOR MATERIAL IMPROVEMENT - not an implementation failure")

    summary = {
        "experiment": "Phase 17 / Exp 8 - DP mechanism (INFRASTRUCTURE)",
        "not_a_predictive_utility_experiment": True,
        "arms": {CONTROL_ARM: "frozen Gaussian baseline (flat clip)",
                 CANDIDATE_ARM: "per-group clipping + Mahalanobis-accounted Gaussian"},
        "seeds": list(SEEDS), "n_repetitions": N_REPETITIONS,
        "statistical_procedure": {
            "paired_difference": "Delta_i = SNR_D0,i - SNR_D1,i",
            "df": DF, "t_crit": T_CRIT, "ddof": 1,
            "note": "frozen design SS 20; matches the project's documented "
                    "paired-difference convention (df = k-1)",
        },
        "per_arm": per_arm,
        "P0_hard_gate": p0,
        "P1_privacy_budget": {"passed": bool(p1_ok), "detail": p1_detail,
                              "delta": DELTA,
                              "budget_per_update": EPS_PER_UPDATE_MAX,
                              "cumulative_budget": EPS_CUMULATIVE_MAX},
        "P2_material_improvement": p2,
        "P3_cryptographic_verification": bool(p3_ok),
        "P4_reproducibility": p4,
        "overall_verdict": verdict,
        "diagnostic_only_47_91": {
            "distortion_ratio_mean": {
                arm: sum(float(by_arm[arm][s]["distortion_ratio"]) for s in SEEDS) / len(SEEDS)
                for arm in (CONTROL_ARM, CANDIDATE_ARM)},
            "note": "l2_after/l2_before. Diagnostic ONLY - never used to judge P2 "
                    "(design SS 17.2).",
        },
        "caveat": ("Infrastructure result. Exp 8 makes no predictive-utility claim "
                   "(design SS 1, SS 24). H0 is a valid scientific outcome (design SS 18)."),
        "input_sha": {"exp8_metrics.csv": sha256_file(mpath),
                      "d0_reference_check.json": sha256_file(gate_path)},
    }

    dest = os.path.join(a.exp_dir, "exp8_summary.json")
    with open(dest, "w", encoding="utf-8") as fh:
        json.dump(summary, fh, indent=2, sort_keys=True, default=float)

    delta_doc = {
        "schema": "exp8_delta/1",
        "pairing": "same seed for D0 and D1 (design SS 16)",
        "per_seed": [{"seed": s,
                      "snr_D0": snr[CONTROL_ARM][i],
                      "snr_D1": snr[CANDIDATE_ARM][i],
                      "delta": p2["per_seed_delta"][i]}
                     for i, s in enumerate(SEEDS)],
        "paired": p2["paired"],
        "relative_improvement_factor": p2["relative_improvement_factor"],
        "k_min": K_MIN,
        "conditions": {"A": p2["condition_A_mean_positive"],
                       "B": p2["condition_B_ci_excludes_zero"],
                       "C": p2["condition_C_ratio_ge_kmin"]},
        "P2_met": p2["P2_met"],
    }
    with open(os.path.join(a.exp_dir, "exp8_delta.json"), "w", encoding="utf-8") as fh:
        json.dump(delta_doc, fh, indent=2, sort_keys=True, default=float)

    # ---- console ------------------------------------------------------------
    print("\nper-arm SNR (lower is better)")
    print(f"  {'arm':4s} {'mean':>12s} {'sd':>10s} {'min':>12s} {'max':>12s} "
          f"{'analytical':>12s} {'dev':>10s}")
    for arm in (CONTROL_ARM, CANDIDATE_ARM):
        d = per_arm[arm]
        print(f"  {arm:4s} {d['mean']:>12.6f} {d['sd']:>10.6f} {d['min']:>12.6f} "
              f"{d['max']:>12.6f} {d['analytical_snr']:>12.6f} "
              f"{d['empirical_minus_analytical']:>+10.6f}")

    print("\npaired analysis (design SS 20)")
    print(f"  mean Delta          : {p2['paired']['mean']:+.6f}")
    print(f"  SD / SE             : {p2['paired']['sd']:.6f} / {p2['paired']['se']:.6f}")
    print(f"  95% CI (df={DF}, t={T_CRIT:.6f}) : "
          f"[{p2['paired']['ci95_lo']:+.6f}, {p2['paired']['ci95_hi']:+.6f}]")
    print(f"  relative improvement: {p2['relative_improvement_factor']:.6f}x "
          f"(k_min = {K_MIN})")

    print("\nP2 conditions (T3 + k_min, ALL required)")
    print(f"  A mean(Delta) > 0            : {p2['condition_A_mean_positive']}")
    print(f"  B 95% CI excludes 0          : {p2['condition_B_ci_excludes_zero']}")
    print(f"  C ratio >= {K_MIN}             : {p2['condition_C_ratio_ge_kmin']}")

    print(f"\nP0 {p0_ok}  P1 {p1_ok}  P2 {p2['P2_met']}  P3 {p3_ok}  P4 {p4_ok}")
    print(f"VERDICT: {verdict}")
    print(f"written: {dest}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
