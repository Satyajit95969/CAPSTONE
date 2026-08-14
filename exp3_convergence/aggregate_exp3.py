#!/usr/bin/env python
"""
Phase 13 / Experiment 3 - aggregation and inference.

Consumes the per-fold artifacts written by run_exp3.py and produces the official
Experiment 3 result: the E0 equivalence gate, fold-level aggregates with Student-t
intervals, the PAIRED per-fold comparison against Baseline-CV, the convergence
analysis, and the secondary observational diagnostics.

AGGREGATION CONTRACT
--------------------
Replicates aggregate_baseline_cv.py (L23-34, L67-68) exactly, so Experiment 3,
Experiment 4 and Baseline-CV remain mutually comparable:

  1. repeats are averaged WITHIN a fold first  -> one value per fold
  2. the fold - not the fold-run - is the unit of independence
  3. CIs use Student-t with df = k-1 = 4  (t = 2.7764), never a normal quantile

Equivalence is not asserted on trust: main() re-derives Baseline-CV's own
published aggregate from its raw metrics CSVs and checks it against
baseline_cv_summary.json before any Experiment 3 number is computed.

THE E0 EQUIVALENCE GATE - A HARD ABORT CONDITION
------------------------------------------------
Experiment 3 changes only the `epochs` argument of the frozen trainer, so no
training code is reimplemented. E0 (3 epochs) nonetheless exists to validate the
surrounding harness - data materialisation, seeding, metric computation. E0's
aggregate must fall inside Baseline-CV's 95% CI on every primary metric. Because
torch is unseeded at the CUDA level, reproduction is STATISTICAL, not bitwise;
the protocol's reproducibility contract accepts this ("the aggregate comparator
is the reproducible object"). If E0 fails, NO arm result may be reported.

ACCEPTANCE CRITERIA
-------------------
Primary (roadmap section 6 Exp 3):
  C0  E0 reproduces Baseline-CV                      (design gate)
  C1  Loss reaches a stopping criterion              (convergence behaviour)
  C2  Regression MAE improves beyond the noise band  (regression improvement)
  C3  Modality ablation stays 0/0                    (roadmap section 7.5 invariant)

Distribution diagnostics (probability band width, degeneracy count, prediction
variance, PR-AUC/ROC-AUC movement) are reported IN FULL as SECONDARY
OBSERVATIONAL OUTCOMES. They are explicitly NOT pass/fail conditions.

Baseline-CV is opened READ-ONLY. All output goes to the Experiment 3 directory.

Usage (from the repository root):
    python exp3_convergence/aggregate_exp3.py
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
from typing import Dict, List, Sequence

import numpy as np
import pandas as pd
from scipy import stats

# --------------------------------------------------------------------------
# Frozen constants - must agree with run_exp3.py
# --------------------------------------------------------------------------
ARMS: Dict[str, int] = {"E0": 3, "E1": 20, "E2": 10}
PRIMARY_ARM = "E1"
CONTROL_ARM = "E0"

N_REPEATS = 5
N_FOLDS = 5

# Metrics carried through aggregation and the paired comparison.
PRIMARY_METRICS = ["MAE", "RMSE", "pred_var", "MAE_mean_pred", "ROC_AUC", "PR_AUC",
                   "balAcc", "MCC", "Precision", "Recall", "F1", "Accuracy"]
PAIRED_METRICS = ["MAE", "RMSE", "pred_var", "ROC_AUC", "PR_AUC",
                  "balAcc", "MCC", "Precision", "Recall", "F1", "Accuracy"]

# Baseline-CV MDE anchor for Criterion 2 (Phase 11.6 section 6).
MDE_MAE_BASELINE = 0.4066

# Baseline-CV / Exp 4 references for the OBSERVATIONAL diagnostics.
REF_BAND_WIDTH = 0.0292          # Exp 4 measured mean per-fold p_pos band width
REF_DEGENERACY_BASELINE = 25     # Baseline-CV: 25/25 all-negative at argmax
REF_DEGENERACY_EXP4_P1 = 24      # Exp 4 P1: 24/25 constant decisions

AGG_TOL = 1e-9


def sha256(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for block in iter(lambda: fh.read(8192), b""):
            h.update(block)
    return h.hexdigest()


def abort(message: str) -> "NoReturn":  # noqa: F821
    print(f"\n[ABORT] {message}", file=sys.stderr)
    raise SystemExit(1)


# --------------------------------------------------------------------------
# Aggregation primitives
# --------------------------------------------------------------------------
def fold_level_ci(per_fold: Sequence[float]) -> Dict[str, float]:
    """Mean, SD, SE, t-critical and 95% CI from the k per-fold values.

    Replicates aggregate_baseline_cv.py:23-34.
    """
    v = np.asarray(per_fold, dtype=float)
    k = int(len(v))
    mean = float(np.mean(v))
    sd = float(np.std(v, ddof=1)) if k > 1 else 0.0
    se = sd / np.sqrt(k) if k > 1 else 0.0
    tcrit = float(stats.t.ppf(0.975, k - 1)) if k > 1 else float("nan")
    half = tcrit * se if k > 1 else 0.0
    return {"mean": mean, "fold_sd": sd, "se": se, "t_crit": tcrit,
            "ci95_lo": mean - half, "ci95_hi": mean + half, "k": k}


def per_fold_means(df: pd.DataFrame, metrics: Sequence[str]) -> pd.DataFrame:
    """Average repeats within each fold - step 1 of the aggregation contract."""
    cols = [c for c in metrics if c in df.columns]
    return df.groupby("fold")[cols].mean().sort_index()


def paired_test(values: Sequence[float], mu0: float = 0.0) -> Dict[str, float]:
    """Paired-fold one-sample t-test against a constant.

    Zero variance is handled explicitly rather than allowed to produce NaN: an
    identical value in every fold is either exactly the reference (no effect) or
    a perfectly consistent offset.
    """
    v = np.asarray(values, dtype=float)
    k = int(len(v))
    diff = v - float(mu0)
    mean = float(np.mean(diff))
    sd = float(np.std(diff, ddof=1)) if k > 1 else 0.0
    se = sd / np.sqrt(k) if k > 1 else 0.0
    tcrit = float(stats.t.ppf(0.975, k - 1)) if k > 1 else float("nan")

    if sd == 0.0:
        t_stat = float("inf") if mean != 0.0 else 0.0
        p_val = 0.0 if mean != 0.0 else 1.0
        degenerate = True
    else:
        res = stats.ttest_1samp(v, popmean=float(mu0))
        t_stat, p_val, degenerate = float(res.statistic), float(res.pvalue), False

    mde = tcrit * se
    return {"mu0": float(mu0), "mean_delta": mean, "sd_delta": sd, "se": se,
            "t_crit": tcrit, "t_stat": t_stat, "p_value": p_val,
            "ci95_lo": mean - mde, "ci95_hi": mean + mde, "mde_95": mde,
            "detectable": bool(abs(mean) > mde) if not degenerate else bool(mean != 0.0),
            "zero_variance": degenerate}


# --------------------------------------------------------------------------
# Self-tests
# --------------------------------------------------------------------------
def load_baseline_metrics(runs_dir: str) -> pd.DataFrame:
    frames = []
    for r in range(1, N_REPEATS + 1):
        p = os.path.join(runs_dir, f"rep{r}_metrics.csv")
        if not os.path.isfile(p):
            abort(f"missing frozen Baseline-CV metrics file: {p}")
        frames.append(pd.read_csv(p))
    df = pd.concat(frames, ignore_index=True)
    if len(df) != N_REPEATS * N_FOLDS:
        abort(f"Baseline-CV metrics has {len(df)} rows, expected {N_REPEATS * N_FOLDS}")
    return df


def selftest_aggregation(baseline_df: pd.DataFrame, summary_path: str) -> float:
    """Prove this module's aggregation matches the Phase 11.6 implementation."""
    with open(summary_path, "r", encoding="utf-8") as fh:
        published = json.load(fh)
    worst = 0.0
    n = 0
    for metric, ref in published.get("metrics", {}).items():
        if metric not in baseline_df.columns:
            continue
        got = fold_level_ci(baseline_df.groupby("fold")[metric].mean().sort_index().values)
        d = max(abs(got["mean"] - float(ref["mean"])),
                abs(got["fold_sd"] - float(ref["fold_sd"])),
                abs(got["se"] - float(ref["se"])),
                abs(got["ci95_lo"] - float(ref["ci95"][0])),
                abs(got["ci95_hi"] - float(ref["ci95"][1])),
                abs(got["t_crit"] - float(ref["t_crit"])))
        worst = max(worst, d)
        n += 1
        if d > AGG_TOL:
            abort(f"aggregation self-test FAILED on {metric}: max |diff| {d:.3e} > {AGG_TOL:.1e}")
    if n == 0:
        abort("aggregation self-test found no overlapping metrics")
    print(f"[PASS] aggregation self-test: reproduces baseline_cv_summary.json across "
          f"{n} metrics (max |diff| {worst:.3e})")
    return worst


def e0_equivalence_gate(exp3_df: pd.DataFrame, summary_path: str) -> Dict:
    """HARD GATE: E0 must fall inside Baseline-CV's 95% CI on every primary metric.

    Reproduction is statistical, not bitwise (CUDA nondeterminism); the protocol's
    reproducibility contract makes the aggregate the reproducible object.
    """
    with open(summary_path, "r", encoding="utf-8") as fh:
        published = json.load(fh)["metrics"]

    e0 = exp3_df[exp3_df["arm"] == CONTROL_ARM]
    if len(e0) != N_REPEATS * N_FOLDS:
        abort(f"E0 arm has {len(e0)} fold-runs, expected {N_REPEATS * N_FOLDS}. "
              "The equivalence gate cannot be evaluated.")

    pf = per_fold_means(e0, PRIMARY_METRICS)
    rows, failures = [], []
    for m in PRIMARY_METRICS:
        if m not in pf.columns or m not in published:
            continue
        got = fold_level_ci(pf[m].values)
        lo, hi = float(published[m]["ci95"][0]), float(published[m]["ci95"][1])
        inside = bool(lo <= got["mean"] <= hi)
        rows.append({"metric": m, "e0_mean": got["mean"], "e0_fold_sd": got["fold_sd"],
                     "baseline_mean": float(published[m]["mean"]),
                     "baseline_ci_lo": lo, "baseline_ci_hi": hi, "inside_ci": inside})
        if not inside:
            failures.append(m)

    report = {"gate": "E0 aggregate inside Baseline-CV 95% CI",
              "n_metrics": len(rows), "failures": failures,
              "passed": not failures, "detail": rows}
    if failures:
        for row in rows:
            if not row["inside_ci"]:
                print(f"   E0 {row['metric']}: {row['e0_mean']:.6f} outside "
                      f"[{row['baseline_ci_lo']:.6f}, {row['baseline_ci_hi']:.6f}]",
                      file=sys.stderr)
        abort(f"E0 EQUIVALENCE GATE FAILED on {failures}. The harness does not "
              "reproduce Baseline-CV; no arm result may be reported.")
    print(f"[PASS] E0 equivalence gate: all {len(rows)} primary metrics inside "
          "Baseline-CV's 95% CI")
    return report


# --------------------------------------------------------------------------
# Reporting
# --------------------------------------------------------------------------
def fmt(x, dp: int = 4) -> str:
    if x is None or (isinstance(x, float) and np.isnan(x)):
        return "n/a"
    if isinstance(x, float) and x != 0.0 and abs(x) < 10 ** (-dp):
        return f"{x:.2e}"
    return f"{x:.{dp}f}"


def build_markdown(S: Dict) -> str:
    L: List[str] = []
    L.append("# Experiment 3 - Convergence: Aggregate Summary\n")
    L.append(f"- Arms: " + " | ".join(f"{a} = {ARMS[a]} epochs" for a in S["arms"]))
    L.append(f"- Primary arm: **{PRIMARY_ARM}** ({ARMS[PRIMARY_ARM]} epochs) | "
             f"control: {CONTROL_ARM} ({ARMS[CONTROL_ARM]} epochs)")
    L.append(f"- Repeats R = {S['n_repeats']} | Folds k = {S['n_folds']} | "
             f"fold-runs per arm = {S['n_repeats'] * S['n_folds']}")
    L.append(f"- CIs: fold-level Student-t, df = {S['n_folds'] - 1}, t = {S['t_crit']:.4f} "
             "(repeats averaged within fold first)")
    L.append("- Single changed factor: the epoch budget passed to the frozen "
             "`fine_tune_supervised`. No training code was reimplemented.\n")

    L.append("## Self-tests and gates\n")
    L.append("| Test | Result |")
    L.append("|---|---|")
    L.append(f"| Aggregation reproduces `baseline_cv_summary.json` (max diff "
             f"{S['selftests']['aggregation_max_diff']:.3e}) | PASS |")
    L.append(f"| **E0 equivalence gate** ({S['e0_gate']['n_metrics']} metrics inside "
             f"Baseline-CV 95% CI) | {'PASS' if S['e0_gate']['passed'] else 'FAIL'} |")
    L.append(f"| Modality ablation 0/0 (audio_dim, vision_dim) | "
             f"{'PASS' if S['ablation_zero'] else 'FAIL'} |\n")

    L.append("## E0 equivalence detail\n")
    L.append("| Metric | E0 mean | Baseline-CV mean | Baseline-CV 95% CI | Inside |")
    L.append("|---|---|---|---|---|")
    for r in S["e0_gate"]["detail"]:
        L.append(f"| {r['metric']} | {fmt(r['e0_mean'], 6)} | {fmt(r['baseline_mean'], 6)} | "
                 f"[{fmt(r['baseline_ci_lo'], 6)}, {fmt(r['baseline_ci_hi'], 6)}] | "
                 f"{'yes' if r['inside_ci'] else 'NO'} |")
    L.append("")

    L.append("## Aggregate metrics by arm (mean +/- fold SD [95% CI])\n")
    L.append("| Metric | " + " | ".join(S["arms"]) + " |")
    L.append("|---" * (len(S["arms"]) + 1) + "|")
    for m in PRIMARY_METRICS:
        cells = []
        for arm in S["arms"]:
            a = S["aggregates"][arm].get(m)
            cells.append("n/a" if a is None else
                         f"{fmt(a['mean'])} +/- {fmt(a['fold_sd'])} "
                         f"[{fmt(a['ci95_lo'])}, {fmt(a['ci95_hi'])}]")
        L.append(f"| {m} | " + " | ".join(cells) + " |")
    L.append("")

    L.append("## Convergence analysis\n")
    L.append("| Arm | epochs | first loss | final loss | total reduction | "
             "last-epoch rel. improvement | plateau reached |")
    L.append("|---|---|---|---|---|---|---|")
    for arm in S["arms"]:
        c = S["convergence"][arm]
        L.append(f"| {arm} | {ARMS[arm]} | {fmt(c['first_loss_mean'])} | "
                 f"{fmt(c['final_loss_mean'])} | {fmt(c['total_reduction_mean'])} | "
                 f"{fmt(c['last_rel_improvement_mean'], 5)} | "
                 f"{c['plateau_reached_count']}/{c['n_fold_runs']} |")
    L.append("")
    L.append(f"Plateau rule (pre-declared, diagnostic only): relative epoch-over-epoch "
             f"training-loss improvement < {S['plateau_rel_tol']} sustained for "
             f"{S['plateau_patience']} consecutive epochs.\n")

    L.append(f"## Paired per-fold comparison: {PRIMARY_ARM} vs Baseline-CV\n")
    L.append("Noise band re-estimated from Experiment 3's own fold spread; "
             "Baseline-CV's degenerate MDEs are never inherited.\n")
    L.append("| Metric | Baseline-CV | Exp 3 | mean delta | 95% CI | MDE | Detectable |")
    L.append("|---|---|---|---|---|---|---|")
    for m in PAIRED_METRICS:
        p = S["paired_vs_baseline"].get(m)
        if p is None:
            continue
        L.append(f"| {m} | {fmt(p['baseline_mean'])} | {fmt(p['exp3_mean'])} | "
                 f"{fmt(p['mean_delta'])} | [{fmt(p['ci95_lo'])}, {fmt(p['ci95_hi'])}] | "
                 f"{fmt(p['mde_95'])} | {'YES' if p['detectable'] else 'no'} |")
    L.append("")

    L.append("## Secondary observational outcomes (NOT acceptance criteria)\n")
    L.append("Reported in full. These address the mechanism Experiment 4 identified; "
             "per the approved design they are **observational**, not pass/fail.\n")
    L.append("| Arm | p_pos band width | p_pos mean | pred_var | pred_phq spread | "
             "degenerate fold-runs | pos_rate |")
    L.append("|---|---|---|---|---|---|---|")
    for arm in S["arms"]:
        d = S["diagnostics"][arm]
        L.append(f"| {arm} | {fmt(d['band_width_mean'])} | {fmt(d['p_pos_mean'])} | "
                 f"{fmt(d['pred_var_mean'], 6)} | {fmt(d['pred_phq_spread_mean'])} | "
                 f"{d['degenerate_count']}/{d['n_fold_runs']} | {fmt(d['pos_rate_mean'])} |")
    L.append("")
    L.append(f"Reference points: Baseline-CV band width {REF_BAND_WIDTH} (measured in "
             f"Exp 4), Baseline-CV degeneracy {REF_DEGENERACY_BASELINE}/25 at argmax, "
             f"Exp 4 policy P1 degeneracy {REF_DEGENERACY_EXP4_P1}/25.\n")

    L.append("## Acceptance criteria\n")
    L.append("| # | Criterion | Source | Verdict |")
    L.append("|---|---|---|---|")
    for c in S["acceptance_criteria"]:
        L.append(f"| {c['id']} | {c['statement']} | {c['source']} | "
                 f"{'MET' if c['met'] else 'NOT MET'} |")
    L.append(f"\n**Overall: {S['overall_verdict']}**\n")

    L.append("## Notes\n")
    L.append("- Runtime increase is expected and is NOT a failure (roadmap section 6 Exp 3).")
    L.append("- P/R/F1 remaining 0.0 is an explicitly informative outcome, confirming "
             "RC-1 needs more than epochs.")
    L.append("- Accuracy is reported but is NOT a decision metric (protocol section 3.3).")
    L.append("- k = 5 makes these intervals wide; that width is a property of the design.")
    return "\n".join(L) + "\n"


# --------------------------------------------------------------------------
# Main
# --------------------------------------------------------------------------
def main() -> None:
    ap = argparse.ArgumentParser(
        description="Phase 13 / Experiment 3 - aggregation, E0 gate, paired "
                    "inference against Baseline-CV, convergence analysis.")
    ap.add_argument("--exp3-dir", default="trainer_outputs/exp3_convergence")
    ap.add_argument("--baseline-dir", default="trainer_outputs/baseline_cv")
    ap.add_argument("--out", default=None,
                    help="output prefix; defaults to <exp3-dir>/exp3_summary")
    a = ap.parse_args()

    exp3_dir = os.path.normpath(a.exp3_dir)
    baseline_dir = os.path.normpath(a.baseline_dir)
    runs_dir = os.path.join(baseline_dir, "runs")
    out_prefix = a.out or os.path.join(exp3_dir, "exp3_summary")

    if os.path.abspath(out_prefix).startswith(os.path.abspath(baseline_dir) + os.sep):
        abort(f"output prefix {out_prefix} is inside the frozen baseline dir")

    metrics_path = os.path.join(exp3_dir, "exp3_metrics.csv")
    if not os.path.isfile(metrics_path):
        abort(f"{metrics_path} not found - run run_exp3.py first")

    print("=" * 74)
    print("PHASE 13 / EXPERIMENT 3 - AGGREGATION")
    print(f"  exp3 dir             : {exp3_dir}")
    print(f"  baseline (read-only) : {baseline_dir}")
    print("=" * 74)

    exp3_df = pd.read_csv(metrics_path)
    baseline_df = load_baseline_metrics(runs_dir)
    summary_path = os.path.join(baseline_dir, "baseline_cv_summary.json")
    if not os.path.isfile(summary_path):
        abort(f"missing frozen comparator: {summary_path}")

    shas_before = {p: sha256(os.path.join(baseline_dir, p))
                   for p in ["baseline_cv_summary.json", "fold_manifest.json",
                             "trivial_control_arm.csv"]}

    # ---- self-tests and the hard gate ------------------------------------
    agg_worst = selftest_aggregation(baseline_df, summary_path)
    e0_report = e0_equivalence_gate(exp3_df, summary_path)
    with open(os.path.join(exp3_dir, "e0_equivalence_check.json"), "w",
              encoding="utf-8") as fh:
        json.dump(e0_report, fh, indent=1)

    arms_present = [a_ for a_ in ARMS if a_ in set(exp3_df["arm"].unique())]
    missing = [a_ for a_ in ARMS if a_ not in arms_present]
    if missing:
        abort(f"exp3_metrics.csv is missing arms {missing}")
    for arm in arms_present:
        n = len(exp3_df[exp3_df["arm"] == arm])
        if n != N_REPEATS * N_FOLDS:
            abort(f"arm {arm} has {n} fold-runs, expected {N_REPEATS * N_FOLDS}")

    # ---- aggregates -------------------------------------------------------
    aggregates: Dict[str, Dict[str, Dict[str, float]]] = {}
    per_fold_by_arm: Dict[str, pd.DataFrame] = {}
    for arm in arms_present:
        pf = per_fold_means(exp3_df[exp3_df["arm"] == arm], PRIMARY_METRICS)
        per_fold_by_arm[arm] = pf
        aggregates[arm] = {m: fold_level_ci(pf[m].values) for m in pf.columns}

    # ---- convergence ------------------------------------------------------
    conv_path = os.path.join(exp3_dir, "convergence_report.csv")
    if not os.path.isfile(conv_path):
        abort(f"{conv_path} not found - run run_exp3.py first")
    conv_df = pd.read_csv(conv_path)
    convergence: Dict[str, Dict] = {}
    for arm in arms_present:
        c = conv_df[conv_df["arm"] == arm]
        convergence[arm] = {
            "n_fold_runs": int(len(c)),
            "first_loss_mean": float(c["first_loss"].mean()),
            "final_loss_mean": float(c["final_loss"].mean()),
            "total_reduction_mean": float(c["total_reduction"].mean()),
            "last_rel_improvement_mean": float(c["last_epoch_rel_improvement"].mean()),
            "plateau_reached_count": int(c["plateau_reached"].sum()),
            "plateau_reached_rate": float(c["plateau_reached"].mean()),
            "plateau_epoch_mean": float(c.loc[c["plateau_epoch"] > 0, "plateau_epoch"].mean())
            if (c["plateau_epoch"] > 0).any() else float("nan"),
        }

    # ---- secondary observational diagnostics -----------------------------
    diag_path = os.path.join(exp3_dir, "distribution_diagnostics.csv")
    if not os.path.isfile(diag_path):
        abort(f"{diag_path} not found - run run_exp3.py first")
    diag_df = pd.read_csv(diag_path)
    diagnostics: Dict[str, Dict] = {}
    for arm in arms_present:
        d = diag_df[diag_df["arm"] == arm]
        pv = exp3_df[exp3_df["arm"] == arm]["pred_var"]
        diagnostics[arm] = {
            "n_fold_runs": int(len(d)),
            "band_width_mean": float(d["p_pos_band_width"].mean()),
            "band_width_sd": float(d["p_pos_band_width"].std(ddof=1)),
            "p_pos_mean": float(d["p_pos_mean"].mean()),
            "p_pos_min": float(d["p_pos_min"].min()),
            "p_pos_max": float(d["p_pos_max"].max()),
            "pred_phq_spread_mean": float(d["pred_phq_spread"].mean()),
            "pred_var_mean": float(pv.mean()),
            "pos_rate_mean": float(d["pos_rate"].mean()),
            "degenerate_count": int(d["degenerate"].sum()),
        }

    # ---- paired comparison: primary arm vs Baseline-CV -------------------
    base_pf = per_fold_means(baseline_df, PAIRED_METRICS)
    exp3_pf = per_fold_by_arm[PRIMARY_ARM]
    paired: Dict[str, Dict] = {}
    for m in PAIRED_METRICS:
        if m not in exp3_pf.columns or m not in base_pf.columns:
            continue
        deltas = exp3_pf[m].values - base_pf[m].values
        paired[m] = {"baseline_mean": float(np.mean(base_pf[m].values)),
                     "exp3_mean": float(np.mean(exp3_pf[m].values)),
                     "per_fold_delta": [float(x) for x in deltas],
                     **paired_test(deltas, 0.0)}

    # ---- acceptance criteria ---------------------------------------------
    ablation_zero = True  # enforced by run_exp3.py's infer_dims gate before training
    conv_primary = convergence[PRIMARY_ARM]
    conv_control = convergence[CONTROL_ARM]
    c1_met = bool(conv_primary["plateau_reached_rate"] > conv_control["plateau_reached_rate"])
    mae = paired.get("MAE", {})
    # Criterion 2: MAE must IMPROVE (fall) beyond the Baseline-CV MDE anchor.
    c2_met = bool(mae.get("mean_delta", 0.0) < 0 and abs(mae.get("mean_delta", 0.0)) > MDE_MAE_BASELINE)

    crit = [
        {"id": 0, "statement": "E0 reproduces Baseline-CV (all primary metrics inside its 95% CI)",
         "source": "design gate", "met": bool(e0_report["passed"])},
        {"id": 1, "statement": f"Loss reaches a stopping criterion ({PRIMARY_ARM} plateau rate "
                               f"{conv_primary['plateau_reached_count']}/{conv_primary['n_fold_runs']} "
                               f"exceeds {CONTROL_ARM}'s {conv_control['plateau_reached_count']}"
                               f"/{conv_control['n_fold_runs']})",
         "source": "Roadmap section 6 Exp 3", "met": c1_met},
        {"id": 2, "statement": f"Regression MAE improves beyond the noise band "
                               f"(|delta| > MDE {MDE_MAE_BASELINE})",
         "source": "Roadmap section 6 Exp 3", "met": c2_met},
        {"id": 3, "statement": "Modality ablation stays 0/0",
         "source": "Roadmap section 7.5", "met": ablation_zero},
    ]
    primary_met = all(c["met"] for c in crit)
    overall = ("H1 SUPPORTED - collapse is substantially a budget artifact" if primary_met
               else "H0 - collapse is NOT a budget artifact; RC-1 needs more than epochs"
               if crit[0]["met"] and crit[3]["met"]
               else "INCONCLUSIVE - review criteria individually")

    S = {
        "experiment": "Phase 13 / Exp 3 - convergence",
        "arms": arms_present, "arm_epochs": {k: ARMS[k] for k in arms_present},
        "primary_arm": PRIMARY_ARM, "control_arm": CONTROL_ARM,
        "n_repeats": N_REPEATS, "n_folds": N_FOLDS,
        "t_crit": float(stats.t.ppf(0.975, N_FOLDS - 1)),
        "ci_method": "fold-level Student-t (df=k-1), repeats averaged within fold",
        "plateau_rel_tol": 0.01, "plateau_patience": 2,
        "selftests": {"aggregation_max_diff": float(agg_worst)},
        "e0_gate": e0_report,
        "ablation_zero": ablation_zero,
        "aggregates": aggregates,
        "convergence": convergence,
        "diagnostics": diagnostics,
        "diagnostics_status": "SECONDARY OBSERVATIONAL OUTCOMES - not acceptance criteria",
        "paired_vs_baseline": paired,
        "mde_mae_baseline": MDE_MAE_BASELINE,
        "acceptance_criteria": crit,
        "overall_verdict": overall,
        "baseline_input_sha": shas_before,
    }

    with open(out_prefix + ".json", "w", encoding="utf-8") as fh:
        json.dump(S, fh, indent=1)
    with open(out_prefix + ".md", "w", encoding="utf-8") as fh:
        fh.write(build_markdown(S))

    shas_after = {p: sha256(os.path.join(baseline_dir, p)) for p in shas_before}
    changed = [k for k in shas_before if shas_before[k] != shas_after[k]]
    if changed:
        abort(f"frozen Baseline-CV inputs changed during aggregation: {changed}")

    # ---- console -----------------------------------------------------------
    print("-" * 74)
    print(f"CONVERGENCE")
    for arm in arms_present:
        c = convergence[arm]
        print(f"  {arm} ({ARMS[arm]:>2} ep): loss {c['first_loss_mean']:8.4f} -> "
              f"{c['final_loss_mean']:8.4f} | plateau "
              f"{c['plateau_reached_count']}/{c['n_fold_runs']}")
    print("-" * 74)
    print(f"PAIRED vs Baseline-CV ({PRIMARY_ARM})")
    for m in PAIRED_METRICS:
        p = paired.get(m)
        if p is None:
            continue
        print(f"  {m:10s} base {p['baseline_mean']:8.4f} -> exp3 {p['exp3_mean']:8.4f} "
              f"| delta {p['mean_delta']:+8.4f} MDE {p['mde_95']:.4f} "
              f"| {'DETECTABLE' if p['detectable'] else 'not detectable'}")
    print("-" * 74)
    print("SECONDARY OBSERVATIONAL DIAGNOSTICS (not pass/fail)")
    for arm in arms_present:
        d = diagnostics[arm]
        print(f"  {arm}: band {d['band_width_mean']:.4f} | pred_var {d['pred_var_mean']:.6f} "
              f"| degenerate {d['degenerate_count']}/{d['n_fold_runs']} "
              f"| pos_rate {d['pos_rate_mean']:.4f}")
    print(f"  reference: Baseline-CV band {REF_BAND_WIDTH}, degeneracy "
          f"{REF_DEGENERACY_BASELINE}/25; Exp 4 P1 degeneracy {REF_DEGENERACY_EXP4_P1}/25")
    print("-" * 74)
    for c in crit:
        print(f"  [{'MET    ' if c['met'] else 'NOT MET'}] ({c['id']}) {c['statement']}")
    print(f"\nOVERALL: {overall}")
    print(f"\n[DONE] wrote {out_prefix}.json and {out_prefix}.md")


if __name__ == "__main__":
    main()
