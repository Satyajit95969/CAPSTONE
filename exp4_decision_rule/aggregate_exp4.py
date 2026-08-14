#!/usr/bin/env python
"""
Phase 12 / Experiment 4 - aggregation and inference.

Consumes the per-fold artifacts written by run_exp4_decision_rule.py and
produces the official Experiment 4 result: fold-level aggregates with Student-t
confidence intervals, the PAIRED per-fold comparison against Baseline-CV, and
the approved Criterion-3 discrimination test against the corpus prevalence.

AGGREGATION CONTRACT
--------------------
The aggregation replicates aggregate_baseline_cv.py (lines 23-34, 67-68)
exactly, because Exp 4 must be comparable to Baseline-CV metric-for-metric:

  1. repeats are averaged WITHIN a fold first  -> one value per fold
  2. the fold - not the fold-run - is the unit of independence
  3. CIs use Student-t with df = k-1 = 4  (t = 2.7764), never a normal quantile

The R x k = 25 fold-runs are correlated (repeats share data), so pooling them
as if iid would understate uncertainty and produce a falsely tight noise band.
That defect was found and fixed in the Phase 11.6 audit; it is not reintroduced
here. Equivalence is not asserted on trust: main() re-derives Baseline-CV's own
published aggregate from its raw metrics CSVs and checks it against
baseline_cv_summary.json before any Exp 4 number is computed.

NOISE BAND - A NECESSARY DEVIATION
----------------------------------
Baseline-CV's MDE for F1 / balAcc / MCC is exactly 0.0000 because those metrics
were degenerate-constant at the argmax operating point. Phase 11.7 section 5
warns that this "must never be read as 'any change is detectable'". Exp 4
therefore re-estimates its noise band from the SD of its OWN paired per-fold
differences, and never inherits Baseline-CV's degenerate MDE.

Baseline-CV is opened READ-ONLY. All output goes to the Exp 4 artifact
directory. Nothing under trainer_outputs/baseline_cv is written.

Usage (from the repository root):
    python exp4_decision_rule/aggregate_exp4.py
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
from typing import Dict, List, Sequence, Tuple

import numpy as np
import pandas as pd
from scipy import stats

# --------------------------------------------------------------------------
# Frozen constants - must agree with run_exp4_decision_rule.py
# --------------------------------------------------------------------------
N_REPEATS = 5
N_FOLDS = 5
N_PARTICIPANTS = 188
N_POSITIVE = 45
CORPUS_PREVALENCE = N_POSITIVE / N_PARTICIPANTS      # 0.2393617021276596

PRIMARY_POLICY = "P1"
SENSITIVITY_POLICIES = ["P2", "P3"]
BASELINE_ARM = "baseline"

# Threshold-conditional metrics: these are what Exp 4 can legitimately move.
THRESHOLD_METRICS = ["Precision", "Recall", "F1", "balAcc", "MCC", "Accuracy", "pos_rate"]

# Threshold-free / regression metrics: invariant to tau, reported as evidence
# that the model was untouched.
INVARIANT_METRICS = ["ROC_AUC", "PR_AUC", "MAE", "RMSE", "pred_var"]

# Metrics carried into the paired comparison against Baseline-CV.
PAIRED_METRICS = ["Precision", "Recall", "F1", "balAcc", "MCC", "Accuracy"]

AGG_TOL = 1e-9          # tolerance for the aggregation self-test
INVARIANT_TOL = 1e-9    # tolerance for the arm-identity self-test


def sha256(path: str) -> str:
    """SHA-256 of a file, streamed."""
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for block in iter(lambda: fh.read(8192), b""):
            h.update(block)
    return h.hexdigest()


def abort(message: str) -> "NoReturn":  # noqa: F821
    """Stop. A failed gate means the inference is untrustworthy, so we exit
    rather than emit a caveated result that could be quoted out of context."""
    print(f"\n[ABORT] {message}", file=sys.stderr)
    raise SystemExit(1)


# --------------------------------------------------------------------------
# Aggregation primitives
# --------------------------------------------------------------------------
def fold_level_ci(per_fold: Sequence[float]) -> Dict[str, float]:
    """Mean, SD, SE, t-critical and 95% CI from the k per-fold values.

    Replicates aggregate_baseline_cv.py:23-34. The fold is the unit of
    independence: folds are participant-disjoint, so k - not R*k - is the
    effective sample size.
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
    """Average repeats within each fold -> one row per fold.

    This is step 1 of the aggregation contract and must happen before any
    interval is formed.
    """
    return df.groupby("fold")[list(metrics)].mean().sort_index()


def one_sample_test(values: Sequence[float], mu0: float) -> Dict[str, float]:
    """Paired-fold one-sample t-test of `values` against the constant `mu0`.

    Used for (a) the paired delta against Baseline-CV and (b) the Criterion-3
    test against the prevalence floor.

    A zero SD is handled explicitly rather than allowed to yield NaN: if every
    fold gives an identical value, the difference is either exactly zero (no
    effect, p = 1) or a perfectly consistent offset (t unbounded, p -> 0). Both
    are reported honestly rather than silently propagating NaN.
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
        t_stat = float(res.statistic)
        p_val = float(res.pvalue)
        degenerate = False

    mde = tcrit * se
    return {
        "mu0": float(mu0),
        "mean_delta": mean,
        "sd_delta": sd,
        "se": se,
        "t_crit": tcrit,
        "t_stat": t_stat,
        "p_value": p_val,
        "ci95_lo": mean - tcrit * se,
        "ci95_hi": mean + tcrit * se,
        "mde_95": mde,
        "detectable": bool(abs(mean) > mde) if not degenerate else bool(mean != 0.0),
        "zero_variance": degenerate,
    }


# --------------------------------------------------------------------------
# Self-tests - run before any Exp 4 number is produced
# --------------------------------------------------------------------------
def load_baseline_metrics(runs_dir: str) -> pd.DataFrame:
    """Concatenate Baseline-CV's 5 frozen per-repeat metric tables."""
    frames = []
    for r in range(1, N_REPEATS + 1):
        path = os.path.join(runs_dir, f"rep{r}_metrics.csv")
        if not os.path.isfile(path):
            abort(f"missing frozen Baseline-CV metrics file: {path}")
        frames.append(pd.read_csv(path))
    df = pd.concat(frames, ignore_index=True)
    if len(df) != N_REPEATS * N_FOLDS:
        abort(f"Baseline-CV metrics has {len(df)} rows, expected {N_REPEATS * N_FOLDS}")
    return df


def selftest_aggregation(baseline_df: pd.DataFrame, summary_path: str) -> Dict[str, float]:
    """Prove this module's aggregation is equivalent to the Phase 11.6 one.

    Re-derives Baseline-CV's published aggregate from its raw metrics CSVs and
    compares against baseline_cv_summary.json. If they agree, fold_level_ci and
    per_fold_means are demonstrably the same computation that produced the
    frozen comparator - so any Exp 4 vs Baseline-CV difference is a real
    difference, not an artefact of two aggregation implementations.
    """
    with open(summary_path, "r", encoding="utf-8") as fh:
        published = json.load(fh)

    worst: Dict[str, float] = {}
    for metric, ref in published.get("metrics", {}).items():
        if metric not in baseline_df.columns:
            continue
        pf = baseline_df.groupby("fold")[metric].mean().sort_index().values
        got = fold_level_ci(pf)
        deltas = {
            "mean": abs(got["mean"] - float(ref["mean"])),
            "fold_sd": abs(got["fold_sd"] - float(ref["fold_sd"])),
            "se": abs(got["se"] - float(ref["se"])),
            "ci_lo": abs(got["ci95_lo"] - float(ref["ci95"][0])),
            "ci_hi": abs(got["ci95_hi"] - float(ref["ci95"][1])),
            "t_crit": abs(got["t_crit"] - float(ref["t_crit"])),
        }
        worst[metric] = float(max(deltas.values()))
        if worst[metric] > AGG_TOL:
            abort(f"aggregation self-test FAILED on {metric}: max |diff| "
                  f"{worst[metric]:.3e} > {AGG_TOL:.1e}. This module does not "
                  "reproduce the Phase 11.6 aggregation; comparisons would be invalid.")

    if not worst:
        abort("aggregation self-test found no overlapping metrics to check")
    print(f"[PASS] aggregation self-test: reproduces baseline_cv_summary.json "
          f"across {len(worst)} metrics (max |diff| {max(worst.values()):.3e})")
    return worst


def selftest_baseline_arm(exp4_df: pd.DataFrame, baseline_df: pd.DataFrame) -> float:
    """Confirm Exp 4's 'baseline' arm reproduces Baseline-CV fold-for-fold.

    The runner re-derives the argmax rule from p_pos. If that arm does not match
    the frozen values, the re-scoring pipeline cannot be trusted to evaluate any
    other decision rule.
    """
    arm = exp4_df[exp4_df["policy"] == BASELINE_ARM]
    if len(arm) != N_REPEATS * N_FOLDS:
        abort(f"baseline arm has {len(arm)} rows, expected {N_REPEATS * N_FOLDS}")

    merged = arm.merge(baseline_df, on=["repeat", "fold"], suffixes=("_e4", "_bcv"))
    if len(merged) != N_REPEATS * N_FOLDS:
        abort("baseline arm could not be joined 1:1 with Baseline-CV on (repeat, fold)")

    worst = 0.0
    for metric in PAIRED_METRICS + INVARIANT_METRICS:
        a, b = f"{metric}_e4", f"{metric}_bcv"
        if a not in merged.columns or b not in merged.columns:
            continue
        d = float(np.nanmax(np.abs(merged[a].to_numpy(float) - merged[b].to_numpy(float))))
        worst = max(worst, d)
        if d > INVARIANT_TOL:
            abort(f"baseline-arm self-test FAILED on {metric}: max |diff| {d:.3e}")
    print(f"[PASS] baseline-arm self-test: reproduces Baseline-CV fold-for-fold "
          f"(max |diff| {worst:.3e})")
    return worst


def check_invariants_across_arms(exp4_df: pd.DataFrame) -> float:
    """The threshold-free and regression metrics must be identical in EVERY arm.

    They are mathematically independent of tau, so any variation across arms
    would prove the arms were not scored on the same predictions.
    """
    worst = 0.0
    for metric in INVARIANT_METRICS:
        spread = exp4_df.groupby(["repeat", "fold"])[metric].agg(lambda s: s.max() - s.min())
        d = float(np.nanmax(spread.to_numpy(dtype=float)))
        worst = max(worst, d)
        if d > INVARIANT_TOL:
            abort(f"invariant {metric} varies across policy arms by {d:.3e} - "
                  "the arms were not scored on identical predictions")
    print(f"[PASS] tau-invariance: ROC-AUC/PR-AUC/MAE/RMSE/pred_var identical "
          f"across all arms (max spread {worst:.3e})")
    return worst


# --------------------------------------------------------------------------
# Reporting
# --------------------------------------------------------------------------
def fmt(x: float, dp: int = 4) -> str:
    """Format a float for the markdown table, keeping tiny values readable."""
    if x is None or (isinstance(x, float) and np.isnan(x)):
        return "n/a"
    if isinstance(x, float) and x != 0.0 and abs(x) < 10 ** (-dp):
        return f"{x:.2e}"
    return f"{x:.{dp}f}"


def build_markdown(summary: Dict) -> str:
    """Render the human-readable comparator document."""
    L: List[str] = []
    L.append("# Experiment 4 - Decision Rule: Aggregate Summary\n")
    L.append(f"- Repeats R = {summary['n_repeats']} | Folds k = {summary['n_folds']} "
             f"| fold-runs = {summary['n_fold_runs']}")
    L.append(f"- Primary policy: **{PRIMARY_POLICY}** (leave-one-fold-out Youden-J); "
             f"sensitivity: {', '.join(SENSITIVITY_POLICIES)}")
    L.append(f"- CIs: fold-level Student-t, df = k-1 = {summary['n_folds'] - 1}, "
             f"t = {summary['t_crit']:.4f} (repeats averaged within fold first)")
    L.append("- Single changed factor: the decision function. The model was never "
             "loaded, and no training occurred.\n")

    L.append("## Self-tests\n")
    L.append("| Test | Max abs diff | Result |")
    L.append("|---|---|---|")
    L.append(f"| Aggregation reproduces `baseline_cv_summary.json` | "
             f"{summary['selftests']['aggregation_max_diff']:.3e} | PASS |")
    L.append(f"| Baseline arm reproduces Baseline-CV fold-for-fold | "
             f"{summary['selftests']['baseline_arm_max_diff']:.3e} | PASS |")
    L.append(f"| tau-invariant metrics identical across arms | "
             f"{summary['selftests']['invariance_max_spread']:.3e} | PASS |\n")

    L.append("## Aggregate metrics by policy (mean +/- fold SD [95% CI])\n")
    L.append("| Metric | " + " | ".join(summary["policies"]) + " |")
    L.append("|---" * (len(summary["policies"]) + 1) + "|")
    for metric in THRESHOLD_METRICS:
        cells = []
        for pol in summary["policies"]:
            a = summary["aggregates"][pol][metric]
            cells.append(f"{fmt(a['mean'])} +/- {fmt(a['fold_sd'])} "
                         f"[{fmt(a['ci95_lo'])}, {fmt(a['ci95_hi'])}]")
        L.append(f"| {metric} | " + " | ".join(cells) + " |")
    L.append("")

    L.append("## Invariants (must equal Baseline-CV - model untouched)\n")
    L.append("| Metric | Exp 4 (all arms) | Baseline-CV |")
    L.append("|---|---|---|")
    for metric in INVARIANT_METRICS:
        e = summary["invariants"][metric]
        L.append(f"| {metric} | {fmt(e['exp4_mean'], 6)} | {fmt(e['baseline_mean'], 6)} |")
    L.append("")

    L.append(f"## Paired per-fold comparison: {PRIMARY_POLICY} vs Baseline-CV\n")
    L.append("Per-fold differences over the frozen manifest; the noise band is "
             "re-estimated from Exp 4's own fold spread, never inherited from "
             "Baseline-CV's degenerate MDE.\n")
    L.append("| Metric | Baseline-CV | Exp 4 | mean delta | 95% CI | MDE | Detectable |")
    L.append("|---|---|---|---|---|---|---|")
    for metric in PAIRED_METRICS:
        pr = summary["paired_vs_baseline"][metric]
        L.append(f"| {metric} | {fmt(pr['baseline_mean'])} | {fmt(pr['exp4_mean'])} | "
                 f"{fmt(pr['mean_delta'])} | [{fmt(pr['ci95_lo'])}, {fmt(pr['ci95_hi'])}] | "
                 f"{fmt(pr['mde_95'])} | {'YES' if pr['detectable'] else 'no'} |")
    L.append("")

    c3 = summary["criterion3_precision_vs_prevalence"]
    L.append("## Criterion 3 - discrimination test (approved)\n")
    L.append(f"Per-fold precision of **{PRIMARY_POLICY}** tested against the corpus "
             f"prevalence floor {CORPUS_PREVALENCE:.10f}, paired-fold, df = 4.\n")
    L.append(f"- per-fold precision: {[fmt(v) for v in c3['per_fold_precision']]}")
    L.append(f"- mean delta vs prevalence: **{fmt(c3['mean_delta'])}** "
             f"[{fmt(c3['ci95_lo'])}, {fmt(c3['ci95_hi'])}]")
    L.append(f"- t({summary['n_folds'] - 1}) = {fmt(c3['t_stat'], 4)}, "
             f"p = {fmt(c3['p_value'], 5)}")
    L.append(f"- **precision exceeds prevalence: "
             f"{'YES' if c3['exceeds_prevalence'] else 'NO'}**\n")

    rr = summary["random_ranker_control"]
    L.append("## Random-ranker control (equal positive rate)\n")
    L.append(f"- Exp 4 F1 {fmt(rr['exp4_f1_mean'])} vs random-ranker F1 "
             f"{fmt(rr['random_f1_mean'])} at the same positive rate")
    L.append(f"- mean delta {fmt(rr['mean_delta'])} "
             f"[{fmt(rr['ci95_lo'])}, {fmt(rr['ci95_hi'])}], "
             f"detectable: {'YES' if rr['detectable'] else 'no'}\n")

    L.append("## Roadmap success criteria\n")
    L.append("| # | Criterion | Source | Verdict |")
    L.append("|---|---|---|---|")
    for c in summary["success_criteria"]:
        L.append(f"| {c['id']} | {c['statement']} | {c['source']} | "
                 f"{'MET' if c['met'] else 'NOT MET'} |")
    L.append(f"\n**Overall: {summary['overall_verdict']}**\n")

    L.append("## Notes\n")
    L.append("- Accuracy is reported but is NOT a decision metric (protocol section 3.3); "
             "an all-negative predictor scores the majority rate.")
    L.append("- `threshold_sweep.csv` (policy P4) is **descriptive only**. Selecting an "
             "operating point from it after seeing these results would convert a "
             "pre-registered experiment into a post-hoc optimisation, so no headline "
             "number is drawn from it.")
    L.append("- k = 5 makes these intervals wide. That width is a property of the "
             "design, not a defect in the measurement.")
    return "\n".join(L) + "\n"


# --------------------------------------------------------------------------
# Main
# --------------------------------------------------------------------------
def main() -> None:
    ap = argparse.ArgumentParser(
        description="Phase 12 / Experiment 4 - aggregation, paired inference "
                    "against Baseline-CV, and the Criterion-3 discrimination test.")
    ap.add_argument("--exp4-dir", default="trainer_outputs/exp4_decision_rule",
                    help="directory containing exp4_metrics.csv (written by the runner)")
    ap.add_argument("--baseline-dir", default="trainer_outputs/baseline_cv",
                    help="READ-ONLY frozen Baseline-CV directory")
    ap.add_argument("--out", default=None,
                    help="output prefix; defaults to <exp4-dir>/exp4_summary")
    args = ap.parse_args()

    exp4_dir = os.path.normpath(args.exp4_dir)
    baseline_dir = os.path.normpath(args.baseline_dir)
    runs_dir = os.path.join(baseline_dir, "runs")
    out_prefix = args.out or os.path.join(exp4_dir, "exp4_summary")

    # Structural guard: never write into the frozen comparator.
    if os.path.abspath(out_prefix).startswith(os.path.abspath(baseline_dir) + os.sep):
        abort(f"output prefix {out_prefix} is inside the frozen baseline dir")

    metrics_path = os.path.join(exp4_dir, "exp4_metrics.csv")
    if not os.path.isfile(metrics_path):
        abort(f"{metrics_path} not found - run run_exp4_decision_rule.py first")

    print("=" * 74)
    print("PHASE 12 / EXPERIMENT 4 - AGGREGATION")
    print(f"  exp4 dir             : {exp4_dir}")
    print(f"  baseline (read-only) : {baseline_dir}")
    print("=" * 74)

    exp4_df = pd.read_csv(metrics_path)
    baseline_df = load_baseline_metrics(runs_dir)
    summary_path = os.path.join(baseline_dir, "baseline_cv_summary.json")
    if not os.path.isfile(summary_path):
        abort(f"missing frozen comparator: {summary_path}")

    shas_before = {p: sha256(os.path.join(baseline_dir, p))
                   for p in ["baseline_cv_summary.json", "fold_manifest.json",
                             "trivial_control_arm.csv"]}

    # ---- self-tests ------------------------------------------------------
    agg_worst = selftest_aggregation(baseline_df, summary_path)
    arm_worst = selftest_baseline_arm(exp4_df, baseline_df)
    inv_worst = check_invariants_across_arms(exp4_df)

    policies = [BASELINE_ARM, PRIMARY_POLICY] + SENSITIVITY_POLICIES
    present = set(exp4_df["policy"].unique())
    missing = [p for p in policies if p not in present]
    if missing:
        abort(f"exp4_metrics.csv is missing policy arms {missing}")

    # ---- aggregates per policy ------------------------------------------
    aggregates: Dict[str, Dict[str, Dict[str, float]]] = {}
    per_fold_by_policy: Dict[str, pd.DataFrame] = {}
    for pol in policies:
        sub = exp4_df[exp4_df["policy"] == pol]
        cols = [c for c in THRESHOLD_METRICS + ["tau"] if c in sub.columns]
        pf = per_fold_means(sub, cols)
        per_fold_by_policy[pol] = pf
        aggregates[pol] = {m: fold_level_ci(pf[m].values) for m in cols}

    # ---- invariants ------------------------------------------------------
    base_pf_all = per_fold_means(baseline_df, INVARIANT_METRICS)
    exp4_pf_inv = per_fold_means(exp4_df[exp4_df["policy"] == PRIMARY_POLICY],
                                 INVARIANT_METRICS)
    invariants = {
        m: {"exp4_mean": float(np.mean(exp4_pf_inv[m].values)),
            "baseline_mean": float(np.mean(base_pf_all[m].values)),
            "max_abs_diff": float(np.max(np.abs(exp4_pf_inv[m].values
                                                - base_pf_all[m].values)))}
        for m in INVARIANT_METRICS
    }
    for m, v in invariants.items():
        if v["max_abs_diff"] > INVARIANT_TOL:
            abort(f"invariant {m} differs from Baseline-CV by "
                  f"{v['max_abs_diff']:.3e} - training contamination suspected")

    # ---- paired comparison: primary policy vs Baseline-CV ----------------
    base_pf = per_fold_means(baseline_df, PAIRED_METRICS)
    exp4_pf = per_fold_by_policy[PRIMARY_POLICY]
    paired: Dict[str, Dict[str, float]] = {}
    for m in PAIRED_METRICS:
        deltas = exp4_pf[m].values - base_pf[m].values
        test = one_sample_test(deltas, 0.0)
        paired[m] = {
            "baseline_mean": float(np.mean(base_pf[m].values)),
            "exp4_mean": float(np.mean(exp4_pf[m].values)),
            "per_fold_delta": [float(x) for x in deltas],
            **test,
        }

    # ---- Criterion 3: precision vs prevalence (paired-fold) --------------
    prec_pf = exp4_pf["Precision"].values
    c3 = one_sample_test(prec_pf, CORPUS_PREVALENCE)
    c3["per_fold_precision"] = [float(x) for x in prec_pf]
    c3["prevalence"] = CORPUS_PREVALENCE
    # "Exceeds" requires both a positive point estimate and a CI that clears the
    # floor - a mean above prevalence whose interval straddles it is not evidence.
    c3["exceeds_prevalence"] = bool(c3["mean_delta"] > 0 and c3["ci95_lo"] > 0)

    # ---- random-ranker control at matched positive rate ------------------
    sub_primary = exp4_df[exp4_df["policy"] == PRIMARY_POLICY]
    rr_pf = per_fold_means(sub_primary, ["F1", "rand_F1"])
    rr_delta = rr_pf["F1"].values - rr_pf["rand_F1"].values
    rr_test = one_sample_test(rr_delta, 0.0)
    random_ranker = {
        "exp4_f1_mean": float(np.mean(rr_pf["F1"].values)),
        "random_f1_mean": float(np.mean(rr_pf["rand_F1"].values)),
        "per_fold_delta": [float(x) for x in rr_delta],
        **rr_test,
    }

    # ---- success criteria -------------------------------------------------
    inv_path = os.path.join(exp4_dir, "invariants_check.json")
    regression_unchanged = False
    if os.path.isfile(inv_path):
        with open(inv_path, "r", encoding="utf-8") as fh:
            regression_unchanged = bool(json.load(fh).get("all_within_tolerance", False))

    crit = [
        {"id": 1,
         "statement": "Recall > 0 and F1 > 0",
         "source": "Roadmap section 6 Exp 4",
         "met": bool(aggregates[PRIMARY_POLICY]["Recall"]["mean"] > 0
                     and aggregates[PRIMARY_POLICY]["F1"]["mean"] > 0)},
        {"id": 2,
         "statement": "Regression MAE unchanged (no training occurred)",
         "source": "Roadmap section 6 Exp 4",
         "met": bool(regression_unchanged
                     and invariants["MAE"]["max_abs_diff"] <= INVARIANT_TOL)},
        {"id": 3,
         "statement": f"Precision exceeds prevalence {CORPUS_PREVALENCE:.4f} "
                      "(paired-fold, CI clears the floor)",
         "source": "approved addition",
         "met": bool(c3["exceeds_prevalence"])},
        {"id": 4,
         "statement": "balAcc > 0.5 and MCC > 0 beyond Exp 4's own noise band",
         "source": "approved addition",
         "met": bool(paired["balAcc"]["mean_delta"] > 0 and paired["balAcc"]["detectable"]
                     and paired["MCC"]["mean_delta"] > 0 and paired["MCC"]["detectable"])},
    ]
    primary_met = crit[0]["met"] and crit[1]["met"] and crit[2]["met"]
    overall = ("H1 SUPPORTED - discrimination demonstrated" if primary_met
               else "H0 - thresholding recovered quantity without demonstrated discrimination"
               if crit[0]["met"] and crit[1]["met"]
               else "INCONCLUSIVE - review criteria individually")

    summary = {
        "experiment": "Phase 12 / Exp 4 - decision rule",
        "n_repeats": N_REPEATS, "n_folds": N_FOLDS,
        "n_fold_runs": int(N_REPEATS * N_FOLDS),
        "t_crit": float(stats.t.ppf(0.975, N_FOLDS - 1)),
        "ci_method": "fold-level Student-t (df=k-1), repeats averaged within fold",
        "primary_policy": PRIMARY_POLICY,
        "sensitivity_policies": SENSITIVITY_POLICIES,
        "policies": policies,
        "prevalence": CORPUS_PREVALENCE,
        "selftests": {"aggregation_max_diff": float(max(agg_worst.values())),
                      "baseline_arm_max_diff": float(arm_worst),
                      "invariance_max_spread": float(inv_worst)},
        "aggregates": aggregates,
        "invariants": invariants,
        "paired_vs_baseline": paired,
        "criterion3_precision_vs_prevalence": c3,
        "random_ranker_control": random_ranker,
        "success_criteria": crit,
        "overall_verdict": overall,
        "baseline_input_sha": shas_before,
    }

    # ---- write + verify inputs untouched ---------------------------------
    with open(out_prefix + ".json", "w", encoding="utf-8") as fh:
        json.dump(summary, fh, indent=1)
    with open(out_prefix + ".md", "w", encoding="utf-8") as fh:
        fh.write(build_markdown(summary))

    shas_after = {p: sha256(os.path.join(baseline_dir, p)) for p in shas_before}
    changed = [k for k in shas_before if shas_before[k] != shas_after[k]]
    if changed:
        abort(f"frozen Baseline-CV inputs changed during aggregation: {changed}")

    # ---- console -----------------------------------------------------------
    print("-" * 74)
    print(f"AGGREGATE - primary policy {PRIMARY_POLICY}")
    for m in THRESHOLD_METRICS:
        a = aggregates[PRIMARY_POLICY][m]
        print(f"  {m:10s} {a['mean']:8.4f} +/- {a['fold_sd']:.4f}  "
              f"CI[{a['ci95_lo']:7.4f}, {a['ci95_hi']:7.4f}]")
    print("-" * 74)
    print("PAIRED vs Baseline-CV")
    for m in PAIRED_METRICS:
        p = paired[m]
        print(f"  {m:10s} base {p['baseline_mean']:7.4f} -> exp4 {p['exp4_mean']:7.4f} "
              f"| delta {p['mean_delta']:+7.4f} MDE {p['mde_95']:.4f} "
              f"| {'DETECTABLE' if p['detectable'] else 'not detectable'}")
    print("-" * 74)
    print(f"CRITERION 3  precision {np.mean(prec_pf):.4f} vs prevalence "
          f"{CORPUS_PREVALENCE:.4f} | delta {c3['mean_delta']:+.4f} "
          f"CI[{c3['ci95_lo']:.4f}, {c3['ci95_hi']:.4f}] "
          f"| exceeds: {c3['exceeds_prevalence']}")
    print("-" * 74)
    for c in crit:
        print(f"  [{'MET    ' if c['met'] else 'NOT MET'}] ({c['id']}) {c['statement']}")
    print(f"\nOVERALL: {overall}")
    print(f"\n[DONE] wrote {out_prefix}.json and {out_prefix}.md")


if __name__ == "__main__":
    main()
