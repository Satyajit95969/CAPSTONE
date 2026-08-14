#!/usr/bin/env python
"""
Phase 14 / Experiment 5 - aggregation and inference.

Consumes the per-fold artifacts written by run_exp5.py and produces the official
Experiment 5 result: the W0 equivalence gate (HARD ABORT), fold-level aggregates
with Student-t intervals, the PAIRED per-fold comparison against Baseline-CV and
- threshold-conditionally - against Experiment 4, the Criterion-5 discrimination
test, and the Criterion-6 degeneracy comparison.

AGGREGATION CONTRACT
--------------------
Replicates aggregate_baseline_cv.py (L23-34, L67-68) exactly, so Baseline-CV,
Exp 3, Exp 4 and Exp 5 remain mutually comparable:

  1. repeats are averaged WITHIN a fold first  -> one value per fold
  2. the fold - not the fold-run - is the unit of independence
  3. CIs use Student-t with df = k-1 = 4  (t = 2.7764), never a normal quantile

Equivalence is not asserted on trust: main() re-derives Baseline-CV's own
published aggregate from its raw metrics CSVs and checks it against
baseline_cv_summary.json before any Experiment 5 number is computed.

THE W0 EQUIVALENCE GATE - AUTHORITATIVE HARD ABORT
--------------------------------------------------
Experiment 5 is the first experiment in the sequence that must REIMPLEMENT the
training loop: the loss is constructed inside the frozen `fine_tune_supervised`
(trainer_mentalbert_daic.py:312) and the trainer must not be edited. Its
one-factor claim therefore rests on evidence rather than on structure, unlike
Exp 4 (no model loaded) and Exp 3 (`epochs` is a parameter).

W0 (weight=None) exercises the reimplemented loop with the CE weighting removed,
which is by definition the frozen computation. Its aggregate must fall INSIDE
Baseline-CV's 95% CI on every primary metric. If it does not, the loop is not
equivalent and NO W1/W2/W3 result may be reported - this module aborts.

`run_exp5.py` prints an advisory version of this check immediately after
training; the decision authority is here.

ACCEPTANCE CRITERIA (design section 11)
---------------------------------------
  C0  W0 reproduces Baseline-CV                              HARD ABORT gate
  C1  PR-AUC detectably above Baseline-CV 0.3941             roadmap
  C2  Recall and F1 improve over Baseline-CV (0/0)           roadmap
  C3  Recall and F1 improve over Exp 4 (0.5422/0.2215)       roadmap, threshold-conditional
  C4  Regression MAE not materially degraded (MDE 0.4066)    roadmap
  C5  Discrimination, not quantity                           carried forward from Exp 4
  C6  Decisions non-degenerate (vs Exp 4's 24/25)            carried forward from Exp 4

Criterion 5 is load-bearing: Exp 4 proved Recall/F1 can rise with no
discrimination, and Exp 3 proved prediction variance can rise with no accuracy
gain. Without C5, criteria 1-3 could be satisfied a third time by a model that
merely predicts more positives.

Baseline-CV, Exp 3 and Exp 4 are opened READ-ONLY. All output goes to the
Experiment 5 artifact directory.

Usage (from the repository root):
    python exp5_imbalance_objective/aggregate_exp5.py
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
from typing import Dict, List, Optional, Sequence

import numpy as np
import pandas as pd
from scipy import stats

# --------------------------------------------------------------------------
# Frozen constants - must agree with run_exp5.py
# --------------------------------------------------------------------------
ARMS: List[str] = ["W0", "W1", "W2", "W3"]
PRIMARY_ARM = "W1"
CONTROL_ARM = "W0"
SENSITIVITY_ARMS = ["W2", "W3"]

N_REPEATS = 5
N_FOLDS = 5
N_PARTICIPANTS = 188
N_POSITIVE = 45
CORPUS_PREVALENCE = N_POSITIVE / N_PARTICIPANTS      # 0.2393617021276596

PRIMARY_METRICS = ["MAE", "RMSE", "pred_var", "MAE_mean_pred", "ROC_AUC", "PR_AUC",
                   "balAcc", "MCC", "Precision", "Recall", "F1", "Accuracy"]
PAIRED_METRICS = ["MAE", "RMSE", "pred_var", "ROC_AUC", "PR_AUC",
                  "balAcc", "MCC", "Precision", "Recall", "F1", "Accuracy"]

# Baseline-CV MDE anchors (Phase 11.6 section 6).
MDE_MAE_BASELINE = 0.4066
MDE_PR_AUC_BASELINE = 0.0950

# Exp 4 references for Criteria 3 and 6 (Phase 12, policy P1).
EXP4_RECALL = 0.5422
EXP4_F1 = 0.2215
EXP4_DEGENERACY = 24            # of 25 fold-runs
BASELINE_DEGENERACY = 25        # of 25 fold-runs, at argmax

AGG_TOL = 1e-9


def sha256(path: str) -> str:
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
    """Average repeats within each fold - step 1 of the aggregation contract."""
    cols = [c for c in metrics if c in df.columns]
    return df.groupby("fold")[cols].mean().sort_index()


def paired_test(values: Sequence[float], mu0: float = 0.0) -> Dict[str, float]:
    """Paired-fold one-sample t-test against a constant.

    Zero variance is handled explicitly rather than allowed to produce NaN: an
    identical value in every fold is either exactly the reference (no effect) or
    a perfectly consistent offset. Both matter here, because Baseline-CV's
    Recall/F1 are exactly 0 in every fold.
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


def random_ranker_f1(prevalence: float, pos_rate: float) -> float:
    """Expected F1 of a RANDOM ranker predicting `pos_rate` positive on a set
    with the given prevalence: 2*p*r / (p + r). This is the Criterion-5 control
    that distinguishes discrimination from merely predicting more positives.

    MUST be evaluated PER FOLD-RUN using that fold's own prevalence, then
    averaged - never from an averaged pos_rate. The function is non-linear in
    pos_rate, so by Jensen's inequality the two differ substantially: on the
    Exp 4 artifacts, per-fold-run averaging gives 0.211901 (the value recorded in
    exp4_summary.json) while evaluating at the mean pos_rate 0.5316 gives
    0.330089. Experiment 4 computed it per fold-run
    (run_exp4_decision_rule.py:519, `random_ranker_control(fold_prevalence,
    tm["pos_rate"])`), so Experiment 5 must do the same or Criterion 5 would not
    be comparable across the two experiments.
    """
    denom = prevalence + pos_rate
    return float(2.0 * prevalence * pos_rate / denom) if denom > 0 else 0.0


# --------------------------------------------------------------------------
# Self-tests and the authoritative gate
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
    """Prove this module's aggregation matches the Phase 11.6 implementation.

    If it does, any Exp 5 vs Baseline-CV difference is a real difference rather
    than an artefact of two aggregation implementations disagreeing.
    """
    with open(summary_path, "r", encoding="utf-8") as fh:
        published = json.load(fh)
    worst, n = 0.0, 0
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


def w0_equivalence_gate(exp5_df: pd.DataFrame, summary_path: str,
                        out_dir: str) -> Dict:
    """AUTHORITATIVE HARD GATE (design section 10).

    W0 must fall inside Baseline-CV's 95% CI on every primary metric. Because
    Experiment 5 reimplements the training loop, this is the evidence - not the
    assertion - that the reimplementation is equivalent to the frozen one.

    Reproduction is expected to be statistical, not bitwise (torch is unseeded at
    the CUDA level). Experiment 3's E0 in fact reproduced Baseline-CV exactly,
    but Exp 5 runs different code and must not assume that.
    """
    with open(summary_path, "r", encoding="utf-8") as fh:
        published = json.load(fh)["metrics"]

    w0 = exp5_df[exp5_df["arm"] == CONTROL_ARM]
    if len(w0) != N_REPEATS * N_FOLDS:
        abort(f"W0 arm has {len(w0)} fold-runs, expected {N_REPEATS * N_FOLDS}. "
              "The equivalence gate cannot be evaluated, so no arm result may be reported.")

    pf = per_fold_means(w0, PRIMARY_METRICS)
    rows, failures = [], []
    for m in PRIMARY_METRICS:
        if m not in pf.columns or m not in published:
            continue
        got = fold_level_ci(pf[m].values)
        lo, hi = float(published[m]["ci95"][0]), float(published[m]["ci95"][1])
        inside = bool(lo <= got["mean"] <= hi)
        rows.append({"metric": m, "w0_mean": got["mean"], "w0_fold_sd": got["fold_sd"],
                     "baseline_mean": float(published[m]["mean"]),
                     "baseline_ci_lo": lo, "baseline_ci_hi": hi, "inside_ci": inside})
        if not inside:
            failures.append(m)

    report = {"gate": "W0 aggregate inside Baseline-CV 95% CI",
              "authoritative": True, "n_metrics": len(rows),
              "failures": failures, "passed": not failures, "detail": rows}
    with open(os.path.join(out_dir, "w0_equivalence_check.json"), "w", encoding="utf-8") as fh:
        json.dump(report, fh, indent=1)

    if failures:
        for row in rows:
            if not row["inside_ci"]:
                print(f"   W0 {row['metric']}: {row['w0_mean']:.6f} outside "
                      f"[{row['baseline_ci_lo']:.6f}, {row['baseline_ci_hi']:.6f}]",
                      file=sys.stderr)
        abort(f"W0 EQUIVALENCE GATE FAILED on {failures}. The reimplemented training "
              "loop does not reproduce Baseline-CV; no W1/W2/W3 result may be reported.")
    print(f"[PASS] W0 equivalence gate (AUTHORITATIVE): all {len(rows)} primary "
          "metrics inside Baseline-CV's 95% CI")
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
    L.append("# Experiment 5 - Imbalance-Aware Objective: Aggregate Summary\n")
    L.append("- Arms: " + " | ".join(
        f"{a}{' (PRIMARY)' if a == PRIMARY_ARM else ' (control)' if a == CONTROL_ARM else ''}"
        for a in S["arms"]))
    L.append(f"- Repeats R = {S['n_repeats']} | Folds k = {S['n_folds']} | "
             f"fold-runs per arm = {S['n_repeats'] * S['n_folds']}")
    L.append(f"- CIs: fold-level Student-t, df = {S['n_folds'] - 1}, "
             f"t = {S['t_crit']:.4f} (repeats averaged within fold first)")
    L.append("- Single changed factor: the class weighting of the cross-entropy term. "
             "Epochs (3) and the decision rule (argmax) are held at Baseline-CV's "
             "values; Exps 3 and 4 both returned H0 and contributed no accepted change.\n")

    L.append("## Self-tests and gates\n")
    L.append("| Test | Result |")
    L.append("|---|---|")
    L.append(f"| Aggregation reproduces `baseline_cv_summary.json` (max diff "
             f"{S['selftests']['aggregation_max_diff']:.3e}) | PASS |")
    L.append(f"| **W0 equivalence gate (AUTHORITATIVE, hard abort)** - "
             f"{S['w0_gate']['n_metrics']} metrics inside Baseline-CV 95% CI | "
             f"{'PASS' if S['w0_gate']['passed'] else 'FAIL'} |\n")

    L.append("## W0 equivalence detail\n")
    L.append("| Metric | W0 mean | Baseline-CV mean | Baseline-CV 95% CI | Inside |")
    L.append("|---|---|---|---|---|")
    for r in S["w0_gate"]["detail"]:
        L.append(f"| {r['metric']} | {fmt(r['w0_mean'], 6)} | {fmt(r['baseline_mean'], 6)} | "
                 f"[{fmt(r['baseline_ci_lo'], 6)}, {fmt(r['baseline_ci_hi'], 6)}] | "
                 f"{'yes' if r['inside_ci'] else 'NO'} |")
    L.append("")

    L.append("## Class weights applied\n")
    L.append("| Arm | w_neg (mean) | w_pos (mean) | scheme |")
    L.append("|---|---|---|---|")
    for arm in S["arms"]:
        w = S["class_weights"].get(arm, {})
        L.append(f"| {arm} | {fmt(w.get('w_neg_mean'), 6)} | {fmt(w.get('w_pos_mean'), 6)} | "
                 f"{w.get('scheme', '')} |")
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

    L.append(f"## Paired per-fold comparison: {PRIMARY_ARM} vs Baseline-CV\n")
    L.append("Noise band re-estimated from Experiment 5's own fold spread; "
             "Baseline-CV's degenerate MDEs are never inherited.\n")
    L.append("| Metric | Baseline-CV | Exp 5 | mean delta | 95% CI | MDE | Detectable |")
    L.append("|---|---|---|---|---|---|---|")
    for m in PAIRED_METRICS:
        p = S["paired_vs_baseline"].get(m)
        if p is None:
            continue
        L.append(f"| {m} | {fmt(p['baseline_mean'])} | {fmt(p['exp5_mean'])} | "
                 f"{fmt(p['mean_delta'])} | [{fmt(p['ci95_lo'])}, {fmt(p['ci95_hi'])}] | "
                 f"{fmt(p['mde_95'])} | {'YES' if p['detectable'] else 'no'} |")
    L.append("")

    L.append(f"## Threshold-conditional comparison: {PRIMARY_ARM} @ argmax vs Exp 4 @ policy P1\n")
    L.append("**[CAVEAT]** Experiment 4's Recall/F1 were measured at threshold policy P1; "
             "Experiment 5's are measured at argmax. This is NOT a like-for-like "
             "comparison and is reported threshold-conditionally because roadmap "
             "section 6 requires it. The like-for-like comparison is the Baseline-CV "
             "table above.\n")
    L.append("**[NOTE]** Exp 4's PR-AUC is identical to Baseline-CV's (0.394064) because "
             "PR-AUC is invariant to the decision rule. Criterion 1 therefore reduces "
             "to a comparison against Baseline-CV alone.\n")
    L.append("| Metric | Exp 4 (P1) | Exp 5 (argmax) | delta |")
    L.append("|---|---|---|---|")
    for m, ref in [("Recall", EXP4_RECALL), ("F1", EXP4_F1)]:
        e5 = S["aggregates"][PRIMARY_ARM][m]["mean"]
        L.append(f"| {m} | {fmt(ref)} | {fmt(e5)} | {fmt(e5 - ref)} |")
    L.append("")

    c5 = S["criterion5_discrimination"]
    L.append("## Criterion 5 - discrimination, not quantity (carried forward from Exp 4)\n")
    L.append(f"Per-fold precision of **{PRIMARY_ARM}** tested against the corpus "
             f"prevalence floor {CORPUS_PREVALENCE:.10f}, paired-fold, df = 4; and F1 "
             "against the random-ranker control at the same positive rate.\n")
    L.append(f"- per-fold precision: {[fmt(v) for v in c5['per_fold_precision']]}")
    L.append(f"- mean delta vs prevalence: **{fmt(c5['precision_vs_prevalence']['mean_delta'])}** "
             f"[{fmt(c5['precision_vs_prevalence']['ci95_lo'])}, "
             f"{fmt(c5['precision_vs_prevalence']['ci95_hi'])}], "
             f"t({S['n_folds'] - 1}) = {fmt(c5['precision_vs_prevalence']['t_stat'])}, "
             f"p = {fmt(c5['precision_vs_prevalence']['p_value'], 5)}")
    L.append(f"- **precision exceeds prevalence: "
             f"{'YES' if c5['precision_exceeds_prevalence'] else 'NO'}**")
    L.append(f"- Exp 5 F1 {fmt(c5['exp5_f1_mean'])} vs random-ranker F1 "
             f"{fmt(c5['random_ranker_f1_mean'])} at the same positive rate; "
             f"delta {fmt(c5['f1_vs_random']['mean_delta'])} "
             f"[{fmt(c5['f1_vs_random']['ci95_lo'])}, {fmt(c5['f1_vs_random']['ci95_hi'])}]")
    L.append(f"- **F1 exceeds the random-ranker control: "
             f"{'YES' if c5['f1_exceeds_random'] else 'NO'}**\n")

    c6 = S["criterion6_degeneracy"]
    L.append("## Criterion 6 - decisions non-degenerate (carried forward from Exp 4)\n")
    L.append("| Arm | degenerate fold-runs | mean pos_rate | mean p_pos band width |")
    L.append("|---|---|---|---|")
    for arm in S["arms"]:
        d = c6["by_arm"][arm]
        L.append(f"| {arm} | {d['degenerate_count']}/{d['n_fold_runs']} | "
                 f"{fmt(d['pos_rate_mean'])} | {fmt(d['band_width_mean'])} |")
    L.append(f"\nReference points: Baseline-CV **{BASELINE_DEGENERACY}/25** degenerate at "
             f"argmax; Exp 4 policy P1 **{EXP4_DEGENERACY}/25**.")
    L.append(f"- **{PRIMARY_ARM} degeneracy materially below Exp 4's "
             f"{EXP4_DEGENERACY}/25: {'YES' if c6['met'] else 'NO'}**\n")

    L.append("## Acceptance criteria\n")
    L.append("| # | Criterion | Source | Verdict |")
    L.append("|---|---|---|---|")
    for c in S["acceptance_criteria"]:
        L.append(f"| {c['id']} | {c['statement']} | {c['source']} | "
                 f"{'MET' if c['met'] else 'NOT MET'} |")
    L.append(f"\n**Overall: {S['overall_verdict']}**\n")

    L.append("## Sensitivity arms\n")
    L.append("W2 and W3 are pre-registered sensitivity analyses, reported for context. "
             "No headline claim may be drawn from them: selecting an arm after seeing "
             "results would convert a pre-registered experiment into a post-hoc search.\n")
    L.append("| Metric | " + " | ".join(SENSITIVITY_ARMS) + " |")
    L.append("|---" * (len(SENSITIVITY_ARMS) + 1) + "|")
    for m in ["PR_AUC", "ROC_AUC", "Precision", "Recall", "F1", "MAE"]:
        cells = [fmt(S["aggregates"][a][m]["mean"]) for a in SENSITIVITY_ARMS]
        L.append(f"| {m} | " + " | ".join(cells) + " |")
    L.append("")

    L.append("## Notes\n")
    L.append("- Accuracy is reported but is NOT a decision metric (protocol section 3.3); "
             "an all-negative predictor scores the majority rate.")
    L.append("- Criterion 5 is load-bearing: Exp 4 proved Recall/F1 can rise with no "
             "discrimination, and Exp 3 proved prediction variance can rise with no "
             "accuracy gain. Criteria 1-3 alone could be met by predicting more positives.")
    L.append("- k = 5 makes these intervals wide; that width is a property of the design.")
    return "\n".join(L) + "\n"


# --------------------------------------------------------------------------
# Main
# --------------------------------------------------------------------------
def main() -> None:
    ap = argparse.ArgumentParser(
        description="Phase 14 / Experiment 5 - aggregation, W0 gate, paired "
                    "inference against Baseline-CV and Exp 4, Criteria 5 and 6.")
    ap.add_argument("--exp5-dir", default="trainer_outputs/exp5_imbalance_objective")
    ap.add_argument("--baseline-dir", default="trainer_outputs/baseline_cv")
    ap.add_argument("--exp4-dir", default="trainer_outputs/exp4_decision_rule")
    ap.add_argument("--out", default=None,
                    help="output prefix; defaults to <exp5-dir>/exp5_summary")
    a = ap.parse_args()

    exp5_dir = os.path.normpath(a.exp5_dir)
    baseline_dir = os.path.normpath(a.baseline_dir)
    exp4_dir = os.path.normpath(a.exp4_dir)
    runs_dir = os.path.join(baseline_dir, "runs")
    out_prefix = a.out or os.path.join(exp5_dir, "exp5_summary")

    for frozen in [baseline_dir, exp4_dir]:
        if os.path.abspath(out_prefix).startswith(os.path.abspath(frozen) + os.sep):
            abort(f"output prefix {out_prefix} is inside frozen directory {frozen}")

    metrics_path = os.path.join(exp5_dir, "exp5_metrics.csv")
    if not os.path.isfile(metrics_path):
        abort(f"{metrics_path} not found - run run_exp5.py first")

    print("=" * 74)
    print("PHASE 14 / EXPERIMENT 5 - AGGREGATION")
    print(f"  exp5 dir             : {exp5_dir}")
    print(f"  baseline (read-only) : {baseline_dir}")
    print(f"  exp4     (read-only) : {exp4_dir}")
    print("=" * 74)

    exp5_df = pd.read_csv(metrics_path)
    baseline_df = load_baseline_metrics(runs_dir)
    summary_path = os.path.join(baseline_dir, "baseline_cv_summary.json")
    if not os.path.isfile(summary_path):
        abort(f"missing frozen comparator: {summary_path}")
    exp4_summary_path = os.path.join(exp4_dir, "exp4_summary.json")
    if not os.path.isfile(exp4_summary_path):
        abort(f"missing frozen comparator: {exp4_summary_path}")

    shas_before = {p: sha256(p) for p in
                   [summary_path, exp4_summary_path,
                    os.path.join(baseline_dir, "fold_manifest.json"),
                    os.path.join(baseline_dir, "trivial_control_arm.csv")]}

    # ---- self-test, then the authoritative gate --------------------------
    agg_worst = selftest_aggregation(baseline_df, summary_path)
    w0_report = w0_equivalence_gate(exp5_df, summary_path, exp5_dir)

    arms_present = [x for x in ARMS if x in set(exp5_df["arm"].unique())]
    missing = [x for x in ARMS if x not in arms_present]
    if missing:
        abort(f"exp5_metrics.csv is missing arms {missing}")
    for arm in arms_present:
        n = len(exp5_df[exp5_df["arm"] == arm])
        if n != N_REPEATS * N_FOLDS:
            abort(f"arm {arm} has {n} fold-runs, expected {N_REPEATS * N_FOLDS}")

    # ---- aggregates -------------------------------------------------------
    aggregates: Dict[str, Dict[str, Dict[str, float]]] = {}
    per_fold_by_arm: Dict[str, pd.DataFrame] = {}
    for arm in arms_present:
        pf = per_fold_means(exp5_df[exp5_df["arm"] == arm], PRIMARY_METRICS)
        per_fold_by_arm[arm] = pf
        aggregates[arm] = {m: fold_level_ci(pf[m].values) for m in pf.columns}

    # ---- class weights actually applied ----------------------------------
    schemes = {"W0": "uniform (weight=None)", "W1": "balanced n/(2*n_c)",
               "W2": "sqrt balanced", "W3": "fixed 1/prevalence"}
    class_weights: Dict[str, Dict] = {}
    wpath = os.path.join(exp5_dir, "class_weights.csv")
    if os.path.isfile(wpath):
        W = pd.read_csv(wpath)
        for arm in arms_present:
            w = W[W["arm"] == arm]
            class_weights[arm] = {"w_neg_mean": float(w["w_neg"].mean()),
                                  "w_pos_mean": float(w["w_pos"].mean()),
                                  "scheme": schemes.get(arm, "")}
    else:
        for arm in arms_present:
            class_weights[arm] = {"w_neg_mean": None, "w_pos_mean": None,
                                  "scheme": schemes.get(arm, "")}

    # ---- paired comparison: primary arm vs Baseline-CV -------------------
    base_pf = per_fold_means(baseline_df, PAIRED_METRICS)
    exp5_pf = per_fold_by_arm[PRIMARY_ARM]
    paired: Dict[str, Dict] = {}
    for m in PAIRED_METRICS:
        if m not in exp5_pf.columns or m not in base_pf.columns:
            continue
        deltas = exp5_pf[m].values - base_pf[m].values
        paired[m] = {"baseline_mean": float(np.mean(base_pf[m].values)),
                     "exp5_mean": float(np.mean(exp5_pf[m].values)),
                     "per_fold_delta": [float(x) for x in deltas],
                     **paired_test(deltas, 0.0)}

    # ---- Criterion 5: discrimination, not quantity -----------------------
    degen_path = os.path.join(exp5_dir, "degeneracy_report.csv")
    if not os.path.isfile(degen_path):
        abort(f"{degen_path} not found - run run_exp5.py first")
    DG = pd.read_csv(degen_path)

    prec_pf = exp5_pf["Precision"].values
    prec_test = paired_test(prec_pf, CORPUS_PREVALENCE)
    # "Exceeds" requires a positive point estimate AND a CI clearing the floor:
    # a mean above prevalence whose interval straddles it is not evidence.
    prec_exceeds = bool(prec_test["mean_delta"] > 0 and prec_test["ci95_lo"] > 0)

    dg_primary = DG[DG["arm"] == PRIMARY_ARM].copy()
    # Evaluate the control PER FOLD-RUN using that fold's own prevalence, then
    # average within fold - matching run_exp4_decision_rule.py:519 exactly. See
    # random_ranker_f1() for why averaging pos_rate first would be wrong.
    dg_primary["rand_F1"] = [random_ranker_f1(float(p), float(r)) for p, r in
                             zip(dg_primary["prevalence"], dg_primary["pos_rate"])]
    rand_f1_pf = dg_primary.groupby("fold")["rand_F1"].mean().sort_index().values
    pos_rate_pf = dg_primary.groupby("fold")["pos_rate"].mean().sort_index().values
    f1_pf = exp5_pf["F1"].values
    f1_vs_rand = paired_test(f1_pf - rand_f1_pf, 0.0)
    f1_exceeds = bool(f1_vs_rand["mean_delta"] > 0 and f1_vs_rand["ci95_lo"] > 0)

    criterion5 = {
        "per_fold_precision": [float(x) for x in prec_pf],
        "prevalence": CORPUS_PREVALENCE,
        "precision_vs_prevalence": prec_test,
        "precision_exceeds_prevalence": prec_exceeds,
        "exp5_f1_mean": float(np.mean(f1_pf)),
        "random_ranker_f1_mean": float(np.mean(rand_f1_pf)),
        "per_fold_random_f1": [float(x) for x in rand_f1_pf],
        "f1_vs_random": f1_vs_rand,
        "f1_exceeds_random": f1_exceeds,
        "met": bool(prec_exceeds and f1_exceeds),
    }

    # ---- Criterion 6: degeneracy -----------------------------------------
    by_arm = {}
    for arm in arms_present:
        d = DG[DG["arm"] == arm]
        by_arm[arm] = {"n_fold_runs": int(len(d)),
                       "degenerate_count": int(d["degenerate"].sum()),
                       "pos_rate_mean": float(d["pos_rate"].mean()),
                       "band_width_mean": float(d["p_pos_band_width"].mean())}
    primary_degen = by_arm[PRIMARY_ARM]["degenerate_count"]
    criterion6 = {"by_arm": by_arm,
                  "exp4_reference": EXP4_DEGENERACY,
                  "baseline_reference": BASELINE_DEGENERACY,
                  "primary_degenerate_count": primary_degen,
                  "met": bool(primary_degen < EXP4_DEGENERACY)}

    # ---- acceptance criteria ---------------------------------------------
    pr = paired.get("PR_AUC", {})
    c1 = bool(pr.get("mean_delta", 0.0) > 0 and pr.get("detectable", False))
    rec, f1 = paired.get("Recall", {}), paired.get("F1", {})
    c2 = bool(rec.get("mean_delta", 0.0) > 0 and f1.get("mean_delta", 0.0) > 0)
    e5_rec = aggregates[PRIMARY_ARM]["Recall"]["mean"]
    e5_f1 = aggregates[PRIMARY_ARM]["F1"]["mean"]
    c3 = bool(e5_rec > EXP4_RECALL and e5_f1 > EXP4_F1)
    mae = paired.get("MAE", {})
    # Not materially degraded: MAE must not RISE by more than the roadmap anchor.
    c4 = bool(mae.get("mean_delta", 0.0) <= MDE_MAE_BASELINE)

    crit = [
        {"id": 0, "statement": "W0 reproduces Baseline-CV (all primary metrics inside its 95% CI)",
         "source": "design gate", "met": bool(w0_report["passed"])},
        {"id": 1, "statement": f"PR-AUC detectably above Baseline-CV 0.3941",
         "source": "Roadmap section 6 Exp 5", "met": c1},
        {"id": 2, "statement": "Recall and F1 improve over Baseline-CV (0.0000/0.0000)",
         "source": "Roadmap section 6 Exp 5", "met": c2},
        {"id": 3, "statement": f"Recall and F1 improve over Exp 4 "
                               f"({EXP4_RECALL}/{EXP4_F1}) [threshold-conditional]",
         "source": "Roadmap section 6 Exp 5", "met": c3},
        {"id": 4, "statement": f"Regression MAE not materially degraded (MDE {MDE_MAE_BASELINE})",
         "source": "Roadmap section 6 Exp 5", "met": c4},
        {"id": 5, "statement": f"Discrimination, not quantity: precision > prevalence "
                               f"{CORPUS_PREVALENCE:.4f} AND F1 > random-ranker control",
         "source": "carried forward from Exp 4", "met": criterion5["met"]},
        {"id": 6, "statement": f"Decisions non-degenerate (below Exp 4's {EXP4_DEGENERACY}/25)",
         "source": "carried forward from Exp 4", "met": criterion6["met"]},
    ]

    gate_ok = crit[0]["met"]
    primary_met = gate_ok and c1 and criterion5["met"]
    overall = ("H1 SUPPORTED - class weighting produced discrimination" if primary_met
               else "H0 - class weighting relocated the boundary without demonstrated discrimination"
               if gate_ok else "INVALID - W0 equivalence gate failed")

    S = {
        "experiment": "Phase 14 / Exp 5 - imbalance-aware objective",
        "arms": arms_present, "primary_arm": PRIMARY_ARM, "control_arm": CONTROL_ARM,
        "sensitivity_arms": SENSITIVITY_ARMS,
        "n_repeats": N_REPEATS, "n_folds": N_FOLDS,
        "t_crit": float(stats.t.ppf(0.975, N_FOLDS - 1)),
        "ci_method": "fold-level Student-t (df=k-1), repeats averaged within fold",
        "prevalence": CORPUS_PREVALENCE,
        "selftests": {"aggregation_max_diff": float(agg_worst)},
        "w0_gate": w0_report,
        "class_weights": class_weights,
        "aggregates": aggregates,
        "paired_vs_baseline": paired,
        "exp4_reference": {"Recall": EXP4_RECALL, "F1": EXP4_F1,
                           "degeneracy": EXP4_DEGENERACY,
                           "caveat": "Exp 4 metrics were measured at threshold policy P1; "
                                     "Exp 5 at argmax. Not like-for-like; reported "
                                     "threshold-conditionally per roadmap section 6."},
        "criterion5_discrimination": criterion5,
        "criterion6_degeneracy": criterion6,
        "mde_anchors": {"MAE": MDE_MAE_BASELINE, "PR_AUC": MDE_PR_AUC_BASELINE},
        "acceptance_criteria": crit,
        "overall_verdict": overall,
        "frozen_input_sha": shas_before,
    }

    with open(out_prefix + ".json", "w", encoding="utf-8") as fh:
        json.dump(S, fh, indent=1)
    with open(out_prefix + ".md", "w", encoding="utf-8") as fh:
        fh.write(build_markdown(S))

    shas_after = {p: sha256(p) for p in shas_before}
    changed = [k for k in shas_before if shas_before[k] != shas_after[k]]
    if changed:
        abort(f"frozen comparator artifacts changed during aggregation: {changed}")

    # ---- console -----------------------------------------------------------
    print("-" * 74)
    print(f"AGGREGATE - primary arm {PRIMARY_ARM}")
    for m in ["MAE", "pred_var", "ROC_AUC", "PR_AUC", "Precision", "Recall", "F1",
              "balAcc", "MCC", "Accuracy"]:
        a_ = aggregates[PRIMARY_ARM][m]
        print(f"  {m:10s} {a_['mean']:8.4f} +/- {a_['fold_sd']:.4f}  "
              f"CI[{a_['ci95_lo']:7.4f}, {a_['ci95_hi']:7.4f}]")
    print("-" * 74)
    print("PAIRED vs Baseline-CV")
    for m in PAIRED_METRICS:
        p = paired.get(m)
        if p is None:
            continue
        print(f"  {m:10s} base {p['baseline_mean']:8.4f} -> exp5 {p['exp5_mean']:8.4f} "
              f"| delta {p['mean_delta']:+8.4f} MDE {p['mde_95']:.4f} "
              f"| {'DETECTABLE' if p['detectable'] else 'not detectable'}")
    print("-" * 74)
    print(f"CRITERION 5  precision {np.mean(prec_pf):.4f} vs prevalence "
          f"{CORPUS_PREVALENCE:.4f} | delta {prec_test['mean_delta']:+.4f} "
          f"CI[{prec_test['ci95_lo']:.4f}, {prec_test['ci95_hi']:.4f}] "
          f"| exceeds: {prec_exceeds}")
    print(f"             F1 {np.mean(f1_pf):.4f} vs random-ranker "
          f"{np.mean(rand_f1_pf):.4f} | delta {f1_vs_rand['mean_delta']:+.4f} "
          f"| exceeds: {f1_exceeds}")
    print(f"CRITERION 6  {PRIMARY_ARM} degenerate {primary_degen}/25 "
          f"(Exp 4 P1 {EXP4_DEGENERACY}/25, Baseline-CV {BASELINE_DEGENERACY}/25) "
          f"| met: {criterion6['met']}")
    print("-" * 74)
    for c in crit:
        print(f"  [{'MET    ' if c['met'] else 'NOT MET'}] ({c['id']}) {c['statement']}")
    print(f"\nOVERALL: {overall}")
    print(f"\n[DONE] wrote {out_prefix}.json and {out_prefix}.md")


if __name__ == "__main__":
    main()
