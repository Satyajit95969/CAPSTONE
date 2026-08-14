#!/usr/bin/env python
"""
Phase 15 / Experiment 6 - aggregation and inference.

Consumes the per-fold artifacts written by run_exp6.py and produces the official
Experiment 6 result: the L0 equivalence gate (HARD ABORT), fold-level aggregates
with Student-t intervals, the PAIRED per-fold comparison against Baseline-CV and
against Experiment 5's W1 arm, the Criterion-5 discrimination test, the
Criterion-6 degeneracy comparison, and the OBSERVATIONAL loss/gradient
decomposition.

AGGREGATION CONTRACT
--------------------
Replicates aggregate_baseline_cv.py (L23-34, L67-68) exactly, so Baseline-CV,
Exp 3, Exp 4, Exp 5 and Exp 6 remain mutually comparable:

  1. repeats are averaged WITHIN a fold first  -> one value per fold
  2. the fold - not the fold-run - is the unit of independence
  3. CIs use Student-t with df = k-1 = 4  (t = 2.7764), never a normal quantile

Equivalence is not asserted on trust: main() re-derives Baseline-CV's own
published aggregate from its raw metrics CSVs and checks it against
baseline_cv_summary.json before any Experiment 6 number is computed.

THE L0 EQUIVALENCE GATE - AUTHORITATIVE HARD ABORT
---------------------------------------------------
Like Experiment 5, Experiment 6 must REIMPLEMENT the training loop: the
coefficient is a literal inside the frozen `fine_tune_supervised`
(trainer_mentalbert_daic.py:324) and the trainer must not be edited. Its
one-factor claim therefore rests on evidence rather than on structure.

L0 (LAMBDA = 0.5) exercises the reimplemented loop at the frozen coefficient,
which is by definition the frozen computation. Its aggregate must fall INSIDE
Baseline-CV's 95% CI on every primary metric. If it does not, the loop is not
equivalent and NO L1/L2/L3 result may be reported - this module aborts.

`run_exp6.py` prints an advisory version of this check immediately after
training; the decision authority is here.

ACCEPTANCE CRITERIA (design section 11) - EVALUATED ON L1 ONLY
---------------------------------------------------------------
  C0  L0 reproduces Baseline-CV                              HARD ABORT gate
  C1  PR-AUC detectably above Baseline-CV 0.3941             roadmap
  C2  Recall and F1 improve over Baseline-CV (0/0)           roadmap
  C3  Recall and F1 improve over Exp 5 W1 (0.0756/0.0295)    roadmap comparator
  C4  Regression MAE not materially degraded (MDE 0.4066)    roadmap
  C5  Discrimination, not quantity                           carried forward from Exps 4, 5
  C6  Decisions non-degenerate (vs Exp 5 W1's 24/25)         carried forward from Exps 4, 5

Design section 11 is explicit that C1-C6 are evaluated on the PRIMARY arm L1;
L2 and L3 are sensitivity arms and do not determine the verdict.

Criterion 5 is load-bearing: Exp 4 proved Recall/F1 can rise with no
discrimination, Exp 5 proved the same at argmax, and Exp 3 proved prediction
variance can rise with no accuracy gain. Without C5, criteria 1-3 could be
satisfied a fourth time by a model that merely predicts more positives.

OBSERVATIONAL ANALYSES - CANNOT AFFECT THE VERDICT
---------------------------------------------------
Two analyses are reported in full but are explicitly NOT acceptance criteria:

  * the loss/gradient decomposition (design section 15.1), which measures the
    encoder gradient each loss term contributes - the quantity design section 7.1
    notes was inferred from loss magnitudes rather than measured;
  * the lambda-ordering analysis (design section 11.1-M), demoted from a proposed
    "C7" because with 4 arms one specific ordering arises by chance with
    probability 1/24 ~ 4.2%.

Neither is consulted when computing `overall_verdict`.

Baseline-CV, Exp 3, Exp 4 and Exp 5 are opened READ-ONLY. All output goes to the
Experiment 6 artifact directory.

Usage (from the repository root):
    python exp6_loss_rebalance/aggregate_exp6.py
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
# Frozen constants - must agree with run_exp6.py
# --------------------------------------------------------------------------
ARM_LAMBDA: Dict[str, float] = {"L0": 0.5, "L1": 0.0144, "L2": 0.1, "L3": 0.0}
ARMS: List[str] = ["L0", "L1", "L2", "L3"]
PRIMARY_ARM = "L1"
CONTROL_ARM = "L0"
SENSITIVITY_ARMS = ["L2", "L3"]
ZERO_LAMBDA_ARM = "L3"          # regression supervision disabled

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
BASELINE_MEAN_PREDICTOR_MAE = 4.9135    # Exp-0a bound, Criterion 4

# Exp 5 references for Criteria 3 and 6 (Phase 14, arm W1, measured AT ARGMAX).
EXP5_ARM = "W1"
EXP5_RECALL = 0.0756
EXP5_F1 = 0.0295
EXP5_DEGENERACY = 24            # of 25 fold-runs
BASELINE_DEGENERACY = 25        # of 25 fold-runs, at argmax

AGG_TOL = 1e-9
EXP5_CONSISTENCY_TOL = 1e-6     # re-derived Exp 5 aggregate vs its gated summary


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
# Aggregation primitives - IDENTICAL to aggregate_exp5.py
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
    0.330089. Experiments 4 and 5 both computed it per fold-run, so Experiment 6
    must do the same or Criterion 5 would not be comparable across them.
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

    If it does, any Exp 6 vs Baseline-CV difference is a real difference rather
    than an artefact of two aggregation implementations disagreeing. Runs BEFORE
    any Experiment 6 number is computed.
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


def load_exp5_primary(exp5_dir: str) -> pd.DataFrame:
    """Per-fold means of Exp 5's W1 arm, validated against its SHA-gated summary.

    C3 and the Exp 5 paired comparison need PER-FOLD values, which live in
    exp5_metrics.csv - a file the pre-registration does not SHA-gate (A.1 gates
    exp5_summary.json, the file C3's constants are quoted from). Rather than
    trust the CSV, selftest_exp5_consistency() re-derives its aggregate and
    checks it against the gated summary: if the CSV had been altered, the two
    would disagree. The gated artifact validates the ungated one.
    """
    p = os.path.join(exp5_dir, "exp5_metrics.csv")
    if not os.path.isfile(p):
        abort(f"missing frozen Exp 5 comparator: {p} (needed for C3 and the "
              "paired Exp 5 comparison)")
    df = pd.read_csv(p)
    w1 = df[df["arm"] == EXP5_ARM]
    if len(w1) != N_REPEATS * N_FOLDS:
        abort(f"Exp 5 arm {EXP5_ARM} has {len(w1)} fold-runs, expected "
              f"{N_REPEATS * N_FOLDS}")
    return per_fold_means(w1, PAIRED_METRICS)


def selftest_exp5_consistency(exp5_pf: pd.DataFrame, exp5_summary_path: str) -> float:
    """Validate the ungated exp5_metrics.csv against the SHA-gated exp5_summary.json.

    Also confirms the frozen C3 constants (0.0756 / 0.0295) are the values that
    Exp 5 actually recorded, so a transcription slip in this module's constants
    cannot silently change a criterion.
    """
    with open(exp5_summary_path, "r", encoding="utf-8") as fh:
        pub = json.load(fh)["aggregates"][EXP5_ARM]
    worst, n = 0.0, 0
    for m in PAIRED_METRICS:
        if m not in exp5_pf.columns or m not in pub:
            continue
        got = fold_level_ci(exp5_pf[m].values)
        d = max(abs(got["mean"] - float(pub[m]["mean"])),
                abs(got["fold_sd"] - float(pub[m]["fold_sd"])))
        worst = max(worst, d)
        n += 1
    if worst > EXP5_CONSISTENCY_TOL:
        abort(f"Exp 5 consistency self-test FAILED: exp5_metrics.csv re-derives to an "
              f"aggregate {worst:.3e} from the SHA-gated exp5_summary.json. One of the "
              "two frozen Exp 5 artifacts has been altered; refusing to proceed.")
    for name, const, pubval in [("Recall", EXP5_RECALL, float(pub["Recall"]["mean"])),
                                ("F1", EXP5_F1, float(pub["F1"]["mean"]))]:
        if abs(const - round(pubval, 4)) > 1e-9:
            abort(f"C3 constant EXP5_{name.upper()} = {const} does not match Exp 5's "
                  f"recorded {name} {pubval:.6f} (4 d.p. {round(pubval, 4)}).")
    print(f"[PASS] Exp 5 consistency self-test: exp5_metrics.csv agrees with the "
          f"SHA-gated exp5_summary.json across {n} metrics (max |diff| {worst:.3e}); "
          "C3 constants confirmed")
    return worst


def l0_equivalence_gate(exp6_df: pd.DataFrame, summary_path: str,
                        out_dir: str) -> Dict:
    """AUTHORITATIVE HARD GATE (design section 10).

    L0 must fall inside Baseline-CV's 95% CI on every primary metric. Because
    Experiment 6 reimplements the training loop, this is the evidence - not the
    assertion - that the reimplementation is equivalent to the frozen one, and it
    is what makes the uniform code path of design section 10.1 transitive: L0
    certifies the identical statements that L1, L2 and L3 execute.

    Reproduction is expected to be statistical, not bitwise (torch is unseeded at
    the CUDA level). Experiment 3's E0 and Experiment 5's W0 both in fact
    reproduced Baseline-CV exactly, but Exp 6 runs different code and must not
    assume that.
    """
    with open(summary_path, "r", encoding="utf-8") as fh:
        published = json.load(fh)["metrics"]

    l0 = exp6_df[exp6_df["arm"] == CONTROL_ARM]
    if len(l0) != N_REPEATS * N_FOLDS:
        abort(f"L0 arm has {len(l0)} fold-runs, expected {N_REPEATS * N_FOLDS}. "
              "The equivalence gate cannot be evaluated, so no arm result may be reported.")

    pf = per_fold_means(l0, PRIMARY_METRICS)
    rows, failures = [], []
    for m in PRIMARY_METRICS:
        if m not in pf.columns or m not in published:
            continue
        got = fold_level_ci(pf[m].values)
        lo, hi = float(published[m]["ci95"][0]), float(published[m]["ci95"][1])
        inside = bool(lo <= got["mean"] <= hi)
        rows.append({"metric": m, "l0_mean": got["mean"], "l0_fold_sd": got["fold_sd"],
                     "baseline_mean": float(published[m]["mean"]),
                     "baseline_ci_lo": lo, "baseline_ci_hi": hi, "inside_ci": inside})
        if not inside:
            failures.append(m)

    report = {"gate": "L0 aggregate inside Baseline-CV 95% CI",
              "authoritative": True, "n_metrics": len(rows),
              "control_lambda": ARM_LAMBDA[CONTROL_ARM],
              "failures": failures, "passed": not failures, "detail": rows}
    with open(os.path.join(out_dir, "l0_equivalence_check.json"), "w", encoding="utf-8") as fh:
        json.dump(report, fh, indent=1)

    if failures:
        for row in rows:
            if not row["inside_ci"]:
                print(f"   L0 {row['metric']}: {row['l0_mean']:.6f} outside "
                      f"[{row['baseline_ci_lo']:.6f}, {row['baseline_ci_hi']:.6f}]",
                      file=sys.stderr)
        abort(f"L0 EQUIVALENCE GATE FAILED on {failures}. The reimplemented training "
              "loop does not reproduce Baseline-CV; no L1/L2/L3 result may be reported.")
    print(f"[PASS] L0 equivalence gate (AUTHORITATIVE): all {len(rows)} primary "
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
    L.append("# Experiment 6 - Loss-Term Rebalancing: Aggregate Summary\n")
    L.append("- Arms: " + " | ".join(
        f"{a} (lambda={ARM_LAMBDA[a]})"
        f"{' PRIMARY' if a == PRIMARY_ARM else ' control' if a == CONTROL_ARM else ''}"
        for a in S["arms"]))
    L.append(f"- Repeats R = {S['n_repeats']} | Folds k = {S['n_folds']} | "
             f"fold-runs per arm = {S['n_repeats'] * S['n_folds']}")
    L.append(f"- CIs: fold-level Student-t, df = {S['n_folds'] - 1}, "
             f"t = {S['t_crit']:.4f} (repeats averaged within fold first)")
    L.append("- Single changed factor: the regression coefficient lambda. Epochs (3), "
             "the decision rule (argmax) and the unweighted CE term are held at "
             "Baseline-CV's values; Exps 3, 4 and 5 all returned H0 and contributed "
             "no accepted change.")
    L.append(f"- **Criteria C1-C6 are evaluated on the primary arm {PRIMARY_ARM} only** "
             f"(design section 11); {' and '.join(SENSITIVITY_ARMS)} are sensitivity arms.\n")

    L.append("## Self-tests and gates\n")
    L.append("| Test | Result |")
    L.append("|---|---|")
    L.append(f"| Aggregation reproduces `baseline_cv_summary.json` (max diff "
             f"{S['selftests']['aggregation_max_diff']:.3e}) | PASS |")
    L.append(f"| `exp5_metrics.csv` agrees with SHA-gated `exp5_summary.json` (max diff "
             f"{S['selftests']['exp5_consistency_max_diff']:.3e}) | PASS |")
    L.append(f"| **L0 equivalence gate (AUTHORITATIVE, hard abort)** - "
             f"{S['l0_gate']['n_metrics']} metrics inside Baseline-CV 95% CI | "
             f"{'PASS' if S['l0_gate']['passed'] else 'FAIL'} |\n")

    L.append("## L0 equivalence detail\n")
    L.append("| Metric | L0 mean | Baseline-CV mean | Baseline-CV 95% CI | Inside |")
    L.append("|---|---|---|---|---|")
    for r in S["l0_gate"]["detail"]:
        L.append(f"| {r['metric']} | {fmt(r['l0_mean'], 6)} | {fmt(r['baseline_mean'], 6)} | "
                 f"[{fmt(r['baseline_ci_lo'], 6)}, {fmt(r['baseline_ci_hi'], 6)}] | "
                 f"{'yes' if r['inside_ci'] else 'NO'} |")
    L.append("")

    L.append("## Coefficient applied\n")
    L.append("| Arm | lambda | Basis | Role |")
    L.append("|---|---|---|---|")
    for arm in S["arms"]:
        c = S["coefficients"][arm]
        L.append(f"| {arm} | {c['lambda']} | {c['basis']} | {c['role']} |")
    L.append("")

    L.append("## Aggregate metrics by arm (mean +/- fold SD [95% CI])\n")
    L.append(f"**[CAVEAT]** Arm {ZERO_LAMBDA_ARM} sets lambda = 0, so the `phq_mu` head "
             "receives NO supervision. Its MAE, RMSE and pred_var characterise an "
             "UNSUPERVISED head and are descriptive only - they are not regression "
             "performance (design section 11.0).\n")
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
    L.append("Noise band re-estimated from Experiment 6's own fold spread; "
             "Baseline-CV's degenerate MDEs are never inherited.\n")
    L.append("| Metric | Baseline-CV | Exp 6 | mean delta | 95% CI | MDE | Detectable |")
    L.append("|---|---|---|---|---|---|---|")
    for m in PAIRED_METRICS:
        p = S["paired_vs_baseline"].get(m)
        if p is None:
            continue
        L.append(f"| {m} | {fmt(p['baseline_mean'])} | {fmt(p['exp6_mean'])} | "
                 f"{fmt(p['mean_delta'])} | [{fmt(p['ci95_lo'])}, {fmt(p['ci95_hi'])}] | "
                 f"{fmt(p['mde_95'])} | {'YES' if p['detectable'] else 'no'} |")
    L.append("")

    L.append(f"## Paired per-fold comparison: {PRIMARY_ARM} vs Exp 5 {EXP5_ARM}\n")
    L.append("**[LIKE-FOR-LIKE]** Unlike Experiment 5's comparison against Exp 4, this one "
             "needs no threshold caveat: Exp 5 was measured at argmax and 3 epochs, "
             "identical to Experiment 6. Both are paired on the same frozen folds.\n")
    L.append("| Metric | Exp 5 W1 | Exp 6 | mean delta | 95% CI | MDE | Detectable |")
    L.append("|---|---|---|---|---|---|---|")
    for m in PAIRED_METRICS:
        p = S["paired_vs_exp5"].get(m)
        if p is None:
            continue
        L.append(f"| {m} | {fmt(p['exp5_mean'])} | {fmt(p['exp6_mean'])} | "
                 f"{fmt(p['mean_delta'])} | [{fmt(p['ci95_lo'])}, {fmt(p['ci95_hi'])}] | "
                 f"{fmt(p['mde_95'])} | {'YES' if p['detectable'] else 'no'} |")
    L.append("")

    c5 = S["criterion5_discrimination"]
    L.append("## Criterion 5 - discrimination, not quantity (carried forward from Exps 4, 5)\n")
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
    L.append(f"- Exp 6 F1 {fmt(c5['exp6_f1_mean'])} vs random-ranker F1 "
             f"{fmt(c5['random_ranker_f1_mean'])} at the same positive rate; "
             f"delta {fmt(c5['f1_vs_random']['mean_delta'])} "
             f"[{fmt(c5['f1_vs_random']['ci95_lo'])}, {fmt(c5['f1_vs_random']['ci95_hi'])}]")
    L.append(f"- **F1 exceeds the random-ranker control: "
             f"{'YES' if c5['f1_exceeds_random'] else 'NO'}**\n")

    c6 = S["criterion6_degeneracy"]
    L.append("## Criterion 6 - decisions non-degenerate (carried forward from Exps 4, 5)\n")
    L.append("| Arm | lambda | degenerate fold-runs | mean pos_rate | mean p_pos band width |")
    L.append("|---|---|---|---|---|")
    for arm in S["arms"]:
        d = c6["by_arm"][arm]
        L.append(f"| {arm} | {ARM_LAMBDA[arm]} | {d['degenerate_count']}/{d['n_fold_runs']} | "
                 f"{fmt(d['pos_rate_mean'])} | {fmt(d['band_width_mean'])} |")
    L.append(f"\nReference points: Baseline-CV **{BASELINE_DEGENERACY}/25** degenerate at "
             f"argmax; Exp 5 {EXP5_ARM} **{EXP5_DEGENERACY}/25**; Exp 4 policy P1 24/25.")
    L.append(f"- **{PRIMARY_ARM} degeneracy below Exp 5's "
             f"{EXP5_DEGENERACY}/25: {'YES' if c6['met'] else 'NO'}**\n")

    L.append("## Acceptance criteria\n")
    L.append(f"Evaluated on the primary arm {PRIMARY_ARM} (C0 on {CONTROL_ARM}).\n")
    L.append("| # | Criterion | Source | Verdict |")
    L.append("|---|---|---|---|")
    for c in S["acceptance_criteria"]:
        L.append(f"| {c['id']} | {c['statement']} | {c['source']} | "
                 f"{'MET' if c['met'] else 'NOT MET'} |")
    L.append(f"\n**Overall: {S['overall_verdict']}**\n")

    # ---- observational blocks ------------------------------------------------
    ld = S["observational_loss_decomposition"]
    L.append("## OBSERVATIONAL - loss and gradient decomposition\n")
    L.append("**These measurements are NOT acceptance criteria and did not contribute to "
             "the verdict above** (design section 15.1). They exist because design "
             "section 7.1 records that lambda = 0.0144 is LOSS-VALUE parity, not "
             "GRADIENT parity: `lambda*MSE = CE` does not imply "
             "`||grad(lambda*MSE)|| = ||grad(CE)||`. The columns below measure the "
             "encoder gradient directly rather than inferring it from loss magnitudes.\n")
    L.append("| Arm | lambda | loss_cls | lambda*loss_reg | loss ratio | "
             "\\|grad cls\\| | \\|grad reg\\| | grad ratio |")
    L.append("|---|---|---|---|---|---|---|---|")
    for arm in S["arms"]:
        d = ld["by_arm"].get(arm)
        if d is None:
            continue
        L.append(f"| {arm} | {ARM_LAMBDA[arm]} | {fmt(d['loss_cls'], 6)} | "
                 f"{fmt(d['lambda_loss_reg'], 6)} | {fmt(d['ratio'], 4)} | "
                 f"{fmt(d['grad_norm_cls'], 6)} | {fmt(d['grad_norm_reg'], 6)} | "
                 f"{fmt(d['grad_norm_ratio'], 4)} |")
    L.append(f"\n- Design section 4 estimated the loss ratio at Baseline-CV's operating "
             f"point as **34.8 : 1** from frozen artifacts; the {CONTROL_ARM} row above is "
             "the measured value at the same coefficient.")
    L.append(f"- **{ZERO_LAMBDA_ARM} zero-reference:** grad_norm_reg = "
             f"{fmt(ld['zero_reference_grad_norm_reg'], 8)} "
             f"(exactly 0 expected; instrument check).\n")

    mo = S["observational_lambda_ordering"]
    L.append("## OBSERVATIONAL - lambda-ordering analysis\n")
    L.append("**NOT an acceptance criterion** (design section 11.1-M). Proposed as \"C7\" "
             "in an earlier draft and demoted before execution: with 4 arms one specific "
             "ordering arises by chance with probability 1/24 ~ 4.2%, and k = 4 affords no "
             "power for a formal trend test. Reported because it is mechanistically "
             "informative, never as pass/fail.\n")
    L.append("Arms in order of DECREASING lambda:\n")
    L.append("| Metric | " + " | ".join(f"{a} ({ARM_LAMBDA[a]})" for a in mo["arm_order"])
             + " | monotone non-decreasing |")
    L.append("|---" * (len(mo["arm_order"]) + 2) + "|")
    for m, row in mo["by_metric"].items():
        cells = " | ".join(fmt(v) for v in row["values"])
        L.append(f"| {m} | {cells} | {'yes' if row['monotone_non_decreasing'] else 'no'} |")
    L.append("")

    L.append("## Sensitivity arms\n")
    L.append(f"{' and '.join(SENSITIVITY_ARMS)} are pre-registered sensitivity analyses, "
             "reported for context. No headline claim may be drawn from them: selecting "
             "an arm after seeing results would convert a pre-registered experiment into "
             "a post-hoc search.\n")
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
             "discrimination, Exp 5 proved the same at argmax, and Exp 3 proved prediction "
             "variance can rise with no accuracy gain. Criteria 1-3 alone could be met by "
             "predicting more positives.")
    L.append(f"- `phq_logsigma` is untrained in every arm and in Baseline-CV; this is "
             "inherited from the frozen trainer, not introduced by Experiment 6 "
             "(design section 10.2).")
    L.append("- k = 5 makes these intervals wide; that width is a property of the design.")
    return "\n".join(L) + "\n"


# --------------------------------------------------------------------------
# Main
# --------------------------------------------------------------------------
def main() -> None:
    ap = argparse.ArgumentParser(
        description="Phase 15 / Experiment 6 - aggregation, L0 gate, paired "
                    "inference against Baseline-CV and Exp 5, Criteria 5 and 6.")
    ap.add_argument("--exp6-dir", default="trainer_outputs/exp6_loss_rebalance")
    ap.add_argument("--baseline-dir", default="trainer_outputs/baseline_cv")
    ap.add_argument("--exp5-dir", default="trainer_outputs/exp5_imbalance_objective")
    ap.add_argument("--out", default=None,
                    help="output prefix; defaults to <exp6-dir>/exp6_summary")
    a = ap.parse_args()

    exp6_dir = os.path.normpath(a.exp6_dir)
    baseline_dir = os.path.normpath(a.baseline_dir)
    exp5_dir = os.path.normpath(a.exp5_dir)
    runs_dir = os.path.join(baseline_dir, "runs")
    out_prefix = a.out or os.path.join(exp6_dir, "exp6_summary")

    for frozen in [baseline_dir, exp5_dir, "trainer_outputs/exp4_decision_rule",
                   "trainer_outputs/exp3_convergence"]:
        if os.path.abspath(out_prefix).startswith(os.path.abspath(frozen) + os.sep):
            abort(f"output prefix {out_prefix} is inside frozen directory {frozen}")

    metrics_path = os.path.join(exp6_dir, "exp6_metrics.csv")
    if not os.path.isfile(metrics_path):
        abort(f"{metrics_path} not found - run run_exp6.py first")

    print("=" * 74)
    print("PHASE 15 / EXPERIMENT 6 - AGGREGATION")
    print(f"  exp6 dir             : {exp6_dir}")
    print(f"  baseline (read-only) : {baseline_dir}")
    print(f"  exp5     (read-only) : {exp5_dir}")
    print("=" * 74)

    exp6_df = pd.read_csv(metrics_path)
    baseline_df = load_baseline_metrics(runs_dir)
    summary_path = os.path.join(baseline_dir, "baseline_cv_summary.json")
    if not os.path.isfile(summary_path):
        abort(f"missing frozen comparator: {summary_path}")
    exp5_summary_path = os.path.join(exp5_dir, "exp5_summary.json")
    if not os.path.isfile(exp5_summary_path):
        abort(f"missing frozen comparator: {exp5_summary_path}")

    shas_before = {p: sha256(p) for p in
                   [summary_path, exp5_summary_path,
                    os.path.join(exp5_dir, "exp5_metrics.csv"),
                    os.path.join(baseline_dir, "fold_manifest.json"),
                    os.path.join(baseline_dir, "trivial_control_arm.csv")]
                   if os.path.isfile(p)}

    # ---- self-tests, then the authoritative gate --------------------------
    agg_worst = selftest_aggregation(baseline_df, summary_path)
    exp5_pf = load_exp5_primary(exp5_dir)
    exp5_worst = selftest_exp5_consistency(exp5_pf, exp5_summary_path)
    l0_report = l0_equivalence_gate(exp6_df, summary_path, exp6_dir)

    arms_present = [x for x in ARMS if x in set(exp6_df["arm"].unique())]
    missing = [x for x in ARMS if x not in arms_present]
    if missing:
        abort(f"exp6_metrics.csv is missing arms {missing}")
    for arm in arms_present:
        n = len(exp6_df[exp6_df["arm"] == arm])
        if n != N_REPEATS * N_FOLDS:
            abort(f"arm {arm} has {n} fold-runs, expected {N_REPEATS * N_FOLDS}")
        if "lambda" in exp6_df.columns:
            lams = sorted(set(np.round(exp6_df[exp6_df["arm"] == arm]["lambda"], 10)))
            if lams != [round(ARM_LAMBDA[arm], 10)]:
                abort(f"arm {arm} recorded lambda {lams}, expected "
                      f"[{ARM_LAMBDA[arm]}] - the single changed factor is not what "
                      "the pre-registration declared")

    # ---- aggregates -------------------------------------------------------
    aggregates: Dict[str, Dict[str, Dict[str, float]]] = {}
    per_fold_by_arm: Dict[str, pd.DataFrame] = {}
    for arm in arms_present:
        pf = per_fold_means(exp6_df[exp6_df["arm"] == arm], PRIMARY_METRICS)
        per_fold_by_arm[arm] = pf
        aggregates[arm] = {m: fold_level_ci(pf[m].values) for m in pf.columns}

    bases = {"L0": "the frozen default", "L1": "loss-value parity 0.5503/38.3111",
             "L2": "intermediate, 5x reduction", "L3": "regression term disabled"}
    roles = {"L0": "equivalence control", "L1": "PRIMARY",
             "L2": "sensitivity", "L3": "bound - sensitivity"}
    coefficients = {arm: {"lambda": ARM_LAMBDA[arm], "basis": bases.get(arm, ""),
                          "role": roles.get(arm, "")} for arm in arms_present}

    # ---- paired comparisons: primary arm vs Baseline-CV and vs Exp 5 W1 ---
    base_pf = per_fold_means(baseline_df, PAIRED_METRICS)
    exp6_pf = per_fold_by_arm[PRIMARY_ARM]

    paired: Dict[str, Dict] = {}
    for m in PAIRED_METRICS:
        if m not in exp6_pf.columns or m not in base_pf.columns:
            continue
        deltas = exp6_pf[m].values - base_pf[m].values
        paired[m] = {"baseline_mean": float(np.mean(base_pf[m].values)),
                     "exp6_mean": float(np.mean(exp6_pf[m].values)),
                     "per_fold_delta": [float(x) for x in deltas],
                     **paired_test(deltas, 0.0)}

    paired5: Dict[str, Dict] = {}
    for m in PAIRED_METRICS:
        if m not in exp6_pf.columns or m not in exp5_pf.columns:
            continue
        deltas = exp6_pf[m].values - exp5_pf[m].values
        paired5[m] = {"exp5_mean": float(np.mean(exp5_pf[m].values)),
                      "exp6_mean": float(np.mean(exp6_pf[m].values)),
                      "per_fold_delta": [float(x) for x in deltas],
                      **paired_test(deltas, 0.0)}

    # ---- Criterion 5: discrimination, not quantity -----------------------
    degen_path = os.path.join(exp6_dir, "degeneracy_report.csv")
    if not os.path.isfile(degen_path):
        abort(f"{degen_path} not found - run run_exp6.py first")
    DG = pd.read_csv(degen_path)

    prec_pf = exp6_pf["Precision"].values
    prec_test = paired_test(prec_pf, CORPUS_PREVALENCE)
    # "Exceeds" requires a positive point estimate AND a CI clearing the floor:
    # a mean above prevalence whose interval straddles it is not evidence.
    prec_exceeds = bool(prec_test["mean_delta"] > 0 and prec_test["ci95_lo"] > 0)

    dg_primary = DG[DG["arm"] == PRIMARY_ARM].copy()
    # Evaluate the control PER FOLD-RUN using that fold's own prevalence, then
    # average within fold - matching run_exp4_decision_rule.py:519 and
    # aggregate_exp5.py exactly. See random_ranker_f1() for why averaging
    # pos_rate first would be wrong.
    dg_primary["rand_F1"] = [random_ranker_f1(float(p), float(r)) for p, r in
                             zip(dg_primary["prevalence"], dg_primary["pos_rate"])]
    rand_f1_pf = dg_primary.groupby("fold")["rand_F1"].mean().sort_index().values
    f1_pf = exp6_pf["F1"].values
    f1_vs_rand = paired_test(f1_pf - rand_f1_pf, 0.0)
    f1_exceeds = bool(f1_vs_rand["mean_delta"] > 0 and f1_vs_rand["ci95_lo"] > 0)

    criterion5 = {
        "per_fold_precision": [float(x) for x in prec_pf],
        "prevalence": CORPUS_PREVALENCE,
        "precision_vs_prevalence": prec_test,
        "precision_exceeds_prevalence": prec_exceeds,
        "exp6_f1_mean": float(np.mean(f1_pf)),
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
                  "exp5_reference": EXP5_DEGENERACY,
                  "baseline_reference": BASELINE_DEGENERACY,
                  "primary_degenerate_count": primary_degen,
                  "met": bool(primary_degen < EXP5_DEGENERACY)}

    # ---- OBSERVATIONAL: loss / gradient decomposition --------------------
    # Reported in full; consulted by NO acceptance criterion (design 15.1).
    decomp_path = os.path.join(exp6_dir, "loss_decomposition.csv")
    dec_cols = ["lambda", "loss_cls", "loss_reg", "lambda_loss_reg", "ratio",
                "grad_norm_cls", "grad_norm_reg", "grad_norm_ratio"]
    obs_decomp: Dict[str, object] = {"available": False, "by_arm": {},
                                     "zero_reference_grad_norm_reg": float("nan"),
                                     "note": "OBSERVATIONAL ONLY - not an acceptance "
                                             "criterion; see design section 15.1"}
    if os.path.isfile(decomp_path):
        DC = pd.read_csv(decomp_path)
        present = [c for c in dec_cols if c in DC.columns]
        obs_decomp["available"] = True
        obs_decomp["by_arm"] = {
            arm: {c: float(DC[DC["arm"] == arm][c].mean()) for c in present}
            for arm in arms_present if (DC["arm"] == arm).any()}
        if ZERO_LAMBDA_ARM in obs_decomp["by_arm"]:
            obs_decomp["zero_reference_grad_norm_reg"] = \
                obs_decomp["by_arm"][ZERO_LAMBDA_ARM].get("grad_norm_reg", float("nan"))
    else:
        print(f"[WARN] {decomp_path} not found - the observational decomposition will "
              "be omitted. This does NOT affect any acceptance criterion.")

    # ---- OBSERVATIONAL: lambda ordering ----------------------------------
    # Demoted from a proposed "C7" before execution (design 11.1-M). Reported,
    # never pass/fail, and NOT consulted by overall_verdict.
    arm_order = sorted(arms_present, key=lambda x: -ARM_LAMBDA[x])
    obs_order: Dict[str, object] = {
        "arm_order": arm_order,
        "lambda_order": [ARM_LAMBDA[x] for x in arm_order],
        "by_metric": {},
        "note": "OBSERVATIONAL ONLY - not an acceptance criterion; demoted from the "
                "proposed C7 because with 4 arms one ordering arises by chance with "
                "p ~ 1/24 = 4.2%. See design section 11.1-M."}
    for m in ["PR_AUC", "ROC_AUC", "Recall", "F1", "Precision"]:
        vals = [float(aggregates[x][m]["mean"]) for x in arm_order]
        obs_order["by_metric"][m] = {
            "values": vals,
            "monotone_non_decreasing": bool(all(
                vals[i] <= vals[i + 1] + 1e-12 for i in range(len(vals) - 1)))}
    if os.path.isfile(degen_path):
        bw = [float(by_arm[x]["band_width_mean"]) for x in arm_order]
        obs_order["by_metric"]["p_pos_band_width"] = {
            "values": bw,
            "monotone_non_decreasing": bool(all(
                bw[i] <= bw[i + 1] + 1e-12 for i in range(len(bw) - 1)))}

    # ---- acceptance criteria (PRIMARY ARM ONLY) --------------------------
    pr = paired.get("PR_AUC", {})
    c1 = bool(pr.get("mean_delta", 0.0) > 0 and pr.get("detectable", False))
    rec, f1 = paired.get("Recall", {}), paired.get("F1", {})
    c2 = bool(rec.get("mean_delta", 0.0) > 0 and f1.get("mean_delta", 0.0) > 0)
    e6_rec = aggregates[PRIMARY_ARM]["Recall"]["mean"]
    e6_f1 = aggregates[PRIMARY_ARM]["F1"]["mean"]
    c3 = bool(e6_rec > EXP5_RECALL and e6_f1 > EXP5_F1)
    mae = paired.get("MAE", {})
    e6_mae = aggregates[PRIMARY_ARM]["MAE"]["mean"]
    # Not materially degraded: MAE must not RISE by more than the roadmap anchor,
    # AND must stay below the Exp-0a mean-predictor bound (design section 11 C4).
    c4 = bool(mae.get("mean_delta", 0.0) <= MDE_MAE_BASELINE
              and e6_mae < BASELINE_MEAN_PREDICTOR_MAE)

    crit = [
        {"id": 0, "statement": "L0 reproduces Baseline-CV (all primary metrics inside its 95% CI)",
         "source": "design gate", "met": bool(l0_report["passed"])},
        {"id": 1, "statement": "PR-AUC detectably above Baseline-CV 0.3941",
         "source": "Roadmap section 6 Exp 6", "met": c1},
        {"id": 2, "statement": "Recall and F1 improve over Baseline-CV (0.0000/0.0000)",
         "source": "Roadmap section 6 Exp 6", "met": c2},
        {"id": 3, "statement": f"Recall and F1 improve over Exp 5 {EXP5_ARM} "
                               f"({EXP5_RECALL}/{EXP5_F1})",
         "source": "Roadmap section 6 Exp 6", "met": c3},
        {"id": 4, "statement": f"Regression MAE not materially degraded (MDE "
                               f"{MDE_MAE_BASELINE}, mean-predictor bound "
                               f"{BASELINE_MEAN_PREDICTOR_MAE})",
         "source": "Roadmap section 6 Exp 6", "met": c4},
        {"id": 5, "statement": f"Discrimination, not quantity: precision > prevalence "
                               f"{CORPUS_PREVALENCE:.4f} AND F1 > random-ranker control",
         "source": "carried forward from Exps 4, 5", "met": criterion5["met"]},
        {"id": 6, "statement": f"Decisions non-degenerate (below Exp 5 {EXP5_ARM}'s "
                               f"{EXP5_DEGENERACY}/25)",
         "source": "carried forward from Exps 4, 5", "met": criterion6["met"]},
    ]

    # The verdict consults C0, C1 and C5 ONLY - never the observational blocks.
    gate_ok = crit[0]["met"]
    primary_met = gate_ok and c1 and criterion5["met"]
    overall = ("H1 SUPPORTED - loss-term rebalancing produced discrimination" if primary_met
               else "H0 - loss-term rebalancing did not produce demonstrated discrimination"
               if gate_ok else "INVALID - L0 equivalence gate failed")

    S = {
        "experiment": "Phase 15 / Exp 6 - loss-term rebalancing",
        "arms": arms_present, "primary_arm": PRIMARY_ARM, "control_arm": CONTROL_ARM,
        "sensitivity_arms": SENSITIVITY_ARMS,
        "arm_lambda": {k: ARM_LAMBDA[k] for k in arms_present},
        "criteria_evaluated_on": PRIMARY_ARM,
        "n_repeats": N_REPEATS, "n_folds": N_FOLDS,
        "t_crit": float(stats.t.ppf(0.975, N_FOLDS - 1)),
        "ci_method": "fold-level Student-t (df=k-1), repeats averaged within fold",
        "prevalence": CORPUS_PREVALENCE,
        "selftests": {"aggregation_max_diff": float(agg_worst),
                      "exp5_consistency_max_diff": float(exp5_worst)},
        "l0_gate": l0_report,
        "coefficients": coefficients,
        "aggregates": aggregates,
        "paired_vs_baseline": paired,
        "paired_vs_exp5": paired5,
        "exp5_reference": {"arm": EXP5_ARM, "Recall": EXP5_RECALL, "F1": EXP5_F1,
                           "degeneracy": EXP5_DEGENERACY,
                           "note": "Exp 5 was measured at argmax and 3 epochs, identical "
                                   "to Exp 6, so this comparison is like-for-like and "
                                   "needs no threshold caveat."},
        "criterion5_discrimination": criterion5,
        "criterion6_degeneracy": criterion6,
        "observational_loss_decomposition": obs_decomp,
        "observational_lambda_ordering": obs_order,
        "mde_anchors": {"MAE": MDE_MAE_BASELINE, "PR_AUC": MDE_PR_AUC_BASELINE},
        "l3_regression_caveat": (f"Arm {ZERO_LAMBDA_ARM} sets lambda = 0, so phq_mu "
                                 "receives no supervision; its MAE/RMSE/pred_var are "
                                 "descriptive only and are not regression performance "
                                 "(design section 11.0)."),
        "acceptance_criteria": crit,
        "overall_verdict": overall,
        "verdict_inputs": ["C0", "C1", "C5"],
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
    print(f"AGGREGATE - primary arm {PRIMARY_ARM} (lambda = {ARM_LAMBDA[PRIMARY_ARM]})")
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
        print(f"  {m:10s} base {p['baseline_mean']:8.4f} -> exp6 {p['exp6_mean']:8.4f} "
              f"| delta {p['mean_delta']:+8.4f} MDE {p['mde_95']:.4f} "
              f"| {'DETECTABLE' if p['detectable'] else 'not detectable'}")
    print("-" * 74)
    print(f"PAIRED vs Exp 5 {EXP5_ARM} (like-for-like: argmax, 3 epochs)")
    for m in PAIRED_METRICS:
        p = paired5.get(m)
        if p is None:
            continue
        print(f"  {m:10s} exp5 {p['exp5_mean']:8.4f} -> exp6 {p['exp6_mean']:8.4f} "
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
          f"(Exp 5 {EXP5_ARM} {EXP5_DEGENERACY}/25, Baseline-CV {BASELINE_DEGENERACY}/25) "
          f"| met: {criterion6['met']}")
    print("-" * 74)
    for c in crit:
        print(f"  [{'MET    ' if c['met'] else 'NOT MET'}] ({c['id']}) {c['statement']}")
    print(f"\nOVERALL: {overall}")
    print(f"  (verdict consults {S['verdict_inputs']} only; the loss/gradient "
          "decomposition and the lambda-ordering analysis are OBSERVATIONAL)")

    if obs_decomp["available"]:
        print("-" * 74)
        print("OBSERVATIONAL - loss/gradient decomposition (NOT an acceptance criterion)")
        for arm in arms_present:
            d = obs_decomp["by_arm"].get(arm, {})
            if not d:
                continue
            print(f"  {arm} (lam={ARM_LAMBDA[arm]:<7}) loss_cls {d.get('loss_cls', float('nan')):9.4f} "
                  f"| lam*loss_reg {d.get('lambda_loss_reg', float('nan')):10.4f} "
                  f"| loss ratio {d.get('ratio', float('nan')):8.3f} "
                  f"| |g_cls| {d.get('grad_norm_cls', float('nan')):8.4f} "
                  f"| |g_reg| {d.get('grad_norm_reg', float('nan')):8.4f} "
                  f"| grad ratio {d.get('grad_norm_ratio', float('nan')):8.3f}")

    print(f"\n[DONE] wrote {out_prefix}.json and {out_prefix}.md")


if __name__ == "__main__":
    main()
