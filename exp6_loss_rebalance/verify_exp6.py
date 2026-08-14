#!/usr/bin/env python
"""
Phase 15 / Experiment 6 - independent post-hoc auditor.

Re-derives every Experiment 6 result from the raw per-fold prediction files and
checks it against what run_exp6.py and aggregate_exp6.py actually wrote. Its
purpose is to catch a defect in those two scripts, so it deliberately does NOT
import them: every quantity it compares against is recomputed here from first
principles.

A verifier that reuses the code under test can only confirm the code is
self-consistent; this one can confirm it is correct.

It also does not import torch, transformers, or the frozen trainer. Auditing does
not require a model.

WHAT IS AUDITED
---------------
  A  artifact inventory        expected set DERIVED from the run design, never a
                               hard-coded literal (literals produced false
                               failures in the Exp 4 `A3` and Exp 3 `J2` auditors)
  B  frozen-input integrity    Baseline-CV / Exp 3 / Exp 4 / Exp 5 / trainer SHAs
  C  fold integrity            participant order, coverage, train/test disjoint,
                               argmax decision rule reproduced from p_pos
  D  metric re-derivation      every metric recomputed from raw predictions
  E  the single changed factor lambda per arm, and that NO class weighting entered
  F  L0 equivalence gate       independently re-evaluated against Baseline-CV CIs
  G  degeneracy diagnostics    band width, pos_rate, degeneracy from raw preds
  H  aggregation re-derivation summary means / CIs / paired deltas vs Baseline-CV
  I  Exp 5 comparator          paired deltas vs Exp 5 W1, and the C3 constants
  J  Criterion 5               precision-vs-prevalence and F1-vs-random-ranker,
                               the latter evaluated PER FOLD-RUN (Jensen)
  K  Criterion 6               degeneracy counts vs Exp 5 W1's 24/25
  L  criteria C0-C6            every recorded verdict re-derived, on L1 only
  M  loss decomposition        columns, gradient norms, L3 zero-reference
  N  observational isolation   the observational blocks did not move the verdict
  O  summary artifacts         JSON structure and markdown content
  P  containment               nothing written outside the Experiment 6 directory

All findings are collected; the audit does not stop at the first failure, because
knowing whether a defect is isolated or systemic is more useful than knowing only
that one exists. Exit code is non-zero if any check fails.

Usage (from the repository root):
    python exp6_loss_rebalance/verify_exp6.py
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
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    balanced_accuracy_score,
    f1_score,
    matthews_corrcoef,
    mean_absolute_error,
    precision_score,
    recall_score,
    roc_auc_score,
)

# --------------------------------------------------------------------------
# Constants restated independently, NOT imported from the scripts under test.
# A disagreement here is itself a finding worth surfacing.
# --------------------------------------------------------------------------
FROZEN_TRAINER_SHA = "65b1902e2ba8dd996c6960db1a6595d84fd48500f4d8707cb79688093d7a230b"
FROZEN_INPUT_SHA = {
    "trainer_outputs/baseline_cv/fold_manifest.json":
        "b9a7a91f1bdd43c78d3cd1ee0c36e4ce3fb8be4b0971dcff3ff62ee5af8fca2f",
    "trainer_outputs/baseline_cv/trivial_control_arm.csv":
        "2ecbfee3225910034a959fe949c7ad027cfa7d41e9bef8040374578ff8c5d572",
    "trainer_outputs/baseline_cv/baseline_cv_summary.json":
        "f065f5e1ff7f8e5ed4d122a8031c9cc12425802e756697d5512a915674871aac",
    "trainer_outputs/exp4_decision_rule/exp4_summary.json":
        "5defdae2a20d0abc164611e8cbe6b8034e3f65c593d8c9b3e38266a1800aa6f2",
    "trainer_outputs/exp3_convergence/exp3_summary.json":
        "9dd135b6b79af06a85e64a5aa8c1896d19df8059dd89c053e50e1cffe20af2f9",
    "trainer_outputs/exp5_imbalance_objective/exp5_summary.json":
        "70ad13ffc4db633419595761c0396fc20dd1b62400ee74238be1f06b73ed48d3",
}

ARM_LAMBDA: Dict[str, float] = {"L0": 0.5, "L1": 0.0144, "L2": 0.1, "L3": 0.0}
ARMS: List[str] = ["L0", "L1", "L2", "L3"]
PRIMARY_ARM = "L1"
CONTROL_ARM = "L0"
SENSITIVITY_ARMS = ["L2", "L3"]
ZERO_LAMBDA_ARM = "L3"
FROZEN_MSE_COEFF = 0.5

N_REPEATS = 5
N_FOLDS = 5
N_PARTICIPANTS = 188
N_POSITIVE = 45
CORPUS_PREVALENCE = N_POSITIVE / N_PARTICIPANTS
BINARIZE_THRESHOLD = 10.0
ARGMAX_TAU = 0.5

MDE_MAE_BASELINE = 0.4066
MDE_PR_AUC_BASELINE = 0.0950
BASELINE_MEAN_PREDICTOR_MAE = 4.9135

EXP5_ARM = "W1"
EXP5_RECALL = 0.0756
EXP5_F1 = 0.0295
EXP5_DEGENERACY = 24
BASELINE_DEGENERACY = 25

PRIMARY_METRICS = ["MAE", "RMSE", "pred_var", "MAE_mean_pred", "ROC_AUC", "PR_AUC",
                   "balAcc", "MCC", "Precision", "Recall", "F1", "Accuracy"]
PAIRED_METRICS = ["MAE", "RMSE", "pred_var", "ROC_AUC", "PR_AUC",
                  "balAcc", "MCC", "Precision", "Recall", "F1", "Accuracy"]

PRED_SCHEMA = ["participant_id", "phq", "phq_bin", "pred_phq", "p_pos", "pred_class"]
DECOMP_SCHEMA = ["arm", "repeat", "fold", "lambda", "loss_cls", "loss_reg",
                 "lambda_loss_reg", "ratio", "grad_norm_cls", "grad_norm_reg",
                 "grad_norm_ratio"]
DEGEN_SCHEMA = ["arm", "repeat", "fold", "p_pos_min", "p_pos_max", "p_pos_band_width",
                "p_pos_mean", "p_pos_sd", "pred_phq_spread", "pos_rate", "prevalence",
                "degenerate", "n_pred_pos"]
# Columns that would indicate Exp 5's factor leaked into Exp 6.
FORBIDDEN_METRIC_COLS = ["w_neg", "w_pos", "n_pos_train", "n_neg_train"]

TOL = 1e-9


class Audit:
    """Collects pass/fail results so the full picture is reported at once."""

    def __init__(self) -> None:
        self.n = 0
        self.failures: List[str] = []
        self.notes: List[str] = []

    def check(self, label: str, ok: bool, detail: str = "") -> bool:
        self.n += 1
        if not ok:
            self.failures.append(f"{label}{(' - ' + detail) if detail else ''}")
        return bool(ok)

    def note(self, text: str) -> None:
        self.notes.append(text)

    def report(self) -> bool:
        print("=" * 74)
        print(f"checks run : {self.n}")
        print(f"FAILURES   : {len(self.failures)}")
        for f in self.failures:
            print(f"   FAIL: {f}")
        if self.notes:
            print(f"notes      : {len(self.notes)}")
            for t in self.notes:
                print(f"   {t}")
        ok = not self.failures
        print(f"\nEXP 6 AUDIT PASS : {ok}")
        return ok


def sha256(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for block in iter(lambda: fh.read(8192), b""):
            h.update(block)
    return h.hexdigest()


def close(a, b, tol: float = TOL) -> bool:
    """NaN-tolerant comparison: two NaNs are equal, matching the single-class
    convention where AUC is undefined."""
    try:
        fa, fb = float(a), float(b)
    except (TypeError, ValueError):
        return a == b
    if np.isnan(fa) and np.isnan(fb):
        return True
    return bool(abs(fa - fb) <= tol)


# --------------------------------------------------------------------------
# Independent recomputation primitives
# --------------------------------------------------------------------------
def recompute_metrics(y_phq, y_bin, pred_phq, p_pos, pred_class,
                      train_mean) -> Dict[str, float]:
    """The full metric suite, recomputed from scratch (numpy var uses ddof=0,
    matching the frozen Baseline-CV driver)."""
    two = len(np.unique(y_bin)) > 1
    yt = np.asarray(y_phq, float)
    pp = np.asarray(pred_phq, float)
    return {
        "MAE": float(mean_absolute_error(yt, pp)),
        "RMSE": float(np.sqrt(np.mean((yt - pp) ** 2))),
        "pred_var": float(np.var(pp)),
        "MAE_mean_pred": float(mean_absolute_error(yt, np.full(len(yt), train_mean))),
        "ROC_AUC": float(roc_auc_score(y_bin, p_pos)) if two else float("nan"),
        "PR_AUC": float(average_precision_score(y_bin, p_pos)) if two else float("nan"),
        "balAcc": float(balanced_accuracy_score(y_bin, pred_class)),
        "MCC": float(matthews_corrcoef(y_bin, pred_class)) if two else 0.0,
        "Precision": float(precision_score(y_bin, pred_class, zero_division=0)),
        "Recall": float(recall_score(y_bin, pred_class, zero_division=0)),
        "F1": float(f1_score(y_bin, pred_class, zero_division=0)),
        "Accuracy": float(accuracy_score(y_bin, pred_class)),
    }


def random_ranker_f1(prevalence: float, pos_rate: float) -> float:
    """Expected F1 of a random ranker at the given positive rate.

    Evaluated PER FOLD-RUN using that fold's own prevalence, then averaged -
    never from an averaged pos_rate. The function is non-linear, so by Jensen's
    inequality the two differ materially (0.211901 vs 0.330089 on the Exp 4
    artifacts). Experiments 4 and 5 computed it per fold-run, so this audit does
    too.
    """
    denom = prevalence + pos_rate
    return float(2.0 * prevalence * pos_rate / denom) if denom > 0 else 0.0


def fold_ci(values: Sequence[float]) -> Dict[str, float]:
    v = np.asarray(values, dtype=float)
    k = len(v)
    m = float(np.mean(v))
    sd = float(np.std(v, ddof=1)) if k > 1 else 0.0
    se = sd / np.sqrt(k) if k > 1 else 0.0
    tc = float(stats.t.ppf(0.975, k - 1)) if k > 1 else float("nan")
    return {"mean": m, "fold_sd": sd, "se": se, "t_crit": tc,
            "ci95_lo": m - tc * se, "ci95_hi": m + tc * se}


def paired_test(values: Sequence[float], mu0: float = 0.0) -> Dict[str, float]:
    """Independent reimplementation, including the zero-variance guard that the
    degenerate Baseline-CV Recall/F1 columns require."""
    v = np.asarray(values, dtype=float)
    k = int(len(v))
    diff = v - float(mu0)
    mean = float(np.mean(diff))
    sd = float(np.std(diff, ddof=1)) if k > 1 else 0.0
    se = sd / np.sqrt(k) if k > 1 else 0.0
    tc = float(stats.t.ppf(0.975, k - 1)) if k > 1 else float("nan")
    mde = tc * se
    detectable = bool(mean != 0.0) if sd == 0.0 else bool(abs(mean) > mde)
    return {"mean_delta": mean, "sd_delta": sd, "se": se, "mde_95": mde,
            "ci95_lo": mean - mde, "ci95_hi": mean + mde, "detectable": detectable,
            "zero_variance": sd == 0.0}


# --------------------------------------------------------------------------
# Main audit
# --------------------------------------------------------------------------
def main() -> None:
    ap = argparse.ArgumentParser(
        description="Phase 15 / Experiment 6 - independent post-hoc audit.")
    ap.add_argument("--exp6-dir", default="trainer_outputs/exp6_loss_rebalance")
    ap.add_argument("--baseline-dir", default="trainer_outputs/baseline_cv")
    ap.add_argument("--exp5-dir", default="trainer_outputs/exp5_imbalance_objective")
    ap.add_argument("--parquet", default="daic_records.parquet")
    ap.add_argument("--tol", type=float, default=TOL)
    a = ap.parse_args()

    exp6 = os.path.normpath(a.exp6_dir)
    base = os.path.normpath(a.baseline_dir)
    exp5 = os.path.normpath(a.exp5_dir)
    runs = os.path.join(base, "runs")
    tol = a.tol
    A = Audit()

    print("=" * 74)
    print("PHASE 15 / EXPERIMENT 6 - INDEPENDENT AUDIT")
    print(f"  exp6 dir             : {exp6}")
    print(f"  baseline (read-only) : {base}")
    print(f"  exp5     (read-only) : {exp5}")
    print(f"  tolerance            : {tol:.1e}")
    print("=" * 74)

    if not os.path.isdir(exp6):
        print(f"\n[ABORT] {exp6} not found - run the runner and aggregator first",
              file=sys.stderr)
        raise SystemExit(1)

    # ---- A. artifact inventory (DERIVED, never a literal) -----------------
    named = {"exp6_metrics.csv", "loss_decomposition.csv", "degeneracy_report.csv",
             "l0_equivalence_check.json", "loop_correspondence.md",
             "exp6_summary.json", "exp6_summary.md"}
    preds = {f"{arm}_rep{r}_fold{f}_preds.csv"
             for arm in ARMS for r in range(1, N_REPEATS + 1)
             for f in range(1, N_FOLDS + 1)}
    mets = {f"{arm}_rep{r}_metrics.csv"
            for arm in ARMS for r in range(1, N_REPEATS + 1)}
    expected = named | preds | mets
    entries = os.listdir(exp6)
    present = {f for f in entries if os.path.isfile(os.path.join(exp6, f))}
    A.check("A1 all expected artifacts present", expected <= present,
            f"missing {sorted(expected - present)[:6]}")
    A.check("A2 no unexpected artifacts", not (present - expected),
            f"unexpected {sorted(present - expected)[:6]}")
    A.check(f"A3 artifact count == {len(expected)} (derived from the run design)",
            len(expected & present) == len(expected),
            f"found {len(expected & present)}")
    A.check("A4 no subdirectories in the artifact dir",
            not [d for d in entries if os.path.isdir(os.path.join(exp6, d))
                 and d != "__pycache__"])
    # duplicate detection: identical content under two names would mean a fold-run
    # was copied rather than computed
    digests: Dict[str, List[str]] = {}
    for f in sorted(present):
        digests.setdefault(sha256(os.path.join(exp6, f)), []).append(f)
    dupes = {h: v for h, v in digests.items() if len(v) > 1}
    A.check("A5 no duplicate artifact contents", not dupes,
            f"{[v for v in dupes.values()][:3]}")
    A.note(f"expected artifact count derived as {len(named)} named + {len(preds)} preds "
           f"+ {len(mets)} metrics = {len(expected)}")

    # ---- B. frozen-input integrity ---------------------------------------
    for path, want in FROZEN_INPUT_SHA.items():
        got = sha256(path) if os.path.isfile(path) else "MISSING"
        A.check(f"B1 {os.path.basename(path)} SHA frozen", got == want, f"got {got}")
    tp = "trainer_mentalbert_daic.py"
    if os.path.isfile(tp):
        A.check("B2 trainer SHA frozen", sha256(tp) == FROZEN_TRAINER_SHA)
    else:
        A.check("B2 trainer file locatable", False, tp)

    # ---- load -------------------------------------------------------------
    with open(os.path.join(base, "fold_manifest.json"), "r", encoding="utf-8") as fh:
        manifest = json.load(fh)
    A.check("C-1 manifest participants == 188",
            manifest.get("n_participants") == N_PARTICIPANTS)
    A.check("C-2 manifest folds == 5", len(manifest.get("folds", [])) == N_FOLDS)

    M = pd.read_csv(os.path.join(exp6, "exp6_metrics.csv"))
    DC = pd.read_csv(os.path.join(exp6, "loss_decomposition.csv"))
    DG = pd.read_csv(os.path.join(exp6, "degeneracy_report.csv"))
    base_metrics = pd.concat([pd.read_csv(os.path.join(runs, f"rep{r}_metrics.csv"))
                              for r in range(1, N_REPEATS + 1)], ignore_index=True)

    n_runs = len(ARMS) * N_REPEATS * N_FOLDS
    for name, df in [("exp6_metrics", M), ("loss_decomposition", DC),
                     ("degeneracy_report", DG)]:
        A.check(f"A6 {name} row count == {n_runs}", len(df) == n_runs, f"found {len(df)}")
        cells = set(zip(df["arm"], df["repeat"], df["fold"]))
        full = {(x, r, f) for x in ARMS for r in range(1, N_REPEATS + 1)
                for f in range(1, N_FOLDS + 1)}
        A.check(f"A7 {name} grid complete and unique",
                cells == full and len(cells) == len(df))
    # schemas
    A.check("A8 loss_decomposition schema", set(DECOMP_SCHEMA) <= set(DC.columns),
            f"missing {sorted(set(DECOMP_SCHEMA) - set(DC.columns))}")
    A.check("A9 degeneracy_report schema", set(DEGEN_SCHEMA) <= set(DG.columns),
            f"missing {sorted(set(DEGEN_SCHEMA) - set(DG.columns))}")
    A.check("A10 exp6_metrics carries lambda", "lambda" in M.columns)

    # training-fold PHQ means, for MAE_mean_pred
    train_means: Dict[int, float] = {}
    phq_by_pid: Dict[int, float] = {}
    try:
        import pyarrow.parquet as pq
        recs = pq.read_table(a.parquet).to_pandas()
        phq_by_pid = {int(r.participant_id): float(r.phq_score) for r in recs.itertuples()}
        for fold in manifest["folds"]:
            ids = [int(x) for x in fold["train_ids"]]
            train_means[int(fold["fold"])] = float(np.mean([phq_by_pid[i] for i in ids]))
        A.check("C0 parquet readable for train-mean re-derivation", True)
    except Exception as exc:
        A.check("C0 parquet readable for train-mean re-derivation", False, str(exc))

    # ---- C-G. per fold-run re-derivation ---------------------------------
    for arm in ARMS:
        lam = ARM_LAMBDA[arm]
        for r in range(1, N_REPEATS + 1):
            seen: List[int] = []
            for f in range(1, N_FOLDS + 1):
                tag = f"{arm} rep{r} fold{f}"
                pth = os.path.join(exp6, f"{arm}_rep{r}_fold{f}_preds.csv")
                if not A.check(f"C1 {tag} preds file present", os.path.isfile(pth)):
                    continue
                d = pd.read_csv(pth)
                A.check(f"C1b {tag} prediction schema", list(d.columns) == PRED_SCHEMA,
                        f"got {list(d.columns)}")

                spec = manifest["folds"][f - 1]
                ids = [int(x) for x in d["participant_id"]]
                seen.extend(ids)
                A.check(f"C2 {tag} participant order == manifest test_ids",
                        ids == [int(x) for x in spec["test_ids"]])
                A.check(f"C3 {tag} NO LEAKAGE: train/test disjoint",
                        not (set(ids) & set(int(x) for x in spec["train_ids"])))
                A.check(f"C3b {tag} prediction count == manifest test size",
                        len(d) == len(spec["test_ids"]))

                y_phq = d["phq"].to_numpy(float)
                y_bin = (y_phq > BINARIZE_THRESHOLD).astype(int)
                A.check(f"C4 {tag} phq_bin consistent with PHQ > 10.0",
                        np.array_equal(y_bin, d["phq_bin"].to_numpy(int)))
                if phq_by_pid:
                    A.check(f"C4b {tag} phq matches the frozen dataset",
                            all(close(phq_by_pid[int(i)], v, 1e-9)
                                for i, v in zip(ids, y_phq)))
                pred_phq = d["pred_phq"].to_numpy(float)
                p_pos = d["p_pos"].to_numpy(float)
                pred_class = d["pred_class"].to_numpy(int)
                # The decision rule must remain argmax - Exps 4 and 5 were both
                # H0, so no threshold policy was accepted into the configuration.
                A.check(f"C5 {tag} pred_class == argmax (p_pos >= 0.5)",
                        np.array_equal(pred_class, (p_pos >= ARGMAX_TAU).astype(int)))

                row = M[(M["arm"] == arm) & (M["repeat"] == r) & (M["fold"] == f)]
                if A.check(f"D0 {tag} metric row present", len(row) == 1):
                    row = row.iloc[0]
                    tm = train_means.get(f, float(row.get("MAE_mean_pred", np.nan)))
                    rec = recompute_metrics(y_phq, y_bin, pred_phq, p_pos, pred_class, tm)
                    for m in PRIMARY_METRICS:
                        if m == "MAE_mean_pred" and f not in train_means:
                            continue
                        A.check(f"D1 {tag} {m} re-derived", close(rec[m], row[m], tol),
                                f"audit={rec[m]!r} artifact={row[m]!r}")
                    # E. the single changed factor
                    A.check(f"E1 {tag} lambda == pre-declared {lam}",
                            close(row["lambda"], lam, 1e-12),
                            f"artifact={row.get('lambda')!r}")

                # G. degeneracy diagnostics re-derived
                drow = DG[(DG["arm"] == arm) & (DG["repeat"] == r) & (DG["fold"] == f)]
                if A.check(f"G0 {tag} diagnostics row present", len(drow) == 1):
                    drow = drow.iloc[0]
                    A.check(f"G1 {tag} band width",
                            close(float(p_pos.max() - p_pos.min()),
                                  drow["p_pos_band_width"], tol))
                    pr = float(np.mean(pred_class))
                    A.check(f"G2 {tag} pos_rate", close(pr, drow["pos_rate"], tol))
                    A.check(f"G3 {tag} degenerate flag",
                            bool(drow["degenerate"]) == bool(pr in (0.0, 1.0)))
                    A.check(f"G4 {tag} fold prevalence",
                            close(float(np.mean(y_bin)), drow["prevalence"], tol))
                    A.check(f"G5 {tag} n_pred_pos",
                            int(drow["n_pred_pos"]) == int(np.sum(pred_class == 1)))

                # M. loss decomposition row
                crow = DC[(DC["arm"] == arm) & (DC["repeat"] == r) & (DC["fold"] == f)]
                if A.check(f"M0 {tag} decomposition row present", len(crow) == 1):
                    crow = crow.iloc[0]
                    A.check(f"M1 {tag} decomposition lambda", close(crow["lambda"], lam, 1e-12))
                    A.check(f"M2 {tag} lambda_loss_reg == lambda * loss_reg",
                            close(float(crow["lambda_loss_reg"]),
                                  lam * float(crow["loss_reg"]), 1e-6))
                    lc = float(crow["loss_cls"])
                    if lc != 0.0:
                        A.check(f"M3 {tag} loss ratio",
                                close(float(crow["ratio"]),
                                      float(crow["lambda_loss_reg"]) / lc, 1e-6))
                    gc = float(crow["grad_norm_cls"])
                    A.check(f"M4 {tag} grad norms non-negative",
                            gc >= 0.0 and float(crow["grad_norm_reg"]) >= 0.0)
                    if gc > 0.0:
                        A.check(f"M5 {tag} grad ratio",
                                close(float(crow["grad_norm_ratio"]),
                                      float(crow["grad_norm_reg"]) / gc, 1e-6))
                    if arm == ZERO_LAMBDA_ARM:
                        # L3 zero-reference: lambda = 0 must contribute exactly no
                        # encoder gradient. This doubles as a check on the instrument.
                        A.check(f"M6 {tag} L3 zero-reference grad_norm_reg == 0",
                                float(crow["grad_norm_reg"]) == 0.0,
                                f"got {crow['grad_norm_reg']!r}")
                        A.check(f"M7 {tag} L3 lambda_loss_reg == 0",
                                close(float(crow["lambda_loss_reg"]), 0.0, 1e-12))

            A.check(f"C6 {arm} rep{r} covers {N_PARTICIPANTS} unique participants",
                    len(seen) == N_PARTICIPANTS and len(set(seen)) == N_PARTICIPANTS,
                    f"rows={len(seen)} unique={len(set(seen))}")

    # ---- E. no class weighting entered (Exp 5's factor must not leak) -----
    A.check("E2 control arm lambda == frozen coefficient",
            close(ARM_LAMBDA[CONTROL_ARM], FROZEN_MSE_COEFF, 1e-12))
    A.check("E3 no class-weight columns in exp6_metrics.csv",
            not [c for c in FORBIDDEN_METRIC_COLS if c in M.columns],
            f"found {[c for c in FORBIDDEN_METRIC_COLS if c in M.columns]}")
    A.check("E4 no class_weights.csv artifact (CE is unweighted in Exp 6)",
            not os.path.isfile(os.path.join(exp6, "class_weights.csv")))
    A.check("E5 exactly four distinct lambda values across the run",
            sorted(set(np.round(M["lambda"].astype(float), 10)))
            == sorted(round(v, 10) for v in ARM_LAMBDA.values()))

    # ---- F. L0 equivalence gate, independently re-evaluated ---------------
    with open(os.path.join(base, "baseline_cv_summary.json"), "r", encoding="utf-8") as fh:
        published = json.load(fh)["metrics"]
    l0 = M[M["arm"] == CONTROL_ARM]
    outside: List[str] = []
    if A.check("F0 L0 arm complete", len(l0) == N_REPEATS * N_FOLDS):
        pf = l0.groupby("fold")[PRIMARY_METRICS].mean().sort_index()
        for m in PRIMARY_METRICS:
            if m not in published:
                continue
            got = fold_ci(pf[m].values)
            lo, hi = float(published[m]["ci95"][0]), float(published[m]["ci95"][1])
            if not (lo <= got["mean"] <= hi):
                outside.append(m)
        A.check("F1 L0 aggregate inside Baseline-CV 95% CI on all primary metrics",
                not outside, f"outside: {outside}")
        gp = os.path.join(exp6, "l0_equivalence_check.json")
        if A.check("F2 l0_equivalence_check.json present", os.path.isfile(gp)):
            with open(gp, "r", encoding="utf-8") as fh:
                rep = json.load(fh)
            A.check("F3 recorded gate verdict matches re-derivation",
                    bool(rep.get("passed")) == (not outside))
            A.check("F4 gate is flagged authoritative", bool(rep.get("authoritative")))
            A.check("F5 gate records the control lambda",
                    close(rep.get("control_lambda", -1), FROZEN_MSE_COEFF, 1e-12))

    # ---- I. Exp 5 comparator ---------------------------------------------
    e5_metrics_path = os.path.join(exp5, "exp5_metrics.csv")
    exp5_pf = None
    if A.check("I0 exp5_metrics.csv available", os.path.isfile(e5_metrics_path)):
        E5 = pd.read_csv(e5_metrics_path)
        w1 = E5[E5["arm"] == EXP5_ARM]
        A.check("I1 Exp 5 W1 complete", len(w1) == N_REPEATS * N_FOLDS)
        exp5_pf = w1.groupby("fold")[PAIRED_METRICS].mean().sort_index()
        with open(os.path.join(exp5, "exp5_summary.json"), "r", encoding="utf-8") as fh:
            e5pub = json.load(fh)["aggregates"][EXP5_ARM]
        worst = max(abs(fold_ci(exp5_pf[m].values)["mean"] - float(e5pub[m]["mean"]))
                    for m in PAIRED_METRICS if m in e5pub)
        A.check("I2 exp5_metrics.csv agrees with SHA-gated exp5_summary.json",
                worst <= 1e-6, f"max |diff| {worst:.3e}")
        A.check("I3 C3 constant EXP5_RECALL matches Exp 5's record",
                close(EXP5_RECALL, round(float(e5pub["Recall"]["mean"]), 4), 1e-9))
        A.check("I4 C3 constant EXP5_F1 matches Exp 5's record",
                close(EXP5_F1, round(float(e5pub["F1"]["mean"]), 4), 1e-9))

    # ---- H-O. aggregation, criteria, summaries ---------------------------
    sp = os.path.join(exp6, "exp6_summary.json")
    c5 = c6 = {}
    if A.check("H0 exp6_summary.json present", os.path.isfile(sp)):
        with open(sp, "r", encoding="utf-8") as fh:
            S = json.load(fh)

        A.check("H1 primary arm is L1", S.get("primary_arm") == PRIMARY_ARM)
        A.check("H2 control arm is L0", S.get("control_arm") == CONTROL_ARM)
        A.check("H3 t_crit == t(0.975, 4)",
                close(S.get("t_crit", 0), float(stats.t.ppf(0.975, N_FOLDS - 1)), 1e-9))
        A.check("H4 prevalence recorded exactly",
                close(S.get("prevalence", 0), CORPUS_PREVALENCE, 1e-15))
        A.check("H4b criteria evaluated on the primary arm",
                S.get("criteria_evaluated_on") == PRIMARY_ARM)
        A.check("H4c recorded arm lambdas match the pre-declared ladder",
                {k: float(v) for k, v in S.get("arm_lambda", {}).items()} == ARM_LAMBDA)

        for arm in ARMS:
            pf = M[M["arm"] == arm].groupby("fold")[PRIMARY_METRICS].mean().sort_index()
            for m in PRIMARY_METRICS:
                got = fold_ci(pf[m].values)
                ref = S.get("aggregates", {}).get(arm, {}).get(m)
                if A.check(f"H5 {arm} {m} aggregate present", ref is not None):
                    for key in ["mean", "fold_sd", "se", "ci95_lo", "ci95_hi"]:
                        A.check(f"H6 {arm} {m}.{key} re-derived",
                                close(got[key], float(ref[key]), tol))

        bpf = base_metrics.groupby("fold")[PAIRED_METRICS].mean().sort_index()
        epf = M[M["arm"] == PRIMARY_ARM].groupby("fold")[PAIRED_METRICS].mean().sort_index()
        for m in PAIRED_METRICS:
            deltas = epf[m].values - bpf[m].values
            ref = S.get("paired_vs_baseline", {}).get(m)
            if A.check(f"H7 paired-vs-baseline {m} present", ref is not None):
                mine = paired_test(deltas, 0.0)
                A.check(f"H8 paired {m} mean_delta re-derived",
                        close(mine["mean_delta"], float(ref["mean_delta"]), tol))
                A.check(f"H9 paired {m} MDE re-derived",
                        close(mine["mde_95"], float(ref["mde_95"]), tol))
                A.check(f"H10 paired {m} detectability re-derived",
                        bool(mine["detectable"]) == bool(ref["detectable"]))
                A.check(f"H11 paired {m} per-fold deltas re-derived",
                        all(close(x, y, tol) for x, y in
                            zip(deltas, ref.get("per_fold_delta", []))))

        if exp5_pf is not None:
            for m in PAIRED_METRICS:
                deltas = epf[m].values - exp5_pf[m].values
                ref = S.get("paired_vs_exp5", {}).get(m)
                if A.check(f"I5 paired-vs-exp5 {m} present", ref is not None):
                    mine = paired_test(deltas, 0.0)
                    A.check(f"I6 paired-vs-exp5 {m} mean_delta re-derived",
                            close(mine["mean_delta"], float(ref["mean_delta"]), tol))
                    A.check(f"I7 paired-vs-exp5 {m} MDE re-derived",
                            close(mine["mde_95"], float(ref["mde_95"]), tol))
                    A.check(f"I8 paired-vs-exp5 {m} detectability re-derived",
                            bool(mine["detectable"]) == bool(ref["detectable"]))

        # ---- J. Criterion 5 ----------------------------------------------
        c5 = S.get("criterion5_discrimination", {})
        if A.check("J0 Criterion-5 block present", bool(c5)):
            prec_pf = epf["Precision"].values
            pt = paired_test(prec_pf, CORPUS_PREVALENCE)
            A.check("J1 precision-vs-prevalence mean delta re-derived",
                    close(pt["mean_delta"],
                          float(c5["precision_vs_prevalence"]["mean_delta"]), tol))
            expect_prec = bool(pt["mean_delta"] > 0 and pt["ci95_lo"] > 0)
            A.check("J2 precision_exceeds_prevalence consistent with its own CI",
                    bool(c5.get("precision_exceeds_prevalence")) == expect_prec)
            A.check("J3 Criterion-5 mu0 == corpus prevalence",
                    close(float(c5["precision_vs_prevalence"].get("mu0", np.nan)),
                          CORPUS_PREVALENCE, 1e-15))
            # random-ranker control: PER FOLD-RUN, then averaged (Jensen)
            dgp = DG[DG["arm"] == PRIMARY_ARM].copy()
            dgp["rand_F1"] = [random_ranker_f1(float(p), float(q)) for p, q in
                              zip(dgp["prevalence"], dgp["pos_rate"])]
            rand_pf = dgp.groupby("fold")["rand_F1"].mean().sort_index().values
            A.check("J4 random-ranker F1 mean re-derived (per fold-run, then averaged)",
                    close(float(np.mean(rand_pf)),
                          float(c5.get("random_ranker_f1_mean", np.nan)), tol),
                    f"audit={float(np.mean(rand_pf)):.6f} "
                    f"artifact={c5.get('random_ranker_f1_mean')}")
            ft = paired_test(epf["F1"].values - rand_pf, 0.0)
            A.check("J5 F1-vs-random mean delta re-derived",
                    close(ft["mean_delta"], float(c5["f1_vs_random"]["mean_delta"]), tol))
            expect_f1 = bool(ft["mean_delta"] > 0 and ft["ci95_lo"] > 0)
            A.check("J6 f1_exceeds_random consistent with its own CI",
                    bool(c5.get("f1_exceeds_random")) == expect_f1)
            A.check("J7 Criterion-5 met == both sub-tests",
                    bool(c5.get("met")) == bool(expect_prec and expect_f1))
            A.note(f"Criterion 5: precision {np.mean(prec_pf):.4f} vs prevalence "
                   f"{CORPUS_PREVALENCE:.4f} (exceeds={c5.get('precision_exceeds_prevalence')}); "
                   f"F1 {np.mean(epf['F1'].values):.4f} vs random-ranker "
                   f"{np.mean(rand_pf):.4f} (exceeds={c5.get('f1_exceeds_random')})")

        # ---- K. Criterion 6 ----------------------------------------------
        c6 = S.get("criterion6_degeneracy", {})
        if A.check("K0 Criterion-6 block present", bool(c6)):
            for arm in ARMS:
                d = DG[DG["arm"] == arm]
                ref = c6.get("by_arm", {}).get(arm, {})
                A.check(f"K1 {arm} degenerate count re-derived",
                        int(d["degenerate"].sum()) == int(ref.get("degenerate_count", -1)),
                        f"audit={int(d['degenerate'].sum())} artifact={ref.get('degenerate_count')}")
                A.check(f"K2 {arm} pos_rate mean re-derived",
                        close(float(d["pos_rate"].mean()), float(ref.get("pos_rate_mean", np.nan)), tol))
                A.check(f"K3 {arm} band width mean re-derived",
                        close(float(d["p_pos_band_width"].mean()),
                              float(ref.get("band_width_mean", np.nan)), tol))
            prim = int(DG[DG["arm"] == PRIMARY_ARM]["degenerate"].sum())
            A.check("K4 Criterion-6 references Exp 5's 24/25",
                    int(c6.get("exp5_reference", -1)) == EXP5_DEGENERACY)
            A.check("K5 Criterion-6 met == primary degeneracy below Exp 5's",
                    bool(c6.get("met")) == bool(prim < EXP5_DEGENERACY))
            A.note(f"Criterion 6: {PRIMARY_ARM} degenerate {prim}/25 "
                   f"(Exp 5 {EXP5_ARM} {EXP5_DEGENERACY}/25, "
                   f"Baseline-CV {BASELINE_DEGENERACY}/25)")

        # ---- L. acceptance criteria --------------------------------------
        crit = {c["id"]: c for c in S.get("acceptance_criteria", [])}
        A.check("L0 all seven criteria recorded", set(crit) == set(range(7)),
                f"found ids {sorted(crit)}")
        if set(crit) == set(range(7)):
            A.check("L1 C0 == L0 gate verdict", bool(crit[0]["met"]) == (not outside))
            pr_ = S["paired_vs_baseline"].get("PR_AUC", {})
            A.check("L2 C1 == PR-AUC positive and detectable",
                    bool(crit[1]["met"]) == bool(pr_.get("mean_delta", 0) > 0
                                                 and pr_.get("detectable", False)))
            rec_, f1_ = (S["paired_vs_baseline"].get("Recall", {}),
                         S["paired_vs_baseline"].get("F1", {}))
            A.check("L3 C2 == Recall and F1 above Baseline-CV",
                    bool(crit[2]["met"]) == bool(rec_.get("mean_delta", 0) > 0
                                                 and f1_.get("mean_delta", 0) > 0))
            e6r = S["aggregates"][PRIMARY_ARM]["Recall"]["mean"]
            e6f = S["aggregates"][PRIMARY_ARM]["F1"]["mean"]
            A.check("L4 C3 == Recall and F1 above Exp 5 W1",
                    bool(crit[3]["met"]) == bool(e6r > EXP5_RECALL and e6f > EXP5_F1))
            mae_ = S["paired_vs_baseline"].get("MAE", {})
            e6mae = S["aggregates"][PRIMARY_ARM]["MAE"]["mean"]
            A.check("L5 C4 == MAE within MDE AND below the mean-predictor bound",
                    bool(crit[4]["met"]) == bool(mae_.get("mean_delta", 0) <= MDE_MAE_BASELINE
                                                 and e6mae < BASELINE_MEAN_PREDICTOR_MAE))
            A.check("L6 C5 == Criterion-5 met", bool(crit[5]["met"]) == bool(c5.get("met")))
            A.check("L7 C6 == Criterion-6 met", bool(crit[6]["met"]) == bool(c6.get("met")))
            A.note(f"overall verdict recorded: {S.get('overall_verdict')}")
            for i in sorted(crit):
                A.note(f"criterion {i}: {'MET' if crit[i]['met'] else 'NOT MET'}")

            # ---- N. observational isolation ------------------------------
            gate_ok = bool(crit[0]["met"])
            expect_h1 = bool(gate_ok and crit[1]["met"] and crit[5]["met"])
            v = str(S.get("overall_verdict", ""))
            A.check("N1 verdict derives from C0, C1 and C5 only",
                    (v.startswith("H1") == expect_h1)
                    and (v.startswith("INVALID") == (not gate_ok)))
            A.check("N2 verdict_inputs recorded as C0/C1/C5",
                    S.get("verdict_inputs") == ["C0", "C1", "C5"])
            obs_d = S.get("observational_loss_decomposition", {})
            obs_o = S.get("observational_lambda_ordering", {})
            A.check("N3 loss-decomposition block flagged observational",
                    "not an acceptance criterion" in str(obs_d.get("note", "")).lower())
            A.check("N4 lambda-ordering block flagged observational",
                    "not an acceptance criterion" in str(obs_o.get("note", "")).lower())
            A.check("N5 no criterion references an observational block",
                    not any("observational" in str(c).lower() for c in crit.values()))

        # ---- M. decomposition aggregates in the summary -------------------
        obs_d = S.get("observational_loss_decomposition", {})
        if obs_d.get("available"):
            for arm in ARMS:
                ref = obs_d.get("by_arm", {}).get(arm, {})
                sub = DC[DC["arm"] == arm]
                for col in ["loss_cls", "lambda_loss_reg", "grad_norm_cls",
                            "grad_norm_reg"]:
                    if col in ref and col in sub.columns:
                        A.check(f"M8 {arm} {col} aggregate re-derived",
                                close(float(sub[col].mean()), float(ref[col]), 1e-9))
            A.check("M9 L3 zero-reference recorded as 0",
                    close(float(obs_d.get("zero_reference_grad_norm_reg", np.nan)),
                          0.0, 1e-12))

        A.check("H12 L3 regression caveat recorded",
                "descriptive only" in str(S.get("l3_regression_caveat", "")).lower())
        A.check("H13 Exp 5 comparator noted as like-for-like",
                "like-for-like" in str(S.get("exp5_reference", {}).get("note", "")).lower())
        A.check("H14 self-tests recorded at zero difference",
                close(S.get("selftests", {}).get("aggregation_max_diff", 1.0), 0.0, 1e-12))

    # ---- O. markdown summary ---------------------------------------------
    mdp = os.path.join(exp6, "exp6_summary.md")
    if A.check("O0 exp6_summary.md present", os.path.isfile(mdp)):
        md = open(mdp, encoding="utf-8").read()
        A.check("O1 markdown names the experiment", "Loss-Term Rebalancing" in md)
        A.check("O2 markdown reports the L0 gate", "L0 equivalence" in md)
        A.check("O3 markdown flags observational blocks",
                "NOT acceptance criteria" in md or "NOT an acceptance criterion" in md)
        A.check("O4 markdown carries the L3 caveat", "UNSUPERVISED" in md)
        A.check("O5 markdown states criteria arm",
                f"C1-C6 are evaluated on the primary arm {PRIMARY_ARM}" in md)
        A.check("O6 markdown lists all seven criteria",
                all(f"| {i} |" in md for i in range(7)))
        A.check("O7 markdown reports both paired comparisons",
                "vs Baseline-CV" in md and f"vs Exp 5 {EXP5_ARM}" in md)

    # ---- P. containment ----------------------------------------------------
    for frozen in [base, exp5, "trainer_outputs/exp4_decision_rule",
                   "trainer_outputs/exp3_convergence"]:
        if not os.path.isdir(frozen):
            continue
        stray = [fn for _r, _d, fs in os.walk(frozen) for fn in fs if "exp6" in fn.lower()]
        A.check(f"P1 no Exp 6 artifact inside {os.path.basename(frozen)}",
                not stray, f"stray {stray}")
    required = {os.path.basename(p) for p in FROZEN_INPUT_SHA
                if os.path.dirname(p).endswith("baseline_cv")}
    top = {f for f in os.listdir(base) if os.path.isfile(os.path.join(base, f))}
    A.check("P2 required baseline inputs present", required <= top,
            f"missing {sorted(required - top)}")
    A.check("P3 baseline runs/ complete (5 metrics + 25 preds)",
            len(os.listdir(os.path.join(base, "runs"))) == 30,
            f"found {len(os.listdir(os.path.join(base, 'runs')))}")

    raise SystemExit(0 if A.report() else 1)


if __name__ == "__main__":
    main()
