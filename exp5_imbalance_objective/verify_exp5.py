#!/usr/bin/env python
"""
Phase 14 / Experiment 5 - independent post-hoc auditor.

Re-derives every Experiment 5 result from the raw per-fold prediction files and
checks it against what run_exp5.py and aggregate_exp5.py actually wrote. Its
purpose is to catch a defect in those two scripts, so it deliberately does NOT
import them: every quantity it compares against is recomputed here from first
principles.

A verifier that reuses the code under test can only confirm the code is
self-consistent; this one can confirm it is correct.

It also does not import torch or transformers. Auditing does not require a model.

WHAT IS AUDITED
---------------
  A  artifact inventory        expected set DERIVED from the run design, never a
                               hard-coded literal (literals produced false
                               failures in the Exp 4 `A3` and Exp 3 `J2` auditors)
  B  frozen-input integrity    Baseline-CV / Exp 3 / Exp 4 / trainer SHAs unchanged
  C  fold integrity            participant order, coverage, train/test disjoint,
                               argmax decision rule reproduced from p_pos
  D  metric re-derivation      every metric recomputed from raw predictions
  E  class weights             each arm's weights re-derived from TRAINING labels
                               only, per the pre-registered W0-W3 formulas
  F  W0 equivalence gate       independently re-evaluated against Baseline-CV CIs
  G  degeneracy diagnostics    band width, pos_rate, degeneracy from raw preds
  H  aggregation re-derivation summary means / CIs / paired deltas
  I  Criterion 5               precision-vs-prevalence and F1-vs-random-ranker,
                               the latter evaluated PER FOLD-RUN (Jensen)
  J  Criterion 6               degeneracy counts vs Exp 4's 24/25
  K  acceptance criteria       every recorded verdict re-derived
  L  containment               nothing written outside the Experiment 5 directory

All findings are collected; the audit does not stop at the first failure, because
knowing whether a defect is isolated or systemic is more useful than knowing only
that one exists. Exit code is non-zero if any check fails.

Usage (from the repository root):
    python exp5_imbalance_objective/verify_exp5.py
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
}

ARMS: List[str] = ["W0", "W1", "W2", "W3"]
PRIMARY_ARM = "W1"
CONTROL_ARM = "W0"
N_REPEATS = 5
N_FOLDS = 5
N_PARTICIPANTS = 188
N_POSITIVE = 45
CORPUS_PREVALENCE = N_POSITIVE / N_PARTICIPANTS
BINARIZE_THRESHOLD = 10.0
ARGMAX_TAU = 0.5

MDE_MAE_BASELINE = 0.4066
EXP4_RECALL = 0.5422
EXP4_F1 = 0.2215
EXP4_DEGENERACY = 24
BASELINE_DEGENERACY = 25

PRIMARY_METRICS = ["MAE", "RMSE", "pred_var", "MAE_mean_pred", "ROC_AUC", "PR_AUC",
                   "balAcc", "MCC", "Precision", "Recall", "F1", "Accuracy"]
PAIRED_METRICS = ["MAE", "RMSE", "pred_var", "ROC_AUC", "PR_AUC",
                  "balAcc", "MCC", "Precision", "Recall", "F1", "Accuracy"]
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
        print(f"\nEXP 5 AUDIT PASS : {ok}")
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


def recompute_class_weights(arm: str, y_train: np.ndarray) -> Dict[str, float]:
    """Independent reimplementation of the pre-registered W0-W3 formulas
    (design section 7). Derived from TRAINING labels only."""
    n = int(len(y_train))
    n_pos = int(np.sum(y_train == 1))
    n_neg = int(np.sum(y_train == 0))
    if arm == "W0":
        w_neg = w_pos = 1.0
    elif arm == "W1":
        w_neg, w_pos = n / (2.0 * n_neg), n / (2.0 * n_pos)
    elif arm == "W2":
        w_neg = float(np.sqrt(n / (2.0 * n_neg)))
        w_pos = float(np.sqrt(n / (2.0 * n_pos)))
    elif arm == "W3":
        w_neg, w_pos = 1.0, 1.0 / CORPUS_PREVALENCE
    else:
        raise ValueError(arm)
    return {"w_neg": float(w_neg), "w_pos": float(w_pos),
            "n_train": n, "n_pos_train": n_pos, "n_neg_train": n_neg}


def random_ranker_f1(prevalence: float, pos_rate: float) -> float:
    """Expected F1 of a random ranker at the given positive rate.

    Evaluated PER FOLD-RUN using that fold's own prevalence, then averaged -
    never from an averaged pos_rate. The function is non-linear, so by Jensen's
    inequality the two differ materially (0.211901 vs 0.330089 on the Exp 4
    artifacts). Experiment 4 computed it per fold-run, so this audit does too.
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
    if sd == 0.0:
        detectable = bool(mean != 0.0)
    else:
        detectable = bool(abs(mean) > mde)
    return {"mean_delta": mean, "sd_delta": sd, "se": se, "mde_95": mde,
            "ci95_lo": mean - mde, "ci95_hi": mean + mde, "detectable": detectable,
            "zero_variance": sd == 0.0}


# --------------------------------------------------------------------------
# Main audit
# --------------------------------------------------------------------------
def main() -> None:
    ap = argparse.ArgumentParser(
        description="Phase 14 / Experiment 5 - independent post-hoc audit.")
    ap.add_argument("--exp5-dir", default="trainer_outputs/exp5_imbalance_objective")
    ap.add_argument("--baseline-dir", default="trainer_outputs/baseline_cv")
    ap.add_argument("--exp4-dir", default="trainer_outputs/exp4_decision_rule")
    ap.add_argument("--parquet", default="daic_records.parquet")
    ap.add_argument("--tol", type=float, default=TOL)
    a = ap.parse_args()

    exp5 = os.path.normpath(a.exp5_dir)
    base = os.path.normpath(a.baseline_dir)
    exp4 = os.path.normpath(a.exp4_dir)
    runs = os.path.join(base, "runs")
    tol = a.tol
    A = Audit()

    print("=" * 74)
    print("PHASE 14 / EXPERIMENT 5 - INDEPENDENT AUDIT")
    print(f"  exp5 dir             : {exp5}")
    print(f"  baseline (read-only) : {base}")
    print(f"  exp4     (read-only) : {exp4}")
    print(f"  tolerance            : {tol:.1e}")
    print("=" * 74)

    if not os.path.isdir(exp5):
        print(f"\n[ABORT] {exp5} not found - run the runner and aggregator first",
              file=sys.stderr)
        raise SystemExit(1)

    # ---- A. artifact inventory (DERIVED, never a literal) -----------------
    named = {"class_weights.csv", "exp5_metrics.csv", "w0_equivalence_check.json",
             "degeneracy_report.csv", "loop_correspondence.md",
             "exp5_summary.json", "exp5_summary.md"}
    preds = {f"{arm}_rep{r}_fold{f}_preds.csv"
             for arm in ARMS for r in range(1, N_REPEATS + 1)
             for f in range(1, N_FOLDS + 1)}
    mets = {f"{arm}_rep{r}_metrics.csv"
            for arm in ARMS for r in range(1, N_REPEATS + 1)}
    expected = named | preds | mets
    present = {f for f in os.listdir(exp5) if os.path.isfile(os.path.join(exp5, f))}
    A.check("A1 all expected artifacts present", expected <= present,
            f"missing {sorted(expected - present)[:6]}")
    A.check("A2 no unexpected artifacts", not (present - expected),
            f"unexpected {sorted(present - expected)[:6]}")
    A.check(f"A3 artifact count == {len(expected)} (derived from the run design)",
            len(expected & present) == len(expected),
            f"found {len(expected & present)}")
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
    M = pd.read_csv(os.path.join(exp5, "exp5_metrics.csv"))
    W = pd.read_csv(os.path.join(exp5, "class_weights.csv"))
    DG = pd.read_csv(os.path.join(exp5, "degeneracy_report.csv"))
    base_metrics = pd.concat([pd.read_csv(os.path.join(runs, f"rep{r}_metrics.csv"))
                              for r in range(1, N_REPEATS + 1)], ignore_index=True)

    n_expected_runs = len(ARMS) * N_REPEATS * N_FOLDS
    A.check(f"A4 exp5_metrics row count == {n_expected_runs}",
            len(M) == n_expected_runs, f"found {len(M)}")
    A.check(f"A5 class_weights row count == {n_expected_runs}",
            len(W) == n_expected_runs, f"found {len(W)}")
    A.check(f"A6 degeneracy_report row count == {n_expected_runs}",
            len(DG) == n_expected_runs, f"found {len(DG)}")

    # training-fold PHQ means, for MAE_mean_pred
    train_means: Dict[int, float] = {}
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
        phq_by_pid = {}

    # ---- C-G. per fold-run re-derivation ---------------------------------
    for arm in ARMS:
        for r in range(1, N_REPEATS + 1):
            seen: List[int] = []
            for f in range(1, N_FOLDS + 1):
                tag = f"{arm} rep{r} fold{f}"
                pth = os.path.join(exp5, f"{arm}_rep{r}_fold{f}_preds.csv")
                if not A.check(f"C1 {tag} preds file present", os.path.isfile(pth)):
                    continue
                d = pd.read_csv(pth)

                spec = manifest["folds"][f - 1]
                ids = [int(x) for x in d["participant_id"]]
                seen.extend(ids)
                A.check(f"C2 {tag} participant order == manifest test_ids",
                        ids == [int(x) for x in spec["test_ids"]])
                A.check(f"C3 {tag} no train/test overlap",
                        not (set(ids) & set(int(x) for x in spec["train_ids"])))

                y_phq = d["phq"].to_numpy(float)
                y_bin = (y_phq > BINARIZE_THRESHOLD).astype(int)
                A.check(f"C4 {tag} phq_bin consistent with PHQ > 10.0",
                        np.array_equal(y_bin, d["phq_bin"].to_numpy(int)))
                pred_phq = d["pred_phq"].to_numpy(float)
                p_pos = d["p_pos"].to_numpy(float)
                pred_class = d["pred_class"].to_numpy(int)
                # The decision rule must remain argmax - Exp 4 was H0, so its
                # threshold policy was NOT accepted into the configuration.
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

                # E. class weights re-derived from TRAINING labels only
                wrow = W[(W["arm"] == arm) & (W["repeat"] == r) & (W["fold"] == f)]
                if A.check(f"E0 {tag} weight row present", len(wrow) == 1):
                    wrow = wrow.iloc[0]
                    if phq_by_pid:
                        y_tr = np.array([1 if phq_by_pid[int(i)] > BINARIZE_THRESHOLD else 0
                                         for i in spec["train_ids"]], int)
                        exp_w = recompute_class_weights(arm, y_tr)
                        for k_ in ["w_neg", "w_pos", "n_train", "n_pos_train", "n_neg_train"]:
                            A.check(f"E1 {tag} {k_} re-derived",
                                    close(exp_w[k_], wrow[k_], tol),
                                    f"audit={exp_w[k_]!r} artifact={wrow[k_]!r}")
                        # the evaluated fold must not influence its own weights
                        A.check(f"E2 {tag} n_train excludes the held-out fold",
                                int(wrow["n_train"]) == len(spec["train_ids"]))
                    if arm == CONTROL_ARM:
                        A.check(f"E3 {tag} W0 weights are uniform",
                                close(wrow["w_neg"], 1.0, tol) and close(wrow["w_pos"], 1.0, tol))

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

            A.check(f"C6 {arm} rep{r} covers {N_PARTICIPANTS} unique participants",
                    len(seen) == N_PARTICIPANTS and len(set(seen)) == N_PARTICIPANTS,
                    f"rows={len(seen)} unique={len(set(seen))}")

    # ---- F. W0 equivalence gate, independently re-evaluated ---------------
    with open(os.path.join(base, "baseline_cv_summary.json"), "r", encoding="utf-8") as fh:
        published = json.load(fh)["metrics"]
    w0 = M[M["arm"] == CONTROL_ARM]
    outside: List[str] = []
    if A.check("F0 W0 arm complete", len(w0) == N_REPEATS * N_FOLDS):
        pf = w0.groupby("fold")[PRIMARY_METRICS].mean().sort_index()
        for m in PRIMARY_METRICS:
            if m not in published:
                continue
            got = fold_ci(pf[m].values)
            lo, hi = float(published[m]["ci95"][0]), float(published[m]["ci95"][1])
            if not (lo <= got["mean"] <= hi):
                outside.append(m)
        A.check("F1 W0 aggregate inside Baseline-CV 95% CI on all primary metrics",
                not outside, f"outside: {outside}")
        gp = os.path.join(exp5, "w0_equivalence_check.json")
        if A.check("F2 w0_equivalence_check.json present", os.path.isfile(gp)):
            with open(gp, "r", encoding="utf-8") as fh:
                rep = json.load(fh)
            A.check("F3 recorded gate verdict matches re-derivation",
                    bool(rep.get("passed")) == (not outside))
            A.check("F4 gate is flagged authoritative", bool(rep.get("authoritative")))

    # ---- H-K. aggregation, criteria -------------------------------------
    sp = os.path.join(exp5, "exp5_summary.json")
    if A.check("H0 exp5_summary.json present", os.path.isfile(sp)):
        with open(sp, "r", encoding="utf-8") as fh:
            S = json.load(fh)

        A.check("H1 primary arm is W1", S.get("primary_arm") == PRIMARY_ARM)
        A.check("H2 control arm is W0", S.get("control_arm") == CONTROL_ARM)
        A.check("H3 t_crit == t(0.975, 4)",
                close(S.get("t_crit", 0), float(stats.t.ppf(0.975, N_FOLDS - 1)), 1e-9))
        A.check("H4 prevalence recorded exactly",
                close(S.get("prevalence", 0), CORPUS_PREVALENCE, 1e-15))

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
            if A.check(f"H7 paired {m} present", ref is not None):
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

        # ---- I. Criterion 5 ----------------------------------------------
        c5 = S.get("criterion5_discrimination", {})
        if A.check("I0 Criterion-5 block present", bool(c5)):
            prec_pf = epf["Precision"].values
            pt = paired_test(prec_pf, CORPUS_PREVALENCE)
            A.check("I1 precision-vs-prevalence mean delta re-derived",
                    close(pt["mean_delta"],
                          float(c5["precision_vs_prevalence"]["mean_delta"]), tol))
            expect_prec = bool(pt["mean_delta"] > 0 and pt["ci95_lo"] > 0)
            A.check("I2 precision_exceeds_prevalence consistent with its own CI",
                    bool(c5.get("precision_exceeds_prevalence")) == expect_prec)
            A.check("I3 Criterion-5 mu0 == corpus prevalence",
                    close(float(c5["precision_vs_prevalence"].get("mu0", np.nan)),
                          CORPUS_PREVALENCE, 1e-15))

            # random-ranker control: PER FOLD-RUN, then averaged (Jensen)
            dgp = DG[DG["arm"] == PRIMARY_ARM].copy()
            dgp["rand_F1"] = [random_ranker_f1(float(p), float(q)) for p, q in
                              zip(dgp["prevalence"], dgp["pos_rate"])]
            rand_pf = dgp.groupby("fold")["rand_F1"].mean().sort_index().values
            A.check("I4 random-ranker F1 mean re-derived (per fold-run, then averaged)",
                    close(float(np.mean(rand_pf)),
                          float(c5.get("random_ranker_f1_mean", np.nan)), tol),
                    f"audit={float(np.mean(rand_pf)):.6f} "
                    f"artifact={c5.get('random_ranker_f1_mean')}")
            ft = paired_test(epf["F1"].values - rand_pf, 0.0)
            A.check("I5 F1-vs-random mean delta re-derived",
                    close(ft["mean_delta"], float(c5["f1_vs_random"]["mean_delta"]), tol))
            expect_f1 = bool(ft["mean_delta"] > 0 and ft["ci95_lo"] > 0)
            A.check("I6 f1_exceeds_random consistent with its own CI",
                    bool(c5.get("f1_exceeds_random")) == expect_f1)
            A.check("I7 Criterion-5 met == both sub-tests",
                    bool(c5.get("met")) == bool(expect_prec and expect_f1))
            A.note(f"Criterion 5: precision {np.mean(prec_pf):.4f} vs prevalence "
                   f"{CORPUS_PREVALENCE:.4f} (exceeds={c5.get('precision_exceeds_prevalence')}); "
                   f"F1 {np.mean(epf['F1'].values):.4f} vs random-ranker "
                   f"{np.mean(rand_pf):.4f} (exceeds={c5.get('f1_exceeds_random')})")

        # ---- J. Criterion 6 ----------------------------------------------
        c6 = S.get("criterion6_degeneracy", {})
        if A.check("J0 Criterion-6 block present", bool(c6)):
            for arm in ARMS:
                d = DG[DG["arm"] == arm]
                ref = c6.get("by_arm", {}).get(arm, {})
                A.check(f"J1 {arm} degenerate count re-derived",
                        int(d["degenerate"].sum()) == int(ref.get("degenerate_count", -1)),
                        f"audit={int(d['degenerate'].sum())} artifact={ref.get('degenerate_count')}")
                A.check(f"J2 {arm} pos_rate mean re-derived",
                        close(float(d["pos_rate"].mean()), float(ref.get("pos_rate_mean", np.nan)), tol))
            prim = int(DG[DG["arm"] == PRIMARY_ARM]["degenerate"].sum())
            A.check("J3 Criterion-6 references Exp 4's 24/25",
                    int(c6.get("exp4_reference", -1)) == EXP4_DEGENERACY)
            A.check("J4 Criterion-6 met == primary degeneracy below Exp 4's",
                    bool(c6.get("met")) == bool(prim < EXP4_DEGENERACY))
            A.note(f"Criterion 6: {PRIMARY_ARM} degenerate {prim}/25 "
                   f"(Exp 4 P1 {EXP4_DEGENERACY}/25, Baseline-CV {BASELINE_DEGENERACY}/25)")

        # ---- K. acceptance criteria --------------------------------------
        crit = {c["id"]: c for c in S.get("acceptance_criteria", [])}
        A.check("K0 all seven criteria recorded", set(crit) == set(range(7)),
                f"found ids {sorted(crit)}")
        if set(crit) == set(range(7)):
            A.check("K1 C0 == W0 gate verdict",
                    bool(crit[0]["met"]) == (not outside))
            pr_ = S["paired_vs_baseline"].get("PR_AUC", {})
            A.check("K2 C1 == PR-AUC positive and detectable",
                    bool(crit[1]["met"]) == bool(pr_.get("mean_delta", 0) > 0
                                                 and pr_.get("detectable", False)))
            rec_, f1_ = S["paired_vs_baseline"].get("Recall", {}), S["paired_vs_baseline"].get("F1", {})
            A.check("K3 C2 == Recall and F1 above Baseline-CV",
                    bool(crit[2]["met"]) == bool(rec_.get("mean_delta", 0) > 0
                                                 and f1_.get("mean_delta", 0) > 0))
            e5r = S["aggregates"][PRIMARY_ARM]["Recall"]["mean"]
            e5f = S["aggregates"][PRIMARY_ARM]["F1"]["mean"]
            A.check("K4 C3 == Recall and F1 above Exp 4 (threshold-conditional)",
                    bool(crit[3]["met"]) == bool(e5r > EXP4_RECALL and e5f > EXP4_F1))
            mae_ = S["paired_vs_baseline"].get("MAE", {})
            A.check("K5 C4 == MAE not degraded beyond MDE",
                    bool(crit[4]["met"]) == bool(mae_.get("mean_delta", 0) <= MDE_MAE_BASELINE))
            A.check("K6 C5 == Criterion-5 met", bool(crit[5]["met"]) == bool(c5.get("met")))
            A.check("K7 C6 == Criterion-6 met", bool(crit[6]["met"]) == bool(c6.get("met")))
            A.note(f"overall verdict recorded: {S.get('overall_verdict')}")
            for i in sorted(crit):
                A.note(f"criterion {i}: {'MET' if crit[i]['met'] else 'NOT MET'}")

        A.check("H12 Exp 4 caveat recorded",
                "threshold" in str(S.get("exp4_reference", {}).get("caveat", "")).lower())

    # ---- L. containment ----------------------------------------------------
    for frozen, n_expected in [(base, 34), (exp4, 31)]:
        stray = [fn for _r, _d, fs in os.walk(frozen) for fn in fs if "exp5" in fn.lower()]
        A.check(f"L1 no Exp 5 artifact inside {os.path.basename(frozen)}",
                not stray, f"stray {stray}")
    # required frozen inputs still present (derived, not a raw recursive count)
    required = {os.path.basename(p) for p in FROZEN_INPUT_SHA
                if os.path.dirname(p).endswith("baseline_cv")}
    top = {f for f in os.listdir(base) if os.path.isfile(os.path.join(base, f))}
    A.check("L2 required baseline inputs present", required <= top,
            f"missing {sorted(required - top)}")
    A.check("L3 baseline runs/ complete (5 metrics + 25 preds)",
            len(os.listdir(os.path.join(base, "runs"))) == 30,
            f"found {len(os.listdir(os.path.join(base, 'runs')))}")

    raise SystemExit(0 if A.report() else 1)


if __name__ == "__main__":
    main()
