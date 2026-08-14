#!/usr/bin/env python
"""
Phase 12 / Experiment 4 - independent post-hoc auditor.

Re-derives every Experiment 4 result from the FROZEN Baseline-CV predictions and
checks it against what run_exp4_decision_rule.py and aggregate_exp4.py actually
wrote. Its purpose is to catch a defect in those two scripts, so it deliberately
does NOT import them: every quantity it compares against is recomputed here from
first principles, out of the raw prediction CSVs.

That independence is the entire point. A verifier that reuses the code under
test can only confirm the code is self-consistent; this one can confirm it is
correct.

WHAT IS AUDITED
---------------
  A  artifact inventory          32 expected files, nothing unexpected
  B  frozen-input integrity      Baseline-CV SHAs unchanged, before and after
  C  leakage                     each LOFO pool provably excludes its own fold
  D  threshold sanity            every tau finite, inside its pool's range,
                                 and consistent across the recorded artifacts
  E  decision re-derivation      pred_class recomputed from (p_pos, tau)
  F  tau-invariants              ROC-AUC/PR-AUC/MAE/RMSE/pred_var == Baseline-CV
  G  baseline-arm reproduction   the degenerate argmax arm is reproduced exactly
  H  participant coverage        188 unique per repeat, order preserved
  I  aggregation re-derivation   summary means/CIs/paired deltas/Criterion 3
  J  containment                 nothing written outside the Exp 4 directory

All findings are collected; the audit does not stop at the first failure,
because knowing whether a defect is isolated or systemic is more useful than
knowing only that one exists. Exit code is non-zero if any check fails.

Usage (from the repository root):
    python exp4_decision_rule/verify_exp4.py
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
# Frozen constants - independently restated, NOT imported from the scripts
# under test. If a constant here disagrees with one there, that is itself a
# finding worth surfacing.
# --------------------------------------------------------------------------
FROZEN_INPUT_SHA = {
    "fold_manifest.json":
        "b9a7a91f1bdd43c78d3cd1ee0c36e4ce3fb8be4b0971dcff3ff62ee5af8fca2f",
    "trivial_control_arm.csv":
        "2ecbfee3225910034a959fe949c7ad027cfa7d41e9bef8040374578ff8c5d572",
    "baseline_cv_summary.json":
        "f065f5e1ff7f8e5ed4d122a8031c9cc12425802e756697d5512a915674871aac",
}

N_REPEATS = 5
N_FOLDS = 5
N_PARTICIPANTS = 188
N_POSITIVE = 45
CORPUS_PREVALENCE = N_POSITIVE / N_PARTICIPANTS
BASELINE_TAU = 0.5

PRIMARY_POLICY = "P1"
FITTED_POLICIES = ["P1", "P2"]          # policies whose tau is fitted on a pool
ALL_POLICIES = ["P1", "P2", "P3"]
BASELINE_ARM = "baseline"

THRESHOLD_METRICS = ["Precision", "Recall", "F1", "balAcc", "MCC", "Accuracy", "pos_rate"]
INVARIANT_METRICS = ["ROC_AUC", "PR_AUC", "MAE", "RMSE", "pred_var"]
PAIRED_METRICS = ["Precision", "Recall", "F1", "balAcc", "MCC", "Accuracy"]

TOL = 1e-9


# --------------------------------------------------------------------------
# Check registry
# --------------------------------------------------------------------------
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
        return ok

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
        print(f"\nEXP 4 AUDIT PASS : {ok}")
        return ok


def sha256(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for block in iter(lambda: fh.read(8192), b""):
            h.update(block)
    return h.hexdigest()


def close(a: float, b: float, tol: float = TOL) -> bool:
    """NaN-tolerant comparison: two NaNs are considered equal, matching the
    convention used for single-class folds where AUC is undefined."""
    if isinstance(a, float) and isinstance(b, float) and np.isnan(a) and np.isnan(b):
        return True
    return bool(abs(float(a) - float(b)) <= tol)


# --------------------------------------------------------------------------
# Independent recomputation primitives
# --------------------------------------------------------------------------
def recompute_threshold_metrics(y: np.ndarray, yhat: np.ndarray) -> Dict[str, float]:
    """Threshold-conditional metrics, recomputed from scratch."""
    two_class = len(np.unique(y)) > 1
    return {
        "Precision": float(precision_score(y, yhat, zero_division=0)),
        "Recall": float(recall_score(y, yhat, zero_division=0)),
        "F1": float(f1_score(y, yhat, zero_division=0)),
        "balAcc": float(balanced_accuracy_score(y, yhat)),
        "MCC": float(matthews_corrcoef(y, yhat)) if two_class else 0.0,
        "Accuracy": float(accuracy_score(y, yhat)),
        "pos_rate": float(np.mean(yhat)),
    }


def recompute_invariants(y: np.ndarray, p_pos: np.ndarray,
                         pred_phq: np.ndarray, phq: np.ndarray) -> Dict[str, float]:
    """tau-invariant metrics, recomputed from scratch (numpy var uses ddof=0,
    matching the frozen Baseline-CV driver)."""
    two_class = len(np.unique(y)) > 1
    yt = np.asarray(phq, dtype=float)
    pp = np.asarray(pred_phq, dtype=float)
    return {
        "ROC_AUC": float(roc_auc_score(y, p_pos)) if two_class else float("nan"),
        "PR_AUC": float(average_precision_score(y, p_pos)) if two_class else float("nan"),
        "MAE": float(mean_absolute_error(yt, pp)),
        "RMSE": float(np.sqrt(np.mean((yt - pp) ** 2))),
        "pred_var": float(np.var(pp)),
    }


def fold_ci(values: Sequence[float]) -> Dict[str, float]:
    """Fold-level Student-t interval, recomputed independently."""
    v = np.asarray(values, dtype=float)
    k = len(v)
    m = float(np.mean(v))
    sd = float(np.std(v, ddof=1)) if k > 1 else 0.0
    se = sd / np.sqrt(k) if k > 1 else 0.0
    tc = float(stats.t.ppf(0.975, k - 1)) if k > 1 else float("nan")
    return {"mean": m, "fold_sd": sd, "se": se, "t_crit": tc,
            "ci95_lo": m - tc * se, "ci95_hi": m + tc * se}


# --------------------------------------------------------------------------
# Main audit
# --------------------------------------------------------------------------
def main() -> None:
    ap = argparse.ArgumentParser(
        description="Phase 12 / Experiment 4 - independent post-hoc audit.")
    ap.add_argument("--exp4-dir", default="trainer_outputs/exp4_decision_rule")
    ap.add_argument("--baseline-dir", default="trainer_outputs/baseline_cv")
    ap.add_argument("--tol", type=float, default=TOL,
                    help="absolute tolerance for numeric re-derivation")
    args = ap.parse_args()

    exp4_dir = os.path.normpath(args.exp4_dir)
    baseline_dir = os.path.normpath(args.baseline_dir)
    runs_dir = os.path.join(baseline_dir, "runs")
    tol = args.tol
    A = Audit()

    print("=" * 74)
    print("PHASE 12 / EXPERIMENT 4 - INDEPENDENT AUDIT")
    print(f"  exp4 dir             : {exp4_dir}")
    print(f"  baseline (read-only) : {baseline_dir}")
    print(f"  tolerance            : {tol:.1e}")
    print("=" * 74)

    if not os.path.isdir(exp4_dir):
        print(f"\n[ABORT] {exp4_dir} not found - run the runner and aggregator first",
              file=sys.stderr)
        raise SystemExit(1)

    # ---- A. artifact inventory -------------------------------------------
    expected = ({"thresholds.csv", "exp4_metrics.csv", "threshold_sweep.csv",
                 "invariants_check.json", "exp4_summary.json", "exp4_summary.md"}
                | {f"rep{r}_fold{f}_rescored.csv"
                   for r in range(1, N_REPEATS + 1) for f in range(1, N_FOLDS + 1)})
    present = set(os.listdir(exp4_dir))
    A.check("A1 all expected artifacts present", expected <= present,
            f"missing {sorted(expected - present)}")
    A.check("A2 no unexpected artifacts", not (present - expected),
            f"unexpected {sorted(present - expected)}")
    # Derived, not hard-coded: the expected count is a property of the expected
    # SET (6 named artifacts + 5x5 rescored CSVs), so it cannot drift out of
    # step with it. A literal here was previously wrong (32 vs the true 31) and
    # made this assertion unsatisfiable even on a perfect run.
    A.check(f"A3 artifact count == {len(expected)} (derived)",
            len(expected & present) == len(expected),
            f"found {len(expected & present)}")

    # ---- B. frozen-input integrity ---------------------------------------
    for name, want in FROZEN_INPUT_SHA.items():
        path = os.path.join(baseline_dir, name)
        got = sha256(path) if os.path.isfile(path) else "MISSING"
        A.check(f"B1 {name} SHA frozen", got == want, f"got {got}")

    inv_path = os.path.join(exp4_dir, "invariants_check.json")
    inv_report: Dict = {}
    if os.path.isfile(inv_path):
        with open(inv_path, "r", encoding="utf-8") as fh:
            inv_report = json.load(fh)
        before = inv_report.get("input_sha_before", {})
        after = inv_report.get("input_sha_after", {})
        A.check("B2 runner recorded before==after SHAs", before == after and bool(before))
        mismatched = [k for k, v in before.items()
                      if os.path.isfile(os.path.join(baseline_dir, k))
                      and sha256(os.path.join(baseline_dir, k)) != v]
        A.check("B3 recorded SHAs still match on disk", not mismatched,
                f"drifted: {mismatched}")
        A.check("B4 runner reported invariants within tolerance",
                bool(inv_report.get("all_within_tolerance", False)))
    else:
        A.check("B2 invariants_check.json present", False)

    # ---- load raw frozen inputs -------------------------------------------
    manifest_path = os.path.join(baseline_dir, "fold_manifest.json")
    with open(manifest_path, "r", encoding="utf-8") as fh:
        manifest = json.load(fh)

    raw: Dict[Tuple[int, int], pd.DataFrame] = {}
    for r in range(1, N_REPEATS + 1):
        for f in range(1, N_FOLDS + 1):
            raw[(r, f)] = pd.read_csv(os.path.join(runs_dir, f"rep{r}_fold{f}_preds.csv"))
    base_metrics = pd.concat(
        [pd.read_csv(os.path.join(runs_dir, f"rep{r}_metrics.csv"))
         for r in range(1, N_REPEATS + 1)], ignore_index=True)

    thresholds = pd.read_csv(os.path.join(exp4_dir, "thresholds.csv"))
    exp4_metrics = pd.read_csv(os.path.join(exp4_dir, "exp4_metrics.csv"))

    A.check("A4 thresholds.csv row count == 75",
            len(thresholds) == N_REPEATS * N_FOLDS * len(ALL_POLICIES),
            f"found {len(thresholds)}")
    A.check("A5 exp4_metrics.csv row count == 100",
            len(exp4_metrics) == N_REPEATS * N_FOLDS * (len(ALL_POLICIES) + 1),
            f"found {len(exp4_metrics)}")

    # ---- C-H. per fold-run re-derivation ----------------------------------
    coverage: Dict[int, List[int]] = {r: [] for r in range(1, N_REPEATS + 1)}

    for r in range(1, N_REPEATS + 1):
        for f in range(1, N_FOLDS + 1):
            tag = f"rep{r}fold{f}"
            df = raw[(r, f)]
            y = df["phq_bin"].to_numpy(dtype=int)
            p = df["p_pos"].to_numpy(dtype=float)
            pred_phq = df["pred_phq"].to_numpy(dtype=float)
            phq = df["phq"].to_numpy(dtype=float)
            ids = [int(x) for x in df["participant_id"]]
            coverage[r].extend(ids)

            # H. order and identity preserved against the manifest
            A.check(f"H1 {tag} participant order == manifest test_ids",
                    ids == [int(x) for x in manifest["folds"][f - 1]["test_ids"]])

            # C. leakage - rebuild the pool independently
            pool_ids: List[int] = []
            pool_y: List[int] = []
            pool_p: List[float] = []
            for g in range(1, N_FOLDS + 1):
                if g == f:
                    continue
                d2 = raw[(r, g)]
                pool_ids.extend(int(x) for x in d2["participant_id"])
                pool_y.extend(int(x) for x in d2["phq_bin"])
                pool_p.extend(float(x) for x in d2["p_pos"])
            A.check(f"C1 {tag} pool excludes evaluated participants",
                    not (set(ids) & set(pool_ids)))

            trow = thresholds[(thresholds["repeat"] == r) & (thresholds["fold"] == f)]
            for pol in ALL_POLICIES:
                pr = trow[trow["policy"] == pol]
                if not A.check(f"D0 {tag} {pol} threshold row present", len(pr) == 1):
                    continue
                pr = pr.iloc[0]
                tau = float(pr["tau"])

                # C. recorded pool provenance must exclude this fold
                recorded = [int(x) for x in str(pr["pool_folds"]).split("|") if x != ""]
                if pol in FITTED_POLICIES:
                    A.check(f"C2 {tag} {pol} pool_folds excludes fold {f}",
                            f not in recorded, f"recorded {recorded}")
                    A.check(f"C3 {tag} {pol} pool_size matches recomputation",
                            int(pr["pool_size"]) == len(pool_ids))
                    A.check(f"C4 {tag} {pol} pool_n_pos matches recomputation",
                            int(pr["pool_n_pos"]) == int(sum(pool_y)))

                # D. threshold sanity
                A.check(f"D1 {tag} {pol} tau finite", np.isfinite(tau))
                if pol in FITTED_POLICIES:
                    A.check(f"D2 {tag} {pol} tau within pool range",
                            min(pool_p) <= tau <= max(pool_p),
                            f"tau={tau:.6f} pool=[{min(pool_p):.6f},{max(pool_p):.6f}]")
                else:
                    A.check(f"D3 {tag} P3 tau == corpus prevalence",
                            close(tau, CORPUS_PREVALENCE, 1e-12))

                # E. decisions re-derived from (p_pos, tau)
                expect = (p >= tau).astype(int)
                res = pd.read_csv(os.path.join(exp4_dir, f"rep{r}_fold{f}_rescored.csv"))
                col = f"pred_class_{pol}"
                if A.check(f"E0 {tag} {col} present", col in res.columns):
                    A.check(f"E1 {tag} {col} reproduces (p_pos >= tau)",
                            np.array_equal(res[col].to_numpy(dtype=int), expect))

                # E. metrics re-derived
                mrow = exp4_metrics[(exp4_metrics["repeat"] == r)
                                    & (exp4_metrics["fold"] == f)
                                    & (exp4_metrics["policy"] == pol)]
                if A.check(f"E2 {tag} {pol} metric row present", len(mrow) == 1):
                    mrow = mrow.iloc[0]
                    rec = recompute_threshold_metrics(y, expect)
                    for k, v in rec.items():
                        A.check(f"E3 {tag} {pol} {k} re-derived",
                                close(v, float(mrow[k]), tol),
                                f"audit={v!r} artifact={float(mrow[k])!r}")
                    A.check(f"D4 {tag} {pol} tau consistent across artifacts",
                            close(tau, float(mrow["tau"]), 1e-12))

            # G. baseline arm - degenerate reproduction
            res = pd.read_csv(os.path.join(exp4_dir, f"rep{r}_fold{f}_rescored.csv"))
            base_expect = (p >= BASELINE_TAU).astype(int)
            A.check(f"G1 {tag} pred_class_baseline == (p_pos >= 0.5)",
                    np.array_equal(res["pred_class_baseline"].to_numpy(dtype=int),
                                   base_expect))
            A.check(f"G2 {tag} baseline arm == stored Baseline-CV pred_class",
                    np.array_equal(base_expect, df["pred_class"].to_numpy(dtype=int)))
            A.check(f"G3 {tag} baseline arm predicts no positives",
                    int(base_expect.sum()) == 0)

            # F. tau-invariants against the frozen Baseline-CV row
            recomputed = recompute_invariants(y, p, pred_phq, phq)
            bref = base_metrics[(base_metrics["repeat"] == r)
                                & (base_metrics["fold"] == f)]
            if A.check(f"F0 {tag} Baseline-CV row present", len(bref) == 1):
                bref = bref.iloc[0]
                for m in INVARIANT_METRICS:
                    A.check(f"F1 {tag} {m} == Baseline-CV",
                            close(recomputed[m], float(bref[m]), tol),
                            f"audit={recomputed[m]!r} baseline={float(bref[m])!r}")
                    for pol in [BASELINE_ARM] + ALL_POLICIES:
                        mr = exp4_metrics[(exp4_metrics["repeat"] == r)
                                          & (exp4_metrics["fold"] == f)
                                          & (exp4_metrics["policy"] == pol)]
                        if len(mr) == 1:
                            A.check(f"F2 {tag} {pol} {m} invariant across arms",
                                    close(recomputed[m], float(mr.iloc[0][m]), tol))

        A.check(f"H2 repeat {r} covers {N_PARTICIPANTS} unique participants",
                len(coverage[r]) == N_PARTICIPANTS
                and len(set(coverage[r])) == N_PARTICIPANTS,
                f"rows={len(coverage[r])} unique={len(set(coverage[r]))}")

    # ---- I. aggregation re-derivation --------------------------------------
    sum_path = os.path.join(exp4_dir, "exp4_summary.json")
    if A.check("I0 exp4_summary.json present", os.path.isfile(sum_path)):
        with open(sum_path, "r", encoding="utf-8") as fh:
            summary = json.load(fh)

        A.check("I1 summary declares 5 repeats x 5 folds",
                summary.get("n_repeats") == N_REPEATS
                and summary.get("n_folds") == N_FOLDS)
        A.check("I2 summary t_crit == t(0.975, 4)",
                close(float(summary.get("t_crit", 0)),
                      float(stats.t.ppf(0.975, N_FOLDS - 1)), 1e-9))
        A.check("I3 primary policy is P1",
                summary.get("primary_policy") == PRIMARY_POLICY)

        prim = exp4_metrics[exp4_metrics["policy"] == PRIMARY_POLICY]
        pf = prim.groupby("fold")[THRESHOLD_METRICS].mean().sort_index()
        for m in THRESHOLD_METRICS:
            got = fold_ci(pf[m].values)
            ref = summary.get("aggregates", {}).get(PRIMARY_POLICY, {}).get(m)
            if A.check(f"I4 aggregate {m} present in summary", ref is not None):
                for key in ["mean", "fold_sd", "se", "ci95_lo", "ci95_hi"]:
                    A.check(f"I5 aggregate {m}.{key} re-derived",
                            close(got[key], float(ref[key]), tol),
                            f"audit={got[key]!r} artifact={float(ref[key])!r}")

        base_pf = base_metrics.groupby("fold")[PAIRED_METRICS].mean().sort_index()
        exp4_pf = prim.groupby("fold")[PAIRED_METRICS].mean().sort_index()
        for m in PAIRED_METRICS:
            deltas = exp4_pf[m].values - base_pf[m].values
            ref = summary.get("paired_vs_baseline", {}).get(m)
            if A.check(f"I6 paired {m} present in summary", ref is not None):
                A.check(f"I7 paired {m} mean_delta re-derived",
                        close(float(np.mean(deltas)), float(ref["mean_delta"]), tol))
                A.check(f"I8 paired {m} per-fold deltas re-derived",
                        all(close(a, b, tol) for a, b in
                            zip(deltas, ref.get("per_fold_delta", []))))

        c3 = summary.get("criterion3_precision_vs_prevalence", {})
        if A.check("I9 Criterion-3 block present", bool(c3)):
            prec = exp4_pf["Precision"].values
            d = prec - CORPUS_PREVALENCE
            A.check("I10 Criterion-3 mean delta re-derived",
                    close(float(np.mean(d)), float(c3.get("mean_delta", np.nan)), tol))
            A.check("I11 Criterion-3 mu0 == corpus prevalence",
                    close(float(c3.get("mu0", np.nan)), CORPUS_PREVALENCE, 1e-12))
            expect_exceeds = bool(np.mean(d) > 0 and float(c3.get("ci95_lo", -1)) > 0)
            A.check("I12 Criterion-3 verdict consistent with its own CI",
                    bool(c3.get("exceeds_prevalence")) == expect_exceeds)
            A.note(f"Criterion 3: mean precision {np.mean(prec):.4f} vs prevalence "
                   f"{CORPUS_PREVALENCE:.4f}, exceeds={c3.get('exceeds_prevalence')}")

        A.note(f"Overall verdict recorded: {summary.get('overall_verdict')}")

    # ---- J. containment -----------------------------------------------------
    stray = []
    for root, _dirs, files in os.walk(baseline_dir):
        for fn in files:
            rel = os.path.relpath(os.path.join(root, fn), baseline_dir)
            if rel.startswith("exp4") or "exp4" in rel:
                stray.append(rel)
    A.check("J1 no Exp 4 artifact written inside baseline dir", not stray,
            f"stray {stray}")
    n_baseline_files = sum(len(fs) for _, _, fs in os.walk(baseline_dir))
    A.check("J2 baseline dir still has 34 files", n_baseline_files == 34,
            f"found {n_baseline_files}")

    ok = A.report()
    raise SystemExit(0 if ok else 1)


if __name__ == "__main__":
    main()
