#!/usr/bin/env python
"""
Phase 13 / Experiment 3 - independent post-hoc auditor.

Re-derives every Experiment 3 result from the raw per-fold prediction files and
checks it against what run_exp3.py and aggregate_exp3.py actually wrote. Its
purpose is to catch a defect in those two scripts, so it deliberately does NOT
import them: every quantity it compares against is recomputed here from first
principles.

A verifier that reuses the code under test can only confirm the code is
self-consistent; this one can confirm it is correct.

It also does not import torch or transformers. Auditing does not require a model.

WHAT IS AUDITED
---------------
  A  artifact inventory        expected set DERIVED from the run design, never a
                               hard-coded literal (a literal produced a false
                               failure in Experiment 4's auditor)
  B  frozen-input integrity    Baseline-CV + trainer SHAs unchanged
  C  fold integrity            participant order, coverage, train/test disjoint
  D  metric re-derivation      every metric recomputed from raw predictions
  E  loss trajectories         one row per (arm, repeat, fold, epoch); budgets honoured
  F  convergence re-derivation plateau rule reapplied to the raw trajectories
  G  diagnostics re-derivation band width, pos_rate, degeneracy from raw preds
  H  E0 equivalence gate       independently re-evaluated against Baseline-CV CIs
  I  aggregation re-derivation summary means/CIs/paired deltas
  J  containment               nothing written outside the Experiment 3 directory

All findings are collected; the audit does not stop at the first failure, because
knowing whether a defect is isolated or systemic is more useful than knowing only
that one exists. Exit code is non-zero if any check fails.

Usage (from the repository root):
    python exp3_convergence/verify_exp3.py
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
    "fold_manifest.json":
        "b9a7a91f1bdd43c78d3cd1ee0c36e4ce3fb8be4b0971dcff3ff62ee5af8fca2f",
    "trivial_control_arm.csv":
        "2ecbfee3225910034a959fe949c7ad027cfa7d41e9bef8040374578ff8c5d572",
    "baseline_cv_summary.json":
        "f065f5e1ff7f8e5ed4d122a8031c9cc12425802e756697d5512a915674871aac",
}

ARMS: Dict[str, int] = {"E0": 3, "E1": 20, "E2": 10}
PRIMARY_ARM = "E1"
CONTROL_ARM = "E0"
N_REPEATS = 5
N_FOLDS = 5
N_PARTICIPANTS = 188
BINARIZE_THRESHOLD = 10.0
PLATEAU_REL_TOL = 0.01
PLATEAU_PATIENCE = 2

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
        print(f"\nEXP 3 AUDIT PASS : {ok}")
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


def recompute_metrics(y_phq, y_bin, pred_phq, p_pos, pred_class, train_mean) -> Dict[str, float]:
    """The full metric suite, recomputed from scratch."""
    two = len(np.unique(y_bin)) > 1
    return {
        "MAE": float(mean_absolute_error(y_phq, pred_phq)),
        "RMSE": float(np.sqrt(np.mean((np.asarray(y_phq, float) - np.asarray(pred_phq, float)) ** 2))),
        "pred_var": float(np.var(pred_phq)),
        "MAE_mean_pred": float(mean_absolute_error(y_phq, np.full(len(y_phq), train_mean))),
        "ROC_AUC": float(roc_auc_score(y_bin, p_pos)) if two else float("nan"),
        "PR_AUC": float(average_precision_score(y_bin, p_pos)) if two else float("nan"),
        "balAcc": float(balanced_accuracy_score(y_bin, pred_class)),
        "MCC": float(matthews_corrcoef(y_bin, pred_class)) if two else 0.0,
        "Precision": float(precision_score(y_bin, pred_class, zero_division=0)),
        "Recall": float(recall_score(y_bin, pred_class, zero_division=0)),
        "F1": float(f1_score(y_bin, pred_class, zero_division=0)),
        "Accuracy": float(accuracy_score(y_bin, pred_class)),
    }


def fold_ci(values: Sequence[float]) -> Dict[str, float]:
    v = np.asarray(values, dtype=float)
    k = len(v)
    m = float(np.mean(v))
    sd = float(np.std(v, ddof=1)) if k > 1 else 0.0
    se = sd / np.sqrt(k) if k > 1 else 0.0
    tc = float(stats.t.ppf(0.975, k - 1)) if k > 1 else float("nan")
    return {"mean": m, "fold_sd": sd, "se": se, "t_crit": tc,
            "ci95_lo": m - tc * se, "ci95_hi": m + tc * se}


def plateau_epoch(losses: Sequence[float]) -> object:
    """Independent reimplementation of the pre-declared plateau rule."""
    streak = 0
    for k in range(1, len(losses)):
        prev, cur = losses[k - 1], losses[k]
        rel = (prev - cur) / prev if prev != 0 else 0.0
        if rel < PLATEAU_REL_TOL:
            streak += 1
            if streak >= PLATEAU_PATIENCE:
                return k + 1
        else:
            streak = 0
    return None


def main() -> None:
    ap = argparse.ArgumentParser(
        description="Phase 13 / Experiment 3 - independent post-hoc audit.")
    ap.add_argument("--exp3-dir", default="trainer_outputs/exp3_convergence")
    ap.add_argument("--baseline-dir", default="trainer_outputs/baseline_cv")
    ap.add_argument("--parquet", default="daic_records.parquet")
    ap.add_argument("--tol", type=float, default=TOL)
    a = ap.parse_args()

    exp3 = os.path.normpath(a.exp3_dir)
    base = os.path.normpath(a.baseline_dir)
    runs = os.path.join(base, "runs")
    tol = a.tol
    A = Audit()

    print("=" * 74)
    print("PHASE 13 / EXPERIMENT 3 - INDEPENDENT AUDIT")
    print(f"  exp3 dir             : {exp3}")
    print(f"  baseline (read-only) : {base}")
    print(f"  tolerance            : {tol:.1e}")
    print("=" * 74)

    if not os.path.isdir(exp3):
        print(f"\n[ABORT] {exp3} not found - run the runner and aggregator first",
              file=sys.stderr)
        raise SystemExit(1)

    # ---- A. artifact inventory (DERIVED, never a literal) -----------------
    named = {"loss_trajectories.csv", "exp3_metrics.csv", "e0_equivalence_check.json",
             "convergence_report.csv", "distribution_diagnostics.csv",
             "exp3_summary.json", "exp3_summary.md"}
    preds = {f"{arm}_rep{r}_fold{f}_preds.csv"
             for arm in ARMS for r in range(1, N_REPEATS + 1)
             for f in range(1, N_FOLDS + 1)}
    mets = {f"{arm}_rep{r}_metrics.csv"
            for arm in ARMS for r in range(1, N_REPEATS + 1)}
    expected = named | preds | mets
    present = set(os.listdir(exp3))
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
    for name, want in FROZEN_INPUT_SHA.items():
        p = os.path.join(base, name)
        got = sha256(p) if os.path.isfile(p) else "MISSING"
        A.check(f"B1 {name} SHA frozen", got == want, f"got {got}")
    tp = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(exp3))),
                      "trainer_mentalbert_daic.py")
    if not os.path.isfile(tp):
        tp = "trainer_mentalbert_daic.py"
    if os.path.isfile(tp):
        A.check("B2 trainer SHA frozen", sha256(tp) == FROZEN_TRAINER_SHA)
    else:
        A.check("B2 trainer file locatable", False, tp)

    # ---- load ------------------------------------------------------------
    with open(os.path.join(base, "fold_manifest.json"), "r", encoding="utf-8") as fh:
        manifest = json.load(fh)
    exp3_df = pd.read_csv(os.path.join(exp3, "exp3_metrics.csv"))
    traj = pd.read_csv(os.path.join(exp3, "loss_trajectories.csv"))
    conv = pd.read_csv(os.path.join(exp3, "convergence_report.csv"))
    diag = pd.read_csv(os.path.join(exp3, "distribution_diagnostics.csv"))
    base_metrics = pd.concat([pd.read_csv(os.path.join(runs, f"rep{r}_metrics.csv"))
                              for r in range(1, N_REPEATS + 1)], ignore_index=True)

    A.check("A4 exp3_metrics row count", len(exp3_df) == len(ARMS) * N_REPEATS * N_FOLDS,
            f"found {len(exp3_df)}")
    A.check("A5 diagnostics row count", len(diag) == len(ARMS) * N_REPEATS * N_FOLDS,
            f"found {len(diag)}")
    A.check("A6 convergence row count", len(conv) == len(ARMS) * N_REPEATS * N_FOLDS,
            f"found {len(conv)}")

    # ---- C-G. per fold-run re-derivation ---------------------------------
    # training-fold PHQ means, needed for MAE_mean_pred, taken from the manifest
    train_means: Dict[int, float] = {}
    try:
        import pyarrow.parquet as pq
        recs = pq.read_table(a.parquet).to_pandas()
        phq_by_pid = {int(r.participant_id): float(r.phq_score) for r in recs.itertuples()}
        for fold in manifest["folds"]:
            ids = [int(x) for x in fold["train_ids"]]
            train_means[int(fold["fold"])] = float(np.mean([phq_by_pid[i] for i in ids]))
        A.check("C0 parquet readable for train-mean re-derivation", True)
    except Exception as exc:  # parquet optional - MAE_mean_pred check is then skipped
        A.check("C0 parquet readable for train-mean re-derivation", False, str(exc))

    for arm in ARMS:
        for r in range(1, N_REPEATS + 1):
            for f in range(1, N_FOLDS + 1):
                tag = f"{arm} rep{r} fold{f}"
                pth = os.path.join(exp3, f"{arm}_rep{r}_fold{f}_preds.csv")
                if not A.check(f"C1 {tag} preds file present", os.path.isfile(pth)):
                    continue
                d = pd.read_csv(pth)

                spec = manifest["folds"][f - 1]
                ids = [int(x) for x in d["participant_id"]]
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
                A.check(f"C5 {tag} pred_class == argmax (p_pos >= 0.5)",
                        np.array_equal(pred_class, (p_pos >= 0.5).astype(int)))

                row = exp3_df[(exp3_df["arm"] == arm) & (exp3_df["repeat"] == r)
                              & (exp3_df["fold"] == f)]
                if A.check(f"D0 {tag} metric row present", len(row) == 1):
                    row = row.iloc[0]
                    tm = train_means.get(f, float(row.get("MAE_mean_pred", np.nan)))
                    rec = recompute_metrics(y_phq, y_bin, pred_phq, p_pos, pred_class, tm)
                    for m in PRIMARY_METRICS:
                        if m == "MAE_mean_pred" and f not in train_means:
                            continue
                        A.check(f"D1 {tag} {m} re-derived",
                                close(rec[m], row[m], tol),
                                f"audit={rec[m]!r} artifact={row[m]!r}")
                    A.check(f"D2 {tag} epochs == arm budget",
                            int(row["epochs"]) == ARMS[arm])

                # G. diagnostics re-derived
                drow = diag[(diag["arm"] == arm) & (diag["repeat"] == r) & (diag["fold"] == f)]
                if A.check(f"G0 {tag} diagnostics row present", len(drow) == 1):
                    drow = drow.iloc[0]
                    A.check(f"G1 {tag} band width", close(float(p_pos.max() - p_pos.min()),
                                                          drow["p_pos_band_width"], tol))
                    A.check(f"G2 {tag} pos_rate", close(float(np.mean(pred_class)),
                                                        drow["pos_rate"], tol))
                    exp_deg = bool(np.mean(pred_class) in (0.0, 1.0))
                    A.check(f"G3 {tag} degenerate flag", bool(drow["degenerate"]) == exp_deg)

                # E-F. trajectory and convergence re-derived
                t = traj[(traj["arm"] == arm) & (traj["repeat"] == r) & (traj["fold"] == f)]
                A.check(f"E1 {tag} trajectory has {ARMS[arm]} epochs",
                        len(t) == ARMS[arm], f"found {len(t)}")
                if len(t) == ARMS[arm]:
                    losses = t.sort_values("epoch")["avg_loss"].tolist()
                    A.check(f"E2 {tag} epochs are 1..{ARMS[arm]}",
                            t.sort_values("epoch")["epoch"].tolist()
                            == list(range(1, ARMS[arm] + 1)))
                    crow = conv[(conv["arm"] == arm) & (conv["repeat"] == r)
                                & (conv["fold"] == f)]
                    if A.check(f"F0 {tag} convergence row present", len(crow) == 1):
                        crow = crow.iloc[0]
                        A.check(f"F1 {tag} first/final loss", close(losses[0], crow["first_loss"], tol)
                                and close(losses[-1], crow["final_loss"], tol))
                        pe = plateau_epoch(losses)
                        A.check(f"F2 {tag} plateau_reached re-derived",
                                bool(crow["plateau_reached"]) == (pe is not None))
                        A.check(f"F3 {tag} plateau_epoch re-derived",
                                int(crow["plateau_epoch"]) == (pe if pe is not None else -1))

        # coverage per arm/repeat
        for r in range(1, N_REPEATS + 1):
            seen: List[int] = []
            for f in range(1, N_FOLDS + 1):
                p = os.path.join(exp3, f"{arm}_rep{r}_fold{f}_preds.csv")
                if os.path.isfile(p):
                    seen += [int(x) for x in pd.read_csv(p)["participant_id"]]
            A.check(f"C6 {arm} rep{r} covers {N_PARTICIPANTS} unique participants",
                    len(seen) == N_PARTICIPANTS and len(set(seen)) == N_PARTICIPANTS,
                    f"rows={len(seen)} unique={len(set(seen))}")

    # ---- H. E0 equivalence gate, independently re-evaluated ---------------
    with open(os.path.join(base, "baseline_cv_summary.json"), "r", encoding="utf-8") as fh:
        published = json.load(fh)["metrics"]
    e0 = exp3_df[exp3_df["arm"] == CONTROL_ARM]
    if A.check("H0 E0 arm complete", len(e0) == N_REPEATS * N_FOLDS):
        pf = e0.groupby("fold")[PRIMARY_METRICS].mean().sort_index()
        outside = []
        for m in PRIMARY_METRICS:
            if m not in published:
                continue
            got = fold_ci(pf[m].values)
            lo, hi = float(published[m]["ci95"][0]), float(published[m]["ci95"][1])
            if not (lo <= got["mean"] <= hi):
                outside.append(m)
        A.check("H1 E0 aggregate inside Baseline-CV 95% CI on all primary metrics",
                not outside, f"outside: {outside}")
        gp = os.path.join(exp3, "e0_equivalence_check.json")
        if A.check("H2 e0_equivalence_check.json present", os.path.isfile(gp)):
            with open(gp, "r", encoding="utf-8") as fh:
                rep = json.load(fh)
            A.check("H3 recorded gate verdict matches re-derivation",
                    bool(rep.get("passed")) == (not outside))

    # ---- I. aggregation re-derivation -------------------------------------
    sp = os.path.join(exp3, "exp3_summary.json")
    if A.check("I0 exp3_summary.json present", os.path.isfile(sp)):
        with open(sp, "r", encoding="utf-8") as fh:
            S = json.load(fh)
        A.check("I1 primary arm is E1", S.get("primary_arm") == PRIMARY_ARM)
        A.check("I2 t_crit == t(0.975, 4)",
                close(S.get("t_crit", 0), float(stats.t.ppf(0.975, N_FOLDS - 1)), 1e-9))
        A.check("I3 diagnostics flagged as observational",
                "not acceptance criteria" in str(S.get("diagnostics_status", "")).lower())

        for arm in ARMS:
            pf = exp3_df[exp3_df["arm"] == arm].groupby("fold")[PRIMARY_METRICS].mean().sort_index()
            for m in PRIMARY_METRICS:
                got = fold_ci(pf[m].values)
                ref = S.get("aggregates", {}).get(arm, {}).get(m)
                if A.check(f"I4 {arm} {m} aggregate present", ref is not None):
                    for key in ["mean", "fold_sd", "se", "ci95_lo", "ci95_hi"]:
                        A.check(f"I5 {arm} {m}.{key} re-derived",
                                close(got[key], float(ref[key]), tol))

        bpf = base_metrics.groupby("fold")[PAIRED_METRICS].mean().sort_index()
        epf = exp3_df[exp3_df["arm"] == PRIMARY_ARM].groupby("fold")[PAIRED_METRICS].mean().sort_index()
        for m in PAIRED_METRICS:
            deltas = epf[m].values - bpf[m].values
            ref = S.get("paired_vs_baseline", {}).get(m)
            if A.check(f"I6 paired {m} present", ref is not None):
                A.check(f"I7 paired {m} mean_delta re-derived",
                        close(float(np.mean(deltas)), float(ref["mean_delta"]), tol))
                A.check(f"I8 paired {m} per-fold deltas re-derived",
                        all(close(x, y, tol) for x, y in
                            zip(deltas, ref.get("per_fold_delta", []))))

        A.note(f"overall verdict recorded: {S.get('overall_verdict')}")
        for c in S.get("acceptance_criteria", []):
            A.note(f"criterion {c['id']}: {'MET' if c['met'] else 'NOT MET'}")

    # ---- J. containment ----------------------------------------------------
    stray = [os.path.relpath(os.path.join(root, fn), base)
             for root, _d, fs in os.walk(base) for fn in fs if "exp3" in fn.lower()]
    A.check("J1 no Exp 3 artifact inside the baseline dir", not stray, f"stray {stray}")
    # Derived, not a literal: assert the artifacts Experiment 3 actually consumes
    # are intact, rather than a raw recursive count. A count couples the audit to
    # bundling choices - baseline_cv_summary.md is a human-readable rendering of
    # the .json, is read by no script, and is legitimately absent from the Colab
    # bundle. B1 already SHA-gates the three inputs that are consumed.
    required = set(FROZEN_INPUT_SHA)
    top_level = {f for f in os.listdir(base) if os.path.isfile(os.path.join(base, f))}
    A.check("J2a required baseline inputs present", required <= top_level,
            f"missing {sorted(required - top_level)}")
    n_runs = len(os.listdir(os.path.join(base, "runs")))
    A.check("J2b baseline runs/ complete (5 metrics + 25 preds)", n_runs == 30,
            f"found {n_runs}")

    raise SystemExit(0 if A.report() else 1)


if __name__ == "__main__":
    main()
