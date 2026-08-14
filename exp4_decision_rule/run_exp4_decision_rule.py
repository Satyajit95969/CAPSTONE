#!/usr/bin/env python
"""
Phase 12 / Experiment 4 - Decision Rule.

Recovers classification decisions from the FROZEN Baseline-CV predictions by
replacing the trainer's implicit argmax decision rule (tau = 0.5) with a
pre-declared, leakage-free operating point.

THE SINGLE CHANGED FACTOR
-------------------------
The decision function mapping predicted probability -> predicted class:

    Baseline-CV : pred_class = argmax(softmax(logits))        == (p_pos >= 0.5)
    Experiment 4: pred_class = (p_pos >= tau_f)               tau_f by policy

Nothing else changes. This script NEVER instantiates a model, NEVER trains, and
deliberately does NOT import torch / transformers / trainer_mentalbert_daic.
It is arithmetic over the 25 frozen prediction CSVs produced by Phase 11.6.

That isolation is what makes the one-factor guarantee a PROOF rather than a
convention: the threshold-free metrics (ROC-AUC, PR-AUC) and every regression
metric (MAE, RMSE, pred_var) are mathematically invariant to tau, so verifying
they are bit-unchanged against Baseline-CV proves nothing but the decision rule
moved. Those invariants are asserted before any result is written.

PRE-DECLARED THRESHOLD POLICIES (design-frozen; no post-hoc additions)
---------------------------------------------------------------------
  P1  PRIMARY     Leave-one-fold-out Youden-J. For repeat r, fold f: pool the
                  held-out predictions of the OTHER FOUR folds of the SAME
                  repeat, choose tau maximising (TPR - FPR) on that pool, then
                  apply it to fold f. No participant's own prediction or label
                  can influence their own threshold -> zero leakage by
                  construction. Repeats stay independent, so the
                  training-stochasticity component remains measurable.
  P2  sensitivity Prevalence-matched, also leave-one-fold-out: tau is the
                  (1 - prevalence) quantile of the pooled p_pos.
  P3  sensitivity Fixed constant tau = 45/188 = 0.23936... declared in advance.
                  A zero-fit control; arbitrary by design, and expected to
                  produce a high positive rate. It exists to bound the family,
                  not to be recommended.
  P4  descriptive Full threshold sweep over tau in [0.14, 0.45]. A CURVE, never
                  a selection mechanism. Reported for sensitivity only.

Baseline-CV is opened READ-ONLY and its SHA-256 digests are verified before and
after the run. All artifacts are written under trainer_outputs/exp4_decision_rule/.

Usage (from the repository root):
    python exp4_decision_rule/run_exp4_decision_rule.py
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
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
    roc_curve,
)

# --------------------------------------------------------------------------
# Frozen constants - Phase 11.6 record. Any mismatch aborts the run.
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
CORPUS_PREVALENCE = N_POSITIVE / N_PARTICIPANTS      # 0.2393617021276596
BASELINE_TAU = 0.5                                    # the argmax decision rule

# P4 descriptive sweep grid. Spans the observed p_pos range measured in Phase
# 11.7 ([0.1404, 0.4422]) with margin on both sides.
SWEEP_LO, SWEEP_HI, SWEEP_STEP = 0.14, 0.45, 0.005

# Metrics that are mathematically invariant to the decision threshold. These
# MUST reproduce the Baseline-CV values; any deviation means the wrong
# predictions were loaded or training contamination occurred.
INVARIANT_METRICS = ["ROC_AUC", "PR_AUC", "MAE", "RMSE", "pred_var"]


# --------------------------------------------------------------------------
# Integrity helpers
# --------------------------------------------------------------------------
def sha256(path: str) -> str:
    """SHA-256 of a file, streamed so large files do not load into memory."""
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for block in iter(lambda: fh.read(8192), b""):
            h.update(block)
    return h.hexdigest()


def abort(message: str) -> "NoReturn":  # noqa: F821
    """Stop the run. Exp 4 has no partial-success mode: a failed gate means the
    result is not trustworthy, so we exit rather than continue and warn."""
    print(f"\n[ABORT] {message}", file=sys.stderr)
    raise SystemExit(1)


def snapshot_input_shas(baseline_dir: str, runs_dir: str) -> Dict[str, str]:
    """Digest every Baseline-CV file this run will read.

    Re-taken after execution and compared, proving the frozen inputs were
    treated as read-only.
    """
    shas: Dict[str, str] = {}
    for name in FROZEN_INPUT_SHA:
        shas[name] = sha256(os.path.join(baseline_dir, name))
    for r in range(1, N_REPEATS + 1):
        for f in range(1, N_FOLDS + 1):
            key = f"runs/rep{r}_fold{f}_preds.csv"
            shas[key] = sha256(os.path.join(runs_dir, f"rep{r}_fold{f}_preds.csv"))
        key = f"runs/rep{r}_metrics.csv"
        shas[key] = sha256(os.path.join(runs_dir, f"rep{r}_metrics.csv"))
    return shas


# --------------------------------------------------------------------------
# Loading and pre-flight validation
# --------------------------------------------------------------------------
def load_preds(runs_dir: str) -> Dict[Tuple[int, int], pd.DataFrame]:
    """Read the 25 frozen held-out prediction files, keyed by (repeat, fold).

    Files are opened read-only; nothing is written back.
    """
    preds: Dict[Tuple[int, int], pd.DataFrame] = {}
    for r in range(1, N_REPEATS + 1):
        for f in range(1, N_FOLDS + 1):
            path = os.path.join(runs_dir, f"rep{r}_fold{f}_preds.csv")
            if not os.path.isfile(path):
                abort(f"missing frozen prediction file: {path}")
            preds[(r, f)] = pd.read_csv(path)
    return preds


def load_baseline_metrics(runs_dir: str) -> pd.DataFrame:
    """Concatenate the 5 frozen per-repeat metric tables (the comparator)."""
    frames = []
    for r in range(1, N_REPEATS + 1):
        path = os.path.join(runs_dir, f"rep{r}_metrics.csv")
        if not os.path.isfile(path):
            abort(f"missing frozen metrics file: {path}")
        frames.append(pd.read_csv(path))
    return pd.concat(frames, ignore_index=True)


def verify_inputs(preds: Dict[Tuple[int, int], pd.DataFrame],
                  manifest: dict,
                  observed_shas: Dict[str, str]) -> None:
    """Stage A pre-flight gates. Every one must pass before computation starts.

    Checks: frozen SHA digests; manifest shape; per-fold participant sets equal
    the manifest's test_ids; train/test disjointness; total row count; and that
    each repeat covers all 188 participants exactly once (no duplication, no
    omission).
    """
    for name, expected in FROZEN_INPUT_SHA.items():
        got = observed_shas[name]
        if got != expected:
            abort(f"{name} SHA {got} != frozen {expected}. "
                  "Baseline-CV has been altered; refusing to run.")

    if manifest.get("n_participants") != N_PARTICIPANTS:
        abort(f"manifest n_participants={manifest.get('n_participants')} != {N_PARTICIPANTS}")
    if len(manifest.get("folds", [])) != N_FOLDS:
        abort(f"manifest has {len(manifest.get('folds', []))} folds, expected {N_FOLDS}")

    required_cols = {"participant_id", "phq", "phq_bin", "pred_phq", "p_pos", "pred_class"}
    total_rows = 0
    for r in range(1, N_REPEATS + 1):
        seen: List[int] = []
        for f in range(1, N_FOLDS + 1):
            df = preds[(r, f)]
            missing = required_cols - set(df.columns)
            if missing:
                abort(f"rep{r}_fold{f}_preds.csv missing columns {sorted(missing)}")

            fold_spec = manifest["folds"][f - 1]
            test_ids = [int(x) for x in fold_spec["test_ids"]]
            train_ids = set(int(x) for x in fold_spec["train_ids"])
            got_ids = [int(x) for x in df["participant_id"].tolist()]

            if got_ids != test_ids:
                abort(f"rep{r} fold{f}: participant_id order/content differs from "
                      "the frozen manifest test_ids")
            overlap = set(got_ids) & train_ids
            if overlap:
                abort(f"rep{r} fold{f}: {len(overlap)} participants appear in BOTH "
                      "train and test - fold integrity violated")

            seen.extend(got_ids)
            total_rows += len(df)

        if len(seen) != N_PARTICIPANTS or len(set(seen)) != N_PARTICIPANTS:
            abort(f"repeat {r} covers {len(seen)} rows / {len(set(seen))} unique "
                  f"participants, expected {N_PARTICIPANTS} / {N_PARTICIPANTS}")

    expected_rows = N_PARTICIPANTS * N_REPEATS
    if total_rows != expected_rows:
        abort(f"total prediction rows {total_rows} != {expected_rows}")

    print(f"[PASS] pre-flight: {N_REPEATS}x{N_FOLDS} folds, {total_rows} predictions, "
          f"frozen SHAs verified, zero train/test overlap")


# --------------------------------------------------------------------------
# Threshold selection - the scientific core
# --------------------------------------------------------------------------
def lofo_pool(preds: Dict[Tuple[int, int], pd.DataFrame],
              repeat: int, fold: int) -> Tuple[pd.DataFrame, List[int]]:
    """Leave-one-fold-out selection pool.

    Returns the concatenated held-out predictions of every fold of `repeat`
    EXCEPT `fold`, together with the fold indices used.

    This is the leakage gate. The evaluated fold's rows are asserted absent
    from the pool, so no participant's own prediction or label can influence
    the threshold applied to them. Pooling within a single repeat (rather than
    across repeats) keeps the repeats statistically independent, preserving the
    training-stochasticity component of the noise band.
    """
    other = [f for f in range(1, N_FOLDS + 1) if f != fold]
    pool = pd.concat([preds[(repeat, f)] for f in other], ignore_index=True)

    # Hard leakage assertion - never soften this into a warning.
    evaluated_ids = set(int(x) for x in preds[(repeat, fold)]["participant_id"])
    pool_ids = set(int(x) for x in pool["participant_id"])
    leaked = evaluated_ids & pool_ids
    if leaked:
        abort(f"LEAKAGE: rep{repeat} fold{fold} - {len(leaked)} evaluated "
              f"participants present in the selection pool")

    if pool["phq_bin"].nunique() < 2:
        abort(f"rep{repeat} fold{fold}: selection pool is single-class; "
              "Youden-J is undefined")

    return pool, other


def select_tau_youden(pool: pd.DataFrame) -> float:
    """P1 (PRIMARY): threshold maximising Youden's J = TPR - FPR on the pool.

    sklearn's roc_curve prepends a sentinel threshold above the maximum score
    (np.inf in modern versions) representing "predict nothing positive". That
    point has J = 0 and is excluded, because selecting it would reproduce the
    degenerate all-negative baseline rather than choose an operating point.
    """
    y = pool["phq_bin"].to_numpy(dtype=int)
    p = pool["p_pos"].to_numpy(dtype=float)

    fpr, tpr, thresholds = roc_curve(y, p)
    finite = np.isfinite(thresholds) & (thresholds <= p.max())
    if not finite.any():
        abort("no finite ROC threshold within the observed probability range")

    j = (tpr - fpr)[finite]
    tau = float(thresholds[finite][int(np.argmax(j))])

    if not (p.min() <= tau <= p.max()):
        abort(f"P1 selected tau={tau:.6f} outside the pool range "
              f"[{p.min():.6f}, {p.max():.6f}]")
    return tau


def select_tau_prevalence(pool: pd.DataFrame,
                          prevalence: float = CORPUS_PREVALENCE) -> float:
    """P2 (sensitivity): threshold placing `prevalence` of the pool above it.

    Uses only the pool's probability distribution - the evaluated fold's labels
    and predictions are never consulted.
    """
    p = pool["p_pos"].to_numpy(dtype=float)
    tau = float(np.quantile(p, 1.0 - prevalence))
    if not (p.min() <= tau <= p.max()):
        abort(f"P2 selected tau={tau:.6f} outside the pool range")
    return tau


def apply_threshold(p_pos: np.ndarray, tau: float) -> np.ndarray:
    """Decision rule. `>=` matches sklearn's roc_curve threshold semantics, so
    a tau chosen by select_tau_youden reproduces the operating point it was
    selected at."""
    return (np.asarray(p_pos, dtype=float) >= tau).astype(int)


# --------------------------------------------------------------------------
# Metrics
# --------------------------------------------------------------------------
def threshold_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
    """Threshold-conditional metrics - the experimental arm.

    MCC is defined as 0.0 on a single-class truth vector, matching the frozen
    driver's convention (run_baseline_cv.py) so the arms remain comparable.
    """
    two_class = len(np.unique(y_true)) > 1
    return {
        "Precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "Recall": float(recall_score(y_true, y_pred, zero_division=0)),
        "F1": float(f1_score(y_true, y_pred, zero_division=0)),
        "balAcc": float(balanced_accuracy_score(y_true, y_pred)),
        "MCC": float(matthews_corrcoef(y_true, y_pred)) if two_class else 0.0,
        "Accuracy": float(accuracy_score(y_true, y_pred)),
        "pos_rate": float(np.mean(y_pred)),
    }


def invariant_metrics(y_true: np.ndarray, p_pos: np.ndarray,
                      pred_phq: np.ndarray, phq: np.ndarray) -> Dict[str, float]:
    """Threshold-free and regression metrics - must equal Baseline-CV.

    The formulas replicate run_baseline_cv.py exactly:
      - RMSE     : sqrt(mean((y - p)^2))                        (its rmse(), L73-74)
      - pred_var : float(np.var(pred_phq))  -> numpy default ddof=0   (L178)
      - MAE      : sklearn mean_absolute_error                        (L177)
      - ROC/PR   : roc_auc_score / average_precision_score on p_pos   (L181-182)

    The frozen driver is deliberately NOT imported to reuse these helpers,
    because importing it would pull in trainer_mentalbert_daic (and therefore
    torch/transformers), breaking this script's isolation from the model. The
    numerical match against the stored Baseline-CV values is the proof of
    equivalence instead.
    """
    two_class = len(np.unique(y_true)) > 1
    y = np.asarray(phq, dtype=float)
    p = np.asarray(pred_phq, dtype=float)
    return {
        "ROC_AUC": float(roc_auc_score(y_true, p_pos)) if two_class else float("nan"),
        "PR_AUC": float(average_precision_score(y_true, p_pos)) if two_class else float("nan"),
        "MAE": float(mean_absolute_error(y, p)),
        "RMSE": float(np.sqrt(np.mean((y - p) ** 2))),
        "pred_var": float(np.var(p)),
    }


def random_ranker_control(prevalence: float, pos_rate: float) -> Dict[str, float]:
    """Expected performance of a RANDOM ranker predicting `pos_rate` positive
    on a set with the given prevalence:

        precision = prevalence      (a random selection inherits the base rate)
        recall    = pos_rate
        F1        = 2*prevalence*pos_rate / (prevalence + pos_rate)

    This is the Criterion-3 control: it distinguishes genuine discrimination
    from merely predicting more positives. A thresholded model that only
    recovers QUANTITY will match these numbers; one that discriminates will
    exceed them on precision.
    """
    denom = prevalence + pos_rate
    f1 = (2.0 * prevalence * pos_rate / denom) if denom > 0 else 0.0
    return {
        "rand_Precision": float(prevalence),
        "rand_Recall": float(pos_rate),
        "rand_F1": float(f1),
    }


def threshold_sweep(preds: Dict[Tuple[int, int], pd.DataFrame],
                    grid: np.ndarray) -> pd.DataFrame:
    """P4: descriptive sweep across the grid for every fold-run.

    This characterises sensitivity to tau. It is NOT a selection mechanism -
    no headline number may be drawn from it, because choosing a point on this
    curve after seeing it would be exactly the post-hoc optimisation the
    pre-registration forbids.
    """
    rows = []
    for (r, f), df in sorted(preds.items()):
        y = df["phq_bin"].to_numpy(dtype=int)
        p = df["p_pos"].to_numpy(dtype=float)
        for tau in grid:
            m = threshold_metrics(y, apply_threshold(p, float(tau)))
            rows.append({"repeat": r, "fold": f, "tau": float(tau), **m})
    return pd.DataFrame(rows)


# --------------------------------------------------------------------------
# Main
# --------------------------------------------------------------------------
def main() -> None:
    ap = argparse.ArgumentParser(
        description="Phase 12 / Experiment 4 - decision-rule re-scoring "
                    "of the frozen Baseline-CV predictions.")
    ap.add_argument("--baseline-dir", default="trainer_outputs/baseline_cv",
                    help="READ-ONLY frozen Baseline-CV directory")
    ap.add_argument("--out-dir", default="trainer_outputs/exp4_decision_rule",
                    help="artifact directory (created; never inside baseline-dir)")
    ap.add_argument("--invariant-tol", type=float, default=1e-9,
                    help="absolute tolerance for the Baseline-CV invariant check "
                         "(CSV round-trip of float64 is not guaranteed bit-exact)")
    args = ap.parse_args()

    baseline_dir = os.path.normpath(args.baseline_dir)
    runs_dir = os.path.join(baseline_dir, "runs")
    out_dir = os.path.normpath(args.out_dir)

    # Structural guard: the output directory must never sit inside the frozen
    # Baseline-CV tree, so a path mistake cannot corrupt the comparator.
    if os.path.abspath(out_dir).startswith(os.path.abspath(baseline_dir) + os.sep) \
            or os.path.abspath(out_dir) == os.path.abspath(baseline_dir):
        abort(f"out-dir {out_dir} is inside the frozen baseline dir {baseline_dir}")
    os.makedirs(out_dir, exist_ok=True)

    print("=" * 74)
    print("PHASE 12 / EXPERIMENT 4 - DECISION RULE")
    print("  single changed factor : decision function (tau), model untouched")
    print(f"  baseline (read-only)  : {baseline_dir}")
    print(f"  artifacts             : {out_dir}")
    print(f"  corpus prevalence     : {CORPUS_PREVALENCE:.10f}  ({N_POSITIVE}/{N_PARTICIPANTS})")
    print("=" * 74)

    # ---- Stage A: load + pre-flight -------------------------------------
    manifest_path = os.path.join(baseline_dir, "fold_manifest.json")
    if not os.path.isfile(manifest_path):
        abort(f"missing frozen manifest: {manifest_path}")
    with open(manifest_path, "r", encoding="utf-8") as fh:
        manifest = json.load(fh)

    preds = load_preds(runs_dir)
    baseline_metrics = load_baseline_metrics(runs_dir)
    shas_before = snapshot_input_shas(baseline_dir, runs_dir)
    verify_inputs(preds, manifest, shas_before)

    # ---- Stage B: threshold selection + re-scoring ----------------------
    threshold_rows: List[dict] = []
    metric_rows: List[dict] = []
    invariant_deltas: Dict[str, List[float]] = {m: [] for m in INVARIANT_METRICS}

    for r in range(1, N_REPEATS + 1):
        for f in range(1, N_FOLDS + 1):
            df = preds[(r, f)]
            y = df["phq_bin"].to_numpy(dtype=int)
            p = df["p_pos"].to_numpy(dtype=float)
            pred_phq = df["pred_phq"].to_numpy(dtype=float)
            phq = df["phq"].to_numpy(dtype=float)
            fold_prevalence = float(np.mean(y))

            pool, pool_folds = lofo_pool(preds, r, f)
            taus = {
                "P1": select_tau_youden(pool),
                "P2": select_tau_prevalence(pool),
                "P3": CORPUS_PREVALENCE,
            }
            for policy, tau in taus.items():
                threshold_rows.append({
                    "repeat": r, "fold": f, "policy": policy, "tau": tau,
                    "pool_size": int(len(pool)),
                    "pool_n_pos": int(pool["phq_bin"].sum()),
                    "pool_folds": "|".join(str(x) for x in pool_folds),
                })

            # Invariants: recomputed from the frozen predictions, then checked
            # against the stored Baseline-CV row. Identical by construction if
            # and only if the same predictions were loaded and nothing trained.
            inv = invariant_metrics(y, p, pred_phq, phq)
            ref = baseline_metrics[(baseline_metrics["repeat"] == r)
                                   & (baseline_metrics["fold"] == f)]
            if len(ref) != 1:
                abort(f"expected exactly one Baseline-CV row for rep{r} fold{f}, "
                      f"found {len(ref)}")
            ref = ref.iloc[0]
            for m in INVARIANT_METRICS:
                delta = abs(inv[m] - float(ref[m]))
                invariant_deltas[m].append(delta)
                if delta > args.invariant_tol:
                    abort(f"INVARIANT VIOLATION rep{r} fold{f}: {m} "
                          f"exp4={inv[m]!r} baseline={float(ref[m])!r} "
                          f"|diff|={delta:.3e} > tol {args.invariant_tol:.1e}")

            # Baseline arm self-test: reproduce the argmax rule from p_pos and
            # confirm it matches the stored pred_class. If this fails, the
            # pipeline cannot be trusted to evaluate any other rule.
            baseline_pred = apply_threshold(p, BASELINE_TAU)
            stored_pred = df["pred_class"].to_numpy(dtype=int)
            if not np.array_equal(baseline_pred, stored_pred):
                abort(f"rep{r} fold{f}: reconstructed argmax decision differs from "
                      "the stored pred_class - cannot reproduce Baseline-CV")

            rescored = pd.DataFrame({
                "participant_id": df["participant_id"].to_numpy(),
                "phq": phq,
                "phq_bin": y,
                "p_pos": p,
                "pred_phq": pred_phq,
                "pred_class_baseline": baseline_pred,
            })

            for arm, tau in [("baseline", BASELINE_TAU)] + list(taus.items()):
                yhat = baseline_pred if arm == "baseline" else apply_threshold(p, tau)
                if arm != "baseline":
                    rescored[f"pred_class_{arm}"] = yhat
                tm = threshold_metrics(y, yhat)
                metric_rows.append({
                    "repeat": r, "fold": f, "policy": arm, "tau": float(tau),
                    "n_test": int(len(df)), "n_pos": int(y.sum()),
                    "prevalence": fold_prevalence,
                    **tm,
                    **random_ranker_control(fold_prevalence, tm["pos_rate"]),
                    **inv,
                })

            rescored.to_csv(os.path.join(out_dir, f"rep{r}_fold{f}_rescored.csv"),
                            index=False)

        print(f"[repeat {r}] 5 folds re-scored | "
              f"P1 tau range "
              f"{min(t['tau'] for t in threshold_rows if t['repeat'] == r and t['policy'] == 'P1'):.4f}"
              f"-"
              f"{max(t['tau'] for t in threshold_rows if t['repeat'] == r and t['policy'] == 'P1'):.4f}")

    thresholds = pd.DataFrame(threshold_rows)
    metrics = pd.DataFrame(metric_rows)
    thresholds.to_csv(os.path.join(out_dir, "thresholds.csv"), index=False)
    metrics.to_csv(os.path.join(out_dir, "exp4_metrics.csv"), index=False)

    # ---- P4 descriptive sweep -------------------------------------------
    grid = np.round(np.arange(SWEEP_LO, SWEEP_HI + SWEEP_STEP / 2, SWEEP_STEP), 4)
    threshold_sweep(preds, grid).to_csv(
        os.path.join(out_dir, "threshold_sweep.csv"), index=False)

    # ---- Stage C: post-execution integrity ------------------------------
    shas_after = snapshot_input_shas(baseline_dir, runs_dir)
    changed = [k for k in shas_before if shas_before[k] != shas_after[k]]
    if changed:
        abort(f"frozen Baseline-CV inputs were modified during the run: {changed}")

    invariants_report = {
        "tolerance": args.invariant_tol,
        "n_fold_runs": int(N_REPEATS * N_FOLDS),
        "max_abs_diff": {m: float(max(v)) for m, v in invariant_deltas.items()},
        "all_within_tolerance": bool(
            all(max(v) <= args.invariant_tol for v in invariant_deltas.values())),
        "baseline_arm_reproduced": True,
        "input_sha_before": shas_before,
        "input_sha_after": shas_after,
        "inputs_unchanged": True,
    }
    with open(os.path.join(out_dir, "invariants_check.json"), "w", encoding="utf-8") as fh:
        json.dump(invariants_report, fh, indent=1)

    # ---- Console summary -------------------------------------------------
    print("-" * 74)
    print("INVARIANTS vs Baseline-CV (max |diff| over 25 fold-runs)")
    for m in INVARIANT_METRICS:
        print(f"  {m:9s} {max(invariant_deltas[m]):.3e}")
    print(f"  all within tolerance {args.invariant_tol:.1e} : "
          f"{invariants_report['all_within_tolerance']}")
    print("  baseline arm reproduces stored pred_class : True")
    print("  frozen inputs unchanged                   : True")

    print("-" * 74)
    print("PER-POLICY MEANS over 25 fold-runs (descriptive; "
          "paired inference is aggregate_exp4.py's job)")
    view = (metrics.groupby("policy")[
                ["tau", "pos_rate", "Precision", "Recall", "F1",
                 "balAcc", "MCC", "Accuracy", "rand_Precision", "rand_F1"]]
            .mean().round(4))
    print(view.to_string())

    print("-" * 74)
    print("artifacts written:")
    for name in sorted(os.listdir(out_dir)):
        print(f"  {name}")
    print(f"\n[DONE] Experiment 4 re-scoring complete -> {out_dir}")
    print("Next: aggregate_exp4.py (fold-level CIs, paired delta vs Baseline-CV, "
          "Criterion-3 test).")


if __name__ == "__main__":
    main()
