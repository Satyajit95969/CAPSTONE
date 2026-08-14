#!/usr/bin/env python
"""
Baseline-CV aggregation (Phase 11.6).

Reads all per-repeat metric CSVs produced by run_baseline_cv.py, computes the
official Version 2 comparator: aggregate mean +/- SD +/- CI for every primary
metric, decomposes the noise band into partition vs training-stochasticity
components, and estimates the Minimum Detectable Effect (MDE).

Usage:
    python aggregate_baseline_cv.py --runs-dir trainer_outputs/baseline_cv/runs \
        --out trainer_outputs/baseline_cv/baseline_cv_summary
"""
import argparse, glob, json, os
import numpy as np
import pandas as pd
from scipy import stats

PRIMARY = ["MAE", "RMSE", "pred_var", "MAE_mean_pred",
           "ROC_AUC", "PR_AUC", "balAcc", "MCC", "Precision", "Recall", "F1", "Accuracy"]


def fold_level_ci(per_fold):
    """95% CI from the k per-fold values using Student-t (the fold is the unit of
    independence; folds are participant-disjoint, so k is the effective n — NOT R*k,
    whose rows are correlated). Point estimate = mean of per-fold means."""
    v = np.asarray(per_fold, float)
    k = len(v)
    m = float(np.mean(v))
    sd = float(np.std(v, ddof=1)) if k > 1 else 0.0
    se = sd / np.sqrt(k) if k > 1 else 0.0
    tcrit = float(stats.t.ppf(0.975, k - 1)) if k > 1 else float("nan")
    half = tcrit * se if k > 1 else 0.0
    return m, sd, se, tcrit, m - half, m + half


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs-dir", default="trainer_outputs/baseline_cv/runs")
    ap.add_argument("--out", default="trainer_outputs/baseline_cv/baseline_cv_summary")
    a = ap.parse_args()

    files = sorted(glob.glob(os.path.join(a.runs_dir, "rep*_metrics.csv")))
    if not files:
        raise SystemExit(f"No rep*_metrics.csv in {a.runs_dir}. Run run_baseline_cv.py first.")
    df = pd.concat([pd.read_csv(f) for f in files], ignore_index=True)
    R = sorted(df["repeat"].unique())
    K = sorted(df["fold"].unique())
    print(f"[INFO] loaded {len(df)} rows: {len(R)} repeats x {len(K)} folds")

    if len(R) < 2:
        print(f"[WARN] only R={len(R)} repeat(s): training-stochasticity SD is unmeasurable (needs R>=2). "
              "Run more repeats before using these numbers to gate Phase 12.")

    summary = {"n_repeats": len(R), "n_folds": len(K), "n_rows": int(len(df)),
               "ci_method": "fold-level Student-t (df=k-1)", "metrics": {}}
    md = ["# Baseline-CV Aggregate Summary (Phase 11.6)\n",
          f"- Repeats R = {len(R)}  |  Folds k = {len(K)}  |  total fold-runs = {len(df)}\n",
          "- CIs and MDE use **fold-level Student-t** (the fold is the unit of independence; "
          "the R*k rows are correlated and are NOT treated as iid).\n",
          "\n| Metric | Mean | Fold SD | 95% CI (t) | Partition SD | Training-stoch SD |",
          "|---|---|---|---|---|---|"]

    for metric in PRIMARY:
        if metric not in df.columns:
            continue
        # average over repeats first -> one value per fold (the independent unit)
        per_fold_mean = df.groupby("fold")[metric].mean().values
        m, sd_fold, se, tcrit, lo, hi = fold_level_ci(per_fold_mean)

        partition_sd = sd_fold  # across-fold variability = partition component

        # Training-stochasticity: SD across repeats within a fold, averaged over folds.
        # Undefined when R<2 (single value per fold) -> report None.
        if len(R) > 1:
            within = df.groupby("fold")[metric].std(ddof=1)
            train_sd = float(np.nanmean(within.values))
        else:
            train_sd = None
        train_str = f"{train_sd:.4f}" if train_sd is not None else "n/a (R<2)"

        summary["metrics"][metric] = dict(mean=m, fold_sd=sd_fold, se=se, t_crit=tcrit,
                                          ci95=[lo, hi], partition_sd=partition_sd,
                                          training_stochasticity_sd=train_sd)
        md.append(f"| {metric} | {m:.4f} | {sd_fold:.4f} | [{lo:.4f}, {hi:.4f}] | {partition_sd:.4f} | {train_str} |")

    # ---- Minimum Detectable Effect (paired planning estimate) ----
    # Every future experiment reuses the SAME manifest => Version 2 is compared to
    # Baseline-CV per fold (paired). This is a PLANNING estimate of the noise scale
    # from Baseline-CV's own fold spread; the operative test at experiment time uses
    # the SD of the paired per-fold differences. Student-t, df = k-1.
    mde = {}
    for metric in ["MAE", "RMSE", "ROC_AUC", "PR_AUC", "balAcc", "MCC", "F1"]:
        if metric not in df.columns:
            continue
        per_fold_mean = df.groupby("fold")[metric].mean().values
        _, sd_fold, se, tcrit, _, _ = fold_level_ci(per_fold_mean)
        mde[metric] = dict(paired_se=se, t_crit=tcrit, mde_95=(tcrit * se if len(K) > 1 else float("nan")))
    summary["MDE"] = mde
    md.append("\n## Minimum Detectable Effect (paired planning estimate, 95%, Student-t)\n")
    md.append("| Metric | Paired SE | t(0.975, k-1) | MDE (95%) |")
    md.append("|---|---|---|---|")
    for k, v in mde.items():
        md.append(f"| {k} | {v['paired_se']:.4f} | {v['t_crit']:.3f} | {v['mde_95']:.4f} |")

    md.append("\n## Interpretation notes\n")
    md.append("- **Fold SD** = SD of the per-fold metric (repeats averaged first); **Partition SD** = same "
              "across-fold quantity; **Training-stoch SD** = run-to-run component across repeats.\n")
    md.append("- **CIs/MDE use fold-level Student-t** (df = k-1), because the fold — not the fold-run — is the "
              "unit of independence. Pooling R*k correlated rows with a normal quantile would understate "
              "uncertainty and make the noise band too tight.\n")
    md.append("- **MDE (95%)** is a *planning* estimate of the noise scale from Baseline-CV's fold spread. The "
              "operative accept/reject test at experiment time uses the SD of the **paired per-fold differences** "
              "(V2_f - BaselineCV_f). A Version 2 change smaller than the MDE is reported as *no detectable effect*.\n")
    md.append("- Accuracy is reported but is NOT a decision metric (protocol §3.3).\n")

    with open(a.out + ".json", "w") as f:
        json.dump(summary, f, indent=1)
    with open(a.out + ".md", "w") as f:
        f.write("\n".join(md) + "\n")
    print("\n".join(md))
    print(f"\n[DONE] wrote {a.out}.json and {a.out}.md")


if __name__ == "__main__":
    main()
