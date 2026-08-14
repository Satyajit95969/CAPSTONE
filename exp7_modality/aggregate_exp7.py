#!/usr/bin/env python3
"""
aggregate_exp7.py - Phase 16 / Experiment 7 aggregation and acceptance.

Owns the AUTHORITATIVE A0 equivalence gate (HARD ABORT), the fold-level
aggregates, the paired deltas against Baseline-CV, and the final verdict.

THE A0 EQUIVALENCE GATE - AUTHORITATIVE HARD ABORT
--------------------------------------------------
A0 is text-only on the MULTIMODAL artifact. It constructs
MultiModalModel(audio_dim=None, vision_dim=None), so no encoder is built, the
fusion input is 768, and the torch RNG stream is consumed exactly as Baseline-CV
consumed it. A0 therefore isolates ONE question: did swapping the artifact change
the text-only result?

If A0's fold-level mean falls outside Baseline-CV's published 95% CI on any
primary metric, the artifact swap was NOT inert, Experiment 7 is confounded, and
NO A1/A2/A3 number may be reported. This module aborts in that case.

Mirrors aggregate_exp6.py's l0_equivalence_gate exactly: same per-fold means,
same fold-level Student-t CI (df = k-1), same 12 primary metrics, same
inside-CI test.

CI CONVENTION (unchanged from Baseline-CV / Exps 3-6)
-----------------------------------------------------
Repeats are averaged WITHIN fold, then the CI is taken across the k=5 fold means
with Student-t, df = k-1 = 4, t_crit = 2.7764451051977987.

Usage:
    python aggregate_exp7.py
"""

from __future__ import annotations

import argparse
import ast
import datetime
import hashlib
import json
import os
import platform
import sys
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

N_REPEATS = 5
N_FOLDS = 5
CONTROL_ARM = "A0"
PRIMARY_ARM = "A3"
ARMS = ["A0", "A1", "A2", "A3"]
ARM_LABEL = {"A0": "text only", "A1": "text + audio",
             "A2": "text + vision", "A3": "text + audio + vision"}
ARM_MODALITY = {"A0": (0, 0), "A1": (154, 0), "A2": (0, 84), "A3": (154, 84)}
FUSION_IN = {"A0": 768, "A1": 896, "A2": 896, "A3": 1024}

PRIMARY_METRICS = ["MAE", "RMSE", "pred_var", "MAE_mean_pred", "ROC_AUC", "PR_AUC",
                   "balAcc", "MCC", "Precision", "Recall", "F1", "Accuracy"]
T_CRIT = 2.7764451051977987          # Student-t, df = 4
AGG_TOL = 1e-9

# MDE anchors carried forward from Baseline-CV (Exp 6 used the same).
MDE = {"MAE": 0.4066, "PR_AUC": 0.0950}


def abort(msg: str) -> "NoReturn":  # noqa: F821
    print(f"\n[ABORT] {msg}", file=sys.stderr)
    raise SystemExit(1)


def sha256(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for b in iter(lambda: fh.read(8192), b""):
            h.update(b)
    return h.hexdigest()


def per_fold_means(df: pd.DataFrame, metrics: Sequence[str]) -> pd.DataFrame:
    """Average repeats within fold - the established convention."""
    cols = [m for m in metrics if m in df.columns]
    return df.groupby("fold")[cols].mean().sort_index()


def fold_level_ci(values: np.ndarray) -> Dict[str, float]:
    v = np.asarray(values, float)
    v = v[np.isfinite(v)]
    k = len(v)
    if k < 2:
        return {"mean": float(v.mean()) if k else float("nan"), "fold_sd": float("nan"),
                "se": float("nan"), "ci95_lo": float("nan"), "ci95_hi": float("nan"), "k": k}
    mean = float(v.mean())
    sd = float(v.std(ddof=1))
    se = sd / np.sqrt(k)
    half = T_CRIT * se
    return {"mean": mean, "fold_sd": sd, "se": float(se),
            "ci95_lo": mean - half, "ci95_hi": mean + half, "k": k}


def a0_equivalence_gate(df: pd.DataFrame, baseline_summary: str,
                        out_dir: str) -> Dict:
    """AUTHORITATIVE HARD GATE. Identical construction to Exp 6's L0 gate."""
    with open(baseline_summary, "r", encoding="utf-8") as fh:
        published = json.load(fh)["metrics"]

    a0 = df[df["arm"] == CONTROL_ARM]
    if len(a0) != N_REPEATS * N_FOLDS:
        abort(f"A0 arm has {len(a0)} fold-runs, expected {N_REPEATS * N_FOLDS}. "
              "The equivalence gate cannot be evaluated, so no arm may be reported.")

    pf = per_fold_means(a0, PRIMARY_METRICS)
    rows, failures = [], []
    for m in PRIMARY_METRICS:
        if m not in pf.columns or m not in published:
            continue
        got = fold_level_ci(pf[m].values)
        lo, hi = float(published[m]["ci95"][0]), float(published[m]["ci95"][1])
        inside = bool(lo <= got["mean"] <= hi)
        rows.append({"metric": m, "a0_mean": got["mean"], "a0_fold_sd": got["fold_sd"],
                     "baseline_mean": float(published[m]["mean"]),
                     "baseline_ci_lo": lo, "baseline_ci_hi": hi, "inside_ci": inside})
        if not inside:
            failures.append(m)

    report = {"gate": "A0 aggregate inside Baseline-CV 95% CI",
              "authoritative": True, "n_metrics": len(rows),
              "control_modality": "text only (audio_dim=None, vision_dim=None)",
              "failures": failures, "passed": not failures, "detail": rows}
    os.makedirs(out_dir, exist_ok=True)
    with open(os.path.join(out_dir, "a0_equivalence_check.json"), "w",
              encoding="utf-8") as fh:
        json.dump(report, fh, indent=1)

    if failures:
        for r in rows:
            if not r["inside_ci"]:
                print(f"   A0 {r['metric']}: {r['a0_mean']:.6f} outside "
                      f"[{r['baseline_ci_lo']:.6f}, {r['baseline_ci_hi']:.6f}]",
                      file=sys.stderr)
        abort(f"A0 EQUIVALENCE GATE FAILED on {failures}. Swapping the artifact "
              "changed the text-only result, so Experiment 7 is confounded and no "
              "A1/A2/A3 result may be reported.")
    print(f"[PASS] A0 equivalence gate (AUTHORITATIVE): all {len(rows)} primary "
          "metrics inside Baseline-CV's 95% CI")
    return report


def paired_delta(df: pd.DataFrame, arm: str, baseline_summary: str) -> Dict:
    """Per-fold paired delta of `arm` against Baseline-CV's published fold means.

    Baseline-CV publishes aggregate CIs, not per-fold values, so the paired test
    is anchored on A0's own per-fold means - which the gate has just certified
    equivalent to Baseline-CV. This keeps the comparison paired by fold.
    """
    a0 = per_fold_means(df[df["arm"] == CONTROL_ARM], PRIMARY_METRICS)
    ax = per_fold_means(df[df["arm"] == arm], PRIMARY_METRICS)
    out: Dict[str, Dict] = {}
    for m in PRIMARY_METRICS:
        if m not in a0.columns or m not in ax.columns:
            continue
        d = ax[m].values - a0[m].values
        ci = fold_level_ci(d)
        out[m] = {"a0_mean": float(np.mean(a0[m].values)),
                  "arm_mean": float(np.mean(ax[m].values)),
                  "per_fold_delta": [float(x) for x in d],
                  "mean_delta": ci["mean"], "se": ci["se"],
                  "ci95_lo": ci["ci95_lo"], "ci95_hi": ci["ci95_hi"],
                  "significant": bool(np.isfinite(ci["ci95_lo"]) and
                                      (ci["ci95_lo"] > 0 or ci["ci95_hi"] < 0)),
                  "exceeds_mde": (bool(abs(ci["mean"]) > MDE[m]) if m in MDE else None)}
    return out


# --------------------------------------------------------------------------
# SERIALISATION-ONLY ADDITIONS (exp7_receipt.json / exp7_delta.json)
#
# Everything below writes extra views of results that are ALREADY computed. No
# metric, gate, seed, fold, arm or hyperparameter is touched, and nothing here
# feeds back into exp7_summary.json.
#
# paired_delta() above is deliberately NOT refactored to share code with
# pairwise_delta() below: its output shape is embedded in exp7_summary.json, and
# a refactor - however clean - would risk changing that. Duplicated arithmetic is
# the correct trade here.
# --------------------------------------------------------------------------
DELTA_PAIRS: List[Tuple[str, str]] = [
    ("A1", "A0"), ("A2", "A0"), ("A3", "A0"), ("A3", "A1"), ("A3", "A2"),
]


def pairwise_delta(df: pd.DataFrame, lhs: str, rhs: str) -> Dict[str, Dict]:
    """Per-fold paired delta (lhs - rhs) for every primary metric.

    Paired BY FOLD, using the same per-fold-mean convention and the same
    fold-level Student-t CI as every other comparison in this phase series.
    """
    a = per_fold_means(df[df["arm"] == lhs], PRIMARY_METRICS)
    b = per_fold_means(df[df["arm"] == rhs], PRIMARY_METRICS)
    out: Dict[str, Dict] = {}
    for m in PRIMARY_METRICS:
        if m not in a.columns or m not in b.columns:
            continue
        d = a[m].values - b[m].values
        ci = fold_level_ci(d)
        out[m] = {
            "lhs_mean": float(np.mean(a[m].values)),
            "rhs_mean": float(np.mean(b[m].values)),
            "per_fold_delta": [float(x) for x in d],
            "mean_delta": ci["mean"], "fold_sd": ci["fold_sd"], "se": ci["se"],
            "ci95_lo": ci["ci95_lo"], "ci95_hi": ci["ci95_hi"], "k": ci["k"],
            "significant": bool(np.isfinite(ci["ci95_lo"]) and
                                (ci["ci95_lo"] > 0 or ci["ci95_hi"] < 0)),
            "exceeds_mde": (bool(abs(ci["mean"]) > MDE[m]) if m in MDE else None),
        }
    return out


def runner_constants(runner_path: str) -> Dict[str, object]:
    """Read run_exp7.py's frozen constants by parsing its source.

    AST-parsed rather than imported (importing would pull torch, transformers and
    the trainer into an aggregation step) and rather than re-declared here
    (a second copy could silently drift from the runner that actually trained).
    The runner source remains the single source of truth.
    """
    wanted = {"BASE_SEED", "EPOCHS", "BATCH_SIZE", "LR", "GRAD_CLIP", "LAMBDA",
              "MAX_LEN", "BINARIZE_THRESHOLD", "N_REPEATS", "N_FOLDS",
              "N_PARTICIPANTS", "BERT_MODEL", "AUDIO_DIM", "VISION_DIM"}
    out: Dict[str, object] = {}
    try:
        with open(runner_path, "r", encoding="utf-8") as fh:
            tree = ast.parse(fh.read())
    except (OSError, SyntaxError) as exc:
        return {"_error": f"{type(exc).__name__}: {exc}"}
    for node in tree.body:
        if isinstance(node, ast.Assign) and isinstance(node.targets[0], ast.Name):
            name = node.targets[0].id
            if name in wanted:
                try:
                    out[name] = ast.literal_eval(node.value)
                except (ValueError, SyntaxError):
                    pass
    return out


def gpu_info() -> Dict[str, object]:
    """Best-effort CUDA/GPU description. Absent torch is reported, not fatal."""
    info: Dict[str, object] = {"platform": platform.platform(),
                               "python": platform.python_version()}
    try:
        import torch
        info["torch"] = torch.__version__
        info["cuda_available"] = bool(torch.cuda.is_available())
        info["device_count"] = int(torch.cuda.device_count())
        if torch.cuda.is_available():
            info["device_name"] = torch.cuda.get_device_name(0)
            info["cuda_version"] = torch.version.cuda
            props = torch.cuda.get_device_properties(0)
            info["total_memory_bytes"] = int(props.total_memory)
            info["capability"] = f"{props.major}.{props.minor}"
    except Exception as exc:  # noqa: BLE001 - reported, never fatal
        info["torch_error"] = f"{type(exc).__name__}: {exc}"
    return info


def sha_or_none(path: str) -> Optional[str]:
    return sha256(path) if os.path.isfile(path) else None


def write_delta(df: pd.DataFrame, exp_dir: str) -> Tuple[str, Dict]:
    """exp7_delta.json - derived ONLY from the official metrics on disk."""
    present = set(df["arm"].unique())
    deltas: Dict[str, Dict] = {}
    skipped: List[str] = []
    for lhs, rhs in DELTA_PAIRS:
        key = f"{lhs}_minus_{rhs}"
        if lhs in present and rhs in present:
            deltas[key] = pairwise_delta(df, lhs, rhs)
        else:
            skipped.append(key)
    payload = {
        "schema": "exp7_delta/1",
        "experiment": "Phase 16 / Exp 7 - multimodal activation",
        "source": "exp7_metrics.csv (official run output); no value is hard-coded",
        "pairing": "per-fold paired; repeats averaged within fold",
        "ci_method": "fold-level Student-t (df=k-1)",
        "t_crit": T_CRIT,
        "mde_anchors": MDE,
        "arm_labels": ARM_LABEL,
        "pairs": [f"{a}_minus_{b}" for a, b in DELTA_PAIRS],
        "pairs_skipped_missing_arm": skipped,
        "metrics": PRIMARY_METRICS,
        "deltas": deltas,
        "note": ("A positive mean_delta means the left arm scored HIGHER. For MAE "
                 "and RMSE lower is better, so a positive delta is a REGRESSION "
                 "there. 'significant' means the 95% CI excludes zero."),
    }
    dest = os.path.join(exp_dir, "exp7_delta.json")
    with open(dest, "w", encoding="utf-8") as fh:
        json.dump(payload, fh, indent=2, sort_keys=True, default=float)
    return dest, payload


def write_receipt(df: pd.DataFrame, exp_dir: str, baseline_dir: str,
                  gate: Dict, ablation: Optional[Dict]) -> Tuple[str, Dict]:
    """exp7_receipt.json - provenance of the official run.

    Every field is read from the run outputs, the frozen artifacts on disk, or
    the runner source. Nothing is asserted from memory.
    """
    runner = os.path.join(os.path.dirname(os.path.abspath(__file__)), "run_exp7.py")
    consts = runner_constants(runner)

    seeds_observed = sorted({int(s) for s in df["seed"].unique()}) if "seed" in df else []
    per_arm_runs = {a: int((df["arm"] == a).sum()) for a in sorted(df["arm"].unique())}
    train_seconds = float(df["seconds"].sum()) if "seconds" in df else None
    abl_seconds = None
    if ablation:
        abl_seconds = float(sum(r.get("seconds", 0.0)
                                for r in (ablation.get("results") or {}).values()))

    outputs = ["exp7_metrics.csv", "modality_config.csv", "exp7_summary.json",
               "a0_equivalence_check.json", "exp7_delta.json",
               "exp7_modality_ablation.json"]
    payload = {
        "schema": "exp7_receipt/1",
        "experiment": "Phase 16 / Exp 7 - multimodal activation",
        "version": "1",
        "factor": "input modality",
        "generated_utc": datetime.datetime.now(datetime.timezone.utc)
                                  .strftime("%Y-%m-%dT%H:%M:%SZ"),
        "frozen_input_sha": {
            "trainer_mentalbert_daic.py": sha_or_none("trainer_mentalbert_daic.py"),
            "daic_records.parquet": sha_or_none("daic_records.parquet"),
            "dataset_build/daic_records_multimodal.parquet":
                sha_or_none("dataset_build/daic_records_multimodal.parquet"),
            "fold_manifest.json":
                sha_or_none(os.path.join(baseline_dir, "fold_manifest.json")),
            "baseline_cv_summary.json":
                sha_or_none(os.path.join(baseline_dir, "baseline_cv_summary.json")),
            "trivial_control_arm.csv":
                sha_or_none(os.path.join(baseline_dir, "trivial_control_arm.csv")),
            "exp3_summary.json":
                sha_or_none("trainer_outputs/exp3_convergence/exp3_summary.json"),
            "exp4_summary.json":
                sha_or_none("trainer_outputs/exp4_decision_rule/exp4_summary.json"),
            "exp5_summary.json":
                sha_or_none("trainer_outputs/exp5_imbalance_objective/exp5_summary.json"),
            "exp6_summary.json":
                sha_or_none("trainer_outputs/exp6_loss_rebalance/exp6_summary.json"),
        },
        "training_configuration": {
            "source": "parsed from exp7_modality/run_exp7.py (single source of truth)",
            **consts,
        },
        "seeding": {
            "rule_stage1_per_repeat": "seed = BASE_SEED + repeat",
            "rule_stage2_per_fold": "torch.manual_seed(seed * 100 + fold)",
            "construction_order": "seed -> model -> dataset -> dataloader -> loop",
            "seeds_observed": seeds_observed,
        },
        "protocol": {
            "n_folds": N_FOLDS, "n_repeats": N_REPEATS,
            "fold_runs_per_arm": N_REPEATS * N_FOLDS,
            "fold_runs_total": N_REPEATS * N_FOLDS * len(ARMS),
            "fold_runs_observed": per_arm_runs,
            "ci_method": "fold-level Student-t (df=k-1), repeats averaged within fold",
            "t_crit": T_CRIT,
            "primary_metrics": PRIMARY_METRICS,
        },
        "arms": {a: {"label": ARM_LABEL[a],
                     "audio_dim": ARM_MODALITY[a][0] or None,
                     "vision_dim": ARM_MODALITY[a][1] or None,
                     "fusion_in": FUSION_IN[a]} for a in ARMS},
        "control_arm": CONTROL_ARM,
        "primary_arm": PRIMARY_ARM,
        "a0_gate": {"passed": bool(gate.get("passed")),
                    "failures": gate.get("failures", []),
                    "n_metrics": gate.get("n_metrics")},
        "environment": gpu_info(),
        "runtime_seconds": {
            "cv_training_total": train_seconds,
            "ablation_total": abl_seconds,
        },
        "output_artifact_sha": {f: sha_or_none(os.path.join(exp_dir, f))
                                for f in outputs},
        "note": ("Serialisation-only artifact. It records the run; it does not "
                 "influence any metric, gate or verdict."),
    }
    dest = os.path.join(exp_dir, "exp7_receipt.json")
    with open(dest, "w", encoding="utf-8") as fh:
        json.dump(payload, fh, indent=2, sort_keys=True, default=float)
    return dest, payload


def main() -> int:
    ap = argparse.ArgumentParser(description="Aggregate Experiment 7")
    ap.add_argument("--exp-dir", default="trainer_outputs/exp7_modality")
    ap.add_argument("--baseline-dir", default="trainer_outputs/baseline_cv")
    ap.add_argument("--ablation", default=None,
                    help="exp7_modality_ablation.json (default: <exp-dir>/…)")
    a = ap.parse_args()
    os.chdir(REPO)

    metrics_path = os.path.join(a.exp_dir, "exp7_metrics.csv")
    if not os.path.isfile(metrics_path):
        abort(f"missing {metrics_path}; run run_exp7.py first")
    baseline_summary = os.path.join(a.baseline_dir, "baseline_cv_summary.json")
    if not os.path.isfile(baseline_summary):
        abort(f"missing frozen Baseline-CV summary: {baseline_summary}")

    df = pd.read_csv(metrics_path)
    print("=" * 74)
    print("EXPERIMENT 7 AGGREGATION")
    print("=" * 74)
    print(f"[INFO] {len(df)} fold-runs, arms {sorted(df['arm'].unique())}")

    # ---- modality routing self-test ---------------------------------------
    for arm in sorted(df["arm"].unique()):
        s = df[df["arm"] == arm]
        ad, vd = ARM_MODALITY.get(arm, (None, None))
        if not (s["audio_dim"] == ad).all() or not (s["vision_dim"] == vd).all():
            abort(f"{arm}: recorded modality dims contradict the pre-declared "
                  f"configuration {ARM_MODALITY[arm]}")
        if not (s["fusion_in"] == FUSION_IN[arm]).all():
            abort(f"{arm}: fusion input is not the expected {FUSION_IN[arm]}")
    print("[OK] modality routing self-test: every arm matches its pre-declared config")

    # ---- THE GATE ----------------------------------------------------------
    gate = a0_equivalence_gate(df, baseline_summary, a.exp_dir)

    # ---- aggregates --------------------------------------------------------
    aggregates: Dict[str, Dict] = {}
    for arm in ARMS:
        s = df[df["arm"] == arm]
        if s.empty:
            continue
        pf = per_fold_means(s, PRIMARY_METRICS)
        aggregates[arm] = {m: fold_level_ci(pf[m].values)
                           for m in PRIMARY_METRICS if m in pf.columns}

    deltas = {arm: paired_delta(df, arm, baseline_summary)
              for arm in ARMS if arm != CONTROL_ARM and not df[df["arm"] == arm].empty}

    ablation_path = a.ablation or os.path.join(a.exp_dir, "exp7_modality_ablation.json")
    ablation: Optional[Dict] = None
    if os.path.isfile(ablation_path):
        with open(ablation_path, "r", encoding="utf-8") as fh:
            ablation = json.load(fh)

    # ---- acceptance criteria (pre-registered) ------------------------------
    crit = []
    crit.append({"id": "C0", "statement": "A0 reproduces Baseline-CV on all primary "
                                          "metrics (artifact swap is inert)",
                 "met": bool(gate["passed"]), "role": "HARD GATE"})
    if ablation:
        pc = ablation.get("primary_criterion", {})
        crit.append({"id": "C1", "statement": "audio ablation non-zero",
                     "met": bool(pc.get("audio_nonzero")), "role": "activation"})
        crit.append({"id": "C2", "statement": "vision ablation non-zero",
                     "met": bool(pc.get("vision_nonzero")), "role": "activation"})
        crit.append({"id": "C3", "statement": "text contribution remains positive",
                     "met": bool(pc.get("text_positive")), "role": "guard"})
    if "A3" in deltas:
        d = deltas["A3"].get("PR_AUC", {})
        crit.append({"id": "C4",
                     "statement": "A3 PR-AUC improves over A0 beyond the MDE (0.0950)",
                     "met": bool(d.get("exceeds_mde") and d.get("mean_delta", 0) > 0),
                     "role": "utility (not required for activation)"})

    # REPORTING GUARD (no scientific behaviour changes here).
    #
    # C1/C2/C3 are appended only when the ablation artifact exists. In an
    # intermediate aggregation - the A0-only run of the execution sequence -
    # `crit` holds C0 alone, so `all(...)` over an empty activation set is
    # VACUOUSLY TRUE and would report activation on the strength of the A0 gate
    # alone, with no ablation evidence whatsoever.
    #
    # The guard below only decides WHICH verdict sentence is emitted. When the
    # ablation evidence is present the expression, the criteria and the verdict
    # wording are exactly what they were before. A0 gate mathematics, criteria
    # definitions and abort behaviour are untouched: a failing A0 has already
    # hard-aborted inside a0_equivalence_gate() long before this line.
    activation_criteria = [c for c in crit if c["role"] == "activation"]
    activation_evaluated = bool(activation_criteria)
    if activation_evaluated:
        activation = all(c["met"] for c in crit
                         if c["role"] in ("HARD GATE", "activation"))
        verdict = ("ACTIVATION DEMONSTRATED - audio and vision reach the graph with the "
                   "text-only control intact" if activation else
                   "ACTIVATION NOT DEMONSTRATED")
    else:
        activation = None
        verdict = ("A0 EQUIVALENCE GATE PASSED - proceed to A1/A2/A3. "
                   "ACTIVATION NOT EVALUATED: the ablation artifact is absent, so "
                   "criteria C1/C2/C3 were not assessed and no activation claim is "
                   "made by this aggregation.")

    summary = {
        "experiment": "Phase 16 / Exp 7 - multimodal activation",
        "factor": "input modality",
        "arms": ARMS,
        "arm_labels": ARM_LABEL,
        "arm_modality": {k: {"audio_dim": v[0], "vision_dim": v[1],
                             "fusion_in": FUSION_IN[k]} for k, v in ARM_MODALITY.items()},
        "control_arm": CONTROL_ARM,
        "primary_arm": PRIMARY_ARM,
        "n_repeats": N_REPEATS, "n_folds": N_FOLDS, "t_crit": T_CRIT,
        "ci_method": "fold-level Student-t (df=k-1), repeats averaged within fold",
        "mde_anchors": MDE,
        "a0_gate": gate,
        "aggregates": aggregates,
        "paired_vs_A0": deltas,
        "ablation": ablation,
        "acceptance_criteria": crit,
        # Machine-readable companion to overall_verdict: False means the ablation
        # artifact was absent and C1/C2/C3 were never assessed, so no activation
        # claim is being made. It does not alter the verdict when evidence exists.
        "activation_evaluated": activation_evaluated,
        "overall_verdict": verdict,
        "caveat": ("A non-zero ablation establishes graph presence, not predictive "
                   "usefulness. Utility is the paired CV deltas (C4)."),
        "frozen_input_sha": {
            "trainer_mentalbert_daic.py": sha256("trainer_mentalbert_daic.py"),
            "daic_records.parquet": sha256("daic_records.parquet"),
            "dataset_build/daic_records_multimodal.parquet":
                sha256("dataset_build/daic_records_multimodal.parquet"),
            os.path.join(a.baseline_dir, "fold_manifest.json"):
                sha256(os.path.join(a.baseline_dir, "fold_manifest.json")),
        },
    }
    dest = os.path.join(a.exp_dir, "exp7_summary.json")
    with open(dest, "w", encoding="utf-8") as fh:
        json.dump(summary, fh, indent=2, sort_keys=True, default=float)

    # ---- console table ------------------------------------------------------
    print("\nper-arm fold-level aggregates (mean [95% CI])")
    hdr = f"{'arm':4s} {'modality':26s} {'MAE':>22s} {'PR_AUC':>22s} {'F1':>10s}"
    print(hdr)
    for arm in ARMS:
        if arm not in aggregates:
            continue
        ag = aggregates[arm]
        def fmt(m):
            g = ag.get(m, {})
            return (f"{g.get('mean', float('nan')):.4f} "
                    f"[{g.get('ci95_lo', float('nan')):.3f},{g.get('ci95_hi', float('nan')):.3f}]")
        print(f"{arm:4s} {ARM_LABEL[arm]:26s} {fmt('MAE'):>22s} {fmt('PR_AUC'):>22s} "
              f"{ag.get('F1', {}).get('mean', float('nan')):>10.4f}")

    print("\nacceptance criteria")
    for c in crit:
        print(f"  {c['id']}  [{'MET' if c['met'] else 'NOT MET'}]  {c['statement']}")
    print(f"\nVERDICT: {verdict}")
    print(f"written: {dest}")

    # ---- serialisation-only outputs ---------------------------------------
    # Written AFTER the summary and the verdict, from results already computed.
    # Delta first, so the receipt can hash it among the output artifacts.
    delta_path, delta_payload = write_delta(df, a.exp_dir)
    n_pairs = len(delta_payload["deltas"])
    print(f"written: {delta_path}  ({n_pairs} arm pair(s)"
          + (f", skipped {delta_payload['pairs_skipped_missing_arm']}"
             if delta_payload["pairs_skipped_missing_arm"] else "") + ")")

    receipt_path, _ = write_receipt(df, a.exp_dir, a.baseline_dir, gate, ablation)
    print(f"written: {receipt_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
