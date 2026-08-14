#!/usr/bin/env python
"""
Phase 13 / Experiment 3 - Convergence.

Tests whether prior-collapse is a BUDGET ARTIFACT by training the frozen,
unmodified MentalBERT trainer under larger epoch budgets over the frozen
Baseline-CV fold manifest.

THE SINGLE CHANGED FACTOR
-------------------------
The epoch budget passed to the frozen trainer:

    Baseline-CV / Exp 4 : fine_tune_supervised(..., epochs=3)
    Experiment 3        : fine_tune_supervised(..., epochs=N)

Nothing else. The optimizer, learning rate, batch size, loss (CE + 0.5*MSE,
unweighted), gradient clipping, architecture, tokenizer, dataset, folds, seeds
and the argmax decision rule all live INSIDE the frozen trainer or the frozen
manifest and are untouched.

WHY THE ONE-FACTOR GUARANTEE IS UNUSUALLY STRONG HERE
----------------------------------------------------
`epochs` is a PARAMETER of the frozen `fine_tune_supervised`, not a line inside
it. Unlike Experiment 5 - where the loss function would have to be reimplemented
- Experiment 3 changes an argument. No training code is reimplemented, so the
training loop needs no equivalence proof. The E0 arm (3 epochs) exists to
validate only the surrounding harness: data materialisation, seeding, metric
computation and artifact writing.

Note this script DOES import torch and transformers, unlike the Exp 4 scripts.
That is inherent: Experiment 3 must train. It never edits the trainer.

CONVERGENCE WITHOUT A VALIDATION SPLIT
--------------------------------------
Baseline-CV trains on the full 4/5 partition (`val_dataset=None`). Carving a
validation set out of it to drive early stopping would shrink the training set -
a SECOND factor. This design therefore establishes convergence from the
TRAINING-LOSS trajectory, which needs no held-out data and changes no data
volume. The frozen trainer already prints that trajectory; this script captures
stdout to harvest it. Capturing output is observation, not modification.

The epoch budgets are FIXED IN ADVANCE. Convergence is then DEMONSTRATED from
the trajectory, never used to choose which epoch to report - that would be
post-hoc selection.

PRE-DECLARED ARMS
-----------------
  E0   3 epochs   equivalence control (must reproduce Baseline-CV)
  E1  20 epochs   PRIMARY
  E2  10 epochs   sensitivity

Baseline-CV is opened READ-ONLY and SHA-verified before and after. All artifacts
are written under trainer_outputs/exp3_convergence/.

Usage (from the repository root, with the repo root on PYTHONPATH):
    PYTHONPATH=. python exp3_convergence/run_exp3.py
    PYTHONPATH=. python exp3_convergence/run_exp3.py --arms E1 --repeats 3
"""
from __future__ import annotations

import argparse
import contextlib
import hashlib
import io
import json
import os
import re
import sys
import time
from typing import Dict, List, Sequence, Tuple

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader
from transformers import AutoTokenizer
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

import trainer_mentalbert_daic as T  # FROZEN trainer - imported, never modified

# --------------------------------------------------------------------------
# Frozen constants - Phase 11.6 / Phase 12 record. Any mismatch aborts.
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

# Pre-declared epoch budgets. Adding or extending an arm after seeing results is
# forbidden by the pre-registration.
ARMS: Dict[str, int] = {"E0": 3, "E1": 20, "E2": 10}
PRIMARY_ARM = "E1"
CONTROL_ARM = "E0"

N_REPEATS = 5
N_FOLDS = 5
N_PARTICIPANTS = 188

# Frozen hyperparameters - defaults reproduce the Phase 9.5 / Baseline-CV recipe.
BASE_SEED = 1000
BATCH_SIZE = 8
LR = 2e-5
MAX_LEN = 128
BINARIZE_THRESHOLD = 10.0
BERT_MODEL = "mental/mental-bert-base-uncased"

# Pre-declared plateau rule (a REPORTED DIAGNOSTIC, not a selector):
# relative epoch-over-epoch training-loss improvement below PLATEAU_REL_TOL
# sustained for PLATEAU_PATIENCE consecutive epochs.
PLATEAU_REL_TOL = 0.01
PLATEAU_PATIENCE = 2

_LOSS_RE = re.compile(r"\[supervised\]\s+epoch\s+(\d+)/(\d+)\s+avg_loss=([-\d.eE+]+)")


# --------------------------------------------------------------------------
# Integrity helpers
# --------------------------------------------------------------------------
def sha256(path: str) -> str:
    """SHA-256 of a file, streamed."""
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for block in iter(lambda: fh.read(8192), b""):
            h.update(block)
    return h.hexdigest()


def abort(message: str) -> "NoReturn":  # noqa: F821
    """Stop. Experiment 3 has no partial-success mode: a failed gate means the
    result is not trustworthy."""
    print(f"\n[ABORT] {message}", file=sys.stderr)
    raise SystemExit(1)


def snapshot_input_shas(baseline_dir: str) -> Dict[str, str]:
    """Digest the frozen Baseline-CV files this run reads, plus the trainer."""
    shas = {n: sha256(os.path.join(baseline_dir, n)) for n in FROZEN_INPUT_SHA}
    shas["trainer_mentalbert_daic.py"] = sha256(T.__file__)
    return shas


# --------------------------------------------------------------------------
# Metric computation - replicates run_baseline_cv.py exactly
# --------------------------------------------------------------------------
def rmse(y: Sequence[float], p: Sequence[float]) -> float:
    """Identical to run_baseline_cv.py rmse() (L73-74)."""
    return float(np.sqrt(np.mean((np.asarray(y, float) - np.asarray(p, float)) ** 2)))


def true_phq(records: List[dict]) -> np.ndarray:
    """Identical resolution to run_baseline_cv.py true_phq() (L77-86)."""
    out = []
    for r in records:
        v = r.get("phq_score") or r.get("phq") or r.get("target_phq")
        try:
            out.append(float(v))
        except Exception:
            out.append(0.0)
    return np.array(out, float)


def infer_dims(records: List[dict]) -> Tuple[object, object]:
    """Replicates run_baseline_cv.py infer_dims() (L50-70). On the text-only
    DAIC parquet this returns (None, None) - the frozen architecture."""
    audio_dim = None
    vision_dim = None
    for r in records:
        f = r.get("features") or {}
        if isinstance(f, dict) and "audio" in f:
            a = f["audio"]
            if isinstance(a, dict) and a.get("wav2vec2"):
                audio_dim = len(a["wav2vec2"]); break
            if isinstance(a, dict) and a.get("egemaps"):
                audio_dim = len(a["egemaps"].keys()); break
    for r in records:
        f = r.get("features") or {}
        if isinstance(f, dict) and "video" in f and isinstance(f["video"], dict) \
                and f["video"].get("densenet"):
            vision_dim = len(f["video"]["densenet"]); break
        if any(str(k).startswith("neuron_") for k in r.keys()):
            vision_dim = len([k for k in r.keys() if str(k).startswith("neuron_")]); break
    return audio_dim, vision_dim


def fold_metrics(y_phq: np.ndarray, y_bin: np.ndarray, pred_phq: np.ndarray,
                 p_pos: np.ndarray, pred_class: np.ndarray,
                 train_mean: float) -> Dict[str, float]:
    """The Baseline-CV metric suite, computed with the same calls and the same
    conventions (numpy var uses ddof=0; MCC is 0.0 on a single-class fold)."""
    two_class = len(np.unique(y_bin)) > 1
    return {
        "MAE": float(mean_absolute_error(y_phq, pred_phq)),
        "RMSE": rmse(y_phq, pred_phq),
        "pred_var": float(np.var(pred_phq)),
        "pred_min": float(pred_phq.min()),
        "pred_max": float(pred_phq.max()),
        "MAE_mean_pred": float(mean_absolute_error(
            y_phq, np.full(len(y_phq), train_mean))),
        "ROC_AUC": float(roc_auc_score(y_bin, p_pos)) if two_class else float("nan"),
        "PR_AUC": float(average_precision_score(y_bin, p_pos)) if two_class else float("nan"),
        "balAcc": float(balanced_accuracy_score(y_bin, pred_class)),
        "MCC": float(matthews_corrcoef(y_bin, pred_class)) if two_class else 0.0,
        "Precision": float(precision_score(y_bin, pred_class, zero_division=0)),
        "Recall": float(recall_score(y_bin, pred_class, zero_division=0)),
        "F1": float(f1_score(y_bin, pred_class, zero_division=0)),
        "Accuracy": float(accuracy_score(y_bin, pred_class)),
    }


def distribution_diagnostics(p_pos: np.ndarray, pred_phq: np.ndarray,
                             pred_class: np.ndarray) -> Dict[str, float]:
    """SECONDARY OBSERVATIONAL OUTCOMES - reported in full, never pass/fail.

    These address the mechanism Experiment 4 identified (a probability band
    0.0292 wide against a fitted-threshold spread of 0.2241, collapsing 24 of 25
    decisions). They are diagnostics, not acceptance criteria: the primary
    criteria remain convergence behaviour and regression improvement.
    """
    pos_rate = float(np.mean(pred_class))
    return {
        "p_pos_min": float(p_pos.min()),
        "p_pos_max": float(p_pos.max()),
        "p_pos_band_width": float(p_pos.max() - p_pos.min()),
        "p_pos_mean": float(p_pos.mean()),
        "p_pos_sd": float(np.std(p_pos, ddof=1)) if len(p_pos) > 1 else 0.0,
        "pred_phq_spread": float(pred_phq.max() - pred_phq.min()),
        "pos_rate": pos_rate,
        "degenerate": bool(pos_rate == 0.0 or pos_rate == 1.0),
    }


# --------------------------------------------------------------------------
# Convergence analysis
# --------------------------------------------------------------------------
def parse_loss_trajectory(captured: str, expected_epochs: int) -> List[float]:
    """Harvest per-epoch avg_loss from the frozen trainer's own stdout.

    The trainer prints '[supervised] epoch k/N avg_loss=X' once per epoch
    (trainer_mentalbert_daic.py L330). Capturing that output observes the
    trainer; it does not modify it.
    """
    losses = []
    for m in _LOSS_RE.finditer(captured):
        epoch, total, val = int(m.group(1)), int(m.group(2)), float(m.group(3))
        if total != expected_epochs:
            abort(f"trainer reported a budget of {total} epochs, expected {expected_epochs}")
        losses.append((epoch, val))
    losses.sort(key=lambda t: t[0])
    seq = [v for _, v in losses]
    if len(seq) != expected_epochs:
        abort(f"captured {len(seq)} epoch losses, expected {expected_epochs}. "
              "The trainer's print format may have changed.")
    return seq


def plateau_epoch(losses: Sequence[float],
                  rel_tol: float = PLATEAU_REL_TOL,
                  patience: int = PLATEAU_PATIENCE) -> object:
    """First epoch at which the pre-declared plateau rule is satisfied.

    Rule: relative improvement (L[k-1] - L[k]) / L[k-1] < rel_tol, sustained for
    `patience` consecutive epochs. Returns the 1-based epoch index at which the
    run of small improvements completes, or None if never satisfied.

    This is a REPORTED DIAGNOSTIC. The budget was fixed in advance; this value
    never selects which epoch's metrics are reported.
    """
    streak = 0
    for k in range(1, len(losses)):
        prev, cur = losses[k - 1], losses[k]
        rel = (prev - cur) / prev if prev != 0 else 0.0
        if rel < rel_tol:
            streak += 1
            if streak >= patience:
                return k + 1  # 1-based epoch index
        else:
            streak = 0
    return None


# --------------------------------------------------------------------------
# Main
# --------------------------------------------------------------------------
def main() -> None:
    ap = argparse.ArgumentParser(
        description="Phase 13 / Experiment 3 - convergence (epoch budget as the "
                    "single factor) over the frozen Baseline-CV folds.")
    ap.add_argument("--parquet", default="daic_records.parquet")
    ap.add_argument("--baseline-dir", default="trainer_outputs/baseline_cv",
                    help="READ-ONLY frozen Baseline-CV directory")
    ap.add_argument("--out-dir", default="trainer_outputs/exp3_convergence")
    ap.add_argument("--arms", default=",".join(ARMS),
                    help="comma-separated arm ids to run (default all)")
    ap.add_argument("--repeats", default="1,2,3,4,5",
                    help="comma-separated repeat indices (default all)")
    ap.add_argument("--base-seed", type=int, default=BASE_SEED)
    ap.add_argument("--bert-model-name", default=BERT_MODEL)
    ap.add_argument("--allow-cpu", action="store_true",
                    help="override the GPU guard (NOT for official runs)")
    a = ap.parse_args()

    baseline_dir = os.path.normpath(a.baseline_dir)
    out_dir = os.path.normpath(a.out_dir)

    # Structural guard: never write inside the frozen comparator.
    if os.path.abspath(out_dir).startswith(os.path.abspath(baseline_dir) + os.sep) \
            or os.path.abspath(out_dir) == os.path.abspath(baseline_dir):
        abort(f"out-dir {out_dir} is inside the frozen baseline dir {baseline_dir}")

    arms = [x.strip() for x in a.arms.split(",") if x.strip()]
    repeats = [int(x) for x in a.repeats.split(",") if x.strip()]
    for arm in arms:
        if arm not in ARMS:
            abort(f"unknown arm {arm!r}; pre-declared arms are {sorted(ARMS)}")

    # ---- device guard -----------------------------------------------------
    device = "cuda" if torch.cuda.is_available() else "cpu"
    if device != "cuda" and not a.allow_cpu:
        abort("Experiment 3 MUST run on GPU. Baseline-CV is a Colab GPU result; "
              "running on CPU changes a SECOND factor (device) and reintroduces "
              "the confound this phase series exists to avoid. "
              "(--allow-cpu exists only for non-official smoke tests.)")

    # ---- integrity gates --------------------------------------------------
    tsha = sha256(T.__file__)
    if tsha != FROZEN_TRAINER_SHA:
        abort(f"trainer SHA {tsha} != frozen {FROZEN_TRAINER_SHA}. Not the verified trainer.")
    shas_before = snapshot_input_shas(baseline_dir)
    for name, want in FROZEN_INPUT_SHA.items():
        if shas_before[name] != want:
            abort(f"{name} SHA {shas_before[name]} != frozen {want}. "
                  "Baseline-CV has been altered; refusing to run.")
    print(f"[OK] frozen trainer SHA verified: {tsha}")
    print("[OK] frozen Baseline-CV inputs verified")

    os.makedirs(out_dir, exist_ok=True)

    manifest_path = os.path.join(baseline_dir, "fold_manifest.json")
    with open(manifest_path, "r", encoding="utf-8") as fh:
        manifest = json.load(fh)
    if manifest.get("n_participants") != N_PARTICIPANTS:
        abort(f"manifest n_participants={manifest.get('n_participants')} != {N_PARTICIPANTS}")
    if len(manifest.get("folds", [])) != N_FOLDS:
        abort(f"manifest has {len(manifest.get('folds', []))} folds, expected {N_FOLDS}")

    # ---- data -------------------------------------------------------------
    all_records = T.read_parquet_records(a.parquet)
    by_pid = {int(r.get("participant_id")): r for r in all_records}
    if len(by_pid) != N_PARTICIPANTS:
        abort(f"parquet holds {len(by_pid)} participants, expected {N_PARTICIPANTS}")

    audio_dim, vision_dim = infer_dims(all_records)
    if audio_dim is not None or vision_dim is not None:
        abort(f"infer_dims returned ({audio_dim}, {vision_dim}); the roadmap invariant "
              "requires audio/vision to remain absent before Exp 7")
    print(f"[OK] infer_dims = (None, None) - text-only architecture preserved")

    tokenizer = AutoTokenizer.from_pretrained(a.bert_model_name)
    print(f"[INFO] device={device} arms={arms} repeats={repeats}")
    print("=" * 74)

    traj_rows: List[dict] = []
    metric_rows: List[dict] = []
    conv_rows: List[dict] = []
    diag_rows: List[dict] = []

    for arm in arms:
        budget = ARMS[arm]
        for r in repeats:
            out_metrics = os.path.join(out_dir, f"{arm}_rep{r}_metrics.csv")
            if os.path.isfile(out_metrics):
                print(f"[skip] {arm} repeat {r} already complete")
                continue

            seed = a.base_seed + r
            torch.manual_seed(seed)
            np.random.seed(seed)
            rows_this_repeat: List[dict] = []

            for fold in manifest["folds"]:
                fno = int(fold["fold"])
                tr_recs = [by_pid[int(i)] for i in fold["train_ids"]]
                te_recs = [by_pid[int(i)] for i in fold["test_ids"]]

                # Leakage gate - structural, but asserted rather than assumed.
                if set(int(i) for i in fold["train_ids"]) & set(int(i) for i in fold["test_ids"]):
                    abort(f"fold {fno}: train/test overlap in the frozen manifest")

                t0 = time.time()
                # Per-fold reseed, identical to run_baseline_cv.py (L145), so the
                # E0 arm can reproduce Baseline-CV.
                torch.manual_seed(seed * 100 + fno)

                model = T.MultiModalModel(a.bert_model_name, audio_dim=audio_dim,
                                          vision_dim=vision_dim, device=device)
                model.to(device)
                tr_ds = T.MultiModalDataset(tr_recs, tokenizer, max_len=MAX_LEN)
                te_ds = T.MultiModalDataset(te_recs, tokenizer, max_len=MAX_LEN)

                # THE SINGLE CHANGED FACTOR: epochs=budget.
                # val_dataset=None keeps the training set at its full size - no
                # validation carve-out, so training-set size is not a second factor.
                # stdout is captured to harvest the trainer's own per-epoch loss.
                buf = io.StringIO()
                with contextlib.redirect_stdout(buf):
                    model = T.fine_tune_supervised(
                        model, tr_ds, epochs=budget, batch_size=BATCH_SIZE,
                        lr=LR, device=device, val_dataset=None)
                losses = parse_loss_trajectory(buf.getvalue(), budget)

                te_loader = DataLoader(te_ds, batch_size=BATCH_SIZE, shuffle=False,
                                       collate_fn=T.collate_batch)
                preds = T.run_inference(model, te_loader, device=device)

                y_phq = true_phq(te_recs)
                y_bin = (y_phq > BINARIZE_THRESHOLD).astype(int)
                pred_phq = np.array([p["pred_phq"] for p in preds], float)
                p_pos = np.array([p["pred_class_probs"][1] for p in preds], float)
                pred_class = np.array([p["pred_class"] for p in preds], int)
                train_mean = float(np.mean(true_phq(tr_recs)))
                seconds = round(time.time() - t0, 1)

                pd.DataFrame({
                    "participant_id": [int(i) for i in fold["test_ids"]],
                    "phq": y_phq, "phq_bin": y_bin,
                    "pred_phq": pred_phq, "p_pos": p_pos, "pred_class": pred_class,
                }).to_csv(os.path.join(out_dir, f"{arm}_rep{r}_fold{fno}_preds.csv"),
                          index=False)

                m = fold_metrics(y_phq, y_bin, pred_phq, p_pos, pred_class, train_mean)
                diag = distribution_diagnostics(p_pos, pred_phq, pred_class)
                base = {"arm": arm, "epochs": budget, "repeat": r, "fold": fno,
                        "seed": seed, "n_test": len(te_recs), "n_pos": int(y_bin.sum())}
                rows_this_repeat.append({**base, **m, "seconds": seconds})
                diag_rows.append({**base, **diag})

                for k, v in enumerate(losses, start=1):
                    traj_rows.append({**base, "epoch": k, "avg_loss": v})

                pe = plateau_epoch(losses)
                conv_rows.append({
                    **base,
                    "final_loss": losses[-1],
                    "first_loss": losses[0],
                    "total_reduction": losses[0] - losses[-1],
                    "last_epoch_improvement": (losses[-2] - losses[-1]) if len(losses) > 1 else float("nan"),
                    "last_epoch_rel_improvement": ((losses[-2] - losses[-1]) / losses[-2])
                    if len(losses) > 1 and losses[-2] != 0 else float("nan"),
                    "plateau_epoch": pe if pe is not None else -1,
                    "plateau_reached": pe is not None,
                    "seconds": seconds,
                })

                print(f"[{arm} rep{r} fold{fno}] loss {losses[0]:.4f}->{losses[-1]:.4f} "
                      f"plateau@{pe if pe else '-'} MAE={m['MAE']:.4f} "
                      f"pred_var={m['pred_var']:.5f} band={diag['p_pos_band_width']:.4f} "
                      f"R={m['Recall']:.3f} ({seconds}s)")

            pd.DataFrame(rows_this_repeat).to_csv(out_metrics, index=False)
            metric_rows.extend(rows_this_repeat)
            print(f"[done] {arm} repeat {r} -> {out_metrics}")

    # ---- consolidated artifacts ------------------------------------------
    def _merge(path: str, new_rows: List[dict]) -> None:
        """Append to a consolidated CSV, de-duplicating on the run key so that
        resumed invocations do not double-count."""
        df_new = pd.DataFrame(new_rows)
        if df_new.empty:
            return
        if os.path.isfile(path):
            df_new = pd.concat([pd.read_csv(path), df_new], ignore_index=True)
        keys = [c for c in ["arm", "repeat", "fold", "epoch"] if c in df_new.columns]
        df_new = df_new.drop_duplicates(subset=keys, keep="last")
        df_new.to_csv(path, index=False)

    _merge(os.path.join(out_dir, "exp3_metrics.csv"), metric_rows)
    _merge(os.path.join(out_dir, "loss_trajectories.csv"), traj_rows)
    _merge(os.path.join(out_dir, "convergence_report.csv"), conv_rows)
    _merge(os.path.join(out_dir, "distribution_diagnostics.csv"), diag_rows)

    # ---- post-execution integrity ----------------------------------------
    shas_after = snapshot_input_shas(baseline_dir)
    changed = [k for k in shas_before if shas_before[k] != shas_after[k]]
    if changed:
        abort(f"frozen inputs were modified during the run: {changed}")

    print("-" * 74)
    print("frozen inputs unchanged                : True")
    print(f"frozen trainer SHA still verified      : {shas_after['trainer_mentalbert_daic.py'] == FROZEN_TRAINER_SHA}")
    if metric_rows:
        summary = (pd.DataFrame(metric_rows)
                   .groupby("arm")[["MAE", "pred_var", "ROC_AUC", "PR_AUC",
                                    "Recall", "F1", "seconds"]].mean().round(4))
        print("\nper-arm means over the fold-runs executed in this invocation")
        print(summary.to_string())
    print(f"\n[DONE] Experiment 3 -> {out_dir}")
    print("Next: aggregate_exp3.py (E0 equivalence gate, paired delta vs Baseline-CV, "
          "convergence analysis, observational diagnostics).")


if __name__ == "__main__":
    main()
