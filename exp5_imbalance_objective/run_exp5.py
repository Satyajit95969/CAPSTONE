#!/usr/bin/env python
"""
Phase 14 / Experiment 5 - Imbalance-Aware Objective.

Tests whether replacing the trainer's UNWEIGHTED cross-entropy with a
CLASS-WEIGHTED cross-entropy produces discrimination, over the frozen
Baseline-CV fold manifest.

THE SINGLE CHANGED FACTOR
-------------------------
The class weighting of the cross-entropy term:

    Baseline-CV / Exp 3 / Exp 4 : nn.CrossEntropyLoss()                # uniform
    Experiment 5                : nn.CrossEntropyLoss(weight=w_fold)   # pre-declared

Nothing else. Architecture, backbone, optimizer, learning rate, epochs (3),
batch size (8), grad-clip (1.0), the MSE term and its 0.5 coefficient, tokenizer,
max_length, folds, seeds, dataset and the argmax decision rule are all held at
Baseline-CV's values.

WHY THIS SCRIPT REIMPLEMENTS THE TRAINING LOOP
----------------------------------------------
Experiment 4 changed the decision rule and never loaded a model - its one-factor
claim was a proof. Experiment 3 changed `epochs`, which is a PARAMETER of the
frozen `fine_tune_supervised`, so no training code was reimplemented; its E0
control reproduced Baseline-CV exactly (max diff 0 across all 25 fold-runs).

Experiment 5 can do neither. The loss is CONSTRUCTED INSIDE
`fine_tune_supervised` (trainer_mentalbert_daic.py:312), and the trainer is
frozen at SHA 65b1902e...a230b and must not be edited. This script therefore
implements its own training loop - a SINGLE code path parameterised by `weight` -
importing MultiModalModel, MultiModalDataset, collate_batch, run_inference,
read_parquet_records and AdamW unchanged from the frozen trainer.

That makes the W0 arm (weight=None) load-bearing in a way Experiment 3's E0 was
not: it is the evidence, rather than the assertion, that the reimplemented loop
is equivalent to the frozen one. See _train_one_fold() for the line-by-line
correspondence, which is also written to loop_correspondence.md as a shipped
artifact.

AdamW NOTE: the optimizer is taken from the frozen trainer's own namespace
(T.AdamW), which is transformers.optimization.AdamW - eps=1e-6, weight_decay=0.0,
correct_bias=True. torch.optim.AdamW is NOT equivalent (eps=1e-8,
weight_decay=0.01, and a different bias-correction placement) and must never be
substituted.

PRE-DECLARED ARMS
-----------------
  W0  uniform (weight=None)            EQUIVALENCE CONTROL - must reproduce Baseline-CV
  W1  balanced, per training fold      PRIMARY
  W2  square-root balanced             sensitivity
  W3  fixed prevalence-inverse         sensitivity, zero-fit control

Weights derive from TRAINING-fold labels only; the held-out fold contributes
nothing to their computation.

Baseline-CV, Exp 3 and Exp 4 artifacts are opened READ-ONLY and SHA-verified
before and after. All output goes to trainer_outputs/exp5_imbalance_objective/.

Usage (from the repository root, with the repo root on PYTHONPATH):
    PYTHONPATH=. python exp5_imbalance_objective/run_exp5.py
    PYTHONPATH=. python exp5_imbalance_objective/run_exp5.py --arms W1 --repeats 3
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import time
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
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
# Frozen constants - Phase 11.6 / 12 / 13 record. Any mismatch aborts.
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

# Pre-declared weighting schemes. Adding, tuning or substituting a scheme after
# results are seen is forbidden by the pre-registration.
ARMS: List[str] = ["W0", "W1", "W2", "W3"]
PRIMARY_ARM = "W1"
CONTROL_ARM = "W0"

N_REPEATS = 5
N_FOLDS = 5
N_PARTICIPANTS = 188
N_POSITIVE = 45
CORPUS_PREVALENCE = N_POSITIVE / N_PARTICIPANTS      # 0.2393617021276596

# Frozen hyperparameters - Baseline-CV's values. Exps 3 and 4 both returned H0,
# so neither contributed an accepted change: epochs reverts to 3 and the decision
# rule reverts to argmax.
BASE_SEED = 1000
EPOCHS = 3
BATCH_SIZE = 8
LR = 2e-5
GRAD_CLIP = 1.0
MSE_COEFF = 0.5
MAX_LEN = 128
BINARIZE_THRESHOLD = 10.0
BERT_MODEL = "mental/mental-bert-base-uncased"

PRIMARY_METRICS = ["MAE", "RMSE", "pred_var", "MAE_mean_pred", "ROC_AUC", "PR_AUC",
                   "balAcc", "MCC", "Precision", "Recall", "F1", "Accuracy"]

LOOP_CORRESPONDENCE = """\
# Experiment 5 - training-loop correspondence

`run_exp5.py::_train_one_fold` versus the frozen
`trainer_mentalbert_daic.py::fine_tune_supervised` (lines 308-330,
SHA 65b1902e...a230b). The CE `weight` argument is the ONLY difference.

| Frozen `fine_tune_supervised` | `_train_one_fold` | Identical? |
|---|---|---|
| `model.to(device)` | `model.to(device)` | yes |
| `DataLoader(dataset, batch_size, shuffle=True, collate_fn=collate_batch)` | same call, `T.collate_batch` | yes |
| `optimizer = AdamW(model.parameters(), lr=lr)` | `T.AdamW(model.parameters(), lr=lr)` | yes - same class object from the frozen module |
| `cls_loss_fn = nn.CrossEntropyLoss()` | `nn.CrossEntropyLoss(weight=w)` | **THE SINGLE CHANGED FACTOR** (`w=None` in W0 reproduces the frozen call exactly) |
| `reg_loss_fn = nn.MSELoss()` | `nn.MSELoss()` | yes |
| `for epoch in range(epochs)` | same, `epochs=3` | yes |
| `model.train()` | `model.train()` | yes |
| `batch = {k: v.to(device) ...}` | same comprehension | yes |
| `optimizer.zero_grad()` | `optimizer.zero_grad()` | yes |
| `model(input_ids, attention_mask, audio_vec=..., vision_vec=...)` | same call and kwargs | yes |
| `loss_cls = cls_loss_fn(logits, batch["label"])` | same | yes |
| `loss_reg = reg_loss_fn(reg_pred, batch["phq"])` | same | yes |
| `loss = loss_cls + 0.5 * loss_reg` | `loss_cls + MSE_COEFF * loss_reg`, `MSE_COEFF = 0.5` | yes |
| `loss.backward()` | `loss.backward()` | yes |
| `clip_grad_norm_(model.parameters(), 1.0)` | same, `GRAD_CLIP = 1.0` | yes |
| `optimizer.step()` | `optimizer.step()` | yes |
| `total_loss += loss.item(); steps += 1` | same | yes |
| `avg = total_loss/steps if steps else 0.0` | same | yes |

**Ordering of RNG consumption is preserved**: the DataLoader is constructed with
`shuffle=True` at the same point relative to model construction and seeding, so
W0 consumes the torch RNG in the same sequence as the frozen loop.

`nn.CrossEntropyLoss(weight=None)` is by definition identical to
`nn.CrossEntropyLoss()`, so the W0 arm exercises the frozen computation exactly.
The W0 equivalence gate in `aggregate_exp5.py` tests this empirically rather than
resting on the argument above.
"""


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
    """Stop. Experiment 5 has no partial-success mode."""
    print(f"\n[ABORT] {message}", file=sys.stderr)
    raise SystemExit(1)


def snapshot_input_shas() -> Dict[str, str]:
    """Digest every frozen artifact this run reads, plus the trainer."""
    shas = {p: sha256(p) for p in FROZEN_INPUT_SHA if os.path.isfile(p)}
    shas["trainer_mentalbert_daic.py"] = sha256(T.__file__)
    return shas


# --------------------------------------------------------------------------
# Class weighting - THE SINGLE CHANGED FACTOR
# --------------------------------------------------------------------------
def compute_class_weights(arm: str, train_labels: np.ndarray) -> Tuple[Optional[torch.Tensor],
                                                                      Dict[str, float]]:
    """Pre-declared weighting schemes (design section 7).

    Weights derive from the TRAINING partition's labels only - the held-out fold
    contributes nothing, so no label information leaks into the operating point.

    Returns (weight_tensor_or_None, provenance_dict). The tensor is ordered
    [w_negative, w_positive] to match CrossEntropyLoss's class-index convention.
    """
    n = int(len(train_labels))
    n_pos = int(np.sum(train_labels == 1))
    n_neg = int(np.sum(train_labels == 0))
    if n_pos == 0 or n_neg == 0:
        abort(f"training partition is single-class (pos={n_pos}, neg={n_neg}); "
              "class weighting is undefined")

    if arm == "W0":
        # Uniform - reproduces nn.CrossEntropyLoss() exactly.
        w_neg = w_pos = 1.0
        weight = None
    elif arm == "W1":
        # Balanced: w_c = n_train / (2 * n_c)
        w_neg = n / (2.0 * n_neg)
        w_pos = n / (2.0 * n_pos)
        weight = torch.tensor([w_neg, w_pos], dtype=torch.float32)
    elif arm == "W2":
        # Square-root balanced - milder correction.
        w_neg = float(np.sqrt(n / (2.0 * n_neg)))
        w_pos = float(np.sqrt(n / (2.0 * n_pos)))
        weight = torch.tensor([w_neg, w_pos], dtype=torch.float32)
    elif arm == "W3":
        # Fixed prevalence-inverse, declared in advance - a zero-fit control.
        w_neg = 1.0
        w_pos = 1.0 / CORPUS_PREVALENCE
        weight = torch.tensor([w_neg, w_pos], dtype=torch.float32)
    else:
        abort(f"unknown arm {arm!r}; pre-declared arms are {ARMS}")

    return weight, {"w_neg": float(w_neg), "w_pos": float(w_pos),
                    "n_train": n, "n_pos_train": n_pos, "n_neg_train": n_neg}


# --------------------------------------------------------------------------
# Training loop - single code path, parameterised by `weight`
# --------------------------------------------------------------------------
def _train_one_fold(model, dataset, weight: Optional[torch.Tensor],
                    device: str) -> Tuple[object, List[float]]:
    """Reimplementation of the frozen `fine_tune_supervised` loop.

    Line-for-line correspondence is documented in LOOP_CORRESPONDENCE (written to
    loop_correspondence.md). The CE `weight` argument is the only difference;
    with weight=None this is the frozen computation exactly.

    The optimizer is T.AdamW - transformers.optimization.AdamW, taken from the
    frozen trainer's own namespace. torch.optim.AdamW differs in eps (1e-8 vs
    1e-6), weight_decay (0.01 vs 0.0) and bias-correction placement, and must
    never be substituted here.
    """
    model.to(device)
    loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True,
                        collate_fn=T.collate_batch)
    optimizer = T.AdamW(model.parameters(), lr=LR)
    cls_loss_fn = nn.CrossEntropyLoss(weight=weight.to(device) if weight is not None else None)
    reg_loss_fn = nn.MSELoss()

    losses: List[float] = []
    for epoch in range(EPOCHS):
        model.train()
        total_loss = 0.0
        steps = 0
        for batch in loader:
            batch = {k: (v.to(device) if isinstance(v, torch.Tensor) else v)
                     for k, v in batch.items()}
            optimizer.zero_grad()
            logits, reg_pred, _ = model(batch["input_ids"], batch["attention_mask"],
                                        audio_vec=batch.get("audio_vec"),
                                        vision_vec=batch.get("video_vec"))
            loss_cls = cls_loss_fn(logits, batch["label"])
            loss_reg = reg_loss_fn(reg_pred, batch["phq"])
            loss = loss_cls + MSE_COEFF * loss_reg
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), GRAD_CLIP)
            optimizer.step()
            total_loss += loss.item()
            steps += 1
        avg = total_loss / steps if steps else 0.0
        losses.append(avg)
        print(f"    [weighted] epoch {epoch + 1}/{EPOCHS} avg_loss={avg:.4f}")
    return model, losses


# --------------------------------------------------------------------------
# Metrics - replicate run_baseline_cv.py exactly
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
    """Replicates run_baseline_cv.py infer_dims() (L50-70). On the text-only DAIC
    parquet this returns (None, None) - the frozen architecture."""
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
    """The Baseline-CV metric suite, same calls and same conventions
    (numpy var uses ddof=0; MCC is 0.0 on a single-class fold)."""
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


def degeneracy_diagnostics(p_pos: np.ndarray, pred_phq: np.ndarray,
                           pred_class: np.ndarray, prevalence: float) -> Dict[str, float]:
    """Criterion-6 observables plus the distribution diagnostics.

    Experiment 4 measured 24/25 degenerate decisions under policy P1 and a
    per-fold probability band 0.0292 wide; Experiment 3's E1 measured 22/25.
    These are the reference points Criterion 6 is judged against.
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
        "prevalence": float(prevalence),
        "degenerate": bool(pos_rate == 0.0 or pos_rate == 1.0),
        "n_pred_pos": int(np.sum(pred_class == 1)),
    }


# --------------------------------------------------------------------------
# Consolidated-artifact merge
# --------------------------------------------------------------------------
def merge_csv(path: str, new_rows: List[dict], keys: Sequence[str]) -> None:
    """Append rows to a consolidated CSV, de-duplicating on the run key.

    Called after EVERY (arm, repeat) rather than only at end of run. An
    end-of-run-only consolidation in Experiment 3 lost an entire arm when an
    invocation was interrupted: the per-repeat files were written inside the loop
    but the consolidated file was not. Merging incrementally means an interrupted
    run leaves consolidated artifacts consistent with whatever completed.
    """
    df_new = pd.DataFrame(new_rows)
    if df_new.empty:
        return
    if os.path.isfile(path):
        df_new = pd.concat([pd.read_csv(path), df_new], ignore_index=True)
    subset = [c for c in keys if c in df_new.columns]
    df_new = df_new.drop_duplicates(subset=subset, keep="last")
    df_new = df_new.sort_values(subset).reset_index(drop=True)
    df_new.to_csv(path, index=False)


# --------------------------------------------------------------------------
# W0 equivalence check (advisory here; authoritative gate is in the aggregator)
# --------------------------------------------------------------------------
def w0_advisory_check(out_dir: str, baseline_summary_path: str) -> Optional[Dict]:
    """Report whether the W0 arm falls inside Baseline-CV's 95% CI.

    This is ADVISORY. The authoritative W0 equivalence gate - the hard abort
    condition of design section 10 - lives in aggregate_exp5.py, which owns the
    acceptance criteria. Surfacing it here lets a failure be seen immediately
    after training rather than at aggregation time, but this function never
    aborts: a partially-complete W0 arm (a resumed run) is not evidence of
    non-equivalence.
    """
    metrics_path = os.path.join(out_dir, "exp5_metrics.csv")
    if not (os.path.isfile(metrics_path) and os.path.isfile(baseline_summary_path)):
        return None
    df = pd.read_csv(metrics_path)
    w0 = df[df["arm"] == CONTROL_ARM]
    if len(w0) != N_REPEATS * N_FOLDS:
        print(f"\n[W0 advisory] arm incomplete ({len(w0)}/{N_REPEATS * N_FOLDS} "
              "fold-runs) - equivalence not assessable in this invocation")
        return None

    with open(baseline_summary_path, "r", encoding="utf-8") as fh:
        published = json.load(fh)["metrics"]
    per_fold = w0.groupby("fold")[PRIMARY_METRICS].mean().sort_index()

    rows, outside = [], []
    for m in PRIMARY_METRICS:
        if m not in published:
            continue
        mean = float(np.mean(per_fold[m].values))
        lo, hi = float(published[m]["ci95"][0]), float(published[m]["ci95"][1])
        inside = bool(lo <= mean <= hi)
        rows.append({"metric": m, "w0_mean": mean, "baseline_mean": float(published[m]["mean"]),
                     "baseline_ci_lo": lo, "baseline_ci_hi": hi, "inside_ci": inside})
        if not inside:
            outside.append(m)

    print("\n[W0 advisory] equivalence against Baseline-CV 95% CI")
    for r in rows:
        flag = "ok " if r["inside_ci"] else "OUT"
        print(f"    {flag} {r['metric']:14s} {r['w0_mean']:10.6f}  "
              f"[{r['baseline_ci_lo']:.6f}, {r['baseline_ci_hi']:.6f}]")
    if outside:
        print(f"    -> {len(outside)} metric(s) OUTSIDE: {outside}")
        print("    -> aggregate_exp5.py will ABORT on this. Investigate before reporting.")
    else:
        print("    -> all metrics inside; the authoritative gate is aggregate_exp5.py")
    return {"advisory": True, "n_metrics": len(rows), "outside": outside,
            "passed_advisory": not outside, "detail": rows}


# --------------------------------------------------------------------------
# Main
# --------------------------------------------------------------------------
def main() -> None:
    ap = argparse.ArgumentParser(
        description="Phase 14 / Experiment 5 - imbalance-aware objective "
                    "(CE class weighting as the single factor).")
    ap.add_argument("--parquet", default="daic_records.parquet")
    ap.add_argument("--baseline-dir", default="trainer_outputs/baseline_cv",
                    help="READ-ONLY frozen Baseline-CV directory")
    ap.add_argument("--out-dir", default="trainer_outputs/exp5_imbalance_objective")
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

    # Structural guard: never write inside a frozen comparator.
    for frozen in [baseline_dir, "trainer_outputs/exp4_decision_rule",
                   "trainer_outputs/exp3_convergence"]:
        fa = os.path.abspath(frozen)
        if os.path.abspath(out_dir) == fa or os.path.abspath(out_dir).startswith(fa + os.sep):
            abort(f"out-dir {out_dir} is inside frozen directory {frozen}")

    arms = [x.strip() for x in a.arms.split(",") if x.strip()]
    repeats = [int(x) for x in a.repeats.split(",") if x.strip()]
    for arm in arms:
        if arm not in ARMS:
            abort(f"unknown arm {arm!r}; pre-declared arms are {ARMS}")

    # ---- device guard -----------------------------------------------------
    device = "cuda" if torch.cuda.is_available() else "cpu"
    if device != "cuda" and not a.allow_cpu:
        abort("Experiment 5 MUST run on GPU. Baseline-CV is a Colab GPU result; "
              "running on CPU changes a SECOND factor (device) and reintroduces "
              "the confound this phase series exists to avoid. "
              "(--allow-cpu exists only for non-official smoke tests.)")

    # ---- integrity gates --------------------------------------------------
    tsha = sha256(T.__file__)
    if tsha != FROZEN_TRAINER_SHA:
        abort(f"trainer SHA {tsha} != frozen {FROZEN_TRAINER_SHA}. Not the verified trainer.")
    shas_before = snapshot_input_shas()
    for path, want in FROZEN_INPUT_SHA.items():
        if not os.path.isfile(path):
            abort(f"missing frozen artifact: {path}")
        if shas_before[path] != want:
            abort(f"{path} SHA {shas_before[path]} != frozen {want}. "
                  "A frozen comparator has been altered; refusing to run.")
    print(f"[OK] frozen trainer SHA verified: {tsha}")
    print("[OK] frozen Baseline-CV / Exp 3 / Exp 4 artifacts verified")

    # The optimizer must be the frozen trainer's AdamW, not torch's.
    if getattr(T.AdamW, "__module__", "") != "transformers.optimization":
        abort(f"T.AdamW resolves to {T.AdamW.__module__}.{T.AdamW.__name__}; expected "
              "transformers.optimization.AdamW. torch.optim.AdamW is NOT equivalent "
              "(eps 1e-8 vs 1e-6, weight_decay 0.01 vs 0.0).")
    print(f"[OK] optimizer is {T.AdamW.__module__}.{T.AdamW.__name__}")

    os.makedirs(out_dir, exist_ok=True)
    with open(os.path.join(out_dir, "loop_correspondence.md"), "w", encoding="utf-8") as fh:
        fh.write(LOOP_CORRESPONDENCE)

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
    print("[OK] infer_dims = (None, None) - text-only architecture preserved")

    tokenizer = AutoTokenizer.from_pretrained(a.bert_model_name)
    print(f"[INFO] device={device} arms={arms} repeats={repeats} "
          f"epochs={EPOCHS} lr={LR} batch={BATCH_SIZE}")
    print(f"[INFO] single changed factor: CE class weighting "
          f"(prevalence {CORPUS_PREVALENCE:.10f})")
    print("=" * 74)

    metrics_path = os.path.join(out_dir, "exp5_metrics.csv")
    weights_path = os.path.join(out_dir, "class_weights.csv")
    degen_path = os.path.join(out_dir, "degeneracy_report.csv")

    for arm in arms:
        for r in repeats:
            out_metrics = os.path.join(out_dir, f"{arm}_rep{r}_metrics.csv")
            if os.path.isfile(out_metrics):
                # Re-merge the existing rows rather than skipping silently. An
                # Experiment 3 invocation skipped completed pairs AND bypassed the
                # accumulator, so a resumed run permanently omitted that arm from
                # the consolidated file.
                print(f"[skip] {arm} repeat {r} already complete - re-merging its rows")
                merge_csv(metrics_path, pd.read_csv(out_metrics).to_dict("records"),
                          ["arm", "repeat", "fold"])
                continue

            seed = a.base_seed + r
            torch.manual_seed(seed)
            np.random.seed(seed)
            rows_this_repeat: List[dict] = []
            weight_rows: List[dict] = []
            degen_rows: List[dict] = []

            for fold in manifest["folds"]:
                fno = int(fold["fold"])
                tr_ids = [int(i) for i in fold["train_ids"]]
                te_ids = [int(i) for i in fold["test_ids"]]
                if set(tr_ids) & set(te_ids):
                    abort(f"fold {fno}: train/test overlap in the frozen manifest")
                tr_recs = [by_pid[i] for i in tr_ids]
                te_recs = [by_pid[i] for i in te_ids]

                t0 = time.time()
                # Per-fold reseed, identical to run_baseline_cv.py (L145), so the
                # W0 arm can reproduce Baseline-CV.
                torch.manual_seed(seed * 100 + fno)

                # THE SINGLE CHANGED FACTOR - weights from TRAINING labels only.
                y_train = (true_phq(tr_recs) > BINARIZE_THRESHOLD).astype(int)
                weight, prov = compute_class_weights(arm, y_train)

                model = T.MultiModalModel(a.bert_model_name, audio_dim=audio_dim,
                                          vision_dim=vision_dim, device=device)
                tr_ds = T.MultiModalDataset(tr_recs, tokenizer, max_len=MAX_LEN)
                te_ds = T.MultiModalDataset(te_recs, tokenizer, max_len=MAX_LEN)

                model, losses = _train_one_fold(model, tr_ds, weight, device)

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
                    "participant_id": te_ids, "phq": y_phq, "phq_bin": y_bin,
                    "pred_phq": pred_phq, "p_pos": p_pos, "pred_class": pred_class,
                }).to_csv(os.path.join(out_dir, f"{arm}_rep{r}_fold{fno}_preds.csv"),
                          index=False)

                base = {"arm": arm, "repeat": r, "fold": fno, "seed": seed,
                        "n_test": len(te_recs), "n_pos": int(y_bin.sum())}
                m = fold_metrics(y_phq, y_bin, pred_phq, p_pos, pred_class, train_mean)
                d = degeneracy_diagnostics(p_pos, pred_phq, pred_class,
                                           float(np.mean(y_bin)))
                rows_this_repeat.append({**base, **m, **prov,
                                         "final_train_loss": losses[-1],
                                         "seconds": seconds})
                weight_rows.append({**base, **prov})
                degen_rows.append({**base, **d})

                print(f"  [{arm} rep{r} fold{fno}] w=({prov['w_neg']:.4f},{prov['w_pos']:.4f}) "
                      f"MAE={m['MAE']:.4f} PR_AUC={m['PR_AUC']:.4f} P={m['Precision']:.4f} "
                      f"R={m['Recall']:.3f} F1={m['F1']:.4f} pos_rate={d['pos_rate']:.3f} "
                      f"({seconds}s)")

            # Per-repeat file, then IMMEDIATE consolidation (Experiment 3 lesson).
            pd.DataFrame(rows_this_repeat).to_csv(out_metrics, index=False)
            merge_csv(metrics_path, rows_this_repeat, ["arm", "repeat", "fold"])
            merge_csv(weights_path, weight_rows, ["arm", "repeat", "fold"])
            merge_csv(degen_path, degen_rows, ["arm", "repeat", "fold"])
            print(f"[done] {arm} repeat {r} -> {out_metrics} (consolidated)")

    # ---- post-execution integrity ----------------------------------------
    shas_after = snapshot_input_shas()
    changed = [k for k in shas_before if shas_before[k] != shas_after[k]]
    if changed:
        abort(f"frozen artifacts were modified during the run: {changed}")

    print("-" * 74)
    print("frozen artifacts unchanged             : True")
    print(f"frozen trainer SHA still verified      : "
          f"{shas_after['trainer_mentalbert_daic.py'] == FROZEN_TRAINER_SHA}")

    if os.path.isfile(metrics_path):
        df = pd.read_csv(metrics_path)
        summary = (df.groupby("arm")[["MAE", "pred_var", "ROC_AUC", "PR_AUC",
                                      "Precision", "Recall", "F1", "seconds"]]
                   .mean().round(4))
        print("\nper-arm means over all fold-runs recorded so far")
        print(summary.to_string())
        if os.path.isfile(degen_path):
            dg = pd.read_csv(degen_path)
            print("\ndegeneracy (Criterion 6 observable; Exp 4 P1 reference = 24/25)")
            for arm in sorted(dg["arm"].unique()):
                s = dg[dg["arm"] == arm]
                print(f"  {arm}: {int(s['degenerate'].sum())}/{len(s)} degenerate | "
                      f"mean pos_rate {s['pos_rate'].mean():.4f} | "
                      f"mean band {s['p_pos_band_width'].mean():.4f}")

    w0_advisory_check(out_dir, os.path.join(baseline_dir, "baseline_cv_summary.json"))

    print(f"\n[DONE] Experiment 5 -> {out_dir}")
    print("Next: aggregate_exp5.py (W0 equivalence gate [HARD ABORT], paired delta "
          "vs Baseline-CV and Exp 4, Criterion-5 discrimination test, Criterion-6 "
          "degeneracy comparison).")


if __name__ == "__main__":
    main()
