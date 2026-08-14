#!/usr/bin/env python3
"""
run_exp7.py - Phase 16 / Experiment 7: multimodal activation.

THE SINGLE CHANGED FACTOR IS INPUT MODALITY.

Forked from run_exp6.py. Every hyperparameter, seed, fold, metric, CI convention
and output convention is copied unchanged; the only thing that varies across arms
is the (audio_dim, vision_dim) pair handed to the frozen MultiModalModel:

    A0  text only            (None, None)   fusion input  768   CONTROL
    A1  text + audio         (154,  None)   fusion input  896
    A2  text + vision        (None,   84)   fusion input  896
    A3  text + audio + vision(154,   84)    fusion input 1024

No trainer modification is required for any of this: MultiModalModel already
gates encoder construction on `dim is not None and dim > 0` (trainer lines
247-251), and forward() ignores audio_vec/vision_vec when the matching encoder is
absent. Passing the full batch to every arm is therefore harmless and keeps ONE
code path across arms - the same uniformity argument Exp 6 used for lambda.

WHY A0 IS A VALID CONTROL
-------------------------
A0 constructs (None, None), so no SmallMLP is built, the fusion input is 768, and
the torch RNG stream is consumed exactly as Baseline-CV consumed it. A0 therefore
tests one thing only: does swapping the ARTIFACT (text-only parquet -> multimodal
parquet) change the text-only result? It must not. A0 is a HARD GATE; if it does
not reproduce Baseline-CV, no A1/A2/A3 number may be reported.

A DECLARED, UNAVOIDABLE ASYMMETRY
---------------------------------
A1/A2/A3 construct SmallMLP encoders BEFORE FusionHead, so their fusion head
initialises from a different position in the RNG stream than A0's. That is
inseparable from adding a modality - the extra parameters must be drawn from
somewhere - and is a property of the factor, not a confound. It is declared here
rather than discovered later.

THE ABLATION IS NOT COMPUTED HERE
---------------------------------
No historical runner has ever called modality_ablation_importance(). Every
published ablation number came from executing the frozen trainer AS A SCRIPT,
where its main() sets the module global `args` that the ablation needs for its
tokenizer (trainer line 522, comment: "used in modality_ablation_importance for
tokenizer retrieval"). Reproducing that execution semantics is the job of
run_exp7_ablation.py, which invokes the trainer through its own CLI in a
subprocess. Nothing here monkey-patches, injects globals, or edits the trainer.

Usage:
    python run_exp7.py --arms A0                    # the gate, run this FIRST
    python run_exp7.py --arms A1,A2,A3              # only after A0 passes
    python run_exp7.py --smoke --allow-cpu          # 1 arm, 1 repeat, 1 fold
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
from sklearn.metrics import (accuracy_score, average_precision_score,
                             balanced_accuracy_score, f1_score,
                             matthews_corrcoef, mean_absolute_error,
                             precision_score, recall_score, roc_auc_score)

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import trainer_mentalbert_daic as T  # FROZEN trainer - imported, never modified

# --------------------------------------------------------------------------
# Frozen SHAs. Every one of these was measured before this runner existed.
# --------------------------------------------------------------------------
FROZEN_TRAINER_SHA = "65b1902e2ba8dd996c6960db1a6595d84fd48500f4d8707cb79688093d7a230b"
FROZEN_TEXT_PARQUET_SHA = "9a241851760727779fcaa4d50041b8bdc4406fe6b66ddbbd7105f4ce4fc55f00"
FROZEN_MM_PARQUET_SHA = "1ac9f53e6102ec0dbaab84dcfdfa3f2e70f2b4a1867a841ea2b7e3ba24c67a95"

FROZEN_INPUT_SHA = {
    "trainer_outputs/baseline_cv/fold_manifest.json":
        "b9a7a91f1bdd43c78d3cd1ee0c36e4ce3fb8be4b0971dcff3ff62ee5af8fca2f",
    "trainer_outputs/baseline_cv/trivial_control_arm.csv":
        "2ecbfee3225910034a959fe949c7ad027cfa7d41e9bef8040374578ff8c5d572",
    "trainer_outputs/baseline_cv/baseline_cv_summary.json":
        "f065f5e1ff7f8e5ed4d122a8031c9cc12425802e756697d5512a915674871aac",
    "trainer_outputs/exp3_convergence/exp3_summary.json":
        "9dd135b6b79af06a85e64a5aa8c1896d19df8059dd89c053e50e1cffe20af2f9",
    "trainer_outputs/exp4_decision_rule/exp4_summary.json":
        "5defdae2a20d0abc164611e8cbe6b8034e3f65c593d8c9b3e38266a1800aa6f2",
    "trainer_outputs/exp5_imbalance_objective/exp5_summary.json":
        "70ad13ffc4db633419595761c0396fc20dd1b62400ee74238be1f06b73ed48d3",
    # Exp 6 is the immediately-preceding accepted experiment; it is a context
    # comparator here, not a binding one (Exp 6 returned H0, so the binding
    # comparator for A0 remains Baseline-CV - the chained-comparator policy
    # PHASE_15_EXP6_RESULTS.md flagged for restatement in this design).
    "trainer_outputs/exp6_loss_rebalance/exp6_summary.json":
        "bea7478594f6af98943ddda2e997e8bc8780abf57c9c2ee720f9426cd2acc94c",
}

# --------------------------------------------------------------------------
# Pre-declared arms. The modality configuration is fixed BEFORE any result is
# seen; adding, removing or re-routing an arm afterwards is forbidden.
# --------------------------------------------------------------------------
ARM_MODALITY: Dict[str, Dict[str, Optional[int]]] = {
    "A0": {"audio_dim": None, "vision_dim": None},
    "A1": {"audio_dim": 154,  "vision_dim": None},
    "A2": {"audio_dim": None, "vision_dim": 84},
    "A3": {"audio_dim": 154,  "vision_dim": 84},
}
ARM_LABEL = {"A0": "text only", "A1": "text + audio",
             "A2": "text + vision", "A3": "text + audio + vision"}
ARMS: List[str] = ["A0", "A1", "A2", "A3"]
CONTROL_ARM = "A0"
PRIMARY_ARM = "A3"

AUDIO_DIM = 154
VISION_DIM = 84
BERT_HIDDEN = 768
EXPECTED_FUSION_IN = {"A0": 768, "A1": 896, "A2": 896, "A3": 1024}

# --------------------------------------------------------------------------
# Frozen hyperparameters - identical to Baseline-CV and Exps 3-6.
# --------------------------------------------------------------------------
N_REPEATS = 5
N_FOLDS = 5
N_PARTICIPANTS = 188
N_POSITIVE = 45
BASE_SEED = 1000
EPOCHS = 3
BATCH_SIZE = 8
LR = 2e-5
GRAD_CLIP = 1.0
LAMBDA = 0.5              # the frozen regression coefficient; NOT a factor here
MAX_LEN = 128
BINARIZE_THRESHOLD = 10.0
BERT_MODEL = "mental/mental-bert-base-uncased"

PRIMARY_METRICS = ["MAE", "RMSE", "pred_var", "MAE_mean_pred", "ROC_AUC", "PR_AUC",
                   "balAcc", "MCC", "Precision", "Recall", "F1", "Accuracy"]


def sha256(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for b in iter(lambda: fh.read(8192), b""):
            h.update(b)
    return h.hexdigest()


def abort(message: str) -> "NoReturn":  # noqa: F821
    """Stop. Experiment 7 has no partial-success mode."""
    print(f"\n[ABORT] {message}", file=sys.stderr)
    raise SystemExit(1)


def snapshot_input_shas(root: str) -> Dict[str, str]:
    out = {p: sha256(os.path.join(root, p))
           for p in FROZEN_INPUT_SHA if os.path.isfile(os.path.join(root, p))}
    out["trainer_mentalbert_daic.py"] = sha256(T.__file__)
    return out


def resolve_modality(arm: str) -> Tuple[Optional[int], Optional[int]]:
    if arm not in ARM_MODALITY:
        abort(f"unknown arm {arm!r}; pre-declared arms are {ARMS}")
    m = ARM_MODALITY[arm]
    return m["audio_dim"], m["vision_dim"]


# --------------------------------------------------------------------------
# Training loop - byte-for-byte the Exp 6 loop with lam pinned at 0.5.
# --------------------------------------------------------------------------
def _train_one_fold(model, dataset, device: str) -> Tuple[object, List[float]]:
    """The frozen fine_tune_supervised loop, lam = LAMBDA = 0.5 (frozen default).

    UNIFORM CODE PATH: identical statements execute for every arm. The only
    thing that differs between arms is which encoders `model` owns, which was
    decided at construction. That is what makes the A0 gate transitive.

    Optimizer is T.AdamW (transformers.optimization.AdamW) taken from the frozen
    trainer's namespace - torch.optim.AdamW differs in eps and weight_decay and
    must never be substituted.
    """
    model.to(device)
    loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True,
                        collate_fn=T.collate_batch)
    optimizer = T.AdamW(model.parameters(), lr=LR)
    cls_loss_fn = nn.CrossEntropyLoss()
    reg_loss_fn = nn.MSELoss()

    losses: List[float] = []
    for epoch in range(EPOCHS):
        model.train()
        total, steps = 0.0, 0
        for batch in loader:
            batch = {k: (v.to(device) if isinstance(v, torch.Tensor) else v)
                     for k, v in batch.items()}
            optimizer.zero_grad()
            logits, reg_pred, _ = model(batch["input_ids"], batch["attention_mask"],
                                        audio_vec=batch.get("audio_vec"),
                                        vision_vec=batch.get("video_vec"))
            loss_cls = cls_loss_fn(logits, batch["label"])
            loss_reg = reg_loss_fn(reg_pred, batch["phq"])
            if not torch.isfinite(loss_reg):
                abort(f"loss_reg non-finite ({loss_reg.item()}) epoch {epoch+1} "
                      f"step {steps+1}")
            loss = loss_cls + LAMBDA * loss_reg
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), GRAD_CLIP)
            optimizer.step()
            total += loss.item()
            steps += 1
        avg = total / steps if steps else 0.0
        losses.append(avg)
        print(f"    [modality] epoch {epoch+1}/{EPOCHS} avg_loss={avg:.4f}")
    return model, losses


# --------------------------------------------------------------------------
# Metrics - replicate run_baseline_cv.py / run_exp6.py exactly
# --------------------------------------------------------------------------
def rmse(y: Sequence[float], p: Sequence[float]) -> float:
    return float(np.sqrt(np.mean((np.asarray(y, float) - np.asarray(p, float)) ** 2)))


def true_phq(records: List[dict]) -> np.ndarray:
    out = []
    for r in records:
        v = r.get("phq_score") or r.get("phq") or r.get("target_phq")
        try:
            out.append(float(v))
        except Exception:
            out.append(0.0)
    return np.array(out, float)


def infer_dims(records: List[dict]) -> Tuple[object, object]:
    """Verbatim from run_exp6.py / run_baseline_cv.py."""
    audio_dim = vision_dim = None
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


def fold_metrics(y_phq, y_bin, pred_phq, p_pos, pred_class, train_mean) -> Dict[str, float]:
    two_class = len(np.unique(y_bin)) > 1
    return {
        "MAE": float(mean_absolute_error(y_phq, pred_phq)),
        "RMSE": rmse(y_phq, pred_phq),
        "pred_var": float(np.var(pred_phq)),
        "pred_min": float(pred_phq.min()),
        "pred_max": float(pred_phq.max()),
        "MAE_mean_pred": float(mean_absolute_error(y_phq, np.full(len(y_phq), train_mean))),
        "ROC_AUC": float(roc_auc_score(y_bin, p_pos)) if two_class else float("nan"),
        "PR_AUC": float(average_precision_score(y_bin, p_pos)) if two_class else float("nan"),
        "balAcc": float(balanced_accuracy_score(y_bin, pred_class)),
        "MCC": float(matthews_corrcoef(y_bin, pred_class)) if two_class else 0.0,
        "Precision": float(precision_score(y_bin, pred_class, zero_division=0)),
        "Recall": float(recall_score(y_bin, pred_class, zero_division=0)),
        "F1": float(f1_score(y_bin, pred_class, zero_division=0)),
        "Accuracy": float(accuracy_score(y_bin, pred_class)),
    }


def merge_csv(path: str, rows: List[dict], keys: Sequence[str]) -> None:
    """Consolidate immediately (the Experiment 3 resumed-run lesson)."""
    new = pd.DataFrame(rows)
    if os.path.isfile(path):
        old = pd.read_csv(path)
        new = pd.concat([old, new], ignore_index=True)
        new = new.drop_duplicates(subset=list(keys), keep="last")
    new = new.sort_values(list(keys)).reset_index(drop=True)
    new.to_csv(path, index=False)


def a0_advisory_check(out_dir: str, baseline_summary_path: str) -> Optional[Dict]:
    """ADVISORY only - the authoritative A0 gate lives in aggregate_exp7.py.

    Mirrors l0_advisory_check in run_exp6.py: surfacing failure immediately after
    training is useful, but a partially-complete arm is not evidence of
    non-equivalence, so this never aborts.
    """
    mp = os.path.join(out_dir, "exp7_metrics.csv")
    if not (os.path.isfile(mp) and os.path.isfile(baseline_summary_path)):
        return None
    df = pd.read_csv(mp)
    a0 = df[df["arm"] == CONTROL_ARM]
    if len(a0) != N_REPEATS * N_FOLDS:
        print(f"\n[A0 advisory] arm incomplete ({len(a0)}/{N_REPEATS*N_FOLDS} "
              "fold-runs) - equivalence not assessable in this invocation")
        return None
    with open(baseline_summary_path, "r", encoding="utf-8") as fh:
        published = json.load(fh)["metrics"]
    per_fold = a0.groupby("fold")[PRIMARY_METRICS].mean().sort_index()
    rows, outside = [], []
    for m in PRIMARY_METRICS:
        if m not in published:
            continue
        mean = float(np.mean(per_fold[m].values))
        lo, hi = float(published[m]["ci95"][0]), float(published[m]["ci95"][1])
        inside = bool(lo <= mean <= hi)
        rows.append({"metric": m, "a0_mean": mean,
                     "baseline_mean": float(published[m]["mean"]),
                     "baseline_ci_lo": lo, "baseline_ci_hi": hi, "inside_ci": inside})
        if not inside:
            outside.append(m)
    print("\n[A0 advisory] equivalence against Baseline-CV 95% CI")
    for r in rows:
        print(f"    {'ok ' if r['inside_ci'] else 'OUT'} {r['metric']:14s} "
              f"{r['a0_mean']:10.6f}  [{r['baseline_ci_lo']:.6f}, {r['baseline_ci_hi']:.6f}]")
    if outside:
        print(f"    -> {len(outside)} metric(s) OUTSIDE: {outside}")
        print("    -> aggregate_exp7.py will ABORT. A1/A2/A3 must not be reported.")
    else:
        print("    -> all metrics inside; authoritative gate is aggregate_exp7.py")
    return {"advisory": True, "n_metrics": len(rows), "outside": outside,
            "passed_advisory": not outside, "detail": rows}


# --------------------------------------------------------------------------
# Main
# --------------------------------------------------------------------------
def main() -> None:
    ap = argparse.ArgumentParser(
        description="Phase 16 / Experiment 7 - multimodal activation "
                    "(INPUT MODALITY as the single factor).")
    ap.add_argument("--parquet", default="dataset_build/daic_records_multimodal.parquet")
    ap.add_argument("--text-parquet", default="daic_records.parquet",
                    help="frozen text-only artifact, for the byte-identity check")
    ap.add_argument("--baseline-dir", default="trainer_outputs/baseline_cv")
    ap.add_argument("--out-dir", default="trainer_outputs/exp7_modality")
    ap.add_argument("--arms", default=",".join(ARMS))
    ap.add_argument("--repeats", default="1,2,3,4,5")
    ap.add_argument("--base-seed", type=int, default=BASE_SEED)
    ap.add_argument("--bert-model-name", default=BERT_MODEL)
    ap.add_argument("--allow-cpu", action="store_true",
                    help="NON-OFFICIAL smoke tests only; CPU changes a second factor")
    ap.add_argument("--smoke", action="store_true",
                    help="1 repeat x 1 fold per arm; never an official result")
    ap.add_argument("--root", default=None, help="repo root (default: parent of this file)")
    a = ap.parse_args()

    root = a.root or os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    os.chdir(root)
    out_dir = a.out_dir
    baseline_dir = a.baseline_dir

    arms = [x.strip() for x in a.arms.split(",") if x.strip()]
    for arm in arms:
        if arm not in ARMS:
            abort(f"unknown arm {arm!r}; pre-declared arms are {ARMS}")
    repeats = [int(x) for x in a.repeats.split(",") if x.strip()]
    if a.smoke:
        repeats = repeats[:1]

    print("=" * 74)
    print("EXPERIMENT 7 - MULTIMODAL ACTIVATION (single factor: INPUT MODALITY)")
    print("=" * 74)

    # ---- device guard (identical policy to Exp 6) -------------------------
    device = "cuda" if torch.cuda.is_available() else "cpu"
    if device != "cuda" and not a.allow_cpu:
        abort("Experiment 7 MUST run on GPU. Baseline-CV and Exps 3-6 are Colab GPU "
              "results; running on CPU changes a SECOND factor (device) and would "
              "invalidate the A0 equivalence gate. "
              "(--allow-cpu exists only for non-official smoke tests.)")
    if device != "cuda":
        print("[WARN] running on CPU with --allow-cpu: NON-OFFICIAL smoke run only")

    # ---- integrity gates ---------------------------------------------------
    tsha = sha256(T.__file__)
    if tsha != FROZEN_TRAINER_SHA:
        abort(f"trainer SHA {tsha} != frozen {FROZEN_TRAINER_SHA}. Not the verified trainer.")
    print(f"[OK] frozen trainer SHA verified: {tsha}")

    if not os.path.isfile(a.parquet):
        abort(f"multimodal parquet not found: {a.parquet}")
    msha = sha256(a.parquet)
    if msha != FROZEN_MM_PARQUET_SHA:
        abort(f"multimodal parquet SHA {msha} != verified {FROZEN_MM_PARQUET_SHA}. "
              "This is not the S3/S4-verified artifact.")
    print(f"[OK] multimodal parquet SHA verified: {msha}")

    if os.path.isfile(a.text_parquet):
        tp = sha256(a.text_parquet)
        if tp != FROZEN_TEXT_PARQUET_SHA:
            abort(f"frozen text parquet SHA {tp} != {FROZEN_TEXT_PARQUET_SHA}")
        print(f"[OK] frozen text parquet SHA verified: {tp}")
    else:
        abort(f"frozen text parquet not found: {a.text_parquet} "
              "(required for the byte-identity check)")

    for path, want in FROZEN_INPUT_SHA.items():
        if want is None:
            continue
        if not os.path.isfile(path):
            abort(f"missing frozen artifact: {path}")
        got = sha256(path)
        if got != want:
            abort(f"{path} SHA {got} != frozen {want}. A frozen comparator changed.")
    print("[OK] frozen Baseline-CV / Exp 3 / Exp 4 / Exp 5 artifacts verified")

    if getattr(T.AdamW, "__module__", "") != "transformers.optimization":
        abort(f"T.AdamW resolves to {T.AdamW.__module__}; expected "
              "transformers.optimization. torch.optim.AdamW is NOT equivalent.")
    print(f"[OK] optimizer is {T.AdamW.__module__}.{T.AdamW.__name__}")

    shas_before = snapshot_input_shas(root)

    # ---- folds -------------------------------------------------------------
    manifest_path = os.path.join(baseline_dir, "fold_manifest.json")
    with open(manifest_path, "r", encoding="utf-8") as fh:
        manifest = json.load(fh)
    if manifest.get("n_participants") != N_PARTICIPANTS:
        abort(f"manifest n_participants={manifest.get('n_participants')} != {N_PARTICIPANTS}")
    if len(manifest.get("folds", [])) != N_FOLDS:
        abort(f"manifest has {len(manifest.get('folds', []))} folds, expected {N_FOLDS}")
    print(f"[OK] frozen fold manifest: {N_FOLDS} folds, {N_PARTICIPANTS} participants")

    # ---- data --------------------------------------------------------------
    all_records = T.read_parquet_records(a.parquet)
    by_pid = {int(r.get("participant_id")): r for r in all_records}
    if len(by_pid) != N_PARTICIPANTS:
        abort(f"parquet holds {len(by_pid)} participants, expected {N_PARTICIPANTS}")

    audio_dim_seen, vision_dim_seen = infer_dims(all_records)
    if (audio_dim_seen, vision_dim_seen) != (AUDIO_DIM, VISION_DIM):
        abort(f"infer_dims returned ({audio_dim_seen}, {vision_dim_seen}); Experiment 7 "
              f"requires ({AUDIO_DIM}, {VISION_DIM}). The roadmap invariant retires HERE, "
              "but only for the verified multimodal artifact.")
    print(f"[OK] infer_dims = ({audio_dim_seen}, {vision_dim_seen})")

    # ---- byte-identity of the frozen columns (the one-factor evidence) -----
    text_df = pd.read_parquet(a.text_parquet)
    mm_df = pd.read_parquet(a.parquet)
    for col in ("participant_id", "text", "phq_score"):
        if list(text_df[col]) != list(mm_df[col]):
            abort(f"column '{col}' differs between the frozen text parquet and the "
                  "multimodal parquet; the artifact swap is NOT inert and Exp 7 "
                  "would be confounded")
    print("[OK] participant_id / text / phq_score byte-identical to the frozen artifact")

    tokenizer = AutoTokenizer.from_pretrained(a.bert_model_name)
    print(f"[INFO] device={device} arms={arms} repeats={repeats} "
          f"epochs={EPOCHS} lr={LR} batch={BATCH_SIZE} lambda={LAMBDA}")
    print("[INFO] single changed factor: INPUT MODALITY "
          f"{ {k: ARM_MODALITY[k] for k in arms} }")
    if a.smoke:
        print("[INFO] SMOKE MODE: 1 repeat x 1 fold per arm - NOT an official result")
    print("=" * 74)

    os.makedirs(out_dir, exist_ok=True)
    metrics_path = os.path.join(out_dir, "exp7_metrics.csv")
    modality_path = os.path.join(out_dir, "modality_config.csv")

    for arm in arms:
        adim, vdim = resolve_modality(arm)
        for r in repeats:
            out_metrics = os.path.join(out_dir, f"{arm}_rep{r}_metrics.csv")
            if os.path.isfile(out_metrics) and not a.smoke:
                print(f"[skip] {arm} repeat {r} already complete - re-merging its rows")
                merge_csv(metrics_path, pd.read_csv(out_metrics).to_dict("records"),
                          ["arm", "repeat", "fold"])
                continue

            # --- SEEDING STAGE 1 of 2: per repeat (run_baseline_cv.py:125-126)
            seed = a.base_seed + r
            torch.manual_seed(seed)
            np.random.seed(seed)
            rows_this_repeat: List[dict] = []
            mod_rows: List[dict] = []

            folds = manifest["folds"][:1] if a.smoke else manifest["folds"]
            for fold in folds:
                fno = int(fold["fold"])
                tr_ids = [int(i) for i in fold["train_ids"]]
                te_ids = [int(i) for i in fold["test_ids"]]
                if set(tr_ids) & set(te_ids):
                    abort(f"fold {fno}: train/test overlap in the frozen manifest")
                tr_recs = [by_pid[i] for i in tr_ids]
                te_recs = [by_pid[i] for i in te_ids]

                t0 = time.time()
                # --- SEEDING STAGE 2 of 2: per fold (run_baseline_cv.py:145),
                # immediately before model construction. The order
                # seed -> model -> dataset -> dataloader -> loop is NORMATIVE.
                torch.manual_seed(seed * 100 + fno)

                # THE SINGLE CHANGED FACTOR: which encoders exist.
                model = T.MultiModalModel(a.bert_model_name, audio_dim=adim,
                                          vision_dim=vdim, device=device)
                fusion_in = model.fusion.fc1.in_features
                if fusion_in != EXPECTED_FUSION_IN[arm]:
                    abort(f"{arm}: fusion input {fusion_in} != "
                          f"{EXPECTED_FUSION_IN[arm]}; modality routing is wrong")
                if bool(model.has_audio) != (adim is not None):
                    abort(f"{arm}: has_audio={model.has_audio} contradicts audio_dim={adim}")
                if bool(model.has_vision) != (vdim is not None):
                    abort(f"{arm}: has_vision={model.has_vision} contradicts vision_dim={vdim}")

                tr_ds = T.MultiModalDataset(tr_recs, tokenizer, max_len=MAX_LEN)
                te_ds = T.MultiModalDataset(te_recs, tokenizer, max_len=MAX_LEN)

                model, losses = _train_one_fold(model, tr_ds, device)

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
                rows_this_repeat.append({
                    **base, "modality": ARM_LABEL[arm],
                    "audio_dim": adim if adim is not None else 0,
                    "vision_dim": vdim if vdim is not None else 0,
                    "fusion_in": fusion_in, "lambda": LAMBDA, **m,
                    "final_train_loss": losses[-1], "seconds": seconds})
                mod_rows.append({**base, "modality": ARM_LABEL[arm],
                                 "audio_dim": adim if adim is not None else 0,
                                 "vision_dim": vdim if vdim is not None else 0,
                                 "fusion_in": fusion_in,
                                 "has_audio": bool(model.has_audio),
                                 "has_vision": bool(model.has_vision),
                                 "n_params": sum(p.numel() for p in model.parameters())})

                print(f"  [{arm} rep{r} fold{fno}] {ARM_LABEL[arm]:24s} "
                      f"fusion={fusion_in} MAE={m['MAE']:.4f} PR_AUC={m['PR_AUC']:.4f} "
                      f"R={m['Recall']:.3f} F1={m['F1']:.4f} ({seconds}s)")

            suffix = "_smoke" if a.smoke else ""
            pd.DataFrame(rows_this_repeat).to_csv(
                out_metrics.replace(".csv", f"{suffix}.csv"), index=False)
            if not a.smoke:
                merge_csv(metrics_path, rows_this_repeat, ["arm", "repeat", "fold"])
                merge_csv(modality_path, mod_rows, ["arm", "repeat", "fold"])
            print(f"[done] {arm} repeat {r}")

    # ---- post-execution integrity -----------------------------------------
    shas_after = snapshot_input_shas(root)
    changed = [k for k in shas_before if shas_before[k] != shas_after[k]]
    if changed:
        abort(f"frozen artifacts were modified during the run: {changed}")
    print("-" * 74)
    print("frozen artifacts unchanged             : True")
    print(f"frozen trainer SHA still verified      : "
          f"{shas_after['trainer_mentalbert_daic.py'] == FROZEN_TRAINER_SHA}")

    if os.path.isfile(metrics_path) and not a.smoke:
        df = pd.read_csv(metrics_path)
        cols = ["audio_dim", "vision_dim", "fusion_in", "MAE", "RMSE", "pred_var",
                "ROC_AUC", "PR_AUC", "Precision", "Recall", "F1", "seconds"]
        print("\nper-arm means over all fold-runs recorded so far")
        print(df.groupby("arm")[cols].mean().round(4).to_string())
        a0_advisory_check(out_dir, os.path.join(baseline_dir, "baseline_cv_summary.json"))

    print(f"\n[DONE] Experiment 7 -> {out_dir}")
    print("Next: run_exp7_ablation.py (frozen-trainer script-mode ablation), "
          "then aggregate_exp7.py (A0 equivalence gate [HARD ABORT] + paired deltas).")


if __name__ == "__main__":
    main()
