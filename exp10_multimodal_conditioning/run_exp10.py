#!/usr/bin/env python3
"""
run_exp10.py - Phase 19 / Experiment 10 runner (GPU).

Authoritative specification: PHASE_19_EXP10_DESIGN.md (FROZEN). Every constant,
arm, threshold, seed and rule is imported from exp10_common.py, which transcribes
that document. Nothing here may be tuned.

SCOPE - design SS 5. This runner adds NO differential privacy, NO masking, NO
compression, NO payload measurement and NO federation. It trains three arms under
one changed factor and records task metrics.

THE SINGLE CHANGED FACTOR IS FEATURE CONDITIONING:

    M0  text-only            (None, None)  fusion 768   raw        CONTROL/GATE
    M1  text+audio+vision    (154,  84)    fusion 1024  raw        CONTROL
    M2  text+audio+vision    (154,  84)    fusion 1024  CONDITIONED TREATMENT

M1 and M2 are architecturally identical - same class, same parameter count, same
tensor shapes, same RNG consumption (design SS 6). Only the feature VALUES differ.

FROZEN PER-FOLD ORDER (design SS 23):
    1. compute fold conditioning statistics from train_ids ONLY   (no RNG)
    2. torch.manual_seed(seed * 100 + fold)
    3. model = T.MultiModalModel(...)                             per arm
    4. records = deepcopy(originals); for M2 replace audio/vision values
    5. T.MultiModalDataset -> DataLoader(collate_fn=T.collate_batch)
    6. frozen training loop -> inference on that fold's test set

The frozen trainer is IMPORTED and never modified: no edit to
_extract_audio_vec/_extract_video_vec, no MultiModalDataset subclass, no
replacement collate. Conditioning is applied by substituting values inside a DEEP
COPY of each record's `features` dict, between read_parquet_records() and
MultiModalDataset (design SS 23).

Usage:
    python exp10_multimodal_conditioning/run_exp10.py
    python exp10_multimodal_conditioning/run_exp10.py --arms M0
    python exp10_multimodal_conditioning/run_exp10.py --smoke --allow-cpu  # NOT official
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from typing import Any, Dict, List, Sequence

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

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import exp10_common as C  # noqa: E402
import trainer_mentalbert_daic as T  # noqa: E402  FROZEN trainer - never modified

# design SS 21.1 - frozen column order.
METRIC_COLUMNS: List[str] = [
    "arm", "audio_dim", "vision_dim", "conditioned", "repeat", "fold", "seed",
    "n_test", "n_pos", "fusion_in", "n_params", "lambda",
    "MAE", "RMSE", "pred_var", "pred_min", "pred_max", "MAE_mean_pred",
    "ROC_AUC", "PR_AUC", "balAcc", "MCC", "Precision", "Recall", "F1", "Accuracy",
    "final_train_loss", "train_seconds", "eval_seconds",
]


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


def fold_metrics(y_phq, y_bin, pred_phq, p_pos, pred_class, train_mean
                 ) -> Dict[str, float]:
    """Verbatim from run_exp7.py / run_exp9_task.py / run_baseline_cv.py.

    Threshold metrics (Precision/Recall/F1/balAcc/MCC/Accuracy) are computed and
    RECORDED but are never acceptance inputs (design SS 16).
    """
    two_class = len(np.unique(y_bin)) > 1
    return {
        "MAE": float(mean_absolute_error(y_phq, pred_phq)),
        "RMSE": rmse(y_phq, pred_phq),
        "pred_var": float(np.var(pred_phq)),
        "pred_min": float(pred_phq.min()),
        "pred_max": float(pred_phq.max()),
        "MAE_mean_pred": float(mean_absolute_error(y_phq,
                                                   np.full(len(y_phq), train_mean))),
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
    """Consolidate immediately (the Experiment 3 resumed-run lesson).

    design SS 21.4 - resumability is per REPEAT and nothing more. A completed
    repeat is re-merged, never re-run. No other checkpointing behaviour exists.
    """
    new = pd.DataFrame(rows)
    if os.path.isfile(path):
        old = pd.read_csv(path)
        new = pd.concat([old, new], ignore_index=True)
        new = new.drop_duplicates(subset=list(keys), keep="last")
    new = new.sort_values(list(keys)).reset_index(drop=True)
    new.to_csv(path, index=False)


def train_one_fold(model, dataset, device: str):
    """The frozen recipe: EPOCHS 3, LR 2e-5, BATCH 8, lambda 0.5, GRAD_CLIP 1.0.

    Byte-for-byte the loop Exps 6, 7 and 9 ran. The optimizer is T.AdamW
    (transformers.optimization.AdamW) taken from the frozen trainer's namespace -
    torch.optim.AdamW differs in eps and weight_decay and must never be
    substituted (design SS 10).
    """
    model.to(device)
    loader = DataLoader(dataset, batch_size=C.BATCH_SIZE, shuffle=True,
                        collate_fn=T.collate_batch)
    optimizer = T.AdamW(model.parameters(), lr=C.LR)
    cls_loss_fn = nn.CrossEntropyLoss()
    reg_loss_fn = nn.MSELoss()

    losses: List[float] = []
    for epoch in range(C.EPOCHS):
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
                C.abort(f"loss_reg non-finite ({loss_reg.item()}) epoch {epoch+1} "
                        f"step {steps+1}")
            loss = loss_cls + C.LAMBDA * loss_reg
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), C.GRAD_CLIP)
            optimizer.step()
            total += loss.item()
            steps += 1
        avg = total / steps if steps else 0.0
        losses.append(avg)
        print(f"    [exp10] epoch {epoch+1}/{C.EPOCHS} avg_loss={avg:.4f}")
    return model, losses


def g0_advisory_check(out_dir: str) -> Dict[str, Any]:
    """ADVISORY ONLY. The authoritative G0 gate lives in aggregate_exp10.py.

    Mirrors a0_advisory_check / k0_advisory_check: surfacing a gate failure right
    after training is useful, but a partially-complete arm is not evidence of
    non-equivalence, so this never aborts and never decides anything.
    """
    mp = os.path.join(out_dir, "exp10_metrics.csv")
    if not os.path.isfile(mp):
        return {"advisory": True, "assessable": False}
    df = pd.read_csv(mp)
    m0 = df[df["arm"] == C.CONTROL_ARM]
    if len(m0) != C.N_REPEATS * C.N_FOLDS:
        print(f"\n[G0 advisory] control incomplete ({len(m0)}/"
              f"{C.N_REPEATS*C.N_FOLDS} fold-runs) - gate not assessable yet")
        return {"advisory": True, "assessable": False}
    per_fold = m0.groupby("fold")[C.PRIMARY_METRIC].mean().sort_index()
    stats = C.fold_level_ci(list(per_fold.values))
    lo, hi = C.ROC_AUC_CI
    inside = bool(lo <= stats["mean"] <= hi)
    print(f"\n[G0 advisory] M0 fold-level ROC-AUC mean {stats['mean']:.6f} "
          f"vs Baseline-CV CI [{lo:.6f}, {hi:.6f}] -> "
          f"{'inside' if inside else 'OUTSIDE'}")
    if not inside:
        print("    -> aggregate_exp10.py will mark the interpretation VOID "
              "(design SS 14). Do not re-run, re-seed or adjust anything.")
    return {"advisory": True, "assessable": True,
            "m0_mean": stats["mean"], "inside_ci": inside}


def build_fold_conditioners(records_by_pid: Dict[int, dict],
                            train_ids: List[int], test_ids: List[int],
                            all_records: List[dict]) -> Dict[str, Any]:
    """design SS 23 step 1 - statistics from TRAIN ids ONLY. No RNG.

    Returns the two conditioners plus the manifest entry and the negative-control
    result for this fold.
    """
    train_recs = [records_by_pid[i] for i in train_ids]
    test_recs = [records_by_pid[i] for i in test_ids]
    A_tr, V_tr = C.extract_modality_matrices(train_recs)
    A_te, V_te = C.extract_modality_matrices(test_recs)
    A_all, V_all = C.extract_modality_matrices(all_records)

    audio_cond = C.FoldConditioner(A_tr, "audio")
    vision_cond = C.FoldConditioner(V_tr, "vision")

    # design SS 7.4 - the zero-variance sets are a measured property; a deviation
    # means the dataset changed underneath the design.
    if tuple(audio_cond.zero_variance_idx) != C.EXPECTED_AUDIO_ZERO_VARIANCE_IDX:
        C.abort(f"audio zero-variance indices {audio_cond.zero_variance_idx} != "
                f"frozen {C.EXPECTED_AUDIO_ZERO_VARIANCE_IDX} (design SS 7.4)")
    if tuple(vision_cond.zero_variance_idx) != C.EXPECTED_VISION_ZERO_VARIANCE_IDX:
        C.abort(f"vision zero-variance indices {vision_cond.zero_variance_idx} != "
                f"frozen {C.EXPECTED_VISION_ZERO_VARIANCE_IDX} (design SS 7.4)")

    # design SS 12.3 - negative control, HARD ABORT on failure.
    nc = [C.negative_control(A_tr, A_te, A_all, audio_cond, "audio"),
          C.negative_control(V_tr, V_te, V_all, vision_cond, "vision")]

    z_tr = np.concatenate([audio_cond.apply(A_tr).ravel(),
                           vision_cond.apply(V_tr).ravel()])
    z_te = np.concatenate([audio_cond.apply(A_te).ravel(),
                           vision_cond.apply(V_te).ravel()])

    entry = {
        "n_train": len(train_ids),
        "train_ids_sha256": C.sha256_bytes(
            json.dumps(sorted(int(i) for i in train_ids)).encode()),
        "audio": audio_cond.manifest(),
        "vision": vision_cond.manifest(),
        "conditioned_train_min": float(z_tr.min()),
        "conditioned_train_max": float(z_tr.max()),
        "conditioned_test_min": float(z_te.min()),
        "conditioned_test_max": float(z_te.max()),
    }
    return {"audio": audio_cond, "vision": vision_cond,
            "entry": entry, "negative_control": nc}


def main() -> int:
    ap = argparse.ArgumentParser(
        description="Phase 19 / Experiment 10 - multimodal feature conditioning")
    ap.add_argument("--parquet", default=C.MULTIMODAL_PARQUET)
    ap.add_argument("--text-parquet", default=C.TEXT_PARQUET)
    ap.add_argument("--baseline-dir",
                    default=os.path.dirname(C.FOLD_MANIFEST))
    ap.add_argument("--out-dir", default=C.OUT_DIRNAME)
    ap.add_argument("--arms", default=",".join(C.ARMS))
    ap.add_argument("--repeats", default=",".join(str(r) for r in C.REPEATS))
    ap.add_argument("--base-seed", type=int, default=C.BASE_SEED)
    ap.add_argument("--bert-model-name", default=C.BERT_MODEL)
    ap.add_argument("--allow-cpu", action="store_true",
                    help="NON-OFFICIAL smoke tests only; CPU changes a second factor")
    ap.add_argument("--smoke", action="store_true",
                    help="1 repeat x 1 fold; never an official result")
    ap.add_argument("--root", default=None)
    a = ap.parse_args()

    root = a.root or C.REPO
    os.chdir(root)
    t_start = time.time()

    arms = [x.strip() for x in a.arms.split(",") if x.strip()]
    for arm in arms:
        if arm not in C.ARMS:
            C.abort(f"unknown arm {arm!r}; the pre-registered arms are "
                    f"{list(C.ARMS)}. design SS 29.1 forbids adding an arm.")
    repeats = [int(x) for x in a.repeats.split(",") if x.strip()]
    if a.smoke:
        repeats = repeats[:1]

    print("=" * 78)
    print("EXPERIMENT 10 - MULTIMODAL FEATURE CONDITIONING")
    print("single factor: conditioning of the audio/vision feature vectors")
    print("no DP - no masking - no payload - no federation (design SS 5)")
    print("=" * 78)

    # ---- device guard (design SS 19) ---------------------------------------
    device = "cuda" if torch.cuda.is_available() else "cpu"
    if device != "cuda" and not a.allow_cpu:
        C.abort("Experiment 10 MUST run on GPU. Baseline-CV and Exps 3-9 are "
                "Colab GPU results; running on CPU changes a SECOND factor "
                "(device) and would invalidate the G0 equivalence gate "
                "(design SS 14, SS 19). --allow-cpu exists only for "
                "non-official smoke tests.")
    if device != "cuda":
        print("[WARN] running on CPU with --allow-cpu: NON-OFFICIAL smoke run only")

    # ---- design SS 13 / SS 26: frozen-input gate, IN the execution path ------
    frozen_before = C.snapshot_frozen()
    C.gate_frozen_before(frozen_before)
    print(f"[OK] {len(C.FROZEN_SHA)} frozen inputs verified before execution")

    import transformers
    if transformers.__version__ != C.REQUIRED_TRANSFORMERS:
        C.abort(f"transformers {transformers.__version__} != required "
                f"{C.REQUIRED_TRANSFORMERS} (design SS 10). The frozen trainer "
                "imports AdamW from transformers, removed in 4.46+, and older "
                "versions change the BERT state_dict key count.")
    if getattr(T.AdamW, "__module__", "") != "transformers.optimization":
        C.abort(f"T.AdamW resolves to {T.AdamW.__module__}; expected "
                "transformers.optimization. torch.optim.AdamW is NOT equivalent.")
    print(f"[OK] transformers {transformers.__version__}; optimizer is "
          f"{T.AdamW.__module__}.{T.AdamW.__name__}")

    # ---- folds: the frozen manifest is authoritative (design SS 9) ----------
    manifest_path = os.path.join(a.baseline_dir, "fold_manifest.json")
    with open(manifest_path, "r", encoding="utf-8") as fh:
        manifest = json.load(fh)
    if manifest.get("n_participants") != C.N_PARTICIPANTS:
        C.abort(f"manifest n_participants={manifest.get('n_participants')} != "
                f"{C.N_PARTICIPANTS}")
    if manifest.get("n_pos") != C.N_POSITIVE or manifest.get("n_neg") != C.N_NEGATIVE:
        C.abort(f"manifest class counts {manifest.get('n_pos')}/"
                f"{manifest.get('n_neg')} != {C.N_POSITIVE}/{C.N_NEGATIVE}")
    if len(manifest.get("folds", [])) != C.N_FOLDS:
        C.abort(f"manifest has {len(manifest.get('folds', []))} folds, expected "
                f"{C.N_FOLDS}")
    print(f"[OK] frozen fold manifest: {C.N_FOLDS} folds, {C.N_PARTICIPANTS} "
          f"participants, {C.N_POSITIVE} pos / {C.N_NEGATIVE} neg")

    # ---- data --------------------------------------------------------------
    all_records = T.read_parquet_records(a.parquet)
    by_pid = {int(r.get("participant_id")): r for r in all_records}
    if len(by_pid) != C.N_PARTICIPANTS:
        C.abort(f"parquet holds {len(by_pid)} participants, expected "
                f"{C.N_PARTICIPANTS}")

    # design SS 12.4 - byte-identity assertion. Without it the fold manifest,
    # which names the TEXT parquet, does not apply to the multimodal artifact.
    text_df = pd.read_parquet(a.text_parquet)
    mm_df = pd.read_parquet(a.parquet)
    for col in C.BYTE_IDENTICAL_COLUMNS:
        if list(text_df[col]) != list(mm_df[col]):
            C.abort(f"column '{col}' differs between the frozen text parquet and "
                    "the multimodal parquet; the fold manifest does not apply "
                    "(design SS 12.4: HARD ABORT)")
    print(f"[OK] {', '.join(C.BYTE_IDENTICAL_COLUMNS)} byte-identical to the "
          "frozen text artifact - the fold manifest applies")

    # design SS 12.7 - coverage is descriptive only, never a model input.
    coverage = {
        "used_as_input": False,
        "audio_coverage": {"min": float(mm_df["audio_coverage"].min()),
                           "mean": float(mm_df["audio_coverage"].mean()),
                           "max": float(mm_df["audio_coverage"].max())},
        "video_coverage": {"min": float(mm_df["video_coverage"].min()),
                           "mean": float(mm_df["video_coverage"].mean()),
                           "max": float(mm_df["video_coverage"].max())},
    }

    tokenizer = AutoTokenizer.from_pretrained(a.bert_model_name)
    os.makedirs(a.out_dir, exist_ok=True)
    metrics_path = os.path.join(a.out_dir, "exp10_metrics.csv")

    print(f"[INFO] device={device} arms={arms} repeats={repeats} "
          f"epochs={C.EPOCHS} lr={C.LR} batch={C.BATCH_SIZE} lambda={C.LAMBDA}")
    print(f"[INFO] conditioning: {C.CONDITIONING_METHOD}, ddof={C.DDOF}, "
          f"{C.ZERO_VARIANCE_RULE}, epsilon={C.CONDITIONING_EPSILON}")
    if a.smoke:
        print("[INFO] SMOKE MODE: 1 repeat x 1 fold - NOT an official result")
    print("=" * 78)

    cond_manifest: Dict[str, Any] = {}
    negative_controls: List[Dict[str, Any]] = []

    for r in repeats:
        rep_out = os.path.join(a.out_dir, f"rep{r}_metrics.csv")
        if os.path.isfile(rep_out) and not a.smoke:
            print(f"[skip] repeat {r} already complete - re-merging its rows")
            merge_csv(metrics_path, pd.read_csv(rep_out).to_dict("records"),
                      ["arm", "repeat", "fold"])
            continue

        seed = a.base_seed + r          # design SS 11
        rows_this_repeat: List[dict] = []
        folds = manifest["folds"][:1] if a.smoke else manifest["folds"]

        for fold in folds:
            fno = int(fold["fold"])
            tr_ids = [int(i) for i in fold["train_ids"]]
            te_ids = [int(i) for i in fold["test_ids"]]
            if set(tr_ids) & set(te_ids):
                C.abort(f"fold {fno}: train/test overlap in the frozen manifest "
                        "(design SS 12.5: HARD ABORT)")
            if (len(tr_ids), len(te_ids)) != C.EXPECTED_FOLD_SIZES.get(fno, ()):
                C.abort(f"fold {fno}: sizes ({len(tr_ids)}, {len(te_ids)}) != "
                        f"frozen {C.EXPECTED_FOLD_SIZES.get(fno)}")

            # --- STEP 1: fold conditioning statistics, TRAIN ONLY, no RNG -----
            if str(fno) not in cond_manifest:
                fc = build_fold_conditioners(by_pid, tr_ids, te_ids, all_records)
                cond_manifest[str(fno)] = fc["entry"]
                negative_controls.extend(
                    [{**n, "fold": fno} for n in fc["negative_control"]])
            else:
                fc = build_fold_conditioners(by_pid, tr_ids, te_ids, all_records)

            tr_raw = [by_pid[i] for i in tr_ids]
            te_raw = [by_pid[i] for i in te_ids]
            y_phq = true_phq(te_raw)
            y_bin = (y_phq > C.BINARIZE_THRESHOLD).astype(int)
            train_mean = float(np.mean(true_phq(tr_raw)))

            for arm in arms:
                spec = C.ARM_SPEC[arm]
                t0 = time.time()

                # --- STEP 2: seeding, immediately before model construction ---
                torch.manual_seed(seed * 100 + fno)

                # --- STEP 3: fresh model every fold (design SS 12.6) ----------
                model = T.MultiModalModel(a.bert_model_name,
                                          audio_dim=spec["audio_dim"],
                                          vision_dim=spec["vision_dim"],
                                          device=device)
                fusion_in = model.fusion.fc1.in_features
                n_params = sum(p.numel() for p in model.parameters())
                if fusion_in != spec["fusion_in"]:
                    C.abort(f"{arm}: fusion input {fusion_in} != frozen "
                            f"{spec['fusion_in']} (design SS 6)")
                if n_params != spec["n_params"]:
                    C.abort(f"{arm}: {n_params} params != frozen "
                            f"{spec['n_params']} (design SS 26: HARD ABORT)")
                if len(model.state_dict()) != spec["n_keys"]:
                    C.abort(f"{arm}: {len(model.state_dict())} state_dict keys != "
                            f"frozen {spec['n_keys']} (design SS 6)")

                # --- STEP 4: DEEP COPIES; conditioned values for M2 only ------
                if spec["conditioned"]:
                    tr_recs = C.condition_records(tr_raw, fc["audio"], fc["vision"])
                    te_recs = C.condition_records(te_raw, fc["audio"], fc["vision"])
                else:
                    tr_recs = C.copy_records(tr_raw)
                    te_recs = C.copy_records(te_raw)

                # --- STEP 5: dataset -> dataloader ---------------------------
                tr_ds = T.MultiModalDataset(tr_recs, tokenizer, max_len=C.MAX_LEN)
                te_ds = T.MultiModalDataset(te_recs, tokenizer, max_len=C.MAX_LEN)

                # --- STEP 6: frozen training loop, then inference -------------
                model, losses = train_one_fold(model, tr_ds, device)
                train_seconds = round(time.time() - t0, 1)

                t1 = time.time()
                te_loader = DataLoader(te_ds, batch_size=C.BATCH_SIZE,
                                       shuffle=False, collate_fn=T.collate_batch)
                preds = T.run_inference(model, te_loader, device=device)
                pred_phq = np.array([p["pred_phq"] for p in preds], float)
                p_pos = np.array([p["pred_class_probs"][1] for p in preds], float)
                pred_class = np.array([p["pred_class"] for p in preds], int)
                m = fold_metrics(y_phq, y_bin, pred_phq, p_pos, pred_class,
                                 train_mean)
                eval_seconds = round(time.time() - t1, 1)

                pd.DataFrame({
                    "participant_id": te_ids, "phq": y_phq, "phq_bin": y_bin,
                    "pred_phq": pred_phq, "p_pos": p_pos, "pred_class": pred_class,
                }).to_csv(os.path.join(a.out_dir,
                                       f"{arm}_rep{r}_fold{fno}_preds.csv"),
                          index=False)

                rows_this_repeat.append({
                    "arm": arm,
                    "audio_dim": spec["audio_dim"] if spec["audio_dim"] else 0,
                    "vision_dim": spec["vision_dim"] if spec["vision_dim"] else 0,
                    "conditioned": bool(spec["conditioned"]),
                    "repeat": r, "fold": fno, "seed": seed,
                    "n_test": len(te_raw), "n_pos": int(y_bin.sum()),
                    "fusion_in": fusion_in, "n_params": n_params,
                    "lambda": C.LAMBDA, **m,
                    "final_train_loss": losses[-1],
                    "train_seconds": train_seconds, "eval_seconds": eval_seconds,
                })
                print(f"  [{arm} rep{r} fold{fno}] {spec['label']:<42s} "
                      f"ROC_AUC={m['ROC_AUC']:.6f} PR_AUC={m['PR_AUC']:.4f} "
                      f"MAE={m['MAE']:.4f} ({train_seconds}s)")

                del model, tr_ds, te_ds, tr_recs, te_recs
                if device == "cuda":
                    torch.cuda.empty_cache()

        suffix = "_smoke" if a.smoke else ""
        pd.DataFrame(rows_this_repeat)[METRIC_COLUMNS].to_csv(
            rep_out.replace(".csv", f"{suffix}.csv"), index=False)
        if not a.smoke:
            merge_csv(metrics_path, rows_this_repeat, ["arm", "repeat", "fold"])
        print(f"[done] repeat {r}")

    # ---- design SS 13: post-run re-verification ------------------------------
    frozen_after = C.gate_frozen_after(frozen_before)
    print("-" * 78)
    print("frozen artifacts unchanged during the run: True")

    if not a.smoke:
        cpath = os.path.join(a.out_dir, "exp10_conditioning_manifest.json")
        with open(cpath, "w", encoding="utf-8") as fh:
            json.dump({
                "schema": "exp10_conditioning/1",
                "design": {"path": C.DESIGN_PATH, "sha256": C.DESIGN_SHA},
                "method": C.CONDITIONING_METHOD,
                "ddof": C.DDOF,
                "zero_variance_rule": C.ZERO_VARIANCE_RULE,
                "epsilon": C.CONDITIONING_EPSILON,
                "applies_to_arms": [C.TREATMENT_ARM],
                "audio_dim": C.AUDIO_DIM, "vision_dim": C.VISION_DIM,
                "folds": cond_manifest,
                "negative_control": {
                    "checked": True,
                    "fold_local_confirmed": all(n["fold_local_confirmed"]
                                                for n in negative_controls),
                    "detail": negative_controls},
            }, fh, indent=2, sort_keys=True)

        advisory = g0_advisory_check(a.out_dir)

        rpath = os.path.join(a.out_dir, "exp10_receipt.json")
        with open(rpath, "w", encoding="utf-8") as fh:
            json.dump({
                "schema": "exp10_receipt/1",
                "experiment": C.EXPERIMENT,
                "design": {"path": C.DESIGN_PATH, "sha256": C.DESIGN_SHA},
                "generated_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
                "session_id": f"exp10-{int(t_start)}",
                "device": device,
                "frozen_input_sha": {"pre_run": frozen_before,
                                     "post_run": frozen_after, "unchanged": True},
                "protocol": {"n_folds": C.N_FOLDS, "n_repeats": C.N_REPEATS,
                             "n_arms": len(C.ARMS),
                             "fold_runs": C.N_FOLDS * C.N_REPEATS * len(C.ARMS),
                             "generator": manifest.get("generator"),
                             "parquet": os.path.basename(a.parquet),
                             "manifest_sha256": C.sha256_file(manifest_path)},
                "recipe": {"epochs": C.EPOCHS, "lr": C.LR,
                           "batch_size": C.BATCH_SIZE, "max_len": C.MAX_LEN,
                           "lambda": C.LAMBDA, "grad_clip": C.GRAD_CLIP,
                           "optimizer": f"{T.AdamW.__module__}.{T.AdamW.__name__}",
                           "bert_model": a.bert_model_name,
                           "binarize_threshold": C.BINARIZE_THRESHOLD},
                "arms": {arm: {k: C.ARM_SPEC[arm][k] for k in
                               ("audio_dim", "vision_dim", "fusion_in",
                                "n_params", "n_keys", "conditioned")}
                         for arm in C.ARMS},
                "seeding": {"base_seed": C.BASE_SEED,
                            "rule": "seed = 1000 + repeat; per fold "
                                    "torch.manual_seed(seed*100 + fold)",
                            "order": "seed -> model -> dataset -> dataloader -> loop",
                            "conditioning_consumes_rng": False},
                "leakage_controls": {"deepcopy_used": True,
                                     "train_only_statistics": True,
                                     "negative_control_passed": True,
                                     "byte_identity_asserted": True,
                                     "checkpoint_reuse": False},
                "coverage": coverage,
                "dp": {"applied": False}, "masking": {"applied": False},
                "payload": {"measured": False}, "federation": {"applied": False},
                "environment": {"python": sys.version.split()[0],
                                "torch": torch.__version__,
                                "cuda": torch.version.cuda,
                                "gpu": (torch.cuda.get_device_name(0)
                                        if torch.cuda.is_available() else None),
                                "transformers": transformers.__version__,
                                "platform": sys.platform},
                "runtime_seconds": round(time.time() - t_start, 2),
                "advisory_g0": advisory,
                "output_artifact_sha": {
                    "exp10_metrics.csv": C.sha256_file(metrics_path),
                    "exp10_conditioning_manifest.json": C.sha256_file(cpath),
                },
                "scope_caveat": C.SCOPE_CAVEAT,
            }, fh, indent=2, sort_keys=True, default=float)
        print(f"[OK] conditioning manifest -> {cpath}")
        print(f"[OK] receipt -> {rpath}")

    print(f"\n[DONE] Experiment 10 -> {a.out_dir}")
    print("Next: aggregate_exp10.py (G0 gate + P1-P4), then verify_exp10.py")
    return 0


if __name__ == "__main__":
    sys.exit(main())
