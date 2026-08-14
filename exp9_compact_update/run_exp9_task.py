#!/usr/bin/env python3
"""
run_exp9_task.py - Phase 18 / Experiment 9, COLAB (GPU) half: the task signal.

Authoritative specification: PHASE_18_EXP9_DESIGN.md (FROZEN), SS 19-SS 24, SS 27.
Constants come from exp9_common.py; nothing here may be tuned.

WHY A SECOND RUNNER EXISTS
--------------------------
design SS 27 splits Experiment 9 across two environments on purpose. The Terminal
half (run_exp9.py) owns the frozen mentalbert_delta.pt, the DP draws and the
transport payload. This half owns the 25 fold trainings, the per-fold deltas and
the 4 x 25 ROC-AUC evaluations, and it must run where the GPU is: Baseline-CV and
Exps 3-8 are all Colab GPU results, and device is not permitted to become a
second factor. Compression happens IN-FLIGHT here so ~11 GB of full deltas never
moves between environments (design SS 21.2, SS 27).

WHAT THIS HALF MEASURES - and what it deliberately does not
-----------------------------------------------------------
design SS 21.2 is explicit:

    for each of the 25 (repeat, fold) runs:
        train under the frozen recipe        -> trained_state
        delta_i = trained_state - base_state
        for each arm k:
            apply S_k, reconstruct, evaluate ROC-AUC on that fold's test set

There is NO DP noise in the task loop, and that is not an omission. design SS 21.1
assigns the DP/SNR half to the frozen mentalbert_delta.pt and the task half to
the 25 per-fold deltas, and design SS 22 requires K0's ROC-AUC to reproduce
Baseline-CV - a noise-free quantity. A noised K0 could not satisfy its own gate,
so the design is internally consistent only under the reading implemented here:
the task half measures the effect of the MASK alone, at k as the single factor.

THE MECHANISM IS NOT TOP-K (design SS 3, SS 32.8). The mask is built from key
names and shapes; no delta value selects a coordinate.

DATASET RULE (design SS 20)
---------------------------
The frozen fold manifest is AUTHORITATIVE and names daic_records.parquet. The
multimodal parquet exists in this repo and must NOT be substituted: the frozen
delta is a text-only object (fusion input 768) and swapping the artifact would
change a second factor and break the SS 22 gate.

Usage (Colab):
    python exp9_compact_update/run_exp9_task.py
    python exp9_compact_update/run_exp9_task.py --smoke --allow-cpu   # NOT official
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

import exp9_common as C  # noqa: E402
import trainer_mentalbert_daic as T  # noqa: E402  FROZEN trainer - never modified

PRIMARY_METRICS = ["MAE", "RMSE", "pred_var", "MAE_mean_pred", "ROC_AUC", "PR_AUC",
                   "balAcc", "MCC", "Precision", "Recall", "F1", "Accuracy"]


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


def fold_metrics(y_phq, y_bin, pred_phq, p_pos, pred_class, train_mean) -> Dict[str, float]:
    """Verbatim from run_exp7.py / run_exp6.py / run_baseline_cv.py."""
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


def train_one_fold(model, dataset, device: str):
    """The frozen recipe: EPOCHS 3, LR 2e-5, BATCH 8, lambda 0.5, GRAD_CLIP 1.0.

    Byte-for-byte the loop Exps 6 and 7 ran, with lambda pinned at the frozen
    0.5. The optimizer is T.AdamW (transformers.optimization.AdamW) taken from
    the frozen trainer's namespace - torch.optim.AdamW differs in eps and
    weight_decay and must never be substituted.

    Capturing the delta afterwards is a READ-OUT; the recipe is unchanged
    (design SS 21.2).
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
        print(f"    [exp9] epoch {epoch+1}/{C.EPOCHS} avg_loss={avg:.4f}")
    return model, losses


def k0_advisory_check(out_dir: str) -> Dict[str, Any] | None:
    """ADVISORY ONLY. The authoritative K0 gate lives in aggregate_exp9.py.

    Mirrors a0_advisory_check in run_exp7.py: surfacing a gate failure right
    after training is useful, but a partially-complete arm is not evidence of
    non-equivalence, so this never aborts and never decides anything.
    """
    mp = os.path.join(out_dir, "exp9_task_metrics.csv")
    if not os.path.isfile(mp):
        return None
    df = pd.read_csv(mp)
    k0 = df[df["arm"] == C.CONTROL_ARM]
    if len(k0) != C.N_REPEATS * C.N_FOLDS:
        print(f"\n[K0 advisory] control incomplete ({len(k0)}/"
              f"{C.N_REPEATS*C.N_FOLDS} fold-runs) - gate not assessable yet")
        return None
    per_fold = k0.groupby("fold")["ROC_AUC"].mean().sort_index()
    stats = C.fold_level_ci(list(per_fold.values))
    lo, hi = C.ROC_AUC_CI
    inside = bool(lo <= stats["mean"] <= hi)
    print(f"\n[K0 advisory] ROC-AUC fold-level mean {stats['mean']:.6f} "
          f"vs Baseline-CV CI [{lo:.6f}, {hi:.6f}] -> "
          f"{'inside' if inside else 'OUTSIDE'}")
    if not inside:
        print("    -> aggregate_exp9.py will HARD ABORT the task interpretation "
              "(design SS 22). No compressed-arm task acceptance may be reported.")
    return {"advisory": True, "k0_mean": stats["mean"], "inside_ci": inside}


def main() -> int:
    ap = argparse.ArgumentParser(
        description="Phase 18 / Experiment 9 - task half (5x5 CV, 4 arms)")
    ap.add_argument("--parquet", default=C.TASK_PARQUET)
    ap.add_argument("--baseline-dir", default=os.path.join("trainer_outputs",
                                                           "baseline_cv"))
    ap.add_argument("--out-dir", default=C.OUT_DIRNAME)
    ap.add_argument("--arms", default=",".join(C.ARMS))
    ap.add_argument("--repeats", default="1,2,3,4,5")
    ap.add_argument("--base-seed", type=int, default=C.BASE_SEED_TASK)
    ap.add_argument("--bert-model-name", default=C.BERT_MODEL)
    ap.add_argument("--allow-cpu", action="store_true",
                    help="NON-OFFICIAL smoke tests only; CPU changes a second factor")
    ap.add_argument("--smoke", action="store_true",
                    help="1 repeat x 1 fold; never an official result")
    ap.add_argument("--root", default=None)
    a = ap.parse_args()

    root = a.root or os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    os.chdir(root)

    arms = [x.strip() for x in a.arms.split(",") if x.strip()]
    for arm in arms:
        if arm not in C.ARMS:
            C.abort(f"unknown arm {arm!r}; the pre-registered ladder is "
                    f"{list(C.ARMS)}. design SS 32.4 forbids adding an arm.")
    repeats = [int(x) for x in a.repeats.split(",") if x.strip()]
    if a.smoke:
        repeats = repeats[:1]

    print("=" * 78)
    print("EXPERIMENT 9 - TASK HALF (5x5 CV; single arm-level factor: k)")
    print(f"mechanism: {C.MECHANISM_NAME}  --  NOT top-k (design SS 3, SS 32.8)")
    print("=" * 78)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    if device != "cuda" and not a.allow_cpu:
        C.abort("Experiment 9's task half MUST run on GPU. Baseline-CV and Exps 3-8 "
                "are Colab GPU results; running on CPU changes a SECOND factor "
                "(device) and would invalidate the K0 equivalence gate (design "
                "SS 22). --allow-cpu exists only for non-official smoke tests.")
    if device != "cuda":
        print("[WARN] running on CPU with --allow-cpu: NON-OFFICIAL smoke run only")

    # ---- frozen-input gate, IN the execution path (design SS 28.1, SS 29) ----
    frozen_before = C.snapshot_frozen(C.COLAB_REQUIRED)
    C.gate_frozen_before(frozen_before, C.COLAB_REQUIRED)
    print(f"[OK] {len(C.COLAB_REQUIRED)} frozen inputs verified before execution")

    if getattr(T.AdamW, "__module__", "") != "transformers.optimization":
        C.abort(f"T.AdamW resolves to {T.AdamW.__module__}; expected "
                "transformers.optimization. torch.optim.AdamW is NOT equivalent.")
    print(f"[OK] optimizer is {T.AdamW.__module__}.{T.AdamW.__name__}")

    # ---- folds: the frozen manifest is authoritative (design SS 20) ----------
    manifest_path = os.path.join(a.baseline_dir, "fold_manifest.json")
    with open(manifest_path, "r", encoding="utf-8") as fh:
        manifest = json.load(fh)
    if manifest.get("n_participants") != C.N_PARTICIPANTS:
        C.abort(f"manifest n_participants={manifest.get('n_participants')} != "
                f"{C.N_PARTICIPANTS}")
    if len(manifest.get("folds", [])) != C.N_FOLDS:
        C.abort(f"manifest has {len(manifest.get('folds', []))} folds, expected "
                f"{C.N_FOLDS}")
    if manifest.get("parquet") and os.path.basename(a.parquet) != \
            os.path.basename(str(manifest["parquet"])):
        C.abort(f"the frozen manifest names {manifest['parquet']!r} but --parquet is "
                f"{a.parquet!r}. design SS 20: the manifest is AUTHORITATIVE and no "
                "other dataset may be substituted.")
    print(f"[OK] frozen fold manifest: {C.N_FOLDS} folds, {C.N_PARTICIPANTS} "
          f"participants, parquet={manifest.get('parquet')}")

    all_records = T.read_parquet_records(a.parquet)
    by_pid = {int(r.get("participant_id")): r for r in all_records}
    if len(by_pid) != C.N_PARTICIPANTS:
        C.abort(f"parquet holds {len(by_pid)} participants, expected "
                f"{C.N_PARTICIPANTS}")

    tokenizer = AutoTokenizer.from_pretrained(a.bert_model_name)
    os.makedirs(a.out_dir, exist_ok=True)
    metrics_path = os.path.join(a.out_dir, "exp9_task_metrics.csv")

    print(f"[INFO] device={device} arms={arms} repeats={repeats} "
          f"epochs={C.EPOCHS} lr={C.LR} batch={C.BATCH_SIZE} lambda={C.LAMBDA}")
    print(f"[INFO] single arm-level factor: k = "
          f"{ {x: C.K_VALUES[x] for x in arms} }")
    if a.smoke:
        print("[INFO] SMOKE MODE: 1 repeat x 1 fold - NOT an official result")
    print("=" * 78)

    order = None
    struct = None

    for r in repeats:
        rep_out = os.path.join(a.out_dir, f"task_rep{r}_metrics.csv")
        if os.path.isfile(rep_out) and not a.smoke:
            print(f"[skip] repeat {r} already complete - re-merging its rows")
            merge_csv(metrics_path, pd.read_csv(rep_out).to_dict("records"),
                      ["arm", "repeat", "fold"])
            continue

        # --- SEEDING STAGE 1 of 2: per repeat (design SS 20) -----------------
        seed = a.base_seed + r
        torch.manual_seed(seed)
        np.random.seed(seed)
        rows_this_repeat: List[dict] = []

        folds = manifest["folds"][:1] if a.smoke else manifest["folds"]
        for fold in folds:
            fno = int(fold["fold"])
            tr_ids = [int(i) for i in fold["train_ids"]]
            te_ids = [int(i) for i in fold["test_ids"]]
            if set(tr_ids) & set(te_ids):
                C.abort(f"fold {fno}: train/test overlap in the frozen manifest")
            tr_recs = [by_pid[i] for i in tr_ids]
            te_recs = [by_pid[i] for i in te_ids]

            t0 = time.time()
            # --- SEEDING STAGE 2 of 2: per fold, immediately before model
            # construction. The order seed -> model -> dataset -> dataloader ->
            # loop is NORMATIVE (design SS 20).
            torch.manual_seed(seed * 100 + fno)

            model = T.MultiModalModel(a.bert_model_name, audio_dim=C.AUDIO_DIM,
                                      vision_dim=C.VISION_DIM, device=device)
            fusion_in = model.fusion.fc1.in_features
            if fusion_in != C.EXPECTED_FUSION_IN:
                C.abort(f"fusion input {fusion_in} != {C.EXPECTED_FUSION_IN}; this is "
                        "not the text-only architecture the frozen delta describes")

            # base_state BEFORE training - the reference for delta_i (design SS 21.2)
            base_state = {k: v.detach().cpu().clone()
                          for k, v in model.state_dict().items()}

            if order is None:
                numels = {k: int(v.numel()) for k, v in base_state.items()}
                order = C.build_ordering(list(base_state.keys()), numels)
                struct = C.verify_ordering(order)
                nesting = C.verify_nesting(order)
                print(f"[OK] ordering O: d={struct['d']:,} keys={struct['n_keys']} "
                      f"groups verified; nesting={nesting['nested']}")
                with open(os.path.join(a.out_dir, "exp9_task_mask_manifest.json"),
                          "w", encoding="utf-8") as fh:
                    json.dump({"schema": "exp9_task_mask_manifest/1",
                               "design_sha256": C.DESIGN_SHA,
                               "mechanism": C.MECHANISM_NAME, "magnitude_based": False,
                               "structure": struct, "nesting": nesting,
                               "k_values": C.K_VALUES}, fh, indent=2, sort_keys=True)

            tr_ds = T.MultiModalDataset(tr_recs, tokenizer, max_len=C.MAX_LEN)
            te_ds = T.MultiModalDataset(te_recs, tokenizer, max_len=C.MAX_LEN)

            model, losses = train_one_fold(model, tr_ds, device)
            train_seconds = round(time.time() - t0, 1)

            # delta_i = trained_state - base_state, via the FROZEN trainer helper.
            delta = T.compute_state_delta(
                base_state, {k: v.detach().cpu() for k, v in model.state_dict().items()})
            # design SS 21.2: compressed IN-FLIGHT; the full delta is never persisted.

            te_loader = DataLoader(te_ds, batch_size=C.BATCH_SIZE, shuffle=False,
                                   collate_fn=T.collate_batch)
            y_phq = true_phq(te_recs)
            y_bin = (y_phq > C.BINARIZE_THRESHOLD).astype(int)
            train_mean = float(np.mean(true_phq(tr_recs)))

            for arm in arms:
                k = C.K_VALUES[arm]
                t1 = time.time()
                masked = C.apply_mask_to_delta(delta, order, k)
                recon = {kk: (base_state[kk] + masked[kk]) for kk in masked}
                del masked
                # 207-key reconstruction; strict=True must hold (design SS 11)
                model.load_state_dict(recon, strict=True)
                del recon
                model.to(device)

                preds = T.run_inference(model, te_loader, device=device)
                pred_phq = np.array([p["pred_phq"] for p in preds], float)
                p_pos = np.array([p["pred_class_probs"][1] for p in preds], float)
                pred_class = np.array([p["pred_class"] for p in preds], int)
                m = fold_metrics(y_phq, y_bin, pred_phq, p_pos, pred_class, train_mean)

                pd.DataFrame({
                    "participant_id": te_ids, "phq": y_phq, "phq_bin": y_bin,
                    "pred_phq": pred_phq, "p_pos": p_pos, "pred_class": pred_class,
                }).to_csv(os.path.join(a.out_dir,
                                       f"{arm}_rep{r}_fold{fno}_preds.csv"),
                          index=False)

                rows_this_repeat.append({
                    "arm": arm, "k": k, "repeat": r, "fold": fno, "seed": seed,
                    "n_test": len(te_recs), "n_pos": int(y_bin.sum()),
                    "fusion_in": fusion_in, "lambda": C.LAMBDA, **m,
                    "final_train_loss": losses[-1],
                    "train_seconds": train_seconds,
                    "eval_seconds": round(time.time() - t1, 1)})

                print(f"  [{arm} rep{r} fold{fno}] k={k:>12,} "
                      f"ROC_AUC={m['ROC_AUC']:.6f} PR_AUC={m['PR_AUC']:.4f} "
                      f"MAE={m['MAE']:.4f}")

            del delta, base_state, model
            if device == "cuda":
                torch.cuda.empty_cache()

        suffix = "_smoke" if a.smoke else ""
        pd.DataFrame(rows_this_repeat).to_csv(
            rep_out.replace(".csv", f"{suffix}.csv"), index=False)
        if not a.smoke:
            merge_csv(metrics_path, rows_this_repeat, ["arm", "repeat", "fold"])
        print(f"[done] repeat {r}")

    frozen_after = C.gate_frozen_after(frozen_before)
    print("-" * 78)
    print("frozen artifacts unchanged during the run: True")

    if not a.smoke:
        k0_advisory_check(a.out_dir)
        with open(os.path.join(a.out_dir, "exp9_task_receipt.json"), "w",
                  encoding="utf-8") as fh:
            json.dump({
                "schema": "exp9_task_receipt/1",
                "experiment": C.EXPERIMENT,
                "half": "task (5x5 CV, ROC-AUC)",
                "design": {"path": C.DESIGN_PATH, "sha256": C.DESIGN_SHA},
                "generated_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
                "device": device,
                "frozen_input_sha": {"pre_run": frozen_before,
                                     "post_run": frozen_after, "unchanged": True},
                "protocol": {"n_folds": C.N_FOLDS, "n_repeats": C.N_REPEATS,
                             "fold_runs_per_arm": C.N_FOLDS * C.N_REPEATS,
                             "generator": manifest.get("generator"),
                             "parquet": manifest.get("parquet"),
                             "manifest_sha256": C.sha256_file(manifest_path)},
                "recipe": {"epochs": C.EPOCHS, "lr": C.LR,
                           "batch_size": C.BATCH_SIZE, "max_len": C.MAX_LEN,
                           "lambda": C.LAMBDA, "grad_clip": C.GRAD_CLIP,
                           "bert_model": a.bert_model_name,
                           "optimizer": f"{T.AdamW.__module__}.{T.AdamW.__name__}",
                           "binarize_threshold": C.BINARIZE_THRESHOLD},
                "seeding": {"rule": "seed = 1000 + repeat; per fold "
                                    "torch.manual_seed(seed*100 + fold)",
                            "order": "seed -> model -> dataset -> dataloader -> loop"},
                "mask": {"mechanism": C.MECHANISM_NAME, "magnitude_based": False,
                         "k_values": C.K_VALUES,
                         "structure": struct},
                "dp_noise_in_task_half": False,
                "dp_note": ("design SS 21.1 assigns DP/SNR to the frozen "
                            "mentalbert_delta.pt; the task half measures the mask "
                            "alone, which is what SS 22's noise-free K0 gate requires"),
                "environment": {"python": sys.version.split()[0],
                                "platform": sys.platform, "torch": torch.__version__},
            }, fh, indent=2, sort_keys=True, default=float)

    print(f"\n[DONE] Experiment 9 task half -> {a.out_dir}")
    print("Bring back trainer_outputs/exp9_compact_update/ in full, then run "
          "aggregate_exp9.py and verify_exp9.py on the Terminal.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
