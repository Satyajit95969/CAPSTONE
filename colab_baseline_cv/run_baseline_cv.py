#!/usr/bin/env python
"""
Baseline-CV fold driver (Phase 11.6).

Evaluates the FROZEN, UNMODIFIED MentalBERT trainer under participant-level
cross-validation, using the frozen fold manifest.

This script does NOT modify the trainer. It imports and calls the trainer's own
functions (MultiModalModel, fine_tune_supervised, run_inference, MultiModalDataset,
collate_batch, read_parquet_records). Architecture, optimizer (AdamW), loss
(CE + 0.5*MSE), grad-clip (1.0), epochs, batch size, lr, tokenizer, and
preprocessing are therefore the trainer's UNCHANGED code.

THE ONLY CHANGE vs Phase 9.5: the single seed-42 90/10 split is replaced by the
participant-level stratified folds in fold_manifest.json. Per fold the model is
trained on the 4/5 train partition (val_dataset=None => trains on the FULL
partition, since the internal 10% holdout is exactly what CV replaces) and
evaluated on the held-out 1/5 test partition.

Run ONCE PER REPEAT:
    python run_baseline_cv.py --repeat 1
    python run_baseline_cv.py --repeat 2
    ...
Each invocation processes all 5 folds for that repeat.
"""
import argparse, json, os, hashlib, time
import numpy as np
import pandas as pd
import pyarrow.parquet as pq
import torch
from torch.utils.data import DataLoader
from transformers import AutoTokenizer
from sklearn.metrics import (mean_absolute_error, roc_auc_score, average_precision_score,
    balanced_accuracy_score, matthews_corrcoef, precision_score, recall_score, f1_score,
    accuracy_score)

import trainer_mentalbert_daic as T  # FROZEN trainer, imported not modified

FROZEN_TRAINER_SHA = "65b1902e2ba8dd996c6960db1a6595d84fd48500f4d8707cb79688093d7a230b"


def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(8192), b""):
            h.update(chunk)
    return h.hexdigest()


def infer_dims(records):
    """Replicate the trainer's own audio/vision dim inference (main(), L533-551).
    On the text-only DAIC parquet this yields (None, None) => identical architecture
    to the frozen run."""
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
        if isinstance(f, dict) and "video" in f and isinstance(f["video"], dict) and f["video"].get("densenet"):
            vision_dim = len(f["video"]["densenet"]); break
        if any(str(k).startswith("neuron_") for k in r.keys()):
            vision_dim = len([k for k in r.keys() if str(k).startswith("neuron_")]); break
    return audio_dim, vision_dim


def rmse(y, p):
    return float(np.sqrt(np.mean((np.asarray(y, float) - np.asarray(p, float)) ** 2)))


def true_phq(records):
    # identical resolution to the trainer's y_true (L603-607)
    out = []
    for r in records:
        v = r.get("phq_score") or r.get("phq") or r.get("target_phq")
        try:
            out.append(float(v))
        except Exception:
            out.append(0.0)
    return np.array(out, float)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--parquet", default="daic_records.parquet")
    ap.add_argument("--manifest", default="trainer_outputs/baseline_cv/fold_manifest.json")
    ap.add_argument("--out-dir", default="trainer_outputs/baseline_cv/runs")
    ap.add_argument("--repeat", type=int, required=True, help="repeat index r (1..R)")
    # Frozen hyperparameters — DO NOT CHANGE (defaults reproduce Phase 9.5 recipe)
    ap.add_argument("--epochs", type=int, default=3)
    ap.add_argument("--batch-size", type=int, default=8)
    ap.add_argument("--lr", type=float, default=2e-5)
    ap.add_argument("--bert-model-name", default="mental/mental-bert-base-uncased")
    ap.add_argument("--binarize-threshold", type=float, default=10.0)
    ap.add_argument("--base-seed", type=int, default=1000, help="per-repeat torch seed = base_seed + repeat")
    ap.add_argument("--allow-cpu", action="store_true", help="override the GPU guard (NOT for official runs)")
    a = ap.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    if device != "cuda" and not a.allow_cpu:
        raise SystemExit(
            "[ABORT] Baseline-CV MUST run on GPU. The frozen baseline is a Colab GPU result; "
            "running on CPU changes a SECOND factor (device) vs Phase 9.5 and reintroduces the "
            "confound this phase exists to avoid. Use a Colab GPU runtime. "
            "(--allow-cpu exists only for non-official smoke tests.)")

    # Integrity: confirm the frozen trainer is the verified one
    tsha = sha256(T.__file__)
    if tsha != FROZEN_TRAINER_SHA:
        raise SystemExit(f"[ABORT] trainer SHA {tsha} != frozen {FROZEN_TRAINER_SHA}. Not the verified trainer.")
    print(f"[OK] frozen trainer SHA verified: {tsha}")

    os.makedirs(a.out_dir, exist_ok=True)
    manifest = json.load(open(a.manifest))
    assert manifest["n_participants"] == 188, manifest["n_participants"]

    # per-repeat seed: makes each repeat regenerable; distinct seeds across repeats
    # sample training stochasticity (protocol P-3 as clarified in 11.5).
    seed = a.base_seed + a.repeat
    torch.manual_seed(seed)
    np.random.seed(seed)

    all_records = T.read_parquet_records(a.parquet)
    by_pid = {}
    for r in all_records:
        by_pid[int(r.get("participant_id"))] = r
    tokenizer = AutoTokenizer.from_pretrained(a.bert_model_name)
    audio_dim, vision_dim = infer_dims(all_records)
    print(f"[INFO] repeat={a.repeat} seed={seed} device={device} audio_dim={audio_dim} vision_dim={vision_dim}")

    rows = []
    for fold in manifest["folds"]:
        fno = fold["fold"]
        tr_recs = [by_pid[i] for i in fold["train_ids"]]
        te_recs = [by_pid[i] for i in fold["test_ids"]]
        t0 = time.time()

        # reseed per fold so a single fold is regenerable independent of fold order
        torch.manual_seed(seed * 100 + fno)

        model = T.MultiModalModel(a.bert_model_name, audio_dim=audio_dim, vision_dim=vision_dim, device=device)
        model.to(device)
        tr_ds = T.MultiModalDataset(tr_recs, tokenizer, max_len=128)
        te_ds = T.MultiModalDataset(te_recs, tokenizer, max_len=128)

        # val_dataset=None => the internal 10% holdout (which CV replaces) is removed;
        # the model trains on the FULL train partition. Nothing else changes.
        model = T.fine_tune_supervised(model, tr_ds, epochs=a.epochs, batch_size=a.batch_size,
                                       lr=a.lr, device=device, val_dataset=None)

        te_loader = DataLoader(te_ds, batch_size=a.batch_size, shuffle=False, collate_fn=T.collate_batch)
        preds = T.run_inference(model, te_loader, device=device)

        y = true_phq(te_recs)
        yb = (y > a.binarize_threshold).astype(int)
        pred_phq = np.array([p["pred_phq"] for p in preds], float)
        p_pos = np.array([p["pred_class_probs"][1] for p in preds], float)
        pred_class = np.array([p["pred_class"] for p in preds], int)

        # persist held-out predictions (reproducibility)
        pred_path = os.path.join(a.out_dir, f"rep{a.repeat}_fold{fno}_preds.csv")
        pd.DataFrame({"participant_id": fold["test_ids"], "phq": y, "phq_bin": yb,
                      "pred_phq": pred_phq, "p_pos": p_pos, "pred_class": pred_class}).to_csv(pred_path, index=False)

        # ---- primary metrics (protocol §3) ----
        train_mean = float(np.mean(true_phq(tr_recs)))
        two_class = len(np.unique(yb)) > 1
        m = dict(
            repeat=a.repeat, fold=fno, seed=seed, n_test=len(te_recs), n_pos=int(yb.sum()),
            # regression
            MAE=mean_absolute_error(y, pred_phq), RMSE=rmse(y, pred_phq),
            pred_var=float(np.var(pred_phq)), pred_min=float(pred_phq.min()), pred_max=float(pred_phq.max()),
            MAE_mean_pred=mean_absolute_error(y, np.full(len(y), train_mean)),
            # classification (ranking = threshold-free; P/R/F1/balAcc/MCC at trainer argmax=0.5)
            ROC_AUC=roc_auc_score(yb, p_pos) if two_class else float("nan"),
            PR_AUC=average_precision_score(yb, p_pos) if two_class else float("nan"),
            balAcc=balanced_accuracy_score(yb, pred_class),
            MCC=matthews_corrcoef(yb, pred_class) if two_class else 0.0,
            Precision=precision_score(yb, pred_class, zero_division=0),
            Recall=recall_score(yb, pred_class, zero_division=0),
            F1=f1_score(yb, pred_class, zero_division=0),
            Accuracy=accuracy_score(yb, pred_class),
            seconds=round(time.time() - t0, 1),
        )
        rows.append(m)
        print(f"[fold {fno}] MAE={m['MAE']:.4f} pred_var={m['pred_var']:.4f} "
              f"ROC_AUC={m['ROC_AUC']:.4f} PR_AUC={m['PR_AUC']:.4f} balAcc={m['balAcc']:.4f} "
              f"MCC={m['MCC']:.4f} R={m['Recall']:.3f} ({m['seconds']}s)")

    out_csv = os.path.join(a.out_dir, f"rep{a.repeat}_metrics.csv")
    pd.DataFrame(rows).to_csv(out_csv, index=False)
    print(f"[DONE] repeat {a.repeat} -> {out_csv}")


if __name__ == "__main__":
    main()
