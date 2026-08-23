#!/usr/bin/env python3
"""
scripts/run_arm3_local_baseline.py

STEP 12 — ARM 3 (local only): single client, no aggregation, no DP. The
centralised upper bound for the privacy-utility comparison in
docs/IMPLEMENTATION_NOTES.md, "Step 12 findings".

Fully offline — no gRPC, no TPM, no MongoDB, no orchestrator, no DP agent.
Trains once on the Fix E2 train split (149 records), evaluates on the same
held-out 37 records ARM 1 / ARM 2 use, and reports the same metric set.

Usage:
    .venv\\Scripts\\python.exe scripts\\run_arm3_local_baseline.py --seed 101
"""

from __future__ import annotations

import argparse
import json
import statistics
import sys
from pathlib import Path

REPO_ROOT = Path(r"D:\Download D\BE PIPELINE\Capstone-")
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
sys.path.append(str(REPO_ROOT / "installer" / "runtime"))

import torch
from torch.utils.data import DataLoader
from transformers import AutoTokenizer

from agents.trainer.trainer_mentalbert_privacy import (
    MultiModalModel,
    MultiModalDataset,
    read_parquet_records,
    collate_batch,
    train_model,
    stratified_split,
    MENTALBERT_PRETRAIN,
    MULTIMODAL_MAX_LEN,
    FREEZE_TEXT_ENCODER,
    LOCAL_SAVE_DIR,
    SUPERVISED_LR,
    SUPERVISED_EPOCHS,
    PHQ_POSITIVE_THRESHOLD,
)

PARQUET_PATH = REPO_ROOT / "dataset_build" / "daic_records_multimodal_participant_only.parquet"
DEFAULT_DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


def infer_dims(records):
    audio_dim = None
    vision_dim = None
    for r in records:
        f = r.get("features") or {}
        audio = f.get("audio")
        if isinstance(audio, dict) and isinstance(audio.get("wav2vec2"), list):
            audio_dim = len(audio["wav2vec2"])
            break
    for r in records:
        f = r.get("features") or {}
        if isinstance(f, dict) and "video" in f and isinstance(f["video"], dict):
            v = f["video"]
            if v.get("densenet") and isinstance(v["densenet"], (list, tuple)):
                vision_dim = len(v["densenet"])
                break
    return audio_dim, vision_dim


def main() -> int:
    ap = argparse.ArgumentParser(description="Step 12 ARM 3 — local-only centralised baseline")
    ap.add_argument("--seed", type=int, required=True)
    ap.add_argument("--epochs", type=int, default=SUPERVISED_EPOCHS)
    ap.add_argument("--lr", type=float, default=SUPERVISED_LR)
    ap.add_argument("--batch-size", type=int, default=8)
    ap.add_argument("--device", default=DEFAULT_DEVICE)
    args = ap.parse_args()

    device = args.device
    records = read_parquet_records(str(PARQUET_PATH))
    train_records, eval_records = stratified_split(records)
    audio_dim, vision_dim = infer_dims(records)

    print(f"[info] seed={args.seed} epochs={args.epochs} lr={args.lr} device={device}")
    print(f"[info] train={len(train_records)} eval={len(eval_records)} "
          f"(eval positive={sum(1 for r in eval_records if float(r['phq_score']) >= PHQ_POSITIVE_THRESHOLD)})")

    torch.manual_seed(args.seed)
    model = MultiModalModel(MENTALBERT_PRETRAIN, audio_dim=audio_dim, vision_dim=vision_dim, device=device)
    if FREEZE_TEXT_ENCODER:
        for p in model.bert.parameters():
            p.requires_grad_(False)
    model.to(device)

    tokenizer = AutoTokenizer.from_pretrained(MENTALBERT_PRETRAIN)
    ds = MultiModalDataset(train_records, tokenizer, max_len=MULTIMODAL_MAX_LEN)
    eval_ds = MultiModalDataset(eval_records, tokenizer, max_len=MULTIMODAL_MAX_LEN)

    result = train_model(
        ds, model,
        output_dir=str(LOCAL_SAVE_DIR / "step12_arm3_scratch"),
        epochs=args.epochs, batch_size=args.batch_size, lr=args.lr, device=device,
        eval_dataset=eval_ds,
    )
    model.load_state_dict(torch.load(result["model_path"], map_location=device))
    model.eval()

    eval_loader = DataLoader(eval_ds, batch_size=args.batch_size, shuffle=False, collate_fn=collate_batch)
    y_pred_cls, y_pos_prob = [], []
    with torch.no_grad():
        for b in eval_loader:
            b = {k: (v.to(device) if isinstance(v, torch.Tensor) else v) for k, v in b.items()}
            logits, _, _ = model(b["input_ids"], b["attention_mask"],
                                 audio_vec=b.get("audio_vec"), vision_vec=b.get("video_vec"))
            probs = torch.softmax(logits, dim=1)
            y_pred_cls.extend(probs.argmax(dim=1).cpu().tolist())
            y_pos_prob.extend(probs[:, 1].cpu().tolist())

    n_pred_positive = sum(y_pred_cls)
    prob_stats = {
        "min": float(min(y_pos_prob)),
        "max": float(max(y_pos_prob)),
        "mean": float(statistics.mean(y_pos_prob)),
        "stdev": float(statistics.stdev(y_pos_prob)) if len(y_pos_prob) > 1 else 0.0,
    }

    out = {
        "seed": args.seed,
        "n_eval": len(eval_records),
        "accuracy": result["metrics"]["accuracy"],
        "precision": result["metrics"]["precision"],
        "recall": result["metrics"]["recall"],
        "f1": result["metrics"]["f1"],
        "mae": result["metrics"]["mae"],
        "predicted_positive": int(n_pred_positive),
        "predicted_negative": int(len(y_pred_cls) - n_pred_positive),
        "prob_positive_distribution": prob_stats,
        "epsilon": None,  # ARM 3: no DP applied at all — not "no privacy noise", DP is simply not in this path
    }

    print()
    print("=" * 60)
    print("ARM 3 — LOCAL-ONLY BASELINE (no aggregation, no DP)")
    print("=" * 60)
    for k, v in out.items():
        print(f"  {k}: {v}")
    print("=" * 60)
    print("[RESULT_JSON] " + json.dumps(out))
    return 0


if __name__ == "__main__":
    sys.exit(main())
