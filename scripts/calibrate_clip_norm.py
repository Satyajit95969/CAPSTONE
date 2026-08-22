#!/usr/bin/env python3
"""
scripts/calibrate_clip_norm.py

OFFLINE measurement only. Runs local training N times against the current
Fix A (participant-only parquet, max_len=512) + Fix B (frozen text encoder,
281,254 trainable params) configuration, and records the RAW delta L2 norm
each time - BEFORE apply_safety_to_delta() and BEFORE DPAgent ever sees it.

Purpose: measure the true sensitivity of this training procedure before
anyone picks a clip_norm to calibrate against it. Does NOT recommend or write
a clip_norm into any config - that is a separate, later step.

Does NOT touch:
  - DPAgent / dp_agent.py (no clipping, no noise)
  - EncryptionAgent
  - gRPC / the orchestrator / any network call
  - MongoDB
  - the frozen dataset_build/daic_records_multimodal.parquet or its .bak
  - the participant-only parquet (read-only)

Mirrors the real trainer's supervised path exactly:
    read_parquet_records() -> infer audio/vision dims -> MultiModalModel
    -> freeze bert (Fix B, if FREEZE_TEXT_ENCODER) -> train_model()
    -> compute_filtered_delta() -> raw L2 norm

Warm-start is intentionally OFF (global_model_path=None) in every run: this
mirrors current live conditions (round 1, no global model yet - see the
Fix B verification run's "Global Model: SKIP"). If warm-start is exercised
in a later round from a clean (non-poisoned) global model, delta sensitivity
could shift and this measurement would need to be re-run - noted in the
report, not solved here.

Usage:
    .venv\\Scripts\\python.exe scripts\\calibrate_clip_norm.py [--runs N] [--epochs E]

Runs entirely from the venv with no orchestrator or mongod required.
"""

from __future__ import annotations

import argparse
import statistics
import sys
import time
from pathlib import Path

REPO_ROOT = Path(r"D:\Download D\BE PIPELINE\Capstone-")
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
sys.path.append(str(REPO_ROOT / "installer" / "runtime"))

import torch
from transformers import AutoTokenizer

from agents.trainer.trainer_mentalbert_privacy import (
    MultiModalModel,
    MultiModalDataset,
    read_parquet_records,
    train_model,
    compute_filtered_delta,
    stratified_split,
    MENTALBERT_PRETRAIN,
    MULTIMODAL_MAX_LEN,
    FREEZE_TEXT_ENCODER,
    LOCAL_SAVE_DIR,
    SUPERVISED_LR,
    SUPERVISED_EPOCHS,
)

# Same parquet Fix A points the live pipeline at (runtime/pipeline.py's
# _MULTIMODAL_PARQUET). Hardcoded here rather than imported to avoid pulling
# in pipeline.py's gRPC/runtime dependencies for what is meant to be a
# standalone, offline script.
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


def one_calibration_run(train_records, eval_records, tokenizer, audio_dim, vision_dim, epochs, batch_size, lr, run_idx):
    """Replicates orchestrate()'s supervised path up to the raw delta L2 norm -
    nothing after it (no clamp, no DP, no encryption, no upload).

    Fix E4: trains on the TRAIN split only (Fix E2's stratified_split), not
    the full 186 records - the live pipeline hasn't trained on all 186 since
    Fix E2 landed, and calibrating against a different training-set size than
    what's actually running would measure the wrong thing."""
    model = MultiModalModel(MENTALBERT_PRETRAIN, audio_dim=audio_dim, vision_dim=vision_dim, device=DEFAULT_DEVICE)

    # Fix B: identical freeze logic to orchestrate() - unconditional, before
    # base_state capture, no warm-start (global_model_path=None every run).
    if FREEZE_TEXT_ENCODER:
        for p in model.bert.parameters():
            p.requires_grad_(False)

    model.to(DEFAULT_DEVICE)
    base_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}

    ds = MultiModalDataset(train_records, tokenizer, max_len=MULTIMODAL_MAX_LEN)
    eval_ds = MultiModalDataset(eval_records, tokenizer, max_len=MULTIMODAL_MAX_LEN)
    result = train_model(
        ds, model,
        output_dir=str(LOCAL_SAVE_DIR / "calibration_scratch"),
        epochs=epochs, batch_size=batch_size, lr=lr, device=DEFAULT_DEVICE,
        eval_dataset=eval_ds,
    )
    model.load_state_dict(torch.load(result["model_path"], map_location=DEFAULT_DEVICE))
    after = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}

    delta = compute_filtered_delta(base_state, after, model)
    delta_l2 = torch.sqrt(sum((v.float().norm() ** 2) for v in delta.values())).item()

    print(f"[run {run_idx}] delta L2 norm (pre-clamp, pre-DP) = {delta_l2:.6f}  "
          f"(final_avg_loss={result.get('final_avg_loss')})")
    return delta_l2


def main() -> int:
    ap = argparse.ArgumentParser(description="Offline delta-L2-norm calibration for clip_norm (measurement only)")
    ap.add_argument("--runs", type=int, default=10, help="number of independent local training runs (default 10)")
    ap.add_argument("--epochs", type=int, default=SUPERVISED_EPOCHS, help="epochs per run, matches orchestrate()/SUPERVISED_EPOCHS default")
    ap.add_argument("--batch-size", type=int, default=8, help="matches orchestrate() default (8)")
    ap.add_argument("--lr", type=float, default=SUPERVISED_LR, help="matches orchestrate()/SUPERVISED_LR default")
    args = ap.parse_args()

    print("=" * 60)
    print("CLIP-NORM CALIBRATION (offline, measurement only)")
    print("=" * 60)
    print(f"Parquet          : {PARQUET_PATH}")
    print(f"FREEZE_TEXT_ENCODER = {FREEZE_TEXT_ENCODER}")
    print(f"MULTIMODAL_MAX_LEN  = {MULTIMODAL_MAX_LEN}")
    print(f"Device           : {DEFAULT_DEVICE}")
    print(f"Runs             : {args.runs}")
    print(f"Epochs/run       : {args.epochs}")
    print("No DP, no clamp, no encryption, no gRPC, no MongoDB in this script.")
    print()

    if not PARQUET_PATH.exists():
        print(f"[FATAL] parquet not found: {PARQUET_PATH}")
        return 2

    records = read_parquet_records(str(PARQUET_PATH))
    print(f"[info] loaded {len(records)} records")
    # Fix E4: calibrate against what actually trains live - the Fix E2 train
    # split (149 records), not the full 186. Fixed split (seed=42), same as
    # every live run; only the model init varies run to run below.
    train_records, eval_records = stratified_split(records)
    print(f"[info] train={len(train_records)} eval={len(eval_records)} (Fix E2 stratified split, seed=42)")
    audio_dim, vision_dim = infer_dims(records)
    print(f"[info] inferred audio_dim={audio_dim}, vision_dim={vision_dim}")

    tokenizer = AutoTokenizer.from_pretrained(MENTALBERT_PRETRAIN)

    norms = []
    for i in range(1, args.runs + 1):
        t0 = time.time()
        norm = one_calibration_run(train_records, eval_records, tokenizer, audio_dim, vision_dim,
                                    args.epochs, args.batch_size, args.lr, i)
        norms.append(norm)
        print(f"         ({time.time() - t0:.1f}s)")

    norms_sorted = sorted(norms)
    n = len(norms_sorted)

    def pct(p):
        idx = min(n - 1, int(round(p * (n - 1))))
        return norms_sorted[idx]

    print()
    print("=" * 60)
    print("DELTA L2 NORM DISTRIBUTION (pre-clamp, pre-DP)")
    print("=" * 60)
    print(f"  n      = {n}")
    print(f"  min    = {norms_sorted[0]:.6f}")
    print(f"  median = {statistics.median(norms_sorted):.6f}")
    print(f"  p90    = {pct(0.90):.6f}")
    print(f"  max    = {norms_sorted[-1]:.6f}")
    print(f"  mean   = {statistics.mean(norms_sorted):.6f}")
    if n > 1:
        print(f"  stdev  = {statistics.stdev(norms_sorted):.6f}")
    print()
    print("Current dp_agent.py clip_norm default: 1.0")
    print(f"Ratio (current clip_norm / measured max)    = {1.0 / norms_sorted[-1]:.2f}x too loose")
    print(f"Ratio (current clip_norm / measured median) = {1.0 / statistics.median(norms_sorted):.2f}x too loose")
    print("=" * 60)
    print("No clip_norm has been written anywhere. Measurement only.")

    return 0


if __name__ == "__main__":
    sys.exit(main())
