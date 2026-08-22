#!/usr/bin/env python3
"""
scripts/sweep_lr_epochs.py

OFFLINE measurement only, same guarantee as calibrate_clip_norm.py: no
orchestrator, no gRPC, no MongoDB, no receipts, no DP, no encryption, no
upload. Extends that script's approach to sweep learning rate x epochs for
Fix E4, using the CURRENT Fix E3 configuration (class-weighted CE + normalized
regression loss) as the fixed baseline.

Sweeps lr x epochs, 3 fresh-init seeds per cell, on the SAME stratified
train/eval split every time (EVAL_SPLIT_SEED=42, unchanged) - only the model's
random initialization (and the resulting training trajectory) varies across
seeds, isolating lr/epoch effects from split effects.

For each cell, reports (averaged over 3 seeds, plus each seed's own values):
    held-out F1, precision, recall, accuracy, MAE
    predicted positive count / 37
    eval probability distribution (min/mean/max)
    degenerate-seed count (all-one-class prediction on the eval set)
    delta L2 norm (pre-clamp, pre-DP - same measurement as calibrate_clip_norm.py)

Ranks cells by (fewest degenerate seeds, then mean F1 descending). Reports
only - does not modify orchestrate()/pipeline.py or pick a value to wire in.

Usage:
    .venv\\Scripts\\python.exe scripts\\sweep_lr_epochs.py
"""

from __future__ import annotations

import statistics
import sys
import time
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
    train_model,
    compute_filtered_delta,
    collate_batch,
    MENTALBERT_PRETRAIN,
    MULTIMODAL_MAX_LEN,
    FREEZE_TEXT_ENCODER,
    stratified_split,
    LOCAL_SAVE_DIR,
    PHQ_POSITIVE_THRESHOLD,
)

PARQUET_PATH = REPO_ROOT / "dataset_build" / "daic_records_multimodal_participant_only.parquet"
DEFAULT_DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

LR_GRID = [2e-5, 1e-4, 5e-4, 1e-3, 5e-3]
EPOCHS_GRID = [1, 3, 5, 10]
SEEDS = [1, 2, 3]
BATCH_SIZE = 8


def one_run(train_ds, eval_ds, tokenizer, lr, epochs, seed):
    torch.manual_seed(seed)
    model = MultiModalModel(MENTALBERT_PRETRAIN, audio_dim=154, vision_dim=84, device=DEFAULT_DEVICE)
    if FREEZE_TEXT_ENCODER:
        for p in model.bert.parameters():
            p.requires_grad_(False)
    model.to(DEFAULT_DEVICE)
    base_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}

    result = train_model(
        train_ds, model,
        output_dir=str(LOCAL_SAVE_DIR / "sweep_scratch"),
        epochs=epochs, batch_size=BATCH_SIZE, lr=lr, device=DEFAULT_DEVICE,
        eval_dataset=eval_ds,
    )
    model.load_state_dict(torch.load(result["model_path"], map_location=DEFAULT_DEVICE))
    after = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
    delta = compute_filtered_delta(base_state, after, model)
    delta_l2 = torch.sqrt(sum((v.float().norm() ** 2) for v in delta.values())).item()

    # Raw eval-set probabilities (train_model()'s own metrics don't expose these)
    eval_loader = DataLoader(eval_ds, batch_size=BATCH_SIZE, shuffle=False, collate_fn=collate_batch)
    model.eval()
    probs_all = []
    with torch.no_grad():
        for b in eval_loader:
            b = {k: (v.to(DEFAULT_DEVICE) if isinstance(v, torch.Tensor) else v) for k, v in b.items()}
            logits, _, _ = model(b["input_ids"], b["attention_mask"],
                                  audio_vec=b.get("audio_vec"), vision_vec=b.get("video_vec"))
            probs_all.extend(torch.softmax(logits, dim=1)[:, 1].cpu().tolist())

    n_pred_pos = sum(1 for p in probs_all if p >= 0.5)
    degenerate = n_pred_pos == 0 or n_pred_pos == len(probs_all)

    m = result["metrics"]
    return {
        "lr": lr, "epochs": epochs, "seed": seed,
        "accuracy": m["accuracy"], "precision": m["precision"], "recall": m["recall"],
        "f1": m["f1"], "mae": m["mae"],
        "n_pred_pos": n_pred_pos, "n_eval": len(probs_all),
        "prob_min": min(probs_all), "prob_mean": statistics.mean(probs_all), "prob_max": max(probs_all),
        "delta_l2": delta_l2,
        "degenerate": degenerate,
    }


def main() -> int:
    print("=" * 70)
    print("LR x EPOCHS SWEEP (offline, Fix E4 investigation)")
    print("=" * 70)
    print(f"lr grid     : {LR_GRID}")
    print(f"epochs grid : {EPOCHS_GRID}")
    print(f"seeds       : {SEEDS}")
    print(f"total cells : {len(LR_GRID) * len(EPOCHS_GRID)}   total runs: {len(LR_GRID) * len(EPOCHS_GRID) * len(SEEDS)}")
    print("No DP, no clamp, no encryption, no gRPC, no MongoDB in this script.")
    print()

    records = read_parquet_records(str(PARQUET_PATH))
    train_records, eval_records = stratified_split(records)  # fixed split, seed=42, unchanged across the whole sweep
    tokenizer = AutoTokenizer.from_pretrained(MENTALBERT_PRETRAIN)
    train_ds = MultiModalDataset(train_records, tokenizer, max_len=MULTIMODAL_MAX_LEN)
    eval_ds = MultiModalDataset(eval_records, tokenizer, max_len=MULTIMODAL_MAX_LEN)
    print(f"[info] train={len(train_records)} eval={len(eval_records)} (fixed split, same for every cell)")
    print()

    all_results = []
    t_sweep0 = time.time()
    cell_idx = 0
    total_cells = len(LR_GRID) * len(EPOCHS_GRID)
    for lr in LR_GRID:
        for epochs in EPOCHS_GRID:
            cell_idx += 1
            cell_results = []
            for seed in SEEDS:
                t0 = time.time()
                r = one_run(train_ds, eval_ds, tokenizer, lr, epochs, seed)
                elapsed = time.time() - t0
                cell_results.append(r)
                print(f"[cell {cell_idx}/{total_cells}] lr={lr:g} epochs={epochs} seed={seed}  "
                      f"f1={r['f1']:.4f} acc={r['accuracy']:.4f} pred_pos={r['n_pred_pos']}/{r['n_eval']} "
                      f"degenerate={r['degenerate']} delta_l2={r['delta_l2']:.6f}  ({elapsed:.1f}s)")
            all_results.extend(cell_results)

    total_elapsed = time.time() - t_sweep0
    print()
    print(f"[info] sweep complete in {total_elapsed/60:.1f} minutes")
    print()

    # ---- aggregate per cell ----
    cells = {}
    for r in all_results:
        key = (r["lr"], r["epochs"])
        cells.setdefault(key, []).append(r)

    print("=" * 70)
    print("PER-CELL SUMMARY (averaged over 3 seeds)")
    print("=" * 70)
    summary_rows = []
    for (lr, epochs), runs in cells.items():
        n_degenerate = sum(1 for r in runs if r["degenerate"])
        mean_f1 = statistics.mean(r["f1"] for r in runs)
        mean_prec = statistics.mean(r["precision"] for r in runs)
        mean_rec = statistics.mean(r["recall"] for r in runs)
        mean_acc = statistics.mean(r["accuracy"] for r in runs)
        mean_mae = statistics.mean(r["mae"] for r in runs)
        mean_delta = statistics.mean(r["delta_l2"] for r in runs)
        min_prob = min(r["prob_min"] for r in runs)
        max_prob = max(r["prob_max"] for r in runs)
        mean_prob = statistics.mean(r["prob_mean"] for r in runs)
        summary_rows.append({
            "lr": lr, "epochs": epochs, "n_degenerate": n_degenerate,
            "mean_f1": mean_f1, "mean_precision": mean_prec, "mean_recall": mean_rec,
            "mean_accuracy": mean_acc, "mean_mae": mean_mae, "mean_delta_l2": mean_delta,
            "prob_range": (min_prob, mean_prob, max_prob),
            "per_seed_f1": [r["f1"] for r in runs],
            "per_seed_pred_pos": [r["n_pred_pos"] for r in runs],
            "per_seed_delta_l2": [r["delta_l2"] for r in runs],
        })

    summary_rows.sort(key=lambda x: (x["n_degenerate"], -x["mean_f1"]))

    for row in summary_rows:
        print(f"lr={row['lr']:g} epochs={row['epochs']:<3d} "
              f"degenerate={row['n_degenerate']}/3  "
              f"mean_f1={row['mean_f1']:.4f} (per-seed: {[f'{v:.4f}' for v in row['per_seed_f1']]})  "
              f"mean_prec={row['mean_precision']:.4f} mean_rec={row['mean_recall']:.4f} "
              f"mean_acc={row['mean_accuracy']:.4f} mean_mae={row['mean_mae']:.4f}")
        print(f"    pred_pos/37 per seed: {row['per_seed_pred_pos']}   "
              f"prob_range=(min={row['prob_range'][0]:.4f}, mean={row['prob_range'][1]:.4f}, max={row['prob_range'][2]:.4f})")
        print(f"    delta_l2 per seed: {[f'{v:.6f}' for v in row['per_seed_delta_l2']]}  mean={row['mean_delta_l2']:.6f}")

    print()
    print("=" * 70)
    print("Ranked by (fewest degenerate seeds, then mean F1 descending) - see above, best first.")
    print("No cell has been selected or wired into the pipeline. Report only.")
    print("=" * 70)

    return 0


if __name__ == "__main__":
    sys.exit(main())
