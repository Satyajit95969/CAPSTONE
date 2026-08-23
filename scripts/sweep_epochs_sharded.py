#!/usr/bin/env python3
"""
scripts/sweep_epochs_sharded.py

STEP 21a — OFFLINE epoch sweep for the sharded regime (docs/IMPLEMENTATION_
NOTES.md, "Step 20" section). No orchestrator, no gRPC, no MongoDB, no DP, no
encryption, no upload — same guarantee as calibrate_clip_norm.py /
sweep_lr_epochs.py.

Step 20's Run A (IID, 3 shards) measured F1 collapse to 0.0 with ~50
records/client at epochs=10 (the value Step 10a's 60-run sweep tuned for
149 records). Two plausible causes, deliberately not disentangled until now:
  (a) ~50 records with only ~15 positives is too little data to learn from,
      regardless of training length — a data-volume floor.
  (b) epochs=10 was tuned for 149 records (190 optimizer steps); 50 records
      at batch=8 gives ~7 steps/epoch, ~70 steps at epochs=10 — undertrained.

This sweeps epochs in {10, 20, 30, 50} at the CURRENT lr=1e-4, 3 seeds/cell,
training on shard 0 of 3 (IID, shard_seed=20240) — the EXACT SAME 50-record
shard Step 20's Run A client 0 used — and evaluating on the same GLOBAL
held-out 37 every cell shares with every measurement since Step 12. Also
runs the full 149-record baseline at epochs=10, 3 seeds, for like-for-like
single-client context (not the aggregated number Run A reported).

Report only. Chooses nothing, changes no default, does not modify
orchestrate()/pipeline.py, does not re-run the live 3-client pipeline.

Usage:
    .venv\\Scripts\\python.exe scripts\\sweep_epochs_sharded.py
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
    stratified_shard,
    LOCAL_SAVE_DIR,
    PHQ_POSITIVE_THRESHOLD,
)

PARQUET_PATH = REPO_ROOT / "dataset_build" / "daic_records_multimodal_participant_only.parquet"
DEFAULT_DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

SHARD_LR = 1e-4          # current default (Fix E4), unchanged, not swept
EPOCHS_GRID = [10, 20, 30, 50]
SEEDS = [1, 2, 3]         # matches Step 10a's sweep_lr_epochs.py convention
BATCH_SIZE = 8

# Step 20 Run A's exact shard-0 configuration — must reproduce size=50,
# positive=15, negative=35 (see docs/IMPLEMENTATION_NOTES.md, Step 20 table).
SHARD_N = 3
SHARD_ID = 0
SHARD_SEED = 20240

FIX_E4_RANGE = (0.625, 0.729)


def one_run(train_ds, eval_ds, tokenizer, epochs, seed, label):
    torch.manual_seed(seed)
    model = MultiModalModel(MENTALBERT_PRETRAIN, audio_dim=154, vision_dim=84, device=DEFAULT_DEVICE)
    if FREEZE_TEXT_ENCODER:
        for p in model.bert.parameters():
            p.requires_grad_(False)
    model.to(DEFAULT_DEVICE)
    base_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}

    result = train_model(
        train_ds, model,
        output_dir=str(LOCAL_SAVE_DIR / "sweep_sharded_scratch"),
        epochs=epochs, batch_size=BATCH_SIZE, lr=SHARD_LR, device=DEFAULT_DEVICE,
        eval_dataset=eval_ds,
    )
    model.load_state_dict(torch.load(result["model_path"], map_location=DEFAULT_DEVICE))
    after = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
    delta = compute_filtered_delta(base_state, after, model)
    delta_l2 = torch.sqrt(sum((v.float().norm() ** 2) for v in delta.values())).item()

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
        "label": label, "epochs": epochs, "seed": seed, "n_train": len(train_ds),
        "accuracy": m["accuracy"], "precision": m["precision"], "recall": m["recall"],
        "f1": m["f1"], "mae": m["mae"],
        "n_pred_pos": n_pred_pos, "n_eval": len(probs_all),
        "delta_l2": delta_l2,
        "degenerate": degenerate,
    }


def print_cell_table(title, rows_by_key):
    print()
    print("=" * 78)
    print(title)
    print("=" * 78)
    summary = []
    for key, runs in rows_by_key.items():
        n_degenerate = sum(1 for r in runs if r["degenerate"])
        mean_f1 = statistics.mean(r["f1"] for r in runs)
        mean_acc = statistics.mean(r["accuracy"] for r in runs)
        mean_mae = statistics.mean(r["mae"] for r in runs)
        mean_delta = statistics.mean(r["delta_l2"] for r in runs)
        in_range = FIX_E4_RANGE[0] <= mean_delta <= FIX_E4_RANGE[1]
        summary.append({
            "key": key, "n_degenerate": n_degenerate, "mean_f1": mean_f1,
            "mean_accuracy": mean_acc, "mean_mae": mean_mae, "mean_delta_l2": mean_delta,
            "in_fix_e4_range": in_range,
            "per_seed_f1": [r["f1"] for r in runs],
            "per_seed_pred_pos": [r["n_pred_pos"] for r in runs],
            "per_seed_delta_l2": [r["delta_l2"] for r in runs],
        })
    summary.sort(key=lambda x: (x["n_degenerate"], -x["mean_f1"]))
    for row in summary:
        print(f"{row['key']}  degenerate={row['n_degenerate']}/{len(SEEDS)}  "
              f"mean_f1={row['mean_f1']:.4f} (per-seed: {[f'{v:.4f}' for v in row['per_seed_f1']]})  "
              f"mean_acc={row['mean_accuracy']:.4f} mean_mae={row['mean_mae']:.4f}")
        print(f"    pred_pos per seed: {row['per_seed_pred_pos']}   "
              f"delta_l2 per seed: {[f'{v:.6f}' for v in row['per_seed_delta_l2']]}  "
              f"mean={row['mean_delta_l2']:.6f}  in_Fix_E4_range[{FIX_E4_RANGE[0]},{FIX_E4_RANGE[1]}]={row['in_fix_e4_range']}")
    return summary


def main() -> int:
    print("=" * 78)
    print("STEP 21a — EPOCH SWEEP, SHARDED REGIME (offline, no gRPC/Mongo/DP/upload)")
    print("=" * 78)
    print(f"epochs grid : {EPOCHS_GRID}   lr (fixed) : {SHARD_LR}   seeds : {SEEDS}")
    print(f"shard       : {SHARD_ID}/{SHARD_N}, shard_seed={SHARD_SEED} (Step 20 Run A's exact shard 0)")
    print()

    records = read_parquet_records(str(PARQUET_PATH))
    train_records, eval_records = stratified_split(records)  # fixed, seed=42 — the GLOBAL held-out 37
    shard_records = stratified_shard(train_records, n_shards=SHARD_N, shard_id=SHARD_ID, seed=SHARD_SEED)

    shard_pos = sum(1 for r in shard_records if float(r["phq_score"]) >= PHQ_POSITIVE_THRESHOLD)
    print(f"[info] shard: n={len(shard_records)} positive={shard_pos} negative={len(shard_records) - shard_pos} "
          f"(Step 20 Run A client 0 reported n=50 positive=15 negative=35 — {'MATCH' if len(shard_records) == 50 and shard_pos == 15 else 'MISMATCH, investigate before trusting results below'})")
    print(f"[info] full baseline train set: n={len(train_records)}")
    print(f"[info] eval (global, shared, unchanged): n={len(eval_records)}")

    tokenizer = AutoTokenizer.from_pretrained(MENTALBERT_PRETRAIN)
    shard_ds = MultiModalDataset(shard_records, tokenizer, max_len=MULTIMODAL_MAX_LEN)
    full_ds = MultiModalDataset(train_records, tokenizer, max_len=MULTIMODAL_MAX_LEN)
    eval_ds = MultiModalDataset(eval_records, tokenizer, max_len=MULTIMODAL_MAX_LEN)

    t_sweep0 = time.time()

    # ---- shard sweep: epochs in {10,20,30,50}, 3 seeds each ----
    shard_cells = {}
    total_cells = len(EPOCHS_GRID)
    for ci, epochs in enumerate(EPOCHS_GRID, start=1):
        runs = []
        for seed in SEEDS:
            t0 = time.time()
            r = one_run(shard_ds, eval_ds, tokenizer, epochs, seed, label=f"shard(n=50) epochs={epochs}")
            elapsed = time.time() - t0
            runs.append(r)
            print(f"[shard cell {ci}/{total_cells}] epochs={epochs} seed={seed}  "
                  f"f1={r['f1']:.4f} acc={r['accuracy']:.4f} pred_pos={r['n_pred_pos']}/{r['n_eval']} "
                  f"degenerate={r['degenerate']} delta_l2={r['delta_l2']:.6f}  ({elapsed:.1f}s)")
        shard_cells[f"epochs={epochs}"] = runs

    # ---- full-149 baseline: epochs=10, 3 seeds (context, not aggregated) ----
    baseline_runs = []
    for seed in SEEDS:
        t0 = time.time()
        r = one_run(full_ds, eval_ds, tokenizer, 10, seed, label="full(n=149) epochs=10")
        elapsed = time.time() - t0
        baseline_runs.append(r)
        print(f"[baseline] full n=149 epochs=10 seed={seed}  "
              f"f1={r['f1']:.4f} acc={r['accuracy']:.4f} pred_pos={r['n_pred_pos']}/{r['n_eval']} "
              f"degenerate={r['degenerate']} delta_l2={r['delta_l2']:.6f}  ({elapsed:.1f}s)")

    total_elapsed = time.time() - t_sweep0
    print(f"\n[info] sweep complete in {total_elapsed/60:.1f} minutes")

    shard_summary = print_cell_table("SHARD (n=50) SWEEP — ranked by (fewest degenerate, then mean F1)", shard_cells)
    print_cell_table("FULL (n=149) BASELINE, epochs=10 — single-client context, NOT the aggregated Step 14 ARM 1 number", {"epochs=10 (full 149)": baseline_runs})

    print()
    print("=" * 78)
    print("READ (report only — nothing chosen, no default changed)")
    print("=" * 78)
    any_nondegenerate = any(row["n_degenerate"] < len(SEEDS) for row in shard_summary)
    any_in_range = any(row["in_fix_e4_range"] for row in shard_summary)
    print(f"Any shard cell with < {len(SEEDS)} degenerate seeds: {any_nondegenerate}")
    print(f"Any shard cell with mean delta_l2 inside Fix E4 range {FIX_E4_RANGE}: {any_in_range}")
    print("See conversation / docs/IMPLEMENTATION_NOTES.md Step 21 section for the data-volume-floor")
    print("vs. training-length read on these numbers.")
    print("=" * 78)

    return 0


if __name__ == "__main__":
    sys.exit(main())
