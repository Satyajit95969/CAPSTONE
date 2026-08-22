#!/usr/bin/env python3
"""
multimodal_smoke_test.py

Standalone, read-only smoke test for the multimodal (text+audio+video)
federated learning path. Does NOT touch MongoDB, the orchestrator, TPM,
mTLS, or any network. Reuses the existing MultiModalModel/MultiModalDataset/
collate_batch/read_parquet_records exactly as-is — no trainer code changes.

Run:
    .venv\\Scripts\\python.exe multimodal_smoke_test.py
"""
import sys
import time
import tracemalloc
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
sys.path.append(str(REPO_ROOT / "installer" / "runtime"))

PARQUET_PATH = REPO_ROOT / "dataset_build" / "daic_records_multimodal.parquet"


def main():
    tracemalloc.start()

    from agents.trainer.trainer_mentalbert_privacy import (
        read_parquet_records, MultiModalDataset, MultiModalModel, collate_batch,
    )
    from transformers import AutoTokenizer
    from torch.utils.data import DataLoader

    print("=" * 60)
    print("PHASE 1 — LOAD REAL PARQUET, VERIFY TEXT/AUDIO/VIDEO")
    print("=" * 60)
    t0 = time.time()
    records = read_parquet_records(str(PARQUET_PATH))
    print(f"Source                : {PARQUET_PATH}")
    print(f"Total records loaded  : {len(records)}")
    r0 = records[0]
    print(f"Sample participant_id : {r0.get('participant_id')}")
    print(f"Sample text (60 chars): {r0.get('text','')[:60]!r}")
    audio_vec = r0.get("features", {}).get("audio", {}).get("wav2vec2")
    video_vec = r0.get("features", {}).get("video", {}).get("densenet")
    print(f"Sample audio dim      : {len(audio_vec) if audio_vec else 0}")
    print(f"Sample video dim      : {len(video_vec) if video_vec else 0}")
    print(f"Sample phq_score      : {r0.get('phq_score')}")
    print(f"Load time             : {time.time()-t0:.3f} sec")
    print("[OK] Phase 1 complete")
    print()

    print("=" * 60)
    print("PHASE 2 — INSTANTIATE MULTIMODAL MODEL, PROVE ENCODERS EXIST")
    print("=" * 60)
    N = min(3, len(records))
    subset = records[:N]
    audio_dim = len(audio_vec) if audio_vec else None
    video_dim = len(video_vec) if video_vec else None
    print(f"Rows used for this smoke test : {N} (RAM-safe subset, not all {len(records)})")
    print(f"[MULTIMODAL] audio dimension = {audio_dim}")
    print(f"[MULTIMODAL] video dimension = {video_dim}")

    tokenizer = AutoTokenizer.from_pretrained(
        str(Path.home() / ".federated" / "models" / "mentalbert")
    )
    ds = MultiModalDataset(subset, tokenizer)
    model = MultiModalModel(
        str(Path.home() / ".federated" / "models" / "mentalbert"),
        audio_dim=audio_dim,
        vision_dim=video_dim,
        device="cpu",
    )
    print(f"[MULTIMODAL] audio encoder = {'ACTIVE' if model.has_audio else 'INACTIVE'}")
    print(f"[MULTIMODAL] video encoder = {'ACTIVE' if model.has_vision else 'INACTIVE'}")
    print(f"[MULTIMODAL] fusion input dimension = {model.fusion.fc1.in_features}")
    total_params = sum(p.numel() for p in model.parameters())
    print(f"[MULTIMODAL] total parameters = {total_params:,}")
    assert model.has_audio, "audio_encoder must be ACTIVE for real audio_dim"
    assert model.has_vision, "vision_encoder must be ACTIVE for real vision_dim"
    print("[OK] Phase 2 complete — both encoders confirmed ACTIVE")
    print()

    print("=" * 60)
    print("PHASE 3 — LOCAL SMOKE TEST, REAL TENSOR SHAPES")
    print("=" * 60)
    loader = DataLoader(ds, batch_size=N, shuffle=False, collate_fn=collate_batch)
    batch = next(iter(loader))
    print(f"[MULTIMODAL] text batch shape  = {tuple(batch['input_ids'].shape)}")
    print(f"[MULTIMODAL] audio batch shape = {tuple(batch['audio_vec'].shape) if batch['audio_vec'] is not None else None}")
    print(f"[MULTIMODAL] video batch shape = {tuple(batch['video_vec'].shape) if batch['video_vec'] is not None else None}")

    model.eval()
    import torch
    with torch.no_grad():
        logits, reg_pred, _ = model(
            batch["input_ids"], batch["attention_mask"],
            audio_vec=batch["audio_vec"], vision_vec=batch["video_vec"],
        )
    print(f"Forward pass logits shape      : {tuple(logits.shape)}")
    print(f"Forward pass reg_pred shape    : {tuple(reg_pred.shape)}")
    print("[OK] Phase 3 complete — forward pass ran with real audio+video tensors")
    print()

    current, peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()
    print("=" * 60)
    print(f"Python-allocated memory (tracemalloc) — current: {current/1e6:.1f} MB, peak: {peak/1e6:.1f} MB")
    print("(This tracks Python-level allocations only, not process RSS — see the")
    print(" process-level measurement in Phase 8 of the live E2E run.)")
    print("=" * 60)


if __name__ == "__main__":
    main()
