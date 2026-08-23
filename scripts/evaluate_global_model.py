#!/usr/bin/env python3
"""
scripts/evaluate_global_model.py

STEP 12 — evaluate the AGGREGATED GLOBAL MODEL (post-aggregation, post-DP)
on the Fix E2 held-out 37 records. Nothing in this project has ever done this
before Step 12 — every prior verification run measured local, pre-DP,
pre-aggregation metrics only.

Reconstructs the model as base_state[trainable_keys] + aggregated_delta,
NOT as pipeline.py's live warm-start does. See
docs/IMPLEMENTATION_NOTES.md, "Step 12 findings" section, for why:

  Defect A: the aggregator persists an averaged DELTA (after - before), never
            weights. The live client warm-start path
            (trainer_mentalbert_privacy.orchestrate(), "Phase 10: warm-start")
            loads that delta via model.load_state_dict(global_state,
            strict=False) directly onto a freshly-initialised model, as if
            the delta WERE the weights. This script does not replicate that
            bug - it does the mathematically correct reconstruction instead,
            so Step 12 measures privacy cost, not this unrelated defect.
            pipeline.py itself is not modified; the live bug stays visible.

  Defect B: no run in this project seeds the model constructor, so without
            GLOBAL_INIT_SEED, three "clients" would have three different
            random bases and "base_state + aggregated_delta" would not even
            be well-defined (whose base?). This script requires --seed to
            equal the GLOBAL_INIT_SEED value the driving script exported to
            all three clients of the round being evaluated, and reconstructs
            that shared base deterministically by re-seeding and
            re-constructing the model exactly as orchestrate() did.

Does NOT touch: gRPC, TPM, crypto, Rust, or pipeline.py's warm-start loader.
Reads the aggregated artifact directly from MongoDB GridFS (the plaintext
.pt the aggregator itself writes and uploads - see aggregator.py's
run_job(): the GridFS copy of the global model is not SecureStore-encrypted,
only the per-client uploads are).

Usage:
    .venv\\Scripts\\python.exe scripts\\evaluate_global_model.py \\
        --seed 101 --round-id 1 --db federated_multimodal
"""

from __future__ import annotations

import argparse
import io
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
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, mean_absolute_error

from agents.trainer.trainer_mentalbert_privacy import (
    MultiModalModel,
    MultiModalDataset,
    read_parquet_records,
    collate_batch,
    stratified_split,
    MENTALBERT_PRETRAIN,
    MULTIMODAL_MAX_LEN,
    FREEZE_TEXT_ENCODER,
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


def fetch_aggregated_delta(mongo_uri: str, db_name: str, round_id: int):
    """Looks up the `global_models` document by round_id and fetches its
    file_id from GridFS — the same lookup pipeline.py's own
    _download_global_model() does server-side, NOT a guessed filename.

    Note: round_id here is the orchestrator's bookkeeping round_id (the
    round this model becomes AVAILABLE for, i.e. aggregated_round + 1 —
    server.rs's own comment: "Inserts a global_models document ... for
    round N+1"). This differs from the aggregator's own GridFS filename
    (global_model_round_{aggregated_round}.pt, aggregator.py:655) — round 1's
    aggregation produces a global_models doc with round_id=2 pointing at a
    file named global_model_round_1.pt. Querying by the global_models
    document (as done here) sidesteps that naming mismatch entirely.

    Plaintext torch pickle — the aggregator uploads it unencrypted (see
    aggregator.py's run_job(), no SecureStore call on model_bytes)."""
    import gridfs
    from pymongo import MongoClient
    from bson.objectid import ObjectId

    client = MongoClient(mongo_uri)
    db = client[db_name]
    fs = gridfs.GridFS(db)

    doc = db["global_models"].find_one({"round_id": round_id})
    if doc is None:
        client.close()
        raise RuntimeError(
            f"No global_models document with round_id={round_id} in database "
            f"{db_name!r}. Has aggregation completed and been recorded yet? "
            f"(round_id here is the orchestrator's N+1 bookkeeping value, not "
            f"the round that was aggregated.)"
        )
    file_id = doc["file_id"] if isinstance(doc["file_id"], ObjectId) else ObjectId(doc["file_id"])
    raw = fs.get(file_id).read()
    client.close()

    buf = io.BytesIO(raw)
    delta = torch.load(buf, map_location="cpu", weights_only=False)
    if not isinstance(delta, dict):
        raise TypeError(f"Expected a state dict from GridFS, got {type(delta).__name__}")
    return delta


def reconstruct_absolute_state(aggregated_delta: dict, seed: int, device: str, label: str = ""):
    """Reconstructs base_state[trainable_keys] + aggregated_delta -> absolute
    weights (Defect A workaround: base + delta, never load_state_dict(delta)
    directly). Returns (model, new_state, trainable_keys, delta_l2,
    audio_dim, vision_dim). `model` already has new_state loaded
    (strict=False — bert stays at its pretrained checkpoint value, untouched).

    Used by evaluate_delta() (held-out scoring) AND by Step 14's multi-round
    driver (scripts/run_step14_multiround.py), which writes `new_state`
    itself into GridFS so the NEXT round's clients warm-start from genuine
    absolute weights through pipeline.py's completely unmodified, otherwise-
    buggy (Defect A) load_state_dict(global_state, strict=False) call —
    that call is only wrong when fed a bare delta; fed real weights, as here,
    it is exactly correct with zero code changes."""
    records = read_parquet_records(str(PARQUET_PATH))
    audio_dim, vision_dim = infer_dims(records)
    print(f"[info] audio_dim={audio_dim} vision_dim={vision_dim} device={device}")

    print(f"[info] aggregated delta{' (' + label + ')' if label else ''}: {len(aggregated_delta)} keys, "
          f"{sum(v.numel() for v in aggregated_delta.values()):,} params")
    delta_l2 = torch.sqrt(sum((v.float().norm() ** 2) for v in aggregated_delta.values())).item()
    print(f"[info] aggregated delta L2 norm = {delta_l2:.6f}")

    # ── Reconstruct the SAME shared base the 3 clients used ────────────────
    # Must mirror orchestrate() exactly: seed -> construct -> (freeze bert,
    # for an identical trainable-key set — doesn't affect eval values since
    # bert's own weights are the pretrained checkpoint either way).
    torch.manual_seed(int(seed))
    print(f"[STEP12-SEED] reconstructing base with GLOBAL_INIT_SEED={seed}")
    model = MultiModalModel(MENTALBERT_PRETRAIN, audio_dim=audio_dim, vision_dim=vision_dim, device=device)
    if FREEZE_TEXT_ENCODER:
        for p in model.bert.parameters():
            p.requires_grad_(False)

    trainable_keys = {n for n, p in model.named_parameters() if p.requires_grad}
    base_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items() if k in trainable_keys}

    delta_keys = set(aggregated_delta.keys())
    if delta_keys != trainable_keys:
        extra = sorted(delta_keys - trainable_keys)
        missing = sorted(trainable_keys - delta_keys)
        raise AssertionError(
            f"Aggregated delta key set does not match this reconstruction's trainable "
            f"key set. extra={extra[:5]} missing={missing[:5]}. Either --seed doesn't "
            f"match what the 3 clients used, or the architecture (audio_dim/vision_dim) "
            f"differs from the round that produced this aggregate."
        )

    # ── Defect A workaround: base + delta, not load_state_dict(delta) ──────
    new_state = {k: base_state[k] + aggregated_delta[k].to(base_state[k].dtype) for k in trainable_keys}
    missing, unexpected = model.load_state_dict(new_state, strict=False)
    print(f"[info] reconstruction load: {len(missing)} missing keys (expected: bert, untouched), "
          f"{len(unexpected)} unexpected keys (expected: 0)")
    assert len(unexpected) == 0, f"Unexpected keys after loading reconstructed state: {unexpected}"

    return model, new_state, trainable_keys, delta_l2, audio_dim, vision_dim


def evaluate_delta(aggregated_delta: dict, seed: int, device: str, label: str = ""):
    """Reconstructs base_state[trainable_keys] + aggregated_delta and evaluates
    on the Fix E2 held-out 37, given an ALREADY-COMPUTED aggregated delta dict
    (from GridFS via fetch_aggregated_delta(), or from any other aggregation —
    e.g. scripts/aggregate_offline.py's offline AggregatorAgent(mode=...) call
    for Step 13's mean-vs-trimmed_mean comparison). Contains no round_id /
    GridFS logic itself — that's the caller's job (see fetch_aggregated_delta()
    and aggregate_offline.py's fetch_round_updates())."""
    model, new_state, trainable_keys, delta_l2, audio_dim, vision_dim = reconstruct_absolute_state(
        aggregated_delta, seed, device, label
    )
    records = read_parquet_records(str(PARQUET_PATH))
    train_records, eval_records = stratified_split(records)
    print(f"[info] eval records = {len(eval_records)} "
          f"(positive={sum(1 for r in eval_records if float(r['phq_score']) >= PHQ_POSITIVE_THRESHOLD)}, "
          f"negative={sum(1 for r in eval_records if float(r['phq_score']) < PHQ_POSITIVE_THRESHOLD)})")

    model.to(device)
    model.eval()

    tokenizer = AutoTokenizer.from_pretrained(MENTALBERT_PRETRAIN)
    eval_ds = MultiModalDataset(eval_records, tokenizer, max_len=MULTIMODAL_MAX_LEN)
    eval_loader = DataLoader(eval_ds, batch_size=8, shuffle=False, collate_fn=collate_batch)

    y_true_cls, y_pred_cls, y_pos_prob = [], [], []
    y_true_phq, y_pred_phq = [], []
    with torch.no_grad():
        for b in eval_loader:
            b = {k: (v.to(device) if isinstance(v, torch.Tensor) else v) for k, v in b.items()}
            logits, reg_pred, _ = model(b["input_ids"], b["attention_mask"],
                                        audio_vec=b.get("audio_vec"), vision_vec=b.get("video_vec"))
            probs = torch.softmax(logits, dim=1)
            preds_cls = probs.argmax(dim=1)
            y_true_cls.extend(b["label"].cpu().tolist())
            y_pred_cls.extend(preds_cls.cpu().tolist())
            y_pos_prob.extend(probs[:, 1].cpu().tolist())
            y_true_phq.extend(b["phq"].cpu().tolist())
            y_pred_phq.extend(reg_pred.cpu().tolist())

    acc = accuracy_score(y_true_cls, y_pred_cls)
    prec, rec, f1, _ = precision_recall_fscore_support(y_true_cls, y_pred_cls, average="binary", zero_division=0)
    mae = mean_absolute_error(y_true_phq, y_pred_phq)
    n_pred_positive = sum(y_pred_cls)

    prob_stats = {
        "min": float(min(y_pos_prob)),
        "max": float(max(y_pos_prob)),
        "mean": float(statistics.mean(y_pos_prob)),
        "stdev": float(statistics.stdev(y_pos_prob)) if len(y_pos_prob) > 1 else 0.0,
    }

    result = {
        "seed": seed,
        "label": label,
        "n_eval": len(eval_records),
        "aggregated_delta_l2": delta_l2,
        "accuracy": float(acc),
        "precision": float(prec),
        "recall": float(rec),
        "f1": float(f1),
        "mae": float(mae),
        "predicted_positive": int(n_pred_positive),
        "predicted_negative": int(len(y_pred_cls) - n_pred_positive),
        "prob_positive_distribution": prob_stats,
    }

    print()
    print("=" * 60)
    print("GLOBAL MODEL — HELD-OUT EVALUATION (post-aggregation)")
    print("=" * 60)
    for k, v in result.items():
        print(f"  {k}: {v}")
    print("=" * 60)
    print("[RESULT_JSON] " + json.dumps(result))
    return result


def reconstruct_and_evaluate(seed: int, round_id: int, mongo_uri: str, db_name: str, device: str):
    """CLI path (Step 12): fetch the orchestrator's own GridFS-aggregated
    (trimmed_mean, live) delta for round_id, then evaluate_delta() it."""
    aggregated_delta = fetch_aggregated_delta(mongo_uri, db_name, round_id)
    return evaluate_delta(aggregated_delta, seed, device, label=f"round_id={round_id}")


def main() -> int:
    ap = argparse.ArgumentParser(description="Step 12 — evaluate the aggregated global model on held-out data")
    ap.add_argument("--seed", type=int, required=True, help="GLOBAL_INIT_SEED used by the 3 clients of this round")
    ap.add_argument("--round-id", type=int, default=1)
    ap.add_argument("--db", default="federated_multimodal")
    ap.add_argument("--mongo-uri", default="mongodb://localhost:27017")
    ap.add_argument("--device", default=DEFAULT_DEVICE)
    args = ap.parse_args()

    reconstruct_and_evaluate(args.seed, args.round_id, args.mongo_uri, args.db, args.device)
    return 0


if __name__ == "__main__":
    sys.exit(main())
