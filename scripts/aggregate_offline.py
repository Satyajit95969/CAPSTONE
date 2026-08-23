#!/usr/bin/env python3
"""
scripts/aggregate_offline.py

STEP 13 — offline aggregation-mode comparison workaround (docs/IMPLEMENTATION_
NOTES.md, "Step 13" section). The Rust orchestrator hardcodes
"mode": "trimmed_mean" (server.rs:1499) with no env/config/request override —
there is no way to select mean aggregation for a live round without editing
Rust source, which is out of scope here.

Workaround: let the live pipeline run completely untouched for the 3 client
submissions (real gRPC/TPM/mTLS/DP, real GridFS uploads, real epsilon — the
orchestrator's own automatic trimmed_mean aggregation still fires as always
and is simply not used by this script). Separately, this script fetches the
SAME 3 clients' encrypted uploads directly from `model_updates` (by
round_id) and calls the *unmodified* AggregatorAgent class
(server/aggregator_agent/aggregator.py — not edited, not touched) with
whatever `mode` is requested. Same decryption code, same class, just a
different constructor argument and a different caller than the Rust
orchestrator's subprocess invocation.

This lets multiple aggregation modes be evaluated on IDENTICAL noisy data
from a single live round — isolating the aggregator's own contribution from
run-to-run training/noise variance (this is what lets Step 13b's mean vs.
trimmed_mean comparison, both at noise_multiplier=1.0, share one live round
instead of needing two).

Usage (aggregate only):
    .venv\\Scripts\\python.exe scripts\\aggregate_offline.py \\
        --round-id 1 --mode mean

Usage (aggregate AND evaluate in one step):
    .venv\\Scripts\\python.exe scripts\\aggregate_offline.py \\
        --round-id 1 --mode mean --seed 303
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPO_ROOT = Path(r"D:\Download D\BE PIPELINE\Capstone-")
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
sys.path.append(str(REPO_ROOT / "installer" / "runtime"))

from server.aggregator_agent.aggregator import AggregatorAgent

DEFAULT_DEVICE = None  # resolved lazily to avoid importing torch unless evaluating


def fetch_round_updates(mongo_uri: str, db_name: str, round_id: int):
    """Reads the 3 client uploads for round_id straight from `model_updates`
    (device_id, file_id, ...) — the same GridFS ObjectIds the orchestrator's
    own run_aggregation() would have passed to the aggregator, just read here
    directly instead of via the Rust subprocess job."""
    from pymongo import MongoClient

    client = MongoClient(mongo_uri)
    db = client[db_name]
    docs = list(db["model_updates"].find({"round_id": round_id, "verified": True}))
    client.close()

    if not docs:
        raise RuntimeError(f"No verified model_updates for round_id={round_id} in {db_name!r}")

    updates = [
        {"gridfs_id": str(d["file_id"]), "scheme": "AES-GCM-SecureStore", "enc_uri": None, "nonce": None}
        for d in docs
    ]
    print(f"[info] fetched {len(updates)} verified update(s) for round_id={round_id}: "
          f"{[str(d['_id']) for d in docs]}")
    return updates


def aggregate_offline(mongo_uri: str, db_name: str, round_id: int, mode: str, trim_ratio: float = 0.1):
    updates = fetch_round_updates(mongo_uri, db_name, round_id)
    agent = AggregatorAgent(mode=mode, trim_ratio=trim_ratio)
    aggregated = agent.aggregate_updates(updates)
    return aggregated


def main() -> int:
    ap = argparse.ArgumentParser(description="Step 13 — offline aggregation-mode comparison")
    ap.add_argument("--round-id", type=int, required=True)
    ap.add_argument("--mode", choices=["mean", "trimmed_mean", "median", "coordinate_median"], required=True)
    ap.add_argument("--trim-ratio", type=float, default=0.1)
    ap.add_argument("--db", default="federated_multimodal")
    ap.add_argument("--mongo-uri", default="mongodb://localhost:27017")
    ap.add_argument("--seed", type=int, default=None, help="if given, also evaluate on held-out data")
    ap.add_argument("--device", default=None)
    args = ap.parse_args()

    aggregated = aggregate_offline(args.mongo_uri, args.db, args.round_id, args.mode, args.trim_ratio)

    if args.seed is None:
        n_params = sum(v.numel() for v in aggregated.values())
        print(f"[info] aggregated (mode={args.mode}): {len(aggregated)} keys, {n_params:,} params")
        return 0

    import torch
    from evaluate_global_model import evaluate_delta
    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    evaluate_delta(aggregated, args.seed, device, label=f"round_id={args.round_id} mode={args.mode}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
