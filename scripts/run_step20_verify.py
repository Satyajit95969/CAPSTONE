#!/usr/bin/env python3
"""
scripts/run_step20_verify.py

STEP 20c — verifies Step 20b's stratified sharding with ONE real 3-client
round (not multi-round — Step 20 is about per-client data partitioning, not
the round trajectory Steps 14/16-18 already covered). Each of the 3 clients
gets a DIFFERENT CLIENT_SHARD_ID (0/1/2) but the SAME CLIENT_N_SHARDS,
CLIENT_SHARD_SEED, and (if --mode noniid) CLIENT_NONIID_ALPHA — so all three
independently compute the identical full partition and each returns its own
disjoint slice (see stratified_shard()'s docstring for why this needs no
cross-client coordination).

Real gRPC/TPM/mTLS/DP pipeline, unmodified — same offline mean-aggregation +
evaluation pattern as Steps 13/14/16-19 (aggregate_offline.py,
evaluate_global_model.py). No Rust/security touched.

Usage:
    .venv\\Scripts\\python.exe scripts\\run_step20_verify.py \\
        --mode iid --orch-log C:\\path\\to\\orch.log
    .venv\\Scripts\\python.exe scripts\\run_step20_verify.py \\
        --mode noniid --alpha 0.5 --orch-log C:\\path\\to\\orch.log
"""

from __future__ import annotations

import argparse
import json
import os
import re
import subprocess
import sys
import time
from pathlib import Path

REPO_ROOT = Path(r"D:\Download D\BE PIPELINE\Capstone-")
PYTHON = str(REPO_ROOT / ".venv" / "Scripts" / "python.exe")
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(REPO_ROOT / "scripts"))
sys.path.insert(0, str(REPO_ROOT / "installer" / "runtime"))

DELTA_L2_RE = re.compile(r"Delta L2 norm before clamp\s*:\s*([0-9.]+)")
FIX_E4_RANGE = (0.625, 0.729)  # scripts/calibrate_clip_norm.py, N=30, post-Fix-E4


def run(cmd, env=None, stdin_path=None, timeout=180, display_tail=3000):
    print(f"[run] {' '.join(cmd)}")
    run_env = dict(env) if env is not None else os.environ.copy()
    run_env["PYTHONIOENCODING"] = "utf-8"
    stdin_fh = open(stdin_path, "r") if stdin_path else None
    try:
        proc = subprocess.run(
            cmd, cwd=str(REPO_ROOT), env=run_env, stdin=stdin_fh,
            capture_output=True, text=True, encoding="utf-8", timeout=timeout,
        )
    finally:
        if stdin_fh:
            stdin_fh.close()
    sys.stdout.write(proc.stdout[-display_tail:])
    if proc.returncode != 0:
        sys.stderr.write(proc.stderr[-display_tail:])
        raise RuntimeError(f"Command failed (exit {proc.returncode}): {' '.join(cmd)}")
    return proc.stdout  # FULL, untruncated — for parsing, not just display


def ensure_blank_stdin(path: Path, n_lines: int = 260):
    if not path.exists():
        path.write_text("\n" * n_lines)


def wait_for_global_model(round_id: int, db_name: str, timeout: float = 90.0):
    import pymongo
    client = pymongo.MongoClient("mongodb://localhost:27017")
    db = client[db_name]
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if db["global_models"].find_one({"round_id": round_id}) is not None:
            client.close()
            return
        time.sleep(1.0)
    client.close()
    raise TimeoutError(f"No global_models document for round {round_id} after {timeout}s")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", choices=["iid", "noniid"], required=True)
    ap.add_argument("--n-shards", type=int, default=3)
    ap.add_argument("--alpha", type=float, default=0.5, help="Dirichlet concentration, noniid mode only")
    ap.add_argument("--seed", type=int, default=303, help="GLOBAL_INIT_SEED — shared model init across clients")
    ap.add_argument("--shard-seed", type=int, default=20240, help="CLIENT_SHARD_SEED")
    ap.add_argument("--arm", choices=["none", "dp"], default="none",
                     help="none (default) matches Step 14 ARM 1's 0.6667 baseline this step compares against — "
                          "DP noise would dominate regardless of partitioning and defeat the comparison")
    ap.add_argument("--orch-log", required=True)
    ap.add_argument("--db", default="federated_multimodal")
    ap.add_argument("--scratch-dir", default=str(REPO_ROOT / "trainer_outputs"))
    args = ap.parse_args()

    scratch = Path(args.scratch_dir)
    scratch.mkdir(parents=True, exist_ok=True)
    blank_stdin = scratch / "step12_blank_stdin.txt"
    ensure_blank_stdin(blank_stdin)

    print("=" * 70)
    print(f"STEP 20c VERIFY — mode={args.mode} arm={args.arm} n_shards={args.n_shards} "
          f"alpha={args.alpha if args.mode == 'noniid' else 'n/a'} seed={args.seed} shard_seed={args.shard_seed}")
    print("=" * 70)

    run([PYTHON, "scripts/reset_all_federated_dbs.py"])
    run([PYTHON, "enroll_step5.py", args.orch_log], timeout=60)

    base_env = os.environ.copy()
    base_env["PIPELINE_MODE"] = "multimodal"
    base_env["GLOBAL_INIT_SEED"] = str(args.seed)
    base_env["DP_MECHANISM"] = "gaussian" if args.arm == "dp" else "none"
    base_env["DP_NOISE_MULTIPLIER"] = "1.0" if args.arm == "dp" else "0.0"
    base_env["CLIENT_N_SHARDS"] = str(args.n_shards)
    base_env["CLIENT_SHARD_SEED"] = str(args.shard_seed)
    if args.mode == "noniid":
        base_env["CLIENT_NONIID_ALPHA"] = str(args.alpha)

    per_client = []
    for shard_id in range(args.n_shards):
        client_env = dict(base_env)
        client_env["CLIENT_SHARD_ID"] = str(shard_id)
        print(f"\n--- client {shard_id + 1}/{args.n_shards}  (CLIENT_SHARD_ID={shard_id}) ---")
        out = run([PYTHON, "run_client_multimodal.py"], env=client_env, stdin_path=str(blank_stdin), timeout=300)

        m = DELTA_L2_RE.search(out)
        delta_l2 = float(m.group(1)) if m else None
        shard_line = [l for l in out.splitlines() if l.startswith("[STEP20-SHARD]")]
        print(f"[parsed] shard_id={shard_id} delta_l2_before_clamp={delta_l2} shard_report={shard_line[-1] if shard_line else '(not found)'}")
        per_client.append({"shard_id": shard_id, "delta_l2_before_clamp": delta_l2,
                            "shard_report": shard_line[-1] if shard_line else None})

    wait_for_global_model(round_id=2, db_name=args.db)

    os.environ["MONGO_DATABASE"] = args.db
    from aggregate_offline import aggregate_offline
    from evaluate_global_model import evaluate_delta
    import torch
    device = "cuda" if torch.cuda.is_available() else "cpu"

    aggregated_delta = aggregate_offline(mongo_uri="mongodb://localhost:27017", db_name=args.db, round_id=1, mode="mean")
    eval_result = evaluate_delta(aggregated_delta, args.seed, device, label=f"step20 mode={args.mode}")

    deltas = [c["delta_l2_before_clamp"] for c in per_client if c["delta_l2_before_clamp"] is not None]
    out_of_range = [d for d in deltas if not (FIX_E4_RANGE[0] <= d <= FIX_E4_RANGE[1])]

    result = {
        "mode": args.mode,
        "n_shards": args.n_shards,
        "alpha": args.alpha if args.mode == "noniid" else None,
        "per_client": per_client,
        "per_client_delta_l2_spread": {"min": min(deltas) if deltas else None, "max": max(deltas) if deltas else None,
                                        "range": (max(deltas) - min(deltas)) if deltas else None},
        "fix_e4_calibration_range": list(FIX_E4_RANGE),
        "deltas_outside_fix_e4_range": out_of_range,
        "aggregated_delta_l2": eval_result["aggregated_delta_l2"],
        "f1": eval_result["f1"],
        "accuracy": eval_result["accuracy"],
        "predicted_positive": eval_result["predicted_positive"],
        "mae": eval_result["mae"],
        # arm=none: true epsilon is inf (no noise, no bound) — NOT the pipeline.py
        # fallback value (1.0) that lands in the receipt when a mechanism returns a
        # non-finite epsilon (see run_step14_multiround.py's module docstring / Step 12
        # notes on this exact fallback). arm=dp: real, data-independent RDP value.
        "epsilon": (5.302585 if args.arm == "dp" else float("inf")),
        "arm": args.arm,
    }
    print("\n[STEP20C_RESULT_JSON] " + json.dumps(result))
    return 0


if __name__ == "__main__":
    sys.exit(main())
