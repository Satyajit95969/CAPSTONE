#!/usr/bin/env python3
"""
scripts/run_step14_multiround.py

STEP 14 — multi-round privacy-utility trajectory (docs/IMPLEMENTATION_NOTES.md,
"Step 14" section). Every prior measurement (Step 12, Step 13) is round 1 —
one noisy draw, aggregated once. This runs N_ROUNDS sequential rounds for one
arm, with warm-start genuinely exercised, using the SAME Defect A/B
workaround established in Step 12/13 — never pipeline.py's buggy live
warm-start loader, which is left completely untouched.

Mechanism (no Rust, no security touched):
  1. Round r's 3 clients submit as normal through the real, unmodified
     pipeline (gRPC/TPM/mTLS/DP/GridFS upload).
  2. The orchestrator's own automatic trimmed_mean aggregation fires as
     always (server.rs: triggered at updates.len()>=3) — its OUTPUT is a
     buggy raw delta (Defect A) and is never used for anything below; letting
     it run is simply unavoidable (it's automatic) and harmless.
  3. This script computes ITS OWN mean-aggregated delta for round r's 3
     uploads (scripts/aggregate_offline.py's aggregate_offline(), unmodified
     AggregatorAgent class), adds it to a running `cumulative_delta`
     (elementwise sum across rounds — this IS the correct multi-round FedAvg
     accumulation: M_r = M_0 + sum_{i<=r}(mean_i)).
  4. Evaluates M_r = base_state(seed) + cumulative_delta on the held-out 37
     (evaluate_global_model.reconstruct_absolute_state() + the same eval
     loop evaluate_delta() uses).
  5. Before round r+1's clients run, PATCHES the `global_models` MongoDB
     document for round_id=r+1 (Mongo write only — same generic collection
     the orchestrator already reads from, no Rust/security code involved) so
     its `file_id` points to a NEWLY UPLOADED GridFS object containing M_r as
     genuine ABSOLUTE weights, not a delta. pipeline.py's existing, UNCHANGED
     warm-start call (`model.load_state_dict(global_state, strict=False)`)
     is only wrong when fed a delta (Defect A) — fed real weights, as here,
     it is exactly correct with zero code changes to pipeline.py or
     trainer_mentalbert_privacy.py.

Epsilon: per-round epsilon is real and data-independent (same RDP formula as
Step 12/13). "Cumulative epsilon" is reported as the NAIVE additive bound
(sum of per-round epsilons) — this system has no true sequential multi-round
RDP accountant (CLAUDE.md's own documented defect #7: "Privacy accounting is
per-round only; no cumulative eps across rounds"). The additive bound is a
loose upper bound, not a tight composition, and is labeled as such — not
fabricated as if it were the real accountant's output.

STEP 18 — FedAdam / FedYogi (--update-rule). Server-side only, no Rust, no
protobuf, no client changes — same offline-aggregation-plus-GridFS-patch
pattern as Steps 13/14/17. Replaces the plain additive accumulation
(cumulative += mean(client deltas)) with the Reddi et al. 2020 "Adaptive
Federated Optimization" update rule: treat the round's mean-aggregated client
delta as a pseudo-gradient, run it through a server-side Adam/Yogi optimizer
(m/v tracked in-process across this driver's round loop, exactly like
cumulative_delta already is — no new MongoDB collection), and accumulate the
OPTIMIZER'S output, not the raw pseudo-gradient.

This is not a checkbox exercise. Step 14/17 measured DP noise accumulating as
sqrt(r) at a STABLE ~260 L2 magnitude per round (noise_multiplier=1.0,
clip_norm=0.85, mean-of-3). FedAdam/FedYogi's v (running second-moment
estimate) is exactly a running estimate of that per-coordinate magnitude — if
the noise component really is stable round to round, v should converge to
reflect it, and m_hat/(sqrt(v_hat)+tau) would then divide it back down to a
roughly unit-scale, per-coordinate step before eta_s rescales it — a
normalization plain averaging has no mechanism for. Whether this actually
damps the noise enough to matter is the open, genuinely unknown question
Step 18b measures — not assumed here.

Usage:
    .venv\\Scripts\\python.exe scripts\\run_step14_multiround.py \\
        --arm dp --rounds 5 --seed 303 --update-rule fedadam \\
        --orch-log C:\\path\\to\\orch.log
    .venv\\Scripts\\python.exe scripts\\run_step14_multiround.py \\
        --arm none --rounds 5 --seed 303 \\
        --orch-log C:\\path\\to\\orch.log
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path

import torch  # module level, not inside main() — fedadam_step()/fedyogi_step() below need it as a global

REPO_ROOT = Path(r"D:\Download D\BE PIPELINE\Capstone-")
PYTHON = str(REPO_ROOT / ".venv" / "Scripts" / "python.exe")
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(REPO_ROOT / "scripts"))
sys.path.insert(0, str(REPO_ROOT / "installer" / "runtime"))

ARM_ENV = {
    "none": {"DP_MECHANISM": "none", "DP_NOISE_MULTIPLIER": "0.0"},
    "dp": {"DP_MECHANISM": "gaussian", "DP_NOISE_MULTIPLIER": "1.0"},
}
ARM_PER_ROUND_EPSILON = {"none": None, "dp": 5.302585}

# Step 19: the update rules moved to scripts/fl_optimizers.py so
# scripts/demo_fl_algorithms.py can import the SAME code rather than a copy.
from fl_optimizers import (  # noqa: E402
    init_moments, fedadam_step, fedyogi_step,
    FEDADAM_ETA_S, FEDADAM_BETA1, FEDADAM_BETA2, FEDADAM_TAU,
    FEDYOGI_ETA_S, FEDYOGI_TAU, FEDYOGI_BETA1, FEDYOGI_BETA2,
)


def run(cmd, env=None, stdin_path=None, timeout=180):
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
    sys.stdout.write(proc.stdout[-3000:])
    if proc.returncode != 0:
        sys.stderr.write(proc.stderr[-3000:])
        raise RuntimeError(f"Command failed (exit {proc.returncode}): {' '.join(cmd)}")
    return proc.stdout


def ensure_blank_stdin(path: Path, n_lines: int = 260):
    if not path.exists():
        path.write_text("\n" * n_lines)


def wait_for_global_model(round_id: int, db_name: str, timeout: float = 90.0):
    import pymongo
    client = pymongo.MongoClient("mongodb://localhost:27017")
    db = client[db_name]
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        doc = db["global_models"].find_one({"round_id": round_id})
        if doc is not None:
            client.close()
            return
        time.sleep(1.0)
    client.close()
    raise TimeoutError(f"No global_models document for round {round_id} after {timeout}s")


def patch_global_model(db_name: str, round_id: int, new_state: dict):
    """Uploads new_state (absolute weights, 16 keys) to a fresh GridFS object
    and repoints global_models[round_id].file_id at it — see module docstring.
    Pure MongoDB write; no Rust/security code touched."""
    import io
    import torch
    import gridfs
    from pymongo import MongoClient

    client = MongoClient("mongodb://localhost:27017")
    db = client[db_name]
    fs = gridfs.GridFS(db)

    buf = io.BytesIO()
    torch.save(new_state, buf)
    file_id = fs.put(buf.getvalue(), filename=f"global_model_round_{round_id}_step14_patched.pt")

    result = db["global_models"].update_one({"round_id": round_id}, {"$set": {"file_id": file_id}})
    if result.matched_count == 0:
        client.close()
        raise RuntimeError(
            f"No global_models document with round_id={round_id} to patch — "
            f"the orchestrator's own automatic aggregation should have created one."
        )
    client.close()
    print(f"[patch] global_models[round_id={round_id}].file_id -> {file_id} "
          f"(genuine absolute weights, {sum(v.numel() for v in new_state.values()):,} params)")


def add_deltas(a: dict, b: dict) -> dict:
    if not a:
        return {k: v.clone() for k, v in b.items()}
    assert set(a.keys()) == set(b.keys()), "cumulative_delta / this_round_delta key mismatch"
    return {k: a[k] + b[k].to(a[k].dtype) for k in a}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--arm", choices=["none", "dp"], required=True)
    ap.add_argument("--rounds", type=int, default=5)
    ap.add_argument("--seed", type=int, default=303)
    ap.add_argument("--orch-log", required=True)
    ap.add_argument("--db", default="federated_multimodal")
    ap.add_argument("--scratch-dir", default=str(REPO_ROOT / "trainer_outputs"))
    ap.add_argument("--update-rule", choices=["plain", "fedadam", "fedyogi"], default="plain",
                     help="Step 18: server-side accumulation rule. plain = Step 14/17 behaviour (unchanged).")
    args = ap.parse_args()

    scratch = Path(args.scratch_dir)
    scratch.mkdir(parents=True, exist_ok=True)
    blank_stdin = scratch / "step12_blank_stdin.txt"
    ensure_blank_stdin(blank_stdin)

    print("=" * 70)
    print(f"STEP 14/18 MULTI-ROUND — arm={args.arm} rounds={args.rounds} seed={args.seed} update_rule={args.update_rule}")
    if args.update_rule == "fedadam":
        print(f"[STEP18] FedAdam: eta_s={FEDADAM_ETA_S} beta1={FEDADAM_BETA1} beta2={FEDADAM_BETA2} tau={FEDADAM_TAU}")
    elif args.update_rule == "fedyogi":
        print(f"[STEP18] FedYogi: eta_s={FEDYOGI_ETA_S} beta1={FEDYOGI_BETA1} beta2={FEDYOGI_BETA2} tau={FEDYOGI_TAU}")
    print("=" * 70)

    # 1. reset + enroll ONCE for this arm (not per round)
    run([PYTHON, "scripts/reset_all_federated_dbs.py"])
    run([PYTHON, "enroll_step5.py", args.orch_log], timeout=60)

    client_env = os.environ.copy()
    client_env["PIPELINE_MODE"] = "multimodal"
    client_env["GLOBAL_INIT_SEED"] = str(args.seed)
    client_env.update(ARM_ENV[args.arm])

    # aggregator.py reads MONGO_DATABASE from the environment at IMPORT time
    # (aggregator.py:85) — must be set before `from aggregate_offline import
    # aggregate_offline` below, not inside the round loop.
    os.environ["MONGO_DATABASE"] = args.db

    device = "cuda" if torch.cuda.is_available() else "cpu"
    from aggregate_offline import aggregate_offline
    from evaluate_global_model import evaluate_delta, reconstruct_absolute_state

    per_round_eps = ARM_PER_ROUND_EPSILON[args.arm]
    cumulative_delta: dict = {}
    m_state: dict = {}
    v_state: dict = {}
    trajectory = []

    for r in range(1, args.rounds + 1):
        print(f"\n{'='*70}\nROUND {r}/{args.rounds}  (arm={args.arm})\n{'='*70}")
        for i in range(1, 4):
            print(f"\n--- round {r}, client {i}/3 ---")
            run([PYTHON, "run_client_multimodal.py"], env=client_env, stdin_path=str(blank_stdin), timeout=300)

        wait_for_global_model(round_id=r + 1, db_name=args.db)

        # This script's own clean mean aggregation of round r's 3 uploads —
        # NOT the orchestrator's own (buggy, unused) trimmed_mean output.
        # This is the "pseudo-gradient" Reddi et al. 2020 feeds to the
        # server-side optimizer — raw, before any FedAdam/FedYogi transform.
        raw_pseudo_gradient = aggregate_offline(
            mongo_uri="mongodb://localhost:27017", db_name=args.db, round_id=r, mode="mean"
        )

        if args.update_rule == "plain":
            applied_delta = raw_pseudo_gradient
        else:
            if not m_state:
                m_state = init_moments(raw_pseudo_gradient)
                v_state = {k: torch.full_like(v.float(), (FEDADAM_TAU if args.update_rule == "fedadam" else FEDYOGI_TAU) ** 2)
                           for k, v in raw_pseudo_gradient.items()}
            step_fn = fedadam_step if args.update_rule == "fedadam" else fedyogi_step
            m_state, v_state, applied_delta = step_fn(m_state, v_state, raw_pseudo_gradient)

        cumulative_delta = add_deltas(cumulative_delta, applied_delta)

        raw_l2 = torch.sqrt(sum((v.float().norm() ** 2) for v in raw_pseudo_gradient.values())).item()
        applied_l2 = torch.sqrt(sum((v.float().norm() ** 2) for v in applied_delta.values())).item()
        m_norm = torch.sqrt(sum((v.float().norm() ** 2) for v in m_state.values())).item() if m_state else None
        v_norm = torch.sqrt(sum((v.float().norm() ** 2) for v in v_state.values())).item() if v_state else None
        print(f"[STEP18] round={r} update_rule={args.update_rule} raw_pseudo_gradient_l2={raw_l2:.6f} "
              f"applied_delta_l2={applied_l2:.6f} m_norm={m_norm} v_norm={v_norm}")

        eval_result = evaluate_delta(cumulative_delta, args.seed, device, label=f"arm={args.arm} after_round={r}")

        cum_eps = None if per_round_eps is None else round(per_round_eps * r, 6)
        row = dict(eval_result)
        row["arm"] = args.arm
        row["round"] = r
        row["update_rule"] = args.update_rule
        row["raw_pseudo_gradient_l2"] = raw_l2
        # this_round_delta_l2 kept as the name Step 14/17 used, for direct
        # comparability — it means "how much the cumulative model moved this
        # round," i.e. the APPLIED delta (== raw pseudo-gradient when plain).
        row["this_round_delta_l2"] = applied_l2
        row["m_norm"] = m_norm
        row["v_norm"] = v_norm
        row["per_round_epsilon"] = per_round_eps
        row["cumulative_epsilon_naive_additive_bound"] = cum_eps
        trajectory.append(row)
        print("[ROUND_RESULT_JSON] " + json.dumps(row))

        if r < args.rounds:
            _, new_state, _, _, _, _ = reconstruct_absolute_state(cumulative_delta, args.seed, device)
            patch_global_model(args.db, round_id=r + 1, new_state=new_state)

    print("\n" + "=" * 70)
    print(f"MULTI-ROUND COMPLETE — arm={args.arm}, update_rule={args.update_rule}, {len(trajectory)} rounds")
    print("=" * 70)
    print("[TRAJECTORY_JSON] " + json.dumps(trajectory))
    return 0


if __name__ == "__main__":
    sys.exit(main())
