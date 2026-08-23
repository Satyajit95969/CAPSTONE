#!/usr/bin/env python3
"""
scripts/run_step12_combo.py

STEP 12b driver for ONE (arm, seed) combination of the live comparison
(ARM 1 "no privacy" / ARM 2 "current DP"). Not used for ARM 3 (that's fully
offline — see run_arm3_local_baseline.py).

For the given arm/seed:
  1. reset_all_federated_dbs.py            (clean state — also wipes "devices")
  2. enroll_step5.py <orch-log>            (re-enroll: reset wiped devices)
  3. run_client_multimodal.py  x3          (GLOBAL_INIT_SEED + DP_* env set
                                             identically for all 3 — this
                                             round's "3 clients" are 3
                                             sequential submissions from the
                                             single enrolled device, matching
                                             every prior verification run in
                                             this project)
  4. poll MongoDB for global_models round_id=1 (aggregation is async,
     triggered server-side once updates.len() >= 3 — server.rs:702)
  5. scripts/evaluate_global_model.py --seed <seed> --round-id 1

Prints one final "[COMBO_RESULT_JSON] {...}" line merging the eval result
with the arm's known (data-independent) epsilon, for the outer report to
collect.

Requires mongod and the Rust orchestrator already running.

Usage:
    .venv\\Scripts\\python.exe scripts\\run_step12_combo.py \\
        --arm dp --seed 101 --orch-log C:\\path\\to\\orch.log
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path

REPO_ROOT = Path(r"D:\Download D\BE PIPELINE\Capstone-")
PYTHON = str(REPO_ROOT / ".venv" / "Scripts" / "python.exe")

ARM_ENV = {
    # ARM 1: no privacy. mechanism="none" makes DPAgent.add_noise() a no-op
    # (dp_agent.py:148); clipping (self.clip=0.85) still applies but is inert
    # (measured delta L2 ~0.62-0.67 < 0.85). True epsilon is undefined/inf —
    # NOT the pipeline.py fallback value of 1.0 that lands in the receipt
    # (pipeline.py's fallback exists to keep the receipt non-degenerate when
    # a mechanism reports inf/None; it is not a real epsilon and must not be
    # reported as one here).
    "none": {"DP_MECHANISM": "none", "DP_NOISE_MULTIPLIER": "0.0"},
    # ARM 2: current DP config, unchanged (gaussian, noise_multiplier=1.0,
    # clip_norm=0.85 default). eps = 5.302585 at delta=1e-5 (RDP, T=1) —
    # data-independent, computed directly rather than scraped from output.
    "dp": {"DP_MECHANISM": "gaussian", "DP_NOISE_MULTIPLIER": "1.0"},
}

ARM_EPSILON = {
    "none": None,   # reported as "inf (no privacy)" by the caller, not a number
    "dp": 5.302585,
}


def run(cmd, env=None, stdin_path=None, timeout=180):
    print(f"[run] {' '.join(cmd)}")
    # Child processes here print non-ASCII characters (e.g. '→' in
    # train_model()'s log lines). With stdout/stderr piped (not a real
    # console), Python on Windows falls back to the system ANSI codepage
    # (cp1252) unless told otherwise, which can't encode them and crashes
    # the child. Forcing UTF-8 here is a subprocess-environment fix scoped
    # to this driver — it does not touch trainer_mentalbert_privacy.py.
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
    sys.stdout.write(proc.stdout[-4000:])
    if proc.returncode != 0:
        sys.stderr.write(proc.stderr[-4000:])
        raise RuntimeError(f"Command failed (exit {proc.returncode}): {' '.join(cmd)}")
    return proc.stdout


def ensure_blank_stdin(path: Path, n_lines: int = 260):
    if not path.exists():
        path.write_text("\n" * n_lines)


def wait_for_global_model(round_id: int, db_name: str, timeout: float = 60.0) -> None:
    import pymongo
    client = pymongo.MongoClient("mongodb://localhost:27017")
    db = client[db_name]
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        doc = db["global_models"].find_one({"round_id": round_id})
        if doc is not None:
            client.close()
            print(f"[info] global_models document for round {round_id} found after aggregation")
            return
        time.sleep(1.0)
    client.close()
    raise TimeoutError(f"No global_models document for round {round_id} after {timeout}s — aggregation did not complete")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--arm", choices=["none", "dp"], required=True)
    ap.add_argument("--seed", type=int, required=True)
    ap.add_argument("--orch-log", required=True, help="Rust orchestrator's stdout log file, for OTP scraping")
    ap.add_argument("--db", default="federated_multimodal")
    ap.add_argument("--scratch-dir", default=str(REPO_ROOT / "trainer_outputs"))
    args = ap.parse_args()

    scratch = Path(args.scratch_dir)
    scratch.mkdir(parents=True, exist_ok=True)
    blank_stdin = scratch / "step12_blank_stdin.txt"
    ensure_blank_stdin(blank_stdin)

    print("=" * 70)
    print(f"STEP 12 COMBO — arm={args.arm} seed={args.seed}")
    print("=" * 70)

    # 1. reset
    run([PYTHON, "scripts/reset_all_federated_dbs.py"])

    # 2. enroll (reset wiped "devices")
    run([PYTHON, "enroll_step5.py", args.orch_log], timeout=60)

    # 3. three sequential client submissions, identical env for the round
    client_env = os.environ.copy()
    client_env["PIPELINE_MODE"] = "multimodal"
    client_env["GLOBAL_INIT_SEED"] = str(args.seed)
    client_env.update(ARM_ENV[args.arm])

    for i in range(1, 4):
        print(f"\n--- client submission {i}/3 (arm={args.arm}, seed={args.seed}) ---")
        run([PYTHON, "run_client_multimodal.py"], env=client_env, stdin_path=str(blank_stdin), timeout=300)

    # 4. wait for async server-side aggregation (server.rs: triggered at
    # updates.len() >= 3). The global_models bookkeeping document the
    # orchestrator writes is round_id = N+1 (server.rs's own comment: "for
    # round N+1") — aggregating round 1's updates produces a doc with
    # round_id=2, NOT round_id=1. See evaluate_global_model.py's
    # fetch_aggregated_delta() docstring for the full explanation.
    GLOBAL_MODEL_ROUND_ID = 2
    wait_for_global_model(round_id=GLOBAL_MODEL_ROUND_ID, db_name=args.db, timeout=90)

    # 5. evaluate
    eval_out = run([PYTHON, "scripts/evaluate_global_model.py",
                    "--seed", str(args.seed), "--round-id", str(GLOBAL_MODEL_ROUND_ID), "--db", args.db],
                   timeout=180)

    eval_result = None
    for line in eval_out.splitlines():
        if line.startswith("[RESULT_JSON] "):
            eval_result = json.loads(line[len("[RESULT_JSON] "):])
            break
    if eval_result is None:
        raise RuntimeError("evaluate_global_model.py did not print a [RESULT_JSON] line")

    combo_result = dict(eval_result)
    combo_result["arm"] = args.arm
    combo_result["epsilon"] = ARM_EPSILON[args.arm]

    print("\n" + "=" * 70)
    print(f"COMBO COMPLETE — arm={args.arm} seed={args.seed}")
    print("=" * 70)
    print("[COMBO_RESULT_JSON] " + json.dumps(combo_result))
    return 0


if __name__ == "__main__":
    sys.exit(main())
