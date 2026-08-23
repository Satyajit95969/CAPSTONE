#!/usr/bin/env python3
"""
scripts/run_step13_combo.py

STEP 13b driver: for ONE noise_multiplier value, submits ONE live 3-client
round (seed fixed at 303 per the approved design — the non-degenerate DP seed
from Step 12), then evaluates it under one or more offline aggregation modes
via scripts/aggregate_offline.py — the SAME 3 encrypted uploads reused across
every --modes value given, so a mean-vs-trimmed_mean comparison at a fixed
noise_multiplier is a controlled comparison on identical noisy data, not two
separate noisy draws.

Requires mongod and the Rust orchestrator already running (freshly restarted
against an empty DB — see docs/IMPLEMENTATION_NOTES.md's Step 12 notes on why
restart must follow reset, not precede it).

Also computes and reports the real epsilon (via the live _rdp_to_dp(), not
estimated) and whether it meets the project's eps<=8 target, for every mode
evaluated (epsilon depends only on noise_multiplier/clip_norm/delta — it does
not vary by aggregation mode, so it's the same value for every --modes entry
in one combo, but is reported per result row as the user asked).

Usage:
    .venv\\Scripts\\python.exe scripts\\run_step13_combo.py \\
        --noise-multiplier 1.0 --modes mean,trimmed_mean --seed 303 \\
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

REPO_ROOT = Path(r"D:\Download D\BE PIPELINE\Capstone-")
PYTHON = str(REPO_ROOT / ".venv" / "Scripts" / "python.exe")

EPS_TARGET = 8.0
CLIP_NORM = 0.85
DELTA = 1e-5


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
    sys.stdout.write(proc.stdout[-4000:])
    if proc.returncode != 0:
        sys.stderr.write(proc.stderr[-4000:])
        raise RuntimeError(f"Command failed (exit {proc.returncode}): {' '.join(cmd)}")
    return proc.stdout


def ensure_blank_stdin(path: Path, n_lines: int = 260):
    if not path.exists():
        path.write_text("\n" * n_lines)


def real_epsilon(noise_multiplier: float) -> float:
    if str(REPO_ROOT) not in sys.path:
        sys.path.insert(0, str(REPO_ROOT))
    sys.path.insert(0, str(REPO_ROOT / "installer" / "runtime"))
    from agents.dp.dp_agent import _rdp_to_dp
    return _rdp_to_dp(noise_multiplier=noise_multiplier, clip_norm=CLIP_NORM, delta=DELTA)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--noise-multiplier", type=float, required=True)
    ap.add_argument("--modes", required=True, help="comma-separated: mean,trimmed_mean")
    ap.add_argument("--seed", type=int, default=303)
    ap.add_argument("--orch-log", required=True)
    ap.add_argument("--db", default="federated_multimodal")
    ap.add_argument("--scratch-dir", default=str(REPO_ROOT / "trainer_outputs"))
    args = ap.parse_args()

    modes = [m.strip() for m in args.modes.split(",")]

    scratch = Path(args.scratch_dir)
    scratch.mkdir(parents=True, exist_ok=True)
    blank_stdin = scratch / "step12_blank_stdin.txt"
    ensure_blank_stdin(blank_stdin)

    print("=" * 70)
    print(f"STEP 13 COMBO — noise_multiplier={args.noise_multiplier} seed={args.seed} modes={modes}")
    print("=" * 70)

    eps = real_epsilon(args.noise_multiplier)
    eps_pass = eps <= EPS_TARGET
    print(f"[info] real epsilon (RDP, T=1, clip={CLIP_NORM}, delta={DELTA}) = {eps:.6f}  "
          f"eps<=8: {'PASS' if eps_pass else 'FAIL'}")

    # 1. reset
    run([PYTHON, "scripts/reset_all_federated_dbs.py"])

    # 2. enroll (reset wiped "devices")
    run([PYTHON, "enroll_step5.py", args.orch_log], timeout=60)

    # 3. three sequential client submissions, identical env for the round
    client_env = os.environ.copy()
    client_env["PIPELINE_MODE"] = "multimodal"
    client_env["GLOBAL_INIT_SEED"] = str(args.seed)
    client_env["DP_MECHANISM"] = "gaussian"
    client_env["DP_NOISE_MULTIPLIER"] = str(args.noise_multiplier)

    for i in range(1, 4):
        print(f"\n--- client submission {i}/3 (nm={args.noise_multiplier}, seed={args.seed}) ---")
        run([PYTHON, "run_client_multimodal.py"], env=client_env, stdin_path=str(blank_stdin), timeout=300)

    # 4. evaluate each requested aggregation mode on the SAME 3 uploads
    #
    # aggregator.py reads its OWN Mongo database name from the MONGO_DATABASE
    # env var at import time (aggregator.py:85, default "federated") — it is
    # NOT parameterised by aggregate_offline.py's --db argument, which only
    # controls fetch_round_updates()'s own query. Without this env var set
    # here, AggregatorAgent._decrypt_from_gridfs() looks the file up in the
    # wrong database and raises gridfs.errors.NoFile even though the 3
    # updates were found correctly moments earlier.
    agg_env = os.environ.copy()
    agg_env["MONGO_DATABASE"] = args.db

    combo_results = []
    for mode in modes:
        print(f"\n--- offline aggregation: mode={mode} (round_id=1) ---")
        out = run([PYTHON, "scripts/aggregate_offline.py",
                   "--round-id", "1", "--mode", mode, "--seed", str(args.seed), "--db", args.db],
                  env=agg_env, timeout=180)
        eval_result = None
        for line in out.splitlines():
            if line.startswith("[RESULT_JSON] "):
                eval_result = json.loads(line[len("[RESULT_JSON] "):])
                break
        if eval_result is None:
            raise RuntimeError(f"aggregate_offline.py (mode={mode}) did not print a [RESULT_JSON] line")

        row = dict(eval_result)
        row["mode"] = mode
        row["noise_multiplier"] = args.noise_multiplier
        row["epsilon"] = eps
        row["eps_le_8"] = eps_pass
        combo_results.append(row)
        print("[CELL_RESULT_JSON] " + json.dumps(row))

    print("\n" + "=" * 70)
    print(f"COMBO COMPLETE — noise_multiplier={args.noise_multiplier} ({len(combo_results)} cell(s))")
    print("=" * 70)
    return 0


if __name__ == "__main__":
    sys.exit(main())
