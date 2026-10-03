#!/usr/bin/env python3
"""
scripts/run_fedprox_modec.py

Measures FedProx's effect on client drift under Mode C (non-IID Dirichlet
sharding), see docs/IMPLEMENTATION_NOTES.md "FedProx".

For every mu in --mus and every repeat, runs one ROUND of 3 clients (shards
0/1/2 of the same Dirichlet partition). All three clients of a round share one
GLOBAL_INIT_SEED, so they start from identical trainable weights - the
round-0 "global model" the proximal term pulls towards. The harness proves
that rather than assuming it: client 0's logged sha256 of its initial
trainable weights is handed to clients 1 and 2 as EXPECTED_INIT_HASH (they
refuse to train on a mismatch), and the three logged hashes are compared
again here. Any mismatch aborts the whole run.

The same seeds are reused for every mu (a paired design): repeat r starts
from the same initial weights at mu=0, 1, 10 and 100.

SCOPE: each client is the live trainer entry point (orchestrate(), the same
call runtime/pipeline.py makes, same kwargs) run in its own process. DP,
encryption, upload and aggregation are NOT run - delta L2 and the local
metrics measured here are produced before any of them, and running the
server would aggregate after 3 uploads and push later rounds onto the
warm-start path (Defect A).

Usage:
    .venv\\Scripts\\python.exe scripts\\run_fedprox_modec.py
    .venv\\Scripts\\python.exe scripts\\run_fedprox_modec.py --mus 0 10 --repeats 1
"""

from __future__ import annotations

import argparse
import json
import os
import re
import statistics
import subprocess
import sys
import time
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
PYTHON = str(REPO_ROOT / ".venv" / "Scripts" / "python.exe")
PARQUET = REPO_ROOT / "dataset_build" / "daic_records_multimodal_participant_only.parquet"
OUT_DIR = REPO_ROOT / "trainer_outputs" / "fedprox_modec"

N_SHARDS = 3
SHARD_SEED = 20240
NONIID_ALPHA = 0.5
BASE_INIT_SEED = 1000   # repeat r uses GLOBAL_INIT_SEED = BASE_INIT_SEED + r

# Same call, same kwargs, as runtime/pipeline.py's trainer stage in multimodal
# mode (round 1, no global model).
CLIENT_SNIPPET = (
    "import sys;"
    f"sys.path.insert(0, r'{REPO_ROOT}');"
    f"sys.path.insert(0, r'{REPO_ROOT / 'installer' / 'runtime'}');"
    "import agents.trainer.trainer_mentalbert_privacy as T;"
    f"T.orchestrate(input_path=r'{PARQUET}', session_id=sys.argv[1], mode='supervised',"
    " epochs=T.SUPERVISED_EPOCHS, batch_size=8, lr=T.SUPERVISED_LR, round_id=1, max_samples=0)"
)

PATTERNS = {
    "init_hash": (r"\[INIT-HASH\].*sha256=([0-9a-f]{64})", str),
    "init_l2": (r"\[INIT-HASH\].*l2=([0-9.]+)", float),
    "shard_size": (r"\[STEP20-SHARD\].*size=(\d+)", int),
    "shard_pos": (r"\[STEP20-SHARD\].*positive=(\d+)", int),
    "shard_neg": (r"\[STEP20-SHARD\].*negative=(\d+)", int),
    "trainable": (r"Trainable params\s*:\s*([0-9,]+)", lambda s: int(s.replace(",", ""))),
    "delta_l2": (r"Delta L2 norm before clamp\s*:\s*([0-9.]+)", float),
    "accuracy": (r"\n\s*accuracy\s*:\s*([0-9.]+)", float),
    "precision": (r"\n\s*precision\s*:\s*([0-9.]+)", float),
    "recall": (r"\n\s*recall\s*:\s*([0-9.]+)", float),
    "f1": (r"\n\s*f1\s*:\s*([0-9.]+)", float),
}
LAST_PROX_RE = re.compile(r"epoch (\d+)/\d+ avg_task_loss=([0-9.]+) avg_loss_prox=([0-9.]+) "
                          r"dist_from_ref_at_epoch_end=([0-9.]+)")


def run_client(mu: float, rep: int, shard: int, expected_hash: str | None) -> dict:
    env = os.environ.copy()
    for k in ("MULTIMODAL_PARQUET_PATH", "FUSION_HIDDEN_DIM", "XAI_ENABLED"):
        env.pop(k, None)
    env.update({
        "PYTHONUTF8": "1", "PYTHONIOENCODING": "utf-8", "PIPELINE_MODE": "multimodal",
        "GLOBAL_INIT_SEED": str(BASE_INIT_SEED + rep), "FEDPROX_MU": repr(float(mu)),
        "CLIENT_SHARD_ID": str(shard), "CLIENT_N_SHARDS": str(N_SHARDS),
        "CLIENT_SHARD_SEED": str(SHARD_SEED), "CLIENT_NONIID_ALPHA": str(NONIID_ALPHA),
    })
    if expected_hash:
        env["EXPECTED_INIT_HASH"] = expected_hash
    else:
        env.pop("EXPECTED_INIT_HASH", None)

    tag = f"mu{mu:g}_rep{rep}_shard{shard}"
    t0 = time.time()
    proc = subprocess.run(
        [PYTHON, "-c", CLIENT_SNIPPET, f"fedprox-{tag}"], cwd=str(REPO_ROOT), env=env,
        input="\n" * 260,   # blank answer to every physician-feedback prompt (keeps the real label)
        capture_output=True, text=True, encoding="utf-8", errors="replace", timeout=900,
    )
    text = proc.stdout + "\n" + proc.stderr
    (OUT_DIR / f"{tag}.log").write_text(text, encoding="utf-8")
    if proc.returncode != 0:
        raise RuntimeError(f"client {tag} exited {proc.returncode} - see {OUT_DIR / (tag + '.log')}\n"
                           f"{text[-1500:]}")

    row = {"mu": mu, "rep": rep, "shard": shard, "seed": BASE_INIT_SEED + rep,
           "seconds": round(time.time() - t0, 1)}
    for key, (pat, cast) in PATTERNS.items():
        m = re.search(pat, text)
        if not m:
            raise RuntimeError(f"client {tag}: could not find {key!r} in its log")
        row[key] = cast(m.group(1))
    prox = LAST_PROX_RE.findall(text)
    row["final_avg_loss_prox"] = float(prox[-1][2]) if prox else None
    row["final_dist_from_ref"] = float(prox[-1][3]) if prox else None
    if (mu > 0) != bool(prox):
        raise RuntimeError(f"client {tag}: proximal logging {'missing' if mu > 0 else 'present'} at mu={mu}")
    # NEITHER = at least one held-out prediction differs from the rest.
    if row["recall"] == 0.0 and abs(row["accuracy"] - 26 / 37) < 5e-4:
        row["mode"] = "all-neg"
    elif row["recall"] == 1.0 and abs(row["accuracy"] - 11 / 37) < 5e-4:
        row["mode"] = "all-pos"
    else:
        row["mode"] = "NEITHER"
    return row


def var(xs: list[float]) -> float:
    return statistics.variance(xs) if len(xs) > 1 else float("nan")


def main() -> int:
    ap = argparse.ArgumentParser(description="FedProx drift measurement under Mode C")
    ap.add_argument("--mus", type=float, nargs="+", default=[0.0, 1.0, 10.0, 100.0])
    ap.add_argument("--repeats", type=int, default=5)
    args = ap.parse_args()

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    rows: list[dict] = []
    hash_by_seed: dict[int, str] = {}
    for mu in args.mus:
        for rep in range(1, args.repeats + 1):
            round_rows = []
            for shard in range(N_SHARDS):
                expected = round_rows[0]["init_hash"] if round_rows else None
                row = run_client(mu, rep, shard, expected)
                round_rows.append(row)
                print(f"mu={mu:g} rep={rep} shard={shard} size={row['shard_size']} "
                      f"({row['shard_pos']}+/{row['shard_neg']}-) delta_l2={row['delta_l2']:.6f} "
                      f"f1={row['f1']:.4f} acc={row['accuracy']:.4f} {row['mode']} "
                      f"init={row['init_hash'][:12]} {row['seconds']}s", flush=True)
            hashes = {r["init_hash"] for r in round_rows}
            if len(hashes) != 1:
                raise RuntimeError(f"ABORT: clients of round mu={mu} rep={rep} did not share one "
                                   f"initialisation: {[r['init_hash'] for r in round_rows]}")
            seed = BASE_INIT_SEED + rep
            if hash_by_seed.setdefault(seed, round_rows[0]["init_hash"]) != round_rows[0]["init_hash"]:
                raise RuntimeError(f"ABORT: seed {seed} gave a different initialisation at mu={mu} "
                                   f"than at an earlier mu - the design is no longer paired")
            rows.extend(round_rows)
            (OUT_DIR / "results.json").write_text(json.dumps(rows, indent=1), encoding="utf-8")

    print("\n=== PER-SHARD delta L2 (mean / variance over repeats) and F1 ===")
    print(f"{'mu':>6} {'shard':>5} {'size':>5} {'pos/neg':>8} {'n':>2} {'mean dL2':>9} {'var dL2':>10} "
          f"{'min':>7} {'max':>7} {'mean F1':>8} {'modes (pos/neg/NEITHER)':>24}")
    for mu in args.mus:
        for shard in range(N_SHARDS):
            rs = [r for r in rows if r["mu"] == mu and r["shard"] == shard]
            d = [r["delta_l2"] for r in rs]
            modes = [sum(r["mode"] == m for r in rs) for m in ("all-pos", "all-neg", "NEITHER")]
            print(f"{mu:>6g} {shard:>5} {rs[0]['shard_size']:>5} "
                  f"{str(rs[0]['shard_pos']) + '/' + str(rs[0]['shard_neg']):>8} {len(rs):>2} "
                  f"{statistics.fmean(d):>9.5f} {var(d):>10.3e} {min(d):>7.4f} {max(d):>7.4f} "
                  f"{statistics.fmean(r['f1'] for r in rs):>8.4f} {'/'.join(map(str, modes)):>24}")
    print("\n=== PER-ROUND spread across the 3 clients (variance of the 3 delta L2 values) ===")
    for mu in args.mus:
        vs = []
        for rep in range(1, args.repeats + 1):
            d = [r["delta_l2"] for r in rows if r["mu"] == mu and r["rep"] == rep]
            vs.append(var(d))
        print(f"mu={mu:>5g}  mean within-round variance={statistics.fmean(vs):.3e}  "
              f"per round: {' '.join(f'{v:.2e}' for v in vs)}")
    print(f"\nshared-initialisation check: PASSED for all {len(rows) // N_SHARDS} rounds "
          f"({len(hash_by_seed)} distinct seeds, each identical across clients and across mu)")
    print(f"results: {OUT_DIR / 'results.json'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
