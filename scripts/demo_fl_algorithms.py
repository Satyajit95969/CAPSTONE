#!/usr/bin/env python3
"""
scripts/demo_fl_algorithms.py

STEP 19 — self-contained FedAvg / FedAdam / FedYogi demonstration.

Runs in under 3 minutes, fully offline: no orchestrator, no gRPC, no
MongoDB, no TPM. Safe to run live in a meeting on any machine with this
repo and its .venv.

Two things this script does NOT do:
  - It does NOT re-run the live 5-round experiments (Step 17, Step 18b).
    Their results are already measured and are embedded below as constants,
    each cited to the exact log file and commit they came from — see
    MEASURED_TRAJECTORIES below. Nothing here is fabricated or re-derived.
  - It does NOT duplicate the FedAdam/FedYogi algorithm code. Part 3 below
    imports fedavg_step/fedadam_step/fedyogi_step from
    scripts/fl_optimizers.py — the exact same functions
    scripts/run_step14_multiround.py used to produce the numbers in Part 2.

Usage:
    .venv\\Scripts\\python.exe scripts\\demo_fl_algorithms.py
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

REPO_ROOT = Path(r"D:\Download D\BE PIPELINE\Capstone-")
sys.path.insert(0, str(REPO_ROOT / "scripts"))

import torch  # noqa: E402
from fl_optimizers import fedavg_step, fedadam_step, fedyogi_step, init_moments  # noqa: E402

# ============================================================================
# PART 1 data: measured 5-round trajectories, DP enabled (noise_multiplier=1.0,
# clip_norm=0.85), mean aggregation, seed=303, LR_DECAY=0.7 (default).
#
# SOURCE — exact numbers, not re-derived, not estimated:
#   plain/FedAvg : trainer_outputs/step17_dp.log      [TRAJECTORY_JSON] line
#   FedAdam      : trainer_outputs/step18_fedadam.log [TRAJECTORY_JSON] line
#   FedYogi      : trainer_outputs/step18_fedyogi.log [TRAJECTORY_JSON] line
# Commit: 48b0cbb ("Step 18: FedAdam/FedYogi — 180x noise damping confirmed,
# utility still 0; momentum concentrates residual noise direction")
# Full analysis: docs/IMPLEMENTATION_NOTES.md, "Step 17" and "Step 18" sections.
#
# plain/FedAvg has no separate raw-vs-applied distinction (FedAvg's
# server-side step is the identity — see fl_optimizers.fedavg_step) so
# raw_l2 == applied_l2 there by construction, not by omission.
# ============================================================================

MEASURED_TRAJECTORIES = {
    "plain (FedAvg)": [
        {"round": 1, "raw_l2": 260.3603515625,   "applied_l2": 260.3603515625,   "f1": 0.0,                "accuracy": 0.7027027027027027, "pred_pos": 0,  "cum_eps": 5.302585},
        {"round": 2, "raw_l2": 260.76434326171875, "applied_l2": 260.76434326171875, "f1": 0.4583333333333333, "accuracy": 0.2972972972972973, "pred_pos": 37, "cum_eps": 10.60517},
        {"round": 3, "raw_l2": 260.5811767578125, "applied_l2": 260.5811767578125, "f1": 0.0,                "accuracy": 0.7027027027027027, "pred_pos": 0,  "cum_eps": 15.907755},
        {"round": 4, "raw_l2": 259.7243347167969, "applied_l2": 259.7243347167969, "f1": 0.0,                "accuracy": 0.7027027027027027, "pred_pos": 0,  "cum_eps": 21.21034},
        {"round": 5, "raw_l2": 259.9881896972656, "applied_l2": 259.9881896972656, "f1": 0.0,                "accuracy": 0.7027027027027027, "pred_pos": 0,  "cum_eps": 26.512925},
    ],
    "FedAdam": [
        {"round": 1, "raw_l2": 259.8680725097656,  "applied_l2": 1.438315749168396,  "f1": 0.0, "accuracy": 0.7027027027027027, "pred_pos": 0, "cum_eps": 5.302585},
        {"round": 2, "raw_l2": 260.31390380859375, "applied_l2": 1.4844391345977783, "f1": 0.0, "accuracy": 0.7027027027027027, "pred_pos": 0, "cum_eps": 10.60517},
        {"round": 3, "raw_l2": 260.7292175292969,  "applied_l2": 1.4468852281570435, "f1": 0.0, "accuracy": 0.7027027027027027, "pred_pos": 0, "cum_eps": 15.907755},
        {"round": 4, "raw_l2": 259.8692321777344,  "applied_l2": 1.3968515396118164, "f1": 0.0, "accuracy": 0.7027027027027027, "pred_pos": 0, "cum_eps": 21.21034},
        {"round": 5, "raw_l2": 259.8161315917969,  "applied_l2": 1.3437772989273071, "f1": 0.0, "accuracy": 0.7027027027027027, "pred_pos": 0, "cum_eps": 26.512925},
    ],
    "FedYogi": [
        {"round": 1, "raw_l2": 260.43536376953125, "applied_l2": 14.386189460754395, "f1": 0.0, "accuracy": 0.7027027027027027, "pred_pos": 0, "cum_eps": 5.302585},
        {"round": 2, "raw_l2": 260.4217224121094,  "applied_l2": 14.829349517822266, "f1": 0.0, "accuracy": 0.7027027027027027, "pred_pos": 0, "cum_eps": 10.60517},
        {"round": 3, "raw_l2": 260.34210205078125, "applied_l2": 14.460721969604492, "f1": 0.0, "accuracy": 0.7027027027027027, "pred_pos": 0, "cum_eps": 15.907755},
        {"round": 4, "raw_l2": 260.4133605957031,  "applied_l2": 13.941147804260254, "f1": 0.0, "accuracy": 0.7027027027027027, "pred_pos": 0, "cum_eps": 21.21034},
        {"round": 5, "raw_l2": 260.0072937011719,  "applied_l2": 13.412020683288574, "f1": 0.0, "accuracy": 0.7027027027027027, "pred_pos": 0, "cum_eps": 26.512925},
    ],
}


def print_header(title: str):
    print()
    print("=" * 100)
    print(title)
    print("=" * 100)


def part1_comparison_table():
    print_header("PART 1 — MEASURED 5-ROUND TRAJECTORY (DP enabled, noise_multiplier=1.0, mean aggregation, seed=303)")
    print("Source: trainer_outputs/step17_dp.log, step18_fedadam.log, step18_fedyogi.log — commit 48b0cbb.")
    print("Nothing below was re-run for this demo. See docs/IMPLEMENTATION_NOTES.md, Step 17/18 sections, for full analysis.\n")

    header = f"{'algorithm':<16} {'round':>5} {'raw pseudo-grad L2':>19} {'applied delta L2':>17} {'damping':>9} {'F1':>7} {'accuracy':>9} {'pred+/37':>9} {'cum eps':>9}"
    print(header)
    print("-" * len(header))
    for algo, rounds in MEASURED_TRAJECTORIES.items():
        for row in rounds:
            damping = row["raw_l2"] / row["applied_l2"] if row["applied_l2"] else float("inf")
            print(f"{algo:<16} {row['round']:>5} {row['raw_l2']:>19.3f} {row['applied_l2']:>17.4f} "
                  f"{damping:>8.2f}x {row['f1']:>7.4f} {row['accuracy']:>9.4f} {row['pred_pos']:>6}/37 {row['cum_eps']:>9.3f}")
        print()

    print("Reading this: 'raw pseudo-gradient L2' is the mean-aggregated, DP-noised client")
    print("delta before any server-side transform — identical magnitude (~260) across all three")
    print("rows, confirming the aggregation step itself is unchanged. 'applied delta L2' is what")
    print("actually gets added to the global model that round. For plain/FedAvg these are the same")
    print("number (FedAvg's server step is the identity). FedAdam damps ~180x; FedYogi ~18x —")
    print("10x eta_s (1e-2 vs 1e-3) accounts for almost exactly that 10x gap. F1 stays at 0 in")
    print("9 of 10 DP-arm rounds regardless — see Part 4 for why damping the noise didn't recover")
    print("utility.")


def part2_no_privacy_reference():
    print_header("PART 2 — FOR CONTEXT: THE NO-PRIVACY SIGNAL THIS DAMPED NOISE IS COMPETING AGAINST")
    print("Source: trainer_outputs/step16_none.log (Step 16's LR-decayed ARM 1, DP disabled entirely).")
    print("Same seed=303, same LR_DECAY=0.7, same 5 rounds — the reference used to compute noise/signal ratios below.\n")
    signal = [0.576, 0.419, 0.317, 0.233, 0.173]
    print(f"{'round':>5} {'no-privacy signal L2':>22} {'FedAdam applied L2':>20} {'FedAdam ratio':>14} {'FedYogi applied L2':>20} {'FedYogi ratio':>14}")
    for i, s in enumerate(signal):
        adam = MEASURED_TRAJECTORIES["FedAdam"][i]["applied_l2"]
        yogi = MEASURED_TRAJECTORIES["FedYogi"][i]["applied_l2"]
        print(f"{i+1:>5} {s:>22.3f} {adam:>20.4f} {adam/s:>13.2f}x {yogi:>20.4f} {yogi/s:>13.2f}x")
    print("\nNote the FedAdam ratio DEGRADES (2.50x -> 7.77x) across rounds, not improves — LR decay")
    print("(Step 16) shrinks the genuine signal every round while FedAdam's damped-noise floor stays")
    print("roughly flat. Same tension Step 17 found in plain averaging, still visible one layer down.")


def part3_synthetic_mechanism_demo():
    print_header("PART 3 — SYNTHETIC MECHANISM DEMO (NOT the real experiment — illustrative only)")
    print("Feeds a fresh, stable ~260-L2-magnitude random vector through fedavg_step / fedadam_step /")
    print("fedyogi_step for 5 synthetic 'rounds', imported directly from scripts/fl_optimizers.py —")
    print("the exact functions that produced Part 1's real numbers. This is here so the damping")
    print("mechanism (v converging to reflect noise magnitude) is visible in seconds, on data with a")
    print("known, controlled magnitude — NOT a re-run of the real 149-record training experiment.\n")

    torch.manual_seed(42)
    d = 5000  # small synthetic dimensionality — fast, not the real 281,254
    target_l2 = 260.0
    per_coord_std = target_l2 / (d ** 0.5)

    template = {"synthetic": torch.zeros(d)}
    m_adam = init_moments(template)
    v_adam = {"synthetic": torch.full((d,), (0.001) ** 2)}   # FedAdam's tau
    m_yogi = init_moments(template)
    v_yogi = {"synthetic": torch.full((d,), (0.001) ** 2)}   # FedYogi's tau

    header = f"{'round':>5} {'raw L2':>10} | {'FedAvg applied':>15} | {'FedAdam applied':>16} {'m_norm':>9} {'v_norm':>9} | {'FedYogi applied':>16} {'m_norm':>9} {'v_norm':>9}"
    print(header)
    print("-" * len(header))
    for r in range(1, 6):
        g = {"synthetic": torch.randn(d) * per_coord_std}
        raw_l2 = g["synthetic"].norm().item()

        avg_applied = fedavg_step(g)
        avg_l2 = avg_applied["synthetic"].norm().item()

        m_adam, v_adam, adam_applied = fedadam_step(m_adam, v_adam, g)
        adam_l2 = adam_applied["synthetic"].norm().item()
        adam_m_norm = m_adam["synthetic"].norm().item()
        adam_v_norm = v_adam["synthetic"].norm().item()

        m_yogi, v_yogi, yogi_applied = fedyogi_step(m_yogi, v_yogi, g)
        yogi_l2 = yogi_applied["synthetic"].norm().item()
        yogi_m_norm = m_yogi["synthetic"].norm().item()
        yogi_v_norm = v_yogi["synthetic"].norm().item()

        print(f"{r:>5} {raw_l2:>10.3f} | {avg_l2:>15.3f} | {adam_l2:>16.5f} {adam_m_norm:>9.3f} {adam_v_norm:>9.5f} | "
              f"{yogi_l2:>16.5f} {yogi_m_norm:>9.3f} {yogi_v_norm:>9.5f}")

    print("\nFedAvg's applied step stays at the raw noise scale every round (no mechanism to change it).")
    print("FedAdam/FedYogi's applied step shrinks as v_norm grows — v is learning the noise magnitude")
    print("and dividing it back out. This is the SAME mechanism Part 1's real 281,254-parameter run")
    print("used; only the dimensionality (5,000 here vs 281,254 there) differs, for speed.")


def part4_plain_english_summary():
    print_header("PART 4 — WHAT EACH ALGORITHM DOES (plain English)")
    print("""FedAvg (plain averaging)
  Every client trains locally and uploads its update. The server just averages
  them and applies the average directly. No memory of past rounds, no
  adjustment for how noisy or spread-out the updates are. Simplest possible
  rule — and here, DP noise (per-round L2 ~260) swamps genuine signal
  (~0.6) by ~450x, so the average is almost entirely noise.

FedAdam
  Same client-side training as FedAvg. On the server, wraps the averaged
  update in an Adam optimizer: tracks a running average of the update
  direction (m) and a running average of its squared magnitude (v), then
  divides the step by sqrt(v) before applying it — same idea Adam uses for
  ordinary gradient descent, just run once per FEDERATED ROUND instead of
  once per training step. Measured here: v successfully learned the DP
  noise's ~260 magnitude and divided the applied step down ~180x. But m
  (the running average of DIRECTION) also picked up whatever small residual
  direction 5 rounds of random noise happened to average out to — and
  applied that same residual, consistently, every round. Utility did not
  recover; see Part 1.

FedYogi
  Identical to FedAdam except how v is updated — instead of a smooth
  exponential average, v moves toward the new squared-magnitude value by a
  fixed step in whichever direction shrinks the gap. Designed to behave
  better than Adam when gradients are noisy/heavy-tailed. Measured here: v
  behaved almost identically to FedAdam's (same magnitude tracked, same
  ~260 noise floor learned), but FedYogi's 10x larger step size (eta_s=1e-2
  vs FedAdam's 1e-3) meant ~10x less damping (~18x vs ~180x) and,
  correspondingly, no utility recovery either.""")


def main() -> int:
    t0 = time.time()
    print("STEP 19 — FedAvg / FedAdam / FedYogi self-contained demonstration")
    print("Fully offline: no orchestrator, no gRPC, no MongoDB, no TPM.")

    part1_comparison_table()
    part2_no_privacy_reference()
    part3_synthetic_mechanism_demo()
    part4_plain_english_summary()

    elapsed = time.time() - t0
    print_header(f"DONE in {elapsed:.1f}s")
    return 0


if __name__ == "__main__":
    sys.exit(main())
