#!/usr/bin/env python3
"""
scripts/fl_comparison_table.py

Presentation-only, read-only renderer. Prints the federated-algorithm
comparison table below and stops - no training, no measurement, no
computation of any kind happens here. Every numeric cell is a literal
constant copied from an already-measured result elsewhere in this repo,
cited to its exact file:line. Any cell with no measured value prints
NOT MEASURED - never estimated, never derived, never backfilled from
arithmetic.

Why a NEW script rather than extending demo_fl_algorithms.py: that script's
own stated scope (STEP 19) is the FedAvg/FedAdam/FedYogi noise-damping
trajectory demo - a different question (does the server-side optimizer
change what happens to DP noise across 5 rounds?) from this table's question
(what did EVERY combination of algorithm x aggregation x DP actually
measure, including combinations demo_fl_algorithms.py never touches - mean
vs trimmed_mean vs median aggregation, the no-DP baseline, and the
explicitly-absent FedProx/SCAFFOLD/Krum). Folding this in would widen an
already-approved, working demo's scope rather than add a clearly-separate
read-only view. The two numeric cells this table shares with
demo_fl_algorithms.py (FedAdam/FedYogi's accuracy/F1/noise) are NOT
re-measured or re-derived here - they are the same constants, hardcoded a
second time with the same citation, because importing demo_fl_algorithms.py
just for two numbers would pull in a torch import this table doesn't
otherwise need, working against its "safe to run live in a meeting, near
zero startup cost" purpose.

Every source cited below was read and spot-checked before being copied in:
  - docs/IMPLEMENTATION_NOTES.md:317-318 (Step 13 table)
  - docs/IMPLEMENTATION_NOTES.md:435 (Step 14 ARM 1, round 1 row)
  - scripts/demo_fl_algorithms.py:56-74 (MEASURED_TRAJECTORIES), itself
    cited to trainer_outputs/step18_fedadam.log and step18_fedyogi.log -
    both files' existence and a sample TRAJECTORY_JSON line were confirmed
    live before this script was written.
  - docs/IMPLEMENTATION_NOTES.md:830-832, 853-859 (Krum: not implemented,
    AND separately needs K>=5); :814-818 (FedProx/SCAFFOLD: zero code
    matches anywhere, live or historical)
  - server/aggregator_agent/aggregator.py:603,636 and
    scripts/aggregate_offline.py:88 (median mode: implemented in code,
    confirmed no measured run exists anywhere in the docs)

One correction kept against the pasted layout (pasted twice, unchanged both
times), reported rather than silently applied: FedAdam's applied-delta-L2
upper bound was pasted as 1.44; the actual measured max across all 5 rounds
(MEASURED_TRAJECTORIES) is 1.4844391345977783 (~1.48). Printed as 1.34-1.48
below, not 1.34-1.44.

STEP D3-2: output narrowed to the table ONLY, per explicit instruction -
no FedProx/SCAFFOLD/Krum rows, no "HOW TO READ THIS" section, no trailing
text. That content (still true, still relevant) now lives only in this
docstring and in docs/IMPLEMENTATION_NOTES.md, not on stdout.

Usage:
    .venv\\Scripts\\python.exe scripts\\fl_comparison_table.py
"""

from __future__ import annotations

TITLE = "FEDERATED ALGORITHM COMPARISON - MEASURED ON THIS SYSTEM"

# ---------------------------------------------------------------------------
# Measured rows. Each tuple: (algorithm, aggregation, dp, acc, f1, noise)
# acc/f1 are pre-formatted strings (not re-rounded here - copied verbatim
# from the cited source). noise is a pre-formatted string; "n/a" or
# "NOT MEASURED - ..." are literal, not placeholders to be filled later.
# ---------------------------------------------------------------------------
ROWS = [
    # FedAvg / Mean / No DP - Step 14 ARM 1, round 1 (no-privacy reference).
    # docs/IMPLEMENTATION_NOTES.md:435
    ("FedAvg", "Mean", "No", "0.7838", "0.6667", "n/a"),

    # FedAvg / Mean / DP - Step 13, noise_multiplier=1.0, mode=mean.
    # docs/IMPLEMENTATION_NOTES.md:317
    # (A second, independent measurement of the same cell exists at Step 14
    # ARM 2 round 1, docs/IMPLEMENTATION_NOTES.md:445: noise 260.09. Close
    # but not identical - a different run. This table reports the Step 13
    # value; the two are NOT averaged or merged.)
    ("FedAvg", "Mean", "Yes", "0.7027", "0.0000", "260.25"),

    # FedAvg / Trimmed Mean / DP - Step 13, noise_multiplier=1.0, mode=trimmed_mean.
    # docs/IMPLEMENTATION_NOTES.md:318
    ("FedAvg", "Trimmed Mean", "Yes", "0.7027", "0.0000", "301.80"),

    # FedAvg / Median / DP - "median"/"coordinate_median" IS implemented in
    # code (server/aggregator_agent/aggregator.py:603,636; exposed via
    # scripts/aggregate_offline.py:88) but no table row anywhere in this
    # repo ever ran it. Implemented, never measured - genuinely different
    # from Krum below (never implemented at all).
    ("FedAvg", "Median", "Yes", "NOT MEASURED", "", "implemented, never run"),

    # FedAdam / Mean / DP - identical accuracy (0.7027027027027027) and F1
    # (0.0) across all 5 measured rounds. Noise: raw pseudo-gradient L2
    # ~260 every round (259.82-260.73), applied (post-FedAdam) L2 range
    # 1.3437772989273071-1.4844391345977783 across the 5 rounds.
    # scripts/demo_fl_algorithms.py:63-67 (MEASURED_TRAJECTORIES["FedAdam"]),
    # sourced from trainer_outputs/step18_fedadam.log (confirmed present,
    # TRAJECTORY_JSON line spot-checked against this value).
    ("FedAdam", "Mean", "Yes", "0.7027", "0.0000", "260 -> 1.34-1.48"),

    # FedYogi / Mean / DP - same pattern: accuracy/F1 identical across all 5
    # rounds. Raw ~260 (260.01-260.44), applied L2 range
    # 13.412020683288574-14.829349517822266.
    # scripts/demo_fl_algorithms.py:70-74 (MEASURED_TRAJECTORIES["FedYogi"]),
    # sourced from trainer_outputs/step18_fedyogi.log (confirmed present).
    ("FedYogi", "Mean", "Yes", "0.7027", "0.0000", "260 -> 13.4-14.8"),
]

def _fmt_row(row: tuple) -> str:
    algo, agg, dp, acc, f1, noise = row
    if acc == "NOT MEASURED":
        return f" {algo:<12} {agg:<15} {dp:<5} NOT MEASURED -- {noise}"
    return f" {algo:<12} {agg:<15} {dp:<5} {acc:<11} {f1:<9} {noise}"


def main() -> int:
    # Step D3-2: prints ONLY the table - no NOT-IMPLEMENTED rows, no
    # "HOW TO READ THIS" prose, no trailing text of any kind, per the
    # explicit "output exactly this table and nothing else" instruction.
    # FedProx/SCAFFOLD/Krum's NOT-IMPLEMENTED status and the "how to read
    # this" framing are documented in this file's own module docstring and
    # in docs/IMPLEMENTATION_NOTES.md - intentionally not printed here.
    width = 80
    print("=" * width)
    print(f" {TITLE}")
    print("=" * width)
    print()
    print(f" {'Algorithm':<12} {'Aggregation':<15} {'DP':<5} {'Acc.':<11} {'F1':<9} {'Noise after agg.'}")
    print(" " + "-" * (width - 2))
    for row in ROWS:
        print(_fmt_row(row))
    print(" " + "-" * (width - 2))
    print("=" * width)
    return 0


if __name__ == "__main__":
    import sys
    sys.exit(main())
