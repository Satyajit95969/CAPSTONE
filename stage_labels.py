#!/usr/bin/env python3
"""
stage_labels.py  (Phase 6 - Step 8 staging helper)

Consolidates the AVEC2017 DAIC-WOZ split label files into the single, unambiguous
label sheet that build_daic_records.py expects:

    <src>/train_split_Depression_AVEC2017*.csv
    <src>/dev_split_Depression_AVEC2017.csv
    <src>/full_test_split.csv
        |  columns: Participant_ID, PHQ8_Binary, PHQ8_Score, Gender, ...
        v
    ./labels/Detailed_PHQ8_Labels.csv   columns: Participant_ID, PHQ8_Score

WHY: build_daic_records.py (and Ritik's format_daic_to_lda.py) select the PHQ
column by "PHQ" in the name. In the raw AVEC header, PHQ8_Binary precedes
PHQ8_Score, so the heuristic would pick the BINARY (0/1) column and collapse every
severe case to label 0 (proven: pid 308, real PHQ8_Score=22, would become phq=1).
Emitting ONLY Participant_ID + PHQ8_Score makes the heuristic unambiguous — a pure
label-staging step, no change to any trainer/build source.

AVEC2017 split IDs are disjoint, so train+dev+test concatenate without collisions.
Blank rows (e.g. the ',,,' line in full_test_split.csv) are dropped.

Usage:
    python stage_labels.py                                 # D:/datasets/DAIC-WOZ -> ./labels/Detailed_PHQ8_Labels.csv
    python stage_labels.py --src "D:/datasets/DAIC-WOZ"
"""

import argparse
import sys
from pathlib import Path

import pandas as pd

ID_COL = "Participant_ID"
SCORE_COL = "PHQ8_Score"


def collect_label_csvs(src: Path):
    # Any CSV in src carrying both required columns (auto-covers the 3 AVEC splits,
    # ignores unrelated CSVs such as COVAREP/FORMANT feature files).
    found = []
    for p in sorted(src.glob("*.csv")):
        try:
            cols = pd.read_csv(p, nrows=0).columns
        except Exception:
            continue
        if ID_COL in cols and SCORE_COL in cols:
            found.append(p)
    return found


def main() -> int:
    ap = argparse.ArgumentParser(description="Stage AVEC2017 splits -> Detailed_PHQ8_Labels.csv (Phase 6 Step 8)")
    ap.add_argument("--src", type=Path, default=Path("D:/datasets/DAIC-WOZ"),
                    help="Folder with AVEC2017 split CSVs (default: D:/datasets/DAIC-WOZ)")
    ap.add_argument("--out", type=Path, default=Path("./labels/Detailed_PHQ8_Labels.csv"))
    args = ap.parse_args()

    if not args.src.exists():
        print(f"[ERROR] src not found: {args.src}")
        return 2

    csvs = collect_label_csvs(args.src)
    if not csvs:
        print(f"[ERROR] no CSVs with columns [{ID_COL}, {SCORE_COL}] under {args.src}")
        return 1
    print(f"[INFO] label sources: {[p.name for p in csvs]}")

    frames = []
    for p in csvs:
        df = pd.read_csv(p, usecols=lambda c: c in (ID_COL, SCORE_COL))
        frames.append(df[[ID_COL, SCORE_COL]])
    merged = pd.concat(frames, ignore_index=True)

    before = len(merged)
    merged = merged.dropna(subset=[ID_COL, SCORE_COL])
    # Coerce: ID -> integer-like string ("300.0" -> "300"), score -> numeric.
    merged[SCORE_COL] = pd.to_numeric(merged[SCORE_COL], errors="coerce")
    merged = merged.dropna(subset=[SCORE_COL])
    merged[ID_COL] = merged[ID_COL].apply(lambda v: str(int(float(v))))
    merged = merged.drop_duplicates(subset=[ID_COL], keep="first").sort_values(ID_COL)

    args.out.parent.mkdir(parents=True, exist_ok=True)
    merged.to_csv(args.out, index=False)
    print(f"[WRITE] {len(merged)} labels (from {before} rows) -> {args.out}")
    print(f"[INFO]  PHQ8_Score range: {int(merged[SCORE_COL].min())}..{int(merged[SCORE_COL].max())}  "
          f"positives(>10)={int((merged[SCORE_COL] > 10).sum())}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
