#!/usr/bin/env python3
"""
normalize_transcripts.py  (Phase 6 - Step 8 staging helper)

DAIC-WOZ transcript CSVs are TAB-delimited with columns:
    start_time <TAB> stop_time <TAB> speaker <TAB> value

build_daic_records.py reads transcripts with pandas' default (comma) parser and
picks the column whose name contains "text"/"value". On a tab file that parser
yields a SINGLE column, so timestamps + speaker labels get embedded into the text
fed to MentalBERT. This helper normalizes the staged transcripts into clean,
comma-delimited, value-only CSVs in a SEPARATE destination tree so that
build_daic_records.py consumes clean utterance text with NO change to that script.

Non-destructive: reads ./data/<pid>_P/<pid>_TRANSCRIPT.csv (raw staged) and writes
./data_norm/<pid>_P/<pid>_TRANSCRIPT.csv (clean). Idempotent; raw files untouched.
Preserves Ritik's whole-session, NO-speaker-filter convention (all rows kept).

Usage:
    python normalize_transcripts.py                       # ./data -> ./data_norm
    python normalize_transcripts.py --limit 2 --dry-run
"""

import argparse
import sys
from pathlib import Path

import pandas as pd


def find_transcript_csv(folder: Path):
    # Case-insensitive (TRANSCRIPT.csv vs Transcript.csv) — same rule as build_daic_records.py.
    for child in sorted(folder.iterdir()):
        if child.is_file() and child.name.lower().endswith("_transcript.csv"):
            return child
    return None


def read_transcript(tpath: Path) -> pd.DataFrame:
    # DAIC transcripts are tab-delimited. Read explicitly; fall back to a sniffing
    # parser if the expected 'value' column is absent.
    df = pd.read_csv(tpath, sep="\t")
    if "value" not in [c.strip().lower() for c in df.columns]:
        df = pd.read_csv(tpath, sep=None, engine="python")
    df.columns = [c.strip() for c in df.columns]
    return df


def normalize_one(tpath: Path) -> pd.DataFrame | None:
    df = read_transcript(tpath)
    val_col = next((c for c in df.columns if c.lower() == "value"), None)
    if val_col is None:
        val_col = next((c for c in df.columns if "value" in c.lower() or "text" in c.lower()), None)
    if val_col is None:
        return None
    spk_col = next((c for c in df.columns if c.lower() == "speaker"), None)

    out = pd.DataFrame()
    if spk_col is not None:
        out["speaker"] = df[spk_col].astype(str)
    out["value"] = df[val_col].astype(str)
    # Drop empty / NaN utterances so they don't become "nan" tokens downstream.
    out = out[out["value"].str.strip().ne("") & out["value"].str.strip().str.lower().ne("nan")]
    return out.reset_index(drop=True)


def main() -> int:
    ap = argparse.ArgumentParser(description="Normalize DAIC tab-transcripts -> clean CSV (Phase 6 Step 8)")
    ap.add_argument("--src", type=Path, default=Path("./data"),
                    help="Root containing staged <pid>_P/ folders (default: ./data)")
    ap.add_argument("--dest", type=Path, default=Path("./data_norm"),
                    help="Output root for normalized transcripts (default: ./data_norm)")
    ap.add_argument("--limit", type=int, default=0, help="Only first N participants (0 = all)")
    ap.add_argument("--dry-run", action="store_true", help="Report actions without writing")
    args = ap.parse_args()

    if not args.src.exists():
        print(f"[ERROR] src not found: {args.src}")
        return 2

    folders = sorted(args.src.glob("*_P"))
    if args.limit:
        folders = folders[:args.limit]
    if not folders:
        print(f"[ERROR] no <pid>_P folders under {args.src}")
        return 2

    ok = miss = err = 0
    for folder in folders:
        pid = folder.name.replace("_P", "")
        tpath = find_transcript_csv(folder)
        if tpath is None:
            miss += 1
            print(f"[skip] {pid}: no *_transcript.csv")
            continue
        try:
            out = normalize_one(tpath)
        except Exception as e:  # noqa: BLE001 - report and continue
            err += 1
            print(f"[FAIL] {pid}: {type(e).__name__}: {e}")
            continue
        if out is None or out.empty:
            miss += 1
            print(f"[skip] {pid}: no usable 'value' rows")
            continue

        dest_dir = args.dest / folder.name
        dest_file = dest_dir / f"{pid}_TRANSCRIPT.csv"
        if args.dry_run:
            print(f"[dry] {pid}: {len(out)} rows -> {dest_file}")
        else:
            dest_dir.mkdir(parents=True, exist_ok=True)
            out.to_csv(dest_file, index=False)   # proper comma CSV with quoting
            print(f"[ OK ] {pid}: {len(out)} rows -> {dest_file}")
        ok += 1

    print(f"\n[SUMMARY] normalized={ok} skipped={miss} errors={err}  dest={args.dest}")
    return 0 if err == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
