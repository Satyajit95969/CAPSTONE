#!/usr/bin/env python3
"""
build_daic_records.py  (Phase 6 - Step 6 helper; Track C input builder)

Builds the PLAINTEXT records file consumed by trainer_mentalbert_daic.py (Track C)
from already-staged DAIC-WOZ transcripts and the PHQ-8 label sheet:

    ./data/<pid>_P/<pid>_TRANSCRIPT.csv   (staged by extract_daic_woz.py)
    ./labels/Detailed_PHQ8_Labels.csv     (PHQ-8 labels)
        |
        v
    ./data/daic_records.parquet           (one row per participant; text + phq_score)

This is the "missing preparation stage" identified in Phase 6 Steps 3-5. It mirrors
format_daic_to_lda.py Steps 1-2 EXACTLY (same column-detection heuristics, same
whole-session transcript concatenation) but deliberately:
  * stops BEFORE the MentalBERT embedding step (Track C tokenizes raw text itself), and
  * writes a PLAINTEXT parquet (NO SecureStore / NO .enc — Track C cannot read an
    encrypted wrapper).

It does NOT train, does NOT modify any existing source file, and does NOT change
architecture. Raw phq_score is stored unbinarized; the trainer derives the label.

Usage:
    python build_daic_records.py                       # ./data + ./labels -> ./data/daic_records.parquet
    python build_daic_records.py --limit 2 --dry-run   # validate on first 2 participants
    python build_daic_records.py --format jsonl --out ./data/daic_records.jsonl
"""

import argparse
import json
import sys
from pathlib import Path

import pandas as pd


def find_transcript_csv(folder: Path) -> Path | None:
    """
    Case-INSENSITIVE lookup of the participant transcript inside <pid>_P/.

    extract_daic_woz.py writes "<pid>_TRANSCRIPT.csv" (uppercase) while
    format_daic_to_lda.py reads "<pid>_Transcript.csv" (mixed case). On Windows
    these resolve to the same file, but Colab/Linux is case-SENSITIVE, so we match
    on a lowercased suffix instead of a fixed-case filename.
    """
    for child in sorted(folder.iterdir()):
        if child.is_file() and child.name.lower().endswith("_transcript.csv"):
            return child
    return None


def pick_text_column(df) -> str | None:
    # Same heuristic as format_daic_to_lda.py line 91: prefer "text" or "value".
    return next((c for c in df.columns if "text" in c.lower() or "value" in c.lower()), None)


def load_phq_map(labels_path: Path):
    if not labels_path.exists():
        raise FileNotFoundError(f"Missing PHQ label file at {labels_path}")
    labels_df = pd.read_csv(labels_path)
    phq_col = next((c for c in labels_df.columns if "PHQ" in c.upper()), None)
    if not phq_col:
        raise KeyError("No PHQ score column found in labels file!")
    id_col = next((c for c in labels_df.columns if "Participant" in c or "ID" in c), None)
    if not id_col:
        raise KeyError("No participant ID column found!")
    phq_map = dict(zip(labels_df[id_col].astype(str), labels_df[phq_col]))
    return phq_map, id_col, phq_col


def build_records(data_dir: Path, labels_path: Path, limit: int):
    phq_map, id_col, phq_col = load_phq_map(labels_path)
    print(f"[INFO] labels: id_col={id_col!r} phq_col={phq_col!r} entries={len(phq_map)}")

    records = []
    skipped_no_phq = skipped_no_file = skipped_no_textcol = skipped_empty = 0

    for folder in sorted(data_dir.glob("*_P")):
        if limit and len(records) >= limit:
            break
        pid = folder.name.replace("_P", "")

        phq = phq_map.get(pid)
        if phq is None or (isinstance(phq, float) and pd.isna(phq)):
            skipped_no_phq += 1
            print(f"[skip] {pid}: no PHQ label")
            continue

        tpath = find_transcript_csv(folder)
        if tpath is None:
            skipped_no_file += 1
            print(f"[skip] {pid}: no *_transcript.csv in {folder.name}")
            continue

        try:
            df_t = pd.read_csv(tpath)
        except Exception as e:  # noqa: BLE001 - report and continue
            skipped_no_file += 1
            print(f"[skip] {pid}: unreadable transcript ({type(e).__name__}: {e})")
            continue

        text_col = pick_text_column(df_t)
        if not text_col:
            skipped_no_textcol += 1
            print(f"[skip] {pid}: no text/value column (cols={list(df_t.columns)})")
            continue

        full_text = " ".join(df_t[text_col].astype(str).tolist()).strip()
        if not full_text:
            skipped_empty += 1
            print(f"[skip] {pid}: empty transcript")
            continue

        # RAW phq_score (NOT binarized). Track C derives label = 1 if phq > 10.0.
        records.append({
            "participant_id": pid,
            "text": full_text,
            "phq_score": float(phq),
        })

    print(f"[SUMMARY] built={len(records)}  skipped: no_phq={skipped_no_phq} "
          f"no_file={skipped_no_file} no_textcol={skipped_no_textcol} empty={skipped_empty}")
    return records


def write_records(records, out_path: Path, fmt: str):
    out_path.parent.mkdir(parents=True, exist_ok=True)
    if fmt == "parquet":
        # Plaintext parquet via pandas/pyarrow. No encryption.
        pd.DataFrame(records).to_parquet(out_path, index=False)
    elif fmt == "jsonl":
        with open(out_path, "w", encoding="utf-8") as f:
            for r in records:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")
    elif fmt == "csv":
        pd.DataFrame(records).to_csv(out_path, index=False)
    else:
        raise ValueError(f"unknown format: {fmt}")
    print(f"[WRITE] {len(records)} records -> {out_path}  (plaintext {fmt})")


def main() -> int:
    ap = argparse.ArgumentParser(description="Build Track-C daic_records file (Phase 6 Step 6)")
    ap.add_argument("--data-dir", type=Path, default=Path("./data"),
                    help="Root containing <pid>_P/ folders (default: ./data)")
    ap.add_argument("--labels", type=Path, default=Path("./labels/Detailed_PHQ8_Labels.csv"),
                    help="PHQ-8 labels CSV (default: ./labels/Detailed_PHQ8_Labels.csv)")
    ap.add_argument("--out", type=Path, default=Path("./data/daic_records.parquet"),
                    help="Output records file (default: ./data/daic_records.parquet)")
    ap.add_argument("--format", choices=["parquet", "jsonl", "csv"], default="parquet")
    ap.add_argument("--limit", type=int, default=0, help="Only first N participants (0 = all)")
    ap.add_argument("--dry-run", action="store_true", help="Build in memory but do not write")
    args = ap.parse_args()

    if not args.data_dir.exists():
        print(f"[ERROR] data dir not found: {args.data_dir}")
        return 2

    records = build_records(args.data_dir, args.labels, args.limit)
    if not records:
        print("[ERROR] no records built — nothing to write")
        return 1

    if args.dry_run:
        print("[DRY-RUN] not writing. First record preview:")
        preview = dict(records[0])
        preview["text"] = preview["text"][:120] + (" ..." if len(preview["text"]) > 120 else "")
        print(json.dumps(preview, ensure_ascii=False, indent=2))
        return 0

    write_records(records, args.out, args.format)
    return 0


if __name__ == "__main__":
    sys.exit(main())
