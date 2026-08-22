#!/usr/bin/env python3
"""
scripts/rebuild_participant_only_parquet.py

Rebuilds the `text` column of dataset_build/daic_records_multimodal.parquet
from the raw DAIC-WOZ transcripts, keeping ONLY participant speech.

Fixes three compounding defects found in the existing text pipeline:
  (i)   speaker contamination - the frozen parquet's `text` field concatenates
        BOTH Ellie (virtual interviewer) and the participant, in session
        order, with Ellie's near-identical fixed script always first.
  (ii)  a wrong CSV separator (comma) used against transcripts that are
        actually tab-separated (present in build_daic_records.py and
        format_daic_to_lda.py) - this script parses with sep='\\t' explicitly.
  (iii) [not fixed here - see Step 2b] a 128-token truncation window applied
        later in the trainer, downstream of this script.

This script does NOT touch any existing builder, does NOT overwrite the
frozen dataset_build/daic_records_multimodal.parquet, and does NOT wire a new
parquet into the live pipeline. It only produces a new, separate file for
review:

    dataset_build/daic_records_multimodal_participant_only.parquet

Inputs (read-only):
    D:\\TeraBoxDownload\\Dataset\\<pid>_P.zip   raw DAIC-WOZ archives
        - opened read-only; the transcript member is read entirely into
          memory and parsed directly from the zip - nothing is ever
          extracted to disk.
    dataset_build/daic_records_multimodal.parquet   frozen, read-only
        - source of phq_score / features / audio_coverage / video_coverage,
          which are carried across UNCHANGED, joined on participant_id.
          Only `text` is rebuilt.

Usage:
    .venv\\Scripts\\python.exe scripts\\rebuild_participant_only_parquet.py

Exit codes:
    0 = built with zero failures, 1 = built but one or more participants
    failed (see [FAIL] lines), 2 = fatal setup error (nothing written)
"""

from __future__ import annotations

import sys
import zipfile
from pathlib import Path

import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

ARCHIVE_DIR = Path(r"D:\TeraBoxDownload\Dataset")
ORIGINAL_PARQUET = Path("dataset_build/daic_records_multimodal.parquet")
OUT_PARQUET = Path("dataset_build/daic_records_multimodal_participant_only.parquet")

EXPECTED_COLUMNS = {"start_time", "stop_time", "speaker", "value"}
SPEAKER_LABEL = "Participant"
OUTPUT_COLUMNS = ["participant_id", "text", "phq_score", "features", "audio_coverage", "video_coverage"]


def find_archives() -> list[Path]:
    return sorted(ARCHIVE_DIR.glob("*_P.zip"))


def participant_id_from_zip(zp: Path) -> str:
    return zp.stem.replace("_P", "")


def load_participant_text(zp: Path) -> tuple[str, str, int, int]:
    """
    Read <pid>_TRANSCRIPT.csv directly from the zip (in memory), filter to
    speaker == "Participant", join by start_time order.

    Returns (participant_id, text, participant_turn_count, empty_turns_dropped).
    Raises ValueError with a clear message on any structural problem -
    caller is responsible for catching and logging it ("fail loudly").
    """
    pid = participant_id_from_zip(zp)

    with zipfile.ZipFile(zp) as z:
        names = z.namelist()
        # Exclude macOS AppleDouble resource-fork siblings ("._<name>") and
        # __MACOSX/ entries - they match the same suffix but are not real CSV
        # data (participant 487's archive has exactly this: both
        # "._487_TRANSCRIPT.csv" and the real "487_TRANSCRIPT.csv").
        candidates = [
            n for n in names
            if n.lower().endswith("_transcript.csv")
            and not Path(n).name.startswith("._")
            and "__MACOSX" not in n
        ]
        if not candidates:
            raise ValueError(f"{pid}: no *_TRANSCRIPT.csv member in archive (members={names})")
        tname = candidates[0]
        with z.open(tname) as f:
            df = pd.read_csv(f, sep="\t")

    missing = EXPECTED_COLUMNS - set(df.columns)
    if missing:
        raise ValueError(
            f"{pid}: missing expected column(s) {sorted(missing)} after sep='\\t' parse; "
            f"found columns={list(df.columns)}"
        )

    participant_rows = df[df["speaker"] == SPEAKER_LABEL]
    if participant_rows.empty:
        found_speakers = sorted(df["speaker"].unique().tolist())
        raise ValueError(
            f"{pid}: zero rows with speaker == {SPEAKER_LABEL!r}; "
            f"speakers present in this transcript: {found_speakers}"
        )

    participant_rows = participant_rows.sort_values("start_time")

    # A handful of participant turns have NaN `value` in the raw corpus
    # (blank/inaudible turns). astype(str) on pandas' native "string" dtype
    # does NOT stringify these to "nan" the way it does for legacy "object"
    # dtype - it leaves them as an actual float, which crashes str.join().
    # Correct behaviour is to drop them (an empty turn contributes no text),
    # not crash and not inject the literal word "nan" into the corpus.
    non_empty = participant_rows[participant_rows["value"].notna()]
    empty_dropped = len(participant_rows) - len(non_empty)
    if non_empty.empty:
        raise ValueError(
            f"{pid}: all {len(participant_rows)} participant turn(s) have NaN value "
            f"(nothing usable after dropping empty turns)"
        )

    text = " ".join(non_empty["value"].astype(str).tolist()).strip()
    return pid, text, len(non_empty), empty_dropped


def main() -> int:
    archives = find_archives()
    print(f"[INFO] archive dir : {ARCHIVE_DIR}")
    print(f"[INFO] found {len(archives)} '<pid>_P.zip' archives")

    if not archives:
        print(f"[FATAL] no archives found under {ARCHIVE_DIR}")
        return 2

    if not ORIGINAL_PARQUET.exists():
        print(f"[FATAL] original parquet not found: {ORIGINAL_PARQUET}")
        return 2

    orig_table = pq.read_table(str(ORIGINAL_PARQUET))
    orig_df = orig_table.to_pandas()
    orig_df["participant_id"] = orig_df["participant_id"].astype(str)
    orig_lookup = orig_df.set_index("participant_id")
    print(f"[INFO] original parquet: {len(orig_df)} rows, columns={list(orig_df.columns)}")
    print()

    rows = []
    comparison_rows = []
    failures = []

    for zp in archives:
        pid_guess = participant_id_from_zip(zp)
        try:
            pid, new_text, turn_count, empty_dropped = load_participant_text(zp)
        except Exception as e:  # noqa: BLE001 - report and continue, this is a data audit
            msg = f"{pid_guess}: {e}"
            failures.append(msg)
            print(f"[FAIL] {msg}")
            continue

        if pid not in orig_lookup.index:
            msg = f"{pid}: present in raw archives but NOT in original parquet (participant_id mismatch)"
            failures.append(msg)
            print(f"[FAIL] {msg}")
            continue

        orig_row = orig_lookup.loc[pid]
        old_text = str(orig_row["text"])
        old_len = len(old_text)
        new_len = len(new_text)
        reduction_pct = (1.0 - new_len / old_len) * 100.0 if old_len else 0.0

        rows.append({
            "participant_id": pid,
            "text": new_text,
            "phq_score": orig_row["phq_score"],
            "features": orig_row["features"],
            "audio_coverage": orig_row["audio_coverage"],
            "video_coverage": orig_row["video_coverage"],
        })
        comparison_rows.append({
            "participant_id": pid,
            "old_chars": old_len,
            "new_chars": new_len,
            "reduction_pct": reduction_pct,
            "participant_turns": turn_count,
            "empty_turns_dropped": empty_dropped,
        })
        if empty_dropped:
            print(f"[INFO] {pid}: dropped {empty_dropped} participant turn(s) with NaN value")

    # Participants present in the original parquet but with no archive at all
    archive_pids = {participant_id_from_zip(z) for z in archives}
    missing_from_archives = sorted(set(orig_lookup.index) - archive_pids)
    for pid in missing_from_archives:
        msg = f"{pid}: present in original parquet but no <pid>_P.zip archive found"
        failures.append(msg)
        print(f"[FAIL] {msg}")

    # ---- Comparison table ----
    print()
    print(f"{'pid':<6} {'old_chars':>10} {'new_chars':>10} {'reduction%':>11} {'p_turns':>8}")
    print("-" * 50)
    total_old = total_new = total_turns = 0
    for r in sorted(comparison_rows, key=lambda x: int(x["participant_id"])):
        print(f"{r['participant_id']:<6} {r['old_chars']:>10} {r['new_chars']:>10} "
              f"{r['reduction_pct']:>10.1f}% {r['participant_turns']:>8}")
        total_old += r["old_chars"]
        total_new += r["new_chars"]
        total_turns += r["participant_turns"]

    n = len(comparison_rows)
    overall_reduction = (1.0 - total_new / total_old) * 100.0 if total_old else 0.0
    print("-" * 50)
    print(f"[TOTALS] rows built={n}  total_old_chars={total_old}  total_new_chars={total_new}  "
          f"overall_reduction={overall_reduction:.1f}%  total_participant_turns={total_turns}")

    if n != 188:
        print(f"[WARN] expected 188 rows (original row count), built {n}")
    if failures:
        print(f"\n[FAILURES] {len(failures)} participant(s) could not be processed:")
        for f in failures:
            print(f"  - {f}")
    else:
        print("\n[OK] zero failures across all participants")

    if not rows:
        print("[FATAL] no rows built - aborting without writing output")
        return 2

    out_df = pd.DataFrame(rows)[OUTPUT_COLUMNS]
    out_table = pa.Table.from_pandas(out_df, preserve_index=False)
    try:
        out_table = out_table.cast(orig_table.schema)
    except Exception as e:
        print(f"[FATAL] output schema does not match original schema and could not be cast: {e}")
        print(f"  original schema: {orig_table.schema}")
        print(f"  built schema   : {out_table.schema}")
        return 2

    OUT_PARQUET.parent.mkdir(parents=True, exist_ok=True)
    pq.write_table(out_table, str(OUT_PARQUET))
    print(f"\n[WRITE] {len(out_df)} rows -> {OUT_PARQUET}")
    print(f"[SCHEMA] {out_table.schema}")

    return 0 if not failures else 1


if __name__ == "__main__":
    sys.exit(main())
