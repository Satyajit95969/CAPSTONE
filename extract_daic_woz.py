#!/usr/bin/env python3
"""
extract_daic_woz.py  (Phase 5 - Step 2 staging helper)

Stages DAIC-WOZ archives into the per-participant folder layout that the
ORIGINAL loaders expect:

    <dest>/<pid>_P/<pid>_TRANSCRIPT.csv     (transcript-only, default)
    <dest>/<pid>_P/<all archive files>      (--full)

This is a NON-DESTRUCTIVE staging tool:
  * Source .zip archives are opened READ-ONLY and never modified or deleted.
  * Existing extracted files are skipped (idempotent) unless --force.
  * No source files of the project are touched.

It does NOT train, does NOT modify any loader, and does NOT change architecture.

Default (recommended for the current text loaders):
    python extract_daic_woz.py                 # transcript-only -> ./data
Validate on a single participant first:
    python extract_daic_woz.py --limit 1 --dry-run
Full multimodal stage (large; target a drive with space, e.g. D:):
    python extract_daic_woz.py --full --dest "D:/datasets/DAIC-WOZ/extracted"
"""

import argparse
import sys
import zipfile
from pathlib import Path

DEFAULT_SRC = Path("D:/datasets/DAIC-WOZ")
DEFAULT_DEST = Path("./data")
TRANSCRIPT_SUFFIX = "_TRANSCRIPT.csv"   # actual DAIC filename (uppercase)


def participant_id(zip_path: Path) -> str:
    # "300_P.zip" -> "300"
    return zip_path.stem.replace("_P", "")


def find_zips(src: Path):
    return sorted(src.glob("*_P.zip"))


def stage_transcript_only(zf: zipfile.ZipFile, pid: str, out_dir: Path,
                          force: bool, dry_run: bool) -> str:
    target = out_dir / f"{pid}_TRANSCRIPT.csv"
    if target.exists() and not force:
        return "skip (exists)"
    # locate the transcript entry case-insensitively
    entry = next((n for n in zf.namelist()
                  if n.lower().endswith("_transcript.csv")), None)
    if entry is None:
        return "MISSING transcript entry"
    if dry_run:
        return f"would write {target.name}"
    out_dir.mkdir(parents=True, exist_ok=True)
    with zf.open(entry) as fsrc:
        target.write_bytes(fsrc.read())
    return f"wrote {target.name}"


def stage_full(zf: zipfile.ZipFile, out_dir: Path,
               force: bool, dry_run: bool) -> str:
    names = [n for n in zf.namelist() if not n.endswith("/")]
    written = 0
    skipped = 0
    for n in names:
        target = out_dir / Path(n).name  # archives are flat at root
        if target.exists() and not force:
            skipped += 1
            continue
        if dry_run:
            written += 1
            continue
        out_dir.mkdir(parents=True, exist_ok=True)
        with zf.open(n) as fsrc:
            target.write_bytes(fsrc.read())
        written += 1
    verb = "would write" if dry_run else "wrote"
    return f"{verb} {written} file(s), skipped {skipped}"


def main() -> int:
    ap = argparse.ArgumentParser(description="DAIC-WOZ staging helper (Phase 5 Step 2)")
    ap.add_argument("--src", type=Path, default=DEFAULT_SRC,
                    help=f"Folder containing *_P.zip archives (default: {DEFAULT_SRC})")
    ap.add_argument("--dest", type=Path, default=DEFAULT_DEST,
                    help=f"Destination root for <pid>_P folders (default: {DEFAULT_DEST})")
    ap.add_argument("--full", action="store_true",
                    help="Extract ALL files (large). Default extracts transcript only.")
    ap.add_argument("--limit", type=int, default=0,
                    help="Process only the first N archives (0 = all). Use for validation.")
    ap.add_argument("--force", action="store_true",
                    help="Overwrite existing extracted files.")
    ap.add_argument("--dry-run", action="store_true",
                    help="Report actions without writing anything.")
    args = ap.parse_args()

    if not args.src.exists():
        print(f"[ERROR] source not found: {args.src}")
        return 2

    zips = find_zips(args.src)
    if args.limit > 0:
        zips = zips[:args.limit]
    if not zips:
        print(f"[ERROR] no *_P.zip archives found in {args.src}")
        return 2

    mode = "FULL" if args.full else "TRANSCRIPT-ONLY"
    print(f"[INFO] mode={mode}  src={args.src}  dest={args.dest.resolve()}  "
          f"archives={len(zips)}  dry_run={args.dry_run}")

    ok = miss = err = 0
    for zp in zips:
        pid = participant_id(zp)
        out_dir = args.dest / f"{pid}_P"
        try:
            with zipfile.ZipFile(zp, "r") as zf:   # read-only
                if args.full:
                    msg = stage_full(zf, out_dir, args.force, args.dry_run)
                else:
                    msg = stage_transcript_only(zf, pid, out_dir, args.force, args.dry_run)
            if "MISSING" in msg:
                miss += 1
                print(f"[WARN] {pid}_P: {msg}")
            else:
                ok += 1
                print(f"[ OK ] {pid}_P: {msg}")
        except zipfile.BadZipFile:
            err += 1
            print(f"[FAIL] {pid}_P: corrupt/unreadable archive")
        except Exception as e:  # noqa: BLE001 - report and continue staging
            err += 1
            print(f"[FAIL] {pid}_P: {type(e).__name__}: {e}")

    print(f"\n[SUMMARY] processed={len(zips)} ok={ok} missing_transcript={miss} errors={err}")
    if not args.dry_run and not args.full:
        present = len(list(args.dest.glob(f"*_P/*{TRANSCRIPT_SUFFIX}")))
        print(f"[VALIDATE] transcripts present under {args.dest}: {present}")
    return 0 if err == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
