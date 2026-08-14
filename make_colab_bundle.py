#!/usr/bin/env python
"""
make_colab_bundle.py — build the Phase 11.6 Baseline-CV Colab upload bundle.

Why this exists instead of Compress-Archive:
  Windows PowerShell 5.1's Compress-Archive stores entry names with BACKSLASH
  separators, violating ZIP APPNOTE 4.4.17.1 (which mandates '/'). Windows
  extractors tolerate it; Linux/Colab does not — it treats the backslash as a
  literal filename character and produces flat files named
  "trainer_outputs\\baseline_cv\\fold_manifest.json".

This writer sets every arcname explicitly with forward slashes, then re-reads
the RAW central directory to prove no backslash survived.

Usage:  python make_colab_bundle.py
"""
import hashlib
import io
import json
import os
import re
import sys
import zipfile

BS = chr(92)  # backslash
OUT = "baseline_cv_colab.zip"
FROZEN_TRAINER_SHA = "65b1902e2ba8dd996c6960db1a6595d84fd48500f4d8707cb79688093d7a230b"

# (source path on disk, arcname inside the zip — ALWAYS forward slashes)
FILES = [
    ("trainer_mentalbert_daic.py", "trainer_mentalbert_daic.py"),
    ("daic_records.parquet", "daic_records.parquet"),
    ("colab_baseline_cv/run_baseline_cv.py", "colab_baseline_cv/run_baseline_cv.py"),
    ("colab_baseline_cv/aggregate_baseline_cv.py", "colab_baseline_cv/aggregate_baseline_cv.py"),
    ("colab_baseline_cv/PHASE_11_BASELINE_CV_AUDIT.md", "colab_baseline_cv/PHASE_11_BASELINE_CV_AUDIT.md"),
    ("colab_baseline_cv/PHASE_11_BASELINE_CV_COLAB_PACKAGE.md", "colab_baseline_cv/PHASE_11_BASELINE_CV_COLAB_PACKAGE.md"),
    ("trainer_outputs/baseline_cv/fold_manifest.json", "trainer_outputs/baseline_cv/fold_manifest.json"),
    ("trainer_outputs/baseline_cv/trivial_control_arm.csv", "trainer_outputs/baseline_cv/trivial_control_arm.csv"),
]


def fail(msg):
    print("[FAIL] " + msg)
    sys.exit(1)


def main():
    missing = [src for src, _ in FILES if not os.path.isfile(src)]
    if missing:
        fail("missing source file(s): " + ", ".join(missing) +
             "\n       Run this from the repository root.")

    if os.path.exists(OUT):
        os.remove(OUT)

    with zipfile.ZipFile(OUT, "w", zipfile.ZIP_DEFLATED) as z:
        for src, arc in FILES:
            if BS in arc:
                fail("arcname contains a backslash: " + arc)
            z.write(src, arcname=arc)

    # ---- verify: read the RAW central directory, bypassing zipfile's
    # ---- os.sep normalisation (which silently hides this defect on Windows)
    raw = open(OUT, "rb").read()
    names = []
    for m in re.finditer(b"PK\x01\x02", raw):
        off = m.start()
        n = int.from_bytes(raw[off + 28:off + 30], "little")
        names.append(raw[off + 46:off + 46 + n].decode("utf-8", "replace"))

    bad = [n for n in names if BS in n]
    if bad:
        fail("backslash separators still present: " + repr(bad))
    if len(names) != len(FILES):
        fail("expected %d entries, found %d" % (len(FILES), len(names)))

    # ---- verify payload integrity from inside the archive
    with zipfile.ZipFile(OUT) as z:
        if z.testzip() is not None:
            fail("archive is corrupt")
        sha = hashlib.sha256(z.read("trainer_mentalbert_daic.py")).hexdigest()
        man = json.loads(z.read("trainer_outputs/baseline_cv/fold_manifest.json"))
        try:
            import pyarrow.parquet as pq
            df = pq.read_table(io.BytesIO(z.read("daic_records.parquet"))).to_pandas()
            rows = len(df)
            pos = int((df["phq_score"] > 10).sum())
            neg = int((df["phq_score"] <= 10).sum())
        except Exception as exc:  # pyarrow absent — not fatal for packaging
            rows = pos = neg = "?"
            print("[WARN] could not read parquet inside zip: %s" % exc)

    print("=== RAW entry names in central directory ===")
    for n in sorted(names):
        print("    " + n)

    ok_sha = (sha == FROZEN_TRAINER_SHA)
    ok_man = (man.get("n_participants") == 188 and len(man.get("folds", [])) == 5)
    ok_data = (rows == 188 and pos == 45 and neg == 143)

    print()
    print("------ BUNDLE VERIFICATION ------")
    print("archive          : %s (%s bytes)" % (OUT, format(os.path.getsize(OUT), ",")))
    print("entries          : %d" % len(names))
    print("backslashes      : 0")
    print("SPEC-COMPLIANT   : True")
    print("trainer SHA      : %s" % sha)
    print("SHA frozen       : %s" % ok_sha)
    print("manifest         : n=%s folds=%s" % (man.get("n_participants"), len(man.get("folds", []))))
    print("dataset          : rows=%s pos=%s neg=%s" % (rows, pos, neg))
    print("READY TO UPLOAD  : %s" % (ok_sha and ok_man and ok_data))

    if not (ok_sha and ok_man and ok_data):
        sys.exit(1)


if __name__ == "__main__":
    main()
