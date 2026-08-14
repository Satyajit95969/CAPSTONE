#!/usr/bin/env python3
"""
verify_daic_records.py  (Phase 6 - Step 6 helper)

Verifies that a records file satisfies trainer_mentalbert_daic.py (Track C)
WITHOUT modifying the trainer and WITHOUT training or downloading MentalBERT.

It imports the trainer module and calls the trainer's OWN read_parquet_records()
so the parsing path is exactly what training will use. It then replays the
trainer's field-access and audio/vision dim-inference logic (copied read-only from
trainer_mentalbert_daic.py main(): lines 147-152, 533-549) to confirm every record
yields a usable (text, phq, label) example. No model/tokenizer is constructed, so
nothing is downloaded.

Usage:
    python verify_daic_records.py ./data/daic_records.parquet
"""

import argparse
import importlib.util
import sys
from pathlib import Path

import pandas as pd

THIS_DIR = Path(__file__).resolve().parent
TRAINER_PATH = THIS_DIR / "trainer_mentalbert_daic.py"


def load_trainer_module():
    # Import the existing trainer as a module (does not run main()).
    spec = importlib.util.spec_from_file_location("trainer_mentalbert_daic", TRAINER_PATH)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def infer_dims(records):
    # Verbatim logic from trainer main() lines 533-549.
    audio_dim = None
    vision_dim = None
    for r in records:
        f = r.get("features") or {}
        if isinstance(f, dict) and "audio" in f:
            a = f["audio"]
            if isinstance(a, dict) and a.get("wav2vec2"):
                audio_dim = len(a["wav2vec2"]); break
            if isinstance(a, dict) and a.get("egemaps"):
                audio_dim = len(a["egemaps"].keys()); break
    for r in records:
        f = r.get("features") or {}
        if isinstance(f, dict) and "video" in f and isinstance(f["video"], dict) and f["video"].get("densenet"):
            vision_dim = len(f["video"]["densenet"]); break
        if any(str(k).startswith("neuron_") for k in r.keys()):
            neuron_keys = sorted([kk for kk in r.keys() if str(kk).startswith("neuron_")])
            vision_dim = len(neuron_keys); break
    return audio_dim, vision_dim


def check_record(r):
    # Mirrors MultiModalDataset.__getitem__ field access (lines 141-152).
    text = (r.get("transcript") or r.get("text") or "")
    text = str(text).strip()
    # Presence-by-KEY (not Python truthiness): a legitimate PHQ8_Score of 0.0 is
    # falsy, so the trainer's `a or b or ...` chain collapses it to None. Here we
    # select the first phq field that is actually present and non-NaN so that a
    # zero score counts as parseable. phq_val/label still match the trainer exactly.
    phq = next((r[k] for k in ("phq_score", "phq", "target_phq", "label_phq")
                if k in r and r[k] is not None
                and not (isinstance(r[k], float) and pd.isna(r[k]))), None)
    try:
        phq_val = float(phq) if phq is not None else 0.0
        phq_ok = phq is not None
    except Exception:
        phq_val = 0.0
        phq_ok = False
    label = 1 if phq_val > 10.0 else 0
    return text, phq_val, label, phq_ok, bool(text)


def main() -> int:
    ap = argparse.ArgumentParser(description="Verify Track-C records file (Phase 6 Step 6)")
    ap.add_argument("path", help="records file (.parquet/.json/.jsonl/.csv)")
    args = ap.parse_args()

    p = Path(args.path)
    if not p.exists():
        print(f"[FAIL] file not found: {p}")
        return 2

    T = load_trainer_module()
    print(f"[OK] imported trainer module: {TRAINER_PATH.name}")

    # Use the trainer's OWN loader — strongest possible compatibility check.
    records = T.read_parquet_records(str(p))
    print(f"[OK] trainer.read_parquet_records loaded {len(records)} records")

    if not records:
        print("[FAIL] zero records")
        return 1

    n_text_ok = n_phq_ok = pos = neg = 0
    bad = []
    for i, r in enumerate(records):
        text, phq_val, label, phq_ok, text_ok = check_record(r)
        n_text_ok += text_ok
        n_phq_ok += phq_ok
        pos += (label == 1)
        neg += (label == 0)
        if not text_ok or not phq_ok:
            bad.append((i, r.get("participant_id"), text_ok, phq_ok))

    audio_dim, vision_dim = infer_dims(records)

    print("\n=== VERIFICATION ===")
    print(f"records            : {len(records)}")
    print(f"non-empty text     : {n_text_ok}/{len(records)}")
    print(f"parseable phq      : {n_phq_ok}/{len(records)}")
    print(f"label balance      : pos(>10)={pos}  neg={neg}")
    print(f"inferred audio_dim : {audio_dim}   (None => text-only, expected for first run)")
    print(f"inferred vision_dim: {vision_dim}   (None => text-only, expected for first run)")

    # Sample preview
    s = records[0]
    stext = str(s.get("transcript") or s.get("text") or "")
    print("\n[sample row 0]")
    print(f"  participant_id : {s.get('participant_id')}")
    print(f"  phq_score      : {s.get('phq_score')}")
    print(f"  text[:100]     : {stext[:100]!r}")

    if bad:
        print(f"\n[FAIL] {len(bad)} record(s) missing text or phq: {bad[:10]}")
        return 1
    print("\n[PASS] file satisfies trainer_mentalbert_daic.py (Track C) schema.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
