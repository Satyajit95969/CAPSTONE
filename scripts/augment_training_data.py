#!/usr/bin/env python3
"""
scripts/augment_training_data.py

Augments the TRAINING partition only of
dataset_build/daic_records_multimodal_participant_only.parquet, to test
whether more training data changes this pipeline's genuine-discrimination
rate (see docs/IMPLEMENTATION_NOTES.md, "Fusion-head capacity sweep" and
the 55-run n=15 sweep this script's output is meant to be compared against).

Produces TWO new parquet files, never touching the original:

  Arm 2 (text windowing only):
    dataset_build/daic_records_multimodal_participant_only_windowed.parquet
    = 186 real rows + windowed rows derived from eligible TRAIN rows only.
    Class ratio unchanged (~29.5% positive) - windowing is applied
    proportionally to both classes.

  Arm 3 (windowing + minority SMOTE-style interpolation):
    dataset_build/daic_records_multimodal_participant_only_augmented.parquet
    = everything in Arm 2 + synthetic positive rows from feature-space
    interpolation between pairs of real positive TRAIN records.
    Class ratio shifted toward ~45% positive.

LEAKAGE SAFETY (non-negotiable, enforced structurally AND asserted):
  - stratified_split() (imported from the live trainer module, unmodified
    logic) is run FIRST, exactly as the live pipeline runs it, to get the
    real 149 train / 37 eval partition.
  - Every augmentation technique below reads ONLY from train_records.
    eval_records is never touched, never read for augmentation inputs -
    it is held in a separate variable for the final leakage-safety
    assertions only.
  - stratified_split() itself was changed (2026-10-02,
    installer/runtime/agents/trainer/trainer_mentalbert_privacy.py) to
    restrict eval candidates to non-augmented records, specifically so
    that loading either output file through the unmodified live pipeline
    reproduces the identical 37 held-out IDs - proven, not just asserted,
    both here and in IMPLEMENTATION_NOTES.md.

LABELS: every augmented row's phq_score is a real, unperturbed value
copied from one real parent record - never generated, never interpolated,
never inferred. The binary label (phq_score >= 10.0) therefore always
matches a real patient's real ground truth.

PROVENANCE: two new columns on every row (originals included):
  - is_augmented (bool): False for all 186 real rows, True for synthetic
    rows.
  - source_record_id (str): for a real row, its own participant_id. For a
    windowed row, the single real participant_id it was derived from. For
    a SMOTE row (derived from TWO real parents), "pidA+pidB" - documented
    here since the spec anticipates one source id but interpolation
    inherently has two.

Usage:
    .venv\\Scripts\\python.exe scripts\\augment_training_data.py
"""

from __future__ import annotations

import json
import random
import sys
from pathlib import Path
from typing import Any, Dict, List, Tuple

import pandas as pd

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(REPO_ROOT / "installer" / "runtime"))

import agents.trainer.trainer_mentalbert_privacy as T  # noqa: E402

SOURCE_PARQUET = REPO_ROOT / "dataset_build" / "daic_records_multimodal_participant_only.parquet"
WINDOWED_OUT = REPO_ROOT / "dataset_build" / "daic_records_multimodal_participant_only_windowed.parquet"
AUGMENTED_OUT = REPO_ROOT / "dataset_build" / "daic_records_multimodal_participant_only_augmented.parquet"

# Separate, clearly-labeled seed for augmentation randomness (which pairs to
# interpolate, which alpha) - distinct from EVAL_SPLIT_SEED (42, used by
# stratified_split itself) so the two are never confused, though both use
# the project's standard value of 42 for reproducibility.
AUGMENT_SEED = 42
WINDOW_TOKENS = 512          # matches MULTIMODAL_MAX_LEN
MIN_TOKENS_FOR_SECOND_WINDOW = 1024   # need two full, non-overlapping windows
N_SMOTE_POSITIVE = 74        # see docs/IMPLEMENTATION_NOTES.md for the arithmetic
SMOTE_ALPHA_RANGE = (0.2, 0.8)   # avoid near-duplicate interpolation near the endpoints

EXPECTED_HELD_OUT_IDS = {
    "301", "302", "306", "316", "318", "326", "329", "334", "335", "338",
    "347", "349", "351", "352", "355", "364", "374", "377", "388", "393",
    "399", "403", "404", "411", "415", "441", "442", "448", "451", "452",
    "461", "469", "479", "480", "482", "485", "489",
}


def load_records() -> List[Dict[str, Any]]:
    df = pd.read_parquet(SOURCE_PARQUET)
    records = df.to_dict(orient="records")
    for r in records:
        r["is_augmented"] = False
        r["source_record_id"] = r["participant_id"]
    return records


def get_tokenizer():
    from transformers import AutoTokenizer
    try:
        return AutoTokenizer.from_pretrained(T.MENTALBERT_PRETRAIN)
    except Exception as e:
        print(f"[augment] Could not load MentalBERT tokenizer ({e}); "
              f"falling back to bert-base-uncased (same WordPiece vocab family, "
              f"adequate for computing a real second text window).")
        return AutoTokenizer.from_pretrained("bert-base-uncased")


def make_windowed_rows(train_records: List[Dict[str, Any]], tokenizer) -> List[Dict[str, Any]]:
    """
    For each train record whose real transcript has >= MIN_TOKENS_FOR_SECOND_WINDOW
    tokens, emit one new row whose text is the SECOND real, non-overlapping
    512-token window of that SAME transcript (tokens[512:1024], decoded back
    to text) - genuinely different real speech the participant actually
    produced, not paraphrased or invented. Audio/video features, phq_score,
    and coverage fields are copied unchanged from the source row, since they
    describe the whole session, not a sub-span of it.
    """
    out = []
    eligible = 0
    for r in train_records:
        ids = tokenizer.encode(r["text"], add_special_tokens=False)
        if len(ids) < MIN_TOKENS_FOR_SECOND_WINDOW:
            continue
        eligible += 1
        window2_ids = ids[WINDOW_TOKENS:2 * WINDOW_TOKENS]
        window2_text = tokenizer.decode(window2_ids, skip_special_tokens=True)

        new_row = dict(r)  # copies features (same JSON string), phq_score, coverage, etc.
        new_row["text"] = window2_text
        new_row["participant_id"] = f"aug_win_{r['participant_id']}"
        new_row["is_augmented"] = True
        new_row["source_record_id"] = r["participant_id"]
        out.append(new_row)

    print(f"[augment] text windowing: {eligible}/{len(train_records)} train records "
          f"eligible (>= {MIN_TOKENS_FOR_SECOND_WINDOW} tokens) -> {len(out)} new rows")
    return out


def make_smote_rows(train_records: List[Dict[str, Any]], n: int, seed: int) -> List[Dict[str, Any]]:
    """
    Minority-only SMOTE-style interpolation: pick two DISTINCT real positive
    train records, interpolate their 154-dim audio and 84-dim video feature
    vectors at a random alpha in SMOTE_ALPHA_RANGE, keep the resulting row
    labeled with the SAME real phq_score/text/coverage as whichever parent
    alpha favors (alpha >= 0.5 -> parent A, else parent B) - a single,
    coherent, real provenance for every non-interpolated field, never an
    invented or interpolated label.
    """
    positives = [r for r in train_records if float(r["phq_score"]) >= T.PHQ_POSITIVE_THRESHOLD]
    assert len(positives) >= 2, "Need at least 2 positive train records to interpolate between."

    rng = random.Random(seed)
    out = []
    for i in range(n):
        a, b = rng.sample(positives, 2)
        alpha = rng.uniform(*SMOTE_ALPHA_RANGE)

        feat_a = json.loads(a["features"])
        feat_b = json.loads(b["features"])
        audio_a, audio_b = feat_a["audio"]["wav2vec2"], feat_b["audio"]["wav2vec2"]
        video_a, video_b = feat_a["video"]["densenet"], feat_b["video"]["densenet"]
        assert len(audio_a) == len(audio_b) == 154
        assert len(video_a) == len(video_b) == 84

        interp_audio = [alpha * x + (1 - alpha) * y for x, y in zip(audio_a, audio_b)]
        interp_video = [alpha * x + (1 - alpha) * y for x, y in zip(video_a, video_b)]

        primary, primary_feat = (a, feat_a) if alpha >= 0.5 else (b, feat_b)
        new_features = {
            "audio": {"spec": primary_feat["audio"]["spec"], "wav2vec2": interp_audio},
            "video": {"spec": primary_feat["video"]["spec"], "densenet": interp_video},
        }

        new_row = dict(primary)
        new_row["features"] = json.dumps(new_features)
        new_row["participant_id"] = f"aug_smote_{i:04d}"
        new_row["is_augmented"] = True
        new_row["source_record_id"] = f"{a['participant_id']}+{b['participant_id']}"
        out.append(new_row)

    print(f"[augment] SMOTE-style minority interpolation: {len(positives)} real positive "
          f"train records available, generated {len(out)} synthetic positive rows")
    return out


def assert_leakage_safe(all_rows: List[Dict[str, Any]], held_out_ids: set) -> None:
    for r in all_rows:
        if r["is_augmented"]:
            sources = r["source_record_id"].split("+")
            for s in sources:
                assert s not in held_out_ids, (
                    f"LEAKAGE: augmented row {r['participant_id']} derives from "
                    f"held-out record {s}."
                )
        else:
            assert r["source_record_id"] == r["participant_id"], (
                f"Real row {r['participant_id']} has an unexpected source_record_id "
                f"{r['source_record_id']!r} (expected it to equal its own id)."
            )


def verify_split_reproduces(all_rows: List[Dict[str, Any]], expected_train_n: int, label: str) -> None:
    """Re-run the (patched) stratified_split on the FULL output pool and
    confirm it still recovers exactly the original 37 held-out ids, proving
    the stratified_split() fix works on THIS file, not just the original."""
    train, ev = T.stratified_split(all_rows)
    ev_ids = {r["participant_id"] for r in ev}
    assert ev_ids == EXPECTED_HELD_OUT_IDS, (
        f"[{label}] stratified_split() on the augmented pool returned a DIFFERENT "
        f"eval set than the original 37. Got: {sorted(ev_ids)}"
    )
    assert all(not r["is_augmented"] for r in ev), f"[{label}] an augmented row leaked into eval."
    assert len(train) == expected_train_n, (
        f"[{label}] expected {expected_train_n} train rows, got {len(train)}."
    )
    print(f"[augment] [{label}] VERIFIED: stratified_split() on the {len(all_rows)}-row output "
          f"reproduces the exact same 37 held-out ids; train={len(train)}, eval={len(ev)}.")


def write_parquet(rows: List[Dict[str, Any]], path: Path) -> None:
    df = pd.DataFrame(rows)
    # Keep dtypes byte-compatible with the original: phq_score/coverage as
    # float64, everything else as string/object, is_augmented as bool.
    df["phq_score"] = df["phq_score"].astype("float64")
    df["audio_coverage"] = df["audio_coverage"].astype("float64")
    df["video_coverage"] = df["video_coverage"].astype("float64")
    df["is_augmented"] = df["is_augmented"].astype(bool)
    df.to_parquet(path, index=False)
    print(f"[augment] wrote {len(df)} rows -> {path}")


def main() -> int:
    records = load_records()
    train_records, eval_records = T.stratified_split(records)
    assert len(train_records) == 149 and len(eval_records) == 37
    held_out_ids = {r["participant_id"] for r in eval_records}
    assert held_out_ids == EXPECTED_HELD_OUT_IDS, "stratified_split() did not reproduce the expected 37 ids."
    print(f"[augment] base split confirmed: {len(train_records)} train / {len(eval_records)} eval")

    tokenizer = get_tokenizer()
    windowed_rows = make_windowed_rows(train_records, tokenizer)

    # ---- Arm 2: real + windowing only ----
    arm2_rows = records + windowed_rows
    assert_leakage_safe(arm2_rows, held_out_ids)
    verify_split_reproduces(arm2_rows, expected_train_n=149 + len(windowed_rows), label="Arm 2 (windowing)")
    write_parquet(arm2_rows, WINDOWED_OUT)

    # ---- Arm 3: real + windowing + minority SMOTE ----
    smote_rows = make_smote_rows(train_records, N_SMOTE_POSITIVE, AUGMENT_SEED)
    arm3_rows = arm2_rows + smote_rows
    assert_leakage_safe(arm3_rows, held_out_ids)
    verify_split_reproduces(
        arm3_rows,
        expected_train_n=149 + len(windowed_rows) + len(smote_rows),
        label="Arm 3 (windowing + SMOTE)",
    )
    write_parquet(arm3_rows, AUGMENTED_OUT)

    # ---- Final summary ----
    def class_counts(rows):
        pos = sum(1 for r in rows if float(r["phq_score"]) >= 10.0)
        return pos, len(rows) - pos

    arm2_train, _ = T.stratified_split(arm2_rows)
    arm3_train, _ = T.stratified_split(arm3_rows)
    a2p, a2n = class_counts(arm2_train)
    a3p, a3n = class_counts(arm3_train)
    print()
    print("=== FINAL COUNTS ===")
    print(f"Arm 1 (original):          train=149  (44 pos / 105 neg, {44/149:.1%} positive)")
    print(f"Arm 2 (+ windowing):       train={len(arm2_train)}  ({a2p} pos / {a2n} neg, {a2p/len(arm2_train):.1%} positive)")
    print(f"Arm 3 (+ windowing+SMOTE): train={len(arm3_train)}  ({a3p} pos / {a3n} neg, {a3p/len(arm3_train):.1%} positive)")
    print()
    print("Held-out eval set (37, verified identical across all 3 arms):", sorted(held_out_ids))
    return 0


if __name__ == "__main__":
    sys.exit(main())
