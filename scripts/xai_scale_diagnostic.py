#!/usr/bin/env python3
"""
scripts/xai_scale_diagnostic.py

STEP A4 (Phase A) — offline diagnostic only. Does NOT modify
trainer_mentalbert_privacy.py or any live pipeline file. No orchestrator, no
gRPC, no MongoDB, no DP, no upload.

Question: Step A3's Integrated Gradients report found aggregate modality
attribution text=0.1314 / audio=0.7911 / vision=0.0775, with audio ranked
#1 in all 37/37 held-out samples with zero exceptions. Is that a genuine
signal, or a raw input-scale artefact of unnormalized IG (wav2vec2's raw
154-dim audio_vec may simply have a larger natural L2 norm than the BERT
embedding output or the 84-dim DenseNet vision_vec, which would inflate
IG's raw attribution sum regardless of "importance")?

Method (avoids re-running the ~370s IG pass — reuses its already-computed
per-sample raw attributions from Step A3's report file):
  1. Load the SAME saved local checkpoint that produced that report
     (~/.federated/data/secure_store/mentalbert_privacy_subset.pt) and the
     SAME held-out 37 (stratified_split(), unchanged default seed=42).
  2. Measure each modality's raw input L2 norm per sample (cheap, forward-only,
     no gradients).
  3. Scale-normalize IG's raw attribution by that sample's input norm; report
     the normalized split alongside the original unnormalized one.
  4. Cross-check against modality_ablation_importance() (line 1172-1223) - a
     fundamentally different, scale-aware-by-construction mechanism (measures
     prediction shift when a modality is zeroed, not gradient magnitude) - run
     on the SAME model and the SAME 37 records, both as one aggregate batch
     and per-sample for a 37/37 ranking-consistency comparison.
  5. State a plain conclusion: real, artefact, or undetermined.

Usage:
    .venv\\Scripts\\python.exe scripts\\xai_scale_diagnostic.py
    .venv\\Scripts\\python.exe scripts\\xai_scale_diagnostic.py --xai-report <path>
"""

from __future__ import annotations

import argparse
import glob
import json
import sys
from pathlib import Path

REPO_ROOT = Path(r"D:\Download D\BE PIPELINE\Capstone-")
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
sys.path.append(str(REPO_ROOT / "installer" / "runtime"))

import torch
from torch.utils.data import DataLoader

from agents.trainer.trainer_mentalbert_privacy import (
    MultiModalModel, MultiModalDataset, read_parquet_records, stratified_split,
    collate_batch, modality_ablation_importance,
    MENTALBERT_PRETRAIN, MULTIMODAL_MAX_LEN, FREEZE_TEXT_ENCODER, LOCAL_SAVE_DIR,
)
from transformers import AutoTokenizer

PARQUET_PATH = REPO_ROOT / "dataset_build" / "daic_records_multimodal_participant_only.parquet"
CHECKPOINT_PATH = LOCAL_SAVE_DIR / "mentalbert_privacy_subset.pt"
DEFAULT_DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


def _latest_xai_report() -> Path:
    explain_dir = Path.home() / ".federated" / "data" / "explain_logs"
    candidates = sorted(explain_dir.glob("xai_ig_*.json"), key=lambda p: p.stat().st_mtime)
    if not candidates:
        raise FileNotFoundError(f"No xai_ig_*.json found under {explain_dir} - run Step A3 first.")
    return candidates[-1]


def mean_std(vals):
    n = len(vals)
    m = sum(vals) / n
    var = sum((v - m) ** 2 for v in vals) / n
    return m, var ** 0.5


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--xai-report", default=None, help="path to a Step A3 xai_ig_*.json; default: most recent")
    args = ap.parse_args()

    xai_report_path = Path(args.xai_report) if args.xai_report else _latest_xai_report()
    print(f"[info] using IG report: {xai_report_path}")
    xai_report = json.loads(xai_report_path.read_text())
    per_sample_ig = {s["record_id"]: s for s in xai_report["per_sample"]}
    print(f"[info] IG report eval_metrics: {xai_report['eval_metrics']}")
    print(f"[info] IG report aggregate_normalized (raw, unscaled): {xai_report['aggregate_normalized']}")

    if not CHECKPOINT_PATH.exists():
        raise FileNotFoundError(f"{CHECKPOINT_PATH} not found - Step A3's checkpoint was expected to still be on disk.")
    print(f"[info] loading checkpoint: {CHECKPOINT_PATH} (mtime={CHECKPOINT_PATH.stat().st_mtime})")

    records = read_parquet_records(str(PARQUET_PATH))
    train_records, eval_records = stratified_split(records)
    tokenizer = AutoTokenizer.from_pretrained(MENTALBERT_PRETRAIN)
    ds_eval = MultiModalDataset(eval_records, tokenizer, max_len=MULTIMODAL_MAX_LEN)
    print(f"[info] held-out set: {len(ds_eval)} records (stratified_split(), default seed=42, unchanged)")

    model = MultiModalModel(MENTALBERT_PRETRAIN, audio_dim=154, vision_dim=84, device=DEFAULT_DEVICE)
    state = torch.load(CHECKPOINT_PATH, map_location=DEFAULT_DEVICE)
    model.load_state_dict(state)
    if FREEZE_TEXT_ENCODER:
        for p in model.bert.parameters():
            p.requires_grad_(False)
    model.to(DEFAULT_DEVICE)
    model.eval()

    # ---- sanity check: alignment between this run's eval_records and the
    # IG report's per_sample record_ids must match, or the scale-normalization
    # below would silently pair the wrong sample's norm with the wrong
    # sample's attribution ----
    mismatches = []
    for idx, r in enumerate(eval_records):
        record_id = r.get("participant_id") or r.get("session_id") or r.get("id") or f"eval_{idx}"
        if record_id not in per_sample_ig:
            mismatches.append(record_id)
    if mismatches:
        print(f"[FATAL] {len(mismatches)} eval records have no matching entry in the IG report "
              f"(split mismatch - aborting rather than silently misaligning): {mismatches[:5]}")
        return 1
    print("[info] record_id alignment check passed - every eval record matches an IG report entry")

    # ---- TASK 1: raw input magnitudes per modality ----
    text_norms, audio_norms, vision_norms = [], [], []
    per_sample_norms = {}
    for idx in range(len(ds_eval)):
        item = ds_eval[idx]
        input_ids = item["input_ids"].unsqueeze(0).to(DEFAULT_DEVICE)
        audio_vec = item["audio_vec"].unsqueeze(0).to(DEFAULT_DEVICE)
        video_vec = item["video_vec"].unsqueeze(0).to(DEFAULT_DEVICE)
        with torch.no_grad():
            real_embeds = model.bert.embeddings(input_ids=input_ids)
        t_norm = real_embeds.norm().item()
        a_norm = audio_vec.norm().item()
        v_norm = video_vec.norm().item()
        record = eval_records[idx]
        record_id = record.get("participant_id") or record.get("session_id") or record.get("id") or f"eval_{idx}"
        per_sample_norms[record_id] = {"text": t_norm, "audio": a_norm, "vision": v_norm}
        text_norms.append(t_norm)
        audio_norms.append(a_norm)
        vision_norms.append(v_norm)

    t_mean, t_std = mean_std(text_norms)
    a_mean, a_std = mean_std(audio_norms)
    v_mean, v_std = mean_std(vision_norms)

    print()
    print("=" * 78)
    print("TASK 1 — RAW INPUT MAGNITUDE PER MODALITY (L2 norm, mean/std over 37 samples)")
    print("=" * 78)
    print(f"  text embedding output : mean={t_mean:.4f}  std={t_std:.4f}")
    print(f"  audio_vec              : mean={a_mean:.4f}  std={a_std:.4f}")
    print(f"  vision_vec              : mean={v_mean:.4f}  std={v_std:.4f}")
    print(f"  ratio audio/text  : {a_mean / t_mean:.2f}x")
    print(f"  ratio audio/vision: {a_mean / v_mean:.2f}x")
    print(f"  ratio text/vision : {t_mean / v_mean:.2f}x")

    # ---- TASK 2: scale-normalized attribution ----
    scale_norm_per_sample = []
    for record_id, ig_s in per_sample_ig.items():
        norms = per_sample_norms[record_id]
        raw = ig_s["raw"]
        scaled = {
            "text": raw["text"] / norms["text"] if norms["text"] > 0 else 0.0,
            "audio": raw["audio"] / norms["audio"] if norms["audio"] > 0 else 0.0,
            "vision": raw["vision"] / norms["vision"] if norms["vision"] > 0 else 0.0,
        }
        total = sum(scaled.values())
        norm_scaled = {k: (v / total if total > 0 else 0.0) for k, v in scaled.items()}
        scale_norm_per_sample.append((record_id, norm_scaled))

    agg_scaled = {
        k: sum(d[k] for _, d in scale_norm_per_sample) / len(scale_norm_per_sample)
        for k in ("text", "audio", "vision")
    }
    dominant_scaled = {"text": 0, "audio": 0, "vision": 0}
    for _, d in scale_norm_per_sample:
        dominant_scaled[max(d, key=d.get)] += 1

    print()
    print("=" * 78)
    print("TASK 2 — SCALE-NORMALIZED ATTRIBUTION (raw IG attribution / input L2 norm)")
    print("=" * 78)
    print(f"  UNNORMALIZED (Step A3, as reported)  : {xai_report['aggregate_normalized']}")
    print(f"  SCALE-NORMALIZED (this diagnostic)   : "
          f"text={agg_scaled['text']:.4f} audio={agg_scaled['audio']:.4f} vision={agg_scaled['vision']:.4f}")
    print(f"  Scale-normalized dominant-modality count across 37 samples: {dominant_scaled}")

    # ---- TASK 3: ablation cross-check ----
    loader_all = DataLoader(ds_eval, batch_size=len(ds_eval), shuffle=False, collate_fn=collate_batch)
    batch_all = next(iter(loader_all))
    ablation_agg = modality_ablation_importance(model, batch_all, device=DEFAULT_DEVICE)
    ablation_scores = {"text": ablation_agg["text_score"], "audio": ablation_agg["audio_score"],
                        "vision": ablation_agg["vision_score"]}
    ablation_total = sum(ablation_scores.values())
    ablation_share = {k: (v / ablation_total if ablation_total > 0 else 0.0) for k, v in ablation_scores.items()}

    dominant_ablation = {"text": 0, "audio": 0, "vision": 0}
    for idx in range(len(ds_eval)):
        item = ds_eval[idx]
        single = {k: (v.unsqueeze(0) if isinstance(v, torch.Tensor) else v) for k, v in item.items()}
        one_batch = {
            "input_ids": single["input_ids"], "attention_mask": single["attention_mask"],
            "audio_vec": single["audio_vec"], "video_vec": single["video_vec"],
        }
        res = modality_ablation_importance(model, one_batch, device=DEFAULT_DEVICE)
        scores = {"text": res["text_score"], "audio": res["audio_score"], "vision": res["vision_score"]}
        dominant_ablation[max(scores, key=scores.get)] += 1

    print()
    print("=" * 78)
    print("TASK 3 — ABLATION CROSS-CHECK (modality_ablation_importance(), zero-one-modality-out)")
    print("=" * 78)
    print(f"  raw ablation scores (whole-batch, all 37 at once) : {ablation_scores}")
    print(f"  normalized share                                   : "
          f"text={ablation_share['text']:.4f} audio={ablation_share['audio']:.4f} vision={ablation_share['vision']:.4f}")
    print(f"  ablation dominant-modality count across 37 samples : {dominant_ablation}")

    # ---- TASK 4: conclusion ----
    ig_raw_dominant = "audio"  # established fact from Step A3: 37/37
    scaled_dominant = max(agg_scaled, key=agg_scaled.get)
    ablation_dominant = max(ablation_share, key=ablation_share.get)

    print()
    print("=" * 78)
    print("TASK 4 — CONCLUSION")
    print("=" * 78)
    print(f"  Raw (unnormalized) IG dominant modality      : {ig_raw_dominant} (37/37 samples, Step A3)")
    print(f"  Scale-normalized IG dominant modality        : {scaled_dominant} "
          f"({dominant_scaled[scaled_dominant]}/37 samples)")
    print(f"  Ablation dominant modality                   : {ablation_dominant} "
          f"({dominant_ablation[ablation_dominant]}/37 samples)")
    if ig_raw_dominant == ablation_dominant and ig_raw_dominant == scaled_dominant:
        print("  -> All three methods agree: audio dominance appears REAL, not merely a scale artefact.")
    elif ig_raw_dominant != ablation_dominant:
        print(f"  -> Raw IG says '{ig_raw_dominant}', ablation (scale-aware by construction) says "
              f"'{ablation_dominant}'. These DISAGREE -> raw IG's audio dominance is most likely an "
              "ARTEFACT of input scale, not a genuine finding.")
    else:
        print("  -> Mixed signal between methods - UNDETERMINED, needs more investigation before Phase B.")

    return 0


if __name__ == "__main__":
    sys.exit(main())
