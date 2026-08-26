#!/usr/bin/env python3
"""
scripts/xai_step_a6_verify.py

STEP A6 — offline verification that Step A5's in-distribution mean baselines
(compute_modality_baselines(), trainer_mentalbert_privacy.py) resolve the
scale artefact found in Step A4. Does NOT modify any live pipeline file.
No orchestrator, no gRPC, no MongoDB, no DP, no upload.

Uses the SAME saved local checkpoint and the SAME held-out 37
(stratified_split(), unchanged default seed=42) as Steps A3/A4.

Reports, for both the OLD (zero/PAD) and NEW (in-distribution mean)
baselines:
  - IG aggregate modality split (OLD reused verbatim from Step A3's saved
    report - no need to re-run the ~370s pass twice for a value that didn't
    change; NEW computed fresh via the now-updated xai_integrated_gradients())
  - ablation aggregate modality split (both OLD and NEW computed fresh here
    - ablation has no gradient pass, so re-running both costs seconds, not
    minutes)
  - per-sample dominant-modality counts for both methods under both baseline
    regimes
  - a plain convergence verdict

Usage:
    .venv\\Scripts\\python.exe scripts\\xai_step_a6_verify.py
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path

REPO_ROOT = Path(r"D:\Download D\BE PIPELINE\Capstone-")
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
sys.path.append(str(REPO_ROOT / "installer" / "runtime"))

import torch
from torch.utils.data import DataLoader

from agents.trainer.trainer_mentalbert_privacy import (
    MultiModalModel, MultiModalDataset, read_parquet_records, stratified_split,
    collate_batch, modality_ablation_importance, xai_integrated_gradients,
    compute_modality_baselines,
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


def dominant_counts_from_per_sample(per_sample, key="normalized"):
    counts = {"text": 0, "audio": 0, "vision": 0}
    for s in per_sample:
        d = s[key]
        counts[max(d, key=d.get)] += 1
    return counts


def main() -> int:
    old_ig_report_path = _latest_xai_report()
    print(f"[info] OLD (zero/PAD baseline) IG reference: {old_ig_report_path}")
    old_ig_report = json.loads(old_ig_report_path.read_text())
    old_ig_agg = old_ig_report["aggregate_normalized"]
    old_ig_dominant = dominant_counts_from_per_sample(old_ig_report["per_sample"])

    if not CHECKPOINT_PATH.exists():
        raise FileNotFoundError(f"{CHECKPOINT_PATH} not found.")
    print(f"[info] loading checkpoint: {CHECKPOINT_PATH}")

    records = read_parquet_records(str(PARQUET_PATH))
    train_records, eval_records = stratified_split(records)
    tokenizer = AutoTokenizer.from_pretrained(MENTALBERT_PRETRAIN)
    ds_eval = MultiModalDataset(eval_records, tokenizer, max_len=MULTIMODAL_MAX_LEN)
    print(f"[info] train={len(train_records)} eval={len(ds_eval)} (stratified_split(), seed=42, unchanged)")

    model = MultiModalModel(MENTALBERT_PRETRAIN, audio_dim=154, vision_dim=84, device=DEFAULT_DEVICE)
    model.load_state_dict(torch.load(CHECKPOINT_PATH, map_location=DEFAULT_DEVICE))
    if FREEZE_TEXT_ENCODER:
        for p in model.bert.parameters():
            p.requires_grad_(False)
    model.to(DEFAULT_DEVICE)
    model.eval()

    # ---- Step A5 baselines: compute once (cached to disk), report where + norms ----
    t0 = time.time()
    baselines = compute_modality_baselines(train_records, tokenizer, DEFAULT_DEVICE)
    t_baseline = time.time() - t0
    print()
    print("=" * 78)
    print("STEP A5 BASELINE CACHE")
    print("=" * 78)
    from agents.trainer.trainer_mentalbert_privacy import XAI_BASELINE_CACHE_PATH
    print(f"  cache path        : {XAI_BASELINE_CACHE_PATH}")
    print(f"  cache exists      : {XAI_BASELINE_CACHE_PATH.exists()}")
    print(f"  computed/loaded in: {t_baseline:.2f}s")
    print(f"  num_train_records : {baselines['num_train_records']}")
    print(f"  num_text_tokens_averaged: {baselines['num_text_tokens_averaged']}")
    print(f"  L2 norms          : text={baselines['text_mean_embedding_l2_norm']:.2f}  "
          f"audio={baselines['audio_mean_l2_norm']:.2f}  vision={baselines['vision_mean_l2_norm']:.2f}")
    print(f"  (for reference, OLD zero-baseline L2 norms were all 0.0; the real per-sample "
          f"input norms measured in Step A4 were text~383.6 audio~6381.6 vision~574.1)")

    # ---- NEW ablation (mean baseline), aggregate + per-sample ----
    loader_all = DataLoader(ds_eval, batch_size=len(ds_eval), shuffle=False, collate_fn=collate_batch)
    batch_all = next(iter(loader_all))

    old_ablation_agg = modality_ablation_importance(model, batch_all, device=DEFAULT_DEVICE)
    new_ablation_agg = modality_ablation_importance(
        model, batch_all, device=DEFAULT_DEVICE,
        audio_baseline=baselines["audio_mean"], vision_baseline=baselines["vision_mean"],
        text_baseline_embedding=baselines["text_mean_embedding"],
    )

    def _share(agg):
        s = {"text": agg["text_score"], "audio": agg["audio_score"], "vision": agg["vision_score"]}
        total = sum(s.values())
        return {k: (v / total if total > 0 else 0.0) for k, v in s.items()}

    old_ablation_share = _share(old_ablation_agg)
    new_ablation_share = _share(new_ablation_agg)

    old_ablation_dominant = {"text": 0, "audio": 0, "vision": 0}
    new_ablation_dominant = {"text": 0, "audio": 0, "vision": 0}
    for idx in range(len(ds_eval)):
        item = ds_eval[idx]
        one_batch = {
            "input_ids": item["input_ids"].unsqueeze(0), "attention_mask": item["attention_mask"].unsqueeze(0),
            "audio_vec": item["audio_vec"].unsqueeze(0), "video_vec": item["video_vec"].unsqueeze(0),
        }
        old_res = modality_ablation_importance(model, one_batch, device=DEFAULT_DEVICE)
        new_res = modality_ablation_importance(
            model, one_batch, device=DEFAULT_DEVICE,
            audio_baseline=baselines["audio_mean"], vision_baseline=baselines["vision_mean"],
            text_baseline_embedding=baselines["text_mean_embedding"],
        )
        old_scores = {"text": old_res["text_score"], "audio": old_res["audio_score"], "vision": old_res["vision_score"]}
        new_scores = {"text": new_res["text_score"], "audio": new_res["audio_score"], "vision": new_res["vision_score"]}
        old_ablation_dominant[max(old_scores, key=old_scores.get)] += 1
        new_ablation_dominant[max(new_scores, key=new_scores.get)] += 1

    print()
    print("=" * 78)
    print("ABLATION: OLD (zero/empty-string) vs NEW (in-distribution mean)")
    print("=" * 78)
    print(f"  OLD normalized share : {old_ablation_share}")
    print(f"  OLD dominant/37      : {old_ablation_dominant}")
    print(f"  NEW normalized share : {new_ablation_share}")
    print(f"  NEW dominant/37      : {new_ablation_dominant}")

    # ---- NEW IG (mean baseline) - the expensive ~370s re-run ----
    print()
    print("=" * 78)
    print(f"RE-RUNNING INTEGRATED GRADIENTS with Step A5 mean baselines "
          f"({len(ds_eval)} samples, n_steps=25) - this takes several minutes")
    print("=" * 78)
    t_ig0 = time.time()
    new_ig_summary = xai_integrated_gradients(
        model, ds_eval, eval_records, tokenizer, "step-a6-verify", DEFAULT_DEVICE,
        eval_metrics={"note": "Step A6 re-verification run, same checkpoint as Step A3"},
        baselines=baselines, n_steps=25,
        out_dir=Path.home() / ".federated" / "data" / "explain_logs",
    )
    t_ig = time.time() - t_ig0
    print(f"[info] new IG run complete in {t_ig:.2f}s -> {new_ig_summary['json_path']}")

    new_ig_agg = new_ig_summary["aggregate_normalized"]
    new_ig_dominant = dominant_counts_from_per_sample(new_ig_summary["per_sample"])

    print()
    print("=" * 78)
    print("IG: OLD (zero/PAD, Step A3) vs NEW (in-distribution mean, Step A6)")
    print("=" * 78)
    print(f"  OLD aggregate normalized : {old_ig_agg}")
    print(f"  OLD dominant/37          : {old_ig_dominant}")
    print(f"  NEW aggregate normalized : {new_ig_agg}")
    print(f"  NEW dominant/37          : {new_ig_dominant}")

    # ---- convergence verdict ----
    new_ig_top = max(new_ig_agg, key=new_ig_agg.get)
    new_ablation_top = max(new_ablation_share, key=new_ablation_share.get)
    ig_unanimous = new_ig_dominant[new_ig_top] == len(ds_eval)
    ablation_unanimous = new_ablation_dominant[new_ablation_top] == len(ds_eval)
    converge = new_ig_top == new_ablation_top

    print()
    print("=" * 78)
    print("CONVERGENCE VERDICT")
    print("=" * 78)
    print(f"  New IG top modality       : {new_ig_top}  ({new_ig_dominant[new_ig_top]}/{len(ds_eval)} samples, "
          f"unanimous={ig_unanimous})")
    print(f"  New ablation top modality : {new_ablation_top}  "
          f"({new_ablation_dominant[new_ablation_top]}/{len(ds_eval)} samples, unanimous={ablation_unanimous})")
    if converge:
        print(f"  -> CONVERGE: both methods now agree on '{new_ig_top}' as the top modality under "
              "consistent in-distribution baselines. The mechanism can be trusted to rank modalities "
              "(with the usual caveat that this is one client, one checkpoint, one held-out split - "
              "not yet validated across clients/rounds).")
    else:
        print(f"  -> STILL DIVERGE: IG says '{new_ig_top}', ablation says '{new_ablation_top}'. "
              "Baseline-kind consistency did not resolve the disagreement. Phase B stays BLOCKED - "
              "do not proceed on this mechanism until the divergence is understood.")

    return 0


if __name__ == "__main__":
    sys.exit(main())
