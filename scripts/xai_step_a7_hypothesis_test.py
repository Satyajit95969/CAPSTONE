#!/usr/bin/env python3
"""
scripts/xai_step_a7_hypothesis_test.py

STEP A7 — offline only. Does NOT modify any live pipeline file. No
orchestrator, no gRPC, no MongoDB, no DP, no upload.

Tests the hypothesis (proposed in Step A6's report) that ablation's
instability under the new in-distribution baselines - a near-random 11/20/6
per-sample split, vs IG's unanimous 37/37 - comes from ablation being a
single discrete jump from baseline to input, while IG integrates over
n_steps path points, and that many predictions sitting near the 0.5 decision
boundary make a discrete jump more exposed to local nonlinearity than a
smoothed path integral.

Four checks, all using the SAME saved local checkpoint and the SAME held-out
37 (stratified_split(), seed=42, unchanged) as Steps A3-A6, and the SAME
Step A5 in-distribution mean baselines (loaded from the existing cache, not
recomputed):

  1. IG per-sample dominant-modality counts at n_steps in {5, 10, 25, 50}.
     n_steps=25 is REUSED from Step A6's saved report (session_id
     "step-a6-verify") rather than re-run, to save ~400s. n_steps=100 is
     DROPPED per the explicit runtime budget instruction (>45 min estimated
     for all five together; reusing 25 and dropping 100 keeps this run to
     roughly 20-25 minutes).
  2. Per-sample text-score distribution (mean/std/min/max) at n_steps=25,
     read directly from Step A6's existing per-sample data - checks whether
     IG's unanimity is genuine per-patient discrimination or a suspiciously
     flat, near-identical score for every sample.
  3. Spearman rank correlation, per sample, between IG's (text, audio,
     vision) triplet (n_steps=25) and ablation's (text, audio, vision)
     triplet (freshly computed here under the SAME mean baselines - Step A6
     only kept ablation's per-sample ARGMAX, not the full triplet, so this
     is recomputed, cheaply: no gradients).
  4. The same per-sample IG-vs-ablation agreement, split by prediction
     confidence: confident (predicted positive prob <0.3 or >0.7) vs
     uncertain (0.3-0.7).

Usage:
    .venv\\Scripts\\python.exe scripts\\xai_step_a7_hypothesis_test.py
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
from scipy.stats import spearmanr

from agents.trainer.trainer_mentalbert_privacy import (
    MultiModalModel, MultiModalDataset, read_parquet_records, stratified_split,
    modality_ablation_importance, xai_integrated_gradients, compute_modality_baselines,
    MENTALBERT_PRETRAIN, MULTIMODAL_MAX_LEN, FREEZE_TEXT_ENCODER, LOCAL_SAVE_DIR,
)
from transformers import AutoTokenizer

PARQUET_PATH = REPO_ROOT / "dataset_build" / "daic_records_multimodal_participant_only.parquet"
CHECKPOINT_PATH = LOCAL_SAVE_DIR / "mentalbert_privacy_subset.pt"
DEFAULT_DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
EXPLAIN_DIR = Path.home() / ".federated" / "data" / "explain_logs"
N_STEPS_FRESH = [5, 10, 50]  # 25 reused from Step A6, 100 dropped per runtime budget


def mean_std_min_max(vals):
    n = len(vals)
    m = sum(vals) / n
    var = sum((v - m) ** 2 for v in vals) / n
    return {"mean": m, "std": var ** 0.5, "min": min(vals), "max": max(vals)}


def dominant_counts(per_sample_list, key="normalized"):
    counts = {"text": 0, "audio": 0, "vision": 0}
    for s in per_sample_list:
        d = s[key]
        counts[max(d, key=d.get)] += 1
    return counts


def run_ig_chunked(model, ds_eval, eval_records, device, baselines, n_steps, internal_batch_size):
    """
    Diagnostic-only, memory-safe re-implementation of xai_integrated_gradients()'s
    per-sample loop, with captum's internal_batch_size set to avoid CUDA OOM at
    high n_steps on a 4GB card (n_steps=50 OOM'd via the live function's
    unchunked call - confirmed live: torch.AcceleratorError: CUDA error: out
    of memory, inside the text-branch BERT forward). internal_batch_size makes
    captum process the n_steps interpolation points in smaller sequential
    sub-batches instead of one large virtual batch - slower, not less correct.

    Lives ONLY in this diagnostic script - xai_integrated_gradients() in
    trainer_mentalbert_privacy.py is NOT modified, per Step A7's "implement
    nothing [in the live pipeline]" instruction. Returns a list of per-sample
    dicts with the same normalized/predicted_positive_prob shape the live
    function's per_sample entries have, so downstream item-1/3/4 code can
    treat this and the live function's output identically.
    """
    from captum.attr import IntegratedGradients

    model.eval()
    text_mean_embedding = baselines["text_mean_embedding"].to(device)
    audio_mean = baselines["audio_mean"].to(device)
    vision_mean = baselines["vision_mean"].to(device)

    def forward_text(embeds, audio_vec, vision_vec, attention_mask, dummy_ids):
        logits, _, _ = model(dummy_ids, attention_mask, audio_vec=audio_vec,
                              vision_vec=vision_vec, inputs_embeds=embeds)
        return torch.softmax(logits, dim=1)[:, 1]

    def forward_av(audio_vec, vision_vec, embeds, attention_mask, dummy_ids):
        logits, _, _ = model(dummy_ids, attention_mask, audio_vec=audio_vec,
                              vision_vec=vision_vec, inputs_embeds=embeds)
        return torch.softmax(logits, dim=1)[:, 1]

    ig_text = IntegratedGradients(forward_text)
    ig_av = IntegratedGradients(forward_av)

    per_sample = []
    for idx in range(len(ds_eval)):
        item = ds_eval[idx]
        input_ids = item["input_ids"].unsqueeze(0).to(device)
        attention_mask = item["attention_mask"].unsqueeze(0).to(device)
        audio_vec = item["audio_vec"].unsqueeze(0).to(device)
        video_vec = item["video_vec"].unsqueeze(0).to(device)

        with torch.no_grad():
            real_embeds = model.bert.embeddings(input_ids=input_ids)
            seq_len = input_ids.shape[1]
            baseline_embeds = text_mean_embedding.view(1, 1, -1).expand(1, seq_len, -1).contiguous()
            audio_baseline = audio_mean.unsqueeze(0).expand_as(audio_vec).contiguous()
            vision_baseline = vision_mean.unsqueeze(0).expand_as(video_vec).contiguous()
            base_probs = torch.softmax(
                model(input_ids, attention_mask, audio_vec=audio_vec, vision_vec=video_vec)[0], dim=1
            )
            pred_prob_pos = base_probs[0, 1].item()

        attr_embeds = ig_text.attribute(
            inputs=real_embeds, baselines=baseline_embeds,
            additional_forward_args=(audio_vec, video_vec, attention_mask, input_ids),
            n_steps=n_steps, internal_batch_size=internal_batch_size,
        )
        text_raw = attr_embeds[0].sum(dim=-1).abs().sum().item()

        attr_av = ig_av.attribute(
            inputs=(audio_vec, video_vec), baselines=(audio_baseline, vision_baseline),
            additional_forward_args=(real_embeds, attention_mask, input_ids),
            n_steps=n_steps, internal_batch_size=internal_batch_size,
        )
        audio_raw = attr_av[0][0].abs().sum().item()
        vision_raw = attr_av[1][0].abs().sum().item()

        total = text_raw + audio_raw + vision_raw
        norm = ({"text": text_raw / total, "audio": audio_raw / total, "vision": vision_raw / total}
                if total > 0 else {"text": 0.0, "audio": 0.0, "vision": 0.0})

        record = eval_records[idx]
        record_id = record.get("participant_id") or record.get("session_id") or record.get("id") or f"eval_{idx}"
        per_sample.append({"record_id": record_id, "normalized": norm, "predicted_positive_prob": pred_prob_pos})

    return per_sample


def find_step_a6_report() -> Path:
    candidates = sorted(EXPLAIN_DIR.glob("xai_ig_step-a6-verify_*.json"), key=lambda p: p.stat().st_mtime)
    if not candidates:
        raise FileNotFoundError("No Step A6 report (xai_ig_step-a6-verify_*.json) found - run Step A6 first.")
    return candidates[-1]


def main() -> int:
    t_start = time.time()

    step_a6_path = find_step_a6_report()
    print(f"[info] reusing Step A6 (n_steps=25) report: {step_a6_path}")
    step_a6_report = json.loads(step_a6_path.read_text())
    ig_by_nsteps = {25: step_a6_report}

    if not CHECKPOINT_PATH.exists():
        raise FileNotFoundError(f"{CHECKPOINT_PATH} not found.")
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

    baselines = compute_modality_baselines(train_records, tokenizer, DEFAULT_DEVICE)
    print(f"[info] Step A5 baselines loaded from cache (L2 norms: "
          f"text={baselines['text_mean_embedding_l2_norm']:.2f} "
          f"audio={baselines['audio_mean_l2_norm']:.2f} vision={baselines['vision_mean_l2_norm']:.2f})")

    # ---- ITEM 1 (part): fresh IG runs at n_steps in {5, 10, 50} ----
    # n_steps=5 and 10 already succeeded and were saved to disk in the run
    # that later OOM'd at n_steps=50 (unchunked, 4GB card) - reuse those
    # files rather than recomputing. n_steps=50 uses run_ig_chunked() (this
    # script only, internal_batch_size=10) to fit in VRAM.
    for ns in N_STEPS_FRESH:
        existing = sorted(EXPLAIN_DIR.glob(f"xai_ig_step-a7-nsteps{ns}_*.json"), key=lambda p: p.stat().st_mtime)
        if existing and ns != 50:
            print(f"\n[info] reusing already-computed n_steps={ns} report: {existing[-1]}")
            ig_by_nsteps[ns] = json.loads(existing[-1].read_text())
            continue
        print(f"\n[info] running IG at n_steps={ns} ...")
        t0 = time.time()
        if ns == 50:
            per_sample_50 = run_ig_chunked(model, ds_eval, eval_records, DEFAULT_DEVICE, baselines,
                                            n_steps=50, internal_batch_size=10)
            agg_raw = {}  # not tracked in the chunked path - only normalized per-sample needed downstream
            agg_norm = {
                k: sum(s["normalized"][k] for s in per_sample_50) / len(per_sample_50)
                for k in ("text", "audio", "vision")
            }
            summary = {"per_sample": per_sample_50, "aggregate_normalized": agg_norm}
            elapsed = time.time() - t0
            print(f"[info] n_steps=50 (chunked, internal_batch_size=10) done in {elapsed:.1f}s")
        else:
            summary = xai_integrated_gradients(
                model, ds_eval, eval_records, tokenizer, f"step-a7-nsteps{ns}", DEFAULT_DEVICE,
                eval_metrics={"note": f"Step A7 n_steps sweep, n_steps={ns}"},
                baselines=baselines, n_steps=ns, out_dir=EXPLAIN_DIR,
            )
            elapsed = time.time() - t0
            print(f"[info] n_steps={ns} done in {elapsed:.1f}s -> {summary['json_path']}")
        ig_by_nsteps[ns] = summary

    # ---- ITEM 1: per-sample dominance counts at each n_steps ----
    print("\n" + "=" * 78)
    print("ITEM 1 — IG PER-SAMPLE DOMINANCE COUNTS BY n_steps")
    print("=" * 78)
    for ns in sorted(ig_by_nsteps.keys()):
        dc = dominant_counts(ig_by_nsteps[ns]["per_sample"])
        agg = ig_by_nsteps[ns]["aggregate_normalized"]
        print(f"  n_steps={ns:3d}: dominant/37={dc}   aggregate_normalized="
              f"text={agg['text']:.4f} audio={agg['audio']:.4f} vision={agg['vision']:.4f}")

    # ---- ITEM 2: per-sample text score distribution at n_steps=25 ----
    text_scores_25 = [s["normalized"]["text"] for s in ig_by_nsteps[25]["per_sample"]]
    stats_25 = mean_std_min_max(text_scores_25)
    print("\n" + "=" * 78)
    print("ITEM 2 — PER-SAMPLE TEXT SCORE DISTRIBUTION (n_steps=25, new baselines)")
    print("=" * 78)
    print(f"  mean={stats_25['mean']:.4f}  std={stats_25['std']:.4f}  "
          f"min={stats_25['min']:.4f}  max={stats_25['max']:.4f}  range={stats_25['max']-stats_25['min']:.4f}")

    # ---- ITEM 3 setup: fresh per-sample ablation (full triplet) under NEW baselines ----
    print("\n[info] computing fresh per-sample ablation triplets under Step A5 baselines ...")
    ablation_per_sample = {}
    for idx in range(len(ds_eval)):
        item = ds_eval[idx]
        one_batch = {
            "input_ids": item["input_ids"].unsqueeze(0), "attention_mask": item["attention_mask"].unsqueeze(0),
            "audio_vec": item["audio_vec"].unsqueeze(0), "video_vec": item["video_vec"].unsqueeze(0),
        }
        res = modality_ablation_importance(
            model, one_batch, device=DEFAULT_DEVICE,
            audio_baseline=baselines["audio_mean"], vision_baseline=baselines["vision_mean"],
            text_baseline_embedding=baselines["text_mean_embedding"],
        )
        scores = {"text": res["text_score"], "audio": res["audio_score"], "vision": res["vision_score"]}
        total = sum(scores.values())
        norm = {k: (v / total if total > 0 else 0.0) for k, v in scores.items()}
        record = eval_records[idx]
        record_id = record.get("participant_id") or record.get("session_id") or record.get("id") or f"eval_{idx}"
        ablation_per_sample[record_id] = norm

    # ---- ITEM 3: Spearman correlation per sample, IG(n_steps=25) vs ablation ----
    modalities = ["text", "audio", "vision"]
    correlations = []
    agreement_flags = []  # top-1 match, for item 4
    confidences = []
    for s in ig_by_nsteps[25]["per_sample"]:
        rid = s["record_id"]
        ig_vec = [s["normalized"][m] for m in modalities]
        ab_vec = [ablation_per_sample[rid][m] for m in modalities]
        rho, _ = spearmanr(ig_vec, ab_vec)
        correlations.append(rho if rho == rho else 0.0)  # guard NaN (all-tied ranks)
        ig_top = max(s["normalized"], key=s["normalized"].get)
        ab_top = max(ablation_per_sample[rid], key=ablation_per_sample[rid].get)
        agreement_flags.append(ig_top == ab_top)
        confidences.append(s["predicted_positive_prob"])

    corr_stats = mean_std_min_max(correlations)
    print("\n" + "=" * 78)
    print("ITEM 3 — PER-SAMPLE SPEARMAN CORRELATION: IG(n_steps=25) vs ABLATION (new baselines)")
    print("=" * 78)
    print(f"  n=37 samples, 3 categories each (text/audio/vision) - note: with only 3 items per")
    print(f"  ranking, Spearman rho is inherently coarse/discrete (possible values cluster at")
    print(f"  {{1.0, 0.5, -0.5, -1.0}} for untied triplets), not a smooth statistic.")
    print(f"  mean={corr_stats['mean']:.4f}  std={corr_stats['std']:.4f}  "
          f"min={corr_stats['min']:.4f}  max={corr_stats['max']:.4f}")
    from collections import Counter
    rho_hist = Counter(round(r, 2) for r in correlations)
    print(f"  distribution: {dict(sorted(rho_hist.items()))}")
    print(f"  top-1 (argmax) agreement rate: {sum(agreement_flags)}/37 = {sum(agreement_flags)/37:.4f}")

    # ---- ITEM 4: split by prediction confidence ----
    confident_idx = [i for i, p in enumerate(confidences) if p < 0.3 or p > 0.7]
    uncertain_idx = [i for i, p in enumerate(confidences) if 0.3 <= p <= 0.7]

    def _group_stats(idxs):
        if not idxs:
            return {"n": 0, "mean_rho": None, "agreement_rate": None}
        rhos = [correlations[i] for i in idxs]
        agree = [agreement_flags[i] for i in idxs]
        return {
            "n": len(idxs),
            "mean_rho": sum(rhos) / len(rhos),
            "agreement_rate": sum(agree) / len(agree),
        }

    conf_stats = _group_stats(confident_idx)
    unc_stats = _group_stats(uncertain_idx)

    print("\n" + "=" * 78)
    print("ITEM 4 — AGREEMENT BY PREDICTION CONFIDENCE")
    print("=" * 78)
    print(f"  confident (prob<0.3 or >0.7): n={conf_stats['n']:2d}  "
          f"mean_spearman_rho={conf_stats['mean_rho']}  top1_agreement_rate={conf_stats['agreement_rate']}")
    print(f"  uncertain (0.3<=prob<=0.7)  : n={unc_stats['n']:2d}  "
          f"mean_spearman_rho={unc_stats['mean_rho']}  top1_agreement_rate={unc_stats['agreement_rate']}")
    print(f"  all predicted_positive_prob values: {sorted(round(p, 3) for p in confidences)}")

    total_elapsed = time.time() - t_start
    print(f"\n[info] total runtime: {total_elapsed/60:.1f} minutes")
    return 0


if __name__ == "__main__":
    sys.exit(main())
