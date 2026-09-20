"""
trainer_agent/trainer_mentalbert_privacy.py

Multimodal trainer agent supporting:
 - autonomous (inference + explainability)
 - supervised (physician-corrected fine-tune -> delta)
 - rl (online REINFORCE-style updates based on clinician reward)

Safety policy (applied automatically):
 - per-parameter absolute delta clamp (max_param_change)
 - global delta L2 norm scaling (max_global_delta_norm)
 - gradient clipping during optimization
 - conservative default learning rates and small batch sizes for RL

Usage:
  python trainer_mentalbert_privacy.py --mode autonomous --input ./session.parquet
  python trainer_mentalbert_privacy.py --mode supervised --input ./session.parquet --epochs 2
  python trainer_mentalbert_privacy.py --mode rl --input ./session.parquet --epochs 1
"""

import os
import sys
import json
import argparse
import math
import random
import time
import tempfile
from pathlib import Path
from typing import List, Dict, Any, Optional, Tuple

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from transformers import AutoModel, AutoTokenizer, AdamW

import pandas as pd
import pyarrow.parquet as pq

from core.centralized_secure_store import SecureStore
from core.centralised_receipts import CentralReceiptManager
from core import reporting as rpt

from installer.security.integrity import integrity_guard
integrity_guard()

# ---------- Config / Defaults ----------
MENTALBERT_PRETRAIN = str(Path.home() / ".federated" / "models" / "mentalbert")
DEFAULT_DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
LOCAL_SAVE_DIR = Path.home() / ".federated" / "data" / "secure_store"
LOCAL_SAVE_DIR.mkdir(parents=True, exist_ok=True)

# Fix A: tokenizer truncation window. Was hardcoded to MultiModalDataset's
# class default of 128 (see class below), which - even after removing
# interviewer text - discarded ~95% of participant speech (median session is
# ~1,460 tokens). 512 is BERT/MentalBERT's absolute positional-embedding
# ceiling, not a full fix, but the largest value this architecture supports.
# Overridable per-run; MultiModalDataset's own default is left at 128 as a
# generic fallback for any other caller that doesn't pass max_len explicitly.
MULTIMODAL_MAX_LEN = int(os.environ.get("MULTIMODAL_MAX_LEN", "512"))

# Fix B (CLAUDE.md defect #1): the text encoder (109,482,240 of 109,763,494
# params, 99.76%) is frozen so it is never trained, never diffed, never
# DP-noised, and never transmitted. DP noise grows as sqrt(param count); this
# cuts the noised surface from ~110M to 281,254 params (audio_encoder +
# vision_encoder + fusion), without touching noise_multiplier or the RDP
# accounting - the privacy guarantee is unchanged. Default true; set to
# "0"/"false" for an unfrozen A/B comparison run.
FREEZE_TEXT_ENCODER = os.environ.get("FREEZE_TEXT_ENCODER", "true").strip().lower() not in ("0", "false", "no", "")

# 2026-09-20 fusion-head capacity sweep: FusionHead.hidden was a hardcoded
# constructor default (256). fc1 (Linear(1024, hidden)) holds 93.3% of the
# 281,254 trainable params (262,400 of 281,254 at hidden=256) - the single
# highest-leverage capacity knob, and DP noise norm scales as sqrt(total
# trainable params), so this also controls DP noise. Total trainable params
# as a function of this value: 1029*FUSION_HIDDEN_DIM + 17830 (audio_encoder
# + vision_encoder fixed overhead = 17,826; verified exact at hidden=256).
#
# Default changed 256 -> 56 (2026-09-20) after a 70-run sweep
# (docs/IMPLEMENTATION_NOTES.md, "Fusion-head capacity sweep"): at n=15,
# hidden=56 (75,454 params) and hidden=256 (281,254 params) are
# statistically indistinguishable on genuine-discrimination rate (5/15 vs
# 6/15 - a one-run difference, within sampling noise). hidden=256 scored
# marginally higher (40.0% vs 33.3%) - recorded honestly, not hidden -  but
# since utility is equivalent, hidden=56 is preferred for 48% less DP noise
# (174.90 vs 338.18 measured) at 3.7x fewer parameters. Overridable; set to
# "256" to reproduce the old architecture/historical runs byte-for-byte.
FUSION_HIDDEN_DIM = int(os.environ.get("FUSION_HIDDEN_DIM", "56"))

# Phase A / STEP A2: Integrated-Gradients attribution over the LOCAL, PRE-DP
# model on held-out data (docs/IMPLEMENTATION_NOTES.md - Phase A design).
# Default OFF: diagnostic-only, never required for training/upload to
# succeed. Local disk only - never added to the delta, a receipt, or a gRPC
# message; runs strictly after training/eval and before DP, but nothing it
# produces is ever read by the DP/encryption/upload path.
XAI_ENABLED = os.environ.get("XAI_ENABLED", "0").strip().lower() not in ("0", "false", "no", "")
XAI_N_STEPS = int(os.environ.get("XAI_N_STEPS", "25"))
XAI_EXPLAIN_DIR = Path.home() / ".federated" / "data" / "explain_logs"

# Step 12 / Defect B (docs/IMPLEMENTATION_NOTES.md): unset by default, so
# ordinary runs are byte-for-byte unchanged - MultiModalModel(...) gets
# PyTorch's normal random init, exactly as before this variable existed. When
# set, orchestrate() seeds torch immediately before constructing the model
# (so multiple client processes given the same GLOBAL_INIT_SEED start
# audio_encoder/vision_encoder/fusion from an identical initial state - the
# common-base precondition FedAvg/trimmed-mean assumes and this system never
# had) and then re-randomises the RNG right after construction, so training
# stochasticity (dropout, etc.) still differs per client - not three
# bit-identical replicas. Does not change what is trained, only what the
# random init happens to be.
GLOBAL_INIT_SEED = os.environ.get("GLOBAL_INIT_SEED")

# Fix E3, loss rebalance ONLY (lr/epochs/init untouched this step - see
# docs/IMPLEMENTATION_NOTES.md). Two problems, measured in Step 9a:
#   1. loss_cls vs loss_reg: unweighted CrossEntropyLoss on a 30/70 imbalance
#      plus an MSE regression term on raw PHQ (0-23) produced a combined loss
#      where loss_reg was 5-40x loss_cls at every step - the regression term
#      structurally dominates the shared trunk's gradient (fc1 grad norm was
#      5-27x more driven by phq_mu than by classifier).
#   2. torch.nn.utils.clip_grad_norm_(..., 1.0) saturates every step (raw
#      norms measured 1163-3291), and clipping preserves direction - so the
#      regression-dominated direction wasn't a lr problem, it was a loss-
#      balance problem. Fixed here; lr is Fix E4's problem, not this one.
CLASS_WEIGHT_ENABLED = os.environ.get("CLASS_WEIGHT_ENABLED", "true").strip().lower() not in ("0", "false", "no", "")
REG_LOSS_WEIGHT = float(os.environ.get("REG_LOSS_WEIGHT", "0.5"))
# PHQ-8's own fixed clinical scale (8 items x 0-3 each), NOT derived from this
# corpus's observed max (23) - using the clinical ceiling keeps the
# normalization stable across different datasets, and gives loss_reg a
# principled 0-1 target scale instead of raw 0-23 MSE, whose squared-error
# magnitude was the actual cause of point 1 above. This is a scale constant,
# not a hyperparameter meant to be swept - no env override.
PHQ_SCALE_MAX = 24.0
# How often (in optimizer steps) to log the loss-component and gradient-norm
# breakdown below. This is a standing diagnostic capability now, not a one-off
# ad hoc script - Step 9a's numbers were gathered by hand; this makes them
# visible on every run.
TRAIN_LOG_INTERVAL = int(os.environ.get("TRAIN_LOG_INTERVAL", "5"))

# Fix E4: training volume and learning rate for the supervised path. lr=2e-5
# was a full-BERT-fine-tuning rate, wrong for a randomly-initialised 281K
# head; epochs=1 (19 steps) was thin regardless of lr. Chosen from
# scripts/sweep_lr_epochs.py's offline grid (5 lr x 4 epochs x 3 seeds, 60
# runs): lr=1e-4/epochs=10 had the best mean F1 (0.5729) among the 0/3-
# degenerate cells, and the tightest per-seed spread (0.550-0.609) of the top
# tier - the 5e-3 cells scored comparably on average (0.54-0.57) but swung
# 0.47-0.76 per seed, which is variance, not a better-trained model. Every
# epochs=1 cell was degenerate in 2 or 3 of 3 seeds at every lr tried - one
# epoch was never viable, independent of learning rate.
# Single source of truth for both values - runtime/pipeline.py imports these
# constants rather than hardcoding its own copy, so the two files cannot
# drift apart the way pipeline.py's old hardcoded "epochs": 1 did.
SUPERVISED_LR = float(os.environ.get("SUPERVISED_LR", "1e-4"))
SUPERVISED_EPOCHS = int(os.environ.get("SUPERVISED_EPOCHS", "10"))

# Step 16: round-aware LR decay. Step 15 diagnosed WHY multi-round training
# diverges (Step 14): lr and epochs were byte-identical every round with no
# schedule, and torch.nn.utils.clip_grad_norm_(..., 1.0) saturates on
# effectively every step (measured fc1_grad_norm 59.9-454.5 against a ceiling
# of 1.0) - so every round takes an essentially fixed-size step regardless of
# how close the model already is to a good solution. This does not touch the
# grad-norm clip, loss balance, or safety clamps (Step 15's explicit
# constraint) - it only makes the step size shrink round over round.
# lr_r = SUPERVISED_LR * (LR_DECAY ** (round_id - 1)), so round 1 is
# unaffected (LR_DECAY**0 = 1) and every later round decays further.
# LR_DECAY=1.0 is a valid, explicit "no decay" setting - it reproduces the
# Step 14 baseline exactly, not a special-cased default.
LR_DECAY = float(os.environ.get("LR_DECAY", "0.7"))

# Safety hyperparameters (tunable)
# Fix E5: recalibrated from the old 1e-3, which was set for the pre-Fix-E4
# regime (19 steps, lr=2e-5) and had gone stale the same way clip_norm had -
# measured (N=1, current regime: lr=1e-4, epochs=10) it was clamping 33.10%
# of all 281,254 trainable params on every single run (93,083 params), not
# occasionally - a routine limiter masquerading as a rare backstop. Global
# per-parameter |delta| distribution: median=0.00059, p90=0.00207,
# p99=0.00403, p99.9=0.00594, max=0.00916 (phq_mu was the most volatile
# submodule proportionally at 70.8% exceeding the old threshold; fc1
# dominated by absolute count simply because it's 93.3% of all trainable
# params). 0.02 gives ~2.2x headroom over the observed max and is fully
# inert on the measured run (0% clamped) - deliberately more margin than the
# L2-norm clamps use, because per-parameter maxima are a noisier statistic
# than an aggregate L2 norm (L2 benefits from central-limit averaging across
# 281K dimensions - see the CV note in docs/IMPLEMENTATION_NOTES.md - a
# single parameter's worst case does not), so N=1 doesn't support as tight a
# margin as the N=30 L2 calibrations did.
#
# IMPORTANT: compute_filtered_delta() (used by scripts/calibrate_clip_norm.py
# to calibrate DP's clip_norm) does NOT apply this clamp or the global-norm
# one below - it measures the raw, pre-safety-clamp delta. That is only a
# valid calibration for DP's clip_norm as long as BOTH safety clamps stay
# inert under normal operation (as intended here). If either is ever
# tightened enough to routinely engage again, clip_norm must be recalibrated
# too - see docs/IMPLEMENTATION_NOTES.md.
DEFAULT_MAX_PARAM_CHANGE = float(os.environ.get("MAX_PARAM_CHANGE", "0.02"))  # per-parameter absolute clamp on delta
# Fix C, recalibrated again for Fix E4 (lr/epochs changed the delta scale
# entirely - see scripts/calibrate_clip_norm.py, N=30: max shifted from
# 0.1177 to 0.7294). This is a general safety backstop applied BEFORE
# encryption, upstream of and separate from dp_agent.py's own
# clip_norm=0.85 (applied AFTER decryption, on the already safety-clamped
# delta - see process_local_update()). Set to 1.7 (2x clip_norm, same ratio
# as the original 0.3-vs-0.15 pairing) so DP's tighter 0.85 threshold is
# always the one that actually engages/binds; this backstop only fires for
# deltas beyond ~2.3x today's observed max, i.e. genuine training
# instability, not normal operation. Deliberately above clip_norm, not
# below it: if this were tighter than clip_norm, it would always clip first
# and clip_norm would never fire, severing the tie between the clip
# threshold and the noise scale that Fix C restored - confirmed empirically
# this was about to happen at the old 0.3 once Fix E4 landed (measured min
# delta 0.6247 already exceeded it).
DEFAULT_MAX_GLOBAL_DELTA_NORM = 1.7   # max L2 norm of delta state (after per-param clamp will be scaled down to this)
RL_PHQ_RANGE = 30.0                   # normalization range for PHQ when computing reward


# ---------- Dataset ----------
class MultiModalDataset(Dataset):
    def __init__(self, records: List[Dict[str, Any]], tokenizer, max_len: int = 128):
        self.records = records
        self.tokenizer = tokenizer
        self.max_len = max_len

    def __len__(self):
        return len(self.records)

    def _extract_audio_vec(self, r):
        feats = r.get("features") or {}
        audio = feats.get("audio")
        if not isinstance(audio, dict):
            return None

        # 1) Prefer wav2vec2 embeddings if present
        if isinstance(audio.get("wav2vec2"), list):
            return torch.tensor(audio["wav2vec2"], dtype=torch.float32)

        # 2) Fallback: use all numeric audio features
        numeric_vals = []
        for k, v in audio.items():
            if isinstance(v, (int, float)):
                numeric_vals.append(float(v))

        if numeric_vals:
            return torch.tensor(numeric_vals, dtype=torch.float32)

        return None

    def _extract_video_vec(self, r):
        feats = r.get("features") or {}
        if isinstance(feats, dict) and "video" in feats and isinstance(feats["video"], dict):
            v = feats["video"]
            if v.get("densenet") and isinstance(v["densenet"], (list, tuple)):
                return torch.tensor(v["densenet"], dtype=torch.float32)
            if v.get("densenet_csv") and isinstance(v["densenet_csv"], str):
                try:
                    arr = [float(x) for x in v["densenet_csv"].split(",") if x.strip() != ""]
                    return torch.tensor(arr, dtype=torch.float32)
                except:
                    return None
        # fallback: check top-level neuron_* keys
        neuron_keys = sorted([k for k in r.keys() if str(k).startswith("neuron_")])
        if neuron_keys:
            arr = [float(r[k]) for k in neuron_keys]
            return torch.tensor(arr, dtype=torch.float32)
        return None

    def __getitem__(self, idx):
        r = self.records[idx]
        def _safe_text(r) -> str:
            """Return transcript text, defaulting to empty string."""
            return (r.get("transcript") or r.get("text") or "").strip()

        text = _safe_text(r)
        enc = self.tokenizer(
            text,
            truncation=True,
            padding="max_length",
            max_length=self.max_len,
            return_tensors="pt"
        )
        input_ids = enc["input_ids"].squeeze(0)
        attention_mask = enc["attention_mask"].squeeze(0)

        audio_vec = self._extract_audio_vec(r)
        video_vec = self._extract_video_vec(r)

        phq = r.get("phq_score") or r.get("phq") or r.get("target_phq") or r.get("label_phq")
        try:
            phq_val = float(phq) if phq is not None else 0.0
        except:
            phq_val = 0.0
        label = 1 if phq_val >= 10.0 else 0

        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "audio_vec": audio_vec,
            "video_vec": video_vec,
            "phq": torch.tensor(phq_val, dtype=torch.float32),
            "label": torch.tensor(label, dtype=torch.long),
        }


# ---------- Model components ----------
class SmallMLP(nn.Module):
    def __init__(self, in_dim: int, out_dim: int = 128, hidden: Optional[int] = None, dropout: float = 0.2):
        super().__init__()
        hid = hidden or max(32, in_dim // 4)
        self.net = nn.Sequential(
            nn.Linear(in_dim, hid),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hid, out_dim),
            nn.ReLU(),
        )

    def forward(self, x):
        return self.net(x)


class FusionHead(nn.Module):
    def __init__(self, in_dim: int, hidden: int = 256, num_classes: int = 2):
        super().__init__()
        self.fc1 = nn.Linear(in_dim, hidden)
        self.act = nn.ReLU()
        self.drop = nn.Dropout(0.2)
        self.classifier = nn.Linear(hidden, num_classes)
        # For RL policy, we output mu and log_sigma for PHQ (continuous action)
        self.phq_mu = nn.Linear(hidden, 1)
        self.phq_logsigma = nn.Linear(hidden, 1)

    def forward(self, x):
        h = self.drop(self.act(self.fc1(x)))
        logits = self.classifier(h)
        mu = self.phq_mu(h).squeeze(-1)
        log_sigma = self.phq_logsigma(h).squeeze(-1)
        return logits, mu, log_sigma


class MultiModalModel(nn.Module):
    def __init__(self, bert_name: str, audio_dim: Optional[int], vision_dim: Optional[int], device: str = DEFAULT_DEVICE):
        super().__init__()
        self.device = device
        self.bert = AutoModel.from_pretrained(bert_name)
        bert_hidden = self.bert.config.hidden_size

        self.has_audio = audio_dim is not None and audio_dim > 0
        self.has_vision = vision_dim is not None and vision_dim > 0

        self.audio_encoder = SmallMLP(audio_dim, out_dim=128) if self.has_audio else None
        self.vision_encoder = SmallMLP(vision_dim, out_dim=128) if self.has_vision else None

        fusion_input_dim = bert_hidden + (128 if self.has_audio else 0) + (128 if self.has_vision else 0)
        self.fusion = FusionHead(fusion_input_dim, hidden=FUSION_HIDDEN_DIM)

    def forward(self, input_ids, attention_mask, audio_vec=None, vision_vec=None, rl_mode=False, sample_action=False, inputs_embeds=None):
        # inputs_embeds (Phase A / STEP A2): optional precomputed text embedding
        # tensor, e.g. an Integrated-Gradients interpolation between a PAD
        # baseline and the real embedding output. When given, it REPLACES the
        # input_ids embedding lookup so attribution can be computed w.r.t. a
        # differentiable embedding tensor instead of discrete token ids (token
        # ids have no gradient). input_ids is still required in this case only
        # to determine batch size upstream by callers, never used here when
        # inputs_embeds is provided. Every existing call site omits
        # inputs_embeds (defaults to None) and is therefore byte-for-byte
        # unaffected by this parameter's addition.
        if inputs_embeds is not None:
            bert_out = self.bert(inputs_embeds=inputs_embeds, attention_mask=attention_mask)
        else:
            bert_out = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        pooled = bert_out.last_hidden_state[:, 0, :]

        audio_enc = None
        vision_enc = None
        if self.has_audio:
            if audio_vec is not None:
                audio_enc = self.audio_encoder(audio_vec)
            else:
                audio_enc = torch.zeros((pooled.size(0), 128), device=pooled.device)
        if self.has_vision:
            if vision_vec is not None:
                vision_enc = self.vision_encoder(vision_vec)
            else:
                vision_enc = torch.zeros((pooled.size(0), 128), device=pooled.device)

        parts = [pooled]
        if audio_enc is not None:
            parts.append(audio_enc)
        if vision_enc is not None:
            parts.append(vision_enc)
        fused = torch.cat(parts, dim=1)

        logits, mu, log_sigma = self.fusion(fused)
        # reg prediction for supervised MSE is mu (deterministic)
        reg_pred = mu

        # RL: if sampling, draw action from N(mu, sigma)
        if rl_mode and sample_action:
            sigma = torch.exp(log_sigma).clamp(min=1e-4)
            noise = torch.randn_like(mu)
            action = mu + sigma * noise
            # compute log_prob manually
            var = sigma * sigma
            log_prob = -0.5 * (((action - mu) ** 2) / var + 2 * log_sigma + math.log(2 * math.pi))
            return logits, reg_pred, (action, log_prob)
        else:
            return logits, reg_pred, (mu, log_sigma)


# ---------- Utilities ----------
def read_parquet_records(path: str) -> List[Dict[str, Any]]:
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(path)

    # -------- Load records --------
    if p.suffix == ".jsonl":
        rows = []
        _CANONICAL_STORE = Path.home() / ".federated" / "data" / "secure_store"
 
        store = SecureStore(
            agent="lda",
            root=_CANONICAL_STORE,
        )

        with open(p, "r", encoding="utf-8") as f:
            for line in f:
                if not line.strip():
                    continue
                m = json.loads(line)

                enc_uri = m["uri"]
                row_idx = m["row"]

                # decrypt parquet
                parquet_bytes = store.decrypt_read(enc_uri)

                with tempfile.NamedTemporaryFile(delete=False, suffix=".parquet") as tf:
                    tf.write(parquet_bytes)
                    temp_path = tf.name

                # 🔥 IMPORTANT: file is CLOSED here

                table = pq.read_table(temp_path)

                # cleanup
                os.remove(temp_path)
                df = table.to_pandas()
                rows.append(df.iloc[row_idx].to_dict())

        records = rows
    elif p.suffix == ".parquet":
        table = pq.read_table(str(p))
        df = table.to_pandas()
        records = df.to_dict(orient="records")
    elif p.suffix in (".json", ".jsonl"):
        records = []
        with open(p, "r", encoding="utf-8") as f:
            if p.suffix == ".json":
                records = json.load(f)
            else:
                for line in f:
                    if line.strip():
                        records.append(json.loads(line))
    else:
        df = pd.read_csv(str(p))
        records = df.to_dict(orient="records")

    # -------- Parse embedded JSON fields --------
    for r in records:
        for key in ("features", "derived"):
            if key in r and isinstance(r[key], str):
                try:
                    r[key] = json.loads(r[key])
                except Exception:
                    pass

    # -------- FILTER BAD TRANSCRIPTS (FIXED) --------
    def _filter_records(records):
        """
        Filter records before training.
        - "failed": ASR crashed entirely (infrastructure problem) → drop
        - "empty": no speech detected in segment (normal) → keep with empty text
        - "ok": transcript present → keep
        - missing key: keep (older data without status field)
        """
        filtered = []
        dropped = 0
        for r in records:
            derived = r.get("derived") or {}
            # derived may be a JSON string (from parquet)
            if isinstance(derived, str):
                try:
                    import json
                    derived = json.loads(derived)
                except Exception:
                    derived = {}
            status = derived.get("transcript_status", "ok")
            if status == "failed":
                dropped += 1
                continue
            filtered.append(r)
    
        if dropped:
            import logging
            logging.getLogger(__name__).warning(
                "Dropped %d records with transcript_status=failed", dropped
            )
    
        if not filtered:
            raise RuntimeError(
                "All records were dropped (transcript_status=failed for all). "
                "Check your ASR setup."
            )
        return filtered
    
    records = _filter_records(records)
    return records

def collate_batch(batch):
    input_ids = torch.stack([b["input_ids"] for b in batch], dim=0)
    attention_mask = torch.stack([b["attention_mask"] for b in batch], dim=0)

    # pad audio vectors
    audio_list = [b["audio_vec"] for b in batch]
    if any(a is not None for a in audio_list):
        max_a = max([a.size(0) if a is not None else 0 for a in audio_list])
        padded_as = []
        for a in audio_list:
            if a is None:
                padded_as.append(torch.zeros(max_a))
            elif a.size(0) < max_a:
                padded_as.append(torch.cat([a, torch.zeros(max_a - a.size(0))], dim=0))
            else:
                padded_as.append(a)
        audio_batch = torch.stack(padded_as, dim=0)
    else:
        audio_batch = None

    # pad video vectors
    video_list = [b["video_vec"] for b in batch]
    if any(v is not None for v in video_list):
        max_v = max([v.size(0) if v is not None else 0 for v in video_list])
        padded_vs = []
        for v in video_list:
            if v is None:
                padded_vs.append(torch.zeros(max_v))
            elif v.size(0) < max_v:
                padded_vs.append(torch.cat([v, torch.zeros(max_v - v.size(0))], dim=0))
            else:
                padded_vs.append(v)
        video_batch = torch.stack(padded_vs, dim=0)
    else:
        video_batch = None

    phq = torch.stack([b["phq"] for b in batch], dim=0)
    label = torch.stack([b["label"] for b in batch], dim=0)

    return {
        "input_ids": input_ids,
        "attention_mask": attention_mask,
        "audio_vec": audio_batch,
        "video_vec": video_batch,
        "phq": phq,
        "label": label,
    }


# ---------- Training / Inference ----------
def run_inference(model: MultiModalModel, dataloader: DataLoader, device: str = DEFAULT_DEVICE):
    model.eval()
    results = []
    with torch.no_grad():
        for b in dataloader:
            inputs = {k: (v.to(device) if isinstance(v, torch.Tensor) else v) for k, v in b.items()}
            logits, reg_pred, _ = model(inputs["input_ids"], inputs["attention_mask"],
                                       audio_vec=inputs.get("audio_vec"), vision_vec=inputs.get("video_vec"),
                                       rl_mode=False, sample_action=False)
            probs = torch.softmax(logits, dim=1).cpu().numpy()
            preds_class = probs.argmax(axis=1)
            preds_phq = reg_pred.cpu().numpy()
            for i in range(preds_phq.shape[0]):
                results.append({
                    "pred_class": int(preds_class[i]),
                    "pred_class_probs": probs[i].tolist(),
                    "pred_phq": float(preds_phq[i]),
                })
    return results


def fine_tune_supervised(model: MultiModalModel, dataset: MultiModalDataset, epochs: int = SUPERVISED_EPOCHS, batch_size: int = 8, lr: float = SUPERVISED_LR, device: str = DEFAULT_DEVICE):
    model.to(device)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True, collate_fn=collate_batch)
    # Fix B: only trainable (requires_grad=True) params get an optimizer slot.
    # Frozen params (the text encoder, when FREEZE_TEXT_ENCODER=true) never
    # receive gradients anyway, but excluding them here is explicit rather
    # than relying on AdamW silently no-op'ing on grad=None params.
    optimizer = AdamW([p for p in model.parameters() if p.requires_grad], lr=lr)
    cls_loss_fn = nn.CrossEntropyLoss()
    reg_loss_fn = nn.MSELoss()

    model.train()
    for epoch in range(epochs):
        total_loss = 0.0
        steps = 0
        for batch in loader:
            batch = {k: (v.to(device) if isinstance(v, torch.Tensor) else v) for k, v in batch.items()}
            optimizer.zero_grad()
            logits, reg_pred, _ = model(batch["input_ids"], batch["attention_mask"],
                                        audio_vec=batch.get("audio_vec"), vision_vec=batch.get("video_vec"))
            loss_cls = cls_loss_fn(logits, batch["label"])
            loss_reg = reg_loss_fn(reg_pred, batch["phq"])
            loss = loss_cls + 0.5 * loss_reg
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            total_loss += loss.item()
            steps += 1
        if steps:
            print(f"[supervised] epoch {epoch+1}/{epochs} avg_loss={total_loss/steps:.4f}")
    return model

# ---------- Unified supervised training + evaluation + explainability ----------
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, mean_absolute_error

def train_model(dataset: MultiModalDataset, model: MultiModalModel,
                output_dir: str = "./trainer_outputs",
                epochs: int = SUPERVISED_EPOCHS, batch_size: int = 8, lr: float = SUPERVISED_LR,
                device: str = DEFAULT_DEVICE,
                eval_dataset: Optional[MultiModalDataset] = None):
    """
    Complete supervised fine-tuning on labeled PHQ data + evaluation + explainability.
    Produces model weights, metrics.json, and explain.txt.

    Fix E2: `dataset` is the TRAINING split only. If `eval_dataset` is given,
    metrics (accuracy/precision/recall/F1/MAE) are computed on it instead of
    on `dataset` - evaluating on training data is not a defensible measurement
    of anything. `eval_dataset=None` is a backward-compatible fallback for any
    other caller (there is currently exactly one call site, in orchestrate(),
    which always passes eval_dataset - see the stratified split below).
    """
    os.makedirs(output_dir, exist_ok=True)
    model.to(device)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True, collate_fn=collate_batch)
    # Fix E2: separate, non-shuffled loader over the held-out split. Falls
    # back to the training loader only if no eval split was provided at all
    # (keeps this function usable standalone without forcing a split).
    eval_loader = (
        DataLoader(eval_dataset, batch_size=batch_size, shuffle=False, collate_fn=collate_batch)
        if eval_dataset is not None else loader
    )

    # Fix B: only trainable (requires_grad=True) params get an optimizer slot.
    # Frozen params (the text encoder, when FREEZE_TEXT_ENCODER=true) never
    # receive gradients anyway, but excluding them here is explicit rather
    # than relying on AdamW silently no-op'ing on grad=None params.
    optimizer = AdamW([p for p in model.parameters() if p.requires_grad], lr=lr)

    # Fix E3, point 1: class weights derived from the ACTUAL dataset passed
    # in (never hardcoded), so they stay correct if the split ever changes.
    # Standard inverse-frequency ("balanced") scheme: weight[c] = N / (K *
    # count[c]) - makes each class contribute equally to the expected loss,
    # the standard correction for CrossEntropyLoss under class imbalance.
    train_labels = [
        1 if float((r.get("phq_score") or r.get("phq") or 0.0)) >= PHQ_POSITIVE_THRESHOLD else 0
        for r in dataset.records
    ]
    n_total = len(train_labels)
    n_pos = sum(train_labels)
    n_neg = n_total - n_pos
    if CLASS_WEIGHT_ENABLED and n_pos > 0 and n_neg > 0:
        w_neg = n_total / (2.0 * n_neg)
        w_pos = n_total / (2.0 * n_pos)
        class_weights = torch.tensor([w_neg, w_pos], dtype=torch.float32, device=device)
        rpt.kv("Class weights (neg, pos)", f"({w_neg:.4f}, {w_pos:.4f})  from train n_neg={n_neg} n_pos={n_pos}")
    else:
        class_weights = None
        rpt.kv("Class weights", "disabled (CLASS_WEIGHT_ENABLED=false or single-class split)")
    cls_loss_fn = nn.CrossEntropyLoss(weight=class_weights)
    reg_loss_fn = nn.MSELoss()

    rpt.subheader(f"TRAINING LOOP ({epochs} epoch(s))")
    rpt.kv("Regression loss weight (REG_LOSS_WEIGHT)", REG_LOSS_WEIGHT)
    rpt.kv("PHQ normalization scale", PHQ_SCALE_MAX)
    model.train()
    final_avg_loss = None
    _shapes_printed = False
    global_step = 0
    for epoch in range(epochs):
        total_loss = 0.0
        for b in loader:
            b = {k: (v.to(device) if isinstance(v, torch.Tensor) else v) for k, v in b.items()}
            if not _shapes_printed:
                print(f"[MULTIMODAL] text batch shape  = {tuple(b['input_ids'].shape)}")
                if b.get("audio_vec") is not None:
                    print(f"[MULTIMODAL] audio batch shape = {tuple(b['audio_vec'].shape)}")
                else:
                    print("[MULTIMODAL] audio batch shape = None (no audio in this batch)")
                if b.get("video_vec") is not None:
                    print(f"[MULTIMODAL] video batch shape = {tuple(b['video_vec'].shape)}")
                else:
                    print("[MULTIMODAL] video batch shape = None (no video in this batch)")
                _shapes_printed = True
            optimizer.zero_grad()
            logits, reg_pred, _ = model(b["input_ids"], b["attention_mask"],
                                        audio_vec=b.get("audio_vec"), vision_vec=b.get("video_vec"))
            loss_cls = cls_loss_fn(logits, b["label"])
            # Fix E3, point 2: MSE computed in normalized (phq/24) space, not
            # raw 0-23 PHQ units - raw-scale MSE is what produced the 5-40x
            # dominance measured in Step 9a. Normalizing by the fixed clinical
            # scale (not this corpus's observed max) is the principled fix
            # the coefficient below no longer has to compensate for a unit
            # mismatch, only for genuine task-importance weighting.
            loss_reg = reg_loss_fn(reg_pred / PHQ_SCALE_MAX, b["phq"] / PHQ_SCALE_MAX)
            loss_reg_weighted = REG_LOSS_WEIGHT * loss_reg
            loss = loss_cls + loss_reg_weighted
            loss.backward()

            # Fix E3, point 4: standing instrumentation (was ad hoc in Step
            # 9a). Gradient norms captured BEFORE clip_grad_norm_ - post-clip
            # they are uniformly rescaled to 1.0 and uninformative.
            if global_step == 0 or global_step % TRAIN_LOG_INTERVAL == 0:
                fc1_grad_norm = model.fusion.fc1.weight.grad.norm().item() if model.fusion.fc1.weight.grad is not None else float("nan")
                classifier_grad_norm = model.fusion.classifier.weight.grad.norm().item() if model.fusion.classifier.weight.grad is not None else float("nan")
                phq_mu_grad_norm = model.fusion.phq_mu.weight.grad.norm().item() if model.fusion.phq_mu.weight.grad is not None else float("nan")
                print(
                    f"[train_model] step={global_step} "
                    f"loss_cls={loss_cls.item():.4f} "
                    f"loss_reg(unweighted,norm)={loss_reg.item():.4f} "
                    f"loss_reg(weighted)={loss_reg_weighted.item():.4f} "
                    f"fc1_grad_norm={fc1_grad_norm:.4f} "
                    f"classifier_grad_norm={classifier_grad_norm:.4f} "
                    f"phq_mu_grad_norm={phq_mu_grad_norm:.4f}"
                )
                rpt.kv(
                    f"step {global_step}",
                    f"loss_cls={loss_cls.item():.4f} loss_reg(unweighted)={loss_reg.item():.4f} "
                    f"loss_reg(weighted)={loss_reg_weighted.item():.4f} "
                    f"fc1_grad={fc1_grad_norm:.4f} cls_grad={classifier_grad_norm:.4f} phq_mu_grad={phq_mu_grad_norm:.4f}",
                    indent=2,
                )

            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            total_loss += loss.item()
            global_step += 1
        final_avg_loss = total_loss / len(loader)
        print(f"[train_model] epoch {epoch+1}/{epochs} avg_loss={final_avg_loss:.4f}")
        rpt.kv(f"Epoch {epoch+1}/{epochs} average loss", f"{final_avg_loss:.4f}", indent=2)

    # ---------- Evaluation (Fix E2: on the held-out split, never on `loader`) ----------
    model.eval()
    y_true_cls, y_pred_cls = [], []
    y_true_phq, y_pred_phq = [], []
    with torch.no_grad():
        for b in eval_loader:
            b = {k: (v.to(device) if isinstance(v, torch.Tensor) else v) for k, v in b.items()}
            logits, reg_pred, _ = model(b["input_ids"], b["attention_mask"],
                                        audio_vec=b.get("audio_vec"), vision_vec=b.get("video_vec"))
            probs = torch.softmax(logits, dim=1)
            preds_cls = probs.argmax(dim=1)
            y_true_cls.extend(b["label"].cpu().tolist())
            y_pred_cls.extend(preds_cls.cpu().tolist())
            y_true_phq.extend(b["phq"].cpu().tolist())
            y_pred_phq.extend(reg_pred.cpu().tolist())

    acc = accuracy_score(y_true_cls, y_pred_cls) if y_true_cls else 0.0
    prec, rec, f1, _ = precision_recall_fscore_support(y_true_cls, y_pred_cls, average="binary") if y_true_cls else (0.0, 0.0, 0.0, None)
    mae = mean_absolute_error(y_true_phq, y_pred_phq) if y_true_phq else 0.0

    metrics = {
        "accuracy": float(acc),
        "precision": float(prec),
        "recall": float(rec),
        "f1": float(f1),
        "mae": float(mae),
    }

    # ---------- Explainability ----------
    explain_path = os.path.join(output_dir, "explain.txt")
    try:
        with open(explain_path, "w") as f:
            f.write("=== Explainability Report ===\n")
            f.write(f"Metrics: {json.dumps(metrics, indent=2)}\n\n")
            try:
                first_batch = next(iter(loader))
                importance = modality_ablation_importance(model, first_batch, device=device)
                f.write("Modality contributions (approx):\n")
                for k, v in importance["raw"].items():
                    f.write(f"  {k}: {v:.4f}\n")
            except Exception as e:
                f.write(f"[WARN] Explainability failed: {e}\n")
    except Exception as e:
        print(f"[WARN] Could not write explain.txt: {e}")

    # ---------- Save everything ----------
    model_path = os.path.join(output_dir, "mentalbert_privacy_subset.pt")
    torch.save(model.state_dict(), model_path)

    # ---- write metrics.json for downstream consumers ----
    metrics_json_path = os.path.join(output_dir, "metrics.json")
    try:
        with open(metrics_json_path, "w") as mf:
            json.dump(metrics, mf, indent=2)
        print(f"[train_model] metrics saved → {metrics_json_path}")
    except Exception as e:
        print(f"[WARN] Could not write metrics.json: {e}")

    # Also write a receipt-like file (existing behavior)
    with open(os.path.join(output_dir, "metrics_report.json"), "w") as f:
        json.dump({"model_path": model_path, "metrics": metrics}, f, indent=2)

    print(f"[train_model] model saved → {model_path}")
    return {
        "model_path": model_path,
        "metrics_path": metrics_json_path,
        "explain_path": explain_path,
        "metrics": metrics,
        "final_avg_loss": final_avg_loss,
    }

# ---------- RL Update (REINFORCE) ----------
class MovingBaseline:
    def __init__(self, momentum: float = 0.9):
        self.momentum = momentum
        self.value = 0.0
        self.inited = False

    def update(self, r: float):
        if not self.inited:
            self.value = r
            self.inited = True
        else:
            self.value = self.momentum * self.value + (1 - self.momentum) * r
        return self.value

def rl_update_reinforce(model: MultiModalModel, dataset: MultiModalDataset, epochs: int = 1, batch_size: int = 4, lr: float = 1e-5, device: str = DEFAULT_DEVICE, supervised_lambda: float = 0.0):
    """
    Simple REINFORCE per-sample using clinician-corrected PHQ as reward signal.
    supervised_lambda: mixing coefficient to add supervised MSE loss (helps stabilize).
    """
    model.to(device)
    loader = DataLoader(dataset, batch_size=1, shuffle=True, collate_fn=collate_batch)  # sample-level for RL
    # Fix B: only trainable (requires_grad=True) params get an optimizer slot.
    # Frozen params (the text encoder, when FREEZE_TEXT_ENCODER=true) never
    # receive gradients anyway, but excluding them here is explicit rather
    # than relying on AdamW silently no-op'ing on grad=None params.
    optimizer = AdamW([p for p in model.parameters() if p.requires_grad], lr=lr)
    baseline = MovingBaseline(momentum=0.9)
    model.train()

    for epoch in range(epochs):
        total_loss = 0.0
        steps = 0
        for batch in loader:
            # each batch will be size 1 (sample-level)
            batch = {k: (v.to(device) if isinstance(v, torch.Tensor) else v) for k, v in batch.items()}
            # sample an action
            logits, reg_pred, (action, log_prob) = model(batch["input_ids"], batch["attention_mask"],
                                                        audio_vec=batch.get("audio_vec"), vision_vec=batch.get("video_vec"),
                                                        rl_mode=True, sample_action=True)
            # clinician-corrected PHQ is in batch["phq"]
            target_phq = batch["phq"].squeeze(0).item()
            # compute reward in [0,1]
            error = abs(action.squeeze(0).item() - target_phq)
            norm_error = min(error / RL_PHQ_RANGE, 1.0)
            reward = 1.0 - norm_error

            # baseline subtraction
            b = baseline.update(reward)
            adv = reward - b

            # REINFORCE loss (negative reward-weighted logprob)
            logp = log_prob.squeeze(0)
            rl_loss = -adv * logp.mean()

            # optional supervised MSE loss
            mse_loss = nn.functional.mse_loss(action.squeeze(0), batch["phq"])
            loss = rl_loss + supervised_lambda * mse_loss

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

            total_loss += loss.item()
            steps += 1
        if steps:
            print(f"[rl] epoch {epoch+1}/{epochs} avg_loss={total_loss/steps:.6f}")
    return model


# ---------- Delta computation & safety ----------
def compute_state_delta(before: Dict[str, torch.Tensor], after: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
    delta = {}
    for k in after:
        if k in before and before[k].shape == after[k].shape:
            delta[k] = (after[k].detach().cpu() - before[k].detach().cpu()).clone()
        else:
            delta[k] = after[k].detach().cpu().clone()
    return delta


# Fix B: sanity ceiling for the trainable-delta param count. Real ceiling is
# audio_encoder + vision_encoder + fusion = 281,254 params (measured); this
# constant is intentionally looser so it doesn't need updating for small
# architecture tweaks, while still catching the failure mode that matters -
# the text encoder (109,482,240 params) ending up in the delta because the
# freeze or the filter silently stopped working.
_TRAINABLE_DELTA_PARAM_CEILING = 1_000_000


def compute_filtered_delta(
    base_state: Dict[str, torch.Tensor],
    after: Dict[str, torch.Tensor],
    model: "MultiModalModel",
) -> Dict[str, torch.Tensor]:
    """
    Fix B: restrict the delta to trainable parameters only.

    state_dict() always returns every parameter regardless of requires_grad -
    setting requires_grad_(False) on the text encoder does NOT by itself
    shrink the delta, the DP noise surface, or the upload payload. This is
    the actual filter: the trainable key set is derived fresh from
    model.named_parameters() (never a hardcoded key list or string-prefix
    match), so it automatically tracks whatever is actually frozen right now.

    Fails loudly (AssertionError) rather than silently drifting if:
      - the resulting delta contains any key outside the trainable set, or
      - the resulting delta is missing any trainable key, or
      - the delta's total param count exceeds a sanity ceiling while
        FREEZE_TEXT_ENCODER is enabled (the text encoder ending up in the
        delta despite the freeze being "on").

    model.state_dict() has zero non-parameter buffers in this architecture
    (verified empirically: state_dict().keys() == named_parameters() keys,
    215 == 215 for the full model) - audio_encoder/vision_encoder/fusion are
    plain Linear/ReLU/Dropout stacks with no BatchNorm/LayerNorm running
    stats and no registered buffers, so named_parameters()-based filtering
    captures 100% of what these submodules need. If a future architecture
    change adds a buffer to a trainable submodule, it would silently NOT
    appear in this delta - there is no buffer-aware fallback here by design,
    since none is currently needed.
    """
    trainable_keys = {n for n, p in model.named_parameters() if p.requires_grad}
    base_trainable  = {k: v for k, v in base_state.items() if k in trainable_keys}
    after_trainable = {k: v for k, v in after.items()      if k in trainable_keys}

    delta = compute_state_delta(base_trainable, after_trainable)

    delta_keys = set(delta.keys())
    extra   = delta_keys - trainable_keys
    missing = trainable_keys - delta_keys
    assert not extra, (
        f"delta contains {len(extra)} key(s) outside the trainable set - "
        f"the trainable-key filter is broken: {sorted(extra)[:5]}"
    )
    assert not missing, (
        f"delta is missing {len(missing)} trainable key(s) - a trainable "
        f"parameter silently vanished from the delta: {sorted(missing)[:5]}"
    )

    delta_param_count = sum(v.numel() for v in delta.values())
    if FREEZE_TEXT_ENCODER:
        assert delta_param_count < _TRAINABLE_DELTA_PARAM_CEILING, (
            f"delta has {delta_param_count:,} trainable params - expected "
            f"< {_TRAINABLE_DELTA_PARAM_CEILING:,} with FREEZE_TEXT_ENCODER=true. "
            f"The text encoder is likely not frozen."
        )

    return delta


# Fix E2: fraction of each class held out for evaluation, and the fixed seed
# that makes the split reproducible. Both intentionally module-level
# constants rather than buried literals, so they're visible and overridable
# the same way the other Fix A/B/C/D knobs are.
EVAL_SPLIT_FRACTION = 0.2
EVAL_SPLIT_SEED = 42


def stratified_split(
    records: List[Dict[str, Any]],
    test_frac: float = EVAL_SPLIT_FRACTION,
    seed: int = EVAL_SPLIT_SEED,
) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    """
    Fix E2: deterministic, seeded, stratified train/eval split.

    Splits `records` into (train_records, eval_records), preserving the
    positive/negative ratio (label = phq_score >= PHQ_POSITIVE_THRESHOLD) in
    BOTH splits independently - each class's indices are shuffled separately
    under the same fixed seed before slicing, so the minority (positive)
    class isn't at the mercy of a single global shuffle landing badly.

    Deterministic by construction: same records + same seed always produces
    the identical split. This is required, not incidental - every client
    currently loads the same shared parquet (see orchestrate()'s docstring
    disclosures elsewhere in this project), so a non-deterministic split
    would make one run's numbers incomparable to the next, and in any future
    genuinely-partitioned multi-client scenario would risk different clients
    drawing different train/eval boundaries over data they happen to share.

    Must be called AFTER label assembly (every record needs a real, final
    phq_score already set) - never before.
    """
    pos_idx = [i for i, r in enumerate(records) if float(r["phq_score"]) >= PHQ_POSITIVE_THRESHOLD]
    neg_idx = [i for i, r in enumerate(records) if float(r["phq_score"]) < PHQ_POSITIVE_THRESHOLD]

    rng = random.Random(seed)
    rng.shuffle(pos_idx)
    rng.shuffle(neg_idx)

    n_pos_eval = max(1, round(len(pos_idx) * test_frac)) if pos_idx else 0
    n_neg_eval = max(1, round(len(neg_idx) * test_frac)) if neg_idx else 0

    eval_idx = set(pos_idx[:n_pos_eval]) | set(neg_idx[:n_neg_eval])
    train_records = [r for i, r in enumerate(records) if i not in eval_idx]
    eval_records  = [r for i, r in enumerate(records) if i in eval_idx]
    return train_records, eval_records


# Step 20: env-gated, default unset ("off"). With CLIENT_SHARD_ID or
# CLIENT_N_SHARDS unset, orchestrate() never calls stratified_shard() at all
# - train_records stays exactly what stratified_split() returned, byte-
# identical to every measurement from Step 12 onward. CLIENT_SHARD_SEED has a
# fixed default so a sharded run is reproducible without the caller having to
# supply one; it is inert unless sharding is actually enabled.
CLIENT_SHARD_ID      = os.environ.get("CLIENT_SHARD_ID")
CLIENT_N_SHARDS       = os.environ.get("CLIENT_N_SHARDS")
CLIENT_SHARD_SEED     = int(os.environ.get("CLIENT_SHARD_SEED", "20240"))
CLIENT_NONIID_ALPHA   = os.environ.get("CLIENT_NONIID_ALPHA")


def _chunk_evenly(idx_list: List[int], n_shards: int) -> List[List[int]]:
    """n_shards contiguous pieces differing in size by at most 1 element."""
    n = len(idx_list)
    base, rem = divmod(n, n_shards)
    chunks, start = [], 0
    for i in range(n_shards):
        size = base + (1 if i < rem else 0)
        chunks.append(idx_list[start:start + size])
        start += size
    return chunks


def _chunks_from_counts(idx_list: List[int], counts: List[int]) -> List[List[int]]:
    chunks, start = [], 0
    for c in counts:
        chunks.append(idx_list[start:start + c])
        start += c
    return chunks


def _dirichlet_sample(rng: random.Random, alpha: float, k: int) -> List[float]:
    """Dirichlet(alpha,...,alpha) sample of dimension k via the standard
    normalised-Gamma construction (draw k independent Gamma(alpha,1) values,
    normalise by their sum) - stdlib-only (random.Random.gammavariate), no
    numpy dependency, deterministic under a seeded random.Random instance."""
    gammas = [rng.gammavariate(alpha, 1.0) for _ in range(k)]
    total = sum(gammas)
    if total <= 0:  # astronomically unlikely for alpha > 0; fail safe to uniform
        return [1.0 / k] * k
    return [g / total for g in gammas]


def _proportions_to_counts(proportions: List[float], total: int) -> List[int]:
    """Largest-remainder rounding: proportions * total, rounded down, then
    the leftover units go to the largest fractional remainders - guarantees
    sum(counts) == total exactly, unlike naive round()."""
    raw = [p * total for p in proportions]
    counts = [int(x) for x in raw]
    remainder = total - sum(counts)
    order = sorted(range(len(raw)), key=lambda i: raw[i] - counts[i], reverse=True)
    for i in range(remainder):
        counts[order[i]] += 1
    return counts


def stratified_shard(
    train_records: List[Dict[str, Any]],
    n_shards: int,
    shard_id: int,
    seed: int,
    noniid_alpha: Optional[float] = None,
    max_resample_attempts: int = 100,
) -> List[Dict[str, Any]]:
    """
    Step 20: N-way partition of an ALREADY-COMPUTED train split. Never called
    on, or given, eval_records - see stratified_split()'s docstring and the
    single call site in orchestrate(), which only ever shards the 149-record
    train_records stratified_split() itself already carved out. The held-out
    37 is never touched by this function and stays global and identical
    across every client, every shard, every run.

    IID mode (noniid_alpha=None): each class's indices are shuffled once
    (same discipline as stratified_split()) and divided into n_shards
    near-equal contiguous pieces (_chunk_evenly) - shard sizes differ by at
    most 1 record per class, each shard's class ratio tracks the corpus's
    own ratio closely.

    Non-IID mode (noniid_alpha set): each shard's SHARE of each class is
    drawn from a Dirichlet(noniid_alpha) distribution over n_shards - low
    alpha concentrates a class into fewer shards (label skew), high alpha
    approaches the IID split. Uses ALL real records; nothing is oversampled,
    undersampled, or synthesised - only how the same records are divided
    across shards changes.

    Every client calls this with the SAME (n_shards, seed, noniid_alpha) and
    a DIFFERENT shard_id - the full n_shards-way partition is recomputed
    identically by each call (deterministic in everything except shard_id),
    so shards are guaranteed disjoint and exhaustive without any
    cross-client coordination.

    If a draw would leave any shard single-class or empty, the ENTIRE
    partition (not just this shard) is redrawn with seed+1, seed+2, ... -
    all clients redraw identically since none of their inputs differ except
    shard_id, so shards stay consistent with each other even after a
    resample. Logged when it happens. Raises AssertionError if no valid
    partition is found within max_resample_attempts.
    """
    assert 0 <= shard_id < n_shards, f"shard_id={shard_id} out of range for n_shards={n_shards}"

    pos_idx_base = [i for i, r in enumerate(train_records) if float(r["phq_score"]) >= PHQ_POSITIVE_THRESHOLD]
    neg_idx_base = [i for i, r in enumerate(train_records) if float(r["phq_score"]) < PHQ_POSITIVE_THRESHOLD]

    for attempt in range(max_resample_attempts):
        trial_seed = seed + attempt
        rng = random.Random(trial_seed)
        pos_idx = pos_idx_base[:]
        neg_idx = neg_idx_base[:]
        rng.shuffle(pos_idx)
        rng.shuffle(neg_idx)

        if noniid_alpha is None:
            pos_chunks = _chunk_evenly(pos_idx, n_shards)
            neg_chunks = _chunk_evenly(neg_idx, n_shards)
        else:
            pos_counts = _proportions_to_counts(_dirichlet_sample(rng, noniid_alpha, n_shards), len(pos_idx))
            neg_counts = _proportions_to_counts(_dirichlet_sample(rng, noniid_alpha, n_shards), len(neg_idx))
            pos_chunks = _chunks_from_counts(pos_idx, pos_counts)
            neg_chunks = _chunks_from_counts(neg_idx, neg_counts)

        if all(len(pos_chunks[s]) > 0 and len(neg_chunks[s]) > 0 for s in range(n_shards)):
            if attempt > 0:
                print(f"[STEP20-SHARD] resampled partition after {attempt} retr{'y' if attempt == 1 else 'ies'} "
                      f"(seed {seed} -> {trial_seed}) to avoid a single-class/empty shard")
            shard_indices = sorted(set(pos_chunks[shard_id]) | set(neg_chunks[shard_id]))
            return [train_records[i] for i in shard_indices]

    raise AssertionError(
        f"stratified_shard: no partition with n_shards={n_shards}, alpha={noniid_alpha} "
        f"avoiding a single-class/empty shard was found in {max_resample_attempts} attempts "
        f"starting at seed={seed}. With {len(pos_idx_base)} positive / {len(neg_idx_base)} "
        f"negative train records, this alpha/n_shards combination may be too extreme."
    )


def apply_safety_to_delta(delta: Dict[str, torch.Tensor], max_param_change: float = DEFAULT_MAX_PARAM_CHANGE, max_global_norm: float = DEFAULT_MAX_GLOBAL_DELTA_NORM) -> Dict[str, torch.Tensor]:
    # Fix E5: these two clamps are meant to be rare safety backstops, not
    # routine limiters - engagement must be visible, not silent. This is
    # exactly the check that would have caught DEFAULT_MAX_PARAM_CHANGE
    # going stale (33% of params clamped every run) immediately instead of
    # being found after the fact during an unrelated investigation.
    n_params_total = sum(t.numel() for t in delta.values())

    # per-parameter clamp
    n_params_clamped = 0
    for k in list(delta.keys()):
        t = delta[k]
        n_params_clamped += (t.abs() > max_param_change).sum().item()
        delta[k] = t.clamp(min=-max_param_change, max=max_param_change)

    if n_params_clamped > 0:
        frac = n_params_clamped / max(n_params_total, 1)
        msg = (f"per-parameter safety clamp ENGAGED: {n_params_clamped:,}/{n_params_total:,} "
               f"params ({frac*100:.2f}%) exceeded +/-{max_param_change} and were clamped")
        print(f"[SAFETY-CLAMP] {msg}")
        rpt.warn(msg)

    # compute global norm
    total_sq = 0.0
    for k in delta:
        total_sq += (delta[k].float().norm() ** 2).item()
    global_norm = math.sqrt(total_sq)

    if global_norm > max_global_norm:
        scale = max_global_norm / (global_norm + 1e-12)
        msg = (f"global-norm safety clamp ENGAGED: delta L2={global_norm:.6f} exceeded "
               f"max_global_norm={max_global_norm}, rescaled by factor {scale:.6f}")
        print(f"[SAFETY-CLAMP] {msg}")
        rpt.warn(msg)
        for k in delta:
            delta[k] = (delta[k].float() * scale).clone()
    return delta

def save_encrypted_delta(delta_state: Dict[str, torch.Tensor], store: SecureStore, session_id: str, rm: CentralReceiptManager) -> Tuple[str, str]:
    import io, torch as _torch
    buf = io.BytesIO()
    _torch.save(delta_state, buf)
    payload = buf.getvalue()
    rel = f"{session_id}/local_updates/{os.urandom(8).hex()}.pt.enc"
    uri = store.encrypt_write(f"file://{store.root / rel}", payload)
    receipt = rm.create_receipt(
        agent="trainer-agent",
        operation="local_update",
        params={"size_bytes": len(payload)},
        outputs=[uri],
        session_id=session_id,
    )
    rrel = f"{session_id}/receipts/trainer_update_{os.urandom(6).hex()}.json.enc"
    ruri = store.encrypt_write(f"file://{store.root / rrel}", json.dumps(receipt).encode())
    return uri, ruri


# ---------- Explainability (modality ablation) ----------
def modality_ablation_importance(
    model: MultiModalModel, sample_batch: Dict[str, torch.Tensor], device: str = DEFAULT_DEVICE,
    audio_baseline: Optional[torch.Tensor] = None,
    vision_baseline: Optional[torch.Tensor] = None,
    text_baseline_embedding: Optional[torch.Tensor] = None,
):
    """
    audio_baseline / vision_baseline / text_baseline_embedding (Step A5):
    optional in-distribution "removed" values to ablate TO, replacing the
    original zero-vector / empty-string ablation. Default None preserves the
    EXACT original zero/empty-string behaviour unchanged - this keeps the
    pre-existing, unconditional "autonomous" mode call site
    (orchestrate(), mode="autonomous") byte-for-byte identical to before
    Step A5, since it does not pass these new parameters. Callers that want
    the corrected, scale-consistent behaviour (Step A6's diagnostic, and any
    future opt-in) pass compute_modality_baselines()'s output explicitly.
    See docs/IMPLEMENTATION_NOTES.md "Step A4/A5" for why zero was found to
    be a biased baseline for audio/vision (16.6x / 11.1x scale mismatch vs
    text) and why this is gated per-call rather than a global default.
    audio_baseline/vision_baseline: 1-D tensors broadcastable to the batch's
    audio_vec/video_vec shape. text_baseline_embedding: 1-D [hidden] tensor,
    broadcast across every token position.
    """
    model.eval()
    with torch.no_grad():
        inputs = {k: (v.to(device) if isinstance(v, torch.Tensor) else v) for k, v in sample_batch.items()}
        logits, reg, _ = model(inputs["input_ids"], inputs["attention_mask"], audio_vec=inputs.get("audio_vec"), vision_vec=inputs.get("video_vec"))
        base_pos = torch.softmax(logits, dim=1)[:, 1].mean().item()
        base_phq = reg.mean().item()
        results = {}
        # audio ablation
        if inputs.get("audio_vec") is not None:
            if audio_baseline is not None:
                zero_a = audio_baseline.to(device).unsqueeze(0).expand_as(inputs["audio_vec"])
            else:
                zero_a = torch.zeros_like(inputs["audio_vec"])
            l_a, r_a, _ = model(inputs["input_ids"], inputs["attention_mask"], audio_vec=zero_a, vision_vec=inputs.get("video_vec"))
            p_a = torch.softmax(l_a, dim=1)[:, 1].mean().item()
            r_a_val = r_a.mean().item()
            results["audio_posdelta"] = abs(base_pos - p_a)
            results["audio_phqdelta"] = abs(base_phq - r_a_val)
        else:
            results["audio_posdelta"] = 0.0
            results["audio_phqdelta"] = 0.0
        # vision ablation
        if inputs.get("video_vec") is not None:
            if vision_baseline is not None:
                zero_v = vision_baseline.to(device).unsqueeze(0).expand_as(inputs["video_vec"])
            else:
                zero_v = torch.zeros_like(inputs["video_vec"])
            l_v, r_v, _ = model(inputs["input_ids"], inputs["attention_mask"], audio_vec=inputs.get("audio_vec"), vision_vec=zero_v)
            p_v = torch.softmax(l_v, dim=1)[:, 1].mean().item()
            r_v_val = r_v.mean().item()
            results["vision_posdelta"] = abs(base_pos - p_v)
            results["vision_phqdelta"] = abs(base_phq - r_v_val)
        else:
            results["vision_posdelta"] = 0.0
            results["vision_phqdelta"] = 0.0
        # text ablation: mean-embedding baseline (Step A5) if given, else the
        # original empty-string baseline (unchanged default)
        try:
            if text_baseline_embedding is not None:
                seq_len = inputs["input_ids"].shape[1]
                bsz = inputs["input_ids"].shape[0]
                mean_embeds = text_baseline_embedding.to(device).view(1, 1, -1).expand(bsz, seq_len, -1)
                l_t, r_t, _ = model(inputs["input_ids"], inputs["attention_mask"], audio_vec=inputs.get("audio_vec"),
                                     vision_vec=inputs.get("video_vec"), inputs_embeds=mean_embeds)
            else:
                tokenizer = AutoTokenizer.from_pretrained(MENTALBERT_PRETRAIN)
                empty = tokenizer([""] * inputs["input_ids"].shape[0], padding=True, truncation=True, return_tensors="pt")
                empty_ids = empty["input_ids"].to(device)
                empty_mask = empty["attention_mask"].to(device)
                l_t, r_t, _ = model(empty_ids, empty_mask, audio_vec=inputs.get("audio_vec"), vision_vec=inputs.get("video_vec"))
            p_t = torch.softmax(l_t, dim=1)[:, 1].mean().item()
            r_t_val = r_t.mean().item()
            results["text_posdelta"] = abs(base_pos - p_t)
            results["text_phqdelta"] = abs(base_phq - r_t_val)
        except Exception:
            results["text_posdelta"] = 0.0
            results["text_phqdelta"] = 0.0

        agg = {
            "audio_score": (results["audio_posdelta"] + results["audio_phqdelta"]) / 2.0,
            "vision_score": (results["vision_posdelta"] + results["vision_phqdelta"]) / 2.0,
            "text_score": (results["text_posdelta"] + results["text_phqdelta"]) / 2.0,
            "raw": results
        }
        return agg


# ---------- Phase A / STEP A5: in-distribution modality baselines ----------
XAI_BASELINE_CACHE_PATH = Path.home() / ".federated" / "data" / "xai_baselines" / "modality_baselines.pt"


def compute_modality_baselines(
    train_records: List[Dict[str, Any]],
    tokenizer,
    device: str,
    cache_path: Optional[Path] = None,
    force_recompute: bool = False,
) -> Dict[str, Any]:
    """
    Step A5: in-distribution baseline vectors for XAI attribution, computed
    once over the TRAIN split only (never eval - using eval data to build a
    baseline would leak information about the held-out set into the
    "removed" reference point). Cached to disk so it is deterministic and
    reused across runs/clients rather than recomputed from randomness.

    Replaces the zero-vector baseline for audio/vision found in Step A4 to
    be a large, systematically biased perturbation (audio_vec's real
    magnitude is ~16.6x text's and ~11.1x vision's, so zero is a far more
    extreme, out-of-distribution "removed" state for audio than for the
    other two modalities - this, not output scale, was the actual root
    cause of Step A3's audio-dominant result). For full consistency, also
    replaces text's PAD-token baseline with a genuine mean-embedding
    baseline: PAD is in-distribution in the sense of being a real
    embedding-table row, but it is one specific token's embedding, not a
    distributional average - the audio/vision baselines below ARE averages,
    so text is switched to match (mean of real, attended/non-PAD token
    embeddings across the train corpus).

    All three baselines are computed independent of any locally fine-tuned
    weights: audio_mean/vision_mean are raw MultiModalDataset feature
    averages, and text_mean_embedding only touches the FROZEN (Fix B) BERT
    embeddings layer. This makes the cached baselines deterministic and
    identical across every client/round that shares the same pretrained
    MENTALBERT_PRETRAIN checkpoint.
    """
    cache_path = Path(cache_path) if cache_path is not None else XAI_BASELINE_CACHE_PATH
    if cache_path.exists() and not force_recompute:
        return torch.load(cache_path, map_location=device)

    from transformers import AutoModel
    bert = AutoModel.from_pretrained(MENTALBERT_PRETRAIN).to(device)
    bert.eval()
    for p in bert.parameters():
        p.requires_grad_(False)

    ds_train = MultiModalDataset(train_records, tokenizer, max_len=MULTIMODAL_MAX_LEN)
    audio_sum = vision_sum = text_embed_sum = None
    text_token_count = 0
    n = len(ds_train)

    with torch.no_grad():
        for i in range(n):
            item = ds_train[i]
            input_ids = item["input_ids"].unsqueeze(0).to(device)
            attention_mask = item["attention_mask"].to(device)
            emb = bert.embeddings(input_ids=input_ids)[0]        # [seq_len, hidden]
            real = emb[attention_mask.bool()]                     # [n_real_tokens, hidden]
            s = real.sum(dim=0)
            text_embed_sum = s if text_embed_sum is None else text_embed_sum + s
            text_token_count += real.size(0)

            a = item["audio_vec"].to(device)
            audio_sum = a.clone() if audio_sum is None else audio_sum + a
            v = item["video_vec"].to(device)
            vision_sum = v.clone() if vision_sum is None else vision_sum + v

    audio_mean = (audio_sum / n).cpu()
    vision_mean = (vision_sum / n).cpu()
    text_mean_embedding = (text_embed_sum / text_token_count).cpu()

    baselines = {
        "audio_mean": audio_mean,
        "vision_mean": vision_mean,
        "text_mean_embedding": text_mean_embedding,
        "num_train_records": n,
        "num_text_tokens_averaged": text_token_count,
        "audio_mean_l2_norm": audio_mean.norm().item(),
        "vision_mean_l2_norm": vision_mean.norm().item(),
        "text_mean_embedding_l2_norm": text_mean_embedding.norm().item(),
        "source": "stratified_split() train split, computed via compute_modality_baselines()",
    }
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(baselines, cache_path)
    return baselines


# ---------- Phase A / STEP A2: Integrated Gradients attribution ----------
XAI_SCOPE_DISCLAIMER = (
    "SCOPE: this attribution is computed on the LOCAL, PRE-DP model "
    "(post-fine-tuning, before clip/noise/DP, before encryption/upload) - "
    "the model that reaches roughly F1~0.57 on this client's held-out split. "
    "It is NOT computed on the aggregated global model produced under "
    "differential privacy: that model is degenerate on this corpus (F1=0, "
    "single-class predictions on all 37 held-out records - see Steps 12-18 "
    "in docs/IMPLEMENTATION_NOTES.md), so attribution on it would explain "
    "nothing meaningful. This is a deliberate, documented scope choice, not "
    "an oversight. Nothing in this report left the client device: it is not "
    "part of the uploaded payload, any receipt, or any gRPC message, and it "
    "has no effect on epsilon or the DP-noised delta."
)


def xai_integrated_gradients(
    model: "MultiModalModel",
    ds_eval,
    eval_records: List[Dict[str, Any]],
    tokenizer,
    session_id: str,
    device: str,
    eval_metrics: Dict[str, float],
    baselines: Dict[str, Any],
    n_steps: int = XAI_N_STEPS,
    out_dir: Optional[Path] = None,
) -> Dict[str, Any]:
    """
    Per-sample Integrated Gradients attribution over ds_eval's held-out
    records, on the model AS PASSED IN (caller's responsibility to pass the
    local, pre-DP model - see XAI_SCOPE_DISCLAIMER).

    Text: IntegratedGradients over the BERT embeddings-layer output, via
    MultiModalModel.forward's inputs_embeds parameter - gradients flow
    through the frozen encoder to that output (verified empirically, STEP
    A1) so Fix B's frozen text encoder does not block this. Audio/vision:
    IntegratedGradients jointly over (audio_vec, vision_vec), text held
    fixed at its real (non-baseline) embedding. Attribution target is the
    model's positive-class probability (softmax(logits)[:, 1]) - the same
    quantity modality_ablation_importance and physician_feedback_cli already
    treat as "the" prediction.

    baselines: output of compute_modality_baselines() (Step A5) - in-
    distribution mean vectors for text/audio/vision, replacing the original
    PAD-token / zero-vector baselines found in Step A4 to bias the result
    toward whichever modality has the largest raw input scale (audio,
    16.6x/11.1x larger than text/vision) rather than genuine importance.

    Writes explain_logs/xai_ig_<session_id>_<timestamp>.json and .txt to
    out_dir (default XAI_EXPLAIN_DIR). Returns the aggregate summary dict
    (also used by orchestrate() to print the modality split).
    """
    from captum.attr import IntegratedGradients

    out_dir = Path(out_dir) if out_dir is not None else XAI_EXPLAIN_DIR
    out_dir.mkdir(parents=True, exist_ok=True)

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
        audio_vec = item["audio_vec"].unsqueeze(0).to(device) if item.get("audio_vec") is not None else None
        video_vec = item["video_vec"].unsqueeze(0).to(device) if item.get("video_vec") is not None else None
        true_label = int(item["label"].item())

        with torch.no_grad():
            real_embeds = model.bert.embeddings(input_ids=input_ids)
            seq_len = input_ids.shape[1]
            baseline_embeds = text_mean_embedding.view(1, 1, -1).expand(1, seq_len, -1).contiguous()
            audio_baseline = audio_mean.unsqueeze(0).expand_as(audio_vec).contiguous() if audio_vec is not None else None
            vision_baseline = vision_mean.unsqueeze(0).expand_as(video_vec).contiguous() if video_vec is not None else None
            base_probs = torch.softmax(
                model(input_ids, attention_mask, audio_vec=audio_vec, vision_vec=video_vec)[0], dim=1
            )
            pred_prob_pos = base_probs[0, 1].item()

        attr_embeds = ig_text.attribute(
            inputs=real_embeds, baselines=baseline_embeds,
            additional_forward_args=(audio_vec, video_vec, attention_mask, input_ids),
            n_steps=n_steps,
        )
        per_token = attr_embeds[0].sum(dim=-1)  # [seq_len], signed
        text_raw = per_token.abs().sum().item()

        av_inputs = tuple(v for v in (audio_vec, video_vec) if v is not None)
        av_baselines = tuple(v for v in (audio_baseline, vision_baseline) if v is not None)
        attr_av = ig_av.attribute(
            inputs=av_inputs, baselines=av_baselines,
            additional_forward_args=(real_embeds, attention_mask, input_ids),
            n_steps=n_steps,
        )
        if not isinstance(attr_av, tuple):
            attr_av = (attr_av,)
        audio_raw = attr_av[0][0].abs().sum().item() if audio_vec is not None else 0.0
        vision_raw = attr_av[1 if audio_vec is not None else 0][0].abs().sum().item() if video_vec is not None else 0.0

        total_raw = text_raw + audio_raw + vision_raw
        if total_raw > 0:
            norm = {"text": text_raw / total_raw, "audio": audio_raw / total_raw, "vision": vision_raw / total_raw}
        else:
            norm = {"text": 0.0, "audio": 0.0, "vision": 0.0}

        tokens = tokenizer.convert_ids_to_tokens(input_ids[0].detach().cpu().tolist())
        mask_list = attention_mask[0].detach().cpu().tolist()
        per_token_list = per_token.detach().cpu().tolist()
        token_scores = [(tok, sc) for tok, sc, m in zip(tokens, per_token_list, mask_list) if m == 1]
        top_tokens = sorted(token_scores, key=lambda x: abs(x[1]), reverse=True)[:10]

        record = eval_records[idx] if idx < len(eval_records) else {}
        record_id = record.get("participant_id") or record.get("session_id") or record.get("id") or f"eval_{idx}"

        per_sample.append({
            "eval_index": idx,
            "record_id": record_id,
            "true_label": true_label,
            "predicted_positive_prob": pred_prob_pos,
            "raw": {"text": text_raw, "audio": audio_raw, "vision": vision_raw},
            "normalized": norm,
            "top_tokens": [{"token": t, "score": s} for t, s in top_tokens],
        })

    n = len(per_sample)
    agg_raw = {
        k: sum(s["raw"][k] for s in per_sample) / n for k in ("text", "audio", "vision")
    } if n else {"text": 0.0, "audio": 0.0, "vision": 0.0}
    agg_total = sum(agg_raw.values())
    agg_normalized = (
        {k: v / agg_total for k, v in agg_raw.items()} if agg_total > 0
        else {"text": 0.0, "audio": 0.0, "vision": 0.0}
    )

    def _stats(key):
        vals = [s["normalized"][key] for s in per_sample]
        if not vals:
            return {"mean": 0.0, "std": 0.0, "min": 0.0, "max": 0.0}
        mean = sum(vals) / len(vals)
        var = sum((v - mean) ** 2 for v in vals) / len(vals)
        return {"mean": mean, "std": var ** 0.5, "min": min(vals), "max": max(vals)}

    variability = {k: _stats(k) for k in ("text", "audio", "vision")}

    summary = {
        "scope_disclaimer": XAI_SCOPE_DISCLAIMER,
        "session_id": session_id,
        "generated_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "eval_metrics": eval_metrics,
        "n_steps": n_steps,
        "num_samples": n,
        "baseline_kind": "in-distribution mean vectors (Step A5) - text: mean of real "
                          "attended-token embeddings over train; audio/vision: mean feature "
                          "vector over train. Replaces the earlier zero/PAD baseline found "
                          "in Step A4 to bias attribution toward audio's larger raw scale.",
        "baseline_l2_norms": {
            "text_mean_embedding": baselines.get("text_mean_embedding_l2_norm"),
            "audio_mean": baselines.get("audio_mean_l2_norm"),
            "vision_mean": baselines.get("vision_mean_l2_norm"),
        },
        "aggregate_raw": agg_raw,
        "aggregate_normalized": agg_normalized,
        "per_sample_variability_of_normalized_scores": variability,
        "per_sample": per_sample,
    }

    ts_ms = int(time.time() * 1000)
    base_name = f"xai_ig_{session_id}_{ts_ms}"
    json_path = out_dir / f"{base_name}.json"
    txt_path = out_dir / f"{base_name}.txt"

    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, default=float)

    lines = []
    lines.append("=" * 78)
    lines.append("PHASE A XAI REPORT - INTEGRATED GRADIENTS MODALITY & TOKEN ATTRIBUTION")
    lines.append("=" * 78)
    lines.append("")
    lines.append(XAI_SCOPE_DISCLAIMER)
    lines.append("")
    lines.append(f"session_id      : {session_id}")
    lines.append(f"generated_at    : {summary['generated_at']}")
    lines.append(f"n_steps         : {n_steps}")
    lines.append(f"num_samples     : {n}")
    lines.append(f"eval_metrics    : {eval_metrics}")
    lines.append(f"baseline_kind   : {summary['baseline_kind']}")
    lines.append(f"baseline_l2_norms: {summary['baseline_l2_norms']}")
    lines.append("")
    lines.append("-" * 78)
    lines.append("AGGREGATE MODALITY ATTRIBUTION (mean over held-out set)")
    lines.append("-" * 78)
    lines.append(f"  raw        : text={agg_raw['text']:.6f}  audio={agg_raw['audio']:.6f}  vision={agg_raw['vision']:.6f}")
    lines.append(
        f"  normalized : text={agg_normalized['text']:.4f}  audio={agg_normalized['audio']:.4f}  "
        f"vision={agg_normalized['vision']:.4f}"
    )
    lines.append("")
    lines.append("  per-sample variability of normalized scores (mean / std / min / max):")
    for k in ("text", "audio", "vision"):
        v = variability[k]
        lines.append(f"    {k:7s}: mean={v['mean']:.4f}  std={v['std']:.4f}  min={v['min']:.4f}  max={v['max']:.4f}")
    lines.append("")
    lines.append("-" * 78)
    lines.append(f"PER-SAMPLE DETAIL ({n} records)")
    lines.append("-" * 78)
    for s in per_sample:
        lines.append("")
        lines.append(
            f"[{s['eval_index']}] record={s['record_id']}  true_label={s['true_label']}  "
            f"pred_pos_prob={s['predicted_positive_prob']:.4f}"
        )
        lines.append(
            f"    normalized: text={s['normalized']['text']:.4f} audio={s['normalized']['audio']:.4f} "
            f"vision={s['normalized']['vision']:.4f}"
        )
        lines.append(
            f"    raw       : text={s['raw']['text']:.6f} audio={s['raw']['audio']:.6f} "
            f"vision={s['raw']['vision']:.6f}"
        )
        top_str = ", ".join(f"{t['token']}({t['score']:.4f})" for t in s["top_tokens"])
        lines.append(f"    top tokens: {top_str}")

    with open(txt_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))

    summary["json_path"] = str(json_path)
    summary["txt_path"] = str(txt_path)
    return summary


# ---------- Physician CLI for supervised correction ----------
PHQ_POSITIVE_THRESHOLD = 10.0  # matches label = 1 if phq_val >= 10.0 in MultiModalDataset

def physician_feedback_cli(preds: List[Dict[str, Any]], texts: List[str]) -> Tuple[List[float], List[bool]]:
    """
    Returns (corrected, provided_by_physician) - two parallel lists.

    corrected[i] is the value to use if the caller decides to apply it:
    the physician's typed number when they entered one, otherwise the
    model's own pre-training prediction (the only sane value to show/log,
    NOT a claim about ground truth).

    provided_by_physician[i] is True only when the physician typed a value
    that parsed as a float - i.e. an explicit, real correction. It is False
    for blank input AND for invalid/unparseable input. Fix E1: this flag is
    what the caller (orchestrate()) uses to distinguish "a human actually
    corrected this" from "nothing was entered" - the caller must never treat
    the fallback corrected[i] value (model's own guess) as a real label.
    """
    corrected = []
    provided_by_physician = []
    rpt.header("PHYSICIAN FEEDBACK LOOP")
    for i, (p, t) in enumerate(zip(preds, texts)):
        snippet = (t or "")[:260].replace("\n", " ")
        probs = p["pred_class_probs"]
        predicted_class = 1 if probs[1] >= 0.5 else 0
        rpt.subheader(f"Sample {i+1}")
        print(f"  Text (truncated) : {snippet} ...")
        rpt.kv("Raw model output (PHQ)", f"{p['pred_phq']:.2f}", indent=2)
        rpt.kv("Negative probability", f"{probs[0]:.3f}", indent=2)
        rpt.kv("Positive probability", f"{probs[1]:.3f}", indent=2)
        rpt.kv("Classification threshold", "0.5 (class prob) / PHQ>=10.0 (label)", indent=2)
        rpt.kv("Predicted class", predicted_class, indent=2)
        val = input("Enter corrected PHQ (or Enter to keep): ").strip()
        if val:
            try:
                corrected_val = float(val)
                corrected.append(corrected_val)
                provided_by_physician.append(True)
                rpt.kv("Physician correction", f"{corrected_val:.2f}", indent=2)
                rpt.kv("Correction applied", "YES", indent=2)
            except:
                print("invalid -> keeping model value")
                corrected.append(float(p["pred_phq"]))
                provided_by_physician.append(False)
                rpt.kv("Physician correction", "invalid input, discarded", indent=2)
                rpt.kv("Correction applied", "NO", indent=2)
        else:
            corrected.append(float(p["pred_phq"]))
            provided_by_physician.append(False)
            rpt.kv("Physician correction", "Not provided", indent=2)
            rpt.kv("Correction applied", "NO", indent=2)
    return corrected, provided_by_physician


# ---------- Orchestrator ----------
def orchestrate(
    input_path: str,
    session_id: str,
    mode: str = "autonomous",
    device: str = None,
    epochs: int = SUPERVISED_EPOCHS,
    batch_size: int = 8,
    lr: float = SUPERVISED_LR,
    rl_supervised_lambda: float = 0.0,
    max_samples=None,
    safety_params=None,
    global_model_path: str = None,   # ← Phase 10: now an explicit parameter
    round_id: int = None,            # ← Step 16: explicit, not inferred from global_model_path
    **kwargs
):
    """
    Orchestrate training/inference.

    global_model_path: optional path to a global model state_dict (.pt file)
                       received from the server.  When provided the model is
                       initialised from these weights before local fine-tuning,
                       implementing the federated averaging warm-start.
    round_id: optional (Step 16). When given, the effective supervised
                       learning rate is lr * (LR_DECAY ** (round_id - 1)) -
                       round 1 unaffected, later rounds decayed. round_id is
                       explicit and always testable, unlike inferring "this is
                       a warm-started round" from global_model_path being set
                       (which would silently do nothing if the global model
                       were ever absent for a later round).
    """
    import os, json, tempfile, torch
    from pathlib import Path
    from torch.utils.data import DataLoader

    # resolve device
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"

    # ── Step 16: round-aware LR decay ─────────────────────────────────────────
    # lr passed in is the BASE rate (SUPERVISED_LR unless overridden by the
    # caller); round_id, when given, decays it further. LR_DECAY=1.0 makes
    # this a no-op (lr_r == lr for every round), reproducing Step 14 exactly.
    effective_lr = lr
    if round_id is not None:
        effective_lr = lr * (LR_DECAY ** (round_id - 1))
    rpt.subheader("STEP 16 — ROUND-AWARE LR DECAY")
    rpt.kv("Round ID", round_id if round_id is not None else "(not provided)")
    rpt.kv("Base learning rate", lr)
    rpt.kv("LR_DECAY", LR_DECAY)
    rpt.kv("Effective learning rate this round", effective_lr)
    print(f"[STEP16-LR] round_id={round_id} base_lr={lr} LR_DECAY={LR_DECAY} effective_lr={effective_lr}")
    lr = effective_lr

    # ── inline imports so the rest of the file remains unchanged ─────────────
    from transformers import AutoTokenizer
 
    # These are defined in the same file — import via module self-reference
    import sys
    _mod = sys.modules[__name__]
 
    read_parquet_records    = _mod.read_parquet_records
    MultiModalDataset       = _mod.MultiModalDataset
    MultiModalModel         = _mod.MultiModalModel
    collate_batch           = _mod.collate_batch
    run_inference           = _mod.run_inference
    train_model             = _mod.train_model
    rl_update_reinforce     = _mod.rl_update_reinforce
    compute_state_delta     = _mod.compute_state_delta
    apply_safety_to_delta   = _mod.apply_safety_to_delta
    save_encrypted_delta    = _mod.save_encrypted_delta
    modality_ablation_importance = _mod.modality_ablation_importance
    physician_feedback_cli  = _mod.physician_feedback_cli
    SecureStore             = _mod.SecureStore
    CentralReceiptManager   = _mod.CentralReceiptManager
    MENTALBERT_PRETRAIN     = _mod.MENTALBERT_PRETRAIN
    LOCAL_SAVE_DIR          = _mod.LOCAL_SAVE_DIR
    DEFAULT_MAX_PARAM_CHANGE     = _mod.DEFAULT_MAX_PARAM_CHANGE
    DEFAULT_MAX_GLOBAL_DELTA_NORM = _mod.DEFAULT_MAX_GLOBAL_DELTA_NORM
 
    # ── load records ──────────────────────────────────────────────────────────
    if input_path.startswith("file://") and input_path.endswith(".enc"):
        _CANONICAL_STORE = Path.home() / ".federated" / "data" / "secure_store"
        store_lda = SecureStore(agent="lda", root=_CANONICAL_STORE)
        manifest_bytes = store_lda.decrypt_read(input_path)
        with tempfile.NamedTemporaryFile(suffix=".jsonl", delete=False) as tf:
            tf.write(manifest_bytes)
            local_manifest_path = tf.name
        records = read_parquet_records(local_manifest_path)
    else:
        records = read_parquet_records(input_path)
 
    if max_samples:
        records = records[:max_samples]
    print(f"[info] loaded {len(records)} records")
 
    tokenizer = AutoTokenizer.from_pretrained(MENTALBERT_PRETRAIN)
    ds        = MultiModalDataset(records, tokenizer, max_len=MULTIMODAL_MAX_LEN)
 
    # infer modality dims
    audio_dim  = None
    vision_dim = None
    for r in records:
        f     = r.get("features") or {}
        audio = f.get("audio")
        if isinstance(audio, dict):
            if isinstance(audio.get("wav2vec2"), list):
                audio_dim = len(audio["wav2vec2"])
                break
            numeric = [v for v in audio.values() if isinstance(v, (int, float))]
            if numeric:
                audio_dim = len(numeric)
                break
    for r in records:
        f = r.get("features") or {}
        if isinstance(f, dict) and "video" in f and isinstance(f["video"], dict):
            v = f["video"]
            if v.get("densenet") and isinstance(v["densenet"], (list, tuple)):
                vision_dim = len(v["densenet"])
                break
        if any(k.startswith("neuron_") for k in r.keys()):
            vision_dim = len(sorted(k for k in r if k.startswith("neuron_")))
            break
 
    print(f"[info] inferred audio_dim={audio_dim}, vision_dim={vision_dim}")

    rpt.subheader("MODEL CONFIGURATION")
    rpt.kv("Architecture", "MentalBERT + audio/vision fusion head")
    rpt.kv("Records loaded", len(records))
    rpt.kv("Audio dim", audio_dim)
    rpt.kv("Vision dim", vision_dim)

    # Step 12 / Defect B workaround (docs/IMPLEMENTATION_NOTES.md): only
    # active when GLOBAL_INIT_SEED is explicitly set. Seeds immediately
    # before construction so audio_encoder/vision_encoder/fusion's random
    # init is reproducible across processes given the same seed, then
    # re-randomises right after so training stochasticity still differs.
    if GLOBAL_INIT_SEED is not None:
        torch.manual_seed(int(GLOBAL_INIT_SEED))
        print(f"[STEP12-SEED] GLOBAL_INIT_SEED={GLOBAL_INIT_SEED} — model init made deterministic for this run")
        rpt.kv("Model init seed (GLOBAL_INIT_SEED)", GLOBAL_INIT_SEED)

    model = MultiModalModel(
        MENTALBERT_PRETRAIN,
        audio_dim=audio_dim,
        vision_dim=vision_dim,
        device=device
    )

    if GLOBAL_INIT_SEED is not None:
        reseed = torch.seed()
        print(f"[STEP12-SEED] RNG re-randomized after init (torch.seed()={reseed}) — training stochasticity independent per client")

    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    rpt.kv("Total parameters", f"{total_params:,}")
    rpt.kv("Trainable parameters", f"{trainable_params:,}")
    rpt.kv("Frozen parameters", f"{total_params - trainable_params:,}")
    print(f"[MULTIMODAL] audio encoder = {'ACTIVE' if model.has_audio else 'INACTIVE'}")
    print(f"[MULTIMODAL] video encoder = {'ACTIVE' if model.has_vision else 'INACTIVE'}")
    print(f"[MULTIMODAL] fusion input dimension = {model.fusion.fc1.in_features}")
    print(f"[MULTIMODAL] total parameters = {total_params:,}")

    # ── Phase 10: warm-start from global model ────────────────────────────────
    rpt.subheader("WARM-START")
    if global_model_path and Path(global_model_path).exists():
        try:
            global_state = torch.load(global_model_path, map_location=device)
            # Support both raw state_dict and wrapped {"state_dict": ...} formats
            if isinstance(global_state, dict) and "state_dict" in global_state:
                global_state = global_state["state_dict"]
            missing, unexpected = model.load_state_dict(global_state, strict=False)
            print(
                f"[FL] Warm-started from global model: "
                f"{len(missing)} missing keys, {len(unexpected)} unexpected keys"
            )
            rpt.kv("Initialization", "Global model (warm-start)")
            rpt.kv("Global model path", global_model_path)
            rpt.kv("Missing keys", len(missing))
            rpt.kv("Unexpected keys", len(unexpected))
            rpt.ok("Model loaded successfully")
        except Exception as e:
            print(f"[FL] Warning: could not load global model ({e}) — starting from scratch")
            rpt.warn(f"Could not load global model ({e}) — starting from scratch")
    else:
        if global_model_path:
            print(f"[FL] global_model_path provided but file not found: {global_model_path}")
            rpt.warn(f"global_model_path provided but file not found: {global_model_path}")
        print("[FL] No global model — using random initialisation")
        rpt.kv("Initialization", "Random / local pretrained (no global model)")
    # ─────────────────────────────────────────────────────────────────────────

    # ── Fix B: freeze the text encoder ──────────────────────────────────────
    # Applied unconditionally (independent of whether warm-start ran) so
    # behaviour is identical on the very first round and every round after.
    # audio_encoder + vision_encoder + fusion (281,254 params) stay trainable.
    rpt.subheader("FIX B — TEXT ENCODER FREEZE")
    if FREEZE_TEXT_ENCODER:
        for p in model.bert.parameters():
            p.requires_grad_(False)
        frozen_now = sum(p.numel() for p in model.bert.parameters())
        print(f"[FIX-B] Text encoder frozen: {frozen_now:,} params (FREEZE_TEXT_ENCODER=true)")
        rpt.kv("Text encoder frozen", f"YES ({frozen_now:,} params)")
    else:
        print("[FIX-B] FREEZE_TEXT_ENCODER=false — full model left trainable (comparison mode)")
        rpt.kv("Text encoder frozen", "NO (FREEZE_TEXT_ENCODER=false)")
    # ─────────────────────────────────────────────────────────────────────────

    model.to(device)
    base_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
 
    loader = DataLoader(ds, batch_size=batch_size, shuffle=False, collate_fn=collate_batch)
    preds  = run_inference(model, loader, device=device)
    for r, p in zip(records, preds):
        r["_pred_phq"]         = p["pred_phq"]
        r["_pred_class_probs"] = p["pred_class_probs"]
 
    store = SecureStore(agent="trainer", root=LOCAL_SAVE_DIR)
    rm    = CentralReceiptManager(agent="trainer-agent")
 
    if mode == "autonomous":
        sample = next(iter(loader), None)
        explanations = modality_ablation_importance(model, sample, device=device) if sample else {}
        payload   = json.dumps({"preds": preds, "explainability": explanations}, default=float).encode()
        out_rel   = f"{session_id}/inference/results_{os.urandom(6).hex()}.json.enc"
        out_uri   = store.encrypt_write(f"file://{store.root / out_rel}", payload)
        receipt   = rm.create_receipt(
            agent="trainer-agent", operation="inference",
            params={"count": len(preds)}, outputs=[out_uri], session_id=session_id,
        )
        rrel = f"{session_id}/receipts/inference_{os.urandom(6).hex()}.json.enc"
        ruri = store.encrypt_write(f"file://{store.root / rrel}", json.dumps(receipt).encode())
        print(f"[done] inference -> {out_uri}")
        return {"inference_uri": out_uri, "receipt_uri": ruri, "explainability": explanations}
 
    elif mode == "supervised":
        texts = [r.get("transcript") or r.get("text") or "" for r in records]

        # Fix E1: snapshot ground-truth presence and the parquet's real
        # label distribution BEFORE anything can be mutated, so the
        # assertion below has an independent baseline to check against.
        had_ground_truth = [r.get("phq_score") is not None for r in records]
        parquet_snapshot = [float(r["phq_score"]) if r.get("phq_score") is not None else None for r in records]
        parquet_positive_count = sum(1 for v in parquet_snapshot if v is not None and v >= PHQ_POSITIVE_THRESHOLD)
        parquet_negative_count = sum(1 for v in parquet_snapshot if v is not None and v < PHQ_POSITIVE_THRESHOLD)

        corrected_phq, provided_by_physician = physician_feedback_cli(preds, texts)

        # Fix E1: this loop is the actual bug fix. The old code unconditionally
        # did `r["phq_score"] = float(cp)` for every record, and cp defaults to
        # the model's own pre-training prediction whenever the physician left
        # the prompt blank (physician_feedback_cli's fallback) - which silently
        # replaced the real PHQ-8 label with the model's untrained guess on
        # every automated run (there is no human in this loop). Every metric
        # this project has reported to date was measured against those
        # self-referential fabricated labels, not the corpus.
        #
        # Fix: apply cp only when it represents a genuine input - either an
        # explicit physician correction (always respected, even overriding
        # real ground truth - this preserves the correction feature), or the
        # model's own fallback ONLY when there was no ground truth to begin
        # with (mirrors the RL branch's need_cli guard a few dozen lines
        # down). A record that already has a real label and gets blank
        # physician input keeps its real label, full stop.
        label_source = []   # "parquet" | "physician" | "model-fallback", per record
        n_physician_corrections = 0
        for i, r in enumerate(records):
            if provided_by_physician[i]:
                r["phq_score"] = float(corrected_phq[i])
                label_source.append("physician")
                n_physician_corrections += 1
            elif had_ground_truth[i]:
                label_source.append("parquet")   # untouched - this is the fix
            else:
                r["phq_score"] = float(corrected_phq[i])
                label_source.append("model-fallback")

        rpt.subheader("LABEL SOURCE (Fix E1)")
        rpt.kv("From parquet (ground truth preserved)", label_source.count("parquet"))
        rpt.kv("From physician (explicit correction)", label_source.count("physician"))
        rpt.kv("From model fallback (no ground truth, no correction)", label_source.count("model-fallback"))
        rpt.kv("Physician corrections applied", n_physician_corrections)

        final_positive = sum(1 for r in records if float(r["phq_score"]) >= PHQ_POSITIVE_THRESHOLD)
        final_negative = len(records) - final_positive
        rpt.kv("Final label distribution: positive", final_positive)
        rpt.kv("Final label distribution: negative", final_negative)

        # Fix E1: fail loudly, before training, if labels have collapsed to a
        # single class - this must never again be able to happen silently.
        assert final_positive > 0 and final_negative > 0, (
            f"Label set has collapsed to a single class going into training "
            f"(positive={final_positive}, negative={final_negative}, "
            f"n={len(records)}) - this is exactly the Fix E1 regression this "
            f"assertion exists to catch."
        )
        # When nothing legitimately altered the labels (no physician
        # corrections, no model-fallback records - i.e. every record had
        # real ground truth and kept it), the final distribution must match
        # the parquet's own distribution exactly.
        if n_physician_corrections == 0 and label_source.count("model-fallback") == 0:
            assert final_positive == parquet_positive_count and final_negative == parquet_negative_count, (
                f"Label distribution drifted from the parquet with zero "
                f"physician corrections and zero model-fallback labels "
                f"applied: final=({final_positive} pos / {final_negative} neg), "
                f"parquet=({parquet_positive_count} pos / {parquet_negative_count} neg). "
                f"Real ground truth is being altered somewhere other than an "
                f"explicit physician correction."
            )

        # Fix E2: stratified, seeded, deterministic train/eval split. Must run
        # after label assembly (above) so it splits on final, real labels.
        train_records, eval_records = stratified_split(records)
        train_pos = sum(1 for r in train_records if float(r["phq_score"]) >= PHQ_POSITIVE_THRESHOLD)
        train_neg = len(train_records) - train_pos
        eval_pos  = sum(1 for r in eval_records if float(r["phq_score"]) >= PHQ_POSITIVE_THRESHOLD)
        eval_neg  = len(eval_records) - eval_pos

        rpt.subheader("TRAIN/EVAL SPLIT (Fix E2)")
        rpt.kv("Split fraction (eval)", EVAL_SPLIT_FRACTION)
        rpt.kv("Split seed", EVAL_SPLIT_SEED)
        rpt.kv("Train size", f"{len(train_records)}  (positive={train_pos}, negative={train_neg})")
        rpt.kv("Eval size", f"{len(eval_records)}  (positive={eval_pos}, negative={eval_neg})")

        # Fix E2: fail loudly rather than silently evaluate (or train) on a
        # single-class split - this is a real risk on a small, imbalanced
        # corpus and must be caught before it can produce another
        # undefined-metric collapse.
        assert train_pos > 0 and train_neg > 0, (
            f"Train split has collapsed to a single class "
            f"(positive={train_pos}, negative={train_neg}, n={len(train_records)}) - "
            f"stratified_split() failed to preserve both classes."
        )
        assert eval_pos > 0 and eval_neg > 0, (
            f"Eval split has collapsed to a single class "
            f"(positive={eval_pos}, negative={eval_neg}, n={len(eval_records)}) - "
            f"stratified_split() failed to preserve both classes."
        )

        # Step 20: per-client sharding of TRAIN records only — eval_records
        # (the held-out 37, just asserted non-degenerate above) is never
        # passed to stratified_shard() and is untouched below. Default OFF:
        # with either env var unset, train_records is exactly what
        # stratified_split() returned, above — byte-identical to every
        # measurement from Step 12 onward.
        partition_mode = "full"
        if CLIENT_SHARD_ID is not None and CLIENT_N_SHARDS is not None:
            shard_id = int(CLIENT_SHARD_ID)
            n_shards = int(CLIENT_N_SHARDS)
            noniid_alpha = float(CLIENT_NONIID_ALPHA) if CLIENT_NONIID_ALPHA is not None else None
            partition_mode = "non-iid" if noniid_alpha is not None else "iid"
            train_records = stratified_shard(
                train_records, n_shards=n_shards, shard_id=shard_id,
                seed=CLIENT_SHARD_SEED, noniid_alpha=noniid_alpha,
            )
            train_pos = sum(1 for r in train_records if float(r["phq_score"]) >= PHQ_POSITIVE_THRESHOLD)
            train_neg = len(train_records) - train_pos
            assert train_pos > 0 and train_neg > 0, (
                f"Shard {shard_id}/{n_shards} collapsed to a single class "
                f"(positive={train_pos}, negative={train_neg}, n={len(train_records)}) - "
                f"stratified_shard()'s own resample-on-collapse should have prevented this."
            )
            achieved_ratio = train_pos / len(train_records) if train_records else 0.0

            rpt.subheader("STEP 20 — CLIENT DATA PARTITION")
            rpt.kv("Partition mode", partition_mode)
            rpt.kv("Shard ID", f"{shard_id}/{n_shards}")
            rpt.kv("Shard size", f"{len(train_records)}  (positive={train_pos}, negative={train_neg})")
            rpt.kv("Achieved positive ratio", f"{achieved_ratio:.4f}")
            rpt.kv("Shard seed", CLIENT_SHARD_SEED)
            if noniid_alpha is not None:
                rpt.kv("Dirichlet alpha", noniid_alpha)
            print(f"[STEP20-SHARD] mode={partition_mode} shard={shard_id}/{n_shards} "
                  f"size={len(train_records)} positive={train_pos} negative={train_neg} "
                  f"achieved_ratio={achieved_ratio:.4f} seed={CLIENT_SHARD_SEED} alpha={noniid_alpha}")
        else:
            rpt.subheader("STEP 20 — CLIENT DATA PARTITION")
            rpt.kv("Partition mode", "full (CLIENT_SHARD_ID/CLIENT_N_SHARDS unset — every client trains on all 149)")
            print(f"[STEP20-SHARD] mode=full size={len(train_records)} positive={train_pos} negative={train_neg}")

        rpt.subheader("SUPERVISED FINE-TUNING")
        rpt.kv("Epochs", epochs)
        rpt.kv("Batch size", batch_size)
        rpt.kv("Learning rate", lr)
        rpt.kv("Optimizer", "AdamW")
        t_train0 = time.time()
        ds_sup  = MultiModalDataset(train_records, tokenizer, max_len=MULTIMODAL_MAX_LEN)
        ds_eval = MultiModalDataset(eval_records, tokenizer, max_len=MULTIMODAL_MAX_LEN)
        result = train_model(ds_sup, model, output_dir=str(LOCAL_SAVE_DIR),
                             epochs=epochs, batch_size=batch_size, lr=lr, device=device,
                             eval_dataset=ds_eval)
        train_elapsed = time.time() - t_train0
        model.load_state_dict(torch.load(result["model_path"], map_location=device))
        after  = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
        delta  = compute_filtered_delta(base_state, after, model)

        delta_norm_before = math.sqrt(sum((v.float().norm() ** 2).item() for v in delta.values()))

        sparams = safety_params or {}
        max_param_change = sparams.get("max_param_change", DEFAULT_MAX_PARAM_CHANGE)
        max_global_norm  = sparams.get("max_global_norm", DEFAULT_MAX_GLOBAL_DELTA_NORM)
        delta_safe = apply_safety_to_delta(
            delta,
            max_param_change=max_param_change,
            max_global_norm=max_global_norm,
        )
        delta_norm_after = math.sqrt(sum((v.float().norm() ** 2).item() for v in delta_safe.values()))

        rpt.subheader("EVALUATION METRICS (post-training, on local data)")
        if result.get("final_avg_loss") is not None:
            rpt.kv("final_avg_loss", f"{result['final_avg_loss']:.4f}", indent=2)
        for mk, mv in result["metrics"].items():
            rpt.kv(mk, f"{mv:.4f}", indent=2)

        # ── Phase A / STEP A2: XAI attribution (local pre-DP model, held-out
        # data). Default off (XAI_ENABLED); diagnostic only - a captum or
        # attribution failure must never break training or the upload path
        # that follows this block, hence the broad except.
        rpt.subheader("XAI — INTEGRATED GRADIENTS (Phase A, local pre-DP model)")
        if XAI_ENABLED:
            try:
                t_xai0 = time.time()
                xai_baselines = compute_modality_baselines(train_records, tokenizer, device)
                rpt.kv("Baseline source", xai_baselines["source"], indent=2)
                rpt.kv("Baseline L2 norms",
                       f"text={xai_baselines['text_mean_embedding_l2_norm']:.2f} "
                       f"audio={xai_baselines['audio_mean_l2_norm']:.2f} "
                       f"vision={xai_baselines['vision_mean_l2_norm']:.2f}", indent=2)
                xai_summary = xai_integrated_gradients(
                    model, ds_eval, eval_records, tokenizer, session_id, device,
                    eval_metrics=result["metrics"], baselines=xai_baselines, n_steps=XAI_N_STEPS,
                )
                xai_elapsed = time.time() - t_xai0
                ar = xai_summary["aggregate_raw"]
                an = xai_summary["aggregate_normalized"]
                print(f"[STEP-A2-XAI] {xai_summary['num_samples']} samples attributed in {xai_elapsed:.2f}s "
                      f"(n_steps={XAI_N_STEPS})")
                print(f"[STEP-A2-XAI] aggregate modality split (normalized): "
                      f"text={an['text']:.4f} audio={an['audio']:.4f} vision={an['vision']:.4f}")
                rpt.kv("Samples attributed", xai_summary["num_samples"], indent=2)
                rpt.kv("Runtime", f"{xai_elapsed:.2f} sec", indent=2)
                rpt.kv("Modality split (normalized)",
                       f"text={an['text']:.4f} audio={an['audio']:.4f} vision={an['vision']:.4f}", indent=2)
                rpt.kv("Modality split (raw)",
                       f"text={ar['text']:.6f} audio={ar['audio']:.6f} vision={ar['vision']:.6f}", indent=2)
                rpt.kv("JSON report", xai_summary["json_path"], indent=2)
                rpt.kv("Text report", xai_summary["txt_path"], indent=2)
                rpt.ok("XAI attribution written to local disk (explain_logs/) — never uploaded")
            except Exception as e:
                print(f"[STEP-A2-XAI] WARNING: attribution failed, continuing without it: {e}")
                rpt.warn(f"XAI attribution failed (non-fatal, training/upload unaffected): {e}")
        else:
            rpt.kv("Status", "disabled (XAI_ENABLED not set)", indent=2)

        trainable_param_count = sum(p.numel() for p in model.parameters() if p.requires_grad)
        frozen_param_count    = sum(p.numel() for p in model.parameters() if not p.requires_grad)
        delta_param_count     = sum(v.numel() for v in delta.values())

        rpt.subheader("DELTA / SAFETY CLAMP")
        rpt.kv("Trainable params", f"{trainable_param_count:,}")
        rpt.kv("Frozen params", f"{frozen_param_count:,}")
        rpt.kv("Delta key count", len(delta))
        rpt.kv("Delta tensor count", len(delta))
        rpt.kv("Delta param count", f"{delta_param_count:,}")
        rpt.kv("Per-parameter clamp", f"±{max_param_change}")
        rpt.kv("Max global delta norm", max_global_norm)
        rpt.kv("Delta L2 norm before clamp", f"{delta_norm_before:.6f}")
        rpt.kv("Delta L2 norm after clamp", f"{delta_norm_after:.6f}")
        rpt.kv("Training time", f"{train_elapsed:.2f} sec")

        update_uri, receipt_uri = save_encrypted_delta(delta_safe, store, session_id, rm)
        print(f"[done] supervised training -> delta saved {update_uri}")
        rpt.ok(f"Local update saved: {update_uri}")
        return {"local_update_uri": update_uri, "update_receipt_uri": receipt_uri}
 
    elif mode == "rl":
        texts    = [r.get("transcript") or r.get("text") or "" for r in records]
        need_cli = not any(r.get("phq_score") is not None for r in records)
        if need_cli:
            # Fix E1 changed physician_feedback_cli()'s return signature to a
            # (values, provided_by_physician) tuple; this branch only runs
            # when NO record has ground truth (need_cli guard above), so
            # there is nothing to protect and the second element is unused
            # here - unlike the supervised branch below.
            corrected_phq, _provided_by_physician = physician_feedback_cli(preds, texts)
            for r, cp in zip(records, corrected_phq):
                r["phq_score"] = float(cp)
        else:
            for r in records:
                r["phq_score"] = float(r.get("phq_score") or r.get("phq") or 0.0)
        ds_rl = MultiModalDataset(records, tokenizer, max_len=MULTIMODAL_MAX_LEN)
        model = rl_update_reinforce(model, ds_rl, epochs=epochs, batch_size=1,
                                    lr=lr, device=device,
                                    supervised_lambda=rl_supervised_lambda)
        after  = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
        delta  = compute_filtered_delta(base_state, after, model)
        sparams = safety_params or {}
        delta_safe = apply_safety_to_delta(
            delta,
            max_param_change=sparams.get("max_param_change", DEFAULT_MAX_PARAM_CHANGE),
            max_global_norm=sparams.get("max_global_norm", DEFAULT_MAX_GLOBAL_DELTA_NORM),
        )
        update_uri, receipt_uri = save_encrypted_delta(delta_safe, store, session_id, rm)
        print(f"[done] RL training -> delta saved {update_uri}")
        return {"local_update_uri": update_uri, "update_receipt_uri": receipt_uri}
 
    else:
        raise ValueError("mode must be 'autonomous'|'supervised'|'rl'")

# ---------- CLI ----------
def main():
    parser = argparse.ArgumentParser(description="Trainer Agent (MentalBERT multimodal) with supervised + RL modes")
    parser.add_argument("--mode", choices=["autonomous", "supervised", "rl"], required=True)
    parser.add_argument("--input", required=True, help="parquet/json/csv path from LDA")
    parser.add_argument("--device", default=DEFAULT_DEVICE)
    parser.add_argument("--epochs", type=int, default=SUPERVISED_EPOCHS)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--lr", type=float, default=SUPERVISED_LR)
    parser.add_argument("--rl-supervised-lambda", type=float, default=0.0, help="mix supervised MSE into RL updates")
    parser.add_argument("--max-samples", type=int, default=None)
    parser.add_argument("--max-param-change", type=float, default=DEFAULT_MAX_PARAM_CHANGE)
    parser.add_argument("--max-global-delta-norm", type=float, default=DEFAULT_MAX_GLOBAL_DELTA_NORM)
    parser.add_argument("--session-id", required=True)
    args = parser.parse_args()

    safety = {"max_param_change": args.max_param_change, "max_global_norm": args.max_global_delta_norm}
    res = orchestrate(
        input_path=args.input,
        session_id=args.session_id,
        mode=args.mode,
        device=args.device,
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        rl_supervised_lambda=args.rl_supervised_lambda,
        max_samples=args.max_samples,
        safety_params=safety
    )
    print(json.dumps(res, indent=2, default=str))


if __name__ == "__main__":
    main()
