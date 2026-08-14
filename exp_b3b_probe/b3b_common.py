#!/usr/bin/env python3
"""
b3b_common.py - Phase 21 / B-3B frozen constants and shared helpers.

Every value here transcribes PHASE_21_B3B_DESIGN.md (FROZEN). Nothing may be
tuned. If a value is not in the design, it does not belong here.

B-3B is a SECOND B-3 attempt. H0 is a legitimate result (design SS 12, SS 19).
"""

from __future__ import annotations

import hashlib
import json
import os
import sys
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

EXPERIMENT = "Phase 21 / B-3B - compact task-trained probe (roadmap Row 9, 2nd attempt)"
DESIGN_PATH = "PHASE_21_B3B_DESIGN.md"
OUT_DIRNAME = "trainer_outputs/b3b_probe"

SCOPE_CAVEAT = (
    "B-3B is a compact-representation experiment and a SECOND B-3 attempt. It is "
    "not expected to succeed; H0 is a legitimate result. It makes no claim about "
    "clinical utility, and a successful result would NOT unblock Roadmap Row 10 "
    "on its own: Row 10 requires B-2 AND B-3, and B-2 (Exp 8) returned H0."
)

# ==========================================================================
# Representation - design SS 3 (inherited)
# ==========================================================================
BERT_MODEL = "mental/mental-bert-base-uncased"
BERT_REVISION = "24809aa822c76639760d0d934742d1b42f89942f"
MAX_LENGTH = 128
TRUNCATION = True
PADDING = "max_length"
POOLING = "last_hidden_state[:, 0, :]"
EMBED_DIM = 768
N_PARTICIPANTS = 188
N_POSITIVE = 45
N_NEGATIVE = 143
BINARIZE_THRESHOLD = 10.0          # label = phq_score > 10.0

# Reused C-2 embedding artifact (design SS 3). Verified, never modified.
EMBEDDINGS_PATH = "trainer_outputs/c2_multiclient/c2_embeddings.npz"
EMBEDDINGS_SHA = "70df4e49b4fea83c2464eba3e41e5b8ffa856fff4c70dc7b7b8d054cf488ed3c"

# ==========================================================================
# Architecture - design SS 4 (repository-supported, FROZEN, no ladder)
# ==========================================================================
IN_DIM = 768
HIDDEN = 256
NUM_CLASSES = 2
HEAD_N_PARAMS = 197_892
HEAD_N_TENSORS = 8
HEAD_TENSORS: Dict[str, Tuple[int, ...]] = {
    "fc1.weight": (256, 768), "fc1.bias": (256,),
    "classifier.weight": (2, 256), "classifier.bias": (2,),
    "phq_mu.weight": (1, 256), "phq_mu.bias": (1,),
    "phq_logsigma.weight": (1, 256), "phq_logsigma.bias": (1,),
}

# ==========================================================================
# Objective + recipe - design SS 5, SS 6 (repository-supported, FROZEN)
# ==========================================================================
LOSS = "CrossEntropyLoss(logits, label) + 0.5 * MSELoss(mu, phq)"
LAMBDA_REG = 0.5
OPTIMIZER = "transformers.AdamW"
LR = 1e-3                    # design SS 6 - head-on-frozen-features precedent
EPOCHS = 3
BATCH_SIZE = 8
SHUFFLE = True
DROPOUT = 0.2
GRAD_CLIP = 1.0
RANKING_SCORE = "softmax(logits)[:, 1]"   # positive-class probability

# ==========================================================================
# Arms - design SS 9
# ==========================================================================
ARMS: Tuple[str, ...] = ("N", "D")
CONTROL_ARM = "N"            # no DP
TREATMENT_ARM = "D"          # DP applied
ARM_LABEL = {
    "N": "no-DP control (identical in every respect except DP)",
    "D": "DP arm (clip C=1.0 + Gaussian sigma_eff=1.0)",
}

# ==========================================================================
# DP - design SS 10 (inherited, FROZEN)
# ==========================================================================
MECHANISM = "gaussian"
CLIP_NORM = 1.0
NOISE_MULTIPLIER = 1.0
SIGMA_EFF = NOISE_MULTIPLIER / CLIP_NORM
DELTA_DP = 1e-05
EPS_PER_UPDATE = 5.302585092994046
COMPOSITION = "RDP single-composition, T = 1 (output perturbation)"

# ==========================================================================
# Evaluation - design SS 11 (inherited from Exp 9)
# ==========================================================================
FOLD_MANIFEST = "trainer_outputs/baseline_cv/fold_manifest.json"
N_FOLDS = 5
FOLDS: Tuple[int, ...] = (1, 2, 3, 4, 5)
N_REPEATS = 5
REPEATS: Tuple[int, ...] = (1, 2, 3, 4, 5)
BASE_SEED = 1000
DF = 4
T_CRIT = 2.7764451051977987
METRIC = "ROC-AUC (positive-class probability)"


def repeat_seed(repeat: int) -> int:
    """design SS 11: seed = 1000 + repeat."""
    return BASE_SEED + repeat


def fold_seed(repeat: int, fold: int) -> int:
    """design SS 11: torch.manual_seed(seed * 100 + fold) before model construction."""
    return repeat_seed(repeat) * 100 + fold


# ==========================================================================
# Acceptance A / B / C - design SS 12 (FROZEN, may never move)
# ==========================================================================
G2_PAYLOAD_BYTES = 1_579_963          # A
G3_NSR = 544.341809                   # B
ROC_AUC_CI = (0.5755177227457472, 0.6911434704671701)   # C part 1
ROC_AUC_MDE = 0.05781287386071144                        # C part 2
BASELINE_ROC_AUC = 0.6333305966064586
TRANSPORT_EXPANSION = 1_579_963 / 1_185_335   # Exp 9 SS 17 measured factor

# ==========================================================================
# Frozen input SHAs - design SS 3, SS 8, SS 11
# ==========================================================================
FROZEN_SHA: Dict[str, Optional[str]] = {
    "PHASE_21_B3B_DESIGN.md": None,     # pinned into artifacts at run time
    "daic_records.parquet":
        "9a241851760727779fcaa4d50041b8bdc4406fe6b66ddbbd7105f4ce4fc55f00",
    "trainer_outputs/baseline_cv/fold_manifest.json":
        "b9a7a91f1bdd43c78d3cd1ee0c36e4ce3fb8be4b0971dcff3ff62ee5af8fca2f",
    "trainer_outputs/baseline_cv/baseline_cv_summary.json":
        "f065f5e1ff7f8e5ed4d122a8031c9cc12425802e756697d5512a915674871aac",
    "trainer_mentalbert_daic.py":
        "65b1902e2ba8dd996c6960db1a6595d84fd48500f4d8707cb79688093d7a230b",
    "dp_agent/dp_agent.py":
        "758642fff57695cb970af88789c3b6a17c77f2b01d16303ff96230fce5798732",
    "PHASE_18_EXP9_DESIGN.md":
        "cf361986e4f058259e21fefd548b335ad9fb91758696d57fdf6c762254bf288a",
    EMBEDDINGS_PATH: EMBEDDINGS_SHA,
}


def abort(message: str) -> "NoReturn":  # noqa: F821
    """Fail closed. B-3B has no partial-success mode."""
    print(f"\n[ABORT] {message}", file=sys.stderr)
    raise SystemExit(1)


def repo_key(path: str) -> str:
    return path.replace("\\", "/")


def sha256_file(path: str) -> Optional[str]:
    if not os.path.isfile(path):
        return None
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def design_sha() -> Optional[str]:
    return sha256_file(os.path.join(REPO, DESIGN_PATH))


def binarize(phq: float) -> int:
    """design SS 5: label = phq_score > 10.0."""
    return int(float(phq) > BINARIZE_THRESHOLD)


def load_folds(manifest: Dict[str, Any]) -> List[Dict[str, Any]]:
    """Frozen participant-level stratified 5-fold, design SS 11."""
    folds = manifest["folds"]
    if len(folds) != N_FOLDS:
        abort(f"fold manifest holds {len(folds)} folds, expected {N_FOLDS}")
    out = []
    for f in folds:
        tr, te = [int(x) for x in f["train_ids"]], [int(x) for x in f["test_ids"]]
        if set(tr) & set(te):
            abort(f"fold {f['fold']}: train/test overlap")
        out.append({"fold": int(f["fold"]), "train_ids": tr, "test_ids": te})
    return out


def epsilon() -> float:
    """eps via the FROZEN dp_agent conversion. Never reimplemented."""
    sys.path.insert(0, os.path.join(REPO, "dp_agent"))
    from dp_agent import _rdp_to_dp  # type: ignore
    return _rdp_to_dp(noise_multiplier=NOISE_MULTIPLIER,
                      clip_norm=CLIP_NORM, delta=DELTA_DP)


def roc_auc(scores: Sequence[float], labels: Sequence[int]) -> float:
    """ROC-AUC from the positive-class probability (design SS 5, SS 11)."""
    from sklearn.metrics import roc_auc_score
    y = np.asarray(labels)
    if len(set(y.tolist())) < 2:
        return float("nan")       # degenerate fold - reported, never imputed
    return float(roc_auc_score(y, np.asarray(scores, dtype=np.float64)))


def mean_sd(v: Sequence[float]) -> Tuple[float, float]:
    a = np.asarray(v, dtype=np.float64)
    if a.size == 0:
        return float("nan"), float("nan")
    return float(np.mean(a)), (float(np.std(a, ddof=1)) if a.size > 1 else 0.0)


def in_ci(x: float) -> bool:
    return bool(ROC_AUC_CI[0] <= x <= ROC_AUC_CI[1])
