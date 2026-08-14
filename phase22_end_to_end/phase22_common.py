#!/usr/bin/env python3
"""
phase22_common.py - Phase 22 frozen constants and shared helpers.

Every value transcribes PHASE_22_DESIGN.md (FROZEN,
SHA-256 3c18108309c17cbff332204aa761411d6600fe584488391741dfcd2a9208036f).
Nothing here may be tuned.

PHASE 22 IS AN ENGINEERING DEMONSTRATION - NOT roadmap Row 10 / C-1.
B-2 = H0, B-3 = H0 twice, Row 10 remains BLOCKED (design SS 1, SS 23).
"""

from __future__ import annotations

import hashlib
import os
import sys
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

EXPERIMENT = "Phase 22 - end-to-end multimodal federated pipeline demonstration"
DESIGN_PATH = "PHASE_22_DESIGN.md"
DESIGN_SHA = "3c18108309c17cbff332204aa761411d6600fe584488391741dfcd2a9208036f"
OUT_DIRNAME = "trainer_outputs/phase22_end_to_end"

SCOPE_CAVEAT = (
    "Phase 22 is an ENGINEERING DEMONSTRATION. It does NOT attempt, satisfy or "
    "partially satisfy Roadmap Row 10 / C-1. B-2 (Exp 8) returned H0 and B-3 "
    "returned H0 twice (Exp 9, B-3B); Row 10 remains BLOCKED. Engineering "
    "success does NOT imply task success (design SS 2)."
)

# ==========================================================================
# Data - design SS 4
# ==========================================================================
DATA_PARQUET = "dataset_build/daic_records_multimodal.parquet"
BASE_PARQUET = "daic_records.parquet"
FOLD_MANIFEST = "trainer_outputs/baseline_cv/fold_manifest.json"
N_PARTICIPANTS = 188
N_POSITIVE = 45
N_NEGATIVE = 143
BINARIZE_THRESHOLD = 10.0

# ==========================================================================
# Representation / model - design SS 5
# ==========================================================================
BERT_MODEL = "mental/mental-bert-base-uncased"
BERT_REVISION = "24809aa822c76639760d0d934742d1b42f89942f"
MAX_LENGTH = 128
TEXT_DIM = 768
AUDIO_DIM = 154
VISION_DIM = 84
FUSION_DIM = TEXT_DIM + 128 + 128          # 1024
MODEL_N_PARAMS = 109_763_494
MODEL_N_KEYS = 215
DROPOUT = 0.2

BERT_ARTIFACT_SHA: Dict[str, str] = {
    "pytorch_model.bin":
        "c4f90fa5f0b991c48eb99afe41c8883dccb2b7e51012b33d2635925b9cde8764",
    "tokenizer.json":
        "5fd1c882abbd30517dced455a2c9768945ec726b96727927e4959348d9de550b",
    "vocab.txt":
        "07eced375cec144d27c900241f3e339478dec958f92fddbc551f295c992038a3",
    "config.json":
        "79ee28a4e33b49f535209c8aaa5eaa7550344345bb3fecb13f179c686749d0d9",
    "tokenizer_config.json":
        "9a8ed9b01c8a56b555dcdc31cd526ba9e488cfc16ee5a101b3ce64af81d34f3d",
    "special_tokens_map.json":
        "303df45a03609e4ead04bc3dc1536d0ab19b5358db685b6f3da123d05ec200e3",
}


def hf_snapshot_dir() -> str:
    return os.path.join(
        os.path.expanduser("~"), ".cache", "huggingface", "hub",
        "models--mental--mental-bert-base-uncased", "snapshots", BERT_REVISION)


# ==========================================================================
# Local training recipe - design SS 6 (FROZEN)
# ==========================================================================
OPTIMIZER = "transformers.AdamW"
LR = 2e-5
EPOCHS = 3
BATCH_SIZE = 8
SHUFFLE = True
GRAD_CLIP = 1.0
LOSS = "CrossEntropyLoss(logits, label) + 0.5 * MSELoss(mu, phq)"
LAMBDA_REG = 0.5

# ==========================================================================
# Federated topology - design SS 7
# ==========================================================================
N_CLIENTS = 5
N_ROUNDS = 1
PARTITION_RULE = "A2 non-IID: sort by (phq_score DESC, participant_id ASC), fill sequentially"

# ==========================================================================
# Arms - design SS 8
# ==========================================================================
ARMS: Tuple[str, ...] = ("N", "D")
CONTROL_ARM = "N"
TREATMENT_ARM = "D"
ARM_LABEL = {
    "N": "no-DP control (identical in every respect except DP)",
    "D": "DP arm (clip C=1.0 + Gaussian sigma_eff=1.0)",
}

# ==========================================================================
# DP - design SS 9 (FROZEN, inherited)
# ==========================================================================
MECHANISM = "gaussian"
CLIP_NORM = 1.0
NOISE_MULTIPLIER = 1.0
SIGMA_EFF = NOISE_MULTIPLIER / CLIP_NORM
DELTA_DP = 1e-05
EPS_PER_UPDATE = 5.302585092994046
COMPOSITION = "RDP single-composition, T = 1 (output perturbation)"
ROUND_EPS_BASIS = (
    "Clients hold DISJOINT participant sets, so parallel composition applies: "
    "round-level epsilon = max over clients = 5.302585092994046, NOT the sum "
    "(design SS 9)."
)

# ==========================================================================
# Aggregation - design SS 10 (FROZEN)
# ==========================================================================
AGG_MODE = "trimmed_mean"
TRIM_RATIO = 0.1
AGGREGATOR_PATH = "server/aggregator_agent/aggregator.py"


def trim_window(n: int, trim_ratio: float = TRIM_RATIO) -> Tuple[int, int, int]:
    lower = max(1, int(trim_ratio * n))
    upper = n - lower
    return lower, upper, upper - lower


# ==========================================================================
# Evaluation - design SS 11
# ==========================================================================
N_FOLDS = 5
FOLDS: Tuple[int, ...] = (1, 2, 3, 4, 5)
N_REPEATS = 1
BASELINE_ROC_AUC = 0.6333305966064586
BASELINE_CI = (0.5755177227457472, 0.6911434704671701)
PRIMARY_METRIC = "ROC-AUC (positive-class probability)"

# ==========================================================================
# Seeds - design SS 14
# ==========================================================================
BASE_SEED = 1000
REPEAT = 1


def fold_seed(fold: int) -> int:
    """design SS 14: (1000 + 1) * 100 + fold -> 100101..100105."""
    return (BASE_SEED + REPEAT) * 100 + fold


def client_seed(fold: int, client_index: int) -> int:
    """design SS 14: fold_seed * 10 + client_index."""
    return fold_seed(fold) * 10 + client_index


def dp_seed(fold: int, client_index: int) -> int:
    """design SS 14: dp_seed = client_seed."""
    return client_seed(fold, client_index)


# ==========================================================================
# Frozen input SHAs - design SS 3
# ==========================================================================
FROZEN_SHA: Dict[str, Optional[str]] = {
    "PHASE_22_DESIGN.md": DESIGN_SHA,
    "dataset_build/daic_records_multimodal.parquet":
        "1ac9f53e6102ec0dbaab84dcfdfa3f2e70f2b4a1867a841ea2b7e3ba24c67a95",
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
    "server/aggregator_agent/aggregator.py":
        "59b4d4838cecfaa49f8341320c4d1fdb55fa3b2b6ca1717e29f273f820535d86",
}


# ==========================================================================
# Utilities
# ==========================================================================
def abort(message: str) -> "NoReturn":  # noqa: F821
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
    return int(float(phq) > BINARIZE_THRESHOLD)


def load_folds(manifest: Dict[str, Any]) -> List[Dict[str, Any]]:
    folds = manifest["folds"]
    if len(folds) != N_FOLDS:
        abort(f"fold manifest holds {len(folds)} folds, expected {N_FOLDS}")
    out = []
    for f in folds:
        tr = [int(x) for x in f["train_ids"]]
        te = [int(x) for x in f["test_ids"]]
        if set(tr) & set(te):
            abort(f"fold {f['fold']}: train/test overlap - LEAKAGE")
        out.append({"fold": int(f["fold"]), "train_ids": tr, "test_ids": te})
    return out


# ==========================================================================
# Client partition - design SS 7.1
# ==========================================================================
def client_sizes(n: int, n_clients: int = N_CLIENTS) -> List[int]:
    """Deterministic even split (design SS 7.1). No free parameter.

        base = n // n_clients ; rem = n % n_clients
        sizes = [base + 1] * rem + [base] * (n_clients - rem)

    n = 150 -> (30, 30, 30, 30, 30)   n = 151 -> (31, 30, 30, 30, 30)
    """
    base, rem = divmod(n, n_clients)
    return [base + 1] * rem + [base] * (n_clients - rem)


def a2_partition_training(train_ids: Sequence[int],
                          phq: Dict[int, float]) -> List[List[int]]:
    """A2 non-IID rule applied to the fold's TRAINING participants only.

    design SS 7.1: the C-2 helper hardcodes CLIENT_SIZES summing to 188 (the whole
    population) and therefore cannot be called here without leaking the held-out
    fold. The RULE is reused verbatim; only the per-fold sizes are derived.

        sort by (phq_score DESC, participant_id ASC), fill sequentially
    """
    ordered = sorted(train_ids, key=lambda pid: (-phq[pid], pid))
    sizes = client_sizes(len(ordered))
    out, k = [], 0
    for s in sizes:
        out.append(ordered[k:k + s])
        k += s
    if k != len(ordered):
        abort(f"partition consumed {k} of {len(ordered)}")
    return out


def check_partition(parts: Sequence[Sequence[int]], expect_ids: set,
                    test_ids: set) -> Dict[str, Any]:
    """design SS 7.2 - disjoint, exhaustive over TRAIN, and no test leakage."""
    flat: List[int] = []
    for p in parts:
        flat.extend(p)
    overlaps = []
    for i in range(len(parts)):
        for j in range(i + 1, len(parts)):
            inter = set(parts[i]) & set(parts[j])
            if inter:
                overlaps.append((i + 1, j + 1, sorted(inter)))
    leaked = sorted(set(flat) & test_ids)
    return {
        "sizes": [len(p) for p in parts],
        "n_total": len(flat), "n_unique": len(set(flat)),
        "duplicates": len(flat) - len(set(flat)),
        "coverage_ok": set(flat) == expect_ids,
        "disjoint": not overlaps, "overlaps": overlaps,
        "test_leakage": leaked, "no_leakage": not leaked,
    }


def partition_stats(client_ids: Sequence[int],
                    phq: Dict[int, float]) -> Dict[str, Any]:
    v = [phq[i] for i in client_ids]
    pos = sum(1 for x in v if binarize(x))
    return {
        "n": len(client_ids), "pos": pos, "neg": len(client_ids) - pos,
        "pos_rate": pos / len(client_ids) if client_ids else float("nan"),
        "phq_mean": float(np.mean(v)), "phq_min": float(np.min(v)),
        "phq_max": float(np.max(v)), "phq_sd": float(np.std(v)),
    }


# ==========================================================================
# DP accounting - frozen dp_agent, never reimplemented
# ==========================================================================
def epsilon() -> float:
    sys.path.insert(0, os.path.join(REPO, "dp_agent"))
    from dp_agent import _rdp_to_dp  # type: ignore
    return _rdp_to_dp(noise_multiplier=NOISE_MULTIPLIER,
                      clip_norm=CLIP_NORM, delta=DELTA_DP)


# ==========================================================================
# Metrics - design SS 11
# ==========================================================================
def task_metrics(scores: Sequence[float], labels: Sequence[int]) -> Dict[str, float]:
    """ROC-AUC primary; accuracy NEVER reported alone (design SS 11)."""
    from sklearn.metrics import (accuracy_score, average_precision_score,
                                 f1_score, precision_score, recall_score,
                                 roc_auc_score)
    y = np.asarray(labels)
    s = np.asarray(scores, dtype=np.float64)
    pred = (s >= 0.5).astype(int)
    both = len(set(y.tolist())) >= 2
    return {
        "roc_auc": float(roc_auc_score(y, s)) if both else float("nan"),
        "pr_auc": float(average_precision_score(y, s)) if both else float("nan"),
        "accuracy": float(accuracy_score(y, pred)),
        "precision": float(precision_score(y, pred, zero_division=0)),
        "recall": float(recall_score(y, pred, zero_division=0)),
        "f1": float(f1_score(y, pred, zero_division=0)),
        "n": int(len(y)), "n_pos": int(y.sum()),
    }


def mean_sd(v: Sequence[float]) -> Tuple[float, float]:
    a = np.asarray([x for x in v if np.isfinite(x)], dtype=np.float64)
    if a.size == 0:
        return float("nan"), float("nan")
    return float(np.mean(a)), (float(np.std(a, ddof=1)) if a.size > 1 else 0.0)


def in_baseline_ci(x: float) -> bool:
    return bool(BASELINE_CI[0] <= x <= BASELINE_CI[1])
