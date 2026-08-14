#!/usr/bin/env python3
"""
c2_common.py - Phase 20 / C-2 shared frozen constants and primitives.

Authoritative specification: PHASE_20_C2_DESIGN.md (FROZEN). Every constant here
is transcribed from that document. None may be changed.

SCOPE - design SS 21. C-2 is an AGGREGATION / MANIPULATION experiment. It computes
NO task metric: no ROC-AUC, accuracy, F1, MAE, RMSE or PR-AUC, and introduces no
held-out evaluation set (design SS 14). Nothing in this module computes one, and
nothing may be added that does.

WHAT C-2 MEASURES
    primary   : PRE-DP pairwise client-update divergence (L2 and cosine)
    secondary : post-DP divergence, per-coordinate trim counts, distance of the
                trimmed-mean aggregate from the ordinary mean and from the
                coordinate median
    validity  : W1 (non-IID) pre-DP divergence MUST EXCEED W0 (stratified) -
                directional only, NO numeric threshold (design SS 15)
"""

from __future__ import annotations

import hashlib
import json
import math
import os
import sys
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if REPO not in sys.path:
    sys.path.insert(0, REPO)

# FROZEN import - the single source of the RDP -> (eps, delta) conversion.
from dp_agent.dp_agent import _rdp_to_dp  # noqa: E402


# ==========================================================================
# Identity - design SS 1
# ==========================================================================
EXPERIMENT = "Phase 20 / C-2 - non-IID multi-client topology (roadmap Row 11)"
DESIGN_PATH = "PHASE_20_C2_DESIGN.md"
OUT_DIRNAME = "trainer_outputs/c2_multiclient"

SCOPE_CAVEAT = (
    "C-2 is an aggregation/manipulation experiment. It measures whether the A2 "
    "non-IID topology produced genuinely divergent client updates relative to "
    "the stratified/IID-like control, and how the frozen trimmed-mean aggregator "
    "behaved on those updates. It makes NO claim - explicit or implied - about "
    "task performance, model generalisation, clinical utility or predictive "
    "validity. No task metric is computed (design SS 14, SS 21)."
)

# ==========================================================================
# Arms and topology - design SS 3, SS 4, SS 5
# ==========================================================================
ARMS: Tuple[str, ...] = ("W0", "W1")
CONTROL_ARM = "W0"          # stratified / IID-like - NEVER "perfectly IID"
TREATMENT_ARM = "W1"        # non-IID A2
ARM_LABEL = {
    "W0": "stratified / IID-like control (frozen 5-fold test partitions)",
    "W1": "non-IID A2 treatment (phq_score DESC, participant_id ASC)",
}

N_CLIENTS = 5
CLIENT_SIZES: Tuple[int, ...] = (38, 38, 38, 37, 37)
N_PARTICIPANTS = 188
N_POSITIVE = 45
N_NEGATIVE = 143
BINARIZE_THRESHOLD = 10.0          # for REPORTING class counts only, never a target

# ==========================================================================
# Representation - design SS 6. PINNED.
# ==========================================================================
BERT_MODEL = "mental/mental-bert-base-uncased"
BERT_REVISION = "24809aa822c76639760d0d934742d1b42f89942f"
MAX_LENGTH = 128
TRUNCATION = True
PADDING = "max_length"
POOLING = "last_hidden_state[:, 0, :]"      # CLS, no .mean(dim=0)
EMBED_DIM = 768

# design SS 6 - pinned artifact hashes of the local HF snapshot.
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
    """Local HF snapshot for the PINNED revision (design SS 6)."""
    return os.path.join(
        os.path.expanduser("~"), ".cache", "huggingface", "hub",
        "models--mental--mental-bert-base-uncased", "snapshots", BERT_REVISION)


# design SS 6.1 - prohibited representation sources, asserted absent from use.
PROHIBITED_REPRESENTATION_SOURCES: Tuple[str, ...] = (
    "secure_store/sess-DAICWOZ/encrypted/text_embeddings.parquet.enc",
    "format_daic_to_lda.py",
    "create_dp_comparison.py",
)

# ==========================================================================
# Data and shared initialisation - design SS 7, SS 8
# ==========================================================================
DATA_PARQUET = "daic_records.parquet"
FOLD_MANIFEST = "trainer_outputs/baseline_cv/fold_manifest.json"
PROBE_PATH = "trainer_outputs/local_probe_base.pt"

PROBE_TENSORS: Dict[str, Tuple[int, ...]] = {
    "fc1.weight": (384, 768),
    "fc1.bias": (384,),
    "fc2.weight": (1, 384),
    "fc2.bias": (1,),
}
PROBE_N_PARAMS = 295_681
PROBE_N_TENSORS = 4

# ==========================================================================
# Local training recipe - design SS 9
# ==========================================================================
LOSS = "MSELoss"
OPTIMIZER = "Adam"
LR = 1e-3
EPOCHS = 1
BATCH = "full"
GRAD_CLIP = 1.0
TARGET = "phq_score"        # CONTINUOUS regression target (design SS 7)

# ==========================================================================
# Differential privacy - design SS 11. Inherited, never varied.
# ==========================================================================
MECHANISM = "gaussian"
CLIP_NORM = 1.0
NOISE_MULTIPLIER = 1.0
SIGMA_EFF = NOISE_MULTIPLIER / CLIP_NORM
DELTA_DP = 1e-05
EPS_PER_UPDATE = 5.302585092994046
COMPOSITION = "RDP single-composition, T = 1 (output perturbation)"

# ==========================================================================
# Aggregation - design SS 12
# ==========================================================================
AGG_MODE = "trimmed_mean"
TRIM_RATIO = 0.1
AGGREGATOR_PATH = "server/aggregator_agent/aggregator.py"


def trim_window(n: int, trim_ratio: float = TRIM_RATIO) -> Tuple[int, int, int]:
    """(lower, upper, kept) exactly as AggregatorAgent._aggregate_tensor computes.

    Source: server/aggregator_agent/aggregator.py - lower = max(1, int(r*N)),
    upper = N - lower, mean(sorted[lower:upper]); raises if lower >= upper.
    At n = 5, trim_ratio 0.1 -> (1, 4, 3): three of five retained (design SS 12).
    """
    lower = max(1, int(trim_ratio * n))
    upper = n - lower
    return lower, upper, upper - lower


# ==========================================================================
# Repeats and seed hierarchy - design SS 13
# ==========================================================================
N_REPEATS = 5
REPEATS: Tuple[int, ...] = (1, 2, 3, 4, 5)
BASE_SEED = 1000


def repeat_seed(repeat: int) -> int:
    """design SS 13: repeat_seed = 1000 + repeat."""
    return BASE_SEED + repeat


def client_seed(repeat: int, client_index: int) -> int:
    """design SS 13: client_seed = repeat_seed * 100 + client_index."""
    return repeat_seed(repeat) * 100 + client_index


def dp_seed(repeat: int) -> int:
    """design SS 13: DP seed = 1000 + repeat."""
    return BASE_SEED + repeat


# ==========================================================================
# Frozen input SHAs - design SS 16, SS 19
# ==========================================================================
FROZEN_SHA: Dict[str, str] = {
    "PHASE_20_C2_DESIGN.md": None,          # filled at first freeze; see design_sha()
    "daic_records.parquet":
        "9a241851760727779fcaa4d50041b8bdc4406fe6b66ddbbd7105f4ce4fc55f00",
    "trainer_outputs/baseline_cv/fold_manifest.json":
        "b9a7a91f1bdd43c78d3cd1ee0c36e4ce3fb8be4b0971dcff3ff62ee5af8fca2f",
    "trainer_outputs/baseline_cv/baseline_cv_summary.json":
        "f065f5e1ff7f8e5ed4d122a8031c9cc12425802e756697d5512a915674871aac",
    "trainer_outputs/local_probe_base.pt":
        "d21f95ab4169ba8bf270e2ae900d2d205ddb4c3aba3c04ebae4d0ec8d9ba0993",
    "trainer_mentalbert_daic.py":
        "65b1902e2ba8dd996c6960db1a6595d84fd48500f4d8707cb79688093d7a230b",
    "dp_agent/dp_agent.py":
        "758642fff57695cb970af88789c3b6a17c77f2b01d16303ff96230fce5798732",
}
# The design SHA is recorded at run time rather than hardcoded, because this file
# is written alongside the design. It is pinned into every artifact.


# ==========================================================================
# Utilities
# ==========================================================================
def abort(message: str) -> "NoReturn":  # noqa: F821
    """Fail closed. C-2 has no partial-success mode (design SS 19)."""
    print(f"\n[ABORT] {message}", file=sys.stderr)
    raise SystemExit(1)


def repo_key(path: str) -> str:
    """Canonical repo-relative forward-slash key (platform-neutral)."""
    return path.replace("\\", "/")


def frozen_sha_for(path: str) -> Optional[str]:
    """Registered SHA, or None if unpinned. Fail-closed: never raises KeyError."""
    return FROZEN_SHA.get(repo_key(path))


def sha256_file(path: str) -> Optional[str]:
    if not os.path.isfile(path):
        return None
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for b in iter(lambda: fh.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


def sha256_bytes(b: bytes) -> str:
    return hashlib.sha256(b).hexdigest()


def design_sha() -> Optional[str]:
    return sha256_file(os.path.join(REPO, DESIGN_PATH))


def vector_digest(v: np.ndarray) -> str:
    """SHA-256 over the raw float64 buffer - enables bitwise recomputation."""
    return sha256_bytes(np.ascontiguousarray(v, dtype=np.float64).tobytes())


# ==========================================================================
# Partitions - design SS 4, SS 5
# ==========================================================================
def control_partition(manifest: Dict[str, Any]) -> List[List[int]]:
    """Arm W0 - the frozen 5-fold TEST partitions (design SS 4).

    STRATIFIED / IID-LIKE. Never describe this as perfectly IID: it is stratified
    on PHQ_binary only, and its PHQ means span 5.5405-7.7895.
    """
    folds = sorted(manifest["folds"], key=lambda f: int(f["fold"]))
    return [[int(i) for i in f["test_ids"]] for f in folds]


def a2_partition(records: Sequence[Tuple[int, float]]) -> List[List[int]]:
    """Arm W1 - A2 (design SS 5). Deterministic, no free parameter.

        sort by (phq_score DESC, participant_id ASC)
        fill clients sequentially with CLIENT_SIZES

    Ties in phq_score are broken by participant_id ascending, so the partition is
    fully determined by the frozen data.
    """
    ordered = sorted(records, key=lambda t: (-t[1], t[0]))
    out: List[List[int]] = []
    k = 0
    for size in CLIENT_SIZES:
        out.append([pid for pid, _ in ordered[k:k + size]])
        k += size
    return out


def partition_stats(client_ids: Sequence[int], phq: Dict[int, float]) -> Dict[str, Any]:
    v = [phq[i] for i in client_ids]
    pos = sum(1 for x in v if x > BINARIZE_THRESHOLD)
    return {
        "n": len(client_ids), "pos": pos, "neg": len(client_ids) - pos,
        "pos_rate": pos / len(client_ids) if client_ids else float("nan"),
        "phq_mean": float(np.mean(v)), "phq_median": float(np.median(v)),
        "phq_min": float(np.min(v)), "phq_max": float(np.max(v)),
        "phq_sd": float(np.std(v)),          # population sd, reporting only
    }


def check_partition(parts: Sequence[Sequence[int]], expect_ids: set) -> Dict[str, Any]:
    """design SS 16.1/16.2 - disjoint, exhaustive, correct sizes."""
    flat: List[int] = []
    for p in parts:
        flat.extend(p)
    sizes = tuple(len(p) for p in parts)
    overlaps = []
    for i in range(len(parts)):
        for j in range(i + 1, len(parts)):
            inter = set(parts[i]) & set(parts[j])
            if inter:
                overlaps.append((i + 1, j + 1, sorted(inter)))
    return {
        "sizes": sizes,
        "sizes_ok": sizes == CLIENT_SIZES,
        "n_total": len(flat),
        "n_unique": len(set(flat)),
        "duplicates": len(flat) - len(set(flat)),
        "coverage_ok": set(flat) == expect_ids,
        "disjoint": not overlaps,
        "overlaps": overlaps,
    }


# ==========================================================================
# Divergence - design SS 14. Both measures reported; no threshold.
# ==========================================================================
def l2_distance(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.linalg.norm(np.asarray(a, np.float64) - np.asarray(b, np.float64)))


def cosine_distance(a: np.ndarray, b: np.ndarray) -> float:
    """1 - cosine similarity. Repository precedent: create_dp_comparison.py:293,311."""
    x = np.asarray(a, np.float64)
    y = np.asarray(b, np.float64)
    nx, ny = np.linalg.norm(x), np.linalg.norm(y)
    if nx == 0.0 or ny == 0.0:
        return float("nan")
    return float(1.0 - float(np.dot(x, y) / (nx * ny)))


def pairwise_divergence(vectors: Sequence[np.ndarray]) -> Dict[str, Any]:
    """All C(n,2) pairs. Returns both L2 and cosine, plus descriptive summaries."""
    n = len(vectors)
    pairs, l2s, coss = [], [], []
    for i in range(n):
        for j in range(i + 1, n):
            d2 = l2_distance(vectors[i], vectors[j])
            dc = cosine_distance(vectors[i], vectors[j])
            pairs.append(f"{i + 1}-{j + 1}")
            l2s.append(d2)
            coss.append(dc)
    return {
        "pairs": pairs, "n_pairs": len(pairs),
        "l2": l2s, "cosine": coss,
        "l2_mean": float(np.mean(l2s)) if l2s else float("nan"),
        "l2_min": float(np.min(l2s)) if l2s else float("nan"),
        "l2_max": float(np.max(l2s)) if l2s else float("nan"),
        "cosine_mean": float(np.mean(coss)) if coss else float("nan"),
        "cosine_min": float(np.min(coss)) if coss else float("nan"),
        "cosine_max": float(np.max(coss)) if coss else float("nan"),
    }


# ==========================================================================
# DP - design SS 11. Applied to the k-dimensional flattened delta.
# ==========================================================================
def epsilon() -> float:
    """eps via the FROZEN dp_agent conversion. Never reimplemented."""
    return _rdp_to_dp(noise_multiplier=NOISE_MULTIPLIER,
                      clip_norm=CLIP_NORM, delta=DELTA_DP)
