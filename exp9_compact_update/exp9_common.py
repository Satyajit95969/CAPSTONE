#!/usr/bin/env python3
"""
exp9_common.py - Phase 18 / Experiment 9 shared frozen constants and primitives.

Authoritative specification: PHASE_18_EXP9_DESIGN.md (FROZEN),
SHA-256 cf361986e4f058259e21fefd548b335ad9fb91758696d57fdf6c762254bf288a.
Every constant here is transcribed from that document. None may be changed.

WHY THIS MODULE EXISTS
----------------------
The priority ordering O, the mask family S_k, the k ladder and every threshold
are used by the Terminal runner, the Colab task runner, the aggregator AND the
verifier. Duplicating them four times would create four places for them to
drift, and drift in O or k is precisely the one-factor violation design SS 29
declares a HARD ABORT. This module is the single definition; it is a support
module, not an additional experimental component.

THE MECHANISM IS NOT TOP-K (design SS 3, SS 6, SS 32.8)
-------------------------------------------------------
The selection implemented here is PUBLIC ARCHITECTURE-PRIORITY COORDINATE
SELECTION. It is a deterministic function of the ARCHITECTURE and k alone:
state-dict key names, tensor shapes, frozen key order, and row-major flattened
index order. It never reads, ranks, sorts, thresholds or otherwise inspects a
single delta VALUE. Magnitude, energy, gradients, Fisher information, validation
or test performance, and every other private or data-derived statistic are
forbidden inputs (design SS 6.3) - including as a post-hoc *validation* of O,
which would reintroduce the SS 3 privacy defect at the design level.

Consequently S_k may be published before any data is touched and leaks nothing,
so the index set need not be transmitted and eps is unchanged (design SS 10).
"""

from __future__ import annotations

import hashlib
import io
import math
import os
import re
import sys
from typing import Any, Dict, List, Optional, Sequence, Tuple

import torch

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if REPO not in sys.path:
    sys.path.insert(0, REPO)

# FROZEN import - the single source of the RDP -> (eps, delta) conversion.
# dp_agent/dp_agent.py is frozen (design SS 5, SS 31): imported, never edited,
# never reimplemented.
from dp_agent.dp_agent import _rdp_to_dp  # noqa: E402


# ==========================================================================
# Identity - design SS 1
# ==========================================================================
EXPERIMENT = "Phase 18 / Exp 9 - compact update representation"
DESIGN_PATH = "PHASE_18_EXP9_DESIGN.md"
DESIGN_SHA = "cf361986e4f058259e21fefd548b335ad9fb91758696d57fdf6c762254bf288a"
MECHANISM_NAME = "public architecture-priority coordinate selection"
OUT_DIRNAME = os.path.join("trainer_outputs", "exp9_compact_update")

# ==========================================================================
# Target objects - design SS 21.1. Two objects, two halves, deliberately disjoint.
# ==========================================================================
DELTA_PATH = os.path.join("trainer_outputs", "mentalbert_delta.pt")
EXPECTED_D = 109_680_132
EXPECTED_N_KEYS = 207

# ==========================================================================
# Frozen k ladder - design SS 8. K0 is the uncompressed control.
# ==========================================================================
ARMS: Tuple[str, ...] = ("K0", "K1", "K2", "K3")
K_VALUES: Dict[str, int] = {
    "K0": 109_680_132,
    "K1": 1_096_801,
    "K2": 295_681,
    "K3": 10_968,
}
CONTROL_ARM = "K0"
ARM_LABEL = {
    "K0": "k = d, no reduction (control)",
    "K1": "k = 1.0000 % of d",
    "K2": "k = 0.2696 % of d (probe-dimension anchor)",
    "K3": "k = 0.0100 % of d",
}

# ==========================================================================
# Frozen privacy parameters - design SS 5, SS 12
# ==========================================================================
CLIP_NORM = 1.0
NOISE_MULTIPLIER = 1.0
SIGMA_EFF = NOISE_MULTIPLIER / CLIP_NORM          # = 1.0
DELTA_DP = 1e-05
EPS_PER_UPDATE_MAX = 5.302585092994046
EPS_CUMULATIVE_MAX = 15.9078
N_UPDATES_PER_ROUND = 3
EPS_MAX_CEILING = 100          # ceiling only, NOT the acceptance target (SS 5)
COMPOSITION = "RDP single-composition, T = 1 (output perturbation)"

# ==========================================================================
# DP/SNR half - design SS 14, SS 26
# ==========================================================================
SNR_DEFINITION = "l2_norm_after / clip_norm"
N_REPETITIONS = 5
BASE_SEED_DP = 1000
SEEDS: Tuple[int, ...] = tuple(BASE_SEED_DP + i for i in range(1, N_REPETITIONS + 1))
# -> (1001, 1002, 1003, 1004, 1005); identical seeds across arms (design SS 26)

# ==========================================================================
# Acceptance thresholds - design SS 15, SS 17, SS 22, SS 23, SS 24
# FROZEN. Never moved, including if K2 lands marginally (design SS 32.2).
# ==========================================================================
G2_PAYLOAD_MAX_BYTES = 1_579_963          # SS 17; P9 SS 7 measured constant
G3_SNR_MAX = 544.341809                   # SS 15; Exp 8 measured D0 mean SNR
ROC_AUC_CI = (0.5755177227457472, 0.6911434704671701)   # SS 23
ROC_AUC_MDE = 0.05781287386071144                        # SS 24
BASELINE_ROC_AUC_MEAN = 0.6333305966064586               # SS 22 reference
T_CRIT = 2.7764451051977987                              # SS 19, df = 5 - 1 = 4
DF = 4

# Baseline expansion factor, pre-declared in design SS 17 for disclosure only.
# It is NEVER substituted for a measurement (design SS 16).
BASELINE_RAW_BYTES = 1_185_335
BASELINE_TRANSPORT_BYTES = 1_579_963

# ==========================================================================
# Task half - design SS 19, SS 20
# ==========================================================================
FOLD_MANIFEST = os.path.join("trainer_outputs", "baseline_cv", "fold_manifest.json")
BASELINE_SUMMARY = os.path.join("trainer_outputs", "baseline_cv",
                                "baseline_cv_summary.json")
N_FOLDS = 5
N_REPEATS = 5
N_PARTICIPANTS = 188
N_POSITIVE = 45
N_NEGATIVE = 143
BASE_SEED_TASK = 1000
EPOCHS = 3
BATCH_SIZE = 8
LR = 2e-5
GRAD_CLIP = 1.0
LAMBDA = 0.5
MAX_LEN = 128
BINARIZE_THRESHOLD = 10.0
BERT_MODEL = "mental/mental-bert-base-uncased"
TASK_METRIC = "ROC_AUC"

# The fold manifest is AUTHORITATIVE for the dataset (design SS 20). It names
# daic_records.parquet, and the frozen mentalbert_delta.pt is a TEXT-ONLY object
# (fusion.fc1.in_features == 768, i.e. audio_dim = vision_dim = None), so the two
# agree. The multimodal parquet must NOT be substituted here even though it
# exists: doing so would change a second factor and break the SS 22 K0 gate.
TASK_PARQUET = "daic_records.parquet"
AUDIO_DIM = None
VISION_DIM = None
EXPECTED_FUSION_IN = 768

# ==========================================================================
# Transport path - design SS 16, SS 18
# ==========================================================================
CHUNK_BYTES = 1024 * 1024      # 1 MB chunks (ARCHITECTURE.md:337)

# ==========================================================================
# Frozen input SHAs - design SS 28.1, SS 29, SS 31.
# Snapshotted BEFORE execution and re-verified AFTER, inside the execution path.
# ==========================================================================
FROZEN_SHA: Dict[str, str] = {
    "PHASE_18_EXP9_DESIGN.md":
        "cf361986e4f058259e21fefd548b335ad9fb91758696d57fdf6c762254bf288a",
    "trainer_mentalbert_daic.py":
        "65b1902e2ba8dd996c6960db1a6595d84fd48500f4d8707cb79688093d7a230b",
    "dp_agent/dp_agent.py":
        "758642fff57695cb970af88789c3b6a17c77f2b01d16303ff96230fce5798732",
    "trainer_outputs/mentalbert_delta.pt":
        "e1342c4bf1b9374d80b661fb7a8023a9189b59be49cb8cb0ada7e9c955d87ea5",
    "daic_records.parquet":
        "9a241851760727779fcaa4d50041b8bdc4406fe6b66ddbbd7105f4ce4fc55f00",
    "dataset_build/daic_records_multimodal.parquet":
        "1ac9f53e6102ec0dbaab84dcfdfa3f2e70f2b4a1867a841ea2b7e3ba24c67a95",
    "trainer_outputs/baseline_cv/fold_manifest.json":
        "b9a7a91f1bdd43c78d3cd1ee0c36e4ce3fb8be4b0971dcff3ff62ee5af8fca2f",
    "trainer_outputs/baseline_cv/baseline_cv_summary.json":
        "f065f5e1ff7f8e5ed4d122a8031c9cc12425802e756697d5512a915674871aac",
    "trainer_outputs/baseline_cv/trivial_control_arm.csv":
        "2ecbfee3225910034a959fe949c7ad027cfa7d41e9bef8040374578ff8c5d572",
    "trainer_outputs/local_probe_base.pt":
        "d21f95ab4169ba8bf270e2ae900d2d205ddb4c3aba3c04ebae4d0ec8d9ba0993",
    "trainer_outputs/exp3_convergence/exp3_summary.json":
        "9dd135b6b79af06a85e64a5aa8c1896d19df8059dd89c053e50e1cffe20af2f9",
    "trainer_outputs/exp4_decision_rule/exp4_summary.json":
        "5defdae2a20d0abc164611e8cbe6b8034e3f65c593d8c9b3e38266a1800aa6f2",
    "trainer_outputs/exp5_imbalance_objective/exp5_summary.json":
        "70ad13ffc4db633419595761c0396fc20dd1b62400ee74238be1f06b73ed48d3",
    "trainer_outputs/exp6_loss_rebalance/exp6_summary.json":
        "bea7478594f6af98943ddda2e997e8bc8780abf57c9c2ee720f9426cd2acc94c",
    "trainer_outputs/exp8_dp_mechanism/exp8_summary.json":
        "99540734c92717f130d1110f4ecddef23b15c5c964ba47356ec4559b170e05a3",
    "PHASE_17_EXP8_DESIGN.md":
        "bc7d77d0457811429112967d746c30ddf8a4e9d99bd41220f65cf5d90378c6a7",
    "FINAL_EXP8_CLOSURE.md":
        "58e95264334bb8e7f1307fb28cf39fb444020178d307a01ee611274d1c1fda53",
    "FINAL_EXP7_CLOSURE.md":
        "928eb7b0d6c5eab7cd3c68a631d97518053774d784ab33905a682d8bcc0d36eb",
}

# Frozen inputs that must be present on the Colab side only. The Terminal half
# does not touch them, so a Colab-only bundle is checked against this subset.
COLAB_REQUIRED: Tuple[str, ...] = (
    "PHASE_18_EXP9_DESIGN.md",
    "trainer_mentalbert_daic.py",
    "daic_records.parquet",
    "trainer_outputs/baseline_cv/fold_manifest.json",
    "trainer_outputs/baseline_cv/baseline_cv_summary.json",
)


# ==========================================================================
# Utilities
# ==========================================================================
def abort(message: str) -> "NoReturn":  # noqa: F821
    """Fail closed. Experiment 9 has no partial-success mode (design SS 29)."""
    print(f"\n[ABORT] {message}", file=sys.stderr)
    raise SystemExit(1)


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


def tensor_digest(t: torch.Tensor) -> str:
    """Bitwise SHA-256 of a tensor's raw buffer.

    This is what makes the design SS 28.10 reproducibility check REAL: two draws
    at the same seed must be byte-identical, not merely close in norm.
    """
    return sha256_bytes(t.detach().cpu().contiguous().numpy().tobytes())


# ==========================================================================
# Frozen-input integrity - design SS 28.1, SS 29, SS 31
# ==========================================================================
def snapshot_frozen(subset: Optional[Sequence[str]] = None) -> Dict[str, Optional[str]]:
    keys = list(FROZEN_SHA) if subset is None else list(subset)
    return {p: sha256_file(p) for p in keys}


def gate_frozen_before(snap: Dict[str, Optional[str]],
                       required: Optional[Sequence[str]] = None) -> None:
    """Pre-run gate, evaluated IN the execution path (design SS 31).

    A later verifier invocation is not a substitute: the gate has to hold at the
    instant the experiment runs.
    """
    req = list(snap) if required is None else list(required)
    missing = [p for p in req if snap.get(p) is None]
    if missing:
        abort(f"frozen input(s) missing before execution: {missing}")
    bad = [p for p in snap
           if snap[p] is not None and p in FROZEN_SHA and snap[p] != FROZEN_SHA[p]]
    if bad:
        detail = "; ".join(f"{p}: got {snap[p]} want {FROZEN_SHA[p]}" for p in bad)
        abort(f"frozen input SHA mismatch BEFORE execution: {detail}")


def gate_frozen_after(before: Dict[str, Optional[str]]) -> Dict[str, Optional[str]]:
    """Post-run re-verification. Any change DURING the run is a HARD ABORT (SS 29)."""
    after = snapshot_frozen(list(before))
    changed = [p for p in before if before[p] != after.get(p)]
    if changed:
        abort(f"frozen artifacts were modified DURING the run: {changed}")
    bad = [p for p in after
           if after[p] is not None and p in FROZEN_SHA and after[p] != FROZEN_SHA[p]]
    if bad:
        abort(f"frozen input SHA mismatch AFTER execution: {bad}")
    return after


# ==========================================================================
# THE PUBLIC MASK - design SS 6, SS 7
# ==========================================================================
_LAYER_RE = re.compile(r"^bert\.encoder\.layer\.(\d+)\.")

# design SS 7 - group sizes, transcribed from the frozen document.
EXPECTED_GROUP_PARAMS: Dict[str, int] = {
    "fusion.*": 197_892,
    "bert.pooler.dense": 590_592,
    "bert.encoder.layer.{0..11}": 85_054_464,     # 12 x 7,087,872
    "bert.embeddings.*": 23_837_184,
}
EXPECTED_PER_LAYER_PARAMS = 7_087_872
EXPECTED_GROUP_KEYS: Dict[str, int] = {
    "fusion.*": 8,
    "bert.pooler.dense": 2,
    "bert.encoder.layer.{0..11}": 192,            # 12 x 16
    "bert.embeddings.*": 5,
}


def group_rank(key: str) -> Tuple[int, int]:
    """Priority rank of a state-dict key under the frozen ordering O (design SS 6.1).

    INPUTS: the key NAME only. No tensor, no shape, no value is consulted here.

    Ordering:
        0  fusion.*                     (task head - randomly initialised at
                                         construction, so it is the component
                                         that must be learned from scratch)
        1  bert.pooler.dense
        2  bert.encoder.layer.0 .. 11   (ascending layer index)
        3  bert.embeddings.*
    """
    if key.startswith("fusion."):
        return (0, 0)
    if key.startswith("bert.pooler."):
        return (1, 0)
    m = _LAYER_RE.match(key)
    if m:
        return (2, int(m.group(1)))
    if key.startswith("bert.embeddings."):
        return (3, 0)
    # An unrecognised key would silently land at the end of O and change the
    # mask. That is a structural change to the experiment, not a detail.
    return (99, 0)


def group_name(key: str) -> str:
    r = group_rank(key)[0]
    return {0: "fusion.*", 1: "bert.pooler.dense",
            2: "bert.encoder.layer.{0..11}", 3: "bert.embeddings.*"}.get(r, "UNKNOWN")


def build_ordering(state_keys: Sequence[str],
                   numels: Dict[str, int]) -> List[Tuple[str, int, int]]:
    """Construct the frozen ordering O (design SS 6.1).

    PROHIBITED-INPUT GUARANTEE (design SS 6.3): the only arguments are key NAMES
    and element COUNTS. No tensor values are accessible to this function, so the
    ordering it returns cannot be a function of the private delta even by
    accident. Callers must pass numels derived from `.numel()` / shapes only.

    Within each structural group the FROZEN state_dict key order is preserved
    (Python dicts are insertion-ordered and `sorted` is stable, so the original
    order survives), and within each tensor the row-major flattened index order
    applies when the mask is used.

    Returns an ordered list of (key, numel, cumulative_end_exclusive).
    """
    idx = {k: i for i, k in enumerate(state_keys)}
    unknown = [k for k in state_keys if group_rank(k)[0] == 99]
    if unknown:
        abort(f"state-dict key(s) not covered by the frozen ordering O: {unknown}. "
              "The mask family is defined only over the frozen 207-key architecture "
              "(design SS 6.1, SS 7).")
    ordered = sorted(state_keys, key=lambda k: (group_rank(k), idx[k]))
    out: List[Tuple[str, int, int]] = []
    run = 0
    for k in ordered:
        run += numels[k]
        out.append((k, numels[k], run))
    return out


def verify_ordering(order: List[Tuple[str, int, int]]) -> Dict[str, Any]:
    """Assert the ordering matches design SS 7 exactly. HARD ABORT otherwise (SS 29)."""
    total = order[-1][2] if order else 0
    if total != EXPECTED_D:
        abort(f"d = {total} != frozen {EXPECTED_D} (design SS 29)")
    if len(order) != EXPECTED_N_KEYS:
        abort(f"{len(order)} keys != frozen {EXPECTED_N_KEYS} (design SS 29)")

    per_group_params: Dict[str, int] = {}
    per_group_keys: Dict[str, int] = {}
    per_layer: Dict[int, int] = {}
    for key, n, _ in order:
        g = group_name(key)
        per_group_params[g] = per_group_params.get(g, 0) + n
        per_group_keys[g] = per_group_keys.get(g, 0) + 1
        r = group_rank(key)
        if r[0] == 2:
            per_layer[r[1]] = per_layer.get(r[1], 0) + n

    for g, want in EXPECTED_GROUP_PARAMS.items():
        if per_group_params.get(g) != want:
            abort(f"group {g}: {per_group_params.get(g)} params != frozen {want} "
                  "(design SS 7)")
    for g, want in EXPECTED_GROUP_KEYS.items():
        if per_group_keys.get(g) != want:
            abort(f"group {g}: {per_group_keys.get(g)} keys != frozen {want} "
                  "(design SS 7)")
    if sorted(per_layer) != list(range(12)):
        abort(f"encoder layers present = {sorted(per_layer)}; expected 0..11")
    for li, n in per_layer.items():
        if n != EXPECTED_PER_LAYER_PARAMS:
            abort(f"bert.encoder.layer.{li}: {n} params != frozen "
                  f"{EXPECTED_PER_LAYER_PARAMS} (design SS 7)")

    # Monotone, gap-free prefix structure - the property the whole mask family
    # rests on.
    run = 0
    for key, n, cum in order:
        run += n
        if cum != run:
            abort(f"ordering O is not a contiguous prefix at {key}")

    return {"d": total, "n_keys": len(order),
            "group_params": per_group_params, "group_keys": per_group_keys,
            "per_layer_params": per_layer}


def mask_boundary(order: List[Tuple[str, int, int]], k: int) -> Tuple[int, str, int]:
    """Locate k under O: (index of the boundary key, its name, offset inside it).

    offset == numel means the mask ends exactly on that key's last coordinate.
    """
    if k < 0 or k > (order[-1][2] if order else 0):
        abort(f"k = {k} is outside [0, d]")
    run = 0
    for i, (key, n, _) in enumerate(order):
        if run + n >= k:
            return i, key, k - run
        run += n
    abort(f"k = {k} could not be located under O")   # unreachable; fail closed


def verify_nesting(order: List[Tuple[str, int, int]]) -> Dict[str, Any]:
    """Verify S_K3 subset S_K2 subset S_K1 subset S_K0 (design SS 9, SS 28.4).

    Every mask is a PREFIX of one fixed ordering, so nesting is equivalent to the
    k values being strictly increasing. That equivalence is checked here, not
    assumed, and the boundary of each k is recorded so the verifier can confirm
    no arm silently re-pooled or re-ordered.
    """
    ks = [K_VALUES[a] for a in ("K3", "K2", "K1", "K0")]
    strictly_increasing = all(ks[i] < ks[i + 1] for i in range(len(ks) - 1))
    if not strictly_increasing:
        abort(f"k ladder is not strictly increasing: {ks} - masks would not nest "
              "(design SS 29: one-factor violated)")
    detail = {}
    for arm in ARMS:
        i, key, off = mask_boundary(order, K_VALUES[arm])
        detail[arm] = {"k": K_VALUES[arm], "boundary_key": key,
                       "boundary_key_index": i, "offset_in_key": off,
                       "fraction_of_d": K_VALUES[arm] / order[-1][2]}
    return {"nested": True, "ascending_k": ks,
            "rule": "S_k = first k positions under O; prefixes of one ordering nest",
            "arms": detail}


def apply_mask_to_delta(delta: Dict[str, torch.Tensor],
                        order: List[Tuple[str, int, int]],
                        k: int) -> Dict[str, torch.Tensor]:
    """Return a FULL 207-key dict with coordinates outside S_k set to zero.

    design SS 11: the receiver places the release at positions S_k and zeros
    elsewhere, producing a full state dict so load_state_dict(strict=True) and
    the whole cryptographic pipeline operate unchanged.

    The boundary tensor is split by flattened (row-major) index, exactly as
    design SS 6.1 specifies.
    """
    bidx, _, off = mask_boundary(order, k)
    out: Dict[str, torch.Tensor] = {}
    for i, (key, n, _) in enumerate(order):
        t = delta[key]
        if i < bidx:
            out[key] = t.detach().cpu().clone()
        elif i > bidx:
            out[key] = torch.zeros_like(t, device="cpu")
        else:
            flat = t.detach().cpu().flatten().clone()
            if off < flat.numel():
                flat[off:] = 0.0
            out[key] = flat.view(t.shape)
    return out


def compact_vector(delta: Dict[str, torch.Tensor],
                   order: List[Tuple[str, int, int]],
                   k: int) -> torch.Tensor:
    """The k retained coordinates as a 1-D float32 vector, in O order.

    This is the object the DP mechanism acts on (design SS 11) and the object
    that is transported (design SS 10: the index set is public, so only values
    travel). Built by CONCATENATION IN ORDER - never by ranking values.
    """
    bidx, _, off = mask_boundary(order, k)
    parts: List[torch.Tensor] = []
    for i, (key, n, _) in enumerate(order):
        if i < bidx:
            parts.append(delta[key].detach().cpu().flatten().to(torch.float32))
        elif i == bidx:
            if off > 0:
                parts.append(
                    delta[key].detach().cpu().flatten().to(torch.float32)[:off])
            break
    v = torch.cat(parts) if parts else torch.zeros(0, dtype=torch.float32)
    if int(v.numel()) != k:
        abort(f"compact vector has {int(v.numel())} coordinates, expected k = {k}")
    return v


# ==========================================================================
# DP mechanism - design SS 11, SS 12
# ==========================================================================
def epsilon() -> float:
    """eps via the FROZEN dp_agent conversion. Identical for every arm.

    design SS 10: the mask is public, so no selection step consumes budget and
    T stays 1. eps is a function of sigma_eff = noise_multiplier / clip_norm
    ALONE - it does not depend on k, which is exactly why d is the only lever
    left (B-3 / design SS 2).
    """
    return _rdp_to_dp(noise_multiplier=NOISE_MULTIPLIER,
                      clip_norm=CLIP_NORM, delta=DELTA_DP)


def analytical_snr(k: int) -> float:
    """SNR_analytical(k) = sigma_eff * sqrt(k)  (design SS 13).

    REGISTERED IN ADVANCE AND ANALYTICAL ONLY. design SS 13 forbids using this
    as an acceptance input; G3 is judged on the MEASURED mean (design SS 15).
    """
    return SIGMA_EFF * math.sqrt(k)


def dp_release(vec: torch.Tensor, seed: int) -> Dict[str, Any]:
    """Clip to C = 1.0 then add isotropic N(0, sigma^2 I_k)  (design SS 11, SS 12).

    The release is over the k RETAINED COORDINATES ONLY. Noising the dense
    109,680,132-dimensional reconstruction would give SNR = sqrt(d) regardless
    of k and would nullify the experiment (design SS 11, "Critical").

    Sensitivity is Delta = C = 1.0 by construction - it is the L2 sensitivity of
    the clipped release, computed from clip_norm here, not a hardcoded constant
    (design SS 12).
    """
    v = vec.detach().cpu().to(torch.float32)
    l2_before = float(torch.norm(v, p=2))
    scale = min(1.0, CLIP_NORM / (l2_before + 1e-12))
    clipped = v * scale
    sensitivity = CLIP_NORM
    sigma = SIGMA_EFF * sensitivity                   # = 1.0

    torch.manual_seed(seed)                           # design SS 26
    noise = torch.normal(0.0, sigma, size=clipped.shape)
    noisy = clipped + noise

    l2_after = float(torch.norm(noisy, p=2))
    return {
        "k": int(v.numel()),
        "seed": seed,
        "l2_before": l2_before,
        "clip_scale": scale,
        "l2_signal_after_clip": float(torch.norm(clipped, p=2)),
        "l2_noise": float(torch.norm(noise, p=2)),
        "l2_after": l2_after,
        "snr": l2_after / CLIP_NORM,                  # design SS 14
        "sensitivity": sensitivity,
        "sigma": sigma,
        "draw_sha256": tensor_digest(noisy),
        "noisy": noisy,
    }


# ==========================================================================
# Transport payload - design SS 16, SS 17, SS 18
# ==========================================================================
def serialise_compact(vec: torch.Tensor, arm: str, k: int) -> bytes:
    """torch.save of the compact object (design SS 16 step 1).

    Only VALUES travel: design SS 10 states that because the mask is public the
    receiver reconstructs S_k from O and k, so indices need not be transmitted.
    k and the arm label are carried so the receiver knows which prefix to use;
    they are public metadata, not index data.
    """
    buf = io.BytesIO()
    torch.save({"experiment": "exp9", "arm": arm, "k": int(k),
                "mechanism": MECHANISM_NAME, "values": vec.contiguous()}, buf)
    return buf.getvalue()


def measure_transport_payload(payload: bytes, uri_name: str, out_dir: str,
                              store_agent: str = "exp9-compact") -> Dict[str, Any]:
    """The FROZEN payload measurement, identical for every arm (design SS 16).

        torch.save bytes
          -> AES-GCM encryption (SecureStore.encrypt_write)
          -> chunked GridFS upload representation (1 MB chunks)
          -> payload_hash = SHA-256 of ALL uploaded bytes
          -> G2 quantity = total uploaded byte count

    "Raw 4k float bytes must NOT be substituted" (design SS 16). raw_4k_bytes is
    reported alongside as a diagnostic ONLY; G2 is judged on
    total_uploaded_bytes.

    NOTHING IS UPLOADED. The chunking is the *upload representation* of the
    encrypted object - the bytes are split and hashed exactly as the transport
    would, and counted. No network call is made anywhere in this module.
    """
    from pathlib import Path
    from centralized_secure_store import SecureStore

    store_root = Path(out_dir) / "secure_store"
    store = SecureStore(agent=store_agent, root=store_root)
    enc_path = store_root / f"{uri_name}.enc"
    uri = f"file://{enc_path}"
    store.encrypt_write(uri, payload)
    with open(enc_path, "rb") as fh:
        uploaded = fh.read()

    # Round-trip proof that the measured bytes really are the transported object.
    back = store.decrypt_read(uri)
    if sha256_bytes(back) != sha256_bytes(payload):
        abort(f"AES-GCM round trip failed for {uri_name}: the measured payload is "
              "not a faithful encoding of the compact object")

    chunks = [uploaded[i:i + CHUNK_BYTES] for i in range(0, len(uploaded), CHUNK_BYTES)]
    chunk_hashes = [sha256_bytes(c) for c in chunks]
    total_uploaded = sum(len(c) for c in chunks)
    if total_uploaded != len(uploaded):
        abort("chunking lost bytes; the payload measurement is not faithful")

    return {
        "serialised_bytes": len(payload),
        "serialised_sha256": sha256_bytes(payload),
        "encrypted_bytes": len(uploaded),
        "n_chunks": len(chunks),
        "chunk_bytes": CHUNK_BYTES,
        "chunk_sha256": chunk_hashes,
        "total_uploaded_bytes": total_uploaded,       # <- the G2 quantity
        "payload_hash": sha256_bytes(uploaded),
        "encrypted_path": str(enc_path),
    }


# ==========================================================================
# Statistics - design SS 19, SS 23, SS 24
# ==========================================================================
def mean(xs: Sequence[float]) -> float:
    return float(sum(xs) / len(xs)) if xs else float("nan")


def sd(xs: Sequence[float], ddof: int = 1) -> float:
    n = len(xs)
    if n - ddof <= 0:
        return float("nan")
    m = mean(xs)
    return math.sqrt(sum((x - m) ** 2 for x in xs) / (n - ddof))


def fold_level_ci(values: Sequence[float]) -> Dict[str, float]:
    """Fold-level Student-t 95 % CI, df = k - 1 = 4, t_crit frozen (design SS 19).

    `values` are the FOLD means (repeats already averaged within fold), which is
    the project's standing convention for every CV experiment.
    """
    m = mean(values)
    s = sd(values, ddof=1)
    se = s / math.sqrt(len(values)) if values else float("nan")
    return {"mean": m, "sd": s, "se": se, "n": len(values),
            "df": len(values) - 1, "t_crit": T_CRIT,
            "ci95_lo": m - T_CRIT * se, "ci95_hi": m + T_CRIT * se}
