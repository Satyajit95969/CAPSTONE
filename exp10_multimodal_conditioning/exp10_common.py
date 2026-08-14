#!/usr/bin/env python3
"""
exp10_common.py - Phase 19 / Experiment 10 shared frozen constants and primitives.

Authoritative specification: PHASE_19_EXP10_DESIGN.md (FROZEN),
SHA-256 71fdb1b3c6b3c4bc402e688769a8aaae447283dfbe5ed62dcc3076f9c7b59a20.
Every constant here is transcribed from that document. None may be changed.

WHY THIS MODULE EXISTS
----------------------
The arms, the conditioning rule, the thresholds and the frozen SHAs are used by
the runner, the aggregator AND the verifier. Duplicating them three times would
create three places for them to drift, and drift in the conditioning rule or a
threshold is exactly what design SS 29 forbids. This module is the single
definition; it is a support module, not an additional experimental component.

WHAT THIS EXPERIMENT IS - and is NOT
------------------------------------
Experiment 10 tests ONE factor: conditioning of the pre-extracted audio and
vision feature vectors (design SS 1, SS 7). It adds NO differential privacy, NO
masking, NO compression, NO payload measurement and NO federation (design SS 5).
Nothing in this module computes epsilon, builds an ordering O, or serialises an
object for transport, and nothing may be added that does.
"""

from __future__ import annotations

import copy
import hashlib
import math
import os
import sys
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if REPO not in sys.path:
    sys.path.insert(0, REPO)


# ==========================================================================
# Identity - design SS 1, SS 2
# ==========================================================================
EXPERIMENT = "Phase 19 / Exp 10 - multimodal feature conditioning"
DESIGN_PATH = "PHASE_19_EXP10_DESIGN.md"
DESIGN_SHA = "71fdb1b3c6b3c4bc402e688769a8aaae447283dfbe5ed62dcc3076f9c7b59a20"
OUT_DIRNAME = "trainer_outputs/multimodal_exp"       # design SS 20, forward slashes

SCIENTIFIC_QUESTION = (
    "Does fold-safe conditioning of the pre-extracted audio and vision features "
    "change the multimodal ranking result under the frozen 5x5 CV protocol - i.e. "
    "is Experiment 7's multimodal degradation attributable to unconditioned "
    "feature scale rather than to the modalities lacking predictive information?"
)

SCOPE_CAVEAT = (
    "A positive result supports the feature-scale hypothesis FOR THIS "
    "ARCHITECTURE. A negative result establishes only that THIS conditioning "
    "strategy does not rescue THIS fusion architecture; it must NOT be "
    "interpreted as proof that audio and vision modalities carry no predictive "
    "information in general. Experiment 10 makes no privacy, compression, "
    "payload or federation claim, and no claim of demonstrated predictive "
    "utility (design SS 30)."
)

MDE_TRANSFER_ASSUMPTION = (
    "The ROC-AUC, PR-AUC and MAE MDEs are pre-existing constants published in "
    "baseline_cv_summary.json before Experiment 10 existed, but they were "
    "derived from TEXT-ONLY Baseline-CV fold variance. Their transfer to a "
    "MULTIMODAL contrast is a PRE-REGISTERED ASSUMPTION, not an established "
    "fact. It is frozen and may not be revised after any result (design SS 17)."
)

# ==========================================================================
# Frozen arms - design SS 6
# ==========================================================================
ARMS: Tuple[str, ...] = ("M0", "M1", "M2")
CONTROL_ARM = "M0"          # equivalence gate
RAW_ARM = "M1"              # raw-multimodal control
TREATMENT_ARM = "M2"        # conditioned multimodal

ARM_SPEC: Dict[str, Dict[str, Any]] = {
    "M0": {"audio_dim": None, "vision_dim": None, "conditioned": False,
           "fusion_in": 768, "n_params": 109_680_132, "n_keys": 207,
           "label": "text-only equivalence gate"},
    "M1": {"audio_dim": 154, "vision_dim": 84, "conditioned": False,
           "fusion_in": 1024, "n_params": 109_763_494, "n_keys": 215,
           "label": "raw multimodal control (regenerates Exp 7 A3)"},
    "M2": {"audio_dim": 154, "vision_dim": 84, "conditioned": True,
           "fusion_in": 1024, "n_params": 109_763_494, "n_keys": 215,
           "label": "conditioned multimodal treatment"},
}

AUDIO_DIM = 154
VISION_DIM = 84

# ==========================================================================
# Frozen conditioning rule - design SS 7
# ==========================================================================
DDOF = 0                      # population standard deviation (design SS 7.2)
ZERO_VARIANCE_RULE = "sigma_safe = sigma if sigma > 0 else 1.0"
CONDITIONING_METHOD = ("per-feature standardisation, train-fold statistics only")
CONDITIONING_EPSILON = None   # design SS 7.2 - NO epsilon of any kind

# design SS 7.4 - measured, identical in the full dataset and every train fold.
EXPECTED_AUDIO_ZERO_VARIANCE_IDX: Tuple[int, ...] = (16, 17, 68, 69, 70, 71,
                                                     72, 73, 74, 75)
EXPECTED_VISION_ZERO_VARIANCE_IDX: Tuple[int, ...] = ()

# ==========================================================================
# Frozen data and folds - design SS 8, SS 9
# ==========================================================================
MULTIMODAL_PARQUET = "dataset_build/daic_records_multimodal.parquet"
TEXT_PARQUET = "daic_records.parquet"
FOLD_MANIFEST = "trainer_outputs/baseline_cv/fold_manifest.json"
BASELINE_SUMMARY = "trainer_outputs/baseline_cv/baseline_cv_summary.json"

N_PARTICIPANTS = 188
N_POSITIVE = 45
N_NEGATIVE = 143
N_FOLDS = 5
N_REPEATS = 5
FOLD_IDS: Tuple[int, ...] = (1, 2, 3, 4, 5)     # design SS 9 - 1..5, NOT 0..4
EXPECTED_FOLD_SIZES: Dict[int, Tuple[int, int]] = {
    1: (150, 38), 2: (150, 38), 3: (150, 38), 4: (151, 37), 5: (151, 37),
}
BYTE_IDENTICAL_COLUMNS: Tuple[str, ...] = ("participant_id", "text", "phq_score")

# ==========================================================================
# Frozen training recipe - design SS 10
# ==========================================================================
BERT_MODEL = "mental/mental-bert-base-uncased"
EPOCHS = 3
LR = 2e-5
BATCH_SIZE = 8
MAX_LEN = 128
LAMBDA = 0.5
GRAD_CLIP = 1.0
BINARIZE_THRESHOLD = 10.0
REQUIRED_TRANSFORMERS = "4.44.0"

# ==========================================================================
# Frozen protocol and seeding - design SS 11
# ==========================================================================
BASE_SEED = 1000
REPEATS: Tuple[int, ...] = (1, 2, 3, 4, 5)
DF = 4
T_CRIT = 2.7764451051977987

# ==========================================================================
# Frozen thresholds - design SS 14, SS 17, SS 18
# ==========================================================================
BASELINE_ROC_AUC_MEAN = 0.6333305966064586
ROC_AUC_CI: Tuple[float, float] = (0.5755177227457472, 0.6911434704671701)
ROC_AUC_MDE = 0.05781287386071144
PR_AUC_MDE = 0.09501814822834184
MAE_MDE = 0.40660852779350554
CHANCE_ROC_AUC = 0.5

PRIMARY_METRIC = "ROC_AUC"
SECONDARY_METRIC = "PR_AUC"
GUARD_METRIC = "MAE"
DIAGNOSTIC_METRIC = "pred_var"
# design SS 16 - reported, NEVER acceptance inputs
NON_ACCEPTANCE_METRICS: Tuple[str, ...] = (
    "Precision", "Recall", "F1", "balAcc", "MCC", "Accuracy")

# ==========================================================================
# Frozen input SHAs - design SS 13.
# Keys are CANONICAL REPO-RELATIVE FORWARD-SLASH paths (design SS 22, and the
# Windows os.path.join defect found in Experiment 9). Never index this dict with
# an unnormalised path; use frozen_sha_for().
# ==========================================================================
FROZEN_SHA: Dict[str, str] = {
    "PHASE_19_EXP10_DESIGN.md":
        "71fdb1b3c6b3c4bc402e688769a8aaae447283dfbe5ed62dcc3076f9c7b59a20",
    "dataset_build/daic_records_multimodal.parquet":
        "1ac9f53e6102ec0dbaab84dcfdfa3f2e70f2b4a1867a841ea2b7e3ba24c67a95",
    "daic_records.parquet":
        "9a241851760727779fcaa4d50041b8bdc4406fe6b66ddbbd7105f4ce4fc55f00",
    "trainer_mentalbert_daic.py":
        "65b1902e2ba8dd996c6960db1a6595d84fd48500f4d8707cb79688093d7a230b",
    "trainer_outputs/baseline_cv/fold_manifest.json":
        "b9a7a91f1bdd43c78d3cd1ee0c36e4ce3fb8be4b0971dcff3ff62ee5af8fca2f",
    "trainer_outputs/baseline_cv/baseline_cv_summary.json":
        "f065f5e1ff7f8e5ed4d122a8031c9cc12425802e756697d5512a915674871aac",
    "trainer_outputs/baseline_cv/trivial_control_arm.csv":
        "2ecbfee3225910034a959fe949c7ad027cfa7d41e9bef8040374578ff8c5d572",
    "PHASE_16_EXP7_DESIGN.md":
        "098b7bb11c57012c3e8c56a790423a1741fc7e65654b34f0a87e1f4c59fb6be1",
    "FINAL_EXP7_CLOSURE.md":
        "928eb7b0d6c5eab7cd3c68a631d97518053774d784ab33905a682d8bcc0d36eb",
    "PHASE_18_EXP9_DESIGN.md":
        "cf361986e4f058259e21fefd548b335ad9fb91758696d57fdf6c762254bf288a",
}

# Subset the Colab task runner must gate on (the design file, the trainer, the
# dataset, the manifest and the gate reference).
COLAB_REQUIRED: Tuple[str, ...] = (
    "PHASE_19_EXP10_DESIGN.md",
    "trainer_mentalbert_daic.py",
    "dataset_build/daic_records_multimodal.parquet",
    "daic_records.parquet",
    "trainer_outputs/baseline_cv/fold_manifest.json",
    "trainer_outputs/baseline_cv/baseline_cv_summary.json",
)

# design SS 12.6 - never loaded, under any circumstance.
FORBIDDEN_CHECKPOINTS: Tuple[str, ...] = (
    "trainer_outputs/exp7_modality/ablation_multimodal/model.pt",
    "trainer_outputs/exp7_modality/ablation_multimodal/delta.pt",
    "trainer_outputs/exp7_modality/ablation_text_only/model.pt",
    "trainer_outputs/exp7_modality/ablation_text_only/delta.pt",
    "trainer_outputs/mentalbert_delta.pt",
    "trainer_outputs/local_probe_base.pt",
)


# ==========================================================================
# Utilities
# ==========================================================================
def abort(message: str) -> "NoReturn":  # noqa: F821
    """Fail closed. Experiment 10 has no partial-success mode (design SS 26)."""
    print(f"\n[ABORT] {message}", file=sys.stderr)
    raise SystemExit(1)


def repo_key(path: str) -> str:
    """Canonical repo-relative key for a FROZEN_SHA lookup.

    PLATFORM-NEUTRAL: on POSIX os.sep is already "/", so this is a no-op; on
    Windows it converts os.path.join output to the forward-slash form the dict
    is keyed with (design SS 22).
    """
    return path.replace("\\", "/")


def frozen_sha_for(path: str) -> Optional[str]:
    """Registered SHA for a path, or None if it is not pinned.

    Returning None instead of raising keeps callers FAIL-CLOSED: an unpinned or
    misspelt path becomes a recorded FAIL rather than an uncontrolled KeyError.
    """
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


def vector_digest(v: np.ndarray) -> str:
    """SHA-256 over the raw float64 buffer of a statistic vector (design SS 21.2).

    Enables bitwise independent recomputation by the verifier.
    """
    return sha256_bytes(np.ascontiguousarray(v, dtype=np.float64).tobytes())


# ==========================================================================
# Frozen-input integrity - design SS 13, SS 21.5, SS 26
# ==========================================================================
def snapshot_frozen(subset: Optional[Sequence[str]] = None) -> Dict[str, Optional[str]]:
    keys = list(FROZEN_SHA) if subset is None else [repo_key(p) for p in subset]
    return {p: sha256_file(p) for p in keys}


def gate_frozen_before(snap: Dict[str, Optional[str]],
                       required: Optional[Sequence[str]] = None) -> None:
    """Pre-run gate, evaluated IN the execution path (design SS 28)."""
    req = list(snap) if required is None else [repo_key(p) for p in required]
    missing = [p for p in req if snap.get(p) is None]
    if missing:
        abort(f"frozen input(s) missing before execution: {missing}")
    bad = []
    for p, got in snap.items():
        want = frozen_sha_for(p)
        if got is not None and want is not None and got != want:
            bad.append(f"{p}: got {got} want {want}")
    if bad:
        abort("frozen input SHA mismatch BEFORE execution: " + "; ".join(bad))


def gate_frozen_after(before: Dict[str, Optional[str]]) -> Dict[str, Optional[str]]:
    """Post-run re-verification. Any change DURING the run is a HARD ABORT."""
    after = snapshot_frozen(list(before))
    changed = [p for p in before if before[p] != after.get(p)]
    if changed:
        abort(f"frozen artifacts were modified DURING the run: {changed}")
    bad = [p for p, got in after.items()
           if got is not None and frozen_sha_for(p) is not None
           and got != frozen_sha_for(p)]
    if bad:
        abort(f"frozen input SHA mismatch AFTER execution: {bad}")
    return after


def assert_no_forbidden_checkpoint_loaded(loaded_paths: Sequence[str]) -> None:
    """design SS 12.6 - no checkpoint may ever be loaded as an initialisation."""
    bad = [p for p in loaded_paths if repo_key(p) in FORBIDDEN_CHECKPOINTS]
    if bad:
        abort(f"forbidden checkpoint load attempted: {bad} (design SS 12.6)")


# ==========================================================================
# THE CONDITIONER - design SS 7. PURE ARITHMETIC, ZERO RNG.
# ==========================================================================
class FoldConditioner:
    """Per-feature standardisation fitted on ONE fold's TRAIN partition only.

    design SS 7.1:
        mu[j]         = mean over TRAIN participants only
        sigma[j]      = population std (ddof = 0) over TRAIN only
        sigma_safe[j] = sigma[j] if sigma[j] > 0 else 1.0
        x'[j]         = (x[j] - mu[j]) / sigma_safe[j]

    NO EPSILON (design SS 7.2). NO feature dropping. NO global statistics. NO
    test statistics.

    ZERO RNG (design SS 11): this class calls no random function of any kind.
    numpy.random, torch.rand*, torch.randperm and random.* are never referenced,
    so all three arms share a byte-identical torch RNG stream.
    """

    def __init__(self, train_matrix: np.ndarray, name: str) -> None:
        X = np.asarray(train_matrix, dtype=np.float64)
        if X.ndim != 2 or X.shape[0] == 0:
            abort(f"{name}: conditioner needs a non-empty 2-D train matrix, "
                  f"got shape {X.shape}")
        self.name = name
        self.n_train = int(X.shape[0])
        self.n_features = int(X.shape[1])
        self.mu = X.mean(axis=0)
        self.sigma = X.std(axis=0, ddof=DDOF)          # design SS 7.2: ddof = 0
        # design SS 7.2 - the ONLY guard; no additive epsilon anywhere.
        self.sigma_safe = np.where(self.sigma > 0.0, self.sigma, 1.0)
        self.zero_variance_idx = tuple(int(i) for i in
                                       np.where(self.sigma <= 0.0)[0])

    def apply(self, matrix: np.ndarray) -> np.ndarray:
        """Apply the TRAIN-derived statistics. Used for train AND test."""
        X = np.asarray(matrix, dtype=np.float64)
        if X.shape[1] != self.n_features:
            abort(f"{self.name}: expected {self.n_features} features, "
                  f"got {X.shape[1]}")
        Z = (X - self.mu) / self.sigma_safe
        if not np.isfinite(Z).all():
            abort(f"{self.name}: conditioning produced non-finite values "
                  "(design SS 26: HARD ABORT)")
        return Z

    def manifest(self) -> Dict[str, Any]:
        return {
            "mu_sha256": vector_digest(self.mu),
            "sigma_sha256": vector_digest(self.sigma),
            "zero_variance_indices": list(self.zero_variance_idx),
            "n_zero_variance": len(self.zero_variance_idx),
        }


def extract_modality_matrices(records: Sequence[Dict[str, Any]]
                              ) -> Tuple[np.ndarray, np.ndarray]:
    """Pull the audio/vision vectors out of already-JSON-decoded records.

    Reads exactly the keys the FROZEN trainer reads
    (trainer_mentalbert_daic.py:113 features["audio"]["wav2vec2"],
     trainer_mentalbert_daic.py:124 features["video"]["densenet"]).
    Read-only: the records are not modified here.
    """
    A: List[List[float]] = []
    V: List[List[float]] = []
    for r in records:
        f = r.get("features") or {}
        if not isinstance(f, dict):
            abort("record 'features' is not a dict; read_parquet_records should "
                  "have JSON-decoded it (trainer_mentalbert_daic.py:90-96)")
        a = (f.get("audio") or {}).get("wav2vec2")
        v = (f.get("video") or {}).get("densenet")
        if a is None or v is None:
            abort(f"record {r.get('participant_id')!r} is missing audio or "
                  "vision features")
        A.append([float(x) for x in a])
        V.append([float(x) for x in v])
    Am = np.asarray(A, dtype=np.float64)
    Vm = np.asarray(V, dtype=np.float64)
    if Am.shape[1] != AUDIO_DIM or Vm.shape[1] != VISION_DIM:
        abort(f"feature dims ({Am.shape[1]}, {Vm.shape[1]}) != frozen "
              f"({AUDIO_DIM}, {VISION_DIM}) (design SS 8)")
    return Am, Vm


def condition_records(records: Sequence[Dict[str, Any]],
                      audio_cond: FoldConditioner,
                      vision_cond: FoldConditioner) -> List[Dict[str, Any]]:
    """Return DEEP COPIES with conditioned audio/vision values substituted.

    design SS 12.2 - MANDATORY DEEP COPY. The discovery audit established that
    run_exp7.py builds `by_pid = {pid: r}` and that train/test lists hold THE
    SAME dict objects. Mutating in place would corrupt every fold AND leak M2's
    conditioned values into M1. copy.deepcopy is the guarantee, not a style
    choice.
    """
    A, V = extract_modality_matrices(records)
    Az = audio_cond.apply(A)
    Vz = vision_cond.apply(V)
    out: List[Dict[str, Any]] = []
    for i, r in enumerate(records):
        rc = copy.deepcopy(r)                      # design SS 12.2
        rc["features"]["audio"]["wav2vec2"] = [float(x) for x in Az[i]]
        rc["features"]["video"]["densenet"] = [float(x) for x in Vz[i]]
        out.append(rc)
    return out


def copy_records(records: Sequence[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Deep copies with values UNCHANGED - used by M0 and M1 (design SS 12.2)."""
    return [copy.deepcopy(r) for r in records]


def negative_control(train_matrix: np.ndarray, test_matrix: np.ndarray,
                     full_matrix: np.ndarray, cond: FoldConditioner,
                     name: str) -> Dict[str, Any]:
    """Prove the statistics were FOLD-LOCAL, not global (design SS 12.3).

    Conditions the test partition with the fold's train statistics and with
    FULL-DATASET statistics, and asserts the two differ. If they did not, the
    fitting was not fold-local and the experiment would be contaminated.
    """
    full = np.asarray(full_matrix, dtype=np.float64)
    mu_full = full.mean(axis=0)
    sd_full = full.std(axis=0, ddof=DDOF)
    sd_full_safe = np.where(sd_full > 0.0, sd_full, 1.0)
    z_fold = cond.apply(test_matrix)
    z_full = (np.asarray(test_matrix, dtype=np.float64) - mu_full) / sd_full_safe
    differs = not np.allclose(z_fold, z_full, rtol=0.0, atol=0.0)
    max_abs_diff = float(np.max(np.abs(z_fold - z_full))) if z_fold.size else 0.0
    if not differs:
        abort(f"negative control FAILED for {name}: fold-local conditioning is "
              "identical to full-dataset conditioning; statistics were not "
              "fold-local (design SS 12.3, SS 26: HARD ABORT)")
    return {"modality": name, "fold_local_confirmed": True,
            "max_abs_difference_vs_global": max_abs_diff}


# ==========================================================================
# Statistics - design SS 11, SS 15, SS 18
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
    """Fold-level Student-t 95 % CI, df = k - 1 = 4, t_crit frozen (design SS 11).

    `values` are the FOLD means (repeats already averaged within fold), the
    project's standing convention for every CV experiment.
    """
    m = mean(values)
    s = sd(values, ddof=1)
    se = s / math.sqrt(len(values)) if values else float("nan")
    return {"mean": m, "sd": s, "se": se, "n": len(values),
            "df": len(values) - 1, "t_crit": T_CRIT,
            "ci95_lo": m - T_CRIT * se, "ci95_hi": m + T_CRIT * se}


def fold_means(rows: Sequence[Dict[str, str]], arm: str, metric: str
               ) -> Dict[int, float]:
    """Average the repeats WITHIN each fold (design SS 11)."""
    buckets: Dict[int, List[float]] = {}
    for r in rows:
        if r["arm"] != arm:
            continue
        v = float(r[metric])
        if math.isnan(v):
            abort(f"{arm} repeat {r['repeat']} fold {r['fold']}: {metric} is "
                  "NaN; a degenerate fold cannot be averaged")
        buckets.setdefault(int(r["fold"]), []).append(v)
    return {f: mean(v) for f, v in sorted(buckets.items())}


def paired_contrast(a_folds: Dict[int, float], b_folds: Dict[int, float]
                    ) -> Dict[str, Any]:
    """Paired within-fold contrast a - b, fold-level Student-t (design SS 15).

    Sign convention is the CALLER's responsibility and is documented at each
    call site in aggregate_exp10.py (design SS 18).
    """
    folds = sorted(set(a_folds) & set(b_folds))
    if len(folds) != N_FOLDS:
        abort(f"paired contrast needs {N_FOLDS} common folds, got {len(folds)}")
    deltas = {f: a_folds[f] - b_folds[f] for f in folds}
    st = fold_level_ci(list(deltas.values()))
    return {"per_fold": {str(f): deltas[f] for f in folds}, **st}
