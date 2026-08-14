#!/usr/bin/env python3
"""
run_exp8.py - Phase 17 / Experiment 8: DP mechanism (INFRASTRUCTURE experiment).

Authoritative specification: PHASE_17_EXP8_DESIGN.md (FROZEN). Every constant,
threshold, allocation, seed and criterion below is taken from that document and
may not be changed. Section references (design SS n) mark the provenance of each
critical block.

THIS IS NOT A PREDICTIVE-UTILITY EXPERIMENT (design SS 1, SS 24). No task metric is
computed, no model is trained, no dataset or fold is read.

TWO ARMS ONLY (design SS 10, SS 11):
    D0  frozen Gaussian baseline           flat clip C=1.0, sigma=1.0
    D1  per-group clipping + Mahalanobis-accounted Gaussian

Both arms consume the IDENTICAL pre-DP object (design SS 19) and are evaluated at
an IDENTICAL privacy budget enforced by the equivalence condition of design SS 11.2.

The frozen RDP->(eps,delta) conversion is IMPORTED from dp_agent, never
reimplemented and never modified (design SS 8, SS 22).

Usage:
    python exp8_dp_mechanism/run_exp8.py
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import io
import json
import math
import os
import sys
import time
from typing import Any, Dict, List, Optional, Tuple

import torch

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO)

# FROZEN import - the single source of the RDP -> (eps, delta) conversion.
# dp_agent/dp_agent.py is frozen (design SS 8, SS 22): imported, never edited.
from dp_agent.dp_agent import _rdp_to_dp  # noqa: E402


# ==========================================================================
# FROZEN CONSTANTS - design SS 9, SS 10, SS 12, SS 13, SS 15, SS 16
# ==========================================================================
TARGET_PATH = os.path.join(REPO, "trainer_outputs", "local_probe_base.pt")

# design SS 9 - target object. d is FIXED; reducing it is forbidden (SS 8.1).
EXPECTED_D = 295_681
FROZEN_GROUP_ORDER: Tuple[str, ...] = (
    "fc1.weight", "fc1.bias", "fc2.weight", "fc2.bias",
)
EXPECTED_GROUP_D: Dict[str, int] = {
    "fc1.weight": 294_912,
    "fc1.bias": 384,
    "fc2.weight": 384,
    "fc2.bias": 1,
}

# design SS 10 - control arm D0
D0_CLIP_NORM = 1.0
D0_NOISE_MULTIPLIER = 1.0
D0_SIGMA_EFF = D0_NOISE_MULTIPLIER / D0_CLIP_NORM        # = 1.0

# design SS 12 - privacy parameters
DELTA = 1e-05
EPS_PER_UPDATE_MAX = 5.302585092994046
EPS_CUMULATIVE_MAX = 15.9078
N_UPDATES_PER_ROUND = 3
EPS_MAX_CEILING = 100          # ceiling only, NOT the acceptance target (SS 12)
COMPOSITION = "RDP single-composition, T = 1 (output perturbation)"

# design SS 13 - SNR. Both arms share the same global signal budget, so the
# denominator is identical and the arms are directly comparable.
GLOBAL_CLIP_NORM = 1.0

# design SS 15.1, SS 16 - protocol and seeding
N_REPETITIONS = 5
BASE_SEED = 1000
SEEDS: Tuple[int, ...] = tuple(BASE_SEED + i for i in range(1, N_REPETITIONS + 1))
# -> (1001, 1002, 1003, 1004, 1005)

# Numerical tolerance for the privacy-equivalence assertion (design SS 19).
EQUIV_TOL = 1e-9

ARMS: Tuple[str, ...] = ("D0", "D1")
ARM_LABEL = {
    "D0": "frozen Gaussian baseline (flat clip)",
    "D1": "per-group clipping + Mahalanobis-accounted Gaussian",
}

# design SS 10 - published D0 L2-after band (P9 SS 6.2 / SS 8):
# 543.02, 544.04, 543.75, 543.73, 542.61, 544.67; sweep 543.49.
D0_BAND_LO, D0_BAND_HI = 540.0, 548.0

# ==========================================================================
# FROZEN INPUT SHAs - design SS 19 (hard abort on change), SS 21, SS 22
# Snapshotted BEFORE execution and re-verified AFTER, inside this runner.
# A separate verifier invocation is explicitly NOT sufficient (design SS 19).
# ==========================================================================
FROZEN_SHA: Dict[str, str] = {
    "trainer_mentalbert_daic.py":
        "65b1902e2ba8dd996c6960db1a6595d84fd48500f4d8707cb79688093d7a230b",
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
    "trainer_outputs/exp3_convergence/exp3_summary.json":
        "9dd135b6b79af06a85e64a5aa8c1896d19df8059dd89c053e50e1cffe20af2f9",
    "trainer_outputs/exp4_decision_rule/exp4_summary.json":
        "5defdae2a20d0abc164611e8cbe6b8034e3f65c593d8c9b3e38266a1800aa6f2",
    "trainer_outputs/exp5_imbalance_objective/exp5_summary.json":
        "70ad13ffc4db633419595761c0396fc20dd1b62400ee74238be1f06b73ed48d3",
    "trainer_outputs/exp6_loss_rebalance/exp6_summary.json":
        "bea7478594f6af98943ddda2e997e8bc8780abf57c9c2ee720f9426cd2acc94c",
}
# Frozen inputs with no published constant to pin against. Their SHAs are
# captured at run start and re-checked at run end, so any change DURING the run
# is still a hard abort (design SS 19).
FROZEN_UNPINNED: Tuple[str, ...] = (
    "dp_agent/dp_agent.py",
    "trainer_outputs/local_probe_base.pt",
    "PHASE_17_EXP8_DESIGN.md",
    "FINAL_EXP7_CLOSURE.md",
)


def abort(message: str) -> "NoReturn":  # noqa: F821
    """Fail closed. Experiment 8 has no partial-success mode (design SS 19)."""
    print(f"\n[ABORT] {message}", file=sys.stderr)
    raise SystemExit(1)


def sha256_file(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for b in iter(lambda: fh.read(8192), b""):
            h.update(b)
    return h.hexdigest()


def sha256_bytes(b: bytes) -> str:
    return hashlib.sha256(b).hexdigest()


# ==========================================================================
# F1 - frozen-input SHA integrity, IN THE EXECUTION PATH (design SS 19, SS 22)
# ==========================================================================
def snapshot_frozen() -> Dict[str, Optional[str]]:
    """SHA-256 of every frozen input. Taken before AND after execution."""
    snap: Dict[str, Optional[str]] = {}
    for p in list(FROZEN_SHA) + list(FROZEN_UNPINNED):
        snap[p] = sha256_file(p) if os.path.isfile(p) else None
    return snap


def gate_frozen_before(snap: Dict[str, Optional[str]]) -> None:
    """Pre-run gate: every pinned frozen input must match its published SHA.

    design SS 19 - a mismatch is a HARD ABORT, evaluated here in the real
    execution path. A later verifier invocation is explicitly NOT sufficient.
    """
    missing = [p for p, v in snap.items() if v is None]
    if missing:
        abort(f"frozen input(s) missing before execution: {missing}")
    bad = [p for p, want in FROZEN_SHA.items() if snap.get(p) != want]
    if bad:
        detail = "; ".join(f"{p}: got {snap.get(p)} want {FROZEN_SHA[p]}" for p in bad)
        abort(f"frozen input SHA mismatch BEFORE execution: {detail}")


def gate_frozen_after(before: Dict[str, Optional[str]]) -> Dict[str, Optional[str]]:
    """Post-run re-verification (design SS 22). Any change during the run aborts."""
    after = snapshot_frozen()
    changed = [p for p in before if before[p] != after.get(p)]
    if changed:
        abort(f"frozen artifacts were modified DURING the run: {changed}")
    bad = [p for p, want in FROZEN_SHA.items() if after.get(p) != want]
    if bad:
        abort(f"frozen input SHA mismatch AFTER execution: {bad}")
    return after


# ==========================================================================
# F4 - experiment-local HMAC key (design SS 22: writes only under out_dir)
# ==========================================================================
def local_receipt_key_source(out_dir: str) -> str:
    """Provision an Exp-8-local HMAC key and return a 'file:' key_source.

    CentralReceiptManager's DEFAULT constructor writes an HMAC key to
    ~/.local_data_agent_receipt_key when absent (centralised_receipts.py:25-30),
    which is OUTSIDE the authorised output directory and would violate design
    SS 22. The manager already supports an explicit key_source of the form
    'file:<path>', so Exp 8 provisions its own key INSIDE out_dir and passes it.

    Consequences, all intended:
      * nothing outside trainer_outputs/exp8_dp_mechanism/ is created or touched;
      * the pre-existing global key is neither read, modified nor deleted;
      * cryptographic strength is unchanged - still HMAC-SHA256 over a 32-byte
        secret from secrets.token_bytes (a CSPRNG independent of the torch RNG,
        so it cannot perturb the seeded experimental noise stream).
    """
    import base64
    import secrets

    key_dir = os.path.join(out_dir, "keys")
    os.makedirs(key_dir, exist_ok=True)
    key_path = os.path.join(key_dir, "exp8_receipt_key.b64")
    if not os.path.isfile(key_path):
        # Provisioned BEFORE any draw, so it cannot touch the experimental RNG.
        with open(key_path, "w", encoding="utf-8") as fh:
            fh.write(base64.b64encode(secrets.token_bytes(32)).decode())
    return f"file:{key_path}"


# ==========================================================================
# Target object - design SS 9
# ==========================================================================
def load_target() -> Tuple[Dict[str, torch.Tensor], List[Tuple[str, int, float]], str]:
    """Load the frozen Track-A probe and its four frozen tensor groups.

    Returns (state_dict, groups, sha) where groups is an ordered list of
    (name, d_g, ||v_g||_2) in FROZEN_GROUP_ORDER.

    Hard-aborts if d != 295,681 or the group structure differs (design SS 19).
    """
    if not os.path.isfile(TARGET_PATH):
        abort(f"target object not found: {TARGET_PATH}")
    sha = sha256_file(TARGET_PATH)
    sd = torch.load(TARGET_PATH, map_location="cpu", weights_only=False)
    if not isinstance(sd, dict):
        abort("target object is not a state_dict")

    missing = [k for k in FROZEN_GROUP_ORDER if k not in sd]
    extra = [k for k in sd if k not in FROZEN_GROUP_ORDER]
    if missing or extra:
        abort(f"group structure differs from the frozen specification; "
              f"missing={missing} extra={extra}")

    groups: List[Tuple[str, int, float]] = []
    for name in FROZEN_GROUP_ORDER:
        t = sd[name].detach().cpu().flatten().to(torch.float32)
        d_g = int(t.numel())
        if d_g != EXPECTED_GROUP_D[name]:
            abort(f"group {name}: d_g={d_g} != frozen {EXPECTED_GROUP_D[name]}")
        groups.append((name, d_g, float(torch.norm(t, p=2))))

    d = sum(g[1] for g in groups)
    # design SS 8.1 - d is FIXED. Any deviation is a hard abort.
    if d != EXPECTED_D:
        abort(f"d={d} != frozen {EXPECTED_D}; dimensionality must not change")
    return sd, groups, sha


def flatten(sd: Dict[str, torch.Tensor]) -> torch.Tensor:
    """Flatten in FROZEN_GROUP_ORDER (deterministic, matches dp_agent's float32)."""
    return torch.cat(
        [sd[k].detach().cpu().flatten().to(torch.float32) for k in FROZEN_GROUP_ORDER]
    )


# ==========================================================================
# Allocation - design SS 11.3, SS 11.4
# ==========================================================================
def allocate(groups: List[Tuple[str, int, float]],
             sigma_eff: float = D0_SIGMA_EFF) -> Dict[str, Any]:
    """FROZEN per-group allocation.

    design SS 11.4:   C_g proportional to ||v_g||_2,  normalised so sum_g C_g^2 = 1
    design SS 11.3:   u_g := C_g/sigma_g  proportional to (C_g^2 d_g)^(1/4)
                      subject to   sum_g (C_g/sigma_g)^2 = 1/sigma_eff^2
                      optimum      min ||n||_2 = sigma_eff * sum_g C_g sqrt(d_g)

    ANTI-TUNING (design SS 11.4): this allocation is closed-form and fixed in
    advance. It must never be re-chosen after observing any result.
    """
    names = [g[0] for g in groups]
    d_g = [float(g[1]) for g in groups]
    v_g = [g[2] for g in groups]

    # C_g proportional to ||v_g||, normalised so that sum C_g^2 = 1
    denom = math.sqrt(sum(x * x for x in v_g))
    if denom <= 0.0:
        abort("target object has zero norm; allocation undefined")
    C_g = [x / denom for x in v_g]

    # a_g = C_g^2 d_g ;  u_g^2 = K * sqrt(a_g)  with  sum u_g^2 = 1/sigma_eff^2
    a_g = [C_g[i] ** 2 * d_g[i] for i in range(len(groups))]
    sqrt_a = [math.sqrt(x) for x in a_g]
    K = (1.0 / (sigma_eff ** 2)) / sum(sqrt_a)
    u_g = [math.sqrt(K * s) for s in sqrt_a]
    sigma_g = [C_g[i] / u_g[i] for i in range(len(groups))]

    # design SS 11.2 - Mahalanobis sensitivity of the non-isotropic Gaussian
    delta_sigma_sq = sum((C_g[i] / sigma_g[i]) ** 2 for i in range(len(groups)))

    return {
        "names": names,
        "d_g": [int(x) for x in d_g],
        "v_g": v_g,
        "C_g": C_g,
        "sigma_g": sigma_g,
        "u_g": u_g,
        "sum_C_g_squared": sum(c * c for c in C_g),
        "delta_sigma_sq": delta_sigma_sq,
        "delta_sigma": math.sqrt(delta_sigma_sq),
        # design SS 11.3 closed form
        "analytical_noise_norm": sigma_eff * sum(C_g[i] * math.sqrt(d_g[i])
                                                 for i in range(len(groups))),
    }


# ==========================================================================
# Privacy accounting - design SS 11.2, SS 12
# ==========================================================================
def epsilon_from_mahalanobis(delta_sigma: float, delta: float = DELTA) -> float:
    """eps via the FROZEN conversion, driven by the Mahalanobis sensitivity.

    design SS 11.2:  eps_RDP(alpha) = alpha * Delta_Sigma^2 / 2
    dp_agent._rdp_to_dp(nm, clip, delta) computes with sigma_eff = nm/clip:
                     eps_RDP(alpha) = alpha / (2 * sigma_eff^2)

    Setting sigma_eff := 1/Delta_Sigma makes the two identical, so the frozen
    conversion is reused verbatim rather than reimplemented.
    """
    if delta_sigma <= 0.0:
        return math.inf
    return _rdp_to_dp(noise_multiplier=1.0 / delta_sigma,
                      clip_norm=1.0, delta=delta)


# ==========================================================================
# Mechanisms - design SS 10 (D0), SS 11 (D1)
# ==========================================================================
def run_d0(sd: Dict[str, torch.Tensor], seed: int) -> Dict[str, Any]:
    """D0 - frozen Gaussian baseline (design SS 10).

    Reproduces dp_agent's computation at its frozen settings: flat L2 clip to
    clip_norm=1.0, then isotropic Gaussian with scale = noise_multiplier *
    sensitivity, where sensitivity == clip_norm == 1.0 (the two coincide exactly
    at the baseline, design SS 11.1).
    """
    flat = flatten(sd)
    l2_before = float(torch.norm(flat, p=2))
    clipped = flat * (D0_CLIP_NORM / (l2_before + 1e-12)) if l2_before > D0_CLIP_NORM else flat
    sigma = D0_SIGMA_EFF * D0_CLIP_NORM          # = 1.0

    torch.manual_seed(seed)                       # design SS 16
    noise = torch.normal(0.0, sigma, size=clipped.shape)
    noisy = clipped + noise

    l2_after = float(torch.norm(noisy, p=2))
    delta_sigma = D0_CLIP_NORM / sigma            # = 1/sigma_eff
    return {
        "arm": "D0", "seed": seed,
        "l2_before": l2_before,
        "l2_signal_after_clip": float(torch.norm(clipped, p=2)),
        "l2_noise": float(torch.norm(noise, p=2)),
        "l2_after": l2_after,
        "snr": l2_after / GLOBAL_CLIP_NORM,       # design SS 13
        "distortion_ratio": l2_after / l2_before, # diagnostic only (design SS 17.2)
        "delta_sigma": delta_sigma,
        "epsilon": epsilon_from_mahalanobis(delta_sigma),
        "draw_sha256": tensor_digest(noisy),          # F3
        "noisy": noisy,
    }


def run_d1(sd: Dict[str, torch.Tensor], alloc: Dict[str, Any],
           seed: int) -> Dict[str, Any]:
    """D1 - per-group clipping + Mahalanobis-accounted Gaussian (design SS 11).

    Per group g:  v_g <- v_g * min(1, C_g/||v_g||)      (design SS 11.1)
                  noise ~ N(0, sigma_g^2 I_{d_g})       (design SS 11.3)
    Privacy equivalence sum_g (C_g/sigma_g)^2 = 1/sigma_eff^2 is asserted by the
    caller before any result is accepted (design SS 19).
    """
    torch.manual_seed(seed)                       # design SS 16 - same seed as D0

    parts_signal, parts_noisy, noise_sq = [], [], 0.0
    for i, name in enumerate(FROZEN_GROUP_ORDER):
        t = sd[name].detach().cpu().flatten().to(torch.float32)
        n_g = float(torch.norm(t, p=2))
        C = alloc["C_g"][i]
        scale = min(1.0, C / (n_g + 1e-12))
        clipped_g = t * scale
        noise_g = torch.normal(0.0, alloc["sigma_g"][i], size=clipped_g.shape)
        noise_sq += float(torch.sum(noise_g * noise_g))
        parts_signal.append(clipped_g)
        parts_noisy.append(clipped_g + noise_g)

    signal = torch.cat(parts_signal)
    noisy = torch.cat(parts_noisy)
    flat = flatten(sd)
    l2_before = float(torch.norm(flat, p=2))
    l2_after = float(torch.norm(noisy, p=2))

    return {
        "arm": "D1", "seed": seed,
        "l2_before": l2_before,
        "l2_signal_after_clip": float(torch.norm(signal, p=2)),
        "l2_noise": math.sqrt(noise_sq),
        "l2_after": l2_after,
        "snr": l2_after / GLOBAL_CLIP_NORM,       # design SS 13
        "distortion_ratio": l2_after / l2_before, # diagnostic only (design SS 17.2)
        "delta_sigma": alloc["delta_sigma"],
        "epsilon": epsilon_from_mahalanobis(alloc["delta_sigma"]),
        "draw_sha256": tensor_digest(noisy),          # F3
        "noisy": noisy,
    }


def tensor_digest(t: torch.Tensor) -> str:
    """Bitwise SHA-256 of a tensor's raw buffer.

    F3: this is what makes the reproducibility check REAL - two draws at the
    same seed must produce byte-identical tensors, not merely similar norms.
    """
    return sha256_bytes(t.detach().cpu().contiguous().numpy().tobytes())


def unflatten(flat: torch.Tensor, sd: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
    """Rebuild a state_dict in FROZEN_GROUP_ORDER (for the strict=True check)."""
    out, idx = {}, 0
    for name in FROZEN_GROUP_ORDER:
        shape = sd[name].shape
        n = sd[name].numel()
        out[name] = flat[idx:idx + n].view(shape).clone()
        idx += n
    return out


# ==========================================================================
# Cryptographic verification - design SS 14
# ==========================================================================
def crypto_verify(noisy: torch.Tensor, sd: Dict[str, torch.Tensor],
                  arm: str, seed: int, session_id: str,
                  out_dir: str, key_source: str) -> Dict[str, Any]:
    """Exercise the frozen cryptographic stack (design SS 14).

    Layers exercised here: AES-GCM at rest (SecureStore), HMAC-SHA256 receipt +
    verify (CentralReceiptManager), SHA-256 full-hash match, and
    load_state_dict(strict=True) with per-tensor equality.

    Layers belonging to the FEDERATED TRANSPORT subsystem (TPM ECDSA P-256
    signing, chunked GridFS upload with per-chunk SHA-256) are NOT invoked by a
    local mechanism-level run. They are reported as not_exercised with an
    explicit reason - never silently marked as passing, never stubbed
    (design SS 14, SS 25.6).
    """
    result: Dict[str, Any] = {"layers": {}, "ok": True}

    def record(layer: str, status: str, detail: str = "") -> None:
        result["layers"][layer] = {"status": status, "detail": detail}
        if status == "FAIL":
            result["ok"] = False

    # --- serialise + SHA-256 full hash -----------------------------------
    try:
        rebuilt = unflatten(noisy, sd)
        buf = io.BytesIO()
        torch.save(rebuilt, buf)
        payload = buf.getvalue()
        digest = sha256_bytes(payload)
        record("sha256_full_hash", "PASS", digest)
    except Exception as exc:  # noqa: BLE001
        record("sha256_full_hash", "FAIL", f"{type(exc).__name__}: {exc}")
        return result

    # --- AES-GCM at rest via the frozen SecureStore ----------------------
    try:
        from centralized_secure_store import SecureStore
        from pathlib import Path
        store_root = Path(out_dir) / "secure_store"
        store = SecureStore(agent="exp8-dp", root=store_root)
        uri = f"file://{store_root / f'{arm}_seed{seed}.pt.enc'}"
        store.encrypt_write(uri, payload)
        back = store.decrypt_read(uri)
        if sha256_bytes(back) != digest:
            record("aes_gcm_at_rest", "FAIL", "round-trip hash mismatch")
            return result
        record("aes_gcm_at_rest", "PASS", "encrypt/decrypt round-trip hash match")
    except Exception as exc:  # noqa: BLE001
        record("aes_gcm_at_rest", "FAIL", f"{type(exc).__name__}: {exc}")
        return result

    # --- load_state_dict(strict=True) + per-tensor equality --------------
    try:
        loaded = torch.load(io.BytesIO(back), map_location="cpu", weights_only=False)
        model_like = torch.nn.ParameterDict(
            {k.replace(".", "_"): torch.nn.Parameter(torch.zeros_like(v))
             for k, v in rebuilt.items()})
        model_like.load_state_dict(
            {k.replace(".", "_"): v for k, v in loaded.items()}, strict=True)
        per_tensor = all(torch.equal(loaded[k], rebuilt[k]) for k in rebuilt)
        if not per_tensor:
            record("strict_load_and_tensor_equality", "FAIL", "per-tensor mismatch")
            return result
        record("strict_load_and_tensor_equality", "PASS",
               "strict=True load clean; all tensors equal")
    except Exception as exc:  # noqa: BLE001
        record("strict_load_and_tensor_equality", "FAIL", f"{type(exc).__name__}: {exc}")
        return result

    # --- HMAC-SHA256 receipt + verify ------------------------------------
    try:
        from centralised_receipts import CentralReceiptManager
        # F4: explicit Exp-8-local key; never the home-directory default.
        rm = CentralReceiptManager(agent="exp8-dp", key_source=key_source)
        receipt = rm.create_receipt(
            agent="exp8-dp", session_id=session_id, operation="exp8_dp_mechanism",
            params={"arm": arm, "seed": seed, "sha256": digest}, outputs=[uri])
        rdir = os.path.join(out_dir, "receipts")
        os.makedirs(rdir, exist_ok=True)
        rpath = rm.write_receipt(receipt, out_dir=rdir)
        local = rpath[len("file://"):] if str(rpath).startswith("file://") else str(rpath)
        if not rm.verify(local):
            record("hmac_receipt", "FAIL", "HMAC verification returned False")
            return result
        record("hmac_receipt", "PASS", f"verified {os.path.basename(local)}")
        record("hmac_chain", "PASS" if "hmac_chain" in receipt or
               "prev" in json.dumps(receipt) else "NOT_EXERCISED",
               "chain field presence checked on the emitted receipt")
    except Exception as exc:  # noqa: BLE001
        record("hmac_receipt", "FAIL", f"{type(exc).__name__}: {exc}")
        return result

    # --- federated transport layers: honestly reported, never stubbed ----
    record("tpm_ecdsa_p256", "NOT_EXERCISED",
           "federated upload path not invoked by a local mechanism-level run "
           "(design SS 15); reported, not assumed to pass")
    record("gridfs_chunked_sha256", "NOT_EXERCISED",
           "federated upload path not invoked by a local mechanism-level run "
           "(design SS 15); reported, not assumed to pass")
    return result


# ==========================================================================
# Main
# ==========================================================================
def main() -> int:
    ap = argparse.ArgumentParser(
        description="Phase 17 / Experiment 8 - DP mechanism (infrastructure)")
    ap.add_argument("--out-dir", default=os.path.join(
        REPO, "trainer_outputs", "exp8_dp_mechanism"))
    ap.add_argument("--target", default=TARGET_PATH)
    args = ap.parse_args()
    os.chdir(REPO)

    t0 = time.time()
    print("=" * 74)
    print("EXPERIMENT 8 - DP MECHANISM (INFRASTRUCTURE; NOT predictive utility)")
    print("=" * 74)

    # F1 / design SS 19 - pre-run frozen-input gate, IN the execution path.
    frozen_before = snapshot_frozen()
    gate_frozen_before(frozen_before)
    print(f"[OK] frozen inputs verified before execution "
          f"({len(FROZEN_SHA)} pinned, {len(FROZEN_UNPINNED)} snapshotted)")

    sd, groups, target_sha = load_target()
    d = sum(g[1] for g in groups)
    print(f"[OK] target {os.path.basename(args.target)}  d={d}  sha={target_sha[:32]}...")
    for name, d_g, v in groups:
        print(f"      {name:<12s} d_g={d_g:>7d}  ||v_g||={v:.6f}")

    alloc = allocate(groups)
    # design SS 19 - hard aborts on the allocation invariants
    if abs(alloc["sum_C_g_squared"] - 1.0) > EQUIV_TOL:
        abort(f"sum_g C_g^2 = {alloc['sum_C_g_squared']!r} != 1")
    if abs(alloc["delta_sigma_sq"] - 1.0 / D0_SIGMA_EFF ** 2) > EQUIV_TOL:
        abort("privacy-equivalence condition sum_g (C_g/sigma_g)^2 = 1/sigma_eff^2 "
              f"violated: {alloc['delta_sigma_sq']!r}")
    print(f"[OK] allocation: sum C_g^2 = {alloc['sum_C_g_squared']:.15f}")
    print(f"[OK] privacy equivalence: sum (C_g/sigma_g)^2 = "
          f"{alloc['delta_sigma_sq']:.15f}  (required {1.0/D0_SIGMA_EFF**2})")

    eps_d0 = epsilon_from_mahalanobis(D0_CLIP_NORM / (D0_SIGMA_EFF * D0_CLIP_NORM))
    eps_d1 = epsilon_from_mahalanobis(alloc["delta_sigma"])
    for arm, eps in (("D0", eps_d0), ("D1", eps_d1)):
        if eps > EPS_PER_UPDATE_MAX + 1e-12:
            abort(f"{arm}: eps={eps!r} exceeds the frozen budget {EPS_PER_UPDATE_MAX}")
        if eps * N_UPDATES_PER_ROUND > EPS_CUMULATIVE_MAX + 1e-9:
            abort(f"{arm}: cumulative eps exceeds {EPS_CUMULATIVE_MAX}")
    print(f"[OK] eps  D0={eps_d0:.15f}  D1={eps_d1:.15f}  (budget {EPS_PER_UPDATE_MAX})")

    analytical = {
        "D0": D0_SIGMA_EFF * math.sqrt(d) / GLOBAL_CLIP_NORM,
        "D1": alloc["analytical_noise_norm"] / GLOBAL_CLIP_NORM,
    }
    print(f"[OK] analytical SNR  D0={analytical['D0']:.4f}  D1={analytical['D1']:.4f}")

    os.makedirs(args.out_dir, exist_ok=True)
    session_id = f"exp8-{int(time.time())}"
    # F4 - provision the Exp-8-local HMAC key BEFORE any draw, so the CSPRNG
    # call cannot interleave with the seeded experimental noise stream.
    key_source = local_receipt_key_source(args.out_dir)
    print(f"[OK] receipt key: {key_source} (inside the authorised output dir)")
    rows: List[Dict[str, Any]] = []

    # design SS 15, SS 16 - r = 5 paired draws, same seed for both arms
    for seed in SEEDS:
        for arm in ARMS:
            r = run_d0(sd, seed) if arm == "D0" else run_d1(sd, alloc, seed)
            cv = crypto_verify(r.pop("noisy"), sd, arm, seed, session_id,
                               args.out_dir, key_source)
            if not cv["ok"]:
                abort(f"cryptographic verification FAILED for {arm} seed {seed}: "
                      f"{cv['layers']}")
            r["crypto_ok"] = cv["ok"]
            r["crypto_layers"] = json.dumps(cv["layers"], sort_keys=True)
            r["analytical_snr"] = analytical[arm]
            r["analytical_deviation"] = r["snr"] - analytical[arm]
            rows.append(r)
            print(f"  [{arm} seed={seed}] l2_after={r['l2_after']:.4f} "
                  f"SNR={r['snr']:.4f} eps={r['epsilon']:.12f} crypto=OK")

    # ---- F3 (a): identical seeds must reproduce identical draws ------------
    # design SS 17 P4. Re-executes every draw and compares the BITWISE tensor
    # digest against the recorded one. This is an actual measurement, not an
    # assertion, and its result feeds the verdict via exp8_receipt.json.
    print("[STEP] reproducibility re-draw (design SS 17 P4)")
    repro_detail: List[Dict[str, Any]] = []
    for r in rows:
        arm, seed = r["arm"], int(r["seed"])
        again = run_d0(sd, seed) if arm == "D0" else run_d1(sd, alloc, seed)
        same = (again["draw_sha256"] == r["draw_sha256"])
        repro_detail.append({"arm": arm, "seed": seed, "identical": bool(same),
                             "draw_sha256": r["draw_sha256"]})
        if not same:
            abort(f"reproducibility FAILED: {arm} seed {seed} did not reproduce "
                  f"a byte-identical draw")
    repro_ok = all(x["identical"] for x in repro_detail)
    print(f"[{'OK' if repro_ok else 'FAIL'}] {len(repro_detail)}/{len(rows)} draws "
          f"reproduced byte-identically at the same seed")

    cols = ["arm", "seed", "l2_before", "l2_signal_after_clip", "l2_noise",
            "l2_after", "snr", "analytical_snr", "analytical_deviation",
            "distortion_ratio", "delta_sigma", "epsilon", "draw_sha256",
            "crypto_ok", "crypto_layers"]
    mpath = os.path.join(args.out_dir, "exp8_metrics.csv")
    with open(mpath, "w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=cols)
        w.writeheader()
        for r in rows:
            w.writerow({c: r.get(c) for c in cols})

    gate = {
        "gate": "D0 reproduces the frozen baseline; privacy equivalence holds",
        "target_sha256": target_sha, "d": d,
        "group_structure": [{"name": n, "d_g": dg, "l2": v} for n, dg, v in groups],
        "allocation": {k: alloc[k] for k in
                       ("names", "d_g", "C_g", "sigma_g", "sum_C_g_squared",
                        "delta_sigma_sq", "delta_sigma", "analytical_noise_norm")},
        "epsilon": {"D0": eps_d0, "D1": eps_d1,
                    "budget_per_update": EPS_PER_UPDATE_MAX,
                    "cumulative_D0": eps_d0 * N_UPDATES_PER_ROUND,
                    "cumulative_D1": eps_d1 * N_UPDATES_PER_ROUND,
                    "cumulative_budget": EPS_CUMULATIVE_MAX,
                    "delta": DELTA, "composition": COMPOSITION,
                    "eps_max_ceiling_only": EPS_MAX_CEILING},
        "analytical_snr": analytical,
        "seeds": list(SEEDS),
    }
    gpath = os.path.join(args.out_dir, "d0_reference_check.json")
    with open(gpath, "w", encoding="utf-8") as fh:
        json.dump(gate, fh, indent=2, sort_keys=True)

    # ---- F1 / design SS 22: post-run re-verification of every frozen input --
    frozen_after = gate_frozen_after(frozen_before)
    print("[OK] frozen inputs re-verified after execution: unchanged")

    # ---- F2 / design SS 21: the authorised exp8_receipt.json ---------------
    elapsed = round(time.time() - t0, 2)
    receipt = {
        "schema": "exp8_receipt/1",
        "experiment": "Phase 17 / Exp 8 - DP mechanism (INFRASTRUCTURE)",
        "generated_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "session_id": session_id,
        "frozen_input_sha": {
            "pinned": {p: frozen_after[p] for p in FROZEN_SHA},
            "pinned_all_match": True,      # gate_frozen_before/after would have aborted
            "unpinned_snapshot": {p: frozen_after[p] for p in FROZEN_UNPINNED},
            "unchanged_pre_post": True,    # gate_frozen_after would have aborted
        },
        "target_object": {"path": os.path.relpath(args.target, REPO),
                          "sha256": target_sha, "d": d,
                          "groups": {n: dg for n, dg, _ in groups}},
        "parameters": {
            "arms": list(ARMS), "arm_labels": ARM_LABEL,
            "D0": {"mechanism": "gaussian", "clip_norm": D0_CLIP_NORM,
                   "noise_multiplier": D0_NOISE_MULTIPLIER,
                   "sigma_eff": D0_SIGMA_EFF},
            "D1": {"C_g": alloc["C_g"], "sigma_g": alloc["sigma_g"],
                   "sum_C_g_squared": alloc["sum_C_g_squared"],
                   "delta_sigma_sq": alloc["delta_sigma_sq"]},
            "delta": DELTA, "composition": COMPOSITION,
            "epsilon_per_update": {"D0": eps_d0, "D1": eps_d1},
            "epsilon_budget_per_update": EPS_PER_UPDATE_MAX,
            "epsilon_cumulative_budget": EPS_CUMULATIVE_MAX,
            "n_updates_per_round": N_UPDATES_PER_ROUND,
            "eps_max_ceiling_only": EPS_MAX_CEILING,
            "global_clip_norm": GLOBAL_CLIP_NORM,
            "snr_definition": "l2_norm_after / clip_norm",
        },
        "seeds": {"base_seed": BASE_SEED, "n_repetitions": N_REPETITIONS,
                  "seeds": list(SEEDS),
                  "rule": "seed_i = BASE_SEED + i, same seed for D0 and D1"},
        "reproducibility": {"checked": True, "all_draws_identical": bool(repro_ok),
                            "n_draws_rechecked": len(repro_detail),
                            "detail": repro_detail},
        "environment": {"python": sys.version.split()[0], "platform": sys.platform,
                        "torch": torch.__version__},
        "runtime_seconds": elapsed,
        "output_artifact_sha": {
            "exp8_metrics.csv": sha256_file(mpath),
            "d0_reference_check.json": sha256_file(gpath),
        },
        "note": ("Provenance artifact. Infrastructure experiment; no task metric "
                 "and no predictive-utility claim (design SS 1, SS 24)."),
    }
    rpath = os.path.join(args.out_dir, "exp8_receipt.json")
    with open(rpath, "w", encoding="utf-8") as fh:
        json.dump(receipt, fh, indent=2, sort_keys=True, default=float)

    print("-" * 74)
    print(f"[DONE] {len(rows)} draws -> {mpath}   ({elapsed:.1f}s)")
    print(f"[DONE] receipt -> {rpath}")
    print("Next: aggregate_exp8.py (paired analysis, T3 + k_min), then verify_exp8.py")
    return 0


if __name__ == "__main__":
    sys.exit(main())
