#!/usr/bin/env python3
"""
scripts/fl_optimizers.py

Server-side FL optimization update rules — FedAvg (identity), FedAdam, FedYogi
(Reddi et al. 2020, "Adaptive Federated Optimization"). Extracted from
scripts/run_step14_multiround.py (Step 18) into its own module so
run_step14_multiround.py and scripts/demo_fl_algorithms.py (Step 19) both
import the SAME code — no duplication, no risk of the demo silently drifting
from what was actually measured.

None of this touches Rust, protobuf, gRPC, or the client. These functions
operate on an already-aggregated client delta (the "pseudo-gradient") that
scripts/aggregate_offline.py produces exactly as it always has — see
run_step14_multiround.py's own module docstring for the full offline-
aggregation-plus-GridFS-patch mechanism these update rules plug into.

Hyperparameters are CLAUDE.md's stated values, env-overridable. CLAUDE.md
gives eta_s/beta1/beta2 for FedAdam and eta_s/tau for FedYogi only — it does
not state FedAdam's tau or FedYogi's beta1/beta2. Reddi et al. 2020
conventionally shares one tau and one (beta1, beta2) pair across both
variants, so the missing values below borrow the OTHER algorithm's stated
value rather than inventing an unrelated number — a documented, deliberate
choice, not a silent gap.
"""

from __future__ import annotations

import os

import torch

FEDADAM_ETA_S = float(os.environ.get("FEDADAM_ETA_S", "1e-3"))   # CLAUDE.md
FEDADAM_BETA1 = float(os.environ.get("FEDADAM_BETA1", "0.9"))    # CLAUDE.md
FEDADAM_BETA2 = float(os.environ.get("FEDADAM_BETA2", "0.999"))  # CLAUDE.md
FEDADAM_TAU   = float(os.environ.get("FEDADAM_TAU", "1e-3"))     # NOT in CLAUDE.md — borrowed from FedYogi's stated tau

FEDYOGI_ETA_S = float(os.environ.get("FEDYOGI_ETA_S", "1e-2"))   # CLAUDE.md
FEDYOGI_TAU   = float(os.environ.get("FEDYOGI_TAU", "1e-3"))     # CLAUDE.md
FEDYOGI_BETA1 = float(os.environ.get("FEDYOGI_BETA1", "0.9"))    # NOT in CLAUDE.md — borrowed from FedAdam's stated beta1
FEDYOGI_BETA2 = float(os.environ.get("FEDYOGI_BETA2", "0.999"))  # NOT in CLAUDE.md — borrowed from FedAdam's stated beta2


def init_moments(template: dict):
    """m_0 = 0. v_0 must be seeded separately as tau^2 (Reddi et al. 2020's
    own recommended init — avoids dividing by ~0 on the very first round,
    before v has seen any data); the caller picks FedAdam's or FedYogi's tau."""
    return {k: torch.zeros_like(v.float()) for k, v in template.items()}


def fedavg_step(pseudo_grad: dict):
    """FedAvg's server-side "update rule" is the identity — the aggregated
    client delta IS the update, unchanged. No m, no v, no state. Included
    here (not just inlined as "do nothing" at call sites) so a caller can
    treat all three algorithms uniformly: step_fn(pseudo_grad) -> applied."""
    return {k: v.clone() for k, v in pseudo_grad.items()}


def fedadam_step(m: dict, v: dict, pseudo_grad: dict):
    """One FedAdam update. Mutates and returns (m, v, applied_delta)."""
    applied = {}
    for k, g in pseudo_grad.items():
        g = g.float()
        m[k] = FEDADAM_BETA1 * m[k] + (1 - FEDADAM_BETA1) * g
        v[k] = FEDADAM_BETA2 * v[k] + (1 - FEDADAM_BETA2) * (g ** 2)
        applied[k] = FEDADAM_ETA_S * m[k] / (v[k].sqrt() + FEDADAM_TAU)
    return m, v, applied


def fedyogi_step(m: dict, v: dict, pseudo_grad: dict):
    """One FedYogi update — differs from FedAdam only in the v update rule
    (Reddi et al. 2020, Algorithm 2): v moves toward g^2 by a fixed
    (1-beta2)*g^2 step in the DIRECTION of sign(v - g^2), rather than FedAdam's
    exponential-moving-average blend. Intended to behave better than Adam
    under heavy-tailed/large gradient noise — directly relevant to this
    project's ~260-magnitude DP noise component. Mutates and returns
    (m, v, applied_delta)."""
    applied = {}
    for k, g in pseudo_grad.items():
        g = g.float()
        g2 = g ** 2
        m[k] = FEDYOGI_BETA1 * m[k] + (1 - FEDYOGI_BETA1) * g
        v[k] = v[k] - (1 - FEDYOGI_BETA2) * torch.sign(v[k] - g2) * g2
        applied[k] = FEDYOGI_ETA_S * m[k] / (v[k].sqrt() + FEDYOGI_TAU)
    return m, v, applied


def l2_norm(state: dict) -> float:
    """L2 norm across every tensor in a state dict — the metric used
    throughout this project's reporting (Steps 12-18)."""
    return torch.sqrt(sum((v.float().norm() ** 2) for v in state.values())).item()
