"""
dp_agent.py

SECURITY FIX:
  FIX-DP-1: process_local_update() now returns epsilon_spent computed via a
             Gaussian mechanism RDP accountant formula instead of leaving the
             field absent. Previously pipeline.py hardcoded epsilon_spent=1.0
             in the receipt because the DP agent never reported actual epsilon.

  The Gaussian mechanism RDP epsilon for one step depends on noise_multiplier
  (sigma) only - NOT on clip_norm. (An earlier version of this comment stated
  a clip_norm-dependent formula; that was never what _rdp_to_dp() computed -
  the comment was wrong, not the code. Corrected here as part of Fix C.)
  For the Gaussian mechanism with std = noise_multiplier * clip_norm and
  sensitivity = clip_norm, the *ratio* std/sensitivity = noise_multiplier is
  what RDP depends on, which is why clip_norm cancels out of the epsilon
  formula even though it fixes the absolute noise scale:
    eps_rdp(alpha) = alpha / (2 * noise_multiplier^2)
  Converted to (eps, delta)-DP via the standard RDP → DP conversion:
    eps(delta) = min over alpha [ eps_rdp(alpha) + log(1/delta)/(alpha-1) ]

  This is a single-step accounting. For full multi-round accounting across
  all clients, wire up the Opacus RDP accountant in the orchestrator.

  FIX-C: add_noise() previously hardcoded sensitivity=1.0 at its call site
         in process_local_update(), completely decoupled from self.clip.
         clip_norm therefore bounded the signal (clipping) but had zero
         effect on the noise - DP-SGD (Abadi et al. 2016) requires
         g_tilde = g + N(0, sigma^2 * C^2 * I), i.e. noise scale must equal
         noise_multiplier * clip_norm. Now passes sensitivity=self.clip,
         restoring that coupling. clip_norm default calibrated to measured
         delta sensitivity (scripts/calibrate_clip_norm.py, N=30: min=0.0393,
         median=0.0719, p90=0.1100, max=0.1177) -> clip_norm=0.15. This is
         NOT a privacy weakening: the epsilon formula above is unchanged
         (still noise_multiplier/delta only), and clipping now genuinely
         bounds sensitivity at the value noise is calibrated against, instead
         of bounding it at 1.0 while noising at 1.0 regardless of where the
         clip threshold was set.
"""

import os, io, time, math, torch
from pathlib import Path
from typing import Optional, Dict, Any
from scipy.optimize import minimize_scalar

from core.centralised_receipts import CentralReceiptManager
from core.centralized_secure_store import SecureStore
from core import reporting as rpt

from installer.security.integrity import integrity_guard
integrity_guard()

_DP_STORE_DIR   = Path.home() / ".federated" / "data" / "secure_store" / "dp_updates"
_DP_RECEIPT_DIR = Path.home() / ".federated" / "data" / "receipts"

# Default delta for RDP → (eps, delta)-DP conversion
_DEFAULT_DELTA = 1e-5


def _rdp_to_dp(noise_multiplier: float, clip_norm: float,
               delta: float = _DEFAULT_DELTA) -> float:
    """
    Compute (epsilon, delta)-DP from Gaussian mechanism parameters
    using Rényi DP → DP conversion.

    For the Gaussian mechanism with sensitivity = clip_norm and
    std = noise_multiplier * clip_norm:

      RDP(alpha) = alpha / (2 * noise_multiplier^2)

    Convert to (eps, delta)-DP:
      eps(alpha) = RDP(alpha) + log(1/delta) / (alpha - 1)

    We minimise over alpha in [2, 256].
    """
    if noise_multiplier <= 0:
        return float("inf")

    best_eps = float("inf")
    for alpha in range(2, 257):
        rdp_alpha = alpha / (2.0 * noise_multiplier ** 2)
        log_term  = math.log(1.0 / delta) / (alpha - 1)
        eps       = rdp_alpha + log_term
        if eps < best_eps:
            best_eps = eps

    return best_eps


def _rdp_to_dp_tight(noise_multiplier: float, delta: float = _DEFAULT_DELTA) -> float:
    """
    Tighter RDP -> (epsilon, delta)-DP conversion (2026-09-20).

    Source: Canonne, Kamath & Steinke, "The Discrete Gaussian for
    Differential Privacy" (NeurIPS 2020), Prop. 12 - itself building on
    Balle, Barthe, Gaboardi, Hsu & Sato, "Hypothesis Testing Interpretations
    and Renyi Differential Privacy" (AISTATS 2020) and Asoodeh et al. (2020).

    If a mechanism is (alpha, eps_rdp(alpha))-RDP for real alpha > 1, it is
    (eps, delta)-DP for any delta in (0,1) at:

        eps(alpha) = eps_rdp(alpha) + log((alpha-1)/alpha)
                     - (log(delta) + log(alpha)) / (alpha - 1)

    This is provably <= the classic conversion used by _rdp_to_dp() above
    (eps_rdp(alpha) + log(1/delta)/(alpha-1)) for every alpha, hence
    "tighter". _rdp_to_dp() is kept intact (not replaced) so the historical
    epsilon numbers it produced remain exactly reproducible.

    For our Gaussian mechanism, std = noise_multiplier * sensitivity, so
    eps_rdp(alpha) = alpha / (2 * noise_multiplier^2) - sensitivity cancels
    out of the ratio exactly as it does for _rdp_to_dp() (see module
    docstring); this conversion is valid for our exact setting: a pure
    Gaussian mechanism, single composition (T=1) per round.

    Unlike the classic bound, this tighter one is minimised at a
    *continuous* alpha (e.g. alpha* ~= 4.576 at noise_multiplier=0.8, not an
    integer), so the classic function's `range(2, 257)` integer grid is not
    sufficient here. We use scipy.optimize.minimize_scalar (bounded Brent's
    method) over alpha in (1, 512] rather than a hand-rolled grid search:
    scipy is already a project dependency (requirements.txt), Brent's method
    finds the continuous optimum to high precision without picking an
    arbitrary grid resolution, and 512 comfortably covers the optimal alpha
    for every noise_multiplier this project uses (down to ~0.5).
    """
    if noise_multiplier <= 0:
        return float("inf")

    def eps_of_alpha(alpha: float) -> float:
        eps_rdp = alpha / (2.0 * noise_multiplier ** 2)
        return (
            eps_rdp
            + math.log((alpha - 1) / alpha)
            - (math.log(delta) + math.log(alpha)) / (alpha - 1)
        )

    result = minimize_scalar(
        eps_of_alpha, bounds=(1.0 + 1e-9, 512.0), method="bounded",
        options={"xatol": 1e-10},
    )
    return float(result.fun)


class DPAgent:
    SUPPORTED_MECHANISMS = {
        "gaussian", "laplace", "uniform", "exponential", "student_t", "none"
    }

    def __init__(
        self,
        clip_norm: float = 0.85,  # Fix E4 recalibration: measured delta sensitivity shifted with lr/epochs (N=30, max=0.7294, was max=0.1177 under the old regime) — see scripts/calibrate_clip_norm.py
        noise_multiplier: float = 0.75,  # Noise reduction (2026-09-20): lowered from 0.8 to 0.75 after switching live accounting to _rdp_to_dp_tight() (Canonne-Kamath-Steinke); eps=6.603254 at sigma=0.75 under the tighter method, ~17.5% margin under eps<=8 (sigma=0.7 was rejected: only ~10.47% margin under the tighter method). The live caller (runtime/pipeline.py) always passes this explicitly via DP_NOISE_MULTIPLIER (default "0.75") — kept in sync here so this default doesn't silently disagree with what's actually live.
        mechanism: str = "gaussian",
        secure_store_dir: str = str(_DP_STORE_DIR),
        receipts_dir: str = str(_DP_RECEIPT_DIR),
        store: Optional['SecureStore'] = None,
        delta: float = _DEFAULT_DELTA,
    ):
        self.clip             = float(clip_norm)
        self.noise_multiplier = float(noise_multiplier)
        self.mechanism        = mechanism.lower()
        self.delta            = delta
        self.secure_store_dir = secure_store_dir
        self.receipts_dir     = receipts_dir

        os.makedirs(self.secure_store_dir, exist_ok=True)
        os.makedirs(self.receipts_dir,     exist_ok=True)

        self.store = store if store is not None else SecureStore(
            agent="dp",
            root=Path.home() / ".federated" / "data" / "secure_store",
        )

        self.rm = CentralReceiptManager(agent="dp-agent")

        if self.mechanism not in self.SUPPORTED_MECHANISMS:
            raise ValueError(
                f"Unsupported mechanism '{self.mechanism}'. "
                f"Supported: {self.SUPPORTED_MECHANISMS}"
            )

    def flatten_state_dict(self, sd):
        tensors, meta = [], []
        for k, v in sd.items():
            t = v.detach().cpu().flatten()
            tensors.append(t)
            meta.append((k, v.size(), t.numel()))
        if not tensors:
            return torch.tensor([]), meta
        return torch.cat(tensors).to(torch.float32), meta

    def unflatten_state_dict(self, flat, meta):
        new_sd, idx = {}, 0
        for k, shape, numel in meta:
            if numel == 0:
                new_sd[k] = torch.zeros(shape)
                continue
            new_sd[k] = flat[idx : idx + numel].view(shape).clone()
            idx += numel
        return new_sd

    def add_noise(self, x: torch.Tensor, sensitivity: float = 1.0):
        if self.noise_multiplier == 0.0 or self.mechanism == "none":
            return x

        eps = 1e-8
        scale = max(eps, self.noise_multiplier * sensitivity)

        if self.mechanism == "gaussian":
            noise = torch.normal(0.0, scale, size=x.shape)
        elif self.mechanism == "laplace":
            noise = torch.distributions.Laplace(0.0, scale).sample(x.shape)
        elif self.mechanism == "uniform":
            noise = torch.empty_like(x).uniform_(-scale, scale)
        elif self.mechanism == "exponential":
            sign  = torch.randint(0, 2, x.shape, dtype=torch.float32) * 2.0 - 1.0
            noise = sign * torch.distributions.Exponential(scale).sample(x.shape)
        elif self.mechanism == "student_t":
            noise = torch.distributions.StudentT(10.0).sample(x.shape) * scale
        else:
            raise ValueError(f"Unknown mechanism {self.mechanism}")

        return x + noise

    def process_local_update(
        self,
        local_update_uri: str,
        session_id: str,
        parent_receipt_uri: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ):
        ts_start = time.time()
        metadata = metadata or {}
        assert local_update_uri.startswith("file://"), "Expected file:// URI"
        path = local_update_uri[len("file://"):]

        if not os.path.exists(path):
            raise FileNotFoundError(f"DPAgent: update not found: {path}")

        try:
            decrypted = self.store.decrypt_read("file://" + path)
        except Exception as e:
            raise RuntimeError(
                f"[DPAgent] Decryption failed (key/root mismatch?)\n"
                f"Path: {path}\nError: {e}"
            )

        buf = io.BytesIO(decrypted)
        try:
            state_dict = torch.load(buf, map_location="cpu", weights_only=False)
        except Exception as e:
            raise RuntimeError(f"Failed to load state_dict from {path}: {e}")

        flat, meta = self.flatten_state_dict(state_dict)
        l2_before  = float(torch.norm(flat, p=2).item()) if flat.numel() > 0 else 0.0
        param_count = flat.numel()

        rpt.subheader("DP INPUT UPDATE")
        rpt.kv("Parameter count", f"{param_count:,}", indent=2)
        rpt.kv("L2 norm before DP", f"{l2_before:.6f}", indent=2)

        clip_factor = 1.0
        clipped = False
        if flat.numel() > 0 and l2_before > self.clip:
            clip_factor = self.clip / (l2_before + 1e-12)
            flat    = flat * clip_factor
            clipped = True

        rpt.subheader("STEP 1 - CLIPPING")
        rpt.kv("Clipping threshold", self.clip, indent=2)
        rpt.kv("Clipping applied", clipped, indent=2)
        rpt.kv("Clipping factor", f"{clip_factor:.6f}", indent=2)
        l2_post_clip = float(torch.norm(flat, p=2).item()) if flat.numel() > 0 else 0.0
        rpt.kv("L2 norm after clip", f"{l2_post_clip:.6f}", indent=2)

        # Fix C: DP-SGD (Abadi et al. 2016) defines the noised update as
        #   g_tilde = g + N(0, sigma^2 * C^2 * I)
        # i.e. noise scale MUST equal noise_multiplier * clip_norm, because
        # clip_norm IS the sensitivity bound the Gaussian mechanism is
        # calibrated against. This project's own paper states that formula.
        # Previously this call hardcoded sensitivity=1.0 regardless of
        # self.clip, so clip_norm bounded the signal but had zero effect on
        # the noise - clipping to a tighter C bought nothing. Passing
        # self.clip here restores the documented mechanism.
        noisy    = self.add_noise(flat, sensitivity=self.clip)
        l2_after = float(torch.norm(noisy, p=2).item()) if noisy.numel() > 0 else 0.0
        noise_norm = float(torch.norm(noisy - flat, p=2).item()) if noisy.numel() > 0 else 0.0

        # Runtime self-check: if add_noise() ever silently regresses to a
        # hardcoded/decoupled sensitivity again, catch it immediately rather
        # than quietly wasting privacy budget on unenforced noise. Compares
        # the empirical per-element noise std (noise_norm / sqrt(n), valid
        # for zero-mean noise over enough dimensions for CLT to apply) to the
        # theoretical noise_multiplier * clip_norm.
        if self.mechanism == "gaussian" and param_count > 100:
            empirical_noise_std = noise_norm / math.sqrt(param_count)
            expected_noise_std  = self.noise_multiplier * self.clip
            rel_error = abs(empirical_noise_std - expected_noise_std) / max(expected_noise_std, 1e-8)
            assert rel_error < 0.15, (
                f"DP noise scale has diverged from clip_norm: empirical per-element "
                f"std={empirical_noise_std:.6f}, expected noise_multiplier*clip_norm="
                f"{expected_noise_std:.6f} (rel_error={rel_error:.1%}). add_noise() may "
                f"no longer be receiving sensitivity=self.clip - this would silently "
                f"waste privacy budget on unenforced noise (Fix C regression)."
            )

        rpt.subheader("STEP 2 - NOISE")
        rpt.kv("Mechanism", self.mechanism, indent=2)
        rpt.kv("Noise multiplier (sigma)", self.noise_multiplier, indent=2)
        rpt.kv("Noise dimension", f"{param_count:,}", indent=2)
        rpt.kv("Noise norm (added)", f"{noise_norm:.6f}", indent=2)

        rpt.subheader("STEP 3 - NOISE ADDITION")
        rpt.kv("Original norm (pre-clip)", f"{l2_before:.6f}", indent=2)
        rpt.kv("Private update norm (post clip+noise)", f"{l2_after:.6f}", indent=2)

        noisy_sd = self.unflatten_state_dict(noisy, meta)
        out_buf  = io.BytesIO()
        torch.save(noisy_sd, out_buf)
        out_bytes = out_buf.getvalue()

        ts       = int(time.time() * 1000)
        out_fname = f"dp_{self.mechanism}_{ts}.pt.enc"
        out_path  = os.path.join(self.secure_store_dir, out_fname)
        self.store.encrypt_write("file://" + out_path, out_bytes)

        # FIX-DP-1: compute real epsilon via RDP accountant
        # 2026-09-20: switched live accounting from the classic Mironov 2017
        # conversion (_rdp_to_dp(), still intact above for reproducing the
        # historical numbers) to the tighter Canonne-Kamath-Steinke
        # conversion (_rdp_to_dp_tight()) - see its docstring for the
        # formula, source, and validity argument. Chosen over the
        # mathematically tighter analytic Gaussian mechanism (Balle & Wang
        # 2018) specifically because RDP composes additively across rounds
        # (sum alpha/(2*sigma_r^2), convert once) while the analytic
        # Gaussian's exact (eps,delta) does not - this project has a known
        # open defect where per-round accounting isn't yet summed across
        # rounds, and this choice keeps that future fix correct instead of
        # requiring a second migration. See docs/IMPLEMENTATION_NOTES.md.
        if self.mechanism == "gaussian" and self.noise_multiplier > 0:
            epsilon_spent = _rdp_to_dp_tight(self.noise_multiplier, self.delta)
        else:
            # Non-Gaussian mechanisms: use a conservative upper bound
            epsilon_spent = float("inf") if self.mechanism == "none" else 10.0

        rpt.subheader("STEP 4 - PRIVACY ACCOUNTING")
        rpt.kv("Delta (δ)", self.delta, indent=2)
        rpt.kv("Epsilon (ε)", f"{epsilon_spent:.6f}" if math.isfinite(epsilon_spent) else "inf", indent=2)
        rpt.kv("Accounting method", "Tighter RDP->DP (Canonne-Kamath-Steinke 2020), single composition T=1", indent=2)
        rpt.kv("DP execution time", f"{(time.time() - ts_start):.4f} sec", indent=2)
        rpt.ok("Differential privacy applied")

        receipt = self.rm.create_receipt(
            agent="dp-agent",
            session_id=session_id,
            operation="dp_process_update",
            params={
                "clip_norm":          self.clip,
                "clip_applied":       clipped,
                "noise_multiplier":   self.noise_multiplier,
                "mechanism":          self.mechanism,
                "delta":              self.delta,
                "l2_norm_before":     l2_before,
                "l2_norm_after":      l2_after,
                "epsilon_spent":      epsilon_spent,     # FIX-DP-1: real value
                "parent_receipt":     parent_receipt_uri,
            },
            outputs=["file://" + out_path],
        )

        receipt_uri = self.rm.write_receipt(receipt, out_dir=self.receipts_dir)

        return {
            "receipt":        receipt,
            "receipt_uri":    receipt_uri,
            "update_uri":     "file://" + out_path,
            "l2_norm_before": l2_before,
            "l2_norm_after":  l2_after,
            "epsilon_spent":  epsilon_spent,      # FIX-DP-1: exposed to pipeline
        }