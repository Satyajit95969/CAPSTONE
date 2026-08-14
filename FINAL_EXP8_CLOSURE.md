# FINAL EXPERIMENT 8 CLOSURE

**Status: CLOSED.** Verdict recorded by the frozen aggregator, verbatim:

> `H0 / FAIL FOR MATERIAL IMPROVEMENT - not an implementation failure`

Authoritative specification: `PHASE_17_EXP8_DESIGN.md` (FROZEN, unmodified,
SHA-256 `bc7d77d0457811429112967d746c30ddf8a4e9d99bd41220f65cf5d90378c6a7`).

---

## 1. Experiment identification

| Field | Value |
|---|---|
| Experiment | Phase 17 / Experiment 8 — DP mechanism |
| Class | **INFRASTRUCTURE / mechanism-level** |
| Single changed factor | DP path (clipping / noise-allocation strategy) |
| Roadmap row | `PHASE_10_ROADMAP.md:256`; root cause RC-4; backlog item B-2 |
| Executed | 2026-08-11, runtime 3.15 s |
| Environment | Python 3.11.9, torch 2.8.0+cpu, win32 |

## 2. Objective

Determine whether the DP path can materially improve post-noise signal
preservation on the Track-A probe **at unchanged privacy budget, unchanged
target dimension, and with cryptographic verification intact.**

## 3. Frozen design summary

| Item | Value |
|---|---|
| Target | `trainer_outputs/local_probe_base.pt`, **d = 295,681**, √d = 543.77 |
| Groups | `fc1.weight` 294,912 · `fc1.bias` 384 · `fc2.weight` 384 · `fc2.bias` 1 |
| Arms | **D0** and **D1** only |
| D0 | Gaussian, `clip_norm` 1.0, `noise_multiplier` 1.0, σ_eff 1.0 |
| D1 | Per-group clipping + Mahalanobis accounting; `C_g ∝ ‖v_g‖₂`, `Σ C_g² = 1`; σ_g from the frozen closed form `min‖n‖ = σ_eff·Σ_g C_g√d_g` |
| Privacy | δ = 1e-05; ε ≤ **5.302585092994046**/update; cumulative ≤ **15.9078**; RDP single-composition, T = 1; `eps_max` 100 = ceiling only |
| SNR | `l2_norm_after / clip_norm` — lower is better |
| Protocol | r = **5** paired draws; seeds **1001–1005**; same seed for D0 and D1; **no participant CV** |
| Acceptance | **T3 + k_min = 1.01**; paired 95 % CI, df = 4, t_crit = 2.7764451051977987 |
| Boundary | **No dimensionality reduction** — that lever belongs to Exp 9 |

## 4. Actual D0 results

| seed | l2_before | signal after clip | l2_after | **SNR** | ε |
|---|---|---|---|---|---|
| 1001 | 11.3448 | 1.000000 | 544.6531 | **544.653137** | 5.302585092994046 |
| 1002 | 11.3448 | 1.000000 | 544.4047 | **544.404724** | 5.302585092994046 |
| 1003 | 11.3448 | 1.000000 | 543.3787 | **543.378723** | 5.302585092994046 |
| 1004 | 11.3448 | 1.000000 | 544.3627 | **544.362671** | 5.302585092994046 |
| 1005 | 11.3448 | 1.000000 | 544.9098 | **544.909790** | 5.302585092994046 |

mean **544.341809** · SD 0.581261 · min 543.378723 · max 544.909790

## 5. Actual D1 results

| seed | l2_before | signal after clip | l2_after | **SNR** | ε |
|---|---|---|---|---|---|
| 1001 | 11.3448 | 1.000000 | 544.5829 | **544.582886** | 5.302585092994046 |
| 1002 | 11.3448 | 1.000000 | 544.3542 | **544.354248** | 5.302585092994046 |
| 1003 | 11.3448 | 1.000000 | 543.3168 | **543.316833** | 5.302585092994046 |
| 1004 | 11.3448 | 1.000000 | 544.2942 | **544.294189** | 5.302585092994046 |
| 1005 | 11.3448 | 1.000000 | 544.8611 | **544.861084** | 5.302585092994046 |

mean **544.281848** · SD 0.583692 · min 543.316833 · max 544.861084

## 6. Paired differences (Δᵢ = SNR_D0,ᵢ − SNR_D1,ᵢ)

| seed | Δ |
|---|---|
| 1001 | **+0.070251** |
| 1002 | **+0.050476** |
| 1003 | **+0.061890** |
| 1004 | **+0.068481** |
| 1005 | **+0.048706** |

mean Δ = **+0.059961** · SD 0.009986 · SE 0.004466
**95 % CI = [+0.047562, +0.072360]** (k = 5, df = 4, t_crit = 2.7764451051977987) — excludes zero.

## 7. Relative improvement

**1.000110×** (`1.0001101652346194`) against the frozen **k_min = 1.01**.

## 8. P0–P4 results

| ID | Criterion | Result |
|---|---|---|
| **P0** | Control / integrity hard gate | **PASS** — reference record present; D0 within published band [540, 548] (min 543.3787, max 544.9098); D0 ε exact; identical pre-DP object across all 10 draws; d = 295,681; privacy equivalence Σ(C_g/σ_g)² = 1 |
| **P1** | Privacy-budget preservation | **PASS** — ε = 5.302585092994046 for **both** arms; cumulative 15.9078 ≤ 15.9078; δ = 1e-05; composition unchanged |
| **P2** | Material improvement (T3 + k_min) | **FAIL** — A True, B True, **C False** |
| **P3** | Cryptographic verification | **PASS** |
| **P4** | Reproducibility | **PASS** |

**P2 conditions individually:**

| | Condition | Result |
|---|---|---|
| A | mean(Δ) > 0 | **True** (+0.059961) |
| B | 95 % CI excludes 0 | **True** ([+0.047562, +0.072360]) |
| C | ratio ≥ 1.01 | **False** (1.000110) |

## 9. Analytical prediction vs empirical result

| arm | analytical SNR | empirical mean | deviation | rel. |
|---|---|---|---|---|
| D0 | 543.765574 | 544.341809 | +0.576235 | +0.106 % |
| D1 | 543.707441 | 544.281848 | +0.574407 | +0.106 % |

| | Registered (design §23) | Measured |
|---|---|---|
| Relative improvement | ≈ **1.000107×** | **1.000110×** |

Agreement to **3 ppm**. The uniform +0.106 % offset in both arms is the
`√(‖signal‖² + ‖noise‖²)` term — `l2_after` includes the unit-norm signal
whereas the registered analytical figure is the noise norm alone. It is
identical in both arms and cancels in the paired difference.

## 10. Cryptographic verification status

**PASS** — `crypto_ok = True` for all 10 draws; no layer reported FAIL.

- **Exercised:** `sha256_full_hash`, `aes_gcm_at_rest`,
  `strict_load_and_tensor_equality`, `hmac_receipt`.
- **NOT_EXERCISED** (reported, never marked PASS): `tpm_ecdsa_p256`,
  `gridfs_chunked_sha256`, `hmac_chain` — the federated transport path is not
  invoked by a local mechanism-level run.

## 11. Reproducibility status

**PASS** — **10/10 draws reproduced byte-identically** at the same seed,
compared by SHA-256 of the raw tensor buffer (bitwise, not a tolerance).
Frozen SHAs unchanged pre/post, recorded by the runner.

## 12. Frozen SHA integrity status

**PASS.** Runner pre-run gate passed; runner post-run re-verification reported
unchanged; independent post-pipeline re-check of all **14** frozen artifacts
returned **0 failures**, including `PHASE_17_EXP8_DESIGN.md`,
`dp_agent/dp_agent.py`, `trainer_outputs/local_probe_base.pt` and every
Experiment 7 artifact. `verify_exp8.py`: **26 checks, 0 failures,
fail_closed = True**. Exp 8 wrote only under
`trainer_outputs/exp8_dp_mechanism/`.

**Result artifacts (SHA-256):**

```
fcd06f10680f6dbd74275c5dc598443dbe1c761faae6124ad5539eef2b7a84ca  exp8_metrics.csv
99540734c92717f130d1110f4ecddef23b15c5c964ba47356ec4559b170e05a3  exp8_summary.json
3b74a8526796af2ce3f5b939578c9d35001b8461288d1a6b7a30039b10b062bd  exp8_delta.json
a6e2cb94ad6281d01c38016afbf6117103f3a169968e03195a0b66611581a98c  exp8_receipt.json
d43782886c4549e4856fb7d1ed00fb47a67194fe8fd860a5157e66e649030fad  d0_reference_check.json
dfb1d8cfeb3a33747bfc1aa36a751c2b6f482bcfbbc815caa246d31232b2e00e  exp8_verification.json
```

## 13. Final H0 interpretation

The candidate mechanism produced a **real, statistically unambiguous, and
physically negligible** improvement. All five paired differences were positive
and the 95 % CI excluded zero (conditions A and B met), but the relative
improvement of **1.000110×** falls **≈ 91× short** of the pre-registered
practical floor of 1.01 (condition C not met).

This is exactly the outcome design §23 registered in advance, and exactly the
failure mode the k_min floor was frozen to prevent: a detectable effect being
mis-reported as a material one. `fc1.weight` carries 99.74 % of the dimensions
and 99.81 % of the signal energy, so the allocation `C_g ∝ ‖v_g‖₂` sits
essentially on the Cauchy–Schwarz equality point `C_g ∝ √d_g` at which the
per-group gain vanishes.

**Every integrity and implementation criterion passed (P0, P1, P3, P4).**
H0 is a legitimate scientific result and is **not** an implementation failure.

**Established finding:** at fixed ε and fixed d, a privacy-equivalent mechanism
change alone **cannot materially improve post-noise SNR**. The measurement
matched the closed-form prediction to 3 ppm, which independently validates both
the SNR and the ε implementations.

## 14. Predictive utility

**Experiment 8 did NOT demonstrate predictive utility, and makes no
predictive-utility claim.** No task metric was computed, no model was trained,
and no dataset or fold was read. Exp 8 is an infrastructure / DP-mechanism
experiment and must never be reported as a privacy–utility tradeoff — Exp 7
established that predictive utility has not been demonstrated, and Exp 8 does
not revisit that question.

## 15. Handoff — Exp 9 owns the dimensionality lever

Since ε depends only on σ_eff and SNR = σ_eff·√d, **d is the only remaining
lever**, as `PHASE_10_ROADMAP.md:228` states: *"Since SNR = 1/√d, **d is the
lever**."* Dimensionality reduction was explicitly **out of scope** for Exp 8
(design §8.1) and belongs to **Experiment 9 / item B-3** (compact update
representation, federated payload).

Exp 8's H0 does not block Exp 9; it strengthens its motivation by demonstrating
empirically that no mechanism-level alternative exists at fixed d.

---

**Experiment 8 status: CLOSED.**
The C-condition failure is a scientific result, not an artifact or integrity issue.
