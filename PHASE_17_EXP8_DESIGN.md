# PHASE 17 / EXPERIMENT 8 — DP MECHANISM
## Experiment Design — **FROZEN PRE-REGISTRATION**

> **Status: APPROVED AND FROZEN.** All design decisions are settled. No parameter,
> threshold, allocation, seed or criterion in this document may be changed after
> any result is observed.

> **This is a DP-mechanism / update-signal-preservation (INFRASTRUCTURE)
> experiment.** It is **NOT** a predictive-utility experiment and **NOT** a
> privacy–utility tradeoff experiment. It must never claim that predictive
> utility has been preserved. Experiment 7 is CLOSED and established that
> predictive utility has **not** been demonstrated; Exp 8 does not revisit that.

---

## 1. Experiment identification

| Field | Value | Source |
|---|---|---|
| Phase / Experiment | Phase 17 / Experiment 8 — DP mechanism | `PHASE_10_ROADMAP.md:256` |
| Single changed factor | **DP path** (clipping / noise-allocation strategy) | roadmap row 8 |
| Objective (roadmap wording) | "Make privacy survivable" | roadmap row 8 |
| Comparator | **P9 §6.2 + §8** | roadmap row 8 |
| Root cause | **RC-4** | `PHASE_10_ROADMAP.md:193` |
| Backlog item | **B-2 — Dimension-aware DP mechanism** | `PHASE_10_ROADMAP.md:227` |
| Experiment class | **Infrastructure / mechanism-level** | approved |

### 1.1 Frozen decisions

| # | Decision | Status |
|---|---|---|
| 1 | Target = Track-A probe, d = 295,681, √d = 543.77. Not MentalBERT. Not merged with Exp 9 | **FROZEN** |
| 2 | Control = D0 (Gaussian, clip 1.0, nm 1.0, δ 1e-05) | **FROZEN** |
| 3 | Mechanism-level; **no 5×5 participant CV**; no predictive-performance claim | **FROZEN** |
| 4 | Single factor = DP mechanism/path | **FROZEN** |
| 5 | Candidate = per-group clipping + Mahalanobis accounting, `C_g ∝ ‖v_g‖₂` | **FROZEN** |
| 6 | 47.91 = diagnostic only, **never** a criterion | **FROZEN** |
| 7 | Material improvement = **T3** (paired significance) **+ k_min = 1.01** | **FROZEN** |
| 8 | Repetitions **r = 5** | **FROZEN** |
| 9 | Seeds **1001–1005**, paired across arms, `BASE_SEED = 1000` | **FROZEN** |
| 10 | Paired 95 % CI + practical floor; analytical expectation primary | **FROZEN** |
| 11 | Cryptographic stack preserved exactly | **FROZEN** |
| 12 | Proceed as infrastructure experiment | **FROZEN** |
| 13 | **d = 295,681 fixed** — no dimensionality reduction (Exp 9 boundary) | **FROZEN** |

## 2. Motivation

`PHASE_10_ROADMAP.md:193` (RC-4, Critical):

> "mechanism `none` → L2-after **1.00** (the clip alone); `gaussian` at nm 1.0 →
> L2-after **543.49**; and 543² ≈ the probe's ~296k parameters. Clipping is
> **dimension-independent** while noise scales as **√d**, so SNR = 1/√d. **This
> is not an unfavourable tradeoff — it is the absence of one.**"

## 3. Objective

Determine whether the DP path can materially improve post-noise signal
preservation on the Track-A probe **at unchanged privacy budget, unchanged target
dimension, and with cryptographic verification intact.**

Specifically: (a) validate the DP accounting; (b) validate the SNR
implementation; (c) test the pre-registered per-group mechanism; (d) empirically
evaluate the analytical √d limitation; (e) establish whether mechanism change
alone can materially improve SNR at fixed ε and fixed d.

## 4. Scientific question

> Can the DP mechanism achieve materially better post-noise SNR than the ~543 : 1
> reference while keeping the declared privacy budget unchanged and preserving
> cryptographic verification?

## 5. Hypothesis (H₁)

> A valid dimension-aware / per-group mechanism preserving the same privacy
> guarantee materially improves post-noise SNR over D0.

## 6. Null hypothesis (H₀)

> A mechanism change that preserves the same privacy budget and target dimension
> does not materially improve post-noise SNR over D0.

**§23 registers H₀ as the analytically predicted outcome.** H₀ is a valid
scientific result and must **not** be reported as an implementation failure.

## 7. Single changed factor

**The DP path** — the clipping/noise-allocation strategy. Nothing else varies.

## 8. Frozen factors

| Frozen | Value / SHA |
|---|---|
| `trainer_mentalbert_daic.py` | `65b1902e…a230b` |
| `daic_records.parquet` | `9a241851…5f00` |
| `dataset_build/daic_records_multimodal.parquet` | `1ac9f53e…7a95` |
| `fold_manifest.json` | `b9a7a91f…fca2f` |
| **All Exp 7 artifacts, `exp7_results.zip`, `FINAL_EXP7_CLOSURE.md`** | CLOSED, untouched |
| `dp_agent/dp_agent.py` | **frozen** — candidates live in a new module |
| Update object `local_probe_base.pt` | identical for every arm and draw |
| Privacy budget (ε, δ, composition) | §12 |
| **Target dimension d = 295,681** | **fixed** — §8.1 |
| Cryptographic verification | §14 |
| Task-level seed rules of Exps 3–7 | untouched, not shared with the DP path |

Exp 8 trains no model and touches no dataset, fold, or task metric.

### 8.1 Exp 9 boundary — **FROZEN**

**d = 295,681 is fixed.** No dimensionality reduction, compression, random
projection, sketching, quantisation, parameter pruning, sparsification, low-rank
factorisation, compact representation, or any other reduction of d is permitted
in Exp 8. All of these belong to **Exp 9 / item B-3**, whose declared factor is
the federated payload. Any result obtained by reducing d is **out of scope and
must be rejected**, not reported.

## 9. Target object — **FROZEN**

`trainer_outputs/local_probe_base.pt` — Track-A 4-tensor linear probe.

| Evidence | Value |
|---|---|
| P9 §6.1 | "The Track-A 4-tensor probe (`local_probe_base.pt`, 1,185,335 B) — **not** the MentalBERT model." |
| Measured d | **295,681** |
| √d | **543.77** |
| Observed L2-after | 543.02, 544.04, 543.75, 543.73, 542.61, 544.67; sweep 543.49 |

Measured group structure (frozen inputs to §11 and §23):

| Group | d_g | √d_g | ‖v_g‖₂ | share of d | share of energy |
|---|---|---|---|---|---|
| `fc1.weight` | 294,912 | 543.06 | 11.3231 | 99.74 % | 99.81 % |
| `fc1.bias` | 384 | 19.60 | 0.4009 | 0.13 % | 0.12 % |
| `fc2.weight` | 384 | 19.60 | 0.5743 | 0.13 % | 0.16 % |
| `fc2.bias` | 1 | 1.00 | 0.0158 | 0.0003 % | 0.003 % |
| **Total** | **295,681** | **543.77** | **11.3448** | | |

‖v‖₂ = 11.3448 reproduces the published `l2_norm_before` exactly.

## 10. Control arm — D0 — **FROZEN**

| Parameter | Value |
|---|---|
| Mechanism | Gaussian |
| `clip_norm` C | 1.0 |
| `noise_multiplier` | 1.0 |
| σ_eff = nm / clip | 1.0 |
| δ | 1e-05 |
| ε per update | 5.302585092994046 |
| Expected L2-after | ≈ 543–544 |

**D0 is a HARD GATE** (A0/L0/W0/E0 precedent). If it does not reproduce the
published band, the harness is faulty and no candidate result may be reported.

## 11. Candidate arm — D1 — **FROZEN**

### 11.1 Explicit definitions (resolving the hardcoded-sensitivity issue)

`dp_agent.add_noise` computes `scale = noise_multiplier * sensitivity` with
**`sensitivity` hardcoded to 1.0** (`dp_agent.py:200`), while clipping normalises
the signal to `clip_norm`. At `clip_norm = 1.0` the two coincide, which is why D0
is internally consistent. **A candidate that changes clipping must not silently
inherit D0's accounting.** All four quantities are therefore defined explicitly:

| Quantity | Definition |
|---|---|
| **Clipping norm** | Per group g: `v_g ← v_g · min(1, C_g/‖v_g‖₂)` ⇒ ‖ṽ_g‖₂ ≤ C_g |
| **Sensitivity** | L2 sensitivity of the *clipped* release; per group Δ_g = C_g — the same convention D0 uses (Δ = clip = 1.0). **Never a hardcoded constant.** |
| **Noise scale** | Per group σ_g, calibrated to Δ_g via §11.3 — **never** to a fixed 1.0 |
| **ε computation** | Mahalanobis sensitivity of the non-isotropic Gaussian, §11.2, through the unmodified `_rdp_to_dp` conversion |

### 11.2 Privacy accounting — derived, not invented

The release is a Gaussian mechanism with **diagonal, non-isotropic** covariance
Σ = diag(σ_g² I_{d_g}). Its RDP guarantee depends on the Mahalanobis sensitivity:

```
Δ_Σ²  =  sup ‖Σ^(-1/2)(f(x) − f(x'))‖²  =  Σ_g ‖δ_g‖²/σ_g²  ≤  Σ_g (C_g/σ_g)²

ε_RDP(α)  =  α · Δ_Σ² / 2
```

**D0 is the one-group special case:** Δ_Σ = C/σ = 1/σ_eff, giving
ε_RDP(α) = α/(2σ_eff²) — **exactly `dp_agent.py:43`.** The derivation reduces to
the frozen implementation, which is what makes it admissible.

**Privacy-equivalence condition (D1 carries the SAME guarantee as D0):**

```
        Σ_g (C_g / σ_g)²  =  (C / σ)²  =  1/σ_eff²        [ = 1 at D0's values ]
```

Because ε is obtained from Δ_Σ by the *same* RDP→(ε,δ) conversion (min over
α ∈ [2,256], `_rdp_to_dp`), satisfying this equality yields a **bit-identical ε**,
not merely a comparable one. δ = 1e-05 and composition (T = 1, output
perturbation) are unchanged. Violation to numerical tolerance is a **HARD ABORT**
(§19).

### 11.3 Optimal allocation — closed form

```
minimise   E‖n‖² = Σ_g σ_g² d_g      subject to   Σ_g (C_g/σ_g)² = 1/σ_eff²

Lagrange:  u_g := C_g/σ_g  ∝  (C_g² d_g)^(1/4)

Optimum:   min ‖n‖₂  =  σ_eff · Σ_g C_g √d_g
```

**D0 check:** one group, C = 1 ⇒ σ_eff·√d = 543.77. ✓

**Cauchy–Schwarz bound.** With Σ_g C_g² = C² = 1:

```
Σ_g C_g √d_g  ≤  √(Σ C_g²)·√(Σ d_g)  =  √d      equality iff  C_g ∝ √d_g
```

So D1 is **never worse** than D0, and strictly better **only** to the extent the
allocation departs from `C_g ∝ √d_g`. This is the sole mathematical opening at
fixed ε and fixed d.

### 11.4 Pre-declared allocation — **FROZEN**

```
C_g  ∝  ‖v_g‖₂        normalised so that   Σ_g C_g² = 1
σ_g  from the §11.3 closed form
```

Sensitivity budget allocated in proportion to measured group signal energy. This
is a **closed-form allocation fixed in advance**, not a tuned parameter.

**ANTI-TUNING CLAUSE (binding).** `C_g` **must not** be re-chosen, re-weighted,
re-parameterised or re-optimised after observing any empirical result, and the
candidate mechanism must not be modified in response to the §23 prediction. Doing
either voids the pre-registration. If a different allocation is ever of interest,
it must be pre-registered as a **separate, additional arm** before execution —
never substituted for this one.

## 12. Privacy parameters — **FROZEN**

| Parameter | Value | Role |
|---|---|---|
| δ | **1e-05** | frozen |
| ε per update | **≤ 5.302585092994046** | acceptance constraint |
| Cumulative ε | **≤ 15.9078** per 3-update round | acceptance constraint |
| Composition | **RDP single-composition, T = 1** (output perturbation) | `dp_agent.py:14-48` |
| RDP bound | ε_RDP(α) = α/(2σ_eff²) | `dp_agent.py:43` |
| Conversion | ε(δ) = min_{α∈[2,256]} [ε_RDP(α) + log(1/δ)/(α−1)] | `dp_agent.py:45` |
| `eps_max` | **100 — ceiling only, NOT the acceptance target** | roadmap `:194` |

Verified: `_rdp_to_dp(1.0, 1.0, 1e-5)` = **5.302585092994046** (machine
precision); ×3 = **15.9078**. ε depends **only** on σ_eff and is **independent of
d** — confirmed for (nm, clip) = (1,1), (2,2), (10,10), (0.5,0.5).

## 13. SNR definition — **FROZEN**

```
SNR  :=  l2_norm_after / clip_norm
```

Roadmap `:31` — "noise of norm 543 — a **543 : 1** noise-to-signal ratio". Both
fields are recorded in the receipt (`dp_agent.py:191, 201`).

As implemented, ‖signal‖ = C after clipping and ‖noise‖ ≈ σ√d, so
**SNR = σ_eff·√d** = 1.0 × 543.77 ≈ the published 543–544. ✓

**Lower is better.** SNR = 1 : 1 is parity between signal and noise.

## 14. Cryptographic verification — **FROZEN, PRESERVE EXACTLY**

| Layer | Implementation |
|---|---|
| Receipt HMAC | HMAC-SHA256, `hmac.compare_digest` (`centralised_receipts.py:67, 70-84`) |
| Receipt chaining | `hmac_chain` present |
| Signature | TPM-backed **ECDSA P-256**, key `FederatedDeviceKey` |
| At rest | **AES-GCM** (`SecureStore.encrypt_write`) |
| Transport | chunked GridFS, **per-chunk SHA-256** |
| Download | **full-hash match**, `load_state_dict(strict=True)`, per-tensor equality |

Re-run and record pass/fail per arm per draw. **No part may be modified, relaxed,
stubbed or bypassed to make the experiment pass.**

## 15. Experimental protocol — **FROZEN**

- Output perturbation, **T = 1**, on a single fixed update vector.
- **No participant CV.** Nothing in the DP source or roadmap ties folds,
  participants or `fold_manifest.json` to the DP path.
- Both arms consume the **identical** pre-DP object (SHA-checked, §21).
- **r = 5** seeded, paired draws per arm.
- Per draw record: `l2_before`, `l2_after`, SNR, ε, crypto pass/fail, seed, arm.

### 15.1 Repetitions — **FROZEN: r = 5**

Exactly five paired repetitions. **No repetitions may be added after results are
observed.** Rationale: aligns with the project's dominant 5-repeat convention and
yields df = 4, matching the established `t_crit` (§20).

## 16. Seeding — **FROZEN**

```
BASE_SEED = 1000
seed_i    = BASE_SEED + i          for  i ∈ {1, 2, 3, 4, 5}
          = 1001, 1002, 1003, 1004, 1005
```

`torch.manual_seed(seed_i)` immediately before each draw, **the same seed_i used
for both D0 and D1** so draws are paired and draw-to-draw variance is removed
from the comparison.

The **task-level** seed rules of Exps 3–7 are untouched and are not shared with
the DP path; the numeric coincidence is deliberate reuse of a project convention,
not a coupling.

## 17. Acceptance criteria — **FROZEN**

Grouped per the approved structure. **No task metric appears anywhere.**

| ID | Group | Criterion |
|---|---|---|
| **P0** | control / integrity | D0 reproduces the published baseline (L2-after in the P9 band, ε = 5.302585092994046 exactly); both arms consume an identical pre-DP object |
| **P1** | privacy-budget preservation | ε ≤ 5.302585092994046/update; cumulative ≤ 15.9078; δ = 1e-05; composition unchanged; privacy-equivalence condition §11.2 satisfied; recomputed ε = declared ε for every arm |
| **P2** | SNR outcome | **T3 + k_min = 1.01** — §17.3 |
| **P3** | cryptographic verification | Full stack (§14) passes for every arm and every draw |
| **P4** | reproducibility | Identical seeds reproduce identical draws; frozen SHAs unchanged pre/post |

Roadmap constraint, carried verbatim:

> "**A utility gain bought by weakening an already-weak budget is not a gain —
> the privacy guarantee is the constraint, not the objective.**"

### 17.1 No pre-existing SNR threshold existed

The project was searched exhaustively. Every prior use of "materially" binds to a
separately-derived anchor — MAE MDE **0.4066** (`PHASE_14_EXP5_DESIGN.md:240`),
prevalence floor **0.2394** (`PHASE_12_EXP4_DESIGN.md:49`), PR-AUC MDE **0.0950**
(Exp 7 C4), "prediction variance materially > 0" (`PHASE_11_BASELINE_CV.md:192`).
**No SNR anchor was ever defined.** §17.3 is therefore an approved new threshold,
not an inherited one.

### 17.2 The 47.91 diagnostic — **NOT an Exp 8 criterion**

| Question | Answer |
|---|---|
| Definition | `distortion = dp_l2_after / dp_l2_before` — `create_dp_comparison.py:815` |
| Arithmetic | 543.49 / 11.3448 = **47.9065** |
| Denominator | the **pre-clip** norm 11.3448 — *not* the post-clip norm 1.0 used by 543 : 1 |
| Role in P9 §8 | a **reported column** in a 30-row sweep table |
| Referenced by roadmap row 8? | **No** — 0 occurrences; row 8 cites only 543 : 1 |
| Classification | **reported diagnostic ratio** |

> **47.91 is NOT an Experiment 8 acceptance criterion, threshold, comparator or
> target.** It remains solely the already-established diagnostic distortion ratio
> `l2_after / l2_before`. It may be *reported* for continuity with P9 §8; it may
> **never** be used to judge P2.
>
> The two ratios have **different denominators** and are related by
> 543 = 47.91 × 11.3448, so treating 47.91 as an improvement multiplier on a
> 543 : 1 baseline would be a **category error**.

### 17.3 Material improvement — **T3 + k_min = 1.01 — FROZEN**

Paired per-seed improvement, for i ∈ {1…5}:

```
Δ_i  =  SNR_D0,i  −  SNR_D1,i               (positive Δ ⇒ D1 has less noise)
```

**P2 requires ALL THREE:**

| | Condition | Meaning |
|---|---|---|
| **A** | `mean(Δ_i) > 0` | mean paired improvement is positive |
| **B** | the paired 95 % CI for `mean(Δ)` **excludes 0** | statistical evidence of improvement |
| **C** | `mean(SNR_D0) / mean(SNR_D1)  ≥  1.01` | **practical floor, k_min = 1.01** |

**Interpretation.** The candidate must demonstrate **at least a 1 % reduction in
the noise-to-signal ratio**, *in addition to* statistical evidence of
improvement. Condition C exists precisely because at d = 295,681 the noise norm
concentrates so tightly (§20) that a physically negligible effect could still
satisfy B alone; without C, a ~0.01 % effect could be mis-reported as material.

## 18. Failure criteria

- P2 reached by **increasing ε or δ, or weakening composition** → arm **rejected**.
- P2 reached by **reducing d** → **out of scope** (§8.1, Exp 9's factor) → rejected.
- Any cryptographic verification failure → arm rejected.
- Any of A, B, C unmet → **H₀ / FAIL FOR MATERIAL IMPROVEMENT.**

> **H₀ is a valid scientific outcome and must NOT be reported as an
> implementation failure.** It is a measured finding about the mechanism.

## 19. Abort conditions

| Condition | Action |
|---|---|
| D0 fails to reproduce the P9 band or ε | **HARD ABORT** |
| Recomputed ε ≠ declared ε for any arm | **HARD ABORT** |
| Privacy-equivalence condition §11.2 unsatisfied to numerical tolerance | **HARD ABORT** |
| Arms consume different pre-DP objects | **HARD ABORT** (one-factor) |
| Any frozen SHA changes during the run | **HARD ABORT** |
| Cryptographic verification cannot be executed | **HARD ABORT** |
| d ≠ 295,681 for any arm | **HARD ABORT** (§8.1) |

## 20. Statistical procedure — **FROZEN**

**No participant CV**, so the Exps 3–7 *fold-level* procedure does not transfer
directly. However, the project **does** have a documented paired-difference
convention, and it applies exactly.

**Documented project convention.** `aggregate_exp7.py::paired_delta` /
`pairwise_delta` form per-unit paired differences and pass them to
`fold_level_ci`, which computes:

```
mean(Δ) ± t_crit · sd(Δ, ddof=1) / √k          with  df = k − 1
```

With **k = r = 5 paired draws ⇒ df = 4**, the applicable critical value is the
project's existing constant:

```
t_crit  =  2.7764451051977987          (verified: scipy.stats.t.ppf(0.975, df=4)
                                        is bit-identical to the project constant)
```

**Formula documented explicitly**, as required, so no foreign statistical
convention is silently imported:

```
Δ_i      = SNR_D0,i − SNR_D1,i ,  i = 1…5
Δ̄        = (1/5) Σ Δ_i
s_Δ      = sqrt( Σ (Δ_i − Δ̄)² / (5 − 1) )            [ddof = 1]
SE       = s_Δ / √5
CI₉₅     = [ Δ̄ − 2.7764451051977987 · SE ,  Δ̄ + 2.7764451051977987 · SE ]
```

**Required reporting, per arm and overall:**

- mean SNR per arm · SD per arm · min/max per arm
- mean paired Δ · SD of paired Δ
- paired 95 % CI
- relative improvement factor `mean(SNR_D0)/mean(SNR_D1)`
- **analytical expected SNR** per arm
- **empirical vs analytical deviation** per arm

**The statistical test is secondary to the analytical expectation.** At
d = 295,681, ‖n‖₂ concentrates with relative SD ≈ 1/√(2d) ≈ **0.130 %**,
consistent with the published spread (543.02–544.67, ≈0.2 %). Repeated draws
therefore serve primarily to **verify the implementation against closed-form
predictions**; a material empirical-vs-analytical mismatch is an implementation
fault, not a discovery.

## 21. Artifacts and SHA requirements

```
trainer_outputs/exp8_dp_mechanism/
    exp8_metrics.csv          arm × draw: seed, l2_before, l2_after, SNR, eps,
                              distortion_ratio (diagnostic), crypto_ok
    exp8_summary.json         aggregates, P0–P4, verdict
    exp8_delta.json           paired Δ per seed, mean, SD, 95% CI, ratio
    exp8_receipt.json         SHAs, params, environment, runtime, seeds
    d0_reference_check.json   the HARD GATE record
```

SHA pins: frozen trainer, both parquets, fold manifest, Baseline-CV and Exp
3/4/5/6 summaries, **Exp 7 summary**, `dp_agent/dp_agent.py`, and the **update
object** `local_probe_base.pt` pre-DP (identical across arms and draws).

## 22. Rollback / integrity

- Writes **only** under `trainer_outputs/exp8_dp_mechanism/`.
- `dp_agent/dp_agent.py` **frozen**; D1 lives in a **new** module (Exp 7
  precedent: fork, never edit).
- No Exp 7 artifact, dataset, trainer or prior runner touched.
- Post-run SHA re-verification of every frozen input.
- MongoDB `federated` state untouched (`PHASE_10_ROADMAP.md:84`).

## 23. Registered analytical prediction — **recorded before any run**

Applying §11.3's closed form to the measured group structure (§9), with
`C_g ∝ ‖v_g‖₂`, Σ C_g² = 1:

| Arm | ‖noise‖₂ | SNR |
|---|---|---|
| **D0** (flat) | 543.7656 | **≈ 543.77 : 1** |
| **D1** (optimal per-group, identical ε) | 543.7074 | **≈ 543.71 : 1** |
| **Expected gain** | | **≈ 1.0001× (1.000107×)** |

**Therefore Exp 8 is expected to FAIL the k_min = 1.01 practical-significance
criterion** — the predicted gain is ≈ 0.0107 %, roughly **94× short** of the
required 1 %. **H₀ is the registered expected outcome.**

**Why the gain vanishes.** `fc1.weight` holds **99.74 % of the dimensions and
99.81 % of the signal energy**. The Cauchy–Schwarz equality condition for zero
gain is `C_g ∝ √d_g`; allocating `C_g ∝ ‖v_g‖₂` on this object lands almost
exactly on it, because the energy is spread across the *large* tensor rather than
concentrated in a small one. Per-group allocation helps only when signal
concentrates in **low-dimensional** groups; in this probe it does not.
Exploiting the bound further would require driving `C_g → 0` on `fc1.weight`,
discarding 99.8 % of the signal — dimensionality reduction, **forbidden by §8.1**.

**Binding consequence.** This prediction is registered *in advance* precisely so
it cannot be used to justify changing the design. Per §11.4, the mechanism and
allocation **must not** be modified in response to it.

**Why the experiment still has value.** It converts an analytical argument into
measured evidence: (a) empirically validates the SNR and ε implementations
against closed-form predictions; (b) produces a rigorous, citable demonstration
that **mechanism change alone cannot escape √d at fixed ε and fixed d** — the
strongest possible motivation for Exp 9, which owns the only remaining lever;
(c) the closed-form bound `min‖n‖ = σ_eff·Σ_g C_g√d_g` and its equality condition
are themselves a contribution.

## 24. Relationship to Exp 7 and future Exp 9 / 10

**Exp 7 (CLOSED, untouched).** Activation demonstrated (audio 2.5518971, vision
0.4041132 from exactly 0.0); **predictive utility NOT demonstrated** (C4 FAIL,
PR-AUC Δ = −0.097949). Exp 8 makes no task claim and does not revisit C4.

**Framing.** Roadmap item B-2 schedules DP work *"after the task-level work
because a privacy mechanism that preserves signal is only useful once there **is**
signal to preserve."* Exps 3–6 each returned **H₀** and Exp 7's C4 failed, so that
precondition is unmet. Under the approved decision, Exp 8 proceeds explicitly as
an **infrastructure** experiment and must **not** be reported as a
privacy–utility tradeoff — there is no utility to trade.

**Exp 9** owns d-reduction, the only lever §23 leaves open. **Exp 10** requires
*"DAIC task metrics hold"* — currently vacuous, and unaffected by Exp 8's outcome.

## 25. Scientific-integrity clauses — **BINDING**

1. The candidate mechanism and `C_g` allocation are **fixed** (§11.4). No
   post-hoc tuning.
2. `k_min = 1.01`, `r = 5`, and seeds 1001–1005 are **fixed**. No adjustment
   after results.
3. **No additional candidate arms** may be run and selected among; any further
   arm must be pre-registered before execution.
4. ε, δ and composition may **never** be weakened to obtain better SNR.
5. **d may never be reduced** (§8.1).
6. Cryptographic verification may never be relaxed to make an arm pass.
7. **H₀ must be reported as a legitimate scientific result**, not as failure.
8. Exp 8 must never claim predictive utility has been preserved.

---

**Pre-registration status: FROZEN AND APPROVED.**
**Implementation has NOT begun and is NOT authorised by this document.**
