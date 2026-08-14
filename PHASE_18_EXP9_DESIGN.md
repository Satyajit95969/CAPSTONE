# PHASE 18 / EXPERIMENT 9 — COMPACT UPDATE REPRESENTATION
## Experiment Design — **FROZEN PRE-REGISTRATION**

> **Status: APPROVED AND FROZEN.** No parameter, threshold, mask, ordering, seed,
> criterion or scope in this document may be changed after any result is observed.

> **This experiment does NOT use top-k magnitude sparsification.** The selection
> mechanism is **public architecture-priority coordinate selection** — it is
> **not magnitude-based** and never inspects the private delta. See §3–§4.

> **This is an infrastructure / update-representation experiment.** It makes no
> claim of predictive utility beyond what the frozen Baseline-CV supports.

---

## 1. Experiment identity and objective

| Field | Value |
|---|---|
| Phase / Experiment | Phase 18 / Experiment 9 — compact update representation |
| Roadmap row | `PHASE_10_ROADMAP.md:257` |
| Backlog item | **B-3** (`PHASE_10_ROADMAP.md:228`), root cause **RC-4** |
| Single arm-level factor | **k** — the number of coordinates retained |
| Control arm | **K0** (k = d, no reduction) |
| Class | Infrastructure / mechanism-level, with a cross-validated task-signal criterion |

**Objective.** Determine the smallest **publicly-specified** compact representation
of a DAIC-trained model update that (a) collapses the transport payload, (b)
materially improves post-noise SNR at an unchanged privacy budget, and (c)
retains the demonstrated DAIC ranking signal.

## 2. Relationship to roadmap Row 9 and B-3

**Row 9 (verbatim):** *"Compact update representation | federated payload | Find
the smallest DAIC-derived object that survives DP | Exp 6/7 task metrics + Exp 8
SNR | Update dimensionality and payload collapse **while DAIC task metrics
hold**. A compact update that has lost the signal is not a success — it is Track A
with extra steps."*

**B-3 (verbatim):** *"what is the smallest object derived from the DAIC-trained
model that still carries DAIC signal and survives the DP path? Since SNR = 1/√d,
**d is the lever**."*

Exp 9 changes **d** — the lever B-3 names. Row 9's comparator, *"Exp 6/7 task
metrics"*, is why the task criterion is evaluated under the same **5×5 CV**
protocol Exps 6 and 7 used (§19).

## 3. Why true top-k magnitude sparsification was abandoned

Top-k selects `S = argtop-k(|δ|)`, a **deterministic function of the private
DAIC-trained delta**, and releases `S` **in the clear**. The Gaussian mechanism
bounds the privacy loss of the *values* only. For adjacent datasets whose top-k
sets differ, the output distributions have **disjoint support on the index
component**, so the privacy loss on that component is unbounded (ε = ∞) and δ
would have to approach 1.

**Private selection was audited quantitatively and is infeasible at the frozen
budget.** In zCDP terms the approved budget is ρ ≈ 0.5006 (ε = 5.302585092994046
at δ = 1e-05, attained at integer α = 6). The Gaussian **value release alone**
costs ρ = Δ²/(2σ²) = **0.500**. Essentially nothing remains for selection:

- **Peeling / report-noisy-max:** even reallocating half the entire budget to
  selection (ρ_sel = 0.25, which by itself raises σ_eff to √2 and worsens SNR by
  41 %), ρ_per = ρ_sel/k gives ε_per ≈ 1.3 × 10⁻³ at k = 295,681 and
  ≈ 6.8 × 10⁻³ at k = 10,968 — selection indistinguishable from uniform. Meaningful
  selection (ε_per ~ O(1)) affords **k ≈ 1**.
- **One-shot exponential mechanism over k-subsets:** the utility gap scales as
  ≈ 4k^1.5·ln(d/k)/ε_sel against a maximum score ≈ √d, requiring
  **ε_sel ≳ 3.6 × 10⁵** at k = 295,681 — five orders of magnitude over budget.

Raising ε is forbidden by roadmap row 8: *"A utility gain bought by weakening an
already-weak budget is not a gain — the privacy guarantee is the constraint, not
the objective."*

> **Top-k's signal advantage IS its data-dependence, and that is exactly what
> destroys the privacy guarantee. The two cannot be separated.**

## 4. Re-scoped research question — stated plainly

| | Question |
|---|---|
| **Original (B-3)** | *"What is the smallest **DAIC-derived** object that still carries DAIC signal and survives the DP path?"* — permits the compression to be chosen **using the data**. |
| **Experiment 9** | *"What is the smallest **publicly-specified** object derived from the DAIC-trained model that retains DAIC task signal and survives the DP path?"* — the compression is fixed **in advance, without reference to the data**. |

**Experiment 9's question is strictly weaker than B-3's.** Exp 9 does **not**
search for, and cannot identify, the globally optimal data-dependent compressed
object. A negative result here does **not** establish that no compact
DAIC-derived object survives DP — only that the **publicly-specified** family
defined in §6 does not. This distinction must be preserved in every report.

## 5. Frozen privacy parameters

| Parameter | Value | Source |
|---|---|---|
| Mechanism | Gaussian | Exp 8 / P9 §6.2 |
| `clip_norm` C | **1.0** | frozen |
| `noise_multiplier` | **1.0** | frozen |
| σ_eff = nm / clip | **1.0** | frozen |
| δ | **1e-05** | frozen |
| ε per update | **≤ 5.302585092994046** | acceptance constraint |
| Cumulative ε | **≤ 15.9078** per 3-update round | acceptance constraint |
| Composition | **RDP single-composition, T = 1** (output perturbation) | `dp_agent.py:14-48` |
| RDP bound | ε_RDP(α) = α/(2σ_eff²) | `dp_agent.py:43` |
| Conversion | ε(δ) = min over integer α ∈ [2,256] of [ε_RDP(α) + ln(1/δ)/(α−1)] | `dp_agent.py:45` |
| `eps_max` | **100 — ceiling only, NOT the acceptance target** | roadmap `:194` |

ε is computed by the **frozen** `dp_agent._rdp_to_dp`, imported and never
reimplemented or modified.

## 6. Frozen mask definition — public architecture-priority coordinate selection

**Not magnitude-based. Never inspects the delta.**

### 6.1 Priority ordering **O**

A single fixed ordering over all 109,680,132 coordinate positions:

```
1.  fusion.*                     (8 keys,     197,892 params)
2.  bert.pooler.dense            (2 keys,     590,592 params)
3.  bert.encoder.layer.0         (16 keys,  7,087,872 params)
4.  bert.encoder.layer.1         (16 keys,  7,087,872 params)
    …
14. bert.encoder.layer.11        (16 keys,  7,087,872 params)
15. bert.embeddings.*            (5 keys,  23,837,184 params)
```

**Within each structural group:** frozen `state_dict` key order.
**Within each tensor:** deterministic flattened index order (row-major).

### 6.2 The mask

```
S_k  =  the first k coordinate positions under O
```

### 6.3 Prohibited inputs to O — binding

The ordering **must never** use: delta magnitude · delta energy · gradients ·
Fisher information computed from DAIC · validation or test performance · any
private or data-derived statistic whatsoever.

**Justification for O is exclusively architectural**, a priori, and requires no
access to the delta: the `fusion` block is the **task head** — randomly
initialised at construction and therefore the task-specific component that must
be learned from scratch — followed by the pooler, then encoder layers, then
embeddings. This is documented structure (`ARCHITECTURE.md`; P9 §3.1;
`trainer_mentalbert_daic.py:225-239` `FusionHead`).

> **Methodological guardrail:** verifying this ordering empirically against the
> delta's measured per-group energy would make O data-dependent and reintroduce
> the §3 privacy defect at the design level. **It must not be done.**

## 7. Exact state-dict ordering and group sizes

Measured from `mentalbert_delta.pt` (207 keys, 109,680,132 params):

| Rank | Group | Keys | Params | Cumulative |
|---|---|---|---|---|
| 1 | `fusion.*` | 8 | 197,892 | 197,892 |
| 2 | `bert.pooler.dense` | 2 | 590,592 | 788,484 |
| 3–14 | `bert.encoder.layer.{0..11}` | 16 each | 7,087,872 each | 85,842,948 |
| 15 | `bert.embeddings.*` | 5 | 23,837,184 | **109,680,132** |

**`fusion.*` keys in frozen order:** `fusion.fc1.weight` (256,768)=196,608 ·
`fusion.fc1.bias`=256 · `fusion.classifier.weight` (2,256)=512 ·
`fusion.classifier.bias`=2 · `fusion.phq_mu.weight`=256 · `fusion.phq_mu.bias`=1 ·
`fusion.phq_logsigma.weight`=256 · `fusion.phq_logsigma.bias`=1.

**`bert.embeddings.*`:** `word_embeddings` 23,440,896 · `position_embeddings`
393,216 · `token_type_embeddings` 1,536 · `LayerNorm.weight` 768 ·
`LayerNorm.bias` 768.

## 8. Frozen k ladder

| Arm | k | k/d | Where k falls under O |
|---|---|---|---|
| **K0** | **109,680,132** | 100 % | all coordinates (control) |
| **K1** | **1,096,801** | 1.0000 % | fusion + pooler (788,484) + 308,317 into `bert.encoder.layer.0` |
| **K2** | **295,681** | 0.2696 % | fusion (197,892) + 97,789 into `bert.pooler.dense` |
| **K3** | **10,968** | 0.0100 % | entirely within `fusion.fc1.weight` |

K3 lies wholly inside the task head — the most favourable position O affords a
10,968-coordinate budget.

## 9. Nested-mask property

Because every mask is a prefix of the same ordering:

```
S_K3  ⊂  S_K2  ⊂  S_K1  ⊂  S_K0
```

The four arms are a **monotone family**, not four unrelated experiments.
**k is the only arm-level factor**; O and the prefix rule are constants of the
experiment, exactly as the fold manifest and training recipe are.

## 10. Public-mask privacy argument

`S_k` is a deterministic function of the **architecture and k alone**. It is
independent of the private delta, may be published before any data is touched,
and therefore **leaks nothing**.

The release is a Gaussian mechanism applied to k clipped values, with the index
set public. Consequently:

- **ε = 5.302585092994046 remains valid, unchanged**
- **δ = 1e-05 unchanged**
- **T = 1 unchanged** — no selection step consumes budget
- The `dp_agent` accounting applies **verbatim**, with Δ = clip = 1.0, σ = 1.0

**Because the mask is public, indices need not be transmitted** — the receiver
reconstructs `S_k` from O and k.

## 11. DP mechanism

Applied to the **k-dimensional** vector, never to the dense reconstruction:

```
δ_k   = δ restricted to S_k                       (k coordinates)
δ̃_k   = δ_k · min(1, C / ‖δ_k‖₂)                  C = 1.0
ỹ     = δ̃_k + n,   n ~ N(0, σ² I_k),  σ = σ_eff · C = 1.0
```

> **Critical:** noising the dense 109,680,132-dimensional reconstruction would
> give SNR = √d regardless of k, nullifying the experiment. The release is over
> the **k retained coordinates only**.

**Reconstruction:** the receiver places `ỹ` at positions `S_k` and zeros
elsewhere, producing a full **207-key** state dict, so `load_state_dict(strict=True)`
and the entire cryptographic pipeline operate unchanged.

## 12. Clipping and noise definitions

| Quantity | Definition |
|---|---|
| Clipping norm | C = 1.0, applied to the k-vector |
| Sensitivity | Δ = C = 1.0 (L2 sensitivity of the clipped release) — **not a hardcoded constant** |
| Noise scale | σ = σ_eff · C = 1.0, isotropic over ℝᵏ |
| ε | via frozen `_rdp_to_dp` at σ_eff = 1.0, δ = 1e-05 |

## 13. Analytical SNR predictions — registered in advance

**Analytical only. Never an acceptance input.**

```
SNR_analytical(k) = σ_eff · √k        (σ_eff = 1.0)
```

| Arm | k | Analytical SNR |
|---|---|---|
| K0 | 109,680,132 | **10,472.83** |
| K1 | 1,096,801 | **1,047.28** |
| K2 | 295,681 | **543.7656** |
| K3 | 10,968 | **104.73** |

K2's value coincides with Exp 8's D0 analytical SNR by construction — K2 is the
probe-dimension anchor.

## 14. Empirical SNR measurement

```
SNR_empirical = l2_norm_after / clip_norm
```

`l2_norm_after` is measured on the noised k-vector; `clip_norm` = 1.0.
Measured on the **frozen `mentalbert_delta.pt`** (§21.1), over r = 5 seeded draws
per arm. Because clipping normalises ‖signal‖ to 1 regardless of contents, SNR
depends only on (k, σ_eff) — so the DP/SNR half legitimately uses the single
frozen artifact.

Reported per arm: mean, SD, min, max, and **empirical-vs-analytical deviation**.
A material mismatch against §13 is an implementation fault, not a discovery.

## 15. G3 acceptance criterion

```
G3:  measured mean SNR  <  544.341809
```

**Basis:** Exp 8's measured D0 mean SNR — the *"Exp 8 SNR"* comparator row 9
names. Lower is better. No invented multiplier.

**Exp 8's k_min = 1.01 is explicitly NOT reused.** A ratio criterion is vacuous
here: SNR = √k makes the improvement analytically guaranteed, and every rung
would exceed 1.01× by two to four orders of magnitude.

## 16. Complete transport payload — definition

**G2 is measured on the complete transport payload**, defined as the exact bytes
covered by `payload_hash` — i.e. **all uploaded bytes** (`ARCHITECTURE.md:343`,
`:494`, `:817`).

**Frozen measurement procedure, identical for every arm:**

```
compact object (post-DP)
  → torch.save serialisation
  → AES-GCM encryption (SecureStore)
  → chunked GridFS upload representation (1 MB chunks)
  → payload_hash = SHA-256 of ALL uploaded bytes
  → G2 quantity = total uploaded byte count
```

**Raw 4k float bytes must NOT be substituted.** The serialisation, encryption,
chunking and counting procedure is frozen so the same definition applies to all
four arms.

## 17. G2 threshold

```
G2:  complete transport payload  ≤  1,579,963 bytes
```

**Basis:** the measured per-upload payload of the completed baseline federation
round (P9 §7: *"Per-upload payload | 1,579,963 B streamed in 2 chunks"*) — the
only federation round the project ever completed end-to-end. An existing measured
project constant; nothing invented.

**Pre-declared disclosure.** The baseline expansion factor was
1,579,963 / 1,185,335 ≈ **1.333**. Under a public mask the raw content is 4k
bytes, so predicted transport payloads are approximately:

| Arm | raw 4k | predicted transport (~1.333×) | vs 1,579,963 B |
|---|---|---|---|
| K0 | 438,789,406 (dense) | ≈ 585 MB | **far over** |
| K1 | 4,387,204 | ≈ 5.85 MB | **over** |
| K2 | 1,182,724 | ≈ 1.577 MB | **marginal — near the threshold** |
| K3 | 43,872 | ≈ 58.5 KB | **under** |

> **K2 is expected to sit at the threshold. This is a pre-declared, structural
> consequence of the ladder and the transport encoding — K2 was chosen as the
> probe-dimension anchor, so any probe-derived payload threshold is marginal at
> K2 by construction. It is NOT a post-hoc adjustment, and the threshold will not
> be moved whichever side K2 lands on.**

## 18. Serialisation / encryption / chunking path

Exactly the existing verified stack, unmodified: `torch.save` → **AES-GCM** at
rest (`SecureStore.encrypt_write`) → chunked GridFS (1 MB) with **per-chunk
SHA-256** → **TPM ECDSA P-256** receipt → **HMAC-SHA256** receipt chaining →
download validation by **full-hash match**, `load_state_dict(strict=True)` and
per-tensor equality (P9 §13 line 220).

Layers not invoked by a local run must be reported **NOT_EXERCISED**, never
marked PASS, and never stubbed.

## 19. 5×5 CV task evaluation

Task metrics are evaluated under the **same protocol Exps 6 and 7 used**, because
row 9's comparator is *"Exp 6/7 task metrics."*

- 5 folds × 5 repeats = **25 fold-runs per arm**
- Repeats averaged **within** fold; fold-level Student-t, **df = 4**,
  **t_crit = 2.7764451051977987**
- Metric: **ROC-AUC** — the only Baseline-CV metric whose 95 % CI excludes chance

## 20. Fold manifest and frozen training recipe

**Fold manifest** — `trainer_outputs/baseline_cv/fold_manifest.json`,
SHA `b9a7a91f1bdd43c78d3cd1ee0c36e4ce3fb8be4b0971dcff3ff62ee5af8fca2f`:

| Field | Value |
|---|---|
| Protocol | participant-level stratified 5-fold |
| Generator | `sklearn.StratifiedKFold(n_splits=5, shuffle=True, random_state=42)` |
| Stratified on | PHQ_binary (PHQ > 10) |
| Population | 188 participants, 45 pos / 143 neg |
| Folds | train/test = 150/38, 150/38, 150/38, 151/37, 151/37; **zero overlap** |

**Training recipe (frozen, identical for every arm):** EPOCHS 3 · LR 2e-05 ·
BATCH_SIZE 8 · MAX_LEN 128 · λ 0.5 · GRAD_CLIP 1.0 ·
`transformers.optimization.AdamW` · `mental/mental-bert-base-uncased` ·
BINARIZE_THRESHOLD 10.0.

**Seeding:** `seed = 1000 + repeat`; per fold `torch.manual_seed(seed*100 + fold)`
immediately before model construction; order **seed → model → dataset →
dataloader → loop** is normative.

## 21. Per-fold delta construction and object split

### 21.1 Two objects, two halves — deliberate and disjoint

| Half | Object |
|---|---|
| **Payload + DP/SNR** | the frozen **`mentalbert_delta.pt`** (d = 109,680,132, 207 keys) |
| **Task signal (ROC-AUC)** | the **25 per-fold deltas** |

This split is required: SNR is delta-independent after clipping (§14), whereas
task metrics must be CV-comparable to the Baseline-CV comparator.

### 21.2 Per-fold procedure

```
for each of the 25 (repeat, fold) runs:
    train under the frozen recipe        → trained_state
    delta_i = trained_state − base_state
    for each arm k:
        apply S_k, reconstruct, evaluate ROC-AUC on that fold's test set
```

Baseline-CV never persisted models or deltas (`run_baseline_cv.py` contains no
`torch.save`), so the 25 trainings must be re-run to capture deltas. **Capturing
the delta is a read-out; the recipe is unchanged.**

**Deltas must be compressed in-flight and must NOT be persisted at full size** —
25 × 438 MB ≈ 11 GB is unnecessary. Only the compressed objects and metrics are
retained.

## 22. K0 equivalence gate — MANDATORY

**K0 must be demonstrated to reproduce Baseline-CV. It must never be asserted.**

```
K0 gate: K0's fold-level ROC-AUC mean must lie inside
         [0.5755177227457472, 0.6911434704671701]
```

Reference: Baseline-CV ROC-AUC mean **0.6333305966064586**.

**Precedent, not proof:** Experiment 7's A0 arm re-ran this exact recipe and
reproduced Baseline-CV's mean on all twelve primary metrics to every recorded
digit. That establishes the recipe is reproducible; it does **not** discharge the
gate, which must be evaluated on this run's own K0 output.

> **If K0 fails the gate: ABORT the task-signal interpretation. No compressed-arm
> task acceptance may be reported.** (A0/D0/L0/W0/E0 precedent.)

## 23. ROC-AUC CI criterion (B8 part 1)

```
Arm's fold-level ROC-AUC mean ∈ [0.5755177227457472, 0.6911434704671701]
```

The frozen Baseline-CV published 95 % CI. Pre-existing; not invented.

## 24. ROC-AUC MDE degradation criterion (B8 part 2)

```
Paired per-fold degradation vs K0 must not exceed
0.05781287386071144  in the adverse direction
```

**Basis:** `baseline_cv_summary.json` → `MDE.ROC_AUC.mde_95` =
0.05781287386071144 (paired_se 0.02082262449651161, t_crit 2.7764451051977987).
A pre-existing frozen project constant, published before Exp 9 existed, and the
same convention Exp 5 (MAE MDE 0.4066) and Exp 7 (PR-AUC MDE 0.0950) used for
*"must not materially degrade."*

Paired **within fold**: fold i's arm against fold i's K0.

## 25. B8 conjunction logic and overall acceptance

**Per-arm task acceptance (C) requires BOTH §23 AND §24.**

**Per-arm overall acceptance requires ALL THREE:**

| | Criterion |
|---|---|
| **A** | complete transport payload ≤ 1,579,963 B (§17) |
| **B** | measured mean SNR < 544.341809 (§15) |
| **C** | ROC-AUC inside CI **and** paired degradation ≤ MDE (§23 + §24) |

**Experiment-level result: the set of passing arms. The smallest passing k is the
answer to the §4 question.** All three criteria are reported **separately for
every arm regardless of the conjunction**, because a partial pattern — e.g.
*"K3 collapsed payload and SNR but lost the ranking signal"* — is precisely the
boundary measurement Exp 9 exists to produce.

**If no arm passes, that is H₀ — a legitimate scientific result, not an
implementation failure.**

## 26. Seed and repetition structure

**DP/SNR half** (explicitly re-approved for Exp 9, not inherited):

| Field | Value |
|---|---|
| r | **5** |
| BASE_SEED | **1000** |
| Seeds | **1001, 1002, 1003, 1004, 1005** |
| Same seeds across arms | Yes |
| Statistics | **Descriptive only** — mean, SD, min, max, deviation vs analytical |
| Inferential test | **None.** Arms differ by orders of magnitude; concentration is ≈1/√(2k) (0.13 % at K2, 0.68 % at K3). Acceptance is the absolute G3 threshold |

**Task half:** the §19/§20 CV convention. The two systems share **no seeds, no
statistics and no acceptance logic**.

## 27. Terminal vs Colab execution split

| Terminal (CPU) | Colab (GPU) |
|---|---|
| Load frozen `mentalbert_delta.pt` | 25 fold trainings (frozen recipe) |
| Construct ordering O, masks S_k | Per-fold delta capture |
| Compression + **transport payload measurement** | **Compression at each k, in-flight** |
| DP draws (4 arms × 5 seeds) | Reconstruction + ROC-AUC evaluation (4 × 25 = 100 inference runs) |
| Cryptographic verification | Returns metrics + compressed objects only |
| Aggregation, verification, receipts | |

Compression for the **task** half must occur on Colab in-flight to avoid moving
~11 GB of full deltas between environments.

**Estimates from measured project evidence:** Terminal ~20–45 min; Colab ~15–20 min
(25 trainings at Exp 7's measured 18 s/fold-run ≈ 8 min, plus inference).

## 28. Preflight checks

Before any draw or training:

1. Frozen-input SHA snapshot and gate (pre-run), re-verified post-run
2. `d == 109,680,132` and 207 keys on the target
3. Group sizes match §7 exactly
4. `S_k` nesting verified: `S_K3 ⊂ S_K2 ⊂ S_K1 ⊂ S_K0`
5. Mask construction demonstrably independent of delta values
6. ε recomputed = 5.302585092994046 for every arm; cumulative ≤ 15.9078
7. Fold manifest SHA = `b9a7a91f…`; 5 folds; zero train/test overlap
8. Transport payload procedure identical across arms
9. Cryptographic stack executes; NOT_EXERCISED layers reported explicitly
10. Reproducibility: identical seeds reproduce byte-identical draws

## 29. Failure and abort conditions

| Condition | Action |
|---|---|
| K0 fails the equivalence gate (§22) | **HARD ABORT** of task interpretation |
| Any frozen SHA changes during the run | **HARD ABORT** |
| Recomputed ε ≠ declared ε for any arm | **HARD ABORT** |
| `d ≠ 109,680,132` or group sizes differ | **HARD ABORT** |
| Mask not nested, or mask differs by arm beyond k | **HARD ABORT** (one-factor violated) |
| Cryptographic verification fails | **HARD ABORT** |
| A criterion met by raising ε, δ, or weakening composition | Arm **rejected**, not reported as success |
| No arm satisfies A ∧ B ∧ C | **H₀** — legitimate result |

## 30. Reporting tables (required)

1. **Per-arm payload:** k, raw bytes, serialised, encrypted, chunked, total uploaded bytes, vs 1,579,963 B
2. **Per-arm SNR:** per-seed values, mean, SD, min, max, analytical, deviation
3. **Per-arm task:** 25 fold-runs, fold-level mean, 95 % CI, CI-containment, paired-vs-K0 delta, MDE comparison
4. **A/B/C matrix** per arm, plus the conjunction
5. **Privacy table:** ε, cumulative ε, δ, composition per arm
6. **Cryptographic table:** per layer, per arm, incl. NOT_EXERCISED
7. **Frozen SHA table:** pre-run and post-run

## 31. Reproducibility requirements

- Ordering O, masks, k values, seeds, thresholds all fixed in this document
- Identical seeds must reproduce byte-identical DP draws (verified, not asserted)
- Frozen SHA gate before and after execution, inside the execution path
- All artifacts written only under `trainer_outputs/exp9_compact_update/`
- `dp_agent/dp_agent.py`, `trainer_mentalbert_daic.py`, both parquets, the fold
  manifest, and every Exp 3–8 artifact remain **frozen and untouched**

## 32. No post-hoc changes — binding

1. **O, the mask family, and k values are fixed.** No re-ordering, no re-pooling,
   no k adjustment after any result.
2. **G2 = 1,579,963 B and G3 = 544.341809 are fixed**, including if K2 lands
   marginally on either side of G2.
3. **§23 and §24 thresholds are fixed** and are pre-existing project constants.
4. **No additional arm** may be run and selected among; any further arm must be
   pre-registered before execution.
5. **ε, δ, composition may never be weakened** to obtain a better result.
6. **The mask may never be constructed from, or validated against, delta
   magnitude, task results, or any private statistic.**
7. **H₀ must be reported as a legitimate scientific result.**
8. **The mechanism must never be described as "top-k."**

## 33. Known limitations

1. **The mask is signal-blind.** A prefix of O captures whatever energy happens to
   lie in the highest-priority groups; it is not chosen to maximise retained
   signal. Expected retention is materially worse than data-dependent selection
   would achieve.
2. **The ordering O is architecturally motivated but not project-prescribed.** No
   source specifies a priority ordering; O is an approved design decision.
3. **K2 is marginal against G2 by construction** (§17).
4. **Predictive utility is not established.** Exps 3–6 returned H₀ and Exp 7's C4
   failed. The ROC-AUC anchor is a *ranking* signal whose CI excludes chance — it
   is **not** demonstrated classification performance.
5. **Task and DP/SNR halves use different objects** (§21.1) — justified, but they
   are not a single measurement on a single artifact.
6. **The 25 trainings are re-runs**, so K0 equivalence is a genuine empirical gate
   that could fail.
7. **Federated transport layers** (TPM ECDSA, GridFS) are not exercised by a local
   run and are reported NOT_EXERCISED.

## 34. What this experiment does NOT do

> **Experiment 9 does not solve the original data-dependent optimal-compression
> question.** It cannot identify the smallest DAIC-derived object, because the
> only mechanisms capable of doing so are data-dependent and therefore invalid at
> the frozen privacy budget (§3).
>
> A negative result establishes only that **the publicly-specified family defined
> by O fails** — not that no compact DAIC-derived object survives DP. A positive
> result establishes that **a publicly-specified compact object suffices**, which
> is a stronger operational claim but a narrower scientific one than B-3 posed.
>
> Exp 9 makes **no predictive-utility claim** beyond preservation of the frozen
> Baseline-CV ranking signal.

---

**Pre-registration status: FROZEN AND APPROVED.**
**Implementation has NOT begun and is NOT authorised by this document.**
