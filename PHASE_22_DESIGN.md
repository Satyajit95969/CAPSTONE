# PHASE 22 — END-TO-END MULTIMODAL FEDERATED PIPELINE DEMONSTRATION
## Experiment Design — **FROZEN PRE-REGISTRATION**

> **Status: APPROVED AND FROZEN.** No parameter, threshold, partition, seed, metric,
> architecture or scope in this document may be changed after any result is observed.

> ## 🚨 PHASE 22 IS AN ENGINEERING DEMONSTRATION — NOT ROADMAP ROW 10 / C-1
>
> **B-2 (Exp 8) returned H₀. B-3 returned H₀ twice (Exp 9, B-3B). Row 10 / C-1 requires
> B-2 AND B-3 to succeed and therefore REMAINS BLOCKED.** Phase 22 does not attempt,
> satisfy, or partially satisfy Row 10, and **must never be described as doing so.**
> `PHASE_10_ROADMAP.md` is not modified by this phase.

**Tag legend** — every value carries one:
**[A]** inherited from a frozen project specification · **[B]** existing repository-supported value ·
**[C]** newly approved Phase 22 decision · **[U]** unresolved (none remain).

---

## 1. Identity and objective

| Field | Value | Tag |
|---|---|---|
| Phase | **22** — end-to-end multimodal federated demonstration | **[C]** |
| Type | **Engineering demonstration** (not a roadmap row) | **[C]** |
| Roadmap status | Does **not** address Row 10; Row 10 stays blocked | **[A]** |

**Demonstration flow — what Phase 22 exercises end to end:**

```
DAIC-WOZ → text + audio + vision → multimodal model → 5 local clients
   → local training → genuine delta (w_after − w_before) → differential privacy
   → federated aggregation (trimmed mean) → global model → held-out DAIC evaluation
```

**Objective [C]:** demonstrate that the complete pipeline executes correctly end to end,
and **honestly measure the resulting task performance** — without pre-declaring what that
performance will be.

## 2. Two kinds of success — BINDING distinction

| | Definition |
|---|---|
| **ENGINEERING SUCCESS** | The complete pipeline executes correctly and passes independent verification. |
| **SCIENTIFIC / TASK SUCCESS** | The resulting global model demonstrates meaningful task discrimination. |

> **The first does NOT imply the second.** Phase 22 may achieve engineering success and
> still produce a task-degenerate model. Both outcomes are reported plainly. **[C]**

## 3. Frozen input SHA-256

Verified before **and** after execution. **None of these may be modified.**

| Artifact | SHA-256 | Tag |
|---|---|---|
| `dataset_build/daic_records_multimodal.parquet` | `1ac9f53e6102ec0dbaab84dcfdfa3f2e70f2b4a1867a841ea2b7e3ba24c67a95` | **[B]** |
| `daic_records.parquet` | `9a241851760727779fcaa4d50041b8bdc4406fe6b66ddbbd7105f4ce4fc55f00` | **[A]** |
| `trainer_outputs/baseline_cv/fold_manifest.json` | `b9a7a91f1bdd43c78d3cd1ee0c36e4ce3fb8be4b0971dcff3ff62ee5af8fca2f` | **[A]** |
| `trainer_outputs/baseline_cv/baseline_cv_summary.json` | `f065f5e1ff7f8e5ed4d122a8031c9cc12425802e756697d5512a915674871aac` | **[A]** |
| `trainer_mentalbert_daic.py` | `65b1902e2ba8dd996c6960db1a6595d84fd48500f4d8707cb79688093d7a230b` | **[A]** |
| `dp_agent/dp_agent.py` | `758642fff57695cb970af88789c3b6a17c77f2b01d16303ff96230fce5798732` | **[A]** |
| `server/aggregator_agent/aggregator.py` | `59b4d4838cecfaa49f8341320c4d1fdb55fa3b2b6ca1717e29f273f820535d86` | **[B]** |
| `exp_c2_multiclient/c2_common.py` | `9c9bfd42e0eebc2cb1ee2734343994476edf590d837818602c6ab108dbbc7211` | **[B]** |

**MentalBERT pinned revision `24809aa822c76639760d0d934742d1b42f89942f`** — 6 artifact SHAs
verified as in Phases 20–21. **[A]**

## 4. Data

`dataset_build/daic_records_multimodal.parquet` — **verified in the Phase 22 readiness audit**:

| Property | Verified value | Tag |
|---|---|---|
| Participants | **188**, all unique, 0 duplicates | **[B]** |
| Class balance | **45 positive / 143 negative** at PHQ > 10 | **[A]** |
| Text | 0 nulls, 0 empty strings | **[B]** |
| Audio | matrix **(188, 154)** — `features["audio"]["wav2vec2"]`, all finite, 0 all-zero rows | **[B]** |
| Vision | matrix **(188, 84)** — `features["video"]["densenet"]`, all finite, 0 all-zero rows | **[B]** |
| Coverage | audio min 0.1161 / mean 0.4619; video min 0.7407 / mean 0.9485; **0 at zero coverage** | **[B]** |
| ID alignment | participant IDs **identical in order** to frozen `daic_records.parquet` | **[B]** |
| Label agreement | PHQ elementwise identical to the frozen parquet | **[B]** |

Loaded with the frozen `read_parquet_records` (`trainer_mentalbert_daic.py:68`). **[A]**

## 5. Model — reused unmodified

`MultiModalModel(bert_name, audio_dim=154, vision_dim=84)` — `trainer_mentalbert_daic.py:241`. **[A]**

```
text   → MentalBERT → last_hidden_state[:,0,:]            → 768
audio  → SmallMLP(154 → 128)                              → 128
vision → SmallMLP(84  → 128)                              → 128
              concat → 1024 → FusionHead(1024, 256, 2) → (logits[2], mu, log_sigma)
```

| Field | Value | Tag |
|---|---|---|
| Total parameters | **109,763,494** | **[B]** |
| `state_dict` keys | **215** | **[B]** |
| Dropout | **0.2** (inside `FusionHead`) | **[B]** |

**Reused without modification [A]:** `MultiModalModel` · `FusionHead` · `SmallMLP` ·
`MultiModalDataset` (incl. `_extract_audio_vec:109`, `_extract_video_vec:120`) ·
`collate_batch:162` · `fine_tune_supervised:308` · `run_inference` ·
`compute_state_delta:348` · `read_parquet_records:68`.

**No component of the frozen trainer is reimplemented.**

## 6. Local training recipe — frozen

| Field | Value | Source | Tag |
|---|---|---|---|
| Optimizer | `transformers.AdamW` | `:311` | **[A]** |
| Learning rate | **2e-5** | `:308`, `:509`, Exp 9 §20 | **[A]** |
| Epochs | **3** | `:507`, Exp 9 §20 | **[A]** |
| Batch size | **8** | `:308`, `:508` | **[A]** |
| Shuffle | **True** | `:310` | **[A]** |
| Gradient clipping | **1.0** | `:326` | **[A]** |
| `max_length` | **128** | frozen family | **[A]** |
| Loss | `CrossEntropyLoss(logits, label) + 0.5 * MSELoss(mu, phq)` | `:322–324` | **[A]** |
| Label | `PHQ_binary = phq_score > 10.0` | frozen | **[A]** |

## 7. Federated topology — 5 clients, non-IID A2

| Field | Value | Tag |
|---|---|---|
| Clients | **5** | **[C]** |
| Partition rule | **A2 (non-IID)** — `sort by (phq_score DESC, participant_id ASC)`, fill sequentially | **[B]** |
| Rounds | **1** | **[C]** |

### 7.1 Why the C-2 helper cannot be called directly — **and what is reused**

`exp_c2_multiclient/c2_common.py:275 a2_partition()` hardcodes
`CLIENT_SIZES = (38, 38, 38, 37, 37)`, which sums to **188 — the whole population**.
Phase 22 requires clients to partition **only each fold's training participants**
(150 or 151), so the function cannot be invoked as-is. **[B]**

**Reused:** the deterministic A2 *rule* — sort by `(phq_score DESC, participant_id ASC)`,
fill clients sequentially, ties broken by `participant_id` ascending. **No new
partitioning strategy is invented.** **[C]**

**Newly specified:** the per-fold client sizes, by a deterministic even split. **[C]**

```
n     = len(train_ids)                 # 150 or 151
base  = n // 5 ;  rem = n % 5
sizes = [base + 1] * rem + [base] * (5 - rem)

n = 150 →  (30, 30, 30, 30, 30)
n = 151 →  (31, 30, 30, 30, 30)
```

This has **no free parameter** and is fully determined by the frozen fold manifest.

### 7.2 Leakage control — BINDING

> For every evaluation fold:
> 1. the fold's **test participants are held out entirely**;
> 2. the **training participants are all remaining participants** (150 or 151);
> 3. **clients partition ONLY the training participants** — disjointly and exhaustively;
> 4. **the held-out fold is NEVER used for any local training**, on any client;
> 5. the aggregated global model is evaluated **only** on that held-out fold.
>
> The validator and the verifier each independently assert train ∩ test = ∅ per fold and
> per client. **[C]**

## 8. Arms — the no-DP control is mandatory

| Arm | Definition | Tag |
|---|---|---|
| **N** | **no-DP control** — aggregate the raw deltas | **[C]** |
| **D** | **DP arm** — clip + Gaussian noise, then aggregate | **[C]** |

**Both arms derive from the SAME local training run per (fold, client).** They are therefore
identical in data, partition, seed, initialisation and training, **differing only in whether
DP is applied.** This is the B-3B pattern and it is what makes the diagnosis in §12 possible. **[C]**

## 9. Differential privacy — frozen, unchanged

| Field | Value | Tag |
|---|---|---|
| Mechanism | **Gaussian output perturbation** | **[A]** |
| Clipping norm C | **1.0** | **[A]** |
| Noise multiplier | **1.0** (σ_eff = 1.0) | **[A]** |
| ε per client update | **5.302585092994046** | **[A]** |
| δ | **1e-05** | **[A]** |
| Composition | **RDP single-composition, T = 1** | **[A]** |
| Accounting | frozen `dp_agent._rdp_to_dp`, **recomputed live** | **[A]** |

Applied to the flattened genuine delta: clip to L2 ≤ C, then add isotropic `N(0, (σ_eff·C)²)`.

**Round-level accounting [C]:** the 5 clients hold **disjoint** participant sets (§7.2), so
**parallel composition** applies and the round-level guarantee is
**ε = 5.302585092994046, δ = 1e-05** — the maximum over clients, *not* the sum. Disjointness
is asserted by both the validator and the verifier; if it ever failed, the run aborts.

**No privacy parameter may be tuned** — roadmap Row 8: *"A utility gain bought by weakening
an already-weak budget is not a gain."* **[A]**

## 10. Aggregation — frozen, unchanged

| Field | Value | Tag |
|---|---|---|
| Implementation | `AggregatorAgent._aggregate_state_dicts:422` | **[A]** |
| Mode | **`trimmed_mean`** | **[A]** |
| `trim_ratio` | **0.1** | **[A]** |
| N clients | 5 → `lower=1, upper=4` → **keeps 3 of 5** per coordinate | **[A]** |

The frozen aggregator validates fail-loud (identical sorted key sets, identical shapes,
byte-identical non-float buffers) and aggregates **parameter-wise**, which also bounds peak
memory. **Global model = `w_before + aggregated_delta`.** **[C]**

## 11. Evaluation

| Field | Value | Tag |
|---|---|---|
| Manifest | frozen `fold_manifest.json` (SHA `b9a7a91f…`) | **[A]** |
| Protocol | participant-level stratified 5-fold, PHQ > 10 | **[A]** |
| Splits | 150/38, 150/38, 150/38, 151/37, 151/37 — zero overlap | **[A]** |
| **Primary metric** | **ROC-AUC** from `softmax(logits)[:, 1]` | **[A]** |
| Also reported | Accuracy · Precision · Recall · F1 · PR-AUC | **[C]** |
| Comparator | Baseline-CV ROC-AUC **0.6333305966064586**, 95 % CI **[0.5755177227457472, 0.6911434704671701]** | **[A]** |

> **Accuracy is never presented alone.** Baseline-CV records accuracy 0.7606 with
> F1 = Recall = Precision = **0.0** — the accuracy illusion. **[A]**

**No numeric acceptance threshold is defined and none may be invented.** Phase 22 is a
demonstration; the Baseline-CV CI is a **scientific comparator**, not a pass/fail gate. **[C]**

## 12. Expected DP effect — PRE-REGISTERED, not post-hoc

At full multimodal dimensionality **d = 109,763,494**, with frozen σ_eff = 1.0:

```
expected Gaussian noise norm  ≈  σ_eff · √d  =  √109,763,494  ≈  10,477
clipped signal norm           =  C  =  1.0
```

**Exp 8 measured 10,472.8 at essentially this dimensionality** (its K0/full-state arm), and
Exp 9's K0 arm recorded the same order. This is a **confirmed measurement, not a hypothesis.**

> **BINDING — registered in advance:** **the possibility that the DP global model collapses
> to chance is an EXPECTED, PRE-REGISTERED outcome, not a post-hoc explanation.**
>
> **However, the design does NOT pre-declare the final task result.** The actual ROC-AUC of
> both arms **must still be measured and reported as measured.** Registering the expectation
> is not permission to assume it. **[C]**

## 13. The diagnostic the no-DP control provides — BINDING

The N arm distinguishes two fundamentally different conclusions, which must never be
conflated in reporting:

| Observation | Conclusion |
|---|---|
| **N performs well, D collapses** | The multimodal federated model **can** learn; **DP destroys it**. |
| **N already collapses** | The model is **weak before DP**; DP is not the limiting factor. |

> B-3B is precedent: its no-DP control scored 0.559165298 — itself outside the Baseline-CV
> CI — which located that failure **upstream of DP**. The same diagnostic logic applies here
> and **must be applied before attributing any Phase 22 outcome to differential privacy.** **[C]**

## 14. Seeds and determinism

| Field | Rule | Tag |
|---|---|---|
| Base seed | **1000** | **[A]** |
| Fold seed | `fold_seed = (1000 + 1) * 100 + fold` = **100101 … 100105** | **[C]** |
| Client seed | `client_seed = fold_seed * 10 + client_index` | **[C]** |
| DP seed | `dp_seed = client_seed`, re-seeded per client so the D arm is reproducible | **[C]** |
| Ordering | **seed → model → dataset → dataloader → loop is NORMATIVE** | **[A]** |

**All clients within a fold start from the identical global initial model `w_before`**,
constructed once per fold under `fold_seed` **before** any client training, and shared
byte-identically. The verifier asserts this. **[C]**

Repeats = **1** (§15). Training is stochastic (dropout 0.2 + `shuffle=True`), so single-run
results carry run-to-run variance that is **not** estimated here; no confidence interval over
repeats is computed or claimed. **[C]**

## 15. Execution count

| Quantity | Value | Tag |
|---|---|---|
| Folds | **5** | **[A]** |
| Clients per fold | **5** | **[C]** |
| Repeats | **1** | **[C]** |
| **Local trainings** | **5 × 5 = 25** | **[C]** |
| **Federated executions** (aggregate → global → evaluate) | **5 folds × 2 arms = 10** | **[C]** |
| Global-model evaluations | **10** | **[C]** |

Local training is **shared between arms** (§8), so 25 trainings yield all 10 federated
executions. This is both the minimum compute and the strongest guarantee that N and D differ
only by DP.

## 16. Portability and execution environment

| Field | Value | Tag |
|---|---|---|
| Target | **Colab GPU** (Tesla T4 class), via the existing bundle pattern | **[C]** |
| Local | `torch 2.8.0+cpu`, **CUDA unavailable** | **[B]** |
| Requirement | Code must be **device-agnostic** (`--device`), running identically on CPU and GPU | **[C]** |

Reference point: Exp 7 ran **100 fold-runs in 1,612.4 s on a Tesla T4**, each on ~150 training
participants. Phase 22's 25 client-trainings use ~30 participants each, so it is a materially
smaller job. **No runtime is predicted here**; the receipt records the measured value. **[C]**

**Memory note [C]:** a flattened delta is 109,763,494 × 4 B ≈ **439 MB**. Clipping and noise
must be applied **one client at a time**, releasing each buffer before the next, and
aggregation must use the frozen parameter-wise path (§10) rather than materialising all five
flat vectors at once.

## 17. Reporting — 13 items, reported independently

No single verdict is produced. Each is reported on its own:

1. Pipeline execution status · 2. Data integrity · 3. Client training completion ·
4. Genuine delta generation · 5. DP application · 6. ε accounting · 7. Aggregation ·
8. Global model reconstruction · 9. Held-out evaluation · 10. **No-DP (N) performance** ·
11. **DP (D) performance** · 12. **DP degradation (N − D)** · 13. **Comparison with Baseline-CV**

Plus the mandatory §2 distinction between **engineering success** and **task success**, and
the §13 diagnostic. **If task performance is poor, report it honestly. If the DP arm
collapses, report it honestly. If the no-DP arm also collapses, report that honestly.** **[C]**

## 18. Artifacts

All confined to **`trainer_outputs/phase22_end_to_end/`**. No existing artifact is modified. **[C]**

1. `phase22_partitions.json` — per-fold client partitions + leakage assertions
2. `phase22_clients.csv` — 25 local trainings (loss, delta norm, clipping, seeds)
3. `phase22_dp.json` — clipping fractions, norms, noise, ε accounting
4. `phase22_aggregation.json` — trim window, per-arm aggregation statistics
5. `phase22_eval.json` — per-fold, per-arm ROC-AUC + Accuracy/Precision/Recall/F1/PR-AUC
6. `phase22_receipt.json` — frozen SHAs pre/post, protocol, environment, runtime
7. `phase22_summary.json` — the 13 reporting items
8. `verify_phase22_report.json` — independent verification

## 19. Implementation structure

`phase22_end_to_end/` — **isolated; no previous experiment directory is modified**: **[C]**

| Module | Role |
|---|---|
| `phase22_common.py` | frozen constants, SHAs, A2-on-training-subset partition, seeds, metrics |
| `validate_phase22.py` | pre-execution, **fail-closed**; must exit 0 before any training |
| `run_phase22.py` | 25 local trainings → deltas → DP → aggregation → global → evaluation |
| `aggregate_phase22.py` | the 13 reporting items, N vs D, Baseline-CV comparison |
| `verify_phase22.py` | independent **fail-closed** verification |

## 20. Validation requirements (pre-execution, fail-closed)

Frozen input hashes · dataset shape · participant count (188) · modality dimensions
(768 / 154 / 84) · no missing modalities · fold integrity · client partition integrity
(disjoint, exhaustive over the training set, correct sizes) · **no train/test participant
overlap** · model architecture (109,763,494 params / 215 keys) · optimizer · epochs ·
batch size · DP parameters · aggregation parameters · output isolation. **[C]**

## 21. Verification requirements (independent, fail-closed)

`PASS` only if independently recomputed and matched; `FAIL` on mismatch; **`PENDING` if
evidence is absent — and PENDING counts against the verdict, never for it.** Exit non-zero
unless all PASS. **[A]**

Independently verify: participant partitioning · training/evaluation separation · model
parameter count · **`delta = w_after − w_before`** · clipping · DP parameters · ε accounting ·
aggregation · global state reconstruction · evaluation predictions · final metrics.

> **The verifier must not simply trust the runner's output.** **[A]**

## 22. No post-hoc changes — BINDING

1. Architecture, modality dimensions and parameter count are fixed.
2. lr, epochs, batch size, dropout, clipping, optimizer, loss and λ = 0.5 are fixed.
3. Client count, partition rule and per-fold sizes are fixed.
4. All DP parameters are fixed. **Clipping behaviour is measured, never tuned.**
5. Aggregation mode and `trim_ratio` are fixed.
6. The fold manifest, seeds and repeat count are fixed.
7. **No acceptance threshold exists, and none may be introduced.**
8. **No experiment may be added after seeing results.**
9. A degenerate task result is reported as measured, never re-tuned.

## 23. Scope

> Phase 22 is an **engineering demonstration** of the complete multimodal federated pipeline.
> It measures, and reports honestly, the task performance that pipeline produces.
>
> **It does NOT attempt, satisfy, or partially satisfy Roadmap Row 10 / C-1.** B-2 remains H₀,
> B-3 remains H₀ twice, and Row 10 remains blocked. Phase 22 makes **no claim** about clinical
> utility or predictive validity, and **engineering success does not imply task success.**
