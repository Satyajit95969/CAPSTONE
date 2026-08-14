# PHASE 20 / C-2 — NON-IID MULTI-CLIENT TOPOLOGY (ROADMAP ROW 11)
## Experiment Design — **FROZEN PRE-REGISTRATION**

> **Status: APPROVED AND FROZEN.** No parameter, threshold, partition, seed,
> metric or scope in this document may be changed after any result is observed.

> **C-2 computes NO task metric.** No ROC-AUC, accuracy, F1, MAE, RMSE or any
> other task-performance quantity. No held-out evaluation set exists. See §14.

> **C-2 does NOT unblock Roadmap Row 10.** Row 10 requires B-2 *and* B-3 to
> succeed; both returned H₀. See §22.

**Tag legend** — every value below carries one:
**[A]** inherited from a frozen source · **[B]** repository-supported ·
**[C]** newly approved C-2 decision · **[U]** unresolved (none remain).

---

## 1. Identity and research question

| Field | Value | Tag |
|---|---|---|
| Phase / item | Phase 20 / C-2, roadmap Row 11, backlog C-2 | **[A]** |
| Weakness | RC-5 | **[A]** |
| Single changed factor | **client partition (topology)** | **[B]** |

**Frozen question — `PHASE_10_ROADMAP.md:259`, verbatim:**

```
Non-IID multi-client | client topology | Exercise trimmed-mean on genuinely
divergent updates | P9 SS 6.4, SS 6.5 (devices = 1) | devices > 1 with
heterogeneous partitions. [H] Task metrics may degrade under non-IID - that is
a legitimate scientific finding, not a failure.
```

**Authorisation — `PHASE_10_ROADMAP.md:235`, verbatim:** *"**The one Future item
NOT blocked by B-2/B-3** — it can be studied on the existing probe and could be
pulled forward if an independent federation result is needed."* **[A]**

## 2. Scientific motivation

`PHASE_10_ROADMAP.md:186`, verbatim: *"`devices = 1`; three sequential uploads on
loopback within 45 s. The aggregation trigger is a **count (≥3), not an identity
check**. **Trimmed-mean — whose entire purpose is robustness across divergent
clients — is averaging three copies of the same file that differ only by their DP
noise draw.**"*

`P9 §6.5`: *"Single enrolled device … The '3 clients' are 3 sequential submissions
from the **same** device."*

C-2 replaces that with genuinely distinct client partitions and measures what the
frozen trimmed-mean aggregator does with the resulting divergent updates. **[A]**

## 3. Client topology

| Field | Value | Tag |
|---|---|---|
| n clients | **5** | **[C]** |
| Client sizes | **38 / 38 / 38 / 37 / 37** | **[B]** |
| Population | 188 participants, 45 pos / 143 neg (PHQ > 10) | **[A]** |

**Why n = 5, recorded so it is not re-litigated:** at n = 3 the frozen
`trim_ratio = 0.1` yields `lower = max(1, int(0.3)) = 1`, `upper = 2`, retaining
**one** value per coordinate — a coordinate-wise median that barely exercises
trimming. At n = 5 it retains **3 of 5**. n = 5 is also the only value with a
frozen, verified, disjoint, exhaustive partition already in the repository. **[C]**

## 4. Arm W0 — stratified / IID-like control

| Field | Value | Tag |
|---|---|---|
| Source | `trainer_outputs/baseline_cv/fold_manifest.json` **test** partitions | **[A]** |
| SHA-256 | `b9a7a91f1bdd43c78d3cd1ee0c36e4ce3fb8be4b0971dcff3ff62ee5af8fca2f` | **[A]** |
| Generator | `sklearn.StratifiedKFold(n_splits=5, shuffle=True, random_state=42)` | **[A]** |
| Stratified on | `PHQ_binary(PHQ > 10)` | **[A]** |

Measured properties (verification, read-only):

| client | n | pos | neg | pos_rate | PHQ mean | median | min | max |
|---|---|---|---|---|---|---|---|---|
| 1 | 38 | 9 | 29 | 0.2368 | 6.6842 | 5.0 | 0 | 21 |
| 2 | 38 | 9 | 29 | 0.2368 | 7.7895 | 7.0 | 0 | 20 |
| 3 | 38 | 9 | 29 | 0.2368 | 6.8947 | 5.5 | 0 | 20 |
| 4 | 37 | 9 | 28 | 0.2432 | 6.4595 | 5.0 | 0 | 22 |
| 5 | 37 | 9 | 28 | 0.2432 | 5.5405 | 3.0 | 0 | 23 |

> **TERMINOLOGY — BINDING.** This arm is **"stratified / IID-like"**. It is
> stratified on the *binary* label; its PHQ means span 5.5405–7.7895, so it is
> **not** homogeneous on the continuous target. **It must never be described as
> perfectly IID.** **[C]**

## 5. Arm W1 — non-IID treatment (A2)

**Deterministic rule [C]:**

```
sort participants by (phq_score DESC, participant_id ASC)
fill clients sequentially with sizes 38, 38, 38, 37, 37
```

Ties in `phq_score` are broken by `participant_id` ascending, so the partition is
fully determined with **no free parameter**.

Measured properties (verification, read-only):

| client | n | pos | neg | PHQ mean | median | min | max | sd | PHQ range |
|---|---|---|---|---|---|---|---|---|---|
| 1 | 38 | 38 | 0 | 16.0789 | 16.0 | 12.0 | 23.0 | 3.0555 | [12, 23] |
| 2 | 38 | 7 | 31 | 9.2632 | 9.0 | 7.0 | 11.0 | 1.2708 | [7, 11] |
| 3 | 38 | 0 | 38 | 5.3158 | 5.0 | 3.0 | 7.0 | 1.2588 | [3, 7] |
| 4 | 37 | 0 | 37 | 2.1622 | 2.0 | 1.0 | 3.0 | 0.7171 | [1, 3] |
| 5 | 37 | 0 | 37 | 0.2973 | 0.0 | 0.0 | 1.0 | 0.4571 | [0, 1] |

**Declared in advance:** client 5 has PHQ ∈ [0, 1] with sd 0.4571. Under MSE
regression its update will be dominated by driving the output toward 0. That is
the intended divergence, stated here rather than discovered in the results. **[C]**

**A2 was chosen over A1** because A1's `participant_id` tie-break produced an
*unintended* monotone severity gradient among its negative clients (PHQ means
4.8684 → 3.6486 → 3.0811), i.e. a hidden second factor. A2 makes the gradient
explicit and controlled. **[C]**

## 6. Representation — **FROZEN AND PINNED**

| Field | Value | Tag |
|---|---|---|
| Model | `mental/mental-bert-base-uncased` | **[A]** |
| **Revision** | **`24809aa822c76639760d0d934742d1b42f89942f`** | **[C]** |
| `max_length` | **128** | **[A]** |
| truncation | `True` | **[A]** |
| padding | `"max_length"` | **[A]** |
| Pooling | `last_hidden_state[:, 0, :]` — CLS, **no `.mean(dim=0)`** | **[A]** |
| Dimension | **768** | **[A]** |

**Pinned artifact SHA-256 [C]** (from the local HF snapshot):

```
c4f90fa5f0b991c48eb99afe41c8883dccb2b7e51012b33d2635925b9cde8764  pytorch_model.bin
5fd1c882abbd30517dced455a2c9768945ec726b96727927e4959348d9de550b  tokenizer.json
07eced375cec144d27c900241f3e339478dec958f92fddbc551f295c992038a3  vocab.txt
79ee28a4e33b49f535209c8aaa5eaa7550344345bb3fecb13f179c686749d0d9  config.json
9a8ed9b01c8a56b555dcdc31cd526ba9e488cfc16ee5a101b3ce64af81d34f3d  tokenizer_config.json
303df45a03609e4ead04bc3dc1536d0ab19b5358db685b6f3da123d05ec200e3  special_tokens_map.json
```

**No prior experiment pinned this revision.** Pinning it is new, and is what makes
the C-2 representation reproducible. **[C]**

### 6.1 Prohibited representation sources — BINDING **[C]**

- **NOT** `secure_store/sess-DAICWOZ/encrypted/text_embeddings.parquet.enc`
- **NOT** `max_length = 256`
- **NOT** any other BERT checkpoint or unpinned revision
- **NOT** `create_dp_comparison.py` as a dependency
- `format_daic_to_lda.py` and its artifact are **read-never, write-never**

### 6.2 Extraction path **[C]**

A new C-2-only module reads `daic_records.parquet` and writes a new C-2 artifact.
Justification for equivalence with the frozen family: `MultiModalModel.__init__`
builds `self.bert = AutoModel.from_pretrained(bert_name)` — a *pretrained-only*
encoder at construction — and `forward` takes `last_hidden_state[:, 0, :]`
(`trainer_mentalbert_daic.py:256`). A standalone `AutoModel` at the pinned
revision with `max_length=128` reproduces that vector exactly. Padding does not
affect CLS (index 0, pads masked). **[B]**

## 7. Data

| Field | Value | Tag |
|---|---|---|
| Source | `daic_records.parquet` | **[A]** |
| SHA-256 | `9a241851760727779fcaa4d50041b8bdc4406fe6b66ddbbd7105f4ce4fc55f00` | **[A]** |
| Rows / participants | 188 / 188, 0 duplicates, 0 nulls | **[A]** |
| Target | **continuous `phq_score`**, range [0, 23] | **[C]** |

## 8. Shared initialisation

| Field | Value | Tag |
|---|---|---|
| Object | `trainer_outputs/local_probe_base.pt` | **[A]** |
| SHA-256 | `d21f95ab4169ba8bf270e2ae900d2d205ddb4c3aba3c04ebae4d0ec8d9ba0993` | **[A]** |
| Role | **`w_before` — shared initialisation ONLY** | **[C]** |
| Tensors | `fc1.weight (384,768)`, `fc1.bias (384)`, `fc2.weight (1,384)`, `fc2.bias (1)` | **[A]** |
| Parameters | **295,681** | **[A]** |

> **It is NOT the client update and NOT the federated payload.** Its provenance —
> fitted to `sample_texts/sample1.txt`, a single 43-byte sentence, with no DAIC
> signal (`PHASE_10_ROADMAP.md:185`, `P9:234`) — is scientifically irrelevant in
> this role: it is a common origin shared by every client in every arm and
> repeat, and it cancels in `w_after − w_before`. **[C]**

Every client in every arm and repeat starts from **this exact object**. No client
starts from another client's weights. No checkpoint is reused across clients or
repeats. **[C]**

## 9. Local client training

| Field | Value | Tag |
|---|---|---|
| Architecture | `Linear(768,384) → ReLU → Linear(384,1)` | **[B]** |
| Loss | `torch.nn.MSELoss()` | **[B]** |
| Optimizer | `torch.optim.Adam` | **[B]** |
| Learning rate | **1e-3** | **[C]** |
| Epochs | **1** | **[C]** |
| Batch | **full batch** (single forward/backward per epoch) | **[B]** |
| Gradient clipping | **1.0** | **[B]** |
| Initialisation | `w_before` loaded via `load_state_dict(strict=True)` | **[C]** |

## 10. Delta computation

```
delta_i = compute_state_delta(w_before, w_after_i)
```

`trainer_mentalbert_daic.py:348` (FROZEN) — `delta[k] = (after[k] − before[k])`.
This is the repository's established convention, used at `:382` and mirrored in
`trainer_agent/trainer_mentalbert_privacy.py:584`. **[A]**

## 11. Differential privacy — inherited, unvaried

| Field | Value | Tag |
|---|---|---|
| Mechanism | Gaussian | **[A]** |
| ε per update | **5.302585092994046** | **[A]** |
| δ | **1e-05** | **[A]** |
| `clip_norm` | **1.0** | **[A]** |
| `noise_multiplier` | **1.0** | **[A]** |
| Composition | RDP single-composition, **T = 1** | **[A]** |
| ε source | frozen `dp_agent._rdp_to_dp`, imported never reimplemented | **[A]** |

DP is applied identically in both arms. **No DP parameter may vary.** C-2 is not
a DP experiment. **[A]**

## 12. Aggregation

| Field | Value | Tag |
|---|---|---|
| Engine | `server/aggregator_agent/aggregator.py::AggregatorAgent` | **[A]** |
| Mode | `trimmed_mean` | **[A]** |
| `trim_ratio` | **0.1** | **[A]** |
| Retained at n = 5 | `lower=1, upper=4` → **3 of 5** per coordinate | **[A]** |

The **frozen aggregator is used directly**, not reimplemented, so C-2 measures the
real trimming logic. **[C]**

## 13. Repeats and seed hierarchy

| Field | Value | Tag |
|---|---|---|
| Repeats | **5** | **[C]** |
| `repeat_seed` | `1000 + repeat`, repeats 1..5 → 1001..1005 | **[C]** |
| `client_seed` | `repeat_seed * 100 + client_index` | **[C]** |
| DP seed | `1000 + repeat` | **[C]** |

Identical seeds are used in both arms. The exact mapping is recorded in the seed
manifest artifact. **[C]**

## 14. Metrics — **NO TASK METRIC IS COMPUTED**

> **BINDING.** C-2 computes **no** ROC-AUC, accuracy, F1, MAE, RMSE, PR-AUC or any
> other task-performance quantity. **No held-out evaluation set is introduced.**
> All 188 participants remain assigned exactly once per arm. **[C]**

**Primary — manipulation check [C]:** pre-DP pairwise client-update divergence,
reported as **both** L2 distance and cosine distance (10 pairs at n = 5).

**Why pre-DP is primary [B]:** Exp 8 measured post-noise L2 ≈ **544** against a
clipped signal norm of **1.0**. Post-DP divergence is ~99.8 % noise and cannot
discriminate topologies. Only the pre-DP update carries the partition signal.

**Secondary [C]:** post-DP pairwise divergence (L2, cosine) · per-coordinate
trimmed-client counts · distance of the trimmed-mean aggregate from the ordinary
mean · distance from the coordinate median.

**Also reported [A]:** DP accounting (ε, δ, cumulative, composition) and integrity
verification.

**Precedent for the metrics [B]:** L2 via `torch.norm(t, p=2)`
(`dp_agent.py:191,201`, `run_exp8.py:259`); cosine via
`sklearn.metrics.pairwise` (`create_dp_comparison.py:293,311`).

## 15. Acceptance — reporting only

> **There is NO numeric acceptance threshold, and none may be invented.**
> `PHASE_10_ROADMAP.md:259` states task degradation under non-IID is *"a
> legitimate scientific finding, not a failure"*, and supplies no numeric
> criterion for Row 11. **[A]**

**The single logical validity gate [C]:**

```
non-IID (W1) pre-DP client-update divergence
        MUST EXCEED
stratified/IID-like (W0) pre-DP client-update divergence
```

Directional only — **no floor, no margin, no MDE.** If it fails, the manipulation
did not produce divergent clients and nothing was exercised; that is reported as
such, not as a numeric failure.

## 16. Integrity and leakage controls

1. Clients pairwise disjoint; every participant exactly once per arm; 188/188
   coverage in **both** arms. **[C]**
2. Both arms cover the **same** 188 participants; only membership differs. **[C]**
3. Frozen-SHA gate before **and** after execution, inside the execution path. **[A]**
4. Model revision and artifact hashes verified before any forward pass. **[C]**
5. Fresh model per client, loaded from `w_before` with `strict=True`. **[C]**
6. No checkpoint reuse across clients, arms or repeats. **[C]**
7. Deterministic seeding recorded in a seed manifest. **[C]**
8. All artifacts written **only** under `trainer_outputs/c2_multiclient/`. **[C]**
9. Fail-closed verification with independent recomputation. **[A]**

## 17. Single-factor principle — BINDING

Identical across arms: model · revision · tokenizer · representation ·
preprocessing · initialisation · optimizer · learning rate · epochs · loss ·
batch · gradient clipping · seeds · DP · aggregation · repeats.

**ONLY client membership differs.** **[C]**

## 18. Artifacts

Under `trainer_outputs/c2_multiclient/`:

1. `c2_model_receipt.json` — revision + artifact hashes + representation spec
2. `c2_embeddings.npz` — 188 × 768 (+ participant ids)
3. `c2_partitions.json` — both arms, per-client participant ids and statistics
4. `c2_seed_manifest.json` — exact seed mapping
5. `c2_client_updates.csv` — per client: pre/post-DP norms, clip scale, ε
6. `c2_divergence_pre_dp.json` — L2 and cosine matrices per arm/repeat
7. `c2_divergence_post_dp.json` — same, post-DP
8. `c2_aggregation.json` — trimmed-mean results, trim counts, distances
9. `c2_receipt.json` — frozen SHAs pre/post, environment, runtime
10. `c2_summary.json` — validity gate, all metrics
11. `c2_acceptance.csv` — validity-gate row
12. `c2_verification.json` — fail-closed verification

**No task-metric artifact exists.** **[C]**

## 19. Failure and abort conditions — HARD ABORT

Model revision mismatch · model artifact hash mismatch · frozen artifact SHA
mismatch · participant duplication or omission · client overlap · arm population
mismatch · NaN/Inf anywhere · state_dict mismatch · delta shape mismatch · DP
parameter deviation · `trim_ratio` deviation · unexpected checkpoint reuse · seed
mapping deviation · output path collision with a frozen artifact · aggregation
failure · integrity verification failure. **[A]**

## 20. Known limitations — recorded in advance

1. **`max_length = 128` against long transcripts.** Median transcript is 9,877
   characters; 128 tokens covers roughly the first ~500. This is **inherited from
   the frozen family and must not be changed**. **[A]**
2. **Exp 8 norms are not comparable.** Exp 8's ‖v‖₂ = 11.3448 and its SNR/G2/G3
   constants were measured on a **full-state** object that the producer labelled a
   "delta" (`create_dp_comparison.py:1163,1167`). C-2 uses a genuine
   `w_after − w_before`, whose norm will differ and whose clipping at C = 1.0 will
   bind differently. **Do not compare C-2 delta norms or SNR to Exp 8's.** **[C]**
3. **`local_probe_base.pt` carries no DAIC signal** — neutralised by its shared-
   initialisation role (§8), but stated. **[A]**
4. **Client 5 of A2 has near-zero target variance** (§5). **[C]**
5. **The stratified control is not perfectly IID** (§4). **[C]**
6. **n = 5 is small for trimmed-mean**; 3 of 5 retained. **[C]**

## 21. Scope

> C-2 is an **aggregation / manipulation experiment**. It measures whether the A2
> non-IID topology produced genuinely divergent client updates relative to the
> stratified/IID-like control, and how the frozen trimmed-mean aggregator behaved
> on those updates.
>
> **It makes NO claim — explicit or implied — about task performance, model
> generalisation, clinical utility or predictive validity.** No task metric is
> computed, so no such claim is available to it.

## 22. Why C-2 does not unblock Row 10

`PHASE_10_ROADMAP.md:234` — C-1 (Row 10) *"is **structurally impossible** until
B-2 and B-3 succeed."* `:274` — *"Exps 8 → 10 are a **strict dependency chain**."*
**B-2 (Exp 8) returned H₀; B-3 (Exp 9) returned H₀.** C-2 is an enabler for B-2's
distributed-noise mechanism, **not a substitute for either.** **[A]**

## 23. No post-hoc changes — BINDING

1. Arms fixed at W0 and W1. No arm added, removed or selected among.
2. Partition rules fixed; A2 may not be re-sorted or re-sized.
3. n, client sizes, `trim_ratio`, repeats and the seed hierarchy are fixed.
4. All DP parameters fixed.
5. The representation and its pinned revision are fixed.
6. **No task metric may be added.**
7. **No numeric acceptance threshold may be introduced.**
8. A failed validity gate is reported as a failed manipulation, never re-tuned.

---

**Pre-registration status: FROZEN AND APPROVED.**
**Implementation is authorised; execution requires separate approval after validation.**
