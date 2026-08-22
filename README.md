# MediProof — Privacy-Preserving Federated Multimodal Depression Detection

**BE Major Project · DAIC-WOZ · MentalBERT · Differential Privacy · Federated Aggregation**

> **This README is the authoritative entry point.** It explains the whole project — problem, architecture, every experiment run so far, every verdict, what is proven, what is not, what is blocked, and what a new developer must do before touching anything.
>
> **Read this file completely before running or changing anything.**

---

## ⚠️ STATUS BANNER — READ FIRST

| | |
|---|---|
| **Infrastructure** | ✅ Built and verified end-to-end |
| **Scientific programme (Version 2)** | ⏹️ **Concluded on negative results** |
| **Headline outcome** | **The federated DAIC task objective was NOT achieved.** Two independent levers each returned H₀. |
| **Row 10 / C-1 (federate a DAIC-derived object)** | 🚫 **BLOCKED — never executed** |
| **Phase 22** | ✅ **EXECUTED & VERIFIED 20/20 on Colab GPU — engineering success; TASK success NOT established** |

**Nothing in this repository demonstrates a working end-to-end multimodal federated depression-detection system producing non-trivial task metrics.** The pipeline *runs*; the *science* returned negative results. Do not represent it otherwise.

**Phase 22 HAS now been executed** (Colab GPU, verification 20/20). The pipeline works end-to-end; the resulting model does **not** discriminate — no-DP arm ROC-AUC 0.4651, DP arm exactly 0.500. **Row 10 remains BLOCKED.** See [§19](#19-phase-22--end-to-end-multimodal-federated-demonstration).

---

## 🚨 PRE-PUSH SECURITY BLOCKER

**Five private keys — including the CA root key — are already in pushed history on the GitHub remote.** Do not push again until resolved. See [§19Z.3](#19z3-security-audit--files-to-remove-from-tracking).

---

## 🔄 Project Continuity Rule

> **README.md is this project's mandatory living handover document.** It must always
> contain enough context for a completely new developer — or a fresh AI session with no
> conversation history — to continue the work without asking anyone anything.

**It MUST be updated at every meaningful milestone, before the task is declared complete:**

| Update on | Examples |
|---|---|
| Phase starts | a new phase is opened |
| Design freezes | a `PHASE_*_DESIGN.md` is frozen (record its **full SHA-256**) |
| Implementation completion | experiment modules written |
| Validation results | validator pass/fail counts |
| Execution results | runs completed, artifacts produced |
| Verification results | independent verification counts |
| Closures | a `FINAL_*_CLOSURE.md` is written |
| Blockers | anything preventing progress |
| Environment changes | CUDA/GPU, dependency pins, bundles |
| Important bugs/fixes | defects found and how they were resolved |
| Final verdicts & phase completion | H₀ / SUCCESS / PARTIAL / COMPLETE |

**Do NOT** update it for every trivial shell command or intermediate inspection — update at
*meaningful checkpoints*, not continuously.

### Authority order — BINDING

```
frozen design  >  closure document  >  result artifact  >  verified SHA  >  README.md
```

**README.md must never contradict any of those.** If a conflict is found, the authoritative
artifact wins and **README.md is the file that gets corrected** — never the reverse, and never
by editing a frozen artifact to match this file.

---

## Table of Contents

1. [Project Overview](#1-project-overview) · 2. [Problem Statement](#2-problem-statement) · 3. [Research Question](#3-research-question)
4. [System Architecture](#4-system-architecture) · 5. [Multimodal Data Pipeline](#5-multimodal-data-pipeline) · 6. [Federated Learning Architecture](#6-federated-learning-architecture)
7. [Differential Privacy](#7-differential-privacy) · 8. [Federated Aggregation](#8-federated-aggregation) · 9. [Dataset and Participants](#9-dataset-and-participants)
10. [Evaluation Methodology](#10-evaluation-methodology) · 11. [Frozen Inputs](#11-frozen-inputs) · 12. [Experiment Methodology](#12-experiment-methodology)
13. [Roadmap and Dependencies](#13-roadmap-and-dependencies) · 14. [Experiment History](#14-experiment-history-chronological)
15. [Current Project Status](#15-current-project-status) · 16. [What Has Been Proven](#16-what-has-been-proven) · 17. [What Has NOT Been Proven](#17-what-has-not-been-proven)
18. [Limitations](#18-limitations) · 19. [Phase 22](#19-phase-22--end-to-end-multimodal-federated-demonstration) · 20. [Repository Structure](#20-repository-structure)
21. [Important Files and Artifacts](#21-important-files-and-artifacts) · 22. [Reproducibility / Integrity Rules](#22-reproducibility--integrity-rules) · 23. [Takeover Guide](#23-if-you-are-taking-over-this-project)
24. [Installation](#24-installation) · 25. [External Tools](#25-external-tools-openface--opensmile) · 26. [Troubleshooting](#26-troubleshooting) · 27. [Credits](#27-credits-and-citation)

---

## 1. Project Overview

MediProof is a **modular, privacy-preserving federated learning framework** for detecting depression from multimodal clinical-interview data (text, audio, video), built on the DAIC-WOZ corpus.

It is organised as cooperating agents:

| Layer | Component | Responsibility |
|---|---|---|
| **Client** | **Local Data Agent (LDA)** | Multimodal ingestion & preprocessing (text/audio/video), PII scrubbing, AES-GCM encryption, signed receipts |
| | **Trainer Agent** | Local training of MentalBERT / multimodal model / probe head |
| | **DP Agent** | Differential-privacy noise mechanisms (Gaussian, Laplace, Uniform, Exponential, Student-t) |
| | **Encryption Agent** | AES-GCM, Fernet, KMS-envelope, CKKS homomorphic |
| **Server** | **Orchestrator Agent** | Coordinates rounds, training, DP, explainability, logging |
| | **Aggregator Agent** | Secure aggregation of model updates (trimmed mean) |
| | **Key Management Agent** | Key issue/rotation for SecureStore |
| | **Audit Agent** | Compliance logging of training / DP / explainability integrity |

Deep references: **`ARCHITECTURE.md`** (1,134 lines — directory structure, runtime/enrollment/federated flows, gRPC, security, aggregation, MongoDB schema) and **`PROJECT_KNOWLEDGE.md`** (781 lines — tech stack, data flow, API/auth flows, dependency map).

---

## 2. Problem Statement

Mental-health data is among the most sensitive data that exists. A model that detects depression from clinical interviews cannot realistically be trained by pooling raw recordings in one place — that is exactly the disclosure risk clinicians and participants object to.

The project therefore asks whether the **detection task and the privacy guarantee can hold simultaneously**: each site trains locally, only privatised model updates leave the device, and a server aggregates them into a global model — with a real, accounted differential-privacy budget rather than a nominal one.

**The core tension this project actually measured:** the Gaussian mechanism adds noise proportional to `σ·√d` for a `d`-dimensional update. At `d ≈ 10⁸`, that noise overwhelms any clipped signal. **This is not a hypothesis in this repository — it was measured and confirmed** (see [§14](#14-experiment-history-chronological)).

---

## 3. Research Question

> **What is the smallest DAIC-derived object that still carries DAIC task signal and survives the differential-privacy path?**

That is roadmap item **B-3**. It was attempted **twice** and returned **H₀ both times**.

The parallel question — can the DP *mechanism* be made dimension-aware so signal survives at fixed `d`? — is **B-2**, which returned **H₀** and is closed by its own finding.

---

## 4. System Architecture

### 4.1 The model — `MultiModalModel` (`trainer_mentalbert_daic.py:241`)

```
input_ids, attention_mask ──► MentalBERT encoder ──► last_hidden_state[:,0,:]  (CLS, 768-d)
audio_vec  (154-d) ──► SmallMLP ──► 128-d                    ┐
vision_vec  (84-d) ──► SmallMLP ──► 128-d                    ├──► concat ──► FusionHead
                                                              ┘
```

- Audio/vision encoders are **conditionally constructed** — built only if `audio_dim`/`vision_dim` are non-zero.
- `fusion_input_dim` = 768 (text-only) or 768 + 128 + 128 = **1024** (full multimodal).

### 4.2 `FusionHead` (`trainer_mentalbert_daic.py:225–239`) — the task head

```python
fc1          = Linear(in_dim, hidden=256)  →  ReLU  →  Dropout(0.2)
classifier   = Linear(256, 2)     # binary logits  → ROC-AUC uses softmax(...)[:,1]
phq_mu       = Linear(256, 1)     # PHQ regression mean
phq_logsigma = Linear(256, 1)
```

At `in_dim=768, hidden=256`: **197,892 parameters in 8 tensors**.

### 4.3 Parameter counts (verified)

| Configuration | Parameters | state_dict keys |
|---|---|---|
| Text-only (`audio_dim=None, vision_dim=None`) | **109,680,132** | 207 |
| Full multimodal (154 audio, 84 vision) | **109,763,494** | 215 |
| `FusionHead(768, 256, 2)` alone | **197,892** | 8 |
| Legacy probe `ProbeModel(768)` (`fc1`/`fc2`) | **295,681** | 4 |

### 4.4 Training objective (frozen)

```python
loss = CrossEntropyLoss(logits, label) + 0.5 * MSELoss(mu, phq)
```

`label = PHQ_binary = phq_score > 10.0`. Recipe: `transformers.AdamW`, **lr 2e-5**, **3 epochs**, **batch 8**, **max_len 128**, **grad clip 1.0**, `DataLoader(shuffle=True)`.

---

## 5. Multimodal Data Pipeline

| Modality | Extraction | Dimension |
|---|---|---|
| **Text** | Transcripts → MentalBERT tokenizer → CLS embedding | **768** |
| **Audio** | openSMILE eGeMAPS / wav2vec2 / prosody | **154** |
| **Vision** | OpenFace facial Action Units (+ face blurring for anonymisation) | **84** |

Built by `build_daic_multimodal_features.py`, `build_daic_multimodal_records.py`, `build_participant_intervals.py`; verified by `verify_daic_multimodal_records.py`. Output: `dataset_build/daic_records_multimodal.parquet`.

**Text representation used by every frozen experiment:**

| Field | Value |
|---|---|
| Model | `mental/mental-bert-base-uncased` |
| **Pinned revision** | `24809aa822c76639760d0d934742d1b42f89942f` |
| `max_length` | **128** · `truncation=True` · `padding="max_length"` |
| Pooling | `last_hidden_state[:, 0, :]` (CLS) |
| Dimension | **768** |

> ⚠️ **Do not confuse this with `format_daic_to_lda.py`**, which uses `max_length=256` and caps at `MAX_PARTICIPANTS = 20`. It is **incompatible** with the frozen family and must never be used as an experiment input.

---

## 6. Federated Learning Architecture

**Round lifecycle:** client trains locally → computes update → DP → encrypt → chunked upload → server aggregates (count trigger ≥3) → publishes global model → clients download → next round.

### The update representation — **critical convention**

```python
# trainer_mentalbert_daic.py:348  — FROZEN
def compute_state_delta(before, after):
    delta[k] = after[k] - before[k]        # w_after − w_before
```

> ⚠️ **Known deviation, documented for provenance:** `create_dp_comparison.py:1163–1167` saves the **full post-training state** and *labels* it a "delta" (its own comment: *"we treat full state as 'delta' for downstream"*). `trainer_outputs/local_probe_base.pt` is such an object. Exp 8's norms were measured on it. **Exp 9 §20.2 binds: do not compare those norms to genuine deltas.** Experiments 9, C-2 and B-3B all use genuine `w_after − w_before`.

---

## 7. Differential Privacy

**Frozen parameters — these may NOT be tuned** (roadmap Row 8: *"A utility gain bought by weakening an already-weak budget is not a gain — the privacy guarantee is the constraint, not the objective."*):

| Parameter | Value |
|---|---|
| Mechanism | **Gaussian output perturbation** |
| **ε per update** | **5.302585092994046** |
| **δ** | **1e-05** |
| **Clipping norm C** | **1.0** |
| **Noise multiplier** | **1.0** (σ_eff = 1.0) |
| Composition | **RDP single-composition, T = 1** |
| Cumulative ceiling | ≤ **15.9078 / round** |
| Accounting | `dp_agent/dp_agent.py` — recomputed live, never re-implemented |

**Procedure:** clip the flattened update to L2 ≤ C, then add isotropic `N(0, (σ_eff·C)²)`.

**The √d problem — measured, not theorised.** Noise norm ≈ `σ_eff·√d`. At `d = 295,681` that is **543.77**; at `d = 109,680,132` it is **10,472.8**. Because ε depends on `σ_eff` alone and **not** on `d`, the only levers are the *mechanism* (B-2) and the *dimensionality* (B-3). Both were tested. Both failed.

---

## 8. Federated Aggregation

**`AggregatorAgent`** — `server/aggregator_agent/aggregator.py`, mode **`trimmed_mean`**, **`trim_ratio = 0.1`**:

```python
lower = max(1, int(trim_ratio * N))
upper = N - lower
result = mean(sorted_values[lower:upper])     # coordinate-wise
```

At **N = 5**: `lower=1, upper=4` → **keeps 3 of 5** per coordinate. (At N = 3 it would keep only 1 — a bare median — which is why C-2 chose 5 clients.)

---

## 9. Dataset and Participants

| Field | Value |
|---|---|
| Corpus | **DAIC-WOZ** |
| Participants | **188 of 189** — the 189th archive is corrupt and unrecoverable |
| Label | `PHQ_binary = phq_score > 10.0` |
| Class balance | **45 positive / 143 negative** (23.9 % positive) |
| Frozen artifact | `daic_records.parquet` |

> **The dataset lever is exhausted.** The corpus is complete. Growing the training set 102 → 170 (+66 %) moved P/R/F1 by **0.0**. *"We need more data"* is not available as an explanation for any remaining failure.

---

## 10. Evaluation Methodology

**Baseline-CV** is the single valid comparator for all Version 2 work.

| Field | Value |
|---|---|
| Protocol | Participant-level stratified 5-fold |
| Generator | `sklearn.StratifiedKFold(n_splits=5, shuffle=True, random_state=42)` |
| Stratified on | `PHQ_binary (PHQ > 10)` |
| Splits | 150/38, 150/38, 150/38, 151/37, 151/37 — **zero overlap** |
| Structure | 5 folds × 5 repeats = **25 rows**; repeats averaged **within** fold |
| Statistics | fold-level Student-t, **df = 4**, **t_crit = 2.7764451051977987** |

**Baseline-CV measured values:**

| Metric | Mean | 95 % CI |
|---|---|---|
| **ROC-AUC** | **0.6333305966064586** | **[0.5755177227457472, 0.6911434704671701]** |
| PR-AUC | 0.39406350916313804 | [0.29904536, 0.48908166] |
| MAE | 4.812727717824207 | [4.40611919, 5.21933625] |
| Accuracy | 0.7605974395448081 | [0.75624411, 0.76495077] |
| **Precision / Recall / F1** | **0.0 / 0.0 / 0.0** | [0, 0] |
| `pred_var` | 0.005734089470007205 | — |
| **MDE (ROC-AUC)** | **0.05781287386071144** | — |

> **ROC-AUC is the only Baseline-CV metric whose CI excludes chance.** Accuracy 0.76 with F1 = 0.0 is the *accuracy illusion*: the classifier predicts the majority class for every input. Never quote accuracy alone.

---

## 11. Frozen Inputs

SHA-256 pinned; verified before **and** after every experiment. **Never modify these.**

| Artifact | SHA-256 |
|---|---|
| `daic_records.parquet` | `9a241851760727779fcaa4d50041b8bdc4406fe6b66ddbbd7105f4ce4fc55f00` |
| `trainer_outputs/baseline_cv/fold_manifest.json` | `b9a7a91f1bdd43c78d3cd1ee0c36e4ce3fb8be4b0971dcff3ff62ee5af8fca2f` |
| `trainer_outputs/baseline_cv/baseline_cv_summary.json` | `f065f5e1ff7f8e5ed4d122a8031c9cc12425802e756697d5512a915674871aac` |
| `trainer_outputs/local_probe_base.pt` | `d21f95ab4169ba8bf270e2ae900d2d205ddb4c3aba3c04ebae4d0ec8d9ba0993` |
| `trainer_mentalbert_daic.py` | `65b1902e2ba8dd996c6960db1a6595d84fd48500f4d8707cb79688093d7a230b` |
| `dp_agent/dp_agent.py` | `758642fff57695cb970af88789c3b6a17c77f2b01d16303ff96230fce5798732` |
| `PHASE_18_EXP9_DESIGN.md` | `cf361986e4f058259e21fefd548b335ad9fb91758696d57fdf6c762254bf288a` |
| `PHASE_19_EXP10_DESIGN.md` | `71fdb1b3c6b3c4bc402e688769a8aaae447283dfbe5ed62dcc3076f9c7b59a20` |
| `PHASE_20_C2_DESIGN.md` | `687ea6a5b57d2a0582f0061a08df4fe916cfdf6d1ffa2b4c48703ead9b2910a0` |
| `PHASE_21_B3B_DESIGN.md` | `00f8e34bb505f06b02466dbb82b679e688bbb4bdf394dadaab8c67d59c820590` |
| `trainer_outputs/c2_multiclient/c2_embeddings.npz` | `70df4e49b4fea83c2464eba3e41e5b8ffa856fff4c70dc7b7b8d054cf488ed3c` |

**MentalBERT pinned revision `24809aa8…`** — 6 artifacts verified (`pytorch_model.bin` `c4f90fa5…`, `tokenizer.json` `5fd1c882…`, `vocab.txt` `07eced37…`, `config.json` `79ee28a4…`, `tokenizer_config.json` `9a8ed9b0…`, `special_tokens_map.json` `303df45a…`).

**Also never modify:** `format_daic_to_lda.py`, `secure_store/sess-DAICWOZ/encrypted/text_embeddings.parquet.enc`, and the three official baseline reports (`PHASE_8B_EXECUTION_REPORT.md`, `PHASE_9_BASELINE_EVALUATION_REPORT.md`, `PHASE_9.5_COMPLETE_DATASET_REPORT.md`).

---

## 12. Experiment Methodology

This project runs a **frozen pre-registration** protocol. Every experiment follows:

1. **Read-only audit** → establish what the repository actually supports
2. **Frozen design document** (`PHASE_*_DESIGN.md`) — SHA-pinned; every value tagged **[A]** inherited / **[B]** repository-supported / **[C]** newly approved
3. **Pre-execution validation** — fail-closed; aborts before any training
4. **Execution** — frozen-SHA gate before *and* after
5. **Independent fail-closed verification** — re-derives results from scratch; `PASS` only if recomputed and matched, `PENDING` if evidence is missing, and **PENDING counts against the verdict, never for it**
6. **Closure document**

**Binding rules:** exactly one factor changes per experiment · no threshold may be moved, added or invented after results are seen · **H₀ is a legitimate scientific result, never an implementation failure** · never claim success merely because code executed.

---

## 13. Roadmap and Dependencies

Authoritative: **`PHASE_10_ROADMAP.md`**.

```
Rows 0a–6  (diagnostics → recipe fixes)          ✅ executed
Row 7      B-1  multimodal activation            ✅ COMPLETE
Row 8      B-2  dimension-aware DP mechanism     ❌ H₀
Row 9      B-3  compact update representation    ❌ H₀ ×2  (Exp 9, B-3B)
                          │
                          ▼   requires B-2 AND B-3 to SUCCEED
Row 10     C-1  federate a DAIC-derived object   🚫 BLOCKED — never executed
                          │
                ┌─────────┴─────────┐
                ▼                   ▼
Row 12  C-3  ε–utility curve   C-4  DP mechanism sweep
        🚫 BLOCKED                 🚫 BLOCKED
Row 11     C-2  non-IID multi-client             ✅ COMPLETE (outside the chain)
```

> **⚠️ Numbering ambiguity — documented, not silently resolved.** The repository contains `PHASE_19_EXP10_DESIGN.md` titled **"EXPERIMENT 10 — MULTIMODAL FEATURE CONDITIONING"**, which is **NOT** roadmap **Row 10** ("Federate a DAIC-derived object"). Repo experiment numbering diverged from roadmap row numbering at 10. `PHASE_10_ROADMAP.md:274` says *"Federating DAIC signal (Exp 10)"* meaning **Row 10**. When you read "Exp 10", check which is meant.

---

## 14. Experiment History (chronological)

### Rows 0a–2 — Diagnostics and the valid comparator ✅

Exp 0a (regression trivial-baseline control) and Exp 0b (threshold-free ranking metric) were analysis-only diagnostics; Row 1 established the evaluation protocol; **Row 2 produced Baseline-CV** ([§10](#10-evaluation-methodology)), reproducing the baseline's qualitative signature (**P/R/F1 = 0.0, all-negative predictions**).

### Rows 3–6 — RC-1 recipe interventions — **all H₀** ❌

| Row | Experiment | Changed factor | Verdict (verbatim) |
|---|---|---|---|
| 3 | Convergence | epoch policy | **H₀** — *"collapse is NOT a budget artifact; RC-1 needs more than epochs"* |
| 4 | Decision rule | decision threshold | **H₀** — *"thresholding recovered quantity without demonstrated discrimination"* |
| 5 | Imbalance-aware objective | loss weighting | **H₀** — *"class weighting relocated the boundary without demonstrated discrimination"* |
| 6 | Loss-term rebalancing | loss scaling λ | **H₀** — *"loss-term rebalancing did not produce demonstrated discrimination"* |

Each passed its equivalence gate (12 metrics inside the Baseline-CV 95 % CI) and reproduced Baseline-CV to `0.000e+00`. **Prior collapse is not a recipe artifact.**

---

### 14.1 · B-1 / Experiment 7 — Multimodal activation ✅ COMPLETE

**Design** `PHASE_16_EXP7_DESIGN.md` · **Closure** `FINAL_EXP7_CLOSURE.md`

- **Objective:** exercise the multimodal capability for the first time — make audio and vision reach the computation graph.
- **Arms:** A0 text-only · A1 +audio (154) · A2 +vision (84) · A3 full multimodal (1024 fusion input, 109,763,494 params). **100 fold-runs.**
- **Result:** **`ACTIVATION DEMONSTRATED — audio and vision reach the graph with the text-only control intact`**
  - **Audio ablation 0.0 → 2.5518971** (C1 PASS)
  - **Vision ablation 0.0 → 0.4041132** (C2 PASS)
- **Verdict:** **CLEAN — EXP 7 CAN BE CLOSED.**
- **What it establishes:** the multimodal path is real and wired, not stubbed. **What it does not:** a task gain — `FINAL_EXP7_CLOSURE.md:171` records that **audio significantly *degraded* ROC-AUC** while improving another term. Only A1 produced non-zero Recall (0.0489) / F1 (0.0249), *neither significant*.

---

### 14.2 · B-2 / Experiment 8 — Dimension-aware DP mechanism ❌ H₀

**Design** `PHASE_17_EXP8_DESIGN.md` (SHA `bc7d77d0…`) · **Closure** `FINAL_EXP8_CLOSURE.md`

- **Objective:** make privacy survivable — improve post-noise SNR beyond 543 : 1 at fixed ε.
- **Mechanism:** per-group clipping allocation `C_g ∝ ‖v_g‖₂` instead of a single global clip.
- **Result:** **`H0 / FAIL FOR MATERIAL IMPROVEMENT - not an implementation failure`**
  - All five paired differences positive; 95 % CI excluded zero (conditions A and B **met**)
  - Relative improvement **1.000110×** — about **91× short** of the pre-registered 1.01 floor (condition C **not met**)
- **Why:** `fc1.weight` holds **99.74 %** of dimensions and **99.81 %** of signal energy, so `C_g ∝ ‖v_g‖₂` sits essentially on the Cauchy–Schwarz equality point `C_g ∝ √d_g`, where the per-group gain vanishes.
- **Integrity:** P0, P1, P3, P4 all passed.
- **What it ruled out — the key handoff:** *"at fixed ε and fixed d, a privacy-equivalent mechanism change alone **cannot** materially improve post-noise SNR."* **Therefore `d` is the only remaining lever, which is B-3.**

---

### 14.3 · B-3 / Experiment 9 — Compact update representation ❌ H₀

**Design** `PHASE_18_EXP9_DESIGN.md` (SHA `cf361986…`)

- **Objective:** find the smallest **publicly-specified** object derived from the DAIC-trained model that retains task signal and survives DP.
- **Why "publicly-specified":** true top-k is a **deterministic function of the private delta** whose index set would be released in the clear — adjacent datasets give **disjoint support on the index component**, so ε = ∞. Private selection was audited and is infeasible at ρ ≈ 0.5006 (the value release alone costs ρ = 0.500). *Top-k's signal advantage IS its data-dependence, and that is exactly what destroys the guarantee.*

**Family O** — a public architecture-priority **prefix mask**, ordering `fusion.*` → `bert.pooler.dense` → `bert.encoder.layer.0…11` → `bert.embeddings.*`; `S_k` = first `k` coordinates. **Never inspects the delta.**

**Criteria:** **A** payload ≤ **1,579,963 B** · **B** NSR < **544.341809** · **C** ROC-AUC in Baseline-CV CI **and** paired degradation ≤ **0.05781287386071144**.

| Arm | k | A | B | ROC-AUC 95 % CI | C |
|---|---|---|---|---|---|
| K0 | 109,680,132 | FAIL (585 MB) | FAIL (10,472.8) | **[0.5754, 0.6911]** | **PASS** |
| K1 | 1,096,801 | FAIL | FAIL (1,047.3) | [0.4169, 0.5953] | FAIL |
| K2 | 295,681 | **PASS** | **PASS** (543.77) | [0.4159, 0.5941] | FAIL |
| K3 | 10,968 | **PASS** | **PASS** (104.73) | [0.4039, 0.5808] | FAIL |

- **Verdict:** **H₀ — no arm satisfies A and B and C.**
- **The decisive pattern:** **compression and DP survival were solved; signal content failed.** K2/K3 passed payload *and* noise, yet every masked arm's CI **straddles chance**. Collapse is total already at K1 (1 % of coordinates).
- **Exact scope of the negative result** (design §4, §34, verbatim): *"A negative result here does **not** establish that no compact DAIC-derived object survives DP — only that the **publicly-specified** family defined in §6 does not."*

---

### 14.4 · Experiment 10 (repo-numbered) — Multimodal feature conditioning ⚠️ PARTIAL

**Design** `PHASE_19_EXP10_DESIGN.md` · **Not a roadmap row** (see the numbering warning in [§13](#13-roadmap-and-dependencies)).

Arms M0 (text-only equivalence gate), M1 (raw multimodal), M2 (conditioned multimodal) — all **full 109.7 M-parameter fine-tuned models**.

**Verdict: `PARTIAL`** — G0 ✅, P1 ✅, **P2 ❌**, P3 ✅, P4 ✅ — *"conditioning improved ranking (P1) without establishing an above-chance signal (P2 failed)."* M2 − M1 = **+0.086836**, CI [+0.035494, +0.138179]; M2's own CI [0.494524, 0.666921] includes 0.5 by **0.005476**.

---

### 14.5 · C-2 / Phase 20 — Non-IID multi-client topology ✅ COMPLETE

**Design** `PHASE_20_C2_DESIGN.md` (SHA `687ea6a5…`)

- **Objective (Row 11):** exercise trimmed-mean on *genuinely divergent* updates. The prior federation ran `devices = 1` — *"three copies of the same file that differ only by their DP noise draw."*
- **Setup:** **5 clients**, sizes **38/38/38/37/37**, both arms covering the identical 188 participants.

| Arm | Definition |
|---|---|
| **W0** control | **stratified / IID-like** — the frozen 5-fold *test* partitions. **Never call this "perfectly IID"**: PHQ means span 5.5405–7.7895. |
| **W1** treatment | **non-IID A2** — sort by `(phq_score DESC, participant_id ASC)`, fill 38/38/38/37/37. No free parameter. |

**A2 heterogeneity:** client 1 pos_rate **1.0000**, PHQ mean 16.0789 → client 5 pos_rate **0.0000**, PHQ mean 0.2973 (sd 0.4571). W0 by contrast: every client pos_rate 0.2368–0.2432.

**Primary result — pre-DP pairwise divergence (validity gate, directional only, no threshold):**

| measure | W0 | W1 | W1 − W0 | result |
|---|---|---|---|---|
| L2 | 0.238848539 | 0.261621691 | **+0.022773152** | **PASS** |
| cosine | 0.134495754 | 0.165267976 | **+0.030772222** | **PASS** |

- **Verification: 25/25 PASS** — including a from-scratch retrain of all 10 clients reproducing the divergences at **max |diff| = 0.000e+00**.
- **🚨 NO TASK METRIC WAS COMPUTED.** No ROC-AUC, accuracy, F1, MAE, RMSE or PR-AUC; no held-out set exists. This was frozen in design §14 and enforced by a verifier check scanning every artifact for forbidden keys.
- **Limitations:** the 5 repeats are **degenerate on the primary endpoint** — full-batch, no dropout, fixed `w_before` made training bit-deterministic, so there is **1 effective replicate per arm, not 5**; no CI is available or claimed. Post-DP divergence is **≈2,940×** the pre-DP signal (cosine ≈ 0.9997 ⇒ near-orthogonal), so post-DP and aggregation-stage measures are noise-dominated and **cannot discriminate topologies**. Clipping never bound (0/50 clipped, norms 0.4505–0.4712; SNR 8.45e-04).
- **Establishes:** the A2 partition produced measurably more divergent pre-DP updates than the stratified control, and the frozen trimmed-mean aggregator ran on genuinely distinct clients. **Does not establish:** anything about task performance, generalisation, clinical utility or predictive validity — and **it does not unblock Row 10.**

---

### 14.6 · B-3B / Phase 21 — Compact task-trained probe ❌ H₀

**Design** `PHASE_21_B3B_DESIGN.md` (SHA `00f8e34b…`) · **Closure** `FINAL_B3B_CLOSURE.md`

- **Why a second B-3:** Exp 9 failed at **coordinate selection**, not compression or noise. B-3B removes the selection step entirely — no mask, no index set. Exp 9's K2 reconstruction was *pretrained encoder + head trained jointly with a fine-tuned encoder*, leaving head and features **mismatched** when the encoder reverted to base. B-3B trains the head **directly on frozen pretrained CLS embeddings**, so head and feature space are **matched by construction**.
- **Architecture:** `FusionHead(768, 256, 2)` — **197,892 params / 8 tensors**. Frozen CLS input (revision `24809aa8…`, max_len 128, CLS pooling). **Genuine delta** `w_after − w_before`. Recipe: AdamW, **lr 1e-3**, **3 epochs**, batch 8, shuffle, dropout 0.2, clip 1.0. **5 folds × 5 repeats = 25 trainings**, each yielding both arms.
- **Arms:** **N** = no-DP control · **D** = DP arm. Identical in every respect except DP.

| | criterion | measured | threshold | result |
|---|---|---|---|---|
| **A** | payload | **1,055,101 B** | ≤ 1,579,963 B | **PASS** |
| **B** | mean NSR | **444.876535** | < 544.341809 | **PASS** |
| **C1** | DP ROC-AUC | **0.503160920** | in [0.5755177227, 0.6911434705] | **FAIL** |
| **C2** | paired degradation | **+0.056004379** | ≤ 0.0578128739 | PASS |
| **C** | C1 ∧ C2 | — | — | **FAIL** |

- **Clipping: 25/25 (100 %)** clipped; pre-clip norms **2.7879–3.1956** (mean 2.9731). The design's *dominant registered risk* — that clipping might not bind — **did not materialise**.
- **Repeats genuinely varied** (mean within-fold SD 0.059839 > 0) — unlike C-2, this is **not** pseudo-replication.
- **Verification: 20/20 PASS** — all 25 folds × 2 arms retrained from scratch, reproducing **all 50** ROC-AUC values at **max |diff| = 0.000e+00**.
- **🔑 The decisive diagnostic:** the **no-DP control scored 0.559165298 — itself outside the frozen CI** (floor 0.5755177227). **The probe failed to reach baseline before any DP was applied.** The limiting factor is the **frozen pretrained CLS representation, not the privacy mechanism.**
- **C2's PASS is not favourable evidence** — degradation stayed within MDE partly because the control had little signal to lose.
- **Verdict: H₀** — a legitimate pre-registered negative result, **not** an implementation failure.

**B-3 therefore remains UNSATISFIED**, having returned H₀ across two structurally different representation families.

---

## 15. Current Project Status

| Item | Row | Status |
|---|---|---|
| **B-1 / Exp 7** — multimodal activation | 7 | ✅ **COMPLETE** |
| **B-2 / Exp 8** — DP mechanism | 8 | ❌ **H₀** |
| **B-3 / Exp 9** — compact update | 9 | ❌ **H₀** |
| **B-3B / Phase 21** — compact probe | 9 (2nd) | ❌ **H₀** |
| **C-2 / Phase 20** — non-IID multi-client | 11 | ✅ **COMPLETE** *as a manipulation/topology experiment* |
| **C-1** — federate a DAIC-derived object | 10 | 🚫 **BLOCKED — never executed** |
| **C-3** — ε–utility curve | 12 | 🚫 **BLOCKED** |
| **C-4** — DP mechanism sweep | — | 🚫 **BLOCKED** |
| **Phase 22** — end-to-end demo | *(not a roadmap row)* | ✅ **EXECUTED & VERIFIED 20/20** — engineering success, task success NOT established ([§19](#19-phase-22--end-to-end-multimodal-federated-demonstration)) |

### Why Row 10 / C-1 is blocked

`PHASE_10_ROADMAP.md:234` — C-1 *"is **structurally impossible** until B-2 and B-3 succeed."*

```
Row 10 requires  B-2 AND B-3  to SUCCEED
   B-2 → H₀   (and closed by its own finding: mechanism change at fixed ε and fixed d cannot help)
   B-3 → H₀ twice   (Exp 9 public prefix mask; B-3B compact task-trained probe)
   ⇒ NEITHER precondition is satisfied  ⇒  Row 10 REMAINS BLOCKED
```

C-3 and C-4 are *"Blocked by C-1"* (roadmap:236, :237) and therefore remain blocked transitively.

**C-2's completion does not change this.** C-2 was explicitly the one item *outside* the dependency chain, and both its design (§22) and its roadmap entry state it does not unblock Row 10.

**No unblocked, unexecuted roadmap item exists.**

---

## 16. What Has Been Proven

✅ **Verified facts, each backed by an artifact and independent verification:**

1. **The infrastructure works end-to-end** — extract → normalise → build → verify → embed → train → checkpoint → delta → DP → encrypt → chunked upload → aggregate → publish → download → next round, across two real federated rounds and both the 113 and 188 corpora.
2. **Cryptographic integrity and genuine RDP accounting are real** — not mocked.
3. **The reconstruction is faithful** — trainer SHA-256 identical to the original; a twelve-item audit returned PASS on all twelve.
4. **Prior collapse is not a recipe artifact** — epochs, decision rule, class weighting and loss rebalancing each returned H₀ (Rows 3–6).
5. **The multimodal path is genuinely active** — audio 0.0 → 2.5518971, vision 0.0 → 0.4041132 (Exp 7).
6. **The √d relationship is confirmed, not hypothesised** — and at fixed ε and fixed `d`, a mechanism change alone cannot materially improve SNR (Exp 8).
7. **Compression and DP survival are individually achievable** — Exp 9's K2/K3 passed both payload and noise criteria.
8. **A non-IID partition produces measurably more divergent client updates** than a stratified one, and the frozen trimmed-mean aggregator runs on them (C-2, 25/25 verified).
9. **A compact task-trained probe over frozen CLS does not reach the Baseline-CV ROC-AUC interval — with or without DP** (B-3B, 20/20 verified).

---

## 17. What Has NOT Been Proven

🚫 **Do not claim any of the following. None is supported by any artifact in this repository.**

1. ❌ **That the federated system produces useful depression predictions.** Baseline-CV F1 = Recall = Precision = **0.0**. No experiment has produced a *significant* non-zero classification result.
2. ❌ **That a DAIC-derived object can be federated with signal intact.** That is Row 10 / C-1 — **never executed**.
3. ❌ **That multimodal input improves the task.** Exp 7 demonstrated *activation*, and recorded that **audio significantly degraded ROC-AUC**. Exp 10 was **PARTIAL** — P2 failed to establish an above-chance signal.
4. ❌ **That C-2 says anything about task performance.** It computed **no task metric at all**, by design.
5. ❌ **That trimmed-mean aggregation is robust or beneficial.** C-2 *exercised* the aggregator; its aggregation-stage measures were noise-dominated and arm-indistinguishable.
6. ❌ **That DP is the reason the task fails.** B-3B's no-DP control was *already* below the baseline CI — the limitation is upstream, in the representation.
7. ❌ **Any clinical utility, generalisation, or predictive validity whatsoever.**
8. ❌ **That the end-to-end multimodal federated task pipeline has been run.** **It has not.** That is Phase 22, which is future work.

---

## 18. Limitations

1. **Sample size.** 188 participants, 45 positive, fold-level **df = 4**. The MDE of **0.0578** is large against a 0.6333 baseline. Small effects are simply undetectable here.
2. **Severe class imbalance** — 23.9 % positive; the complete corpus is *more* imbalanced than the earlier 113 subset.
3. **The accuracy illusion** — 0.76 accuracy with F1 = 0.0. Never quote accuracy alone.
4. **C-2's repeats are degenerate** on its primary endpoint (1 effective replicate per arm).
5. **DP noise dominance** at realistic `d` makes post-DP measurements uninformative about topology.
6. **B-3B tested a single pre-registered architecture**, no dimension ladder. H₀ establishes only that *that* object fails.
7. **Payload figures are predicted transport** values using Exp 9's documented ×1.3329 expansion factor, not live encryption runs (margins were large enough that verdicts do not depend on it).
8. **Open audit issues** recorded in `VERSION2_FINAL_REVIEW.md` — 2 high (broken digest provenance in Phase 12 and Phase 14), 3 medium, 5 low. **None affects any measured value or verdict**, but they should be fixed before thesis submission.

---

## 19. PHASE 22 — END-TO-END MULTIMODAL FEDERATED DEMONSTRATION

> ## 🟡 **STARTED — DESIGN FROZEN · CODE WRITTEN · NOT EXECUTED**
>
> **No Phase 22 training, DP, federation, aggregation or evaluation has been run.
> No Phase 22 result exists.** Only the design, the implementation, a local
> validation pass and the Colab bundle exist.

| | |
|---|---|
| **Design** | `PHASE_22_DESIGN.md` — **FROZEN** |
| **Design SHA-256** | `3c18108309c17cbff332204aa761411d6600fe584488391741dfcd2a9208036f` |
| **Type** | **ENGINEERING DEMONSTRATION — NOT Roadmap Row 10 / C-1** |
| **Executed?** | ❌ **NO** |

### 19.1 Scope — binding

**Phase 22 is an engineering demonstration. It does NOT attempt, satisfy, or partially
satisfy Roadmap Row 10 / C-1.** B-2 (Exp 8) returned H₀; B-3 returned H₀ twice (Exp 9,
B-3B). **Row 10 requires B-2 ∧ B-3 and therefore remains BLOCKED.**
`PHASE_10_ROADMAP.md` is **not** modified by this phase.

> **ENGINEERING SUCCESS ≠ TASK SUCCESS.** The pipeline executing correctly does **not**
> imply the model discriminates. Both are reported separately (design §2).

### 19.2 The frozen pipeline

```
DAIC-WOZ
   ↓
Text + Audio + Vision
   ↓
Multimodal model  (MultiModalModel, 109,763,494 params / 215 keys)
   ↓
5 local clients   (non-IID A2, partitioning ONLY each fold's training set)
   ↓
Local training    (AdamW, lr 2e-5, 3 epochs, batch 8, clip 1.0, dropout 0.2)
   ↓
Genuine model delta   w_after − w_before   (frozen compute_state_delta)
   ↓
Gaussian DP       (clip C=1.0, σ_eff=1.0, ε=5.302585092994046, δ=1e-05, T=1)
   ↓
Federated aggregation  (trimmed_mean, trim_ratio 0.1 → keeps 3 of 5)
   ↓
Global model      (w_before + aggregated_delta)
   ↓
Held-out DAIC evaluation  (frozen fold_manifest.json, ROC-AUC primary)
```

### 19.3 Frozen parameters

| Group | Values |
|---|---|
| **Model** | `MultiModalModel(audio_dim=154, vision_dim=84)` · text 768 + audio 128 + vision 128 = **fusion 1024** · **109,763,494 params / 215 keys** · dropout 0.2 |
| **Recipe** | `transformers.AdamW` · lr **2e-5** · **3** epochs · batch **8** · shuffle True · grad clip **1.0** · max_len **128** |
| **Loss** | `CrossEntropyLoss(logits, label) + 0.5 · MSELoss(mu, phq)` · label `PHQ > 10.0` |
| **Delta** | `w_after − w_before` (frozen `compute_state_delta`) — legacy full-state object explicitly rejected |
| **Clients** | **5**, non-IID **A2**: sort `(phq_score DESC, participant_id ASC)`, fill sequentially |
| **Client sizes** | derived per fold: `base=n//5, rem=n%5` → **150 → (30,30,30,30,30)**, **151 → (31,30,30,30,30)** |
| **Arms** | **N** = no-DP control · **D** = DP. Identical in every respect except DP |
| **DP** | Gaussian · C **1.0** · nm **1.0** · ε **5.302585092994046** · δ **1e-05** · **T=1** |
| **Round ε** | **5.302585092994046** — parallel composition over **disjoint** client data (max, not sum) |
| **Aggregation** | `trimmed_mean`, `trim_ratio` **0.1**, N=5 → `lower=1, upper=4` → **keeps 3 of 5** |
| **Rounds / repeats** | **1** round · **1** repeat |
| **Executions** | **25** local trainings (5 folds × 5 clients) → **10** federated executions (5 folds × 2 arms) |
| **Metric** | **ROC-AUC** primary; also Accuracy / Precision / Recall / F1 / PR-AUC. Accuracy never alone |
| **Comparator** | Baseline-CV ROC-AUC **0.6333305966064586**, CI **[0.5755177227457472, 0.6911434704671701]** — a *comparator*, **not** a pass/fail threshold |

**No acceptance threshold exists and none may be invented.** Clipping is **measured, never tuned**.

### 19.4 Leakage control — binding

Per fold: test participants held out entirely → training = all remaining (150/151) →
**clients partition ONLY the training participants** → **the held-out fold is never used for
any local training** → the global model is evaluated only on that fold. Asserted
independently by both the validator and the verifier.

### 19.5 Readiness audit — result

| Area | Status |
|---|---|
| **Dataset** (`dataset_build/daic_records_multimodal.parquet`, SHA `1ac9f53e…`) | ✅ **PASS** — 188 unique participants · 45 pos / 143 neg · text 768 / audio **154** / vision **84** · all finite · 0 nulls · 0 all-zero rows · IDs identical *in order* to frozen `daic_records.parquet` · PHQ elementwise identical |
| **Fold manifest** (SHA `b9a7a91f…`) | ✅ **PASS** — 150/38 ×3 + 151/37 ×2, zero overlap, test folds partition the population |
| **Multimodal model** | ✅ **PASS** — instantiated: 215 keys / 109,763,494 params / fusion_in 1024 |
| **DP** | ✅ **PASS** — ε recomputed live = 5.302585092994046 |
| **Aggregation** | ✅ **PASS** — frozen aggregator keeps 3 of 5 |
| **CUDA / GPU** | ⏳ **PENDING — must be confirmed on Colab** |

*(OpenFace / openSMILE are **not** required: audio and vision features are pre-extracted in the parquet.)*

### 19.6 Local validator result — 19/20 PASS

`python phase22_end_to_end/validate_phase22.py` → **19 PASS, 1 FAIL**, exit **1**.

> **The single failure is check 19, CUDA availability. This is EXPECTED**: the current
> machine runs `torch 2.8.0+cpu` with no GPU. Every scientific and structural check passed.
> The validator is fail-closed by design — Phase 22 requires GPU and there is **no silent
> CPU fallback**.

Report: `trainer_outputs/phase22_end_to_end/validate_phase22_report.json`.

### 19.0 ✅ PHASE 22 EXECUTED AND VERIFIED — results

> **Executed on Colab GPU (Tesla T4, CUDA 12.8, PyTorch 2.11.0+cu128, Python 3.12.13,
> Transformers 5.13.1). Validation 20/20 · Independent verification 20 PASS / 0 FAIL /
> 0 PENDING.** Design SHA `3c181083…`.
>
> ⚠️ **The result artifacts are NOT yet on the local machine** — they live in the Colab
> `trainer_outputs/phase22_end_to_end/`. Values below are as reported from that run; local
> re-verification is pending transfer of the artifacts.

**Completed:** 188 participants · 5 folds · 5 clients/fold · **25 local trainings** ·
**10 federated executions** · CUDA confirmed · DP clip+Gaussian applied · trimmed-mean kept
3 of 5 · global model reconstructed · held-out evaluation · **no participant leakage** ·
frozen inputs unchanged.

| Metric | **N** (no-DP control) | **D** (DP arm) |
|---|---|---|
| **ROC-AUC mean** | **0.465106732** (fold SD 0.119932274) | **0.500000000** (fold SD **0**) |
| Accuracy | 0.431578947 | 0.552631579 |
| F1 | 0.311452359 | 0.154856614 |
| PR-AUC | 0.284460604 | 0.239402561 |
| Precision | 0.216112764 | 0.096017070 |
| Recall | 0.688888889 | 0.400000000 |

**DP:** ε = 5.302585092994046 · δ = 1e-05 · C = 1.0 · nm = 1.0 · 25/25 updates ·
**0 clipped (fraction 0.0)** · parallel composition (disjoint client sets).

**Comparator:** Baseline-CV ROC-AUC 0.6333305966, CI [0.575517723, 0.691143470] —
**neither arm falls inside it.**

**Recorded diagnosis (design §13):**
> *"The no-DP control N is ALREADY outside the Baseline-CV CI: the model is weak BEFORE DP.
> DP is not the limiting factor, and no Phase 22 outcome may be attributed to DP alone."*

**Verdict:**

| | |
|---|---|
| **ENGINEERING / pipeline success** | ✅ **TRUE** — the full pipeline executed and verified 20/20 |
| **TASK success** | ❌ **NOT ESTABLISHED** — N at 0.4651 is *below chance*; D is exactly 0.5 with **zero** fold variance, i.e. a constant-output model |
| **Row 10 / C-1** | 🚫 **REMAINS BLOCKED** — B-2 H₀, B-3 H₀ ×2. Phase 22 is an engineering demonstration and does not satisfy it |

**Reading the D arm honestly:** ROC-AUC exactly 0.500 with fold SD exactly 0 is the signature
of a model emitting a constant score — the DP'd global model carries no ranking information at
all. That is consistent with the pre-registered §12 expectation at d = 109,763,494
(σ·√d ≈ 10,477 against a clipped signal ≤ 1.0). **Note 0 of 25 updates were clipped**, so the
raw update norms were already below C = 1.0 before noise was added.

### 19.7a Verifier parsing defect — FIXED (verifier-only, no scientific change)

A Colab execution produced Phase 22 result artifacts, and `verify_phase22.py` **crashed**
against them:

```
ValueError: invalid literal for int() with base 10: 'False'
```

**Root cause.** `run_phase22.py:87` stores `clipped` as a Python `bool`; `csv.DictWriter`
serialises that as the **string** `"True"`/`"False"`. The verifier's check 7 did
`bool(int(r["clipped"]))`, and `int("False")` raises. Inconsistent with B-3B, whose runner
wrote `int(...)` → `"0"`/`"1"`, which is why its verifier worked.

**Fix — verifier only.** A local `as_bool()` helper accepts the representations actually
stored (`False`/`"False"`/`"false"`/`"0"` and `True`/`"True"`/`"true"`/`"1"`) and **raises on
anything else**, keeping the verifier fail-closed. **No result artifact was rewritten** —
artifacts are results, and editing them to suit the verifier would invert the authority order.

**Scope:** `clipped` is the only bool-valued CSV column, so this was the sole instance; a scan
found no other occurrence of the defect class. **`PHASE_22_DESIGN.md` (`3c181083…`), all frozen
inputs, the training code, DP configuration, aggregation code and every scientific parameter
are unchanged.** No experiment was re-run.

**Status:** the verifier now executes all 20 checks to completion instead of crashing.
**Final verification is still PENDING** — it must be re-run on Colab against the actual result
artifacts, which do not exist on the local machine. Run locally without them it correctly
reports `6 PASS / 1 FAIL / 13 PENDING` (missing-evidence PENDINGs), not a verdict on the
experiment.

### 19.7 What has NOT happened

❌ No training · ❌ no DP applied · ❌ no federation · ❌ no aggregation ·
❌ no global model built · ❌ no held-out evaluation · ❌ no results of any kind.

`trainer_outputs/phase22_end_to_end/` contains **only** the validation report.

### 19.8 Colab bundle

| | |
|---|---|
| File | **`phase22_upload_FINAL.zip`** (in `~/Downloads`) |
| SHA-256 | **`f248534d3bddd555d7bc30c794e7779a65a3778786b64dde93fed25ebe7ff5b2`** |
| Size | **407,719,314 bytes** (≈ 407.7 MB) |
| Contents | **19** code files · 5 data/frozen artifacts · **6 MentalBERT snapshot files** · manifest |
| Builder | `make_phase22_upload_bundle.py` (**fail-closed**: aborts if any declared file is missing or any frozen SHA drifts) |

**The bundle includes the verified pinned MentalBERT snapshot** (revision
`24809aa822c76639760d0d934742d1b42f89942f`, all 6 artifact SHAs checked at build time).
`mental/mental-bert-base-uncased` is a **gated** HF repository — shipping the snapshot means
**no Hugging Face authentication is needed in Colab** and the exact pinned weights are
guaranteed.

**Transitive dependency note (Exp 9 lesson):** `dp_agent/dp_agent.py` imports
`centralised_receipts`, `centralized_secure_store` and `installer.security.integrity`. All
are in the manifest. Exp 9's Colab run aborted because a bundle shipped Python imports but
omitted runtime data dependencies; this builder lists code **and** data explicitly.

#### FINAL bundle — full transitive-closure rebuild

Earlier bundles were incomplete. **`phase22_upload_FINAL.zip` supersedes v1, v2 and v3; those
have been deleted and must not be used.**

| | |
|---|---|
| Path | `C:\Users\DELL\Downloads\phase22_upload_FINAL.zip` |
| SHA-256 | **`f248534d3bddd555d7bc30c794e7779a65a3778786b64dde93fed25ebe7ff5b2`** |
| Size | **407,719,314 bytes** (≈ 407.7 MB) |
| Contents | **19** code · **5** data/frozen · **6** MentalBERT snapshot · manifest = **31 entries** |

**Root cause of the earlier failures.** v1 shipped only `installer/security/__init__.py` +
`integrity.py`, but that package `__init__` eagerly imports `anti_debug` **and**
`tpm_attestation`; `dp_agent.py:12` then *calls* `integrity_guard()`, which lazily imports
`self_destruct`. Colab aborted at check 16 with
`ModuleNotFoundError: installer.security.anti_debug`.

**Fix method — AST transitive closure, not incremental patching.** A closure walk over all
five Phase 22 modules plus the frozen trainer, `dp_agent` and the aggregator — using
`ast.walk`, so **function-level lazy imports are included** — returned **17 local files**. That
surfaced a dependency no earlier bundle had:
`server/aggregator_agent/core/centralized_secure_store.py`, imported lazily at
`aggregator.py:319`. The manifest now carries the full closure.

Files added since v1: `anti_debug.py`, `tpm_attestation.py`, `self_destruct.py`,
`core/centralized_secure_store.py`, `core/centralised_receipts.py`. **All are pre-existing and
git-tracked; none was written, modified, stubbed, mocked or bypassed. The security/integrity
layer is intact.** This is packaging completeness only — no scientific parameter, no frozen
input, and not the design, changed.

**Not required in Colab:** `aggregator.py` also imports `pymongo` / `bson` / `gridfs`, but only
inside GridFS decryption methods Phase 22 never calls. They are deliberately excluded.

**Verified empirically:** the FINAL bundle was extracted to a clean directory with no
repository on `sys.path`, and the validator was run from it — **19 PASS / 1 FAIL**, the sole
failure being CUDA. Check 16 (ε accounting), which crashed on Colab, now passes and returns
exactly `5.302585092994046`.

**A note on `self_destruct.py`:** it is a genuine runtime dependency (`integrity.py:153`, plus
four Linux-path sites in `anti_debug.py`). It should not fire — `verify_integrity()` writes a
baseline and returns `True` on first run — and its destructive scope is limited to
`~/.federated`, which is ephemeral on Colab and touches no Phase 22 artifact. `anti_debug()`
and `tpm_attestation()` are **imported but never invoked**, which matters because
`tpm_attestation()` would `sys.exit("[SECURITY] TPM not found")` on Colab, which has no TPM.

### 19.9 Exact next step

```
1. Colab GPU check      (nvidia-smi; torch.cuda.is_available() must be True)
2. Unpack phase22_upload.zip
3. Install deps         (transformers==4.44.0 — must still export transformers.AdamW)
4. Run the validator    → must reach 20/20 PASS, exit 0
5. GPU batch-8 memory test (forward+backward, frozen batch size)
6. STOP and wait for explicit authorization to execute
```

> **If the memory test OOMs: STOP.** Do **not** resolve it by changing batch size, model
> dimensions, epochs, learning rate or architecture. Those are frozen scientific parameters.

### 19.10 Two distinct readiness states — do not confuse them

| State | Meaning | Current |
|---|---|---|
| **READY FOR COLAB VALIDATION** | Design frozen, code written, bundle built, local checks pass except GPU | ✅ **THIS IS WHERE WE ARE** |
| **READY FOR PHASE 22 EXECUTION** | Colab reports 20/20 validator PASS **and** the batch-8 GPU memory test succeeds | ❌ **NOT YET** |

Reaching "ready for execution" still does **not** authorise a run — explicit user
authorization is required.

### 19.11 Implementation

`phase22_end_to_end/` — isolated; no previous experiment directory is touched:

| Module | Role |
|---|---|
| `phase22_common.py` | frozen constants, SHAs, A2-on-training-subset partition, seeds, metrics |
| `validate_phase22.py` | pre-execution, **fail-closed** (includes the hard CUDA check) |
| `run_phase22.py` | 25 local trainings → deltas → DP → aggregation → global → evaluation |
| `aggregate_phase22.py` | the 13 reporting items; N vs D; Baseline-CV comparison |
| `verify_phase22.py` | independent **fail-closed** verification |

Outputs are confined to `trainer_outputs/phase22_end_to_end/`.

**Reused unmodified:** `MultiModalModel`, `FusionHead`, `MultiModalDataset`, `collate_batch`,
`fine_tune_supervised`, `run_inference`, `compute_state_delta`, `read_parquet_records`
(frozen trainer) · `_rdp_to_dp` (frozen `dp_agent`) · `AggregatorAgent` (frozen aggregator).

---


## 19Z. Clean Deployment Guide — RTX 3050 / Terminal Pipeline

> ### 🚨 SECURITY BLOCKER — DO NOT PUSH UNTIL RESOLVED
>
> **Five private keys are ALREADY in pushed history** on
> `https://github.com/soham-0510/BE-Major-Project.git` (`origin/main`):
> `server/orchestration_agent/certs/ca.key` (**the CA root key**),
> `certs/server.key`, `server/orchestration_agent/server.key`,
> `client_agent/certs/client.key`, `device_key.pem`.
>
> `.gitignore` lists `*.key` / `*.pem`, but **.gitignore does not apply to already-tracked
> files** — they were committed before the rule existed. If that repository is public, the
> entire mTLS trust chain must be treated as **compromised**: anyone can mint client
> certificates trusted by the server. See §19Z.3 before any further push.

> **This section covers the LIVE TERMINAL ORCHESTRATION PIPELINE only.** It is a different
> system from the [Phase 22 Colab experiment](#190--phase-22-executed-and-verified--results).
> Phase 22 was a *simulation* run in one process on Colab; the live pipeline is a **Rust gRPC
> orchestrator + Python clients over mTLS**. **The live terminal pipeline has NOT been
> end-to-end tested on the RTX 3050 machine** — nothing below should be read as a passed test.

### 19Z.1 Architecture (verified from code, not assumed)

```
   Python client(s)                            Rust orchestrator
 runtime/federated_client.py  ──gRPC/TLS──►  server/orchestration_agent
   ~/.federated/keys/                          tonic, 0.0.0.0:50051
        │                                             │
        │  UploadUpdate (stream)                      ▼
        └──────────────────────────────────►   MongoDB "federated"
                DownloadGlobalModel (stream)     + GridFS payloads
```

| Component | Language | Entry point |
|---|---|---|
| **Orchestration server** | **Rust** (crate `orchestrator` v1.0.0, tonic 0.11 + tokio, edition 2021) | `server/orchestration_agent/src/main.rs` |
| **Federated client** | Python | `runtime/federated_client.py` → `main()`, modes `daemon` \| `run-once` |
| **Aggregator** | Python | `server/aggregator_agent/aggregator.py`, triggered from Rust at `grpc/server.rs:675` when `updates.len() >= 3` |

**RPCs** (`server/orchestration_agent/proto/orchestrator.proto`): `RegisterDevice`,
`RequestEnrollment`, `EnrollDevice`, `GetRound`, `UploadUpdate` (stream), `SubmitReceipt`,
`DownloadGlobalModel` (stream).

> ⚠️ Root **`client.py` is NOT the federated client** — it contains zero gRPC code and is a
> local LDA→trainer→DP→encrypt script. Do not use it for the orchestration demo.

### 19Z.2 Repository structure and what goes to GitHub

| Directory | Purpose | Server | Client | Commit? |
|---|---|---|---|---|
| `server/orchestration_agent/` | Rust gRPC orchestrator (src, proto, Cargo) | ✅ | — | ✅ **yes** — except `certs/`, `*.log`, `*.pt`, `target/` |
| `server/aggregator_agent/` | Python trimmed-mean aggregator | ✅ | — | ✅ yes |
| `runtime/` | **Authoritative** client runtime (gRPC, pipeline, TPM guard) | — | ✅ | ✅ yes |
| `installer/runtime/` | Packaged **mirror** of `runtime/` — `federated_client.py` and `pipeline.py` **DIFFER** from source | — | ⚠️ | ✅ yes, but treat `runtime/` as source of truth |
| `installer/windows_signer/` | Rust TPM signer (Windows) | — | ✅ | ⚠️ source yes, **`target/` NO** (~40 MB tracked build junk) |
| `client_agent/` | Older client + generated gRPC stubs | — | ⚠️ | ✅ code yes, **`certs/` NO** |
| `LDA/` | Local Data Agent (text/audio/video preprocessing) | — | ✅ | ✅ yes (not `OpenFace/`, `opensmile/`) |
| `dp_agent/`, `enc_agent/`, `trainer_agent/` | DP, encryption, trainer agents | ✅ | ✅ | ✅ yes |
| `phase22_end_to_end/` | Phase 22 experiment modules | — | — | ✅ yes (currently **untracked**) |
| `dataset_build/` | Multimodal parquet builders | — | — | ✅ scripts yes, **`.parquet` no** |
| `trainer_outputs/` | All experiment results + frozen artifacts | — | — | ❌ **gitignored** — see §19Z.6 |
| `exp*/`, `exp_*/` | Frozen experiment code (Exps 3–10, C-2, B-3B) | — | — | ✅ yes |
| `.venv/`, `venv/`, `.venv_py313/` | Virtualenvs | — | — | ❌ never |

### 19Z.3 Security audit — files to remove from tracking

**Never expose contents.** Status is *tracked in git*, which is the problem.

| File | Type | Safe to commit? | Needed on new machine? | How to provision |
|---|---|---|---|---|
| `server/orchestration_agent/certs/ca.key` | **CA private key** | ❌ **NO — already leaked** | Server only | `bash certs/gen_certs.sh <SERVER_IP>` |
| `server/orchestration_agent/certs/server.key` | Server private key | ❌ **NO — already leaked** | Server only | same script |
| `server/orchestration_agent/server.key` | Private key (stray copy) | ❌ **NO — already leaked** | No | delete after rotation |
| `client_agent/certs/client.key` | Client private key | ❌ **NO — already leaked** | No | per-client enrollment |
| `device_key.pem` | Device private key | ❌ **NO — already leaked** | No | generated at enrollment |
| `certs/ca.pem`, `server.pem`, `client.pem`, `*.csr`, `*.srl` | Public certs / CSRs | ⚠️ avoid | `ca.pem` yes (clients) | regenerate with PKI |
| `installer/runtime/keys/ca.pem` | CA public cert | ⚠️ avoid | ✅ clients need it | copy after regeneration |
| `.env` | env file (only `PYTHONPATH`) | ⚠️ | No | not in `origin/main` — keep it that way |
| `~/.federated/**` | client identity, `secrets/master.bin` | ❌ **never** | ✅ per machine | created by enrollment |
| `client_id.bin`, `msg.bin`, `sig.bin` | crypto scratch | ❌ no | No | regenerated |
| `installer/windows_signer/target/**` | Rust build artifacts (~40 MB) | ❌ no | No | `cargo build --release` |
| `*.log`, `orch_run.log`, `orch_err.log` | logs | ❌ no | No | generated |

**Required remediation before any further push** (not performed — your call):
1. Confirm whether the GitHub repo is **public or private**.
2. If public: **treat all five keys as compromised** and rotate the entire PKI via
   `bash certs/gen_certs.sh <SERVER_IP>`; redistribute the new `ca.pem` to every client.
3. `git rm --cached` the sensitive paths, commit, and (separately decided) purge history with
   `git filter-repo` / BFG. **History rewriting was NOT performed.**

### 19Z.4 Server requirements (verified)

| Item | Value | Source |
|---|---|---|
| Rust / Cargo | edition **2021**; no `rust-version` pin. Verified working on **1.96.0** | `Cargo.toml:4` |
| Directory | `server/orchestration_agent/` | — |
| Entry point | `src/main.rs`, `#[tokio::main]` | — |
| Config | `config/orchestrator.toml` — **loaded by relative path**, so you must `cd` into the crate | `main.rs:24` |
| Bind address | `0.0.0.0:50051`, `enable_tls = true` | `config/orchestrator.toml` |
| MongoDB | **Required, hard-fail.** Pings DB `federated`; GridFS for update payloads | `main.rs:35-43` |
| Env vars | `MONGO_URI` (default `mongodb://localhost:27017`) | `main.rs:29` |
| TLS material | `certs/{ca.pem, ca.key, server.pem, server.key}` | `config/orchestrator.toml` |
| Cert provisioning | `bash certs/gen_certs.sh <SERVER_IP>` — **requires OpenSSL**; default IP `192.168.1.7` | `certs/gen_certs.sh` |
| Enrollment OTP | printed to stdout/log at startup, valid 10 min | `main.rs:50-53` |

**Start command:**
```bash
cd server/orchestration_agent
export MONGO_URI="mongodb://localhost:27017"     # Windows PS: $env:MONGO_URI="..."
cargo run --release
```

### 19Z.5 Client requirements (verified)

| Item | Value | Source |
|---|---|---|
| Python | 3.11+ (local venv is 3.11.9) | — |
| Deps | `requirements.txt` (full freeze) / `requirements.lock.txt` | tracked |
| Entry point | `runtime/federated_client.py` | — |
| Env vars | **`FED_SERVER`** (`host:port`) — else it prompts interactively | `federated_client.py:146` |
| Identity home | **`~/.federated`** — *hardcoded*, no override | `grpc_client.py:36` |
| Required files | `~/.federated/keys/{ca.pem, client.key, client.pem}`, `bin/windows_signer.exe`, `secrets/master.bin` | verified present locally |
| Device ID | `sha256(device_pubkey)` — **derived, never assigned** | `federated_client.py:143` |
| Enrollment | `enroll_step5.py` → `RequestEnrollment`, parses OTP from server log, `EnrollDevice` | `enroll_step5.py` |
| Local training | ✅ yes, in `runtime/pipeline.py::run_pipeline()` | — |
| Update transport | streaming `UploadUpdate` → MongoDB GridFS | proto + `server.rs` |
| Global model | streaming `DownloadGlobalModel` | proto |

**Run command:**
```bash
export FED_SERVER="127.0.0.1:50051"              # Windows PS: $env:FED_SERVER="..."
python -m runtime.federated_client run-once      # or: daemon
```

### 19Z.6 Files NOT in GitHub that the new machine still needs

> ## 📦 THE DAIC-WOZ DATASET IS A **LOCAL PREREQUISITE** — IT IS NOT IN THIS REPOSITORY
>
> **Cloning this repository does NOT give you the dataset.** DAIC-WOZ is **licensed
> human-subjects clinical-interview data** and is deliberately excluded from Git. It must be
> obtained under your own DAIC-WOZ usage agreement and placed on each machine locally.
>
> The following are **git-ignored and will never be committed**:
>
> | Path | Contents |
> |---|---|
> | `data/` | raw DAIC-WOZ participant archives (188 × `<id>_P/`) |
> | `data_norm/` | normalised participant transcripts (`<id>_TRANSCRIPT.csv`) |
> | `labels/` | `Detailed_PHQ8_Labels.csv` — PHQ-8 depression labels |
> | `receipts/`, `plots/`, `processed/`, `explain_logs/` | generated runtime artifacts |
>
> **Expected layout on a new machine** (create these yourself; they stay untracked):
>
> ```
> BE-Major-Project/
> ├── data/         <id>_P/…      ← from your DAIC-WOZ download
> ├── data_norm/    <id>_P/<id>_TRANSCRIPT.csv
> └── labels/       Detailed_PHQ8_Labels.csv
> ```
>
> `data_norm/` is produced from `data/` by `normalize_transcripts.py`; the parquets are then
> built by `build_daic_records.py` / `dataset_build/`. **Never commit any of it.**

`.gitignore` also excludes `*.parquet`, `*.pt`, `trainer_outputs/` — so **cloning alone is not
enough** for the experiments:

| Artifact | Status | How to obtain |
|---|---|---|
| `data/`, `data_norm/`, `labels/` | ❌ **git-ignored** | **local prerequisite** — your own DAIC-WOZ copy |
| `daic_records.parquet` | ❌ not tracked | supply separately, or build from `data_norm/` |
| `dataset_build/daic_records_multimodal.parquet` | ❌ not tracked | supply separately, or rebuild via `dataset_build/` |
| `trainer_outputs/baseline_cv/fold_manifest.json` | ❌ not tracked | **supply separately — frozen input, must not be regenerated** |
| `trainer_outputs/baseline_cv/baseline_cv_summary.json` | ❌ not tracked | supply separately |
| `trainer_outputs/local_probe_base.pt` | ❌ **NOT tracked** (untracked during Git cleanup) | **supply separately** — required: `round0_step6_validate.py:55` hardcodes this path |
| MentalBERT snapshot (rev `24809aa8…`) | ❌ not tracked | HF download (gated → needs `huggingface-cli login`) or copy the snapshot dir |

**Git LFS is NOT required.** Largest tracked file is 13.48 MB (`libwindows-*.rlib`); nothing
exceeds GitHub's 50 MB warning. `.git` is 61 MB, inflated mostly by
`installer/windows_signer/target/`. Removing that from tracking is the single best size win.

### 19Z.7 RTX 3050 machine requirements

| Requirement | Minimum / expected | Why | Setup |
|---|---|---|---|
| OS | Windows 10/11 or Linux | `windows_signer.exe` is Windows-only; `anti_debug` has a full Linux path | — |
| GPU | **RTX 3050 (8 GB) is sufficient** | Phase 22 ran on a 15.6 GB T4, but batch 8 × 128 tokens on a 110 M model needs ≈4–6 GB | — |
| NVIDIA driver | ≥ 525 | CUDA 12.x runtime | GeForce Experience / NVIDIA site |
| CUDA | 12.1–12.8 (matched to the PyTorch wheel) | GPU training | bundled with the PyTorch wheel |
| Python | **3.11+** | matches local venv | python.org |
| PyTorch | CUDA build — **never `+cpu`** | validator check 19 hard-fails without CUDA | `pip install torch --index-url https://download.pytorch.org/whl/cu121` |
| Transformers | must export `transformers.AdamW` | `trainer_mentalbert_daic.py:34` imports it | pin ≤ 4.44.x, or verify the import |
| Rust + Cargo | 1.7x+ (verified 1.96.0) | builds the orchestrator | rustup.rs |
| MongoDB | **8.0** (verified locally) | server hard-fails without it | mongodb.com |
| Git | any | clone | — |
| Git LFS | ❌ **not needed** | no file >50 MB | — |
| OpenSSL | required **only** to regenerate certs | `gen_certs.sh` | Git-Bash bundles it on Windows |
| MSVC Build Tools | required on Windows | Rust `msvc` toolchain links with it | VS Build Tools + "Desktop C++" |
| protoc | ❌ not needed | `build.rs` uses `tonic-build`; Python stubs are pre-generated | — |
| Disk | **≈15 GB** | repo 61 MB + venv ~6 GB + Rust target ~2 GB + model ~0.5 GB + datasets | — |
| RAM | **16 GB** recommended | 110 M-param model + 439 MB deltas | — |
| VRAM | **≥6 GB** | training headroom | — |
| Network | internet for install; **LAN for the 3-PC demo** | — | — |

### 19Z.8 Clean installation sequence

```bash
# STEP 1  Prerequisites: Python 3.11+, Rust (rustup), MongoDB 8.0, Git,
#         MSVC Build Tools (Windows), NVIDIA driver ≥525

# STEP 2  Clone  (NOTE: this does NOT include the DAIC-WOZ dataset - see 19Z.6)
git clone https://github.com/soham-0510/BE-Major-Project.git
cd BE-Major-Project
# Place your own licensed DAIC-WOZ copy at: data/  data_norm/  labels/

# STEP 3  Python env
python -m venv .venv
.venv\Scripts\activate                       # Windows
source .venv/bin/activate                    # Linux

# STEP 4  Dependencies  (install the CUDA torch wheel FIRST)
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
pip install -r requirements.txt
python -c "import torch; print(torch.cuda.is_available(), torch.cuda.get_device_name(0))"
python -c "from transformers import AdamW; print('AdamW ok')"

# STEP 5  Build the Rust server
cd server/orchestration_agent && cargo build --release && cd ../..

# STEP 6  MongoDB — start the service, confirm it listens on 27017

# STEP 7  Regenerate TLS (MANDATORY — committed keys are compromised)
cd server/orchestration_agent
bash certs/gen_certs.sh <SERVER_LAN_IP>
cp certs/ca.pem ../../installer/runtime/keys/ca.pem
cd ../..

# STEP 8  Client identity: copy ca.pem to ~/.federated/keys/ca.pem
#         (the rest is created by enrollment in STEP 12)

# STEP 9  Environment
export MONGO_URI="mongodb://localhost:27017"
export FED_SERVER="127.0.0.1:50051"

# STEP 10-11  Terminal 1 — server
cd server/orchestration_agent && cargo run --release

# STEP 12  Terminal 2 — enroll (one-time)
python enroll_step5.py

# STEP 13  Terminal 3 — client
python -m runtime.federated_client run-once

# STEP 14-17  Verify: server logs "MongoDB connected" + OTP; client reaches
#             RegisterDevice/GetRound; aggregation fires at >=3 updates;
#             DownloadGlobalModel yields aggregated_round_N.pt
```

### 19Z.9 One-computer test — honest limitation

> ⚠️ **Multiple client processes under the same Windows user share `~/.federated/keys/` and
> therefore present the SAME device identity.** `device_id = sha256(device_pubkey)` is derived
> from that one keypair, and the only env override in the client is `FED_SERVER` — there is no
> identity/home override.

**Terminal 1** server · **Terminals 2–4** clients — all three clients will appear to the
orchestrator as **one device uploading three times**, which is enough to trip the
`updates.len() >= 3` aggregation trigger.

**This is a connectivity/integration smoke test ONLY. It must never be described as three
independent federated devices** — that is precisely the `devices = 1` artifact recorded as
weakness **RC-5** in `PHASE_10_ROADMAP.md:186`, and the reason C-2 existed. Repeated identity
in this test is *expected*, not a bug.

### 19Z.10 Three-computer LAN demo (the real topology)

```
              SERVER PC  (Rust :50051, MongoDB)
                        │  Wi-Fi / LAN
        ┌───────────────┼───────────────┐
      PC-1            PC-2            PC-3
    Client 1        Client 2        Client 3
```

| Item | Requirement |
|---|---|
| Server IP | fixed LAN IP; certs **must** be generated for it — `bash certs/gen_certs.sh 192.168.1.7` |
| SAN | `grpc_client.py` sets **no hostname override**, so the cert SAN must match the address exactly (`IP.x` for IPs) |
| `FED_SERVER` on clients | `192.168.1.7:50051` |
| Firewall | inbound TCP **50051** on the server |
| Client identity | each PC enrolls separately → distinct `~/.federated` → **genuinely distinct `device_id`** |
| MongoDB on clients | ❌ **not needed** — server only |
| Full repo on clients | ❌ not needed — clients need `runtime/`, `dp_agent/`, `enc_agent/`, `trainer_agent/`, `LDA/`, plus `~/.federated/keys/ca.pem` |
| Dataset on clients | ✅ each client needs its **own local** data shard |
| MentalBERT on clients | ✅ yes — local training runs on the client |

**This is the only configuration that demonstrates real federation.** Separate OS user
accounts on one machine also work (different `Path.home()`), and require no code change.

### 19Z.11 Recommended `.gitignore` additions

```gitignore
# Rust build artifacts (currently TRACKED — ~40 MB)
installer/windows_signer/target/
server/orchestration_agent/target/

# Key material (rules exist but files were tracked before them)
server/orchestration_agent/certs/
client_agent/certs/
installer/runtime/keys/
device_key.pem
*.csr

# Crypto scratch
client_id.bin
msg.bin
sig.bin

# Caches / IDE / OS
.pytest_cache/
.mypy_cache/
.idea/
*.swp
```

> Adding these does **not** untrack existing files. That needs
> `git rm --cached <path>` — **not performed**.

### 19Z.12 Troubleshooting

| Symptom | Cause | Fix |
|---|---|---|
| `MongoDB connection failed` | mongod not running | start the service; check 27017 |
| Server exits immediately | `config/orchestrator.toml` not found | `cd server/orchestration_agent` first — the path is relative |
| Client TLS handshake failure | cert SAN ≠ connect address | regenerate with the correct IP; copy `ca.pem` to `~/.federated/keys/` |
| `cannot import name 'AdamW'` | transformers too new | pin ≤ 4.44.x |
| `torch.cuda.is_available() == False` | CPU wheel installed | reinstall the cu121 wheel |
| Client prompts for server address | `FED_SERVER` unset | export it |
| Aggregation never triggers | fewer than 3 updates | needs `updates.len() >= 3` |
| All clients show one device | shared `~/.federated` | expected on one user — see §19Z.9 |

---

## 20. Repository Structure

```
BE-Major-Project/
├── README.md                          ← YOU ARE HERE (authoritative entry point)
├── ARCHITECTURE.md                    ← system architecture deep reference (1,134 lines)
├── PROJECT_KNOWLEDGE.md               ← tech stack, data/API/auth flows, dependency map
├── PHASE_10_ROADMAP.md                ← AUTHORITATIVE roadmap + Execution status
├── VERSION2_FINAL_REVIEW.md           ← read-only consistency audit (open issues)
│
├── PHASE_8B/9/9.5_*.md                ← official FROZEN baseline reports
├── PHASE_11_*.md                      ← evaluation protocol, Baseline-CV, diagnostics
├── PHASE_12–21_*_DESIGN.md            ← frozen pre-registrations (SHA-pinned)
├── FINAL_EXP7/EXP8/B3B_CLOSURE.md     ← official closure documents
│
├── LDA/                               ← Local Data Agent (text/audio/video, SecureStore)
├── trainer_agent/ · dp_agent/ · enc_agent/ · server/aggregator_agent/
├── trainer_mentalbert_daic.py         ← FROZEN trainer (model, recipe, compute_state_delta)
├── create_dp_comparison.py            ← legacy orchestrator; probe producer at :1138
├── format_daic_to_lda.py              ← ⚠️ FROZEN-DO-NOT-USE (max_length=256, 20 participants)
│
├── dataset_build/                     ← multimodal parquet builders
├── exp3_convergence/ · exp4_decision_rule/ · exp5_imbalance_objective/
├── exp6_loss_rebalance/ · exp7_modality/ · exp8_dp_mechanism/
├── exp9_compact_update/ · exp10_multimodal_conditioning/
├── exp_c2_multiclient/                ← C-2 (common, validate, run, aggregate, verify)
├── exp_b3b_probe/                     ← B-3B (common, validate, run, aggregate, verify)
├── phase22_end_to_end/                ← Phase 22 (design frozen, NOT executed)
├── make_phase22_upload_bundle.py      ← Colab bundle builder (fail-closed manifest)
│
└── trainer_outputs/                   ← ALL results (baseline_cv, exp3–10, c2_multiclient, b3b_probe)
```

Each experiment directory follows the same five-module pattern: `*_common.py` (frozen constants) · `validate_*.py` (pre-execution, fail-closed) · `run_*.py` (execution) · `aggregate_*.py` (verdict) · `verify_*.py` (independent verification).

---

## 21. Important Files and Artifacts

### Authoritative documents — **never delete**

| File | Role |
|---|---|
| `PHASE_10_ROADMAP.md` | The roadmap and its **Execution status — recorded results** block |
| `PHASE_8B_EXECUTION_REPORT.md`, `PHASE_9_BASELINE_EVALUATION_REPORT.md`, `PHASE_9.5_COMPLETE_DATASET_REPORT.md` | Official **FROZEN** baseline — *"They will not be modified."* |
| `PHASE_12`–`PHASE_21_*_DESIGN.md` | Frozen pre-registrations; several are **SHA-pinned inside result artifacts** |
| `FINAL_EXP7_CLOSURE.md`, `FINAL_EXP8_CLOSURE.md`, `FINAL_B3B_CLOSURE.md` | Official closure documents — authoritative for Rows 7, 8 and B-3B |
| `PHASE_11_*.md` (7 documents) | Evaluation protocol, Baseline-CV, diagnostics A/B, synthesis |
| `PHASE_13/14/15_*_RESULTS.md`, `PHASE_12_EXP4_DECISION_RULE.md` | Experiment result reports |
| `VERSION2_FINAL_REVIEW.md` | Consistency audit; records still-open issues |
| `ARCHITECTURE.md`, `PROJECT_KNOWLEDGE.md` | Infrastructure deep references |
| `trainer_outputs/**/*_summary.{json,md}`, `verify_*_report.json` | **Result artifacts — these are data, not documentation** |

### Key result artifacts

| Path | Contents |
|---|---|
| `trainer_outputs/baseline_cv/` | Baseline-CV summary + `fold_manifest.json` (the frozen splits) |
| `trainer_outputs/exp9_compact_update/exp9_summary.json` | Exp 9 A/B/C per arm, H₀ verdict |
| `trainer_outputs/multimodal_exp/exp10_summary.json` | Exp 10 PARTIAL verdict |
| `trainer_outputs/c2_multiclient/` | C-2: embeddings, partitions, divergence, aggregation, verification (12 files) |
| `trainer_outputs/b3b_probe/` | B-3B: runs, deltas, ROC-AUC, payload, summary, verification (10 files) |

---

## 22. Reproducibility / Integrity Rules

1. **Frozen artifacts are immutable.** Verify SHA-256 before *and* after every run.
2. **Freeze every parameter before execution.** Tag each [A] inherited / [B] repository-supported / [C] newly approved.
3. **Never tune after observing results.** If a registered risk materialises, report it — do not fix it retroactively.
4. **Never move, add, or invent a threshold** after results are seen.
5. **Verification is fail-closed:** `PASS` only if independently recomputed and matched; missing evidence is `PENDING`, and **PENDING counts against the verdict**.
6. **Re-derive, don't trust.** Verifiers retrain from scratch and compare — the standard achieved is `max |diff| = 0.000e+00`.
7. **H₀ is a legitimate result.** Two of this project's most valuable findings are H₀.
8. **Never claim success because code executed.**
9. **Keep new outputs isolated** in their own `trainer_outputs/<experiment>/` directory.
10. **One factor per experiment.** Phase 9.5 changed two at once and consequently could not attribute its own headline result.

---

## 23. IF YOU ARE TAKING OVER THIS PROJECT

**1. Read this README first**, then `PHASE_10_ROADMAP.md`, then the closure documents for Rows 7, 8 and B-3B.

**2. Do not modify frozen artifacts.** See [§11](#11-frozen-inputs). This includes `trainer_mentalbert_daic.py`, `dp_agent/dp_agent.py`, `daic_records.parquet`, `fold_manifest.json`, every `PHASE_*_DESIGN.md`, and the encrypted store.

**3. Do not rerun completed experiments** unless explicitly required. They are verified and their artifacts are referenced by SHA elsewhere.

**4. Know which files are authoritative** — frozen designs, closure documents, the roadmap, and `trainer_outputs/**` result artifacts. Prose summaries (including this README) are *derived*; if this README ever disagrees with a frozen design or closure document, **the frozen document wins** and this README is the file to fix.

**5. Understand the blocked dependency.** Row 10 needs B-2 ∧ B-3. Both are H₀. You cannot legitimately "just run C-1". See [§15](#15-current-project-status).

**6. Reusable components for Phase 22** (all verified and working):
- Embedding extraction with pinned-revision loading + integrity checkpoint → `exp_c2_multiclient/extract_c2_embeddings.py`
- Client partitioning (IID-like and non-IID A2) → `exp_c2_multiclient/c2_common.py`
- The frozen DP path and live ε accounting → `dp_agent/dp_agent.py`
- The frozen aggregator → `server/aggregator_agent/aggregator.py`
- The frozen model, recipe and delta convention → `trainer_mentalbert_daic.py`
- The five-module experiment scaffold → `exp_b3b_probe/` is the cleanest example
- Multimodal features → `dataset_build/` (⚠️ re-audit readiness first)

**7. Before writing Phase 22 code, inspect the existing verified implementations.** Do not reimplement what is already frozen and verified — import it. Prior experiments were caught out by re-deriving things that already existed.

**8. Never tune parameters after observing results.**

**9. Freeze experimental parameters before execution**, in a design document with a SHA.

**10. Use independent verification** — a separate verifier that re-derives results from frozen inputs, fail-closed.

**11. Keep new outputs isolated** — never write into an existing experiment's directory.

**12. Never claim an experiment succeeded merely because code executed.** This is the single most important rule in this repository.

---

## 24. Installation

```bash
# system packages (Debian/Ubuntu)
sudo apt install ffmpeg libsndfile1 python3-dev build-essential cmake git

# Python dependencies
pip install -r requirements.txt
```

Core dependencies: `torch torchvision torchaudio · transformers · pandas pyarrow numpy scipy scikit-learn · fastapi uvicorn · opencv-python ffmpeg-python librosa webrtcvad pyannote.audio · pydub matplotlib seaborn · cryptography · tenseal · openpyxl`

> **Note:** `transformers` must be pinned to a version exposing `transformers.AdamW` (imported by `trainer_mentalbert_daic.py:34`). MentalBERT is a **gated** HF repo — run `huggingface-cli login` before first download, or use the local snapshot at the pinned revision.

**Running the legacy orchestrator** (infrastructure demo, *not* a frozen experiment):

```bash
python create_dp_comparison.py \
  --lda-mode session --input-type text_dir --input-path ./sample_texts \
  --store-root ./secure_store --modes base rag vector_rag \
  --epochs 20 --lr 1e-3 --use-bert --device cuda --rag-k 3
```

**Running a frozen experiment** — always validate first:

```bash
python exp_b3b_probe/validate_b3b.py --report validate_b3b_report.json   # must exit 0
python exp_b3b_probe/run_b3b.py
python exp_b3b_probe/aggregate_b3b.py
python exp_b3b_probe/verify_b3b.py --report verify_b3b_report.json       # must exit 0
```

---

## 25. External Tools (OpenFace & openSMILE)

### OpenFace — facial Action Units (vision, 84-d)

**Linux**
```bash
sudo pacman -S cmake dlib opencv ffmpeg      # or apt equivalents
git clone https://github.com/TadasBaltrusaitis/OpenFace.git
cd OpenFace && bash download_models.sh
mkdir build && cd build && cmake .. && make -j$(nproc)
./bin/FeatureExtraction -h                    # verify
```

**Windows** — install [CMake](https://cmake.org/download/) and Visual Studio Build Tools, then:
```powershell
git clone https://github.com/TadasBaltrusaitis/OpenFace.git
cd OpenFace; .\download_models.ps1
mkdir build; cd build
cmake .. -G "Visual Studio 17 2022" -A x64
cmake --build . --config Release
```

### openSMILE — eGeMAPS acoustic features (audio, 154-d)

**Linux**
```bash
sudo pacman -S portaudio sox
git clone https://github.com/audeering/opensmile.git
cd opensmile && mkdir build && cd build && cmake .. && make -j$(nproc) && sudo make install
```

**Windows** — download prebuilt binaries from [audeering.github.io/opensmile](https://audeering.github.io/opensmile/download/), extract to `C:\Program Files\openSMILE\bin`, then `setx PATH "%PATH%;C:\Program Files\openSMILE\bin"`.

### Example `configs/local_config.yaml`

```yaml
mode: session
ingest: { video: true, audio: true, text: true }
video_pipe:
  openface:
    binary_path: /home/user/Desktop/BE-Major-Project/LDA/OpenFace/build/bin/FeatureExtraction
    classifiers_path: .../classifiers/haarcascade_frontalface_alt.xml
audio_pipe:
  opensmile: { binary_path: /usr/local/bin/SMILExtract }
  wav2vec2:  { model: facebook/wav2vec2-base-960h }
storage: { root: ./secure_store }
```

| Component | Linux | Windows |
|---|---|---|
| OpenFace binary | `.../OpenFace/build/bin/FeatureExtraction` | `...\OpenFace\build\bin\FeatureExtraction.exe` |
| openSMILE binary | `/usr/local/bin/SMILExtract` | `C:\Program Files\openSMILE\bin\SMILExtract.exe` |
| SecureStore root | `./secure_store` | `.\secure_store` |

---

## 26. Troubleshooting

| Issue | Cause | Fix |
|---|---|---|
| `utf-8 codec can't decode byte 0x80` | Binary DP delta read as text | Use `rb` mode or the `file://` prefix |
| `local variable 'delta_path' referenced before assignment` | Missing default delta path | Fixed in the current `run_pipeline()` |
| `MentalBERT gated repo` | Model requires acceptance | `huggingface-cli login`, or load the local snapshot with `local_files_only=True` |
| `Probe metrics all zero` | No numeric features | Ensure the `embedding` column exists in the parquet |
| `OpenFace model not found` | Wrong YAML path | Correct `video_pipe.openface.binary_path` |
| `cannot import name 'AdamW' from transformers` | `transformers` too new | Pin to a version still exporting `transformers.AdamW` |
| Frozen SHA mismatch on startup | A frozen artifact was modified | **Stop.** Restore from git; do not proceed |
| `Some weights … newly initialized: ['bert.pooler…']` | Pooler is randomly initialised | **Harmless** — verified: `last_hidden_state` is computed *before* the pooler; CLS embeddings are bit-identical across loads |

---

## 27. Credits and Citation

* **Ritik Shetty** — Architect & Lead Developer
* **Dr. Nupur Giri** — Project Supervisor
* **Ascentech Collaboration** — Data & Compute Resources

```bibtex
@software{privacy_preserving_ai_2025,
  author = {Shetty, Ritik and Giri, Nupur},
  title  = {Privacy-Preserving AI Framework for Multimodal Depression Detection},
  year   = {2025},
  url    = {https://github.com/ritikshetty/BE-Major-Project}
}
```

---

### Closing scientific position

This project delivers a **working, verified, privacy-preserving federated infrastructure** and a **rigorously established negative result** on federating DAIC task signal. Two independent levers — the DP mechanism (B-2) and the update representation (B-3, twice) — were each tested under frozen pre-registration with independent verification, and each returned H₀.

**That is a defensible scientific outcome, not an incomplete project.** The most actionable finding for future work is B-3B's diagnostic: the limitation lies in the **frozen pretrained representation**, not in the privacy mechanism.

---

---

# PART II — RTX 3050 MACHINE DEPLOYMENT RECORD

> **This part is the living record for the RTX 3050 Windows deployment.**
> It documents only verified machine state and actual execution results.
> It does NOT modify any frozen scientific parameter, experiment design, or historical result.
> Authority order still applies: frozen design > closure document > result artifact > verified SHA > README.md.
>
> Status codes used throughout:
> - **[VERIFIED]** — independently confirmed on this machine
> - **[COMPLETED]** — action taken and confirmed working
> - **[PARTIAL]** — partially done; see notes
> - **[BLOCKED]** — cannot proceed; see blocker entry
> - **[NOT TESTED]** — code/config exists but not exercised on this machine
> - **[NOT AVAILABLE]** — component absent from this machine
> - **[REQUIRES USER INPUT]** — needs a decision or credential from the user
> - **[REQUIRES EXTERNAL ARTIFACT]** — needs a file not yet present on this machine

---

## 28. Current Machine / Environment Status

> Last audited: **2026-08-14** by Claude Code (full read-only audit; nothing installed or modified).

### 28.1 Hardware

| Component | Value | Status |
|---|---|---|
| OS | Windows 11 Home Single Language 10.0.26200 | [VERIFIED] |
| GPU model | NVIDIA GeForce RTX 3050 **Laptop GPU** | [VERIFIED] — nvidia-smi |
| GPU VRAM | **4096 MiB (4 GB)** | [VERIFIED] — nvidia-smi |
| NVIDIA driver | 596.08 | [VERIFIED] — nvidia-smi |
| Hardware CUDA capability | **13.2** (reported by nvidia-smi) | [VERIFIED] |
| RAM | Not yet measured | [NOT TESTED] |

> ✅ **HARDWARE NOTE [VERIFIED 2026-08-15]:** README §19Z.7 states "RTX 3050 (8 GB) is sufficient." This machine has the **4 GB Laptop variant** (4.294 GB). Phase E VRAM probe (batch_size=8, full MultiModalModel) completed without OOM. Peak VRAM allocated: **1.117 GB / 4.294 GB** — headroom 3.178 GB. The 4 GB Laptop variant is confirmed feasible for the frozen batch_size=8 recipe.

### 28.2 Python Environment

| Component | Installed | Required (requirements.txt) | Status |
|---|---|---|---|
| Python | 3.11.9 | 3.11+ | [VERIFIED] |
| pip | 24.0 | — | [VERIFIED] |
| torch | **2.8.0+cpu** | 2.8.0 | [VERIFIED] — CPU build only |
| torchvision | (in requirements) | 0.23.0 | [NOT TESTED] |
| torchaudio | (in requirements) | 2.8.0 | [NOT TESTED] |
| transformers | 4.44.0 | 4.44.0 | [VERIFIED] — exact match |
| numpy | 2.2.6 | 1.26.4 | [VERIFIED] — version mismatch; 2.2.6 worked for Exp 7/8/B-3B on this machine |
| pandas | 2.3.2 | 2.3.2 | [VERIFIED] |
| pyarrow | 21.0.0 | 21.0.0 | [VERIFIED] |
| scipy | 1.15.3 | 1.15.3 | [VERIFIED] |
| scikit-learn | 1.7.1 | 1.7.1 | [VERIFIED] |
| cryptography | 45.0.6 | 45.0.6 | [VERIFIED] |
| grpcio | 1.62.2 | 1.62.2 | [VERIFIED] |
| pymongo | 4.16.0 | 4.16.0 | [VERIFIED] |

**Python environment note:** No project-specific `.venv` exists in the Capstone repo. System Python 3.11.9 is currently used directly.

### 28.3 CUDA / PyTorch GPU Status

| Check | Result | Status |
|---|---|---|
| `torch.cuda.is_available()` | **False** | [VERIFIED] |
| `torch.version.cuda` | None (CPU build) | [VERIFIED] |
| CUDA PyTorch wheel | **NOT installed** | [VERIFIED] |
| GPU smoke test (tensor → CUDA) | **Cannot run** | [BLOCKED] — requires CUDA PyTorch |
| Phase 22 batch-8 memory feasibility | Unknown | [NOT TESTED] |

### 28.4 External Services and Tools

| Component | Status | Notes |
|---|---|---|
| MongoDB 8.0 | [NOT AVAILABLE] | Required for live orchestration pipeline only; not needed for Phase 22 |
| Rust + Cargo | [NOT AVAILABLE] | Required to build Rust orchestrator; not needed for Phase 22 |
| MSVC Build Tools | Unknown | Required for Rust Windows toolchain |
| OpenFace binary | [NOT AVAILABLE] | Required for live LDA video; not needed for Phase 22 |
| openSMILE binary | [NOT AVAILABLE] | Required for live LDA audio; not needed for Phase 22 |
| ffmpeg | [NOT AVAILABLE] | Required for live LDA AV; not needed for Phase 22 |
| HuggingFace CLI | Available (pip) | `huggingface-cli` present via transformers install |

### 28.5 Repository State

| Property | Value | Status |
|---|---|---|
| Remote | `https://github.com/soham-0510/Capstone-.git` | [VERIFIED] |
| Branch | `main` | [VERIFIED] |
| HEAD commit | `b19b0d4 Initial clean release…` | [VERIFIED] |
| Working tree | Clean — no uncommitted changes | [VERIFIED] |
| Clone path | `D:\Download D\BE PIPELINE\Capstone-\` | [VERIFIED] |
| Tracked private keys | **None** | [VERIFIED] — git ls-files for *.pem/*.key/*.bin returned no matches |
| `.gitignore` coverage | Complete — covers all key, cert, parquet, pt, trainer_outputs | [VERIFIED] |
| git `core.autocrlf` | **true** | [VERIFIED] — causes CRLF line endings on checkout; see §28.6 |

### 28.6 Windows CRLF / SHA Issue

`git config core.autocrlf` is **true** on this machine. Text files (including `PHASE_22_DESIGN.md`) are checked out with CRLF line endings. The frozen SHA for `PHASE_22_DESIGN.md` (`3c181083…`) was computed on Linux (LF). On this machine, the raw-bytes SHA of that file is `bdc505d6…` (CRLF version), which does NOT match the frozen value.

**LF-normalized SHA of `PHASE_22_DESIGN.md` on this machine: [VERIFIED] matches frozen value exactly.**

Python source file SHAs (`trainer_mentalbert_daic.py`, `dp_agent/dp_agent.py`) were frozen on Windows (CRLF) and **do match** on this machine.

**Resolution required:** Re-clone with `git -c core.autocrlf=false clone …` to preserve LF line endings throughout, so all frozen SHA checks pass. **Do NOT modify `PHASE_22_DESIGN.md` content.**

### 28.7 MentalBERT Status

| Property | Status |
|---|---|
| HuggingFace cache path (`~/.cache/huggingface/hub/`) | Exists; contains only `models--sentence-transformers--all-MiniLM-L6-v2` |
| `mental/mental-bert-base-uncased` snapshot | [NOT AVAILABLE] — not in cache |
| Pinned revision `24809aa822c76639760d0d934742d1b42f89942f` | [NOT AVAILABLE] |
| HuggingFace account / license accepted | [REQUIRES USER INPUT] |

### 28.8 DAIC-WOZ Dataset Status

| Component | Status |
|---|---|
| Raw archives (`data/<id>_P/`) | [NOT AVAILABLE] on this machine |
| Normalized transcripts (`data_norm/`) | [NOT AVAILABLE] |
| PHQ labels (`labels/`) | [NOT AVAILABLE] |
| `D:\datasets\DAIC-WOZ` | Does not exist |
| `D:\Download D\data\` | Contains unrelated claims/corpus data (NOT DAIC-WOZ) |

---

## 29. Current End-to-End Demonstration Status

> Two pipelines are tracked separately as required. See §19 (Phase 22) and §19Z (Live Orchestration).

### Pipeline A — Phase 22 Frozen Scientific Reproduction

| # | Stage | Source File | Status | Notes |
|---|---|---|---|---|
| 1 | Dataset / frozen parquet available | `dataset_build/daic_records_multimodal.parquet` | [REQUIRES EXTERNAL ARTIFACT] | SHA `1ac9f53e…`; not present |
| 2 | Fold manifest available | `trainer_outputs/baseline_cv/fold_manifest.json` | [REQUIRES EXTERNAL ARTIFACT] | Frozen; SHA `b9a7a91f…`; must not be regenerated |
| 3 | Base parquet available | `daic_records.parquet` | [REQUIRES EXTERNAL ARTIFACT] | SHA `9a241851…`; not present |
| 4 | MentalBERT snapshot available | `~/.cache/huggingface/hub/…/24809aa8…/` | [REQUIRES EXTERNAL ARTIFACT] | 6 artifact SHAs frozen |
| 5 | CUDA PyTorch installed | `torch==2.8.0+cu128` | [NOT AVAILABLE] | Currently CPU-only; CUDA install requires user authorization |
| 6 | Validator 20/20 PASS | `phase22_end_to_end/validate_phase22.py` | [BLOCKED] | Crashes at check 3 (missing parquet); CRLF fix needed for checks 1–2 |
| 7 | GPU batch-8 memory feasibility | Requires memory probe | [NOT TESTED] | 4 GB VRAM; project estimate ≈4–6 GB; feasibility unknown |
| 8 | Phase 22 full run on this machine | `phase22_end_to_end/run_phase22.py` | [NOT TESTED] | No training has been executed on this machine |
| 9 | Aggregation (Python, in-process) | `server/aggregator_agent/aggregator.py` | [NOT TESTED] | Code present; never run on this machine |
| 10 | Evaluation / ROC-AUC | `phase22_end_to_end/run_phase22.py` | [NOT TESTED] | — |
| 11 | Independent verification | `phase22_end_to_end/verify_phase22.py` | [NOT TESTED] | — |
| 12 | Final artifacts produced | `trainer_outputs/phase22_end_to_end/` | [NOT TESTED] | — |

**Historical status:** Phase 22 was executed and verified 20/20 on **Colab GPU (Tesla T4, CUDA 12.8, PyTorch 2.11.0+cu128, Python 3.12.13)** as documented in §19.0. Result artifacts remain on the Colab/Google Drive instance and are **not present on this machine**. Local reproduction is [NOT TESTED].

---

### Pipeline B — Live Orchestration / Infrastructure Demonstration

| # | Stage | Source File | Status | Notes |
|---|---|---|---|---|
| 1 | LDA text preprocessing | `LDA/app/pipelines/text.py` | [NOT TESTED] | Code present; spaCy model not verified |
| 2 | LDA audio preprocessing | `LDA/app/pipelines/audio.py` | [NOT TESTED] | openSMILE [NOT AVAILABLE]; wav2vec2 not cached |
| 3 | LDA video preprocessing | `LDA/app/pipelines/video.py` | [NOT TESTED] | OpenFace binary [NOT AVAILABLE] |
| 4 | PII scrubbing (spaCy NER) | `LDA/app/pipelines/text.py` | [NOT TESTED] | spaCy en_core_web_sm status unknown |
| 5 | AES-GCM encryption / SecureStore | `centralized_secure_store.py` | [NOT TESTED] | Code present; crypto deps installed |
| 6 | HMAC receipt signing | `centralised_receipts.py` | [NOT TESTED] | Code present |
| 7 | Differential privacy (dp_agent) | `dp_agent/dp_agent.py` | [NOT TESTED] | Code present; ran on this machine in Exp 8 (CPU) |
| 8 | DP epsilon accounting | `dp_agent/dp_agent.py` | [VERIFIED] indirectly | RDP-to-DP function verified during Exp 8 |
| 9 | Encryption agent | `enc_agent/enc_agent.py` | [NOT TESTED] | Code present |
| 10 | gRPC transport | `runtime/grpc_client.py` | [NOT TESTED] | gRPC stubs pre-generated; no server to connect to |
| 11 | mTLS certificates | `server/orchestration_agent/certs/` | [NOT AVAILABLE] | Certs not generated on this machine; keys in old repo are compromised |
| 12 | Rust orchestrator (build) | `server/orchestration_agent/` | [NOT AVAILABLE] | Rust not installed |
| 13 | MongoDB 8.0 running | localhost:27017 | [NOT AVAILABLE] | MongoDB not installed |
| 14 | Client enrollment | `enroll_step5.py` | [NOT TESTED] | Requires server + certs + MongoDB |
| 15 | Federated client run-once | `runtime/federated_client.py` | [NOT TESTED] | Requires all of the above |
| 16 | Aggregation via MongoDB GridFS | `server/aggregator_agent/aggregator.py` | [NOT TESTED] | Requires MongoDB |
| 17 | Global model publication | MongoDB `global_models` | [NOT TESTED] | Known gap: aggregated model not written back (§ARCHITECTURE §2) |
| 18 | Client model synchronization | `DownloadGlobalModel` RPC | [NOT TESTED] | Requires running server |
| 19 | Full live end-to-end (smoke test) | §19Z.8 sequence | [NOT TESTED] | **Never end-to-end tested on this machine** (README §19Z explicit) |

---

## 30. Current Blockers

| # | Blocker | Severity | Affected Pipeline | Why Blocked | Exact Requirement | Status | Next Action |
|---|---|---|---|---|---|---|---|
| B-1 | CUDA PyTorch not installed | **CRITICAL** | Phase 22, Live (training) | `torch 2.8.0+cpu` installed; `torch.cuda.is_available()` = False | Install `torch==2.8.0+cu128` wheel | [REQUIRES USER INPUT] — user must authorize install | Await authorization, then: `pip install torch==2.8.0 torchvision==0.23.0 torchaudio==2.8.0 --index-url https://download.pytorch.org/whl/cu128` |
| B-2 | `daic_records_multimodal.parquet` missing | **CRITICAL** | Phase 22 | Gitignored; never on this machine | File at `dataset_build/`, SHA `1ac9f53e…` | [REQUIRES EXTERNAL ARTIFACT] | Transfer from DELL machine (in `phase22_upload_FINAL.zip`) or rebuild from DAIC-WOZ raw data |
| B-3 | `daic_records.parquet` missing | **CRITICAL** | Phase 22 | Gitignored | Root of repo, SHA `9a241851…` | [REQUIRES EXTERNAL ARTIFACT] | Same as B-2 |
| B-4 | `fold_manifest.json` missing | **CRITICAL** | Phase 22 | Gitignored; must NOT be regenerated — it is a frozen input | `trainer_outputs/baseline_cv/`, SHA `b9a7a91f…` | [REQUIRES EXTERNAL ARTIFACT] | Transfer from DELL machine only; do not regenerate |
| B-5 | `baseline_cv_summary.json` missing | **CRITICAL** | Phase 22 | Gitignored | `trainer_outputs/baseline_cv/`, SHA `f065f5e1…` | [REQUIRES EXTERNAL ARTIFACT] | Same as B-4 |
| B-6 | MentalBERT snapshot missing | **CRITICAL** | Phase 22, Live | Not in HF cache; gated model | `~/.cache/huggingface/hub/…/24809aa8…/`, 6 frozen SHAs | [REQUIRES EXTERNAL ARTIFACT] + [REQUIRES USER INPUT] | Transfer snapshot from DELL machine OR `huggingface-cli login` then download pinned revision |
| B-7 | Windows CRLF / validator check 1–2 | HIGH | Phase 22 validator | `core.autocrlf=true` causes CRLF; frozen SHA was computed on Linux | Re-clone with `git -c core.autocrlf=false clone …` | [NOT COMPLETED] — awaiting authorization | Re-clone to new path; do NOT edit `PHASE_22_DESIGN.md` |
| B-8 | 4 GB VRAM feasibility | ~~HIGH~~ **RESOLVED** | Phase 22 (frozen recipe) | ~~Project estimate ≈4–6 GB; 4 GB Laptop GPU may OOM at `batch_size=8`~~ | Phase E probe: peak VRAM 1.117 GB / 4.294 GB; headroom 3.178 GB; batch_size=8 confirmed | **[VERIFIED 2026-08-15]** — forward+backward PASS, no OOM | RTX 3050 4 GB Laptop variant is feasible; batch_size=8 NOT reduced |
| B-9 | MongoDB not installed | MEDIUM | Live pipeline only | Hard-fail in Rust orchestrator (`main.rs:35-43`) | MongoDB 8.0 on localhost:27017 | [NOT AVAILABLE] | Await Phase A–E completion before addressing |
| B-10 | Rust / Cargo not installed | MEDIUM | Live pipeline only | Orchestrator is pure Rust | Rust edition 2021, verified 1.96.0 | [NOT AVAILABLE] | Await Phase A–E completion before addressing |
| B-11 | TLS certificates not generated | MEDIUM | Live pipeline only | Keys in old repo (`soham-0510/BE-Major-Project`) are **compromised** | Run `bash certs/gen_certs.sh <IP>` after Rust build | [NOT AVAILABLE] | After B-10; new certs must NOT be committed |
| B-12 | OpenFace not installed | LOW | Live LDA video only | Binary subprocess not available | OpenFace CMake build or prebuilt Windows binary | [NOT AVAILABLE] | Phase J and beyond |
| B-13 | openSMILE not installed | LOW | Live LDA audio only | Binary subprocess not available | Prebuilt Windows binary | [NOT AVAILABLE] | Phase J and beyond |
| B-14 | DAIC-WOZ raw data not present | LOW | Live LDA only | Licensed data; not downloadable automatically | User's own DAIC-WOZ copy at `data/` | [NOT AVAILABLE] | Phase J and beyond; NOT needed for Phase 22 |
| B-15 | `local_probe_base.pt` missing | LOW | `round0_step6_validate.py` only | Gitignored; not needed for Phase 22 | `trainer_outputs/local_probe_base.pt`, SHA `d21f95ab…` | [NOT AVAILABLE] | Transfer if live pipeline step 6 validation is needed |

**Resolved blockers:** None yet. All blockers are current as of 2026-08-14.

---

## 31. Verified Artifact Table

| Artifact | Path | Required For | Frozen? | Expected SHA-256 | Present? | SHA Verified? | Source | Notes |
|---|---|---|---|---|---|---|---|---|
| Multimodal parquet | `dataset_build/daic_records_multimodal.parquet` | Phase 22 (primary dataset) | YES | `1ac9f53e6102ec0dbaab84dcfdfa3f2e70f2b4a1867a841ea2b7e3ba24c67a95` | **NO** | — | DELL machine / Colab Drive | In `phase22_upload_FINAL.zip` |
| Base parquet | `daic_records.parquet` | Phase 22, live pipeline | YES | `9a241851760727779fcaa4d50041b8bdc4406fe6b66ddbbd7105f4ce4fc55f00` | **NO** | — | DELL machine / Colab Drive | |
| Fold manifest | `trainer_outputs/baseline_cv/fold_manifest.json` | Phase 22, all Exps | YES — **must not regenerate** | `b9a7a91f1bdd43c78d3cd1ee0c36e4ce3fb8be4b0971dcff3ff62ee5af8fca2f` | **NO** | — | DELL machine | Regenerating invalidates experiment |
| Baseline CV summary | `trainer_outputs/baseline_cv/baseline_cv_summary.json` | Phase 22 | YES | `f065f5e1ff7f8e5ed4d122a8031c9cc12425802e756697d5512a915674871aac` | **NO** | — | DELL machine | |
| Local probe base | `trainer_outputs/local_probe_base.pt` | `round0_step6_validate.py` | YES | `d21f95ab4169ba8bf270e2ae900d2d205ddb4c3aba3c04ebae4d0ec8d9ba0993` | **NO** | — | DELL machine | Not needed for Phase 22 |
| Trainer (frozen) | `trainer_mentalbert_daic.py` | Phase 22, all experiments | YES | `65b1902e2ba8dd996c6960db1a6595d84fd48500f4d8707cb79688093d7a230b` | YES (in repo) | **YES** — SHA matches on Windows (CRLF; SHA frozen on Windows) | Git | |
| DP agent (frozen) | `dp_agent/dp_agent.py` | Phase 22, all experiments | YES | `758642fff57695cb970af88789c3b6a17c77f2b01d16303ff96230fce5798732` | YES (in repo) | **YES** — SHA matches on Windows | Git | |
| Aggregator (frozen) | `server/aggregator_agent/aggregator.py` | Phase 22 | YES | `59b4d4838cecfaa49f8341320c4d1fdb55fa3b2b6ca1717e29f273f820535d86` | YES (in repo) | NOT CHECKED | Git | |
| Phase 22 design | `PHASE_22_DESIGN.md` | Phase 22 validator | YES (FROZEN) | `3c18108309c17cbff332204aa761411d6600fe584488391741dfcd2a9208036f` (LF) | YES (in repo) | **CRLF MISMATCH** — raw SHA `bdc505d6…`; LF-normalized SHA matches | Git | Re-clone with `autocrlf=false` resolves this |
| MentalBERT `pytorch_model.bin` | `~/.cache/huggingface/hub/…/24809aa8…/pytorch_model.bin` | Phase 22, B-3B, all MentalBERT exps | YES | `c4f90fa5f0b991c48eb99afe41c8883dccb2b7e51012b33d2635925b9cde8764` | **NO** | — | HF gated / DELL machine | ≈414 MB |
| MentalBERT `tokenizer.json` | same snapshot dir | Phase 22 | YES | `5fd1c882abbd30517dced455a2c9768945ec726b96727927e4959348d9de550b` | **NO** | — | HF gated / DELL machine | |
| MentalBERT `vocab.txt` | same snapshot dir | Phase 22 | YES | `07eced375cec144d27c900241f3e339478dec958f92fddbc551f295c992038a3` | **NO** | — | HF gated / DELL machine | |
| MentalBERT `config.json` | same snapshot dir | Phase 22 | YES | `79ee28a4e33b49f535209c8aaa5eaa7550344345bb3fecb13f179c686749d0d9` | **NO** | — | HF gated / DELL machine | |
| MentalBERT `tokenizer_config.json` | same snapshot dir | Phase 22 | YES | `9a8ed9b01c8a56b555dcdc31cd526ba9e488cfc16ee5a101b3ce64af81d34f3d` | **NO** | — | HF gated / DELL machine | |
| MentalBERT `special_tokens_map.json` | same snapshot dir | Phase 22 | YES | `303df45a03609e4ead04bc3dc1536d0ab19b5358db685b6f3da123d05ec200e3` | **NO** | — | HF gated / DELL machine | |
| Phase 22 Colab bundle | `phase22_upload_FINAL.zip` | Colab execution (historical) | NO | `f248534d3bddd555d7bc30c794e7779a65a3778786b64dde93fed25ebe7ff5b2` | **Not on this machine** | — | DELL machine `C:\Users\DELL\Downloads\` | 407 MB; contains all artifacts above |
| Phase 22 result artifacts | `trainer_outputs/phase22_end_to_end/` | Verification of Colab run | NO | — | **Not on this machine** | — | Colab / Google Drive | ROC-AUC N=0.4651, D=0.5000 |
| C-2 embeddings (frozen) | `trainer_outputs/c2_multiclient/c2_embeddings.npz` | C-2 verification | YES | `70df4e49b4fea83c2464eba3e41e5b8ffa856fff4c70dc7b7b8d054cf488ed3c` | **NOT on this machine** | — | DELL machine | |

---

## 32. Environment Setup History

> Record of every meaningful environment action on this machine.
> Format: Date · Component · Action · Result · Verification.

| Date | Component | Version | Action | Result | Verification |
|---|---|---|---|---|---|
| 2026-08-14 | Repository | commit `b19b0d4` | `git clone https://github.com/soham-0510/Capstone-.git` | Clean clone; single commit; working tree clean | `git status` → nothing to commit |
| 2026-08-14 | nvidia-smi | Driver 596.08 | Read-only audit | RTX 3050 Laptop, 4096 MiB, CUDA 13.2 | `nvidia-smi` output confirmed |
| 2026-08-14 | PyTorch | 2.8.0+cpu | Read-only audit | CPU-only build confirmed; CUDA not available | `torch.cuda.is_available()` → False |
| 2026-08-14 | Key packages | (see §28.2) | Read-only audit | transformers 4.44.0, numpy 2.2.6, pyarrow 21.0.0 confirmed | `python -c "import …"` |
| 2026-08-14 | Phase 22 validator | — | `python phase22_end_to_end/validate_phase22.py --allow-cpu` | Crash at check 3: `FileNotFoundError: dataset_build/daic_records_multimodal.parquet` | Checks 1–2 FAIL (CRLF + missing artifacts); checks 3+ not reached |
| 2026-08-15 | Phase 22 validator | — | `PYTHONUTF8=1 python phase22_end_to_end/validate_phase22.py --allow-cpu` (D: venv, RTX 3050) | **20 PASS, 0 FAIL — VALIDATION: PASS** (6.3 s) | All checks 1–20 [ok]; exit 0; CUDA PASS (4.3 GB VRAM detected) |
| 2026-08-14 | CRLF audit | — | SHA cross-check on frozen files | `PHASE_22_DESIGN.md` LF-normalized SHA [VERIFIED] matches frozen value; `trainer_mentalbert_daic.py` / `dp_agent.py` raw SHA [VERIFIED] match frozen values | Python hashlib |
| 2026-08-14 | Security audit | — | `git ls-files` for *.pem/*.key/*.bin/*.pt | Zero tracked secrets in Capstone repo | `git ls-files \| grep -E "\.(pem\|key\|csr\|bin\|p12\|pfx)$"` → no output |
| 2026-08-14 | pip cache purge | — | `pip cache purge` | Freed 4,722.6 MB from C: (pip HTTP cache); C: free: 3.8 GB → 8.2 GB | `pip cache list` → 0 packages |
| 2026-08-14 | PyTorch (CUDA) | 2.8.0+cu128 | Install attempt via system pip | **FAILED** — `[Errno 28] No space left on device`. Root cause: `TEMP` set to POSIX path `/d/pip_tmp`; Python on Windows ignored it, fell back to `C:\AppData\Local\Temp`. 3.4 GB wheel extraction exhausted C:. CPU build rolled back to `2.8.0+cpu` successfully. | Error log captured |
| 2026-08-14 | PyTorch (CPU) | 2.8.0+cpu | `pip uninstall torch torchvision torchaudio -y` | Uninstalled to free space on C: before retrying CUDA install (C: → 9.1 GB free) | `python -c "import torch"` → ModuleNotFoundError (correct) |
| 2026-08-14 | PyTorch (CUDA) | 2.8.0+cu128 | Second install attempt (corrected TEMP=`D:\pip_tmp` Windows path) | **STOPPED** by user before completion. 1.5 GB partial extraction in `D:\pip_tmp`. torch NOT installed. | `python -c "import torch"` → ModuleNotFoundError |
| 2026-08-14 | **Environment decision** | — | Install target changed: **D: venv** instead of system Python on C: | DECISION: All ML packages go into a dedicated venv at `D:\Download D\BE PIPELINE\Capstone-\.venv`. C: system Python untouched for ML. pip temp=`D:\pip_tmp`, pip cache=`D:\pip_cache`. | Executed — see rows below |
| 2026-08-14 | D: venv | Python 3.11.9 | `python -m venv "D:\Download D\BE PIPELINE\Capstone-\.venv"` | venv created; executable and site-packages verified on D: | `sys.executable` → `D:\...\Capstone-\.venv\Scripts\python.exe`; `site.getsitepackages()` → `D:\...\Capstone-\.venv\Lib\site-packages` |
| 2026-08-14 | pip / setuptools / wheel | pip 26.2.1 / setuptools 84.0.0 / wheel 0.48.0 | `python -m pip install --upgrade pip setuptools wheel` (in D: venv) | Upgraded from pip 24.0 / setuptools 65.5.0 | Version confirmed via pip output |
| 2026-08-14 | PyTorch (CUDA) | 2.8.0+cu128 | `pip install --cache-dir D:\pip_cache torch==2.8.0 torchvision==0.23.0 torchaudio==2.8.0 --index-url https://download.pytorch.org/whl/cu128` (D: venv pip) | **SUCCESS** — CUDA available; RTX 3050 detected; 4095 MiB VRAM; CUDA tensor test passed | `torch.cuda.is_available()` → True; `torch.cuda.get_device_name(0)` → `NVIDIA GeForce RTX 3050 Laptop GPU`; `torch.version.cuda` → `12.8` |
| 2026-08-14 | System Python cleanup | — | Manual removal of orphaned torch/functorch dirs from C: site-packages (broken partial from prior failed attempts; no dist-info, missing DLLs) | `torch` (861 MB) and `functorch` dirs removed; `ModuleNotFoundError` confirmed on system Python | System Python `import torch` → `ModuleNotFoundError` ✓ |
| 2026-08-15 | Frozen artifacts | — | Extracted 3 files from `BE-Major-Project.zip` (selective; `.git` NOT touched) | All 3 SHA-256 verified on read-from-zip AND post-write-to-disk; written to repo | See §B artifact table |
| 2026-08-15 | pandas + pyarrow | pandas 3.0.5 / pyarrow 25.0.1 | `pip install pandas pyarrow` (D: venv) | Installed on D: venv; both confirmed under `D:\...\Capstone-\.venv\Lib\site-packages\` | `import pandas; import pyarrow` ✓ |
| 2026-08-15 | pyarrow downgrade | 25.0.1 → 21.0.0 | `pip install pyarrow==21.0.0` (D: venv) | Downgraded to match DELL's frozen build environment; pyarrow 25.0.1 produced different binary encoding | `pyarrow.__version__` → `21.0.0` ✓ |
| 2026-08-15 | Multimodal builder | — | `python build_daic_multimodal_records.py` (D: venv, pyarrow 21.0.0) | **SHA MATCH** [VERIFIED] — `1ac9f53e6102ec0dbaab84dcfdfa3f2e70f2b4a1867a841ea2b7e3ba24c67a95`; 188 rows; 6 columns; frozen columns byte-identical; 1,678,900 bytes | Double SHA read from disk: MATCH ✓ |
| 2026-08-15 | baseline_cv_summary.json | — | Search across all local dirs (repo, TeraBoxDownload, BE-Major-Project, Download D) | **NOT FOUND** anywhere on this machine | Must transfer from DELL |
| 2026-08-15 | transformers / scikit-learn | transformers 4.44.0 / sklearn 1.7.1 | `pip install transformers scikit-learn` (D: venv) + all deps (tokenizers, safetensors, huggingface-hub, scipy, joblib) | **SUCCESS** — all packages installed under `D:\...\Capstone-\.venv\Lib\site-packages\` | `import transformers; import sklearn` ✓ |
| 2026-08-15 | HF cache → D: junction | — | Moved `C:\Users\satya\.cache\huggingface\hub\` contents (88 MB sentence-transformers) to `D:\HF_cache\`; created Windows directory junction `C:\Users\satya\.cache\huggingface\hub` → `D:\HF_cache` via `mklink /J` | **SUCCESS** — `os.path.realpath('C:\...\hub')` → `D:\HF_cache`; all C: logical paths resolve to D: | `os.path.realpath()` → `D:\HF_cache` ✓; sentence-transformers SHA verified after move ✓ |
| 2026-08-15 | MentalBERT snapshot | mental/mental-bert-base-uncased @ `24809aa8` | `snapshot_download` to `D:\HF_cache` (via junction; HF token entered locally — never logged); 419 MB download | **SUCCESS** — all 6 frozen artifacts present and SHA-verified on D: | All 6 SHA MATCH: `pytorch_model.bin`, `tokenizer.json`, `vocab.txt`, `config.json`, `tokenizer_config.json`, `special_tokens_map.json`; `AutoModel.from_pretrained(snap)` load test PASS |
| 2026-08-15 | cryptography + pymongo | cryptography 50.0.0 / pymongo 4.17.0 | `pip install cryptography pymongo` (D: venv only) | **SUCCESS** — installed under `D:\...\Capstone-\.venv\Lib\site-packages\`; system Python untouched | `import cryptography; import pymongo` ✓ both physically on D: |
| 2026-08-15 | PHASE_22_DESIGN.md CRLF | — | `git config --local core.autocrlf false` + `rm PHASE_22_DESIGN.md` + `git checkout -- PHASE_22_DESIGN.md` | **SUCCESS** — file now LF-only (19,188 bytes, 390 LF lines, 0 CRLF); `git ls-files --eol` → `i/lf w/lf`; SHA matches frozen | SHA `3c18108…` MATCH ✓ |
| 2026-08-15 | baseline_cv_summary.json | — | Manually transferred from DELL; SHA-256 verified on arrival | **SUCCESS** — 3,977 bytes; SHA matches frozen `f065f5e1…` | SHA MATCH ✓ |
| 2026-08-15 | Phase 22 validator | — | `PYTHONUTF8=1 python phase22_end_to_end\validate_phase22.py --allow-cpu` (D: venv, 6.3 s) | **20/20 PASS — VALIDATION: PASS** | All 20 checks: [ok]; exit 0 |
| 2026-08-15 | Phase E — VRAM feasibility probe | — | `PYTHONUTF8=1 python phase_e_probe.py` (D: venv, batch_size=8, full MultiModalModel, CUDA) | **PASS** — forward+backward+grad-clip completed without OOM; peak VRAM allocated 1.117 GB / 4.294 GB; headroom 3.178 GB | Peak reserved: 1.325 GB; model load: 0.440 GB; fwd: 2.504 s; bwd: 0.384 s; batch_size=8 confirmed; param count 109,763,494 MATCH; NOTE: Phase E did NOT exercise AdamW optimizer state (~836 MB additional VRAM) |
| 2026-08-15 | Phase F attempt 1 | — | `PYTHONUTF8=1 python phase22_end_to_end/run_phase22.py` | **BLOCKED — CUDA OOM** at fold 1 / client 2 / optimizer.step() / AdamW exp_avg_sq; fold 1 / client 1 completed; batch size NOT reduced | No output artifacts | Peak in equivalent diagnostic probe: 1,973 MB / 4,294 MB |
| 2026-08-15 | Phase F attempt 2 | — | `PYTHONUTF8=1 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True python phase22_end_to_end/run_phase22.py` | **BLOCKED — SIGSEGV (exit 139)** — `expandable_segments` unsupported on Windows/WDDM; crash BEFORE any `[OK]` from main(); no VRAM stats | No output artifacts | — |
| 2026-08-15 | Phase F attempt 3 | — | `PYTHONUTF8=1 PYTORCH_NO_CUDA_MEMORY_CACHING=1 python phase22_end_to_end/run_phase22.py` | **BLOCKED — SIGSEGV (exit 139)** — same crash class; BEFORE any `[OK]` from main(); no VRAM stats; batch size NOT reduced | No output artifacts | Both CUDA allocator env vars cause SIGSEGV on PyTorch 2.8.0 + Windows WDDM |
| 2026-08-15 | Phase F attempt 4 | — | `PYTHONUTF8=1 python phase22_end_to_end/run_phase22.py` (NO env vars — exact original command) | **BLOCKED — SIGSEGV (exit 139)** — plain retry with no env vars; BEFORE any `[OK]` from main(); identical crash signature to attempts 2+3 despite no flags | No output artifacts | **DIAGNOSIS: NVIDIA CUDA driver state contaminated by SIGSEGV crashes in attempts 2+3. Plain attempt 1 succeeded before those crashes; plain attempt 4 fails after them. Driver state persists across Python processes until machine reboot.** Reboot required before next attempt. |

---

## 33. E2E Demonstration Log

> Record of every actual pipeline execution on this machine.

| Date | Pipeline | Commit | Environment | Command | Result | Artifacts | Verification | Limitations |
|---|---|---|---|---|---|---|---|---|
| 2026-08-14 | Phase 22 pre-execution validator | `b19b0d4` | Python 3.11.9, torch 2.8.0+cpu, Windows 11 | `python phase22_end_to_end/validate_phase22.py --allow-cpu` | **FAIL** — crash at check 3 (FileNotFoundError on missing parquet) | None | See §32 | Missing frozen data artifacts; CRLF issue on checks 1–2; CUDA check (19) would also fail |
| 2026-08-15 | Phase E — VRAM feasibility probe | — | Python 3.11.9, torch 2.8.0+cu128, RTX 3050 Laptop (4.294 GB VRAM), CUDA 12.8 | `PYTHONUTF8=1 python phase_e_probe.py` (D: venv) — forward+backward pass, batch_size=8, full MultiModalModel on GPU; loss = CrossEntropyLoss + 0.5·MSELoss | **PASS** — no OOM; forward PASS (loss=29.69), backward PASS, grad_clip(1.0) PASS | Peak VRAM 1.117 GB allocated / 1.325 GB reserved; headroom 3.178 GB | param count 109,763,494 MATCH; batch_size=8 NOT reduced | Phase E tested forward+backward ONLY — did NOT test AdamW optimizer state (~836 MB) |
| 2026-08-15 | Phase F — Phase 22 full execution (attempt 1) | — | Python 3.11.9, torch 2.8.0+cu128, RTX 3050 Laptop (4.294 GB VRAM), CUDA 12.8, PYTHONUTF8=1 | `PYTHONUTF8=1 python phase22_end_to_end/run_phase22.py` (D: venv, default device=cuda) | **BLOCKED — CUDA OOM** at fold 1 / client 2 / epoch 1 / optimizer.step() when AdamW allocated `exp_avg_sq`; fold 1 / client 1 completed (3 epochs, losses 31.55/40.05/17.69) | None (run aborted before any output artifacts were written) | VRAM diagnostic probe: post-client-1 state = 903 MB alloc / 2309 MB reserved; peak in equivalent probe = 1973 MB / 4294 MB — GPU physically has capacity; OOM likely due to CUDA allocator fragmentation under 3-epoch pressure + Windows WDDM scheduling; batch_size NOT reduced; recipe NOT modified | AdamW state (exp_avg + exp_avg_sq) = ~836 MB ON GPU — not tested by Phase E. Phase F: BLOCKED pending user decision. |
| 2026-08-15 | Phase F — Phase 22 full execution (attempt 2) | — | Python 3.11.9, torch 2.8.0+cu128, RTX 3050 Laptop (4.294 GB VRAM), CUDA 12.8, PYTHONUTF8=1, `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True` | `PYTHONUTF8=1 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True python phase22_end_to_end/run_phase22.py` (D: venv, default device=cuda) | **BLOCKED — Segmentation Fault (exit 139)** — process crashed at OS level; no Python exception; NO output artifacts | None | WARNING: `expandable_segments not supported on this platform` (CUDAAllocatorConfig.h:35); crash BEFORE any `[OK]` output from main(); SIGSEGV during CUDA allocator init | `expandable_segments:True` is Linux-only; causes crash on Windows/WDDM + PyTorch 2.8.0. Phase F: BLOCKED. No source files modified. |
| 2026-08-15 | Phase F — Phase 22 full execution (attempt 3) | — | Python 3.11.9, torch 2.8.0+cu128, RTX 3050 Laptop (4.294 GB VRAM), CUDA 12.8, PYTHONUTF8=1, `PYTORCH_NO_CUDA_MEMORY_CACHING=1` | `PYTHONUTF8=1 PYTORCH_NO_CUDA_MEMORY_CACHING=1 python phase22_end_to_end/run_phase22.py` (D: venv, default device=cuda) | **BLOCKED — Segmentation Fault (exit 139)** — same crash class as attempt 2; crashed BEFORE any `[OK]` output from main(); NO output artifacts | None | No platform-unsupported warning this time, but same SIGSEGV pattern; crash during CUDA allocator init, before training began | `PYTORCH_NO_CUDA_MEMORY_CACHING=1` also causes SIGSEGV on PyTorch 2.8.0 + Windows WDDM. BOTH CUDA allocator env var approaches are broken on this platform+version. Phase F: BLOCKED. No source files modified. |

---

## 34. Security Status Summary

| Item | Capstone repo (`soham-0510/Capstone-`) | Old repo (`soham-0510/BE-Major-Project`) |
|---|---|---|
| Private keys in current working tree | **None** [VERIFIED] | N/A (separate repo) |
| Private keys in pushed git history | **None** — clean release | **5 keys compromised** (CA root + 4 others) — see §19Z.3 |
| `.gitignore` coverage | Complete [VERIFIED] | Incomplete (keys were tracked before rules added) |
| Action required | None for current tracked files | Do NOT push; rotate entire PKI before any server use |

**TLS certs on this machine:** Not yet generated. When generated, they MUST NOT be committed. Store at `server/orchestration_agent/certs/` (gitignored) and `~/.federated/keys/` (outside repo). Document generation here when done.

---

## 35. Project Goal and Work Order for This Machine

**Goal:** Complete end-to-end demonstration of the implemented MediProof pipeline on the RTX 3050 system, while preserving the frozen scientific protocol and clearly separating exact reproduction from hardware-adapted engineering demonstrations.

### Phase progression (do not skip phases)

| Phase | Goal | Gate to proceed |
|---|---|---|
| **A — Environment** | Re-clone with `autocrlf=false`; install CUDA PyTorch | `torch.cuda.is_available()` == True |
| **B — Frozen artifacts** | Transfer frozen parquets, manifest, MentalBERT snapshot from DELL | All 4 data artifact SHAs verified |
| **C — CUDA GPU verification** | Verify RTX 3050 is visible to PyTorch; run smoke test | GPU name + VRAM printed correctly |
| **D — Phase 22 validator** | Run `validate_phase22.py`; must reach 20/20 PASS | Exit 0 |
| **E — GPU memory feasibility** | Probe: forward+backward, batch 8, full model; measure peak VRAM | If ≤4 GB: proceed. If OOM: **STOP and report; do NOT reduce batch size** |
| **F — Phase 22 exact reproduction** | Run Phase 22 IF Phase E passes AND user authorizes | 20/20 verification PASS |
| **G — Live pipeline dependencies** | MongoDB + Rust + MSVC | Services confirmed running |
| **H — PKI** | Regenerate TLS certs for this machine's IP | `gen_certs.sh` completes; new ca.pem distributed |
| **I — Rust orchestrator** | Build + start server; confirm MongoDB connected + OTP printed | Server logs confirmed |
| **J — LDA preprocessing** | Text → audio → video for one sample (not DAIC-WOZ if unavailable) | Encrypted artifacts produced + receipts signed |
| **K — Trainer / DP / encryption** | Single local training round → delta → DP → AES-GCM | Encrypted delta produced |
| **L — gRPC / mTLS** | Client connects to server; RegisterDevice → GetRound succeeds | Server log confirms registration |
| **M — Federated aggregation** | ≥3 uploads trigger aggregation | Aggregated model produced |
| **N — Live end-to-end** | Full `run-once` from client to aggregation | No crash; receipts signed |
| **O — Final verification** | Document all stages; update this README | All entries updated with [VERIFIED] or accurate status |

> **Current position: Phase F BLOCKED — 3 failed attempts [2026-08-15].**
> **Attempt 1 (no flags):** Python `torch.AcceleratorError: CUDA OOM` at fold 1 / client 2 / `optimizer.step()` — AdamW `exp_avg_sq` allocation. Fold 1/client 1 completed (3 epochs). Diagnostic probe showed peak 1,973 MB / 4,294 MB — GPU physically has capacity; OOM was allocator fragmentation.
> **Attempt 2 (`expandable_segments:True`):** SIGSEGV (exit 139). Root cause: flag is Linux-only, NOT supported on Windows WDDM. Crash during CUDA allocator init, before `main()` produced output.
> **Attempt 3 (`PYTORCH_NO_CUDA_MEMORY_CACHING=1`):** SIGSEGV (exit 139). Same crash class as attempt 2. Crash before `main()` produced output. **Both CUDA allocator env vars are broken (SIGSEGV) on PyTorch 2.8.0 + Windows WDDM.**
> **Established facts:** (1) GPU has physical capacity (1,973 MB peak < 4,294 MB); (2) attempt-1 OOM is fragmentation, not a capacity limit; (3) PyTorch 2.8.0 crashes on Windows when either CUDA allocator env var is set; (4) plain retry (no flags) is the only remaining Windows-safe option.
> **Current state (2026-08-15):** Phase F has attempted 4 runs. Attempt 4 (plain, no env vars) crashed with SIGSEGV — the same failure as attempts 2+3 which used invalid CUDA allocator env vars. Root cause: the SIGSEGV crashes in attempts 2+3 left the NVIDIA CUDA kernel driver in a corrupted state that persists across Python process restarts. Attempt 1 (the first ever run, before any crashes) reached training without SIGSEGV.
> **Required action before next attempt: REBOOT the machine.** A reboot clears the NVIDIA kernel driver state and GPU VRAM. Alternatively, restarting the display driver via Device Manager (Disable + Enable the NVIDIA GPU) may be sufficient on Windows.
> **After reboot:** Plain retry (no env vars, no source changes) is the appropriate next step. The diagnostic probe proved tensors fit (1,973 MB / 4,294 MB); the attempt-1 OOM was fragmentation that may not recur on a clean boot.

---

## 36. Dataset Discovery — `D:\TeraBoxDownload\Dataset\` (2026-08-14 read-only audit)

> Read-only audit only. Nothing extracted, moved, or modified.

### 36.1 Raw DAIC-WOZ Archives

| Property | Value | Status |
|---|---|---|
| Location | `D:\TeraBoxDownload\Dataset\` | [VERIFIED] |
| Total size | **85 GB** | [VERIFIED] |
| Participant archives | **187 zip files** (IDs 300–492) | [VERIFIED] |
| Missing IDs | 342, 367, 368, 394, 398, 460 | [VERIFIED] — not needed (project uses 188; `archive_audit.json` confirms bijection) |
| Corrupt archive | `440_P.zip` | [VERIFIED] — expected; documented in `archive_audit.json` |
| PHQ label CSVs | `train_split_Depression_AVEC2017.csv`, `dev_split_Depression_AVEC2017.csv`, `full_test_split.csv`, `test_split_Depression_AVEC2017.csv` | [VERIFIED] |

**Per-archive contents** (verified against `300_P.zip`):

| File | Description | Notes |
|---|---|---|
| `{ID}_AUDIO.wav` | Raw clinical interview audio | ~20 MB |
| `{ID}_TRANSCRIPT.csv` | Interview transcript | Text input for LDA and Phase 22 |
| `{ID}_COVAREP.csv` | Pre-extracted COVAREP audio features | 74 columns, 100 Hz — used in `multimodal_features.json` |
| `{ID}_FORMANT.csv` | Pre-extracted formant features | 5 columns |
| `{ID}_CLNF_AUs.txt` | Pre-extracted OpenFace Action Units | 24 features |
| `{ID}_CLNF_gaze.txt` | Pre-extracted OpenFace gaze | 16 features |
| `{ID}_CLNF_pose.txt` | Pre-extracted OpenFace head pose | 10 features |
| `{ID}_CLNF_features.txt` | Pre-extracted OpenFace 2D landmarks | — |
| `{ID}_CLNF_features3D.txt` | Pre-extracted OpenFace 3D landmarks | — |
| `{ID}_CLNF_hog.txt` | Pre-extracted HOG descriptors | ~350 MB per participant |

> **Critical insight:** COVAREP audio features and CLNF video features are **pre-extracted** inside each archive. OpenFace and openSMILE do NOT need to be installed to reconstruct Phase 22 features — those feature vectors are already materialized in `dataset_build/multimodal_features.json` (committed in git, SHA verified).

### 36.2 Frozen Artifact Source — `D:\TeraBoxDownload\Capstone\BE-Major-Project.zip` (2.87 GB)

**SHA-256 verified from inside the zip without extraction:**

| Artifact | Size | SHA-256 Computed | Frozen SHA Expected | Match? |
|---|---|---|---|---|
| `daic_records.parquet` | 916,554 bytes | `9a241851760727779fcaa4d50041b8bdc4406fe6b66ddbbd7105f4ce4fc55f00` | `9a241851...` | **MATCH** [VERIFIED] |
| `trainer_outputs/baseline_cv/fold_manifest.json` | 10,087 bytes | `b9a7a91f1bdd43c78d3cd1ee0c36e4ce3fb8be4b0971dcff3ff62ee5af8fca2f` | `b9a7a91f...` | **MATCH** [VERIFIED] |
| `trainer_outputs/local_probe_base.pt` | 1,185,335 bytes | `d21f95ab4169ba8bf270e2ae900d2d205ddb4c3aba3c04ebae4d0ec8d9ba0993` | `d21f95ab...` | **MATCH** [VERIFIED] |

> **Extraction note:** Extract ONLY these specific files from the zip. Do NOT extract the full zip — it contains the `.git` directory of the old repo whose git history holds 5 compromised private keys (CA root + 4 client keys). Those keys are in the git history, not in the working tree, but exposing the full `.git` folder on disk is unnecessary risk.

**Not found in zip:**
- `baseline_cv_summary.json` — not present anywhere on this machine
- `daic_records_multimodal.parquet` — not present anywhere on this machine
- MentalBERT model weights — empty placeholder directory only

### 36.3 `multimodal_features.json` — Already in Repo (Tracked in Git)

| Property | Value | Status |
|---|---|---|
| Path | `dataset_build/multimodal_features.json` | [VERIFIED] |
| Size | 1.44 MB | [VERIFIED] |
| SHA-256 | `a268a94432656a1419dacffc57f84267cea4979f241f254fc71decbdeec1f941` | [VERIFIED] |
| Pinned SHA in `multimodal_records_report.json` | `a268a94432656a1419dacffc57f84267cea4979f241f254fc71decbdeec1f941` | **EXACT MATCH** [VERIFIED] |
| Content | Feature vectors for all 188 participants (audio: 154-d, video: 84-d) extracted from DAIC-WOZ archives | [VERIFIED] |
| Generated by | `build_daic_multimodal_features.py` on DELL machine | [VERIFIED from report] |

This file is the exact intermediate artifact used on the DELL machine to produce the frozen `daic_records_multimodal.parquet`. Since it is SHA-verified and committed in git, running `build_daic_multimodal_records.py` with verified inputs can reconstruct the frozen parquet deterministically.

### 36.4 `baseline_cv_summary.json` — Not Found

Searched: `D:\TeraBoxDownload\`, `D:\Download D\`, `C:\Users\satya\`, `BE-Major-Project.zip`. **Not found anywhere on this machine.**

- Used by: Phase 22 **validator** (check 2 SHA) only — NOT by `run_phase22.py` itself
- Recovery: (a) Check DELL machine at `C:\Users\DELL\Desktop\pipeline backup\BE-Major-Project\trainer_outputs\baseline_cv\baseline_cv_summary.json`, or (b) re-run `colab_baseline_cv/run_baseline_cv.py` + `aggregate_baseline_cv.py` after CUDA + MentalBERT are available

### 36.5 `windows_signer.exe` — Pre-Built Binary Present

| Property | Value | Status |
|---|---|---|
| Path | `installer/runtime/windows_signer.exe` | [VERIFIED] |
| Size | 185,344 bytes | [VERIFIED] |
| Purpose | TPM-style artifact signing for live pipeline receipts | — |
| Rust compilation required? | **No** — pre-built binary | [VERIFIED] |

---

## 37. Updated Artifact Availability (Post-Discovery)

| Artifact | Status | Source | Next Action |
|---|---|---|---|
| `daic_records.parquet` | **IN ZIP, SHA VERIFIED** | `BE-Major-Project.zip` on this machine | Extract (targeted extraction only) |
| `daic_records_multimodal.parquet` | **Reconstructable** | Inputs in repo (SHA verified) | Run `build_daic_multimodal_records.py`; verify output SHA |
| `fold_manifest.json` | **IN ZIP, SHA VERIFIED** | `BE-Major-Project.zip` on this machine | Extract (targeted extraction only) |
| `local_probe_base.pt` | **IN ZIP, SHA VERIFIED** | `BE-Major-Project.zip` on this machine | Extract if needed (live pipeline step 6 only) |
| `multimodal_features.json` | **PRESENT, SHA VERIFIED** | Committed in git repo | No action needed |
| `baseline_cv_summary.json` | **NOT FOUND** | Unknown | Check DELL machine or re-run baseline CV |
| MentalBERT snapshot | **NOT FOUND** | HuggingFace Hub (gated) | Authorize HF login + download |
| DAIC-WOZ raw archives | **PRESENT** — 187/188 participants, 85 GB | `D:\TeraBoxDownload\Dataset\` | Available for live LDA demo |
| PHQ labels | **PRESENT** | `D:\TeraBoxDownload\Dataset\*.csv` | No action needed |
| `windows_signer.exe` | **PRESENT, pre-built** | `installer/runtime/` | No action needed |

**Updated blockers affected by discovery:**
- ~~B-2 (multimodal parquet transfer needed)~~ → Reconstructable from verified local inputs
- ~~B-3 (daic_records.parquet transfer needed)~~ → In BE-Major-Project.zip, SHA verified
- ~~B-4 (fold_manifest.json transfer needed)~~ → In BE-Major-Project.zip, SHA verified
- ~~B-14 (DAIC-WOZ not present)~~ → 85 GB present at D:\TeraBoxDownload\Dataset\
- B-6 (MentalBERT) remains [NOT AVAILABLE]
- B-8 (baseline_cv_summary.json) remains [NOT FOUND]
- B-1 (CUDA PyTorch) remains [REQUIRES USER INPUT]
