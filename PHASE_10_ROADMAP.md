# PHASE 10 — OFFICIAL VERSION 2 ROADMAP
## Federated, Privacy-Preserving Mental-Health Detection Pipeline (DAIC-WOZ)

**Phase:** 10 — Baseline Analysis & Version 2 Research Design
**Status:** ✅ **COMPLETE**
**Constituent work:** 10.1 Evidence Collection · 10.2 Root Cause Analysis · 10.3 Weakness Prioritization · 10.4 Version 2 Research Design
**Authority:** This document supersedes nothing and modifies nothing. It is the strategic roadmap for all work after Phase 9.5.

Throughout: **[E]** = evidence established in the frozen baseline reports · **[H]** = research hypothesis, not yet demonstrated.

---

# SECTION 1 — Executive Summary

## 1.1 Why Phase 10 was necessary

Phase 9.5 completed the reconstruction of the original implementation and re-ran it, unchanged, on the complete DAIC-WOZ corpus. It produced a headline result — **PHQ regression MAE improved 8.7652 → 6.6547 (−24.1 %)** — and a stated caveat: the classifier remained degenerate.

Phase 10 was necessary because **that caveat was larger than it appeared, and the headline was weaker than it appeared.** Neither could be established without a systematic audit, because the baseline's failures are **silent**: the pipeline runs end-to-end, every verification gate returns `[PASS]`, the cryptography verifies, and every artifact is well-formed. Nothing crashes. A project that triaged by visible symptoms would have concluded the system was healthy and merely under-tuned.

Phase 10 asked a different question — not *"what broke?"* but *"what do the numbers actually license us to claim?"* The answer required tracing every reported metric back to its mechanism.

## 1.2 What was learned from the validated baseline

Four findings, in order of consequence:

**1. The system has never detected a single depressed participant.** [E] Precision, recall and F1 are **exactly 0.0** in both the 113-participant and the 188-participant runs. All 11 and all 18 validation predictions were negative. The proof of the mechanism is exact: the model's predicted positive probability **is the class prior** — 0.365 when the corpus was 32.7 % positive, 0.20 when it was 23.9 % positive. The classifier learned the base rate, not the task.

**2. The one claimed gain is currently unattributed.** [E] The regression head's predictions span **4.27–4.34** across all 11 samples, then **5.03–5.09** across all 18 — a spread of **0.06 on a 0–23 scale**. The model assigns essentially the same PHQ score to every participant. A near-constant predictor's MAE moves when the constant relocates or the evaluation split changes — and between the two runs, **both** changed. The −24.1 % improvement is a real measurement whose cause is **not demonstrated to be learning**.

**3. The federated, private, multimodal claims are not supported by their own evidence.** [E] The federated global model derives from `sample_texts/sample1.txt`, a public demo file — it contains **no DAIC data whatsoever**. The DP mechanism clips updates to L2 norm 1.0 and then adds noise of norm **543** — a **543 : 1** noise-to-signal ratio, confirmed by the baseline's own sweep. Audio and video contribute **exactly 0.0**, because the pipeline discards both modalities at file selection, three stages before the model — despite every one of the 189 archives containing them.

**4. Phase 9.5's true contribution was eliminating the last excuse.** [E] The corpus is now **complete** (188 of 189; the 189th archive is corrupt and unrecoverable). The training set grew 102 → 170 (+66 %) and P/R/F1 moved **0.0 → 0.0**. *"We need more data"* is permanently unavailable as an explanation for anything. Every remaining failure lies in the training recipe, the privacy mechanism, or the data pipeline — **none in the data**.

## 1.3 Why Version 2 is now justified

Version 2 is justified not because the baseline underperforms, but because **the baseline cannot currently substantiate any of the four claims the project is built on** — depression classification, PHQ regression, multimodal fusion, and privacy-preserving federation.

Phase 10 traced all 25 documented weaknesses to **seven root causes (RC-1 … RC-7)**, and established a decisive asymmetry:

> **RC-6 and RC-7 (engineering defects, reproducibility gaps) generate 7 of the 25 weaknesses — 28 % of the count and 0 % of the Critical severity. RC-1 through RC-5 generate all six Critical weaknesses and produce no crashes at all.**
>
> **A team triaging this system by visible symptoms would fix exactly the wrong half.**

Critically, the six Critical weaknesses are also among the **cheapest to investigate** — four of them are Low-to-Medium complexity and **High compatibility with the original architecture**. Version 2 is therefore justified on the strongest possible grounds: the highest-severity problems are the most tractable, the dataset excuse is exhausted, and the causes are now identified rather than suspected.

---

# SECTION 2 — Baseline Status: FROZEN

The following three reports are hereby **FROZEN** and constitute the **official reference baseline** for the remainder of the Capstone Project:

| Report | Role | Status |
|---|---|---|
| **`PHASE_8B_EXECUTION_REPORT.md`** | The original end-to-end execution record — canonical command sequence, live training log, federated round trace, documented failure modes (§5.2, §5.3, §5.4, §5.5) | **FROZEN** |
| **`PHASE_9_BASELINE_EVALUATION_REPORT.md`** | The original-implementation baseline on 113 participants — Track C and Track A metrics, DP accounting, federation record, 12 documented limitations | **FROZEN** |
| **`PHASE_9.5_COMPLETE_DATASET_REPORT.md`** | The complete-dataset baseline on 188 participants (official Google Colab training) — the primary comparator for all Version 2 work | **FROZEN** |

## Explicit declarations

1. **These reports will NOT be modified.** Not corrected, not amended, not retro-fitted with Version 2 results, not rewritten under an improved evaluation protocol. They are the historical record of what the original implementation did.

2. **All future experiments must compare against them.** No Version 2 claim is admissible unless it is stated relative to a specific, cited figure in one of these three reports.

3. **The frozen baseline numbers are:**

| Quantity | Frozen value | Source |
|---|---|---|
| Dataset | 188 records · pos 45 / neg 143 · 440 excluded · 487 recovered | P9.5 §2.3, §4 |
| Split | train 170 / val 18 (`np.random.seed(42)`, val_split 0.1) | P9.5 §6 |
| **Regression MAE** | **6.654737075169881** | P9.5 §6.1 |
| **Accuracy** | **0.7222222222222222** | P9.5 §6.1 |
| **Precision / Recall / F1** | **0.0 / 0.0 / 0.0** | P9.5 §6.1 |
| **Modality (text / audio / video)** | **2.3365 / 0.0 / 0.0** | P9.5 §6.2 |
| Validation predictions | 18 rows, **all class 0**, P(pos) 0.199–0.214, pred_phq 5.03–5.09 | P9.5 §7.1 |
| ε per update / cumulative | 5.302585092994046 / 15.9078 per round | P9 §6.2 |
| DP: mechanism / clip / noise / δ | Gaussian / 1.0 / 1.0 / 1e-5 · L2-before 11.3448 · L2-after ≈543 | P9 §6.2, §8 |
| Global model | round_id 2 · hash `626dfc2a…d1b6` · 1,184,989 B · **derived from demo corpus** | P9 §6.3 |
| Federation topology | 1 device · 3 sequential uploads · ≈45 s | P9 §6.4, §6.5 |
| Training wall-clock | **not measured** (no timer emitted) | P9 §7 |

4. **The 113-participant baseline is preserved** in `trainer_outputs/phase9_113_baseline/`; the unofficial local 188 run is preserved in `trainer_outputs/local_188_preliminary/` and remains **superseded** by the Colab result.

5. **MongoDB `federated` remains untouched** — 6 model_updates / 6 receipts / 1 global_model (round_id 2) / 8 fs.files / 52 fs.chunks / 1 device. This is the standing federation evidence and must not be wiped.

---

# SECTION 3 — Validated Strengths of Version 1

Stated conservatively. Every item below is supported by a specific artifact; **none is a claim about model quality**, and that distinction is deliberate.

## 3.1 Successful reconstruction of the senior implementation

[E] The original implementation was recovered, executed unchanged, and re-executed on the complete corpus with the trainer **SHA-256 identical** to the repo original (`65b1902e…a230b`). Phase 9.5's twelve-item audit returned **PASS on all twelve** criteria: no source file modified, no architecture change, no hyperparameter change, no DP/aggregation change. **The reconstruction is faithful, and this is a genuine and non-trivial achievement** — it is what makes every finding in Phase 10 trustworthy.

## 3.2 The full pipeline executes end-to-end

[E] extract → normalize → build → verify → LDA embed → train → checkpoint → delta → DP → encrypt → chunked upload → aggregate → publish → download → next round. The pipeline completed on real DAIC transcripts across two independent federated rounds (26 Jun, 28 Jun) and on both the 113 and 188 corpora.

## 3.3 Cryptographic integrity is real and verified

[E] This is the baseline's strongest verified property, and it should be protected in Version 2:
- Per-chunk **SHA-256**; full-model hash match on download.
- `load_state_dict(strict=True)` clean; **per-tensor equality verified** after round-trip.
- **TPM-backed ECDSA P-256** receipts (Windows Platform Crypto Provider, key `FederatedDeviceKey`).
- **HMAC receipt chaining**; receipt scheme `AES-GCM-DP-ECDSA`, `verified = True`.
- **AES-GCM** encryption at rest; 492 receipt JSONs on disk as a full historical trail.

## 3.4 Genuine differential-privacy accounting

[E] ε = 5.302585092994046 per update is **computed by a real RDP accountant, not hard-coded**, and tracked cumulatively against `eps_max`. The *accounting* is correct. (Section 4 explains why the *guarantee* is nonetheless vacuous — the arithmetic is sound; the object it protects is not.)

## 3.5 Working infrastructure integration

[E] Rust orchestrator (gRPC :50051, mTLS enabled) · MongoDB v8.0 with GridFS chunked storage · TPM key enrollment · trimmed-mean aggregation firing correctly at the ≥3-update threshold · full round transition (Round 1 → Round 2, `global_model_available = True`).

## 3.6 Stable, reproducible infrastructure behaviour

[E] Two independent runs produced **identical** DP parameters (ε, L2-before-clip 11.3448), **identical** global-model schema, and **identical** classification metrics (0.5455 / 0 / 0 / 0). Phase 8B's live log and Phase 9's persisted artifacts agree across every cross-checked quantity.

## 3.7 Canonical model fidelity

[E] The exact **gated** MentalBERT (`mental/mental-bert-base-uncased`, vocab 30522) was used; a mismatched local substitute was **correctly rejected** rather than silently accepted.

## 3.8 Monotone learning signal

[E] Training loss fell **37.3740 → 27.5487 → 23.6329** and validation MAE improved **9.9550 → 9.1430 → 8.7315**, monotonically, every epoch. **This confirms the training loop is wired correctly** — the optimizer, the loss, and the gradient path all function. The failure documented in Section 4 is *not* a broken training loop.

---

## ⚠ What these strengths do NOT establish

Stated explicitly, so that Section 3 is not misread:

- They do **not** establish that the system detects depression. [E] It never has (P/R/F1 = 0.0).
- They do **not** establish that federation preserves mental-health signal. [E] The federated object contains none.
- They do **not** establish a working privacy–utility tradeoff. [E] No utility survives the mechanism (543 : 1).
- They do **not** establish multimodal capability. [E] Audio and video contribute exactly 0.0.

**The infrastructure is sound. The science it carries is not yet demonstrated.** That gap is precisely what Version 2 exists to close.

---

# SECTION 4 — Validated Weaknesses

Grouped by domain; each traced to its root cause (RC-1 … RC-7) from Phase 10.2 and its severity from Phase 10.3.

## 4.1 Dataset

| Weakness | RC | Sev | Explanation |
|---|---|---|---|
| **Class imbalance (23.9 % positive)** | RC-1 | High | [E] pos 45 / neg 143. The **complete** corpus is *more* imbalanced than the 113 subset (32.7 % → 23.9 %), and the 440 exclusion removed a **positive** (PHQ8 = 19.0). This is the input condition RC-1 exploits, and it is why the accuracy illusion strengthened. |
| **The dataset lever is exhausted** | — | Medium | [E] **Not a defect — the single most consequential fact for Version 2.** 188 of 189 is the complete corpus. Training set +66 % moved P/R/F1 by 0.0. No future failure can be blamed on data volume. |
| **Participant 440 permanently excluded** | — | Low | [E] Upstream file corruption (EOCD signature absent). Unfixable by any project code. Costs one positive case. |
| **Label-source drift** | RC-7 | Medium | [E] `full_test_split.csv` renamed `PHQ8_Score` → `PHQ_Score`. `stage_labels.py` now harvests **141 labels instead of 189**, silently dropping 47 participants — **19 of them in the new 416–492 range**. The 189-entry sheet is **orphaned**: correct, but unreproducible by the code that made it. |
| **Extractor AppleDouble defect** | RC-7 | Medium | [E] A first-match heuristic selected a 4 KB macOS `._` sidecar over the real transcript. **Detected only because the sidecar's bytes happened to break UTF-8 decoding** — a cleanly-decoding sidecar "would have silently produced a garbage record." Affected 487 only (all 189 audited). |

## 4.2 Model

| Weakness | RC | Sev | Explanation |
|---|---|---|---|
| **The trained model can never be federated** | RC-4 | High | [E] Three independent blockers, any one sufficient: (a) √d noise at 109M params ⇒ ~10⁴ : 1 SNR; (b) the upload driver **hardcodes** its source; (c) 438 MB × 3 exceeds the **15 s** receipt timeout. **This is why the project has two disconnected tracks that "never share weights."** |
| **The multimodal architecture is dead code** | RC-3 | Critical | [E] `audio_encoder` / `vision_encoder` are **conditionally constructed on dim > 0**, and the condition never fires. Present in code, unexercised in every run. |

## 4.3 Training — *the performance failure*

| Weakness | RC | Sev | Explanation |
|---|---|---|---|
| **Classifier collapsed to the class prior** | RC-1 | **Critical** | [E] P/R/F1 = **0.0** in both runs; 11/11 and 18/18 predictions negative. **P(positive) = 0.365 at a 32.7 % prior; 0.20 at a 23.9 % prior — the output *is* the prior.** The primary task has never functioned. |
| **Regression head is also near-constant** | RC-1 | **Critical** | [E] `pred_phq` spread of **0.06 on a 0–23 scale**. The model assigns essentially the same PHQ to everyone. **The −24.1 % MAE gain is therefore not demonstrated to be learning.** |
| **Undertrained — 3 epochs, not converged** | RC-1 | High | [E] Loss still falling by **3.92/epoch** at the cut-off; val MAE still improving. The epoch count is a **default, not a convergence criterion** — no early stopping exists. Prior-collapse is the *early* solution the budget never allowed escaping. |
| **Evaluation on 11 then 18 samples** | RC-2 | **Critical** | [E] `val_split = 0.1` on n ≈ 10². One flipped prediction moves accuracy by **5.6 %**. **No metric in the baseline can support or refute any hypothesis at this sample size** — including any future comparison against it. |

## 4.4 Multimodal

| Weakness | RC | Sev | Explanation |
|---|---|---|---|
| **Audio and video contribute exactly 0.0** | RC-3 | **Critical** | [E] **All 189 archives contain audio (`.wav`, COVAREP, FORMANT) and video (CLNF AUs/features/gaze/hog/pose).** Extraction stages transcripts only; the record schema has no audio/vision columns; the encoders therefore resolve to `None`. Ablation returns **exactly 0.0** — absence-from-graph, not weak contribution. **The signal is discarded three stages before the model.** |
| **LDA capped at 20 participants** | RC-3 | Low | [E] Hardcoded `MAX_PARTICIPANTS = 20`, preserved by mandate. **Conclusively inert:** LDA outputs are **byte-identical across both phases** (91,927 B / 159 B / shape (20,4)) — the 66 % dataset upgrade changed this stage by *zero bytes*. Coverage *degraded* 18 % → 11 % as a result of the upgrade. |

## 4.5 Federated Learning

| Weakness | RC | Sev | Explanation |
|---|---|---|---|
| **The federated model contains no DAIC signal** | RC-5 | **Critical** | [E] The global model derives from `local_probe_base.pt`, fitted to `sample_texts/sample1.txt`. Probe metrics: accuracy **1.0** with P/R/F1 **0.0** — the signature of a **single-class demo corpus**. **Every federated, DP and aggregation result in this project was obtained on an object with no mental-health signal in it.** |
| **"Federation" is one device submitting three times** | RC-5 | High | [E] `devices = 1`; three sequential uploads on loopback within 45 s. The aggregation trigger is a **count (≥3), not an identity check**. **Trimmed-mean — whose entire purpose is robustness across divergent clients — is averaging three copies of the same file that differ only by their DP noise draw.** |
| **Federation is not restartable** | RC-6 | Medium | [E] Round state lives **in memory** while artifacts persist to Mongo, and the two are never reconciled. Symptoms: a mid-round restart **orphans uploads**; a fresh orchestrator re-seeds Round 1 and its aggregation **collides with the existing `round_id = 2`** under a unique index. The only documented recovery is a **destructive wipe of five collections**. **This engineering defect dictated an experimental decision** — it is why Phase 9.5 could not re-run federation. |

## 4.6 Privacy

| Weakness | RC | Sev | Explanation |
|---|---|---|---|
| **The DP mechanism destroys the update (543 : 1)** | RC-4 | **Critical** | [E] Confirmed by the baseline's own sweep: mechanism `none` → L2-after **1.00** (the clip alone); `gaussian` at nm 1.0 → L2-after **543.49**; and **543² ≈ the probe's ~296k parameters**. Clipping is **dimension-independent** while noise scales as **√d**, so SNR = 1/√d. **This is not an unfavourable tradeoff — it is the absence of one, because no utility survives.** |
| **Weak ε, spent on the wrong object** | RC-5/RC-4 | High | [E] ε = 5.3026/update against `eps_max = 100` — the guard would not fire until ~round 19, so **it never constrained anything**. The accountant is genuine; **the ε is spent protecting a public demo file.** No DAIC participant's privacy was ever actually at stake. Note the compounding: even at this *permissive* setting, utility is already destroyed. |
| **The DP sweep cannot produce a result** | RC-5 | High | [E] Accuracy **flat at 1.0** and MAE flat at 1.3204 across **6 mechanisms × 5 noise levels × 3 corpora (90 rows)**. A metric that is 1.0 for every model **cannot degrade**. **The headline privacy finding — "no tradeoff observable" — describes the corpus, not the mechanism.** |

## 4.7 Engineering / Reproducibility

| Weakness | RC | Sev | Explanation |
|---|---|---|---|
| **The 113 and 188 validation sets are not comparable** | RC-7/RC-2 | High | [E] `np.random.seed(42)` fixes the RNG but **not the permutation** when array length changes. The two val sets are **different, non-nested participant sets** of different sizes with different class balances. **The entire Phase 9.5 comparison table compares two different test sets.** |
| **Torch is unseeded** | RC-7 | Medium | [E] Only the split is seeded. Drift: MAE 8.7315 vs 8.7652; 6.7058 vs 6.6547. **Telling detail:** acc/P/R/F1 are *identical* across re-runs — **the classification metrics are stable precisely because they are degenerate.** |
| **The official 188 run is a black box** | RC-7 | Medium | [E] Final metrics only — **no loss curve, no timer.** Whether the official run converged is **unobservable**. The CPU → GPU migration produced **no quantified speedup** because neither run was timed. |
| **Environment fragility** | RC-7 | Low | [E] `transformers` pinned to 4.44.0 solely to preserve a **deprecated `AdamW` import**; external gated-model dependency. |

---

# SECTION 5 — Version 2 Research Priorities

## A. IMMEDIATE VERSION 2 — the first experiments

| # | Item | RC | Why it belongs here |
|---|---|---|---|
| **A-0** | **Diagnostics: trivial-baseline control for regression; threshold-free ranking metric for classification** | RC-1 | **Cost nothing and may redefine the problem.** [E] The baseline **never computed** a mean-predictor control, and **never computed any ranking metric** (no ROC-AUC, no PR-AUC, no balanced accuracy, no MCC — neither report contains one). Both can be derived from artifacts that already exist. **[H]** If the probabilities (0.199–0.214) *rank* participants correctly, then a large part of the classification failure is a **decision-rule defect, not a learning defect** — and the cheapest experiment in the roadmap becomes the highest-yield. **[H]** If MAE 6.6547 equals the mean-predictor's MAE, the regression head has learned nothing and the −24.1 % claim must be formally withdrawn. Running expensive work before these two analyses risks optimizing a system whose actual failure mode is unidentified. |
| **A-1** | **Evaluation protocol** — participant-level cross-validation, frozen stratified splits, imbalance-appropriate metrics | RC-2, RC-7 | **The enabling weakness.** [E] At n = 18 with non-nested splits, **no Version 2 improvement could be shown to work, and no failure could be shown to have failed.** Fixing the model before fixing the ruler produces results that cannot be defended. This is a precondition, not a parallel workstream. |
| **A-2** | **Baseline re-measurement** — the *original, unmodified* trainer under the A-1 protocol | — | **Procedural necessity.** [E] The frozen Phase 9.5 numbers were computed on an 18-sample split that the new protocol replaces. **Comparing a cross-validated Version 2 result against an 18-sample baseline figure would repeat the exact error Phase 9.5 committed.** This produces `Baseline-CV`, the valid comparator. |
| **A-3** | **Convergence** — train to a stopping criterion instead of a fixed 3 epochs | RC-1 | [E] Loss was still falling by **3.92/epoch** at the cut-off. The least invasive of the three RC-1 conditions, and a documented deficiency in its own right. **Low complexity, Low risk, High compatibility.** |
| **A-4** | **Decision rule** — calibrated operating point instead of argmax at 0.5 | RC-1 | [E] **The model's probabilities span 0.199–0.214 — under a 0.5 threshold, no input can *ever* be classified positive.** The threshold is unreachable by construction. This changes **no training at all** and is therefore the safest possible intervention. |
| **A-5** | **Imbalance-aware objective** | RC-1, RC-9 | [E] Directly targets the documented condition — "unweighted CE on an imbalanced set, no class weighting." |
| **A-6** | **Loss-term rebalancing** (CE vs 0.5·MSE) | RC-1 | [E] Loss = `CE + 0.5·MSE`, and the reported totals (37.37 → 23.63) are far too large for 2-class CE to explain — the MSE term necessarily dominates the gradient budget. *(Per Phase 10.2 §J: inferred from magnitude, not measured — the per-term losses were never instrumented.)* Most invasive of the RC-1 block, hence last within it. |

## B. LATER VERSION 2 — only after Immediate is complete

| # | Item | RC | Why it belongs here (and not earlier) |
|---|---|---|---|
| **B-1** | **Activate the multimodal path** — supply non-zero audio/vision dimensions | RC-3 | **Severity is Critical; position is Later — deliberately.** [E] The system is currently 100 % textual *and* both heads are degenerate. **[H]** Introducing new modalities into a collapsed model would confound two effects: a modality contribution and a recipe fix. Establishing a **functioning** text baseline first means any subsequent audio/video gain is attributable **to the modality**. **Compatibility is High** — [E] the encoders already exist and are conditionally constructed; **the capability does not need to be built, it needs to be fed.** The work is in the data pipeline, not the model. |
| **B-2** | **Dimension-aware DP mechanism** | RC-4 | [E] The √d relationship is **confirmed**, not hypothesized (543² ≈ 296k params). It is the structural cause of the federation failure. It is scheduled after the task-level work because a privacy mechanism that preserves signal is only useful once there *is* signal to preserve. **First item with Medium (not High) architecture compatibility** — it changes the DP path, and that should be stated openly. |
| **B-3** | **Compact DAIC-derived update representation** | RC-4 | [E] The research question is precise: **what is the smallest object derived from the DAIC-trained model that still carries DAIC signal and survives the DP path?** [E] Since SNR = 1/√d, **d is the lever** — this follows from the confirmed mechanism, not from preference. **Highest-risk item in the Immediate/Later scope, and the most likely to fail** — which is acceptable *because* it is scheduled after the cheap wins, not before them. |

## C. FUTURE RESEARCH — intentionally deferred

| # | Item | RC | Why deferred |
|---|---|---|---|
| **C-1** | **Federate a DAIC-derived object** | RC-5 | **Critical severity, deferred position — and this is the clearest illustration of the difference between severity and readiness.** [E] It is **structurally impossible** until B-2 and B-3 succeed. Attempting it first would reproduce, exactly, the failure the baseline already recorded. It is the *payoff* of B-2 + B-3, not an independent workstream. |
| **C-2** | **Non-IID multi-client simulation** — ✅ **COMPLETE** (Phase 20; see §6 *Execution status*) | RC-5 | [E] Currently `devices = 1` and the three uploads differ **only by their DP noise draw**. **The one Future item NOT blocked by B-2/B-3** — it can be studied on the existing probe and could be pulled forward if an independent federation result is needed. Deferred by dependency-free choice, not necessity. **[E] That provision was exercised: C-2 was pulled forward and executed ahead of Row 10 as Phase 20. Manipulation/topology check PASS; no task metric computed by design. It does NOT unblock C-1 — B-2 and B-3 remain required.** |
| **C-3** | **ε–utility curve on a real corpus** | RC-5/RC-4 | [E] Blocked by C-1: the budget cannot be meaningfully studied until it protects something real. **Low complexity, Low risk — cheap *because* it is late.** |
| **C-4** | **DP mechanism sweep on a non-degenerate corpus** | RC-5 | [E] Blocked by C-1: 90 rows of saturated metric (accuracy flat at 1.0) prove the sweep cannot move until the corpus can degrade. A *consequence*, not a task. |

---

# SECTION 6 — Official Experiment Sequence

**Binding rules:** every experiment modifies **exactly one factor** relative to the immediately preceding *accepted* configuration. No experiment may be merged with another. [E] The justification is empirical: **Phase 9.5 changed two factors at once (dataset 113 → 188 *and* CPU → GPU) and consequently cannot attribute its own headline result.**

| # | Experiment | Single changed factor | Objective | Comparator | Success criteria |
|---|---|---|---|---|---|
| **0a** | Regression trivial-baseline control | *none — analysis only* | Determine whether the regression head learned anything | **P9.5 §6.1** (MAE 6.654737075169881) | A **definitive answer**. **[H]** Model MAE ≈ mean-predictor MAE ⇒ head confirmed a mean-estimator and **the −24.1 % claim is formally withdrawn**. Materially below ⇒ genuine signal established for the first time. |
| **0b** | Threshold-free ranking metric | *none — analysis only* | Determine whether the classifier's probabilities rank correctly | **P9.5 §7.1** (`eval_preds.csv`: P(pos) 0.199–0.214, all class 0) | A **definitive answer**. **[H]** AUC materially above chance ⇒ W#1 is substantially a **decision-rule defect** and Exp 4 becomes high-yield. AUC at chance ⇒ Exps 4–5 become mandatory. |
| **1** | Evaluation protocol | evaluation only | Make every future claim falsifiable | — | Metrics over all **188** participants with fold variance / CIs, on **frozen stratified participant-level splits**. |
| **2** | **Baseline re-measurement** | *nothing — original trainer, new protocol* | Produce `Baseline-CV`, the valid comparator | **P9.5 §6.1, §7.1** | `Baseline-CV` must reproduce the baseline's **qualitative signature: P/R/F1 = 0.0, all-negative predictions.** If it does not, the re-measurement is faulty — **not the baseline**. |
| **3** | Convergence | epoch policy | Establish whether prior-collapse is a budget artifact | `Baseline-CV` | Loss reaches a stopping criterion; regression MAE improves. **P/R/F1 may legitimately remain 0.0 — that is an informative outcome**, confirming RC-1 needs more than epochs. Runtime **will** increase (expected, not a failure). Modality ablation must stay 0/0. |
| **4** | Decision rule | decision threshold | Recover non-zero recall without retraining | `Baseline-CV` | **Recall > 0 and F1 > 0** — the first non-zero classification result in the project's history. **Regression MAE must be unchanged** (no training occurred; any movement indicates contamination). |
| **5** | Imbalance-aware objective | loss weighting | Improve classification beyond thresholding alone | `Baseline-CV` **and Exp 4** | Recall / F1 / PR-AUC improve over **both**. Regression MAE must not materially degrade. |
| **6** | Loss-term rebalancing | loss scaling | Break the joint-collapse of both heads | `Baseline-CV` **and Exp 5** | Classification improves **and/or** the **regression prediction spread widens** — [E] baseline spread is **0.06 on a 0–23 scale**; predicted-value variance is itself a success metric, independent of MAE. **Failure condition:** classification improves while MAE degrades toward the Exp-0a mean-predictor bound ⇒ one collapse merely traded for the other. |
| **7** | Multimodal activation | input modality | Exercise the multimodal capability for the first time | Best accepted text-only config (Exp 6) **+ P9.5 §6.2** | **`modality_ablation.json`: audio and/or video becomes non-zero.** [E] Baseline is **exactly 0.0** — the cleanest, most unambiguous success criterion in the roadmap. **Text contribution must remain materially positive** (baseline 2.3365); a collapse in text would mean fusion is discarding language, not augmenting it. |
| **8** | DP mechanism | DP path | Make privacy survivable | **P9 §6.2 + §8** | **Post-noise SNR materially better than 543 : 1**, at **ε ≤ 5.302585/update** and **cumulative ≤ 15.9078/round**. [E] **A utility gain bought by weakening an already-weak budget is not a gain — the privacy guarantee is the constraint, not the objective.** Cryptographic verification must remain intact. |
| **9** | Compact update representation | federated payload | Find the smallest DAIC-derived object that survives DP | Exp 6/7 task metrics + Exp 8 SNR | Update dimensionality and payload collapse **while DAIC task metrics hold**. [E] **A compact update that has lost the signal is not a success — it is Track A with extra steps.** That is precisely the trap the baseline fell into. |
| **10** | Federate a DAIC-derived object | federation source | Make the federation claim true | **P9 §4.4 + §6.3** | The **aggregated** model, evaluated on DAIC held-out data, produces **non-trivial** metrics. [E] Baseline global model scores **accuracy 1.0 with P/R/F1 = 0.0** (single-class demo). **Success = the aggregate is no longer trivially perfect.** All cryptographic guarantees preserved. |
| **11** | Non-IID multi-client | client topology | Exercise trimmed-mean on genuinely divergent updates | **P9 §6.4, §6.5** (`devices = 1`) | `devices` > 1 with heterogeneous partitions. **[H]** Task metrics may *degrade* under non-IID — **that is a legitimate scientific finding, not a failure.** The baseline offers no evidence either way. |
| **12** | DP sweep / ε–utility curve | sweep corpus | Produce a privacy–utility tradeoff for the first time | **P9 §8, §13.2** | **The sweep produces a non-flat curve.** [E] Baseline: accuracy **flat at 1.0** across **90 rows**. Success = utility **actually degrades** as noise rises — i.e. a tradeoff exists at all. ε accounting must remain a genuine RDP computation. |

## Execution status — recorded results

Rows are recorded here **only** when a frozen pre-registration, an execution and an
independent verification all exist. Absence from this table means *not yet recorded
here* — it does not mean not executed. [E]

| Row | Item | Status | Evidence |
|---|---|---|---|
| **9** | **B-3 — Compact update representation** (2nd attempt, B-3B) | ❌ **H₀ — pre-registered negative result, NOT an implementation failure.** A (payload) **PASS** at 1,055,101 B and B (NSR) **PASS** at 444.876535 with 25/25 updates clipped, but C (task signal) **FAIL**: DP ROC-AUC 0.503160920 lies below the frozen CI floor 0.5755177227. **The failure is upstream of DP** — the no-DP control scored 0.559165298, itself outside the frozen CI. Independent verification: 20/20 PASS. **B-3 remains UNSATISFIED** (H₀ twice, across two structurally different representation families). | `PHASE_21_B3B_DESIGN.md` (frozen, SHA `00f8e34b…c820590`) · `FINAL_B3B_CLOSURE.md` · `trainer_outputs/b3b_probe/b3b_summary.json` · `trainer_outputs/b3b_probe/verify_b3b_report.json` · **A 1,055,101 B ≤ 1,579,963 B** · **B NSR 444.876535 < 544.341809** · **C1 0.503160920 vs CI floor 0.5755177227** · **C2 +0.056004379 ≤ MDE 0.0578128739** · **verification 20 PASS / 0 FAIL / 0 PENDING** |
| **11** | **C-2 — Non-IID multi-client** | ✅ **COMPLETE — manipulation/topology check PASS.** Five disjoint clients with heterogeneous A2 partition produced higher pre-DP client-update divergence than the stratified/IID-like control on both L2 and cosine. **No task metric was computed by design; therefore no task-performance claim is made.** Independent verification: 25/25 PASS. **C-2 does not unblock Row 10; B-2 and B-3 remain required.** | `PHASE_20_C2_DESIGN.md` (frozen, SHA `687ea6a5…b2910a0`) · `trainer_outputs/c2_multiclient/c2_summary.json` · `trainer_outputs/c2_multiclient/verify_c2_report.json` · **L2 0.238848539 → 0.261621691** · **cosine 0.134495754 → 0.165267976** · **verification 25 PASS / 0 FAIL / 0 PENDING** |

**[E] Row 11 was executed ahead of Row 10**, under the provision recorded at
`PHASE_10_ROADMAP.md:235` — *"The one Future item NOT blocked by B-2/B-3 … could be
pulled forward if an independent federation result is needed."* The Section 6
single-factor rule is not violated: C-2's one changed factor is the **client
partition**, held against a stratified/IID-like control drawn from the frozen
`fold_manifest.json`, not against Row 10.

**[E] Scope of the Row 11 result.** C-2 is an aggregation/manipulation experiment.
It establishes that the A2 partition produced genuinely divergent client updates and
that the frozen trimmed-mean aggregator ran on them. **It establishes nothing about
task performance, generalisation, clinical utility or predictive validity**, and its
post-DP and aggregation-stage measures are noise-dominated (SNR 8.45e-04 at
d = 295,681) and cannot discriminate topologies. The primary endpoint has **one
effective replicate per arm**, so no confidence interval is available or claimed.

> **Rows 7 and 8 (B-2) have been executed and have recorded verdicts held in their
> own closure documents (`FINAL_EXP7_CLOSURE.md`, `FINAL_EXP8_CLOSURE.md`), which
> remain the authoritative record for those experiments. Their omission from this
> block is deliberate** — it must not be read as absence of a result.

## Why this order is scientifically justified

**Diagnostics precede everything because they cost nothing and may redefine the problem.** Two zero-cost analyses could collapse the scope of the entire roadmap.

**Measurement precedes models because it gates falsifiability.** [E] At n = 18, one sample is worth 5.6 % accuracy; any improvement would be indistinguishable from noise and any failure equally unfalsifiable.

**Re-measurement (Exp 2) is not optional.** It is the procedural crux: without it, every downstream comparison repeats Phase 9.5's own methodological error.

**Exps 3 → 6 attack RC-1 in ascending order of invasiveness,** so each result is attributable. [E] The reports name **three joint conditions** for prior-collapse — 3 epochs, unweighted CE, imbalance. **Changing them together would make it impossible to say which one mattered.**

**Exp 7 (multimodal) follows the RC-1 block** so that any audio/video gain is attributable to the modality rather than to a concurrent recipe fix.

**Exps 8 → 10 are a strict dependency chain, not a priority ordering.** [E] Federating DAIC signal (Exp 10) is *Critical* — more severe than Exps 8 and 9 — yet is scheduled **last of the three, because it is structurally impossible until they succeed.**

**Exps 11 → 12 are consequences.** Cheap *because* they are late.

---

# SECTION 7 — Evaluation Strategy

## 7.1 The frozen baseline and the Version 2 protocol coexist

This is the governing principle of all Version 2 evaluation:

> **The Phase 9 / Phase 9.5 reports remain FROZEN historical references. They are never rewritten, never re-scored, never re-run under the new protocol.**
>
> **The improved protocol becomes the Version 2 evaluation methodology — for Version 2 experiments only.**
>
> **`Baseline-CV` (Experiment 2) is the bridge between them: the original, unmodified trainer, re-measured under the Version 2 protocol.**

## 7.2 The three-layer comparison model

| Layer | What it is | Used for |
|---|---|---|
| **Historical baseline** | The frozen figures in P8B / P9 / P9.5 — MAE 6.6547, acc 0.7222, P/R/F1 = 0.0, text/audio/video 2.3365/0/0, computed on an 18-sample split | The permanent record of **what the original implementation did**. Cited for *qualitative* claims (e.g. "audio contribution was exactly 0.0") and for the diagnostics in Exps 0a/0b. |
| **`Baseline-CV`** | The original trainer under the Version 2 protocol | The **quantitative comparator** for every Version 2 experiment from Exp 3 onward. |
| **Version 2 results** | Each single-factor experiment | Compared against `Baseline-CV` **and** against the immediately preceding accepted configuration. |

## 7.3 Mandatory reporting discipline

Every Version 2 result must state **which layer it is compared against.** A claim of the form *"Version 2 achieves MAE X, an improvement on the 6.6547 baseline"* is **inadmissible** if X was computed under cross-validation, because the two numbers come from different evaluation universes. [E] Conflating them would be the identical error that voided Phase 9.5's own comparison table.

## 7.4 Two metrics that must never be used naively

**Accuracy is not a success metric anywhere in Version 2.** [E] For an all-negative predictor, accuracy ≡ the negative rate: 6/11 = 0.5455, then 13/18 = 0.7222 — *exactly* the majority-class rate. **A Version 2 experiment that recovers recall will very likely *reduce* accuracy below 0.7222, and that is a success, not a regression.** Any evaluation that ranks configurations by accuracy will systematically prefer the degenerate model. The frozen reports already say this: *"Reporting this as improved classification would be wrong."*

**MAE improvement is not, by itself, evidence of learning.** [E] The baseline's `pred_phq` spread of **0.06 on a 0–23 scale** means MAE can move purely by relocating a constant. **Every Version 2 regression claim must therefore be reported alongside (a) the trivial mean-predictor MAE from Exp 0a and (b) the prediction-variance statistic.** This requirement exists precisely because Phase 9.5's −24.1 % headline could not survive it.

## 7.5 Universal invariants

A violation invalidates the run **regardless of its headline metric**:
- **Modality ablation = exactly 0.0 for audio and video in all experiments before Exp 7.** Any non-zero value earlier means an uncontrolled change entered the pipeline.
- **Dataset held constant:** 188 records, pos 45 / neg 143, 440 excluded, 487 recovered.
- **Exactly one changed factor** per experiment.
- **Cryptographic verification preserved** in all federated experiments (SHA-256 chunks, hash match, `strict=True` load, TPM ECDSA, HMAC chaining). [E] These already work — they must not be traded away.

---

# SECTION 8 — Scope Boundaries

## 8.1 IN SCOPE for Version 2

| In scope | Traces to |
|---|---|
| **Training-recipe research** — convergence policy, decision-threshold calibration, imbalance-aware objectives, loss-term balance | RC-1 (Weaknesses #1, #2, #8, #9) |
| **Evaluation-protocol research** — participant-level cross-validation, frozen stratified splits, imbalance-appropriate and threshold-free metrics, uncertainty reporting | RC-2 (Weaknesses #6, #13) |
| **Multimodal data-pipeline research** — carrying the audio and video streams that already exist in all 189 archives into the record schema, so the existing encoders activate | RC-3 (Weakness #5) |
| **DP-mechanism research** — dimension-aware clipping, noise calibration, update-dimensionality reduction | RC-4 (Weaknesses #3, #7) |
| **Federated-substance research** — federating a DAIC-derived object; non-IID multi-client simulation; ε–utility curves on a real corpus | RC-5 (Weaknesses #4, #10, #11, #12) |
| **Diagnostic analyses of existing artifacts** — trivial-baseline controls, ranking metrics | RC-1 |

## 8.2 OUT OF SCOPE for Version 2

| Out of scope | Why (Phase 10.3 evidence) |
|---|---|
| **Fixing the orchestrator's round-state volatility / `round_id` collision** | [E] **PERF = 0.** Real operational pain — it already cost the project an experiment — but it **cannot move a single metric.** RC-6 produces **zero Critical weaknesses**. Engineering hygiene, not research. |
| **Raising the LDA 20-participant cap** | [E] **Conclusively inert.** LDA outputs are **byte-identical across both phases** — the 66 % dataset upgrade changed this stage by *zero bytes*. It feeds **no reported metric**. Preserved by mandate. |
| **Raising the 15 s SubmitReceipt timeout** | [E] A **symptom**, not a cause. Removing it changes nothing while the √d DP problem stands; it resolves automatically if Exp 9 succeeds. |
| **Adding wall-clock instrumentation as a workstream** | [E] Affects runtime *reporting* only. Zero metric impact. (Worth capturing as a side-effect of Exp 2 — never as a task.) |
| **Unpinning `transformers` / removing the gated-model dependency** | [E] Supply-chain risk with **zero** metric impact. |
| **"Fixing" the DP global-model hash nondeterminism** | [E] **Not a defect — a correctness property.** DP *requires* fresh noise. Fixing it would break the guarantee. |
| **Recovering participant 440** | [E] Upstream file corruption. Unfixable by any project code. Remains a documented exclusion. |
| **Rewriting, re-scoring, or amending the frozen baseline reports** | Section 2. They are the historical record. |
| **Bundling multiple changes into one experiment** | [E] The error that voided Phase 9.5's own attribution. |
| **Introducing research directions not justified in Phase 10** | Every recommendation must trace to a validated weakness ID. |

## 8.3 Deferred but NOT ignorable — an engineering-hygiene track

Two weaknesses scored **PERF = 0** and are therefore excluded from the Version 2 *research* scope — **but only because a human intercepted both:**

- [E] The **extractor AppleDouble defect** was caught **solely because the sidecar's bytes happened to break UTF-8 decoding**; a cleanly-decoding sidecar "would have silently produced a garbage record."
- [E] The **label-source drift** would have **silently dropped 19 of the 76 newly-added participants**, gutting the entire point of Phase 9.5, with no error raised.

These are **silent-corruption hazards, not cosmetic debt.** They return nothing in performance and should consume no research effort — but **every Version 2 experiment inherits the same data-staging path, and a data-integrity failure would invalidate the experiment running on top of it.** They belong in a hygiene track running *alongside* Version 2, not inside it.

---

# SECTION 9 — Phase 10 Completion Statement

## 9.1 What Phase 10 accomplished

| Sub-phase | Delivered |
|---|---|
| **10.1 — Evidence Collection** | A complete, interpretation-free extraction of every factual claim in the Phase 9 and Phase 9.5 reports: dataset, configuration, metrics, classification and regression behaviour, federation findings, runtime observations, and all 26 stated limitations. |
| **10.2 — Root Cause Analysis** | All 25 distinct weaknesses traced to **seven root causes (RC-1 … RC-7)**. Established that the reports' 26 separately-listed limitations are not 26 independent problems. Confirmed the √d noise mechanism **from the baseline's own sweep arithmetic** (`none` → L2-after 1.00; `gaussian` → 543.49; 543² ≈ 296k params). Identified that **both prediction heads collapsed to the label prior** — proven by P(positive) tracking the class prior across both runs (0.365 at a 32.7 % prior, 0.20 at a 23.9 % prior) and by a `pred_phq` spread of **0.06 on a 0–23 scale**. |
| **10.3 — Weakness Prioritization** | All 25 weaknesses scored on performance, scientific validity, usability and reproducibility; assigned severity; ranked 1–25. Established the decisive asymmetry: **RC-6 and RC-7 generate 28 % of the weakness count and 0 % of the Critical severity.** Identified the **Top Five** research drivers (one per root cause RC-1 … RC-5) and the items that should **not** be worked on. |
| **10.4 — Version 2 Research Design** | A 12-experiment, single-factor sequence with objectives, comparators, and success criteria for every Critical and High weakness; the dependency chain; the deferral rationale; and the evaluation strategy that keeps the frozen baseline intact. |
| **10.5 — This roadmap** | The consolidated strategic document governing all remaining Capstone work. |

## 9.2 The finding that defines Version 2

> [E] **Phase 9.5's real contribution was not the MAE number — it was the elimination of the last available excuse.**
>
> With the corpus complete (188 of 189) and precision, recall and F1 still **exactly 0.0**, *"we need more data"* is permanently unavailable as an explanation for anything. Every remaining failure in this system is traceable to **RC-1 through RC-5** — all five of which sit in the training recipe, the privacy mechanism, or the data pipeline. **None sits in the data.**

And the finding that defines the *approach*:

> [E] **The six Critical weaknesses are the quietest defects in the system.** Nothing crashes, every gate returns `[PASS]`, the cryptography verifies, and the pipeline runs end-to-end — and yet not one of the project's four headline claims is currently supported by its own evidence. **Triage by symptom would fix exactly the wrong half.**

## 9.3 Formal completion

**✅ Reproduction is complete.**
The original implementation was recovered, executed unchanged, and re-run on the complete DAIC-WOZ corpus. The trainer is SHA-256 identical to the original; the Phase 9.5 audit passed all twelve criteria. **No source file was modified across the reconstruction.**

**✅ The baseline is frozen.**
`PHASE_8B_EXECUTION_REPORT.md`, `PHASE_9_BASELINE_EVALUATION_REPORT.md` and `PHASE_9.5_COMPLETE_DATASET_REPORT.md` are the official reference baseline. **They will not be modified.** All future experiments compare against them. MongoDB `federated` and the preserved artifact directories remain untouched.

**✅ The Version 2 roadmap is established.**
Seven root causes identified. Twenty-five weaknesses prioritized. Twelve single-factor experiments sequenced, each with an objective, a comparator, and falsifiable success criteria. Scope boundaries drawn in both directions — what to pursue, and what to deliberately leave alone.

**✅ Future work proceeds through controlled experiments, one change at a time.**
This is not a stylistic preference. [E] It is the direct lesson of the baseline itself: **Phase 9.5 changed two factors at once and consequently cannot attribute its own headline result.** Version 2 will not repeat that. Every experiment changes exactly one factor, is compared against a valid comparator, and is reported with the distinction between **evidence** and **hypothesis** made explicit.

---

**PHASE 10 COMPLETE — BASELINE ANALYZED, FROZEN, AND SUPERSEDED BY A CONTROLLED RESEARCH PROGRAM.**

*No code generated. No implementation modified. No architecture redesigned. No experiments merged. Every recommendation traceable to a validated Phase 10 finding.*
