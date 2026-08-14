# PHASE 11.4 — OFFICIAL VERSION 2 EVALUATION PROTOCOL
## The Methodology Every Version 2 Experiment Must Follow

**Phase:** 11.4 — Evaluation Protocol Design (Experiment 1, design half)
**Type:** Methodology specification — no training, no code, no implementation
**Authority basis:** `PHASE_10_ROADMAP.md` §6 (Exp 1), §7 (Evaluation Strategy) · Phase 11.0–11.3 diagnostic evidence
**Status:** This document governs the *evaluation* of all Version 2 experiments. It does not modify the roadmap; it operationalizes §7 of it.

Convention used everywhere in Version 2: **[E]** = observed evidence (a measured quantity) · **[I]** = interpretation. This document also marks each rule as **[MANDATORY]** or **[OPTIONAL]**.

---

## 1. Purpose

Phases 11.1–11.3 established, at verified precision, that both heads of the frozen model collapsed to the label prior, and that the frozen 18-sample evaluation is too small to measure effect sizes (ROC-AUC 95% CI [0.250, 0.787]). Those two facts define what an evaluation protocol must guarantee:

1. that a Version 2 result can be **distinguished from a trivial predictor** (11.1 showed MAE alone cannot do this — a constant scored 6.6545), and
2. that a Version 2 result can be **distinguished from noise** (11.2 showed a single 18-sample split cannot do this).

This protocol makes both guarantees mandatory and uniform, so that every Version 2 experiment is comparable, falsifiable, and honestly reported against the frozen baseline.

---

## 2. Principles

| # | Principle | Rule | Trace |
|---|---|---|---|
| P-1 | **Frozen baseline is immutable** | **[MANDATORY]** The three frozen reports (8B, 9, 9.5) and their numbers are never rewritten, re-scored, or amended. They are the historical record. | Roadmap §2, §7.1 |
| P-2 | **One factor at a time** | **[MANDATORY]** Each experiment changes exactly one factor relative to the immediately preceding *accepted* configuration. No bundled changes. | Roadmap §6; [E] Phase 9.5 changed two factors and cannot attribute its result |
| P-3 | **Reproducibility** | **[MANDATORY]** Every experiment fixes and records the **split seed / frozen manifest**, code version (SHA), and environment. Reproducibility here means the **manifest, environment, and aggregate statistics (mean ± CI over repeats) are regenerable** — *not* that a single training run is bitwise-identical. Training stochasticity (torch) is deliberately left free across repeats to measure the noise band (§7). A result whose **aggregate statistics** cannot be regenerated is not a result. | Roadmap §7.5; RC-7; Phase 11 DP-2 |
| P-4 | **Participant-level independence** | **[MANDATORY]** No participant may appear in both training and evaluation within any fold. Splitting is by participant, never by row/segment. | 11.3 §6; RC-2 |
| P-5 | **Evidence-first conclusions** | **[MANDATORY]** Every claim separates [E] from [I]; no causal claim without a controlled single-factor comparison and a stated uncertainty. | 11.1–11.3; Roadmap §7.3 |
| P-6 | **Trivial-baseline controls are permanent** | **[MANDATORY]** Every result ships with its trivial controls (mean/median predictor for regression; majority-class predictor for classification). A model is only credited for what it does *above* these. | 11.1; Roadmap §7.4 |
| P-7 | **Universal invariants** | **[MANDATORY]** Dataset held constant (188 / 45 / 143); modality ablation = exactly 0.0 for audio/video before Exp 7; cryptographic verification preserved in federated experiments. A violation invalidates the run regardless of its headline. | Roadmap §7.5 |

---

## 3. Primary Metrics — [MANDATORY]

Primary metrics are **required in every experiment** and are the only metrics on which accept/reject decisions may be based.

### 3.1 Regression (PHQ score)

| Metric | Why included | Trace |
|---|---|---|
| **MAE** | The frozen baseline's headline metric; retained for continuity. **[E] By itself it is insufficient** — 11.1 showed a constant achieves MAE 6.6545. | 11.1 |
| **RMSE** | Complements MAE by penalizing large errors more heavily; a constant predictor and a discriminating one with the same MAE differ in RMSE. Exposes whether errors are uniform (constant) or heteroscedastic (learning). | Justified addition |
| **Prediction variance / spread** | **The decisive companion to MAE.** [E] 11.1 found spread 0.058 on a 0–23 scale (var 3.5e-4). A regression head that has learned *must* produce participant-varying predictions; variance near zero signals collapse **regardless of MAE**. | 11.1; Roadmap §7.4 |
| **Trivial-baseline gap (Δ vs mean & median predictor)** | The only way to certify learning. **[E] A Version 2 regression claim is inadmissible unless MAE beats the mean-predictor baseline (frozen floor: 6.3882).** | 11.1; Roadmap §7.4 |

### 3.2 Classification (PHQ > 10)

| Metric | Why included | Trace |
|---|---|---|
| **ROC-AUC** | **Threshold-free ranking** — the core question of 11.2. Independent of the decision boundary that collapsed the frozen output. Frozen floor: **0.523**. | 11.2 |
| **PR-AUC (Average Precision)** | The imbalance-appropriate ranking metric (23.9% positive). More informative than ROC-AUC under skew. Frozen floor **0.311** vs prevalence **0.278**; a credible classifier must clear prevalence by more than the noise band. | 11.2; RC-1 |
| **Balanced Accuracy** | Mean of per-class recall; **immune to the majority-class illusion.** [E] Plain accuracy of the frozen all-negative model is 0.7222 = the negative rate. Balanced accuracy of that model is 0.5 — the honest number. | 11.2; Roadmap §7.4 |
| **MCC (Matthews Correlation Coefficient)** | Single-number summary robust to imbalance; 0 for any degenerate/all-one-class predictor. A clean "did it beat trivial?" scalar. | Justified addition |
| **Precision / Recall / F1** at a **declared** operating point | Retained for continuity with the frozen reports (P/R/F1 = 0/0/0). **[MANDATORY]** the operating point (threshold) must be **declared and justified**, not silently defaulted to 0.5, and must be **selected on the training / inner-validation folds only — never on the held-out test fold** (that would be a leakage vector; see Risk R-2). | 11.2; Roadmap §6 #4 |

**Threshold-free vs threshold-dependent primaries [clarification].** ROC-AUC and PR-AUC are **threshold-free**. Balanced accuracy, MCC, and P/R/F1 are computed **at the declared operating point** (§3.2 last row) and must never be reported at the degenerate 0.5 default on a collapsed model — that merely reproduces the frozen 0/0/0. When a model is degenerate, the threshold-free metrics carry the primary verdict and the operating-point metrics are reported as **threshold-conditional**.

### 3.3 Accuracy — [MANDATORY EXCLUSION]

**Plain accuracy is never a primary metric and never drives an accept/reject decision.** [E] For an all-negative predictor accuracy equals the negative rate (6/11 = 0.5455, then 13/18 = 0.7222). [I] A Version 2 experiment that recovers recall will very likely *lower* accuracy below 0.7222 — and that is a success, not a regression. Accuracy may be **reported for completeness only**, always beside balanced accuracy. (Roadmap §7.4.)

---

## 4. Secondary Diagnostics — [OPTIONAL]

These **support interpretation** of the primary metrics. They may accompany an experiment and are encouraged where they clarify a result, but **they never replace a primary metric and never, on their own, justify accepting an experiment.**

| Diagnostic | What it clarifies |
|---|---|
| **Prediction distribution / histogram** | Whether outputs collapse (the 11.1/11.2 signature) or spread across participants. |
| **Calibration (reliability curve)** | Whether predicted probabilities mean what they claim. [E] 11.2 found P(pos) pinned near the prior (0.205); calibration shows if Version 2 has broken that pinning. |
| **Threshold sensitivity sweep** | How P/R/F1 move across thresholds — *characterization, not optimization on the test fold.* | 
| **Confusion matrix** | The concrete error structure behind balanced accuracy / MCC. |
| **Regression residual analysis** | Whether residuals are structured (learning) or flat around a constant (collapse); pairs with the variance metric. |
| **Per-fold breakdown** | Exposes whether a headline is driven by one lucky fold. |

---

## 5. Comparison Framework — [MANDATORY]

### 5.1 The three-layer comparison model (Roadmap §7.2)

| Layer | What it is | Used for |
|---|---|---|
| **Historical baseline** | Frozen P8B/P9/P9.5 figures (MAE 6.6547, acc 0.7222, P/R/F1 0/0/0, ablation 2.3365/0/0), on the 18-sample split | **Qualitative** claims only (e.g. "audio contribution was exactly 0.0") and the 11.1/11.2 diagnostics. **Never** a quantitative target for a cross-validated number. |
| **`Baseline-CV`** | The original, unmodified trainer under this protocol (Phase 11.6) | **The quantitative comparator** for every Version 2 experiment from Exp 3 onward. |
| **Version 2 result** | Each single-factor experiment | Compared against `Baseline-CV` **and** the immediately preceding accepted configuration. |

### 5.2 Mandatory reporting discipline (Roadmap §7.3)

- **[MANDATORY]** Every result states **which layer** it is compared against.
- **[MANDATORY]** A claim of the form *"Version 2 MAE X beats the 6.6547 baseline"* is **inadmissible** if X was computed under cross-validation — the two come from different evaluation universes. [E] Conflating them is the error that voided Phase 9.5's own comparison table.
- **[MANDATORY]** Every comparison reports **fold variance / confidence intervals**, not point estimates alone.
- **[MANDATORY]** The **improvement must exceed the empirical noise band** (the minimum detectable effect established by `Baseline-CV` in 11.6). A change within the noise band is reported as "no detectable effect."
- **[MANDATORY — clarification]** The numeric **"frozen floor" values cited in §3** (ROC-AUC 0.523, PR-AUC 0.311, mean-predictor MAE 6.3882) are **18-sample reference points, illustrative only.** The **operative** quantitative floors for accept/reject are their **`Baseline-CV` re-measurements** (with CIs) from Phase 11.6. This preserves §5.1: no 18-sample frozen number is ever a direct cross-validated target.

### 5.3 Required artifacts per experiment

| Artifact | Requirement |
|---|---|
| **Primary-metrics table** (§3), with trivial-baseline controls | **[MANDATORY]** |
| **Comparison table** vs `Baseline-CV` and vs previous accepted config, with CIs | **[MANDATORY]** |
| **Single-changed-factor statement** and seed/split/SHA/environment record | **[MANDATORY]** |
| **Universal-invariant check** (dataset constant; ablation 0/0 pre-Exp 7; crypto intact) | **[MANDATORY]** |
| Secondary-diagnostic plots (distribution, calibration, residuals, confusion) | **[OPTIONAL]** |

---

## 6. Interpretation Guidelines — [MANDATORY]

1. **Separate [E] from [I] in every statement.** A measured number is [E]; any statement about *why* is [I].
2. **No causal claim without a single-factor control.** [E] Phase 9.5's inability to attribute its −24.1% headline (two factors changed) is the standing example. Attribution requires that exactly one factor differ from the comparator.
3. **MAE improvement is not evidence of learning.** [E] 11.1: a constant scored 6.6545. Every regression claim must be reported beside (a) the mean-predictor MAE and (b) prediction variance.
4. **A high accuracy is not evidence of classification.** [E] 11.2/§7.4: 0.7222 is the negative rate. Judge classification by ROC-AUC, PR-AUC, balanced accuracy, MCC.
5. **A moved threshold is not a learned classifier.** [E] 11.2: re-thresholding recovers recall (1.0 at t=0.20) but precision stays at prevalence (0.385) when ranking is at chance. Distinguish *mechanical* recall recovery from *discrimination*.
6. **State uncertainty with every effect size.** A point estimate without a CI or fold variance is not a finding.
7. **An informative null is a valid outcome.** [E] Roadmap §6: P/R/F1 may legitimately remain 0.0 after Exp 3 — that is evidence RC-1 needs more than epochs, not a failed experiment.

---

## 7. Statistical Considerations

*(No calculations required here; this section sets the standard of caution.)*

- **Small-sample fragility of the frozen split.** [E] The frozen evaluation is n = 18 (5 positives). One flipped prediction moves accuracy 5.6%; the ranking AUC 95% CI is [0.250, 0.787]; the regression trivial-gap of ~0.27 MAE carries wide uncertainty. [I] Point estimates from this split are directional, not conclusive.
- **Why `Baseline-CV` (Phase 11.6) is still required.** [I] The frozen numbers cannot serve as a quantitative comparator for cross-validated Version 2 results (different evaluation universe) and cannot themselves resolve effect sizes at this sample size. `Baseline-CV` re-measures the *unmodified* trainer under this protocol over all 188 participants, yielding the valid comparator **and** the empirical noise band / minimum detectable effect that §5.2 requires. Until it exists, no Version 2 quantitative comparison is admissible.
- **Interpret significance cautiously until cross-validation is complete.** [I] Any p-value, CI, or "significant improvement" computed on an 18-sample split (as in 11.2, where permutation p = 0.446) is provisional. Significance claims become meaningful only once effects are measured across participant-level folds with reported variance. Until then, prefer the language "no detectable effect / directional signal" over "significant / not significant."
- **Cross-validation design (recommended, to be frozen in Phase 11.6).** [I] Participant-level **stratified k-fold**, repeated, with the split manifest frozen and versioned. **k = 5 is the recommended default**: with 45 positives it yields ≈9 positives per held-out fold — enough to compute PR-AUC/recall without the ~4-positive fragility a larger k would reintroduce. Repeats (R) characterize run-to-run variance and set the noise band. The **fold count and manifest are finalized and frozen in 11.5/11.6**, not here; this protocol only fixes the *method* (participant-level, stratified, frozen, repeated).

---

## 8. Reporting Template — [MANDATORY]

Every Version 2 experiment report uses this structure, in this order:

```
# EXPERIMENT <n> — <title>
1. Objective            — the single question, and the roadmap experiment it maps to
2. Single Changed Factor — exactly what differs from the comparator (one factor)
3. Configuration Record  — seeds, frozen split manifest ID, code SHA, environment
4. Primary Metrics       — §3 table, WITH trivial-baseline controls (mandatory)
5. Comparison            — vs Baseline-CV and vs previous accepted config; CIs/fold variance (mandatory)
6. Universal-Invariant Check — dataset 188/45/143; ablation 0/0 (pre-Exp 7); crypto intact
7. Secondary Diagnostics — distributions / calibration / residuals / confusion (optional)
8. Interpretation        — [E] vs [I]; uncertainty stated; no unsupported causal claim
9. Decision              — accept / reject / informative-null, against the pre-stated success criterion
```

**Accept/reject rule [MANDATORY]:** an experiment is accepted only if a **primary** metric improves over `Baseline-CV` by **more than the noise band**, the trivial-baseline controls are cleared, and every universal invariant holds. Secondary diagnostics may explain a result but cannot rescue one that fails this rule.

---

## 9. Conclusion

This protocol converts the lessons of Phases 11.1–11.3 into binding evaluation rules. Its non-negotiables: the frozen baseline is immutable; experiments change one factor at a time; participant-level independence is absolute; trivial-baseline controls ship with every result; regression is judged by MAE **plus** variance **plus** the trivial gap; classification is judged by ROC-AUC / PR-AUC / balanced accuracy / MCC — **never** by plain accuracy; every comparison names its layer and reports uncertainty; and no quantitative Version 2 comparison is admissible until `Baseline-CV` (Phase 11.6) supplies the valid comparator and the noise band.

[I] The protocol deliberately makes it *hard* to claim success and *easy* to detect collapse — because the frozen baseline demonstrated that a system can pass every gate, run end-to-end, and still have learned nothing. Version 2 will be measured so that this cannot happen silently again.

**Phase 11.4 outcome: the Version 2 evaluation methodology is specified. The split manifest and fold count are finalized and frozen in Phase 11.5/11.6.**

---

*Methodology only. No implementation, no training, no code change, no architecture change, no roadmap modification. Mandatory rules and optional analyses are marked throughout; every rule traces to a completed diagnostic (11.0–11.3) or to `PHASE_10_ROADMAP.md` §6–§7.*
