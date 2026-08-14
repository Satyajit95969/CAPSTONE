# PHASE 11.3 — DIAGNOSTIC SYNTHESIS
## Integrated Evidence Assessment of the Frozen Baseline

**Phase:** 11.3 — Synthesis of Diagnostics A and B
**Type:** Analysis / synthesis only — no experiments, no retraining, no code change
**Inputs:** `PHASE_11_DIAG_A_REGRESSION_CONTROL.md` (11.1) · `PHASE_11_DIAG_B_THRESHOLD_FREE_ANALYSIS.md` (11.2)
**Roadmap trace:** `PHASE_10_ROADMAP.md` §5, §6 (Exp 0a/0b), §7 (Evaluation Strategy), §9 (root causes RC-1…RC-7)

Throughout: **[E]** = observed evidence measured from frozen artifacts in 11.1/11.2 · **[I]** = interpretation. Every claim traces to a completed diagnostic; no new measurement is introduced.

---

## 1. Executive Summary

Two independent, threshold-aware and threshold-free diagnostics were run on the frozen Phase 9.5 evaluation, each after a **verification gate** that reproduced the frozen metrics exactly (regression MAE `6.654737075169881`; classification acc `0.7222`, P/R/F1 `0/0/0`). The reconstruction is therefore trustworthy, and the two diagnostics converge on a single conclusion:

> [I] **Both output heads of the frozen model have collapsed to the label prior. Neither head discriminates between participants.** The regression head emits a near-constant ≈5.05 (variance 3.5e-4); the classification head emits P(positive) ≈ 0.205 ≈ the class prior (variance 3.2e-5), with ranking indistinguishable from chance (ROC-AUC 0.523, 95% CI [0.250, 0.787]).

[I] This is a **single shared failure mode expressed twice**, not two separate defects. It directly supports Phase 10 root cause **RC-1** (prior-collapse driven by the training recipe) and, as a by-product of the diagnostics' own inability to reach conclusive verdicts at n = 18, it directly supports **RC-2** (the evaluation is too small to be falsifiable). The privacy, federation, and multimodal root causes (RC-3…RC-6) were **not exercised** by these diagnostics and remain as documented in the frozen reports.

---

## 2. Evidence Review

### 2.1 Phase 11.1 — Regression Trivial-Baseline Control (Experiment 0a)

**Verification.** [E] The reconstructed validation targets reproduced the frozen model MAE to full precision (`6.654737075169881`, bit-exact), confirming the 18 validation participants and their PHQ labels.

**Observed [E]:**
- Model prediction spread = **0.058** on a 0–23 scale (5.027–5.086); prediction variance **3.5e-4**; mean prediction 5.055.
- A single constant fixed at the model's own mean prediction (5.055) reproduces its MAE to within **0.000264**.
- Mean Predictor (train mean 6.506) → MAE **6.3882**; the model (6.6547) is **worse by 0.267**.
- Median Predictor (train median 5.0) → MAE **6.6667**; the model is **tied** (better by 0.012, ≈0.18%).

**Interpretation [I]:** The regression head is a **constant/mean estimator**. It did not learn participant-level information; its MAE is that of a fixed number, and that number is not even MAE-optimal (it loses to the training mean). The Phase 9.5 −24.1% MAE headline is **withdrawn as evidence of learning** (the numeric value remains a valid frozen record); the drop is fully explained by a constant relocating between two different validation sets.

### 2.2 Phase 11.2 — Threshold-Free Ranking Analysis (Experiment 0b)

**Verification.** [E] The reconstructed labels reproduced the frozen classification metrics exactly (acc 0.7222; P/R/F1 0/0/0; all 18 predictions negative).

**Observed [E]:**
- P(positive) range = **0.0153** (0.199–0.214); variance **3.2e-5**; mean **0.205**.
- Mean P(pos) 0.205 ≈ the class prior (full corpus 0.239; validation 0.278).
- **ROC-AUC 0.523**, bootstrap 95% CI **[0.250, 0.787]**, permutation p = **0.446**.
- PR-AUC 0.311 vs prevalence baseline 0.278.
- Mean P(pos) is **lower** for depressed (0.203) than non-depressed (0.205) — rank direction slightly wrong; Spearman ρ 0.315 (p 0.20) and Kendall τ 0.205 (p 0.25), neither significant.
- Threshold sweep: the frozen 0.5 boundary sits far above the whole probability cluster; lowering to 0.20 yields recall 1.0 but precision 0.385 ≈ prevalence.

**Interpretation [I]:** The classifier carries **no detectable ranking information**. The all-negative output is a **mechanical threshold artifact** (all probabilities ≪ 0.5), but re-thresholding recovers only the *quantity* of positives, not *discrimination*, because the underlying ordering is at chance. The failure is substantially a **learning** failure, not a decision-rule defect.

---

## 3. Cross-Diagnostic Findings

[I] Placing the two heads side by side reveals one mechanism, not two:

| Property | Regression head (11.1) | Classification head (11.2) | Shared pattern |
|---|---|---|---|
| **Prediction collapse** | [E] spread 0.058 / 23 (0.25% of range); var 3.5e-4 | [E] spread 0.0153; var 3.2e-5 | [I] Both emit a near-constant; participant-level variance is negligible in both. |
| **Prior tracking** | [E] constant ≈5.05, near the training central tendency (mean 6.51 / median 5.0) | [E] P(pos) ≈0.205 ≈ class prior (0.239 / 0.278) | [I] Each head has settled on its own **label prior**, the trivial minimizer of its loss term. |
| **Participant discrimination** | [E] worse than mean predictor; tied with median predictor | [E] ROC-AUC 0.523, CI straddles chance; wrong rank direction | [I] Neither head separates participants beyond a constant. |

**Synthesis [I]:** A model whose CE head outputs the base rate for everyone and whose MSE head outputs the central PHQ for everyone is a model that minimized its loss by learning the **marginal label distribution** rather than the **conditional** map from text to outcome. This is exactly the "collapsed to the prior" behaviour Phase 10 named. The two diagnostics are independent confirmations — one on the raw regression scale, one on the probability rank scale — of the same collapse.

**Relationship to Phase 10 root causes [I]:**
- The shared collapse is the signature of **RC-1** (three joint conditions: 3 epochs, unweighted CE, class imbalance). Both heads reaching their respective priors is the predicted symptom.
- The diagnostics' inability to *resolve* the collapse conclusively — the AUC CI [0.250, 0.787], the n=18-fragile MAE gaps — is itself a live demonstration of **RC-2** (evaluation too small to falsify anything).

---

## 4. Root Cause Mapping

Support status is assessed **strictly against what diagnostics 11.1 and 11.2 measured.** "Still unverified" means *not exercised by these diagnostics* — several such causes remain independently documented in the frozen reports; the synthesis simply adds no new evidence for them.

| RC | Phase 10 claim | Status after 11.1/11.2 | Why |
|---|---|---|---|
| **RC-1** | Both heads collapsed to the label prior (training recipe) | **Directly supported** | [E] Regression = constant ≈5.05 (var 3.5e-4), worse than mean predictor; classification P(pos) ≈ prior (var 3.2e-5), AUC ≈ chance. Two independent confirmations of prior-collapse. |
| **RC-2** | Evaluation on n≈18 cannot support/refute any hypothesis | **Directly supported** | [E] Both diagnostics hit the sample-size wall: AUC 95% CI [0.250, 0.787]; MAE gaps of ~0.27 with wide uncertainty. The diagnostics could establish *internal* collapse but not *external* effect sizes — precisely RC-2's prediction. |
| **RC-3** | Audio/video contribute exactly 0.0 (dead multimodal path) | **Still unverified (by these diagnostics)** | [I] 11.1/11.2 analyzed text-derived predictions only; modality ablation was not re-exercised. RC-3 remains supported by frozen Phase 9.5 §6.2 independently, but this synthesis adds nothing to it. |
| **RC-4** | DP √d mechanism destroys the update (543:1) | **Still unverified (by these diagnostics)** | [I] No privacy pathway was touched. Remains as documented in frozen P9 §6.2/§8. |
| **RC-5** | Federated object carries no DAIC signal (demo probe) | **Still unverified (by these diagnostics)** | [I] No federation artifact was analyzed. Remains as documented in frozen P9 §6.3. |
| **RC-6** | Federation not restartable (round state in memory) | **Still unverified (by these diagnostics)** | [I] Orchestrator state was not exercised; out of scope for ML-metric diagnostics. |
| **RC-7** | Reproducibility gaps (only split seeded; torch unseeded; non-nested val sets) | **Partially supported** | [E] The diagnostics *confirmed* the sub-claim that "only the split is seeded": the seeded split reproduced the frozen metrics **exactly**, so the split is deterministic. [I] But the *harmful* RC-7 claims — torch-unseeded run-to-run drift, cross-length permutation non-comparability, black-box official run — were not exercised and remain unverified here. |

---

## 5. Version 2 Implications

*The roadmap is not modified. Experiment order is unchanged. Only interpretation and priority emphasis shift, consistent with the roadmap's own 0a/0b branch logic.*

[I] Consequences of the confirmed joint prior-collapse:

- **Exp 3 (Convergence)** — remains the least-invasive first probe. The collapse is consistent with an under-trained model that reached only the prior; whether more training alone widens either head's output variance is now a concrete, measurable question. **Priority: unchanged (first).**
- **Exp 4 (Decision rule)** — reframed. 11.2 shows it will **mechanically** yield the project's first non-zero recall (roadmap §6 #4 already predicts this), but its ceiling is low because the ordering is at chance. **Value: confirm the mechanical relationship and set a non-zero baseline — not a fix. Expectation lowered.**
- **Exp 5 (Imbalance-aware objective)** and **Exp 6 (Loss-term rebalancing)** — **elevated in emphasis.** Both diagnostics point to the *training objective* as where genuine signal must be created: 11.1 (regression head sits at central tendency) motivates Exp 6's variance-widening success criterion; 11.2 (classifier at prior, no ranking) motivates Exp 5's imbalance correction. These are the experiments most likely to move the collapse.
- **§7.4 reporting floor is now populated.** Every future regression claim must be reported against **mean-predictor MAE 6.3882** and **prediction variance 3.5e-4**; every future classification claim against **ROC-AUC 0.523 (CI [0.25, 0.79])** and **PR-AUC 0.311 vs 0.278 prevalence**.

[I] Net effect: the diagnostics **raise confidence** that Version 2's classification and regression gains must originate in the training recipe (Exps 3, 5, 6) and **lower** the expected payoff of treating the decision threshold (Exp 4) as a standalone remedy. No experiment is added, removed, reordered, or merged.

---

## 6. Remaining Unknowns — Why Phase 11.6 (Baseline-CV) Is Still Required

[E] The single fact that gates every quantitative conclusion above: **n = 18, with 5 positives.** This produces unknowns the frozen baseline **cannot** resolve:

1. **The true ranking ability is unmeasured.** [E] AUC 95% CI [0.250, 0.787] cannot exclude either "clearly worse than chance" or "moderately useful." A point estimate of 0.523 is uninformative at this width. [I] Whether any latent ranking signal exists can only be settled by evaluating over all 188 participants with cross-validation.
2. **The true regression effect size is unmeasured.** [E] The ~0.27 MAE deficit versus the mean predictor is itself an n=18 estimate. [I] The *qualitative* "constant predictor" finding is robust (it is an internal property of the predictions), but the *quantitative* gap is not.
3. **Generalization vs. this particular split is unknown.** [E] Both diagnostics describe one seeded 18-sample split. [I] Phase 10 §7.2 already establishes that the frozen 18-sample numbers are **not** a valid comparator for cross-validated Version 2 results.

[I] **Phase 11.6 (`Baseline-CV`, Experiment 2) is therefore mandatory, not optional.** It re-measures the *original, unmodified* trainer under participant-level cross-validation over all 188 participants, producing the valid comparator (with fold variance and CIs) that Exps 3–12 must be judged against. These diagnostics are **fast, cheap, directional signals**; `Baseline-CV` is the **authority**. The two diagnostics set the hypotheses (both heads collapsed; ranking at chance) that 11.6 will test at adequate power. Nothing in this synthesis overrides that ordering.

---

## 7. Conclusion

[E] After verification gates that reproduced the frozen metrics exactly, two independent diagnostics established:
- the **regression head is a constant/mean estimator** (spread 0.058/23, worse than the mean predictor), and
- the **classification head tracks the class prior with ranking at chance** (P(pos) var 3.2e-5, ROC-AUC 0.523, CI [0.250, 0.787], wrong rank direction).

[I] These are one failure mode — **joint prior-collapse** — expressed on both output scales. The finding **directly supports RC-1**, **directly supports RC-2** (through the diagnostics' own power limits), **partially supports RC-7** (split determinism confirmed; harmful drift untested), and **leaves RC-3…RC-6 unverified** by these diagnostics (they remain independently documented in the frozen reports).

[I] For Version 2, this sharpens — without changing — the roadmap: the training-recipe experiments (Exps 3, 5, 6) are where classification and regression signal must be created; the decision-rule experiment (Exp 4) is a mechanical baseline, not a fix. All quantitative conclusions are bounded by n = 18 and must be re-established at power by **Phase 11.6 (`Baseline-CV`)**, which the frozen baseline cannot substitute for.

**Phase 11.3 outcome: diagnostics integrated. Joint prior-collapse confirmed (RC-1, RC-2). Version 2 interpretation sharpened; experiment order unchanged; Baseline-CV confirmed as the required next authority.**

---

*Analysis and synthesis only. No experiments performed, no model run, no retraining, no predictions regenerated. No trainer, dataset, label, frozen report, or roadmap modified. Every conclusion traces to Phase 11.1 or 11.2, with observed evidence [E] distinguished from interpretation [I].*

---

## ⚠ PARTIALLY SUPERSEDED — read `PHASE_11_7_BASELINE_CV_INTERPRETATION.md` before citing this document

**Added 2026-08-06. Nothing above has been altered.**

This synthesis inherits Phase 11.2's ranking conclusion, which **Phase 11.7 superseded** at full statistical power (188 participants, 25 fold-runs, versus 11.2's single 18-participant set).

| Claim in this document | Status after Phase 11.7 |
|---|---|
| "ranking indistinguishable from chance (ROC-AUC 0.523, 95 % CI [0.250, 0.787])" | ❌ **SUPERSEDED** — ROC-AUC **0.6333**, CI **[0.5755, 0.6911]**, excluding 0.5; 25/25 fold-runs above chance; sign test p = 5.96e-8 |
| "The classifier carries no detectable ranking information" | ❌ **SUPERSEDED** — a real, weak, consistently-directed ordering exists |
| "The failure is substantially a **learning** failure, not a decision-rule defect" | ⚠️ **PARTIALLY SUPERSEDED** — it is **both**. Learning is weak (AUC 0.633, not 0.85), *and* the decision rule discards the ordering entirely |
| **"Exp 4 … its ceiling is low because the ordering is at chance. Expectation lowered."** | ❌ **SUPERSEDED** — Phase 11.7 promoted Exp 4 to the next experiment on the evidence that a real ordering existed and the argmax boundary was unreachable (max p_pos 0.4422 across 940 predictions) |

**What survives, and it is the core of this document.** The **joint prior-collapse** finding (RC-1, RC-2) — that both output heads collapsed to the label prior — is **confirmed and strengthened** by Baseline-CV. So is this document's own governing judgement, at `:110`:

> *"**Phase 11.6 (`Baseline-CV`, Experiment 2) is therefore mandatory, not optional** … These diagnostics are fast, cheap, directional signals; `Baseline-CV` is the **authority**."*

**[I] That call was correct, and it is what produced the correction now recorded here.** This document explicitly designated itself provisional and named the measurement that would adjudicate it; the supersession is that instruction being carried out, not a failure of the synthesis.

**Subsequent record.** Experiment 4 (`PHASE_12_EXP4_DECISION_RULE.md`), Experiment 3 (`PHASE_13_EXP3_CONVERGENCE.md`) and Experiment 5 (`PHASE_14_EXP5_RESULTS.md`) all returned H₀, closing the roadmap's Immediate block.
