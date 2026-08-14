# PHASE 11.2 — DIAGNOSTIC B: THRESHOLD-FREE RANKING ANALYSIS
## Experiment 0b of the Official Phase 10 Roadmap

**Phase:** 11.2 — Second scientific diagnostic of Version 2
**Type:** Analysis only — no training, no code change, no prediction regeneration
**Roadmap trace:** `PHASE_10_ROADMAP.md` §5 (A-0), §6 (Experiment 0b), §7.4
**Comparator:** `PHASE_9.5_COMPLETE_DATASET_REPORT.md` §7.1 (frozen `eval_preds.csv`, all class 0, P(pos) 0.199–0.214)

Throughout: **[E]** = observed evidence (measured from frozen artifacts) · **[I]** = interpretation.

---

## 1. Objective

The frozen classifier predicts **every** validation participant as negative (P/R/F1 = 0.0). This diagnostic asks a **threshold-free** question: *ignoring the 0.5 decision rule, do depressed participants nonetheless receive higher positive-class probabilities than non-depressed participants?*

The goal is **not** to improve, retrain, or re-threshold the classifier. It is to determine whether **useful probability ordering already exists** beneath the collapsed decision — which determines whether the classification failure is a *decision-rule* problem (cheap to fix) or a *learning* problem (requires recipe changes).

---

## 2. Methodology

The analysis reuses the **verified** validation reconstruction established in Phase 11.1.

**Split replay (read-only).** The trainer's split is deterministic (`trainer_mentalbert_daic.py` L560–566): `idx = shuffle(arange(188), seed=42)`, `val_idx = idx[:18]`, evaluation with `shuffle=False`. Replaying these operations recovers the 18 validation participants, their true PHQ, and the binarized labels (`PHQ > 10`), aligned one-to-one with the rows of `eval_preds.csv`.

**Verification gate.** Before any ranking claim, the reconstructed labels were scored against the frozen `pred_class` column:

| Metric | Reconstructed | Frozen (P9.5 §6.1) | Match |
|---|---|---|---|
| Accuracy | 0.7222222222 | 0.7222222222222222 | ✅ |
| Precision / Recall / F1 | 0.0 / 0.0 / 0.0 | 0.0 / 0.0 / 0.0 | ✅ |
| All predictions negative | True | True | ✅ |

[E] The reconstruction reproduces the frozen classification behaviour exactly; the 18 labels and their probability alignment are therefore confirmed correct.

**Probability representation used.** `eval_preds.csv` stores `pred_class_probs` as a two-element softmax pair per row. [E] These pairs sum to 1.0 for every row, i.e. `[P(negative), P(positive)]`. **P(positive) = element 1** is the correct threshold-free score and is used throughout. No raw logits or alternative confidence artifacts were saved for the Track-C classifier, so this is the only probability representation available — and it is the appropriate one (softmax posterior for the positive class). *(The Track-A linear-probe `metrics_*.json` files are a separate federated unit with no DAIC labels and are not classification outputs for this model; they are not used.)*

**Ranking statistics** are threshold-independent by construction: ROC-AUC, Average Precision (PR-AUC), the Mann–Whitney U rank-sum (does P(pos) rank higher for depressed participants?), and Spearman/Kendall rank correlation against the *continuous* PHQ score. Because n = 18, each is accompanied by an uncertainty statement (bootstrap CI and permutation test on the AUC).

**Threshold sensitivity** sweeps the decision threshold across the observed probability range to characterise how predictions *would* change — **without** selecting or optimizing any threshold.

---

## 3. Frozen Artifacts Used

All inputs are frozen Phase 9.5 artifacts; nothing was modified or regenerated.

| Artifact | Role |
|---|---|
| `trainer_outputs/eval_preds.csv` (18 rows) | `pred_class`, `pred_class_probs` — the classifier outputs under analysis |
| `daic_records.parquet` (188 records) | True PHQ → binarized labels; split replay |
| `trainer_mentalbert_daic.py` (SHA `65b1902e…a230b`) | Read only, to confirm split/eval/binarize logic |
| `PHASE_9.5_…REPORT.md` §6.1, §7.1 | Frozen comparator values |

**Reconstructed validation set (verified, n = 18):** 5 positives / 13 negatives (prevalence 0.278). Binarized labels (eval order): `0,0,0,0,0,0,1,1,0,0,0,1,0,1,0,1,0,0`.

---

## 4. Probability Analysis

[E] Descriptive statistics of P(positive) over the 18 validation participants:

| Statistic | Value |
|---|---|
| Minimum | 0.198528 |
| Maximum | 0.213817 |
| **Range** | **0.015290** |
| Mean | 0.204708 |
| Median | 0.201219 |
| Variance | 0.00003152 |
| Std. deviation | 0.005614 |

Descriptive histogram (bin counts across [0.19, 0.195, 0.20, 0.205, 0.21, 0.215]): **0, 5, 6, 1, 6**.

**Two observations:**

- [E] **The probabilities collapse into a ~0.015-wide interval** (0.199–0.214). The entire spread of positive-class confidence across all participants is 1.5 percentage points. Variance is 3.2e-5.
- [E] **The mean P(pos) = 0.205 sits at the class prior.** The full-corpus positive rate is 0.239 and the validation positive rate is 0.278; the model assigns essentially the base rate as the positive probability to everyone. [I] This is the classifier-side signature of the same **prior-collapse** identified in Phase 11.1 for the regression head — both heads emit a near-constant that tracks the label prior rather than the participant.

[E] **Determinacy of the all-negative output:** every P(pos) is ≤ 0.214, far below the 0.5 argmax boundary, so `argmax` selects the negative class for all 18 inputs by construction.

---

## 5. Ranking Metrics

[E] Threshold-free ranking, with uncertainty:

| Metric | Value | Reference / uncertainty |
|---|---|---|
| **ROC-AUC** | **0.523077** | Chance = 0.500; bootstrap 95% CI **[0.250, 0.787]**; permutation p(AUC ≥ obs) = **0.446** |
| P(true AUC ≤ 0.5) | 0.464 (bootstrap) | ≈ coin-flip whether ordering even beats chance |
| **Average Precision (PR-AUC)** | **0.310505** | Baseline (prevalence) = 0.277778 → +0.033 |
| Mean P(pos) \| depressed | **0.202989** | — |
| Mean P(pos) \| non-depressed | **0.205369** | depressed score **lower** (wrong direction) |
| Mann–Whitney U (depressed > non) | 34.0 | one-sided p = **0.462** (not significant) |
| Spearman ρ (P(pos) vs PHQ) | 0.3154 | p = 0.202 (not significant) |
| Kendall τ (P(pos) vs PHQ) | 0.2050 | p = 0.249 (not significant) |

[E] **The point estimates are mutually inconsistent and none is significant.** ROC-AUC is trivially above chance (0.523) but its 95% CI spans from clearly-below-chance to strongly-above (0.25–0.79); the permutation test cannot distinguish it from label-shuffled noise (p = 0.446). The mean positive-class probability is actually *lower* for depressed participants (0.203 vs 0.205), i.e. the binary rank direction is slightly wrong, while the continuous rank correlations (ρ = 0.32, τ = 0.21) lean weakly positive — but neither approaches significance.

[I] Taken together, **the frozen artifacts contain no statistically detectable ranking signal.** The scatter of these estimates around chance is exactly what near-random ordering produces at n = 18.

---

## 6. Threshold Analysis

[E] Prediction counts and metrics as the decision threshold sweeps down through the probability range (no optimization — sensitivity only):

| Threshold | # pos | # neg | Recall | Precision | F1 |
|---|---|---|---|---|---|
| 0.500 (frozen) | 0 | 18 | 0.000 | 0.000 | 0.000 |
| 0.300 | 0 | 18 | 0.000 | 0.000 | 0.000 |
| 0.220 | 0 | 18 | 0.000 | 0.000 | 0.000 |
| 0.213 | 1 | 17 | 0.000 | 0.000 | 0.000 |
| 0.210 | 6 | 12 | 0.200 | 0.167 | 0.182 |
| 0.205 | 7 | 11 | 0.200 | 0.143 | 0.167 |
| **0.200** | 13 | 5 | **1.000** | 0.385 | 0.556 |
| 0.199 | 17 | 1 | 1.000 | 0.294 | 0.455 |
| ≤ 0.195 | 18 | 0 | 1.000 | 0.278 | 0.435 |

**Two observations:**

- [E] **The all-negative output is mechanically a threshold artifact.** Nothing is positive until the threshold drops below ~0.214, and the transition from "all negative" to "all positive" occurs entirely within the narrow 0.199–0.214 band. The frozen 0.5 threshold sits far above the whole collapsed cluster, so it *guarantees* all-negative predictions regardless of ordering.
- [E] **Lowering the threshold recovers quantity, not quality.** At threshold 0.200 recall reaches 1.0, but precision is 0.385 — barely above the 0.278 prevalence a coin flip at that positivity rate would deliver. [I] Because the probabilities carry no reliable ordering (§5), sweeping the threshold merely relabels a near-randomly-ordered cluster; it produces positive predictions but not *discriminating* ones.

---

## 7. Scientific Interpretation

**A. Does the classifier contain meaningful ranking information?**
[E] **No detectable ranking information.** ROC-AUC = 0.523 with 95% CI [0.250, 0.787], permutation p = 0.446, a slightly-wrong binary rank direction, and non-significant rank correlations. [I] On the frozen evidence there is no usable probability ordering; depressed and non-depressed participants receive statistically indistinguishable scores. [I] At n = 18 (5 positives) this is **directionally informative, not conclusive** — the CI cannot exclude a modest true signal — but there is **no positive evidence** for one, and the best estimate is "at chance."

**B. Is the all-negative behaviour caused primarily by the threshold or by poor probability separation?**
[E/I] **Both, at different levels — and the distinction is the key finding.** *Mechanically*, the all-negative output is caused by the **threshold**: every probability (~0.20) lies far below 0.5, so argmax is always negative, and a lower threshold would flip predictions (§6). *Substantively*, the probabilities exhibit **near-total absence of separation** (range 0.015, AUC ≈ chance), so the threshold is not hiding a good ranking — there is no good ranking to hide. [I] The threshold explains *why the predictions are all one class*; the collapsed, prior-tracking probabilities explain *why re-thresholding cannot rescue the classifier*.

**C. Would later calibration or threshold experiments be scientifically justified?**
[I] **A threshold experiment (roadmap Exp 4) remains justified — but with recalibrated expectations.** It is the cheapest possible intervention and will, as the roadmap already predicts (§6 #4), mechanically produce the project's first **non-zero recall**. Its scientific value is now precise: to *confirm the mechanical threshold relationship* and *establish a non-zero classification baseline*, **not** to deliver a discriminating classifier. [E] Calibration cannot manufacture separation that is absent (AUC ≈ 0.5); [I] therefore neither thresholding nor calibration is the *fix* — genuine classification signal must be created in **training**.

**D. Which future Version 2 experiments become more or less important?**
[I] This is the roadmap's **"AUC at chance" branch of Experiment 0b**, whose stated consequence is: *"Exps 4–5 become mandatory"* (§6 #0b). Concretely:
- **Exp 4 (decision rule)** — still run, reframed as a mechanical baseline that yields non-zero recall, *not* a discrimination fix. Its ceiling is now known to be low.
- **Exp 5 (imbalance-aware objective)** and **Exp 6 (loss-term rebalancing)** — **rise in importance.** The evidence says the classifier never learned to rank; the corrective must act on the *training objective*, not the output threshold.
- **Exp 3 (convergence)** — remains the least-invasive first probe of whether the collapse is a budget artifact.
- The **decision-rule-defect hypothesis is not supported**; the failure is substantially a **learning** failure. This *lowers* the expected payoff of Exp 4-as-a-fix while *raising* that of Exps 5–6.

No experiments are added, removed, reordered, or merged; only priority emphasis shifts, consistent with the roadmap's own 0b-branch logic.

---

## 8. Implications for the Version 2 Roadmap

*Explanation only — the roadmap is not modified.*

- **Joint prior-collapse is now confirmed on both heads.** Phase 11.1 showed the regression head emits a near-constant at the training central tendency; Phase 11.2 shows the classifier emits P(pos) ≈ the class prior. [I] This coherent two-sided collapse strengthens the roadmap's RC-1 framing and the rationale for the **Exp 3 → 6 recipe block** attacking one factor at a time.
- **§7.4's threshold-free reporting requirement is now populated with evidence.** Every future classification claim must be reported against these frozen-baseline ranking figures: **ROC-AUC 0.523 (CI [0.25, 0.79]), PR-AUC 0.311 vs 0.278 prevalence.** A Version 2 classifier is only credible when its AUC clears this floor by more than the noise band.
- **Supersession is explicit.** [E] The bootstrap CI [0.250, 0.787] is precisely why these diagnostics are *fast signals, not verdicts*. The authoritative ranking measurement is deferred to **Experiment 2 / Phase 11.6 (`Baseline-CV`)**, which recomputes AUC/PR-AUC under participant-level cross-validation over all 188 participants (§7.2). Nothing here overrides that; it sets the hypothesis it will test.

---

## 9. Conclusion

[E] The reconstruction reproduced the frozen classification behaviour exactly (acc 0.7222; P/R/F1 = 0/0/0; all-negative), validating the analysis. On that verified basis:

- [E] The classifier's positive-class probabilities **collapse into a 0.015-wide band around the class prior** (mean 0.205, variance 3.2e-5).
- [E] **Ranking is indistinguishable from chance** — ROC-AUC 0.523 (95% CI [0.250, 0.787], permutation p = 0.446), a slightly-wrong binary rank direction, and non-significant rank correlations.
- [E] The **all-negative output is a mechanical threshold artifact** (all probabilities ≪ 0.5); [I] but re-thresholding recovers only *quantity* of positives (recall 1.0 at threshold 0.20, precision 0.385 ≈ prevalence), **not discrimination**, because the underlying ordering carries no reliable signal.

[I] **The classifier contains no detectable useful ranking information; the classification failure is substantially a learning failure, not merely a decision-rule defect.** A threshold/calibration experiment (Exp 4) is still worth running as the cheapest route to a first non-zero recall, but the genuine fix must come from the training-objective experiments (Exps 5–6, with Exp 3 as the convergence probe). At n = 18 this is directionally strong but not conclusive, and the authoritative measurement is the cross-validated `Baseline-CV` in Phase 11.6.

**Experiment 0b outcome: AUC at chance — decision-rule-defect hypothesis not supported; Exps 4–5 (and 6) become mandatory. Branch B confirmed.**

---

*No code executed against the model. No retraining. No predictions regenerated. No trainer, dataset, label, or frozen report modified. Every conclusion is supported by a measured quantity, with observed evidence [E] distinguished from interpretation [I].*

---

## ⚠ SUPERSEDED — read `PHASE_11_7_BASELINE_CV_INTERPRETATION.md` before citing this document

**Added 2026-08-06. Nothing above has been altered.**

**This document's central conclusion — "no detectable ranking information", "the best estimate is *at chance*", "Experiment 0b outcome: AUC at chance" — was superseded by Phase 11.7.**

Phase 11.2 measured a single 18-participant evaluation set (5 positives). Phase 11.6 re-measured the *same frozen model* over all **188 participants** under participant-level 5-fold cross-validation with 5 repeats, and Phase 11.7 interpreted the result:

| Phase 11.2 (n = 18) | Phase 11.7 (n = 188, 25 fold-runs) |
|---|---|
| ROC-AUC **0.523**, 95 % CI **[0.250, 0.787]** | ROC-AUC **0.6333**, 95 % CI **[0.5755, 0.6911]** — **excludes 0.5** |
| permutation p = 0.446 — indistinguishable from shuffled labels | sign test p = **5.96e-8**; **25/25** fold-runs above chance |
| rank direction "slightly wrong" (depressed scored lower) | rank direction **correct**: +0.002726 pooled, consistent within every fold |

**[I] Phase 11.2's analysis was correct for the evidence available to it.** Its own §132 states the finding is *"directionally informative, not conclusive"* and that the confidence interval *"cannot exclude a modest true signal"*. That caveat proved decisive: the true value lies outside the lower half of the interval, and the width of the CI — not an error of method — is what made the point estimate uninformative.

**What survives.** The mechanical finding that the all-negative output is a **threshold artifact** (all probabilities ≪ 0.5) is confirmed and strengthened: Phase 11.7 measured a maximum p_pos of **0.4422** across 940 predictions, making the argmax boundary structurally unreachable.

**What does not.** The inference that *no usable ordering exists* is withdrawn, and with it Phase 11.3's consequent downgrade of Experiment 4. A real ordering exists; the decision rule discards all of it.

**Roadmap consequence.** The stated *"Exps 4–5 become mandatory"* branch outcome remains valid in effect — Experiments 4 and 5 were both executed — but it was reached by the **other** branch. See `PHASE_11_7_BASELINE_CV_INTERPRETATION.md` §6 for the itemised supersession table and `PHASE_12_EXP4_DECISION_RULE.md` for what Experiment 4 subsequently found.
