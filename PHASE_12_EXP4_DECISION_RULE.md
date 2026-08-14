# PHASE 12 / EXPERIMENT 4 — DECISION RULE
## Official Result Report

**Phase:** 12 · **Experiment:** 4 of the Official Experiment Sequence
**Type:** Re-scoring of the frozen Baseline-CV predictions under pre-declared decision rules — no training, no model instantiated
**Pre-registration:** `PHASE_12_EXP4_DESIGN.md` (written and frozen before execution)
**Authority:** `PHASE_10_ROADMAP.md` §5 A-4, §6 Exp 4, §7 · `PHASE_11_EVALUATION_PROTOCOL.md`
**Comparator:** `Baseline-CV` (Phase 11.6), frozen
**Convention:** **[E]** = measured from the Experiment 4 artifacts · **[I]** = interpretation

**Outcome: H₀ accepted.** Threshold relocation recovered non-zero Recall and F1 but did **not** demonstrate discrimination.

---

## 1. Objective

**[R]** *"Recover non-zero recall without retraining."*

Determine how much of the ranking information Baseline-CV established (ROC-AUC 0.6333, CI [0.5755, 0.6911]) could be converted into usable classification decisions by replacing the trainer's implicit `argmax` decision rule with a pre-declared, leakage-free operating point — while changing nothing about the model.

## 2. Research question

> Given a model whose positive-class probabilities never reach 0.5, does relocating the decision threshold recover classification performance **materially better than chance at the same positive rate** — or does it merely recover the *quantity* of positive predictions without *discrimination*?

## 3. Experimental design

Exp 4 re-scored the 25 frozen Baseline-CV prediction files (940 held-out predictions) under four pre-declared decision rules. **[E]** No model was instantiated; the scripts import neither `torch`, `transformers`, nor the frozen trainer. Design R = 5 repeats × k = 5 folds = 25 fold-runs, identical to Baseline-CV.

Aggregation reproduces the Baseline-CV contract exactly: repeats averaged within fold first, then fold-level Student-t, df = k−1 = 4, t = 2.7764.

## 4. Controlled variables

| Held constant | Mechanism |
|---|---|
| Model weights, architecture | Never loaded |
| Training (epochs, lr, loss, optimizer) | No training performed |
| Predicted probabilities `p_pos`, `pred_phq` | Read verbatim from the frozen prediction files |
| Fold partition, dataset, labels, seeds | `fold_manifest.json`, SHA-verified before and after |
| Evaluation protocol and aggregation | Identical to Baseline-CV |
| **Decision function** | **THE SINGLE CHANGED FACTOR** |

## 5. Threshold-selection policies

**[E]** All four were declared in `PHASE_12_EXP4_DESIGN.md` §6 before execution. Selected thresholds across the 25 fold-runs:

| Policy | Role | τ min | τ median | τ max |
|---|---|---|---|---|
| **P1** | **PRIMARY** — leave-one-fold-out Youden-J | 0.187337 | 0.247235 | 0.411394 |
| P2 | sensitivity — prevalence-matched, LOFO | 0.275830 | 0.332252 | 0.405700 |
| P3 | sensitivity — fixed constant 45/188 | 0.239362 | 0.239362 | 0.239362 |
| P4 | descriptive sweep only — no result drawn from it | — | — | — |

**[E]** Every P1/P2 selection pool contained 150 or 151 participants with 36 positives, drawn exclusively from the other four folds of the same repeat.

## 6. Validation checks

**[E] All integrity gates passed.**

| Self-test | Max abs diff | Result |
|---|---|---|
| Aggregation reproduces `baseline_cv_summary.json` | 0.000e+00 | PASS |
| Baseline arm reproduces Baseline-CV fold-for-fold | 3.659e-19 | PASS |
| τ-invariant metrics identical across all arms | 0.000e+00 | PASS |

**[E] τ-invariants versus Baseline-CV** — these are mathematically independent of the threshold, so their being unchanged is a *proof* that nothing but the decision rule moved:

| Metric | Exp 4 | Baseline-CV | Max abs diff |
|---|---|---|---|
| ROC-AUC | 0.633331 | 0.633331 | 0.000e+00 |
| PR-AUC | 0.394064 | 0.394064 | 5.551e-17 |
| MAE | 4.812728 | 4.812728 | 8.882e-16 |
| RMSE | 6.189593 | 6.189593 | 8.882e-16 |
| pred_var | 0.005734 | 0.005734 | 7.524e-17 |

**[E]** Frozen inputs unchanged; baseline arm reproduced the stored `pred_class` exactly; independent audit `verify_exp4.py` reports **2061 checks, 0 failures**.

**[E] One verifier defect was found and corrected.** The auditor's artifact-count check asserted 32 files against an expected set of 31 names (6 named + 25 rescored), making it unsatisfiable by construction. The count is now derived from the expected set. **[E]** No experiment logic, metric, output, or conclusion was affected; the runner and aggregator SHAs are unchanged from the pre-registration record.

## 7. Results

### 7.1 Aggregate metrics by policy — [E]

Mean ± fold SD [95% CI], fold-level Student-t, df = 4.

| Metric | baseline | **P1 (primary)** | P2 | P3 |
|---|---|---|---|---|
| Precision | 0.0000 ± 0.0000 | **0.1431 ± 0.0367** [0.0976, 0.1887] | 0.0835 ± 0.0594 | 0.1608 ± 0.0290 |
| Recall | 0.0000 ± 0.0000 | **0.5422 ± 0.2561** [0.2242, 0.8602] | 0.2400 ± 0.1519 | 0.6133 ± 0.2181 |
| F1 | 0.0000 ± 0.0000 | **0.2215 ± 0.0766** [0.1264, 0.3166] | 0.1146 ± 0.0744 | 0.2498 ± 0.0612 |
| balAcc | 0.5000 ± 0.0000 | **0.5070 ± 0.0156** [0.4876, 0.5263] | 0.5117 ± 0.0161 | 0.5011 ± 0.0109 |
| MCC | 0.0000 ± 0.0000 | **0.0131 ± 0.0292** [−0.0232, 0.0494] | 0.0228 ± 0.0316 | −0.0082 ± 0.0240 |
| Accuracy | 0.7606 ± 0.0035 | 0.4894 ± 0.1564 | 0.6537 ± 0.0752 | 0.4431 ± 0.1261 |
| pos_rate | 0.0000 ± 0.0000 | 0.5316 ± 0.2786 | 0.2221 ± 0.1449 | 0.6116 ± 0.2299 |

### 7.2 The decisive observation — decision degeneracy — [E]

The repeat-averaged table above conceals the mechanism. Examining all 25 P1 fold-runs individually:

| Outcome | Count |
|---|---|
| τ ≤ the evaluated fold's minimum p_pos ⇒ **predict ALL positive** | **13 / 25** |
| τ > the evaluated fold's maximum p_pos ⇒ **predict NONE positive** | **11 / 25** |
| τ inside the evaluated fold's probability range | **1 / 25** |

**[E]** The mean width of a single fold's entire p_pos band is **0.0292**, while the selected τ ranges over **0.2241** — a spread roughly eight times wider than the band it must land inside.

**[E]** In 24 of 25 fold-runs the decision was degenerate, and balanced accuracy was exactly 0.5000 and MCC exactly 0.0000 in every one of those runs — the arithmetic consequence of a constant prediction.

**[E] The single non-degenerate fold-run** (repeat 5, fold 2; τ = 0.2332 inside the band [0.2225, 0.2500]):

| Precision | Recall | F1 | balAcc | MCC | pos_rate |
|---|---|---|---|---|---|
| 0.4545 | 0.5556 | 0.5000 | 0.6743 | 0.3268 | 0.2895 |

### 7.3 Decomposition of the headline precision — [E]

| Fold-run outcome | Count | Precision contributed |
|---|---|---|
| All-negative (`zero_division=0`) | 11 | 0.0000 |
| All-positive | 13 | = that fold's prevalence (0.2368 or 0.2432) |
| Non-degenerate | 1 | 0.4545 |
| **Mean over 25** | **25** | **0.1431** |

## 8. Statistical analysis

### 8.1 Paired per-fold comparison, P1 vs Baseline-CV — [E]

Noise band re-estimated from Exp 4's own fold spread, never inherited from Baseline-CV's degenerate MDE.

| Metric | Baseline-CV | Exp 4 | mean Δ | 95% CI | MDE | Detectable |
|---|---|---|---|---|---|---|
| Precision | 0.0000 | 0.1431 | +0.1431 | [0.0976, 0.1887] | 0.0456 | **YES** |
| Recall | 0.0000 | 0.5422 | +0.5422 | [0.2242, 0.8602] | 0.3180 | **YES** |
| F1 | 0.0000 | 0.2215 | +0.2215 | [0.1264, 0.3166] | 0.0951 | **YES** |
| balAcc | 0.5000 | 0.5070 | +0.0070 | [−0.0124, 0.0263] | 0.0194 | no |
| MCC | 0.0000 | 0.0131 | +0.0131 | [−0.0232, 0.0494] | 0.0363 | no |
| Accuracy | 0.7606 | 0.4894 | −0.2712 | [−0.4631, −0.0792] | 0.1919 | **YES** |

### 8.2 Criterion 3 — discrimination test — [E]

Per-fold precision of P1 tested against the corpus prevalence 0.2393617021, paired-fold, df = 4.

- Per-fold precision: **0.1421, 0.0909, 0.1421, 0.1946, 0.1459**
- Mean Δ vs prevalence: **−0.09623**, 95% CI **[−0.141791, −0.050669]**
- **t(4) = −5.8641, p = 0.004222**
- **Precision exceeds prevalence: NO**

### 8.3 Random-ranker control at equal positive rate — [E]

- Exp 4 F1 **0.221480** vs random-ranker F1 **0.211901**
- Mean Δ **+0.009579**, 95% CI **[−0.017016, 0.036174]**
- **Detectable: no**

## 9. Interpretation

**[I] Why H₀ was accepted despite Recall and F1 improving.** Recall rose from 0.0000 to 0.5422 and F1 from 0.0000 to 0.2215, both far beyond their noise bands. Neither is evidence of discrimination:

1. **[E]** F1 is statistically indistinguishable from a random ranker predicting at the same positive rate (Δ +0.009579, CI [−0.017016, 0.036174], not detectable). A model that discriminated would exceed that control.
2. **[E]** Balanced accuracy (Δ +0.0070 vs MDE 0.0194) and MCC (Δ +0.0131 vs MDE 0.0363) — the two metrics that cannot be inflated by simply predicting more positives — did **not** move detectably. Both CIs include zero.
3. **[E]** Precision is significantly *below* prevalence (t(4) = −5.8641, p = 0.004222).
4. **[E]** Accuracy fell 0.7606 → 0.4894. Per protocol §3.3 this is not itself a failure, but combined with 1–3 it confirms the additional positives were not preferentially the true positives.

**[I]** Recall and F1 rose because the classifier began emitting positives at all — an arithmetic consequence of lowering the threshold, not a gain in ordering. This is precisely the *quantity-without-discrimination* outcome the research question was constructed to distinguish, and Criterion 3 was added specifically because Criteria 1–2 alone could not.

**[I] The mechanism, and the limitation it exposes.** The measured cause of the null is not that the ranking vanished — ROC-AUC remained 0.6333 throughout, unchanged by construction. It is that **the operating point could not be transferred between folds.** [E] Each fold's probability band is ~0.0292 wide, while the LOFO-selected τ varies over 0.2241 across runs; consequently the transferred threshold landed outside the evaluated fold's range in **24 of 25** runs, collapsing the decision to a constant.

**[I]** This is a *calibration* failure, not a demonstrated absence of exploitable signal. Phase 11.7 §4 anticipated the underlying condition — *"each fold-run is a separately trained model with its own probability offset"* — and warned that only within-fold rank statistics are interpretable for this model. Exp 4 has now measured the operational cost of that fact.

**[I]** The single fold-run whose threshold happened to land inside the band (Precision 0.4545, Recall 0.5556, F1 0.5000, balAcc 0.6743, MCC 0.3268) is consistent with exploitable signal being present when the operating point is correctly placed. **[I] It is one observation out of 25 and supports no quantitative claim whatsoever** — it is reported because omitting it would misrepresent the evidence, not because it constitutes a result.

**[I] What Exp 4 does and does not establish.** It establishes that a *threshold transferred across separately-trained folds* does not yield discrimination on this model. It does **not** establish that no decision rule could, because 24 of 25 runs never tested a non-trivial partition. Per pre-registration §14, no alternative threshold policy was searched for after seeing these results.

## 10. Comparison against Baseline-CV

| Property | Baseline-CV | Exp 4 (P1) | Changed? |
|---|---|---|---|
| ROC-AUC / PR-AUC | 0.633331 / 0.394064 | 0.633331 / 0.394064 | **No** — invariant by construction |
| MAE / RMSE / pred_var | 4.812728 / 6.189593 / 0.005734 | identical | **No** |
| Precision / Recall / F1 | 0.0000 / 0.0000 / 0.0000 | 0.1431 / 0.5422 / 0.2215 | Yes — all detectable |
| balAcc / MCC | 0.5000 / 0.0000 | 0.5070 / 0.0131 | No — neither detectable |
| Accuracy | 0.7606 | 0.4894 | Yes (not a decision metric) |
| Decision degeneracy | 25/25 all-negative | 24/25 constant (13 all-pos, 11 all-neg) | Form changed, degeneracy persisted |

**[I]** Baseline-CV was degenerate in one direction; Exp 4 is degenerate in both. The threshold-free evidence of ranking (ROC-AUC 0.6333) is unchanged and unexploited in both.

## 11. Acceptance criteria

| # | Criterion | Source | Verdict |
|---|---|---|---|
| 1 | Recall > 0 and F1 > 0 | Roadmap §6 Exp 4 | **MET** — 0.5422 / 0.2215 |
| 2 | Regression MAE unchanged | Roadmap §6 Exp 4 | **MET** — max diff 8.882e-16 |
| 3 | Precision exceeds prevalence 0.2394 (paired-fold, CI clears the floor) | approved addition | **NOT MET** — Δ −0.09623, CI [−0.1418, −0.0507] |
| 4 | balAcc > 0.5 and MCC > 0 beyond Exp 4's own noise band | approved addition | **NOT MET** — neither detectable |

## 12. Hypothesis verdict

> **H₀ ACCEPTED.** *"Thresholding recovers positives at approximately the rate implied by the threshold, with precision ≈ prevalence — the AUC advantage is too small to survive discretisation."*

**[E]** Criteria 1 and 2 met; Criteria 3 and 4 not met. **[I]** H₁ required precision materially above the prevalence floor; the measured precision is significantly below it, and F1 is indistinguishable from a random ranker at the same positive rate.

**[I]** The pre-registered verdict stands. §9 records that the measured *mechanism* is threshold non-transferability rather than an absence of ranking signal — a distinction that constrains what may be concluded, not the verdict itself.

## 13. Implications for the roadmap

**[I] Phase 11.7's recommendation is partially superseded by its own consequence.** Phase 11.7 promoted Exp 4 on the evidence that ranking existed (ROC-AUC 0.6333, CI excluding 0.5) and the argmax boundary was unreachable. Both premises remain true and unchallenged. What Exp 4 adds is that **the ranking, though real, is not exploitable by a transferred threshold**, because the probability distribution is too narrow and too fold-dependent to carry an operating point.

**[I] Why this justifies moving to Experiment 5 (imbalance-aware objective).** The roadmap's next single-factor step is Exp 5, and Exp 4's mechanism strengthens rather than weakens the case for it:

1. **[E]** The immediate obstacle is a probability band of mean width **0.0292** that shifts between folds. **[I]** A decision rule cannot repair a distribution that narrow — no threshold is robust to fold-to-fold offsets several times larger than the band itself.
2. **[R]** Exp 5 changes the **loss weighting**, directly targeting *"unweighted CE on an imbalanced set, no class weighting"* — the documented condition that produces a compressed, prior-tracking probability distribution.
3. **[I]** Widening and stabilising the probability distribution is a **precondition** for any decision-rule intervention to succeed. Exp 4 established that the cheap intervention is exhausted; the remaining levers are in the training objective, exactly where the roadmap places Exps 5 and 6.
4. **[E]** Exp 4 satisfies the roadmap's stated prerequisite for Exp 5, whose comparator is *"Baseline-CV and Exp 4"* — both now exist as measured artifacts.
5. **[R]** Roadmap §6 Exp 5 success criteria — *"Recall / F1 / PR-AUC improve over both"* — are now anchored to real numbers rather than to a degenerate zero baseline.

**[I] One observation for the project owner, not a roadmap change.** Exp 4's failure mode is calibration, and no roadmap experiment currently isolates calibration as a single factor. Whether that warrants an experiment is a governance decision outside this report's scope; it is recorded here because the evidence produced it, not because it is recommended.

**[I]** Exps 3 and 6 are unaffected. Exp 3 (convergence) still targets the regression arm, where Baseline-CV recorded a powered null.

## 14. Conclusion

**[E]** Experiment 4 changed exactly one factor — the decision function — and proved it: ROC-AUC, PR-AUC, MAE, RMSE and prediction variance are identical to Baseline-CV to within 8.882e-16, and the independent auditor returned 2061 checks with 0 failures.

**[E]** Relocating the threshold produced the project's first non-zero Recall (0.5422) and F1 (0.2215), both detectable against Baseline-CV. **[E]** It did not produce discrimination: precision (0.1431) is significantly below prevalence (t(4) = −5.8641, p = 0.004222), F1 is indistinguishable from a random ranker at equal positive rate (Δ +0.009579, CI [−0.017016, 0.036174]), and neither balanced accuracy nor MCC moved detectably.

**[E]** The measured cause is decision degeneracy: in 24 of 25 fold-runs the leave-one-fold-out threshold fell outside the evaluated fold's probability band, collapsing the decision to a constant.

**[I]** The classification failure identified in Phase 11.7 as *composite* — weak learning plus an unreachable decision boundary — is now better resolved. The decision-rule component cannot be fixed in isolation while the probability distribution remains ~0.03 wide and fold-dependent. The remaining lever is the training objective.

**Phase 12 / Experiment 4 outcome: H₀ accepted. Recall and F1 recovered; discrimination not demonstrated. Criteria 1–2 met, 3–4 not met. Proceed to Experiment 5.**

---

*Result report only. No experiment re-run, no metric recomputed, no roadmap modified, no Experiment 4 output altered. Every [E] statement is traceable to `trainer_outputs/exp4_decision_rule/` (`exp4_summary.json`, `exp4_summary.md`, `exp4_metrics.csv`, `thresholds.csv`, `invariants_check.json`) or to the frozen Baseline-CV artifacts; every [I] statement is labelled as interpretation. Pre-registration: `PHASE_12_EXP4_DESIGN.md`.*
