# PHASE 14 / EXPERIMENT 5 — IMBALANCE-AWARE OBJECTIVE
## Official Result Report

**Phase:** 14 · **Experiment:** 5 of the Official Experiment Sequence
**Type:** Retraining of the frozen architecture under class-weighted cross-entropy over the frozen Baseline-CV folds
**Pre-registration:** `PHASE_14_EXP5_DESIGN.md` (written and frozen before execution)
**Authority:** `PHASE_10_ROADMAP.md` §5 A-3, §6 Exp 5, §7 · `PHASE_11_EVALUATION_PROTOCOL.md`
**Comparator:** `Baseline-CV` (Phase 11.6), frozen · secondary context: `Exp 4` (Phase 12) and `Exp 3` (Phase 13), frozen
**Convention:** **[E]** = measured from the Experiment 5 artifacts · **[I]** = interpretation · **[R]** = roadmap reference

**Outcome: H₀ accepted.** Class weighting **relocated the decision boundary without producing discrimination.** Positive predictions appeared for the first time under argmax, but they are of random-ranker quality or worse.

---

## 1. Introduction

**[E]** Experiment 5 is the third and final experiment in the Immediate block of the roadmap's RC-1 investigation, and the last of the three cheap interventions the roadmap names. Experiments 4 and 3 both returned H₀, excluding the decision rule and the epoch budget respectively. **[R]** RC-1 names three joint conditions for prior-collapse — *3 epochs · unweighted CE · class imbalance* — and Experiment 5 tests the only untested pair among them: the unweighted objective and the imbalance it is applied to.

**[I]** This report closes Phase 14. It reports measurement and interpretation only; it does not redesign the roadmap, does not initiate further experiments, and every number below is traceable to `trainer_outputs/exp5_imbalance_objective/`.

**[E]** The result is not a null in the weak sense of "nothing happened." Something measurable and highly systematic happened — it simply was not discrimination. Section 9 isolates the mechanism precisely.

## 2. Objective

**[R]** *"Establish whether the unweighted cross-entropy objective on an imbalanced corpus is the operative cause of prior-collapse."*

Determine whether re-weighting the classification loss in inverse proportion to class frequency causes the classification head to escape the label prior and produce genuine positive-class discrimination, with the regression head and every other factor held fixed.

## 3. Comparator configuration

**[E]** The accepted configuration is **unchanged from Baseline-CV**, because Experiments 4 and 3 both returned H₀ and therefore contributed no accepted change:

| Factor | Value in Exp 5 | Provenance |
|---|---|---|
| Epoch budget | **3 (reverted)** | Exp 3 returned H₀ — 20 epochs not adopted |
| Decision rule | **argmax, τ = 0.5 (reverted)** | Exp 4 returned H₀ — threshold policy not adopted |
| Loss weighting | **the single changed factor** | this experiment |
| Folds, seeds, tokenizer, LR, batch size, clipping, MSE coefficient | frozen | Baseline-CV |

**[E]** The one-factor rule therefore holds strictly against Baseline-CV: exactly one factor differs. **[E]** The frozen trainer `trainer_mentalbert_daic.py` was **not modified** (SHA `65b1902e…a230b` verified at startup and after the run).

**[I] One structural caution, carried forward verbatim from Phase 13 §9.** Unlike Experiment 3 — where `epochs` is a *parameter* of the frozen `fine_tune_supervised` and no training code had to be rewritten — Experiment 5's factor lives *inside* that function. The training loop had to be reimplemented to inject a weighted `CrossEntropyLoss`. The W0 equivalence gate therefore does real load-bearing work that Exp 3's E0 gate did not have to do, and Phase 13 explicitly flagged it as the load-bearing element of this design. Section 5 reports how it discharged that duty.

## 4. Experimental setup

**[E] Pre-declared arms**, all four using the identical training loop and differing only in the `weight` tensor passed to `nn.CrossEntropyLoss`:

| Arm | Scheme | Formula | Role |
|---|---|---|---|
| **W0** | uniform | `weight = None` | **equivalence control** |
| **W1** | balanced | `w_c = n / (2·n_c)` | **primary** |
| W2 | sqrt-balanced | `w_c = √(n / (2·n_c))` | sensitivity |
| W3 | fixed inverse-prevalence | `w_neg = 1`, `w_pos = 1/0.2394` | sensitivity |

**[E]** Weights are computed from **training-fold labels only**; the held-out fold never influences its own weights. The independent auditor re-derived all 100 weight pairs from the fold manifest and confirmed this (§8).

**[E] Design:** R = 5 repeats × k = 5 folds = 25 fold-runs per arm, **100 fold-runs total**. Corpus prevalence 45/188 = **0.2393617021276596**. Binarisation PHQ > 10.0.

**[E] Aggregation:** repeats averaged within fold first, then fold-level Student-t, df = k−1 = 4, **t = 2.7764451051977987** — identical to Baseline-CV, Exp 4 and Exp 3, so all four remain mutually comparable.

**[E] Execution summary.** Colab GPU, frozen trainer SHA verified at startup, optimizer confirmed as `transformers.optimization.AdamW` (a hard abort would have fired had it resolved to `torch.optim.AdamW`, which differs in eps, weight decay and bias-correction placement).

| Arm | Fold-runs | Total compute | Mean per fold-run | Final train loss (mean) |
|---|---|---|---|---|
| W0 | 25 | 416.3 s | 16.7 s | 21.1187 |
| **W1** | **25** | **403.1 s** | **16.1 s** | **21.2357** |
| W2 | 25 | 401.5 s | 16.1 s | 21.1947 |
| W3 | 25 | 401.6 s | 16.1 s | 21.2463 |
| **Total** | **100** | **1622.5 s (27.0 min)** | | |

**[E]** W0's mean final training loss, **21.1187**, is identical to Experiment 3's E0 arm final loss of **21.1187** — an independent cross-experiment corroboration that the two harnesses reproduce the same 3-epoch optimisation.

**[I]** Final training loss is **not comparable across arms**: each arm minimises a different objective, so W1's higher value (21.2357) does not mean it optimised worse. It is reported for provenance only, never as evidence.

**[E] Artifacts:** 127 files — 100 per-fold prediction files, 20 per-repeat metrics files, 7 consolidated artifacts. Consolidated row counts are exact: `exp5_metrics.csv` 100, `class_weights.csv` 100, `degeneracy_report.csv` 100, each with W0 = W1 = W2 = W3 = 25 and every `(arm, repeat, fold)` cell present exactly once.

**[E] Audit:** `verify_exp5.py` — **3443 checks, 0 failures, EXP 5 AUDIT PASS: True, exit code 0.** The auditor imports neither `run_exp5.py` nor `aggregate_exp5.py`, and neither torch nor transformers; every metric, weight, aggregate, confidence interval and criterion verdict was re-derived from the raw prediction files. **[E]** Aggregation self-test reproduces `baseline_cv_summary.json` at max abs diff **0.000e+00**.

## 5. Acceptance criteria

**[E]** Seven criteria, pre-registered before execution. C0 is a hard gate; C5 and C6 were carried forward from Exp 4's H₀ specifically to prevent a repeat of the failure mode Exp 4 exposed.

| # | Criterion | Source |
|---|---|---|
| **0** | W0 reproduces Baseline-CV (all primary metrics inside its 95 % CI) | design gate — **hard abort** |
| 1 | PR-AUC detectably above Baseline-CV 0.3941 | Roadmap §6 Exp 5 |
| 2 | Recall and F1 improve over Baseline-CV (0.0000 / 0.0000) | Roadmap §6 Exp 5 |
| 3 | Recall and F1 improve over Exp 4 (0.5422 / 0.2215) *[threshold-conditional]* | Roadmap §6 Exp 5 |
| 4 | Regression MAE not materially degraded (MDE 0.4066) | Roadmap §6 Exp 5 |
| **5** | **Discrimination, not quantity:** precision > prevalence 0.2394 **AND** F1 > random-ranker control | carried forward from Exp 4 |
| **6** | Decisions non-degenerate (below Exp 4's 24/25) | carried forward from Exp 4 |

**[E]** Per the approved design, the distribution diagnostics of §9 are **reported in full but are not acceptance criteria** — they are secondary observational outcomes only.

## 6. W0 equivalence gate

**[E] PASSED, and at the strongest possible level.** The gate is recorded as `authoritative: true` with `failures: []` across all 12 primary metrics.

**[E]** The pre-registration anticipated only *statistical* equivalence — W0's aggregate falling inside Baseline-CV's 95 % CI. What was measured is far stronger:

| Test | Result |
|---|---|
| W0 aggregate inside Baseline-CV 95 % CI, 12/12 metrics | **PASS** |
| max \|W0 mean − Baseline-CV mean\| over 12 metrics | **0.000e+00** |
| max \|Δ `pred_phq`\| over all 25 fold-runs (940 predictions) | **0.000000e+00** |
| max \|Δ `p_pos`\| over all 25 fold-runs (940 predictions) | **0.000000e+00** |

**[E] W0's predictions are bit-identical to Baseline-CV on every one of the 940 held-out predictions.**

**[I]** This fully discharges the structural caution Phase 13 raised. The reimplemented training loop does not merely approximate the frozen `fine_tune_supervised` — with `weight=None` it reproduces it exactly, prediction for prediction. Every W1/W2/W3 difference from Baseline-CV is therefore attributable **solely to the loss weighting**, with no residual implementation risk. This is the same guarantee Exp 3's E0 gate delivered, obtained here despite the loop having been rewritten.

## 7. Results by criterion (C0–C6)

| # | Criterion | Measured | Verdict |
|---|---|---|---|
| **0** | W0 reproduces Baseline-CV | max diff **0**, 12/12 inside CI | **MET** |
| 1 | PR-AUC detectably above 0.3941 | 0.3902, Δ **−0.0039**, MDE 0.0273 | **NOT MET** |
| 2 | Recall and F1 above Baseline-CV | 0.0756 > 0 and 0.0295 > 0 | **MET** |
| 3 | Recall and F1 above Exp 4 | 0.0756 < 0.5422; 0.0295 < 0.2215 | **NOT MET** |
| 4 | MAE not materially degraded | Δ **−0.0013**, anchor 0.4066 | **MET** |
| **5** | **Discrimination** | precision **0.0184 < 0.2394**; F1 **0.0295 < 0.0309** | **NOT MET** |
| **6** | Non-degenerate decisions | W1 **24/25** degenerate, not below Exp 4's 24 | **NOT MET** |

**[I] C2 MET alongside C5 NOT MET is the crux of this experiment, and it is not a contradiction.** C2 is a bare inequality against exact zeros — any positive prediction at all satisfies it. C5 asks whether those predictions carry information. They do not. C2 measures *quantity*; C5 measures *quality*. The design deliberately included both, precisely because Exp 4 had already demonstrated that the first can be satisfied while the second fails.

**[E]** C4 is met in the strict sense — MAE moved by −0.0013 against an anchor of 0.4066 — but **[I]** this reflects that the regression head was barely perturbed at all, not that weighting protected it (§9).

## 8. Statistical analysis

### 8.1 Aggregate metrics by arm — [E]

Mean ± fold SD, fold-level Student-t, df = 4.

| Metric | W0 (uniform) | **W1 (balanced)** | W2 (sqrt) | W3 (1/prev) |
|---|---|---|---|---|
| MAE | 4.8127 ± 0.3275 | **4.8114 ± 0.3253** | 4.8117 ± 0.3262 | 4.8108 ± 0.3255 |
| RMSE | 6.1896 ± 0.3521 | 6.1832 ± 0.3464 | 6.1856 ± 0.3476 | 6.1819 ± 0.3469 |
| pred_var | 0.0057 ± 0.0079 | 0.0059 ± 0.0083 | 0.0058 ± 0.0081 | 0.0059 ± 0.0084 |
| MAE_mean_pred | 4.9135 ± 0.4041 | 4.9135 ± 0.4041 | 4.9135 ± 0.4041 | 4.9135 ± 0.4041 |
| ROC-AUC | 0.6333 ± 0.0466 | 0.6267 ± 0.0420 | 0.6272 ± 0.0418 | 0.6262 ± 0.0371 |
| PR-AUC | 0.3941 ± 0.0765 | 0.3902 ± 0.0599 | 0.3935 ± 0.0720 | 0.3881 ± 0.0601 |
| balAcc | 0.5000 ± 0.0000 | 0.4978 ± 0.0050 | 0.5000 ± 0.0000 | 0.5056 ± 0.0107 |
| MCC | 0.0000 ± 0.0000 | **−0.0118 ± 0.0263** | 0.0000 ± 0.0000 | 0.0124 ± 0.0191 |
| Precision | 0.0000 ± 0.0000 | 0.0184 ± 0.0252 | 0.0000 ± 0.0000 | 0.0530 ± 0.0549 |
| Recall | 0.0000 ± 0.0000 | 0.0756 ± 0.1038 | 0.0000 ± 0.0000 | 0.1822 ± 0.1808 |
| F1 | 0.0000 ± 0.0000 | 0.0295 ± 0.0405 | 0.0000 ± 0.0000 | 0.0797 ± 0.0805 |
| Accuracy | 0.7606 ± 0.0035 | 0.7179 ± 0.0592 | 0.7606 ± 0.0035 | 0.6743 ± 0.0869 |

**[E] W2 is metric-for-metric identical to W0 on every classification metric** — balAcc 0.5000, MCC 0.0000, Precision 0.0000, Recall 0.0000, F1 0.0000, Accuracy 0.7606. Section 9 explains why, and the explanation is exact.

**[E] W1's MCC is negative** (−0.0118 ± 0.0263). **[I]** A negative Matthews correlation means the positive predictions are anti-correlated with the true labels — worse than uninformative, though the interval [−0.0444, 0.0209] includes zero.

### 8.2 Paired per-fold comparison: W1 vs Baseline-CV — [E]

Noise band re-estimated from Experiment 5's own fold spread, never inherited from Baseline-CV's degenerate MDEs.

| Metric | Baseline-CV | W1 | mean Δ | 95 % CI | MDE | Detectable |
|---|---|---|---|---|---|---|
| MAE | 4.8127 | 4.8114 | −0.0013 | [−0.0048, 0.0022] | 0.0035 | no |
| RMSE | 6.1896 | 6.1832 | −0.0064 | [−0.0138, 0.0010] | 0.0074 | no |
| pred_var | 0.0057 | 0.0059 | +0.0001 | [−0.0004, 0.0007] | 0.0006 | no |
| ROC-AUC | 0.6333 | 0.6267 | −0.0067 | [−0.0197, 0.0064] | 0.0131 | no |
| PR-AUC | 0.3941 | 0.3902 | −0.0039 | [−0.0312, 0.0234] | 0.0273 | no |
| balAcc | 0.5000 | 0.4978 | −0.0022 | [−0.0084, 0.0039] | 0.0062 | no |
| MCC | 0.0000 | −0.0118 | −0.0118 | [−0.0444, 0.0209] | 0.0326 | no |
| Precision | 0.0000 | 0.0184 | +0.0184 | [−0.0129, 0.0496] | 0.0312 | no |
| Recall | 0.0000 | 0.0756 | +0.0756 | [−0.0533, 0.2044] | 0.1288 | no |
| F1 | 0.0000 | 0.0295 | +0.0295 | [−0.0207, 0.0798] | 0.0503 | no |
| Accuracy | 0.7606 | 0.7179 | −0.0427 | [−0.1152, 0.0299] | 0.0726 | no |

**[E] Not one of the eleven paired comparisons is detectable.** Every confidence interval includes zero.

**[E] Per-fold Recall deltas:** +0.2000, 0.0000, 0.0000, +0.1778, 0.0000. **[E] Per-fold Accuracy deltas:** −0.1053, 0.0000, 0.0000, −0.1081, 0.0000.

**[I]** The two vectors are supported on the *same two folds*. In folds 1 and 4 — the only folds where W1 emitted any positive at all — Recall rose and Accuracy fell by comparable magnitudes. In the other three folds W1 is exactly Baseline-CV. **[I]** This is the signature of a boundary shift trading false positives for true positives at roughly the prevailing base rate, not of improved separation.

### 8.3 Ranking ability is preserved — [E]

| Arm | ROC-AUC | 95 % CI | Excludes 0.5 |
|---|---|---|---|
| W0 | 0.6333 | [0.5755, 0.6911] | **yes** |
| **W1** | **0.6267** | **[0.5745, 0.6788]** | **yes** |
| W2 | 0.6272 | [0.5753, 0.6791] | **yes** |
| W3 | 0.6262 | [0.5802, 0.6723] | **yes** |

**[E]** All four arms retain above-chance ranking, and no arm's ROC-AUC moved detectably from Baseline-CV.

**[I]** This is a crucial qualification and must not be lost in the null result. The underlying **ranking signal established in Phase 11.7 survives class weighting entirely intact.** The model orders participants better than chance both before and after the intervention. What weighting failed to fix is the conversion of that ordering into decisions — and, as §9 shows, it did not even attempt to fix it in the way one might assume.

## 9. Class-weight analysis

**[E]** All 100 weight pairs were independently re-derived by the auditor from the fold manifest, using training labels only, and matched the recorded values exactly.

| Arm | w_neg | w_pos | ratio w_pos/w_neg | distinct pairs |
|---|---|---|---|---|
| W0 | 1.000000 | 1.000000 | 1.0000 | 1 |
| **W1** | 0.657895 / 0.656522 | 2.083333 / 2.097222 | **3.1667 / 3.1944** | 2 |
| W2 | 0.811107 / 0.810260 | 1.443376 / 1.448179 | 1.7795 / 1.7873 | 2 |
| W3 | 1.000000 | 4.177778 | 4.1778 | 1 |

**[E]** W1 and W2 each show two distinct pairs because fold training-set sizes alternate between **150 and 151** participants (36 positive, 114–115 negative; training prevalence 0.2384–0.2400). W3's weights are constant by construction, being derived from the fixed corpus prevalence rather than from fold composition.

**[E] Weights were correctly data-dependent and leak-free:** `n_train` equals the manifest's training-fold size in all 100 cases, so the evaluated fold never contributed to its own weights.

**[I]** The four arms span a **4.2×** range of positive-class emphasis — from uniform to inverse-prevalence — which is a substantial intervention, not a marginal one. The ordering by ratio is W0 (1.00) < W2 (1.78) < W1 (3.17) < W3 (4.18). That ordering becomes the key to §10.

## 10. Prediction distribution analysis

**[E]** This is the section that identifies the mechanism, and the finding is unusually clean.

| Arm | ratio | p_pos min | p_pos mean | **p_pos max** | **band width** | pred_phq spread |
|---|---|---|---|---|---|---|
| W0 | 1.00 | 0.140420 | 0.268154 | 0.442162 | **0.029216** | 0.300402 |
| W2 | 1.78 | 0.182213 | 0.320197 | 0.496435 | **0.029236** | 0.298473 |
| **W1** | **3.17** | 0.232980 | **0.390228** | 0.567042 | **0.029362** | 0.294758 |
| W3 | 4.18 | 0.257059 | 0.425587 | 0.609557 | **0.029237** | 0.293059 |

**[E] The probability band width is invariant.** Across a 4.2× range of class weighting it varies only between **0.029216 and 0.029362** — a total spread of **0.000146**, or **0.5 %** of the band itself. Baseline-CV's band width is **0.0292**; Exp 4's P1 band was likewise **0.0292**.

**[E] The band's *location* moved monotonically with the weight ratio.** Mean `p_pos` rose 0.268154 → 0.320197 → 0.390228 → 0.425587 in exact rank order of w_pos/w_neg.

**[I] Class weighting translates the probability distribution without dilating it.** The model's *relative* ordering of participants is preserved — hence the unchanged ROC-AUC of §8.3 — while the whole compressed band slides upward toward τ = 0.5. Weighting adjusts *where the model sits*, not *how much it distinguishes*. Prior-collapse in the sense of an extremely narrow output band is completely untouched by the intervention.

**[E] The regression head was essentially unaffected:** `pred_var` 0.0057 → 0.0059 (paired Δ +0.0001, MDE 0.0006) and `pred_phq` spread 0.3004 → 0.2948, a slight *decrease*.

**[I]** This is exactly as expected — the weight tensor enters only the classification cross-entropy, leaving the MSE regression term untouched — and it is why C4 was met. The regression head was not protected by the intervention; it was simply not addressed by it. **[I]** C4's satisfaction should therefore carry no positive weight in the verdict.

### 10.1 Why W2 behaved identically to W0 — [E]

**[E]** W2's maximum probability across all 940 predictions is **0.496435**. The argmax threshold is 0.500000. The gap is **0.003565**.

| Arm | global p_pos max | gap to τ = 0.5 | predictions ≥ 0.5 | positives emitted |
|---|---|---|---|---|
| W0 | 0.442162 | +0.057838 | **0 / 940** | 0 |
| W2 | 0.496435 | **+0.003565** | **0 / 940** | 0 |
| W1 | 0.567042 | −0.067042 | **74 / 940** | 74 |
| W3 | 0.609557 | −0.109557 | **163 / 940** | 163 |

**[I]** W2 produced exactly zero positive predictions, and therefore metrics identical to W0, **not because sqrt-balancing failed to change the model but because its translation fell 0.0036 short of the threshold.** Had W2's band shifted by another 0.4 % of a probability unit, its classification metrics would have jumped discontinuously from the W0 column to something resembling W1's.

**[I]** This is the sharpest available demonstration that the reported classification metrics are a **threshold-crossing artifact of a rigid, translated band**, not a measure of learned separation. It also independently reconfirms Phase 11.7's structural finding — Baseline-CV's ceiling of 0.4422 leaves argmax unreachable by 0.0578 — and shows that weighting moves that ceiling without widening the band beneath it.

## 11. Degeneracy analysis

**[E]** A fold-run is *degenerate* when its predicted positive rate is exactly 0.0 or 1.0.

| Arm | degenerate fold-runs | pos_rate mean | positives emitted | fold-runs with Recall > 0 |
|---|---|---|---|---|
| W0 | **25 / 25** | 0.0000 | 0 / 940 | 0 / 25 |
| W2 | **25 / 25** | 0.0000 | 0 / 940 | 0 / 25 |
| **W1** | **24 / 25** | 0.0789 | 74 / 940 | **2 / 25** |
| W3 | **23 / 25** | 0.1737 | 163 / 940 | 5 / 25 |

**Reference points — [E]:** Baseline-CV **25/25** at argmax · Exp 4 policy P1 **24/25** · Exp 3 arm E1 (20 epochs) **22/25**.

**[E] C6 requires the primary arm to fall *below* Exp 4's 24/25. W1 recorded exactly 24/25 — equal, not below. C6 is NOT MET.**

**[E]** W1's non-degenerate behaviour is confined to two fold-runs: (repeat 2, fold 1) and (repeat 4, fold 4). W3's to five: (1,1), (1,4), (2,1), (4,4), (5,3).

**[I]** Even the most aggressive arm tested — W3, at inverse-prevalence weighting — leaves **23 of 25** fold-runs emitting a single constant class. **[I]** Notably, Exp 3's convergence arm achieved *better* degeneracy (22/25) than any weighting arm here, which is consistent with §10: convergence widened the band, whereas weighting only slid it.

## 12. Random-ranker comparison

**[E]** The control is the expected F1 of a random ranker emitting the same number of positives at the same fold prevalence, evaluated **per fold-run using that fold's own prevalence and then averaged** — the identical convention Experiment 4 used. Because the function is non-linear, evaluating it at an averaged positive rate instead would differ materially by Jensen's inequality; the independent auditor confirmed the per-fold-run path by reproducing Exp 4's stored value 0.211901 exactly.

| Fold | W1 F1 | Random-ranker F1 | Δ |
|---|---|---|---|
| 1 | 0.076596 | 0.076596 | **+0.000000** |
| 2 | 0.000000 | 0.000000 | 0.000000 |
| 3 | 0.000000 | 0.000000 | 0.000000 |
| 4 | 0.071111 | 0.077838 | **−0.006727** |
| 5 | 0.000000 | 0.000000 | 0.000000 |
| **mean** | **0.029541** | **0.030887** | **−0.001345** |

**[E]** Paired test: mean Δ **−0.0013453**, 95 % CI [−0.005081, +0.002390], MDE 0.003735, t(4) = −1.0000, p = 0.3739 — **not detectable**, and numerically below zero.

**[I] In the only two folds where the model committed to any positive prediction, its F1 exactly equalled the random-ranker control in one and fell below it in the other.** The positives W1 emits are not merely weakly informative; they are indistinguishable from what would be obtained by selecting the same number of participants at random. This is the single most direct measurement of the experiment's central question, and it answers it negatively.

**[E] Precision against prevalence** — the second half of C5:

| Quantity | Value |
|---|---|
| W1 precision (mean of per-fold) | **0.018363** |
| Per-fold precision | 0.047368, 0.000000, 0.000000, 0.044444, 0.000000 |
| Corpus prevalence (μ₀) | **0.239362** |
| mean Δ | **−0.220999** |
| 95 % CI | [−0.252246, −0.189752] |
| t(4) | **−19.6370** |
| p | **3.9663e-05** |

**[E] The precision shortfall is strongly detectable in the adverse direction** — the CI lies entirely below zero, and p = 0.00004.

**[I]** A classifier whose precision is an order of magnitude below the base rate is selecting *against* the positive class among those it flags. Of the 74 positives W1 emitted, the fraction that were correct is far lower than would arise from indiscriminate selection.

## 13. Comparison against Baseline-CV

**[E]** Summarised from §6, §8.2 and §10:

| Dimension | Baseline-CV | W1 | Assessment |
|---|---|---|---|
| Equivalence control | — | W0 bit-identical, max diff **0** | comparison is exact |
| MAE | 4.8127 | 4.8114 | Δ −0.0013, not detectable |
| PR-AUC | 0.3941 | 0.3902 | Δ −0.0039, **slightly down** |
| ROC-AUC | 0.6333 | 0.6267 | Δ −0.0067, not detectable, both exclude 0.5 |
| Recall / F1 | 0.0000 / 0.0000 | 0.0756 / 0.0295 | positive but not detectable |
| Accuracy | 0.7606 | 0.7179 | Δ **−0.0427** |
| p_pos band width | 0.0292 | 0.0294 | **unchanged** |
| p_pos mean | 0.2682 | 0.3902 | **+0.1221 — translated** |
| Degeneracy | 25/25 | 24/25 | −1 fold-run |

**[I]** Against its own frozen comparator, Experiment 5 improved nothing detectably, moved three discrimination metrics (PR-AUC, ROC-AUC, Accuracy) slightly *downward*, and changed exactly one thing decisively: **the location of the probability band.**

## 14. Comparison against Experiment 4

**[E] Reported threshold-conditionally.** Exp 4's Recall 0.5422 and F1 0.2215 were measured at threshold policy **P1**; Exp 5 was measured at **argmax**. These are **not like-for-like**, and the design records this caveat in `exp5_summary.json` under `exp4_reference`. C3 is therefore a directional check, not a controlled contrast.

| Quantity | Exp 4 (P1, tuned threshold) | Exp 5 (W1, argmax) |
|---|---|---|
| Recall | 0.5422 | 0.0756 |
| F1 | 0.2215 | 0.0295 |
| Precision vs prevalence | 0.1431 < 0.2394, t(4) = −5.8641, p = 0.004222 | 0.0184 < 0.2394, t(4) = −19.6370, p = 0.000040 |
| F1 vs random ranker | 0.2215 vs 0.211901, Δ **+0.009579**, not detectable | 0.029541 vs 0.030887, Δ **−0.001345**, not detectable |
| Degenerate fold-runs | 24 / 25 | 24 / 25 |
| p_pos band width | 0.0292 | 0.0294 |

**[I] The two experiments arrive at the same structural conclusion by opposite routes.** Exp 4 moved the threshold down to the band; Exp 5 moved the band up to the threshold. Both produced positive predictions, both failed precision-against-prevalence, both left F1 statistically indistinguishable from a random ranker, and both ended at 24/25 degenerate with an unchanged band width of ~0.0292.

**[I]** That convergence is itself evidence. Two independent interventions on opposite sides of the same decision boundary yield the same outcome, which points to the limitation residing in **the width of the probability band** — that is, in the representation — rather than in the placement of either the threshold or the band.

## 15. Comparison against Experiment 3

**[E]** Experiment 3 is the informative contrast because it is the only intervention so far that changed the band's *shape* rather than its position.

| Quantity | Baseline-CV | Exp 3 E1 (20 epochs) | **Exp 5 W1 (balanced CE)** |
|---|---|---|---|
| pred_var | 0.0057 | **9.4247** | 0.0059 |
| Paired Δ pred_var | — | **+9.4190, detectable** | +0.0001, not detectable |
| pred_phq spread | 0.3004 | **8.2133** | 0.2948 |
| p_pos band width | 0.0292 | **0.3002** | 0.0294 |
| p_pos mean | 0.2682 | 0.1838 | **0.3902** |
| Degenerate fold-runs | 25/25 | **22/25** | 24/25 |
| MAE | 4.8127 | 4.6272 (Δ −0.1856, n.d.) | 4.8114 (Δ −0.0013, n.d.) |
| ROC-AUC | 0.6333 | 0.6929 | 0.6267 |
| Verdict | — | **H₀** | **H₀** |

**[I] The two interventions are close to orthogonal, and neither is sufficient.**

- **Exp 3 dilated the band** — variance up 1,600×, band width up 10×, spread up 27× — while leaving its centre low and accuracy unmoved.
- **Exp 5 translated the band** — centre up +0.1221 — while leaving its width unchanged to within 0.5 %.

**[I]** Phase 13 concluded that convergence "broke the *prediction* collapse without improving *predictive accuracy*." Experiment 5 now adds the complementary result: weighting **relocates the decision boundary without improving predictive accuracy either.** Neither the shape nor the position of the output distribution is the operative constraint on discrimination.

**[I]** Phase 13 §8 anticipated this precisely, warning that Exp 4 showed Recall/F1 can rise with no discrimination, and Exp 3 showed variance can rise with no accuracy — *"both are ways of appearing to improve without improving"* — and requiring any Exp 5 claim to clear discrimination criteria rather than variance or positive-rate criteria. C5 and C6 were written for exactly this contingency, and they are what caught it.

## 16. Threats to validity

**[E] Addressed and discharged:**

| Threat | Mitigation | Status |
|---|---|---|
| Reimplemented training loop differs from frozen trainer | W0 gate, bit-identical on 940 predictions | **discharged** |
| Wrong optimizer silently substituted | hard abort unless `transformers.optimization.AdamW`; eps 1e-6, wd 0.0 verified | **discharged** |
| Weights leaking held-out labels | all 100 pairs re-derived from training folds only by an independent auditor | **discharged** |
| Fold or threshold drift from Baseline-CV | manifest order and `pred_class == (p_pos ≥ 0.5)` re-checked on all 100 files | **discharged** |
| Aggregation divergence from prior phases | self-test reproduces `baseline_cv_summary.json`, max diff 0.000e+00 | **discharged** |
| Post-hoc criterion selection | all 7 criteria and all 3 script SHAs pre-registered before execution | **discharged** |
| Random-ranker control mis-specified (Jensen) | per-fold-run convention re-derived; reproduces Exp 4's 0.211901 exactly | **discharged** |
| Verifier merely re-running the aggregator | auditor imports neither script, nor torch/transformers; 3443 independent checks | **discharged** |

**[E] Acknowledged and not fully controllable:**

- **[E]** Exp 4 comparison is threshold-conditional (§14) and is not a controlled contrast.
- **[I]** Final training loss is not comparable across arms, since each minimises a different objective; it is reported for provenance only.
- **[E]** Only three weighting schemes plus the control were tested. A scheme outside the 1.0–4.18 ratio range was not examined, and the pre-registration forbids extending it post hoc.
- **[I]** The bit-identical W0 result implies the pipeline is effectively deterministic for this configuration, so the 5 repeats measure fold assignment variability rather than training stochasticity. This narrows what "repeat" contributes but does not affect fold-level inference, which is the basis of every interval reported.

## 17. Limitations

**[E]** k = 5 yields wide intervals (t = 2.7764); W1's MAE CI spans [4.4075, 5.2153]. **[E]** With 45 positives in total, classification metrics remain the higher-variance side: W1's Precision CI [−0.0129, 0.0496], Recall CI [−0.0533, 0.2044] and F1 CI [−0.0207, 0.0798] all include negative values, reflecting fold-level variance rather than meaningful bounds. **[E]** W1's non-degenerate behaviour rests on **2 of 25** fold-runs, so C5's discrimination test is effectively supported by two folds; this is a genuine weakness of the estimate, though it points in the same direction as W3's larger five-fold support. **[I]** A null result cannot prove that no weighting scheme could work — the claim is bounded to "inverse-frequency weighting up to a 4.18× ratio does not produce discrimination at 3 epochs," not "no objective reweighting could." **[E]** Nothing here bears on RC-3 (multimodal), RC-4 (DP), RC-5 (federation) or RC-6 (restartability); Experiment 5 is a text-only, non-federated, non-private measurement. **[E]** Weighting was applied only to the classification cross-entropy; the MSE regression term and the 0.5 coefficient were untouched, so no claim is made about weighted regression objectives.

## 18. Final verdict

> **H₀ ACCEPTED. Class weighting relocated the decision boundary without demonstrated discrimination.**

**[E]** The verdict rests on the conjunction of Criteria 0, 5 and 6, and that conjunction is what makes it decisive rather than merely negative:

- **[E] Criterion 0** establishes that the harness reproduces Baseline-CV **bit-identically**, so every difference is attributable to the loss weighting alone and to nothing else.
- **[E] Criterion 5** establishes that the intervention produced **no discrimination**: precision 0.0184 against prevalence 0.2394 (t(4) = −19.64, p = 0.00004 — detectable in the adverse direction), and F1 0.0295 against a random-ranker control of 0.0309, equal in one supporting fold and below it in the other.
- **[E] Criterion 6** establishes that the intervention did not even relieve degeneracy relative to Exp 4: **24/25** fold-runs remain degenerate, exactly Exp 4's figure.

**[I]** Had the weighting failed to change the model at all, the experiment would have been inconclusive — one could argue the intervention was too weak. That escape is closed: the weighting demonstrably worked, shifting mean `p_pos` by **+0.1221** and pushing 74 predictions across a threshold that Baseline-CV could not reach at all. **Because the intervention took effect *and* discrimination did not follow, the unweighted objective is eliminated as the explanation for prior-collapse.**

**[I] The most important nuance — position and width are separable.** The probability band's width is invariant at **0.0292–0.0294** across a 4.2× range of class weighting, while its centre moves monotonically with that weighting. W2's arrest **0.0036** short of the threshold makes the point unarguable: whether an arm emits zero positives or dozens is decided by where a rigid band lands relative to τ, not by how well the model separates the classes. Combined with Exp 3 — which widened the band tenfold without improving accuracy — **the shape and position of the output distribution are both now excluded as the operative constraint.**

**[I]** What survives intact is the ranking signal: all four arms retain ROC-AUC confidence intervals excluding 0.5, unchanged from Baseline-CV. The model does order participants better than chance. The deficit is that this ordering is compressed into a band roughly 0.03 wide, and no intervention tested so far has widened it in a way that carries information.

## 19. Roadmap consequence

**[R]** RC-1 names three joint conditions for prior-collapse: **3 epochs · unweighted CE · class imbalance**.

**[E] All three cheap interventions in the Immediate block are now formally excluded:**

| Experiment | Phase | Factor tested | Verdict | What it excluded |
|---|---|---|---|---|
| **Exp 4** | 12 | decision rule (threshold) | **H₀** | Threshold relocation cannot recover discrimination — precision below prevalence, F1 indistinguishable from a random ranker, 24/25 degenerate |
| **Exp 3** | 13 | epoch budget | **H₀** | Convergence cannot recover accuracy — the model converged and MAE did not detectably move |
| **Exp 5** | 14 | imbalance-aware objective | **H₀** | Reweighting cannot recover discrimination — the band translates but does not dilate; precision falls further below prevalence |

**[I] The Immediate block is now closed, and closed cleanly.** Each of the three interventions was pre-registered, independently audited, and returned H₀ under criteria written before execution. No accepted change has entered the configuration, so the accepted configuration remains exactly Baseline-CV — which means all three results are mutually comparable and none is contingent on another.

**[I] By elimination, the operative constraint is upstream of the classifier head.** The decision threshold, the optimisation budget, and the loss weighting have each been tested and excluded. What remains untested is whether the frozen text representation carries recoverable positive-class signal at all, given that its output band is 0.03 wide and its ROC-AUC is 0.63.

**[I]** This is a stronger position than any single positive result would have provided. Three independent negative results with a bit-identical control arm constitute a well-supported claim about *where the problem is not* — and that claim is defensible precisely because the criteria were fixed in advance and the audits were independent.

## 20. Recommendation for the next phase

**[I]** These are recommendations for the user's consideration, not a roadmap modification and not an initiated experiment.

1. **[I] Document the Immediate block as closed.** Three pre-registered H₀ results with a bit-identical equivalence control is a complete, defensible unit of work and a natural chapter boundary for the capstone report. The negative results are the finding.

2. **[I] The evidence points to representation capacity as the next question.** The consistent picture — ranking preserved at ROC-AUC ≈ 0.63, band width immovable at ≈ 0.03, three interventions on the decision layer all null — suggests the next informative measurement concerns what the frozen MentalBERT features can support, not how the head is trained on them. **[R]** Whether that is pursued is a roadmap decision that belongs to the user.

3. **[E] Version 2 artifacts remain git-ignored and working-tree-only.** All 289 files across `baseline_cv/` (34), `exp3_convergence/` (97), `exp4_decision_rule/` (31) and `exp5_imbalance_objective/` (127) are excluded by `.gitignore:20` (`*.csv`) and `.gitignore:114` (`trainer_outputs/`). **[I]** Four phases of audited experimental evidence exist only in the working tree. Securing it is independent of any scientific decision and is the one item worth acting on regardless of direction.

4. **[E] Two documentation defects remain open** and are recorded here for completeness: the `PHASE_10_ROADMAP.md` §6 Exp-0b numbering inconsistency (the branch text names "Exp 3" while describing Exp 4's intervention), and the cosmetic unused loop variable in `verify_exp5.py`'s containment check. **[I]** The latter has no functional effect and was deliberately left uncorrected, since changing it would alter the script's SHA and invalidate the Appendix A.2 pre-registration.

## 21. Artifact references

```
trainer_outputs/exp5_imbalance_objective/          127 artifacts
├── exp5_summary.json / .md                aggregates, CIs, paired deltas, verdicts
├── exp5_metrics.csv                       100 rows (W0/W1/W2/W3 x 25 fold-runs)
├── class_weights.csv                      100 rows — per-fold w_neg, w_pos, train composition
├── degeneracy_report.csv                  100 rows — band, pos_rate, degeneracy, prevalence
├── w0_equivalence_check.json              the authoritative hard gate result
├── loop_correspondence.md                 frozen-loop / reimplemented-loop correspondence
├── {arm}_rep{r}_fold{f}_preds.csv         100 held-out prediction files
└── {arm}_rep{r}_metrics.csv               20 per-repeat metric tables
```

**Provenance:** frozen trainer `65b1902e…a230b` · fold manifest `b9a7a91f…af8fca2f` · dataset `9a241851…4fc55f00` · trivial control `2ecbfee3…f8c5d572` · Baseline-CV summary `f065f5e1…74871aac` · Exp 4 summary `5defdae2…1800aa6f2` · `run_exp5.py` `42345081…5cbb7f` · `aggregate_exp5.py` `66bbac11…e701ca57` · `verify_exp5.py` `57843f65…2b6b8`. Pre-registration: `PHASE_14_EXP5_DESIGN.md` (`1149b9ba…48c0a`), Appendix A.2 complete before execution.

**Audit:** `verify_exp5.py` — 3443 checks, 0 failures, EXP 5 AUDIT PASS: True, exit code 0.

---

*Result report only. No experiment re-run, no metric recomputed, no roadmap modified, no Experiment 5 output altered. Every [E] statement is traceable to `trainer_outputs/exp5_imbalance_objective/` or to the frozen Baseline-CV, Exp 3 and Exp 4 artifacts; every [I] statement is labelled as interpretation; every [R] statement cites the roadmap.*
