# PHASE 15 / EXPERIMENT 6 — LOSS-TERM REBALANCING
## Official Result Report

**Phase:** 15 · **Experiment:** 6 of the Official Experiment Sequence
**Type:** Retraining of the frozen architecture under a varied regression-loss coefficient over the frozen Baseline-CV folds
**Pre-registration:** `PHASE_15_EXP6_DESIGN.md` (`120e2d99…3b6a3`, written and frozen before any Experiment 6 code existed)
**Authority:** `PHASE_10_ROADMAP.md` §5 A, §6 Exp 6, §7 · `PHASE_11_EVALUATION_PROTOCOL.md`
**Comparators:** `Baseline-CV` (Phase 11.6) — binding · `Exp 5` (Phase 14) — like-for-like · `Exp 3` (Phase 13) — reference
**Convention:** **[E]** = measured from the Experiment 6 artifacts · **[I]** = interpretation · **[R]** = roadmap reference

**Outcome: H₀ accepted.** Loss-term rebalancing **dilated the probability band without moving it across the decision boundary.** For the first time in this project the intervention is confirmed to have operated on the mechanism it targeted — gradient competition was measurably reversed — and discrimination still did not follow.

---

## 1. Executive summary

**[E]** Experiment 6 varied the single scalar λ multiplying the regression MSE term across a pre-declared ladder — L0 = 0.5 (the frozen default), L1 = 0.0144 (primary), L2 = 0.1, L3 = 0.0 — over 100 fold-runs on the frozen folds, with every other factor held at Baseline-CV's values.

**[E] Three findings define the result:**

1. **The intervention worked at the mechanism level.** The measured encoder gradient ratio ‖∇reg‖ / ‖∇cls‖ fell **11.5313 → 2.5523 → 0.4126 → 0.0000** across decreasing λ. At the primary arm the regression term no longer dominates; it is outweighed by classification more than two-to-one.
2. **The probability band dilated, monotonically.** Band width rose **0.029216 → 0.043155 → 0.143786 → 0.245101**, a **4.92×** widening at L1 and **8.39×** at L3 — the first dilation any objective-side intervention has produced.
3. **No arm emitted a single positive prediction.** `p_pos ≥ 0.5` occurred **0 / 940** times in every arm; all four are **25/25 degenerate**; Recall, Precision, F1 and MCC are exactly **0.0000** throughout.

**[E]** Ranking improved detectably against Baseline-CV — ROC-AUC **0.6333 → 0.6713**, Δ **+0.0380**, CI [0.0031, 0.0728] against an MDE of 0.0348 — but PR-AUC's rise (**+0.0475**) did not clear its noise band (MDE 0.0754), so **C1 is not met**.

**[E]** Regression degraded: L1's MAE **4.9372** is **above** the Exp-0a mean-predictor bound of **4.9135**, and RMSE rose detectably (**+0.3953**, MDE 0.1271).

**[I]** The roadmap's stated failure condition for this experiment is *"classification improves while MAE degrades toward the Exp-0a mean-predictor bound ⇒ one collapse merely traded for the other."* What was observed is that half of it: MAE crossed the bound **without** classification improving.

## 2. Experimental objective

**[R]** Roadmap §6 Exp 6: *"**Loss-term rebalancing** · single changed factor: **loss scaling** · Objective: **Break the joint-collapse of both heads**."*

Determine whether the fixed 0.5 coefficient on the regression term is the operative cause of prior-collapse, by varying that coefficient alone while the epoch budget, the decision rule and the class weighting are all held at Baseline-CV's values — Exps 3, 4 and 5 all returned H₀ and therefore contributed no accepted change.

**[E] The single changed factor**, at `trainer_mentalbert_daic.py:324`:

```python
loss = loss_cls + LAMBDA * loss_reg      # frozen: LAMBDA = 0.5
```

Both terms backpropagate through **one shared MentalBERT encoder** (`self.bert`, `:245`) feeding two heads (`self.classifier` `:231`, `self.phq_mu` `:232`), so the two objectives compete for the same representation.

## 3. Verification status

**[E] Independent audit:** `verify_exp6.py` — **3973 checks, 0 failures, EXP 6 AUDIT PASS: True, exit code 0.** The auditor imports neither `run_exp6.py` nor `aggregate_exp6.py`, and neither torch, transformers nor the frozen trainer; every constant is restated independently and every metric re-derived from the raw prediction files.

**[E] Self-tests:**

| Test | Max abs diff | Result |
|---|---|---|
| Aggregation reproduces `baseline_cv_summary.json` | 0.000e+00 | PASS |
| `exp5_metrics.csv` agrees with SHA-gated `exp5_summary.json` | 0.000e+00 | PASS |
| **L0 equivalence gate** — 12 primary metrics inside Baseline-CV's 95 % CI | — | PASS |

**[E] Artifacts:** 127 files — 100 per-fold prediction files, 20 per-repeat metrics files, 7 consolidated artifacts. Consolidated row counts are exact: `exp6_metrics.csv` 100, `loss_decomposition.csv` 100, `degeneracy_report.csv` 100, each with L0 = L1 = L2 = L3 = 25 and every `(arm, repeat, fold)` cell present exactly once.

**[E] Installation note.** The Colab results archive carried a nested wrapper directory `exp6_loss_rebalance/`, unlike Experiment 5's flat archive. The wrapper was detected and stripped during extraction; all 127 files were verified byte-identical to the archive, and the installed directory is flat.

**[E] Execution summary.** Colab GPU, frozen trainer SHA verified at startup and after the run, optimizer confirmed as `transformers.optimization.AdamW`.

| Arm | λ | Fold-runs | Total compute | Mean per fold-run | Final train loss (mean) |
|---|---|---|---|---|---|
| L0 | 0.5 | 25 | 450.5 s | 18.02 s | 21.1187 |
| L2 | 0.1 | 25 | 452.1 s | 18.08 s | 4.7190 |
| **L1** | **0.0144** | **25** | **451.5 s** | **18.06 s** | **1.2277** |
| L3 | 0.0 | 25 | 449.9 s | 18.00 s | 0.5405 |
| **Total** | | **100** | **1804.0 s (30.1 min)** | | |

**[E]** L0's mean final training loss, **21.1187**, is identical to Baseline-CV's and to Experiment 3's E0 arm.

**[I]** Final training loss is **not comparable across arms** — each minimises a different objective, so L3's 0.5405 reflects a smaller objective, not better optimisation. It is reported for provenance only.

## 4. Aggregate metrics

**[E]** Mean ± fold SD, fold-level Student-t, df = 4, t = 2.7764451051977987. Arms ordered by **decreasing λ**.

| Metric | L0 (λ=0.5) | L2 (λ=0.1) | **L1 (λ=0.0144)** | L3 (λ=0.0) |
|---|---|---|---|---|
| MAE | 4.8127 ± 0.3275 | 4.8150 ± 0.3362 | **4.9372 ± 0.4507** | 6.6398 ± 0.8135 |
| RMSE | 6.1896 ± 0.3521 | 6.2134 ± 0.3530 | **6.5849 ± 0.3914** | 8.8184 ± 0.5915 |
| pred_var | 0.0057 ± 0.0079 | 0.0061 ± 0.0088 | **0.0089 ± 0.0107** | 0.0048 ± 0.0024 |
| MAE_mean_pred | 4.9135 ± 0.4041 | 4.9135 ± 0.4041 | 4.9135 ± 0.4041 | 4.9135 ± 0.4041 |
| **ROC-AUC** | 0.6333 ± 0.0466 | 0.6624 ± 0.0452 | **0.6713 ± 0.0280** | 0.6708 ± 0.0403 |
| **PR-AUC** | 0.3941 ± 0.0765 | 0.4234 ± 0.0868 | **0.4416 ± 0.0690** | 0.4403 ± 0.0799 |
| balAcc | 0.5000 ± 0.0000 | 0.5000 ± 0.0000 | 0.5000 ± 0.0000 | 0.5000 ± 0.0000 |
| MCC | 0.0000 ± 0.0000 | 0.0000 ± 0.0000 | 0.0000 ± 0.0000 | 0.0000 ± 0.0000 |
| Precision | 0.0000 ± 0.0000 | 0.0000 ± 0.0000 | 0.0000 ± 0.0000 | 0.0000 ± 0.0000 |
| Recall | 0.0000 ± 0.0000 | 0.0000 ± 0.0000 | 0.0000 ± 0.0000 | 0.0000 ± 0.0000 |
| F1 | 0.0000 ± 0.0000 | 0.0000 ± 0.0000 | 0.0000 ± 0.0000 | 0.0000 ± 0.0000 |
| Accuracy | 0.7606 ± 0.0035 | 0.7606 ± 0.0035 | 0.7606 ± 0.0035 | 0.7606 ± 0.0035 |

**[E] Every classification decision metric is identical across all four arms** — balAcc 0.5000, MCC 0.0000, Precision 0.0000, Recall 0.0000, F1 0.0000, Accuracy 0.7606 — because no arm predicted a single positive.

**[E] L1 primary-arm confidence intervals:** ROC-AUC [0.6365, 0.7061] · PR-AUC [0.3559, 0.5273] · MAE [4.3776, 5.4968] · pred_var [−0.0045, 0.0222].

**[E] Ranking is above chance in every arm** — ROC-AUC CIs: L0 [0.5755, 0.6911], L2 [0.6063, 0.7184], L1 [0.6365, 0.7061], L3 [0.6207, 0.7209]; all four exclude 0.5.

**[E] L3 caveat (design §11.0).** λ = 0 leaves `phq_mu` unsupervised, so L3's MAE 6.6398, RMSE 8.8184 and pred_var 0.0048 characterise an **untrained head** and are descriptive only — not regression performance.

## 5. Acceptance criteria C0–C6

**[E]** Evaluated on the primary arm **L1** only (C0 on L0), per design §11. Verdicts as recorded and as independently re-derived by the auditor.

| # | Criterion | Measured | Verdict |
|---|---|---|---|
| **0** | L0 reproduces Baseline-CV | 12/12 inside CI, max diff **0** | **MET** |
| 1 | PR-AUC detectably above 0.3941 | 0.4416, Δ **+0.0475**, MDE 0.0754 | **NOT MET** |
| 2 | Recall and F1 above Baseline-CV | 0.0000 vs 0.0000, Δ = 0 | **NOT MET** |
| 3 | Recall and F1 above Exp 5 W1 | 0.0000 < 0.0756; 0.0000 < 0.0295 | **NOT MET** |
| 4 | MAE not materially degraded | Δ +0.1245 ≤ 0.4066 ✓ but MAE **4.9372 > 4.9135** ✗ | **NOT MET** |
| **5** | **Discrimination, not quantity** | precision **0.0000 < 0.2394**; F1 0.0000 vs random 0.0000 | **NOT MET** |
| **6** | Non-degenerate decisions | **25/25**, not below Exp 5's 24/25 | **NOT MET** |

**[E]** `overall_verdict`: *"H0 — loss-term rebalancing did not produce demonstrated discrimination"*, computed from `verdict_inputs: ["C0", "C1", "C5"]` alone.

**[I] C4 fails on its second clause, and that matters.** The paired MAE change (+0.1245) stays inside the roadmap's 0.4066 anchor, but L1's absolute MAE **4.9372 exceeds the Exp-0a mean-predictor bound 4.9135**. The design wrote C4 as a conjunction precisely so a model could not pass by drifting slowly past the trivial baseline; that clause is what caught this.

**[I] C2 and C3 fail here in a way they did not in Experiment 5.** Exp 5 produced *some* positives, so C2 passed on a bare inequality. Experiment 6 produced none at all, so C2 fails outright and C3 fails against Exp 5's own small non-zero values. **Experiment 6 is, on decision metrics, strictly behind Experiment 5.**

## 6. L0 equivalence gate

**[E] PASSED, at the strongest available level.** Recorded `authoritative: true`, `failures: []`, `control_lambda: 0.5`, 12/12 metrics inside Baseline-CV's 95 % CI.

| Test | Result |
|---|---|
| L0 aggregate inside Baseline-CV 95 % CI, 12/12 metrics | **PASS** |
| max \|L0 mean − Baseline-CV mean\| over 12 metrics | **0.000e+00** |
| max \|Δ `pred_phq`\| over all 25 fold-runs (940 predictions) | **0.000000e+00** |
| max \|Δ `p_pos`\| over all 25 fold-runs (940 predictions) | **0.000000e+00** |

**[E] L0's predictions are bit-identical to Baseline-CV on all 940 held-out predictions.**

**[I]** This discharges the structural caution the design raised in §10. λ is a *literal inside* the frozen `fine_tune_supervised`, so the training loop had to be reimplemented; L0 proves the reimplementation reproduces the frozen computation exactly rather than approximately. **[I]** Because §10.1 forbade any `λ == 0` branch, all four arms execute identical statements — so the gate is **transitive**: certifying L0 certifies the code path L1, L2 and L3 ran. Every difference below is attributable to λ alone.

## 7. Gradient instrumentation results

**[E]** Encoder gradient norms, measured on the first step of each epoch via `torch.autograd.grad` on `model.bert` parameters, **before** the combined backward — so no forward pass was re-run and the optimizer update was unaffected.

| Arm | λ | ‖∇<sub>enc</sub> loss_cls‖ | ‖∇<sub>enc</sub> λ·loss_reg‖ | **gradient ratio** |
|---|---|---|---|---|
| L0 | 0.5 | 2.8214 | 30.8070 | **11.5313** |
| L2 | 0.1 | 2.5297 | 6.2138 | **2.5523** |
| **L1** | **0.0144** | **2.5690** | **1.0297** | **0.4126** |
| L3 | 0.0 | 2.7023 | **0.0000** | **0.0000** |

**[E] The L3 zero-reference is exactly 0.0000**, confirming both that λ = 0 contributes no encoder gradient and that the instrument itself is correctly wired.

**[E] The gradient ratio falls monotonically with λ and crosses parity between L2 and L1.** At the primary arm the regression term contributes **less than half** the encoder gradient that classification does.

**[E] ‖∇ loss_cls‖ is essentially invariant** across the ladder (2.5297 – 2.8214, a spread of 0.2917), so the ratio moves because the regression contribution collapses, not because the classification signal changes.

**[I] This is the measurement design §7.1 committed to making, and it did the work it was designed for.** λ = 0.0144 was derived as *loss-value* parity; the design recorded in advance that loss-value parity does not imply gradient parity. It did not: at L0 the loss ratio is **41.265** while the gradient ratio is **11.5313**. Had only loss values been recorded, the null would have been unfalsifiable — one could always argue gradient parity was never reached. **It was reached, and passed.**

## 8. Loss decomposition analysis

**[E]** Per-arm means over the logged steps (3 per fold-run).

| Arm | λ | loss_cls | loss_reg | λ·loss_reg | **loss ratio** |
|---|---|---|---|---|---|
| L0 | 0.5 | 0.6249 | 51.2420 | 25.6210 | **41.265** |
| L2 | 0.1 | 0.5914 | 51.8012 | 5.1801 | **8.761** |
| **L1** | **0.0144** | **0.5790** | **56.2663** | **0.8102** | **1.397** |
| L3 | 0.0 | 0.5778 | 72.0982 | 0.0000 | **0.000** |

**[E] The design's §4 estimate is corroborated.** It computed the loss ratio at Baseline-CV's operating point as **34.8 : 1** from frozen artifacts alone; the measured L0 value is **41.265 : 1** — the same order and the same conclusion, with the difference attributable to the estimate having used held-out MSE (38.3111) where the measurement uses the training MSE (51.2420).

**[E] `loss_reg` rises as λ falls** — 51.2420 → 51.8012 → 56.2663 → 72.0982 — while `loss_cls` falls slightly, 0.6249 → 0.5778.

**[I]** That is exactly what de-weighting predicts: the regression head is optimised less hard and its error grows, most sharply at L3 where it receives no gradient at all. The classification loss improves only marginally (−0.0471 from L0 to L3), which is the first quantitative hint that freeing encoder capacity did not translate into much better classification even in-sample.

## 9. Comparison against Baseline-CV

**[E]** Paired per-fold, noise band re-estimated from Experiment 6's own fold spread.

| Metric | Baseline-CV | L1 | mean Δ | 95 % CI | MDE | Detectable |
|---|---|---|---|---|---|---|
| MAE | 4.8127 | 4.9372 | +0.1245 | [−0.1256, 0.3745] | 0.2501 | no |
| **RMSE** | 6.1896 | 6.5849 | **+0.3953** | **[0.2682, 0.5224]** | 0.1271 | **YES** |
| pred_var | 0.0057 | 0.0089 | +0.0031 | [−0.0008, 0.0071] | 0.0040 | no |
| **ROC-AUC** | 0.6333 | 0.6713 | **+0.0380** | **[0.0031, 0.0728]** | 0.0348 | **YES** |
| PR-AUC | 0.3941 | 0.4416 | +0.0475 | [−0.0278, 0.1229] | 0.0754 | no |
| balAcc | 0.5000 | 0.5000 | 0.0000 | [0.0000, 0.0000] | 0.0000 | no |
| MCC | 0.0000 | 0.0000 | 0.0000 | [0.0000, 0.0000] | 0.0000 | no |
| Precision | 0.0000 | 0.0000 | 0.0000 | [0.0000, 0.0000] | 0.0000 | no |
| Recall | 0.0000 | 0.0000 | 0.0000 | [0.0000, 0.0000] | 0.0000 | no |
| F1 | 0.0000 | 0.0000 | 0.0000 | [0.0000, 0.0000] | 0.0000 | no |
| Accuracy | 0.7606 | 0.7606 | 0.0000 | [0.0000, 0.0000] | 0.0000 | no |

**[E] Exactly two changes are detectable, and they point in opposite directions:** ROC-AUC improved (+0.0380) and RMSE degraded (+0.3953).

**[E] Per-fold MAE deltas:** +0.1057, +0.4247, +0.1404, +0.0927, **−0.1411** — four folds worse, one better.
**[E] Per-fold ROC-AUC deltas:** +0.0326, +0.0460, **−0.0077**, +0.0643, +0.0548 — four folds better, one worse.

**[I]** The picture is a modest, real gain in *ordering* bought at the cost of a real loss in *regression fit*, with no change whatsoever in decisions.

## 10. Comparison against Experiment 5

**[E] Like-for-like.** Experiment 5 ran at argmax and 3 epochs — identical to Experiment 6 — so unlike Exp 5's own comparison against Exp 4, no threshold caveat applies. Both are paired on the same frozen folds.

| Metric | Exp 5 W1 | Exp 6 L1 | mean Δ | 95 % CI | MDE | Detectable |
|---|---|---|---|---|---|---|
| MAE | 4.8114 | 4.9372 | +0.1258 | [−0.1275, 0.3790] | 0.2532 | no |
| **RMSE** | 6.1832 | 6.5849 | **+0.4017** | **[0.2725, 0.5310]** | 0.1293 | **YES** |
| pred_var | 0.0059 | 0.0089 | +0.0030 | [−0.0005, 0.0064] | 0.0035 | no |
| **ROC-AUC** | 0.6267 | 0.6713 | **+0.0446** | **[0.0115, 0.0778]** | 0.0331 | **YES** |
| PR-AUC | 0.3902 | 0.4416 | +0.0514 | [−0.0019, 0.1047] | 0.0533 | no |
| balAcc | 0.4978 | 0.5000 | +0.0022 | [−0.0039, 0.0084] | 0.0062 | no |
| MCC | −0.0118 | 0.0000 | +0.0118 | [−0.0209, 0.0444] | 0.0326 | no |
| Precision | 0.0184 | 0.0000 | −0.0184 | [−0.0496, 0.0129] | 0.0312 | no |
| Recall | 0.0756 | 0.0000 | −0.0756 | [−0.2044, 0.0533] | 0.1288 | no |
| F1 | 0.0295 | 0.0000 | −0.0295 | [−0.0798, 0.0207] | 0.0503 | no |
| Accuracy | 0.7179 | 0.7606 | +0.0427 | [−0.0299, 0.1152] | 0.0726 | no |

**[E] Structural comparison:**

| Quantity | Baseline-CV | Exp 5 W1 (weighting) | **Exp 6 L1 (rebalancing)** |
|---|---|---|---|
| p_pos band width | 0.029216 | 0.029362 (**unchanged**) | **0.143786 (×4.92)** |
| p_pos mean | 0.268154 | 0.390228 (**translated up**) | 0.212100 (translated **down**) |
| global max p_pos | 0.442162 | 0.567042 | 0.407019 |
| predictions ≥ 0.5 | 0 / 940 | **74 / 940** | **0 / 940** |
| degenerate fold-runs | 25/25 | 24/25 | **25/25** |

**[I] The two experiments are near-perfect complements, and neither succeeds.** Experiment 5 *translated* a rigid band upward until part of it crossed τ, producing positives that proved to be of random-ranker quality. Experiment 6 *dilated* the band — the thing Exp 5 could not do — but simultaneously re-centred it **downward**, so its wider distribution still lies entirely below the threshold. **[E]** Exp 6 has the better ordering (ROC-AUC +0.0446, detectable) and the worse decisions (0 positives versus 74).

## 11. Interpretation of the gradient findings

**[I]** The gradient measurements change what this null result means, and they are the most valuable thing Experiment 6 produced.

**[E] The causal chain the design hypothesised was confirmed at every step except the last:**

| Step | Predicted | Measured |
|---|---|---|
| λ ↓ reduces the regression gradient at the encoder | yes | ratio 11.5313 → 0.4126 → 0.0000 ✅ |
| freed capacity widens the probability band | yes | 0.029216 → 0.143786 → 0.245101 ✅ |
| a wider band produces positive predictions | yes | **0 / 940 in every arm** ❌ |
| positives carry discrimination | yes | **no positives to assess** ❌ |

**[I] Gradient competition was real, and removing it was not sufficient.** The band's *width* was genuinely constrained by the regression term — that is now measured, not inferred. But width was never the binding constraint on decisions: **position was, and rebalancing moved position the wrong way.**

**[E] The threshold-crossing table makes this exact:**

| Arm | λ | band width | global max p_pos | gap to τ = 0.5 | crossings |
|---|---|---|---|---|---|
| L0 | 0.5 | 0.029216 | 0.442162 | +0.057838 | 0 / 940 |
| L2 | 0.1 | 0.043155 | 0.304342 | +0.195658 | 0 / 940 |
| L1 | 0.0144 | 0.143786 | 0.407019 | +0.092981 | 0 / 940 |
| L3 | 0.0 | 0.245101 | 0.484034 | **+0.015966** | 0 / 940 |

**[E] L3 — with the regression term removed entirely — came within 0.015966 of the threshold and still never crossed it.**

**[I]** That is the sharpest statement Experiment 6 supports: the maximum achievable effect of loss rebalancing, measured at its own upper bound, is a distribution that approaches but does not reach the decision boundary. **[I]** It also means Experiment 6's null cannot be dismissed as too weak an intervention — the bounding arm was included in the pre-registration precisely to close that escape, and it closed it.

## 12. Why the null result is scientifically interpretable

**[I]** Design §19 pre-declared the conditions under which an H₀ here would be decisive rather than merely negative. All of them are satisfied.

1. **[E] The control is exact.** L0 is bit-identical to Baseline-CV across 940 predictions, and the uniform code path makes that guarantee transitive to every treatment arm. No implementation risk remains.
2. **[E] The intervention demonstrably took effect.** Gradient ratio 11.5313 → 0.4126, band width ×4.92. This is not a case of a factor that failed to move.
3. **[E] Gradient parity was crossed, not merely approached.** At L1 the ratio is 0.4126 — the regression term is now the *minority* contributor. The confound §7.1 warned about (loss-value parity ≠ gradient parity) is measured and excluded.
4. **[E] The bound was tested.** L3 removes the regression objective entirely and still yields 0 positives.
5. **[E] The criteria were fixed before execution** and the digests recorded in Appendix A.2 before any artifact existed.

**[I]** Had the gradient norms not been recorded, this report could only have said *"we reduced a coefficient and nothing happened"* — leaving open whether the coefficient reduction ever reached the shared encoder. It did, by a factor of **28×** in the gradient ratio, and discrimination still did not appear. **That converts an ambiguous null into an informative one.**

## 13. Discussion

**[E] What changed:** ordering (ROC-AUC +0.0380 vs Baseline-CV, +0.0446 vs Exp 5, both detectable), band width (×4.92), and encoder gradient balance (×28 shift in ratio).

**[E] What did not change:** every decision metric. balAcc 0.5000, MCC 0.0000, Precision 0.0000, Recall 0.0000, F1 0.0000, Accuracy 0.7606 — identical in all four arms and identical to Baseline-CV.

**[E] What degraded:** RMSE (+0.3953, detectable) and MAE, which crossed the mean-predictor bound at L1 (4.9372 > 4.9135) and collapsed at L3 (6.6398).

**[I] The regression head's degradation is expected and is not a defect** — de-weighting a term is supposed to worsen it. What matters is that the trade was not rewarded: **regression accuracy was spent and classification decisions bought nothing.** That is the roadmap's "one collapse merely traded for the other" failure mode, realised in its least favourable form, where only the cost side materialised.

**[E] The λ-ordering analysis (observational, not a criterion).** Ordered by decreasing λ (L0, L2, L1, L3): PR-AUC [0.3941, 0.4234, 0.4416, 0.4403] and ROC-AUC [0.6333, 0.6624, 0.6713, 0.6708] are **not** monotone — L3 dips slightly below L1 on both. Band width [0.029216, 0.043155, 0.143786, 0.245101] **is** monotone. Recall, Precision and F1 are trivially monotone at 0.

**[I]** The near-monotone ranking curve peaking at L1 rather than L3 is consistent with a mild optimum around gradient parity, but **[I]** with four arms and k = 5 this is not evidence — it is why the design demoted the ordering analysis from a criterion to an observation before execution, and that decision should be respected in any write-up.

## 14. Threats to validity

**[E] Addressed and discharged:**

| Threat | Mitigation | Status |
|---|---|---|
| Reimplemented loop differs from frozen trainer | L0 gate, bit-identical on 940 predictions | **discharged** |
| Gate does not cover treatment arms | uniform code path, no `λ == 0` branch (§10.1) | **discharged** |
| Gradient logging perturbs training | `autograd.grad` before backward; no forward re-run, no `.grad` write; L0 still bit-identical | **discharged** |
| Instrument mis-wired | L3 zero-reference measured **exactly 0.0000** | **discharged** |
| Loss-value parity ≠ gradient parity | gradient norms measured directly; parity crossed | **discharged** |
| Intervention too weak to conclude | L3 bounding arm (λ = 0) included | **discharged** |
| Wrong optimizer substituted | hard abort unless `transformers.optimization.AdamW` | **discharged** |
| Fold or threshold drift | manifest order and `pred_class == (p_pos ≥ 0.5)` re-checked on all 100 files | **discharged** |
| Aggregation divergence | self-test reproduces `baseline_cv_summary.json` at 0.000e+00 | **discharged** |
| Ungated Exp 5 comparator | `exp5_metrics.csv` re-derived against SHA-gated `exp5_summary.json` at 0.000e+00 | **discharged** |
| Post-hoc criterion selection | 7 criteria and 3 script SHAs pre-registered before execution | **discharged** |
| Observational metrics influencing the verdict | `verdict_inputs: ["C0","C1","C5"]`; auditor re-derives and enforces | **discharged** |

**[E] Acknowledged and not fully controllable:**

- **[E]** Gradient norms were sampled on the first step of each epoch (3 per fold-run), not every step; they characterise the arm, not every update.
- **[E]** L3's regression metrics describe an unsupervised head and are excluded from interpretation by design §11.0.
- **[E]** Four λ values were tested. No value between 0.0144 and 0.1, or below 0.0144 but above 0, was examined, and the pre-registration forbids extending the ladder post hoc.
- **[I]** The near-bit-identical L0 result implies the pipeline is effectively deterministic for this configuration, so the 5 repeats measure fold-assignment variability rather than training stochasticity.

## 15. Final verdict

> **H₀ ACCEPTED. Loss-term rebalancing dilated the probability band without producing discrimination.**

**[E]** The verdict rests on Criteria 0, 1 and 5, and its force comes from what the instrumentation adds:

- **[E] Criterion 0** establishes the harness reproduces Baseline-CV **bit-identically**, so every difference is attributable to λ and to nothing else.
- **[E] Criterion 1** establishes that PR-AUC rose (+0.0475) but **not detectably** (MDE 0.0754).
- **[E] Criterion 5** establishes there was **no discrimination to assess**: precision is exactly 0.0000 against a prevalence of 0.2394, because the classifier emitted no positives in any of 940 predictions.

**[I] The decisive nuance — the constraint is position, not width.** Version 2 has now separated the two properties of the output distribution and eliminated both:

| Experiment | Factor | Effect on the band | Decisions |
|---|---|---|---|
| Exp 3 | epochs | width 0.0292 → 0.3002 | 22/25 degenerate, no accuracy gain |
| Exp 5 | class weighting | **position** +0.1221, width unchanged | 24/25 degenerate, random-quality positives |
| **Exp 6** | **loss coefficient** | **width ×4.92, position −0.0561** | **25/25 degenerate, no positives** |

**[I]** Three interventions have moved the output distribution in three different ways — widened it, translated it up, widened it while translating it down — and none produced a decision that carries information. **[I]** What survives every one of them is the ranking signal: ROC-AUC excludes 0.5 in all four Exp 6 arms and in Baseline-CV, and Exp 6 improved it detectably. The model orders participants better than chance; nothing tested has converted that ordering into decisions.

## 16. Roadmap consequence

**[R]** RC-1 names three joint conditions for prior-collapse: **3 epochs · unweighted CE · class imbalance**.

**[E] Four experiments, four H₀ results, four excluded factors:**

| Experiment | Phase | Factor | Verdict | What it excluded |
|---|---|---|---|---|
| Exp 4 | 12 | decision rule | **H₀** | Threshold relocation cannot recover discrimination |
| Exp 3 | 13 | epoch budget | **H₀** | Convergence cannot recover accuracy |
| Exp 5 | 14 | imbalance-aware objective | **H₀** | Reweighting translates the band but does not dilate it |
| **Exp 6** | **15** | **loss-term rebalancing** | **H₀** | **Rebalancing dilates the band but does not move it across τ — even at λ = 0** |

**[I] The objective-side interventions are now exhausted.** The decision threshold, the optimisation budget, the class weighting and the loss-term balance have each been tested against a bit-identical control, with pre-registered criteria and independent audits. Every one returned H₀, so **no accepted change has entered the configuration and the accepted configuration remains exactly Baseline-CV** — which keeps all five measurements mutually comparable.

**[I] By elimination, and now with direct gradient evidence, the constraint is upstream of the loss.** Experiment 6 measured that the shared encoder can be freed of regression competition entirely and still yields a probability distribution that does not reach the decision boundary. That is a statement about what the frozen text representation supports, not about how the head is trained on it.

**[R]** Roadmap §6 places **Exp 7 (multimodal activation)** next, comparator *"best accepted text-only config (Exp 6)"*. **[I]** Since Exp 6 returned H₀, the best accepted text-only configuration is still Baseline-CV — the same chained-comparator situation `VERSION2_FINAL_REVIEW.md` M3 recorded and design §11.2 resolved for this experiment. That policy will need restating for Exp 7.

## 17. Recommendations [I]

**[I]** These are recommendations for the user's consideration. No roadmap modification is made and no experiment is initiated.

1. **[I] Treat the objective-side block as closed.** Four pre-registered H₀ results with a bit-identical control, one of them instrumented at the gradient level, is a complete and defensible unit of work. The negative results, taken together, are a stronger claim than any single positive result would have been.

2. **[I] The evidence now points specifically at representation capacity.** Ranking sits at ROC-AUC ≈ 0.63–0.67 and improves slightly when the encoder is freed for classification, but the probability distribution never reaches τ = 0.5 even at λ = 0. **[R]** Whether that is pursued — and whether it precedes Exp 7 — is a roadmap decision belonging to the user.

3. **[I] Resolve the chained-comparator policy before Exp 7 is designed.** Design §11.2 settled it for Exp 6 (binding comparator = Baseline-CV, prior experiment = like-for-like reference). The same wording will serve Exp 7, but it should be stated in that design rather than inherited silently.

4. **[E] Version 2 artifacts remain git-ignored and working-tree-only.** All 416 files across `baseline_cv/` (34), `exp3_convergence/` (97), `exp4_decision_rule/` (31), `exp5_imbalance_objective/` (127) and `exp6_loss_rebalance/` (127) are excluded by `.gitignore:20` (`*.csv`) and `:114` (`trainer_outputs/`). **[I]** Five phases of audited evidence exist only in the working tree; securing it is independent of any scientific decision and remains the one item worth acting on regardless of direction.

## 18. Provenance

**[E] Frozen inputs**, SHA-verified before and after the run and recorded in `exp6_summary.json`:

```
65b1902e…a230b   trainer_mentalbert_daic.py
9a241851…4fc55f00 daic_records.parquet
b9a7a91f…5af8fca2f trainer_outputs/baseline_cv/fold_manifest.json
2ecbfee3…f8c5d572 trainer_outputs/baseline_cv/trivial_control_arm.csv
f065f5e1…674871aac trainer_outputs/baseline_cv/baseline_cv_summary.json
5defdae2…1800aa6f2 trainer_outputs/exp4_decision_rule/exp4_summary.json
9dd135b6…20af2f9  trainer_outputs/exp3_convergence/exp3_summary.json
70ad13ff…3ed48d3  trainer_outputs/exp5_imbalance_objective/exp5_summary.json
a1692ac4…f60bdc84 trainer_outputs/exp5_imbalance_objective/exp5_metrics.csv
```

**[E] Implementation code**, pre-registered in `PHASE_15_EXP6_DESIGN.md` Appendix A.2 **before execution**:

```
5933bc29eff056c9f976c67cde51edb2b311461b3368bd276dc2024cbdd5c5f2  exp6_loss_rebalance/run_exp6.py
546ad22f89ef3581aeccbbf930286a2d7dc06cc7f0d009cc9559160675194799  exp6_loss_rebalance/aggregate_exp6.py
fc7c0f742cbd062728c17cc4a3cd53e226330d2cee3c640862d26694aa74d61d  exp6_loss_rebalance/verify_exp6.py
```

**[E] Pre-registration:** `PHASE_15_EXP6_DESIGN.md` SHA-256 `120e2d995ba30ab16586fb80b7a8cdc5e77ad860c6c61790032ca30d40d3b6a3`.
**[E] Audit:** `verify_exp6.py` — 3973 checks, 0 failures, EXP 6 AUDIT PASS: True, exit code 0.

## 19. Appendix — artifact inventory

```
trainer_outputs/exp6_loss_rebalance/          127 artifacts
├── exp6_summary.json / .md                aggregates, CIs, paired deltas, verdicts
├── exp6_metrics.csv                       100 rows (L0/L1/L2/L3 x 25 fold-runs)
├── loss_decomposition.csv                 100 rows — loss_cls, loss_reg,
│                                          lambda_loss_reg, ratio, grad_norm_cls,
│                                          grad_norm_reg, grad_norm_ratio
├── degeneracy_report.csv                  100 rows — band, pos_rate, degeneracy
├── l0_equivalence_check.json              the authoritative hard gate result
├── loop_correspondence.md                 frozen-loop / reimplemented-loop mapping
├── {arm}_rep{r}_fold{f}_preds.csv         100 held-out prediction files
└── {arm}_rep{r}_metrics.csv               20 per-repeat metric tables
```

**[E] Statistical protocol:** repeats averaged within fold first, then fold-level Student-t, df = k−1 = 4, **t = 2.7764451051977987** — identical to Baseline-CV and Exps 3, 4 and 5, so all five remain mutually comparable. Corpus prevalence **0.2393617021276596**. MDE anchors: MAE **0.4066**, PR-AUC **0.0950**.

---

*Result report only. No experiment re-run, no metric recomputed, no roadmap modified, no Experiment 6 output altered, no previous phase document changed. Every **[E]** statement is traceable to `trainer_outputs/exp6_loss_rebalance/` or to the frozen Baseline-CV, Exp 3, Exp 4 and Exp 5 artifacts; every **[I]** statement is labelled as interpretation; every **[R]** statement cites the roadmap.*
