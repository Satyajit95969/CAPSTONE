# PHASE 13 / EXPERIMENT 3 — CONVERGENCE
## Official Result Report

**Phase:** 13 · **Experiment:** 3 of the Official Experiment Sequence
**Type:** Retraining of the frozen trainer under larger epoch budgets over the frozen Baseline-CV folds
**Pre-registration:** `PHASE_13_EXP3_DESIGN.md` (written and frozen before execution)
**Authority:** `PHASE_10_ROADMAP.md` §5 A-3, §6 Exp 3, §7 · `PHASE_11_EVALUATION_PROTOCOL.md`
**Comparator:** `Baseline-CV` (Phase 11.6), frozen · secondary context: `Exp 4` (Phase 12), frozen
**Convention:** **[E]** = measured from the Experiment 3 artifacts · **[I]** = interpretation

**Outcome: H₀ accepted.** Prior-collapse is **not** a budget artifact. Training to convergence broke the *prediction* collapse but produced **no detectable improvement in accuracy**.

---

## 1. Objective

**[R]** *"Establish whether prior-collapse is a budget artifact."*

Determine whether the collapse of both output heads to the label prior is a consequence of the fixed 3-epoch training budget — a default rather than a convergence criterion — by training the otherwise-identical frozen model under larger epoch budgets.

## 2. Methodology

**[E]** The single changed factor is the `epochs` argument passed to the frozen `fine_tune_supervised`. Because `epochs` is a **parameter** of the frozen function rather than a line inside it, no training code was reimplemented — the optimizer, learning rate, batch size, loss (`CE + 0.5·MSE`, unweighted), gradient clipping, architecture and tokenizer all remain inside the untouched trainer.

**[E] Pre-declared arms:** E0 = 3 epochs (equivalence control) · **E1 = 20 epochs (primary)** · E2 = 10 epochs (sensitivity). Design R = 5 repeats × k = 5 folds = 25 fold-runs per arm, 75 total.

**[E]** Convergence was established from the **training-loss trajectory**, harvested by capturing the frozen trainer's own per-epoch stdout. No validation split was carved out of the training partition, which would have reduced training-set size — a second factor. Training-set size remained 150/151 per fold.

**[E] Plateau rule (pre-declared, diagnostic only):** relative epoch-over-epoch training-loss improvement < 0.01 sustained for 2 consecutive epochs. The budgets were fixed in advance; convergence was *demonstrated* from the trajectory, never used to select which epoch to report.

**[E] Aggregation:** repeats averaged within fold first, then fold-level Student-t, df = k−1 = 4, t = 2.7764 — identical to Baseline-CV and Exp 4, so all three remain mutually comparable.

## 3. Execution summary

**[E]** Executed on Colab GPU with the frozen trainer SHA verified at startup (`65b1902e…a230b`) and `infer_dims → (None, None)` confirming the text-only architecture.

| Arm | Epochs | Fold-runs | Total compute | Mean per fold-run |
|---|---|---|---|---|
| E0 | 3 | 25 | 389.8 s | 15.6 s |
| **E1** | **20** | **25** | **2552.4 s** | **102.1 s** |
| E2 | 10 | 25 | 1289.0 s | 51.6 s |
| **Total** | | **75** | **4231.2 s (70.5 min)** | |

**[R]** Runtime increase is expected and is not a failure.

**[E] Artifacts:** 97 files — 75 per-fold prediction files, 15 per-repeat metrics files, 7 consolidated artifacts. Consolidated row counts are exact: `exp3_metrics.csv` 75, `distribution_diagnostics.csv` 75, `convergence_report.csv` 75, `loss_trajectories.csv` 825 (E0 = 75, E1 = 500, E2 = 250 — 25 fold-runs × each arm's budget).

**[E] Execution note.** An earlier invocation was interrupted, leaving the consolidated CSVs incomplete while the per-fold artifacts survived. The run reported here is a **complete, uninterrupted re-execution**; all consolidated files were written by a single invocation. No result in this report derives from a partial run.

## 4. Audit summary

**[E] Independent audit:** `verify_exp3.py` — **2461 checks, 0 failures, EXP 3 AUDIT PASS: True.** The auditor imports neither `run_exp3.py` nor `aggregate_exp3.py`, and neither torch nor transformers; every metric, trajectory, plateau verdict and diagnostic was re-derived from the raw prediction files.

**[E] Self-tests:**

| Test | Max abs diff | Result |
|---|---|---|
| Aggregation reproduces `baseline_cv_summary.json` | 0.000e+00 | PASS |
| **E0 equivalence gate** — 12 primary metrics inside Baseline-CV's 95 % CI | — | PASS |
| Modality ablation 0/0 (audio_dim, vision_dim) | — | PASS |

**[E] The E0 gate passed at a stronger level than the design required.** The pre-registration anticipated only *statistical* equivalence, since torch is unseeded at the CUDA level. In fact E0's predictions are **numerically identical to Baseline-CV on all 25 fold-runs** — maximum absolute difference in `pred_phq` and `p_pos` of **exactly 0**. Every E0 aggregate matches Baseline-CV to six decimal places.

**[I]** This is the strongest possible form of the harness-equivalence guarantee: the Experiment 3 harness does not merely approximate Baseline-CV, it reproduces it exactly. Any E1/E2 difference from Baseline-CV is therefore attributable solely to the epoch budget.

**[E] One verifier defect was found and corrected.** Check `J2` asserted a raw recursive file count (`== 34`) against the baseline directory, which coupled the audit to a bundling choice rather than to anything the experiment consumes: `baseline_cv_summary.md` is read by no script and was legitimately absent from the Colab bundle. `J2` was replaced by `J2a` (required SHA-gated inputs present) and `J2b` (`runs/` complete at 30 files), both derived rather than hard-coded. **[E]** No experiment logic, metric, output or conclusion was affected; `run_exp3.py` and `aggregate_exp3.py` remain at their pre-registered SHAs. Recorded as an erratum in `PHASE_13_EXP3_DESIGN.md` Appendix A.

## 5. Statistical results

### 5.1 Convergence analysis — [E]

| Arm | Epochs | First loss | Final loss | Total reduction | Last-epoch rel. improvement | Plateau reached |
|---|---|---|---|---|---|---|
| E0 | 3 | 31.7856 | 21.1187 | 10.6669 | 0.11178 | **0 / 25** |
| **E1** | **20** | 31.7856 | **9.6644** | **22.1213** | 0.05956 | **9 / 25** |
| E2 | 10 | 31.7856 | 17.5635 | 14.2221 | 0.01966 | 6 / 25 |

**[E]** Where reached, the plateau occurred at a mean epoch of 10.1 (range 7–17) for E1 and 8.3 (range 7–10) for E2.

**[I]** E0 reaching plateau in **zero** of 25 fold-runs, with an 11.2 % relative improvement still occurring at its final epoch, confirms empirically what Phase 10 recorded from the Phase 9.5 log: the 3-epoch budget stops training mid-descent. E1 reduced the loss to less than half E0's final value.

### 5.2 Aggregate metrics by arm — [E]

Mean ± fold SD [95 % CI], fold-level Student-t, df = 4.

| Metric | E0 (3 ep) | **E1 (20 ep)** | E2 (10 ep) |
|---|---|---|---|
| MAE | 4.8127 ± 0.3275 | **4.6272 ± 0.2793** | 4.6594 ± 0.3625 |
| RMSE | 6.1896 ± 0.3521 | 5.9878 ± 0.3995 | 5.7871 ± 0.3390 |
| **pred_var** | **0.0057 ± 0.0079** | **9.4247 ± 2.0454** | 0.6177 ± 0.1725 |
| MAE_mean_pred | 4.9135 ± 0.4041 | 4.9135 ± 0.4041 | 4.9135 ± 0.4041 |
| ROC-AUC | 0.6333 ± 0.0466 | 0.6929 ± 0.0742 | 0.6164 ± 0.0481 |
| PR-AUC | 0.3941 ± 0.0765 | 0.4719 ± 0.1282 | 0.4040 ± 0.0779 |
| balAcc | 0.5000 ± 0.0000 | 0.5098 ± 0.0149 | 0.5000 ± 0.0000 |
| MCC | 0.0000 ± 0.0000 | 0.0265 ± 0.0366 | 0.0000 ± 0.0000 |
| Precision | 0.0000 ± 0.0000 | 0.0629 ± 0.0912 | 0.0000 ± 0.0000 |
| Recall | 0.0000 ± 0.0000 | 0.0267 ± 0.0398 | 0.0000 ± 0.0000 |
| F1 | 0.0000 ± 0.0000 | 0.0345 ± 0.0483 | 0.0000 ± 0.0000 |
| Accuracy | 0.7606 ± 0.0035 | 0.7617 ± 0.0028 | 0.7606 ± 0.0035 |

### 5.3 Paired per-fold comparison: E1 vs Baseline-CV — [E]

Noise band re-estimated from Experiment 3's own fold spread, never inherited from Baseline-CV's degenerate MDEs.

| Metric | Baseline-CV | E1 | mean Δ | 95 % CI | MDE | Detectable |
|---|---|---|---|---|---|---|
| MAE | 4.8127 | 4.6272 | −0.1856 | [−0.7189, 0.3478] | 0.5334 | no |
| RMSE | 6.1896 | 5.9878 | −0.2018 | [−0.8783, 0.4746] | 0.6764 | no |
| **pred_var** | **0.0057** | **9.4247** | **+9.4190** | **[6.8710, 11.9669]** | 2.5479 | **YES** |
| ROC-AUC | 0.6333 | 0.6929 | +0.0595 | [−0.0384, 0.1574] | 0.0979 | no |
| PR-AUC | 0.3941 | 0.4719 | +0.0779 | [−0.1161, 0.2718] | 0.1939 | no |
| balAcc | 0.5000 | 0.5098 | +0.0098 | [−0.0087, 0.0283] | 0.0185 | no |
| MCC | 0.0000 | 0.0265 | +0.0265 | [−0.0190, 0.0719] | 0.0455 | no |
| Precision | 0.0000 | 0.0629 | +0.0629 | [−0.0504, 0.1762] | 0.1133 | no |
| Recall | 0.0000 | 0.0267 | +0.0267 | [−0.0227, 0.0760] | 0.0494 | no |
| F1 | 0.0000 | 0.0345 | +0.0345 | [−0.0254, 0.0945] | 0.0599 | no |
| Accuracy | 0.7606 | 0.7617 | +0.0011 | [−0.0019, 0.0041] | 0.0030 | no |

**[E] Per-fold MAE deltas:** −0.3008, −0.0680, −0.4318, −0.6206, **+0.4934**. Mean −0.1856 against an MDE of 0.5334 (and the roadmap's Baseline-CV anchor of 0.4066). **[I]** Four folds improved and one degraded; the direction is favourable but inconsistent, and the magnitude does not clear the noise band under either threshold.

### 5.4 Secondary observational outcomes — [E], NOT acceptance criteria

Per the approved design these are reported in full and treated as observational, never pass/fail.

| Arm | p_pos band width | p_pos mean | pred_var | pred_phq spread | Degenerate fold-runs | pos_rate |
|---|---|---|---|---|---|---|
| E0 | 0.0292 | 0.2682 | 0.005734 | 0.3004 | **25 / 25** | 0.0000 |
| **E1** | **0.3002** | 0.1838 | **9.4247** | **8.2133** | **22 / 25** | 0.0118 |
| E2 | 0.0796 | 0.1995 | 0.617723 | 1.9914 | 25 / 25 | 0.0000 |

Reference points: Baseline-CV band width **0.0292**, Baseline-CV degeneracy **25/25** at argmax, Exp 4 policy P1 degeneracy **24/25**.

**[E]** E1 produced non-zero Recall, F1 and MCC in **3 of 25** fold-runs. E0 and E2 produced them in **0 of 25**.

## 6. Acceptance criteria and verdicts

| # | Criterion | Source | Verdict |
|---|---|---|---|
| **0** | E0 reproduces Baseline-CV (all primary metrics inside its 95 % CI) | design gate | **MET** — reproduced *exactly*, max diff 0 |
| **1** | Loss reaches a stopping criterion (E1 plateau rate exceeds E0's) | Roadmap §6 Exp 3 | **MET** — 9/25 vs 0/25 |
| **2** | Regression MAE improves beyond the noise band (\|Δ\| > MDE 0.4066) | Roadmap §6 Exp 3 | **NOT MET** — Δ −0.1856, MDE 0.5334 |
| **3** | Modality ablation stays 0/0 | Roadmap §7.5 | **MET** |

## 7. Final conclusion

> **H₀ ACCEPTED. Prior-collapse is NOT a budget artifact; RC-1 needs more than epochs.**

**[E]** The verdict rests on the conjunction of Criteria 1 and 2, and that conjunction is what makes it decisive rather than merely negative:

- **Criterion 0** establishes that the harness reproduces Baseline-CV **exactly**, so the arms are strictly comparable.
- **Criterion 1** establishes that the model genuinely **did** train further: loss fell 31.7856 → 9.6644 (a 22.12 reduction versus E0's 10.67), and 9 of 25 fold-runs reached the pre-declared plateau where E0 reached it in none. This was not another truncated run.
- **Criterion 2** then establishes that this additional training bought **no detectable improvement in regression accuracy** — Δ −0.1856 against an MDE of 0.5334.

**[I]** Had the loss failed to plateau, the experiment would have been inconclusive: one could argue the budget was still insufficient. Because convergence was demonstrated *and* accuracy did not move, **the epoch budget is eliminated as the explanation for prior-collapse.**

### 7.1 The most important nuance — collapse and accuracy are separable

**[E]** Prediction variance rose from **0.0057 to 9.4247** — a paired increase of **+9.4190**, CI [6.8710, 11.9669] against an MDE of 2.5479, and the **only detectable change in the entire paired comparison.** The `pred_phq` spread rose from 0.3004 to 8.2133; the probability band widened from 0.0292 to 0.3002; degeneracy fell from 25/25 to 22/25; and 3 fold-runs produced non-zero Recall and F1 for the first time under argmax.

**[I] Convergence therefore broke the *prediction* collapse without improving *predictive accuracy*.** The model moved from emitting a near-constant value to emitting varied values that are, within the noise band, equally wrong. This distinction matters and should not be elided:

- The **Phase 11.1 finding** — that the regression head is a *constant estimator* — is now **conditional on the 3-epoch budget**. It is not an intrinsic property of the architecture or the objective. Under 20 epochs the head is no longer constant.
- The **Phase 11.7 finding** — that the head does not beat a mean predictor — **survives**. E1's MAE (4.6272) still does not detectably beat the in-run mean-predictor control (4.9135) by more than the noise band, and Criterion 2 is not met.

**[I]** Convergence relocated the failure rather than resolving it. That is a genuine advance in understanding: prediction variance is now a *solved* sub-problem, and the remaining deficit is discriminative accuracy.

## 8. Implications for RC-1

**[R]** RC-1 names three joint conditions for prior-collapse: **3 epochs · unweighted CE · class imbalance**.

**[E] Two of the roadmap's cheap interventions are now formally excluded:**

| Experiment | Factor tested | Verdict | What it excluded |
|---|---|---|---|
| **Exp 4** (Phase 12) | decision rule (threshold) | H₀ | Threshold relocation cannot recover discrimination — precision fell below prevalence, F1 was indistinguishable from a random ranker, 24/25 decisions degenerate |
| **Exp 3** (Phase 13) | epoch budget | H₀ | Convergence cannot recover accuracy — the model converged and MAE did not detectably move |

**[I] By elimination, the remaining RC-1 conditions are the training objective itself: unweighted cross-entropy and class imbalance.** Neither the decision rule nor the training budget is the operative constraint.

**[I]** Exp 3 also sharpens what Exp 5 must be judged on. Exp 4 established that Recall/F1 gains can occur with no discrimination whatsoever; Exp 3 now establishes that prediction *variance* can rise dramatically with no accuracy gain. **Both are ways of appearing to improve without improving.** Any Exp 5 claim must therefore clear discrimination and accuracy criteria, not variance or positive-rate criteria.

## 9. Why Experiment 5 is now justified

1. **[E] The roadmap's prerequisite is satisfied.** Exp 5's mandated comparator is *"Baseline-CV **and** Exp 4"* — both exist as measured, audited artifacts. Exp 3's completion additionally removes the attribution debt that would have existed had Exp 5 run first.
2. **[E] The accepted configuration is unchanged.** Exps 3 and 4 both returned H₀, so neither contributed an accepted change. Exp 5's epoch policy remains **3** and its decision rule remains **argmax**, preserving the one-factor rule relative to Baseline-CV.
3. **[I] Attribution is now clean.** Had Exp 5 run before Exp 3 and succeeded, we could not have excluded convergence as an alternative explanation. That ambiguity no longer exists: if class weighting moves the metrics, neither the threshold nor the budget can account for it.
4. **[R] Exp 5 changes the loss weighting**, directly targeting the documented condition *"unweighted CE on an imbalanced set, no class weighting"* — the only untested RC-1 condition remaining in the Immediate block.
5. **[I] Exp 3 supplies a concrete design input.** Because 20 epochs alone widens the distribution without improving accuracy, Exp 5's success criteria must be anchored to **discrimination** (precision above prevalence, PR-AUC above Baseline-CV) rather than to variance or positive count — a criterion set Exp 4's H₀ already motivated and Exp 3 now independently reinforces.

**[I] One structural caution carried into Phase 14.** Exp 3's one-factor guarantee was unusually strong because `epochs` is a *parameter* of the frozen function. Exp 5's factor — the loss — lives *inside* `fine_tune_supervised` and cannot be changed without reimplementing the training loop. Its W0 equivalence gate therefore does real work that Exp 3's E0 gate did not have to do, and it should be treated as the load-bearing element of that design.

## 10. Limitations

**[E]** k = 5 yields wide intervals (t = 2.7764); E1's MAE CI spans [4.2804, 4.9739]. **[E]** With 45 positives in total, classification metrics remain the higher-variance side; E1's Precision CI includes negative values, reflecting fold-level variance rather than a meaningful bound. **[E]** The plateau rule was satisfied in only 9 of 25 E1 fold-runs, so "convergence" is demonstrated at the arm level rather than universally per fold-run. **[I]** A budget beyond 20 epochs was not tested, and the pre-registration explicitly forbids extending it post hoc — the claim is bounded to "up to 20 epochs does not improve accuracy," not "no budget could." **[E]** Nothing here bears on RC-3 (multimodal), RC-4 (DP), RC-5 (federation) or RC-6 (restartability); Exp 3 is a text-only, non-federated, non-private measurement. **[E]** E2's intermediate position (pred_var 0.6177, degeneracy 25/25) shows the variance increase is budget-dependent and non-linear, but three budget points cannot characterise that curve.

## 11. Artifact references

```
trainer_outputs/exp3_convergence/          97 artifacts
├── exp3_summary.json / .md                aggregates, CIs, paired deltas, verdicts
├── exp3_metrics.csv                       75 rows (E0/E1/E2 x 25 fold-runs)
├── loss_trajectories.csv                  825 rows (E0 75, E1 500, E2 250)
├── convergence_report.csv                 75 rows — plateau epoch, losses
├── distribution_diagnostics.csv           75 rows — observational outcomes
├── e0_equivalence_check.json              the hard gate result
├── {arm}_rep{r}_fold{f}_preds.csv         75 held-out prediction files
└── {arm}_rep{r}_metrics.csv               15 per-repeat metric tables
```

**Provenance:** frozen trainer `65b1902e…a230b` · fold manifest `b9a7a91f…af8fca2f` · dataset `9a241851…4fc55f00` · `run_exp3.py` `def7f220…5f36f` · `aggregate_exp3.py` `faa3bd4a…41a40` · `verify_exp3.py` `4a1de483…5dfe` (post-erratum; pre-registered as `c31481c6…285cb`). Pre-registration: `PHASE_13_EXP3_DESIGN.md`.

---

*Result report only. No experiment re-run, no metric recomputed, no roadmap modified, no Experiment 3 output altered. Every [E] statement is traceable to `trainer_outputs/exp3_convergence/` or to the frozen Baseline-CV artifacts; every [I] statement is labelled as interpretation.*
