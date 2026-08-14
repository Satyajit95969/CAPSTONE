# PHASE 11.7 — SCIENTIFIC INTERPRETATION OF BASELINE-CV

**Type:** Analysis and synthesis only — no experiment, no retraining, no code delivered, no artifact modified
**Inputs:** `PHASE_11_BASELINE_CV.md` (closed) · `baseline_cv_summary.json` · `baseline_cv_summary.md` · `runs/rep{1..5}_metrics.csv` (25 fold-runs) · `runs/rep{r}_fold{f}_preds.csv` (940 held-out predictions) · `trivial_control_arm.csv`
**Convention:** **[E]** = measured from the above · **[I]** = interpretation

---

## 1. Purpose

Phase 11.6 produced the numbers. Phase 11.7 asks what they mean, and — critically — which of the Phase 11.1–11.3 conclusions, all drawn at n = 18, survive contact with adequate statistical power.

The headline is a genuine reversal: **the frozen model discriminates between participants better than chance, and it has never once been observed to do so.** Both halves of that sentence are true simultaneously, and reconciling them is the substance of this analysis.

---

## 2. Regression: the collapse is confirmed and the null result is robust

**[E] Aggregate.** MAE **4.8127 ± 0.3275**, CI [4.4061, 5.2193]. In-run mean-predictor control **4.9135 ± 0.4041** — reproducing the frozen control arm to four decimals (4.9135 = 4.9135), independently confirming the fold partition and label handling.

**[E] Paired analysis.** Per-fold differences (model − its own control): −0.0942, +0.4982, +0.0016, −0.0724, −0.8371. Paired mean **Δ̄ = −0.1008**, paired SD 0.4776, SE 0.2136. The operative accept/reject statistic — the paired per-fold test the protocol mandates — gives **MDE₉₅ = 0.5930**. The planning MDE reported in §6 is 0.4066.

**[I] The model does not beat the mean predictor.** |Δ̄| = 0.1008 clears neither threshold, and the sign of the per-fold differences is inconsistent (2 positive, 3 negative). This is a **null result with adequate power to be called a null result**, not an absence of evidence. Note also that the operative paired MDE (0.593) *exceeds* the planning estimate (0.407): the model–control differences are more variable across folds than Baseline-CV's own fold spread implies. **[I] Phase 12 should quote the paired figure, not the planning figure**, or it will systematically over-claim detectability.

**[E] RMSE tells a sharper story.** Model 6.1896 versus control 5.8752 — the model is **worse by 0.3144**.

**[I] This is diagnostic, not incidental.** The mean predictor is by construction the L2-optimal constant. A model that loses to it on RMSE while roughly matching it on MAE is behaving as a *slightly mis-located constant*. That is a precise characterisation of the failure mode, and it is stronger evidence than MAE alone.

**[E] Prediction collapse persists.** Aggregate `pred_var` **0.005734**, CI **[−0.004089, 0.015557]** — includes zero. Across all 940 held-out predictions, `pred_phq` spans **3.7946 to 5.0673** (range 1.27 on a 0–23 scale); pooled SD of predictions **0.2541** against a true PHQ SD of **5.8590**, a ratio of ~23:1. Per fold-run, `pred_var` ranges 0.000059–0.027887.

**[I] The regression head remains a constant estimator.** The observed range is wider than the frozen baseline's 0.06 spread, but that comparison is not like-for-like: 1.27 spans 25 *separately trained models*, whereas 0.06 came from one. Within a single fold-run the variance remains statistically indistinguishable from zero. **RC-1 holds for regression, now demonstrated at power.**

---

## 3. Classification: two results that appear contradictory

**[E] Threshold-conditional metrics are exactly degenerate.** Precision = Recall = F1 = **0.0000**, balanced accuracy = **0.5000**, MCC = **0.0000** — on every one of the 25 fold-runs, with zero variance from either noise source. Accuracy **0.7606 ± 0.0035**, identical to the majority-class rate.

**[E] Threshold-free metrics are not.** ROC-AUC **0.6333 ± 0.0466**, CI **[0.5755, 0.6911]** — the interval excludes 0.5. PR-AUC **0.3941 ± 0.0765**, CI [0.2990, 0.4891], against a prevalence baseline of 0.2394; the CI lower bound exceeds prevalence.

**[E] Three independent confirmations that the AUC result is not chance:**
- **25 of 25** individual fold-runs have AUC > 0.5 (min 0.5476, median 0.6369, max 0.7663). Sign test **p = 5.96 × 10⁻⁸**.
- One-sample *t* on the five per-fold means versus 0.5: **t = 6.4032, p = 0.00306**.
- The fold-level Student-*t* CI excludes 0.5 outright.

**[I]** These are not one marginal test. The effect is present in *every* partition and every repeat.

---

## 4. Why ROC-AUC > 0.5 with P/R/F1 = 0 changes the understanding of the model

This is the central scientific result of Baseline-CV, and it is mechanically explicable.

**[E] The decisive measurement.** Across all **940** held-out predictions, positive-class probability spans **0.1404 to 0.4422**. Predictions with p ≥ 0.5: **0 of 940**. Predictions assigned class 1: **0 of 940**.

**[I] The 0.5 decision boundary is unreachable by construction.** It is not that the classifier decides badly at the margin — no input the model has ever seen produces a probability that could cross argmax. Precision, recall, F1, MCC and balanced accuracy are therefore not measuring the model's discriminative capacity at all; they are measuring the distance between the probability distribution and an arbitrary constant. Their being exactly zero, with exactly zero variance, is the signature of a **structurally unreachable threshold**, not of a model that cannot distinguish anything.

**[I] The two families of metrics are answering different questions.** ROC-AUC and PR-AUC ask *does the model order participants correctly?* — answer: yes, modestly but consistently. P/R/F1 ask *does the model, at argmax 0.5, name anyone depressed?* — answer: never, and it never could.

**[E] One caveat, stated because it constrains the strength of the claim.** Pooling all 940 predictions, mean p_pos is 0.270153 for depressed participants versus 0.267427 for non-depressed — a separation of only **+0.002726**, though now in the **correct direction**.

**[I] The pooled gap understates within-fold discrimination and must not be read as the effect size.** Each fold-run is a separately trained model with its own probability offset; pooling adds between-model variance that swamps the within-fold ordering. AUC is computed *within* each fold-run, which is why it registers signal the pooled comparison hides. This is itself a methodological lesson: **for this model, only within-fold rank statistics are interpretable.**

**[I] The honest characterisation.** AUC 0.633 is *real but weak* — well below any clinically useful threshold (conventionally ≥ 0.75), and its CI reaches down to 0.576. The correct statement is: **the model has acquired a small amount of genuine ordering information from text, which the training recipe compresses into a probability band far too narrow and too low to survive a fixed 0.5 cut.** The failure is now demonstrably *composite*: a weak-learning failure **and** a decision-rule failure, where previously only the former was believed to exist.

---

## 5. Noise structure and its consequences for experiment design

**[E] Decomposition (partition SD vs training-stochasticity SD, ratio):**

| Metric | Partition | Training | Ratio |
|---|---|---|---|
| MAE | 0.3275 | 0.0387 | **8.46×** |
| RMSE | 0.3521 | 0.0738 | 4.77× |
| ROC-AUC | 0.0466 | 0.0428 | **1.09×** |
| PR-AUC | 0.0765 | 0.0525 | 1.46× |
| balAcc / MCC / P / R / F1 | 0.0000 | 0.0000 | — |

**[I] Regression and ranking have structurally different noise regimes.** For MAE, *which participants land in a fold* dominates run-to-run training variation by more than 8:1 — so a single-run regression comparison is nearly all partition noise, and the paired frozen-fold design is doing almost all the work. For ROC-AUC the two components are of equal magnitude, meaning **repeats are not optional for classification claims**: a Phase 12 experiment reporting an AUC change from one run would be quoting roughly half the true noise band. R ≥ 5 must be retained.

**[I] The zero variance on the threshold-conditional metrics is not stability.** It is degeneracy. Those metrics acquire a real noise band only once a configuration produces non-constant predictions — so their MDE of 0.0000 in §6 must never be read as "any change is detectable."

**[E] Confidence intervals** use fold-level Student-*t*, df = k−1 = 4, t = 2.7764 throughout, with repeats averaged within fold before the interval is formed.

**[I] This is the correct unit of independence** and materially wider than pooling the 25 correlated fold-runs under a normal quantile would have produced. The intervals are honest; k = 5 makes them wide, and that width is a property of the design, not a defect in the measurement.

---

## 6. Comparison with Phase 11.1–11.3: confirmed, refined, superseded

| Prior conclusion | Source | Status after Baseline-CV |
|---|---|---|
| Regression head is a constant/mean estimator | 11.1 | ✅ **CONFIRMED at power.** `pred_var` CI includes zero; pooled prediction SD 0.2541 vs true 5.8590. |
| Model does not beat the mean predictor on MAE | 11.1 | ✅ **CONFIRMED, and strengthened.** Paired Δ̄ = −0.1008 against operative MDE 0.5930 — a powered null. RMSE is *worse* than control by 0.3144. |
| The −24.1 % MAE headline is not evidence of learning | 11.1 | ✅ **CONFIRMED.** |
| Predictions are all-negative under argmax 0.5 | 11.2 / 9.5 | ✅ **CONFIRMED exactly.** 0 of 940 positive predictions. This is Roadmap §6's Exp-2 success criterion — the re-measurement is faithful. |
| Accuracy ≡ majority rate | 11.2 | ✅ **CONFIRMED.** 0.7606 model = 0.7606 control. |
| Only the split is seeded / partition drives variability | 11.3 (RC-7 partial) | ✅ **CONFIRMED and quantified.** Partition:training = 8.46× on MAE. |
| Evaluation at n = 18 cannot support or refute a hypothesis | 11.3 (RC-2) | ✅ **CONFIRMED — and vindicated.** The n=18 diagnostic reached the *opposite* ranking conclusion. RC-2 was not a caveat; it was load-bearing. |
| **"The classifier carries no detectable ranking information"** | **11.2** | ❌ **SUPERSEDED.** AUC 0.6333, CI [0.5755, 0.6911]; 25/25 fold-runs above chance; sign test p = 5.96e-8. |
| **"ROC-AUC 0.523, CI [0.250, 0.787], permutation p = 0.446"** | **11.2** | ❌ **SUPERSEDED.** Point estimate was uninformative at that CI width; the true value lies outside its lower half. |
| **"Rank direction slightly wrong" (depressed scored lower)** | **11.2** | ❌ **SUPERSEDED.** Direction is correct: +0.002726 pooled, and within-fold ordering is consistently correct across all 25 runs. |
| **"The failure is substantially a learning failure, not a decision-rule defect"** | **11.3** | ⚠️ **PARTIALLY SUPERSEDED.** It is *both*. Learning is weak (AUC 0.633, not 0.85); but a real ordering exists that the decision rule discards entirely. |
| **"Exp 4's ceiling is low… expectation lowered"** | **11.3** | ❌ **SUPERSEDED.** See §7. |
| **"Joint prior-collapse — one failure mode expressed twice"** | **11.3** | ⚠️ **REFINED.** The regression head is genuinely collapsed. The classification head is *compressed and mis-thresholded*, not informationally empty. The two heads fail differently. |
| RC-3…RC-6 (multimodal, DP, federation, restartability) | 11.3 | ➖ **Not exercised.** Baseline-CV adds no evidence; they stand as documented in the frozen reports. |

**[I] The methodological lesson is as important as the scientific one.** Phase 11.3 reasoned correctly from the evidence it had and reached a conclusion that was wrong, because n = 18 could not distinguish AUC 0.52 from AUC 0.63. Phase 10's insistence that measurement precede modelling (Exp 1 → Exp 2 before Exp 3+) is retrospectively justified: had Version 2 proceeded straight to training-recipe experiments, it would have been optimising against a mischaracterised failure mode.

---

## 7. Recommended next experiment: **Exp 4 (decision rule)**, promoted ahead of Exp 3

The roadmap sequences Exp 3 (convergence) → **Exp 4 (decision rule)** → 5 → 6, on the reasoning that experiments should ascend in invasiveness. Baseline-CV does not change that principle — it changes which experiment the evidence now points at.

**Justification, each item traced to Baseline-CV evidence:**

1. **[E] A real ordering exists, and the current decision rule discards 100 % of it.** AUC 0.6333 (CI excludes 0.5) with 0/940 positive predictions. Exp 4 is the *only* experiment that directly targets this gap.
2. **[E] The threshold is provably unreachable, not merely badly chosen.** Max observed p_pos = 0.4422 < 0.5. Any threshold at or above 0.4422 yields identical all-negative output — so this is a structural defect with a deterministic remedy.
3. **[E] Exp 4's success criterion is now plausibly reachable.** Roadmap §6 requires "Recall > 0 and F1 > 0 — the first non-zero classification result in the project's history." With genuine ranking present, a calibrated operating point should produce non-trivial recall; under 11.3's chance-ranking assumption it would only have produced recall proportional to prevalence.
4. **[E] It is the cheapest experiment available.** It changes no training whatsoever, requires **no GPU**, and reuses the 25 existing prediction files already in the repository. Roadmap §5 already names it "the safest possible intervention."
5. **[E] Its invariant is verifiable.** Roadmap §6 requires regression MAE to be *unchanged* under Exp 4 — trivially checkable, since no training occurs, and any movement signals contamination.
6. **[E] Conversely, Exp 3's target has just been shown not to be detectably movable by evidence of this kind.** Convergence addresses the regression collapse; Baseline-CV shows the regression head fails a powered test against a constant with Δ 0.1008 vs MDE 0.5930. Exp 3 remains scientifically warranted, but it costs a full GPU retrain to probe the arm where Baseline-CV found no detectable signal, while Exp 4 costs nothing to probe the arm where it did.

**[I] This is a reordering, not a redefinition.** Both experiments remain in scope, each still changes exactly one factor, and both compare paired-per-fold against the frozen manifest. Reordering the roadmap is a governance decision and remains the project owner's.

**[I] A deliberate abstention.** This analysis has *not* computed what a prevalence-matched or Youden-optimal threshold would yield, though the artifacts permit it. Doing so would be conducting Exp 4 without its protocol — selecting an operating point on the same folds used to measure it, with no declared selection rule. Exp 4 must specify its threshold policy *in advance* and report it threshold-conditionally, or it will produce an optimistically biased result of exactly the kind this phase series exists to prevent.

---

## 8. Limitations of this analysis

**[E]** k = 5 yields wide intervals (t = 2.7764); AUC's CI spans 0.576–0.691. **[E]** 45 positives in total; classification remains the higher-variance side. **[E]** R = 5 bounds training stochasticity but does not eliminate it, and for ranking metrics it is comparable to partition noise. **[I]** AUC 0.633 is modest — this analysis establishes that ordering information *exists*, not that it is sufficient for any clinical purpose. **[E]** Nothing here bears on RC-3 (multimodal), RC-4 (DP), RC-5 (federation) or RC-6 (restartability); Baseline-CV is a text-only, non-federated, non-private measurement.

---

## 9. Conclusion

**[E]** Baseline-CV reproduces the frozen baseline's qualitative signature exactly — P/R/F1 = 0.0000 and all-negative predictions on all 25 fold-runs — satisfying Roadmap §6's Exp-2 criterion and confirming the re-measurement is faithful.

**[E]** Beneath that identical surface it reveals something the 18-sample evaluation could not: the regression head is a confirmed constant estimator that fails a powered comparison against the mean predictor, while the classification head carries **genuine, weak, consistently-directed ranking information** (ROC-AUC 0.6333, CI [0.5755, 0.6911]; 25/25 fold-runs above chance) that is **entirely destroyed by an unreachable 0.5 decision boundary** (max p_pos 0.4422 across 940 predictions).

**[I]** The project's classification failure is therefore not one problem but two, with different remedies and very different costs. The expensive one — creating stronger signal — belongs to the training-recipe experiments. The cheap one — not discarding the signal already present — is Exp 4, requires no training, and is now the highest-yield next step the evidence supports.

**Phase 11.7 outcome: Baseline-CV interpreted. Regression collapse confirmed at power. Ranking-at-chance conclusion (11.2) and Exp-4 downgrade (11.3) superseded. Exp 4 recommended as the next experiment.**

---

*Analysis only. No experiment performed, no model run, no code delivered, no document, artifact, roadmap or result modified. `PHASE_11_BASELINE_CV.md` verified unchanged at SHA `56b2cc46…accb`. Every [E] claim is derived from the Phase 11.6 artifacts named above; every [I] claim is labelled as interpretation.*
