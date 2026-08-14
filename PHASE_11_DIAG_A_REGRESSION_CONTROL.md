# PHASE 11.1 — DIAGNOSTIC A: REGRESSION TRIVIAL-BASELINE CONTROL
## Experiment 0a of the Official Phase 10 Roadmap

**Phase:** 11.1 — First scientific diagnostic of Version 2
**Type:** Analysis only — no training, no code change, no prediction regeneration
**Roadmap trace:** `PHASE_10_ROADMAP.md` §5 (A-0), §6 (Experiment 0a), §7.4
**Comparator:** `PHASE_9.5_COMPLETE_DATASET_REPORT.md` §6.1 (frozen MAE 6.654737075169881)

Throughout: **[E]** = observed evidence (measured from frozen artifacts) · **[I]** = interpretation.

---

## 1. Objective

Determine whether the MentalBERT regression head learned meaningful **participant-level** information, or whether it behaves like a **trivial constant predictor** (one that ignores the input and emits a fixed value).

The Phase 9.5 baseline reported a headline regression result — **MAE improved 8.7652 → 6.6547 (−24.1 %)** — but also reported (§7.1) that the model's predicted PHQ values spanned only ~0.06 on a 0–23 scale. Experiment 0a exists to resolve the resulting question directly: *is 6.6547 evidence of learning, or the MAE of a constant?*

---

## 2. Methodology

The diagnostic compares the frozen model's regression MAE against the MAE of trivial predictors evaluated on the **identical validation participants**.

**The central obstacle:** the frozen `eval_preds.csv` contains only `pred_phq` — it carries **no participant IDs and no true PHQ labels**. A trivial-baseline comparison is only valid if it is computed on the *same* 18 targets the model was scored against. Those targets therefore had to be reconstructed and then **verified**.

**Reconstruction (read-only replay of the frozen split):**
The trainer's split is deterministic (`trainer_mentalbert_daic.py` lines 560–566):
```
idx = np.arange(n)            # n = 188, parquet row order
np.random.seed(42)
np.random.shuffle(idx)
vs  = int(n * 0.1)            # = 18
val_idx = idx[:vs]            # the 18 validation records, in eval order
```
Evaluation runs with `shuffle=False` (line 599), so the rows of `eval_preds.csv` correspond one-to-one, in order, to `val_idx`. Replaying these exact NumPy operations on the frozen parquet recovers the 18 validation participants and their true PHQ scores.

**Verification gate (the check that licenses everything else):**
The reconstructed true targets were aligned to the frozen `pred_phq` column and the model MAE was recomputed. It reproduced the frozen value **exactly**:

| Quantity | Value |
|---|---|
| Model MAE (recomputed from reconstruction) | `6.654737075169881` |
| Model MAE (frozen, P9.5 §6.1) | `6.654737075169881` |
| Bit-exact match | **✅ True** |

[E] Because the reconstruction reproduces the frozen MAE to full floating-point precision, the identified validation participants and their PHQ targets are confirmed correct. All subsequent baselines are computed on these verified targets.

**Trivial predictors.** A trivial predictor ignores the input and emits one constant. The fair, model-comparable constants are derived from the **training** labels only (the model likewise never saw the validation labels):
- **Mean Predictor** — always predict `mean(PHQ_train)`.
- **Median Predictor** — always predict `median(PHQ_train)` (the median is the MAE-minimizing constant of a distribution).

For completeness, three **reference** constants are also reported (oracle val-mean, oracle val-median, and a constant fixed at the model's own average prediction). The oracle constants use the validation labels and are therefore *not* fair competitors — they serve only as the theoretical MAE floor.

---

## 3. Data Used

All inputs are frozen Phase 9.5 artifacts. Nothing was modified or regenerated.

| Artifact | Role |
|---|---|
| `daic_records.parquet` (188 records) | Source of true PHQ scores and split replay |
| `trainer_outputs/eval_preds.csv` (18 rows) | Frozen model predictions (`pred_phq`) |
| `trainer_outputs/training_report.json` | Frozen headline MAE 6.654737075169881 |
| `trainer_mentalbert_daic.py` (SHA `65b1902e…a230b`) | Read only, to confirm split/eval logic |

**Reconstructed validation set (18 participants, verified):**

- Participant IDs: `315, 316, 318, 343, 346, 356, 361, 367, 368, 377, 411, 418, 426, 450, 459, 463, 467, 488`
- True PHQ (eval order): `0, 0, 3, 2, 7, 0, 23, 16, 6, 9, 0, 20, 9, 16, 10, 19, 10, 0`
- Range 0–23; **5 of 18 positive** (PHQ > 10); val mean 8.333, val median 8.0.
- Training set: 170 participants; **train mean 6.5059, train median 5.0**.

---

## 4. Baseline Calculations

[E] Measured values on the 18 verified validation targets:

| Predictor | Constant emitted | MAE | Mean prediction | Prediction variance |
|---|---|---|---|---|
| **MentalBERT model** | ~5.03–5.09 (near-constant) | **6.654737** | 5.054871 | **0.000354** |
| **Mean Predictor** (train mean) | 6.505882 | **6.388235** | 6.505882 | 0.000000 |
| **Median Predictor** (train median) | 5.000000 | **6.666667** | 5.000000 | 0.000000 |
| *[ref] Oracle mean (val mean)* | 8.333333 | *6.333333* | 8.333333 | 0.000000 |
| *[ref] Oracle median (val median)* | 8.000000 | *6.333333* | 8.000000 | 0.000000 |
| *[ref] Constant at model's own mean pred* | 5.054871 | *6.654473* | 5.054871 | 0.000000 |

[E] **The model's own prediction spread is 0.058468** (min 5.027390, max 5.085858, std 0.018819). A constant fixed at the model's average prediction (5.054871) scores MAE **6.654473** — within **0.000264** of the model's actual 6.654737. [I] The model's 18 predictions are, for scoring purposes, a single constant; their 0.058 of variation changes MAE by ~0.0003.

---

## 5. Results

Three observations, each directly measured:

**R1 — The model is a constant predictor.** [E] Prediction variance is 0.000354 versus a target variance of ~46; the full prediction range (0.058) is 0.25 % of the label range (0–23). A constant at the model's own mean matches its MAE to within 0.0003.

**R2 — The model does not beat the fair trivial mean.** [E] Model MAE 6.654737 is **higher (worse)** than the Mean Predictor's 6.388235 by **0.266502**. The model only edges the Median Predictor (6.666667) by **0.011930**, and it loses to both oracle constants (6.333333) by **0.321404**.

**R3 — The model's constant is sub-optimal.** [E] The model emits ≈5.05, close to the train median (5.0) but far from the val-set MAE-optimal constant (8.0). [I] Its narrow win over the median baseline is a coincidence of where its constant landed, not participant-level discrimination.

---

## 6. Comparison Table

*"Difference vs Model" = predictor MAE − model MAE. Negative ⇒ the trivial predictor is **better** than the model.*

| Predictor | MAE | Difference vs Model | Interpretation |
|---|---|---|---|
| **MentalBERT model** | 6.654737 | — | [E] Near-constant output (variance 3.5e-4); emits ≈5.05 for every participant. |
| **Mean Predictor** (train mean 6.51) | 6.388235 | **−0.266502** | [E] Trivial predictor is **better** than the model. [I] The model adds no value over "predict the average." |
| **Median Predictor** (train median 5.0) | 6.666667 | **+0.011930** | [E] Statistically tied with the model. [I] The model's edge is ~0.18 % of MAE and reflects its constant sitting near the train median, not learning. |

---

## 7. Interpretation

**A. Is the regression model meaningfully better than a trivial predictor?**
[E] **No.** It is worse than the training-mean predictor (by 0.267 MAE) and effectively tied with the training-median predictor (by 0.012 MAE). [I] On this validation set it does not outperform the simplest possible baseline.

**B. How large is the improvement?**
[E] There is **no improvement — the sign is negative** against the most natural trivial baseline (the mean). Against the median baseline the "improvement" is 0.012 MAE (0.18 %), which is smaller than the 0.0003 MAE contributed by the model's entire prediction spread. [I] There is no measurable participant-level gain to report.

**C. Is the improvement likely to represent genuine learning?**
[E] **No.** Genuine regression learning requires predictions that *vary with the participant*; the model's predictions vary by 0.058 on a 0–23 scale (variance 3.5e-4), and a fixed constant reproduces its MAE to within 0.0003. [I] The head has learned a single number — approximately the central tendency of the training PHQ distribution — and applies it to everyone. This is the definition of a mean/constant estimator, not a model that has learned to distinguish participants.

**D. Does this finding strengthen or weaken the Phase 9.5 regression result?**
[E/I] It **weakens** the interpretation of the −24.1 % headline as evidence of learning. [E] The 113-participant run predicted a constant ≈4.30 (MAE 8.7652); the 188-participant run predicts a constant ≈5.05 (MAE 6.6547). Between the two runs, both the emitted constant **and** the validation set changed. [I] A constant predictor's MAE moves whenever its constant relocates or the evaluation set changes — exactly the two things that changed — so the −24.1 % drop is fully consistent with a constant moving nearer the new validation distribution, with **no learning required**. This is the outcome the roadmap anticipated for Experiment 0a: *"Model MAE ≈ mean-predictor MAE ⇒ head confirmed a mean-estimator and the −24.1 % claim is formally withdrawn"* (§6). The claim is hereby formally withdrawn **as evidence of learning** (the measured MAE value itself remains a valid, frozen historical figure).

**Uncertainty, stated honestly.** [E] The comparison set is n = 18 (5 positives), so the *exact* MAE gaps (e.g. the 0.267 deficit vs the mean) carry wide uncertainty and should not be over-read as a precise effect size. [I] However, conclusions A–D do **not** depend on sample size: the finding that the model is a constant predictor is an **internal property of its own 18 predictions** (spread 0.058, variance 3.5e-4, matched by a fixed constant to within 0.0003) and holds regardless of how many participants it is scored on.

---

## 8. Implications for the Version 2 Roadmap

*This section explains implications only. It does not modify the roadmap.*

The regression head behaves as a trivial predictor. Per the Phase 10 roadmap's own conditional logic, this raises the priority of the regression / loss-balancing line of work:

- **Experiment 6 — Loss-term rebalancing (`CE + 0.5·MSE`)** [E-roadmap §5 A-6, §6 #6] becomes higher-priority. The diagnostic confirms both heads are collapsed; the regression head emitting a single constant is consistent with the roadmap's hypothesis that the loss geometry lets both heads settle on their respective priors. Widening the **prediction variance** (not merely lowering MAE) is the relevant success signal — as §6 #6 and §7.4 already specify.
- **Experiment 3 — Convergence** [§6 #3] retains its place as the least-invasive first test: the constant-output behaviour is exactly what an under-trained head that has only reached the prior would produce. Whether more training alone widens the prediction spread is now a concrete, measurable question.
- **§7.4's mandatory reporting rule is reinforced by direct evidence:** every future regression claim must be reported alongside (a) the trivial mean-predictor MAE and (b) the prediction-variance statistic. This diagnostic supplies the first such numbers — **mean-predictor MAE 6.3882** and **model prediction variance 3.5e-4** — as the reference floor against which any Version 2 regression result must be judged.

No new research directions are introduced. No experiment is added, removed, reordered, or merged. The finding adjusts *priority emphasis* within the existing sequence, consistent with the roadmap's stated 0a-branch behaviour.

---

## 9. Conclusion

[E] The reconstruction reproduced the frozen model MAE (6.654737075169881) exactly, validating the 18-participant target set. On that verified set, the MentalBERT regression head:

- emits a **near-constant** ≈5.05 (prediction variance 3.5e-4; range 0.058 on a 0–23 scale);
- is **worse** than a training-mean predictor (6.6547 vs 6.3882) and **tied** with a training-median predictor (6.6547 vs 6.6667);
- is reproduced to within 0.0003 MAE by a single fixed constant.

[I] **The regression head has not learned meaningful participant-level information; it is a constant/mean estimator.** Consequently, the Phase 9.5 −24.1 % MAE result is **not** evidence of learning and is formally withdrawn as such (the numeric value remains a valid frozen record). The regression / loss-balancing experiments (notably Experiment 6, supported by Experiment 3) are correspondingly elevated in priority within the unchanged Version 2 roadmap.

**Experiment 0a outcome: the regression head is a trivial predictor. Branch A confirmed.**

---

*No code executed against the model. No retraining. No predictions regenerated. No trainer, dataset, label, or frozen report modified. Every conclusion is supported by a measured quantity, with observed evidence [E] distinguished from interpretation [I].*
