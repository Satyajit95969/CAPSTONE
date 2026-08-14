# PHASE 11.6 — BASELINE-CV
## The Official Version 2 Quantitative Comparator (Experiment 2)

**Phase:** 11.6 — Baseline re-measurement under participant-level cross-validation
**Type:** Measurement of the *unmodified* frozen trainer under a new evaluation protocol — no model improvement
**Authority:** `PHASE_10_ROADMAP.md` §6 (Exp 2), §7 · `PHASE_11_EVALUATION_PROTOCOL.md` (validated in 11.5)
**Comparator role:** Every Phase 12 experiment (Exp 3 onward) compares against this document.

Convention: **[E]** = measured evidence · **[I]** = interpretation. All model metrics below were executed on the mandated Colab GPU; no value in this document is fabricated or estimated.

---

## 0. Execution-Integrity Statement (read first)

Baseline-CV has **two components**, and both are now complete:

| Component | Needs model training? | Status |
|---|---|---|
| **Frozen fold manifest** (participant-level stratified 5-fold) | No | ✅ **Produced and frozen** — real artifact |
| **Trivial-baseline control arm** (mean/median predictor; majority-class; constant-ranker) | No | ✅ **Computed** — real numbers, all 5 folds |
| **Partition-induced noise component** | No | ✅ **Measured** — real |
| **Model per-fold metrics** (MentalBERT MAE/RMSE/ROC-AUC/PR-AUC/balAcc/MCC/P/R/F1) | **Yes** | ✅ **Measured** — Colab GPU (Tesla T4), R = 5 |
| **Training-stochasticity noise + final MDE** | **Yes (R repeats)** | ✅ **Measured** — R = 5, 25 fold-runs |

**Why the model metrics were executed on Colab GPU.** [E] The local environment is **CPU-only** (`torch 2.8.0+cpu`, `cuda_available = False`, `device_count = 0`); the frozen Phase 9.5 baseline was trained on **Colab GPU**. [I] Running Baseline-CV on CPU would have changed **two** factors relative to Phase 9.5 — the evaluation protocol *and* the compute device — which is precisely the two-factor confound (§6 of the roadmap: "Phase 9.5 changed two factors at once and cannot attribute its own headline result"). Baseline-CV must be a **one-factor** change (evaluation protocol only), so the device was held at **Colab GPU**. The validated protocol makes this mandatory (Phase 11.4 §5.1; roadmap Phase 11 planning). [E] The driver enforces this: `run_baseline_cv.py` aborts on CPU unless explicitly overridden. The execution used the **frozen, unmodified** trainer (SHA verified at startup); the as-executed record is §9.

Every number below is real, computed evidence.

---

## 1. Methodology

Baseline-CV re-measures the **original trainer, unchanged**, under the Version 2 evaluation protocol. The **only** change from Phase 9.5 is the evaluation methodology: a single seed-42 90/10 split is replaced by participant-level stratified 5-fold cross-validation, repeated, with the metric suite of `PHASE_11_EVALUATION_PROTOCOL.md`.

Held constant (verified in Phase 11.0): trainer SHA `65b1902e…a230b`, architecture `MultiModalModel`, loss `CE + 0.5·MSE`, optimizer, epochs = 3, batch = 8, lr = 2e-5, binarize threshold 10.0, backbone `mental/mental-bert-base-uncased`, dataset (188 / 45 / 143), labels, preprocessing. No hyperparameter, threshold, optimizer, loss, data, or preprocessing change.

---

## 2. Cross-Validation Design

| Design element | Choice | Justification |
|---|---|---|
| Split unit | **Participant** | [E] 188 rows = 188 unique participant_ids (one row per participant), so record-level folds *are* participant-level folds — independence (P-4) is structural, zero leakage possible. |
| Scheme | **Stratified k-fold**, `shuffle=True`, `random_state=42` | Preserves the 23.9% positive rate in every fold. |
| **k = 5** | ≈9 positives / held-out fold | [E] Only 45 positives exist; k = 5 yields the largest fold count that keeps ≥9 positives per test fold, avoiding the ~4-positive fragility a larger k reintroduces (Phase 11.4 §7). |
| Repeats **R** | **R = 5** (as executed) | Characterizes training stochasticity (torch unseeded, per protocol P-3 as clarified in 11.5). Recommended R ≥ 5. |
| Manifest | **Frozen & versioned** | `trainer_outputs/baseline_cv/fold_manifest.json` — every future experiment reuses these exact folds, enabling **paired** comparisons (§8). |

**Frozen manifest fold composition [E]:**

| Fold | n_test | n_pos | prevalence | train_mean(PHQ) | train_median(PHQ) |
|---|---|---|---|---|---|
| 1 | 38 | 9 | 0.237 | 6.680 | 5.0 |
| 2 | 38 | 9 | 0.237 | 6.400 | 4.5 |
| 3 | 38 | 9 | 0.237 | 6.627 | 5.0 |
| 4 | 37 | 9 | 0.243 | 6.735 | 5.0 |
| 5 | 37 | 9 | 0.243 | 6.960 | 6.0 |

Artifact: `trainer_outputs/baseline_cv/fold_manifest.json` (full train/test participant-ID lists per fold).

---

## 3. Fold-by-Fold Results

### 3.1 Trivial-baseline control arm — ✅ REAL (no training required)

These are the permanent control floors the protocol (P-6) requires to ship with Baseline-CV. Computed on each held-out fold.

**Regression (constant predictors; prediction variance = 0 by construction):**

| Fold | MAE (mean-pred) | RMSE (mean-pred) | MAE (median-pred) | RMSE (median-pred) |
|---|---|---|---|---|
| 1 | 4.8084 | 5.5824 | 4.6842 | 5.8310 |
| 2 | 4.2842 | 5.5812 | 4.8947 | 6.3277 |
| 3 | 5.2961 | 6.2960 | 5.2632 | 6.5695 |
| 4 | 4.9515 | 5.8633 | 4.8649 | 6.0359 |
| 5 | 5.2271 | 6.0533 | 4.8378 | 5.9024 |

**Classification (majority-class predictor + constant-score ranker):**

| Fold | ROC-AUC | PR-AUC | Balanced Acc | MCC | Precision | Recall | F1 | Accuracy |
|---|---|---|---|---|---|---|---|---|
| 1 | 0.500 | 0.2368 | 0.500 | 0.0 | 0.0 | 0.0 | 0.0 | 0.7632 |
| 2 | 0.500 | 0.2368 | 0.500 | 0.0 | 0.0 | 0.0 | 0.0 | 0.7632 |
| 3 | 0.500 | 0.2368 | 0.500 | 0.0 | 0.0 | 0.0 | 0.0 | 0.7632 |
| 4 | 0.500 | 0.2432 | 0.500 | 0.0 | 0.0 | 0.0 | 0.0 | 0.7568 |
| 5 | 0.500 | 0.2432 | 0.500 | 0.0 | 0.0 | 0.0 | 0.0 | 0.7568 |

Artifact: `trainer_outputs/baseline_cv/trivial_control_arm.csv`.

### 3.2 Model (MentalBERT) per-fold results — ✅ REAL

Repeats averaged within each fold (the fold is the unit of independence). Source: `trainer_outputs/baseline_cv/runs/rep{1..5}_metrics.csv`.

| Fold | MAE | RMSE | Pred.Var | ROC-AUC | PR-AUC | balAcc | MCC | P | R | F1 |
|---|---|---|---|---|---|---|---|---|---|---|
| 1 | 4.7143 | 5.9214 | 0.0003 | 0.6050 | 0.3456 | 0.5000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 |
| 2 | 4.7824 | 6.1968 | 0.0008 | 0.6483 | 0.4819 | 0.5000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 |
| 3 | 5.2978 | 6.7871 | 0.0003 | 0.7027 | 0.4606 | 0.5000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 |
| 4 | 4.8791 | 6.0930 | 0.0183 | 0.5802 | 0.3003 | 0.5000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 |
| 5 | 4.3901 | 5.9495 | 0.0090 | 0.6306 | 0.3819 | 0.5000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 |

Artifacts: `runs/rep{r}_metrics.csv` (5) and `runs/rep{r}_fold{f}_preds.csv` (25 held-out prediction files).

---

## 4. Aggregate Metrics

### 4.1 Control arm — ✅ REAL (mean ± SD across 5 folds; 95% CI ≈ 1.96·SE, k = 5 illustrative)

| Metric | Mean | SD | 95% CI |
|---|---|---|---|
| MAE (mean-predictor) | **4.9135** | 0.4041 | [4.559, 5.268] |
| RMSE (mean-predictor) | 5.8752 | 0.3087 | [5.605, 6.146] |
| MAE (median-predictor) | 4.9090 | 0.2140 | [4.721, 5.097] |
| RMSE (median-predictor) | 6.1333 | 0.3091 | [5.862, 6.404] |
| ROC-AUC (constant) | 0.5000 | 0.0000 | [0.500, 0.500] |
| PR-AUC (constant) | 0.2394 | 0.0035 | [0.236, 0.242] |
| Balanced Acc (majority) | 0.5000 | 0.0000 | [0.500, 0.500] |
| MCC (majority) | 0.0000 | 0.0000 | [0.000, 0.000] |
| Accuracy (majority) | 0.7606 | 0.0035 | [0.758, 0.764] |

*(k = 5 CIs are wide and illustrative; they exist to convey uncertainty, not precision.)*

### 4.2 Model aggregate — ✅ REAL (mean ± fold SD; 95% CI = fold-level Student-t, df = k−1 = 4, t = 2.7764)

| Metric | Mean | Fold SD | 95% CI | Partition SD | Training-stoch SD |
|---|---|---|---|---|---|
| **MAE** | **4.8127** | 0.3275 | [4.4061, 5.2193] | 0.3275 | 0.0387 |
| RMSE | 6.1896 | 0.3521 | [5.7524, 6.6268] | 0.3521 | 0.0738 |
| **Prediction variance** | **0.005734** | 0.007911 | [−0.004089, 0.015557] | 0.007911 | 0.003329 |
| MAE (mean-predictor, in-run control) | 4.9135 | 0.4041 | [4.4117, 5.4153] | 0.4041 | 0.0000 |
| **ROC-AUC** | **0.6333** | 0.0466 | [0.5755, 0.6911] | 0.0466 | 0.0428 |
| **PR-AUC** | **0.3941** | 0.0765 | [0.2990, 0.4891] | 0.0765 | 0.0525 |
| Balanced Acc | 0.5000 | 0.0000 | [0.5000, 0.5000] | 0.0000 | 0.0000 |
| MCC | 0.0000 | 0.0000 | [0.0000, 0.0000] | 0.0000 | 0.0000 |
| Precision | 0.0000 | 0.0000 | [0.0000, 0.0000] | 0.0000 | 0.0000 |
| Recall | 0.0000 | 0.0000 | [0.0000, 0.0000] | 0.0000 | 0.0000 |
| F1 | 0.0000 | 0.0000 | [0.0000, 0.0000] | 0.0000 | 0.0000 |
| Accuracy | 0.7606 | 0.0035 | [0.7562, 0.7650] | 0.0035 | 0.0000 |

*(R = 5 repeats × k = 5 folds = 25 fold-runs. The in-run `MAE_mean_pred` reproduces the frozen control-arm value 4.9135 of §4.1, independently confirming the fold partition and label handling.)*

---

## 5. Noise-Band Estimation

The total noise band of a metric has two sources; only one is measurable without training:

| Noise source | Measurable now? | Value |
|---|---|---|
| **Partition-induced** (metric moves with fold composition) | ✅ Yes (via control arm) | [E] Regression: **SD ≈ 0.40 MAE**, 0.31 RMSE across folds. Classification trivial metrics: SD ≈ 0 for ROC-AUC/balAcc/MCC, ≈ 0.004 for PR-AUC (tracks prevalence). |
| **Training-stochasticity** (run-to-run drift across R repeats) | ✅ Yes (R = 5) | [E] MAE **0.0387** · RMSE 0.0738 · pred_var 0.0033 · ROC-AUC **0.0428** · PR-AUC **0.0525** · balAcc / MCC / P / R / F1 / Accuracy **0.0000**. |

[E] **Measured partition floor:** the mean-predictor MAE varies by **SD 0.40** purely from which participants land in a fold — with *no model involved*. [I] This is a genuine lower bound on the regression noise band: no Version 2 regression effect smaller than the partition variability can be claimed from a single-run comparison. [E] **Both components are now measured.** For regression, partition (0.3275 MAE) dominates training-stochasticity (0.0387) by roughly 8:1. For the ranking metrics the two are comparable (ROC-AUC 0.0466 vs 0.0428; PR-AUC 0.0765 vs 0.0525), so classification comparisons must budget for both. [E] The threshold-conditional metrics (balAcc / MCC / P / R / F1) have **zero** variance from either source — they are degenerate-constant, and carry no model variance to measure.

---

## 6. Minimum Detectable Effect (MDE)

**Methodology (as executed):**
1. Run the frozen model over the 5 frozen folds × R repeats → per-fold, per-repeat metrics.
2. Because every future experiment reuses the **same frozen manifest**, comparisons are **paired per fold**. Compute the per-fold difference `Δ_f = V2_f − BaselineCV_f`; the relevant statistic is the paired mean `Δ̄` with standard error `SE = SD(Δ_f)/√k`.
3. **MDE ≈ t₀.₉₅,df·SE_total** (one-sided) — the smallest `Δ̄` that clears the noise band. A change with `|Δ̄| < MDE` is reported as **"no detectable effect."**
4. The paired design is deliberately more powerful than comparing two independent aggregates, and is the reason the manifest is frozen.

**Final MDE — ✅ REAL** (paired planning estimate, 95%, Student-t, df = k−1 = 4):

| Metric | Paired SE | t(0.975, 4) | **MDE (95%)** |
|---|---|---|---|
| MAE | 0.1464 | 2.776 | **0.4066** |
| RMSE | 0.1575 | 2.776 | **0.4372** |
| ROC-AUC | 0.0208 | 2.776 | **0.0578** |
| PR-AUC | 0.0342 | 2.776 | **0.0950** |
| balAcc | 0.0000 | 2.776 | 0.0000 |
| MCC | 0.0000 | 2.776 | 0.0000 |
| F1 | 0.0000 | 2.776 | 0.0000 |

[I] A Version 2 change smaller than the MDE for its metric is reported as **"no detectable effect."** [I] The zero MDE on balAcc / MCC / F1 is an artifact of their degeneracy at the frozen operating point, **not** a claim that any change is detectable — those metrics acquire a real noise band only once a configuration produces non-constant predictions.

---

## 7. Official Version 2 Comparator

**This is the table Phase 12 experiments compare against.** Both columns are real, measured evidence.

| Metric | Trivial control (REAL, mean±SD) | Baseline-CV model (mean±SD) | Operative floor for Exp 3+ |
|---|---|---|---|
| **Regression** | | | |
| MAE | mean-pred **4.9135 ± 0.404** | **4.8127 ± 0.327** | must beat control **and** exceed MDE 0.4066 below Baseline-CV |
| RMSE | mean-pred 5.8752 ± 0.309 | 6.1896 ± 0.352 | MDE 0.4372 |
| Prediction variance | 0 (constant) | **0.005734 ± 0.007911** | must be **materially > 0** (anti-collapse) |
| **Classification** | | | |
| ROC-AUC | **0.500 ± 0.000** | **0.6333 ± 0.0466** | must exceed Baseline-CV by > MDE 0.0578 |
| PR-AUC | **0.2394 ± 0.004** (= prevalence) | **0.3941 ± 0.0765** | must exceed Baseline-CV by > MDE 0.0950 |
| Balanced Acc | 0.500 ± 0.000 | 0.5000 ± 0.0000 | must exceed 0.5 |
| MCC | 0.000 ± 0.000 | 0.0000 ± 0.0000 | must exceed 0 |
| P / R / F1 @ declared op-point | 0 / 0 / 0 | 0.0000 / 0.0000 / 0.0000 | reported threshold-conditionally |
| Accuracy | 0.7606 ± 0.004 | 0.7606 ± 0.0035 | **not** a decision metric (§3.3) |

[I] **The control arm already encodes the acceptance geometry:** any Version 2 classifier must clear ROC-AUC 0.5, PR-AUC 0.239, balanced accuracy 0.5, and MCC 0 by more than the measured noise band (§6 MDE); any Version 2 regressor must beat mean-predictor MAE ≈ 4.91 **and** produce non-zero prediction variance. Beating accuracy 0.76 is explicitly *not* a success (it is the majority rate).

---

## 8. Scientific Interpretation

**Stability across folds [E].** The control arm is highly stable: classification trivial metrics are essentially invariant across folds (SD ≈ 0), and regression trivial MAE varies by SD 0.40 — modest, and driven purely by fold composition. [I] The stratified manifest is well-balanced (every fold 23.7–24.3% positive), so fold-composition noise is controlled but non-zero for regression.

**Generalization [I].** Because folds are participant-disjoint and stratified, the model aggregate estimates generalization far more honestly than the single 18-sample split ever could. Whether the model's prior-collapse (established in 11.1–11.3) persists under CV is the central hypothesis this comparator tested at power.

**Remaining limitations [E/I].** (a) The **model metrics are measured** over R = 5 repeats × k = 5 folds (§4.2). (b) k = 5 CIs are wide (illustrative). (c) With 45 positives total, classification metrics remain the higher-variance side and R repeats are needed to bound them. (d) The trivial classification controls are degenerate by construction and therefore contribute no model-variance information.

**Relationship to the frozen historical baseline [I].** Per protocol §5.1 and the roadmap §7.3, the frozen 18-sample figures (MAE 6.6547, acc 0.7222, P/R/F1 0/0/0) are a **historical record, not a numeric comparator** for these cross-validated numbers — they inhabit a different evaluation universe (a single 90/10 split vs 5-fold CV, different train-set sizes and central tendencies). They are cited here **only as historical context** and are deliberately **not** differenced against the CV control arm. The Baseline-CV *model* aggregate (once produced) — not the frozen 6.6547 — becomes the sole quantitative comparator for Version 2.

---

## 9. Execution Record for the Model Runs (as executed)

*Ran the **frozen, unmodified** trainer. Changed only data-in and predictions-out (Phase 11.5 architectural constraint). No trainer/model/loss/hyperparameter edit.*

For each fold f ∈ {1..5} and repeat r ∈ {1..5}, on **Colab GPU (Tesla T4)**:
1. From `fold_manifest.json`, materialize `train_fold_f.parquet` (train_ids) and `test_fold_f.parquet` (test_ids) as **subsets** of `daic_records.parquet` (no data modification — pure selection).
2. **Train** with the exact frozen command:
   `python trainer_mentalbert_daic.py --input train_fold_f.parquet --mode supervised --epochs 3 --batch-size 8 --lr 2e-5 --binarize --binarize-threshold 10.0 --out-path .../fold_f_rep_r.pt` (device auto-selects GPU).
3. **Evaluate on the held-out fold** by running the trainer's own `MultiModalModel` + `run_inference` over `test_fold_f.parquet` using the fold checkpoint — the trainer's existing inference path, no logic change — to obtain `pred_phq` and `pred_class_probs` for the test participants.
4. Compute the §2 metric suite (via the same sklearn calls used in 11.1/11.2) on each held-out fold; store per-fold, per-repeat.
5. Aggregate → fill §3.2, §4.2; compute training-stochasticity SD → complete §5; finalize MDE → §6; fill §7.

**Reproducibility record:** frozen manifest `trainer_outputs/baseline_cv/fold_manifest.json` (188 participants, 5 folds) · trainer SHA `65b1902e…a230b` (verified at driver startup on every invocation) · `PYTHONPATH=<repo root> python colab_baseline_cv/run_baseline_cv.py --repeat r` for r = 1…5, all other parameters at frozen defaults (epochs 3, batch 8, lr 2e-5, threshold 10.0, base-seed 1000) · R = 5 · per-repeat seed 1000+r, per-fold seed seed×100+f · GPU Tesla T4 · `transformers==4.44.0` · backbone `mental/mental-bert-base-uncased` (vocab 30522) · total compute 419.9 s across 25 fold-runs · 25 per-fold prediction CSVs retained.

---

## 10. Conclusion

[E] The training-free foundation of Baseline-CV is **complete and real**: a frozen, participant-level, stratified 5-fold manifest (well-balanced, zero-leakage-by-construction), a full trivial-baseline control arm (regression mean-predictor MAE **4.9135 ± 0.404**; classification ROC-AUC **0.5**, PR-AUC **0.239**, balanced accuracy **0.5**, MCC **0**), and a measured partition-noise floor (regression SD ≈ 0.40 MAE).

[E] The **model** side is now **complete**: executed on Colab GPU (Tesla T4) with R = 5 repeats over the frozen folds, using the SHA-verified frozen trainer. The model aggregate (§4.2), the two-component noise band (§5) and the final MDE (§6) complete the official comparator. Baseline-CV reproduces the frozen baseline's qualitative signature exactly — **P / R / F1 = 0.0000 and all-negative predictions on every fold and every repeat** — which is the Exp-2 success criterion of roadmap §6 and confirms the re-measurement is faithful rather than faulty.

**Phase 11.6 outcome: COMPLETE. Fold manifest frozen; trivial control arm, model comparator, two-component noise band and final MDE all measured (real). The frozen trainer is unmodified; the only change from Phase 9.5 is the evaluation methodology.**

---

*Frozen trainer only — not modified. No hyperparameter, threshold, optimizer, loss, data, or preprocessing change. No roadmap modification. Real evidence [E] is separated from interpretation [I]; every model metric is measured, none fabricated. Artifacts: `trainer_outputs/baseline_cv/fold_manifest.json`, `trivial_control_arm.csv`, `baseline_cv_summary.json`, `baseline_cv_summary.md`, `runs/rep{1..5}_metrics.csv`, `runs/rep{r}_fold{f}_preds.csv` (25), and this report.*
