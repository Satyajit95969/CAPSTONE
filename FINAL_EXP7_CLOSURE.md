# FINAL EXPERIMENT 7 CLOSURE

**Status: CLOSED.** Audit verdict: **CLEAN — EXP 7 CAN BE CLOSED**.

This document is derived solely from the completed Experiment 7 artifacts
(`exp7_results.zip`, SHA-256 `b0d5f303145e8461ff18d0f9f43f4c8da8833748de13af5a9a2a64b0891233d7`)
and the final read-only audit. No result has been modified, re-run or reinterpreted.

---

## 1. Experiment identification

| Field | Value |
|---|---|
| Experiment | Phase 16 / Experiment 7 — multimodal activation |
| Single changed factor | **input modality** |
| Control arm | A0 |
| Primary arm | A3 |
| Run completed (UTC) | 2026-08-09T13:11:54Z |
| Receipt schema | `exp7_receipt/1`, version 1 |
| Result bundle | `exp7_results.zip`, 166.1 KB, 142 entries |

## 2. Objective

To exercise the multimodal architecture for the first time in the project's history,
and to test whether adding audio and vision to the text-only model (a) reaches the
computation graph at all, and (b) improves predictive performance.

Prior to this experiment the modality ablation had returned **exactly 0.0** for both
audio and vision in every phase — absence from the computation graph, not weak
contribution.

## 3. Experimental design

One factor, four pre-registered arms, everything else frozen. 5×5 repeated
participant-level cross-validation on the frozen fold manifest, 25 fold-runs per arm,
**100 fold-runs total**. Arms differ only in the `(audio_dim, vision_dim)` pair passed
to the frozen `MultiModalModel`; a single code path executes for all arms because
`forward()` ignores modality vectors whose encoder was not constructed.

A0 is a **hard gate**: it is text-only on the *multimodal* artifact, so it isolates the
single question of whether swapping the artifact changed the text-only result. Had A0
failed, no A1/A2/A3 result could have been reported.

## 4. A0–A3 arm definitions

| Arm | Modality | audio_dim | vision_dim | Fusion input | Parameters | Role |
|---|---|---|---|---|---|---|
| **A0** | text only | None (0) | None (0) | **768** | 109,680,132 | HARD GATE / control |
| **A1** | text + audio | **154** | None (0) | **896** | 109,723,782 | audio contribution |
| **A2** | text + vision | None (0) | **84** | **896** | 109,719,844 | vision contribution |
| **A3** | text + audio + vision | **154** | **84** | **1024** | 109,763,494 | full multimodal, PRIMARY |

Verified per fold-run in `exp7_metrics.csv` and independently in `modality_config.csv`
(including `has_audio` / `has_vision` flags).

## 5. Dataset / sample information

| Field | Value |
|---|---|
| Corpus | DAIC-WOZ, 188 participants |
| Positives / negatives | 45 / 143 |
| Prevalence | 0.2393617 (PHQ-8 > 10) |
| Per-fold test sizes | 38, 38, 38, 37, 37 — sums to 188 |
| Positives per fold | 9, 9, 9, 9, 9 |
| Text artifact | `daic_records.parquet` (frozen) |
| Multimodal artifact | `dataset_build/daic_records_multimodal.parquet` (154-d audio, 84-d vision) |
| Frozen columns | `participant_id`, `text`, `phq_score` byte-identical between the two artifacts |

## 6. Seeds and CV protocol

| Field | Value |
|---|---|
| Folds / repeats | 5 / 5 → 25 fold-runs per arm, 100 total |
| BASE_SEED | 1000 |
| Stage-1 seeding (per repeat) | `seed = BASE_SEED + repeat` |
| Seeds observed | 1001, 1002, 1003, 1004, 1005 |
| Stage-2 seeding (per fold) | `torch.manual_seed(seed * 100 + fold)` |
| Construction order (normative) | seed → model → dataset → dataloader → loop |
| Fold source | frozen `fold_manifest.json`, participant-level stratified |
| CI method | fold-level Student-t (df = k−1 = 4), repeats averaged within fold |
| t_crit | 2.7764451051977987 |
| MDE anchors | MAE 0.4066, PR_AUC 0.0950 |

## 7. Hyperparameters (all frozen, identical to Baseline-CV and Exps 3–6)

| Parameter | Value |
|---|---|
| EPOCHS | 3 |
| LR | 2e-05 |
| BATCH_SIZE | 8 |
| MAX_LEN | 128 |
| LAMBDA (regression coefficient) | 0.5 |
| GRAD_CLIP | 1.0 |
| BINARIZE_THRESHOLD | 10.0 |
| Optimizer | `transformers.optimization.AdamW` |
| Encoder | `mental/mental-bert-base-uncased` |

## 8. C0–C4 results

| ID | Result | Role | Statement |
|---|---|---|---|
| **C0** | **PASS** | HARD GATE | A0 reproduces Baseline-CV on all primary metrics (artifact swap is inert) |
| **C1** | **PASS** | activation | audio ablation non-zero |
| **C2** | **PASS** | activation | vision ablation non-zero |
| **C3** | **PASS** (major caveat, §16) | guard | text contribution remains positive |
| **C4** | **FAIL** | utility | A3 PR-AUC improves over A0 beyond the MDE (0.0950) |

Recorded verdict (verbatim):

> `ACTIVATION DEMONSTRATED - audio and vision reach the graph with the text-only control intact`

## 9. A0 equivalence result

**PASSED — authoritative, 12/12 metrics, zero failures.** A0 completed all 25 fold-runs
(folds 1–5 × repeats 1–5, each cell exactly once).

| Metric | A0 mean | Baseline mean | Baseline 95 % CI | Inside |
|---|---|---|---|---|
| MAE | 4.812728 | 4.812728 | [4.406119, 5.219336] | True |
| RMSE | 6.189593 | 6.189593 | [5.752370, 6.626815] | True |
| pred_var | 0.005734 | 0.005734 | [−0.004089, 0.015557] | True |
| MAE_mean_pred | 4.913480 | 4.913480 | [4.411692, 5.415269] | True |
| ROC_AUC | 0.633331 | 0.633331 | [0.575518, 0.691143] | True |
| PR_AUC | 0.394064 | 0.394064 | [0.299045, 0.489082] | True |
| balAcc | 0.500000 | 0.500000 | [0.500000, 0.500000] | True |
| MCC | 0.000000 | 0.000000 | [0.000000, 0.000000] | True |
| Precision | 0.000000 | 0.000000 | [0.000000, 0.000000] | True |
| Recall | 0.000000 | 0.000000 | [0.000000, 0.000000] | True |
| F1 | 0.000000 | 0.000000 | [0.000000, 0.000000] | True |
| Accuracy | 0.760597 | 0.760597 | [0.756244, 0.764951] | True |

**A0 did not merely fall inside the confidence interval — it reproduced Baseline-CV's
mean to every recorded digit on all twelve metrics.** The artifact swap is demonstrably
inert and the one-factor design holds.

## 10. Full multimodal performance summary

Fold-level means with 95 % CI (k = 5):

| Metric | A0 | A1 | A2 | A3 |
|---|---|---|---|---|
| MAE ↓ | 4.8127 [4.406, 5.219] | 4.8434 [4.459, 5.228] | 4.8570 [4.532, 5.182] | 4.8567 [4.513, 5.201] |
| RMSE ↓ | 6.1896 [5.752, 6.627] | 5.9943 [5.599, 6.390] | 5.9477 [5.632, 6.263] | 6.0094 [5.695, 6.324] |
| ROC_AUC ↑ | 0.6333 [0.576, 0.691] | 0.4717 [0.427, 0.516] | 0.5825 [0.493, 0.672] | 0.4939 [0.433, 0.555] |
| PR_AUC ↑ | 0.3941 [0.299, 0.489] | 0.2938 [0.236, 0.351] | 0.3619 [0.265, 0.459] | 0.2961 [0.250, 0.342] |
| Recall ↑ | 0.0000 | 0.0489 [−0.047, 0.145] | 0.0000 | 0.0000 |
| F1 ↑ | 0.0000 | 0.0249 [−0.020, 0.070] | 0.0000 | 0.0000 |
| Precision ↑ | 0.0000 | 0.0170 [−0.013, 0.047] | 0.0000 | 0.0000 |
| Accuracy ↑ | 0.7606 [0.756, 0.765] | 0.7318 [0.682, 0.782] | 0.7552 [0.737, 0.773] | 0.7606 [0.756, 0.765] |
| balAcc ↑ | 0.5000 | 0.4979 [0.484, 0.511] | 0.4964 [0.487, 0.506] | 0.5000 |
| MCC ↑ | 0.0000 | −0.0027 [−0.028, 0.022] | −0.0090 [−0.034, 0.016] | 0.0000 |
| pred_var | 0.0057 [−0.004, 0.016] | 0.0737 [0.050, 0.098] | 0.0946 [0.042, 0.147] | 0.0949 [0.055, 0.135] |
| MAE_mean_pred | 4.9135 [4.412, 5.415] | 4.9135 | 4.9135 | 4.9135 |

↑ higher is better; ↓ lower is better.

Classification remained **degenerate** (Recall = F1 = Precision = 0) in A0, A2 and A3.
Only A1 produced any non-zero Recall (0.0489) and F1 (0.0249), neither significant.

## 11. A1 vs A0 findings (text + audio)

| Metric | Better | Mean Δ | 95 % CI | Significant | Exceeds MDE |
|---|---|---|---|---|---|
| PR_AUC | higher | −0.100252 | [−0.246263, +0.045759] | No | True |
| ROC_AUC | higher | **−0.161612** | [−0.219103, −0.104121] | **Yes (degradation)** | — |
| RMSE | lower | **−0.195273** | [−0.385706, −0.004841] | **Yes (improvement)** | — |
| MAE | lower | +0.030638 | [−0.268375, +0.329650] | No | False |
| pred_var | — | **+0.067941** | [+0.048019, +0.087862] | **Yes** | — |

Audio significantly **degraded** ranking (ROC-AUC) while significantly **improving**
RMSE and widening prediction variance.

## 12. A2 vs A0 findings (text + vision)

| Metric | Better | Mean Δ | 95 % CI | Significant | Exceeds MDE |
|---|---|---|---|---|---|
| PR_AUC | higher | −0.032196 | [−0.184700, +0.120307] | No | False |
| ROC_AUC | higher | −0.050851 | [−0.158225, +0.056523] | No | — |
| RMSE | lower | **−0.241859** | [−0.478610, −0.005107] | **Yes (improvement)** | — |
| MAE | lower | +0.044234 | [−0.330997, +0.419464] | No | False |
| pred_var | — | **+0.088912** | [+0.042786, +0.135037] | **Yes** | — |

Vision produced the largest RMSE improvement and the least ranking damage, but no
significant ranking gain.

## 13. A3 vs A0 findings (full multimodal)

| Metric | Better | Mean Δ | 95 % CI | Significant | Exceeds MDE |
|---|---|---|---|---|---|
| PR_AUC | higher | **−0.097949** | [−0.211979, +0.016081] | No | True |
| ROC_AUC | higher | **−0.139444** | [−0.175209, −0.103680] | **Yes (degradation)** | — |
| RMSE | lower | −0.180196 | [−0.365197, +0.004805] | No | — |
| MAE | lower | +0.044019 | [−0.305337, +0.393376] | No | False |
| pred_var | — | **+0.089146** | [+0.047930, +0.130362] | **Yes** | — |

Additional pairwise comparisons — **no significant differences in either**:

- **A3 − A1**: PR_AUC +0.002303 [−0.038185, +0.042790]; ROC_AUC +0.022167; RMSE +0.015077
- **A3 − A2**: PR_AUC −0.065753 [−0.195412, +0.063907]; ROC_AUC −0.088593; RMSE +0.061663

Combining both modalities produced no detectable benefit over either alone.

## 14. C4 failure and exact reason

```
A3 − A0  PR_AUC
  A3 mean         :  0.296114
  A0 mean         :  0.394064
  mean_delta      : -0.097949          <- NEGATIVE (degradation)
  fold_sd / se    :  0.091837 / 0.041071
  95% CI          : [-0.211979, +0.016081]   k = 5
  significant     :  False  (CI includes zero)
  MDE (PR_AUC)    :  0.0950
  exceeds_mde     :  True   (|-0.097949| > 0.0950)
  per-fold deltas : [-0.064656, -0.214329, -0.124688, +0.035505, -0.121577]
```

**C4 failed because A3 PR-AUC decreased relative to A0 by 0.097949; the point estimate exceeded the MDE in magnitude but in the unfavorable direction, and the 95% CI included zero.**

The criterion requires `exceeds_mde AND mean_delta > 0`. The magnitude cleared the MDE
but the sign is negative, so the conjunction is False. Four of five folds moved negative.
This is **not** a case of "the improvement was too small to detect": the point estimate
indicates degradation of nearly one full MDE, which the fold-level variance cannot
resolve in either direction.

## 15. Ablation findings

Produced by the frozen trainer invoked through its **own command-line entry point** in a
subprocess — no monkey-patching, no injected globals, no trainer modification
(`monkey_patched: false`, `trainer_modified: false`; trainer SHA `65b1902e…` unchanged).

| Run | audio_score | vision_score | text_score |
|---|---|---|---|
| Text-only control | 0.0000000 | 0.0000000 | **2.2707760** |
| **Multimodal** | **2.5518971** | **0.4041132** | **0.0215758** |
| Phase 9.5 (historical) | 0.0 | 0.0 | 2.3365455 |

- **Audio became non-zero**: 0.0 → 2.5518971 (C1 PASS)
- **Vision became non-zero**: 0.0 → 0.4041132 (C2 PASS)
- **Text remained positive**: 0.0215758 > 0 (C3 PASS)
- The text-only control reproduces Phase 9.5 (2.2708 vs 2.3365, 2.8 % apart), validating
  the ablation harness.

Trainer logs confirm identical execution semantics to Phase 9.5:
`inferred dims audio_dim=154 vision_dim=84`, `train=170 val=18`.

## 16. Text-contribution reduction caveat (99.05 %)

**The text ablation score fell from 2.2707760 (text-only control) to 0.0215758
(multimodal): a reduction of 2.2492002, or 99.05 %.** Measured against the Phase 9.5
reference of 2.3365455 the reduction is 99.08 %. Component-wise, `text_posdelta` fell
0.22722 → 0.029627 and `text_phqdelta` fell 4.314332 → 0.013525.

Three statements must be kept strictly separate:

1. **C3 passes on its literal wording** — the criterion is "text contribution remains
   positive", and 0.0215758 > 0.
2. **A ~99 % reduction is a substantial adverse caveat.** The roadmap anticipated exactly
   this failure mode: *"a collapse in text would mean fusion is discarding language, not
   augmenting it."* On this evidence the fused model's output is dominated by audio
   (2.5519) with language contributing almost nothing.
3. **A non-zero ablation score demonstrates graph presence/contribution, not predictive
   usefulness.** Predictive usefulness is decided by the cross-validated arms, where C4
   failed.

## 17. Activation conclusion

**PASS.** For the first time in the project's history the multimodal architecture is
exercised. Audio and vision moved from exactly 0.0 — their value in every prior phase —
to 2.5518971 and 0.4041132. Encoder construction was confirmed per fold-run (fusion
widths 896/896/1024 against A0's 768, with matching parameter counts), and the text-only
control reproduced Baseline-CV exactly. **The modality pathway is real and live.**

## 18. Predictive-utility conclusion

**NOT DEMONSTRATED.** C4 failed. No arm significantly improved any ranking metric over
A0. ROC-AUC significantly **degraded** in A1 (−0.1616) and A3 (−0.1394). PR-AUC point
estimates fell in all three multimodal arms. A3 showed no detectable benefit over either
A1 or A2. Classification remained degenerate (Recall = F1 = 0) in A0, A2 and A3.

The only significant favourable movements were RMSE reductions in A1 (−0.1953) and A2
(−0.2419), and an increase in prediction variance across all multimodal arms
(0.0057 → 0.074–0.095) indicating the regression head escaped its near-constant output —
a change in behaviour, not a demonstration of better discrimination.

## 19. Integrity / SHA status

Frozen input SHAs recorded in `exp7_receipt.json`:

```
65b1902e2ba8dd996c6960db1a6595d84fd48500f4d8707cb79688093d7a230b  trainer_mentalbert_daic.py
9a241851760727779fcaa4d50041b8bdc4406fe6b66ddbbd7105f4ce4fc55f00  daic_records.parquet
1ac9f53e6102ec0dbaab84dcfdfa3f2e70f2b4a1867a841ea2b7e3ba24c67a95  dataset_build/daic_records_multimodal.parquet
b9a7a91f1bdd43c78d3cd1ee0c36e4ce3fb8be4b0971dcff3ff62ee5af8fca2f  fold_manifest.json
f065f5e1ff7f8e5ed4d122a8031c9cc12425802e756697d5512a915674871aac  baseline_cv_summary.json
2ecbfee3225910034a959fe949c7ad027cfa7d41e9bef8040374578ff8c5d572  trivial_control_arm.csv
9dd135b6b79af06a85e64a5aa8c1896d19df8059dd89c053e50e1cffe20af2f9  exp3_summary.json
5defdae2a20d0abc164611e8cbe6b8034e3f65c593d8c9b3e38266a1800aa6f2  exp4_summary.json
70ad13ffc4db633419595761c0396fc20dd1b62400ee74238be1f06b73ed48d3  exp5_summary.json
bea7478594f6af98943ddda2e997e8bc8780abf57c9c2ee720f9426cd2acc94c  exp6_summary.json
```

**None of these ten files is contained in `exp7_results.zip`.** For all ten:
**"SHA recorded in receipt but independently unverifiable from the supplied result ZIP."**
A separate, weaker cross-check against the local repository copies found all ten matching;
that is not independent verification from the result bundle.

**Output artifacts, which ARE in the bundle, were independently re-hashed: 6/6 MATCH** —
`exp7_summary.json`, `exp7_metrics.csv`, `exp7_delta.json`, `a0_equivalence_check.json`,
`modality_config.csv`, `exp7_modality_ablation.json`. (`exp7_receipt.json` cannot hash
itself.)

**Consistency audit: NO CONTRADICTIONS.** All summary aggregates and all pairwise deltas
were independently recomputed from the raw 100 fold-runs with **maximum difference
0.000e+00**. Arm counts, seeds, runtime, modality dimensions, gate status and C4 agree
across every artifact. The 20 per-repeat CSVs sum to the 100-row consolidated file, and
100 per-fold prediction files are present.

## 20. Runtime / GPU information

| Field | Value |
|---|---|
| GPU | Tesla T4 (capability 7.5, 15,637,086,208 B ≈ 14.6 GB), 1 device |
| CUDA | available, version 12.8 |
| PyTorch | 2.11.0+cu128 |
| Python | 3.12.13 |
| Platform | Linux-6.6.122+-x86_64-with-glibc2.35 |
| CV training runtime | **1612.4 s (26.9 min)** — matches Σ`seconds` in `exp7_metrics.csv` exactly |
| Ablation runtime | **78.0 s** |
| Total | ≈ 28.2 min |

## 21. Limitations

1. **Sample size.** n = 188 with 45 positives; ~9 positives per test fold. Fold-level
   confidence intervals are correspondingly wide, and the A3−A0 PR-AUC interval spans zero.
2. **Training budget.** 3 epochs, inherited frozen from Baseline-CV. Exp 3 established
   that the budget is not the binding constraint on prior-collapse, but the multimodal
   encoders are newly initialised and were given no additional budget.
3. **Ablation protocol.** Produced from a **single 90/10 split** (train = 170, val = 18),
   matching Phase 9.5 — not the 5×5 CV. Comparable to the historical numbers and to the
   text-only control, and to nothing else.
4. **Ablation semantics.** A non-zero score establishes only that a modality reaches the
   computation graph and perturbs the output. It is not a measure of predictive value.
5. **Feature-space limitations.** 10 of the 154 audio dimensions are constant across the
   entire corpus (COVAREP column 10 and HMPDM_0–3), contributing no information.
6. **Unresolved gating decision.** The FORMANT voicing-gate ambiguity recorded during
   Stage S2 was implemented on the literal reading of the agreed rule
   (`FORMANT_VOICED_GATED = False`) and remained unconfirmed by design.
7. **RNG asymmetry (declared in advance).** A1/A2/A3 construct `SmallMLP` encoders before
   `FusionHead`, so their fusion head initialises from a different RNG position than A0's.
   This is inseparable from adding a modality and is a property of the factor, not a
   confound.
8. **Degenerate classification.** Recall and F1 remain 0 in A0, A2 and A3, so several
   classification metrics carry no discriminative information in this experiment.

## 22. Final scientific conclusion

Experiment 7 tested input modality as a single pre-registered factor across four arms
(25 fold-runs each; 5×5 participant-level cross-validation on the frozen fold manifest;
seeds 1001–1005; λ = 0.5; 3 epochs; 100 fold-runs total).

**Multimodal activation was demonstrated, but predictive utility was not demonstrated.**

The text-only control A0 reproduced Baseline-CV on all twelve primary metrics to every
recorded digit, establishing that substituting the multimodal artifact left the text-only
result untouched. Against that control, audio and vision ablation scores moved from
exactly 0.0 to 2.5518971 and 0.4041132 respectively, with encoder construction confirmed
per fold-run. The multimodal architecture — dead code since the project's inception — is
now genuinely exercised.

Predictive performance did not improve.
**C4 failed because A3 PR-AUC decreased relative to A0 by 0.097949; the point estimate exceeded the MDE in magnitude but in the unfavorable direction, and the 95% CI included zero.**
ROC-AUC significantly degraded in
A1 (−0.1616, CI [−0.219103, −0.104121]) and A3 (−0.1394, CI [−0.175209, −0.103680]).
Combining modalities gave no detectable benefit over either alone. The only significant
favourable movements were RMSE reductions in A1 and A2 and an increase in prediction
variance, the latter reflecting escape from a near-constant regression output rather than
improved discrimination.

The text-contribution collapse of 99.05 % (2.2707760 → 0.0215758) is the most consequential
secondary finding. C3 passes on its literal wording because the value remains positive,
but the fused model's output is dominated by audio while language contributes almost
nothing — the failure mode the roadmap explicitly anticipated, in which fusion displaces
language rather than augmenting it.

Experiment 7 therefore achieved its stated objective of multimodal activation and did not
establish that the added modalities improve depression detection.

---

**Audit verdict: CLEAN — EXP 7 CAN BE CLOSED.**
The C4 failure is a scientific result, not an artifact or integrity issue.
