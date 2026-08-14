# PHASE 19 / EXPERIMENT 10 — MULTIMODAL FEATURE CONDITIONING
## Experiment Design — **FROZEN PRE-REGISTRATION**

> **Status: APPROVED AND FROZEN.** No parameter, threshold, arm, statistic,
> seed, criterion or scope in this document may be changed after any result is
> observed.

> **This experiment adds no DP, no masking, no compression, no payload
> measurement and no federation.** Those levers belong to Experiments 8, 9 and
> the roadmap rows that follow. See §5 and §29.

> **This is a precondition experiment.** It does not claim multimodal utility.
> It tests one identified confounder in Experiment 7. See §4 and §30.

---

## 1. Experiment identity and objective

| Field | Value |
|---|---|
| Phase / Experiment | Phase 19 / Experiment 10 — multimodal feature conditioning |
| Single arm-level factor | **conditioning of the pre-extracted audio and vision feature vectors** |
| Control arms | **M0** (text-only equivalence gate) and **M1** (raw multimodal) |
| Treatment arm | **M2** (conditioned multimodal) |
| Class | Task-signal / preprocessing experiment with a cross-validated criterion |
| Predecessors | Exp 7 (multimodal activation, C4 FAILED) · Exp 9 (compact update, H₀) |

**Objective.** Determine whether the multimodal ranking degradation measured in
Experiment 7 is attributable to **unconditioned feature scale** rather than to
the audio and vision modalities lacking predictive information.

## 2. Frozen scientific question

```
Does fold-safe conditioning of the pre-extracted audio and vision features
change the multimodal ranking result under the frozen 5x5 CV protocol - i.e.
is Experiment 7's multimodal degradation attributable to unconditioned feature
scale rather than to the modalities lacking predictive information?
```

## 3. Motivating evidence — measured, not assumed

Measured on the frozen multimodal artifact (§13.1) during the Phase 19 discovery
audit:

| Quantity | Audio (154-d) | Vision (84-d) |
|---|---|---|
| Raw min / max | −10.7358 / **4477.6146** | −133.5116 / **752.4710** |
| Global std | 503.9587 | 62.2077 |
| Per-column \|mean\| median → max | 0.2038 → **4341.3849** | 0.3866 → **564.9573** |
| Columns with \|mean\| > 1000 | **4** / 154 | 0 / 84 |
| Cross-column dynamic range | **4.341 × 10¹⁵** | 3.142 × 10⁴ |

Recorded batch tensors (`dataset_build/multimodal_verification.json`, batch 0)
confirm these values reach the model: `audio_vec` max **4431.21**, `video_vec`
max **752.47**, while BERT CLS activations are O(1).

**No normalisation exists anywhere in the pipeline.** A repository-wide search
for `BatchNorm`, `LayerNorm(`, `StandardScaler`, `normalize`, `zscore`,
`MinMax`, `scaler` across `trainer_mentalbert_daic.py`,
`exp7_modality/run_exp7.py`, `exp7_modality/aggregate_exp7.py`,
`build_daic_multimodal_records.py` and `build_daic_multimodal_features.py`
returns **zero occurrences**.

`SmallMLP(154 -> 38 -> 128)` is `Linear -> ReLU -> Dropout -> Linear -> ReLU`
with default initialisation (~±1/sqrt(154) ≈ ±0.08). Inputs of magnitude 4×10³
produce pre-activations orders of magnitude above the 768-d CLS block they are
concatenated with in `fusion.fc1 (256, 1024)`. At lr 2e-05 over 3 epochs the
fusion layer has no realistic opportunity to rescale that.

**This makes Exp 7's result ambiguous between two explanations, and Exp 7 could
not separate them because it varied only modality presence.**

## 4. What Experiment 7 established, and what it did not

From `FINAL_EXP7_CLOSURE.md` (SHA `928eb7b0d6c5eab7cd3c68a631d97518053774d784ab33905a682d8bcc0d36eb`),
carried here as **historical context only** (see §27):

| Established | Not established |
|---|---|
| A0 reproduced Baseline-CV on all twelve primary metrics | Whether multimodal *can* work |
| Modality activation is real (ablation 0.0 → audio 0.9339 / vision 0.5757) | Why A3 degraded |
| A3 ROC-AUC **0.4939 [0.433, 0.555]** — CI **includes 0.5** | Any predictive utility |
| A3 − A0 ROC-AUC **−0.139444 [−0.175209, −0.103680]** — significant degradation | That feature conditioning was adequate |
| C4 **FAILED**; classification degenerate (P = R = F1 = 0) in A0, A2, A3 | |

## 5. Relationship to Experiments 8 and 9 — and what is deliberately out of scope

Exp 8 returned H₀: at fixed ε and fixed d, a privacy-equivalent mechanism change
alone cannot materially improve post-noise SNR. Exp 9 returned H₀: the
publicly-specified architecture-priority mask family collapses payload and SNR
but loses the ranking signal (K2 0.505000, K3 0.492362 — both straddling chance).

Roadmap row 9 states the governing principle: *"A compact update that has lost
the signal is not a success."* Applying compression, DP or federation to a
multimodal model whose ROC-AUC CI already includes chance would measure the
compression of nothing.

**Therefore the following are OUT OF SCOPE for Experiment 10 and must not appear
in its implementation, artifacts or claims:**

- differential privacy of any kind (no clipping, no noise, no ε accounting)
- masking, compression, ordering **O**, or any k ladder
- transport payload measurement, serialisation for upload, encryption
- federation, aggregation across clients, GridFS, TPM, receipts chaining
- the Exp 9 G2 threshold (1,579,963 B), which was derived from a **text-only**
  federation round and is not transferable

## 6. Frozen arms

Exactly **three** arms. No fourth arm may be added, and no arm may be selected
among after any result is observed.

| Arm | `audio_dim` | `vision_dim` | Features | fusion input | Params | Keys | Role |
|---|---|---|---|---|---|---|---|
| **M0** | `None` | `None` | — | 768 | **109,680,132** | **207** | text-only equivalence gate |
| **M1** | 154 | 84 | **raw** | 1024 | **109,763,494** | **215** | raw-multimodal control; regenerates Exp 7 A3 |
| **M2** | 154 | 84 | **conditioned** | 1024 | **109,763,494** | **215** | treatment |

M1 and M2 are **architecturally identical**: same model class, same parameter
count, same tensor shapes, same RNG consumption. **Only the feature values
differ.** This removes the A0-vs-A1/A2/A3 initialisation asymmetry Exp 7 had to
declare, and is what makes the paired M2 − M1 contrast clean.

## 7. Frozen conditioning method

### 7.1 Definition

For **each fold** independently, and for **each feature index j** independently:

```
mu[j]         = mean over the fold's TRAIN participants only
sigma[j]      = population standard deviation (ddof = 0) over TRAIN only
sigma_safe[j] = sigma[j]  if sigma[j] > 0  else  1.0
x'[j]         = (x[j] - mu[j]) / sigma_safe[j]
```

Applied identically and independently to the **154 audio** features and the
**84 vision** features. Both partitions of the fold — train **and** test — are
transformed with the **train-derived** `mu` and `sigma_safe`.

### 7.2 Frozen parameters

| Parameter | Value |
|---|---|
| `ddof` | **0** (population standard deviation) |
| Zero-variance rule | `sigma_safe = sigma if sigma > 0 else 1.0` |
| Epsilon | **none** — no `1e-8` or any other additive term |
| Feature dropping | **forbidden** — audio stays 154-d, vision stays 84-d |
| Method | per-feature standardisation **only** — no quantile, min-max, log, robust or power transform |
| Applies to | M2 only |

`ddof = 0` is binding on the implementation, the artifacts, the aggregation and
the independent verification alike.

### 7.3 Prohibited statistic sources — binding

Conditioning statistics may **never** be computed from:

- the complete 188-participant dataset
- the test partition of any fold
- any other fold
- any other arm
- any cached statistic from a previous run

### 7.4 Zero-variance columns — measured behaviour

Exactly **10 audio columns have zero variance**, identically in the full dataset
and in all five train folds:

```
idx 16, 17   covarep_10_undocumented_mean / _std
idx 68 - 75  covarep_36..39_HMPDM_0..3_mean / _std
vision: 0 zero-variance columns (minimum sigma 5.975e-03)
```

Their constant value is **exactly 0.0**. Under §7.1 they map to
`(0.0 − 0.0) / 1.0 = 0.0`, verified to be **exactly 0 in train and test for all
five folds**. For these ten columns M1 and M2 are therefore numerically
identical; the policy is a no-op for them, not merely well-defined.

Verified across 5 folds × 2 modalities × {train, test}: **0 non-finite values**.

### 7.5 Measured effect of the transform (fold 1, train statistics applied to test)

| | raw | conditioned train | conditioned test |
|---|---|---|---|
| audio | [−10.74, 4477.61] | [−6.80, 8.28] | [−4.03, 7.26] |
| vision | [−133.51, 752.47] | [−4.22, 6.46] | [−3.86, 4.84] |

`|z| > 10` in the test partition: audio **0** of 5852, vision **0** of 3192.

This is reported as a pre-execution property of the transform, **not** as a
result and **not** as an acceptance input.

## 8. Frozen data

| Item | Value |
|---|---|
| Training artifact | `dataset_build/daic_records_multimodal.parquet` |
| SHA-256 | `1ac9f53e6102ec0dbaab84dcfdfa3f2e70f2b4a1867a841ea2b7e3ba24c67a95` |
| Rows / participants | 188 / 188 (one row each) |
| Schema | `participant_id:string · text:string · phq_score:double · features:string(JSON) · audio_coverage:double · video_coverage:double` |
| Modality dims | audio **154**, vision **84**, present for all 188 records |
| Labels | PHQ range 0–23; binarised at **PHQ > 10** → **45 positive / 143 negative** |
| Byte-identity reference | `daic_records.parquet`, SHA `9a241851760727779fcaa4d50041b8bdc4406fe6b66ddbbd7105f4ce4fc55f00` |

Neither artifact may be regenerated, re-derived or modified.

**Feature access path (frozen, unmodified):** `T.read_parquet_records()` JSON-decodes
the `features` string column into a dict (`trainer_mentalbert_daic.py:90-96`);
`MultiModalDataset._extract_audio_vec` reads `features["audio"]["wav2vec2"]`
(line 109) and `_extract_video_vec` reads `features["video"]["densenet"]`
(line 120). Both perform `torch.tensor(list, dtype=float32)` and **no transform**.

## 9. Frozen folds

| Field | Value |
|---|---|
| Manifest | `trainer_outputs/baseline_cv/fold_manifest.json` |
| SHA-256 | `b9a7a91f1bdd43c78d3cd1ee0c36e4ce3fb8be4b0971dcff3ff62ee5af8fca2f` |
| Protocol | participant-level stratified 5-fold |
| Generator | `sklearn.StratifiedKFold(n_splits=5, shuffle=True, random_state=42)` |
| Stratified on | PHQ_binary (PHQ > 10) |
| Population | 188 participants, 45 pos / 143 neg |
| Folds (id, train, test) | (1, 150, 38) · (2, 150, 38) · (3, 150, 38) · (4, 151, 37) · (5, 151, 37) |
| Overlap | **zero** in every fold; test sets partition all 188 exactly once |

**Fold ids are 1–5, not 0–4.** The manifest may not be regenerated.

The manifest's `parquet` field names `daic_records.parquet`. Its use with the
multimodal artifact is licensed **only** by the byte-identity assertion of §12.4,
which the runner must perform and abort on.

## 10. Frozen training recipe

Identical for every arm; taken unchanged from Baseline-CV and Exps 3–9.

| Parameter | Value |
|---|---|
| Model | `MultiModalModel` (`trainer_mentalbert_daic.py:241`) |
| BERT | `mental/mental-bert-base-uncased` (**gated** — authentication required) |
| EPOCHS | **3** |
| LR | **2e-05** |
| BATCH_SIZE | **8** |
| MAX_LEN | **128** |
| λ (regression coefficient) | **0.5** |
| GRAD_CLIP | **1.0** |
| Optimizer | **`transformers.optimization.AdamW`** — `torch.optim.AdamW` is NOT equivalent and must be rejected |
| BINARIZE_THRESHOLD | **10.0** |
| Loss | `CrossEntropyLoss + λ · MSELoss` |
| Scheduler | none |
| Early stopping | none |
| transformers | **4.44.0** (pinned) |
| Frozen trainer SHA | `65b1902e2ba8dd996c6960db1a6595d84fd48500f4d8707cb79688093d7a230b` |

## 11. Frozen protocol and seeding

| Field | Value |
|---|---|
| Folds × repeats × arms | 5 × 5 × 3 = **75 fold-runs** |
| BASE_SEED | **1000** |
| Per-repeat seed | `seed = 1000 + repeat`, repeats 1..5 → 1001..1005 |
| Per-fold seed | `torch.manual_seed(seed * 100 + fold)` **immediately before model construction** |
| Order | **seed → model → dataset → dataloader → loop** — NORMATIVE |
| Statistics | repeats averaged **within** fold; fold-level Student-t |
| df | **4** |
| t_crit | **2.7764451051977987** |

**Conditioning must consume ZERO RNG.** It is pure arithmetic. The following are
forbidden anywhere on the conditioning path: `numpy.random`, `torch.rand`,
`torch.randn`, `torch.randperm`, `random.*`, or any shuffling. This guarantees
all three arms share a byte-identical torch RNG stream.

`torch.backends.cudnn.deterministic` and `torch.use_deterministic_algorithms`
**must remain unset**, because the frozen recipe does not set them and enabling
them would alter the RNG/kernel stream relative to Baseline-CV and could disturb
the §14 gate. This is a deliberate decision, not an oversight.

## 12. Leakage controls — binding

### 12.1 Train-only statistics
Conditioning statistics come exclusively from `fold["train_ids"]`. Asserted in
the runner and independently recomputed by the verifier.

### 12.2 Deep copy — mandatory
The discovery audit established that `run_exp7.py:441` builds
`by_pid = {int(r["participant_id"]): r for r in all_records}` and that
`tr_recs` / `te_recs` are lists of **the same dict objects**. In-place mutation
of `features` would corrupt every fold **and** leak M2's conditioned values into
M1.

**Every record must be `copy.deepcopy`-ed before any value is replaced.** The
implementation must guarantee:

- M0 never receives conditioned values
- M1 always receives raw values
- M2 receives only that fold's train-derived conditioned values
- conditioning one fold cannot modify another fold
- conditioning one arm cannot modify another arm

### 12.3 Negative control
The runner must assert that the M2 test-partition values differ from what
**full-dataset** statistics would produce, proving the statistics were fold-local.
Failure is a HARD ABORT.

### 12.4 Byte-identity assertion
Before training, assert `participant_id`, `text` and `phq_score` are identical
between the multimodal parquet and the frozen text parquet. Failure is a HARD
ABORT — without it the fold manifest does not apply.

### 12.5 Fold integrity
Manifest SHA gated; zero train/test overlap re-asserted per fold; test sets must
partition all 188 exactly once.

### 12.6 Checkpoint exclusion — binding
`trainer_outputs/exp7_modality/ablation_multimodal/model.pt` (and `delta.pt`, and
every other existing checkpoint) **must never be loaded**. It was trained on a
single 90/10 split that overlaps every CV test fold. **Every fold constructs a
fresh model.** No warm start, no initialisation reuse, no resumption from weights.

### 12.7 Coverage columns
`audio_coverage` and `video_coverage` are **IGNORED as model inputs**. They must
not be used for filtering, imputation, weighting, conditioning or acceptance.
They may be reported descriptively only. (Discovery confirmed zero references to
them in the trainer, the Exp 7 runner, the aggregator or the verifier.)

## 13. Frozen input SHAs

Snapshotted before execution and re-verified after, inside the execution path.

```
1ac9f53e6102ec0dbaab84dcfdfa3f2e70f2b4a1867a841ea2b7e3ba24c67a95  dataset_build/daic_records_multimodal.parquet
9a241851760727779fcaa4d50041b8bdc4406fe6b66ddbbd7105f4ce4fc55f00  daic_records.parquet
65b1902e2ba8dd996c6960db1a6595d84fd48500f4d8707cb79688093d7a230b  trainer_mentalbert_daic.py
b9a7a91f1bdd43c78d3cd1ee0c36e4ce3fb8be4b0971dcff3ff62ee5af8fca2f  trainer_outputs/baseline_cv/fold_manifest.json
f065f5e1ff7f8e5ed4d122a8031c9cc12425802e756697d5512a915674871aac  trainer_outputs/baseline_cv/baseline_cv_summary.json
2ecbfee3225910034a959fe949c7ad027cfa7d41e9bef8040374578ff8c5d572  trainer_outputs/baseline_cv/trivial_control_arm.csv
098b7bb11c57012c3e8c56a790423a1741fc7e65654b34f0a87e1f4c59fb6be1  PHASE_16_EXP7_DESIGN.md
928eb7b0d6c5eab7cd3c68a631d97518053774d784ab33905a682d8bcc0d36eb  FINAL_EXP7_CLOSURE.md
cf361986e4f058259e21fefd548b335ad9fb91758696d57fdf6c762254bf288a  PHASE_18_EXP9_DESIGN.md
```

Any change during a run is a **HARD ABORT**.

## 14. M0 equivalence gate — MANDATORY

**M0 must be demonstrated to reproduce Baseline-CV. It must never be asserted.**

```
M0 gate: M0's fold-level ROC-AUC mean must lie inside
         [0.5755177227457472, 0.6911434704671701]
```

Reference: Baseline-CV ROC-AUC mean **0.6333305966064586**
(`baseline_cv_summary.json`, SHA `f065f5e1…871aac`).

**Precedent, not proof:** Exp 7's A0 reproduced Baseline-CV on all twelve primary
metrics, and Exp 9's K0 passed this gate at 0.6332539682539682. That establishes
the recipe is reproducible; it does **not** discharge the gate, which must be
evaluated on this run's own M0 output.

> **If M0 fails the gate: the experiment interpretation is INVALID. Do not
> interpret M1 vs M2. Do not claim conditioning success or failure. Report the
> measurement and the gate failure; report no conditioning verdict.**

## 15. Primary metric and primary contrast

| Field | Value |
|---|---|
| Primary metric | **ROC-AUC** |
| Primary contrast | **paired within-fold M2 − M1** |
| Pairing | fold i's M2 against fold i's M1 |
| Statistic | fold-level mean of the 5 paired differences, Student-t, df = 4, t_crit = 2.7764451051977987 |

ROC-AUC is chosen because it is the only Baseline-CV metric whose 95 % CI
excludes chance, it is the metric Exp 9 used, and it is the metric on which
Exp 7's A3 significantly degraded.

## 16. Secondary, guard and diagnostic metrics

| Metric | Role |
|---|---|
| **PR-AUC** | secondary — prevalence-aware at 0.2394; published MDE 0.09501814822834184 |
| **MAE** | guard — detects the regression head collapsing while ranking improves |
| **pred_var** | diagnostic — Exp 6 established prediction spread as informative |
| Precision / Recall / F1 / balAcc / MCC / Accuracy | **reported, NEVER acceptance inputs** |

Threshold-based classification metrics are excluded from acceptance because
Exps 3–7 repeatedly produced degenerate classification (P = R = F1 = 0). They are
recorded for completeness and may be degenerate again; that is expected and is
not a failure of this experiment.

## 17. The MDE transfer assumption — declared

```
ROC-AUC MDE = 0.05781287386071144
PR-AUC  MDE = 0.09501814822834184
MAE     MDE = 0.40660852779350554
```

All three are **pre-existing project constants** from
`baseline_cv_summary.json`, published before Experiment 10 existed, and are the
same convention Exps 5, 7 and 9 used for "must not materially degrade".

> **DECLARED ASSUMPTION.** These MDEs were derived from **text-only** Baseline-CV
> fold variance (ROC-AUC paired_se 0.02082262449651161, fold_sd
> 0.046560803844152295). Their transfer to a **multimodal** contrast is a
> **pre-registered assumption, not an established fact.** Multimodal fold variance
> may differ. This assumption is frozen here and may not be revised after any
> result is observed. Any post-hoc adjustment of an MDE is forbidden by §29.

## 18. Acceptance criteria

Every constant below is pre-existing. **No threshold was invented to make this
experiment pass.**

### 18.1 Gate — G0 (mandatory, evaluated first)

```
G0: M0 fold-level ROC-AUC mean inside [0.5755177227457472, 0.6911434704671701]
```

Failure ⇒ interpretation VOID (§14). No further criterion may be reported as
passing.

### 18.2 Primary — P1 (the hypothesis test)

**Both conditions required.**

```
P1a: mean paired (M2 - M1) fold-level ROC-AUC  >=  0.05781287386071144
P1b: the 95 % CI of that paired difference EXCLUDES 0
     (fold-level Student-t, df = 4, t_crit = 2.7764451051977987)
```

Direction: **positive means M2 is better than M1.** `delta_i = ROC_AUC(M2, fold i) − ROC_AUC(M1, fold i)`.

### 18.3 Signal existence — P2

```
P2: M2's fold-level ROC-AUC 95 % CI EXCLUDES 0.5
```

0.5 is the chance level for ROC-AUC — a mathematical constant, not a chosen
threshold.

### 18.4 Non-inferiority to text — P3

```
P3: mean paired (M0 - M2) fold-level ROC-AUC  <=  0.05781287386071144
```

Direction: **positive means M2 is WORSE than M0**, i.e. adverse. P3 passes when
the adverse degradation does not exceed the MDE.

### 18.5 Regression guard — P4

```
P4: mean paired (MAE(M2) - MAE(M0))  <=  0.40660852779350554
```

Direction: **positive means M2 has HIGHER error than M0**, i.e. adverse. MAE is
an error metric — lower is better.

### 18.6 Secondary — S1 (reported, not part of the conjunction)

```
S1: mean paired (M2 - M1) fold-level PR-AUC, with its 95 % CI, compared
    descriptively against the PR-AUC MDE 0.09501814822834184
```

### 18.7 Conjunction and verdict

**P1, P2, P3 and P4 are reported SEPARATELY for every arm and contrast,
regardless of the conjunction**, because a partial pattern is itself the
boundary measurement this experiment exists to produce.

| Outcome | Condition | Interpretation |
|---|---|---|
| **Conditioning supported** | G0 ∧ P1 ∧ P2 ∧ P3 ∧ P4 | the feature-scale hypothesis is supported **for this architecture** |
| **Partial** | G0 ∧ P1 but ¬P2 | conditioning improved ranking without establishing above-chance signal |
| **H₀** | G0 ∧ ¬P1 | this conditioning strategy does not rescue this fusion architecture |
| **VOID** | ¬G0 | the harness did not reproduce Baseline-CV; no conditioning verdict |

**H₀ is a legitimate scientific result and must never be reported as an
implementation failure.**

## 19. Environment and execution

| Field | Value |
|---|---|
| Device | **CUDA required.** The runner must abort on CPU unless `--allow-cpu` is passed, which is for non-official smoke tests only |
| Rationale | Baseline-CV and Exps 3–9 are Colab GPU results; device would otherwise become a second factor and invalidate G0 |
| GPU | Colab **T4 (16 GB) sufficient**; L4/A100 optional |
| VRAM | ~4–6 GB |
| System RAM | ~2–3 GB |
| transformers | **4.44.0** pinned |
| HF access | `mental/mental-bert-base-uncased` is **gated**; authentication required |

Must be recorded in the receipt: Python version, PyTorch version, CUDA version,
GPU name, transformers version, platform.

## 20. Output isolation

```
All Experiment 10 artifacts are written ONLY under
    trainer_outputs/multimodal_exp/
```

This directory must not exist before the run. **No existing artifact may be
overwritten.** In particular nothing under `trainer_outputs/exp9_compact_update/`,
`trainer_outputs/exp7_modality/` or `trainer_outputs/baseline_cv/` may be
created, modified or deleted.

## 21. Artifact schemas — frozen

### 21.1 `exp10_metrics.csv`
75 data rows, one per (arm, repeat, fold). Columns, in order:

```
arm, audio_dim, vision_dim, conditioned, repeat, fold, seed, n_test, n_pos,
fusion_in, n_params, lambda,
MAE, RMSE, pred_var, pred_min, pred_max, MAE_mean_pred,
ROC_AUC, PR_AUC, balAcc, MCC, Precision, Recall, F1, Accuracy,
final_train_loss, train_seconds, eval_seconds
```

`arm ∈ {M0, M1, M2}`; `conditioned ∈ {False, True}` (True only for M2);
`fold ∈ {1..5}`; `repeat ∈ {1..5}`; `seed = 1000 + repeat`.

### 21.2 `exp10_conditioning_manifest.json`
```json
{
  "schema": "exp10_conditioning/1",
  "design": {"path": "PHASE_19_EXP10_DESIGN.md", "sha256": "<this document>"},
  "method": "per-feature standardisation, train-fold statistics only",
  "ddof": 0,
  "zero_variance_rule": "sigma_safe = sigma if sigma > 0 else 1.0",
  "epsilon": null,
  "applies_to_arms": ["M2"],
  "audio_dim": 154, "vision_dim": 84,
  "folds": {
    "<fold id 1..5>": {
      "n_train": 150,
      "train_ids_sha256": "<sha256 of the sorted train id list>",
      "audio": {"mu_sha256": "...", "sigma_sha256": "...",
                "zero_variance_indices": [16,17,68,69,70,71,72,73,74,75],
                "n_zero_variance": 10},
      "vision": {"mu_sha256": "...", "sigma_sha256": "...",
                 "zero_variance_indices": [], "n_zero_variance": 0},
      "conditioned_train_min": 0.0, "conditioned_train_max": 0.0,
      "conditioned_test_min": 0.0,  "conditioned_test_max": 0.0
    }
  },
  "negative_control": {"checked": true, "fold_local_confirmed": true}
}
```
`mu_sha256` / `sigma_sha256` are SHA-256 over the raw float64 buffer of the
statistic vector, enabling bitwise independent recomputation.

### 21.3 Prediction CSVs
`{arm}_rep{r}_fold{f}_preds.csv`, 75 files, columns:
```
participant_id, phq, phq_bin, pred_phq, p_pos, pred_class
```

### 21.4 Per-repeat checkpoints
`rep{1..5}_metrics.csv` — same columns as §21.1, restricted to that repeat.
Used for resumability; a completed repeat is re-merged, never re-run.

### 21.5 `exp10_receipt.json`
```json
{
  "schema": "exp10_receipt/1",
  "experiment": "Phase 19 / Exp 10 - multimodal feature conditioning",
  "design": {"path": "PHASE_19_EXP10_DESIGN.md", "sha256": "..."},
  "generated_utc": "...", "session_id": "...", "device": "cuda",
  "frozen_input_sha": {"pre_run": {...}, "post_run": {...}, "unchanged": true},
  "protocol": {"n_folds": 5, "n_repeats": 5, "n_arms": 3, "fold_runs": 75,
               "generator": "...", "parquet": "...", "manifest_sha256": "..."},
  "recipe": {"epochs": 3, "lr": 2e-05, "batch_size": 8, "max_len": 128,
             "lambda": 0.5, "grad_clip": 1.0, "optimizer": "...",
             "bert_model": "...", "binarize_threshold": 10.0},
  "arms": {"M0": {"audio_dim": null, "vision_dim": null, "fusion_in": 768,
                  "n_params": 109680132, "n_keys": 207, "conditioned": false},
           "M1": {"audio_dim": 154, "vision_dim": 84, "fusion_in": 1024,
                  "n_params": 109763494, "n_keys": 215, "conditioned": false},
           "M2": {"audio_dim": 154, "vision_dim": 84, "fusion_in": 1024,
                  "n_params": 109763494, "n_keys": 215, "conditioned": true}},
  "seeding": {"base_seed": 1000,
              "rule": "seed = 1000 + repeat; per fold torch.manual_seed(seed*100 + fold)",
              "order": "seed -> model -> dataset -> dataloader -> loop",
              "conditioning_consumes_rng": false},
  "leakage_controls": {"deepcopy_used": true, "train_only_statistics": true,
                       "negative_control_passed": true,
                       "byte_identity_asserted": true,
                       "checkpoint_reuse": false},
  "coverage": {"used_as_input": false,
               "audio_coverage": {"min": 0.116073, "mean": 0.46188808510638285, "max": 0.723923},
               "video_coverage": {"min": 0.740728, "mean": 0.948508005319149, "max": 0.99677}},
  "dp": {"applied": false}, "masking": {"applied": false},
  "payload": {"measured": false}, "federation": {"applied": false},
  "environment": {"python": "...", "torch": "...", "cuda": "...",
                  "gpu": "...", "transformers": "...", "platform": "..."},
  "runtime_seconds": 0.0,
  "output_artifact_sha": {"exp10_metrics.csv": "...",
                          "exp10_conditioning_manifest.json": "..."}
}
```

### 21.6 `exp10_summary.json`
```json
{
  "schema": "exp10_summary/1",
  "design": {"path": "PHASE_19_EXP10_DESIGN.md", "sha256": "..."},
  "thresholds": {"baseline_roc_auc_mean": 0.6333305966064586,
                 "roc_auc_ci": [0.5755177227457472, 0.6911434704671701],
                 "roc_auc_mde": 0.05781287386071144,
                 "pr_auc_mde": 0.09501814822834184,
                 "mae_mde": 0.40660852779350554,
                 "chance_roc_auc": 0.5,
                 "t_crit": 2.7764451051977987, "df": 4},
  "arms": {"<M0|M1|M2>": {"status": "MEASURED|PENDING", "n_fold_runs": 25,
            "fold_means": {"1": 0.0, "...": 0.0},
            "mean": 0.0, "sd": 0.0, "se": 0.0, "n": 5, "df": 4,
            "t_crit": 2.7764451051977987, "ci95_lo": 0.0, "ci95_hi": 0.0,
            "inside_baseline_ci": false, "excludes_chance": false,
            "PR_AUC": {...}, "MAE": {...}, "pred_var": {...}}},
  "g0_gate": {"status": "MEASURED|PENDING", "m0_fold_level_mean": 0.0,
              "baseline_ci": [0.5755177227457472, 0.6911434704671701],
              "passed": false, "precedent_note": "..."},
  "contrasts": {
    "P1_M2_minus_M1_ROC_AUC": {"per_fold": {"1": 0.0}, "mean": 0.0, "sd": 0.0,
        "ci95": [0.0, 0.0], "direction": "positive = M2 better than M1",
        "P1a_meets_mde": false, "P1b_ci_excludes_zero": false, "P1_pass": false},
    "P3_M0_minus_M2_ROC_AUC": {"...": "direction: positive = M2 worse than M0 (adverse)",
        "P3_pass": false},
    "P4_MAE_M2_minus_M0": {"...": "direction: positive = M2 higher error (adverse)",
        "P4_pass": false},
    "S1_M2_minus_M1_PR_AUC": {"...": "reported, not part of the conjunction"}},
  "acceptance": {"G0": false, "P1": false, "P2": false, "P3": false, "P4": false,
                 "conjunction": "SUPPORTED|PARTIAL|H0|VOID|PENDING",
                 "verdict": "<verbatim text per SS 18.7>"},
  "scope_caveat": "<verbatim SS 30>",
  "mde_transfer_assumption": "<verbatim SS 17>"
}
```

### 21.7 `exp10_acceptance.csv`
```
criterion,quantity,value,threshold,direction,pass
G0,M0 fold-level ROC-AUC mean,...,"[0.5755177227457472, 0.6911434704671701]",inside,...
P1a,mean paired M2-M1 ROC-AUC,...,0.05781287386071144,>=,...
P1b,95% CI of paired M2-M1,...,0,excludes,...
P2,M2 fold-level ROC-AUC CI,...,0.5,excludes,...
P3,mean paired M0-M2 ROC-AUC,...,0.05781287386071144,<=,...
P4,mean paired MAE M2-M0,...,0.40660852779350554,<=,...
S1,mean paired M2-M1 PR-AUC,...,0.09501814822834184,reported,...
```

### 21.8 `exp10_verification.json`
```json
{"schema": "exp10_verification/1",
 "design": {"path": "PHASE_19_EXP10_DESIGN.md", "sha256": "..."},
 "fail_closed": true, "n_pass": 0, "n_fail": 0, "n_pending": 0,
 "overall": "PASS|FAIL|PENDING",
 "checks": [{"group": 1, "check": "...", "status": "PASS|FAIL|PENDING", "detail": "..."}]}
```

**No artifact may contain an invented experimental value.** Any field that
exists only after execution must be written as an explicit `PENDING`, never
fabricated.

## 22. Verification requirements — FAIL CLOSED

`verify_exp10.py` must **recompute rather than trust**, and must satisfy:

- any mismatch ⇒ **FAIL** (never a silent accept)
- missing evidence ⇒ **PENDING** (never PASS)
- non-zero exit unless every check is PASS
- a missing or non-boolean stored verdict for a complete arm ⇒ **FAIL**, never a
  default substituted from the recomputed value
- path constants used as dictionary keys must be normalised to forward slashes
  before lookup (the Windows `os.path.join` defect found in Exp 9)

Verification groups:

| # | Group |
|---|---|
| 1 | frozen design SHA |
| 2 | frozen artifact SHAs (§13) |
| 3 | arm definitions, parameter counts, key counts, fusion inputs |
| 4 | fold counts: 5 folds, 25 fold-runs per arm, 75 total, seeds `1000 + repeat` |
| 5 | fold integrity: manifest SHA, zero overlap, 188 partitioned exactly once |
| 6 | **conditioning recomputation**: μ and σ (ddof = 0) recomputed from `train_ids`, compared against the recorded digests |
| 7 | **zero-variance policy**: the 10 audio indices identified independently; conditioned values exactly 0 |
| 8 | **leakage**: statistics ≠ full-dataset statistics; M1 values raw; M0 unconditioned |
| 9 | G0 gate recomputed from the raw fold means |
| 10 | P1/P2/P3/P4 recomputed and compared against `exp10_summary.json` |
| 11 | `exp10_acceptance.csv` agrees with the recomputation |
| 12 | thresholds in the summary are the frozen constants; design SHA pinned |
| 13 | no arm outside {M0, M1, M2}; no DP / mask / payload artifact present |

## 23. Implementation boundaries

**Create exactly these five files. Modify nothing.**

```
exp10_multimodal_conditioning/
    exp10_common.py        frozen constants; fold-safe conditioner; statistics
    run_exp10.py           GPU runner: 3 arms x 5 folds x 5 repeats
    aggregate_exp10.py     G0 gate; P1-P4; acceptance matrix
    verify_exp10.py        fail-closed independent recomputation
make_exp10_upload_bundle.py
```

**Insertion point (frozen).** Conditioning is applied **after
`T.read_parquet_records()` and before `T.MultiModalDataset(...)`**, by replacing
the value lists inside a **deep copy** of each record's `features` dict. This
changes no frozen file, no model class, no tensor shape and no parameter count.

**Forbidden implementations:** editing `_extract_audio_vec` / `_extract_video_vec`;
subclassing `MultiModalDataset`; replacing `T.collate_batch`; any change to
`trainer_mentalbert_daic.py`.

**Per-fold execution order (frozen):**

```
1. compute fold conditioning statistics from train_ids only   (no RNG)
2. torch.manual_seed(seed * 100 + fold)
3. model = T.MultiModalModel(bert, audio_dim, vision_dim, device)   per arm
4. records = deepcopy(originals); for M2 replace audio/vision values
5. T.MultiModalDataset(...) -> DataLoader(collate_fn=T.collate_batch)
6. frozen training loop -> inference on that fold's test set
```

## 24. Upload bundle requirements

`make_exp10_upload_bundle.py` must:

- perform **static AST-based project-local import-closure validation**, resolving
  absolute and relative imports and including parent-package `__init__.py` files;
  build fails if any reachable project-local module is unshipped
- SHA-pin every required frozen artifact and fail on mismatch
- assert forbidden paths are absent: any `*.pt` model or delta,
  `trainer_outputs/exp9_compact_update/**`, `trainer_outputs/exp7_modality/**`,
  `trainer_outputs/mentalbert_delta.pt`, `trainer_outputs/local_probe_base.pt`,
  and every Exp 10 result artifact
- emit and print a SHA manifest of the final bundle
- never upload anything

## 25. Runtime expectations

**Measured anchor:** 18.0 s per fold-run on Colab T4 (Exp 7, same recipe and
architecture).

| | T4 | L4 (estimate) | A100 (estimate) |
|---|---|---|---|
| 75 training runs | **~23 min** | ~13 min | ~8 min |
| 75 inference passes | ~2–4 min | ~1–3 min | ~1–2 min |
| Conditioning (5 folds × 238 features) | **< 1 s** | — | — |
| Aggregation + verification (local CPU) | ~1 min | — | — |
| **Total** | **~27–32 min** | ~16–19 min | ~10–13 min |

Only the 18.0 s/fold-run figure is **measured**; the rest are **estimates**.

## 26. Failure and abort conditions

| Condition | Action |
|---|---|
| M0 fails the G0 gate (§14) | **interpretation VOID** — no conditioning verdict |
| Any frozen SHA changes during the run | **HARD ABORT** |
| Byte-identity assertion fails (§12.4) | **HARD ABORT** |
| Negative control fails (§12.3) | **HARD ABORT** |
| Any non-finite value after conditioning | **HARD ABORT** |
| Train/test overlap in any fold | **HARD ABORT** |
| M1 and M2 parameter counts differ | **HARD ABORT** |
| Conditioning consumes RNG | **HARD ABORT** |
| Any checkpoint loaded as an initialisation | **HARD ABORT** |
| A fold fails | **STOP and report** the exact repeat/fold and error; do not skip, reseed or retry with changed parameters |
| No criterion passes | **H₀** — a legitimate result |

## 27. Handling of Experiment 7's prose-only result

`trainer_outputs/exp7_modality/` contains **no `exp7_metrics.csv` and no
summary**; only the ablation artifact survives. Exp 7's cross-validated numbers
exist solely as prose in `FINAL_EXP7_CLOSURE.md`.

**That document is historical context only.** It may be cited for motivation. It
must **never** be used as a comparator, a gate input, or SHA-pinned evidence.

Arm **M1 regenerates the raw-multimodal configuration as a first-class,
machine-readable artifact** inside this experiment. If M1 lands near the recorded
0.4939 that corroborates the closure narrative; if it does not, the discrepancy
is a reproducibility finding about environments, and the closure text must not be
treated as ground truth against it.

## 28. Reproducibility requirements

- Arms, conditioning method, `ddof`, zero-variance rule, seeds, folds and
  thresholds are all fixed in this document
- Conditioning statistics recorded per fold as digests, enabling bitwise
  independent recomputation
- Frozen SHA gate before **and** after execution, inside the execution path
- All artifacts written only under `trainer_outputs/multimodal_exp/`
- `trainer_mentalbert_daic.py`, both parquets, the fold manifest, all Baseline-CV
  artifacts, and every Exp 3–9 artifact remain **frozen and untouched**

## 29. No post-hoc changes — binding

1. **Arms are fixed at M0, M1, M2.** No arm added, removed or selected among.
2. **The conditioning method is fixed** — per-feature standardisation, ddof = 0,
   `sigma_safe = sigma if sigma > 0 else 1.0`, no epsilon, no feature dropping.
   No alternative transform may be tried and compared.
3. **All thresholds are fixed**, including the MDE transfer of §17.
4. **The G0 gate may not be relaxed, re-centred or re-derived.**
5. **No metric may be promoted to primary after results are seen.**
6. **Threshold classification metrics may never become acceptance inputs.**
7. **H₀ must be reported as a legitimate scientific result.**
8. **Coverage columns may never enter the model or the acceptance path.**

## 30. Scope — what this experiment does and does not establish

> **A positive result supports the feature-scale hypothesis FOR THIS
> ARCHITECTURE.** It establishes that unconditioned feature scale was a material
> cause of the degradation Experiment 7 measured, under this fusion design, this
> recipe, and these pre-extracted features.
>
> **A negative result establishes only that THIS conditioning strategy does not
> rescue THIS fusion architecture.** It must **NOT** be interpreted as proof that
> audio and vision modalities carry no predictive information in general, nor as
> evidence about other fusion designs, other feature extractors, other
> normalisation strategies, or other datasets.
>
> **Experiment 10 makes no privacy, compression, payload or federation claim of
> any kind.** It measures a task signal under a single preprocessing factor.
>
> **Experiment 10 makes no claim of demonstrated clinical or predictive utility.**
> The ROC-AUC anchor is a *ranking* signal. Exps 3–6 returned H₀ and Exp 7's C4
> failed; nothing here revisits those verdicts.

## 31. Known limitations

1. **The MDE is transferred from text-only variance** (§17) — a declared
   assumption, not an established fact.
2. **Audio coverage is uneven**: mean 0.4619, with 111 of 188 participants below
   0.50. Conditioning does not repair uneven support; it is reported, not
   corrected.
3. **10 audio features are constant at 0.0** and carry no information in either
   arm; the effective audio dimensionality is 144 even though the tensor is 154-d.
4. **Only one conditioning method is tested.** A negative result does not
   generalise to other transforms — testing several would be threshold-shopping
   and is forbidden by §29.2.
5. **The G0 gate could fail.** It is a genuine empirical gate, not a formality.
6. **M1 may not reproduce Exp 7's recorded 0.4939**, since the environments
   differ; see §27.
7. **Determinism is not bitwise-guaranteed across GPU types**, because the frozen
   recipe does not set `cudnn.deterministic` (§11).
8. **Modality attribution is not resolved.** M2 conditions audio and vision
   together; if M2 succeeds, which modality drove it is a pre-registered
   follow-up, not a claim of this experiment.

---

**Pre-registration status: FROZEN AND APPROVED.**
**Implementation has NOT begun and is NOT authorised by this document.**
