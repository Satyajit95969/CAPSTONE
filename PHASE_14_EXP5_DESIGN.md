# PHASE 14 / EXPERIMENT 5 — IMBALANCE-AWARE OBJECTIVE
## Experiment Design (pre-registration)

**Phase:** 14 · **Experiment:** 5 of the Official Experiment Sequence
**Status:** PRE-REGISTERED — written before any Experiment 5 result exists
**Date:** 2026-08-04
**Authority:** `PHASE_10_ROADMAP.md` §5 A-5, §6 Exp 5, §7 · `PHASE_11_EVALUATION_PROTOCOL.md`
**Comparators:** `Baseline-CV` (Phase 11.6) **and** `Exp 4` (Phase 12), both frozen
**Convention:** **[R]** roadmap text · **[E]** measured in a completed phase · **[I]** design decision

> **Pre-registration statement.** This document was written and frozen **before**
> any Experiment 5 code was executed. At the time of writing,
> `trainer_outputs/exp5_imbalance_objective/` did not exist and no Experiment 5
> result had been computed. The weighting schemes in §7 are the complete
> pre-declared set; no scheme may be added, tuned, or substituted after results
> are seen. Appendix A records the SHA-256 digests of the frozen inputs and
> comparators. **Implementation digests are recorded in Appendix A.2 as an
> addendum once the scripts are written and before execution** — the scripts do
> not yet exist at the time of this pre-registration.

---

## 1. Objective

**[R]** *"Improve classification beyond thresholding alone."*

Determine whether replacing the trainer's **unweighted** cross-entropy with a
**class-weighted** cross-entropy produces a classifier that discriminates between
participants — the capability Experiments 4 and 3 have now separately shown
cannot be recovered by the decision rule or by the training budget.

## 2. Research question

> **[E]** Exp 4 (H₀) established that relocating the decision threshold recovers
> the *quantity* of positive predictions without discrimination: precision
> 0.1431 fell significantly **below** prevalence 0.2394 (t(4) = −5.8641,
> p = 0.004222), F1 0.2215 was indistinguishable from a random ranker at equal
> positive rate (Δ +0.009579, CI [−0.017016, 0.036174]), and 24 of 25 decisions
> were degenerate.
>
> **[E]** Exp 3 (H₀) established that training to convergence widens the
> prediction distribution without improving accuracy: `pred_var` rose 0.0057 →
> 9.4247 (the only detectable change, CI [6.8710, 11.9669]) while MAE moved only
> −0.1856 against an MDE of 0.5334.
>
> **Does an imbalance-aware objective produce discrimination — measured
> threshold-free by PR-AUC and ROC-AUC, and threshold-conditionally at the
> unchanged argmax rule — rather than a third variety of movement-without-improvement?**

## 3. Motivation

**[R]** RC-1 names three joint conditions for prior-collapse: **3 epochs ·
unweighted CE · class imbalance**.

**[E] Two of the three cheap interventions are now formally excluded:**

| Experiment | Factor | Verdict | What it eliminated |
|---|---|---|---|
| **Exp 4** | decision rule | H₀ | The threshold is not the operative constraint |
| **Exp 3** | epoch budget | H₀ | The training budget is not the operative constraint |

**[I] By elimination, the remaining RC-1 conditions are the training objective
itself.** Exp 5 changes the class weighting of the cross-entropy term and nothing
else. It is the last untested condition in the Immediate block, and the roadmap
places it exactly here.

**[I] Exp 3 additionally supplies a design constraint.** It demonstrated that a
model can acquire large prediction variance while remaining no more accurate.
Any Exp 5 success criterion must therefore be anchored to **discrimination**, not
to variance, positive count, or recall alone.

## 4. Scientific background

**[E] The frozen trainer's loss** (`trainer_mentalbert_daic.py:308-324`, SHA
`65b1902e…a230b`):

```
optimizer   = AdamW(model.parameters(), lr=lr)
cls_loss_fn = nn.CrossEntropyLoss()          # unweighted
reg_loss_fn = nn.MSELoss()
...
loss_cls = cls_loss_fn(logits, batch["label"])
loss_reg = reg_loss_fn(reg_pred, batch["phq"])
loss     = loss_cls + 0.5 * loss_reg
torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
```

**[E]** Corpus prevalence is 45/188 = **0.2393617021276596**; every fold carries
9 positives against 28–29 negatives.

**[I]** Under unweighted CE on a 24 %-positive corpus, the loss-minimising early
solution is to predict the majority class and place probabilities near the prior.
**[E]** That is what Baseline-CV recorded: P/R/F1 = 0, balanced accuracy exactly
0.5000, probabilities in a band 0.0292 wide centred at 0.2682.

**[I]** Class weighting alters each sample's contribution to the CE gradient so
the minority class is not economically ignorable. The mechanistic prediction is a
probability distribution that is **better separated between classes**, not merely
wider — Exp 3 already produced width without benefit.

**[I] A technical point that keeps Exps 5 and 6 separable.** PyTorch's
`CrossEntropyLoss(weight=w, reduction='mean')` normalises by the sum of sample
weights, so the CE term's *scale* is approximately preserved. Class weighting
therefore changes the **relative weighting between classes**, not the CE : MSE
ratio — which is Exp 6's factor.

## 5. Hypotheses

**H₁ (primary) [I].** A class-weighted objective produces genuine discrimination:
**PR-AUC detectably above Baseline-CV's 0.3941**, with precision above the
prevalence floor and F1 above the random-ranker control at equal positive rate.

**H₀ (null) [I].** Class weighting relocates the decision boundary without
improving *ordering*: PR-AUC and ROC-AUC remain within their noise bands of
Baseline-CV, and any Recall/F1 gain at argmax is again indistinguishable from a
random ranker at the same positive rate.

**[I]** Both are informative. H₀ would establish that none of the three RC-1
conditions is individually sufficient, elevating **Exp 6** (loss-term
rebalancing) and making the case that the conditions are genuinely *joint*.

## 6. Single independent variable

> **The class weighting of the cross-entropy term.**

```
Baseline-CV / Exp 3 / Exp 4 :  nn.CrossEntropyLoss()                 # w = uniform
Experiment 5                :  nn.CrossEntropyLoss(weight=w_fold)    # w pre-declared
```

Nothing else: same architecture, backbone, optimizer, learning rate, **epochs
= 3**, batch size, grad-clip, tokenizer, max_length, folds, seeds, dataset, MSE
term and its 0.5 coefficient, and **the same argmax decision rule**.

**[I] Both the decision rule and the epoch policy revert to Baseline-CV's
values.** Exps 3 and 4 returned H₀, so neither was accepted into the
configuration. Carrying forward Exp 4's threshold or Exp 3's 20-epoch budget
would change two factors.

## 7. Weighting schemes — pre-declared [APPROVED]

| ID | Scheme | Definition | Role |
|---|---|---|---|
| **W0** | **Uniform** | `weight = None` | **Equivalence control — not an experimental arm.** Validates that the reimplemented training loop reproduces Baseline-CV |
| **W1** | **Balanced — PRIMARY** | `w_c = n_train / (2 · n_c)`, computed **per training fold from training labels only** | The experimental arm |
| **W2** | Square-root balanced | `w_c = sqrt(n_train / (2 · n_c))` | Sensitivity — milder correction |
| **W3** | Prevalence-inverse, fixed | `w_pos = 1 / 0.2393617021276596`, `w_neg = 1` | Sensitivity — zero-fit control, declared in advance |

**[I]** Weights derive from **training-fold labels only**; the held-out fold
contributes nothing to their computation. All four are declared here, before any
is computed. **[I]** W3 is deliberately arbitrary — it exists to bound the family,
not to be recommended.

## 8. Controlled variables

| Held constant | Mechanism |
|---|---|
| Architecture | `T.MultiModalModel(bert, audio_dim=None, vision_dim=None)`, frozen trainer |
| Backbone | `mental/mental-bert-base-uncased`, vocab 30522 |
| Optimizer / lr / batch | AdamW / 2e-5 / 8 |
| **Epochs** | **3** — Baseline-CV's value; Exp 3 was H₀ and contributed no accepted change |
| Grad-clip | 1.0 |
| Regression term | `MSELoss`, coefficient **0.5**, **unweighted** |
| Tokenizer / max_length | `AutoTokenizer`, 128 |
| Dataset / labels / binarisation | 188 records, 45/143, PHQ > 10.0 |
| Training-set size | 150/151 per fold — no validation carve-out |
| Folds | `fold_manifest.json`, SHA `b9a7a91f…af8fca2f` |
| Seeds | `base_seed + repeat`; per-fold `seed×100 + fold` |
| **Decision rule** | **argmax (τ = 0.5)** — Baseline-CV's value |
| Device | **Colab GPU** — held constant with Baseline-CV |
| R × k | 5 × 5 = 25 fold-runs per arm |

## 9. Experimental protocol

1. Materialise per-fold train/test partitions from the frozen manifest
   (selection only, no data modification).
2. For each arm ∈ {W0, W1, W2, W3}, repeat r ∈ 1…5, fold f ∈ 1…5:
   a. Seed identically to Baseline-CV (`seed = base + r`; per-fold `seed×100 + f`).
   b. Compute `w_fold` from the **training** partition's labels (W0 → `None`).
   c. Construct the frozen `MultiModalModel`.
   d. Train 3 epochs with `CrossEntropyLoss(weight=w_fold) + 0.5·MSELoss`,
      grad-clip 1.0, AdamW lr 2e-5, batch 8.
   e. Run the frozen `run_inference` on the held-out fold.
   f. Record `pred_phq`, `p_pos`, `pred_class` (argmax), the full metric suite,
      the weight vector, and the distribution diagnostics.
3. Aggregate: repeats averaged within fold, then fold-level Student-t, df = 4,
   t = 2.7764.
4. Paired per-fold comparison against Baseline-CV and, threshold-conditionally,
   against Exp 4.

## 10. The one-factor problem — and how it is solved [APPROVED]

**[I] This is the most consequential design issue, and Exp 5 is structurally
weaker here than Exps 3 and 4. It is stated plainly rather than glossed.**

- **[E] Exp 4's** one-factor claim was a *proof*: it never loaded a model, so
  verifying the τ-invariant metrics were unchanged proved nothing but the
  decision rule moved.
- **[E] Exp 3's** was nearly as strong: `epochs` is a **parameter** of the frozen
  `fine_tune_supervised`, so no training code was reimplemented — and its E0
  control reproduced Baseline-CV **exactly** (max diff 0 across all 25 fold-runs).
- **Exp 5 cannot do either.** The loss is constructed *inside*
  `fine_tune_supervised` (line 312), and **[I] the trainer must not be edited**.
  Exp 5 therefore requires a driver implementing its own training loop, importing
  `MultiModalModel`, `MultiModalDataset`, `collate_batch`, `run_inference` and
  `read_parquet_records` unchanged from the frozen trainer.

**[I] The W0 equivalence control is what restores rigour.** The reimplemented
loop is a *single* code path parameterised by `weight`. Running it with
`weight=None` must reproduce Baseline-CV.

> **W0 EQUIVALENCE GATE — HARD ABORT [APPROVED].** W0's aggregate ROC-AUC,
> PR-AUC, MAE, RMSE, pred_var, P/R/F1, balAcc, MCC and Accuracy must each fall
> **inside Baseline-CV's 95 % CI**. If W0 fails, the loop is not equivalent and
> **no W1/W2/W3 result may be reported** — the experiment aborts.

**[I]** Reproduction is expected to be **statistical, not bitwise**: torch is
unseeded at the CUDA level, and the protocol's reproducibility contract accepts
this (*"the aggregate comparator is the reproducible object"*). **[E]** Exp 3's E0
in fact reproduced Baseline-CV exactly, which suggests the seeding is
deterministic on this stack — but Exp 5 must not *assume* that, because it runs
different code.

**[I]** The implementation must additionally carry a **line-by-line
correspondence table** between the reimplemented loop and `fine_tune_supervised`
lines 308–330, with the CE `weight` argument as the sole difference. That table
is a required artifact of the result report.

## 11. Acceptance criteria [APPROVED]

**[R] Roadmap §6 Exp 5:** *"Recall / F1 / PR-AUC improve over **both**
[Baseline-CV and Exp 4]. Regression MAE must not materially degrade."*

| # | Criterion | Source | Test |
|---|---|---|---|
| **0** | **W0 reproduces Baseline-CV** — every primary metric inside its 95 % CI | design gate | **HARD ABORT** on failure |
| **1** | **PR-AUC detectably above Baseline-CV 0.3941** | Roadmap | paired per-fold; Δ̄ must exceed the noise band re-estimated from Exp 5's own fold spread |
| **2** | **Recall and F1 improve over Baseline-CV** (0.0000 / 0.0000) at argmax | Roadmap | paired per-fold |
| **3** | **Recall and F1 improve over Exp 4** (0.5422 / 0.2215) | Roadmap | paired per-fold, reported **threshold-conditionally** — see §12 |
| **4** | **Regression MAE not materially degraded** | Roadmap | Δ̄ vs Baseline-CV 4.8127 must not exceed MDE **0.4066** in the adverse direction |
| **5** | **Discrimination, not quantity** *(carried forward from Exp 4)* | approved addition | precision above prevalence **0.2394** **and** F1 above the random-ranker control at equal positive rate — both paired-fold, CI clearing the floor |
| **6** | **Decisions non-degenerate** *(carried forward from Exp 4)* | approved addition | count of fold-runs with `pos_rate ∈ {0, 1}` materially below Exp 4's **24/25** |

**[I] Criterion 5 is the load-bearing one.** **[E]** Exp 4 proved that Recall/F1
gains can occur with no discrimination whatsoever, and **[E]** Exp 3 proved that
prediction variance can rise with no accuracy gain. Without Criterion 5,
criteria 1–3 could again be satisfied by a model that merely predicts more
positives.

**[I] Criterion 6** follows from Exp 4's measured mechanism: the degeneracy count
is the most direct observable of whether the decision has become non-trivial.
**[E]** References: Baseline-CV 25/25 degenerate at argmax; Exp 4 policy P1
24/25; Exp 3 E1 22/25.

## 12. Two comparison caveats the roadmap's wording requires

**[I] (a) "PR-AUC improve over both" is partly vacuous.** **[E]** Exp 4's PR-AUC
is *identical* to Baseline-CV's (0.394064) — it was invariant to the decision
rule by construction. Criterion 1 therefore reduces to "improve over
Baseline-CV". Stated so the result report does not imply two independent
comparisons.

**[I] (b) Comparing F1 across different decision rules is not like-for-like.**
**[E]** Exp 4's F1 = 0.2215 was measured at threshold policy P1; Exp 5's will be
measured at argmax. Both are reported, with the primary like-for-like comparison
being **Exp 5 @ argmax vs Baseline-CV @ argmax**, and the roadmap-mandated Exp 4
comparison reported **threshold-conditionally and explicitly flagged**.

## 13. Failure criteria

| Condition | Meaning |
|---|---|
| **W0 outside Baseline-CV's CI on any primary metric** | Loop not equivalent — **abort, report nothing** |
| PR-AUC / ROC-AUC within noise of Baseline-CV | **H₀** — weighting did not improve ordering |
| Recall/F1 up but precision ≈ or below prevalence | **H₀** — Exp 4's outcome reproduced by a different route |
| Degeneracy count ≈ 24/25 | Decisions still trivial; the operative constraint is elsewhere |
| MAE degrades beyond MDE 0.4066 | Classification bought at the cost of regression — **[R]** Exp 6's stated failure mode |
| Any audio/vision dimension non-zero | **Abort** — Roadmap §7.5 invariant: ablation stays 0/0 before Exp 7 |
| Fold manifest / dataset / seed drift | **Abort** — comparability destroyed |
| Trainer SHA ≠ `65b1902e…a230b` | **Abort** |

**[I] Explicit prohibition.** No weighting scheme may be added, tuned, or
substituted after results are seen. W0–W3 are the complete pre-registered set. If
H₀ obtains, report it and proceed to Exp 6 — **do not search for a better
weighting**.

## 14. Validation checks

**Pre-execution:** trainer SHA · manifest SHA · dataset 188/45/143 ·
`infer_dims → (None, None)` · GPU present (abort on CPU) · frozen-input SHAs ·
artifact directory absent.

**During:** class weights derived from **training** labels only, asserted per
fold · weight vector recorded per (arm, repeat, fold) · W0 asserted to pass
`weight=None` · per-fold `participant_id` set ≡ manifest `test_ids` ·
train/test disjoint.

**Post-execution:**
1. **W0 equivalence gate** against Baseline-CV's 95 % CIs
2. Each repeat covers 188 unique participants exactly once
3. Modality ablation 0/0 across all arms
4. Baseline-CV, Exp 3 and Exp 4 artifacts SHA-unchanged
5. Aggregation self-test — reproduces `baseline_cv_summary.json` from its raw metrics
6. Independent audit re-derives every metric, weight and diagnostic from the raw predictions
7. **Artifact-count check derived from the run design, never a hard-coded
   literal** — **[E]** literals produced false failures in both the Exp 4 (`A3`)
   and Exp 3 (`J2`) auditors
8. **Consolidated CSVs written per (arm, repeat), not only at end of run** —
   **[E]** end-of-run-only consolidation lost E0 and part of E1 when an Exp 3
   invocation was interrupted

## 15. Statistical methodology

Identical to Baseline-CV, Exp 3 and Exp 4, so all four remain mutually
comparable:

- Repeats averaged **within fold** first → k = 5 independent values
- Fold-level Student-t, df = k−1 = 4, **t = 2.7764**
- Paired per-fold: `Δ_f = Exp5_f − Comparator_f`; `SE = SD(Δ_f)/√5`; MDE = t·SE
- **[I]** Noise bands re-estimated from **Exp 5's own** fold spread; Baseline-CV's
  degenerate MDEs (F1/balAcc/MCC = 0.0000) are never inherited
- Criterion 5 tested as a one-sample paired-fold t against the prevalence
  constant, df = 4; the criterion is met only when the mean difference is
  positive **and** the 95 % CI lower bound clears the floor
- Two-sided α = 0.05; every effect size reported with its CI; no p-value without
  an interval

## 16. Comparator methodology

| Layer | Values | Use |
|---|---|---|
| **Baseline-CV** | MAE 4.8127 ± 0.3275 · RMSE 6.1896 ± 0.3521 · pred_var 0.0057 ± 0.0079 · ROC-AUC 0.6333 ± 0.0466 · PR-AUC 0.3941 ± 0.0765 · balAcc 0.5000 · MCC 0.0000 · P/R/F1 0/0/0 · Acc 0.7606 ± 0.0035 | **primary quantitative comparator** — paired per fold |
| **MDE anchors** | MAE **0.4066** · RMSE 0.4372 · ROC-AUC 0.0578 · PR-AUC **0.0950** | acceptance thresholds |
| **Exp 4 (policy P1)** | Precision 0.1431 ± 0.0367 · Recall 0.5422 ± 0.2561 · F1 0.2215 ± 0.0766 · balAcc 0.5070 · MCC 0.0131 · pos_rate 0.5316 · degeneracy 24/25 | Criterion 3 (threshold-conditional) and Criterion 6 reference |
| **Exp 3 (E1) context** | pred_var 9.4247 · band width 0.3002 · degeneracy 22/25 · MAE 4.6272 · ROC-AUC 0.6929 · PR-AUC 0.4719 | observational reference only — **not** a comparator (different epoch budget) |
| **Controls** | prevalence **0.2393617021276596** · random-ranker F1 at equal positive rate · mean-predictor MAE 4.9135 | shipped with the result (protocol P-6) |

**[I]** Exp 3 is **not** a paired comparator for Exp 5: it ran at 20 epochs where
Exp 5 runs at 3. Its diagnostics are cited only to contextualise distribution
width and degeneracy.

## 17. Required artifacts

```
trainer_outputs/exp5_imbalance_objective/
├── class_weights.csv                    (arm, repeat, fold, w_neg, w_pos,
│                                         n_train, n_pos_train, n_neg_train)
├── {arm}_rep{r}_fold{f}_preds.csv       100 files  (4 arms x 25 fold-runs)
├── {arm}_rep{r}_metrics.csv             20 files   (4 arms x 5 repeats)
├── exp5_metrics.csv                     consolidated (arm, repeat, fold) x metrics
├── w0_equivalence_check.json            W0 vs Baseline-CV CI gate — the abort condition
├── degeneracy_report.csv                pos_rate distribution per arm vs Exp 4's 24/25
├── loop_correspondence.md               line-by-line map to fine_tune_supervised
├── exp5_summary.json                    aggregates, CIs, paired deltas vs both comparators
└── exp5_summary.md                      human-readable comparator table
```

**Artifact count: 100 + 20 + 7 = 127.** **[I]** The verifier must **derive** this
from the run design rather than assert the literal.

**[I] Expected outputs — structure only. No value is predicted.**

## 18. Rollback strategy

```
Remove-Item trainer_outputs\exp5_imbalance_objective -Recurse -Force
```

**[I]** Complete and sufficient. Baseline-CV, Exp 3 and Exp 4 artifacts are
opened read-only and SHA-verified before and after; the frozen trainer is
imported, never written. Re-runs are idempotent and skip completed
(arm, repeat) pairs. A failed W0 gate leaves artifacts on disk for diagnosis, but
**no result may be reported from them**.

## 19. One-factor justification

| Dimension | Baseline-CV | Exp 4 | Exp 3 | **Exp 5** | Changed vs accepted config? |
|---|---|---|---|---|---|
| Architecture / backbone | MentalBERT | *not loaded* | identical | identical | No |
| Optimizer / lr / batch / clip | AdamW 2e-5 / 8 / 1.0 | *no training* | identical | identical | No |
| **Epochs** | **3** | 3 (not retrained) | 20 (H₀) | **3 (reverted)** | No — Exp 3 was H₀ |
| MSE term and 0.5 coefficient | yes | *no training* | identical | identical | No |
| Data / folds / labels / seeds | frozen | identical | identical | identical | No |
| **Decision rule** | **argmax** | P1 threshold (H₀) | argmax | **argmax (reverted)** | No — Exp 4 was H₀ |
| Evaluation protocol | 5×5, fold-level t | identical | identical | identical | No |
| **CE class weighting** | **uniform** | uniform | uniform | **weighted (W1)** | **✅ the single factor** |

**[I] Confirmation.** The immediately preceding **accepted** configuration is
**Baseline-CV** — Exps 3 and 4 both returned H₀ and contributed no accepted
change. Exp 5 modifies exactly one factor relative to it: the class weighting of
the CE term. The epoch budget and decision rule both revert to Baseline-CV's
values precisely to preserve this.

## 20. Implementation plan

| Script | Responsibility |
|---|---|
| `exp5_imbalance_objective/run_exp5.py` | Imports the frozen trainer's model/dataset/collate/inference; implements the **single parameterised** training loop; derives per-fold weights; trains and evaluates 4 arms × 25 fold-runs; writes predictions, weights, metrics, degeneracy report, consolidated CSVs **per (arm, repeat)** |
| `exp5_imbalance_objective/aggregate_exp5.py` | Fold-level aggregation; **W0 equivalence gate (hard abort)**; paired Δ vs Baseline-CV and Exp 4; Criterion-5 discrimination test; Criterion-6 degeneracy comparison; criteria verdicts |
| `exp5_imbalance_objective/verify_exp5.py` | Independent auditor — imports neither of the above; re-derives every metric, weight derivation, leakage check, coverage and diagnostic from raw predictions; **derived** artifact count |

**Execution order:** `run_exp5.py` → `aggregate_exp5.py` → `verify_exp5.py`.

**Estimated cost [I]:** extrapolating from **[E]** Baseline-CV's measured 419.9 s
per 25 fold-runs at 3 epochs — ≈ 7 min per arm × 4 arms ≈ **28–35 min** on a
Colab T4.

## 21. Repository locations

```
CREATE  exp5_imbalance_objective/run_exp5.py
CREATE  exp5_imbalance_objective/aggregate_exp5.py
CREATE  exp5_imbalance_objective/verify_exp5.py
CREATE  trainer_outputs/exp5_imbalance_objective/     (127 artifacts)
UPDATE  PHASE_14_EXP5_DESIGN.md Appendix A.2          (implementation SHAs, before execution)

READ-ONLY  trainer_outputs/baseline_cv/**  ·  trainer_outputs/exp4_decision_rule/**
           trainer_outputs/exp3_convergence/**  ·  trainer_mentalbert_daic.py
           daic_records.parquet
UNTOUCHED  all frozen reports, prior pre-registrations, and the roadmap
```

## 22. Verification strategy

Three-stage, mirroring Exps 3 and 4: **runner** (in-line gates, aborts on
integrity failure) → **aggregator** (W0 gate, aggregation self-test, paired
inference) → **independent auditor** (re-derives everything from raw predictions,
importing neither predecessor).

Reported back: pre-flight gates · **W0 equivalence table** · per-fold class
weights · aggregate table for all four arms · paired Δ vs Baseline-CV with MDEs ·
threshold-conditional Δ vs Exp 4 · **Criterion-5 discrimination test** ·
**Criterion-6 degeneracy counts vs 24/25** · loop correspondence table ·
modality-ablation invariant · runtime · audit check count and failures · final
SHAs and `git status`.

---

## Appendix A — Pre-registration record

### A.1 Frozen inputs and comparators (read-only)

Digests taken **before any Experiment 5 code existed**.
`trainer_outputs/exp5_imbalance_objective/` did not exist at the time of writing.

```
65b1902e2ba8dd996c6960db1a6595d84fd48500f4d8707cb79688093d7a230b  trainer_mentalbert_daic.py
9a241851760727779fcaa4d50041b8bdc4406fe6b66ddbbd7105f4ce4fc55f00  daic_records.parquet
b9a7a91f1bdd43c78d3cd1ee0c36e4ce3fb8be4b0971dcff3ff62ee5af8fca2f  trainer_outputs/baseline_cv/fold_manifest.json
2ecbfee3225910034a959fe949c7ad027cfa7d41e9bef8040374578ff8c5d572  trainer_outputs/baseline_cv/trivial_control_arm.csv
f065f5e1ff7f8e5ed4d122a8031c9cc12425802e756697d5512a915674871aac  trainer_outputs/baseline_cv/baseline_cv_summary.json
5defdae2a20d0abc164611e8cbe6b8034e3f65c593d8c9b3e38266a1800aa6f2  trainer_outputs/exp4_decision_rule/exp4_summary.json
9dd135b6b79af06a85e64a5aa8c1896d19df8059dd89c053e50e1cffe20af2f9  trainer_outputs/exp3_convergence/exp3_summary.json
```

### A.2 Implementation code — addendum (2026-08-04)

Digests recorded after writing and BEFORE execution.
`trainer_outputs/exp5_imbalance_objective/` did not exist at the time of writing.

```
4234508133c74c413f5a6498de0f3993e55bf0e579cadc3f4d0380c01a5cbb7f  exp5_imbalance_objective/run_exp5.py
66bbac11345445740b0f31cf7ae148b9bd8dea651b5541a31e6def68e701ca57  exp5_imbalance_objective/aggregate_exp5.py
57843f65a6b83c9015301fb37f34b2c8652f8b800f8efbcd495b86b017d2b6b8  exp5_imbalance_objective/verify_exp5.py
```

**`run_exp5.py` verification before recording:** compiles (`py_compile`); the
four weighting schemes reproduce their §7 formulas exactly on a 150/36/114 fold
(W1 0.657895/2.083333 · W2 0.811107/1.443376 · W3 1.0/4.177778 · W0 `None`);
`T.AdamW` confirmed to be `transformers.optimization.AdamW`
(eps 1e-6, weight_decay 0.0, correct_bias True) with a hard abort if it ever
resolves elsewhere; `nn.CrossEntropyLoss(weight=None)` confirmed bit-identical
to `nn.CrossEntropyLoss()`, so the W0 arm exercises the frozen computation
exactly.

**`aggregate_exp5.py` verification before recording:** compiles (`py_compile`);
imports neither torch, transformers nor the frozen trainer; the aggregation
contract reproduces `baseline_cv_summary.json` across all 12 metrics at
**max |diff| 0.000e+00** against the real Baseline-CV artifacts; the
zero-variance guard was tested against the degenerate case that Baseline-CV's
Recall/F1 present (exactly 0 in every fold). **One defect was found and
corrected during verification:** the Criterion-5 random-ranker control must be
evaluated per fold-run using that fold's own prevalence and then averaged, as
Experiment 4 did (`run_exp4_decision_rule.py:519`); evaluating it at the
fold-mean `pos_rate` differs by 0.118 through Jensen's inequality. The corrected
path was validated by replaying Experiment 4's own artifacts and reproducing its
stored value **0.211901 exactly**.

**`verify_exp5.py` verification before recording:** compiles (`py_compile`);
imports neither torch, transformers, the frozen trainer, nor either script under
test — every constant is restated independently. The expected artifact count is
**derived** from the run design (7 named + 100 preds + 20 metrics = 127), never a
literal, following the false failures the Exp 4 `A3` and Exp 3 `J2` literals
produced. Primitives validated against real prior artifacts: the W0–W3 weight
formulas reproduce §7 exactly on a 150/36/114 fold; `fold_ci` reproduces
`baseline_cv_summary.json` at **max |diff| 0.000e+00**; and the per-fold-run
random-ranker control reproduces Experiment 4's stored **0.211901** exactly, so a
regression to the fold-mean shortcut would be caught.

**[E] Pre-registration complete.** All three implementation digests are recorded
above, before any execution. `trainer_outputs/exp5_imbalance_objective/` does not
exist at the time of this record.

**Rollback:** delete `trainer_outputs/exp5_imbalance_objective/`.

---

*Design and pre-registration only. No experiment executed, no code written, no
repository artifact modified. Approved decisions incorporated: W1 (balanced,
per-training-fold) primary; W2 (square-root balanced) and W3 (fixed
prevalence-inverse) as pre-registered sensitivity analyses; W0 equivalence gate
retained as a hard abort condition; Criteria 5 and 6 carried forward from
Experiment 4; one-factor rule preserved with epochs and decision rule reverted to
Baseline-CV values; frozen trainer and all prior experiment artifacts unchanged.*
