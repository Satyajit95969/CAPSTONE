# PHASE 13 / EXPERIMENT 3 — CONVERGENCE
## Experiment Design (pre-registration)

**Phase:** 13 · **Experiment:** 3 of the Official Experiment Sequence
**Status:** PRE-REGISTERED — written before any Experiment 3 result exists
**Date:** 2026-08-03
**Authority:** `PHASE_10_ROADMAP.md` §5 A-3, §6 Exp 3, §7 · `PHASE_11_EVALUATION_PROTOCOL.md`
**Comparator:** `Baseline-CV` (Phase 11.6), frozen · secondary context: `Exp 4` (Phase 12), frozen
**Convention:** **[R]** roadmap text · **[E]** measured in a completed phase · **[I]** design decision

> **Pre-registration statement.** This document was written and frozen **before**
> `run_exp3.py` was executed. At the time of writing,
> `trainer_outputs/exp3_convergence/` did not exist and no Experiment 3 result
> had been computed. The epoch budgets in §7 are the complete pre-declared set;
> no budget may be added, extended, or substituted after results are seen.
> Appendix A records the SHA-256 digests of the frozen inputs and of the exact
> implementation code, fixing the analysis pipeline as of this date.

---

## 1. Objective

**[R]** *"Establish whether prior-collapse is a budget artifact."*

Determine whether the collapse of both output heads to the label prior is a
consequence of the **fixed 3-epoch training budget** — a default rather than a
convergence criterion — by training the otherwise-identical frozen model under
larger epoch budgets and measuring whether the collapse resolves.

## 2. Research question

> **[E]** At the Phase 9.5 cut-off, training loss was still falling by **3.92 per
> epoch** (37.3740 → 27.5487 → 23.6329) and validation MAE still improving
> (9.9550 → 9.1430 → 8.7315). **[E]** Baseline-CV then measured, at power, a
> regression head indistinguishable from a constant (pred_var 0.005734, CI
> [−0.004089, 0.015557] including zero) and a classifier whose probability band
> is only **0.0292** wide.
>
> **Does training to convergence — rather than to an arbitrary 3-epoch default —
> widen the probability distribution, move regression MAE beyond its noise band,
> or resolve either collapse?**

## 3. Motivation

**[E]** Exp 4 returned H₀. Its measured mechanism was not absent ranking
(ROC-AUC remained 0.6333) but **threshold non-transferability**: a per-fold
probability band of 0.0292 against a fitted-τ spread of 0.2241, collapsing 24 of
25 decisions to a constant.

**[I]** That located the obstacle upstream of the decision rule, in the training
recipe — but did not identify *which* recipe condition. **[R]** RC-1 names three
joint conditions: *3 epochs, unweighted CE, class imbalance*. Exp 3 tests the
first, and it is the only one for which direct measured evidence of deficiency
exists: the loss was demonstrably still descending when training stopped.

**[I]** It is also **[R]** *"the least invasive of the three RC-1 conditions…
Low complexity, Low risk, High compatibility."* Under the roadmap's
ascending-invasiveness principle it precedes Exp 5.

## 4. Scientific background

**[E] The frozen trainer** (`trainer_mentalbert_daic.py:308-346`, SHA
`65b1902e…a230b`) exposes `epochs` as a **function parameter**:

```
def fine_tune_supervised(model, dataset, epochs=1, batch_size=8, lr=2e-5,
                         device="cpu", val_dataset=None):
    optimizer = AdamW(model.parameters(), lr=lr)
    for epoch in range(epochs):
        ...
        loss = cls_loss_fn(logits, label) + 0.5 * reg_loss_fn(reg_pred, phq)
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        print(f"[supervised] epoch {epoch+1}/{epochs} avg_loss={avg:.4f}")
```

**[I] Two consequences shape this design:**

1. **The epoch budget changes without touching a line of frozen code.** Unlike
   Exp 5 — where the loss lives *inside* the loop and would require
   reimplementation — Exp 3's factor is an argument. This yields the same
   strength of one-factor guarantee Exp 4 enjoyed.
2. **The frozen trainer already prints the per-epoch loss.** The trajectory
   needed to demonstrate convergence is harvested by **capturing stdout**, which
   is observation, not modification.

**[E] There is no validation split available.** `run_baseline_cv.py` calls
`fine_tune_supervised(..., val_dataset=None)`, training on the full 4/5
partition. **[I]** Carving a validation set out of it to drive early stopping
would reduce training-set size — a **second factor**. This design therefore does
**not** use validation-driven early stopping. Convergence is established from the
**training-loss trajectory**, which requires no held-out data and changes no data
volume.

## 5. Hypotheses

**H₁ (primary) [I].** The collapse is substantially a budget artifact. Under a
larger epoch budget, training loss reaches a plateau **and** regression MAE
improves beyond MDE 0.4066.

**H₀ (null) [I].** The collapse is not a budget artifact. Training loss
plateaus, yet the prior-collapse signature persists: regression MAE does not
improve beyond its noise band and P/R/F1 remain 0.0 at argmax.

**[R] H₀ is explicitly an acceptable and informative outcome:** *"P/R/F1 may
legitimately remain 0.0 — that is an informative outcome, confirming RC-1 needs
more than epochs."*

## 6. Single independent variable

> **The epoch budget passed to `fine_tune_supervised`.**

```
Baseline-CV / Exp 4 :  epochs = 3
Experiment 3        :  epochs = N,  N pre-declared
```

Nothing else. **[I]** In particular the decision rule reverts to **argmax** —
Exp 4 returned H₀ and contributed no accepted change, so the accepted
configuration remains Baseline-CV.

## 7. Epoch-budget arms — pre-declared [APPROVED]

| ID | Budget | Role |
|---|---|---|
| **E0** | **3 epochs** | **Equivalence control** — not an experimental arm. Validates that the Exp 3 harness reproduces Baseline-CV |
| **E1** | **20 epochs** | **PRIMARY** |
| **E2** | **10 epochs** | Sensitivity — intermediate point, characterises the trajectory |

**[I] Why 20 is primary rather than an adaptive criterion.** The research
question is *"is collapse a budget artifact?"*, and the most decisive test is the
largest defensible budget: if collapse persists at 20 epochs with the loss
plateaued, budget is eliminated as the explanation. Declaring an adaptive
stopping rule as primary would require selecting the reported epoch *after seeing
trajectories* — post-hoc selection of exactly the kind the Exp 4 pre-registration
forbids. **The budget is fixed in advance; convergence is then *demonstrated*
from the trajectory, never used to choose the result.**

**[I] The plateau rule is pre-declared as a reported diagnostic, not a
selector:** relative epoch-over-epoch training-loss improvement
`(L[k−1] − L[k]) / L[k−1] < 0.01`, sustained for **2** consecutive epochs. The
returned value is the 1-based epoch at which that run completes.

## 8. Controlled variables

| Held constant | Mechanism |
|---|---|
| Architecture | `T.MultiModalModel(bert, audio_dim=None, vision_dim=None)`, frozen trainer |
| Backbone | `mental/mental-bert-base-uncased`, vocab 30522 |
| Optimizer / lr / batch | AdamW / 2e-5 / 8 — inside the frozen function |
| Loss | `CrossEntropyLoss() + 0.5·MSELoss()`, **unweighted** — inside the frozen function |
| Grad-clip | 1.0 — inside the frozen function |
| Tokenizer / max_length | `AutoTokenizer`, 128 |
| Dataset / labels / binarisation | 188 records, 45/143, PHQ > 10.0 |
| **Training-set size** | 150/151 per fold — **unchanged**; no validation carve-out |
| Folds | `fold_manifest.json`, SHA `b9a7a91f…af8fca2f` |
| Seeds | `base_seed + repeat`; per-fold `seed×100 + fold` — identical to Baseline-CV |
| Decision rule | **argmax (τ = 0.5)** |
| Device | **Colab GPU** — held constant with Baseline-CV |
| R × k | 5 × 5 = 25 fold-runs per arm |

## 9. Experimental protocol

1. Materialise per-fold train/test partitions from the frozen manifest
   (selection only).
2. For each arm ∈ {E0, E1, E2}, repeat r ∈ 1…5, fold f ∈ 1…5:
   a. Seed identically to Baseline-CV (`seed = base + r`; per-fold `seed×100 + f`).
   b. Construct the frozen `MultiModalModel`.
   c. Call the **frozen, unmodified** `fine_tune_supervised(..., epochs=N,
      val_dataset=None)`, **capturing stdout** to harvest per-epoch `avg_loss`.
   d. Call the **frozen** `run_inference` on the held-out fold.
   e. Record predictions, the full metric suite, the loss trajectory, wall-clock,
      and the distribution diagnostics.
3. Aggregate: repeats averaged within fold, then fold-level Student-t, df = 4,
   t = 2.7764.
4. Paired per-fold comparison against Baseline-CV.

## 10. The one-factor guarantee

**[I]** `epochs` is a **parameter of the frozen function**, not a line inside it.
No training code is reimplemented, so the training loop needs no equivalence
proof. **E0 validates only the surrounding harness** — data materialisation,
seeding, metric computation, artifact writing.

**[I] The E0 gate is statistical, not bitwise.** torch is unseeded at the CUDA
level, so E0 cannot be expected to reproduce Baseline-CV exactly. The protocol's
reproducibility contract already accepts this: *"the aggregate comparator is the
reproducible object."*

## 11. Acceptance criteria [APPROVED]

**[R] Roadmap §6 Exp 3:** *"Loss reaches a stopping criterion; regression MAE
improves. P/R/F1 may legitimately remain 0.0… Runtime will increase (expected,
not a failure). Modality ablation must stay 0/0."*

| # | Criterion | Source | Test |
|---|---|---|---|
| **0** | **E0 reproduces Baseline-CV** — every primary metric inside Baseline-CV's 95 % CI | design gate | **HARD ABORT** on failure; no arm result may be reported |
| **1** | **Loss reaches a stopping criterion** — E1's plateau rate exceeds E0's | Roadmap | plateau rule of §7, reported per fold-run |
| **2** | **Regression MAE improves** beyond the noise band | Roadmap | paired per-fold vs Baseline-CV 4.8127; Δ̄ must be **negative** and \|Δ̄\| > MDE **0.4066** |
| **3** | **Modality ablation stays 0/0** | Roadmap §7.5 | `infer_dims → (None, None)`, gated before training |

**[R]** Runtime increase is expected and is **not** a failure.

## 12. Secondary observational outcomes — NOT acceptance criteria [APPROVED]

**[I]** The following are reported **in full** and treated as **observational**,
never as pass/fail conditions. **The primary acceptance criteria remain
convergence behaviour and regression improvement as defined in the roadmap.**

- p_pos band width per fold-run — Exp 4 reference **0.0292**
- p_pos min / max / mean / SD
- degeneracy count (`pos_rate ∈ {0, 1}`) — Baseline-CV reference **25/25** at argmax; Exp 4 policy P1 reference **24/25**
- prediction variance — Baseline-CV reference **0.005734**
- `pred_phq` spread
- ROC-AUC / PR-AUC movement — Baseline-CV references **0.6333 / 0.3941**

**[I] Why they are reported at all.** They address the mechanism Exp 4
identified. **[I] Why they are not criteria.** Elevating them would add
acceptance conditions the roadmap does not specify for this experiment, and
would risk an experiment being judged on a dimension it was not designed to move.

## 13. Failure criteria

| Condition | Meaning |
|---|---|
| **E0 outside Baseline-CV's CI on any primary metric** | Harness not equivalent — **abort, report nothing** |
| Loss does not plateau even at 20 epochs | **[I]** Budget question unresolved; report honestly, do **not** extend the budget post hoc |
| MAE unchanged and P/R/F1 = 0 | **H₀** — collapse is not a budget artifact |
| MAE degrades beyond MDE 0.4066 adversely | Overfitting harms the regression head — informative, reported as such |
| Any audio/vision dimension non-zero | **Abort** — Roadmap §7.5 invariant violated |
| Fold manifest / dataset / seed drift | **Abort** — comparability destroyed |
| Trainer SHA ≠ `65b1902e…a230b` | **Abort** |

**[I] Explicit prohibition.** No epoch budget may be added, extended, or
substituted after results are seen. E0/E1/E2 are the complete pre-registered set.
If the loss has not plateaued at 20 epochs, that is the finding.

## 14. Validation checks

**Pre-execution:** trainer SHA · manifest SHA · dataset 188/45/143 ·
`infer_dims → (None, None)` · GPU present (abort on CPU) · frozen-input SHAs.

**During:** per-fold `participant_id` set ≡ manifest `test_ids` · train/test
disjoint · per-epoch loss captured for every fold-run · epochs executed ==
declared budget.

**Post-execution:**
1. **E0 equivalence gate** against Baseline-CV's 95 % CIs
2. Each repeat covers 188 unique participants exactly once
3. Frozen inputs SHA-unchanged before and after
4. Aggregation self-test — reproduces `baseline_cv_summary.json` from its raw metrics
5. Independent audit re-derives every metric, trajectory, plateau verdict and
   diagnostic from the raw predictions
6. **Artifact-count check derived from the run design, never a hard-coded
   literal** — **[E]** a literal produced a false failure in Exp 4's auditor

## 15. Statistical methodology

Identical to Baseline-CV and Exp 4, so all three remain mutually comparable:

- Repeats averaged **within fold** first → k = 5 independent values
- Fold-level Student-t, df = k−1 = 4, **t = 2.7764**
- Paired per-fold: `Δ_f = Exp3_f − BaselineCV_f`; `SE = SD(Δ_f)/√5`; MDE = t·SE
- **[I]** Noise bands re-estimated from **Exp 3's own** fold spread; Baseline-CV's
  degenerate MDEs (F1/balAcc/MCC = 0.0000) are never inherited
- Two-sided α = 0.05; every effect size reported with its CI
- **[I]** Convergence is reported descriptively, not as a hypothesis test

## 16. Comparator methodology

| Layer | Values | Use |
|---|---|---|
| **Baseline-CV** | MAE 4.8127 ± 0.3275 · RMSE 6.1896 ± 0.3521 · pred_var 0.005734 · ROC-AUC 0.6333 ± 0.0466 · PR-AUC 0.3941 ± 0.0765 · P/R/F1 0/0/0 · balAcc 0.5000 · MCC 0.0000 · Acc 0.7606 | **primary quantitative comparator** — paired per fold |
| **MDE anchor** | MAE **0.4066** | Criterion 2 threshold |
| **Exp 4 context** | band width 0.0292 · degeneracy 24/25 (P1) · Baseline-CV degeneracy 25/25 at argmax | §12 observational reference; **not** a paired comparator |
| **Controls** | mean-predictor MAE 4.9135 · prevalence 0.2394 | shipped with the result (protocol P-6) |

**[I]** Exp 4 is **not** a paired comparator for Exp 3. Its threshold-conditional
metrics were measured at policy P1; Exp 3 measures at argmax.

## 17. Required artifacts

```
trainer_outputs/exp3_convergence/
├── loss_trajectories.csv               (arm, repeat, fold, epoch, avg_loss)
├── {arm}_rep{r}_fold{f}_preds.csv      75 files  (3 arms x 25 fold-runs)
├── {arm}_rep{r}_metrics.csv            15 files  (3 arms x 5 repeats)
├── exp3_metrics.csv                    consolidated (arm, repeat, fold) x metrics
├── e0_equivalence_check.json           E0 gate result — the abort condition
├── convergence_report.csv              per fold-run: plateau epoch, losses, rate
├── distribution_diagnostics.csv        SECONDARY observational outcomes
├── exp3_summary.json                   aggregates, CIs, paired deltas, verdicts
└── exp3_summary.md                     human-readable comparator table
```

**Artifact count: 75 + 15 + 7 = 97.** **[I]** The verifier **derives** this from
the run design rather than asserting the literal.

**[I] Expected outputs — structure only. No value is predicted.**

## 18. Rollback strategy

```
Remove-Item trainer_outputs\exp3_convergence -Recurse -Force
```

**[I]** Complete and sufficient. Baseline-CV and Exp 4 artifacts are opened
read-only and SHA-verified before and after; the frozen trainer is imported,
never written. Re-runs are idempotent and skip completed (arm, repeat) pairs. A
failed E0 gate leaves artifacts on disk for diagnosis, but **no result may be
reported from them**.

## 19. One-factor justification

| Dimension | Baseline-CV | Exp 4 | **Exp 3** | Changed vs accepted config? |
|---|---|---|---|---|
| Architecture / backbone | MentalBERT | *not loaded* | identical | No |
| Optimizer / lr / batch / clip | AdamW 2e-5 / 8 / 1.0 | *no training* | identical (inside frozen fn) | No |
| Loss (CE + 0.5·MSE, unweighted) | yes | *no training* | identical (inside frozen fn) | No |
| Training-set size | 150/151 | — | identical — **no val carve-out** | No |
| Data / folds / labels / seeds | frozen | identical | identical | No |
| Decision rule | argmax | P1 threshold | **argmax (reverted)** | No — Exp 4 was H₀ |
| Evaluation protocol | 5×5, fold-level t | identical | identical | No |
| **Epoch budget** | **3** | 3 (not retrained) | **20 (E1)** | **✅ the single factor** |

**[I] Confirmation.** The immediately preceding **accepted** configuration is
Baseline-CV. Exp 3 changes exactly one factor: the `epochs` argument. The
decision rule reverts to argmax and the training-set size is preserved precisely
to keep this true.

## 20. Implementation

| Script | Responsibility |
|---|---|
| `exp3_convergence/run_exp3.py` | Imports the frozen trainer; calls `fine_tune_supervised` **unmodified** with the arm's budget; captures stdout for per-epoch loss; runs frozen `run_inference`; writes predictions, trajectories, metrics, diagnostics |
| `exp3_convergence/aggregate_exp3.py` | Fold-level aggregation; **E0 equivalence gate (hard abort)**; paired Δ vs Baseline-CV; convergence analysis; observational diagnostics; criteria verdicts |
| `exp3_convergence/verify_exp3.py` | Independent auditor — imports neither of the above and neither torch nor transformers; re-derives every metric, trajectory, plateau verdict and diagnostic from raw predictions |

**Estimated cost [I]:** extrapolating from **[E]** Baseline-CV's measured 419.9 s
per 25 fold-runs at 3 epochs — E0 ≈ 7 min · E2 ≈ 11–15 min · E1 ≈ 18–25 min →
**≈ 36–47 min** on a Colab T4.

---

## Appendix A — Pre-registration record

Digests taken **before execution**. `trainer_outputs/exp3_convergence/` did not
exist at the time of writing.

**Frozen inputs (read-only):**

```
65b1902e2ba8dd996c6960db1a6595d84fd48500f4d8707cb79688093d7a230b  trainer_mentalbert_daic.py
b9a7a91f1bdd43c78d3cd1ee0c36e4ce3fb8be4b0971dcff3ff62ee5af8fca2f  trainer_outputs/baseline_cv/fold_manifest.json
2ecbfee3225910034a959fe949c7ad027cfa7d41e9bef8040374578ff8c5d572  trainer_outputs/baseline_cv/trivial_control_arm.csv
f065f5e1ff7f8e5ed4d122a8031c9cc12425802e756697d5512a915674871aac  trainer_outputs/baseline_cv/baseline_cv_summary.json
9a241851760727779fcaa4d50041b8bdc4406fe6b66ddbbd7105f4ce4fc55f00  daic_records.parquet
```

**Implementation code (fixed as of this pre-registration):**

```
def7f22001299313b1ac3509f463faf88101dd0f091608eaf5e9d953b1a5f36f  exp3_convergence/run_exp3.py
faa3bd4a9b136020f9617ccfe1a0335f6ad420eb6741c132f276529133341a40  exp3_convergence/aggregate_exp3.py
c31481c6a814a99eeb3d0f79345389239525383f04d52db2997776eb0ed285cb  exp3_convergence/verify_exp3.py
```

**Verification performed before pre-registration:** all three scripts compile
(`py_compile`); `aggregate_exp3.py` and `verify_exp3.py` import neither torch,
transformers, nor the frozen trainer; `verify_exp3.py` imports neither script
under test; the `ARMS` constant `{E0: 3, E1: 20, E2: 10}` is stated identically
in all three; the plateau rule and metric helpers were unit-tested against the
Phase 9.5 logged losses (37.3740 → 27.5487 → 23.6329) and confirmed to use
`np.var` with `ddof=0`, matching the frozen driver.

**Execution order:** `run_exp3.py` → `aggregate_exp3.py` → `verify_exp3.py`.

**Rollback:** delete `trainer_outputs/exp3_convergence/`.

---

### Erratum 1 — `verify_exp3.py` (2026-08-04, approved)

**Change.** Audit check `J2` asserted a hard-coded literal — that the frozen
Baseline-CV directory contains exactly 34 files. It is replaced by two derived
checks:

- **J2a** — the SHA-gated inputs the experiment consumes are present
  (`fold_manifest.json`, `trivial_control_arm.csv`, `baseline_cv_summary.json`)
- **J2b** — `runs/` is complete at 30 files (5 per-repeat metrics + 25 per-fold
  predictions)

**Reason.** The literal coupled the audit to a *bundling* decision rather than to
anything the experiment consumes. `baseline_cv_summary.md` is a human-readable
rendering of the `.json`, is read by no Experiment 3 script, and was legitimately
absent from the Colab upload bundle — so `J2` reported a failure on a
scientifically valid run. This is the same class of defect corrected in the
Experiment 4 verifier, and the fix follows the same principle: derive the
expectation, never hard-code it.

**SHA update.**

```
pre-registered : c31481c6a814a99eeb3d0f79345389239525383f04d52db2997776eb0ed285cb
post-erratum   : 4a1de4833cba505e6f2bc027dc7f8336c4ea9553cdf697131f853151f64f5dfe
```

**Scope.** `run_exp3.py` and `aggregate_exp3.py` are **unchanged** and remain at
their pre-registered digests. No experiment logic, metric, output, acceptance
criterion, statistical method or conclusion is affected: the verifier is an
independent auditor that sits outside the analysis pipeline, and relaxing an
unsatisfiable assertion cannot alter a measured result. The post-erratum verifier
was run against the completed Experiment 3 artifacts and returned **2461 checks,
0 failures**.

---

*Design and pre-registration only. No experiment had been executed at the time of
writing. Approved decisions incorporated: E0 = 3 / E1 = 20 (primary) / E2 = 10;
E0 equivalence gate retained as a hard abort condition; distribution diagnostics
reported in full as secondary observational outcomes and explicitly NOT
acceptance criteria; primary acceptance criteria remain convergence behaviour and
regression improvement as defined in the roadmap.*
