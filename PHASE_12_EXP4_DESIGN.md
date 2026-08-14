# PHASE 12 / EXPERIMENT 4 — DECISION RULE
## Experiment Design (pre-registration)

**Phase:** 12 · **Experiment:** 4 of the Official Experiment Sequence
**Status:** PRE-REGISTERED — written before any Experiment 4 result exists
**Date:** 2026-08-02
**Roadmap authority:** `PHASE_10_ROADMAP.md` §5 A-4, §6 Exp 4, §7 · `PHASE_11_EVALUATION_PROTOCOL.md`
**Comparator:** `Baseline-CV` (Phase 11.6), frozen
**Convention:** **[R]** roadmap text · **[E]** measured in a completed phase · **[I]** design decision

> **Pre-registration statement.** This document was written and frozen **before**
> `run_exp4_decision_rule.py` was executed. At the time of writing,
> `trainer_outputs/exp4_decision_rule/` did not exist and no Experiment 4 result
> had been computed. The threshold policies in §6 are the complete pre-declared
> set; no policy may be added, tuned, or substituted after results are seen.
> Appendix A records the SHA-256 digests of the frozen inputs and of the exact
> implementation code, so the analysis pipeline is fixed as of this date.

---

## 1. Objective

**[R]** *"Recover non-zero recall without retraining."*

Determine how much of the ranking information Baseline-CV proved exists **[E]**
(ROC-AUC 0.6333, CI [0.5755, 0.6911]) can be converted into usable
classification decisions by replacing the trainer's implicit `argmax` decision
rule with a pre-declared, leakage-free operating point — **while changing
nothing about the model**.

## 2. Research question

> Given a model whose positive-class probabilities never exceed **0.4422**
> **[E]** and therefore never cross the 0.5 argmax boundary, does relocating the
> decision threshold recover classification performance that is **materially
> better than chance at the same positive rate** — or does it merely recover the
> *quantity* of positive predictions without *discrimination*?

That second clause is the substance. **[E]** Phase 11.2 observed at n = 18 that
lowering the threshold to 0.20 gave recall 1.0 with precision 0.385 ≈ prevalence
— quantity, not discrimination. Baseline-CV's AUC now predicts a different
outcome, and this experiment tests it at power.

## 3. Hypothesis

**H₁ (primary) [I].** Because ordering information is present and consistently
directed **[E]** (25/25 fold-runs AUC > 0.5; sign test p = 5.96 × 10⁻⁸), a
threshold placed inside the observed probability range will yield **Recall > 0
and F1 > 0**, and **precision materially above the 0.2394 prevalence floor**.

**H₀ (null) [I].** Thresholding recovers positives at approximately the rate
implied by the threshold, with precision ≈ prevalence — i.e. the AUC advantage
is too small to survive discretisation, and the classification failure is
predominantly a learning failure after all.

**[I]** Both outcomes are scientifically informative. H₀ would *reinstate* Phase
11.3's position and make Exps 5–6 mandatory; H₁ delivers the project's first
non-zero classification result.

## 4. Single independent variable

> **The decision function mapping predicted probability → predicted class.**

Baseline-CV: `pred_class = argmax(softmax(logits))`, equivalent to a fixed
threshold τ = 0.5.
Experiment 4: `pred_class = 1 if p_pos ≥ τ_f else 0`, where τ_f is chosen by a
**pre-declared policy**.

Nothing else. No weights, no training, no data, no folds, no seeds.

## 5. Controlled variables

| Held constant | Mechanism |
|---|---|
| Model weights | Never loaded — no model is instantiated |
| Predicted probabilities `p_pos` | Read verbatim from the 25 frozen `rep{r}_fold{f}_preds.csv` |
| Regression predictions `pred_phq` | Read verbatim; never recomputed |
| Fold partition | `fold_manifest.json`, unchanged, byte-verified |
| Dataset | 188 participants, 45 pos / 143 neg |
| Labels, binarisation threshold (PHQ > 10.0) | From the frozen preds files |
| Trainer, hyperparameters, seeds, epochs, backbone | Not invoked |
| Aggregation statistics | Same as Baseline-CV: repeats averaged within fold, then fold-level Student-t, df = 4, t = 2.7764 |
| R and k | R = 5, k = 5 → 25 fold-runs |

**[I]** No GPU, no Hugging Face access, no `transformers`, no training. This
experiment is arithmetic over existing CSVs.

## 6. Threshold-selection policy

**[E] A hard constraint discovered in the artifacts:** `run_baseline_cv.py` saved
**only test-fold predictions** and **no model checkpoints**. There are therefore
no train-fold probabilities on disk, and no way to obtain them without
retraining — which would violate the one-factor rule. This rules out the
conventional "fit τ on the training split" approach.

**P1 — PRIMARY — Leave-one-fold-out (nested) selection, per repeat. [APPROVED]**
For each repeat *r* and fold *f*: pool the held-out predictions of the **other
four folds of the same repeat**, select τ by maximising **Youden's J**
(TPR − FPR) on that pool, then apply τ to fold *f*.

- No participant's own prediction or label influences their own threshold ⇒
  **zero leakage by construction**.
- Repeats remain independent ⇒ the training-stochasticity component stays
  measurable.
- Uses only data already on disk.

**Pre-declared secondary policies — sensitivity analysis only. [APPROVED]**

| ID | Policy | Rationale |
|---|---|---|
| **P2** | **Prevalence-matched**, LOFO — τ set so the predicted positive rate equals the corpus prevalence 0.2394 | Operating point chosen without reference to labels in the evaluated fold |
| **P3** | **Fixed constant τ = 0.2394** (corpus prevalence, 45/188), declared in advance | Zero-fit control; leakage-free trivially. Arbitrary by design and expected to produce a high positive rate — it bounds the family, it is not a recommendation |
| **P4** | **Full threshold sweep**, τ ∈ [0.14, 0.45] | **Descriptive curve only — not a selection mechanism.** Reported to characterise sensitivity; no headline number may be drawn from it |

**[I]** All four are declared here, before any is computed. The headline result
is **P1**; P2–P4 are sensitivity analysis. Reporting is strictly
**threshold-conditional** — every P/R/F1 figure carries the policy that produced
it.

## 7. Success criteria

**[R] Roadmap §6 Exp 4 — primary, must both hold:**

1. **Recall > 0 and F1 > 0** — the first non-zero classification result in the
   project's history
2. **Regression MAE unchanged** — no training occurred; any movement indicates
   contamination

**[APPROVED ADDITION] Discrimination criterion, not merely a quantity criterion:**

3. **Precision must exceed the prevalence baseline 0.2394**, evaluated by
   **paired fold analysis rather than a single aggregate**. Operationally: the
   per-fold precision values (repeats averaged within fold) are tested against
   the constant 0.2394 by a one-sample Student-t on k = 5, df = 4; the criterion
   is met only when the mean difference is positive **and the 95 % CI lower
   bound clears the floor**. A point estimate above prevalence whose interval
   straddles it is not evidence.
   Reported alongside: the **random-ranker control at equal positive rate** —
   expected precision = prevalence, recall = ρ, F1 = 2·0.2394·ρ/(0.2394+ρ).

   **Why this was added.** Criteria 1–2 alone can be satisfied by a model with
   zero discrimination: any threshold below 0.4422 produces positives, so
   Recall > 0 and F1 > 0 are nearly guaranteed and would not distinguish H₁ from
   H₀. Criterion 3 is what makes the experiment falsifiable.

4. **[APPROVED ADDITION]** Balanced accuracy > 0.5 and MCC > 0, each exceeding
   the noise band re-estimated from Exp 4's own fold spread.

## 8. Failure criteria

| Condition | Meaning |
|---|---|
| Recall = 0 or F1 = 0 under all policies | Threshold relocation cannot recover positives — contradicts the p_pos evidence; investigate implementation before concluding |
| Precision ≈ 0.2394 with elevated recall | **H₀ confirmed.** Quantity without discrimination. Exps 5–6 become mandatory; Phase 11.3's position is reinstated |
| **MAE / RMSE / pred_var differ from Baseline-CV by any amount** | **Contamination — abort.** No training occurred; these must be unchanged |
| **ROC-AUC / PR-AUC differ from Baseline-CV** | **Abort.** These are threshold-free; a change proves the wrong predictions were loaded |
| Selected τ ≥ 0.4422 | Degenerate — reproduces Baseline-CV exactly; indicates a policy bug |
| Threshold selection touches the evaluated fold | **Leakage — abort and redesign** |

## 9. Metrics to report

**Threshold-conditional (the experimental arm):** Precision · Recall · F1 ·
Balanced accuracy · MCC · Accuracy · predicted positive rate · selected τ per
(repeat, fold)

**Threshold-free (must be identical to Baseline-CV — reported as invariants):**
ROC-AUC · PR-AUC

**Regression (must be identical — reported as invariants):** MAE · RMSE ·
pred_var

**Controls shipped with the result (protocol P-6):** majority-class predictor ·
random-ranker-at-equal-positive-rate · prevalence 0.2394

Per fold and aggregated: mean ± fold SD ± 95 % CI, fold-level Student-t, df = 4.

## 10. Statistical comparison against Baseline-CV

**Paired per-fold**, exploiting the frozen manifest exactly as the protocol
intends:

```
Δ_f = Exp4_f − BaselineCV_f          (repeats averaged within fold first)
Δ̄   = mean(Δ_f),  SE = SD(Δ_f)/√5,  t(0.975, 4) = 2.7764
```

**[I] A necessary deviation on the noise band.** Baseline-CV's MDE for
F1/balAcc/MCC is **0.0000** because those metrics were degenerate-constant.
Phase 11.7 §5 warns this *"must never be read as 'any change is detectable'."*
Therefore the operative noise band for Exp 4 is **re-estimated from Exp 4's own
fold-level spread**, not inherited from Baseline-CV's degenerate MDE. For Recall
and F1, BaselineCV_f = 0 exactly, so Δ_f = Exp4_f and the test reduces to a
one-sample t of Exp 4's per-fold values against zero.

Two-sided α = 0.05 throughout. Effect sizes reported with CIs; no p-value
reported without its interval.

## 11. Required inputs

| Input | Location | Verification |
|---|---|---|
| 25 × `rep{r}_fold{f}_preds.csv` | `trainer_outputs/baseline_cv/runs/` | SHA recorded pre/post; read-only |
| 5 × `rep{r}_metrics.csv` | `trainer_outputs/baseline_cv/runs/` | comparator; read-only |
| `fold_manifest.json` | `trainer_outputs/baseline_cv/` | SHA `b9a7a91f…af8fca2f` |
| `baseline_cv_summary.json` | `trainer_outputs/baseline_cv/` | SHA `f065f5e1…4871aac` |
| `trivial_control_arm.csv` | `trainer_outputs/baseline_cv/` | SHA `2ecbfee3…f8c5d572` |

**Not required:** GPU · model checkpoints · `daic_records.parquet` ·
`trainer_mentalbert_daic.py` · Hugging Face access · MongoDB.

## 12. Exact artifacts to be generated

**[APPROVED]** All output confined to `trainer_outputs/exp4_decision_rule/`:

```
trainer_outputs/exp4_decision_rule/
├── thresholds.csv                  (repeat, fold, policy, tau, pool_size,
│                                    pool_n_pos, pool_folds)
├── rep{r}_fold{f}_rescored.csv     25 files: participant_id, phq, phq_bin,
│                                   p_pos, pred_phq, pred_class_baseline,
│                                   pred_class_P1/P2/P3
├── exp4_metrics.csv                (repeat, fold, policy) × metrics + invariants
│                                   + random-ranker control
├── threshold_sweep.csv             P4 descriptive curve
├── invariants_check.json           AUC / PR-AUC / MAE / RMSE / pred_var identity proof
├── exp4_summary.json               aggregates, CIs, paired Δ, Criterion-3 test
└── exp4_summary.md                 human-readable comparator table
```

**Total: 32 files.** Nothing under `trainer_outputs/baseline_cv/` is written to.
**[APPROVED] Baseline-CV remains frozen.**

## 13. Validation checks

**Pre-execution:** input SHAs match the Phase 11.6 record · 25 preds files, 940
rows · every fold's `participant_id` set equals the manifest's `test_ids` ·
train/test disjoint per fold.

**During:** τ selection pool for (r, f) excludes fold f entirely · selected τ
lies within the observed p_pos range · predicted positive rate recorded per
fold.

**Post-execution — the decisive invariants:**

1. **ROC-AUC and PR-AUC identical to Baseline-CV** per fold-run (0.6333 / 0.3941
   aggregate) — proves the same predictions were used
2. **MAE, RMSE, pred_var identical** (4.8127 / 6.1896 / 0.005734) — proves no
   training contamination
3. `pred_class_baseline` reproduces Baseline-CV's degenerate arm exactly:
   P/R/F1 = 0, balAcc = 0.5, Accuracy = 0.7606
4. Every participant appears exactly once per repeat; 188 unique
5. Input file SHAs unchanged after the run

**[I]** Check 3 is a self-test: the pipeline must reproduce the *baseline*
decision rule from the same inputs before its result under a new rule can be
trusted.

**Numerical tolerance.** The invariant comparison uses an absolute tolerance of
**1e-9** rather than exact equality, because `pred_var`, `MAE`, `RMSE` and the
AUCs are recomputed from `pred_phq` / `p_pos` after a **CSV round-trip**, which
is not guaranteed bit-exact for float64. The actual maximum absolute difference
per metric is reported in `invariants_check.json`, so the realised precision is
visible rather than assumed.

## 14. Stop conditions

**Abort immediately** on: any input SHA mismatch · any threshold-free or
regression metric differing from Baseline-CV · leakage detected in τ selection ·
row/participant count mismatch · any write attempted under
`trainer_outputs/baseline_cv/`.

**Halt and report (do not iterate):** if H₀ obtains — precision ≈ prevalence —
**do not search for a better threshold policy.** That would convert a
pre-registered experiment into a post-hoc optimisation and is precisely the
failure mode this design exists to prevent. Report H₀ and proceed to Exp 5.

**Explicit prohibition:** no threshold policy may be added, tuned, or
substituted after results are seen. P1–P4 above are the complete pre-registered
set.

## 15. How the one-factor principle is preserved

**[R] §6 binding rule:** *"every experiment modifies exactly one factor relative
to the immediately preceding accepted configuration."*

| Dimension | Baseline-CV | Exp 4 | Changed? |
|---|---|---|---|
| Model weights / architecture | MentalBERT `MultiModalModel` | *not loaded* | No |
| Training (epochs, lr, loss, optimizer) | 3 / 2e-5 / CE+0.5·MSE / AdamW | *no training* | No |
| Data, folds, labels, seeds | frozen | identical files | No |
| Predicted probabilities | computed | **read verbatim** | No |
| Evaluation protocol / aggregation | 5×5, fold-level t | identical | No |
| **Decision function** | **argmax (τ = 0.5)** | **τ_f by policy P1** | **✅ the single factor** |

**[I]** The change is provably isolated because the experiment never
instantiates a model. The threshold-free metrics (ROC-AUC, PR-AUC) and all
regression metrics are *mathematically invariant* to τ — so verifying they are
unchanged is a **proof**, not a heuristic, that nothing but the decision rule
moved. This is the strongest one-factor guarantee available anywhere in the
roadmap.

---

## Appendix A — Pre-registration record

Digests taken **before execution**. `trainer_outputs/exp4_decision_rule/` did
not exist at the time of writing.

**Frozen inputs (Baseline-CV, read-only):**

```
b9a7a91f1bdd43c78d3cd1ee0c36e4ce3fb8be4b0971dcff3ff62ee5af8fca2f  fold_manifest.json
2ecbfee3225910034a959fe949c7ad027cfa7d41e9bef8040374578ff8c5d572  trivial_control_arm.csv
f065f5e1ff7f8e5ed4d122a8031c9cc12425802e756697d5512a915674871aac  baseline_cv_summary.json
```

**Implementation code (fixed as of this pre-registration):**

```
6ead0a78530946d986e09198f663e5680739a040008545a5b6eeb75ca6cbc3f6  exp4_decision_rule/run_exp4_decision_rule.py
59d54f25faf962b1fd2ac0d629cfb21df0fc3abe020853771327d049c8e76ff4  exp4_decision_rule/aggregate_exp4.py
5dcfe619b4da7bccf276186c7c2c5aa175a9fbbfef6dd61ccbade5590c68b3a8  exp4_decision_rule/verify_exp4.py
```

**Execution order:** `run_exp4_decision_rule.py` → `aggregate_exp4.py` →
`verify_exp4.py`. The third is an independent auditor: it imports neither of the
first two and re-derives every result from the raw frozen predictions.

**Rollback:** delete `trainer_outputs/exp4_decision_rule/`. Baseline-CV inputs
are opened read-only and SHA-verified before and after, so re-runs are
idempotent and the repository returns to its pre-execution state.

---

### Erratum 1 — `verify_exp4.py` (2026-08-03, approved)

**Change.** Audit check `A3` asserted a hard-coded literal — that the experiment
directory contains exactly 32 artifacts — against an expected set built from 31
names (6 named artifacts + 25 rescored prediction files). The count is now
**derived** from that set (`len(expected)`) rather than stated independently of
it.

**Reason.** The literal and the expected set were two separate statements of the
same quantity, and they disagreed: the arithmetic behind the literal was wrong.
`A3` was therefore unsatisfiable by construction — it would have reported a
failure on a perfect run, and did. This is the same class of defect later
corrected in the Experiment 3 verifier (`J2`), and the fix follows the same
principle: derive the expectation, never hard-code it.

**SHA update.**

```
pre-registered : 5dcfe619b4da7bccf276186c7c2c5aa175a9fbbfef6dd61ccbade5590c68b3a8
post-erratum   : 4820215fa343b4e536fbe6e5e57f11ee36aa95e5965bbd93b1d14142cf3272b9
```

**Scope.** `run_exp4_decision_rule.py` and `aggregate_exp4.py` are **unchanged**
and remain at their pre-registered digests. No experiment logic, metric, output,
acceptance criterion, statistical method or conclusion is affected: the verifier
is an independent auditor that sits outside the analysis pipeline, and relaxing
an unsatisfiable assertion cannot alter a measured result. The post-erratum
verifier was run against the completed Experiment 4 artifacts and returned
**2061 checks, 0 failures**.

---

*Design and pre-registration only. No experiment had been executed at the time
of writing. Approved decisions incorporated: Criterion 3 adopted (paired-fold);
P1 primary; P2–P4 sensitivity only; artifacts confined to
`trainer_outputs/exp4_decision_rule/`; Baseline-CV frozen.*
