# PHASE 15 / EXPERIMENT 6 — LOSS-TERM REBALANCING
## Experiment Design (pre-registration)

**Phase:** 15 · **Experiment:** 6 of the Official Experiment Sequence
**Status:** PRE-REGISTERED — written before any Experiment 6 code exists
**Date:** 2026-08-06
**Authority:** `PHASE_10_ROADMAP.md` §5 A, §6 Exp 6, §7 · `PHASE_11_EVALUATION_PROTOCOL.md`
**Comparators:** `Baseline-CV` (Phase 11.6) — binding · `Exp 5` (Phase 14) — like-for-like reference · `Exp 3` (Phase 13) — reference
**Convention:** **[R]** roadmap text · **[E]** measured in a completed phase · **[I]** design decision

> **Pre-registration statement.** This document was written and frozen **before**
> any Experiment 6 code was written or executed. At the time of writing,
> `exp6_loss_rebalance/` does not exist and
> `trainer_outputs/exp6_loss_rebalance/` does not exist. No Experiment 6 result
> exists in any form. Appendix A records the SHA-256 of every frozen input; A.2
> is reserved for implementation digests and **must be completed before
> execution**, exactly as in Phases 12, 13 and 14.

> **Frozen-record statement.** Version 2 is closed. This document modifies
> nothing in Phases 10–14. Every value quoted from a completed phase is a
> read-only citation of a frozen artifact.

---

## 1. Objective

**[R]** Roadmap §6 Exp 6: *"**Loss-term rebalancing** · single changed factor: **loss scaling** · Objective: **Break the joint-collapse of both heads**."*

Determine whether the fixed **0.5 coefficient on the regression term** in the frozen composite objective is the operative cause of prior-collapse, by varying that coefficient alone while every other factor — including the epoch budget, the decision rule and the class weighting — is held at its Baseline-CV value.

## 2. Research question

The frozen trainer minimises, at `trainer_mentalbert_daic.py:324`:

```python
loss = loss_cls + 0.5 * loss_reg          # CE + λ·MSE, λ = 0.5
```

Both terms backpropagate through **one shared MentalBERT encoder** (`self.bert`, `:245`) which feeds two heads (`self.classifier` `:231`, `self.phq_mu` `:232`). The two objectives therefore compete for the same representation.

**Does the classification head fail to discriminate because it is being out-competed for encoder capacity by a regression term roughly 35 times its magnitude?**

## 3. Motivation — what Version 2 has established

**[E] Three interventions on the classification decision layer have all returned H₀:**

| Experiment | Factor | Verdict | What it excluded |
|---|---|---|---|
| Exp 4 (Phase 12) | decision threshold | **H₀** | Threshold relocation cannot recover discrimination |
| Exp 3 (Phase 13) | epoch budget | **H₀** | Convergence cannot recover accuracy |
| Exp 5 (Phase 14) | class weighting | **H₀** | Reweighting translates the probability band but does not dilate it |

**[E] The structural finding common to all three.** The held-out `p_pos` band width is **0.029216** under Baseline-CV and moves by less than 0.5 % across a 4.2× range of class weighting (Exp 5: 0.029216 / 0.029362 / 0.029236 / 0.029237). Exp 4 measured the same **0.0292**. Only Exp 3 moved it — to 0.3002, via more epochs — and that produced no accuracy gain.

**[E] Ranking survives every intervention.** ROC-AUC confidence intervals exclude 0.5 in all Exp 5 arms and in Baseline-CV (0.6333, CI [0.5755, 0.6911]). A real ordering exists; nothing tested so far converts it into decisions.

## 4. Scientific background — the quantitative case

**[E] The loss decomposition, computed from frozen artifacts only.**

At the collapsed operating point that Baseline-CV actually occupies:

| Term | Value | Share of total |
|---|---|---|
| Classification CE at the label prior, −[p·ln p + (1−p)·ln(1−p)], p = 0.2393617021276596 | **0.5503** | **2.79 %** |
| Regression 0.5 · MSE, from RMSE 6.1896 ⇒ MSE 38.3111 | **19.1555** | **97.21 %** |
| **Total** | **19.7059** | |

**[E] Ratio regression : classification = 34.8 : 1.**

**[E] Corroboration from an independent artifact.** Baseline-CV's mean final *training* loss is **21.1187** (`exp5_metrics.csv`, arm W0; identical to Exp 3's E0 arm). The held-out prediction above is 19.7059 — the same quantity to within the train/test gap, confirming the decomposition is not an artefact of the estimate.

**[I] This single number reinterprets all three prior H₀ results.**

- **Exp 5 re-weighted 2.79 % of the objective.** Class weighting scales `loss_cls`. Even W3's 4.18× positive weighting acted on a term contributing under 3 % of the gradient reaching the shared encoder. That the probability band *translated* by +0.1221 without *dilating* is exactly what a small perturbation to a dominated term predicts.
- **Exp 3's behaviour follows too.** Given 20 epochs, the optimiser continued descending the term that dominates: `pred_var` rose 0.0057 → **9.4247** (the regression head), while classification F1 reached only 0.0345. The budget was spent where the gradient was.
- **The immovable 0.0292 band is the signature of an encoder optimised for something else.**

**[I]** The coefficient 0.5 has no recorded justification anywhere in the project. It is a default, exactly as the 3-epoch budget was — and Exp 3 established that testing an unjustified default is worthwhile even when the result is H₀.

**[I]** The scale asymmetry is structural, not incidental: MSE is computed on a **0–23 PHQ scale**, so its natural magnitude is ~10², while binary CE is bounded near ~10⁰. Any fixed coefficient of order 1 will be dominated by the regression term. **This is the first experiment in Version 2 to target the shared representation rather than the classification head.**

## 5. Hypotheses

**H₀ (null).** Varying the regression coefficient λ over the pre-declared ladder produces **no detectable improvement in classification discrimination** relative to Baseline-CV, under the acceptance criteria of §11.

**H₁ (primary).** Reducing λ from 0.5 to parity (λ = 0.0144) frees encoder capacity for the classification objective, producing a **detectable increase in PR-AUC** and precision **above prevalence**, without degrading MAE beyond the pre-registered noise band.

**H₂ (secondary, roadmap-derived).** **[R]** *"Classification improves **and/or** the regression prediction spread widens."* Reducing λ widens the `p_pos` band beyond the immovable 0.0292 observed in Exps 4 and 5.

**[I] Directional prediction recorded in advance.** If the joint-collapse hypothesis is correct, classification metrics should improve **monotonically as λ decreases**, and the λ = 0 arm should bound the maximum achievable gain. A non-monotone result would falsify the capacity-competition mechanism even if some arm improves.

## 6. Single independent variable

**The scalar coefficient λ multiplying the regression MSE term, and nothing else.**

```python
loss = loss_cls + LAMBDA * loss_reg       # LAMBDA is the only quantity that varies
```

`loss_cls` is **unweighted** `CrossEntropyLoss()` — Exp 5 returned H₀, so class weighting was **not** adopted. `loss_reg` is unmodified `MSELoss()`. The PHQ target, its scale, the architecture, the optimizer and the tokenizer are untouched.

## 7. Coefficient ladder — pre-declared

| Arm | λ | Basis | Role |
|---|---|---|---|
| **L0** | **0.5** | the frozen default | **equivalence control** |
| **L1** | **0.0144** | **parity** — makes the two terms equal at Baseline-CV's operating point (0.5503 / 38.3111 = 0.014365, rounded to 4 s.f.) | **primary** |
| L2 | 0.1 | intermediate, 5× reduction | sensitivity |
| L3 | 0.0 | regression term disabled | **bound** — sensitivity |

**[I] Why parity is the primary.** It is the only value on the ladder derived from measurement rather than chosen for roundness: it is computed from two frozen quantities (CE at the label prior and Baseline-CV's held-out MSE) and equalises the two terms' contributions at the point the model actually occupies. It is fixed in this document before execution and will not be re-derived post hoc.

### 7.1 λ = 0.0144 is loss-value parity, NOT gradient parity [F5]

**[I] This limitation is stated explicitly because the design's mechanism and its anchor are not the same quantity.**

λ = 0.0144 equalises the two terms' **loss values** at Baseline-CV's operating point: λ·MSE = CE ⇒ λ = 0.5503 / 38.3111. What actually governs competition for the shared encoder, however, is the **gradient** each term contributes to `self.bert`:

```
λ · MSE  =  CE        does NOT imply        ‖∇_encoder (λ · MSE)‖  =  ‖∇_encoder CE‖
```

The two gradients depend on each term's curvature and on the Jacobians of two different heads (`phq_mu` and `classifier`), neither of which is measurable from the frozen artifacts. **The 34.8 : 1 figure in §4 is a ratio of loss magnitudes standing in for a ratio that has never been measured in this project.**

**[I] λ is nevertheless retained at 0.0144**, for three reasons: it is derived from frozen measurement rather than chosen; it is pre-registered here before execution and will not be re-derived post hoc; and the ladder (0.5 / 0.1 / 0.0144 / 0.0) brackets it across the full 35× span, so the sensitivity arms absorb the uncertainty in where true gradient parity lies. Selecting a different λ after observing gradient norms would be a post-hoc choice and is forbidden.

**[I] The gap is closed by measurement, not by argument.** §15 adds encoder gradient norms to `loss_decomposition.csv`, converting §4's central inference into a direct observation. That instrument is valuable whichever way the experiment resolves — see §15.1.

**[I] Why λ = 0 is included.** It bounds what loss rebalancing can possibly buy. If classification does not improve even when the regression term is removed entirely, the joint-collapse hypothesis is dead and the deficit lies in the representation itself — a decisive negative result. It also directly instruments the roadmap's stated failure condition. **[E]** MAE in this arm is expected to degrade sharply toward or past the mean-predictor bound of 4.9135; that is the arm's purpose, not a defect.

**[I]** The ladder spans the full 35× gap between the frozen default and parity, in a monotone sequence, so §5's monotonicity prediction is testable.

## 8. Controlled variables — all frozen at Baseline-CV values

| Factor | Value | Why this value |
|---|---|---|
| Epochs | **3 (reverted)** | Exp 3 returned H₀ — 20 epochs not adopted |
| Decision rule | **argmax, τ = 0.5 (reverted)** | Exp 4 returned H₀ — threshold policy not adopted |
| Class weighting | **uniform, `weight=None` (reverted)** | Exp 5 returned H₀ — weighting not adopted |
| Folds | `fold_manifest.json`, frozen | identical participants per fold as all prior phases |
| Repeats / folds | R = 5, k = 5 → 25 fold-runs per arm | identical to Baseline-CV, Exps 3, 4, 5 |
| Learning rate / batch / clip | 2e-5 / 8 / 1.0 | frozen trainer |
| Optimizer | `transformers.optimization.AdamW` | eps 1e-6, wd 0.0, correct_bias True |
| Model / tokenizer | `mental/mental-bert-base-uncased` | frozen trainer |
| Binarisation | PHQ > 10.0 | frozen protocol |
| Modality | text-only; audio_dim = vision_dim = 0 | roadmap §7.5 invariant |

## 9. Experimental protocol

1. Load the frozen fold manifest; verify every input SHA against Appendix A.1; **abort on any mismatch**.
2. For each arm L0 → L3, each repeat r ∈ 1..5:
   **Repeat-level seeding**, before entering the fold loop:
   ```
   seed = base_seed + r                # base_seed = 1000
   torch.manual_seed(seed)
   np.random.seed(seed)
   ```
   Then for each fold f ∈ 1..5, in this **exact order**:
   ```
   a.  torch.manual_seed(seed * 100 + f)        # per-fold reseed
   b.  model = MultiModalModel(...); model.to(device)
   c.  tr_ds, te_ds = MultiModalDataset(...)    # datasets
   d.  DataLoader(tr_ds, batch_size=8, shuffle=True, collate_fn=collate_batch)
   e.  training loop — 3 epochs, loss = loss_cls + LAMBDA * loss_reg
   f.  run_inference(model, te_loader) → {arm}_rep{r}_fold{f}_preds.csv
   g.  record the loss decomposition row (§15)
   ```
3. Consolidate after **every** (arm, repeat) pair — never only at the end.
4. Aggregate: repeats averaged **within fold first**, then fold-level Student-t, df = k−1 = 4, **t = 2.7764451051977987**.
5. Evaluate the **L0 equivalence gate** (§14) — hard abort on failure.
6. Evaluate acceptance criteria; write summary artifacts.
7. Independent audit re-deriving every result from raw predictions.

### 9.1 Seeding protocol — normative [F1]

**[E] The frozen driver seeds twice, and both are mandatory.**

| Point | Statement | Source |
|---|---|---|
| Per repeat | `torch.manual_seed(seed)` · `np.random.seed(seed)`, `seed = 1000 + r` | `run_baseline_cv.py:125-126` |
| **Per fold** | `torch.manual_seed(seed * 100 + f)`, **immediately before model construction** | `run_baseline_cv.py:145` |

**[E]** Experiment 5's runner replicated both (`run_exp5.py:580-581` and `:598`, the latter commented *"Per-fold reseed, identical to run_baseline_cv.py (L145)"*), and that is why its W0 arm was **bit-identical** to Baseline-CV.

**[I] The ordering seed → model → dataset/dataloader → training loop is normative, not stylistic.** `FusionHead` contains `nn.Dropout(0.2)` (`trainer_mentalbert_daic.py:230`) and the training `DataLoader` uses `shuffle=True`, so both consume the torch RNG stream. Any deviation in seeding *or* in the order in which these objects are constructed changes the stream and **will** break L0 bit-identity. This is the single most likely cause of a gate failure and is therefore specified here rather than left to implementation.

**[I] The consolidation rule at step 3 is not optional.** Experiment 3's first execution lost an entire arm because its skip path bypassed the accumulator. Experiment 5 designed the defect out; Experiment 6 inherits that design.

## 10. The one-factor problem — and how it is solved

**[I]** λ is a **literal inside** the frozen `fine_tune_supervised`, not a parameter of it. As in Experiment 5, the training loop must be reimplemented to vary it. Experiment 3 did not face this — `epochs` is a genuine parameter.

**The L0 arm is the guarantee.** L0 sets λ = 0.5, reproducing the frozen expression exactly. **[E]** In Experiment 5 the analogous W0 arm proved **bit-identical** to Baseline-CV — maximum absolute difference of **0** in both `pred_phq` and `p_pos` across all 940 held-out predictions. The same standard applies here.

**[I]** The reimplemented loop must be **line-for-line identical** to `fine_tune_supervised` except for the coefficient: same optimizer construction, same `DataLoader(shuffle=True)`, same `collate_batch`, same `clip_grad_norm_(1.0)`, same ordering of `zero_grad` / `backward` / `step`, and the seeding protocol of §9.1. A `loop_correspondence.md` artifact will document each line's provenance, as in Experiment 5.

**[I] The gate is the load-bearing element of this design.** If L0 does not reproduce Baseline-CV, no L1/L2/L3 result may be reported. This is stated as a hard abort in §14, not a warning.

### 10.1 Uniform code path — normative [F6]

**[I] All four arms must execute an identical code path.** The loss is computed unconditionally as:

```python
loss = loss_cls + LAMBDA * loss_reg       # LAMBDA ∈ {0.5, 0.1, 0.0144, 0.0}
```

**A `if LAMBDA == 0: loss = loss_cls` branch is explicitly forbidden.** Although the two forms produce identical gradients, a branch creates a code path that L0, L1 and L2 never traverse, so the L0 gate would no longer certify the path L3 actually runs. Uniformity is what makes the gate transitive across arms.

**[I]** `loss_reg` must additionally be checked finite before the backward pass: `0.0 * NaN = NaN`, so the λ = 0 arm is not automatically protected from a non-finite regression term by its coefficient.

### 10.2 Inherited behaviour — `phq_logsigma` [F7]

**[E]** `FusionHead` defines three heads — `classifier` (`:231`), `phq_mu` (`:232`) and `phq_logsigma` (`:233`) — but the frozen objective at `:324` consumes only `logits` and `reg_pred` (= `phq_mu`) via `CrossEntropyLoss` and `MSELoss`. **`phq_logsigma` receives no gradient from the supervised loss.**

**[I] This is inherited from the frozen trainer and is not introduced, altered or exercised by Experiment 6.** The head is untrained in Baseline-CV, in Exps 3, 4 and 5, and in all four Experiment 6 arms alike. It is recorded here only so that a reader does not mistake it for an artefact of loss rebalancing. It has no bearing on the one-factor rule: the head is equally untrained in the control and in every treatment arm.

## 11. Acceptance criteria

Pre-registered, in evaluation order. **C0 is a hard gate.**

**[I] Evaluation arm — normative [F3].** **Criteria C1–C6 are evaluated on the primary arm L1 (λ = 0.0144) only.** L2 and L3 are **sensitivity arms**: their metrics are reported in full and inform interpretation, but they do not determine the H₀/H₁ verdict. C0 is evaluated on L0 by construction.

| # | Criterion | Arm | Threshold | Source |
|---|---|---|---|---|
| **0** | **L0 reproduces Baseline-CV** on all 12 primary metrics (aggregate inside Baseline-CV's 95 % CI) | L0 | hard abort | design gate |
| 1 | **PR-AUC detectably above** Baseline-CV 0.3941 | **L1** | paired per-fold Δ > 0 and \|Δ\| > MDE | Roadmap §6 Exp 6 (*"classification improves"*) |
| 2 | Recall and F1 above Baseline-CV (0.0000 / 0.0000) | **L1** | Δ > 0 | Roadmap §6 Exp 6 |
| 3 | Recall and F1 above **Exp 5 W1** (0.0756 / 0.0295) | **L1** | Δ > 0 | Roadmap §6 Exp 6 comparator |
| 4 | **MAE not materially degraded** | **L1** | Δ ≤ MDE 0.4066 **and** MAE < mean-predictor 4.9135 | Roadmap §6 Exp 6 failure condition |
| **5** | **Discrimination, not quantity:** precision > prevalence 0.2394 **AND** F1 > random-ranker control | **L1** | both, per-fold-run control | carried forward from Exps 4 and 5 |
| **6** | Decisions non-degenerate — below Exp 5 W1's **24/25** | **L1** | strict | carried forward from Exps 4 and 5 |

**[I] Criteria 5 and 6 are the load-bearing ones and are non-negotiable.** Exp 4 showed Recall/F1 can rise with **no** discrimination; Exp 5 showed the same at argmax. Criterion 2 alone is a bare inequality against exact zeros that any positive prediction satisfies. **A result meeting Criteria 1–4 but failing 5 is H₀.**

**[I] On C3's information content.** Exp 5 W1's Recall 0.0756 and F1 0.0295 sit close enough to Baseline-CV's exact zeros that most results clearing C2 by a meaningful margin will also clear C3. C3 is retained for roadmap compliance (§11.2) but carries little independent weight in interpretation.

### 11.0 L3 regression metrics are descriptive only [F3]

**[E]** In arm L3, λ = 0, so `loss_reg` contributes exactly zero gradient and the `phq_mu` head receives **no supervision whatsoever**. Its `pred_phq` output is an untrained random projection of the encoder representation.

**[I] Therefore MAE, RMSE and prediction variance for L3 are descriptive only and must NOT be interpreted as regression performance.** They characterise an unsupervised head, not a degraded one. This is also why C4 is evaluated on L1 alone: applied to L3 it would fail by construction, reporting the arm's design rather than a finding. The same caution applies to any cross-arm regression comparison involving L3.

### 11.1-M Monotonicity — observational only [F2]

**[I] The λ-ordering analysis is NOT an acceptance criterion.** It was proposed as "C7" in an earlier draft and has been **demoted to a directional observational outcome**. It is reported in full and is mechanistically informative, but it **does not contribute to the H₀/H₁ verdict** and cannot cause acceptance or rejection.

**Reported as:** fold-level mean PR-AUC, Recall, F1 and `p_pos` band width for each arm, tabulated in decreasing λ order (L0 = 0.5 → L2 = 0.1 → L1 = 0.0144 → L3 = 0.0), with the observed ordering stated plainly.

**[I] Why it cannot be a criterion.** With four arms, one specific ordering arises by chance with probability 1/24 ≈ 4.2 % under the null — close enough to a conventional significance level that a pass/fail monotonicity test would be near coin-flip in both directions, and k = 4 affords no power for a formal trend test. Treating it as pass/fail would import exactly the kind of weakly-supported criterion that Exps 4 and 5 taught this project to distrust.

**[I]** This is the same treatment given to Experiment 3's distribution diagnostics and Experiment 5's degeneracy diagnostics: **report in full, treat as secondary observational outcomes, never pass/fail.**

### 11.1 On the roadmap's prediction-spread criterion

**[R]** Roadmap §6 Exp 6 offers *"classification improves **and/or** the regression prediction spread widens"* and calls predicted-value variance *"itself a success metric, independent of MAE."*

**[I] Prediction spread is reported in full but is NOT an acceptance criterion in this experiment.** **[E]** Experiment 3 measured `pred_var` rising from 0.0057 to **9.4247** — a 1,600-fold increase, the only detectable change in its entire paired comparison — while MAE did not detectably move and the experiment returned H₀. Spread is now known to be satisfiable without any accuracy or discrimination gain.

**[I]** This is the identical treatment applied to Experiment 3's distribution diagnostics and Experiment 5's: report fully, treat as secondary observational outcomes, never pass/fail. The roadmap is frozen and unmodified; what changes is the evidentiary weight assigned to one of its criteria, on evidence the roadmap could not have had.

### 11.2 Comparator policy — resolving review finding M3

**[E]** `VERSION2_FINAL_REVIEW.md` M3 records that the roadmap's chained comparators assume each experiment is *accepted*, an assumption three consecutive H₀ results left unmet. **[R]** Roadmap §6 Exp 6 names the comparator as *"`Baseline-CV` **and Exp 5**"*.

**[I] Policy adopted for this experiment, stated explicitly because the review flagged it as a decision requiring approval:**

1. **`Baseline-CV` is the binding comparator.** Exp 5 returned H₀ and contributed no accepted change, so the accepted configuration remains exactly Baseline-CV.
2. **Exp 5 W1 is a like-for-like reference** and Criterion 3 binds against it. Unlike Exp 4, **Exp 5 ran at argmax and 3 epochs — identical to this experiment** — so the comparison is genuinely controlled and needs no threshold caveat.
3. **Exp 3 E1 is a reference only**, at a different epoch budget; any comparison is reported budget-conditionally and binds nothing.

## 12. Failure criteria

The experiment is **invalid** (not merely negative) if any of these occur:

- **L0 fails the equivalence gate** → the reimplemented loop is not the frozen loop; no result may be reported.
- Modality ablation is non-zero for audio or video (roadmap §7.5 invariant).
- Any frozen input SHA changes during the run.
- Fold membership deviates from `fold_manifest.json`.
- `pred_class` is not exactly `p_pos ≥ 0.5` (the reverted decision rule).
- The optimizer resolves to anything other than `transformers.optimization.AdamW`.
- Artifact count ≠ the derived expectation.

**[I]** An H₀ outcome is a **result**, not a failure. Three H₀ results have already produced the strongest claims in Version 2.

## 13. Statistical methodology

Identical to Baseline-CV, Exp 3, Exp 4 and Exp 5, so all five remain mutually comparable:

- Repeats averaged **within fold first**, then fold-level Student-t, df = k−1 = **4**, **t = 2.7764451051977987**.
- Paired per-fold Δ against Baseline-CV's own per-fold values.
- **MDE re-estimated from Experiment 6's own fold spread**, never inherited from Baseline-CV's degenerate zero-variance columns.
- Roadmap MDE anchors retained for reference: **MAE 0.4066**, **PR-AUC 0.0950**.
- Random-ranker control evaluated **per fold-run** using that fold's own prevalence, then averaged — never from an averaged positive rate (Jensen's inequality; this convention reproduces Exp 4's stored 0.211901 exactly).
- Zero-variance guard for degenerate columns (Baseline-CV's Recall/F1 are exactly 0 in every fold).

## 14. Validation checks

**L0 equivalence gate — AUTHORITATIVE, hard abort:**

| Check | Requirement |
|---|---|
| L0 aggregate inside Baseline-CV 95 % CI | all 12 primary metrics |
| L0 vs Baseline-CV raw predictions | reported; bit-identity expected per Exp 5 precedent |
| Aggregation self-test | reproduces `baseline_cv_summary.json`, max diff 0.000e+00 |

**Per-run checks:** frozen SHAs · participant order matches manifest · train/test disjoint · `phq_bin` consistent with PHQ > 10.0 · `pred_class == (p_pos ≥ 0.5)` · λ recorded per fold-run equals the arm's declared value · modality ablation 0/0.

## 15. Required artifacts

All written **only** to `trainer_outputs/exp6_loss_rebalance/`.

| Artifact | Rows / count |
|---|---|
| `{arm}_rep{r}_fold{f}_preds.csv` | **100** (4 arms × 5 repeats × 5 folds) |
| `{arm}_rep{r}_metrics.csv` | **20** |
| `exp6_metrics.csv` | 100 rows |
| `loss_decomposition.csv` | 100 rows — mean `loss_cls`, `loss_reg`, `λ·loss_reg`, ratio, λ, **and encoder gradient norms (§15.1)** |
| `degeneracy_report.csv` | 100 rows |
| `l0_equivalence_check.json` | the authoritative gate result |
| `loop_correspondence.md` | frozen-loop ↔ reimplemented-loop line correspondence |
| `exp6_summary.json` / `.md` | aggregates, CIs, paired deltas, criteria verdicts |
| **Total** | **127** = 100 + 20 + 7 named |

**[I]** `loss_decomposition.csv` is new to this experiment and is its most important diagnostic: it measures directly the quantity §4 estimated from frozen artifacts, turning a computed inference into a measurement.

### 15.1 Encoder gradient norms — observational instrument [F5]

**[I]** `loss_decomposition.csv` additionally records, per fold-run, the mean over logged steps of:

| Column | Quantity |
|---|---|
| `grad_norm_cls` | ‖∇<sub>encoder</sub> `loss_cls`‖₂ |
| `grad_norm_reg` | ‖∇<sub>encoder</sub> (λ · `loss_reg`)‖₂ |
| `grad_norm_ratio` | `grad_norm_reg` / `grad_norm_cls` |

taken with respect to the shared `self.bert` parameters, on a fixed pre-declared subset of training steps (the first step of each epoch), via `torch.autograd.grad(..., retain_graph=True)` on each term separately. The logging must not alter the optimisation: the recorded gradients are computed for measurement only, and the parameter update continues to derive from the single combined `loss.backward()` of §10.1.

**[I] These measurements are strictly observational. They are NOT acceptance criteria and cannot influence the H₀/H₁ verdict**, which rests solely on C0–C6 as defined in §11. They are recorded because they resolve the §7.1 limitation empirically:

- **Under H₁** they show the mechanism operating — the ratio falls as λ falls, and discrimination follows.
- **Under H₀** they are decisive in a way the loss decomposition alone cannot be: they distinguish *"gradient competition was rebalanced and discrimination still did not follow"* — which would exhaust the objective-rebalancing hypothesis and point squarely at representation capacity — from *"loss-value parity never achieved gradient parity"*, a confound that would otherwise leave the null uninterpretable.

**[E]** Arm L3 (λ = 0) provides a zero-reference: `grad_norm_reg` must be exactly 0.0 there, which doubles as a correctness check on the instrument itself.

**[I]** Because these columns are observational, a defect in the gradient-logging path cannot invalidate the experiment's verdict — but the verifier must still confirm the L3 zero-reference and that logging did not perturb the optimisation, by checking L0 bit-identity (§14), which would break first if it had.

## 16. Rollback strategy

```
Remove-Item trainer_outputs\exp6_loss_rebalance -Recurse -Force
```

Frozen inputs are opened **read-only** and SHA-verified before and after; re-runs are idempotent. No Version 2 artifact, document or script is written to at any point. Nothing outside `trainer_outputs/exp6_loss_rebalance/` is created except the three implementation scripts.

## 17. One-factor justification

| Factor | Baseline-CV | Exp 6 | Changed |
|---|---|---|---|
| Regression coefficient λ | 0.5 | **0.5 / 0.0144 / 0.1 / 0.0** | ✅ **the single factor** |
| Epochs | 3 | 3 | no |
| Decision rule | argmax | argmax | no |
| Class weighting | uniform | uniform | no |
| Folds / seeds / LR / batch / clip / model | frozen | frozen | no |

**Exactly one factor differs from the accepted configuration.**

## 18. Verification strategy

Three scripts, executed in order, mirroring Phases 12–14:

1. `run_exp6.py` — trains all 100 fold-runs; verifies frozen SHAs and the optimizer identity at startup; consolidates after every (arm, repeat).
2. `aggregate_exp6.py` — evaluates the L0 gate (hard abort), aggregates, evaluates all 8 criteria.
3. `verify_exp6.py` — **independent auditor**: imports neither predecessor, nor torch, nor transformers; re-derives every metric from raw predictions; derives the expected artifact count rather than hard-coding it; independently re-evaluates the L0 gate.

**[I]** The derived-count rule is not stylistic. Hard-coded literals produced false failures in Exp 4 (`A3`) and Exp 3 (`J2`), both recorded as errata.

## 19. Expected outputs and interpretation

| Outcome | Reading |
|---|---|
| C0 fails | Loop not equivalent; **no result reportable** |
| C1–C7 met | **H₁ accepted** — loss-term dominance was the operative constraint; λ enters the accepted configuration |
| C5 fails while C2/C3 pass | **H₀** — more positives, no discrimination; the Exp 4 / Exp 5 pattern repeats one level deeper |
| C7 fails while some arm improves | mechanism falsified; improvement needs another account |
| L3 (λ = 0) shows no gain | **decisive H₀** — even removing regression entirely does not free discrimination; the deficit is in the representation, and the RC-1 objective block is exhausted |

**[I]** The final row is the highest-information outcome available. It would close the RC-1 loop with all four cheap interventions excluded and redirect the project to representation capacity — a well-supported conclusion rather than an assumption.

## 20. Implementation plan

Three scripts, **one at a time, each approved before the next**, exactly as Phases 12–14. **No Python will be written until this document is approved.** After each script is written its SHA-256 is computed and recorded in Appendix A.2; **execution may not begin until all three digests are recorded.**

## 21. Repository locations

```
exp6_loss_rebalance/run_exp6.py            (to be created)
exp6_loss_rebalance/aggregate_exp6.py      (to be created)
exp6_loss_rebalance/verify_exp6.py         (to be created)
trainer_outputs/exp6_loss_rebalance/       (created by the run — does not exist yet)
```

---

## Appendix A — Pre-registration record

### A.1 Frozen inputs and comparators (read-only)

Digests taken **before** any Experiment 6 code was written.

```
65b1902e2ba8dd996c6960db1a6595d84fd48500f4d8707cb79688093d7a230b  trainer_mentalbert_daic.py
9a241851760727779fcaa4d50041b8bdc4406fe6b66ddbbd7105f4ce4fc55f00  daic_records.parquet
b9a7a91f1bdd43c78d3cd1ee0c36e4ce3fb8be4b0971dcff3ff62ee5af8fca2f  trainer_outputs/baseline_cv/fold_manifest.json
2ecbfee3225910034a959fe949c7ad027cfa7d41e9bef8040374578ff8c5d572  trainer_outputs/baseline_cv/trivial_control_arm.csv
f065f5e1ff7f8e5ed4d122a8031c9cc12425802e756697d5512a915674871aac  trainer_outputs/baseline_cv/baseline_cv_summary.json
5defdae2a20d0abc164611e8cbe6b8034e3f65c593d8c9b3e38266a1800aa6f2  trainer_outputs/exp4_decision_rule/exp4_summary.json
9dd135b6b79af06a85e64a5aa8c1896d19df8059dd89c053e50e1cffe20af2f9  trainer_outputs/exp3_convergence/exp3_summary.json
70ad13ffc4db633419595761c0396fc20dd1b62400ee74238be1f06b73ed48d3  trainer_outputs/exp5_imbalance_objective/exp5_summary.json
```

**[I] The Exp 5 comparator is SHA-gated like every other frozen input [F4].** An
earlier draft deferred its digest to implementation time, which was inconsistent:
**Criterion C3 binds against Exp 5 W1's Recall 0.0756 and F1 0.0295**, so
`exp5_summary.json` is a load-bearing input to the acceptance decision, not
background context. It is now verified at startup on the same footing as the
fold manifest and the Baseline-CV summary, and a mismatch is an abort condition
under §12.

**Frozen constants used in this design, all cited from the artifacts above:**

```
prevalence                 0.2393617021276596
t_crit (df = 4)            2.7764451051977987
MDE anchors                MAE 0.4066 · PR-AUC 0.0950
Baseline-CV MAE            4.8127        mean-predictor bound  4.9135
Baseline-CV PR-AUC         0.3941        ROC-AUC  0.6333 [0.5755, 0.6911]
Baseline-CV pred_var       0.005734      p_pos band width      0.029216
Baseline-CV degeneracy     25/25         Exp 5 W1 degeneracy   24/25
Exp 5 W1 Recall / F1       0.0756 / 0.0295
Derived parity coefficient 0.5503 / 38.3111 = 0.014365 → λ = 0.0144
```

### A.2 Implementation code

```
5933bc29eff056c9f976c67cde51edb2b311461b3368bd276dc2024cbdd5c5f2  exp6_loss_rebalance/run_exp6.py
546ad22f89ef3581aeccbbf930286a2d7dc06cc7f0d009cc9559160675194799  exp6_loss_rebalance/aggregate_exp6.py
fc7c0f742cbd062728c17cc4a3cd53e226330d2cee3c640862d26694aa74d61d  exp6_loss_rebalance/verify_exp6.py
```

**`run_exp6.py` verification before recording:** compiles (`py_compile`); 47
internal consistency checks pass against the executable AST, with docstrings
stripped so prose cannot satisfy a structural assertion. **No `λ == 0` branch
exists in code** — the loss is a single unconditional statement outside any
conditional (§10.1), so the L0 gate certifies the identical path all four arms
run; `loss_reg` is checked finite every step because `0.0 * NaN = NaN`. Both
Baseline-CV seeding stages are reproduced in the normative order of §9.1
(`run_baseline_cv.py:125-126`, then `:145`), with exactly two `torch.manual_seed`
calls and the per-fold reseed immediately preceding model construction. `T.AdamW`
confirmed to be `transformers.optimization.AdamW` with a hard abort if it ever
resolves elsewhere; `nn.CrossEntropyLoss()` is unweighted and `nn.MSELoss()`
unmodified, so λ is the sole difference from the frozen loop. Gradient-norm
logging uses `torch.autograd.grad` **before** the combined backward: it re-runs no
forward pass, so the RNG stream is untouched, and writes nothing to `.grad`, so
the optimizer update still derives solely from `loss.backward()`. All six frozen
inputs are SHA-gated, including `exp5_summary.json` (C3's binding comparator),
and the write-containment guard covers all four frozen output directories.

**`aggregate_exp6.py` verification before recording:** compiles (`py_compile`);
37 checks pass; imports neither torch, transformers nor the frozen trainer. The
aggregation self-test reproduces `baseline_cv_summary.json` across 12 metrics at
**max |diff| 0.000e+00** and runs **before** the L0 gate and before any Exp 6
number is computed. A second self-test validates the ungated `exp5_metrics.csv`
— needed for the per-fold Exp 5 comparison — against the SHA-gated
`exp5_summary.json` at **0.000e+00**, so the gated artifact certifies the ungated
one, and it confirms the C3 constants 0.0756 / 0.0295 are what Experiment 5
actually recorded. `fold_level_ci` and `paired_test` were verified **numerically
identical** to `aggregate_exp5.py` over 200 random draws (max |diff| 0.000e+00),
evidencing that no statistical methodology changed. The L0 gate is an
authoritative hard abort; C1–C6 were verified to source only from `PRIMARY_ARM`;
and `overall_verdict` is computed from C0, C1 and C5 alone, with the two
observational blocks provably absent from that expression.

**`verify_exp6.py` verification before recording:** compiles (`py_compile`); 33
checks pass; imports neither torch, transformers, the frozen trainer, nor either
script under test — every constant is restated independently, so a disagreement
with the runner or aggregator surfaces as a finding rather than passing silently.
The expected artifact count is **derived** from the run design (7 named + 100
preds + 20 metrics = 127), never a literal, following the false failures the Exp 4
`A3` and Exp 3 `J2` literals produced. Primitives validated against real prior
artifacts: `fold_ci` reproduces `baseline_cv_summary.json` at **0.000e+00**; the
per-fold-run random-ranker control reproduces Experiment 4's stored **0.211901**
exactly; the C3 constants match Experiment 5's record; and the primitives are
numerically identical to `verify_exp5.py` over 200 draws. The audit additionally
re-derives the L0 gate, asserts the **L3 zero-reference** (`grad_norm_reg` exactly
0 at λ = 0, which is simultaneously a physics check and an instrument check), and
**mechanically enforces observational isolation** by recomputing the verdict from
C0/C1/C5 alone and requiring the recorded verdict to match.

**[E] Pre-registration complete.** All three implementation digests are recorded
above, before any execution. `trainer_outputs/exp6_loss_rebalance/` does not exist
at the time of this record.

---

*Design and pre-registration only. No experiment has been executed, no artifact
created, no Version 2 document or output modified. Every **[E]** value is a
read-only citation of a frozen Phase 11–14 artifact; every **[R]** value is quoted
from the frozen roadmap; every **[I]** statement is a design decision.*
