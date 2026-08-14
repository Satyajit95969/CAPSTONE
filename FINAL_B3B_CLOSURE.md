# FINAL B-3B CLOSURE

**Status: CLOSED.** Verdict recorded by the frozen acceptance logic, verbatim:

> `H0`  —  `A=True B=True C=False`

Authoritative specification: `PHASE_21_B3B_DESIGN.md` (FROZEN, unmodified,
SHA-256 `00f8e34bb505f06b02466dbb82b679e688bbb4bdf394dadaab8c67d59c820590`).

This document is derived solely from the completed B-3B artifacts under
`trainer_outputs/b3b_probe/` and the independent verification. No result has been
modified, re-run or reinterpreted.

---

## 1. Experiment identification

| Field | Value |
|---|---|
| Phase / item | Phase 21 / B-3B, roadmap **Row 9**, **second** B-3 attempt |
| Weakness | RC-4 |
| Single changed factor | the **representation family** (vs Experiment 9) |
| Design | `PHASE_21_B3B_DESIGN.md`, SHA `00f8e34b…c820590` |
| Runs | 5 repeats × 5 folds = **25 trainings**, each yielding both arms |
| Runtime | 22.5 s (terminal, CPU) |

## 2. Objective

Does a compact, task-trained DAIC-derived probe — whose transmitted object
*itself* contains the task signal — survive the frozen DP mechanism better than
the Family-O coordinate-prefix representation that failed in Experiment 9?

Experiment 9 achieved A ∧ B but never C. B-3B removes Exp 9's coordinate-selection
step entirely: there is no mask, no index set and no selection of any kind.

## 3. Frozen design summary

| Field | Value |
|---|---|
| Representation | MentalBERT CLS, revision `24809aa822c76639760d0d934742d1b42f89942f`, `max_length` 128, `truncation=True`, `padding="max_length"`, `last_hidden_state[:, 0, :]`, 768-d, 188 participants |
| Architecture | `FusionHead(in_dim=768, hidden=256, num_classes=2)` — **197,892 params in 8 tensors** |
| Objective | `CrossEntropyLoss(logits, label) + 0.5 * MSELoss(mu, phq)`, label = PHQ > 10.0 |
| Ranking score | `softmax(logits)[:, 1]` (positive-class probability) |
| Recipe | `transformers.AdamW`, lr **1e-3**, **3** epochs, batch **8**, shuffle, dropout **0.2**, grad clip **1.0** |
| Transmitted object | genuine delta `w_after − w_before` via the frozen `compute_state_delta` |
| DP | Gaussian, ε **5.302585092994046**, δ **1e-05**, C **1.0**, nm **1.0**, T = 1 |
| Evaluation | frozen `fold_manifest.json`, 5 folds, participant-level stratification on PHQ > 10, ROC-AUC, repeats averaged within fold, df = 4 |

Every value was fixed **before execution**. No parameter was tuned at any point.

## 4. Arms

| Arm | Definition |
|---|---|
| **N** | no-DP control — reconstruct `w_before + delta`, evaluate |
| **D** | DP arm — clip `delta` to C = 1.0, add Gaussian noise, reconstruct, evaluate |

Both arms derive from the **same** training run per (repeat, fold), so they are
identical in embeddings, partitions, seeds, initialisation, training and
evaluation, **differing only in whether DP is applied.**

## 5. Actual results — arm N (no-DP control)

| fold | 1 | 2 | 3 | 4 | 5 |
|---|---|---|---|---|---|
| ROC-AUC | 0.575862 | 0.647510 | 0.537931 | 0.531746 | 0.502778 |

**Mean 0.559165298**, fold SD 0.055825076.

## 6. Actual results — arm D (DP)

| fold | 1 | 2 | 3 | 4 | 5 |
|---|---|---|---|---|---|
| ROC-AUC | 0.534866 | 0.499234 | 0.478927 | 0.540079 | 0.462698 |

**Mean 0.503160920**, fold SD 0.033941449.

## 7. Criterion A — payload

```
A:  complete transport payload  <=  1,579,963 B
```

| Field | Value |
|---|---|
| d | 197,892 |
| Raw (4 B/param) | 791,568 B |
| Transport (× 1.3329, Exp 9 §17 measured factor) | **1,055,101 B** |
| Threshold | 1,579,963 B |
| **Result** | **PASS** |

## 8. Criterion B — noise / NSR, and clipping

```
B:  measured mean NSR  <  544.341809
```

| Field | Value |
|---|---|
| Analytical √d | 444.850537 |
| **Measured mean NSR** | **444.876535** (SD 0.146178) |
| Deviation vs analytical | +0.006 % |
| Threshold | 544.341809 |
| **Result** | **PASS** |

**Clipping — measured, never tuned (design §13):**

| Field | Value |
|---|---|
| Updates clipped | **25 / 25 (100 %)** |
| Pre-clip norm range | 2.7879 – 3.1956 |
| Pre-clip norm mean | 2.9731 |
| Post-clip norm | 1.0 (bound bound on every run) |
| Clipping was binding | **Yes** |

## 9. Criterion C — task signal

```
C  =  C1 AND C2
C1:  ROC-AUC mean inside [0.5755177227457472, 0.6911434704671701]
C2:  paired within-fold degradation vs the no-DP control <= 0.05781287386071144
```

| Part | Measured | Threshold | Result |
|---|---|---|---|
| **C1** | DP ROC-AUC mean **0.503160920** | CI floor **0.5755177227** | **FAIL** |
| **C2** | paired degradation **+0.056004379** | MDE **0.0578128739** | **PASS** |
| **C** | C1 ∧ C2 | — | **FAIL** |

The DP arm falls **0.0724** below the CI floor and sits essentially at chance.

Paired degradation by fold: +0.040996, +0.148276, +0.059004, −0.008333, +0.040079.

> **C2's PASS is not evidence that DP preserved signal.** Degradation stayed
> within the MDE in part because the no-DP control itself carried little signal
> to lose (§11). This is recorded as measured, without favourable interpretation.

## 10. A/B/C conjunction and verdict

| | Criterion | Result |
|---|---|---|
| **A** | payload ≤ 1,579,963 B | **PASS** |
| **B** | mean NSR < 544.341809 | **PASS** |
| **C** | ROC-AUC in CI **and** degradation ≤ MDE | **FAIL** |

```
SUCCESS = A AND B AND C     ->     NOT SATISFIED
VERDICT: H0
```

No threshold was moved, added or invented at any point.

## 11. The registered dominant risk did NOT materialise

Design §16.1 registered, in advance, that the clipping bound might not bind:
C-2 had measured probe delta norms of 0.4505–0.4712 with **0 / 50** clipped, which
would have given a realised NSR ≈ 989 and failed B on arithmetic alone.

**That did not occur.** The frozen lr = 1e-3 over 3 epochs with batched updates
produced deltas of norm ≈ 2.97 — roughly **6.6×** the clipping bound — so clipping
bound on **every one of the 25 runs** and criterion B passed on its own merits.

The learning rate was frozen **before** execution (design §6) and was **not**
revisited afterwards. **B-3B did not fail for the reason the design most feared.**

## 12. Repeat structure — genuine stochasticity

Design §11 justified 5 repeats on the grounds that `shuffle=True` and
`Dropout(0.2)` inject stochasticity, unlike C-2's bit-deterministic full-batch
probe.

**Confirmed empirically:** mean within-fold SD across repeats = **0.059839 > 0**
(arm N 0.053516, arm D 0.066162). The repeats genuinely varied. This is **not**
pseudo-replication, and the fold-level aggregation is sound.

## 13. Independent verification status

**20 PASS, 0 FAIL, 0 PENDING** — `trainer_outputs/b3b_probe/verify_b3b_report.json`.

Fail-closed: missing evidence counts as PENDING against the verdict, never for it.
The verifier re-derived the result rather than trusting the runner:

- **retrained all 25 folds × 2 arms from scratch** with an independent
  implementation and reproduced all **50** recorded ROC-AUC values with
  **max |diff| = 0.000e+00**
- recomputed A, B, C1, C2 and the overall verdict independently
- recomputed clipping counts and NSR independently
- recomputed ε live via the frozen `dp_agent`
- confirmed the MDE equals `baseline_cv_summary.json` and that **no acceptance
  threshold was moved, added or invented**
- confirmed the summary does **not** claim Row 10 is unblocked

## 14. Frozen SHA integrity status

All **7** pinned frozen inputs verified **before and after** execution — unchanged:
`daic_records.parquet`, `fold_manifest.json`, `baseline_cv_summary.json`,
`trainer_mentalbert_daic.py`, `dp_agent/dp_agent.py`, `PHASE_18_EXP9_DESIGN.md`,
and the reused embedding artifact.

The C-2 embedding artifact
(`trainer_outputs/c2_multiclient/c2_embeddings.npz`, SHA
`70df4e49b4fea83c2464eba3e41e5b8ffa856fff4c70dc7b7b8d054cf488ed3c`) was reused
**read-only**. **C-2 was not re-run and no C-2 artifact was modified.**

## 15. Final H0 interpretation — the failure is upstream of DP

**The decisive diagnostic is the no-DP control.** Arm N — the identical probe with
**no privacy noise at all** — scored **0.559165298**, which is **also outside** the
frozen CI (floor 0.5755177227).

> **The compact task-trained probe failed to reach the frozen baseline CI before
> any DP was applied.** The limiting factor is the frozen pretrained MentalBERT
> CLS representation, not the privacy mechanism.

This is design §16 risk **#2** materialising, not risk #1. B-3B therefore did not
cleanly test *"does a compact object survive DP"* — the object carried
insufficient signal to begin with.

**H₀ here is a legitimate, pre-registered scientific result, and is NOT an
implementation failure.** Every integrity, structural and implementation criterion
passed: 20/20 independent verification, bit-exact reproduction of all 50 values,
frozen artifacts unchanged, criteria A and B both met, and nothing tuned.

**Established finding:** a compact task-trained head over frozen pretrained
MentalBERT CLS embeddings does **not** reach the frozen Baseline-CV ROC-AUC
interval, with or without the frozen DP mechanism.

## 16. Limitations

1. The failure is upstream of DP (§15); B-3B does not isolate DP's contribution.
2. C2's PASS is not favourable evidence (§9).
3. Small sample: 188 participants, 45 positive, df = 4; the frozen MDE 0.0578 is
   large relative to a 0.6333 baseline.
4. The payload is a **predicted** transport figure using Exp 9's documented
   ×1.3329 factor, not a live encryption run. The margin is large enough that the
   verdict does not depend on it.
5. Single pre-registered architecture, **no dimension ladder** (design §4). H₀
   establishes only that **this** compact object fails — not that no compact
   DAIC-derived object survives DP.
6. Fold 2 contributes the largest degradation (+0.148276 against +0.040996,
   +0.059004, −0.008333, +0.040079); at 5 folds this is not separable from noise.

## 17. Roadmap consequence

**B-3 remains UNSATISFIED.** Row 9 has now returned H₀ **twice**, across two
structurally different representation families:

| attempt | family | verdict |
|---|---|---|
| Experiment 9 | public architecture-priority coordinate-prefix mask | H₀ |
| **B-3B** | compact task-trained probe over frozen CLS embeddings | **H₀** |

**Row 10 / C-1 remains BLOCKED.** `PHASE_10_ROADMAP.md:234` requires **B-2 and
B-3** to succeed. B-2 (Experiment 8) returned H₀; B-3 has now returned H₀ twice.
**Neither precondition is satisfied.**

C-3 and C-4 remain blocked behind C-1.

> B-3B does **not** unblock Row 10, and no claim to the contrary is made anywhere
> in its artifacts. `b3b_summary.json` records `b3_satisfied: false` and
> `row_10_unblocked: false`, and independent verification check 18 enforces this.
