# PHASE 21 / B-3B — SECOND B-3 ATTEMPT: COMPACT TASK-TRAINED PROBE
## Experiment Design — **FROZEN PRE-REGISTRATION**

> **Status: APPROVED AND FROZEN.** No parameter, threshold, partition, seed,
> metric, architecture or scope in this document may be changed after any result
> is observed.

> **This is a SECOND B-3 attempt. Failure is a valid scientific result.** The
> roadmap designates B-3 *"the highest-risk item … and the most likely to fail."*
> This design does **not** assume or predict success. §16 lists, in advance, the
> strongest reasons it may return H₀.

> **B-3B does NOT unblock Roadmap Row 10 by itself.** Row 10 requires B-2 **and**
> B-3 to succeed. B-2 (Exp 8) returned H₀ and is closed as an approach. Even a
> fully successful B-3B leaves Row 10 blocked. See §15.

**Tag legend** — every value carries one:
**[A]** inherited from a frozen source · **[B]** repository-supported ·
**[C]** newly approved decision · **[U]** unresolved (none remain).

---

## 1. Identity and research question

| Field | Value | Tag |
|---|---|---|
| Phase / item | Phase 21 / B-3B, roadmap Row 9 (second attempt) | **[A]** |
| Weakness | RC-4 | **[A]** |
| Single changed factor | **the representation family** (vs Exp 9) | **[C]** |

**Research question [C]:**

> Does a compact, task-trained DAIC-derived probe — whose transmitted object
> *itself* contains the task signal — survive the frozen DP mechanism better than
> the Family-O coordinate-prefix representation that failed in Experiment 9?

Formally: **is there a compact DAIC-derived object satisfying A (payload),
B (noise) and C (task signal) simultaneously?** Exp 9 achieved A ∧ B but never C.

## 2. Why this is a genuinely different B-3 attempt

Exp 9 failed at **coordinate selection**, not at compression or noise. K2 and K3
**passed** payload and SNR; every masked arm's ROC-AUC CI straddled chance:

| arm | k | A payload | B SNR | ROC-AUC 95 % CI | C task |
|---|---|---|---|---|---|
| K0 | 109,680,132 | FAIL | FAIL | [0.5754, 0.6911] | **PASS** |
| K1 | 1,096,801 | FAIL | FAIL | [0.4169, 0.5953] | FAIL |
| K2 | 295,681 | **PASS** | **PASS** | [0.4159, 0.5941] | FAIL |
| K3 | 10,968 | **PASS** | **PASS** | [0.4039, 0.5808] | FAIL |

**The mechanism B-3B removes [C].** Exp 9's K2 reconstruction was *pretrained
encoder + fine-tuned fusion head*, because coordinates outside `S_k` received a
zero delta and the encoder reverted to base. That head had been trained **jointly
with the encoder**, so it had learned to read a *fine-tuned* embedding space;
reverting the encoder left head and features **mismatched**. B-3B trains the head
**directly on the frozen pretrained CLS embeddings** — head and feature space are
**matched by construction**. There is no mask, no index set, no selection step.

**No repository experiment has ever measured whether a probe on frozen pretrained
CLS carries DAIC signal.** Exp 10's M0/M1/M2 are all full 109.7 M-parameter
fine-tuned models (`PHASE_19_EXP10_DESIGN.md:116–118`), not frozen-embedding
probes. That gap is exactly what B-3B closes. **[B]**

## 3. Representation — inherited, nothing invented

| Field | Value | Tag |
|---|---|---|
| Model | `mental/mental-bert-base-uncased` | **[A]** |
| Revision | `24809aa822c76639760d0d934742d1b42f89942f` | **[A]** |
| max_length | **128** | **[A]** |
| truncation | **True** | **[A]** |
| padding | **"max_length"** | **[A]** |
| Pooling | `last_hidden_state[:, 0, :]` (CLS) | **[A]** |
| Dimension | **768** | **[A]** |
| Population | **188** DAIC participants | **[A]** |

**Artifact reuse [C]:** the embeddings are the C-2 artifact
`trainer_outputs/c2_multiclient/c2_embeddings.npz`
(SHA `70df4e49b4fea83c2464eba3e41e5b8ffa856fff4c70dc7b7b8d054cf488ed3c`),
produced under **exactly this configuration**, verified 8/8 on its integrity
checkpoint and shown to reproduce **bit-exactly** from the pinned revision
(max |diff| 0.000e+00). Re-extraction would recompute an identical array. The SHA
is verified before use; **C-2 is not re-run and none of its artifacts are
modified.**

## 4. Architecture — repository-supported, frozen

`FusionHead(in_dim=768, hidden=256, num_classes=2)` —
`trainer_mentalbert_daic.py:225–239`. **[B]**

```
fc1        Linear(768, 256)   →  ReLU  →  Dropout(0.2)
classifier Linear(256, 2)     ← positive-class probability feeds ROC-AUC
phq_mu     Linear(256, 1)     ← regression mean
phq_logsigma Linear(256, 1)
```

| tensor | params |
|---|---|
| `fc1.weight` / `fc1.bias` | 196,608 / 256 |
| `classifier.weight` / `.bias` | 512 / 2 |
| `phq_mu.weight` / `.bias` | 256 / 1 |
| `phq_logsigma.weight` / `.bias` | 256 / 1 |
| **total** | **197,892 in 8 tensors** |

This equals `fusion.*` in Exp 9 §6.1 (*"8 keys, 197,892 params"*) exactly. **[B]**

**No dimension ladder.** One pre-registered architecture only. **[C]**

## 5. Training objective — repository-supported, frozen

```
loss = CrossEntropyLoss(logits, label) + 0.5 * MSELoss(mu, phq)
```

`trainer_mentalbert_daic.py:312–324`; λ = 0.5 matches Exp 9 §20. **[B]**

- `label` = **PHQ_binary = phq_score > 10.0** (`BINARIZE_THRESHOLD 10.0`) — 45 pos
  / 143 neg. **[A]**
- `mu` = the `phq_mu` head against continuous `phq_score`. **[B]**

**ROC-AUC is computed from the positive-class classifier probability**
`softmax(logits)[:, 1]`. The regression term only shapes the shared `fc1`; it is
never used as the ranking score. **[C]**

CE-only was considered and **rejected**: it would leave `phq_mu` and
`phq_logsigma` (514 params) at zero gradient — dead dimensions still absorbing DP
noise — and it has no repository precedent. **[C]**

## 6. Training recipe — every value repository-supported

| Field | Value | Source | Tag |
|---|---|---|---|
| Optimizer | `transformers.AdamW` | `:34`, `:311` | **[B]** |
| Learning rate | **1e-3** | `create_dp_comparison.py:1380` — the repository's only head-on-frozen-features precedent (used by C-2) | **[C]** |
| Epochs | **3** | `:507` CLI default; Exp 9 §20 frozen recipe | **[B]** |
| Batch size | **8** | `:308`, `:508`, Exp 9 §20 — all agree | **[B]** |
| Shuffle | **True** | `:310` | **[B]** |
| Dropout | **0.2** | `:229` | **[B]** |
| Gradient clipping | **1.0** | `:326`; Exp 9 §20 | **[B]** |

**On the learning rate [C]:** 2e-5 (the other repository value) is a *BERT
fine-tuning* rate for a 109.7 M-parameter encoder. 1e-3 is the rate the
repository uses when training a **head on frozen features**. This choice was made
**before execution** and **may never be changed on the basis of observed
results** (§12, §17).

**No repository pathway trains FusionHead alone on precomputed embeddings**
(verified: no `fusion.parameters()`, `head.parameters()`, `freeze` or
`requires_grad` usage exists). `fine_tune_supervised` optimises
`model.parameters()` end-to-end from `input_ids`. **The hyperparameter values are
inherited; the training pathway is necessarily new code.** Stated openly. **[C]**

## 7. Initialisation

`w_before` = the **seeded random initialisation** of `FusionHead`, constructed
fresh per (repeat, fold). **[C]**

`trainer_outputs/local_probe_base.pt` **cannot** be reused: it is a
295,681-parameter `fc1`/`fc2` object, incompatible in both key set and shape with
FusionHead's 197,892 parameters in 8 tensors. Exp 9 likewise took its `fusion.*`
delta against a randomly initialised head. **[B]**

## 8. Transmitted object

```
delta = compute_state_delta(w_before, w_after)      # w_after − w_before
```

The **frozen** implementation, `trainer_mentalbert_daic.py:348`. **[A]**

Full post-training state is **NOT** transmitted. Exp 9 §20.2 records that Exp 8's
norms came from a full-state object mislabelled a delta
(`create_dp_comparison.py:1163–1167`) and must not be compared to genuine deltas.
**[A]**

## 9. Arms — DP and a matching no-DP control

| Arm | Definition |
|---|---|
| **N** (control) | reconstruct `w_before + delta`, evaluate. **No DP.** |
| **D** (treatment) | clip `delta` to C = 1.0, add Gaussian noise, reconstruct `w_before + noisy_delta`, evaluate. |

The arms share **identical** embeddings, partitions, seeds, initialisation,
training and evaluation. **They differ only in whether DP is applied.** **[C]**

The control is required because `baseline_cv_summary.json` persists **aggregate
statistics only** (`ROC_AUC.fold_sd`, means, MDE) and contains **no per-fold
ROC-AUC values**, so the §11 paired within-fold comparison has no other
reference. **[B]**

## 10. DP — inherited and frozen, unchanged

| Field | Value | Tag |
|---|---|---|
| Mechanism | Gaussian output perturbation | **[A]** |
| ε per update | **5.302585092994046** | **[A]** |
| δ | **1e-05** | **[A]** |
| Clipping norm C | **1.0** | **[A]** |
| Noise multiplier | **1.0** | **[A]** |
| σ_eff | **1.0** | **[A]** |
| Composition | RDP single-composition, **T = 1** | **[A]** |
| Accounting | the frozen `dp_agent/dp_agent.py` conversion, recomputed live | **[A]** |

Roadmap Row 8 is binding: *"A utility gain bought by weakening an already-weak
budget is not a gain — the privacy guarantee is the constraint, not the
objective."* **No privacy parameter may be tuned.** **[A]**

## 11. Evaluation — Exp 9's framework, inherited

| Field | Value | Tag |
|---|---|---|
| Fold manifest | `trainer_outputs/baseline_cv/fold_manifest.json`, SHA `b9a7a91f…8fca2f` | **[A]** |
| Protocol | participant-level stratified 5-fold | **[A]** |
| Generator | `StratifiedKFold(n_splits=5, shuffle=True, random_state=42)` | **[A]** |
| Stratified on | PHQ_binary (PHQ > 10) | **[A]** |
| Splits | train/test 150/38, 150/38, 150/38, 151/37, 151/37; zero overlap | **[A]** |
| Metric | **ROC-AUC** from positive-class probability | **[A]** |
| Structure | 5 folds × 5 repeats = **25 runs per arm** | **[A]** |
| Aggregation | repeats averaged **within** fold; fold-level statistics | **[A]** |
| df / t_crit | **4** / **2.7764451051977987** | **[A]** |
| Seeding | `seed = 1000 + repeat`; per fold `torch.manual_seed(seed*100 + fold)` **before model construction**; order seed → model → dataset → dataloader → loop is **normative** | **[A]** |

**5 repeats are scientifically justified here, not pseudo-replication [C]:**
`shuffle=True` (`:310`) and `Dropout(0.2)` (`:229`) both inject stochasticity, and
at batch_size 8 over ~150 training participants there are ~19 shuffled batches per
epoch. This is materially unlike C-2, whose full-batch, dropout-free probe was
bit-deterministic and whose repeats were therefore degenerate. **The runner
measures and reports whether the repeats actually varied**; if they do not, that
is reported as a limitation and no cross-repeat dispersion is claimed.

## 12. Success criteria — the frozen B-3 A/B/C framework

| | Criterion | Threshold | Tag |
|---|---|---|---|
| **A** | complete transport payload | **≤ 1,579,963 B** | **[A]** |
| **B** | measured mean NSR | **< 544.341809** | **[A]** |
| **C** | ROC-AUC mean inside the frozen CI **AND** paired within-fold degradation vs the no-DP control ≤ MDE | CI **[0.5755177227457472, 0.6911434704671701]**, MDE **0.05781287386071144** | **[A]** |

```
SUCCESS  =  A AND B AND C          (all three, on the DP arm)
H0       =  any of A, B, C fails
```

All three are reported **separately regardless of the conjunction**, because a
partial pattern is itself the boundary measurement (Exp 9 §25). **No threshold,
margin or criterion may be invented, moved or added after results are seen.**
H₀ is a legitimate scientific result, not an implementation failure. **[A]**

## 13. Clipping — measured, never tuned

Pre-registered reporting, **before any result is seen** [C]:

- fraction of updates clipped (`‖delta‖₂ > C`)
- pre-clip norm distribution (mean, min, max, SD)
- post-clip norm
- resulting effective NSR = `‖noise‖₂ / ‖clipped delta‖₂`

> **BINDING.** If the updates fail to reach the clipping bound, the training
> recipe **must not** be changed in response. That outcome is reported as a
> measured property and, if it causes B to fail, as H₀.

## 14. Privacy and leakage

The Exp 9 §3 defect — releasing a **data-dependent index set** in the clear,
giving disjoint support and ε = ∞ on the index component — is **structurally
absent**. B-3B releases the complete parameter vector of a **fixed, publicly
declared architecture**: no mask, no index set, no top-k, no selection. Only
values travel, and values are exactly what the Gaussian mechanism bounds. **[C]**

> **BINDING.** Architecture, dimensionality and every element of the transmitted
> structure are fixed **a priori**, before any participant outcome is examined.
> No data-dependent mask, top-k selection, architecture selection or index
> release is permitted at any point. Selecting any of them by validation
> performance would make the structure data-dependent and reintroduce the §3
> defect at design level, exactly as Exp 9 §6.3 warns. **[C]**

Metadata check: the fold manifest, the binarisation threshold and the
architecture are public project constants. No per-participant information is
released through structure or metadata. **[C]**

## 15. Roadmap dependency — B-3B alone does not unblock Row 10

`PHASE_10_ROADMAP.md:234` — Row 10 / C-1 *"is structurally impossible until B-2
and B-3 succeed."* **B-2 (Exp 8) returned H₀** and is closed as an approach by its
own established finding: *"at fixed ε and fixed d, a privacy-equivalent mechanism
change alone cannot materially improve post-noise SNR."*

> **Therefore a successful B-3B satisfies only the B-3 side of the dependency.
> Row 10 remains BLOCKED unless the roadmap's dependency rule is formally
> reconsidered — which this experiment does not do and may not assume.** **[A]**

`PHASE_10_ROADMAP.md` is **not modified** by this experiment. **[C]**

## 16. Scientific risks — registered in advance

1. **★ The clipping bound may not bind — the dominant risk.** Exp 9's measured
   NSR matched √d because a 109.7 M-parameter delta has norm ≫ C = 1.0, so
   clipping normalised it. **C-2 measured probe delta norms of 0.4505–0.4712 with
   0/50 clipped.** If B-3B's delta behaves similarly, realised NSR
   ≈ 444.85/0.45 ≈ **989**, which **fails B outright**. lr = 1e-3 over 3 epochs
   with batched updates should produce a larger delta than C-2's single
   full-batch step, but **this is a prediction, not a guarantee**, and the recipe
   is frozen either way.
2. **The pretrained CLS space may not be separable for depression.** If frozen
   MentalBERT CLS carries no PHQ signal, the probe has nothing to transmit and C
   fails regardless of DP.
3. **A 197,892-parameter head is not obviously better than K2's 295,681.** The
   defence is the head/feature *matching* argument (§2), which is reasoned but
   not yet measured.
4. **Sample size.** 188 participants, 45 positive, fold-level df = 4. The frozen
   MDE 0.0578 is large relative to a 0.6333 baseline.
5. **The dropout/shuffle stochasticity may be small**, leaving repeats
   near-degenerate despite §11's justification.
6. **The roadmap designates B-3 the item most likely to fail.** A second H₀ is a
   realistic and acceptable outcome.

## 17. No post-hoc changes — BINDING

1. Architecture, `in_dim`, `hidden`, `num_classes` are fixed. No ladder.
2. lr, epochs, batch size, dropout, clipping, optimizer are fixed.
3. The loss and λ = 0.5 are fixed.
4. All DP parameters are fixed.
5. The fold manifest, seeds and repeat structure are fixed.
6. A, B and C and their thresholds are fixed. **None may be moved or added.**
7. Clipping behaviour is **measured, never tuned** (§13).
8. A failed conjunction is reported as **H₀**, never re-tuned.
9. The representation and its pinned revision are fixed.

## 18. Artifacts

1. `b3b_embeddings_receipt.json` — reused C-2 embedding SHA + verification
2. `b3b_runs.csv` — all 50 runs (2 arms × 5 repeats × 5 folds)
3. `b3b_deltas.json` — per-run norms, clipping, NSR
4. `b3b_roc_auc.json` — per-fold / per-repeat ROC-AUC, both arms
5. `b3b_payload.json` — payload measurement
6. `b3b_receipt.json` — frozen SHAs, environment, protocol
7. `b3b_summary.json` — A/B/C verdict
8. `b3b_acceptance.csv` — the A/B/C row
9. `verify_b3b_report.json` — independent verification

All confined to `trainer_outputs/b3b_probe/`. **No existing artifact is
modified.** **[C]**

## 19. Scope

> B-3B is a **compact-representation experiment**. It measures whether a
> task-trained probe over frozen MentalBERT CLS embeddings retains DAIC ranking
> signal through the frozen DP path, under the frozen A/B/C criteria.
>
> It is a **second attempt** at B-3 and is **not expected to succeed**. H₀ is a
> legitimate result. It makes no claim about clinical utility, and a successful
> result would **not** unblock Row 10 on its own (§15).
