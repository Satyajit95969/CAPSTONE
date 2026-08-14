# PHASE 11.6 — BASELINE-CV EXECUTION PACKAGE AUDIT
## Final Implementation Audit Before Colab GPU Execution

**Type:** Code + methodology audit — no training executed, trainer not modified
**Scope:** `run_baseline_cv.py`, `aggregate_baseline_cv.py`, `PHASE_11_BASELINE_CV_COLAB_PACKAGE.md`
**Method:** Line-by-line cross-check of the driver against the frozen trainer's actual code paths (`MultiModalModel`, `FusionHead`, `collate_batch`, `run_inference`, `fine_tune_supervised`), plus local execution of every non-training path and a synthetic-data test of the aggregator.

Convention: **[PASS]** verified correct · **[FIXED]** defect found and corrected this audit · **[OBS]** deliberate design note, not a defect.

---

## 1. Frozen trainer imported and used without modification — [PASS]

- `import trainer_mentalbert_daic as T`; all model/training/inference via `T.*`. The trainer source is never edited.
- The driver hard-codes the frozen SHA and **verifies it at startup** (`sha256(T.__file__) == 65b1902e…a230b`), aborting on mismatch. [E] Confirmed locally: `sha256 == frozen`.
- The trainer `.py` on disk is byte-identical to the Phase 11.0-verified file (unchanged this phase).

## 2. No architecture / optimizer / scheduler / loss / tokenizer / preprocessing / hyperparameter change — [PASS]

| Element | Source | Verified |
|---|---|---|
| Architecture | `T.MultiModalModel(bert, audio_dim, vision_dim, device)`; `infer_dims` replicates the trainer's own dim logic | [E] returns `(None, None)` → text-only, identical to frozen |
| Optimizer | `AdamW` **inside** `T.fine_tune_supervised` (unchanged) | [PASS] |
| Scheduler | None exists in the trainer; **none introduced** | [PASS] |
| Loss | `CE + 0.5·MSE` + grad-clip 1.0, **inside** `T.fine_tune_supervised` | [PASS] |
| Tokenizer | `AutoTokenizer.from_pretrained("mental/mental-bert-base-uncased")`, `max_len=128` (matches `MultiModalDataset` default) | [PASS] |
| Preprocessing | `T.MultiModalDataset`, `T.collate_batch` (unchanged) | [PASS] |
| Hyperparameters | epochs 3, batch 8, lr 2e-5, threshold 10.0 — driver defaults = frozen values | [PASS] |
| **Single change** | data partition from the manifest + `val_dataset=None` (train on full 4/5 partition) | [PASS] — the one intended factor |

**[OBS]** The driver sets a per-repeat torch seed (`1000+r`) for regenerability; this is RNG control, **not** a hyperparameter/recipe change. It is a deliberate, documented deviation from the frozen *unseeded* run: distinct seeds across repeats sample the training-stochasticity noise reproducibly (protocol P-3 as clarified in 11.5). Does not affect architecture, optimizer, or loss.

## 3. Participant-level fold manifest applied correctly — [PASS]

- Loads `fold_manifest.json`; asserts `n_participants == 188`; builds `by_pid` index and subsets records by `train_ids`/`test_ids`. Fold identity comes **only** from the manifest, not the CLI.
- [E] Verified locally: folds train 150/150/150/151/151, test 38/38/38/37/37, 9 positives per test fold.

## 4. Every participant appears in exactly one test fold — [PASS]

[E] Verified locally: the union of the five `test_ids` sets = **188 participants, tested exactly once** (`StratifiedKFold` guarantees this; confirmed empirically that Σ test sizes = 188 with no repeats).

## 5. No train/test leakage possible — [PASS]

- One row per participant ⇒ participant-level = record-level; folds are participant-disjoint. [E] `train_ids ∩ test_ids = ∅` verified for all 5 folds.
- Test predictions come from a `shuffle=False` loader over the held-out records only; no test row enters `fine_tune_supervised`. **Structural — leakage is impossible by construction.**
- **[PASS] Alignment traced:** `te_recs = [by_pid[i] for i in test_ids]` (built in `test_ids` order) → `MultiModalDataset` preserves order → `run_inference(shuffle=False)` preserves order → `pred_phq[i] ↔ te_recs[i] ↔ test_ids[i]`. The saved `participant_id` column, the true labels, and the predictions are mutually aligned. No off-by-one/reorder bug.

## 6. Aggregation — mean, SD, CI, MDE — [FIXED]

**Defect found (statistical, medium severity):** the original aggregator computed 95% CIs and the MDE with a **normal 1.96 multiplier over the pooled R×k rows**. Two problems: (a) the R×k fold-runs are **not iid** (repeats share data; folds are the independent unit), so pooling inflates the effective n and narrows the CI; (b) with k = 5, the normal quantile understates the interval versus Student-t (t₀.₉₇₅,₄ = 2.776 vs 1.96, ~42% wider). Net effect: **the noise band and MDE were too tight**, risking false "detectable effect" claims across all of Phase 12.

**[FIXED]** The aggregator now:
- averages repeats within a fold first, then computes **fold-level Student-t** CIs (df = k−1) — the fold is the unit of independence;
- computes the **MDE with Student-t** and labels it a *planning* estimate, with the operative accept/reject test defined as the SD of the **paired per-fold differences** (V2_f − BaselineCV_f);
- reports mean, fold SD, partition SD, and training-stochasticity SD (the last decomposition labeled approximate).

[E] Verified on synthetic data: R=5 → `t_crit = 2.776`, training-SD measured; both paths exit 0.

**[FIXED] Edge case (R = 1):** within-fold SD across a single repeat is undefined (`ddof=1` → NaN). Now guarded — training-stochasticity SD is reported as `n/a (R<2)` with an explicit warning that more repeats are required before the numbers gate Phase 12. [E] Verified: R=1 path exits 0, reports `None`.

## 7. Artifacts sufficient to reproduce Baseline-CV — [PASS]

- Per fold/repeat: `rep{r}_fold{f}_preds.csv` = `participant_id, phq, phq_bin, pred_phq, p_pos, pred_class` → **every metric is recomputable/auditable** from raw predictions.
- Per repeat: `rep{r}_metrics.csv`. Aggregate: `baseline_cv_summary.{json,md}`.
- Provenance recorded: trainer SHA, per-repeat seeds (`1000+r`), manifest generator (`StratifiedKFold(5, shuffle=True, random_state=42)`), CLI, R, GPU model, `transformers==4.44.0`.
- **Reproducibility contract:** manifest + environment + seeds + aggregate statistics are regenerable (single-run GPU nondeterminism is part of the measured noise). Sufficient.

## 8. Bugs, edge cases, and residual risks

| ID | Finding | Class | Status |
|---|---|---|---|
| A | CI/MDE used normal-approx over correlated pooled rows → noise band too tight | Statistical, medium | **[FIXED]** fold-level Student-t |
| B | R=1 → training-stochasticity SD = NaN | Edge case | **[FIXED]** guarded + warning |
| C | Prediction↔label↔participant ordering | Correctness | **[PASS]** traced, aligned |
| D | `true_phq` `or`-chain treats genuine PHQ 0.0 as 0.0 | Consistency | **[PASS]** matches the trainer's own `y_true` exactly; no distortion |
| E | Per-repeat torch seeding vs frozen unseeded run | Methodology | **[OBS]** deliberate, documented; samples noise reproducibly |
| F | CV trains on 150–151 vs frozen 170 (fewer steps/epoch) | Methodology | **[OBS]** inherent to CV, not a confound (it *is* the split change) |
| G | Gated MentalBERT auth; `transformers==4.44.0` pin; GPU required | Operational | **[OBS]** documented; driver **aborts on CPU** |
| H | Noise decomposition is approximate variance-components, not formal ANOVA | Reporting | **[OBS]** labeled approximate |

No unresolved correctness bug remains. The two defects (A, B) were in the **aggregator**, which runs *after* the GPU run on saved artifacts — so they never affected the run itself and are now corrected. The **driver** (what executes on GPU) was clean as written.

---

## Certification

The driver is confound-free (frozen trainer, SHA-checked, GPU-guarded, one changed factor), the manifest is applied correctly with zero possible leakage and full-coverage single-test-fold assignment, the artifacts are reproducibility-complete, and the aggregator's statistics are now correct (fold-level Student-t CIs and MDE, R<2 guarded). All non-training paths and the aggregator were executed and verified locally; no training was run and the trainer was not modified.

> ### ✅ READY FOR COLAB EXECUTION

**Pre-flight reminders (from the package §7 checklist):** GPU runtime, `transformers==4.44.0`, HF auth for the gated model, run `--repeat 1..R` (R ≥ 5 recommended), then aggregate. On completion, return `trainer_outputs/baseline_cv/` to populate the `[PENDING-GPU]` rows of `PHASE_11_BASELINE_CV.md`.

---

*Code and methodology audit only. No training executed. Trainer not modified. Fixes applied this audit: `aggregate_baseline_cv.py` (fold-level Student-t CIs/MDE + R<2 guard) and one reconciling line in the package doc. Driver and manifest unchanged (verified correct).*
