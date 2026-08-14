# Phase 9.5 — Complete-Dataset Rerun of the Original Implementation
## DAIC-WOZ 113 → 188 Participants (Official Google Colab Baseline)

**Project:** Federated, Privacy-Preserving Mental-Health Detection Pipeline (DAIC-WOZ)
**Phase:** 9.5 — Rerun the ORIGINAL implementation on the COMPLETE DAIC-WOZ dataset
**Execution dates:** 11–13 July 2026
**Hosts:** Windows 11 Pro (preprocessing, local) + **Google Colab GPU (official training)**
**Status:** ✅ **COMPLETE**

> **Scope & integrity note.** Ritik's implementation was preserved exactly. **No source file was modified**
> (the trainer uploaded to Colab is SHA-256 identical to the repo original). No architecture change, no
> optimization, no Version-2 feature, no explainability addition, no BioClinicalBERT, no hyperparameter
> change, no DP/aggregation change. The ONLY change is the **dataset size: 113 → 188 participants**.
> Every number below is copied verbatim from a real artifact.

---

## 1. Objective

Re-execute the original pipeline unchanged, replacing the partial 113-participant subset with the
complete DAIC-WOZ corpus, and quantify what the larger dataset does to the baseline.

---

## 2. Dataset Inspection & Integrity

### 2.1 Complete corpus (`D:\datasets\DAIC-WOZ`)

| Property | Value |
|---|---|
| Files in folder | 201 zips + 4 split CSVs + documentation PDF |
| **Unique participant archives** (`*_P.zip`) | **189** |
| ID range | 300 – 492 |
| IDs never shipped by DAIC-WOZ | 342, 394, 398, 460 (expected gaps) |
| `_2` duplicate archives | 10 (`471–480_P_2.zip`) — byte-identical re-downloads; **auto-excluded** by the extractor's `*_P.zip` glob |
| Modalities inside each archive | transcript, audio (`.wav`, COVAREP, FORMANT), video (CLNF AUs/features/features3D/gaze/hog/pose) |
| Transcript availability | **189 / 189** |

### 2.2 Integrity findings (2 anomalies, both resolved)

| # | Participant | Finding | Resolution |
|---|---|---|---|
| 1 | **440** | `440_P.zip` is **truncated** — "End-of-central-directory signature not found"; `zipfile.is_zipfile` → `False`. Not openable. | **PERMANENTLY EXCLUDED** (user decision). Dataset-integrity exclusion only — not an architecture/preprocessing/model/hyperparameter change. 440's label is `PHQ8_Score = 19.0` (a **positive** case), so its exclusion removes one positive. |
| 2 | **487** | Extractor staged a **4 KB macOS AppleDouble sidecar** (`._487_TRANSCRIPT.csv`, `com.apple.quarantine` metadata) instead of the real transcript. Cause: `extract_daic_woz.py`'s `next(... endswith("_transcript.csv"))` first-match heuristic matches the `._`-prefixed file, which is stored first in the archive. The **real** `487_TRANSCRIPT.csv` (20,406 B, valid UTF-8, 299 rows) was intact in the zip. | **RECOVERED** (user decision). The real transcript was staged over the sidecar — a pure **data-staging correction, no code change**. Audit of all 189 archives confirmed **487 is the only affected participant**. |

> **Latent tooling defect (documented, NOT fixed):** `extract_daic_woz.py` selects the first namelist entry
> ending in `_transcript.csv`, so a macOS `._` sidecar wins over the real file. Left unmodified per the
> preserve-exactly rule. It surfaced only because 487's sidecar bytes broke UTF-8 decoding; had the sidecar
> decoded cleanly it would have silently produced a garbage record.

### 2.3 Effective dataset

**189 archives − 1 corrupt (440) = 188 participants.**

| | Phase 9 baseline | **Phase 9.5** |
|---|---|---|
| Participants | 113 (IDs 300–415) | **188** (IDs 300–492) |
| Additional participants | — | **+75 net** (76 new IDs 416–492, minus corrupt 440) |
| Class balance (PHQ8 > 10) | pos 37 / neg 76 (32.7 % pos) | **pos 45 / neg 143 (23.9 % pos)** |

---

## 3. Label Staging — Source-Data Drift (important)

`stage_labels.py` collects only CSVs containing the exact column `PHQ8_Score`. **The source CSVs on `D:` have
drifted since Phase 8B:**

- `full_test_split.csv` now carries **`PHQ_Score`**, not `PHQ8_Score` → it is **no longer collected**.
- `train_split_Depression_AVEC2017 (1).csv` was renamed (the `(1)` suffix is gone).

Re-running `stage_labels.py` today would therefore harvest only **train + dev = 141 labels**, silently dropping
the 47 test-split participants — **19 of them in the new 416–492 range** (421, 424, 431, 432, 435, 438, 442,
450, 452, 453, 461, 462, 465, 466, 467, 469, 470, 480, 481). That would corrupt the complete-dataset build.

**Resolution (approved standing decision):** the existing `labels/Detailed_PHQ8_Labels.csv` — the original
implementation's own output, **189 entries, IDs 300–492, verified full coverage of all 189 participants** —
was **reused as-is**. `stage_labels.py` was deliberately **not re-run**, because doing so would overwrite a
correct artifact with a broken one. This preserves the original Phase 8B *output* exactly.

---

## 4. Preprocessing (rerun, unchanged scripts)

Exact Phase 8B §4.1 sequence, same flags, no code touched:

| Stage | Command | Result |
|---|---|---|
| Extract | `extract_daic_woz.py` | `processed=189 ok=188 missing_transcript=0 errors=1` (the 1 error = 440, accepted) · 188 transcripts staged |
| Normalize | `normalize_transcripts.py` | `normalized=188 skipped=0 errors=0` |
| Labels | *(reused — see §3)* | 189 entries, full coverage |
| Build records | `build_daic_records.py --data-dir ./data_norm --out ./daic_records.parquet` | `built=188  skipped: no_phq=0 no_file=0 no_textcol=0 empty=0` |
| **Verify (gate)** | `verify_daic_records.py ./daic_records.parquet` | **[PASS]** |

**Verification gate output:**
```
records            : 188
non-empty text     : 188/188
parseable phq      : 188/188
label balance      : pos(>10)=45  neg=143
inferred audio_dim : None
inferred vision_dim: None
[PASS] file satisfies trainer_mentalbert_daic.py (Track C) schema.
```

---

## 5. LDA Processing (unchanged, 20-participant cap preserved)

`format_daic_to_lda.py` has `MAX_PARTICIPANTS = 20` **hardcoded (Ritik's design — preserved)**. It reads the
first 20 sorted folders (300–319), which are identical between the 113 and 188 datasets.

| Artifact | Phase 8B | Phase 9.5 |
|---|---|---|
| Participants embedded | 20 | **20** (unchanged by design) |
| Dataframe | (20, 4) | **(20, 4)** |
| `text_embeddings.parquet.enc` | 91,927 B | **91,927 B** (byte-identical) |
| `lda_manifest.jsonl` | 159 B | **159 B** |

**The 20-cap means LDA embeddings still cover < 11 % of the 188 records.** This is a carried-forward baseline
limitation, not something the dataset upgrade addresses.

---

## 6. MentalBERT Training — OFFICIAL (Google Colab, GPU)

**Environment:** Google Colab GPU. Trainer uploaded byte-for-byte identical to the repo original
(SHA-256 `65b1902e2ba8dd996c6960db1a6595d84fd48500f4d8707cb79688093d7a230b`).
Colab origin confirmed in the artifact: `model_path: /content/trainer_outputs/mentalbert_privacy_subset.pt`.

**Hyperparameters — every original value preserved:**
`--mode supervised --epochs 3 --batch-size 8 --lr 2e-5 --val-split 0.1 --binarize --binarize-threshold 10.0`,
model `mental/mental-bert-base-uncased`, `max_param_change 1e-3`, `max_global_norm 1.0`.
`--device` omitted so the trainer's **own default** (`cuda if available else cpu`) selected the GPU — an
execution-environment difference only, not a hyperparameter.

**Split:** `np.random.seed(42)`, val_split 0.1 → **train = 170, val = 18**.

### 6.1 Final metrics (`trainer_outputs/training_report.json`, verbatim)

| Metric | Value |
|---|---|
| Regression **MAE** | **6.654737075169881** |
| Classification **Accuracy** | **0.7222222222222222** |
| **Precision** | **0.0** |
| **Recall** | **0.0** |
| **F1** | **0.0** |

### 6.2 Modality contribution (`modality_ablation.json`, verbatim)

| Modality | Score | Raw pos-δ | Raw PHQ-δ |
|---|---|---|---|
| **Text** | **2.3365455344319344** | 0.2256280928850174 | 4.447462975978851 |
| Audio | 0.0 | 0.0 | 0.0 |
| Video | 0.0 | 0.0 | 0.0 |

**Still strictly unimodal (text-only).** Audio and vision encoders were never instantiated
(`audio_dim = vision_dim = None`), exactly as at baseline.

---

## 7. Baseline Comparison: 113 → 188

| Metric | Phase 9 baseline (113) | **Phase 9.5 official (188)** | Δ |
|---|---|---|---|
| Records | 113 | **188** | +75 |
| Train / Val | 102 / 11 | **170 / 18** | — |
| Class balance | pos 37 / neg 76 | **pos 45 / neg 143** | pos rate 32.7 % → 23.9 % |
| **MAE** | 8.7652 | **6.6547** | **−2.1104 (−24.1 %)** |
| Accuracy | 0.5455 | 0.7222 | +0.1768 |
| Precision | 0.0 | **0.0** | 0 |
| Recall | 0.0 | **0.0** | 0 |
| F1 | 0.0 | **0.0** | 0 |
| Text / Audio / Video | 1.9956 / 0 / 0 | 2.3365 / 0 / 0 | still unimodal |

### 7.1 Interpretation — read this carefully

1. **The regression head genuinely improved.** MAE fell 8.7652 → 6.6547, a **24 % error reduction**, directly
   attributable to the 66 % larger training set. This is the single real gain of Phase 9.5.

2. **The classifier did NOT improve — it is still degenerate.** `eval_preds.csv` shows **all 18 validation
   samples predicted class 0**, with P(positive) ≈ 0.199–0.214 throughout and `pred_phq` ≈ 5.03–5.09. Precision,
   recall and F1 remain **exactly 0.0** — no positive was ever predicted, so there are no true positives.

3. **The accuracy rise from 0.5455 to 0.7222 is an artifact, not a result.** With an all-negative predictor,
   accuracy is *by definition* the negative rate of the validation split: **13 negatives / 18 = 0.7222**.
   The 188-participant corpus has a lower positive rate (23.9 % vs 32.7 %), so the majority-class baseline is
   simply higher. **Reporting this as improved classification would be wrong.**

4. **Conclusion:** more data improved PHQ *regression* but did not cure the all-negative *classification*
   collapse. The collapse is a property of the training recipe (3 CPU/GPU epochs, unweighted CE on an
   imbalanced set, no class weighting), not of dataset size.

---

## 8. Federated Learning / DP / Aggregation — NOT re-run (by decision)

**Decision (user-approved): the federated round was deliberately not re-executed.** Rationale, established
from the original implementation itself:

- Per Phase 8B §5.2/§5.3, the ~438 MB MentalBERT delta **cannot pass the DP path** (whole-model flatten +
  clip to norm 1.0 with unit noise is degenerate; the upload driver hardcodes its source; 438 MB × 3 exceeds
  the 15 s SubmitReceipt timeout). The federation's aggregation unit is therefore the **Track-A linear probe**
  `local_probe_base.pt` (1,185,335 B, 4 tensors), built from the demo corpus `sample_texts/sample1.txt`.
- **The probe contains no DAIC data whatsoever.** The federated path is consequently **dataset-independent**:
  re-running it on the 188-participant corpus would use the identical probe, identical clip (1.0), identical
  noise multiplier (1.0), identical ε, and identical trimmed-mean rule — yielding a structurally identical
  result that carries **zero information about the 113 → 188 upgrade** (only the random DP noise, and hence
  the model hash, would differ).
- A re-run would additionally have **required destroying** the live Phase 8B/9 federation record: a fresh
  orchestrator seeds Round 1, whose aggregation inserts `global_models[round_id=2]`, colliding with the
  existing `round_id=2` under the unique index (Phase 8B §5.4). The documented remedy is to wipe
  `model_updates`, `receipts`, `global_models`, `fs.files`, `fs.chunks`.

**Therefore the existing federation evidence is PRESERVED INTACT and cited as the standing result.**

**Live MongoDB state at Phase 9.5 close (read-only, unchanged):**
```
model_updates = 6    receipts = 6    global_models = 1 (round_id=2)
fs.files      = 8    fs.chunks = 52  devices = 1
device 9665f030…  (enrolled 2026-06-24)
```

| Federation property | Value (unchanged from Phase 8B/9) |
|---|---|
| Aggregation unit | Track-A 4-tensor probe, 1,185,335 B (**not** MentalBERT, **not** DAIC) |
| DP mechanism / clip / noise / δ | Gaussian / 1.0 / 1.0 / 1e-5 |
| **ε per update** | **5.302585092994046** (RDP single composition) |
| **Cumulative ε per 3-update round** | **15.9078** (< eps_max 100) |
| L2 before clip | 11.3448 |
| Aggregation rule | Trimmed-mean, fires at ≥ 3 updates |
| Published global model | round_id 2, hash `626dfc2af1ea502965c2cc75ab195cc3d73e203fab87753472c5fb51c189d1b6`, 1,184,989 B |
| Topology | Single enrolled device, 3 sequential submissions (simulated federation) |

---

## 9. Limitations (carried forward + new)

**Unchanged from the Phase 9 baseline (dataset size does not address these):**
1. **Degenerate classifier** — precision = recall = F1 = 0; predicts negative for every sample.
2. **Accuracy is majority-class only** — 0.7222 = 13/18 negatives, not discrimination.
3. **Tiny validation split** — 18 samples; classification metrics remain statistically fragile.
4. **Unimodal in practice** — audio & vision contribute exactly 0; the multimodal architecture is unexercised.
5. **LDA capped at 20 participants** (`MAX_PARTICIPANTS = 20`) — now covers < 11 % of records (was < 18 %).
6. **The trained model is not the federated model** — MentalBERT delta cannot pass the DP path.
7. **Federated model has no DAIC signal** — the global model derives from the demo `sample_texts` probe.
8. **Weak privacy budget** — ε = 5.30/update (cumulative 15.91/round); noise dominates signal.
9. **Single-node "federation"** — one device submitting 3× sequentially; no client heterogeneity.
10. **No training wall-clock instrumentation** — the trainer emits no timer.
11. **Unseeded training** — only the split is seeded (`np.random.seed(42)`); torch is not, so metrics vary
    run-to-run (the unofficial local 188 run gave MAE 6.7058 vs Colab's 6.6547).

**New in Phase 9.5:**
12. **Participant 440 permanently excluded** — corrupt archive; removes one positive case (PHQ8 = 19.0).
13. **Extractor AppleDouble defect** — `._`-sidecar first-match bug (affected 487 only); documented, not fixed.
14. **Label-source drift** — `full_test_split.csv` column renamed `PHQ8_Score` → `PHQ_Score`; `stage_labels.py`
    can no longer reproduce the 189-label sheet from the current CSVs.

---

## 10. Files Generated / Modified

| Path | Content |
|---|---|
| `data/<pid>_P/<pid>_TRANSCRIPT.csv` | 188 staged transcripts (75 new; 487 corrected) |
| `data_norm/<pid>_P/<pid>_TRANSCRIPT.csv` | 188 normalized transcripts |
| `daic_records.parquet` | **188 records** (was 113) |
| `secure_store/sess-DAICWOZ/encrypted/text_embeddings.parquet.enc` | LDA embeddings (20×4), 91,927 B |
| `trainer_outputs/training_report.json` | **OFFICIAL Colab 188 metrics** |
| `trainer_outputs/eval_preds.csv` | 18 validation predictions (all class 0) |
| `trainer_outputs/modality_ablation.json` | text 2.3365 / audio 0 / video 0 |
| `trainer_outputs/mentalbert_delta.receipt.json` | delta receipt |
| `trainer_outputs/phase9_113_baseline/` | **preserved** 113-record baseline artifacts |
| `trainer_outputs/local_188_preliminary/` | unofficial local 188 run (superseded by Colab) |
| `colab_package/` | Colab notebook + package (trainer SHA-verified identical) |
| `labels/Detailed_PHQ8_Labels.csv` | **unchanged** (reused, 189 entries) |
| MongoDB `federated` | **unchanged** (federation preserved) |

**No project source file was modified in Phase 9.5.**

---

## 11. Phase 9.5 PASS/FAIL Audit

| # | Requirement | Status | Evidence |
|---|---|---|---|
| 1 | Original implementation preserved exactly | **PASS** | Trainer SHA-256 identical; no source file edited |
| 2 | No architecture change | **PASS** | `MultiModalModel` untouched; text-only path unchanged |
| 3 | No hyperparameter change | **PASS** | §6 — every arg equals the original default / 8B value |
| 4 | No DP / aggregation / federated change | **PASS** | §8 — federation untouched and not re-run |
| 5 | No Version-2 / explainability / BioClinicalBERT | **PASS** | None introduced |
| 6 | Complete dataset used | **PASS** | 188 of 189 (440 excluded, documented) |
| 7 | Preprocessing rerun on full corpus | **PASS** | §4 — verify gate **[PASS]**, 188 records |
| 8 | Official training on Google Colab | **PASS** | §6 — `model_path: /content/…` confirms Colab |
| 9 | Baseline comparison produced | **PASS** | §7 |
| 10 | Dataset-integrity exclusions recorded | **PASS** | §2.2 (440 excluded, 487 recovered) |
| 11 | Every metric artifact-backed | **PASS** | All values verbatim from `training_report.json` / `modality_ablation.json` / `eval_preds.csv` |
| 12 | No hallucinated / estimated values | **PASS** | Unmeasured items (wall-clock) left explicitly unfilled |

---

## 12. Final Conclusion

Phase 9.5 achieved its objective: **the original implementation was re-executed unchanged on the complete
DAIC-WOZ corpus (188 of 189 participants), with official training performed on Google Colab.**

- **Dataset upgraded 113 → 188** (+75 net participants) with two integrity events fully documented:
  440 permanently excluded (corrupt archive) and 487 recovered (AppleDouble sidecar).
- **The one real gain: PHQ regression MAE improved 8.7652 → 6.6547 (−24.1 %).**
- **The classifier remains degenerate** — precision = recall = F1 = 0, all-negative predictions. The apparent
  accuracy gain (0.5455 → 0.7222) is purely the majority-class rate of a val split with fewer positives, and
  **must not be read as improved detection.**
- **The system remains unimodal** (text-only) and the **federated path remains DAIC-free** (demo probe), both
  unchanged by dataset size.

The larger dataset improved regression but did not fix the classification collapse; that failure is a property
of the original training recipe, not of the data volume.

**PHASE 9.5 COMPLETE — ORIGINAL IMPLEMENTATION RERUN ON THE COMPLETE DAIC-WOZ DATASET.**

---

*No code modified. No architecture redesigned. No hyperparameters changed. No Version-2 features introduced.*
