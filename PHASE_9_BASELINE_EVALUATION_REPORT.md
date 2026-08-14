# Phase 9 — Baseline Evaluation of the Original Senior Project
## Original-Implementation Baseline Reference Report

**Project:** Federated, Privacy-Preserving Mental-Health Detection Pipeline (DAIC-WOZ)
**Phase:** 9 — Baseline Evaluation (read-only)
**Evaluation date:** 9 July 2026
**Host:** Windows 11 Pro (10.0.26200), CPU-only Torch build
**Source execution evaluated:** Phase 8B end-to-end run (26 Jun 2026) + a subsequent identical-configuration re-run (28 Jun 2026)
**Status:** Baseline established from existing artifacts. No training, federation, or aggregation was re-run.

> **Scope & integrity note.** This report reads *only* artifacts already present on disk and in MongoDB from the original implementation's executed runs. Every number below is copied verbatim from a real artifact (`training_report.json`, `modality_ablation.json`, `eval_preds.csv`, `metrics_*.json`, receipt JSONs, DP-comparison CSVs, live MongoDB collections, `PHASE_8B_EXECUTION_REPORT.md`). No code was modified, nothing was retrained, no architecture was redesigned, and no Version-2 improvement is proposed. Where a metric was never emitted by the original tooling, that is stated explicitly rather than invented.

---

## 1. Experimental Setup

### 1.1 Two disjoint execution tracks

The original project contains **two separate, non-connected model tracks**. Understanding the baseline requires keeping them distinct, because they are trained on different data and never share weights:

| | **Track C — MentalBERT DAIC model** | **Track A — Linear probe (federated unit)** |
|---|---|---|
| Script | `trainer_mentalbert_daic.py` | `create_dp_comparison.py` / `LDA` pipeline |
| Trained on | 113 DAIC-WOZ transcripts | demo `sample_texts/sample1.txt` |
| Model | `MultiModalModel` (MentalBERT + fusion head), ~109M params, 207-key state dict | 4-tensor MLP probe (`fc1`, `fc2`), ~1.18 MB |
| Artifact | `mentalbert_privacy_subset.pt` (438.8 MB) | `local_probe_base.pt` (1,185,335 B) |
| Role | The "mental-health model" | The object actually pushed through federation/DP/aggregation |

**Critical baseline fact:** the federated round did **not** federate the trained MentalBERT model. The MentalBERT delta (~438 MB) is not a usable aggregation unit under the original DP path, so the federation was driven by the small Track-A probe instead (documented in Phase 8B §5.2/§5.3). The two tracks are therefore evaluated separately below.

### 1.2 System configuration

| Component | Detail |
|---|---|
| Compute | **CPU only** (torch 2.8.0+cpu); all training/embedding/inference on CPU |
| Python | CPython 3.11 (`.venv`) |
| Key libs | transformers 4.44.0 (pinned for `AdamW` import), scikit-learn 1.7.1, pandas 2.3.2, numpy 1.26.4, cryptography 49.0.0, pymongo 4.16.0 |
| Orchestrator | Rust `orchestrator.exe` (24,990,720 B, debug), `0.0.0.0:50051`, `enable_tls=true` |
| Datastore | MongoDB v8.0, port 27017, DB `federated`, GridFS `fs` bucket |
| TPM | Windows Platform Crypto Provider, key `FederatedDeviceKey` (ECDSA P-256) |

---

## 2. Dataset

| Property | Value | Source |
|---|---|---|
| Corpus | DAIC-WOZ (transcript-only path) | `extract_daic_woz.py` |
| Raw archives | 113 `*_P.zip` | `[SUMMARY] processed=113 ok=113` |
| Labels staged | 189 PHQ-8 entries (from 236 rows, 3 AVEC split CSVs) | `stage_labels.py` |
| PHQ8_Score range | 0 … 23 | label staging |
| Trainer records built | **113** (skipped: 0) | `daic_records.parquet` |
| Text coverage | 113/113 non-empty | `verify_daic_records.py` |
| PHQ coverage | 113/113 parseable | `verify_daic_records.py` |
| Class balance (>10) | pos = 37, neg = 76 (≈32.7% positive) | verification gate |
| Modality present | **text only** (audio_dim=None, vision_dim=None) | inferred dims |
| Binarize threshold | 10.0 (PHQ8 > 10 ⇒ depressed) | trainer default |
| LDA embedding subset | **20** participants (hardcoded `MAX_PARTICIPANTS=20`) | `format_daic_to_lda.py` |

The label-staging log reported `positives(>10)=46` over 189 label rows, while the 113 records actually built carry `pos=37 / neg=76` — the difference is the subset of labels that matched an extracted transcript.

---

## 3. Model

### 3.1 Track-C — MentalBERT `MultiModalModel`

- **Backbone:** `mental/mental-bert-base-uncased` (canonical, vocab 30522; gated HF model, access resolved before run).
- **Heads (fusion):** shared `fc1` → hidden; a classification head (`classifier`, 2-class), and a Gaussian PHQ regression head (`phq_mu`, `phq_logsigma`).
- **Optional encoders:** `audio_encoder` / `vision_encoder` (SmallMLP → 128-dim) — **instantiated only if** audio_dim/vision_dim > 0. For DAIC text-only they are `None`.
- **Parameters / state:** 207-key `OrderedDict` (`bert.*` + `fusion.*`), checkpoint 438,805,659 B.
- **Loss:** `CrossEntropyLoss(binarized label)` + `0.5 · MSELoss(raw PHQ)`, gradient clipped to global-norm 1.0.

### 3.2 Track-A — Linear probe

- 4 tensors: `fc1.weight/bias`, `fc2.weight/bias` — the object serialized as `local_probe_base.pt`, and the schema of the published global model (`keys=['fc1.bias','fc1.weight','fc2.bias','fc2.weight']`).

---

## 4. Training Results

### 4.1 Track-C MentalBERT — training configuration

`--mode supervised --input daic_records.parquet --device cpu --epochs 3 --batch-size 8 --binarize` (lr 2e-5, val split 0.1, threshold 10.0). **Split: train = 102, val = 11.**

### 4.2 Loss and validation-MAE progression (as-run, Phase 8B live log)

| Epoch | avg training loss | val MAE |
|---|---|---|
| 1 / 3 | 37.3740 | 9.9550 |
| 2 / 3 | 27.5487 | 9.1430 |
| 3 / 3 | 23.6329 | 8.7315 |

Training loss decreased monotonically (37.37 → 23.63); validation MAE improved 9.96 → 8.73. Loss had **not** converged at epoch 3.

### 4.3 Final evaluation metrics (from `training_report.json`, latest artifact, 28 Jun re-run)

| Metric | Value |
|---|---|
| Regression **MAE** | **8.7652** |
| Classification **Accuracy** | **0.5455** (6/11) |
| **Precision** | **0.0** |
| **Recall** | **0.0** |
| **F1-score** | **0.0** |

`eval_preds.csv` (11 rows) shows the model predicted **class 0 for every validation sample**, with `pred_phq ≈ 4.27–4.34` for all rows and softmax P(positive) ≈ 0.365–0.376 throughout. This is a **degenerate all-negative predictor**: accuracy 0.5455 simply reflects the negative-majority of the 11-sample split; precision/recall/F1 = 0 because no positive was ever predicted (no true positives).

> **Artifact-vs-log note.** The live Phase 8B log recorded epoch-3 MAE = 8.7315 and final MAE = 8.7315; the persisted `training_report.json` (a later, identically-configured 28 Jun re-run — file mtime 28 Jun 13:40) records MAE = 8.7652. The two runs agree to within ~0.03 MAE; acc/precision/recall/F1 are identical (0.5455 / 0 / 0 / 0). Both are the same undertrained, all-negative baseline.

### 4.4 Track-A probe — evaluation metrics

The federated aggregation unit was trained/evaluated on the demo `sample_texts` corpus, not DAIC. Its metrics are therefore **trivial** and must not be read as DAIC performance:

| Variant | Accuracy | Precision | Recall | F1 | MAE |
|---|---|---|---|---|---|
| `metrics_base.json` | 1.0 | 0.0 | 0.0 | 0.0 | 1.3204 |
| `metrics_rag.json` | 1.0 | 0.0 | 0.0 | 0.0 | 1.6080 |
| `metrics_vector_rag.json` | 1.0 | 0.0 | 0.0 | 0.0 | 2.1529 |

Accuracy 1.0 with precision/recall/F1 = 0 indicates a single-class demo set (all-negative, trivially "100% accurate"). These numbers characterize the plumbing, not depression detection.

---

## 5. Modality Contribution (Text / Audio / Video)

The DAIC path executed is **text-only**; audio and vision encoders were never instantiated (audio_dim = vision_dim = None). The ablation (`modality_ablation.json`, zero-out / empty-string perturbation on the trained MentalBERT) quantifies this directly:

| Modality | Ablation score | Raw pos-delta | Raw PHQ-delta | Interpretation |
|---|---|---|---|---|
| **Text** | **1.9956** | 0.15497 | 3.83631 | **Sole contributor** — zeroing text moves PHQ output by ~3.84 |
| **Audio** | 0.0 | 0.0 | 0.0 | Absent from pipeline — zero contribution |
| **Video** | 0.0 | 0.0 | 0.0 | Absent from pipeline — zero contribution |

**Baseline conclusion on modality:** the original executed system is a **unimodal (text-only) model**. Although `MultiModalModel` and the DP-comparison harness are *architecturally* multimodal, no audio or video features entered the executed DAIC run, so 100% of predictive signal is textual. The "multimodal" capability is present in code but **unexercised** at baseline.

---

## 6. Federated Learning Results

### 6.1 What was federated

The Track-A 4-tensor probe (`local_probe_base.pt`, 1,185,335 B) — **not** the MentalBERT model. Two full rounds were executed (26 Jun and 28 Jun), each: GetRound → DP → AES-GCM encryption → chunked GridFS upload → TPM ECDSA receipt → aggregation at the `≥3` update threshold.

### 6.2 Differential Privacy accounting (identical across all uploads)

| Parameter | Value | Source |
|---|---|---|
| Mechanism | Gaussian | receipt JSONs |
| Clip norm | 1.0 (applied) | `clip_applied: true` |
| Noise multiplier | 1.0 | receipts |
| δ | 1e-05 | receipts |
| **ε per update** | **5.302585092994046** | RDP single-composition |
| L2 norm before clip | 11.3448 | every upload |
| L2 norm after noise | ~543–544 (randomized: 543.02, 544.04, 543.75 on 28 Jun; 543.73, 542.61, 544.67 on 26 Jun) | receipts |
| **Cumulative ε (per 3-update round)** | **15.9078** (< eps_max 100) | 3 × 5.302585 |

### 6.3 Aggregation & publication (26 Jun round, verified in log + Mongo)

| Item | Value |
|---|---|
| Aggregation rule | Trimmed-mean of 3 independent updates, fired on 3rd `SubmitReceipt` |
| Round transition | Round 1 → **Round 2 (Collecting, global_model_available=True)** |
| Global model hash | `626dfc2af1ea502965c2cc75ab195cc3d73e203fab87753472c5fb51c189d1b6` |
| Global model size | 1,184,989 B, 2 chunks, keys `['fc1.bias','fc1.weight','fc2.bias','fc2.weight']` |
| Download validation | Full-hash match, `load_state_dict(strict=True) OK`, per-tensor equality verified |

### 6.4 Live MongoDB state at evaluation time (Phase 9, read-only)

```
model_updates = 6      receipts = 6       global_models = 1 (round_id=2)
fs.files      = 8      fs.chunks = 52     devices = 1
global_model_hash = 626dfc2af1ea502965c2cc75ab195cc3d73e203fab87753472c5fb51c189d1b6
device 9665f030…  enrolled 2026-06-24 14:36:35  last_seen 2026-06-28 08:14:36
receipt scheme = AES-GCM-DP-ECDSA, verified = True, hmac_chain present
```

The 6 updates / 6 receipts / 8 fs.files reflect the **two** executed rounds (3 uploads each) accumulated in the store; a single published global model (round_id=2) is retained. On-disk `receipts/` holds 492 receipt JSONs (full historical agent-receipt trail across all sessions).

### 6.5 Federation topology

Single enrolled device (`9665f030…`), single physical host, loopback orchestrator. The "3 clients" are 3 sequential submissions from the **same** device, not 3 distinct participants — federation is **simulated on one node**.

---

## 7. Runtime Measurements

| Stage | Measurement | Source |
|---|---|---|
| MentalBERT download | 438 MB in 2:17 (≈3.19 MB/s) | LDA log |
| LDA embedding (20 sessions, 768-dim CLS) | ≈6 s @ 2.94 it/s | LDA log |
| MentalBERT training (3 epochs, CPU) | **no wall-clock emitted**; output timestamps place completion ≈14:49 local | trainer (no timer) |
| Federated round upload span (26 Jun) | 10:00:18 → 10:01:03 ≈ **45 s** for 3 uploads + aggregation | orchestrator log (UTC) |
| Aggregation latency | inline within 3rd receipt, < 15 s timeout window | orchestrator log |
| Per-upload payload | 1,579,963 B streamed in 2 chunks (1 MB each) | upload log |

The original trainer emits **no training wall-clock timer**; total training time is not directly recoverable and is left as *not measured* rather than estimated.

---

## 8. DP Noise-Mechanism Sweep (Track-A supplementary baseline)

`dp_noise_mechanism_comparison_base.csv` (30 rows) sweeps 6 mechanisms × 5 noise multipliers on the probe. At noise_multiplier = 1.0:

| Mechanism | Accuracy | MAE | Silhouette | Distortion ratio | L2 after |
|---|---|---|---|---|---|
| gaussian | 1.0 | 1.3204 | 0.6397 | 47.91 | 543.49 |
| laplace | 1.0 | 1.3204 | 0.6488 | 67.59 | 766.74 |
| uniform | 1.0 | 1.3204 | 0.6439 | 27.68 | 314.04 |
| exponential | 1.0 | 1.3204 | 0.6412 | 67.63 | 767.20 |
| student_t | 1.0 | 1.3204 | 0.6453 | 53.61 | 608.16 |
| none | 1.0 | 1.3204 | 0.6404 | 0.088 | 1.00 |

Accuracy/MAE are **flat (1.0 / 1.3204) across every mechanism and noise level** — the demo dataset is too trivial to show any privacy–utility tradeoff. The only thing the sweep varies is the **distortion ratio** (L2 after ÷ before): 27.7×–67.6×, i.e., the added noise is 28–68× larger than the signal it protects.

---

## 9. Strengths (observed, original implementation)

1. **Full pipeline executes end-to-end** on real DAIC transcripts: extract → normalize → build → LDA embed → train → checkpoint → delta → DP → encrypt → upload → aggregate → publish → download → next round.
2. **Cryptographic integrity is real and verified:** per-chunk SHA-256, full-model hash match on download, `load_state_dict(strict=True)` clean, TPM-backed ECDSA P-256 receipts, HMAC receipt chaining, AES-GCM at rest.
3. **Genuine DP accounting:** RDP single-composition ε = 5.3026/update computed by real accountant, not hard-coded; cumulative ε tracked against `eps_max`.
4. **Reproducible & deterministic infrastructure:** two independent runs (26 & 28 Jun) produced consistent DP (ε, L2-before), identical global-model schema, and consistent classification metrics.
5. **Canonical model fidelity:** the exact gated MentalBERT (vocab 30522) was used; a mismatched local substitute was correctly rejected.
6. **Monotone learning signal:** training loss and validation MAE both improved every epoch, confirming the training loop is wired correctly.

---

## 10. Limitations (observed / documented — no fixes proposed)

1. **Degenerate classifier:** precision = recall = F1 = 0 on Track-C; the model predicts the negative class for all 11 validation samples. Accuracy 0.5455 is majority-class only.
2. **Severely undertrained:** 3 CPU epochs, loss still 23.63 (not converged); MAE 8.77 on a 0–24 PHQ scale is a large error (~1.5× the PHQ-8 clinical severity band width).
3. **Tiny evaluation:** validation split = **11 samples**; all classification metrics are statistically fragile.
4. **The trained model is not the federated model:** the ~438 MB MentalBERT delta cannot pass the DP path (whole-model flatten + clip to norm 1.0 with unit noise is degenerate; upload driver hardcodes its source; 438 MB × 3 exceeds the 15 s receipt timeout). Federation ran on a **separate 1.18 MB probe** trained on demo text.
5. **Federated model has no DAIC signal:** the published global model derives from `local_probe_base.pt`, built from `sample_texts/sample1.txt` (metrics acc 1.0 = trivial single-class), never from DAIC.
6. **Unimodal in practice:** audio & vision contribute exactly 0; the multimodal architecture is unexercised.
7. **LDA capped at 20 participants** (`MAX_PARTICIPANTS=20`, hardcoded) — embeddings cover <18% of the 113 records.
8. **Weak privacy budget:** ε = 5.30/update (cumulative 15.91/round) is high (weak) privacy; noise dominates signal (distortion 28–68×), so DP utility on this scale is unmeasurable.
9. **Single-node "federation":** one enrolled device submitting 3× sequentially; no real client heterogeneity, no non-IID split, no multi-party aggregation.
10. **No training wall-clock instrumentation:** total training time is not emitted and cannot be recovered from artifacts.
11. **Volatile round state:** in-memory round state is not rehydrated from Mongo (an orchestrator restart mid-round orphans persisted uploads — encountered in 8B §5.5).
12. **Two-run metric drift:** persisted `training_report.json` (MAE 8.765) differs slightly from the live 8B log (MAE 8.732), reflecting a non-seed-locked re-run rather than a single canonical artifact.

---

## 11. Comparison with the Dissertation / Senior Report

A direct numeric comparison against the senior dissertation **cannot be performed from repository artifacts**: no dissertation file or reported-metric table is present in the repo. `README.md` enumerates *which* metrics the system measures (Accuracy / F1 / MAE / silhouette) but states **no target or claimed values**. The only prior internal record is `PHASE_8B_EXECUTION_REPORT.md`, against which the current artifacts agree:

| Metric | Phase 8B report (live log) | Phase 9 persisted artifact | Agreement |
|---|---|---|---|
| Training loss (ep1→3) | 37.37 → 23.63 | (same run) | ✓ |
| Val MAE (ep1→3) | 9.96 → 8.73 | report MAE 8.765 (28 Jun re-run) | ✓ within 0.03 |
| Accuracy | 0.5455 | 0.5455 | ✓ identical |
| Precision/Recall/F1 | 0 / 0 / 0 | 0 / 0 / 0 | ✓ identical |
| ε per update | 5.302585 | 5.302585 | ✓ identical |
| Cumulative ε | 15.91 | 15.91 (3/round) | ✓ identical |
| Global model hash | 626dfc2a…d1b6 | 626dfc2a…d1b6 (Mongo) | ✓ identical |
| Mongo (single round) | updates 3 / receipts 3 / gm 1 | updates 6 / receipts 6 / gm 1 (2 rounds accumulated) | ✓ consistent |

**Verdict:** the executed metrics are internally consistent and faithfully reproduce the Phase 8B record. No external dissertation figures are available in-repo to benchmark against; if the senior report is later supplied, the baseline numbers in §4–§7 above are the reference set to compare it to.

---

## 12. Baseline Summary Table (reference for future comparison)

| Dimension | Baseline value |
|---|---|
| Dataset | DAIC-WOZ, 113 transcripts, text-only, pos/neg = 37/76 |
| Model (primary) | MentalBERT `MultiModalModel`, ~109M params, text-only |
| Split | train 102 / val 11 |
| Training | 3 epochs, CPU, batch 8, lr 2e-5, CE + 0.5·MSE |
| Training loss (final) | **23.6329** |
| Validation MAE (final) | **8.73** (log) / **8.765** (artifact) |
| Accuracy | **0.5455** |
| Precision / Recall / F1 | **0.0 / 0.0 / 0.0** |
| Text / Audio / Video contribution | **1.9956 / 0 / 0** (unimodal) |
| Federated unit | 4-tensor probe (NOT MentalBERT), 1.18 MB |
| DP: mechanism / clip / noise / δ | Gaussian / 1.0 / 1.0 / 1e-5 |
| ε per update / cumulative | **5.3026 / 15.91** |
| Aggregation | trimmed-mean, ≥3 updates, single node |
| Global model | round_id 2, hash `626dfc2a…d1b6`, 1,184,989 B |
| Federated round runtime | ~45 s (3 uploads + inline aggregation) |
| Rounds executed | 2 (26 Jun, 28 Jun) |

---

---

## 13. Phase 9 Final Validation Audit (verification pass)

**Audit date:** 9 July 2026. **Method:** every number in §1–§12 was re-read directly from its source artifact (JSON, CSV, parquet, source file, live MongoDB) and compared byte-for-byte to the report. No code modified; nothing retrained; no outputs regenerated.

### 13.1 Metric-by-metric verification (all re-confirmed against source)

| Report claim | Source re-read | Match |
|---|---|---|
| MAE 8.7652 / acc 0.5455 / P/R/F1 = 0 | `training_report.json` → mae 8.765163811770352, acc 0.5454545, 0/0/0 | ✓ |
| eval_preds all class 0; phq 4.27–4.34; P(pos) low end | `eval_preds.csv` → classes {0}; phq 4.2742–4.3353; P(pos) 0.3646–0.3758 | ✓ (P(pos) low end corrected 0.366→0.365) |
| Text/Audio/Video = 1.9956 / 0 / 0; posδ 0.15497, phqδ 3.83631 | `modality_ablation.json` verbatim | ✓ |
| metrics base/rag/vrag MAE 1.3204 / 1.6080 / 2.1529, acc 1.0 | `metrics_*.json` verbatim | ✓ |
| Dataset 113 rec, text 113/113, pos 37 / neg 76, PHQ 0–23 | `daic_records.parquet` recomputed | ✓ |
| Loss = CE + 0.5·MSE, grad-clip 1.0 | `trainer_mentalbert_daic.py:322–326` | ✓ |
| LDA cap 20 | `format_daic_to_lda.py:47` `MAX_PARTICIPANTS = 20` | ✓ |
| DP ε 5.302585, clip 1.0, noise 1.0, δ 1e-5; L2-before 11.3448; 28-Jun L2-after 543.02/544.04/543.75 | receipt JSONs verbatim | ✓ |
| DP sweep gaussian nm1.0: silhouette 0.6397, distortion 47.91, L2-after 543.49 | `dp_noise_mechanism_comparison_base.csv` | ✓ |
| Mongo 6/6/1/8/52/1; global model round 2, hash `626dfc2a…d1b6` | live MongoDB `federated` | ✓ |
| Receipts on disk = 492 | `receipts/*.json` count | ✓ |

**No hallucinated, assumed, or estimated values were found.** Every quantitative claim traces to an existing artifact. Items explicitly marked "not measured" (training wall-clock) remain correctly labelled as not emitted by the tooling — no substitute estimate was inserted.

### 13.2 Omitted-metric sweep (Task 4)

Two supplementary Track-A metrics the original implementation *does* emit were not in §1–§12. They are recorded here for completeness; both are trivial and do not alter any baseline conclusion:

- **RAG retrieval latency:** `dp_noise_mechanism_comparison_{rag,vector_rag}.csv` carry `retrieval_avg_latency_ms`, `retrieval_median_latency_ms`, `retrieval_k`. All 30 rows in each file record **avg = median = 0.0 ms, k = 3** — retrieval was effectively instantaneous / unmeasured on the demo corpus. (The `base` sweep has no retrieval columns populated.)
- **RAG / vector-RAG DP sweeps:** beyond the `base` sweep reported in §8, `rag` and `vector_rag` sweeps (30 rows each) exist. Both show identical behaviour — **accuracy flat at 1.0** across all 6 mechanisms × 5 noise levels — differing only in fixed MAE (rag 1.6080, vector_rag 2.1529), consistent with §4.4. No privacy–utility tradeoff is observable, matching the §8 finding.
- **Visualization artifact:** `silhouette_vs_noise_multiplier.png` (a Track-A plot) exists; it visualizes the silhouette column already tabulated in §8.

No DAIC (Track-C) evaluation supported by the original implementation was found to be omitted.

### 13.3 Phase 9 Requirement PASS/FAIL Checklist

| # | Phase 9 requirement | Status | Evidence in report |
|---|---|---|---|
| 1 | Every statement supported by an artifact/log/source/output | **PASS** | §13.1 traceability table |
| 2 | Every metric matches actual artifact values | **PASS** | §13.1 — all ✓ |
| 3 | No assumptions / estimates / hallucinated values | **PASS** | §13.1; "not measured" left unfilled |
| 4 | Missing supported evaluations identified | **PASS** | §13.2 (RAG latency, rag/vrag sweeps — trivial) |
| 5a | Evaluation metrics (acc/precision/recall/F1) | **PASS** | §4.3, §4.4 |
| 5b | PHQ prediction metrics (MAE, regression) | **PASS** | §4.2, §4.3 |
| 5c | Training metrics (loss, val curve) | **PASS** | §4.2 |
| 5d | Runtime metrics | **PASS** | §7 (+ §13.2 RAG latency 0.0 ms) |
| 5e | Modality contribution (text/audio/video) | **PASS** | §5 |
| 5f | Original limitations (no fixes proposed) | **PASS** | §10 (12 items) |
| 5g | Comparison with available execution/report evidence | **PASS** | §11 (dissertation absent, stated; 8B cross-check ✓) |
| 5h | Complete baseline report (setup/dataset/model/training/federated/runtime/strengths/limitations) | **PASS** | §1–§12 |
| 6 | PASS/FAIL checklist produced | **PASS** | this table |
| 7 | Missing sections appended | **PASS** | §13.2 appended (supplementary only) |

### 13.4 Audit conclusion

All Phase 9 requirements are satisfied. Every reported metric is artifact-backed and exact; the single rounding imprecision (P(pos) 0.366→0.365) has been corrected in-place; the only omissions were trivial supplementary Track-A metrics, now recorded in §13.2. No core DAIC baseline evaluation was missing.

**PHASE 9 COMPLETE — ORIGINAL BASELINE ESTABLISHED.**

---

*End of Phase 9 Baseline Evaluation Report. No code modified; no retraining performed; no Version-2 improvements proposed.*
