# Phase 9.5 — Step 6 Colab Execution Checklist (188-Participant Official Training)

Run cells top-to-bottom. ⏱ = approx time. 🖐 = your manual action required.

| # | Cell | What to expect | ⏱ | Expected output |
|---|---|---|---|---|
| 1 | **GPU check** | Confirms a GPU runtime is active | ~5 s | `CUDA available: True` · `Device: Tesla T4` (or similar) |
| 2 | **Install deps** | Installs `transformers==4.44.0` (+sklearn/pandas/pyarrow, `numpy<2`) | 1–3 min | `transformers 4.44.0 | AdamW import OK` |
| 3 | 🖐 **HF login** | Paste HF **read** token (gated model) | ~30 s | `Login successful` |
| 4 | 🖐 **Upload 2 files** | Upload `trainer_mentalbert_daic.py` + `daic_records.parquet` | 15–60 s | `records: 188 | cols: ['participant_id','text','phq_score']` · `pos(>10): 45 | neg: 143` (assert passes) |
| 5 | **Run trainer** | Downloads MentalBERT (438 MB, first time) then 3 GPU epochs | 3–6 min | `[INFO] loaded 188 records` · `[INFO] train=170 val=18` · `epoch 1/3 … 2/3 … 3/3` · `[EVAL] regression MAE=…` · `[EVAL] classification acc=… ` · `[DONE] training complete.` |
| 6 | **Show results** | Prints metrics + lists artifacts | ~2 s | JSON with `mae / acc / precision / recall / f1` + file sizes |
| 7a | 🖐 **Download small files** | Browser downloads 4 metric artifacts | ~15 s | 4 files land in your Downloads folder |
| 7b | 🖐 **Copy to Drive** | Mounts Drive, copies full `trainer_outputs/` | 1–3 min | `copied trainer_outputs -> /content/drive/MyDrive/Phase95_trainer_outputs` |

**Total wall-clock: ≈ 8–15 min.**

---

## Files produced in Colab (`/content/trainer_outputs/`)

| File | Size | |
|---|---|---|
| `training_report.json` | ~0.5 KB | metrics (mae/acc/prec/rec/f1) |
| `eval_preds.csv` | ~1 KB | 18 val-sample predictions |
| `modality_ablation.json` | ~0.25 KB | text/audio/video contribution |
| `mentalbert_delta.receipt.json` | ~0.2 KB | delta receipt |
| `mentalbert_privacy_subset.pt` | ~438 MB | trained checkpoint (207-key state) |
| `mentalbert_delta.pt` | ~438 MB | trained delta |
| `secure_store/mentalbert_delta.pt` | ~438 MB | delta mirrored to store |

## ⬇ MUST download back into local `BE-Major-Project/trainer_outputs/` (needed for Steps 7–9)

- [ ] `training_report.json`
- [ ] `eval_preds.csv`
- [ ] `modality_ablation.json`
- [ ] `mentalbert_delta.receipt.json`

(These 4 are what the Phase 9.5 report/audit and Step 7 read.)

## ☁ May remain on Google Drive (large — NOT required for Steps 7–9)

- `mentalbert_privacy_subset.pt` (~438 MB)
- `mentalbert_delta.pt` (~438 MB)
- `secure_store/mentalbert_delta.pt` (~438 MB)

> Federation (Step 8) uses the Track-A probe, not the MentalBERT delta; the report (Step 9) is driven by
> the metric JSONs. Keep the `.pt` files on Drive as the model-of-record; pull one locally only if you
> later want the checkpoint on disk.

---

## ✅ Step 6 PASS checklist (all must be true)

- [ ] Cell 1: `CUDA available: True`
- [ ] Cell 2: `transformers 4.44.0 | AdamW import OK`
- [ ] Cell 3: HF `Login successful` (no 401 on the gated model in Cell 5)
- [ ] Cell 4: `records: 188`, `pos(>10): 45 | neg: 143`, assert passed
- [ ] Cell 5: `[INFO] train=170 val=18` and `[DONE] training complete.` printed (no traceback)
- [ ] Cell 6: `training_report.json` shows numeric `mae`, `acc`, `precision`, `recall`, `f1`
- [ ] 4 small artifacts downloaded and placed in local `trainer_outputs/`
- [ ] (optional) full `trainer_outputs/` copied to Drive

When every box above is ticked, tell me "Colab done" and I will continue automatically with
**Step 7 (ingest & verify official artifacts + 113→188 comparison) → Step 8 (federation) → Step 9 (final report + audit).**
