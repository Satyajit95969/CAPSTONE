# Phase 9.5 — Step 5: Google Colab Training Package

Official Phase 9.5 baseline training runs on **Google Colab (GPU)** using the complete
**188-participant** DAIC-WOZ dataset. Ritik's implementation is preserved exactly — no code
change, no architecture change, every original hyperparameter retained.

---

## 1. Package folder structure

```
colab_package/
├── Phase95_Colab_Training.ipynb   # the notebook you run in Colab (7 cells)
├── trainer_mentalbert_daic.py     # ORIGINAL trainer, byte-for-byte unmodified
├── daic_records.parquet           # 188 records (participant_id, text, phq_score)
└── README_COLAB.md                # this file
```

## 2. Upload list (upload EXACTLY these two files in Cell 4)

| File | Size | What it is |
|---|---|---|
| `trainer_mentalbert_daic.py` | ~30 KB | Original trainer (no local imports; fully self-contained) |
| `daic_records.parquet` | ~916 KB | 188 records · pos(>10)=45 · neg=143 · text-only |

Nothing else is required — the trainer imports only PyPI packages plus its own inline
`SecureStoreFallback`, and downloads MentalBERT from the Hugging Face Hub at runtime.

## 3. Hyperparameters (preserved from Phase 8B — DO NOT CHANGE)

| Param | Value |
|---|---|
| mode | supervised |
| epochs | 3 |
| batch-size | 8 |
| lr | 2e-5 |
| val-split | 0.1  → train=170, val=18 (seed 42) |
| binarize / threshold | on / 10.0 |
| model | `mental/mental-bert-base-uncased` |
| device | **auto** (Colab GPU via the trainer's own default; the only environment difference) |

## 4. Output locations (produced in Colab at `/content/trainer_outputs/`)

| File | Approx size | Role |
|---|---|---|
| `training_report.json` | ~0.5 KB | **MAE / acc / precision / recall / f1** — primary metrics |
| `eval_preds.csv` | ~1 KB | per-sample validation predictions (18 rows) |
| `modality_ablation.json` | ~0.25 KB | text/audio/video contribution |
| `mentalbert_delta.receipt.json` | ~0.2 KB | delta receipt |
| `mentalbert_privacy_subset.pt` | ~438 MB | trained checkpoint (207-key state dict) |
| `mentalbert_delta.pt` | ~438 MB | trained delta vs base |
| `secure_store/mentalbert_delta.pt` | ~438 MB | delta mirrored to store |

## 5. Download-back procedure (feeds Steps 7–9, which run locally)

- **Cell 7a** browser-downloads the four small artifacts. Place them into the local repo at:
  ```
  BE-Major-Project/trainer_outputs/
      training_report.json
      eval_preds.csv
      modality_ablation.json
      mentalbert_delta.receipt.json
  ```
- **Cell 7b** copies the full `trainer_outputs/` (incl. the two 438 MB `.pt` files) to
  `MyDrive/Phase95_trainer_outputs/`. Only needed if you want the checkpoint locally; the
  Phase 9.5 report/audit is driven by the small metric files above.

> The local `trainer_outputs/` currently holds an **unofficial** local 188-record run
> (`trainer_outputs/local_188_preliminary/`) and the preserved **113 baseline**
> (`trainer_outputs/phase9_113_baseline/`). The Colab download becomes the **official** Phase 9.5
> 188-participant result and supersedes the local run.

## 6. Manual interaction required (Colab only)

You (the human) must do these in the browser — they cannot be automated from this environment:
1. Set the runtime to GPU.
2. Cell 3: paste your Hugging Face read token (gated model).
3. Cell 4: upload the two files.
4. Cell 7: approve the downloads / mount Drive.

Everything else runs unattended. When the Colab run finishes and the artifacts are back in
`trainer_outputs/`, tell me and I will continue automatically with Steps 7 → 8 → 9.
