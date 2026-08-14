# PHASE 11.6 — BASELINE-CV COLAB EXECUTION PACKAGE

**Goal:** Execute the official Baseline-CV experiment — the frozen MentalBERT trainer under participant-level cross-validation — on Google Colab GPU, introducing **no** experimental confound.

**The single change vs Phase 9.5:** the one seed-42 90/10 split is replaced by the frozen participant-level stratified 5-fold manifest. Architecture, optimizer, loss, hyperparameters, tokenizer, preprocessing, labels, and dataset are **unchanged** (the trainer file is byte-identical, SHA `65b1902e…a230b`).

Package pre-verified locally (non-training paths): scripts compile; trainer SHA matches; `infer_dims → (None, None)` (text-only architecture preserved); folds disjoint (zero leakage); 150–151 train / 37–38 test per fold; all 188 participants tested exactly once.

---

## 1. Exact Execution Plan

| Step | Action | Confound control |
|---|---|---|
| 1 | Colab runtime → **GPU** (T4 or better) | [E] Frozen baseline is Colab GPU; GPU keeps **device** constant → the only changed factor is the split. The driver **aborts on CPU** unless `--allow-cpu`. |
| 2 | Install pinned deps: `transformers==4.44.0`, `torch`, `pandas`, `pyarrow`, `scikit-learn` | Preserves the frozen environment (4.44.0 required for the trainer's `AdamW` import). |
| 3 | Authenticate to Hugging Face (gated MentalBERT) | Same model access as the frozen run — not a new factor. |
| 4 | Upload the frozen files: `trainer_mentalbert_daic.py`, `daic_records.parquet`, `trainer_outputs/baseline_cv/fold_manifest.json`, and this `colab_baseline_cv/` folder | Uses the **already-generated** manifest; no re-split. |
| 5 | For **r = 1..R** run `run_baseline_cv.py --repeat r` (each processes all 5 folds) | Per-repeat seed = `1000+r`; distinct seeds sample training stochasticity → noise band. |
| 6 | Run `aggregate_baseline_cv.py` | Produces the official comparator, noise band, and MDE. |
| 7 | Download `trainer_outputs/baseline_cv/` back into the repo | Preserves all artifacts for the final Baseline-CV report. |

**Recommended R = 5** (→ 25 fold-runs). At ~1–3 min/fold on a T4, budget ~1–2 GPU-hours.

---

## 2. Required File Structure

```
/content/be_project/
├── trainer_mentalbert_daic.py            # FROZEN trainer (SHA 65b1902e…a230b) — NOT modified
├── daic_records.parquet                   # frozen dataset (188 participants)
├── colab_baseline_cv/
│   ├── run_baseline_cv.py                 # fold driver (wraps frozen trainer)
│   ├── aggregate_baseline_cv.py           # aggregation / comparator / MDE
│   └── PHASE_11_BASELINE_CV_COLAB_PACKAGE.md
└── trainer_outputs/
    └── baseline_cv/
        ├── fold_manifest.json             # frozen 5-fold manifest (already generated)
        ├── trivial_control_arm.csv        # already generated (training-free control)
        └── runs/                          # created by the driver
            ├── rep1_fold1_preds.csv …     # held-out predictions per fold/repeat
            ├── rep1_metrics.csv …         # per-repeat metric tables
            ├── baseline_cv_summary.json   # final comparator (aggregation output)
            └── baseline_cv_summary.md
```

> `run_baseline_cv.py` must run with the repo root as the working directory so `import trainer_mentalbert_daic` resolves to the frozen file (its SHA is checked at startup).

---

## 3. Commands / Scripts to Execute Each Fold

**Colab cell 1 — environment**
```bash
%cd /content/be_project
pip -q install "transformers==4.44.0" torch pandas pyarrow scikit-learn
python -c "import torch; assert torch.cuda.is_available(), 'Select a GPU runtime'; print('GPU:', torch.cuda.get_device_name(0))"
```

**Colab cell 2 — Hugging Face auth (gated MentalBERT)**
```python
from huggingface_hub import login
login()   # paste a token with access to mental/mental-bert-base-uncased
```

**Colab cell 3 — run all folds for each repeat**
```bash
for r in 1 2 3 4 5; do
  python colab_baseline_cv/run_baseline_cv.py --repeat $r
done
```
Each invocation: verifies the trainer SHA, loads the frozen manifest, and for every fold trains the frozen trainer on the 4/5 partition (`val_dataset=None` → full partition; the internal 10% holdout that CV replaces is removed) and evaluates on the held-out 1/5. It writes `rep{r}_fold{f}_preds.csv` and `rep{r}_metrics.csv`.

*(No fold-specific command differs — the fold identity comes from the frozen manifest, not the command line. This is deliberate: it makes every fold-run reproducible from `(repeat, manifest)` alone.)*

---

## 4. Aggregation Procedure

**Colab cell 4**
```bash
python colab_baseline_cv/aggregate_baseline_cv.py \
    --runs-dir trainer_outputs/baseline_cv/runs \
    --out trainer_outputs/baseline_cv/baseline_cv_summary
```

The aggregator:
1. Concatenates all `rep*_metrics.csv` (R repeats × 5 folds).
2. For each primary metric computes **mean, fold SD, and a fold-level Student-t 95% CI** (df = k−1; the fold — not the fold-run — is the unit of independence, so the R×k correlated rows are **not** treated as iid).
3. Decomposes the noise band into **partition SD** (across-fold variability of the per-fold mean) and **training-stochasticity SD** (across-repeat variability within a fold, averaged; reported as `n/a` when R < 2).
4. Computes the **MDE** using a **paired** design with Student-t: because every future experiment reuses this same manifest, Version 2 is compared to Baseline-CV *per fold*; MDE(95%) = t(0.975, k−1) · SD(per-fold mean)/√k. This is a *planning* estimate of the noise scale; the operative accept/reject test at experiment time uses the SD of the **paired per-fold differences** (V2_f − BaselineCV_f).
5. Writes `baseline_cv_summary.json` and `.md`.

---

## 5. Final Metrics That Must Be Reported

Per fold **and** aggregated (mean ± SD ± 95% CI):

**Regression**
- MAE, RMSE
- Prediction variance (anti-collapse; must be materially > 0 to indicate learning)
- Mean-predictor MAE (trivial control, per fold)

**Classification**
- ROC-AUC, PR-AUC *(threshold-free)*
- Balanced Accuracy, MCC *(at the trainer's own argmax = 0.5 — unchanged, no threshold optimization)*
- Precision, Recall, F1 *(threshold-conditional, same operating point)*
- Accuracy *(reported only; **not** a decision metric — protocol §3.3)*

**Derived**
- Noise band (total SD; partition vs training-stochasticity components)
- Minimum Detectable Effect (paired, 95%) per metric

These populate the `[PENDING-GPU]` rows of `PHASE_11_BASELINE_CV.md` §3.2, §4.2, §5, §6, §7.

---

## 6. Artifact Structure for Reproducibility

Preserve all of the following (download the whole `trainer_outputs/baseline_cv/` tree):

| Artifact | Purpose |
|---|---|
| `fold_manifest.json` | The frozen partition — the definition of the experiment. Reused by every future experiment. |
| `runs/rep{r}_fold{f}_preds.csv` | Held-out per-participant predictions — allows any metric to be recomputed or audited. |
| `runs/rep{r}_metrics.csv` | Per-repeat, per-fold primary metrics. |
| `baseline_cv_summary.json` / `.md` | The official comparator, noise band, and MDE. |
| **Recorded provenance** (in the final report) | trainer SHA `65b1902e…a230b`; exact CLI; R; per-repeat seeds (`1000+r`); GPU model; `transformers==4.44.0`; manifest generator (`StratifiedKFold(5, shuffle=True, random_state=42)`). |

**Reproducibility contract (protocol P-3, as clarified in 11.5):** the manifest, environment, per-repeat seeds, and **aggregate statistics** are regenerable. A single training run is not required to be bitwise-identical (GPU residual nondeterminism is part of the measured noise); the aggregate comparator is the reproducible object.

---

## 7. Confound Checklist (verify before accepting the run)

- [ ] Ran on **GPU** (driver aborts on CPU) — device held constant vs Phase 9.5.
- [ ] Trainer SHA `65b1902e…a230b` confirmed at startup — architecture/optimizer/loss/hyperparameters unchanged.
- [ ] `infer_dims → (None, None)` — text-only architecture, audio/video absent as in the frozen run (invariant: ablation stays 0/0).
- [ ] Folds from the frozen manifest only — the **only** changed factor is the split.
- [ ] `transformers==4.44.0` — environment preserved.
- [ ] Dataset unchanged: 188 / 45 / 143; labels untouched.
- [ ] All 188 participants tested exactly once across the 5 folds.

If every box is checked, the run is a clean one-factor change and its output is the admissible official Baseline-CV comparator.
