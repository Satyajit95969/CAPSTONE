# Phase 16 / Experiment 7 — Multimodal Activation

**Status:** implemented, gate-tested, smoke-tested. **The official 5×5×4 run has NOT been executed** — it requires GPU (see §8).

**Single changed factor: INPUT MODALITY.**

---

## 1. Arms (pre-registered, fixed before any result was seen)

| Arm | Modality | `audio_dim` | `vision_dim` | Fusion input | Role |
|---|---|---|---|---|---|
| **A0** | text only | `None` | `None` | 768 | **HARD GATE** — artifact equivalence control |
| **A1** | text + audio | 154 | `None` | 896 | does audio contribute |
| **A2** | text + vision | `None` | 84 | 896 | does vision contribute |
| **A3** | text + audio + vision | 154 | 84 | 1024 | full multimodal, **primary** |

Routing needs **no trainer change**: `MultiModalModel.__init__` already gates encoder construction on `dim is not None and dim > 0` (trainer lines 247–251), and `forward()` ignores `audio_vec`/`vision_vec` when the matching encoder is absent. The full batch is passed to every arm, so **one code path** executes for all four — the same uniformity argument Exp 6 used for λ, and what makes the A0 gate transitive.

## 2. What is held frozen

Copied unchanged from Baseline-CV / Exps 3–6: `fold_manifest.json` (5 participant-level folds), 5 repeats = 25 fold-runs/arm, `BASE_SEED=1000`, `EPOCHS=3`, `LR=2e-5`, `BATCH_SIZE=8`, `MAX_LEN=128`, `λ=0.5`, `GRAD_CLIP=1.0`, `T.AdamW` (transformers', **not** torch's), the 12 primary metrics, fold-level Student-t CI (df=4, `t_crit=2.7764451051977987`, repeats averaged within fold), the two-stage seeding (`seed=BASE_SEED+r`, then `torch.manual_seed(seed*100+fold)` immediately before model construction), and the `seed → model → dataset → dataloader → loop` construction order, which is normative because `FusionHead` has `Dropout(0.2)` and the train loader shuffles.

## 3. Why A0 is a valid control

A0 constructs `(None, None)` → no `SmallMLP` is built → fusion input 768 → the torch RNG stream is consumed exactly as Baseline-CV consumed it. A0 therefore isolates one question: **did swapping the artifact change the text-only result?** It must not.

## 4. A declared, unavoidable asymmetry

A1/A2/A3 construct `SmallMLP` encoders **before** `FusionHead`, so their fusion head initialises from a different RNG position than A0's. This is inseparable from adding a modality — the extra parameters must be drawn from somewhere — and is a property of the factor, not a confound. Declared here rather than discovered later.

## 5. The text-ablation problem and its resolution

Trainer line 442:

```python
tokenizer = AutoTokenizer.from_pretrained(args.bert_model_name) if "args" in globals() else None
```

`args` is a **module global assigned by the trainer's own `main()`** (line 522, comment: *"used in modality_ablation_importance for tokenizer retrieval"*). Importing the module — which every runner and verifier does — leaves it undefined, the `else` branch fires, and `text_score` is forced to 0.0. Stage S4 observed exactly this.

**Evidence that script execution is the historical mechanism:** no runner (Baseline-CV, Exp 3/4/5/6) has ever called `modality_ablation_importance` — zero grep hits across all five. Every published number came from executing the trainer as a script:

| Artifact | audio | vision | text |
|---|---|---|---|
| `phase9_113_baseline/` | 0.0 | 0.0 | 1.9956 |
| `local_188_preliminary/` | 0.0 | 0.0 | 2.2672 |
| `trainer_outputs/` (Phase 9.5 official) | 0.0 | 0.0 | **2.3365** |

**Resolution:** `run_exp7_ablation.py` invokes the frozen trainer through its **own CLI in a subprocess**. `args` exists naturally. No source edit, no `T.args` injection, no fake globals, no rewritten ablation, no import of the trainer at all.

**Two consequences recorded in the output artifact:**
- Script mode uses a **single 90/10 split** (`val_split=0.1`), not the 5×5 CV — exactly what Phase 9.5 did (`train=170 val=18`). The number is like-for-like with 2.3365 and is **not** comparable to, and does not substitute for, the cross-validated arm metrics.
- The Phase 9.5 log shows local `.venv` paths and this machine's torch is CPU-only, so **the historical ablation was itself produced on CPU**. Running the Exp 7 ablation on CPU is therefore the *more* comparable execution context.

**Overwrite hazard, guarded:** the trainer writes `modality_ablation.json` into `Path(--out-path).parent`, and its default parent is `trainer_outputs/` — the directory holding the frozen Phase 9.5 artifact. Every invocation is forced into an isolated per-run directory, the script refuses to run with `trainer_outputs/` as its output directory, and it re-checks the protected file's SHA afterwards.

## 6. Acceptance criteria

| ID | Statement | Role |
|---|---|---|
| **C0** | A0 reproduces Baseline-CV on all 12 primary metrics | **HARD GATE** |
| C1 | audio ablation ≠ 0 | activation |
| C2 | vision ablation ≠ 0 | activation |
| C3 | text contribution remains positive | guard |
| C4 | A3 PR-AUC improves over A0 beyond MDE (0.0950) | utility, *not* required for activation |

A non-zero ablation establishes **graph presence, not predictive usefulness**. Utility is C4, from the paired CV deltas.

## 7. Files created

```
exp7_modality/run_exp7.py             CV runner (fork of run_exp6.py)
exp7_modality/run_exp7_ablation.py    frozen-trainer script-mode ablation driver
exp7_modality/aggregate_exp7.py       authoritative A0 gate + aggregates + deltas
make_exp7_upload_bundle.py            Colab GPU bundle
PHASE_16_EXP7_DESIGN.md               this document
```

Nothing existing was modified.

## 8. Why the official run requires GPU

`run_exp7.py` inherits Exp 6's device guard: CPU aborts unless `--allow-cpu`, because device would be a **second factor** and would invalidate the A0 gate against a GPU-produced Baseline-CV.

Measured on this machine: **312 s/fold-run on CPU** → 100 fold-runs ≈ **9.4 hours**. Exp 6 measured **18.0 s/fold-run on Colab GPU** → ≈ **30 minutes**. The official run goes to Colab via `make_exp7_upload_bundle.py`.

## 9. Chained-comparator policy (restated, per `PHASE_15_EXP6_RESULTS.md`)

Exp 6 returned H₀, so the best accepted text-only configuration remains **Baseline-CV**. A0's binding comparator is therefore Baseline-CV, not Exp 6. `exp6_summary.json` is SHA-gated here as a **context** comparator only.
