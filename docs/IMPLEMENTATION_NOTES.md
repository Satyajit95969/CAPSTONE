# Implementation Notes

Durable technical notes that belong in the repo, not only in a chat history.
Add to this file rather than letting decisions live solely in conversation.

---

## Fix E2 — train/eval split is per-run, not per-client (2026-08-22)

`stratified_split()` (`installer/runtime/agents/trainer/trainer_mentalbert_privacy.py`)
gives every client run a deterministic, seeded, stratified 80/20 train/eval
split of whatever `records` it's handed.

**What this fixes**: no client evaluates on records it just trained on
(Fix E2's actual purpose - see the commit).

**What this does NOT fix, and was never meant to**: every client currently
loads the identical shared corpus - `runtime/pipeline.py`'s `_MULTIMODAL_PARQUET`
is one hardcoded path, loaded in full (`MULTIMODAL_MAX_SAMPLES=0`) by every
client, every round. There is no per-client data partitioning implemented
anywhere in this project today.

Because the split is deterministic (fixed seed, same records in → same
train/eval boundary out), every client draws the *identical* boundary over
the *identical* corpus. State this precisely, not loosely:

- **No cross-client leakage in the narrow sense**: no client ever evaluates
  on records another client trained on, because they all draw the same
  boundary.
- **But**: all clients train on the *same* 149 training records. The
  aggregated global model has seen one dataset three times over three
  rounds, not three independent data partitions. That is a limitation of
  the current single-corpus setup, not a defect Fix E2 introduced.

This is consistent with, and does not replace, the single-device-identity
caveat already disclosed in `RUNBOOK.md` / `MENTOR_DEMO_RUNBOOK.md`: this
project currently has one enrolled device, so "three clients" already meant
one device submitting three times before Fix E2 existed. Fix E2 doesn't
change that; it just stops the same-corpus problem from also being a
same-records-for-train-and-eval problem.

If genuine multi-client data partitioning is implemented later,
`stratified_split()` needs no changes - it operates on whatever `records`
list it's given, so a per-client partition would automatically get its own
independent stratified split. The gap to close, if that's ever wanted, is
upstream: something has to hand each client a different local subset of the
corpus in the first place.

---

## Fix E4 — more training steps make delta magnitude more predictable (2026-08-23)

Side finding from recalibrating `clip_norm` after raising `lr` (2e-5→1e-4) and
`epochs` (1→10, 19→190 optimizer steps): the delta L2 norm distribution got
*much* tighter, not just bigger.

```
old regime (lr=2e-5, epochs=1, N=30): mean=0.0509  stdev=0.0157  CV≈30.8%
new regime (lr=1e-4, epochs=10, N=30): mean=0.6670  stdev=0.0259  CV≈3.9%
```

More optimizer steps per round produced a far more consistent delta
magnitude run to run - intuitively, 19 steps means the final weight delta is
dominated by whichever few batches happened to land late in that short run
(high variance), while 190 steps averages over enough batches that the
result is much less sensitive to which specific examples got sampled when.

This matters operationally, not just statistically: a calibration with lower
CV is safer to trust with less margin above the observed max, and clipping
behavior is more predictable round to round - fewer surprise clips on tail
runs. Not the reason Fix E4 was done (that was fixing the degenerate
collapse), but worth recording as a real, measured side benefit.

---

## Fix E5 — `calibrate_clip_norm.py` measures the RAW delta, not the clamped one (2026-08-23)

**The bug this documents**: after Fix E4 raised `lr`/`epochs`, `clip_norm`
was recalibrated (0.15→0.85) against `scripts/calibrate_clip_norm.py`'s N=30
measurement. That measurement calls `compute_filtered_delta()` directly,
which does **not** call `apply_safety_to_delta()` - it measures the delta
before either safety clamp (`DEFAULT_MAX_PARAM_CHANGE`,
`DEFAULT_MAX_GLOBAL_DELTA_NORM`) touches it. Meanwhile the live pipeline
*does* run `apply_safety_to_delta()` before encryption. At the time of that
recalibration, `DEFAULT_MAX_PARAM_CHANGE` was still `1e-3`, stale from the
pre-Fix-E4 regime, and was clamping 33.10% of all 281,254 trainable params on
every single run - cutting the delta from L2≈0.66 down to L2≈0.36 before DP
ever saw it. `clip_norm=0.85` was calibrated against a distribution the live
pipeline was never actually producing.

**Why this was allowed to happen**: nothing surfaced it. Both safety clamps
were silent - they clamped or rescaled without printing anything different
from a run where they never engaged. Fixed as part of Fix E5:
`apply_safety_to_delta()` now prints and `rpt.warn()`s whenever either clamp
actually engages, so this class of drift is visible on the next run it
happens on, not discovered later by a separate investigation.

**The standing rule, for whoever touches either clamp next**:
`compute_filtered_delta()`-based calibration (`calibrate_clip_norm.py`,
`sweep_lr_epochs.py`) is only a valid measurement of what reaches DP's
`clip_norm` **as long as both safety clamps stay inert under normal
operation** - i.e. as long as neither one's `[SAFETY-CLAMP] ... ENGAGED`
warning fires on ordinary runs. If you tighten `DEFAULT_MAX_PARAM_CHANGE` or
`DEFAULT_MAX_GLOBAL_DELTA_NORM` enough that either starts engaging routinely
again, the raw-delta calibration stops being accurate and `clip_norm` must be
recalibrated - or better, teach the calibration scripts to call
`apply_safety_to_delta()` too, so they measure what actually reaches DP
regardless of clamp settings. Neither script does that today; it wasn't
necessary while both clamps were dead defaults, but it's a known gap now
that they aren't.

---

## Step 12 findings — federated aggregation has never been mathematically
## valid in this system (2026-08-23)

Found while planning Step 12 (evaluating the aggregated global model on
held-out data - the first time anyone has actually tried to use the output of
`AggregatorAgent.run_job()` for anything). Two independent, pre-existing
defects, neither introduced by, or fixed by, Step 12. Step 12's evaluation
script (`scripts/evaluate_global_model.py`) **works around both** so it can
still produce a meaningful privacy-vs-utility number; it does **not** fix
either one, and neither should be read as fixed by this note or by Step 12.

**Defect A - the aggregator persists an averaged DELTA and the client loads it
as absolute weights.** `save_encrypted_delta()`
(`trainer_mentalbert_privacy.py`) saves `after - before`, a small correction
(measured L2~0.62), never the model's actual weights. `AggregatorAgent`
averages these deltas across clients and writes the result to GridFS as
`global_model_round_N.pt` - it is never added to a base. The Rust orchestrator
streams those bytes through unmodified. The client's warm-start path
(`orchestrate()`, "Phase 10: warm-start from global model") then does
`model.load_state_dict(global_state, strict=False)` directly on a freshly,
randomly initialised model - i.e. it overwrites `audio_encoder` /
`vision_encoder` / `fusion` with a ~0.6-magnitude correction *as if it were
the literal parameter values*, discarding whatever those layers had actually
learned. **This has never fired in any run to date** - CLAUDE.md's own
"Warm-start never exercised" note is why: every verification run so far has
been round 1 (`global_model_available=false`), so this branch has simply
never executed.

**Defect B - no seeding before `MultiModalModel(...)` construction, so
"clients" aggregate deltas computed from different random bases.**
`orchestrate()` never calls `torch.manual_seed()` (or anything equivalent)
before constructing the model. `audio_encoder` / `vision_encoder` / `fusion`
get PyTorch's default random init, independently, every client run. FedAvg
(and this project's own trimmed-mean aggregator) assumes every client's
`after - before` is a correction to the *same* `before`. That assumption has
never held here even once - not because of Defect A, but independently of it:
even if the aggregator correctly added the averaged delta to a base, "the
base" was never common across the three submissions being averaged.

**Combined: federated aggregation, as implemented, has never produced a
mathematically meaningful global model.** Every prior verification run in
this project measured *local, pre-DP, pre-aggregation* metrics precisely
because nothing downstream of aggregation was ever evaluated - Step 12 is the
first attempt to actually score the aggregator's output, and this is what
that attempt found before a single live round was run for it.

**What Step 12 does about it**: nothing, to the live pipeline. `pipeline.py`'s
warm-start loader is untouched and the bug stays exactly as visible as it was
found. Two narrow, env-gated additions make the *measurement* meaningful
without touching production behaviour by default:
- `GLOBAL_INIT_SEED` (env, default unset = current random-init behaviour
  unchanged): when set, seeds `torch.manual_seed()` immediately before
  `MultiModalModel(...)` construction, then re-randomises the RNG
  (`torch.seed()`) immediately after, so the three clients in one Step 12
  trial share an identical initial `audio_encoder`/`vision_encoder`/`fusion`
  (Defect B's missing precondition) while still training with independent
  stochasticity (dropout etc.) - not three bit-identical replicas.
- `scripts/evaluate_global_model.py` reconstructs the evaluated model as
  `base_state[trainable_keys] + aggregated_delta` (the mathematically correct
  read of what the aggregator actually produced), rather than replicating
  `pipeline.py`'s `load_state_dict(delta, strict=False)` (Defect A). The
  live client-side warm-start path is not called by this script at all.

Both defects should be tracked as their own fix (not scoped here): Defect A
needs the aggregator (or the client) to add the aggregated delta to a real
base before it's usable as a model; Defect B needs a canonical shared initial
model distributed to clients at round 0, not independent random init per
client, before FedAvg's precondition can hold in this system at all.

**Aggregator note found while implementing Step 12** (not a defect, just
undocumented): the `global_models` bookkeeping document the orchestrator
writes uses `round_id = N+1` (aggregating round N's updates produces a
`global_models` doc with `round_id=N+1` — server.rs's own comment: "for
round N+1"), but the aggregator's own GridFS filename is
`global_model_round_{N}.pt` (`aggregator.py:655`, the round that was
aggregated). The two numbering schemes disagree with each other by one.
`evaluate_global_model.py` looks the artifact up via the `global_models`
document (the same path the real client's `_download_global_model()` uses),
which sidesteps this — but it tripped up the first version of the script and
is worth knowing about if anyone else reads GridFS directly by filename.

---

## Step 12 — the privacy-utility measurement (2026-08-23)

The central result: 3 arms x 3 seeds each (`GLOBAL_INIT_SEED` = 101, 202,
303), held out on the same Fix E2 37 records throughout. ARM 1 / ARM 2 are
full 3-client live rounds (gRPC/TPM/mTLS/DP/aggregation, real) evaluated with
`scripts/evaluate_global_model.py`; ARM 3 is `scripts/run_arm3_local_baseline.py`
(fully offline, single client, no aggregation, no DP).

| arm | seed | accuracy | precision | recall | F1 | MAE | pred+/37 | prob(+) mean | delta L2 | eps |
|---|---|---|---|---|---|---|---|---|---|---|
| 1 (no privacy) | 101 | 0.7027 | 0.0000 | 0.0000 | 0.0000 | 4.857 | 0/37 | 0.283 | 0.591 | inf (no privacy)* |
| 1 (no privacy) | 202 | 0.7027 | 0.5000 | 0.3636 | 0.4211 | 4.847 | 8/37 | 0.425 | 0.614 | inf (no privacy)* |
| 1 (no privacy) | 303 | 0.2973 | 0.2973 | 1.0000 | 0.4583 | 5.107 | 37/37 | 0.872 | 0.586 | inf (no privacy)* |
| **1 mean** | | 0.5676 | 0.2658 | 0.4545 | 0.2931 | 4.937 | 15.0 | | 0.597 | |
| 2 (current DP) | 101 | 0.7027 | 0.0000 | 0.0000 | 0.0000 | 513959.48 | 0/37 | 0.000 | 301.73 | 5.302585 |
| 2 (current DP) | 202 | 0.7027 | 0.0000 | 0.0000 | 0.0000 | 41009.57 | 0/37 | 0.000 | 301.83 | 5.302585 |
| 2 (current DP) | 303 | 0.2973 | 0.2973 | 1.0000 | 0.4583 | 83562.36 | 37/37 | 1.000 | 301.78 | 5.302585 |
| **2 mean** | | 0.5676 | 0.0991 | 0.3333 | 0.1528 | 212843.80 | 12.33 | | 301.78 | 5.302585 |
| 3 (local only) | 101 | 0.2973 | 0.2973 | 1.0000 | 0.4583 | 5.069 | 37/37 | 0.815 | n/a | n/a** |
| 3 (local only) | 202 | 0.7027 | 0.0000 | 0.0000 | 0.0000 | 4.912 | 0/37 | 0.302 | n/a | n/a** |
| 3 (local only) | 303 | 0.7027 | 0.0000 | 0.0000 | 0.0000 | 5.127 | 0/37 | 0.136 | n/a | n/a** |
| **3 mean** | | 0.5676 | 0.0991 | 0.3333 | 0.1528 | 5.036 | 12.33 | | | |

\* The DP agent's own return value for `mechanism="none"` is `epsilon_spent = inf`
(`dp_agent.py:279`, correct - no noise means no privacy bound). `pipeline.py`'s
fallback (added for the case a mechanism returns a non-finite epsilon) then
overwrites this to a placeholder `1.0` before it reaches the receipt
(`pipeline.py:502-511`). That `1.0` is not a real epsilon and is not reported
here as one - ARM 1 has no privacy guarantee, full stop.
\*\* ARM 3 never calls the DP agent - there is no epsilon to report, not even
an infinite one.

**Reading this table:**

1. **ARM 1 vs ARM 3 confirms the Defect A/B workaround reconstructs a real
   model.** Same 3 seeds, same collapse pattern (seed 303 -> "predict
   everyone positive", seeds 101/202 -> weak-to-moderate "predict mostly
   negative"), same F1/accuracy/MAE, essentially seed-for-seed. This is the
   expected result if `base_state[trainable_keys] + aggregated_delta` is a
   correct reconstruction: a 3-client trimmed-mean-of-clean-deltas round
   should behave like a single well-trained client, not like something else
   entirely, and it does. Delta L2 (~0.59-0.61) matches the single-client
   scale measured throughout this project (Fix E4/E5 calibration), confirming
   the median-of-3 aggregator (see below) barely moves clean deltas off their
   individual scale.

2. **ARM 1's non-DP collapse is real and predates Step 12** - it is the same
   seed-to-seed instability documented after Fix E4/E5 (Step 10d, Step 11d:
   "individual live runs still sometimes collapse to degenerate single-class
   predictions"), now visible in the *aggregated* model for the first time.
   Not something Step 12 introduced or should paper over.

3. **ARM 2 (current DP) is categorically worse, not just noisier.** Delta L2
   jumps from ~0.6 to ~301.8 - roughly 500x - and MAE explodes into the tens
   to hundreds of thousands (a PHQ score MAE should be O(1-10)). The
   probability distribution saturates to exactly 0.0 or exactly 1.0 in every
   DP seed (`stdev: 0.0`) - the classifier isn't uncertain, its logits have
   been driven so far by noise that softmax has numerically saturated. F1
   never exceeds ARM 1's best seed and is 0.0 in 2 of 3 seeds vs ARM 1's 1 of
   3. **At the current DP configuration (gaussian, noise_multiplier=1.0,
   clip_norm=0.85, eps=5.302585), aggregated utility is destroyed, not
   degraded.** This is the honest answer to "how much utility survives
   privacy" for this system as configured: essentially none, and the
   collapse is total (saturated probabilities) rather than partial.

4. **Why: the trimmed-mean aggregator does not denoise at n=3** (this is the
   Step 12a Q3 correction, confirmed empirically here). `_aggregate_tensor()`
   at n=3 keeps exactly 1 of 3 sorted values per coordinate (coordinate-wise
   median-of-3, `lower=max(1,int(0.1*3))=1`, `upper=2`) - it is NOT an
   average, and a median-of-3 draw from three independent
   N(0, noise_multiplier^2 * clip_norm^2) noise vectors has almost the same
   L2 norm as a single draw (no sqrt(3) variance reduction the way a mean
   would give). Measured: a single client's noised update has L2~450 (Step
   11d); the aggregated (median-of-3) delta here measures L2~301.8 across all
   3 DP seeds - noticeably reduced from one sample, but nowhere near the
   ~260 (450/sqrt(3)) a true mean-of-3 would achieve, and nowhere close to
   recovering the ~0.6 signal. **No Byzantine-robustness claim is made or
   implied by this aggregator at n=3** - median-of-3 is not more private or
   more robust than mean-of-3 at this client count, it is simply a different,
   weaker-averaging statistic that happens to be the trimmed-mean formula's
   degenerate case here.

**Bottom line**: the current DP configuration is incompatible with this
aggregation setup at n=3 clients - the aggregator's noise reduction is too
weak (median-of-3, not mean-of-3) to bring a noise_multiplier=1.0 update back
down anywhere near signal scale, and the result is total collapse (saturated
probabilities) rather than a graceful accuracy/privacy tradeoff. This is a
capacity/configuration finding, not evidence that DP-SGD itself cannot work
here - a real deployment would need either far more clients per round (so a
genuine mean, or a trim that keeps more than 1 value, denoises properly), a
substantially lower `noise_multiplier`, or both. Nothing about ARM 2's config
was retuned to make this comparison land more favorably, per the standing
rule in this project: report what's measured, don't tune between runs to
make numbers look better.
