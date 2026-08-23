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
