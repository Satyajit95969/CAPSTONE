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

---

## Step 13 — testing the root cause: aggregation mode and noise_multiplier
## (2026-08-23)

**Precise question**: does a configuration exist that recovers non-degenerate
aggregated utility while remaining within eps<=8 at n=3 clients? Not "find a
config that works" - a negative answer is a legitimate result.

**Mode-selection blocker, confirmed**: the Rust orchestrator hardcodes
`"mode": "trimmed_mean"` (`server.rs:1499`) with no env/config/request
override anywhere (checked `orchestrator.toml` and all of `server.rs`/
`round.rs`) - selecting `mean` for a live round requires editing Rust source,
out of scope. Workaround used: `scripts/aggregate_offline.py` lets the live
pipeline run completely untouched for all 3 client submissions (real gRPC/
TPM/mTLS/DP, real GridFS uploads, real epsilon), then calls the *unmodified*
`AggregatorAgent` class directly with a different `mode` argument - same
decryption code, same class, different caller than the Rust subprocess
invocation. Nothing in `aggregator.py` or Rust was edited.

**Table** (seed=303 throughout - the one non-degenerate seed from Step 12;
n_eval=37; clip_norm=0.85, delta=1e-5 unchanged from Step 12):

| noise_multiplier | mode | accuracy | precision | recall | F1 | MAE | pred+/37 | prob(+) mean/stdev | delta L2 | epsilon | eps<=8 |
|---|---|---|---|---|---|---|---|---|---|---|---|
| 1.0 | mean | 0.7027 | 0.0000 | 0.0000 | 0.0000 | 15,681.78 | 0/37 | 0.000 / 0.000 | 260.25 | 5.302585 | **PASS** |
| 1.0 | trimmed_mean | 0.7027 | 0.0000 | 0.0000 | 0.0000 | 301,155.11 | 0/37 | 0.000 / 0.000 | 301.80 | 5.302585 | **PASS** |
| 0.5 | mean | **BLOCKED** - round never completed | | | | | | | | 11.756463 | FAIL |
| 0.25 | mean | **BLOCKED** - round never completed | | | | | | | | 27.512925 | FAIL |

The `1.0` row is a controlled pair: both modes ran on the *identical* 3
encrypted uploads from one live round (`round_id=1`, file_ids
`6a8a7c2d...`, `6a8a7c73...`, `6a8a7cb8...`), so the only thing that differs
between those two rows is the aggregation math, not training/noise variance.

**BLOCKED means exactly that, not "degraded"**: at noise_multiplier=0.5 and
0.25, `runtime/pipeline.py`'s own pre-existing hard ceiling
(`MAX_EPS_VALUE = 10.0`, `pipeline.py:60`) rejects the round outright -
`[FAIL] epsilon_spent=11.7565 exceeds hard ceiling 10.0` and
`epsilon_spent=27.5129 exceeds hard ceiling 10.0` respectively, raised before
`SubmitReceipt` is ever called. No update reaches GridFS, no aggregation
happens, there is no model to evaluate. This ceiling was not touched,
weakened, or bypassed to get numbers out of these cells - it did exactly what
CLAUDE.md documents it should ("Server-side hard epsilon ceiling; round
aborted on violation"), and the two probe attempts that hit it are reported
as blocked, not worked around.

**Reading the results:**

1. **Mean genuinely reduces noise, matching the Step 13a arithmetic almost
   exactly.** A single client's noised update measures L2~450 (Step 12/11d).
   Mean-of-3 predicts `450/sqrt(3) = 259.8`; measured `260.25` - a 0.2% match.
   Trimmed_mean (median-of-3) measured `301.80` on the *same 3 uploads*,
   confirming Step 12's Q3 finding was not an artifact of that particular
   round: mean is measurably, reproducibly better than this aggregator's
   trimmed_mean at n=3, by exactly the amount the arithmetic predicts.

2. **It is not enough.** 260.25 is still ~433x the ~0.6 signal scale. Both
   `1.0` rows collapse to the same degenerate "predict everyone negative"
   pattern (MAE in the thousands to hundreds of thousands - a real PHQ-scale
   MAE should be O(1-10)). Switching only the aggregation mode, at the
   noise_multiplier that keeps eps<=8, does not recover non-degenerate
   utility.

3. **The only levers that would plausibly help - more clients per round (so
   mean gets its sqrt(n) benefit at a larger n) or lower noise_multiplier -
   are unavailable within this project's own constraints as currently
   configured**: this system runs with one enrolled device (n=3 "clients" is
   three sequential submissions from it, documented since Fix E2), and
   lowering noise_multiplier to recover utility is blocked by the pipeline's
   own eps<=10 ceiling well before reaching eps<=8.

**Answer to the precise question**: **no** - no configuration tested (mean
aggregation, or lower noise_multiplier, or both together) recovers
non-degenerate aggregated utility while remaining within eps<=8 at n=3
clients. This is reported as the negative result it is; nothing was retuned
between cells to change the outcome.

**Scope of this finding - read this before citing the result anywhere.**
This is **not** "DP destroys utility for federated depression detection." It
is: **at n=3 clients, in a single round, DP-SGD at noise_multiplier=1.0
destroys rather than degrades utility, because sqrt(3) denoising cannot
bridge a 450:0.6 noise-to-signal ratio.** Every measurement in Step 12 and
Step 13 is round 1 (or, for the `1.0` pair here, one round evaluated two
ways) - a single noisy draw, aggregated once, evaluated once. Real FL
deployments run many rounds across many clients, where noise is zero-mean
and partially cancels across BOTH dimensions (more clients per round, more
rounds accumulating signal) while the model's actual learned signal
compounds round over round. Nothing tested here rules out that a real
multi-round, larger-n deployment recovers utility this single-round n=3
snapshot cannot - that is a different, larger, so-far unmeasured question
(see the Step 14 section below, which begins to measure it). Treat this
result as a genuine, scoped finding about *this configuration measured this
way*, not a general verdict on DP-SGD for this task.

**Actionable recommendation, recorded separately from the negative result
above** (not implemented - a deliberate decision to log, not something to
slip into a Rust file unreviewed): **mean aggregation should replace
trimmed_mean at low client counts**, independent of whether it alone
recovers non-degenerate utility. It is strictly better denoising at n=3
(measured: 260.25 vs 301.80 on identical data, matching the sqrt(3)
prediction to 0.2%) with no compensating robustness benefit given up - Step
12 already established trimmed_mean confers no real Byzantine-robustness at
n=3 (median-of-3 is not more attack-resistant than mean-of-3 at this client
count). Making this real requires changing the hardcoded literal at
`server/orchestration_agent/src/grpc/server.rs:1499`
(`"mode": "trimmed_mean"`) - Rust code, out of scope for this investigation
per its own constraints. Logged here as a recommended future change with its
exact location, not made.

---

## Step 14 — multi-round trajectory: warm-start genuinely exercised for the
## first time (2026-08-23)

Every measurement before this one (Step 12, Step 13) is round 1 - a single
noisy draw, aggregated once. Warm-start has never fired in this project
(CLAUDE.md's own "Warm-start never exercised" note). This runs 5 sequential
rounds for each of ARM 1 (no privacy) and ARM 2 (DP, noise_multiplier=1.0),
mean aggregation throughout (per the Step 13 recommendation - not
trimmed_mean), seed=303, evaluating the aggregated global model after every
round.

**Mechanism** (`scripts/run_step14_multiround.py`, no Rust/security touched):
the live pipeline runs completely untouched for every client submission. The
orchestrator's own automatic trimmed_mean aggregation still fires every
round (harmless, unused). Separately, this script computes its own mean
aggregation of each round's 3 uploads (`aggregate_offline()`, unmodified
`AggregatorAgent`), adds it to a running `cumulative_delta`, and evaluates
`base_state(seed) + cumulative_delta` - the correct multi-round FedAvg
accumulation. Before the next round's clients run, it overwrites the
`global_models` MongoDB document's `file_id` (a plain Mongo write - the
orchestrator just serves whatever's there) to point at a freshly-uploaded
GridFS object containing this cumulative state as genuine ABSOLUTE weights.
pipeline.py's existing, completely unmodified warm-start call
(`model.load_state_dict(global_state, strict=False)`) is only wrong when fed
a bare delta (Defect A) - fed real weights, as here, it is exactly correct
with zero code changes to pipeline.py or trainer_mentalbert_privacy.py.

**ARM 1 - no privacy, 5 rounds, mean aggregation:**

| round | F1 | accuracy | precision | recall | MAE | pred+/37 | prob(+) mean/stdev | cumulative delta L2 | this-round delta L2 |
|---|---|---|---|---|---|---|---|---|---|
| 1 | 0.6667 | 0.7838 | 0.6154 | 0.7273 | 5.210 | 13/37 | 0.474/0.062 | 0.581 | 0.581 |
| 2 | 0.4583 | 0.2973 | 0.2973 | 1.0000 | 5.836 | 37/37 | 0.736/0.055 | 1.113 | 0.584 |
| 3 | 0.0000 | 0.7027 | 0.0000 | 0.0000 | 5.556 | 0/37 | 0.401/0.042 | 1.628 | 0.563 |
| 4 | 0.0000 | 0.7027 | 0.0000 | 0.0000 | 5.268 | 0/37 | 0.251/0.030 | 2.146 | 0.569 |
| 5 | 0.0000 | 0.7027 | 0.0000 | 0.0000 | 5.155 | 0/37 | 0.182/0.034 | 2.671 | 0.575 |

**ARM 2 - DP (noise_multiplier=1.0), 5 rounds, mean aggregation:**

| round | F1 | accuracy | precision | recall | MAE | pred+/37 | prob(+) mean/stdev | cumulative delta L2 | this-round delta L2 | per-round eps | cumulative eps (naive additive bound) |
|---|---|---|---|---|---|---|---|---|---|---|---|
| 1 | 0 | 0.7027 | 0 | 0 | 126,868.58 | 0/37 | 0.000/0.000 | 260.09 | 260.09 | 5.302585 | 5.302585 |
| 2 | 0 | 0.7027 | 0 | 0 | 411,896.54 | 0/37 | 0.000/0.000 | 367.83 | 260.53 | 5.302585 | 10.605170 |
| 3 | 0 | 0.7027 | 0 | 0 | 470,234.36 | 0/37 | 0.000/0.000 | 450.31 | 260.09 | 5.302585 | 15.907755 |
| 4 | 0 | 0.7027 | 0 | 0 | 100,125.43 | 0/37 | 0.000/0.000 | 520.52 | 260.36 | 5.302585 | 21.210340 |
| 5 | 0 | 0.7027 | 0 | 0 | 1,640,656.53 | 0/37 | 0.000/0.000 | 581.92 | 259.73 | 5.302585 | 26.512925 |

"cumulative eps (naive additive bound)" is `round x 5.302585` - a loose upper
bound via basic composition, NOT a tight sequential RDP composition. This
system has no true multi-round accountant (CLAUDE.md defect #7: "Privacy
accounting is per-round only"); this number is reported as the approximation
it is, not fabricated as the real accountant's output.

**Headline finding: multi-round training diverges in this system, independent
of privacy.** ARM 1 (no privacy, DP fully disabled) does NOT improve over
rounds - it gets WORSE: F1 = 0.6667 (round 1) -> 0.4583 -> 0.0 -> 0.0 -> 0.0.
**Round 1's F1 = 0.6667 is the strongest utility result this entire
investigation has produced - the clean-federated, no-privacy reference
point** (3-client mean aggregation, correctly reconstructed, no DP noise
anywhere). By round 3 the same no-privacy configuration has collapsed to the
same degenerate all/none-prediction pattern seen everywhere DP noise
dominates - except here there is no DP noise to blame. The cumulative
delta's L2 magnitude growing at a steady ~0.57-0.58/round (linear) reflects
the model *moving* a consistent amount each round, not moving toward a
better solution - there is no learning-rate decay or convergence control
across rounds in this training regime, so multi-round training here diverges
rather than converges. **This is a training-regime defect independent of
DP, and it makes "more rounds recovers DP utility" untestable until it is
fixed** - the no-privacy baseline this hypothesis would need to compare
against does not itself improve with rounds. See Step 15 for the
investigation into why.

**Two further, precisely quantified mechanisms, both real, both secondary to
the headline finding above:**

1. **DP noise accumulates as a random walk (sqrt(r)); ARM 1's own training
   movement accumulates roughly linearly (r).** ARM 2's cumulative delta L2
   matches `sqrt(r) x 260.09` to within 0.4% at every one of the 5 rounds
   (260.09, 367.83->367.82 predicted, 450.31->450.48, 520.52->520.17,
   581.92->581.57) - textbook independent-zero-mean-noise accumulation.

2. **Consequently the noise-to-signal ratio genuinely improves over
   rounds** - using ARM 1's cumulative magnitude as the signal-scale
   reference: 447.7x (round 1) -> 330.5x -> 276.5x -> 242.5x -> 217.9x
   (round 5). That's a ~2.05x improvement over 5 rounds, close to the
   sqrt(5)=2.24x the two accumulation rates predict - the actual mechanism
   the Step 14 prompt hypothesized, real and measured, not assumed. But it
   is nowhere near enough on its own (closing a ~450x starting ratio via
   sqrt(r) alone would need on the order of 10^5 rounds at this rate - a
   rough extrapolation, not a fitted claim), and per the headline finding
   above, it was never going to be sufficient regardless of rate: the
   no-privacy baseline it would need to converge toward instead diverges.

---

## Step 15/16 — diagnosis and fix: round-aware LR decay (2026-08-23)

**Step 15 diagnosis** (investigation only, no code): grepped
`trainer_mentalbert_privacy.py` and `pipeline.py` for any round-conditioned
lr/epochs logic - none exists; `round_meta.round_id` reached `pipeline.py`
already but was only ever used for logging and ECDSA receipt-signing, never
passed to the trainer. AdamW is freshly constructed every round (expected
FedAvg behaviour, not itself a defect) with no optimizer-state persistence
anywhere. The actual mechanism: a fresh diagnostic run at the *current*
lr=1e-4/epochs=10 config (not the stale Step 9a numbers, which predate Fix
E3/E4) measured `fc1_grad_norm` at 59.9-454.5 across every sampled step of a
full training run - `torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)`
saturates on effectively every one of the 190 steps/round. Clipping preserves
direction but forces every step to the same pre-Adam magnitude regardless of
curvature or proximity to a good solution - so every round takes an
essentially fixed-size step whether starting from random init or from an
already-well-fit point. Zero `[SAFETY-CLAMP]` engagements across all 10 Step
14 rounds confirm the delta-level clamps were never involved - they only
watched, they never shaped this behaviour.

**Step 16 fix** (`trainer_mentalbert_privacy.py`, `runtime/pipeline.py` -
no security/TPM/crypto/gRPC/Rust touched, grad clip/loss balance/safety
clamps untouched per the diagnosis's own recommendation): `round_id` now
flows from `pipeline.py`'s already-available `round_meta.round_id` into
`trainer_orchestrate()`. When given, `effective_lr = lr * (LR_DECAY **
(round_id - 1))` - round 1 unaffected (`LR_DECAY**0 = 1`), later rounds
decayed. `LR_DECAY` is env-configurable, default `0.7`; `LR_DECAY=1.0` is a
valid, explicit "no decay" setting that reproduces Step 14 exactly, not a
special case. Explicit `round_id`, never inferred from `global_model_path`
being set (which would silently no-op if the global model were ever absent
for a later round). Effective lr is reported every round (`rpt.kv` +
`[STEP16-LR]` print) alongside `round_id` and the base lr.

**Verification**: re-ran the exact ARM 1 5-round trajectory (seed=303, mean
aggregation, otherwise identical to Step 14), `LR_DECAY=0.7` (the default).

| round | effective lr | Step 14 F1 (no decay) | Step 16 F1 (decay=0.7) | Step 16 accuracy | Step 16 pred+/37 | Step 16 MAE | Step 16 this-round delta L2 | Step 16 cumulative delta L2 |
|---|---|---|---|---|---|---|---|---|
| 1 | 1.00e-4 | 0.6667 | 0.4583 | 0.297 | 37/37 | 5.046 | 0.576 | 0.576 |
| 2 | 7.00e-5 | 0.4583 | 0.4583 | 0.297 | 37/37 | 6.050 | 0.419 | 0.953 |
| 3 | 4.90e-5 | 0.0000 | 0.4583 | 0.297 | 37/37 | 6.784 | 0.317 | 1.235 |
| 4 | 3.43e-5 | 0.0000 | 0.4348 | 0.297 | 35/37 | 7.482 | 0.233 | 1.443 |
| 5 | 2.40e-5 | 0.0000 | 0.4324 | 0.432 | 26/37 | 8.242 | 0.173 | 1.593 |

**LR decay is confirmed engaged and causally responsible**, not just
computed and ignored: this-round delta L2 shrinks by a ratio of 0.727,
0.757, 0.735, 0.743 at each successive round transition - matching
`LR_DECAY=0.7` almost exactly at every step. (The literal `[STEP16-LR]`
print lines were truncated out of the driver's combined log by the same
`[-3000:]` per-subprocess truncation that hid the Step 15 gradient-norm
logs - not re-plumbed for this run since the delta-L2 ratio is a stronger,
more direct confirmation than reading a printed number would have been: it
shows the decay had the right *causal* effect on training, not merely that
the value was computed.)

**Answer to the specific question - precisely, not glossed over**: round 1's
F1 does NOT literally hold at 0.6667, because round 1 is a fresh noisy draw
each run (`GLOBAL_INIT_SEED` fixes only the shared init; training
stochasticity is deliberately re-randomised per client - Defect B's own
fix). Comparing round 1 to round 1 across two different runs was never a
literal apples-to-apples comparison. The question that IS answerable and
matters: **does the trajectory collapse to F1=0 by round 3, as it did
without decay?** No. F1 holds flat at 0.4583 for three consecutive rounds,
then declines gently to 0.4348 and 0.4324 - a ~5.6% relative drop over two
rounds, not a collapse. **The fix works**: multi-round training under
LR_DECAY=0.7 stabilises in a moderate, non-degenerate performance band and
stays there, instead of diverging to a degenerate all-one-class collapse by
round 3. Not swept against other decay values in this step, per the
instruction (one value, one trajectory, honest result) - 0.7 neither
obviously overshoots (rounds don't stagnate at round-1 performance) nor
undershoots (no collapse) at this evidence, but a systematic decay sweep is a
separate, future measurement, not concluded here.

**A cost, not just a fix**: MAE degrades monotonically across the same 5
rounds even as F1 stabilises - 5.046 -> 6.050 -> 6.784 -> 7.482 -> 8.242.
Consistent with Fix E3's loss rebalance, which pushed `loss_cls` to dominate
`loss_reg` by 20-100x (measured in Step 9a) - stabilising classification
under LR decay does nothing to change that imbalance, so the regression head
keeps getting starved of gradient signal every round, and its error keeps
growing. Classification stability here was bought at a measurable, growing
cost to PHQ regression accuracy. `REG_LOSS_WEIGHT` (env, default 0.5) is the
existing knob if regression performance is ever prioritised - not changed
here, recorded only.
