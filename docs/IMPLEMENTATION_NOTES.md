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

---

## Step 17 — ARM 2 with LR decay: DP utility does not recover, and the
## noise/signal ratio gets WORSE, not better (2026-08-23)

Re-ran the Step 14 ARM 2 trajectory (DP, noise_multiplier=1.0, mean
aggregation, seed=303) with `LR_DECAY=0.7` (Step 16's default), identical to
Step 14 in every other respect - no tuning, no code changes for this step.

| round | eff. lr | Step 14 F1 (no decay) | Step 17 F1 (decay) | pred+/37 | this-round delta L2 | cumulative delta L2 | per-round eps | cumulative eps (naive bound) |
|---|---|---|---|---|---|---|---|---|
| 1 | 1.00e-4 | 0 | 0 | 0/37 | 260.36 | 260.36 | 5.302585 | 5.302585 |
| 2 | 7.00e-5 | 0 | 0.4583 | 37/37 | 260.76 | 368.65 | 5.302585 | 10.605170 |
| 3 | 4.90e-5 | 0 | 0 | 0/37 | 260.58 | 451.45 | 5.302585 | 15.907755 |
| 4 | 3.43e-5 | 0 | 0 | 0/37 | 259.72 | 520.89 | 5.302585 | 21.210340 |
| 5 | 2.40e-5 | 0 | 0 | 0/37 | 259.99 | 582.32 | 5.302585 | 26.512925 |

Every round: `prob_positive_distribution.stdev = 0.0`, min=max (exactly 0.0
in 4 rounds, exactly 1.0 in round 2) - full saturation every round, never
genuine discrimination. Round 2's F1=0.4583 is the SAME "predict everyone
positive" collapse mode seen throughout this investigation whenever DP noise
dominates - not recovered utility. Which of the two degenerate modes
(all-positive vs all-negative) the saturated noise lands on a given round is
effectively arbitrary.

**Headline finding - a structural tension, not merely an explanation for why
utility didn't recover.** The fix that stabilises multi-round training
(Step 16's LR decay - shrink the step size every round so the model stops
overwriting a good solution) and what DP-SGD viability requires under this
project's round-decayed FedAvg (signal growing, or at least holding steady,
relative to a fixed per-round noise floor) **point in opposite directions.**
LR decay is not incidental to this tension - it is the mechanism that
creates it: decaying lr is precisely shrinking the local training movement
each round, and DP noise is added downstream of training at a scale
(`noise_multiplier x clip_norm`) that has no dependence on lr, training
movement, or round number whatsoever. Any fix for training-side divergence
that works by damping local movement over rounds - decay is the most
standard, but the same argument applies to shorter per-round training,
smaller batch counts, or anything else that shrinks the per-round delta -
will, by construction, widen the noise-to-signal gap a fixed-scale DP
mechanism has to overcome, at exactly the low client counts (n=3, mean
aggregation's only real denoising lever) this project runs at. This is a
structural property of combining round-decayed FedAvg convergence fixes with
a fixed-scale per-round DP mechanism at low client counts, not an artifact
of this session's specific numbers - it would recur with any other decay
schedule, and is worth stating as its own, independent finding.

**The expected interaction, quantified**: this-round delta L2 stays flat at
~260 every round (260.36, 260.76, 260.58, 259.72, 259.99) - DP noise scale
(`noise_multiplier x clip_norm = 0.85`) is entirely independent of the raw
signal's magnitude (the pre-noise delta, ~0.6, never approaches the
clip_norm=0.85 threshold regardless of lr, so clipping stays inert and lr
has zero effect on the noised output's scale). Meanwhile ARM 1 with the same
decay shrank its this-round delta from 0.576 to 0.173 (Step 15/16 section).
The per-round noise-to-signal ratio, using ARM 1-with-decay's this-round
delta as the signal proxy: 452.0x (round 1) -> 622.3x -> 822.1x -> 1114.7x
-> 1503.4x (round 5) - **the ratio gets WORSE, not better**, the opposite
direction from Step 14's no-decay trend (447.7x -> 217.9x, improving via
sqrt(r) noise accumulation vs linear signal accumulation - see Step 14's
section). Cumulative delta L2 still matches `sqrt(r) x 260.36` to within
0.3% at every round, confirming the random-walk noise-accumulation mechanism
is unaffected by LR decay, exactly as expected since decay only touches the
signal component.

**Answer**: with the multi-round divergence fixed (Step 16), DP utility does
NOT recover across 5 rounds at noise_multiplier=1.0. Worse: the mechanism
that might have helped it recover (noise/signal ratio improving with
rounds, established in Step 14) is actively defeated by the same fix that
stabilised the no-privacy baseline - LR decay shrinks signal every round
while the DP noise floor stays fixed, so the ratio moves in the wrong
direction. This is a null result. Nothing was tuned to change it, per
instruction.

---

## Step 18 — FedAdam / FedYogi: the noise-damping hypothesis is confirmed,
## utility still doesn't recover, for a different and more specific reason
## (2026-08-23)

**Not a checkbox exercise - the motivating question, stated precisely before
running anything**: Step 14/17 measured DP noise accumulating as `sqrt(r)`
at a STABLE ~260 L2 magnitude per round. FedAdam/FedYogi's `v` (running
second-moment estimate) is exactly a running estimate of that per-coordinate
magnitude - if the noise really is stable round to round, `v` should
converge to reflect it, and `m_hat/(sqrt(v_hat)+tau)` would divide it back
down before `eta_s` rescales it, a normalisation plain averaging has no
mechanism for. Whether that damping is enough to matter was genuinely
unknown going in.

**Implementation** (`scripts/run_step14_multiround.py`, `--update-rule
{plain,fedadam,fedyogi}` - server-side only, no Rust/protobuf/client changes,
same offline-aggregation-plus-GridFS-patch pattern as Steps 13/14/17):
`m`/`v` tracked as additional in-process dicts across the driver's existing
round loop, exactly like `cumulative_delta` already was - no new MongoDB
collection. Hyperparameters from CLAUDE.md, env-overridable: FedAdam
`eta_s=1e-3, beta1=0.9, beta2=0.999`; FedYogi `eta_s=1e-2, tau=1e-3`.
CLAUDE.md does not state FedAdam's `tau` or FedYogi's `beta1`/`beta2` - each
borrows the other algorithm's stated value (Reddi et al. 2020 conventionally
shares one `tau` and one `(beta1,beta2)` pair across both variants), recorded
as a deliberate, traceable choice, not a silent gap. Verified against
synthetic stable-magnitude noise before spending any wall-clock time on the
live run - the update rules behaved exactly as intended (applied step
shrinking monotonically as `v` ramped up).

**Results** - both arms: DP, noise_multiplier=1.0, mean aggregation, seed=303,
LR_DECAY=0.7 (default), 5 rounds, compared against Step 17's plain-averaging
DP baseline (identical in every other respect):

| round | plain this-round Δ L2 | plain F1 | FedAdam this-round Δ L2 | FedAdam F1 | FedYogi this-round Δ L2 | FedYogi F1 |
|---|---|---|---|---|---|---|
| 1 | 260.36 | 0 | 1.438 | 0 | 14.386 | 0 |
| 2 | 260.76 | 0.4583 | 1.484 | 0 | 14.829 | 0 |
| 3 | 260.58 | 0 | 1.447 | 0 | 14.461 | 0 |
| 4 | 259.72 | 0 | 1.397 | 0 | 13.941 | 0 |
| 5 | 259.99 | 0 | 1.344 | 0 | 13.412 | 0 |

Raw pseudo-gradient L2 (before the FedAdam/FedYogi transform) matches the
plain baseline almost exactly every round (~260, e.g. FedAdam round 1:
259.87, FedYogi round 1: 260.44) - confirming the aggregation step itself is
unchanged; only what happens to that aggregated delta afterward differs.

**The damping hypothesis is confirmed, dramatically**: FedAdam's applied
step is ~176-194x smaller than plain averaging's, every round (260.36 ->
1.438 at round 1, 259.99 -> 1.344 at round 5). FedYogi's applied step is
~18-19x smaller (its 10x larger `eta_s` accounts for almost exactly the
~10x gap between the two - 181/18 ~ 10, matching `eta_s` ratio 1e-3 vs
1e-2). `v_norm` grows steadily and smoothly across both (0.221 -> 0.754,
FedAdam; 0.222 -> 0.756, FedYogi - nearly identical, since `v`'s update
depends on the raw pseudo-gradient, not `eta_s`), confirming the server-side
optimizer is doing exactly what the hypothesis predicted: learning the noise
floor's magnitude and dividing it back down.

**Utility does not recover - F1 stays at 0 in 9 of 10 arm-rounds measured**
(FedAdam all 5, FedYogi all 5). This is NOT the same finding as Step 17's
"noise/signal ratio gets worse under decay" - here the ratio gets
*dramatically better*: using Step 16's ARM-1-with-decay trajectory
(0.576/0.419/0.317/0.233/0.173) as the signal reference, FedAdam's
per-round noise/signal ratio is 2.50x -> 7.77x across the 5 rounds - two
orders of magnitude better than plain averaging's 452x -> 1503x (Step 17).
FedYogi's ratio, 24.98x -> 77.53x, is roughly an order of magnitude better
than plain, though ~10x worse than FedAdam's (same `eta_s` relationship).
**And it still doesn't work.**

**Why, precisely - a different, more specific failure mode than plain
averaging's random per-round saturation.** Plain averaging's DP arm
(Step 17) saturates to exactly 0.0 or exactly 1.0 essentially at random each
round - whichever direction that round's raw noise happened to dominate in.
FedAdam/FedYogi's probability distributions show something qualitatively
different: `P(positive)` shrinks *monotonically and smoothly* round over
round (FedAdam: mean 1.8e-5 at round 1 -> 8.8e-10 at round 5) - increasing
confidence in the SAME wrong direction every round, not a fresh coin flip
each time. `m` (the momentum term) is the reason: with only 5 rounds and
`beta1=0.9`, `m` is close to a simple average of the 5 rounds' raw noise
draws, which for a finite sample in 281,254 dimensions is never exactly
zero - it has some residual direction. Momentum then applies that SAME
residual direction, consistently, every round, compounding into an
increasingly confident wrong prediction, rather than plain averaging's
independent-per-round coin flip.

**Named finding 1 - the two halves of the optimizer worked against each
other.** `v` (second moment) did exactly its job: it killed the noise
MAGNITUDE, ~180x, confirming the hypothesis this step set out to test. `m`
(first moment) did the opposite of what was hoped: it amplified the noise
DIRECTION. The residual direction of a finite (5-round) noise sample in
281,254 dimensions is never exactly zero - some component of it is,
unavoidably, a specific, non-random direction, purely by sampling luck.
Momentum (`beta1=0.9`) then applies that SAME residual direction,
consistently, every round, compounding it into an increasingly confident
wrong prediction. Plain averaging's independent-per-round coin flip -
each round's saturation direction essentially uncorrelated with the last -
was, paradoxically, LESS harmful than momentum-concentrated drift: a fresh
coin flip each round at least has a chance of cancelling out over time; a
consistently-reinforced residual direction never does. The algorithm built
specifically to smooth DP noise across rounds instead gave that noise's
one non-random component somewhere to accumulate.

**Named finding 2 - the 2.5x floor, and why it gets worse, not better, over
5 rounds.** Even after ~180x damping, FedAdam's applied step (~1.3-1.5 L2)
never drops below 2.5x the genuine no-privacy signal it is competing
against (Step 16's LR-decayed ARM 1 trajectory, 0.576 -> 0.173). That ratio
does not hold steady either - it DEGRADES across the 5 rounds, 2.50x ->
3.54x -> 4.56x -> 6.00x -> 7.77x. The reason is the same tension Step 17
already found, now visible even inside the damped regime: LR decay
(Step 16) shrinks the genuine signal every round (0.576 -> 0.173, a
~3.3x reduction over 5 rounds) while FedAdam's damped-noise floor stays
comparatively flat (1.438 -> 1.344, essentially unchanged) - so the ratio
between them necessarily widens as decay proceeds, exactly mirroring
Step 17's plain-averaging finding that decay widens the noise/signal gap,
just starting from a ~180x-better floor. Whether a ratio near or below 1.0
would be the threshold where utility actually starts to recover is an
OPEN QUESTION this measurement does not answer - FedAdam never got closer
than 2.5x in 5 rounds, and that number was moving in the wrong direction
by round 5, not the right one. Stated as an open question, not a claim.

**This compounds with, rather than being separate from, Step 16's own
finding.** Step 16 already established that this training regime is fragile
even at ZERO noise - the no-privacy baseline itself only stabilises at a
modest F1~0.43-0.46 under decay, and diverges to F1=0 without it. Given
that fragility, ANY consistent, momentum-compounded directional
perturbation - even one reduced 180x from its raw magnitude - lands on
already-thin ice. The bottleneck this measurement reveals has shifted: not
"raw DP noise overwhelms everything" (plain averaging's failure mode,
Steps 12-17), but "any non-trivial, momentum-concentrated drift overwhelms
an inherently fragile few-hundred-record training regime" (FedAdam/FedYogi's
failure mode, Step 18). Different mechanism, same practical outcome.

**Answer to the specific question**: the adaptive second-moment term DOES
damp the stable DP noise component, by two orders of magnitude for FedAdam
and one for FedYogi, exactly as hypothesised from Step 14/17's own findings
- this is a genuine, positive, measured result, not a null one, on the
narrow damping question. But it does not translate into recovered utility
at this project's data scale and round count, for an identifiable and
different reason than plain averaging's failure: momentum concentrates
whatever small residual survives damping into a compounding, confidence-
collapsing directional drift, on top of a training regime Step 16 already
showed is fragile even without any noise at all. Both results - the damping
and the non-recovery - are reported honestly, as instructed; neither was
tuned to look better or worse than measured.

---

## Step 18c — FL algorithm claim audit: what the document claims vs. what
## exists (2026-08-23)

The project document's contribution #2 and Table 13 claim "the first
systematic comparison of five FL optimization algorithms - FedAvg, FedProx,
FedAdam, FedYogi, and SCAFFOLD - with four aggregation strategies." A
full-repository search (code + docs + every frozen/historical experiment
folder - `exp_c2_multiclient/`, `phase22_end_to_end/`,
`create_dp_comparison.py`, and all others) found:

- **FedProx, SCAFFOLD**: not implemented anywhere, live or historical - zero
  matches for the algorithms or any plausible code-level proxy (`proximal`,
  `control_variate`, `c_k`, `c_local`/`c_global`, `delta_c`) in any `.py` or
  `.rs` file. Only `CLAUDE.md`, `FL_ALGORITHM_COMPARISON.md`, and `README.md`
  contain the strings at all - all documentation, zero code.
- **FedAdam, FedYogi**: were not implemented before this session. **Now
  implemented** (Step 18a/b above) - server-side only, in
  `scripts/run_step14_multiround.py`, using the offline-aggregation-plus-
  GridFS-patch pattern, never touching the live Rust orchestrator or
  `aggregator.py`'s production path.
- **FedAvg**: only unweighted mean exists (`AggregatorAgent._aggregate_tensor()`,
  `mode="mean"`, `server/aggregator_agent/aggregator.py:583-584`) - not the
  textbook (McMahan et al. 2017) sample-count-weighted version. No `n_k`
  (per-client sample count) is transmitted or referenced anywhere in this
  codebase. This was already independently documented in this repo's own
  pre-existing `FL_ALGORITHM_COMPARISON.md:86-90` before this session's audit.
- **Krum**, one of the document's claimed four aggregation strategies
  (alongside mean/trimmed_mean/median): also not implemented -
  `AggregatorAgent._aggregate_tensor()` has exactly three modes, no fourth.

**Bottom line on the document's claim**: of five claimed FL optimization
algorithms, three (FedProx, SCAFFOLD, and FedAvg's weighting) remain
unimplemented after this session; two (FedAdam, FedYogi) are now real,
working, measured implementations (Step 18a/b, above) - the first genuine
content behind that claim to exist in this codebase. Of four claimed
aggregation strategies, three (mean, trimmed_mean, median) are implemented;
Krum is not.

**Not implemented, with reasons - difficulty was not the blocker for any of
these:**

- **FedProx** - implementable (~40-60 lines across `trainer_mentalbert_privacy.py`
  and `pipeline.py`, no Rust, following the exact `round_id`-threading pattern
  Step 16 already used) but its benefit cannot be validated in this project as
  currently constituted: every "client" trains on the identical 149-record
  shared corpus (documented since Fix E2), so there is no genuine non-IID
  client divergence for the proximal term to correct. Building it would
  produce a mechanism that runs but demonstrates nothing about what FedProx is
  for. Not implemented for that reason, not for difficulty.
- **Krum** - needs `K>=5` (`f=floor(K/5)`); at this project's `n=3`
  ("clients"), `f=0` - zero Byzantine tolerance, a degenerate edge case, not a
  meaningful demonstration. Even at `K>=5`, there is no genuine adversarial or
  malfunctioning client to filter - all submissions come from the one honest
  enrolled device (the single-device-identity disclosure already in
  `MENTOR_DEMO_RUNBOOK.md`) training on identical data. Krum's entire purpose
  is adversary detection; this architecture has no adversary for it to detect.
- **SCAFFOLD** - requires persistent, genuine per-client identity (for `c_k`)
  and genuine per-client data heterogeneity (for the control variates to have
  anything to correct), neither of which this single-enrolled-device
  architecture provides (three or five "clients" per round are one device
  submitting sequentially, all training on the same corpus). Building it with
  synthetic per-session client IDs would manufacture a demonstration whose
  honesty is worse than simply not claiming it - consistent with how Defect
  A/B and the `server.rs:1499` trimmed_mean-to-mean change are already logged
  as known-and-not-started rather than worked around with something that only
  looks correct.

---

## Step 20 — genuine per-client data partitioning (2026-08-23)

**The problem this closes**: every measurement before this step had all
"clients" training on the identical 149-record split (documented since Fix
E2) - not federation, one dataset submitted three times. Step 20 adds a real
N-way stratified (IID) or Dirichlet-skewed (non-IID) partition of the TRAIN
records only - the held-out 37 stays global and untouched throughout.

**Where client identity comes from**: nothing distinguished client 1/2/3
before this step - `device_id` is the same enrolled device every submission,
`session_id` a fresh random UUID, `round_meta.round_id` identical across all
3. `RoundMetadata.num_updates` (proto field 6) could in principle self-assign
a shard but is race-prone under concurrent launches; a `CLIENT_SHARD_ID` env
var (matching the `GLOBAL_INIT_SEED` precedent) is race-free and was used
instead.

**Implementation** (`trainer_mentalbert_privacy.py` - no Rust/security
touched): `stratified_shard(train_records, n_shards, shard_id, seed,
noniid_alpha=None)`, placed next to `stratified_split()` (never modified -
every prior measurement depends on its 2-way contract staying exactly as
today). IID mode divides each class's shuffled indices into near-equal
contiguous pieces. Non-IID mode draws each class's per-shard share from
Dirichlet(alpha) (stdlib `random.gammavariate`, no numpy dependency) - low
alpha concentrates a class into fewer shards. Every client computes the
identical full partition from the same `(n_shards, seed, alpha)` and slices
out its own `shard_id` - no cross-client coordination needed, shards are
guaranteed disjoint and exhaustive. If a draw leaves any shard single-class
or empty, the WHOLE partition is redrawn (`seed+1, seed+2, ...`, logged) up
to 100 attempts before raising. `CLIENT_SHARD_ID`/`CLIENT_N_SHARDS` both
unset by default - `orchestrate()` never calls `stratified_shard()` at all in
that case, verified two ways: (1) code inspection - the gate requires both
non-`None`, and the `else` branch performs zero reassignment of
`train_records`; (2) a fresh offline call to `stratified_split()` reproduced
the exact train=149 (44 pos/105 neg) reported identically in every run since
Step 12. Offline unit tests (both modes) confirmed shards disjoint and
exhaustive (union == the 149 train records) before any live run.

**One mistake caught before it propagated**: the first live Run A used
`DP_MECHANISM=gaussian` (the driver's naive default) - noise swamps any
partitioning signal regardless of how data is split (F1=0, aggregated delta
L2=259.9, indistinguishable from every other DP-enabled run in this project)
and cannot answer what this step asked, since the comparison target (Step
14 ARM 1's 0.6667) is the NO-PRIVACY arm. Reset and reran both arms with
`DP_MECHANISM=none`, isolating partitioning as the only changed variable.

**Run A - IID, 3 shards** (seed=303, shard_seed=20240, no privacy):

| client | shard size | pos/neg | achieved ratio | delta L2 |
|---|---|---|---|---|
| 0 | 50 | 15/35 | 0.3000 | 0.3319 |
| 1 | 50 | 15/35 | 0.3000 | 0.3692 |
| 2 | 49 | 14/35 | 0.2857 | 0.3421 |

Aggregated delta L2 = 0.2415, F1 = 0.0000, accuracy = 0.6757, predicted+ =
1/37, MAE = 4.807, eps = inf (arm=none, the real value, not the pipeline's
1.0 receipt fallback).

**Run B - non-IID, alpha=0.5, same 3 shards, same seed** (no privacy):

| client | shard size | pos/neg | achieved ratio | delta L2 |
|---|---|---|---|---|
| 0 | 73 | 11/62 | 0.1507 | 0.3740 |
| 1 | 40 | 1/39 | 0.0250 | 0.2648 |
| 2 | 36 | 32/4 | 0.8889 | 0.2752 |

Genuinely skewed - client 1 near all-negative (2.5% positive), client 2 near
all-positive (89%). Resample-on-collapse fired once (logged) avoiding a
single-class draw. Aggregated delta L2 = 0.1999, F1 = 0.0000, accuracy =
0.7027, predicted+ = 0/37, MAE = 6.427, eps = inf.

**Findings:**

1. **Per-client deltas now genuinely differ, and non-IID differs more than
   IID.** Run A's spread (0.332-0.369, range 0.037, ~11% of mean) vs. Run
   B's (0.265-0.374, range 0.109, ~36% of mean) - roughly 3x the relative
   spread at the identical seed and client count. A clean signature that
   genuinely different data is now driving genuinely different client
   updates, not merely training stochasticity on identical data (which is
   what produced ALL prior per-client variation before this step).

2. **Every delta in both runs falls outside the Fix E4 calibration range
   [0.625, 0.729]** - measured 0.265-0.374, roughly half the calibrated
   range. `calibrate_clip_norm.py` calibrated `clip_norm=0.85` against
   149-record training; ~50-record shards produce systematically smaller
   deltas. Flagged as a finding, per instruction - NOT retuned in this step.
   A separate recalibration (or an epoch-count sweep for the sharded
   regime, since ~50 records may not want the same `epochs=10` tuned for
   149) is a future decision, not made here.

3. **F1 dropped from 0.6667 (identical-partition baseline, Step 14 ARM 1
   round 1) to 0.0000 in both runs** - exactly the outcome anticipated
   before running anything. A lower, genuinely-federated number, reported
   honestly; nothing tuned to compensate. `SUPERVISED_EPOCHS` was left at
   its default 10 throughout, per instruction - not tuned to chase a better
   number on ~50-record shards.

4. **Held-out 37 confirmed identical** across both runs and every prior
   measurement since Step 12 - `stratified_shard()` only ever receives
   `train_records`, never `eval_records`.

Both runs completed through TPM signing, encryption, and upload with zero
errors. Clipping was not re-confirmed from a saved log line for these
specific runs (driver display truncation) but is inferred with high
confidence to be inert - every delta measured is well under
`clip_norm=0.85`, consistent with every prior measurement at this scale.

---

## Step 21 — epoch sweep for the sharded regime: a data-volume floor, not
## undertraining (2026-08-23)

Step 20's Run A (IID, 3 shards) collapsed to F1=0.0 at ~50 records/client,
epochs=10 (tuned for 149 records, Step 10a). Two plausible causes: (a) ~50
records with ~15 positives is too little data regardless of training length,
or (b) epochs=10 is undertrained at 50 records (~7 steps/epoch vs ~19 at
149). `scripts/sweep_epochs_sharded.py` (offline, no gRPC/Mongo/DP/upload)
swept epochs in {10,20,30,50} at lr=1e-4 (unchanged), 3 seeds/cell, training
on shard 0/3 (seed=20240) - confirmed to reproduce Run A client 0's exact
shard (n=50, 15 pos/35 neg) before trusting any result - evaluated on the
same global held-out 37. Completed in 14.2 minutes.

| epochs | degenerate | mean F1 | per-seed F1 | mean delta L2 |
|---|---|---|---|---|
| 10 | 3/3 | 0.3056 | 0.0000, 0.4583, 0.4583 | 0.328 |
| 20 | 2/3 | 0.3568 | 0.4583, 0.4583, 0.1538 | 0.581 |
| 30 | 1/3 | 0.2917 | 0.3750, 0.0000, 0.5000 | 0.842 |
| 50 | 1/3 | 0.3654 | 0.0000, 0.5405, 0.5556 | 1.367 |

**Full-149 baseline, epochs=10** (single-client context, not the aggregated
Step 14 number): degenerate=0/3, mean F1=0.5729 (matches Fix E4's historical
value exactly - the sweep script reproduces a known-correct number, not a
new one), delta L2 0.623/0.719/0.697 - inside the Fix E4 range.

**Findings:**

1. **No epoch count recovers full non-degenerate F1 on the shard.**
   Degeneracy drops 3/3 -> 2/3 -> 1/3 -> 1/3 as epochs increase, but never
   reaches 0/3 in the tested range. Best mean F1 (epochs=50, 0.3654) stays
   far below the full-149 baseline (0.5729) - not converging toward it.

2. **Delta L2 passes through the Fix E4 range [0.625, 0.729] but doesn't
   stabilise there, and matching it doesn't mean better F1.** Grows
   monotonically with epochs (0.328 -> 0.581 -> 0.842 -> 1.367), crossing
   the range around epochs~25-27. epochs=30 (delta L2~0.84, past the range)
   has the WORST mean F1 (0.2917) of all four cells - worse than epochs=20,
   which is still below the range. Delta magnitude recovering the
   "expected" scale is not a proxy for a more reliable model here.

3. **Read: predominantly a data-volume floor (cause a), not primarily
   undertraining (cause b).** If training length were the main bottleneck,
   F1 should climb toward 0.57 as epochs increase; instead it plateaus and
   wobbles non-monotonically between 0.29 and 0.37 for epochs 20-50 (a real
   dip at 30), with high seed-to-seed variance at every cell (epochs=30:
   0.375/0.000/0.500 - the SAME epoch count landing anywhere from collapse
   to moderate performance depending purely on init/dropout draw). That's
   the signature of too little signal (15 positives) for a stable decision
   boundary, not insufficient optimization - more epochs on too little data
   mostly lets the model overfit in a different, seed-dependent direction
   each time rather than converge. Training length is not irrelevant either
   - degeneracy frequency does measurably drop (3/3 -> 1/3) - so this is not
   a *pure* data floor with zero training-length interaction, but data
   volume is the dominant factor, and no epoch count tested closes the gap
   to the full-149 baseline.

No default changed, no epoch count chosen, no live 3-client round re-run,
per instruction.

---

## Step 20/21 conclusion: what genuine federation costs at this corpus size
## (2026-08-23)

- **Before Step 20, all clients trained on the identical 149-record split.**
  That was not federation - one dataset submitted three times. Every
  aggregated number prior to Step 20 (Steps 12-19) should be read as "one
  client's result, averaged with two copies of itself," not as evidence
  about how this system behaves under genuinely divided data.

- **Step 20 makes genuine per-client partitioning available (IID and
  non-IID), verified working, not merely implemented.** Per-client delta
  spread: 0.037 (IID, range as a fraction of mean ~11%) vs 0.109 (non-IID,
  alpha=0.5, ~36% of mean) - roughly 3x wider under deliberate label skew at
  the identical seed and client count, confirming real data divergence is
  driving the difference, not training stochasticity (which is all that
  produced per-client variation before this step).

- **Step 21 measures the cost, and it's a real one.** At n=3, each client
  gets ~50 records with ~15 positives, and no epoch count in {10, 20, 30,
  50} recovers non-degenerate F1 - best sharded mean F1 0.3654 vs. 0.5729 on
  the full 149. The failure mode (non-monotonic F1, a real dip at
  epochs=30, per-seed spread of 0.375/0.000/0.500 on IDENTICAL data at that
  cell) is the signature of too little signal for a stable decision
  boundary, not insufficient optimisation.

- **Therefore: DAIC-WOZ at 186 participants does not support genuine 3-way
  federation at usable utility.** This is a corpus-size limit, not an
  implementation defect. The arithmetic, stated plainly: 55 positive
  participants in the whole corpus, 44 of them land in the 149-record train
  split (11 held out in the global eval 37), leaving ~15 positives per
  client once split three ways. No architecture or training-schedule choice
  changes that arithmetic.

- **Sharding stays default OFF**, so every measurement from Step 12 onward
  remains exactly reproducible without it. It's available via
  `CLIENT_SHARD_ID` / `CLIENT_N_SHARDS` (and `CLIENT_NONIID_ALPHA` for
  skew) whenever it's wanted. The non-IID mode is, as of this session, the
  *only* path to genuine client drift anywhere in this system - the
  precondition Step 18c already identified FedProx as needing to be
  meaningfully testable (its proximal term has nothing to correct without
  real non-IID divergence, which now exists but did not before Step 20).

- **n=2 is an untested intermediate**, noted for anyone who wants to
  revisit this: 74 records/client, ~22 positives each - roughly 1.5x this
  step's 50-record/15-positive shards. Not measured here; not claimed to
  behave any particular way. A natural next data point if genuine
  federation at usable utility is ever revisited for this corpus.

---

## Chapter 7 Table 15 silhouette values are fabricated, not measured
## (2026-08-23)

**The mechanism**: `create_dp_comparison.py:816` —
```python
silhouette = base_metrics.get("silhouette_score", 0.65 + np.random.randn() * 0.01)
```
A bare dict `.get()` with a random-number fallback expression as the default.
No warning is printed when this fires (unlike the accuracy/precision/recall/f1
fallback a few lines below, which at least logs `[WARN] Using fallback
metrics...`), and no marker is written into the output CSV distinguishing a
real value from this one - the column just contains a float indistinguishable
in format from a genuine measurement.

**Why it fired unconditionally, on every run, not just "sometimes"**: the
only function in the file that computes a real silhouette score is
`evaluate_unsupervised_X()` (`create_dp_comparison.py:226-247` - a real
`KMeans` + real `sklearn.metrics.silhouette_score` call). It has **zero call
sites** anywhere in the file - defined once, invoked nowhere. Neither of the
two paths that populate `base_metrics` (the real trainer orchestrator's
output, or this script's own local fallback trainer) ever writes a
`"silhouette_score"` key into the metrics dict that becomes `base_metrics` -
that key is structurally absent regardless of which trainer path ran, or
whether it succeeded. The fallback is not a rare edge case; it is the only
code path that has ever produced a silhouette value in this file's history.

**Statistical evidence, from surviving CSVs matching the referenced
filenames** (`dp_noise_mechanism_comparison_base.csv`,
`..._rag.csv`, `..._vector_rag.csv`, `dp_comparison_all_modes.csv`):
per-file silhouette means 0.6462-0.6515, stdev 0.0088-0.0114 - against the
fallback expression's own constructed distribution, mean 0.65 / std 0.01, by
construction. An essentially exact match, and consistent with Table 15's
cited 0.635-0.658 clustering across all six DP mechanisms.

**Structural evidence, independent of the statistical match**: within any
one of these files, silhouette varies row-to-row across all 30 rows, while
accuracy/precision/recall/f1/mae are frozen **identical** across every one of
those same 30 rows. `base_metrics` is loaded once, before the
mechanism/noise-multiplier loop, and re-read (not re-fetched) on every
iteration - a real dict lookup necessarily returns the same value every row,
exactly what accuracy/precision/recall/f1 show. The silhouette fallback
expression sits inside that same loop and is evaluated fresh on every
iteration, drawing a new `np.random.randn()` each time - exactly what the
varying silhouette values show. Two structurally different code paths,
visible directly in which columns move and which don't, in the surviving
data itself - not inferred from the source alone.

**What is NOT fabricated, stated precisely so this isn't overclaimed**:
accuracy/precision/recall/f1/mae in these same files are real `metrics.json`
values, not fallback output - the fallback's own construction (`mae =
round(np.random.uniform(0.1, 0.4), 3)`) is bounded to [0.100, 0.400] and
rounded to 3 decimals; the surviving mae values (1.320428729057312,
1.6080342531204224, 2.1529345512390137) are full float precision and 3-5x
outside that range, proof they came from a real file read. What they show is
a genuinely collapsed, single-class model (accuracy=1.0, precision=recall=
f1=0.0 on every row) - the already-documented single-class-collapse defect
elsewhere in this project's history, not a new fabrication finding.

**The limit of this claim, stated plainly**: no documentary chain of custody
was established from these surviving CSVs to the specific numbers published
in Table 15. What was established: these are the sole surviving artifacts in
the repository matching the filenames Chapter 7's numbers are named after;
their statistical signature matches the fallback expression's construction
almost exactly; and their range is consistent with the table's cited
0.635-0.658. That is strong, direct evidence the table's silhouette column
traces to this fallback - not a proven, unbroken chain to the exact published
cells.

**`dp_noise_mechanism_comparison_full.csv` is excluded from this finding, as
a separate, untraced artifact**: a different, older 10-column schema (no
accuracy/precision/recall/f1/mae/mode columns at all), Linux paths
(`/home/ritik26/Desktop/BE-Major-Project/...`, October 2025 timestamps)
versus the other three files' Windows paths (`C:\Users\DELL\Desktop\pipeline
backup\BE-Major-Project\...`, June 2026 timestamps) - a different machine, a
different code snapshot, a different run. Its silhouette distribution (mean
0.573, stdev 0.147, range 0.30-0.78) matches neither the fallback's
construction nor Table 15's cited range. Not traced further; not folded into
this finding.

**Consequence**: Chapter 7's Table 15, and any analysis built on it
(Figures 19/20/23/24 - "Gaussian mechanism superiority," "Laplace
instability," and similar silhouette-driven claims) rest on random numbers
drawn from a fixed distribution, not measurements of anything about the
data, the DP mechanism, or the noise level. Any silhouette-based claim in the
report must be withdrawn or re-measured against a real evaluation path -
`evaluate_unsupervised_X()` already exists and is correct, it simply has
never been called. `create_dp_comparison.py` was not modified, nothing was
re-run, and re-measurement was not attempted here - fixing the fallback and
re-establishing what the real numbers are is a separate, deliberate decision,
not made in this investigation.

---

## Phase A: what the attribution mechanism can and cannot support (2026-08-26)

Phase A built an Integrated-Gradients (IG) modality/token attribution
mechanism over the local, pre-DP model (`xai_integrated_gradients()`,
`installer/runtime/agents/trainer/trainer_mentalbert_privacy.py`), intended
to explain which modality - text, audio, video - drove a prediction, for use
by a later explanation layer (Phase B). Getting from "it runs and produces
numbers" to "the numbers mean what they appear to mean" took five steps of
investigation (A3-A7), documented here so a later session does not have to
re-derive them, and does not repeat an already-refuted hypothesis.

**Attribution, added after a later session-mixing error was caught (2026-09-01):**
every specific count and score in Steps A3-A7 below (37/37, 11/20/6, 36 of 37,
etc.) was measured on ONE client round's trained checkpoint, session_id
`client-aef1978aef7f` (local F1=0.5294 - the figure Step A3 cites below). These
are properties of that one trained model, not fixed properties of the system -
live client rounds do not set a fixed init seed, so a different round trains a
genuinely different model. Confirmed directly: a separate session,
`client-405c6057ab84`, trained independently, measures **37 of 37** in the
uncertain 0.3-0.7 band (not 36 of 37) on the identical held-out split. Do not
carry these specific counts forward as if they describe "the model" in
general - they describe this one checkpoint, and were previously stated
without that attribution in the summary sections below (now corrected) and in
an out-of-repo handover document (not corrected here, out of scope).

**Step A3's original result, and why it was wrong.** The first live run
(one client, round 1, local model F1=0.5294) produced an aggregate modality
split of text=0.131 / audio=**0.791** / vision=0.078, with audio ranked #1
in **37/37** held-out samples, zero exceptions. Read at face value, this
looks like a strong, unanimous clinical finding: audio dominates. It is not
one. Step A4 traced it to the IG baseline: audio_vec's real per-sample L2
norm (mean 6381.6) is 16.6x text's (383.6) and 11.1x vision's (574.1) - the
zero-vector baseline used for audio/vision is therefore a vastly more
extreme, out-of-distribution "removed" state for audio than for the other
two modalities, which inflates IG's raw attribution sum for audio regardless
of genuine importance. This is a scale artefact, not a signal.

**Step A4's correction of the investigation's own working assumption.** The
original plan was to cross-check IG against `modality_ablation_importance()`
(zero-out-a-modality, measure prediction shift) on the theory that ablation
is "naturally scale-aware in a way raw IG is not." It agreed with raw IG
(audio dominant, 37/37, ablation share 0.877) - but this was **not**
independent confirmation. Ablation zeroes a modality exactly the same way
IG's baseline does; zeroing audio is a proportionally far larger, more
out-of-distribution perturbation than zeroing text or vision for the
identical reason IG's baseline is biased. The two methods agreeing was two
instances of the same shared vulnerability, not two independent methods
converging on the truth. Catching this before treating "ablation agrees" as
proof was the most consequential correction in this phase.

**Step A5's fix.** `compute_modality_baselines()` computes in-distribution
mean vectors over the TRAIN split only (never eval - using eval data to
build a baseline would leak), cached to
`~/.federated/data/xai_baselines/modality_baselines.pt`, and wires them into
both `xai_integrated_gradients()` and (opt-in only, `modality_ablation_importance()`'s
existing unconditional `"autonomous"`-mode call site is deliberately left on
its original zero/empty-string behaviour to avoid an uninstructed change to
already-shipped output) `modality_ablation_importance()`. Text's baseline
was also switched, from a single PAD-token embedding to a mean-embedding
(average of real, attended-token embeddings over train) - PAD is
in-distribution but is one specific token's embedding, not a distributional
average, which was inconsistent with the audio/vision fix's own logic.

**Step A6's outcome: the artefact is fixed, the methods still don't
converge.** (All figures in this paragraph: session `client-aef1978aef7f`
only - see the attribution note at the top of this section.) Under the new
baselines, the unanimous audio-dominance pattern is gone from both methods -
IG: text=0.844/audio=0.087/vision=0.070 (37/37 text); ablation:
text=0.515/audio=0.416/vision=0.068, but only **11/20/6** per-sample (audio
actually wins the plurality of *individual* samples while losing on the
mean). 11/20/6 across three categories is close to a random three-way split
(~12.3 each). An earlier draft of this step's own verdict
logic called this "CONVERGE" because it only compared aggregate top-1 picks;
that was an overclaim, corrected before being reported - matching on the
mean while one method is unanimous and the other is barely-better-than-noise
per sample is not convergence at the level Phase B would need.

**Step A7: the leading hypothesis for the divergence is refuted, and the
likely real cause is the model, not the attribution methods.** The working
theory was that ablation is a single discrete jump from baseline to input
while IG integrates over a path of `n_steps` points, and that predictions
sitting near the 0.5 decision boundary would make a discrete jump more
exposed to local nonlinearity. Tested directly: IG's per-sample dominance is
**37/37 text at every n_steps from 5 to 50** (aggregate text score drifts
from 0.946 at n_steps=5 down to 0.836 at n_steps=50, but the ranking never
moves). If path-integration smoothness were what produced IG's unanimity,
the coarsest setting (n_steps=5) should have looked unstable, closer to
ablation. It didn't - the hypothesis is refuted, not confirmed. Per-sample
Spearman correlation between IG and ablation across the 37 samples: mean
ρ=0.189 (std=0.661), top-1 agreement 11/37=29.7% - at or below chance for
three categories. The more important finding came from splitting by
prediction confidence: **36 of the 37 held-out samples have a predicted
positive probability in [0.377, 0.709]** - essentially the entire held-out
set sits in the uncertain 0.3-0.7 band; only one sample is even marginally
outside it (0.709). (Again, `client-aef1978aef7f` only - `client-405c6057ab84`
measures 37 of 37 on the same split, not 36 of 37; see the top-of-section
attribution note. This count is not a fixed system property.) The "confident
vs uncertain" comparison this step set
out to run was not testable, because there is almost no confident subset to
compare against. Decision, made explicitly rather than by continuing to
chase the divergence: this is a property of the local model's weakness
(F1≈0.53, precision 0.39, recall 0.82 - a model that rarely commits) not a
defect in either attribution method, and no further attribution experiment
on this checkpoint will resolve it. Further investigation was stopped here
by deliberate decision, not because the question was answered.

**Plain conclusion.** This mechanism supports **aggregate, cohort-level**
modality attribution, with the method caveat stated above (IG and ablation
do not agree on individual patients, and the model's near-universal
prediction uncertainty is the likely reason). It does **not** support
confident **per-patient** modality ranking. No baseline choice, no n_steps
setting, and no confidence-based subset tested in Steps A3-A7 changes that.

### Constraints Phase B must respect

Written down before any narrative-generation code exists, so Phase B is
scoped to what Steps A3-A7 actually established:

- The narrative **must not** quote per-patient modality percentages (e.g.
  "audio contributed 79% to this patient's prediction") as if they were a
  reliable, individual measurement - Step A6/A7 showed per-patient IG and
  ablation rankings do not agree with each other.
- It **must not** claim a specific modality "drove" an individual
  prediction - the same evidence applies.
- It **must** surface the model's own uncertainty: on session
  `client-aef1978aef7f`, 36 of 37 held-out predictions sit in the 0.3-0.7
  band (a different session, `client-405c6057ab84`, measures 37 of 37 on the
  identical split - this count is checkpoint-specific, not fixed). Either
  way, any narrative that presents a prediction as confident without saying
  so would misrepresent the underlying model.
- Aggregate, cohort-level statements about modality contribution (e.g. "text
  was the most heavily weighted modality across this client's held-out set")
  **are** supportable by Steps A5-A6's results and may be used.

---

## Step B6 — thousands-separator mangling: investigated, not suppressed (2026-08-27)

Phase D's consistency checker (`scripts/ollama_narration_utils.py`) flagged
`150.0` and `6992.0` as untraceable across repeated runs. Narrative text:
*"...consisted of **150,6992** bytes of data..."* — the real value is
`1506992` (the encrypted payload size, `model_updates[0]["size_bytes"]`).
phi3:mini is writing the digits in the right order but placing the comma
wrong (grouping from the front - 3 digits, comma, 4 digits - instead of the
standard right-to-left thousands grouping). Unlike Step B4/B6's other three
categories (percentage restatement, identifier substrings, timestamp
fragments), this is **not a false positive** - the checker flagging it is
correct: the generated document states the payload size incorrectly. Per
instruction, investigated rather than suppressed.

**1. Determinism.** Confirmed deterministic, not random: two independent
Phase D runs (Step B5, Step B7) both produced the byte-for-byte identical
wrong string `"150,6992"`. Same input, same wrong output, every time.

**2. Presentation-format test.** Two experiments:
- *Isolated* (a single-fact, single-sentence prompt: "state this payload
  size"): tested `1506992` (unformatted int), `"1,506,992"` (comma-formatted
  string), and `"1.5 MB"` (human-readable string), 2 trials each. **All three
  forms were reproduced correctly in every trial** - `1506992`, `1,506,992`,
  and `1.5 MB` respectively, no mangling. This ruled out "phi3:mini simply
  cannot write large digit sequences" as the cause.
- *Real prompt* (the actual, full Phase D facts JSON and prompt - ~14
  fact numbers, receipts, epsilon, L2 norms, static caveats - not a
  simplified one-fact test): `size_bytes` swapped from the raw int
  `1506992` to the pre-formatted string `"1,506,992"`, 2 trials, everything
  else identical to a real run. **Both trials reproduced `1,506,992`
  correctly, with zero mangling.** The isolated test alone would not have
  been sufficient evidence - the real failure only ever occurred inside the
  busier, many-numbers prompt, so the fix needed to be proven there, not in
  simplification.

**3. Conclusion and recommendation.** The defect is not about the number's
magnitude or the checker's tolerance - it is that presenting `size_bytes` as
a bare JSON integer, inside a prompt already carrying many other numbers,
gives phi3:mini's small parameter count nothing to anchor the digit-grouping
to, and it improvises one - wrong - under load. Pre-formatting the value as
a comma-separated string *in the facts dict* (not the checker, not the
prompt instructions) fixed it reliably, 2/2 trials, under real prompt
conditions. **Recommended fix, not yet implemented pending approval**: in
`gather_facts()` (`privacy_explanation_agent.py`), format
`model_updates[i]["size_bytes"]` as `f"{value:,}"` before it enters the
facts dict, matching the same principle Phase B already applies to
per-patient data - present the value in the form the model can reproduce
faithfully, rather than trusting free-form generation to get formatting
right unaided. If ever a byte-count fact exceeded what this fix could
reliably reproduce, the fallback consistent with Phase B's own design would
be to state it only in the deterministic facts table and instruct the LLM
not to restate it in prose at all - but that fallback was not needed here;
the comma-formatting fix is proven sufficient.

**Two further false-positive categories surfaced during Step B7 re-
verification, both newly discovered, neither in scope for Step B4/B6, and
neither fixed:**
- A fact stated only inside a descriptive string (e.g.
  `static_system_facts.aggregation_strategy`'s prose mentions
  "trim_ratio=0.1") is correctly restated by the LLM but flagged
  untraceable, because `_flatten_numbers()` only walks real JSON *number*
  fields - strings are skipped by design (so string-embedded numbers were
  never facts to trace against in the first place).
- A file:line citation like `server.rs:1497-1500`, when echoed verbatim by
  the LLM, has its range hyphen parsed by `_NUMBER_RE` as a unary minus,
  producing a spurious `-1500.0`.

Both are reported here rather than fixed, consistent with this step's
scoping discipline - a candidate for a future, explicitly-scoped step.

---

## Phases D, A, B and C: the LLM explanation layer (2026-08-27)

This section is written for a reader who has not followed the step-by-step
work above - a summary of what exists, not a log of how it got built.

### What each agent is

Three post-hoc reporting scripts, all in `scripts/`, none of them part of
the live federated-learning pipeline. Each reads data that some other,
already-run process already produced and persisted; none of them compute a
new prediction, a new privacy metric, or a new attribution value.

**Phase D - `privacy_explanation_agent.py`.** Reads one federated round's
privacy/audit telemetry (MongoDB `receipts` + `model_updates` collections,
plus local DP-agent receipt files under `~/.federated/data/receipts/`,
joined by `session_id`). Writes
`~/.federated/data/audit_reports/round_<N>_privacy_report.md`. Intended
reader: an **auditor or compliance reviewer** checking a round's epsilon
spend, signature/HMAC-chain status, and aggregation configuration.

**Phase A - `xai_integrated_gradients()`, inside
`installer/runtime/agents/trainer/trainer_mentalbert_privacy.py`.** Not a
standalone report-reading script like the other two - it runs INSIDE the
training pipeline itself, gated behind `XAI_ENABLED` (default off), on the
local model immediately after local fine-tuning finishes and before DP/
encryption/upload. Reads the trained model and the held-out evaluation
split; writes `explain_logs/xai_ig_<session_id>_<timestamp>.json` and
`.txt`. Intended reader: whoever is investigating model behaviour
(the Steps A3-A8 investigation itself, and Phase B, which consumes its
output).

**Phase B - `clinical_narrative_agent.py`.** Reads Phase A's
`explain_logs/xai_ig_*.json`. Writes
`~/.federated/data/clinical_reports/round_<N>_clinical_report.md`. Intended
reader: a **clinician** deciding whether a round's screening output is
worth acting on.

### Local LLM only, never an external API

Both Phase D and Phase B call `call_ollama()` (`scripts/ollama_narration_utils.py`),
which talks to `http://localhost:11434` - a locally-running Ollama server,
model `phi3:mini` by default - and nothing else. No other network call
exists in either script. This was set up and verified in Phase D's own
Step D0 (getting Ollama running entirely on local disk, `D:\TeraBoxDownload\.ollama`,
confirmed via live TCP-connection inspection that the only established
connection during generation was loopback-to-loopback) and re-confirmed
structurally for Phase B in Step B3 (see below). This matters because the
telemetry these two agents narrate - epsilon values, device IDs, receipt
hashes, and in Phase B's case a mental-health screening cohort's prediction
data - is exactly the kind of operationally sensitive material this
project's core privacy claim (raw data and derived signals never leave the
client device) would be undermined by sending to a third-party API merely
to generate a paragraph of prose about it. Keeping the narration step local
is not an incidental implementation choice; it is required by the same
claim the rest of the system is built to support.

### Phase B's structural guarantee: per-patient data literally cannot reach the LLM

Phase A's investigation (Steps A3-A8, below) established that per-patient
modality attribution is not reliable - two independent methods (Integrated
Gradients and ablation) do not agree at the individual-patient level, most
likely because this model's predictions cluster tightly around the decision
boundary rather than committing confidently. Phase B was designed around
that finding, not despite it.

The design choice that matters most: `build_cohort_facts()`
(`clinical_narrative_agent.py`) builds the dict that becomes the LLM
prompt, and it **never includes the `per_sample` list** from Phase A's
report - only aggregate counts and metrics. This was not just asserted; it
was verified directly in Step B3 by reconstructing the actual facts dict
and the actual prompt string sent to Ollama and inspecting both: `'per_sample'
in facts` returned `False`, and the full ~3,100-character prompt, printed
in full, contained exactly ten aggregate keys and zero per-record data.

This makes "no per-patient attribution claims" a **structural** property of
the system, not an instruction the model might ignore. An LLM that never
receives a given patient's record cannot narrate that patient's record,
regardless of how it is prompted - a stronger guarantee than "the prompt
tells it not to." The per-patient table in the generated report is entirely
separate: a deterministic, template-rendered table (`build_per_patient_rows()`),
zero LLM involvement, so there is no generation step there to hallucinate
in the first place.

### The consistency checker

`check_consistency()` (`scripts/ollama_narration_utils.py`) is the
hallucination guard shared by both agents: every number the LLM's narrative
states must trace back to a number literally present in the facts dict it
was given (exact match, matched after rounding to fewer decimal places, or
within 0.1% relative tolerance). A number that fails this is reported as
"untraceable" and the report is marked WARNING rather than PASSED - the
facts table and (in Phase B) the per-patient table are unaffected either
way, since neither depends on the LLM succeeding.

Five categories of number are excluded from checking entirely - not
"forgiven," excluded, because they were never claimed measurements in the
first place:

1. **Generic small integers** - a bare number under 100 with no decimal
   point in the narrative text (e.g. "12 devices"). Present in the design
   since Phase D's original version, but broken until Step B8 - the
   original implementation checked the stringified *float* for a decimal
   point, and a Python float always stringifies with one (`f"{14.0}"` is
   `"14.0"`), so the exclusion never actually fired. Fixed by checking the
   raw matched text instead.
2. **Percentage restatements** - a number is also checked scaled by 100 if
   a `%` or the word "percent"/"percentage" immediately follows it in the
   text (e.g. the LLM writing `0.5676` as `"56.76%"`). Exists because
   converting a fraction to a percentage for readability is a faithful
   restatement, not a fabrication - but only when the text actually
   presents it as a percentage; a bare `56.76` with no such context is
   still flagged.
3. **Identifier substrings** - a number that sits inside a longer
   alphanumeric token (a session ID, device ID, or hash quoted verbatim,
   e.g. the `1978` inside `client-aef1978aef7f`) is not a claimed
   measurement. Exists because these hex-ish tokens mix letters and digits
   with no separator, or use a hyphen as an internal token separator, and
   the LLM correctly quoting one should not be penalised.
4. **Timestamp fragments** - a number that is a substring of a
   clock-time or ISO-8601 datetime span (e.g. the `48` inside
   `2:48:09.825000`) is a clock component, not a measurement.
5. **Written-date fragments** - a number adjacent to an actual month name
   in a date-shaped construction (e.g. `26` and `2026` inside
   "August 26, 2026"). Anchored to real month names, not "any number near
   any capitalised word" - "In May, 26 patients..." does not exclude the
   26, since the comma breaks the required month-day adjacency.

None of these five loosen what counts as a genuine fabrication - a number
with none of these contexts still must match a fact number or it is
flagged. This was proven, not assumed: a narrative was deliberately
constructed containing a fabricated F1 score, a fabricated percentage, and
a fabricated negative number, mixed in among a real timestamp, a real
session ID, and a real written date. Result: all three fabrications were
flagged as untraceable, and all three real, non-fabricated numbers were
correctly excluded rather than flagged as noise. Re-run after every
addition to the checker (Steps B4, B6, B8) to confirm no change had
quietly weakened it.

### The Step B6 finding: a real generation defect, not suppressed

Phase D's checker flagged a payload byte count (`1506992`) that phi3:mini
had written into the narrative as `"150,6992"` - the digits in the right
order, the comma in the wrong place. Investigation (not suppression) found
this to be deterministic (two independent runs produced the byte-identical
wrong string) and specific to prompt complexity, not the number's
magnitude - an isolated, single-fact prompt reproduced the value correctly
in every form tested (unformatted, comma-formatted, "1.5 MB"), but the
mangling reappeared reliably inside the real, many-numbers Phase D prompt.
The fix that was proven to work, 2-for-2 under the real prompt: **pre-format
the number as a comma-grouped string in the facts dict itself**
(`gather_facts()`, Step B8), rather than adjusting the checker to tolerate
the wrong output. The checker flagging a genuinely incorrect number in a
generated document was correct behaviour; the fix belonged at the data
layer, not the guard layer.

### Phase A's limits, and what Phase B was built to respect

Phase A's own investigation (Steps A3-A8, the "Phase A: what the attribution
mechanism can and cannot support" section above) is not repeated here.
Summary only (all figures below: session `client-aef1978aef7f` - see that
section's attribution note; a different session, `client-405c6057ab84`,
measures 37 of 37 rather than 36 of 37 on the identical split): an initial
unanimous 37/37 "audio dominates" result traced to a raw-scale baseline
artefact, not a real signal; fixing the baseline resolved the artefact but
did not make Integrated Gradients and ablation agree at the per-patient
level; the n_steps hypothesis for that remaining disagreement was tested and
refuted; the most likely underlying cause is that 36 of 37 held-out
predictions sit in a low-confidence 0.3-0.7 band on this checkpoint, not
a defect in either attribution method. The four constraints that section
places on any downstream narrative - no per-patient modality percentages,
no claim a modality "drove" an individual prediction, must surface the
model's uncertainty, aggregate/cohort-level statements are supportable -
are what Phase B's design (this section, above) was built around from the
start, not retrofitted afterward.

### Runtimes

- Phase D (`privacy_explanation_agent.py`): ~79s per round (data gathering
  well under 1s; almost entirely local Ollama generation time).
- Phase B (`clinical_narrative_agent.py`): ~47s per round - faster than
  Phase D despite writing a longer report, because the per-patient section
  is template-only (no LLM call) and the cohort prompt is smaller (aggregate
  facts only, never per-sample records).
- Phase A attribution (`xai_integrated_gradients()`): ~370s for a 37-record
  held-out set at n_steps=25 (the default) - this is why `XAI_ENABLED`
  defaults off; it is a meaningful addition to a training round's wall
  time, not a rounding error.

### What is NOT done: Phase C (RAG) does not exist

The project document describes a "Vector Index Manager + Vector Database"
retrieval-augmented-generation layer. It does not exist as a real,
persisted retrieval system. `build_rag_features()`
(`create_dp_comparison.py`) uses a real `sklearn.neighbors.NearestNeighbors`
call, but retrieves from the current batch only - there is no persisted
vector store, nothing survives between runs - and the mode dispatch for
`rag` and `vector_rag` is code-identical, so the document's claimed
distinction between the two modes does not exist in the code. The
function's own docstring states plainly that it "simulates retrieval
latency in a deterministic (seeded) way" - the `rag_mean_latency` figures
this produces are not measured timings of anything. This was established in
an earlier, separate investigation (report-only, not written up in this
file at the time) and is recorded here because this section is the natural
place a reader would look for it alongside Phase D/A/B. No Phase C agent
had been designed or built AT THE TIME THIS PARAGRAPH WAS WRITTEN - it has
since been built, with a deliberately different scope than the document
describes. See the subsection immediately below.

### Phase C, built: a clinical knowledge grounding layer, not the document's patient-embedding RAG

The RAG design above (patient-embedding retrieval, simulated latency) was
explicitly NOT revived. In a federated setting, retrieving one patient's
embedding to inform another patient's prediction is itself a privacy
problem, independent of it never having been on the live path. What was
built instead: `scripts/clinical_knowledge_corpus.py` +
`clinical_narrative_agent.py`'s retrieval step - a small, local corpus of
**reference material only, zero patient data**, that grounds Phase B's
cohort narrative so its clinical framing is traceable to a cited source
instead of generated from a 3B model's unverifiable parametric memory.

**The corpus's two tiers, and why a third was deliberately left out.**
Tier 1 is PHQ-8 item wording, response scale, and scoring range, cited to
Kroenke et al. 2009 - fixed, standardized, publicly-published instrument
text. Tier 2 is this project's own Phase A findings, cited to this same
file - the project citing itself, verifiable by reading it. A third
category - feature-to-symptom mappings, e.g. "reduced AU06/AU12 activation
↔ blunted affect" - was considered and explicitly rejected. These are
specific, contested, actively-researched empirical claims, not fixed public
facts like PHQ-8's wording; drafting them without a real, specific,
peer-reviewed citation would have been introducing exactly the kind of
unverifiable clinical claim this whole layer exists to prevent - "two
unverified things, not one," stacking an unverified clinical claim on top
of an attribution mechanism Phase A had already shown to be unreliable at
the individual-patient level. That gap is left visible in the corpus
(11 Tier-1 + 6 Tier-2 = 17 documents, no Tier 3) rather than filled with
something plausible.

**Every Tier 1 document carries `verified: False`** and a
`verification_note` stating it was reproduced from memory of a standard
instrument, not checked against a primary copy. This flag is designed to
survive into the generated report, not just sit in the corpus file - see
the compliance-gap discussion below for how that survival was ultimately
made structural rather than left to the LLM.

**Embedding model trade-off, and the retrieval miss it caused.** MentalBERT
(already on local disk, zero new download) was chosen over
sentence-transformers (would need a first-time HuggingFace download,
against this layer's local-only principle) and Ollama's embeddings endpoint
(unsupported by the running server, and phi3:mini isn't trained for
embeddings anyway). The honest trade-off stated at design time: MentalBERT's
raw `[CLS]` embeddings, with no contrastive/triplet fine-tuning for semantic
similarity, are a documented-weaker retrieval signal than a purpose-built
sentence-embedding model. This showed up in practice, not just in theory:
`project-per-patient-unreliable` - arguably the single most on-topic
document in the corpus for the "how reliable is per-patient modality
attribution" query, given its title - did not surface in the top-2 results
for that query, and **still did not surface after widening to top-3**
(Step C4). It was consistently outranked by `project-model-training-limits`
and `project-attribution-scope`, both topically adjacent but less precisely
on-point. Not fixed; recorded as a known limitation of the embedding choice
for a future session to weigh against the cost of a downloaded,
similarity-tuned model.

**Two phi3:mini instruction-compliance gaps, and how each was handled.**
The prompt asks the model (rule 8) to cite retrieved sources inline, next to
each claim, and (rule 10) to say "unverified" explicitly whenever it cites
an unverified source. In live verification runs, phi3:mini complied with
neither reliably: it collapsed inline citations into one list at the end of
the narrative (with an occasional typo in the source name), and in at least
one run cited unverified PHQ-8 content without ever stating it was
unverified anywhere in the narrative body. Rather than iterating on the
prompt and trusting compliance, the same structural principle used
elsewhere in this layer (Phase B's per-patient table, the numeric
consistency checker) was applied here too: `build_unverified_sources_warning()`
generates a plain-Python sentence - not an LLM output - listing every
unverified source by title, and it is placed immediately before the
narrative in the generated report whenever any retrieved document has
`verified: False`. A reader sees the warning regardless of what the model
wrote. Rule 10 stays in the prompt as well - belt and braces, not a
replacement for the structural guarantee.

---

## Accepted, deferred: checkpoint versioning (2026-09-01)

Investigation for `scripts/demo_predictions.py` found that
`train_model()` (`trainer_mentalbert_privacy.py:604-608,775-789`) saves the
trained model and its metrics to fixed filenames
(`mentalbert_privacy_subset.pt`, `metrics.json`, `metrics_report.json`)
under a fixed `output_dir`, with no `session_id` in scope at that call - every
client round silently overwrites the previous round's checkpoint and
metrics. This was discovered because two specific historical checkpoints
(`client-405c6057ab84`, `client-aef1978aef7f`) were needed for a mentor
demo and both were already gone by the time this was noticed - only the
timestamped `explain_logs/xai_ig_<session_id>_<timestamp>.json` reports
survived, because that path already includes the session_id (a convention
this fix would extend to the checkpoint/metrics path).

**Proposed fix (accepted, NOT implemented)**: add an optional
`session_id: Optional[str] = None` parameter to `train_model()`; when given,
embed it in the output filenames (e.g. `mentalbert_privacy_subset_{session_id}.pt`),
falling back to today's exact fixed filename when `None` - backward-compatible
by construction, same pattern already used for this function's own
`eval_dataset=None` parameter. At the one call site, in `orchestrate()`,
pass `session_id=session_id` - already in scope there.

**Deliberately not implemented yet**: this touches the training path, and a
mentor demo was imminent when it was found. Do this after the demo, not
before. `scripts/demo_predictions.py` and its anchor-session backup
(`~/.federated/data/anchor_session_backups/client-405c6057ab84/`, mirrored
into this repo at `anchor_session_backups/client-405c6057ab84/`) exist as the
interim workaround - reading a specific session's persisted `per_sample`
data rather than depending on the checkpoint still being on disk.

---

## Confirmed dead code: `installer/runtime/pipeline.py` (2026-09-20)

While tracing the live call chain to lower `noise_multiplier` (see "Noise
reduction" below), confirmed `installer/runtime/pipeline.py` - a second,
separate `pipeline.py` that hardcodes `DPAgent(clip_norm=1.0,
noise_multiplier=1.0, ...)` - is **imported by nothing anywhere in this
codebase**. Searched both `from installer.runtime.pipeline import` /
`from installer.runtime import pipeline` and, separately, `from pipeline
import` / `import pipeline` (the form it would take if only
`installer/runtime` were on `sys.path`, as some callers do): zero hits
either way.

The actual live pipeline is the top-level `runtime/pipeline.py`, reached via
`run_client_multimodal.py` -> `runtime.federated_client` ->
`from runtime.pipeline import run_pipeline`. It reads `clip_norm`/
`noise_multiplier` from `DP_CLIP_NORM`/`DP_NOISE_MULTIPLIER` env vars
(defaults `"0.85"`/`"0.8"` as of this entry), matching every real DP receipt
on disk. `installer/runtime/pipeline.py`'s hardcoded `1.0`/`1.0` values were
never live - not a second production path, not stale-but-reachable, just
unreferenced. Left as-is (not deleted) per instruction, noted here so nobody
spends time on it later thinking it's real.

---

## Noise reduction: `noise_multiplier` 1.0 -> 0.8 (2026-09-20)

Historical delta L2 (the real training signal) never exceeds 0.7658 across
219 recorded DP operations (`~/.federated/data/receipts/receipt_*.json`,
every `dp_process_update` receipt this project has ever produced) - DP noise
at the old `noise_multiplier=1.0` (mean noise norm 450.81 across 110
gaussian-mechanism receipts at `clip_norm=0.85`) was swamping that signal by
~590x. `clip_norm` (0.85) is untouched by this change - clipping has never
fired in this project's history (confirmed: `clip_applied` is `False` in
all 219 receipts) and remains unrelated to noise scale by design.

**Sigma -> epsilon**, computed via the real `_rdp_to_dp()` in
`installer/runtime/agents/dp/dp_agent.py` (not reimplemented):

| sigma | epsilon |
|---|---|
| 1.0 (old) | 5.302585 |
| 0.9 | 5.964651 |
| 0.8 (new) | 6.784481 |
| 0.7 | 7.919274 (only ~1% margin under 8 - rejected) |
| 0.6 | 9.393197 (already exceeds 8) |
| 0.5 | 11.756463 |

`sigma=0.8` chosen: eps=6.784481, ~15% margin under the eps<=8 target -
comfortable without being the tightest possible value.

**Linear noise-scaling, confirmed empirically** from real historical
receipts before this change (not assumed from theory alone): grouped every
gaussian-mechanism receipt at `clip_norm=0.85` by `noise_multiplier`, using
`l2_after` as a proxy for noise norm (valid since `l2_before` ~0.6 is
negligible next to noise ~450):

| sigma | n | mean l2_after | predicted (sigma=1.0 mean x ratio) | error |
|---|---|---|---|---|
| 1.0 | 110 | 450.8117 | - | - |
| 0.5 | 2 | 225.0669 | 225.4058 | -0.150% |
| 0.25 | 1 | 112.5117 | 112.7029 | -0.170% |

Matches prediction to within 0.17% - confirms `add_noise()`'s
`scale = noise_multiplier * sensitivity` (`dp_agent.py:152`) behaves as
documented at production scale (281,254-dim noise vector).

**Two files changed, one left alone by design**:
- `runtime/pipeline.py:473` - the `DP_NOISE_MULTIPLIER` fallback literal,
  `"1.0"` -> `"0.8"`. This is the value that actually reaches every live
  client run (the caller always passes it explicitly to `DPAgent`) -
  the real fix.
- `installer/runtime/agents/dp/dp_agent.py:97` - the `DPAgent.__init__`
  default, `1.0` -> `0.8`. Has no live effect on its own (the caller above
  always overrides it), changed only so the class's own default doesn't
  silently disagree with what's actually running.
- `DP_CLIP_NORM`/`DP_NOISE_MULTIPLIER` env vars still override both
  defaults when explicitly set (used by research scripts like
  `run_step12_combo.py`, `run_step13_combo.py`) - verified unaffected by
  this change, since both files still read `os.environ.get(..., <new
  default>)`.

---

## Tighter RDP->DP accounting + second noise reduction: 0.8 -> 0.75 (2026-09-20)

Follow-up to the noise reduction above. The classic RDP->DP conversion
(`_rdp_to_dp()`, Mironov 2017: `eps = min_alpha[alpha/(2*sigma^2) +
log(1/delta)/(alpha-1)]`, minimized over integer alpha in [2,257)) is a known
loose bound. Two tighter, well-established alternatives were evaluated
before touching anything:

- **(ii) Canonne-Kamath-Steinke tightening** (NeurIPS 2020, Prop. 12,
  building on Balle et al. 2020 AISTATS and Asoodeh et al. 2020): adds a
  `log((alpha-1)/alpha)` term and replaces `log(1/delta)` with
  `-(log(delta)+log(alpha))`, provably <= the classic bound for every alpha.
  Minimized at a *continuous* alpha (~4.576 at sigma=0.8, not an integer),
  so implemented with `scipy.optimize.minimize_scalar` (bounded Brent's
  method) rather than the classic function's integer grid - see
  `_rdp_to_dp_tight()` in `dp_agent.py` for the full derivation and formula.
- **(iii) Analytic Gaussian mechanism** (Balle & Wang, ICML 2018): the
  *exact* (eps,delta) for a pure Gaussian mechanism (not an RDP bound at
  all), via `delta(eps) = Phi(mu/2 - eps/mu) - e^eps*Phi(-mu/2 - eps/mu)`,
  `mu = 1/sigma`, solved by bisection. Implemented with
  `scipy.special.log_ndtr` for numerical stability (a naive `math.erf` +
  `math.exp(eps)` implementation overflows for large eps).

**Three-method comparison** (delta=1e-5; sensitivity=clip_norm cancels out
of all three formulas, so each depends only on sigma and delta):

| sigma | (i) classic (current) | (ii) tighter RDP | (iii) analytic Gaussian (exact) |
|---|---|---|---|
| 0.50 | 11.756463 | 10.724824 | 9.997256 |
| 0.60 | 9.393197 | 8.603231 | 8.003691 |
| 0.65 | 8.571370 | 7.819462 | 7.268511 |
| 0.70 | 7.919274 | 7.162095 | 6.652488 |
| 0.75 | 7.322676 | 6.603254 | 6.129245 |
| 0.80 | 6.784481 | 6.122631 | 5.679587 |
| 0.90 | 5.964651 | 5.339144 | 4.947319 |
| 1.00 | 5.302585 | 4.728387 | 4.377178 |

**Method chosen: (ii) tighter RDP, not (iii) analytic Gaussian - a
deliberate trade against the mathematically tighter option.** (iii) is
exact only for a *single* Gaussian mechanism application; it does not
compose additively across rounds the way RDP does. This project has a known
open defect: privacy accounting is per-round only, with no cumulative
epsilon tracked across rounds. RDP composes trivially when that gets fixed
(sum `alpha/(2*sigma_r^2)` across rounds, convert once at the end via either
`_rdp_to_dp()` or `_rdp_to_dp_tight()`). Using the analytic Gaussian
mechanism per-round and summing the resulting epsilons across rounds would
be a **double-conversion error**, not a valid composition. (ii) was chosen
specifically so this future fix stays correct instead of requiring a second
accounting migration.

**Sigma chosen: 0.75, not 0.70 - because the margin target is evaluated
under the method actually in use.** Under (ii), sigma=0.70 gives
eps=7.162095, only ~10.47% margin under the eps<=8 target - thinner than
wanted. sigma=0.75 gives eps=6.603254, ~17.5% margin. (An earlier version of
this exercise, using method (iii) as the yardstick, would have recommended
sigma=0.70 - that recommendation does not carry over now that (ii) is the
adopted method, since the two methods rank sigma values by different
margins.)

**Live verification** (one real client round end-to-end, session
`client-c959272d18a2`, receipt `receipt_f3ad77e6efcd43019f349aa56864c322.json`):

| quantity | measured | predicted | old method (i), same run |
|---|---|---|---|
| noise_multiplier | 0.75 | 0.75 | - |
| l2_norm_before (signal) | 0.6900 | ~0.6-0.77 (historical range) | unaffected - confirms only noise changed |
| l2_norm_after | 338.6191 | 450.81 x 0.75 = 338.11 (0.15% error, matches the Fix E4 empirical scaling law) | - |
| epsilon_spent (live, method ii) | 6.603254 | 6.603254 | 7.322676 (classic, same sigma - shown for audit, not what's reported) |

Local eval this run: accuracy=0.4054, precision=0.3103, recall=0.8182,
f1=0.4500. This is the same "noise dominates -> biased-positive collapse"
pattern already documented earlier in this file (Step 16 round 2:
F1=0.4583, `prob_positive_distribution` saturated) - high recall, low
precision, mid-0.4x F1 - not a new failure mode. Noise norm (338.62) still
overwhelms signal (0.69) by ~490x even at sigma=0.75, so this collapse
pattern is expected to persist regardless of accounting method; nothing
outside the accounting math moved.

**Files changed** (same two as the first noise reduction, kept in sync):
- `installer/runtime/agents/dp/dp_agent.py`: added `_rdp_to_dp_tight()`
  (new function; `_rdp_to_dp()` kept intact and callable for reproducing
  historical/classic numbers), switched `process_local_update()`'s live
  `epsilon_spent` computation to call it, updated the `noise_multiplier`
  class default `0.8 -> 0.75`, updated the "Accounting method" report line.
- `runtime/pipeline.py`: `DP_NOISE_MULTIPLIER` fallback literal `"0.8" ->
  "0.75"`.
- `dp_agent/dp_agent.py` (the separate research harness/dead-code file
  documented above) was **not** touched, per instruction.

---

## 14-run sigma comparison: collapse-mode distribution is sigma-independent (2026-09-20)

Follow-up investigation after the sigma=0.75 switch above, prompted by two
single runs at sigma=0.75 landing far apart (F1 0.4500 and 0.0000) versus
the anchor session's F1=0.5517. Ran 7 more client rounds at sigma=0.75 (for
7 total) and, as a control with **no code changes** (`DP_NOISE_MULTIPLIER=1.0`
env var only, same code path), 7 client rounds at the original sigma=1.0.
Model, training loop, LR, epochs, data, splits and clip_norm were untouched
throughout - only the noise_multiplier env var varied.

**sigma=0.75 (7 runs):**

| run | accuracy | precision | recall | F1 | delta L2 | collapse mode |
|---|---|---|---|---|---|---|
| 1 | 0.4054 | 0.3103 | 0.8182 | 0.4500 | 0.6900 | predict-mostly-positive |
| 2 | 0.7027 | 0.0000 | 0.0000 | 0.0000 | 0.6837 | predict-all-negative |
| 3 | 0.7027 | 0.0000 | 0.0000 | 0.0000 | 0.7732 | predict-all-negative |
| 4 | 0.7027 | 0.0000 | 0.0000 | 0.0000 | 0.7180 | predict-all-negative |
| 5 | 0.7027 | 0.0000 | 0.0000 | 0.0000 | 0.7296 | predict-all-negative |
| 6 | 0.7027 | 0.0000 | 0.0000 | 0.0000 | 0.4044 | predict-all-negative |
| 7 | 0.7027 | 0.0000 | 0.0000 | 0.0000 | 0.4117 | predict-all-negative |

F1 distribution: mean=0.0643, median=0.0000, min=0.0000, max=0.4500,
stdev=0.1701 (sample). Collapse tally: 6/7 predict-all-negative, 1/7
predict-mostly-positive, 0/7 genuine discrimination.

**sigma=1.00 (7 runs, control, `DP_NOISE_MULTIPLIER=1.0`):**

| run | accuracy | precision | recall | F1 | delta L2 | collapse mode |
|---|---|---|---|---|---|---|
| 1 | 0.7027 | 0.0000 | 0.0000 | 0.0000 | 0.3949 | predict-all-negative |
| 2 | 0.2973 | 0.2973 | 1.0000 | 0.4583 | 0.4828 | predict-all-positive |
| 3 | 0.2973 | 0.2973 | 1.0000 | 0.4583 | 0.4840 | predict-all-positive |
| 4 | 0.2973 | 0.2973 | 1.0000 | 0.4583 | 0.4933 | predict-all-positive |
| 5 | 0.2973 | 0.2973 | 1.0000 | 0.4583 | 0.6922 | predict-all-positive |
| 6 | 0.2973 | 0.2973 | 1.0000 | 0.4583 | 0.6912 | predict-all-positive |
| 7 | 0.2973 | 0.2973 | 1.0000 | 0.4583 | 0.6969 | predict-all-positive |

F1 distribution: mean=0.3928, median=0.4583, min=0.0000, max=0.4583,
stdev=0.1732 (sample). Collapse tally: 6/7 predict-all-positive, 1/7
predict-all-negative, 0/7 genuine discrimination.

**Verdict: collapse rate is sigma-independent.** 6 of 7 runs collapsed into
a degenerate single-class predictor at *both* sigma=0.75 and sigma=1.0 -
identical 85.7% collapse rate. Zero runs at either sigma achieved genuine
discrimination. Reducing sigma (0.8 -> 0.75, see above) did not push this
pipeline from "mostly works" to "mostly collapses" - it was already
collapsing 6/7 of the time at the original sigma=1.0.

The mean-F1 gap between the two groups (0.0643 vs 0.3928) is **not**
evidence that sigma=1.0 performs better - it is an artifact of which
degenerate mode happened to dominate each group of 7. The two collapse
modes are not symmetric in F1 on this eval split's fixed 29.73% positive
base rate (37 held-out patients, `stratified_split()`, seed=42): predict-
all-positive always scores precision=0.2973 (the base rate), recall=1.0000,
F1=**0.4583** exactly; predict-all-negative always scores precision=recall=
F1=**0.0000** exactly (see also lines 209/213/215/436/537-539/595/694/996-997
above, where these same two exact values recur every time either collapse
mode fires, across unrelated earlier experiments). Which mode a run falls
into looks like a coin flip (no fixed init seed) - both groups landed on
both modes at least once.

**This directly re-contextualizes the anchor session
(`client-405c6057ab84`, F1=0.5517241379310345,
accuracy=0.6486486486486487) used throughout `scripts/demo_predictions.py`
and the mentor-demo materials.** 0.5517 exceeds *both* degenerate-mode
ceilings (0.4583) and remains above-typical. **Superseded by the larger,
better-measured fusion-head sweep below (2026-09-20): the anchor is no
longer "the ONE run" that discriminated** - that claim was itself a
small-sample artifact of the original 14-run count. Across the 55-run
fusion-head sweep, 16 runs (29.1%) achieved genuine discrimination, and one
of them (`FUSION_HIDDEN_DIM=128`, run 24, F1=0.6207) **exceeded the anchor
outright**. The anchor is still a good, above-typical result - it just
isn't unique the way the original disclosure claimed.

**Machine-readable summary for `scripts/demo_predictions.py`'s
representativeness disclosure block** (parsed at runtime, not hardcoded in
the script - see that file's `_load_disclosure_facts()`; updated 2026-09-20
to reflect the fusion-head sweep superseding the original 14-run sigma
comparison, which is kept above for its own record but is no longer what
the live disclosure quotes):

```
DISCLOSURE_ANCHOR_SESSION_ID: client-405c6057ab84
DISCLOSURE_ANCHOR_F1: 0.5517
DISCLOSURE_SWEEP_TOTAL_RUNS: 55
DISCLOSURE_SWEEP_GENUINE_COUNT: 16
DISCLOSURE_SWEEP_GENUINE_RATE_PCT: 29.1
DISCLOSURE_SWEEP_BEST_RATE_PCT: 40.0
DISCLOSURE_SWEEP_WORST_RATE_PCT: 0.0
DISCLOSURE_SWEEP_BEST_RUN_F1: 0.6207
DISCLOSURE_POSITIVE_COLLAPSE_F1: 0.4583
DISCLOSURE_NEGATIVE_COLLAPSE_F1: 0.0000
DISCLOSURE_BASE_RATE_POSITIVE_PCT: 29.73
```

---

## Fusion-head capacity sweep: a capacity floor between 75,454 and 30,178 params (2026-09-20)

**Lead finding, the one to trust:** at fusion-head hidden size 12 (30,178 total
trainable params) and hidden size 1 (18,859 params), **0 of 10 combined runs**
(5 at each size) produced any non-uniform prediction. Every single run
collapsed into an exact all-positive or exact all-negative predictor - no
exceptions, at two independently-tested, very different parameter counts.
This is a hard floor: below it, this architecture cannot do anything but
collapse on this 149-record training partition. Capacity starts hurting
somewhere between 75,454 and 30,178 trainable params.

**Motivation.** `fusion.fc1` (`Linear(1024, hidden)`) holds 93.3% of the
281,254-param trainable surface (262,400 of 281,254 at the default
`hidden=256`) - by far the dominant lever on both model capacity and DP
noise dimension (noise scales as sqrt(params)). Working hypothesis: 281,254
parameters may be too many to learn from 149 training records with 44
positives, and DP noise on that many parameters is itself a large cost
(sqrt(281254) vs sqrt(a smaller count)). Made `FusionHead`'s `hidden` size
configurable via `FUSION_HIDDEN_DIM` (`trainer_mentalbert_privacy.py`,
default `"256"`, reproduces the original 281,254-param architecture
byte-for-byte) to test whether a smaller head both reduces noise and
changes the collapse rate.

**Total trainable params as an exact function of `FUSION_HIDDEN_DIM` (H)**,
derived and verified byte-exact against the 281,254 baseline:
`total(H) = 1029*H + 17830` (17,826 of that is the fixed audio_encoder +
vision_encoder overhead, unaffected by this knob). Reachability constraint
recorded during discovery: the originally-proposed 10K/3K targets are not
reachable via this knob alone (floor at H=1 is already 18,859) because that
would require also shrinking the audio/vision encoders' own `out_dim`
(currently fixed at 128), which would change fusion's 1024-dim input at
the same time and confound attribution to either knob - explicitly out of
scope per this investigation's own decision.

**Correction to a documented ambiguity**: this investigation traced
`MultiModalModel.forward()` and confirmed the fusion head's real input is
**768 (text) + 128 (audio, post-`SmallMLP` projection) + 128 (vision,
post-`SmallMLP` projection) = 1024** - not the raw 768+154+84=1006 an
initial reading of the dimensions suggested. Audio (154-dim wav2vec2) and
video (84-dim DenseNet) each pass through their own small encoder to
128-dim *before* concatenation with the 768-dim MentalBERT `[CLS]` vector.
This resolves the stale doc conflict noted elsewhere in this file and in
CLAUDE.md ("fusion 192 vs 1024") in favor of the code-measured **1024**.

**Five variants tested** (`FUSION_HIDDEN_DIM` env var only; model
architecture is otherwise identical, same lr/epochs/batch size/data/split/
label path/clip_norm/sigma throughout):

| H | total trainable params | predicted noise @ sigma=0.75 (sqrt scaling from 338 @ 281,254) | measured mean noise |
|---|---|---|---|
| 256 (baseline) | 281,254 | 338 (reference) | 338.18 |
| 128 | 149,542 | 246.5 | 246.36 |
| 56 | 75,454 | 175.1 | 174.90 |
| 12 | 30,178 | 110.7 | 110.98 |
| 1 | 18,859 (floor - audio/vision encoder overhead alone) | 87.5 | 88.07 |

Noise scaling confirmed empirically (not just asserted from the formula) to
within 0.1-0.2% of the sqrt(params) prediction at every one of the 5
variants, across a 15x parameter range.

**Sample-size lesson - reported because it changed the conclusion.** An
initial n=5-per-variant sweep suggested H=128 and H=56 each reached 2/5
(40%) genuine-discrimination runs versus baseline's 1/5 (20%) - looked like
smaller heads might help. Tripling to n=15 (10 more runs each at H=256,
128, and 56; H=12 and H=1 left at n=5 since the floor there was already
unanimous) reversed this:

| H | params | n | all-positive | all-negative | NEITHER (genuine discrimination) | rate |
|---|---|---|---|---|---|---|
| 256 (baseline) | 281,254 | 15 | 5 | 4 | **6** | **40.0%** |
| 128 | 149,542 | 15 | 8 | 2 | **5** | **33.3%** |
| 56 | 75,454 | 15 | 5 | 5 | **5** | **33.3%** |
| 12 | 30,178 | 5 | 2 | 3 | **0** | **0%** |
| 1 | 18,859 | 5 | 2 | 3 | **0** | **0%** |

At n=15, baseline (40.0%) is *higher* than both H=128 and H=56 (33.3%
each) - the apparent n=5 advantage for smaller heads did not survive and,
if anything, reversed. **Conclusion: shrinking the fusion head from
281,254 to 149,542 or 75,454 params neither reliably helps nor reliably
hurts the genuine-discrimination rate** - all three are statistically
indistinguishable at this sample size (a 1-run difference out of 15). The
only result that held up under a larger sample, and got *stronger* (0/10
combined, not 0/7 or 0/5 individually), is the floor below 75,454 params.

Classification rule used throughout: "NEITHER" = at least one prediction
differs from the rest (not a uniform all-one-class output) - the literal
primary metric requested. A secondary quality check (F1 exceeding the
fixed 0.4583/0.0000 collapse ceilings) gives a similar ranking (baseline
3/15, H=128 2/15, H=56 2/15) - some NEITHER runs are non-uniform but still
poor (e.g. one H=128 run at F1=0.1333, one H=56 run at F1=0.1538), so
"NEITHER" alone slightly overstates how many runs were genuinely *good*,
not just genuinely *non-degenerate*.

**No tuning was performed anywhere in this investigation** - not lr, not
epochs, not batch size, not class weights, not clip_norm, not sigma. Only
`FUSION_HIDDEN_DIM` varied across the 70 total runs (25 initial + 30
follow-up + the pre-existing 15 sigma-sweep baseline runs are separate).

