# FL Algorithms in This Project — Summary

For a reader who has not followed the investigation. What's actually
implemented, where it lives, what was measured, and what isn't done and why.
Live demo: `scripts/demo_fl_algorithms.py` (runs in ~2s, fully offline).
Full analysis: `docs/IMPLEMENTATION_NOTES.md`, "Step 12" through "Step 18c".

## What each algorithm does

### FedAvg — plain federated averaging
Every client trains locally on its own data and uploads its parameter
update (`Δ_k`). The server averages the updates across `N` clients and
applies the average directly to the global model. No history, no
adjustment for how noisy or spread-out the updates are.

```
Δ_t = (1/N) * Σ_k Δ_k              (server-side average)
M_{t+1} = M_t + Δ_t                (applied directly)
```

### FedAdam — server-side Adam on the aggregated update
Client-side training is identical to FedAvg. On the server, the averaged
update `Δ_t` is treated as a "pseudo-gradient" and run through one step of
the Adam optimizer (Reddi et al. 2020, *Adaptive Federated Optimization*):
a running average of the update's *direction* (`m`) and a running average
of its *squared magnitude* (`v`), then the step is divided by `sqrt(v)`
before being scaled by a server learning rate `eta_s` and applied.

```
m_t = beta1*m_{t-1} + (1-beta1)*Δ_t
v_t = beta2*v_{t-1} + (1-beta2)*Δ_t^2
M_{t+1} = M_t + eta_s * m_t / (sqrt(v_t) + tau)
```
Hyperparameters used (CLAUDE.md, env-overridable): `eta_s=1e-3, beta1=0.9,
beta2=0.999`. CLAUDE.md does not state `tau` for FedAdam specifically — this
implementation borrows FedYogi's stated `tau=1e-3` (Reddi et al. 2020
conventionally shares one `tau` across both variants; noted as a deliberate,
documented choice, not a silent gap).

### FedYogi — server-side Yogi on the aggregated update
Identical to FedAdam except how `v` is updated: instead of a smooth
exponential average, `v` moves toward the new squared-magnitude value by a
*fixed-size* step in whichever direction shrinks the gap. Designed (Reddi et
al. 2020) to behave better than Adam under noisy/heavy-tailed gradients.

```
m_t = beta1*m_{t-1} + (1-beta1)*Δ_t
v_t = v_{t-1} - (1-beta2)*sign(v_{t-1} - Δ_t^2)*Δ_t^2
M_{t+1} = M_t + eta_s * m_t / (sqrt(v_t) + tau)
```
Hyperparameters used (CLAUDE.md): `eta_s=1e-2, tau=1e-3`. `beta1`/`beta2`
not stated for FedYogi specifically — borrowed from FedAdam's stated
`0.9`/`0.999`, same documented-not-silent convention as above.

## Where each lives in this codebase

| Algorithm | File : function | Pipeline stage |
|---|---|---|
| FedAvg (unweighted mean) | `server/aggregator_agent/aggregator.py`, `AggregatorAgent._aggregate_tensor()`, `mode="mean"` | **Aggregation** — combining N clients' uploaded deltas into one |
| FedAvg (server step) | `scripts/fl_optimizers.py`, `fedavg_step()` (identity — the aggregated delta IS the applied update) | **Post-aggregation server optimizer** (trivial/absent for FedAvg) |
| FedAdam | `scripts/fl_optimizers.py`, `fedadam_step()` | **Post-aggregation server optimizer** — runs on the already-aggregated delta, after `AggregatorAgent` produces it |
| FedYogi | `scripts/fl_optimizers.py`, `fedyogi_step()` | **Post-aggregation server optimizer**, same stage as FedAdam |
| Driver that exercises all three across real multi-round federated learning | `scripts/run_step14_multiround.py`, `--update-rule {plain,fedadam,fedyogi}` | orchestrates: client training (unmodified, real gRPC/DP/TPM pipeline) → aggregation (`aggregate_offline()`, unmodified `AggregatorAgent`) → server optimizer (this file) → warm-start patch for the next round |

None of FedAvg/FedAdam/FedYogi touch **client-side training**
(`trainer_mentalbert_privacy.py`) — all three see identical client behavior;
they differ only in what happens to the already-aggregated update
afterward, server-side.

## Measured results

5 rounds, DP enabled (`noise_multiplier=1.0`, `clip_norm=0.85`), mean
aggregation, seed=303, `LR_DECAY=0.7` (Step 16's default). Full precision
and provenance (exact log file + commit) in
`scripts/demo_fl_algorithms.py`'s `MEASURED_TRAJECTORIES`; full narrative
analysis in `docs/IMPLEMENTATION_NOTES.md`'s Step 17/18 sections.

| round | plain applied Δ L2 | plain F1 | FedAdam applied Δ L2 | FedAdam F1 | FedYogi applied Δ L2 | FedYogi F1 |
|---|---|---|---|---|---|---|
| 1 | 260.36 | 0.0000 | 1.438 | 0.0000 | 14.386 | 0.0000 |
| 2 | 260.76 | 0.4583 | 1.484 | 0.0000 | 14.829 | 0.0000 |
| 3 | 260.58 | 0.0000 | 1.447 | 0.0000 | 14.461 | 0.0000 |
| 4 | 259.72 | 0.0000 | 1.397 | 0.0000 | 13.941 | 0.0000 |
| 5 | 259.99 | 0.0000 | 1.344 | 0.0000 | 13.412 | 0.0000 |

Raw pseudo-gradient L2 (before any server optimizer runs) is ~260 every
round, for all three — the aggregation step itself is unchanged; only what
happens to it afterward differs. **F1 = 0 in 9 of 10 DP-arm rounds
regardless of which server optimizer is used.**

## Two named findings (Step 18)

**1. The two halves of the optimizer worked against each other.** `v`
(second moment) did exactly what it was hypothesized to do: it learned the
DP noise's stable ~260 magnitude and divided it back out — ~180x damping
for FedAdam, ~18x for FedYogi (the ~10x gap between them tracks their ~10x
`eta_s` difference almost exactly). `m` (first moment) did the opposite of
what was hoped: the residual direction of a *finite*, 5-round noise sample
in 281,254 dimensions is never exactly zero, and momentum (`beta1=0.9`)
applied that same residual direction consistently, every round — compounding
it into an increasingly confident wrong prediction rather than letting it
cancel out. Plain averaging's independent, uncorrelated per-round coin flip
was, paradoxically, less harmful than momentum-concentrated drift.

**2. The 2.5x floor, and it gets worse, not better, over 5 rounds.** Even
after ~180x damping, FedAdam's applied step (~1.3-1.5 L2) never dropped
below 2.5x the genuine no-privacy signal it was competing against (Step 16's
LR-decayed baseline, 0.576 → 0.173 over the same 5 rounds). That ratio
*degraded* across rounds — 2.50x → 3.54x → 4.56x → 6.00x → 7.77x — because
LR decay shrinks the real signal every round while the damped-noise floor
stays roughly flat. Same tension Step 17 found in plain averaging, still
visible one layer inside the damped regime. Whether a ratio near or below
1.0 would be the threshold where utility actually recovers is an **open
question** this measurement did not reach — not a claim.

## Current limitations — read this before anyone asks

- **These three run in the research harness (`scripts/run_step14_multiround.py`),
  not the live Rust orchestrator.** The orchestrator hardcodes
  `"mode": "trimmed_mean"` at `server/orchestration_agent/src/grpc/server.rs:1499`,
  with no config, env var, or request field to change it — none of this
  work touched that file. FedAdam/FedYogi work by letting the live pipeline
  run completely unmodified for every client submission (real gRPC/TPM/mTLS/
  DP/GridFS), then computing the *actual* desired aggregation+optimizer step
  offline and patching the `global_models` MongoDB document so the next
  round's clients warm-start from it correctly. This is a legitimate,
  auditable measurement of what these algorithms do to real client data
  through the real DP/encryption/signing pipeline — but it is not (yet) how
  a live deployment of this system would run one of these algorithms
  automatically.
- **FedAvg here is unweighted mean, equivalent to the textbook algorithm
  (McMahan et al. 2017) only because every client in this project trains on
  an identical shared data partition.** No per-client sample count (`n_k`)
  is transmitted or used anywhere in this codebase. If clients ever have
  different amounts of data, this stops being FedAvg in the textbook sense
  and would need real `n_k/n` weighting added to
  `AggregatorAgent._aggregate_tensor()`.
- **"Cumulative epsilon" in the results table is a naive additive upper
  bound** (`round × 5.302585`), not a tight sequential RDP composition —
  this project has no true multi-round privacy accountant (a known,
  documented gap, not something Step 18 introduced or resolved).

## Not implemented — FedProx, SCAFFOLD, Krum

- **FedProx** — implementable (~40-60 lines, no Rust, following the exact
  `round_id`-threading pattern already used for LR decay) but its benefit
  cannot be validated in this project as built: every client trains on the
  identical shared 149-record corpus, so there is no genuine non-IID client
  divergence for FedProx's proximal term to correct. It would run, but
  demonstrate nothing about what FedProx is for.
- **Krum** — needs `K≥5` clients (`f = floor(K/5)`); at this project's `n=3`,
  `f=0` — zero Byzantine tolerance, a degenerate case, not a meaningful
  demonstration. Even at `K≥5`, every submission in this project comes from
  one honestly-behaving enrolled device training on identical data — there
  is no genuine adversarial or malfunctioning client for Krum to filter.
- **SCAFFOLD** — requires persistent, genuine per-client identity (for its
  control variate `c_k`) and genuine per-client data heterogeneity (for the
  control variates to correct anything). This project has neither: "clients"
  per round are one enrolled device submitting sequentially, all training on
  the same corpus. Building it with synthetic per-session client IDs would
  manufacture a demonstration less honest than not claiming it.

## Project document claim vs. reality

The project document's contribution #2 and Table 13 claim "the first
systematic comparison of five FL optimization algorithms — FedAvg, FedProx,
FedAdam, FedYogi, and SCAFFOLD." As of this document: **two of the five
(FedAdam, FedYogi) are now real, measured implementations** (above); **two
(FedProx, SCAFFOLD) were never implemented**, anywhere in this codebase, live
or in any frozen/historical experiment folder; and **FedAvg exists only as
unweighted mean**, not the sample-weighted textbook version. Of the
document's separately claimed four aggregation strategies (mean,
trimmed_mean, median, Krum), three are implemented and Krum is not.
