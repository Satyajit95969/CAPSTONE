# Federated Learning Algorithm Comparison

Code-grounded reference for this project. Every claim below is anchored to a
specific file/function — nothing here is assumed from "how FL papers usually
work." Where the project's own naming (`trimmed_mean`, `mean`, `median`)
differs from academic naming, both are given.

## What is actually implemented

Source: `server/aggregator_agent/aggregator.py`, class `AggregatorAgent`,
methods `_aggregate_state_dicts()` / `_aggregate_tensor()`.

The aggregator receives one full PyTorch `state_dict` per client (decrypted
from the client's DP-noised, AES-GCM-encrypted delta), validates that every
client shares an identical key set and per-key tensor shape, then aggregates
**key-by-key, coordinate-by-coordinate** across clients. Three modes are
implemented and selectable via `mode=`/`trim_ratio=` on `AggregatorAgent.__init__`:

| Mode (code) | Academic name | Formula (per coordinate, N clients) |
|---|---|---|
| `mean` | Unweighted FedAvg | `mean(x_1, ..., x_N)` |
| `trimmed_mean` (**default**, `trim_ratio=0.1`) | Coordinate-wise Trimmed Mean (Yin et al. 2018, Byzantine-robust distributed learning) | sort the N client values at that coordinate, drop the lowest and highest `trim_ratio*N` values, mean the rest |
| `median` / `coordinate_median` | Coordinate-wise Median | `median(x_1, ..., x_N)` per coordinate |

**Not implemented anywhere in this codebase:** FedAvg's sample-count
weighting (`n_k/n`), FedProx's proximal term, FedAdam/FedYogi/FedAdagrad
server-side adaptive optimizers, SCAFFOLD control variates, or any
personalization/clustering scheme. If you see these terms in an academic
comparison table, they are **ALTERNATIVE/AVAILABLE BUT NOT IMPLEMENTED** —
they don't exist as code in this repo, not "implemented but unused."

## IMPLEMENTED vs CONFIGURED vs CURRENTLY EXECUTED

| Algorithm | Implemented | Configured (server default) | Currently executed in this E2E flow |
|---|---|---|---|
| `mean` (unweighted FedAvg) | YES — `_aggregate_tensor()` | No | No (not selected) |
| `trimmed_mean` (default) | YES | **YES** — hardcoded in `server.rs::run_aggregation()`: `"mode": "trimmed_mean", "trim_ratio": 0.1` | **YES** |
| `coordinate_median` | YES — `_aggregate_tensor()` | No | No (not selected) |

The Rust orchestrator (`server/orchestration_agent/src/grpc/server.rs`,
`run_aggregation()`) builds the aggregation job it sends to the Python
subprocess with `mode` and `trim_ratio` **hardcoded**, not read from any
config file or CLI flag. To actually run `mean` or `median` in this system
today, that hardcoded JSON literal in `server.rs` would need to change —
`AggregatorAgent` itself already supports all three without modification.

## Per-algorithm detail

### `trimmed_mean` — Coordinate-wise Trimmed Mean (currently executed)
- **Objective:** robust estimation of the coordinate mean under a fraction of
  adversarial/outlier client updates, without needing to identify which
  clients are outliers.
- **Client-side behavior:** unchanged from standard FL — each client trains
  locally (`trainer_mentalbert_privacy.py`), computes a state-dict delta,
  clips+DP-noises it (`dp_agent.py`), encrypts it (`enc_agent.py`), and
  uploads. The client does nothing algorithm-specific; robustness is a
  server-side aggregation property only.
- **Server-side behavior:** for each parameter tensor key, flatten to
  `(N, D)`, sort each of the D columns across the N clients, drop the
  lowest/highest `trim_ratio*N` values per column, mean the remainder,
  reshape back. Implemented in pure PyTorch float32 (`_aggregate_tensor()`),
  explicitly avoiding `numpy.mean()`'s float64 promotion.
- **Non-IID handling:** trimmed mean does not adapt to non-IID data
  directly — it protects against outlier *magnitude*, not systematic
  client drift. With highly non-IID clients, some legitimate updates can be
  trimmed away if a coordinate's true update spread happens to be wide. No
  personalization or clustering is implemented to compensate.
- **Communication cost:** identical to plain FedAvg — one full state_dict
  upload per client per round, one full state_dict download per round
  (`_stream_update()` / `_download_global_model()`, 1MB chunks).
- **Computational cost (server):** O(N·D log N) per round for the sort step
  across all D parameters (~110M+ for MentalBERT-scale models) — more
  expensive than `mean`'s O(N·D), cheaper than most Byzantine-robust
  alternatives (e.g. Krum's O(N²·D)).
- **Why used here:** the project's threat model treats clients as
  semi-trusted (untrusted client, per `client_agent.py`'s own docstring:
  "Autonomous UNTRUSTED client"); a single compromised or malfunctioning
  client should not be able to arbitrarily corrupt the global model the way
  an unweighted mean would allow.

### `mean` — Unweighted Federated Averaging (implemented, not executed)
- **Objective:** minimize the average of client loss functions,
  `min_w (1/N) Σ F_k(w)`.
- **Client-side / server-side behavior:** identical client side; server
  simply means each coordinate across all N clients, no sorting/trimming.
- **Note:** this is *not* the textbook FedAvg (McMahan et al. 2017), which
  weights each client by `n_k/n` (its local sample count). No sample counts
  are transmitted or used anywhere in this codebase — every client's
  contribution is weighted equally regardless of how many records it
  trained on.
- **Non-IID handling:** none — most vulnerable of the three to a single bad
  or unrepresentative client, since one outlier update shifts every
  coordinate's mean directly.
- **Communication / computational cost:** cheapest of the three (O(N·D),
  no sort).
- **Why not used as default:** no built-in resistance to a malicious or
  malfunctioning client in an untrusted-client threat model.

### `coordinate_median` — Coordinate-wise Median (implemented, not executed)
- **Objective:** robust estimation of a "typical" client value per
  coordinate; classic Byzantine-robust baseline (Yin et al. 2018).
- **Client-side / server-side behavior:** identical client side; server
  takes the per-coordinate median (`torch.median`) instead of a trimmed mean.
- **Non-IID handling:** most robust of the three to individual outliers
  (breakdown point ~50%), but the median can be a biased estimator of the
  true mean under skewed per-coordinate distributions from non-IID clients.
- **Communication cost:** same as the others.
- **Computational cost:** similar to `trimmed_mean` (also requires a sort
  per coordinate); slightly cheaper since there's no separate trim step.
- **Why not used as default:** trimmed mean was chosen instead, likely as a
  middle ground — it uses more of the client signal (only the extremes are
  discarded, not everything but one point) while retaining robustness.
  This is a design choice visible in the hardcoded default; the code
  comments don't state the reasoning explicitly, so treat this line as
  informed inference, not a documented fact.

## Differential Privacy — not an aggregation algorithm, but interacts with it

DP noise (`dp_agent.py`) is applied **per-client, before upload**, not as
part of aggregation. Mechanism: L2 clip to `clip_norm=1.0`, then additive
noise (Gaussian by default; Laplace/uniform/exponential/Student-t also
implemented), single composition (T=1), RDP→(ε,δ)-DP conversion via the
Mironov 2017 bound. This is orthogonal to which aggregation algorithm runs
server-side — any of the three aggregation modes can consume DP-noised
updates unchanged, since aggregation only sees the already-noised state
dicts.
