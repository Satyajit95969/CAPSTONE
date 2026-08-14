#!/usr/bin/env python3
"""
run_c2.py - Phase 20 / C-2 experiment runner (Modules B-H).

Authoritative specification: PHASE_20_C2_DESIGN.md (FROZEN). Every constant comes
from c2_common.py, which transcribes that document. Nothing here may be tuned.

    5 repeats x 2 arms x 5 clients = 50 client trainings

For every client:
    w_before (frozen local_probe_base.pt, shared initialisation)
      -> train locally on that client's partition
      -> w_after
      -> delta = compute_state_delta(w_before, w_after)      [FROZEN convention]
      -> record PRE-DP divergence
      -> DP (Gaussian, clip 1.0, nm 1.0, T=1)
      -> record POST-DP divergence
      -> aggregate with the REAL frozen AggregatorAgent (trimmed_mean, 0.1)

NO TASK METRIC IS COMPUTED (design SS 14). There is no ROC-AUC, accuracy, F1,
MAE, RMSE or PR-AUC anywhere in this file, and no held-out evaluation set exists.
C-2 is an aggregation/manipulation experiment (design SS 21).

SEEDING (design SS 13) - identical in both arms:
    repeat_seed = 1000 + repeat
    client_seed = repeat_seed * 100 + client_index
    dp_seed     = 1000 + repeat, re-seeded at the DP stage of EACH arm so that
                  corresponding clients receive identical noise across arms

Usage:
    python exp_c2_multiclient/run_c2.py
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import sys
import time
from typing import Any, Dict, List, Tuple

import numpy as np
import torch
import torch.nn as nn

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import c2_common as C  # noqa: E402
import trainer_mentalbert_daic as T  # noqa: E402  FROZEN - compute_state_delta only


class ProbeModel(nn.Module):
    """768 -> 384 -> 1, matching local_probe_base.pt exactly (design SS 8, SS 9).

    Key names fc1.weight / fc1.bias / fc2.weight / fc2.bias are identical to the
    frozen probe, so load_state_dict(strict=True) succeeds and the delta keys
    match by construction.
    """

    def __init__(self) -> None:
        super().__init__()
        self.fc1 = nn.Linear(C.EMBED_DIM, 384)
        self.fc2 = nn.Linear(384, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.fc2(torch.relu(self.fc1(x)))


def flatten_delta(delta: Dict[str, torch.Tensor]) -> np.ndarray:
    """Flatten in the frozen probe key order (design SS 8)."""
    return np.concatenate([
        delta[k].detach().cpu().numpy().astype(np.float64).ravel()
        for k in C.PROBE_TENSORS])


def train_client(w_before: Dict[str, torch.Tensor], X: np.ndarray, y: np.ndarray,
                 seed: int) -> Tuple[Dict[str, torch.Tensor], float]:
    """One client, one local training run (design SS 9).

    MSELoss, Adam, lr 1e-3, 1 epoch, FULL BATCH, grad clip 1.0.

    Determinism note: with full-batch training, no dropout and no shuffling, the
    run is deterministic given w_before and the client's data. The seed governs
    the parameter initialisation, which is then immediately overwritten by
    w_before via load_state_dict(strict=True). It is set and recorded because
    design SS 13 requires a declared, deterministic seed hierarchy.
    """
    torch.manual_seed(seed)
    model = ProbeModel()
    model.load_state_dict(w_before, strict=True)       # design SS 8, SS 16.5
    model.train()

    xb = torch.tensor(X, dtype=torch.float32)
    yb = torch.tensor(y, dtype=torch.float32).view(-1, 1)

    optim = torch.optim.Adam(model.parameters(), lr=C.LR)
    criterion = nn.MSELoss()
    loss_val = float("nan")
    for _ in range(C.EPOCHS):
        optim.zero_grad()
        out = model(xb)
        loss = criterion(out, yb)
        if not torch.isfinite(loss):
            C.abort(f"non-finite loss during client training (seed={seed})")
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), C.GRAD_CLIP)
        optim.step()
        loss_val = float(loss.item())

    w_after = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
    return w_after, loss_val


def dp_release(flat: np.ndarray) -> Dict[str, Any]:
    """Clip to C = 1.0 then add isotropic N(0, sigma^2) (design SS 11).

    The caller seeds the global torch RNG immediately before the per-arm DP
    stage, so corresponding clients in W0 and W1 receive identical noise.
    """
    v = torch.tensor(flat, dtype=torch.float32)
    l2_before = float(torch.norm(v, p=2))
    scale = min(1.0, C.CLIP_NORM / (l2_before + 1e-12))
    clipped = v * scale
    sigma = C.SIGMA_EFF * C.CLIP_NORM                 # = 1.0
    noise = torch.normal(0.0, sigma, size=clipped.shape)
    noisy = clipped + noise
    if not torch.isfinite(noisy).all():
        C.abort("non-finite value after DP noise")
    return {
        "l2_before": l2_before,
        "clip_scale": scale,
        "clipped": clipped.numpy().astype(np.float64),
        "l2_after_clip": float(torch.norm(clipped, p=2)),
        "l2_noise": float(torch.norm(noise, p=2)),
        "noisy": noisy.numpy().astype(np.float64),
        "l2_after": float(torch.norm(noisy, p=2)),
    }


def aggregate(noisy_flats: List[np.ndarray], agg) -> Dict[str, Any]:
    """Aggregate with the REAL frozen AggregatorAgent (design SS 12).

    Also records the trim bookkeeping the design requires: per-coordinate trimmed
    client counts, and the distance of the trimmed-mean aggregate from the
    ordinary mean and from the coordinate median.
    """
    stacked = torch.tensor(np.stack(noisy_flats), dtype=torch.float32)
    trimmed = agg._aggregate_tensor(stacked)          # frozen trimming logic
    trimmed_np = trimmed.numpy().astype(np.float64)

    arr = np.stack(noisy_flats)                       # (n_clients, d)
    lower, upper, kept = C.trim_window(arr.shape[0])
    order = np.argsort(arr, axis=0, kind="stable")    # ranks along client axis
    trimmed_mask = np.zeros_like(arr, dtype=bool)
    for pos in list(range(0, lower)) + list(range(upper, arr.shape[0])):
        trimmed_mask[order[pos], np.arange(arr.shape[1])] = True
    trim_counts = trimmed_mask.sum(axis=1).astype(int).tolist()

    ordinary_mean = arr.mean(axis=0)
    coord_median = np.median(arr, axis=0)
    return {
        "trimmed_mean": trimmed_np,
        "trim_window": {"lower": lower, "upper": upper, "kept": kept,
                        "n_clients": int(arr.shape[0])},
        "trim_counts_per_client": trim_counts,
        "trim_count_total": int(trimmed_mask.sum()),
        "coordinates": int(arr.shape[1]),
        "l2_trimmed_vs_ordinary_mean": C.l2_distance(trimmed_np, ordinary_mean),
        "cos_trimmed_vs_ordinary_mean": C.cosine_distance(trimmed_np, ordinary_mean),
        "l2_trimmed_vs_coord_median": C.l2_distance(trimmed_np, coord_median),
        "cos_trimmed_vs_coord_median": C.cosine_distance(trimmed_np, coord_median),
        "l2_trimmed_mean_norm": float(np.linalg.norm(trimmed_np)),
    }


def main() -> int:
    ap = argparse.ArgumentParser(description="Phase 20 / C-2 experiment runner")
    ap.add_argument("--out-dir", default=os.path.join(C.REPO, C.OUT_DIRNAME))
    a = ap.parse_args()
    os.chdir(C.REPO)
    t0 = time.time()

    print("=" * 78)
    print("C-2 - NON-IID MULTI-CLIENT TOPOLOGY (roadmap Row 11)")
    print("single factor: client partition. NO task metric (design SS 14).")
    print("=" * 78)

    # ---- frozen gate, in the execution path (design SS 16.3) --------------
    frozen_before = {p: C.sha256_file(p) for p in C.FROZEN_SHA}
    bad = [p for p, w in C.FROZEN_SHA.items()
           if w is not None and frozen_before[p] != w]
    if bad:
        C.abort(f"frozen input SHA mismatch before execution: {bad}")
    print(f"[OK] frozen inputs verified; design={C.design_sha()}")

    # ---- embeddings -------------------------------------------------------
    npz = os.path.join(a.out_dir, "c2_embeddings.npz")
    if not os.path.isfile(npz):
        C.abort(f"embedding artifact missing: {npz} (run extract_c2_embeddings.py)")
    z = np.load(npz, allow_pickle=False)
    ids = [int(x) for x in z["participant_id"]]
    E = z["embedding"].astype(np.float64)
    Y = z["phq_score"].astype(np.float64)
    idx = {p: i for i, p in enumerate(ids)}
    phq = {p: float(Y[i]) for p, i in idx.items()}
    if E.shape != (C.N_PARTICIPANTS, C.EMBED_DIM):
        C.abort(f"embedding shape {E.shape} != ({C.N_PARTICIPANTS}, {C.EMBED_DIM})")
    print(f"[OK] embeddings {E.shape} sha={C.sha256_file(npz)[:32]}...")

    # ---- partitions -------------------------------------------------------
    with open(C.FOLD_MANIFEST, "r", encoding="utf-8") as fh:
        manifest = json.load(fh)
    parts = {C.CONTROL_ARM: C.control_partition(manifest),
             C.TREATMENT_ARM: C.a2_partition([(p, phq[p]) for p in ids])}
    expect = set(ids)
    for arm, pl in parts.items():
        st = C.check_partition(pl, expect)
        if not (st["disjoint"] and st["coverage_ok"] and st["sizes_ok"]
                and st["duplicates"] == 0):
            C.abort(f"{arm} partition invalid: {st}")
        print(f"[OK] {arm}: sizes={st['sizes']} disjoint={st['disjoint']} "
              f"coverage=188/188  ({C.ARM_LABEL[arm]})")
    if {i for p in parts[C.CONTROL_ARM] for i in p} != \
       {i for p in parts[C.TREATMENT_ARM] for i in p}:
        C.abort("arm population mismatch (design SS 16.2)")
    print("[OK] both arms cover the same 188 participants")

    # ---- shared initialisation -------------------------------------------
    w_before = torch.load(C.PROBE_PATH, map_location="cpu", weights_only=False)
    w_before = {k: v.detach().cpu().clone() for k, v in w_before.items()}
    n_params = int(sum(v.numel() for v in w_before.values()))
    if n_params != C.PROBE_N_PARAMS or set(w_before) != set(C.PROBE_TENSORS):
        C.abort(f"w_before mismatch: {n_params} params, keys {sorted(w_before)}")
    print(f"[OK] shared initialisation w_before: {n_params:,} params, "
          f"{len(w_before)} tensors")

    # ---- frozen aggregator -------------------------------------------------
    import importlib.util as iu
    spec = iu.spec_from_file_location("c2_agg",
                                      os.path.join(C.REPO, C.AGGREGATOR_PATH))
    agg_mod = iu.module_from_spec(spec)
    spec.loader.exec_module(agg_mod)
    agg = agg_mod.AggregatorAgent(mode=C.AGG_MODE, trim_ratio=C.TRIM_RATIO)
    lo, up, kept = C.trim_window(C.N_CLIENTS)
    print(f"[OK] frozen AggregatorAgent mode={agg.mode} trim_ratio={agg.trim_ratio} "
          f"-> keeps {kept} of {C.N_CLIENTS}")

    eps = C.epsilon()
    if abs(eps - C.EPS_PER_UPDATE) > 1e-12:
        C.abort(f"recomputed eps {eps!r} != frozen {C.EPS_PER_UPDATE}")
    print(f"[OK] eps = {eps:.15f}  delta = {C.DELTA_DP}  clip = {C.CLIP_NORM}  "
          f"nm = {C.NOISE_MULTIPLIER}  T = 1")
    print("=" * 78)

    client_rows: List[Dict[str, Any]] = []
    div_pre: Dict[str, Any] = {}
    div_post: Dict[str, Any] = {}
    agg_out: Dict[str, Any] = {}
    seed_manifest: List[Dict[str, Any]] = []
    n_trainings = 0

    for repeat in C.REPEATS:
        for arm in C.ARMS:
            pre_flats: List[np.ndarray] = []
            post_flats: List[np.ndarray] = []
            per_client: List[Dict[str, Any]] = []

            # ---- local training (deltas are arm/repeat specific) -----------
            for ci, members in enumerate(parts[arm], start=1):
                cseed = C.client_seed(repeat, ci)
                rows = [idx[p] for p in members]
                Xc, Yc = E[rows], Y[rows]
                w_after, loss = train_client(w_before, Xc, Yc, cseed)
                delta = T.compute_state_delta(w_before, w_after)   # FROZEN
                if set(delta) != set(w_before):
                    C.abort(f"delta key mismatch r{repeat} {arm} c{ci}")
                for kk in delta:
                    if tuple(delta[kk].shape) != tuple(w_before[kk].shape):
                        C.abort(f"delta shape mismatch at {kk}")
                flat = flatten_delta(delta)
                if flat.size != C.PROBE_N_PARAMS or not np.isfinite(flat).all():
                    C.abort(f"bad delta r{repeat} {arm} c{ci}: size={flat.size}")
                pre_flats.append(flat)
                per_client.append({"client": ci, "n": len(members),
                                   "client_seed": cseed, "final_loss": loss,
                                   "delta_l2": float(np.linalg.norm(flat)),
                                   **C.partition_stats(members, phq)})
                n_trainings += 1
                seed_manifest.append({"repeat": repeat, "arm": arm, "client": ci,
                                      "repeat_seed": C.repeat_seed(repeat),
                                      "client_seed": cseed,
                                      "dp_seed": C.dp_seed(repeat)})

            # ---- PRE-DP divergence: the PRIMARY measurement ---------------
            div_pre[f"{arm}|r{repeat}"] = C.pairwise_divergence(pre_flats)

            # ---- DP: re-seeded per (repeat, arm) so arms match -------------
            torch.manual_seed(C.dp_seed(repeat))
            for ci in range(C.N_CLIENTS):
                rel = dp_release(pre_flats[ci])
                post_flats.append(rel["noisy"])
                per_client[ci].update({
                    "dp_seed": C.dp_seed(repeat),
                    "l2_before_dp": rel["l2_before"],
                    "clip_scale": rel["clip_scale"],
                    "l2_after_clip": rel["l2_after_clip"],
                    "l2_noise": rel["l2_noise"],
                    "l2_after_dp": rel["l2_after"],
                    "epsilon": eps, "delta_dp": C.DELTA_DP,
                    "clip_norm": C.CLIP_NORM,
                    "noise_multiplier": C.NOISE_MULTIPLIER,
                    "mechanism": C.MECHANISM,
                })

            div_post[f"{arm}|r{repeat}"] = C.pairwise_divergence(post_flats)
            agg_out[f"{arm}|r{repeat}"] = aggregate(post_flats, agg)

            for row in per_client:
                client_rows.append({"repeat": repeat, "arm": arm, **row})

            dp = div_pre[f"{arm}|r{repeat}"]
            print(f"  [r{repeat} {arm}] pre-DP L2 mean={dp['l2_mean']:.6e} "
                  f"cos mean={dp['cosine_mean']:.6e} | "
                  f"post-DP L2 mean={div_post[f'{arm}|r{repeat}']['l2_mean']:.4f} | "
                  f"trims={agg_out[f'{arm}|r{repeat}']['trim_counts_per_client']}")

    if n_trainings != C.N_REPEATS * len(C.ARMS) * C.N_CLIENTS:
        C.abort(f"{n_trainings} client trainings, expected "
                f"{C.N_REPEATS * len(C.ARMS) * C.N_CLIENTS}")

    # ---- post-run frozen gate ---------------------------------------------
    frozen_after = {p: C.sha256_file(p) for p in C.FROZEN_SHA}
    changed = [p for p in frozen_before if frozen_before[p] != frozen_after[p]]
    if changed:
        C.abort(f"frozen artifacts modified DURING the run: {changed}")
    print("-" * 78)
    print(f"[OK] {n_trainings} client trainings; frozen artifacts unchanged")

    # ---- artifacts ---------------------------------------------------------
    os.makedirs(a.out_dir, exist_ok=True)

    cols = ["repeat", "arm", "client", "n", "pos", "neg", "pos_rate", "phq_mean",
            "phq_median", "phq_min", "phq_max", "phq_sd", "client_seed", "dp_seed",
            "final_loss", "delta_l2", "l2_before_dp", "clip_scale",
            "l2_after_clip", "l2_noise", "l2_after_dp", "epsilon", "delta_dp",
            "clip_norm", "noise_multiplier", "mechanism"]
    cpath = os.path.join(a.out_dir, "c2_client_updates.csv")
    with open(cpath, "w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=cols)
        w.writeheader()
        for r in client_rows:
            w.writerow({c: r.get(c) for c in cols})

    ppath = os.path.join(a.out_dir, "c2_partitions.json")
    with open(ppath, "w", encoding="utf-8") as fh:
        json.dump({"schema": "c2_partitions/1",
                   "design": {"path": C.DESIGN_PATH, "sha256": C.design_sha()},
                   "arms": {arm: {"label": C.ARM_LABEL[arm],
                                  "clients": [{"client": i + 1,
                                               "participant_ids": sorted(m),
                                               **C.partition_stats(m, phq)}
                                              for i, m in enumerate(pl)]}
                            for arm, pl in parts.items()}}, fh,
                  indent=2, sort_keys=True, default=float)

    spath = os.path.join(a.out_dir, "c2_seed_manifest.json")
    with open(spath, "w", encoding="utf-8") as fh:
        json.dump({"schema": "c2_seed_manifest/1",
                   "rule": {"repeat_seed": "1000 + repeat",
                            "client_seed": "repeat_seed * 100 + client_index",
                            "dp_seed": "1000 + repeat, re-seeded per (repeat, arm) "
                                       "so corresponding clients receive identical "
                                       "noise in both arms"},
                   "entries": seed_manifest}, fh, indent=2, sort_keys=True)

    for name, obj in (("c2_divergence_pre_dp.json", div_pre),
                      ("c2_divergence_post_dp.json", div_post)):
        with open(os.path.join(a.out_dir, name), "w", encoding="utf-8") as fh:
            json.dump({"schema": name.replace(".json", "/1"),
                       "design": {"path": C.DESIGN_PATH, "sha256": C.design_sha()},
                       "measures": ["l2", "cosine"],
                       "note": "pre-DP is the PRIMARY manipulation check "
                               "(design SS 14); post-DP is secondary because DP "
                               "noise dominates it",
                       "by_arm_repeat": obj}, fh, indent=2, sort_keys=True,
                      default=float)

    apath = os.path.join(a.out_dir, "c2_aggregation.json")
    with open(apath, "w", encoding="utf-8") as fh:
        json.dump({"schema": "c2_aggregation/1",
                   "design": {"path": C.DESIGN_PATH, "sha256": C.design_sha()},
                   "aggregator": {"module": C.AGGREGATOR_PATH,
                                  "mode": C.AGG_MODE, "trim_ratio": C.TRIM_RATIO,
                                  "lower": lo, "upper": up, "kept": kept},
                   "by_arm_repeat": {k: {kk: vv for kk, vv in v.items()
                                         if kk != "trimmed_mean"}
                                     for k, v in agg_out.items()}}, fh,
                  indent=2, sort_keys=True, default=float)

    elapsed = round(time.time() - t0, 2)
    rpath = os.path.join(a.out_dir, "c2_receipt.json")
    with open(rpath, "w", encoding="utf-8") as fh:
        json.dump({
            "schema": "c2_receipt/1", "experiment": C.EXPERIMENT,
            "design": {"path": C.DESIGN_PATH, "sha256": C.design_sha()},
            "generated_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            "frozen_input_sha": {"pre_run": frozen_before, "post_run": frozen_after,
                                 "unchanged": True},
            "protocol": {"n_repeats": C.N_REPEATS, "n_arms": len(C.ARMS),
                         "n_clients": C.N_CLIENTS, "client_trainings": n_trainings,
                         "client_sizes": list(C.CLIENT_SIZES)},
            "representation": {"model": C.BERT_MODEL, "revision": C.BERT_REVISION,
                               "max_length": C.MAX_LENGTH, "padding": C.PADDING,
                               "pooling": C.POOLING, "embed_dim": C.EMBED_DIM,
                               "embeddings_sha256": C.sha256_file(npz)},
            "shared_initialisation": {"path": C.PROBE_PATH,
                                      "sha256": C.sha256_file(C.PROBE_PATH),
                                      "n_params": n_params,
                                      "role": "w_before ONLY - not the payload"},
            "recipe": {"loss": C.LOSS, "optimizer": C.OPTIMIZER, "lr": C.LR,
                       "epochs": C.EPOCHS, "batch": C.BATCH,
                       "grad_clip": C.GRAD_CLIP, "target": C.TARGET},
            "dp": {"mechanism": C.MECHANISM, "epsilon": eps, "delta": C.DELTA_DP,
                   "clip_norm": C.CLIP_NORM,
                   "noise_multiplier": C.NOISE_MULTIPLIER,
                   "composition": C.COMPOSITION, "applied": True},
            "aggregation": {"mode": C.AGG_MODE, "trim_ratio": C.TRIM_RATIO,
                            "kept_of_n": f"{kept} of {C.N_CLIENTS}"},
            "integrity": {"checkpoint_reuse": False,
                          "shared_init_identical_all_clients": True,
                          "arms_same_population": True,
                          "clients_disjoint": True},
            "task_metric_computed": False,
            "scope_caveat": C.SCOPE_CAVEAT,
            "environment": {"python": sys.version.split()[0],
                            "torch": torch.__version__, "platform": sys.platform},
            "runtime_seconds": elapsed,
        }, fh, indent=2, sort_keys=True, default=float)

    print(f"[DONE] {n_trainings} trainings in {elapsed:.1f}s -> {a.out_dir}")
    print("Next: aggregate_c2.py (validity gate), then verify_c2.py")
    return 0


if __name__ == "__main__":
    sys.exit(main())
