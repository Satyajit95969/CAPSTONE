#!/usr/bin/env python3
"""
run_phase22.py - Phase 22 end-to-end multimodal federated demonstration.

Authoritative specification: PHASE_22_DESIGN.md (FROZEN, SHA 3c181083...).

    5 folds x 5 clients = 25 local trainings
    5 folds x 2 arms    = 10 federated executions (training is SHARED by arms)

Per fold:
    seed -> build the ONE shared global init w_before
    partition the fold's TRAINING participants into 5 non-IID A2 clients
    for each client: load w_before -> train -> delta = w_after - w_before
    arm N: aggregate raw deltas
    arm D: clip + Gaussian noise each delta, then aggregate
    global = w_before + aggregated_delta   ->  evaluate on the HELD-OUT fold

PHASE 22 IS AN ENGINEERING DEMONSTRATION - NOT Row 10 / C-1.
Engineering success does NOT imply task success (design SS 2).
Clipping is MEASURED, never tuned (design SS 22.4).
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
from torch.utils.data import DataLoader

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import phase22_common as C  # noqa: E402
from trainer_mentalbert_daic import (  # noqa: E402  ALL FROZEN
    MultiModalModel, MultiModalDataset, collate_batch, compute_state_delta,
    fine_tune_supervised, read_parquet_records)


def float_keys(sd: Dict[str, torch.Tensor]) -> List[str]:
    """Only floating-point tensors receive DP noise.

    The frozen aggregator states non-floating buffers (e.g. int64 position_ids)
    must be byte-identical across clients. Their delta is exactly zero and adding
    Gaussian noise to them would corrupt the reconstructed model, so they are
    passed through untouched.
    """
    return [k for k, v in sd.items() if v.is_floating_point()]


def flatten(delta: Dict[str, torch.Tensor], keys: List[str]) -> torch.Tensor:
    return torch.cat([delta[k].detach().cpu().reshape(-1).float() for k in keys])


def unflatten(flat: torch.Tensor, delta: Dict[str, torch.Tensor],
              keys: List[str]) -> Dict[str, torch.Tensor]:
    out = {k: v.detach().cpu().clone() for k, v in delta.items()}
    off = 0
    for k in keys:
        n = delta[k].numel()
        out[k] = flat[off:off + n].view(delta[k].shape).to(delta[k].dtype)
        off += n
    if off != flat.numel():
        C.abort(f"unflatten consumed {off} of {flat.numel()}")
    return out


def dp_release(delta: Dict[str, torch.Tensor], keys: List[str], seed: int
               ) -> Tuple[Dict[str, torch.Tensor], Dict[str, Any]]:
    """Clip to C then add isotropic N(0,(sigma_eff*C)^2) - design SS 9."""
    flat = flatten(delta, keys)
    l2 = float(torch.norm(flat, p=2))
    scale = min(1.0, C.CLIP_NORM / (l2 + 1e-12))
    clipped = flat * scale
    torch.manual_seed(seed)
    noise = torch.normal(0.0, C.SIGMA_EFF * C.CLIP_NORM, size=clipped.shape)
    noisy = clipped + noise
    if not torch.isfinite(noisy).all():
        C.abort("non-finite value after DP noise")
    stats = {
        "l2_pre_clip": l2, "clip_scale": scale, "clipped": bool(l2 > C.CLIP_NORM),
        "l2_post_clip": float(torch.norm(clipped, p=2)),
        "l2_noise": float(torch.norm(noise, p=2)),
        "nsr": float(torch.norm(noise, p=2)) / max(float(torch.norm(clipped, p=2)), 1e-12),
        "d_noised": int(flat.numel()),
    }
    out = unflatten(noisy, delta, keys)
    del flat, clipped, noise, noisy
    return out, stats


def evaluate(model: MultiModalModel, state: Dict[str, torch.Tensor],
             records: List[Dict[str, Any]], tokenizer, device: str
             ) -> Dict[str, float]:
    model.load_state_dict(state, strict=True)
    model.to(device)
    model.eval()
    ds = MultiModalDataset(records, tokenizer, max_len=C.MAX_LENGTH)
    dl = DataLoader(ds, batch_size=C.BATCH_SIZE, shuffle=False,
                    collate_fn=collate_batch)
    scores, labels = [], []
    with torch.no_grad():
        for b in dl:
            b = {k: (v.to(device) if isinstance(v, torch.Tensor) else v)
                 for k, v in b.items()}
            logits, _, _ = model(b["input_ids"], b["attention_mask"],
                                 audio_vec=b.get("audio_vec"),
                                 vision_vec=b.get("video_vec"))
            scores.extend(torch.softmax(logits, dim=1)[:, 1].cpu().numpy().tolist())
    labels = [C.binarize(r["phq_score"]) for r in records]
    return C.task_metrics(scores, labels)


def main() -> int:
    ap = argparse.ArgumentParser(description="Phase 22 runner")
    ap.add_argument("--out-dir", default=os.path.join(C.REPO, C.OUT_DIRNAME))
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--allow-cpu", action="store_true")
    a = ap.parse_args()
    os.chdir(C.REPO)
    t0 = time.time()

    print("=" * 78)
    print("PHASE 22 - END-TO-END MULTIMODAL FEDERATED DEMONSTRATION")
    print("ENGINEERING DEMONSTRATION - NOT Row 10 / C-1 (design SS 1, SS 23)")
    print("=" * 78)

    if a.device != "cuda" and not a.allow_cpu:
        C.abort("CUDA required (design SS 16). Pass --allow-cpu to override "
                "deliberately; the receipt will record the device used.")

    frozen_before = {p: C.sha256_file(p) for p in C.FROZEN_SHA}
    bad = [p for p, w in C.FROZEN_SHA.items()
           if w is not None and frozen_before[p] != w]
    if bad:
        C.abort(f"frozen input SHA mismatch: {bad}")
    print(f"[OK] frozen inputs verified; design={C.design_sha()}")

    snap = C.hf_snapshot_dir()
    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(snap, local_files_only=True)

    records = read_parquet_records(C.DATA_PARQUET)
    by_id = {int(r["participant_id"]): r for r in records}
    phq = {int(r["participant_id"]): float(r["phq_score"]) for r in records}
    with open(C.FOLD_MANIFEST, "r", encoding="utf-8") as fh:
        folds = C.load_folds(json.load(fh))
    print(f"[OK] {len(records)} records; {len(folds)} frozen folds; "
          f"eps={C.epsilon():.15f} delta={C.DELTA_DP} C={C.CLIP_NORM} T=1")
    print(f"[OK] device={a.device}")

    import importlib.util as iu
    spec = iu.spec_from_file_location("p22agg",
                                      os.path.join(C.REPO, C.AGGREGATOR_PATH))
    am = iu.module_from_spec(spec)
    spec.loader.exec_module(am)
    agg = am.AggregatorAgent(mode=C.AGG_MODE, trim_ratio=C.TRIM_RATIO)
    lo, up, kept = C.trim_window(C.N_CLIENTS)
    print(f"[OK] aggregator {agg.mode} trim_ratio={agg.trim_ratio} "
          f"-> keeps {kept} of {C.N_CLIENTS}")
    print("-" * 78)

    client_rows: List[Dict[str, Any]] = []
    partitions: Dict[str, Any] = {}
    evals: Dict[str, Dict[str, Any]] = {arm: {} for arm in C.ARMS}
    aggr: Dict[str, Any] = {}
    n_trainings = 0

    for fs in folds:
        f = fs["fold"]
        train_ids, test_ids = fs["train_ids"], fs["test_ids"]

        # ---- ONE shared global initialisation for this fold ---------------
        torch.manual_seed(C.fold_seed(f))
        model = MultiModalModel(snap, audio_dim=C.AUDIO_DIM,
                                vision_dim=C.VISION_DIM, device=a.device)
        w_before = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
        n_params = int(sum(v.numel() for v in w_before.values()))
        if n_params != C.MODEL_N_PARAMS or len(w_before) != C.MODEL_N_KEYS:
            C.abort(f"model mismatch: {n_params} params, {len(w_before)} keys")
        fkeys = float_keys(w_before)

        # ---- clients partition ONLY the training participants -------------
        parts = C.a2_partition_training(train_ids, phq)
        st = C.check_partition(parts, set(train_ids), set(test_ids))
        if not (st["disjoint"] and st["coverage_ok"] and st["no_leakage"]
                and st["duplicates"] == 0):
            C.abort(f"fold {f} partition invalid (LEAKAGE RISK): {st}")
        partitions[f"fold{f}"] = {
            "train_n": len(train_ids), "test_n": len(test_ids),
            "sizes": st["sizes"], "disjoint": st["disjoint"],
            "coverage_ok": st["coverage_ok"], "no_test_leakage": st["no_leakage"],
            "clients": [{"client": i + 1, "participant_ids": sorted(p),
                         **C.partition_stats(p, phq)} for i, p in enumerate(parts)],
        }

        raw_deltas: List[Dict[str, torch.Tensor]] = []
        dp_deltas: List[Dict[str, torch.Tensor]] = []

        for ci, members in enumerate(parts, start=1):
            cseed = C.client_seed(f, ci)
            torch.manual_seed(cseed)
            model.load_state_dict(w_before, strict=True)
            model.to(a.device)
            ds = MultiModalDataset([by_id[p] for p in members], tokenizer,
                                   max_len=C.MAX_LENGTH)
            fine_tune_supervised(model, ds, epochs=C.EPOCHS,
                                 batch_size=C.BATCH_SIZE, lr=C.LR,
                                 device=a.device, val_dataset=None)
            w_after = {k: v.detach().cpu().clone()
                       for k, v in model.state_dict().items()}
            delta = compute_state_delta(w_before, w_after)      # FROZEN
            if set(delta) != set(w_before):
                C.abort(f"delta key mismatch fold{f} client{ci}")
            raw_deltas.append(delta)

            noisy, dstats = dp_release(delta, fkeys, C.dp_seed(f, ci))
            dp_deltas.append(noisy)
            n_trainings += 1
            client_rows.append({
                "fold": f, "client": ci, "n_train": len(members),
                "client_seed": cseed, "dp_seed": C.dp_seed(f, ci),
                **C.partition_stats(members, phq), **dstats,
                "epsilon": C.EPS_PER_UPDATE, "delta_dp": C.DELTA_DP,
            })
            del w_after

        # ---- aggregate, reconstruct, evaluate - both arms ------------------
        for arm, deltas in (("N", raw_deltas), ("D", dp_deltas)):
            agg_delta = agg._aggregate_state_dicts(deltas)       # FROZEN
            global_state = {k: (w_before[k] + agg_delta[k]).to(w_before[k].dtype)
                            if w_before[k].is_floating_point() else w_before[k]
                            for k in w_before}
            m = C.mean_sd([float(torch.norm(agg_delta[k].float(), p=2))
                           for k in fkeys])
            aggr[f"{arm}|fold{f}"] = {
                "mode": C.AGG_MODE, "trim_ratio": C.TRIM_RATIO,
                "lower": lo, "upper": up, "kept_of_n": f"{kept} of {C.N_CLIENTS}",
                "agg_delta_l2_mean_per_tensor": m[0],
            }
            evals[arm][f"fold{f}"] = evaluate(model, global_state,
                                              [by_id[p] for p in test_ids],
                                              tokenizer, a.device)
            del agg_delta, global_state

        print(f"  [fold {f}] sizes={st['sizes']}  "
              f"N roc_auc={evals['N'][f'fold{f}']['roc_auc']:.6f}  "
              f"D roc_auc={evals['D'][f'fold{f}']['roc_auc']:.6f}  "
              f"clipped={sum(1 for r in client_rows[-5:] if r['clipped'])}/5")

        del raw_deltas, dp_deltas, w_before, model
        if a.device == "cuda":
            torch.cuda.empty_cache()

    if n_trainings != C.N_FOLDS * C.N_CLIENTS:
        C.abort(f"{n_trainings} trainings, expected {C.N_FOLDS * C.N_CLIENTS}")

    frozen_after = {p: C.sha256_file(p) for p in C.FROZEN_SHA}
    changed = [p for p in frozen_before if frozen_before[p] != frozen_after[p]]
    if changed:
        C.abort(f"frozen artifacts modified DURING the run: {changed}")
    print("-" * 78)
    print(f"[OK] {n_trainings} local trainings; frozen artifacts unchanged")

    # ---- artifacts ---------------------------------------------------------
    os.makedirs(a.out_dir, exist_ok=True)
    cols = ["fold", "client", "n_train", "n", "pos", "neg", "pos_rate", "phq_mean",
            "phq_min", "phq_max", "phq_sd", "client_seed", "dp_seed",
            "l2_pre_clip", "clip_scale", "clipped", "l2_post_clip", "l2_noise",
            "nsr", "d_noised", "epsilon", "delta_dp"]
    with open(os.path.join(a.out_dir, "phase22_clients.csv"), "w", newline="",
              encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=cols)
        w.writeheader()
        for r in client_rows:
            w.writerow({c: r.get(c) for c in cols})

    pre = [r["l2_pre_clip"] for r in client_rows]
    n_clipped = sum(1 for r in client_rows if r["clipped"])
    dp_block = {
        "mechanism": C.MECHANISM, "clip_norm": C.CLIP_NORM,
        "noise_multiplier": C.NOISE_MULTIPLIER, "sigma_eff": C.SIGMA_EFF,
        "epsilon_per_update": C.epsilon(), "delta": C.DELTA_DP,
        "composition": C.COMPOSITION, "round_epsilon": C.EPS_PER_UPDATE,
        "round_epsilon_basis": C.ROUND_EPS_BASIS,
        "n_updates": len(client_rows), "n_clipped": n_clipped,
        "fraction_clipped": n_clipped / len(client_rows),
        "pre_clip_norm_mean": float(np.mean(pre)),
        "pre_clip_norm_min": float(np.min(pre)),
        "pre_clip_norm_max": float(np.max(pre)),
        "measured_nsr_mean": float(np.mean([r["nsr"] for r in client_rows])),
        "d_noised": client_rows[0]["d_noised"],
        "expected_noise_norm_sqrt_d": float(np.sqrt(client_rows[0]["d_noised"])),
        "note": "clipping measured, never tuned (design SS 22.4)",
    }

    for name, obj in (("phase22_partitions.json", partitions),
                      ("phase22_dp.json", dp_block),
                      ("phase22_aggregation.json", aggr),
                      ("phase22_eval.json", evals)):
        with open(os.path.join(a.out_dir, name), "w", encoding="utf-8") as fh:
            json.dump({"schema": name.replace(".json", "/1"),
                       "design": {"path": C.DESIGN_PATH, "sha256": C.design_sha()},
                       "data": obj}, fh, indent=2, sort_keys=True, default=float)

    elapsed = round(time.time() - t0, 2)
    with open(os.path.join(a.out_dir, "phase22_receipt.json"), "w",
              encoding="utf-8") as fh:
        json.dump({
            "schema": "phase22_receipt/1", "experiment": C.EXPERIMENT,
            "design": {"path": C.DESIGN_PATH, "sha256": C.design_sha()},
            "generated_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            "frozen_input_sha": {"pre_run": frozen_before, "post_run": frozen_after,
                                 "unchanged": True},
            "protocol": {"folds": C.N_FOLDS, "clients": C.N_CLIENTS,
                         "repeats": C.N_REPEATS, "rounds": C.N_ROUNDS,
                         "local_trainings": n_trainings,
                         "federated_executions": C.N_FOLDS * len(C.ARMS),
                         "arms": list(C.ARMS), "arm_labels": C.ARM_LABEL},
            "model": {"class": "MultiModalModel", "text_dim": C.TEXT_DIM,
                      "audio_dim": C.AUDIO_DIM, "vision_dim": C.VISION_DIM,
                      "fusion_dim": C.FUSION_DIM, "n_params": C.MODEL_N_PARAMS,
                      "n_keys": C.MODEL_N_KEYS, "revision": C.BERT_REVISION},
            "recipe": {"optimizer": C.OPTIMIZER, "lr": C.LR, "epochs": C.EPOCHS,
                       "batch_size": C.BATCH_SIZE, "grad_clip": C.GRAD_CLIP,
                       "dropout": C.DROPOUT, "loss": C.LOSS},
            "transmitted_object": "genuine delta w_after - w_before "
                                  "(frozen compute_state_delta)",
            "dp": dp_block,
            "aggregation": {"mode": C.AGG_MODE, "trim_ratio": C.TRIM_RATIO,
                            "kept_of_n": f"{kept} of {C.N_CLIENTS}"},
            "environment": {"python": sys.version.split()[0],
                            "torch": torch.__version__,
                            "cuda_available": bool(torch.cuda.is_available()),
                            "device_used": a.device,
                            "gpu": torch.cuda.get_device_name(0)
                            if torch.cuda.is_available() else "NONE"},
            "scope_caveat": C.SCOPE_CAVEAT, "runtime_seconds": elapsed,
        }, fh, indent=2, sort_keys=True, default=float)

    print(f"[DONE] {n_trainings} trainings + {C.N_FOLDS*len(C.ARMS)} federated "
          f"executions in {elapsed:.1f}s -> {a.out_dir}")
    print("Next: aggregate_phase22.py, then verify_phase22.py")
    return 0


if __name__ == "__main__":
    sys.exit(main())
