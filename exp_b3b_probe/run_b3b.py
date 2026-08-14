#!/usr/bin/env python3
"""
run_b3b.py - Phase 21 / B-3B experiment runner.

Authoritative specification: PHASE_21_B3B_DESIGN.md (FROZEN).

    5 repeats x 5 folds = 25 trainings; each yields BOTH arms.

Per (repeat, fold):
    torch.manual_seed(fold_seed)            <- BEFORE model construction (SS 11)
    w_before = FusionHead(768, 256, 2)      <- seeded random init
    train 3 epochs, AdamW lr=1e-3, batch 8, shuffle, CE + 0.5*MSE, clip 1.0
    delta = compute_state_delta(w_before, w_after)        <- FROZEN
    arm N : evaluate w_before + delta                     (no DP)
    arm D : evaluate w_before + DP(delta)                 (clip 1.0 + N(0,1))

Both arms come from the SAME training run, so they are identical in every
respect except DP application (design SS 9).

Clipping is MEASURED, NEVER TUNED (design SS 13). H0 is a legitimate result.
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
from torch.utils.data import DataLoader, TensorDataset

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import b3b_common as C  # noqa: E402
from trainer_mentalbert_daic import FusionHead, compute_state_delta  # noqa: E402 FROZEN
from transformers import AdamW  # noqa: E402


def flatten(delta: Dict[str, torch.Tensor]) -> np.ndarray:
    return np.concatenate([delta[k].detach().cpu().numpy().astype(np.float64).ravel()
                           for k in C.HEAD_TENSORS])


def unflatten(vec: np.ndarray) -> Dict[str, torch.Tensor]:
    out, off = {}, 0
    for k, shape in C.HEAD_TENSORS.items():
        n = int(np.prod(shape))
        out[k] = torch.tensor(vec[off:off + n], dtype=torch.float32).reshape(shape)
        off += n
    if off != C.HEAD_N_PARAMS:
        C.abort(f"unflatten consumed {off} of {C.HEAD_N_PARAMS}")
    return out


def train_one(X: np.ndarray, y: np.ndarray, phq: np.ndarray, seed: int
              ) -> Tuple[Dict[str, torch.Tensor], Dict[str, torch.Tensor], float]:
    """One (repeat, fold) training run under the frozen recipe (design SS 6)."""
    torch.manual_seed(seed)                       # SS 11: BEFORE model construction
    head = FusionHead(in_dim=C.IN_DIM, hidden=C.HIDDEN, num_classes=C.NUM_CLASSES)
    w_before = {k: v.detach().cpu().clone() for k, v in head.state_dict().items()}

    ds = TensorDataset(torch.tensor(X, dtype=torch.float32),
                       torch.tensor(y, dtype=torch.long),
                       torch.tensor(phq, dtype=torch.float32))
    loader = DataLoader(ds, batch_size=C.BATCH_SIZE, shuffle=C.SHUFFLE)

    opt = AdamW(head.parameters(), lr=C.LR)
    cls_loss_fn = nn.CrossEntropyLoss()
    reg_loss_fn = nn.MSELoss()
    last = float("nan")
    for _ in range(C.EPOCHS):
        head.train()
        tot, steps = 0.0, 0
        for xb, yb, pb in loader:
            opt.zero_grad()
            logits, mu, _ = head(xb)
            loss = cls_loss_fn(logits, yb) + C.LAMBDA_REG * reg_loss_fn(mu, pb)
            if not torch.isfinite(loss):
                C.abort(f"non-finite loss (seed={seed})")
            loss.backward()
            torch.nn.utils.clip_grad_norm_(head.parameters(), C.GRAD_CLIP)
            opt.step()
            tot += float(loss.item()); steps += 1
        last = tot / steps if steps else float("nan")

    w_after = {k: v.detach().cpu().clone() for k, v in head.state_dict().items()}
    return w_before, w_after, last


def evaluate(state: Dict[str, torch.Tensor], X: np.ndarray, y: np.ndarray) -> float:
    """ROC-AUC from the positive-class probability (design SS 5, SS 11)."""
    head = FusionHead(in_dim=C.IN_DIM, hidden=C.HIDDEN, num_classes=C.NUM_CLASSES)
    head.load_state_dict(state, strict=True)
    head.eval()
    with torch.no_grad():
        logits, _, _ = head(torch.tensor(X, dtype=torch.float32))
        p_pos = torch.softmax(logits, dim=1)[:, 1].cpu().numpy()
    return C.roc_auc(p_pos, y)


def dp_release(flat: np.ndarray, seed: int) -> Dict[str, Any]:
    """Clip to C then add isotropic N(0, sigma_eff^2) (design SS 10)."""
    v = torch.tensor(flat, dtype=torch.float32)
    l2 = float(torch.norm(v, p=2))
    scale = min(1.0, C.CLIP_NORM / (l2 + 1e-12))
    clipped = v * scale
    g = torch.Generator().manual_seed(seed)       # deterministic noise draw
    noise = torch.normal(0.0, C.SIGMA_EFF * C.CLIP_NORM, size=clipped.shape,
                         generator=g)
    noisy = clipped + noise
    if not torch.isfinite(noisy).all():
        C.abort("non-finite value after DP noise")
    l2c = float(torch.norm(clipped, p=2))
    l2n = float(torch.norm(noise, p=2))
    return {"l2_pre_clip": l2, "clip_scale": scale, "clipped": l2 > C.CLIP_NORM,
            "l2_post_clip": l2c, "l2_noise": l2n,
            "nsr": l2n / l2c if l2c > 0 else float("inf"),
            "noisy": noisy.numpy().astype(np.float64)}


def main() -> int:
    ap = argparse.ArgumentParser(description="Phase 21 / B-3B runner")
    ap.add_argument("--out-dir", default=os.path.join(C.REPO, C.OUT_DIRNAME))
    a = ap.parse_args()
    os.chdir(C.REPO)
    t0 = time.time()

    print("=" * 78)
    print("B-3B - COMPACT TASK-TRAINED PROBE (roadmap Row 9, SECOND attempt)")
    print("H0 is a legitimate result. Clipping is measured, never tuned.")
    print("=" * 78)

    frozen_before = {p: C.sha256_file(p) for p in C.FROZEN_SHA}
    bad = [p for p, w in C.FROZEN_SHA.items()
           if w is not None and frozen_before[p] != w]
    if bad:
        C.abort(f"frozen input SHA mismatch: {bad}")
    print(f"[OK] frozen inputs verified; design={C.design_sha()}")

    z = np.load(C.EMBEDDINGS_PATH, allow_pickle=False)
    ids = [int(x) for x in z["participant_id"]]
    E = z["embedding"].astype(np.float64)
    PHQ = z["phq_score"].astype(np.float64)
    idx = {p: i for i, p in enumerate(ids)}
    Y = np.array([C.binarize(v) for v in PHQ], dtype=np.int64)
    print(f"[OK] embeddings {E.shape} (reused C-2 artifact, SHA verified); "
          f"pos={int(Y.sum())} neg={int((1-Y).sum())}")

    with open(C.FOLD_MANIFEST, "r", encoding="utf-8") as fh:
        folds = C.load_folds(json.load(fh))
    print(f"[OK] {len(folds)} frozen folds; eps={C.epsilon():.15f} "
          f"delta={C.DELTA_DP} C={C.CLIP_NORM} nm={C.NOISE_MULTIPLIER} T=1")
    print("-" * 78)

    rows: List[Dict[str, Any]] = []
    auc: Dict[str, Dict[int, Dict[int, float]]] = {ar: {f: {} for f in C.FOLDS}
                                                   for ar in C.ARMS}
    n_runs = 0

    for r in C.REPEATS:
        for fspec in folds:
            f = fspec["fold"]
            seed = C.fold_seed(r, f)
            tr = [idx[p] for p in fspec["train_ids"]]
            te = [idx[p] for p in fspec["test_ids"]]

            w_before, w_after, loss = train_one(E[tr], Y[tr], PHQ[tr], seed)
            delta = compute_state_delta(w_before, w_after)          # FROZEN
            if set(delta) != set(C.HEAD_TENSORS):
                C.abort(f"delta key mismatch r{r} f{f}")
            flat = flatten(delta)
            if flat.size != C.HEAD_N_PARAMS or not np.isfinite(flat).all():
                C.abort(f"bad delta r{r} f{f}: size={flat.size}")

            # arm N - no DP
            auc_n = evaluate(w_after, E[te], Y[te])
            # arm D - identical training, DP applied to the same delta
            rel = dp_release(flat, seed)
            noisy_state = {k: w_before[k] + v
                           for k, v in unflatten(rel["noisy"]).items()}
            auc_d = evaluate(noisy_state, E[te], Y[te])

            auc["N"][f][r] = auc_n
            auc["D"][f][r] = auc_d
            n_runs += 1
            rows.append({
                "repeat": r, "fold": f, "seed": seed,
                "n_train": len(tr), "n_test": len(te),
                "final_loss": loss, "delta_l2": float(np.linalg.norm(flat)),
                "l2_pre_clip": rel["l2_pre_clip"], "clip_scale": rel["clip_scale"],
                "clipped": int(rel["clipped"]), "l2_post_clip": rel["l2_post_clip"],
                "l2_noise": rel["l2_noise"], "nsr": rel["nsr"],
                "roc_auc_N": auc_n, "roc_auc_D": auc_d,
                "epsilon": C.EPS_PER_UPDATE, "delta_dp": C.DELTA_DP,
            })
        rn = [rows[-5 + i]["roc_auc_N"] for i in range(5)]
        rd = [rows[-5 + i]["roc_auc_D"] for i in range(5)]
        print(f"  [repeat {r}] N mean={np.mean(rn):.6f}  D mean={np.mean(rd):.6f}  "
              f"clipped={sum(rows[-5+i]['clipped'] for i in range(5))}/5  "
              f"NSR mean={np.mean([rows[-5+i]['nsr'] for i in range(5)]):.2f}")

    if n_runs != C.N_REPEATS * C.N_FOLDS:
        C.abort(f"{n_runs} runs, expected {C.N_REPEATS * C.N_FOLDS}")

    frozen_after = {p: C.sha256_file(p) for p in C.FROZEN_SHA}
    changed = [p for p in frozen_before if frozen_before[p] != frozen_after[p]]
    if changed:
        C.abort(f"frozen artifacts modified DURING the run: {changed}")
    print("-" * 78)
    print(f"[OK] {n_runs} trainings; frozen artifacts unchanged")

    # ---- payload (A) -------------------------------------------------------
    raw_bytes = C.HEAD_N_PARAMS * 4
    transport = raw_bytes * C.TRANSPORT_EXPANSION
    payload = {
        "d": C.HEAD_N_PARAMS, "bytes_per_param": 4, "raw_bytes": raw_bytes,
        "transport_expansion": C.TRANSPORT_EXPANSION,
        "transport_bytes": transport,
        "threshold_bytes": C.G2_PAYLOAD_BYTES,
        "criterion_A_pass": bool(transport <= C.G2_PAYLOAD_BYTES),
        "basis": ("Exp 9 SS 17 convention: 4 bytes/param raw, times the measured "
                  "baseline transport expansion 1,579,963 / 1,185,335. Reported as "
                  "a predicted transport payload, not a live encryption run; the "
                  "margin is large enough that the verdict is insensitive to the "
                  "exact factor."),
    }

    # ---- clipping + NSR (B) ------------------------------------------------
    pre = [r["l2_pre_clip"] for r in rows]
    nsrs = [r["nsr"] for r in rows]
    n_clipped = sum(r["clipped"] for r in rows)
    nsr_mean, nsr_sd = C.mean_sd(nsrs)
    clipping = {
        "n_runs": len(rows), "n_clipped": n_clipped,
        "fraction_clipped": n_clipped / len(rows),
        "clipping_was_binding": n_clipped > 0,
        "pre_clip_norm_mean": float(np.mean(pre)),
        "pre_clip_norm_min": float(np.min(pre)),
        "pre_clip_norm_max": float(np.max(pre)),
        "pre_clip_norm_sd": float(np.std(pre, ddof=1)),
        "post_clip_norm_mean": float(np.mean([r["l2_post_clip"] for r in rows])),
        "noise_norm_mean": float(np.mean([r["l2_noise"] for r in rows])),
        "analytical_nsr_sqrt_d": float(np.sqrt(C.HEAD_N_PARAMS)),
        "measured_nsr_mean": nsr_mean, "measured_nsr_sd": nsr_sd,
        "threshold": C.G3_NSR,
        "criterion_B_pass": bool(nsr_mean < C.G3_NSR),
        "note": "measured, never tuned (design SS 13)",
    }

    os.makedirs(a.out_dir, exist_ok=True)
    cols = ["repeat", "fold", "seed", "n_train", "n_test", "final_loss",
            "delta_l2", "l2_pre_clip", "clip_scale", "clipped", "l2_post_clip",
            "l2_noise", "nsr", "roc_auc_N", "roc_auc_D", "epsilon", "delta_dp"]
    with open(os.path.join(a.out_dir, "b3b_runs.csv"), "w", newline="",
              encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=cols)
        w.writeheader()
        for r in rows:
            w.writerow({c: r[c] for c in cols})

    with open(os.path.join(a.out_dir, "b3b_roc_auc.json"), "w",
              encoding="utf-8") as fh:
        json.dump({"schema": "b3b_roc_auc/1",
                   "design": {"path": C.DESIGN_PATH, "sha256": C.design_sha()},
                   "metric": C.METRIC, "ranking_score": C.RANKING_SCORE,
                   "by_arm_fold_repeat": {ar: {str(f): {str(r): auc[ar][f][r]
                                                        for r in C.REPEATS}
                                               for f in C.FOLDS} for ar in C.ARMS},
                   }, fh, indent=2, sort_keys=True)

    for name, obj in (("b3b_payload.json", payload),
                      ("b3b_deltas.json", {"clipping": clipping,
                                           "per_run": rows})):
        with open(os.path.join(a.out_dir, name), "w", encoding="utf-8") as fh:
            json.dump({"schema": name.replace(".json", "/1"),
                       "design": {"path": C.DESIGN_PATH, "sha256": C.design_sha()},
                       **({"payload": obj} if "payload" in name else obj)},
                      fh, indent=2, sort_keys=True, default=float)

    with open(os.path.join(a.out_dir, "b3b_embeddings_receipt.json"), "w",
              encoding="utf-8") as fh:
        json.dump({"schema": "b3b_embeddings_receipt/1",
                   "reused_artifact": C.EMBEDDINGS_PATH,
                   "sha256_expected": C.EMBEDDINGS_SHA,
                   "sha256_measured": C.sha256_file(C.EMBEDDINGS_PATH),
                   "verified": C.sha256_file(C.EMBEDDINGS_PATH) == C.EMBEDDINGS_SHA,
                   "model": C.BERT_MODEL, "revision": C.BERT_REVISION,
                   "max_length": C.MAX_LENGTH, "padding": C.PADDING,
                   "pooling": C.POOLING, "embed_dim": C.EMBED_DIM,
                   "n_participants": C.N_PARTICIPANTS,
                   "note": "C-2 artifact reused read-only; C-2 was NOT re-run and "
                           "no C-2 artifact was modified (design SS 3)."},
                  fh, indent=2, sort_keys=True)

    elapsed = round(time.time() - t0, 2)
    with open(os.path.join(a.out_dir, "b3b_receipt.json"), "w",
              encoding="utf-8") as fh:
        json.dump({
            "schema": "b3b_receipt/1", "experiment": C.EXPERIMENT,
            "design": {"path": C.DESIGN_PATH, "sha256": C.design_sha()},
            "generated_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            "frozen_input_sha": {"pre_run": frozen_before, "post_run": frozen_after,
                                 "unchanged": True},
            "protocol": {"n_repeats": C.N_REPEATS, "n_folds": C.N_FOLDS,
                         "runs": n_runs, "arms": list(C.ARMS),
                         "arm_labels": C.ARM_LABEL},
            "architecture": {"class": "FusionHead", "in_dim": C.IN_DIM,
                             "hidden": C.HIDDEN, "num_classes": C.NUM_CLASSES,
                             "n_params": C.HEAD_N_PARAMS,
                             "n_tensors": C.HEAD_N_TENSORS},
            "recipe": {"optimizer": C.OPTIMIZER, "lr": C.LR, "epochs": C.EPOCHS,
                       "batch_size": C.BATCH_SIZE, "shuffle": C.SHUFFLE,
                       "dropout": C.DROPOUT, "grad_clip": C.GRAD_CLIP,
                       "loss": C.LOSS, "lambda": C.LAMBDA_REG},
            "transmitted_object": "genuine delta w_after - w_before "
                                  "(frozen compute_state_delta)",
            "dp": {"mechanism": C.MECHANISM, "epsilon": C.epsilon(),
                   "delta": C.DELTA_DP, "clip_norm": C.CLIP_NORM,
                   "noise_multiplier": C.NOISE_MULTIPLIER,
                   "composition": C.COMPOSITION},
            "second_attempt": True, "scope_caveat": C.SCOPE_CAVEAT,
            "environment": {"python": sys.version.split()[0],
                            "torch": torch.__version__, "platform": sys.platform},
            "runtime_seconds": elapsed,
        }, fh, indent=2, sort_keys=True, default=float)

    print(f"[A] payload {transport:,.0f} B vs {C.G2_PAYLOAD_BYTES:,} B -> "
          f"{'PASS' if payload['criterion_A_pass'] else 'FAIL'}")
    print(f"[B] NSR {nsr_mean:.4f} vs {C.G3_NSR} -> "
          f"{'PASS' if clipping['criterion_B_pass'] else 'FAIL'}   "
          f"(clipped {n_clipped}/{len(rows)}, pre-clip norm "
          f"{np.min(pre):.4f}-{np.max(pre):.4f})")
    print(f"[DONE] {n_runs} trainings in {elapsed:.1f}s -> {a.out_dir}")
    print("Next: aggregate_b3b.py (A/B/C verdict), then verify_b3b.py")
    return 0


if __name__ == "__main__":
    sys.exit(main())
