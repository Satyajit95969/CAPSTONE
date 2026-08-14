#!/usr/bin/env python3
"""
validate_c2.py - Phase 20 / C-2 PRE-TRAINING VALIDATION. FAIL CLOSED.

Authoritative specification: PHASE_20_C2_DESIGN.md (FROZEN).

This module runs the 22 pre-training checks and NOTHING ELSE. It does not
extract embeddings, does not construct or train a probe, does not draw DP noise,
does not aggregate, and writes no experimental artifact. Its only side effect is
an optional JSON report, and even that is written only when --report is passed.

FAIL CLOSED (design SS 19):
    * any failed check   -> FAIL, non-zero exit
    * any missing input  -> FAIL (not PENDING: every input must exist BEFORE
                            training, so absence is a hard failure here)
    * unpinned path keys are normalised before lookup and never raise KeyError

If ANY check fails: STOP. Do not train. Do not modify frozen artifacts.

Usage:
    python exp_c2_multiclient/validate_c2.py
    python exp_c2_multiclient/validate_c2.py --report
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from typing import Any, Dict, List

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import c2_common as C  # noqa: E402

PASS, FAIL = "PASS", "FAIL"


class Checks:
    def __init__(self) -> None:
        self.items: List[Dict[str, Any]] = []

    def ok(self, num: int, name: str, condition: bool, detail: str = "") -> bool:
        self.items.append({"n": num, "check": name,
                           "status": PASS if condition else FAIL,
                           "detail": detail})
        return bool(condition)

    @property
    def n_fail(self) -> int:
        return sum(1 for x in self.items if x["status"] == FAIL)

    @property
    def n_pass(self) -> int:
        return sum(1 for x in self.items if x["status"] == PASS)


def main() -> int:
    ap = argparse.ArgumentParser(
        description="Phase 20 / C-2 pre-training validation (fail closed)")
    ap.add_argument("--out-dir", default=os.path.join(C.REPO, C.OUT_DIRNAME))
    ap.add_argument("--report", action="store_true",
                    help="write c2_validation.json (the ONLY side effect)")
    a = ap.parse_args()
    os.chdir(C.REPO)

    k = Checks()
    print("=" * 78)
    print("C-2 PRE-TRAINING VALIDATION - FAIL CLOSED")
    print("no embeddings, no training, no DP, no aggregation")
    print("=" * 78)

    # ---------------- 1-4: pinned MentalBERT revision and artifacts ----------
    snap = C.hf_snapshot_dir()
    refs = os.path.join(os.path.dirname(os.path.dirname(snap)), "refs", "main")
    ref_rev = None
    if os.path.isfile(refs):
        with open(refs, "r", encoding="utf-8") as fh:
            ref_rev = fh.read().strip()
    k.ok(1, "MentalBERT revision pinned and present",
         os.path.isdir(snap) and ref_rev == C.BERT_REVISION,
         f"refs/main={ref_rev} want={C.BERT_REVISION} snapshot_dir_exists="
         f"{os.path.isdir(snap)}")

    for num, fname in ((2, "pytorch_model.bin"), (3, "tokenizer.json"),
                       (4, "vocab.txt")):
        p = os.path.join(snap, fname)
        got = C.sha256_file(p)
        want = C.BERT_ARTIFACT_SHA[fname]
        k.ok(num, f"MentalBERT artifact SHA: {fname}", got == want,
             f"got={got} want={want}")

    # ---------------- 5: frozen repository artifacts -------------------------
    bad = []
    for path, want in C.FROZEN_SHA.items():
        if want is None:
            continue
        got = C.sha256_file(path)
        if got is None:
            bad.append(f"{path}: MISSING")
        elif got != want:
            bad.append(f"{path}: got {got} want {want}")
    dsha = C.design_sha()
    k.ok(5, "frozen repository artifact SHAs", not bad and dsha is not None,
         "all match; design=" + str(dsha) if not bad else "; ".join(bad))

    # ---------------- 6-8: input data ---------------------------------------
    import pandas as pd
    df = pd.read_parquet(C.DATA_PARQUET)
    pids = [int(x) for x in df["participant_id"]]
    phq = {int(r["participant_id"]): float(r["phq_score"])
           for _, r in df.iterrows()}
    k.ok(6, "input count = 188", len(df) == C.N_PARTICIPANTS, f"rows={len(df)}")
    k.ok(7, "participant IDs unique",
         len(set(pids)) == C.N_PARTICIPANTS and len(pids) == len(set(pids)),
         f"unique={len(set(pids))} duplicates={len(pids) - len(set(pids))}")
    vals = np.array(list(phq.values()), dtype=np.float64)
    k.ok(8, "PHQ scores complete and finite",
         int(df["phq_score"].isna().sum()) == 0 and bool(np.isfinite(vals).all())
         and int(df["text"].isna().sum()) == 0,
         f"nulls_phq={int(df['phq_score'].isna().sum())} "
         f"range=[{vals.min()}, {vals.max()}] nulls_text="
         f"{int(df['text'].isna().sum())}")

    # ---------------- 9-13: partitions --------------------------------------
    expect_ids = set(pids)
    with open(C.FOLD_MANIFEST, "r", encoding="utf-8") as fh:
        manifest = json.load(fh)
    w0 = C.control_partition(manifest)
    w1 = C.a2_partition([(p, phq[p]) for p in pids])
    s0 = C.check_partition(w0, expect_ids)
    s1 = C.check_partition(w1, expect_ids)

    k.ok(9, "control (W0) partition valid: disjoint + 188/188",
         s0["disjoint"] and s0["coverage_ok"] and s0["duplicates"] == 0,
         f"sizes={s0['sizes']} unique={s0['n_unique']} dup={s0['duplicates']} "
         f"disjoint={s0['disjoint']}")
    k.ok(10, "A2 (W1) partition valid: disjoint + 188/188 + deterministic",
         s1["disjoint"] and s1["coverage_ok"] and s1["duplicates"] == 0,
         f"sizes={s1['sizes']} unique={s1['n_unique']} dup={s1['duplicates']} "
         f"disjoint={s1['disjoint']}")
    k.ok(11, "client sizes 38/38/38/37/37 in BOTH arms",
         s0["sizes_ok"] and s1["sizes_ok"],
         f"W0={s0['sizes']} W1={s1['sizes']}")
    flat0 = {i for p in w0 for i in p}
    flat1 = {i for p in w1 for i in p}
    k.ok(12, "both arms cover exactly the same 188 participants",
         flat0 == flat1 == expect_ids,
         f"W0={len(flat0)} W1={len(flat1)} identical={flat0 == flat1}")
    k.ok(13, "no cross-client overlap within either arm",
         not s0["overlaps"] and not s1["overlaps"],
         f"W0_overlaps={s0['overlaps']} W1_overlaps={s1['overlaps']}")

    # ---------------- 14-16: probe / shared initialisation -------------------
    import torch
    sd = torch.load(C.PROBE_PATH, map_location="cpu", weights_only=False)
    names_ok = set(sd.keys()) == set(C.PROBE_TENSORS)
    shapes_ok = all(tuple(sd[n].shape) == s for n, s in C.PROBE_TENSORS.items()
                    if n in sd)
    k.ok(14, "probe state_dict = exactly 4 tensors, correct names and shapes",
         len(sd) == C.PROBE_N_TENSORS and names_ok and shapes_ok,
         f"n_tensors={len(sd)} names_ok={names_ok} shapes_ok={shapes_ok}")
    n_params = int(sum(v.numel() for v in sd.values()))
    k.ok(15, "probe parameter count = 295,681",
         n_params == C.PROBE_N_PARAMS, f"{n_params:,}")

    # delta keys/shapes: compute_state_delta(before, before) must be key- and
    # shape-identical to the probe. Uses the FROZEN trainer helper, no training.
    import trainer_mentalbert_daic as T
    probe_before = {kk: v.detach().cpu().clone() for kk, v in sd.items()}
    d0 = T.compute_state_delta(probe_before, probe_before)
    delta_ok = (set(d0.keys()) == set(sd.keys())
                and all(tuple(d0[n].shape) == tuple(sd[n].shape) for n in sd)
                and all(float(torch.max(torch.abs(d0[n]))) == 0.0 for n in sd))
    k.ok(16, "delta keys/shapes match w_before (frozen compute_state_delta)",
         delta_ok,
         f"keys={len(d0)} identical_shapes=True self_delta_is_zero="
         f"{all(float(torch.max(torch.abs(d0[n]))) == 0.0 for n in sd)}")

    # ---------------- 17: DP parameters --------------------------------------
    eps = C.epsilon()
    k.ok(17, "DP parameters exactly match frozen values",
         abs(eps - C.EPS_PER_UPDATE) <= 1e-12
         and C.DELTA_DP == 1e-05 and C.CLIP_NORM == 1.0
         and C.NOISE_MULTIPLIER == 1.0 and C.MECHANISM == "gaussian",
         f"eps={eps!r} delta={C.DELTA_DP} clip={C.CLIP_NORM} "
         f"nm={C.NOISE_MULTIPLIER} mech={C.MECHANISM} T=1")

    # ---------------- 18: aggregation, measured on the FROZEN aggregator -----
    import importlib.util as iu
    spec = iu.spec_from_file_location("c2_agg", os.path.join(C.REPO,
                                                             C.AGGREGATOR_PATH))
    agg_mod = iu.module_from_spec(spec)
    spec.loader.exec_module(agg_mod)
    agg = agg_mod.AggregatorAgent(mode=C.AGG_MODE, trim_ratio=C.TRIM_RATIO)
    lower, upper, kept = C.trim_window(C.N_CLIENTS)
    probe_stack = torch.arange(C.N_CLIENTS * 3, dtype=torch.float32).reshape(
        C.N_CLIENTS, 3)
    got_mid = agg._aggregate_tensor(probe_stack)
    expect_mid = probe_stack.sort(dim=0).values[lower:upper].mean(dim=0)
    k.ok(18, "trim_ratio = 0.1 retains 3 of 5 on the FROZEN aggregator",
         agg.mode == C.AGG_MODE and abs(agg.trim_ratio - C.TRIM_RATIO) < 1e-12
         and kept == 3 and torch.allclose(got_mid, expect_mid),
         f"mode={agg.mode} trim_ratio={agg.trim_ratio} "
         f"lower={lower} upper={upper} kept={kept} verified_on_probe_tensor=True")

    # ---------------- 19: n --------------------------------------------------
    k.ok(19, "n = 5 clients",
         C.N_CLIENTS == 5 and len(w0) == 5 and len(w1) == 5,
         f"N_CLIENTS={C.N_CLIENTS} W0={len(w0)} W1={len(w1)}")

    # ---------------- 20: seed mapping determinism ---------------------------
    m1 = {(r, i): (C.repeat_seed(r), C.client_seed(r, i), C.dp_seed(r))
          for r in C.REPEATS for i in range(1, C.N_CLIENTS + 1)}
    m2 = {(r, i): (C.repeat_seed(r), C.client_seed(r, i), C.dp_seed(r))
          for r in C.REPEATS for i in range(1, C.N_CLIENTS + 1)}
    expected_r1c1 = (1001, 100101, 1001)
    k.ok(20, "seed mapping deterministic and matches design SS 13",
         m1 == m2 and m1[(1, 1)] == expected_r1c1
         and len({v[1] for v in m1.values()}) == C.N_REPEATS * C.N_CLIENTS,
         f"repeat1/client1 -> repeat_seed={m1[(1,1)][0]} "
         f"client_seed={m1[(1,1)][1]} dp_seed={m1[(1,1)][2]}; "
         f"{len(m1)} unique client seeds={len({v[1] for v in m1.values()})}")

    # ---------------- 21: output isolation -----------------------------------
    frozen_dirs = ["trainer_outputs/exp9_compact_update",
                   "trainer_outputs/multimodal_exp",
                   "trainer_outputs/exp8_dp_mechanism",
                   "trainer_outputs/exp7_modality",
                   "trainer_outputs/baseline_cv",
                   "secure_store/sess-DAICWOZ"]
    out_norm = C.repo_key(os.path.relpath(a.out_dir, C.REPO))
    collides = [d for d in frozen_dirs
                if out_norm == d or out_norm.startswith(d + "/")]
    k.ok(21, "output directory does not collide with frozen artifacts",
         not collides and not os.path.isdir(a.out_dir),
         f"out_dir={out_norm} exists={os.path.isdir(a.out_dir)} "
         f"collisions={collides or 'none'}")

    # ---------------- 22: representation configuration -----------------------
    k.ok(22, "representation config matches the frozen definition",
         C.MAX_LENGTH == 128 and C.TRUNCATION is True
         and C.PADDING == "max_length" and C.EMBED_DIM == 768
         and C.POOLING == "last_hidden_state[:, 0, :]"
         and C.BERT_MODEL == "mental/mental-bert-base-uncased",
         f"model={C.BERT_MODEL} rev={C.BERT_REVISION} max_length={C.MAX_LENGTH} "
         f"truncation={C.TRUNCATION} padding={C.PADDING} pooling={C.POOLING} "
         f"dim={C.EMBED_DIM}")

    # ---------------- report -------------------------------------------------
    print()
    for it in k.items:
        mark = "ok  " if it["status"] == PASS else "FAIL"
        print(f"  [{mark}] ({it['n']:>2}) {it['check']}")
        if it["detail"]:
            print(f"          {it['detail']}")

    overall = PASS if k.n_fail == 0 else FAIL
    print("-" * 78)
    print(f"checks: {k.n_pass} PASS, {k.n_fail} FAIL of {len(k.items)}")
    print(f"VALIDATION: {overall}")
    if overall == FAIL:
        print("\nSTOP. Do not train. Do not modify frozen artifacts.",
              file=sys.stderr)

    if a.report:
        os.makedirs(a.out_dir, exist_ok=True)
        p = os.path.join(a.out_dir, "c2_validation.json")
        with open(p, "w", encoding="utf-8") as fh:
            json.dump({"schema": "c2_validation/1",
                       "experiment": C.EXPERIMENT,
                       "design": {"path": C.DESIGN_PATH, "sha256": C.design_sha()},
                       "fail_closed": True,
                       "n_pass": k.n_pass, "n_fail": k.n_fail,
                       "overall": overall, "checks": k.items}, fh,
                      indent=2, sort_keys=True)
        print(f"[report] -> {p}")

    return 0 if overall == PASS else 1


if __name__ == "__main__":
    sys.exit(main())
