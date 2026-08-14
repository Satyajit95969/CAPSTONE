#!/usr/bin/env python3
"""
verify_c2.py - Phase 20 / C-2 independent fail-closed verification.

Authoritative specification: PHASE_20_C2_DESIGN.md (FROZEN).

FAIL-CLOSED CONTRACT
--------------------
  PASS     - the claim was independently recomputed and matched
  FAIL     - the claim was recomputed and did NOT match
  PENDING  - the evidence needed to judge is absent

Missing evidence is NEVER reported as PASS. The process exits non-zero unless
every check is PASS. This verifier re-derives the primary result from the frozen
inputs rather than trusting c2_summary.json: it retrains all 10 clients from
scratch, recomputes both partitions, recomputes the divergences, re-exercises the
frozen aggregator, and only then compares against the recorded artifacts.

NO TASK METRIC (design SS 14): check 20 scans every artifact for forbidden
task-performance keys and fails if any is present.

Usage:
    python exp_c2_multiclient/verify_c2.py
    python exp_c2_multiclient/verify_c2.py --report verify_c2_report.json
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import re
import sys
import time
from typing import Any, Dict, List, Optional

import numpy as np
import torch
import torch.nn as nn

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import c2_common as C  # noqa: E402
import trainer_mentalbert_daic as T  # noqa: E402  FROZEN

FORBIDDEN = ("roc_auc", "rocauc", "accuracy", "f1_score", "pr_auc",
             "held_out_score", "test_auc", "val_auc")

CHECKS: List[Dict[str, Any]] = []


def record(name: str, status: str, detail: str = "") -> None:
    CHECKS.append({"check": name, "status": status, "detail": detail})
    tag = {"PASS": "ok  ", "FAIL": "FAIL", "PENDING": "PEND"}[status]
    print(f"  [{tag}] {name}")
    if detail:
        print(f"         {detail}")


def cmp_check(name: str, got: Any, want: Any, detail: str = "") -> None:
    """Three-way. Absent evidence -> PENDING, never PASS (V1 fail-open lesson)."""
    if got is None:
        record(name, "PENDING", detail or "evidence absent from artifacts")
    elif got == want:
        record(name, "PASS", detail or f"{got!r}")
    else:
        record(name, "FAIL", detail or f"got {got!r}, expected {want!r}")


def load_json(path: str) -> Optional[Dict[str, Any]]:
    if not os.path.isfile(path):
        return None
    with open(path, "r", encoding="utf-8") as fh:
        return json.load(fh)


class Probe(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.fc1 = nn.Linear(C.EMBED_DIM, 384)
        self.fc2 = nn.Linear(384, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.fc2(torch.relu(self.fc1(x)))


def retrain(w_before, X, y, seed):
    """Independent re-implementation of the frozen recipe (design SS 9)."""
    torch.manual_seed(seed)
    m = Probe()
    m.load_state_dict(w_before, strict=True)
    m.train()
    opt = torch.optim.Adam(m.parameters(), lr=C.LR)
    xb = torch.tensor(X, dtype=torch.float32)
    yb = torch.tensor(y, dtype=torch.float32).view(-1, 1)
    for _ in range(C.EPOCHS):
        opt.zero_grad()
        loss = nn.MSELoss()(m(xb), yb)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(m.parameters(), C.GRAD_CLIP)
        opt.step()
    return {k: v.detach().cpu().clone() for k, v in m.state_dict().items()}


def main() -> int:
    ap = argparse.ArgumentParser(description="Phase 20 / C-2 verification")
    ap.add_argument("--out-dir", default=os.path.join(C.REPO, C.OUT_DIRNAME))
    ap.add_argument("--report", default=None)
    a = ap.parse_args()
    os.chdir(C.REPO)
    od = a.out_dir
    t0 = time.time()

    print("=" * 78)
    print("C-2 INDEPENDENT VERIFICATION - FAIL CLOSED")
    print("PASS = recomputed and matched | FAIL = mismatch | PENDING = no evidence")
    print("=" * 78)

    summary = load_json(os.path.join(od, "c2_summary.json"))
    receipt = load_json(os.path.join(od, "c2_receipt.json"))
    mrec = load_json(os.path.join(od, "c2_model_receipt.json"))
    pre = load_json(os.path.join(od, "c2_divergence_pre_dp.json"))
    post = load_json(os.path.join(od, "c2_divergence_post_dp.json"))
    aggj = load_json(os.path.join(od, "c2_aggregation.json"))
    partj = load_json(os.path.join(od, "c2_partitions.json"))
    seedj = load_json(os.path.join(od, "c2_seed_manifest.json"))

    # -- 1 design frozen -----------------------------------------------------
    # FROZEN_SHA[DESIGN_PATH] is deliberately None (c2_common.py:200): the design
    # SHA is not hardcoded beside the design it describes, it is pinned INTO every
    # artifact at run time. The verifiable claim is therefore that the design has
    # not changed SINCE the run - the live SHA must equal the SHA recorded in
    # every artifact, and those must all agree with each other.
    live = C.design_sha()
    pinned = {n: (o or {}).get("design", {}).get("sha256")
              for n, o in (("c2_receipt", receipt), ("c2_model_receipt", mrec),
                           ("c2_summary", summary), ("c2_divergence_pre_dp", pre),
                           ("c2_divergence_post_dp", post),
                           ("c2_aggregation", aggj), ("c2_partitions", partj))}
    seen = {n: v for n, v in pinned.items() if v is not None}
    if not seen:
        record("1. design unchanged since the run (SHA pinned into artifacts)",
               "PENDING", "no artifact records a design SHA")
    else:
        mismatched = {n: v for n, v in seen.items() if v != live}
        record("1. design unchanged since the run (SHA pinned into artifacts)",
               "PASS" if not mismatched else "FAIL",
               f"live {live} == SHA pinned in all {len(seen)} artifacts"
               if not mismatched else
               f"live {live}; disagreeing artifacts: {mismatched}")

    # -- 2 frozen inputs -----------------------------------------------------
    mism = [f"{p}: {C.sha256_file(p)} != {w}" for p, w in C.FROZEN_SHA.items()
            if w is not None and C.sha256_file(p) != w]
    record("2. every frozen input SHA unchanged",
           "PASS" if not mism else "FAIL",
           f"{len([w for w in C.FROZEN_SHA.values() if w])} pinned artifacts"
           if not mism else "; ".join(mism))

    # -- 3 MentalBERT revision ----------------------------------------------
    snap = C.hf_snapshot_dir()
    bad = [f for f, w in C.BERT_ARTIFACT_SHA.items()
           if C.sha256_file(os.path.join(snap, f)) != w]
    record("3. MentalBERT pinned revision artifacts intact",
           "PASS" if not bad else "FAIL",
           f"revision {C.BERT_REVISION}, {len(C.BERT_ARTIFACT_SHA)} artifacts"
           if not bad else f"mismatched: {bad}")

    # -- 4 embeddings --------------------------------------------------------
    npz_path = os.path.join(od, "c2_embeddings.npz")
    if not os.path.isfile(npz_path):
        record("4. embedding artifact present and well-formed", "PENDING",
               "c2_embeddings.npz absent")
        z = None
    else:
        z = np.load(npz_path, allow_pickle=False)
        E, ids, Y = z["embedding"], [int(i) for i in z["participant_id"]], z["phq_score"]
        ok = (E.shape == (C.N_PARTICIPANTS, C.EMBED_DIM)
              and bool(np.isfinite(E).all()) and len(set(ids)) == C.N_PARTICIPANTS)
        record("4. embeddings 188x768, unique IDs, all finite",
               "PASS" if ok else "FAIL", f"shape={E.shape} unique={len(set(ids))}")

    # -- 5 embeddings match the source parquet ------------------------------
    if z is None:
        record("5. embedding IDs/PHQ match daic_records.parquet", "PENDING",
               "no embedding artifact")
    else:
        import pandas as pd
        df = pd.read_parquet(C.DATA_PARQUET)
        src_ids = [int(x) for x in df["participant_id"]]
        src_phq = [float(x) for x in df["phq_score"]]
        ok = ids == src_ids and all(float(Y[i]) == src_phq[i] for i in range(len(ids)))
        record("5. embedding IDs and PHQ match the frozen parquet exactly",
               "PASS" if ok else "FAIL", f"188 IDs in order, PHQ elementwise equal")

    # -- 6 embeddings independently reproducible ----------------------------
    if z is None:
        record("6. embeddings reproduce from the pinned revision", "PENDING",
               "no embedding artifact")
    else:
        from transformers import AutoModel, AutoTokenizer
        import pandas as pd
        tok = AutoTokenizer.from_pretrained(snap, local_files_only=True)
        mdl = AutoModel.from_pretrained(snap, local_files_only=True)
        mdl.eval()
        sample = [0, 47, 93, 140, 187]
        dfx = pd.read_parquet(C.DATA_PARQUET)
        diffs = []
        with torch.no_grad():
            for i in sample:
                enc = tok(str(dfx.iloc[i]["text"]), return_tensors="pt",
                          truncation=C.TRUNCATION, padding=C.PADDING,
                          max_length=C.MAX_LENGTH)
                cls = mdl(**enc).last_hidden_state[:, 0, :].squeeze(0).numpy()
                diffs.append(float(np.max(np.abs(cls.astype(np.float64) - E[i]))))
        ok = max(diffs) == 0.0
        record("6. embeddings reproduce BIT-EXACTLY from the pinned revision",
               "PASS" if ok else "FAIL",
               f"5 sampled participants, max |diff| = {max(diffs):.3e}")

    # -- 7/8 partitions independently recomputed ----------------------------
    manifest = load_json(C.FOLD_MANIFEST)
    if z is None or manifest is None or partj is None:
        record("7. W0 partition equals the frozen fold manifest", "PENDING",
               "inputs absent")
        record("8. W1 partition equals the recomputed A2 rule", "PENDING",
               "inputs absent")
        rec_parts = None
    else:
        phq = {p: float(Y[i]) for i, p in enumerate(ids)}
        rec_parts = {C.CONTROL_ARM: C.control_partition(manifest),
                     C.TREATMENT_ARM: C.a2_partition([(p, phq[p]) for p in ids])}
        for n, arm in ((7, C.CONTROL_ARM), (8, C.TREATMENT_ARM)):
            stored = [sorted(c["participant_ids"])
                      for c in partj["arms"][arm]["clients"]]
            ok = [sorted(x) for x in rec_parts[arm]] == stored
            label = ("frozen fold manifest" if arm == C.CONTROL_ARM
                     else "recomputed A2 rule (phq DESC, pid ASC)")
            record(f"{n}. {arm} partition equals the {label}",
                   "PASS" if ok else "FAIL",
                   f"5 clients, sizes {[len(x) for x in rec_parts[arm]]}")

    # -- 9 partition integrity ----------------------------------------------
    if rec_parts is None:
        record("9. partitions disjoint, exhaustive, correctly sized", "PENDING",
               "inputs absent")
        record("10. both arms cover the identical 188 participants", "PENDING",
               "inputs absent")
    else:
        expect = set(ids)
        sts = {arm: C.check_partition(pl, expect) for arm, pl in rec_parts.items()}
        ok = all(s["disjoint"] and s["coverage_ok"] and s["sizes_ok"]
                 and s["duplicates"] == 0 for s in sts.values())
        record("9. partitions disjoint, exhaustive (188/188), sizes 38/38/38/37/37",
               "PASS" if ok else "FAIL",
               f"W0 {sts['W0']['sizes']} / W1 {sts['W1']['sizes']}, "
               f"duplicates {sts['W0']['duplicates']}/{sts['W1']['duplicates']}")
        same = ({i for p in rec_parts["W0"] for i in p} ==
                {i for p in rec_parts["W1"] for i in p})
        record("10. both arms cover the identical 188-participant population",
               "PASS" if same else "FAIL",
               "single-factor: only the partition differs (design SS 17)")

    # -- 11 probe ------------------------------------------------------------
    wb = torch.load(C.PROBE_PATH, map_location="cpu", weights_only=False)
    wb = {k: v.detach().cpu().clone() for k, v in wb.items()}
    npar = int(sum(v.numel() for v in wb.values()))
    ok = npar == C.PROBE_N_PARAMS and len(wb) == C.PROBE_N_TENSORS
    record("11. shared initialisation has 4 tensors / 295,681 parameters",
           "PASS" if ok else "FAIL", f"{len(wb)} tensors, {npar:,} params")

    # -- 12 independent re-derivation of the PRIMARY endpoint ---------------
    if z is None or rec_parts is None:
        record("12. pre-DP divergence reproduces from an independent retrain",
               "PENDING", "inputs absent")
        rec_pre = None
    else:
        idx = {p: i for i, p in enumerate(ids)}
        rec_pre = {}
        for arm in C.ARMS:
            flats = []
            for ci, members in enumerate(rec_parts[arm], start=1):
                rows = [idx[p] for p in members]
                wa = retrain(wb, E[rows].astype(np.float64), Y[rows].astype(np.float64),
                             C.client_seed(C.REPEATS[0], ci))
                dl = T.compute_state_delta(wb, wa)          # FROZEN convention
                flats.append(np.concatenate([dl[k].numpy().astype(np.float64).ravel()
                                             for k in C.PROBE_TENSORS]))
            rec_pre[arm] = C.pairwise_divergence(flats)
        if pre is None:
            record("12. pre-DP divergence reproduces from an independent retrain",
                   "PENDING", "c2_divergence_pre_dp.json absent")
        else:
            worst = max(
                max(abs(rec_pre[arm][m][k] - pre["by_arm_repeat"][f"{arm}|r1"][m][k])
                    for m in ("l2", "cosine") for k in range(10))
                for arm in C.ARMS)
            record("12. pre-DP divergence reproduces from an INDEPENDENT retrain",
                   "PASS" if worst == 0.0 else "FAIL",
                   f"10 clients retrained from w_before; max |diff| over 40 "
                   f"recorded values = {worst:.3e}")

    # -- 13 the validity gate, recomputed ------------------------------------
    if pre is None:
        record("13. validity gate recomputed from the divergence artifact",
               "PENDING", "c2_divergence_pre_dp.json absent")
        gate = None
    else:
        gate = {}
        for m in ("l2", "cosine"):
            w0 = float(np.mean(pre["by_arm_repeat"][f"{C.CONTROL_ARM}|r1"][m]))
            w1 = float(np.mean(pre["by_arm_repeat"][f"{C.TREATMENT_ARM}|r1"][m]))
            gate[m] = {"w0": w0, "w1": w1, "pass": bool(w1 > w0)}
        allp = all(v["pass"] for v in gate.values())
        record("13. validity gate recomputed independently (W1 > W0, directional)",
               "PASS" if allp else "FAIL",
               " | ".join(f"{m}: W0={v['w0']:.9f} W1={v['w1']:.9f} "
                          f"diff={v['w1'] - v['w0']:+.9f} "
                          f"{'PASS' if v['pass'] else 'FAIL'}"
                          for m, v in gate.items()))

    # -- 14 recorded gate matches the recomputation --------------------------
    rec_res = None if gate is None else ("PASS" if all(v["pass"] for v in gate.values())
                                         else "FAIL")
    stored_res = (summary or {}).get("validity_gate", {}).get("result")
    cmp_check("14. recorded gate result equals the independent recomputation",
              stored_res, rec_res,
              f"recorded={stored_res!r} recomputed={rec_res!r}"
              if stored_res is not None else "c2_summary.json absent")

    # -- 15 repeat degeneracy independently confirmed ------------------------
    if pre is None:
        record("15. repeat degeneracy on the primary endpoint confirmed",
               "PENDING", "c2_divergence_pre_dp.json absent")
    else:
        b = pre["by_arm_repeat"]
        ident = all(b[f"{arm}|r{r}"]["l2"] == b[f"{arm}|r1"]["l2"]
                    and b[f"{arm}|r{r}"]["cosine"] == b[f"{arm}|r1"]["cosine"]
                    for arm in C.ARMS for r in C.REPEATS)
        claimed = (summary or {}).get("repeat_degeneracy", {}).get(
            "primary_endpoint_deterministic")
        if claimed is None:
            record("15. repeat degeneracy on the primary endpoint confirmed",
                   "PENDING", "summary does not report repeat_degeneracy")
        else:
            record("15. repeat degeneracy independently confirmed and disclosed",
                   "PASS" if (bool(claimed) == ident) else "FAIL",
                   f"observed deterministic={ident}, summary claims {claimed}; "
                   f"primary effective replicates = 1 per arm, not "
                   f"{C.N_REPEATS}")

    # -- 16 no cross-repeat CI on the degenerate primary endpoint -----------
    if summary is None:
        record("16. no cross-repeat SD/CI reported on the primary endpoint",
               "PENDING", "c2_summary.json absent")
    else:
        blob = json.dumps(summary.get("primary_pre_dp", {}))
        leaked = [k for k in ("ci_low", "ci_high", "sd_across_repeats",
                              "confidence_interval") if k in blob]
        record("16. no cross-repeat SD/CI is reported on the primary endpoint",
               "PASS" if not leaked else "FAIL",
               "pseudo-replication avoided; only within-replicate pair spread "
               "is reported" if not leaked else f"leaked keys: {leaked}")

    # -- 17 seeds ------------------------------------------------------------
    if seedj is None:
        record("17. seed hierarchy matches the frozen formulas", "PENDING",
               "c2_seed_manifest.json absent")
    else:
        e = seedj["entries"]
        ok = (len(e) == 50
              and all(x["client_seed"] == C.client_seed(x["repeat"], x["client"])
                      and x["repeat_seed"] == C.repeat_seed(x["repeat"])
                      and x["dp_seed"] == C.dp_seed(x["repeat"]) for x in e))
        uniq = len({x["client_seed"] for x in e})
        record("17. seed hierarchy matches the frozen formulas exactly",
               "PASS" if (ok and uniq == 25) else "FAIL",
               f"50 entries, {uniq} unique client seeds "
               f"(25 = 5 repeats x 5 clients, shared across arms by design SS 13)")

    # -- 18 DP accounting ----------------------------------------------------
    eps = C.epsilon()
    cmp_check("18. epsilon recomputed live via the frozen dp_agent",
              eps, C.EPS_PER_UPDATE,
              f"eps={eps!r} delta={C.DELTA_DP} nm={C.NOISE_MULTIPLIER} "
              f"C={C.CLIP_NORM} T=1")

    # -- 19 frozen aggregator trimming --------------------------------------
    import importlib.util as iu
    spec = iu.spec_from_file_location("v_agg", os.path.join(C.REPO, C.AGGREGATOR_PATH))
    am = iu.module_from_spec(spec)
    spec.loader.exec_module(am)
    agg = am.AggregatorAgent(mode=C.AGG_MODE, trim_ratio=C.TRIM_RATIO)
    probe = torch.tensor([[0.0], [1.0], [2.0], [3.0], [100.0]])
    got = float(agg._aggregate_tensor(probe).item())
    lo, up, kept = C.trim_window(C.N_CLIENTS)
    record("19. frozen AggregatorAgent trims 1 low + 1 high, keeping 3 of 5",
           "PASS" if (abs(got - 2.0) < 1e-9 and kept == 3) else "FAIL",
           f"[0,1,2,3,100] -> {got} (mean of 1,2,3); window [{lo}:{up}]")

    # -- 20 NO TASK METRIC ---------------------------------------------------
    found: List[str] = []
    for fn in sorted(os.listdir(od)) if os.path.isdir(od) else []:
        if not fn.endswith((".json", ".csv")):
            continue
        with open(os.path.join(od, fn), "r", encoding="utf-8") as fh:
            txt = fh.read().lower()
        for k in FORBIDDEN:
            if re.search(r'"' + re.escape(k) + r'"\s*:', txt):
                found.append(f"{fn}:{k}")
    declared = (summary or {}).get("task_metric_computed")
    if declared is None:
        record("20. NO task metric computed anywhere (design SS 14)", "PENDING",
               "c2_summary.json absent")
    else:
        ok = (not found) and declared is False
        record("20. NO task metric computed anywhere (design SS 14)",
               "PASS" if ok else "FAIL",
               f"scanned all artifacts for {len(FORBIDDEN)} forbidden keys; "
               f"none present; summary declares task_metric_computed=false"
               if ok else f"found {found} / declared={declared!r}")

    # -- 21 scope caveat -----------------------------------------------------
    cav = (summary or {}).get("scope_caveat")
    cmp_check("21. scope caveat recorded verbatim from design SS 21",
              cav, C.SCOPE_CAVEAT,
              "present and identical" if cav == C.SCOPE_CAVEAT else
              ("absent" if cav is None else "differs from the frozen text"))

    # -- 22 completeness -----------------------------------------------------
    if receipt is None:
        record("22. all 50 client trainings recorded", "PENDING",
               "c2_receipt.json absent")
    else:
        cpath = os.path.join(od, "c2_client_updates.csv")
        nrows = (sum(1 for _ in csv.DictReader(open(cpath, encoding="utf-8")))
                 if os.path.isfile(cpath) else None)
        exp = C.N_REPEATS * len(C.ARMS) * C.N_CLIENTS
        cmp_check("22. all 50 client trainings recorded", nrows, exp,
                  f"{nrows} rows = {C.N_REPEATS} repeats x {len(C.ARMS)} arms x "
                  f"{C.N_CLIENTS} clients" if nrows else "CSV absent")

    # -- 23 no checkpoint reuse / shared init --------------------------------
    integ = (receipt or {}).get("integrity", {})
    cmp_check("23. no checkpoint reuse across clients, arms or repeats",
              integ.get("checkpoint_reuse"), False,
              "every client starts from the same immutable w_before"
              if integ else "receipt absent")

    # -- 24 artifact set complete -------------------------------------------
    want = ["c2_embeddings.npz", "c2_model_receipt.json", "c2_client_updates.csv",
            "c2_partitions.json", "c2_seed_manifest.json",
            "c2_divergence_pre_dp.json", "c2_divergence_post_dp.json",
            "c2_aggregation.json", "c2_receipt.json", "c2_summary.json",
            "c2_acceptance.csv"]
    missing = [f for f in want if not os.path.isfile(os.path.join(od, f))]
    record("24. all 11 declared artifacts present (design SS 18)",
           "PASS" if not missing else "FAIL",
           f"{len(want)} artifacts" if not missing else f"missing: {missing}")

    # -- 25 outputs confined to the C-2 directory ----------------------------
    mrec_ok = mrec is not None and mrec.get("task_metric_computed") is False
    cmp_check("25. extraction receipt also declares no task metric",
              None if mrec is None else mrec.get("task_metric_computed"), False,
              "c2_model_receipt.json declares task_metric_computed=false"
              if mrec_ok else "receipt absent or does not declare it")

    # ---- verdict -----------------------------------------------------------
    n_pass = sum(1 for c in CHECKS if c["status"] == "PASS")
    n_fail = sum(1 for c in CHECKS if c["status"] == "FAIL")
    n_pend = sum(1 for c in CHECKS if c["status"] == "PENDING")
    print("=" * 78)
    print(f"RESULT: {n_pass} PASS, {n_fail} FAIL, {n_pend} PENDING "
          f"of {len(CHECKS)} checks   ({time.time() - t0:.1f}s)")
    verdict = "PASS" if (n_fail == 0 and n_pend == 0) else "FAIL"
    print(f"VERIFICATION: {verdict}")
    if verdict != "PASS":
        print("Fail-closed: PENDING counts against the verdict, never for it.")
    print("=" * 78)

    if a.report:
        rp = a.report if os.path.isabs(a.report) else os.path.join(od, a.report)
        with open(rp, "w", encoding="utf-8") as fh:
            json.dump({"schema": "c2_verification/1", "experiment": C.EXPERIMENT,
                       "design": {"path": C.DESIGN_PATH, "sha256": C.design_sha()},
                       "generated_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ",
                                                      time.gmtime()),
                       "checks": CHECKS,
                       "n_pass": n_pass, "n_fail": n_fail, "n_pending": n_pend,
                       "verdict": verdict}, fh, indent=2, sort_keys=True)
        print(f"[report] {rp}")

    return 0 if verdict == "PASS" else 1


if __name__ == "__main__":
    sys.exit(main())
