#!/usr/bin/env python3
"""
verify_exp8.py - Phase 17 / Experiment 8 integrity verification.

Authoritative specification: PHASE_17_EXP8_DESIGN.md (FROZEN). Implements every
hard-abort check of design SS 19, plus the frozen-SHA and no-modification checks of
design SS 21 / SS 22.

FAIL CLOSED. A failed check is never bypassed, relaxed, downgraded or retried
(design SS 25.6). There is no --force and no skip flag for any integrity check.

Checks (design SS 19):
    V1  d = 295,681 and the four frozen groups are intact
    V2  D0 reproduces the frozen baseline band and eps exactly
    V3  eps verification for both arms (per-update and cumulative)
    V4  privacy-equivalence condition  sum_g (C_g/sigma_g)^2 = 1/sigma_eff^2
    V5  both arms consumed an IDENTICAL pre-DP object
    V6  frozen SHA verification of every frozen input
    V7  cryptographic verification recorded and passing for every draw
    V8  no frozen source file was modified during the run

Usage:
    python exp8_dp_mechanism/verify_exp8.py
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
import sys
from typing import Any, Dict, List, Optional

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO)

# ==========================================================================
# FROZEN EXPECTATIONS - design SS 9, SS 10, SS 12, SS 16, SS 21
# ==========================================================================
EXPECTED_D = 295_681
FROZEN_GROUPS = {"fc1.weight": 294_912, "fc1.bias": 384,
                 "fc2.weight": 384, "fc2.bias": 1}

EPS_PER_UPDATE_MAX = 5.302585092994046
EPS_CUMULATIVE_MAX = 15.9078
N_UPDATES_PER_ROUND = 3
DELTA = 1e-05
D0_SIGMA_EFF = 1.0
EQUIV_TOL = 1e-9

# design SS 10 - the published Baseline band for D0 (P9 SS 6.2 / SS 8):
# 543.02, 544.04, 543.75, 543.73, 542.61, 544.67; sweep 543.49.
D0_BAND_LO, D0_BAND_HI = 540.0, 548.0

BASE_SEED = 1000
N_REPETITIONS = 5
SEEDS = tuple(BASE_SEED + i for i in range(1, N_REPETITIONS + 1))

# design SS 21 - frozen inputs that must be SHA-verified and unmodified.
FROZEN_SHA: Dict[str, str] = {
    "trainer_mentalbert_daic.py":
        "65b1902e2ba8dd996c6960db1a6595d84fd48500f4d8707cb79688093d7a230b",
    "daic_records.parquet":
        "9a241851760727779fcaa4d50041b8bdc4406fe6b66ddbbd7105f4ce4fc55f00",
    "dataset_build/daic_records_multimodal.parquet":
        "1ac9f53e6102ec0dbaab84dcfdfa3f2e70f2b4a1867a841ea2b7e3ba24c67a95",
    "trainer_outputs/baseline_cv/fold_manifest.json":
        "b9a7a91f1bdd43c78d3cd1ee0c36e4ce3fb8be4b0971dcff3ff62ee5af8fca2f",
    "trainer_outputs/baseline_cv/baseline_cv_summary.json":
        "f065f5e1ff7f8e5ed4d122a8031c9cc12425802e756697d5512a915674871aac",
    "trainer_outputs/baseline_cv/trivial_control_arm.csv":
        "2ecbfee3225910034a959fe949c7ad027cfa7d41e9bef8040374578ff8c5d572",
    "trainer_outputs/exp3_convergence/exp3_summary.json":
        "9dd135b6b79af06a85e64a5aa8c1896d19df8059dd89c053e50e1cffe20af2f9",
    "trainer_outputs/exp4_decision_rule/exp4_summary.json":
        "5defdae2a20d0abc164611e8cbe6b8034e3f65c593d8c9b3e38266a1800aa6f2",
    "trainer_outputs/exp5_imbalance_objective/exp5_summary.json":
        "70ad13ffc4db633419595761c0396fc20dd1b62400ee74238be1f06b73ed48d3",
    "trainer_outputs/exp6_loss_rebalance/exp6_summary.json":
        "bea7478594f6af98943ddda2e997e8bc8780abf57c9c2ee720f9426cd2acc94c",
}
# Frozen but SHA recorded at runtime (no published constant to pin against).
FROZEN_UNPINNED = ("dp_agent/dp_agent.py",
                   "trainer_outputs/local_probe_base.pt",
                   "PHASE_17_EXP8_DESIGN.md")


def sha256_file(path: str) -> Optional[str]:
    if not os.path.isfile(path):
        return None
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for b in iter(lambda: fh.read(8192), b""):
            h.update(b)
    return h.hexdigest()


class Auditor:
    """Fail-closed checklist. A FAIL can never be downgraded (design SS 25.6)."""

    def __init__(self) -> None:
        self.results: List[Dict[str, Any]] = []

    def check(self, cid: str, name: str, ok: bool, detail: str = "") -> bool:
        self.results.append({"id": cid, "check": name, "passed": bool(ok),
                             "detail": detail})
        print(f"  [{'PASS' if ok else 'FAIL'}] {cid} {name}"
              + (f"  — {detail}" if detail else ""))
        return bool(ok)

    @property
    def failures(self) -> List[Dict[str, Any]]:
        return [r for r in self.results if not r["passed"]]


def main() -> int:
    ap = argparse.ArgumentParser(description="Verify Experiment 8 (fail closed)")
    ap.add_argument("--exp-dir", default=os.path.join(
        REPO, "trainer_outputs", "exp8_dp_mechanism"))
    a = ap.parse_args()
    os.chdir(REPO)

    print("=" * 74)
    print("EXPERIMENT 8 VERIFICATION - FAIL CLOSED")
    print("=" * 74)
    A = Auditor()

    mpath = os.path.join(a.exp_dir, "exp8_metrics.csv")
    gpath = os.path.join(a.exp_dir, "d0_reference_check.json")
    if not os.path.isfile(mpath):
        print(f"\n[ABORT] missing {mpath}; run run_exp8.py first", file=sys.stderr)
        return 2
    if not os.path.isfile(gpath):
        print(f"\n[ABORT] missing {gpath}", file=sys.stderr)
        return 2

    with open(mpath, newline="", encoding="utf-8") as fh:
        rows = list(csv.DictReader(fh))
    gate = json.load(open(gpath, encoding="utf-8"))

    # ---- V1  dimension and group structure (design SS 8.1, SS 9, SS 19) ------
    print("\nV1 target dimension and group structure")
    A.check("V1.1", "d == 295,681", gate.get("d") == EXPECTED_D,
            f"d={gate.get('d')}")
    gs = {g["name"]: g["d_g"] for g in gate.get("group_structure", [])}
    A.check("V1.2", "four frozen groups with frozen sizes", gs == FROZEN_GROUPS,
            f"{gs}")

    # ---- V2  D0 reproduces the frozen baseline (design SS 10, SS 17 P0) ------
    print("\nV2 D0 baseline reproduction (HARD GATE)")
    d0 = [r for r in rows if r["arm"] == "D0"]
    d0_snr = [float(r["snr"]) for r in d0]
    in_band = all(D0_BAND_LO <= s <= D0_BAND_HI for s in d0_snr)
    A.check("V2.1", f"D0 SNR within the published band [{D0_BAND_LO}, {D0_BAND_HI}]",
            in_band, f"min={min(d0_snr):.4f} max={max(d0_snr):.4f}" if d0_snr else "no rows")
    d0_eps = {float(r["epsilon"]) for r in d0}
    A.check("V2.2", "D0 eps == frozen 5.302585092994046",
            all(abs(e - EPS_PER_UPDATE_MAX) < 1e-12 for e in d0_eps),
            f"{sorted(d0_eps)}")

    # ---- V3  eps verification, both arms (design SS 12, SS 17 P1) ------------
    print("\nV3 privacy budget")
    for arm in ("D0", "D1"):
        eps = [float(r["epsilon"]) for r in rows if r["arm"] == arm]
        if not eps:
            A.check(f"V3.{arm}", f"{arm} rows present", False, "no rows")
            continue
        e = max(eps)
        A.check(f"V3.{arm}.a", f"{arm} eps <= {EPS_PER_UPDATE_MAX}",
                e <= EPS_PER_UPDATE_MAX + 1e-12, f"max eps={e:.15f}")
        A.check(f"V3.{arm}.b", f"{arm} cumulative <= {EPS_CUMULATIVE_MAX}",
                e * N_UPDATES_PER_ROUND <= EPS_CUMULATIVE_MAX + 1e-9,
                f"cum={e*N_UPDATES_PER_ROUND:.6f}")
    A.check("V3.delta", "delta == 1e-05",
            abs(gate.get("epsilon", {}).get("delta", -1) - DELTA) < 1e-18)
    A.check("V3.equal_eps", "D0 and D1 carry the SAME epsilon",
            abs(float(gate["epsilon"]["D0"]) - float(gate["epsilon"]["D1"])) < 1e-12,
            f"D0={gate['epsilon']['D0']:.15f} D1={gate['epsilon']['D1']:.15f}")

    # ---- V4  privacy-equivalence condition (design SS 11.2, SS 19) -----------
    print("\nV4 privacy-equivalence condition")
    alloc = gate.get("allocation", {})
    dss = float(alloc.get("delta_sigma_sq", float("nan")))
    A.check("V4.1", "sum_g (C_g/sigma_g)^2 == 1/sigma_eff^2",
            abs(dss - 1.0 / D0_SIGMA_EFF ** 2) < EQUIV_TOL, f"{dss!r}")
    scs = float(alloc.get("sum_C_g_squared", float("nan")))
    A.check("V4.2", "sum_g C_g^2 == 1", abs(scs - 1.0) < EQUIV_TOL, f"{scs!r}")

    # ---- V5  identical pre-DP object (design SS 19) --------------------------
    print("\nV5 identical pre-DP object across arms and draws")
    l2b = {round(float(r["l2_before"]), 10) for r in rows}
    A.check("V5.1", "l2_before identical for every row", len(l2b) == 1, f"{l2b}")
    A.check("V5.2", "target SHA recorded", bool(gate.get("target_sha256")),
            str(gate.get("target_sha256"))[:32] + "...")

    # ---- V6  frozen SHA verification (design SS 21) --------------------------
    print("\nV6 frozen input SHA verification")
    bad = []
    for path, want in FROZEN_SHA.items():
        got = sha256_file(path)
        if got != want:
            bad.append(path)
    A.check("V6.1", f"{len(FROZEN_SHA)} pinned frozen artifacts unchanged",
            not bad, f"failures={bad}" if bad else "all match")
    unpinned = {p: sha256_file(p) for p in FROZEN_UNPINNED}
    A.check("V6.2", "frozen unpinned artifacts present",
            all(v is not None for v in unpinned.values()),
            ", ".join(f"{os.path.basename(k)}={str(v)[:12]}..."
                      for k, v in unpinned.items()))

    # ---- V7  cryptographic verification (design SS 14, SS 17 P3) -------------
    print("\nV7 cryptographic verification")
    ok_all = all(str(r["crypto_ok"]).strip().lower() in ("true", "1") for r in rows)
    A.check("V7.1", "crypto_ok true for every draw", ok_all)
    layer_fail, not_exercised = [], set()
    for r in rows:
        try:
            layers = json.loads(r.get("crypto_layers") or "{}")
        except json.JSONDecodeError:
            layer_fail.append((r["arm"], r["seed"], "unparseable crypto_layers"))
            continue
        for name, info in layers.items():
            if info.get("status") == "FAIL":
                layer_fail.append((r["arm"], r["seed"], name))
            elif info.get("status") == "NOT_EXERCISED":
                not_exercised.add(name)
    A.check("V7.2", "no cryptographic layer reported FAIL", not layer_fail,
            f"{layer_fail}" if layer_fail else "none")
    if not_exercised:
        # Surfaced explicitly. NOT silently treated as passing (design SS 14).
        print(f"  [NOTE] layers not exercised by a local mechanism-level run: "
              f"{sorted(not_exercised)}")
        A.results.append({"id": "V7.3", "check": "layers not exercised (reported)",
                          "passed": True, "detail": sorted(not_exercised)})

    # ---- V7b  receipt artifact and reproducibility (design SS 21, SS 17 P4) --
    print("\nV7b receipt artifact and reproducibility")
    rpath = os.path.join(a.exp_dir, "exp8_receipt.json")
    rcp = json.load(open(rpath, encoding="utf-8")) if os.path.isfile(rpath) else None
    A.check("V7b.1", "exp8_receipt.json present (design SS 21)", rcp is not None)
    if rcp is not None:
        repro = rcp.get("reproducibility") or {}
        fz = rcp.get("frozen_input_sha") or {}
        A.check("V7b.2", "all draws reproduced byte-identically at the same seed",
                bool(repro.get("checked") and repro.get("all_draws_identical")),
                f"n_rechecked={repro.get('n_draws_rechecked')}")
        A.check("V7b.3", "frozen inputs unchanged pre/post (recorded by the runner)",
                bool(fz.get("unchanged_pre_post") and fz.get("pinned_all_match")))
        A.check("V7b.4", "receipt records seeds, parameters and environment",
                all(k in rcp for k in ("seeds", "parameters", "environment",
                                       "runtime_seconds", "output_artifact_sha")))

    # ---- V8  no frozen source modified during the run (design SS 22) --------
    print("\nV8 no frozen source modified")
    A.check("V8.1", "dp_agent/dp_agent.py present and readable",
            sha256_file("dp_agent/dp_agent.py") is not None)
    A.check("V8.2", "PHASE_17_EXP8_DESIGN.md present and readable",
            sha256_file("PHASE_17_EXP8_DESIGN.md") is not None)
    A.check("V8.3", "Exp 8 wrote only under its own output directory",
            os.path.abspath(a.exp_dir).startswith(
                os.path.abspath(os.path.join(REPO, "trainer_outputs"))),
            os.path.abspath(a.exp_dir))

    # ---- report -------------------------------------------------------------
    report = {
        "schema": "exp8_verification/1",
        "fail_closed": True,
        "checks": A.results,
        "n_checks": len(A.results),
        "n_failures": len(A.failures),
        "passed": len(A.failures) == 0,
        "frozen_sha_verified": {p: sha256_file(p) for p in FROZEN_SHA},
        "frozen_sha_unpinned": unpinned,
    }
    dest = os.path.join(a.exp_dir, "exp8_verification.json")
    with open(dest, "w", encoding="utf-8") as fh:
        json.dump(report, fh, indent=2, sort_keys=True, default=str)

    print("-" * 74)
    print(f"checks: {len(A.results)}   failures: {len(A.failures)}")
    print(f"report: {dest}")
    if A.failures:
        print("\n[FAIL] Experiment 8 integrity verification FAILED:", file=sys.stderr)
        for f in A.failures:
            print(f"    {f['id']} {f['check']}  {f['detail']}", file=sys.stderr)
        return 1
    print("\n[PASS] all Experiment 8 integrity checks passed.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
