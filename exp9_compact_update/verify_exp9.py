#!/usr/bin/env python3
"""
verify_exp9.py - Phase 18 / Experiment 9 independent verification. FAIL CLOSED.

Authoritative specification: PHASE_18_EXP9_DESIGN.md (FROZEN), SS 28-SS 32.

This verifier RECOMPUTES rather than trusts. Every quantity it checks is derived
again from the raw artifacts (exp9_dp_metrics.csv, exp9_payload.json,
exp9_task_metrics.csv, exp9_mask_manifest.json) and compared against the frozen
constants and against what exp9_summary.json claims. A disagreement between the
raw data and the summary is a FAIL, not a rounding note.

FAIL CLOSED (implementation brief SS 13):
    * any mismatch produces FAIL - never a silent accept
    * missing evidence produces PENDING - never PASS
    * the process exit code is non-zero unless every check is PASS

Verified here:
    1  frozen design SHA
    2  frozen artifact SHAs
    3  k values
    4  nested masks
    5  public mask construction (rebuilt independently from key names + shapes)
    6  DP parameters
    7  seeds
    8  payload measurements
    9  SNR measurements
   10  fold counts
   11  K0 equivalence
   12  ROC-AUC CI criterion
   13  MDE criterion
   14  absence of post-hoc arm changes

Usage:
    python exp9_compact_update/verify_exp9.py
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import sys
from typing import Any, Dict, List, Optional

import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import exp9_common as C  # noqa: E402

PASS, FAIL, PENDING = "PASS", "FAIL", "PENDING"
TOL = 1e-9


class Checks:
    """Accumulator. Fail-closed: unknown is never treated as success."""

    def __init__(self) -> None:
        self.items: List[Dict[str, Any]] = []

    def add(self, group: int, name: str, status: str, detail: str = "") -> None:
        self.items.append({"group": group, "check": name, "status": status,
                           "detail": detail})

    def ok(self, group: int, name: str, condition: bool, detail: str = "") -> bool:
        self.add(group, name, PASS if condition else FAIL, detail)
        return bool(condition)

    @property
    def n_fail(self) -> int:
        return sum(1 for x in self.items if x["status"] == FAIL)

    @property
    def n_pending(self) -> int:
        return sum(1 for x in self.items if x["status"] == PENDING)

    @property
    def n_pass(self) -> int:
        return sum(1 for x in self.items if x["status"] == PASS)


def read_csv_rows(path: str) -> List[Dict[str, str]]:
    with open(path, "r", newline="", encoding="utf-8") as fh:
        return list(csv.DictReader(fh))


def load_json(path: str) -> Optional[Dict[str, Any]]:
    if not os.path.isfile(path):
        return None
    with open(path, "r", encoding="utf-8") as fh:
        return json.load(fh)


def repo_key(path: str) -> str:
    """Canonical repo-relative key for a C.FROZEN_SHA lookup.

    exp9_common builds its path constants with os.path.join, which yields
    BACKSLASHES on Windows (`trainer_outputs\\baseline_cv\\fold_manifest.json`),
    while C.FROZEN_SHA is keyed with forward-slash literals
    (`trainer_outputs/baseline_cv/fold_manifest.json`). Indexing the dict with an
    unnormalised constant therefore raises KeyError on Windows.

    Normalising here is PLATFORM-NEUTRAL: on POSIX os.sep is already "/", so the
    replacement is a no-op and Linux behaviour is unchanged. The frozen constants
    themselves are not touched - only the key used for the lookup.

    Applies to every repo-relative path constant that carries a directory
    component: C.FOLD_MANIFEST, C.BASELINE_SUMMARY, C.DELTA_PATH, C.OUT_DIRNAME.
    (C.TASK_PARQUET and C.DESIGN_PATH have no directory part and are unaffected.)
    """
    return path.replace("\\", "/")


def frozen_sha_for(path: str) -> Optional[str]:
    """The frozen SHA registered for a path, or None if it is not pinned.

    Returning None instead of raising keeps the caller FAIL-CLOSED: an unpinned
    or misspelt path becomes a recorded FAIL rather than an unhandled KeyError
    that would discard every check computed so far.
    """
    return C.FROZEN_SHA.get(repo_key(path))


def main() -> int:
    ap = argparse.ArgumentParser(
        description="Phase 18 / Experiment 9 verification (fail closed)")
    ap.add_argument("--out-dir", default=os.path.join(C.REPO, C.OUT_DIRNAME))
    ap.add_argument("--delta", default=os.path.join(C.REPO, C.DELTA_PATH))
    a = ap.parse_args()
    os.chdir(C.REPO)

    k = Checks()
    print("=" * 78)
    print("EXPERIMENT 9 VERIFICATION - FAIL CLOSED")
    print("=" * 78)

    # ================= 1. frozen design SHA =================================
    got = C.sha256_file(C.DESIGN_PATH)
    k.ok(1, "frozen design SHA", got == C.DESIGN_SHA,
         f"{C.DESIGN_PATH}: got {got}, want {C.DESIGN_SHA}")

    # ================= 2. frozen artifact SHAs ==============================
    bad = []
    for p, want in C.FROZEN_SHA.items():
        g = C.sha256_file(p)
        if g is None:
            bad.append(f"{p}: MISSING")
        elif g != want:
            bad.append(f"{p}: got {g} want {want}")
    k.ok(2, "frozen artifact SHAs", not bad,
         "all match" if not bad else "; ".join(bad))

    # ================= 5. public mask construction (rebuilt) ================
    # Rebuilt here from the frozen delta's KEY NAMES AND SHAPES ONLY, entirely
    # independently of whatever the runner recorded (design SS 6.3, SS 28.5).
    order = None
    if os.path.isfile(a.delta):
        sd = torch.load(a.delta, map_location="cpu", weights_only=False)
        numels = {kk: int(v.numel()) for kk, v in sd.items()}     # shapes only
        order = C.build_ordering(list(sd.keys()), numels)
        struct = C.verify_ordering(order)
        del sd
        k.ok(5, "ordering O rebuilt from key names + shapes",
             struct["d"] == C.EXPECTED_D and struct["n_keys"] == C.EXPECTED_N_KEYS,
             f"d={struct['d']:,} keys={struct['n_keys']}")
        k.ok(5, "group sizes match design SS 7",
             all(struct["group_params"][g] == v
                 for g, v in C.EXPECTED_GROUP_PARAMS.items()),
             json.dumps(struct["group_params"], sort_keys=True))
    else:
        k.add(5, "ordering O rebuilt from key names + shapes", PENDING,
              f"{a.delta} not present")

    mask_manifest = load_json(os.path.join(a.out_dir, "exp9_mask_manifest.json"))
    if mask_manifest is None:
        k.add(5, "mask manifest recorded by the runner", PENDING,
              "exp9_mask_manifest.json not present")
    else:
        k.ok(5, "mask declared non-magnitude-based",
             mask_manifest.get("magnitude_based") is False
             and mask_manifest.get("mechanism") == C.MECHANISM_NAME
             and mask_manifest.get("prohibited_inputs_used") == [],
             f"mechanism={mask_manifest.get('mechanism')!r}")
        if order is not None:
            recorded = [e["key"] for e in mask_manifest.get("ordering", [])]
            k.ok(5, "recorded ordering identical to the independent rebuild",
                 recorded == [o[0] for o in order],
                 f"{len(recorded)} keys recorded")

    # ================= 3. k values ==========================================
    frozen_k = {"K0": 109_680_132, "K1": 1_096_801, "K2": 295_681, "K3": 10_968}
    k.ok(3, "k ladder matches design SS 8", C.K_VALUES == frozen_k,
         json.dumps(C.K_VALUES, sort_keys=True))
    if mask_manifest is not None:
        k.ok(3, "k ladder recorded by the runner is unchanged",
             {kk: int(v) for kk, v in mask_manifest.get("k_values", {}).items()}
             == frozen_k, json.dumps(mask_manifest.get("k_values"), sort_keys=True))

    # ================= 4. nested masks ======================================
    if order is not None:
        nest = C.verify_nesting(order)
        ks = [C.K_VALUES[x] for x in ("K3", "K2", "K1", "K0")]
        k.ok(4, "S_K3 subset S_K2 subset S_K1 subset S_K0",
             nest["nested"] and all(ks[i] < ks[i + 1] for i in range(3)),
             f"ascending k = {ks}; prefixes of one ordering")
        if mask_manifest is not None:
            rec = mask_manifest.get("nesting", {}).get("arms", {})
            same = all(rec.get(arm, {}).get("boundary_key")
                       == nest["arms"][arm]["boundary_key"]
                       and rec.get(arm, {}).get("offset_in_key")
                       == nest["arms"][arm]["offset_in_key"] for arm in C.ARMS)
            k.ok(4, "mask boundaries match the independent rebuild", same,
                 json.dumps({x: nest["arms"][x]["boundary_key"] for x in C.ARMS}))
    else:
        k.add(4, "S_K3 subset S_K2 subset S_K1 subset S_K0", PENDING,
              "frozen delta not present")

    # ================= 6/7/9. DP parameters, seeds, SNR =====================
    dp_path = os.path.join(a.out_dir, "exp9_dp_metrics.csv")
    summary = load_json(os.path.join(a.out_dir, "exp9_summary.json"))
    if not os.path.isfile(dp_path):
        for g, n in ((6, "DP parameters"), (7, "seeds"), (9, "SNR measurements")):
            k.add(g, n, PENDING, "exp9_dp_metrics.csv not present")
        dp_rows: List[Dict[str, str]] = []
    else:
        dp_rows = read_csv_rows(dp_path)
        eps_recomputed = C.epsilon()
        k.ok(6, "eps recomputed by the frozen dp_agent",
             abs(eps_recomputed - C.EPS_PER_UPDATE_MAX) <= 1e-12,
             f"{eps_recomputed!r}")
        k.ok(6, "recorded eps identical for every arm and equal to the budget",
             all(abs(float(r["epsilon"]) - C.EPS_PER_UPDATE_MAX) <= 1e-12
                 for r in dp_rows),
             f"{len({r['epsilon'] for r in dp_rows})} distinct value(s)")
        k.ok(6, "clip_norm / noise_multiplier / sigma_eff / delta frozen",
             all(abs(float(r["clip_norm"]) - C.CLIP_NORM) <= TOL
                 and abs(float(r["noise_multiplier"]) - C.NOISE_MULTIPLIER) <= TOL
                 and abs(float(r["sigma_eff"]) - C.SIGMA_EFF) <= TOL
                 and abs(float(r["delta_dp"]) - C.DELTA_DP) <= TOL
                 for r in dp_rows),
             f"C={C.CLIP_NORM} nm={C.NOISE_MULTIPLIER} sigma_eff={C.SIGMA_EFF} "
             f"delta={C.DELTA_DP}")
        k.ok(6, "sensitivity equals clip_norm (design SS 12, not hardcoded)",
             all(abs(float(r["sensitivity"]) - C.CLIP_NORM) <= TOL for r in dp_rows),
             "Delta = C = 1.0")
        k.ok(6, "cumulative eps within budget",
             C.EPS_PER_UPDATE_MAX * C.N_UPDATES_PER_ROUND
             <= C.EPS_CUMULATIVE_MAX + 1e-9,
             f"{C.EPS_PER_UPDATE_MAX * C.N_UPDATES_PER_ROUND} <= "
             f"{C.EPS_CUMULATIVE_MAX}")

        seeds_by_arm = {arm: sorted(int(r["seed"]) for r in dp_rows
                                    if r["arm"] == arm) for arm in C.ARMS}
        present = {arm: s for arm, s in seeds_by_arm.items() if s}
        k.ok(7, "seeds are the frozen 1001..1005 in every measured arm",
             bool(present) and all(s == sorted(C.SEEDS) for s in present.values()),
             json.dumps(present))
        k.ok(7, "identical seeds across arms (design SS 26)",
             len({tuple(s) for s in present.values()}) <= 1,
             f"{len(present)} arm(s) measured")

        for arm in C.ARMS:
            vals = [float(r["snr"]) for r in dp_rows if r["arm"] == arm]
            if not vals:
                k.add(9, f"SNR {arm}", PENDING, "arm not measured")
                continue
            if len(vals) != C.N_REPETITIONS:
                k.add(9, f"SNR {arm}", FAIL,
                      f"{len(vals)} draws, expected {C.N_REPETITIONS}")
                continue
            m = C.mean(vals)
            an = C.analytical_snr(C.K_VALUES[arm])
            k.ok(9, f"SNR {arm}: k recorded matches the frozen ladder",
                 all(int(r["k"]) == C.K_VALUES[arm]
                     for r in dp_rows if r["arm"] == arm),
                 f"k={C.K_VALUES[arm]:,}")
            # A large gap between measurement and the closed form is an
            # IMPLEMENTATION FAULT, not a discovery (design SS 14).
            k.ok(9, f"SNR {arm}: measured agrees with sigma_eff*sqrt(k)",
                 abs(m - an) / an < 0.05,
                 f"mean {m:.6f} vs analytical {an:.6f} "
                 f"({100*(m-an)/an:+.4f} %)")
            if summary:
                claimed = summary.get("snr", {}).get(arm, {}).get("mean")
                if claimed is not None:
                    k.ok(9, f"SNR {arm}: summary matches the raw CSV",
                         abs(float(claimed) - m) <= 1e-9,
                         f"summary {claimed} vs recomputed {m}")
                b = summary.get("snr", {}).get(arm, {}).get("criterion_B_pass")
                if b is not None:
                    k.ok(9, f"criterion B {arm} recomputed",
                         bool(b) == bool(m < C.G3_SNR_MAX),
                         f"mean {m:.6f} vs G3 {C.G3_SNR_MAX}")

    # ================= 8. payload measurements ==============================
    pay = load_json(os.path.join(a.out_dir, "exp9_payload.json"))
    if pay is None:
        k.add(8, "payload measurements", PENDING, "exp9_payload.json not present")
    else:
        k.ok(8, "G2 threshold recorded is the frozen constant",
             int(pay.get("g2_threshold_bytes", -1)) == C.G2_PAYLOAD_MAX_BYTES,
             f"{pay.get('g2_threshold_bytes')} vs {C.G2_PAYLOAD_MAX_BYTES}")
        for arm in C.ARMS:
            p = pay.get("arms", {}).get(arm)
            if not p:
                k.add(8, f"payload {arm}", PENDING, "arm not measured")
                continue
            total = int(p["total_uploaded_bytes"])
            k.ok(8, f"payload {arm}: uploaded bytes equal the encrypted object",
                 total == int(p["encrypted_bytes"]),
                 f"{total:,} B in {p['n_chunks']} chunk(s)")
            k.ok(8, f"payload {arm}: chunking is 1 MB and covers every byte",
                 int(p["chunk_bytes"]) == C.CHUNK_BYTES
                 and int(p["n_chunks"]) == max(1, math.ceil(total / C.CHUNK_BYTES)),
                 f"{p['n_chunks']} chunk(s) of {p['chunk_bytes']:,} B")
            k.ok(8, f"payload {arm}: raw 4k not substituted for the measurement",
                 int(p["raw_4k_bytes"]) == 4 * C.K_VALUES[arm]
                 and total != int(p["raw_4k_bytes"]),
                 f"raw {p['raw_4k_bytes']:,} B vs measured {total:,} B")
            k.ok(8, f"criterion A {arm} recomputed",
                 bool(p["g2_pass"]) == (total <= C.G2_PAYLOAD_MAX_BYTES),
                 f"{total:,} vs {C.G2_PAYLOAD_MAX_BYTES:,}")

    # ================= 10-13. task half =====================================
    task_path = os.path.join(a.out_dir, "exp9_task_metrics.csv")
    if not os.path.isfile(task_path):
        for g, n in ((10, "fold counts"), (11, "K0 equivalence"),
                     (12, "ROC-AUC CI criterion"), (13, "MDE criterion")):
            k.add(g, n, PENDING, "task half not executed")
        task_rows: List[Dict[str, str]] = []
    else:
        task_rows = read_csv_rows(task_path)
        man = load_json(C.FOLD_MANIFEST) or {}
        fm_want = frozen_sha_for(C.FOLD_MANIFEST)
        k.ok(10, "fold manifest SHA is the frozen one",
             fm_want is not None and C.sha256_file(C.FOLD_MANIFEST) == fm_want,
             repo_key(C.FOLD_MANIFEST) if fm_want is not None
             else f"{repo_key(C.FOLD_MANIFEST)} is not pinned in FROZEN_SHA")
        folds = man.get("folds", [])
        k.ok(10, "5 folds, 188 participants, 45/143 split",
             len(folds) == C.N_FOLDS
             and man.get("n_participants") == C.N_PARTICIPANTS
             and man.get("n_pos") == C.N_POSITIVE
             and man.get("n_neg") == C.N_NEGATIVE,
             f"{len(folds)} folds, {man.get('n_participants')} participants")
        k.ok(10, "zero train/test overlap in every fold",
             all(not (set(map(int, f["train_ids"])) & set(map(int, f["test_ids"])))
                 for f in folds),
             f"{len(folds)} folds checked")
        for arm in C.ARMS:
            n = len([r for r in task_rows if r["arm"] == arm])
            if n == 0:
                k.add(10, f"fold-runs {arm}", PENDING, "arm not measured")
            else:
                k.ok(10, f"fold-runs {arm} = 25",
                     n == C.N_FOLDS * C.N_REPEATS,
                     f"{n} fold-runs")
        k.ok(10, "seeds follow seed = 1000 + repeat",
             all(int(r["seed"]) == C.BASE_SEED_TASK + int(r["repeat"])
                 for r in task_rows), "checked on every fold-run")

        def fold_means(arm: str) -> Dict[int, float]:
            b: Dict[int, List[float]] = {}
            for r in task_rows:
                if r["arm"] == arm:
                    b.setdefault(int(r["fold"]), []).append(float(r[C.TASK_METRIC]))
            return {f: C.mean(v) for f, v in sorted(b.items())}

        k0 = fold_means(C.CONTROL_ARM)
        if len(k0) != C.N_FOLDS:
            k.add(11, "K0 equivalence", PENDING,
                  f"K0 has {len(k0)}/{C.N_FOLDS} folds")
            for g, n in ((12, "ROC-AUC CI criterion"), (13, "MDE criterion")):
                k.add(g, n, PENDING, "K0 incomplete")
        else:
            st = C.fold_level_ci(list(k0.values()))
            lo, hi = C.ROC_AUC_CI
            gate = lo <= st["mean"] <= hi
            k.ok(11, "K0 reproduces Baseline-CV (design SS 22 gate)", gate,
                 f"K0 fold-level mean {st['mean']:.6f} vs CI "
                 f"[{lo:.6f}, {hi:.6f}]; reference "
                 f"{C.BASELINE_ROC_AUC_MEAN:.6f}")
            if summary:
                claimed = summary.get("k0_gate", {}).get("passed")
                if claimed is not None:
                    k.ok(11, "summary K0 verdict matches the recomputation",
                         bool(claimed) == bool(gate), f"summary says {claimed}")
            for arm in C.ARMS:
                fm = fold_means(arm)
                if len(fm) != C.N_FOLDS:
                    k.add(12, f"CI criterion {arm}", PENDING, "arm incomplete")
                    k.add(13, f"MDE criterion {arm}", PENDING, "arm incomplete")
                    continue
                s = C.fold_level_ci(list(fm.values()))
                inside = lo <= s["mean"] <= hi
                deg = C.mean([k0[f] - fm[f] for f in sorted(fm)])
                if not gate:
                    k.add(12, f"CI criterion {arm}", FAIL,
                          "K0 gate failed; criterion C is VOID (design SS 22)")
                    k.add(13, f"MDE criterion {arm}", FAIL,
                          "K0 gate failed; criterion C is VOID (design SS 22)")
                    continue
                mde_ok = deg <= C.ROC_AUC_MDE
                # The stored verdict is fetched with NO DEFAULT. Defaulting it to
                # the recomputed value would make the comparison trivially true and
                # turn a missing field into a silent PASS - the cross-check between
                # the raw fold data and the published summary exists precisely to
                # catch that case.
                arm_summary = (summary or {}).get("task", {}).get(arm, {})
                claimed_ci = arm_summary.get("criterion_C1_ci_pass")
                claimed_mde = arm_summary.get("criterion_C2_mde_pass")

                if summary is None:
                    # Missing evidence is PENDING, never PASS: there is no stored
                    # verdict to agree or disagree with.
                    k.add(12, f"CI criterion {arm} recomputed", PENDING,
                          "exp9_summary.json absent - nothing to cross-check "
                          f"(independently recomputed: mean {s['mean']:.6f} in "
                          f"[{lo:.6f}, {hi:.6f}] -> {inside})")
                elif not isinstance(claimed_ci, bool):
                    # The arm is complete and the K0 gate passed, so the summary
                    # owes a boolean verdict here. Absent or non-boolean is an
                    # inconsistency between the summary and its own inputs.
                    k.add(12, f"CI criterion {arm} recomputed", FAIL,
                          f"arm is complete ({C.N_FOLDS} folds) but the summary's "
                          f"criterion_C1_ci_pass is {claimed_ci!r}, not a boolean "
                          f"verdict (independently recomputed -> {inside})")
                else:
                    k.ok(12, f"CI criterion {arm} recomputed",
                         claimed_ci == inside,
                         f"summary {claimed_ci} vs recomputed {inside}; "
                         f"mean {s['mean']:.6f} in [{lo:.6f}, {hi:.6f}]")

                if summary is None:
                    k.add(13, f"MDE criterion {arm} recomputed", PENDING,
                          "exp9_summary.json absent - nothing to cross-check "
                          f"(independently recomputed: paired degradation "
                          f"{deg:+.6f} vs MDE {C.ROC_AUC_MDE} -> {mde_ok})")
                elif not isinstance(claimed_mde, bool):
                    k.add(13, f"MDE criterion {arm} recomputed", FAIL,
                          f"arm is complete ({C.N_FOLDS} folds) but the summary's "
                          f"criterion_C2_mde_pass is {claimed_mde!r}, not a boolean "
                          f"verdict (independently recomputed -> {mde_ok})")
                else:
                    k.ok(13, f"MDE criterion {arm} recomputed",
                         claimed_mde == mde_ok,
                         f"summary {claimed_mde} vs recomputed {mde_ok}; "
                         f"paired degradation {deg:+.6f} vs MDE {C.ROC_AUC_MDE}")

    # ================= 14. absence of post-hoc arm changes ==================
    seen_arms = set()
    for rows, key in ((dp_rows, "arm"), (task_rows, "arm")):
        seen_arms |= {r[key] for r in rows}
    if pay:
        seen_arms |= set(pay.get("arms", {}))
    extra = sorted(seen_arms - set(C.ARMS))
    k.ok(14, "no arm outside the pre-registered ladder", not extra,
         f"observed {sorted(seen_arms) or 'none'}; frozen {list(C.ARMS)}")
    if dp_rows:
        k.ok(14, "recorded k values match the frozen ladder exactly",
             all(int(r["k"]) == C.K_VALUES[r["arm"]] for r in dp_rows),
             "DP metrics")
    if task_rows:
        k.ok(14, "task k values match the frozen ladder exactly",
             all(int(r["k"]) == C.K_VALUES[r["arm"]] for r in task_rows),
             "task metrics")
    if summary:
        th = summary.get("thresholds", {})
        k.ok(14, "thresholds in the summary are the frozen constants",
             int(th.get("G2_payload_bytes", -1)) == C.G2_PAYLOAD_MAX_BYTES
             and abs(float(th.get("G3_snr", -1)) - C.G3_SNR_MAX) <= 1e-12
             and [float(x) for x in th.get("roc_auc_ci", [])] == list(C.ROC_AUC_CI)
             and abs(float(th.get("roc_auc_mde", -1)) - C.ROC_AUC_MDE) <= 1e-15,
             json.dumps(th, sort_keys=True, default=float))
        k.ok(14, "summary pins the frozen design SHA",
             summary.get("design", {}).get("sha256") == C.DESIGN_SHA,
             str(summary.get("design", {}).get("sha256")))
    else:
        k.add(14, "summary threshold audit", PENDING,
              "exp9_summary.json not present")

    # ================= report ===============================================
    print()
    for it in k.items:
        mark = {PASS: "ok ", FAIL: "FAIL", PENDING: "..."}[it["status"]]
        print(f"  [{mark}] ({it['group']:>2}) {it['check']}")
        if it["detail"]:
            print(f"         {it['detail']}")

    overall = FAIL if k.n_fail else (PENDING if k.n_pending else PASS)
    print("-" * 78)
    print(f"checks: {k.n_pass} PASS, {k.n_fail} FAIL, {k.n_pending} PENDING")
    print(f"VERIFICATION: {overall}"
          + ("  (fail closed: PENDING is never PASS)" if overall == PENDING else ""))

    os.makedirs(a.out_dir, exist_ok=True)
    vpath = os.path.join(a.out_dir, "exp9_verification.json")
    with open(vpath, "w", encoding="utf-8") as fh:
        json.dump({"schema": "exp9_verification/1",
                   "experiment": C.EXPERIMENT,
                   "design": {"path": C.DESIGN_PATH, "sha256": C.DESIGN_SHA},
                   "fail_closed": True,
                   "n_pass": k.n_pass, "n_fail": k.n_fail,
                   "n_pending": k.n_pending,
                   "overall": overall, "checks": k.items}, fh,
                  indent=2, sort_keys=True)
    print(f"[DONE] -> {vpath}")
    return 0 if overall == PASS else 1


if __name__ == "__main__":
    sys.exit(main())
