#!/usr/bin/env python3
"""
verify_exp10.py - Phase 19 / Experiment 10 independent verification. FAIL CLOSED.

Authoritative specification: PHASE_19_EXP10_DESIGN.md (FROZEN), SS 22.

This verifier RECOMPUTES rather than trusts. Fold means, confidence intervals,
the G0 gate, every contrast, the conditioning statistics and the zero-variance
policy are all derived again from the RAW artifacts (exp10_metrics.csv, the
frozen parquet, the frozen fold manifest) and only then compared against what
exp10_summary.json and exp10_acceptance.csv claim. A disagreement between the
raw data and the published summary is a FAIL, not a rounding note.

FAIL CLOSED (design SS 22):
    * any mismatch produces FAIL - never a silent accept
    * missing evidence produces PENDING - never PASS
    * a missing or non-boolean stored verdict for a complete arm produces FAIL,
      and the recomputed value is NEVER substituted as the expected one
    * path constants are normalised to forward slashes before any FROZEN_SHA
      lookup, and an unpinned path fails closed instead of raising KeyError
    * the process exit code is non-zero unless every check is PASS

Verification groups (design SS 22):
     1 frozen design SHA                     8 leakage
     2 frozen artifact SHAs                  9 G0 gate recomputation
     3 arm definitions                      10 P1/P2/P3/P4 recomputation
     4 fold counts and seeds                11 acceptance.csv agreement
     5 fold integrity                       12 thresholds and design pin
     6 conditioning recomputation           13 scope: no extra arm, no DP artifact
     7 zero-variance policy

Usage:
    python exp10_multimodal_conditioning/verify_exp10.py
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import sys
from typing import Any, Dict, List, Optional

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import exp10_common as C  # noqa: E402

PASS, FAIL, PENDING = "PASS", "FAIL", "PENDING"
VOID = "VOID"
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


def stored_bool(container: Any, key: str) -> Any:
    """Fetch a stored verdict with NO DEFAULT (design SS 22).

    Returns the raw value so the caller can distinguish 'absent' (None) and
    'present but not a boolean' from a genuine True/False. Defaulting to the
    recomputed value would make the comparison trivially true and turn a missing
    field into a silent PASS.
    """
    if not isinstance(container, dict):
        return None
    return container.get(key)


def main() -> int:
    ap = argparse.ArgumentParser(
        description="Phase 19 / Experiment 10 verification (fail closed)")
    ap.add_argument("--out-dir", default=os.path.join(C.REPO, C.OUT_DIRNAME))
    ap.add_argument("--parquet", default=os.path.join(C.REPO, C.MULTIMODAL_PARQUET))
    a = ap.parse_args()
    os.chdir(C.REPO)

    k = Checks()
    print("=" * 78)
    print("EXPERIMENT 10 VERIFICATION - FAIL CLOSED")
    print("=" * 78)

    summary = load_json(os.path.join(a.out_dir, "exp10_summary.json"))
    cond_manifest = load_json(os.path.join(a.out_dir,
                                           "exp10_conditioning_manifest.json"))
    receipt = load_json(os.path.join(a.out_dir, "exp10_receipt.json"))
    metrics_path = os.path.join(a.out_dir, "exp10_metrics.csv")
    acc_path = os.path.join(a.out_dir, "exp10_acceptance.csv")

    # ================= 1. frozen design SHA =================================
    got = C.sha256_file(C.DESIGN_PATH)
    k.ok(1, "frozen design SHA", got == C.DESIGN_SHA,
         f"{C.DESIGN_PATH}: got {got}, want {C.DESIGN_SHA}")

    # ================= 2. frozen artifact SHAs ==============================
    bad = []
    for p in C.FROZEN_SHA:
        want = C.frozen_sha_for(p)          # normalised lookup, never raw index
        g = C.sha256_file(p)
        if want is None:
            bad.append(f"{p}: NOT PINNED")
        elif g is None:
            bad.append(f"{p}: MISSING")
        elif g != want:
            bad.append(f"{p}: got {g} want {want}")
    k.ok(2, "frozen artifact SHAs", not bad,
         "all match" if not bad else "; ".join(bad))

    # ================= 3. arm definitions ===================================
    frozen_arms = {
        "M0": (None, None, False, 768, 109_680_132, 207),
        "M1": (154, 84, False, 1024, 109_763_494, 215),
        "M2": (154, 84, True, 1024, 109_763_494, 215),
    }
    k.ok(3, "exactly three pre-registered arms",
         tuple(C.ARMS) == ("M0", "M1", "M2"), str(list(C.ARMS)))
    k.ok(3, "arm specifications match design SS 6",
         all((C.ARM_SPEC[m]["audio_dim"], C.ARM_SPEC[m]["vision_dim"],
              C.ARM_SPEC[m]["conditioned"], C.ARM_SPEC[m]["fusion_in"],
              C.ARM_SPEC[m]["n_params"], C.ARM_SPEC[m]["n_keys"]) == v
             for m, v in frozen_arms.items()),
         "M0 207/109,680,132 - M1 and M2 215/109,763,494")
    k.ok(3, "M1 and M2 architecturally identical",
         C.ARM_SPEC["M1"]["n_params"] == C.ARM_SPEC["M2"]["n_params"]
         and C.ARM_SPEC["M1"]["n_keys"] == C.ARM_SPEC["M2"]["n_keys"]
         and C.ARM_SPEC["M1"]["fusion_in"] == C.ARM_SPEC["M2"]["fusion_in"],
         "same class, params, shapes; only feature values differ")

    # ================= 4/5. fold counts, seeds, fold integrity ==============
    man = load_json(C.FOLD_MANIFEST) or {}
    man_want = C.frozen_sha_for(C.FOLD_MANIFEST)
    k.ok(5, "fold manifest SHA is the frozen one",
         man_want is not None and C.sha256_file(C.FOLD_MANIFEST) == man_want,
         C.repo_key(C.FOLD_MANIFEST) if man_want is not None
         else f"{C.repo_key(C.FOLD_MANIFEST)} is not pinned in FROZEN_SHA")
    folds = man.get("folds", [])
    k.ok(5, "5 folds, 188 participants, 45/143",
         len(folds) == C.N_FOLDS and man.get("n_participants") == C.N_PARTICIPANTS
         and man.get("n_pos") == C.N_POSITIVE and man.get("n_neg") == C.N_NEGATIVE,
         f"{len(folds)} folds, {man.get('n_participants')} participants")
    k.ok(5, "fold ids are 1..5",
         sorted(int(f["fold"]) for f in folds) == list(C.FOLD_IDS),
         str(sorted(int(f["fold"]) for f in folds)))
    k.ok(5, "zero train/test overlap in every fold",
         all(not (set(map(int, f["train_ids"])) & set(map(int, f["test_ids"])))
             for f in folds), f"{len(folds)} folds checked")
    all_test = set()
    for f in folds:
        all_test |= set(map(int, f["test_ids"]))
    k.ok(5, "test sets partition all 188 exactly once",
         len(all_test) == C.N_PARTICIPANTS
         and sum(len(f["test_ids"]) for f in folds) == C.N_PARTICIPANTS,
         f"{len(all_test)} unique test ids")

    if not os.path.isfile(metrics_path):
        for g, n in ((4, "fold counts"), (6, "conditioning recomputation"),
                     (7, "zero-variance policy"), (8, "leakage"),
                     (9, "G0 gate"), (10, "P1/P2/P3/P4"), (11, "acceptance.csv")):
            k.add(g, n, PENDING, "exp10_metrics.csv not present")
        rows: List[Dict[str, str]] = []
    else:
        rows = read_csv_rows(metrics_path)
        k.ok(4, "75 fold-runs total", len(rows) == C.N_FOLDS * C.N_REPEATS * 3,
             f"{len(rows)} rows")
        for arm in C.ARMS:
            n = len([r for r in rows if r["arm"] == arm])
            if n == 0:
                k.add(4, f"fold-runs {arm}", PENDING, "arm not measured")
            else:
                k.ok(4, f"fold-runs {arm} = 25", n == C.N_FOLDS * C.N_REPEATS,
                     f"{n} fold-runs")
        k.ok(4, "seeds follow seed = 1000 + repeat",
             all(int(r["seed"]) == C.BASE_SEED + int(r["repeat"]) for r in rows),
             "checked on every fold-run")
        k.ok(4, "conditioned flag matches the arm specification",
             all(str(r["conditioned"]).strip().lower()
                 == str(C.ARM_SPEC[r["arm"]]["conditioned"]).lower()
                 for r in rows),
             "M0/M1 raw, M2 conditioned")
        k.ok(4, "recorded parameter counts match the frozen arms",
             all(int(r["n_params"]) == C.ARM_SPEC[r["arm"]]["n_params"]
                 for r in rows), "per fold-run")

    # ================= 6/7/8. conditioning recomputed from the parquet ======
    if cond_manifest is None:
        for g, n in ((6, "conditioning recomputation"),
                     (7, "zero-variance policy"), (8, "leakage")):
            k.add(g, n, PENDING, "exp10_conditioning_manifest.json not present")
    elif not os.path.isfile(a.parquet):
        for g, n in ((6, "conditioning recomputation"),
                     (7, "zero-variance policy"), (8, "leakage")):
            k.add(g, n, PENDING, f"{a.parquet} not present")
    else:
        k.ok(6, "conditioning parameters are the frozen ones",
             cond_manifest.get("ddof") == C.DDOF
             and cond_manifest.get("zero_variance_rule") == C.ZERO_VARIANCE_RULE
             and cond_manifest.get("epsilon") is None
             and cond_manifest.get("applies_to_arms") == [C.TREATMENT_ARM],
             f"ddof={cond_manifest.get('ddof')} "
             f"epsilon={cond_manifest.get('epsilon')!r} "
             f"arms={cond_manifest.get('applies_to_arms')}")

        import pandas as pd  # local import: only needed when artifacts exist
        df = pd.read_parquet(a.parquet)
        recs = []
        for _, r in df.iterrows():
            f = json.loads(r["features"]) if isinstance(r["features"], str) \
                else r["features"]
            recs.append({"participant_id": int(r["participant_id"]),
                         "features": f})
        by_pid = {r["participant_id"]: r for r in recs}
        A_all, V_all = C.extract_modality_matrices(recs)
        idx = {p: i for i, p in enumerate(r["participant_id"] for r in recs)}

        digest_ok, zero_ok, leak_ok = True, True, True
        details: List[str] = []
        for f in folds:
            fno = str(int(f["fold"]))
            entry = cond_manifest.get("folds", {}).get(fno)
            if not isinstance(entry, dict):
                digest_ok = False
                details.append(f"fold {fno}: no manifest entry")
                continue
            tr = [idx[int(i)] for i in f["train_ids"]]
            te = [idx[int(i)] for i in f["test_ids"]]
            for name, X in (("audio", A_all), ("vision", V_all)):
                cond = C.FoldConditioner(X[tr], name)
                rec = entry.get(name, {})
                if (rec.get("mu_sha256") != C.vector_digest(cond.mu)
                        or rec.get("sigma_sha256") != C.vector_digest(cond.sigma)):
                    digest_ok = False
                    details.append(f"fold {fno} {name}: digest mismatch")
                want_zero = (list(C.EXPECTED_AUDIO_ZERO_VARIANCE_IDX)
                             if name == "audio"
                             else list(C.EXPECTED_VISION_ZERO_VARIANCE_IDX))
                if (list(cond.zero_variance_idx) != want_zero
                        or rec.get("zero_variance_indices") != want_zero):
                    zero_ok = False
                    details.append(f"fold {fno} {name}: zero-variance set mismatch")
                if cond.zero_variance_idx:
                    zi = list(cond.zero_variance_idx)
                    if not (np.all(cond.apply(X[tr])[:, zi] == 0.0)
                            and np.all(cond.apply(X[te])[:, zi] == 0.0)):
                        zero_ok = False
                        details.append(f"fold {fno} {name}: zero-var cols not exactly 0")
                # design SS 12.3 - fold-local statistics must differ from global
                mu_g = X.mean(axis=0)
                sd_g = X.std(axis=0, ddof=C.DDOF)
                z_fold = cond.apply(X[te])
                z_glob = (X[te] - mu_g) / np.where(sd_g > 0.0, sd_g, 1.0)
                if np.allclose(z_fold, z_glob, rtol=0.0, atol=0.0):
                    leak_ok = False
                    details.append(f"fold {fno} {name}: statistics are NOT fold-local")
            if int(entry.get("n_train", -1)) != len(f["train_ids"]):
                digest_ok = False
                details.append(f"fold {fno}: n_train mismatch")
        k.ok(6, "conditioning mu/sigma digests recomputed (ddof=0, train only)",
             digest_ok, "; ".join(details[:4]) if details else
             f"{len(folds)} folds x 2 modalities re-derived and matched")
        k.ok(7, "zero-variance policy: 10 audio indices, conditioned to exactly 0",
             zero_ok,
             f"audio {list(C.EXPECTED_AUDIO_ZERO_VARIANCE_IDX)}, vision none")
        k.ok(8, "statistics are fold-local, not global (negative control)",
             leak_ok, "recomputed independently for every fold and modality")
        nc = (cond_manifest.get("negative_control") or {})
        k.ok(8, "runner recorded a passing negative control",
             nc.get("checked") is True and nc.get("fold_local_confirmed") is True,
             f"checked={nc.get('checked')} confirmed={nc.get('fold_local_confirmed')}")
        if receipt is None:
            k.add(8, "leakage controls recorded in the receipt", PENDING,
                  "exp10_receipt.json not present")
        else:
            lc = receipt.get("leakage_controls", {})
            k.ok(8, "leakage controls recorded in the receipt",
                 lc.get("deepcopy_used") is True
                 and lc.get("train_only_statistics") is True
                 and lc.get("negative_control_passed") is True
                 and lc.get("byte_identity_asserted") is True
                 and lc.get("checkpoint_reuse") is False,
                 json.dumps(lc, sort_keys=True))

    # ================= 9/10/11. gate, contrasts, acceptance =================
    if not rows:
        pass  # already recorded PENDING above
    else:
        def fmeans(arm: str, metric: str) -> Dict[int, float]:
            return C.fold_means([r for r in rows if r["arm"] == arm], arm, metric)

        complete = {a: len([r for r in rows if r["arm"] == a])
                    == C.N_FOLDS * C.N_REPEATS for a in C.ARMS}
        if not complete.get(C.CONTROL_ARM):
            k.add(9, "G0 gate", PENDING, f"{C.CONTROL_ARM} incomplete")
            gate_pass = None
        else:
            m0 = fmeans(C.CONTROL_ARM, C.PRIMARY_METRIC)
            st = C.fold_level_ci(list(m0.values()))
            lo, hi = C.ROC_AUC_CI
            gate_pass = bool(lo <= st["mean"] <= hi)
            k.ok(9, "M0 reproduces Baseline-CV (design SS 14 gate)", gate_pass,
                 f"M0 fold-level mean {st['mean']:.6f} vs CI [{lo:.6f}, {hi:.6f}]; "
                 f"reference {C.BASELINE_ROC_AUC_MEAN:.6f}")
            if summary is None:
                k.add(9, "summary G0 verdict matches the recomputation", PENDING,
                      "exp10_summary.json absent - nothing to cross-check")
            else:
                claimed = stored_bool(summary.get("g0_gate", {}), "passed")
                if not isinstance(claimed, bool):
                    k.add(9, "summary G0 verdict matches the recomputation", FAIL,
                          f"g0_gate.passed is {claimed!r}, not a boolean verdict "
                          f"(recomputed -> {gate_pass})")
                else:
                    k.ok(9, "summary G0 verdict matches the recomputation",
                         claimed == gate_pass,
                         f"summary {claimed} vs recomputed {gate_pass}")

        if not all(complete.values()):
            for g, n in ((10, "P1/P2/P3/P4"), (11, "acceptance.csv")):
                k.add(g, n, PENDING, "not every arm is complete")
        else:
            roc = {a: fmeans(a, C.PRIMARY_METRIC) for a in C.ARMS}
            mae = {a: fmeans(a, C.GUARD_METRIC) for a in C.ARMS}
            pr = {a: fmeans(a, C.SECONDARY_METRIC) for a in C.ARMS}
            p1 = C.paired_contrast(roc["M2"], roc["M1"])
            p3 = C.paired_contrast(roc["M0"], roc["M2"])
            p4 = C.paired_contrast(mae["M2"], mae["M0"])
            s1 = C.paired_contrast(pr["M2"], pr["M1"])
            m2 = C.fold_level_ci(list(roc["M2"].values()))
            rec = {
                "P1": bool(p1["mean"] >= C.ROC_AUC_MDE
                           and (p1["ci95_lo"] > 0.0 or p1["ci95_hi"] < 0.0)),
                "P2": bool(m2["ci95_lo"] > C.CHANCE_ROC_AUC
                           or m2["ci95_hi"] < C.CHANCE_ROC_AUC),
                "P3": bool(p3["mean"] <= C.ROC_AUC_MDE),
                "P4": bool(p4["mean"] <= C.MAE_MDE),
            }
            k.add(10, "S1 PR-AUC contrast recomputed (reported, not acceptance)",
                  PASS, f"mean {s1['mean']:+.6f} vs MDE {C.PR_AUC_MDE}")
            for name, val, detail in (
                    ("P1", rec["P1"], f"paired M2-M1 {p1['mean']:+.6f} "
                                      f"CI [{p1['ci95_lo']:+.6f}, {p1['ci95_hi']:+.6f}] "
                                      f"vs MDE {C.ROC_AUC_MDE}"),
                    ("P2", rec["P2"], f"M2 CI [{m2['ci95_lo']:.6f}, {m2['ci95_hi']:.6f}] "
                                      f"vs chance {C.CHANCE_ROC_AUC}"),
                    ("P3", rec["P3"], f"paired M0-M2 {p3['mean']:+.6f} "
                                      f"vs MDE {C.ROC_AUC_MDE} (adverse)"),
                    ("P4", rec["P4"], f"paired MAE M2-M0 {p4['mean']:+.6f} "
                                      f"vs MDE {C.MAE_MDE} (adverse)")):
                if summary is None:
                    k.add(10, f"{name} recomputed", PENDING,
                          f"exp10_summary.json absent - nothing to cross-check "
                          f"(independently recomputed -> {val}; {detail})")
                    continue
                claimed = stored_bool(summary.get("acceptance", {}), name)
                if gate_pass is False:
                    k.ok(10, f"{name} recorded VOID after a failed G0 gate",
                         claimed == VOID,
                         f"summary {claimed!r}; design SS 14 voids the "
                         "interpretation")
                elif not isinstance(claimed, bool):
                    k.add(10, f"{name} recomputed", FAIL,
                          f"acceptance.{name} is {claimed!r}, not a boolean "
                          f"verdict (independently recomputed -> {val})")
                else:
                    k.ok(10, f"{name} recomputed", claimed == val,
                         f"summary {claimed} vs recomputed {val}; {detail}")

            if not os.path.isfile(acc_path):
                k.add(11, "acceptance.csv agreement", PENDING,
                      "exp10_acceptance.csv not present")
            else:
                arows = {r["criterion"]: r for r in read_csv_rows(acc_path)}
                expected = {"G0", "P1a", "P1b", "P2", "P3", "P4", "S1"}
                k.ok(11, "acceptance.csv has the frozen criterion rows",
                     set(arows) == expected,
                     f"{sorted(arows)} vs {sorted(expected)}")
                pairs = [("P1a", bool(p1["mean"] >= C.ROC_AUC_MDE)),
                         ("P1b", bool(p1["ci95_lo"] > 0.0 or p1["ci95_hi"] < 0.0)),
                         ("P2", rec["P2"]),
                         ("P3", rec["P3"]), ("P4", rec["P4"])]
                agree = True
                bad_rows = []
                for cid, val in pairs:
                    cell = (arows.get(cid) or {}).get("pass")
                    if gate_pass is False:
                        if str(cell) != VOID:
                            agree = False
                            bad_rows.append(f"{cid}={cell!r} (expected VOID)")
                    elif str(cell).strip() not in ("True", "False"):
                        agree = False
                        bad_rows.append(f"{cid}={cell!r} not a boolean")
                    elif (str(cell).strip() == "True") != val:
                        agree = False
                        bad_rows.append(f"{cid}={cell} vs recomputed {val}")
                k.ok(11, "acceptance.csv agrees with the recomputation", agree,
                     "; ".join(bad_rows) if bad_rows else "all criterion rows match")

    # ================= 12. thresholds and design pin ========================
    if summary is None:
        k.add(12, "summary threshold audit", PENDING,
              "exp10_summary.json not present")
    else:
        th = summary.get("thresholds", {})
        k.ok(12, "thresholds in the summary are the frozen constants",
             abs(float(th.get("baseline_roc_auc_mean", -1))
                 - C.BASELINE_ROC_AUC_MEAN) <= 1e-15
             and [float(x) for x in th.get("roc_auc_ci", [])] == list(C.ROC_AUC_CI)
             and abs(float(th.get("roc_auc_mde", -1)) - C.ROC_AUC_MDE) <= 1e-15
             and abs(float(th.get("pr_auc_mde", -1)) - C.PR_AUC_MDE) <= 1e-15
             and abs(float(th.get("mae_mde", -1)) - C.MAE_MDE) <= 1e-15
             and float(th.get("chance_roc_auc", -1)) == C.CHANCE_ROC_AUC
             and abs(float(th.get("t_crit", -1)) - C.T_CRIT) <= 1e-15
             and int(th.get("df", -1)) == C.DF,
             json.dumps(th, sort_keys=True, default=float))
        k.ok(12, "summary pins the frozen design SHA",
             summary.get("design", {}).get("sha256") == C.DESIGN_SHA,
             str(summary.get("design", {}).get("sha256")))
        k.ok(12, "scope caveat and MDE-transfer assumption recorded verbatim",
             summary.get("scope_caveat") == C.SCOPE_CAVEAT
             and summary.get("mde_transfer_assumption") == C.MDE_TRANSFER_ASSUMPTION,
             "design SS 17, SS 30")

    # ================= 13. scope: no extra arm, no DP artifact ==============
    seen = {r["arm"] for r in rows} if rows else set()
    extra = sorted(seen - set(C.ARMS))
    k.ok(13, "no arm outside the pre-registered ladder", not extra,
         f"observed {sorted(seen) or 'none'}; frozen {list(C.ARMS)}")
    if summary is not None:
        k.ok(13, "summary declares no DP, masking, payload or federation",
             (summary.get("dp") or {}).get("applied") is False
             and (summary.get("masking") or {}).get("applied") is False
             and (summary.get("payload") or {}).get("measured") is False
             and (summary.get("federation") or {}).get("applied") is False,
             "design SS 5")
    forbidden_out = []
    if os.path.isdir(a.out_dir):
        for root, _, files in os.walk(a.out_dir):
            for f in files:
                if f.endswith((".pt", ".bin", ".safetensors", ".enc")):
                    forbidden_out.append(os.path.join(root, f))
    k.ok(13, "no model delta, checkpoint or encrypted payload in the output",
         not forbidden_out, str(forbidden_out) if forbidden_out else
         "output contains only metrics, manifests and receipts")

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
    vpath = os.path.join(a.out_dir, "exp10_verification.json")
    with open(vpath, "w", encoding="utf-8") as fh:
        json.dump({"schema": "exp10_verification/1",
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
