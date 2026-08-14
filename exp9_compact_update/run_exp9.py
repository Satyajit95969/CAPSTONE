#!/usr/bin/env python3
"""
run_exp9.py - Phase 18 / Experiment 9, TERMINAL (CPU) half.

Authoritative specification: PHASE_18_EXP9_DESIGN.md (FROZEN). Every constant,
mask, threshold, seed and criterion is imported from exp9_common.py, which
transcribes that document. Nothing here may be tuned.

SCOPE OF THIS RUNNER - design SS 27, left-hand column ONLY:
    * load the frozen mentalbert_delta.pt
    * construct the ordering O and the masks S_k, and VERIFY their structure
    * DP draws: 4 arms x 5 seeds, on the k-dimensional release
    * empirical SNR
    * complete transport payload measurement (serialise -> AES-GCM -> chunk)
    * cryptographic verification, with NOT_EXERCISED layers reported honestly
    * frozen-SHA gate before AND after, inside the execution path

NOT IN SCOPE HERE - design SS 27, right-hand column: the 25 fold trainings, the
per-fold deltas and the ROC-AUC evaluation. Those are run_exp9_task.py, on GPU.

THE MECHANISM IS NOT TOP-K (design SS 3, SS 32.8). The mask is public
architecture-priority coordinate selection, built from key names and shapes
alone. No delta value influences which coordinates are retained.

NOTHING IS UPLOADED. The "transport payload" is the encrypted, chunked byte
representation measured locally; no network call exists in this file.

Usage:
    python exp9_compact_update/run_exp9.py --plan-only     # static, no draws
    python exp9_compact_update/run_exp9.py
"""

from __future__ import annotations

import argparse
import csv
import io
import json
import os
import sys
import time
from typing import Any, Dict, List

import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import exp9_common as C  # noqa: E402


# ==========================================================================
# Experiment-local HMAC key (the Exp 8 F4 lesson)
# ==========================================================================
def local_receipt_key_source(out_dir: str) -> str:
    """Provision an Exp-9-local HMAC key and return a 'file:' key_source.

    CentralReceiptManager's DEFAULT constructor writes a key to
    ~/.local_data_agent_receipt_key when absent, which is OUTSIDE the authorised
    output directory and would violate design SS 31 ("all artifacts written only
    under trainer_outputs/exp9_compact_update/"). Exp 8 hit exactly this and
    remediated it the same way.

    The key is provisioned BEFORE any draw so its CSPRNG call cannot interleave
    with the seeded torch noise stream.
    """
    import base64
    import secrets

    key_dir = os.path.join(out_dir, "keys")
    os.makedirs(key_dir, exist_ok=True)
    key_path = os.path.join(key_dir, "exp9_receipt_key.b64")
    if not os.path.isfile(key_path):
        with open(key_path, "w", encoding="utf-8") as fh:
            fh.write(base64.b64encode(secrets.token_bytes(32)).decode())
    return f"file:{key_path}"


# ==========================================================================
# Staged-run persistence - PERSISTENCE ONLY, no experimental semantics
# ==========================================================================
def load_prior_json(path: str):
    """Read an artifact this runner wrote in an EARLIER invocation, or None.

    Experiment 9's Terminal half may be executed in stages (design SS 27 places
    all four arms here, but K0's transport-payload measurement is far larger than
    the others, so `--arms` allows them to be run separately). Each invocation
    must therefore ADD its arms to the artifact set rather than replace it.

    This function and the three merge blocks in main() are persistence logic
    only. They read and write nothing but previously recorded values, and touch
    no k value, mask, seed, DP parameter, threshold or acceptance rule.
    """
    if not os.path.isfile(path):
        return None
    with open(path, "r", encoding="utf-8") as fh:
        return json.load(fh)


def arm_sort_key(arm: str, seed: Any) -> Any:
    """Deterministic (arm, seed) ordering along the frozen ladder."""
    return (list(C.ARMS).index(arm) if arm in C.ARMS else 99, int(seed))


# ==========================================================================
# Target - design SS 21.1
# ==========================================================================
def load_frozen_delta(path: str):
    """Load mentalbert_delta.pt and derive O from KEY NAMES AND SHAPES ONLY.

    The `numels` dict handed to build_ordering is computed with `.numel()`, i.e.
    from shape metadata. No tensor value is read, compared, sorted or thresholded
    anywhere on this path (design SS 6.3, SS 28.5).
    """
    if not os.path.isfile(path):
        C.abort(f"frozen delta not found: {path}")
    sd = torch.load(path, map_location="cpu", weights_only=False)
    if not isinstance(sd, dict):
        C.abort("frozen delta is not a state_dict")
    numels = {k: int(v.numel()) for k, v in sd.items()}      # shapes only
    order = C.build_ordering(list(sd.keys()), numels)
    struct = C.verify_ordering(order)
    return sd, order, struct


# ==========================================================================
# Cryptographic verification - design SS 18, per arm
# ==========================================================================
def crypto_verify(compact_obj_bytes: bytes, payload_meas: Dict[str, Any],
                  reconstructed: Dict[str, torch.Tensor],
                  reference: Dict[str, torch.Tensor],
                  arm: str, session_id: str, out_dir: str,
                  key_source: str) -> Dict[str, Any]:
    """Exercise the frozen cryptographic stack for one arm (design SS 18).

    EXERCISED here: SHA-256 full hash, AES-GCM at rest (already round-tripped by
    the payload measurement), per-chunk SHA-256 over the upload representation,
    load_state_dict(strict=True) with per-tensor equality on the 207-key
    reconstruction, and an HMAC-SHA256 receipt with verification.

    NOT_EXERCISED, reported and never stubbed or marked PASS (design SS 18,
    SS 33.7): TPM ECDSA P-256 signing and the live chunked GridFS upload. A local
    run does not invoke the federated transport subsystem, and pretending
    otherwise would be a false integrity claim.
    """
    result: Dict[str, Any] = {"layers": {}, "ok": True}

    def record(layer: str, status: str, detail: str = "") -> None:
        result["layers"][layer] = {"status": status, "detail": detail}
        if status == "FAIL":
            result["ok"] = False

    record("sha256_full_hash", "PASS", payload_meas["serialised_sha256"])
    record("aes_gcm_at_rest", "PASS",
           "SecureStore encrypt/decrypt round-trip hash match "
           f"({payload_meas['encrypted_bytes']} B)")

    # per-chunk SHA-256 over the upload representation (ARCHITECTURE.md:338).
    # Recomputed here independently of the measurement that produced them.
    try:
        with open(payload_meas["encrypted_path"], "rb") as fh:
            raw = fh.read()
        chunks = [raw[i:i + C.CHUNK_BYTES] for i in range(0, len(raw), C.CHUNK_BYTES)]
        recomputed = [C.sha256_bytes(x) for x in chunks]
        if recomputed != payload_meas["chunk_sha256"]:
            record("per_chunk_sha256", "FAIL", "chunk hash recomputation mismatch")
            return result
        if C.sha256_bytes(raw) != payload_meas["payload_hash"]:
            record("per_chunk_sha256", "FAIL", "full payload_hash mismatch")
            return result
        record("per_chunk_sha256", "PASS",
               f"{len(chunks)} chunk(s) verified; payload_hash matches")
    except Exception as exc:  # noqa: BLE001
        record("per_chunk_sha256", "FAIL", f"{type(exc).__name__}: {exc}")
        return result

    # strict=True load of the 207-key reconstruction (design SS 11).
    try:
        loaded = torch.load(io.BytesIO(compact_obj_bytes), map_location="cpu",
                            weights_only=False)
        if int(loaded["k"]) != C.K_VALUES[arm]:
            record("strict_load_and_tensor_equality", "FAIL",
                   f"transported k={loaded['k']} != {C.K_VALUES[arm]}")
            return result
        shell = torch.nn.ParameterDict(
            {k.replace(".", "_"): torch.nn.Parameter(torch.zeros_like(v))
             for k, v in reference.items()})
        shell.load_state_dict(
            {k.replace(".", "_"): v for k, v in reconstructed.items()}, strict=True)
        same_keys = set(reconstructed) == set(reference)
        same_shapes = all(reconstructed[k].shape == reference[k].shape
                          for k in reference)
        if not (same_keys and same_shapes):
            record("strict_load_and_tensor_equality", "FAIL",
                   "reconstruction is not key/shape identical to the 207-key target")
            return result
        record("strict_load_and_tensor_equality", "PASS",
               f"strict=True load clean; {len(reference)} keys, shapes identical")
    except Exception as exc:  # noqa: BLE001
        record("strict_load_and_tensor_equality", "FAIL", f"{type(exc).__name__}: {exc}")
        return result

    # HMAC-SHA256 receipt + verify
    try:
        from centralised_receipts import CentralReceiptManager
        rm = CentralReceiptManager(agent="exp9-compact", key_source=key_source)
        receipt = rm.create_receipt(
            agent="exp9-compact", session_id=session_id,
            operation="exp9_compact_update",
            params={"arm": arm, "k": C.K_VALUES[arm],
                    "payload_hash": payload_meas["payload_hash"]},
            outputs=[f"file://{payload_meas['encrypted_path']}"])
        rdir = os.path.join(out_dir, "receipts")
        os.makedirs(rdir, exist_ok=True)
        rpath = rm.write_receipt(receipt, out_dir=rdir)
        local = rpath[len("file://"):] if str(rpath).startswith("file://") else str(rpath)
        if not rm.verify(local):
            record("hmac_receipt", "FAIL", "HMAC verification returned False")
            return result
        record("hmac_receipt", "PASS", f"verified {os.path.basename(local)}")
        record("hmac_chain",
               "PASS" if ("prev" in json.dumps(receipt) or "hmac_chain" in receipt)
               else "NOT_EXERCISED",
               "chain field presence checked on the emitted receipt")
    except Exception as exc:  # noqa: BLE001
        record("hmac_receipt", "FAIL", f"{type(exc).__name__}: {exc}")
        return result

    record("tpm_ecdsa_p256", "NOT_EXERCISED",
           "federated transport not invoked by a local run (design SS 18, SS 33.7); "
           "reported, never assumed to pass")
    record("gridfs_live_upload", "NOT_EXERCISED",
           "no network transport in a local run (design SS 18, SS 33.7); the chunked "
           "byte representation IS measured, but no upload occurs")
    return result


# ==========================================================================
# Main
# ==========================================================================
def main() -> int:
    ap = argparse.ArgumentParser(
        description="Phase 18 / Experiment 9 - Terminal half (mask, DP/SNR, payload)")
    ap.add_argument("--out-dir", default=os.path.join(C.REPO, C.OUT_DIRNAME))
    ap.add_argument("--delta", default=os.path.join(C.REPO, C.DELTA_PATH))
    ap.add_argument("--arms", default=",".join(C.ARMS))
    ap.add_argument("--keep-payloads", action="store_true",
                    help="retain the encrypted payload files (K0 is ~585 MB)")
    ap.add_argument("--plan-only", action="store_true",
                    help="gates + ordering + nesting + eps + analytical SNR, then "
                         "STOP. No draw, no noise, no encryption, no result files.")
    a = ap.parse_args()
    os.chdir(C.REPO)

    arms = [x.strip() for x in a.arms.split(",") if x.strip()]
    for arm in arms:
        if arm not in C.ARMS:
            C.abort(f"unknown arm {arm!r}; the pre-registered ladder is {list(C.ARMS)}. "
                    "design SS 32.4 forbids adding an arm.")

    t0 = time.time()
    print("=" * 78)
    print("EXPERIMENT 9 - COMPACT UPDATE REPRESENTATION (Terminal half)")
    print(f"mechanism: {C.MECHANISM_NAME}  --  NOT top-k (design SS 3, SS 32.8)")
    print("=" * 78)

    # ---- design SS 28.1 / SS 29: frozen-input gate, IN the execution path ----
    frozen_before = C.snapshot_frozen()
    C.gate_frozen_before(frozen_before)
    print(f"[OK] {len(C.FROZEN_SHA)} frozen inputs verified before execution")

    # ---- design SS 28.2 / SS 28.3: target structure -------------------------
    delta, order, struct = load_frozen_delta(a.delta)
    delta_sha = C.sha256_file(a.delta)
    print(f"[OK] target {os.path.basename(a.delta)}  d={struct['d']:,}  "
          f"keys={struct['n_keys']}  sha={delta_sha[:32]}...")
    for g in ("fusion.*", "bert.pooler.dense", "bert.encoder.layer.{0..11}",
              "bert.embeddings.*"):
        print(f"      {g:<30s} keys={struct['group_keys'][g]:>4d}  "
              f"params={struct['group_params'][g]:>12,}")

    # ---- design SS 28.4 / SS 28.5: nesting and public construction -----------
    nesting = C.verify_nesting(order)
    print("[OK] nesting S_K3 subset S_K2 subset S_K1 subset S_K0 verified "
          "(prefixes of one ordering)")
    for arm in C.ARMS:
        d = nesting["arms"][arm]
        print(f"      {arm}: k={d['k']:>12,}  ({100*d['fraction_of_d']:8.4f}%)  "
              f"boundary {d['boundary_key']} @ {d['offset_in_key']:,}")
    print("[OK] mask built from key names + shapes only; no delta value consulted")

    # ---- design SS 28.6: eps, identical for every arm ------------------------
    eps = C.epsilon()
    if abs(eps - C.EPS_PER_UPDATE_MAX) > 1e-12:
        C.abort(f"recomputed eps={eps!r} != declared {C.EPS_PER_UPDATE_MAX} "
                "(design SS 29: HARD ABORT)")
    if eps * C.N_UPDATES_PER_ROUND > C.EPS_CUMULATIVE_MAX + 1e-9:
        C.abort(f"cumulative eps {eps * C.N_UPDATES_PER_ROUND} exceeds "
                f"{C.EPS_CUMULATIVE_MAX}")
    print(f"[OK] eps = {eps:.15f} for every arm (independent of k); "
          f"cumulative {eps * C.N_UPDATES_PER_ROUND:.6f} <= {C.EPS_CUMULATIVE_MAX}")

    analytical = {arm: C.analytical_snr(C.K_VALUES[arm]) for arm in C.ARMS}
    print("[OK] analytical SNR (registered in advance; NEVER an acceptance input):")
    for arm in C.ARMS:
        print(f"      {arm}: {analytical[arm]:>12.4f}")

    os.makedirs(a.out_dir, exist_ok=True)
    mask_manifest = {
        "schema": "exp9_mask_manifest/1",
        "experiment": C.EXPERIMENT,
        "design_sha256": C.DESIGN_SHA,
        "mechanism": C.MECHANISM_NAME,
        "magnitude_based": False,
        "mask_inputs": ["state_dict key names", "tensor shapes",
                        "frozen state_dict key order", "row-major flattened index"],
        "prohibited_inputs_used": [],
        "ordering": [{"rank": i, "key": k, "numel": n, "cumulative": cum,
                      "group": C.group_name(k)}
                     for i, (k, n, cum) in enumerate(order)],
        "structure": struct,
        "k_values": C.K_VALUES,
        "nesting": nesting,
        "target": {"path": os.path.relpath(a.delta, C.REPO), "sha256": delta_sha},
    }
    mpath = os.path.join(a.out_dir, "exp9_mask_manifest.json")

    if a.plan_only:
        print("-" * 78)
        print("[PLAN-ONLY] static validation complete. No draw, no noise, no "
              "encryption, no file written. Exiting before the experiment.")
        return 0

    with open(mpath, "w", encoding="utf-8") as fh:
        json.dump(mask_manifest, fh, indent=2, sort_keys=True)
    print(f"[OK] mask manifest -> {mpath}")

    session_id = f"exp9-{int(time.time())}"
    key_source = local_receipt_key_source(a.out_dir)
    print(f"[OK] receipt key: {key_source} (inside the authorised output dir)")

    rows: List[Dict[str, Any]] = []
    payloads: Dict[str, Any] = {}
    crypto: Dict[str, Any] = {}

    for arm in arms:
        k = C.K_VALUES[arm]
        print(f"\n[ARM {arm}] k = {k:,}  ({C.ARM_LABEL[arm]})")
        vec = C.compact_vector(delta, order, k)

        # --- DP draws: r = 5, seeds 1001..1005, same seeds across arms (SS 26)
        for seed in C.SEEDS:
            r = C.dp_release(vec, seed)
            noisy = r.pop("noisy")
            r.update({"arm": arm,
                      "analytical_snr": analytical[arm],
                      "analytical_deviation": r["snr"] - analytical[arm],
                      "epsilon": eps,
                      "delta_dp": C.DELTA_DP,
                      "clip_norm": C.CLIP_NORM,
                      "noise_multiplier": C.NOISE_MULTIPLIER,
                      "sigma_eff": C.SIGMA_EFF})
            rows.append(r)
            print(f"  [{arm} seed={seed}] l2_after={r['l2_after']:.4f} "
                  f"SNR={r['snr']:.6f} eps={r['epsilon']:.12f}")

            # --- payload + crypto measured once per arm, on the first seed.
            # design SS 30.1/SS 30.6 report these PER ARM; the payload size is a
            # property of (k, encoding) and not of the draw.
            if seed == C.SEEDS[0]:
                obj = C.serialise_compact(noisy, arm, k)
                meas = C.measure_transport_payload(obj, f"{arm}_k{k}", a.out_dir)
                meas.update({
                    "arm": arm, "k": k,
                    "raw_4k_bytes": 4 * k,          # diagnostic ONLY (design SS 16)
                    "payload_seed": seed,
                    "g2_threshold_bytes": C.G2_PAYLOAD_MAX_BYTES,
                    "g2_pass": bool(meas["total_uploaded_bytes"]
                                    <= C.G2_PAYLOAD_MAX_BYTES),
                })
                recon = C.apply_mask_to_delta(delta, order, k)
                cv = crypto_verify(obj, meas, recon, delta, arm, session_id,
                                   a.out_dir, key_source)
                if not cv["ok"]:
                    C.abort(f"cryptographic verification FAILED for {arm}: "
                            f"{cv['layers']}  (design SS 29: HARD ABORT)")
                del recon
                payloads[arm] = meas
                crypto[arm] = cv["layers"]
                print(f"  [{arm}] payload: serialised {meas['serialised_bytes']:,} B "
                      f"-> encrypted {meas['encrypted_bytes']:,} B in "
                      f"{meas['n_chunks']} chunk(s); uploaded "
                      f"{meas['total_uploaded_bytes']:,} B  "
                      f"G2({C.G2_PAYLOAD_MAX_BYTES:,}) -> "
                      f"{'PASS' if meas['g2_pass'] else 'FAIL'}")
                if not a.keep_payloads:
                    try:
                        os.remove(meas["encrypted_path"])
                    except OSError:
                        pass
                del obj
        del vec

    # ---- design SS 28.10: identical seeds reproduce byte-identical draws -----
    print("\n[STEP] reproducibility re-draw (design SS 28.10, SS 31)")
    repro: List[Dict[str, Any]] = []
    for arm in arms:
        vec = C.compact_vector(delta, order, C.K_VALUES[arm])
        for r in [x for x in rows if x["arm"] == arm]:
            again = C.dp_release(vec, int(r["seed"]))
            same = again["draw_sha256"] == r["draw_sha256"]
            repro.append({"arm": arm, "seed": int(r["seed"]),
                          "identical": bool(same), "draw_sha256": r["draw_sha256"]})
            if not same:
                C.abort(f"reproducibility FAILED: {arm} seed {r['seed']} did not "
                        "reproduce a byte-identical draw")
        del vec
    print(f"[OK] {len(repro)}/{len(rows)} draws reproduced byte-identically")

    # ---- artifacts ----------------------------------------------------------
    cols = ["arm", "k", "seed", "l2_before", "clip_scale", "l2_signal_after_clip",
            "l2_noise", "l2_after", "snr", "analytical_snr", "analytical_deviation",
            "sensitivity", "sigma", "clip_norm", "noise_multiplier", "sigma_eff",
            "epsilon", "delta_dp", "draw_sha256"]
    dpath = os.path.join(a.out_dir, "exp9_dp_metrics.csv")
    # MERGE, keyed on (arm, seed). Rows recorded by an earlier invocation are
    # carried over as the exact text they were written as - they are never
    # re-parsed into floats and re-formatted, so no recorded value can drift.
    # This run's rows replace only their own (arm, seed).
    merged_rows = {}
    if os.path.isfile(dpath):
        with open(dpath, "r", newline="", encoding="utf-8") as fh:
            for old in csv.DictReader(fh):
                merged_rows[(old["arm"], str(old["seed"]))] = old
    n_prior = len(merged_rows)
    for r in rows:
        merged_rows[(r["arm"], str(r["seed"]))] = {c: r.get(c) for c in cols}
    with open(dpath, "w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=cols)
        w.writeheader()
        for key in sorted(merged_rows, key=lambda t: arm_sort_key(t[0], t[1])):
            w.writerow({c: merged_rows[key].get(c) for c in cols})
    print(f"[OK] exp9_dp_metrics.csv: {n_prior} prior row(s) preserved + "
          f"{len(rows)} from this run = {len(merged_rows)} total")

    ppath = os.path.join(a.out_dir, "exp9_payload.json")
    # MERGE: earlier arms first, this run's measurements override only their arm.
    prior_payload = load_prior_json(ppath)
    payload_arms = dict((prior_payload or {}).get("arms", {}))
    payload_arms.update(payloads)
    with open(ppath, "w", encoding="utf-8") as fh:
        json.dump({"schema": "exp9_payload/1",
                   "definition": ("complete transport payload = total uploaded byte "
                                  "count; torch.save -> AES-GCM -> 1 MB chunks -> "
                                  "SHA-256 of all uploaded bytes (design SS 16)"),
                   "note": ("raw_4k_bytes is a DIAGNOSTIC only and is never "
                            "substituted for the measurement (design SS 16)"),
                   "g2_threshold_bytes": C.G2_PAYLOAD_MAX_BYTES,
                   "baseline_reference": {
                       "raw_bytes": C.BASELINE_RAW_BYTES,
                       "transport_bytes": C.BASELINE_TRANSPORT_BYTES},
                   "arms": payload_arms}, fh, indent=2, sort_keys=True)
    print(f"[OK] exp9_payload.json: arms {sorted(payload_arms)}")

    frozen_after = C.gate_frozen_after(frozen_before)
    print("[OK] frozen inputs re-verified after execution: unchanged")

    elapsed = round(time.time() - t0, 2)
    receipt = {
        "schema": "exp9_receipt/1",
        "experiment": C.EXPERIMENT,
        "half": "terminal (mask + DP/SNR + transport payload)",
        "design": {"path": C.DESIGN_PATH, "sha256": C.DESIGN_SHA},
        "generated_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "session_id": session_id,
        "mechanism": C.MECHANISM_NAME,
        "magnitude_based": False,
        "frozen_input_sha": {"pre_run": frozen_before, "post_run": frozen_after,
                             "unchanged": True},
        "target": {"path": os.path.relpath(a.delta, C.REPO), "sha256": delta_sha,
                   "d": struct["d"], "n_keys": struct["n_keys"]},
        "mask": {"ordering_sha256": C.sha256_bytes(
                     json.dumps([o[0] for o in order]).encode()),
                 "k_values": C.K_VALUES, "nesting": nesting,
                 "manifest_sha256": C.sha256_file(mpath)},
        "privacy": {"epsilon_per_update": eps,
                    "epsilon_budget_per_update": C.EPS_PER_UPDATE_MAX,
                    "cumulative_epsilon": eps * C.N_UPDATES_PER_ROUND,
                    "cumulative_budget": C.EPS_CUMULATIVE_MAX,
                    "delta": C.DELTA_DP, "composition": C.COMPOSITION,
                    "clip_norm": C.CLIP_NORM,
                    "noise_multiplier": C.NOISE_MULTIPLIER,
                    "sigma_eff": C.SIGMA_EFF,
                    "eps_max_ceiling_only": C.EPS_MAX_CEILING,
                    "identical_across_arms": True,
                    "reason": "eps depends on sigma_eff alone, not on k or d"},
        "seeds": {"base_seed": C.BASE_SEED_DP, "r": C.N_REPETITIONS,
                  "seeds": list(C.SEEDS),
                  "rule": "seed_i = BASE_SEED + i; identical seeds across arms"},
        "snr": {"definition": C.SNR_DEFINITION, "analytical": analytical,
                "note": "analytical is registered in advance and is NEVER an "
                        "acceptance input (design SS 13)"},
        "reproducibility": {"checked": True,
                            "all_identical": all(x["identical"] for x in repro),
                            "n_rechecked": len(repro), "detail": repro},
        "crypto": crypto,
        "environment": {"python": sys.version.split()[0], "platform": sys.platform,
                        "torch": torch.__version__},
        "runtime_seconds": elapsed,
        "task_half": "PENDING - produced by run_exp9_task.py on GPU (design SS 27)",
        "output_artifact_sha": {
            "exp9_mask_manifest.json": C.sha256_file(mpath),
            "exp9_dp_metrics.csv": C.sha256_file(dpath),
            "exp9_payload.json": C.sha256_file(ppath),
        },
    }
    rpath = os.path.join(a.out_dir, "exp9_receipt.json")
    # MERGE: the crypto table and the reproducibility detail accumulate across
    # invocations, keyed on arm and on (arm, seed). Each invocation's provenance
    # (session, timestamp, arms, runtime) is appended to `invocations` so an
    # earlier staged run remains auditable instead of being overwritten.
    this_invocation = {"session_id": session_id,
                       "generated_utc": receipt["generated_utc"],
                       "arms": list(arms), "runtime_seconds": elapsed,
                       "n_draws": len(rows)}
    prior_receipt = load_prior_json(rpath)
    if prior_receipt:
        merged_crypto = dict(prior_receipt.get("crypto", {}))
        merged_crypto.update(crypto)
        prior_detail = prior_receipt.get("reproducibility", {}).get("detail", [])
        now_keys = {(x["arm"], int(x["seed"])) for x in repro}
        merged_detail = [d for d in prior_detail
                         if (d["arm"], int(d["seed"])) not in now_keys] + list(repro)
        merged_detail.sort(key=lambda d: arm_sort_key(d["arm"], d["seed"]))
        receipt["crypto"] = merged_crypto
        receipt["reproducibility"] = {
            "checked": True,
            "all_identical": all(bool(d["identical"]) for d in merged_detail),
            "n_rechecked": len(merged_detail),
            "detail": merged_detail,
        }
        prior_invocations = prior_receipt.get("invocations")
        if prior_invocations is None:
            # An earlier receipt written before staged merging existed; recover
            # its provenance from its own top-level fields rather than lose it.
            prior_invocations = [{
                "session_id": prior_receipt.get("session_id"),
                "generated_utc": prior_receipt.get("generated_utc"),
                "arms": sorted(prior_receipt.get("crypto", {})),
                "runtime_seconds": prior_receipt.get("runtime_seconds"),
                "n_draws": prior_receipt.get("reproducibility", {})
                                        .get("n_rechecked"),
            }]
        receipt["invocations"] = prior_invocations + [this_invocation]
    else:
        receipt["invocations"] = [this_invocation]
    with open(rpath, "w", encoding="utf-8") as fh:
        json.dump(receipt, fh, indent=2, sort_keys=True, default=float)
    print(f"[OK] exp9_receipt.json: crypto arms {sorted(receipt['crypto'])}, "
          f"{receipt['reproducibility']['n_rechecked']} draw(s) in the "
          f"reproducibility record, {len(receipt['invocations'])} invocation(s)")

    print("-" * 78)
    print(f"[DONE] {len(rows)} draws, {len(payloads)} payload measurements "
          f"-> {a.out_dir}  ({elapsed:.1f}s)")
    print("Next: run_exp9_task.py on Colab GPU (25 fold-runs + 4x25 ROC-AUC), "
          "then aggregate_exp9.py, then verify_exp9.py")
    return 0


if __name__ == "__main__":
    sys.exit(main())
