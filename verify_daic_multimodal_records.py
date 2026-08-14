#!/usr/bin/env python3
"""
verify_daic_multimodal_records.py  (Phase 16 - Stage S4: trainer compatibility)

Proves that the FROZEN trainer consumes the new multimodal parquet without a
single line of modification.

This stage does NOT train. No optimizer is constructed, no gradient is taken, no
weight is updated, no checkpoint is written. Every forward pass runs under
torch.no_grad() with the model in eval() mode.

WHAT "USE THE TRAINER'S OWN CODE" MEANS HERE
--------------------------------------------
Everything that can be exercised through the frozen module is:

    T.read_parquet_records()          record loading and JSON decoding
    T.MultiModalDataset               __len__ / __getitem__ / vector extraction
    T.collate_batch                   batching and padding
    T.MultiModalModel                 construction, forward, fusion
    T.modality_ablation_importance()  the Experiment 7 success criterion

The trainer is imported read-only. It is never edited, and never monkey-patched:
no attribute of the imported module is assigned to at any point.

A KNOWN CONSEQUENCE OF NOT MONKEY-PATCHING
------------------------------------------
trainer_mentalbert_daic.py line 442 reads:

    tokenizer = AutoTokenizer.from_pretrained(args.bert_model_name) \
                if "args" in globals() else None

`globals()` there is the TRAINER MODULE's namespace. `args` is created only when
the trainer runs as __main__ (argparse in its own main()). Importing the module -
which any verifier must do - leaves `args` undefined, so the text branch takes
the `else` path and returns text_posdelta = text_phqdelta = 0.0.

That zero is an artifact of execution context, NOT a statement about the data.
Phase 9.5 recorded text = 2.3365 precisely because it executed the trainer as a
script. Injecting `T.args` would make the number appear, but that is exactly the
monkey-patching this stage is forbidden to do, so the frozen result is reported
as-is and a clearly-labelled SUPPLEMENTARY diagnostic establishes separately that
text is still a live input to the graph.

WHAT THE ABLATION DOES AND DOES NOT SHOW
----------------------------------------
The model here is MentalBERT plus a RANDOMLY INITIALISED fusion head; nothing is
trained. A non-zero audio/vision delta therefore proves the modality is present
in the computation graph and reaches the fusion head - which is precisely the
"absence-from-graph vs weak contribution" distinction the roadmap's Exp 7
criterion is about. It is NOT evidence of predictive value. Only a trained run
can speak to that.

Inputs (read-only):
    trainer_mentalbert_daic.py                      frozen trainer
    dataset_build/daic_records_multimodal.parquet   frozen Stage S3 artifact

Output:
    dataset_build/multimodal_verification.json
    exit code   0 = compatible, 1 = verification failure, 2 = fatal setup error

Usage:
    python verify_daic_multimodal_records.py
    python verify_daic_multimodal_records.py --parquet daic_records.parquet   # must FAIL
    python verify_daic_multimodal_records.py --batches 3 --batch-size 8

NOTE: run with the project venv (.venv/Scripts/python.exe).
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np

DEFAULT_PARQUET = Path("./dataset_build/daic_records_multimodal.parquet")
DEFAULT_OUT = Path("./dataset_build/multimodal_verification.json")
TRAINER_PATH = Path("./trainer_mentalbert_daic.py")

FROZEN_TRAINER_SHA = "65b1902e2ba8dd996c6960db1a6595d84fd48500f4d8707cb79688093d7a230b"
BERT_MODEL = "mental/mental-bert-base-uncased"

EXPECTED_RECORDS = 188
AUDIO_DIM = 154
VISION_DIM = 84
MAX_LEN = 128          # the frozen recipe's tokenizer length
BATCH_SIZE = 8         # the frozen recipe's batch size
N_BATCHES = 3
SEED = 1000            # BASE_SEED used across Baseline-CV and Exps 3-6

SEV_FAIL, SEV_WARN, SEV_INFO = "FAIL", "WARN", "INFO"


class Reporter:
    """ASCII-only reporting, matching the S0-S3 idiom (console is cp1252)."""

    def __init__(self, verbose: bool = False) -> None:
        self.findings: List[Dict[str, Any]] = []
        self.verbose = verbose

    def add(self, sev: str, check: str, msg: str) -> None:
        self.findings.append({"severity": sev, "check": check, "message": msg})
        if sev == SEV_FAIL or (sev == SEV_WARN and self.verbose):
            print(f"[{sev}] {check}: {msg}")

    def fail(self, c: str, m: str) -> None:
        self.add(SEV_FAIL, c, m)

    def warn(self, c: str, m: str) -> None:
        self.add(SEV_WARN, c, m)

    def info(self, c: str, m: str) -> None:
        self.add(SEV_INFO, c, m)

    def count(self, sev: str) -> int:
        return sum(1 for f in self.findings if f["severity"] == sev)

    def digest(self) -> str:
        canon = sorted((f["severity"], f["check"], f["message"]) for f in self.findings)
        return hashlib.sha256(
            json.dumps(canon, sort_keys=True, ensure_ascii=True).encode("utf-8")).hexdigest()


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for block in iter(lambda: fh.read(1 << 16), b""):
            h.update(block)
    return h.hexdigest()


def load_trainer():
    """Import the frozen trainer WITHOUT executing its main().

    Identical mechanism to verify_daic_records.py. No attribute of the returned
    module is ever assigned to - that would be monkey-patching.
    """
    spec = importlib.util.spec_from_file_location("trainer_mentalbert_daic", TRAINER_PATH)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def tensor_stats(t) -> Dict[str, Any]:
    a = t.detach().cpu().numpy()
    return {"shape": list(a.shape), "dtype": str(a.dtype),
            "finite": bool(np.isfinite(a).all()) if a.dtype.kind == "f" else True,
            "min": float(a.min()) if a.size else None,
            "max": float(a.max()) if a.size else None}


def main() -> int:
    ap = argparse.ArgumentParser(
        description="Verify frozen-trainer compatibility with the multimodal parquet (S4)")
    ap.add_argument("--parquet", type=Path, default=DEFAULT_PARQUET)
    ap.add_argument("--out", type=Path, default=DEFAULT_OUT)
    ap.add_argument("--batches", type=int, default=N_BATCHES)
    ap.add_argument("--batch-size", type=int, default=BATCH_SIZE)
    ap.add_argument("--expect-dims", default=f"{AUDIO_DIM},{VISION_DIM}",
                    help="expected (audio,vision); used by negative tests")
    ap.add_argument("--skip-model", action="store_true",
                    help="skip model construction / forward / ablation (data checks only)")
    ap.add_argument("--verbose", action="store_true")
    args = ap.parse_args()

    t0 = time.time()
    rep = Reporter(verbose=args.verbose)

    if not TRAINER_PATH.is_file():
        print(f"[FATAL] frozen trainer not found: {TRAINER_PATH}")
        return 2
    if not args.parquet.is_file():
        print(f"[FATAL] parquet not found: {args.parquet}")
        return 2
    try:
        exp_a, exp_v = (int(x) for x in args.expect_dims.split(","))
    except ValueError:
        print("[FATAL] --expect-dims must be 'audio,vision'")
        return 2

    print("=" * 78)
    print("MULTIMODAL TRAINER-COMPATIBILITY VERIFICATION (Stage S4)")
    print("NO TRAINING - no optimizer, no gradients, no weight updates")
    print("=" * 78)
    print(f"[INFO] trainer : {TRAINER_PATH.resolve()}")
    print(f"[INFO] parquet : {args.parquet.resolve()}")
    print("-" * 78)

    # ---- 1. frozen trainer, imported read-only ----------------------------
    print("[STEP 1/7] importing the frozen trainer")
    tsha = sha256_file(TRAINER_PATH)
    if tsha != FROZEN_TRAINER_SHA:
        rep.fail("A1.trainer_sha", f"trainer SHA {tsha} != frozen {FROZEN_TRAINER_SHA}")
        print("[FATAL] not the verified trainer - refusing to continue")
        return 1
    print(f"[ OK ] trainer SHA verified: {tsha[:32]}...")

    try:
        import torch
        from torch.utils.data import DataLoader
        from transformers import AutoTokenizer
        T = load_trainer()
    except Exception as exc:  # noqa: BLE001
        print(f"[FATAL] cannot import trainer or dependencies: {type(exc).__name__}: {exc}")
        return 2
    # Guard the no-monkey-patch promise: `args` must NOT be present in the
    # trainer's namespace, and this stage must not put it there.
    trainer_has_args = hasattr(T, "args")
    print(f"[ OK ] trainer imported (module-level 'args' present: {trainer_has_args})")

    # ---- 2. records via the trainer's own loader ---------------------------
    print("[STEP 2/7] loading records with T.read_parquet_records()")
    records = T.read_parquet_records(str(args.parquet))
    n = len(records)
    if n != EXPECTED_RECORDS:
        rep.fail("A2.records", f"loaded {n} records, expected {EXPECTED_RECORDS}")
    print(f"[ OK ] {n} records loaded")

    # infer_dims: replicated from trainer main() lines 535-549, which is not an
    # importable function in the frozen module. The extraction that MATTERS is
    # done below with the dataset's own methods.
    audio_dim = vision_dim = None
    for r in records:
        f = r.get("features") or {}
        if isinstance(f, dict) and "audio" in f:
            a = f["audio"]
            if isinstance(a, dict) and a.get("wav2vec2"):
                audio_dim = len(a["wav2vec2"]); break
            if isinstance(a, dict) and a.get("egemaps"):
                audio_dim = len(a["egemaps"].keys()); break
    for r in records:
        f = r.get("features") or {}
        if isinstance(f, dict) and "video" in f and isinstance(f["video"], dict) \
                and f["video"].get("densenet"):
            vision_dim = len(f["video"]["densenet"]); break
        if any(str(k).startswith("neuron_") for k in r.keys()):
            vision_dim = len([k for k in r.keys() if str(k).startswith("neuron_")]); break
    if (audio_dim, vision_dim) != (exp_a, exp_v):
        rep.fail("A3.infer_dims",
                 f"infer_dims = ({audio_dim}, {vision_dim}), expected ({exp_a}, {exp_v})")
    print(f"[ OK ] infer_dims = ({audio_dim}, {vision_dim})")

    # ---- 3. per-record extraction through the dataset ----------------------
    print("[STEP 3/7] per-record extraction via T.MultiModalDataset")
    try:
        tokenizer = AutoTokenizer.from_pretrained(BERT_MODEL)
    except Exception as exc:  # noqa: BLE001
        print(f"[FATAL] tokenizer unavailable ({type(exc).__name__}: {exc})")
        return 2
    ds = T.MultiModalDataset(records, tokenizer, max_len=MAX_LEN)

    if len(ds) != n:
        rep.fail("A4.len", f"__len__ = {len(ds)}, expected {n}")

    n_text = n_audio = n_vision = 0
    bad_audio: List[str] = []
    bad_vision: List[str] = []
    nonfinite: List[str] = []
    for i in range(len(ds)):
        rec = records[i]
        pid = str(rec.get("participant_id"))
        a = ds._extract_audio_vec(rec)
        v = ds._extract_video_vec(rec)
        txt = str(rec.get("transcript") or rec.get("text") or "").strip()
        if txt:
            n_text += 1
        if a is None:
            bad_audio.append(pid)
        else:
            n_audio += 1
            if tuple(a.shape) != (exp_a,):
                bad_audio.append(f"{pid}:shape{tuple(a.shape)}")
            if not bool(torch.isfinite(a).all()):
                nonfinite.append(f"{pid}:audio")
        if v is None:
            bad_vision.append(pid)
        else:
            n_vision += 1
            if tuple(v.shape) != (exp_v,):
                bad_vision.append(f"{pid}:shape{tuple(v.shape)}")
            if not bool(torch.isfinite(v).all()):
                nonfinite.append(f"{pid}:vision")

    if n_text != n:
        rep.fail("A5.text", f"{n - n_text} record(s) have empty text")
    if bad_audio:
        rep.fail("A6.audio", f"missing/wrong-shape audio vectors: {bad_audio[:10]}")
    if bad_vision:
        rep.fail("A6.vision", f"missing/wrong-shape vision vectors: {bad_vision[:10]}")
    if nonfinite:
        rep.fail("A7.nonfinite", f"non-finite tensor values: {nonfinite[:10]}")
    print(f"[ OK ] text {n_text}/{n}, audio {n_audio}/{n} ({exp_a}-d), "
          f"vision {n_vision}/{n} ({exp_v}-d), no None vectors")

    # __getitem__ contract
    item = ds[0]
    for key in ("input_ids", "attention_mask", "audio_vec", "video_vec", "phq", "label"):
        if key not in item:
            rep.fail("A8.getitem", f"__getitem__ missing key '{key}'")
    if "input_ids" in item and tuple(item["input_ids"].shape) != (MAX_LEN,):
        rep.fail("A8.getitem", f"input_ids shape {tuple(item['input_ids'].shape)}")

    # ---- 4/5. batching through T.collate_batch -----------------------------
    print("[STEP 4/7] batching via T.collate_batch + DataLoader")
    torch.manual_seed(SEED)
    loader = DataLoader(ds, batch_size=args.batch_size, shuffle=False,
                        collate_fn=T.collate_batch)
    batch_report: List[Dict[str, Any]] = []
    batches = []
    for bi, batch in enumerate(loader):
        if bi >= args.batches:
            break
        batches.append(batch)
        info = {"batch": bi, "tensors": {}}
        for key in ("input_ids", "attention_mask", "audio_vec", "video_vec", "phq", "label"):
            t = batch.get(key)
            if t is None:
                rep.fail("A9.collate", f"batch {bi}: '{key}' is None "
                                       "(collate saw no values for this modality)")
                continue
            info["tensors"][key] = tensor_stats(t)
        b = batch["input_ids"].shape[0]
        expect = {"input_ids": (b, MAX_LEN), "attention_mask": (b, MAX_LEN),
                  "audio_vec": (b, exp_a), "video_vec": (b, exp_v),
                  "phq": (b,), "label": (b,)}
        for key, want in expect.items():
            t = batch.get(key)
            if t is not None and tuple(t.shape) != want:
                rep.fail("A9.shape", f"batch {bi}: {key} shape {tuple(t.shape)} != {want}")
        # Silent-padding detection: collate zero-fills a None entry to the batch
        # max. With every record carrying a full-length vector, an all-zero row
        # would mean a modality went missing without raising.
        for key in ("audio_vec", "video_vec"):
            t = batch.get(key)
            if t is not None:
                zero_rows = int((t.abs().sum(dim=1) == 0).sum().item())
                info[f"{key}_zero_rows"] = zero_rows
                if zero_rows:
                    rep.fail("A10.silent_padding",
                             f"batch {bi}: {zero_rows} all-zero {key} row(s)")
        batch_report.append(info)
    print(f"[ OK ] {len(batches)} batches; shapes "
          f"input_ids{tuple(batches[0]['input_ids'].shape)} "
          f"audio{tuple(batches[0]['audio_vec'].shape)} "
          f"vision{tuple(batches[0]['video_vec'].shape)}")

    forward_report: Dict[str, Any] = {"run": False}
    ablation: Dict[str, Any] = {"run": False}
    supplementary: Dict[str, Any] = {"run": False}

    if args.skip_model:
        print("[STEP 5/7] model construction SKIPPED (--skip-model)")
        print("[STEP 6/7] forward passes SKIPPED")
        print("[STEP 7/7] ablation SKIPPED")
        rep.warn("A0.skip_model", "model, forward and ablation checks were skipped")
    else:
        # ---- 6. model construction + forward passes (NO TRAINING) ----------
        print(f"[STEP 5/7] constructing T.MultiModalModel(audio_dim={exp_a}, "
              f"vision_dim={exp_v})")
        torch.manual_seed(SEED)   # fusion head / encoders are randomly initialised
        model = T.MultiModalModel(BERT_MODEL, audio_dim=exp_a, vision_dim=exp_v,
                                  device="cpu")
        model.eval()   # eval() disables dropout -> deterministic forward
        if not model.has_audio:
            rep.fail("A11.encoder", "audio encoder was not constructed")
        if not model.has_vision:
            rep.fail("A11.encoder", "vision encoder was not constructed")
        fusion_in = model.fusion.fc1.in_features
        expected_fusion = model.bert.config.hidden_size + 128 + 128
        if fusion_in != expected_fusion:
            rep.fail("A11.fusion",
                     f"fusion input {fusion_in} != {expected_fusion} "
                     "(bert_hidden + 128 audio + 128 vision)")
        n_params = sum(p.numel() for p in model.parameters())
        print(f"[ OK ] encoders present; fusion input {fusion_in}; "
              f"{n_params:,} parameters")

        print("[STEP 6/7] forward passes (torch.no_grad, eval mode)")
        cls_loss_fn = torch.nn.CrossEntropyLoss()
        reg_loss_fn = torch.nn.MSELoss()
        per_batch = []
        with torch.no_grad():
            for bi, batch in enumerate(batches):
                logits, reg_pred, _ = model(
                    batch["input_ids"], batch["attention_mask"],
                    audio_vec=batch.get("audio_vec"),
                    vision_vec=batch.get("video_vec"))
                ok = bool(torch.isfinite(logits).all() and torch.isfinite(reg_pred).all())
                if not ok:
                    rep.fail("A12.forward", f"batch {bi}: non-finite forward output")
                loss_cls = cls_loss_fn(logits, batch["label"])
                loss_reg = reg_loss_fn(reg_pred, batch["phq"])
                loss = loss_cls + 0.5 * loss_reg   # the frozen objective
                if not bool(torch.isfinite(loss)):
                    rep.fail("A13.loss", f"batch {bi}: non-finite loss")
                per_batch.append({
                    "batch": bi,
                    "logits": tensor_stats(logits),
                    "reg_pred": tensor_stats(reg_pred),
                    "loss_cls": float(loss_cls.item()),
                    "loss_reg": float(loss_reg.item()),
                    "loss_total": float(loss.item()),
                })
        # Nothing may have accumulated gradients.
        if any(p.grad is not None for p in model.parameters()):
            rep.fail("A14.no_training", "gradients were accumulated - this stage must not train")
        forward_report = {"run": True, "batches": per_batch,
                          "objective": "CrossEntropy + 0.5 * MSE (frozen)"}
        print(f"[ OK ] {len(per_batch)} forward passes; "
              f"loss_total={[round(b['loss_total'], 4) for b in per_batch]}; "
              "no gradients accumulated")

        # ---- 7. the Experiment 7 success criterion -------------------------
        print("[STEP 7/7] T.modality_ablation_importance()")
        agg = T.modality_ablation_importance(model, batches[0], device="cpu")
        ablation = {"run": True, **agg}
        a_s, v_s, t_s = agg["audio_score"], agg["vision_score"], agg["text_score"]
        if a_s == 0.0:
            rep.fail("A15.audio_zero",
                     "audio_score is exactly 0.0 - the modality is absent from the graph")
        if v_s == 0.0:
            rep.fail("A15.vision_zero",
                     "vision_score is exactly 0.0 - the modality is absent from the graph")
        if t_s <= 0.0:
            # Expected under module import: see the header. Recorded as a WARN
            # with its precise cause, never silently patched away.
            rep.warn("A16.text_zero",
                     "text_score is 0.0 because trainer line 442 requires module-level "
                     "'args', which exists only when the trainer runs as __main__; "
                     "this is an execution-context artifact, not a data property")
        print(f"[ OK ] ablation: audio={a_s:.6g}  vision={v_s:.6g}  text={t_s:.6g}")

        # Supplementary, clearly separated from the frozen result: show that text
        # is still a live input by re-running forward with the attention mask
        # zeroed. Uses the trainer's own model.forward; it does NOT substitute
        # for, or get merged into, the frozen ablation numbers above.
        with torch.no_grad():
            b0 = batches[0]
            base_logits, base_reg, _ = model(b0["input_ids"], b0["attention_mask"],
                                             audio_vec=b0["audio_vec"],
                                             vision_vec=b0["video_vec"])
            base_pos = float(torch.softmax(base_logits, 1)[:, 1].mean())
            base_phq = float(base_reg.mean())
            mask0 = torch.zeros_like(b0["attention_mask"])
            t_logits, t_reg, _ = model(b0["input_ids"], mask0,
                                       audio_vec=b0["audio_vec"],
                                       vision_vec=b0["video_vec"])
            t_pos = float(torch.softmax(t_logits, 1)[:, 1].mean())
            t_phq = float(t_reg.mean())
        supplementary = {
            "run": True,
            "method": "attention_mask zeroed; model.forward from the frozen trainer",
            "text_posdelta": abs(base_pos - t_pos),
            "text_phqdelta": abs(base_phq - t_phq),
            "text_score": (abs(base_pos - t_pos) + abs(base_phq - t_phq)) / 2.0,
            "note": ("SUPPLEMENTARY diagnostic only - not a substitute for the frozen "
                     "ablation, and not comparable to Phase 9.5's 2.3365"),
        }
        if supplementary["text_score"] == 0.0:
            rep.fail("A17.text_absent",
                     "text has no effect on the output even when fully masked - "
                     "the text pathway is not live")
        print(f"[ OK ] supplementary text-in-graph check: "
              f"score={supplementary['text_score']:.6g}")

    # ---- determinism of loading and batching -------------------------------
    ds2 = T.MultiModalDataset(T.read_parquet_records(str(args.parquet)),
                              tokenizer, max_len=MAX_LEN)
    loader2 = DataLoader(ds2, batch_size=args.batch_size, shuffle=False,
                         collate_fn=T.collate_batch)
    det_ok = True
    for bi, b2 in enumerate(loader2):
        if bi >= len(batches):
            break
        for key in ("input_ids", "attention_mask", "audio_vec", "video_vec", "phq", "label"):
            x, y = batches[bi].get(key), b2.get(key)
            if (x is None) != (y is None):
                det_ok = False
            elif x is not None and not bool(torch.equal(x, y)):
                det_ok = False
    if not det_ok:
        rep.fail("A18.determinism", "reloading produced different batches")
    print(f"[ OK ] deterministic loading and batching: {det_ok}")

    # ---- report ------------------------------------------------------------
    elapsed = time.time() - t0
    n_fail, n_warn = rep.count(SEV_FAIL), rep.count(SEV_WARN)
    out = {
        "schema": "daic_multimodal_verification/1",
        "generated_by": "verify_daic_multimodal_records.py",
        "trained": False,
        "inputs": {
            "trainer": str(TRAINER_PATH.resolve()),
            "trainer_sha256": tsha,
            "trainer_sha_matches_frozen": tsha == FROZEN_TRAINER_SHA,
            "trainer_module_args_present": trainer_has_args,
            "parquet": str(args.parquet.resolve()),
            "parquet_sha256": sha256_file(args.parquet),
            "bert_model": BERT_MODEL,
        },
        "infer_dims": {"audio_dim": audio_dim, "vision_dim": vision_dim,
                       "expected": [exp_a, exp_v]},
        "records": {"loaded": n, "expected": EXPECTED_RECORDS,
                    "with_text": n_text, "with_audio": n_audio, "with_vision": n_vision},
        "dataset": {"len": len(ds), "max_len": MAX_LEN,
                    "getitem_keys": sorted(item.keys())},
        "batch_statistics": {"batch_size": args.batch_size,
                             "batches_checked": len(batches),
                             "batches": batch_report},
        "forward_pass": forward_report,
        "ablation": ablation,
        "supplementary_text_check": supplementary,
        "determinism": {"loading_and_batching_identical": det_ok},
        "validation_summary": {"fail": n_fail, "warn": n_warn,
                               "compatible": n_fail == 0},
        "findings": rep.findings,
        "digest": rep.digest(),
        "runtime_seconds": round(elapsed, 2),
    }
    try:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        with open(args.out, "w", encoding="utf-8") as fh:
            json.dump(out, fh, indent=2, sort_keys=True, ensure_ascii=True)
    except OSError as exc:
        print(f"[FATAL] cannot write verification report: {exc}")
        return 2

    print("-" * 78)
    print("=== VERIFICATION SUMMARY ===")
    print(f"records / dims      : {n} / ({audio_dim}, {vision_dim})")
    print(f"text/audio/vision   : {n_text}/{n_audio}/{n_vision}")
    if ablation.get("run"):
        print(f"ablation            : audio={ablation['audio_score']:.6g} "
              f"vision={ablation['vision_score']:.6g} text={ablation['text_score']:.6g}")
    print(f"FAIL / WARN         : {n_fail} / {n_warn}")
    print(f"digest              : {rep.digest()}")
    print(f"runtime             : {elapsed:.2f}s")
    print(f"report              : {args.out.resolve()}")

    if n_fail:
        print("\n[FAIL] the frozen trainer CANNOT consume this parquet as intended.")
        by: Dict[str, int] = {}
        for f in rep.findings:
            if f["severity"] == SEV_FAIL:
                by[f["check"]] = by.get(f["check"], 0) + 1
        for c, k in sorted(by.items()):
            print(f"    {c}: {k}")
        return 1

    print("\n[PASS] the frozen trainer consumes the multimodal parquet unmodified.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
