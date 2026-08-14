#!/usr/bin/env python3
"""
run_exp7_ablation.py - Phase 16 / Experiment 7: the modality-ablation criterion.

WHY THIS IS A SEPARATE SCRIPT AND NOT PART OF run_exp7.py
---------------------------------------------------------
No experiment runner in this repository has ever called
modality_ablation_importance(). Every published ablation number - Phase 9 (113),
the 188 preliminary, and the Phase 9.5 official 2.3365 - was produced by
executing trainer_mentalbert_daic.py AS A SCRIPT.

That matters because of trainer line 442:

    tokenizer = AutoTokenizer.from_pretrained(args.bert_model_name) \
                if "args" in globals() else None

`args` is a MODULE GLOBAL that the trainer's own main() assigns (line 522,
comment: "used in modality_ablation_importance for tokenizer retrieval"). When
the trainer is imported as a module - which every runner and verifier does -
`args` does not exist, the else branch fires, and text_score is 0.0. Stage S4
observed exactly this.

The fix is NOT to inject `T.args`, patch the module, or rewrite the ablation.
The fix is to invoke the frozen trainer through its OWN COMMAND-LINE ENTRY POINT
in a subprocess, which is the execution semantics that produced every historical
number. This script does that and nothing else. It does not import the trainer.

WHAT SCRIPT MODE ACTUALLY DOES - AND THE LIMIT THAT IMPOSES
-----------------------------------------------------------
Trainer main() in --mode supervised performs a SINGLE 90/10 split
(val_split=0.1), not the 5x5 CV. Phase 9.5's log records exactly this:
"train=170 val=18". So the number produced here is like-for-like comparable with
the historical 2.3365, and is NOT comparable with, and does not substitute for,
the cross-validated metrics from run_exp7.py. Both facts are recorded in the
output.

OVERWRITE HAZARD - GUARDED
--------------------------
The trainer writes modality_ablation.json, training_report.json, eval_preds.csv
and the model/delta files into Path(--out-path).parent. Its DEFAULT out-path is
./trainer_outputs/mentalbert_privacy_subset.pt, whose parent is trainer_outputs/
- the directory holding the FROZEN Phase 9.5 modality_ablation.json. Running the
trainer with defaults would silently destroy that artifact. Every invocation here
is therefore forced into an isolated per-run directory, and the script refuses to
proceed if the resolved output directory is trainer_outputs/ itself.

Usage:
    python run_exp7_ablation.py                 # multimodal + text-only control
    python run_exp7_ablation.py --arms mm       # multimodal only
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

REPO = Path(__file__).resolve().parent.parent
TRAINER = REPO / "trainer_mentalbert_daic.py"

FROZEN_TRAINER_SHA = "65b1902e2ba8dd996c6960db1a6595d84fd48500f4d8707cb79688093d7a230b"
FROZEN_TEXT_PARQUET_SHA = "9a241851760727779fcaa4d50041b8bdc4406fe6b66ddbbd7105f4ce4fc55f00"
FROZEN_MM_PARQUET_SHA = "1ac9f53e6102ec0dbaab84dcfdfa3f2e70f2b4a1867a841ea2b7e3ba24c67a95"

# The frozen Phase 9.5 artifact this script must never touch.
PROTECTED = REPO / "trainer_outputs" / "modality_ablation.json"
PROTECTED_SHA_KNOWN = None  # captured at runtime, re-checked afterwards

# Historical reference, for context only - NOT an acceptance threshold.
HISTORICAL = {
    "phase9_113_baseline": {"audio": 0.0, "vision": 0.0, "text": 1.9956400692462921},
    "local_188_preliminary": {"audio": 0.0, "vision": 0.0, "text": 2.2671899050474167},
    "phase95_official": {"audio": 0.0, "vision": 0.0, "text": 2.3365455344319344},
}

# Frozen trainer recipe - identical to Phase 9.5's invocation.
EPOCHS = 3
BATCH_SIZE = 8
LR = 2e-5
BINARIZE_THRESHOLD = 10.0
BERT_MODEL = "mental/mental-bert-base-uncased"


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for b in iter(lambda: fh.read(8192), b""):
            h.update(b)
    return h.hexdigest()


def abort(msg: str) -> "NoReturn":  # noqa: F821
    print(f"\n[ABORT] {msg}", file=sys.stderr)
    raise SystemExit(1)


def run_trainer(parquet: Path, out_dir: Path, device: str,
                timeout_s: int) -> Dict[str, Any]:
    """Invoke the FROZEN trainer through its own CLI. No import, no patching."""
    out_dir.mkdir(parents=True, exist_ok=True)
    resolved = out_dir.resolve()
    if resolved == (REPO / "trainer_outputs").resolve():
        abort("refusing to run with output directory trainer_outputs/ - that would "
              "overwrite the frozen Phase 9.5 modality_ablation.json")

    cmd = [
        sys.executable, str(TRAINER),
        "--input", str(parquet),
        "--mode", "supervised",
        "--device", device,
        "--epochs", str(EPOCHS),
        "--batch-size", str(BATCH_SIZE),
        "--lr", str(LR),
        "--binarize",
        "--binarize-threshold", str(BINARIZE_THRESHOLD),
        "--bert-model-name", BERT_MODEL,
        "--out-path", str(out_dir / "model.pt"),
        "--delta-path", str(out_dir / "delta.pt"),
        "--store-root", str(out_dir / "secure_store"),
    ]
    print(f"  [exec] {' '.join(cmd[1:6])} ... --out-path {out_dir/'model.pt'}")
    t0 = time.time()
    proc = subprocess.run(cmd, cwd=str(REPO), capture_output=True, text=True,
                          timeout=timeout_s)
    elapsed = round(time.time() - t0, 1)
    log_path = out_dir / "trainer_stdout.log"
    log_path.write_text((proc.stdout or "") + "\n=== STDERR ===\n" + (proc.stderr or ""),
                        encoding="utf-8")

    ablation_path = out_dir / "modality_ablation.json"
    report_path = out_dir / "training_report.json"
    result: Dict[str, Any] = {
        "command": cmd[1:], "returncode": proc.returncode, "seconds": elapsed,
        "log": str(log_path), "ablation_file": str(ablation_path),
    }
    if proc.returncode != 0:
        result["error"] = (proc.stderr or "")[-800:]
        return result
    if not ablation_path.is_file():
        result["error"] = "trainer completed but wrote no modality_ablation.json"
        return result
    with open(ablation_path, "r", encoding="utf-8") as fh:
        result["ablation"] = json.load(fh)
    if report_path.is_file():
        with open(report_path, "r", encoding="utf-8") as fh:
            result["training_report"] = json.load(fh)
    return result


def main() -> int:
    ap = argparse.ArgumentParser(
        description="Experiment 7 modality ablation via the frozen trainer's own CLI")
    ap.add_argument("--parquet", default="dataset_build/daic_records_multimodal.parquet")
    ap.add_argument("--text-parquet", default="daic_records.parquet")
    ap.add_argument("--out-dir", default="trainer_outputs/exp7_modality")
    ap.add_argument("--arms", default="mm,text",
                    help="'mm' = multimodal, 'text' = frozen text-only control")
    ap.add_argument("--device", default=None)
    ap.add_argument("--timeout", type=int, default=7200)
    a = ap.parse_args()
    os.chdir(REPO)

    print("=" * 74)
    print("EXPERIMENT 7 - MODALITY ABLATION (frozen trainer, script execution)")
    print("=" * 74)

    if not TRAINER.is_file():
        abort(f"frozen trainer not found: {TRAINER}")
    tsha = sha256(TRAINER)
    if tsha != FROZEN_TRAINER_SHA:
        abort(f"trainer SHA {tsha} != frozen {FROZEN_TRAINER_SHA}")
    print(f"[OK] frozen trainer SHA verified: {tsha}")

    mm = Path(a.parquet)
    txt = Path(a.text_parquet)
    if not mm.is_file():
        abort(f"multimodal parquet not found: {mm}")
    if sha256(mm) != FROZEN_MM_PARQUET_SHA:
        abort(f"multimodal parquet SHA mismatch - not the S3/S4-verified artifact")
    print(f"[OK] multimodal parquet SHA verified")
    if txt.is_file() and sha256(txt) != FROZEN_TEXT_PARQUET_SHA:
        abort("frozen text parquet SHA mismatch")

    protected_before = sha256(PROTECTED) if PROTECTED.is_file() else None
    if protected_before:
        print(f"[OK] protecting frozen Phase 9.5 ablation: {protected_before[:32]}...")

    device = a.device or ("cuda" if _cuda_available() else "cpu")
    print(f"[INFO] device={device}  epochs={EPOCHS} batch={BATCH_SIZE} lr={LR}")
    print("[INFO] script mode uses a SINGLE 90/10 split (val_split=0.1), exactly as "
          "Phase 9.5 did - NOT the 5x5 CV")
    print("-" * 74)

    arms = [x.strip() for x in a.arms.split(",") if x.strip()]
    out_root = Path(a.out_dir)
    results: Dict[str, Any] = {}

    if "mm" in arms:
        print("[ARM mm] multimodal parquet (audio 154 + vision 84)")
        results["multimodal"] = run_trainer(mm, out_root / "ablation_multimodal",
                                            device, a.timeout)
    if "text" in arms and txt.is_file():
        print("[ARM text] frozen text-only parquet - reproduces the historical 0/0/x pattern")
        results["text_only"] = run_trainer(txt, out_root / "ablation_text_only",
                                           device, a.timeout)

    # ---- the frozen artifact must be untouched ----------------------------
    if protected_before:
        after = sha256(PROTECTED) if PROTECTED.is_file() else None
        if after != protected_before:
            abort("the frozen Phase 9.5 modality_ablation.json was modified - "
                  "this must never happen")
        print(f"\n[OK] frozen Phase 9.5 modality_ablation.json unchanged")
    if sha256(TRAINER) != FROZEN_TRAINER_SHA:
        abort("trainer changed during execution")

    # ---- verdict ------------------------------------------------------------
    findings: List[str] = []
    mm_abl = (results.get("multimodal") or {}).get("ablation") or {}
    audio = mm_abl.get("audio_score")
    vision = mm_abl.get("vision_score")
    text = mm_abl.get("text_score")
    criterion = {
        "audio_nonzero": bool(audio not in (None, 0.0)),
        "vision_nonzero": bool(vision not in (None, 0.0)),
        "text_positive": bool(text is not None and text > 0.0),
    }
    if not criterion["audio_nonzero"]:
        findings.append("audio_score is zero or missing")
    if not criterion["vision_nonzero"]:
        findings.append("vision_score is zero or missing")
    if not criterion["text_positive"]:
        findings.append("text_score is not positive")

    out = {
        "schema": "exp7_modality_ablation/1",
        "experiment": "Phase 16 / Exp 7 - multimodal activation",
        "execution_semantics": {
            "method": "frozen trainer invoked via its own CLI in a subprocess",
            "why": ("modality_ablation_importance needs the module global `args`, "
                    "which trainer main() assigns (line 522). Importing the module "
                    "leaves it undefined and forces text_score to 0.0."),
            "monkey_patched": False,
            "trainer_modified": False,
            "split": "single 90/10 (val_split=0.1), NOT the 5x5 CV",
            "comparable_to": "the historical Phase 9.5 ablation (train=170 val=18)",
            "not_comparable_to": "run_exp7.py cross-validated metrics",
        },
        "frozen_sha": {
            "trainer": tsha,
            "multimodal_parquet": FROZEN_MM_PARQUET_SHA,
            "phase95_ablation_unchanged": True if protected_before else None,
        },
        "historical_reference": HISTORICAL,
        "results": results,
        "primary_criterion": criterion,
        "criterion_met": not findings,
        "findings": findings,
        "caveat": ("A non-zero ablation establishes only that the modality reaches "
                   "the computation graph. It is NOT evidence of predictive "
                   "usefulness; that comes from the cross-validated arms."),
    }
    out_root.mkdir(parents=True, exist_ok=True)
    dest = out_root / "exp7_modality_ablation.json"
    with open(dest, "w", encoding="utf-8") as fh:
        json.dump(out, fh, indent=2, sort_keys=True)

    print("-" * 74)
    print("=== ABLATION SUMMARY ===")
    for name, res in results.items():
        abl = res.get("ablation") or {}
        if abl:
            print(f"  {name:12s} audio={abl.get('audio_score'):.6g}  "
                  f"vision={abl.get('vision_score'):.6g}  "
                  f"text={abl.get('text_score'):.6g}   ({res['seconds']}s)")
        else:
            print(f"  {name:12s} FAILED: {res.get('error', 'unknown')[:120]}")
    print(f"\n  historical Phase 9.5 : audio=0.0 vision=0.0 "
          f"text={HISTORICAL['phase95_official']['text']:.6g}")
    print(f"  criterion met        : {out['criterion_met']}  {findings}")
    print(f"  written              : {dest}")
    return 0 if out["criterion_met"] else 1


def _cuda_available() -> bool:
    try:
        import torch
        return bool(torch.cuda.is_available())
    except Exception:
        return False


if __name__ == "__main__":
    sys.exit(main())
