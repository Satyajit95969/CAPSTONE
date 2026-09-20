#!/usr/bin/env python3
"""
scripts/demo_predictions.py

Presentation-only, read-only display of one client round's local, pre-DP,
pre-aggregation model predictions on its 37 held-out patients. Does NOT
train, re-train, re-measure, or re-run anything - reads a single existing
artifact (a Phase A explain_logs/xai_ig_<session_id>_<timestamp>.json report)
and renders it.

ANCHOR SESSION: client-405c6057ab84 (hardcoded default). Chosen because it is
the ONLY client round on disk whose recorded metrics match this project's
mentor-demo target numbers (accuracy 0.6486486486486487, precision
0.4444444444444444, recall 0.7272727272727273, f1 0.5517241379310345) AND
whose own local DP receipt independently matches the "what happens after DP"
numbers (l2_norm_before=0.7657614350318909, l2_norm_after=449.96197509765625,
ratio=587.6007259087384x). See docs/IMPLEMENTATION_NOTES.md's "Session
attribution correction" section for the investigation that established this
and why two OTHER checkpoints' numbers must not be mixed with this one's.

Does NOT reload mentalbert_privacy_subset.pt. That checkpoint is overwritten
in place by every client round (no versioning) and the file that produced
THIS session's predictions no longer exists on disk - confirmed during this
script's own investigation. All predictions below come from the anchor
report's persisted `per_sample` array, computed at the time the correct
model was still current, never re-inferred here.

Reads from ~/.federated/data/anchor_session_backups/client-405c6057ab84/
(the permanent backup made when this script was built) if present, falling
back to the live explain_logs/ path - so this demo survives even if a later
client round's artifacts churn through explain_logs in ways that don't
overwrite THIS session's uniquely-timestamped files (they haven't been at
risk of overwrite - only the fixed-filename .pt/metrics.json ever were - but
the backup is read first regardless, as the more deliberately-preserved copy).

Usage:
    .venv\\Scripts\\python.exe scripts\\demo_predictions.py
    .venv\\Scripts\\python.exe scripts\\demo_predictions.py --xai-report <path>
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

IMPLEMENTATION_NOTES_PATH = Path(__file__).resolve().parent.parent / "docs" / "IMPLEMENTATION_NOTES.md"

# Keys read from IMPLEMENTATION_NOTES.md's "Fusion-head capacity sweep"
# section (2026-09-20 - supersedes the earlier 14-run sigma comparison,
# whose 6/7-collapse figure was itself a small-sample artifact) for the
# mandatory representativeness disclosure (block1b, below). Not hardcoded
# here - see _load_disclosure_facts().
_DISCLOSURE_KEYS = (
    "DISCLOSURE_SWEEP_TOTAL_RUNS",
    "DISCLOSURE_SWEEP_GENUINE_COUNT",
    "DISCLOSURE_SWEEP_GENUINE_RATE_PCT",
    "DISCLOSURE_SWEEP_BEST_RATE_PCT",
    "DISCLOSURE_SWEEP_WORST_RATE_PCT",
    "DISCLOSURE_SWEEP_BEST_RUN_F1",
    "DISCLOSURE_POSITIVE_COLLAPSE_F1",
    "DISCLOSURE_NEGATIVE_COLLAPSE_F1",
    "DISCLOSURE_BASE_RATE_POSITIVE_PCT",
)

ANCHOR_SESSION_ID = "client-405c6057ab84"
ANCHOR_BACKUP_PATH = (
    Path.home() / ".federated" / "data" / "anchor_session_backups" / ANCHOR_SESSION_ID
    / "xai_ig_client-405c6057ab84_1787825775504.json"
)
ANCHOR_LIVE_PATH = (
    Path.home() / ".federated" / "data" / "explain_logs"
    / "xai_ig_client-405c6057ab84_1787825775504.json"
)

# Every real client round's session_id is "client-<hex>" (pipeline.py:
# f"client-{uuid.uuid4().hex[:12]}"). Every diagnostic-script artifact this
# project has ever produced uses a "step-<name>" session_id instead
# (step-a6-verify, step-a7-nsteps5, step-a7-nsteps10 - enumerated by listing
# every session_id ever seen in explain_logs/ during this task's
# investigation). This script refuses to run against anything that isn't a
# real client round - a diagnostic artifact was never a genuine held-out
# evaluation of a deployable model, and showing it as one would misrepresent
# what it is.
REAL_SESSION_ID_RE = re.compile(r"^client-[0-9a-f]+$")

CONFIDENT_LOW, CONFIDENT_HIGH = 0.3, 0.7  # Step A7's threshold, reused as-is
WIDTH = 80


def _line(char: str = "=") -> str:
    return char * WIDTH


def find_xai_report(explicit_path: str | None) -> Path:
    if explicit_path:
        p = Path(explicit_path)
        if not p.exists():
            raise FileNotFoundError(f"--xai-report path does not exist: {p}")
        return p
    if ANCHOR_BACKUP_PATH.exists():
        return ANCHOR_BACKUP_PATH
    if ANCHOR_LIVE_PATH.exists():
        return ANCHOR_LIVE_PATH
    raise FileNotFoundError(
        f"Anchor session report not found at either "
        f"{ANCHOR_BACKUP_PATH} or {ANCHOR_LIVE_PATH}. "
        f"Pass --xai-report to point at a different real client-round report."
    )


def load_and_validate(path: Path) -> dict:
    data = json.loads(path.read_text())
    session_id = data.get("session_id", "")
    if not REAL_SESSION_ID_RE.match(session_id):
        raise SystemExit(
            f"REFUSING TO RUN: session_id {session_id!r} in {path} does not "
            f"match a real client round (expected 'client-<hex>'). This looks "
            f"like a diagnostic-script artifact (e.g. 'step-a6-verify', "
            f"'step-a7-nsteps5') from the Phase A investigation scripts, not a "
            f"genuine held-out evaluation of a deployable model. Refusing to "
            f"display it as one. Pass --xai-report to point at a real "
            f"client-<hex> session's report instead."
        )
    return data


def compute_confusion(per_sample: list[dict]) -> dict:
    tp = fp = tn = fn = 0
    for s in per_sample:
        true = s["true_label"]
        pred = 1 if s["predicted_positive_prob"] >= 0.5 else 0
        if true == 1 and pred == 1:
            tp += 1
        elif true == 0 and pred == 1:
            fp += 1
        elif true == 0 and pred == 0:
            tn += 1
        elif true == 1 and pred == 0:
            fn += 1
    return {"tp": tp, "fp": fp, "tn": tn, "fn": fn}


def verify_against_recorded_metrics(cm: dict, recorded: dict, n: int) -> list[str]:
    """Recomputes accuracy/precision/recall/F1 from the per-sample confusion
    matrix and compares against the report's own recorded eval_metrics.
    Returns a list of mismatch descriptions (empty if everything matches).
    Per the hard rule: if these disagree, the caller must STOP and report
    the discrepancy, never silently pick one number to display."""
    tp, fp, tn, fn = cm["tp"], cm["fp"], cm["tn"], cm["fn"]
    computed = {
        "accuracy": (tp + tn) / n if n else float("nan"),
        "precision": tp / (tp + fp) if (tp + fp) else 0.0,
        "recall": tp / (tp + fn) if (tp + fn) else 0.0,
    }
    computed["f1"] = (
        2 * computed["precision"] * computed["recall"] / (computed["precision"] + computed["recall"])
        if (computed["precision"] + computed["recall"]) else 0.0
    )
    mismatches = []
    for key in ("accuracy", "precision", "recall", "f1"):
        rec_val = recorded.get(key)
        comp_val = computed[key]
        if rec_val is None or abs(rec_val - comp_val) > 1e-9:
            mismatches.append(f"{key}: recorded={rec_val} computed={comp_val}")
    return mismatches


def block1_scope_header(session_id: str) -> None:
    print(_line())
    print(" SCOPE OF THESE RESULTS")
    print(_line())
    print()
    print(f" Session: {session_id}")
    print(" These are LOCAL, PRE-DP, PRE-aggregation results on 37 held-out")
    print(" patients this model never saw during training. They are honest")
    print(" held-out scores.")
    print()
    print(" They say NOTHING about the aggregated global model under")
    print(" differential privacy, which is measured separately and collapses")
    print(" (see Block 6 below).")
    print()


def _load_disclosure_facts() -> dict[str, str]:
    """Parses the machine-readable DISCLOSURE_* block from
    docs/IMPLEMENTATION_NOTES.md's "Fusion-head capacity sweep" section for
    block1b's mandatory representativeness disclosure, rather than
    hardcoding those figures in this script. Fails loudly - not a silent
    fallback - if the doc or any required key is missing: showing this
    disclosure with wrong or stale numbers would be worse than refusing to
    run, same reasoning as verify_against_recorded_metrics() above."""
    if not IMPLEMENTATION_NOTES_PATH.exists():
        raise SystemExit(
            f"REFUSING TO DISPLAY: cannot find {IMPLEMENTATION_NOTES_PATH} to "
            f"read the mandatory representativeness disclosure facts from."
        )
    text = IMPLEMENTATION_NOTES_PATH.read_text(encoding="utf-8")
    facts: dict[str, str] = {}
    for key in _DISCLOSURE_KEYS:
        m = re.search(rf"^{key}:\s*(.+?)\s*$", text, re.MULTILINE)
        if not m:
            raise SystemExit(
                f"REFUSING TO DISPLAY: {IMPLEMENTATION_NOTES_PATH} is missing "
                f"required disclosure key {key!r}. Refusing to show the "
                f"representativeness disclosure with an incomplete figure."
            )
        facts[key] = m.group(1)
    return facts


def block1b_representativeness_disclosure(session_id: str, recorded: dict) -> None:
    """Mandatory, never-suppressed disclosure of how representative this
    session's result is against the 55-run fusion-head capacity sweep
    documented in IMPLEMENTATION_NOTES.md (2026-09-20; supersedes the
    original 14-run sigma comparison, whose 0/7-genuine figure at each
    sigma was itself a small-sample artifact - a fresh baseline batch and
    a tripled sample both moved that number well above zero). This
    session's own F1 comes from its already-loaded report (the actual
    ground truth for this run); every sweep figure is read via
    _load_disclosure_facts(), not hardcoded."""
    facts = _load_disclosure_facts()
    this_f1 = recorded["f1"]
    pos_ceiling = float(facts["DISCLOSURE_POSITIVE_COLLAPSE_F1"])
    neg_ceiling = float(facts["DISCLOSURE_NEGATIVE_COLLAPSE_F1"])
    total_runs = facts["DISCLOSURE_SWEEP_TOTAL_RUNS"]
    genuine_count = facts["DISCLOSURE_SWEEP_GENUINE_COUNT"]
    genuine_rate = facts["DISCLOSURE_SWEEP_GENUINE_RATE_PCT"]
    best_rate = facts["DISCLOSURE_SWEEP_BEST_RATE_PCT"]
    worst_rate = facts["DISCLOSURE_SWEEP_WORST_RATE_PCT"]
    best_run_f1 = float(facts["DISCLOSURE_SWEEP_BEST_RUN_F1"])
    base_rate = facts["DISCLOSURE_BASE_RATE_POSITIVE_PCT"]
    exceeds_both_ceilings = this_f1 > max(pos_ceiling, neg_ceiling)

    print(_line())
    print(" REPRESENTATIVENESS DISCLOSURE - READ BEFORE THE RESULTS BELOW")
    print(_line())
    print()
    if exceeds_both_ceilings:
        print(f" This session ({session_id}, F1={this_f1:.4f}) achieved genuine class")
        print(f" discrimination - consistent with roughly {genuine_rate}% of runs")
        print(f" ({genuine_count} of {total_runs}) in a capacity sweep across fusion-head")
        print(" sizes. This is an above-typical result, but NOT unique: at least one")
        print(f" other measured run reached F1={best_run_f1:.4f}, exceeding this session.")
    else:
        print(f" This session ({session_id}, F1={this_f1:.4f}) does NOT exceed the fixed")
        print(" degenerate-mode ceilings below - it is consistent with the majority")
        print(f" collapse pattern seen in {total_runs} measured runs ({genuine_count} of")
        print(f" which, {genuine_rate}%, achieved genuine discrimination).")
    print()
    print(f" Across a {total_runs}-run sweep varying fusion-head size (same data, lr,")
    print(" epochs, batch size, splits and DP settings throughout), the genuine-")
    print(f" discrimination rate ranged from {worst_rate}% (smallest heads tested -")
    print(f" total collapse, no exceptions) up to {best_rate}% depending on head size.")
    print(" No head size tested reliably avoids collapse in most runs.")
    print()
    print(" The two degenerate modes have fixed scores on this eval split's")
    print(f" {base_rate}% positive base rate: predict-all-positive gives F1 exactly")
    print(f" {facts['DISCLOSURE_POSITIVE_COLLAPSE_F1']}, predict-all-negative gives F1")
    print(f" exactly {facts['DISCLOSURE_NEGATIVE_COLLAPSE_F1']}.")
    if exceeds_both_ceilings:
        print(f" This session's {this_f1:.4f} exceeds both ceilings, which is what makes")
        print(" it non-degenerate - genuinely useful, but one good result among many")
        print(" measured runs, not the sole exception to an otherwise-universal rule.")
    print()


def block2_per_patient_table(per_sample: list[dict]) -> None:
    print(_line())
    print(" PER-PATIENT PREDICTIONS (37 held-out patients, anonymised)")
    print(_line())
    print()
    print(" No real patient identifiers are shown - each row is an anonymous")
    print(" sequential ID assigned only for this display.")
    print()

    rows = []
    for i, s in enumerate(per_sample):
        prob = s["predicted_positive_prob"]
        true_label = s["true_label"]
        pred_class = 1 if prob >= 0.5 else 0
        correct = (pred_class == true_label)
        band = "CONFIDENT" if (prob < CONFIDENT_LOW or prob > CONFIDENT_HIGH) else "UNCERTAIN"
        rows.append({
            "anon_id": f"P{i+1:02d}",
            "true_label": true_label,
            "prob": prob,
            "pred_class": pred_class,
            "correct": correct,
            "band": band,
        })

    # Confident rows first (there may be very few - genuinely rare on this
    # checkpoint, see Block 4), then uncertain - visually separable per spec.
    confident_rows = [r for r in rows if r["band"] == "CONFIDENT"]
    uncertain_rows = [r for r in rows if r["band"] == "UNCERTAIN"]

    header = f" {'ID':<5} {'True':<6} {'Prob':<8} {'Pred':<6} {'Result':<10} {'Band'}"
    print(header)
    print(" " + "-" * (WIDTH - 2))
    for group_name, group in (("CONFIDENT", confident_rows), ("UNCERTAIN", uncertain_rows)):
        if not group:
            continue
        print(f" -- {group_name} ({len(group)}) --")
        for r in group:
            label_str = "PHQ+" if r["true_label"] == 1 else "PHQ-"
            pred_str = "PHQ+" if r["pred_class"] == 1 else "PHQ-"
            result_str = "correct" if r["correct"] else "WRONG"
            print(f" {r['anon_id']:<5} {label_str:<6} {r['prob']:<8.4f} {pred_str:<6} {result_str:<10} {r['band']}")
    print()
    return rows


def block3_confusion_matrix(cm: dict, recorded: dict) -> None:
    tp, fp, tn, fn = cm["tp"], cm["fp"], cm["tn"], cm["fn"]
    print(_line())
    print(" CONFUSION MATRIX")
    print(_line())
    print()
    print("                        Predicted PHQ+       Predicted PHQ-")
    print(f"   Actual PHQ+          {tp:>3} (caught)        {fn:>3} (missed)")
    print(f"   Actual PHQ-          {fp:>3} (false alarm)   {tn:>3} (correctly cleared)")
    print()
    print(f" Accuracy:  {recorded['accuracy']:.4f}")
    print(f" Precision: {recorded['precision']:.4f}")
    print(f" Recall:    {recorded['recall']:.4f}")
    print(f" F1:        {recorded['f1']:.4f}")
    print()
    print(" High recall with lower precision means the model errs toward")
    print(" flagging - for a SCREENING tool this is the safer error to make:")
    print(" a false alarm gets ruled out by a clinician, a missed patient")
    print(" does not.")
    print()


def block4_confidence_distribution(rows: list[dict]) -> None:
    n = len(rows)
    n_uncertain = sum(1 for r in rows if r["band"] == "UNCERTAIN")
    pct = 100.0 * n_uncertain / n if n else 0.0
    print(_line())
    print(" CONFIDENCE DISTRIBUTION")
    print(_line())
    print()
    print(f" {n_uncertain} of {n} predictions ({pct:.1f}%) fall in the 0.3-0.7")
    print(" uncertain band on THIS session's model.")
    print()
    print(" Separately, a documented system-level finding (Steps A3-A8,")
    print(" docs/IMPLEMENTATION_NOTES.md, measured on a DIFFERENT client")
    print(" round's checkpoint, session client-aef1978aef7f - NOT this one):")
    print(" per-patient attribution is not reliable, because most predictions")
    print(" sit in this same uncertain band AND two independent attribution")
    print(" methods (Integrated Gradients and a modality-ablation check)")
    print(" disagree with each other at the individual-patient level (IG")
    print(" 37/37 one modality; ablation 11/20/6 - close to a random split).")
    print(" This is stated as a limitation of the system, not softened.")
    print()


def block5_explainability(aggregate_normalized: dict) -> None:
    print(_line())
    print(" EXPLAINABILITY (COHORT LEVEL ONLY)")
    print(_line())
    print()
    print(" Aggregate modality attribution across this session's 37 held-out")
    print(" patients (Integrated Gradients, cohort-level average):")
    for modality in ("text", "audio", "vision"):
        v = aggregate_normalized.get(modality)
        if v is not None:
            print(f"   {modality:<8s} {v:.4f}  ({v*100:.1f}%)")
    print()
    print(" This means: averaged across the whole held-out group, text")
    print(" contributed most to the model's predictions, audio and vision")
    print(" less. This is a COHORT-level figure - no claim is made about")
    print(" which modality drove any single patient's prediction.")
    print()
    print(" Corrected-reference finding: an earlier measurement using a")
    print(" zero-vector reference for audio/vision found audio dominant at")
    print(" a similar level. That was traced to audio's raw values being")
    print(" roughly 16.6x text's and 11.1x vision's - a zero reference point")
    print(" is far more extreme, out-of-distribution for audio than for the")
    print(" other two, which inflated its apparent importance regardless of")
    print(" genuine effect. Fixing the reference to a proper in-distribution")
    print(" average (the modality means above) reversed the result.")
    print()


def block6_after_dp(dp_receipt: dict | None) -> None:
    print(_line())
    print(" WHAT HAPPENS AFTER DIFFERENTIAL PRIVACY")
    print(_line())
    print()
    if dp_receipt:
        before = dp_receipt["params"]["l2_norm_before"]
        after = dp_receipt["params"]["l2_norm_after"]
        ratio = after / before
        print(f" This session's own local update, before/after DP noise:")
        print(f"   Local signal (pre-DP delta L2):  {before:.4f}")
        print(f"   Post-DP noise (delta L2):         {after:.4f}")
        print(f"   Ratio:                             {ratio:.1f}x")
    else:
        print(" NOT MEASURED - no local DP receipt found for this session.")
    print()
    print(" The aggregated GLOBAL model, after DP and aggregation across")
    print(" clients, measured across four separate experiments:")
    print("   accuracy 0.7027  (= 26 of 37 - exactly what predicting")
    print("                      'not depressed' for every patient scores)")
    print("   F1       0.0000")
    print(" Sources: docs/IMPLEMENTATION_NOTES.md Step 13 (mean aggregation,")
    print(" line 317; trimmed-mean, line 318), Step 14 ARM 2 round 1 (line")
    print(" 445), and scripts/demo_fl_algorithms.py's FedAdam/FedYogi")
    print(" trajectories (all 5 rounds each).")
    print()
    print(" Scope: this says DP-SGD fails at n=3 clients, in this")
    print(" configuration, on this corpus size. It is NOT a general claim")
    print(" about differential privacy.")
    print()


def find_dp_receipt(session_id: str) -> dict | None:
    """Reads this session's local DP receipt if present (backup copy first,
    then live receipts/ dir) - used only for Block 6's before/after/ratio
    numbers. Returns None (-> NOT MEASURED in Block 6) if not found; never
    estimated."""
    backup_dir = Path.home() / ".federated" / "data" / "anchor_session_backups" / session_id
    if backup_dir.exists():
        for p in backup_dir.glob("receipt_*.json"):
            try:
                data = json.loads(p.read_text())
            except (json.JSONDecodeError, OSError):
                continue
            if data.get("session_id") == session_id and data.get("operation") == "dp_process_update":
                return data
    live_dir = Path.home() / ".federated" / "data" / "receipts"
    if live_dir.exists():
        for p in live_dir.glob("*.json"):
            try:
                data = json.loads(p.read_text())
            except (json.JSONDecodeError, OSError):
                continue
            if data.get("session_id") == session_id and data.get("operation") == "dp_process_update":
                return data
    return None


def main() -> int:
    ap = argparse.ArgumentParser(description="Read-only display of one client round's held-out predictions")
    ap.add_argument("--xai-report", default=None,
                     help="path to a real client-round xai_ig_*.json; default: the anchor session")
    args = ap.parse_args()

    report_path = find_xai_report(args.xai_report)
    data = load_and_validate(report_path)
    session_id = data["session_id"]
    per_sample = data["per_sample"]
    n = len(per_sample)

    cm = compute_confusion(per_sample)
    mismatches = verify_against_recorded_metrics(cm, data["eval_metrics"], n)
    if mismatches:
        print("REFUSING TO DISPLAY: computed metrics disagree with the report's "
              "own recorded eval_metrics. This means either the per_sample data "
              "or the recorded aggregate is wrong - displaying either number "
              "without resolving this would risk showing a false one.",
              file=sys.stderr)
        for m in mismatches:
            print(f"  MISMATCH: {m}", file=sys.stderr)
        return 1

    dp_receipt = find_dp_receipt(session_id)

    block1_scope_header(session_id)
    block1b_representativeness_disclosure(session_id, data["eval_metrics"])
    rows = block2_per_patient_table(per_sample)
    block3_confusion_matrix(cm, data["eval_metrics"])
    block4_confidence_distribution(rows)
    block5_explainability(data.get("aggregate_normalized", {}))
    block6_after_dp(dp_receipt)

    print(_line())
    print(" Source report:")
    print(f"   {report_path}")
    print(f" Held-out patients: {n}  (Fix E2 stratified_split(), seed=42, deterministic)")
    print(_line())
    return 0


if __name__ == "__main__":
    sys.exit(main())
