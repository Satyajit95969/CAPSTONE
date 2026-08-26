#!/usr/bin/env python3
"""
scripts/clinical_narrative_agent.py

Phase B — Clinical Narrative Agent.

Turns Phase A's Integrated-Gradients attribution output (explain_logs/xai_ig_*.json)
into a clinician-readable report. Client-side, local disk only, never uploaded
- the same constraint Phase A itself was built under. Does not touch the live
pipeline, does not compute any new attribution or prediction, does not call
any external network service.

Design (approved, Step B1/B2), driven directly by the four constraints
docs/IMPLEMENTATION_NOTES.md's "Phase A: what the attribution mechanism can
and cannot support" section places on this agent:
    - No per-patient modality percentages
    - No claim that a specific modality "drove" an individual prediction
    - Must surface the model's uncertainty (36/37 held-out predictions sit in
      the 0.3-0.7 band, on the checkpoint that finding was measured against)
    - Aggregate, cohort-level modality statements ARE supportable

Structure:
    Part 1 - COHORT narrative. One Ollama call. The prompt is built from an
    AGGREGATE-ONLY facts dict - it never contains per_sample records. This
    makes the "no per-patient claims" constraint structural, not just an
    instruction the LLM might ignore: it cannot narrate what it was never
    given.
    Part 2 - PER-PATIENT table. Fully deterministic, zero LLM involvement.
    Prediction + calibrated confidence band + a fixed review recommendation.
    No modality column, no free-text "reason." The abstention rule (Step B2
    point 3) uses the SAME 0.3-0.7 band Step A7 already established, not a
    new threshold.

The facts tables and per-patient table are written unconditionally, even if
Ollama is unreachable - report generation never depends on the LLM
succeeding (Step B2 point 4).

Usage:
    .venv\\Scripts\\python.exe scripts\\clinical_narrative_agent.py
    .venv\\Scripts\\python.exe scripts\\clinical_narrative_agent.py --xai-report <path>
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

from pymongo import MongoClient

sys.path.insert(0, str(Path(__file__).resolve().parent))
from ollama_narration_utils import call_ollama, check_consistency  # noqa: E402

DEFAULT_EXPLAIN_DIR = Path.home() / ".federated" / "data" / "explain_logs"
DEFAULT_OUT_DIR = Path.home() / ".federated" / "data" / "clinical_reports"
DEFAULT_OLLAMA_URL = "http://localhost:11434"
DEFAULT_MONGO_URI = "mongodb://localhost:27017"
DEFAULT_DB = "federated_multimodal"

# Confidence band threshold - Step A7's own threshold, reused deliberately
# rather than inventing a new one (Step B2 point 3).
CONFIDENT_LOW, CONFIDENT_HIGH = 0.3, 0.7

# ---------------------------------------------------------------------------
# Fixed, approved safety wording (Step B1, approved with two additions in
# Step B2's instructions). None of this is LLM-generated - it is identical
# on every report, by design, so its wording cannot drift or be
# LLM-paraphrased into something weaker.
# ---------------------------------------------------------------------------
DOCUMENT_DISCLAIMER = (
    "This report is a decision-support artifact only. It does not diagnose "
    "depression or any mental health condition, and it must not be used as "
    "the sole basis for a clinical decision. All predictions come from an "
    "automated model that reaches F1≈{f1:.2f} on its own held-out "
    "evaluation set — a screening aid with substantial error, not a "
    "diagnostic instrument. This model was trained on {n_train} records from "
    "a single research corpus (DAIC-WOZ). It has not been validated on any "
    "clinical population, and its performance on patients unlike that corpus "
    "is unknown. Every prediction in this report requires clinician review "
    "before any action is taken."
)

UNCERTAINTY_STATEMENT = (
    "On this round's held-out evaluation set (n={n_eval}), the model's "
    "predicted probability fell between {low:.1f} and {high:.1f} for "
    "{n_uncertain} of {n_eval} patients ({pct:.0f}%). The model rarely "
    "commits confidently to either class. A probability of 0.55 and a "
    "probability of 0.65 should be read as functionally similar — both "
    "reflect low confidence, not meaningfully different risk levels."
)

PER_PATIENT_HEADER_NOTE = (
    "This table intentionally does not attribute individual predictions to "
    "text, audio, or video. Per-patient attribution was tested and found "
    "unreliable (Phase A, Steps A3-A7, docs/IMPLEMENTATION_NOTES.md) — a "
    "false per-patient reason is more harmful than none. Aggregate, "
    "cohort-level modality attribution is reported in the section above, and "
    "the raw per-sample attribution data, with its documented limitations, "
    "is in explain_logs/."
)

ABSTENTION_TEXT = "Insufficient signal for automated interpretation. Clinician review required."

OUTSIDE_BAND_TEXT = (
    "Predicted probability ({prob:.2f}) falls outside this cohort's typical "
    "near-boundary range ({low:.1f}–{high:.1f}). This does not indicate a "
    "reliable individual finding — the model's overall reliability on "
    "this evaluation set is limited (F1≈{f1:.2f}) — and clinician "
    "review is still required."
)

METHOD_CAVEAT = (
    "Modality attribution (text/audio/vision) is reported at the cohort "
    "level only. Two independent methods (Integrated Gradients and a "
    "modality-ablation check) were tested for per-patient reliability and "
    "found NOT to agree at the individual level (Steps A6-A7, "
    "docs/IMPLEMENTATION_NOTES.md) — most likely because this model's "
    "predictions cluster tightly around the decision boundary rather than "
    "committing confidently either way. The aggregate, cohort-level split "
    "below remains supportable; per-patient attribution does not."
)


def _find_latest_xai_report(explain_dir: Path) -> Path:
    candidates = sorted(explain_dir.glob("xai_ig_*.json"), key=lambda p: p.stat().st_mtime)
    if not candidates:
        raise FileNotFoundError(
            f"No xai_ig_*.json found under {explain_dir} - run Phase A "
            "(XAI_ENABLED=1 client round, or one of the scripts/xai_*.py diagnostics) first."
        )
    return candidates[-1]


def _lookup_num_train_records() -> int | None:
    """Reads num_train_records from Step A5's cached baseline file, if
    present. Avoids hardcoding the 149-record figure inline; falls back to
    None (disclaimer omits the exact count rather than guessing) if the
    cache isn't there."""
    cache_path = Path.home() / ".federated" / "data" / "xai_baselines" / "modality_baselines.pt"
    if not cache_path.exists():
        return None
    try:
        import torch
        data = torch.load(cache_path, map_location="cpu")
        return data.get("num_train_records")
    except Exception:
        return None


def _lookup_round_id(session_id: str, mongo_uri: str, db_name: str) -> int | None:
    try:
        client = MongoClient(mongo_uri, serverSelectionTimeoutMS=5000)
        client.admin.command("ping")
        db = client[db_name]
        doc = db["model_updates"].find_one({"session_id": session_id})
        client.close()
        return doc["round_id"] if doc else None
    except Exception:
        return None


def build_cohort_facts(xai_report: dict, round_id: int | None) -> dict:
    """
    Aggregate-only facts. Deliberately excludes xai_report["per_sample"] -
    this dict is fed directly into the LLM prompt, and the per-patient
    section is built separately (see build_per_patient_rows()) from data
    that never reaches the prompt. This is what makes "no per-patient
    claims in the narrative" structural rather than instructional.
    """
    per_sample = xai_report.get("per_sample", [])
    n_eval = len(per_sample)
    n_uncertain = sum(
        1 for s in per_sample
        if CONFIDENT_LOW <= s.get("predicted_positive_prob", 0.5) <= CONFIDENT_HIGH
    )

    return {
        "round_id": round_id,
        "session_id": xai_report.get("session_id"),
        "num_train_records": _lookup_num_train_records(),
        "num_patients_in_eval_set": n_eval,
        "eval_metrics": xai_report.get("eval_metrics", {}),
        "modality_attribution_aggregate_normalized": xai_report.get("aggregate_normalized", {}),
        "baseline_kind": xai_report.get("baseline_kind"),
        "n_steps": xai_report.get("n_steps"),
        "prediction_confidence": {
            "num_uncertain_0.3_to_0.7": n_uncertain,
            "num_outside_band": n_eval - n_uncertain,
            "band_low": CONFIDENT_LOW,
            "band_high": CONFIDENT_HIGH,
        },
        "method_caveat": METHOD_CAVEAT,
    }


def render_cohort_facts_markdown(facts: dict) -> str:
    em = facts.get("eval_metrics", {})
    an = facts.get("modality_attribution_aggregate_normalized", {})
    pc = facts["prediction_confidence"]
    lines = []
    lines.append(f"## Round {facts['round_id']} — cohort facts (session: `{facts['session_id']}`)\n")
    lines.append(
        "**Every number in this section comes directly from Phase A's attribution report "
        "(`explain_logs/`). Nothing here was computed or estimated by the LLM.** The narrative "
        "below was generated from exactly this data — check it against this table, not the "
        "other way around.\n"
    )
    lines.append("| Field | Value |")
    lines.append("|---|---|")
    lines.append(f"| Training records (this local model) | {facts.get('num_train_records', 'unknown')} |")
    lines.append(f"| Patients in held-out evaluation set | {facts['num_patients_in_eval_set']} |")
    for k, v in em.items():
        if isinstance(v, (int, float)):
            lines.append(f"| Eval metric: {k} | {v:.4f} |")
        else:
            lines.append(f"| Eval metric: {k} | {v} |")
    for k, v in an.items():
        lines.append(f"| Modality attribution (aggregate, normalized): {k} | {v:.4f} |")
    lines.append(f"| IG attribution baseline kind | {facts.get('baseline_kind', 'n/a')} |")
    lines.append(f"| Predictions in uncertain band ({pc['band_low']:.1f}-{pc['band_high']:.1f}) | "
                  f"{pc['num_uncertain_0.3_to_0.7']} of {facts['num_patients_in_eval_set']} |")
    lines.append(f"| Predictions outside uncertain band | {pc['num_outside_band']} of {facts['num_patients_in_eval_set']} |")
    lines.append("")
    lines.append(f"**Method caveat**: {facts['method_caveat']}\n")
    return "\n".join(lines)


def build_cohort_prompt(facts: dict) -> str:
    facts_json = json.dumps(facts, indent=2, default=str)
    return f"""You are a clinical-report narration assistant. Below is a JSON object of
ALREADY-COMPUTED, COHORT-LEVEL facts about one federated learning round's
local model, evaluated on its own held-out patient set. Your job is to
narrate these facts in plain English for a clinician reader. You do not
compute anything, and you have NOT been given any individual patient's data
- only cohort-level aggregates.

STRICT RULES:
1. Use ONLY numbers that literally appear in the JSON below. Do not compute,
   estimate, round differently, or introduce any new number.
2. You have not been given any per-patient record. Do not refer to "this
   patient" or any individual case, real or hypothetical - only the cohort
   as a whole.
3. Do not claim any modality (text/audio/video) "drove" or "explains" any
   individual prediction - you have no individual data to base that on.
   You MAY describe the aggregate modality split given in the JSON as a
   cohort-level pattern.
4. Explicitly restate, in your own words, the "method_caveat" field's point:
   that per-patient attribution was tested and found unreliable, and that
   only the aggregate split is supportable.
5. Explicitly restate the prediction_confidence numbers: most of this
   cohort's predictions are low-confidence (near the decision boundary), and
   say plainly that this limits how much weight any single prediction should
   be given.
6. Do not make a diagnostic or clinical claim of any kind. This is a
   description of model behavior on a held-out evaluation set, not a
   statement about any patient's mental health.
7. Write 3-5 short paragraphs. No headings, no bullet lists, no markdown.

JSON facts:
{facts_json}

Now write the cohort narrative.
"""


def build_per_patient_rows(per_sample: list[dict], f1: float | None) -> list[dict]:
    rows = []
    for s in per_sample:
        prob = s.get("predicted_positive_prob", 0.5)
        predicted_class = "positive" if prob >= 0.5 else "negative"
        in_band = CONFIDENT_LOW <= prob <= CONFIDENT_HIGH
        band_label = "near-boundary (low confidence)" if in_band else "outside typical range"
        if in_band:
            message = ABSTENTION_TEXT
        else:
            message = OUTSIDE_BAND_TEXT.format(prob=prob, low=CONFIDENT_LOW, high=CONFIDENT_HIGH,
                                                f1=(f1 if f1 is not None else float("nan")))
        rows.append({
            "record_id": s.get("record_id"),
            "predicted_class": predicted_class,
            "predicted_probability": prob,
            "confidence_band": band_label,
            "message": message,
        })
    return rows


def render_per_patient_markdown(rows: list[dict]) -> str:
    lines = []
    lines.append("## Per-patient table\n")
    lines.append(PER_PATIENT_HEADER_NOTE + "\n")
    lines.append("| Record | Predicted class | Predicted probability | Confidence band | Note |")
    lines.append("|---|---|---|---|---|")
    for r in rows:
        lines.append(
            f"| {r['record_id']} | {r['predicted_class']} | {r['predicted_probability']:.4f} | "
            f"{r['confidence_band']} | {r['message']} |"
        )
    lines.append("")
    return "\n".join(lines)


def main() -> int:
    ap = argparse.ArgumentParser(description="Phase B — Clinical Narrative Agent")
    ap.add_argument("--xai-report", default=None, help="path to a Phase A xai_ig_*.json; default: most recent")
    ap.add_argument("--round-id", type=int, default=None, help="default: looked up via MongoDB session_id")
    ap.add_argument("--model", default="phi3:mini")
    ap.add_argument("--out", default=str(DEFAULT_OUT_DIR))
    ap.add_argument("--explain-dir", default=str(DEFAULT_EXPLAIN_DIR))
    ap.add_argument("--mongo-uri", default=DEFAULT_MONGO_URI)
    ap.add_argument("--db", default=DEFAULT_DB)
    ap.add_argument("--ollama-url", default=DEFAULT_OLLAMA_URL)
    ap.add_argument("--ollama-timeout", type=int, default=240)
    args = ap.parse_args()

    t_start = time.time()

    xai_report_path = Path(args.xai_report) if args.xai_report else _find_latest_xai_report(Path(args.explain_dir))
    print(f"[clinical-agent] reading Phase A report: {xai_report_path}")
    xai_report = json.loads(xai_report_path.read_text())
    per_sample = xai_report.get("per_sample", [])

    round_id = args.round_id
    if round_id is None:
        round_id = _lookup_round_id(xai_report.get("session_id", ""), args.mongo_uri, args.db)
    round_label = str(round_id) if round_id is not None else f"unknown_{xai_report.get('session_id', 'session')}"
    if round_id is None:
        print(f"[clinical-agent] WARNING: could not resolve round_id from MongoDB "
              f"(session_id={xai_report.get('session_id')!r}); using {round_label!r} for the filename")

    cohort_facts = build_cohort_facts(xai_report, round_id)
    t_gather = time.time() - t_start
    print(f"[clinical-agent] cohort facts assembled in {t_gather:.2f}s "
          f"({cohort_facts['num_patients_in_eval_set']} patients in eval set)")

    cohort_facts_md = render_cohort_facts_markdown(cohort_facts)
    prompt = build_cohort_prompt(cohort_facts)
    narrative, llm_error, t_llm = call_ollama(prompt, args.model, args.ollama_url, args.ollama_timeout)

    consistency = None
    if narrative:
        consistency = check_consistency(narrative, cohort_facts)
        status = "PASSED" if consistency["passed"] else "WARNING - untraceable numbers found"
        print(f"[clinical-agent] factual-consistency check: {status} "
              f"({consistency['num_checked_as_claims']} numeric claims checked "
              f"against {consistency['num_fact_numbers_available']} known facts)")
        if not consistency["passed"]:
            print(f"[clinical-agent]   untraceable: {consistency['untraceable_numbers']}")
    else:
        print(f"[clinical-agent] cohort narrative unavailable: {llm_error}")

    em = cohort_facts.get("eval_metrics", {})
    f1 = em.get("f1") if isinstance(em.get("f1"), (int, float)) else None
    per_patient_rows = build_per_patient_rows(per_sample, f1)
    per_patient_md = render_per_patient_markdown(per_patient_rows)

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"round_{round_label}_clinical_report.md"

    n_eval = cohort_facts["num_patients_in_eval_set"]
    n_uncertain = cohort_facts["prediction_confidence"]["num_uncertain_0.3_to_0.7"]
    doc = []
    doc.append(f"# Clinical Screening Report — Round {round_label}\n")
    doc.append(f"Generated {datetime.now(timezone.utc).isoformat()} by `clinical_narrative_agent.py` "
               f"(local Ollama model: `{args.model}`, source: `{xai_report_path.name}`)\n")
    n_train = cohort_facts.get("num_train_records")
    doc.append("> " + DOCUMENT_DISCLAIMER.format(
        f1=(f1 if f1 is not None else float("nan")),
        n_train=(n_train if n_train is not None else "an undetermined number of"),
    ) + "\n")
    if n_eval > 0:
        doc.append("> " + UNCERTAINTY_STATEMENT.format(
            n_eval=n_eval, low=CONFIDENT_LOW, high=CONFIDENT_HIGH,
            n_uncertain=n_uncertain, pct=100.0 * n_uncertain / n_eval,
        ) + "\n")

    doc.append(cohort_facts_md)

    doc.append("## Factual-consistency check (cohort narrative)\n")
    if consistency is not None:
        doc.append(f"**Result: {'PASSED' if consistency['passed'] else 'WARNING — narrative contains numbers not traceable to the facts above'}**\n")
        doc.append(
            f"- Numeric claims extracted: {consistency['num_checked_as_claims']} "
            f"(of {consistency['num_candidates_extracted']} numbers found)\n"
            f"- Fact numbers available to check against: {consistency['num_fact_numbers_available']}\n"
        )
        if not consistency["passed"]:
            doc.append(f"- **Untraceable numbers:** {consistency['untraceable_numbers']}\n")
    else:
        doc.append("Not run — no narrative was generated (see below).\n")

    doc.append("## Cohort narrative (LLM-generated, local Ollama)\n")
    if narrative:
        doc.append(narrative + "\n")
    else:
        doc.append(f"*Cohort narrative unavailable — {llm_error}*\n"
                    f"\nThe facts table and per-patient table above/below are unaffected; "
                    f"they do not depend on the LLM.\n")

    doc.append(per_patient_md)

    t_total = time.time() - t_start
    doc.append("---\n")
    doc.append(
        f"*Runtime: {t_total:.2f}s total — facts gathering {t_gather:.2f}s, "
        f"LLM generation {t_llm:.2f}s, per-patient table generation is template-only "
        f"(no LLM call, negligible time). No external network calls: MongoDB at "
        f"{args.mongo_uri}, Ollama at {args.ollama_url} — both localhost only.*\n"
    )

    out_path.write_text("\n".join(doc), encoding="utf-8")

    print(f"[clinical-agent] report written to {out_path}")
    print(f"[clinical-agent] total runtime {t_total:.2f}s (gather {t_gather:.2f}s, llm {t_llm:.2f}s)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
