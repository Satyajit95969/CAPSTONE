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
import hashlib
import json
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

from pymongo import MongoClient

sys.path.insert(0, str(Path(__file__).resolve().parent))
from ollama_narration_utils import call_ollama, check_consistency  # noqa: E402
from clinical_knowledge_corpus import CORPUS, corpus_hash  # noqa: E402

DEFAULT_EXPLAIN_DIR = Path.home() / ".federated" / "data" / "explain_logs"
DEFAULT_OUT_DIR = Path.home() / ".federated" / "data" / "clinical_reports"
DEFAULT_OLLAMA_URL = "http://localhost:11434"
DEFAULT_MONGO_URI = "mongodb://localhost:27017"
DEFAULT_DB = "federated_multimodal"

# Phase C: local-only clinical knowledge retrieval. MentalBERT (already on
# disk, no new download - see Step C1 investigation) for embeddings; the
# path is redefined here rather than importing it from
# trainer_mentalbert_privacy.py, which would pull in a heavy, unrelated
# transitive dependency (the full training module) just for one constant.
MENTALBERT_PRETRAIN = str(Path.home() / ".federated" / "models" / "mentalbert")
CORPUS_CACHE_DIR = Path.home() / ".federated" / "data" / "clinical_knowledge_cache"
CORPUS_EMBEDDINGS_CACHE = CORPUS_CACHE_DIR / "corpus_embeddings.pt"

# Fixed topic queries (Step C2 point 3) - NOT dynamically generated from the
# cohort facts or the LLM's own output, so retrieval stays deterministic and
# auditable: the same three queries run every time, tied to the report's
# three known content areas (what PHQ-8 is, the modality-attribution method
# caveat, the confidence/uncertainty framing).
TOPIC_QUERIES = [
    "What does the PHQ-8 depression screening questionnaire measure, and what are its items and scoring?",
    "How reliable is per-patient modality attribution across text, audio, and video for this model?",
    "What does it mean when a model's prediction confidence is near the decision boundary?",
]
RETRIEVAL_TOP_K = 3  # Step C4: widened from 2 - a 17-document corpus costs
# nothing to search wider, and top_k=2 missed project-per-patient-unreliable
# for query 2 despite it being the most on-topic document for that query.

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

# Step C4 item 2: deterministic, NOT LLM-generated. The prompt (rule 10)
# asks phi3:mini to say "unverified" when it cites unverified retrieved
# content, and it did not reliably comply (Step C3's verification run cited
# phq8-overview, verified=False, without ever saying so). Rather than
# trusting a stronger prompt to fix that, this sentence is generated by
# Python from the retrieved docs' own verified flags and placed adjacent to
# the narrative regardless of what the LLM wrote - the same structural
# principle as the per-patient table and the "Sources used" section: a
# reader-facing safety statement should not depend on model compliance when
# it can instead be a fact the code itself asserts. Rule 10 stays in the
# prompt too - belt and braces, not a replacement.
UNVERIFIED_SOURCES_WARNING_TEMPLATE = (
    "**Note (machine-generated, not from the LLM):** the narrative below may "
    "cite the following source(s), which are UNVERIFIED — reproduced from "
    "memory of a standard instrument, not checked against a primary copy: "
    "{titles}. Treat any content attributed to these sources as likely-"
    "correct but unconfirmed, independent of whether the narrative itself "
    "says so.\n"
)


def build_unverified_sources_warning(retrieved_docs: list[dict]) -> str | None:
    """Returns the deterministic warning sentence if any retrieved doc is
    unverified, else None (nothing to prepend)."""
    unverified = [d for d in retrieved_docs if not d.get("verified", True)]
    if not unverified:
        return None
    titles = ", ".join(f'"{d["title"]}"' for d in unverified)
    return UNVERIFIED_SOURCES_WARNING_TEMPLATE.format(titles=titles)


def _embed_texts(texts: list[str], device: str) -> "list[list[float]]":
    """MentalBERT [CLS] embeddings for a list of short strings (Step C1
    choice: already on local disk, zero new download, zero new
    dependency - see docs/IMPLEMENTATION_NOTES.md Step C1 investigation)."""
    import torch
    from transformers import AutoModel, AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(MENTALBERT_PRETRAIN)
    model = AutoModel.from_pretrained(MENTALBERT_PRETRAIN).to(device)
    model.eval()
    out = []
    with torch.no_grad():
        for text in texts:
            inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=128).to(device)
            hidden = model(**inputs).last_hidden_state[:, 0, :]  # [CLS]
            out.append(hidden[0].cpu().tolist())
    return out


def compute_or_load_corpus_embeddings(device: str, cache_path: Path | None = None,
                                       force_recompute: bool = False) -> dict:
    """Step C2 point 2: caches the corpus's MentalBERT embeddings to disk,
    following Step A5's compute_modality_baselines() caching pattern.
    Invalidation: the cache stores clinical_knowledge_corpus.corpus_hash()
    at write time; if the CURRENT corpus's hash doesn't match, the cache is
    stale (the corpus changed since it was written) and is recomputed
    rather than silently served."""
    import torch

    cache_path = cache_path or CORPUS_EMBEDDINGS_CACHE
    current_hash = corpus_hash()
    if cache_path.exists() and not force_recompute:
        cached = torch.load(cache_path, map_location="cpu")
        if cached.get("corpus_hash") == current_hash:
            return cached
        print(f"[clinical-agent] corpus changed since embeddings were cached "
              f"({cached.get('corpus_hash', '?')[:8]}... -> {current_hash[:8]}...), recomputing")

    doc_ids = [doc["id"] for doc in CORPUS]
    texts = [f"{doc['title']}. {doc['text']}" for doc in CORPUS]
    embeddings = _embed_texts(texts, device)
    cached = {"corpus_hash": current_hash, "doc_ids": doc_ids, "embeddings": embeddings}
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(cached, cache_path)
    return cached


def retrieve_context_detailed(queries: list[str], device: str, top_k: int = RETRIEVAL_TOP_K):
    """The single retrieval code path (Step C5: retrieve_context() below is
    now a thin wrapper over this - visibility changes must not create a
    second way retrieval could disagree with itself). MentalBERT embeddings,
    sklearn NearestNeighbors cosine (Step C1's recommendation: a corpus this
    size needs no approximate-nearest-neighbor index).

    Returns (retrieved_docs, per_query, cached):
      - retrieved_docs: the union of top_k matches across all queries,
        deduplicated, each entry carrying its full corpus record plus which
        queries matched it and its best similarity score - same shape
        retrieve_context() has always returned.
      - per_query: one entry per query, each with that query's own top_k
        matches and THAT query's own similarity score (not deduped/maxed) -
        what Step C5's console output and scripts/demo_rag.py need to show
        "different queries retrieve different documents" honestly.
      - cached: the corpus-embeddings cache dict (corpus_hash, doc_ids,
        embeddings) - exposed so callers can print corpus size/hash without
        a second cache load.
    """
    import numpy as np
    from sklearn.neighbors import NearestNeighbors

    cached = compute_or_load_corpus_embeddings(device)
    doc_by_id = {doc["id"]: doc for doc in CORPUS}
    corpus_matrix = np.array(cached["embeddings"], dtype=np.float32)

    k = min(top_k, len(cached["doc_ids"]))
    nn = NearestNeighbors(n_neighbors=k, metric="cosine")
    nn.fit(corpus_matrix)

    query_vecs = np.array(_embed_texts(queries, device), dtype=np.float32)
    distances, indices = nn.kneighbors(query_vecs)

    per_query = []
    retrieved: dict[str, dict] = {}
    for qi, query in enumerate(queries):
        matches = []
        for rank, idx in enumerate(indices[qi]):
            doc_id = cached["doc_ids"][idx]
            doc = doc_by_id[doc_id]
            similarity = 1.0 - float(distances[qi][rank])
            matches.append({"id": doc_id, "title": doc["title"], "tier": doc["tier"],
                             "verified": doc["verified"], "similarity": similarity})
            entry = retrieved.setdefault(doc_id, {**doc, "matched_queries": [], "similarity": similarity})
            entry["matched_queries"].append(query)
            entry["similarity"] = max(entry["similarity"], similarity)
        per_query.append({"query": query, "matches": matches})

    return list(retrieved.values()), per_query, cached


def retrieve_context(queries: list[str], device: str, top_k: int = RETRIEVAL_TOP_K) -> list[dict]:
    """Thin wrapper preserving the pre-Step-C5 return contract (used by
    main() below and by any external caller) - see
    retrieve_context_detailed() for the actual retrieval logic and the
    richer per-query data."""
    retrieved_docs, _per_query, _cached = retrieve_context_detailed(queries, device, top_k)
    return retrieved_docs


def print_retrieval_console_summary(per_query: list[dict], retrieved_docs: list[dict], cached: dict) -> None:
    """Step C5 point 1: compact, ~20-line console visibility into what
    retrieval actually did, printed right after retrieve_context_detailed()
    returns - corpus size/hash, each fixed query, its top-k matches with
    similarity/tier/verified, and the total unique-document count."""
    print(f"[clinical-agent] --- Phase C retrieval detail ---")
    print(f"[clinical-agent] corpus: {len(CORPUS)} documents, hash={cached['corpus_hash'][:12]}...")
    for pq in per_query:
        q = pq["query"]
        q_short = q if len(q) <= 78 else q[:75] + "..."
        print(f"[clinical-agent] query: \"{q_short}\"")
        for m in pq["matches"]:
            v = "verified" if m["verified"] else "UNVERIFIED"
            print(f"[clinical-agent]     sim={m['similarity']:.4f}  tier={m['tier']}  [{v:10s}]  {m['title']}")
    print(f"[clinical-agent] {len(retrieved_docs)} unique document(s) retrieved across {len(per_query)} queries")


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
8. The JSON below also contains a "retrieved_context" list. Each entry has a
   "title", "text", "source" citation, and a "verified" flag. If you use ANY
   information from retrieved_context in your narrative, you MUST cite it by
   its title in parentheses, e.g. "(source: PHQ-8 scoring range)". Do not
   present retrieved_context content as your own knowledge.
9. Do NOT make any clinical claim, definition, or interpretation that is not
   directly supported by retrieved_context or the facts above it. If
   retrieved_context does not cover something, do not fill the gap from your
   own training - only state what is grounded in what you were given.
10. If a retrieved_context entry has "verified": false, you MUST say so when
    citing it (e.g., "unverified PHQ-8 wording, source: ...") - never present
    it as confirmed fact.

JSON facts:
{facts_json}

Now write the cohort narrative.
"""


def render_sources_used_markdown(retrieved_docs: list[dict]) -> str:
    """Step C2 point 6: every retrieved document, its citation, and its
    verified flag, so a reader can check what the cohort narrative was
    grounded in without having to open the corpus file."""
    lines = []
    lines.append("## Sources used\n")
    if not retrieved_docs:
        lines.append("*No sources retrieved (retrieval unavailable or corpus empty for this run - "
                      "the cohort narrative above was not grounded in any retrieved reference text).*\n")
        return "\n".join(lines)
    lines.append(
        "Retrieved from `scripts/clinical_knowledge_corpus.py` against this report's fixed topic "
        "queries. The cohort narrative above was instructed to cite any of these it used, and to say "
        "so explicitly if a source is unverified.\n"
    )
    lines.append("| Title | Verified | Source | Tier |")
    lines.append("|---|---|---|---|")
    for doc in sorted(retrieved_docs, key=lambda d: (-d.get("similarity", 0), d["id"])):
        verified_label = "yes" if doc.get("verified") else "**NO - unverified**"
        lines.append(f"| {doc['title']} | {verified_label} | {doc['source']} | {doc['tier']} |")
    lines.append("")
    for doc in sorted(retrieved_docs, key=lambda d: (-d.get("similarity", 0), d["id"])):
        lines.append(f"**{doc['title']}** (`{doc['id']}`, verified={doc.get('verified')}): {doc['text']}")
        lines.append(f"  *{doc.get('verification_note', '')}*\n")
    return "\n".join(lines)


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
    ap.add_argument("--no-retrieval", action="store_true",
                     help="skip clinical-knowledge retrieval (Phase C) - narrative generated ungrounded")
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

    # ── Phase C: clinical knowledge retrieval, between build_cohort_facts()
    # and the prompt build (Step C2 point 4). retrieve_context() only ever
    # queries clinical_knowledge_corpus.CORPUS, which contains zero patient
    # data - so attaching its output to cohort_facts does not touch the
    # structural no-per-patient-data guarantee (facts still never contains
    # per_sample). Wrapped in try/except: retrieval failing must not break
    # narrative generation or the deterministic tables (Step B2 point 4's
    # principle, extended to this new step).
    t_retrieval0 = time.time()
    retrieved_docs = []
    if not args.no_retrieval:
        try:
            import torch
            device = "cuda" if torch.cuda.is_available() else "cpu"
            retrieved_docs, per_query, cached = retrieve_context_detailed(TOPIC_QUERIES, device)
            print_retrieval_console_summary(per_query, retrieved_docs, cached)
        except Exception as e:
            print(f"[clinical-agent] WARNING: clinical-knowledge retrieval failed, "
                  f"continuing without grounding: {e}")
    t_retrieval = time.time() - t_retrieval0
    cohort_facts["retrieved_context"] = [
        {"id": d["id"], "title": d["title"], "text": d["text"], "source": d["source"], "verified": d["verified"]}
        for d in retrieved_docs
    ]

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
            f"- Excluded as generic small integers (bare, no decimal point, under 100 - not a claim): "
            f"{consistency['num_excluded_as_small_integer']}\n"
            f"- Traceable as a percentage restatement (e.g. 0.5676 written as \"56.76%\"): "
            f"{consistency['num_traceable_as_percentage']}\n"
            f"- Excluded as identifier substrings (digits inside a session/device ID or hash, not a claim): "
            f"{consistency['num_excluded_as_identifier_substring']}\n"
            f"- Excluded as timestamp fragments (clock-time or ISO-8601 datetime components, not a claim): "
            f"{consistency['num_excluded_as_timestamp']}\n"
            f"- Excluded as written-date fragments (e.g. \"August 26, 2026\", not a claim): "
            f"{consistency['num_excluded_as_written_date']}\n"
        )
        if not consistency["passed"]:
            doc.append(f"- **Untraceable numbers:** {consistency['untraceable_numbers']}\n")
    else:
        doc.append("Not run — no narrative was generated (see below).\n")

    doc.append("## Cohort narrative (LLM-generated, local Ollama)\n")
    unverified_warning = build_unverified_sources_warning(retrieved_docs)
    if unverified_warning:
        doc.append(unverified_warning)
    if narrative:
        doc.append(narrative + "\n")
    else:
        doc.append(f"*Cohort narrative unavailable — {llm_error}*\n"
                    f"\nThe facts table and per-patient table above/below are unaffected; "
                    f"they do not depend on the LLM.\n")

    doc.append(render_sources_used_markdown(retrieved_docs))

    doc.append(per_patient_md)

    t_total = time.time() - t_start
    doc.append("---\n")
    doc.append(
        f"*Runtime: {t_total:.2f}s total — facts gathering {t_gather:.2f}s, "
        f"clinical-knowledge retrieval {t_retrieval:.2f}s, LLM generation {t_llm:.2f}s, "
        f"per-patient table generation is template-only (no LLM call, negligible time). "
        f"No external network calls: MongoDB at {args.mongo_uri}, Ollama at {args.ollama_url}, "
        f"MentalBERT embeddings run locally — all localhost/local disk only.*\n"
    )

    out_path.write_text("\n".join(doc), encoding="utf-8")

    print(f"[clinical-agent] report written to {out_path}")
    print(f"[clinical-agent] total runtime {t_total:.2f}s "
          f"(gather {t_gather:.2f}s, retrieval {t_retrieval:.2f}s, llm {t_llm:.2f}s)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
