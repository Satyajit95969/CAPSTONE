#!/usr/bin/env python3
"""
scripts/demo_rag_comparison.py

STEP C5 — the strongest RAG demonstration: the SAME cohort facts, from the
SAME Phase A report, narrated TWICE by the SAME local Ollama model - once
WITHOUT retrieval (ungrounded, the model's own unaided framing) and once
WITH retrieval (grounded, cited to the corpus) - printed side by side.

Two live Ollama calls, ~150s total. Progress is printed throughout so a live
demo does not look hung. The ungrounded narrative is expensive to regenerate
and doesn't change run to run in any way that matters for this comparison,
so it is cached to disk after the first run and reused - pass
--force-regenerate to bypass the cache (e.g. after a prompt-wording change).

This script reuses clinical_narrative_agent.py's real functions
(build_cohort_facts, build_cohort_prompt, retrieve_context_detailed) and
ollama_narration_utils.py's real call_ollama/check_consistency - it does not
reimplement or approximate either code path.

Usage:
    .venv\\Scripts\\python.exe scripts\\demo_rag_comparison.py
    .venv\\Scripts\\python.exe scripts\\demo_rag_comparison.py --force-regenerate
"""

from __future__ import annotations

import argparse
import json
import re
import sys
import time
from pathlib import Path

REPO_ROOT = Path(r"D:\Download D\BE PIPELINE\Capstone-")
sys.path.insert(0, str(REPO_ROOT / "scripts"))

from ollama_narration_utils import call_ollama, check_consistency  # noqa: E402
from clinical_narrative_agent import (  # noqa: E402
    build_cohort_facts, build_cohort_prompt, retrieve_context_detailed,
    print_retrieval_console_summary, _find_latest_xai_report, _lookup_round_id,
    TOPIC_QUERIES, DEFAULT_EXPLAIN_DIR, DEFAULT_MONGO_URI, DEFAULT_DB,
)

UNGROUNDED_CACHE = Path.home() / ".federated" / "data" / "demo_cache" / "rag_comparison_ungrounded.json"


def print_header(title: str):
    print()
    print("=" * 100)
    print(title)
    print("=" * 100)


def wrap_print(text: str, width: int = 96, prefix: str = "  "):
    import textwrap
    for para in text.split("\n\n"):
        for line in textwrap.wrap(para, width=width):
            print(prefix + line)
        print()


def count_citations(narrative: str, retrieved_docs: list[dict]) -> int:
    """How many of the retrieved documents' titles appear (verbatim or via
    their id) anywhere in the narrative - a rough but honest proxy for
    'did the model actually cite what it was given', used only for this
    demo's summary line, not as a correctness guarantee."""
    count = 0
    for d in retrieved_docs:
        if d["title"].lower() in narrative.lower() or d["id"].lower() in narrative.lower():
            count += 1
    return count


def mentions_unverified(narrative: str) -> bool:
    return bool(re.search(r"unverif", narrative, re.IGNORECASE))


def main() -> int:
    ap = argparse.ArgumentParser(description="Phase C RAG comparison demo — ungrounded vs grounded narrative")
    ap.add_argument("--xai-report", default=None)
    ap.add_argument("--model", default="phi3:mini")
    ap.add_argument("--ollama-url", default="http://localhost:11434")
    ap.add_argument("--ollama-timeout", type=int, default=240)
    ap.add_argument("--explain-dir", default=str(DEFAULT_EXPLAIN_DIR))
    ap.add_argument("--mongo-uri", default=DEFAULT_MONGO_URI)
    ap.add_argument("--db", default=DEFAULT_DB)
    ap.add_argument("--force-regenerate", action="store_true",
                     help="ignore the cached ungrounded narrative and regenerate it")
    args = ap.parse_args()

    t_start = time.time()
    print("STEP C5 — RAG comparison demo: ungrounded vs grounded cohort narrative")
    print("Two live Ollama calls, ~150s total. This is not a mockup - both narratives are")
    print("generated fresh by phi3:mini (the ungrounded one may be served from cache; see below).\n")

    xai_report_path = Path(args.xai_report) if args.xai_report else _find_latest_xai_report(Path(args.explain_dir))
    print(f"[demo] Phase A report: {xai_report_path}")
    xai_report = json.loads(xai_report_path.read_text())
    round_id = _lookup_round_id(xai_report.get("session_id", ""), args.mongo_uri, args.db)
    cohort_facts = build_cohort_facts(xai_report, round_id)
    print(f"[demo] cohort facts assembled: round={round_id} session={cohort_facts['session_id']} "
          f"eval_set={cohort_facts['num_patients_in_eval_set']}\n")

    # ── UNGROUNDED (no retrieved_context) — cached ──────────────────────────
    print_header("GENERATING NARRATIVE 1/2 — UNGROUNDED (no retrieval, model's own framing)")
    ungrounded_facts = dict(cohort_facts)
    ungrounded_facts["retrieved_context"] = []
    ungrounded_prompt = build_cohort_prompt(ungrounded_facts)

    ungrounded_narrative = None
    from_cache = False
    if UNGROUNDED_CACHE.exists() and not args.force_regenerate:
        try:
            cached = json.loads(UNGROUNDED_CACHE.read_text())
            if cached.get("session_id") == cohort_facts["session_id"] and cached.get("model") == args.model:
                ungrounded_narrative = cached["narrative"]
                from_cache = True
                print(f"[demo] using cached ungrounded narrative (from {cached.get('generated_at', '?')}) "
                      f"— pass --force-regenerate to bypass")
        except Exception:
            pass

    if ungrounded_narrative is None:
        print("[demo] calling Ollama (no cache hit)...")
        t0 = time.time()
        ungrounded_narrative, err, elapsed = call_ollama(ungrounded_prompt, args.model, args.ollama_url, args.ollama_timeout)
        print(f"[demo] done in {elapsed:.1f}s" + (f" (error: {err})" if err else ""))
        if ungrounded_narrative:
            UNGROUNDED_CACHE.parent.mkdir(parents=True, exist_ok=True)
            UNGROUNDED_CACHE.write_text(json.dumps({
                "session_id": cohort_facts["session_id"], "model": args.model,
                "narrative": ungrounded_narrative,
                "generated_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            }, indent=2))

    # ── GROUNDED (with retrieval) — always live ─────────────────────────────
    print_header("GENERATING NARRATIVE 2/2 — GROUNDED (with Phase C retrieval)")
    import torch
    device = "cuda" if torch.cuda.is_available() else "cpu"
    retrieved_docs, per_query, cached_embeddings = retrieve_context_detailed(TOPIC_QUERIES, device)
    print_retrieval_console_summary(per_query, retrieved_docs, cached_embeddings)

    grounded_facts = dict(cohort_facts)
    grounded_facts["retrieved_context"] = [
        {"id": d["id"], "title": d["title"], "text": d["text"], "source": d["source"], "verified": d["verified"]}
        for d in retrieved_docs
    ]
    grounded_prompt = build_cohort_prompt(grounded_facts)
    print("\n[demo] calling Ollama (grounded, not cached — always live)...")
    t0 = time.time()
    grounded_narrative, err, elapsed = call_ollama(grounded_prompt, args.model, args.ollama_url, args.ollama_timeout)
    print(f"[demo] done in {elapsed:.1f}s" + (f" (error: {err})" if err else ""))

    # ── side by side ─────────────────────────────────────────────────────
    print_header("SIDE BY SIDE")
    print(f"[Narrative 1 — UNGROUNDED{' (cached)' if from_cache else ''}]\n")
    if ungrounded_narrative:
        wrap_print(ungrounded_narrative)
    else:
        print("  (unavailable)\n")

    print("-" * 100)
    print("\n[Narrative 2 — GROUNDED, Phase C retrieval]\n")
    if grounded_narrative:
        wrap_print(grounded_narrative)
    else:
        print("  (unavailable)\n")

    # ── summary ──────────────────────────────────────────────────────────
    print_header("SUMMARY OF THE DIFFERENCE")
    n_unverified = sum(1 for d in retrieved_docs if not d["verified"])
    if ungrounded_narrative:
        ug_cites = count_citations(ungrounded_narrative, retrieved_docs)
        ug_flags_unverified = mentions_unverified(ungrounded_narrative)
        print(f"Ungrounded narrative: {ug_cites} corpus source(s) mentioned by title/id "
              f"(it was never given any — {ug_cites} would mean incidental overlap, not real citation).")
        print(f"                      mentions \"unverified\" anywhere: {ug_flags_unverified}")
    if grounded_narrative:
        g_cites = count_citations(grounded_narrative, retrieved_docs)
        g_flags_unverified = mentions_unverified(grounded_narrative)
        print(f"Grounded narrative:   {g_cites} of {len(retrieved_docs)} retrieved source(s) mentioned by title/id.")
        print(f"                      mentions \"unverified\" anywhere in its own text: {g_flags_unverified}")
        print(f"                      {n_unverified} of the retrieved sources are actually unverified "
              f"(PHQ-8 content) — the deterministic warning that always precedes the narrative in a real "
              f"report (Step C4) covers this regardless of whether the LLM itself said so.")
    if ungrounded_narrative and grounded_narrative:
        consistency = check_consistency(grounded_narrative, grounded_facts)
        print(f"\nGrounded narrative's factual-consistency check: "
              f"{'PASSED' if consistency['passed'] else 'WARNING'} "
              f"({consistency['num_checked_as_claims']} claims checked, "
              f"{len(consistency['untraceable_numbers'])} untraceable).")

    elapsed_total = time.time() - t_start
    print_header(f"DONE in {elapsed_total:.1f}s")
    return 0


if __name__ == "__main__":
    sys.exit(main())
