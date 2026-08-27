#!/usr/bin/env python3
"""
scripts/demo_rag.py

STEP C5 — self-contained Phase C (clinical knowledge retrieval) demonstration.

Runs in under 60 seconds, fully offline: no orchestrator, no gRPC, no
MongoDB, no TPM, no live client round. Safe to run live in a meeting on any
machine with this repo, its .venv, and the MentalBERT checkpoint already on
local disk (~/.federated/models/mentalbert).

Imports the real corpus (clinical_knowledge_corpus.CORPUS) and the real
retrieval function (clinical_narrative_agent.retrieve_context_detailed) -
this demo does not maintain its own copy of either. What you see here is
exactly what a live clinical_narrative_agent.py run retrieves.

Usage:
    .venv\\Scripts\\python.exe scripts\\demo_rag.py
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

REPO_ROOT = Path(r"D:\Download D\BE PIPELINE\Capstone-")
sys.path.insert(0, str(REPO_ROOT / "scripts"))

from clinical_knowledge_corpus import CORPUS, corpus_hash  # noqa: E402
from clinical_narrative_agent import (  # noqa: E402
    retrieve_context_detailed, TOPIC_QUERIES, RETRIEVAL_TOP_K,
)


def print_header(title: str):
    print()
    print("=" * 100)
    print(title)
    print("=" * 100)


def part1_the_corpus():
    print_header("PART 1 — THE CORPUS (scripts/clinical_knowledge_corpus.py)")
    print(f"{len(CORPUS)} documents. hash={corpus_hash()[:16]}... "
          "(changes if the corpus content changes - invalidates the embedding cache).\n")
    print("This corpus contains ZERO patient data - every document below is either a")
    print("published clinical instrument's own wording, or this project's own findings")
    print("about its model, never a patient record, transcript, or prediction.\n")

    header = f"{'id':<32} {'tier':>4}  {'verified':>8}  {'title'}"
    print(header)
    print("-" * 100)
    for doc in CORPUS:
        v = "yes" if doc["verified"] else "NO"
        print(f"{doc['id']:<32} {doc['tier']:>4}  {v:>8}  {doc['title']}")
    print()
    tier1 = [d for d in CORPUS if d["tier"] == 1]
    tier2 = [d for d in CORPUS if d["tier"] == 2]
    print(f"Tier 1 (PHQ-8 instrument text, cited to Kroenke et al. 2009): {len(tier1)} documents, "
          f"all verified=False (see Part 4).")
    print(f"Tier 2 (this project's own findings, cited to docs/IMPLEMENTATION_NOTES.md): "
          f"{len(tier2)} documents, all verified=True.")
    print(f"\nSource citation (Tier 1): {tier1[0]['source']}")


def part2_retrieval_in_action():
    print_header("PART 2 — RETRIEVAL IN ACTION (the 3 fixed topic queries, live)")
    print("Same MentalBERT embeddings + sklearn cosine NearestNeighbors, same corpus, same")
    print(f"top_k={RETRIEVAL_TOP_K} as a real clinical_narrative_agent.py run - not a mockup.\n")

    import torch
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"[info] embedding queries + corpus on device={device} ...")
    t0 = time.time()
    retrieved_docs, per_query, cached = retrieve_context_detailed(TOPIC_QUERIES, device)
    elapsed = time.time() - t0
    print(f"[info] retrieval complete in {elapsed:.2f}s\n")

    for pq in per_query:
        print(f"QUERY: \"{pq['query']}\"")
        for rank, m in enumerate(pq["matches"]):
            v = "verified" if m["verified"] else "UNVERIFIED"
            print(f"  #{rank+1}  sim={m['similarity']:.4f}  tier={m['tier']}  [{v:10s}]  {m['title']}")
        print()

    all_ids = set()
    for pq in per_query:
        all_ids.update(m["id"] for m in pq["matches"])
    print(f"{len(retrieved_docs)} unique document(s) retrieved across {len(TOPIC_QUERIES)} queries.")
    overlap = len(per_query[0]["matches"]) + len(per_query[1]["matches"]) + len(per_query[2]["matches"]) - len(all_ids)
    print(f"Different queries retrieve different (though sometimes overlapping) documents: "
          f"{len(all_ids)} unique out of {sum(len(pq['matches']) for pq in per_query)} total slots "
          f"({overlap} shared matches, mostly the two project-findings documents that are relevant "
          "to more than one query).")


def part3_what_and_why():
    print_header("PART 3 — WHAT THIS RAG LAYER DOES, AND WHY IT IS BUILT THIS WAY")
    print("""\
The project document's original RAG design retrieved from PATIENT embeddings in
the current training batch - one patient's data informing another patient's
prediction. That was never revived here. In a federated-learning setting where
raw patient data is supposed to never leave the client device, retrieving one
patient's embedding to help explain a DIFFERENT patient's prediction is itself
a privacy problem - and it was never on Phase B's live reporting path anyway
(create_dp_comparison.py's version is a research-harness function, disconnected
from clinical_narrative_agent.py).

This RAG layer retrieves ONLY from a small, fixed corpus of reference material -
no patient data of any kind ever enters the vector store. The purpose is
different too: not to inform a prediction, but to GROUND a plain-English report
about an already-made prediction, so its clinical framing traces to a cited
source instead of a 3B-parameter model's uncited, unverifiable parametric
memory. Every fact the LLM's narrative uses can be checked against a real
citation, the same principle the numeric consistency-checker already applies
to numbers (Phase D/B): don't trust the model's unaided output when the
alternative is checking it against something real.""")


def part4_deliberately_excluded():
    print_header("PART 4 — WHAT WAS DELIBERATELY LEFT OUT")
    print("""\
A third corpus tier was considered and rejected: feature-to-symptom mappings
(e.g. "reduced AU06/AU12 facial-action-unit activation <-> blunted affect", or
any audio-prosody-to-symptom claim). Unlike PHQ-8's fixed, standardized,
publicly-published wording, these are specific, contested, actively-researched
empirical claims that vary by study and population - not something an LLM
(or this project's author) should draft without a real, specific, peer-reviewed
citation.

Drafting placeholder feature-to-symptom content to "fill out" the corpus would
have stacked one unverified thing (an invented clinical mapping) on top of
another already-established one: Phase A's own investigation (Steps A3-A8,
docs/IMPLEMENTATION_NOTES.md) found this system's attribution mechanism
unreliable at the individual-patient level in the first place. Two unverified
things is worse than one, and worse than a visible gap.

This gap is left visible - the corpus has 17 documents and 0 feature-to-symptom
mappings - rather than filled with something plausible-sounding.""")


def main() -> int:
    t0 = time.time()
    print("STEP C5 — Phase C (clinical knowledge retrieval) self-contained demonstration")
    print("Fully offline: no orchestrator, no gRPC, no MongoDB, no TPM, no live client round.")

    part1_the_corpus()
    part2_retrieval_in_action()
    part3_what_and_why()
    part4_deliberately_excluded()

    elapsed = time.time() - t0
    print_header(f"DONE in {elapsed:.1f}s")
    return 0


if __name__ == "__main__":
    sys.exit(main())
