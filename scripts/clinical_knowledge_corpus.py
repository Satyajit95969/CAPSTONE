#!/usr/bin/env python3
"""
scripts/clinical_knowledge_corpus.py

Phase C — the clinical knowledge corpus, as structured data, not free text.
Contains NO patient data of any kind - only reference material Phase B's
cohort narrative can retrieve and cite. Small and inspectable by design: a
human should be able to read the whole thing in a few minutes (see
CORPUS below - it's the entire module).

Two tiers only, per the approved Step C1/C2 scope decision:

  TIER 1 - PHQ-8 item wording, response scale, and scoring range, cited to
  Kroenke et al. 2009. Every Tier 1 document carries verified=False and an
  explicit verification_note: this text was reproduced from memory of a
  standard, widely-published instrument, NOT copied from or checked against
  a primary source document. That flag is designed to survive into the
  generated report - a clinician reading a PHQ-8 quotation in Phase B's
  output should be able to see it is unverified. If someone later checks
  this text against a primary copy of the instrument, the flag should be
  cleared deliberately (edit this file, change verified to True and update
  verification_note to say what was checked against), never silently.

  TIER 2 - this project's own findings, drawn from
  docs/IMPLEMENTATION_NOTES.md's "Phase A: what the attribution mechanism
  can and cannot support" section. This is the project citing itself, so
  verified=True - a reader can check every one of these documents against
  that file directly.

NOT included, deliberately: feature-to-symptom mappings (e.g. facial-
action-unit-to-affect claims, prosody-to-symptom claims). These are
specific, contested, actively-researched empirical claims, not fixed public
facts like PHQ-8's wording - drafting them without a real, specific,
peer-reviewed citation would be introducing exactly the kind of
unverifiable clinical claim this whole phase exists to avoid. This gap is
left visible rather than filled with something plausible; do not add
placeholder feature-to-symptom documents here.
"""

from __future__ import annotations

import hashlib
import json

PHQ8_CITATION = (
    "Kroenke K, Strine TW, Spitzer RL, Williams JB, Berry JT, Mokdad AH. "
    "The PHQ-8 as a measure of current depression in the general population. "
    "J Affect Disord. 2009;114(1-3):163-173."
)

_UNVERIFIED_NOTE = (
    "Reproduced from memory of the standard, widely-published PHQ-8 "
    "instrument. NOT verified against a primary copy of the instrument or "
    "the cited paper. Treat this text as likely-correct but unconfirmed."
)

_PROJECT_NOTE = (
    "This project's own investigation, documented in "
    "docs/IMPLEMENTATION_NOTES.md - verifiable by reading that file "
    "directly, not an external or unverifiable claim."
)

CORPUS: list[dict] = [
    # ---- TIER 1: PHQ-8 (Kroenke et al. 2009), all verified=False ----
    {
        "id": "phq8-overview",
        "tier": 1,
        "title": "What the PHQ-8 measures",
        "text": (
            "The PHQ-8 (Patient Health Questionnaire-8) is an 8-item "
            "self-report screening instrument for depression severity, "
            "derived from the PHQ-9 by removing the ninth item (thoughts of "
            "self-harm). Respondents rate how often each of 8 symptoms "
            "bothered them over the prior 2 weeks. It is a SCREENING tool, "
            "not a diagnostic instrument - a high score indicates symptoms "
            "consistent with depression severity, not a clinical diagnosis, "
            "which requires clinical evaluation."
        ),
        "source": PHQ8_CITATION,
        "verified": False,
        "verification_note": _UNVERIFIED_NOTE,
        "tags": ["phq8", "overview", "screening"],
    },
    {
        "id": "phq8-item-1",
        "tier": 1,
        "title": "PHQ-8 item 1",
        "text": "Little interest or pleasure in doing things.",
        "source": PHQ8_CITATION,
        "verified": False,
        "verification_note": _UNVERIFIED_NOTE,
        "tags": ["phq8", "item_wording"],
    },
    {
        "id": "phq8-item-2",
        "tier": 1,
        "title": "PHQ-8 item 2",
        "text": "Feeling down, depressed, or hopeless.",
        "source": PHQ8_CITATION,
        "verified": False,
        "verification_note": _UNVERIFIED_NOTE,
        "tags": ["phq8", "item_wording"],
    },
    {
        "id": "phq8-item-3",
        "tier": 1,
        "title": "PHQ-8 item 3",
        "text": "Trouble falling or staying asleep, or sleeping too much.",
        "source": PHQ8_CITATION,
        "verified": False,
        "verification_note": _UNVERIFIED_NOTE,
        "tags": ["phq8", "item_wording"],
    },
    {
        "id": "phq8-item-4",
        "tier": 1,
        "title": "PHQ-8 item 4",
        "text": "Feeling tired or having little energy.",
        "source": PHQ8_CITATION,
        "verified": False,
        "verification_note": _UNVERIFIED_NOTE,
        "tags": ["phq8", "item_wording"],
    },
    {
        "id": "phq8-item-5",
        "tier": 1,
        "title": "PHQ-8 item 5",
        "text": "Poor appetite or overeating.",
        "source": PHQ8_CITATION,
        "verified": False,
        "verification_note": _UNVERIFIED_NOTE,
        "tags": ["phq8", "item_wording"],
    },
    {
        "id": "phq8-item-6",
        "tier": 1,
        "title": "PHQ-8 item 6",
        "text": (
            "Feeling bad about yourself - or that you are a failure, or "
            "have let yourself or your family down."
        ),
        "source": PHQ8_CITATION,
        "verified": False,
        "verification_note": _UNVERIFIED_NOTE,
        "tags": ["phq8", "item_wording"],
    },
    {
        "id": "phq8-item-7",
        "tier": 1,
        "title": "PHQ-8 item 7",
        "text": (
            "Trouble concentrating on things, such as reading the newspaper "
            "or watching television."
        ),
        "source": PHQ8_CITATION,
        "verified": False,
        "verification_note": _UNVERIFIED_NOTE,
        "tags": ["phq8", "item_wording"],
    },
    {
        "id": "phq8-item-8",
        "tier": 1,
        "title": "PHQ-8 item 8",
        "text": (
            "Moving or speaking so slowly that other people could have "
            "noticed - or the opposite, being so fidgety or restless that "
            "you have been moving around a lot more than usual."
        ),
        "source": PHQ8_CITATION,
        "verified": False,
        "verification_note": _UNVERIFIED_NOTE,
        "tags": ["phq8", "item_wording"],
    },
    {
        "id": "phq8-response-scale",
        "tier": 1,
        "title": "PHQ-8 response scale",
        "text": (
            "Each of the 8 items is rated on how often it occurred over the "
            "prior 2 weeks, using a 4-point scale: 0 = Not at all, "
            "1 = Several days, 2 = More than half the days, 3 = Nearly "
            "every day."
        ),
        "source": PHQ8_CITATION,
        "verified": False,
        "verification_note": _UNVERIFIED_NOTE,
        "tags": ["phq8", "scoring", "response_scale"],
    },
    {
        "id": "phq8-scoring-range",
        "tier": 1,
        "title": "PHQ-8 scoring range and severity bands",
        "text": (
            "Total PHQ-8 score is the sum of the 8 item scores, ranging "
            "0-24. Commonly reported severity bands: 0-4 none/minimal, "
            "5-9 mild, 10-14 moderate, 15-19 moderately severe, 20-24 "
            "severe. A cutoff of 10 or higher is commonly used in the "
            "literature as the threshold for a positive depression screen."
        ),
        "source": PHQ8_CITATION,
        "verified": False,
        "verification_note": _UNVERIFIED_NOTE,
        "tags": ["phq8", "scoring", "severity", "threshold"],
    },

    # ---- TIER 2: this project's own findings, all verified=True ----
    {
        "id": "project-attribution-scope",
        "tier": 2,
        "title": "What the attribution model is, and what it is not",
        "text": (
            "Modality attribution in this system is computed on the LOCAL, "
            "PRE-DP model for one client's round - the model that reaches "
            "roughly F1~0.53 on that client's own held-out split. It is NOT "
            "computed on the aggregated global model produced under "
            "differential privacy, which is degenerate on this corpus "
            "(F1=0, single-class predictions on all held-out records). This "
            "is a deliberate scope choice, not an oversight."
        ),
        "source": "docs/IMPLEMENTATION_NOTES.md, 'Phase A: what the attribution mechanism can and cannot support'",
        "verified": True,
        "verification_note": _PROJECT_NOTE,
        "tags": ["attribution", "scope", "methodology"],
    },
    {
        "id": "project-baseline-artefact",
        "tier": 2,
        "title": "The original modality-dominance finding was a scale artefact",
        "text": (
            "An early measurement found one modality (audio) unanimously "
            "dominant across every held-out sample. Investigation traced "
            "this to the attribution baseline: a zero-vector baseline is a "
            "far more extreme, out-of-distribution 'removed' state for a "
            "large-magnitude feature than for a small-magnitude one, which "
            "inflates that modality's apparent attribution regardless of "
            "genuine importance. Switching to in-distribution mean-vector "
            "baselines for all three modalities removed this artefact."
        ),
        "source": "docs/IMPLEMENTATION_NOTES.md, Steps A4-A5",
        "verified": True,
        "verification_note": _PROJECT_NOTE,
        "tags": ["attribution", "methodology", "limitation"],
    },
    {
        "id": "project-per-patient-unreliable",
        "tier": 2,
        "title": "Per-patient modality attribution does not agree across methods",
        "text": (
            "After correcting the baseline artefact, two independent "
            "attribution methods (Integrated Gradients and a modality-"
            "ablation check) were tested against each other on the same 37 "
            "held-out records. They agreed on the cohort-level aggregate "
            "pattern, but did NOT agree at the individual-patient level - "
            "one method was unanimous across all patients while the other "
            "was close to a random three-way split. Per-patient modality "
            "attribution from this mechanism should not be trusted; "
            "aggregate, cohort-level attribution is supportable."
        ),
        "source": "docs/IMPLEMENTATION_NOTES.md, Step A6",
        "verified": True,
        "verification_note": _PROJECT_NOTE,
        "tags": ["attribution", "limitation", "per_patient", "reliability"],
    },
    {
        "id": "project-uncertainty-root-cause",
        "tier": 2,
        "title": "Most predictions on this corpus are low-confidence",
        "text": (
            "The leading hypothesis for why attribution methods disagree "
            "per-patient (that it was caused by how finely the attribution "
            "method integrates its calculation) was tested directly and "
            "refuted. The more likely explanation: on this held-out "
            "evaluation set, the model's predicted probability sat between "
            "0.3 and 0.7 - close to the decision boundary, not confidently "
            "toward either class - for the large majority of patients. A "
            "model that rarely commits confidently produces attribution "
            "and prediction values that are inherently less stable per "
            "individual case, even though its aggregate, cohort-level "
            "behaviour remains measurable."
        ),
        "source": "docs/IMPLEMENTATION_NOTES.md, Step A7",
        "verified": True,
        "verification_note": _PROJECT_NOTE,
        "tags": ["uncertainty", "confidence", "decision_boundary", "limitation"],
    },
    {
        "id": "project-b8-constraints",
        "tier": 2,
        "title": "What this report is designed to say, and not say, about individual patients",
        "text": (
            "Because per-patient attribution is not reliable, this "
            "reporting system is designed so that: it does not quote "
            "per-patient modality percentages; it does not claim any "
            "modality 'drove' an individual prediction; it surfaces the "
            "model's own uncertainty rather than presenting a prediction as "
            "confident; and it treats aggregate, cohort-level modality "
            "statements as the supportable level of claim. These are "
            "structural design constraints, not just guidance to the "
            "narrative-generation step."
        ),
        "source": "docs/IMPLEMENTATION_NOTES.md, 'Phase A: what the attribution mechanism can and cannot support' (constraints section)",
        "verified": True,
        "verification_note": _PROJECT_NOTE,
        "tags": ["constraints", "scope", "per_patient", "methodology"],
    },
    {
        "id": "project-model-training-limits",
        "tier": 2,
        "title": "This model's training data and validation status",
        "text": (
            "The local model discussed in this report was trained on 149 "
            "records from a single research corpus (DAIC-WOZ), reaching "
            "roughly F1~0.53 on its own held-out split of that same corpus. "
            "It has not been trained or validated on any clinical "
            "population, and its behaviour on patients unlike that research "
            "corpus is unknown."
        ),
        "source": "docs/IMPLEMENTATION_NOTES.md, Phase A/B sections; trainer_mentalbert_privacy.py stratified_split()",
        "verified": True,
        "verification_note": _PROJECT_NOTE,
        "tags": ["training_data", "validation", "limitation"],
    },
]


def corpus_hash() -> str:
    """Stable hash of the corpus content, used as a cache-invalidation key
    for retrieve_context()'s embedding cache (Step C2 point 2) - if this
    file's CORPUS changes, the hash changes, and the cache is recomputed
    rather than silently serving stale embeddings for changed text."""
    canonical = json.dumps(CORPUS, sort_keys=True, ensure_ascii=True)
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


if __name__ == "__main__":
    print(f"{len(CORPUS)} documents, corpus_hash={corpus_hash()[:16]}...")
    for doc in CORPUS:
        print(f"  [{doc['id']}] tier={doc['tier']} verified={doc['verified']} tags={doc['tags']}")
