#!/usr/bin/env python3
"""
scripts/privacy_explanation_agent.py

Phase D — Privacy Explanation Agent.

Post-hoc reporting tool over ALREADY-PERSISTED privacy/audit telemetry for one
completed federated learning round. Does not touch the live pipeline, does
not compute any new privacy metric, does not call any external network
service.

Design (approved, Step D1/D2):
    1. Gather structured facts from two sources for a given --round-id:
         - MongoDB: receipts, model_updates collections (round-level facts)
         - Local disk: ~/.federated/data/receipts/*.json DP-agent receipts
           (mechanism/clip/noise/L2 detail), correlated to the round via the
           session_id that MongoDB's model_updates document already stores
           for that round (exact join key — NOT a time-window guess).
    2. Assemble one flat facts dict. This dict is the single source of truth
       for both the printed table and the LLM prompt.
    3. Render the facts as Markdown tables FIRST.
    4. Feed the SAME facts dict (not raw telemetry, not a free-form summary)
       to a local Ollama model with a fixed prompt that forbids computation.
       The narrative is written BELOW the table so a reader can check it
       against the facts it was given.
    5. Run a small numeric factual-consistency check on the narrative: every
       number the LLM wrote must trace back to a number in the facts dict.
       This is a hallucination guard, not an LLM judging an LLM.

The facts table is written unconditionally, even if Ollama is unreachable —
report generation never depends on the LLM succeeding.

Usage:
    .venv\\Scripts\\python.exe scripts\\privacy_explanation_agent.py --round-id 1
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

DEFAULT_RECEIPTS_DIR = Path.home() / ".federated" / "data" / "receipts"
DEFAULT_OUT_DIR = Path.home() / ".federated" / "data" / "audit_reports"
DEFAULT_OLLAMA_URL = "http://localhost:11434"
DEFAULT_MONGO_URI = "mongodb://localhost:27017"

# Facts about the SYSTEM as built, not about any one round. Sourced from code
# inspection (file:line cited inline), not from a live lookup — there is
# nothing to look up, because none of these are persisted per-round.
STATIC_SYSTEM_FACTS = {
    "aggregation_strategy": (
        "trimmed_mean, trim_ratio=0.1. This is a hardcoded constant in "
        "run_aggregation() (server/orchestration_agent/src/grpc/server.rs:1497-1500), "
        "not read from any per-round or persisted configuration. Every round uses it."
    ),
    "signature_verification": (
        "Reported per receipt below as 'verified': true. This is NOT a per-row "
        "outcome distinguishing verified from unverified receipts — SubmitReceipt "
        "rejects any receipt with an invalid ECDSA signature before it is ever "
        "written to the database (server.rs:528-542). Every stored receipt "
        "necessarily passed signature verification; the field records that "
        "precondition, not a check performed on stored data. Correct framing: "
        "signature verification is enforced at ingest."
    ),
    "hmac_chain_integrity": (
        "Every receipt carries an 'hmac_chain' value linking it to the previous "
        "receipt in the same round (server.rs:629-660). However, RECEIPT_CHAIN_KEY "
        "is not set in any launch script or config file in this repository "
        "(grep-confirmed across the codebase) — the orchestrator therefore falls "
        "back to a fresh random ephemeral key on every process start "
        "(server.rs:92-104). The chain IS computed and stored, but it is only "
        "verifiable within the lifetime of the single orchestrator process that "
        "wrote it. It does NOT establish tamper-evidence across orchestrator "
        "restarts in this deployment."
    ),
    "epsilon_accounting_scope": (
        "epsilon_spent on each receipt is a per-round, single-composition RDP "
        "value (Mironov 2017 accountant, T=1 — installer/runtime/agents/dp/dp_agent.py). "
        "It is not a multi-round privacy loss figure. Any cross-round total in "
        "this report ('cumulative_epsilon_naive_upper_bound') is a plain "
        "arithmetic sum of independent per-round values, NOT a tight RDP "
        "composition across rounds — it should be read as a conservative upper "
        "bound, not a measured cumulative privacy loss."
    ),
}


def _short(s: str, n: int = 16) -> str:
    return s if len(s) <= n else s[:n] + "…"


def gather_facts(round_id: int, db_name: str, mongo_uri: str, receipts_dir: Path) -> dict:
    """Collect all facts for one round. Raises RuntimeError with a helpful
    message (listing available round_ids) if the round has no data."""
    client = MongoClient(mongo_uri, serverSelectionTimeoutMS=5000)
    try:
        client.admin.command("ping")
    except Exception as e:
        raise RuntimeError(
            f"Cannot reach MongoDB at {mongo_uri}: {e}. Is mongod running?"
        )
    db = client[db_name]

    receipts = list(db["receipts"].find({"round_id": round_id}))
    model_updates = list(db["model_updates"].find({"round_id": round_id}))

    if not receipts and not model_updates:
        available = sorted(db["receipts"].distinct("round_id"))
        client.close()
        raise RuntimeError(
            f"No receipts or model_updates found for round_id={round_id} in "
            f"database {db_name!r}. Rounds with data: {available}"
        )

    # global_models documents are keyed by the round the model becomes
    # AVAILABLE for, i.e. aggregated_round + 1 (server.rs comment, confirmed
    # against evaluate_global_model.py's own documented convention). Query
    # that offset explicitly rather than assuming round_id.
    global_model_doc = db["global_models"].find_one({"round_id": round_id + 1})
    client.close()

    # ---- correlate local DP-agent disk receipts via session_id (exact join,
    # not a time-window guess) ----
    session_ids = {mu.get("session_id") for mu in model_updates if mu.get("session_id")}
    session_to_device = {mu.get("session_id"): mu.get("device_id") for mu in model_updates}

    dp_local_receipts = []
    if receipts_dir.exists():
        for f in receipts_dir.glob("*.json"):
            try:
                data = json.loads(f.read_text())
            except (json.JSONDecodeError, OSError):
                continue
            if data.get("operation") != "dp_process_update":
                continue
            sid = data.get("session_id")
            if sid not in session_ids:
                continue
            p = data.get("params", {})
            dp_local_receipts.append({
                "session_id": sid,
                "device_id": session_to_device.get(sid),
                "mechanism": p.get("mechanism"),
                "clip_norm": p.get("clip_norm"),
                "clip_applied": p.get("clip_applied"),
                "noise_multiplier": p.get("noise_multiplier"),
                "delta": p.get("delta"),
                "l2_norm_before": p.get("l2_norm_before"),
                "l2_norm_after": p.get("l2_norm_after"),
                "epsilon_spent": p.get("epsilon_spent"),
                "timestamp": data.get("timestamp"),
                "source_file": f.name,
            })

    receipts_out = []
    for r in receipts:
        receipts_out.append({
            "device_id": r.get("device_id"),
            "epsilon_spent": r.get("epsilon_spent"),
            "scheme": r.get("scheme"),
            "enc_handle": str(r.get("enc_handle")),
            "payload_hash": r.get("payload_hash"),
            "timestamp": str(r.get("timestamp")),
            "verified": r.get("verified"),
            "hmac_chain": r.get("hmac_chain"),
        })

    model_updates_out = []
    for mu in model_updates:
        raw_size = mu.get("size_bytes")
        model_updates_out.append({
            "device_id": mu.get("device_id"),
            "session_id": mu.get("session_id"),
            # Step B8: pre-formatted as a comma-grouped string, not a bare
            # int - proven 2/2 trials under the real prompt (Step B6
            # investigation, docs/IMPLEMENTATION_NOTES.md) to fix phi3:mini
            # reliably mangling large digit-groups (1506992 -> "150,6992")
            # when writing it out in a busy, many-numbers narrative. A bare
            # int is still available as size_bytes_raw for any caller that
            # needs the actual number rather than the display string.
            "size_bytes": f"{raw_size:,}" if isinstance(raw_size, (int, float)) else raw_size,
            "size_bytes_raw": raw_size,
            "upload_time": str(mu.get("upload_time")),
            "verified": mu.get("verified"),
        })

    round_epsilon_total = sum(
        r["epsilon_spent"] for r in receipts_out
        if isinstance(r["epsilon_spent"], (int, float))
    )

    # naive cumulative across every round <= round_id (see STATIC_SYSTEM_FACTS
    # epsilon_accounting_scope — explicitly NOT a tight composition)
    client2 = MongoClient(mongo_uri, serverSelectionTimeoutMS=5000)
    db2 = client2[db_name]
    cumulative = sum(
        rc.get("epsilon_spent", 0.0)
        for rc in db2["receipts"].find({"round_id": {"$lte": round_id}})
        if isinstance(rc.get("epsilon_spent"), (int, float))
    )
    client2.close()

    facts = {
        "round_id": round_id,
        "database": db_name,
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "receipts": receipts_out,
        "model_updates": model_updates_out,
        "dp_local_receipts": dp_local_receipts,
        "aggregates": {
            "num_receipts": len(receipts_out),
            "num_unique_devices": len({r["device_id"] for r in receipts_out}),
            "round_epsilon_total": round_epsilon_total,
            "cumulative_epsilon_naive_upper_bound_through_this_round": cumulative,
            "global_model_produced": global_model_doc is not None,
            "global_model_queried_round_id": round_id + 1,
        },
        "static_system_facts": STATIC_SYSTEM_FACTS,
    }
    return facts


def render_facts_markdown(facts: dict) -> str:
    a = facts["aggregates"]
    lines = []
    lines.append(f"## Round {facts['round_id']} — facts (database: `{facts['database']}`)\n")
    lines.append(
        "**Every number in this section comes directly from MongoDB "
        "(`receipts`, `model_updates`, `global_models`) or from local DP-agent "
        "receipt files. Nothing in this section was computed or estimated by "
        "the LLM.** The narrative further below was generated from exactly "
        "this data — check it against these tables, not the other way around.\n"
    )

    lines.append("### Round summary\n")
    lines.append("| Field | Value |")
    lines.append("|---|---|")
    lines.append(f"| Round ID | {facts['round_id']} |")
    lines.append(f"| Receipts submitted | {a['num_receipts']} |")
    lines.append(f"| Unique devices | {a['num_unique_devices']} |")
    lines.append(f"| Round epsilon total (sum of this round's receipts) | {a['round_epsilon_total']:.6f} |")
    lines.append(
        f"| Cumulative epsilon through round {facts['round_id']} "
        f"(naive additive upper bound — see caveat below) | "
        f"{a['cumulative_epsilon_naive_upper_bound_through_this_round']:.6f} |"
    )
    lines.append(
        f"| Global model produced for round {a['global_model_queried_round_id']} | "
        f"{'yes' if a['global_model_produced'] else 'no'} |"
    )
    lines.append("| Aggregation strategy (system constant, not per-round data) | trimmed_mean, trim_ratio=0.1 |")
    lines.append("")

    lines.append("### Server-side receipts (MongoDB `receipts`)\n")
    lines.append("| Device (short) | Epsilon spent | Scheme | Enc handle (short) | Verified* |")
    lines.append("|---|---|---|---|---|")
    for r in facts["receipts"]:
        lines.append(
            f"| {_short(r['device_id'] or '')} | {r['epsilon_spent']:.6f} | "
            f"{r['scheme']} | {_short(r['enc_handle'])} | {r['verified']} |"
        )
    lines.append(
        "\n*\\*'Verified' is enforced at ingest — an invalid signature never "
        "reaches storage, so this column cannot distinguish stored receipts "
        "from each other. See caveats below.*\n"
    )

    lines.append("### Client-side DP mechanism detail (local disk receipts, joined by session_id)\n")
    if facts["dp_local_receipts"]:
        lines.append("| Session | Mechanism | Clip norm | Clip applied | Noise multiplier | L2 before | L2 after | Epsilon |")
        lines.append("|---|---|---|---|---|---|---|---|")
        for d in facts["dp_local_receipts"]:
            eps = d["epsilon_spent"]
            eps_str = f"{eps:.6f}" if isinstance(eps, (int, float)) else str(eps)
            lines.append(
                f"| {_short(d['session_id'] or '')} | {d['mechanism']} | {d['clip_norm']} | "
                f"{d['clip_applied']} | {d['noise_multiplier']} | "
                f"{d['l2_norm_before']:.6f} | {d['l2_norm_after']:.6f} | {eps_str} |"
            )
    else:
        lines.append(
            "*No local DP-agent receipt files matched this round's session IDs "
            f"under `{DEFAULT_RECEIPTS_DIR}`.*"
        )
    lines.append("")

    lines.append("### System-level caveats (apply to every round, not just this one)\n")
    for key, text in facts["static_system_facts"].items():
        lines.append(f"- **{key}**: {text}")
    lines.append("")

    return "\n".join(lines)


def build_prompt(facts: dict) -> str:
    facts_json = json.dumps(facts, indent=2, default=str)
    return f"""You are a compliance-reporting assistant. Below is a JSON object of
ALREADY-VERIFIED facts about one federated learning round, gathered from a
database and signed receipt files. Your job is to NARRATE these facts in
plain English for a non-technical reader. You do not compute anything.

STRICT RULES:
1. Use ONLY numbers that literally appear in the JSON below. Do not compute,
   estimate, round differently, convert units, or introduce any new number.
2. If you state a number, copy it as it appears in the JSON.
3. Explicitly and plainly mention the three caveats under "static_system_facts":
   that signature verification is enforced at ingest (not a per-row check),
   that the HMAC chain is only verifiable within one orchestrator process
   lifetime (RECEIPT_CHAIN_KEY was never configured), and that epsilon
   figures are per-round single-composition values (any cumulative figure is
   a naive additive upper bound, not a tight composition).
4. Do not invent participant counts, clinical outcomes, or model predictions.
   This JSON is infrastructure/privacy telemetry only.
5. Write 3-5 short paragraphs. No headings, no bullet lists, no markdown.

JSON facts:
{facts_json}

Now write the narrative.
"""


def main() -> int:
    ap = argparse.ArgumentParser(description="Phase D — Privacy Explanation Agent")
    ap.add_argument("--round-id", type=int, required=True)
    ap.add_argument("--db", default="federated_multimodal")
    ap.add_argument("--model", default="phi3:mini")
    ap.add_argument("--out", default=str(DEFAULT_OUT_DIR))
    ap.add_argument("--mongo-uri", default=DEFAULT_MONGO_URI)
    ap.add_argument("--receipts-dir", default=str(DEFAULT_RECEIPTS_DIR))
    ap.add_argument("--ollama-url", default=DEFAULT_OLLAMA_URL)
    ap.add_argument("--ollama-timeout", type=int, default=240,
                     help="generous read timeout in seconds for LLM generation")
    args = ap.parse_args()

    t_start = time.time()

    print(f"[privacy-agent] gathering facts for round_id={args.round_id} "
          f"from db={args.db!r} and {args.receipts_dir} ...")
    try:
        facts = gather_facts(args.round_id, args.db, args.mongo_uri, Path(args.receipts_dir))
    except RuntimeError as e:
        print(f"[privacy-agent] ERROR: {e}", file=sys.stderr)
        return 1
    t_gather = time.time() - t_start
    print(f"[privacy-agent] facts gathered in {t_gather:.2f}s "
          f"({facts['aggregates']['num_receipts']} receipts, "
          f"{len(facts['dp_local_receipts'])} local DP receipts matched)")

    facts_md = render_facts_markdown(facts)

    prompt = build_prompt(facts)
    narrative, llm_error, t_llm = call_ollama(prompt, args.model, args.ollama_url, args.ollama_timeout)

    consistency = None
    if narrative:
        consistency = check_consistency(narrative, facts)
        status = "PASSED" if consistency["passed"] else "WARNING — untraceable numbers found"
        print(f"[privacy-agent] factual-consistency check: {status} "
              f"({consistency['num_checked_as_claims']} numeric claims checked "
              f"against {consistency['num_fact_numbers_available']} known facts)")
        if not consistency["passed"]:
            print(f"[privacy-agent]   untraceable: {consistency['untraceable_numbers']}")
    else:
        print(f"[privacy-agent] narrative unavailable: {llm_error}")

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"round_{args.round_id}_privacy_report.md"

    doc = []
    doc.append(f"# Privacy & Audit Report — Round {args.round_id}\n")
    doc.append(f"Generated {facts['generated_at']} by `privacy_explanation_agent.py` "
               f"(local Ollama model: `{args.model}`)\n")
    doc.append(
        "> **How to read this report.** The tables below are the complete set of "
        "facts this report is based on, gathered directly from MongoDB and signed "
        "local receipts. The LLM never computes anything — it only narrates the "
        "table beneath it into plain English. Check the narrative against the "
        "table, not the other way around.\n"
    )
    doc.append(facts_md)

    doc.append("## Factual-consistency check\n")
    if consistency is not None:
        doc.append(f"**Result: {'PASSED' if consistency['passed'] else 'WARNING — narrative contains numbers not traceable to the facts above'}**\n")
        doc.append(
            f"- Numeric claims extracted from the narrative: {consistency['num_checked_as_claims']} "
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
            doc.append(f"- **Untraceable numbers (present in the narrative, not found in the facts):** "
                        f"{consistency['untraceable_numbers']}\n")
            doc.append("  Treat the narrative below with caution — verify manually against the tables above.\n")
    else:
        doc.append("Not run — no narrative was generated (see below).\n")

    doc.append("## Narrative (LLM-generated, local Ollama)\n")
    if narrative:
        doc.append(narrative + "\n")
    else:
        doc.append(f"*Narrative unavailable — {llm_error}*\n"
                    f"\nThe facts tables above are unaffected; they do not depend on the LLM.\n")

    t_total = time.time() - t_start
    doc.append("---\n")
    doc.append(
        f"*Runtime: {t_total:.2f}s total — data gathering {t_gather:.2f}s, "
        f"LLM generation {t_llm:.2f}s. No external network calls: MongoDB at "
        f"{args.mongo_uri}, Ollama at {args.ollama_url} — both localhost only.*\n"
    )

    out_path.write_text("\n".join(doc), encoding="utf-8")

    print(f"[privacy-agent] report written to {out_path}")
    print(f"[privacy-agent] total runtime {t_total:.2f}s "
          f"(gather {t_gather:.2f}s, llm {t_llm:.2f}s)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
