# VERSION 2 — FINAL CONSISTENCY REVIEW
## Read-only audit of Phases 10 – 14

**Scope:** `PHASE_10_ROADMAP.md` · the seven Phase 11 documents · Phase 12 (Exp 4) · Phase 13 (Exp 3) · Phase 14 (Exp 5), together with their experiment directories, scripts and artifacts.
**Method:** automated cross-checking of every full and abbreviated SHA-256 reference against files on disk; every backticked path and inter-document reference resolved; experiment numbering, terminology, appendix structure, heading duplication and artifact counts compared across all fourteen documents.
**Mode:** **read-only.** Nothing was modified, and no fix in this document has been applied.

---

## Verdict

**The repository is NOT yet fully internally consistent.** Twelve issues were found: **2 high**, **3 medium**, **5 low**, plus **5 informational** items that are *not* defects but are recorded because a reviewer would otherwise flag them.

**None of the issues affects any measured value, any statistical conclusion, or any experimental verdict.** Every number in every result report reconciles with its artifacts. The two high-severity items are both **broken digest provenance** — audit-trail defects, not data defects.

**The scientific record is sound. The citation and navigation layer needs work before thesis writing.**

---

## Summary table

| # | Severity | Area | Issue |
|---|---|---|---|
| **H1** | **HIGH** | Phase 12 | `verify_exp4.py` pre-registration digest matches no file on disk; no erratum recorded |
| **H2** | **HIGH** | Phase 14 | Four malformed abbreviated SHA references in the results report's provenance line |
| **M1** | MEDIUM | Phase 11 | Superseded conclusions (11.2, 11.3) carry no supersession notice in their own documents |
| **M2** | MEDIUM | Phase 10 | Roadmap §6 Exp-0b branch names Exp 3 while describing Exp 4's intervention |
| **M3** | MEDIUM | Phase 10 / 14 | Roadmap's "improve over Exp 4" criterion is not like-for-like across decision rules |
| **L1** | LOW | Phases 12–14 | Appendix structure diverges across the three pre-registrations |
| **L2** | LOW | Phases 12–14 | Result-report filenames follow no single convention |
| **L3** | LOW | Phase 11 | Two Phase 11.6 documents live outside the repository root |
| **L4** | LOW | Entry points | `README.md` and `PROJECT_KNOWLEDGE.md` predate Experiments 3, 4 and 5 |
| **L5** | LOW | Housekeeping | Five stale bundle archives and two redundant bundlers at root |
| I1–I5 | INFO | — | Verified-correct items that resemble defects |

---

# HIGH SEVERITY

## H1 — Phase 12's pre-registration digest matches no file on disk

**Location:** `PHASE_12_EXP4_DESIGN.md:322` (Appendix A — Pre-registration record)

**Recorded:**
```
5dcfe619b4da7bccf276186c7c2c5aa175a9fbbfef6dd61ccbade5590c68b3a8  exp4_decision_rule/verify_exp4.py
```

**Actual on disk:**
```
4820215fa343b4e536fbe6e5e57f11ee36aa95e5965bbd93b1d14142cf3272b9  exp4_decision_rule/verify_exp4.py
```

**Root cause.** During Experiment 4's execution the verifier's `A3` check asserted an artifact count of 32 against an expected set containing 31 names, making it unsatisfiable by construction. The check was corrected to derive the count from the expected set. `PHASE_12_EXP4_DECISION_RULE.md:77` describes this correction in prose — but **records no digest**, and Appendix A of the design document was never amended.

**Why this is high severity.** The pre-registration record's entire function is to let a third party verify that the code which produced the results is the code that was registered before execution. For `verify_exp4.py` that chain is currently broken: the registered digest identifies no artifact that exists, and the report that describes the change does not supply the replacement. Experiment 4's *runner* and *aggregator* digests are both intact and verified, so **no measured value is in question** — but an examiner checking provenance will find a dangling hash.

**The contrast that makes this actionable.** Phase 13 encountered the *identical* class of defect — its `J2` check hard-coded a file count — and handled it correctly:

> `PHASE_13_EXP3_DESIGN.md:382` — `### Erratum 1 — verify_exp3.py (2026-08-04, approved)`
> `PHASE_13_EXP3_CONVERGENCE.md:210` — ``verify_exp3.py`` `4a1de483…5dfe` **(post-erratum; pre-registered as** `c31481c6…285cb`**)**

Phase 13 records both digests, labels which is which, and dates the change. Phase 12 records neither. **The remedy is to bring Phase 12 up to Phase 13's standard**, not to alter any code.

---

## H2 — Four malformed abbreviated SHA references in the Phase 14 results report

**Location:** `PHASE_14_EXP5_RESULTS.md` §21, Provenance line

| Cited in report | Actual digest ends | Status |
|---|---|---|
| `2ecbfee3…78d572` | `…578ff8c5d572` | **wrong suffix** |
| `f065f5e1…871faac` | `…5674871aac` | **wrong suffix** |
| `5defdae2…1800aa6f` | `…1800aa6f2` | **truncated by one character** |
| `66bbac11…8ca57` | `…68e701ca57` | **wrong suffix** |

Six further abbreviations on the same line — `65b1902e…a230b`, `b9a7a91f…af8fca2f`, `9a241851…4fc55f00`, `42345081…5cbb7f`, `57843f65…2b6b8`, `1149b9ba…48c0a` — **resolve correctly.**

**Why this is high severity.** These four references cannot be used to verify the artifacts they name. A reader attempting to confirm the frozen inputs of the project's final experiment will find four of ten digests unmatchable.

**Mitigating factor.** These are *abbreviations in a prose provenance line*. The authoritative full digests live in `PHASE_14_EXP5_DESIGN.md` Appendix A.1/A.2 and in `exp5_summary.json`'s `frozen_input_sha` block — **all verified correct in this review**. No measured value, criterion or verdict is affected, and the independent audit (3443 checks, 0 failures) is unaffected.

**Origin.** These are transcription errors I introduced when writing the provenance line. They are reported here rather than silently corrected because this review is read-only.

---

# MEDIUM SEVERITY

## M1 — Superseded Phase 11 conclusions carry no supersession notice

**Locations:** `PHASE_11_DIAG_B_THRESHOLD_FREE_ANALYSIS.md` (Phase 11.2) · `PHASE_11_DIAGNOSTIC_SYNTHESIS.md` (Phase 11.3)

Phase 11.2 concludes, at n = 18:

> `:132` — *"**No detectable ranking information.** ROC-AUC = 0.523 with 95% CI [0.250, 0.787] … the best estimate is 'at chance.'"*
> `:171` — *"**Experiment 0b outcome: AUC at chance** — decision-rule-defect hypothesis not supported."*

Phase 11.3 carries this forward and lowers Exp 4's expected value:

> `:94` — *"its ceiling is low because the ordering is at chance … **Expectation lowered.**"*

**Phase 11.7 explicitly overturns all of this** at full power (n = 188, 25 fold-runs):

> `PHASE_11_7_BASELINE_CV_INTERPRETATION.md:101` — *"**SUPERSEDED.** AUC 0.6333, CI [0.5755, 0.6911]; 25/25 fold-runs above chance; sign test p = 5.96e-8."*
> `:105` — *"'Exp 4's ceiling is low… expectation lowered' — **SUPERSEDED.**"*

**The defect is directional.** Phase 11.7 knows it supersedes 11.2 and 11.3; **11.2 and 11.3 do not know they have been superseded.** Neither document carries a forward pointer. A reader — or a thesis chapter drafted from them — opening either file directly encounters "ranking is indistinguishable from chance" as a live conclusion of the project.

**This is the single most likely source of a factual error in written-up work**, because 11.2 and 11.3 are the natural documents to cite when narrating the diagnostic phase. A one-line banner at the head of each would resolve it; the underlying analyses were correct for the evidence they had and should not be rewritten.

---

## M2 — Roadmap §6 Exp-0b branch names the wrong experiment

**Location:** `PHASE_10_ROADMAP.md:248`

> *"**[H]** AUC materially above chance ⇒ W#1 is substantially a **decision-rule defect** and **Exp 3** becomes high-yield. AUC at chance ⇒ Exps 4–5 become mandatory."*

The roadmap's own table, three rows below on lines 251–252, defines:

| # | Experiment | Single changed factor | Line |
|---|---|---|---|
| **3** | Convergence | epoch policy | `:251` |
| **4** | Decision rule | decision threshold | `:252` |

A *decision-rule defect* implicates **Exp 4**, not Exp 3. The branch text and the intervention it describes disagree.

**Consequence, and why it is only medium.** The conditional resolved to its *other* branch — Phase 11.2 measured AUC at chance — so the mis-numbered clause was never acted upon. Furthermore Phase 11.7 later reversed the underlying finding and correctly promoted **Exp 4**, which the project then executed as Phase 12. **No decision was taken on the erroneous text.** It remains a live error in the governing document, and it is the branch a reader would consult to understand why Exp 4 was run.

---

## M3 — The roadmap's "improve over Exp 4" criterion is not like-for-like

**Locations:** `PHASE_10_ROADMAP.md:253` · `PHASE_14_EXP5_DESIGN.md` C3 · `PHASE_14_EXP5_RESULTS.md` §14

The roadmap specifies Experiment 5's comparator as:

> *"`Baseline-CV` **and Exp 4** … Recall / F1 / PR-AUC improve over **both**."*

Experiment 4's Recall 0.5422 and F1 0.2215 were measured at **threshold policy P1**. Experiment 5 was measured at **argmax**, because Exp 4 returned H₀ and its threshold policy was therefore never adopted into the accepted configuration. The two figures are not comparable, and a criterion demanding improvement over a tuned-threshold Recall is close to unsatisfiable for any experiment operating at argmax.

**Both Phase 14 documents handle this correctly** — C3 is labelled *[threshold-conditional]*, the caveat is recorded in `exp5_summary.json` under `exp4_reference`, and §14 states plainly that the comparison is "not like-for-like." **The defect is in the roadmap criterion, not in the experiment.**

**This will recur.** Roadmap §6 #6 specifies Exp 6's comparator as *"`Baseline-CV` **and Exp 5**"* — and Exp 5 also returned H₀, so it likewise contributes no accepted configuration. The roadmap's chained-comparator design implicitly assumes each experiment is accepted; three consecutive H₀ results have left that assumption unmet. **Worth resolving before Exp 6 is designed**, though it does not invalidate anything already done.

---

# LOW SEVERITY

## L1 — Appendix structure diverges across the three pre-registrations

| Document | Structure |
|---|---|
| `PHASE_12_EXP4_DESIGN.md` | undivided *"Appendix A — Pre-registration record"*; no A.1/A.2 labels; no erratum section |
| `PHASE_13_EXP3_DESIGN.md` | Appendix A + *"Erratum 1"*; no A.1/A.2 labels |
| `PHASE_14_EXP5_DESIGN.md` | labelled **A.1** (frozen inputs) and **A.2** (implementation code) |

Only Phase 14 is self-describing. Cross-references such as *"Appendix A.2"* in Phase 14 have no counterpart in Phases 12 and 13, so the citation form is not portable across the three experiments. This is a presentation inconsistency, not an error — each appendix contains the right content for its phase.

## L2 — Result-report filenames follow no single convention

```
PHASE_12_EXP4_DECISION_RULE.md     ← names the intervention
PHASE_13_EXP3_CONVERGENCE.md       ← names the intervention
PHASE_14_EXP5_RESULTS.md           ← says "RESULTS"
```

All three are result reports. Two are named after the factor tested, one after its role. From filenames alone a reader cannot distinguish design documents from result documents in Phases 12 and 13 without opening them. (`PHASE_11_BASELINE_CV.md` is a fourth pattern again.)

## L3 — Two Phase 11.6 documents live outside the repository root

```
colab_baseline_cv/PHASE_11_BASELINE_CV_AUDIT.md            # PHASE 11.6 — BASELINE-CV EXECUTION PACKAGE AUDIT
colab_baseline_cv/PHASE_11_BASELINE_CV_COLAB_PACKAGE.md    # PHASE 11.6 — BASELINE-CV COLAB EXECUTION PACKAGE
```

Every other `PHASE_*.md` document sits at the repository root. These two are referenced by no root document and are easily missed when enumerating the Phase 11 record. Their content is legitimate Phase 11.6 material.

## L4 — The entry-point documents predate Experiments 3, 4 and 5

| Document | Latest phase referenced | Mentions of Exp 3 / 4 / 5 |
|---|---|---|
| `README.md` | none ≥ 10 | 0 / 0 / 0 |
| `PROJECT_KNOWLEDGE.md` | Phase 12 | 0 / 0 / 0 |

The two documents a new reader opens first describe a repository state three phases old. Neither mentions that the Immediate block is complete or that three experiments returned H₀. `ARCHITECTURE.md` is infrastructure-scoped and legitimately silent on experiments.

## L5 — Stale bundle archives and redundant bundlers at root

```
baseline_cv_colab.zip     835,585 B    2026-08-01
exp3_colab.zip            875,384 B    2026-08-03
exp3_convergence.zip       78,314 B    2026-08-04
exp3_upload.zip           875,653 B    2026-08-04
exp5_upload.zip           891,772 B    2026-08-05
```

All five are regenerable build outputs. Additionally `make_exp3_zip.py` and `make_exp3_upload_bundle.py` are two bundlers for the same experiment, with no document stating which supersedes the other. No correctness impact; purely repository hygiene.

---

# INFORMATIONAL — verified correct, recorded to prevent re-flagging

**I1 — `PHASE_13_EXP3_DESIGN.md`'s unresolvable digest `c31481c6…285cb` is correct.** It is the *pre-erratum* `verify_exp3.py` digest, explicitly labelled as such in both the design document and the result report. A digest matching no file is exactly right here — it records what was registered before a documented, approved correction. **This is the model H1 should follow.**

**I2 — `PHASE_10_ROADMAP.md:78`'s `626dfc2a…d1b6` is not a repository file.** It is the Phase 9 global-model hash (round_id 2, 1,184,989 B). Correctly unresolvable against the working tree.

**I3 — Three "missing" file references are templates, not broken links.** `train_fold_f.parquet` and `test_fold_f.parquet` (`PHASE_11_BASELINE_CV.md`) use `f` as a fold-index placeholder; `full_test_split.csv` (`PHASE_10_ROADMAP.md`) names a planned artifact. All other backticked paths across all fourteen documents resolve.

**I4 — The runtime-mirror landmine is currently clean.** `colab_package/trainer_mentalbert_daic.py` and `colab_package/daic_records.parquet` are **byte-identical** to the root originals (`65b1902e…`, `9a241851…`). This is a known divergence risk in this repository and is presently not diverged.

**I5 — Artifact inventories reconcile exactly.**

| Directory | On disk | Claimed in docs |
|---|---|---|
| `baseline_cv/` | 34 | 34 ✅ |
| `exp3_convergence/` | 97 | 97 ✅ |
| `exp4_decision_rule/` | 31 | 31 ✅ |
| `exp5_imbalance_objective/` | 127 | 127 ✅ |

All **289** files remain git-ignored (`.gitignore:20` `*.csv`, `:114` `trainer_outputs/`) and exist **only in the working tree.** This is not a consistency defect, but it is the largest single risk to the project: four phases of audited evidence have no backup.

---

# What this review confirms as sound

These were checked and found correct; they are recorded because a consistency review that reports only faults misrepresents the state of the work.

**No contradictory conclusions among the experiment reports.** Phases 12, 13 and 14 each returned H₀, and each correctly characterises the others. Phase 13 §8 anticipated Exp 5's failure mode — *"both are ways of appearing to improve without improving"* — and Phase 14's criteria C5 and C6 were written from it and are what caught the null. Phase 12's self-qualification (`:205`, Phase 11.7's recommendation "partially superseded by its own consequence") is accurate.

**No duplicated sections.** Zero repeated headings within any of the fourteen documents.

**All inter-document references resolve.** Every `PHASE_*.md` cited from another phase document exists.

**Terminology is consistent.** *"prior-collapse"* (25 uses) is used uniformly; the two uses of *"prediction collapse"* are a deliberate Phase 13 distinction between collapse of the output distribution and collapse of accuracy, and that distinction is explicitly drawn where used. `Baseline-CV` is capitalised consistently in all 222 occurrences.

**Experiment numbering is otherwise sound.** Apart from M2, every reference to Exps 0a, 0b, 1–9 is consistent with the roadmap §6 table. The execution order Exp 2 → Exp 4 → Exp 3 → Exp 5 is deliberate, justified in `PHASE_13_EXP3_DESIGN.md` and re-justified in `PHASE_13_EXP3_CONVERGENCE.md` §9, and does not constitute a numbering defect.

**Statistical protocol is uniform across all four measurements.** Fold-level Student-*t*, df = 4, t = 2.7764451051977987, repeats averaged within fold first — identical in Baseline-CV, Exp 4, Exp 3 and Exp 5, so all four remain mutually comparable. Prevalence is recorded to full precision (0.2393617021276596) everywhere it appears.

**Every pre-registration preceded its execution.** All three design documents record digests taken before the corresponding `trainer_outputs/` directory existed, and each states so explicitly.

---

# Recommended remediation order

No fix has been applied. Ranked by risk to written work:

1. **H1** — add an erratum to `PHASE_12_EXP4_DESIGN.md` Appendix A recording `verify_exp4.py`'s post-correction digest `4820215f…72b9` alongside the pre-registered `5dcfe619…b3a8`, following Phase 13's format exactly. *Restores the only broken provenance chain in the project.*
2. **H2** — correct the four abbreviated digests in `PHASE_14_EXP5_RESULTS.md` §21.
3. **M1** — add a supersession banner to `PHASE_11_DIAG_B_THRESHOLD_FREE_ANALYSIS.md` and `PHASE_11_DIAGNOSTIC_SYNTHESIS.md` pointing to Phase 11.7. *Highest risk of propagating a false statement into the thesis.*
4. **M2** — correct "Exp 3" → "Exp 4" in `PHASE_10_ROADMAP.md:248`.
5. **M3** — decide how chained comparators behave when the preceding experiment returns H₀, before Exp 6 is designed.
6. **I5** — commit or otherwise back up the 289 artifact files. *Independent of every other item and of any scientific decision.*
7. **L1–L5** — presentation and hygiene; safe to defer.

---

## Readiness statement

**The experimental record — Baseline-CV, Exp 4, Exp 3, Exp 5 — is internally consistent, fully audited, and scientifically ready for thesis writing.** Every measured value reconciles with its artifacts; every verdict follows from pre-registered criteria; no two documents assert contradictory findings about the same measurement.

**The documentation layer requires the two high-severity provenance fixes and the M1 supersession banners before write-up.** Until M1 is addressed, the Phase 11.2 and 11.3 documents will assert, without qualification, a conclusion the project has formally superseded.

---

*Read-only review. No file was modified, no experiment re-run, no artifact regenerated, no code changed. Every finding cites a file and line or a computed digest, and every claim in this document was derived from the repository as it stands.*
