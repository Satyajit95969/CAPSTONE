# PHASE 11.5 — EVALUATION PROTOCOL VALIDATION
## Methodology Audit of PHASE_11_EVALUATION_PROTOCOL.md

**Phase:** 11.5 — Validate the Version 2 evaluation protocol before it governs any experiment
**Type:** Methodology audit — no training, no code, no implementation, no roadmap change
**Subject:** `PHASE_11_EVALUATION_PROTOCOL.md` (Phase 11.4)
**Basis:** Completed diagnostics 11.0–11.3 · `PHASE_10_ROADMAP.md` §6–§7

Convention: **[OBS]** = an observation made during the audit · **[REC]** = a recommendation. Applied fixes are marked **[APPLIED]**.

---

## 1. Validation Objective

Confirm that the Phase 11.4 protocol is **internally consistent, scientifically defensible, executable with the current project, and roadmap-compatible** — *before* it is used to judge Version 2 experiments. A protocol that ships with an internal contradiction or a leakage vector would silently corrupt every experiment that trusts it, so genuine flaws found here are corrected by surgical clarification (not rewrite), and everything else is confirmed sound.

---

## 2. Internal Consistency

Systematic cross-check of metric definitions, reporting rules, comparison framework, and interpretation guidance. Four genuine issues were found; all are clarifications (the protocol's *intent* was coherent, but the wording admitted a contradictory reading). No fatal flaw; no rewrite required.

| ID | [OBS] Issue | Severity | Class | Resolution |
|---|---|---|---|---|
| **C-1** | **P-3 (Reproducibility) vs §7 (noise band via repeats) contradicted each other.** P-3 read "fixes and records its seeds… a result that cannot be regenerated is not a result," which implies bitwise determinism (torch seeded). §7 requires **R repeats** to characterize run-to-run variance — which requires torch to be **un**seeded. As written, one rule demanded determinism and the other demanded stochasticity. | Genuine (contradiction) | **Clarification** | **[APPLIED]** P-3 now defines reproducibility as regenerability of the **manifest, environment, and aggregate statistics (mean ± CI)** — *not* single-run bitwise identity — and states torch stochasticity is deliberately free across repeats. Aligns with Phase 11 roadmap **DP-2** and RC-7. |
| **C-2** | **Operating-point selection was a latent leakage vector.** §3.2 required the threshold be "chosen without peeking at the test fold" but did not say *where* it is chosen, leaving an implementer free to tune it on the held-out fold. | Genuine (leakage risk) | **Clarification** | **[APPLIED]** §3.2 now mandates the operating point be **selected on the training / inner-validation folds only, never the held-out test fold**, cross-referencing Risk R-2. |
| **C-3** | **Threshold-dependent metrics listed beside threshold-free ones without distinction.** Balanced accuracy, MCC, and P/R/F1 depend on the operating point; ROC-AUC/PR-AUC do not. A reader could report balanced accuracy at the degenerate 0.5 default and call it a primary verdict. | Minor (misuse risk) | **Clarification** | **[APPLIED]** Added a note under §3.2 separating threshold-free primaries (carry the verdict on a degenerate model) from threshold-conditional ones. |
| **C-4** | **§3 "frozen floor" numbers conflicted with §5.1/§5.2.** §3 cites AUC 0.523 / PR-AUC 0.311 / mean-predictor MAE 6.3882 as floors a V2 result "must clear," but §5 forbids using any 18-sample frozen number as a quantitative target for a cross-validated result. | Genuine (contradiction) | **Clarification** | **[APPLIED]** Added a §5.2 clause: the §3 "frozen floor" values are **illustrative 18-sample reference points**; the **operative** floors are their `Baseline-CV` re-measurements. Preserves §5.1. |

[OBS] With C-1…C-4 clarified, no remaining rule contradicts another. The metric hierarchy (primary vs secondary vs excluded), the three-layer comparison model, and the interpretation guidelines are now mutually consistent.

---

## 3. Evidence Traceability

Every **[MANDATORY]** rule was traced to a completed diagnostic or a roadmap clause. Result: **all mandatory rules are traceable.** Two metrics are methodological additions (sound, but not derived from project-specific evidence) and are flagged transparently.

| Rule / metric | Traces to | Verdict |
|---|---|---|
| P-1 immutable baseline | Roadmap §2, §7.1 | ✅ traced |
| P-2 one factor | Roadmap §6; [E] Phase 9.5 two-factor confound | ✅ traced |
| P-3 reproducibility | Roadmap §7.5; RC-7; DP-2 | ✅ traced |
| P-4 participant independence | 11.3 §6; RC-2 | ✅ traced |
| P-5 evidence-first | 11.1–11.3; Roadmap §7.3 | ✅ traced |
| P-6 trivial-baseline controls | 11.1; Roadmap §7.4 | ✅ traced |
| P-7 universal invariants | Roadmap §7.5 | ✅ traced |
| Regression: MAE, variance, trivial-gap | 11.1; Roadmap §7.4 | ✅ traced |
| Classification: ROC-AUC, PR-AUC, balanced acc | 11.2; Roadmap §7.4 | ✅ traced |
| Accuracy exclusion | 11.2; Roadmap §7.4 | ✅ traced |
| Three-layer comparison / noise band | Roadmap §7.2, §7.3; 11.6 | ✅ traced |
| **RMSE** | Task-sanctioned ("RMSE if applicable"); general methodology | ⚠️ **[OBS]** not project-evidence-derived — retained as a sound, low-cost complement |
| **MCC** | General methodology (imbalance-robust scalar) | ⚠️ **[OBS]** the one metric added beyond roadmap/diagnostics/task; retained as sound. Not evidence-derived. |

[OBS] **No mandatory rule lacks a basis.** RMSE and MCC are the only additions without a project-specific evidence trace; both are standard, defensible, and non-burdensome, and neither can *lower* rigor (they only add information). [REC] Keep both, with their "methodological addition" provenance recorded (done here). No change required.

---

## 4. Feasibility Assessment

Can every mandatory step actually be executed in later Version 2 experiments? **Yes**, with two non-blocking burden notes and one dependency.

| Item | Feasible? | [OBS] Note |
|---|---|---|
| All primary metrics (MAE, RMSE, variance, ROC-AUC, PR-AUC, balanced acc, MCC, P/R/F1) | ✅ | Standard library computations from predictions + labels; trivially available. |
| Trivial-baseline controls per experiment | ✅ | Cheap; already demonstrated in 11.1/11.2. |
| Participant-level stratified k-fold, repeated | ✅ | Executable on 188 records. **[OBS] Burden F-1:** cost is `R × k` training runs per experiment (~R×5). Across ~10 experiments this is real compute, but each run is the short 3-epoch trainer on Colab GPU — manageable. [REC] Choose R in 11.6 to balance noise-band precision vs compute; this is a parameter choice, not a flaw. |
| Frozen split manifest + noise band | ✅ (dependency) | **[OBS] D-1:** both are produced in **Phase 11.6**; every "exceeds noise band" / "Baseline-CV comparison" rule is inert until 11.6 completes. The ordering (11.6 before Exp 3) already guarantees this. No protocol change needed. |
| Calibration reliability curve (optional) | ✅ | **[OBS] F-2:** at ≈9 positives/held-out fold, calibration curves are low-resolution. Acceptable because calibration is **[OPTIONAL]**; it informs, never decides. No action. |
| Confusion matrix, residual analysis, per-fold breakdown (optional) | ✅ | All computable from stored predictions. |

[OBS] **No mandatory rule is impossible or unnecessarily burdensome.** The only cost concentration (repeated k-fold) is intrinsic to the scientific goal (measuring the noise band the whole protocol depends on), not an accident of the protocol.

---

## 5. Risk Review

| Risk | [OBS] Exposure | How the protocol mitigates it |
|---|---|---|
| **R-1 Statistical misuse** (point estimates, over-claimed significance) | High if unmanaged — 11.2 showed n=18 gives AUC CI [0.25, 0.79] | §5.2 mandates CIs/fold variance; §7 mandates caution until CV; §6 rule 6 forbids effect sizes without uncertainty. **Mitigated.** |
| **R-2 Data leakage** (participant across folds; threshold tuned on test) | Would silently inflate every metric | P-4 mandates participant-level splitting; **C-2 fix** now mandates operating-point selection on inner folds only. **Mitigated (strengthened this phase).** |
| **R-3 Inconsistent reporting** (experiments not comparable) | Would break cross-experiment comparison | §8 fixed reporting template + §5.3 required-artifacts list enforce one structure. **Mitigated.** |
| **R-4 Reproducibility failure** (RC-7 recurrence) | Torch unseeded, manifest drift | P-3 (clarified by **C-1**) mandates frozen manifest + recorded environment + aggregate-statistic regenerability; noise handled by repeats, not hidden. **Mitigated.** |
| **R-5 Overinterpretation** (MAE↑ = learning; accuracy = classification; threshold = classifier) | This is the *exact* failure Phase 10 documented | §3.3 accuracy exclusion; §6 rules 3–5; P-6 trivial controls; the variance + trivial-gap primaries. **Mitigated — this is the protocol's core purpose.** |
| **R-6 Premature-comparator error** (comparing CV result to 18-sample frozen number) | Would repeat Phase 9.5's voided table | §5.1/§5.2 three-layer model; **C-4 fix** relabels frozen floors as illustrative. **Mitigated (strengthened this phase).** |

[OBS] Every identified risk maps to an existing mitigation; the two that were only *partially* covered (R-2 threshold leakage, R-6 frozen-floor conflation) were closed by the C-2 and C-4 clarifications applied this phase.

---

## 6. Required Revisions

All revisions are **Clarification-class**; none is Structural; none required rewriting the protocol's design. Editorial notes are recorded but need no change.

| ID | Class | Status |
|---|---|---|
| C-1 Reproducibility vs repeats contradiction | Clarification | **[APPLIED]** to §2 P-3 |
| C-2 Operating-point leakage vector | Clarification | **[APPLIED]** to §3.2 |
| C-3 Threshold-free vs threshold-dependent primaries | Clarification | **[APPLIED]** as note under §3.2 |
| C-4 "Frozen floor" vs three-layer model conflict | Clarification | **[APPLIED]** to §5.2 |
| E-1 MCC provenance (methodological addition) | Editorial | Recorded here; no protocol change (metric retained as sound) |
| F-1 Repeated-k-fold compute burden | Editorial (feasibility) | Recorded; parameter R to be set in 11.6 |
| F-2 Calibration low-resolution at ~9 positives/fold | Editorial (feasibility) | Recorded; diagnostic is optional |

[OBS] No new **evaluation rules** were introduced. The four applied changes are clarifications that resolve validated ambiguities/contradictions; each either removed a contradiction (C-1, C-4) or closed a leakage/misuse vector (C-2, C-3). This is consistent with the audit constraint "avoid introducing new evaluation rules unless required to resolve a validated flaw."

---

## 7. Final Verdict

> ### ✅ PROTOCOL VALIDATED (with four clarifications applied)

[OBS] The Phase 11.4 evaluation protocol is **internally consistent** (after C-1…C-4), **fully evidence-traceable** (every mandatory rule maps to a diagnostic or roadmap clause; RMSE/MCC flagged as sound methodological additions), **practically executable** (only intrinsic compute cost, no impossible step), and **roadmap-compatible** (operationalizes §7 without altering §6). The four contradictions/vectors found were genuine but wording-level; all are now closed by surgical clarification, and no design change or rewrite was needed.

[REC] The protocol is **cleared to govern Version 2 experiments** — subject to the one standing dependency (D-1): its `Baseline-CV` comparator and empirical noise band must be produced in **Phase 11.6** before any accept/reject decision (Exp 3 onward) is made. Nothing in this audit alters the roadmap or the experiment order.

**Phase 11.5 outcome: methodology audit complete; protocol consistent, traceable, feasible, and roadmap-aligned; four clarifications applied; validated for Version 2 use pending Baseline-CV (11.6).**

---

*Audit only. No training, no predictions, no code/architecture change, no roadmap modification. The only files touched were four clarification edits to `PHASE_11_EVALUATION_PROTOCOL.md` (the audit subject) and this report. Observations [OBS] are separated from recommendations [REC]; no new evaluation rule was introduced beyond resolving validated flaws.*
