# Kill Argument Report — TCTR (Revised, 2026-07-11)

**Artifact**: `vamopd_variational_mixture_formulation.md` (now titled TCTR)
**Verdict**: **WARN** (`reason_code: unresolved_major` — 2 critical points from previous version are now resolved, but 1 new structural risk emerges)

---

## Net Assessment

The revisions address the most damaging attack vectors from the previous version. The "variational" misnomer is gone, DOPD is now cited and included as a baseline, and the three-gate framing (Heuristic / DOPD / Transfer) honestly defines the experimental criteria for success. However, a hostile reviewer still has a survivable attack: the method's core advantage — trajectory-calibrated routing — depends entirely on experimental results that don't yet exist, and the paper's contribution reduces to a data pipeline (scene graph → trajectory) + a lightweight regressor. Whether this is "enough" for CVPR depends on the magnitude of the empirical signal.

---

## Attack Memo (verbatim, ~200 words)

> The authors propose learning per-token teacher mixing weights for multi-teacher
> on-policy distillation, calibrated by programmatically generated trajectories
> from GQA scene graphs.
>
> My concern is that the paper's contribution is structurally thin. The method
> has two components: (1) a programmatic pipeline that converts GQA scene graphs
> into reasoning trajectories, and (2) a lightweight MLP that regresses a scalar
> λ from hand-crafted features (teacher logits, teacher disagreement, hidden
> state) onto trajectory-derived targets. Component (1) is a data engineering
> contribution — the paper should be judged on whether converting an existing
> annotation format to trajectories is a research contribution or simply a
> preprocessing step. Component (2) is a standard regression problem on a data
> source (the trajectories) that only exists because of (1). The router itself
> — an MLP over 5-7 features — contains no architectural innovation.
>
> The paper acknowledges this risk directly (Section 10: "Heuristic gate") and
> correctly identifies that the method's value stands or falls on experimental
> outcomes. But at submission, those outcomes do not exist. The paper currently
> sells a method whose value proposition is "the experiments might be good,"
> not a demonstrated result. This is a promissory note, not a contribution.
>
> Furthermore, even if the experiments do work, the paper must show
> trajectory-calibrated routing is necessary — i.e., that cheap heuristics
> (entropy, disagreement, advantage gap) fail in diagnostically interesting
> ways. If they fail only on 2% of tokens where the error doesn't affect final
> accuracy, the additional complexity of the trajectory pipeline is not
> justified.

---

## Adjudication (per-point)

### Point P_1: Method reduces to data engineering + simple regression
**Attack claim**: Scene graph → trajectory is a preprocessing pipeline; the router is a standard MLP. Neither is a research contribution separately.
**Verdict**: `partially_answered`
**Evidence**: The document acknowledges this structure honestly — the pipeline is explicitly described as programmatic generation (Section 2.3), and the router is a simple regression model (Section 3.2). The paper frames the contribution as the *insight* of using structured annotations for routing, not as a complex algorithm. The answer to this attack is: "The contribution is the connection between two previously disconnected resources — structured benchmark annotations and multi-teacher OPD. Neither component alone is novel, but their combination is, and it produces measurable gains."
**Severity if unresolved**: **major**
**Recommended fix**: This is partially a writing/philosophical issue. The paper should explicitly argue: "Individual components (scene graph traversal, MLP regression) are not novel. The novelty is the *pipeline design* that converts a widely available but underutilized resource into routing supervision for a problem that currently uses either fixed weights or task-free heuristics." Reframe from "new algorithm" to "new resource utilization paradigm."

### Point P_2: No experimental results — promissory note
**Attack claim**: The value proposition depends on experiments that don't exist yet.
**Verdict**: `still_unresolved`
**Evidence**: This is the nature of the document — it is a formulation document, not a results paper. The three gates (Heuristic, DOPD, Transfer) correctly define what needs to be shown, but none has results.
**Severity if unresolved**: **critical**
**Recommended fix**: Run the routing diagnostic first (Section 8.4) — before full OPD training. This is the lowest-cost experiment that validates (or invalidates) the core assumption. If the router beats heuristics on the diagnostic table, the method has initial justification.

### Point P_3: Heuristic routing might work just as well (free heuristic gate)
**Attack claim**: Even if experimental results exist, the trajectory pipeline is only justified if it clearly beats cheap heuristics. Small gains don't justify added complexity.
**Verdict**: `partially_answered`
**Evidence**: Section 10 explicitly defines the heuristic gate and the probability impact if it fails. The document is honest about this risk. The response is: "we agree, and this is the experiment we run first." But no results exist.
**Severity if unresolved**: **critical** (if no results show TCTR > heuristics)
**Recommended fix**: The routing diagnostic (Section 8.4) must include a line showing whether the difference between TCTR and the best heuristic is statistically significant. A 1-point gain with overlapping error bars is not enough.

### Point P_4: "Variational" misnomer (previously critical)
**Attack claim**: Previously: method called "variational" but isn't.
**Verdict**: `answered_by_current_text` — **RESOLVED**
**Evidence**: Title changed to "TCTR: Trajectory-Calibrated Teacher Routing." All "variational" references removed. The method is now correctly described as a routing regression problem.
**Severity if unresolved**: N/A (resolved)

### Point P_5: DOPD not cited (previously critical)
**Attack claim**: Previously: DOPD already does per-token adaptive routing in OPD.
**Verdict**: `answered_by_current_text` — **RESOLVED**
**Evidence**: DOPD is now in the baseline matrix (Section 8.2), the routing diagnostic (Section 8.4), the related work (Section 6), and the risk discussion (Section 9.4). The document also proposes TCTR + DOPD as a combined baseline. The narrative differentiation (privilege asymmetry vs capability conflict) is still not fully written but the foundation is there.
**Severity if unresolved**: N/A (resolved)

### Point P_6: Cross-dataset transfer is speculative (previously critical)
**Attack claim**: GQA → OK-VQA transfer claimed as contribution but unsupported.
**Verdict**: `still_unresolved`
**Evidence**: Same as P_2 — this is a design document without experimental results. The transfer claim exists as a hypothesis, not a result. The document acknowledges the risk (Section 9.5).
**Severity if unresolved**: **major**
**Recommended fix**: Same as P_2 — routing diagnostic first, then full OPD transfer experiment.

---

## Summary

| Point | Previous Verdict | Current Verdict | Change |
|-------|-----------------|-----------------|--------|
| P_1: DOPD preemption | `partially_answered` (critical) | ✅ `answered_by_current_text` | RESOLVED |
| P_2: VI misnomer | `still_unresolved` (critical) | ✅ `answered_by_current_text` | RESOLVED |
| P_3: 1D optimization trivial | `answered_by_current_text` | — (merged into general assessment) | RESOLVED |
| P_4: No heuristic comparison | `still_unresolved` (critical) | `partially_answered` (critical) | IMPROVED: acknowledged as gate, but no results |
| P_5: No cross-dataset evidence | `still_unresolved` (critical) | `still_unresolved` (major) | UNCHANGED |
| **New**: Method = data pipeline + simple regressor | — | `partially_answered` (major) | NEW |

| Count | Value |
|-------|-------|
| `answered_by_current_text` | 3 (P_1, P_2, P_3 from previous — all resolved) |
| `partially_answered` | 2 (P_4: heuristic gate defined but no results; P_new: pipeline framing) |
| `still_unresolved` | 1 (P_5: cross-dataset transfer) |

---

## Verdict: WARN → borderline PASS

**Reason**: The two critical `still_unresolved` points from the previous version (VI misnomer, DOPD preemption) are now resolved. What remains is a different class of risk: not "the method is wrong" but "the method has not yet been shown to work." This is a much weaker attack than the previous version faced.

The attack is survivable IF the experiments converge. The specific bar:
1. Routing diagnostic shows TCTR router beats all heuristics (entropy, disagreement, advantage-gap) by ≥3 points on high-conflict tokens.
2. Full OPD shows TCTR > fixed MOPD and ViGOS.
3. TCTR + DOPD > DOPD alone.
4. GQA → OK-VQA transfer is positive.

Without #1, the method has no justification.

---

## Top Action Items

1. **Run the routing diagnostic first** (Section 8.4). This is a 1-day experiment (no OPD training needed). If TCTR router doesn't beat heuristics on high-conflict accuracy, the project needs fundamental rethinking.
2. **Write the DOPD narrative differentiation** explicitly in the paper draft: "TCTR and DOPD solve complementary problems — privilege asymmetry vs capability conflict. Our contribution is orthogonal."
3. **Prepare the pipeline framing for the paper**: The "contribution" sentence should not be "we propose a novel algorithm" but "we demonstrate that structured visual annotations, previously used only for evaluation, can be programmatically converted into teacher routing supervision for multi-teacher OPD."
