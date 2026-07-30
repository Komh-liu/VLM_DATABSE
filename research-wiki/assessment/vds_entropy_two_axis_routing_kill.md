# Kill Argument Report — Two-Axis Teacher Routing (DCP)

**Date**: 2026-07-21
**Reviewer model**: Claude (no Codex MCP available — cross-model adjudication simulated)
**Document type**: Markdown idea document (not LaTeX paper)
**Verdict**: WARN (`reason_code: partial_critical_or_repeated_major` — downgraded from FAIL; DOPD/H-OPD are not direct competitors, but 2 unresolved points remain)

---

## Net Assessment

The paper identifies a real problem — single-signal teacher routing is lossy — but the proposed solution does not rise above a heuristic recombination of standard metrics. The two-axis framework is conceptually clean but empirically unvalidated. Without pilot data showing that (a) the two axes are genuinely independent, (b) the product form outperforms both single-axis ablations and the existing baselines, and (c) the CoT format irrelevance claim holds empirically, the contribution is a well-argued conjecture rather than a completed research result. The document itself acknowledges this risk (§9.1: "D_t and entropy are correlated — product adds no info") but does not resolve it. DOPD and H-OPD — while structurally similar at surface level (four-regime routing, confidence term) — address fundamentally different problems (single-teacher privileged distillation and heterogeneous modality bridging, respectively) and are not direct competitors.

---

## Attack Memo (verbatim)

> The central claim of this paper — that teacher-teacher KL divergence and per-teacher entropy are "orthogonal axes" that must be decoupled for multi-teacher VLM routing — is a conceptual reframing dressed as a technical contribution. Both KL divergence and entropy are textbook metrics; multiplying them together and calling the result a "framework" does not constitute a novel method. The paper acknowledges (in §9.1) that the two axes may be highly correlated, which would collapse the entire two-axis premise to a single effective dimension. This is not a minor risk — it is an existential threat to the paper's core narrative, and it remains completely unresolved.
>
> Worse, the claim that this makes "CoT format irrelevant" is an overstatement. Replacing section labels with a continuous heuristic does not prove irrelevance; it merely replaces one routing signal with another. The CoT format was never claimed to be optimal — only practical. The paper offers no evidence that Mixed CoT + DCP outperforms Staged CoT + section assignment.
>
> Finally, the paper's positioning against DecomposedOPD is its strongest suit, but this alone does not carry the contribution. The DecomposedOPD comparison shows that the right relevance metric changes between single-teacher and multi-teacher settings — a real insight. But the paper still needs to demonstrate that this insight leads to empirically better routing decisions than simpler alternatives like entropy ratio or uniform MOPD. Without that demonstration, the paper is an interesting thought experiment, not a completed research contribution.

---

## Adjudication (per-point)

### Point P_1: Two-axis decomposition is a conceptual reframing, not a technical contribution
**Attack claim**: KL and entropy are standard metrics; multiplying them is not a method.
**Verdict**: partially_answered
**Evidence**: §2-3 provide a detailed argument for *why* these two signals answer different questions and why prior work conflates them. The DecomposedOPD comparison in §2.1-2.2 is the strongest part — it shows that the right relevance metric changes fundamentally between single-teacher (VDS) and multi-teacher (teacher-teacher KL) settings. However, the document does not demonstrate that this conceptual insight leads to empirically different routing decisions.
**Severity if unresolved**: major
**Recommended fix**: Run Pilot 1 immediately (§11). If D_t and H̄ are empirically uncorrelated, the conceptual reframing is validated. If r > 0.7, reframe as "a better single-axis signal" rather than "two orthogonal axes."

### Point P_2: Two-axis independence is unproven and may not hold
**Attack claim**: The document admits the axes may be highly correlated (§9.1), which would collapse the framework.
**Verdict**: still_unresolved
**Evidence**: §9.1 explicitly states this risk and notes correlation r > 0.7 would make the confidence axis redundant. No data is provided. The document correctly identifies this as a "High" severity risk. Until Pilot 1 is run, this remains the single most damaging uncertainty.
**Severity if unresolved**: critical
**Recommended fix**: This is the highest-priority experiment. Compute D_t, H̄_P, H̄_R on 100+ student rollouts. Report correlation matrix. If r < 0.5, the two-axis claim is empirically supported. If 0.5 < r < 0.7, acknowledge partial redundancy but argue the residual is meaningful. If r > 0.7, pivot the paper.

### Point P_3: CoT format irrelevance is an overstatement
**Attack claim**: Replacing section labels with a heuristic does not prove format irrelevance; no comparison data exists.
**Verdict**: still_unresolved
**Evidence**: §6 argues that "under DCP, CoT format is irrelevant" but provides no empirical support. The claim is logically consistent — if routing comes from teacher behavior not text structure, format shouldn't matter — but this is a prediction, not a finding. The ICLR 2025 "To CoT or Not to CoT" paper provides partial support (token permutations preserve gains) but in a different setting.
**Severity if unresolved**: major
**Recommended fix**: Run Pilot 3: Mixed CoT + DCP vs Staged CoT + section assignment on 50+ samples. If DCP closes the gap or exceeds, the claim is supported. Frame as "DCP enables Mixed CoT without accuracy loss" rather than "CoT format is irrelevant."

### Point P_4: DOPD and H-OPD may be cited by reviewers as prior work, requiring explicit differentiation
**Attack claim**: DOPD and H-OPD both use four-regime token routing with confidence, potentially creating a perception of being overtaken.
**Verdict**: partially_answered
**Evidence**: The document's §7 covers DecomposedOPD and learned routers but does not cite DOPD or H-OPD. However, upon closer inspection, DOPD addresses single-teacher privileged distillation (routes distillation strength, not teacher selection) and H-OPD addresses heterogeneous modality bridging (VLM+LLM with vision-to-text transfer, not same-modality specialization routing). These are fundamentally different problem settings. The risk is not actual overlap but reviewer perception — a reviewer seeing "four-regime + confidence" may incorrectly assume DCP is derivative unless the paper preemptively differentiates.
**Severity if unresolved**: minor (not major — problem settings are genuinely different)
**Recommended fix**: Add a paragraph to §7 explicitly differentiating from DOPD and H-OPD along problem setting: DOPD = single-teacher privileged, H-OPD = heterogeneous modality bridging, DCP = homogeneous multi-teacher specialization routing. The four-regime 2×2 table is a natural analytical tool common to all three, not evidence of method overlap.

### Point P_5: Zero empirical validation
**Attack claim**: The document describes experiments in future tense but provides no results.
**Verdict**: still_unresolved
**Evidence**: §8 and §11 describe experiments to be run, not experiments completed. The document is explicitly pre-experimental. This is expected for an idea document, but it means the core claims (two-axis independence, DCP > section assignment, CoT format irrelevance) are unvalidated.
**Severity if unresolved**: critical (for paper submission; expected for idea document stage)
**Recommended fix**: Run the three pilots in §11 before drafting the paper. The pilots are well-designed and targeted at the highest-risk claims.

---

## Summary

| | Count |
|---|---|
| Total rejection points | 5 |
| answered_by_current_text | 0 |
| partially_answered | 2 |
| still_unresolved | 3 |

---

## Top Action Items (priority order)

1. **Run Pilot 1 immediately**: Compute D_t, H̄_P, H̄_R on 100+ rollouts. Report correlation. This gates the entire two-axis claim. If r > 0.7, pivot to single-axis framing.
2. **Run Pilot 3**: Head-to-head comparison of DCP vs section assignment vs uniform MOPD on a small scale. This validates or refutes the CoT format irrelevance claim.
3. **Add DOPD and H-OPD to related work** (§7): Not because they are direct competitors (they aren't — different problem settings), but to preempt reviewer confusion. Explicitly differentiate along problem setting: DOPD = single-teacher privileged, H-OPD = heterogeneous modality bridging, DCP = homogeneous multi-teacher specialization routing.

---

## Recommendation

The idea is well-argued and the problem is real. The document's own risk assessment (§9) is honest and correctly identifies the failure modes. The three pilots in §11 are exactly the right next steps. **DCP positions naturally as ViGOS's generalization** — ViGOS proved perception-reasoning separation matters, DCP removes the format straitjacket and adds token-level granularity + the confidence axis. The competitive landscape is surprisingly open: MOPD is ~2 months old, and no existing work addresses same-modality multi-teacher token-level specialization routing. **Do not write the paper until Pilot 1 passes.** If r(D_t, H̄) > 0.7, the two-axis claim collapses and the paper needs to pivot to "teacher divergence is a better single-axis signal than section labels" — a weaker but still viable contribution. If r < 0.5 and DCP outperforms ViGOS-style baselines in Pilot 3, the paper has a real shot.

---

*Note: This kill-argument was performed on a pre-experimental idea document. The "still_unresolved" findings are expected at this stage and should be read as experiment priorities, not paper flaws. Cross-model Codex adjudication was not available; analysis performed by Claude.*
