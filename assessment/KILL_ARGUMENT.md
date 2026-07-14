# Kill Argument Report — Capability-Decoupled Supervision for Multi-Teacher VLM Distillation

**Date:** 2026-07-03
**Reviewer:** Manual adversarial analysis (Codex MCP unavailable — degraded execution)
**Verdict:** **WARN** (`reason_code: partial_critical_or_repeated_major`)
**Context:** Claims-stage analysis with pilot experiment data (GO_NO_GO_REPORT.md) and 2026-07-03 WebSearch results

---

## Net Assessment

This claims-stage analysis identifies **two critical unresolved issues** and **four partially-answered major issues**. The most damaging attack is the combination of (1) **Drive-KD + IGA jointly preempting the gradient conflict discovery claim** — Drive-KD already found cross-capability gradient conflicts specifically in VLM multi-teacher distillation, and IGA already built the gradient conflict analytic toolkit — and (2) **experimental evidence at pilot scale on only 2 benchmarks** being insufficient to establish either the generality of the phenomenon or the necessity of all three method components.

The "dual problem" framing (gradient conflict + style dominance are independent but compounding) remains the strongest defense. But the prior-work picture now includes BOTH a text-only gradient conflict paper (IGA) AND a VLM-specific multi-teacher capability gradient conflict paper (Drive-KD). The authors must now differentiate against two prior findings, not one.

The method's novelty is better defended than the discovery claim: the specific combination of capability-segment routing + segment-level preference + multi-teacher judging is genuinely absent from prior work, even if each component is individually precedented. The key question is whether ablation experiments will show all three components are necessary.

---

## Attack Memo (Adversarial Rejection Paragraph)

> The paper claims to discover that multi-teacher token-level distillation in dense VLMs causes gradient interference between teachers with different capabilities. This is not a new discovery. Drive-KD (Lian et al., 2025) already identified "cross-capability gradient conflicts" in multi-teacher VLM distillation, decomposing autonomous driving into perception-reasoning-planning and proposing Asymmetric Gradient Projection to resolve them. Invariant Gradient Alignment (IGA, Jun 2026) further provided the analytic toolkit — per-dimension gradient variance, SVD projection, conflict masks — for detecting and suppressing gradient conflicts in distillation. Replacing "perception-reasoning-planning" with "visual-knowledge" and calling it a discovery is a domain shift, not a contribution. If there is something fundamentally different about visual-knowledge gradient interference compared to perception-reasoning-planning interference, the paper does not establish it.
>
> The method combines three established ideas: capability decomposition (Drive-KD), segment-level DPO (fDPO, NeurIPS 2025), and preference optimization replacing KL (StepOPSD, May 2026). Each component has clear precedent. The combination is the only new element, but "A+B+C" requires strong ablation evidence that each component is individually necessary and that the combination exceeds the sum of its parts. No such evidence exists.
>
> The experimental evidence is preliminary almost to the point of anecdotal: two benchmarks (A-OKVQA, V*Bench) at pilot scale. The headline finding — that content-token gradient cosine drops from ~1.0 (same-teacher) to significantly lower with negative values (cross-teacher) — could reflect idiosyncratic properties of the specific teacher-student pair rather than a general phenomenon. Similarly, the ">50% KL from non-content tokens" figure may depend on specific prompt templates or answer distributions. Without replication across model families, dataset scales, and teacher configurations, these numbers cannot support the breadth of claims the paper makes.
>
> Finally, the capability segment decomposition assumes answers can be cleanly partitioned into visual, knowledge, and reasoning spans. In practice, many tokens serve multiple capabilities simultaneously. The paper provides no mechanism for handling mixed-capability tokens and no measurement of what fraction of tokens are cleanly classifiable.

---

## Adjudication (Per-Point)

### Point P_1: Drive-KD Already Found Cross-Capability Gradient Conflict in VLM Multi-Teacher KD
**Attack claim:** Drive-KD (2025) already identified cross-capability gradient conflicts and proposed AGP. The visual-knowledge axis is a domain shift, not a new discovery.
**Verdict:** `partially_answered`
**Evidence:** The authors distinguish their work from Drive-KD on two axes: (a) capability taxonomy — visual/knowledge vs. perception/reasoning/planning, and (b) domain — general VQA vs. autonomous driving. However, both papers study "multi-teacher VLM distillation where teachers with different capabilities produce conflicting gradients on shared parameters." The *phenomenon* is the same. The question is whether the visual-knowledge conflict has qualitatively different characteristics (e.g., different token-level distribution, different interaction with answer correctness, different mitigation requirements) than Drive-KD's perception-reasoning-planning conflict. The current framing does not establish this.
**Severity if unresolved:** `critical`
**Recommended fix:** (1) Cite Drive-KD explicitly and distinguish the conflict *type* with specificity: "Drive-KD studies sequential pipeline conflicts where one capability's output feeds the next; we study parallel capability conflicts where visual and knowledge signals compete on the same output token." (2) If possible, measure gradient conflict patterns using both Drive-KD's taxonomy AND the proposed taxonomy on the same data, showing the conflict *structure* differs.

### Point P_2: IGA Already Built the Gradient Conflict Detection Toolkit
**Attack claim:** IGA (Jun 2026) provides per-dimension gradient variance, SVD projection, and conflict masks — the same analytic approach. The extension from cross-domain to cross-capability is incremental.
**Verdict:** `partially_answered`
**Evidence:** The v7 analysis already addressed this: IGA is (a) text-only, (b) same-capability cross-domain (math→medicine→law reasoning), (c) uses gradient masking rather than decoupling. The token-level capability-mixing analysis is genuinely absent from IGA. However, a reviewer will note that IGA's methodology could be applied to any distillation setting — the analytic toolkit transfers. The burden is on the authors to show that capability decoupling is *better* than gradient masking, not just different.
**Severity if unresolved:** `major`
**Recommended fix:** Add IGA-at-VLM as a baseline in experiments. If capability decoupling outperforms gradient masking, the "different and better" claim is empirically supported. If not, the contribution is limited to "we applied IGA's method to VLM" — which is not sufficient.

### Point P_3: Experimental Evidence Is Pilot-Scale, Cannot Support General Claims
**Attack claim:** Two benchmarks at pilot scale. Gradient cosine drop and >50% KL-from-non-content may be idiosyncratic. No cross-model, cross-dataset, or statistical validation.
**Verdict:** `still_unresolved`
**Evidence:** The claims explicitly acknowledge "pilot 实验" but make broad statements ("Dense VLM 的 multi-teacher token-level distillation 存在直接梯度干扰") that imply generality. This is the single most damaging attack because it undermines ALL claims simultaneously. If the phenomenon doesn't replicate, neither the discovery nor the method matters.
**Severity if unresolved:** `critical`
**Recommended fix:** Expand to minimum 4-5 benchmarks (add OK-VQA, TextVQA, VizWiz, InfoSeek at minimum). Test 2+ model families. Report bootstrap CIs on gradient cosine differences. Until then, scope all claims with "preliminary evidence suggests" and explicitly list generalization as a limitation.

### Point P_4: Method Is A+B+C — Ablation Evidence Missing
**Attack claim:** Capability decomposition (Drive-KD), segment-level DPO (fDPO), and preference-over-KL (StepOPSD) are each individually precedented. The combination requires ablation evidence.
**Verdict:** `partially_answered`
**Evidence:** The dual-problem framing provides a conceptual defense: each component targets a different problem (routing → gradient conflict, preference → style dominance), so all are necessary. This is logically coherent but requires empirical validation. No ablation experiments are reported. If any single component could be removed without significant degradation, the "all necessary" claim fails.
**Severity if unresolved:** `major`
**Recommended fix:** Run and report: (1) CCD without capability routing (all teachers supervise all segments via preference), (2) CCD with segment-level KL instead of preference, (3) CCD with single-teacher segment preference (fDPO-style). Each ablation isolates one component. If any degradation is small (<2%), that component is not load-bearing.

### Point P_5: Capability Segment Boundaries Are Ill-Defined — Mixed Tokens Unhandled
**Attack claim:** Many tokens serve multiple capabilities simultaneously. The method assumes clean segmentability. No mixed-token handling mechanism exists.
**Verdict:** `still_unresolved`
**Evidence:** This attack was not present in the v7 analysis but emerged from the current adversarial review. The method's core operation — routing teachers to capability segments — depends on the assumption that segments are cleanly separable. If a significant fraction of tokens are genuinely mixed (e.g., "the red car" requires visual recognition + world knowledge that this object is a car), hard segment boundaries will misroute teacher supervision.
**Severity if unresolved:** `major`
**Recommended fix:** (1) Measure what fraction of answer tokens in A-OKVQA/V*Bench are unambiguously classifiable into a single capability. (2) If >20% are mixed, propose a soft-boundary mechanism (teacher weight decays near segment boundaries) or overlapping segments. (3) If >50% are mixed, the method's premise is fundamentally challenged — this must be acknowledged.

### Point P_6: Non-Content Token KL Claim May Not Generalize
**Attack claim:** The >50% figure could depend on specific prompt templates, answer length distributions, or teacher-student pairs. Sensitivity analysis is missing.
**Verdict:** `partially_answered`
**Evidence:** The measurement methodology (token classification → KL contribution decomposition) is sound in principle. But a single number without sensitivity analysis is fragile — if the number drops to 35% with a different prompt template, the "style dominance" narrative weakens substantially.
**Severity if unresolved:** `major`
**Recommended fix:** Report how the non-content KL fraction varies with: (a) different prompt templates (short/long/chat-style), (b) short vs. long ground-truth answers, (c) different teacher-student pairs. If the fraction is stable (e.g., 45-55% across conditions), the claim is robust. If it varies widely, the narrative must be more nuanced.

### Point P_7: Negative Gradient Cosine May Be Noise, Not Interference
**Attack claim:** Negative cosine values could arise from stochastic gradient noise in small-scale experiments rather than systematic teacher conflict.
**Verdict:** `partially_answered`
**Evidence:** The same-teacher baseline (cosine ~1.0) provides a strong control — if noise were the cause, same-teacher gradients would also show substantial variance. However, statistical testing is needed to rule out the possibility that cross-teacher negative cosines reflect a small number of outlier batches rather than a systematic pattern.
**Severity if unresolved:** `minor`
**Recommended fix:** Report bootstrap confidence intervals on gradient cosine distributions. Test statistical significance of same-teacher vs. cross-teacher cosine difference. If the negative cosine tail is statistically significant and replicable across seeds, the noise explanation is ruled out.

---

## Summary

| | Count |
|---|---|
| Total rejection points | 7 |
| `answered_by_current_text` | 0 |
| `partially_answered` | 4 (P_1, P_2, P_4, P_6, P_7) |
| `still_unresolved` | 2 (P_3: pilot scale; P_5: mixed tokens) |

**Verdict: WARN** — Two critical-severity unresolved issues:
- **P_3 (critical):** Experimental evidence is pilot-scale. Cannot support generality claims.
- **P_5 (major):** Capability segment boundaries may not be cleanly definable. No mixed-token handling.

Two critical/major partially-answered issues:
- **P_1 (critical):** Drive-KD preemption of gradient conflict discovery
- **P_2 (major):** IGA preemption of gradient conflict methodology

**Verdict compared to v7:** Risk is **slightly higher** than v7's assessment, primarily due to the addition of Drive-KD as a VLM-specific prior work that v7's attack memo did not account for. The "domain shift" defense that worked against IGA alone (text-only → VLM) does not work against Drive-KD (already VLM).

---

## Top Action Items (Priority Order)

1. **【BLOCKING — P_3】Scale experiments before submitting:** 2 benchmarks → 4-5 + 2 model families + statistical tests. This gates everything else. Without this, even a perfectly framed paper cannot withstand review.
2. **【BLOCKING — P_1+P_2】Address Drive-KD + IGA preemption:** Add both to related work with explicit differentiation. Add IGA-at-VLM and Drive-KD-AGP-at-VLM as baselines. Show that capability decoupling outperforms both gradient masking (IGA) and gradient projection (Drive-KD).
3. **【MAJOR — P_5】Address mixed-capability tokens:** Measure clean-vs-mixed token fraction. If mixed fraction is significant, propose soft-boundary mechanism or acknowledge as limitation.
4. **【MAJOR — P_4】Run ablation experiments:** Demonstrate that capability routing, segment-level preference, and multi-teacher design are each individually necessary. Without this, the method is "A+B+C."
5. **【MINOR — P_6+P_7】Sensitivity + statistical analysis:** Bootstrap CIs on gradient cosine; sensitivity analysis on non-content KL fraction across prompt templates and model pairs.

---

## Recommendation

**Go signal confirmed — proceed to full experiments.**

The pilot experiments (content cosine 0.327, negative rate 30.4%, non-content KL 54.0%) confirm both problems exist at measurable scale. The kill-argument exercise surfaces the key vulnerabilities that full experiments must address:

1. **Scale experiments** (4-5 benchmarks, 2 model families, statistical testing). Pilot data is strong but on 120 samples. Must demonstrate generality.
2. **Add Drive-KD and IGA baselines.** If capability decoupling empirically beats gradient projection (Drive-KD) and gradient masking (IGA), the "different and better" claim is validated.
3. **Add ablation experiments.** If all three components (capability routing, teacher isolation, segment preference) are individually necessary, the "A+B+C" criticism is neutralized.

If these three conditions are met, the paper moves from WARN to PASS.

---

## Review Tracing

- **Codex MCP:** Not called (unavailable). Manual adversarial analysis performed with 2026-07-03 WebSearch results and GO_NO_GO_REPORT.md pilot data.
- **Attack methodology:** Single-commitment rejection paragraph (~240 words) targeting four axes: (1) Drive-KD + IGA preemption of gradient conflict discovery, (2) A+B+C building-block combination, (3) insufficient experimental scale, (4) mixed-capability token ambiguity.
- **Adjudication:** 7 atomic rejection points, each with specific evidence assessment and actionable fixes.
