# Research Review — Two-Axis Teacher Routing (DCP)

**Date**: 2026-07-21
**Reviewer**: Claude (no Codex MCP; xhigh-effort analysis simulated)
**Document**: `vds_entropy_two_axis_routing.md`
**Venue target**: CVPR 2027 / NeurIPS 2027
**Review type**: Pre-experimental idea assessment

---

## Overall Assessment

**Current score**: N/A (pre-experimental)
**Projected score if pilots pass**: 6.5-7.5/10 (borderline accept at CVPR)
**Projected score if pilots fail**: 4-5/10 (reject)

The document identifies a genuine gap in multi-teacher VLM distillation — the conflation of teacher relevance and teacher confidence into single routing signals — and proposes a clean conceptual solution. The writing is clear, the related work coverage is thorough (with notable gaps for DOPD and H-OPD), and the experimental design is well-motivated. The critical weakness is the absence of any empirical validation for the central claim that the two axes are genuinely independent.

---

## Strengths

### 1. Clear problem identification (§1)
The argument that section labels (and single-signal routing in general) conflate two distinct questions — "which teacher has unique info?" and "is that teacher confident?" — is well-articulated and convincing. The table in §1.2 showing how each existing approach misses one axis is an effective rhetorical device.

### 2. Strong positioning against DecomposedOPD (§2)
The comparison between DecomposedOPD's VDS ("does this token need the image?") and DCP's teacher-teacher KL ("where does π_P's visual specialization add value?") is the strongest technical argument in the document. The insight that the right relevance metric changes fundamentally between single-teacher and multi-teacher settings is non-obvious and valuable.

### 3. Computational simplicity (§5.5)
DCP genuinely requires zero extra forward passes, zero learned parameters, and one backward. This is a real practical advantage over learned routers and gradient-space methods. The cost table in §5.5 makes this case clearly.

### 4. Honest risk assessment (§9)
The self-identified risks are correctly prioritized and honestly stated. The paper-ending scenarios in §9.3 show the authors understand what would falsify their claims — a sign of intellectual rigor that reviewers appreciate.

### 5. Well-designed pilots (§11)
The three pilot experiments directly target the highest-risk claims: (1) axis independence, (2) teacher differentiation, (3) head-to-head comparison. This is exactly the right experimental strategy.

---

## Weaknesses

### Critical

**W1: No empirical evidence for the central claim (§3, §9.1).** The entire framework rests on the assumption that D_t and teacher entropy are sufficiently independent to justify two axes. If they're correlated at r > 0.7, the paper as currently framed cannot exist. This is self-acknowledged but unresolved. At least a small-scale diagnostic on public VLM checkpoints would substantially strengthen the document.

### Major

**W2: ViGOS is the most important baseline but is under-discussed.** The document's §7 mentions ViGOS only in passing (it appears in §7.5's comparison table as "ViGOS-style hard separation"). This is backwards — ViGOS should be the primary comparison point. ViGOS proved that perception-reasoning separation matters; DCP claims to be the strictly better implementation of the same insight. The related work should lead with ViGOS, and the experiments should include ViGOS-style section assignment as the most important baseline.

**W3: CoT format irrelevance claim is overstated (§6).** The logical argument is sound but the empirical claim ("Mixed CoT works natively") is unsupported. The document would be stronger claiming "DCP *enables* Mixed CoT" rather than "CoT format is irrelevant." The distinction matters: the former is a practical benefit of DCP; the latter is a stronger claim that requires proof.

**W4: Teacher construction is hand-waved (§9.2 Q2).** The document says teachers are "fine-tuned with RL on vision-heavy tasks" vs "reasoning-heavy tasks." This is vague. What specific datasets? What RL objectives? How different are the resulting teachers? The teacher differentiation check (Pilot 2) addresses whether teachers are different enough, but not how to make them different if they aren't.

### Minor

**W5: The τ hyperparameter (§3.1).** Default τ = 0.5 nats is proposed with "clear semantic interpretation." But the sensitivity to τ is not discussed beyond a note in §9.1. For a zero-parameter method, having one parameter that needs tuning somewhat undermines the "zero-parameter" claim. Either provide a principled way to set τ, or show that results are insensitive to τ across a reasonable range.

**W6: Extension to K > 2 teachers (§9.2 Q4).** The document dismisses this as "conceptually straightforward" but left for future work. For a general framework claim, this is a limitation. At minimum, sketch the extension explicitly (pairwise divergence matrix + confidence vector → softmax over teachers) rather than hand-waving.

**W7: No discussion of inference-time behavior.** DCP is a training-time routing mechanism. At inference, the student operates alone without teachers. The document should discuss whether the routing pattern learned during training transfers to better student behavior at inference, or whether there's a train-test mismatch.

---

## Claims Audit

| Claim | Location | Type | Verifiability | Status |
|-------|----------|------|---------------|--------|
| Single-signal routing conflates two questions | §1.2 | Conceptual | Medium (requires showing that divergence and entropy are empirically distinct) | Unvalidated |
| KL(π_P ‖ π_R) captures specialization gap | §2.3 | Technical | High (can be tested via teacher ablation) | Unvalidated |
| DCP outperforms section assignment | §8.2 | Empirical | High (direct comparison) | Not run |
| CoT format is irrelevant under DCP | §6 | Empirical | High (compare Mixed vs Staged CoT) | Not run |
| Product form > sum > max | §8.2 | Empirical | High (ablation) | Not run |
| DCP approaches learned router performance | §8.2 | Empirical | High (comparison) | Not run |
| Four-regime analysis reveals patterns missed by single-axis | §8.3 | Analytical + Empirical | Medium | Not run |

---

## Experimental Design Review

### What's well-designed

- **Pilot 1** (axis correlation check) is the right first experiment and would gate further work
- **Four-regime analysis** (§8.3) is a strong diagnostic that could become a key figure
- **Section label vs divergence misalignment plot** (§8.3) is a compelling visualization idea
- **Full baseline matrix** (§8.4, 12 methods) is comprehensive

### What's missing

- **Teacher quality baseline**: Single-teacher OPD with each teacher alone tells us whether multi-teacher is even needed. If visual-teacher-only OPD matches DCP, the routing problem is moot.
- **Oracle/upper-bound routing**: What if we had perfect token-level teacher labels? This establishes the ceiling for routing methods.
- **Statistical significance**: 47K training samples — how many runs per method? Error bars?

---

## Competitive Landscape Assessment

| Method | Setting | Routing Signal | Granularity | CoT-aware | Status |
|--------|---------|---------------|-------------|-----------|--------|
| **ViGOS** | Single model, dual input modes | Section labels (hard) | Section-level | Yes (requires Staged CoT) | arXiv Jun 2026 |
| DecomposedOPD | Single-teacher strong→weak | VDS (± image) | Token-level | No | ICML 2026 Spotlight |
| DOPD | Single-teacher privileged | Advantage gap × confidence | Token-level | No | arXiv Jun 2026 |
| H-OPD | Multi-teacher heterogeneous | Confidence + candidate scoring | Token-level | No | arXiv Jul 2026 |
| TCTR | Multi-teacher | Learned router | Token-level | Yes | Idea |
| **DCP** | **Multi-teacher specialist** | **Divergence × confidence** | **Token-level** | **Yes (format-agnostic)** | **Idea** |

**DCP's unique positioning**: The natural generalization of ViGOS from section-level hard routing to token-level continuous routing. DCP is the only method that (a) explicitly argues two-axis decomposition is necessary for multi-teacher routing, (b) operates at token-level without learned parameters, and (c) makes CoT format a free choice rather than an architectural constraint. ViGOS is the primary baseline; DOPD and H-OPD address different problem settings.

---

## Venue Recommendation

| Venue | Fit | Reasoning |
|-------|-----|-----------|
| **CVPR 2027** | Best fit | VLM distillation + visual reasoning theme. Two-axis framework is visual enough for CVPR audience. |
| NeurIPS 2027 | Good fit | Method + analysis. Right level of contribution if experiments are strong. |
| ICML 2027 | Stretch | ICML prefers deeper theory; DCP is more empirical/conceptual. |
| ICLR 2028 | Possible | If the two-axis insight generalizes beyond VLMs. |

---

## Recommended Next Steps (Priority Order)

1. **Run Pilot 1 this week.** This is the gate. If axes are correlated, the paper needs fundamental restructuring.
2. **Run Pilot 3 at small scale (50-100 samples).** Include ViGOS-style section assignment as the primary baseline. Even noisy results on DCP vs ViGOS-style vs uniform MOPD would transform the document from "idea" to "preliminary evidence."
3. **Restructure §7 to lead with ViGOS.** Frame as: ViGOS proved separation is valuable; DCP removes the format straitjacket and adds the confidence axis.
4. **Soften the CoT format claim.** Change from "CoT format is irrelevant" to "DCP enables format-agnostic routing" until experiments support the stronger claim.
5. **Specify teacher construction.** Even if not yet executed, provide concrete dataset/objective choices so reviewers can assess feasibility.

---

## Bottom Line

The idea is sound and well-argued. DCP positions naturally as ViGOS's generalization from hard section-level routing to continuous token-level routing. The document correctly identifies its own risks. The three pilots are the right next experiments. The competitive landscape is surprisingly open — MOPD is only ~2 months old as a concept, and no existing work addresses same-modality multi-teacher token-level specialization routing. The main risk is empirical (axis correlation, Pilot 1), not competitive.

---

*Note: This review was performed on a pre-experimental idea document. No Codex MCP cross-model review was available. All analysis by Claude.*
