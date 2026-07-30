# Novelty Check Report — Two-Axis Teacher Routing (DCP)

**Date**: 2026-07-21
**Method**: Multi-source literature search (arXiv, Semantic Scholar, Google Scholar, top venue proceedings)
**Cross-model verification**: Not available (no Codex MCP); analysis performed by Claude based on search results

---

## Proposed Method

Two-axis teacher routing (DCP: Divergence-Confidence Product) for multi-teacher VLM on-policy distillation. Decouples token-level teacher assignment into two orthogonal signals — teacher-teacher KL divergence (relevance: "which teacher has unique information?") and per-teacher entropy (confidence: "is that teacher reliable?") — then combines them via product to produce per-token routing weights, at zero extra forward passes and zero learned parameters.

---

## Core Claims

### Claim 1: Two-axis decomposition is necessary for multi-teacher routing
> "No single scalar can simultaneously capture which teacher has unique information AND whether that teacher is confident."
- **Novelty**: MEDIUM-HIGH
- **Closest**: H-OPD (2607.02592) does multi-teacher token-level arbitration with confidence, but its routing problem is modality bridging (VLM + text-only LLM), not specialization routing (two same-modality VLMs with different RL specializations). The explicit factorization into two named orthogonal axes with the argument that they answer "different questions" is not present in any prior work.
- **Delta**: The four-regime 2×2 partition is a natural analytical tool (DOPD also uses one), but DCP is the only work that (a) explicitly argues single-signal routing is fundamentally lossy because it conflates two questions, (b) proposes a specific product-form solution, and (c) connects routing to CoT format choice. The conceptual argument is the contribution, not the 2×2 table.

### Claim 2: KL(π_P ‖ π_R) is the right relevance metric for multi-teacher routing
> "Teacher-teacher divergence... isolates the specialization gap"
- **Novelty**: MEDIUM
- **Closest**: DecomposedOPD's VDS measures within-model ± image divergence. UniKD, EWAD use teacher-teacher KL as a generic disagreement signal. The novel part is the *argument for why* KL(π_P ‖ π_R) is the right metric specifically for multi-teacher specialist routing: shared base VLM → language priors near-identical → residual KL isolates RL specialization gap. This justification is new even though the metric is standard.
- **Delta**: The DecomposedOPD comparison (§2.1-2.2) showing that the right relevance metric fundamentally changes between single-teacher (VDS) and multi-teacher (teacher-teacher KL) settings is a non-obvious insight.

### Claim 3: Zero-parameter, zero-overhead routing (product form)
> "One forward, one backward. Zero extra passes. Zero learned parameters. One hyperparameter."
- **Novelty**: LOW-MEDIUM
- **Closest**: DE-MKD (2024) uses entropy-weighted multi-teacher aggregation (sample-level). Section-based assignment is also zero-parameter. The routing is computed, not learned — many prior methods do this (entropy ratio, disagreement ratio, etc.).
- **Delta**: The specific product form D_t·(1-H̄_P) is a new combination, but the individual components are standard. Risk of being seen as "yet another heuristic combination."

### Claim 4: CoT format becomes irrelevant under two-axis routing
> "Under DCP routing, Mixed CoT works natively... Staged CoT's section labels become dead weight."
- **Novelty**: MEDIUM
- **Closest**: "To CoT or Not to CoT?" (ICLR 2025) shows CoT format/ordering is irrelevant for distillation gains — token permutations preserve improvements. But that's for single-teacher text distillation, not multi-teacher VLM routing.
- **Delta**: The claim that section labels are *strictly worse* (not just unnecessary) because they lose the confidence axis is a stronger, more specific claim than prior work. However, this is currently unsupported by experiments.

### Claim 5: Four-regime token analysis taxonomy
> "Regime I (High D_t, Low H_P): Strong visual... Regime II (High D_t, High H_P): Weak visual..."
- **Novelty**: LOW-MEDIUM (analytical contribution)
- **Closest**: DOPD also has a four-regime table (high/low advantage × high/low confidence), but the axes mean different things. TA-OPD partitions tokens by teachability.
- **Delta**: The four-regime labels are domain-specific to VLM perception/reasoning specialization (not privilege illusion). Unique contribution is showing that Regime II tokens (high divergence + low visual confidence) are invisible to both section labels and single-axis routing — a concrete, testable prediction.

---

## Closest Prior Work

| Paper | Year | Venue | Overlap | Key Difference |
|-------|------|-------|---------|----------------|
| **ViGOS** (2606.19120) | 2026 | arXiv | Perception-reasoning separation, staged CoT | Hard section-level routing; DCP is its natural generalization to continuous token-level routing |
| **DecomposedOPD** | 2026 | ICML Spotlight | Token-level signal decomposition | Single-teacher strong→weak distillation; different problem setting |
| **H-OPD** (2607.02592) | 2026 | arXiv | Multi-teacher, token-level, confidence-aware | Heterogeneous teachers (VLM+LLM), modality bridging; not specialization routing |
| **DOPD** (2606.30626) | 2026 | arXiv | Four-regime token routing | Single-teacher privileged; routes distillation strength not teacher selection |
| **TA-OPD** (2605.26844) | 2026 | arXiv | Token teachability | Single-teacher; "which tokens are learnable" not "which teacher" |

---

## Overall Novelty Assessment

- **Score**: 6.5/10
- **Recommendation**: PROCEED
- **Key differentiator**: The *explicit decoupling argument* ("one signal cannot answer two questions") is the strongest conceptual contribution. No prior work makes this specific argument for multi-teacher VLM routing with same-modality specialist teachers. DOPD (single-teacher privileged) and H-OPD (heterogeneous modality bridging) address fundamentally different problems, so they are not direct competitors.
- **Risk**: 
  1. **The two-axis claim may collapse empirically.** If D_t and H̄ are highly correlated (content words → high divergence + low entropy; function words → low divergence + medium entropy), the "two-axis" story reduces to one effective dimension. The pilot experiment checking this correlation is existentially important.
  2. **"Zero-parameter heuristic" is a hard sell at top venues** unless it substantially outperforms learned routers or reveals surprising insights. The contribution needs to be framed as *understanding* (why two axes matter) rather than *engineering* (a new routing formula).
  3. **Reviewers may incorrectly lump DCP with DOPD/H-OPD** due to superficial structural similarities (four-regime table, confidence term). The paper must preemptively differentiate along problem setting, not just method.

### Suggested Positioning

- **Position DCP as ViGOS's natural generalization.** ViGOS proved that perception and reasoning benefit from different supervision, but used the crudest possible routing mechanism (hard section labels). DCP shows that the same insight, implemented at token-level with continuous teacher-behavior-driven routing, is both simpler (no format dependency) and more effective (confidence-gated, handles Regime II/IV).
- **Frame as an analysis/understanding paper, not a method paper.** The core contribution is the *argument* that section labels conflate two orthogonal questions. DCP is the natural consequence of this insight, not the primary contribution.
- **Lead with the four-regime diagnostics.** Show that Regime II tokens (high divergence, low visual confidence) are invisible to both section labels and single-axis routing, and that this class of tokens is both common and important.
- **Emphasize the CoT format irrelevance result** — if experiments show DCP + Mixed CoT ≥ ViGOS-style section assignment + Staged CoT, that's a practically useful finding even if the routing formula is simple.

---

*Search conducted 2026-07-21. Cross-model verification via Codex MCP not available — all analysis performed by Claude based on web search results.*
