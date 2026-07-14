# Novelty Check Report — TCTR (Revised, 2026-07-11)

**Artifact**: `vamopd_variational_mixture_formulation.md` (now titled TCTR)

---

## Proposed Method

Per-token continuous teacher routing in multi-teacher OPD for VLM visual reasoning, calibrated by programmatically generated trajectories from GQA scene graphs. A lightweight router $r_\phi(z(s))$ predicts $\lambda(s) \in [0,1]$ — the per-state mixture weight between perception teacher $\pi_P$ and reasoning teacher $\pi_R$ — trained on automatically labeled trajectory states via a soft operation prior and teacher support objective. At inference time, the router generalizes to student rollout states without requiring scene graph annotations.

---

## Core Claims

### Claim 1: Scene graph → routing trajectory pipeline
**Novelty: HIGH** — No prior work found

Programmatic conversion of GQA scene graphs into per-token teacher routing trajectories with soft operation priors. This uniquely exploits structured visual annotations that exist in benchmarks but no other distillation method uses. Search across arXiv 2024-2026 found zero papers combining scene graph programmatic trajectories with multi-teacher OPD routing.

**Key differentiator**: The trajectory is not hand-annotated or distilled from large models — it's derived from structured scene graph operations. This makes the trajectory cost essentially free.

### Claim 2: Continuous λ ∈ [0,1] routing calibrated by pseudo-optimal trajectories
**Novelty: MEDIUM** — Closest: **DOPD (2606.30626, Jun 2026)**

DOPD also does per-token adaptive routing in OPD with 4 discrete regimes. The differences:
- DOPD: teacher vs student routing (solves **privilege illusion**)
- TCTR: teacher vs teacher routing (solves **capability conflict**)
- DOPD: 4 discrete regimes based on **advantage gap heuristic**
- TCTR: continuous λ ∈ [0,1] calibrated by **trajectory + teacher support**
- These are **complementary** routing problems, not identical

DOPD's strength: no external data needed (advantage gap is on-the-fly). TCTR's strength: trajectory calibration may provide more reliable signals than heuristic classification. Which is better is an empirical question.

**Risk**: A reviewer could argue "both do per-token routing in OPD, the difference is one detail." Mitigation: frame as orthogonal problems (privilege asymmetry vs capability conflict), show TCTR + DOPD > DOPD alone.

### Claim 3: Soft operation prior from scene graph structure
**Novelty: MEDIUM-HIGH**

Using structured operation types (object / attribute / relation / logic / answer) as soft routing priors with continuous $\mu_{\text{SG}}(o_t)$ and confidence $c_{\text{SG}}(o_t)$ per operation. This is a natural way to inject benchmark-level knowledge into routing without hard labels.

**Closest**: ViGOS uses format parsing (hard binary), Decomposed OPD uses fixed priority (always visual). Soft operation prior is more flexible but the mechanism itself is straightforward.

### Claim 4: Sparse-to-dense routing generalization
**Novelty: MEDIUM-LOW**

Training router on trajectory states → deploying on student rollout states. This is a standard supervised-to-on-policy transfer problem. The consistency regularization (Section 4) is a standard technique. The novelty is in the **application setting** (teacher routing), not the mechanism.

### Claim 5: Cross-dataset routing transfer (GQA → OK-VQA)
**Novelty: MEDIUM**

If the router trained solely on GQA scene graph trajectories improves OK-VQA performance, this would demonstrate routing features (teacher disagreement, teacher support) are dataset-agnostic. This is a strong empirical finding if it holds.

---

## Closest Prior Work

| Paper | Venue | Date | Overlap | Key Difference |
|-------|-------|------|---------|----------------|
| **DOPD** (2606.30626) | arXiv | Jun 2026 | Per-token adaptive OPD routing, VLM experiments | Teacher-student routing (privilege), 4 discrete regimes, advantage-gap heuristic |
| **HEED** (2605.17093) | arXiv | May 2026 | Per-token adaptive weighting in VLM distillation | Density-based weights, single teacher, no routing |
| **HAWAII** | NeurIPS 2025 | 2025 | Router-based multi-teacher vision distillation | Discrete teacher selection, encoder only, no OPD |
| **ViGOS** (2606.19120) | CVPR 2026 | Jun 2026 | Hard perception/reasoning separation | Format-based hard {0,1}, no routing model |
| **Decomposed OPD** | ICML 2026 Spotlight | 2026 | Gradient conflict in VLM distillation | Fixed always-prioritize-visual, no per-token |
| **MoCA** | ICML 2026 Spotlight | 2025 | Modality-aware credit in VLM | Binary attribution (seeing vs thinking), not routing |
| **MOPD** (2606.30406) | arXiv | Jun 2026 | Multi-teacher OPD baseline | Fixed α, no per-token adaptation |

---

## Overall Novelty Assessment

- **Score: 6.0/10** (↑ from 5.5 in previous version — "variational" misnomer removed, DOPD now addressed)
- **Recommendation: PROCEED** — with DOPD as the primary novelty concern
- **Key differentiator**: Scene graph → trajectory pipeline is genuinely novel. No other method uses structured visual annotations for teacher routing in OPD.
- **Primary risk**: DOPD's per-token routing paradigm may be viewed as "this space has been explored." Must demonstrate trajectory-calibrated routing between **capability teachers** provides value beyond DOPD's **privilege-aware teacher-student** routing.

### Risk Matrix

| Risk | Severity | Status |
|------|----------|--------|
| DOPD overlap | **CRITICAL** | Partially mitigated: now in baseline matrix; needs clear narrative differentiation |
| "Variational" misnomer | ~~CRITICAL~~ **RESOLVED** | Removed entirely ✓ |
| Trivial 1D optimization | **MODERATE** | Document acknowledges this; router + generalization is real contribution |
| Scene graph → free data | **MODERATE** | Now framed as strength (programmatic generation) ✓ |
| Heuristic routing beats TCTR | **CRITICAL** | Acknowledged as "free heuristic gate" (Section 10); needs experimental resolution |

### Suggested Positioning

> "Existing structured visual annotations — GQA scene graphs in this work — can be programmatically converted into per-token teacher routing labels. This turns trajectory dependency from a cost into an advantage. The resulting router generalizes to student rollout states, providing continuous teacher mixtures that outperform both fixed-weight mixing and heuristic-based per-token routing."

Differentiation from DOPD:
```
DOPD: "How should the student decide whether to trust the teacher or itself?"
TCTR: "Given two capable but conflicting teachers, which one should supervise this state?"
→ Complementary. DOPD addresses privilege asymmetry; TCTR addresses capability conflict.
→ Best evidence: TCTR + DOPD > DOPD alone.
```
