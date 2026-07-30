# Two-Axis Teacher Routing: Divergence × Confidence for Format-Agnostic Multi-Teacher VLM Distillation

> CoT section labels conflate two orthogonal questions. Teacher-teacher divergence answers one, teacher entropy answers the other. Together they make CoT format irrelevant to teacher assignment.

---

## Core Thesis

**CoT section labels attempt to answer "which teacher is relevant?" with a coarse, text-based heuristic. They fail because (1) section boundaries don't align with token-level teacher divergence, and (2) they completely ignore a second, equally important question: "is the relevant teacher confident?"**

We propose a two-axis framework that decouples these questions:

- **Axis 1 (Relevance)**: Teacher-teacher divergence — how much unique information does the visual teacher have at this token?
- **Axis 2 (Confidence)**: Teacher entropy — is each teacher's prediction reliable?

The product of these two axes — **Divergence × Confidence (DCP)** — provides a principled, format-agnostic alternative to section-based teacher assignment. Under this framework, CoT format is irrelevant to distillation: the routing signal comes from teacher behavior, not from text structure. No extra forward passes. No learned parameters. One backward per step.

---

## 1. The Problem: One Signal Cannot Answer Two Questions

### 1.1 What section labels try to do

Staged CoT's section-based teacher assignment makes an implicit claim:

> If a token is inside a `<perception>` section, the visual teacher is relevant and should supervise it.

This claim conflates two distinct questions:

| Question | What section labels assume | Reality |
|----------|--------------------------|---------|
| **Q1: Does the visual teacher have unique information?** | Yes, for all perception tokens | Teacher divergence varies from 0 (function words) to high (content words) within the same section |
| **Q2: Is the visual teacher confident?** | Not considered at all | Visual teacher entropy varies across tokens, even when divergence is high |

The section label is a **1-bit proxy for Q1 that ignores Q2 entirely**. It routes supervision based on a semantic convention — not based on what the teachers actually know.

### 1.2 Why existing approaches also conflate

Every single-signal approach to token-level teacher routing suffers from the same conflation:

| Approach | Signal | Answers Q1? | Answers Q2? |
|----------|--------|------------|------------|
| Section labels | Text structure | Crudely (1-bit) | ❌ |
| Entropy ratio | H(π_R) / (H_R+H_P) | ❌ (entropy ≠ relevance) | ✅ |
| Teacher KL alone | KL(π_P ‖ π_R) | ✅ (divergence = unique info) | ❌ |
| Gradient cos | cos(g_P, g_R) | ❌ | ❌ (measures conflict, not confidence) |

**No single scalar can simultaneously capture which teacher has unique information AND whether that teacher is confident.** These are orthogonal dimensions.

---

## 2. Why Teacher-Teacher Divergence, Not Decomposed-Style VDS

### 2.1 The asymmetry in DecomposedOPD (single teacher)

DecomposedOPD decomposes a single teacher's prediction via Bayes rule:

$$
\log \pi_T(\tau \mid I, x) = \underbrace{\log \pi_T(\tau \mid x)}_{\text{Language Prior}} + \underbrace{\log \pi_T(I \mid \tau, x)}_{\text{Visual Likelihood}} - \underbrace{\log \pi_T(I \mid x)}_{\text{Evidence (constant)}}
$$

The key mechanism is the constructed target distribution $q_T^*$ for the visual loss:

$$
q_T^*(\tau \mid I, x) \propto \underbrace{p_{\theta_S}(\tau \mid x)}_{\text{Student's language prior}} \cdot \underbrace{q_T(I \mid \tau, x)}_{\text{Teacher's visual likelihood}}
$$

**$q_T^*$ uses the student's own language prior as its base and grafts on only the teacher's visual information gain.** The only thing in $q_T^*$ that differs from the student's own distribution is the teacher's visual prior — everything else is the student's own. This clean isolation means $\mathcal{L}_{\text{Vis}} = \text{KL}(p_S \parallel q_T^*)$ transfers ONLY visual capability without contaminating the student's language style.

In contrast, $\mathcal{L}_{\text{Lang}} = \text{KL}(p_S(\cdot|\text{text}) \parallel q_T(\cdot|\text{text}))$ transfers the teacher's language prior — a refinement of a capability the student already has (language modeling), not a new capability.

This is the fundamental asymmetry DecomposedOPD exploits:

| Component | What it transfers | Nature |
|-----------|------------------|--------|
| $\mathcal{L}_{\text{Vis}}$ (via $q_T^*$) | Teacher's visual prior | **New capability** (student can't interpret images well) |
| $\mathcal{L}_{\text{Lang}}$ | Teacher's language prior | **Refinement** (student already models language) |

→ VGS biases toward $\mathcal{L}_{\text{Vis}}$ because visual grounding is the bottleneck AND $q_T^*$ provides a clean, contamination-free visual signal. Natural visual > language.

### 2.2 The different asymmetry in DCP (two teachers)

In DCP, both teachers see the image and both are fine-tuned from the same base VLM. Their language priors are nearly identical — the same architecture, same pretraining, same basic multimodal knowledge. The **only systematic difference** comes from RL specialization:

| Teacher | RL training emphasis | Specialized in |
|---------|---------------------|----------------|
| π_P (Visual) | Vision-heavy RL (VQA, grounding, spatial) | Object attributes, spatial relations, counting |
| π_R (Reasoning) | Reasoning-heavy RL (math, logic, planning) | Deduction, comparison, multi-step reasoning |

Unlike DecomposedOPD — where only one direction (visual) provides a genuinely new capability — **both teachers in DCP provide new capabilities**: π_P provides visual grounding expertise the student lacks, π_R provides reasoning expertise the student lacks. The routing problem is therefore fundamentally different: not "bias toward visual," but "identify which expertise is needed at each token."

### 2.3 The right relevance metric: teacher-teacher divergence

Since both teachers share the same base VLM, their language priors are near-identical. On a purely linguistic token (e.g., "the", "therefore"), both predict from the same language distribution → KL(π_P ‖ π_R) ≈ 0. On a visually-dependent token (e.g., "red", "chair"), π_P's vision-specialized training changes its prediction relative to π_R's base-level visual capability → KL(π_P ‖ π_R) > 0.

**KL(π_P ‖ π_R) thus isolates the specialization gap — specifically, π_P's enhanced visual perception relative to π_R.** It is not measuring generic "disagreement"; it is measuring where specialized RL training has caused π_P's predictions to diverge from π_R's. Given the shared base, the primary systematic source of this divergence is visual specialization.

$$
\boxed{D_t = \frac{D_{\text{KL}}(\pi_P(\cdot|s_t) \;\|\; \pi_R(\cdot|s_t))}{D_{\text{KL}}(\pi_P(\cdot|s_t) \;\|\; \pi_R(\cdot|s_t)) + \tau}}
$$

- $D_t \to 1$: π_P's visual specialization strongly changes its prediction → **visual teacher has unique information** → visual teacher is relevant
- $D_t \to 0$: teachers predict identically (shared base language prior dominates) → no specialization gap → **either teacher suffices**

**Why this, not Decomposed-style VDS?**

| | KL(π_R(·\|I) ‖ π_R(·\|text)) | KL(π_P ‖ π_R) |
|---|---|---|
| Question | "Does this token need the image?" | "Where does π_P's visual specialization add value?" |
| Best for | Single-teacher decomposition | **Multi-teacher routing** |
| Extra compute | Text-only forward needed | **None (from existing forwards)** |
| Information source | One model ± image | **Two models with different specializations** |

DecomposedOPD's VDS answers "how much does the image change π_R's prediction?" — the right question when decomposing one teacher. DCP answers "where does π_P's visual specialization exceed π_R's?" — the right question when routing between two specialized teachers.

### 2.4 Connection to DecomposedOPD

The two metrics — DecomposedOPD's VDS and our $D_t$ — measure related but distinct things:

| | DecomposedOPD VDS | Our $D_t$ |
|---|---|---|
| What it measures | Image impact on one model | Specialization gap between two models |
| Formula | KL(π_R(·\|I) ‖ π_R(·\|text)) | KL(π_P(·\|I) ‖ π_R(·\|I)) |
| Extra compute | Text-only forward needed | **None (from existing forwards)** |
| Best for | Single-teacher decomposition | **Multi-teacher routing** |

They are likely correlated (high-VDS tokens tend to have high teacher divergence), but $D_t$ additionally captures cases where π_P and π_R disagree for reasons beyond pure visual dependency — such as inductive bias differences from their specialized training.

---

## 3. The Two-Axis Decomposition

### 3.1 Axis 1: Relevance — Teacher Divergence

$$
D_t = \frac{D_{\text{KL}}(\pi_P \| \pi_R)}{D_{\text{KL}}(\pi_P \| \pi_R) + \tau}
$$

$\tau$ is a temperature controlling the sigmoid steepness. Default $\tau=0.5$ (nats). Meaning: when the two teachers differ by 0.5 nats, relevance is 50%. This is the only hyperparameter in DCP — and it has a clear semantic interpretation.

- $D_t$ high: π_P and π_R give substantially different next-token distributions → visual teacher has unique information
- $D_t$ low: teachers agree → both are saying the same thing → visual teacher adds nothing unique

### 3.2 Axis 2: Confidence — Teacher Entropy

$$
\bar{H}_P = \frac{H(\pi_P(\cdot|s_t))}{\log |V|}, \qquad \bar{H}_R = \frac{H(\pi_R(\cdot|s_t))}{\log |V|}
$$

- Low $\bar{H}$ → concentrated probability mass → teacher is confident → prediction is reliable
- High $\bar{H}$ → flat distribution → teacher is uncertain → prediction is unreliable

**Why confidence matters independently of relevance**: A teacher can have highly relevant information (D_t high: unique visual knowledge) but low confidence (H̄_P high: the image is ambiguous). In this case, blindly trusting the visual teacher would inject noise. The confidence axis down-weights this case.

### 3.3 Why two axes instead of one

Section labels are a **1-bit quantization of the divergence axis** — and they completely miss the confidence axis:

```
Section label view:
  Perception section → "divergence is high" → assign visual teacher
  Reasoning section  → "divergence is low"  → assign language teacher

Reality:
  Within perception: divergence varies continuously from 0 to high
  Within reasoning:  divergence is mostly low, but can spike
  Both sections:      teacher confidence varies independently of divergence
```

A two-axis framework captures what section labels miss:
- High divergence but visual teacher uncertain → reduce visual weight despite high relevance
- Low divergence but language teacher uncertain → reduce language weight despite assumed sufficiency
- Both axes moderate each other → supervision quality = relevance × confidence

---

## 4. The Four Regimes

The product Divergence × Confidence partitions tokens into four naturally interpretable regimes:

```
                    High Confidence              Low Confidence
                    ───────────────              ──────────────
High D_t     │  REGIME I: Strong visual    │  REGIME II: Weak visual
(visual has  │  D_t↑, H_P↓                 │  D_t↑, H_P↑
unique info) │  → Full visual supervision  │  → Cautious visual
             │  e.g., "red", "chair"       │  e.g., ambiguous objects
             │                              │
Low D_t      │  REGIME III: Strong lang    │  REGIME IV: Weak lang
(teachers    │  D_t↓, H_R↓                 │  D_t↓, H_R↑
 agree)      │  → Full language supervis.  │  → Cautious language
             │  e.g., "therefore", "the"   │  e.g., rare reasoning step
```

### Why each regime matters

**Regime I (high divergence, visual confident)**: Visual teacher has unique information AND is certain about it. The core case for visual distillation. These are what section labels attempt to capture — object names, attributes, spatial relations.

**Regime II (high divergence, visual uncertain)**: Visual teacher knows something the language teacher doesn't, but isn't sure about it — occluded objects, fine-grained distinctions, ambiguous scenes. Single-axis routing (KL-only) would give full weight here. DCP reduces weight to prevent the visual teacher from confidently giving wrong answers. **This regime is invisible to section labels and to Route 3.**

**Regime III (low divergence, language confident)**: Teachers agree and language teacher is certain. Pure reasoning tokens. Section labels handle these correctly — but so does DCP, automatically, without needing section labels.

**Regime IV (low divergence, language uncertain)**: Teachers agree but both are uncertain — rare logic patterns, ambiguous deductions. Reducing weight prevents overfitting to teacher uncertainty. **Also invisible to section labels.**

**Section labels only distinguish I+II from III+IV — and even that distinction is noisy within sections.** They miss the I vs. II and III vs. IV splits entirely.

---

## 5. Method: Divergence-Confidence Product Routing (DCP)

### 5.1 Per-teacher weight

$$
w_P = D_t \cdot (1 - \bar{H}_P), \qquad w_R = (1 - D_t) \cdot (1 - \bar{H}_R)
$$

Intuition: Visual weight = (does visual teacher have unique info?) × (is it confident?). Language weight = (is language teacher sufficient?) × (is it confident?).

### 5.2 Normalized routing weight

$$
\boxed{\alpha_t = \frac{D_t \cdot (1 - \bar{H}_P)}{D_t \cdot (1 - \bar{H}_P) + (1 - D_t) \cdot (1 - \bar{H}_R)}}
$$

### 5.3 Distillation loss

$$
\boxed{\mathcal{L} = \sum_t \left[\alpha_t \cdot \text{KL}(\pi_\theta \| \pi_P) + (1-\alpha_t) \cdot \text{KL}(\pi_\theta \| \pi_R)\right]}
$$

**One forward, one backward. Zero extra passes. Zero learned parameters. One hyperparameter ($\tau$, with semantic meaning).**

### 5.4 Algorithm

```
For each training step:
  1. Student rollout: ŷ ~ π_θ(·|I, Q) using Mixed CoT prompt
  
  2. Teacher forward (with image) — done once, shared:
     p_P = π_P(·|I, Q, ŷ_<t)     # visual teacher
     p_R = π_R(·|I, Q, ŷ_<t)     # reasoning teacher
  
  3. Compute routing signals from p_P, p_R (no extra forwards):
     D_t = KL(p_P || p_R) / (KL(p_P || p_R) + τ)    # divergence
     H̄_P = H(p_P) / log|V|                           # visual confidence
     H̄_R = H(p_R) / log|V|                           # language confidence
  
  4. Routing weight:
     α_t = D_t·(1-H̄_P) / [D_t·(1-H̄_P) + (1-D_t)·(1-H̄_R)]
  
  5. Loss and update:
     L = Σ_t [α_t·KL(π_θ||p_P) + (1-α_t)·KL(π_θ||p_R)]
     L.backward()  # single backward
```

### 5.5 Computational cost

| Method | Forward passes | Backward passes | Extra parameters |
|--------|---------------|-----------------|------------------|
| Uniform MOPD | 1 | 1 | 0 |
| Staged + section assignment | 1 | 1 | 0 |
| PCGrad | 1 | **2** | 0 |
| DCP (Ours) | 1 | **1** | 0 |
| Learned router (TCTR) | 1 | 1 | ~1M |

DCP matches the computational cost of uniform MOPD and section-based assignment. The only additional cost is the element-wise operations to compute KL, entropy, and α_t from the teacher log-probabilities — negligible compared to the forward/backward passes.

---

## 6. Why CoT Format Becomes Irrelevant

### 6.1 The routing signal is format-independent

$D_t$ and teacher entropy are computed from **teacher model outputs**, not from **text structure**. They depend on:
- The image content
- The student's generated prefix $\hat{y}_{<t}$
- The teacher models' parameters

None of these depend on whether the text says `<perception>` or interleaves visual and linguistic reasoning.

### 6.2 Mixed CoT is now the natural choice

Under section-based assignment, Staged CoT was "necessary" because routing needed section labels. Under DCP routing:

- **Mixed CoT works natively**: No section labels needed. Routing is handled by Divergence × Confidence at every token.
- **Staged CoT's section labels become dead weight**: They provide no additional routing information beyond what teacher behavior already captures — and they lose the confidence axis entirely.
- **CoT format choice reduces to efficiency**: Mixed CoT uses ~33% fewer tokens with comparable accuracy → Mixed CoT dominates.

### 6.3 Section labels are strictly worse

| What section labels provide | What DCP provides |
|---|---|
| Binary routing (perception → visual, reasoning → language) | Continuous routing based on teacher behavior |
| No confidence modulation | Confidence-gated via teacher entropy |
| Requires explicit format structure | Format-agnostic |
| 1-bit approximation of divergence axis | Full divergence computation |
| Ignores Regime II and IV | Handles all four regimes |

**Section labels are not just unnecessary — they lose information.** Using section labels when teacher behavior signals are available is like using a binary classifier when you have the continuous logits.

---

## 7. Related Work

### 7.1 ViGOS: Section-Based Perception-Reasoning Separation

ViGOS (arXiv:2606.19120) is the closest prior work to DCP. It introduces a staged CoT format with `<description>` (image-only perception) and `<think>` (reasoning) sections, using the same model with different input modes to supervise different CoT segments. ViGOS demonstrates that perception and reasoning benefit from different supervision — a key insight DCP builds on.

| | ViGOS | DCP (Ours) |
|---|---|---|
| Core insight | Perception and reasoning need different supervision | Same |
| Routing signal | CoT section labels (hard) | Teacher behavior (continuous) |
| Routing granularity | Section-level | Token-level |
| CoT format | Requires Staged CoT | Format-agnostic |
| Teacher | Same model, different input modes | Two independently specialized teachers |
| Confidence axis | Not considered | Explicitly modeled |

**Key difference**: ViGOS uses the CoT format itself as the routing mechanism — `<description>` tokens → visual supervision, `<think>` tokens → reasoning supervision. DCP decouples routing from format: teacher-teacher divergence and teacher entropy provide a continuous, token-level signal that works with any CoT format.

**DCP as ViGOS's natural generalization**: ViGOS proves that perception-reasoning separation is valuable. DCP shows that section labels are a 1-bit quantization of what should be a continuous signal, and that teacher confidence — completely unmodeled by ViGOS — is equally important for routing quality.

### 7.2 DecomposedOPD (ICML 2026 Spotlight)

DecomposedOPD decomposes a single teacher's predictions into language prior and visual likelihood via Bayes rule, biasing OPD toward the visual component. It operates in a single-teacher strong-to-weak distillation setting (large teacher → small student), unlike DCP's multi-teacher same-scale setting.

**Connection**: Both share the insight that teacher supervision quality varies across tokens. DecomposedOPD's VDS measures "how much does the image change this model's prediction?" — the right question for single-teacher decomposition. DCP's teacher-teacher KL measures "where does π_P's specialization exceed π_R's?" — the right question for multi-teacher routing. The two frameworks are complementary: DecomposedOPD addresses within-teacher signal decomposition; DCP addresses across-teacher signal selection.

### 7.3 DOPD and H-OPD: Different Problem Settings

**DOPD** (arXiv:2606.30626) addresses single-teacher privileged distillation. Its advantage gap routes *distillation strength* (how much to trust the teacher vs. the student's own policy on each token), not *teacher selection* (which of two specialist teachers to trust). DOPD's four-regime partition (advantage gap × confidence) uses a similar analytical structure to DCP, but the first axis measures a fundamentally different quantity.

**H-OPD** (arXiv:2607.02592) addresses heterogeneous multi-teacher distillation where one teacher sees images and another does not. Its core technical problem is modality bridging (vision-to-language description transfer); its arbitration mechanism selects between teachers with different input modalities. DCP's teachers are both VLMs with the same input modality but different RL specializations — no cross-modal translation is needed, and the routing problem is specialization selection rather than modality selection.

### 7.4 Entropy-based gating (EGRSD, SEAD, CAKD, DE-MKD)

All use teacher and/or student entropy to determine **whether** to distill at a token (or **how much**). All are **single-teacher** methods. DE-MKD uses entropy ratio for multi-teacher weighting but at the **sample level** (image classification), not token level.

**DCP differs**: (1) Multi-teacher **token-level** routing (which teacher), not single-teacher gating (whether to distill). (2) Second axis (divergence) is essential — entropy alone conflates relevance and confidence.

### 7.5 Teacher disagreement methods (UniKD, EWAD, RAPS-DA)

Use teacher-teacher KL/JSD as a routing or weighting signal. But teacher-teacher divergence is used as a **single scalar** — it conflates relevance and confidence.

**DCP differs**: DCP explicitly decouples divergence (relevance) from entropy (confidence). This is the core conceptual contribution — not the signals themselves, but the recognition that they answer different questions.

### 7.6 Gradient-space methods (PCGrad, AE-KD, ATTITTUD)

Operate in gradient space with 2× backward overhead. Address conflict, not routing.

**DCP differs**: Operates in distribution space with 1× backward. Addresses routing (which teacher for which token), not conflict resolution.

### 7.7 Learned routers (TCTR, COMPACT, VAMOPD)

Learn a router network $\lambda_\phi(s_t)$ to predict per-token teacher weights.

**DCP differs**: Zero learned parameters. Weights are **computed** from teacher behavior, not **learned** from data. This eliminates the off-trajectory generalization problem and the need for router training data. The routing is also interpretable — you can inspect why α_t is high or low for any token.

---

## 8. Experimental Design

### 8.1 Core claim to validate

> Divergence × Confidence routing outperforms section-based teacher assignment, and makes CoT format irrelevant to distillation quality.

### 8.2 Primary experiments

| Experiment | What it proves |
|------------|---------------|
| **Mixed CoT + DCP vs. Staged CoT + section assignment** | DCP with Mixed CoT ≥ section assignment with Staged CoT → format is irrelevant, DCP is better |
| **Mixed CoT + DCP vs. Mixed CoT + uniform MOPD** | DCP adds value over naive multi-teacher averaging |
| **Staged CoT + DCP vs. Staged CoT + section assignment** | DCP improves even within Staged CoT (section labels are worse even in their "home" format) |
| **DCP vs. divergence-only routing** | Ablation: confidence axis matters |
| **DCP vs. entropy-only routing** | Ablation: relevance axis matters |
| **DCP vs. product vs. sum vs. max** | Ablation: product form is justified |
| **DCP vs. learned router (TCTR-style)** | DCP approaches learned router performance with zero parameters |
| **DCP (teacher-teacher KL) vs. DCP (Decomposed-style VDS)** | Ablation: teacher divergence is the right relevance metric for multi-teacher |

### 8.3 Diagnostic experiments

#### Four-regime analysis

```
For each token in the training set:
  1. Compute D_t, H̄_P, H̄_R
  2. Classify into Regime I/II/III/IV
  3. Report:
     - Distribution of tokens across regimes
     - Per-regime downstream accuracy improvement from DCP
     - I > II and III > IV (confidence modulation works)
     - I+III >> II+IV (relevance modulation works)
     - Which regime benefits most from DCP vs. uniform MOPD?
```

#### Section label vs. teacher divergence misalignment

```
Plot: x-axis = token position in Staged CoT sequence
      y-axis = D_t (teacher divergence)
      Color = teacher entropy (blue = confident, red = uncertain)
      Overlay: section boundary marker

Key observation:
  - D_t varies continuously; no jump at section boundary
  - High-entropy visual tokens exist within perception sections (Regime II)
  - Section labels are a 1-bit quantization of a continuous signal
```

#### Correlation between Decomposed-style VDS and teacher divergence

```
Scatter plot: x = DecomposedOPD VDS (π_R ± image), y = D_t (π_P vs π_R)
  - Expected: positive correlation (visual tokens show high values in both)
  - But D_t may capture additional divergence beyond pure visual dependency
  - Report Spearman ρ and cases where they disagree
```

### 8.4 Baselines

```
1.  Base Student (no distillation)
2.  Single-teacher OPD (visual teacher only)
3.  Single-teacher OPD (reasoning teacher only)
4.  MOPD uniform (α = 0.5 fixed)
5.  MOPD + section-based assignment (Staged CoT only)
6.  MOPD + divergence-only routing (α_t from D_t alone)
7.  MOPD + entropy-only routing (α_t from H ratio)
8.  MOPD + teacher-KL routing (Route 3: KL-based PoE mixture)
9.  MOPD + DCP (ours)
10. MOPD + DCP with Decomposed-style VDS (for ablation)
11. MOPD + learned router (TCTR-style)
12. MOPD + Protective PCGrad
```

### 8.5 Metrics

- Downstream task accuracy (VSR1, GQA, OK-VQA)
- Per-regime accuracy breakdown
- Training efficiency (wall time per step vs. accuracy gain)
- Routing sharpness: $\text{std}(\alpha_t)$ across tokens (higher = more decisive routing)
- Fraction of tokens where $|\alpha_t - 0.5| > 0.2$ (meaningfully routed tokens)

---

## 9. Risk Assessment

### 9.1 Known risks

| Risk | Severity | Mitigation |
|------|----------|------------|
| **D_t and entropy are correlated** — product adds no info | High | Diagnostic: compute correlation. If r > 0.7, the confidence axis may be redundant. But even then, the four-regime taxonomy provides conceptual value, and the framework can fall back to divergence-only routing |
| **Section assignment isn't actually worse** — DCP ≤ section assignment | High | Pilot experiment is the first thing to run. If section assignment matches DCP, pivot from "section labels are harmful" to "section labels are unnecessary, DCP is simpler" |
| **Teacher entropy patterns are identical** (H_P ≈ H_R everywhere) | High | The confidence axis loses routing power. Degenerates to divergence-only routing. Still contributes the divergence axis (which section labels miss), but loses half the framework |
| **Two-axis story is "obvious in hindsight"** | Medium | Mitigated if (a) experiments show non-trivial regime distributions, (b) the four-regime analysis reveals patterns that single-axis methods miss |
| **τ hyperparameter needs tuning** | Low | τ has clear semantics (nats of divergence for 50% relevance). Default 0.5. Sensitivity analysis in experiments |
| **Teacher construction** — need genuinely different teachers | Medium | Different RL training objectives naturally produce different specialization. Verify entropy pattern difference in pilot. If teachers are too similar, routing degenerates — but then multi-teacher KD itself is questionable |

### 9.2 Predicted reviewer responses

**Q: "Why is divergence × confidence the right combination? Why not learn the combination?"**

A: The product form has a natural probabilistic interpretation: the probability that the visual teacher should supervise token t is P(has unique info) × P(prediction is reliable). If either condition fails, the probability goes to zero. This is a design choice — but we ablate product vs. sum vs. max vs. learned combination. If a learned combination substantially outperforms the product, that would be interesting. But the product form works as a strong zero-parameter baseline with clear semantics.

**Q: "How do you construct the visual and reasoning teachers?"**

A: Starting from the same base VLM, fine-tune π_P with RL on vision-heavy tasks (VQA, spatial reasoning, visual grounding) and π_R with RL on reasoning-heavy tasks (math, logic, planning). The training objective difference — not token-level annotations — produces the specialization that DCP exploits. We do NOT use section labels for teacher training.

**Q: "Why teacher-teacher KL rather than DecomposedOPD's VDS?"**

A: DecomposedOPD's VDS measures "how much does the image change model X's prediction?" — appropriate when comparing the same model with and without the image. DCP compares two different models (π_P vs π_R) on the same multimodal input — the natural metric is their output divergence. We include DCP with Decomposed-style VDS as an ablation baseline.

**Q: "Why only two teachers? What about K > 2?"**

A: The visual-reasoning dichotomy is the most fundamental and common specialization in VLM distillation. For K > 2, the framework extends to pairwise divergence and confidence scores, normalized via softmax. The two-teacher case is the most important setting and the one where section labels are most commonly used. Extension to K>2 is conceptually straightforward but left for future work.

### 9.3 Paper-ending scenarios

1. **D_t and H_P/H_R are highly correlated** (r > 0.85) → the confidence axis is redundant → framework reduces to divergence-only → contribution is "teacher divergence > section labels" (weaker but still a contribution)
2. **Section assignment baseline beats DCP** → the premise is empirically falsified → paper in current form cannot exist → need to understand why and potentially pivot
3. **Teacher entropy patterns are identical** → confidence axis provides zero routing information → DCP = divergence-only → half the framework is lost → need to investigate why teachers aren't differentiated
4. **DCP = uniform MOPD** (α_t ≈ 0.5 everywhere) → teacher divergence is never large enough to produce decisive routing → multi-teacher KD provides no benefit over single-teacher → questions the entire MOPD motivation

---

## 10. Paper Positioning

### 10.1 One-sentence

> CoT section labels conflate two orthogonal signals for multi-teacher routing — teacher divergence and teacher confidence. Decoupling them makes CoT format irrelevant to distillation quality, at zero extra computational cost.

### 10.2 Contribution summary

| # | Contribution | Type |
|---|-------------|------|
| 1 | **Two-axis decomposition**: Multi-teacher routing requires answering two independent questions — which teacher has unique information? (divergence) and is that information reliable? (confidence). All prior work conflates them into a single signal | Conceptual |
| 2 | **Divergence-Confidence Product (DCP)**: A zero-parameter, zero-overhead routing mechanism combining teacher-teacher divergence with per-teacher entropy | Method |
| 3 | **CoT format irrelevance**: Under two-axis routing, CoT section structure provides no additional value for teacher assignment → format choice reduces to token efficiency → Mixed CoT dominates | Theoretical |
| 4 | **Four-regime taxonomy**: A framework for analyzing token-level teacher supervision quality. Reveals failure modes (Regime II: high relevance + low confidence) invisible to both section-based and single-axis approaches | Analytical |

### 10.3 Venue fit

| Venue | Fit | Reasoning |
|-------|-----|-----------|
| CVPR 2027 | 45-55% | VLM distillation + visual reasoning theme. Two-axis framework is conceptually clean. Needs strong experiments |
| NeurIPS 2027 | 40-50% | Method + analysis. Right level of contribution |
| ICLR 2028 | 35-45% | More theory-heavy than our contribution |
| ICML 2027 | 30-40% | ICML prefers deeper theory |

---

## 11. Next Steps

### Critical path (must do before committing)

- [ ] **Pilot 1**: Compute D_t, H_P, H_R on 100 Mixed CoT student rollouts. Check:
  - Distribution of D_t (does it deviate from 0?)
  - Correlation between D_t and teacher entropy
  - Four-regime token distribution (do Regime II and IV exist?)
  - Distribution of α_t (does routing deviate from 0.5?)
- [ ] **Pilot 2**: Teacher differentiation check — are H_P and H_R substantially different on key tokens? If not, teachers need more specialization.
- [ ] **Pilot 3**: Tiny distillation run (50 samples) — DCP vs. section assignment vs. uniform MOPD vs. divergence-only vs. entropy-only

### If pilots pass

- [ ] Full experiment matrix (Section 8)
- [ ] Four-regime diagnostic analysis
- [ ] Section label vs. divergence misalignment visualization
- [ ] Write Introduction + Method + Experiments

---

*Last updated: 2026-07-16*
