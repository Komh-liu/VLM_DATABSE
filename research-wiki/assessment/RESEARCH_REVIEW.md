# Research Review — TCTR (Revised, 2026-07-11)

**Artifact**: `vamopd_variational_mixture_formulation.md` (now titled TCTR)
**Review type**: Deep technical review (methods + claims + experimental design)

---

## 1. Overall Assessment

**Score: 6.5/10** (↑ from 5.5 — structural issues resolved, method is now honest and well-framed)

The revisions successfully addressed the three blocking issues from the previous review:
1. ✅ "Variational" misnomer → removed entirely, now "Trajectory-Calibrated Teacher Routing"
2. ✅ DOPD not cited → now in baseline matrix, routing diagnostic, and risk discussion
3. ✅ No heuristic comparison → Section 8.4 (routing diagnostic) and Section 10 (three gates) explicitly define the experimental bar

The narrative is now significantly more honest and defensible. The method is framed as a data utilization contribution (structured annotations → routing supervision) rather than a variational inference contribution. This framing is more accurate and harder to attack.

**Recommendation**: Weak Accept → Major Revisions. The experimental results will determine the final verdict. Submit for CVPR 2027 if experiments converge positively by October 2026.

---

## 2. Strengths

### S1: Problem framing is clear and well-motivated
The document now opens with the core question: "Which teacher should supervise which student-visited state?" This is a clean, operational framing. The key observation — "GQA scene graph is not just an evaluation resource" — is the right hook for the method.

### S2: Three-gate framework (Heuristic / DOPD / Transfer)
Section 10's probability breakdown by gate conditions is a honest and useful self-assessment. This should be included in the paper and would be well-received by reviewers who value clean experimental design.

### S3: Baseline matrix is comprehensive
The expanded baseline matrix (Section 8.2) includes SFT, OPD, MOPD, entropy routing, disagreement routing, DOPD-style routing, ViGOS hard separation, DOPD, TCTR + DOPD combination, and 4 ablations. This is thorough. The routing diagnostic (Section 8.4) with per-category accuracy is well-designed.

### S4: Off-trajectory evaluation (Section 8.6)
The divergence-bucket evaluation (student vs trajectory prefix divergence) directly tests the sparse-to-dense generalization claim. This would be diagnostic gold.

### S5: Trajectory quality evaluation (Section 8.5)
Five dimensions with specific metrics: answer consistency, scene-graph faithfulness, naturalness, routing distribution, teacher agreement. This shows experimental maturity.

### S6: Risk aware
Section 9 lists 7 risks, from trajectory quality to teacher-pair dependence. This intellectual honesty strengthens the document.

---

## 3. Remaining Weaknesses

### W1: Pipeline novelty framing is fragile (MAJOR)

**Problem**: The method is structured as:
```
Scene graph → Trajectory (programmatic generation)
                           ↓
                      Router training (MLP regression)
                           ↓
                      Dense OPD (standard KL distillation)
```

A reviewer can argue: "The first stage is data engineering, the second is supervised learning, the third is standard OPD. What is the research contribution?"

**Response**: The contribution is in the *connection between stages*: no prior work uses structured annotations to calibrate teacher routing. This is a valid defense, but the paper must explicitly argue this, not assume it.

**Fix**: Include a "Contribution" paragraph that clearly states: "Our contribution is neither the trajectory generation algorithm nor the router architecture. It is the demonstration that structured visual annotations — previously used only for evaluation — can serve as routing supervision for multi-teacher OPD, and that the learned routing transfers to settings without such annotations." This preempts the "just data engineering" critique.

### W2: Router features are hand-crafted, not learned (MAJOR)

**Problem**: The router input $z(s)$ is hand-designed:
$$
z(s) = [h_\theta(s), D_{cand}, D_{KL}(\pi_P\|\pi_R), D_{KL}(\pi_R\|\pi_P), \log\pi_P - \log\pi_R]
$$

This is a fixed feature engineering choice. A more principled approach might learn which aspects of teacher distributions matter for routing. The current approach works but is not "deep" — a reviewer could call it "feature engineering on top of feature engineering."

**Fix**: (a) Show that learned features (via a small transformer over teacher logits) do not significantly outperform hand-crafted features, or (b) include an ablation that drops each feature to show which ones matter. This turns a weakness into experimental evidence of insight.

### W3: No negative results analysis (MODERATE)

**Problem**: The document doesn't discuss what happens when routing is wrong. If $\lambda_\phi(s) \approx 0.2$ but the state actually needs perception (λ ≈ 0.8), what is the impact on the student's learning? Is there error compounding?

**Fix**: Add a short analysis: either theoretical (bounded routing error → bounded KL degradation) or empirical (measure routing error vs student gain correlation). This is particularly important for the off-trajectory case, where routing errors are most likely.

### W4: Second model family not included in baseline plan (MODERATE)

**Problem**: The experimental plan assumes Qwen2.5-VL as the primary model. For CVPR, a second model family (LLaVA, InternVL, or Idefics) significantly reduces the "only tested on one architecture" criticism.

**Fix**: Even one partial experiment on a second architecture (e.g., LLaVA-1.6 on GQA held-out with the routing diagnostic) would be sufficient.

### W5: Trajectory paraphrasing quality control (MODERATE)

**Problem**: The trajectory generation pipeline uses a strong VLM (Qwen2.5-VL-72B) for paraphrasing scene graph skeletons into natural language. This introduces a potential confound: are improvements from the routing method or from the privileged information in the teacher trajectory?

**Fix**: Include a baseline where the teacher trajectory SFT uses the same paraphrased trajectories. The current baseline matrix includes "Scene-graph trajectory SFT," which addresses this. Ensure the filtering criteria (Section 2.3) are strict enough that the trajectories are not leaking privileged information.

---

## 4. Experimental Design Review

### What improved since the previous version

| Aspect | Previous | Current |
|--------|----------|---------|
| DOPD baseline | Not mentioned | Included ✓ |
| Heuristic baselines | Not mentioned | Entropy, disagreement, DOPD-style included ✓ |
| Routing diagnostic | Not present | Section 8.4 with per-category accuracy ✓ |
| Three gates | Not present | Section 10 ✓ |
| Off-trajectory eval | Brief mention | Section 8.6 with divergence buckets ✓ |
| Trajectory quality eval | Not present | Section 8.5 with 5 metrics ✓ |
| Combined TCTR + DOPD | Not present | Section 8.2 ✓ |

### Recommended experiment priority

```
Week 1:   Routing diagnostic (Section 8.4) — 1-2 days, no OPD
          → Decision gate: if TCTR router ≤ best heuristic on high-conflict accuracy, STOP

Week 2-3: Data efficiency sweep (1K/5K trajectories, 1%/5%/10% ratios)
          → Decision gate: if SFT trajectory baseline > TCTR, rethink router design

Week 4-5: Full OPD on GQA with all baselines
          λ(t) visualization on 10 examples

Week 6:   Off-trajectory evaluation (Section 8.6)
          Routing feature ablation

Week 7:   Cross-dataset transfer (GQA → OK-VQA)

Week 8:   Second model family (partial)
          Statistical significance on all results
```

### 3 critical experimental questions

1. **Does trajectory routing beat free heuristics?** (determines method necessity)
2. **Does routing quality degrade under student prefix divergence?** (determines sparse-to-dense generalization claim)
3. **Does GQA-trained routing transfer to OK-VQA?** (determines cross-dataset contribution)

---

## 5. Related Work Gaps

### Addressed now:
- ✅ DOPD (2606.30626)
- ✅ MOPD (2606.30406)
- ✅ CaMOPD (2605.27115)
- ✅ ViGOS (2606.19120)
- ✅ Decomposed OPD (ICML 2026)
- ✅ MoCA (ICML 2026)

### Still missing:
- **HEED (2605.17093)**: Density-weighted per-token VLM distillation — closest on per-token weighting mechanism. Should be cited.
- **BOLT (ICLR 2026)**: Budget-aware routing + decision-aligned distillation for multimodal QA — routing in distillation context.

---

## 6. Venue Fit

| Venue | Deadline | Fit | Probability |
|-------|----------|-----|-------------|
| **CVPR 2027** | ~Nov 2026 | **Good** — GQA + visual reasoning + routing is natural CVPR territory | **50-60%** (↑ from previous) |
| NeurIPS 2027 | ~May 2027 | Moderate — empirical focus is more CVPR than NeurIPS | 40-50% |
| ICLR 2028 | ~Oct 2027 | Moderate — if router theory is deepened | 45-55% |
| ECCV 2028 | ~Mar 2027 | Good fallback — empirical paper welcome | 60-70% |

**Recommendation**: Target CVPR 2027. The revisions fixed the structural issues; the remaining unknown is experimental signal strength. If experiments converge by October, submit to CVPR. If the routing diagnostic (Week 1) is promising, continue. If not, pivot to a different framing or target ECCV.

---

## 7. Action Items

### Before paper writing
1. **Run routing diagnostic** (Section 8.4) — 1-2 days, determines whether to continue
2. **Write DOPD narrative differentiation** — prepare one paragraph explaining complementary problem settings
3. **Implement trajectory generation pipeline** — start with 1K GQA scene graphs

### Before CVPR submission
4. Full baseline matrix on GQA
5. Off-trajectory evaluation
6. Cross-dataset transfer
7. λ(t) visualization
8. Feature ablation study
9. Statistical significance on all metrics

### Nice to have
10. Second model family experiment
11. HEED and BOLT citation
12. Routing error analysis (Section 3 — W3)

---

## 8. Claims Matrix

| Outcome | Claim | Venue |
|---------|-------|-------|
| TCTR ≈ heuristic routing | "Routing diagnostic shows learned router is unnecessary" | — (stop) |
| TCTR > heuristics on routing diagnostic | "Trajectory-calibrated routing provides superior per-token teacher assignments" | Workshop |
| + full OPD: TCTR > fixed MOPD + ViGOS | "Continuous trajectory-calibrated routing outperforms fixed mixing and task-free routing" | ECCV |
| + DOPD: TCTR + DOPD > DOPD | "Capability conflict routing is complementary to privilege-aware routing" | CVPR |
| + transfer: GQA → OK-VQA works | "Routing trained on structured annotations generalizes across visual reasoning benchmarks" | CVPR/NeurIPS |
| + second model family | "TCTR is architecture-agnostic and scales across VLM families" | CVPR Oral |
