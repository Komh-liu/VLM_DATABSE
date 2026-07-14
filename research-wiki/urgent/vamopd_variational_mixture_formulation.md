# TCTR: Trajectory-Calibrated Teacher Routing for Multi-Teacher OPD

> 将 multi-teacher OPD 在 mixed visual reasoning 场景下的监督冲突，形式化为一个 **structured evidence -> pseudo-optimal trajectory -> dense teacher routing** 问题：GQA scene graph 可以程序化构造近似最优推理轨迹，并在每个 token/span 上生成连续 teacher-mixture target。该 target 允许相邻 token 使用不同专家监督，也允许两个专家联合监督。

---

## 1. Core Motivation

Multi-teacher OPD 的关键问题不是缺少 teacher，而是不同 teacher 在同一 student-visited state 上会给出不同监督方向：

$$
s_t=(I,Q,\hat{y}_{<t}),\qquad \hat{y}\sim\pi_\theta
$$

两个 teacher：

- **Perception teacher** $\pi_P$：擅长 object / attribute / relation grounding
- **Reasoning teacher** $\pi_R$：擅长 comparison / logic / answer aggregation

固定权重 MOPD 使用：

$$
\alpha\pi_P(\cdot\mid s)+(1-\alpha)\pi_R(\cdot\mid s)
$$

但 mixed visual reasoning 中，不同 state 需要不同 teacher。核心问题是：

$$
\boxed{
\text{Which teacher should supervise which student-visited state?}
}
$$

TCTR 的关键观察：

> GQA scene graph 不只是 evaluation resource。它可以程序化生成 pseudo-optimal reasoning trajectories，并为每个 token/span 提供连续 teacher-mixture supervision。

---

## 2. Structured Evidence -> Routing Trajectory

### 2.1 GQA scene graph 的训练价值

GQA scene graph 包含 objects、attributes、relations，以及问题所需的组合推理结构。例如：

```
Scene graph:
  objects: chair, table, cup
  attributes: chair.color=red, table.material=wood
  relations: chair left_of table, cup on table
  question: "What is on the right of the red chair?"

Generated interleaved trajectory:
  Step 1: decompose the question into target and relation
          λ is biased toward reasoning, but may still use visual context
  Step 2: locate and verify the red chair
          λ is biased toward perception, but may still use question context
  Step 3: use the relation constraint to decide where to inspect next
          λ may be mixed because relation reasoning and visual grounding interact
  Step 4: verify the candidate object from the image / scene graph
          λ is biased toward perception with reasoning context
  Step 5: aggregate the verified evidence and answer
          λ is biased toward reasoning, grounded by previous visual evidence
```

这里不显示离散专家标签，因为相邻 token 完全可能需要不同专家监督，单个 span 内也可能需要联合监督。我们为每个 token/span 生成连续 routing target：

$$
\lambda_t\in[0,1]
$$

其中 $\lambda_t$ 越接近 1，teacher mixture 越偏向 $\pi_P$；越接近 0，越偏向 $\pi_R$；中间值表示联合监督。

注意这里不是“先看完再推理”的两段式流程，而是：

$$
\text{decide what to inspect}
\rightarrow
\text{verify evidence}
\rightarrow
\text{update the next constraint}
\rightarrow
\text{verify new evidence}
\rightarrow
\text{aggregate answer}
$$

因此 TCTR 学到的是 token/span-level 连续 routing，而不是把输出模板硬切成 perception segment 和 reasoning segment。

这把 trajectory dependency 从“昂贵标注”变成“程序化生成”：

$$
\mathcal{D}_{\text{GQA-SG}}
\xrightarrow{\text{programmatic generation}}
\mathcal{D}_{\text{traj}}
=
\{(x_i,\tau_i,\lambda_i)\}_{i=1}^N
$$

其中：

$$
\tau_i=(v_1,\ldots,v_T),
\qquad
\lambda_i=(\lambda_1,\ldots,\lambda_T)
$$

### 2.2 Operation-to-routing prior

令第 $t$ 步 scene graph operation 为：

$$
o_t\in
\{\text{decompose},\text{object},\text{attribute},\text{relation},\text{verify},\text{comparison},\text{logic},\text{answer}\}
$$

operation 不直接决定硬标签，而是给出一个软先验：

$$
\mu_{\text{SG}}(o_t)\in[0,1],
\qquad
c_{\text{SG}}(o_t)\ge 0
$$

$\mu_{\text{SG}}$ 表示该 operation 对 teacher mixture 的倾向，$c_{\text{SG}}$ 表示这个倾向的强度。例如 object / attribute / verify 的 $\mu_{\text{SG}}$ 可以偏高，decompose / logic / answer 的 $\mu_{\text{SG}}$ 可以偏低，relation 的 $\mu_{\text{SG}}$ 可以更接近中间。它们都是软先验，不是硬标签。

最终 $\lambda_t$ 还会被两个 teacher 对 pseudo-optimal token 的支持程度修正：

$$
p_P^t=\pi_P(v_t^{\text{traj}}\mid s_t),
\qquad
p_R^t=\pi_R(v_t^{\text{traj}}\mid s_t)
$$

所以即使两个相邻 token 属于相似 operation，它们也可以得到不同的 $\lambda_t$；同一个 span 内也可以出现连续变化的 teacher mixture。

### 2.3 Trajectory generation pipeline

推荐的构造流程：

```
1. Parse GQA question and scene graph.
2. Extract interleaved evidence-seeking and constraint-updating operations.
3. Generate a structured reasoning skeleton with soft operation priors, alternating "what to inspect" and "what was verified".
4. Use strong VLM / LLM to paraphrase the skeleton into natural reasoning.
5. Preserve span-level routing priors during paraphrasing.
6. Filter by answer consistency and scene-graph faithfulness.
```

质量过滤信号：

- final answer 与 GQA answer 一致
- trajectory 中的 object / attribute / relation 都存在于 scene graph
- paraphrase 不引入 scene graph 外对象
- routing label 分布不过度塌缩到一个 teacher
- teacher likelihood 不极端反对该 trajectory span

---

## 3. Teacher Router

### 3.1 Teacher mixture target

在 student state $s$ 上，定义 teacher mixture：

$$
a_\lambda(v\mid s)
=
\lambda(s)\pi_P(v\mid s)
+
(1-\lambda(s))\pi_R(v\mid s)
$$

学习一个 router：

$$
\lambda_\phi(s)=r_\phi(z(s))
$$

其中 $z(s)$ 可以包含：

$$
z(s)=
\left[
h_\theta(s),\;
D_{\text{cand}}(\pi_P,\pi_R),\;
D_{\text{KL}}(\pi_P\|\pi_R),\;
D_{\text{KL}}(\pi_R\|\pi_P),\;
\log\pi_P(v^{\text{traj}}\mid s)-\log\pi_R(v^{\text{traj}}\mid s)
\right]
$$

这里的核心不是 teacher confidence，而是 teacher 对 pseudo-optimal token / candidate set 的相对支持。entropy 可以作为 baseline 或辅助特征，但不应成为主要依据。

### 3.2 Router calibration loss

在 generated trajectory states 上，先用 soft operation prior 和 teacher 对 pseudo-optimal token 的支持程度得到连续 target：

$$
\boxed{
\lambda_t^*
=
\arg\min_{\lambda\in[0,1]}
\left[
-\log
\left(
\lambda p_P^t+(1-\lambda)p_R^t
\right)
+
c_{\text{SG}}(o_t)
\left(
\lambda-\mu_{\text{SG}}(o_t)
\right)^2
\right]
}
$$

其中：

$$
p_P^t=\pi_P(v_t^{\text{traj}}\mid s_t),
\qquad
p_R^t=\pi_R(v_t^{\text{traj}}\mid s_t)
$$

这一步允许相邻 token 得到不同 $\lambda_t^*$，也允许 $\lambda_t^*$ 落在中间区间表示联合监督。然后训练 router：

$$
\boxed{
\mathcal{L}_{\text{route}}(\phi)
=
\mathbb{E}_{s\sim\mathcal{D}_{\text{SG-traj}}}
\left[
w(s)
\left(
\lambda_\phi(s)-\lambda_t^*
\right)^2
\right]
}
$$

$w(s)$ 是 label confidence，可以来自：

- scene graph parser confidence
- operation 类型置信度
- teacher likelihood margin

例如：

$$
w(s)=
\left|
\pi_P(v^{\text{traj}}\mid s)
-
\pi_R(v^{\text{traj}}\mid s)
\right|
$$

### 3.3 Dense OPD loss

在 student rollout states 上：

$$
s\sim d_{\pi_\theta}
$$

router 产生 dense teacher mixture：

$$
a_{\lambda_\phi}(v\mid s)
=
\lambda_\phi(s)\pi_P(v\mid s)
+
(1-\lambda_\phi(s))\pi_R(v\mid s)
$$

student 的 OPD loss：

$$
\boxed{
\mathcal{L}_{\text{TCTR}}(\theta)
=
\mathbb{E}_{s\sim d_{\pi_\theta}}
\left[
D_{\text{KL}}
\left(
\pi_\theta(\cdot\mid s)
\parallel
a_{\lambda_\phi}(\cdot\mid s)
\right)
\right]
}
$$

总目标：

$$
\mathcal{L}
=
\mathcal{L}_{\text{TCTR}}
+
\beta\mathcal{L}_{\text{route}}
+
\gamma\mathcal{L}_{\text{smooth}}
+
\delta\mathcal{L}_{\text{traj-imitation}}
$$

其中 trajectory imitation 是辅助项。主贡献是：

$$
\boxed{
\text{soft operation prior + teacher support}
\rightarrow
\text{router}
\rightarrow
\text{dense OPD on student states}
}
$$

---

## 4. Off-Trajectory Generalization

scene graph 轨迹覆盖的是 expert states：

$$
s_t^{\text{traj}}=(I,Q,v_{<t}^{\text{traj}})
$$

OPD 训练访问的是 student states：

$$
s_t^\theta=(I,Q,\hat{y}_{<t})
$$

通常：

$$
s_t^\theta\neq s_t^{\text{traj}}
$$

因此 TCTR 的真正技术点是 sparse-to-dense generalization：

$$
s\sim\mathcal{D}_{\text{SG-traj}}
\quad\Longrightarrow\quad
s\sim d_{\pi_\theta}
$$

可以加入 consistency regularization：

$$
\mathcal{L}_{\text{cons}}
=
\mathbb{E}_{s,s'\in\mathcal{N}(s)}
\left[
\left(\lambda_\phi(s)-\lambda_\phi(s')\right)^2
\right]
$$

其中 $\mathcal{N}(s)$ 可以是相邻前缀、轻微扰动前缀、或 teacher-assisted rollout 产生的近邻 state。

---

## 5. Training Algorithm

```
Algorithm: Trajectory-Calibrated Teacher Routing

Input:
  Student π_θ
  Teachers π_P, π_R
  GQA scene graphs D_SG
  OPD training prompts D_opd
  Router r_φ

Stage A: Generate routing trajectories
  For each GQA sample (image, question, scene graph, answer):
    1. Parse required scene graph operations.
    2. Generate an interleaved skeleton:
       reason what to inspect -> verify visual evidence -> reason next constraint.
    3. Paraphrase skeleton with strong VLM / LLM.
    4. Preserve span-level soft operation priors μ_SG, c_SG.
    5. Filter by answer consistency and scene-graph faithfulness.

Stage B: Train router
  For each generated trajectory state s_t:
    1. Query π_P(·|s_t), π_R(·|s_t).
    2. Build feature z(s_t).
    3. Compute λ_t* from teacher support and soft operation prior.
    4. Optimize λ_φ(s_t) against λ_t*.

Stage C: Dense multi-teacher OPD
  For each prompt x in D_opd:
    1. Student rollout: ŷ ~ π_θ(·|x).
    2. For each student state s_t=(I,Q,ŷ_<t):
       a. Query π_P(·|s_t), π_R(·|s_t).
       b. Predict λ_φ(s_t)=r_φ(z(s_t)).
       c. Construct a_{λ_φ}(·|s_t).
       d. Update student with KL(π_θ(·|s_t) || a_{λ_φ}(·|s_t)).
```

---

## 6. 与相关路线的区别

### 6.1 相比 trajectory SFT

Trajectory SFT 直接拟合：

$$
\mathcal{L}_{\text{SFT}}=-\log\pi_\theta(\tau\mid x)
$$

TCTR 学的是：

$$
\text{structured-evidence trajectory}
\rightarrow
\text{teacher router}
\rightarrow
\text{dense supervision on student states}
$$

### 6.2 相比 ViGOS hard separation

ViGOS 依赖格式模板：

$$
\lambda_{\text{hard}}(s)=
\begin{cases}
1,& s\in\langle\text{description}\rangle\\
0,& s\in\langle\text{think}\rangle
\end{cases}
$$

TCTR 不依赖 `<description>` / `<think>` 分段，而是从 pseudo-optimal trajectory 中学习连续 routing weight。相邻 token 可以对应不同 $\lambda$，单个 span 也可以由两个 teacher 联合监督。

### 6.3 相比 fixed MOPD / confidence routing

Fixed MOPD 使用全局 $\alpha$。Confidence routing 主要依赖 entropy，但 teacher 可能 confidently wrong。TCTR 使用 pseudo-optimal trajectory 上的 teacher support 和 soft operation prior 校准 teacher mixture：

$$
\lambda_\phi(s)
=
f(
\text{state representation},
\text{teacher support for pseudo-optimal tokens},
\text{teacher disagreement},
\text{soft trajectory prior}
)
$$

### 6.4 相比只用 scene graph 做 evaluation

常见做法是用 GQA scene graph 做 error attribution。TCTR 把 scene graph 变成 training signal：

$$
\text{scene graph}
\rightarrow
\text{pseudo-optimal trajectory}
\rightarrow
\text{router training}
\rightarrow
\text{dense OPD}
$$

---

## 7. 主要贡献

| 内容 | 定位 |
|------|------|
| Structured-evidence trajectories | 用 GQA scene graph 构造近似最优 reasoning trajectories |
| Soft teacher-mixture targets | 用 soft operation prior 和 teacher 对 pseudo-optimal token 的支持得到连续 $\lambda_t^*$ |
| Sparse-to-dense OPD | 将 trajectory anchors 泛化到 student rollout states |
| Conflict-aware routing | 在 teacher disagreement 高的 state 上学习更合适的 teacher mixture |
| Cross-dataset transfer | 用 GQA scene graph 学到的 router 迁移到无 scene graph 的 OK-VQA |

---

## 8. 实验设计

### 8.1 数据设置

```
Trajectory source:
  GQA scene graph -> generated reasoning trajectories
  pilot scale: 1K, 5K
  scaling ratios: 1%, 5%, 10%

Dense OPD training:
  GQA remaining split
  optional mixed VQA prompts without trajectories

Evaluation:
  GQA held-out
  GQA compositional / high-hop subset
  OK-VQA without OK-VQA scene graphs or trajectories
```

### 8.2 Baselines

```
Method                              GQA held-out   GQA-Compose   OK-VQA   Evidence-Wrong ↓
──────────────────────────────────────────────────────────────────────────────────────────
Base student                         xx.x           xx.x          xx.x     xx.x
Scene-graph trajectory SFT            xx.x           xx.x          xx.x     xx.x
Vanilla OPD                           xx.x           xx.x          xx.x     xx.x
Fixed-weight MOPD                     xx.x           xx.x          xx.x     xx.x
Confidence / entropy routing          xx.x           xx.x          xx.x     xx.x
Disagreement routing                  xx.x           xx.x          xx.x     xx.x
DOPD-style advantage-gap routing       xx.x           xx.x          xx.x     xx.x
ViGOS-style hard separation            xx.x           xx.x          xx.x     xx.x
DOPD                                  xx.x           xx.x          xx.x     xx.x
TCTR (ours)                           xx.x           xx.x          xx.x     xx.x
TCTR + DOPD                            xx.x           xx.x          xx.x     xx.x
  - w/o soft operation prior             xx.x           xx.x          xx.x     xx.x
  - w/o disagreement features           xx.x           xx.x          xx.x     xx.x
  - binarized routing                    xx.x           xx.x          xx.x     xx.x
  - no off-trajectory regularization    xx.x           xx.x          xx.x     xx.x
```

### 8.3 必须展示的实验信号

1. **Free trajectory construction**：展示 scene graph 自动生成 trajectory 的质量、通过率和规模。
2. **Data efficiency**：1K / 5K / 1% / 5% GQA scene-graph trajectories 下，TCTR 优于 trajectory SFT。
3. **Free heuristic gate**：TCTR 必须优于 entropy routing、disagreement routing、DOPD-style advantage-gap routing。只讨论“confident $\neq$ correct”不够，必须用 routing diagnostic 证明启发式确实会错。
4. **Dense OPD gain**：TCTR 优于 fixed MOPD、confidence routing、ViGOS-style hard separation 和 DOPD；最好证明 TCTR + DOPD 进一步优于 DOPD。
5. **Off-trajectory generalization**：student 偏离 expert trajectory 后，routing 仍然有效。
6. **Cross-dataset transfer**：不使用 OK-VQA scene graph / trajectory，TCTR 在 OK-VQA 上仍有收益。
7. **Conflict-region gain**：teacher disagreement 高的 token 上收益更大。
8. **Teacher-pair transfer**：换一组 perception / reasoning teacher 后，router 或 routing features 仍有部分可迁移性。

### 8.4 Routing diagnostic

在完整 OPD 训练前，先做低成本 routing diagnostic。用 GQA scene graph 生成近似最优轨迹，再由 soft prior + teacher support 得到连续 $\lambda_t^*$，比较免费启发式和 learned router：

| Method | Overall routing acc | High-conflict acc | Relation/comparison acc |
|--------|---------------------|-------------------|-------------------------|
| Entropy routing | xx.x | xx.x | xx.x |
| Disagreement routing | xx.x | xx.x | xx.x |
| DOPD-style advantage-gap routing | xx.x | xx.x | xx.x |
| TCTR router | xx.x | xx.x | xx.x |

如果 TCTR 不能明显优于这些无需训练、无需 scene graph 的启发式，方法必要性会明显下降。

### 8.5 Trajectory quality evaluation

自动生成 trajectory 需要单独评估：

| 指标 | 目的 |
|------|------|
| Answer consistency | 生成轨迹是否导向 GQA answer |
| Scene-graph faithfulness | 轨迹是否只使用图中存在的 object / attribute / relation |
| Naturalness | paraphrased trajectory 是否接近真实 VLM 推理风格 |
| Routing distribution | 连续 $\lambda_t^*$ 是否不过度塌缩到 0 或 1 |
| Teacher agreement | teacher likelihood 是否极端反对生成轨迹 |

建议先做 1K-5K pilot：

```
1. scene graph template generation
2. Qwen2.5-VL-72B / strong VLM paraphrasing
3. automatic filtering
4. human spot-check 100 examples
5. train router and test GQA held-out / OK-VQA
```

### 8.6 Off-trajectory evaluation

```
For each GQA trajectory:
  1. Let student roll out from the same prompt.
  2. Measure prefix divergence from generated trajectory.
  3. Bucket states by divergence distance: 0, 1-5, 6-10, 10+ tokens.
  4. Evaluate routing quality / teacher agreement / final accuracy per bucket.
```

关键结果：

$$
\text{TCTR gain at divergence}>0
$$

### 8.7 GQA scene graph error attribution

GQA scene graph 还能支持 error decomposition：

```
Perception error:
  wrong object / attribute / relation grounding

Reasoning error:
  wrong comparison / logical composition / answer aggregation

Evidence-wrong:
  answer correct but visual evidence chain wrong
```

TCTR 应该在 evidence-wrong 和 teacher-conflict token 上降低错误率。

---

## 9. 风险与边界

1. **Trajectory quality**：scene graph 可以免费生成轨迹，但模板轨迹可能不自然。需要 paraphrasing、filtering 和人工抽检。

2. **Routing triviality**：单 token likelihood 的 $\lambda$ 优化本身很简单，论文贡献必须放在 soft operation prior、teacher support refinement 和 sparse-to-dense routing。

3. **Free heuristic risk**：entropy、disagreement、advantage-gap routing 几乎免费。如果它们接近 TCTR，scene graph router 的必要性不足。

4. **DOPD / MoCA competition**：DOPD 已经做 token-level adaptive OPD routing，MoCA 已经做 perception-reasoning credit assignment。TCTR 必须定位为 trajectory-calibrated multi-teacher OPD routing，并把 DOPD / MoCA 作为相关工作和实验对照。

5. **Transfer risk**：GQA 主要覆盖 visual relational reasoning，迁移到 OK-VQA 需要证明 router 学到的是 teacher-utility pattern，而不是 scene graph 模板。

6. **Teacher-pair dependence**：router 可能过拟合某一对 teachers，需要 teacher-pair transfer 或 feature-level ablation 支撑。

7. **Generation bias**：paraphrasing 模型可能引入自己的推理风格。需要比较 template-only、paraphrased、teacher-generated 三种 trajectory。

---

## 10. CVPR Positioning

这个版本的优势不在复杂数学，而在资源利用和问题设置：

> Existing structured evidence, instantiated with GQA scene graphs in this work, can be converted into pseudo-optimal trajectories for calibrating continuous teacher-mixture targets in multi-teacher OPD.

如果实验成立，claim 比“少量外部 expert trajectory”更强，因为 trajectory cost 从弱点变成优势。

当前 CVPR 概率取决于三个 gate：

| Gate | 通过条件 | 影响 |
|------|----------|------|
| Heuristic gate | TCTR 明显优于 entropy / disagreement / advantage-gap routing | 决定方法必要性 |
| DOPD gate | TCTR + DOPD 优于 DOPD | 证明与强 OPD routing 互补 |
| Transfer gate | 不用 OK-VQA scene graph 仍提升 OK-VQA | 证明不是 GQA 模板过拟合 |

概率估计：

- 三个 gate 都通过：$55\%-65\%$
- 只通过 routing diagnostic 和 GQA，下游 transfer 一般：$35\%-45\%$
- 免费启发式接近 TCTR：$20\%-30\%$

---

*Last updated: 2026-07-11*
