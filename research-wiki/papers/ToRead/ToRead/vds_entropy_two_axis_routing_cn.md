# 双轴教师路由：Divergence × Confidence — 格式无关的多教师 VLM 蒸馏

> CoT section 标签用一个信号回答两个正交问题。Teacher-teacher divergence 回答其中一个，teacher entropy 回答另一个。二者结合使 CoT 格式对教师路由变得无关。

---

## 核心论点

**CoT section 标签试图用粗糙的文本启发式回答"哪个教师相关？"。它之所以失败，是因为：(1) section 边界与 token 级的教师差异不对齐；(2) 完全忽略了第二个同等重要的问题——"相关教师是否确信？"**

我们提出一个将这两个问题解耦的双轴框架：

- **轴 1（相关性, Relevance）**：Teacher-teacher divergence — 视觉教师在这个 token 上有多少独特信息？
- **轴 2（置信度, Confidence）**：Teacher entropy — 每个教师的预测是否可靠？

两个轴的乘积 — **Divergence × Confidence Product（DCP）** — 提供了对 section-based 教师分配的 principled、格式无关的替代方案。在此框架下，CoT 格式对蒸馏质量无关：路由信号来自教师行为，而非文本结构。无需额外 forward。无学习参数。每步一次 backward。

---

## 1. 问题：一个信号无法回答两个问题

### 1.1 Section label 试图做什么

Staged CoT 的 section-based 教师分配隐含了一个断言：

> 如果 token 在 `<perception>` section 内，视觉教师相关，应该监督它。

这个断言混淆了两个不同的问题：

| 问题 | Section label 的假设 | 现实 |
|------|---------------------|------|
| **Q1: 视觉教师有独特信息吗？** | 是的，对所有 perception token | Teacher divergence 在同一 section 内从 0（虚词）到高（内容词）连续变化 |
| **Q2: 视觉教师确信吗？** | 完全不考虑 | 视觉教师 entropy 在 token 间变化，即使 divergence 很高 |

Section label 是 Q1 的 **1-bit proxy，完全忽略 Q2**。它基于语义约定而非教师实际知识来做路由。

### 1.2 为什么现有方法也都混淆

所有单信号 token 级教师路由方法都有同样的混淆问题：

| 方法 | 信号 | 回答 Q1？ | 回答 Q2？ |
|------|------|-----------|-----------|
| Section labels | 文本结构 | 粗糙（1-bit） | ❌ |
| Entropy ratio | H(π_R) / (H_R+H_P) | ❌（entropy ≠ relevance） | ✅ |
| Teacher KL alone | KL(π_P ‖ π_R) | ✅（divergence = 独特信息） | ❌ |
| Gradient cos | cos(g_P, g_R) | ❌ | ❌（测量冲突，不是置信度） |

**没有任何单一标量能同时捕捉"哪个教师有独特信息"和"该教师是否确信"。** 这两个是正交的维度。

---

## 2. 为什么用 Teacher-Teacher Divergence 而非 Decomposed-Style VDS

### 2.1 DecomposedOPD 的非对称性（单教师）

DecomposedOPD 通过 Bayes 规则将单教师的预测分解：

$$
\log \pi_T(\tau \mid I, x) = \underbrace{\log \pi_T(\tau \mid x)}_{\text{Language Prior}} + \underbrace{\log \pi_T(I \mid \tau, x)}_{\text{Visual Likelihood}} - \underbrace{\log \pi_T(I \mid x)}_{\text{Evidence（常数）}}
$$

关键机制在于视觉 loss 的 target 分布 $q_T^*$ 的构造：

$$
q_T^*(\tau \mid I, x) \propto \underbrace{p_{\theta_S}(\tau \mid x)}_{\text{学生的语言 prior}} \cdot \underbrace{q_T(I \mid \tau, x)}_{\text{教师的视觉似然}}
$$

**$q_T^*$ 以学生自己的语言 prior 为底座，仅嫁接教师的视觉信息增益。** $q_T^*$ 中与学生自身分布唯一不同的就是教师的视觉 prior——其余全是学生自己的。这种干净的隔离意味着 $\mathcal{L}_{\text{Vis}} = \text{KL}(p_S \parallel q_T^*)$ 只传递视觉能力，不污染学生的语言风格。

相比之下，$\mathcal{L}_{\text{Lang}} = \text{KL}(p_S(\cdot|\text{text}) \parallel q_T(\cdot|\text{text}))$ 传递教师的语言 prior——这是对学生**已有能力**的语言建模的精炼，而不是新能力。

这就是 DecomposedOPD 利用的根本非对称性：

| Component | 传递什么 | 性质 |
|-----------|---------|------|
| $\mathcal{L}_{\text{Vis}}$（通过 $q_T^*$） | 教师的视觉 prior | **新能力**（学生不擅长图像解释） |
| $\mathcal{L}_{\text{Lang}}$ | 教师的语言 prior | **精炼**（学生已经会语言建模） |

→ VGS 偏向 $\mathcal{L}_{\text{Vis}}$ 是因为视觉 grounding 是瓶颈，且 $q_T^*$ 提供了干净、不污染的视觉信号。天然地 visual > language。

### 2.2 DCP 的非对称性（双教师）

在 DCP 中，两个教师都看到图像，且都从同一个 base VLM 微调而来。它们的语言 prior 几乎相同——相同的架构、相同的预训练、相同的基本多模态知识。**唯一的系统性差异**来自 RL 专业化：

| 教师 | RL 训练重点 | 专长 |
|------|------------|------|
| π_P（视觉） | Vision-heavy RL（VQA, grounding, spatial） | 物体属性、空间关系、计数 |
| π_R（推理） | Reasoning-heavy RL（math, logic, planning） | 演绎、比较、多步推理 |

与 DecomposedOPD 不同——那里只有一个方向（视觉）提供真正的新能力——**DCP 中两个教师都提供新能力**：π_P 提供学生缺乏的视觉 grounding 专长，π_R 提供学生缺乏的推理专长。因此路由问题根本不同：不是"偏向视觉"，而是"在每个 token 上识别需要哪种专长"。

### 2.3 正确的 relevance 度量：Teacher-teacher divergence

由于两个教师共享同一个 base VLM，它们的语言 prior 几乎相同。在纯语言 token（如 "the"、"因此"）上，两个教师从相同的语言分布做预测 → KL(π_P ‖ π_R) ≈ 0。在视觉依赖 token（如 "红色"、"椅子"）上，π_P 经过 vision-specialized 训练后，其预测相对于 π_R 的 base-level 视觉能力发生了变化 → KL(π_P ‖ π_R) > 0。

**KL(π_P ‖ π_R) 因此隔离了专长差距（specialization gap）——具体来说，是 π_P 相对于 π_R 增强的视觉感知能力。** 它不是在测量泛泛的 "disagreement"；它测量的是专业化 RL 训练导致 π_P 的预测与 π_R 发生偏离的位置。由于共享 base，这种偏离的首要系统性来源就是视觉专业化。

$$
\boxed{D_t = \frac{D_{\text{KL}}(\pi_P(\cdot|s_t) \;\|\; \pi_R(\cdot|s_t))}{D_{\text{KL}}(\pi_P(\cdot|s_t) \;\|\; \pi_R(\cdot|s_t)) + \tau}}
$$

- $D_t \to 1$：π_P 的视觉专业化显著改变了其预测 → **视觉教师有独特信息** → 视觉教师相关
- $D_t \to 0$：教师预测相同（共享 base 语言 prior 主导）→ 无专长差距 → **任一教师都可以**

**为什么不用 Decomposed-style VDS？**

| | KL(π_R(·\|I) ‖ π_R(·\|text)) | KL(π_P ‖ π_R) |
|---|---|---|
| 回答的问题 | "这个 token 需要看图吗？" | "π_P 的视觉专长在哪里比 π_R 更强？" |
| 最适合 | 单教师分解 | **多教师路由** |
| 额外计算 | 需要 text-only forward | **无（从已有 forward 计算）** |
| 信息来源 | 一个模型 ± 图像 | **两个不同专业化的模型** |

DecomposedOPD 的 VDS 回答"图像改变了 π_R 的预测多少？"——分解一个教师时是正确的。DCP 回答"π_P 的视觉专长在哪里超过 π_R？"——在两个专业化教师之间路由时是正确的。

### 2.4 与 DecomposedOPD 的关联

两个度量——DecomposedOPD 的 VDS 和我们的 $D_t$——测量相关但不同的事物：

| | DecomposedOPD VDS | 我们的 $D_t$ |
|---|---|---|
| 测量什么 | 图像对单一模型的影响 | 两个模型间的专长差距 |
| 公式 | KL(π_R(·\|I) ‖ π_R(·\|text)) | KL(π_P(·\|I) ‖ π_R(·\|I)) |
| 额外计算 | 需要 text-only forward | **无（从已有 forward 计算）** |
| 最适合 | 单教师分解 | **多教师路由** |

它们可能相关（高 VDS token 倾向于有高 teacher divergence），但 $D_t$ 还捕获了 π_P 和 π_R 因专业化训练导致的归纳偏置差异——这些超出了纯粹视觉依赖的范畴。

---

## 3. 双轴分解

### 3.1 轴 1：相关性 — Teacher Divergence

$$
D_t = \frac{D_{\text{KL}}(\pi_P \| \pi_R)}{D_{\text{KL}}(\pi_P \| \pi_R) + \tau}
$$

$\tau$ 是控制 sigmoid 陡峭程度的温度。默认 $\tau=0.5$（nats）。含义：当两个教师相差 0.5 nats 时，relevance 为 50%。这是 DCP 唯一的超参数——且有清晰的语义解释。

- $D_t$ 高：π_P 和 π_R 给出显著不同的 next-token 分布 → 视觉教师有独特信息
- $D_t$ 低：教师预测一致 → 视觉教师没有独特信息

### 3.2 轴 2：置信度 — Teacher Entropy

$$
\bar{H}_P = \frac{H(\pi_P(\cdot|s_t))}{\log |V|}, \qquad \bar{H}_R = \frac{H(\pi_R(\cdot|s_t))}{\log |V|}
$$

- 低 $\bar{H}$ → 概率质量集中 → 教师确信 → 预测可靠
- 高 $\bar{H}$ → 分布平坦 → 教师不确定 → 预测不可靠

**为什么置信度独立于相关性**：一个教师可以有高度相关的信息（D_t 高：独特的视觉知识）但低置信度（H̄_P 高：图像模糊）。这种情况下，盲目信任视觉教师会注入噪声。置信度轴对这种情况进行降权。

### 3.3 为什么需要双轴而非单轴

Section label 是 divergence 轴的 **1-bit 量化**——且完全遗漏了 confidence 轴：

```
Section label 视角：
  Perception section → "divergence 高" → 分配视觉教师
  Reasoning section  → "divergence 低" → 分配语言教师

现实：
  Perception 内部：divergence 从 0 到高连续变化
  Reasoning 内部：divergence 大多低，但会突增
  两个 section 内：teacher confidence 独立于 divergence 变化
```

双轴框架捕获了 section label 遗漏的东西：
- 高 divergence 但视觉教师不确定 → 虽相关但降权
- 低 divergence 但语言教师不确定 → 虽假定足够但降权
- 两个轴互相制约 → 监督质量 = relevance × confidence

---

## 4. 四个 Regime

Divergence × Confidence 的乘积将 token 分为四个自然可解释的 regime：

```
                    高 Confidence              低 Confidence
                    ─────────────              ─────────────
高 D_t     │  REGIME I: 强视觉              │  REGIME II: 弱视觉
(视觉教师   │  D_t↑, H_P↓                   │  D_t↑, H_P↑
有独特信息) │  → 完全视觉监督               │  → 谨慎视觉监督
           │  如："红色"、"椅子"            │  如：模糊的物体

低 D_t     │  REGIME III: 强语言            │  REGIME IV: 弱语言
(教师预测   │  D_t↓, H_R↓                   │  D_t↓, H_R↑
 一致)     │  → 完全语言监督               │  → 谨慎语言监督
           │  如："因此"、"the"            │  如：罕见推理步骤
```

### 每个 regime 的意义

**Regime I（高 divergence，视觉确信）**：视觉教师有独特信息且确信。视觉蒸馏的核心场景。这是 section label 试图捕获的情况——物体名称、属性、空间关系。

**Regime II（高 divergence，视觉不确定）**：视觉教师知道语言教师不知道的东西，但自己也不确定——被遮挡的物体、细粒度区分、模糊场景。单轴路由（仅 KL）会在这里给全权重。DCP 降权以防止视觉教师自信地给出错误答案。**这个 regime 对 section label 和单轴方法完全不可见。**

**Regime III（低 divergence，语言确信）**：教师一致且语言教师确信。纯推理 token。Section label 处理正确——但 DCP 也能自动处理，无需 section label。

**Regime IV（低 divergence，语言不确定）**：教师一致但都不确信——罕见逻辑模式、模糊推理步骤。降权防止过拟合教师的不确定性。**对 section label 同样不可见。**

**Section label 只能区分 I+II 和 III+IV——且即使在 section 内部这个区分也是有噪声的。** I vs II 和 III vs IV 的区分被完全遗漏。

---

## 5. 方法：Divergence-Confidence Product Routing（DCP）

### 5.1 每个教师的权重

$$
w_P = D_t \cdot (1 - \bar{H}_P), \qquad w_R = (1 - D_t) \cdot (1 - \bar{H}_R)
$$

直觉：视觉权重 = （视觉教师有独特信息吗？）×（它确信吗？）。语言权重 = （语言教师足够吗？）×（它确信吗？）。

### 5.2 归一化路由权重

$$
\boxed{\alpha_t = \frac{D_t \cdot (1 - \bar{H}_P)}{D_t \cdot (1 - \bar{H}_P) + (1 - D_t) \cdot (1 - \bar{H}_R)}}
$$

### 5.3 蒸馏 Loss

$$
\boxed{\mathcal{L} = \sum_t \left[\alpha_t \cdot \text{KL}(\pi_\theta \| \pi_P) + (1-\alpha_t) \cdot \text{KL}(\pi_\theta \| \pi_R)\right]}
$$

**一次 forward，一次 backward。零额外 forward。零学习参数。一个超参数（$\tau$，有语义含义）。**

### 5.4 算法

```
每步训练：
  1. Student rollout: ŷ ~ π_θ(·|I, Q)，使用 Mixed CoT prompt
  
  2. Teacher forward（带图像）— 执行一次，共享：
     p_P = π_P(·|I, Q, ŷ_<t)     # 视觉教师
     p_R = π_R(·|I, Q, ŷ_<t)     # 推理教师
  
  3. 从 p_P, p_R 计算路由信号（无需额外 forward）：
     D_t = KL(p_P || p_R) / (KL(p_P || p_R) + τ)    # divergence
     H̄_P = H(p_P) / log|V|                            # 视觉置信度
     H̄_R = H(p_R) / log|V|                            # 语言置信度
  
  4. 路由权重：
     α_t = D_t·(1-H̄_P) / [D_t·(1-H̄_P) + (1-D_t)·(1-H̄_R)]
  
  5. Loss 和更新：
     L = Σ_t [α_t·KL(π_θ||p_P) + (1-α_t)·KL(π_θ||p_R)]
     L.backward()  # 单次 backward
```

### 5.5 计算开销

| 方法 | Forward 次数 | Backward 次数 | 额外参数 |
|------|-------------|--------------|---------|
| Uniform MOPD | 1 | 1 | 0 |
| Staged + section assignment | 1 | 1 | 0 |
| PCGrad | 1 | **2** | 0 |
| DCP（Ours） | 1 | **1** | 0 |
| Learned router（TCTR） | 1 | 1 | ~1M |

DCP 的计算开销与 uniform MOPD 和 section-based assignment 相同。唯一的额外成本是从教师 log-probabilities 计算 KL、entropy 和 α_t 的逐元素运算——与 forward/backward 相比可忽略不计。

---

## 6. 为什么 CoT 格式变得无关

### 6.1 路由信号是格式无关的

$D_t$ 和 teacher entropy 从**教师模型输出**计算，不从**文本结构**计算。它们依赖：
- 图像内容
- 学生生成的 prefix $\hat{y}_{<t}$
- 教师模型的参数

这些都不依赖于文本是否写 `<perception>` 或是交错视觉和语言推理。

### 6.2 Mixed CoT 现在成为自然选择

在 section-based assignment 下，Staged CoT 是"必要的"因为路由需要 section label。在 DCP 路由下：

- **Mixed CoT 原生可用**：无需 section label。路由在每个 token 上由 Divergence × Confidence 处理。
- **Staged CoT 的 section label 变成死重**：它不能提供教师行为之外的额外路由信息——且完全丢失了 confidence 轴。
- **CoT 格式选择退化为效率问题**：Mixed CoT 使用 ~33% 更少的 token 且准确率相当 → Mixed CoT 占优。

### 6.3 Section label 严格更差

| Section label 提供什么 | DCP 提供什么 |
|---|---|
| 二值路由（perception → visual, reasoning → language） | 基于教师行为的连续路由 |
| 无置信度调制 | 通过 teacher entropy 进行置信度门控 |
| 需要显式格式结构 | 格式无关 |
| divergence 轴的 1-bit 近似 | 完整的 divergence 计算 |
| 忽略 Regime II 和 IV | 处理全部四个 regime |

**Section label 不仅不必要——它丢失信息。** 在教师行为信号可用时使用 section label，就像在有连续 logits 时使用二值分类器。

---

## 7. 相关工作

### 7.1 DecomposedOPD（ICML 2026 Spotlight）

| | DecomposedOPD | DCP（Ours） |
|---|---|---|
| 教师 | 单教师，分解 | **两个独立教师** |
| Relevance 度量 | VDS（同一模型 ± 图像） | **Teacher-teacher divergence** |
| 问题 | OPD 中视觉 component 权重不足 | **哪个教师监督哪个 token** |
| 额外计算 | Text-only forward for VDS | **零额外计算** |
| CoT 格式 | 未讨论 | **核心论点** |

**关键区别**：DecomposedOPD 将一个教师分解为语言和视觉 components，并偏向视觉 component（因为只有它有独特的图像信息）。在 DCP 中，两个教师都有独特信息——视觉教师在 perception token 上，推理教师在 logic token 上。路由问题根本不同。

**关联**：两者共享"教师质量非对称"的洞察——并非所有监督信号同等有价值。DecomposedOPD 在单教师内部解决这个问题；DCP 跨两个教师解决。双轴框架（relevance × confidence）可视为泛化：DecomposedOPD 聚焦于 relevance 轴（VDS），而 DCP 展示了 confidence 轴在多教师场景中同样重要。

### 7.2 基于 Entropy 的门控方法（EGRSD, SEAD, CAKD, DE-MKD）

都使用教师和/或学生 entropy 来确定**是否**在某个 token 做蒸馏（或**多少**）。全部是**单教师**方法。DE-MKD 使用 entropy ratio 做多教师加权，但是**样本级**（图像分类），不是 token 级。

**DCP 的区别**：(1) 多教师 **token 级**路由（哪个教师），而非单教师门控（是否蒸馏）。(2) 第二轴（divergence）至关重要——单独的 entropy 混淆了 relevance 和 confidence。

### 7.3 Teacher disagreement 方法（UniKD, EWAD, RAPS-DA）

使用 teacher-teacher KL/JSD 作为路由或加权信号。但 teacher-teacher divergence 被用作**单一标量**——混淆了 relevance 和 confidence。

**DCP 的区别**：DCP 显式解耦 divergence（relevance）和 entropy（confidence）。这是核心概念贡献——不是信号本身，而是认识到它们回答不同的问题。

### 7.4 梯度空间方法（PCGrad, AE-KD, ATTITTUD）

在梯度空间操作，有 2× backward 开销。解决冲突，不是路由。

**DCP 的区别**：在分布空间操作，1× backward。解决路由（哪个教师对哪个 token），不是冲突解决。

### 7.5 Learned routers（TCTR, COMPACT, VAMOPD）

学习一个路由器网络 $\lambda_\phi(s_t)$ 来预测每个 token 的教师权重。

**DCP 的区别**：零学习参数。权重是**计算**出来的（来自教师行为），不是**学习**出来的（来自数据）。这消除了 off-trajectory generalization 问题和路由器训练数据的需求。路由也是可解释的——可以检查任何 token 上 α_t 为什么高或低。

---

## 8. 实验设计

### 8.1 核心待验证声明

> Divergence × Confidence 路由优于 section-based 教师分配，且使 CoT 格式对蒸馏质量无关。

### 8.2 主要实验

| 实验 | 证明什么 |
|------|---------|
| **Mixed CoT + DCP vs. Staged CoT + section assignment** | DCP + Mixed CoT ≥ section assignment + Staged CoT → 格式无关，DCP 更好 |
| **Mixed CoT + DCP vs. Mixed CoT + uniform MOPD** | DCP 相对朴素多教师平均有增益 |
| **Staged CoT + DCP vs. Staged CoT + section assignment** | DCP 在 Staged CoT 内也能改善（section label 在自己的"主场"也输） |
| **DCP vs. divergence-only routing** | Ablation：confidence 轴有意义 |
| **DCP vs. entropy-only routing** | Ablation：relevance 轴有意义 |
| **DCP: product vs. sum vs. max** | Ablation：乘积形式合理 |
| **DCP vs. learned router（TCTR-style）** | DCP 以零参数逼近 learned router 性能 |
| **DCP（teacher-teacher KL）vs. DCP（Decomposed-style VDS）** | Ablation：teacher divergence 是多教师场景正确的 relevance 度量 |

### 8.3 诊断实验

#### 四 Regime 分析

```
对训练集中的每个 token：
  1. 计算 D_t, H̄_P, H̄_R
  2. 分类到 Regime I/II/III/IV
  3. 报告：
     - 各 regime 的 token 分布
     - DCP 在各 regime 的下游准确率提升
     - I > II 且 III > IV（confidence 调制有效）
     - I+III >> II+IV（relevance 调制有效）
     - 哪个 regime 从 DCP 获益最大（vs. uniform MOPD）？
```

#### Section label vs. teacher divergence 错位

```
Plot: x 轴 = Staged CoT 序列中的 token 位置
      y 轴 = D_t（teacher divergence）
      Color = teacher entropy（蓝 = confident, 红 = uncertain）
      Overlay: section 边界标记

关键观察：
  - D_t 连续变化；section 边界处无跳跃
  - Perception section 内存在高 entropy 的视觉 token（Regime II）
  - Section label 是连续信号的 1-bit 量化
```

#### Decomposed-style VDS 与 teacher divergence 的相关性

```
Scatter plot: x = DecomposedOPD VDS（π_R ± image）, y = D_t（π_P vs π_R）
  - 预期：正相关（视觉 token 在两者上都高）
  - 但 D_t 可能捕获纯视觉依赖之外的额外 divergence
  - 报告 Spearman ρ 和两者不一致的情况
```

### 8.4 Baselines

```
1.  Base Student（无蒸馏）
2.  单教师 OPD（仅视觉教师）
3.  单教师 OPD（仅推理教师）
4.  MOPD uniform（α = 0.5 固定）
5.  MOPD + section-based assignment（仅 Staged CoT）
6.  MOPD + divergence-only routing（α_t 仅来自 D_t）
7.  MOPD + entropy-only routing（α_t 来自 H ratio）
8.  MOPD + teacher-KL routing（Route 3: KL-based PoE mixture）
9.  MOPD + DCP（ours）
10. MOPD + DCP with Decomposed-style VDS（ablation）
11. MOPD + learned router（TCTR-style）
12. MOPD + Protective PCGrad
```

### 8.5 评估指标

- 下游任务准确率（VSR1, GQA, OK-VQA）
- 各 regime 准确率分解
- 训练效率（每步 wall time vs. 准确率增益）
- 路由锐度：各 token 间 $\text{std}(\alpha_t)$（越高 = 路由越果断）
- $|\alpha_t - 0.5| > 0.2$ 的 token 比例（有意义路由的 token）

---

## 9. 风险评估

### 9.1 已知风险

| 风险 | 严重度 | 缓解 |
|------|--------|------|
| **D_t 和 entropy 高度相关** — 乘积不增加信息 | 高 | 诊断：计算相关性。若 r > 0.7，confidence 轴可能冗余。但即便如此，四 regime 分类仍提供概念价值，框架可退化为 divergence-only routing |
| **Section assignment 并不更差** — DCP ≤ section assignment | 高 | Pilot 实验第一件事。若 section assignment 匹配 DCP，从"section label 有害"转向"section label 不必要，DCP 更简洁" |
| **教师 entropy 模式相同**（处处 H_P ≈ H_R） | 高 | Confidence 轴失去路由能力。退化为 divergence-only routing。仍贡献 divergence 轴（section label 遗漏的），但框架丢了一半 |
| **双轴 story 是"事后显而易见"** | 中 | 若 (a) 实验展示 nontrivial regime 分布，(b) 四 regime 分析揭示了单轴方法遗漏的模式，则可缓解 |
| **τ 超参数需要调优** | 低 | τ 有清晰语义（50% relevance 对应的 nats）。默认 0.5。实验中做 sensitivity 分析 |
| **教师构造** — 需要真正不同的教师 | 中 | 不同 RL 训练目标自然产生不同专长。Pilot 中验证 entropy 模式差异。若教师太相似，路由退化——但此时多教师 KD 本身值得怀疑 |

### 9.2 预期的审稿人回应

**Q："为什么 divergence × confidence 是正确的组合？为什么不学出来？"**

A：乘积形式有自然的概率解释：视觉教师应监督 token t 的概率 = P（有独特信息）× P（预测可靠）。任一条件不满足，概率归零。这是一个设计选择——但我们 ablation product vs. sum vs. max vs. learned combination。若 learned combination 显著优于 product，那会很有趣。但 product 是一个强零参数 baseline，有清晰语义。

**Q："如何构造视觉和推理教师？"**

A：从同一个 base VLM 出发，用 vision-heavy tasks（VQA, spatial reasoning, visual grounding）做 RL 微调 π_P，用 reasoning-heavy tasks（math, logic, planning）做 RL 微调 π_R。训练目标差异——而非 token 级标注——产生 DCP 利用的专业化。教师训练中**不使用** section label。

**Q："为什么用 teacher-teacher KL 而不是 DecomposedOPD 的 VDS？"**

A：DecomposedOPD 的 VDS 测量"图像改变了模型 X 的预测多少？"——适合比较同一模型 ± 图像。DCP 比较两个不同模型（π_P vs π_R）在同一多模态输入上的输出——自然的度量是它们的输出 divergence。我们将 Decomposed-style VDS 作为 ablation baseline。

**Q："为什么只有两个教师？K > 2 呢？"**

A：视觉-推理二分是 VLM 蒸馏中最基本和最常见的专业化。对 K > 2，框架扩展到 pairwise divergence 和 confidence scores，通过 softmax 归一化。两教师情况是最重要的场景，也是 section label 最常用的场景。K>2 扩展概念上直接，留作 future work。

### 9.3 Paper-ending 场景

1. **D_t 和 H_P/H_R 高度相关**（r > 0.85）→ confidence 轴冗余 → 框架退化为 divergence-only → 贡献为"teacher divergence > section labels"（更弱但仍是贡献）
2. **Section assignment baseline 击败 DCP** → 前提被实验证伪 → 当前形式的论文无法存在 → 需理解原因，可能 pivot
3. **教师 entropy 模式相同** → confidence 轴不提供路由信息 → DCP = divergence-only → 框架丢了一半 → 需调查为何教师未分化
4. **DCP = uniform MOPD**（处处 α_t ≈ 0.5）→ teacher divergence 永远不够大到产生果断路由 → 多教师 KD 无优于单教师 → 质疑整个 MOPD 动机

---

## 10. 论文定位

### 10.1 一句话

> CoT section label 将多教师路由的两个正交信号——teacher divergence 和 teacher confidence——混为一谈。解耦它们使 CoT 格式对蒸馏质量无关，零额外计算开销。

### 10.2 贡献总结

| # | 贡献 | 类型 |
|---|------|------|
| 1 | **双轴分解**：多教师路由需要回答两个独立问题——哪个教师有独特信息？（divergence）该信息是否可靠？（confidence）。所有现有工作将它们混为一个信号 | 概念 |
| 2 | **Divergence-Confidence Product（DCP）**：零参数、零额外计算开销的路由机制，结合 teacher-teacher divergence 和 per-teacher entropy | 方法 |
| 3 | **CoT 格式无关性**：在双轴路由下，CoT section 结构对教师分配不提供额外价值 → 格式选择退化为 token 效率 → Mixed CoT 占优 | 理论 |
| 4 | **四 Regime 分类**：分析 token 级教师监督质量的框架。揭示了 section-based 和单轴方法不可见的失败模式（Regime II：高 relevance + 低 confidence） | 分析 |

### 10.3 投稿方向

| 会议 | 概率 | 理由 |
|------|------|------|
| CVPR 2027 | 45-55% | VLM 蒸馏 + 视觉推理主题。双轴框架概念干净。需要强实验 |
| NeurIPS 2027 | 40-50% | 方法 + 分析。贡献层级合适 |
| ICLR 2028 | 35-45% | 偏向理论，我们的贡献更偏方法 |
| ICML 2027 | 30-40% | ICML 偏好更深的理论 |

---

## 11. 下一步

### 关键路径（commit 之前必须完成）

- [ ] **Pilot 1**：在 100 个 Mixed CoT student rollout 上计算 D_t, H_P, H_R。检查：
  - D_t 分布（是否偏离 0？）
  - D_t 和 teacher entropy 的相关性
  - 四 regime token 分布（Regime II 和 IV 是否存在？）
  - α_t 分布（路由是否偏离 0.5？）
- [ ] **Pilot 2**：教师差异化检查 — H_P 和 H_R 在关键 token 上是否有实质差异？若无，教师需要更多专业化。
- [ ] **Pilot 3**：小规模蒸馏（50 样本）— DCP vs. section assignment vs. uniform MOPD vs. divergence-only vs. entropy-only

### 若 Pilot 通过

- [ ] 完整实验矩阵（Section 8）
- [ ] 四 regime 诊断分析
- [ ] Section label vs. divergence 错位可视化
- [ ] 撰写 Introduction + Method + Experiments

---

*最后更新：2026-07-21*
