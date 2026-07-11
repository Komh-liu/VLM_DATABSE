# VaMOPD: Variational Mixture-of-Teachers for On-Policy Distillation

> 将 multi-teacher OPD 在 mixed-reasoning 场景下的监督冲突，形式化为一个变分推断问题：每个 token 位置存在隐变量 $\lambda(t)$ 控制 perception teacher 和 reasoning teacher 的混合权重，student 的训练目标是学习这个隐变量下的后验推断。

---

## 1. Problem Setup

### 1.1 Notation

考虑一个多模态输入 $(I, Q)$（图像 + 问题），student 模型自回归采样 rollout：

$$
\hat{y} = (\hat{y}_1, \ldots, \hat{y}_T) \sim \pi_\theta(\cdot \mid I, Q)
$$

两个专长 teacher：
- **Perception teacher** $\pi_P$：擅长视觉 grounding，在需要识别图像细节的 token 上分布更可靠
- **Reasoning teacher** $\pi_R$：擅长逻辑推理，在推理链收束的 token 上分布更可靠

### 1.2 Vanilla MOPD 的问题

Vanilla MOPD 在 student rollout 上用固定权重混合 teacher 信号：

$$
\mathcal{L}_{\text{MOPD}}(\theta) = \mathbb{E}_t\left[ \alpha \cdot D_{\text{KL}}(\pi_\theta(\cdot|t) \parallel \pi_P(\cdot|t)) + (1-\alpha) \cdot D_{\text{KL}}(\pi_\theta(\cdot|t) \parallel \pi_R(\cdot|t)) \right]
$$

其中 $\alpha \in [0,1]$ 是全局固定的，与 token 内容无关。

**问题**：在 mixed-reasoning 轨迹中，不同 token 需要不同 teacher 的监督。施加固定 $\alpha$ 会导致：

- 视觉 grounding token 上被 reasoning teacher 的噪声信号稀释
- 逻辑推理 token 上被 perception teacher 的无信息梯度分散
- 两个 teacher 梯度方向不一致的 token 上出现冲突

---

## 2. Variational Formulation

### 2.1 核心假设

在真实 mixed-reasoning 过程中，每个 token 存在一个**隐能力混合变量** $\lambda(t) \in [0,1]$：

- $\lambda(t) \to 1$：该 token 主要由 visual perception 能力支撑
- $\lambda(t) \to 0$：该 token 主要由 logical reasoning 能力支撑
- $\lambda(t) \approx 0.5$：真正 mixed，两种能力共同作用

我们假设真正的 target next-token distribution 是两 teacher 的 $\lambda(t)$-加权混合：

$$
\boxed{\pi^*(v \mid t) = \lambda(t) \cdot \pi_P(v \mid t) + (1 - \lambda(t)) \cdot \pi_R(v \mid t)}
$$

> **关键洞察**：$\lambda(t)$ 是**不可直接观测的隐变量**。如果 student 能推断 $\lambda(t)$，就能在每个 token 上自适应地混合 teacher 信号。

### 2.2 变分目标

对于一个 token 位置 $t$（为简洁省略条件 $(I, Q, \hat{y}_{<t})$），student 的目标是逼近 unknown target $\pi^*$：

$$
\mathcal{L}_t(\theta) = D_{\text{KL}}\left(\pi_\theta(\cdot) \parallel \pi^*(\cdot)\right)
$$

由于 $\pi^*$ 包含隐变量 $\lambda(t)$，直接优化不可行。我们引入**变分分布** $q(\lambda \mid t)$，导出 Evidence Lower Bound (ELBO)：

$$
\begin{aligned}
\log \pi_\theta(v \mid t) &\geq \mathbb{E}_{q(\lambda|t)}\left[ \log\left( \lambda \cdot \pi_P(v|t) + (1-\lambda) \cdot \pi_R(v|t) \right) \right] \\
&\quad - D_{\text{KL}}\left( q(\lambda|t) \parallel p(\lambda|t) \right)
\end{aligned}
$$

其中 $p(\lambda|t)$ 是 $\lambda$ 的先验分布。

### 2.3 Per-Token Student Objective

最大化 ELBO 等价于最小化：

$$
\boxed{
\mathcal{L}_{\text{VaMOPD}}(\theta) = \mathbb{E}_{t, \hat{y}\sim\pi_\theta}\left[ \min_{q(\lambda|t)} \mathcal{J}_t(\theta, q) \right]
}
$$

其中 per-token objective：

$$
\mathcal{J}_t(\theta, q) = \underbrace{D_{\text{KL}}\left(\pi_\theta(\cdot|t) \parallel \lambda_q \cdot \pi_P(\cdot|t) + (1-\lambda_q) \cdot \pi_R(\cdot|t)\right)}_{\text{distillation loss}}
+ \beta \cdot \underbrace{D_{\text{KL}}\left(q(\lambda|t) \parallel p(\lambda|t)\right)}_{\text{regularization}}
$$

这里 $\lambda_q = \mathbb{E}_{q(\lambda|t)}[\lambda]$ 是后验期望混合权重。

---

## 3. Inference of Mixing Weights

### 3.1 封闭形式的 λ 推断

在每一步训练中，给定 student 当前分布 $\pi_\theta$ 和两 teacher 分布 $\pi_P, \pi_R$，最优 $\lambda$ 是以下一维优化问题的解：

$$
\lambda^*(t) = \arg\min_{\lambda \in [0,1]} D_{\text{KL}}\left( \pi_\theta(\cdot|t) \parallel \lambda \cdot \pi_P(\cdot|t) + (1-\lambda) \cdot \pi_R(\cdot|t) \right)
$$

**定理 1（Identifiability）**：若 $\pi_P(\cdot|t) \neq \pi_R(\cdot|t)$ 且 $\pi_\theta(\cdot|t)$ 不在二者连线的外推方向上，则 $\lambda^*(t)$ 是唯一的。

*证明概要*：$f(\lambda) = D_{\text{KL}}(\pi_\theta \parallel \lambda \pi_P + (1-\lambda) \pi_R)$ 在 $[0,1]$ 上是严格凸函数（KL 散度对于混合分布的凸性），因此存在唯一极小值点。闭区间上严格凸函数的极小值要么在内部驻点，要么在边界。$\square$

**定理 2（梯度形式的解）**：$\lambda^*(t)$ 满足不动点条件：

$$
\lambda = \frac{\mathbb{E}_{v \sim \pi_\theta}\left[ w_P(v, \lambda) \right]}{\mathbb{E}_{v \sim \pi_\theta}\left[ w_P(v, \lambda) + w_R(v, \lambda) \right]}
$$

其中 $w_P(v, \lambda) = \frac{\lambda \pi_P(v)}{\lambda \pi_P(v) + (1-\lambda) \pi_R(v)}$ 是 token $v$ 来自 perception teacher 的后验概率。

*推导*：

$$
\begin{aligned}
\frac{\partial}{\partial \lambda} D_{\text{KL}}(\pi_\theta \parallel \lambda \pi_P + (1-\lambda) \pi_R) &= 0 \\
\sum_v \pi_\theta(v) \cdot \frac{\pi_P(v) - \pi_R(v)}{\lambda \pi_P(v) + (1-\lambda) \pi_R(v)} &= 0 \\
\sum_v \pi_\theta(v) \cdot \frac{1}{\lambda + (1-\lambda) \frac{\pi_R(v)}{\pi_P(v)}} &= \sum_v \pi_\theta(v) \cdot \frac{1}{(1-\lambda) + \lambda \frac{\pi_P(v)}{\pi_R(v)}}
\end{aligned}
$$

该不动点方程可通过少量（通常 10-20 步）二分搜索或 EM 迭代求解，开销远小于一次完整的 forward-backward pass。

### 3.2 Fisher Information 先验

$\lambda(t)$ 的先验应编码"teacher 在该 token 上的确信度"这一先验知识。我们用 teacher 的 token-level Fisher Information：

$$
\begin{aligned}
\mathcal{I}_P(t) &\triangleq \mathbb{E}_{v \sim \pi_P}\left[ \|\nabla_{\text{logits}} \log \pi_P(v \mid t)\|^2 \right] \\
&= \sum_v \pi_P(v) \cdot (1 - \pi_P(v))^2 + \sum_{v \neq u} \pi_P(v) \cdot \pi_P(u)^2 \\
&= 1 - \sum_v \pi_P(v)^2
\end{aligned}
$$

$1 - \sum_v \pi(v)^2$ 是 Gini 不纯度——分布越 peaked（entropy 越低），这个值越大，teacher 在该 token 上越"确信"。

**Fisher 驱动的先验**：

$$
p(\lambda \mid t) = \text{Beta}\left(\lambda \;\middle|\; \alpha_0 \cdot \frac{\mathcal{I}_P(t)}{\mathcal{I}_P(t) + \mathcal{I}_R(t)} + 1,\; \alpha_0 \cdot \frac{\mathcal{I}_R(t)}{\mathcal{I}_P(t) + \mathcal{I}_R(t)} + 1 \right)
$$

其中 $\alpha_0$ 控制先验强度（$\alpha_0 = 0$ 退化为均匀先验）。

**直觉**：若 $\mathcal{I}_P(t) \gg \mathcal{I}_R(t)$，说明 perception teacher 在该 token 上有更强的分布峰 → 先验偏向 $\lambda \to 1$ → 变分后验围绕先验微调。

---

## 4. Joint Training Algorithm

### 4.1 EM-Style Alternating Optimization

```
Algorithm: VaMOPD Training Loop

Input: Student π_θ, teachers π_P, π_R, dataset D
Hyperparams: β (KL regularization), α_0 (prior strength)

For each training step:
  1. Sample batch of prompts (I, Q) ~ D
  
  2. Rollout: student generates ŷ = (ŷ_1, ..., ŷ_T) ~ π_θ
  
  3. E-Step (λ inference, no gradient):
     For each token position t:
       a. Get teacher distributions π_P(·|t), π_R(·|t)
       b. Get student distribution π_θ(·|t)
       c. Compute Fisher prior p(λ|t) using Eq. (Fisher prior)
       d. Solve λ*(t) via bisection on the fixed-point equation
       e. Compute posterior q(λ|t) = Beta with mean λ*(t),
          regularized toward prior by β
  
  4. M-Step (student update):
     For each token position t:
       L_t = D_KL( π_θ(·|t) || λ*(t)·π_P(·|t) + (1-λ*(t))·π_R(·|t) )
             + β · D_KL( q(λ|t) || p(λ|t) )
       
     θ ← θ - η · ∇_θ Σ_t L_t
```

### 4.2 计算开销分析

| 组件 | 复杂度 | 备注 |
|------|--------|------|
| 两 teacher forward | $2 \times O(T \cdot |\mathcal{V}|)$ | 可并行，与 vanilla MOPD 相同 |
| $\lambda$ inference (E-step) | $O(T \cdot K)$ | $K \approx 15$ 迭代，每迭代仅需标量计算 |
| Student forward + backward | $O(T \cdot |\mathcal{V}|)$ | 与 vanilla MOPD 相同 |
| Fisher prior 计算 | $O(T \cdot |\mathcal{V}|)$ | 从 teacher forward 结果直接算，无额外 forward |

**总开销**：约为 vanilla MOPD 的 $1.05\times$——几乎可以忽略的额外成本换来 per-token adaptive mixing。

---

## 5. 关键理论性质

### 5.1 与 Decomposed OPD 的关系

**命题 1（Decomposed OPD 是 VaMOPD 的特例）**：当 $\lambda(t) \equiv \lambda_0$（常数）且 teacher 为 self-distillation 时，VaMOPD 退化为 single-teacher 的 weighted OPD。VGS 的固定偏 visual 策略对应 $\lambda_0 > 0.5$。

**证明**：直接代入即可。

**命题 2（VaMOPD 的严格泛化性）**：存在 mixed-reasoning 分布，使得最优 $\lambda^*(t)$ 随 token 变化而任意 fixed-$\lambda$ 策略严格次优。

**证明构造**：考虑两个 token 位置 $t_1$（纯视觉，$\pi_P \gg \pi_R$）和 $t_2$（纯推理，$\pi_R \gg \pi_P$）。最优策略为 $\lambda(t_1)=1, \lambda(t_2)=0$。任何 fixed-$\lambda$ 的策略至少在其中一个位置产生非零 regret。$\square$

### 5.2 与 ViGOS (Hard Separation) 的关系

ViGOS 的 hard separation 等价于 $\lambda(t)$ 被输出格式显式约束：

$$
\lambda_{\text{ViGOS}}(t) = \begin{cases}
1 & \text{if token } t \text{ in } \langle\text{description}\rangle \text{ segment} \\
0 & \text{if token } t \text{ in } \langle\text{think}\rangle \text{ segment}
\end{cases}
$$

即 $\lambda(t) \in \{0, 1\}$ 且由人工模板决定，而非由数据分布推断。

**VaMOPD 的优势**：
- $\lambda(t) \in [0, 1]$ 连续，允许真正的混合监督
- $\lambda(t)$ 由 student-teacher 分布几何自动推断，不依赖格式先验
- 泛化到没有固定 segment 格式的任意 mixed-reasoning 数据

### 5.3 Token-Level Gradient Conflict Resolution

**命题 3（梯度冲突的变分解）**：当 $\pi_P$ 和 $\pi_R$ 在某 token 上梯度方向明显分歧时（cosine similarity $< 0$），VaMOPD 的 $\lambda^*(t)$ 倾向于推向 0 或 1（即选择一个主导 teacher），而非取中间值。

**直觉证明**：$\lambda \approx 0.5$ 的混合分布在梯度冲突时会产生"模糊的" target distribution（两个 teacher 的高概率区域取并集），其 KL 散度通常不如选取一侧的分布的 KL 散度小。因此优化器自然选择极端 $\lambda$ 值。

这意味着 **VaMOPD 自动解决了 token-level 的梯度冲突**——不需要显式梯度投影（PCGrad）或冲突规避（CAGrad），而是通过变分推断自然路由到正确的 teacher。

---

## 6. 理论贡献总结

| 内容 | 定位 |
|------|------|
| $\lambda(t)$ 隐变量形式化 | **核心理论贡献**：首次将 multi-teacher token-level mixing 形式化为变分推断问题 |
| Fisher Information 先验 | 连接 teacher 确信度与混合权重的统计桥梁 |
| EM 推断算法 | 实用贡献：封闭形式 E-step，$<5\%$ 额外开销 |
| 统一 Decomposed OPD + ViGOS | 理论完备性：已有方法均为 VaMOPD 的特例/限制形式 |
| 自动梯度冲突消解 | 属性：无需显式投影即可处理 teacher 冲突 |

---

## 7. 与论文叙事的对应

```
Abstract hook:
  "We show that multi-teacher on-policy distillation for mixed visual
   reasoning is fundamentally a variational inference problem over a
   latent capability-mixing variable."

Motivation figure:
  Left: same token, π_P pushes one direction, π_R pushes another
  Right: VaMOPD infers λ(t) and flexibly combines both

Method section title:
  "Variational Multi-Teacher Distillation:
   Token-Level Capability Mixing as Latent Variable Inference"

Key equation (boxed):
  λ*(t) = argmin_λ KL(π_θ || λ·π_P + (1-λ)·π_R)

Comparison table:
  | Method     | λ(t)               | λ source      | Format-Free | Error Attr. |
  | Vanilla OPD| ≡ 1.0              | —             | ✓           | ✗           |
  | MOPD       | ≡ α (constant)     | hyperparam    | ✓           | ✗           |
  | ViGOS      | ∈ {0, 1}           | format parse  | ✗           | Partial     |
  | Decomp. OPD| ≡ λ₀ (biased)      | hyperparam    | ✓           | ✗           |
  | VaMOPD     | ∈ [0, 1], adaptive | VI from π_θ   | ✓           | ✓ (GQA SG)  |
```

---

## 8. 实验设计：以 GQA 为主 Benchmark

### 8.1 为什么选 GQA

传统 KB-VQA benchmark（OK-VQA, A-OKVQA, InfoSeek）的语言先验过强，准确率天花板已被纯文本模型逼近，teacher conflict 信号被淹没。GQA 天然适合验证 VaMOPD：

- **组合式问题**：必须同时识别物体（perception）和理解关系（reasoning）
- **场景图标注**：提供 object / attribute / relation 级别的 ground truth，可直接做 error attribution
- **语言先验弱**：GQA 问题由模板生成，分布可控，不看图无法答对
- **CVPR 经典 benchmark**：审稿人认可度高

### 8.2 GQA 问题分解

```
问题类型               Perception 需求       Reasoning 需求        λ*(t) 预期
───────────────────────────────────────────────────────────────────────────
Verify/Object          高（识别物体）        低（存在性判断）       → 1
Verify/Attribute       高（识别属性）        低                      → 1
Query/Color            高（颜色识别）        低                      → 1
Query/Size             高（尺寸识别）        低                      → 1
Verify/Relation        中（识别主体）        高（空间/比较关系）     → 0
Choose/Logical         中（识别候选）        高（AND/OR/NOT）        → 0
Compare/Attribute      高（识别两物体属性）  高（比较推理）          → 0.5
Query/Count            高（识别+计数）       中                      → 0.7
```

### 8.3 Error Attribution via Scene Graph

GQA 的 scene graph 标注允许精确的 error decomposition：

```
Perception Error:
  - Wrong object: 模型识别了错误物体
  - Wrong attribute: 颜色/材质/尺寸判断错
  → λ(t) 应该高 → perception teacher 应该主导

Reasoning Error:
  - Wrong relation: 空间/逻辑关系判断错
  - Wrong logical composition: AND/OR 组合错
  → λ(t) 应该低 → reasoning teacher 应该主导

Evidence-Wrong (Joint Error):
  - Answer coincidentally correct but perception chain wrong
  - 例: "Is the red chair wooden?" → 答 Yes, 但模型把 blue chair 看成 red
  → λ(t) 错误地偏向 reasoning → 需要校正
```

### 8.4 Baseline 矩阵

```
Method                          GQA     GQA-Compose    Evidence-Wrong ↓
────────────────────────────────────────────────────────────────────────
Direct (SFT only)                xx.x         xx.x           xx.x
Mixed CoT                        xx.x         xx.x           xx.x
Vanilla OPSD                     xx.x         xx.x           xx.x
MOPD (fixed α)                   xx.x         xx.x           xx.x
MOPD + PCGrad                    xx.x         xx.x           xx.x
MOPD + CAGrad                    xx.x         xx.x           xx.x
ViGOS-style (hard separation)    xx.x         xx.x           xx.x
VaMOPD (ours)                    xx.x         xx.x           xx.x
  - w/o Fisher prior             xx.x         xx.x           xx.x
  - λ binarized (hard)           xx.x         xx.x           xx.x
```

### 8.5 关键实验信号

1. **GQA-Compose（推理密集型子集）**：VaMOPD 收益 > GQA 全量
2. **Evidence-Wrong Rate**：基于 scene graph 自动检测，VaMOPD 应显著低于 ViGOS
3. **λ(t) 与问题类型的相关性**：Query/Color → λ 高，Verify/Relation → λ 低
4. **High-hop 问题**：多跳组合推理中 λ(t) 变化应更剧烈，收益更大

---

## 9. 开放问题

1. **多 teacher 推广**：perception / knowledge / reasoning 三者时，λ 从标量变为 2-simplex，从 Beta 变为 Dirichlet。推导直接成立。

2. **λ(t) 的时序结构**：当前假设 λ(t) 在不同 token 间独立。真实 mixed reasoning 中相邻 token 的 λ 应高度相关。可加入 Markov prior λ(t) ∣ λ(t−1) 来建模。

3. **On-policy 偏差**：student 自身 rollout 上的 λ 推断可能不稳定。可加入 burn-in period 用 teacher rollout 初始化 λ。

4. **Connection to RL**：λ(t) 可被视作 token-level 的 capability advantage。这连接到了 RLCSD 的 contrastive signal 思路。

---

*Last updated: 2026-07-11*
