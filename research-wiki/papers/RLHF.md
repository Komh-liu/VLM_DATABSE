---
type: paper
node_id: paper:rlhf_note
title: "RLHF 完整讲解笔记：SFT → Reward Model → PPO → DPO"
authors: []
year: 2026
venue: "Note"
external_ids:
  arxiv: null
  doi: null
  s2: null
tags: ["RLHF", "SFT", "PPO", "DPO", "reward-model", "tutorial"]
added: 2026-06-26T00:00:00Z
---

# RLHF 完整讲解笔记：SFT → Reward Model → PPO → DPO

---

## 缘起：为什么需要 RLHF？

纯预训练的大语言模型（LLM）能生成流畅文本，但不会遵循指令、不会对齐人类偏好。RLHF（Reinforcement Learning from Human Feedback）解决的问题就是：**如何让模型学会人类认为"好"的回答？**

RLHF 给出的答案是一个三阶段流水线：

```
Stage 1: SFT            → 模型学会"模仿"（给定指令，生成正确答案）
Stage 2: Reward Model   → 训练一个打分器（学会"判断"回答好坏）
Stage 3: PPO            → 用打分器的信号优化策略（学会"产生"高分回答）
```

DPO 则是 Stage 2 + Stage 3 的替代方案：**不需要显式训练 reward model，直接从偏好数据中优化策略**。

---

## 第一阶段：SFT（有监督微调）

### 问题

预训练模型的目标是"预测下一个 token"，不是"遵循指令"。所以第一步是收集一批人工写的 `(prompt, ideal_response)` 对，让模型先学会模仿。

### 数学目标

给定 prompt $x$，我们希望模型生成理想回答 $y^c$（chosen）的概率最大化：

$$
\max_\theta \prod_{i=1}^{N} \pi_\theta(y_i^c \mid x_i)
$$

取负对数（把最大化变最小化）得到 SFT 损失函数：

$$
\boxed{L_{\text{SFT}}(\theta) = -\mathbb{E}_{(x, y^c) \sim \mathcal{D}} \big[ \log \pi_\theta(y^c \mid x) \big]}
$$

### 💡 直观理解

这本质上就是**交叉熵损失**。模型只需死记硬背"正确答案"，不需要理解"为什么好"或"什么不好"。这是它的根本局限。

### SFT 的局限

SFT 还有一个隐藏问题：**模型在 SFT 中看到的数据都是"唯一正确答案"，但真实世界中回答没有唯一标准。** 模型学会的是机械模仿，而不是对"好坏"的判断能力。

---

## 第二阶段：Reward Model（奖励模型）

### 问题

SFT 模型知道"什么是正确答案"，但不知道"为什么这个答案比那个好"。当面对开放性问题时，模型无法判断回答质量。

### 数据形式

收集人类标注的**偏好对**：给定 prompt $x$，两个回答 $y_c$（chosen）和 $y_r$（rejected），标注者选择哪个更好。

### 数学目标

用 **Bradley-Terry 模型**描述人类偏好：

$$
P(y_c \succ y_r \mid x) = \sigma\big( r_\phi(x, y_c) - r_\phi(x, y_r) \big)
$$

其中 $r_\phi(x, y)$ 是奖励模型的打分，$\sigma(z) = 1/(1+e^{-z})$ 是 Sigmoid 函数。

人类标注告诉了我们"谁赢了"，但我们希望 $r_\phi$ 学会**给赢家高分、给输家低分**。所以训练奖励模型就是最大化"人类偏好"的对数似然：

$$
\max_\phi \mathbb{E}_{(x, y_c, y_r) \sim \mathcal{D}} \big[ \log \sigma\big( r_\phi(x, y_c) - r_\phi(x, y_r) \big) \big]
$$

转化为损失函数（最小化负对数似然）：

$$
\boxed{L_{\text{RM}}(\phi) = -\mathbb{E}_{(x, y_c, y_r)} \big[ \log \sigma\big( r_\phi(x, y_c) - r_\phi(x, y_r) \big) \big]}
$$

### 产出

训练好的 $r_\phi(x, y)$ 可以对**任何**回答打一个标量分数，作为"人类偏好"的代理（proxy）。

### ⚠️ Reward Model 的隐患

RM 是一个 proxy，不是真正的"好"的标准。模型如果无约束地最大化 RM 分数，迟早会 exploit RM 的漏洞（reward hacking）——比如，发现说"根据权威研究"能得高分，就学会在所有回答中加这句话。

---

## 第三阶段：PPO（近端策略优化）

PPO 是 RLHF 的最后一步：用训练好的 reward model $r_\phi$ 来指导策略模型 $\pi_\theta$ 更新，同时防止更新过大导致灾难性遗忘。

### 3.1 RLHF 原始优化目标

RLHF 是一个**带 KL 约束的奖励最大化问题**：

$$
\max_{\pi_\theta} \mathbb{E}_{x \sim \mathcal{D}, y \sim \pi_\theta(y|x)} \big[ r_\phi(x, y) \big] - \beta \, \mathbb{D}_{\text{KL}} \big( \pi_\theta(y|x) \parallel \pi_{\text{ref}}(y|x) \big)
$$

| 符号 | 含义 | 说明 |
|------|------|------|
| $\pi_\theta$ | 当前策略模型 | 参数 $\theta$，初始化为 SFT 产出的模型 |
| $\pi_{\text{ref}}$ | 参考模型（固定） | 通常是 $\pi_{\text{SFT}}$，**不更新** |
| $r_\phi(x, y)$ | 奖励模型打分 | 一个标量，越大越好 |
| $\beta$ | KL 惩罚系数 | 控制"追求高分" vs "不要偏离 SFT"的 trade-off |
| $\mathbb{D}_{\text{KL}}$ | KL 散度 | 衡量两个概率分布的差异 |

### 3.2 修正后的 Reward

实际训练中，KL 项被移到 reward 内部，变成一个**修正后的 reward**：

$$
\boxed{R(x, y) = r_\phi(x, y) - \beta \log \frac{\pi_\theta(y|x)}{\pi_{\text{ref}}(y|x)}}
$$

这样做的原因：如果当前模型对某个 token 的概率比参考模型**高太多**，惩罚项立刻暴涨，拉低修正奖励。这强迫模型只能在参考模型附近"温和地"优化。

### 3.3 重要性采样

PPO 是 on-policy 算法，但为了数据效率，允许用旧策略 $\pi_{\theta_{\text{old}}}$ 采样的数据来更新新策略。通过**重要性采样（Importance Sampling）**实现：

$$
\text{ratio} = \frac{\pi_\theta(y|x)}{\pi_{\theta_{\text{old}}}(y|x)}
$$

如果 ratio = 2，说明新策略现在生成这个回答的概率是旧策略的 2 倍。

$$
L^{\text{PPO}}(\theta) = -\mathbb{E}_{x \sim \mathcal{D}, y \sim \pi_{\theta_{\text{old}}}} \left[ \frac{\pi_\theta(y|x)}{\pi_{\theta_{\text{old}}}(y|x)} \cdot A(x, y) \right]
$$

### 3.4 Advantage（优势函数）

Advantage 衡量"这个回答比平均水平好多少"：

$$
A(x, y) = R(x, y) - \text{baseline}
$$

在标准 PPO 中，baseline 由一个 **critic 网络**（value function）预测。Critic 网络与策略网络同时训练，输入 prompt $x$（或 state），输出预期 reward 的估计值。

- $A > 0$：这个回答好于预期 → 提高它的概率
- $A < 0$：这个回答差于预期 → 降低它的概率

**💡 与 GRPO 的区别**：GRPO（Group Relative Policy Optimization）省去了 critic 网络，直接用**同组回答的 reward 均值**作为 baseline。对同一个 prompt 采样 $G$ 个回答，计算组内 reward 均值和标准差做归一化。这在计算上更轻量，但要求每组有足够多样本才能可靠估计。

### 3.5 Clipping（PPO 核心创新）

如果 ratio 变得太大（比如 > 2），梯度会爆炸，导致**策略崩塌**——模型突然忘记一切。PPO 引入 **Clipping** 来防止这种情况：

$$
\boxed{L^{\text{PPO}}_{\text{clip}}(\theta) = -\mathbb{E}_{x, y} \Big[ \min\big( \text{ratio} \cdot A, \; \text{clip}(\text{ratio}, 1-\epsilon, 1+\epsilon) \cdot A \big) \Big]}
$$

$\epsilon$ 通常取 **0.2**，即 ratio 只能在 [0.8, 1.2] 范围内有效。

| 情况 | 条件 | 行为 |
|------|------|------|
| 好回答，合理提升 | $A > 0$，ratio 在 $[1-\epsilon, 1+\epsilon]$ | 正常提升 |
| **好回答，过度提升** | $A > 0$，ratio > $1+\epsilon$ | **裁掉**！不再奖励 |
| 差回答，合理降低 | $A < 0$，ratio 在 $[1-\epsilon, 1+\epsilon]$ | 正常降低 |
| **差回答，过度降低** | $A < 0$，ratio < $1-\epsilon$ | **裁掉**！不再惩罚 |

**💡 一体感**：$\epsilon$ 通常取 0.2。这意味着 ratio 只能在 [0.8, 1.2] 范围内有效。超出部分不计入梯度。

### 3.6 PPO 的多步迭代更新机制——为什么要用旧数据？怎么做到的？

#### 核心痛点：标准策略梯度的致命缺陷

标准策略梯度（REINFORCE）的更新公式是：

$$
\text{梯度} = \mathbb{E}_{y \sim \pi_\theta} \left[ A \cdot \nabla \log \pi_\theta(y|x) \right]
$$

**问题就在左边这个 $\mathbb{E}_{y \sim \pi_\theta}$** ——它要求数据必须从**当前最新的策略 $\pi_\theta$** 采样。这意味着：**每次更新模型参数后，都必须用新模型重新生成几万条数据**。在 LLM 场景中，一次采样的成本可能高达数万 GPU 小时，这是天文数字。

PPO 的核心贡献：**能否用旧数据 $\pi_{\theta_{\text{old}}}$ 的缓存来更新新模型 $\pi_\theta$？**

#### 数学桥梁：重要性采样

统计学中的**重要性采样（Importance Sampling）** 提供了答案。对于任意函数 $f(y)$，有以下恒等式：

$$
\mathbb{E}_{y \sim \pi_\theta} \big[ f(y) \big] = \mathbb{E}_{y \sim \pi_{\theta_{\text{old}}}} \left[ \frac{\pi_\theta(y|x)}{\pi_{\theta_{\text{old}}}(y|x)} \cdot f(y) \right]
$$

解读这个等式的两端：

| 侧 | 含义 | 能否实现？ |
|----|------|-----------|
| **左边** $\mathbb{E}_{y \sim \pi_\theta}$ | 需要从新策略采样 | ❌ 成本太高 |
| **右边** $\mathbb{E}_{y \sim \pi_{\theta_{\text{old}}}}$ | 只需要从旧策略采样 | ✅ 可用缓存数据 |

**ratio** $= \pi_\theta / \pi_{\theta_{\text{old}}}$ 是修正系数：如果新策略 $\pi_\theta$ 认为某个回答的概率比旧策略 $\pi_{\theta_{\text{old}}}$ 更高（ratio $> 1$），就放大这个样本的影响；反之则缩小。这保证了**即使数据来自旧策略，计算结果在期望上等价于来自新策略**。

#### 构造替代目标函数

将 $f(y) = A \cdot \nabla \log \pi_\theta$ 代入重要性采样恒等式，得到 PPO 的**替代目标（Surrogate Objective）**：

$$
L(\theta) = \mathbb{E}_{y \sim \pi_{\theta_{\text{old}}}} \left[ \frac{\pi_\theta(y|x)}{\pi_{\theta_{\text{old}}}(y|x)} \cdot A \right]
$$

从现在起，所有期望都基于**固定的旧策略数据缓存**。每次更新只需从缓存中随机抽取一个小批量，完全不需要重新生成。

#### 对替代目标求梯度

对 $L(\theta)$ 求关于 $\theta$ 的梯度。由于期望和 $A$ 都来自旧数据、与当前 $\theta$ 无关，可直接移入期望内部：

$$
\nabla L(\theta) = \mathbb{E}_{y \sim \pi_{\theta_{\text{old}}}} \left[ A \cdot \nabla \left( \frac{\pi_\theta}{\pi_{\theta_{\text{old}}}} \right) \right]
$$

根据**对数导数技巧**：$\nabla \left( \frac{\pi_\theta}{\pi_{\theta_{\text{old}}}} \right) = \frac{\pi_\theta}{\pi_{\theta_{\text{old}}}} \cdot \nabla \log \pi_\theta$

得到最终梯度：

$$
\nabla L(\theta) = \mathbb{E}_{y \sim \pi_{\theta_{\text{old}}}} \left[ \frac{\pi_\theta}{\pi_{\theta_{\text{old}}}} \cdot A \cdot \nabla \log \pi_\theta \right]
$$

PyTorch 实际实现中，加负号转成 loss：

$$
Loss = - \mathbb{E}\left[ \text{ratio} \cdot A \right], \quad \text{其中 } \text{ratio} = \frac{\pi_\theta(y|x)}{\pi_{\theta_{\text{old}}}(y|x)}
$$

#### 完整流程：采样一次，更新 K 次

$$
\begin{aligned}
&\textbf{初始化：}\ \pi_\theta \gets \pi_{\text{SFT}} \\
\\
&\textbf{循环直到收敛：} \\
&\qquad \textbf{① 冻结：}\ \pi_{\theta_{\text{old}}} \gets \pi_\theta \quad (\text{复制当前策略，冻结权重，只用于采样}) \\
&\qquad \textbf{② 采样：} \text{用 } \pi_{\theta_{\text{old}}} \text{ 采样一批 } (x, y, A) \text{ 存入缓存} \\
&\qquad \textbf{③ 多步更新：} \text{对缓存做 } K \text{ 次梯度下降} \\
&\qquad\qquad \text{for } t = 1 \text{ to } K: \\
&\qquad\qquad\qquad \text{从缓存中取一个 minibatch} \\
&\qquad\qquad\qquad Loss = -\mathbb{E}\left[ \text{ratio} \cdot A \right] \\
&\qquad\qquad\qquad \theta \gets \theta - \alpha \cdot \nabla_\theta Loss \\
&\qquad\qquad\qquad \text{ratio 在更新中逐步偏离 1，clip 在 } [0.8, 1.2] \text{ 内} \\
&\qquad \textbf{④ 重置：} \text{回到 ①，将新 } \pi_\theta \text{ 作为下一轮的 } \pi_{\theta_{\text{old}}}
\end{aligned}
$$

#### 为什么不是无限次？

随着 $\pi_\theta$ 持续更新，ratio 逐步偏离 1。当越来越多的 ratio 触碰到 $[0.8, 1.2]$ 的边界时，说明 $\pi_\theta$ 已经偏离 $\pi_{\theta_{\text{old}}}$ 太远。Clipping 截断梯度，防止进一步偏移。此时缓存数据的价值被"榨干"，需要重新采样。

#### 效率优势量化

| 算法 | 采样一次能做几步梯度 | 原因 |
|------|---------------------|------|
| **REINFORCE** | 1 步 | 无重要性采样，更新后旧数据立刻失效 |
| **PPO** | $K = 4 \sim 10$ 步 | 重要性采样修正 + Clipping 安全网 |
| **GRPO** | $K = 4 \sim 10$ 步 | 继承 PPO 的多步更新模式 |

**加速**：PPO 每步有效更新的成本为 $(T_{\text{sample}} + K \cdot T_{\text{update}}) / K$，而 REINFORCE 是 $T_{\text{sample}} + T_{\text{update}}$。当采样成本 $T_{\text{sample}} \gg T_{\text{update}}$ 时，PPO 效率接近 $K$ 倍。这个数学小"骗术"——名义上优化新策略，实际上全程用旧数据——省下了 LLM 训练中 99% 的采样成本。

### 3.7 PPO 完整目标

InstructGPT 实际使用的 PPO 损失：

$$
\boxed{L_{\text{PPO}}(\theta) = L^{\text{clip}}_{\text{PPO}}(\theta) + \gamma \cdot L^{\text{SFT}}(\theta)}
$$

（KL 惩罚已内化到修正 reward $R$ 中。$L^{\text{SFT}}$ 项保留一部分 SFT 数据，防止模型在 RL 中遗忘预训练知识。）

---

## 第四阶段：DPO（直接偏好优化）

### 问题

PPO 方案有三点麻烦：
1. 需要训练和维护一个额外的 reward model
2. 需要 tuning PPO 的超参数（clip、KL、lr）
3. 需要 rollout 采样，计算开销大

**DPO 的核心想法**：能不能直接从偏好数据中优化策略，跳过 reward model 和 PPO？

答案是可以——只需要解一道数学题。

### 4.1 从 RLHF 目标出发

回忆 RLHF 的原始目标：

$$
\max_{\pi} \mathbb{E}_{x \sim \mathcal{D}, y \sim \pi(y|x)} \big[ r(x, y) \big] - \beta \, \mathbb{D}_{\text{KL}} \big( \pi(y|x) \parallel \pi_{\text{ref}}(y|x) \big)
$$

对于**每一个**固定的 $x$，展开 KL 散度：

$$
\mathbb{D}_{\text{KL}}(\pi \parallel \pi_{\text{ref}}) = \mathbb{E}_{y \sim \pi} \bigg[ \log \frac{\pi(y|x)}{\pi_{\text{ref}}(y|x)} \bigg] = \sum_y \pi(y|x) \big( \log \pi(y|x) - \log \pi_{\text{ref}}(y|x) \big)
$$

所以点 $x$ 上的目标函数为：

$$
\mathcal{L}_x(\pi) = \sum_y \pi(y|x) \, r(x, y) - \beta \sum_y \pi(y|x) \big( \log \pi(y|x) - \log \pi_{\text{ref}}(y|x) \big)
$$

合并成一个期望：

$$
\mathcal{L}_x(\pi) = \sum_y \pi(y|x) \Big[ r(x, y) - \beta \log \frac{\pi(y|x)}{\pi_{\text{ref}}(y|x)} \Big]
= \mathbb{E}_{y \sim \pi} \Big[ r(x, y) - \beta \log \frac{\pi(y|x)}{\pi_{\text{ref}}(y|x)} \Big]
$$

### 4.2 ⭐ 求解最优策略的闭式解（拉格朗日乘数法）

这是整个 DPO 推导中最核心的步骤。

我们要找在**给定 $x$ 下**、让 $\mathcal{L}_x(\pi)$ 最大的概率分布 $\pi(y|x)$。这是一个带约束的优化问题：

$$
\max_{\pi} \; \mathcal{L}_x(\pi) \quad \text{s.t.} \quad \sum_y \pi(y|x) = 1
$$

用**拉格朗日乘数法**求解。构造拉格朗日函数：

$$
\mathcal{L} = \sum_y \pi(y|x) \Big[ r(x, y) - \beta \log \frac{\pi(y|x)}{\pi_{\text{ref}}(y|x)} \Big] + \lambda \Big( 1 - \sum_y \pi(y|x) \Big)
$$

对 $\pi(y|x)$ 求偏导。回忆 $\frac{\partial}{\partial \pi} (-\pi \log \pi) = -(\log \pi + 1)$，逐项展开：

$$
\frac{\partial}{\partial \pi(y|x)} \Big[ \pi(y|x) \, r(x, y) \Big] = r(x, y)
$$

$$
\frac{\partial}{\partial \pi(y|x)} \Big[ -\beta \pi(y|x) \log \pi(y|x) \Big] = -\beta \big( \log \pi(y|x) + 1 \big)
$$

$$
\frac{\partial}{\partial \pi(y|x)} \Big[ \beta \pi(y|x) \log \pi_{\text{ref}}(y|x) \Big] = \beta \log \pi_{\text{ref}}(y|x)
$$

$$
\frac{\partial}{\partial \pi(y|x)} \Big[ -\lambda \pi(y|x) \Big] = -\lambda
$$

合起来，令偏导为 0：

$$
r(x, y) - \beta \big( \log \pi(y|x) + 1 \big) + \beta \log \pi_{\text{ref}}(y|x) - \lambda = 0
$$

整理：

$$
\beta \log \pi(y|x) = r(x, y) + \beta \log \pi_{\text{ref}}(y|x) - \beta - \lambda
$$

两边除以 $\beta$：

$$
\log \pi(y|x) = \frac{1}{\beta} r(x, y) + \log \pi_{\text{ref}}(y|x) - 1 - \frac{\lambda}{\beta}
$$

取指数：

$$
\pi(y|x) = \pi_{\text{ref}}(y|x) \cdot \exp\!\left( \frac{1}{\beta} r(x, y) \right) \cdot \exp\!\left( -1 - \frac{\lambda}{\beta} \right)
$$

注意 $\exp(-1 - \lambda/\beta)$ **与 $y$ 无关**，它只是一个归一化常数。令：

$$
Z(x) = \sum_{y'} \pi_{\text{ref}}(y'|x) \exp\!\left( \frac{1}{\beta} r(x, y') \right)
$$

则 $\exp(-1 - \lambda/\beta) = 1 / Z(x)$。于是最优策略的闭式解为：

$$
\boxed{\pi^*(y|x) = \frac{1}{Z(x)} \, \pi_{\text{ref}}(y|x) \, \exp\!\left( \frac{1}{\beta} r(x, y) \right)}
$$

**💡 直观理解**：**最优策略就是在参考策略的基础上，对每个 $y$ 按其 reward 指数加权后归一化。** reward 越高的回答，最优策略会分配越高的概率；$\beta$ 越小（KL 惩罚越弱），放大效应越剧烈。

### 4.3 ⭐ 反解 Reward 函数

上面的公式里，$r(x, y)$ 是未知的。但我们可以把公式**反过来**，用已知的 $\pi^*$ 和 $\pi_{\text{ref}}$ 表达 $r$。

对 $\pi^*(y|x)$ 两边取对数：

$$
\log \pi^*(y|x) = \log \pi_{\text{ref}}(y|x) + \frac{1}{\beta} r(x, y) - \log Z(x)
$$

解出 $r(x, y)$：

$$
\boxed{r(x, y) = \beta \log \frac{\pi^*(y|x)}{\pi_{\text{ref}}(y|x)} + \beta \log Z(x)}
$$

**核心信息**：reward 函数可以被**等价地表达**为最优策略和参考策略的概率比（乘以 $\beta$）加一个只与 $x$ 有关的归一化项。

### 4.4 代入 Bradley-Terry 模型——关键一步

Bradley-Terry 模型描述人类偏好的概率：

$$
P(y_c \succ y_r \mid x) = \sigma\big( r(x, y_c) - r(x, y_r) \big)
$$

代入 $r(x, y)$ 表达式，先看差值：

$$
r(x, y_c) - r(x, y_r) = \left[ \beta \log \frac{\pi^*(y_c|x)}{\pi_{\text{ref}}(y_c|x)} + \cancel{\beta \log Z(x)} \right] - \left[ \beta \log \frac{\pi^*(y_r|x)}{\pi_{\text{ref}}(y_r|x)} + \cancel{\beta \log Z(x)} \right]
$$

**$Z(x)$ 在减法中完美消掉了！** 这正是 DPO 推导的魔法时刻——我们无需知道这个归一化常数的确切值。

结果：

$$
r(x, y_c) - r(x, y_r) = \beta \log \frac{\pi^*(y_c|x)}{\pi_{\text{ref}}(y_c|x)} - \beta \log \frac{\pi^*(y_r|x)}{\pi_{\text{ref}}(y_r|x)}
$$

### 4.5 从最优策略到当前策略

我们不知道最优策略 $\pi^*$ 是什么。DPO 做了一个关键假设：**把正在训练的模型 $\pi_\theta$ 视为 $\pi^*$ 的近似**，用 $\pi_\theta$ 代入上式。

于是偏好概率：

$$
P(y_c \succ y_r \mid x) = \sigma\!\left( \beta \log \frac{\pi_\theta(y_c|x)}{\pi_{\text{ref}}(y_c|x)} - \beta \log \frac{\pi_\theta(y_r|x)}{\pi_{\text{ref}}(y_r|x)} \right)
$$

### 4.6 得到 DPO 损失函数

对偏好概率做最大似然估计，取负对数：

$$
\boxed{L_{\text{DPO}}(\theta) = -\mathbb{E}_{(x, y_c, y_r) \sim \mathcal{D}} \left[ \log \sigma\!\left( \beta \log \frac{\pi_\theta(y_c|x)}{\pi_{\text{ref}}(y_c|x)} - \beta \log \frac{\pi_\theta(y_r|x)}{\pi_{\text{ref}}(y_r|x)} \right) \right]}
$$

### 解读 DPO 损失

令 $u = \beta \log \frac{\pi_\theta(y_c|x)}{\pi_{\text{ref}}(y_c|x)} - \beta \log \frac{\pi_\theta(y_r|x)}{\pi_{\text{ref}}(y_r|x)}$，则 $L_{\text{DPO}} = -\mathbb{E}[\log \sigma(u)]$。

- $u > 0$：模型对优选 $\pi_\theta(y_c)$ 的相对提升 > 对拒选 $\pi_\theta(y_r)$ 的相对提升 → $\sigma(u) > 0.5$ → loss 小
- $u < 0$：模型更喜欢拒选 → $\sigma(u) < 0.5$ → loss 大

**与 SFT 的本质区别**：SFT 只看 $y_c$，无条件推高概率；DPO 同时看 $y_c$ 和 $y_r$，强制拉大差距——这是一个**对比（Contrastive）**过程。

### 4.7 DPO 的梯度行为

对 $L_{\text{DPO}}$ 求梯度：

$$
\nabla_\theta L_{\text{DPO}} = -\beta \, \mathbb{E} \Big[ \underbrace{\sigma(-u)}_{\text{自适应权重}} \cdot \underbrace{\big( \nabla_\theta \log \pi_\theta(y_c|x) - \nabla_\theta \log \pi_\theta(y_r|x) \big)}_{\text{拉大差距}} \Big]
$$

- $\sigma(-u)$：当模型**已经正确**（$u$ 大，$\sigma(-u)$ 小）时，梯度被压低；当模型**犯错**（$u$ 小或负，$\sigma(-u)$ 大）时，梯度放大。这是一种**自适应的动态权重**。
- $(\nabla_\theta \log \pi_\theta(y_c) - \nabla_\theta \log \pi_\theta(y_r))$：同时提高 $y_c$ 的概率、降低 $y_r$ 的概率。

---

## 总结对比

| 维度 | SFT | PPO（InstructGPT） | DPO |
|------|-----|-------------------|-----|
| **输入数据** | 单条 prompt + 理想回答 | prompt（+ RM 打分） | prompt + 偏好对 $(y_c, y_r)$ |
| **需要 RM** | ❌ | ✅ | ❌ |
| **需要 $\pi_{\text{ref}}$** | ❌ | ✅ | ✅ |
| **优化目标** | $\max \log \pi_\theta(y_c)$ | $\max R - \beta \cdot \text{KL}$ | $\max \log \sigma(\beta \log \frac{\pi_\theta(y_c)}{\pi_{\text{ref}}(y_c)} - \beta \log \frac{\pi_\theta(y_r)}{\pi_{\text{ref}}(y_r)})$ |
| **核心机制** | 模仿正确答案 | 标量 reward + clip 约束 | 偏好对之间的概率比差距 |
| **计算开销** | 低 | 高（RM + rollout + clip） | 低（一个 forward pass） |
| **调参难度** | 很低 | 很高（$\beta$、clip、lr 等） | 中等（$\beta$） |

### 完整关系图

```
传统 RLHF 方案:
  SFT → Reward Model → PPO
  (学模仿)   (学打分)    (学优化)

DPO 方案:
  SFT → DPO
  (学模仿)   (直接学偏好，跳过 RM + PPO)

公式链路:
  SFT:        L = -log π_θ(y_c | x)
  RM:         L = -log σ(r_φ(x, y_c) - r_φ(x, y_r))
  PPO:        L = -E[min(ratio·A, clip(ratio)·A)]
  DPO:        L = -E[log σ(β·log(π_θ(y_c)/π_ref(y_c)) - β·log(π_θ(y_r)/π_ref(y_r)))]
              ↑ 从 RLHF 目标解最优策略 → 反解 r → 代入 BT → 消去 Z(x) → 得到
```

### DPO 推导全链路

```
Step 1: 写出 RLHF 目标
  max E[r(x,y)] - β·KL(π || π_ref)

Step 2: 拉格朗日求解最优策略（详细推导见 4.2 节）
  π*(y|x) = (1/Z(x)) · π_ref(y|x) · exp(r(x,y)/β)

Step 3: 反解出 r(x,y)
  r(x,y) = β · log(π*(y|x)/π_ref(y|x)) + β·log Z(x)

Step 4: 代入 Bradley-Terry
  P(y_c ≻ y_r | x) = σ(r(x,y_c) - r(x,y_r))
                    = σ(β·log(π*(y_c)/π_ref(y_c)) - β·log(π*(y_r)/π_ref(y_r)))
                    ↑ Z(x) 抵消！

Step 5: 用 π_θ 近似 π*
  L_DPO = -E[log σ(β·log(π_θ(y_c)/π_ref(y_c)) - β·log(π_θ(y_r)/π_ref(y_r)))]
```
