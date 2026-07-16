---
type: paper
short: "DecomposedOPD"
node_id: paper:decomposed_opd_icml2026
title: "Decomposed On-Policy Distillation for Vision-Language Reasoning: Steering Gradients for Visual Grounding"
authors: ["Hee Suk Yoon", "Eunseop Yoon", "Jaehyun Jang", "SooHwan Eom", "Ji Woo Hong", "Mark Hasegawa-Johnson", "Qi Dai", "Chong Luo", "Chang D. Yoo"]
year: 2026
venue: "ICML 2026 (Spotlight)"
external_ids:
  arxiv: "2606.00564"
  doi: ""
  s2: null
tags: ["on-policy-distillation", "visual-grounding", "VLM", "vision-language", "knowledge-distillation", "gradient-decomposition"]
added: 2026-07-14T00:00:00Z
pdf: "decomposed_opd_icml2026.pdf"
---

# Decomposed On-Policy Distillation for Vision-Language Reasoning: Steering Gradients for Visual Grounding

## One-line thesis

> 标准 VLM 的 OPD 目标函数可以数学分解为**语言先验**（Language Prior）和**视觉 Grounding**（Visual Grounding）两个分量；二者的梯度在高视觉依赖 token 上近乎正交，导致标准优化走一条被动的折中路线。通过在梯度层面显式掰向视觉子空间（VGS），打破这种几何对称性，能以极小开销显著提升多模态推理的对齐效果。

## Problem / Gap

现有 VLM 的 OPD（如 Qwen3-VL）把蒸馏限制在文本 token 上，忽略了视觉 grounding 的对齐。直接将标准 OPD 扩展到多模态域时，默认优化一个**单一的整体目标**。本文挑战这一做法：将其数学分解为语言和视觉两个分量，揭示二者之间存在根本性的优化张力——标准梯度是一个"和事佬"，在视觉最需要被优化的 token 上也只取静态折中。

### 问题设定：多模态 OPD 的 trajectory 和 token-level KL

令输入为图像 $I$ 和文本 prompt $x$。学生模型 $p_{\theta_S}$ 自回归采样一条推理轨迹 $\tau = (h_1, \ldots, h_T, \hat{y}) \sim p_{\theta_S}(\cdot \mid I, x)$，其中 $\{h_k\}_{k=1}^T$ 为中间推理步骤（Chain-of-Thought），$\hat{y}$ 为最终答案。教师模型 $q_T$ 是固定的。记 $\tau_{<t}$ 为第 $t$ 步之前的轨迹前缀。

教师 $q_T$ 本身是一个模型，但在**两种不同输入条件**下查询时会得到不同的分布：

- **多模态分布** $q_T(\cdot \mid \tau_{<t}, I, x)$：同时看到图像和文本时输出的 next-token 分布
- **单模态（纯文本）分布** $q_T(\cdot \mid \tau_{<t}, x)$：只看到文本 prompt（不给图像）时输出的分布

学生同样有这两种分布 $p_{\theta_S}(\cdot \mid \tau_{<t}, I, x)$ 和 $p_{\theta_S}(\cdot \mid \tau_{<t}, x)$，除此之外 3.1 节还会构造一个混合目标分布 $q_T^*$（学生的语言先验 + 教师的视觉信息增益）。所有分布都是词表 $\mathcal{V}$ 上的完整 categorical 分布。

## Method

### 2.1 背景：On-Policy Distillation 的两种 KL 形式

> **输入条件说明**：Forward KL 和 Reverse KL 在本文中均定义在**纯文本条件** $(x, \tau_{<t})$ 上（无图像 $I$）。原因：SeqKD 和早期 OPD 工作本来就是在 text-only 设定下提出的。到 2.2 节才将 Reverse KL 推广到多模态（加上 $I$）。

#### Forward KL（SeqKD / Off-Policy）

Trajectory-wise Forward KL 定义为轨迹上逐 token Forward KL 的均值（注意：纯文本条件，无 $I$）：

$$
\ell_{\text{Forward}}(\tau) \triangleq \frac{1}{|\tau|} \sum_{t=1}^{|\tau|} D_{\mathrm{KL}}\big(q_T(\cdot \mid \tau_{<t}, x) \;\big\|\; p_{\theta_S}(\cdot \mid \tau_{<t}, x)\big)
$$

Off-Policy Distillation 在教师自己生成的轨迹上最小化此期望：

$$
\mathcal{L}_{\text{Off-Policy}} = \mathbb{E}_{\tau \sim q_T}[\ell_{\text{Forward}}(\tau)] \tag{1}
$$

> **问题**：Forward KL 是 **mode-covering** 的——它强迫学生覆盖教师分布的所有 mode。在推理任务中，学生容量不足以覆盖教师的全复杂度时，会在不同 mode 之间用低概率转移桥接，导致幻觉和不合理的推理链。此外存在 **exposure bias**：学生从未在自己的自回归错误上训练。

#### Reverse KL（On-Policy）

Trajectory-wise Reverse KL（纯文本条件，无 $I$）：

$$
\ell_{\text{Reverse}}(\tau) \triangleq \frac{1}{|\tau|} \sum_{t=1}^{|\tau|} D_{\mathrm{KL}}\big(p_{\theta_S}(\cdot \mid \tau_{<t}, x) \;\big\|\; q_T(\cdot \mid \tau_{<t}, x)\big)
$$

On-Policy Distillation 在学生自己生成的轨迹上最小化期望：

$$
\mathcal{L}_{\text{On-Policy}} = \mathbb{E}_{\tau \sim p_{\theta_S}}[\ell_{\text{Reverse}}(\tau)] \tag{2}
$$

> **优势**：Reverse KL 是 **mode-seeking** 的——惩罚学生生成教师认为不可能的高概率 token（$q_T \approx 0$ 的区域），优先匹配教师的高置信度推理路径。同时 $\tau \sim p_{\theta_S}$ 使学生学自自己的 rollout，闭合 training-inference gap。

---

### 2.2 标准整体式多模态目标（Standard Monolithic Objective）

将 Reverse KL 从纯文本推广到多模态——在条件中加入图像 $I$，使用教师的**多模态分布** $q_T(\cdot \mid \tau_{<t}, I, x)$：

$$
\ell_{\text{Standard}}(\tau) \triangleq \frac{1}{|\tau|} \sum_{t=1}^{|\tau|} D_{\mathrm{KL}}\big(p_{\theta_S}(\cdot \mid \tau_{<t}, I, x) \;\big\|\; q_T(\cdot \mid \tau_{<t}, I, x)\big) \tag{3}
$$

$$
\mathcal{L}_{\text{Standard}} = \mathbb{E}_{\tau \sim p_{\theta_S}(\cdot \mid I, x)}[\ell_{\text{Standard}}(\tau)] \tag{4}
$$

> **核心问题**：这个目标把监督信号视为一个不可分割的整体，掩盖了**语言先验对齐**和**视觉 Grounding** 的独立贡献。接下来用 Bayes 分解把二者拆开。

---

### 3.1 分解整体式目标（Decomposing the Monolithic Objective）

#### Bayes 分解恒等式

对任意多模态生成器 $p$，应用 Bayes 规则将条件概率分解为**语言先验 + 视觉似然 + 常数项**：

$$
\log p(\tau \mid I, x) = \underbrace{\log p(\tau \mid x)}_{\text{Language Prior}} + \underbrace{\log p(I \mid \tau, x)}_{\text{Visual Likelihood}} - \underbrace{\log p(I \mid x)}_{\text{Evidence (常数)}} \tag{5}
$$

其中 $\log p(I \mid x)$ 仅依赖图像本身，对轨迹 $\tau$ 是常数。将这一恒等式分别应用于学生 $p_{\theta_S}$ 和教师 $q_T$，揭示出标准目标隐式包含两个独立目标。

> **直觉**：$\log p(\tau \mid x)$ 是纯文本分布——给定 prompt $x$ 生成 $\tau$ 的概率，不涉及图像。$\log p(I \mid \tau, x)$ 是给定生成的 $\tau$ 后对图像 $I$ 的解释能力——生成的推理在多大程度上能"回推"出图像内容。证据项 $\log p(I \mid x)$ 是序列无关的标准化常数。

---

#### I. 语言先验对齐（Language Prior Matching, $\mathcal{L}_{\text{Lang}}$）

匹配教师和学生在**纯文本条件**下的分布——注意这里使用的是教师的纯文本分布 $q_T(\cdot \mid \tau_{<t}, x)$（无 $I$），确保推理风格（reasoning style）的迁移不依赖视觉上下文：

$$
\ell_{\text{Lang}}(\tau) \triangleq \frac{1}{|\tau|} \sum_{t=1}^{|\tau|} D_{\mathrm{KL}}\big(p_{\theta_S}(\cdot \mid \tau_{<t}, x) \;\big\|\; q_T(\cdot \mid \tau_{<t}, x)\big) \tag{6}
$$

> **关键细节**：梯度在**单模态分布**（text-only）上计算，但轨迹 $\tau$ 仍从**完整多模态策略** $p_{\theta_S}(\cdot \mid I, x)$ 采样，以维持 on-policy 对齐：

$$
\mathcal{L}_{\text{Lang}} = \mathbb{E}_{\tau \sim p_{\theta_S}(\cdot \mid I, x)}[\ell_{\text{Lang}}(\tau)] \tag{7}
$$

---

#### II. 视觉 Grounding 对齐（Visual Grounding Matching, $\mathcal{L}_{\text{Vis}}$）

此目标隔离 Eq. 5 中的视觉似然项 $\log p(I \mid \tau, x)$。该似然衡量模型的**感知敏感度**——生成的推理在多大程度上能因果性地解释视觉输入。

> **关键技巧**：虽然 $\log p(I \mid \tau, x)$ 在自回归模型中无法直接计算，但 Bayes 规则揭示它等于可计算的 **Visual Information Gain**（后验与先验的对数比）加序列独立常数：

$$
\log p(I \mid \tau, x) = \underbrace{\big[\log p(\tau \mid I, x) - \log p(\tau \mid x)\big]}_{\text{Visual Information Gain}} + \log p(I \mid x) \tag{8}
$$

由于 $\log p(I \mid x)$ 与 $\tau$ 无关，**匹配教师的信息增益在数学上等价于对齐学生的视觉感知**。

然后构造一个**视觉目标分布** $q_T^*$：保留学生的语言先验，但将视觉似然替换为教师的：

$$
q_T^*(\tau \mid I, x) \propto p_{\theta_S}(\tau \mid x) \cdot q_T(I \mid \tau, x) \tag{9}
$$

> **⭐ $q_T^*$ vs $q_T$ 的关键区分**：到这里我们有了三个不同的 token 级分布，必须区分清楚：
>
> | 符号 | 语言先验来源 | 视觉信息增益来源 | 本质 |
> |------|:------------:|:----------------:|------|
> | $q_T(\cdot \mid \tau_{<t}, I, x)$ | 教师 | 教师 | 教师的**完整多模态分布** |
> | $q_T(\cdot \mid \tau_{<t}, x)$ | 教师 | 无（纯文本） | 教师的**纯语言分布** |
> | $q_T^*(\cdot \mid \tau_{<t}, I, x)$ | **学生** | **教师** | **构造的混合分布** |
>
> 为什么需要 $q_T^*$ 而不是直接用 $q_T(\cdot \mid \tau_{<t}, I, x)$？因为 $\mathcal{L}_{\text{Vis}}$ 的目的**仅仅是传递视觉感知能力**。如果直接用教师的完整多模态分布作 target，语言风格也会被教师的覆盖——这就和 $\mathcal{L}_{\text{Lang}}$ 打架了。$q_T^*$ 的设计精妙之处在于：用学生的语言先验做底座，只把教师的视觉信息增量"嫁接"上去，实现了视觉和语言的**干净分离**。

将教师的 Bayesian 展开代入 Eq. 9，得到 log-space 下可计算的 target logits：

$$
\log q_T^*(\tau \mid I, x) = \log p_{\theta_S}(\tau \mid x) + \big(\log q_T(\tau \mid I, x) - \log q_T(\tau \mid x)\big) - \log Z^* \tag{10}
$$

其中 $(\log q_T(\tau \mid I, x) - \log q_T(\tau \mid x))$ 正是教师的**视觉信息增益**（Visual Information Gain）——教师在看到图像后对 $\tau$ 的概率评估提升了多少。这一项是 $q_T^*$ 从教师那里唯一借用的东西。

> **关键细节**：教师的边际证据项 $\log q_T(I \mid x)$ 对所有 token 是标量偏移。在自回归公式中，该常数被吸收到每步 Softmax 的局部分配函数 $Z^*$ 中，因此**无需显式计算**。

最后定义 trajectory-wise 视觉散度及其期望：

$$
\ell_{\text{Vis}}(\tau) \triangleq \frac{1}{|\tau|} \sum_{t=1}^{|\tau|} D_{\mathrm{KL}}\big(p_{\theta_S}(\cdot \mid \tau_{<t}, I, x) \;\big\|\; q_T^*(\cdot \mid \tau_{<t}, I, x)\big) \tag{11}
$$

$$
\mathcal{L}_{\text{Vis}} = \mathbb{E}_{\tau \sim p_{\theta_S}(\cdot \mid I, x)}[\ell_{\text{Vis}}(\tau)]
$$

> **直觉**：KL 的 target 是 $q_T^*$ 而非 $q_T$——迫使学生在**保留自己语言偏好**的同时，吸收教师对视觉信息的敏感度。这实现了视觉和语言两个目标的干净分离。

---

### 📊 动态指标：Visual Dependency Score (VDS)

#### VDS 定义

衡量教师对每个 token 的视觉依赖程度。对教师的两个分布——多模态 $q_T(\cdot \mid \tau_{<t}, I, x)$ 和纯文本 $q_T(\cdot \mid \tau_{<t}, x)$——计算 KL 散度：

$$
\text{VDS}_t = D_{\mathrm{KL}}\big(q_T(\cdot \mid \tau_{<t}, I, x) \;\big\|\; q_T(\cdot \mid \tau_{<t}, x)\big) \tag{12}
$$

> **直觉**：在给定前缀 $\tau_{<t}$ 时，教师的 next-token 分布在有图像 $I$ 和没有图像时差距有多大？差距大 → token 高度依赖视觉 → 高 VDS。
>
> **注意**：VDS 完全基于教师 $q_T$ 的两种输入条件计算，**不涉及 $q_T^*$**。$q_T^*$ 是训练时 $\mathcal{L}_{\text{Vis}}$ 的 KL target，VDS 是分析工具。

所有生成的 token 按 VDS 分为 10 个等频分桶（equal-frequency quantiles）：
- **Bin 0**：最小视觉依赖（纯语言 token，如格式词、连接词）
- **Bin 9**：最大视觉依赖（需要看图的 token，如颜色、数量、空间关系）

---

### 3.2 梯度动力学的几何分析（Geometric Analysis of Gradient Dynamics）

这是全文最重要的分析部分。研究 $\nabla \mathcal{L}_{\text{Standard}}$ 相对于 $\nabla \mathcal{L}_{\text{Lang}}$ 和 $\nabla \mathcal{L}_{\text{Vis}}$ 的几何关系，按 VDS 分桶统计。

#### 分析设定
- 在验证集 prompt 上
- 轨迹从完整多模态学生策略 $\tau \sim p_{\theta_S}(\cdot \mid I, x)$ on-policy 采样
- 对每个 token 计算 VDS 并分桶

#### 发现 I：视觉极端处的正交性（Figure 3-a）

| VDS 区间 | $\nabla \mathcal{L}_{\text{Lang}}$ 与 $\nabla \mathcal{L}_{\text{Vis}}$ 夹角 | 含义 |
|----------|:--------------------------------------------------------------------------:|------|
| Bin 0（低视觉依赖） | $\theta \approx 60^\circ$ | 有一定共享子空间，两者存在协调整的空间 |
| Bin 9（高视觉依赖） | $\theta \approx 92^\circ$（近乎正交） | 二者近乎独立——在一个方向上前进几乎不影响另一个 |
| 整体趋势 | **单调递增** | 视觉依赖越高，两者越正交 |

> **近乎正交的含义**：$\nabla \mathcal{L}_{\text{Lang}} \cdot \nabla \mathcal{L}_{\text{Vis}} \approx 0$。沿 $\nabla \mathcal{L}_{\text{Lang}}$ 走一步，对视觉目标的改善投影近似为 0，反之亦然。这意味着在 Bin 9 的 token 上，**学会说话和学会看是两件完全不同的事**——需要改动的参数/子结构几乎不重叠。

此外，Bin 9 处夹角进入**钝角区域**（$> 90^\circ$），出现破坏性梯度干扰：往视觉方向走一步，会给语言梯度带来**负投影**，意味着"纯视觉方向优化 = 反学习语言先验"。这是 Section 4 中 Language Preservation 的设计动机。

#### 发现 II：标准梯度是被动的角平分线（Figure 3-b, 3-c）

| 对比维度 | $\angle(\nabla \mathcal{L}_{\text{Standard}}, \nabla \mathcal{L}_{\text{Vis}})$ | $\angle(\nabla \mathcal{L}_{\text{Standard}}, \nabla \mathcal{L}_{\text{Lang}})$ |
|----------|:--------------------------------------------------------------------------:|:----------------------------------------------------------------------------:|
| 各 bin 值 | 始终 $\approx 42^\circ$ | 始终 $\approx 50^\circ$ |
| 跨 bin 变化 | 几乎不变 | 几乎不变 |

即使在 Bin 9（最需要视觉的 token），标准梯度仍维持离 $\nabla \mathcal{L}_{\text{Vis}}$ 约 $42^\circ$、离 $\nabla \mathcal{L}_{\text{Lang}}$ 约 $50^\circ$ 的固定方向——**静态地平分两个信号**，对二者的重视程度始终一样（各约一半），完全忽略 token 的实际视觉需求。

#### 核心假设：打破优化对称性（Asymmetric Maturity Hypothesis）

> 标准目标在一个**本质上不对称**的任务上强制施加了几何对称性。由于性能受限于感知瓶颈，且语言梯度在高依赖区与视觉改善正交，标准的折中更新是低效的。我们提出：显式将梯度导向视觉子空间，为优先解决感知歧义引入必要的归纳偏置。

核心支撑逻辑链：
1. 性能瓶颈是视觉 → 应该多看
2. $\nabla \mathcal{L}_{\text{Lang}} \perp \nabla \mathcal{L}_{\text{Vis}}$ → 往语言方向走对视觉没帮助
3. 标准优化 50:50 平分 → 在不需要语言的 token 上浪费了大量优化预算
4. 结论：**把预算掰给视觉**。

---

### 4. 方法：Visual Gradient Steering (VGS)

#### 导向目标（The Steered Objective）

定义 trajectory-wise 导向目标，在标准蒸馏损失上附加视觉辅助项：

$$
\ell_{\text{VGS}}(\tau) \triangleq \ell_{\text{Standard}}(\tau) + \gamma \cdot \ell_{\text{Vis}}(\tau) \tag{13}
$$

其中 $\gamma \geq 0$ 是控制视觉修正强度的**导向系数**。最终损失为期望的导向目标：

$$
\mathcal{L}_{\text{VGS}} = \eta_{\text{VGS}}(\gamma) \cdot \mathbb{E}_{\tau \sim p_{\theta_S}(\cdot \mid I, x)}[\ell_{\text{VGS}}(\tau)] \tag{14}
$$

---

#### 梯度范数归一化（Gradient Norm Normalization）

为保证优化稳定，引入缩放因子 $\eta_{\text{VGS}}(\gamma)$，使导向后的梯度**范数等于原标准梯度的范数**——只改变方向，不改变步长：

$$
\eta_{\text{VGS}}(\gamma) \triangleq \frac{\|\nabla_\theta \mathcal{L}_{\text{Standard}}\|_2}{\|\nabla_\theta \mathcal{L}_{\text{Standard}} + \gamma \nabla_\theta \mathcal{L}_{\text{Vis}}\|_2} \tag{15}
$$

> **设计动机**：将"导向"（由 $\gamma$ 控制）和"学习率"（由优化器控制）解耦。这是多任务学习中广泛采用的原则（如 GradNorm, Chen et al. 2018）。

> **工程简化**：附录 A 验证了训练过程中 $\|\nabla \mathcal{L}_{\text{Standard}}\|$、$\|\nabla \mathcal{L}_{\text{Vis}}\|$、以及二者夹角的余弦均保持稳定。因此将 $\eta_{\text{VGS}}(\gamma)$ 设为仅依赖 $\gamma$ 和模型架构的固定常数（2B: $\eta = 0.41$，4B: $\eta = 0.36$），避免运行时计算开销。

将 Eq. 15 的分子分母展开：

$$
\eta_{\text{VGS}}(\gamma) = \frac{\|\nabla \mathcal{L}_{\text{Standard}}\|_2}{\sqrt{A + B + C}} \tag{23}
$$

其中：
- $A = \|\nabla \mathcal{L}_{\text{Standard}}\|_2^2$
- $B = \gamma^2 \|\nabla \mathcal{L}_{\text{Vis}}\|_2^2$
- $C = 2\gamma \|\nabla \mathcal{L}_{\text{Standard}}\|_2 \cdot \|\nabla \mathcal{L}_{\text{Vis}}\|_2 \cdot \cos\phi$
- $\phi = \angle(\nabla \mathcal{L}_{\text{Standard}}, \nabla \mathcal{L}_{\text{Vis}})$

---

#### 缓解破坏性梯度干扰：Language Preservation (LP)

在高视觉依赖 token（Bin 7-9，即 VDS 的 top 30%）处，$\nabla \mathcal{L}_{\text{Lang}}$ 和 $\nabla \mathcal{L}_{\text{Vis}}$ 的夹角进入钝角区（$> 90^\circ$），纯视觉优化会造成语言先验的"反学习"。为此引入 LP 正则项，**仅选择性地**应用于 VDS 最高的 30% token：

$$
\ell_{\text{LP}}(\tau) \triangleq \frac{1}{|\tau|} \sum_{t=1}^{|\tau|} \mathbf{1}[\text{VDS}_t > Q_{0.7}] \cdot D_{\mathrm{KL}}\big(p_{\theta_S}(\cdot \mid \tau_{<t}, x) \;\big\|\; q_T(\cdot \mid \tau_{<t}, x)\big) \tag{16}
$$

其中 $Q_{0.7}$ 是 VDS 分布的 70 分位数阈值。

> **直觉**：只在视觉导向信号可能造成破坏时（VDS 高、夹角钝），才惩罚偏离教师语言先验的行为。注意 LP 的 KL target 是 $q_T(\cdot \mid \tau_{<t}, x)$（教师纯文本分布），而非 $q_T^*$——LP 保护的是语言，跟视觉 Grounding 的 target 是不同的分布。低中 VDS 区间的 token 不需要保护——$\gamma$ 的导向在那里安全且有效。

---

#### 最终目标：VGS-LP

$$
\ell_{\text{VGS-LP}}(\tau) = \ell_{\text{VGS}}(\tau) + \lambda \ell_{\text{LP}}(\tau) = \ell_{\text{Standard}}(\tau) + \gamma \ell_{\text{Vis}}(\tau) + \lambda \ell_{\text{LP}}(\tau) \tag{17}
$$

$$
\mathcal{L}_{\text{VGS-LP}} = \eta_{\text{VGS}}(\gamma) \cdot \mathbb{E}_{\tau \sim p_{\theta_S}(\cdot \mid I, x)}[\ell_{\text{VGS-LP}}(\tau)] \tag{18}
$$

参数分工：
- $\gamma \geq 1$（通常 $=2.0$）：**进攻**——驱动视觉适应
- $\lambda \approx 0.01$：**防守**——保守地防止灾难性遗忘，仅在视觉-语言冲突区生效

最终效果（Figure 4 验证）：
- $\mathcal{L}_{\text{Vis}}$（视觉 loss）：VGS 显著加速视觉学习，尤其在 Bin 7-9 的高依赖 token 上
- $\mathcal{L}_{\text{Lang}}$（语言 loss）：VGS-only（无 LP）会导致语言先验发散；VGS-LP 成功阻止此退化，同时维持优越的视觉 grounding

---

### 7. 验证 Asymmetric Maturity Hypothesis：逆实验

假设视觉是主要瓶颈，语言已相对成熟。通过构造**对称的语言导向** $L_{\text{Lang-Steer}}$ 来验证：

$$
\mathcal{L}_{\text{Lang-Steer}} = \eta_{\text{Lang}}(\gamma_{\text{lang}}) \cdot \mathbb{E}_{\tau \sim p_{\theta_S}(\cdot \mid I, x)}\big[\ell_{\text{Standard}}(\tau) + \gamma_{\text{lang}} \cdot \ell_{\text{Lang}}(\tau) + \lambda \cdot \ell_{\text{VP}}(\tau)\big] \tag{19}
$$

其中 $\eta_{\text{Lang}}(\gamma)$ 的范数归一化与 VGS 对称：

$$
\eta_{\text{Lang}}(\gamma) \triangleq \frac{\|\nabla_\theta \mathcal{L}_{\text{Standard}}\|_2}{\|\nabla_\theta \mathcal{L}_{\text{Standard}} + \gamma_{\text{lang}} \nabla_\theta \mathcal{L}_{\text{Lang}}\|_2} \tag{20}
$$

$\ell_{\text{VP}}(\tau)$ 是**视觉保护**正则项（与 LP 对称），仅在高视觉依赖 token 上惩罚偏离教师视觉分布：

$$
\ell_{\text{VP}}(\tau) \triangleq \frac{1}{|\tau|} \sum_{t=1}^{|\tau|} \mathbf{1}[\text{VDS}_t > Q_{0.7}] \cdot D_{\mathrm{KL}}\big(p_{\theta_S}(\cdot \mid \tau_{<t}, x) \;\big\|\; q_T^*(\cdot \mid \tau_{<t}, x)\big) \tag{21}
$$

> **结果**：$\gamma_{\text{lang}} > 0$（掰向语言）→ 性能下降至 baseline 以下；$\gamma > 0$（掰向视觉）→ 性能持续提升。确认了视觉才是瓶颈，语言已足够成熟。

---

### 8. RL Fine-Tuning + On-Policy Distillation

VGS 可作为 GRPO 的正则器，与 RL 奖励构成正交约束：

$$
J_{\text{Total}}(\theta) = \mathbb{E}_{\{\tau_i\}_{i=1}^G \sim p_{\theta_S}} \left[ \frac{1}{G} \sum_{i=1}^G \big( (1 - \alpha) \cdot O_{\text{GRPO}}(\tau_i) - \alpha \cdot \ell_{\text{VGS-LP}}(\tau_i) \big) \right] \tag{22}
$$

其中 $O_{\text{GRPO}}(\tau_i)$ 是标准 GRPO 替代目标（clipped advantage，无 KL 惩罚），$\alpha = 0.3$。

> **直觉**：RL 优化最终答案的正确性，VGS 确保推理过程保持视觉 grounding。二者**正交互补**——RL 只管"对错"，VGS 管"过程有没有看图"。

---

### 附录 E. 自适应 Token-Level VGS（Adaptive VGS）

固定 $\gamma$ 对所有 token 一视同仁。自适应版本根据每 token 的 VDS 动态缩放导向强度：

$$
\ell_{\text{Adaptive}}(\tau) \triangleq \frac{1}{|\tau|} \sum_{t=1}^{|\tau|} \big( \ell_{\text{Standard}}(\tau) + \gamma_t \cdot \ell_{\text{Vis}}(\tau) + \lambda \cdot \ell_{\text{LP}}(\tau) \big) \tag{24}
$$

其中 token-level 导向系数 $\gamma_t$ 为分段函数：

$$
\gamma_t = \begin{cases}
0, & \text{VDS}_t \leq Q_{0.4} \quad \text{（无视觉修正）} \\
\gamma/2, & Q_{0.4} < \text{VDS}_t \leq Q_{0.7} \quad \text{（中等修正）} \\
\gamma, & \text{VDS}_t > Q_{0.7} \quad \text{（最大视觉导向）}
\end{cases} \tag{25}
$$

> **结果**：Adaptive VGS 略优于固定 $\gamma$（如 VisualPuzzles: 33.99 vs 30.64），验证了按 token 视觉需求分配导向预算的有效性。但作者将其作为附录——三阶段阈值仍是粗糙设计，平滑的动态路由远未解决。

---

## Key Results

- **VGS vs Standard OPD**：在 7 个 VLM 推理 benchmark 上一致超越 baseline。8B→2B 设置下平均 Acc@1 从 43.74% → 46.10%（+2.37%），8B→4B 从 56.64% → 58.12%（+1.56%）。
- **视觉密集型任务受益最大**：VisualPuzzles（+3.68%）、LogicVista（+3.35%）、MathVerse-VD（+2.08%）提升最显著。
- **语言导向有害**：$\gamma_{\text{lang}} > 0$ 使性能降至 baseline 以下，验证 Asymmetric Maturity Hypothesis。
- **VGS + GRPO 联合**：GRPO + VGS 优于纯 GRPO 和 GRPO + Standard-KD，VGS 的正则化同时防止了 GRPO 的 length explosion。
- **低视觉依赖 / 纯文本任务不受影响**：VGS 在 Geo3K、We-Math、MATH500 等任务上性能与 baseline 持平——掰向视觉不伤害文本推理。
- **计算开销**：VGS 需要额外一次 text-only forward pass（构造 $q_T^*$ 和 $L_{\text{Vis}}$），每步训练时间增加约 1.375×。
- **Adaptive VGS**（附录 E）略优于固定 $\gamma$，但改进有限——说明三阶段硬阈值非最优方案。

---

## Assumptions

1. **Asymmetric Maturity Hypothesis**：视觉感知是当前 VLM 的性能瓶颈，语言先验已相对成熟。此假设在 Qwen3-VL-8B→2B/4B 蒸馏中获得实验验证，但不保证对其他 teacher-student pair 成立。

2. **教师分布的对比信号有效**：VGS 依赖教师的多模态和单模态分布可被可靠区分（$\nabla \mathcal{L}_{\text{Vis}} \not\approx 0$）。如果教师本身对视觉不敏感（mode collapse），VGS 退化为标准 OPD。

3. **语言-视觉二元分解是完备的**：本文仅分解为语言先验和视觉 Grounding。实际 VLM 推理可能涉及更细粒度的子能力（空间推理、OCR、数值推理、跨模态融合），它们在 $\nabla \mathcal{L}_{\text{Vis}}$ 内部可能存在未被发现的冲突。

4. **全局 $\gamma$ 的充分性**：主要实验中所有 token 共享同一 $\gamma$。自适应版（附录 E）证明 per-token 差异化有益但有限——暗示更平滑、更细粒度的 token 级路由是开放问题。

---

## Limitations / Failure Modes

### 论文自述的限制（附录 G）

1. **训练吞吐量开销**：VGS 需要额外的 text-only forward pass 来分解教师的输出分布，每步训练时间增加约 $1.375\times$。在多教师场景下成本乘以教师数。

2. **依赖教师校正**：如果教师模型本身存在 mode collapse——视觉输入对输出分布无影响（$\nabla \mathcal{L}_{\text{Vis}} \approx 0$）——VGS 退化为标准 OPD。方法只能增强视觉 Grounding 的传输，无法修复教师固有的感知盲区。

### 未明说的限制

3. **仅 Qwen3-VL 架构**：视觉编码器架构（ViT 变体）影响梯度正交性程度。不同架构（SigLIP vs. CLIP vs. DINO）的 $\nabla \mathcal{L}_{\text{Vis}}$ 几何可能完全不同，VGS 的超参数（$\gamma$, $\eta$）需要重新校准。

4. **单教师假设**：全部分析基于"一个教师同时提供语言和视觉信号"。当存在多个专门化教师时，分解和导向逻辑需要根本性重新设计——谁的语言先验？谁的视觉 Grounding？

5. **LP 的粗糙性**：Language Preservation 使用全局 70 分位数 hard threshold + 固定 $\lambda = 0.01$。Bin 7（刚过阈值）和 Bin 9（极端冲突）需要不同强度的保护，且阈值应依赖任务类型（纯视觉 vs. 混合）。

6. **仅在推理基准上验证**：无 captioning、VQA、visual grounding、video 上的实验。不同任务类型的 VDS 分布和梯度几何可能差异显著。

7. **无理论保证**：正交性是实验观察（empirical observation），没有理论证明正交性何时成立、何时失败、是否在不同架构/数据分布下保持。

---

## Reusable Ingredients

- **Bayes 分解 OPD 损失**：将多模态 OPD 的 KL 分解为语言先验 + 视觉信息增益的一般框架，可复用于任何 VLM 蒸馏分析。
- **VDS 作为 token 级视觉依赖性诊断工具**：$\text{VDS}_t = D_{\text{KL}}(q_T(\cdot \mid \tau_{<t}, I, x) \parallel q_T(\cdot \mid \tau_{<t}, x))$，可在不依赖额外标注的情况下诊断每个 token 对视觉的需求程度。
- **梯度夹角的 bin 分析**：按 VDS 分桶统计 $\nabla \mathcal{L}_{\text{Lang}}$ 与 $\nabla \mathcal{L}_{\text{Vis}}$ 的夹角，是对优化几何进行细粒度诊断的通用方法——可推广到 $\nabla \mathcal{L}_{\text{Reasoning}}$ vs $\nabla \mathcal{L}_{\text{Knowledge}}$ 等任意分解。
- **梯度范数归一化保持步长稳定**：$\eta(\gamma) \triangleq \frac{\|\nabla L_{\text{Standard}}\|_2}{\|\nabla L_{\text{Standard}} + \gamma \nabla L_{\text{Vis}}\|_2}$，将方向控制和步长控制解耦，可复用于任何多目标梯度组合。
- **逆实验验证瓶颈假设**：对称构造相反方向的导向（视觉→语言），以验证"哪个模态是真正瓶颈"——可复用于任何声称某模态是瓶颈的工作。
- **LP 的选择性正则**：仅在梯度冲突区域（钝角区）施加保护性正则，而非全 token 统一加——选择性保护的原则比 hard threshold 本身更具复用价值。

---

## Open Questions

- VDS 基于教师分布计算。对学生而言，某个 token 的视觉依赖可能与教师完全不同——教师觉得需要看图，学生可能视觉编码器弱到获取不到信号。**学生 VDS** 和 teacher VDS 的交叉分析是缺失的。
- 二元分解（语言 + 视觉）是否完备？在 OCR 任务中，"读图中文字"和"看物体空间关系"真的是同一梯度方向吗？如果 $\nabla \mathcal{L}_{\text{OCR}}$ 和 $\nabla \mathcal{L}_{\text{Spatial}}$ 之间也存在钝角冲突，VGS 的全局 $\gamma$ 无法区分对待。
- 当性能瓶颈不是视觉时（如数学推理是瓶颈，视觉编码器已经很强大），VGS 的逻辑是否完全翻转？此时 $\gamma_{\text{lang}} > 0$ 可能反而有效。
- Adaptive VGS（Eq. 25）的三阶段硬阈值远非最优。能否学习一个**平滑的 $\gamma_t = f(\text{VDS}_t)$ 函数**，甚至让 $\gamma_t$ 同时依赖 VDS 和当前训练阶段？
- $\eta_{\text{VGS}}(\gamma)$ 被近似为固定常数，虽经实验验证在训练中稳定，但在不同数据集、不同 teacher-student pair 下是否仍稳定？如果不稳定，需要什么级别的动态估计？

---

## Claims

- **Claim 1：标准 VLM OPD 的梯度是被动的角平分线。** $\nabla \mathcal{L}_{\text{Standard}}$ 始终取语言和视觉方向的固定折中（$\approx 42^\circ$ vs $\approx 50^\circ$），无论 token 的视觉需求有多大。

- **Claim 2：语言和视觉梯度在高视觉依赖 token 上近乎正交。** Bin 9 上夹角 $\approx 92^\circ$，说明学会说话和学会看是两件近乎独立的事。

- **Claim 3：由于 Asymmetric Maturity，掰向视觉是正确方向。** 语言先验已相对成熟，视觉感知是瓶颈。逆实验（掰向语言）导致性能下降，验证此假设。

- **Claim 4：VGS 以极小开销显著超越 Standard OPD。** $1.375\times$ 训练时间增加换来平均 +2.37%（2B）和 +1.56%（4B）的 Acc@1 提升。

- **Claim 5：VGS 可作为 RL 的正则化器。** VGS + GRPO 联合优化中，VGS 确保过程视觉 grounded，RL 优化最终答案正确性——二者互补。

---

## Connections

- **OPD Rethinking**：OPD 论文揭示了 teacher-student thinking pattern overlap 是 OPD 成功的前提。DecomposedOPD 进一步揭示了：即使在 overlap 成立时，标准的优化几何仍可能导致次优——因为语言和视觉的梯度是被等权平分的。二者分别从"信号是否可用"和"可用信号如何被优化"两个维度切入。

- **MOPD / 多教师蒸馏**：DecomposedOPD 的分解和导向完全基于单教师假设。多教师场景下，$L_{\text{Lang}}$ 和 $L_{\text{Vis}}$ 各来自哪个教师？视觉教师提供 $\nabla L_{\text{Vis}}$、推理教师提供 $\nabla L_{\text{Lang}}$，二者来自不同模型，其梯度夹角可能比单教师场景的 92° 更大或更小。且 LP 中保护的是哪个教师的语言先验？这是一个新的多目标 routing 问题——当前方法无法直接套用。

- **Pivot (Song et al. 2026)**：Pivot 关注视觉 grounding 的评估。DecomposedOPD 提供了从梯度层面诊断 grounding 质量的方法——VDS 和 $\angle(\nabla L_{\text{Lang}}, \nabla L_{\text{Vis}})$ 可以作为 grounding 质量的动态指标。

- **PDCR (Yoon et al. 2026)**：同组工作，将感知和推理分解用于 RL reward 设计。与 DecomposedOPD 的分解思想一脉相承，区别在于 PDCR 用于 RL reward，DecomposedOPD 用于 OPD 梯度导向。

---

## Relevance to This Project

DecomposedOPD 是我们的 MOPD 方法路径上最重要的一篇直接前置工作。它将 OPD 从"一个损失函数"变成"一个可分解的几何问题"，为多教师场景提供了分析语言。

核心迁移点：

1. **从二元分解到多元分解**：DecomposedOPD 证明标准 OPD = $L_{\text{Lang}} + L_{\text{Vis}}$。我们的 MOPD 自然地推广为：标准 MOPD = $\sum_k w_k (L_{\text{Lang}}^{(k)} + L_{\text{Vis}}^{(k)})$，其中 $k$ 索引教师。关键新问题：不同教师的 $L_{\text{Vis}}^{(k)}$ 之间是否存在梯度冲突？教师的专门化程度是否影响各时间步的几何？

2. **从全局 $\gamma$ 到 token 级教师权重矩阵**：DecomposedOPD 用一个全局 $\gamma$ 掰向视觉。MOPD 需要在每个 token 上做 K 维的教师选择，且权重 $(w_1, \ldots, w_K)$ 需随 token 平滑变化。这不是"调一个 $\gamma$"的问题，而是**在线教师路由**问题。

3. **从单瓶颈假设到多瓶颈动态识别**：DecomposedOPD 假设"视觉永远是瓶颈"。MOPD 中不同任务类型的瓶颈不同——数学推理 token 需要推理教师，视觉描述 token 需要视觉教师。需要一种机制动态识别当前 token 的瓶颈模态，并路由到对应教师。

4. **LP 的多教师泛化**：单教师的 LP 只保护语言先验。多教师中"掰"一个教师可能损害其他多个教师的监督信号。需要多目标帕累托优化，而非单正则项。

因此我们的表述应该是：

> DecomposedOPD revealed that in single-teacher OPD, language and visual gradients are nearly orthogonal, and standard optimization passively compromises between them. Our MOPD generalizes this geometric perspective to the multi-teacher regime: when K specialized teachers provide supervision on the same student-visited state, the gradient geometry is no longer a 2D bisector but a K-dimensional simplex requiring token-level routing with smoothness constraints, inter-teacher conflict resolution, and bottleneck-adaptive steering.

---

## Reading Notes

### 核心概念

**全文涉及的分布速查**——这是理解整篇论文最关键的 notation 基础：

| 符号 | 含义 | 出现在 |
|------|------|--------|
| $q_T(\cdot \mid \tau_{<t}, I, x)$ | 教师多模态分布（同时看图文） | Eq.3, 12 |
| $q_T(\cdot \mid \tau_{<t}, x)$ | 教师纯文本分布（不看图） | Eq.6, 12, 16 |
| $p_{\theta_S}(\cdot \mid \tau_{<t}, I, x)$ | 学生多模态分布 | Eq.3, 11 |
| $p_{\theta_S}(\cdot \mid \tau_{<t}, x)$ | 学生纯文本分布 | Eq.6, 10 |
| $q_T^*(\cdot \mid \tau_{<t}, I, x)$ | **构造混合分布**：学生语言先验 + 教师视觉增益 | Eq.9-11, 21 |

> **关键区分**：$q_T$ 是一个模型，上面两行是**同一个模型在不同输入条件下**的分布。$q_T^*$ 不是 $q_T$ 的变体——它是由 $p_{\theta_S}$ 和 $q_T$ 拼接构造出的**新分布**，语言底座来自学生，视觉增量来自教师。

- **Language Prior ($\mathcal{L}_{\text{Lang}}$)**：在纯文本条件 $(x, \tau_{<t})$ 下匹配教师的纯文本分布 $q_T(\cdot \mid \tau_{<t}, x)$，传递推理风格，不涉及视觉上下文。KL target 是 $q_T$（教师），不是 $q_T^*$。
- **Visual Grounding ($\mathcal{L}_{\text{Vis}}$)**：KL target 是 $q_T^*$（而非 $q_T(\cdot \mid \tau_{<t}, I, x)$）——因为只用教师的视觉信息增量嫁接在学生的语言底座上，实现视觉和语言的干净分离。
- **Visual Dependency Score (VDS)**：教师在 token $t$ 的多模态分布与单模态分布的 KL 散度，量化 token 对视觉的依赖程度。是后续所有分析和 LP 正则化的基础。
- **Gradient Orthogonality**：$\nabla \mathcal{L}_{\text{Lang}} \perp \nabla \mathcal{L}_{\text{Vis}}$ 在高 VDS token 上成立——两个目标的优化需要改动模型的不同参数子空间。
- **Passive Bisector**：$\nabla \mathcal{L}_{\text{Standard}}$ 的行为模式——始终平分语言和视觉方向，是"优化对称性"的引擎。
- **Asymmetric Maturity Hypothesis**：视觉是瓶颈，语言已成熟→掰向视觉是正确的方向。

### 机制理解

DecomposedOPD 的本质是**在 OPD 的优化几何上做了手术**。它不改变"用 KL 匹配教师"这件事，只改变"梯度往哪里走"。

将这个机制画成参数空间的几何图景：

```
                     Standard OPD  (Bin 9)
                     ==============
       (visual)
         ^
         |     /  <- grad L_Standard (passive bisector, ~42 deg from visual)
         |    /
         |   /
         |  /
         | /
         O---------->  (language)
              grad L_Lang

         Angle(grad L_Lang, grad L_Vis) ~= 92 deg (nearly orthogonal)
         Standard update splits the difference -> slow visual progress


                     VGS  (same Bin 9 token)
                     ===
       (visual)
         ^
         |  /
         | /   <- grad VGS = eta * (grad Standard + gamma * grad Vis)
         |/       (steered toward visual subspace)
         |
         O---------->  (language)
              grad L_Lang

         Standard: compromise between two near-orthogonal directions
         VGS:      explicitly biased toward visual -> faster grounding
                   LP regularizer prevents language unlearning
```

梯度正交性的物理意义是：提升语言能力和提升视觉感知需要改动模型的不同参数子结构。标准 OPD 把更新向量平分，意味着在所有参数上平均分配——相当于同时在两个近乎独立的子空间上做半速优化。VGS 把更新向量掰向视觉子空间，在瓶颈处集中资源。

### Recipe / 实践细节

- $\gamma = 2.0, \lambda = 0.01$：主导向系数和 LP 权重，经 sensitivity analysis 验证在宽范围内鲁棒。
- $\eta_{\text{VGS}}(2.0) = 0.41$（2B）、$0.36$（4B）：由附录 A 的梯度范数追踪实验预计算，训练中固定不变。
- text-only forward pass：每步额外跑一次不带图像的 forward（用 $x$ 和 $\tau_{<t}$），计算 $q_T(\cdot \mid \tau_{<t}, x)$ 和 $p_{\theta_S}(\cdot \mid \tau_{<t}, x)$。
- $q_T^*$ 的构造在 logit 层面：`student_text_logits + (teacher_mm_logits - teacher_text_logits)`，Softmax 后即为目标分布。注意这里 $q_T^*$ 的 logit = 学生的语言底座 + 教师的视觉增量（多模态 logit 减纯文本 logit），不是直接用教师的原始 logit。
- 统一 system prompt：教师 GRPO 训练和学生 OPD 蒸馏使用同一模板，强制 `<reason>...</reason>` + `\boxed{}` 格式。
- Vision encoder 不冻结：训练中同时更新视觉编码器和 LLM backbone。
- 1 epoch, lr=1e-6, AdamW, batch=512, rollout temperature=1.0, top-p=1.0。
- 8×A100 80GB 单节点。

### 值得复现或借鉴的实验

- **梯度夹角 vs VDS binned analysis（Figure 3）**：在 MOPD 中可推广为——对每个教师的 $\nabla L_{\text{Vis}}^{(k)}$ 和 $\nabla L_{\text{Lang}}^{(k)}$，以及跨教师的 $\nabla L_{\text{Vis}}^{(k)}$ vs $\nabla L_{\text{Vis}}^{(j)}$，做相同的分桶夹角分析。这是理解多教师梯度几何的起点。
- **逆实验验证瓶颈假设（Section 7）**：在 MOPD 中可推广为——分别关闭某个教师，观察哪些能力受损最严重，以识别真正的瓶颈教师。
- **η 的稳定性和近似可行性（Appendix A）**：在 MOPD 中检查多教师组合梯度范数是否仍然稳定，决定了是否可以做类似的常数近似。
- **Adaptive steering 的三阶段阈值（Appendix E）**：在 MOPD 中作为最简单的 baseline——用 VDS 分三段决定掰向哪个教师（低 VDS→语言教师，中 VDS→推理教师，高 VDS→视觉教师）——然后证明我们需要比三阶段更平滑的路由。

### 和 MOPD / 多教师蒸馏的关系

DecomposedOPD 为 MOPD 提供了**分析范式**而非直接可套用的方法。其最重要的遗产是：

1. **证明了"分解→分析几何→定向干预"这条技术路线有效**。MOPD 可以沿着相同的路线走：分解各教师贡献→分析跨教师梯度几何→设计 token 级路由干预。

2. **暴露了全局静态控制信号的不足**。VGS + LP 本质上是一个全局 $\gamma$ + 一个 hard threshold 正则项。Adaptive VGS（附录 E）的三阶段改进也仍是离散的。MOPD 需要的平滑、连续、token 级、多教师的路由信号，是比 VGS 严格更难的问题。

3. **给定了 MOPD 的下界**。对任意一个 token，如果没有 token 级路由，我们能做的最好就是 VGS 式的全局偏见（偏某个教师/某个模态）。VGS 相对于 Standard OPD 的提升，就是 token 级路由理论上能获得的上限增益的一小部分。

多教师时每 token 需要回答的不是"掰多少给视觉"，而是"掰多少给视觉教师 vs 推理教师 vs 知识教师"，且这个决策需要满足：
- Token 间平滑：相邻 token 的教师权重不能剧烈跳变
- 教师间冲突检测：两个教师对同一 token 的梯度方向是否冲突
- 学生能力感知：学生当前在哪个教师擅长领域最弱，应接受更多该教师的信号
