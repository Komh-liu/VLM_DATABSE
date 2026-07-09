---
type: paper
short: "OPD"
node_id: paper:opd_rethinking_2026
title: "Rethinking On-Policy Distillation of Large Language Models: Phenomenology, Mechanism, and Recipe"
authors: ["Yaxuan Li", "Yuxin Zuo", "Bingxiang He", "Jinqian Zhang", "Chaojun Xiao", "Cheng Qian", "Tianyu Yu", "Huan-ang Gao", "Wenkai Yang", "Zhiyuan Liu", "Ning Ding"]
year: 2026
venue: "arXiv"
external_ids:
  arxiv: "2604.13016"
  doi: "10.48550/arXiv.2604.13016"
  s2: null
tags: ["on-policy-distillation", "knowledge-distillation", "LLM", "post-training", "alignment"]
added: 2026-07-06T00:00:00Z
pdf: "opd_rethinking_2026.pdf"
---

# Rethinking On-Policy Distillation of Large Language Models: Phenomenology, Mechanism, and Recipe

## One-line thesis

> 待补充。

## Problem / Gap

虽然OPD已经得到广泛应用，但是很少有人研究为什么更好的模型完全不能优化学生模型，而弱一些的模型可以在初始对齐度更低的情况下成功。

### 问题设定：prompt、response 与 next-token distribution

令输入 prompt 为 $x = (x_1, \ldots, x_n)$，模型回复为 $y = (y_1, \ldots, y_m)$，并记 $y_{<t} \triangleq (y_1, \ldots, y_{t-1})$ 表示到第 $t$ 步之前的回复前缀。论文考虑两个 LLM：学生模型 $\pi_\theta$ 和教师模型 $\pi_T$；二者都会在词表 $\mathcal{V}$ 上定义一个 next-token distribution $\pi(\cdot \mid x, y_{<t})$，也就是给定输入 prompt $x$ 和当前已经生成的前缀 $y_{<t}$ 时，模型对下一个 token 的概率分布。记 $y \sim \pi_\theta(\cdot \mid x)$ 表示回复 $y$ 是由学生模型 $\pi_\theta$ 根据 prompt $x$ 自回归采样生成的。固定数据集记为 $\mathcal{D} = \{(x^{(i)}, y^{(i)})\}_{i=1}^{N}$，其中 response 是教师模型生成的输出；对应的 prompt 集合记为 $\mathcal{D}_x = \{x^{(i)}\}_{i=1}^{N}$。知识蒸馏（KD）的目标是通过最小化教师模型 $\pi_T$ 和学生模型 $\pi_\theta$ 这两个分布之间的差异，把知识从教师模型转移到学生模型。一种标准做法是使用 KL 散度。对于定义在词表 $\mathcal{V}$ 上的两个分布 $P$ 和 $Q$，KL 散度定义为 $D_{\mathrm{KL}}(P \parallel Q) = \sum_{v \in \mathcal{V}} P(v)\log\frac{P(v)}{Q(v)}$。直观地说，KL 散度衡量的是：如果真实分布是 $P$，但用 $Q$ 去近似它，会产生多大的信息损失。

## Method

### On-Policy Distillation objective

OPD 在当前学生模型 $\pi_\theta$ 采样得到的轨迹上计算监督信号。给定 prompt $x \sim \mathcal{D}_x$，学生模型采样一个回复 $\hat{y} = (\hat{y}_1, \ldots, \hat{y}_T) \sim \pi_\theta(\cdot \mid x)$，其中 $T \triangleq |\hat{y}|$ 表示 rollout 长度。随后，学生模型和教师模型都会在学生生成的前缀 $\hat{y}_{<t}$ 上被评估，从而在每一步 $t$ 得到两个 next-token distributions：$p_t(v) \triangleq \pi_\theta(v \mid x, \hat{y}_{<t})$ 和 $q_t(v) \triangleq \pi_T(v \mid x, \hat{y}_{<t})$，其中 $v \in \mathcal{V}$。标准 OPD 形式是在学生生成的轨迹上最小化 sequence-level reverse KL：

$$
\mathcal{L}_{\mathrm{OPD}}(\theta)
= \mathbb{E}_{x \sim \mathcal{D}_x}
\left[
D_{\mathrm{KL}}
\left(
\pi_\theta(\cdot \mid x)
\parallel
\pi_T(\cdot \mid x)
\right)
\right]
$$

利用自回归分解，上面的 sequence-level objective 可以精确分解为 token-level KL：

$$
\mathcal{L}_{\mathrm{OPD}}(\theta)
= \mathbb{E}_{x \sim \mathcal{D}_x, \hat{y} \sim \pi_\theta(\cdot \mid x)}
\left[
\sum_{t=1}^{T}
D_{\mathrm{KL}}(p_t \parallel q_t)
\right]
$$

实践中，不同 OPD 实现主要区别在于如何计算这个逐 token 的 reverse KL：full-vocabulary OPD 直接优化上式；sampled-token OPD 对每个 token-level KL 项使用无偏 Monte Carlo 估计；top-k OPD 则用基于子集的近似替代 full-vocabulary KL。

 
- 采样OPD: 最轻量化的变种，仅仅评估学生采样的token，也是目前最广泛应用的工作。对于采样的序列 $\hat{y}$ 损失函数loss的计算式为 $log \frac {p_t(\hat{y})}{q_t(\hat{y})}$

- 全词表OPD: 对整个词表进行OPD

- Top-k OPD: 只对学生采样中的top-k个词进行计算KL散度

### 📊 动态指标 (Dynamic Metrics)

我们定义学生（Student）和教师（Teacher）模型在步骤 $t$ 时的 **Top-k 集合** 分别为 $S_t^{(p)}$ 和 $S_t^{(q)}$。在后续的实验中，将监控以下指标。

#### 1. 重叠率 (Overlap Ratio)
该指标量化了学生和教师候选空间（candidate spaces）之间的对齐程度。它被定义为同时出现在学生和教师 Top-k 集合中的 Token 的平均占比。

$$
 \mathcal{M}_{\text{overlap}} \triangleq \mathbb{E}_t \left[ \frac{|S_t^{(p)} \cap S_t^{(q)}|}{k} \right] 
$$

> **翻译与解释**：低重叠率表明学生模型的概率质量集中于与教师不同的 Token 集合上，暗示可能存在显著的概率分布分歧或“模式不匹配”（mode mismatch）。反之，当重叠率接近 1.0 时，说明学生模型已经成功定位到了教师模型的支持区域。


#### 2. 重叠 Token 优势 (Overlap-Token Advantage)
为了测量**重叠 Token 集合内部**的分布一致性，定义了 $A_t(v)$，其中 $\tilde{p}_t$ 和 $\tilde{q}_t$ 是在 $S_t^{(p)} \cap S_t^{(q)}$ 上重新归一化后的学生和教师分布。该指标取这一量的均值：

$$
 \mathcal{M}_{\text{adv}} \triangleq \mathbb{E}_t \left[ \frac{1}{|S_t^{(p)} \cap S_t^{(q)}|} \sum_{v \in S_t^{(p)} \cap S_t^{(q)}} A_t(v) \right]  
$$

> **翻译与解释**：当该值接近 0 时，表明学生模型在教师偏好的 Token 上赋予了适当的置信度，实现了高质量的对齐。相反，较大的负值则意味着在重叠的 Token 集内部，学生模型表现出了**过度自信（overconfident）**（即学生的高置信度 $p_t$ 对比教师的低置信度 $q_t$）。


#### 3. 熵与熵差 (Entropy and Entropy Gap)
为了监测分布的特性，我们追踪学生模型 $H(p_t)$ 和教师模型 $H(q_t)$ 两者的**熵**，并将**熵差**定义为：

$$
 \Delta H_t = |H(q_t) - H(p_t)| 
$$

> **翻译与解释**：$\Delta H_t$ 是一种**状态特异性**的模态对齐指标。较大的熵差表明学生在置信度和多样性上与教师存在显著的不匹配。随着训练收敛，该值趋近于 0，意味着学生模型成功地匹配了教师模型在其生成的轨迹上的**不确定性分布（uncertainty profile）**。
> 离线冷启动和教师对齐的prompt选择。
> 1. 离线冷启动，让学生模型在教师生成的rollouts上进行冷启动，使得学生模型能与教师模型的思维模式进行对齐。
> 2. 教师prompt选择，用教师模型的后训练数据加强和教师模型的对其能力。但是这也降低了模型在分布外prompt的熵值。

## Key Results

- **渐进式对齐**。OPD训练中，教师和学生的重叠topk的token逐渐稳定增加，而失败的OPD训练重叠的token有限
- **重叠充分性** 几乎所有的优化效果在重叠部分的top-k的token中出现，只有这些token能够对OPD训练贡献
因此提出了两个方法：**离线冷启动策略** 


## Assumptions

分析OPD成功或者失败的两个因素。

### 思维模式一致性
1. 学生模型和教师模型应该具有兼容的思考模型。（topk采样时最好在token分布高度重叠）如果思考模式不匹配，即使更高得分的模型也无法很好提升学生模型能力
作者使用qwen3-1.7B作为学生模型，qwen3-4B-nothink和qwen3-4B-GRPO作为教师模型，期望是因为base模型是nothink，学生模型会在GRPO版本学习到更好的分布。
![实验图1](../images/OPD_thinking_consistent.png)
### 新知识而不只是规模上的提升
2. 即使二者思维模式一致、教师得分更高，教师也必须提供学生在训练中尚未见过的真正新能力。如果教师模型和学生模型都是在同样的数据集上训练的，在对应的领域内会有相近的分布，导致学生能学习到极少的信号。
成功的OPD是以在学生高概率visit的token空间提供对齐来实现的，这部分token旨在词表中占据一小部分，但是覆盖了绝大部分的情况。也有新的使用自己作为教师模型的方法--OPSD
实验中使用相同模型，在新的数据集上做RL的模型最终在能力提升上比没有RL过的模型效果更好。此外定义gap recovery rate。更强的教师模型，学生模型能够相对学到的内容也更多

## Limitations / Failure Modes

密集的token level的reward不一定能够在长程任务表现良好。在推理深度提升时，reward的质量下降并且后续的token出现不稳定性。令人惊讶的是，即使失败的教师模型也能提供与 rollout 正确性全局相关的奖励信号，这表明失败并非源于信号质量，而是源于局部优化几何结构。

## Reusable Ingredients

- 对于OPD，需要关注教师模型与学生模型之间的thinking patterns。OPD训练会主动获取教师模型的思维模式，并覆盖学生的思维模式。基准测试和更高的分数不代表OPD能够获得新的知识，教师模型应当具有训练过程中尚未见过的知识。

## Open Questions

- 待补充。

## Claims

- 待补充。

## Connections

- **Related:** 待补充。

## Relevance to This Project

待补充。

## Reading Notes

### 核心概念

待补充。

### 机制理解

待补充。

### Recipe / 实践细节

待补充。

### 值得复现或借鉴的实验

待补充。

### 和 MOPD / 多教师蒸馏的关系

待补充。
