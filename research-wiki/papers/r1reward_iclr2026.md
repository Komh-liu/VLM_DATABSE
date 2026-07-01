---
type: paper
node_id: paper:r1reward_iclr2026
title: "R1-Reward: Training Multimodal Reward Model Through Stable Reinforcement Learning"
short: "R1-Reward"
authors: ["Yi-Fan Zhang", "Xingyu Lu", "Xiao Hu", "Chaoyou Fu", "Bin Wen", "Tianke Zhang", "Changyi Liu", "Kaiyu Jiang", "Kaibing Chen", "Kaiyu Tang", "Haojie Ding", "Jiankang Chen", "Fan Yang", "Zhang Zhang", "Tingting Gao", "Di Zhang", "Guorui Zhou", "Liang Wang"]
year: 2026
venue: "ICLR 2026"
external_ids:
  arxiv: "2505.02835"
  doi: null
  s2: null
tags: ["reward-model", "multimodal", "reinforcement-learning", "RLHF", "StableReinforce", "long-CoT", "preference-optimization"]
added: 2026-06-29T00:00:00Z
---

# R1-Reward: 通过稳定强化学习训练多模态奖励模型

**机构**：中科院自动化所 (CASIA) + 清华 (THU) + 快手 (KuaiShou) + 南大 (NJU)
**代码**：https://github.com/yfzhang114/r1_reward
**模型**：https://huggingface.co/yifanzhang114/R1-Reward

---

## One-line thesis

> R1-Reward 首次将多模态 reward modeling 重新定义为 **rule-based RL 任务**：模型不再直接打分，而是先思考（`<think>`）再判断（`<answer>`），通过提出的 **StableReinforce** 算法解决 RL 训练 RM 时的三类不稳定问题（梯度爆炸、低方差 batch 扰动、推理-答案不一致），在 VL Reward-Bench 上提升 8.4%、Multimodal Reward Bench 上提升 14.3%，并观察到模型出现自我反思和纠错的 emergent behavior（"Aha Moment"）。

---

## 核心研究问题

> 现有的多模态 reward model 训练存在两个根本问题：
>
> **1. 隐式 RM（DPO）不够好**：DPO 跳过了显式 RM，用公式 $r = \beta \log(\pi_\theta / \pi_{\text{ref}})$ 反推偏好分数。RewardBench 已经证明这种隐式 RM 在分布外数据上泛化很差（换一个 reference model 甚至崩到随机水平）。多模态场景下问题更严重——视觉信息的加入让偏好判断更复杂，需要显式的推理过程。
>
> **2. 直接用 RL 训练 RM 会崩溃**：把 reward modeling 当成 RL 任务让模型自己推理，逻辑上很自然——但 Reinforce++ / PPO 等现有 RL 算法直接套上去会训练不稳定甚至崩溃。原因是：RM 任务的 reward 信号稀疏（只有最后判断对错）、长 CoT 推理的梯度链条极长、batch 内方差剧烈波动。
>
> R1-Reward 解决的就是这个问题：**如何用 RL 稳定地训练一个会思考的多模态 RM？**


### RL训练奖励模型存在的问题
1. PPO优势为负且当前策略模型和参考模型差距非常大时，由于min策略，模型可能最大化更新参数策略，简单的的clipping失效。
2. GRPO和Reinforce++不够稳定，可能会学习过大或者过小的优势值
3. 模型的推理能力和结果经常存在不连贯性。因为结果导向的奖励模型不会对cot进行有效监督。

- 原有的RM要么用MLLM作为奖励模型，但是该方法严重依赖MLLM的指令遵循能力和理解能力，要么直接训练输出分数，但是又缺乏了模型的可解释性。因此论文提出了一种先思考再打分的RM兼顾可解释性和无偏性。

### 关键洞察：为什么 RM 需要"思考"？

传统 RM（分类器式）直接输出一个标量分数，没有中间推理。但多模态偏好判断往往需要：
- 比较两张图的细节差异（哪张图里有幻觉？）
- 推理两个回答的逻辑一致性（哪个回答更合理？）
- 在模糊场景下权衡多个维度（准确性 vs 完整性 vs 安全性）

这些判断不是"看一眼就能打分"的——需要 **Long-CoT 推理**。R1-Reward 的核心 bet 就是：**让 RM 像 DeepSeek-R1 一样思考，它会变得更准**。

---

## 方法：StableReinforce 算法

### 整体框架

R1-Reward 将 reward modeling 重构为 rule-based RL 任务：

```
输入：Question + Answer A + Answer B
  ↓
模型生成 Long-CoT 推理 → 判断 A 更好还是 B 更好
  ↓
Reward = Format Reward + Result Reward + Consistency Reward
  ↓
用 StableReinforce 更新模型
```

### StableReinforce 的三项创新

直接套用 Reinforce++ 或 PPO 训练 RM 会遇到训练崩溃。StableReinforce 针对性地解决了三类不稳定：

| # | 创新 | 解决的问题 | 机制 |
|---|------|-----------|------|
| **1** | **Pre-CLIP** | 梯度爆炸 / 数值溢出 | 在指数函数**之前**对 log-probability ratio 做 clipping，防止 $\exp(\text{ratio})$ 变成 NaN/Inf |
| **2** | **Advantage Filter（3-sigma 规则）** | 低方差 batch 导致梯度剧烈波动 | 计算 batch 内 advantage 的均值和标准差，只保留 $[-3\sigma, 3\sigma]$ 范围内的 advantage，过滤极端值 |
| **3** | **Consistency Reward** | 推理过程和最终答案不一致（"说一套做一套"） | 用外部 MLLM referee（Qwen2.5-VL-7B-Instruct）检查推理链条是否真的支持最终判断，不一致则扣分 |

**第三项的一致性奖励可以关注一下，作为PRM的重要考量，推理过程必须与结果保持一致。**
#### Pre-CLIP 详解

标准 PPO 的 clipping 发生在 ratio **已经计算之后**：

$$L^{\text{PPO}} = \min(\text{ratio} \cdot A, \; \text{clip}(\text{ratio}, 1-\epsilon, 1+\epsilon) \cdot A)$$

其中 $\text{ratio} = \pi_\theta / \pi_{\theta_{\text{old}}}$。问题是：ratio 本身是先算 $\log \pi_\theta - \log \pi_{\theta_{\text{old}}}$ 再取指数。当两者差异极大时（长 CoT 推理中很常见），$\exp(\cdot)$ 直接溢出。

**Pre-CLIP 的做法**：在取指数**之前** clip 差值：

$$\text{Pre-CLIP ratio} = \exp(\text{clip}(\log \pi_\theta - \log \pi_{\theta_{\text{old}}}, -C, C))$$

这样指数函数的输入永远在安全范围内，数值稳定性得到保证。

#### Advantage Filter 详解

RL 训练中，advantage $A(x, y)$ 衡量"这个回答比平均水平好/差多少"。当 batch 内样本的 reward 方差很小时（比如全是简单样本或全是极难样本），微小的 reward 差异会被归一化放大成极端的 advantage，导致梯度剧烈震荡。

**解决方案**：计算 batch 内 advantage 的均值 $\mu_A$ 和标准差 $\sigma_A$，只保留 $A_i \in [\mu_A - 3\sigma_A, \mu_A + 3\sigma_A]$ 的样本，超出范围的直接丢弃（该步不参与梯度更新）。

#### Consistency Reward 详解

这是 R1-Reward 最巧妙的设计。RL 训练中模型可能学会"作弊"：推理过程写一堆废话，最后猜对答案依然能拿到 Result Reward。

**Consistency Reward 的做法**：用 Qwen2.5-VL-7B-Instruct 作为 referee，给定 `<question, answer A, answer B, model's reasoning, model's final judgment>`，让 referee 判断：**推理过程是否真的支持最终结论？**

- 推理支持结论 → Consistency Reward = 1
- 推理不支持结论 → Consistency Reward = 0

最终的 reward 公式：

$$R = \text{Format} + \text{Result} \times (1 + 0.5 \times \text{Consistency})$$

- Format：输出是否遵循 `<think>...</think><answer>...</answer>` 格式
- Result：最终判断是否正确（0 或 1）
- Consistency：推理-答案一致性（0 或 1）

**设计意图**：当模型判断正确时（Result=1），额外奖励一致性（Consistency），鼓励"真懂"而不是"蒙对"。当模型判断错误时（Result=0），一致性项自然归零。

---

## 渐进式训练策略

### 阶段一：Cold-start SFT

直接用 RL 从零训练会导致 reward 稀疏、探索效率极低。R1-Reward 先用 GPT-4o 生成推理轨迹做 SFT 热身：

1. 收集 200K 偏好样本（多模态 QA + 两个候选回答）
2. 用 GPT-4o 对每个样本生成推理判断（哪个回答更好 + 为什么）
3. **记录 GPT-4o 需要几次尝试才能生成正确判断** → 作为难度标签
4. 用 200K `(question, answer A, answer B, reasoning, judgment)` 数据做 SFT → 得到 **R1-Reward-200K**

### 阶段二：RL on Hard Samples

SFT 后的模型在"简单"样本上已经表现不错，但在 GPT-4o 也需要多次尝试的"困难"样本上仍然不行。RL 阶段**只在困难样本上训练**（GPT-4o ≥2 次尝试或失败的样本）：

- 这些样本的特点是：两个回答差异很细微（都需要仔细推理才能区分）
- 在这些样本上做 RL → 模型学会处理真正需要"思考"的边界 case
- 效率更高（不浪费 RL 算力在已经会的简单样本上）


---

## 实验设计与结果

![R1-Reward在VLbench结果](../images/r1_on_vlbench.png)


多种采样投票显示R1-Reward具有接近100%正确的潜力，但是多数vote还是没有达到最高的准确率情况。
---

## 核心发现

- R1-Reward 的核心贡献不是提出一个新的多模态理解 backbone，而是将多模态 Reward Model 从传统的 scalar scorer 转换为带显式 CoT 的 generative judge：模型先比较候选回答，再输出偏好判断。通过 GPT-4o 生成的 rationale 做 cold-start SFT，模型获得初始的“逐步比较”能力；随后用 StableReinforce 在困难样本上继续强化，使 CoT 不只是格式模仿，而能服务于最终偏好判断。
- 这篇工作的关键技术点在于解决长 CoT RL 训练的不稳定问题。长推理链会放大新旧策略之间的 log-probability 差异，导致 ratio 过大、loss 爆炸甚至 NaN。StableReinforce 通过 Pre-CLIP 在取指数前裁剪 log-ratio，并结合 Advantage Filter 与 Consistency Reward，使带 CoT 的 RM 可以相对稳定地进行 RL 优化。换句话说，它的主要价值是让“会思考的 reward model”变得可训练，而不是单纯证明 CoT prompt 有用。
- 不过，这篇工作对多模态能力本身的增强仍然有限。它主要训练的是“比较与裁判能力”，没有显式建模图文细粒度 grounding，也没有针对文档 VQA、OCR-heavy VQA、区域定位、视觉证据引用等任务做特化设计。因此，虽然它在 VL Reward-Bench 和 Multimodal Reward Bench 上表现突出，但这些 benchmark 更偏通用 reward/judge 能力，不能充分说明模型已经具备强细粒度视觉理解或文档级多模态推理能力。
- 从结果上看，R1-Reward 在 reasoning、hallucination 相关维度提升明显，而 general 维度仍相对较弱。这说明**CoT + RL 更有效地改善了模型的判断逻辑、幻觉识别和答案一致性管理，但并没有从根本上补足视觉感知、细粒度定位或复杂图文对齐能力**。也就是说，它更像是增强了 RM 的“审判推理层”，而不是显著增强了底层视觉 grounding 层。

---

## 与 DPO / BaseReward 的关系

### 与 DPO 的对比

| 维度 | DPO | R1-Reward |
|------|-----|-----------|
| **RM 形式** | 隐式（从策略概率反推） | 显式（独立的 RM，会思考） |
| **推理能力** | ❌ 无推理，只靠概率比 | ✅ Long-CoT 推理 + 自我纠错 |
| **分布外泛化** | 差（RewardBench 已证明） | 好（三个 benchmark 均 SOTA） |
| **训练稳定性** | 好（交叉熵 loss，无需 RL） | 需要 StableReinforce 三项创新 |
| **计算开销** | 低 | 高（RL rollout + referee MLLM） |

**R1-Reward 的 motivation 正是 DPO 的局限**：DPO 跳过了显式 RM，在文本上还行，在多模态上不行——因为多模态偏好判断需要显式推理（因为复杂的定位和推理没有办法只通过结果奖励训练得到？）。R1-Reward 说："我们要回到显式 RM，但用 RL 而不是交叉熵来训练它。"

### 与 BaseReward 的关系

> R1-Reward 是"从 0 到 1"——首次提出用 RL 训练多模态 RM，核心贡献是 StableReinforce 算法。
>
> BaseReward 是"从 1 到 N"——同一批作者在 R1-Reward 基础上系统探索了 MRM 的所有设计维度：Naive-RM / Critic-RM / Generative RM 三种范式比较、head 架构选择、训练策略、数据组合、集成方法等。

---

## 对你的意义

> **1. 理解 RL 训练 RM 的核心难点**：R1-Reward 把 RM 训练中的不稳定性讲得很透彻——为什么梯度会炸（长 CoT 的 log-ratio 差异极大）、为什么低方差 batch 会扰乱训练、为什么推理-答案一致性需要显式约束。这些 insight 对理解任何 RL-based 训练方法都有帮助。
>
> **2. "从隐式到显式" 的趋势判断**：R1-Reward 是"回归显式 RM"路线的旗帜性工作。虽然 DPO 路线（隐式 RM）在纯文本上很成功，但在多模态场景下，让 RM 显式地思考和判断似乎是更优解。这个趋势判断对科研方向选择有意义。
>
> **3. Consistency Reward 的设计思路可复用**：用外部 referee 检查推理-答案一致性是一种通用技巧——不限于 RM 训练，任何需要"模型不仅要答对、还要想对"的场景都可以借鉴。
>
> **4. Test-time scaling 的实用价值**：多数投票（Vote@K）是简单高效的推理增强手段。Vote@5 就能从 71% 提到 85.3%，成本可控、效果显著——在实际部署中直接可用。
>
> **5. Aha Moment 的理论趣味**：自我纠错在 RL 训练中自发涌现（而非 SFT 教会），说明 RL 的探索机制 + Consistency Reward 的正向强化可以催生出 SFT 无法产生的行为。这对理解"RL 到底在学什么"有启发性。
>
> **6. 与 BaseReward 构成完整叙事**：R1-Reward（why）+ BaseReward（what/which）是理解多模态 RM 前沿的必读组合。

---

## 待深入的问题

- StableReinforce 的 Pre-CLIP 和标准 PPO clip 在数学上的精确关系？
- Consistency Reward 中 referee MLLM 的判断准确率本身有多高？referee 犯错时对训练的影响？
- 困难样本筛选（GPT-4o 尝试次数）的可靠性——GPT-4o 的"难度"是否与真实难度分布一致？
- RM 的 Long-CoT 推理能否进一步压缩（15% 的减少后还有多少空间）？
- 该方法能否迁移到纯文本 RM？文本场景下 DPO 已经足够好，RL-trained RM 的优势在哪里？
