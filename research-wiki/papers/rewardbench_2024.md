---
type: paper
node_id: paper:rewardbench_2024
title: "RewardBench"
short: "RewardBench"
authors: ["Nathan Lambert", "Valentina Pyatkin", "Jacob Morrison", "LJ Miranda", "Bill Yuchen Lin", "Khyathi Chandu", "Nouha Dziri", "Sachin Kumar", "Tom Zick", "Yejin Choi", "Noah A. Smith", "Hannaneh Hajishirzi"]
year: 2024
venue: "NeurIPS 2024 Track on Datasets and Benchmarks"
external_ids:
  arxiv: "2403.13787"
  doi: null
  s2: null
tags: ["reward-model", "benchmark", "RLHF", "evaluation", "preference-optimization"]
added: 2026-06-26T00:00:00Z
---

# RewardBench: Evaluating Reward Models for Language Modeling

## One-line thesis

> RewardBench 是首个专门评估 reward model（RM）的综合 benchmark，覆盖 Chat / Reasoning / Safety 三类任务共 20K+ prompt-chosen-rejected 三元组，评测了 80+ 个 RM（包括分类器式的显式 RM 和 DPO 训练中的隐式 RM），系统揭示了 DPO 模型的 scaling benefit、分类器 RM 与生成式 RM 间的持久差距、以及三类不同的 refusal 行为模式。

## 核心研究问题

> 设计了一个评估多种不同架构的奖励模型的框架。同时评估DPO虽然简单但是无法泛化到流行的偏好数据测试集。对于之前的数据集，存在两种问题。一种是旧的数据集人类自己也无法达成一致，所以模型的测试天花板存在上限，无法很好的区分模型是否能够很好的解决问题。（对于reward model的benchmark，我们的目的是RM能够完全近似人类偏好，但是人类自己也无法判断更偏好哪个的数据集不能用来测RM）后续的新数据集有训练集但是没有测试集，意味着模型训练完之后不知道是否能够达成一个很好的proxy效果。

---
### 两种不同的RM策略

## 实验设计
### 数据集构成

RewardBench 的核心是一个包含 prompt-chosen-rejected 三元组的评测集，覆盖四大类别：

| 类别 | 来源数据集 | 评测维度 |
|------|-----------|---------|
| **Chat** | AlpacaEval、MT Bench 变体 | 对话质量、指令遵循 |
| **Reasoning** | LLMBar、PRM Math、HumanEvalPack | 数学推理、代码、逻辑 |
| **Safety** | XSTest、Do Not Answer | 拒绝回答、安全边界 |
| **Prior Sets** | Anthropic HH、SHP、Summarize | 经典偏好数据的分布外测试 |

每个类别的样本都经过精心筛选，确保 chosen 和 rejected 之间有**细粒度、可验证的差异**（如存在 bug、事实错误、逻辑矛盾等），而不是模糊的主观偏好。

### 评测指标

核心指标：**分类准确率**——RM 是否正确识别了 chosen 回答优于 rejected 回答。

此外还分析了：
- Calibration（RM 的分数是否校准）
- Refusal 行为分类
- 不同数据分布下的泛化能力

## 核心发现

> 特定训练的classifier（拆掉unenbedding层换成投影层），与DPO训练的模型反推偏好概率做对比（DPO后推导出回答的概率-DPO前推导出回答的概率）。发现RM会在更广泛的场景表现比DPO场景好（泛化性？）。基于这个，作者比较了scaling law，测试推理能力，突出三种refusal行为。

- 通过一个框架测试不同架构的奖励模型。并且发布了评估时用的不同数据。
- RewardBench 对专职RM和DPO模型的评估方式有着本质不同。对于专职RM（传统分类器），评估过程非常直接：将“提示词+回答”输入模型，模型直接输出一个标量分数，RewardBench通过比较两个回答的分数高低来计算准确率。而对于DPO模型，由于它本身并不输出分数，RewardBench需要额外加载训练时所用的参考模型，通过公式 $$r(x, y) = \beta \log(\pi / \pi_{ref}) $$ 反推出一个“隐式奖励分数”，然后再用这个反推的分数去比大小、算准确率。虽然两者最终都能得出一个准确率用于排行榜排名，但分数的来源和质量截然不同。
- DPO 隐式奖励极度依赖参考模型：论文实验了换一个"错误的" reference model，DPO RM 的性能直接崩到随机水平。
- 从多个维度展示RM的全景。很少有RM在REWARDBENCH数据集上的分数呈高斯分布，更少的RM以0奖励为中心，且测试的没有一个是中心高斯分布。未来的工作应确定为下游RL训练首选的RM输出分布
- 展示现有偏好数据集的局限性。 

### Finding 1：DPO 的 scaling benefit

- 随着 DPO 模型参数量增大（7B → 13B → 70B），其隐式 RM 的质量持续提升
- 表明 DPO 从更大的 backbone 中受益，不仅在生成能力上，也在 implicit reward modeling 上

### Finding 2：三类 RM 的分数对比

RewardBench 评测了三类模型：Sequence Classifier（显式分类器 RM）、DPO 隐式 RM、Generative RM（LLM-as-a-Judge，通过 prompt 生成判断）。

- 显式训练的**分类器 RM**（如 ArmoRM、Starling）在 RewardBench 总分上整体最优
- **DPO 隐式 RM**在大多数子集上表现尚可，但在 **Prior Sets**（前人偏好数据的 test set，属分布外评测）上分数很低——论文结论是 DPO "fail to generalize" 到这些经典偏好测试集
- **LLM-as-a-Judge**（如 Meta-Llama-3-70B-Instruct、prometheus-8x7b-v2.0）得分低于分类器 RM（Tab. 8 原文：*"the best classifier RMs outperform the best generative reward models"*）
- DPO 隐式奖励极度依赖参考模型：换一个 reference model 性能直接崩到随机水平

### Finding 3：三类 refusal 行为

RM 在面对不安全/有害输入时表现出三种不同的拒绝行为：
1. **过度拒绝**：安全但合理的回答也被拒绝
2. **适当拒绝**：仅对不安全内容拒绝
3. **拒绝缺失**：不安全内容通过了 RM 的打分

不同 RM 在这三种行为上差异显著，揭示了 reward model 的安全对齐特性。

### Finding 4：现有偏好数据 test set 的局限

发现已有的偏好评测集（如 Anthropic HH test set）存在：
- 数据泄露（与训练集分布过于接近）
- 区分度不足（大多数 RM 都能达到高准确率）
- 缺乏细粒度、可验证的挑战性样本

RewardBench 通过构建"hard subset"来解决这些问题。


## 与 VL-RewardBench 的关系

VL-RewardBench（Li et al., CVPR 2025）是 RewardBench 的多模态扩展，将评测从纯文本扩展到视觉-语言场景，涵盖 General Multimodal Queries、Visual Hallucination Detection、Complex Multimodal Reasoning 三类任务。两者在方法论上一脉相承，都是设计有细粒度区别的 preference pair 来评测 RM 的真实能力。

## 对你的意义

> 了解RM是怎么工作的，对于表现不好的RM有哪些场景的不足。
