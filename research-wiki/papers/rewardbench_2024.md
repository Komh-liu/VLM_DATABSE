---
type: paper
node_id: paper:rewardbench_2024
title: "RewardBench: Evaluating Reward Models for Language Modeling"
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

> 留给你写

---

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

> 留给你写

### Finding 1：DPO 的 scaling benefit

- 随着 DPO 模型参数量增大（7B → 13B → 70B），其隐式 RM 的质量持续提升
- 表明 DPO 从更大的 backbone 中受益，不仅在生成能力上，也在 implicit reward modeling 上

### Finding 2：分类器 RM vs 生成式 RM 的差距

- 显式训练的**分类器 RM**（如 UltraRM、Starling）在 RewardBench 上整体优于 **DPO 隐式 RM**
- 但在分布外场景下，差距会缩小甚至反转
- 生成式 RM（LLM-as-a-Judge 范式）虽然灵活，但在标准化评测中不如分类器 RM 稳定

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

## 局限

> 留给你写

## 与 VL-RewardBench 的关系

VL-RewardBench（Li et al., CVPR 2025）是 RewardBench 的多模态扩展，将评测从纯文本扩展到视觉-语言场景，涵盖 General Multimodal Queries、Visual Hallucination Detection、Complex Multimodal Reasoning 三类任务。两者在方法论上一脉相承，都是设计有细粒度区别的 preference pair 来评测 RM 的真实能力。

## 对你的意义

> 留给你写
