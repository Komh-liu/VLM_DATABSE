---
type: paper
node_id: paper:vl_rewardbench_cvpr2025
title: "VL-RewardBench"
short: "VL-RewardBench"
authors: ["Lei Li", "Yuancheng Wei", "Zhihui Xie", "Xuqing Yang", "Chenxin An", "Tianyu Liu", "Yifan Song", "Peiyi Wang", "Sujian Li", "Bill Yuchen Lin", "Lingpeng Kong", "Qi Liu"]
year: 2025
venue: "CVPR 2025"
external_ids:
  arxiv: "2411.17451"
  doi: null
  s2: null
tags: ["reward-model", "benchmark", "VL-GenRM", "LLM-as-a-Judge", "multimodal", "evaluation", " hallucination", "RLHF"]
added: 2026-06-28T00:00:00Z
---

# VL-RewardBench: A Challenging Benchmark for Vision-Language Generative Reward Models

## One-line thesis

> VL-RewardBench 是 RewardBench 的多模态扩展，专门评测视觉-语言生成式奖励模型（VL-GenRM，即 LVLM-as-a-Judge 范式）。涵盖 General Multimodal / Hallucination Detection / Complex Reasoning 三类任务共 1,250 个高质量样本，评估 16 个模型，揭示视觉感知是核心瓶颈、推理时 scaling 效果因模型而异、critic training 可大幅提升判断能力（+14.7%）三个关键发现。

## 核心研究问题

> 当前 VL-GenRM（用 LVLM 作为评判器）的评估方法存在两个问题：AI 标注偏好引入系统性偏差，传统学术基准任务过于简单无法区分快速迭代的 LVLM。本文提出 VL-RewardBround truth 标签的基准。

---
### 当前研究现状
在VL RewardBench被提出的时候，最好用的还是llm as judge。然后论文发现如果针对性进行critic训练效果会更好。
1. 模型在视觉任务失败的比推理任务多
2. 模型“推理时拓展”的能力提升受scaling影响
3. 训练VL-genRM评分能够提升模型作为rewardmodel的能力。
4. RM是数据飞轮的重要步骤，能够将模型生成的数据进行打分之后调整模型的输出，使得模型不断能够改进自己的输出。
#### 先前的VL-GenRMs的工作
1. 依赖AI生成的偏好，例如使用GPT4V做注释进行评估
2. 调整传统的学术benchmark和预先定义好的表现，主要关注于VL对齐任务如图片捕捉。


## 实验设计
1. 真实场景的多模态查询
2. 视觉幻觉检测任务（细粒度或者Grounding）
3. 多模态知识和数学推理（兼顾推理）
### 数据集构建

VL-RewardBench 从 7 个数据源中精选 1,250 个偏好对，覆盖三类任务：

| 类别 | 数据源 | 样本数 | 标注方式 |
|------|--------|--------|---------|
| **General Multimodal** | VLFeedback, WildVision | 183 | 已有偏好标签 + 小模型集成过滤 |
| **Hallucination Detection** | POVID, RLAIF-V, RLHF-V | 749 | 已有偏好标签 + 小模型集成过滤 |
| **Complex Reasoning** | MMMU-Pro, MathVerse | 318 | 无偏好标签 → AI 生成候选回答 + 人工验证 |

**集成过滤策略**：用 LLaVA-1.5-7B、LLaVA-1.6-7B、LLaVA-OneVision-7B-si、Qwen

2-VL-7B 四个小模型组成集成，识别所有小模型一致判错的样本作为"挑战集"。三位研究生标注 + 两位审核，每样本约 65 秒，标注者间 Cohen's κ 平均 0.70。

**错误类别标注**：Existence（存在性，59.3%）、Recognition（识别，20.6%）、Attribute（属性，7.7%）、Counting（计数，6.7%）、Other（5.7%）。

### 评测协议

- 每个偏好对评估 K=5 次，随机化回答顺序消除位置偏差
- 多数投票决定最终偏好
- 固定解码参数（temperature=0.2, top-p=0.2）
- 整体准确率 + 宏平均准确率（缓解任务类别不均衡）

---

## 核心发现

### Finding 1：视觉感知是核心瓶颈

- 存在性/识别类任务的错误率 > 67%，而推理类任务平均错误率仅 41.8%
- 即使是 GPT-4o-mini，在 Existence 任务上错误率仍高达 67.9%；Qwen2-VL-7B 在 Recognition 任务上错误率 80.9%
- **结论**：当前 VL-GenRM 的高层推理能力远强于基础视觉感知，首要改进方向是提升视觉感知

### Finding 2：推理时 scaling 效果因模型而异

- GPT-4o：K=1→7 时宏平均准确率从 60.3% 提升到 62.7%，传统 scaling 有效
- GPT-4o-mini：K 增加无明显变化
- 开源模型 Qwen2-VL-72B、Molmo-72B：K 增加反而性能下降（-1.7 ~ -2.6 pp）
- **结论**：文本领域的推理时 scaling 不能直接迁移到 VL-GenRM，需要专门设计

### Finding 3：Critic Training 大幅提升判断能力

- LLaVA-OneVision-7B-ov 经过 critic training 后：
  - Pointwise critic：+14.7%（52.9%）
  - Pairwise critic：+9.2%（47.4%）
- Pointwise 在幻觉子集上更好（+9.1% over pairwise），Pairwise 在推理任务上更好（60.0%）
- **结论**：训练 VL-GenRM "学会判断"是提升评测能力的可靠路径

### Downstream Correlation

- VL-RewardBench 上的表现与 MMMU-Pro 的 Best-of-N 采样提升呈强相关（Pearson r > 0.9 for both Qwen2-VL-7B and LLaVA-OneVision-7B-ov）
- 验证了 VL-RewardBench 能预测 RM 在下游对齐任务中的实际效用

---

## 主实验结果

| 模型 | General | Hallucination | Reasoning | Macro Avg |
|------|---------|---------------|-----------|-----------|
| GPT-4o | 49.1 | 67.6 | 70.5 | **62.4** |
| Gemini-1.5-Pro | 50.8 | 72.5 | 64.2 | **62.5** |
| Claude-3.5-Sonnet | 43.4 | 55.0 | 62.3 | 53.6 |
| GPT-4o-mini | 41.7 | 34.5 | 58.2 | 44.8 |
| Llama-3.2-90B | 42.6 | 57.3 | 61.7 | 53.9 |
| Qwen2-VL-72B | 32.8 | 38.4 | 58.0 | 43.0 |
| Qwen2-VL-7B | 19.1 | 22.4 | 51.1 | 33.9 |

- 即使是 GPT-4o 也仅 62.4%，开源 72B+ 模型大多在随机水平附近挣扎
- 消融实验：同样任务分布的随机采样对，Gemini 系列可达 95%+，证明集成过滤策略有效识别了真正有挑战性的样本

---


## 与 RewardBench 的关系

> VL-RewardBench 是 RewardBench（Lambert et al., NeurIPS 2024）的多模态扩展：两者都采用 prompt-chosen-rejected 三元组格式，通过设计细粒度、可验证的偏好差异来评测 RM。VL-RewardBench 将评测从纯文本扩展到视觉-语言场景，重点关注视觉感知瓶颈（这是纯文本 RM 评测中不存在的维度）。

## 对你的意义

> VL-GenRM 的不靠谱 + critic training 的有效性 → 说明当前直接用 LVLM 做 judge 风险很大，需要用专门训练的评判模型替代。这与 BaseReward 论文的结论一致：需要专门的 multimodal reward model，不能直接拿通用 LVLM 当评判器用。
