---
title: "VLM-R³: Vision-Language Models for Reinforcement Learning-based Reasoning and Referring"
authors: Anonymous
venue: NeurIPS 2025
tags: [grounding, RL, GRPO, adaptive-grounding, REC]
---

## 核心问题

**模型能不能自己决定什么时候做 grounding？** 不是 "始终 ground" (Shikra)，也不是 "在推理步骤中 ground" (Argus)，而是让模型通过 RL 自己学。

## 核心方法: Grounding 作为 RL Action

每一步推理，模型有三个选择：
- **Action 1: 不 ground，继续推理** → 直接输出下一个推理 token
- **Action 2: ground 当前提到的对象** → 输出 `[x1,y1,x2,y2]` 后再继续推理
- **Action 3: 终止推理，输出答案** → `<EOS>`

### 奖励设计
```
R = α × 最终答案正确性 + β × grounding 准确性 (cIoU)
```
如果模型 ground 错了一个位置，即使答案对也会被惩罚（但惩罚权重 β 可调）。

### 训练: GRPO
- 对同一个 question，采样 N 条推理路径
- 每条路径有不同的 ground/不-ground 选择序列
- 组内归一化 advantage → 更新 policy

## 核心发现

1. **自适应策略学到了有意义的 pattern**: 对需要精确空间定位的问题（"红色球在蓝色球的左边还是右边"）模型更倾向于 ground；对纯语义问题（"这是什么动物"）更倾向于不 ground
2. **始终 ground 不是最优的**: 强制 ground 增加推理负担，反而损害表现
3. **RL 自然学到 trade-off**: 不需要手动制定规则

## 为什么是论文路线图的核心

这篇论文是 **Grounding 和 RL 的交叉点**——它证明了 "何时 ground" 是一个 RL 问题。MoCA (ICML 2026) 进一步问 "ground 对了该给多少信用"，iVGR (ICML 2026) 进一步问 "能不能不 ground 也行"。

## 阅读重点

- Action space 定义 (3.1)
- Reward function 设计 (3.2)
- 自适应策略学到的 pattern 分析 (4.3 实验部分)
