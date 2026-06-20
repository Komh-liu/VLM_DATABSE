---
title: "GRIT: Grounded Reasoning via Interleaved BBox-Text Generation with Minimal Supervision"
authors: Anonymous
venue: NeurIPS 2025
tags: [grounding, few-shot, reasoning, bbox-text-interleaving, RL]
---

## 核心问题

**最少需要多少 grounding 数据？** 能不能只用几十个样本？

## 核心方法: BBox+文字交错推理 + 极少样本

```
输入: "红色球在蓝色球的左边还是右边？"
输出: 红色球 [100,200,250,350]
      蓝色球 [300,100,450,250]
      红色球的 x1=100 < 蓝色球的 x1=300
      所以红色球在左边 ✓
```
- BBox 和文字在推理链中交错出现
- 仅需 **20 个** 标注样本！

## 为什么 20 个样本就够

1. **BBox 输出的格式是固定的**——不需要学习复杂的 grounding 表示，只需要学 "什么时候插入 bbox"
2. **预训练 VLM 已经有 grounding 能力**（QwenVL 本身就支持 REC），只需要唤醒
3. **少量样本只是 trigger**：告诉模型 "你可以在推理中输出 bbox"

## 核心洞察

**Grounding 能力已经在预训练中了，不需要大量 grounding 标注来灌输**。少量样本只是教会模型 "你可以这么做"。

## 在你的研究中的意义

- 证明了 **7B 预算下做 grounding 研究是完全可行的**（不需要百万级标注）
- "用 RL 唤醒已有能力" 的思路 → 与 MoCA/iVGR 呼应
- 20 样本的限制在哪？（泛化性、新领域、细粒度）

## 阅读重点

- 数据构造方法（20 个样本怎么选的）
- 交错生成格式
- 与 full-data grounding 方法的对比实验
