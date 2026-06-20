---
title: "Shikra: Unleashing Multimodal LLM's Referential Dialogue Magic"
authors: Keqin Chen, Zhao Zhang, Weili Zeng, Richong Zhang, Feng Zhu, Rui Zhao
venue: arXiv 2023
tags: [grounding, MLLM, coordinate-as-text, REC]
---

## 核心问题

怎么让 LLM 直接输出坐标？**最简单的方法是什么？**

## 核心方法

把 `[x1, y1, x2, y2]` 当普通文本 token 输出。不引入任何特殊结构——没有新 token type，没有新 head，没有离散化。

```
Q: "Where is the cat?"
A: "The cat is at [100, 200, 300, 400]."
          → 数字就是 LLM 的文本 token
```

### 为什么能 work
- LLM 本身就能处理数字（训练数据里全是数字）
- 坐标 = 数字序列 = LLM 原生能力
- 只需要把坐标数字 tokenize，不需要架构改动

### 训练数据
- ReferDialogue 数据集：包含 "refer + dialogue" 的联合数据
- 坐标格式直接嵌入自然语言

## 关键洞察

**Grounding 不需要特殊架构**。这是 "坐标当文本" 路线的开山之作，也是 MLLM grounding 的 baseline。

## 局限（为什么后来被 Ferret 超越）

- 连续坐标 → 离散 token → 精度损失（只能输出整数坐标）
- 对细粒度描述（"猫的左耳朵尖"）精度不够
- 序列长度随坐标数量线性增长

## 在你知识体系中的位置

Grounding 表示演进: **Shikra (坐标当文本) → Ferret (离散 bin) → LISA (SEG token) → iVGR (内化)**

这是第一步——最简单的 grounding 方案。
