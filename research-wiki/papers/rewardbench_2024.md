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

> RewardBench 是首个专门用于评估 reward model（RM）的综合 benchmark，覆盖 Chat、Reasoning、Safety 三类任务，包含 prompt-chosen-rejected 三元组。评测了 80+ 个 RM（包括分类器式 RM 和 DPO 隐式 RM），发现 DPO 模型存在 scaling benefit、分类器 RM 与生成式 RM 之间存在持久差距、以及三类不同的 refusal 行为模式。

## 核心研究问题

> 留给你写

---

## 实验设计

### 数据集构成

> 留给你写

**四大类别**：
- **Chat**：AlpacaEval、MT Bench 变体
- **Reasoning**：LLMBar、PRM Math、HumanEvalPack
- **Safety**：XSTest、Do Not Answer
- **Prior Sets**：Anthropic HH、SHP、Summarize

### 评测指标

> 留给你写

---

## 核心发现

> 留给你写

### Finding 1：DPO 的 scaling benefit

### Finding 2：分类器 RM vs 生成式 RM 的差距

### Finding 3：三类 refusal 行为

### Finding 4：现有偏好数据 test set 的局限

---

## 局限

> 留给你写

---

## 论文引用的相关工作与本文的区别

> 留给你写

## 对你的意义

> 留给你写
