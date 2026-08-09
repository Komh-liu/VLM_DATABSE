---
type: paper
short: "DeFacto"
node_id: paper:defacto_2025
title: "DeFacto: Counterfactual Thinking with Images for Enforcing Evidence-Grounded and Faithful Reasoning"
authors: ["Tianrun Xu", "Haoda Jing", "Ye Li", "Yuquan Wei", "Jun Feng", "Guanyu Chen", "Haichuan Gao", "Tianren Zhang", "Jing Liu", "Feng Chen"]
year: 2025
venue: "ICML 2026"
external_ids:
  arxiv: "2509.20912"
  doi: ""
  s2: null
tags: ["counterfactual-reasoning", "visual-grounding", "evidence-faithfulness", "GRPO", "VLM", "thinking-with-images"]
added: 2026-08-09T00:00:00Z
pdf: "DeFacto_2509.20912.pdf"
---

# DeFacto: Counterfactual Thinking with Images for Enforcing Evidence-Grounded and Faithful Reasoning

## One-line thesis

>

## Problem / Gap

> 现有的方法在确保证据-答案一致性时存在问题


## Method

>  将训练拆解成3种不同范式： 1.正样本 2.事实遮挡 3.随机mask

1. 正样本的case中，



## Contribution

> 构建了一个语言先验的pipeline辅助定位问题相关区域和依托构建出对的数据集DeFacto-100K。以及在数据集基础上设计的3种互补损失函数，以提升答案准确率，结构化推理能力和证据一致性。此外作者还引入了一个人工标注的benchmark以系统评估评估答案准确性之外，答案grounded一致性。通过pipeline训练的模型同时提升了答案的准确率和证据的一致性。


## Experiments

### 实验设置

>使用了Deepeyes和GRIT以量化证明这些问题（定位错误/偶然正确/忠实但错误）

### 主要结果

>

### Ablation 与分析

>

## Key Findings

>

## Limitations

>

## Connections

>之前的工作包括“thinking with image”将直接的视觉步骤加入多模态推理中以解释推理的可解释性和视觉grounding。也有工作通过SFT让模型在cot中生成区域

## Relevance to This Project

>
