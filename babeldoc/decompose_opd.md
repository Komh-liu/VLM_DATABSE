---
type: paper
short: "DecomposedOPD"
node_id: paper:decomposed_opd_icml2026
title: "Decomposed On-Policy Distillation for Vision-Language Reasoning: Steering Gradients for Visual Grounding"
authors: ["Hee Suk Yoon", "Eunseop Yoon", "Jaehyun Jang", "SooHwan Eom", "Ji Woo Hong", "Mark Hasegawa-Johnson", "Qi Dai", "Chong Luo", "Chang D. Yoo"]
year: 2026
venue: "ICML 2026 (Spotlight)"
external_ids:
  arxiv: "2606.00564"
  doi: ""
  s2: null
tags: ["on-policy-distillation", "visual-grounding", "VLM", "vision-language", "knowledge-distillation", "gradient-decomposition"]
added: 2026-07-14T00:00:00Z
pdf: "decomposed_opd_icml2026.pdf"
---

# Decomposed On-Policy Distillation for Vision-Language Reasoning: Steering Gradients for Visual Grounding

## One-line thesis

> 

## Problem / Gap
对于使用OPD做VLM的蒸馏，作者认为可以把损失函数拆分为两个不同的组件，**语言先验**和**视觉Grounding**作者的分析解释了这两个梯度的分量几乎是正交的。说明学生和教师的语言分布进行对齐和视觉感知能力提升几乎是独立的。因此，标准的优化轨迹其实是一条次优的轨迹，只能在两个目标之间取得平衡。作者引入了视觉梯度引导的方法，以动态调整更新向量的方向，优先考虑视觉的子空间。


### 问题设定



## Method


### 



### 📊 动态指标 (Dynamic Metrics)



#### 1. 



#### 2. 



#### 3. 



## Key Results



## Assumptions



## Limitations / Failure Modes



## Reusable Ingredients



## Open Questions



## Claims



## Connections



## Relevance to This Project



## Reading Notes

### 核心概念



### 机制理解



### Recipe / 实践细节



### 值得复现或借鉴的实验



### 和 MOPD / 多教师蒸馏的关系


