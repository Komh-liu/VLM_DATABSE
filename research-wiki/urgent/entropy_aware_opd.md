---
type: paper
short: "EntropyOPD"
node_id: paper:entropy_aware_opd_2026
title: "Entropy-Aware On-Policy Distillation of Language Models"
authors: []
year: 2026
venue: "arXiv"
external_ids:
  arxiv: ""
  doi: ""
  s2: null
tags: ["on-policy-distillation", "knowledge-distillation", "entropy", "LLM", "post-training"]
added: 2026-07-27T00:00:00Z
pdf: ""
---

# Entropy-Aware On-Policy Distillation of Language Models

## One-line thesis

> TODO

## Problem / Gap

> 论文发现在高熵时，Reverse KL会损伤生成的多样性并且产生不稳定的学习信号。RKL是一种mode-seeking模式。会在教师分布有高熵的情况下，降低学生输出的多样性并且产生不稳定的学习信号。尤其在推理时，高熵token往往代表重要的决定token。

在做reverse KL训练时，训练后的学生在高熵token分布上下降明显。这证明了mode seeking无法有效的保存教师的不确定性。
## Method

> 熵感知的OPD，用Forward KL增强模型在教师的熵分布较高时的Reverse KL目标。这在不确定的步骤上捕捉了所有可能的output但是保留了其他地方的准确模仿能力。平衡了mode seeking准确率和mode covering的鲁棒性，同时没有牺牲在线训练的效率。

<img src="../images/EOPD公式.png" alt="公式" style="width: 50%;" />

###  Contribution
1. **分析了多样性退化和训练的不稳定性**。标准的OPD流程导致了训练的多样性崩塌，高熵token大幅度下降。同时在教师不确定的情况下，RKL提供了一种不稳定的梯度信号。
2. **熵感知的OPD（EOPD）**。通过使用一种熵感知的策略动态调整训练目标。通过在低熵token应用RKL和高熵token应用FKL，EOPD高效地将教师的不确定性转化为一个可计算的问题
2. **在推理的benchmark上的改进**。在qwen30.6B，1.7B和4B上开展实验。

## Experiments

TODO

## Key Findings

> RKL能够在自信的预测上高效学习，而FKL则对熵高的情况使用mode covering保留了全局结构。
RKL能够使学生模型再教师自信的情况下高效稳定的学习。但是FKL的mode covering特质迁移了教师的不确定性和整体的结构。

## Limitations

TODO
