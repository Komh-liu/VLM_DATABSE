---
title: "DeepSeek-R1: Incentivizing Reasoning Capability in LLMs via Reinforcement Learning"
authors: Daya Guo, Dejian Yang, Haowei Zhang, Junxiao Song, Ruoyu Zhang, Runxin Xu et al.
venue: arXiv 2025 (DeepSeek)
tags: [RLVR, reasoning, GRPO, rule-based-reward]
---

## 为什么这篇论文在你的阅读清单里

因为 ICML 2026 的多模态 RL 论文（SSL4RL、RuCL、3D-RFT）全都在用 **RLVR (RL with Verifiable Rewards)** 范式。这个范式来自 R1。

## 你需要读的部分

- **Introduction** (理解动机)
- **R1-Zero** (Section 2): 纯 RL 无 SFT → 涌现推理
- **Discussion** (Section 5): 为什么 rule-based reward 比 neural reward model 好
- 跳过: R1 完整版（SFT+RL 混合）、蒸馏、工程细节

## R1-Zero: 纯 RL 无 SFT

```
标准流程:  Pretrain → SFT → RLHF
R1-Zero:   Pretrain → 直接 GRPO (无 SFT!)
```

- 不经过任何监督微调，直接对 base model 做 GRPO
- 奖励函数: 数学题答对 = 1，答错 = 0（**完全规则化，无需人类标注**）
- 结果: 模型自行涌现了 CoT、自我验证、反思等推理行为

## RLVR 范式的核心

```
可验证奖励 (Verifiable Reward) = 
  答案可以自动校验的任务
    ├── 数学题: 最终答案对错
    ├── 代码: 通过测试用例
    ├── Grounding: IoU, cIoU, F1, Recall (全自动可算!)
    └── 推理步骤: 规则检查（格式、一致性...）
```

**关键洞察**: 规则化的奖励函数不会 reward hack——因为规则不会像 neural reward model 那样被过拟合。

## 对你的 grounding 研究的启示

Grounding 天然适配 RLVR：
- IoU > 0.5 = ground 对了 (reward +=1)
- cIoU 连续值可以直接当 reward
- 不需要人类标注 "这个 bbox 好不好"
- 不需要训练一个 neural reward model 来评判 grounding 质量

## 阅读重点
- R1-Zero 的训练流程（怎么做到纯 RL 不崩溃的？）
- Rule-based reward 的设计原则
- "Aha moment" 现象（推理能力涌现）
