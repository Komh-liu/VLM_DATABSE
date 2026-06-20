---
title: "DeepSeekMath: Pushing the Limits of Mathematical Reasoning in Open Language Models"
authors: Zhihong Shao, Peiyi Wang, Qihao Zhu, Runxin Xu, Junxiao Song, Mingchuan Zhang, Y.K. Li, Y. Wu, Daya Guo
venue: arXiv 2024 (DeepSeek)
tags: [GRPO, RL, mathematical-reasoning, policy-optimization]
---

## 为什么这篇论文在你的阅读清单里

不是因为它做数学推理——是因为 **GRPO 算法在这篇论文里被首次提出**。你现在要用的所有工具（VLM-R³、MoCA、iVGR、SSL4RL）全都在用 GRPO。

## 你需要读的部分

**只读 Section 3 (GRPO)**，约 4 页。其余全部跳过。

## GRPO 核心（你需要的理解深度）

### 和 PPO 的区别
- PPO: 需要 Critic/Value Network 估计 advantage → 需要额外的一整个模型
- **GRPO: 不需要 Value Network**。对同一个 prompt 采样 N 个回答，组内归一化 → advantage

### 公式（直觉版）
```
输入: 同一个 question，生成 N 个不同的 output
计算: 每个 output 的 reward r_i (rule-based: 答案对=1, 错=0)
组内归一化: advantage_i = (r_i - mean(r_1..r_N)) / std(r_1..r_N)
策略更新: max Σ log π_θ(output_i | question) × advantage_i
KL 约束: - β × D_KL(π_θ || π_ref)  # 防止偏离太远
```

### 为什么 GRPO 特别适合 VLM
1. VLM 的生成质量很难用 value function 建模 → 不需要 critic 是优势
2. 组内比较天然鲁棒：不需要绝对奖励值准确，只需要相对排名对
3. 奖励函数可以基于规则（IoU, F1, 答案匹配）→ 不会 reward hack

### 关键超参数
- N (group size): 采样数，越大越稳定但越贵
- β (KL penalty): 防止 policy 偏离 reference model
- 学习率: 通常比 SFT 低一个数量级

## 阅读重点
- Section 3.1: GRPO from PPO 的推导
- Section 3.2: 为什么不用 value network（组内归一化的论证）
- 附录的伪代码
