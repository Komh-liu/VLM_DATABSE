---
type: paper
node_id: paper:r1reward_iclr2026
title: "R1-Reward: Training Multimodal Reward Model Through Stable Reinforcement Learning"
authors: ["YiFan Zhang", "Xingyu Lu", "Xiao Hu", "Chaoyou Fu", "Bin Wen", "Tianke Zhang", "Changyi Liu", "Kaiyu Jiang", "Kaibing Chen", "Kaibing Tang", "Haojie Ding", "Jiankang Chen", "Fan Yang", "Zhang Zhang", "Tingting Gao", "Di Zhang", "Guorui Zhou", "Liang Wang"]
year: 2026
venue: "ICLR 2026 Poster"
external_ids:
  arxiv: "2505.02835"
  doi: null
  s2: null
tags: ["RL", "reward-model", "multimodal", "StableReinforce", "MLLM", "preference-optimization"]
added: 2026-06-26T00:00:00Z
---

# R1-Reward: Training Multimodal Reward Model Through Stable Reinforcement Learning

## One-line thesis

将 reward modeling 重新定义为 rule-based RL 任务——模型给定问题+两个答案，用 `<think>` 格式推理哪个更好，以答案是否匹配 ground truth 作为 reward。针对 RL 训练不稳定提出 StableReinforce 算法（Pre-Clip + 3-sigma Advantage Filter + Consistency Reward 三项修复），在 VL Reward-Bench、Multimodal Reward Bench 上超越此前 SOTA。与 BaseReward 为同一团队先后工作，BaseReward 在此基础上进一步改进。

---

## 核心研究问题

> 留给你写

---

## 实验设计

### 训练流程

> 留给你写

**基础模型**：Qwen2.5-VL-7B

**训练策略**：
- **Cold-start SFT**：xx
- **Progressive difficulty**：xx

**Reward 设计**（三项）：
> 留给你写

**评测基准**：
> 留给你写

---

## 核心发现

### Finding 1：RL 训练 reward model 面临的稳定性问题

> 留给你写

---

### Finding 2：StableReinforce 的有效性

> 留给你写

---

### Finding 3：Test-time scaling

> 留给你写

---

## StableReinforce 算法（核心方法）

> 留给你写

### Pre-Clip

### 3-sigma Advantage Filter

### Consistency Reward

---

## 计算成本

> 留给你写

---

## 核心结果

| 对比 | 结果 |
|:---|:---|
| VL Reward-Bench | +8.4% |
| Multimodal Reward Bench | +14.3% |
| MM-RLHF Reward Bench | +3.5% |
| Test-time scaling (Voting@15) | ~100% (Any Correct) |

---

## 局限

> 留给你写

---

## 论文引用的相关工作与本文的区别

### 直接竞争关系

- **[BaseReward](basereward_iclr2026.md)**（ICLR 2026，同一团队）— 两个工作为同一批核心作者（YiFan Zhang, Chaoyou Fu 等）的先后工作。R1-Reward 首次提出用 RL 训练多模态 reward model 的思路；BaseReward 在此基础上系统化探索构建配方（三种范式对比、reward head 设计、数据组合、集成策略等），最终性能超越 R1-Reward（MM-RLHF-Reward Bench ~11% vs R1-Reward +3.5%，VL-Reward Bench ~18% vs R1-Reward +8.4%）。建议先读 R1-Reward 理解核心思路，再读 BaseReward 看完整演化。

### 方法对比关系

- **PPO / GRPO / Reinforce++** — 论文指出这些标准 RL 算法在 reward modeling 场景下存在训练崩溃问题，StableReinforce 的 Pre-Clip 和 Advantage Filter 专门针对这些问题设计
- **DPO** — reward model 的传统训练方式是 DPO（偏好对分类），而本工作改用 RL 训练，属于方法论上的范式转换

> 留给你写更多细节

---

## 关键洞察

> 留给你写

## 对你的意义

> 留给你写

## 后续阅读建议

> 留给你写
