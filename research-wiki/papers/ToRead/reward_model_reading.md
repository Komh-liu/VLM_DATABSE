# Reward Model & Multimodal Reward Model 学习路径

> **方向**: RLHF → Reward Model → Multimodal Reward Model —— 理解 R1-Reward 和 BaseReward 的前提知识
> **目标**: 快速理解 MRM 的范式、训练、评估，以及为什么用 RL 训练 RM

---

## 背景

R1-Reward 和 BaseReward 这两篇论文属于**多模态奖励模型**方向。要理解它们在做什么，需要先了解：

1. **RLHF 中的 reward model** 是什么、起什么作用
2. **DPO** 提供了怎样的替代思路（隐式 RM vs 显式 RM）
3. **多模态 RM** 和文本 RM 有什么不同
4. **用 RL 训练 RM** 为什么比传统交叉熵训练好

---

## 阅读顺序（共 6 篇）

### 第一步：RLHF 大框架

#### 1/6. InstructGPT（NeurIPS 2022）

| 项目 | 内容 |
|------|------|
| **论文** | Training Language Models to Follow Instructions with Human Feedback |
| **作者** | Ouyang et al. (OpenAI) |
| **链接** | `https://arxiv.org/abs/2203.02155` |
| **预计时间** | 1h |

**为什么第一个读：**
RLHF 的开山之作，定义了 **SFT → Reward Model → PPO** 三阶段范式。reward model 的角色就在这里确定的：给 LM 的输出打一个标量分数，指导 PPO 更新。

**要理解的核心概念：**
- Reward model 是**人类偏好的代理（proxy）**
- 它把"人觉得哪个回答好"这种隐式偏好压缩成一个标量
- reward hacking 问题的根源：proxy 不等于真实偏好

---

### 第二步：DPO——省掉显式 RM 的路线

#### 2/6. DPO（NeurIPS 2023 Oral）

| 项目 | 内容 |
|------|------|
| **论文** | Direct Preference Optimization: Your Language Model is Secretly a Reward Model |
| **作者** | Rafailov et al. (Stanford) |
| **链接** | `https://arxiv.org/abs/2305.18290` |
| **预计时间** | 2h |

**为什么第二个读：**
DPO 证明了最优策略与 reward 之间存在解析映射，所以可以跳过显式 RM 训练。这是理解 R1-Reward/BaseReward 的**前提**，因为 R1-Reward 的 motivation 就是"DPO 这种隐式 RM 不够好，我们回到显式 RM 但用 RL 训练它"。

**核心概念：**
- 隐式 RM vs 显式 RM 的 trade-off
- 为什么隐式 RM 在文本上表现好，在多模态上可能不够？

---

### 第三步：Reward Model 评估与训练

#### 3/6. RewardBench（2024）

| 项目 | 内容 |
|------|------|
| **论文** | RewardBench: Evaluating Reward Models for Language Modeling |
| **作者** | Lambert et al. |
| **链接** | `https://arxiv.org/abs/2403.13787` |
| **预计时间** | 30min |

**为什么读：**
当前最通用的 RM 评测基准，定义了评估 RM 的标准协议。VL-RewardBench（多模态版本）正是其扩展。

**核心概念：**
- RM 评估的维度：如何检测 reward hacking？
- 标量 RM 的局限

#### 4/6. Secrets of RLHF in Part I: PPO（2024）

| 项目 | 内容 |
|------|------|
| **论文** | Secrets of RLHF in Large Language Models Part I: PPO |
| **作者** | Zheng et al. |
| **链接** | `https://arxiv.org/abs/2307.04964` |
| **预计时间** | 1h |

**为什么读：**
实操层面深入讲解 RM 训练的细节（reward model overoptimization、数据质量影响、训练不稳定性等）。StableReinforce 解决的很多问题在这里有讨论。

---

### 第四步：多模态 Reward Model 评测

#### 5/6. VL-RewardBench（CVPR 2025）

| 项目 | 内容 |
|------|------|
| **论文** | VL-RewardBench: A Challenging Benchmark for Vision-Language Generative Reward Models |
| **作者** | Li et al. (HKU, SCUT, SJTU, PKU, UW, Allen AI) |
| **链接** | `https://arxiv.org/abs/2411.17451` |
| **预计时间** | 30min |

**为什么读：**
定义了多模态 RM 的评测体系：
- General Multimodal Queries（14.7%）
- Visual Hallucination Detection（59.9%）
- Complex Multimodal Reasoning（25.4%）

理解了这个 benchmark 才知道 R1-Reward 和 BaseReward 在比什么。

---

### 第五步：目标论文

#### 6/6. 先读 R1-Reward，再读 BaseReward

| 顺序 | 论文 | 会议 | 链接 |
|------|------|------|------|
| **6a** | R1-Reward | ICLR 2026 | `https://arxiv.org/abs/2505.02835` |
| **6b** | BaseReward | ICLR 2026 | `https://arxiv.org/abs/2509.16127` |

**为什么先 R1-Reward：**
R1-Reward 是首次用 RL 训练多模态 RM 的思路提出者，是"从 0 到 1"的贡献。核心创新点是 StableReinforce 算法。

**为什么后 BaseReward：**
BaseReward 是同一批作者在 R1-Reward 基础上的系统性探索——它不只是提一个新算法，而是系统地比较了 RM 的**所有设计维度**（范式选择、head 架构、训练策略、数据组合、集成方法等），是"从 1 到 N"的贡献。先读 R1-Reward 理解 why，再读 BaseReward 理解 what/which。

---

## 阅读路线图

```
InstructGPT (RLHF 范式)
    │
    ▼
DPO (隐式 RM 替代路线)
    │
    ├─── RewardBench (RM 评估标准)
    │
    ├─── Secrets of PPO (RM 训练实操)
    │
    ├─── VL-RewardBench (多模态 RM 评测)
    │
    ▼
R1-Reward (用 RL 训练显式 MRM)
    │
    ▼
BaseReward (MRM 系统化构建指南)
```

---

## 相关 PDF 下载

这些论文大部分可在 ToRead/pdf 目录下找到（DPO 已完成下载，InstructGPT 等尚未下载）。需要时可定位后补充。

## 已下载的关联 PDF

| 文件 | 内容 | 在阅读路径中的位置 |
|------|------|-------------------|
| `dpo_neurips2023.pdf` | DPO 原始论文 | 第 2 步 |
| `basereward_iclr2026.pdf` | BaseReward | 第 6b 步 |
| `r1reward_iclr2026.pdf` | R1-Reward（笔记待完善） | 第 6a 步 |
