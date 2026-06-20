---
title: "Bad Seeing or Bad Thinking? Rewarding Perception for Vision-Language Reasoning (MoCA)"
authors: Haozhe Wang, Qixin Xu, Changpeng Wang et al.
venue: ICML 2026 Spotlight
tags: [credit-assignment, perception, reasoning, RL, GRPO, grounding]
---

## 核心问题

VLM 答错了——是**看错了**还是**想错了**？传统 RL 只给 "答对=1, 答错=0" 的终局奖励，完全不区分错误来源。

## 核心方法: Modality-Aware Credit Assignment (MoCA)

### Blindfolded Reasoning（蒙眼推理）代理

```
正常推理:  看图 → 描述 → 推理 → 答案
蒙眼推理:  不看图 → 只有文本 → 推理 → 答案
```

- 如果蒙眼也能答对 → 视觉感知在这个问题上没有贡献 → 不应该给视觉通道奖励
- 如果蒙眼答错、看图答对 → 视觉感知是关键的 → 应该给视觉通道奖励
- 如果看图也答错 → 推理有问题 → 应该惩罚推理通道

### 奖励分解

```
R_total = R_perception + R_reasoning

R_perception: 感知步骤的奖励
  = 看图答对 - 蒙眼答对 (如果 >0，说明视觉有帮助)

R_reasoning: 推理步骤的奖励
  = 蒙眼答对 (纯推理能力)
```

## 核心发现

1. **感知错误经常被误判为推理错误**: 很多表面上的 "推理失败" 其实是模型根本没看到关键信息
2. **独立的感知奖励显著提升细粒度视觉理解**: 需要精确感知的任务（空间关系、计数、属性识别）收益最大
3. **信用分配让模型学会了 "主动看"**: 训练后模型在推理前会更仔细看相关区域

## 为什么是金矿方向

- 这是**第一篇**系统性做多模态信用分配的论文
- 只区分了 "看 vs 想"，太粗糙 → 你可以层次化到 "定位/属性/关系/计数"
- Grounding 是天然的训练信号来源：ground 对了 = 看清了
- 与你的 grounding 背景直接交叉

## 可能的扩展方向（你的研究）
1. **层次化信用分配**: 定位错误 vs 属性识别错误 vs 空间关系错误 vs 计数错误
2. **Grounding-guided credit**: 用 grounding 标注自动构建 credit label（而非 MoCA 的 blindfold 方法）
3. **何时 credit 分配最有价值**: 连接 VLM-R³ 的自适应 grounding

## 阅读重点
- Blindfolded reasoning proxy 怎么实现 (3.1)
- R_perception 和 R_reasoning 的公式 (3.2)
- 实验：哪些任务类型从感知奖励中受益最大 (4.2-4.3)
