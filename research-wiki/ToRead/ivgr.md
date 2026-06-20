---
title: "iVGR: Internalizing Visually Grounded Reasoning for MLLMs with Reinforcement Learning"
authors: Changbin Zhang, Yujie Zhong, Qiang Zhang, Kai Han
venue: ICML 2026 Poster
tags: [grounding-internalization, dual-stream, consistency-reward, GRPO]
---

## 核心问题

**Grounding 能力能不能被 "压缩" 进推理链，推理时不显式输出任何坐标？**

## 核心方法: 双流训练 + 一致性奖励

### 训练时（双流）
```
Stream A (带 grounding):      Stream B (不带 grounding):
  看图                         看图
  → "猫在 [100,200,300,400]"   → "猫在图片的左侧"
  → "所以答案是猫"              → "所以答案是猫"
        │                              │
        └──── 一致性奖励 ──────────────┘
        "两条流的推理结论是否一致？"
```

### 奖励信号
- **答案一致性**: Stream A 和 Stream B 的最终答案是否相同？
- **推理链一致性**: 中间步骤是否逻辑一致（不要求文字相同）？
- 如果 A（带 ground）和 B（不带）推理结果一致 → B 学到了 A 的 grounding 能力（内化）

### 推理时（只用 Stream B）
- **不输出任何坐标**
- 但内部推理链的 attention/latent state 已经吸收了 grounding 信息
- 更轻量、更隐私友好（不需要暴露位置信息）

## 核心发现: 颠覆性

**强制显式 bbox 输出反而损害推理性能**。当模型被迫输出 bbox：
- 推理链长度增加 → 注意力稀释
- "语言 + 坐标" 的多任务冲突
- 对纯语义问题（不需要定位）造成负担

内化 = 模型自己学会 "什么时候需要在内部做 grounding，什么时候不需要"。

## 为什么是金矿方向

- **仅此一篇论文**，方向几乎空白
- 恰好是你轨道 A 表示演进线的终点
- 大量开放问题未回答：
  - 内化 vs 显式的边界条件？
  - 怎么度量 "内化程度"？
  - 内化是否对所有任务类型都有效？

## 阅读重点
- 双流架构和一致性奖励 (3.1-3.2)
- "显式输出损害推理" 的实验证据 (4.2)
- 哪些任务内化最有效、哪些需要显式 ground (4.3)
