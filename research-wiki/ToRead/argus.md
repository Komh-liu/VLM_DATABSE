---
title: "Argus: Object-Centric Grounded Chain-of-Thought for Vision-Language Reasoning"
authors: Anonymous
venue: CVPR 2025
tags: [grounding, chain-of-thought, object-centric, REC]
---

## 核心问题

如果在 CoT 的**每一步**都显式 ground 推理中提到的物体，会发生什么？

## 核心方法: Object-Centric Grounded CoT

```
标准 CoT:
  "图中有三个球。红色球在左边。所以答案是红色球。"

Grounded CoT (Argus):
  "图中有三个球 [b:0,0,300,400]。红色球在 [b:100,200,250,350]。
   蓝色球在 [b:300,100,450,250]。红色球在左边。
   所以答案是红色球。"
```

### 关键设计
- **以物体为中心**: 推理链以物体为单位组织，每个物体 → 描述 + 坐标
- **BBox 嵌入在文本中**: 类似 Shikra，但只在推理中间标注
- **始终 ground**: 只要提到一个物体，就输出它的坐标

## 与 VLM-R³ 的对比（这是核心张力）

| | Argus | VLM-R³ |
|---|-------|--------|
| Grounding 时机 | 始终 ground（每步） | 自适应 ground（RL 选择） |
| 方法 | SFT + 固定模板 | GRPO RL |
| 优点 | 精确、可审计 | 灵活、不冗余 |
| 缺点 | 推理链长、显式输出负担重 | 可能 ground 不够 |

## 为什么需要理解 Argus

它代表 "grounding 应该全程参与推理" 的极端立场。VLM-R³ 是中间立场。iVGR 是另一个极端（完全不 ground）。理解这三者的光谱 → 就是你研究的空间。

## 阅读重点

- Grounded CoT template 的设计 (3.1)
- Object-centric 怎么实现 (3.2)
- 实验结果中 grounding accuracy vs reasoning accuracy 的 trade-off
