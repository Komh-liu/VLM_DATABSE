---
title: "Visual Grounding: A Comprehensive Survey"
authors: Anonymous
venue: TPAMI 2025
tags: [survey, visual-grounding, REC, RES, referring, 354-refs]
---

## 怎么用这篇综述

**不要从头读。354 篇参考文献，从头读会迷失。**

### 当词典用
- 碰到不懂的概念 → 翻目录找到对应章节
- REC vs RES vs RIS 的区别 → Section 2
- Grounding 数据集概览 → Section 3
- 评价指标 (cIoU, F1, AP...) → Section 4 或 Appendix

### 当脉络图用
- Section 5: 方法分类 (Detection-based / MLLM-based / ...)
- 看看分类树，理解 "你读的论文在整个领域的什么位置"

### 当文献索引用
- 对某个方向感兴趣 → 看该节引了哪些论文 → 顺藤摸瓜
- 比如 "MLLM-based grounding" 节 → 找到 Shikra, Ferret 等的原始引文

## 不需要看的
- 检测架构细节章节（如果你不打算改 DETR）
- 实验对比大表（太细节，你的焦点不在这里）
- 3D/视频 grounding 章节（已排除）

## 你的研究焦点在综述中的位置

```
Visual Grounding
├── Task Types
│   ├── REC (Referring Expression Comprehension) ← 你的核心
│   └── RES (Referring Expression Segmentation) ← 次要
├── Methods
│   ├── Detection-based ← 跳过 (MDETR/GLIP/Grounding DINO)
│   └── MLLM-based ← 你的焦点
│       ├── Coordinate-as-text ← Shikra
│       ├── Discrete-bin ← Ferret
│       └── Internalized ← iVGR (太新，综述没覆盖)
└── Trends ← 读这章了解领域方向
```
