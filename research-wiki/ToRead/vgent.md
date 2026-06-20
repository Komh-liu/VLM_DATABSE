---
title: "VGent: Modular Decoupled Reasoning and Grounding for Visual Tasks"
authors: Anonymous
venue: CVPR 2026
tags: [grounding, modular, decoupled, reasoning]
---

## 核心问题

如果把推理和 grounding **完全拆成两个模块**，互不耦合，会怎样？

## 核心方法

```
VGent 架构:
  推理模块 (Reasoning Module)    预测模块 (Prediction Module)
        │                              │
  只负责 "思考"                    只负责 "定位"
  输出文本推理链                   输出 bbox/mask
        │                              │
        └────────── 不共享参数 ──────────┘
                      ↓
                 最终输出
```

- 两个模块完全独立，各自优化各自的目标
- 推理模块: language modeling loss
- 预测模块: grounding loss (L1 + GIoU)
- 多目标 F1 +20%

## 为什么值得读

它代表 grounding 演进方向光谱的另一个极端：
- Argus: grounding 和推理**深度耦合**（每步推理都 ground）
- VLM-R³: **自适应**耦合（RL 选择）
- **VGent: 完全解耦**
- iVGR: **不 ground**（内化）

理解这个光谱 = 理解你的研究空间在哪里。

## 阅读重点
- 两个模块怎么交互（接口设计）
- 解耦的代价是什么（信息损失？）
- 实验结果中两个模块各自的贡献
