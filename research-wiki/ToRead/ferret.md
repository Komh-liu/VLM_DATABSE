---
title: "Ferret: Refer and Ground Anything Anywhere at Any Granularity"
authors: Haoxuan You, Haotian Zhang, Zhe Gan, Xianzhi Du, Bowen Zhang, Zirui Wang, Liangliang Cao, Shih-Fu Chang, Yinfei Yang
venue: ICLR 2024 (Apple)
tags: [grounding, MLLM, hybrid-representation, REC, RES, referring]
---

## 核心问题

Shikra 直接输出坐标数字，但**连续空间和离散 token 之间的 gap 怎么弥合？** 任意形状（点、框、自由曲线）怎么统一表示？

## 核心方法: 混合区域表征 (Hybrid Region Representation)

一个区域 = 三部分信息：
1. **离散坐标 token**: 空间 bin grid 上 `[bin_x, bin_y, bin_w, bin_h]`
2. **连续视觉特征**: 该区域的 ROI-pooled feature（保留细粒度视觉信息）
3. **文本上下文**: 自然语言描述

### 离散化策略
- 图像被划分为空间 grid
- 连续坐标 (x, y) → 离散 bin index
- Bin 分辨率是超参数：越细 = 越精确但序列越长

### 关键创新: 统一表示任意形状
- 点 → `[bin_x, bin_y, <EMPTY>, <EMPTY>]`
- 框 → `[bin_x1, bin_y1, bin_x2, bin_y2]`
- 自由曲线/多边形 → 序列 of bin points + continuous feature

## 核心结果
- 支持 REC (Referring Expression Comprehension) 和 RES (Referring Expression Segmentation)
- 任意粒度：从 "猫" 到 "猫左耳朵第三根毛" 理论上都行
- 表现显著优于 Shikra 等纯文本坐标方法

## 在你知识体系中的位置

Grounding 表示演进: Shikra → **Ferret** → LISA → iVGR

理解 Ferret 的关键是理解 "为什么需要离散化？"——因为 LLM 的 token 空间是离散的，而视觉世界是连续的。离散 bin 是最直接的桥接方案。
