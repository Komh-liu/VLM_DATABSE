---
type: paper
short: "Qwen-VL"
node_id: paper:qwenvl
title: "Qwen-VL: A Versatile Vision-Language Model for Understanding, Localization, Text Reading, and Beyond"
authors: ["Jinze Bai", "Shuai Bai", "Shusheng Yang", "Shijie Wang", "Sinan Tan", "Peng Wang", "Junyang Lin", "Chang Zhou", "Jingren Zhou"]
year: 2023
venue: "arXiv"
external_ids:
  arxiv: "2308.12966"
  doi: null
  s2: null
tags: ["VLM", "vision-language", "multimodal", "visual grounding", "text reading", "Qwen"]
added: 2026-06-02T06:01:26Z
---

# Qwen-VL: A Versatile Vision-Language Model for Understanding, Localization, Text Reading, and Beyond

## One-line thesis

> Qwen-VL 以 Qwen-7B LLM 为基础，搭配 OpenCLIP ViT-bigG 视觉编码器和单层交叉注意力投影器，通过三阶段训练（预训练→多任务→指令微调），在一个统一框架中同时实现图像描述、视觉问答、细粒度视觉定位（grounding）和图像文字阅读（text reading），在同规模通用模型中达到了 SOTA。

## 架构设计

### 整体架构

Qwen-VL 的整体架构遵循"视觉编码器 → 模态连接器 → LLM"的经典范式：

- **视觉编码器**：OpenCLIP 的 ViT-bigG，分辨率为 448×448
- **模态连接器**：单层交叉注意力（cross-attention，约 0.08B 参数），将 256 个视觉 token 投影到 LLM 的输入空间
- **LLM**：Qwen-7B（7.7B 参数）

与 BLIP-2 的 Q-Former（12 层 transformer）不同，Qwen-VL 的感知器采用**单层交叉注意力**，直接连接视觉编码器和 LLM，大幅简化了桥接模块，减少了计算开销。

### 三阶段训练管线

| 阶段 | 内容 | 训练数据量 | 冻结参数 | 可训练参数 |
|------|------|-----------|----------|-----------|
| **阶段一：预训练** | 大规模图文对训练 | 1.4B 样本（从 5B 清洗得到） | LLM | ViT + 交叉注意力 |
| **阶段二：多任务训练** | 引入高分辨率、定位、text reading 数据 | ~50M 样本 | 无 | 全部参数 |
| **阶段三：指令微调** | 多轮对话交互 | ~350K 对话样本 | 无 | 全部参数 |

**阶段一**的核心目标是学习视觉编码器与 LLM 之间的对齐。使用大规模中英文混合图文对数据，在冻结 LLM 的条件下训练 ViT 和交叉注意力层。

**阶段二**采用 448×448 分辨率（阶段一的 2 倍），引入多种任务格式的数据：
- 纯图文描述数据
- 视觉问答数据
- **视觉定位数据**：将边界框坐标 token 化为 `(x1, y1, x2, y2)` 格式，在训练中让模型同时看到图像区域和对应的文本描述
- **Text reading 数据**：检测和识别图像中的文字

**阶段三**构建了约 350K 多轮对话数据，将第二阶段模型转化为能够多轮交互的 Qwen-VL-Chat。

## 核心能力提升

### 1. 视觉理解（Comprehension）

- 在 **ImageNet-1K** 零样本分类任务上达到 75.5% 的 top-1 准确率，显著超过其他同规模 VLM（如 LLaVA-1.5 为 70.7%）
- **VQAv2** test-dev 集上达到 78.8%，超过 LLaVA-1.5-13B（78.5%），以 7B 基座超越 13B 级模型

### 2. 通用视觉问答（VQA）

| Benchmark | Qwen-VL | Qwen-VL-Chat | LLaVA-1.5-13B | InstructBLIP-13B |
|-----------|---------|-------------|---------------|-----------------|
| **VQAv2** (test-dev) | 78.8 | 76.3 | 78.5 | — |
| **OKVQA** (val) | 58.6 | 56.6 | 56.2 | 54.0 |
| **GQA** (test) | 59.3 | 58.4 | 61.0 | 60.4 |
| **TextVQA** (val) | 61.5 | 60.5 | 52.1 | 54.5 |

在 **TextVQA** 上 Qwen-VL 的表现特别突出（61.5 vs LLaVA-1.5 的 52.1），得益于阶段二引入了大量 text reading 数据。

### 3. 视觉定位（Grounding）

Qwen-VL 的定位能力是最具区分度的创新点之一：

- **RefCOCO** testA：75.9%（Qwen-VL-Chat 指令微调后：86.3%）
- **RefCOCO** testB：67.6%（指令微调后：77.4%）
- 以一般模型的参数规模达到了专用定位模型的效果

实现方式是通过**统一 token 化**：边界框坐标作为文本 token 在训练中出现，让模型通过语言建模学习区域与语言之间的对应关系。这对于后 续像 GRIT（在推理链中交替生成文本和 bbox）等工作的出现提供了重要基础。

### 4. 文本阅读（Text Reading）

Qwen-VL 在 OCR 和文档理解任务上表现优异：
- **TextVQA** 61.5%，在同规模模型中领先
- **DocVQA** 文档级文字识别能力显著
- 支持横排和竖排文字、手写和印刷体

### 5. 对话交互

Qwen-VL-Chat 在真实对话 benchmark 上的表现：
- 在开放域多模态对话中生成了更自然、更符合语境的回复
- 支持多轮图文对话、基于图片的推理对话
- 能够理解用户问题指向的图像区域并进行精准回答

## 局限

- 视频理解能力未覆盖（后续 Qwen2-VL 补齐）
- 部分复杂推理场景仍有困难
- 高分辨率密集文本场景可能受限
