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

Qwen-VL 是一系列大规模视觉语言模型（LVLM），以 Qwen-7B LLM 为基础，配备 ViT-bigG 视觉编码器和单层交叉注意力投影器，通过三阶段训练管线，在图像描述、视觉问答、视觉定位和文本阅读等任务上达到了同规模模型的 SOTA 水平。

## Problem / Gap

已有的视觉语言模型通常在单一任务上优化，缺乏在图像理解、定位和文本阅读等多维度能力的统一。如何在单一模型中同时实现：常规图文理解、细粒度视觉定位（grounding）、以及图像中的文字识别（text reading），是该领域的核心挑战。

## Method

1. **视觉编码器**：使用 OpenClip 的 ViT-bigG@448 作为视觉受体，将图像压缩为 256 个 token
2. **输入-输出接口**：单层交叉注意力（cross-attention）将视觉 token 投影到 LLM 输入空间，图像分辨率支持 448×448
3. **三阶段训练管线**：
   - 阶段一：大规模图文对预训练（1.4B 样本），冻结 LLM，仅训练视觉编码器和交叉注意力层
   - 阶段二：多任务训练（~50M 样本），引入更高分辨率和定位数据，训练视觉编码器+交叉注意力
   - 阶段三：指令微调，生成 Qwen-VL-Chat 变体，支持多轮对话交互
4. **定位能力**：通过将边界框坐标 token 化实现 grounding 和 text reading 的统一训练
5. **多语言语料**：构建了多语言多模态清洗语料库

## Key Results

- 在图像描述、VQA、视觉定位等多项 benchmark 上刷新同规模模型的 SOTA
- Qwen-VL-Chat 在真实对话 benchmark 上优于已有的视觉对话模型
- 支持 zero-shot 和 few-shot 设置，泛化能力突出
- 在 text reading 任务上（如 OCR、文档理解）表现优异

## Assumptions

- ViT-bigG 特征空间可被 Qwen-7B 有效利用
- 单层交叉注意力足以桥接视觉和语言模态
- 三阶段训练策略优于端到端联合训练

## Limitations / Failure Modes

- 视频理解能力未在论文中涉及
- 高分辨率场景（如超大图像、密集文本）可能受限
- 模型偏见和幻觉问题未充分评估

## Reusable Ingredients

- 三阶段训练管线（预训练→多任务→指令微调）可泛化到其他 VLM
- ViT-bigG + 单层交叉注意力的简单架构设计思路
- 定位和 text reading 的统一 token 化方案

## Open Questions

- Qwen-VL 系列后续版本（Qwen2-VL）的性能提升情况
- 对于技术报告而言，重要的不是技术报告中能够学习到什么，而是知道这个工作选用这个技术成功的原因。如果benchmark或者选用技术不了解，应该去阅读引用的原文而不是钻研技术报告

## Claims

_TODO._

## Connections

_Edges are recorded in `graph/edges.jsonl`; summarize here for human readers._

## Relevance to This Project

- Qwen-VL 是 VLM 领域的重要工作，展示了如何以相对简单的架构（单层交叉注意力 vs Q-Former）实现多功能 VLM。其定位和 text reading 的统一训练方案对后续研究有重要参考价值。


## Abstract (original)

> In this work, we introduce the Qwen-VL series, a set of large-scale vision-language models (LVLMs) designed to perceive and understand both texts and images. Starting from the Qwen-LM as a foundation, we endow it with visual capacity by the meticulously designed (i) visual receptor, (ii) input-output interface, (iii) 3-stage training pipeline, and (iv) multilingual multimodal cleaned corpus. Beyond the conventional image description and question-answering, we implement the grounding and text-reading ability of Qwen-VLs by aligning image-caption-box tuples. The resulting models, including Qwen-VL and Qwen-VL-Chat, set new records for generalist models under similar model scales on a broad range of visual-centric benchmarks (e.g., image captioning, question answering, visual grounding) and different settings (e.g., zero-shot, few-shot). Moreover, on real-world dialog benchmarks, our instruction-tuned Qwen-VL-Chat also demonstrates superiority compared to existing vision-language chatbots. Code, demo and models are available at https://github.com/QwenLM/Qwen-VL.
