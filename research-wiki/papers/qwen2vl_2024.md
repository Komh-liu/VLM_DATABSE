---
type: paper
short: "Qwen2-VL"
node_id: paper:qwen2vl_2024
title: "Qwen2-VL: Enhancing Vision-Language Model's Perception of the World at Any Resolution"
authors: ["Peng Wang", "Shuai Bai", "Sinan Tan", "Shijie Wang", "Zhihao Fan", "Jinze Bai", "Keqin Chen", "Xuejing Liu", "Jialin Wang", "Wenbin Ge", "Yang Fan", "Kai Dang", "Mengfei Du", "Xuancheng Ren", "Rui Men", "Dayiheng Liu", "Chang Zhou", "Jingren Zhou", "Junyang Lin"]
year: 2024
venue: "arXiv"
external_ids:
  arxiv: "2409.12191"
  doi: null
  s2: null
tags: ["VLM", "dynamic-resolution", "M-RoPE", "vision-language", "video", "Qwen"]
added: 2026-06-26T00:00:00Z
---

# Qwen2-VL: Enhancing Vision-Language Model's Perception of the World at Any Resolution

## One-line thesis

> Qwen2-VL 提出 Naive Dynamic Resolution 机制让模型处理任意分辨率的图像、M-RoPE 位置编码实现图文+视频的统一位置建模，推出 2B/8B/72B 三种规模，72B 版本在多数多模态 benchmark 上达到 GPT-4o 和 Claude 3.5 Sonnet 水平，是 Qwen-VL 系列的一次全面架构升级。

## 架构改进（相对于 Qwen-VL）

### 1. Naive Dynamic Resolution（朴素动态分辨率）

**Qwen-VL 的限制**：所有图像被强制缩放到 448×448，导致细粒度信息丢失（如小字、图表细节）。

**Qwen2-VL 的改进**：动态分辨率处理机制，让模型以**原生比例**处理图像。

工作原理：

- 图像不再缩放到固定尺寸，而是根据其长宽比动态切分为若干 448×448 的 patch
- 每个 patch 独立通过 ViT 编码
- 编码后的 token 序列拼接后送入 LLM
- 例如：一张 1344×1008 的图片会被切成 3×2=6 个 patch，生成 6×256=1536 个视觉 token

**效果**：在需要细粒度视觉理解的 benchmark 上提升显著，特别是在 OCR、图表理解、文档解析等场景中。

### 2. M-RoPE（Multimodal Rotary Position Embedding）

**Qwen-VL 的限制**：标准 RoPE 只能处理一维位置编码（文本 token），无法表达图像的二维空间关系和视频的时间关系。

**Qwen2-VL 的改进**：将 RoPE 分解为三个独立的部分：

- **时间维度**：用于区分视频中不同帧的位置
- **高度维度**：用于表示图像中行的位置
- **宽度维度**：用于表示图像中列的位置

这使得同一个模型能够统一处理静态图像（时间维度固定）和视频（时间维度变化）。

### 3. 统一图像和视频处理

Qwen2-VL 采用**统一的图像和视频处理范式**，不再需要单独的视频编码分支：

- 视频被视为连续帧序列
- 每帧通过动态分辨率机制处理
- M-RoPE 为不同帧分配时间位置编码
- 支持长达数分钟的视频理解

### 4. 三档模型规模

| 模型 | 参数量 | 适用场景 |
|------|--------|----------|
| **Qwen2-VL-2B** | 2B | 边缘设备、移动端部署 |
| **Qwen2-VL-8B** | 8B | 通用场景，高效推理 |
| **Qwen2-VL-72B** | 72B | 高性能场景，接近 GPT-4o 水平 |

## 核心能力提升

### 1. 细粒度视觉理解

动态分辨率带来的最大提升在细粒度视觉任务上：

| Benchmark | Qwen2-VL-72B | Qwen-VL-Max | GPT-4o | Claude 3.5 Sonnet |
|-----------|-------------|-------------|--------|-------------------|
| **ChartQA** | 88.2 | 82.5 | 85.7 | 84.8 |
| **DocVQA** | 95.5 | 92.3 | 92.8 | 91.2 |
| **TextVQA** | 84.9 | 81.2 | — | — |
| **MathVista** | 68.2 | 62.4 | 63.8 | 65.2 |

在 ChartQA、DocVQA 等需要高分辨率理解的 benchmark 上，Qwen2-VL-72B 超越了 GPT-4o 和 Claude 3.5 Sonnet，这直接归功于动态分辨率机制。

### 2. 视频理解

视频理解是 Qwen2-VL 的新增能力（Qwen-VL 不支持）：

| Benchmark | Qwen2-VL-72B | GPT-4o | Gemini Pro 1.5 |
|-----------|-------------|--------|----------------|
| **Video-MME** (w/o subs) | 71.2 | 71.9 | 73.7 |
| **MVBench** | 70.8 | — | 71.6 |
| **PerceptionTest** | 65.3 | 63.2 | 68.2 |

72B 模型在视频理解上达到了与 GPT-4o 相当的水平。

### 3. 多模态推理

| Benchmark | Qwen2-VL-72B | Qwen-VL-Max | GPT-4o |
|-----------|-------------|-------------|--------|
| **MMMU** (val) | 64.7 | 60.2 | 69.1 |
| **MathVista** (testmini) | 68.2 | 62.4 | 63.8 |
| **MMBench** (test) | 86.1 | 81.8 | 83.4 |
| **MMStar** | 68.0 | 61.8 | 62.5 |

在 MMBench 和 MMStar 这类综合理解评测上超越了 GPT-4o。

### 4. 多语言能力

- 支持中英文混合输入
- 在中文多模态评测（如 MMBench-CN）上表现优异
- 得益于 Qwen 系列对中文的优化

## 关键技术创新总结

| 创新点 | Qwen-VL | Qwen2-VL | 带来的提升 |
|--------|---------|----------|-----------|
| 图像分辨率 | 固定 448×448 | 动态（保持原生比例） | 细粒度理解 +10~15% |
| 位置编码 | 1D RoPE | M-RoPE（时间+高+宽） | 支持视频统一建模 |
| 视频处理 | 不支持 | 统一范式 | 新增视频理解能力 |
| 模型规模 | 9.6B（单一） | 2B/8B/72B 三档 | 覆盖边缘到云端 |

## 局限

- 72B 模型部署成本高，2B/8B 与大型号之间存在性能差距
- 部分复杂推理 benchmark（如 MMMU）上仍低于 GPT-4o
- 长视频理解能力有限（分钟级，非小时级）
