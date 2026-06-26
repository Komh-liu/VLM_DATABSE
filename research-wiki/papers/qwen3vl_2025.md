---
type: paper
short: "Qwen3-VL"
node_id: paper:qwen3vl_2025
title: "Qwen3-VL Technical Report"
authors: ["Shuai Bai", "Yuxuan Cai", "Ruizhe Chen", "Keqin Chen", "Xionghui Chen", "Zesen Cheng", "Lianghao Deng", "Wei Ding", "Chang Gao", "Chunjiang Ge", "Wenbin Ge", "Zhifang Guo", "Qidong Huang", "Jie Huang", "Fei Huang", "Binyuan Hui", "Shutong Jiang", "Zhaohai Li", "Mingsheng Li", "Mei Li", "Kaixin Li", "Zicheng Lin", "Junyang Lin", "Xuejing Liu", "Jiawei Liu", "Chenglong Liu", "Yang Liu", "Dayiheng Liu", "Shixuan Liu", "Dunjie Lu", "Ruilin Luo", "Chenxu Lv", "Rui Men", "Lingchen Meng", "Xuancheng Ren", "Xingzhang Ren", "Sibo Song", "Yuchong Sun", "Jun Tang", "Jianhong Tu", "Jianqiang Wan", "Peng Wang", "Pengfei Wang", "Qiuyue Wang", "Yuxuan Wang", "Tianbao Xie", "Yiheng Xu", "Haiyang Xu", "Jin Xu", "Zhibo Yang", "Mingkun Yang", "Jianxin Yang", "An Yang", "Bowen Yu", "Fei Zhang", "Hang Zhang", "Xi Zhang", "Bo Zheng", "Humen Zhong", "Jingren Zhou", "Fan Zhou", "Jing Zhou", "Yuanzhi Zhu", "Ke Zhu"]
year: 2025
venue: "arXiv"
external_ids:
  arxiv: "2511.21631"
  doi: null
  s2: null
tags: ["VLM", "MoE", "long-context", "interleaved-MRoPE", "DeepStack", "Qwen"]
added: 2026-06-26T00:00:00Z
---

# Qwen3-VL Technical Report

## One-line thesis

> Qwen3-VL 是 Qwen 视觉语言系列的第四代，推出 Dense（2B/4B/8B/32B）和 MoE（30B-A3B/235B-A22B）两种架构共 6 种规模，原生支持 256K token 长上下文，引入 interleaved-MRoPE 和 DeepStack 增强视觉-语言对齐，纯文本和多模态能力均超越同规模的前代模型。

## 架构改进（相对于 Qwen2.5-VL）

### 1. 双架构路线：Dense + MoE

Qwen3-VL 首次提供两种架构选择：

| 系列 | 模型规模 | 激活参数 | 特点 |
|------|---------|---------|------|
| **Dense** | 2B / 4B / 8B / 32B | 全部 | 高效紧凑，适合部署 |
| **MoE** | 30B-A3B / 235B-A22B | 3B / 22B | 超大容量，稀疏激活 |

**235B-A22B MoE 模型**的总参数量达 235B，但每次推理仅激活 22B 参数，在控制推理成本的同时提供接近密集 235B 模型的容量。

### 2. Interleaved-MRoPE

在 Qwen2.5-VL 的 M-RoPE（时间+高度+宽度）基础上，升级为 **interleaved-MRoPE**，增强跨模态的位置交互：

- 三维位置编码不再独立作用，而是在注意力计算中交叉交互
- 对于多图输入（如对比两张图片），能更好地建模图与图之间的空间关系
- 对视频帧间的时间一致性建模更强

### 3. DeepStack 集成

DeepStack 是一组多层 ViT 特征增强技术：

- 从 ViT 的多个层级提取特征（而非仅最后一层）
- 浅层特征保留细粒度空间信息（边缘、纹理）
- 深层特征蕴含语义信息（物体类别、场景）
- 通过可学习的融合机制整合多级特征

在需要细粒度定位和检测的任务上特别有效。

### 4. 256K 原生长上下文

Qwen3-VL 原生支持 256K token 的上下文窗口，适用于：

- 长视频理解（数十分钟到数小时）
- 多图对比推理（同时输入几十张图片）
- 图文交错的长文档理解
- 基于大量帧进行事件检测和推理

### 5. 文本时间对齐（Text-based Time Alignment）

Qwen2.5-VL 使用绝对时间编码来处理视频时间信息。Qwen3-VL 进一步演进，引入显式**文本化时间戳**：

- 视频帧的时间位置不再仅由位置编码表达
- 在文本 token 中显式嵌入时间戳描述（如 "Frame at 00:01:35"）
- 模型可以基于时间文本进行推理，更自然地理解时间顺序

## 模型家族对比

| 维度 | Qwen2.5-VL | Qwen3-VL |
|------|-----------|---------|
| 架构 | 仅 Dense | Dense + MoE |
| 规模数 | 3 种 | 6 种 |
| 最大参数量 | 72B | 235B-A22B |
| 上下文长度 | 32K | **256K** |
| 位置编码 | M-RoPE | **Interleaved-MRoPE** |
| 视觉特征 | 单层 ViT 输出 | **DeepStack（多层融合）** |
| 视频时间建模 | 绝对时间编码 | 文本时间戳对齐 |
| 纯文本能力 | 依赖 Qwen2.5 LLM | 超越同尺寸文本 backbone |

## 核心能力提升

### 1. 纯文本理解

Qwen3-VL 的一个独特之处在于：它的纯文本能力在多项 benchmark 上**超过了同尺寸的纯文本 backbone**（一般的 VLM 因为视觉训练导致文本能力下降）。这得益于更好的多模态训练策略。

### 2. 长上下文多模态理解

原生 256K token 上下文窗口带来的能力跃升：

- 支持同时输入数十张图像进行综合推理
- 长视频（>1 小时）的关键事件检索和总结
- 图文交错的大型文档（几百页 PDF）的全局理解

### 3. 多模态推理

- 单图像推理：在 MMMU、MathVista 等复杂推理 benchmark 上持续领先
- 多图像推理：支持对比、排序、差异查找等跨图像推理任务
- 视频推理：长视频事件定位和时间顺序推理达到新水平

### 4. 视觉 Agent 能力延续

继承并增强了 Qwen2.5-VL 的 agent 能力，MoE 大模型在复杂多步操作任务上表现更好。

## 局限

- MoE 模型（235B-A22B）推理部署需要专门的 MoE 推理框架支持
- DeepStack 增加了视觉编码的计算量
- 长上下文（256K）的实际利用率仍在探索中
- 部分 agent 任务上成功率仍有提升空间
