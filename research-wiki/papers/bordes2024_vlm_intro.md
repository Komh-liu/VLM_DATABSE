---
type: paper
node_id: paper:bordes2024_vlm_intro
title: "An Introduction to Vision-Language Modeling"
authors: ["Florian Bordes", "Richard Yuanzhe Pang", "Anurag Ajay", "Alexander C. Li", "Adrien Bardes", "Suzanne Petryk", "Oscar Mañas", "Zhiqiu Lin", "Anas Mahmoud", "Banghua Zhu", "Sara Elkafrawy", "Kathy Zhu", "Yejin Choi", "Antoine Bosselut", "Saining Xie", "Yann LeCun", "Armand Joulin", "Kaiming He", "Hugo Touvron", "Piotr Dollár", "Liangliang Cao", "Hu Xu", "Zhe Gan", "Xi Yin", "Jianfeng Gao", "Lijuan Wang", "Jianwei Yang"]
year: 2024
venue: "arXiv"
external_ids:
  arxiv: "2405.17247"
  doi: null
  s2: null
tags: ["vision-language-models", "survey", "tutorial", "contrastive-learning", "generative-models", "masked-modeling", "pretrained-backbone", "evaluation", "bias", "hallucination", "video-understanding", "instruction-tuning", "rlhf", "grounding", "data-curation"]
added: 2026-05-31T12:00:00Z
---

# An Introduction to Vision-Language Modeling（视觉-语言建模导论）

## 摘要（中文翻译）

随着大语言模型（LLM）的近期流行，许多尝试将其扩展到视觉领域。从能够引导我们穿越陌生环境的视觉助手，到仅凭高层次文本描述生成图像的生成模型，视觉-语言模型（VLM）的应用将深刻影响我们与技术的关系。然而，要提高这些模型的可靠性，仍有许多挑战需要解决。语言是离散的，而视觉存在于一个更高维度的空间中，其中的概念并不总能被轻易离散化。为了更好地理解将视觉映射到语言的机制，我们撰写了这篇 VLM 导论，希望能帮助任何想要进入该领域的人。首先，我们介绍 VLM 是什么、它们如何工作以及如何训练它们。然后，我们介绍并讨论评估 VLM 的方法。虽然本文主要关注将图像映射到语言，但我们也讨论了将 VLM 扩展到视频领域。

## One-line thesis

A comprehensive tutorial and entry guide to vision-language models (VLMs), categorizing training paradigms into four types — contrastive, masked, generative, and pretrained-backbone-based — while providing practical training guidance, responsible evaluation practices, and a discussion of extending VLMs to video.

## 引言（中文翻译）

近年来，语言建模领域取得了令人瞩目的进展。Llama、ChatGPT 等大语言模型（LLM）已经能够解决种类繁多的任务，其使用也变得越来越普及。这些原本主要局限于文本输入的模型，现在被扩展到接受视觉输入。将视觉与语言连接起来，将解锁若干对当前基于 AI 的技术革命至关重要的应用。尽管已有若干工作将 LLM 扩展到视觉领域，但语言与视觉的连接问题并未完全解决。例如，大多数模型难以理解空间关系或计数，除非借助复杂的数据标注工程开销。许多 VLM 也缺乏对属性和顺序的理解，常常忽略输入提示的部分内容，导致需要大量的提示工程才能产生期望结果。部分模型还会产生幻觉，生成既不需要也不相关的内容。因此，开发可靠的模型仍然是一个非常活跃的研究领域。

在本文中，我们介绍视觉-语言模型（VLM）。我们解释 VLM 是什么、如何训练它们，以及如何根据不同研究目标有效评估 VLM。本文不应被视为综述或完整的 VLM 指南——我们并非旨在引用该领域的每一篇工作，也不试图囊括所有最佳实践。相反，我们的目标是为 VLM 研究提供一个清晰易懂的入门介绍，并重点介绍该领域的有效实践。本导论对于想要进入这一领域的学生或其他领域的研究者尤其有用。

我们首先介绍不同的 VLM 训练范式。我们讨论对比学习策略和生成组件。最后，我们介绍使用预训练骨干网络（如 LLM）的 VLM。将 VLM 归类为不同家族并非易事，因为大多数模型都有重叠的组件。但我们希望我们的分类法能够帮助新研究者导航这一领域，并揭示 VLM 背后的内部机制。

接下来，我们介绍训练 VLM 的典型配方。例如，我们涵盖：不同研究目标下哪些数据集是合适的？哪种数据整理策略？我们需要训练文本编码器，还是可以利用预训练的 LLM？对比损失足以实现视觉理解，还是生成组件是关键？我们还介绍用于提升模型性能、接地能力以及更好的对齐能力的常用技术。

虽然提供训练模型的配方是更好地理解 VLM 需求的关键步骤，但对这些模型进行稳健可靠的评估同样重要。近期引入了许多用于评估 VLM 的基准测试，然而其中一些基准测试存在研究者应该知晓的基本局限性。通过讨论 VLM 基准测试的优势和劣势，我们希望揭示改进我们对 VLM 理解所面临的挑战。我们首先讨论评估 VLM 视觉-语言能力的基准测试，然后介绍如何衡量偏差。

下一代 VLM 将能够通过将视频映射到语言来理解视频。然而，视频存在不同于图像的各种挑战。计算成本当然高得多，但还有关于如何通过文本映射时间维度的其他考虑。通过阐明当前从视频中学习的方法，我们希望指出当前需应对的研究挑战。

通过降低进入 VLM 研究的门槛，我们希望为推动 VLM 的负责任发展奠定基础，同时拓展视觉理解的边界。

## 目录结构（点击跳转）

- **[1. 引言](#sec-引言-中文翻译)** — 背景、VLM 面临的挑战、本文定位
- **[2. VLM 的家族分类](#sec-VLM-的四种训练范式)**
  - 2.1 基于 Transformer 的早期 VLM 工作（VisualBERT、ViLBERT）
  - 2.2 基于对比学习的 VLM（CLIP、SigLIP、Llip）+ 能量模型 / InfoNCE 理论
  - 2.3 基于掩码目标的 VLM（FLAVA、MaskVLM）+ 信息论视角
  - 2.4 基于生成的 VLM（CoCa、CM3Leon、Chameleon；用扩散模型做判别任务）
  - 2.5 基于预训练骨干网络的 VLM（Frozen、MiniGPT-4/5/v2、Qwen-VL、BLIP-2）
- **[3. VLM 训练指南](#sec-VLM-训练指南)**
  - 3.1 训练数据（DataComp、启发式过滤、CLIPScore、MetaCLIP；合成数据、数据增强、交织数据、质量评估、人工标注）
  - 3.2 软件与硬件（OpenCLIP、transformers、GPU 预算估算、torch.compile、xformers、FFCV）
  - 3.3 如何选择模型（何时用对比学习 / 掩码 / 生成 / 预训练骨干网）
  - 3.4 提升接地能力（边界框标注、负样本描述）
  - 3.5 提升对齐能力（指令微调、RLHF、LLaVA 系列、多模态上下文学习）
  - 3.6 提升富含文本图像的理解能力（LLaVAR、Monkey、Lumos）
  - 3.7 参数高效微调 PEFT（LoRA、QLoRA、VeRA、DoRA；CoOp、VPT；Adapter；MAPL、LiMBeR）
- **[4. 负责任的 VLM 评估](#sec-负责任的-VLM-评估)**
  - 4.1 视觉-语言能力基准测试（图像描述、文本到图像一致性、VQA、文本中心 VQA、零样本分类、组合推理、密集描述、合成数据评估）
  - 4.2 偏差与差异基准测试（基于分类、基于嵌入、语言偏差警告、训练数据概念影响）
  - 4.3 幻觉基准测试（CHAIR、POPE、GAVIE、MMHal-Bench）
  - 4.4 记忆化基准测试（déjà vu、k-NN 测试、文本随机化）
  - 4.5 红队测试
- **[5. 将 VLM 扩展到视频](#sec-将-VLM-扩展到视频)**
  - 5.1 基于 BERT 的早期视频工作（VideoBERT、MERLOT）
  - 5.2 早期融合 VLM 实现文本生成（VideoOFA）
  - 5.3 使用预训练 LLM（Video-LLaMA、Video-LLaVA、MiniGPT4-Video）
  - 5.4 评估中的机遇（EgoSchema、ActivityNet-QA、基于物理的合成数据）
  - 5.5 视频数据利用的挑战（稀缺的时序标注、名词偏置、计算成本、冗余问题）
- **[6. 结论](#sec-Key-Results)** — 主要发现与开放问题

## Problem / Gap

VLMs remain challenging for newcomers due to the rapid pace of the field, the diversity of training paradigms, and scattered practical guidance. Existing resources either focus narrowly on specific models or assume significant prior knowledge. There was no accessible, unified entry point covering the full VLM landscape — from foundational paradigms through training recipes, evaluation best practices, bias/fairness considerations, hallucination detection, and video extensions — aimed at students and researchers entering the field.

## VLM 的四种训练范式

### 1. 基于对比学习的 VLM（Contrastive-based VLMs）

对比训练从基于能量的模型（EBM）视角理解最好。InfoNCE 损失是核心——利用余弦相似度在表示空间中拉近正例对、推开负例对。

- **CLIP**：在 4 亿图文对上训练，随机初始化视觉和文本编码器，用对比损失映射到共享表示空间。ResNet-101 CLIP 达到 76.2% 零样本分类准确率，匹敌有监督 ResNet。
- **SigLIP**：将 InfoNCE 的多分类目标替换为二分类交叉熵，在小批量下效果更好。
- **Llip**：通过交叉注意力模块将图像编码条件化于目标描述，考虑描述的多样性，提高零样本迁移和检索性能。

### 2. 基于掩码目标的 VLM（Masking-based VLMs）

掩码可视为一种特定形式的去噪自编码器，噪声具有空间结构。

- **FLAVA**：图像编码器使用 ViT，文本编码器使用 Transformer，多模态编码器融合二者隐藏状态。联合训练多模态/单模态掩码建模损失和对比损失，在 35 个任务上达到 SOTA。
- **MaskVLM**：直接在像素空间和文本 Token 空间应用掩码，利用模态间的信息流（文本重建接收图像编码器信息，反之亦然）。
- **信息论视角**：VLM 可理解为解决率失真问题——减少冗余信息（Rate）、最大化预测信息（Distortion）。对比损失可视为无数据重建的压缩，掩码的 entropy bottleneck 受掩码信息量限制。

### 3. 基于生成的 VLM（Generative-based VLMs）

此范式考虑文本和/或图像的生成，训练成本通常最高。

- **CoCa**：除对比损失外，还使用生成损失（多模态文本解码器），无需额外融合模块即可执行 VQA 等任务。
- **CM3Leon**：两阶段训练——检索增强预训练（用 CLIP 检索相关多模态文档）+ 监督微调（多任务指令微调），SOTA 图文生成。
- **Chameleon**：混合模态基础模型，从头端到端训练，统一架构处理图像和文本 Token，早期融合策略实现跨模态推理。
- **用生成模型做判别任务**：通过贝叶斯定理（$p_\theta(c_i|x) = \frac{p(c_i)p_\theta(x|c_i)}{\sum_j p(c_j)p_\theta(x|c_j)}$），条件生成模型可直接用于分类。扩散模型分类虽推理成本高，但具有更好的分布外鲁棒性和组合推理能力。图像分词器基于 VQ-VAE 框架（CNN/ViT 编码器 + 矢量量化层 + 解码器），VIT-VQGAN 使用 ViT 替代 CNN。

### 4. 基于预训练骨干网络的 VLM（VLMs from Pretrained Backbones）

利用开源 LLM（如 Llama），学习图像编码器与 LLM 之间的映射，计算成本更低。

- **Frozen**：首个利用冻结 LLM 的模型。通过轻量级映射网络将视觉特征投影到文本 Token 嵌入，视觉编码器和线性映射从头训练，7B LLM 保持冻结。
- **MiniGPT-4**：使用 BLIP-2 的视觉编码器（Q-Former + ViT），仅训练线性投影层将视觉表示对齐到 Vicuna LLM 输入空间。4 张 A100 GPU 约 10 小时。MiniGPT-v2 通过统一接口和任务标识符支持多种视觉-语言任务。
- **Qwen-VL**：LLM 初始化为 Qwen-7B，视觉编码器基于 ViT-bigG，单层交叉注意力模块压缩视觉表示为固定长度（256）。
- **BLIP-2**：使用 Q-Former（~100-200M 参数）将冻结视觉编码器表示映射到冻结 LLM 输入空间。

## VLM 训练指南

### 3.1 训练数据

数据筛选方法分为三类：
- **启发式过滤**：单模态（文本复杂度、非英语过滤、分辨率/宽高比）+ 多模态（图像分类器筛选、文字检测去重）
- **基于预训练 VLM 排序**：CLIPScore 计算图文嵌入余弦相似度排名；MetaCLIP 使用元数据字符串匹配构建平衡数据集
- **多样性与平衡**：对长尾分布概念进行子采样，但完全平衡不现实——零样本能力主要取决于下游概念在训练数据中的覆盖度

**合成数据**：BLIP/BLIP2 生成描述性合成标注，LLaVA 作为标注模型，但大规模下合成标注多样性上限受限。**数据增强**：CLIP-rocket 使用不对称增强（弱+强），分离投影器（线性 vs 2层MLP），推理时插值融合。**交织数据**：OBELICS（自然交织）+MM1（合成交织），交织数据提升少样本性能。**质量评估**：QuRating、VILA、LAION-aesthetics 分别评估文本/图像/对齐质量，但缺乏整体的多模态质量评估方法。

### 3.2 软件与硬件

- **软件**：OpenCLIP、HuggingFace transformers 提供大多数 VLM 实现
- **GPU 预算**：CLIP 规模训练需 500+ GPU（数十万美元），但使用高质量数据集+掩码策略，从零训练对比模型约需 64 GPU（~1万美元）。利用预训练骨干网则成本更低
- **加速训练**：torch.compile、xformers（高效注意力）、FFCV（高效数据加载）
- **关键超参数**：图像分辨率 > 视觉编码器容量 > 视觉预训练数据 > 模态连接方式；文本+交织+图文对的正确混合比至关重要

### 3.3 如何选择模型

- **对比学习（CLIP）**：适合数据整理、检索、零样本分类；训练简单，表示空间具有图文双重语义
- **掩码**：无批量依赖（无需负样本），小批量友好；但需额外解码器，效率可能不如纯对比方法
- **生成模型**：可输出可视化结果（无需 k-NN 查找），学习隐式联合分布，但计算成本最高
- **预训练骨干网 + LLM**：适合快速构建 VLM；利用已有 LLM 知识，训练成本最低

### 3.4 提升接地能力（Grounding）

- **边界框标注**：提供空间定位信息
- **负样本描述**：训练模型理解"不是什么"，减少幻觉

### 3.5 提升对齐能力（Alignment）

- **指令微调**：LLaVA 系列（LLaVA、LLaVA-1.5、LLaVA-NeXT）将图文对齐到指令遵循格式
- **RLHF**：人类反馈强化学习对齐
- **多模态上下文学习**：在提示中提供图文示例

### 3.6 提升富含文本图像的理解

- **LLaVAR**：增强文本丰富图像的理解
- **Monkey**：提高输入分辨率以增强 OCR 能力
- **Lumos**：专注于文本中心的视觉理解

### 3.7 参数高效微调（PEFT）

- **LoRA 系列**：LoRA、QLoRA（量化）、VeRA（向量随机化）、DoRA（权重分解）
- **提示方法**：CoOp（连续提示优化）、VPT（视觉提示微调）
- **Adapter 方法**：在预训练层之间插入小型可训练瓶颈模块
- **映射方法**：MAPL、LiMBeR——学习视觉与语言之间的轻量级映射

## 负责任的 VLM 评估

### 4.1 视觉-语言能力基准测试

- **图像描述**：COCO Captions、Flickr30k、NoCaps
- **文本到图像一致性**：TIFA、VPEval、Davidsonian
- **VQA**：VQAv2、GQA、OK-VQA（需外部知识）
- **文本中心 VQA**：TextVQA、ST-VQA、DocVQA——精确字符串匹配指标不足
- **零样本分类**：ImageNet、ObjectNet、ImageNet-R/A/Sketch
- **组合推理**：Winoground、ARO、SugarCrepe——argmax 零向量参数可达 100% 准确率的漏洞
- **密集描述与裁剪匹配**：Dense Captioning、Crop-Caption Matching
- **合成数据评估**：利用程序生成的视觉场景进行受控测试

### 4.2 偏差与差异基准测试

- **基于分类的偏差**：FairFace、DollarStreet——测量跨人口统计的分类差异
- **基于嵌入的偏差**：WEAT、SEAT——测量嵌入空间中的刻板印象关联
- **语言偏差警告**：许多基准测试本身存在语言偏差捷径，影响评估有效性
- **训练数据概念影响**：评估特定训练数据概念对下游性能的影响

### 4.3 幻觉基准测试

- **CHAIR**：测量图像描述中的目标幻觉率
- **POPE**：轮询式目标探测评估
- **GAVIE**：基于 GPT-4 的细粒度幻觉评估
- **MMHal-Bench**：多模态幻觉基准

### 4.4 记忆化基准测试

- **déjà vu 记忆化**：检测训练数据记忆
- **k-NN 测试**：通过最近邻检索评估记忆化
- **文本随机化**：对比原始文本与随机化文本的训练动态

### 4.5 红队测试

对抗性评估，识别模型的安全漏洞和失败模式。

## 将 VLM 扩展到视频

### 5.1 早期基于 BERT 的方法

- **VideoBERT**：将 BERT 扩展到视频，使用矢量量化处理视觉 Token
- **MERLOT**：大规模视频-语言预训练

### 5.2 早期融合文本生成

- **VideoOFA**：早期融合视觉和文本模态进行视频文本生成

### 5.3 使用预训练 LLM

- **Video-LLaMA**：使用视频 Q-Former 桥接冻结视觉编码器和 LLM
- **Video-LLaVA**：将 LLaVA 扩展到视频理解
- **MiniGPT4-Video**：MiniGPT-4 的视频版本

### 5.4 评估中的机遇

- **EgoSchema**：自我中心视频理解
- **ActivityNet-QA**：基于活动视频的问答
- **基于物理的合成数据**：测试物理推理能力——当前视频 VLM 在物理推理任务上表现低于随机水平

### 5.5 视频数据的挑战

- 时序标注稀缺
- 名词偏置（模型依赖名词捷径）
- 计算成本（远高于图像）
- 冗余问题（相邻帧高度相似）

## Key Results

- 提供了统一分类法，将主流 VLM 方法归入四种明确定义的范式
- 识别出关键评估陷阱：语言偏差捷径降低基准测试有效性；部分组合推理基准上 argmax 零向量参数可达 100% 准确率
- 当前视频 VLM 在物理推理任务上表现低于随机水平
- 76 页涵盖约 200 篇参考文献，是 VLM 研究者首选的入门资源
- 重要发现：文本导向 VQA 的精确字符串匹配指标不足——会惩罚语义正确但措辞不同的答案

## Assumptions

- 读者具备深度学习和 Transformer 的基本知识
- 四种范式分类法是有用的方向指南，但承认范式边界在实践中是模糊的
- 训练建议基于已发表结果和社区最佳实践，最优方案可能快速演变
- 评估基准分析反映 2024 年初的基准状态

## Limitations / Failure Modes

- 非全面综述——刻意定位为入门/教程，许多专门的子领域仅简要涉及
- 四类分类法是简化；实际模型越来越多地混合多种范式
- 领域发展迅速，某些具体模型推荐可能很快过时（作者已认识到并侧重于通用原则）
- 训练指南推荐主要基于经验；许多 VLM 现象的理论理解仍不完整

## Reusable Ingredients

- 四种范式分类法作为理解和比较 VLM 方法的心智模型
- 实用训练清单：数据过滤启发式方法、GPU 预算估算公式、超参数指南
- 评估协议：多轴评估覆盖能力、偏差、幻觉、记忆化和红队测试
- 视频 VLM 设计模式：早期融合 vs. 预训练 LLM 骨干网方法
- PEFT 决策树：何时使用 LoRA vs. Adapter vs. Prompt Tuning
- 已知基准测试失败模式警告目录（语言偏差、精确匹配问题、组合推理捷径）

## Open Questions

- 如何设计能抵抗语言偏差捷径和随机重复的基准测试？
- 四种 VLM 范式能否统一为单一通用训练配方？
- 如何在不过高计算成本下扩展视频 VLM 以处理长时序推理？
- 什么评估协议能可靠检测 VLM 开放式生成中的幻觉？
- 如何在减轻记忆化的同时不降低模型实用性？

## Claims

- 四种范式分类法为理解截至 2024 年的 VLM 格局提供了充分的框架
- 负责任的评估需要超越标准能力基准，纳入偏差、幻觉、记忆化和红队测试轴
- 当前视频 VLM 基准测试由于名词偏置和有限的时序推理要求，夸大了进展

## Connections

- **Extends:** `paper:radford2021_clip` (CLIP — Section 2.2 中作为对比式 VLM 的典范广泛讨论；CLIPScore 用于全篇数据整理和评估)
- **Extends:** `paper:dosovitskiy2021_vit` (ViT — 作为 FLAVA、BLIP-2、MiniGPT 等的基础视觉编码器骨干网，VIT-VQGAN 中用于编码器/解码器)
- **Extends:** `paper:li2023_blip2` (BLIP-2 — Section 2.5.3 中以 Q-Former 为代表的预训练骨干网 VLM)

## Relevance to This Project

这是截至 2024 年 VLM 领域的权威入门指南。它将本知识图谱中的基础论文（ViT、CLIP、BLIP-2）整合为统一的叙述，提供了适用于任何 VLM 项目的实用训练指南，并建立了严格的评估标准。其四种范式分类法可作为理解新 VLM 论文之间关系的组织框架。
[[liu2021_swin]]
[[radford2021_clip]]
[[dosovitskiy2021_vit]]
[[li2023_blip2]]