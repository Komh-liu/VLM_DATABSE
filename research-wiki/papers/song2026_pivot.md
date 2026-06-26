---
type: paper
node_id: paper:song2026_pivot
title: "RL makes MLLMs see better than SFT"
authors: ["Junha Song", "Sangdoo Yun", "Dongyoon Han", "Jaegul Choo", "Byeongho Heo"]
year: 2026
venue: "ICLR 2026 Poster"
external_ids:
  arxiv: "2510.16333"
  doi: null
  s2: null
tags: ["RL", "DPO", "MLLM", "vision-encoder", "visual-representation", "SFT", "preference-optimization"]
added: 2026-06-22T00:00:00Z
---

# PIVOT: RL makes MLLMs see better than SFT

## 一句话概括

> **"RL produces stronger and more localized visual representations compared to SFT, boosting the ability of the vision encoder for MLLM."** (Page 1, Abstract)

论文通过系统对比 SFT 和 DPO 对 MLLM 中视觉编码器的影响，发现 RL 训练不仅优化了模型输出，更从根本上重塑了视觉编码器的内部表征，使其产生更强、更局部化的视觉特征。基于此发现，作者提出 PIVOT（Preference-Instructed Vision OpTimization）配方，用 <1% 标准视觉预训练的计算成本，使小型编码器超越更大规模的 SOTA 编码器。

---

## 核心研究问题

> "The field lacks a systematic comparison within MLLMs between SFT for instruction-following and RL for preference alignment... leaving a significant void in our understanding of how SFT and RL differ in reshaping visual representations." (Page 2, Introduction)

研究问题：SFT 和 RL（以 DPO 为代表）在 MLLM 后训练中对**视觉编码器的表征**分别产生了什么影响？社区长期以来以 LLM 为中心的假设导致对这一问题的系统性研究缺失。

---

## 实验设计

### 训练流程（Section 3.1 + Section E.1）

| 阶段 | 内容 | 数据 |
|:---|:---|:---|
| Stage 1 | 投影层预训练 + 端到端预训练 | BLIP_LAION_CC_SBU_558K + LLaVA-OV-3.2M |
| Stage 2 后训练 | 分别用 SFT 或 DPO 全参训练，**公平对比** | MPO 数据集，随机采样 20K |

**关键设计**：SFT 和 DPO 使用**完全相同的数据量**，仅在损失函数上不同，以隔离训练策略的影响。

**Loss 定义**（Equation 1, Page 3）：
- SFT: 最大化 $y_i^c$ 的似然
- DPO: $\mathcal{L}_{\text{DPO}} = -\log \sigma(\beta \cdot [\log(\pi_\theta(y_c)/\pi_{\text{ref}}(y_c)) - \log(\pi_\theta(y_r)/\pi_{\text{ref}}(y_r))])$

**评测基准**（Section 3.1）：Cambrian 评测套件，覆盖 16 个任务，分 4 类：General、Knowledge、OCR & Chart、Vision-Centric VQA。

---

## 核心发现

### Finding 1（Page 4）

> "Increasing the capacity of the vision encoder in MLLMs is particularly important for tasks requiring fine-grained visual understanding."

**来源**：Figure 2 中，将 SigLIP2-B/16 替换为 SigLIP2-g/16，DPO-tuned 模型在 Vision-Centric 上差距达 +4.5%，在 OCR & Chart 上达 +10.6%，而在弱视觉任务上仅 +1.9%。

---

### Finding 2（Page 5）

> "Preference alignment (DPO) produces MLLMs with superior performance to SFT, especially on strongly vision-related tasks."

**来源**：Figure 2-3 数据显示：
- Knowledge VQA（ScienceQA, MathVista）上 DPO vs SFT 仅 +0.3%（marginal）
- OCR & Chart VQA 上 DPO 超出 +4.2%（SigLIP2-L/16）
- Vision-Centric VQA 上 DPO 超出 +2.4%
- 即使 scaling LLM 到 7B，DPO 在 OCR & Chart 和 Vision-Centric 上仍分别保持 +3.1% 和 +4.2% 的差距

> "It highlights the superiority of DPO, particularly on tasks requiring detailed visual understanding, and further implies that preference alignment impacts the model's visual processing capabilities, **beyond the language model**."

---

### Finding 3（Page 6）

> "MLLM training not only adapts the language model but also reshapes the visual representations that determine how the model sees an image." DPO不仅仅改变了调整了语言模型的遵从人类偏好能力，也调整了视觉表征。这可以从仅仅使用两个MLP层就能够在分类任务上表现更好得到证明。

**支撑证据**（Section 4.2, Figure 6）：
- ImageNet linear probing 显示 DPO 后编码器 Top-1 准确率比 SFT 高 +1.83%（SigLIP2-So/16 + Qwen-3B）
- DPO 比 SFT 高 +1.96%（SigLIP2-L/16 + Qwen-1.5B）
- **关键结论**："DPO—a prevalent RL method in the LLM community—is more effective than SFT, not only for aligning LLMs but also for **learning visual representations**."

---

### Finding 4（Page 7）

> "DPO steers the vision encoder toward a more fine-grained analysis of visual information, improving its object localization capabilities."

**支撑证据**：
- **Grad-CAM 可视化**（Figure 7, Page 7）：DPO 的梯度信号精确聚焦于问题相关区域，而 SFT 信号分散。论文原话："large gradients primarily occur in question-relevant regions... the signal from DPO is precisely focused on semantically relevant regions."
- **语义分割探测**（Figure 8, Page 7）：在 ADE20K 上用冻结编码器 + 2-layer MLP 做分割，DPO-tuned 编码器在 patch-level recall 上 consistently 优于 SFT。例如 CLIP-L/14 336px 上 DPO 比 SFT 高 1.08%p。
- **定性结果**（Figure 9, Page 8）：DPO-trained 编码器产生更贴近 ground truth 的分割图。

---

### Finding 5（Page 8）

> "The vision encoder benefits from a larger LLM, which provides more informative backward signals for visual representation within an MLLM."

**支撑证据**（Figure 6, 10）：
- DPO 训练下，SigLIP2-So/16 搭配 7B LLM 比搭配 0.5B LLM 的 ImageNet 准确率高 +4.4%（Page 6）
- 表征对齐分数（Huh et al., 2024）随 LLM 规模增大而一致提高（Figure 10）

---

## RL 优势的泛化性验证（Section 5）

论文验证了 RL 优势**不限于 DPO**：

| 算法 | 结果 |
|:---|:---|
| **GRPO**（Table 1, Page 8-9） | 比 SFT 高 +3.1% MLLM 平均分，OCR&Chart +4.3%p，Vision-Centric +3.4%p；ImageNet +1.93%p，Segmentation +1.83%p |
| **PPO**（Table C, Page 22） | LLaVA-1.0-7B 上 PPO 比 SFT 高 +1.5% 平均分，OCR&Chart +3.2%p |
| **MPO**（Table D, Page 23） | 与 DPO 效果相当，均优于 SFT |

> "These results confirm that the benefits of RL over SFT are not specific to DPO but generalize to other RL formulations." (Page 9)

---

## PIVOT 方法（Section 6）

### 定义（Page 9）

> "We reframe this training process into an effective strategy for evolving vision models, which we term **Preference-Instructed Vision OpTimization (PIVOT)**."

**流程**（Section 6.1, Figure 11）：
1. 将现有视觉编码器（CLIP/SigLIP/DINOv2/MAE/MoCo/Supervised ViT）接到 LLM head 上
2. 用 Stage 1 预训练（3M 样本）+ Stage 2 DPO 后训练（20K 偏好对）
3. **拆下**训练好的视觉编码器（冻结）
4. 接到**新的 LLM** 上，仅训练投影层，在 Cambrian-737K 上评测

### 核心结果（Table 2, Page 11）

| 对比 | 结果（Avg VQA） |
|:---|:---|
| SigLIP1-So/14 + PIVOT | **53.2%** > SigLIP2-So/16（**52.4%**）— 超越新一代编码器 |
| SigLIP2-So/16 + PIVOT | **55.6%** > SigLIP2-g/16（**53.9%**）— 2.5× 参数量的编码器，PIVOT 超越大模型 |
| SigLIP2-g/16 + PIVOT | **56.7%** > SigLIP2-g/16 + SFT（**55.4%**）— DPO 优于 SFT |
| CLIP-L/14 + PIVOT | **49.5%** > CLIP-L/14 原始（**46.3%**） |
| DINOv2-g/1B + PIVOT | **43.6%** > 原始（**40.9%**） |
| MAE-h/632M + PIVOT | **39.7%** > 原始（**36.8%**） |
| MoCo-b/86M + PIVOT | **37.5%** > 原始（**35.3%**） |
| ImageNetSup-h/632M + PIVOT | **37.7%** > 原始（**35.5%**） |

### Finding 6（Page 10）

> "Existing vision models possess substantial potential for improvement within MLLMs, which can be unlocked by PIVOT."

---

## 计算成本（Page 2）

> "This enhancement is achieved with just **18 hours of training on 8 H100 GPUs** using a Qwen2.5-1.5B LLM-head. This amounts to **fewer than 1% of GPUs of standard vision pre-training**, with SigLIP2 trained on up to 2K TPUv5e chips."

---

## 局限（Section D.4, Page 31-32）

论文明确承认的局限：

1. **数据量不完美匹配**："each DPO update uses $(I_t, x_t, y_t^c, y_t^r)$，whereas SFT relies only on $(I_t, x_t, y_t^c)$，so the amount of supervision per iteration is not perfectly matched." (Page 31-32)

2. **数据分布可能偏向 DPO**："our comparison relies on preference pairs sourced from the MPO dataset, which may be more favorable to DPO than to SFT." (Page 32)

3. **论文的缓解措施**（Figure 4, Section B.3, B.4）：
   - DPO with 3K 样本（60.4%）> SFT with 40K 样本（59.5%）— 证明 DPO 数据效率更高
   - 在 SFT-friendly 数据上（NoThink, Under30）DPO 仍优于 SFT（Table E）
   - 在数据分布偏移下 DPO 保持鲁棒，SFT 严重下降（Figure B）

---

## 论文引用的相关工作与本文的区别（Section A.3, D.3）

论文在 Related Work (Section A.3, Page 20, Table A) 中对 DPO-based MLLM 工作的定位：

| 论文 | 论文中的描述 |
|:---|:---|
| **RLHF-V** (Yu et al., 2024) | 被列为 CVPR 2024 DPO-based MLLM 工作之一 |
| **OPA-DPO** (Yang et al., 2025c) | "reweight token-level losses on disagreement tokens between chosen and rejected responses" |
| **CHiP** (Fu et al., 2025a) | "incorporating visual preference data reduces perceptual errors in MLLMs" |

**本论文与它们的区别**（论文自述，Section D.3, Page 31）：
> "Prior work has reported performance gains when RL is applied to an MLLM pretrained with SFT... Despite this evidence of RL's effectiveness, SFT has remained the dominant training strategy... Our study aims to strengthen the RL-based MLLM literature by demonstrating the effectiveness of RL in the post-training stage... we compare a model trained with SFT in Stage 1 and DPO in Stage 2 to a model trained with SFT in both stages. Even under this comparison, we find that DPO remains advantageous."

**核心区别**：前述工作大多报告了 "SFT then DPO" 优于 "仅 SFT" 的现象，但**未分析视觉编码器的变化**。本篇是第一个系统性地证明 RL 的优化信号会反向传播并重塑视觉编码器底层表征的工作。

---

## 关键洞察

> 视觉 encoder 在 MLLM 中有 **"进化潜力"**——即使 SOTA 编码器（SigLIP2）仍有巨大提升空间。RL 是解锁这个潜力的钥匙。

## 对你的意义

如果你做多模态 RL（GRPO/DPO 训练 MLLM），这篇论文给出了一个强有力的论点：**RL 不只是让模型"回答更准"，它从根本上改变了模型 "看" 的方式。** 这对 "用 RL 优化 grounding" 方向是直接支撑。

## 后续阅读建议

基于论文实际引用和实验范围：

1. **RLHF-V / OPA-DPO / CHiP** — 重点关注：是否更新了视觉编码器？如果更新了，这些论文是否注意到了编码器变化？（本篇认为它们未做此分析）

2. **Cambrian (Tong et al., 2024a)** — 本篇采用的评测框架和 PIVOT 评估协议均来自 Cambrian，是其方法论基础

3. **Perception Encoder (Bolya et al., 2025)** — "Its language alignment stage follows a strategy similar to the 'SFT' setting in Table 2. Unlike their focus on SFT-driven representation changes, we investigate how **RL** training influences vision representations." (Page 21) — 这是与 PIVOT 最直接可对比的工作

4. **GRPO vs SFT 实验**（Section 5）— 使用 VLM-R1 代码库，如果你的研究方向涉及 GRPO，这是直接相关的实验设置参考
