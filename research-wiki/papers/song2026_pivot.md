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

## 一句话

**RL 训练不仅让 MLLM 答案更准，还能"重写"视觉编码器的内部表征**。用 DPO 训练的视觉编码器产生比 SFT 更强、更局部化的视觉特征，以不到标准视觉预训练 1% 的计算成本超越更大型编码器。

## 阅读遇到问题

第二面高亮处引用的论文，只是用DPO做后训练减少幻觉率，为什么和文章在做的DPO调整视觉表征能够联系起来？DPO在这些论文中的训练中发挥了什么作用？
- RLHF-V: Towards Trustworthy MLLMs via Behavior Alignment from Fine-Grained Correctional Human Feedback 
- OPA-DPO: Mitigating Hallucinations in LVLMs via DPO: On-Policy Data Hold the Key
- CHiP: Cross-modal Hierarchical Direct Preference Optimization for MLLMs

论文在CLIP MAE DINO都进行研究并且取得了一定的进展，后续把这些论文也看看

## 核心问题

MLLM 社区有一个隐含假设：MLLM 的能力主要来自 LLM backbone（参数最大、能力最强）。这导致**没人认真研究视觉 encoder 在 SFT vs RL 训练中到底发生了什么变化**。

本文填补了这个空白：**SFT（用于指令遵循）和 RL（用于偏好对齐）是如何调整 Encoder 表征的？**

## 方法

### 控制实验设计

#### 训练过程
- Stage 1 投影层的图文匹配训练
只训练投影层，用图-文训练数据训练模型的投影层。
- Stage 2 后训练中，用**相同数量的图文数据**、分别用 SFT 和 DPO 训练，公平对比：
在VQA , 视觉Grounding任务 ， 图片查找任务上做全参数
```
LSFT = -log π(y_chosen | I, q)                        ← 标准微调
LDPO = -log σ(β·[log(π_θ(y_chosen)/π_ref(y_chosen))   ← 偏好优化
                - log(π_θ(y_rejected)/π_ref(y_rejected))])
```

### 核心发现

1. **RL > SFT 在视觉密集型任务上**：DPO 在需要细粒度视觉理解的 VQA benchmark 上显著超越 SFT

2. **RL 重写视觉表征**：通过 ImageNet 分类、语义分割、**梯度可视化**等多种实验，证明 RL 的优化信号能传递回视觉 encoder，使其学到：
   - 更强的局部定位能力（物体在哪、边界在哪）
   - 更精确的细粒度信息（纹理、形状）
   - SFT 则停留在 "全局语义" 层面

3. **梯度可视化揭示**：RL 训练中，视觉 encoder 浅层也收到显著梯度信号（SFT 几乎只更新高层），说明 RL 在"重新训练" 视觉 encoder 的基础特征
   > ⚠️ **注意**：论文原文的 Grad-CAM 分析（Section 4.2, Figure 7）实际展示的是梯度在**空间维度**上的分布——DPO 梯度集中于问题相关区域，SFT 梯度散乱。论文并未做 per-block/layer 的梯度 norm 分析。"浅层收到显著梯度"是笔记作者基于以下间接证据的推断：(1) ImageNet linear probe 大幅提升暗示 encoder 整体表征被改写；(2) 分割 mask 更精确说明低级空间信息也被优化；(3) Larger LLM → better encoder（Finding 5）暗示梯度可穿透全 encoder。建议后续回看原文确认是否有 per-layer 分析。

### PIVOT 方法

把上述发现提炼为一个训练范式——**Preference-Instructed Vision OpTimization（PIVOT）**：

- **思想**：用 LLM head + DPO 来"二次锻造"视觉 encoder，训完后拆下来给新的 MLLM 用
- **不是新结构**，而是一个之前研究不充分的 training regime
- **流程**：
  1. 取一个现成的视觉 encoder（CLIP/SigLIP/DINOv2/MAE 等）
  2. 接到 Qwen2.5-1.5B LLM 上，用 3M instruction-following 样本 + 20K preference pairs 做 pretrain + DPO
  3. 训完后**拆下 vision encoder，冻结权重** → 称为 **PIVOT-enhanced encoder**
  4. 把增强后的 encoder 接入新 LLM，重新构建 MLLM（projector-only pretrain + instruction finetune）
- **成本**：18 小时 × 8×H100 → 不到标准视觉预训练的 1% 计算量
- **评估方式**：隔离 vision encoder 能力——接到新的 Qwen2.5-1.5B LLM 上，在 LAION/CC/SBU-558K 做 projector-only pretraining，然后在 Cambrian-737K 做 projector + LLM instruction finetuning

## PIVOT 实验结果

### 跨代超越

| 对比 | 结果 |
|------|------|
| SigLIP1-So/14 + PIVOT vs SigLIP2-So/16（原生） | PIVOT 旧代 encoder **超过**原生新一代（53.2 vs 52.4） |
| SigLIP2-So/16 + PIVOT vs SigLIP2-g/16（原生） | 小模型 + PIVOT **超过**大模型原生（55.6 vs 53.9） |

> **结论**：经过 PIVOT 后，旧一代 encoder 可超过新一代原生 encoder；较小但 PIVOT 过的 encoder 可超过更大的原生 encoder。

### DPO vs SFT

DPO 作为 PIVOT 默认选择，在所有 encoder 上优于 SFT（如 SigLIP2-g/16：DPO 56.7 vs SFT 55.4）。

### 通用性

PIVOT 对 CLIP、DINOv2、MAE、MoCo、ImageNet supervised ViT 等不同类型视觉模型**均有提升**，说明这是一种通用的 encoder 增强方法。

### 模型集成

单独一个 PIVOT 后的 SigLIP1-So/14 可超过一些多 encoder 组合（如 SigLIP1 + ConvNeXt-XXL）。若再与 ConvNeXt-XXL 组合，性能还能进一步提升。

**Finding 6**：现有视觉模型在 MLLM 中还有很大的提升空间，PIVOT 可以释放这种潜力。

## PIVOT 的贡献定位

1. PIVOT 不是一个全新方法，而是一个之前研究不充分的**训练范式**
2. 展示了 PIVOT 后的 encoder 构建 MLLM **明显强于**使用原始 encoder 的 MLLM——哪怕 SigLIP2 这种 SOTA encoder 在 MLLM 场景下仍有提升空间
3. 提供第一个证据：**DPO 不仅可以提升多模态任务表现，也能比 SFT 更积极地重塑视觉特征**，在标准视觉 benchmark 和多模态任务上都更有效

已有工作证明 SFT-style 的 language alignment 可以改善视觉表征，但 RL/DPO 带来的提升更强，未来可继续探索更多 RL 算法来训练视觉 encoder。

## 局限

- DPO 的偏好数据仍然需要人工或高质量数据构造
- 实验主要基于 LLaVA-OneVision 框架，Qwen-VL / InternVL 等架构上的泛化性待验证
- PIVOT 对 LLM backbone 的反向影响未深入研究
- SFT 和 DPO 的比较还可以更公平（如让 SFT 也利用 negative examples），以区分：DPO 的优势到底来自 preference objective，还是来自看到了 rejected response
- 作者对设计新的数据格式感兴趣，尤其是能更好利用 DPO 学习视觉表征的数据，列为未来工作

## 附录补充

- PPO、MPO、GRPO 等其他 RL 算法与 SFT 的比较
- 更适合 SFT 的设置
- Text-only benchmark
- PIVOT 消融实验

## 关键洞察

> 视觉 encoder 在 MLLM 中有 **"进化潜力"**——即使 SOTA 编码器（SigLIP2）仍有巨大提升空间。RL 是解锁这个潜力的钥匙。

> PIVOT 不是新结构，而是一个 training regime。它的卖点是：**不用重新做超大规模视觉预训练，只用少量偏好训练，就可能让现有 vision encoder 更适合 MLLM。**

## 对你的意义

如果你做多模态 RL（GRPO/DPO 训练 MLLM），这篇论文给出了一个强有力的论点：**RL 不只是让模型"回答更准"，它从根本上改变了模型 "看" 的方式。** 这对 "用 RL 优化 grounding" 方向是直接支撑。
