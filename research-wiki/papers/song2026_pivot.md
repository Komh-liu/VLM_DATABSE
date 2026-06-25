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

**RL 训练不仅让 MLLM 答案更准，还能"重写"视觉编码器的内部表征**。用 DPO 训练的视觉编码器产生比 SFT 更强、更局部化的视觉特征，以不到标准视觉预训练 1% 的计算成本超越更大型编码器。作者的结论是RL使得视觉Encoder得到了一个更好的视觉表征
## 阅读遇到问题

第二面高亮处引用的论文，只是用DPO做后训练减少幻觉率，为什么和文章在做的DPO调整视觉表征能够联系起来？DPO在这些论文中的训练中发挥了什么作用？
- RLHF-V: Towards Trustworthy MLLMs via Behavior Alignment from Fine-Grained Correctional Human Feedback 
- OPA-DPO: Mitigating Hallucinations in LVLMs via DPO: On-Policy Data Hold the Key
- CHiP: Cross-modal Hierarchical Direct Preference Optimization for MLLMs
- 
论文在clip MAE DINO都进行研究并且取得了一定的进展，后续把这些论文也看看

## 核心问题

MLLM 社区有一个隐含假设：MLLM 的能力主要来自 LLM backbone（参数最大、能力最强）。这导致**没人认真研究视觉 encoder 在 SFT vs RL 训练中到底发生了什么变化**。

本文填补了这个空白：**SFT --用于指令遵循和 RL 用于调整人类偏好是如何调整Encoder表征的？**

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

### PIVOT 方法

把上述发现提炼为一个简单配方：
- 用 DPO + 1.5B 小 LLM head 训练视觉 encoder
- 训完后，增强过的视觉 encoder **可以拆下来**换到任意 MLLM 里
- 18 小时 × 8×H100 → 不到标准视觉预训练的 1% 计算量

## 核心结果

| 对比 | 结果 |
|------|------|
| SigLIP2-So/16 + PIVOT | **超过** SigLIP2-g/16（参数量大得多） |
| SigLIP1-So/14 + PIVOT | **超过** SigLIP2-So/16（下一代编码器） |
| 通用性 | 在 CLIP / DINOv2 / MAE / MoCo / Supervised ViT 上均有效 |
| MLLM 下游 | 装上 PIVOT encoder 的 MLLM 在 VQA 上显著提升 |

## 关键洞察

> 视觉 encoder 在 MLLM 中有 **"进化潜力"**——即使 SOTA 编码器（SigLIP2）仍有巨大提升空间。RL 是解锁这个潜力的钥匙。

## 局限

- DPO 的偏好数据仍然需要人工或高质量数据构造
- 实验主要基于 LLaVA-OneVision 框架，Qwen-VL / InternVL 等架构上的泛化性待验证
- PIVOT 对 LLM backbone 的反向影响未深入研究

## 对你的意义

如果你做多模态 RL（GRPO/DPO 训练 MLLM），这篇论文给出了一个强有力的论点：**RL 不只是让模型"回答更准"，它从根本上改变了模型 "看" 的方式。** 这对 "用 RL 优化 grounding" 方向是直接支撑。