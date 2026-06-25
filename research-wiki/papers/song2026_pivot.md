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

**RL 训练不仅让 MLLM 答案更准，还能"重写"视觉编码器的内部表征。** 用 DPO 训练的视觉编码器产生比 SFT 更强、更局部化的视觉特征，以不到标准视觉预训练 1% 的计算成本超越更大型编码器。

## 核心问题

MLLM 社区有一个隐含假设：MLLM 的能力主要来自 LLM backbone。这导致**没人认真研究视觉 encoder 在 SFT vs RL 训练中到底发生了什么变化**。本文填补了这个空白。

## 实验设计

### 训练流程

| Stage | 内容 | 训练参数 |
|-------|------|----------|
| Stage 1 | 投影层图文匹配训练 | 仅投影层 |
| Stage 2 | 后训练：VQA / 视觉 Grounding / 图片查找 | 全参数 |

Stage 2 使用**相同数量图文数据**，分别用 SFT 和 DPO 训练，确保公平对比：

```
LSFT = -log π(y_chosen | I, q)

LDPO = -log σ( β · [ log(π_θ(y_chosen) / π_ref(y_chosen))
                    - log(π_θ(y_rejected) / π_ref(y_rejected)) ] )
```

### 模型规模

- LLM: Qwen2.5 (0.5B / 1.5B / 3B / 7B)
- Vision Encoder: SigLIP2 (B/16, L/16, So/16, g/16)
- Projector: 2-layer MLP

## 核心发现

### 1. RL > SFT 在视觉密集型任务上

DPO 在需要细粒度视觉理解的 VQA benchmark 上显著超越 SFT。

### 2. RL 重写视觉表征

通过 ImageNet 分类、语义分割、梯度可视化等实验，证明 RL 优化信号传递回 encoder：

| 维度 | DPO (RL) | SFT |
|------|----------|-----|
| 局部定位 | 强（物体在哪、边界在哪） | 弱 |
| 细粒度信息 | 精确（纹理、形状） | 粗糙 |
| 表征层次 | 深入到低级特征 | 停留在全局语义 |

### 3. 梯度可视化

DPO 梯度**集中在问题相关区域**，SFT 梯度**散乱无焦点**（Section 4.2, Figure 7）。

> ⚠️ 笔记中"RL 浅层也收到显著梯度，SFT 只更新高层"是推断，论文原文未做 per-layer 梯度 norm 分析。依据：(1) ImageNet probe 大幅提升暗示整体表征被改写；(2) 分割 mask 更精确说明低级空间信息也被优化；(3) Larger LLM → better encoder (Finding 5)。

## PIVOT 方法

将上述发现提炼为训练范式——**Preference-Instructed Vision OpTimization**。

**核心思路**：用 LLM head + DPO "二次锻造" vision encoder，训完后拆下来给新 MLLM 用。

**训练流程**：

1. 取现成 encoder（CLIP / SigLIP / DINOv2 / MAE 等）
2. 接 Qwen2.5-1.5B LLM → 3M instruction 样本 + 20K preference pairs → pretrain + DPO
3. 拆下 vision encoder，冻结 → **PIVOT-enhanced encoder**
4. 接入新 LLM 重建 MLLM，在 LAION/CC/SBU-558K 上 projector-only pretrain，再在 Cambrian-737K 上 finetune

**成本**：18h × 8×H100，不到标准视觉预训练的 1%。

**评估方式**：接到新 Qwen2.5-1.5B LLM 上评估，隔离 encoder 本身能力。

## PIVOT 实验结果

### 跨代超越

| 对比 | 结果 |
|------|------|
| SigLIP1-So/14 + PIVOT vs SigLIP2-So/16（原生） | 旧代 **超过** 新一代（53.2 vs 52.4） |
| SigLIP2-So/16 + PIVOT vs SigLIP2-g/16（原生） | 小模型 **超过** 大模型（55.6 vs 53.9） |

### DPO vs SFT

DPO 在所有 encoder 上优于 SFT（如 SigLIP2-g/16：DPO 56.7 vs SFT 55.4），故将 DPO 作为 PIVOT 默认选择。

### 通用性

CLIP、DINOv2、MAE、MoCo、Supervised ViT 均有提升，说明 PIVOT 是通用增强方法。

### 模型集成

单 PIVOT SigLIP1-So/14 超过部分多 encoder 组合（如 SigLIP1 + ConvNeXt-XXL）；组合后还能更强。

> **Finding 6**：现有视觉模型在 MLLM 中仍有很大提升空间，PIVOT 可以释放这种潜力。

## PIVOT 贡献定位

1. 揭示一个之前研究不充分的**训练范式**
2. PIVOT 后的 encoder 构建 MLLM **明显强于**原始 encoder——哪怕 SigLIP2 也有提升空间
3. 首个证据：**DPO 比 SFT 更积极地重塑视觉特征**，在视觉 benchmark 和 MLLM 任务上均更有效

SFT-style language alignment 能改善视觉表征，但 RL/DPO 提升更强，未来可探索更多 RL 算法训练 vision encoder。

## 局限

- DPO 偏好数据仍需人工或高质量构造
- 实验基于 LLaVA-OneVision，Qwen-VL / InternVL 上泛化性待验证
- PIVOT 对 LLM backbone 的反向影响未研究
- SFT vs DPO 比较可更公平（让 SFT 也看 negative examples），以区分：优势来自 preference objective 还是 rejected response
- 需要设计更适配 DPO 视觉表征学习的数据格式（未来工作）

## 附录补充

- PPO、MPO、GRPO 与 SFT 对比
- SFT 友好设置下的实验
- Text-only benchmark
- PIVOT 消融实验

## 关键洞察

> 视觉 encoder 在 MLLM 中有"进化潜力"——即使 SOTA 编码器仍有巨大提升空间，RL 是解锁它的钥匙。
>
> PIVOT 不是新结构，而是 training regime：**不用超大规模预训练，少量偏好数据即可让现有 encoder 更适合 MLLM。**

## 对你的意义

做多模态 RL（GRPO/DPO 训练 MLLM）时，这篇论文提供了强论点：**RL 不只是让模型"回答更准"，它从根本上改变了模型"看"的方式。** 直接支撑"用 RL 优化 grounding"方向。

## 待读相关论文

引用的 DPO 幻觉消减工作，与本文 DPO 调整视觉表征的关联待理解：

- RLHF-V: Towards Trustworthy MLLMs via Behavior Alignment from Fine-Grained Correctional Human Feedback
- OPA-DPO: Mitigating Hallucinations in LVLMs via DPO: On-Policy Data Hold the Key
- CHiP: Cross-modal Hierarchical Direct Preference Optimization for MLLMs

此外论文在 CLIP / MAE / DINO 上均取得进展，后续跟进阅读。
