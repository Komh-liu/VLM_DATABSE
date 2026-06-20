# ToRead — 多模态 Grounding × RL 论文阅读

> **研究方向**: 多模态 Visual Grounding + RL 微调  
> **目标**: 6 个月内产出 7B 量级可独立跑通的实验 → 论文  
> **详细路线**: 见 [learning-roadmap.md](learning-roadmap.md)  
> **ICML 2026 调研**: 见 [icml2026_multimodal_rl_survey.md](icml2026_multimodal_rl_survey.md)

## 调研过滤规则

1. **排除 VLA/具身智能/3D** — 不关注机器人、World Model、3D grounding
2. **优先 7B 可复现** — Qwen2.5-VL-7B 级别
3. **优先开放问题** — 避开已被大团队占领的红海
4. **半年可产出** — 方法轻量，无需大规模预训练
5. **聚焦 Reward Design / Credit Assignment / Grounding Internalization**

---

## 阅读路线（10 篇必读 + 2 篇参考）

### 阶段 1: Grounding 表示入门（Week 1）

| # | 论文 | 概要 | PDF |
|---|------|------|-----|
| 1 | **Shikra** (2023) | 坐标当文本输出。MLLM grounding baseline | [pdf](pdfs/shikra_2023.pdf) |
| 2 | **Ferret** (ICLR 2024) | 离散 bin + 连续特征混合区域表征。现代标准方案 | [pdf](pdfs/ferret_2024.pdf) |

### 阶段 2: 核心张力 — 什么时候做 Grounding?（Week 2-3）

| # | 论文 | 概要 | PDF |
|---|------|------|-----|
| 3 | **VLM-R³** (NeurIPS 2025) | Grounding 时机作为 RL action。自适应选择 | [pdf](pdfs/vlmr3_neurips2025.pdf) |
| 4 | **Argus** (CVPR 2025) | Object-centric Grounded CoT。始终 ground | [pdf](pdfs/argus_cvpr2025.pdf) |
| 5 | **GRIT** (NeurIPS 2025) | BBox+文字交错推理。仅需 20 样本 | [pdf](pdfs/grit_neurips2025.pdf) |

### 阶段 3: RL 工具（Week 4）

| # | 论文 | 概要 | PDF |
|---|------|------|-----|
| 6 | **DeepSeekMath** (2024) | GRPO 原始论文。只读 §3 | [pdf](pdfs/deepseekmath_grpo_2024.pdf) |
| 7 | **DeepSeek-R1** (2025) | RLVR 范式起源。Rule-based verifiable reward | [pdf](pdfs/deepseek_r1_2025.pdf) |

### 阶段 4: ICML 2026 前沿（Week 5-6）

| # | 论文 | 概要 | PDF |
|---|------|------|-----|
| 8 | **MoCA** (ICML 2026 Spotlight) | 感知信用分配。区分 "看错 vs 想错" | [pdf](pdfs/moca_icml2026.pdf) |
| 9 | **iVGR** (ICML 2026) | Grounding 内化。训练时双流，推理时不输出坐标 | [pdf](pdfs/ivgr_icml2026.pdf) |
| 10 | **SSL4RL** (ICML 2026) | SSL 任务当奖励。无需人类标注。代码开源 | [pdf](pdfs/ssl4rl_icml2026.pdf) |

### 参考（不作为必读）

| # | 论文 | 用途 | PDF |
|---|------|------|-----|
| 11 | **VGent** (CVPR 2026) | 解耦推理与 grounding 的极端方案 | [pdf](pdfs/vgent_cvpr2026.pdf) |
| 12 | **Survey** (TPAMI 2025) | 354 篇综述。当词典翻 | [pdf](pdfs/survey_visual_grounding_2025.pdf) |

---

## Grounding 表示演进（读完阶段 1+2 后应能画出）

```
坐标当文本    离散 bin + 连续特征     SEG token 潜空间       Grounding 内化
Shikra  ──→  Ferret  ──→  (LISA)  ──→  iVGR
   │              │                     │                   │
   │              │                     │                   └── 推理时不输出坐标
   │              │                     └── 坐标消失在潜空间
   │              └── 弥合连续-离散 gap
   └── 最简单方案：LLM 原生能力
```

## Grounding 时机光谱（读完阶段 2+3+4 后应能画出）

```
始终 ground      推理中 ground     自适应 ground      解耦             不 ground
Argus           GRIT              VLM-R³            VGent            iVGR
   │               │                  │                │                │
   └── 最精确       └── 最少数据       └── RL 选择      └── 模块化        └── 内化
```

## 核心研究问题（你的潜在贡献空间）

1. **信用分配的层次化** (MoCA 延伸): 定位 vs 属性 vs 关系 vs 计数 → 怎么分别给奖励？
2. **Grounding 内化的边界** (iVGR 延伸): 什么时候必须显式 ground？什么时候内化更好？
3. **Grounding 驱动的幻觉检测** (交叉方向): 幻觉 = grounding failure → 用 grounding 做自修正

## 已读基础（不需要再补）

- ✅ CLIP, ViT — 视觉编码器和对比学习
- ✅ BLIP-2, LLaVA/LLaVA-1.5 — VLM 架构范式
- ✅ QwenVL 系列报告 — Dynamic Resolution, M-RoPE, grounding 能力

## 目录结构

```
ToRead/
  CLAUDE.md                          # 本文件
  learning-roadmap.md                # 详细周计划 + 自检清单
  icml2026_multimodal_rl_survey.md   # ICML 2026 多模态 RL 调研
  *.md                               # 论文概要（每篇 < 2KB）
  pdfs/                              # 论文 PDF（12 篇）
    shikra_2023.pdf
    ferret_2024.pdf
    vlmr3_neurips2025.pdf
    argus_cvpr2025.pdf
    grit_neurips2025.pdf
    vgent_cvpr2026.pdf
    deepseekmath_grpo_2024.pdf
    deepseek_r1_2025.pdf
    moca_icml2026.pdf
    ivgr_icml2026.pdf
    ssl4rl_icml2026.pdf
    survey_visual_grounding_2025.pdf
```
