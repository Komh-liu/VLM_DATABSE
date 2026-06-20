# 阅读路线图：Preference-Optimized Grounding (POG)

> **目标**: 偏好学习替代 GRPO 训练 7B MLLM Grounding，解决 GRPO 三大问题（LLD崩溃、感知脱节、模态失衡）
> **时间**: 2026-06-21 → 2026-12-21（6 个月）
> **投稿目标**: ICLR 2027（~2026 年 9-10 月截稿）
> **硬件估计**: 4×A100 或 2×A/H100

---

## 总览

```
进度条：
[基础阅读 ████████░░░░░░░░░░░░]  4 周
[实验搭建 ████████████░░░░░░░░]  6 周  
[训练调试 ████████████████░░░░]  8 周
[评估写文 ████████████████████] 12 周（约 6 个月）
                             ↑
                 现在（2026-06-21）
```

---

## 第一阶段：领域切入（Week 1，~2026-06-21 → 06-27）

### 目标
> 理解视觉 grounding 是什么、坐标怎么表示、7B MLLM 怎么输出坐标。**不涉及 RL**，先建立 grounding 认知。

### 阅读内容

#### ① Grounding 表示（2 篇）

| 论文 | 页数 | 时间 | 深度要求 |
|------|------|------|---------|
| **Shikra** (arXiv 2023) | ~8 页正文 | 1h | 理解"坐标即文本"范式，知道 `[x1,y1,x2,y2]` 怎么嵌入 LLM |
| **Ferret** (ICLR 2024) | ~10 页 | 2h | 理解离散 bin + 连续特征的混合区域表征 |

**阅读重点**：
- Shikra: §3 Method → 坐标格式设计、数据构造
- Ferret: §3.1 Hybrid Region Representation → 离散化策略

**读完能回答**：
- [ ] LLM 怎么输出坐标？（Shikra vs Ferret 两种方案各怎么做？）
- [ ] 连续坐标 → 离散 token 有什么精度损失？
- [ ] Qwen2.5-VL 的 grounding 格式是哪种？用哪个特殊 token？

#### ② 基础设施阅读（2 篇）

| 论文 | 页数 | 时间 | 深度要求 |
|------|------|------|---------|
| **Qwen2.5-VL 技术报告** | ~30 页 | 2h | §3.2 坐标格式、§4 下游 benchmark（扫读 grounding 相关段） |
| **Survey** (TPAMI 2025) | ~27 页 | 1h | 只翻目录 + §5 方法分类树 + 文献索引 |

**阅读重点**：
- Qwen2.5-VL: §3.2 Dynamic Resolution + M-RoPE, §4.2 REC 结果
- Survey: 方法分类树 → 找到 "你读的论文在领域中的位置"

**读完能回答**：
- [ ] Qwen2.5-VL 支持哪几种 grounding 格式？
- [ ] Dynamic Resolution 是怎么工作的？
- [ ] Grounding 领域的方法分类树是什么？

### ✅ 阶段一产出
- [ ] 能自己写出 Qwen2.5-VL 的 grounding 推理代码
- [ ] 在至少 1 个 REC benchmark（RefCOCO/RefCOCOg）上验证了 7B baseline
- [ ] 写好了 Ref-Adv benchmark 的评估脚本

---

## 第二阶段：GRPO Baseline 拆解（Week 2，~2026-06-28 → 07-04）

### 目标
> 理解你要替代的对象——GRPO 怎么训练 grounding、哪里好哪里坏。**这是你的 motivation 来源。**

### 阅读内容

#### ① GRPO 原文（1 篇）

| 论文 | 页数 | 时间 | 深度要求 |
|------|------|------|---------|
| **DeepSeekMath: GRPO** (2024) | §3 公式推导 | 1.5h | 理解 group-based advantage 估计、KL 正则化、reward normalization |

**阅读重点**：
- §3.1 GRPO 公式推导：`A_i = (r_i - mean(r)) / std(r)`
- §3.2 训练细节

**读完能回答**：
- [ ] GRPO 和 PPO 的核心区别是什么？
- [ ] 为什么 GRPO 不需要 critic model？
- [ ] Group-based advantage 的 group size 怎么选？

#### ② GRPO × Grounding 的问题诊断（2 篇）

| 论文 | 页数 | 时间 | 深度要求 |
|------|------|------|---------|
| **Faithful GRPO** (arXiv 2604.08476) | ~8 页 | 2h | **全文精读** — 你的 motivation 核心来源 |
| **LLDS: GRPO Collapse** (ICML 2026, arXiv 2512.04220) | ~9 页 | 2.5h | **理解 LLD 死亡螺旋机制** — GRPO 不稳定性的根因 |

**阅读重点**：
- Faithful GRPO: §1 发现的 24.5% 不一致率、§3.1 逻辑不一致 + 视觉 grounding 差、§4.2 消融实验
- LLDS: §2 LLD 定义与检测、§3 三阶段崩溃轨迹、§4 LLDS 正则化

**读完能回答**（这是你的论文 story 的核心）：
- [ ] GRPO 训练后存在 "答案对但 grounding 错" 的比例是多少？
- [ ] CoT 不一致率具体指什么？怎么测量？
- [ ] LLD 死亡螺旋的三个阶段是什么？
- [ ] 为什么这会给 DPO 方案创造机会？

#### ③ DeepSeek-R1（快速扫读）

| 论文 | 页数 | 时间 | 深度要求 |
|------|------|------|---------|
| **DeepSeek-R1** (2025) | ~20 页 | 1h | §2 RLVR 范式、cold-start 数据设计 |

**阅读重点**：
- §2: Rule-based verifiable reward → grounding 场景怎么用？
- **不读**: 数学推理部分

### ✅ 阶段二产出
- [ ] 能用文字清晰解释 "为什么 GRPO 不适合 grounding"（3 个原因）
- [ ] 能复述 Faithful GRPO 的 24.5% 不一致率发现
- [ ] 理解 LLD 崩溃机制

---

## 第三阶段：GRPO × Visual Grounding SOTA（Week 3，~2026-07-05 → 07-11）

### 目标
> 理解当前 GRPO + grounding 的 SOTA pipeline 是什么样的，你的直接竞争对手在做什么。

### 阅读内容

#### ① Grounding + RL 的标准方案（3 篇）

| 论文 | 页数 | 时间 | 深度要求 |
|------|------|------|---------|
| **VLM-R³** (NeurIPS 2025) | ~9 页 | 2.5h | **精读** — grounding 时机作为 RL action 的标准方案 |
| **GenSeg-R1** (arXiv 2602.09701) | ~8 页 | 2h | **精读** — GRPO+grounding 的 SOTA setting，是你的直接 baseline |
| **GRIT** (NeurIPS 2025) | ~8 页 | 1.5h | 框架理解 — 极简 RL grounding 的可行性 |

**阅读重点**：
- VLM-R³: §3 R-GRPO 算法、§4 reward design、实验结果中 grounding accuracy vs reasoning accuracy
- GenSeg-R1: §3 方法、实验设置（benchmark, metric, training details）
- GRIT: §3 20 样本 data construction、§4 与 full-data 方法的对比

**读完能回答**：
- [ ] GenSeg-R1 在 RefCOCOg 的 mask cIoU 是多少？（你的 baseline）
- [ ] VLM-R³ 的 R-GRPO 跟标准 GRPO 有什么区别？
- [ ] GRIT 为什么 20 个样本就够？
- [ ] 这些方法的共同弱点是什么？(→ 你的 DPO 方案的优势在哪)

#### ② 补充阅读——Grounding 时机光谱

读完 VLM-R³ / GRIT / Argus 后，你应该能画出：
```
始终 ground      推理中 ground    自适应 ground     不 ground
Argus            GRIT            VLM-R³           iVGR (下周读)
```

### ✅ 阶段三产出
- [ ] 能用一条命令跑通 GenSeg-R1 的评估
- [ ] 记录下了 GRPO baseline 在你目标 benchmark 上的分数
- [ ] 对 "DPO 在哪比 GRPO 好" 有了初步论证

---

## 第四阶段：偏好学习基础设施（Week 4，~2026-07-12 → 07-18）

### 目标
> 理解偏好学习方法论：DPO 理论基础 + 各变体的优劣 + 与 RL 的对比

### 阅读内容

#### ① 偏好学习核心（2 篇）

| 论文 | 页数 | 时间 | 深度要求 |
|------|------|------|---------|
| **DPO** (NeurIPS 2023) | §2-4 | 2h | **精读** — DPO loss、reference model、KL 隐式正则化 |
| **SimPO** (2024) | §3 | 1.5h | 对比阅读 — 无 reference model 的变体 |

**阅读重点**：
- DPO: §3 DPO 推导、§4.1 训练设置
- SimPO: §3 SimPO loss、与 DPO 的对比

**读完能回答**：
- [ ] DPO loss 的推导假设是什么？（Bradley-Terry preference model）
- [ ] Reference model 的作用是什么？
- [ ] SimPO 和 DPO 的核心区别在哪？
- [ ] 你的场景用 DPO 还是 SimPO？为什么？

#### ② 你的竞争方案（2 篇）

| 论文 | 页数 | 时间 | 深度要求 |
|------|------|------|---------|
| **PRPO** (arXiv 2606.08708) | ~9 页 | 2h | 精读 — Token 级信用分配，与 DPO 思路最接近 |
| **MoCA** (ICML 2026 Spotlight) | ~8 页 | 2h | 精读 — 感知信用分配，区分"看错 vs 想错" |

**阅读重点**：
- PRPO: §3 RVD + PAR 公式、§4 实验结果
- MoCA: §3 方法、credit assignment 的层次化设计

**读完能回答**：
- [ ] PRPO 的 "robust visual dependency" 是怎么定义的？
- [ ] MoCA 的 credit assignment 和你的 DPO 方案各有什么优劣？
- [ ] 你能 claim "DPO 方案不需要显式的 credit assignment 机制" 吗？

### ✅ 阶段四产出
- [ ] 写好了 DPO 训练脚本的框架
- [ ] 确定了 DPO 变体选择（DPO vs SimPO vs IPO）
- [ ] 明确了你的方案 vs PRPO/MoCA 的区别和优势

---

## 第五阶段：前沿对抗 + 读剩下的参考（Week 5，~2026-07-19 → 07-25）

### 目标
> 补充 ICML 2026 其他相关前沿工作，确保你的方案在文献中没有被 overclaim

### 阅读内容

#### ① 前沿 grounding × RL（2 篇）

| 论文 | 页数 | 时间 | 深度要求 |
|------|------|------|---------|
| **iVGR** (ICML 2026) | ~8 页 | 2h | 精读 — Grounding 内化的极端方案，你的反面对比 |
| **MGPO** (arXiv 2507.05920) | ~8 页 | 1.5h | 框架理解 — grounding RL 冷启动问题 |

**阅读重点**：
- iVGR: §3 双流训练、为什么推理时不输出坐标
- MGPO: §3 多轮对话模板、cold-start 设计

**读完能回答**：
- [ ] iVGR 的 "grounding 内化" 和你的 DPO grounding 保真度优化，是什么关系？
- [ ] MGPO 的 grounding 冷启动问题，DPO 方案能避免吗？

#### ② 参考阅读（按需翻）

| 论文 | 时间 | 翻什么 |
|------|------|--------|
| **ReVisual-R1** | 1h | §3 staged pipeline 设计，做 ablation 参考 |
| **SSL4RL** (ICML 2026) | 1h | SSL 做 reward 的思想——如果 DPO 效果不好可以备用 |
| **VGent** (CVPR 2026) | 1h | 模块化解耦推理与 grounding 的极端 |
| **Argus** (CVPR 2025) | 1h | Always-ground 的极端，grounded CoT 对比 |

### ✅ 阶段五产出
- [ ] 完成了完整的前人工作梳理（Related Work §2 的初稿）
- [ ] 确认了你的方法还没有被做过
- [ ] 完成了作者对比表（方法 × 指标 × 优势 × 劣势）

---

## 第六阶段及以后：实验 + 论文（Week 6+）

### 实验流水线

```
Week 6-7:   构造 GFP 偏好数据集 (5K-10K pairs)
Week 8-9:   第一轮 DPO 训练 + 调参
Week 10-11: 评估（RefCOCOg + Ref-Adv + GSEval + 保真度指标）
Week 12-14: 消融实验 + 对比 GRPO baseline
Week 15-16: 论文初稿
Week 17-18: 修改 + 投稿（ICLR 2027，预计 9-10 月）
```

### 实验需要回看的内容

当你在实验中遇到问题时，翻这些章节：

| 问题 | 翻什么 |
|------|--------|
| DPO training loss 不收敛 | DPO §3, SimPO §3 |
| 通用能力退化 | DPO §4.1 (KL 正则化) |
| 偏好对质量不够 | GRIT §3, GenSeg-R1 §3 |
| 评估指标不明确 | Faithful GRPO §4.2 (不一致率指标) |
| Grounding baseline 太低 | Qwen2.5-VL 技术报告 §4 |
| 不知道加什么 ablation | MoCA §4, PRPO §4 |

---

## 非必读论文（无需碰）

这些论文与你的方向有部分关联，但不建议在当前 6 个月内读。如果审稿人提了再补：

- **Grounding DINO** / **GLIP** / **MDETR** — DETR-based 方法，MLLM 时代已过时
- **Kosmos-2** / **GLaMM** / **LLaVA-Grounding** — 早期的 MLLM grounding，已被 Shikra/Ferret 覆盖
- **3D Grounding** 方向全部（ReasonGrounder, S²-MLLM, GS-Reasoner）— 跟你的聚焦无关
- **Video Grounding** 方向全部（Video-R1, VTimeCoT, TPO）— 除非你未来想做扩展
- **Agent / Tool Use** 方向全部 — 与 grounding 保真度优化无关

---

## 附录：阅读效率建议

### 5 小时快速启动（如果今天就想动手）

按以下顺序读，每个论文只看关键章节：

```
1h: GenSeg-R1 §1+§3   → baseline 理解
1h: Faithful GRPO §1+§3 → motivation 来源
1h: GRIT §1+§3        → 极简方案证据
1h: DPO §3             → 方法论核心
1h: PRPO §1+§3        → 竞争方案理解
```

### 读完论文后需要问自己的问题

1. 这篇论文的核心 insight 是什么？
2. 它跟我的方案什么关系？（motivation / baseline / competitor / extension）
3. 我需要在实验中复现它的结果吗？
4. 如果被审稿人引用，我该怎么回应？

### Related Work 写作对照表（建议随读随记）

| 工作 | 在你的 paper 中的角色 | 一句话定位 |
|------|---------------------|-----------|
| GenSeg-R1 | GRPO baseline | "GenSeg-R1 achieves SOTA in referring segmentation via GRPO, but suffers from faithfulness issues" |
| Faithful GRPO | 问题诊断 | "Faithful GRPO reveals 24.5% inconsistency between reasoning and grounding" |
| PRPO | 竞争方案 | "PRProposes token-level credit assignment but still relies on unstable RL" |
| MoCA | 竞争方案 | "MoCA disentangles perception vs reasoning errors but requires per-step reward" |
| DPO/SimPO | 方法基础 | "Our method builds on DPO, replacing GRPO with direct preference optimization" |
| iVGR | 对比视角 | "Unlike iVGR which internalizes grounding, we optimize faithfulness at inference" |
