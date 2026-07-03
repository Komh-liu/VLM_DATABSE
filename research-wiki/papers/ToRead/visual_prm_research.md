# 视觉 PRM（Process Reward Model）研究方向调研报告

**日期**：2026-07-02
**调研范围**：2025-2026 年视觉/多模态 Process Reward Model 相关论文

---

## 一、什么是 Visual PRM？

Process Reward Model (PRM) 是给推理链中**每一步**打分的模型——判断这步对不对，而不只是看最终答案。在视觉 VQA 中，PRM 回答的是：

> "模型说的这一步（比如'我检测到了 5 个人'、'提取到了发票总金额 ￥128.00'）是否正确且基于图像证据？"

与 Outcome Reward Model (ORM，只看最终答案) 和 MOPD 的关键区别：

| | ORM | MOPD | PRM |
|---|---|---|---|
| **监督粒度** | Sequence-level（只看最终答案） | Token-level（teacher 的 log-prob） | Step-level（每一步正确性判断） |
| **信号含义** | 答案对不对 | teacher 会生成什么 token | 这一步推理对不对 |
| **能否区分错误类型** | 不能 | 不能 | **能**——可以知道是感知步骤错了还是推理步骤错了 |
| **训练数据需求** | 低（只需要最终答案 GT） | 中（需要 teacher 模型） | 高（需要 step-level correctness 标注） |

---

## 二、已有工作全景图

### 2.1 基础工作：PRM 基准和数据集

| 论文 | 时间 | 出处 | 做了什么 |
|---|---|---|---|
| **VisualPRM + VisualProcessBench** | 2025-03 | ICLR 2026 | 发布 VisualProcessBench（2,866 题，26,950 人工标注 step label）和 VisualPRM400K 训练集。VisualPRM-8B 在 4 个模型尺度上 +5.9 分。**这是视觉 PRM 方向的奠基性工作。** |
| **"What, Whether and How"** | 2025 | AAAI 2026 | 首个 "thinking with images" 范式下的 PRM 评测基准。1,206 人工标注轨迹，定义 7 种细粒度错误类型。发现当前 LVLM 做 PRM 效果很差。 |
| **ViLBench** | 2025 | EMNLP 2025 | VLM 作为 ORM/PRM 的基准测试。GPT-4o CoT 只有 27.3%，说明这很难。 |

### 2.2 方法工作：如何训练更好的 Visual PRM

| 论文 | 时间 | 出处 | 核心方法 | 关键结果 |
|---|---|---|---|---|
| **Athena-PRM** | 2025-06 | TMLR 2026 | 数据高效 PRM。用 weak-strong completer 一致性筛选 noisy label。只需 5K 样本。ORM 初始化 + 负样本上采样。 | VisualProcessBench SOTA（+3.9 F1） |
| **EVPV-PRM** | 2026-03 | arXiv (Qwen团队) | **显式视觉前提验证**。把感知不确定性与逻辑评估解耦——先验证"模型声称看到的"是否真的在图像中，再给推理步骤打分。 | 提升 Best-of-N reranking 和 step-level verification |
| **VL-PRM TTS** | 2025-09 | arXiv | 混合数据合成（MCTS + strong VLM judgment）+ perception-focused supervision。**反直觉发现**：VL-PRM 当 ORM 用有时比当 PRM 用更好。 | test-time scaling 的 insights |
| **Training Data Efficiency in Multimodal PRMs** | 2026-02 | arXiv | Balanced-Information Score (BIS)：只用 **10%** 训练数据达到全量效果。 | 数据效率 |

### 2.3 感知-推理解耦方向的 PRM

| 论文 | 时间 | 出处 | 核心方法 | 与你兴趣的重叠度 |
|---|---|---|---|---|
| **Perceval** | 2026-04 | CVPR 2026 | **感知中心 PRM。** 从 VLM 响应中提取 image-related claims，逐条与视觉证据对比，token-level 错误定位。整合到 RL 训练中做细粒度 advantage 信号。 | ★★★★★ 最相关——做 token-level 感知错误定位 + RL 集成 |
| **Journey Before Destination** | 2026 | EACL 2026 | 无需训练的框架。将推理链分解为 perception vs reasoning 步骤，用 VLM judge 做 step-level faithfulness verification。 | ★★★★☆ 感知/推理分解 + 视觉忠实度验证 |
| **PaLMR** | 2026-03 | CVPR 2026 Findings | 多模态过程对齐。感知对齐数据层 + 过程对齐优化 + 层次化 reward 融合。 | ★★★☆☆ 过程级对齐，偏训练框架 |

### 2.4 免显式 PRM 训练的方法

| 论文 | 时间 | 出处 | 核心方法 |
|---|---|---|---|
| **ProcessThinker** | 2026-04 | ICLR 2026 Workshop | **Rollout-based process reward**。不训练 PRM！对每个中间步骤采样多个续写，用经验成功率作为该步的 reward。 |
| **MRPO** | 2026-06 | arXiv | Step-aware RL。对推理链早期步骤的错误给指数级更大的惩罚。早期推理失败率从 64%→13%。 |

### 2.5 与你的任务域（Counting, Document VQA, Chart）直接相关的工作

| 论文 | 时间 | 出处 | 做了什么 | PRM 相关？ |
|---|---|---|---|---|
| **VisionReasoner** | 2026 | ICLR 2026 | 统一 RL 框架做 detection + segmentation + **counting** + **DocVQA**。Process-level reward + task router。 | 半相关——用 process reward 但不是 PRM |
| **DocThinker** | 2025 | ICCV 2025 | GRPO + 多目标 process reward（Format + Accuracy + **RoI IoU** + Rephrase）。专门针对文档/图表/细粒度理解。 | 半相关——用 multi-objective reward 但不是独立 PRM |
| **Look as You Think** | 2026 | AAAI 2026 | 统一推理和视觉证据归因，做可验证的文档 RAG。 | 低相关 |

---

## 三、竞争格局分析

### 3.1 谁在做什么？

```
                    感知-推理解耦
                        │
        ┌───────────────┼───────────────┐
        │               │               │
    EVPV-PRM        Perceval      Journey Before
    (显式验证)    (token级定位)   Destination(分解)
        
        
                    PRM 训练方法
                        │
        ┌───────────────┼───────────────┐
        │               │               │
    Athena-PRM      VL-PRM TTS     ProcessThinker
    (数据高效)     (test-time)    (免训练PRM)


                    PRM 数据/基准
                        │
        ┌───────────────┼───────────────┐
        │               │               │
  VisualProcessBench  ViLBench    AAAI "What,Whether,How"
    (ICLR 2026)     (EMNLP 2025)    (AAAI 2026)
```

### 3.2 关键的未被占据的空间

| Gap | 描述 | 为什么重要 |
|---|---|---|
| **1. 任务特化的 Visual PRM** | 现有 PRM 都是通用的（数学+科学+逻辑），没有专门针对 counting/doc/chart 的 PRM | 不同任务的错误模式不同——counting 的典型错误（漏检、重检）和文档 VQA 的典型错误（OCR 误读、区域定位错）需要不同的评判标准 |
| **2. 可自动标注的 PRM 训练数据** | 对于感知可验证任务（counting IoU、OCR 编辑距离、chart 数值），step-level correctness 可以**自动规则化标注**，不需要人工 | 这是现有 PRM 工作的最大瓶颈（VisualPRM400K 用了大量人工标注，Athena 用了 MC 估计但仍有噪声） |
| **3. 多误差类型 PRM** | 现有 PRM 只判断"对不对"，不判断"哪类错"。但 counting 的 bbox 错和 count 逻辑错需要不同的反馈 | 让 PRM 输出细粒度错误诊断（"漏检了第3个人"vs"去重逻辑有误"），对 RL 训练的指导意义完全不同 |
| **4. PRM-driven RL for Fine-Grained VQA** | VisionReasoner 和 DocThinker 用了 process reward 但不是独立 PRM。Perceval 做了 PRM+RL 但是通用场景 | 在 specific task domain 上做 PRM-guided GRPO，PRM 提供 dense step-level reward |
| **5. Verifiable PRM（可验证 PRM）** | 通过可验证感知任务自动生成高质量 PRM 训练数据 | 绕开人工标注瓶颈，且可验证 = 零噪声标注 |

---

## 四、最值得考虑的方向

### 方向 1：Task-Specialized Visual PRM for Fine-Grained VQA ★★★★★

**核心思路**：训练一个专门评判 counting、document VQA、chart QA 推理链的 PRM。

**关键优势**：对于这些任务，step-level correctness 可以**自动标注**：
- 数人头："检测到 [x1,y1,x2,y2] 处有人" → 用 GT bbox 算 IoU
- 文档 VQA："提取到文本 '￥128.00'" → 用 GT 文本算编辑距离
- 图表："Q3 数值为 15.2" → 用 GT 数值算误差

这意味着你可以在**没有人工标注**的情况下生成大量高质量的 PRM 训练数据。这是现有 PRM 工作做不到的。

**差异化**：Athena-PRM 用 MC 估计（有噪声），EVPV-PRM 用 VLM judge（可能出错），你用**规则化验证**（零噪声）。这是结构性的优势。

**研究问题**：
- 用可验证标注训练的 task-specialized PRM 是否比通用 PRM 在 fine-grained VQA 上更准确？
- PRM 能否区分不同类型错误（感知 vs 推理）并提供细粒度反馈？
- PRM-guided GRPO 是否比 outcome-only GRPO 在 counting/doc/chart 上更好？

### 方向 2：Verifiable Automatic PRM Data Generation Pipeline ★★★★☆

**核心思路**：不直接训练 PRM，而是构建一个**自动化的 PRM 数据生成 pipeline**。

```
For each VQA question with perception GT:
  1. Model generates reasoning chain with structured steps
  2. Each step is auto-labeled:
     - Perception step → rule-based verification (IoU, edit distance, count match)
     - Reasoning step → consistency check + final answer verification
  3. High-quality auto-labeled data → train PRM
```

**关键优势**：这种方法**只对感知可验证任务有效**——这恰好是你的 niche。通用 VQA 做不到这一点（"描述这张图"没有 GT），但 counting/doc/chart 可以。

**差异化**：VisualPRM400K 是人工标注的；你的是自动标注的。Athena 用 MC 估计（有噪声）；你用规则化（无噪声）。

### 方向 3：Perception-Corrective PRM for RL ★★★★☆

**核心思路**：PRM 不仅打分，还输出**修正信号**——"这一步 bbox 偏左了，应该向右移 10px"。

- Perceval (CVPR 2026) 做 token-level 错误定位但不做修正
- 你的 PRM 可以直接输出 corrective feedback → 更丰富的 RL reward 信号

**风险**：修正信号的构建比较复杂，可能需要额外的训练数据。

---

## 五、推荐：方向 1（Task-Specialized Visual PRM）

这是最干净的方向。理由：

1. **差异化清晰**：现有 PRM 都是通用的，你做 task-specialized。就像 general LLM vs domain-specific LLM 的区别。

2. **数据瓶颈被绕过**：可验证感知任务 → 自动标注 → 零人工成本的高质量 PRM 训练数据。这是你相对于 Athena/VisualPRM/EVPV 的结构性优势。

3. **与你的兴趣完美对齐**：counting、document VQA、chart QA 正是你的目标任务。

4. **已有 CVPR 2026 论文做铺垫**：Perceval 证明了 "perception-centric PRM" 这个方向有价值。你的差异化是 "task-specialized + auto-labeled + fine-grained error typing"。

5. **发表路径**：
   - 强实验结果 → NeurIPS/ICML
   - 中等结果 → CVPR/ICCV
   - 都有明确的贡献（第一个 task-specialized visual PRM + 自动标注 pipeline）

### 对应的研究框架

```
Phase 1: 自动 PRM 数据生成
  - 选 3 个任务：Counting, DocVQA, ChartQA
  - 用 base model 生成推理链（结构化步骤格式）
  - 对每个步骤自动标注：
    - 感知步骤：IoU / 编辑距离 / 数值误差 → binary correctness
    - 推理步骤：基于感知结果 + 最终答案 → binary correctness
  - 产出：~10-50K 自动标注的 step-level correctness 数据

Phase 2: 训练 Task-Specialized PRM
  - Backbone: Qwen2.5-VL-7B
  - 输入：图像 + 问题 + 部分推理链（到第 k 步）
  - 输出：第 k+1 步是否正确（+ 可选的错误类型分类）
  - 对比：通用 PRM (VisualPRM, Athena) vs 我们的 Task-Specialized PRM

Phase 3: PRM-guided GRPO
  - 用 PRM 提供 dense step-level reward
  - 对比：Outcome-only GRPO vs PRM-guided GRPO
  - 关键指标：per-step accuracy, final accuracy, PRM F1

Phase 4 (亮点): Fine-grained Error Analysis
  - PRM 可以输出错误类型（perception error / reasoning error / format error）
  - 分析不同任务的主要失败模式
  - 提供可操作的改进建议
```

---

## 六、需要关注的最新论文

| 论文 | 优先级 | 需要关注的点 |
|---|---|---|
| **VisualPRM / VisualProcessBench** (ICLR 2026) | ★★★★★ | 基准和数据集，你的 PRM 需要在这个 benchmark 上评测 |
| **Athena-PRM** (TMLR 2026) | ★★★★★ | 当前 SOTA，你的主要对比对象 |
| **EVPV-PRM** (arXiv 2603.16253) | ★★★★★ | 感知验证机制，你的 PRM 需要类似的能力 |
| **Perceval** (CVPR 2026) | ★★★★★ | 最接近的工作——perception-centric PRM |
| **VL-PRM TTS** (arXiv 2509.23250) | ★★★★☆ | Test-time scaling insights |
| **ProcessThinker** (arXiv 2606.11209) | ★★★★☆ | 免 PRM 训练的替代方案，你的 baseline |
| **PaLMR** (CVPR 2026) | ★★★☆☆ | 过程对齐 |
| **VisionReasoner** (ICLR 2026) | ★★★☆☆ | Counting+DocVQA 的 RL 方法 |
| **DocThinker** (ICCV 2025) | ★★★☆☆ | 文档任务的 process reward 设计 |
| **"What, Whether and How"** (AAAI 2026) | ★★★☆☆ | PRM 评测基准 |

---

## 七、与之前 MOPD 方向的对比

| | Domain MOPD (v3) | Task-Specialized PRM (本方向) |
|---|---|---|
| **核心机制** | 多教师 on-policy 蒸馏 | Step-level correctness 判断 |
| **方法新颖性** | 低（域迁移） | **中高**（首个 task-specialized visual PRM） |
| **竞争拥挤度** | 中（Keye-VL-2.0, CoPD） | **低**（通用 PRM 有但 task-specialized 无） |
| **数据瓶颈** | 需要训练多个 teacher | **可自动标注**（感知可验证任务的结构性优势） |
| **计算成本** | 3-4 个 teacher RL + MOPD 蒸馏 | 1 个 PRM 训练 + PRM-guided GRPO |
| **发表潜力 (NeurIPS/ICML)** | 10-20% | **25-40%** |
| **发表潜力 (CVPR/ICCV)** | 30-40% | **45-60%** |

**结论**：PRM 方向在几乎每个维度上都优于 MOPD 方向。最关键的差异化——"用可验证感知任务自动生成 PRM 训练数据"——是结构性的优势，不依赖 MOPD 框架，不与其他工作直接冲突。
