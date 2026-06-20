# ICML 2026 多模态 RL 前沿调研（精简版：排除 VLA/具身）

> **调研日期**: 2026-06-18  
> **会议**: ICML 2026, 2026年7月6–11日, 韩国首尔 COEX  
> **规模**: 23,918 投稿 → 6,352 录用 (26.6%) | ~168 Oral (0.7%) | ~536 Spotlight (2.24%)  
> **过滤条件**: 排除所有 VLA/具身/3D/机器人论文，仅保留与轨道 A (Visual Grounding) 和轨道 B (RL → Preference Learning) 相关的方法论工作  
> **目标读者**: 7B 量级预算、6 个月快速产出、寻找开放问题的入门研究者

---

## 目录

1. [过滤后的核心趋势](#过滤后的核心趋势)
2. [Oral & Spotlight 高影响力论文](#oral--spotlight)
3. [Poster 论文（按主题）](#poster-论文)
4. [新手机会评估](#新手机会评估)
5. [推荐入手方向 Top 5](#推荐入手方向)
6. [与你的双轨路线的关联](#与你的双轨路线的关联)

---

## 过滤后的核心趋势

排除 VLA/具身后，ICML 2026 多模态 RL 真正与你相关的趋势有 **四条**：

| # | 趋势 | 一句话 | 拥挤度 | 新手友好度 |
|---|------|--------|--------|-----------|
| 1 | **Reward Design 精细化** | 终局奖励 → 过程奖励 → 段级奖励 → 模态分解奖励 | 🔴 热但仍有空间 | ⭐⭐⭐⭐ |
| 2 | **感知 vs 推理的信用分配** | 区分 "看错了" vs "想错了"，按模态独立给奖励 | 🟡 刚刚开启 | ⭐⭐⭐⭐⭐ |
| 3 | **Grounding 内化** | RL 把视觉定位能力吸收进推理链，推理时不显式输出 | 🟢 几乎空白 | ⭐⭐⭐⭐⭐ |
| 4 | **无人工标注的奖励信号** | SSL/互信息/自反馈替代人类偏好或 AI 评判 | 🟡 方法多元 | ⭐⭐⭐⭐ |

被**排除**的热门方向（与你无关）：
- ❌ VLA + RL 后训练（From Pixels to Tokens, BehaviorVLA, AVA-VLA, VLA-MBPO）
- ❌ World Model + 机器人（DreamDojo, TimeRewarder）
- ❌ 具身探索（GLANCE, EARL, TIC-VLA）
- ❌ 3D 场景理解（3D-RFT）

---

## Oral & Spotlight

> 过滤后仅剩与多模态 RL 方法论直接相关的 Oral/Spotlight

### 🏆 VOTP: Video-Based Optimal Transport for Feedback-Efficient Offline PbRL (Oral)

- **作者**: Tung M. Luu et al. (KAIST)
- **链接**: [arXiv:2606.16856](https://arxiv.org/abs/2606.16856) | [ICML](https://icml.cc/virtual/2026/poster/65169)
- **核心思想**: Video Foundation Model + Optimal Transport → 半监督偏好奖励学习
  - 只需 5–10 个人类偏好标注视频 → OT 对齐生成数千伪标签
  - 对视觉干扰鲁棒
- **与你的关联**: 轨道 B (Offline PbRL → OPRL) 的直接延伸
- **⚠️ 注意**: 实验在 MuJoCo/MetaWorld/Robot，但方法通用，可迁移到非具身偏好学习场景

---

### 🌟 MoCA: Bad Seeing or Bad Thinking? Rewarding Perception for Vision-Language Reasoning (Spotlight)

- **作者**: Haozhe Wang et al.
- **链接**: [arXiv:2605.14054](https://arxiv.org/abs/2605.14054) | [ICML](https://icml.cc/virtual/2026/poster/62726)
- **核心思想**: **Modality-Aware Credit Assignment**
  - 将 VLM 生成分解为感知步骤 + 推理步骤
  - "蒙眼推理" 代理：遮住图让模型纯文本推理，如果也能答对 → 说明感知没贡献 → 不给感知奖励
  - 区分 "看错了" vs "想错了"，分别给独立奖励信号
- **为什么是金矿**: 这个方向刚刚开启。MoCA 是第一篇系统性做这件事的，远未饱和
  - 扩展方向：更细粒度的错误归因（定位错误 vs 属性错误 vs 关系错误）
  - 扩展方向：用 Grounding 标注自动构建 credit label
  - 扩展方向：credit assignment 的 theoretical justification
- **新手友好度**: ⭐⭐⭐⭐⭐ — 核心思路清晰，7B Qwen 可跑，有大量未探索的分支

---

## Poster 论文

### 主题一：RL for MLLM 推理增强 (Grounding 线)

#### iVGR: Internalizing Visually Grounded Reasoning for MLLMs with RL

- **链接**: [arXiv:2605.31096](https://arxiv.org/abs/2605.31096) | [ICML](https://icml.cc/virtual/2026/poster/63651)
- **核心思想**: 将视觉定位能力**内化**到文本推理链中
  - 双流训练 + 一致性奖励：一个流带 bbox，一个流不带，奖励两者推理结果一致
  - **关键发现**: 强制输出显式 bbox 反而损害推理性能
  - 推理时无需 bbox 输出即可精确感知（更轻量、更隐私友好）
- **为什么是金矿**: 完全空白的方向
  - 目前只有这一篇论文
  - "内化" vs "显式" 的 trade-off 完全未被理解
  - 可连接到你轨道 A 的表示演进线（坐标 token → 离散 bin → SEG token → 内化）
- **新手友好度**: ⭐⭐⭐⭐⭐ — 你的轨道 A 阅读线恰好覆盖了 grounding 表示演进的完整脉络

---

#### SAPO: Segment-Aligned Policy Optimization for Multi-Modal Reasoning

- **链接**: [ICML](https://icml.cc/virtual/2026/poster/66325)
- **核心思想**: 策略更新的基本单位从 token/序列 → **连贯推理段落**
  - 段级 MDP 抽象，与推理的自然步骤对齐
  - 训练稳定性和价值估计一致性优于 token 级和序列级方法
- **创新点**: Segment-level RL
- **拥挤度**: 🟡 方法学上有空间（如何自动检测段边界？段长度是否自适应？）
- **新手友好度**: ⭐⭐⭐

---

#### RuCL: Stratified Rubric-Based Curriculum Learning for MLLM Reasoning

- **链接**: [arXiv:2602.21628](https://arxiv.org/abs/2602.21628)
- **核心思想**: 课程学习从 "数据筛选" 转向 **"奖励权重调度"**
  - 基础感知 Rubric（视觉存在、实体抽取）→ 高级推理 Rubric（步骤连贯性、证据落地）
  - 动态权重调度：模型强了就给高阶 Rubric 更大权重
  - **Qwen2.5-VL-7B** +7.83%（7 基准平均）
- **为什么值得关注**: 7B 实验 + 思路清晰 + curriculum learning 视角新颖
- **新手友好度**: ⭐⭐⭐⭐ — 7B 可跑，方向清晰，可扩展（更细的 rubric 粒度和调度策略）

---

#### DMPO: Beyond Mode Collapse — Distribution Matching for Diverse Reasoning

- **链接**: [arXiv:2605.19461](https://arxiv.org/abs/2605.19461) | [GitHub](https://github.com/OliverLeeXZ/DMPO)
- **核心思想**: GRPO 的 reverse KL → mode collapse（只会走一条推理路径）→ DMPO 用 forward KL 保持多样性
  - MM-NP-Bench: 10 个 NP-hard 视觉推理任务
  - 9-12% 相对提升
- **为什么有趣**: 理论视角（KL 方向）→ 实践影响（推理多样性），连接 Info Theory 和 RL
- **新手友好度**: ⭐⭐⭐ — 偏理论，需要扎实的 KL/信息论基础

---

### 主题二：高效多模态 RL — 替代奖励信号

#### SSL4RL: Revisiting Self-supervised Learning as Intrinsic Reward for Visual-Language Reasoning

- **作者**: PKU, MIT, TUM, Meituan
- **链接**: [ICML](https://icml.cc/virtual/2026/poster/60895) | [GitHub](https://github.com/PKU-ML/SSL4RL)
- **核心思想**: 把 SSL 前置任务重构为**可验证的密集 RL 奖励**
  - Rotation prediction / Jigsaw puzzles / Contrastive learning / Patch position → 全可自动校验
  - **无需人类偏好、无需 AI 评判器**
  - +7.39% MMBench / +8.94% SEED-Bench (3B 模型)
  - **+39 pp 关系推理** (MMBench)
  - 发现 "金凤花原理" (Goldilocks principle): SSL 任务难度需匹配模型能力
- **为什么是金矿**: 
  - SSL 任务种类成千上万，目前只试了 4 种
  - "金凤花原理" 暗示需要自适应难度调整 → 可连接 curriculum learning
  - 代码开源，7B 直接可跑
- **新手友好度**: ⭐⭐⭐⭐⭐ — 开源代码 + 清晰扩展路径 + 7B 友好 + PKU 团队代码质量高

---

#### FOCUS-RL: Seeing is Solving — Unlocking Efficient Multimodal RL via View Alignment

- **链接**: [ICML](https://icml.cc/virtual/2026/poster/64627)
- **核心思想**: 利用 VLM 自身的**文本-视觉对齐动态**作为训练信号
  - PVA (Predictive View Accuracy): 用 view alignment 估计样本难度 → 自动课程
  - RVA (Reasoning View Accuracy): 用 CoT 中的 view alignment 反映推理质量 → 密集监督
  - **2.5×–4× 加速收敛**, +4.4 平均准确率
- **为什么有趣**: 零额外成本（只复用 VLM 已有的对齐能力），纯方法论
- **新手友好度**: ⭐⭐⭐⭐ — 方法轻量、即插即用

---

#### MIRL: See First, Reason Later — Mutual Information-Guided RL for VLMs

- **链接**: [ICML](https://icml.cc/virtual/2026/poster/63334)
- **核心思想**: 互信息作为便宜的预筛选信号
  - MI(生成的视觉描述, 输入图像) → 低 MI = 视觉理解失败 → 放弃该轨迹，不浪费采样预算
  - 仅用 10 预采样 + top-6 选择，超越完整 16 轨迹（25% 更少）
- **为什么有趣**: "先看→再想" 的阶段门控思想，与 MoCA 互补
- **新手友好度**: ⭐⭐⭐ — MI 估计需要一些技巧

---

### 主题三：GRPO 改进 (算法层)

#### Durian: Difficulty-Aware Group Normalization for Multimodal LLM Reasoning

- **链接**: [ICML](https://icml.cc/virtual/2026/poster/62082)
- **核心思想**: 解决 GRPO 在 multimodal 场景中 std-based group normalization 不稳定
  - 按难度重新分组（而非随机分组），使组内方差更合理
- **为什么有趣**: GRPO 的稳定性问题被广泛讨论，但解决方案少
- **新手友好度**: ⭐⭐⭐ — 算法改进方向，需要对 GRPO 内部机制有深入理解

---

#### AlphaGRPO: Decompositional Verifiable Reward for Multimodal Generation

- **链接**: [arXiv:2605.12495](https://arxiv.org/abs/2605.12495) | [GitHub](https://github.com/huangrh99/AlphaGRPO)
- **核心思想**: LLM 分解复杂请求 → 原子可验证问题 → 每个原子问题独立校验
  - 支持 Reasoning T2I Generation + Self-Reflective Refinement
- **为什么有趣**: "分解→校验" 的思路可迁移到 grounding
- **新手友好度**: ⭐⭐ — 偏生成方向，与你的 grounding/RL 线距离较远

---

### 主题四：安全对齐与幻觉

#### RLSF-V: Mitigating Hallucinations via Fuzzy Semantic Self-Feedback

- **链接**: [ICML](https://icml.cc/virtual/2026/poster/61964)
- **核心思想**: 从模型自身 logits 构建模糊语义评估 → **on-policy 自反馈** → 减少幻觉
  - 无需外部监督（不用 GPT-4 评判、不用人类标注）
  - AMBER >50% 幻觉率下降
- **为什么是金矿**: 多模态幻觉检测 + RL 自修正 = 开放战场
  - 可连接的 Grounding 视角：幻觉本质是 "visual grounding failure" → 用 grounding 信号做幻觉检测
- **新手友好度**: ⭐⭐⭐⭐ — 与你的 grounding 线可交叉

---

#### Meerkat-VL: Implicit Risk Safety Alignment in MLLMs

- **链接**: [ICML](https://icml.cc/virtual/2026/poster/61928)
- **核心思想**: 发现多模态安全对齐中的**隐式风险**（图片暗示但未明说的危险）
  - 规范性感知自我验证 → 密集可靠奖励
  - +16% 安全性 / +32% 隐式风险增益
- **新手友好度**: ⭐⭐ — 偏产品安全方向

---

#### GenAlign: Unified Alignment Framework via Generative Reward Model

- **链接**: [ICML](https://icml.cc/virtual/2026/poster/62209)
- **核心思想**: Rubric-based 生成式奖励模型（直接生成评分+解释，而非分类打分）
  - 在线位置去偏 → 防止模型利用奖励模型的长度/格式偏见
- **新手友好度**: ⭐⭐⭐ — 奖励模型设计方向

---

#### Robust-U1: Visual Self-Recovery + Dual Reward RL

- **核心思想**: 受损图像 → 先自恢复 → 联合原图推理
  - Flow-GRPO + SSIM(像素) + CLIP(语义) 双奖励
  - R-Bench 0.74 (vs BAGEL 0.58)
- **为什么有趣**: "self-recovery before reasoning" 是新的 reasoning pipeline
- **新手友好度**: ⭐⭐⭐ — 可扩展（更多退化类型、更细粒度的恢复评估）

---

### 主题五：多模态 DPO 修正

#### IC-VCO: In-Context Visual Contrastive Optimization

- **链接**: [arXiv:2605.31312](https://arxiv.org/abs/2605.31312)
- **核心思想**: 修正标准 DPO 在 multimodal 中的 partition function 不匹配
  - Visual Contrast Distillation (VCDist) + 对比样本编辑
- **新手友好度**: ⭐⭐ — 偏理论修正，需要深入理解 DPO 的数学推导

---

#### TUR-DPO: Topology- and Uncertainty-Aware DPO

- **链接**: [arXiv:2605.00224](https://arxiv.org/abs/2605.00224)
- **核心思想**: 不只看最终答案，还奖励**推理过程的拓扑结构**
  - 轻量推理拓扑提取 + 校准不确定性 → 不确定性加权 DPO
- **为什么有趣**: "推理结构作为偏好信号" 是新视角
- **新手友好度**: ⭐⭐⭐ — 偏方法论

---

## 新手机会评估

### 判断标准

| 维度 | 描述 |
|------|------|
| **空间饱和度** | 已发表论文数 / 未解决问题数。越低越好 |
| **7B 可复现性** | 是否用 Qwen2.5-VL-7B 级别模型验证过 |
| **入门门槛** | 需要的前置知识深度 |
| **扩展路径** | 能否在 6 个月内从 idea 走到实验 |
| **差异化空间** | 能否做出与现有工作的本质区别 |

### 各方向评分

```
方向                    拥挤度  空间  7B可跑  入门  扩展性  总分  推荐
──────────────────────────────────────────────────────────────
Grounding 内化 (iVGR系)  🟢极低 ⭐⭐⭐⭐⭐  ✅   ⭐⭐⭐  ⭐⭐⭐⭐⭐  18   🔥🔥🔥
感知信用分配 (MoCA系)    🟡低   ⭐⭐⭐⭐⭐  ✅   ⭐⭐⭐⭐ ⭐⭐⭐⭐   17   🔥🔥🔥
SSL 奖励 (SSL4RL系)     🟡低   ⭐⭐⭐⭐   ✅   ⭐⭐⭐⭐ ⭐⭐⭐⭐⭐  17   🔥🔥🔥
View Alignment (FOCUS系) 🟡低   ⭐⭐⭐⭐   ✅   ⭐⭐⭐⭐ ⭐⭐⭐⭐   16   🔥🔥
幻觉自修正 (RLSF-V系)    🟡低   ⭐⭐⭐⭐   ✅   ⭐⭐⭐  ⭐⭐⭐⭐   15   🔥🔥
Curriculum Reward (RuCL) 🟡低   ⭐⭐⭐    ✅   ⭐⭐⭐⭐ ⭐⭐⭐    13   🔥
段级 RL (SAPO系)        🟡低   ⭐⭐⭐    ✅   ⭐⭐⭐  ⭐⭐⭐    12
DPO 修正 (IC-VCO/TUR)   🔴高   ⭐⭐      ✅   ⭐⭐    ⭐⭐      8
GRPO 稳定性 (Durian)    🔴热   ⭐⭐      ✅   ⭐⭐    ⭐⭐      7
多模态生成 (AlphaGRPO)  🔴热   ⭐⭐      ❌   ⭐     ⭐⭐      5
```

---

## 推荐入手方向 Top 5

### 🥇 Grounding 内化 (iVGR 延伸)

**为什么是最好的切入点**：
- 目前**只有一篇论文**（iVGR），几乎是空白地带
- 恰好坐落在你轨道 A 阅读线的**终结点**：坐标 token → 离散 bin → SEG token → **内化**
- 你的阅读线让你天然理解 "为什么需要内化" 的动机

**半年可做的方向**：
1. **内化的理论解释**: 为什么强制显式 grounding 会损害推理？（信息瓶颈？任务干扰？）
2. **部分内化**: 哪些场景需要显式 grounding，哪些可以内化？做 adaptive 的内化策略
3. **内化程度度量**: 提出一个 metric 衡量 grounding 被吸收进推理链的程度
4. **与 vlmr3 的连接**: 模型自主决定何时 ground → 内化 = 最激进的 "不 ground"

**需要的资源**: Qwen2.5-VL-7B + 4×A100 + GRPO/PPO 训练框架 (如 VERL)

---

### 🥈 感知信用分配 (MoCA 延伸)

**为什么是第二好的切入点**：
- MoCA 是 Spotlight，说明社区认可这个方向的重要性
- 但只有一篇，大量未探索
- 和你的 grounding 背景天然交叉（grounding = perception 的最精确形式）

**半年可做的方向**：
1. **层次化信用分配**: MoCA 的 "感知 vs 推理" 太粗糙 → 细化为 "定位 vs 属性识别 vs 关系理解 vs 计数" 的层次化归因
2. **Grounding-guided credit assignment**: 用 grounding 标注自动生成 perception step 的 credit label（而非 MoCA 的盲fold方法）
3. **跨模态信用分配的统一理论**: 什么时候该 trust vision？什么时候该 trust language？

**需要的资源**: Qwen2.5-VL-7B + 4×A100, grounding 标注数据（RefCOCO/g, Visual Genome）

---

### 🥉 SSL 奖励扩展 (SSL4RL 延伸)

**为什么好**：
- 代码开源，可直接跑
- SSL 任务种类无限 → 自动发现 "最优 SSL 奖励组合" 是开放问题
- 论文中明确提到 "金凤花原理" 但没有深入

**半年可做的方向**：
1. **Dynamic SSL reward selection**: 训练过程中根据模型能力自动切换 SSL 任务
2. **SSL reward composition**: 如何组合多种 SSL 奖励？（加权和？multi-objective RL？）
3. **Grounding-specific SSL tasks**: 设计专门提升 grounding 能力的 SSL 前置任务

**需要的资源**: 复现 SSL4RL + 修改 SSL 任务池

---

### 4️⃣ View Alignment 信号 (FOCUS-RL 延伸)

**为什么好**：
- 零额外成本（只用 VLM 已有的 alignment），方法极简
- PVA/RVA 思路可移植到 grounding 场景
- "alignment as training signal" 是新的 paradigm

**半年可做的方向**：
1. **Grounding-aware View Alignment**: 把 PVA/RVA 扩展为检测 "模型是否看到了正确的区域"
2. **Multi-granularity alignment**: object-level / region-level / relation-level 的对齐信号

---

### 5️⃣ 幻觉自修正 + Grounding (RLSF-V 交叉)

**为什么好**：
- 幻觉是多模态模型的头号问题
- Grounding 是幻觉的天然克星（grounded = 减少幻觉）
- 交叉点完全未被探索

**半年可做的方向**：
1. **Grounding as hallucination detector**: 用 grounding 信号自动检测幻觉 → RL reward
2. **Self-correction through regrounding**: 检测到幻觉 → 重新 grounding → 修正输出

---

## 与你的双轨路线的关联

### 轨道 A (Visual Grounding) — 直接相关论文

| ICML 2026 论文 | 连接你的哪篇阅读 | 连接点 |
|---------------|----------------|--------|
| **iVGR** | LISA, Ferret, VLM-R³ | Grounding 表示演进的终点：从显式到内化 |
| **MoCA** (Spotlight) | VLM-R³, VGent | 何时 ground 的信用分配视角 |
| **RLSF-V** | — | 幻觉 = grounding failure 的连接 |

### 轨道 B (RL → Preference Learning) — 直接相关论文

| ICML 2026 论文 | 连接你的哪篇阅读 | 连接点 |
|---------------|----------------|--------|
| **VOTP** (Oral) | OPRL, Sim-OPRL | 视频偏好学习的少样本扩展 |
| **SSL4RL** | IPO/ΨPO | 无偏好对的奖励设计统一框架 |
| **DMPO** | DPO | Forward KL vs Reverse KL 的理论扩展 |
| **IC-VCO** | DPO | 多模态 DPO 的 partition function 修正 |

---

## 建议阅读优先级

```
必读（高影响力 + 高相关性）：
  1. MoCA (Spotlight)     — 感知信用分配，轨道 A 核心问题
  2. iVGR                  — Grounding 内化，轨道 A 演进终点
  3. SSL4RL                — SSL 奖励，轨道 B 方法论

精读（好方向 + 可操作）：
  4. FOCUS-RL              — View Alignment，极简高效
  5. RuCL                  — Curriculum Reward，7B 实验
  6. RLSF-V               — 幻觉自修正 + Grounding 交叉

泛读（了解趋势）：
  7. VOTP (Oral)          — 视频偏好学习
  8. DMPO                  — Forward KL 理论
  9. SAPO                  — 段级策略优化
 10. Durian                — GRPO 稳定性
```

---

## 数据来源

- [ICML 2026 官网](https://icml.cc/virtual/2026)
- arXiv 检索 (关键词: ICML 2026, multimodal, reinforcement learning, GRPO, visual reasoning)
- 各论文 GitHub 仓库与项目页
- 科技媒体报道 (雷锋网, BAAI, TechWalker)

> **声明**: 本调研基于公开可获取的论文信息。ICML 2026 完整程序尚未最终发布，部分论文状态可能根据最终 program 有所调整。建议 7 月会议正式召开后交叉验证。
