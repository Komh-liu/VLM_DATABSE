# 学习路线图：从 CLIP/ViT/QwenVL 到多模态 Grounding 研究

> **起点**: 读过 CLIP, ViT, QwenVL — 理解 VLM 内部结构（ViT → LLM）  
> **目标**: 6 个月内能独立设计并跑通一个多模态 Grounding + RL 的实验（7B 量级, 4×A100）  
> **排除**: VLA/具身/3D/检测架构改进  
> **创建日期**: 2026-06-18

---

## 总览

```
Week 1:     Grounding 表示入门 (Shikra, Ferret)           → 理解 "MLLM 怎么输出坐标"
Week 2-3:   核心张力 (VLM-R³, Argus, GRIT)                → 理解 "什么时候该做 grounding"
Week 4:     RL 基础 (DeepSeekMath/GRPO, DeepSeek-R1)      → 理解你使用的工具
Week 5-6:   ICML 2026 前沿 (MoCA, iVGR, SSL4RL, FOCUS-RL) → 理解 "2026 年大家在做什么"
Week 7-8:   研究方向收敛 + 实验环境搭建                    → 确定你的具体研究问题
Week 9+:    实验迭代                                       → 跑实验，写论文
```

**你在 Week 0 已经具备的（不需要学）**:  LLaVA 架构、BLIP-2 Q-former、QwenVL 技术报告 → 直接跳过

---

## 可以跳过的论文清单（省时间）

这些论文已经在 "阅读清单" 里，但对你的目标**不是必读**：

| 论文 | 为什么可以跳过 | 如果你有额外时间 |
|------|--------------|---------------|
| **MDETR** (NeurIPS 2021) | DETR 检测器架构，你是 MLLM 用户不需要改检测器 | 看第 3 节 Method Overview + 图 1 |
| **GLIP** (CVPR 2022) | 检测 + grounding 的工业方案，架构细节重 | 看 Introduction + 图 2 |
| **Grounding DINO** (ECCV 2024) | 同上，DINO 扩展 | 知道它存在即可 |
| **LISA** (CVPR 2024) | SEG token 是局部创新，核心 idea 一小时理解 | 看 Figure 2 + 3.1-3.2 节 |
| **OneRef** (NeurIPS 2024) | 单塔架构优化，工程改进 | 跳过 |
| **SimVG** (NeurIPS 2024) | 解耦融合效率优化 | 跳过 |
| **ReasonGrounder** (CVPR 2025) | 3D，已排除 | 跳过 |
| **Track B: DQN/PPO/SAC/CQL/IQL/DT/DPO/IPO...** | 你做 grounding + GRPO 不需要经典 RL | 后面感兴趣再补 |

---

## 阶段 1：Grounding 表示入门（Week 1）

> 目标：理解 MLLM 怎么输出 "猫在 [100,200,300,400]" 这样的坐标

### 必读: Shikra (2023) — 坐标当文本

- **PDF**: `pdfs/shikra_2023.pdf` (657 KB, 很短)
- **核心问题**: 怎么让 LLM 输出坐标？
- **核心做法**: 把 `[x1,y1,x2,y2]` 直接当文本 token 输出，不加任何特殊结构
- **需要读懂的**: 
  - 怎么把坐标数字 tokenize？（直接数字 → text tokens）
  - 训练数据怎么构造？（ReferDialogue 数据集）
  - 为什么这么简单能 work？（LLM 本身就能理解数字）
- **读完后你应该能回答**: "如果我想让 Qwen2.5-VL 输出一个物体的坐标，最少需要改什么？"
- **时间**: 1-2 天

### 必读: Ferret (ICLR 2024) — 混合区域表征

- **PDF**: `pdfs/ferret_2024.pdf` (29 MB, Apple 出品)
- **核心问题**: 坐标连续而 token 离散，怎么弥合这个 gap？
- **核心做法**: 
  - 离散化坐标 → 空间 bin grid
  - 提出 "混合区域表征"：一个区域 = `[bin_x, bin_y, bin_w, bin_h]` + 该区域的连续视觉特征
  - 任意形状（点/框/自由曲线）统一表示为离散 bin + 连续特征
- **需要读懂的**:
  - 为什么坐标不能直接当文本？（精度损失、坐标连续 vs token 离散的本质矛盾）
  - 离散 bin 的分辨率怎么选？（粒度-序列长度 trade-off）
  - Hybrid representation 的三部分：离散坐标 token + 连续视觉特征 + 文本上下文
- **读完后你应该能回答**: "Shikra 和 Ferret 表示坐标的本质区别是什么？各自适合什么场景？"
- **时间**: 2-3 天

### 选读（1 小时内）: LISA (CVPR 2024) — SEG token 机制

- **PDF**: `pdfs/lisa_2024.pdf`
- **为什么不必须**: SEG token 是局部技术创新，不改变你对 "何时 ground" 的理解
- **需要看懂的**: 
  - 怎么用一个特殊 token `<SEG>` 代替坐标输出？
  - `<SEG>` embedding 怎么 decode 成 mask？
  - 为什么说它实现了 "推理分割"？
- **时间**: 1 小时（看 Figure 2 + 3.1-3.2 节）

### 本阶段自检

```
□ 能用一句话解释 Shikra 和 Ferret 的坐标表示分别怎么工作
□ 能画出 "坐标 → LLM token" 的两条路径：直接文本 vs 离散 bin
□ 能说出为什么 Ferret 比 Shikra 更适合 "指猫的左耳朵"（细粒度）
```

---

## 阶段 2：核心张力 — "什么时候做 Grounding?" （Week 3-5）

> 目标：理解 grounding 不是一个独立任务，而是和推理深度交织的

### 必读: VLM-R³ (NeurIPS 2025) — 自适应 Grounding + RL

- **PDF**: `pdfs/vlmr3_neurips2025.pdf` (11 MB)
- **核心问题**: 模型能不能自己决定**什么时候 ground，以及 ground 什么区域**？
- **核心做法**: 
  - 把 grounding 时机视为 RL 的 action
  - 每个推理步骤，模型有三个选择：(1) 不 ground 继续推理 (2) ground 当前提到的对象 (3) 不说话直接输出答案
  - 用 GRPO 训练，奖励 = 最终答案正确性 + grounding 准确性
- **需要读懂的**:
  - 为什么 "何时 ground" 是个 RL 问题？（稀疏奖励，需要探索 trade-off）
  - 三个 action 怎么编码？action space 怎么定义？
  - GRPO 奖励怎么设计？怎么平衡 "答案对" 和 "ground 对"？
  - 自适应策略学到了什么 pattern？
- **读完后你应该能回答**: "VLM-R³ 在什么情况下选择 ground，什么情况下选择不 ground？为什么？"
- **时间**: 3-4 天（这篇是核心中的核心，连接了 grounding 和 RL）

### 必读: Argus (CVPR 2025) — Grounded CoT

- **PDF**: `pdfs/argus_cvpr2025.pdf` (22 MB)
- **核心问题**: 如果在推理链的每一步都显式 ground，会发生什么？
- **核心做法**: 
  - 在 CoT 中间插入 grounding 标注：先是 "我看到了物体 A 在位置 X" → 再基于这个 grounding 推理
  - Object-centric：以物体为中心组织推理链
- **与 VLM-R³ 的对比**: Argus 是 "始终 ground"，VLM-R³ 是 "自适应 ground"
- **需要读懂的**:
  - 为什么 object-centric？（比 "看图" 更精细，比 "看区域" 更结构化）
  - Grounded CoT 的模板长什么样？
  - 始终 ground 的好处和代价分别是什么？
- **时间**: 2-3 天

### 必读: GRIT (NeurIPS 2025) — 极少样本 Grounded 推理

- **PDF**: `pdfs/grit_neurips2025.pdf` (3.6 MB)
- **核心问题**: 最少需要多少 grounding 数据？
- **核心做法**: 
  - 仅 20 个标注样本做 grounded reasoning
  - BBox 和文字在推理链中交错出现
- **为什么重要**: 证明了 grounding 不需要大量标注 → 7B 预算友好
- **时间**: 1-2 天（核心贡献是数据效率，方法细节相对简单）

### 选读: VGent (CVPR 2026) — 模块化解耦

- **PDF**: `pdfs/vgent_cvpr2026.pdf` (10 MB)
- **核心问题**: 如果完全不耦合推理和 grounding，会怎样？
- **核心做法**: 推理模块 + 预测模块 → 两个独立组件
- **时间**: 1-2 天（作为 "解耦" 思想的终点）

### 本阶段自检

```
□ 能画出 "何时 ground" 的光谱：始终 → 自适应 → 模块化 → 解耦 → 不 ground
□ 能解释 VLM-R³ 的 action space 和 reward design
□ 能说出 Argus 和 VLM-R³ 的实验结果分别证明了什么
□ 能思考：GRIT 的 20 样本为什么 work？有什么局限性？
```

---

## 阶段 3：RL 基础 — 理解你的实验工具（Week 4）

> 目标：理解 GRPO 怎么工作 + RLVR 范式怎么来的。不需要学经典 RL（PPO/DPO 等）。

### 必读: DeepSeekMath (Shao et al., 2024) — GRPO 原始论文

- **为什么必须读**: GRPO 是你做实验的核心工具。读原始论文和看博客是不同的理解深度。
- **核心内容**:
  - Section 3 (GRPO): 只有 4 页，逐行读公式
  - 对同一个 prompt 采样 N 个输出 → 组内归一化 → advantage = (r - mean(r)) / std(r)
  - 为什么不需要 value network？（组内归一化就是 baseline）
  - KL 正则化怎么加？（直接加在 loss 里，而非 reward 里：`D_KL(π_θ || π_ref)`）
- **需要读懂的关键问题**:
  - "Group Relative" 到底是什么意思？为什么 "组内排名" 比 "绝对奖励值" 更稳定？
  - GRPO 和 PPO 的核心区别是什么？（无 value network / 无 GAE / 用组内归一化替代）
  - 为什么 GRPO 特别适合 VLM？（VLM 生成的 reward 很难用 value function 建模，组内比较更鲁棒）
- **读完后你应该能回答**: "如果我对 Qwen2.5-VL-7B 用 GRPO 训练，loss 函数长什么样？每个 batch 里发生了什么？"
- **时间**: 1-2 天

### 必读: DeepSeek-R1 (Guo et al., 2025) — RLVR 范式起源

- **为什么必须读**: ICML 2026 的 SSL4RL、RuCL、3D-RFT 全是 RLVR (RL with Verifiable Rewards) 范式。这个范式来自 R1。
- **核心内容**:
  - R1-Zero: 纯 RL 无 SFT → 涌现推理能力。为什么能 work？（rule-based reward 不会 reward hack）
  - 什么是 Rule-based Verifiable Reward？数学题答案对错、代码通过测试 → 自动校验，无需人类标注
  - 为什么 Rule-based reward 比 Neural reward model 更好？（不会过拟合、不会 reward hack、可以 scaling）
  - 对 grounding 的启示: grounding 也有可验证奖励（IoU, cIoU, F1, Recall → 全是自动可算的）
- **读完后你应该能回答**: "R1 的 rule-based reward 思路怎么迁移到 grounding？grounding 的 'rule' 是什么？"
- **时间**: 1 天（Introduction + R1-Zero + Discussion，跳过 R1 的蒸馏和 RLHF 混合部分）

### 实践: 看懂一个 GRPO 训练脚本

- **推荐看**: VERL 或 EasyR1 中 Qwen2.5-VL + GRPO 的训练 example
- **目标**: 能说出 "reward_function() 怎么写"、"advantage 怎么算"、"KL penalty 加在哪"
- **时间**: 半天

### 本阶段自检

```
□ 能手写 GRPO 的伪代码（采样 → 计算 reward → 归一化 → 更新 policy + KL 约束）
□ 能解释为什么 GRPO 不需要 value network
□ 能解释 RLVR 的 "rule-based reward" 和 grounding benchmark 的 metric 之间对应关系
□ 知道 GRPO 训练 Qwen2.5-VL-7B 需要多少 GPU（4×A100 是底线）
```

---

## 阶段 4：ICML 2026 前沿（Week 7-8）

> 目标：看清 2026 年的研究方向，找到自己的切入点

### 必读: MoCA — 感知信用分配（Spotlight）

- **链接**: [arXiv:2605.14054](https://arxiv.org/abs/2605.14054)
- **核心问题**: 区分 "看错了" vs "想错了"
- **为什么是必读**: 2026 年最受关注的 grounding 相关方法论论文
- **时间**: 2-3 天

### 必读: iVGR — Grounding 内化

- **链接**: [arXiv:2605.31096](https://arxiv.org/abs/2605.31096)
- **核心问题**: 把 grounding 能力压缩进 latent，推理时不显式输出坐标
- **为什么是必读**: 代表 grounding 表示演进的终点
- **时间**: 2-3 天

### 选读: SSL4RL — SSL 作为奖励

- **链接**: [ICML](https://icml.cc/virtual/2026/poster/60895)
- **核心问题**: 用 SSL 任务（旋转预测/拼图/对比学习）当可验证奖励
- **时间**: 1-2 天（思路简单但有效，代码开源）

### 选读: FOCUS-RL — View Alignment 信号

- **链接**: [ICML](https://icml.cc/virtual/2026/poster/64627)
- **核心问题**: VLM 自带的图文对齐能力 → 免费训练信号
- **时间**: 1 天

---

## 阶段 5：研究方向收敛（Week 9-10）

> 目标：从学习模式切换到研究模式

### 可选的开放研究方向

#### 🥇 方向 A: Grounding 信用分配（推荐度最高）
```
动机: MoCA 只区分了 "看 vs 想"，太粗糙
你的切入点:
  - 层次化信用分配: 定位错误 vs 属性错误 vs 关系错误 vs 计数错误
  - 用 Grounding 标注自动构建 credit label
  - 把信用分配和 VLM-R³ 的自适应 ground 结合
需要: Qwen2.5-VL-7B + GRPO + 标准 grounding 数据集
新颖度: 高（MoCA 只打了一束光，你是把光分开）
```

#### 🥈 方向 B: Grounding 内化的边界
```
动机: iVGR 说 "内化比显式好"，但这个结论太强了
你的切入点:
  - 什么任务必须显式 ground？（空间推理？计数？关系？）
  - 什么任务内化更好？（颜色识别？存在判断？）
  - 做 adaptive: 根据任务类型自动选择显式 vs 内化
需要: Qwen2.5-VL-7B + iVGR 的 dual-stream 框架 + 多类型 benchmark
新颖度: 高（iVGR 没有研究边界条件）
```

#### 🥉 方向 C: Grounding 驱动的幻觉检测
```
动机: 多模态幻觉本质是 "说了一个没看到的东西" = grounding failure
你的切入点:
  - 训练时: 用 grounding 一致性作为奖励 → 减少幻觉
  - 推理时: 检测 "提到的物体是否被 ground" → 自动标记可能幻觉
  - 连接 RLSF-V 的自反馈思路和 grounding 的精确性
需要: Qwen2.5-VL-7B + Grounding 模型 + 幻觉 benchmark (AMBER/POPE)
新颖度: 中高（grounding × hallucination 交叉点很少有人做）
```

### 实验环境搭建

```
硬性需求:
  - 4×A100 (80G) 或等价 GPU — 这是 7B GRPO 训练的最低配置
  - 如果不方便搞 GPU，考虑用 Qwen2.5-VL-3B 做概念验证

软件栈:
  - 训练框架: VERL / Open-R1 / EasyR1 (选一个)
  - 模型: Qwen2.5-VL-7B-Instruct (HuggingFace 直接拉)
  - Grounding 数据: RefCOCO/RefCOCO+/RefCOCOg (REC), Visual Genome (密集标注)
  - 推理 benchmark: MMBench, SEED-Bench, MMStar, MM-Vet

时间预算:
  - 搭建环境: 1 周
  - 跑通 baseline (复现 VLM-R³ 或 MoCA): 2 周
  - 实现你的改动: 2-4 周
  - 实验 + 消融: 4-6 周
  - 写论文: 3-4 周
  总计: 12-17 周 (3-4 个月可以出初稿)
```

---

## 最小必读清单（10 篇）

| # | 论文 | 为什么必读 | 时间 |
|---|------|----------|------|
| 1 | **Shikra** | 最简单的 grounding 方法，1 天看懂 | 1 天 |
| 2 | **Ferret** | 现代 grounding 表示的标准方案 | 2 天 |
| 3 | **VLM-R³** | 连接 grounding 和 RL 的桥梁论文 | 3-4 天 |
| 4 | **Argus** | Grounded CoT，理解 grounding+推理耦合 | 2 天 |
| 5 | **GRIT** | 极低数据量做 grounded reasoning，7B 友好 | 1 天 |
| 6 | **DeepSeekMath (§3)** | GRPO 原始论文，你的核心工具 | 1-2 天 |
| 7 | **DeepSeek-R1** | RLVR 范式起源，理解 rule-based reward | 1 天 |
| 8 | **MoCA** | 2026 grounding+RL 标杆 (Spotlight) | 2 天 |
| 9 | **iVGR** | Grounding 表示演进终点 | 2 天 |
| 10 | **SSL4RL** | SSL 当奖励，代码开源可直接跑 | 1 天 |

**总计: 约 16-18 天全职。** 3-4 周完成全部必读。

### 你不需要补的（已读过）
- ✅ LLaVA/LLaVA-1.5 — 已理解 VLM 架构范式
- ✅ BLIP-2 — 已理解 Q-former
- ✅ QwenVL 系列报告 — 已理解 Dynamic Resolution + M-RoPE + grounding 格式

### 你不需要现在学的

| 暂不需要 | 理由 |
|---------|------|
| MDETR/GLIP/Grounding DINO | 检测架构线，你是 MLLM 用户 |
| OneRef/SimVG/LISA | 局部创新或工程优化 |
| 所有 Track B 经典 RL | GRPO 不需要 DQN/PPO/SAC/CQL/DPO 前置 |
| 3D grounding | 已排除 |

---

## 学习节奏建议

```
Week 1:  Shikra (1d) + Ferret (2d) + 画 grounding 表示演进图
Week 2:  VLM-R³ (4d) — 慢读，逐节理解
Week 3:  Argus (2d) + GRIT (1d) + 写两者对比笔记
Week 4:  DeepSeekMath §3/GRPO (2d) + DeepSeek-R1 (1d) + VGent 粗读 (0.5d)
Week 5:  MoCA (2d) + iVGR (2d) + SSL4RL 粗读 (1d)
Week 6:  FOCUS-RL (1d) + Survey 当词典翻 + 确定研究方向
Week 7:  搭建实验环境 + 复现一篇 baseline（推荐 VLM-R³ 或 MoCA）
Week 8+: 你的实验
```

---

## 自检清单（读完所有必读论文后）

```
Grounding 表示:
  □ 能对比 Shikra, Ferret, iVGR 三者的坐标表示方式
  □ 能解释 "离散化为什么必要" 和 "离散化的问题是什么"

Grounding 时机:
  □ 能画出 grounding 时机的光谱（始终→自适应→解耦→内化）
  □ 能解释 VLM-R³ 的 action space 和 reward design
  □ 能说出 Argus 和 VLM-R³ 的矛盾在哪里

RL 工具:
  □ 能手写 GRPO 的伪代码
  □ 能解释 advantage 怎么计算，为什么不需要 value network
  □ 知道 GRPO 训练 7B VLM 需要多少 GPU 算力

前沿:
  □ 能说清 MoCA 的 credit assignment 怎么实现
  □ 能说清 iVGR 为什么 "内化比显式好"
  □ 能从 icml2026_multimodal_rl_survey.md 中挑出 3 个你感兴趣的方向
```

---

> **提示**: 每读完一篇必读论文后，写一篇简短的中文笔记（500-1000 字即可）。这个习惯会让你 2 个月后有积累感，而不是 "读了但忘了"。笔记模板见附录。
