# 多模态 RL 顶会论文汇总（2024-2026）

> 筛选范围：NeurIPS, ICML, ICLR, CVPR, ECCV, ICCV
> 研究方向：使用 RL（GRPO/RLVR/DPO/RLHF）训练/微调多模态大模型

---

## 一、NeurIPS 2025（8 篇）

### 训练范式

| 论文 | 核心方法 | 亮点 |
|------|---------|------|
| **MM-UPT** | 无监督 GRPO 自提升 | 无需标注数据，自奖励机制。Qwen2.5-VL-7B: MathVista 66.3%→72.9% |
| **GRIT** | GRPO-GR, BBox+文字交错 | 仅需 20 样本做 grounded reasoning |
| **SRPO** | 两阶段反思感知 GRPO | Stage 1 构造高质量反思数据，Stage 2 反思奖励+GRPO |

### 训练稳定性与鲁棒性

| 论文 | 核心方法 | 亮点 |
|------|---------|------|
| **NoisyGRPO** | 视觉噪声注入 + 贝叶斯优势估计 | 提升小模型探索和泛化能力 |
| **SCS** | 自一致性采样 | 视觉扰动+轨迹重采样去噪。+7.7pp |
| **QoQ-Med** (Oral) | DRPO (域感知 GRPO) | 医学多模态基础模型，层次化奖励缩放 |

### 特定能力

| 论文 | 核心方法 | 亮点 |
|------|---------|------|
| **VideoLLM RL Tuning** | 双奖励（语义+时序）+ 方差感知数据选择 | 视频理解的 GRPO 后训练 |
| **Actial** | SFT+GRPO 激活 3D 空间推理 | Viewpoint-100K 数据集 |

---

## 二、ICLR 2026（6 篇）

### 奖励模型

| 论文 | 核心方法 | 亮点 |
|------|---------|------|
| **R1-Reward** | StableReinforce 算法 | 训练多模态奖励模型。VL Reward-Bench +8.4% |
| **BaseReward** | 系统化研究 MRM pipeline | Qwen2.5-VL backbone，多 benchmark SOTA |

### 视觉感知 × RL

| 论文 | 核心方法 | 亮点 |
|------|---------|------|
| **Perception-R1** | 视觉感知奖励 | 显式激励 MLLM 在 RLVR 中感知视觉内容。仅 1442 训练数据 |
| **PIVOT** | 偏好驱动的视觉优化 | RL 产生比 SFT 更强、更局部化的视觉表征。<1% 计算成本超越更大型编码器 |
| **Visual Jigsaw** | 自监督后训练+RLVR | 打乱→重建视觉输入的排列。提升细粒度感知、时序推理 |

### Tool Use

| 论文 | 核心方法 | 亮点 |
|------|---------|------|
| **VTool-R1** | RL 训练多模态链式思维 | Python 视觉编辑工具。ReFOCUS-TableVQA 71.7% (+10%) |

---

## 三、CVPR 2026（7 篇）

| 论文 | 核心方法 | 亮点 |
|------|---------|------|
| **MSSR** | 熵基优势塑形 | 解决单次 rollout RLVR 不稳定问题。样本高效 |
| **Hallucination in RL Post-Training** | 幻觉即线索框架 | RL 在幻觉条件下也能提升推理 |
| **Two-Stage Entropy RLVR** | 探索→利用两阶段熵优化 | GRPO 噪声鲁棒训练 |
| **EVA** | SFT+KTO+GRPO | 视频 Agent，先规划再感知 |
| **MemoryExplorer** | 多任务奖励 RL 微调 | 具身探索的长时记忆 |
| **EMO-R3** | 结构化情感思维+反思奖励 | 情感推理的 RL |
| **MiniCPM-V 4.5** | 混合 RL 策略（短/长推理模式） | 8B 超越 GPT-4o-latest |

---

## 四、ICCV 2025（3 篇）

| 论文 | 核心方法 | 亮点 |
|------|---------|------|
| **R1-VL** | StepGRPO（逐步稠密奖励） | StepRAR + StepRVR，ICCV 2025 |
| **Visual-RFT** | GRPO + 可验证奖励 | 扩展到视觉感知任务（检测、分类） |
| **Hint-GRPO** | 文本去偏 + 自适应提示 | 解决 GRPO 低数据利用和文本偏置 |

---

## 五、ICML 2026（1 篇）

| 论文 | 核心方法 | 亮点 |
|------|---------|------|
| **Distributional-Aware RL** | GRPO + Concordance Correlation Coefficient 奖励 | 长尾回归，即插即用 |

---

## 六、CVPR 2025 / ICLR 2025 / NeurIPS 2024 / ICML 2024（精选）

| 论文 | 会议 | 年份 | 核心方法 |
|------|------|------|---------|
| **DPO** (Rafailov et al.) | NeurIPS | 2023-24 | 偏好优化奠基作。直接优化偏好对，无需显式奖励模型 |
| **KTO** (Ethayarajh et al.) | ICML | 2024 | 无参考模型的对齐方法 |
| **RLHF-V** (Yu et al.) | CVPR | 2024 | 细粒度人类纠正减少 MLLM 幻觉 |
| **AMP** | NeurIPS | 2024 | 多层级偏好 MDPO + MRHal-Bench |
| **TIS-DPO** | ICLR | 2024 | Token 级重要性采样的 DPO |
| **Online vs Offline Alignment Study** | ICLR | 2025 | 离线 DPO vs 在线 RLHF vs Online-DPO 系统对比 |

---

## 七、研究方向聚类

### 🔴 GRPO 变体（最热，~15 篇）
StepGRPO, NoisyGRPO, Hint-GRPO, DRPO, GRPO-GR, Share-GRPO, R-GRPO...

**核心问题**：标准 GRPO 在 MLLM 上不稳定、低数据利用、模态失衡

### 🟡 奖励设计（~8 篇）
Visual perception reward, 双奖励（语义+时序）, 反思奖励, 稠密步级奖励, CCC-based reward

**核心问题**：MLLM 需要什么样的奖励信号？

### 🟢 训练策略（~5 篇）
无监督 RL, 两阶段（探索→利用）, 混合 RL（短链+长链）, 单次 rollout RL

**核心问题**：怎么让 GRPO 训练更稳定、更高效？

### 🔵 应用领域
医学 (QoQ-Med), 视频 (VideoLLM RL, EVA), 具身 (MemoryExplorer), 情感 (EMO-R3), 空间推理 (Actial)

---

## 八、未覆盖但相关的方向

以下方向与多模态 RL 相关但不在此汇总中（另作调研）：
- **Embodied AI / VLA**: 机器人 + MLLM + RL（RT-2, PaLM-E, Octo 等）
- **RL for Visual Generation**: 用 RL 微调扩散模型
- **RL from Human Feedback (RLHF)**: 传统 RLHF pipeline 在 MLLM 上的应用
