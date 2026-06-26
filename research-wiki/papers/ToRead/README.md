# PDF 清单

> 以下 PDF 文件不在 git 中追踪，仅此清单记录
> 全部来源：arXiv，29 篇

| # | 文件名 | 会议 | 主题 | 已阅读 | 核心内容 |
|---|--------|------|------|--------|----------|
| 1 | `basereward_iclr2026.pdf` | ICLR 2026 | BaseReward | ⬜ | 系统性地探索多模态奖励模型（MRM）的构建配方：对比 Naive-RM / Critic-RM / Generative RM 三种范式，基于 Qwen2.5-VL + 双层 reward head，在 MM-RLHF-Reward Bench 上提升 ~11%，VL-Reward Bench 提升 ~18%，超过 Claude 3.7 Sonnet 和 R1-Reward |
| 2 | `dpo_neurips2023.pdf` | NeurIPS 2023 Oral | DPO | ⬜ | 提出 Direct Preference Optimization，用简单的分类损失替代 RLHF 中的 reward model + PPO 两阶段流程。核心洞察：LM 本身即 reward model，最优策略与 reward 函数之间存在解析映射，无需显式训练 reward model、无需采样、无需大量超参调优 |
| 3 | `grit_neurips2025.pdf` | NeurIPS 2025 | GRIT | ⬜ | 让 MLLM 在推理链中交替生成自然语言与 bounding box 坐标，实现"看图说话 + 指哪打哪"的统一。提出 GRPO-GR 方法，无需推理链标注和 bbox 标签，仅需 ~20 条训练样本即可赋予模型定位推理能力 |
| 4 | `hallucination_rl_cvpr2026.pdf` | CVPR 2026 | Hallucination RL | ⬜ | 提出 Hallucination-as-Cue 框架，研究 RL 后训练中幻觉对多模态推理模型的影响。反直觉发现：即使在纯幻觉诱导设定下做 RL，仍能显著提升推理性能，有时甚至超过标准训练，揭示了 RL 探索过程本身的正面作用 |
| 5 | `mmupt_neurips2025.pdf` | NeurIPS 2025 | MMUPT | ⬜ | 提出 "先 SFT → 再 RL → 最后 Unsupervised Post-Training" 的三阶段持续改进框架。核心创新：基于 majority voting 的自奖励机制实现无监督 GRPO，无需外部标注。Qwen2.5-VL-7B 上 MathVista 从 66.3% 提升至 72.9% |
| 6 | `mssr_cvpr2026.pdf` | CVPR 2026 | MSSR | ⬜ | 提出 Multimodal Stabilized Single-Rollout 框架：只需 1 次 rollout per prompt（而非 GRPO 的多次采样）。关键创新：基于熵的自适应 advantage shaping 防止训练崩溃，在多模态单 rollout 场景下是"必需品"而非"锦上添花"。仅用 GRPO 一半的训练步数达到相同精度 |
| 7 | `noisygrpo_neurips2025.pdf` | NeurIPS 2025 | NoisyGRPO | ⬜ | 两项技术创新：(1) 对视觉输入注入高斯噪声以增强探索多样性；(2) 以噪声水平为先验、轨迹 reward 为似然的贝叶斯 advantage 估计。在小模型（Qwen2.5-VL-3B）上展现出更强的泛化和鲁棒性 |
| 8 | `perception_r1_iclr2026.pdf` | ICLR 2026 | Perception R1 | ⬜ | 指出标准 RLVR 仅优化答案正确性、无法提升视觉感知能力的问题（McNemar 检验验证）。引入视觉感知 reward：用 judge LLM 评估模型生成内容与视觉标注的一致性，仅需 1,442 条训练数据即超越需要 200K 数据的 Vision-R1 |
| 9 | `qoqmed_neurips2025.pdf` | NeurIPS 2025 Oral | QOQMed | ⬜ | 首个开放通用临床基础模型，跨医学图像 + 心电图 + 文本联合推理。提出 Domain-aware Relative Policy Optimization (DRPO)：按领域稀有度层级缩放 reward。macro-F1 超 GRPO 等 critic-free 方法 43%，IoU 超开源模型 10×，逼近 o4-mini |
| 10 | `r1reward_iclr2026.pdf` | ICLR 2026 | R1 Reward | ⬜ | 将 reward modeling 重新定义为 rule-based RL 任务，提出 StableReinforce 算法解决三类训练不稳定：(1) advantage 为负时 refined clipping；(2) 低方差 batch 的鲁棒归一化；(3) 基于 MLLM referee 的一致性 reward。VL Reward-Bench 提升 8.4%，Multimodal Reward Bench 提升 14.3% |
| 11 | `r1vl_iccv2025.pdf` | ICCV 2025 | R1VL | ⬜ | 提出 StepGRPO（Step-wise Group Relative Policy Optimization），用两个规则化 step 级 reward——StepRAR（关键步骤软匹配）和 StepRVR（推理逻辑一致性）——替代仅看最终答案的粗粒度 reward。在 8 个多模态推理 benchmark 上展现强逐步推理能力 |
| 12 | `srpo_neurips2025.pdf` | NeurIPS 2025 | SRPO | ⬜ | 提出两阶段 reflection-aware RL 框架：先 GRPO 增强推理，再引入自我反思机制。基于 Qwen2.5-VL（7B/32B），在 MathVista、MathVision、MathVerse、MMMU-Pro 上全面超越已有方法 |
| 13 | `survey_perception_reason_2025.pdf` | 2025 | Survey: Perception & Reasoning | ⬜ | 91 页大型综述，覆盖 550+ 篇论文。按四阶段发展路线组织：感知驱动模块化推理 → 语言中心短推理（MCoT）→ 语言中心长推理（MM-O1/MM-R1）→ 原生大多模态推理模型（N-LMRM），是理解整个领域的最佳入口 |
| 14 | `twostage_entropy_cvpr2026.pdf` | CVPR 2026 | Two-Stage Entropy | ⬜ | 针对真实标注含噪声的 RLVR 训练问题，提出两阶段 token 级熵优化：第一阶段熵最大化鼓励探索多样性、防止过拟合噪声标签；第二阶段熵最小化鼓励确定性输出、巩固所学。在 Qwen2-VL-2B/7B 和 Qwen2.5-VL-3B 上一致优于已有方法 |
| 15 | `visualrft_iccv2025.pdf` | ICCV 2025 | VisualRFT | ⬜ | 将 DeepSeek-R1 的 GRPO + 可验证 reward 范式扩展到视觉感知任务（细粒度分类、小样本检测、推理定位、开放词汇检测）。设计任务专属 reward 函数（如检测用 IoU）。单样本细粒度分类 +24.3%，COCO 双样本检测 +21.9% |
| 16 | `pivot_iclr2026.pdf` | ICLR 2026 | PIVOT |  ✅ | 核心发现：RL（DPO）训练不仅让 MLLM 答案更准，还能"重写"视觉编码器的内部表征。DPO 训练的视觉编码器产生比 SFT 更强、更局部化的视觉特征，以不到标准视觉预训练 1% 的计算成本超越更大型编码器。在 CLIP/MAE/DINO 三种视觉 backbone 上均验证有效，颠覆了"视觉能力主要由 LLM 继承"的假设 |
| 17 | `survey_sailing_by_stars_2025.pdf` | EMNLP 2025 Findings | Sailing by the Stars | ⬜ | 最全面的 RL 后训练综述。统一梳理 PPO → DPO → GRPO → RLVR 全链条：从 reward 来源（人类/自动）、reward model 设计（标量/批判/隐式）、学习策略（训练期 PPO/DPO/GRPO vs 推理期 Best-of-N/奖励引导解码/自修正）三个维度建立分类体系。附带持续更新的 GitHub 论文库 |
| 18 | `survey_post_training_llm_2025.pdf` | 2025 | Post-Training Survey | ⬜ | 87 页大型综述，首篇系统覆盖 LLM 后训练五大范式：(1) 微调、(2) 对齐（RLHF/DPO/RLAIF）、(3) 推理（CoT/DeepSeek-R1）、(4) 效率（模型压缩/PEFT/知识蒸馏）、(5) 多模态集成与领域适配。追溯从 ChatGPT 对齐策略到 DeepSeek-R1 推理突破的完整演化脉络 |
| 19 | `deepseekmath_grpo_2024.pdf` | 2024 | GRPO | ⬜ | **GRPO 原始论文**。提出 Group Relative Policy Optimization：去掉 critic 模型，用组内 reward 均值做 baseline 估计 advantage，大幅降低 RL 训练内存开销。同时提出 DeepSeekMath 7B（120B math tokens 预训练），MATH 51.7%。RL 后训练领域几乎所有后续工作（DAPO/GSPO/NoisyGRPO 等）均基于此 |
| 20 | `dapo_neurips2025.pdf` | NeurIPS 2025 | DAPO | ⬜ | 字节跳动 & 清华 AIR 联合发布的开源大规模 LLM RL 系统。四项关键技巧修复 GRPO：(1) Clip-Higher 防止熵崩溃；(2) 动态采样过滤低分 rollout；(3) Token 级 policy gradient loss 对长 CoT 至关重要；(4) 超长惩罚 shaping 减少 reward 噪声。Qwen2.5-32B 达 AIME 50 分，超越 DeepSeek-R1-Zero-Qwen-32B（47 分），训练步数仅其 50% |
| 21 | `gspo_qwen_2025.pdf` | 2025.07 | GSPO | ⬜ | 通义千问团队提出 Group Sequence Policy Optimization。核心修正：将 GRPO 的 token 级 importance ratio 替换为**序列级**（几何平均 + 长度归一化），从理论上修复了 GRPO 中 importance sampling 误用导致的高方差噪声和不可逆模型崩溃。MoE 训练天然稳定，无需 Routing Replay。Qwen3 系列的关键训练算法 |
| 22 | `opd_rethinking_2026.pdf` | ICML 2026 Workshop | OPD | ⬜ | 清华 THUNLP 系统性研究 On-Policy Distillation 机制。两大成功条件：(1) Thinking-Pattern Consistency——师生模型必须共享思维模式而非仅看分数；(2) New Knowledge Requirement——教师必须提供学生未见过的能力。发现 OPD 本质是 teacher log-prob 作为 token 级稠密 reward 替代 RL 的稀疏信号，但随着轨迹变长性能退化
| 23 | `instructgpt_neurips2022.pdf` | NeurIPS 2022 | InstructGPT | ⬜ | **RLHF 开山之作**。定义 SFT → Reward Model → PPO 三阶段范式，reward model 作为人类偏好代理（proxy）的核心角色由此确立。理解 reward model 在 RLHF 管线中的位置和 reward hacking 问题的根源 |
| 24 | `rewardbench_2024.pdf` | 2024 | RewardBench | ⬜ | 当前最通用的 RM 评测基准，定义了评估 reward model 的标准协议。VL-RewardBench（多模态版本）是其扩展。理解 RM 评估维度（overoptimization、偏好一致性等）的最佳入口 |
| 25 | `secrets_ppo_2024.pdf` | 2024 | Secrets of PPO | ⬜ | 实操层面深入讲解 RM 训练的细节：reward model overoptimization、数据质量影响、训练不稳定性等。StableReinforce 解决的多项问题在这里有先行讨论 |
| 26 | `vl_rewardbench_cvpr2025.pdf` | CVPR 2025 | VL-RewardBench | ⬜ | 多模态 RM 的标准化评测基准（1250 偏好对），涵盖 General（14.7%）、Hallucination Detection（59.9%）、Complex Reasoning（25.4%）三类任务。性能与下游 Best-of-N 采样的 Pearson r > 0.9 |
| 27 | `qwenvl_2023.pdf` | arXiv 2023 | Qwen-VL | ⬜ | 第一代 Qwen 视觉语言模型。Qwen-7B + ViT-bigG + 单层交叉注意力。三阶段训练（预训练→多任务→指令微调）。支持视觉定位（RefCOCO 75.9% → Chat 86.3%）、text reading（TextVQA 61.5%）和通用 VQA（VQAv2 78.8%），同规模通用模型 SOTA |
| 28 | `qwen2vl_2024.pdf` | arXiv 2024 | Qwen2-VL | ⬜ | 第二代 Qwen 视觉模型。Naive Dynamic Resolution（原生比例切图），M-RoPE（时间+高度+宽度三维位置编码），统一图像与视频处理。2B/8B/72B 三档。72B 在 ChartQA（88.2）、DocVQA（95.5）上超越 GPT-4o |
| 29 | `qwen2_5vl_2025.pdf` | arXiv 2025 | Qwen2.5-VL | ⬜ | 第三代旗舰。从零训练原生动态分辨率 ViT + Window Attention，绝对时间编码支持小时级视频定位。新增计算机/手机操控 agent 能力。文档图表理解全面超越 GPT-4o（DocVQA 96.5、ChartQA 90.1、InfoVQA 84.3）|

---

## 论文关系

```
═══════════════════════════════════════════════════════════
综述入门
  17.Sailing by the Stars (RL 后训练全景 · 最全面)
  18.Post-Training Survey (五大范式 87p · 最系统)
  13.Perception & Reasoning Survey (多模态推理 550+ 篇)
═══════════════════════════════════════════════════════════
RL 算法基础
  2.DPO (去掉 reward model，NeurIPS 2023 Oral)
    │
  19.GRPO (去掉 critic，组内归一化，DeepSeekMath 2024)  ← RL 后训练基石
    │
    ├─ 20.DAPO (四项技巧修复 GRPO，NeurIPS 2025)
    ├─ 21.GSPO (序列级 ratio 修复 token 级噪声，Qwen 2025)
    └─ 22.OPD (teacher log-prob 替代 reward，ICML 2026 Workshop)
═══════════════════════════════════════════════════════════
多模态 RL（GRPO 向视觉扩展）
  1.BaseReward ──→ 10.R1-Reward (多模态 reward model)
    │
  16.PIVOT (DPO 重塑视觉编码器表征，ICLR 2026)
    │
  3.GRIT (视觉定位推理 · 20 样本即可)
  5.MMUPT (无监督多数投票自进化)
  6.MSSR (单 rollout 替代多 rollout · 熵稳训练)
  7.NoisyGRPO (高斯噪声注入 + 贝叶斯 advantage)
  8.Perception-R1 (视觉感知 reward · 仅 1442 样本)
  9.QOQMed (领域感知 RL · 医学多模态 · NeurIPS Oral)
  11.R1VL (step 级 reward 替代答案级)
  12.SRPO (反思感知两阶段 RL)
  14.TwoStage-Entropy (探索→利用 熵调控抗噪)
  15.VisualRFT (RL 用于检测/分类/定位等视觉任务)
═══════════════════════════════════════════════════════════
分析视角
  4.Hallucination RL (幻觉在 RL 中的反直觉角色，CVPR 2026)
═══════════════════════════════════════════════════════════
Foundation（Reward Model 基础）
  23.InstructGPT (RLHF 范式起源 · reward model 角色)
  2.DPO (隐式 reward model · 理解对比 R1-Reward 的前提)
  24.RewardBench (RM 评估标准)
  25.Secrets of PPO (RM 训练实操细节)
  26.VL-RewardBench (多模态 RM 评测)
  10.R1-Reward (用 RL 训练显式 MRM)
  1.BaseReward (MRM 系统化构建指南)
```
