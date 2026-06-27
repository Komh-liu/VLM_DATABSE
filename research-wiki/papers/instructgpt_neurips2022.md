---
type: paper
node_id: paper:instructgpt_neurips2022
title: "Training Language Models to Follow Instructions with Human Feedback"
authors: ["Long Ouyang", "Jeff Wu", "Xu Jiang", "Diogo Almeida", "Carroll L. Wainwright", "Pamela Mishkin", "Chong Zhang", "Sandhini Agarwal", "Katarina Slama", "Alex Ray", "John Schulman", "Jacob Hilton", "Fraser Kelton", "Luke Miller", "Maddie Simens", "Amanda Askell", "Peter Welinder", "Paul Christiano", "Jan Leike", "Ryan Lowe"]
year: 2022
venue: "NeurIPS 2022"
external_ids:
  arxiv: "2203.02155"
  doi: null
  s2: null
short: "InstructGPT"
tags: ["RLHF", "alignment", "instruction-following", "PPO", "reward-model", "SFT"]
added: 2026-06-27T00:00:00Z
---

# InstructGPT

> **原文**: [Training Language Models to Follow Instructions with Human Feedback](https://arxiv.org/abs/2203.02155)
> **作者**: Long Ouyang et al. (OpenAI)
> **会议**: NeurIPS 2022
> **PDF**: `instructgpt_neurips2022.pdf` | `InstructGPT_中文版.pdf` | `InstructGPT_双语版.pdf`

## One-line Thesis

> 更大的模型不会自动更听话——用人类反馈微调，1.3B 的 InstructGPT 比 175B 的 GPT-3 更被人类偏好。

**RLHF 开山之作**，定义了 SFT → Reward Model → PPO 三阶段对齐范式。这是 ChatGPT 背后的方法论基础。

---

## 核心贡献

首次将 **RLHF（Reinforcement Learning from Human Feedback）** 系统化、规模化地应用于指令跟随问题：证明全量微调下，RLHF 可以让小模型（1.3B）在人类偏好上显著超越大模型（175B GPT-3）。

> RLHF 本身并非本文发明（此前用于风格对齐等任务），本文的核心贡献在于将其从零星的实验扩展为可复现的工业级 pipeline。

---

## Method

### 三阶段流水线

| Stage | 做什么 | 数据量 | 算力 | 学到了什么 |
|:---|:---|:---|:---|:---|
| **Stage 1: SFT** | 标注者手写 demo（真人） | 13k prompts | ~5 petaflops/days | 模仿正确答案 |
| **Stage 2: RM** | 标注者对多回答排序 | 31k prompts | pairwise ranking | 判断哪个回答更好 |
| **Stage 3: PPO** | RM 打分 + KL 约束 | 33k prompts | ~60 petaflops/days | 产生高分回答 |

**流程**: SFT → RM → PPO，三个阶段串行，前一阶段的输出是后一阶段的输入。

### Stage 1: SFT（Supervised Fine-Tuning）
- **数据**：标注者手写的 (prompt, ideal response) 对，13k 条
- **方法**：标准语言模型交叉熵损失，**全量微调**
- **局限**：只学会模仿，不理解"为什么好"——模型只见过"唯一正确答案"

### Stage 2: Reward Model（RM）
- **数据**：31k prompts，每个 prompt 多个回答由标注者排序，取所有 $\binom{K}{2}$ 对做 pairwise ranking loss
- **目标**：训练一个标量打分器，作为"人类偏好"的代理（proxy）
- **Bradley-Terry 建模**：$P(y_c \succ y_r \mid x) = \sigma(r_\phi(x, y_c) - r_\phi(x, y_r))$
- **关键细节**：所有 pairwise comparisons 在同一 prompt 内做 loss 平均，减少过拟合
- ⚠️ **隐患**：RM 只是 proxy，不是真正的"好"——reward hacking 的根源
- 使用6B的SFT模型去掉unembedding层，用投影层替代，得到对输入的一个打分。RM对学习率不敏感但是对训练轮次敏感，多轮训练之后很快崩塌

### Stage 3: PPO（Proximal Policy Optimization）
- **数据**：33k prompts（只用 prompt，不需人工标注），**全量微调**
- **Reward 修正**：$R(x, y) = r_\phi(x, y) - \beta \cdot \text{KL}(\pi_\theta \parallel \pi_{\text{SFT}})$
  - **注意**：KL 参照的是 SFT 模型 $\pi_{\text{SFT}}$，而非原始 GPT-3 $\pi_{\text{base}}$
- **PPO-ptx 变体**：混合预训练损失，缓解 alignment tax
  - $\mathcal{L}_{\text{PPO-ptx}} = \mathcal{L}_{\text{PPO}} + \gamma \cdot \mathcal{L}_{\text{pretrain}}$
- **Clipping**：ratio 限制在 [0.8, 1.2]，防止策略崩塌
- **最终模型**：InstructGPT = PPO-ptx

---

## 关键结果

| 指标 | 结果 |
|------|------|
| 人类偏好胜率 | 1.3B InstructGPT > 175B GPT-3（~73% 胜率） |
| TruthfulQA | GPT-3 ~21% → InstructGPT ~38%（true + informative） |
| Toxicity（RealToxicityPrompts） | 无诱导时显著降低，但被故意诱导时可能更毒 |
| 公共 NLP benchmark | 轻微退化，PPO-ptx 缓解了 alignment tax |
| 泛化 | 对训练中未见过的标注者、语言（96% 英文训练数据）、编程任务均有效 |
| Bias（社会偏见） | 相比 GPT-3 **无显著改善**——RLHF 优化 helpfulness，未校正预训练数据偏见 |

---

## 标注细节

- 标注者来自 **Upwork** 和 **ScaleAI**，均为真人，经过仔细筛选
- 训练/验证/测试按 **user ID** 划分，防止信息泄露
- 标注者间一致率约 **73–77%**
- 评测标注者来自 withheld set（未参与训练）

---

## 局限

1. **错误前提**：当 prompt 包含错误前提时，模型倾向于当作真命题处理
2. **过度推诿**：频繁说"作为 AI 语言模型我不能……"而非直接回答简单问题
3. **多约束退化**：当指令包含多个明确约束时性能下降
4. **对抗性有害**：被明确诱导时可能比 GPT-3 产生更多有害内容
5. **简单错误**：仍有低级错误

---

## 历史意义

- 这是 **ChatGPT 背后的方法论基础**
- 开创了 **SFT + RM + PPO** 三阶段范式，至今仍是 alignment 领域的标准框架
- RLHF 本身并非本文发明（此前用于风格对齐），本文贡献是将 RLHF 系统化、规模化

---

## Connections

| 论文 | 关系 |
|------|------|
| `paper:rlhf_note` | RLHF 完整数学推导笔记，基于本文的 SFT→RM→PPO 框架展开至 DPO |
| DPO (NeurIPS 2023 Oral) | 跳过 RM + PPO，用隐式 RM 替代显式奖励模型 |
| GRPO (DeepSeekMath 2024) | 去掉 critic 的 PPO 变体 |
