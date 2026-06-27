# InstructGPT 阅读笔记

> **论文**: Training Language Models to Follow Instructions with Human Feedback
> **作者**: Ouyang, Wu, Jiang, Almeida, Wainwright, Mishkin, Zhang, Agarwal, Slama, Ray, Schulman, Hilton, Kelton, Miller, Simens, Askell, Welinder, Christiano, Leike, Lowe (OpenAI)
> **会议**: NeurIPS 2022
> **链接**: https://arxiv.org/abs/2203.02155
> **PDF**: `instructgpt_neurips2022.pdf` | `InstructGPT_中文版.pdf` | `InstructGPT_双语版.pdf`

---

## 一句话论文

> 更大的模型不会自动更听话——用人类反馈微调，1.3B 的 InstructGPT 比 175B 的 GPT-3 更被人类偏好。

---

## 核心贡献

**RLHF 开山之作**，定义了 SFT → Reward Model → PPO 三阶段对齐范式。这是 ChatGPT 背后的方法论基础。

---

## 三阶段流水线

```
Stage 1: SFT               Stage 2: Reward Model         Stage 3: PPO
┌──────────────────┐       ┌──────────────────┐       ┌──────────────────┐
│ 标注者写 demo      │  →   │ 标注者排偏好对      │  →   │ RM 打分 + KL 约束  │
│ 13k prompts       │       │ 31k prompts       │       │ 33k prompts       │
│ ~5 petaflops/days │       │ pairwise ranking  │       │ ~60 petaflops/days│
└──────────────────┘       └──────────────────┘       └──────────────────┘
  学"模仿正确答案"             学"判断哪个更好"              学"产生高分回答"
```

### Stage 1: SFT（Supervised Fine-Tuning）
- **数据**：标注者手写的 (prompt, ideal response) 对，13k 条
- **方法**：标准语言模型交叉熵损失
- **局限**：只学会模仿，不理解"为什么好"——模型只见过"唯一正确答案"

### Stage 2: Reward Model（RM）
- **数据**：31k prompts，每个多个回答由标注者排序，取所有 $\binom{K}{2}$ 对做 pairwise ranking loss
- **目标**：训练一个标量打分器，作为"人类偏好"的代理（proxy）
- **Bradley-Terry 建模**：$P(y_c \succ y_r | x) = \sigma(r_\phi(x, y_c) - r_\phi(x, y_r))$
- **关键细节**：所有 pairwise comparisons 在同一 prompt 内做 loss 平均，减少过拟合
- ⚠️ **隐患**：RM 只是 proxy，不是真正的"好"——reward hacking 的根源

### Stage 3: PPO（Proximal Policy Optimization）
- **数据**：33k prompts（只用 prompt，不需人工标注）
- **Reward 修正**：$R(x, y) = r_\phi(x, y) - \beta \cdot \text{KL}(\pi_\theta \| \pi_{\text{SFT}})$
- **PPO-ptx 变体**：混合预训练损失，缓解 alignment tax
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
| Bias | 相比 GPT-3 无显著改善 |

---

## 论文自述的局限

1. **错误前提**：当 prompt 包含错误前提时，模型倾向于当作真命题处理
2. **过度推诿**：频繁说"作为 AI 语言模型我不能……"而非直接回答简单问题
3. **多约束退化**：当指令包含多个明确约束时性能下降
4. **对抗性有害**：被明确诱导时可能比 GPT-3 产生更多有害内容
5. **简单错误**：仍有低级错误

---

## 标注细节

- 标注者来自 Upwork 和 ScaleAI，经过仔细筛选
- 训练/验证/测试按 **user ID** 划分，防止信息泄露
- 标注者间一致率 ~73–77%
- 评测标注者来自 withheld set（未参与训练）

---

## 历史意义

- 这是 **ChatGPT 背后的方法论基础**
- RLHF 本身并非本文发明（此前用于风格对齐），本文的贡献是将 RLHF **系统化、规模化**地应用于指令跟随问题
- 开创了"SFT + RM + PPO"三阶段范式，至今仍是 alignment 领域的标准框架

---

## 阅读状态

⬜ 待阅读

---

## 关联论文

| 论文 | 关系 |
|------|------|
| DPO (NeurIPS 2023 Oral) | 跳过 RM + PPO，隐式 RM 替代路线 |
| GRPO (DeepSeekMath 2024) | 去掉 critic 的 PPO 变体 |
| Secrets of PPO (2024) | RM 训练实操细节、overoptimization 讨论 |
| RLHF.md | 本项目 RLHF 全链路数学推导笔记 |
| reward_model_reading.md | 以 RM 为主线的 6 篇阅读路径 |
