# Capability-Decoupled Supervision for Multi-Teacher VLM Distillation

**Status:** Go — pilot experiments passed.
**Target:** ICML 2027 (deadline ~Jan 2027).

---

## 核心问题

VLM 多教师蒸馏中，不同能力 teacher（Visual vs Knowledge）在 **shared content token** 上产生冲突梯度。

### 两个独立但叠加的问题

**Problem 1: Cross-Teacher Gradient Conflict**
- Visual Teacher 和 Knowledge Teacher 在相同 content token 上给出相反优化方向
- Pilot: 30.4% content token 的梯度 cosine 为负

**Problem 2: Style Dominance in Token-Level KL**
- >50% KL 信号分配给非内容 token（功能词、标点、格式）
- 单教师 setting 中已存在，但多教师 setting 中为梯度冲突提供了更多"战场"

### 2 维设置

```
Visual Teacher   ──→ 擅长 counting, visual search, spatial reasoning
Knowledge Teacher ──→ 擅长 commonsense, factual knowledge, categorization
                      ↑
              在 shared content token 上梯度反向冲突
```

---

## 差异化定位

| Prior Work | 做了什么 | 我们的区别 |
|---|---|---|
| DOPD (Jun 2026) | 单 teacher 内 visual vs language **loss** 分解 | 我们分解的是**跨 teacher 信号**，不是 loss |
| CaMOPD (May 2026) | training schedule 缓解冲突 | 我们改变**监督信号类型**（KL → segment-preference） |
| Keye-VL-2.0 (2026) | 13 teacher hand-crafted 路由 | 我们证明 **2 个 teacher 就足够产生显著冲突**，并提出 principled routing |
| StepOPSD (May 2026) | Preference 替代 KL（单教师） | 仅解决 Problem 2；我们同时解决 Problem 1 + 2 |

**核心 narrative:** 问题不在 teacher 数量——2 个 teacher 就够了。问题在 KL loss 对 shared token 的无差别监督。

---

## 梯度冲突分析维度

### 1. Token 级冲突分布

```
content token 内部细分:
  ┌─ 视觉描述 token:  color, shape, position, count
  ├─ 知识推理 token:  because, therefore, category, attribute
  └─ 混合 token:      同时承载视觉确认 + 知识推断
                        ↑ 假设: 冲突最严重
```

### 2. 样本条件分析

```
按样本类型分组:
  ┌─ 纯视觉    V*Bench counting / visual search
  ├─ 纯知识    A-OKVQA commonsense
  └─ 混合      OK-VQA (outside knowledge + image)
                  ↑ 假设: 混合样本冲突最强
                  → 说明不是 teacher 不好，是架构制造了冲突
```

### 3. 层间冲突传播

```
Layer 0-8:   cosine 较高   (底层特征共享)
Layer 9-20:  cosine 下降   (高层语义分化)
Layer 21-28: cosine 最低   (输出层分歧最大)
                                  ↑
Claim: 冲突不是 noise，是高层语义分化 —
teacher 在"理解一致，输出不同"
```

---

## 解决方案

### Capability-Decoupled Supervision

1. **Segment 切分:** Student rollout 按能力切分 → `[VISUAL]` / `[KNOWLEDGE]` segment
2. **教师隔离:** 每个 teacher 只看自己 segment（解决 Problem 1）
3. **偏好排序替代 KL:** Segment-level preference ranking 替代 token-level distribution matching（解决 Problem 2）

```
MOPD（冲突）:
  Visual Teacher ────→ [tok1][tok2]...[tokN]  ← 全局 token-level KL
  Knowledge Teacher ─→ [tok1][tok2]...[tokN]  ← 在 shared token 上冲突

Ours（解耦）:
  Visual Teacher ────→ [VISUAL segment]        ← 只看视觉
  Knowledge Teacher ─→ [KNOWLEDGE segment]     ← 只看知识
  → 每个教师只在自己能力域内提供 segment-level 偏好排序
```

---

## 实验计划

### Phase 1: 梯度冲突深度分析（~2 周）

- [ ] Token 子类标注（visual / knowledge / mixed）→ 冲突热力图
- [ ] 样本条件分组分析（纯视觉 / 纯知识 / 混合）
- [ ] LoRA 参数梯度 cosine（替代 logit-space proxy）
- [ ] 层间 cosine 传播曲线
- [ ] GRPO teacher 替换 LoRA teacher（预期冲突更显著）

### Phase 2: 方法实现（~3 周）

Baselines to implement:
- [ ] Vanilla MOPD (token-level KL, global)
- [ ] Content-masked KL
- [ ] Segment-isolated KL (teacher 隔离但仍用 KL)
- [ ] StepOPSD-at-VLM (单教师 preference)
- [ ] DOPD / VGS baseline
- [ ] **Ours:** Capability-Decoupled Supervision

### Phase 3: 主实验（~4 周）

- [ ] 单能力域验证（Counting / Visual Grounding）
- [ ] 多能力域主实验（OK-VQA, A-OKVQA, InfoSeek）
- [ ] Ablation: 去掉 routing / segment-KL 替代 preference / 单 teacher

### Phase 4: 理论与写作（~4 周）

- [ ] Lemma: Token-Level KL Decomposition
- [ ] Lemma: Capability Gradient Divergence
- [ ] Proposition: Capability Decoupling Reduces Conflict
- [ ] 论文写作 + 投稿

---

## 技术栈

- **Model:** Qwen2.5-VL-3B-Instruct
- **GPU:** RTX 5080 (WSL2)
- **Data:** A-OKVQA, V*Bench, OK-VQA, InfoSeek, Encyclopedic-VQA
- **Framework:** PyTorch + Transformers + LoRA (PEFT)

## 风险

| 风险 | 应对 |
|------|------|
| 参数梯度 cosine 不显著 | Logit-space proxy 已显著；参数梯度是线性变换，预期保持 |
| GRPO teacher 冲突弱于 LoRA | GRPO teacher 能力偏置更强，预期冲突更明显 |
| DOPD/VGS baseline 持平 | 即使持平，我们的方法更简单 + 更可解释 |
| 时间不够 | ICML → NeurIPS 2027 备选 |
