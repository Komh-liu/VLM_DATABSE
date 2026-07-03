# Capability-Decoupled Supervision for Multi-Teacher VLM Distillation

**状态:** Go 信号确认 — 先导实验通过，进入完整实验阶段
**目标:** ICML 2027（Jan 2027 deadline，剩余 ~6.5 个月）

---

## 一、Go / No-Go 结果

### 实验设置

```
模型:   Qwen2.5-VL-3B-Instruct
数据:   A-OKVQA 80 条 + V*Bench 40 条 = 120 条
教师:   Visual LoRA r8 (V*Bench, 60 steps) + Knowledge LoRA r8 (A-OKVQA, 45 steps)
指标:   Logit-space KL gradient cosine（content / function / marker / punct / subword / space）
```

### 核心数字

```
Same teacher baseline:
  content-token gradient cosine: 1.000
  negative rate:                 0.0%         ← sanity check 通过

Different capability, same style (visual vs knowledge):
  content-token mean cosine:     0.327
  content negative rate:         30.4%        ← Problem 1 支持

Same capability, different style:
  content-token mean cosine:     0.429
  content negative rate:         25.2%        ← Problem 2 支持

Non-content KL contribution:
  different-capability setting:  54.0%        ← Problem 2 支持
  different-style setting:       52.7%
```

### Go Criteria 验证

| 条件 | 要求 | 实际 | 状态 |
|---|---|---|---|
| Cross-teacher content cosine 明显低于 same-teacher | mean < 0.5 | 0.327 vs 1.000 | ✅ |
| Negative cosine rate 显著 > 0 | > 15% | 30.4% | ✅ |
| 非内容 token 显著贡献 KL | > 50% | 54.0% | ✅ |
| 结果不是脚本 artifact | same-teacher = 1.000 | 通过 | ✅ |

**结论: GO — 进入完整实验阶段。**

---

## 二、我们发现的问题

### 2.1 Problem 1: Capability Gradient Conflict

不同能力 teacher（visual vs. knowledge）在相同 prompt/style 下，对 content token 产生显著不一致甚至相反的梯度。

```
Same teacher:       content cosine = 1.000, negative rate = 0.0%
Visual vs Knowledge: content cosine = 0.327, negative rate = 30.4%
Visual concise vs descriptive: content cosine = 0.429, negative rate = 25.2%
```

30.4% 的 content token 上，visual teacher 和 knowledge teacher 在告诉 student 同时增加和减少同一个 token 的概率。

### 2.2 Problem 2: Style Dominance in Token-Level KL

Token-level KL 将 >50% 的监督信号分配给非内容 token（功能词、标点、格式标记）。

```
内容 token KL 贡献:   46.0%
非内容 token KL 贡献:  54.0%（其中功能词 38.2%、标点 12.1%、空格 2.3%）
```

这意味着 token-level KL 的大部分优化努力花在了学习教师措辞习惯上，而非传递能力。

### 2.3 两个问题独立但叠加

- Problem 2（风格主导）在单教师 setting 中已存在 → StepOPSD 等 preference 方法部分解决
- Problem 1（梯度冲突）仅在多教师 setting 中存在 → 现有 MOPD 变体均未触及
- 当两个问题同时存在（多教师 + token-level KL），它们相互放大：风格噪声为梯度冲突提供了更多"战场"

---

## 三、解决方案

### Capability-Decoupled Supervision

1. **能力 Segment 切分:** 将 student rollout 按能力切分为 [VISUAL] / [KNOWLEDGE] / [REASON] segment
2. **教师隔离:** 每个 teacher 只监督/评价自己对应的 segment（解决 Problem 1）
3. **偏好排序替代 KL:** Segment-level preference ranking 替代 token-level distribution matching（解决 Problem 2）

```
MOPD（冲突）:
  Visual Teacher ────→ [tok1][tok2]...[tokN]  ← 全局 token-level KL
  Knowledge Teacher ─→ [tok1][tok2]...[tokN]  ← 在 MIXED token 上冲突

Ours（解耦）:
  Visual Teacher ────→ [VISUAL segment]        ← 只看视觉
  Knowledge Teacher ─→ [KNOWLEDGE segment]     ← 只看知识
  → 每个教师只在自己能力域内提供 segment-level 偏好排序
```

---

## 四、当前状态与已完成工作

### 已完成 ✅

- [x] 环境搭建（Qwen2.5-VL-3B, RTX 5080, WSL2）
- [x] Prompt-only probe（验证 pipeline 可行）
- [x] LoRA teacher 训练（visual_lora_r8 + knowledge_lora_r8）
- [x] LoRA teacher conflict probe（120 样本）
- [x] Same-teacher sanity baseline（cosine = 1.000）
- [x] Same-capability different-style 对照
- [x] Token-type KL contribution 分解
- [x] Go/No-Go 判断 → GO

### 关键限制（pilot → paper 需要升级）

```
当前 pilot:
  - 120 样本（A-OKVQA 80 + V*Bench 40）
  - 轻量 LoRA teacher（60/45 steps）
  - Logit-space gradient proxy
  - Token 分类用启发式规则
  - 单模型（Qwen2.5-VL-3B）

论文需要:
  - 扩展 benchmark（OK-VQA, InfoSeek, Encyclopedic-VQA）
  - 更强 teacher（GRPO trained, 独立能力评测）
  - LoRA 参数梯度 cosine（不只是 logit-space proxy）
  - 人工标注 token 分类验证
  - 至少 2 个 model family
```

---

## 五、ICML 2027 路线图

**Deadline:** ~2027 年 1 月中（剩余 ~6.5 个月）

### Phase 1: 实验升级（Jul-Aug 2026, ~6 周）

**1A. 训练更强 Teacher**
```
- Visual Teacher: GRPO trained on TallyQA (counting) + V*Bench (visual search)
- Knowledge Teacher: GRPO trained on OK-VQA + A-OKVQA
- 独立能力评测确认 teacher 确实有能力偏置
- 目标: 比当前 LoRA r8/60step teacher 明显更强的能力特化
```

**1B. 扩展到更多 Benchmark**
```
当前: A-OKVQA (80) + V*Bench (40) = 120 条
目标: A-OKVQA + V*Bench + OK-VQA + InfoSeek + Encyclopedic-VQA = 500+ 条
```

**1C. 参数梯度 Cosine**
```
当前: Logit-space KL gradient proxy
目标: LoRA 参数梯度 cosine（更接近真实训练中的冲突度量）
```

**1D. 实现方法 Baselines**
```
必须实现并对比:
  - Vanilla MOPD (token-level KL, 全局)
  - Content-masked KL (过滤非内容 token 的 KL)
  - Segment-isolated KL (教师只看自己 segment，但仍用 KL)
  - StepOPSD-at-VLM (单教师 preference，无 capability routing)
  - fDPO-style (2-dim single-judge segment DPO)
  - Drive-KD-AGP-at-VLM (gradient projection baseline)
  - Ours: Capability-Decoupled Supervision (segment routing + preference)
```

### Phase 2: 主实验（Sep-Oct 2026, ~8 周）

```
2A. 单能力域验证 (Counting, Visual Grounding)
  → 验证: 单教师 setting 中 segment-preference ≥ token-level KL
  → 预期: 接近或略优（风格主导被解决，但无冲突问题所以差距不大）

2B. 多能力域主实验 (OK-VQA, A-OKVQA, InfoSeek)
  → 验证: 多教师 setting 中 Ours >> Token-KL MOPD
  → 关键: Ours >> StepOPSD-at-VLM（证明多教师隔离是必要的）
  → 关键: Ours >> Drive-KD-AGP（证明 decoupling > projection）

2C. Ablation
  → 去掉 capability routing（所有 teacher 监督所有 segment via preference）
  → Segment-level KL 替代 preference
  → 单 teacher segment-preference（去掉 multi-teacher）
  → 维度数: 1 vs 2 vs 3 capability dimensions
```

### Phase 3: 理论与写作（Nov-Dec 2026, ~8 周）

**3A. 轻量理论组件（ICML 必需）**
```
Lemma 1 (Token-Level KL Decomposition):
  KL loss 分解为 content-token contribution 和 non-content contribution。
  在标准 VLM 蒸馏 setting 中，non-content contribution > 0.5（实验支持）。

Lemma 2 (Capability Gradient Divergence):
  不同能力 teacher 在 content token 上的梯度 cosine 期望
  显著低于同能力 teacher 的梯度 cosine 期望。

Proposition 1 (Capability Decoupling Reduces Conflict):
  将监督按能力段解耦后，content token 上的期望梯度 cosine
  上界提高（冲突降低）。
```

**3B. 论文写作**
```
- Intro: 双问题发现 + 三张 smoking gun Figure
- Related Work: MOPD variants + Drive-KD + IGA + StepOPSD + fDPO
- Method: Capability-Decoupled Supervision
- Experiments: Phase 1 + Phase 2 + Ablation
- Analysis: 梯度冲突的 token 级分析 + KL composition
```

**3C. 投稿**
```
Deadline: ~2027 年 1 月中
备选: 如果 ICML 被拒 → NeurIPS 2027 (May 2027)
```

---

## 六、差异化定位

### 与最接近 Prior Work 的关系

| Prior Work | 做了什么 | 我们与它的区别 |
|---|---|---|
| **Drive-KD** (2025) | VLM 多教师蒸馏中识别跨能力梯度冲突 + AGP | 我们研究 visual-knowledge 冲突（而非 perception-reasoning-planning）；我们用 supervision decoupling（而非 gradient projection） |
| **IGA** (Jun 2026) | 蒸馏中梯度冲突检测 + SVD masking | text-only 跨域（同能力）vs. VLM 跨能力；gradient masking vs. supervision decoupling |
| **StepOPSD** (May 2026) | Preference 替代 KL（单教师 text agent） | 仅解决 Problem 2（风格主导），不触及 Problem 1（多教师冲突） |
| **fDPO** (NeurIPS 2025) | VLM segment-level DPO（2 维，单偏好模型） | 单偏好模型 vs. 多教师能力特化 judges；2 维 vs. 3 维能力域 |
| **MoVE-KD** (CVPR 2025) | Multi-visual-encoder KD + MoLE 冲突缓解 | Visual encoder 蒸馏 vs. answer generation 监督解耦 |
| **CaMOPD** (May 2026) | 识别 recovery-preservation counteraction | 保留 token-level KL；training schedule 方案 vs. 监督信号类型改变 |

### 核心差异化

1. **发现驱动而非方法驱动:** 我们不是 "提出了一个新方法"，而是 "发现了两个被忽视的问题，解法是问题分析的自然推论"
2. **Token 级实证测量:** 30.4% negative cosine + 54% non-content KL — 这些具体数字在文献中不存在
3. **双问题框架:** 两个独立但叠加的问题各自需要一个解决机制 — 这是 principled design，不是 A+B+C 组合

---

## 七、风险与应对

| 风险 | 概率 | 影响 | 应对 |
|---|---|---|---|
| 参数梯度 cosine 不显著 | 低 | 高 | Logit-space proxy 已经显著；参数梯度通常是 logit 梯度的线性变换，预期保持 |
| GRPO teacher 冲突弱于 LoRA teacher | 中 | 高 | GRPO teacher 的能力偏置更强 → 预期冲突更明显。如果反而弱，分析原因 |
| 方法未显著优于 MOPD baseline | 中 | 致命 | Phase 1 先做单能力域（低风险验证），如果单能力域就不 work，调整方法 |
| Drive-KD/IGA baseline 与我们持平 | 中 | 中 | 即使持平，如果我们的方法更简单/更可解释/更低计算，仍有 contribution |
| ICML 审稿人认为无理论不够 | 中 | 中 | Lemma 1-3 提供轻量形式化；如果仍不够 → NeurIPS（对理论要求更低） |
| 时间不够完成所有实验 | 中 | 高 | ICML 被拒 → NeurIPS 2027（+4 个月缓冲）。ICML 是第一枪，不是唯一一枪 |

---

## 八、现在需要做的事（优先级排序）

### 本周启动

1. **训练 GRPO teacher（替换 LoRA teacher）**
   - Visual: TallyQA counting + V*Bench visual search
   - Knowledge: OK-VQA + A-OKVQA
   - 独立能力评测确认 teacher 能力偏置
   - 预期耗时: 2-3 天（含评测）

2. **扩展 benchmark 样本**
   - 从当前 120 条扩展到 500+ 条
   - 加入 OK-VQA, InfoSeek, Encyclopedic-VQA
   - 预期耗时: 1-2 天（数据下载 + 预处理）

### 本月完成

3. **参数梯度 cosine 测量**
   - 从 logit-space proxy 升级到 LoRA 参数梯度
   - 用 GRPO teacher 复现（并预期强化）先导实验结果
   - 预期耗时: 2-3 天

4. **实现方法 baselines**
   - Vanilla MOPD, Content-masked KL, Segment-isolated KL
   - StepOPSD-at-VLM, fDPO-style, Drive-KD-AGP-at-VLM
   - 预期耗时: 2-3 周

### 下月启动

5. **Phase 1 单能力域实验**（Counting）
6. **Phase 2 多能力域主实验**（OK-VQA, A-OKVQA, InfoSeek）
7. **开始写理论部分**（Lemma 1-3 draft）
