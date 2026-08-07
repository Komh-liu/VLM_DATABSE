# 世界知识 VQA × Multi-Teacher Distillation 阅读路线

> **当前方向**：在世界知识 / 多能力 VQA 中诊断 dense VLM 的 multi-teacher token-level KL 蒸馏问题，并验证 capability-decoupled segment preference 是否比 KL 修补版更合理。

---

## 0. 当前论文问题定位

我们不是冲任务 SOTA，而是研究一个训练目标问题：

```text
Multi-teacher token-level KL distillation
是否真的等价于 multi-capability transfer？
```

现阶段假设：

```text
Problem 1: Capability Gradient Conflict
不同能力 teacher 在同一内容 token 上可能给出相反或低一致性的 KL 梯度。

Problem 2: Style-Dominant Supervision
token-level KL 大量惩罚 teacher 风格、格式和功能词，而不是能力正确性。
```

目标实验场景：

```text
世界知识 VQA / knowledge-intensive VQA
= 视觉识别 + 外部知识 + 推理连接
```

这类任务天然适合暴露 visual teacher 与 knowledge teacher 的监督冲突。

---

## 1. 必读主线 A：OPD / MOPD / Token-Level Distillation

这条线回答：

```text
现有 on-policy distillation 怎么做？
它们为什么依赖 token-level dense supervision？
它们有没有已经发现 token/KL 的问题？
```

### 1.1 MOPD: Multi-Teacher On-Policy Distillation

| 项目 | 内容 |
|---|---|
| 论文 | MOPD: Multi-Teacher On-Policy Distillation for Capability Integration in LLM Post-Training |
| 链接 | https://arxiv.org/abs/2606.30406 |
| 优先级 | ★★★★★ |
| 作用 | 直接靶子；定义 multi-teacher OPD/MOPD 叙事 |

**重点看：**

- teacher 是如何按 domain/capability 训练的。
- student rollout 上如何计算 teacher token-level feedback。
- teacher mixing / prompt routing / teacher weighting 机制。
- 是否默认所有 teacher logit/KL 对 student 都有益。
- 是否分析 teacher 间 gradient conflict。

**我们要写清楚的差异：**

```text
MOPD 解决多能力整合效率；
我们诊断 dense VLM 中 token-level multi-teacher supervision 的失效机制。
```

---

### 1.2 CaMOPD: Counteraction-Aware MOPD

| 项目 | 内容 |
|---|---|
| 论文 | Counteraction-Aware Multi-Teacher On-Policy Distillation for General Capability Recovery with Domain Preservation |
| 链接 | https://arxiv.org/abs/2605.27115 |
| 优先级 | ★★★★★ |
| 作用 | 最接近“多 teacher 梯度互相打架”的工作 |

**重点看：**

- recovery-preservation counteraction 如何定义。
- gradient coherence analysis 怎么做。
- alternating training 为什么能缓解冲突。
- 它的冲突是 general recovery vs domain preservation，不是 VLM 内视觉/知识/推理。

**我们的切入：**

```text
CaMOPD 发现跨域 recovery/preservation counteraction；
我们关注世界知识 VQA 中同一 answer 内的 capability-level token conflict。
```

---

### 1.3 Revisiting OPD

| 项目 | 内容 |
|---|---|
| 论文 | Revisiting On-Policy Distillation: Empirical Failure Modes and Simple Fixes |
| 链接 | https://arxiv.org/abs/2603.25562 |
| 优先级 | ★★★★★ |
| 作用 | 支撑 token-level OPD/KL 本身存在 failure modes |

**重点看：**

- sampled-token OPD 为什么 brittle。
- one-token signal imbalance。
- student-generated prefix 上 teacher guidance 为什么不可靠。
- special-token / tokenizer mismatch 如何扭曲训练。
- top-K local support matching 和 special-token masking。

**与我们关系：**

它证明 token-level OPD 不是天然可靠；我们进一步证明在 multi-teacher VLM 中还有 style dominance 和 capability conflict。

---

### 1.4 StepOPSD

| 项目 | 内容 |
|---|---|
| 论文 | StepOPSD: Step-Aware Online Preference Distillation for Agent Reinforcement Learning |
| 链接 | https://arxiv.org/abs/2605.27140 |
| 优先级 | ★★★★☆ |
| 作用 | 从 token-level 转向 step/segment-level credit 的最近邻 |

**重点看：**

- 如何把 trajectory 分解成 step。
- 如何把 token log-prob gap 转成 step advantage。
- 为什么 monolithic string supervision 不适合 heterogeneous trajectory。

**我们的差异：**

```text
StepOPSD 是 agent step credit；
我们是 VLM answer 内 capability segment credit。
```

---

### 1.5 OmniOPD

| 项目 | 内容 |
|---|---|
| 论文 | OmniOPD: Logit-Free On-Policy Distillation via Speculative Verification |
| 链接 | https://arxiv.org/abs/2606.01476 |
| 优先级 | ★★★★☆ |
| 作用 | 反对强依赖 token-level teacher logits，转向 chunk-level semantic verification |

**重点看：**

- 为什么直接依赖 teacher token logits 有局限。
- chunk-level semantic verification 如何替代 logit feedback。
- 是否可以作为我们 segment-level preference 的相关工作。

---

## 2. 必读主线 B：世界知识 / Knowledge-Intensive VQA Benchmarks

这条线回答：

```text
哪些 benchmark 同时需要视觉观察和外部知识？
哪些适合暴露 visual teacher 和 knowledge teacher 的冲突？
哪些可以作为主实验或诊断实验？
```

### 2.1 OK-VQA

| 项目 | 内容 |
|---|---|
| 论文 | OK-VQA: A Visual Question Answering Benchmark Requiring External Knowledge |
| 链接 | https://okvqa.allenai.org/ |
| 优先级 | ★★★★★ |
| 定位 | 世界知识 VQA 基础 benchmark |

**重点看：**

- 问题如何保证需要外部知识。
- answer annotation 和 evaluation protocol。
- 常见 failure：看得见但不知道、知道但没看对。

**用于我们的实验：**

```text
Knowledge teacher / mixed VQA evaluation
```

---

### 2.2 A-OKVQA

| 项目 | 内容 |
|---|---|
| 论文 | A-OKVQA: A Benchmark for Visual Question Answering using World Knowledge |
| 链接 | https://github.com/allenai/aokvqa |
| 优先级 | ★★★★★ |
| 定位 | 约 25K 世界知识 VQA，带 multiple-choice / direct-answer / rationales |

**重点看：**

- rationales 是否能用于 segment analysis。
- multiple-choice 与 direct-answer 的评估差异。
- commonsense vs factual world knowledge 的比例。

**用于我们的实验：**

```text
1. 训练轻量 knowledge LoRA teacher
2. mixed visual+knowledge rollout
3. rationale 辅助 segment labeling / error analysis
```

---

### 2.3 Encyclopedic-VQA

| 项目 | 内容 |
|---|---|
| 论文 | Encyclopedic VQA: Visual questions about detailed properties of fine-grained categories and instances |
| 链接 | https://arxiv.org/html/2306.09224v1 |
| 优先级 | ★★★★★ |
| 定位 | 大规模细粒度实体/百科知识 VQA，含 Wikipedia evidence |

**重点看：**

- 221K QA、1M image-QA samples 的构造。
- controlled knowledge base / Wikipedia evidence 如何标注。
- 视觉实体识别和百科属性查询如何耦合。

**为什么重要：**

它比 OK-VQA 更适合我们的主叙事：

```text
视觉实体识别 + 外部百科知识 + 推理回答
```

这正是 visual teacher 和 knowledge teacher 容易冲突的场景。

---

### 2.4 InfoSeek / OVEN-style Entity VQA

| 项目 | 内容 |
|---|---|
| 论文/项目 | InfoSeek / Open-domain Visual Entity Recognition and knowledge-seeking QA |
| 链接 | https://github.com/open-vision-language/infoseek |
| 优先级 | ★★★★★ |
| 定位 | 实体链接 + Wikidata/Wikipedia 知识密集 VQA |

**重点看：**

- question type taxonomy。
- image entity grounding 与 knowledge answer 的关系。
- 是否能筛出“不能只靠图像 / 不能只靠知识”的 mixed samples。

**用于我们的实验：**

```text
主诊断集：筛 100-500 个 visual+knowledge mixed questions
```

本地注意：

```text
仓库已有部分 InfoSeek 标注/网页资源；
完整图像映射和数据管线仍需补齐。
```

---

### 2.5 MIRAGE

| 项目 | 内容 |
|---|---|
| 论文 | MIRAGE: A Benchmark for Multimodal Information-Seeking and Grounded Reasoning |
| 链接 | https://mirage-benchmark.github.io/ |
| 优先级 | ★★★☆☆ |
| 定位 | 真实专家咨询式 multimodal information seeking |

**重点看：**

- natural user query + image context + expert response。
- grounded reasoning、clarification、long-form answer。
- 稀有生物实体 / 专家知识场景。

**为什么看：**

它不是最适合第一轮训练，但适合作为“世界知识 VQA / 信息寻求型任务正在变难”的背景。

---

## 3. 必读主线 C：Knowledge-Augmented / RAG VQA Methods

这条线回答：

```text
知识型 VQA 中，视觉能力、检索能力、知识推理能力如何拆分？
这些拆分可以怎样映射到我们的 capability teachers？
```

### 3.1 EchoSight

| 项目 | 内容 |
|---|---|
| 论文 | EchoSight: Advancing Visual-Language Models with Wiki Knowledge |
| 链接 | https://aclanthology.org/2024.findings-emnlp.83.pdf |
| 优先级 | ★★★★☆ |
| 定位 | 用 Wiki knowledge 增强 VLM 的 KB-VQA 方法 |

**重点看：**

- 如何注入 wiki knowledge。
- 对 OK-VQA / A-OKVQA / entity-centric VQA 的提升来自哪里。
- 视觉识别错误和知识检索错误如何区分。

---

### 3.2 MI-RAG / Multimodal Iterative RAG

| 项目 | 内容 |
|---|---|
| 论文 | Multimodal Iterative RAG for Knowledge-Intensive Visual Question Answering |
| 链接 | https://openreview.net/forum?id=NCcx1qoh26 |
| 优先级 | ★★★★☆ |
| 定位 | 在 Encyclopedic VQA / InfoSeek / OK-VQA 上做 iterative retrieval + reasoning |

**重点看：**

- 多轮 retrieval 如何改善 KB-VQA。
- 检索证据与图像证据如何交互。
- 哪些错误来自 visual recognition，哪些来自 retrieval/knowledge。

**与我们关系：**

它可以帮助定义：

```text
[VISUAL] entity recognition / image evidence
[KNOWLEDGE] retrieved facts
[REASON] evidence integration
```

---

### 3.3 Multimodal Reranking for Knowledge-Intensive VQA

| 项目 | 内容 |
|---|---|
| 论文 | Multimodal Reranking for Knowledge-Intensive Visual Question Answering |
| 链接 | https://www.haoyangwen.com/pubs/acl2024avlr-multimodal-reranking.pdf |
| 优先级 | ★★★☆☆ |
| 定位 | 提升知识候选排序质量 |

**重点看：**

- knowledge candidate ranking 如何结合图像和问题。
- reranker 与 answer generator 的 train-test discrepancy。
- 这类 reranker 是否可作为 knowledge teacher / verifier。

---

## 4. 必读主线 D：Preference / Reward / PRM Baseline

这条线回答：

```text
如果我们抛弃 token-level KL，相关的 preference / reward baseline 是什么？
```

### 4.1 DPO

| 项目 | 内容 |
|---|---|
| 论文 | Direct Preference Optimization |
| 文件 | `research-wiki/papers/ToRead/dpo_neurips2023.pdf` |
| 优先级 | ★★★★☆ |
| 作用 | segment preference loss 的基础 |

**重点看：**

- DPO loss 如何从 preference pair 直接优化 policy。
- reference model 的作用。
- 我们是否需要 capability-specific reference。

---

### 4.2 VisualPRM / VisualProcessBench

| 项目 | 内容 |
|---|---|
| 论文 | VisualPRM: An Effective Process Reward Model for Multimodal Reasoning |
| 链接 | https://arxiv.org/abs/2503.10291 |
| 优先级 | ★★★★☆ |
| 作用 | 多模态 step-level correctness baseline |

**重点看：**

- step label 如何构造。
- PRM vs ORM vs self-consistency。
- 是否真正 grounding 到图像证据。

---

### 4.3 VL-RewardBench / BaseReward / R1-Reward

| 论文 | 文件/链接 | 优先级 | 作用 |
|---|---|---|---|
| VL-RewardBench | `research-wiki/papers/Already/vl_rewardbench_cvpr2025.pdf` | ★★★★☆ | 多模态 reward model 评估 |
| BaseReward | `research-wiki/papers/ToRead/basereward_iclr2026.pdf` | ★★★★☆ | 强 outcome MRM baseline |
| R1-Reward | `research-wiki/papers/Already/r1reward_iclr2026.pdf` | ★★★☆☆ | generative reward / CoT judge 对照 |

**重点看：**

- reward model 是否能区分 hallucination / reasoning / general。
- outcome-level reward 与 process/segment-level reward 的边界。
- 是否能作为 teacher ranking / reranking baseline。

---

## 5. 推荐阅读顺序

### 2 天快速版

```text
Day 1:
  1. MOPD
  2. CaMOPD
  3. Revisiting OPD
  4. RLCSD
  5. StepOPSD

Day 2:
  6. OK-VQA
  7. A-OKVQA
  8. Encyclopedic-VQA
  9. InfoSeek
```

目标：

```text
确认已发表文献给出的正式 novelty 边界；
把未发表 OPD/MOPD preprint 当作趋势和风险观察，而不是正式 novelty attack；
确认哪些 KB-VQA benchmark 最适合暴露 visual/knowledge conflict。
```

### 1 周完整版

```text
Day 1: MOPD + CaMOPD + Drive-KD（若未发表，仅作 recent/concurrent work）
Day 2: Revisiting OPD + RLCSD + StepOPSD + OmniOPD
Day 3: OK-VQA + A-OKVQA
Day 4: Encyclopedic-VQA + InfoSeek + EchoSight + ReAG
Day 5: PCGrad + CAGrad + GradVac
Day 6: MoVE-KD + AMMKD
Day 7: DPO + Let's Verify Step by Step + Math-Shepherd
```

---

## 6. 读完必须回答的问题

### 机制问题

1. MOPD 是否默认 token-level dense teacher feedback 总是正收益？
2. CaMOPD 的 counteraction 与我们的 capability conflict 是否本质不同？
3. Revisiting OPD 的 special-token masking / top-K matching 是否能解释我们的 style dominance？
4. RLCSD 的 contrastive 去风格是否能迁移到 multi-teacher VLM？
5. StepOPSD / OmniOPD 是否已经足够解决 token-level KL 的问题？
6. 哪些结论来自已发表论文，哪些只是未发表 preprint 的参考？

### Benchmark 问题

7. 哪些 benchmark 的样本必须同时依赖图像和外部知识？
8. 哪些 benchmark 有 evidence / rationale / entity link，可用于自动 segment 或 verifier？
9. OK-VQA / A-OKVQA / InfoSeek / Encyclopedic-VQA 中，哪个最适合主实验，哪个适合诊断实验？

### 方法问题

10. Masked KL 能缓解多少？
11. Segment-isolated KL 能缓解多少？
12. Contrastive capability signal 是否比 masking-only 更能保留冲突关键 token 的学习？
13. 什么时候必须抛弃 KL，转向 segment preference？
14. capability-specific preference 是否比 generic full-answer DPO 更好？

---

## 7. 当前实验路线与论文 baseline

### 诊断实验

```text
1. same teacher sanity baseline
2. same capability + different style
3. different capability + same style
4. different capability + different style
```

核心指标：

```text
content-token gradient cosine
negative cosine rate
cos < 0.2 rate
KL contribution: function/style tokens vs content tokens
```

### 方法 baseline

```text
Base / SFT only
Single-teacher OPD
Vanilla MOPD
Content-Masked KL
Segment-Isolated KL
Content + Segment Masked KL
Full-answer DPO
Generic segment preference
Ours: capability-specific segment preference
```

### 主实验 benchmark 候选

优先：

```text
A-OKVQA
OK-VQA
InfoSeek
Encyclopedic-VQA
V*Bench / visual grounding subset
```

扩展：

```text
MIRAGE
WebQA / ViQuAE / other KB-VQA
```

---

## 8. 当前判断

最值得优先读和优先做的组合是：

```text
MOPD + CaMOPD + Revisiting OPD
        ↓
OK-VQA + A-OKVQA + InfoSeek + Encyclopedic-VQA
        ↓
Masked/Segment KL baselines
        ↓
Capability-specific segment preference
```

如果后续实验能证明：

```text
1. content-token 上确实有 capability gradient conflict；
2. function/style token 对 KL loss 贡献过大；
3. masked/segment KL 只能缓解，不能彻底解决；
4. capability-specific segment preference 在相同 budget 下更好；
```

这个方向就具备冲 ICML / NeurIPS 的完整叙事。
