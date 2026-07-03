# Multimodal PRM 阅读路径

> **方向**: Outcome-level MRM -> Step-level PRM -> Evidence-grounded multimodal process reward
> **目标**: 判断多模态 PRM 是否比普通 MRM 更有研究空间，并形成一个可执行的选题路线。

---

## 核心判断

BaseReward 之后，普通多模态 MRM 的标准 recipe 已经比较完整。继续做一个整体打分器：

```text
r(I, q, y)
```

空间相对有限。更值得看的方向是多模态 PRM：

```text
r(I, q, y_{\le t})
```

即评价每一个中间推理步骤是否仍然被图像、题目和前文支持。重点不再是“最终回答 A/B 哪个更好”，而是：

```text
这个推理过程从哪一步开始脱离视觉证据？
```

---

## 必读主线

### 1. PRM 基础：为什么要从 outcome 转向 process

#### 1.1 Let's Verify Step by Step

| 项目 | 内容 |
|---|---|
| **论文** | Let's Verify Step by Step |
| **作者** | Lightman et al. |
| **会议** | ICLR 2024 |
| **定位** | 文本 PRM 基础论文 |
| **优先级** | ★★★★★ |

**重点看：**

- ORM 和 PRM 的区别。
- step-level label 如何定义。
- first-error labeling 为什么足够有用。
- PRM 如何用于 test-time search。

**要回答的问题：**

```text
多模态 PRM 是否也可以只标到 first visual/logical error？
```

#### 1.2 Math-Shepherd

| 项目 | 内容 |
|---|---|
| **论文** | Math-Shepherd: Verify and Reinforce LLMs Step-by-step without Human Annotations |
| **会议** | ACL 2024 |
| **定位** | 无人工 step label 的 PRM 数据构造 |
| **优先级** | ★★★★☆ |

**重点看：**

- 如何用后续 rollout 的最终正确率反推当前步骤质量。
- hard label / soft label 的构造方式。
- MCTS / rollout 估计是否能迁移到多模态 VQA。

**要回答的问题：**

```text
如果多模态任务没有 step label，能不能用 outcome verifier 自动构造 PRM label？
```

---

### 2. 多模态 PRM 直接相关

#### 2.1 VisualPRM

| 项目 | 内容 |
|---|---|
| **论文** | VisualPRM: An Effective Process Reward Model for Multimodal Reasoning |
| **链接** | `https://arxiv.org/abs/2503.10291` |
| **资源** | VisualPRM400K, VisualProcessBench |
| **定位** | 当前最直接的多模态 PRM baseline |
| **优先级** | ★★★★★ |

**重点看：**

- VisualPRM400K 的自动数据构造 pipeline。
- VisualProcessBench 如何做人类 step-wise correctness 标注。
- PRM vs ORM vs Self-Consistency 的 BoN 比较。
- 是否真正检查 visual grounding，还是主要检查文本推理合理性。

**要回答的问题：**

```text
VisualPRM 已经解决了什么？它没有解决什么？
```

特别关注它的潜在缺口：

```text
step correctness 是否显式绑定视觉证据？
first-error 是否能定位到具体 object/OCR/relation？
PRM 是否能给出 correction，而不仅仅是 binary score？
```

#### 2.2 DreamPRM

| 项目 | 内容 |
|---|---|
| **论文** | DreamPRM: Domain-Reweighted Process Reward Model for Multimodal Reasoning |
| **链接** | `https://arxiv.org/abs/2505.20241` |
| **定位** | 多数据源质量不均衡下的 PRM 训练 |
| **优先级** | ★★★★☆ |

**重点看：**

- 为什么多模态 PRM 比文本 PRM 更容易遇到 domain shift。
- bi-level optimization 如何学习 domain weights。
- 它把 PRM 的瓶颈定位为数据质量和领域覆盖，而不是模型结构。

**要回答的问题：**

```text
如果我们做细粒度 VQA PRM，需要按 OCR/counting/chart/spatial 等领域重加权吗？
```

#### 2.3 GM-PRM

| 项目 | 内容 |
|---|---|
| **论文** | GM-PRM: A Generative Multimodal Process Reward Model for Multimodal Mathematical Reasoning |
| **链接** | `https://arxiv.org/abs/2508.04088` |
| **定位** | 从 binary verifier 转向 generative/corrective PRM |
| **优先级** | ★★★★☆ |

**重点看：**

- step intent / visual alignment / logical soundness 三维分析。
- first erroneous step detection。
- 生成 corrected step 的训练目标。
- Refined-BoN 如何利用 correction 改善候选解质量。

**要回答的问题：**

```text
PRM 是否应该只打分，还是应该指出错误并生成修正步骤？
```

---

### 3. 多模态 step-level RL / reward

#### 3.1 R1VL

| 项目 | 内容 |
|---|---|
| **文件** | `research-wiki/papers/ToRead/r1vl_iccv2025.pdf` |
| **定位** | StepGRPO，多模态 step reward 训练策略 |
| **优先级** | ★★★★★ |

**重点看：**

- StepRAR 和 StepRVR 如何定义。
- 它的 step reward 是否依赖固定 CoT 模板。
- step-level reward 如何进入 GRPO。

**要回答的问题：**

```text
step reward 是训练时辅助信号，还是可以独立训练成一个 reusable PRM？
```

#### 3.2 Perception-R1

| 项目 | 内容 |
|---|---|
| **文件** | `research-wiki/papers/ToRead/perception_r1_iclr2026.pdf` |
| **定位** | 视觉感知 reward |
| **优先级** | ★★★★☆ |

**重点看：**

- 它如何证明标准 RLVR 不一定提升视觉感知。
- visual perception reward 如何构造。
- judge 是否能可靠检查视觉标注一致性。

**要回答的问题：**

```text
多模态 PRM 的 step label 是否必须显式包含 visual perception correctness？
```

#### 3.3 VisualRFT

| 项目 | 内容 |
|---|---|
| **文件** | `research-wiki/papers/ToRead/visualrft_iccv2025.pdf` |
| **定位** | 可验证视觉 reward |
| **优先级** | ★★★★☆ |

**重点看：**

- IoU、分类正确性、定位正确性这类 verifiable reward。
- 如何把视觉感知任务变成 RL 可用 reward。

**要回答的问题：**

```text
PRM 中哪些步骤可以用规则/工具直接验证，而不依赖 LLM judge？
```

#### 3.4 GRIT

| 项目 | 内容 |
|---|---|
| **文件** | `research-wiki/papers/ToRead/grit_neurips2025.pdf` |
| **定位** | 语言推理与 bbox grounding 交替生成 |
| **优先级** | ★★★☆☆ |

**重点看：**

- 自然语言 token 与 bbox token 如何统一生成。
- grounding 推理是否能作为 PRM 的中间状态。

---

### 4. MRM 作为对照，而不是主线

#### 4.1 BaseReward

| 项目 | 内容 |
|---|---|
| **文件** | `research-wiki/papers/ToRead/basereward_iclr2026.pdf` |
| **定位** | 普通 MRM 的强 baseline |
| **优先级** | ★★★★★ |

**重点看：**

- Naive-RM / Critic-RM / Generative RM 的比较。
- 数据、backbone、head、ensemble 的系统消融。
- 它在哪些 benchmark 上已经接近饱和。

**作用：**

```text
BaseReward 是“为什么不继续做普通 MRM”的参照系。
```

#### 4.2 R1-Reward

| 项目 | 内容 |
|---|---|
| **文件** | `research-wiki/papers/ToRead/r1reward_iclr2026.pdf` |
| **定位** | outcome-level generative judge / CoT RM |
| **优先级** | ★★★★☆ |

**重点看：**

- Consistency Reward。
- CoT judge 是否真正 grounded。
- 它与 step-level PRM 的边界。

---

### 5. Test-time search / downstream use

#### 5.1 OPD

| 项目 | 内容 |
|---|---|
| **文件** | `research-wiki/papers/ToRead/opd_rethinking_2026.pdf` |
| **定位** | teacher log-prob 作为 token-level dense reward |
| **优先级** | ★★★★☆ |

**重点看：**

- Thinking-Pattern Consistency。
- On-policy distillation 为什么长轨迹会退化。
- token-level dense reward 和 step-level PRM 的关系。

#### 5.2 DAPO / GSPO / GRPO

| 文件 | 作用 |
|---|---|
| `deepseekmath_grpo_2024.pdf` | 理解 critic-free RL 基础 |
| `dapo_neurips2025.pdf` | token-level loss、动态采样、长度惩罚 |
| `gspo_qwen_2025.pdf` | sequence-level ratio 降低 token-level 噪声 |

**阅读目标：**

```text
如果 PRM 用于训练 policy，而不是只用于 BoN，应采用哪种 RL 稳定化策略？
```

---

## 推荐阅读顺序

### 快速判断版：2 天

```text
Day 1:
  1. VisualPRM
  2. BaseReward
  3. R1VL

Day 2:
  4. GM-PRM
  5. DreamPRM
  6. Perception-R1
```

目标：判断多模态 PRM 的真实空白在哪里。

### 完整推进版：1 周

```text
Day 1: Let's Verify Step by Step + Math-Shepherd
Day 2: VisualPRM
Day 3: GM-PRM + DreamPRM
Day 4: R1VL + Perception-R1
Day 5: VisualRFT + GRIT
Day 6: BaseReward + R1-Reward
Day 7: OPD + DAPO/GSPO 选择性阅读
```

---

## 读完后必须回答的 8 个问题

1. 当前多模态 PRM 的 step label 是人标、自动构造，还是 outcome rollout 反推？
2. 现有 PRM 是否显式检查 visual grounding？
3. PRM 能否定位 first visual error，而不是只判断整步对错？
4. PRM 是否能输出 correction？
5. PRM 对不同 MLLM 的 CoT 风格是否泛化？
6. PRM-guided BoN 是否稳定超过 ORM reranking 和 self-consistency？
7. PRM 能否作为 RL reward 使用，还是只适合 test-time search？
8. 与 BaseReward 这类强 MRM 相比，PRM 的收益来自哪里？

---

## 可能的新选题

### 方向 A：Evidence-grounded Visual PRM

```text
目标：判断每一步推理是否被图像证据支持。
核心：step correctness = visual grounding + logical validity
```

适合细粒度 VQA、OCR、chart、counting、spatial relation。

### 方向 B：First-error Localization PRM

```text
目标：找出多模态推理链中第一个错误步骤。
核心：不仅判断错，还要定位错在哪里。
```

适合构建诊断 benchmark。

### 方向 C：Corrective Multimodal PRM

```text
目标：PRM 发现错误后生成修正步骤。
核心：verifier -> critic -> repair model
```

对标 GM-PRM，但可以从视觉证据维度做差异化。

### 方向 D：PRM-guided Test-time Search

```text
目标：用 PRM 指导 step-level beam/search，而不是只 rerank 完整答案。
核心：固定 base VLM，不训练 policy，只提升 inference。
```

最容易做出低成本 MVP。

