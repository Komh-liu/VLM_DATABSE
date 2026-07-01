# 多模态 PRM 细粒度 VQA 研究步骤

## 研究问题

关注点：

> 使用多模态 PRM 对细粒度 VQA 问题做稠密推理监督；同时考虑不同模型的推理过程不同，寻找统一或高度可复用的监督信号。

建议把问题收窄为：

> 多模态细粒度 VQA 中，能否把不同模型各自不同的 CoT，投影到一套统一的“可验证中间状态”上，再用 PRM 学习这些状态的质量？

核心假设：

> Fine-grained multimodal reasoning does not require model-specific chain-of-thought supervision. A reusable PRM can be learned by supervising model-agnostic visual evidence states, where each reasoning step is validated by grounding, operation correctness, and consistency with the final answer.

换句话说，不直接监督“某个模型应该怎么写推理链”，而是监督更抽象、更可复用的中间视觉证据状态。

## 统一监督信号

推荐把自然语言 CoT 转成结构化 evidence tuple：

```text
(subquestion, visual_evidence, operation, intermediate_answer, validity)
```

字段含义：

| 字段 | 含义 |
|---|---|
| `subquestion` | 当前步骤要解决的子问题 |
| `visual_evidence` | 支持该步骤的视觉证据，如 bbox、region、OCR 文本、图表元素、对象属性 |
| `operation` | 当前推理操作，如定位、计数、比较、OCR、属性识别、关系判断、数学计算 |
| `intermediate_answer` | 当前步骤得到的中间结论 |
| `validity` | 该步骤是否被图像和上下文支持 |

示例：

```text
问题: 左边第二个人手里拿的是什么？

step 1:
  subquestion: 定位“左边第二个人”
  visual_evidence: person bbox / region
  operation: grounding
  intermediate_answer: 左侧第二个人区域
  validity: valid

step 2:
  subquestion: 识别该人物手部附近物体
  visual_evidence: hand-near object region
  operation: object recognition
  intermediate_answer: umbrella
  validity: valid

step 3:
  subquestion: 将中间答案映射到最终回答
  visual_evidence: step 1 + step 2
  operation: answer synthesis
  intermediate_answer: umbrella
  validity: valid
```

这样不同模型可以写完全不同的 CoT，但都能被投影到同一套检查点上。

## 阅读路线

### 第一阶段：补完多模态 RM 主线

1. `research-wiki/papers/ToRead/r1reward_iclr2026.pdf`
   - 重点看 `Consistency Reward`。
   - 关注它如何检查“推理过程是否支持最终判断”。
   - 思考：这个 consistency reward 是否可以从 pairwise reward 扩展到 step-level PRM？

2. `research-wiki/papers/ToRead/basereward_iclr2026.pdf`
   - 重点看多模态 RM 的范式比较：Naive-RM、Critic-RM、Generative RM。
   - 关注它对 backbone、reward head、训练数据、ensemble 的系统消融。
   - 目标：决定你的 PRM 应该是 classifier-style、critic-style，还是 generative judge-style。

### 第二阶段：转向 step-level / process reward

3. `research-wiki/papers/ToRead/r1vl_iccv2025.pdf`
   - 最贴近当前问题。
   - 重点看 StepGRPO、StepRAR、StepRVR。
   - 关注它如何定义“关键步骤软匹配”和“推理逻辑一致性”。
   - 核心问题：这些 step reward 是否依赖某个模型的 CoT 风格？

4. `research-wiki/papers/ToRead/opd_rethinking_2026.pdf`
   - 重点看 teacher log-prob 作为 token-level dense reward 的解释。
   - 关注 Thinking-Pattern Consistency。
   - 这篇对应你的关键风险：不同模型推理模式不一致时，稠密监督可能无法迁移。

### 第三阶段：补视觉证据和可验证 reward

5. `research-wiki/papers/ToRead/perception_r1_iclr2026.pdf`
   - 重点看 visual perception reward。
   - 关注它如何判断生成内容是否与视觉标注一致。
   - 目标：避免 PRM 退化成只看文本的 judge。

6. `research-wiki/papers/ToRead/visualrft_iccv2025.pdf`
   - 重点看任务专属可验证 reward，如 IoU、分类正确性、定位正确性。
   - 目标：学习如何把“视觉感知步骤”变成可计算 reward。

7. `research-wiki/papers/ToRead/grit_neurips2025.pdf`
   - 重点看自然语言和 bounding box 坐标交替生成。
   - 目标：学习如何把推理链与显式视觉 grounding 绑定。

### 第四阶段：补 RL 算法稳定性

8. `research-wiki/papers/ToRead/deepseekmath_grpo_2024.pdf`
   - 理解 GRPO 的组内归一化和 critic-free advantage。

9. `research-wiki/papers/ToRead/dapo_neurips2025.pdf`
   - 重点看 token-level policy gradient loss、动态采样、长度惩罚。

10. `research-wiki/papers/ToRead/gspo_qwen_2025.pdf`
    - 重点看 sequence-level ratio 如何降低 token-level 噪声。

11. `research-wiki/papers/ToRead/noisygrpo_neurips2025.pdf`
    - 重点看视觉输入噪声注入是否能提高多样性和鲁棒性。

## 实验路线

### E0：问题和数据定义

目标：构建一个小而清晰的细粒度 VQA 验证集。

数据建议：

| 类型 | 数据集候选 | 关注能力 |
|---|---|---|
| 细粒度视觉问答 | GQA / VQAv2 subset | 对象、属性、关系、计数 |
| OCR / 文档问答 | TextVQA / DocVQA | 文字定位和读取 |
| 图表 / 数学视觉推理 | ChartQA / MathVista | 图表元素、数值比较、计算 |

最小规模：

- 每类 100-300 个问题即可先跑通。
- 每题保留 image、question、gold answer、question type。
- 优先挑需要多步视觉证据的问题，不要只选一眼能答的问题。

### E1：生成多模型推理轨迹

目标：观察不同模型的 CoT 风格差异。

候选模型：

- Qwen2.5-VL-7B
- InternVL 系列
- LLaVA 系列
- Qwen3-VL，如果本地或 API 可用

每题采样：

- 每个模型生成 4-8 条 reasoning trajectory。
- 保留最终答案、完整 CoT、采样温度、模型名。

记录格式：

```json
{
  "id": "...",
  "image": "...",
  "question": "...",
  "gold_answer": "...",
  "model": "qwen2.5-vl-7b",
  "trajectory_id": 0,
  "reasoning": "...",
  "final_answer": "...",
  "is_final_correct": true
}
```

### E2：把 CoT 投影成 evidence tuple

目标：从模型私有 CoT 中抽取统一中间状态。

方法：

- 用强 VLM / LLM judge 把自然语言 CoT 解析成 evidence tuple。
- 对每个 step 标注 `operation`、`visual_evidence`、`intermediate_answer`。
- 对每个 step 标注 `validity`。

优先支持的 operation：

| operation | 示例 |
|---|---|
| `grounding` | 定位某个对象、人物、区域 |
| `ocr` | 读取图中文字 |
| `attribute` | 判断颜色、形状、材质、状态 |
| `counting` | 计数 |
| `spatial_relation` | 左右、上下、遮挡、包含 |
| `comparison` | 大小、数量、数值比较 |
| `calculation` | 基于图表或视觉数值计算 |
| `answer_synthesis` | 从中间结论合成最终答案 |

### E3：训练 step-level PRM

目标：训练一个判断单步推理是否有效的 PRM。

输入：

```text
image + question + previous_steps + current_step
```

输出：

```text
valid / invalid
```

或：

```text
step_reward in [0, 1]
```

训练目标：

- Binary classification：预测 step validity。
- Ranking loss：同一问题下 valid step 分数高于 invalid step。
- 可选：按 operation 做 multi-task head，观察不同视觉操作的难度。

### E4：PRM rerank / rejection sampling

目标：验证 step-level PRM 是否提升最终 VQA。

流程：

1. 对每题生成 K 条推理轨迹。
2. PRM 给每条轨迹的每个 step 打分。
3. 聚合为 trajectory score。
4. 选择最高分轨迹的 final answer。

聚合函数候选：

```text
mean(step_scores)
min(step_scores)
mean(step_scores) * final_consistency_score
weighted_sum(step_scores by operation)
```

对比 baseline：

| 方法 | 含义 |
|---|---|
| Greedy | 模型单次输出 |
| Majority Vote | 多条轨迹最终答案投票 |
| Outcome RM | 只看最终答案或最终解释打分 |
| Text-only PRM | 不输入图像，只判断文本推理 |
| Multimodal PRM | 输入图像，判断 step validity |

关键 baseline 与目的：

| Baseline | 目的 |
|---|---|
| Existing PRM / MRM | 检验现有好用的 PRM/MRM 直接作为外部 reward，能否迁移到多个架构 VLM 的 rerank 或后训练 |
| Outcome Reward | 检验只监督最终答案是否足够，作为 step-level PRM 的下界对照 |
| Model-specific PRM | 每个模型用自己的 CoT / rollout 训练 PRM，检验模型专属过程监督的上限 |
| Text-only PRM | 去掉图像输入，检查提升是否只是来自语言流畅性和 CoT 风格 |
| Unified Evidence PRM | 使用统一 evidence tuple 训练，检验模型无关的视觉证据监督是否更可复用 |
| Oracle / Rule Reward | 在可验证任务上使用 GT、IoU、OCR match、答案匹配等规则奖励，提供可达到的强参考 |

### E5：跨模型迁移实验

目标：验证监督信号是否统一、可复用。

核心设置：

| 训练 PRM 数据 | 测试 rollouts | 目的 |
|---|---|---|
| Qwen2.5-VL | Qwen2.5-VL | 同模型上限 |
| Qwen2.5-VL | InternVL / LLaVA | 跨模型迁移 |
| 多模型混合 | 单个未见模型 | 泛化能力 |
| 原始 CoT step | 未见模型 | 检查 CoT 风格过拟合 |
| evidence tuple | 未见模型 | 检查统一表示是否更稳 |

关键指标：

- Final VQA accuracy
- PRM step validity accuracy
- PRM-AUC / pairwise ranking accuracy
- Cross-model performance drop
- Text-only vs multimodal gap
- 不同 operation 上的 PRM 准确率

### E6：消融实验

建议消融：

| 消融 | 要回答的问题 |
|---|---|
| 去掉图像输入 | PRM 是否真的使用视觉信息？ |
| 去掉 visual evidence 字段 | 结构化证据是否必要？ |
| 只用最终答案正确性训练 | step-level 监督是否优于 outcome-only？ |
| 用自然语言 CoT step 替代 evidence tuple | 统一表示是否降低模型风格依赖？ |
| 单模型训练 vs 多模型训练 | 多模型数据是否提高可复用性？ |
| mean 聚合 vs min 聚合 | PRM 应该奖励整体质量还是惩罚任一坏步骤？ |

## 最小可行版本

第一轮不要直接做完整 RL。先做 PRM rerank。

MVP：

1. 选 300-900 个细粒度 VQA 样本。
2. 用 Qwen2.5-VL-7B 每题生成 4 条 CoT。
3. 用强 judge 把 CoT 转成 evidence tuple。
4. 标注每个 tuple 的 validity。
5. 训练一个小型 multimodal PRM。
6. 用 PRM rerank K 条候选轨迹。
7. 对比 greedy、majority vote、outcome RM、text-only PRM。
8. 换 InternVL / LLaVA rollouts 测跨模型迁移。

若 MVP 成立，再进入 RL：

- 用 PRM 作为 dense reward。
- 做 StepGRPO 或 PRM-guided GRPO。
- 比较 outcome reward、step reward、mixed reward。

## 预期论文贡献点

可能贡献：

1. 提出 model-agnostic visual evidence state，作为多模态 PRM 的统一监督表示。
2. 证明 evidence-level PRM 比 raw CoT PRM 更能跨模型迁移。
3. 证明 step-level multimodal PRM 比 outcome-only reward 更适合细粒度 VQA。
4. 给出不同视觉推理 operation 的 PRM 难度分析。
5. 将 PRM 用于 reranking 或 RL，提升细粒度 VQA 最终准确率。

## 风险与检查点

| 风险 | 检查方法 | 应对 |
|---|---|---|
| Judge 解析 CoT 不稳定 | 人工抽查 50-100 条 | 固定 schema，加入 few-shot 示例 |
| PRM 只学到文本流畅性 | text-only vs multimodal 对比 | 强制加入视觉证据字段和 hard negatives |
| 跨模型迁移差 | train/test 按模型拆分 | 使用 evidence tuple，混合多模型数据 |
| Step validity 标注噪声大 | 计算 judge 一致性 | 多 judge 投票或只保留高置信样本 |
| rerank 提升来自答案投票而非 PRM | 对比 majority vote | 控制 K 和候选集合 |

## 实验记录模板

```markdown
## Experiment: YYYY-MM-DD short_name

### Goal

### Data
- Dataset:
- Sample size:
- Question types:

### Models
- Generator:
- Judge:
- PRM backbone:

### Training
- Input format:
- Objective:
- Hyperparameters:

### Results
| Method | Accuracy | Step Acc | Cross-model Drop | Notes |
|---|---:|---:|---:|---|

### Findings

### Next
```
