# Dense VLM Multi-Teacher Distillation 理论支撑阅读清单

> 目标：为论文主线补齐理论依据，明确哪些已发表 prior work 会形成正式 novelty 边界，哪些未发表工作只作为参考/趋势/风险观察，以及当前最该精读哪些论文。

## 阅读与 novelty 判定原则

```text
1. Novelty attack 只考虑已发表或正式接收论文。
2. arXiv/未发表论文仍然要读，尤其是 MOPD、Drive-KD、CaMOPD 这类最近工作；
   但它们在写作中主要放在 concurrent work / recent preprints / risk watch。
3. Related work 可以覆盖未发表论文，因为它们能帮助我们定位方法、避免重复实验、提前回应趋势。
4. 正式 novelty 防线要优先建立在已发表的多任务梯度冲突、VLM KD、KB-VQA、process supervision 文献上。
```

## 0. 我们现在最稳的 claim

RLCSD 之后，不要把 novelty 写成“第一次发现 token-level KL 会学风格”。style drift 应该作为已知/近同期问题，用来支撑我们的动机，而不是主贡献。更稳的写法是：

```text
Dense VLM multi-teacher token-level distillation couples heterogeneous teacher signals
inside the same shared parameter space. In knowledge-intensive VQA, visual, knowledge,
and reasoning teachers can induce divergent or opposing gradients on capability-relevant
tokens. This conflict is compounded by the known style-drift problem of token-level
distillation, but cannot be solved by simply masking style tokens because the most
conflicting tokens are often capability-bearing tokens.
```

对应中文：

```text
在 dense VLM 的多教师蒸馏里，不同能力 teacher 的监督不是天然可加的。
它们会在同一个共享参数空间里产生能力相关 token 的梯度冲突；
风格漂移是 token-level 蒸馏的已知/近同期问题，但更关键的是：
冲突最强的 token 往往正是视觉、知识、推理能力相关 token，不能简单 mask 掉。
```

最关键的是把论文定位成：

```text
诊断 dense VLM 多教师蒸馏的机制性失效，并提出 capability-decoupled supervision。
```

方法贡献的强版本：

```text
从 masking-only 升级为 contrastive capability-decoupled supervision：
不删除能力关键 token，而是用正确/错误能力条件的差分消除共享风格项，
保留真正由 visual / knowledge / reasoning capability 支持的监督信号。
```

而不是：

```text
发明 MOPD / 首次发现多教师冲突 / 首次发现 KL 学风格。
```

当前最该读的短清单：

```text
1. RLCSD: 确认 style drift 已经被怎样处理，避免把它当主 novelty。
2. Drive-KD: 确认 VLM 多教师 capability conflict 最近 preprint 的覆盖范围。
3. MOPD: 确认 vanilla multi-teacher OPD 的目标函数和默认假设。
4. CaMOPD: 确认 MOPD counteraction 是否覆盖我们的 content-token conflict。
5. Revisiting OPD: 确认 token-level OPD/KL 的其他 failure modes。
6. PCGrad / CAGrad: 确认已发表 gradient conflict 边界和 baseline。
7. MoVE-KD: 确认已发表 VLM 多专家 KD 边界。
8. A-OKVQA / InfoSeek: 确认实验场景和 segment 标注可能性。
```

## 1. 第一优先级：必须精读，但按发表状态分层

### 1.1 MOPD

PDF: `MOPD_2606.30406.pdf`

为什么读：

- 它是当前方向的直接靶子，定义 multi-teacher on-policy distillation 叙事。
- 它把“多能力 teacher 并行开发，再蒸馏进一个 student”包装成 post-training primitive。
- 它强调 on-policy rollout 和 dense token-level signal，这是我们要诊断的训练目标。
- 如果仍是未发表 preprint，不把它当作正式 novelty attack，而是作为最近趋势和问题设置参考。

重点看：

- teacher 如何按 domain/capability 构建。
- student rollout 上如何拿 teacher token-level feedback。
- 多 teacher 如何路由、混合或调度。
- 是否假设 teacher supervision 可加。
- 是否真的分析 token/gradient conflict。

我们的写法：

```text
MOPD shows the promise of multi-teacher capability integration.
We ask when this promise breaks in dense VLMs.
```

### 1.2 Drive-KD

PDF: `drive_kd_2601.21288.pdf`

为什么读：

- 这是最接近的 VLM 多教师冲突 preprint，因为它在 VLM 多教师蒸馏中明确讨论 cross-capability gradient conflicts。
- 如果它尚未发表，不把它作为正式 novelty attack；但它仍然是必须关注的 concurrent/recent work。
- 它提出 AGP 之类的梯度缓解方法，能帮助我们提前写清楚方法边界。

重点看：

- 它怎么把 driving 拆成 perception / reasoning / planning。
- 它怎么定义和测 cross-capability gradient conflict。
- AGP 是投影什么梯度、作用在哪些层。
- 它的监督主要是 attention / layer-level / driving triad，还是 token-level on-policy KL。

我们的差异：

```text
Drive-KD studies autonomous-driving VLM distillation with capability-specific modules/signals.
We study knowledge-intensive VQA under dense token-level/on-policy multi-teacher supervision,
where conflicts emerge at answer-token and capability-segment level.
```

不要写：

```text
We are the first to identify multi-teacher gradient conflict in VLMs.
```

更稳的写法：

```text
Among published work, generic gradient conflict has been studied in multi-task optimization,
and VLM distillation has explored multi-teacher/multi-expert transfer. Recent preprints
also begin to discuss cross-capability conflict in VLM distillation. We focus on dense
token-level/on-policy multi-teacher supervision for knowledge-intensive VQA.
```

### 1.3 CaMOPD

PDF: `camopd_2605.27115.pdf`

为什么读：

- 它是 MOPD 最近的冲突修补工作。
- 它已经指出 vanilla MOPD 在 incomplete coverage 下会有 counteraction。
- 如果尚未发表，它属于 recent preprint/risk watch，而不是正式 novelty attack。

重点看：

- recovery-preservation counteraction 的形式化。
- gradient coherence / counteraction analysis 怎么做。
- alternating training 和 gap-based selection 如何缓解。
- 它的任务冲突是 general recovery vs domain preservation，不是 VLM 内视觉-知识-推理冲突。

我们的差异：

```text
CaMOPD treats counteraction across recovery/preservation objectives.
We focus on capability-level conflict inside multimodal answers and use content/segment-level diagnosis.
```

### 1.4 Revisiting OPD

PDF: `revisiting_opd_2603.25562.pdf`

为什么读：

- 支撑“OPD/token-level KL 不是天然可靠”的大前提。
- 它能帮我们避免把问题说得太孤立。

重点看：

- sampled-token log-ratio 为什么 brittle。
- long rollout prefix drift 如何影响 teacher guidance。
- special-token / tokenizer / support mismatch 相关问题。
- 它提出的 simple fixes 是否接近 masking / local support matching。

我们的差异：

```text
Revisiting OPD analyzes single-teacher OPD failure modes.
We extend the diagnosis to multi-teacher dense VLM distillation, where teacher disagreement
and capability entanglement introduce additional conflict.
```

## 2. 第二优先级：支撑我们的解决方案

### 2.1 StepOPSD

PDF: `stepopsd_2605.27140.pdf`

作用：

- 支撑“从 token-level supervision 转向 step/segment-level credit”。
- 它是 agent step，不是 VLM capability segment，但逻辑相近。

重点看：

- 如何从 trajectory 拆 step。
- 如何把 token log-prob gap 聚合成 step advantage。
- 为什么 heterogeneous trajectory 不适合整串 token 监督。

### 2.2 OmniOPD

PDF: `omniopd_2606.01476.pdf`

作用：

- 支撑“可以不依赖 teacher logits / token KL，也能做 OPD”。
- 和我们的 segment-level preference 是同一类思想盟友。

重点看：

- logit-free OPD 的动机。
- speculative verification / chunk-level semantic verification。
- 它如何避免 teacher logit access 和 token-level noise。

### 2.3 RLCSD

PDF: `rlcsd_2606.11709.pdf`

作用：

- 直接支撑我们的 style 问题：RLCSD 指出 OPSD 的 dense token signal 会集中在 style tokens，而不是 task-bearing tokens。
- 它把这个问题称为 privilege-induced style drift：带 hint 的 teacher 往往更短、更直接，导致 token gap 学到长度/表达风格。
- 它的解决方式不是简单 mask，而是用 correct hint 与 wrong hint 的 teacher-student gap 做差分，抵消 hint 条件带来的共享风格项。

重点看：

- style tokens vs task-bearing tokens 的统计方式。
- correct-hint gap 与 wrong-hint gap 的 contrastive signal。
- 为什么共享风格项可以通过差分被消掉。
- 它是否声称能扩展到 cross-model on-policy distillation。

对我们的启发：

```text
Masking 只能减少明显非内容 token 的监督，但容易被认为贡献不足；
更严重的是，若冲突最大的 token 正好是能力关键 token，masking 会把需要学习的地方删掉。
因此我们的核心方法不应是 masking-only，而应是 contrastive capability-decoupled supervision：
保留能力关键 token，但用正/负能力条件差分去掉共享风格项和非能力漂移。
```

可以借鉴的形式：

```text
visual_signal =
  gap(visual_teacher | correct visual evidence)
- gap(visual_teacher | corrupted/wrong visual evidence)

knowledge_signal =
  gap(knowledge_teacher | correct evidence/knowledge)
- gap(knowledge_teacher | wrong evidence/knowledge)

reasoning_signal =
  gap(reason_teacher | correct reasoning bridge)
- gap(reason_teacher | invalid bridge)
```

其中：

```text
gap = log p_teacher(token | condition, prefix) - log p_student(token | prefix)
```

这样留下来的不是“teacher 风格上更喜欢这个 token”，而是“正确能力条件相比错误能力条件更支持这个 token”。

RLCSD 之后我们的 novelty 应该这样收缩：

```text
不是：发现 OPD/token-level distillation 会学风格。
而是：在 dense VLM multi-teacher distillation 中，style drift 之外还存在
capability-specialized teachers 对 task-bearing tokens 的直接冲突。
```

RLCSD 不能直接解决我们的原因：

```text
RLCSD 是 self-distillation / privileged-hint contrast；
我们的设置是 cross-teacher / cross-capability VLM distillation。
它可以消除 correct/wrong hint 共享的风格项，
但不能自动决定 visual teacher、knowledge teacher、reasoning teacher
在同一个 answer token 上谁应该更新、谁应该让路。
```

### 2.4 DPKD / GKD

PDF:

- `dpkd_2406.19774.pdf`
- `gkd_on_policy_distillation_2306.13649.pdf`

作用：

- 补齐 KD/OPD 从 off-policy 到 on-policy、从 forward KL 到 generalized KD 的基础。
- 不需要花太多时间，但要知道它们解决的不是多教师能力冲突。

重点看：

- data distribution mismatch / student-generated samples。
- forward KL、reverse KL、JSD 等 divergence 的选择。
- teacher-student mismatch 对蒸馏的影响。

## 3. 第三优先级：梯度冲突理论底座

### 3.1 PCGrad

PDF: `pcgrad_neurips2020.pdf`

作用：

- 多任务梯度冲突的经典基线。
- 支撑我们使用 gradient cosine / negative cosine 作为机制诊断指标。

重点看：

- conflict gradient 的定义。
- projection 公式。
- 何时梯度冲突会伤害 optimization。

### 3.2 CAGrad

PDF: `cagrad_neurips2021.pdf`

作用：

- 比 PCGrad 更强的多目标优化 baseline。
- 如果未来补“梯度投影 baseline”，它是候选。

重点看：

- conflict-averse objective。
- average gradient 与 worst local improvement 的权衡。

### 3.3 GradVac

PDF: `gradvac_2010.05874.pdf`

作用：

- 支撑“梯度相似性可以反映任务关系/迁移关系”。
- 可用于理论段落解释为什么 teacher gradient alignment 是有效诊断。

重点看：

- gradient similarity 与 task/language proximity 的关系。
- 沿训练轨迹统计 gradient cosine 的方法。

## 4. 第四优先级：世界知识 VQA 实验合理性

### 4.1 OK-VQA / A-OKVQA

PDF:

- `okvqa_cvpr2019.pdf`
- `aokvqa_eccv2022.pdf`

作用：

- 证明我们选的任务不是普通 VQA，而是天然需要视觉 + 外部知识。
- A-OKVQA 的 rationales 对 segment analysis 有潜在帮助。

重点看：

- 数据如何保证需要 external/world knowledge。
- direct answer、multiple choice、rationale 的评估设计。
- 错误类型是否能映射到 visual / knowledge / reasoning。

### 4.2 InfoSeek / Encyclopedic-VQA

PDF:

- `infoseek_emnlp2023.pdf`
- `encyclopedic_vqa_iccv2023.pdf`

作用：

- 它们比 OK-VQA 更贴近 fine-grained entity + Wikipedia knowledge。
- 最适合把 visual teacher 和 knowledge teacher 的功能拆开。

重点看：

- 问题是否无法仅靠图像回答。
- 是否有实体、知识源或 evidence。
- 是否支持按实体识别、知识检索、回答推理拆 error。

### 4.3 EchoSight / ReAG

PDF:

- `echosight_emnlp2024.pdf`
- `reag_cvpr2026.pdf`

作用：

- 这是 KB-VQA/RAG-VQA 的强相关方法，不是我们主方法 baseline，但要知道它们代表当前任务解法。
- 它们提醒我们：世界知识 VQA 的强系统通常依赖 retrieval/evidence，而不是纯参数记忆。

重点看：

- retrieval 如何从视觉实体启动。
- reranking / critic / reasoning over retrieved content 如何设计。
- 它们在 InfoSeek / Encyclopedic-VQA 上的评价协议。

## 5. 第五优先级：VLM 多教师/多专家 KD 相关工作

### 5.1 MoVE-KD

PDF: `move_kd_cvpr2025.pdf`

作用：

- VLM 里“多个视觉专家蒸馏到一个模型”的相关工作。
- 它用 LoRA/MoE 保留 teacher 特性，对我们 dense vs routed adapters 的对照很有启发。

重点看：

- 多视觉 encoder 的差异如何处理。
- LoRA/MoE 如何缓解冲突。
- 它是否处理 language-side token supervision。

### 5.2 AMMKD

PDF: `ammkd_2509.00039.pdf`

作用：

- 多模态多教师 KD，且显式提到 gradient space diversity / teacher weighting。
- 适合作为 “multi-teacher KD has conflict-aware weighting, but not our dense token-level VLM MOPD setting” 的 related work。

重点看：

- adaptive teacher weighting。
- gradient diversity 如何使用。
- 任务是 retrieval/lightweight VLP，不是 generative VQA distillation。

## 6. 第六优先级：过程监督/片段偏好理论盟友

### 6.1 Let's Verify Step by Step

PDF: `lets_verify_step_by_step_2305.20050.pdf`

作用：

- 经典 process supervision 论文。
- 支撑“只看 final outcome/token imitation 不足，过程/片段监督更有效”。

### 6.2 Math-Shepherd

PDF: `math_shepherd_acl2024.pdf`

作用：

- 自动构造 step-level process supervision。
- 启发我们未来用自动 judge 构造 capability-segment preference。

重点看：

- step quality 如何自动估计。
- verification 与 reinforcement 两种使用方式。

## 7. 旁支阅读：只需扫读

PDF:

- `full_rollouts_opd_2605.31490.pdf`
- `mad_opd_2605.01347.pdf`
- `revisiting_opd_2603.25562.pdf`
- `opd_rethinking_2026.pdf`

作用：

- 帮我们确认 2026 OPD 社区正在关注 rollout cost、teacher signal reliability、multi-rollout feedback、failure modes。
- 它们不是主威胁，但 related work 里可以一句带过。

## 8. 推荐阅读顺序

第一天，看问题设置和最近风险，不把未发表论文当正式 novelty attack：

1. `MOPD_2606.30406.pdf`
2. `drive_kd_2601.21288.pdf`
3. `camopd_2605.27115.pdf`
4. `revisiting_opd_2603.25562.pdf`

第二天，看实验任务和 benchmark：

1. `okvqa_cvpr2019.pdf`
2. `aokvqa_eccv2022.pdf`
3. `infoseek_emnlp2023.pdf`
4. `encyclopedic_vqa_iccv2023.pdf`
5. `reag_cvpr2026.pdf`

第三天，看方法设计支撑：

1. `stepopsd_2605.27140.pdf`
2. `omniopd_2606.01476.pdf`
3. `rlcsd_2606.11709.pdf`
4. `lets_verify_step_by_step_2305.20050.pdf`
5. `math_shepherd_acl2024.pdf`

第四天，看 baseline 和审稿人会问的替代方案：

1. `pcgrad_neurips2020.pdf`
2. `cagrad_neurips2021.pdf`
3. `gradvac_2010.05874.pdf`
4. `move_kd_cvpr2025.pdf`
5. `ammkd_2509.00039.pdf`

## 9. 目前最应该写进论文 related work 的分组

### Multi-teacher on-policy distillation

MOPD, CaMOPD, MAD-OPD, OmniOPD, Revisiting OPD。

核心观点：

```text
Existing OPD/MOPD methods improve capability integration but mostly treat dense teacher
signals as useful optimization targets. Recent work starts to identify OPD fragility and
cross-objective counteraction, but has not studied dense VLM knowledge-intensive answers
at content-token/capability-segment granularity.
```

写作注意：

```text
如果这些 OPD/MOPD 工作仍是未发表 preprints，放在 recent/concurrent work；
不要用它们来主动削弱自己的 novelty，只用来说明研究问题正在变重要。
```

### Multi-teacher / multi-task gradient conflict

PCGrad, CAGrad, GradVac, Drive-KD, AMMKD。

核心观点：

```text
Gradient conflict is a known multi-task issue. Our contribution is not the generic concept,
but its concrete manifestation in dense VLM multi-teacher token supervision and the
empirical decomposition into capability-relevant content conflict versus surface-form KL.
```

正式 novelty 边界：

```text
PCGrad, CAGrad, GradVac 等已发表工作说明“梯度冲突”不是新概念；
MoVE-KD 等已发表 VLM KD 工作说明“多专家/多教师 VLM 蒸馏”不是空白。
Drive-KD/AMMKD 若未发表，只作为近期参考，不作为正式 novelty attack。
```

### Knowledge-intensive VQA

OK-VQA, A-OKVQA, InfoSeek, Encyclopedic-VQA, EchoSight, ReAG。

核心观点：

```text
Knowledge-intensive VQA naturally decomposes into visual grounding, external/world
knowledge, and reasoning, making it a good stress test for multi-capability distillation.
```

### Process / segment-level supervision

Let's Verify Step by Step, Math-Shepherd, StepOPSD, OmniOPD, RLCSD。

核心观点：

```text
Process supervision suggests that supervision should be assigned at semantically meaningful
steps rather than blindly across all tokens. We instantiate this idea for VLM capability
segments.
```

RLCSD 对我们的额外约束：

```text
Style-dominated dense supervision should not be solved only by deleting tokens.
The stronger solution is to construct contrastive signals where correct and wrong
capability conditions share style but differ in task correctness.
```

## 10. Go / no-go 判断

理论上，这个方向是成立的，但必须避免过宽 claim：

```text
GO:
Dense VLM multi-teacher distillation can exhibit capability-level gradient conflict and
surface-form dominated KL on knowledge-intensive VQA.

NO-GO:
We are the first to discover multi-teacher conflict.
We prove all dense models are worse than MoE.
Token-level KL is always bad.
```

更适合三大会的版本：

```text
We identify and quantify a specific failure mode of dense multi-teacher VLM distillation:
token-level supervision entangles capability transfer with surface-form imitation, and
capability-specialized teachers can disagree on the same content-bearing answer tokens.
Motivated by this diagnosis, we propose capability-decoupled segment supervision.
```
