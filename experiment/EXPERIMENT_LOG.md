# 先导实验日志

## 2026-07-03：环境与 Prompt-only Probe

### 环境

- 本地虚拟环境：`.venv`
- 模型：`experiment/models/Qwen2.5-VL-3B-Instruct`
- GPU：RTX 5080 16GB
- WSL2 下运行 GPU 命令需要：

```bash
LD_LIBRARY_PATH=/usr/lib/wsl/lib .venv/bin/python ...
```

### Prompt-only probe

脚本：

- `experiment/probes/run_logit_conflict_probe.py`

数据：

- `experiment/probes/probe_samples.jsonl`
- 8 条 A-OKVQA
- 4 条 V*Bench

结果：

```text
overall mean cosine: 0.354
A-OKVQA mean:        0.451
V*Bench mean:        0.160
min cosine:         -1.000
```

观察：

- 能测到明显低 cosine / 负 cosine。
- 但大量负值来自格式 marker、标点和功能词。
- 该阶段只能证明“不同 prompt-conditioned teacher 会产生差异化梯度”，不能证明能力差异。

## 2026-07-03：轻量 LoRA Teacher Probe

### 训练数据

脚本：

- `experiment/probes/make_lora_train_data.py`

产物：

- `experiment/probes/lora_data/visual_train.jsonl`
- `experiment/probes/lora_data/knowledge_train.jsonl`

数据构造：

- Visual LoRA：V*Bench，160 条，使用 target object / bbox / visual evidence 构造训练目标。
- Knowledge LoRA：A-OKVQA，160 条，使用 question / choices / rationale / answer 构造训练目标。

### LoRA teacher 训练

脚本：

- `experiment/probes/train_lora_teacher.py`

Visual adapter：

```text
path:       experiment/probes/adapters/visual_lora_r8
rank:       8
steps:      60
mean_loss:  0.800
last_loss:  0.315
```

Knowledge adapter：

```text
path:       experiment/probes/adapters/knowledge_lora_r8
rank:       8
steps:      45
mean_loss:  0.889
last_loss:  1.258
```

备注：

- Knowledge adapter 首次 60 step 训练在 58/60 被系统 kill，未保存。
- 第二次降低到 45 step 后成功保存。

### LoRA teacher conflict probe

脚本：

- `experiment/probes/run_lora_teacher_conflict_probe.py`

关键修正：

- Student logits 使用 base model，无 adapter。
- Visual teacher logits 使用 visual LoRA adapter。
- Knowledge teacher logits 使用 knowledge LoRA adapter。
- 三者使用相同 prompt 和相同输出格式，减少纯 prompt 风格差异。

结果文件：

- `experiment/probes/lora_teacher_conflict_results.jsonl`

结果：

```text
overall mean cosine: 0.466
A-OKVQA mean:        0.473
V*Bench mean:        0.454
min cosine:         -1.000
```

与 prompt-only 对比：

```text
prompt-only overall mean: 0.354
LoRA-teacher overall mean: 0.466
```

解释：

- LoRA 同 prompt 版本的整体冲突比 prompt-only 弱，说明 prompt 风格确实贡献了大量噪声。
- 但局部 token 仍有强负 cosine，说明权重差异 teacher 之间仍存在显著局部梯度分歧。

最负 token 示例：

```text
birthday / cultures / water / substance / cupcakes / holding / glass / toys / black / white
```

同时仍有不少 marker 碎片：

```text
KN / OW / ASON / WER
```

当前判断：

- “teacher 差异化梯度存在”已经有工程证据。
- “能力差异化梯度存在”还没有完全证明，因为 marker 与功能词噪声仍然存在。
- 下一步应做 content-token filtering 和 same-capability/different-style 对照。

## 下一步

优先级最高的下一步：

1. 过滤 marker、标点、功能词，只统计内容 token。
2. 加 same-capability/different-style baseline。
3. 对 visual / knowledge teacher 做小评测，证明二者确实能力偏置不同。
4. 如果 content token 上仍有稳定低 cosine，再升级到 LoRA 参数梯度，而不是 logit-space gradient。

## 2026-07-03：Problem 1 内容词过滤验证

### 目标

验证：

```text
在相同 prompt / 相同输出格式下，
visual teacher 和 knowledge teacher 是否仍然对内容 token 给出不同方向的 KL 梯度。
```

这一步用于支撑 Problem 1：Capability Gradient Conflict。

### 方法

修改脚本：

- `experiment/probes/run_lora_teacher_conflict_probe.py`

新增 token 分类：

- `content`
- `function`
- `marker`
- `punct`
- `subword`
- `space`

核心过滤：

```text
只看 content token，排除 marker、标点、功能词和明显碎片。
```

### 对照设置

#### 1. Different capability, same style

```text
student:           base Qwen2.5-VL-3B
teacher_visual:    visual_lora_r8
teacher_knowledge: knowledge_lora_r8
prompt/style:      identical
```

结果文件：

- `experiment/probes/lora_teacher_conflict_results_tokens.jsonl`

#### 2. Same teacher sanity baseline

```text
student:           base Qwen2.5-VL-3B
teacher_1:         visual_lora_r8
teacher_2:         visual_lora_r8
prompt/style:      identical
```

结果文件：

- `experiment/probes/lora_same_visual_baseline_tokens.jsonl`

### 结果

#### Different capability, same style

全部 token：

```text
n:              640
mean cosine:    0.452
median:         0.845
min:           -1.000
negative rate:  24.4%
cos < 0.2:      28.3%
```

内容 token：

```text
n:              261
mean cosine:    0.335
median:         0.669
min:           -1.000
negative rate:  30.3%
cos < 0.2:      35.2%
```

内容 token 按数据集：

```text
A-OKVQA:
  n:              186
  mean cosine:    0.373
  negative rate:  26.3%
  cos < 0.2:      32.8%

V*Bench:
  n:              75
  mean cosine:    0.242
  negative rate:  40.0%
  cos < 0.2:      41.3%
```

内容 token 按 segment：

```text
VISUAL:
  n:              119
  mean cosine:    0.322
  negative rate:  31.9%

KNOWLEDGE:
  n:              96
  mean cosine:    0.358
  negative rate:  27.1%

REASON:
  n:              46
  mean cosine:    0.321
  negative rate:  32.6%
```

最负内容 token 示例：

```text
water / objects / white / holding / board / substance / black /
colors / blue / birthday / sheep / mud / cupcakes / laptop / glass
```

#### Same teacher sanity baseline

全部 token：

```text
n:              640
mean cosine:    1.000
min:            1.000
negative rate:  0.0%
```

内容 token：

```text
n:              261
mean cosine:    1.000
min:            1.000
negative rate:  0.0%
```

### 判断

这个对照比 prompt-only probe 更干净：

```text
same teacher:
  content cosine ≈ 1.000

different capability teacher, same style:
  content cosine ≈ 0.335
  content negative rate ≈ 30.3%
```

因此可以初步支持：

```text
Problem 1: 不同能力 LoRA teacher 在内容 token 上确实产生差异化、
甚至相反方向的 KL 梯度。
```

当前仍需谨慎：

- teacher 是轻量 SFT LoRA，不是 GRPO teacher。
- token 分类是启发式规则，不是人工标注。
- 样本只有 12 条。
- 当前梯度是 logit-space KL gradient proxy，还不是 LoRA 参数梯度。

但作为 go/no-go evidence，Problem 1 已经从“风格噪声可能解释一切”推进到：

```text
在风格受控后，内容 token 上仍存在显著 teacher gradient divergence。
```

## 2026-07-03：120 样本 Go / No-Go 扩展

完整报告见：

```text
experiment/GO_NO_GO_REPORT.md
```

核心结论：

```text
GO
```

关键数字：

```text
Different capability, same style:
  content-token mean cosine: 0.327
  content negative rate:     30.4%

Same teacher baseline:
  content-token mean cosine: 1.000
  content negative rate:     0.0%

Same capability, different style:
  content-token mean cosine: 0.429
  content negative rate:     25.2%

Non-content KL contribution:
  52.7% - 54.0%
```

判断：

```text
Problem 1: supported at pilot scale
Problem 2: supported at pilot scale
```

仍需补强：

```text
更强 teacher
更多 benchmark
参数梯度版本
真实训练收益
masked/segment KL 与 preference 方法对比
```
