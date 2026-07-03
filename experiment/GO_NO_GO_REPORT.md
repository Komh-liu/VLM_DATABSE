# Go / No-Go 报告：Dense VLM Multi-Teacher KL 冲突假设

日期：2026-07-03

## 结论

当前 pilot 结果支持继续推进。

```text
GO
```

但当前结论仍是 pilot-level，不是最终论文级证明。

最强证据：

```text
same teacher:
  content-token gradient cosine ≈ 1.000

different capability teacher, same style:
  content-token gradient cosine ≈ 0.327
  negative cosine rate ≈ 30.4%

same capability, different style:
  content-token gradient cosine ≈ 0.429
  negative cosine rate ≈ 25.2%
```

这说明：

1. 脚本和 cosine 计算本身没有制造假冲突。
2. 在 prompt/style 受控后，visual LoRA teacher 和 knowledge LoRA teacher 仍在内容 token 上产生明显梯度分歧。
3. 仅改变同一能力 teacher 的表达风格，也会造成显著梯度差异。
4. 非内容 token 对 KL loss 的贡献超过 50%，支持 style/function-token dominance。

---

## 实验设置

### 数据

样本文件：

```text
experiment/probes/probe_samples_120.jsonl
```

组成：

```text
A-OKVQA: 80 条
V*Bench: 40 条
```

### Teacher

当前使用轻量 LoRA teacher：

```text
Visual teacher:
  experiment/probes/adapters/visual_lora_r8
  data: V*Bench
  steps: 60

Knowledge teacher:
  experiment/probes/adapters/knowledge_lora_r8
  data: A-OKVQA
  steps: 45
```

### 指标

对同一个 student rollout，计算：

```text
cosine(
  grad KL(student || teacher_1),
  grad KL(student || teacher_2)
)
```

当前是 logit-space KL gradient proxy。

token 类型：

```text
content / function / marker / punct / subword / space
```

核心关注：

```text
content-token cosine
negative cosine rate
cos < 0.2 rate
KL contribution by token kind
```

---

## 实验 1：Different Capability, Same Style

目的：

```text
验证 Problem 1：在风格和输出格式受控后，
不同能力 teacher 是否仍产生内容 token 梯度冲突。
```

设置：

```text
student:           base Qwen2.5-VL-3B
teacher_visual:    visual_lora_r8
teacher_knowledge: knowledge_lora_r8
prompt/style:      identical
```

结果文件：

```text
experiment/probes/go_nogo_lora_v_vs_k_120.jsonl
```

### 结果

全部 token：

```text
n:              6807
mean cosine:    0.395
median:         0.780
min:           -1.000
negative rate:  27.6%
cos < 0.2:      31.3%
```

内容 token：

```text
n:              2816
mean cosine:    0.327
median:         0.633
min:           -1.000
negative rate:  30.4%
cos < 0.2:      35.3%
```

按数据集：

```text
A-OKVQA content:
  n:              1960
  mean cosine:    0.352
  negative rate:  29.0%
  cos < 0.2:      33.9%

V*Bench content:
  n:              856
  mean cosine:    0.269
  negative rate:  33.5%
  cos < 0.2:      38.3%
```

### 判断

支持 Problem 1。

在相同 prompt/style 下，不同能力 teacher 对内容 token 的梯度不只是轻微偏移，而是有约三成 token 出现负 cosine。

---

## 实验 2：Same Teacher Sanity Baseline

目的：

```text
排除脚本、数值、token alignment 本身制造冲突的可能。
```

设置：

```text
student:   base Qwen2.5-VL-3B
teacher_1: visual_lora_r8
teacher_2: visual_lora_r8
```

结果文件：

```text
experiment/probes/go_nogo_same_visual_120.jsonl
```

### 结果

全部 token：

```text
n:              6807
mean cosine:    1.000
min:            1.000
negative rate:  0.0%
```

内容 token：

```text
n:              2816
mean cosine:    1.000
min:            1.000
negative rate:  0.0%
```

### 判断

sanity baseline 通过。

因此实验 1 的低 cosine 不是由统计流程造成的，而是 teacher 分布差异导致的。

---

## 实验 3：Same Capability, Different Style

目的：

```text
验证 Problem 2：即使能力相同，仅表达风格不同，也会造成 token-level KL 梯度差异。
```

设置：

```text
model:      base Qwen2.5-VL-3B
teacher_a: visual-focused concise style
teacher_b: visual-focused descriptive style
capability: same visual capability
style:      different
```

结果文件：

```text
experiment/probes/go_nogo_visual_style_120.jsonl
```

### 结果

全部 token：

```text
n:              7531
mean cosine:    0.496
median:         0.832
min:           -1.000
negative rate:  22.3%
cos < 0.2:      26.5%
```

内容 token：

```text
n:              3099
mean cosine:    0.429
median:         0.744
min:           -1.000
negative rate:  25.2%
cos < 0.2:      30.1%
```

### 判断

支持 Problem 2。

即使 capability 相同，仅改变表达风格，也会使 token-level KL 产生明显梯度差异。

---

## 实验 4：KL Contribution by Token Kind

目的：

```text
验证 token-level KL 是否被非内容 token 显著占据。
```

### Different Capability, Same Style

KL contribution：

```text
content:   46.0%
function:  38.2%
punct:     12.1%
space:      2.3%
subword:    1.1%
marker:     0.3%
```

非内容 token 总贡献：

```text
54.0%
```

### Same Capability, Different Style

KL contribution：

```text
content:   47.3%
function:  28.7%
punct:     16.7%
space:      4.9%
subword:    1.5%
marker:     0.8%
```

非内容 token 总贡献：

```text
52.7%
```

### 判断

支持 Problem 2。

即使 content token 是能力相关部分，非内容 token 仍贡献超过一半 KL loss。这说明 token-level KL 的 dense signal 很大部分用于学习表达脚手架，而不是能力关键 token。

---

## 当前假设状态

### Problem 1：Capability Gradient Conflict

当前状态：

```text
支持
```

证据：

```text
same teacher content cosine:          1.000
visual vs knowledge content cosine:   0.327
visual vs knowledge negative rate:    30.4%
```

解释：

不同能力 teacher 在相同 prompt/style 下仍产生显著内容 token 梯度分歧。

### Problem 2：Style-Dominant Supervision

当前状态：

```text
支持
```

证据：

```text
same capability different style content cosine: 0.429
same capability different style negative rate:  25.2%
non-content KL contribution:                    52.7%-54.0%
```

解释：

风格、功能词、标点、格式等非内容信号足以制造明显梯度差异，并占据大量 KL loss。

---

## Go / No-Go 判断

### Go 条件

原定 go 条件：

```text
1. content-token 上不同能力 teacher cosine 明显低于 same-teacher baseline
2. negative cosine rate 显著大于 0
3. 风格/非内容 token 对 KL 有显著贡献
4. 结果不是脚本 artifact
```

当前结果：

```text
1. 通过
2. 通过
3. 通过
4. 通过
```

因此：

```text
GO
```

---

## 仍然不能声称什么

当前还不能声称：

```text
1. 最终方法已经优于 MOPD。
2. GRPO teacher 上一定同样成立。
3. 所有世界知识 VQA benchmark 都有同样现象。
4. logit-space gradient proxy 完全等价于参数梯度。
5. token heuristic filter 等价于人工内容 token 标注。
```

---

## 下一步

为了把 pilot 证据升级成论文级证据，需要补：

```text
1. 训练更强 teacher，至少 LoRA teacher 要有独立能力评测。
2. 扩展到 OK-VQA / InfoSeek / Encyclopedic-VQA。
3. 计算 LoRA 参数梯度 cosine，而不只用 logit-space proxy。
4. 实现并对比：
   - vanilla MOPD
   - content-masked KL
   - segment-isolated KL
   - content + segment KL
   - generic DPO / segment preference
   - capability-specific segment preference
5. 验证方法是否降低 conflict，同时提升最终任务指标。
```

