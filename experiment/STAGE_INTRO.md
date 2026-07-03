# LoRA 版先导实验阶段说明

本文档记录当前 `experiment/` 下先导实验的阶段、目标和需要关注的判断标准。

## Stage 0：环境与模型验证

目标是确认本机能稳定跑 Qwen2.5-VL-3B。

当前状态：

- Python 环境在项目本地 `.venv`。
- PyTorch 使用 `torch==2.11.0+cu128`。
- 模型路径为 `experiment/models/Qwen2.5-VL-3B-Instruct`。
- 机器是 WSL2 + RTX 5080 16GB。运行 GPU 命令时需要带：

```bash
LD_LIBRARY_PATH=/usr/lib/wsl/lib .venv/bin/python ...
```

已验证：

- PyTorch 在非沙箱 GPU 环境下可以识别 RTX 5080。
- Qwen2.5-VL-3B 可以加载并完成一条 VLM 推理。
- 单次 bf16 推理峰值显存约 7GB。

## Stage 1：Prompt-only Probe

目标是快速验证 KL/logit conflict 统计链路是否可运行。

做法：

- 不训练 teacher。
- 使用同一个 base model。
- 通过不同 prompt 模拟 student、visual teacher、knowledge teacher。
- 对同一个生成答案计算：

```text
cosine(grad KL(student || visual_teacher),
       grad KL(student || knowledge_teacher))
```

已产物：

- `experiment/probes/probe_samples.jsonl`
- `experiment/probes/logit_conflict_results.jsonl`
- `experiment/probes/run_logit_conflict_probe.py`

已观察到：

- 存在大量低 cosine / 负 cosine token。
- 但许多负值落在格式 marker、标点、功能词上。

结论：

Prompt-only probe 只能说明“不同 teacher 条件会产生不同梯度”，不能证明这是能力差异导致的。它主要用于检查数据、模型、KL 计算和可视化链路。

## Stage 2：轻量 LoRA Teacher

目标是把 teacher 差异从 prompt 风格差异推进到“权重差异”。

当前做法：

- Visual LoRA teacher：
  - 数据：V*Bench。
  - 目标：学习 object / bbox / visual evidence 风格。
  - 输出目录：`experiment/probes/adapters/visual_lora_r8`。

- Knowledge LoRA teacher：
  - 数据：A-OKVQA。
  - 目标：学习 rationale / commonsense / answer explanation 风格。
  - 输出目录：`experiment/probes/adapters/knowledge_lora_r8`。

训练脚本：

- `experiment/probes/make_lora_train_data.py`
- `experiment/probes/train_lora_teacher.py`

注意：

这不是最终论文级 teacher。它是轻量 SFT/LoRA teacher，用来快速检查能力方向是否能被拉开。

## Stage 3：LoRA Teacher Conflict Probe

目标是用相同 prompt、相同格式、不同 LoRA 权重来测 teacher 差异。

脚本：

```bash
LD_LIBRARY_PATH=/usr/lib/wsl/lib \
.venv/bin/python experiment/probes/run_lora_teacher_conflict_probe.py \
  --visual-adapter experiment/probes/adapters/visual_lora_r8 \
  --knowledge-adapter experiment/probes/adapters/knowledge_lora_r8 \
  --limit 12 \
  --max-new-tokens 64
```

需要关注：

- overall mean cosine 是否明显低于 prompt-only 的风格控制 baseline。
- VISUAL / KNOWLEDGE / REASON / ANSWER 各 segment 的 cosine 分布。
- 负 cosine 是否仍主要来自功能词和 marker。
- 如果内容 token 上出现稳定低 cosine，才说明能力差异梯度开始显现。

## Stage 4：Go / No-Go 判断

继续 MOPD 梯度冲突方向的条件：

```text
different capability, same style 的 cosine
明显低于
same capability, different style 的 cosine
```

并且：

- 低 cosine 不只来自 marker、标点和功能词。
- MIXED / REASON / ANSWER 中的内容 token 有稳定冲突。
- visual teacher 与 knowledge teacher 在各自小评测上确实表现出能力偏置。

如果这些条件不成立，应收缩 MOPD 方向，转向 Visual PRM / task-specialized PRM 方向。

## 当前最重要的问题

现在不是证明最终方法有效，而是回答：

```text
风格噪声之外，能力差异是否真的会带来不同方向的 teacher gradient？
```

轻量 LoRA 版实验就是为这个问题服务的。
