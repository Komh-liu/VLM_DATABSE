# 数据集需求交接：VLM 多教师 / MOPD 实验

## 架构总览

| 组件 | 模型 | 训练？ | 输入 | 数据需求 |
|------|------|--------|------|---------|
| 视觉 Teacher π_P | 3B QwenVL（frozen） | ❌ | crop 图像 | bbox → crop |
| 推理 Teacher π_R | 3B QwenVL（RL 后迁移） | ✅ RL | 结构化证据文本 | 证据→推理→答案 |
| Student | 3B QwenVL | OPD 蒸馏 | 全图 + 问题 | 问题 + 答案 |
| λ router | 小网络 | ✅ | state features | scene graph → trajectory |

## 核心逻辑

### π_P 为什么不训练

π_P 和 Student 都是 3B QwenVL，但输入不同（crop vs 全图），同一 frozen 模型产出的分布天然不同 → KL ≠ 0 → 有意义的监督信号。这是 Vision-OPD 的核心思路，不需要额外训练。

### π_R 迁移

π_R 是纯文本推理（结构化证据 → 推理链 → 答案），不和图像交互。在 Vision-SR1-47K 上 RL 训完后，迁移到 GQA 场景图提取的结构化证据上直接可用。不需要和 π_P/OPD 共享视觉数据分布。

### 统一 OPD 平台：GQA

π_P 的 crop 来自 GQA bbox，λ router 的 trajectory 来自 GQA scene graph，OPD 训练在 GQA 问题上。三者同分布，不存在 π_P 先验退化问题。

## 数据集

### 主数据集：GQA（OPD + λ router + π_P crop）

| 项目 | 说明 |
|------|------|
| 来源 | GQA (Hudson & Manning, 2019) |
| 规模 | ~22M 问题，~113K 图像 |
| 标注 | scene graph（objects, attributes, relations）+ bbox + 问题 + 答案 |
| 用途 | π_P crop 输入、λ router trajectory 生成、Student OPD 训练 |
| 获取 | https://cs.stanford.edu/people/dorarad/gqa/download.html |

### π_R 训练数据集：Vision-SR1-47K

| 项目 | 说明 |
|------|------|
| 来源 | Vision-SR1 (arXiv:2508.19652)，GitHub: zli12321/Vision-SR1 |
| 规模 | ~47K RL 样本 + ~9K SFT 冷启动 |
| 标注 | `<visual perception>` + `<think>` + `<answer>` 三段式 |
| 用途 | π_R RL 训练（提取 visual perception 作为结构化证据输入） |
| 获取 | GitHub 公开 |

### 辅助数据

| 数据集 | 用途 | 备注 |
|--------|------|------|
| Visual Genome | GQA bbox 不够时补充 π_P crop 来源 | bbox + 区域描述 5.4M |
| ScienceQA | π_R 补充训练 | text evidence → reasoning |
| RefCOCO+ | π_P 视觉感知评估 | 禁止位置词，强制依赖视觉属性 |

---

## Final Benchmark（和训练数据源隔离）

| Benchmark | 测试能力 | 图像来源 |
|-----------|---------|---------|
| MMStar | 通用视觉推理 | 多种来源 |
| MMBench | 通用 VQA | 多种来源 |
| HallusionBench | 幻觉检测 | 合成/对抗 |
| MathVista | 数学视觉推理 | 数学图表 |
| MMMU | 跨领域推理 | 大学教材 |
| RealWorldQA | 真实场景 | 手机实拍 |
| OCRBench | OCR 能力 | 文字场景 |

全部非 GQA 源。GQA 只用于训练，不用于 evaluation（GQA 答案有同义词歧义问题，不适合评估）。

---

## 数据流

```
GQA (统一平台)
  ├── bbox → crop 图像 → π_P(frozen) → 视觉描述分布
  ├── scene graph → pseudo-optimal trajectory → λ router 训练
  └── 全图 + 问题 → Student OPD 训练（π_P + π_R 联合监督，λ 路由）

Vision-SR1-47K (π_R 训练，训完迁移)
  └── visual perception 文本 → π_R(RL) → 推理链 + 答案

Final Test: 7 benchmarks（非 GQA，非 Vision-SR1 源）
```

---

## 下一步

1. **下载 GQA** — 确认 scene graph + bbox + 问题格式，抽检 20 条
2. **下载 Vision-SR1-47K** — 确认 visual perception 分段可提取为结构化证据，抽检 20 条
3. **Benchmark contamination 检查** — GQA / Vision-SR1 来源和 final test 的 7 个 benchmark 是否有重叠
4. **如果 GQA bbox 精度不够** — Visual Genome 作为 π_P crop 补充
5. **如果 Vision-SR1 推理质量不够** — ScienceQA 作为 π_R 补充
