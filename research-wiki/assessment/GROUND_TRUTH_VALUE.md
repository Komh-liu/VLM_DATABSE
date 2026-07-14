# 核心价值再分析：Ground-Truth 路由标签

> **你的观点**：TCTR 的独特价值不是轨迹便宜，而是 scene graph 可以构造**近乎绝对正确的路径**——它给出了 teacher routing 的 ground-truth 最优定义。

---

## 1. 这才是 TCTR 真正不可替代的东西

对比其他方法的路由信号来源：

| 方法 | 路由信号 | 信号性质 | 可靠性 |
|------|---------|---------|--------|
| **DOPD** | $\|\log\pi_T - \log\pi_S\|$ (advantage gap) | 启发式 | 训练中估计，可能偏移 |
| **ViGOS** | 输出格式解析 (`<description>`/`<think>`) | 格式假设 | 依赖用户遵守格式 |
| **Decomposed OPD** | 固定优先 visual | 先验假设 | 所有 token 一刀切 |
| **MoCA** | modality classifier confidence | 启发式 | 分类器本身有误差 |
| **Confidence routing** | teacher entropy | 启发式 | confidently wrong 时失效 |
| **TCTR** | scene graph → 最优操作序列 | **Ground-truth** | **可验证正确** |

DOPD 的 advantage gap 是在训练中计算的相对量，本质上是"teacher 比 student 自信多少"。这个信号可能正确也可能错误——它没有外部校验。

TCTR 的路由标签来自 scene graph 的结构：
```
Scene graph 说: 这是 object recognition step  → λ* 应偏高
Scene graph 说: 这是 relation reasoning step → λ* 应偏低
```

这不是启发式，不是估计，不是分类器——它是**结构事实**。

## 2. 这意味着什么

TCTR 不是在"猜测什么时候该用哪个 teacher"，而是在**回答一个可以被精确回答的问题**：

> 给定一个最优推理步骤，哪个 teacher 更擅长这个步骤对应的能力？

**Perception teacher** 的分布应该更符合 scene graph 中 object/attribute 操作所在的 token；
**Reasoning teacher** 的分布应该更符合 relation/logic 操作所在的 token。

而 DOPD 的 advantage gap 回答的是：

> 当前 token 上，teacher 和 student 谁更自信？

这是两个完全不同的信息层次。

## 3. 为什么即使轨迹不"自然"也没关系

TCTR 不需要轨迹读起来像人写的。它只需要轨迹的 **routing label 是正确的**。

```
"locate red chair"                                        → λ* = 0.8 (perception)
  ↑ scene graph 说这是一个 object identification step
  ↑ 无论这句话被写成什么自然语言

"compare size of chair and table"                         → λ* = 0.3 (reasoning)
  ↑ scene graph 说这是一个 comparison step
```

模板轨迹 vs 强模型 paraphrase 的区别只是**文本风格**，不是**路由正确性**。如果 routing label 由 scene graph 保证，轨迹文本的"不自然"不影响路由学习质量。router 的输入 $z(s)$ 包含 teacher logits 和 hidden states，不依赖轨迹文本的自然度。

## 4. 这如何改变对比 DOPD 的叙事

修正后的对比框架：

```
DOPD: 优势在于不需要任何外部资源
      局限：路由信号是启发式 (advantage gap)，没有最优性保证

TCTR: 优势在于有 ground-truth 路由标签
      局限：需要 scene graph 标注（但 GQA 已有）

问题变成：
  有 ground-truth 监督的 routing 是否优于启发式 routing？
  
  这是一个实验问题，但 TCTR 有一个 DOPD 永远没有的优势：
  它的路由训练目标有外部可验证的正确性基准。
```

这比"免费数据"的叙事更强。免费数据只是一个资源效率论点。**Ground-truth 标签是一个信息质量论点**——资源效率可以被质疑（"免费但低质量"），但信息质量不容易被质疑。

## 5. Novelty 重新评估

如果核心 claim 是 **ground-truth 路由标签** 而非 **免费轨迹生成**：

| 维度 | Novelty | 理由 |
|------|---------|------|
| Scene graph → 最优路由标签 | **HIGH** | 没有任何方法做这个 |
| 基于 ground-truth 的 λ* 标定 | **HIGH** | DOPD/ViGOS/MoCA 全是启发式 |
| 连续 soft routing | MEDIUM | 与 DOPD 共享 |
| Sparse-to-dense OPD 泛化 | MEDIUM-LOW | 标准技术 |
| 跨数据集迁移 | MEDIUM | 需实验验证 |

**Overall: 6.5/10**（从 6.0 调回并略升——因为"ground-truth 路由标签"比"免费轨迹"更根本）
