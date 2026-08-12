---
type: paper
node_id: paper:optimizer_sgd_adam
title: "SGD 与 Adam 优化器详解"
authors: []
year: 2026
venue: "Note"
external_ids:
  arxiv: null
  doi: null
  s2: null
tags: ["optimizer", "SGD", "Adam", "deep-learning", "tutorial"]
added: 2026-08-11T00:00:00Z
---

# SGD 与 Adam 优化器详解

## One-line thesis

> 从朴素 SGD 到 Adam，优化器的每一次演进都引入了一个必须跨 step 持久化的状态变量，这些递推依赖的状态使得 Adam 在混合精度训练中占据总显存的 75%，成为分布式训练显存优化的首要目标。

---

## 1. 背景：优化问题的结构

### 经验风险最小化

深度学习训练的目标是求解如下非凸随机优化问题：

$$
\theta^* = \arg\min_{\theta \in \mathbb{R}^d} \; \mathcal{L}(\theta) = \arg\min_{\theta} \; \mathbb{E}_{(x,y) \sim \mathcal{D}} \left[ \ell(f_\theta(x), y) \right]
$$

在小批量随机梯度下降（Mini-batch SGD）的框架下，每个 step 使用一个大小为 $B$ 的 mini-batch 计算梯度估计：

$$
g_t = \frac{1}{B} \sum_{i=1}^{B} \nabla_\theta \ell(f_\theta(x_i), y_i)
$$

### 病态 Hessian 问题

神经网络损失函数的一个典型特征是 **Hessian 矩阵的病态条件数**——不同参数维度的曲率差异极大：

$$
\text{cond}(H) = \frac{\lambda_{\max}(H)}{\lambda_{\min}(H)} \gg 1
$$

这意味着损失曲面在参数空间中呈现极狭长的"峡谷"形状。某些方向梯度分量极大（陡峭壁面），某些方向极小（平坦谷底）。固定学习率 $\eta$ 对于陡峭方向过大会导致发散，对于平坦方向过小则收敛缓慢——这是朴素 SGD 的根本缺陷。

---

## 2. 朴素 SGD

### 更新规则

$$
\theta_{t+1} = \theta_t - \eta \cdot g_t
$$

其中 $g_t = \nabla_\theta \mathcal{L}_{\text{batch}}(\theta_t)$ 是当前 mini-batch 上的随机梯度估计。

### 状态存储

朴素 SGD 是**无状态（stateless）**优化器：更新仅依赖当前步的梯度 $g_t$，梯度使用完毕后即可释放。

| 持久状态 | 精度 | 大小 (参数量 $d = 3\text{B}$) |
|----------|------|-------------------------------|
| $\theta$ | fp32 | $3 \times 10^9 \times 4 = 12 \text{ GB}$ |

### 局限性分析

**(a) 统一学习率假设。** 所有参数共享同一个标量步长 $\eta$，完全忽略了参数空间不同维度的尺度差异性。当 Hessian 条件数较大时，必须设置极小的学习率以避免在陡峭方向发散，导致在平坦方向收敛极慢。

**(b) 梯度噪声。** Mini-batch 梯度 $g_t$ 是真实梯度 $\nabla\mathcal{L}(\theta_t)$ 的随机估计，方差 $\propto 1/B$。在高维空间中，纯 SGD 的收敛轨迹存在剧烈的 zig-zag 振荡。

**(c) 收敛率。** 对于强凸光滑目标，SGD 的期望次优性以 $\mathcal{O}(1/t)$ 的速度下降；但在非凸设定下，SGD 可能困在鞍点（saddle point）附近，因为鞍点处梯度接近零但 Hessian 有负特征值，仅靠一阶信息难以逃脱。

---

## 3. SGD with Momentum（动量 SGD）

### 动机

动量方法最早由 Polyak (1964) 在经典优化中提出。在深度学习中，动量对 SGD 的改进体现在两个层面：

1. **沿峡谷底部加速收敛**：在低曲率方向（平坦谷底），动量累积历史梯度信号，使得该方向的等效步长放大到 $\eta / (1-\beta)$，加速前进。
2. **抑制高频振荡**：在高曲率方向（陡峭壁面），正负交替的梯度分量在指数加权平均中相互抵消，减少 zig-zag 幅度。

### 更新规则

$$
\begin{aligned}
m_t &= \beta \cdot m_{t-1} + (1 - \beta) \cdot g_t \\[4pt]
\theta_{t+1} &= \theta_t - \eta \cdot m_t
\end{aligned}
$$

其中 $\beta \in [0, 1)$，通常设定为 $0.9$。

### 递推展开

将 $m_t$ 的递推展开到初始步，可以清楚看到动量的指数衰减窗口：

$$
m_t = (1-\beta) \cdot g_t + (1-\beta)\beta \cdot g_{t-1} + (1-\beta)\beta^2 \cdot g_{t-2} + \cdots
$$

当 $\beta = 0.9$，$(0.9)^{10} \approx 0.35$，即历史梯度的有效窗口约为 $1/(1-\beta) = 10$ 步。梯度方向一致的步累积放大，方向相反的步相互抵消。

### 状态存储增量

动量引入了一项与 $\theta$ 等大的持久状态 $m_t \in \mathbb{R}^d$：

| 持久状态 | 大小 |
|----------|------|
| $\theta$ (fp32) | 12 GB |
| $m_t$ (fp32) | 12 GB |
| **合计** | **24 GB** |

### 为什么 $m_t$ 无法丢弃

$m_t$ 是时刻 $t$ 之前所有梯度的指数衰减加权和。任意时刻 $t+1$ 的计算需要 $m_t$ 作为输入，丢弃则破坏了递推结构，无法恢复。这是**有状态优化器**的本质代价。

---

## 4. AdaGrad：逐参数自适应学习率

### 动机

Momentum SGD 解决了平坦方向的加速问题，但仍未解决**统一全局学习率**的根本缺陷。Duchi et al. (2011) 提出 AdaGrad，核心洞见是：每个参数的学习率应反比于其历史梯度的大小。

### 更新规则

$$
\begin{aligned}
v_t &= v_{t-1} + g_t^2 \qquad (\text{逐元素平方累加}) \\[4pt]
\theta_{t+1} &= \theta_t - \frac{\eta}{\sqrt{v_t} + \varepsilon} \odot g_t
\end{aligned}
$$

其中 $\varepsilon$ 是微小的平滑常数（通常 $10^{-8}$），防止除零；$\odot$ 表示逐元素乘法。

分母 $\sqrt{v_t}$ 为每个参数提供一个独立的、基于历史梯度均方值的缩放因子。历史梯度大的参数自动获得较小的有效学习率，反之亦然。

### 致命缺陷：学习率单调衰减

$v_t$ 是梯度平方的**无限累加**，严格单调递增：

$$
v_{t+1} = v_t + g_t^2 > v_t
$$

因此分母 $\sqrt{v_t}$ 随时间单调递增，有效学习率 $\eta_{\text{eff}}^{(i)} = \eta / \sqrt{v_t^{(i)} + \varepsilon}$ 单调衰减逼近零。在凸优化中这有理论收敛性保证（$\mathcal{O}(1/\sqrt{T})$ regret bound），但在非凸深度神经网络中，训练尚未完成学习率便已降至零——参数更新提前停滞。

---

## 5. RMSprop：指数移动平均替代累加

### 动机

Tieleman & Hinton (2012) 提出 RMSprop，将 AdaGrad 的"无限窗口累加"修改为"有限窗口指数移动平均"：

$$
v_t = \beta_2 \cdot v_{t-1} + (1 - \beta_2) \cdot g_t^2
$$

其中 $\beta_2 = 0.999$。展开递推：

$$
v_t = (1-\beta_2) \cdot g_t^2 + (1-\beta_2)\beta_2 \cdot g_{t-1}^2 + (1-\beta_2)\beta_2^2 \cdot g_{t-2}^2 + \cdots
$$

窗口长度为 $1/(1-\beta_2) = 1000$ 步。超出窗口的历史梯度平方对 $v_t$ 的贡献指数衰减至零，$v_t$ 不再单调递增。在收敛阶段，$v_t$ 稳定在梯度平方的稳态均值附近，有效学习率不再持续衰减。

### 与 Momentum SGD 的对称性

| | Momentum SGD (一阶) | RMSprop (二阶) |
|---|---|---|
| 维护的状态 | $m_t$：梯度的一阶矩 | $v_t$：梯度平方的二阶矩 |
| 递推形式 | $m_t = \beta m_{t-1} + (1-\beta)g_t$ | $v_t = \beta_2 v_{t-1} + (1-\beta_2)g_t^2$ |
| 作用 | 平滑更新方向 | 逐参数缩放步长 |
| 窗口长度 | $\sim 1/(1-\beta)$ | $\sim 1/(1-\beta_2)$ |

两者使用完全相同的指数移动平均机制，但作用于不同的统计量。

---

## 6. Adam：一阶矩 + 二阶矩 + 偏差修正

### 动机

Kingma & Ba (2014) 提出 Adam（**Ada**ptive **M**oment Estimation），将 Momentum 的方向平滑（一阶矩）和 RMSprop 的逐参数步长缩放（二阶矩）统一到一个框架中，并引入偏差修正以解决初始化阶段的估计偏差。

### 完整更新规则

**Step 1: 计算一阶矩和二阶矩**

$$
\begin{aligned}
m_t &= \beta_1 \cdot m_{t-1} + (1 - \beta_1) \cdot g_t \\[4pt]
v_t &= \beta_2 \cdot v_{t-1} + (1 - \beta_2) \cdot g_t^2
\end{aligned}
$$

其中 $g_t^2 = g_t \odot g_t$（逐元素平方）。

**Step 2: 偏差修正**

初始化 $m_0 = 0$, $v_0 = 0$。在训练初期（$t$ 较小时），$m_t$ 和 $v_t$ 是真实一阶矩 $\mathbb{E}[g]$ 和二阶矩 $\mathbb{E}[g^2]$ 的有偏估计——因为初始零向量的权重 $(1-\beta^t)$ 不足，EMA 被显著低估计。偏差修正式展开如下：

$$
\begin{aligned}
m_t &= (1-\beta_1) \sum_{i=1}^{t} \beta_1^{t-i} \cdot g_i \\[4pt]
\mathbb{E}[m_t] &= \mathbb{E}[g] \cdot (1-\beta_1^t) \quad (\text{稳态假设 } \mathbb{E}[g_i] = \mathbb{E}[g])
\end{aligned}
$$

因此无偏估计为 $\hat{m}_t = m_t / (1 - \beta_1^t)$。同理 $\hat{v}_t = v_t / (1 - \beta_2^t)$：

$$
\hat{m}_t = \frac{m_t}{1 - \beta_1^t}, \qquad \hat{v}_t = \frac{v_t}{1 - \beta_2^t}
$$

**Step 3: 参数更新**

$$
\theta_{t+1} = \theta_t - \eta \cdot \frac{\hat{m}_t}{\sqrt{\hat{v}_t} + \varepsilon}
$$

### 超参数及语义

| 超参数 | 典型值 | 语义 |
|--------|--------|------|
| $\eta$ | $10^{-3}$ | 全局学习率（通常用余弦退火或 warmup schedule） |
| $\beta_1$ | $0.9$ | 一阶矩衰减系数，控制动量平滑窗口 $= 1/(1-\beta_1) = 10$ 步 |
| $\beta_2$ | $0.999$ | 二阶矩衰减系数，控制步长自适应窗口 $= 1/(1-\beta_2) = 1000$ 步 |
| $\varepsilon$ | $10^{-8}$ | 数值稳定常数 |

### 为什么 $\beta_1 < \beta_2$

一阶矩 $m_t$ 要求较短的响应窗口（~10 步）以快速适应梯度方向的短期变化；二阶矩 $v_t$ 需要较长的统计窗口（~1000 步）以准确估计梯度幅度的稳态分布。若 $\beta_2$ 太小（窗口太短），步长缩放因子会根据单个 mini-batch 的噪声剧烈波动，破坏训练的稳定性。

### 完整性：Adam 的三个不可约简组件

| 组件 | 去除后的退化为 | 失去的能力 |
|------|---------------|-----------|
| $m_t$（一阶矩） | RMSprop（SGD with $\beta_2$） | 方向平滑与动量加速，易困于鞍点 |
| $v_t$（二阶矩） | SGD with Momentum | 逐参数自适应步长，病态曲率下收敛慢 |
| 偏差修正 $\hat{m}, \hat{v}$ | 无修正的 Adam | 初始训练阶段更新方向严重偏差，warmup 阶段不稳定 |

---

## 7. 状态存储：Adam 的显存账本

### 全量训练（混合精度：fp16 前向反向 + fp32 权重更新）

在混合精度训练方案下（Micikevicius et al., 2018），前向和反向传播使用 fp16 以节省计算和显存，但权重更新环节必须使用 fp32 以保证累加精度。

对于参数量 $d = 3 \times 10^9$（3B）的模型：

| 状态变量 | 精度 | 字节数 | 是否持久 |
|----------|------|--------|----------|
| $\theta_{\text{fp16}}$（前向反向用） | fp16 | $2d = 6 \text{ GB}$ | ✓ |
| $\theta_{\text{fp32}}$（master 副本，用于累加微小更新） | fp32 | $4d = 12 \text{ GB}$ | ✓ |
| $m_t$（Adam 一阶矩） | fp32 | $4d = 12 \text{ GB}$ | ✓ |
| $v_t$（Adam 二阶矩） | fp32 | $4d = 12 \text{ GB}$ | ✓ |
| $g_t$（当前步梯度，reduce 后释放） | fp16 | $2d = 6 \text{ GB}$ | ✗ 每步释放 |
| 激活值 | fp16 | 取决于 batch/seq_len | ✗ |

### 为什么需要 fp32 master 副本？

fp16 的有效精度约为 $10^{-3.3}$ 量级（10 位尾数加 5 位指数）。Adam 中单步参数更新量 $\Delta\theta^{(i)} = \eta \cdot \hat{m}_t^{(i)} / (\sqrt{\hat{v}_t^{(i)}} + \varepsilon)$ 通常在 $10^{-4}$ 至 $10^{-6}$ 量级。如果在 fp16 表示中直接累加：

$$
\theta_{\text{fp16}} \leftarrow \theta_{\text{fp16}} + \Delta\theta_{\text{fp16}}
$$

那么 $\Delta\theta \ll \theta \cdot 2^{-10}$ 时，浮点加法发生 **catastrophic cancellation**——增量直接被吸收消失。必须维护一个 fp32 master 副本，在 fp32 下完成累加后再转换回 fp16 供下一步计算。

### 总账

```
持久优化器状态 = θ_fp32 + m_t + v_t = 12 + 12 + 12 = 36 GB
计算用参数     = θ_fp16                   =  6 GB
────────────────────────────────────────────────────
持久总显存     =                         = 42 GB  （不含激活值）
优化器状态占比 = 36 / 48 ≈ 75%           （相对峰值显存含梯度~48GB）
```

### 不同优化器的持久状态演化

| 优化器 | 持久状态变量 | 持久大小 ($d=3\text{B}$) | 相对朴素 SGD |
|--------|-------------|--------------------------|-------------|
| 朴素 SGD | $\theta$ | 12 GB | 1× |
| SGD + Momentum | $\theta, m_t$ | 24 GB | 2× |
| AdaGrad / RMSprop | $\theta, v_t$ | 24 GB | 2× |
| Adam (fp32) | $\theta, m_t, v_t$ | 36 GB | 3× |
| Adam (混合精度) | $\theta_{\text{fp16}}, \theta_{\text{fp32}}, m_t, v_t$ | 42 GB | 3.5× |

**Adam 的 36 GB 优化器状态（$m_t, v_t, \theta_{\text{fp32}}$）不是设计缺陷，而是其自适应能力的必然代价。**

---

## 8. AdamW：权重衰减与自适应学习率的解耦

### 问题

原始 Adam 论文提出在损失函数中加入 L2 正则化 $\lambda \|\theta\|_2^2$。其梯度为 $\nabla\mathcal{L} + \lambda\theta$，代入 Adam 更新后：

$$
\theta_{t+1} = \theta_t - \eta \cdot \frac{\hat{m}_t + \lambda\theta_t}{\sqrt{\hat{v}_t} + \varepsilon}
$$

由于分母 $\sqrt{\hat{v}_t}$ 是逐参数的，L2 正则化被自适应学习率**扭曲**——梯度较大的参数其 L2 惩罚也被缩小，破坏了正则化的一致性。Loshchilov & Hutter (2019) 提出 AdamW，将权重衰减从自适应学习率中解耦：

$$
\theta_{t+1} = (1 - \eta\lambda) \cdot \theta_t - \eta \cdot \frac{\hat{m}_t}{\sqrt{\hat{v}_t} + \varepsilon}
$$

权重衰减 $-\eta\lambda\theta_t$ 独立于自适应项，不增加额外的持久状态。现代大语言模型训练（GPT、LLaMA 等）普遍使用 AdamW。

---

## 9. 分布式训练视角：为什么优化器状态是首要瓶颈

### 通信 vs 显存的权衡

在数据并行中：
- **梯度 $g_t$**：可以通过 all-reduce 通信后丢弃（每步一次通信）
- **优化器状态 $m_t, v_t, \theta_{\text{fp32}}$**：在所有 worker 之间**完全复制**，无通信但显存冗余 $N$ 倍

对于 175B 参数规模（GPT-3 量级），Adam 优化器状态在纯数据并行每张 GPU 上占用：

$$
175 \times 10^9 \times 12 \text{ bytes} = 2.1 \text{ TB}
$$

这远超任何单 GPU 显存。ZeRO 系列优化的核心正是：**将这 36 GB / 2.1 TB 的优化器状态在数据并行维度上进行分片（partition），将内存从 N× 冗余降为 1×，而通信复杂度不变**。

### ZeRO 的分片层次

| 级别 | 分片内容 | 持久显存 (per GPU, 3B/8GPUs) | 通信量 vs 标准 DP |
|------|---------|------------------------------|-------------------|
| 标准 DP | 无 | 42 GB | 1× |
| ZeRO-1 | $m_t, v_t, \theta_{\text{fp32}}$ | 42/8 = 5.25 GB | 1×（无额外通信） |
| ZeRO-2 | + $g_t$ | $\approx 4.5$ GB | 1× |
| ZeRO-3 | + $\theta_{\text{fp16}}$ | $\approx 3.75$ GB | 1.5× |

ZeRO-1 的通信量与标准数据并行完全一致（$2 \times |\theta|$），仅仅将 all-reduce 替换为 reduce-scatter + all-gather，但消除了优化器状态的 $N$ 倍冗余——这是零额外通信代价换 4× 显存节省。

### Offload：利用 CPU-DRAM

Adam 更新的计算复杂度为 $\mathcal{O}(d)$（逐元素运算），与 GPU 密集的前向/反向传播（$\mathcal{O}(d^2)$ 量级矩阵乘法）形成数个数量级的差异。ZeRO-Offload 利用这一点，将轻量级的优化器状态和更新步骤迁移至 CPU 内存：

- GPU 保留 fp16 模型参数和激活值，承担前向反向的矩阵运算
- CPU 内存存储 $m_t, v_t, \theta_{\text{fp32}}$，承担 Adam 更新和权重衰减
- PCIe 带宽仅需传输每步的梯度（GPU→CPU）和更新后的权重（CPU→GPU）

由于传输的数据量和计算负载远小于前向反向，GPU 和 CPU 在时间线上**高度重叠并行**，训练吞吐几乎不受影响。

---

## 总结

| 演进 | 引入的状态 | 解决的问题 | 持久显存 (3B) |
|------|-----------|-----------|--------------|
| 朴素 SGD | — | — | 12 GB |
| + Momentum | $m_t$ | 病态曲率下的方向加速 | 24 GB |
| + RMSprop | $v_t$ | 逐参数自适应学习率 | 24 GB |
| Adam | $m_t + v_t$ | 方向平滑 + 步长自适应 | 36 GB |
| Adam + 混合精度 | $m_t + v_t + \theta_{\text{fp32}}$ | 低精度加速 + 高精度累加 | 42 GB |

Adam 中的 $m_t$、$v_t$、$\theta_{\text{fp32}}$ 三者构成了优化器状态的"三座大山"。它们各自基于不可约简的递推依赖（$m_t$ 依赖 $m_{t-1}$，$v_t$ 依赖 $v_{t-1}$，$\theta_{\text{fp32}}$ 依赖高精度累加），使得 Adam 相比朴素 SGD 需要 **3.5×** 的持久显存。

## 参考文献

[1] Sutskever, I., Martens, J., Dahl, G., & Hinton, G. (2013). On the importance of initialization and momentum in deep learning. *ICML*.

[2] Duchi, J., Hazan, E., & Singer, Y. (2011). Adaptive subgradient methods for online learning and stochastic optimization. *JMLR*.

[3] Tieleman, T. & Hinton, G. (2012). Lecture 6.5: RMSprop. *Coursera: Neural Networks for Machine Learning*.

[4] Kingma, D. P., & Ba, J. (2014). Adam: A method for stochastic optimization. *arXiv:1412.6980*.

[5] Loshchilov, I., & Hutter, F. (2019). Decoupled weight decay regularization. *ICLR*.

[6] Micikevicius, P., et al. (2018). Mixed precision training. *ICLR*.

[7] Rajbhandari, S., Rasley, J., Ruwase, O., & He, Y. (2020). ZeRO: Memory optimizations toward training trillion parameter models. *SC20*.

[8] Ren, J., et al. (2021). ZeRO-Offload: Democratizing billion-scale model training. *ATC*.
