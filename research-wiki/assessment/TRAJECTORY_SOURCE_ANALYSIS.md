# 假设修正分析：强模型构造轨迹 vs Scene Graph 自动生成

**分析前提**：TCTR 原本假设 GQA scene graph 可以**程序化足量构造**高质量轨迹。现在考虑另一种可能性：scene graph 只能提供骨架，真正的高质量轨迹需要**强模型（如 Qwen2.5-VL-72B）构造小批量（几千条）**。这会如何改变方法的优劣势？

---

## 1. 成本结构的变化

| 维度 | Scene Graph 自动生成 | 强模型构造小批量 |
|------|---------------------|-----------------|
| 单条轨迹成本 | ~$0（程序化） | ~$0.01-0.05（API call） |
| 总成本（5K 轨迹） | ~$0 | ~$50-250 |
| 可控性 | 低（模板固定） | 高（可 prompt 控制风格和格式） |
| 质量天花板 | 低（模板可能不自然） | 高（强模型推理更自然） |
| 扩展性 | 极高（覆盖 GQA 全量 22M） | 中等（受限于预算） |

**核心变化**：5K 轨迹的成本从 $0 变成 ~$100-250。这个级别对于研究项目是可接受的，但"免费"这个叙事优势消失了。

---

## 2. 竞争格局的重新评估

### 方法成本对比

```
Method            External data needed?     Cost structure
──────────────────────────────────────────────────────────────
DOPD              不需要 (advantage gap)     $0
ViGOS             不需要 (格式解析)          $0
Decomposed OPD    不需要 (固定优先)          $0
HEED              不需要 (patch density)     $0
MoCA              不需要 (modality classifier) $0
──────────────────────────────────────────────────────────────
TCTR (scene graph) scene graph → 程序化生成  $0 (但依赖 GQA 标注)
TCTR (强模型)     强模型构造轨迹             $100-250/5K
```

原来 TCTR 的"免费 trajectory"是唯一一个**不需要外部昂贵资源**的 supervised routing 方法。如果变成强模型构造，TCTR 成为唯一一个**需要外部模型调用**的方法 —— 这是一个定位倒退。

### 但问题没有表面那么简单

审稿人会问的关键问题链：

```
Q1: 为什么要用强模型构造成本？
    → 因为 scene graph 模板不够好
    → 为什么不够好？
    → 因为模板轨迹不自然，routing 质量提升有限

Q2: 那这个路由方法本质上是"用一个强模型生成数据来训练另一个模型"？
    → 对，但这是许多 KD 方法的通用范式

Q3: DOPD 不需要任何外部数据，为什么比你差？
    → 这是一个实验问题，需要证明 TCTR > DOPD
```

**关键风险**：一旦引入强模型依赖，TCTR 必须**显著**优于 DOPD 才能 justify 这个额外成本。scene graph 免费版本只需要"不显著差于 DOPD + 提供额外价值"。

---

## 3. 剩余的独特优势

即使改用强模型，TCTR 仍有几个**场景图无法替代**的优势：

### 3.1 Ground-truth 路由标签

场景图的真正价值不仅是免费，更是**带来 ground-truth operation 标签**。强模型生成的轨迹没有 scene graph 的 operation-level 标注，但 TCTR 的 soft operation prior $\mu_{\text{SG}}(o_t)$ 需要 operation 类型。

**解决方案**：强模型生成轨迹后，可以用 scene graph 回溯标注 operation。即：

```
强模型生成轨迹 → 用 scene graph 验证每一步 → 
→ 如果匹配，标注 operation 类型 + 计算 teacher support
→ 如果不匹配，丢弃（避免幻觉）
```

这保留了 scene graph 的校准价值，只是轨迹内容由强模型生成而非模板。

### 3.2 精确的 Error Attribution

即使轨迹由强模型生成，scene graph 的 error decomposition（Section 8.7）仍然成立：

```
Scene graph → Perception error / Reasoning error / Evidence-wrong
```

这是 DOPD、ViGOS、MoCA 都没有的——它们只能在 model distribution 层面分析，不能在 ground-truth 层面归因。这是**TCTR 独有的诊断能力**。

### 3.3 跨数据集迁移（可能仍然成立）

如果 router 学到的是 teacher-support 和 teacher-disagreement 的模式（而不是 scene graph 模板），那么迁移到 OK-VQA 仍然可能成立。但需要实验验证。

---

## 4. 三种轨迹来源的对比

```
                    Programmatic SG     Strong model      Teacher-generated
Trajectory source   template+SG labels  强模型+SG验证      Teacher π itself
─────────────────────────────────────────────────────────────────────────────
Naturalness         ❌ 低                ✅ 高              ✅ 高
Cost                ✅ 免费              ⚠️ ~$100/5K        ✅ 免费（但需要 teacher rollout）
Operation labels    ✅ 自带              ✅ 可回溯标注       ❌ 无（需额外标注）
Routing λ* labels   ✅ 自动              ✅ 自动             ❌ 无
SG faithfulness     ✅ 100%              ⚠️ 需过滤           ❌ 不确定
Scaling              ✅ 无限              ❌ 受预算限制       ✅ 可用 teacher 采样
Novelty (vs DOPD)    ✅ 强（独特资源）     ⚠️ 中              ❌ 弱（标准 self-play）
```

**Teacher-generated** 列特别值得注意：如果用 teacher 本身来生成轨迹（teacher rollout + 筛选正确的），成本也是 $0，但失去了 scene graph 的 operation 标注能力。

---

## 5. 场景图的核心不可替代价值

在仔细分析后，scene graph 的真正价值不是免费，而是：

> **能在生成轨迹的同时，提供每个 token 的 ground-truth operation 类型和 routing 标签。**

DOPD 和 ViGOS 的 routing/分类都是基于**启发式或格式假设**，不是 ground-truth。TCTR 的 $\lambda_t^*$ 构建使用了两个真实信号：
1. Soft operation prior $\mu_{\text{SG}}(o_t)$ — 来自 scene graph 的 operation 标注
2. Teacher support $p_P^t, p_R^t$ — 来自 teacher 对 ground-truth token 的实际概率

强模型生成可以解决"自然度"问题，但**必须保留 scene graph 的回溯标注能力**。

---

## 6. 修正后的策略建议

### 最佳折中方案：Hybrid Pipeline

```
Step 1: 用 scene graph 生成 skeleton + 自动标注 operation 类型 (免费)
Step 2: 用强模型 (Qwen2.5-VL-72B) 将 skeleton paraphrase 成自然轨迹 (小成本)
Step 3: 用 scene graph 回溯验证语义一致性 + 标注 operation 归属 (自动)
Step 4: 计算 teacher support 和 λ* target (自动)
Step 5: 训练 router + dense OPD (标准流程)
```

这个方案：
- 保留 scene graph 的**标注能力**（operation, routing, attribution）
- 解决模板轨迹的**自然度问题**
- 成本可控（5K 轨迹 × ~$0.02 = ~$100）
- 可自动验证（scene graph 一致性检查）

### 和 DOPD 的叙事对比更新

```
DOPD: 无需外部数据，advantage gap 计算路由
      优势：干净，自包含
      劣势：启发式路由，离散 4 类，没有真正的"最优"定义

TCTR (hybrid): 
      用强模型生成少量自然轨迹 + scene graph 标注最优路由
      优势：有 ground-truth 定义 "什么是最优路由"
      劣势：需要少量外部数据 + 依赖 scene graph
```

这个叙事比"scene graph 免费轨迹"更弱，但比"variational"诚实。问题变成：

> **有 ground-truth 标签校准的路由，是否比启发式路由（DOPD）更好？**

这是一个实验问题，不是先验问题。如果实验答案是 Yes，故事仍然成立。

---

## 7. 优劣势总结

### 优势（仍成立）

| 优势 | 强度变化 | 说明 |
|------|---------|------|
| Ground-truth routing labels | ✅ 不变 | scene graph 仍提供唯一的 operation-level 标注 |
| Token-level λ 连续性 | ✅ 不变 | 比 DOPD 4-regime 细粒度，比 ViGOS {0,1} 灵活 |
| Error attribution 能力 | ✅ 不变 | scene graph 仍支持精确的 error decomposition |
| 跨数据集迁移潜力 | ✅ 不变 | router 特征和训练不依赖轨迹来源 |
| TCTR + DOPD 互补 | ✅ 不变 | 两个独立的路由维度，可叠加 |

### 劣势（新增或加重）

| 劣势 | 严重程度 | 说明 |
|------|---------|------|
| "免费数据" 不再是卖点 | ⚠️ **MAJOR** | 现在需要 API 成本，DOPD 完全不需要 |
| 强模型依赖 | ⚠️ MAJOR | 没有免费午餐了，依赖外部模型生成质量 |
| 可复现性 | ⚠️ MODERATE | 强模型 API 可能变化，黑盒生成不可控 |
| 对比 DOPD 的负担更重 | ⚠️ CRITICAL | 必须证明 TCTR > DOPD 且差距够大 justify 外部依赖 |
| Novelty score 下降 | ⚠️ MODERATE | 从 6.0 → ~5.0-5.5 |

### Novelty Score 修正

如果我之前评估 novelty 6.0/10 是基于"免费场景图轨迹"这个假设，那么修正后：

| 维度 | 之前 | 修正后 | 原因 |
|------|------|--------|------|
| 场景图 → 轨迹 pipeline | HIGH | MEDIUM-HIGH | 仍独特，但不再是免费资源利用 |
| 连续 λ routing | MEDIUM | MEDIUM | 不受影响 |
| Soft operation prior | MEDIUM-HIGH | MEDIUM | 仍独特，但只是标注方式，不是方法核心 |
| 跨数据集迁移 | MEDIUM | MEDIUM-LOW | 强模型生成更依赖 GQA 格式 |
| **Overall** | **6.0/10** | **~5.0-5.5/10** | |

---

## 8. 建议

**如果场景图可以构造足够好的轨迹**（经过 paraphrasing 后自然度达到 70%+ 真实水平），优先走 scene graph 路线。$0 成本和独特的资源利用叙事对审稿人来说是一个干净的卖点。

**如果必须强模型才能得到可用轨迹**，建议两个调整：

1. **叙事重心从"免费资源"转移到"最优路由标定"**：
   ```
   "We use a small set of high-quality reasoning trajectories, 
    labeled by scene graph operations, to define what optimal 
    teacher routing looks like. This ground-truth calibration 
    generalizes to dense OPD and outperforms heuristic routing."
   ```

2. **强调 scene graph 的诊断价值，而不是轨迹生成价值**：
   ```
   TCTR 的优势不是在生成轨迹（强模型可以做），
   而是在有 ground-truth 的 routing loss + error attribution。
   ```

3. **实验上必须 pass DOPD gate**：
   - TCTR > DOPD（有强模型数据）
   - TCTR + DOPD > DOPD（路由互补性）
   - 如果不能，方法的存在理由就弱了
