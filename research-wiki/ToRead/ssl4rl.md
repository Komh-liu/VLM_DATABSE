---
title: "SSL4RL: Revisiting Self-supervised Learning as Intrinsic Reward for Visual-Language Reasoning"
authors: Xiaojun Guo, Runyu Zhou, Yifei Wang, Qi Zhang, Chenheng Zhang, Stefanie Jegelka, Xiaohan Wang, Jiajun Chai, Guojun Yin, Wei Lin, Yisen Wang
venue: ICML 2026 Poster (Peking University, MIT, TUM, Meituan)
tags: [SSL, reward-design, GRPO, verifiable-reward, no-human-label]
---

## 核心问题

能不能**完全不用人类标注**来做 VLM 的 RL 微调？不用人类偏好、不用 AI 评判、不用数据标注？

## 核心方法: SSL 任务 → 可验证奖励

把经典 SSL 前置任务重新包装为 RL 奖励函数：

| SSL 任务 | 怎么变成奖励 | 为什么可验证 |
|----------|------------|------------|
| Rotation Prediction | 图旋转了 K×90°，模型预测 K。对了就 +1 | 旋转角度是确定的 |
| Jigsaw Puzzles | 图切成 2×2 块打乱，模型排列回正确顺序 | 排列有唯一解 |
| Contrastive Learning | 同一图的两个 view 产生相似 embedding | 正负对是构造的 |
| Patch Position | Mask 某个 patch，预测它的位置 (x,y) | 位置是已知的 |

### GRPO 训练
```
对每张图:
  构造 SSL 任务 (e.g. 旋转 90°)
  采样 N 个回答: "90°", "180°", "270°", "0°"
  奖励: 对 = 1, 错 = 0
  GRPO 组内归一化 → 更新
```

## 核心发现

1. **SSL 奖励有效**: +7.39% MMBench, +8.94% SEED-Bench (3B 模型)
2. **关系推理提升最大**: +39 pp on MMBench Relation Reasoning——SSL 任务天然需要理解空间关系
3. **金凤花原理 (Goldilocks Principle)**: SSL 任务难度必须匹配模型当前能力。太简单（区分 0°/180°）= 没学到东西。太难（区分 1°）= reward 全为 0 无法训练
4. **多种 SSL 任务组合更好**: 组合优于单一任务

## 为什么在你的阅读清单里

- **代码开源** → 你可以直接跑，不需要从零实现
- SSL 任务种类无限 → 你可以设计 "grounding 专用的 SSL 任务"
- "金凤花原理" 没有深入 → 自适应难度调整可以连接 curriculum learning (RuCL)
- 无人类标注 → 你的实验成本极低

## 可能的扩展方向
1. **Grounding-specific SSL**: 设计专门提升 grounding 的 SSL 前置任务（e.g. 预测物体相对位置、预测遮挡关系）
2. **Dynamic SSL selection**: 训练中根据模型能力自动切换 SSL 任务 → 解决金凤花原理
3. **SSL + rule-based reward 混合**: SSL 提供密集过程奖励 + IoU 提供稀疏终局奖励

## 阅读重点
- 4 种 SSL 任务的构造细节 (3.1-3.4)
- 金凤花原理的实验证据 (4.2-4.3)
- 代码结构（GitHub: PKU-ML/SSL4RL）
