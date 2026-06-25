# 多模态 RL 相关综述论文

## 1. Reinforced MLLM: A Survey on RL-Based Reasoning in Multimodal LLMs

> **发布时间**: 2025-04 (arXiv:2504.21277)
> **作者**: Zhou et al.
> **状态**: 预印本

### 覆盖内容
- 系统性综述 RL 训练多模态大模型的推理能力
- 分类：value-model-free (GRPO, RLOO) vs value-model-based (PPO, RLHF)
- 奖励机制：rule-based verifiable reward, LLM-as-judge, preference-based
- Benchmark 总览：MathVista, MMMU, MMStar, HallusionBench 等
- 公开问题：训练稳定性、奖励 hacking、模态失衡

### 适合谁读
入门首选。读完能建立 RL+MLLM 的全局认知。

---

## 2. A Survey on RL-Powered Visual Reasoning for Multimodal Models

> **发布时间**: 2026-03 (ACM)
> **作者**: Wentao Dai et al.

### 覆盖内容
- 四部分框架：视觉交互模式 → RL 训练范式 → 奖励设计 → 应用领域
- "Thinking with Images" 理念：MLLM 通过 RL 主动探索视觉证据
- 奖励设计是核心关注点：ORM、LLM-as-judge、组合奖励函数
- 视觉交互模式扩展：从 tool-use 到 native zooming/grounding

### 适合谁读
对 reward design 和视觉交互模式感兴趣时读。

---

## 3. Perception, Reason, Think, and Plan: A Survey on Large Multimodal Reasoning Models

> **发布时间**: 2025-05 (arXiv:2505.04921)
> **作者**: Li et al.

### 覆盖内容
- 四阶段发展路线图：Perception → Reason → Think → Plan
- 引入 "Native Large Multimodal Reasoning Models" (N-LMRMs) 概念
- MCoT (Multimodal Chain-of-Thought) 和 multimodal RL 的讨论
- 从 SFT 到 RL 的训练范式演变

### 适合谁读
想了解 MLLM 推理能力发展的宏观视角时读。

---

## 4. The Evolving Landscape of LLM- and VLM-Integrated Reinforcement Learning

> **发布时间**: 2025 (IJCAI 2025 Survey Track)
> **作者**: Schoepp et al.

### 覆盖内容
- LLM/VLM 参与 RL 的三种角色：agent、planner、reward
- 不限于训练 MLLM，也包括用 MLLM 辅助传统 RL
- 发表在一流会议（IJCAI，CORE A*）

### 适合谁读
想了解 MLLM 和 RL 双向关系（不限于"训模型"）时读。

---

## 5. Awesome-RL-for-Multimodal-Foundation-Models

> **维护者**: Wu et al.
> **形式**: GitHub 仓库，持续更新
> **链接**: `github.com/weijiawu/Awesome-RL-for-Multimodal-Foundation-Models`

### 覆盖内容
- 首个面向多模态基础模型 RL 训练的全面资源列表
- 涵盖 MLLM、视觉生成、统一模型、VLA agent
- 持续更新，包含最新论文和代码链接

### 适合谁读
追踪最新论文时用，当动态字典。

---

## 阅读建议

| 如果你想 | 读哪篇 |
|---------|--------|
| 快速入门建立全局认知 | Reinforced MLLM Survey (arXiv:2504) |
| 深入了解 reward design | RL-Powered Visual Reasoning (ACM 2026) |
| 了解 MLLM 推理能力演变 | Perception, Reason, Think, Plan |
| 追踪最新论文 | Awesome-RL GitHub 仓库 |
