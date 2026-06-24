# LLM 后训练 RL 算法学习路线

> **方向**: GRPO · DAPO · GSPO · OPD —— 2024-2026 最新 LLM 后训练 RL 算法
> **目标**: 理解算法演进逻辑，为多模态 RL 研究打基础

---

## 阅读顺序

```
Week 1-2: 综述建立全局认知
  │
  ├─ 1. survey_sailing_by_stars_2025.pdf     ← 从这里开始
  │     EMNLP 2025 Findings · Wu et al.
  │     PPO → DPO → GRPO → RLVR 统一框架
  │     最全面 RL 后训练综述
  │
  └─ 2. survey_post_training_llm_2025.pdf
        87 页 · Tie et al.
        SFT → 对齐 → 推理 → 效率 → 多模态 五大范式

Week 3-4: 核心算法原始论文
  │
  ├─ 3. deepseekmath_grpo_2024.pdf           ← 只读 §3
  │     DeepSeekMath (2024)
  │     GRPO: 去掉 critic，组内归一化当 baseline
  │
  ├─ 4. dapo_neurips2025.pdf
  │     NeurIPS 2025 · ByteDance & 清华
  │     DAPO: 四项技巧修复 GRPO（Clip-Higher/动态采样/Token级loss/超长惩罚）
  │
  ├─ 5. gspo_qwen_2025.pdf
  │     Qwen 团队 (2025.07)
  │     GSPO: 序列级 importance ratio 修复 GRPO 的 token 级噪声
  │
  └─ 6. opd_rethinking_2026.pdf
        ICML 2026 Workshop · 清华 THUNLP
        OPD: 用 teacher log-prob 替代 reward signal

之后: 扩展到多模态
  │
  └─ 读 multimodal_rl_papers.md 中的 MLLM+RL 论文
```

---

## 算法演进脉络

```
PPO (2017) ──→ GRPO (2024) ──→ DAPO (2025)    ← 在 GRPO 上加技巧
                           ──→ GSPO (2025)    ← 改 GRPO 核心机制
              ──→ OPD (2025-2026)              ← 不玩 RL，改用蒸馏
```

---

## 文件清单

| 文件 | 内容 |
|------|------|
| `survey_sailing_by_stars_2025.pdf` | 综述: RL 后训练统一框架 |
| `survey_post_training_llm_2025.pdf` | 综述: LLM 后训练五大范式 |
| `deepseekmath_grpo_2024.pdf` | GRPO 原始论文 |
| `dapo_neurips2025.pdf` | DAPO: 开源 LLM RL 系统 |
| `gspo_qwen_2025.pdf` | GSPO: 序列级策略优化 |
| `opd_rethinking_2026.pdf` | OPD: 重思考 On-Policy 蒸馏 |
| `multimodal_rl_survey.md` | 多模态 RL 综述汇总 |
| `multimodal_rl_papers.md` | 多模态 RL 顶会论文汇总 |
