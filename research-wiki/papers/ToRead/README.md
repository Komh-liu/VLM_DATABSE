# ToRead: Dense VLM Multi-Teacher Distillation

> 当前目录只保留与“dense VLM 多教师蒸馏 / OPD-MOPD / 梯度冲突 / 世界知识 VQA / capability segment supervision”直接相关的论文。

## 主阅读路线

- [理论支撑与 novelty 防线](theory_support_mopd_dense_vlm.md)
- [世界知识 VQA × Multi-Teacher Distillation 阅读路线](world_knowledge_vqa_mopd_reading.md)

## 阅读分层原则

```text
Novelty attack 只考虑已发表/正式接收论文；
未发表 arXiv 工作仍然要读，但只作为 related work、趋势、方法参考和风险观察。
```

## 现在最该读的 10 篇

### A. 先读，决定论文主 claim

| 顺序 | 文件 | 读它是为了回答什么 |
|---|---|---|
| 1 | `rlcsd_2606.11709.pdf` | style drift 已经被谁说过？为什么 masking-only 不够？ |
| 2 | `drive_kd_2601.21288.pdf` | VLM 多教师 capability conflict 最近怎么被讨论？我们和 driving setting 差在哪？ |
| 3 | `MOPD_2606.30406.pdf` | token-level multi-teacher OPD 的原始目标和假设是什么？ |
| 4 | `camopd_2605.27115.pdf` | 现有 MOPD 如何处理 counteraction？和我们的 content-token conflict 差在哪？ |
| 5 | `revisiting_opd_2603.25562.pdf` | OPD/KL token supervision 还有哪些已知 failure modes？ |

### B. 再读，决定正式 novelty 边界

| 顺序 | 文件 | 读它是为了回答什么 |
|---|---|---|
| 6 | `pcgrad_neurips2020.pdf` | 梯度冲突这个概念的已发表边界是什么？ |
| 7 | `cagrad_neurips2021.pdf` | 如果审稿人要求 gradient surgery baseline，该怎么对比？ |
| 8 | `move_kd_cvpr2025.pdf` | 已发表 VLM 多专家/多教师 KD 做到了哪一步？ |

### C. 最后读，决定实验场景

| 顺序 | 文件 | 读它是为了回答什么 |
|---|---|---|
| 9 | `aokvqa_eccv2022.pdf` | rationale/世界知识 VQA 能否支持 segment/capability 标注？ |
| 10 | `infoseek_emnlp2023.pdf` | entity + external knowledge 场景是否更适合主实验？ |

## 完整第一优先级

这些论文需要精读，直接决定 claim 怎么写：

| 文件 | 作用 |
|---|---|
| `MOPD_2606.30406.pdf` | 当前 multi-teacher OPD 主靶子；未发表则不作为 novelty attack |
| `drive_kd_2601.21288.pdf` | 最接近的未发表 VLM 多教师冲突参考；不作为正式 novelty attack |
| `camopd_2605.27115.pdf` | MOPD counteraction / conflict 修补近邻；未发表则作为风险观察 |
| `revisiting_opd_2603.25562.pdf` | OPD token-level failure modes；未发表则作为参考 |
| `opd_rethinking_2026.pdf` | OPD 机制与适用条件；按发表状态决定引用强度 |
| `rlcsd_2606.11709.pdf` | token-level OPSD 风格漂移与 contrastive 去风格；未发表但高度相关 |

## 第二优先级

这些论文支撑方法设计和替代方案：

| 文件 | 作用 |
|---|---|
| `OPD_2604.13016.pdf` | OPD 基础 |
| `stepopsd_2605.27140.pdf` | 从 token-level 转向 step/segment-level credit |
| `omniopd_2606.01476.pdf` | logit-free / chunk-level OPD 思路 |
| `rlcsd_2606.11709.pdf` | correct hint vs wrong hint 差分，抵消 shared style drift |
| `dpkd_2406.19774.pdf` | KD 分布/偏好基础 |
| `gkd_on_policy_distillation_2306.13649.pdf` | generalized KD / on-policy 蒸馏背景 |
| `dpo_neurips2023.pdf` | preference objective 基础，可支撑 segment preference |
| `lets_verify_step_by_step_2305.20050.pdf` | process supervision 经典支撑 |
| `math_shepherd_acl2024.pdf` | 自动 step-level supervision 支撑 |

## 梯度冲突与多教师 KD

| 文件 | 作用 |
|---|---|
| `pcgrad_neurips2020.pdf` | 已发表；gradient conflict 经典定义与正式 novelty 边界 |
| `cagrad_neurips2021.pdf` | 已发表；conflict-averse multi-task optimization |
| `gradvac_2010.05874.pdf` | 已发表；gradient similarity 作为任务关系信号 |
| `move_kd_cvpr2025.pdf` | 已发表；VLM 多视觉专家 KD，LoRA/MoE 缓解冲突 |
| `ammkd_2509.00039.pdf` | 未发表/待确认；多模态多教师 KD 参考，不作为 novelty attack |

## 世界知识 VQA / KB-VQA

| 文件 | 作用 |
|---|---|
| `okvqa_cvpr2019.pdf` | 外部知识 VQA 基础 benchmark |
| `aokvqa_eccv2022.pdf` | world knowledge VQA，含 rationale |
| `infoseek_emnlp2023.pdf` | visual information-seeking questions |
| `encyclopedic_vqa_iccv2023.pdf` | fine-grained entity + Wikipedia knowledge |
| `echosight_emnlp2024.pdf` | KB-VQA RAG 方法相关工作 |
| `reag_cvpr2026.pdf` | KB-VQA reasoning / critic / retrieval 强相关方法 |

## 实验背景

| 文件 | 作用 |
|---|---|
| `qwenvl_2023.pdf` | Qwen-VL 背景 |
| `qwen2vl_2024.pdf` | Qwen2-VL 背景 |
| `qwen2_5vl_2025.pdf` | 当前实验模型背景 |
| `qwen3vl_2025.pdf` | dense / MoE VLM 架构背景 |

## 只扫读

| 文件 | 作用 |
|---|---|
| `DeepSeek-V4_2606.19348.pdf` | 若涉及工业 MOPD 叙事，可扫 |
| `full_rollouts_opd_2605.31490.pdf` | OPD rollout cost / reliability 旁支 |
| `mad_opd_2605.01347.pdf` | multi-rollout / debate OPD 旁支 |

## 当前不保留的旁支

已从本目录清掉以下类型论文：

- 通用 RLHF / PPO / GRPO 技巧论文
- 多模态 reward model 泛读论文
- 医学、多模态检测定位、幻觉 RL 等任务论文
- 与 dense VLM 多教师蒸馏机制关系较弱的大综述

清理原则：

```text
能直接支撑 claim、构成正式 novelty 边界、提供相关参考、构成 baseline、或解释 benchmark 的保留；
只是在“大多模态 RL/后训练”大领域相关的移除。
```
