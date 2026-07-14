#!/usr/bin/env python3
"""
找 A-OKVQA 中 base model 做不对的样本。

目的: 证明 base model 虽然能做对一些题, 但仍有大量做不对的题 → MOPD 有提升空间。
用法: LD_LIBRARY_PATH=/usr/lib/wsl/lib .venv/bin/python experiment/probes/find_hard_samples.py
"""

import sys, os, json, io, random
from pathlib import Path
import pandas as pd
from PIL import Image
import torch
from transformers import Qwen2_5_VLForConditionalGeneration, AutoProcessor
from qwen_vl_utils import process_vision_info

# ==== 配置 ====
MODEL_ID = "Qwen/Qwen2.5-VL-3B-Instruct"
VAL_PARQUET = "experiment/data/aokvqa/data/data/validation-00000-of-00001-b2bd0de231b6326a.parquet"
CACHE_DIR = "experiment/probes/cache_images"
NUM_SAMPLES = 30  # 随机测试 30 个样本

# ==== 加载验证数据 ====
df = pd.read_parquet(VAL_PARQUET)
print(f"验证集总样本数: {len(df)}")

# 随机选样本 (固定 seed 方便复现)
random.seed(42)
indices = random.sample(range(len(df)), min(NUM_SAMPLES, len(df)))

# ==== 加载模型 ====
print(f"加载模型: {MODEL_ID}")
model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
    MODEL_ID,
    torch_dtype=torch.bfloat16,
    device_map="auto",
    trust_remote_code=True,
)
processor = AutoProcessor.from_pretrained(MODEL_ID, trust_remote_code=True)
print("模型加载完成\n")

# ==== 推理循环 ====
results = []
for idx in indices:
    row = df.iloc[idx]
    question_id = row["question_id"]
    question = row["question"]
    choices = row["choices"]  # list of str
    correct_idx = row["correct_choice_idx"]  # 正确答案在 choices 中的索引
    correct_answer = choices[correct_idx]
    rationales = row.get("rationales", [])

    # 保存图片到缓存
    img_bytes = row["image"]["bytes"]
    img = Image.open(io.BytesIO(img_bytes))
    img_path = os.path.join(CACHE_DIR, f"hard_test_{question_id}.jpg")
    img.save(img_path)

    # 构造 VQA prompt (直接让模型生成答案, 不做选择题)
    prompt = f"Answer the following question about the image. Answer as briefly as possible.\nQuestion: {question}\nAnswer:"

    # 构造 Qwen2.5-VL 消息格式
    messages = [
        {
            "role": "user",
            "content": [
                {"type": "image", "image": img_path},
                {"type": "text", "text": prompt},
            ],
        }
    ]

    # 推理
    text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    image_inputs, video_inputs = process_vision_info(messages)
    inputs = processor(
        text=[text], images=image_inputs, videos=video_inputs, padding=True, return_tensors="pt"
    ).to(model.device)

    with torch.no_grad():
        generated_ids = model.generate(**inputs, max_new_tokens=64, do_sample=False)
    generated_ids_trimmed = [out_ids[len(in_ids):] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)]
    output_text = processor.batch_decode(generated_ids_trimmed, skip_special_tokens=True)[0].strip()

    is_correct = correct_answer.lower() in output_text.lower() or output_text.lower() in correct_answer.lower()

    status = "✅ CORRECT" if is_correct else "❌ WRONG"
    print(f"[{question_id}] {status}")
    print(f"  Q: {question}")
    print(f"  Expected: {correct_answer}")
    print(f"  Got: {output_text}")
    print(f"  Choices: {choices}")
    if len(rationales) > 0:
        print(f"  Rationale: {str(rationales[0])[:100]}...")
    print()

    results.append({
        "question_id": question_id,
        "question": question,
        "choices": choices,
        "correct_answer": correct_answer,
        "model_answer": output_text,
        "is_correct": is_correct,
        "image_path": img_path,
        "rationales": rationales,
    })

# ==== 汇总 ====
correct_count = sum(1 for r in results if r["is_correct"])
wrong_count = len(results) - correct_count
print(f"{'='*60}")
print(f"总测试: {len(results)} | 正确: {correct_count} | 错误: {wrong_count}")
print(f"准确率: {correct_count / len(results) * 100:.1f}%")
print(f"{'='*60}")

# 列出所有做错的样本
if wrong_count > 0:
    print("\n\n=== 做错的样本 ===")
    for r in results:
        if not r["is_correct"]:
            print(f"\n[{r['question_id']}] {r['question']}")
            print(f"  正确答案: {r['correct_answer']}")
            print(f"  模型回答: {r['model_answer']}")
            print(f"  图片: {r['image_path']}")

# 保存结果
out_path = "experiment/probes/hard_samples_results.jsonl"
with open(out_path, "w") as f:
    for r in results:
        f.write(json.dumps(r, ensure_ascii=False) + "\n")
print(f"\n结果保存到: {out_path}")
