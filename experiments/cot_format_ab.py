#!/usr/bin/env python3
"""
A/B test: Separated CoT vs Mixed CoT on Vision-SR1-47K.

Hypothesis: Vision-SR1's "see → think" format artificially separates visual
perception from reasoning. If accuracy is similar between separated and mixed
prompts, the dataset doesn't truly require interleaved visual-reasoning tokens
— and thus doesn't properly test MOPD's token-level multi-teacher routing.
"""

import json
import random
import re
from pathlib import Path
from collections import defaultdict
from datetime import datetime

import torch
from datasets import load_dataset
from transformers import Qwen3VLForConditionalGeneration, AutoProcessor
from qwen_vl_utils import process_vision_info

# ============================================================================
# Config
# ============================================================================
MODEL_ID = "Qwen/Qwen3-VL-2B-Instruct"
DATASET_ID = "LMMs-Lab-Turtle/Vision-SR1-47K"
SAMPLE_SIZE = 500
MAX_NEW_TOKENS = 1024
OUTPUT_DIR = Path(__file__).parent / "results"
OUTPUT_DIR.mkdir(exist_ok=True)

# ============================================================================
# Prompt Variants
# ============================================================================

SEPARATED_PROMPT = """You are a helpful assistant. When answering, follow this structure:
1. First, describe what you see in the image in full detail (visual perception).
2. Then, reason step by step to answer the question (chain-of-thought).
3. Finally, output your answer in \\boxed{ANSWER}.

Question: {question}
Options: {options}"""

MIXED_PROMPT = """You are a helpful assistant. Reason step by step to answer the question, looking at the image whenever visual information is needed. Output your final answer in \\boxed{ANSWER}.

Question: {question}
Options: {options}"""


def format_options(options_list):
    """Format options as A/B/C/D string."""
    if not options_list:
        return ""
    return " ".join(options_list)


def extract_answer(text):
    """Extract answer from \\boxed{...} or fallback to last capital letter."""
    # Try \boxed{...}
    m = re.search(r'\\boxed\{([^}]+)\}', text)
    if m:
        ans = m.group(1).strip()
        # Normalize: just take the first letter if it's "A. No" style
        letter = re.match(r'([A-Ea-e])', ans)
        if letter:
            return letter.group(1).upper()
        return ans.upper()

    # Fallback: last standalone capital letter A-E
    letters = re.findall(r'\b([A-E])\b', text)
    if letters:
        return letters[-1]

    return None


def build_messages(question, options_str, image, prompt_template):
    """Build messages list for Qwen3-VL processor."""
    content = [
        {"type": "image", "image": image},
        {"type": "text", "text": prompt_template.format(
            question=question,
            options=options_str
        )},
    ]
    return [{"role": "user", "content": content}]


def run_inference(model, processor, messages):
    """Run single inference. Returns (generated_text, wall_time)."""
    text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    image_inputs, video_inputs = process_vision_info(messages)
    inputs = processor(
        text=[text], images=image_inputs, videos=video_inputs,
        padding=True, return_tensors="pt"
    ).to(model.device)

    with torch.no_grad():
        generated_ids = model.generate(
            **inputs, max_new_tokens=MAX_NEW_TOKENS, do_sample=False
        )

    generated_ids_trimmed = [
        out_ids[len(in_ids):]
        for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
    ]
    output = processor.batch_decode(
        generated_ids_trimmed, skip_special_tokens=True,
        clean_up_tokenization_spaces=False
    )[0]
    return output


def main():
    print(f"Loading dataset: {DATASET_ID}")
    ds = load_dataset(DATASET_ID, split="train")
    print(f"  {len(ds)} examples, {len(set(ds['data_source']))} sources")

    # Stratified sample across data sources
    rng = random.Random(42)
    by_source = defaultdict(list)
    for i, source in enumerate(ds['data_source']):
        by_source[source].append(i)

    indices = []
    for source, idxs in sorted(by_source.items()):
        n = max(1, int(SAMPLE_SIZE * len(idxs) / len(ds)))
        indices.extend(rng.sample(idxs, min(n, len(idxs))))

    rng.shuffle(indices)
    indices = indices[:SAMPLE_SIZE]
    print(f"  Sampled {len(indices)} examples from {len(set(ds[i]['data_source'] for i in indices))} sources")

    # Load model
    print(f"\nLoading model: {MODEL_ID}")
    model = Qwen3VLForConditionalGeneration.from_pretrained(
        MODEL_ID, torch_dtype=torch.bfloat16,
        device_map="auto", attn_implementation="sdpa"
    )
    processor = AutoProcessor.from_pretrained(MODEL_ID)
    print(f"  Model loaded on {model.device}")

    # Run A/B test
    results = []
    correct = {"separated": 0, "mixed": 0}
    total = 0

    for round_num, idx in enumerate(indices):
        example = ds[idx]
        question = example['problem']
        options_str = format_options(example.get('options', []))
        gt_answer = example['answer'].strip().upper()
        image = example['images']

        row = {
            'idx': idx, 'source': example['data_source'],
            'question': question[:200], 'gt': gt_answer
        }

        # Randomize which prompt goes first to avoid order effects
        if rng.random() < 0.5:
            order = [('separated', SEPARATED_PROMPT), ('mixed', MIXED_PROMPT)]
        else:
            order = [('mixed', MIXED_PROMPT), ('separated', SEPARATED_PROMPT)]

        for cond_name, prompt_tmpl in order:
            messages = build_messages(question, options_str, image, prompt_tmpl)
            try:
                output = run_inference(model, processor, messages)
                pred = extract_answer(output)
                is_correct = (pred == gt_answer)
                row[f'{cond_name}_pred'] = pred
                row[f'{cond_name}_correct'] = is_correct
                row[f'{cond_name}_output'] = output[:500]
                if is_correct:
                    correct[cond_name] += 1
            except Exception as e:
                print(f"  ERROR [{cond_name}] idx={idx}: {e}")
                row[f'{cond_name}_pred'] = None
                row[f'{cond_name}_correct'] = False

        results.append(row)
        total += 1

        if (total) % 10 == 0:
            sep_acc = correct['separated'] / total * 100
            mix_acc = correct['mixed'] / total * 100
            print(f"  [{total}/{len(indices)}] separated={sep_acc:.1f}% mixed={mix_acc:.1f}%  delta={mix_acc - sep_acc:+.1f}%")

    # Summary
    sep_acc = correct['separated'] / total * 100
    mix_acc = correct['mixed'] / total * 100
    print(f"\n{'='*60}")
    print(f"RESULTS (n={total})")
    print(f"  Separated CoT:  {correct['separated']}/{total} = {sep_acc:.2f}%")
    print(f"  Mixed CoT:      {correct['mixed']}/{total} = {mix_acc:.2f}%")
    print(f"  Delta:          {mix_acc - sep_acc:+.2f}%")
    print(f"{'='*60}")

    # Per-source breakdown
    source_stats = defaultdict(lambda: {'sep': 0, 'mix': 0, 'n': 0})
    for r in results:
        src = r['source']
        source_stats[src]['n'] += 1
        if r.get('separated_correct'):
            source_stats[src]['sep'] += 1
        if r.get('mixed_correct'):
            source_stats[src]['mix'] += 1

    print("\nPer-source (n>=10):")
    for src, stats in sorted(source_stats.items(), key=lambda x: -x[1]['n']):
        if stats['n'] >= 10:
            s = stats['sep'] / stats['n'] * 100
            m = stats['mix'] / stats['n'] * 100
            print(f"  {src:25s} n={stats['n']:3d}  sep={s:5.1f}%  mix={m:5.1f}%  delta={m-s:+.1f}%")

    # Save
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_file = OUTPUT_DIR / f"cot_format_ab_{ts}.json"
    with open(out_file, 'w') as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    print(f"\nResults saved to {out_file}")


if __name__ == "__main__":
    main()
