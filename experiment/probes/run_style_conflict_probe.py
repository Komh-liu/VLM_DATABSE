#!/usr/bin/env python3
"""Same-capability different-style conflict probe for Qwen2.5-VL."""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

import torch
from qwen_vl_utils import process_vision_info
from transformers import AutoProcessor, Qwen2_5_VLForConditionalGeneration

from run_lora_teacher_conflict_probe import (
    FUNCTION_WORDS,
    MARKER_FRAGMENTS,
    kl_logit_grad,
    kl_per_token,
    segment_labels,
    token_kind,
)


BASE_FORMAT = (
    "Use exactly this format: [VISUAL] ... [KNOWLEDGE] ... "
    "[REASON] ... [ANSWER] ..."
)

STYLE_PAIRS = {
    "visual_style": (
        "Focus on visual evidence only. Use concise, factual wording. Avoid background knowledge.",
        "Focus on visual evidence only. Use rich descriptive wording and mention visible attributes in detail. Avoid background knowledge.",
    ),
    "knowledge_style": (
        "Focus on relevant commonsense or world knowledge. Use concise, factual wording.",
        "Focus on relevant commonsense or world knowledge. Use explanatory wording with more context and connective phrases.",
    ),
}

STUDENT_STYLE = "Answer naturally and accurately while following the required format."


def read_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def user_text(sample: dict, style: str) -> str:
    choices = sample.get("choices") or []
    choice_text = "\nChoices: " + "; ".join(str(x) for x in choices) if choices else ""
    return f"{style}\n{BASE_FORMAT}\nQuestion: {sample['question']}{choice_text}"


def messages_for(sample: dict, style: str) -> list[dict]:
    return [
        {
            "role": "user",
            "content": [
                {"type": "image", "image": sample["image"]},
                {"type": "text", "text": user_text(sample, style)},
            ],
        }
    ]


def encode(processor, sample: dict, style: str, answer: str | None = None):
    messages = messages_for(sample, style)
    text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    if answer is not None:
        text += answer
    image_inputs, video_inputs = process_vision_info(messages)
    return processor(text=[text], images=image_inputs, videos=video_inputs, padding=True, return_tensors="pt")


def generate_answer(model, processor, sample: dict, max_new_tokens: int) -> str:
    inputs = encode(processor, sample, STUDENT_STYLE).to(model.device)
    with torch.inference_mode():
        out = model.generate(**inputs, max_new_tokens=max_new_tokens, do_sample=False)
    trimmed = [o[len(i) :] for i, o in zip(inputs.input_ids, out)]
    return processor.batch_decode(trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False)[0]


def logits_for(model, processor, sample: dict, style: str, answer: str) -> tuple[torch.Tensor, torch.Tensor]:
    prompt_inputs = encode(processor, sample, style).to(model.device)
    full_inputs = encode(processor, sample, style, answer).to(model.device)
    prompt_len = prompt_inputs.input_ids.shape[1]
    full_len = full_inputs.input_ids.shape[1]
    with torch.inference_mode():
        logits = model(**full_inputs).logits[0]
    response_len = full_len - prompt_len
    return logits[prompt_len - 1 : prompt_len - 1 + response_len].float().cpu(), full_inputs.input_ids[0, prompt_len:full_len].cpu()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="experiment/models/Qwen2.5-VL-3B-Instruct")
    parser.add_argument("--samples", type=Path, default=Path("experiment/probes/probe_samples_120.jsonl"))
    parser.add_argument("--out", type=Path, default=Path("experiment/probes/go_nogo_style_120.jsonl"))
    parser.add_argument("--style-pair", choices=sorted(STYLE_PAIRS), default="visual_style")
    parser.add_argument("--limit", type=int, default=120)
    parser.add_argument("--max-new-tokens", type=int, default=64)
    args = parser.parse_args()

    style_a, style_b = STYLE_PAIRS[args.style_pair]
    samples = read_jsonl(args.samples)[: args.limit]
    processor = AutoProcessor.from_pretrained(
        args.model,
        trust_remote_code=True,
        min_pixels=128 * 28 * 28,
        max_pixels=256 * 28 * 28,
    )
    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        args.model,
        torch_dtype=torch.bfloat16,
        device_map="cuda",
        trust_remote_code=True,
    ).eval()

    args.out.parent.mkdir(parents=True, exist_ok=True)
    rows = []
    with args.out.open("w", encoding="utf-8") as f:
        for idx, sample in enumerate(samples, start=1):
            print(f"[{idx}/{len(samples)}] {sample['id']}")
            answer = generate_answer(model, processor, sample, args.max_new_tokens)
            student_logits, response_ids = logits_for(model, processor, sample, STUDENT_STYLE, answer)
            style_a_logits, _ = logits_for(model, processor, sample, style_a, answer)
            style_b_logits, _ = logits_for(model, processor, sample, style_b, answer)
            n = min(len(response_ids), len(student_logits), len(style_a_logits), len(style_b_logits))

            log_p_s = torch.log_softmax(student_logits[:n], dim=-1)
            log_p_a = torch.log_softmax(style_a_logits[:n], dim=-1)
            log_p_b = torch.log_softmax(style_b_logits[:n], dim=-1)
            grad_a = kl_logit_grad(log_p_s, log_p_a)
            grad_b = kl_logit_grad(log_p_s, log_p_b)
            cos = torch.nn.functional.cosine_similarity(grad_a, grad_b, dim=-1)
            kl_a = kl_per_token(log_p_s, log_p_a)
            kl_b = kl_per_token(log_p_s, log_p_b)
            kl_mean = 0.5 * (kl_a + kl_b)

            labels = segment_labels(processor, response_ids[:n])
            token_pieces = [
                processor.tokenizer.decode([token_id], skip_special_tokens=True)
                for token_id in response_ids[:n].tolist()
            ]
            tokens = [
                {
                    "token": token,
                    "segment": label,
                    "kind": token_kind(token),
                    "cosine": float(value),
                    "kl_a": float(ka),
                    "kl_b": float(kb),
                    "kl_mean": float(km),
                }
                for token, label, value, ka, kb, km in zip(
                    token_pieces,
                    labels,
                    cos.tolist(),
                    kl_a.tolist(),
                    kl_b.tolist(),
                    kl_mean.tolist(),
                )
            ]
            by_segment: dict[str, list[float]] = {}
            for t in tokens:
                by_segment.setdefault(t["segment"], []).append(t["cosine"])
            result = {
                "id": sample["id"],
                "dataset": sample["dataset"],
                "task_type": sample["task_type"],
                "style_pair": args.style_pair,
                "question": sample["question"],
                "answer_text": answer,
                "num_tokens": int(n),
                "mean_cosine": float(cos.mean().item()),
                "min_cosine": float(cos.min().item()),
                "mean_kl": float(kl_mean.mean().item()),
                "by_segment": {
                    key: {"count": len(vals), "mean": sum(vals) / len(vals), "min": min(vals)}
                    for key, vals in by_segment.items()
                },
                "most_negative_tokens": sorted(tokens, key=lambda x: x["cosine"])[:10],
                "tokens": tokens,
            }
            f.write(json.dumps(result, ensure_ascii=False) + "\n")
            f.flush()
            rows.append(result)
            print(f"  mean_cos={result['mean_cosine']:.4f} min_cos={result['min_cosine']:.4f}")

    print("summary")
    print("samples", len(rows))
    print("mean_cosine", sum(r["mean_cosine"] for r in rows) / len(rows))
    print("min_cosine", min(r["min_cosine"] for r in rows))
    print("wrote", args.out)


if __name__ == "__main__":
    main()
