#!/usr/bin/env python3
"""Minimal logit-space KL conflict probe for Qwen2.5-VL.

This is an engineering pilot, not the final paper-grade gradient analysis.
It compares the student KL gradient induced by two prompt-conditioned teacher
distributions: visual-focused and knowledge-focused.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
from qwen_vl_utils import process_vision_info
from transformers import AutoProcessor, Qwen2_5_VLForConditionalGeneration


BASE_INSTRUCTION = (
    "Answer with labeled parts: [VISUAL] observations from the image; "
    "[KNOWLEDGE] relevant world knowledge if needed; [REASON] concise reasoning; "
    "[ANSWER] final short answer."
)
STUDENT_STYLE = "Be balanced across visual evidence, knowledge, and reasoning."
VISUAL_STYLE = "Focus on visual evidence only. Prefer image-grounded observations."
KNOWLEDGE_STYLE = "Focus on relevant world knowledge and semantic facts."


def read_jsonl(path: Path) -> list[dict]:
    with path.open(encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def build_user_text(sample: dict, style: str) -> str:
    choices = sample.get("choices") or []
    choice_text = ""
    if choices:
        choice_text = "\nChoices: " + "; ".join(str(x) for x in choices)
    return f"{style}\n{BASE_INSTRUCTION}\nQuestion: {sample['question']}{choice_text}"


def build_messages(sample: dict, style: str) -> list[dict]:
    return [
        {
            "role": "user",
            "content": [
                {"type": "image", "image": sample["image"]},
                {"type": "text", "text": build_user_text(sample, style)},
            ],
        }
    ]


def encode_prompt(processor, sample: dict, style: str, answer: str | None = None):
    messages = build_messages(sample, style)
    text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    if answer is not None:
        text = text + answer
    image_inputs, video_inputs = process_vision_info(messages)
    return processor(
        text=[text],
        images=image_inputs,
        videos=video_inputs,
        padding=True,
        return_tensors="pt",
    )


def generate_answer(model, processor, sample: dict, max_new_tokens: int) -> str:
    inputs = encode_prompt(processor, sample, STUDENT_STYLE).to(model.device)
    with torch.inference_mode():
        output = model.generate(**inputs, max_new_tokens=max_new_tokens, do_sample=False)
    trimmed = [out[len(inp) :] for inp, out in zip(inputs.input_ids, output)]
    return processor.batch_decode(trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False)[0]


def logits_for_answer(model, processor, sample: dict, style: str, answer: str) -> tuple[torch.Tensor, torch.Tensor]:
    prompt_inputs = encode_prompt(processor, sample, style).to(model.device)
    full_inputs = encode_prompt(processor, sample, style, answer).to(model.device)
    prompt_len = prompt_inputs.input_ids.shape[1]
    full_len = full_inputs.input_ids.shape[1]
    response_len = full_len - prompt_len
    if response_len <= 0:
        raise ValueError("empty response after tokenization")

    with torch.inference_mode():
        logits = model(**full_inputs).logits[0]

    pred_logits = logits[prompt_len - 1 : prompt_len - 1 + response_len].float().cpu()
    response_ids = full_inputs.input_ids[0, prompt_len:full_len].cpu()
    return pred_logits, response_ids


def kl_logit_grad(log_p_student: torch.Tensor, log_p_teacher: torch.Tensor) -> torch.Tensor:
    p_student = log_p_student.exp()
    kl = (p_student * (log_p_student - log_p_teacher)).sum(dim=-1, keepdim=True)
    return p_student * ((log_p_student - log_p_teacher) - kl)


def segment_labels(processor, response_ids: torch.Tensor) -> list[str]:
    labels = []
    current = "OTHER"
    decoded_so_far = ""
    for token_id in response_ids.tolist():
        piece = processor.tokenizer.decode([token_id], skip_special_tokens=True)
        decoded_so_far += piece
        upper = decoded_so_far.upper()
        if "[ANSWER]" in upper:
            current = "ANSWER"
        elif "[REASON]" in upper:
            current = "REASON"
        elif "[KNOWLEDGE]" in upper:
            current = "KNOWLEDGE"
        elif "[VISUAL]" in upper:
            current = "VISUAL"
        labels.append(current)
    return labels


def probe_sample(model, processor, sample: dict, max_new_tokens: int) -> dict:
    answer = generate_answer(model, processor, sample, max_new_tokens=max_new_tokens)
    student_logits, response_ids = logits_for_answer(model, processor, sample, STUDENT_STYLE, answer)
    visual_logits, visual_ids = logits_for_answer(model, processor, sample, VISUAL_STYLE, answer)
    knowledge_logits, knowledge_ids = logits_for_answer(model, processor, sample, KNOWLEDGE_STYLE, answer)

    n = min(len(response_ids), len(visual_ids), len(knowledge_ids))
    student_logits = student_logits[:n]
    visual_logits = visual_logits[:n]
    knowledge_logits = knowledge_logits[:n]
    response_ids = response_ids[:n]

    log_p_s = torch.log_softmax(student_logits, dim=-1)
    log_p_v = torch.log_softmax(visual_logits, dim=-1)
    log_p_k = torch.log_softmax(knowledge_logits, dim=-1)
    grad_v = kl_logit_grad(log_p_s, log_p_v)
    grad_k = kl_logit_grad(log_p_s, log_p_k)
    cos = torch.nn.functional.cosine_similarity(grad_v, grad_k, dim=-1)
    labels = segment_labels(processor, response_ids)

    by_segment: dict[str, list[float]] = {}
    for label, value in zip(labels, cos.tolist()):
        by_segment.setdefault(label, []).append(value)
    token_pieces = [
        processor.tokenizer.decode([token_id], skip_special_tokens=True)
        for token_id in response_ids.tolist()
    ]
    most_negative = sorted(
        [
            {"token": token, "segment": label, "cosine": float(value)}
            for token, label, value in zip(token_pieces, labels, cos.tolist())
        ],
        key=lambda x: x["cosine"],
    )[:10]

    return {
        "id": sample["id"],
        "dataset": sample["dataset"],
        "task_type": sample["task_type"],
        "question": sample["question"],
        "answer_text": answer,
        "num_tokens": int(n),
        "mean_cosine": float(cos.mean().item()),
        "min_cosine": float(cos.min().item()),
        "by_segment": {
            key: {
                "count": len(values),
                "mean": sum(values) / len(values),
                "min": min(values),
            }
            for key, values in by_segment.items()
        },
        "most_negative_tokens": most_negative,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="experiment/models/Qwen2.5-VL-3B-Instruct")
    parser.add_argument("--samples", type=Path, default=Path("experiment/probes/probe_samples.jsonl"))
    parser.add_argument("--out", type=Path, default=Path("experiment/probes/logit_conflict_results.jsonl"))
    parser.add_argument("--limit", type=int, default=4)
    parser.add_argument("--max-new-tokens", type=int, default=96)
    args = parser.parse_args()

    samples = read_jsonl(args.samples)[: args.limit]
    processor = AutoProcessor.from_pretrained(
        args.model,
        trust_remote_code=True,
        min_pixels=256 * 28 * 28,
        max_pixels=512 * 28 * 28,
    )
    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        args.model,
        torch_dtype=torch.bfloat16,
        device_map="cuda",
        trust_remote_code=True,
    )
    model.eval()

    args.out.parent.mkdir(parents=True, exist_ok=True)
    results = []
    with args.out.open("w", encoding="utf-8") as f:
        for idx, sample in enumerate(samples, start=1):
            print(f"[{idx}/{len(samples)}] {sample['id']}")
            result = probe_sample(model, processor, sample, args.max_new_tokens)
            f.write(json.dumps(result, ensure_ascii=False) + "\n")
            f.flush()
            results.append(result)
            print(
                f"  mean_cos={result['mean_cosine']:.4f} "
                f"min_cos={result['min_cosine']:.4f} tokens={result['num_tokens']}"
            )

    if results:
        print("\nsummary")
        print("samples", len(results))
        print("mean_cosine", sum(r["mean_cosine"] for r in results) / len(results))
        print("min_cosine", min(r["min_cosine"] for r in results))
        print("wrote", args.out)


if __name__ == "__main__":
    main()
