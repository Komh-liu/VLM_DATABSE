#!/usr/bin/env python3
"""Conflict probe using two trained LoRA teacher adapters with the same prompt."""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

import torch
from peft import PeftModel
from qwen_vl_utils import process_vision_info
from transformers import AutoProcessor, Qwen2_5_VLForConditionalGeneration


INSTRUCTION = (
    "Use exactly this format: [VISUAL] ... [KNOWLEDGE] ... "
    "[REASON] ... [ANSWER] ..."
)

FUNCTION_WORDS = {
    "a",
    "an",
    "the",
    "this",
    "that",
    "these",
    "those",
    "is",
    "are",
    "was",
    "were",
    "be",
    "being",
    "been",
    "am",
    "of",
    "to",
    "in",
    "on",
    "at",
    "for",
    "from",
    "with",
    "by",
    "as",
    "and",
    "or",
    "but",
    "if",
    "then",
    "so",
    "because",
    "which",
    "who",
    "what",
    "where",
    "when",
    "why",
    "how",
    "it",
    "its",
    "their",
    "his",
    "her",
    "they",
    "them",
    "he",
    "she",
    "we",
    "you",
    "i",
    "there",
    "has",
    "have",
    "had",
    "can",
    "could",
    "would",
    "should",
    "will",
    "may",
    "might",
    "must",
    "do",
    "does",
    "did",
    "not",
    "no",
    "yes",
}

MARKER_FRAGMENTS = {
    "visual",
    "knowledge",
    "reason",
    "answer",
    "vis",
    "ual",
    "know",
    "kn",
    "ow",
    "ledge",
    "reas",
    "ason",
    "ans",
    "wer",
}


def token_kind(token: str) -> str:
    stripped = token.strip()
    lower = stripped.lower()
    if not stripped:
        return "space"
    if lower in MARKER_FRAGMENTS or any(part in lower for part in ["visual", "knowledge", "reason", "answer"]):
        return "marker"
    if re.fullmatch(r"[\W_]+", stripped):
        return "punct"
    if lower in FUNCTION_WORDS:
        return "function"
    if stripped.startswith("'") or stripped in {"n't", "'s", "'re", "'ve", "'ll", "'d"}:
        return "function"
    if len(stripped) <= 2 and not stripped.isdigit():
        return "subword"
    return "content"


def read_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def user_text(sample: dict) -> str:
    choices = sample.get("choices") or []
    choice_text = "\nChoices: " + "; ".join(str(x) for x in choices) if choices else ""
    return f"{INSTRUCTION}\nQuestion: {sample['question']}{choice_text}"


def messages_for(sample: dict) -> list[dict]:
    return [
        {
            "role": "user",
            "content": [
                {"type": "image", "image": sample["image"]},
                {"type": "text", "text": user_text(sample)},
            ],
        }
    ]


def encode(processor, sample: dict, answer: str | None = None):
    messages = messages_for(sample)
    text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    if answer is not None:
        text += answer
    image_inputs, video_inputs = process_vision_info(messages)
    return processor(text=[text], images=image_inputs, videos=video_inputs, padding=True, return_tensors="pt")


def generate_answer(model, processor, sample: dict, max_new_tokens: int) -> str:
    inputs = encode(processor, sample).to(model.device)
    with torch.inference_mode():
        out = model.generate(**inputs, max_new_tokens=max_new_tokens, do_sample=False)
    trimmed = [o[len(i) :] for i, o in zip(inputs.input_ids, out)]
    return processor.batch_decode(trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False)[0]


def logits_for(model, processor, sample: dict, answer: str) -> tuple[torch.Tensor, torch.Tensor]:
    prompt_inputs = encode(processor, sample).to(model.device)
    full_inputs = encode(processor, sample, answer).to(model.device)
    prompt_len = prompt_inputs.input_ids.shape[1]
    full_len = full_inputs.input_ids.shape[1]
    with torch.inference_mode():
        logits = model(**full_inputs).logits[0]
    response_len = full_len - prompt_len
    return logits[prompt_len - 1 : prompt_len - 1 + response_len].float().cpu(), full_inputs.input_ids[0, prompt_len:full_len].cpu()


def kl_logit_grad(log_p_student: torch.Tensor, log_p_teacher: torch.Tensor) -> torch.Tensor:
    p_student = log_p_student.exp()
    kl = (p_student * (log_p_student - log_p_teacher)).sum(dim=-1, keepdim=True)
    return p_student * ((log_p_student - log_p_teacher) - kl)


def kl_per_token(log_p_student: torch.Tensor, log_p_teacher: torch.Tensor) -> torch.Tensor:
    p_student = log_p_student.exp()
    return (p_student * (log_p_student - log_p_teacher)).sum(dim=-1)


def segment_labels(processor, response_ids: torch.Tensor) -> list[str]:
    labels, current, text = [], "OTHER", ""
    for token_id in response_ids.tolist():
        text += processor.tokenizer.decode([token_id], skip_special_tokens=True)
        upper = text.upper()
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


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="experiment/models/Qwen2.5-VL-3B-Instruct")
    parser.add_argument("--visual-adapter", type=Path, required=True)
    parser.add_argument("--knowledge-adapter", type=Path, required=True)
    parser.add_argument("--samples", type=Path, default=Path("experiment/probes/probe_samples.jsonl"))
    parser.add_argument("--out", type=Path, default=Path("experiment/probes/lora_teacher_conflict_results.jsonl"))
    parser.add_argument("--limit", type=int, default=12)
    parser.add_argument("--max-new-tokens", type=int, default=96)
    args = parser.parse_args()

    processor = AutoProcessor.from_pretrained(
        args.model,
        trust_remote_code=True,
        min_pixels=128 * 28 * 28,
        max_pixels=256 * 28 * 28,
    )
    base = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        args.model,
        torch_dtype=torch.bfloat16,
        device_map="cuda",
        trust_remote_code=True,
    ).eval()
    visual_model = PeftModel.from_pretrained(base, args.visual_adapter, adapter_name="visual")
    visual_model.load_adapter(args.knowledge_adapter, adapter_name="knowledge")
    visual_model.eval()

    args.out.parent.mkdir(parents=True, exist_ok=True)
    rows = []
    with args.out.open("w", encoding="utf-8") as f:
        for idx, sample in enumerate(read_jsonl(args.samples)[: args.limit], start=1):
            print(f"[{idx}/{args.limit}] {sample['id']}")
            with visual_model.disable_adapter():
                answer = generate_answer(visual_model, processor, sample, args.max_new_tokens)

            with visual_model.disable_adapter():
                student_logits, response_ids = logits_for(visual_model, processor, sample, answer)
            visual_model.set_adapter("visual")
            visual_logits, _ = logits_for(visual_model, processor, sample, answer)
            visual_model.set_adapter("knowledge")
            knowledge_logits, _ = logits_for(visual_model, processor, sample, answer)

            n = min(len(response_ids), len(visual_logits), len(knowledge_logits), len(student_logits))
            log_p_s = torch.log_softmax(student_logits[:n], dim=-1)
            log_p_v = torch.log_softmax(visual_logits[:n], dim=-1)
            log_p_k = torch.log_softmax(knowledge_logits[:n], dim=-1)
            cos = torch.nn.functional.cosine_similarity(
                kl_logit_grad(log_p_s, log_p_v),
                kl_logit_grad(log_p_s, log_p_k),
                dim=-1,
            )
            kl_visual = kl_per_token(log_p_s, log_p_v)
            kl_knowledge = kl_per_token(log_p_s, log_p_k)
            kl_mean = 0.5 * (kl_visual + kl_knowledge)
            labels = segment_labels(processor, response_ids[:n])
            by_segment: dict[str, list[float]] = {}
            for label, value in zip(labels, cos.tolist()):
                by_segment.setdefault(label, []).append(value)
            token_pieces = [
                processor.tokenizer.decode([token_id], skip_special_tokens=True)
                for token_id in response_ids[:n].tolist()
            ]
            token_rows = [
                {
                    "token": token,
                    "segment": label,
                    "kind": token_kind(token),
                    "cosine": float(value),
                    "kl_visual": float(kv),
                    "kl_knowledge": float(kk),
                    "kl_mean": float(km),
                }
                for token, label, value, kv, kk, km in zip(
                    token_pieces,
                    labels,
                    cos.tolist(),
                    kl_visual.tolist(),
                    kl_knowledge.tolist(),
                    kl_mean.tolist(),
                )
            ]
            most_negative = sorted(
                token_rows,
                key=lambda x: x["cosine"],
            )[:10]
            result = {
                "id": sample["id"],
                "dataset": sample["dataset"],
                "task_type": sample["task_type"],
                "question": sample["question"],
                "answer_text": answer,
                "num_tokens": int(n),
                "mean_cosine": float(cos.mean().item()),
                "min_cosine": float(cos.min().item()),
                "mean_kl_visual": float(kl_visual.mean().item()),
                "mean_kl_knowledge": float(kl_knowledge.mean().item()),
                "mean_kl": float(kl_mean.mean().item()),
                "by_segment": {
                    key: {"count": len(vals), "mean": sum(vals) / len(vals), "min": min(vals)}
                    for key, vals in by_segment.items()
                },
                "most_negative_tokens": most_negative,
                "tokens": token_rows,
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
