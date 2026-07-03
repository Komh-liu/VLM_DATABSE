#!/usr/bin/env python3
"""Tiny Qwen2.5-VL LoRA SFT trainer for pilot teacher adapters."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
from peft import LoraConfig, get_peft_model
from qwen_vl_utils import process_vision_info
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm
from transformers import AutoProcessor, Qwen2_5_VLForConditionalGeneration


INSTRUCTION = (
    "Use exactly this format: [VISUAL] ... [KNOWLEDGE] ... "
    "[REASON] ... [ANSWER] ..."
)


class JsonlDataset(Dataset):
    def __init__(self, path: Path, limit: int | None = None):
        rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
        self.rows = rows[:limit] if limit else rows

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, idx: int) -> dict:
        return self.rows[idx]


def user_text(row: dict) -> str:
    choices = row.get("choices") or []
    choice_text = "\nChoices: " + "; ".join(str(x) for x in choices) if choices else ""
    return f"{INSTRUCTION}\nQuestion: {row['question']}{choice_text}"


def collate_one(processor, row: dict, device: torch.device):
    messages = [
        {
            "role": "user",
            "content": [
                {"type": "image", "image": row["image"]},
                {"type": "text", "text": user_text(row)},
            ],
        }
    ]
    prompt = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    full_text = prompt + row["answer"]
    image_inputs, video_inputs = process_vision_info(messages)
    prompt_inputs = processor(
        text=[prompt],
        images=image_inputs,
        videos=video_inputs,
        padding=True,
        return_tensors="pt",
    )
    full_inputs = processor(
        text=[full_text],
        images=image_inputs,
        videos=video_inputs,
        padding=True,
        return_tensors="pt",
    )
    labels = full_inputs.input_ids.clone()
    prompt_len = prompt_inputs.input_ids.shape[1]
    labels[:, :prompt_len] = -100
    labels[labels == processor.tokenizer.pad_token_id] = -100
    full_inputs["labels"] = labels
    return {k: v.to(device) if torch.is_tensor(v) else v for k, v in full_inputs.items()}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="experiment/models/Qwen2.5-VL-3B-Instruct")
    parser.add_argument("--train", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--limit", type=int, default=80)
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--rank", type=int, default=8)
    parser.add_argument("--alpha", type=int, default=16)
    parser.add_argument("--max-steps", type=int, default=80)
    args = parser.parse_args()

    device = torch.device("cuda")
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
    )
    model.config.use_cache = False
    model.gradient_checkpointing_enable()
    lora_config = LoraConfig(
        r=args.rank,
        lora_alpha=args.alpha,
        lora_dropout=0.05,
        bias="none",
        task_type="CAUSAL_LM",
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
    )
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()
    model.train()

    dataset = JsonlDataset(args.train, limit=args.limit)
    loader = DataLoader(dataset, batch_size=1, shuffle=True, collate_fn=lambda batch: batch[0])
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)

    step = 0
    losses: list[float] = []
    pbar = tqdm(total=args.max_steps)
    for _epoch in range(args.epochs):
        for row in loader:
            inputs = collate_one(processor, row, device)
            outputs = model(**inputs)
            loss = outputs.loss
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            optimizer.zero_grad(set_to_none=True)
            step += 1
            losses.append(float(loss.detach().cpu()))
            pbar.update(1)
            pbar.set_postfix(loss=f"{losses[-1]:.3f}")
            if step >= args.max_steps:
                break
        if step >= args.max_steps:
            break
    pbar.close()

    args.out.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(args.out)
    processor.save_pretrained(args.out / "processor")
    metrics = {"steps": step, "mean_loss": sum(losses) / len(losses), "last_loss": losses[-1]}
    (args.out / "train_metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    print(json.dumps(metrics, indent=2))
    print(f"saved adapter to {args.out}")


if __name__ == "__main__":
    main()
