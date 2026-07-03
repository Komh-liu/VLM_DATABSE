#!/usr/bin/env python3
"""Create tiny SFT datasets for lightweight visual/knowledge LoRA teachers."""

from __future__ import annotations

import argparse
import json
from io import BytesIO
from pathlib import Path

import pyarrow.parquet as pq
from PIL import Image


def write_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def make_visual_rows(repo: Path, limit: int) -> list[dict]:
    root = repo / "experiment/data/vstar_bench/vstar_data"
    rows: list[dict] = []
    for json_path in sorted(root.glob("*/*.json")):
        image_path = None
        for ext in [".jpg", ".JPG", ".jpeg", ".png", ".webp"]:
            candidate = json_path.with_suffix(ext)
            if candidate.exists():
                image_path = candidate
                break
        if image_path is None:
            continue
        data = json.loads(json_path.read_text(encoding="utf-8"))
        target = ", ".join(data.get("target_object") or ["the target object"])
        bbox = data.get("bbox") or []
        bbox_text = f" Approximate bounding box: {bbox[0]}." if bbox else ""
        answer = (
            f"[VISUAL] The image contains {target}. {bbox_text} "
            f"The visual evidence should be checked directly in the image. "
            f"[KNOWLEDGE] No external knowledge is needed. "
            f"[REASON] Use the visible object, attributes, and spatial evidence only. "
            f"[ANSWER] {target}."
        )
        rows.append(
            {
                "id": f"visual-{json_path.parent.name}-{json_path.stem}",
                "image": str(image_path),
                "question": data.get("question", "Describe the target object in the image."),
                "choices": data.get("options", []),
                "answer": answer,
            }
        )
        if len(rows) >= limit:
            break
    return rows


def make_knowledge_rows(repo: Path, limit: int) -> list[dict]:
    parquet = repo / "experiment/data/aokvqa/data/data/train-00000-of-00002-c1d24de3bacb5e0c.parquet"
    image_dir = repo / "experiment/probes/lora_train_images"
    image_dir.mkdir(parents=True, exist_ok=True)
    rows: list[dict] = []
    for idx, row in enumerate(pq.read_table(parquet).to_pylist()):
        choices = row.get("choices") or []
        correct = None
        if choices and row.get("correct_choice_idx") is not None:
            correct = choices[int(row["correct_choice_idx"])]
        rationales = row.get("rationales") or []
        rationale = rationales[0] if rationales else ""
        image_path = image_dir / f"aokvqa_train_{idx:05d}.jpg"
        if not image_path.exists():
            image = Image.open(BytesIO(row["image"]["bytes"])).convert("RGB")
            image.save(image_path, quality=92)
        answer = (
            f"[VISUAL] Identify only the key visible clue needed for the question. "
            f"[KNOWLEDGE] {rationale} "
            f"[REASON] Combine the visible clue with the relevant commonsense fact. "
            f"[ANSWER] {correct}."
        )
        rows.append(
            {
                "id": f"knowledge-aokvqa-{idx:05d}",
                "image": str(image_path),
                "question": row["question"],
                "choices": choices,
                "answer": answer,
            }
        )
        if len(rows) >= limit:
            break
    return rows


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo", type=Path, default=Path("."))
    parser.add_argument("--visual-limit", type=int, default=160)
    parser.add_argument("--knowledge-limit", type=int, default=160)
    parser.add_argument("--out-dir", type=Path, default=Path("experiment/probes/lora_data"))
    args = parser.parse_args()

    repo = args.repo.resolve()
    out_dir = repo / args.out_dir
    visual_rows = make_visual_rows(repo, args.visual_limit)
    knowledge_rows = make_knowledge_rows(repo, args.knowledge_limit)
    write_jsonl(out_dir / "visual_train.jsonl", visual_rows)
    write_jsonl(out_dir / "knowledge_train.jsonl", knowledge_rows)
    print(f"visual rows: {len(visual_rows)} -> {out_dir / 'visual_train.jsonl'}")
    print(f"knowledge rows: {len(knowledge_rows)} -> {out_dir / 'knowledge_train.jsonl'}")


if __name__ == "__main__":
    main()
