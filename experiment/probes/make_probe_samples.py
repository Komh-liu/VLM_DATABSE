#!/usr/bin/env python3
"""Build a tiny local probe set for VLM conflict experiments."""

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


def add_aokvqa(rows: list[dict], repo: Path, image_dir: Path, limit: int) -> None:
    parquet = repo / "experiment/data/aokvqa/data/data/validation-00000-of-00001-b2bd0de231b6326a.parquet"
    table = pq.read_table(parquet).to_pylist()
    image_dir.mkdir(parents=True, exist_ok=True)

    for idx, row in enumerate(table[:limit]):
        image_path = image_dir / f"aokvqa_{idx:04d}.jpg"
        if not image_path.exists():
            image = Image.open(BytesIO(row["image"]["bytes"])).convert("RGB")
            image.save(image_path, quality=95)

        choices = row.get("choices") or []
        answer = None
        if choices and row.get("correct_choice_idx") is not None:
            answer = choices[int(row["correct_choice_idx"])]

        rows.append(
            {
                "id": f"aokvqa-{idx:04d}",
                "dataset": "aokvqa",
                "task_type": "mixed_visual_knowledge",
                "image": str(image_path),
                "question": row["question"],
                "choices": choices,
                "answer": answer,
                "rationales": row.get("rationales") or [],
            }
        )


def add_vstar(rows: list[dict], repo: Path, limit: int) -> None:
    root = repo / "experiment/data/vstar_bench/vstar_data"
    count = 0
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
        rows.append(
            {
                "id": f"vstar-{json_path.parent.name}-{json_path.stem}",
                "dataset": "vstar",
                "task_type": "visual_grounding",
                "image": str(image_path),
                "question": data.get("question", ""),
                "choices": data.get("options", []),
                "answer": None,
                "target_object": data.get("target_object"),
                "bbox": data.get("bbox"),
            }
        )
        count += 1
        if count >= limit:
            return


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo", type=Path, default=Path("."))
    parser.add_argument("--out", type=Path, default=Path("experiment/probes/probe_samples.jsonl"))
    parser.add_argument("--aokvqa", type=int, default=8)
    parser.add_argument("--vstar", type=int, default=4)
    args = parser.parse_args()

    repo = args.repo.resolve()
    rows: list[dict] = []
    add_aokvqa(rows, repo, repo / "experiment/probes/cache_images", args.aokvqa)
    add_vstar(rows, repo, args.vstar)
    write_jsonl(repo / args.out, rows)
    print(f"wrote {len(rows)} samples to {repo / args.out}")


if __name__ == "__main__":
    main()
