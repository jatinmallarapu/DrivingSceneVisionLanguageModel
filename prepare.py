"""Split a normalized DriveLM JSONL manifest by scene, never by image or row."""
import argparse
import hashlib
import json
from pathlib import Path


def split_for(scene: str, seed: int) -> str:
    bucket = int(hashlib.sha256(f"{seed}:{scene}".encode()).hexdigest(), 16) % 100
    return "train" if bucket < 80 else "validation" if bucket < 90 else "test"


def prepare(source: Path, output: Path, image_root: Path, seed: int) -> None:
    rows = {split: [] for split in ("train", "validation", "test")}
    ids = set()
    for line in source.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        item = json.loads(line)
        if any(not isinstance(item[key], str) or not item[key].strip()
               for key in ("id", "scene", "image", "question", "answer")):
            raise ValueError("Every row needs nonempty id, scene, image, question and answer")
        if item["id"] in ids:
            raise ValueError(f"Duplicate row ID: {item['id']}")
        ids.add(item["id"])
        image = (image_root / item["image"]).resolve(strict=True)
        if not image.is_relative_to(image_root.resolve()) or not image.is_file():
            raise ValueError("Image must be a file inside the dataset image directory")
        if item.get("choices"):
            choices = item["choices"]
            if not isinstance(choices, dict) or item["answer"] not in choices:
                raise ValueError("Multiple-choice answers must be keys in the choices mapping")
            item["question"] += "\n" + "\n".join(f"{k}. {val}" for k, val in choices.items())
            item["question"] += "\nAnswer with the option letter only."
        item["image"] = str(image)
        rows[split_for(item["scene"], seed)].append(item)
    if any(not group for group in rows.values()):
        raise ValueError("A split is empty. Supply annotations from more scenes.")
    output.mkdir(parents=True, exist_ok=True)
    for split, group in rows.items():
        (output / f"{split}.jsonl").write_text(
            "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in group), encoding="utf-8"
        )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("source", type=Path)
    parser.add_argument("--image-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, default=Path("data/splits"))
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    prepare(args.source, args.output, args.image_root, args.seed)
