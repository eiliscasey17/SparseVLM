#!/usr/bin/env python3

import argparse
import base64
import csv
import json
from pathlib import Path


ALL_OPTIONS = ["A", "B", "C", "D"]


def is_none(value):
    if value is None:
        return True
    if isinstance(value, str) and value.strip().lower() in {"", "nan", "none"}:
        return True
    return False


def build_query(row):
    question = row["question"]
    hint = row.get("hint")
    if not is_none(hint):
        question = f"{hint}\n{question}"

    options = []
    for option_char in ALL_OPTIONS:
        option_value = row.get(option_char)
        if is_none(option_value):
            break
        options.append({"label": option_char, "text": option_value})
        question += f"\n{option_char}. {option_value}"

    return question, options


def load_rows(question_file):
    with question_file.open("r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f, delimiter="\t")
        return list(reader)


def find_row(rows, sample_index=None, row_number=None):
    if sample_index is not None:
        for row in rows:
            if str(row.get("index")) == str(sample_index):
                return row
        raise ValueError(f"Could not find a row with index={sample_index}.")

    if row_number is None:
        row_number = 0

    if row_number < 0 or row_number >= len(rows):
        raise ValueError(f"row_number={row_number} is out of range for {len(rows)} rows.")

    return rows[row_number]


def decode_image_bytes(image_b64):
    return base64.b64decode(image_b64)


def detect_image_extension(image_bytes):
    if image_bytes.startswith(b"\xff\xd8\xff"):
        return "jpg"
    if image_bytes.startswith(b"\x89PNG\r\n\x1a\n"):
        return "png"
    if image_bytes.startswith((b"GIF87a", b"GIF89a")):
        return "gif"
    if image_bytes.startswith(b"RIFF") and image_bytes[8:12] == b"WEBP":
        return "webp"
    return "img"


def main():
    parser = argparse.ArgumentParser(
        description="Extract one MMBench sample, saving its image and associated query.",
    )
    parser.add_argument(
        "--question-file",
        type=Path,
        default=Path("playground/data/eval/mmbench/mmbench_dev_20230712.tsv"),
        help="Path to the MMBench TSV file.",
    )
    parser.add_argument(
        "--sample-index",
        type=str,
        default=None,
        help="Exact value from the TSV 'index' column to extract.",
    )
    parser.add_argument(
        "--row-number",
        type=int,
        default=None,
        help="Zero-based row number to extract when --sample-index is not provided.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("playground/data/eval/mmbench/extracted_samples"),
        help="Directory where the image and query files will be written.",
    )
    parser.add_argument(
        "--output-prefix",
        type=str,
        default=None,
        help="Optional filename prefix. Defaults to mmbench_<index>.",
    )
    args = parser.parse_args()

    rows = load_rows(args.question_file)
    row = find_row(rows, sample_index=args.sample_index, row_number=args.row_number)
    query_text, options = build_query(row)

    sample_id = str(row["index"])
    output_prefix = args.output_prefix or f"mmbench_{sample_id}"

    args.output_dir.mkdir(parents=True, exist_ok=True)

    image_bytes = decode_image_bytes(row["image"])
    image_ext = detect_image_extension(image_bytes)

    image_path = args.output_dir / f"{output_prefix}.{image_ext}"
    query_path = args.output_dir / f"{output_prefix}_query.txt"
    metadata_path = args.output_dir / f"{output_prefix}_metadata.json"

    image_path.write_bytes(image_bytes)
    query_path.write_text(query_text + "\n", encoding="utf-8")

    metadata = {
        "index": row["index"],
        "question": row["question"],
        "hint": row.get("hint"),
        "query": query_text,
        "options": options,
        "answer": row.get("answer"),
        "category": row.get("category"),
        "l2-category": row.get("l2-category"),
        "split": row.get("split"),
        "image_path": str(image_path),
        "query_path": str(query_path),
    }
    metadata_path.write_text(json.dumps(metadata, indent=2), encoding="utf-8")

    print(f"Saved image to: {image_path}")
    print(f"Saved query to: {query_path}")
    print(f"Saved metadata to: {metadata_path}")


if __name__ == "__main__":
    main()
