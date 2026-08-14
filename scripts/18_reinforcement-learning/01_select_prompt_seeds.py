#!/usr/bin/env python3
"""Select grouped human seeds for the GRPO writing-prompt dataset."""

import argparse
from collections import Counter, defaultdict
import datetime as dt
import hashlib
import json
import math
from pathlib import Path
import random

from datasets import load_from_disk


PROJECT_DIR = Path(__file__).resolve().parents[2]
DEFAULT_DATASET = PROJECT_DIR / "data" / "hf-dataset"
DEFAULT_OUTPUT = PROJECT_DIR / "data" / "grpo-prompts"
DEFAULT_COUNTS = {"train": 5_000, "validation": 500, "test": 1_000}
TARGET_LENGTHS = (50, 100, 250, 500, 1_000, 2_000)


def target_words(row):
    recorded = row.get("target_words")
    if isinstance(recorded, int) and recorded > 0:
        return recorded
    words = int(row.get("word_count") or 250)
    return min(TARGET_LENGTHS, key=lambda value: abs(value - words))


def stratum(row):
    collection = str(row.get("source_collection") or "unknown")
    return f"{collection}|{target_words(row)}"


def stable_seed(seed, split, label):
    digest = hashlib.sha256(f"{seed}:{split}:{label}".encode()).digest()
    return int.from_bytes(digest[:8], "big")


def allocate_quotas(rows, count):
    availability = Counter(stratum(row) for row in rows)
    total = len(rows)
    if count > total:
        raise ValueError(f"Requested {count} rows but only {total} are available")

    exact = {
        key: count * available / total for key, available in availability.items()
    }
    quotas = {key: math.floor(value) for key, value in exact.items()}
    remaining = count - sum(quotas.values())
    order = sorted(
        availability,
        key=lambda key: (exact[key] - quotas[key], availability[key], key),
        reverse=True,
    )
    for key in order[:remaining]:
        quotas[key] += 1
    return quotas


def select_rows(
    rows, count, *, split, seed
):
    quotas = allocate_quotas(rows, count)
    by_stratum = defaultdict(list)
    for row in rows:
        by_stratum[stratum(row)].append(row)

    selected = []
    for key, candidates in sorted(by_stratum.items()):
        rng = random.Random(stable_seed(seed, split, key))
        rng.shuffle(candidates)
        selected.extend(candidates[: quotas[key]])

    rng = random.Random(stable_seed(seed, split, "final-order"))
    rng.shuffle(selected)
    if len(selected) != count:
        raise AssertionError(f"Selected {len(selected)} rows instead of {count}")
    return selected


def manifest_row(
    row, *, split, prompt_index
):
    prompt_id = f"{split}-{prompt_index:05d}"
    return {
        "prompt_id": prompt_id,
        "split": split,
        "prompt_file": f"{split}/{prompt_index:05d}.txt",
        "seed_sample_id": int(row["id"]),
        "seed_group_id": str(row["group_id"]),
        "seed_file": str(row["local_file"]),
        "seed_sha256": str(row["sha256"]),
        "seed_word_count": int(row["word_count"]),
        "target_words": target_words(row),
        "source_collection": str(row.get("source_collection") or "unknown"),
        "source_document_id": str(row.get("source_document_id") or ""),
        "source_name": str(row.get("source_name") or ""),
        "source_title": str(row.get("source_title") or ""),
        "source_url": str(row.get("source_url") or ""),
        "source_license": str(row.get("source_license") or ""),
        "source_license_url": str(row.get("source_license_url") or ""),
    }


def write_jsonl(path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8") as file:
        for row in rows:
            file.write(json.dumps(row, ensure_ascii=False) + "\n")
    temporary.replace(path)


def make_summary(rows, seed):
    split_summary = {}
    groups_by_split = {}
    for split in DEFAULT_COUNTS:
        selected = [row for row in rows if row["split"] == split]
        groups = {row["seed_group_id"] for row in selected}
        groups_by_split[split] = groups
        split_summary[split] = {
            "prompts": len(selected),
            "source_groups": len(groups),
            "source_collections": dict(
                sorted(Counter(row["source_collection"] for row in selected).items())
            ),
            "target_words": dict(
                sorted(Counter(str(row["target_words"]) for row in selected).items())
            ),
        }

    for left_index, left in enumerate(DEFAULT_COUNTS):
        for right in list(DEFAULT_COUNTS)[left_index + 1 :]:
            overlap = groups_by_split[left] & groups_by_split[right]
            if overlap:
                raise ValueError(f"{left} and {right} share {len(overlap)} groups")

    return {
        "created_at_utc": dt.datetime.now(dt.UTC).isoformat(),
        "selection_seed": seed,
        "dataset": "rasbt/human-vs-ai-50k",
        "dataset_path": "data/hf-dataset",
        "selection": "Human rows sampled proportionally by source collection and target length within the existing grouped splits.",
        "counts": dict(DEFAULT_COUNTS),
        "splits": split_summary,
        "source_group_overlap": False,
    }


def build_parser():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, default=DEFAULT_DATASET)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--seed", type=int, default=17)
    return parser


def main():
    args = build_parser().parse_args()
    dataset = load_from_disk(str(args.dataset.resolve()))
    manifest = []
    for split, count in DEFAULT_COUNTS.items():
        human_rows = [dict(row) for row in dataset[split] if int(row["label"]) == 0]
        selected = select_rows(human_rows, count, split=split, seed=args.seed)
        manifest.extend(
            manifest_row(row, split=split, prompt_index=index)
            for index, row in enumerate(selected, start=1)
        )

    output = args.output.resolve()
    write_jsonl(output / "manifest.jsonl", manifest)
    summary = make_summary(manifest, args.seed)
    (output / "summary.json").write_text(
        json.dumps(summary, indent=2) + "\n", encoding="utf-8"
    )
    print(f"Wrote {len(manifest):,} prompt seeds to {output}")


if __name__ == "__main__":
    main()
