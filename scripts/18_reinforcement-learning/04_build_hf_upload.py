#!/usr/bin/env python3
"""Build Hub-ready Parquet splits for the human-writing prompt dataset."""

import argparse
from collections import Counter, defaultdict
import hashlib
import json
from pathlib import Path

from datasets import Dataset, Features, Value


PROJECT_DIR = Path(__file__).resolve().parents[2]
DEFAULT_INPUT = PROJECT_DIR / "data" / "grpo-prompts" / "prompts.jsonl"
DEFAULT_OUTPUT = PROJECT_DIR / "data" / "grpo-prompts" / "hf-upload"
SPLIT_COUNTS = {"train": 5_000, "validation": 500, "test": 1_000}
SPLIT_NAMES = tuple(SPLIT_COUNTS)

FEATURES = Features(
    {
        "prompt_id": Value("string"),
        "prompt": Value("string"),
        "split": Value("string"),
        "target_words": Value("int64"),
        "broad_topic": Value("string"),
        "group_id": Value("string"),
        "prompt_word_count": Value("int64"),
        "prompt_sha256": Value("string"),
        "seed_sample_id": Value("int64"),
        "seed_word_count": Value("int64"),
        "source_collection": Value("string"),
        "source_document_id": Value("string"),
        "source_name": Value("string"),
        "source_title": Value("string"),
        "source_url": Value("string"),
        "source_license": Value("string"),
        "source_license_url": Value("string"),
        "generator_provider": Value("string"),
        "generator_model": Value("string"),
        "generation_seed": Value("int64"),
    }
)


def read_jsonl(path):
    with path.open(encoding="utf-8") as file:
        return [json.loads(line) for line in file if line.strip()]


def optional_string(value):
    if value is None:
        return None
    text = str(value).strip()
    return text or None


def make_record(row, dataset_root):
    prompt = str(row["question"]).strip()
    relative_path = Path(str(row["prompt_file"]))
    prompt_path = (dataset_root / relative_path).resolve()
    try:
        prompt_path.relative_to(dataset_root.resolve())
    except ValueError as error:
        raise ValueError(f"Prompt path leaves the dataset root: {relative_path}") from error
    if not prompt_path.is_file():
        raise ValueError(f"Prompt file is missing: {relative_path}")
    file_text = prompt_path.read_text(encoding="utf-8").strip()
    if file_text != prompt:
        raise ValueError(f"Prompt file disagrees with metadata: {relative_path}")
    word_count = len(prompt.split())
    if word_count != int(row["prompt_word_count"]):
        raise ValueError(f"Prompt word count disagrees: {row['prompt_id']}")
    return {
        "prompt_id": str(row["prompt_id"]),
        "prompt": prompt,
        "split": str(row["split"]),
        "target_words": int(row["target_words"]),
        "broad_topic": str(row["broad_topic"]),
        "group_id": str(row["seed_group_id"]),
        "prompt_word_count": word_count,
        "prompt_sha256": hashlib.sha256(prompt.encode("utf-8")).hexdigest(),
        "seed_sample_id": int(row["seed_sample_id"]),
        "seed_word_count": int(row["seed_word_count"]),
        "source_collection": str(row["source_collection"]),
        "source_document_id": str(row["source_document_id"]),
        "source_name": optional_string(row.get("source_name")),
        "source_title": optional_string(row.get("source_title")),
        "source_url": optional_string(row.get("source_url")),
        "source_license": optional_string(row.get("source_license")),
        "source_license_url": optional_string(row.get("source_license_url")),
        "generator_provider": str(row["generator_provider"]),
        "generator_model": str(row["generator_model"]),
        "generation_seed": int(row["generation_seed"]),
    }


def verify_records(records):
    if len(records) != sum(SPLIT_COUNTS.values()):
        raise ValueError(f"Expected 6,500 records, found {len(records):,}")
    for field in ("prompt_id", "prompt", "prompt_sha256"):
        if len({record[field] for record in records}) != len(records):
            raise ValueError(f"Duplicate {field} values found")
    counts = Counter(record["split"] for record in records)
    if dict(counts) != SPLIT_COUNTS:
        raise ValueError(f"Unexpected split counts: {dict(counts)}")
    group_splits = defaultdict(set)
    for record in records:
        group_splits[record["group_id"]].add(record["split"])
    overlap = [group for group, splits in group_splits.items() if len(splits) > 1]
    if overlap:
        raise ValueError(f"Source groups cross splits: {overlap[:5]}")
    models = {record["generator_model"] for record in records}
    if models != {"qwen3.5:4b"}:
        raise ValueError(f"Unexpected generator models: {sorted(models)}")


def dataset_card(split_bytes, parquet_bytes):
    feature_lines = "\n".join(
        f"  - name: {name}\n    dtype: {feature.dtype}"
        for name, feature in FEATURES.items()
    )
    split_lines = "\n".join(
        f"  - name: {split}\n"
        f"    num_bytes: {split_bytes[split]}\n"
        f"    num_examples: {SPLIT_COUNTS[split]}"
        for split in SPLIT_NAMES
    )
    dataset_bytes = sum(split_bytes.values())
    return f"""---
dataset_info:
  features:
{feature_lines}
  splits:
{split_lines}
  download_size: {parquet_bytes}
  dataset_size: {dataset_bytes}
configs:
- config_name: default
  data_files:
  - split: train
    path: data/train-*
  - split: validation
    path: data/validation-*
  - split: test
    path: data/test-*
license: other
task_categories:
- text-generation
language:
- en
pretty_name: Human Writing Prompts 6K
size_categories:
- 1K<n<10K
---

# Human Writing Prompts 6K

This dataset contains 6,500 unique English writing prompts for experiments on
human-style text generation. The prompts were constructed from broad topics
extracted from human-written source texts. The source texts themselves are not
included.

| Split | Prompts |
|---|---:|
| Train | 5,000 |
| Validation | 500 |
| Test | 1,000 |

The source-document groups do not cross split boundaries. Each row retains the
source collection, document, URL, and license metadata of the human text used
as topic inspiration. The prompts were generated with Qwen3.5 4B.

Load all three splits with:

```python
from datasets import load_dataset

dataset = load_dataset("rasbt/human-writing-prompts-6k")
```

More details coming soon.
"""


def build_parser():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    return parser


def main():
    args = build_parser().parse_args()
    input_path = args.input.resolve()
    output_dir = args.output_dir.resolve()
    if output_dir.exists():
        raise SystemExit(f"Output directory already exists: {output_dir}")
    rows = read_jsonl(input_path)
    records = [make_record(row, input_path.parent) for row in rows]
    verify_records(records)
    data_dir = output_dir / "data"
    data_dir.mkdir(parents=True)
    split_bytes = {}
    parquet_bytes = 0
    for split in SPLIT_NAMES:
        dataset = Dataset.from_list(
            [record for record in records if record["split"] == split],
            features=FEATURES,
        )
        parquet_path = data_dir / f"{split}-00000-of-00001.parquet"
        dataset.to_parquet(str(parquet_path))
        reloaded = Dataset.from_parquet(str(parquet_path))
        if len(reloaded) != SPLIT_COUNTS[split] or reloaded.features != FEATURES:
            raise ValueError(f"Saved Parquet verification failed for {split}")
        split_bytes[split] = dataset.data.nbytes
        parquet_bytes += parquet_path.stat().st_size
    (output_dir / "README.md").write_text(
        dataset_card(split_bytes, parquet_bytes), encoding="utf-8"
    )
    print(f"Built Hub upload folder: {output_dir}")
    for split in SPLIT_NAMES:
        print(f"{split}: {SPLIT_COUNTS[split]:,} prompts")
    print("Verified: unique prompts, expected generator, and no group overlap")


if __name__ == "__main__":
    main()
