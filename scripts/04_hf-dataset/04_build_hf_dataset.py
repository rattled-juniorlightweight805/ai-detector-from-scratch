#!/usr/bin/env python3
"""Build and verify a local Hugging Face DatasetDict from data/meta.json."""

import argparse
import hashlib
import json
from collections import Counter, defaultdict
from pathlib import Path

from datasets import ClassLabel, Dataset, DatasetDict, Features, List, Value, load_from_disk
from sklearn.model_selection import StratifiedGroupKFold


PROJECT_DIR = Path(__file__).resolve().parents[2]
DEFAULT_META = PROJECT_DIR / "data" / "meta.json"
DEFAULT_OUTPUT = PROJECT_DIR / "data" / "hf-dataset"
SPLIT_NAMES = ("train", "validation", "test")
SPLIT_TARGETS = {"train": 0.8, "validation": 0.1, "test": 0.1}
N_FOLDS = 10

FEATURES = Features(
    {
        "id": Value("int64"),
        "text": Value("string"),
        "label": ClassLabel(names=["human", "ai"]),
        "split": Value("string"),
        "group_id": Value("string"),
        "local_file": Value("string"),
        "word_count": Value("int64"),
        "sha256": Value("string"),
        "sample_type": Value("string"),
        "text_collection": Value("string"),
        "source_collection": Value("string"),
        "source_document_id": Value("string"),
        "source_name": Value("string"),
        "source_title": Value("string"),
        "source_url": Value("string"),
        "source_license": Value("string"),
        "source_license_url": Value("string"),
        "source_authors": List(Value("string")),
        "attribution_name": Value("string"),
        "attribution_url": Value("string"),
        "generator_provider": Value("string"),
        "generator_model": Value("string"),
        "seed_sample_id": Value("int64"),
        "target_words": Value("int64"),
        "public_hub_eligible": Value("bool"),
    }
)


def optional_string(value):
    if value is None:
        return None
    text = str(value).strip()
    return text or None


def string_list(value):
    if value is None:
        return []
    values = value if isinstance(value, list) else [value]
    return [str(item) for item in values if str(item).strip()]


def generator_model(sample):
    generator = sample.get("generator") or {}
    return optional_string(
        generator.get("model_selection")
        or generator.get("requested_model")
        or generator.get("model")
    )


def safe_data_path(data_dir, relative_file):
    path = (data_dir / relative_file).resolve()
    try:
        path.relative_to(data_dir.resolve())
    except ValueError as error:
        raise SystemExit(f"Metadata path leaves the data directory: {relative_file}") from error
    return path


def source_record(
    sample, human_by_id
):
    if sample["label"] == "human":
        return sample, None
    seed_id = sample.get("seed_sample_id")
    if seed_id is None or int(seed_id) not in human_by_id:
        raise SystemExit(f"AI sample {sample['id']} has no valid human seed")
    seed = human_by_id[int(seed_id)]
    declared_document = sample.get("seed_source_document_id")
    if declared_document and declared_document != seed.get("source_document_id"):
        raise SystemExit(f"AI sample {sample['id']} disagrees with its seed document")
    declared_collection = sample.get("seed_collection")
    if declared_collection and declared_collection != seed.get("collection"):
        raise SystemExit(f"AI sample {sample['id']} disagrees with its seed collection")
    return seed, int(seed_id)


def make_record(
    sample,
    human_by_id,
    data_dir,
):
    source, seed_id = source_record(sample, human_by_id)
    source_collection = optional_string(source.get("collection"))
    source_document_id = optional_string(source.get("source_document_id"))
    if not source_collection or not source_document_id:
        raise SystemExit(f"Sample {sample['id']} has incomplete source provenance")
    if source.get("public_hub_eligible") is not True:
        raise SystemExit(f"Sample {sample['id']} uses a seed that is not Hub-eligible")

    relative_file = str(sample.get("file"))
    path = safe_data_path(data_dir, relative_file)
    if not path.is_file():
        raise SystemExit(f"Dataset file does not exist: {relative_file}")
    raw = path.read_bytes()
    raw_hash = hashlib.sha256(raw).hexdigest()
    text_bytes = raw.rstrip(b"\n")
    text_hash = hashlib.sha256(text_bytes).hexdigest()
    expected_hash = sample.get("sha256")
    if not expected_hash or expected_hash not in {raw_hash, text_hash}:
        raise SystemExit(f"SHA-256 mismatch for {relative_file}")
    try:
        text = text_bytes.decode("utf-8")
    except UnicodeDecodeError as error:
        raise SystemExit(f"Dataset file is not valid UTF-8: {relative_file}") from error
    if not text.strip():
        raise SystemExit(f"Dataset file is empty: {relative_file}")
    count = len(text.split())
    if count != int(sample.get("word_count", -1)):
        raise SystemExit(f"Word-count mismatch for {relative_file}")

    generator = sample.get("generator") or {}
    target_words = (
        sample.get("target_response_words")
        if sample["label"] == "ai"
        else sample.get("target_words")
    )
    source_name = optional_string(source.get("source_name"))
    source_url = optional_string(source.get("source_url"))
    source_authors = string_list(source.get("authors"))
    attribution_name = (
        optional_string(source.get("attribution_name"))
        or (", ".join(source_authors) if source_authors else None)
        or source_name
    )
    attribution_url = optional_string(source.get("attribution_url")) or source_url
    return {
        "id": int(sample["id"]),
        "text": text,
        "label": str(sample["label"]),
        "split": None,
        "group_id": f"{source_collection}:{source_document_id}",
        "local_file": relative_file,
        "word_count": count,
        "sha256": text_hash,
        "sample_type": optional_string(sample.get("sample_type")),
        "text_collection": optional_string(sample.get("collection")),
        "source_collection": source_collection,
        "source_document_id": source_document_id,
        "source_name": source_name,
        "source_title": optional_string(source.get("title")),
        "source_url": source_url,
        "source_license": optional_string(source.get("license")),
        "source_license_url": optional_string(source.get("license_url")),
        "source_authors": source_authors,
        "attribution_name": attribution_name,
        "attribution_url": attribution_url,
        "generator_provider": optional_string(generator.get("provider")),
        "generator_model": generator_model(sample),
        "seed_sample_id": seed_id,
        "target_words": int(target_words) if target_words is not None else None,
        "public_hub_eligible": True,
    }


def load_records(meta_path):
    metadata = json.loads(meta_path.read_text(encoding="utf-8"))
    samples = metadata.get("samples")
    if not isinstance(samples, list):
        raise SystemExit("Metadata has no samples list")
    human_by_id = {
        int(sample["id"]): sample
        for sample in samples
        if sample.get("label") == "human"
    }
    excluded = Counter()
    records = []
    for sample in sorted(samples, key=lambda item: int(item["id"])):
        label = sample.get("label")
        if label not in {"human", "ai"}:
            excluded[str(label)] += 1
            continue
        if sample.get("public_hub_eligible") is not True:
            excluded[f"{label}-not-public"] += 1
            continue
        records.append(make_record(sample, human_by_id, meta_path.parent))
    if not records:
        raise SystemExit("No Hub-eligible human or AI records were found")
    return records, excluded


def assign_splits(records, seed):
    groups = [record["group_id"] for record in records]
    if len(set(groups)) < N_FOLDS:
        raise SystemExit(f"At least {N_FOLDS} source groups are required")
    strata = [
        f"{record['label']}|{record['source_collection']}" for record in records
    ]
    small_strata = {
        name: count for name, count in Counter(strata).items() if count < N_FOLDS
    }
    if small_strata:
        raise SystemExit(f"Split strata contain fewer than {N_FOLDS} rows: {small_strata}")

    assigned = [dict(record) for record in records]
    folds = [None] * len(records)
    splitter = StratifiedGroupKFold(
        n_splits=N_FOLDS,
        shuffle=True,
        random_state=seed,
    )
    indices = list(range(len(records)))
    for fold, (_, held_out) in enumerate(splitter.split(indices, strata, groups)):
        for index in held_out:
            folds[int(index)] = fold
    if any(fold is None for fold in folds):
        raise AssertionError("The splitter did not assign every record")
    for record, fold in zip(assigned, folds):
        record["split"] = "test" if fold == 0 else "validation" if fold == 1 else "train"
    return assigned


def verify_records(records):
    if len({record["id"] for record in records}) != len(records):
        raise SystemExit("Duplicate dataset IDs found")
    if len({record["local_file"] for record in records}) != len(records):
        raise SystemExit("Duplicate local file paths found")
    if len({record["sha256"] for record in records}) != len(records):
        raise SystemExit("Duplicate text hashes found")

    group_splits = defaultdict(set)
    for record in records:
        group_splits[record["group_id"]].add(record["split"])
    leaked = [group for group, splits in group_splits.items() if len(splits) != 1]
    if leaked:
        raise SystemExit(f"Source groups cross dataset splits: {leaked[:5]}")

    total = len(records)
    global_ai_share = sum(record["label"] == "ai" for record in records) / total
    split_counts = {}
    for split in SPLIT_NAMES:
        split_rows = [record for record in records if record["split"] == split]
        if not split_rows:
            raise SystemExit(f"Split is empty: {split}")
        ratio = len(split_rows) / total
        if abs(ratio - SPLIT_TARGETS[split]) > 0.015:
            raise SystemExit(f"Split ratio is outside tolerance: {split}={ratio:.4f}")
        labels = Counter(record["label"] for record in split_rows)
        if set(labels) != {"human", "ai"}:
            raise SystemExit(f"Split does not contain both labels: {split}")
        ai_share = labels["ai"] / len(split_rows)
        if abs(ai_share - global_ai_share) > 0.02:
            raise SystemExit(f"Class balance is outside tolerance in {split}")
        split_counts[split] = {
            "rows": len(split_rows),
            "share": round(ratio, 6),
            "labels": dict(sorted(labels.items())),
            "source_groups": len({record["group_id"] for record in split_rows}),
            "source_collections": dict(
                sorted(Counter(record["source_collection"] for record in split_rows).items())
            ),
            "generator_models": dict(
                sorted(
                    Counter(
                        record["generator_model"] or "not-applicable"
                        for record in split_rows
                    ).items()
                )
            ),
        }

    return {
        "total_rows": total,
        "label_encoding": {"human": 0, "ai": 1},
        "split_method": (
            "10-fold StratifiedGroupKFold over label and source collection; "
            "fold 0=test, fold 1=validation, folds 2-9=train"
        ),
        "source_group_key": "source_collection:source_document_id",
        "source_group_overlap": False,
        "duplicate_text_hashes": 0,
        "splits": split_counts,
    }


def to_dataset_dict(records):
    return DatasetDict(
        {
            split: Dataset.from_list(
                [record for record in records if record["split"] == split],
                features=FEATURES,
            )
            for split in SPLIT_NAMES
        }
    )


def verify_saved_dataset(output_dir, summary):
    loaded = load_from_disk(str(output_dir))
    if not isinstance(loaded, DatasetDict):
        raise SystemExit("Saved output did not reload as a DatasetDict")
    if set(loaded) != set(SPLIT_NAMES):
        raise SystemExit(f"Saved DatasetDict has unexpected splits: {sorted(loaded)}")
    for split in SPLIT_NAMES:
        if len(loaded[split]) != summary["splits"][split]["rows"]:
            raise SystemExit(f"Saved row count changed for {split}")
        label = loaded[split].features["label"]
        if not isinstance(label, ClassLabel) or label.names != ["human", "ai"]:
            raise SystemExit("Saved ClassLabel encoding is incorrect")


def build_parser():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--meta", type=Path, default=DEFAULT_META)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--seed", type=int, default=17)
    return parser


def main():
    args = build_parser().parse_args()
    meta_path = args.meta.resolve()
    output_dir = args.output_dir.resolve()
    if output_dir.exists():
        raise SystemExit(f"Output already exists: {output_dir}")

    records, excluded = load_records(meta_path)
    records = assign_splits(records, args.seed)
    summary = verify_records(records)
    summary["seed"] = args.seed
    summary["excluded_metadata_rows"] = dict(sorted(excluded.items()))

    dataset = to_dataset_dict(records)
    dataset.save_to_disk(str(output_dir))
    (output_dir / "summary.json").write_text(
        json.dumps(summary, indent=2) + "\n", encoding="utf-8"
    )
    verify_saved_dataset(output_dir, summary)

    print(f"Saved DatasetDict to {output_dir}")
    for split in SPLIT_NAMES:
        counts = summary["splits"][split]
        print(
            f"{split}: {counts['rows']:,} rows "
            f"({counts['labels']['human']:,} human, {counts['labels']['ai']:,} AI)"
        )
    print("Verified: no source-group overlap and no duplicate text hashes")


if __name__ == "__main__":
    main()
