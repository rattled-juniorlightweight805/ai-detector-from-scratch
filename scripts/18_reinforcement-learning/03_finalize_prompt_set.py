#!/usr/bin/env python3
"""Finalize and validate the generated GRPO prompt set."""

import argparse
import hashlib
import importlib.util
import json
from pathlib import Path
import random
import sys


PROJECT_DIR = Path(__file__).resolve().parents[2]
DEFAULT_ROOT = PROJECT_DIR / "data" / "grpo-prompts"
GENERATOR_PATH = Path(__file__).with_name("02_generate_ollama_prompts.py")
EXPECTED_SPLIT_COUNTS = {"train": 5_000, "validation": 500, "test": 1_000}
ORIGINAL_TEMPLATE_COUNTS = {"short": 6, "long": 8}


def load_generator():
    spec = importlib.util.spec_from_file_location("grpo_prompt_generator", GENERATOR_PATH)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Could not load {GENERATOR_PATH}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def read_jsonl(path):
    with path.open(encoding="utf-8") as file:
        return [json.loads(line) for line in file if line.strip()]


def index_unique(rows, key, label):
    indexed = {}
    for row in rows:
        value = str(row[key])
        if value in indexed:
            raise ValueError(f"Duplicate {label}: {value}")
        indexed[value] = row
    return indexed


def write_jsonl_atomic(path, rows):
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8") as file:
        for row in rows:
            file.write(json.dumps(row, ensure_ascii=False) + "\n")
    temporary.replace(path)


def choose_unique_question(
    topic,
    target_words,
    starting_index,
    used_questions,
    generator,
):
    templates = (
        generator.SHORT_QUESTION_TEMPLATES
        if target_words <= 100
        else generator.LONG_QUESTION_TEMPLATES
    )
    for offset in range(len(templates)):
        template_index = (starting_index + offset) % len(templates)
        question = templates[template_index].format(topic=topic)
        valid, reason = generator.question_is_valid(question)
        if not valid:
            raise ValueError(f"Invalid finalized question: {reason}: {question}")
        if question not in used_questions:
            return question, template_index
    raise ValueError(f"No unused question template remains for topic: {topic}")


def build_parser():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-root", type=Path, default=DEFAULT_ROOT)
    parser.add_argument("--expected-model", default="qwen3.5:4b")
    return parser


def main():
    args = build_parser().parse_args()
    root = args.dataset_root.resolve()
    generator = load_generator()
    manifest = read_jsonl(root / "manifest.jsonl")
    progress = read_jsonl(root / "generation-progress.jsonl")
    manifest_by_id = index_unique(manifest, "prompt_id", "manifest prompt ID")
    progress_by_id = index_unique(progress, "prompt_id", "progress prompt ID")
    if set(manifest_by_id) != set(progress_by_id):
        missing = sorted(set(manifest_by_id) - set(progress_by_id))
        extra = sorted(set(progress_by_id) - set(manifest_by_id))
        raise ValueError(f"Manifest/progress mismatch: missing={missing[:5]}, extra={extra[:5]}")

    split_counts = {split: 0 for split in EXPECTED_SPLIT_COUNTS}
    groups_by_split = {split: set() for split in EXPECTED_SPLIT_COUNTS}
    used_questions = set()
    finalized_progress = []
    finalized_prompts = []
    reassigned = 0

    for manifest_row in manifest:
        prompt_id = str(manifest_row["prompt_id"])
        split = str(manifest_row["split"])
        split_counts[split] += 1
        groups_by_split[split].add(str(manifest_row["seed_group_id"]))
        progress_row = dict(progress_by_id[prompt_id])
        if progress_row.get("requested_model") != args.expected_model:
            raise ValueError(
                f"{prompt_id} uses {progress_row.get('requested_model')}, "
                f"expected {args.expected_model}"
            )
        topic = str(progress_row["broad_topic"])
        topic_valid, topic_reason = generator.topic_is_valid(topic)
        if not topic_valid:
            raise ValueError(f"{prompt_id} has invalid topic: {topic_reason}")
        template_group = "short" if int(manifest_row["target_words"]) <= 100 else "long"
        selector_seed = 17 + int(manifest_row["seed_sample_id"]) * 101
        original_index = random.Random(selector_seed).randrange(
            ORIGINAL_TEMPLATE_COUNTS[template_group]
        )
        question, template_index = choose_unique_question(
            topic,
            int(manifest_row["target_words"]),
            original_index,
            used_questions,
            generator,
        )
        used_questions.add(question)
        reassigned += template_index != original_index
        question_hash = hashlib.sha256(
            (question.rstrip() + "\n").encode("utf-8")
        ).hexdigest()
        progress_row.update(
            {
                "question": question,
                "prompt_word_count": len(question.split()),
                "prompt_sha256": question_hash,
                "template_index": template_index,
                "original_template_index": original_index,
                "template_reassigned_during_finalization": template_index
                != original_index,
            }
        )
        prompt_path = root / str(manifest_row["prompt_file"])
        generator.atomic_write_text(prompt_path, question)
        finalized_progress.append(progress_row)
        finalized_prompts.append(
            {
                **manifest_row,
                "broad_topic": topic,
                "question": question,
                "prompt_word_count": len(question.split()),
                "prompt_sha256": question_hash,
                "generator_provider": progress_row["provider"],
                "generator_model": progress_row["requested_model"],
                "generation_seed": progress_row["generation_seed"],
            }
        )

    if split_counts != EXPECTED_SPLIT_COUNTS:
        raise ValueError(f"Unexpected split counts: {split_counts}")
    splits = list(EXPECTED_SPLIT_COUNTS)
    for index, left in enumerate(splits):
        for right in splits[index + 1 :]:
            overlap = groups_by_split[left] & groups_by_split[right]
            if overlap:
                raise ValueError(f"Source groups cross {left}/{right}: {sorted(overlap)[:5]}")
    if len(used_questions) != len(manifest):
        raise ValueError("Finalized questions are not unique")

    write_jsonl_atomic(root / "generation-progress.jsonl", finalized_progress)
    write_jsonl_atomic(root / "prompts.jsonl", finalized_prompts)
    summary = {
        "total": len(finalized_prompts),
        "split_counts": split_counts,
        "unique_questions": len(used_questions),
        "template_reassignments": reassigned,
        "expected_model": args.expected_model,
        "source_group_overlap": False,
    }
    (root / "finalization-summary.json").write_text(
        json.dumps(summary, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
