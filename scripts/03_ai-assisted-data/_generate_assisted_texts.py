#!/usr/bin/env python3
"""Shared implementation for paired AI-assisted text generation."""

import argparse
import json
import math
import os
import random
import re
import sys
import time
from collections import Counter
from difflib import SequenceMatcher
from pathlib import Path


AI_DATA_SCRIPT_DIR = Path(__file__).resolve().parents[1] / "02_ai-data"
sys.path.insert(0, str(AI_DATA_SCRIPT_DIR))

from _generate_ai_texts import (  # noqa: E402
    DEFAULT_API_BASES,
    DEFAULT_KEY_ENVS,
    OPENAI_MODELS,
    GenerationError,
    atomic_write_json,
    atomic_write_text,
    call_model,
    dataset_path,
    exclusive_file_lock,
    generator_metadata,
    metadata_lock_path,
    model_slug,
    read_json,
    sha256_text,
    text_for_file,
    utc_now,
    word_count,
)


DEFAULT_META = Path("data/meta.json")
DEFAULT_OUTPUT_FOLDER = Path("data/ai-assisted")
DEFAULT_COUNT = 5_071
DEFAULT_SEED = 17
DEFAULT_TIMEOUT = 300.0
EDIT_LEVELS = ("light", "moderate")

LIGHT_SYSTEM = """Edit the supplied text using only light copyediting.
Correct spelling, grammar, and punctuation. Preserve the author's wording,
voice, paragraph order, examples, facts, technical meaning, code, equations,
URLs, and citations. Do not summarize, expand, or rewrite for style. If the
text is already clean, make the smallest defensible local correction rather
than inventing new content. Return only the complete edited text without a
label, preface, explanation, quotation marks, or code fence.
"""

MODERATE_SYSTEM = """Edit the supplied text for clarity and concision.
Improve awkward wording and sentence structure. You may split, combine, or
locally reorder sentences, but preserve the author's voice, claims, examples,
technical meaning, code, equations, URLs, citations, and overall paragraph
order. Do not add facts, arguments, examples, or conclusions. Return only the
complete edited text without a label, preface, explanation, quotation marks,
or code fence.
"""


def normalize_edit(text):
    text = text.replace("\r\n", "\n").strip()
    fenced = re.fullmatch(r"```(?:text|markdown)?\s*\n?(.*?)\n?```", text, re.I | re.S)
    if fenced:
        text = fenced.group(1).strip()
    text = re.sub(
        r"^(?:edited|revised|corrected)\s+text\s*:\s*",
        "",
        text,
        flags=re.I,
    )
    return text.strip()


def similarity_ratio(original, edited):
    token_pattern = re.compile(r"\b\w+(?:['’-]\w+)*\b")
    original_words = token_pattern.findall(original.lower())
    edited_words = token_pattern.findall(edited.lower())
    if not original_words or not edited_words:
        return 0.0
    sequence_ratio = SequenceMatcher(
        None, original_words, edited_words, autojunk=False
    ).ratio()
    overlap = sum((Counter(original_words) & Counter(edited_words)).values())
    overlap_ratio = 2 * overlap / (len(original_words) + len(edited_words))
    return max(sequence_ratio, overlap_ratio)


def character_similarity_ratio(original, edited):
    return SequenceMatcher(None, original, edited, autojunk=False).ratio()


def validate_edit(original, edited, level):
    if not edited:
        return False, "is empty", 0.0
    if "as a language model" in edited.lower():
        return False, "mentions the generation process", 0.0
    if original.strip() == edited.strip():
        return False, "is unchanged", 1.0

    original_words = word_count(original)
    edited_words = word_count(edited)
    ratio = similarity_ratio(original, edited)
    if level == "light":
        low = max(1, math.floor(original_words * 0.85))
        high = math.ceil(original_words * 1.15)
        if not low <= edited_words <= high:
            return (
                False,
                f"has {edited_words} words; expected {low} to {high}",
                ratio,
            )
        if ratio < 0.82:
            return False, f"rewrites too much for light editing ({ratio:.3f})", ratio
    elif level == "moderate":
        low = max(1, math.floor(original_words * 0.65))
        high = math.ceil(original_words * 1.15)
        if not low <= edited_words <= high:
            return (
                False,
                f"has {edited_words} words; expected {low} to {high}",
                ratio,
            )
        if ratio > 0.97:
            return False, f"changes too little for moderate editing ({ratio:.3f})", ratio
        if ratio < 0.45:
            return False, f"rewrites too much for moderate editing ({ratio:.3f})", ratio
    else:
        raise AssertionError(f"Unknown edit level: {level}")
    return True, "", ratio


def update_counts(metadata):
    samples = metadata.get("samples")
    if not isinstance(samples, list):
        raise SystemExit("Metadata has no samples list")
    counts = Counter(str(sample.get("label")) for sample in samples)
    metadata["counts"] = {
        "total": len(samples),
        "human": counts["human"],
        "ai": counts["ai"],
        "ai_assisted": counts["ai-assisted"],
    }


def profile_name(args):
    return str(getattr(args, "model_selection", None) or args.model)


def job_name(args):
    return f"{args.provider}-{model_slug(profile_name(args))}"


def job_lock_path(args):
    return args.meta.parent / ".locks" / f"assisted-job-{job_name(args)}.lock"


def claims_path(data_dir):
    return data_dir / "ai-assisted-generation-claims.json"


def failures_path(data_dir):
    return data_dir / "ai-assisted-generation-failures.jsonl"


def generator_matches(sample, args):
    generator = sample.get("generator")
    if not isinstance(generator, dict) or generator.get("provider") != args.provider:
        return False
    expected = profile_name(args)
    recorded = {
        str(value)
        for value in (
            generator.get("model_selection"),
            generator.get("requested_model"),
            generator.get("model"),
        )
        if value is not None
    }
    return expected in recorded or args.model in recorded


def matching_rows(metadata, args):
    return [
        sample
        for sample in metadata.get("samples", [])
        if sample.get("label") == "ai-assisted"
        and sample.get("sample_type") == "generated-edit"
        and sample.get("edit_level") in EDIT_LEVELS
        and generator_matches(sample, args)
    ]


def load_claims(data_dir):
    path = claims_path(data_dir)
    if not path.is_file():
        return {"version": 1, "claims": {}}
    claims = read_json(path)
    if claims.get("version") != 1 or not isinstance(claims.get("claims"), dict):
        raise SystemExit(f"Unsupported claims file: {path}")
    return claims


def claim_seed_samples(args):
    data_dir = args.meta.parent
    with exclusive_file_lock(metadata_lock_path(args.meta)):
        metadata = read_json(args.meta)
        rows = matching_rows(metadata, args)
        completed = {
            level: {
                int(sample["seed_sample_id"])
                for sample in rows
                if sample.get("edit_level") == level
            }
            for level in EDIT_LEVELS
        }
        claims = load_claims(data_dir)
        all_claims = claims["claims"]
        key = job_name(args)
        claim = all_claims.get(key)
        if claim is not None:
            if int(claim.get("count", -1)) != args.count or int(
                claim.get("seed", -1)
            ) != args.seed:
                raise SystemExit(
                    f"Existing claim for {key} used another --count or --seed"
                )
            seed_ids = [int(value) for value in claim.get("seed_sample_ids", [])]
        else:
            existing_seed_ids = sorted(
                {int(sample["seed_sample_id"]) for sample in rows}
            )
            if len(existing_seed_ids) > args.count:
                raise SystemExit(
                    f"Found {len(existing_seed_ids)} existing sources for {key}, "
                    f"more than --count {args.count}"
                )
            reserved = {
                int(value)
                for other in all_claims.values()
                if isinstance(other, dict)
                for value in other.get("seed_sample_ids", [])
            }
            reserved.update(
                int(sample["seed_sample_id"])
                for sample in metadata.get("samples", [])
                if sample.get("label") == "ai-assisted"
                and sample.get("seed_sample_id") is not None
            )
            reserved.difference_update(existing_seed_ids)
            human_samples = [
                sample
                for sample in metadata.get("samples", [])
                if sample.get("label") == "human"
                and int(sample["id"]) not in reserved
                and dataset_path(data_dir, str(sample.get("file"))).is_file()
            ]
            random.Random(f"{args.seed}:ai-assisted:{key}").shuffle(human_samples)
            needed = args.count - len(existing_seed_ids)
            if len(human_samples) < needed:
                raise SystemExit(
                    f"Only {len(human_samples)} unclaimed human samples remain; "
                    f"{needed} are required"
                )
            seed_ids = existing_seed_ids + [
                int(sample["id"]) for sample in human_samples[:needed]
            ]
            all_claims[key] = {
                "provider": args.provider,
                "requested_model": args.model,
                "model_selection": getattr(args, "model_selection", None),
                "count": args.count,
                "seed": args.seed,
                "edit_levels": list(EDIT_LEVELS),
                "seed_sample_ids": seed_ids,
                "created_at_utc": utc_now(),
            }
            atomic_write_json(claims_path(data_dir), claims)

        human_by_id = {
            int(sample["id"]): sample
            for sample in metadata.get("samples", [])
            if sample.get("label") == "human"
        }
        missing = [seed_id for seed_id in seed_ids if seed_id not in human_by_id]
        if missing:
            raise SystemExit(f"Claim references missing human samples: {missing[:5]}")
        return [human_by_id[seed_id] for seed_id in seed_ids], completed


def preview_seed_samples(args):
    metadata = read_json(args.meta)
    existing = matching_rows(metadata, args)
    counts = Counter(str(sample.get("edit_level")) for sample in existing)
    human_count = sum(1 for sample in metadata.get("samples", []) if sample.get("label") == "human")
    print(f"Human samples available: {human_count}")
    print(f"Requested sources for this model: {args.count}")
    for level in EDIT_LEVELS:
        print(f"Existing {level} edits for this model: {counts[level]}")
    print("Dry run only. No claims or API requests were made.")


def record_failure(
    args,
    seed_sample,
    level,
    error,
):
    data_dir = args.meta.parent
    record = {
        "created_at_utc": utc_now(),
        "provider": args.provider,
        "model": args.model,
        "model_selection": getattr(args, "model_selection", None),
        "seed_sample_id": int(seed_sample["id"]),
        "edit_level": level,
        "error": error,
    }
    lock_path = data_dir / ".locks" / "assisted-failures.lock"
    with exclusive_file_lock(lock_path):
        path = failures_path(data_dir)
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")
            handle.flush()
            os.fsync(handle.fileno())


def edit_prompt(original):
    return f"Original text:\n<original>\n{original}\n</original>"


def generate_edit(
    args,
    seed_sample,
    original,
    level,
):
    system = LIGHT_SYSTEM if level == "light" else MODERATE_SYSTEM
    prompt = edit_prompt(original)
    max_words = max(100, math.ceil(word_count(original) * 1.20))
    last_error = "no response"
    for attempt in range(1, args.max_retries + 1):
        retry_prompt = prompt
        if attempt > 1:
            retry_prompt += (
                f"\n\nThe previous edit was rejected because it {last_error}. "
                "Apply the requested editing level more carefully and return the full text."
            )
        try:
            result = call_model(
                args,
                system,
                retry_prompt,
                max_words,
                args.seed + int(seed_sample["id"]) * 10_007 + attempt,
            )
        except GenerationError as error:
            last_error = str(error)
            print(
                f"Seed {seed_sample['id']} {level} attempt {attempt} failed: {error}",
                file=sys.stderr,
            )
            if not error.retryable:
                break
            if attempt < args.max_retries:
                time.sleep(min(2**attempt, 10))
            continue
        edited = normalize_edit(result.text)
        valid, reason, ratio = validate_edit(original, edited, level)
        if valid:
            return edited, result, ratio, None
        last_error = reason
        print(
            f"Seed {seed_sample['id']} {level} attempt {attempt} rejected: {reason}",
            file=sys.stderr,
        )
    return None, None, None, last_error


def commit_edit(
    args,
    seed_sample,
    original,
    edited,
    level,
    similarity,
    result,
):
    data_dir = args.meta.parent
    with exclusive_file_lock(metadata_lock_path(args.meta)):
        metadata = read_json(args.meta)
        for sample in matching_rows(metadata, args):
            if (
                int(sample.get("seed_sample_id", -1)) == int(seed_sample["id"])
                and sample.get("edit_level") == level
            ):
                return False

        samples = metadata.get("samples")
        if not isinstance(samples, list):
            raise SystemExit("Metadata has no samples list")
        sample_id = max((int(sample["id"]) for sample in samples), default=0) + 1
        relative_file = f"ai-assisted/{level}/{sample_id}.txt"
        output_path = dataset_path(data_dir, relative_file)
        file_text = text_for_file(edited)
        atomic_write_text(output_path, file_text)
        samples.append(
            {
                "id": sample_id,
                "file": relative_file,
                "label": "ai-assisted",
                "collection": args.provider,
                "source": f"{args.provider}-{profile_name(args)}-{level}",
                "sample_type": "generated-edit",
                "edit_level": level,
                "edit_prompt_version": 1,
                "word_count": word_count(edited),
                "sha256": sha256_text(file_text),
                "source_word_count": word_count(original),
                "source_sha256": sha256_text(text_for_file(original)),
                "word_similarity_ratio": similarity,
                "character_similarity_ratio": character_similarity_ratio(
                    original, edited
                ),
                "character_edit_magnitude": 1.0
                - character_similarity_ratio(original, edited),
                "seed_sample_id": int(seed_sample["id"]),
                "seed_file": seed_sample.get("file"),
                "seed_collection": seed_sample.get("collection"),
                "seed_source_document_id": (
                    seed_sample.get("source_document_id") or seed_sample.get("source")
                ),
                "seed_title": seed_sample.get("title"),
                "seed_source_url": seed_sample.get("source_url"),
                "seed_license": seed_sample.get("license"),
                "seed_license_url": seed_sample.get("license_url"),
                "seed_public_hub_eligible": seed_sample.get("public_hub_eligible"),
                "public_hub_eligible": bool(seed_sample.get("public_hub_eligible")),
                "generator": generator_metadata(args, result.model),
                "batch_usage": result.usage,
                "created_at_utc": utc_now(),
            }
        )
        update_counts(metadata)
        atomic_write_json(args.meta, metadata)
    return True


def completed_counts(args):
    metadata = read_json(args.meta)
    return Counter(
        str(sample.get("edit_level")) for sample in matching_rows(metadata, args)
    )


def generate_dataset(args):
    seeds, completed = claim_seed_samples(args)
    print(f"Claimed human sources: {len(seeds)}")
    print(
        "Existing edits: "
        + ", ".join(f"{level}={len(completed[level])}" for level in EDIT_LEVELS)
    )
    failures = 0
    data_dir = args.meta.parent
    for position, seed_sample in enumerate(seeds, start=1):
        seed_id = int(seed_sample["id"])
        pending_levels = [level for level in EDIT_LEVELS if seed_id not in completed[level]]
        if not pending_levels:
            continue
        original = dataset_path(data_dir, str(seed_sample["file"])).read_text(
            encoding="utf-8"
        ).strip()
        for level in pending_levels:
            edited, result, similarity, error = generate_edit(
                args, seed_sample, original, level
            )
            if edited is None or result is None or similarity is None:
                failures += 1
                record_failure(args, seed_sample, level, str(error))
                continue
            if commit_edit(
                args,
                seed_sample,
                original,
                edited,
                level,
                similarity,
                result,
            ):
                completed[level].add(seed_id)
        print(
            f"Processed source {position}/{len(seeds)}; "
            + ", ".join(f"{level}={len(completed[level])}/{args.count}" for level in EDIT_LEVELS),
            flush=True,
        )

    counts = completed_counts(args)
    incomplete = [level for level in EDIT_LEVELS if counts[level] != args.count]
    if incomplete:
        details = ", ".join(f"{level}={counts[level]}/{args.count}" for level in EDIT_LEVELS)
        raise SystemExit(
            f"Finished with {failures} failed edits and incomplete counts: {details}. "
            "Run the same command again to retry the missing edits."
        )
    print(
        "Done. " + ", ".join(f"{level}={counts[level]}" for level in EDIT_LEVELS)
    )


def build_parser(provider):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=provider != "gemini")
    parser.add_argument("--meta", type=Path, default=DEFAULT_META)
    parser.add_argument("--output-folder", type=Path, default=DEFAULT_OUTPUT_FOLDER)
    parser.add_argument("--count", type=int, default=DEFAULT_COUNT)
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument("--timeout", type=float, default=DEFAULT_TIMEOUT)
    parser.add_argument("--max-retries", type=int, default=4)
    parser.add_argument("--dry-run", action="store_true")
    if provider == "ollama":
        parser.add_argument("--ollama-url", default=DEFAULT_API_BASES["ollama"])
        parser.add_argument("--temperature", type=float, default=0.2)
        parser.add_argument("--top-p", type=float, default=0.9)
        parser.add_argument("--keep-alive", default="30m")
    else:
        parser.add_argument("--api-base", default=DEFAULT_API_BASES[provider])
        parser.add_argument("--api-key-env", default=DEFAULT_KEY_ENVS[provider])
    if provider == "openrouter":
        parser.add_argument("--temperature", type=float, default=0.2)
    if provider == "gemini":
        parser.set_defaults(model="gemini-3.6-flash")
        parser.add_argument(
            "--thinking-level",
            choices=("minimal", "low", "medium", "high"),
            default="minimal",
        )
    return parser


def validate_args(args, provider):
    args.provider = provider
    args.model = args.model.strip()
    if not args.model:
        raise SystemExit("--model must not be empty")
    if provider == "openai":
        if args.model not in OPENAI_MODELS:
            raise SystemExit(f"--model must be one of: {', '.join(OPENAI_MODELS)}")
        args.model_selection = args.model
        args.model, args.reasoning_effort = OPENAI_MODELS[args.model_selection]
    if provider == "ollama":
        args.api_base = args.ollama_url
        args.api_key = None
    elif not args.dry_run:
        args.api_key = os.environ.get(args.api_key_env)
        if not args.api_key:
            raise SystemExit(f"Set {args.api_key_env} before running this script")
    else:
        args.api_key = None
    if args.count <= 0:
        raise SystemExit("--count must be positive")
    if args.timeout <= 0:
        raise SystemExit("--timeout must be positive")
    if args.max_retries <= 0:
        raise SystemExit("--max-retries must be positive")
    if hasattr(args, "temperature") and not 0 <= args.temperature <= 2:
        raise SystemExit("--temperature must be between 0 and 2")
    if hasattr(args, "top_p") and not 0 < args.top_p <= 1:
        raise SystemExit("--top-p must be in (0, 1]")
    args.meta = args.meta.resolve()
    args.output_folder = args.output_folder.resolve()
    if args.output_folder != args.meta.parent / "ai-assisted":
        raise SystemExit("--output-folder must be the data/ai-assisted directory")


def main(provider):
    parser = build_parser(provider)
    args = parser.parse_args()
    validate_args(args, provider)
    print(f"Using {provider} model {args.model}")
    if provider == "openai":
        print(f"Model selection: {args.model_selection}, reasoning: {args.reasoning_effort}")
    if args.dry_run:
        preview_seed_samples(args)
        return
    with exclusive_file_lock(job_lock_path(args), wait=False):
        generate_dataset(args)
