#!/usr/bin/env python3
"""Generate baseline responses for the human-writing prompt test split."""

import argparse
from collections import defaultdict
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import sys
import tempfile
import time

from datasets import load_dataset
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer


SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_DIR = SCRIPT_DIR.parents[2]
TRAINING_SCRIPT = SCRIPT_DIR.parent / "05_train_grpo_human_writing.py"
DEFAULT_OUTPUT = SCRIPT_DIR / "test-cases-original"


def load_training_helpers():
    spec = importlib.util.spec_from_file_location(
        "grpo_human_writing_training",
        TRAINING_SCRIPT,
    )
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Could not load training script: {TRAINING_SCRIPT}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module.render_prompt, module.response_token_limit, module.resolve_device


render_prompt, response_token_limit, resolve_device = load_training_helpers()


def atomic_write_text(path, text):
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        mode="w",
        encoding="utf-8",
        dir=path.parent,
        prefix=f".{path.name}.",
        suffix=".tmp",
        delete=False,
    ) as temporary_file:
        temporary_path = Path(temporary_file.name)
        temporary_file.write(text)
        temporary_file.flush()
        os.fsync(temporary_file.fileno())
    os.replace(temporary_path, path)


def sha256_text(text):
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def dtype_for_device(device, requested):
    if requested == "float32":
        return torch.float32
    if requested == "float16":
        return torch.float16
    if requested == "bfloat16":
        return torch.bfloat16
    if device.type == "cuda":
        return torch.bfloat16
    if device.type == "mps":
        return torch.float16
    return torch.float32


def stable_batches(
    rows, batch_size
):
    if batch_size < 1:
        raise ValueError("batch_size must be at least 1")
    by_target = defaultdict(list)
    for row in rows:
        by_target[int(row["target_words"])].append(row)
    batch_index = 0
    for target_words in sorted(by_target):
        group = by_target[target_words]
        for start in range(0, len(group), batch_size):
            yield batch_index, group[start : start + batch_size]
            batch_index += 1


def record_path(output_dir, prompt_id):
    return output_dir / "records" / f"{prompt_id}.json"


def response_path(output_dir, prompt_id):
    return output_dir / "texts" / f"{prompt_id}.txt"


def load_existing_record(
    output_dir,
    row,
    run_config,
):
    metadata_path = record_path(output_dir, row["prompt_id"])
    text_path = response_path(output_dir, row["prompt_id"])
    if not metadata_path.exists() and not text_path.exists():
        return None
    if not metadata_path.is_file() or not text_path.is_file():
        raise ValueError(f"Incomplete output for {row['prompt_id']}")
    record = json.loads(metadata_path.read_text(encoding="utf-8"))
    with text_path.open("r", encoding="utf-8", newline="") as file:
        text = file.read().strip()
    expected = {
        "prompt_id": row["prompt_id"],
        "prompt_sha256": row["prompt_sha256"],
        "policy_model": run_config["policy_model"],
        "temperature": run_config["temperature"],
        "top_p": run_config["top_p"],
        "max_new_tokens": run_config["max_new_tokens"],
        "seed": run_config["seed"],
        "batch_size": run_config["batch_size"],
        "dtype": run_config["dtype"],
    }
    for key, value in expected.items():
        if record.get(key) != value:
            raise ValueError(
                f"Existing {row['prompt_id']} has a different {key}: "
                f"{record.get(key)!r} != {value!r}"
            )
    if record.get("response_sha256") != sha256_text(text):
        raise ValueError(f"Response hash mismatch for {row['prompt_id']}")
    if record.get("response") != text:
        raise ValueError(f"Response text mismatch for {row['prompt_id']}")
    return record


@torch.inference_mode()
def generate_batch(
    model,
    tokenizer,
    device,
    rows,
    *,
    max_new_tokens,
    temperature,
    top_p,
):
    prompts = [render_prompt(row) for row in rows]
    encoded = tokenizer(
        prompts,
        padding=True,
        add_special_tokens=True,
        return_tensors="pt",
    )
    encoded = {key: value.to(device) for key, value in encoded.items()}
    prompt_width = int(encoded["input_ids"].shape[1])
    generated = model.generate(
        **encoded,
        do_sample=True,
        max_new_tokens=max_new_tokens,
        temperature=temperature,
        top_p=top_p,
        eos_token_id=tokenizer.eos_token_id,
        pad_token_id=tokenizer.pad_token_id,
        use_cache=True,
    )
    response_ids = generated[:, prompt_width:]
    responses = tokenizer.batch_decode(response_ids, skip_special_tokens=True)
    texts = [response.strip() for response in responses]
    token_counts = []
    for row_ids in response_ids:
        count = int(row_ids.ne(tokenizer.pad_token_id).sum().item())
        token_counts.append(count)
    return texts, token_counts


def save_response(
    output_dir,
    row,
    response,
    response_tokens,
    *,
    batch_seed,
    token_limit,
    run_config,
    elapsed_seconds,
):
    text_path = response_path(output_dir, row["prompt_id"])
    relative_text_path = text_path.relative_to(output_dir).as_posix()
    record = {
        "prompt_id": row["prompt_id"],
        "split": "test",
        "prompt": row["prompt"],
        "prompt_sha256": row["prompt_sha256"],
        "target_words": int(row["target_words"]),
        "response_file": relative_text_path,
        "response": response,
        "response_word_count": len(response.split()),
        "response_token_count": response_tokens,
        "response_sha256": sha256_text(response),
        "policy_model": run_config["policy_model"],
        "temperature": run_config["temperature"],
        "top_p": run_config["top_p"],
        "max_new_tokens": run_config["max_new_tokens"],
        "effective_token_limit": token_limit,
        "seed": run_config["seed"],
        "batch_size": run_config["batch_size"],
        "dtype": run_config["dtype"],
        "batch_seed": batch_seed,
        "generation_seconds_per_case": elapsed_seconds,
        "group_id": row["group_id"],
        "source_collection": row["source_collection"],
        "source_document_id": row["source_document_id"],
    }
    atomic_write_text(text_path, response + "\n")
    atomic_write_text(
        record_path(output_dir, row["prompt_id"]),
        json.dumps(record, ensure_ascii=False, indent=2) + "\n",
    )
    return record


def finalize_results(
    output_dir,
    rows,
    run_config,
):
    records = []
    for row in rows:
        record = load_existing_record(output_dir, row, run_config)
        if record is not None:
            records.append(record)
    results_text = "".join(
        json.dumps(record, ensure_ascii=False) + "\n" for record in records
    )
    atomic_write_text(output_dir / "results.jsonl", results_text)
    complete = len(records) == len(rows)
    summary = {
        "dataset": run_config["dataset"],
        "split": "test",
        "expected_cases": len(rows),
        "completed_cases": len(records),
        "complete": complete,
        "policy_model": run_config["policy_model"],
        "temperature": run_config["temperature"],
        "top_p": run_config["top_p"],
        "max_new_tokens": run_config["max_new_tokens"],
        "seed": run_config["seed"],
        "batch_size": run_config["batch_size"],
        "dtype": run_config["dtype"],
    }
    atomic_write_text(
        output_dir / "summary.json",
        json.dumps(summary, indent=2) + "\n",
    )
    return len(records), complete


def parse_args():
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
        description="Generate untrained Qwen3 responses for all test prompts.",
    )
    parser.add_argument(
        "--dataset",
        default="rasbt/human-writing-prompts-6k",
    )
    parser.add_argument(
        "--policy-model",
        default="Qwen/Qwen3-0.6B-Base",
    )
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--device", choices=("auto", "cuda", "mps", "cpu"), default="auto")
    parser.add_argument(
        "--dtype",
        choices=("auto", "float32", "float16", "bfloat16"),
        default="auto",
    )
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--temperature", type=float, default=0.8)
    parser.add_argument("--top-p", type=float, default=0.9)
    parser.add_argument("--max-new-tokens", type=int, default=1616)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--max-batches",
        type=int,
        default=None,
        help="Generate at most this many incomplete batches in this run.",
    )
    args = parser.parse_args()
    if args.batch_size < 1:
        parser.error("--batch-size must be at least 1")
    if args.temperature <= 0:
        parser.error("--temperature must be positive")
    if not 0 < args.top_p <= 1:
        parser.error("--top-p must be in (0, 1]")
    if args.max_new_tokens < 1:
        parser.error("--max-new-tokens must be at least 1")
    if args.max_batches is not None and args.max_batches < 1:
        parser.error("--max-batches must be at least 1")
    return args


def main():
    args = parse_args()
    output_dir = args.output_dir.expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    rows = list(load_dataset(args.dataset, split="test"))
    if len(rows) != 1_000:
        raise ValueError(f"Expected 1,000 test cases, found {len(rows):,}")

    run_config = {
        "dataset": args.dataset,
        "policy_model": args.policy_model,
        "temperature": args.temperature,
        "top_p": args.top_p,
        "max_new_tokens": args.max_new_tokens,
        "seed": args.seed,
        "batch_size": args.batch_size,
        "dtype": args.dtype,
    }
    config_path = output_dir / "run-config.json"
    if config_path.exists():
        existing_config = json.loads(config_path.read_text(encoding="utf-8"))
        if existing_config != run_config:
            raise ValueError(
                "Output directory belongs to a different generation run:\n"
                f"existing={existing_config}\nrequested={run_config}"
            )
    else:
        atomic_write_text(
            config_path,
            json.dumps(run_config, indent=2) + "\n",
        )

    existing = {
        row["prompt_id"]
        for row in rows
        if load_existing_record(output_dir, row, run_config) is not None
    }
    print(f"Existing responses: {len(existing):,}/1,000")
    if len(existing) == len(rows):
        finalize_results(output_dir, rows, run_config)
        print(f"Already complete: {output_dir}")
        return

    device = resolve_device(args.device)
    dtype = dtype_for_device(device, args.dtype)
    print(f"Policy model: {args.policy_model}")
    print(f"Device: {device}; dtype: {dtype}")
    tokenizer = AutoTokenizer.from_pretrained(args.policy_model)
    tokenizer.padding_side = "left"
    if tokenizer.eos_token_id is None:
        raise ValueError("Policy tokenizer must define an EOS token")
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(args.policy_model, dtype=dtype)
    model.to(device)
    model.eval()

    completed = len(existing)
    generated_batches = 0
    for batch_index, batch_rows in stable_batches(rows, args.batch_size):
        missing_ids = {
            row["prompt_id"] for row in batch_rows if row["prompt_id"] not in existing
        }
        if not missing_ids:
            continue
        if args.max_batches is not None and generated_batches >= args.max_batches:
            break
        batch_seed = args.seed + batch_index
        torch.manual_seed(batch_seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(batch_seed)
        token_limit = response_token_limit(
            int(batch_rows[0]["target_words"]),
            args.max_new_tokens,
        )
        start = time.perf_counter()
        responses, token_counts = generate_batch(
            model,
            tokenizer,
            device,
            batch_rows,
            max_new_tokens=token_limit,
            temperature=args.temperature,
            top_p=args.top_p,
        )
        elapsed = time.perf_counter() - start
        per_case_seconds = elapsed / len(batch_rows)
        for row, response, token_count in zip(
            batch_rows, responses, token_counts, strict=True
        ):
            if row["prompt_id"] not in missing_ids:
                continue
            save_response(
                output_dir,
                row,
                response,
                token_count,
                batch_seed=batch_seed,
                token_limit=token_limit,
                run_config=run_config,
                elapsed_seconds=per_case_seconds,
            )
            existing.add(row["prompt_id"])
            completed += 1
        generated_batches += 1
        print(
            f"Generated {completed:,}/1,000 | target={batch_rows[0]['target_words']} "
            f"| batch={len(batch_rows)} | {elapsed:.1f}s"
        )

    completed, complete = finalize_results(output_dir, rows, run_config)
    print(f"Completed responses: {completed:,}/1,000")
    print(f"Output: {output_dir}")
    if not complete:
        print("Run the same command again to resume.")


if __name__ == "__main__":
    main()
