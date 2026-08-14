#!/usr/bin/env python3
"""Generate test responses from one saved GRPO policy checkpoint."""

import argparse
import json
import os
from pathlib import Path
import sys


SCRIPT_DIR = Path(__file__).resolve().parent
BASELINE_SCRIPT = SCRIPT_DIR / "06_generate_original_test_cases.py"
DEFAULT_CHECKPOINT_ROOT = (
    SCRIPT_DIR.parent / "checkpoints" / "05_train_grpo_human_writing"
)


def checkpoint_problem(path):
    if not path.is_dir() or not (path / "config.json").is_file():
        return "config.json is missing"
    weight_patterns = (
        "model*.safetensors",
        "pytorch_model*.bin",
        "model*.index.json",
        "pytorch_model*.index.json",
    )
    weight_files = [
        weight_file
        for pattern in weight_patterns
        for weight_file in path.glob(pattern)
    ]
    if not weight_files:
        return "model weights are missing"
    has_tokenizer = (path / "tokenizer.json").is_file() or (
        path / "tokenizer_config.json"
    ).is_file()
    if not has_tokenizer:
        return "tokenizer files are missing"

    for weight_file in weight_files:
        if weight_file.suffix == ".safetensors":
            try:
                from safetensors import safe_open

                with safe_open(weight_file, framework="pt", device="cpu") as file:
                    if not list(file.keys()):
                        return f"{weight_file.name} contains no tensors"
            except Exception as error:
                return f"{weight_file.name} is invalid: {error}"
        elif weight_file.name.endswith(".index.json"):
            try:
                index = json.loads(weight_file.read_text(encoding="utf-8"))
                shards = set(index["weight_map"].values())
            except Exception as error:
                return f"{weight_file.name} is invalid: {error}"
            missing = sorted(shard for shard in shards if not (path / shard).is_file())
            if missing:
                return f"missing weight shard: {missing[0]}"
    return None


def is_checkpoint(path):
    return checkpoint_problem(path) is None


def available_checkpoints(root):
    if not root.is_dir():
        return []
    return sorted(path for path in root.iterdir() if is_checkpoint(path))


def resolve_checkpoint(value, root):
    supplied = Path(value).expanduser()
    path = supplied if supplied.is_absolute() else root / supplied
    path = path.resolve()
    problem = checkpoint_problem(path)
    if problem is not None:
        raise ValueError(
            f"Checkpoint is incomplete or invalid: {path}. {problem}"
        )
    return path


def default_output_dir(checkpoint):
    return SCRIPT_DIR / f"test-cases-{checkpoint.name}"


def build_generation_command(
    args,
    checkpoint,
    output_dir,
):
    command = [
        sys.executable,
        "-u",
        str(BASELINE_SCRIPT),
        "--dataset",
        args.dataset,
        "--policy-model",
        str(checkpoint),
        "--output-dir",
        str(output_dir),
        "--device",
        args.device,
        "--dtype",
        args.dtype,
        "--batch-size",
        str(args.batch_size),
        "--temperature",
        str(args.temperature),
        "--top-p",
        str(args.top_p),
        "--max-new-tokens",
        str(args.max_new_tokens),
        "--seed",
        str(args.seed),
    ]
    if args.max_batches is not None:
        command.extend(("--max-batches", str(args.max_batches)))
    return command


def build_parser():
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
        description=(
            "Generate the 1,000 test responses from a saved GRPO checkpoint."
        ),
    )
    parser.add_argument(
        "--checkpoint",
        help=(
            "Checkpoint directory name under --checkpoint-root, or an "
            "absolute checkpoint path."
        ),
    )
    parser.add_argument(
        "--checkpoint-root",
        type=Path,
        default=DEFAULT_CHECKPOINT_ROOT,
    )
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--list-checkpoints", action="store_true")
    parser.add_argument(
        "--dataset",
        default="rasbt/human-writing-prompts-6k",
    )
    parser.add_argument(
        "--device",
        choices=("auto", "cuda", "mps", "cpu"),
        default="auto",
    )
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
    parser.add_argument("--max-batches", type=int, default=None)
    return parser


def parse_args():
    parser = build_parser()
    args = parser.parse_args()
    if not args.list_checkpoints and not args.checkpoint:
        parser.error("--checkpoint is required unless --list-checkpoints is used")
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
    checkpoint_root = args.checkpoint_root.expanduser().resolve()
    if args.list_checkpoints:
        if not checkpoint_root.is_dir():
            raise SystemExit(f"Checkpoint root does not exist: {checkpoint_root}")
        candidates = sorted(
            path for path in checkpoint_root.iterdir() if path.is_dir()
        )
        if not candidates:
            raise SystemExit(f"No checkpoint directories found under {checkpoint_root}")
        for checkpoint in candidates:
            problem = checkpoint_problem(checkpoint)
            status = "ready" if problem is None else f"invalid: {problem}"
            print(f"{checkpoint.name}\t{status}")
        return

    checkpoint = resolve_checkpoint(args.checkpoint, checkpoint_root)
    output_dir = (
        args.output_dir.expanduser().resolve()
        if args.output_dir is not None
        else default_output_dir(checkpoint)
    )
    print(f"Checkpoint: {checkpoint}")
    print(f"Output: {output_dir}")
    command = build_generation_command(args, checkpoint, output_dir)
    os.execv(sys.executable, command)


if __name__ == "__main__":
    main()
