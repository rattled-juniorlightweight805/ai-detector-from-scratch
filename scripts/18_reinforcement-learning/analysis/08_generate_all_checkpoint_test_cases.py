#!/usr/bin/env python3
"""Generate test responses at regular GRPO checkpoint intervals."""

import argparse
import importlib.util
from pathlib import Path
import re
import shlex
import subprocess
import sys


SCRIPT_DIR = Path(__file__).resolve().parent
CHECKPOINT_SCRIPT = SCRIPT_DIR / "07_generate_checkpoint_test_cases.py"
DEFAULT_CHECKPOINT_ROOT = (
    SCRIPT_DIR.parent / "checkpoints" / "05_train_grpo_human_writing"
)


def load_checkpoint_helpers():
    spec = importlib.util.spec_from_file_location(
        "checkpoint_test_case_generation",
        CHECKPOINT_SCRIPT,
    )
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Could not load checkpoint script: {CHECKPOINT_SCRIPT}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module.available_checkpoints


available_checkpoints = load_checkpoint_helpers()


def checkpoint_step(checkpoint):
    match = re.fullmatch(r"step-(\d+)", checkpoint.name)
    return int(match.group(1)) if match else None


def select_checkpoints(
    checkpoints, checkpoint_every
):
    if checkpoint_every < 1:
        raise ValueError("checkpoint_every must be at least 1")
    selected = []
    for checkpoint in checkpoints:
        step = checkpoint_step(checkpoint)
        if step is not None and step > 0 and step % checkpoint_every == 0:
            selected.append(checkpoint)
    return selected


def output_dir_for_checkpoint(output_root, checkpoint):
    return output_root / f"test-cases-{checkpoint.name}"


def build_generation_command(
    args,
    checkpoint,
    output_dir,
):
    command = [
        sys.executable,
        "-u",
        str(CHECKPOINT_SCRIPT),
        "--checkpoint",
        str(checkpoint),
        "--output-dir",
        str(output_dir),
        "--dataset",
        args.dataset,
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


def parse_args():
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
        description=(
            "Generate 1,000 test responses at regular GRPO checkpoint "
            "intervals. "
            "Completed checkpoint runs are verified and skipped by the child "
            "generator."
        ),
    )
    parser.add_argument(
        "--checkpoint-root",
        type=Path,
        default=DEFAULT_CHECKPOINT_ROOT,
    )
    parser.add_argument("--output-root", type=Path, default=SCRIPT_DIR)
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
    parser.add_argument(
        "--checkpoint-every",
        type=int,
        default=500,
        help="Evaluate exact step-N checkpoints divisible by this interval.",
    )
    parser.add_argument(
        "--max-batches",
        type=int,
        default=None,
        help="Generate at most this many batches per checkpoint.",
    )
    parser.add_argument(
        "--max-checkpoints",
        type=int,
        default=None,
        help="Process only the first N valid checkpoints.",
    )
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument(
        "--continue-on-error",
        action="store_true",
        help="Continue to later checkpoints after a child process fails.",
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
    if args.checkpoint_every < 1:
        parser.error("--checkpoint-every must be at least 1")
    if args.max_batches is not None and args.max_batches < 1:
        parser.error("--max-batches must be at least 1")
    if args.max_checkpoints is not None and args.max_checkpoints < 1:
        parser.error("--max-checkpoints must be at least 1")
    return args


def main():
    args = parse_args()
    checkpoint_root = args.checkpoint_root.expanduser().resolve()
    output_root = args.output_root.expanduser().resolve()
    checkpoints = select_checkpoints(
        available_checkpoints(checkpoint_root), args.checkpoint_every
    )
    if not checkpoints:
        raise SystemExit(
            f"No valid checkpoints at {args.checkpoint_every}-step intervals "
            f"found under {checkpoint_root}"
        )
    if args.max_checkpoints is not None:
        checkpoints = checkpoints[: args.max_checkpoints]

    print(f"Checkpoint root: {checkpoint_root}")
    print(f"Output root: {output_root}")
    print(f"Checkpoint interval: {args.checkpoint_every:,}")
    print(f"Checkpoints to process: {len(checkpoints):,}")
    failures = []
    for index, checkpoint in enumerate(checkpoints, start=1):
        output_dir = output_dir_for_checkpoint(output_root, checkpoint)
        command = build_generation_command(args, checkpoint, output_dir)
        print(f"\n[{index}/{len(checkpoints)}] {checkpoint.name}", flush=True)
        if args.dry_run:
            print(shlex.join(command))
            continue

        result = subprocess.run(command, check=False)
        if result.returncode == 0:
            continue
        failures.append((checkpoint, result.returncode))
        print(
            f"Checkpoint failed with exit code {result.returncode}: {checkpoint}",
            file=sys.stderr,
        )
        if not args.continue_on_error:
            raise SystemExit(result.returncode)

    if failures:
        print("\nFailed checkpoints:", file=sys.stderr)
        for checkpoint, returncode in failures:
            print(f"  {checkpoint.name}: exit {returncode}", file=sys.stderr)
        raise SystemExit(1)
    if not args.dry_run:
        print(f"\nCompleted {len(checkpoints):,} checkpoint runs.")


if __name__ == "__main__":
    main()
