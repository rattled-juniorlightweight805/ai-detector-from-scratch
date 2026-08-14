#!/usr/bin/env python3
"""Train Qwen3 with no-KL GRPO and an AI-text detector verifier.

Adapted from rlvr_grpo_original_no_kl_batched.py in
https://github.com/rasbt/reasoning-from-scratch.

The original MATH correctness verifier is replaced with a frozen local
AI-text classifier. A rollout receives a high reward when the verifier assigns
it a high human-written probability and its length is close to the requested
target length.
"""

import argparse
import csv
from collections import namedtuple
import json
import math
from pathlib import Path
import time

from datasets import load_dataset
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer


SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_DIR = SCRIPT_DIR.parents[1]
from ai_detector import load_classifier


SCRIPT_NAME = Path(__file__).stem
LOG_DIR = SCRIPT_DIR / "logs" / SCRIPT_NAME
CHECKPOINT_DIR = SCRIPT_DIR / "checkpoints" / SCRIPT_NAME
SAMPLE_LOG_PATH = LOG_DIR / "samples.txt"
METRICS_LOG_PATH = LOG_DIR / "metrics.csv"


Rollout = namedtuple("Rollout", "token_ids prompt_length text")


def resolve_device(requested):
    if requested == "auto":
        if torch.cuda.is_available():
            return torch.device("cuda")
        if torch.backends.mps.is_available():
            return torch.device("mps")
        return torch.device("cpu")
    if requested == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is not available")
    if requested == "mps" and not torch.backends.mps.is_available():
        raise RuntimeError("MPS was requested but is not available")
    if requested not in {"cpu", "cuda", "mps"}:
        raise ValueError("device must be one of: auto, cpu, cuda, mps")
    return torch.device(requested)


def render_prompt(example):
    target_words = int(example["target_words"])
    question = str(example["prompt"]).strip()
    return (
        "Write a standalone answer to the question below in approximately "
        f"{target_words} words. Return only the answer.\n\n"
        f"Question: {question}\n\nAnswer:"
    )


def response_token_limit(target_words, maximum):
    if target_words < 1:
        raise ValueError("target_words must be positive")
    if maximum < 1:
        raise ValueError("maximum must be positive")
    estimated_tokens = math.ceil(target_words * 1.6) + 16
    return min(maximum, max(64, estimated_tokens))


def length_adherence_score(word_count, target_words):
    """Return a smooth 0-1 score based on distance from the target length."""
    if target_words < 1:
        raise ValueError("target_words must be positive")
    if word_count <= 0:
        return 0.0
    return min(word_count, target_words) / max(word_count, target_words)


def compute_human_writing_rewards(
    texts,
    *,
    target_words,
    verifier,
    verifier_batch_size,
):
    if verifier_batch_size < 1:
        raise ValueError("verifier_batch_size must be at least 1")
    safe_texts = [text if text.strip() else "." for text in texts]
    ai_probabilities = verifier.score_many(
        safe_texts,
        batch_size=verifier_batch_size,
    )
    if len(ai_probabilities) != len(texts):
        raise ValueError("Verifier returned an unexpected number of scores")

    rewards = []
    details = []
    for text, ai_probability in zip(texts, ai_probabilities, strict=True):
        ai_probability = float(ai_probability)
        if not math.isfinite(ai_probability) or not 0.0 <= ai_probability <= 1.0:
            raise ValueError("Verifier probabilities must be between 0 and 1")
        word_count = len(text.split())
        human_probability = 1.0 - ai_probability
        length_score = length_adherence_score(word_count, target_words)
        reward = human_probability * length_score
        rewards.append(reward)
        details.append(
            {
                "ai_probability": ai_probability,
                "human_probability": human_probability,
                "length_score": length_score,
                "word_count": word_count,
            }
        )
    return rewards, details


@torch.no_grad()
def sample_responses_batched(
    model,
    tokenizer,
    prompt,
    device,
    batch_size,
    *,
    max_new_tokens,
    temperature,
    top_p,
):
    encoded = tokenizer(
        prompt,
        add_special_tokens=True,
        return_tensors="pt",
    )
    prompt_ids = encoded["input_ids"].to(device)
    prompt_mask = encoded["attention_mask"].to(device)
    prompt_length = int(prompt_ids.shape[1])
    input_ids = prompt_ids.expand(batch_size, -1).clone()
    attention_mask = prompt_mask.expand(batch_size, -1).clone()

    generated = model.generate(
        input_ids=input_ids,
        attention_mask=attention_mask,
        do_sample=True,
        max_new_tokens=max_new_tokens,
        temperature=temperature,
        top_p=top_p,
        eos_token_id=tokenizer.eos_token_id,
        pad_token_id=tokenizer.pad_token_id,
        use_cache=True,
    )

    results = []
    for row in generated:
        new_tokens = row[prompt_length:]
        eos_positions = (new_tokens == tokenizer.eos_token_id).nonzero(
            as_tuple=True
        )[0]
        if len(eos_positions) > 0:
            new_tokens = new_tokens[: int(eos_positions[0]) + 1]
        full_tokens = torch.cat((row[:prompt_length], new_tokens), dim=0)
        text = tokenizer.decode(new_tokens, skip_special_tokens=True).strip()
        results.append(
            Rollout(
                token_ids=full_tokens.detach(),
                prompt_length=prompt_length,
                text=text,
            )
        )
    return results


def sequence_logprob(model, rollout):
    token_ids = rollout.token_ids
    outputs = model(
        input_ids=token_ids.unsqueeze(0),
        attention_mask=torch.ones_like(token_ids).unsqueeze(0),
        use_cache=False,
    )
    logits = outputs.logits.squeeze(0).float()
    targets = token_ids[1:]
    token_logprobs = torch.log_softmax(logits[:-1], dim=-1)
    selected = token_logprobs.gather(
        1, targets.unsqueeze(-1)
    ).squeeze(-1)
    return selected[rollout.prompt_length - 1 :].sum()


def normalized_advantages(rewards, device):
    if len(rewards) < 2:
        raise ValueError("GRPO requires at least two rollouts per prompt")
    reward_tensor = torch.tensor(rewards, dtype=torch.float32, device=device)
    return (reward_tensor - reward_tensor.mean()) / (
        reward_tensor.std(unbiased=False) + 1e-4
    )


def compute_grpo_loss(
    model,
    tokenizer,
    verifier,
    example,
    device,
    *,
    num_rollouts,
    rollout_batch_size,
    verifier_batch_size,
    max_new_tokens,
    temperature,
    top_p,
    skip_zero_advantage,
):
    prompt = render_prompt(example)
    target_words = int(example["target_words"])
    token_limit = response_token_limit(target_words, max_new_tokens)

    was_training = model.training
    model.eval()
    rollouts = []
    remaining = num_rollouts
    while remaining > 0:
        current_batch_size = min(rollout_batch_size, remaining)
        rollouts.extend(
            sample_responses_batched(
                model,
                tokenizer,
                prompt,
                device,
                current_batch_size,
                max_new_tokens=token_limit,
                temperature=temperature,
                top_p=top_p,
            )
        )
        remaining -= current_batch_size
    if was_training:
        model.train()

    rewards, reward_details = compute_human_writing_rewards(
        [rollout.text for rollout in rollouts],
        target_words=target_words,
        verifier=verifier,
        verifier_batch_size=verifier_batch_size,
    )
    advantages = normalized_advantages(rewards, device)
    is_zero_advantage = torch.allclose(
        advantages,
        torch.zeros_like(advantages),
        atol=1e-8,
        rtol=0.0,
    )

    samples = []
    for rollout, reward, details in zip(
        rollouts, rewards, reward_details, strict=True
    ):
        samples.append(
            {
                "text": rollout.text,
                "reward": reward,
                "gen_len": int(rollout.token_ids.numel() - rollout.prompt_length),
                **details,
            }
        )

    if skip_zero_advantage and is_zero_advantage:
        return {
            "loss": 0.0,
            "loss_tensor": None,
            "rewards": rewards,
            "advantages": advantages.detach().cpu().tolist(),
            "is_zero_advantage": True,
            "samples": samples,
        }

    logprobs = torch.stack(
        [sequence_logprob(model, rollout) for rollout in rollouts]
    )
    loss = -(advantages.detach() * logprobs).mean()
    return {
        "loss": float(loss.detach().cpu()),
        "loss_tensor": loss,
        "rewards": rewards,
        "advantages": advantages.detach().cpu().tolist(),
        "is_zero_advantage": is_zero_advantage,
        "samples": samples,
    }


def append_sample_logs(step, example, samples):
    SAMPLE_LOG_PATH.parent.mkdir(parents=True, exist_ok=True)
    with SAMPLE_LOG_PATH.open("a", encoding="utf-8") as file:
        file.write(
            f"[Step {step}] prompt_id={example['prompt_id']} "
            f"target_words={example['target_words']}\n"
        )
        file.write(f"Question: {example['prompt']}\n")
        for index, sample in enumerate(samples[:3], start=1):
            text = sample["text"].replace("\n", "\\n")
            file.write(
                f"  {index}) reward={sample['reward']:.4f} "
                f"human={sample['human_probability']:.4f} "
                f"length={sample['length_score']:.4f} "
                f"words={sample['word_count']}: {text}\n"
            )
        file.write("\n")


def append_metrics(step, total_steps, stats, elapsed):
    METRICS_LOG_PATH.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "step",
        "total_steps",
        "loss",
        "reward_mean",
        "human_probability_mean",
        "length_score_mean",
        "response_words_mean",
        "tokens_per_second",
        "zero_advantage",
    ]
    samples = stats["samples"]
    generated_tokens = sum(sample["gen_len"] for sample in samples)
    row = {
        "step": step,
        "total_steps": total_steps,
        "loss": stats["loss"],
        "reward_mean": sum(stats["rewards"]) / len(stats["rewards"]),
        "human_probability_mean": sum(
            sample["human_probability"] for sample in samples
        )
        / len(samples),
        "length_score_mean": sum(sample["length_score"] for sample in samples)
        / len(samples),
        "response_words_mean": sum(sample["word_count"] for sample in samples)
        / len(samples),
        "tokens_per_second": generated_tokens / elapsed if elapsed > 0 else 0.0,
        "zero_advantage": stats["is_zero_advantage"],
    }
    write_header = not METRICS_LOG_PATH.exists()
    with METRICS_LOG_PATH.open("a", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        if write_header:
            writer.writeheader()
        writer.writerow(row)


def save_checkpoint(model, tokenizer, step, suffix = ""):
    suffix = f"-{suffix}" if suffix else ""
    path = CHECKPOINT_DIR / f"step-{step:05d}{suffix}"
    path.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(path, safe_serialization=True)
    tokenizer.save_pretrained(path)
    return path


def train(
    model,
    tokenizer,
    verifier,
    train_data,
    device,
    args,
):
    total_steps = len(train_data) if args.steps is None else args.steps
    if total_steps < 1:
        raise ValueError("steps must be positive")
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate)
    model.train()
    current_step = 0
    start_time = time.perf_counter()

    try:
        for step_index in range(total_steps):
            step_start = time.perf_counter()
            current_step = step_index + 1
            example = train_data[step_index % len(train_data)]
            stats = compute_grpo_loss(
                model,
                tokenizer,
                verifier,
                example,
                device,
                num_rollouts=args.num_rollouts,
                rollout_batch_size=args.rollout_batch_size,
                verifier_batch_size=args.verifier_batch_size,
                max_new_tokens=args.max_new_tokens,
                temperature=args.temperature,
                top_p=args.top_p,
                skip_zero_advantage=args.skip_zero_advantage_updates,
            )
            if stats["loss_tensor"] is not None:
                optimizer.zero_grad(set_to_none=True)
                stats["loss_tensor"].backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()

            elapsed = time.perf_counter() - step_start
            append_metrics(current_step, total_steps, stats, elapsed)
            if current_step == 1 or current_step % args.log_samples_every == 0:
                append_sample_logs(current_step, example, stats["samples"])
            if args.checkpoint_every and current_step % args.checkpoint_every == 0:
                path = save_checkpoint(model, tokenizer, current_step)
                print(f"Saved checkpoint: {path}")

            reward_mean = sum(stats["rewards"]) / len(stats["rewards"])
            human_mean = sum(
                sample["human_probability"] for sample in stats["samples"]
            ) / len(stats["samples"])
            length_mean = sum(
                sample["length_score"] for sample in stats["samples"]
            ) / len(stats["samples"])
            completed = current_step / total_steps
            eta_seconds = (
                (time.perf_counter() - start_time) / completed
                - (time.perf_counter() - start_time)
            )
            print(
                f"[Step {current_step}/{total_steps}] "
                f"loss={stats['loss']:.4f} reward={reward_mean:.4f} "
                f"human={human_mean:.4f} length={length_mean:.4f} "
                f"step_time={elapsed:.1f}s eta={eta_seconds / 3600:.1f}h"
            )
    except KeyboardInterrupt:
        path = save_checkpoint(model, tokenizer, max(1, current_step), "interrupt")
        print(f"\nInterrupted. Saved checkpoint: {path}")
        return

    path = save_checkpoint(model, tokenizer, total_steps, "final")
    print(f"Saved final model: {path}")


def load_policy(args, device):
    source = args.checkpoint_path or args.policy_model
    dtype = torch.bfloat16 if device.type == "cuda" else torch.float32
    tokenizer = AutoTokenizer.from_pretrained(source)
    if tokenizer.eos_token_id is None:
        raise ValueError("Policy tokenizer must define an EOS token")
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(source, dtype=dtype)
    model.to(device)
    if args.gradient_checkpointing:
        model.gradient_checkpointing_enable()
    model.config.use_cache = False
    return model, tokenizer


def parse_args():
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
        description=(
            "Train Qwen3 with no-KL GRPO on human-writing prompts using "
            "the local AI detector as verifier."
        ),
    )
    parser.add_argument(
        "--dataset",
        default="rasbt/human-writing-prompts-6k",
        help="Hugging Face prompt dataset.",
    )
    parser.add_argument(
        "--policy-model",
        default="Qwen/Qwen3-0.6B-Base",
        help="Initial causal language model.",
    )
    parser.add_argument(
        "--checkpoint-path",
        type=Path,
        default=None,
        help="Optional saved policy directory to continue from.",
    )
    parser.add_argument(
        "--verifier-model",
        default="qwen3-variable",
        help="Classifier API model used for reward scoring.",
    )
    parser.add_argument(
        "--policy-device",
        choices=("auto", "cuda", "mps", "cpu"),
        default="auto",
    )
    parser.add_argument(
        "--verifier-device",
        choices=("auto", "cuda", "mps", "cpu"),
        default="auto",
    )
    parser.add_argument("--steps", type=int, default=None)
    parser.add_argument("--num-rollouts", type=int, default=8)
    parser.add_argument("--rollout-batch-size", type=int, default=8)
    parser.add_argument("--verifier-batch-size", type=int, default=4)
    parser.add_argument("--max-new-tokens", type=int, default=1616)
    parser.add_argument("--temperature", type=float, default=0.8)
    parser.add_argument("--top-p", type=float, default=0.9)
    parser.add_argument("--learning-rate", type=float, default=1e-5)
    parser.add_argument("--checkpoint-every", type=int, default=50)
    parser.add_argument("--log-samples-every", type=int, default=10)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--skip-zero-advantage-updates",
        action="store_true",
    )
    parser.add_argument(
        "--gradient-checkpointing",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    args = parser.parse_args()
    if args.num_rollouts < 2:
        parser.error("--num-rollouts must be at least 2")
    if args.rollout_batch_size < 1:
        parser.error("--rollout-batch-size must be at least 1")
    if args.verifier_batch_size < 1:
        parser.error("--verifier-batch-size must be at least 1")
    if args.temperature <= 0:
        parser.error("--temperature must be positive")
    if not 0 < args.top_p <= 1:
        parser.error("--top-p must be in (0, 1]")
    return args


def main():
    args = parse_args()
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    policy_device = resolve_device(args.policy_device)
    print(f"Policy device: {policy_device}")
    print(f"Verifier device: {args.verifier_device}")
    print(f"Dataset: {args.dataset} [train]")
    train_data = load_dataset(args.dataset, split="train").shuffle(seed=args.seed)
    if not train_data:
        raise ValueError("Training split is empty")

    verifier = load_classifier(
        args.verifier_model,
        device=args.verifier_device,
    )
    model, tokenizer = load_policy(args, policy_device)
    run_config = vars(args).copy()
    run_config["checkpoint_path"] = (
        str(args.checkpoint_path) if args.checkpoint_path else None
    )
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    (LOG_DIR / "run-config.json").write_text(
        json.dumps(run_config, indent=2) + "\n",
        encoding="utf-8",
    )

    train(model, tokenizer, verifier, train_data, policy_device, args)
    if torch.cuda.is_available():
        memory_gb = torch.cuda.max_memory_allocated() / 1024**3
        print(f"Max CUDA memory allocated: {memory_gb:.2f} GB")


if __name__ == "__main__":
    main()
