#!/usr/bin/env python3
"""Benchmark GPT-5.6 Luna across repeated AI-text classification passes."""

import argparse
import hashlib
import http.client
import json
import os
import random
import statistics
import time
import urllib.error
import urllib.request
from collections import namedtuple
from concurrent.futures import FIRST_COMPLETED, ThreadPoolExecutor, wait
from pathlib import Path

import numpy as np
from datasets import load_from_disk


SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_DIR = SCRIPT_DIR.parents[1]
DEFAULT_PROMPT_PATH = SCRIPT_DIR / "PROMPT.md"
DEFAULT_DATASET_PATH = PROJECT_DIR / "data" / "hf-dataset"
DEFAULT_OUTPUT_DIR = SCRIPT_DIR / "results"
DEFAULT_API_BASE = "https://api.openai.com/v1"
DEFAULT_MODEL = "gpt-5.6-luna"
DEFAULT_REASONING_EFFORT = "medium"
DEFAULT_REPEATS = 5
DEFAULT_WORKERS = 8
DEFAULT_THRESHOLD = 50


PromptConfig = namedtuple(
    "PromptConfig", "system_prompt user_template output_schema sha256"
)
TestSample = namedtuple("TestSample", "dataset_index sample_id text label")


class APIError(RuntimeError):
    def __init__(self, message, *, retryable):
        super().__init__(message)
        self.retryable = retryable


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Classify the local Hugging Face test split with GPT-5.6 Luna. "
            "The default benchmark performs five complete passes."
        )
    )
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument(
        "--reasoning-effort",
        choices=("low", "medium", "high"),
        default=DEFAULT_REASONING_EFFORT,
    )
    parser.add_argument(
        "--prompt-path",
        type=Path,
        default=DEFAULT_PROMPT_PATH,
    )
    parser.add_argument(
        "--dataset-path",
        type=Path,
        default=DEFAULT_DATASET_PATH,
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
    )
    parser.add_argument("--repeats", type=int, default=DEFAULT_REPEATS)
    parser.add_argument("--workers", type=int, default=DEFAULT_WORKERS)
    parser.add_argument(
        "--limit",
        type=int,
        help="Use a deterministic, class-balanced subset for a small test run.",
    )
    parser.add_argument("--seed", type=int, default=17)
    parser.add_argument("--threshold", type=int, default=DEFAULT_THRESHOLD)
    parser.add_argument("--timeout", type=float, default=180.0)
    parser.add_argument("--max-retries", type=int, default=5)
    parser.add_argument("--max-output-tokens", type=int, default=2_048)
    parser.add_argument("--warmup-samples", type=int, default=1)
    parser.add_argument("--progress-every", type=int, default=100)
    parser.add_argument("--api-base", default=DEFAULT_API_BASE)
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Rerun passes even when matching completed result files exist.",
    )
    args = parser.parse_args()

    if args.repeats < 1:
        parser.error("--repeats must be at least 1")
    if args.workers < 1:
        parser.error("--workers must be at least 1")
    if args.limit is not None and args.limit < 1:
        parser.error("--limit must be at least 1")
    if not 0 <= args.threshold <= 100:
        parser.error("--threshold must be between 0 and 100")
    if args.timeout <= 0:
        parser.error("--timeout must be positive")
    if args.max_retries < 1:
        parser.error("--max-retries must be at least 1")
    if args.max_output_tokens < 1:
        parser.error("--max-output-tokens must be at least 1")
    if args.warmup_samples < 0:
        parser.error("--warmup-samples cannot be negative")
    if args.progress_every < 1:
        parser.error("--progress-every must be at least 1")
    return args


def fenced_block(markdown, heading, language):
    marker = f"## {heading}\n"
    heading_start = markdown.find(marker)
    if heading_start < 0:
        raise ValueError(f"Prompt document is missing heading: {heading}")
    fence = f"```{language}\n"
    content_start = markdown.find(fence, heading_start + len(marker))
    if content_start < 0:
        raise ValueError(
            f"Prompt document is missing a {language} block under {heading}"
        )
    content_start += len(fence)
    content_end = markdown.find("\n```", content_start)
    if content_end < 0:
        raise ValueError(f"Prompt document has an unclosed block under {heading}")
    return markdown[content_start:content_end].strip()


def load_prompt_config(path):
    markdown = path.read_text(encoding="utf-8")
    system_prompt = fenced_block(markdown, "System prompt", "text")
    user_template = fenced_block(markdown, "User prompt", "text")
    schema_text = fenced_block(markdown, "Structured output schema", "json")
    output_schema = json.loads(schema_text)
    if user_template.count("{{TEXT}}") != 1:
        raise ValueError("The user prompt must contain {{TEXT}} exactly once")
    canonical = json.dumps(
        {
            "system_prompt": system_prompt,
            "user_template": user_template,
            "output_schema": output_schema,
        },
        ensure_ascii=False,
        sort_keys=True,
    )
    return PromptConfig(
        system_prompt=system_prompt,
        user_template=user_template,
        output_schema=output_schema,
        sha256=hashlib.sha256(canonical.encode("utf-8")).hexdigest(),
    )


def load_test_samples(
    dataset_path,
    *,
    limit,
    seed,
):
    dataset = load_from_disk(str(dataset_path))
    test_dataset = dataset["test"]
    all_indices = list(range(len(test_dataset)))

    if limit is not None and limit < len(all_indices):
        by_label = {
            0: [i for i, label in enumerate(test_dataset["label"]) if label == 0],
            1: [i for i, label in enumerate(test_dataset["label"]) if label == 1],
        }
        random_generator = random.Random(seed)
        random_generator.shuffle(by_label[0])
        random_generator.shuffle(by_label[1])
        human_count = (limit + 1) // 2
        ai_count = limit - human_count
        if human_count > len(by_label[0]) or ai_count > len(by_label[1]):
            raise ValueError("The requested class-balanced subset is too large")
        all_indices = by_label[0][:human_count] + by_label[1][:ai_count]
        random_generator.shuffle(all_indices)

    samples = []
    for dataset_index in all_indices:
        row = test_dataset[dataset_index]
        samples.append(
            TestSample(
                dataset_index=dataset_index,
                sample_id=str(row["id"]),
                text=str(row["text"]),
                label=int(row["label"]),
            )
        )
    return samples


def build_payload(
    prompt,
    text,
    *,
    model,
    reasoning_effort,
    max_output_tokens,
):
    return {
        "model": model,
        "instructions": prompt.system_prompt,
        "input": prompt.user_template.replace("{{TEXT}}", text),
        "reasoning": {"effort": reasoning_effort},
        "max_output_tokens": max_output_tokens,
        "text": {
            "format": {
                "type": "json_schema",
                "name": "ai_text_classification",
                "strict": True,
                "schema": prompt.output_schema,
            }
        },
        "store": False,
    }


def http_json(
    url,
    payload,
    api_key,
    timeout,
):
    request = urllib.request.Request(
        url,
        data=json.dumps(payload).encode("utf-8"),
        method="POST",
        headers={
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
            "User-Agent": "ai-detector-dev/chatgpt-baseline",
        },
    )
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            result = json.loads(response.read().decode("utf-8"))
    except urllib.error.HTTPError as error:
        details = error.read().decode("utf-8", errors="replace")
        retryable = error.code in {408, 409, 429} or error.code >= 500
        raise APIError(
            f"OpenAI returned HTTP {error.code}: {details}",
            retryable=retryable,
        ) from error
    except urllib.error.URLError as error:
        raise APIError(
            f"Could not connect to OpenAI: {error.reason}", retryable=True
        ) from error
    except TimeoutError as error:
        raise APIError("OpenAI request timed out", retryable=True) from error
    except http.client.IncompleteRead as error:
        raise APIError(
            f"OpenAI returned an incomplete response: {error}", retryable=True
        ) from error
    except json.JSONDecodeError as error:
        raise APIError(
            f"OpenAI returned invalid JSON: {error}", retryable=True
        ) from error
    if not isinstance(result, dict):
        raise APIError("OpenAI returned a non-object response", retryable=True)
    return result


def output_text(result):
    if result.get("status") != "completed":
        raise APIError(
            "OpenAI response did not complete: "
            f"{result.get('incomplete_details')!r}",
            retryable=True,
        )
    parts = []
    for item in result.get("output", []):
        if not isinstance(item, dict) or item.get("type") != "message":
            continue
        for part in item.get("content", []):
            if not isinstance(part, dict):
                continue
            if part.get("type") == "refusal":
                raise APIError(
                    f"OpenAI refused the request: {part.get('refusal')}",
                    retryable=False,
                )
            if part.get("type") == "output_text":
                parts.append(str(part.get("text", "")))
    text = "".join(parts).strip()
    if not text:
        raise APIError("OpenAI returned no output text", retryable=True)
    return text


def parse_ai_score(result):
    try:
        payload = json.loads(output_text(result))
    except json.JSONDecodeError as error:
        raise APIError(
            f"OpenAI returned malformed structured output: {error}",
            retryable=True,
        ) from error
    if not isinstance(payload, dict):
        raise APIError("Structured output is not an object", retryable=True)
    score = payload.get("ai_score")
    if isinstance(score, bool) or not isinstance(score, int):
        raise APIError("Structured output has a non-integer ai_score", retryable=True)
    if not 0 <= score <= 100:
        raise APIError("Structured output ai_score is outside 0 to 100", retryable=True)
    return score


def classify_sample(
    sample,
    *,
    pass_index,
    prompt,
    model,
    reasoning_effort,
    max_output_tokens,
    api_base,
    api_key,
    timeout,
    max_retries,
    threshold,
    seed,
):
    payload = build_payload(
        prompt,
        sample.text,
        model=model,
        reasoning_effort=reasoning_effort,
        max_output_tokens=max_output_tokens,
    )
    started = time.perf_counter()
    last_error = None
    for attempt in range(1, max_retries + 1):
        try:
            result = http_json(
                f"{api_base.rstrip('/')}/responses",
                payload,
                api_key,
                timeout,
            )
            score = parse_ai_score(result)
            prediction = int(score >= threshold)
            return {
                "dataset_index": sample.dataset_index,
                "sample_id": sample.sample_id,
                "true_label": sample.label,
                "ai_score": score,
                "prediction": prediction,
                "correct": prediction == sample.label,
                "latency_seconds": time.perf_counter() - started,
                "attempts": attempt,
                "response_id": result.get("id"),
                "response_model": result.get("model"),
                "usage": result.get("usage") or {},
            }
        except APIError as error:
            last_error = error
            if not error.retryable or attempt == max_retries:
                break
            jitter = random.Random(
                seed
                + pass_index * 1_000_003
                + sample.dataset_index * 101
                + attempt
            ).uniform(0.0, 1.0)
            time.sleep(min(2**attempt, 30) + jitter)
    assert last_error is not None
    raise last_error


def run_pass(
    samples,
    classify,
    *,
    workers,
    progress_every,
):
    worker_count = min(workers, len(samples))
    iterator = iter(samples)
    in_flight = {}
    results = []
    failures = []
    completed = 0
    executor = ThreadPoolExecutor(
        max_workers=worker_count,
        thread_name_prefix="openai-benchmark",
    )

    def submit_available():
        while len(in_flight) < worker_count:
            try:
                sample = next(iterator)
            except StopIteration:
                return
            in_flight[executor.submit(classify, sample)] = sample

    started = time.perf_counter()
    try:
        submit_available()
        while in_flight:
            done, _ = wait(in_flight, return_when=FIRST_COMPLETED)
            for future in done:
                sample = in_flight.pop(future)
                try:
                    results.append(future.result())
                except Exception as error:  # noqa: BLE001
                    failures.append(
                        {
                            "dataset_index": sample.dataset_index,
                            "sample_id": sample.sample_id,
                            "error": str(error),
                        }
                    )
                completed += 1
                if completed % progress_every == 0 or completed == len(samples):
                    print(
                        f"Completed {completed:,}/{len(samples):,} "
                        f"with {len(failures)} failures",
                        flush=True,
                    )
            submit_available()
    finally:
        executor.shutdown(wait=True, cancel_futures=True)
    elapsed = time.perf_counter() - started
    results.sort(key=lambda row: int(row["dataset_index"]))
    failures.sort(key=lambda row: int(row["dataset_index"]))
    return results, failures, elapsed


def confusion_counts(
    true_labels,
    predictions,
):
    return {
        "true_negative": int(np.sum((true_labels == 0) & (predictions == 0))),
        "false_positive": int(np.sum((true_labels == 0) & (predictions == 1))),
        "false_negative": int(np.sum((true_labels == 1) & (predictions == 0))),
        "true_positive": int(np.sum((true_labels == 1) & (predictions == 1))),
    }


def sum_usage(rows):
    totals = {
        "input_tokens": 0,
        "cached_input_tokens": 0,
        "output_tokens": 0,
        "reasoning_tokens": 0,
        "total_tokens": 0,
    }
    for row in rows:
        usage = row.get("usage") or {}
        input_details = usage.get("input_tokens_details") or {}
        output_details = usage.get("output_tokens_details") or {}
        totals["input_tokens"] += int(usage.get("input_tokens") or 0)
        totals["cached_input_tokens"] += int(
            input_details.get("cached_tokens") or 0
        )
        totals["output_tokens"] += int(usage.get("output_tokens") or 0)
        totals["reasoning_tokens"] += int(
            output_details.get("reasoning_tokens") or 0
        )
        totals["total_tokens"] += int(usage.get("total_tokens") or 0)
    return totals


def summarize_pass(
    rows,
    duration_seconds,
):
    true_labels = np.asarray([row["true_label"] for row in rows], dtype=np.int64)
    predictions = np.asarray([row["prediction"] for row in rows], dtype=np.int64)
    latencies = [float(row["latency_seconds"]) for row in rows]
    retries = sum(max(int(row["attempts"]) - 1, 0) for row in rows)
    return {
        "samples": len(rows),
        "duration_seconds": duration_seconds,
        "accuracy": float(np.mean(predictions == true_labels)),
        "confusion": confusion_counts(true_labels, predictions),
        "throughput_texts_per_second": len(rows) / duration_seconds,
        "mean_request_latency_seconds": statistics.fmean(latencies),
        "request_latency_sample_std_seconds": (
            statistics.stdev(latencies) if len(latencies) > 1 else 0.0
        ),
        "retries": retries,
        "usage": sum_usage(rows),
    }


def benchmark_config(
    args,
    prompt,
    samples,
):
    return {
        "model": args.model,
        "reasoning_effort": args.reasoning_effort,
        "prompt_sha256": prompt.sha256,
        "threshold": args.threshold,
        "max_output_tokens": args.max_output_tokens,
        "workers": args.workers,
        "seed": args.seed,
        "sample_ids_sha256": hashlib.sha256(
            "\n".join(sample.sample_id for sample in samples).encode("utf-8")
        ).hexdigest(),
        "samples_per_pass": len(samples),
    }


def atomic_write_json(path, payload):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, ensure_ascii=False, indent=2)
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)


def read_completed_pass(
    path,
    expected_config,
):
    payload = json.loads(path.read_text(encoding="utf-8"))
    if payload.get("config") != expected_config:
        raise SystemExit(
            f"Existing result does not match this benchmark configuration: {path}"
        )
    rows = payload.get("results")
    if not isinstance(rows, list) or len(rows) != expected_config["samples_per_pass"]:
        raise SystemExit(f"Existing result is incomplete: {path}")
    return payload


def overall_summary(
    pass_payloads,
    config,
):
    pass_summaries = [payload["summary"] for payload in pass_payloads]
    durations = [float(summary["duration_seconds"]) for summary in pass_summaries]
    accuracies = [float(summary["accuracy"]) for summary in pass_summaries]

    ordered_ids = [row["sample_id"] for row in pass_payloads[0]["results"]]
    rows_by_pass = [
        {row["sample_id"]: row for row in payload["results"]}
        for payload in pass_payloads
    ]
    unanimous = 0
    score_stdevs = []
    majority_true = []
    majority_predicted = []
    for sample_id in ordered_ids:
        rows = [mapping[sample_id] for mapping in rows_by_pass]
        scores = [int(row["ai_score"]) for row in rows]
        predictions = [int(row["prediction"]) for row in rows]
        if len(set(predictions)) == 1:
            unanimous += 1
        score_stdevs.append(
            statistics.stdev(scores) if len(scores) > 1 else 0.0
        )
        votes_for_ai = sum(predictions)
        if votes_for_ai * 2 == len(predictions):
            majority_prediction = int(
                statistics.fmean(scores) >= int(config["threshold"])
            )
        else:
            majority_prediction = int(votes_for_ai * 2 > len(predictions))
        majority_true.append(int(rows[0]["true_label"]))
        majority_predicted.append(majority_prediction)

    majority_true_array = np.asarray(majority_true, dtype=np.int64)
    majority_prediction_array = np.asarray(majority_predicted, dtype=np.int64)
    total_usage = {
        key: sum(int(summary["usage"].get(key, 0)) for summary in pass_summaries)
        for key in pass_summaries[0]["usage"]
    }
    return {
        "config": config,
        "passes": len(pass_payloads),
        "total_classifications": len(ordered_ids) * len(pass_payloads),
        "mean_pass_duration_seconds": statistics.fmean(durations),
        "pass_duration_sample_std_seconds": (
            statistics.stdev(durations) if len(durations) > 1 else 0.0
        ),
        "mean_accuracy": statistics.fmean(accuracies),
        "accuracy_sample_std": (
            statistics.stdev(accuracies) if len(accuracies) > 1 else 0.0
        ),
        "unanimous_prediction_rate": unanimous / len(ordered_ids),
        "mean_per_sample_score_std": statistics.fmean(score_stdevs),
        "majority_vote_accuracy": float(
            np.mean(majority_prediction_array == majority_true_array)
        ),
        "majority_vote_confusion": confusion_counts(
            majority_true_array,
            majority_prediction_array,
        ),
        "total_usage": total_usage,
        "pass_summaries": pass_summaries,
    }


def print_overall_summary(summary):
    print(f"\n{summary['passes']}-pass benchmark summary")
    print(f"Classifications: {summary['total_classifications']:,}")
    print(
        "Mean full-pass time: "
        f"{summary['mean_pass_duration_seconds']:.2f} +/- "
        f"{summary['pass_duration_sample_std_seconds']:.2f} seconds"
    )
    print(
        f"Mean accuracy: {summary['mean_accuracy']:.2%} +/- "
        f"{summary['accuracy_sample_std']:.2%}"
    )
    print(
        "Unanimous predictions across passes: "
        f"{summary['unanimous_prediction_rate']:.2%}"
    )
    print(
        "Mean per-sample score standard deviation: "
        f"{summary['mean_per_sample_score_std']:.2f} points"
    )
    print(f"Majority-vote accuracy: {summary['majority_vote_accuracy']:.2%}")
    print(f"Total token usage: {summary['total_usage']}")


def main():
    args = parse_args()
    api_key = os.environ.get("OPENAI_API_KEY")
    if not api_key:
        raise SystemExit("Set OPENAI_API_KEY before running this script")
    if not args.prompt_path.is_file():
        raise SystemExit(f"Prompt file not found: {args.prompt_path}")
    if not args.dataset_path.is_dir():
        raise SystemExit(f"Dataset not found: {args.dataset_path}")

    prompt = load_prompt_config(args.prompt_path)
    samples = load_test_samples(
        args.dataset_path,
        limit=args.limit,
        seed=args.seed,
    )
    config = benchmark_config(args, prompt, samples)
    args.output_dir.mkdir(parents=True, exist_ok=True)

    print(f"Model: {args.model}")
    print(f"Reasoning effort: {args.reasoning_effort}")
    print(f"Test samples per pass: {len(samples):,}")
    print(f"Passes: {args.repeats}")
    print(f"Workers: {min(args.workers, len(samples))}")
    print(f"Total planned classifications: {len(samples) * args.repeats:,}")
    print(f"Results: {args.output_dir}")

    existing_payloads = []
    passes_to_run = []
    for pass_index in range(1, args.repeats + 1):
        result_path = args.output_dir / f"pass-{pass_index}.json"
        if result_path.is_file() and not args.overwrite:
            existing_payloads.append(read_completed_pass(result_path, config))
            print(f"Reusing completed pass {pass_index}: {result_path}")
        else:
            passes_to_run.append(pass_index)

    classify_arguments = {
        "prompt": prompt,
        "model": args.model,
        "reasoning_effort": args.reasoning_effort,
        "max_output_tokens": args.max_output_tokens,
        "api_base": args.api_base,
        "api_key": api_key,
        "timeout": args.timeout,
        "max_retries": args.max_retries,
        "threshold": args.threshold,
        "seed": args.seed,
    }

    if passes_to_run and args.warmup_samples:
        warmup_count = min(args.warmup_samples, len(samples))
        print(f"Running {warmup_count} untimed warm-up classification(s)...")
        for sample in samples[:warmup_count]:
            classify_sample(
                sample,
                pass_index=0,
                **classify_arguments,
            )

    pass_payloads = list(existing_payloads)
    for pass_index in passes_to_run:
        print(f"\nStarting pass {pass_index}/{args.repeats}")

        def classify(sample):
            return classify_sample(
                sample,
                pass_index=pass_index,
                **classify_arguments,
            )

        rows, failures, duration = run_pass(
            samples,
            classify,
            workers=args.workers,
            progress_every=args.progress_every,
        )
        if failures:
            failure_path = args.output_dir / f"pass-{pass_index}-failures.json"
            atomic_write_json(
                failure_path,
                {
                    "config": config,
                    "pass_index": pass_index,
                    "duration_seconds": duration,
                    "successful_results": rows,
                    "failures": failures,
                },
            )
            raise SystemExit(
                f"Pass {pass_index} had {len(failures)} failures. "
                f"Details: {failure_path}"
            )

        summary = summarize_pass(rows, duration)
        payload = {
            "config": config,
            "pass_index": pass_index,
            "summary": summary,
            "results": rows,
        }
        result_path = args.output_dir / f"pass-{pass_index}.json"
        atomic_write_json(result_path, payload)
        pass_payloads.append(payload)
        print(
            f"Pass {pass_index}: {summary['accuracy']:.2%} accuracy in "
            f"{summary['duration_seconds']:.2f} seconds"
        )

    pass_payloads.sort(key=lambda payload: int(payload["pass_index"]))
    if len(pass_payloads) != args.repeats:
        raise SystemExit("Not all requested passes are complete")
    final_summary = overall_summary(pass_payloads, config)
    summary_path = args.output_dir / "summary.json"
    atomic_write_json(summary_path, final_summary)
    print_overall_summary(final_summary)
    print(f"Summary saved to {summary_path}")


if __name__ == "__main__":
    main()
