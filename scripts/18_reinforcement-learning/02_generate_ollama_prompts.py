#!/usr/bin/env python3
"""Generate a resumable GRPO writing-prompt dataset with Ollama."""

import argparse
from concurrent.futures import FIRST_COMPLETED, ThreadPoolExecutor, wait
import datetime as dt
import hashlib
import json
from pathlib import Path
import random
import re
import sys
import threading
import time
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen


PROJECT_DIR = Path(__file__).resolve().parents[2]
DEFAULT_MANIFEST = PROJECT_DIR / "data" / "grpo-prompts" / "manifest.jsonl"
DEFAULT_URL = "http://127.0.0.1:11434"
DEFAULT_MODEL = "qwen3.5:4b"
TOPIC_SYSTEM = """Reduce the supplied material to one broad subject.
Return only a lowercase noun phrase containing 2 to 10 words. Use no names,
acronyms, numbers, punctuation, study details, products, genes, mutations,
chemicals, software packages, organizations, or datasets. Generalize heavily.

Examples:
- a named mutation affecting a lens protein -> cell communication
- a particular image-analysis tool -> automated image analysis
- a study of a named chemical -> hormone exposure and tissue changes
- a bug in a named web framework -> software upgrade safety
"""
TOPIC_BATCH_SYSTEM = """Reduce each supplied item to one broad subject.
Return JSON with a `topics` array containing one string per input item in the
same order. Each string must be a lowercase noun phrase containing 2 to 10 words. Use no
names, acronyms, numbers, punctuation, study details, products, genes,
mutations, chemicals, software packages, organizations, or datasets. Generalize
heavily. Return no commentary outside the JSON object.
"""
SHORT_QUESTION_TEMPLATES = (
    "What should someone know about {topic}?",
    "What factors are most important in {topic}?",
    "What common challenges arise in {topic}?",
    "Which core ideas help explain {topic}?",
    "How can {topic} be understood in practical terms?",
    "What are the main considerations in {topic}?",
    "Which questions are central to {topic}?",
    "What key distinctions matter when discussing {topic}?",
    "How is {topic} commonly approached?",
    "What practical lessons can be drawn from {topic}?",
    "How can a beginner make sense of {topic}?",
    "What tradeoffs commonly arise in {topic}?",
)
LONG_QUESTION_TEMPLATES = (
    "What should a thorough explanation of {topic} cover?",
    "Which principles, challenges, and practical implications are most important when discussing {topic}?",
    "What factors shape {topic}, and how do they interact in practice?",
    "How can someone analyze {topic} from both theoretical and practical perspectives?",
    "What background, current approaches, and unresolved questions are relevant to {topic}?",
    "Which examples best illustrate the key ideas and tradeoffs in {topic}?",
    "How should someone evaluate competing explanations or approaches related to {topic}?",
    "What common misconceptions affect discussions of {topic}, and how can they be corrected?",
    "How can {topic} be explained clearly to someone encountering it for the first time?",
    "What historical context, major developments, and current debates are relevant to {topic}?",
    "Which assumptions shape thinking about {topic}, and when do those assumptions break down?",
    "How do theory, evidence, and practical experience contribute to understanding {topic}?",
    "What questions should guide a careful investigation of {topic}?",
    "How can examples and counterexamples clarify the central ideas in {topic}?",
    "Which methods are commonly used to study or address {topic}, and what are their limitations?",
    "What important tradeoffs arise in {topic}, and how should they be evaluated?",
)


class GenerationError(RuntimeError):
    pass


def read_jsonl(path):
    with path.open(encoding="utf-8") as file:
        return [json.loads(line) for line in file if line.strip()]


def normalize_question(text):
    text = text.strip()
    text = re.sub(r"^```(?:text|markdown)?\s*|\s*```$", "", text, flags=re.I)
    text = re.sub(r"^[\s\-*\d.)]+", "", text).strip()
    text = re.sub(r"^(?:question|prompt)\s*:\s*", "", text, flags=re.I)
    text = text.strip().strip('"').strip()
    return re.sub(r"\s+", " ", text)


def question_is_valid(question):
    words = len(question.split())
    if not question.endswith("?"):
        return False, "does not end with a question mark"
    if question.count("?") != 1:
        return False, "does not contain exactly one question mark"
    if not 6 <= words <= 60:
        return False, f"has {words} words; expected 6 to 60"
    lowered = question.lower()
    forbidden = ("source material", "provided passage", "writing dataset")
    if any(phrase in lowered for phrase in forbidden):
        return False, "mentions the generation setup"
    return True, ""


def normalize_topic(text):
    topic = normalize_question(text).rstrip("?.!").strip().lower()
    return re.sub(r"\s+", " ", topic)


def topic_is_valid(topic):
    words = topic.split()
    if not 2 <= len(words) <= 10:
        return False, f"topic has {len(words)} words; expected 2 to 10"
    if not re.fullmatch(r"[a-z]+(?:[ -][a-z]+)*", topic):
        return False, "topic must be a lowercase phrase without punctuation"
    return True, ""


def inspiration_text(text, *, max_words = 40, max_chars = 1_000):
    inspiration = " ".join(text.split()[:max_words])
    if len(inspiration) <= max_chars:
        return inspiration
    return inspiration[:max_chars].rsplit(" ", 1)[0]


def topic_prompt(row, data_dir):
    seed_path = data_dir / str(row["seed_file"])
    seed_text = seed_path.read_text(encoding="utf-8")
    inspiration = inspiration_text(seed_text)
    domain = str(row.get("source_collection") or "general")
    title = str(row.get("source_title") or row.get("source_name") or "")
    return (
        f"Broad source category: {domain}\n"
        f"Title: {title}\n"
        f"Topic inspiration:\n{inspiration}\n"
    )


def topic_batch_prompt(rows, data_dir):
    items = []
    for row in rows:
        seed_path = data_dir / str(row["seed_file"])
        items.append(
            {
                "id": str(row["prompt_id"]),
                "category": str(row.get("source_collection") or "general"),
                "title": str(
                    row.get("source_title") or row.get("source_name") or ""
                ),
                "excerpt": inspiration_text(
                    seed_path.read_text(encoding="utf-8")
                ),
            }
        )
    return json.dumps({"items": items}, ensure_ascii=False)


def build_question(topic, target_words, selector_seed):
    templates = (
        SHORT_QUESTION_TEMPLATES if target_words <= 100 else LONG_QUESTION_TEMPLATES
    )
    template_index = random.Random(selector_seed).randrange(len(templates))
    return templates[template_index].format(topic=topic), template_index


def post_json(url, payload, timeout):
    request = Request(
        url,
        data=json.dumps(payload).encode("utf-8"),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    try:
        with urlopen(request, timeout=timeout) as response:
            return json.loads(response.read().decode("utf-8"))
    except HTTPError as error:
        details = error.read().decode("utf-8", errors="replace")
        raise GenerationError(f"Ollama returned HTTP {error.code}: {details}") from error
    except (URLError, TimeoutError) as error:
        raise GenerationError(f"Could not reach Ollama: {error}") from error


def generate_question(
    row, data_dir, args
):
    ollama_url = str(row.get("_ollama_url") or args.ollama_url[0])
    topic_request = topic_prompt(row, data_dir)
    topic = ""
    last_error = "no response"
    for attempt in range(1, args.max_retries + 1):
        topic_seed = args.seed + int(row["seed_sample_id"]) * 211 + attempt
        payload = {
            "model": args.model,
            "messages": [
                {"role": "system", "content": TOPIC_SYSTEM},
                {"role": "user", "content": topic_request},
            ],
            "stream": False,
            "think": False,
            "keep_alive": args.keep_alive,
            "options": {
                "temperature": args.temperature,
                "top_p": args.top_p,
                "seed": topic_seed,
                "num_predict": 32,
            },
        }
        try:
            topic_result = post_json(
                f"{ollama_url.rstrip('/')}/api/chat", payload, args.timeout
            )
            message = topic_result.get("message")
            content = message.get("content") if isinstance(message, dict) else None
            if not isinstance(content, str) or not content.strip():
                raise GenerationError("Ollama returned no topic text")
            topic = normalize_topic(content)
            valid, reason = topic_is_valid(topic)
            if not valid:
                raise GenerationError(reason)
        except GenerationError as error:
            last_error = str(error)
            if attempt < args.max_retries:
                time.sleep(min(2**attempt, 8))
            continue
        break
    else:
        raise GenerationError(last_error)

    selector_seed = args.seed + int(row["seed_sample_id"]) * 101
    question, template_index = build_question(
        topic, int(row["target_words"]), selector_seed
    )
    valid, reason = question_is_valid(question)
    if not valid:
        raise GenerationError(reason)
    return {
        "prompt_id": row["prompt_id"],
        "broad_topic": topic,
        "question": question,
        "prompt_word_count": len(question.split()),
        "prompt_sha256": hashlib.sha256(
            (question.rstrip() + "\n").encode("utf-8")
        ).hexdigest(),
        "provider": "ollama",
        "ollama_url": ollama_url,
        "requested_model": args.model,
        "model": str(topic_result.get("model") or args.model),
        "generation_seed": topic_seed,
        "template_index": template_index,
        "attempt": attempt,
        "created_at_utc": dt.datetime.now(dt.UTC).isoformat(),
        "usage": {
            "topic_prompt_eval_count": topic_result.get("prompt_eval_count"),
            "topic_eval_count": topic_result.get("eval_count"),
            "topic_total_duration_ns": topic_result.get("total_duration"),
        },
    }


def generate_question_batch(
    rows, data_dir, args
):
    ollama_url = str(rows[0].get("_ollama_url") or args.ollama_url[0])
    expected_ids = [str(row["prompt_id"]) for row in rows]
    request_text = topic_batch_prompt(rows, data_dir)
    last_error = "no response"
    for attempt in range(1, args.max_retries + 1):
        topic_seed = args.seed + sum(int(row["seed_sample_id"]) for row in rows) * 211 + attempt
        payload = {
            "model": args.model,
            "messages": [
                {"role": "system", "content": TOPIC_BATCH_SYSTEM},
                {"role": "user", "content": request_text},
            ],
            "format": {
                "type": "object",
                "properties": {
                    "topics": {
                        "type": "array",
                        "items": {"type": "string"},
                    }
                },
                "required": ["topics"],
            },
            "stream": False,
            "think": False,
            "keep_alive": args.keep_alive,
            "options": {
                "temperature": args.temperature,
                "top_p": args.top_p,
                "seed": topic_seed,
                "num_predict": max(128, 32 * len(rows)),
            },
        }
        try:
            topic_result = post_json(
                f"{ollama_url.rstrip('/')}/api/chat", payload, args.timeout
            )
            message = topic_result.get("message")
            content = message.get("content") if isinstance(message, dict) else None
            if not isinstance(content, str) or not content.strip():
                raise GenerationError("Ollama returned no batched topic text")
            parsed = json.loads(content)
            returned = parsed.get("topics") if isinstance(parsed, dict) else None
            if not isinstance(returned, list):
                raise GenerationError("batched response has no topics list")
            if len(returned) != len(rows):
                raise GenerationError("batched topic count does not match the request")
            topic_by_id = {}
            invalid_ids = set()
            for row, item in zip(rows, returned):
                if not isinstance(item, str):
                    raise GenerationError("batched response contains a non-string topic")
                prompt_id = str(row["prompt_id"])
                topic = normalize_topic(item)
                valid, reason = topic_is_valid(topic)
                if not valid:
                    invalid_ids.add(prompt_id)
                    continue
                topic_by_id[prompt_id] = topic
        except (GenerationError, json.JSONDecodeError) as error:
            last_error = str(error)
            if attempt < args.max_retries:
                time.sleep(min(2**attempt, 8))
            continue
        break
    else:
        raise GenerationError(last_error)

    batch_id = hashlib.sha256("|".join(expected_ids).encode("utf-8")).hexdigest()[:12]
    results = []
    for row in rows:
        prompt_id = str(row["prompt_id"])
        if prompt_id in invalid_ids:
            results.append(
                generate_question(
                    {**row, "_ollama_url": ollama_url}, data_dir, args
                )
            )
            continue
        topic = topic_by_id[prompt_id]
        selector_seed = args.seed + int(row["seed_sample_id"]) * 101
        question, template_index = build_question(
            topic, int(row["target_words"]), selector_seed
        )
        valid, reason = question_is_valid(question)
        if not valid:
            raise GenerationError(f"{prompt_id}: {reason}")
        results.append(
            {
                "prompt_id": prompt_id,
                "broad_topic": topic,
                "question": question,
                "prompt_word_count": len(question.split()),
                "prompt_sha256": hashlib.sha256(
                    (question.rstrip() + "\n").encode("utf-8")
                ).hexdigest(),
                "provider": "ollama",
                "ollama_url": ollama_url,
                "requested_model": args.model,
                "model": str(topic_result.get("model") or args.model),
                "generation_seed": topic_seed,
                "template_index": template_index,
                "attempt": attempt,
                "batch_id": batch_id,
                "batch_size": len(rows),
                "created_at_utc": dt.datetime.now(dt.UTC).isoformat(),
                "usage": {
                    "batch_prompt_eval_count": topic_result.get("prompt_eval_count"),
                    "batch_eval_count": topic_result.get("eval_count"),
                    "batch_total_duration_ns": topic_result.get("total_duration"),
                },
            }
        )
    return results


def atomic_write_text(path, text):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(text.rstrip() + "\n", encoding="utf-8")
    temporary.replace(path)


def append_jsonl(path, row, lock):
    path.parent.mkdir(parents=True, exist_ok=True)
    with lock, path.open("a", encoding="utf-8") as file:
        file.write(json.dumps(row, ensure_ascii=False) + "\n")
        file.flush()


def valid_existing_question(path):
    if not path.is_file():
        return False
    valid, _ = question_is_valid(path.read_text(encoding="utf-8").strip())
    return valid


def export_prompts(
    manifest, output_root, output_path
):
    completed = 0
    temporary = output_path.with_suffix(output_path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8") as file:
        for row in manifest:
            prompt_path = output_root / str(row["prompt_file"])
            if not valid_existing_question(prompt_path):
                continue
            exported = {
                **row,
                "question": prompt_path.read_text(encoding="utf-8").strip(),
            }
            file.write(json.dumps(exported, ensure_ascii=False) + "\n")
            completed += 1
    temporary.replace(output_path)
    return completed


def run_batched(
    pending,
    manifest,
    output_root,
    data_dir,
    already_complete,
    args,
):
    progress_path = output_root / "generation-progress.jsonl"
    failure_path = output_root / "generation-failures.jsonl"
    write_lock = threading.Lock()
    generated = 0
    failures = 0
    for start in range(0, len(pending), args.batch_size):
        rows = pending[start : start + args.batch_size]
        try:
            results = generate_question_batch(rows, data_dir, args)
        except Exception as batch_error:
            print(
                f"Batch starting at {rows[0]['prompt_id']} failed: {batch_error}; "
                "retrying its rows individually",
                file=sys.stderr,
            )
            results = []
            for row in rows:
                try:
                    results.append(generate_question(row, data_dir, args))
                except Exception as row_error:
                    failures += 1
                    append_jsonl(
                        failure_path,
                        {
                            "prompt_id": row["prompt_id"],
                            "error": str(row_error),
                            "created_at_utc": dt.datetime.now(dt.UTC).isoformat(),
                        },
                        write_lock,
                    )
                    print(
                        f"Failed {row['prompt_id']}: {row_error}", file=sys.stderr
                    )
        row_by_id = {str(row["prompt_id"]): row for row in rows}
        for result in results:
            row = row_by_id[str(result["prompt_id"])]
            prompt_path = output_root / str(row["prompt_file"])
            atomic_write_text(prompt_path, str(result["question"]))
            append_jsonl(progress_path, result, write_lock)
            generated += 1
        overall = already_complete + generated
        print(f"Generated {overall:,}/{len(manifest):,}")
    return generated, failures


def build_parser():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument(
        "--ollama-url",
        action="append",
        help="Ollama base URL. Repeat to distribute requests across servers.",
    )
    parser.add_argument("--workers", type=int, default=1)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--max-retries", type=int, default=4)
    parser.add_argument("--timeout", type=float, default=300)
    parser.add_argument("--seed", type=int, default=17)
    parser.add_argument("--temperature", type=float, default=0.8)
    parser.add_argument("--top-p", type=float, default=0.95)
    parser.add_argument("--keep-alive", default="30m")
    parser.add_argument("--limit", type=int)
    return parser


def main():
    args = build_parser().parse_args()
    if args.ollama_url is None:
        args.ollama_url = [DEFAULT_URL]
    if args.workers < 1:
        raise SystemExit("--workers must be positive")
    if args.batch_size < 1:
        raise SystemExit("--batch-size must be positive")
    if args.batch_size > 1 and args.workers != 1:
        raise SystemExit("batched generation requires --workers 1")
    manifest_path = args.manifest.resolve()
    output_root = manifest_path.parent
    data_dir = output_root.parent
    manifest = read_jsonl(manifest_path)
    pending = [
        row
        for row in manifest
        if not valid_existing_question(output_root / str(row["prompt_file"]))
    ]
    if args.limit is not None:
        pending = pending[: args.limit]
    total_requested = len(pending)
    already_complete = len(manifest) - sum(
        not valid_existing_question(output_root / str(row["prompt_file"]))
        for row in manifest
    )
    print(f"Model: {args.model}")
    print(f"Ollama endpoints: {', '.join(args.ollama_url)}")
    print(f"Manifest prompts: {len(manifest):,}")
    print(f"Already complete: {already_complete:,}")
    print(f"Generating this run: {total_requested:,}")

    if args.batch_size > 1:
        print(f"Batch size: {args.batch_size}")
        _, failures = run_batched(
            pending,
            manifest,
            output_root,
            data_dir,
            already_complete,
            args,
        )
        completed = export_prompts(
            manifest, output_root, output_root / "prompts.jsonl"
        )
        print(f"Complete prompt files: {completed:,}/{len(manifest):,}")
        if failures:
            print(
                f"This run had {failures} failures. Rerun the same command to retry them."
            )
        if completed != len(manifest):
            raise SystemExit(1)
        return

    progress_path = output_root / "generation-progress.jsonl"
    failure_path = output_root / "generation-failures.jsonl"
    write_lock = threading.Lock()
    completed_this_run = 0
    failures = 0

    executor = ThreadPoolExecutor(max_workers=min(args.workers, max(1, len(pending))))
    iterator = iter(pending)
    in_flight = {}
    next_endpoint_index = 0

    def submit_available():
        nonlocal next_endpoint_index
        while len(in_flight) < args.workers:
            try:
                row = next(iterator)
            except StopIteration:
                return
            task_row = {
                **row,
                "_ollama_url": args.ollama_url[
                    next_endpoint_index % len(args.ollama_url)
                ],
            }
            next_endpoint_index += 1
            future = executor.submit(generate_question, task_row, data_dir, args)
            in_flight[future] = row

    try:
        submit_available()
        while in_flight:
            done, _ = wait(in_flight, return_when=FIRST_COMPLETED)
            for future in done:
                row = in_flight.pop(future)
                try:
                    result = future.result()
                except Exception as error:
                    failures += 1
                    append_jsonl(
                        failure_path,
                        {
                            "prompt_id": row["prompt_id"],
                            "error": str(error),
                            "created_at_utc": dt.datetime.now(dt.UTC).isoformat(),
                        },
                        write_lock,
                    )
                    print(f"Failed {row['prompt_id']}: {error}", file=sys.stderr)
                    continue
                prompt_path = output_root / str(row["prompt_file"])
                atomic_write_text(prompt_path, str(result["question"]))
                append_jsonl(progress_path, result, write_lock)
                completed_this_run += 1
                overall = already_complete + completed_this_run
                print(f"Generated {overall:,}/{len(manifest):,}")
            submit_available()
    except KeyboardInterrupt:
        print("Stopping after in-flight requests finish...", file=sys.stderr)
    finally:
        executor.shutdown(wait=True, cancel_futures=True)

    completed = export_prompts(
        manifest, output_root, output_root / "prompts.jsonl"
    )
    print(f"Complete prompt files: {completed:,}/{len(manifest):,}")
    if failures:
        print(f"This run had {failures} failures. Rerun the same command to retry them.")
    if completed != len(manifest):
        raise SystemExit(1)


if __name__ == "__main__":
    main()
