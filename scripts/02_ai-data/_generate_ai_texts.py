#!/usr/bin/env python3
"""Shared implementation for complete AI-text dataset generation.

The public provider scripts call this module in one of two modes. ``fresh``
creates a question from a human seed and then creates the answer that becomes
the AI sample. ``repair`` turns the existing ``generated-question`` rows into
full responses without discarding their questions.
"""

import argparse
import datetime as dt
import fcntl
import hashlib
import http.client
import json
import math
import os
import random
import re
import sys
import time
import urllib.error
import urllib.parse
import urllib.request
from collections import Counter, namedtuple
from concurrent.futures import FIRST_COMPLETED, ThreadPoolExecutor, wait
from contextlib import contextmanager
from copy import deepcopy
from pathlib import Path


DEFAULT_META = Path("data/meta.json")
DEFAULT_OUTPUT_FOLDER = Path("data/ai")
DEFAULT_COUNT = 5_000
DEFAULT_TIMEOUT = 300.0
DEFAULT_SEED = 17
DEFAULT_TOLERANCE = 0.20
DEFAULT_API_BASES = {
    "openai": "https://api.openai.com/v1",
    "openrouter": "https://openrouter.ai/api/v1",
    "gemini": "https://generativelanguage.googleapis.com/v1beta",
    "ollama": "http://localhost:11434",
}
DEFAULT_KEY_ENVS = {
    "openai": "OPENAI_API_KEY",
    "openrouter": "OPENROUTER_API_KEY",
    "gemini": "GEMINI_API_KEY",
}
OPENAI_MODELS = {
    "luna-high": ("gpt-5.6-luna", "high"),
    "luna-medium": ("gpt-5.6-luna", "medium"),
    "sol-medium": ("gpt-5.6-sol", "medium"),
}
SAFE_TOPICS = (
    "software engineering",
    "machine learning",
    "statistics",
    "computer hardware",
    "education",
    "history of technology",
    "data visualization",
    "scientific computing",
)
SAFE_ANGLES = (
    "practical tradeoffs",
    "common failure modes",
    "ways to evaluate results",
    "historical development",
    "implementation choices",
    "communication challenges",
    "resource constraints",
    "ways to teach newcomers",
)
BIOLOGY_PATTERN = re.compile(
    r"\b(?:biolog(?:y|ical)|biochem(?:istry|ical)|biomed(?:ical|icine)|"
    r"biotech(?:nology|nological)?|genom(?:e|es|ic|ics)|genetic(?:s|ally)?|"
    r"protein(?:s)?|enzyme(?:s)?|dna|rna|virus(?:es)?|viral|pathogen(?:s|ic)?|"
    r"bacteri(?:a|al|um)|fung(?:us|i|al)|vaccine(?:s|ation)?|disease(?:s)?|"
    r"clinical|medical|medicine|health(?:care)?|patient(?:s)?|therapy|"
    r"treatment(?:s)?|pharmaceutical(?:s)?|drug(?:s)?|cancer|oncolog(?:y|ical)"
    r")\b",
    re.IGNORECASE,
)

QUESTION_SYSTEM = """Create one standalone, substantive question for a writing dataset.
Return only the question. It must end with a question mark and be answerable
without access to source material. Use the supplied material only as topic
inspiration. Do not mention a passage, article, source, seed, dataset, requested
length, language model, or AI. Do not copy more than five consecutive words
from the supplied material. Avoid facts that require information after 2022.
"""


class GenerationError(RuntimeError):
    def __init__(
        self,
        message,
        *,
        retryable = False,
        code = None,
    ):
        super().__init__(message)
        self.retryable = retryable
        self.code = code


ModelResult = namedtuple("ModelResult", "text model usage")
RepairJob = namedtuple(
    "RepairJob", "sample sample_id target_words prompt_relative question"
)


def utc_now():
    return dt.datetime.now(dt.UTC).isoformat()


def word_count(text):
    return len(text.split())


def text_for_file(text):
    return text.rstrip() + "\n"


def sha256_text(text):
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def model_slug(model):
    slug = re.sub(r"[^a-z0-9._-]+", "-", model.lower()).strip("-._")
    if not slug:
        raise SystemExit("--model must contain at least one letter or number")
    return slug


def normalize_question(text):
    text = text.strip()
    if text.startswith("{"):
        try:
            payload = json.loads(text)
        except json.JSONDecodeError:
            pass
        else:
            if isinstance(payload, dict) and isinstance(payload.get("question"), str):
                text = payload["question"]
    text = re.sub(r"^```(?:text|markdown)?\s*|\s*```$", "", text, flags=re.I)
    text = re.sub(r"^[\s\-*\d.)]+", "", text).strip()
    text = re.sub(r"^(?:question|prompt)\s*:\s*", "", text, flags=re.I)
    text = text.strip().strip('"').strip()
    return re.sub(r"\s+", " ", text)


def normalize_response(text):
    text = text.replace("\r\n", "\n").strip()
    fenced = re.fullmatch(r"```(?:text|markdown)?\s*\n?(.*?)\n?```", text, re.I | re.S)
    if fenced:
        text = fenced.group(1).strip()
    text = re.sub(r"^(?:answer|response)\s*:\s*", "", text, flags=re.I)
    return text.strip()


def question_is_valid(question):
    count = word_count(question)
    if not question.endswith("?"):
        return False, "does not end with a question mark"
    if not 8 <= count <= 80:
        return False, f"has {count} words; expected 8 to 80"
    return True, ""


def stored_prompt_is_valid(prompt):
    """Legacy prompt rows only need usable text for response generation."""
    if not prompt.strip():
        return False, "is empty"
    return True, ""


def response_bounds(target_words, tolerance):
    return (
        max(1, math.floor(target_words * (1 - tolerance))),
        math.ceil(target_words * (1 + tolerance)),
    )


def response_is_valid(
    response, target_words, tolerance
):
    count = word_count(response)
    low, high = response_bounds(target_words, tolerance)
    if not response:
        return False, "is empty"
    if not low <= count <= high:
        return False, f"has {count} words; expected {low} to {high}"
    lowered = response.lower()
    if "as a language model" in lowered or "requested word count" in lowered:
        return False, "mentions the generation process"
    return True, ""


def trim_overlong_response(
    response, target_words, tolerance
):
    """Trim at the latest complete sentence inside the accepted word range."""
    low, high = response_bounds(target_words, tolerance)
    words = list(re.finditer(r"\S+", response))
    if len(words) <= high:
        return response
    for count in range(high, low - 1, -1):
        match = words[count - 1]
        if not re.search(r"[.!?][\"')\]]*$", match.group()):
            continue
        return response[: match.end()].rstrip()
    return response


def read_json(path):
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError as error:
        raise SystemExit(f"File does not exist: {path}") from error
    except json.JSONDecodeError as error:
        raise SystemExit(f"Malformed JSON in {path}: {error}") from error
    if not isinstance(payload, dict):
        raise SystemExit(f"Expected a JSON object in {path}")
    return payload


def atomic_write_text(path, text):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        handle.write(text)
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)


def atomic_write_json(path, payload):
    atomic_write_text(path, json.dumps(payload, ensure_ascii=False, indent=2) + "\n")


@contextmanager
def exclusive_file_lock(path, *, wait = True):
    """Hold an advisory process lock for a short shared-file operation."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a+", encoding="utf-8") as handle:
        flags = fcntl.LOCK_EX if wait else fcntl.LOCK_EX | fcntl.LOCK_NB
        try:
            fcntl.flock(handle.fileno(), flags)
        except BlockingIOError as error:
            raise SystemExit(
                f"Another process is already running this provider/model job: {path.name}"
            ) from error
        try:
            yield
        finally:
            fcntl.flock(handle.fileno(), fcntl.LOCK_UN)


def metadata_lock_path(meta_path):
    return meta_path.parent / ".locks" / "meta.lock"


def model_job_name(args):
    profile = (
        getattr(args, "source_model_selection", None)
        or getattr(args, "model_selection", None)
        or args.model
    )
    return f"{args.provider}-{model_slug(str(profile))}"


def job_lock_path(args):
    return args.meta.parent / ".locks" / f"job-{model_job_name(args)}.lock"


def dataset_path(data_dir, relative_file):
    path = (data_dir / relative_file).resolve()
    try:
        path.relative_to(data_dir.resolve())
    except ValueError as error:
        raise SystemExit(f"Metadata path leaves the dataset directory: {relative_file}") from error
    return path


def update_counts(metadata):
    samples = metadata.get("samples")
    if not isinstance(samples, list):
        raise SystemExit("Metadata has no samples list")
    counts = Counter(str(sample.get("label")) for sample in samples)
    metadata["counts"] = {
        "total": len(samples),
        "human": counts["human"],
        "ai": counts["ai"],
    }


def http_json(
    url,
    payload,
    headers,
    timeout,
    provider_name,
):
    request = urllib.request.Request(
        url,
        data=json.dumps(payload).encode("utf-8"),
        method="POST",
        headers={"Content-Type": "application/json", **headers},
    )
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            result = json.loads(response.read().decode("utf-8"))
    except urllib.error.HTTPError as error:
        details = error.read().decode("utf-8", errors="replace")
        error_code = None
        try:
            error_payload = json.loads(details)
        except json.JSONDecodeError:
            pass
        else:
            if isinstance(error_payload, dict) and isinstance(
                error_payload.get("error"), dict
            ):
                error_code = error_payload["error"].get("code")
        retryable = error.code in {408, 409, 429} or error.code >= 500
        raise GenerationError(
            f"{provider_name} returned HTTP {error.code}: {details}",
            retryable=retryable,
            code=str(error_code) if error_code is not None else None,
        ) from error
    except urllib.error.URLError as error:
        raise GenerationError(
            f"Could not connect to {provider_name}: {error.reason}", retryable=True
        ) from error
    except TimeoutError as error:
        raise GenerationError(f"{provider_name} request timed out", retryable=True) from error
    except http.client.IncompleteRead as error:
        raise GenerationError(
            f"{provider_name} returned an incomplete response: {error}", retryable=True
        ) from error
    except json.JSONDecodeError as error:
        raise GenerationError(
            f"{provider_name} returned invalid JSON: {error}", retryable=True
        ) from error
    if not isinstance(result, dict):
        raise GenerationError(f"{provider_name} returned a non-object response")
    return result


def openai_text(args, system, prompt, max_words):
    max_tokens = max(4_096, math.ceil(max_words * 3.5) + 1_024)
    payload = {
        "model": args.model,
        "instructions": system,
        "input": prompt,
        "reasoning": {"effort": args.reasoning_effort},
        "max_output_tokens": max_tokens,
        "store": False,
    }
    result = http_json(
        f"{args.api_base.rstrip('/')}/responses",
        payload,
        {"Authorization": f"Bearer {args.api_key}"},
        args.timeout,
        "OpenAI",
    )
    if result.get("status") != "completed":
        raise GenerationError(
            f"OpenAI response did not complete: {result.get('incomplete_details')!r}",
            retryable=True,
        )
    parts = []
    for item in result.get("output", []):
        if not isinstance(item, dict) or item.get("type") != "message":
            continue
        for part in item.get("content", []):
            if isinstance(part, dict) and part.get("type") == "output_text":
                parts.append(str(part.get("text", "")))
            if isinstance(part, dict) and part.get("type") == "refusal":
                raise GenerationError(f"OpenAI refused the request: {part.get('refusal')}")
    text = "".join(parts).strip()
    if not text:
        raise GenerationError("OpenAI returned no output text", retryable=True)
    usage = result.get("usage") or {}
    return ModelResult(text, str(result.get("model") or args.model), usage)


def openrouter_text(
    args, system, prompt, max_words
):
    payload = {
        "model": args.model,
        "messages": [
            {"role": "system", "content": system},
            {"role": "user", "content": prompt},
        ],
        "reasoning": {"enabled": False},
        "temperature": args.temperature,
        "max_tokens": max(512, math.ceil(max_words * 1.8) + 256),
        "stream": False,
    }
    result = http_json(
        f"{args.api_base.rstrip('/')}/chat/completions",
        payload,
        {
            "Authorization": f"Bearer {args.api_key}",
            "X-Title": "ai-detector-dev",
        },
        args.timeout,
        "OpenRouter",
    )
    if isinstance(result.get("error"), dict):
        raise GenerationError(f"OpenRouter returned an error: {result['error']}")
    choices = result.get("choices")
    if not isinstance(choices, list) or not choices:
        raise GenerationError("OpenRouter returned no choices", retryable=True)
    choice = choices[0]
    finish_reason = choice.get("finish_reason") if isinstance(choice, dict) else None
    if finish_reason not in {None, "stop", "length"}:
        raise GenerationError(
            f"OpenRouter stopped with finish_reason={finish_reason!r}", retryable=True
        )
    message = choice.get("message") if isinstance(choice, dict) else None
    content = message.get("content") if isinstance(message, dict) else None
    if isinstance(content, list):
        content = "".join(
            str(part.get("text", ""))
            for part in content
            if isinstance(part, dict) and part.get("type") == "text"
        )
    if not isinstance(content, str) or not content.strip():
        raise GenerationError("OpenRouter returned no output text", retryable=True)
    usage = {**(result.get("usage") or {}), "finish_reason": finish_reason}
    return ModelResult(content, str(result.get("model") or args.model), usage)


def gemini_text(args, system, prompt, max_words):
    payload = {
        "model": args.model,
        "system_instruction": system,
        "input": prompt,
        "store": False,
        "generation_config": {
            "thinking_level": args.thinking_level,
            "max_output_tokens": max(1_024, math.ceil(max_words * 2.0) + 512),
        },
    }
    result = http_json(
        f"{args.api_base.rstrip('/')}/interactions",
        payload,
        {"x-goog-api-key": args.api_key},
        args.timeout,
        "Gemini",
    )
    if result.get("status") != "completed":
        raise GenerationError(
            f"Gemini interaction did not complete: status={result.get('status')!r}",
            retryable=True,
        )
    parts = []
    for step in result.get("steps", []):
        if not isinstance(step, dict) or step.get("type") != "model_output":
            continue
        for part in step.get("content", []):
            if isinstance(part, dict) and part.get("type") == "text":
                parts.append(str(part.get("text", "")))
    text = "".join(parts).strip()
    if not text:
        raise GenerationError("Gemini returned no output text", retryable=True)
    return ModelResult(text, str(result.get("model") or args.model), result.get("usage") or {})


def ollama_text(
    args,
    system,
    prompt,
    max_words,
    generation_seed,
):
    payload = {
        "model": args.model,
        "messages": [
            {"role": "system", "content": system},
            {"role": "user", "content": prompt},
        ],
        "stream": False,
        "think": False,
        "keep_alive": args.keep_alive,
        "options": {
            "temperature": args.temperature,
            "top_p": args.top_p,
            "seed": generation_seed,
            "num_predict": max(512, math.ceil(max_words * 1.8) + 256),
        },
    }
    result = http_json(
        f"{args.api_base.rstrip('/')}/api/chat",
        payload,
        {},
        args.timeout,
        "Ollama",
    )
    message = result.get("message")
    content = message.get("content") if isinstance(message, dict) else None
    if not isinstance(content, str) or not content.strip():
        raise GenerationError("Ollama returned no output text", retryable=True)
    usage = {
        "prompt_eval_count": result.get("prompt_eval_count"),
        "eval_count": result.get("eval_count"),
        "total_duration_ns": result.get("total_duration"),
    }
    return ModelResult(content, str(result.get("model") or args.model), usage)


def call_model(
    args,
    system,
    prompt,
    max_words,
    generation_seed,
):
    if args.provider == "openai":
        return openai_text(args, system, prompt, max_words)
    if args.provider == "openrouter":
        return openrouter_text(args, system, prompt, max_words)
    if args.provider == "gemini":
        return gemini_text(args, system, prompt, max_words)
    if args.provider == "ollama":
        return ollama_text(args, system, prompt, max_words, generation_seed)
    raise AssertionError(f"Unknown provider: {args.provider}")


def response_system(target_words, tolerance, previous_words):
    low, high = response_bounds(target_words, tolerance)
    aim = target_words
    retry_note = ""
    if previous_words is not None:
        if previous_words < low:
            aim = high
            retry_note = (
                f" A previous answer had only {previous_words} words. The replacement "
                f"must have at least {low} words, so expand it substantially and aim "
                f"near {high} words."
            )
        else:
            retry_note = (
                f" A previous answer had {previous_words} words and was too long, so "
                "write a shorter complete replacement."
            )
    return (
        "Answer the user's question directly as a self-contained piece of prose. "
        f"Write between {low} and {high} words, aiming for about {aim}."
        f"{retry_note} Return only the answer. Use natural paragraphs where useful. "
        "Do not discuss these instructions, the requested length, a dataset, a "
        "language model, or AI. Do not claim access to a source that the reader "
        "cannot see. Avoid facts that require information after 2022."
    )


def safe_fallback_question(item_seed):
    topic = SAFE_TOPICS[item_seed % len(SAFE_TOPICS)]
    angle = SAFE_ANGLES[(item_seed // len(SAFE_TOPICS)) % len(SAFE_ANGLES)]
    return (
        f"How do {angle} shape practical work in {topic}, and what examples help "
        "explain the most important considerations?"
    )


def add_prompt_replacement_usage(
    result,
    original_question,
    replacement_question,
    reason,
):
    return ModelResult(
        result.text,
        result.model,
        {
            **result.usage,
            "prompt_replacement": {
                "reason": reason,
                "original_prompt_sha256": sha256_text(
                    text_for_file(original_question)
                ),
                "replacement_question": replacement_question,
            },
        },
    )


def generate_question(
    args,
    seed_sample,
    data_dir,
    prompt_path,
):
    if prompt_path.is_file():
        question = prompt_path.read_text(encoding="utf-8").strip()
        valid, reason = question_is_valid(question)
        if valid:
            return question, None, None
        raise SystemExit(f"Existing prompt is invalid at {prompt_path}: {reason}")

    seed_file = dataset_path(data_dir, str(seed_sample["file"]))
    seed_text = seed_file.read_text(encoding="utf-8")
    inspiration = " ".join(seed_text.split()[:160])
    title = str(seed_sample.get("title") or seed_sample.get("source") or "Untitled")
    domain = str(seed_sample.get("collection") or "general")
    if BIOLOGY_PATTERN.search(f"{title}\n{inspiration}"):
        topic = SAFE_TOPICS[int(seed_sample["id"]) % len(SAFE_TOPICS)]
        title = topic
        domain = topic
        inspiration = ""
    target = target_words_for_seed(seed_sample)
    user_prompt = (
        f"Domain: {domain}\nTitle or topic: {title}\n"
        f"The eventual answer should support roughly {target} words.\n"
    )
    if inspiration:
        user_prompt += f"Topic inspiration:\n{inspiration}\n"

    last_error = "no response"
    for attempt in range(1, args.max_retries + 1):
        try:
            result = call_model(
                args,
                QUESTION_SYSTEM,
                user_prompt,
                80,
                args.seed + int(seed_sample["id"]) * 101 + attempt,
            )
        except GenerationError as error:
            last_error = str(error)
            print(f"Question attempt {attempt} failed: {error}", file=sys.stderr)
            if not error.retryable:
                break
            if attempt < args.max_retries:
                time.sleep(min(2**attempt, 10))
            continue
        question = normalize_question(result.text)
        valid, reason = question_is_valid(question)
        if valid:
            atomic_write_text(prompt_path, text_for_file(question))
            return question, {
                "generator": generator_metadata(args, result.model),
                "usage": result.usage,
                "created_at_utc": utc_now(),
            }, None
        last_error = reason
        print(f"Question attempt {attempt} rejected: {reason}", file=sys.stderr)
    return None, None, last_error


def generate_response(
    args, question, target_words, item_seed
):
    active_question = question
    replacement_reason = None
    if args.provider == "openai" and BIOLOGY_PATTERN.search(question):
        active_question = safe_fallback_question(item_seed)
        replacement_reason = "biology-topic-prefilter"
        print(
            f"Row {item_seed}: replacing biology-related prompt with a safe topic",
            file=sys.stderr,
        )
    previous_words = None
    last_error = "no response"
    for attempt in range(1, args.max_retries + 1):
        try:
            result = call_model(
                args,
                response_system(target_words, args.length_tolerance, previous_words),
                active_question,
                target_words,
                args.seed + item_seed * 10_007 + attempt,
            )
        except GenerationError as error:
            last_error = str(error)
            print(
                f"Row {item_seed} response attempt {attempt} failed: {error}",
                file=sys.stderr,
            )
            if (
                args.provider == "openai"
                and error.code == "bio_policy"
                and active_question == question
            ):
                active_question = safe_fallback_question(item_seed)
                replacement_reason = "openai-bio-policy"
                print(
                    f"Row {item_seed}: replacing policy-blocked prompt with a safe topic",
                    file=sys.stderr,
                )
                continue
            if not error.retryable:
                break
            if attempt < args.max_retries:
                jitter = random.Random(
                    args.seed + item_seed * 1_009 + attempt
                ).uniform(0.0, 1.0)
                time.sleep(min(2**attempt, 10) + jitter)
            continue
        response = normalize_response(result.text)
        original_words = word_count(response)
        response = trim_overlong_response(
            response, target_words, args.length_tolerance
        )
        if word_count(response) != original_words:
            result = ModelResult(
                result.text,
                result.model,
                {
                    **result.usage,
                    "response_postprocessing": {
                        "type": "sentence-boundary-trim",
                        "original_word_count": original_words,
                        "final_word_count": word_count(response),
                    },
                },
            )
            print(
                f"Row {item_seed}: trimmed response from {original_words} to "
                f"{word_count(response)} words",
                file=sys.stderr,
            )
        valid, reason = response_is_valid(
            response, target_words, args.length_tolerance
        )
        if valid:
            if replacement_reason is not None:
                result = add_prompt_replacement_usage(
                    result,
                    question,
                    active_question,
                    replacement_reason,
                )
            return response, result, None
        previous_words = word_count(response)
        last_error = reason
        print(
            f"Row {item_seed} response attempt {attempt} rejected: {reason}",
            file=sys.stderr,
        )
    return None, None, last_error


def target_words_for_seed(sample):
    target = sample.get("target_words")
    if isinstance(target, int) and target > 0:
        return target
    words = int(sample.get("word_count") or 250)
    choices = (50, 100, 250, 500, 1_000, 2_000)
    return min(choices, key=lambda value: abs(value - words))


def generator_metadata(args, actual_model):
    metadata = {
        "provider": args.provider,
        "requested_model": args.model,
        "model": actual_model,
        "dataset_seed": args.seed,
    }
    if args.provider == "openai":
        metadata["model_selection"] = args.model_selection
        metadata["reasoning_effort"] = args.reasoning_effort
    elif args.provider == "gemini":
        metadata["thinking_level"] = args.thinking_level
    elif args.provider in {"openrouter", "ollama"}:
        metadata["temperature"] = args.temperature
    if args.provider == "ollama":
        metadata["top_p"] = args.top_p
    return metadata


def generator_matches_model(
    generator,
    provider,
    model,
    model_selection,
    reasoning_effort,
):
    if not isinstance(generator, dict) or generator.get("provider") != provider:
        return False
    recorded = {
        value
        for value in (
            generator.get("requested_model"),
            generator.get("model"),
            generator.get("model_selection"),
        )
        if value is not None
    }
    candidates = {value for value in (model, model_selection) if value is not None}
    if not recorded & candidates:
        return False
    if provider == "openai":
        return generator.get("reasoning_effort") == reasoning_effort
    return True


def matches_requested_model(sample, args):
    return generator_matches_model(
        sample.get("generator"),
        args.provider,
        args.model,
        getattr(args, "model_selection", None),
        getattr(args, "reasoning_effort", None),
    )


def matches_repair_source(sample, args):
    generator = sample.get("generator")
    if sample.get("sample_type") == "generated-response" and isinstance(
        sample.get("prompt_generator"), dict
    ):
        generator = sample["prompt_generator"]
    return generator_matches_model(
        generator,
        args.provider,
        getattr(args, "source_model", args.model),
        getattr(
            args,
            "source_model_selection",
            getattr(args, "model_selection", None),
        ),
        getattr(
            args,
            "source_reasoning_effort",
            getattr(args, "reasoning_effort", None),
        ),
    )


def prompt_relative_path(
    args, sample_id, *, recovered
):
    prefix = "prompts/recovered" if recovered else "prompts"
    model = getattr(args, "source_model", args.model) if recovered else args.model
    group = f"{args.provider}-{model_slug(model)}"
    return f"{prefix}/{group}/{sample_id}.txt"


def materialize_response_prompt(
    args,
    data_dir,
    item_id,
    prompt_relative,
    question,
    result,
    *,
    recovered,
):
    replacement = result.usage.get("prompt_replacement")
    if not isinstance(replacement, dict):
        return prompt_relative, question, result
    replacement_question = replacement.get("replacement_question")
    if not isinstance(replacement_question, str) or not replacement_question.strip():
        return prompt_relative, question, result

    prefix = (
        "prompts/replacements/recovered"
        if recovered
        else "prompts/replacements/fresh"
    )
    model = getattr(args, "source_model", args.model) if recovered else args.model
    group = f"{args.provider}-{model_slug(model)}"
    replacement_relative = f"{prefix}/{group}/{item_id}.txt"
    replacement_path = dataset_path(data_dir, replacement_relative)
    atomic_write_text(replacement_path, text_for_file(replacement_question))
    replacement_metadata = {
        **replacement,
        "original_prompt_file": prompt_relative,
        "replacement_prompt_file": replacement_relative,
    }
    updated_result = ModelResult(
        result.text,
        result.model,
        {**result.usage, "prompt_replacement": replacement_metadata},
    )
    return replacement_relative, replacement_question, updated_result


def seed_provenance(sample):
    return {
        "seed_sample_id": int(sample["id"]),
        "seed_file": sample.get("file"),
        "seed_collection": sample.get("collection"),
        "seed_source_document_id": (
            sample.get("source_document_id") or sample.get("source")
        ),
        "seed_title": sample.get("title"),
        "seed_source_url": sample.get("source_url"),
        "seed_license": sample.get("license"),
        "seed_license_url": sample.get("license_url"),
        "seed_public_hub_eligible": sample.get("public_hub_eligible"),
    }


def response_fields(
    args,
    response,
    result,
    prompt_relative,
    question,
    question_info,
):
    response_file_text = text_for_file(response)
    question_file_text = text_for_file(question)
    fields = {
        "sample_type": "generated-response",
        "word_count": word_count(response),
        "sha256": sha256_text(response_file_text),
        "prompt_file": prompt_relative,
        "prompt_word_count": word_count(question),
        "prompt_sha256": sha256_text(question_file_text),
        "generator": generator_metadata(args, result.model),
        "batch_usage": result.usage,
        "created_at_utc": utc_now(),
    }
    if question_info is not None:
        fields["prompt_generator"] = question_info["generator"]
        fields["prompt_batch_usage"] = question_info["usage"]
        fields["prompt_created_at_utc"] = question_info["created_at_utc"]
    if isinstance(result.usage.get("prompt_replacement"), dict):
        fields["prompt_replacement"] = deepcopy(
            result.usage["prompt_replacement"]
        )
    return fields


def record_failure(data_dir, args, payload):
    path = data_dir / "generation-failures.jsonl"
    record = {
        "created_at_utc": utc_now(),
        "provider": args.provider,
        "model": args.model,
        **payload,
    }
    lock_path = data_dir / ".locks" / "failures.lock"
    with exclusive_file_lock(lock_path):
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")
            handle.flush()
            os.fsync(handle.fileno())


def report_repair_failure(
    data_dir,
    args,
    sample,
    error,
):
    """Record one unusable repair row without stopping the remaining job."""
    record_failure(
        data_dir,
        args,
        {"mode": "repair", "sample_id": sample.get("id"), "error": error},
    )
    print(f"Skipping row {sample.get('id', 'unknown')}: {error}", file=sys.stderr)


def matching_response_rows(
    metadata, args
):
    samples = metadata.get("samples")
    if not isinstance(samples, list):
        raise SystemExit("Metadata has no samples list")
    return [
        sample
        for sample in samples
        if sample.get("label") == "ai"
        and sample.get("sample_type") == "generated-response"
        and matches_requested_model(sample, args)
    ]


def claims_path(data_dir):
    return data_dir / "generation-claims.json"


def read_claims(data_dir):
    path = claims_path(data_dir)
    if not path.exists():
        return {"version": 1, "claims": {}}
    payload = read_json(path)
    if payload.get("version") != 1 or not isinstance(payload.get("claims"), dict):
        raise SystemExit(f"Unsupported generation claims file: {path}")
    return payload


def claim_fresh_candidates(
    args,
):
    """Atomically reserve human seeds for one provider/model job."""
    data_dir = args.meta.parent.resolve()
    with exclusive_file_lock(metadata_lock_path(args.meta)):
        metadata = read_json(args.meta)
        samples = metadata.get("samples")
        if not isinstance(samples, list):
            raise SystemExit("Metadata has no samples list")
        existing = matching_response_rows(metadata, args)
        if len(existing) > args.count:
            raise SystemExit(
                f"Found {len(existing)} completed rows for this model, more than "
                f"--count {args.count}"
            )
        claims = read_claims(data_dir)
        all_claims = claims["claims"]
        claim_key = model_job_name(args)
        claim = all_claims.get(claim_key)
        if claim is not None:
            if int(claim.get("count", -1)) != args.count or int(
                claim.get("seed", -1)
            ) != args.seed:
                raise SystemExit(
                    f"Existing seed claim for {claim_key} used a different --count "
                    "or --seed. Resume with the original values."
                )
            seed_ids = [int(value) for value in claim.get("seed_sample_ids", [])]
        else:
            existing_seed_ids = [int(sample["seed_sample_id"]) for sample in existing]
            reserved_seed_ids = {
                int(sample["seed_sample_id"])
                for sample in samples
                if sample.get("label") == "ai"
                and sample.get("seed_sample_id") is not None
            }
            for other_claim in all_claims.values():
                if not isinstance(other_claim, dict):
                    continue
                reserved_seed_ids.update(
                    int(value) for value in other_claim.get("seed_sample_ids", [])
                )
            reserved_seed_ids.difference_update(existing_seed_ids)
            available = [
                sample
                for sample in samples
                if sample.get("label") == "human"
                and int(sample["id"]) not in reserved_seed_ids
                and dataset_path(data_dir, str(sample.get("file"))).is_file()
            ]
            rng = random.Random(f"{args.seed}:{claim_key}")
            rng.shuffle(available)
            needed = args.count - len(existing)
            if len(available) < needed:
                raise SystemExit(
                    f"Only {len(available)} unclaimed human seeds remain, but "
                    f"{needed} responses are needed"
                )
            seed_ids = existing_seed_ids + [
                int(sample["id"]) for sample in available[:needed]
            ]
            all_claims[claim_key] = {
                "provider": args.provider,
                "requested_model": args.model,
                "model_selection": getattr(args, "model_selection", None),
                "reasoning_effort": getattr(args, "reasoning_effort", None),
                "count": args.count,
                "seed": args.seed,
                "seed_sample_ids": seed_ids,
                "created_at_utc": utc_now(),
            }
            atomic_write_json(claims_path(data_dir), claims)

        completed_seed_ids = {
            int(sample["seed_sample_id"]) for sample in existing
        }
        human_by_id = {
            int(sample["id"]): sample
            for sample in samples
            if sample.get("label") == "human"
        }
        missing_ids = [seed_id for seed_id in seed_ids if seed_id not in human_by_id]
        if missing_ids:
            raise SystemExit(
                f"Seed claim {claim_key} references missing human rows: {missing_ids[:5]}"
            )
        candidates = [
            human_by_id[seed_id]
            for seed_id in seed_ids
            if seed_id not in completed_seed_ids
        ]
        candidates.sort(
            key=lambda sample: not dataset_path(
                data_dir,
                prompt_relative_path(args, int(sample["id"]), recovered=False),
            ).is_file()
        )
        return candidates, len(existing)


def commit_repair_response(
    args,
    sample_id,
    response,
    result,
    prompt_relative,
    question,
    *,
    recovered = False,
):
    """Replace one question row while preserving concurrent metadata changes."""
    data_dir = args.meta.parent.resolve()
    with exclusive_file_lock(metadata_lock_path(args.meta)):
        metadata = read_json(args.meta)
        samples = metadata.get("samples")
        if not isinstance(samples, list):
            raise SystemExit("Metadata has no samples list")
        current = next(
            (sample for sample in samples if int(sample["id"]) == sample_id), None
        )
        if current is None:
            raise SystemExit(f"Metadata row disappeared during repair: {sample_id}")
        if current.get("sample_type") == "generated-response":
            return False
        if current.get("sample_type") != "generated-question":
            raise SystemExit(
                f"Metadata row {sample_id} changed to unexpected sample_type "
                f"{current.get('sample_type')!r}"
            )
        old_generator = deepcopy(current.get("generator"))
        old_usage = deepcopy(current.get("batch_usage"))
        old_created = current.get("created_at_utc")
        sample_path = dataset_path(data_dir, str(current["file"]))
        atomic_write_text(sample_path, text_for_file(response))
        current.update(
            response_fields(
                args,
                response,
                result,
                prompt_relative,
                question,
                None,
            )
        )
        current["prompt_generator"] = old_generator
        current["prompt_batch_usage"] = old_usage
        current["prompt_created_at_utc"] = old_created
        if recovered:
            current["response_recovered_after_interruption"] = True
        update_counts(metadata)
        atomic_write_json(args.meta, metadata)
        return True


def repair_remaining(args, selected_ids):
    with exclusive_file_lock(metadata_lock_path(args.meta)):
        metadata = read_json(args.meta)
        samples = metadata.get("samples")
        if not isinstance(samples, list):
            raise SystemExit("Metadata has no samples list")
        return sum(
            int(sample["id"]) in selected_ids
            and sample.get("sample_type") == "generated-question"
            for sample in samples
        )


def commit_fresh_response(
    args,
    seed_sample,
    output_folder,
    response,
    result,
    prompt_relative,
    question,
    question_info,
):
    """Append one response with collision-free IDs and file allocation."""
    data_dir = args.meta.parent.resolve()
    output_folder = output_folder.resolve()
    seed_id = int(seed_sample["id"])
    with exclusive_file_lock(metadata_lock_path(args.meta)):
        metadata = read_json(args.meta)
        samples = metadata.get("samples")
        if not isinstance(samples, list):
            raise SystemExit("Metadata has no samples list")
        existing = matching_response_rows(metadata, args)
        if any(int(sample["seed_sample_id"]) == seed_id for sample in existing):
            return len(existing)
        output_path = next_numbered_file(output_folder)
        relative_output = output_path.relative_to(data_dir).as_posix()
        next_metadata_id = max(
            (int(sample["id"]) for sample in samples), default=0
        ) + 1
        next_generation_index = max(
            (int(sample.get("generation_index") or 0) for sample in existing),
            default=0,
        ) + 1
        atomic_write_text(output_path, text_for_file(response))
        sample = {
            "id": next_metadata_id,
            "file": relative_output,
            "label": "ai",
            "collection": args.provider,
            "source": f"{args.provider}-{args.model}",
            "generation_index": next_generation_index,
            "target_response_words": target_words_for_seed(seed_sample),
            "public_hub_eligible": bool(seed_sample.get("public_hub_eligible")),
            **seed_provenance(seed_sample),
            **response_fields(
                args,
                response,
                result,
                prompt_relative,
                question,
                question_info,
            ),
        }
        samples.append(sample)
        update_counts(metadata)
        atomic_write_json(args.meta, metadata)
        return len(existing) + 1


def fresh_completed_count(args):
    with exclusive_file_lock(metadata_lock_path(args.meta)):
        return len(matching_response_rows(read_json(args.meta), args))


def complete_repair_job(
    args,
    data_dir,
    job,
    outcome,
):
    response, result, error = outcome
    if response is None or result is None:
        report_repair_failure(
            data_dir,
            args,
            job.sample,
            f"response generation failed: {error}",
        )
        return "failed"

    prompt_relative, question, result = materialize_response_prompt(
        args,
        data_dir,
        job.sample_id,
        job.prompt_relative,
        job.question,
        result,
        recovered=True,
    )
    committed = commit_repair_response(
        args,
        job.sample_id,
        response,
        result,
        prompt_relative,
        question,
    )
    return "generated" if committed else "already-complete"


def run_repair_jobs(
    args,
    data_dir,
    jobs,
    completed,
    total,
):
    if not jobs:
        return completed, 0, False

    worker_count = min(args.workers, len(jobs))
    print(f"Workers: {worker_count}")
    job_iterator = iter(jobs)
    in_flight = {}
    failures = 0
    interrupted = False
    executor = ThreadPoolExecutor(
        max_workers=worker_count,
        thread_name_prefix=f"{args.provider}-repair",
    )

    def submit_available():
        while len(in_flight) < worker_count:
            try:
                job = next(job_iterator)
            except StopIteration:
                return
            future = executor.submit(
                generate_response,
                args,
                job.question,
                job.target_words,
                job.sample_id,
            )
            in_flight[future] = job

    def process_completed(
        futures,
    ):
        nonlocal completed, failures
        for future in futures:
            job = in_flight[future]
            status = complete_repair_job(
                args,
                data_dir,
                job,
                future.result(),
            )
            if status == "generated":
                completed += 1
                print(f"Generated {completed}/{total}")
            elif status == "failed":
                failures += 1
            del in_flight[future]

    try:
        submit_available()
        while in_flight:
            done, _ = wait(in_flight, return_when=FIRST_COMPLETED)
            process_completed(done)
            submit_available()
    except KeyboardInterrupt:
        interrupted = True
        print(
            "Stopping new work. Waiting for in-flight requests so their results "
            "can be committed safely...",
            file=sys.stderr,
        )
        for future in list(in_flight):
            if future.cancel():
                del in_flight[future]
        while in_flight:
            done, _ = wait(in_flight, return_when=FIRST_COMPLETED)
            process_completed(done)
    finally:
        executor.shutdown(wait=True, cancel_futures=True)

    return completed, failures, interrupted


def repair_dataset(args):
    with exclusive_file_lock(metadata_lock_path(args.meta)):
        metadata = read_json(args.meta)
    samples = metadata.get("samples")
    if not isinstance(samples, list):
        raise SystemExit("Metadata has no samples list")
    data_dir = args.meta.parent.resolve()
    matching = [
        sample
        for sample in samples
        if sample.get("label") == "ai" and matches_repair_source(sample, args)
    ]
    matching.sort(key=lambda sample: int(sample.get("generation_index") or sample["id"]))
    if args.count is not None:
        if len(matching) < args.count:
            raise SystemExit(
                f"Found only {len(matching)} matching AI rows, fewer than --count {args.count}"
            )
        matching = matching[: args.count]
    pending = [sample for sample in matching if sample.get("sample_type") == "generated-question"]
    unexpected = [
        sample
        for sample in matching
        if sample.get("sample_type") not in {"generated-question", "generated-response"}
    ]
    if unexpected:
        raise SystemExit(f"Found {len(unexpected)} matching rows with unexpected sample_type")
    selected_ids = {int(sample["id"]) for sample in matching}
    completed = len(matching) - len(pending)
    print(f"Matching rows: {len(matching)}")
    print(f"Already complete: {completed}")
    print(f"Responses remaining: {len(pending)}")
    failures = 0
    jobs = []

    for sample in pending:
        try:
            sample_id = int(sample["id"])
            target = int(sample.get("target_response_words") or 0)
        except (KeyError, TypeError, ValueError) as error:
            failures += 1
            report_repair_failure(
                data_dir,
                args,
                sample,
                f"has invalid repair metadata: {error}",
            )
            continue
        if target <= 0:
            failures += 1
            report_repair_failure(
                data_dir,
                args,
                sample,
                "has no positive target_response_words",
            )
            continue

        try:
            sample_path = dataset_path(data_dir, str(sample["file"]))
            prompt_relative = prompt_relative_path(args, sample_id, recovered=True)
            prompt_path = dataset_path(data_dir, prompt_relative)
            current_text = sample_path.read_text(encoding="utf-8").strip()
            if prompt_path.is_file():
                question = prompt_path.read_text(encoding="utf-8").strip()
                valid, reason = stored_prompt_is_valid(question)
                if not valid:
                    failures += 1
                    report_repair_failure(
                        data_dir,
                        args,
                        sample,
                        f"has an unusable saved prompt: {reason}",
                    )
                    continue
                if current_text != question:
                    valid, reason = response_is_valid(
                        current_text,
                        target,
                        args.length_tolerance,
                    )
                    if valid:
                        recovered_result = ModelResult(current_text, args.model, {})
                        commit_repair_response(
                            args,
                            sample_id,
                            current_text,
                            recovered_result,
                            prompt_relative,
                            question,
                            recovered=True,
                        )
                        completed += 1
                        print(
                            f"Recovered {completed}/{len(matching)} "
                            "from an interrupted write"
                        )
                        continue
                    atomic_write_text(sample_path, text_for_file(question))
                    print(
                        f"Restored prompt for row {sample_id} after an "
                        f"invalid interrupted write ({reason})",
                        file=sys.stderr,
                    )
            else:
                question = current_text
                valid, reason = stored_prompt_is_valid(question)
                if not valid:
                    failures += 1
                    report_repair_failure(
                        data_dir,
                        args,
                        sample,
                        f"has an unusable stored prompt: {reason}",
                    )
                    continue
                atomic_write_text(prompt_path, text_for_file(question))
        except (KeyError, OSError) as error:
            failures += 1
            report_repair_failure(
                data_dir,
                args,
                sample,
                f"could not prepare the stored prompt: {error}",
            )
            continue

        jobs.append(
            RepairJob(
                sample=sample,
                sample_id=sample_id,
                target_words=target,
                prompt_relative=prompt_relative,
                question=question,
            )
        )

    completed, generation_failures, interrupted = run_repair_jobs(
        args,
        data_dir,
        jobs,
        completed,
        len(matching),
    )
    failures += generation_failures

    remaining = repair_remaining(args, selected_ids)
    if interrupted:
        raise SystemExit(
            f"Stopped safely with {remaining} question rows remaining. "
            "Run the same command again to resume."
        )
    if failures or remaining:
        raise SystemExit(
            f"Finished the pass with {failures} failures and {remaining} question rows remaining. "
            "Run the same command again to retry them."
        )
    print(f"Done. All {len(matching)} matching rows contain generated responses.")


def next_numbered_file(output_folder):
    ids = [int(path.stem) for path in output_folder.glob("*.txt") if path.stem.isdigit()]
    return output_folder / f"{max(ids, default=0) + 1}.txt"


def fresh_dataset(args):
    data_dir = args.meta.parent.resolve()
    output_folder = args.output_folder.resolve()
    try:
        output_folder.relative_to(data_dir)
    except ValueError as error:
        raise SystemExit(f"--output-folder must be inside {data_dir}") from error
    output_folder.mkdir(parents=True, exist_ok=True)
    candidates, existing_count = claim_fresh_candidates(args)
    needed = args.count - existing_count

    print(f"Existing responses: {existing_count}")
    print(f"Responses remaining: {needed}")
    generated = existing_count
    failures = 0

    for seed_sample in candidates:
        if generated >= args.count:
            break
        seed_id = int(seed_sample["id"])
        prompt_relative = prompt_relative_path(args, seed_id, recovered=False)
        prompt_path = dataset_path(data_dir, prompt_relative)
        question, question_info, question_error = generate_question(
            args, seed_sample, data_dir, prompt_path
        )
        if question is None:
            failures += 1
            record_failure(
                data_dir,
                args,
                {"mode": "fresh-question", "seed_sample_id": seed_id, "error": question_error},
            )
            continue
        target = target_words_for_seed(seed_sample)
        response, result, response_error = generate_response(args, question, target, seed_id)
        if response is None or result is None:
            failures += 1
            record_failure(
                data_dir,
                args,
                {"mode": "fresh-response", "seed_sample_id": seed_id, "error": response_error},
            )
            continue

        prompt_relative, question, result = materialize_response_prompt(
            args,
            data_dir,
            seed_id,
            prompt_relative,
            question,
            result,
            recovered=False,
        )

        generated = commit_fresh_response(
            args,
            seed_sample,
            output_folder,
            response,
            result,
            prompt_relative,
            question,
            question_info,
        )
        print(f"Generated {generated}/{args.count}")

    generated = fresh_completed_count(args)
    if generated != args.count:
        raise SystemExit(
            f"Generated {generated}/{args.count} responses with {failures} failed attempts. "
            "Run the same command again to continue."
        )
    print(f"Done. The dataset contains {generated} responses for this model.")


def build_parser(provider, mode):
    description = (
        "Repair existing AI question rows by generating their full responses."
        if mode == "repair"
        else "Generate complete AI text samples from human-matched prompts and responses."
    )
    parser = argparse.ArgumentParser(description=description)
    parser.add_argument("--model", required=provider != "gemini")
    parser.add_argument("--meta", type=Path, default=DEFAULT_META)
    parser.add_argument("--count", type=int, default=None if mode == "repair" else DEFAULT_COUNT)
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument("--timeout", type=float, default=DEFAULT_TIMEOUT)
    parser.add_argument("--max-retries", type=int, default=4)
    parser.add_argument("--length-tolerance", type=float, default=DEFAULT_TOLERANCE)
    if mode == "fresh":
        parser.add_argument("--output-folder", type=Path, default=DEFAULT_OUTPUT_FOLDER)
    else:
        parser.add_argument(
            "--workers",
            type=int,
            default=1,
            help="Number of response requests to run concurrently.",
        )
    if provider == "openai" and mode == "repair":
        parser.add_argument(
            "--source-model",
            help=(
                "Original OpenAI model allocation to repair. Defaults to --model."
            ),
        )

    if provider == "ollama":
        parser.add_argument("--ollama-url", default=DEFAULT_API_BASES["ollama"])
        parser.add_argument("--temperature", type=float, default=0.9)
        parser.add_argument("--top-p", type=float, default=0.95)
        parser.add_argument("--keep-alive", default="30m")
    else:
        parser.add_argument("--api-base", default=DEFAULT_API_BASES[provider])
        parser.add_argument("--api-key-env", default=DEFAULT_KEY_ENVS[provider])
    if provider == "openrouter":
        parser.add_argument("--temperature", type=float, default=0.9)
    if provider == "gemini":
        parser.set_defaults(model="gemini-3.6-flash")
        parser.add_argument(
            "--thinking-level",
            choices=("minimal", "low", "medium", "high"),
            default="minimal",
        )
    return parser


def validate_args(args, provider, mode):
    args.provider = provider
    args.model = args.model.strip()
    if not args.model:
        raise SystemExit("--model must not be empty")
    if provider == "openai":
        if args.model not in OPENAI_MODELS:
            choices = ", ".join(OPENAI_MODELS)
            raise SystemExit(f"--model must be one of: {choices}")
        args.model_selection = args.model
        args.model, args.reasoning_effort = OPENAI_MODELS[args.model_selection]
        if mode == "repair":
            source_selection = (args.source_model or args.model_selection).strip()
            if source_selection not in OPENAI_MODELS:
                choices = ", ".join(OPENAI_MODELS)
                raise SystemExit(f"--source-model must be one of: {choices}")
            args.source_model_selection = source_selection
            (
                args.source_model,
                args.source_reasoning_effort,
            ) = OPENAI_MODELS[source_selection]
    if provider == "ollama":
        args.api_base = args.ollama_url
        args.api_key = None
    else:
        args.api_key = os.environ.get(args.api_key_env)
        if not args.api_key:
            raise SystemExit(f"Set {args.api_key_env} before running this script")
    if args.count is not None and args.count <= 0:
        raise SystemExit("--count must be positive")
    if args.timeout <= 0:
        raise SystemExit("--timeout must be positive")
    if args.max_retries <= 0:
        raise SystemExit("--max-retries must be positive")
    if mode == "repair" and args.workers <= 0:
        raise SystemExit("--workers must be positive")
    if not 0 < args.length_tolerance < 1:
        raise SystemExit("--length-tolerance must be between 0 and 1")
    if hasattr(args, "temperature") and not 0 <= args.temperature <= 2:
        raise SystemExit("--temperature must be between 0 and 2")
    if hasattr(args, "top_p") and not 0 < args.top_p <= 1:
        raise SystemExit("--top-p must be in (0, 1]")
    args.meta = args.meta.resolve()
    if mode == "fresh":
        args.output_folder = args.output_folder.resolve()


def main(provider, mode):
    parser = build_parser(provider, mode)
    args = parser.parse_args()
    validate_args(args, provider, mode)
    print(f"Using {provider} model {args.model}")
    if provider == "openai":
        print(f"Model selection: {args.model_selection}, reasoning: {args.reasoning_effort}")
        if mode == "repair" and args.source_model_selection != args.model_selection:
            print(
                f"Repair source: {args.source_model_selection}, "
                f"reasoning: {args.source_reasoning_effort}"
            )
    if mode == "repair":
        print(f"Requested workers: {args.workers}")
    with exclusive_file_lock(job_lock_path(args), wait=False):
        if mode == "repair":
            repair_dataset(args)
        else:
            fresh_dataset(args)
