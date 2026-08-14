#!/usr/bin/env python3
"""Score text with one of the locally trained AI-text classifiers."""

import argparse
import json
import sys
from pathlib import Path
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

from ai_detector import (
    ArtifactNotReadyError,
    artifact_status,
    load_classifier,
    model_registry,
    score_payload,
    validate_text,
)


def parse_args():
    registry = model_registry()
    parser = argparse.ArgumentParser(
        description=(
            "Return a 0-100 AI-generated probability score as JSON. "
            "Text can be provided with --text, --file, or standard input."
        )
    )
    parser.add_argument(
        "--model",
        choices=sorted(registry),
        help="Classifier to use",
    )
    parser.add_argument(
        "--artifact",
        type=Path,
        help="Override the model's default exported artifact path",
    )
    input_group = parser.add_mutually_exclusive_group()
    input_group.add_argument("--text", help="Text to classify")
    input_group.add_argument(
        "--file", type=Path, help="UTF-8 text file to classify"
    )
    parser.add_argument(
        "--device",
        choices=("auto", "cpu", "cuda", "mps"),
        default="auto",
        help="Torch inference device (default: auto)",
    )
    parser.add_argument(
        "--server",
        help=(
            "Use a running classifier server, for example "
            "http://127.0.0.1:8000"
        ),
    )
    parser.add_argument(
        "--timeout",
        type=float,
        default=60.0,
        help="Server request timeout in seconds (default: 60)",
    )
    parser.add_argument(
        "--pretty", action="store_true", help="Indent the JSON output"
    )
    parser.add_argument(
        "--list-models",
        action="store_true",
        help="List model names, export paths, and readiness",
    )
    args = parser.parse_args()

    if not args.list_models and args.model is None and args.server is None:
        parser.error(
            "--model is required unless --server or --list-models is used"
        )
    if args.server is not None and args.model is not None:
        parser.error(
            "--model cannot be used with --server; the server selects the model"
        )
    if args.server is not None and args.artifact is not None:
        parser.error("--artifact cannot be used with --server")
    if args.server is not None and args.device != "auto":
        parser.error("--device cannot be used with --server")
    if args.timeout <= 0:
        parser.error("--timeout must be positive")
    return args


def list_models():
    for spec in model_registry().values():
        ready, status = artifact_status(spec)
        marker = "ready" if ready else status
        print(f"{spec.name:20} {marker}")
        print(f"  {spec.description}")
        print(f"  {spec.artifact_path}")


def read_input_text(args):
    if args.text is not None:
        return validate_text(args.text)
    if args.file is not None:
        if not args.file.is_file():
            raise FileNotFoundError(f"Text file not found: {args.file}")
        return validate_text(args.file.read_text(encoding="utf-8"))
    if sys.stdin.isatty():
        raise ValueError(
            "Provide text with --text, --file, or piped standard input"
        )
    return validate_text(sys.stdin.read())


def request_server_score(
    server_url,
    text,
    *,
    timeout,
):
    base_url = server_url.rstrip("/")
    endpoint = (
        base_url
        if base_url.endswith("/api/check")
        else f"{base_url}/api/check"
    )
    request = Request(
        endpoint,
        data=json.dumps({"text": validate_text(text)}).encode("utf-8"),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    try:
        with urlopen(request, timeout=timeout) as response:
            response_body = response.read().decode("utf-8")
    except HTTPError as error:
        error_body = error.read().decode("utf-8", errors="replace")
        try:
            detail = json.loads(error_body).get("detail", error_body)
        except json.JSONDecodeError:
            detail = error_body
        raise RuntimeError(
            f"Classifier server returned HTTP {error.code}: {detail}"
        ) from error
    except URLError as error:
        raise RuntimeError(
            f"Could not connect to classifier server at {base_url}: "
            f"{error.reason}"
        ) from error

    try:
        payload = json.loads(response_body)
        score = float(payload["score"])
    except (json.JSONDecodeError, KeyError, TypeError, ValueError) as error:
        raise RuntimeError(
            "Classifier server returned an invalid score payload"
        ) from error
    return score_payload(score / 100.0)


def main():
    args = parse_args()
    if args.list_models:
        list_models()
        return

    try:
        text = read_input_text(args)
        if args.server is not None:
            payload = request_server_score(
                args.server,
                text,
                timeout=args.timeout,
            )
        else:
            classifier = load_classifier(
                args.model,
                artifact_path=args.artifact,
                device=args.device,
            )
            ai_probability = classifier.score_many([text], batch_size=1)[0]
            payload = score_payload(ai_probability)
    except (
        ArtifactNotReadyError,
        FileNotFoundError,
        RuntimeError,
        ValueError,
    ) as error:
        raise SystemExit(f"error: {error}") from error

    indent = 2 if args.pretty else None
    print(json.dumps(payload, indent=indent))


if __name__ == "__main__":
    main()
