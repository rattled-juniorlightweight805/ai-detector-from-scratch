#!/usr/bin/env python3
"""Evaluate a reproducible random sample with Pangram's bulk API."""

import argparse
import csv
import hashlib
import json
import math
import os
import time
import urllib.error
import urllib.request
from collections import Counter
from pathlib import Path


BASE_URL = "https://text.external-api.pangram.com"
SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_DIR = SCRIPT_DIR.parents[2]
DEFAULT_META = PROJECT_DIR / "data" / "meta.json"
DEFAULT_ENV_FILE = Path.home() / ".env.pangram"
TERMINAL_STATUSES = {"succeeded", "failed", "partial"}


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--count-per-class", type=int, default=50)
    parser.add_argument("--seed", type=int, default=20260804)
    parser.add_argument("--model", default="default")
    parser.add_argument("--meta", type=Path, default=DEFAULT_META)
    parser.add_argument("--output-dir", type=Path, default=SCRIPT_DIR)
    parser.add_argument("--bulk-unit-price", type=float, default=0.04)
    parser.add_argument("--submit", action="store_true")
    return parser.parse_args()


def read_env_file(path, name):
    if not path.is_file():
        return None
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.startswith(f"{name}="):
            return line.split("=", 1)[1].strip().strip("\"").strip("'")
    return None


def get_api_key():
    key = os.environ.get("PANGRAM_API_KEY") or read_env_file(
        DEFAULT_ENV_FILE, "PANGRAM_API_KEY"
    )
    if not key:
        raise SystemExit(
            "Set PANGRAM_API_KEY or store it as PANGRAM_API_KEY=... in ~/.env.pangram"
        )
    return key


def request_json(
    url,
    api_key,
    *,
    method = "GET",
    payload = None,
):
    body = None
    headers = {"x-api-key": api_key}
    if payload is not None:
        body = json.dumps(payload).encode("utf-8")
        headers["Content-Type"] = "application/json"
    request = urllib.request.Request(url, data=body, headers=headers, method=method)
    try:
        with urllib.request.urlopen(request, timeout=120) as response:
            return json.loads(response.read().decode("utf-8"))
    except urllib.error.HTTPError as error:
        details = error.read().decode("utf-8", errors="replace")
        raise SystemExit(f"Pangram returned HTTP {error.code}: {details}") from error


def stable_rank(seed, label, sample_id):
    value = f"{seed}:{label}:{sample_id}".encode("utf-8")
    return hashlib.sha256(value).hexdigest()


def select_samples(
    meta_path, count_per_class, seed
):
    metadata = json.loads(meta_path.read_text(encoding="utf-8"))
    data_dir = meta_path.parent
    selected = []
    for label in ("human", "ai"):
        candidates = [
            sample
            for sample in metadata["samples"]
            if sample.get("label") == label
            and (data_dir / str(sample.get("file"))).is_file()
        ]
        candidates.sort(
            key=lambda sample: stable_rank(seed, label, int(sample["id"]))
        )
        if len(candidates) < count_per_class:
            raise SystemExit(
                f"Requested {count_per_class} {label} samples but found {len(candidates)}"
            )
        for sample in candidates[:count_per_class]:
            path = data_dir / str(sample["file"])
            text = path.read_text(encoding="utf-8").strip()
            word_count = len(text.split())
            selected.append(
                {
                    "request_id": f"{label}-{sample['id']}",
                    "sample_id": int(sample["id"]),
                    "true_label": label,
                    "file": str(sample["file"]),
                    "word_count": word_count,
                    "pangram_3_billable_units": max(1, math.ceil(word_count / 1000)),
                    "text": text,
                }
            )
    return selected


def public_manifest(samples):
    return [{key: value for key, value in sample.items() if key != "text"} for sample in samples]


def manifest_digest(manifest):
    encoded = json.dumps(manifest, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(encoded).hexdigest()


def compact_result(item, sample):
    result = item["result"]
    windows = result.get("windows") or []
    scores = [
        float(window["ai_assistance_score"])
        for window in windows
        if window.get("ai_assistance_score") is not None
    ]
    return {
        "request_id": sample["request_id"],
        "sample_id": sample["sample_id"],
        "true_label": sample["true_label"],
        "file": sample["file"],
        "word_count": sample["word_count"],
        "version": result.get("version"),
        "prediction_short": result.get("prediction_short"),
        "headline": result.get("headline"),
        "fraction_ai": result.get("fraction_ai"),
        "fraction_ai_assisted": result.get("fraction_ai_assisted"),
        "fraction_human": result.get("fraction_human"),
        "num_ai_segments": result.get("num_ai_segments"),
        "num_ai_assisted_segments": result.get("num_ai_assisted_segments"),
        "num_human_segments": result.get("num_human_segments"),
        "num_windows": len(windows),
        "mean_window_ai_assistance_score": sum(scores) / len(scores) if scores else None,
        "max_window_ai_assistance_score": max(scores) if scores else None,
    }


def build_summary(
    *,
    args,
    bulk_id,
    manifest,
    results,
):
    cross_tab = Counter(
        (row["true_label"], str(row["prediction_short"])) for row in results
    )
    other_predictions = Counter(
        (row["true_label"], str(row["prediction_short"]))
        for row in results
        if row["prediction_short"] not in {"AI", "Human"}
    )
    units = sum(int(row["pangram_3_billable_units"]) for row in manifest)
    return {
        "model_selector": args.model,
        "returned_versions": sorted({str(row["version"]) for row in results}),
        "bulk_id": bulk_id,
        "manifest_sha256": manifest_digest(manifest),
        "seed": args.seed,
        "count_per_class": args.count_per_class,
        "total_samples": len(results),
        "total_words": sum(int(row["word_count"]) for row in manifest),
        "billable_units": units,
        "estimated_bulk_cost_usd": round(units * args.bulk_unit_price, 2),
        "prediction_counts": {
            label: dict(
                Counter(
                    str(row["prediction_short"])
                    for row in results
                    if row["true_label"] == label
                )
            )
            for label in ("human", "ai")
        },
        "binary_matrix_for_exact_ai_human_labels": {
            "tp": cross_tab[("ai", "AI")],
            "fp": cross_tab[("human", "AI")],
            "tn": cross_tab[("human", "Human")],
            "fn": cross_tab[("ai", "Human")],
            "unresolved": [
                {"true_label": key[0], "prediction": key[1], "count": count}
                for key, count in sorted(other_predictions.items())
            ],
        },
    }


def write_json(path, value):
    path.write_text(json.dumps(value, indent=2) + "\n", encoding="utf-8")


def main():
    args = parse_args()
    if args.count_per_class <= 0:
        raise SystemExit("--count-per-class must be positive")
    args.output_dir.mkdir(parents=True, exist_ok=True)

    samples = select_samples(args.meta.resolve(), args.count_per_class, args.seed)
    manifest = public_manifest(samples)
    digest = manifest_digest(manifest)
    units = sum(int(row["pangram_3_billable_units"]) for row in manifest)
    estimated_cost = units * args.bulk_unit_price
    write_json(args.output_dir / "sample-manifest.json", manifest)
    print(
        f"Selected {len(samples)} texts with {sum(row['word_count'] for row in manifest):,} "
        f"words and {units} Pangram 3 bulk units."
    )
    print(f"Estimated cost: ${estimated_cost:.2f}")
    if not args.submit:
        print("Dry run only. Add --submit to call Pangram.")
        return

    summary_path = args.output_dir / "summary.json"
    if summary_path.is_file():
        completed = json.loads(summary_path.read_text(encoding="utf-8"))
        if (
            completed.get("manifest_sha256") == digest
            and completed.get("total_samples") == len(samples)
        ):
            print("The selected sample already has complete saved results.")
            print(json.dumps(completed["prediction_counts"], indent=2))
            return

    api_key = get_api_key()
    models = request_json(f"{BASE_URL}/models", api_key).get("models", [])
    if args.model not in models:
        raise SystemExit(
            f"Model selector {args.model!r} is unavailable. Available models: {models}"
        )

    state_path = args.output_dir / "bulk-job.json"
    if state_path.is_file():
        state = json.loads(state_path.read_text(encoding="utf-8"))
        if state.get("manifest_sha256") != digest or state.get("model") != args.model:
            raise SystemExit("Existing bulk-job.json belongs to another sample or model")
        bulk_id = str(state["bulk_id"])
        print(f"Resuming bulk job {bulk_id}")
    else:
        submitted = request_json(
            f"{BASE_URL}/bulk",
            api_key,
            method="POST",
            payload={
                "items": [
                    {"id": sample["request_id"], "text": sample["text"]}
                    for sample in samples
                ],
                "model": args.model,
            },
        )
        bulk_id = str(submitted["bulk_id"])
        write_json(
            state_path,
            {
                "bulk_id": bulk_id,
                "model": args.model,
                "manifest_sha256": digest,
                "total_items": len(samples),
            },
        )
        print(f"Submitted bulk job {bulk_id}")

    while True:
        status = request_json(f"{BASE_URL}/bulk/{bulk_id}", api_key)
        print(
            f"Status {status.get('status')}: succeeded={status.get('succeeded', 0)}, "
            f"failed={status.get('failed', 0)}",
            flush=True,
        )
        if status.get("status") in TERMINAL_STATUSES:
            break
        time.sleep(3)

    page = request_json(
        f"{BASE_URL}/bulk/{bulk_id}/results?offset=0&limit=1000", api_key
    )
    failed_items = page.get("failed_items") or []
    by_request_id = {sample["request_id"]: sample for sample in samples}
    compact = [
        compact_result(item, by_request_id[str(item["id"])])
        for item in page.get("items", [])
        if item.get("result") is not None
    ]
    compact.sort(key=lambda row: (row["true_label"], row["sample_id"]))
    write_json(args.output_dir / "results.json", compact)
    if compact:
        with (args.output_dir / "results.csv").open(
            "w", encoding="utf-8", newline=""
        ) as handle:
            writer = csv.DictWriter(handle, fieldnames=list(compact[0]))
            writer.writeheader()
            writer.writerows(compact)

    if len(compact) != len(samples) or failed_items:
        write_json(args.output_dir / "failed-items.json", failed_items)
        raise SystemExit(
            f"Expected {len(samples)} results, received {len(compact)} with "
            f"{len(failed_items)} failures"
        )
    summary = build_summary(
        args=args,
        bulk_id=bulk_id,
        manifest=manifest,
        results=compact,
    )
    write_json(summary_path, summary)
    print(json.dumps(summary["prediction_counts"], indent=2))
    print(json.dumps(summary["binary_matrix_for_exact_ai_human_labels"], indent=2))


if __name__ == "__main__":
    main()
