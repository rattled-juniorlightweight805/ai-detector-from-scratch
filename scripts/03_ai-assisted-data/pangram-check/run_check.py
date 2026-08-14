#!/usr/bin/env python3
"""Evaluate the current AI-assisted dataset snapshot with Pangram 3."""

import csv
import hashlib
import importlib.util
import json
import math
import time
from pathlib import Path


SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_DIR = SCRIPT_DIR.parents[2]
META_PATH = PROJECT_DIR / "data" / "meta.json"
SNAPSHOT_PATH = SCRIPT_DIR / "sample-manifest.json"
MODEL = "default"
TERMINAL_STATUSES = {"succeeded", "failed", "partial"}

SHARED_PATH = PROJECT_DIR / "scripts" / "02_ai-data" / "pangram-check" / "run_check.py"
SPEC = importlib.util.spec_from_file_location("shared_pangram_check", SHARED_PATH)
assert SPEC is not None and SPEC.loader is not None
SHARED = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(SHARED)


def write_json(path, value):
    path.write_text(json.dumps(value, indent=2) + "\n", encoding="utf-8")


def manifest_digest(manifest):
    encoded = json.dumps(manifest, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(encoded).hexdigest()


def create_or_load_snapshot():
    if SNAPSHOT_PATH.is_file():
        return json.loads(SNAPSHOT_PATH.read_text(encoding="utf-8"))

    metadata = json.loads(META_PATH.read_text(encoding="utf-8"))
    rows = []
    for sample in metadata["samples"]:
        relative_file = str(sample.get("file"))
        path = META_PATH.parent / relative_file
        if sample.get("label") != "ai-assisted" or not path.is_file():
            continue
        generator = sample.get("generator") or {}
        words = int(sample.get("word_count") or len(path.read_text(encoding="utf-8").split()))
        rows.append(
            {
                "request_id": f"ai-assisted-{sample['id']}",
                "sample_id": int(sample["id"]),
                "file": relative_file,
                "edit_level": sample.get("edit_level"),
                "source_sample_id": sample.get("seed_sample_id"),
                "provider": generator.get("provider"),
                "model": generator.get("model_selection")
                or generator.get("requested_model")
                or generator.get("model"),
                "word_count": words,
                "pangram_3_billable_units": max(1, math.ceil(words / 1000)),
            }
        )
    rows.sort(key=lambda row: row["sample_id"])
    if not rows:
        raise SystemExit("No completed AI-assisted samples were found")
    write_json(SNAPSHOT_PATH, rows)
    return rows


def compact_result(item, sample):
    result = item["result"]
    windows = result.get("windows") or []
    window_scores = [
        float(window["ai_assistance_score"])
        for window in windows
        if window.get("ai_assistance_score") is not None
    ]
    text = (META_PATH.parent / sample["file"]).read_text(encoding="utf-8").strip()

    def percent(key):
        value = result.get(key)
        return round(float(value) * 100, 3) if value is not None else None

    return {
        "sample_id": sample["sample_id"],
        "edit_level": sample["edit_level"],
        "provider": sample["provider"],
        "model": sample["model"],
        "source_sample_id": sample["source_sample_id"],
        "word_count": sample["word_count"],
        "text": text,
        "pangram_version": result.get("version"),
        "prediction": result.get("prediction_short"),
        "headline": result.get("headline"),
        "ai_fraction_percent": percent("fraction_ai"),
        "ai_assisted_fraction_percent": percent("fraction_ai_assisted"),
        "human_fraction_percent": percent("fraction_human"),
        "mean_window_ai_assistance_score_percent": (
            round(100 * sum(window_scores) / len(window_scores), 3)
            if window_scores
            else None
        ),
        "max_window_ai_assistance_score_percent": (
            round(100 * max(window_scores), 3) if window_scores else None
        ),
    }


def main():
    samples = create_or_load_snapshot()
    digest = manifest_digest(samples)
    units = sum(int(sample["pangram_3_billable_units"]) for sample in samples)
    print(f"Snapshot: {len(samples)} texts, {units} Pangram 3 bulk units")
    print(f"Estimated cost: ${units * 0.04:.2f}")

    api_key = SHARED.get_api_key()
    models = SHARED.request_json(f"{SHARED.BASE_URL}/models", api_key).get("models", [])
    if MODEL not in models:
        raise SystemExit(f"Model selector {MODEL!r} is unavailable: {models}")

    state_path = SCRIPT_DIR / "bulk-job.json"
    if state_path.is_file():
        state = json.loads(state_path.read_text(encoding="utf-8"))
        if state.get("manifest_sha256") != digest:
            raise SystemExit("bulk-job.json belongs to another snapshot")
        bulk_id = str(state["bulk_id"])
        print(f"Resuming bulk job {bulk_id}")
    else:
        submitted = SHARED.request_json(
            f"{SHARED.BASE_URL}/bulk",
            api_key,
            method="POST",
            payload={
                "items": [
                    {
                        "id": sample["request_id"],
                        "text": (META_PATH.parent / sample["file"])
                        .read_text(encoding="utf-8")
                        .strip(),
                    }
                    for sample in samples
                ],
                "model": MODEL,
            },
        )
        bulk_id = str(submitted["bulk_id"])
        write_json(
            state_path,
            {
                "bulk_id": bulk_id,
                "model": MODEL,
                "manifest_sha256": digest,
                "total_items": len(samples),
            },
        )
        print(f"Submitted bulk job {bulk_id}")

    while True:
        status = SHARED.request_json(f"{SHARED.BASE_URL}/bulk/{bulk_id}", api_key)
        print(
            f"Status {status.get('status')}: succeeded={status.get('succeeded', 0)}, "
            f"failed={status.get('failed', 0)}",
            flush=True,
        )
        if status.get("status") in TERMINAL_STATUSES:
            break
        time.sleep(3)

    page = SHARED.request_json(
        f"{SHARED.BASE_URL}/bulk/{bulk_id}/results?offset=0&limit=1000", api_key
    )
    by_request_id = {sample["request_id"]: sample for sample in samples}
    rows = [
        compact_result(item, by_request_id[str(item["id"])])
        for item in page.get("items", [])
        if item.get("result") is not None
    ]
    rows.sort(key=lambda row: row["sample_id"])
    write_json(SCRIPT_DIR / "results.json", rows)
    if rows:
        with (SCRIPT_DIR / "results.csv").open("w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)
    failures = page.get("failed_items") or []
    if len(rows) != len(samples) or failures:
        write_json(SCRIPT_DIR / "failed-items.json", failures)
        raise SystemExit(
            f"Expected {len(samples)} results, received {len(rows)} with "
            f"{len(failures)} failures"
        )
    print(f"Saved {len(rows)} result rows to {SCRIPT_DIR / 'results.csv'}")


if __name__ == "__main__":
    main()
