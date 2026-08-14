#!/usr/bin/env python3
"""Benchmark calibrated scikit-learn inference on the complete test split."""

import argparse
import statistics
import time
from pathlib import Path

import numpy as np
from datasets import load_from_disk
from joblib import load


SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_DIR = SCRIPT_DIR.parents[1]
DEFAULT_MODEL_PATH = SCRIPT_DIR / "artifacts" / "logreg-ai-detector.joblib"
DEFAULT_DATASET_PATH = PROJECT_DIR / "data" / "hf-dataset"


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Benchmark predict_proba on every sample in the local test split. "
            "Model and dataset loading are excluded from the timing."
        )
    )
    parser.add_argument(
        "--model-path",
        type=Path,
        default=DEFAULT_MODEL_PATH,
        help=f"Saved joblib model (default: {DEFAULT_MODEL_PATH})",
    )
    parser.add_argument(
        "--dataset-path",
        type=Path,
        default=DEFAULT_DATASET_PATH,
        help=f"Local DatasetDict directory (default: {DEFAULT_DATASET_PATH})",
    )
    parser.add_argument(
        "--repeats",
        type=int,
        default=5,
        help="Number of measured full-test prediction passes (default: 5)",
    )
    args = parser.parse_args()
    if args.repeats < 1:
        parser.error("--repeats must be at least 1")
    return args


def benchmark_predictions(
    model,
    texts,
    repeats,
):
    # Warm up vectorization and prediction before collecting measurements.
    model.predict_proba(texts)

    elapsed_times = []
    probabilities = None
    for repeat in range(1, repeats + 1):
        start_time = time.perf_counter()
        probabilities = model.predict_proba(texts)
        elapsed = time.perf_counter() - start_time
        elapsed_times.append(elapsed)
        print(f"Run {repeat}/{repeats}: {elapsed:.4f} seconds")

    assert probabilities is not None
    return elapsed_times, probabilities


def main():
    args = parse_args()

    if not args.model_path.is_file():
        raise FileNotFoundError(f"Model not found: {args.model_path}")
    if not args.dataset_path.is_dir():
        raise FileNotFoundError(f"Dataset not found: {args.dataset_path}")

    # Loading and conversion to Python and NumPy objects happen before timing.
    model = load(args.model_path)
    dataset = load_from_disk(str(args.dataset_path))
    test_dataset = dataset["test"]
    test_texts = list(test_dataset["text"])
    test_labels = np.asarray(test_dataset["label"], dtype=np.int64)

    print(f"Model: {args.model_path}")
    print(f"Dataset: {args.dataset_path}")
    print(f"Test samples: {len(test_texts):,}")
    print("Running one untimed warm-up pass...")

    elapsed_times, probabilities = benchmark_predictions(
        model,
        test_texts,
        args.repeats,
    )

    mean_seconds = statistics.fmean(elapsed_times)
    std_seconds = (
        statistics.stdev(elapsed_times) if len(elapsed_times) > 1 else 0.0
    )
    mean_ms_per_sample = mean_seconds / len(test_texts) * 1_000
    std_ms_per_sample = std_seconds / len(test_texts) * 1_000
    throughput = len(test_texts) / mean_seconds

    predictions = (probabilities[:, 1] >= 0.5).astype(np.int64)
    accuracy = float(np.mean(predictions == test_labels))

    print(f"\nPrediction time for all {len(test_texts):,} test samples")
    print(f"Mean: {mean_seconds:.4f} seconds")
    print(f"Sample standard deviation: {std_seconds:.4f} seconds")
    print(
        "Mean time per sample: "
        f"{mean_ms_per_sample:.4f} +/- {std_ms_per_sample:.4f} ms"
    )
    print(f"Throughput: {throughput:,.1f} texts/second")
    print(f"Test accuracy: {accuracy:.2%}")


if __name__ == "__main__":
    main()
