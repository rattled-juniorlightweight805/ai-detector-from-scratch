#!/usr/bin/env python3
"""Score Substack notes with every local classifier and plot confusion matrices."""

import argparse
import csv
import gc
from pathlib import Path


SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_DIR = SCRIPT_DIR.parents[1]

from ai_detector import (
    artifact_status,
    load_classifier,
    model_registry,
    score_payload,
)


MODEL_NAMES = (
    "logreg",
    "distilbert",
    "distilbert-lora",
    "distilbert-mica",
    "modernbert",
    "gpt2-variable",
    "gpt2-fixed",
    "qwen3-variable",
    "qwen3-fixed",
)
REQUIRED_COLUMNS = ("Text", "URL", "Human percent", "AI percent")


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Score every row in new-substack-predictions.csv with all nine "
            "local classifiers and create one confusion matrix per model."
        )
    )
    parser.add_argument(
        "--input",
        type=Path,
        default=SCRIPT_DIR / "new-substack-predictions.csv",
        help="Input CSV (default: next to this script)",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=SCRIPT_DIR / "new-substack-model-predictions.csv",
        help="Scored CSV output (default: next to this script)",
    )
    parser.add_argument(
        "--figures-dir",
        type=Path,
        default=SCRIPT_DIR / "confusion-matrices",
        help="Directory for the nine confusion-matrix SVG files",
    )
    parser.add_argument(
        "--device",
        choices=("auto", "cpu", "cuda", "mps"),
        default="auto",
        help="Torch inference device (default: auto)",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=8,
        help="Number of texts per classifier call (default: 8)",
    )
    parser.add_argument(
        "--threshold",
        type=float,
        default=50.0,
        help=(
            "AI-percentage threshold for reference and model labels "
            "(default: 50)"
        ),
    )
    args = parser.parse_args()
    if args.batch_size < 1:
        parser.error("--batch-size must be at least 1")
    if not 0.0 <= args.threshold <= 100.0:
        parser.error("--threshold must be between 0 and 100")
    return args


def parse_percentage(value, *, column, row_number):
    try:
        percentage = float(value.strip().removesuffix("%"))
    except ValueError as error:
        raise ValueError(
            f"Row {row_number}: {column!r} is not a percentage: {value!r}"
        ) from error
    if not 0.0 <= percentage <= 100.0:
        raise ValueError(
            f"Row {row_number}: {column!r} must be between 0 and 100"
        )
    return percentage


def read_rows(path):
    if not path.is_file():
        raise FileNotFoundError(f"Input CSV not found: {path}")
    with path.open(encoding="utf-8-sig", newline="") as file:
        reader = csv.DictReader(file)
        missing = [
            column
            for column in REQUIRED_COLUMNS
            if column not in (reader.fieldnames or [])
        ]
        if missing:
            raise ValueError(
                f"Input CSV is missing columns: {', '.join(missing)}"
            )
        rows = list(reader)
    if not rows:
        raise ValueError("Input CSV contains no data rows")
    for row_number, row in enumerate(rows, start=2):
        if not row["Text"].strip():
            raise ValueError(f"Row {row_number}: 'Text' cannot be empty")
        parse_percentage(
            row["AI percent"], column="AI percent", row_number=row_number
        )
    return rows


def ensure_models_ready():
    registry = model_registry()
    failures = []
    for model_name in MODEL_NAMES:
        ready, status = artifact_status(registry[model_name])
        if not ready:
            failures.append(f"{model_name}: {status}")
    if failures:
        raise RuntimeError("Models are not ready:\n" + "\n".join(failures))


def release_device_cache():
    gc.collect()
    try:
        import torch

        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        if torch.backends.mps.is_available():
            torch.mps.empty_cache()
    except ImportError:
        pass


def score_models(
    texts, *, device, batch_size
):
    predictions = {}
    for model_name in MODEL_NAMES:
        print(f"Scoring {len(texts)} notes with {model_name}...", flush=True)
        classifier = load_classifier(model_name, device=device)
        try:
            probabilities = classifier.score_many(
                texts, batch_size=batch_size
            )
        finally:
            del classifier
            release_device_cache()
        if len(probabilities) != len(texts):
            raise RuntimeError(
                f"{model_name} returned {len(probabilities)} predictions "
                f"for {len(texts)} texts"
            )
        predictions[model_name] = [
            score_payload(probability)["score"] for probability in probabilities
        ]
    return predictions


def write_predictions(
    path,
    rows,
    predictions,
    *,
    threshold,
):
    fieldnames = [
        *REQUIRED_COLUMNS,
        "Reference label",
        *(
            column
            for model_name in MODEL_NAMES
            for column in (
                f"{model_name} AI percent",
                f"{model_name} prediction",
            )
        ),
    ]
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = path.with_suffix(f"{path.suffix}.tmp")
    with temporary_path.open("w", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        for index, source_row in enumerate(rows):
            output_row = {
                column: source_row[column] for column in REQUIRED_COLUMNS
            }
            reference_score = parse_percentage(
                source_row["AI percent"],
                column="AI percent",
                row_number=index + 2,
            )
            output_row["Reference label"] = (
                "AI" if reference_score >= threshold else "Human"
            )
            for model_name in MODEL_NAMES:
                score = predictions[model_name][index]
                output_row[f"{model_name} AI percent"] = score
                output_row[f"{model_name} prediction"] = (
                    "AI" if score >= threshold else "Human"
                )
            writer.writerow(output_row)
    temporary_path.replace(path)


def save_confusion_matrices(
    figures_dir,
    rows,
    predictions,
    *,
    threshold,
):
    import matplotlib.pyplot as plt
    from sklearn.metrics import ConfusionMatrixDisplay, confusion_matrix

    reference_labels = [
        int(
            parse_percentage(
                row["AI percent"],
                column="AI percent",
                row_number=index + 2,
            )
            >= threshold
        )
        for index, row in enumerate(rows)
    ]
    figures_dir.mkdir(parents=True, exist_ok=True)
    for model_name in MODEL_NAMES:
        predicted_labels = [
            int(score >= threshold) for score in predictions[model_name]
        ]
        matrix = confusion_matrix(
            reference_labels, predicted_labels, labels=[0, 1]
        )
        figure, axis = plt.subplots(figsize=(4.4, 4.0))
        display = ConfusionMatrixDisplay(
            confusion_matrix=matrix,
            display_labels=("Human", "AI"),
        )
        display.plot(
            ax=axis,
            cmap="Blues",
            colorbar=False,
            values_format="d",
        )
        axis.set_ylabel("Pangram label")
        axis.set_title(model_name)
        figure.tight_layout()
        output_path = figures_dir / f"{model_name}.svg"
        figure.savefig(output_path, format="svg", bbox_inches="tight")
        plt.close(figure)
        correct = sum(
            expected == predicted
            for expected, predicted in zip(
                reference_labels, predicted_labels, strict=True
            )
        )
        print(
            f"Saved {output_path} "
            f"({correct / len(reference_labels):.1%} agreement)"
        )


def main():
    args = parse_args()
    rows = read_rows(args.input.resolve())
    ensure_models_ready()
    predictions = score_models(
        [row["Text"] for row in rows],
        device=args.device,
        batch_size=args.batch_size,
    )
    write_predictions(
        args.output.resolve(),
        rows,
        predictions,
        threshold=args.threshold,
    )
    save_confusion_matrices(
        args.figures_dir.resolve(),
        rows,
        predictions,
        threshold=args.threshold,
    )
    print(f"Saved predictions to {args.output.resolve()}")


if __name__ == "__main__":
    main()
