#!/usr/bin/env python3
"""Render the saved Pangram results as an mlxtend-style confusion matrix."""

import argparse
import csv
import html
from collections import Counter
from pathlib import Path


SCRIPT_DIR = Path(__file__).resolve().parent
DEFAULT_RESULTS = SCRIPT_DIR / "results.csv"
DEFAULT_OUTPUT = SCRIPT_DIR / "confusion-matrix.svg"


def svg_text(
    x,
    y,
    value,
    *,
    size,
    fill = "#222222",
    weight = 400,
    anchor = "middle",
    rotate = None,
):
    transform = f' transform="rotate({rotate} {x} {y})"' if rotate else ""
    return (
        f'<text x="{x}" y="{y}" text-anchor="{anchor}"{transform} '
        'font-family="Helvetica, Arial, sans-serif" '
        f'font-size="{size}" font-weight="{weight}" fill="{fill}">'
        f"{html.escape(value)}</text>"
    )


def binary_prediction(prediction):
    if prediction == "Mixed":
        return "AI"
    if prediction in {"AI", "Human"}:
        return prediction
    raise SystemExit(f"Unexpected Pangram prediction: {prediction!r}")


def read_matrix(path):
    with path.open(encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    if not rows:
        raise SystemExit(f"No rows found in {path}")
    counts = Counter(
        (row["true_label"], binary_prediction(row["prediction_short"]))
        for row in rows
    )
    return {
        ("AI", "AI"): counts[("ai", "AI")],
        ("AI", "Human"): counts[("ai", "Human")],
        ("Human", "AI"): counts[("human", "AI")],
        ("Human", "Human"): counts[("human", "Human")],
    }


def cell(
    x,
    y,
    *,
    fill,
    count,
    label,
    text_fill,
):
    return "\n".join(
        [
            f'<rect x="{x}" y="{y}" width="260" height="260" fill="{fill}" />',
            svg_text(
                x + 130,
                y + 117,
                str(count),
                size=44,
                fill=text_fill,
                weight=500,
            ),
            svg_text(
                x + 130,
                y + 161,
                label,
                size=17,
                fill=text_fill,
                weight=500,
            ),
        ]
    )


def render(matrix):
    left = 165
    top = 75
    size = 520
    return "\n".join(
        [
            '<?xml version="1.0" encoding="UTF-8"?>',
            '<svg xmlns="http://www.w3.org/2000/svg" width="800" height="760" '
            'viewBox="0 0 800 760">',
            '<rect width="800" height="760" fill="#FFFFFF" />',
            cell(
                left,
                top,
                fill="#174A7E",
                count=matrix[("AI", "AI")],
                label="True positive (TP)",
                text_fill="#FFFFFF",
            ),
            cell(
                left + 260,
                top,
                fill="#F3F7FB",
                count=matrix[("AI", "Human")],
                label="False negative (FN)",
                text_fill="#222222",
            ),
            cell(
                left,
                top + 260,
                fill="#FFFFFF",
                count=matrix[("Human", "AI")],
                label="False positive (FP)",
                text_fill="#222222",
            ),
            cell(
                left + 260,
                top + 260,
                fill="#123B67",
                count=matrix[("Human", "Human")],
                label="True negative (TN)",
                text_fill="#FFFFFF",
            ),
            f'<rect x="{left}" y="{top}" width="{size}" height="{size}" '
            'fill="none" stroke="#4B4B4B" stroke-width="2" />',
            f'<line x1="{left + 260}" y1="{top}" x2="{left + 260}" '
            f'y2="{top + size}" stroke="#D6DCE1" stroke-width="1" />',
            f'<line x1="{left}" y1="{top + 260}" x2="{left + size}" '
            f'y2="{top + 260}" stroke="#D6DCE1" stroke-width="1" />',
            svg_text(left + 130, top + size + 38, "AI", size=18),
            svg_text(left + 390, top + size + 38, "Human", size=18),
            svg_text(left - 25, top + 136, "AI", size=18, anchor="end"),
            svg_text(left - 25, top + 396, "Human", size=18, anchor="end"),
            f'<line x1="{left + 130}" y1="{top + size}" x2="{left + 130}" '
            f'y2="{top + size + 8}" stroke="#4B4B4B" stroke-width="2" />',
            f'<line x1="{left + 390}" y1="{top + size}" x2="{left + 390}" '
            f'y2="{top + size + 8}" stroke="#4B4B4B" stroke-width="2" />',
            f'<line x1="{left - 8}" y1="{top + 130}" x2="{left}" '
            f'y2="{top + 130}" stroke="#4B4B4B" stroke-width="2" />',
            f'<line x1="{left - 8}" y1="{top + 390}" x2="{left}" '
            f'y2="{top + 390}" stroke="#4B4B4B" stroke-width="2" />',
            svg_text(left + size / 2, top + size + 92, "predicted label", size=21),
            svg_text(55, top + size / 2, "true label", size=21, rotate=-90),
            '</svg>',
            '',
        ]
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results", type=Path, default=DEFAULT_RESULTS)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    output = args.output.resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(render(read_matrix(args.results.resolve())), encoding="utf-8")
    print(f"Wrote {output}")


if __name__ == "__main__":
    main()
