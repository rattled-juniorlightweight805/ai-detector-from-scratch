#!/usr/bin/env python3
"""Plot inference throughput and peak GPU memory for selected classifiers."""

import argparse
import csv
from collections import namedtuple
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


SCRIPT_DIR = Path(__file__).resolve().parent
DEFAULT_INPUT = SCRIPT_DIR / "results" / "resource-latency-results.csv"
DEFAULT_OUTPUT = SCRIPT_DIR / "figures" / "resource-latency-results.svg"

MODEL_ORDER = (
    "logreg",
    "distilbert",
    "modernbert",
    "gpt2-variable",
    "qwen3-variable",
)
MODEL_LABELS = {
    "logreg": "Logistic regression",
    "distilbert": "DistilBERT",
    "modernbert": "ModernBERT",
    "gpt2-variable": "GPT-2 variable",
    "qwen3-variable": "Qwen3 0.6B variable",
}

BLUE = "#356A9A"
GRAY = "#777777"
LIGHT_BLUE = "#D7E3EE"
GRID = "#E5E5E5"
TEXT = "#222222"


Result = namedtuple(
    "Result",
    (
        "model",
        "mean_texts_per_second",
        "std_texts_per_second",
        "gpu_memory_allocated_peak_mb",
    ),
)


def optional_float(value):
    if value is None or not value.strip():
        return None
    return float(value)


def load_results(path):
    with path.open(encoding="utf-8", newline="") as file:
        rows = {row["model"]: row for row in csv.DictReader(file)}

    missing = [model for model in MODEL_ORDER if model not in rows]
    if missing:
        raise ValueError(f"Missing requested models: {', '.join(missing)}")

    return [
        Result(
            model=model,
            mean_texts_per_second=float(rows[model]["mean_texts_per_second"]),
            std_texts_per_second=float(rows[model]["std_texts_per_second"]),
            gpu_memory_allocated_peak_mb=optional_float(
                rows[model]["gpu_memory_allocated_peak_mb"]
            ),
        )
        for model in MODEL_ORDER
    ]


def format_throughput(value, standard_deviation):
    if value >= 100:
        return f"{value:,.0f} ± {standard_deviation:.1f}"
    if standard_deviation < 0.1:
        return f"{value:.2f} ± {standard_deviation:.2f}"
    return f"{value:.1f} ± {standard_deviation:.1f}"


def style_axis(axis):
    axis.spines[["top", "right", "left"]].set_visible(False)
    axis.spines["bottom"].set_color("#999999")
    axis.tick_params(axis="y", length=0, colors=TEXT)
    axis.tick_params(axis="x", colors="#555555")
    axis.grid(axis="x", color=GRID, linewidth=0.8, zorder=0)


def create_figure(results):
    labels = [MODEL_LABELS[result.model] for result in results]
    y_positions = np.arange(len(results))
    throughput = np.array([result.mean_texts_per_second for result in results])
    throughput_std = np.array([result.std_texts_per_second for result in results])

    figure, (throughput_axis, memory_axis) = plt.subplots(
        1,
        2,
        figsize=(12, 5.2),
        sharey=True,
        gridspec_kw={"width_ratios": (1.12, 1), "wspace": 0.26},
    )
    figure.patch.set_facecolor("white")
    figure.suptitle(
        "Inference throughput and peak GPU memory",
        x=0.08,
        y=0.97,
        ha="left",
        fontsize=18,
        fontweight="bold",
        color=TEXT,
    )
    figure.text(
        0.08,
        0.91,
        "5,066 test texts, batch size 8  ·  LogReg on CPU, neural models on CUDA  ·  mean ± SD across five runs",
        ha="left",
        fontsize=10.5,
        color="#555555",
    )

    throughput_axis.errorbar(
        throughput,
        y_positions,
        xerr=throughput_std,
        fmt="o",
        markersize=7.5,
        markerfacecolor=BLUE,
        markeredgecolor="white",
        markeredgewidth=0.8,
        ecolor=TEXT,
        elinewidth=1.2,
        capsize=3,
        zorder=3,
    )
    throughput_axis.set_xscale("log")
    throughput_axis.set_xlim(8, 4_600)
    throughput_axis.set_xticks([10, 30, 100, 300, 1_000, 3_000])
    throughput_axis.set_xticklabels(["10", "30", "100", "300", "1,000", "3,000"])
    throughput_axis.set_yticks(y_positions, labels=labels, fontsize=10.5)
    throughput_axis.invert_yaxis()
    throughput_axis.set_title(
        "Throughput",
        loc="left",
        pad=16,
        fontsize=12.5,
        fontweight="bold",
        color=TEXT,
    )
    throughput_axis.set_xlabel("Texts per second (log scale)", labelpad=10, color=TEXT)
    style_axis(throughput_axis)

    for y_position, value, standard_deviation in zip(
        y_positions, throughput, throughput_std
    ):
        throughput_axis.annotate(
            format_throughput(value, standard_deviation),
            xy=(value, y_position),
            xytext=(8, 0),
            textcoords="offset points",
            va="center",
            fontsize=9.5,
            fontweight="bold",
            color=BLUE,
        )

    memory_values = np.array(
        [
            np.nan
            if result.gpu_memory_allocated_peak_mb is None
            else result.gpu_memory_allocated_peak_mb
            for result in results
        ]
    )
    available = ~np.isnan(memory_values)
    memory_axis.hlines(
        y_positions[available],
        0,
        memory_values[available],
        color=LIGHT_BLUE,
        linewidth=3,
        zorder=1,
    )
    memory_axis.scatter(
        memory_values[available],
        y_positions[available],
        s=60,
        color=BLUE,
        edgecolor="white",
        linewidth=0.8,
        zorder=3,
    )
    memory_axis.set_xlim(0, 2_300)
    memory_axis.set_xticks([0, 500, 1_000, 1_500, 2_000])
    memory_axis.set_xticklabels(["0", "500", "1,000", "1,500", "2,000"])
    memory_axis.tick_params(axis="y", labelleft=False)
    memory_axis.set_title(
        "Peak allocated GPU memory",
        loc="left",
        pad=16,
        fontsize=12.5,
        fontweight="bold",
        color=TEXT,
    )
    memory_axis.set_xlabel("Memory (MB)", labelpad=10, color=TEXT)
    style_axis(memory_axis)

    for y_position, value in zip(y_positions[available], memory_values[available]):
        memory_axis.annotate(
            f"{value:,.0f} MB",
            xy=(value, y_position),
            xytext=(8, 0),
            textcoords="offset points",
            va="center",
            fontsize=9.5,
            fontweight="bold",
            color=BLUE,
        )

    cpu_positions = y_positions[~available]
    for y_position in cpu_positions:
        memory_axis.text(
            25,
            y_position,
            "CPU",
            va="center",
            fontsize=9.5,
            fontweight="bold",
            color=GRAY,
        )

    figure.subplots_adjust(left=0.18, right=0.97, top=0.78, bottom=0.19)
    return figure


def build_parser():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    return parser


def main():
    args = build_parser().parse_args()
    results = load_results(args.input)
    figure = create_figure(results)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(args.output, bbox_inches="tight", facecolor="white")
    plt.close(figure)
    print(f"Wrote {args.output}")


if __name__ == "__main__":
    main()
