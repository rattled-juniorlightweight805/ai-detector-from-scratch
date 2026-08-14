#!/usr/bin/env python3
"""Plot word-count distributions for one or two folders of UTF-8 text files."""

import argparse
import html
import math
import statistics
from collections import namedtuple
from pathlib import Path


BINS = (
    ("words ≤50", 0, 50),
    ("51–100", 51, 100),
    ("101–250", 101, 250),
    ("251–500", 251, 500),
    ("501–1,000", 501, 1_000),
    ("1,001–2,000", 1_001, 2_000),
    (">2,000", 2_001, None),
)
DISPLAY_LABELS = {
    "human": "Human-written samples",
    "ai": "AI-generated samples",
}
PANEL_WIDTH = 640
PANEL_HEIGHT = 500
PANEL_GAP = 36
MARGIN_LEFT = 70
MARGIN_RIGHT = 24
MARGIN_TOP = 100
MARGIN_BOTTOM = 115
COLORS = ("#2C7FB8", "#D95F4C")


class Distribution(
    namedtuple("DistributionFields", "label folder word_counts bin_counts")
):
    __slots__ = ()

    @property
    def shares(self):
        total = len(self.word_counts)
        return tuple(count * 100 / total for count in self.bin_counts)


def count_words(path):
    return len(path.read_text(encoding="utf-8").split())


def assign_bin(word_count):
    for index, (_, low, high) in enumerate(BINS):
        if word_count >= low and (high is None or word_count <= high):
            return index
    raise AssertionError(f"No bin for word count {word_count}")


def read_distribution(folder):
    folder = folder.resolve()
    if not folder.is_dir():
        raise SystemExit(f"Folder does not exist: {folder}")
    text_files = sorted(folder.glob("*.txt"))
    if not text_files:
        raise SystemExit(f"Folder contains no .txt files: {folder}")

    word_counts = tuple(count_words(path) for path in text_files)
    bin_counts = [0] * len(BINS)
    for count in word_counts:
        bin_counts[assign_bin(count)] += 1
    return Distribution(
        label=DISPLAY_LABELS.get(folder.name.lower(), folder.name),
        folder=folder,
        word_counts=word_counts,
        bin_counts=tuple(bin_counts),
    )


def svg_text(
    x,
    y,
    value,
    *,
    size = 13,
    anchor = "middle",
    weight = "normal",
    fill = "#222222",
    rotate = None,
):
    transform = (
        f' transform="rotate({rotate} {x:.1f} {y:.1f})"'
        if rotate is not None
        else ""
    )
    return (
        f'<text x="{x:.1f}" y="{y:.1f}"{transform} text-anchor="{anchor}" '
        f'font-family="-apple-system, BlinkMacSystemFont, Segoe UI, sans-serif" '
        f'font-size="{size}" font-weight="{weight}" fill="{fill}">'
        f"{html.escape(value)}</text>"
    )


def nice_axis_max(maximum_share):
    return max(10, math.ceil(maximum_share * 1.15 / 5) * 5)


def render_panel(
    distribution,
    panel_index,
    axis_max,
    color,
):
    offset_x = panel_index * (PANEL_WIDTH + PANEL_GAP)
    chart_left = offset_x + MARGIN_LEFT
    chart_top = MARGIN_TOP
    chart_width = PANEL_WIDTH - MARGIN_LEFT - MARGIN_RIGHT
    chart_height = PANEL_HEIGHT - MARGIN_TOP - MARGIN_BOTTOM
    chart_bottom = chart_top + chart_height
    bar_step = chart_width / len(BINS)
    bar_width = bar_step * 0.68
    elements = []

    elements.append(
        svg_text(
            offset_x + PANEL_WIDTH / 2,
            34,
            distribution.label,
            size=21,
            weight="600",
        )
    )
    elements.append(
        svg_text(
            offset_x + 18,
            chart_top + chart_height / 2,
            "Share of samples",
            size=12,
            fill="#555555",
            rotate=-90,
        )
    )
    counts = distribution.word_counts
    summary = (
        f"n={len(counts):,}  ·  median={statistics.median(counts):,.0f}  ·  "
        f"mean={statistics.fmean(counts):,.0f}  ·  range={min(counts):,}–{max(counts):,}"
    )
    elements.append(
        svg_text(
            offset_x + PANEL_WIDTH / 2,
            59,
            summary,
            size=12,
            fill="#555555",
        )
    )

    for tick in range(6):
        value = axis_max * tick / 5
        y = chart_bottom - chart_height * tick / 5
        elements.append(
            f'<line x1="{chart_left:.1f}" y1="{y:.1f}" '
            f'x2="{chart_left + chart_width:.1f}" y2="{y:.1f}" '
            'stroke="#E3E3E3" stroke-width="1" />'
        )
        elements.append(
            svg_text(
                chart_left - 10,
                y + 4,
                f"{value:.0f}%",
                size=11,
                anchor="end",
                fill="#666666",
            )
        )

    for index, ((label, _, _), count, share) in enumerate(
        zip(BINS, distribution.bin_counts, distribution.shares)
    ):
        center_x = chart_left + (index + 0.5) * bar_step
        height = chart_height * share / axis_max
        y = chart_bottom - height
        elements.append(
            f'<rect x="{center_x - bar_width / 2:.1f}" y="{y:.1f}" '
            f'width="{bar_width:.1f}" height="{height:.1f}" rx="2" '
            f'fill="{color}" />'
        )
        elements.append(
            svg_text(
                center_x,
                y - 7,
                f"{share:.1f}%",
                size=11,
                weight="600",
                fill=color,
            )
        )
        elements.append(
            svg_text(center_x, chart_bottom + 23, label, size=11)
        )
        elements.append(
            svg_text(
                center_x,
                chart_bottom + 43,
                f"n={count:,}",
                size=10,
                fill="#666666",
            )
        )

    elements.append(
        svg_text(
            offset_x + PANEL_WIDTH / 2,
            PANEL_HEIGHT - 20,
            "Sample length (words)",
            size=12,
            fill="#555555",
        )
    )
    return "\n".join(elements)


def render_svg(distributions):
    width = len(distributions) * PANEL_WIDTH + (len(distributions) - 1) * PANEL_GAP
    maximum_share = max(
        share for distribution in distributions for share in distribution.shares
    )
    axis_max = nice_axis_max(maximum_share)
    panels = [
        render_panel(distribution, index, axis_max, COLORS[index])
        for index, distribution in enumerate(distributions)
    ]
    return (
        '<?xml version="1.0" encoding="UTF-8"?>\n'
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" '
        f'height="{PANEL_HEIGHT}" viewBox="0 0 {width} {PANEL_HEIGHT}">\n'
        '<rect width="100%" height="100%" fill="white" />\n'
        + "\n".join(panels)
        + "\n</svg>\n"
    )


def default_output(folders):
    names = [folder.resolve().name for folder in folders]
    return Path("-vs-".join(names) + "-length-distribution.svg")


def build_parser():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "folders",
        type=Path,
        nargs="+",
        help="One or two folders containing UTF-8 .txt files.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        help="Output SVG path. Defaults to a name derived from the folders.",
    )
    return parser


def main():
    args = build_parser().parse_args()
    if len(args.folders) not in {1, 2}:
        raise SystemExit("Pass one folder or two folders for side-by-side plots")
    distributions = [read_distribution(folder) for folder in args.folders]
    output = (args.output or default_output(args.folders)).resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(render_svg(distributions), encoding="utf-8")
    print(f"Wrote {output}")


if __name__ == "__main__":
    main()
