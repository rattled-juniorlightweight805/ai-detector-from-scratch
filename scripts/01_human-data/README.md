&nbsp;
# Human dataset collection

[Project index](../../README.md) · [Stage 02: AI data →](../02_ai-data/)

`01_collect_licensed_human_text.py` collects human-written text published by
December 31, 2022. It writes numbered UTF-8 files to `data/human` and records
the source, date, license, length, and hash in `data/meta.json`.

```bash
./scripts/01_human-data/01_collect_licensed_human_text.py \
  --data-dir data --sources all --target-per-source 3500 \
  --cutoff-date 2022-12-31
```

The collector is resumable, skips duplicate hashes, and contributes at most 12
chunks per source document. Its default chunk targets rotate through 50, 100,
250, 500, and 1,000 words.

To plot the word-count distribution of one folder, or compare two folders side
by side, run:

```bash
./scripts/01_human-data/01_plot_length_distribution.py data/human

./scripts/01_human-data/01_plot_length_distribution.py \
  data/human data/ai --output human-vs-ai-length-distribution.svg
```

The dependency-free script writes an SVG using the same length buckets as the
table below.

&nbsp;
## Current sources and licenses

| Collection | Samples | License composition | Word range |
|---|---:|---|---:|
| `pmc` | 3,500 | CC0 1.0 | 50–1,049 |
| `plos` | 3,500 | 3,069 CC0 1.0; 431 CC BY 4.0 | 50–1,030 |
| `wikimedia` | 3,500 | CC BY-SA 4.0 | 50–1,044 |
| `stackexchange` | 3,498 | 689 CC BY-SA 2.5; 673 CC BY-SA 3.0; 2,136 CC BY-SA 4.0 | 50–1,000 |
| `gutenberg` | 3,500 | Public domain in the United States | 50–1,044 |
| `openstax` | 3,500 | CC BY 4.0 | 50–1,048 |
| `arxiv` | 3,500 | 3,234 CC BY 4.0; 169 CC BY-SA 4.0; 97 CC0 1.0 | 50–1,045 |
| `arxiv-preprints` | 327 | CC BY-SA 4.0 | 50–1,033 |
| `blog` | 73 | CC BY-SA 4.0 | 103–3,981 |
| `ml-q-and-ai` | 108 | CC BY-SA 4.0 | 52–1,021 |
| `personal-blog` | 248 | CC BY-SA 4.0 | 100–6,797 |
| `website` | 101 | CC BY-SA 4.0 | 99–896 |

The last five collections are Sebastian Raschka-owned material released for
this dataset under CC BY-SA 4.0. Stack Overflow licenses follow each
contribution's publication date. arXiv, PMC, and PLOS licenses are checked at
the individual-record level.

&nbsp;
## Current length distribution

The corpus contains 25,355 samples, including 25,237 chunks and 118 full texts.
The minimum is 50 words, the median is 250, the 90th percentile is 1,000, and
the maximum is 6,797 words.

| Words | Samples | Share |
|---:|---:|---:|
| 50 | 3,734 | 14.73% |
| 51–100 | 6,194 | 24.43% |
| 101–250 | 6,208 | 24.48% |
| 251–500 | 5,033 | 19.85% |
| 501–1,000 | 4,015 | 15.84% |
| 1,001–2,000 | 129 | 0.51% |
| More than 2,000 | 42 | 0.17% |
