&nbsp;
# Hugging Face dataset construction

[← Stage 03: AI-assisted data](../03_ai-assisted-data/) · [Project index](../../README.md) · [Stage 05: Logistic regression →](../05_logreg-baseline/)

`04_build_hf_dataset.py` converts the public `data/human` and `data/ai` text
files into a local Hugging Face `DatasetDict` with `train`, `validation`, and
`test` splits. Rows labeled `ai-assisted` are intentionally excluded. The
script does not upload anything.

Run:

```bash
uv run python scripts/04_hf-dataset/04_build_hf_dataset.py
```

The default output is `data/hf-dataset`, which is excluded from Git. The
builder refuses to overwrite an existing output directory. Pass a different
location with `--output-dir` when you want to compare builds.

&nbsp;
## Loading the dataset locally

Load the local `DatasetDict` with:

```python
from datasets import load_from_disk

dataset = load_from_disk("data/hf-dataset")
train_dataset = dataset["train"]
validation_dataset = dataset["validation"]
test_dataset = dataset["test"]
```

&nbsp;
## Loading from the Hugging Face Hub

The published dataset is available at
[`rasbt/human-vs-ai-50k`](https://huggingface.co/datasets/rasbt/human-vs-ai-50k).
Load all three splits with:

```python
from datasets import load_dataset

dataset = load_dataset("rasbt/human-vs-ai-50k")
train_dataset = dataset["train"]
validation_dataset = dataset["validation"]
test_dataset = dataset["test"]
```

To load only one split:

```python
train_dataset = load_dataset("rasbt/human-vs-ai-50k", split="train")
```

&nbsp;
## Split construction

The split is deterministic for a given `--seed`, which defaults to `17`. The
builder uses ten-fold `StratifiedGroupKFold` over the label and source
collection. Fold 0 becomes the test split, fold 1 becomes validation, and the
remaining eight folds become training data.

The group key is `source_collection:source_document_id`. An AI text inherits
the group of its seed human sample. Consequently, chunks from one source
document and AI texts based on those chunks cannot cross split boundaries.
The expected split proportions are approximately 80%, 10%, and 10%.

The seed-17 build currently contains:

| Split | Human | AI | Total |
|---|---:|---:|---:|
| Train | 20,256 | 20,303 | 40,559 |
| Validation | 2,552 | 2,533 | 5,085 |
| Test | 2,547 | 2,519 | 5,066 |
| **Total** | **25,355** | **25,355** | **50,710** |

&nbsp;
## Stored columns

| Column | Meaning |
|---|---|
| `text` | Exact UTF-8 text from the source `.txt` file, excluding its trailing newline |
| `label` | Hugging Face `ClassLabel` with `human=0` and `ai=1` |
| `split` | `train`, `validation`, or `test` |
| `group_id` | Source-lineage group used to prevent split leakage |
| `id`, `local_file`, `sha256`, `word_count` | Local sample identity and integrity fields; the hash covers `text` without its file-ending newline |
| `text_collection`, `sample_type` | How the stored text entered the corpus |
| `source_*` | Source-document provenance and row-level license information |
| `attribution_name`, `attribution_url` | Display-ready attribution; explicit attribution metadata takes precedence, followed by the source author or name and source URL |
| `generator_provider`, `generator_model` | AI generator information; null for human samples |
| `seed_sample_id` | Human seed ID for AI samples; null for human samples |
| `target_words` | Collection or generation length target when available |

For AI rows, `source_license` describes the human seed material. It should not
be interpreted as a separate copyright determination for the generated text.
Because the corpus contains several source licenses, the Hub dataset card uses
`license: other`. The row-level fields retain the applicable license details.

&nbsp;
## Verification

The build stops on missing files, invalid UTF-8, hash or word-count mismatches,
duplicate IDs, duplicate text hashes, source-group leakage, unexpected split
ratios, or excessive class imbalance. It reloads the saved output and checks
the split names, row counts, and `ClassLabel` encoding. A detailed report is
written to `data/hf-dataset/summary.json`.

The saved folder is a local DatasetDict. Do not upload that Arrow directory
directly. In the later Hub-upload stage, load it with `load_from_disk` and use
`DatasetDict.push_to_hub`, which uploads Hub-compatible Parquet shards.
