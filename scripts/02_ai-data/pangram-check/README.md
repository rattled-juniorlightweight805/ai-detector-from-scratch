&nbsp;
# Pangram check

[← Stage 02: AI data](../README.md) · [Project index](../../../README.md)

This folder records a Pangram 3 evaluation on a reproducible random sample of
50 human-written and 50 AI-generated dataset files. Samples are selected
without replacement by sorting each class with a SHA-256 rank derived from the
fixed seed `20260804`.

The API job was originally submitted with 150 samples per class. The requested
scope changed after Pangram had accepted that job, so the saved manifest and
reported results retain only the first 50 hash-ranked samples from each class.
This is the same subset that a fresh run with the current defaults selects.
The source job cost an estimated `$12.92`; scanning only the retained 100 files
would require 107 Pangram 3 bulk units, or an estimated `$4.28`.

Run a free dry run first to write `sample-manifest.json` and show the billable
word-block count:

```bash
./scripts/02_ai-data/pangram-check/run_check.py
```

Submit or resume the bulk job with:

```bash
./scripts/02_ai-data/pangram-check/run_check.py --submit
```

The script reads `PANGRAM_API_KEY` from the environment or from
`~/.env.pangram`. Pangram currently exposes version 3 through the `default`
model selector for this API key. `sample-manifest.json` contains the selected
file IDs without their text. `results.json` and `results.csv` contain the
returned labels and scores without duplicating the source text. `summary.json`
contains prediction counts and an AI-versus-human matrix while preserving any
Mixed or AI-Assisted predictions as unresolved.
