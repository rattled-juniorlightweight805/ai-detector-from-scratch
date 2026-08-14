&nbsp;
# Pangram check

[← Stage 03: AI-assisted data](../README.md) · [Project index](../../../README.md)

`run_check.py` freezes the currently completed AI-assisted files into
`sample-manifest.json`, submits that snapshot to Pangram 3, and saves one row
per text in `results.csv` and `results.json`. Each result row includes the full
text, editing level, generating model, Pangram label, class fractions, and
window-level AI-assistance scores.

The manifest is reused on subsequent runs, so an active generation job cannot
change the evaluation set. Delete the Pangram-check output files only when you
intentionally want to create and pay for a new snapshot.

```bash
./scripts/03_ai-assisted-data/pangram-check/run_check.py
```
