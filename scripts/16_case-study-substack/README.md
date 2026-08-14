&nbsp;
# Substack case study

[← Stage 15: Classifier API](../15_classifier-api/) · [Project index](../../README.md) · [Stage 17: Browser UI →](../17_browser-ui/)

This stage applies all nine local classifiers to a labeled collection of
Substack notes. It compares their predictions and writes one confusion matrix
per model.

Run the analysis from the project directory after downloading the model
artifacts:

```bash
uv run python scripts/16_case-study-substack/predict_substack_notes.py
```

The default input is `new-substack-predictions.csv`. The combined predictions
are written to `new-substack-model-predictions.csv`, and the plots are written
to `confusion-matrices/`.
