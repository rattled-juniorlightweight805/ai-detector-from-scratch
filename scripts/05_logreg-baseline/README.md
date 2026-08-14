&nbsp;
# Logistic regression baseline

[← Stage 04: Hugging Face dataset](../04_hf-dataset/) · [Project index](../../README.md) · [Stage 06: ChatGPT baseline →](../06_chatgpt-baseline/)

`sklearn-baseline.ipynb` trains, calibrates, evaluates, and exports the TF-IDF logistic regression baseline.

&nbsp;
## Inference benchmark

`benchmark_inference.py` measures calibrated `predict_proba` inference on all samples in the local test split. Model loading, dataset loading, and conversion of dataset columns are completed before timing starts. The script also performs one untimed warm-up pass.

From the project directory, run:

```bash
uv run python scripts/05_logreg-baseline/benchmark_inference.py
```

By default, it performs five measured passes and reports each run, the mean, sample standard deviation, mean time per sample, throughput, and test accuracy.

The defaults can be changed with:

```bash
uv run python scripts/05_logreg-baseline/benchmark_inference.py \
  --repeats 10 \
  --model-path scripts/05_logreg-baseline/artifacts/logreg-ai-detector.joblib \
  --dataset-path data/hf-dataset
```

Run `uv run python scripts/05_logreg-baseline/benchmark_inference.py --help` for all options.
