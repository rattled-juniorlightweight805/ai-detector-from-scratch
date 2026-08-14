&nbsp;
# Learning curves

[← Stage 12: Length bias](../12_length-bias/) · [Project index](../../README.md) · [Stage 14: Resource and latency →](../14_resource-latency/)

Each notebook fits a fresh classifier on nested subsets containing 1%, 2.5%,
5%, 10%, 25%, 50%, and 100% of the training split. The validation split stays
fixed, and the test split is not used.

The notebooks are in `notebooks/`:

- `logreg-learning-curves.ipynb`
- `distilbert-learning-curves.ipynb`
- `distilbert-lora-learning-curves.ipynb`
- `distilbert-mica-learning-curves.ipynb`
- `modernbert-learning-curves.ipynb`
- `gpt2-fixed-position-learning-curves.ipynb`
- `gpt2-variable-position-learning-curves.ipynb`
- `qwen3-fixed-position-learning-curves.ipynb`
- `qwen3-variable-position-learning-curves.ipynb`

The neural notebooks use three fixed epochs per point and do not use early
stopping. This gives every point the same training budget. The subset fractions
sum to 1.935, so one complete curve requires almost twice the work of a single
three-epoch full-data run.

ModernBERT, GPT-2, and Qwen3 use the same CUDA and FlashAttention 2 setup as
their final model notebooks. Run JupyterLab from the project root:

```bash
uv run jupyter lab scripts/13_learning-curves/notebooks
```

Each notebook saves its partial results after every completed point and writes
an SVG figure after all points finish. CSV files are written to `results/`,
and the plots are written to `figures/`.
