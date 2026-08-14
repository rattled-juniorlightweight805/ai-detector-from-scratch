&nbsp;
# Resource and latency benchmarks

[← Stage 13: Learning curves](../13_learning-curves/) · [Project index](../../README.md) · [Stage 15: Classifier API →](../15_classifier-api/)

These scripts benchmark the held-out test split with one local classifier at a
time. Model and dataset loading are excluded. Tokenization, device transfer,
model inference, and probability conversion are included.

Each script performs one warm-up batch followed by five measured passes. It
reports full-pass latency, latency per text, texts per second, GPU utilization,
and GPU memory. NVIDIA utilization and device memory come from `nvidia-smi`.
Process-specific allocated and reserved memory come from PyTorch. GPU values
are reported as `n/a` for logistic regression, CPU runs, MPS runs, or systems
without `nvidia-smi`.

Run a single model from the project root, for example:

```bash
uv run python scripts/14_resource-latency/benchmark_distilbert.py
```

Use a different batch size or a small subset for a quick check:

```bash
uv run python scripts/14_resource-latency/benchmark_qwen3_variable.py \
  --batch-size 8 \
  --limit 100
```

The available entry points are:

- `benchmark_logreg.py`
- `benchmark_distilbert.py`
- `benchmark_distilbert_lora.py`
- `benchmark_distilbert_mica.py`
- `benchmark_modernbert.py`
- `benchmark_gpt2_fixed.py`
- `benchmark_gpt2_variable.py`
- `benchmark_qwen3_fixed.py`
- `benchmark_qwen3_variable.py`

Open `notebooks/resource-latency.ipynb` to run all nine scripts as separate
processes and collect the results in `results/resource-latency-results.csv`.
The separate processes ensure that one model's GPU allocation is released
before the next model loads.

Plot the selected model comparison from the collected CSV file with:

```bash
uv run python scripts/14_resource-latency/plot_resource_latency.py
```

This writes `figures/resource-latency-results.svg`. The throughput panel
includes standard-deviation error bars across the five measured passes.
Logistic regression is marked as CPU-only in the GPU-memory panel.

By default, the scripts load `rasbt/human-vs-ai-50k` from the Hugging Face Hub.
Use `--dataset-path data/hf-dataset` to load the local dataset instead.
