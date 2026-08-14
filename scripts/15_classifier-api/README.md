&nbsp;
# Classifier API

[← Stage 14: Resource and latency](../14_resource-latency/) · [Project index](../../README.md) · [Stage 16: Substack case study →](../16_case-study-substack/)

`classify.py` provides one command-line interface for all locally trained
classifiers. It returns the calibrated probability that the input was
AI-generated as a score from 0 to 100:

```json
{"score": 97.3142}
```

Run commands from the project directory.

The logistic regression detector works after the default `uv sync`. Install
the transformer inference group before using any neural detector:

```bash
uv sync --group transformer-inference
```

&nbsp;
## Prepare the models directory

The classifier API reads its default artifacts from the top-level `models/`
directory. Download all published models from the Hugging Face Hub with:

```bash
uv run python scripts/15_classifier-api/download-models.py --fetch
```

To download only one model:

```bash
uv run python scripts/15_classifier-api/download-models.py \
  --fetch \
  --model distilbert
```

Alternatively, copy all complete local training exports into `models/` with:

```bash
uv run python scripts/15_classifier-api/download-models.py --local
```

To copy only one model:

```bash
uv run python scripts/15_classifier-api/download-models.py \
  --local \
  --model distilbert
```

The script downloads or copies into a temporary directory first. It validates
each export before replacing `models/<model-name>/`, so a failed or incomplete
download leaves an existing model unchanged. The top-level `models/` directory
is excluded from Git.

The LoRA and MiCA repositories contain adapter weights. Loading either model
for the first time may also download the shared DistilBERT base model into the
Hugging Face cache.

&nbsp;
## Check the available exports

```bash
uv run python scripts/15_classifier-api/classify.py --list-models
```

A model is listed as ready only when its top-level model directory contains
the required configuration, tokenizer, and weights.

&nbsp;
## Classify text

Logistic regression:

```bash
uv run python scripts/15_classifier-api/classify.py \
  --model logreg \
  --text "Paste the text to classify here."
```

DistilBERT:

```bash
uv run python scripts/15_classifier-api/classify.py \
  --model distilbert \
  --text "Paste the text to classify here."
```

DistilBERT with LoRA:

```bash
uv run python scripts/15_classifier-api/classify.py \
  --model distilbert-lora \
  --text "Paste the text to classify here."
```

DistilBERT with MiCA:

```bash
uv run python scripts/15_classifier-api/classify.py \
  --model distilbert-mica \
  --text "Paste the text to classify here."
```

ModernBERT:

```bash
uv run python scripts/15_classifier-api/classify.py \
  --model modernbert \
  --text "Paste the text to classify here."
```

GPT-2 with a variable-position readout:

```bash
uv run python scripts/15_classifier-api/classify.py \
  --model gpt2-variable \
  --text "Paste the text to classify here."
```

GPT-2 with a fixed-position readout:

```bash
uv run python scripts/15_classifier-api/classify.py \
  --model gpt2-fixed \
  --text "Paste the text to classify here."
```

Qwen3-0.6B with a variable-position readout:

```bash
uv run python scripts/15_classifier-api/classify.py \
  --model qwen3-variable \
  --text "Paste the text to classify here."
```

Qwen3-0.6B with a fixed-position readout:

```bash
uv run python scripts/15_classifier-api/classify.py \
  --model qwen3-fixed \
  --text "Paste the text to classify here."
```

Use a UTF-8 text file instead of `--text`:

```bash
uv run python scripts/15_classifier-api/classify.py \
  --model distilbert \
  --file article.txt
```

Or pipe text through standard input:

```bash
pbpaste | uv run python scripts/15_classifier-api/classify.py \
  --model distilbert
```

Use `--device cpu`, `--device cuda`, or `--device mps` to override automatic
device selection. Use `--artifact PATH` when an exported model is stored at a
non-default location.

&nbsp;
## Keep a model in memory

The regular CLI loads the model for each invocation. Start the local server
when making repeated predictions. Install its dependencies first:

```bash
uv sync --group browser-ui
```

Then start the server:

```bash
uv run python scripts/15_classifier-api/serve.py \
  --model distilbert \
  --device cuda
```

The server listens only on `127.0.0.1` by default. It loads the selected model
once during startup and keeps it in memory until the server is stopped.

Send text through the existing CLI from another terminal:

```bash
uv run python scripts/15_classifier-api/classify.py \
  --server http://127.0.0.1:8000 \
  --text "Paste the text to classify here."
```

The server can also be called directly:

```bash
curl http://127.0.0.1:8000/api/check \
  -H "Content-Type: application/json" \
  -d '{"text": "Paste the text to classify here."}'
```

The response uses the same JSON format as the one-shot CLI:

```json
{"score": 97.3142}
```

Only one model is loaded by each server process. To serve another model, stop
the process and restart it with a different `--model` value or use a different
port for a second server.

&nbsp;
## Python API

```python
from ai_detector import load_classifier, score_payload

classifier = load_classifier("distilbert", device="auto")
probability = classifier.score_many(["Text to classify."], batch_size=1)[0]
result = score_payload(probability)
print(result)
```

The reusable implementation lives in `src/ai_detector`. The scripts in this
folder are command-line and server entry points.
