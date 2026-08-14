&nbsp;
# AI dataset generation

[← Stage 01: Human data](../01_human-data/) · [Project index](../../README.md) · [Stage 03: AI-assisted data →](../03_ai-assisted-data/)

The `02_generate_*_texts.py` scripts create complete AI-written samples. Each
model first generates a question from a human seed and then answers it near the
seed's target length. Only the answer is stored in `data/ai`; the question is
kept in `data/prompts`.

&nbsp;
## Commands


```bash
# Ensure ollama is running

./scripts/02_generate_ollama_questions.py \
  --model qwen3.6:35b \
  --output-folder data/ai \
  --count 5071
```

```bash
export OPENAI_API_KEY="sk-..."

./scripts/02_generate_openai_questions.py \
  --model sol-medium \
  --output-folder data/ai \
  --count 5071
```

```bash
export OPENROUTER_API_KEY="sk-or-v1-..."

./scripts/02_generate_openrouter_questions.py \
  --model kimi-k3 \
  --output-folder data/ai \
  --count 5071
```

```bash
export OPENROUTER_API_KEY="sk-or-v1-..."

./scripts/02_generate_openrouter_questions.py \
  --model deepseek-v4-flash-0731 \
  --output-folder data/ai \
  --count 5071
```



```bash
export GEMINI_API_KEY="your-key"

./scripts/02_generate_gemini_questions.py \
  --model gemini-3.6-flash \
  --output-folder data/ai \
  --count 5071
```

&nbsp;
## Parallel and resumable generation

Different provider/model jobs can run simultaneously. Metadata updates are
locked, fresh jobs claim disjoint human seeds, and a duplicate process for the
same model is rejected. Interrupted jobs resume from completed responses.

Each accepted response must fall within 20 percent of its target length by
default. Successful responses update `data/meta.json` with their model,
provider, target and actual word counts, prompt provenance, and output hash.
Failures are logged and can be retried by running the same command again.
