&nbsp;
# AI-assisted dataset generation

[← Stage 02: AI data](../02_ai-data/) · [Project index](../../README.md) · [Stage 04: Hugging Face dataset →](../04_hf-dataset/)

These scripts create two AI-edited variants from each selected human text:

- `light` corrects spelling, grammar, and punctuation.
- `moderate` improves clarity, concision, and sentence structure.

`--count` is the number of human sources assigned to one model. Each source
produces both edit levels, so `--count 5071` creates 10,142 edited texts per
model. The five model jobs claim disjoint human sources. With 5,071 sources per
job, all 25,355 human samples are used once by one model and produce 50,710
AI-assisted texts in total.

Outputs are stored under `data/ai-assisted/light/` and
`data/ai-assisted/moderate/`. Each metadata row records the source human ID,
editing model, edit level, word- and character-level similarity, character edit
magnitude, license, and generation settings. Keep a human source and both
edited variants in the same train, validation, or test split.

Run the five jobs in separate terminals. Metadata writes and seed claims are
locked, so different model jobs can run concurrently. Do not start the same
model job twice.

```bash
./scripts/03_ai-assisted-data/03_generate_ollama_edits.py \
  --model qwen3.6:35b \
  --count 5071
```

```bash
export OPENAI_API_KEY="sk-..."

./scripts/03_ai-assisted-data/03_generate_openai_edits.py \
  --model sol-medium \
  --count 5071
```

```bash
export OPENROUTER_API_KEY="sk-or-v1-..."

./scripts/03_ai-assisted-data/03_generate_openrouter_edits.py \
  --model moonshotai/kimi-k3 \
  --count 5071
```

```bash
export OPENROUTER_API_KEY="sk-or-v1-..."

./scripts/03_ai-assisted-data/03_generate_openrouter_edits.py \
  --model deepseek-v4-flash-0731 \
  --count 5071
```

```bash
export GEMINI_API_KEY="your-key"

./scripts/03_ai-assisted-data/03_generate_gemini_edits.py \
  --model gemini-3.6-flash \
  --count 5071
```

Add `--dry-run` to validate a command without claiming samples or calling a
model. Interrupted jobs resume from completed edits. Validation failures are
logged and skipped so later sources continue; run the same command again to
retry missing edits.
