[← Stage 17: Browser UI](../17_browser-ui/) · [Project index](../../README.md)

&nbsp;
# 1 Dataset preparation

This folder creates a fresh prompt dataset for the human-writing GRPO
experiment. It does not reuse the questions that generated the AI-detector
training texts.

&nbsp;
## 1.1 Select the human-text seeds

Select 5,000 training, 500 validation, and 1,000 test seeds from the existing
grouped dataset splits:

```bash
uv run python scripts/18_reinforcement-learning/01_select_prompt_seeds.py
```

&nbsp;
## 1.2 Generate the questions

Generate the questions with the local Qwen3.5 model:

```bash
uv run python scripts/18_reinforcement-learning/02_generate_ollama_prompts.py \
  --model qwen3.5:4b
```

If Ollama is running on another machine through an IPv6-bound SSH tunnel, the
topic extraction can process eight seeds per request:

```bash
NO_PROXY=::1,localhost,127.0.0.1 \
uv run python scripts/18_reinforcement-learning/02_generate_ollama_prompts.py \
  --model qwen3.5:4b \
  --ollama-url 'http://[::1]:11434' \
  --batch-size 8
```

Generation is resumable. Questions are written under `data/grpo-prompts`, and
the combined dataset is written to `data/grpo-prompts/prompts.jsonl`.
`manifest.jsonl` records the human seed, source group, license, split, and
target answer length for each question.

Ollama reduces each seed text to a short broad topic. In batch mode, the server
returns an ordered JSON array, and the script verifies its length before
attaching IDs locally. The script then inserts each topic into a deterministic
question template so the final prompt does not expose the original seed text.

&nbsp;
## 1.3 Finalize and validate the dataset

```bash
uv run python scripts/18_reinforcement-learning/03_finalize_prompt_set.py
```

The finalization step rotates exact duplicates to unused templates, verifies
the split counts and source-group isolation, rewrites the hashes, and requires
all 6,500 questions to be unique.

&nbsp;
## 1.4 Build and load the Hugging Face dataset

Build the Hugging Face upload folder:

```bash
uv run python scripts/18_reinforcement-learning/04_build_hf_upload.py
```

The published dataset can be loaded with:

```python
from datasets import load_dataset

dataset = load_dataset("rasbt/human-writing-prompts-6k")
```

&nbsp;
# 2 Training

`05_train_grpo_human_writing.py` adapts the batched no-KL GRPO script from
[`reasoning-from-scratch`](https://github.com/rasbt/reasoning-from-scratch/blob/main/ch06/02_rlvr_grpo_scripts_intro/rlvr_grpo_original_no_kl_batched.py).
It fine-tunes `Qwen/Qwen3-0.6B-Base` on the 5,000 training prompts. The frozen
`qwen3-variable` AI detector serves as the verifier. The validation and test
splits are not used for parameter updates.

&nbsp;
## 2.1 Reward

For a generated response with `w` words and target length `t`, the reward is:

```text
(1 - AI probability) * min(w / t, t / w)
```

This favors text that the verifier considers human-written while discouraging
the policy from obtaining a high reward with an unusually short response.

&nbsp;
## 2.2 Download the verifier

```bash
uv run python scripts/15_classifier-api/download-models.py \
  --fetch \
  --model qwen3-variable
```

&nbsp;
## 2.3 Run a smoke test

Run a one-step GPU test before starting a long run:

```bash
uv run python scripts/18_reinforcement-learning/05_train_grpo_human_writing.py \
  --policy-device cuda \
  --verifier-device cuda \
  --steps 1 \
  --num-rollouts 4 \
  --rollout-batch-size 4 \
  --max-new-tokens 256
```

&nbsp;
## 2.4 Run the full training pass

```bash
uv run python scripts/18_reinforcement-learning/05_train_grpo_human_writing.py \
  --policy-device cuda \
  --verifier-device cuda \
  --steps 5000 \
  --num-rollouts 8 \
  --rollout-batch-size 8 \
  --verifier-batch-size 4 \
  --max-new-tokens 1616 \
  --checkpoint-every 50 \
  --skip-zero-advantage-updates
```

The larger token limit allows the model to attempt the dataset's 1,000-word
targets. Checkpoints and CSV metrics are written under
`scripts/18_reinforcement-learning/checkpoints` and
`scripts/18_reinforcement-learning/logs`, respectively.

&nbsp;
# 3 Analysis

&nbsp;
## 3.1 Generate the original-model test responses

Generate one response from the untrained `Qwen/Qwen3-0.6B-Base` policy for
each of the 1,000 test prompts:

```bash
uv run python \
  scripts/18_reinforcement-learning/analysis/06_generate_original_test_cases.py \
  --device cuda \
  --batch-size 8
```

Generation is resumable. The script uses the same prompt formatter, target
lengths, maximum token count, temperature, and top-p value as the GRPO trainer.
It writes individual responses and a combined `results.jsonl` file under
`scripts/18_reinforcement-learning/analysis/test-cases-original`.

&nbsp;
## 3.2 Generate test responses from a checkpoint

List the complete checkpoints saved by the training script:

```bash
uv run python \
  scripts/18_reinforcement-learning/analysis/07_generate_checkpoint_test_cases.py \
  --list-checkpoints
```

The listing marks each checkpoint as `ready` or `invalid`. An invalid
checkpoint is usually still being synchronized or has an incomplete weights
file.

Generate the test responses for one checkpoint:

```bash
uv run python \
  scripts/18_reinforcement-learning/analysis/07_generate_checkpoint_test_cases.py \
  --checkpoint step-00500 \
  --device cuda \
  --batch-size 8
```

The checkpoint can also be given as an absolute path. By default, the example
above writes to `analysis/test-cases-step-00500`. Each checkpoint gets its own
resumable output directory.

&nbsp;
## 3.3 Generate test responses every 500 checkpoints

Process checkpoints 500, 1,000, and so on sequentially:

```bash
uv run python \
  scripts/18_reinforcement-learning/analysis/08_generate_all_checkpoint_test_cases.py \
  --device cuda \
  --batch-size 8
```

Each checkpoint writes to its own `analysis/test-cases-step-*` directory. Both
the meta script and child generator stop on errors by default, and rerunning
the command resumes incomplete checkpoints without regenerating completed
responses. The interval defaults to 500 and can be changed with
`--checkpoint-every`. Inspect the commands without starting generation with
`--dry-run`.
