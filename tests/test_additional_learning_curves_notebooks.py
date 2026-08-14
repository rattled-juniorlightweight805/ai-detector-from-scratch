
import json
from pathlib import Path


PROJECT_DIR = Path(__file__).resolve().parents[1]
NOTEBOOK_DIR = (
    PROJECT_DIR / "scripts" / "13_learning-curves" / "notebooks"
)

EXPECTED_NOTEBOOKS = {
    "logreg-learning-curves.ipynb": (
        'TfidfVectorizer(',
        'C=3.0',
    ),
    "distilbert-lora-learning-curves.ipynb": (
        'MODEL_NAME = "distilbert/distilbert-base-uncased"',
        'r=8',
        'lora_alpha=16',
    ),
    "distilbert-mica-learning-curves.ipynb": (
        'MODEL_NAME = "distilbert/distilbert-base-uncased"',
        'r=4',
        'init_lora_weights="mica"',
    ),
    "modernbert-learning-curves.ipynb": (
        'MODEL_NAME = "answerdotai/ModernBERT-base"',
        'MAX_LENGTH = 8192',
    ),
    "gpt2-fixed-position-learning-curves.ipynb": (
        'MODEL_NAME = "openai-community/gpt2"',
        'READOUT_POSITION = "fixed"',
    ),
    "gpt2-variable-position-learning-curves.ipynb": (
        'MODEL_NAME = "openai-community/gpt2"',
        'READOUT_POSITION = "variable"',
    ),
    "qwen3-fixed-position-learning-curves.ipynb": (
        'MODEL_NAME = "Qwen/Qwen3-0.6B"',
        'READOUT_POSITION = "fixed"',
    ),
    "qwen3-variable-position-learning-curves.ipynb": (
        'MODEL_NAME = "Qwen/Qwen3-0.6B"',
        'READOUT_POSITION = "variable"',
    ),
}


def load_notebook(filename):
    path = NOTEBOOK_DIR / filename
    return json.loads(path.read_text(encoding="utf-8"))


def notebook_source(notebook):
    return "\n".join(
        "".join(cell.get("source", []))
        for cell in notebook["cells"]
    )


def test_all_additional_notebooks_compile_and_match_their_models():
    for filename, expected_fragments in EXPECTED_NOTEBOOKS.items():
        notebook = load_notebook(filename)
        cell_ids = [cell["id"] for cell in notebook["cells"]]
        assert len(cell_ids) == len(set(cell_ids))

        source = notebook_source(notebook)
        for fragment in expected_fragments:
            assert fragment in source

        for cell in notebook["cells"]:
            if cell["cell_type"] != "code":
                continue
            cell_source = "".join(cell.get("source", []))
            compile(cell_source, f"{filename}:{cell['id']}", "exec")


def test_all_curves_use_nested_training_subsets_and_fixed_validation():
    for filename in EXPECTED_NOTEBOOKS:
        source = notebook_source(load_notebook(filename))
        assert "TRAIN_FRACTIONS = np.asarray(" in source
        assert "def make_distribution_aware_order(" in source
        assert "nested_order[:sample_count]" in source
        assert "assert previous_indices.issubset(current_indices)" in source
        assert 'dataset["train"]' in source
        assert 'dataset["validation"]' in source
        assert 'dataset["test"]' not in source
        assert 'PROJECT_DIR / "scripts" / "13_learning-curves"' in source


def test_neural_curves_start_fresh_and_use_equal_training_budgets():
    neural_notebooks = set(EXPECTED_NOTEBOOKS) - {
        "logreg-learning-curves.ipynb"
    }
    for filename in neural_notebooks:
        source = notebook_source(load_notebook(filename))
        loop_source = source[source.index("for sample_count in train_sizes:"):]
        assert "set_seed(RANDOM_STATE)" in loop_source
        assert "model = build_model()" in loop_source
        assert "NUM_TRAIN_EPOCHS = 3" in source
        assert "num_train_epochs=NUM_TRAIN_EPOCHS" in source
        assert 'eval_strategy="no"' in source
        assert 'save_strategy="no"' in source
        assert "EarlyStoppingCallback" not in source


def test_causal_curves_keep_their_readout_construction():
    fixed_names = (
        "gpt2-fixed-position-learning-curves.ipynb",
        "qwen3-fixed-position-learning-curves.ipynb",
    )
    variable_names = (
        "gpt2-variable-position-learning-curves.ipynb",
        "qwen3-variable-position-learning-curves.ipynb",
    )
    for filename in (
        "gpt2-fixed-position-learning-curves.ipynb",
        "gpt2-variable-position-learning-curves.ipynb",
    ):
        source = notebook_source(load_notebook(filename))
        assert 'tokenizer.add_special_tokens({"pad_token": "<|pad|>"})' in source
        assert "{{" not in source

    for filename in fixed_names:
        source = notebook_source(load_notebook(filename))
        assert "+ [active_tokenizer.eos_token_id]" in source
        assert "+ [0] * padding_length" in source
        assert "DefaultDataCollator()" in source

    for filename in variable_names:
        source = notebook_source(load_notebook(filename))
        assert 'tokenizer.padding_side = "right"' in source
        assert "attention_mask + [1]" in source
        assert "DataCollatorWithPadding(" in source
