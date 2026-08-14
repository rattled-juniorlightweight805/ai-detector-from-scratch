
import json
from pathlib import Path


NOTEBOOK_DIR = (
    Path(__file__).resolve().parents[1]
    / "scripts"
    / "11_qwen3"
)
NOTEBOOK_PATHS = {
    "variable": NOTEBOOK_DIR / "qwen3-variable-position.ipynb",
    "fixed": NOTEBOOK_DIR / "qwen3-fixed-position.ipynb",
}


def load_notebook(variant):
    return json.loads(
        NOTEBOOK_PATHS[variant].read_text(encoding="utf-8")
    )


def notebook_source(notebook):
    return "\n".join(
        "".join(cell.get("source", []))
        for cell in notebook["cells"]
    )


def test_notebook_code_compiles_and_has_unique_cell_ids():
    for variant, path in NOTEBOOK_PATHS.items():
        notebook = load_notebook(variant)
        cell_ids = [cell["id"] for cell in notebook["cells"]]
        assert len(cell_ids) == len(set(cell_ids))

        for cell in notebook["cells"]:
            if cell["cell_type"] != "code":
                continue
            source = "".join(cell.get("source", []))
            compile(source, f"{path.name}:{cell['id']}", "exec")


def test_notebooks_use_qwen3_pretrained_readout_and_padding():
    for variant in NOTEBOOK_PATHS:
        source = notebook_source(load_notebook(variant))

        assert 'MODEL_NAME = "Qwen/Qwen3-0.6B"' in source
        assert "NATIVE_CONTEXT_LENGTH = 32768" in source
        assert "CONTEXT_LENGTH = 1024" in source
        assert "MAX_TEXT_LENGTH = CONTEXT_LENGTH - 1" in source
        assert "tokenizer.eos_token_id is not None" in source
        assert "tokenizer.pad_token_id is not None" in source
        assert "tokenizer.pad_token_id != tokenizer.eos_token_id" in source
        assert "model.config.pad_token_id = tokenizer.pad_token_id" in source
        assert "active_tokenizer.eos_token_id" in source
        assert "AutoModelForSequenceClassification.from_pretrained(" in source
        assert "add_special_tokens({" not in source
        assert "resize_token_embeddings(" not in source


def test_notebooks_require_optimized_cuda_training():
    for variant in NOTEBOOK_PATHS:
        source = notebook_source(load_notebook(variant))

        assert 'find_spec("flash_attn")' in source
        assert 'ATTN_IMPLEMENTATION = "flash_attention_2"' in source
        assert "attn_implementation=ATTN_IMPLEMENTATION" in source
        assert "dtype=model_dtype" in source
        assert 'optim="adamw_torch_fused"' in source
        assert "torch_compile=True" in source
        assert "dataloader_num_workers=4" in source
        assert "dataloader_persistent_workers=True" in source
        assert "dataloader_prefetch_factor=2" in source


def test_variable_notebook_uses_dynamic_right_padding():
    source = notebook_source(load_notebook("variable"))

    assert 'READOUT_POSITION = "variable"' in source
    assert "DataCollatorWithPadding(" in source
    assert "input_ids + [eos_token_id]" in source
    assert "attention_mask + [1]" in source
    assert "padding_length" not in source
    assert 'train_sampling_strategy="group_by_length"' in source


def test_fixed_notebook_places_eos_after_masked_padding():
    source = notebook_source(load_notebook("fixed"))

    assert 'READOUT_POSITION = "fixed"' in source
    assert "DefaultDataCollator()" in source
    assert "padding_length = (" in source
    assert "[active_tokenizer.pad_token_id] * padding_length" in source
    assert "+ [eos_token_id]" in source
    assert "+ [0] * padding_length" in source
    assert "+ [1]" in source
    assert 'train_sampling_strategy="group_by_length"' not in source


def test_test_split_is_held_out_until_final_evaluation():
    for variant in NOTEBOOK_PATHS:
        source = notebook_source(load_notebook(variant))
        trainer_setup = source.index("trainer = Trainer(")
        final_evaluation = source.index("## Final evaluation")

        assert 'train_dataset=tokenized_dataset["train"]' in source
        assert 'eval_dataset=tokenized_dataset["validation"]' in source
        assert (
            'tokenized_dataset["test"]'
            not in source[trainer_setup:final_evaluation]
        )


def test_post_training_predictions_keep_logits_and_labels_aligned():
    for variant in NOTEBOOK_PATHS:
        source = notebook_source(load_notebook(variant))

        assert "validation_output.label_ids" in source
        assert "prediction_output.label_ids" in source
        assert 'split_labels["test"]' in source
        assert "np.testing.assert_array_equal(" in source


def test_notebooks_calibrate_evaluate_and_export_distinct_artifacts():
    expected = {
        "variable": (
            "qwen3-variable-position-confmat.svg",
            'Path("artifacts") / "qwen3-variable-position-ai-detector"',
        ),
        "fixed": (
            "qwen3-fixed-position-confmat.svg",
            'Path("artifacts") / "qwen3-fixed-position-ai-detector"',
        ),
    }

    for variant, (figure_name, artifact_path) in expected.items():
        source = notebook_source(load_notebook(variant))

        assert "def fit_temperature" in source
        assert figure_name in source
        assert artifact_path in source
        assert "trainer.save_model(ARTIFACT_DIR)" in source
        assert "tokenizer.save_pretrained(ARTIFACT_DIR)" in source
        assert '"readout_position": READOUT_POSITION' in source
        assert '"native_context_length": NATIVE_CONTEXT_LENGTH' in source
        assert ".float()" in source


def test_new_notebooks_have_no_stored_outputs():
    for variant in NOTEBOOK_PATHS:
        notebook = load_notebook(variant)
        code_cells = [
            cell
            for cell in notebook["cells"]
            if cell["cell_type"] == "code"
        ]

        assert all(cell["execution_count"] is None for cell in code_cells)
        assert all(cell["outputs"] == [] for cell in code_cells)
