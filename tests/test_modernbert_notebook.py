
import json
from pathlib import Path


NOTEBOOK_PATH = (
    Path(__file__).resolve().parents[1]
    / "scripts"
    / "09_modernbert"
    / "modernbert.ipynb"
)


def load_notebook():
    return json.loads(NOTEBOOK_PATH.read_text(encoding="utf-8"))


def notebook_source(notebook):
    return "\n".join(
        "".join(cell.get("source", []))
        for cell in notebook["cells"]
    )


def test_notebook_code_cells_compile_and_ids_are_unique():
    notebook = load_notebook()
    cell_ids = [cell["id"] for cell in notebook["cells"]]
    assert len(cell_ids) == len(set(cell_ids))

    for cell in notebook["cells"]:
        if cell["cell_type"] != "code":
            continue
        source = "".join(cell.get("source", []))
        compile(source, f"{NOTEBOOK_PATH.name}:{cell['id']}", "exec")


def test_notebook_uses_modernbert_base_at_native_context_length():
    source = notebook_source(load_notebook())

    assert 'MODEL_NAME = "answerdotai/ModernBERT-base"' in source
    assert "NATIVE_CONTEXT_LENGTH = 8192" in source
    assert "MAX_LENGTH = NATIVE_CONTEXT_LENGTH" in source
    assert "AutoModelForSequenceClassification.from_pretrained(" in source


def test_notebook_requires_optimized_cuda_kernels():
    source = notebook_source(load_notebook())

    assert 'find_spec("flash_attn")' in source
    assert 'ATTN_IMPLEMENTATION = "flash_attention_2"' in source
    assert "attn_implementation=ATTN_IMPLEMENTATION" in source
    assert 'optim="adamw_torch_fused"' in source
    assert 'train_sampling_strategy="group_by_length"' in source
    assert "torch_compile=True" in source


def test_notebook_preserves_effective_batch_size_with_accumulation():
    source = notebook_source(load_notebook())

    assert "per_device_train_batch_size=2" in source
    assert "gradient_accumulation_steps=8" in source
    assert '"effective_train_batch_size": 16' in source


def test_test_split_is_held_out_until_final_evaluation():
    source = notebook_source(load_notebook())
    trainer_setup = source.index("trainer = Trainer(")
    final_evaluation = source.index("## Final evaluation")

    assert 'train_dataset=tokenized_dataset["train"]' in source[trainer_setup:]
    assert 'eval_dataset=tokenized_dataset["validation"]' in source[trainer_setup:]
    assert 'tokenized_dataset["test"]' not in source[trainer_setup:final_evaluation]


def test_notebook_calibrates_evaluates_and_exports_model():
    source = notebook_source(load_notebook())

    assert "def fit_temperature" in source
    assert "distilbert-confmat.svg" not in source
    assert "modernbert-confmat.svg" in source
    assert 'Path("artifacts") / "modernbert-ai-detector"' in source
    assert "trainer.save_model(ARTIFACT_DIR)" in source
    assert "tokenizer.save_pretrained(ARTIFACT_DIR)" in source
    assert "training_runtime_seconds" in source


def test_new_notebook_has_no_stored_outputs():
    notebook = load_notebook()
    code_cells = [
        cell for cell in notebook["cells"] if cell["cell_type"] == "code"
    ]

    assert all(cell["execution_count"] is None for cell in code_cells)
    assert all(cell["outputs"] == [] for cell in code_cells)
