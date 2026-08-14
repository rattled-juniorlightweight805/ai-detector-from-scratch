
import json
from pathlib import Path


NOTEBOOK_PATH = (
    Path(__file__).resolve().parents[1]
    / "scripts"
    / "07_distilbert"
    / "distilbert.ipynb"
)


def load_notebook():
    return json.loads(NOTEBOOK_PATH.read_text(encoding="utf-8"))


def notebook_source(notebook):
    return "\n".join(
        "".join(cell.get("source", []))
        for cell in notebook["cells"]
    )


def test_notebook_code_cells_compile():
    notebook = load_notebook()
    for cell in notebook["cells"]:
        if cell["cell_type"] != "code":
            continue
        source = "".join(cell.get("source", []))
        compile(source, f"{NOTEBOOK_PATH.name}:{cell['id']}", "exec")


def test_notebook_keeps_the_test_split_held_out_until_evaluation():
    source = notebook_source(load_notebook())
    trainer_setup = source.index("trainer = Trainer(")
    final_evaluation = source.index("## Final evaluation")

    assert 'train_dataset=tokenized_dataset["train"]' in source[trainer_setup:]
    assert 'eval_dataset=tokenized_dataset["validation"]' in source[trainer_setup:]
    assert 'tokenized_dataset["test"]' not in source[trainer_setup:final_evaluation]


def test_notebook_contains_distilbert_calibration_and_export():
    source = notebook_source(load_notebook())

    assert 'MODEL_NAME = "distilbert/distilbert-base-uncased"' in source
    assert "MAX_LENGTH = 512" in source
    assert "def fit_temperature" in source
    assert "distilbert-confmat.svg" in source
    assert "trainer.save_model(ARTIFACT_DIR)" in source
    assert "tokenizer.save_pretrained(ARTIFACT_DIR)" in source
