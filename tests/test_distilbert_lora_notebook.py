
import json
from pathlib import Path


NOTEBOOK_PATH = (
    Path(__file__).resolve().parents[1]
    / "scripts"
    / "08_distilbert-lora"
    / "distilbert-lora.ipynb"
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


def test_lora_targets_attention_and_saves_the_classification_head():
    source = notebook_source(load_notebook())

    assert 'LORA_TARGET_MODULES = ["q_lin", "v_lin"]' in source
    assert 'MODULES_TO_SAVE = ["pre_classifier", "classifier"]' in source
    assert "task_type=TaskType.SEQ_CLS" in source
    assert "r=LORA_RANK" in source
    assert "model = get_peft_model(base_model, lora_config)" in source
    assert "model.print_trainable_parameters()" in source


def test_test_split_is_held_out_until_final_evaluation():
    source = notebook_source(load_notebook())
    trainer_setup = source.index("trainer = Trainer(")
    final_evaluation = source.index("## Final evaluation")

    assert 'train_dataset=tokenized_dataset["train"]' in source[trainer_setup:]
    assert 'eval_dataset=tokenized_dataset["validation"]' in source[trainer_setup:]
    assert 'tokenized_dataset["test"]' not in source[trainer_setup:final_evaluation]


def test_notebook_calibrates_and_exports_a_reloadable_adapter():
    source = notebook_source(load_notebook())

    assert "def fit_temperature" in source
    assert "trainer.save_model(ARTIFACT_DIR)" in source
    assert "tokenizer.save_pretrained(ARTIFACT_DIR)" in source
    assert "AutoPeftModelForSequenceClassification.from_pretrained(" in source
    assert "distilbert-lora-confmat.svg" in source
