
import json
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
NOTEBOOK_PATH = (
    PROJECT_ROOT
    / "scripts"
    / "08_distilbert-lora"
    / "distilbert-mica.ipynb"
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
    cell_ids = [cell["id"] for cell in notebook["cells"]]
    assert len(cell_ids) == len(set(cell_ids))

    for cell in notebook["cells"]:
        if cell["cell_type"] != "code":
            continue
        source = "".join(cell.get("source", []))
        compile(source, f"{NOTEBOOK_PATH.name}:{cell['id']}", "exec")


def test_mica_uses_minor_component_initialization():
    source = notebook_source(load_notebook())

    assert 'init_lora_weights="mica"' in source
    assert 'MICA_TARGET_MODULES = ["q_lin", "v_lin"]' in source
    assert 'MODULES_TO_SAVE = ["pre_classifier", "classifier"]' in source
    assert "task_type=TaskType.SEQ_CLS" in source
    assert "r=MICA_RANK" in source
    assert "model = get_peft_model(base_model, mica_config)" in source


def test_mica_checks_that_only_a_is_trainable():
    source = notebook_source(load_notebook())

    assert 'if "lora_A" in name' in source
    assert 'if "lora_B" in name' in source
    assert "all(parameter.requires_grad for parameter in mica_a_parameters)" in source
    assert "all(not parameter.requires_grad for parameter in mica_b_parameters)" in source
    assert "torch.count_nonzero(parameter).item() == 0" in source


def test_mica_uses_recommended_starting_point_from_lora_baseline():
    source = notebook_source(load_notebook())

    assert "MICA_RANK = 4" in source
    assert "learning_rate=2e-4" in source


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
    assert "distilbert-mica-confmat.svg" in source
