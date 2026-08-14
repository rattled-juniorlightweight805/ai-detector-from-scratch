
import json
from pathlib import Path

import numpy as np


NOTEBOOK_PATH = (
    Path(__file__).resolve().parents[1]
    / "scripts"
    / "13_learning-curves"
    / "notebooks"
    / "distilbert-learning-curves.ipynb"
)


def load_notebook():
    return json.loads(NOTEBOOK_PATH.read_text(encoding="utf-8"))


def notebook_source():
    notebook = load_notebook()
    return "\n".join(
        "".join(cell.get("source", []))
        for cell in notebook["cells"]
    )


def test_notebook_code_compiles_and_cell_ids_are_unique():
    notebook = load_notebook()
    cell_ids = [cell["id"] for cell in notebook["cells"]]
    assert len(cell_ids) == len(set(cell_ids))

    for cell in notebook["cells"]:
        if cell["cell_type"] != "code":
            continue
        source = "".join(cell.get("source", []))
        compile(source, f"{NOTEBOOK_PATH.name}:{cell['id']}", "exec")


def test_curve_uses_distilbert_and_keeps_test_split_untouched():
    source = notebook_source()

    assert 'MODEL_NAME = "distilbert/distilbert-base-uncased"' in source
    assert 'load_dataset("rasbt/human-vs-ai-50k")' in source
    assert 'dataset["train"]' in source
    assert 'dataset["validation"]' in source
    assert 'dataset["test"]' not in source
    assert 'tokenized_dataset["test"]' not in source


def test_training_subsets_are_nested_and_distribution_aware():
    source = notebook_source()

    assert "TRAIN_FRACTIONS = np.asarray(" in source
    assert "def make_distribution_aware_order(" in source
    assert "source_collections" in source
    assert "labels" in source
    assert "nested_order[:sample_count]" in source
    assert "assert previous_indices.issubset(current_indices)" in source


def test_distribution_aware_order_preserves_small_prefix_balance():
    notebook = load_notebook()
    cell = next(
        cell for cell in notebook["cells"] if cell["id"] == "nested-subsets"
    )
    function_source = "".join(cell["source"]).split(
        "nested_order =", maxsplit=1
    )[0]
    namespace = {"np": np}
    exec(function_source, namespace)

    labels = np.asarray([0] * 60 + [1] * 40)
    sources = np.asarray(
        ["a"] * 45 + ["b"] * 15 + ["a"] * 30 + ["b"] * 10
    )
    order = namespace["make_distribution_aware_order"](
        labels, sources, seed=17
    )

    assert sorted(order.tolist()) == list(range(100))
    strata = np.asarray(
        [f"{label}::{source}" for label, source in zip(labels, sources)]
    )
    for prefix_size in (10, 25, 50):
        prefix_strata = strata[order[:prefix_size]]
        for stratum in np.unique(strata):
            expected = prefix_size * np.mean(strata == stratum)
            actual = np.sum(prefix_strata == stratum)
            assert abs(actual - expected) <= 1


def test_each_curve_point_starts_from_fresh_model_weights():
    source = notebook_source()
    loop_start = source.index("for sample_count in train_sizes:")
    loop_source = source[loop_start:]

    assert "set_seed(RANDOM_STATE)" in loop_source
    assert "AutoModelForSequenceClassification.from_pretrained(" in loop_source
    assert "num_train_epochs=NUM_TRAIN_EPOCHS" in source
    assert 'eval_strategy="no"' in source
    assert 'save_strategy="no"' in source
    assert "EarlyStoppingCallback" not in source


def test_notebook_evaluates_and_plots_train_and_validation_accuracy():
    source = notebook_source()

    assert "trainer.predict(train_subset)" in source
    assert "validation_output = trainer.predict(" in source
    assert 'tokenized_dataset["validation"]' in source
    assert '"training_accuracy"' in source
    assert '"validation_accuracy"' in source
    assert "distilbert-learning-curve-results.csv" in source
    assert "distilbert-learning-curves.svg" in source
    assert 'ax.set_xlabel("Training data used")' in source
    assert 'ax.set_ylabel("Accuracy")' in source
