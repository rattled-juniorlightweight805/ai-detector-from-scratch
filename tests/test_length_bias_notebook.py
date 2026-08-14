
import json
from pathlib import Path


NOTEBOOK_PATH = (
    Path(__file__).resolve().parents[1]
    / "scripts"
    / "12_length-bias"
    / "length-bias.ipynb"
)


def load_notebook():
    return json.loads(NOTEBOOK_PATH.read_text(encoding="utf-8"))


def notebook_source():
    return "\n".join(
        "".join(cell.get("source", []))
        for cell in load_notebook()["cells"]
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


def test_notebook_uses_all_local_classifier_variants():
    source = notebook_source()
    for model_name in (
        "logreg",
        "distilbert",
        "distilbert-lora",
        "distilbert-mica",
        "modernbert",
        "gpt2-variable",
        "gpt2-fixed",
        "qwen3-variable",
        "qwen3-fixed",
    ):
        assert f'"{model_name}"' in source

    assert "artifact_status(spec)" in source
    assert "Unavailable models will be reported" in source


def test_analysis_uses_only_the_held_out_test_split():
    source = notebook_source()

    assert 'dataset = load_dataset("rasbt/human-vs-ai-50k")' in source
    assert 'test_dataset = dataset["test"]' in source
    assert '["train"]' not in source
    assert '["validation"]' not in source


def test_length_bins_match_the_dataset_generation_ranges():
    source = notebook_source()

    assert "LENGTH_BIN_EDGES = [0, 60, 120, 300, 600, np.inf]" in source
    assert '"≤60"' in source
    assert '"61–120"' in source
    assert '"121–300"' in source
    assert '"301–600"' in source
    assert '"601+"' in source


def test_inference_is_cached_and_validated_against_artifact():
    source = notebook_source()

    assert "def artifact_signature(" in source
    assert "REUSE_CACHED_PREDICTIONS = True" in source
    assert "cached_ids != expected_ids" in source
    assert '"artifact_signature"' in source
    assert "classifier.score_many(" in source


def test_summary_distinguishes_score_drift_and_error_rates():
    source = notebook_source()

    assert '"mean_human_score"' in source
    assert '"mean_ai_score"' in source
    assert '"false_positive_rate"' in source
    assert '"false_negative_rate"' in source
    assert '"score_separation"' in source
    assert '"human_length_correlation"' in source
    assert '"ai_length_correlation"' in source


def test_notebook_exports_restrained_small_multiple_figures():
    source = notebook_source()

    assert "length-bias-scores.svg" in source
    assert "length-bias-error-rates.svg" in source
    assert "ax.spines[[\"top\", \"right\"]].set_visible(False)" in source
    assert "ax.legend(" not in source
    assert "fig.legend(" not in source


def test_notebook_has_no_stored_outputs():
    code_cells = [
        cell
        for cell in load_notebook()["cells"]
        if cell["cell_type"] == "code"
    ]
    assert all(cell["execution_count"] is None for cell in code_cells)
    assert all(cell["outputs"] == [] for cell in code_cells)
