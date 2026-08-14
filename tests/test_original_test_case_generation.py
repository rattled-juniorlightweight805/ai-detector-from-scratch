
import importlib.util
from pathlib import Path
import sys

import torch


PROJECT_DIR = Path(__file__).resolve().parents[1]
SCRIPT_PATH = (
    PROJECT_DIR
    / "scripts"
    / "18_reinforcement-learning"
    / "analysis"
    / "06_generate_original_test_cases.py"
)
SPEC = importlib.util.spec_from_file_location("original_generation", SCRIPT_PATH)
assert SPEC is not None and SPEC.loader is not None
MODULE = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = MODULE
SPEC.loader.exec_module(MODULE)


def make_row(prompt_id, target_words):
    return {
        "prompt_id": prompt_id,
        "prompt": "What should this answer discuss?",
        "prompt_sha256": f"hash-{prompt_id}",
        "target_words": target_words,
        "group_id": f"group-{prompt_id}",
        "source_collection": "test-source",
        "source_document_id": f"document-{prompt_id}",
    }


def run_config():
    return {
        "dataset": "example/dataset",
        "policy_model": "example/model",
        "temperature": 0.8,
        "top_p": 0.9,
        "max_new_tokens": 1616,
        "seed": 42,
        "batch_size": 2,
        "dtype": "auto",
    }


def test_batches_share_target_length_and_have_stable_indices():
    rows = [
        make_row("test-1", 100),
        make_row("test-2", 50),
        make_row("test-3", 100),
        make_row("test-4", 100),
    ]

    batches = list(MODULE.stable_batches(rows, batch_size=2))

    assert [(index, [row["prompt_id"] for row in batch]) for index, batch in batches] == [
        (0, ["test-2"]),
        (1, ["test-1", "test-3"]),
        (2, ["test-4"]),
    ]
    assert all(len({row["target_words"] for row in batch}) == 1 for _, batch in batches)


def test_saved_response_can_be_verified_and_resumed(tmp_path):
    row = make_row("test-1", 100)
    config = run_config()

    saved = MODULE.save_response(
        tmp_path,
        row,
        "A complete baseline response.",
        5,
        batch_seed=42,
        token_limit=176,
        run_config=config,
        elapsed_seconds=0.5,
    )
    loaded = MODULE.load_existing_record(tmp_path, row, config)

    assert loaded == saved
    assert (tmp_path / saved["response_file"]).read_text(encoding="utf-8") == (
        "A complete baseline response.\n"
    )


def test_empty_response_is_recorded_and_can_be_resumed(tmp_path):
    row = make_row("test-1", 100)
    config = run_config()

    saved = MODULE.save_response(
        tmp_path,
        row,
        "",
        0,
        batch_seed=42,
        token_limit=176,
        run_config=config,
        elapsed_seconds=0.5,
    )
    loaded = MODULE.load_existing_record(tmp_path, row, config)

    assert loaded == saved
    assert saved["response"] == ""
    assert saved["response_word_count"] == 0
    assert saved["response_token_count"] == 0
    assert (tmp_path / saved["response_file"]).read_text(encoding="utf-8") == "\n"


def test_response_with_crlf_is_verified_without_newline_normalization(tmp_path):
    row = make_row("test-1", 100)
    config = run_config()
    response = "First line.\r\nSecond line."

    saved = MODULE.save_response(
        tmp_path,
        row,
        response,
        6,
        batch_seed=42,
        token_limit=176,
        run_config=config,
        elapsed_seconds=0.5,
    )

    assert MODULE.load_existing_record(tmp_path, row, config) == saved


def test_auto_dtype_uses_float16_on_mps_and_bfloat16_on_cuda():
    assert MODULE.dtype_for_device(torch.device("mps"), "auto") == torch.float16
    assert MODULE.dtype_for_device(torch.device("cuda"), "auto") == torch.bfloat16
