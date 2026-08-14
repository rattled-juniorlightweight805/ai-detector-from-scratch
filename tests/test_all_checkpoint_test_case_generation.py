
import argparse
import importlib.util
from pathlib import Path
import sys


PROJECT_DIR = Path(__file__).resolve().parents[1]
SCRIPT_PATH = (
    PROJECT_DIR
    / "scripts"
    / "18_reinforcement-learning"
    / "analysis"
    / "08_generate_all_checkpoint_test_cases.py"
)
SPEC = importlib.util.spec_from_file_location("all_checkpoint_generation", SCRIPT_PATH)
assert SPEC is not None and SPEC.loader is not None
MODULE = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = MODULE
SPEC.loader.exec_module(MODULE)


def make_args(**overrides):
    values = {
        "dataset": "example/dataset",
        "device": "cuda",
        "dtype": "auto",
        "batch_size": 8,
        "temperature": 0.8,
        "top_p": 0.9,
        "max_new_tokens": 1616,
        "seed": 42,
        "max_batches": None,
    }
    values.update(overrides)
    return argparse.Namespace(**values)


def test_output_directory_uses_checkpoint_name(tmp_path):
    checkpoint = tmp_path / "checkpoints" / "step-00500"

    assert MODULE.output_dir_for_checkpoint(tmp_path, checkpoint) == (
        tmp_path / "test-cases-step-00500"
    )


def test_selects_only_exact_checkpoints_at_requested_interval(tmp_path):
    checkpoints = [
        tmp_path / "step-00001-final",
        tmp_path / "step-00050",
        tmp_path / "step-00500",
        tmp_path / "step-01000",
        tmp_path / "step-01550",
        tmp_path / "step-05000",
        tmp_path / "step-05000-final",
    ]

    assert [
        path.name for path in MODULE.select_checkpoints(checkpoints, 500)
    ] == ["step-00500", "step-01000", "step-05000"]


def test_generation_command_forwards_checkpoint_and_sampling_settings(tmp_path):
    checkpoint = tmp_path / "checkpoints" / "step-00500"
    output = tmp_path / "output"
    command = MODULE.build_generation_command(
        make_args(max_batches=1), checkpoint, output
    )

    assert command[0] == sys.executable
    assert command[2] == str(MODULE.CHECKPOINT_SCRIPT)
    assert command[command.index("--checkpoint") + 1] == str(checkpoint)
    assert command[command.index("--output-dir") + 1] == str(output)
    assert command[command.index("--batch-size") + 1] == "8"
    assert command[-2:] == ["--max-batches", "1"]
