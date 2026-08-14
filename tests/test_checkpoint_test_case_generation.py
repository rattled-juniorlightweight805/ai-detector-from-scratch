
import argparse
import importlib.util
from pathlib import Path
import sys

from safetensors.torch import save_file
import torch


PROJECT_DIR = Path(__file__).resolve().parents[1]
SCRIPT_PATH = (
    PROJECT_DIR
    / "scripts"
    / "18_reinforcement-learning"
    / "analysis"
    / "07_generate_checkpoint_test_cases.py"
)
SPEC = importlib.util.spec_from_file_location("checkpoint_generation", SCRIPT_PATH)
assert SPEC is not None and SPEC.loader is not None
MODULE = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = MODULE
SPEC.loader.exec_module(MODULE)


def make_checkpoint(root, name):
    checkpoint = root / name
    checkpoint.mkdir()
    (checkpoint / "config.json").write_text("{}\n", encoding="utf-8")
    (checkpoint / "tokenizer.json").write_text("{}\n", encoding="utf-8")
    save_file({"weight": torch.ones(1)}, checkpoint / "model.safetensors")
    return checkpoint


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


def test_checkpoint_discovery_only_returns_complete_exports(tmp_path):
    first = make_checkpoint(tmp_path, "step-00100")
    second = make_checkpoint(tmp_path, "step-00050")
    incomplete = tmp_path / "step-00150"
    incomplete.mkdir()
    (incomplete / "config.json").write_text("{}\n", encoding="utf-8")

    assert MODULE.available_checkpoints(tmp_path) == [second, first]


def test_checkpoint_can_be_resolved_by_name_or_absolute_path(tmp_path):
    checkpoint = make_checkpoint(tmp_path, "step-00500")

    assert MODULE.resolve_checkpoint("step-00500", tmp_path) == checkpoint
    assert MODULE.resolve_checkpoint(str(checkpoint), tmp_path) == checkpoint


def test_corrupted_safetensors_checkpoint_is_rejected(tmp_path):
    checkpoint = make_checkpoint(tmp_path, "step-00500")
    (checkpoint / "model.safetensors").write_bytes(b"partial sync")

    problem = MODULE.checkpoint_problem(checkpoint)

    assert problem is not None
    assert "model.safetensors is invalid" in problem


def test_default_output_directory_uses_checkpoint_name(tmp_path):
    checkpoint = make_checkpoint(tmp_path, "step-00500")

    assert MODULE.default_output_dir(checkpoint).name == "test-cases-step-00500"


def test_generation_command_forwards_checkpoint_and_sampling_settings(tmp_path):
    checkpoint = make_checkpoint(tmp_path, "step-00500")
    output = tmp_path / "output"
    command = MODULE.build_generation_command(
        make_args(max_batches=1), checkpoint, output
    )

    assert command[0] == sys.executable
    assert command[2] == str(MODULE.BASELINE_SCRIPT)
    assert command[command.index("--policy-model") + 1] == str(checkpoint)
    assert command[command.index("--output-dir") + 1] == str(output)
    assert command[command.index("--batch-size") + 1] == "8"
    assert command[-2:] == ["--max-batches", "1"]
