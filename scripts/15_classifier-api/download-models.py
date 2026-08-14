"""Populate the top-level models directory from local or Hub exports."""

import argparse
import os
from pathlib import Path
import shutil
import tempfile

from ai_detector import PROJECT_DIR, artifact_status, model_registry


HUB_MODEL_REPOSITORIES = {
    "logreg": "rasbt/ai-text-detector-logreg",
    "distilbert": "rasbt/ai-text-detector-distilbert",
    "distilbert-lora": "rasbt/ai-text-detector-distilbert-lora",
    "distilbert-mica": "rasbt/ai-text-detector-distilbert-mica",
    "modernbert": "rasbt/ai-text-detector-modernbert",
    "gpt2-variable": "rasbt/ai-text-detector-gpt2-variable",
    "gpt2-fixed": "rasbt/ai-text-detector-gpt2-fixed",
    "qwen3-variable": "rasbt/ai-text-detector-qwen3-0.6b-variable",
    "qwen3-fixed": "rasbt/ai-text-detector-qwen3-0.6b-fixed",
}

HUB_IGNORE_PATTERNS = (
    ".gitattributes",
    "LICENSE",
    "README.md",
    "figures/*",
)


def local_export_paths(project_dir):
    scripts_dir = project_dir / "scripts"
    return {
        "logreg": (
            scripts_dir
            / "05_logreg-baseline"
            / "artifacts"
            / "logreg-ai-detector.joblib"
        ),
        "distilbert": (
            scripts_dir
            / "07_distilbert"
            / "artifacts"
            / "distilbert-ai-detector"
        ),
        "distilbert-lora": (
            scripts_dir
            / "08_distilbert-lora"
            / "artifacts"
            / "distilbert-lora-ai-detector"
        ),
        "distilbert-mica": (
            scripts_dir
            / "08_distilbert-lora"
            / "artifacts"
            / "distilbert-mica-ai-detector"
        ),
        "modernbert": (
            scripts_dir
            / "09_modernbert"
            / "artifacts"
            / "modernbert-ai-detector"
        ),
        "gpt2-variable": (
            scripts_dir
            / "10_gpt2"
            / "artifacts"
            / "gpt2-variable-position-ai-detector"
        ),
        "gpt2-fixed": (
            scripts_dir
            / "10_gpt2"
            / "artifacts"
            / "gpt2-fixed-position-ai-detector"
        ),
        "qwen3-variable": (
            scripts_dir
            / "11_qwen3"
            / "artifacts"
            / "qwen3-variable-position-ai-detector"
        ),
        "qwen3-fixed": (
            scripts_dir
            / "11_qwen3"
            / "artifacts"
            / "qwen3-fixed-position-ai-detector"
        ),
    }


def _remove_existing(path):
    if path.is_dir() and not path.is_symlink():
        shutil.rmtree(path)
    elif path.exists() or path.is_symlink():
        path.unlink()


def _copy_file_atomic(source, destination):
    destination.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        dir=destination.parent,
        prefix=f".{destination.name}.",
        suffix=".tmp",
        delete=False,
    ) as temporary_file:
        temporary_path = Path(temporary_file.name)
    try:
        shutil.copy2(source, temporary_path)
        os.replace(temporary_path, destination)
    finally:
        temporary_path.unlink(missing_ok=True)


def _copy_directory_atomic(source, destination):
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary_root = Path(
        tempfile.mkdtemp(
            dir=destination.parent,
            prefix=f".{destination.name}.",
        )
    )
    staged_destination = temporary_root / destination.name
    try:
        shutil.copytree(source, staged_destination)
        _remove_existing(destination)
        os.replace(staged_destination, destination)
    finally:
        shutil.rmtree(temporary_root, ignore_errors=True)


def copy_local_exports(
    project_dir, model_names
):
    destinations = model_registry(project_dir)
    sources = local_export_paths(project_dir)
    copied = []
    unavailable = {}

    for model_name in model_names:
        destination_spec = destinations[model_name]
        source = sources[model_name]
        source_spec = destination_spec._replace(artifact_path=source)
        ready, status = artifact_status(source_spec)
        if not ready:
            unavailable[model_name] = f"{status}: {source}"
            print(f"Unavailable {model_name}: {unavailable[model_name]}")
            continue

        destination = destination_spec.artifact_path
        if source_spec.kind == "sklearn":
            _copy_file_atomic(source, destination)
            metadata_source = source.with_suffix(".json")
            if metadata_source.is_file():
                _copy_file_atomic(
                    metadata_source,
                    destination.with_suffix(".json"),
                )
        else:
            _copy_directory_atomic(source, destination)

        copied.append(model_name)
        print(f"Copied {model_name}: {source} -> {destination}")

    return copied, unavailable


def fetch_hub_exports(
    project_dir,
    model_names,
    *,
    download_snapshot=None,
):
    if download_snapshot is None:
        from huggingface_hub import snapshot_download

        download_snapshot = snapshot_download

    destinations = model_registry(project_dir)
    fetched = []
    unavailable = {}

    for model_name in model_names:
        destination_spec = destinations[model_name]
        repo_id = HUB_MODEL_REPOSITORIES[model_name]
        destination = (
            destination_spec.artifact_path.parent
            if destination_spec.kind == "sklearn"
            else destination_spec.artifact_path
        )
        destination.parent.mkdir(parents=True, exist_ok=True)
        temporary_root = Path(
            tempfile.mkdtemp(
                dir=destination.parent,
                prefix=f".{destination.name}.",
            )
        )
        staged_destination = temporary_root / destination.name

        try:
            download_snapshot(
                repo_id=repo_id,
                local_dir=staged_destination,
                ignore_patterns=HUB_IGNORE_PATTERNS,
            )
            shutil.rmtree(
                staged_destination / ".cache",
                ignore_errors=True,
            )

            staged_artifact = (
                staged_destination / destination_spec.artifact_path.name
                if destination_spec.kind == "sklearn"
                else staged_destination
            )
            staged_spec = destination_spec._replace(
                artifact_path=staged_artifact,
            )
            ready, status = artifact_status(staged_spec)
            if not ready:
                raise RuntimeError(status)

            _remove_existing(destination)
            os.replace(staged_destination, destination)
            fetched.append(model_name)
            print(f"Fetched {model_name}: {repo_id} -> {destination}")
        except Exception as error:
            unavailable[model_name] = f"{type(error).__name__}: {error}"
            print(f"Unavailable {model_name}: {unavailable[model_name]}")
        finally:
            shutil.rmtree(temporary_root, ignore_errors=True)

    return fetched, unavailable


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Populate top-level models/ from local exports or the Hub."
        )
    )
    source_group = parser.add_mutually_exclusive_group(required=True)
    source_group.add_argument(
        "--local",
        action="store_true",
        help="Copy complete exports from the local training folders.",
    )
    source_group.add_argument(
        "--fetch",
        action="store_true",
        help="Download complete exports from the Hugging Face Hub.",
    )
    parser.add_argument(
        "--model",
        action="append",
        dest="models",
        help="Select one registry model. Repeat to select multiple models.",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    registry = model_registry(PROJECT_DIR)
    model_names = args.models or list(registry)
    unknown = sorted(set(model_names) - set(registry))
    if unknown:
        available = ", ".join(registry)
        raise SystemExit(
            f"Unknown model(s): {', '.join(unknown)}. Available: {available}"
        )

    if args.local:
        copied, unavailable = copy_local_exports(PROJECT_DIR, model_names)
        print(
            f"Finished: copied {len(copied)}, "
            f"unavailable {len(unavailable)}."
        )
        if not copied:
            raise SystemExit(1)
    else:
        fetched, unavailable = fetch_hub_exports(PROJECT_DIR, model_names)
        print(
            f"Finished: fetched {len(fetched)}, "
            f"unavailable {len(unavailable)}."
        )
        if unavailable:
            raise SystemExit(1)


if __name__ == "__main__":
    main()
