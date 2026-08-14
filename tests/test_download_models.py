
import importlib.util
from pathlib import Path
import sys


API_DIR = (
    Path(__file__).resolve().parents[1]
    / "scripts"
    / "15_classifier-api"
)
MODULE_PATH = API_DIR / "download-models.py"


def load_module():
    if str(API_DIR) not in sys.path:
        sys.path.insert(0, str(API_DIR))
    spec = importlib.util.spec_from_file_location(
        "download_models", MODULE_PATH
    )
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def test_local_copy_uses_top_level_models_and_skips_incomplete(
    tmp_path,
):
    module = load_module()
    logreg_export = (
        tmp_path
        / "scripts"
        / "05_logreg-baseline"
        / "artifacts"
        / "logreg-ai-detector.joblib"
    )
    logreg_export.parent.mkdir(parents=True)
    logreg_export.write_bytes(b"joblib")
    logreg_export.with_suffix(".json").write_text(
        '{"model": "logreg"}', encoding="utf-8"
    )

    distilbert_export = (
        tmp_path
        / "scripts"
        / "07_distilbert"
        / "artifacts"
        / "distilbert-ai-detector"
    )
    distilbert_export.mkdir(parents=True)
    (distilbert_export / "detector-config.json").write_text(
        "{}", encoding="utf-8"
    )
    (distilbert_export / "model.safetensors").write_bytes(b"weights")

    copied, unavailable = module.copy_local_exports(
        tmp_path,
        ["logreg", "distilbert", "modernbert"],
    )

    assert copied == ["logreg", "distilbert"]
    assert list(unavailable) == ["modernbert"]
    assert (
        tmp_path / "models" / "logreg" / "logreg-ai-detector.joblib"
    ).read_bytes() == b"joblib"
    assert (
        tmp_path / "models" / "logreg" / "logreg-ai-detector.json"
    ).is_file()
    assert (
        tmp_path
        / "models"
        / "distilbert"
        / "detector-config.json"
    ).is_file()


def test_directory_copy_replaces_stale_destination(tmp_path):
    module = load_module()
    source = (
        tmp_path
        / "scripts"
        / "07_distilbert"
        / "artifacts"
        / "distilbert-ai-detector"
    )
    source.mkdir(parents=True)
    (source / "detector-config.json").write_text("{}", encoding="utf-8")
    (source / "model.safetensors").write_bytes(b"new")

    destination = tmp_path / "models" / "distilbert"
    destination.mkdir(parents=True)
    (destination / "stale.bin").write_bytes(b"old")

    copied, unavailable = module.copy_local_exports(
        tmp_path, ["distilbert"]
    )

    assert copied == ["distilbert"]
    assert unavailable == {}
    assert not (destination / "stale.bin").exists()
    assert (destination / "model.safetensors").read_bytes() == b"new"


def test_hub_repository_registry_matches_model_registry():
    module = load_module()

    assert set(module.HUB_MODEL_REPOSITORIES) == set(
        module.model_registry(Path("/project"))
    )


def test_hub_fetch_validates_and_installs_complete_exports(
    tmp_path,
):
    module = load_module()
    calls = []

    def fake_download_snapshot(**kwargs):
        calls.append(kwargs)
        destination = Path(kwargs["local_dir"])
        destination.mkdir(parents=True)
        (destination / ".cache").mkdir()
        (destination / ".cache" / "download-state").write_text(
            "temporary", encoding="utf-8"
        )
        if kwargs["repo_id"].endswith("-logreg"):
            (destination / "logreg-ai-detector.joblib").write_bytes(
                b"joblib"
            )
            (destination / "logreg-ai-detector.json").write_text(
                "{}", encoding="utf-8"
            )
        else:
            (destination / "detector-config.json").write_text(
                "{}", encoding="utf-8"
            )
            (destination / "model.safetensors").write_bytes(b"weights")
        return str(destination)

    stale_destination = tmp_path / "models" / "distilbert"
    stale_destination.mkdir(parents=True)
    (stale_destination / "stale.bin").write_bytes(b"old")

    fetched, unavailable = module.fetch_hub_exports(
        tmp_path,
        ["logreg", "distilbert"],
        download_snapshot=fake_download_snapshot,
    )

    assert fetched == ["logreg", "distilbert"]
    assert unavailable == {}
    assert [call["repo_id"] for call in calls] == [
        "rasbt/ai-text-detector-logreg",
        "rasbt/ai-text-detector-distilbert",
    ]
    assert all(
        call["ignore_patterns"] == module.HUB_IGNORE_PATTERNS
        for call in calls
    )
    assert (
        tmp_path / "models" / "logreg" / "logreg-ai-detector.joblib"
    ).read_bytes() == b"joblib"
    assert not (tmp_path / "models" / "logreg" / ".cache").exists()
    assert not (stale_destination / "stale.bin").exists()
    assert (
        stale_destination / "model.safetensors"
    ).read_bytes() == b"weights"


def test_incomplete_hub_fetch_preserves_ready_destination(
    tmp_path,
):
    module = load_module()
    destination = tmp_path / "models" / "modernbert"
    destination.mkdir(parents=True)
    (destination / "detector-config.json").write_text(
        "{}", encoding="utf-8"
    )
    (destination / "model.safetensors").write_bytes(b"existing")

    def fake_incomplete_download(**kwargs):
        staged = Path(kwargs["local_dir"])
        staged.mkdir(parents=True)
        (staged / "detector-config.json").write_text(
            "{}", encoding="utf-8"
        )
        return str(staged)

    fetched, unavailable = module.fetch_hub_exports(
        tmp_path,
        ["modernbert"],
        download_snapshot=fake_incomplete_download,
    )

    assert fetched == []
    assert "model weights are missing" in unavailable["modernbert"]
    assert (destination / "model.safetensors").read_bytes() == b"existing"
