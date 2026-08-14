import re
import tomllib
from pathlib import Path


PYPROJECT_PATH = Path(__file__).resolve().parents[1] / "pyproject.toml"


def load_pyproject():
    with PYPROJECT_PATH.open("rb") as file:
        return tomllib.load(file)


def dependency_names(dependencies):
    return {
        re.split(r"[<>=!~@\s]", dependency, maxsplit=1)[0]
        for dependency in dependencies
        if isinstance(dependency, str)
    }


def included_groups(dependencies):
    return {
        dependency["include-group"]
        for dependency in dependencies
        if isinstance(dependency, dict)
    }


def test_default_environment_is_limited_to_logreg_inference():
    config = load_pyproject()
    dependencies = dependency_names(config["project"]["dependencies"])

    assert {"huggingface-hub", "numpy", "pyyaml", "scikit-learn"} <= dependencies
    assert dependencies.isdisjoint(
        {
            "accelerate",
            "datasets",
            "evaluate",
            "fastapi",
            "peft",
            "torch",
            "transformers",
            "uvicorn",
        }
    )
    assert config["tool"]["uv"]["default-groups"] == []


def test_feature_groups_keep_heavy_dependencies_explicit():
    groups = load_pyproject()["dependency-groups"]

    assert {"peft", "torch", "transformers"} <= dependency_names(
        groups["transformer-inference"]
    )
    assert "transformer-inference" in included_groups(groups["training"])
    assert {"accelerate", "datasets", "evaluate"} <= dependency_names(
        groups["training"]
    )
    assert {"jupyterlab", "matplotlib", "ipywidgets"} <= dependency_names(
        groups["notebooks"]
    )
    assert {"fastapi", "uvicorn"} <= dependency_names(groups["browser-ui"])


def test_dev_group_collects_the_reproduction_environment():
    groups = load_pyproject()["dependency-groups"]

    assert included_groups(groups["dev"]) == {
        "training",
        "notebooks",
        "browser-ui",
    }
    assert {"httpx", "pytest"} <= dependency_names(groups["dev"])
