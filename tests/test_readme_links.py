import re
from pathlib import Path


PROJECT_DIR = Path(__file__).resolve().parents[1]
STAGE_DIRS = [
    "01_human-data",
    "02_ai-data",
    "03_ai-assisted-data",
    "04_hf-dataset",
    "05_logreg-baseline",
    "06_chatgpt-baseline",
    "07_distilbert",
    "08_distilbert-lora",
    "09_modernbert",
    "10_gpt2",
    "11_qwen3",
    "12_length-bias",
    "13_learning-curves",
    "14_resource-latency",
    "15_classifier-api",
    "16_case-study-substack",
    "17_browser-ui",
    "18_reinforcement-learning",
]
READER_READMES = [
    PROJECT_DIR / "README.md",
    PROJECT_DIR / "scripts" / "01_human-data" / "README.md",
    PROJECT_DIR / "scripts" / "02_ai-data" / "README.md",
    PROJECT_DIR / "scripts" / "02_ai-data" / "pangram-check" / "README.md",
    PROJECT_DIR / "scripts" / "03_ai-assisted-data" / "README.md",
    PROJECT_DIR / "scripts" / "03_ai-assisted-data" / "pangram-check" / "README.md",
    PROJECT_DIR / "scripts" / "04_hf-dataset" / "README.md",
    PROJECT_DIR / "scripts" / "05_logreg-baseline" / "README.md",
    PROJECT_DIR / "scripts" / "13_learning-curves" / "README.md",
    PROJECT_DIR / "scripts" / "14_resource-latency" / "README.md",
    PROJECT_DIR / "scripts" / "15_classifier-api" / "README.md",
    PROJECT_DIR / "scripts" / "16_case-study-substack" / "README.md",
    PROJECT_DIR / "scripts" / "17_browser-ui" / "README.md",
    PROJECT_DIR / "scripts" / "18_reinforcement-learning" / "README.md",
]
LINK_PATTERN = re.compile(r"\[[^\]]+\]\(([^)]+)\)")
HEADING_PATTERN = re.compile(r"^ {0,3}#{1,6}\s")


def github_readmes():
    excluded_parts = {
        ".git",
        ".pytest_cache",
        ".venv",
        "dist",
        "models",
        "node_modules",
    }
    return [
        path
        for path in PROJECT_DIR.rglob("README.md")
        if not excluded_parts.intersection(path.relative_to(PROJECT_DIR).parts)
    ]


def test_root_readme_links_every_numbered_stage():
    source = (PROJECT_DIR / "README.md").read_text(encoding="utf-8")

    for stage_dir in STAGE_DIRS:
        assert f"(scripts/{stage_dir}/)" in source


def test_stage_readmes_link_back_to_the_project_index():
    for readme_path in READER_READMES[1:]:
        source = readme_path.read_text(encoding="utf-8")
        assert "Project index" in source


def test_reader_readme_relative_links_resolve():
    missing = []
    for readme_path in READER_READMES:
        source = readme_path.read_text(encoding="utf-8")
        for target in LINK_PATTERN.findall(source):
            if "://" in target or target.startswith("#"):
                continue
            relative_target = target.split("#", maxsplit=1)[0]
            resolved = (readme_path.parent / relative_target).resolve()
            if not resolved.exists():
                missing.append(f"{readme_path.relative_to(PROJECT_DIR)} -> {target}")

    assert not missing, "Missing README links:\n" + "\n".join(missing)


def test_github_readme_headings_have_nbsp_spacers():
    missing = []
    for readme_path in github_readmes():
        lines = readme_path.read_text(encoding="utf-8").splitlines()
        in_code_fence = False
        for index, line in enumerate(lines):
            stripped = line.lstrip()
            if stripped.startswith(("```", "~~~")):
                in_code_fence = not in_code_fence
                continue
            if in_code_fence or not HEADING_PATTERN.match(line):
                continue
            if index == 0 or lines[index - 1].strip() != "&nbsp;":
                missing.append(
                    f"{readme_path.relative_to(PROJECT_DIR)}:{index + 1}"
                )

    assert not missing, "Headings without &nbsp;:\n" + "\n".join(missing)
