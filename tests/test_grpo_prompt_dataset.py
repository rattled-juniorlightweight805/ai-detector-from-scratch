
import importlib.util
from pathlib import Path
import sys


PROJECT_DIR = Path(__file__).resolve().parents[1]
SCRIPT_DIR = PROJECT_DIR / "scripts" / "18_reinforcement-learning"


def load_module(name, filename):
    spec = importlib.util.spec_from_file_location(name, SCRIPT_DIR / filename)
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def sample_rows():
    rows = []
    for index in range(20):
        rows.append(
            {
                "id": index + 1,
                "source_collection": "blog" if index < 12 else "arxiv",
                "target_words": 100 if index % 2 else 500,
                "word_count": 100 if index % 2 else 500,
            }
        )
    return rows


def test_seed_selection_is_exact_stratified_and_deterministic():
    module = load_module("select_grpo_prompts", "01_select_prompt_seeds.py")
    rows = sample_rows()

    first = module.select_rows(rows, 10, split="train", seed=17)
    second = module.select_rows(rows, 10, split="train", seed=17)

    assert len(first) == 10
    assert [row["id"] for row in first] == [row["id"] for row in second]
    assert sum(row["source_collection"] == "blog" for row in first) == 6
    assert sum(row["source_collection"] == "arxiv" for row in first) == 4


def test_question_validation_rejects_generation_artifacts():
    module = load_module("generate_grpo_prompts", "02_generate_ollama_prompts.py")

    valid, _ = module.question_is_valid(
        "How does attention help a model relate information across a sequence?"
    )
    missing_mark, _ = module.question_is_valid(
        "How does attention help a model relate information across a sequence"
    )
    artifact, _ = module.question_is_valid(
        "What does the provided passage say about attention mechanisms?"
    )

    assert valid
    assert not missing_mark
    assert not artifact


def test_inspiration_has_word_and_character_limits():
    module = load_module("generate_grpo_prompt_limits", "02_generate_ollama_prompts.py")
    text = "longword " * 1_000
    inspiration = module.inspiration_text(text, max_words=160, max_chars=400)

    assert len(inspiration.split()) <= 160
    assert len(inspiration) <= 400


def test_question_construction_is_deterministic_and_valid():
    module = load_module("generate_grpo_prompt_templates", "02_generate_ollama_prompts.py")

    first, first_index = module.build_question("software upgrade safety", 500, 17)
    second, second_index = module.build_question("software upgrade safety", 500, 17)
    valid, _ = module.question_is_valid(first)

    assert first == second
    assert first_index == second_index
    assert "software upgrade safety" in first
    assert valid


def test_finalizer_rotates_away_from_an_existing_question():
    finalizer = load_module("finalize_grpo_prompts", "03_finalize_prompt_set.py")
    generator = finalizer.load_generator()
    topic = "software upgrade safety"
    existing = generator.LONG_QUESTION_TEMPLATES[0].format(topic=topic)

    question, template_index = finalizer.choose_unique_question(
        topic,
        500,
        0,
        {existing},
        generator,
    )

    assert template_index == 1
    assert question != existing
