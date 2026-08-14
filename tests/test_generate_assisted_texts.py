
import importlib.util
import json
from pathlib import Path
from types import SimpleNamespace


MODULE_PATH = (
    Path(__file__).resolve().parents[1]
    / "scripts"
    / "03_ai-assisted-data"
    / "_generate_assisted_texts.py"
)
SPEC = importlib.util.spec_from_file_location("generate_assisted_texts", MODULE_PATH)
assert SPEC is not None and SPEC.loader is not None
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


def test_normalize_edit_removes_wrapper_without_changing_body():
    text = "```text\nEdited text: This sentence is correct.\n```"
    assert MODULE.normalize_edit(text) == "This sentence is correct."


def test_light_edit_accepts_a_local_correction():
    original = "This are a short sentence with one obvious grammar error."
    edited = "This is a short sentence with one obvious grammar error."
    valid, reason, ratio = MODULE.validate_edit(original, edited, "light")
    assert valid, reason
    assert ratio > 0.82


def test_light_edit_rejects_an_unchanged_sample():
    original = "This sentence is already correct."
    valid, reason, ratio = MODULE.validate_edit(original, original, "light")
    assert not valid
    assert reason == "is unchanged"
    assert ratio == 1.0


def test_light_edit_rejects_a_large_rewrite():
    original = "The model reads tokens and produces a contextual representation."
    edited = (
        "A neural system interprets an input sequence before returning useful "
        "features."
    )
    valid, reason, _ = MODULE.validate_edit(original, edited, "light")
    assert not valid
    assert "rewrites too much" in reason


def test_moderate_edit_rejects_a_nearly_unchanged_sample():
    original = " ".join(f"word{index}" for index in range(100))
    edited = original.replace("word50", "replacement50")
    valid, reason, _ = MODULE.validate_edit(original, edited, "moderate")
    assert not valid
    assert "changes too little" in reason


def test_moderate_edit_accepts_sentence_restructuring():
    original = (
        "The model reads each token in sequence. It then creates a contextual "
        "representation for every position. These representations are passed to "
        "the next layer for additional processing."
    )
    edited = (
        "After reading the token sequence, the model builds a contextual "
        "representation at each position. The next layer then processes these "
        "representations further."
    )
    valid, reason, ratio = MODULE.validate_edit(original, edited, "moderate")
    assert valid, reason
    assert 0.45 <= ratio <= 0.97


def test_character_similarity_detects_a_punctuation_edit():
    original = "However the result was useful."
    edited = "However, the result was useful."
    ratio = MODULE.character_similarity_ratio(original, edited)
    assert 0.95 < ratio < 1.0


def test_model_jobs_claim_disjoint_human_sources(tmp_path):
    human_dir = tmp_path / "human"
    human_dir.mkdir()
    samples = []
    for sample_id in range(1, 11):
        relative_file = f"human/{sample_id}.txt"
        (tmp_path / relative_file).write_text(
            f"Human sample {sample_id}.\n", encoding="utf-8"
        )
        samples.append(
            {
                "id": sample_id,
                "file": relative_file,
                "label": "human",
            }
        )
    meta_path = tmp_path / "meta.json"
    meta_path.write_text(json.dumps({"samples": samples}), encoding="utf-8")

    claimed_sets = []
    for job_index in range(5):
        args = SimpleNamespace(
            provider=f"provider{job_index}",
            model=f"model{job_index}",
            meta=meta_path,
            count=2,
            seed=17,
        )
        claimed, _ = MODULE.claim_seed_samples(args)
        claimed_sets.append({int(sample["id"]) for sample in claimed})

    assert set().union(*claimed_sets) == set(range(1, 11))
    assert sum(map(len, claimed_sets)) == 10
    for index, claimed in enumerate(claimed_sets):
        for other in claimed_sets[index + 1 :]:
            assert claimed.isdisjoint(other)
