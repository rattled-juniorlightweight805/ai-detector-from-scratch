
import importlib.util
import hashlib
from pathlib import Path


MODULE_PATH = (
    Path(__file__).resolve().parents[1]
    / "scripts"
    / "04_hf-dataset"
    / "04_build_hf_dataset.py"
)
SPEC = importlib.util.spec_from_file_location("build_hf_dataset", MODULE_PATH)
assert SPEC is not None and SPEC.loader is not None
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


def synthetic_records():
    records = []
    for group_index in range(20):
        collection = "source-a" if group_index < 10 else "source-b"
        group_id = f"{collection}:document-{group_index}"
        for label in ("human", "ai"):
            sample_id = group_index * 2 + (label == "ai")
            records.append(
                {
                    "id": sample_id,
                    "label": label,
                    "split": None,
                    "group_id": group_id,
                    "local_file": f"{label}/{sample_id}.txt",
                    "sha256": f"hash-{sample_id}",
                    "source_collection": collection,
                    "generator_model": "model-a" if label == "ai" else None,
                }
            )
    return records


def test_group_aware_split_has_no_lineage_overlap():
    records = MODULE.assign_splits(synthetic_records(), seed=17)
    splits_by_group = {}
    for record in records:
        splits_by_group.setdefault(record["group_id"], set()).add(record["split"])
    assert set(record["split"] for record in records) == {
        "train",
        "validation",
        "test",
    }
    assert all(len(splits) == 1 for splits in splits_by_group.values())


def test_verification_reports_balanced_splits():
    records = MODULE.assign_splits(synthetic_records(), seed=17)
    summary = MODULE.verify_records(records)
    assert summary["source_group_overlap"] is False
    assert summary["duplicate_text_hashes"] == 0
    assert summary["total_rows"] == 40
    for split in ("train", "validation", "test"):
        assert summary["splits"][split]["labels"]["human"] == summary["splits"][split]["labels"]["ai"]


def test_text_hash_accepts_both_collection_conventions(tmp_path):
    raw = b"Two words.\n"
    hashes = (
        hashlib.sha256(raw).hexdigest(),
        hashlib.sha256(raw.rstrip(b"\n")).hexdigest(),
    )
    for sample_id, expected_hash in enumerate(hashes, start=1):
        relative_file = f"human/{sample_id}.txt"
        path = tmp_path / relative_file
        path.parent.mkdir(exist_ok=True)
        path.write_bytes(raw)
        sample = {
            "id": sample_id,
            "file": relative_file,
            "label": "human",
            "collection": "test-source",
            "source_document_id": f"document-{sample_id}",
            "word_count": 2,
            "sha256": expected_hash,
            "public_hub_eligible": True,
        }
        record = MODULE.make_record(sample, {sample_id: sample}, tmp_path)
        assert record["text"] == "Two words."
        assert record["sha256"] == hashes[1]


def test_attribution_uses_explicit_values_then_source_fallbacks(tmp_path):
    raw = b"Two words.\n"
    expected_hash = hashlib.sha256(raw).hexdigest()

    fallback_path = tmp_path / "human" / "1.txt"
    fallback_path.parent.mkdir(exist_ok=True)
    fallback_path.write_bytes(raw)
    fallback_sample = {
        "id": 1,
        "file": "human/1.txt",
        "label": "human",
        "collection": "personal-blog",
        "source_document_id": "document-1",
        "source_name": "Sebastian Raschka's blog",
        "source_url": "https://example.com/article",
        "authors": ["Sebastian Raschka"],
        "word_count": 2,
        "sha256": expected_hash,
        "public_hub_eligible": True,
    }
    fallback_record = MODULE.make_record(
        fallback_sample, {1: fallback_sample}, tmp_path
    )
    assert fallback_record["attribution_name"] == "Sebastian Raschka"
    assert fallback_record["attribution_url"] == "https://example.com/article"

    explicit_path = tmp_path / "human" / "2.txt"
    explicit_path.write_bytes(raw)
    explicit_sample = {
        **fallback_sample,
        "id": 2,
        "file": "human/2.txt",
        "source_document_id": "document-2",
        "attribution_name": "Explicit attribution",
        "attribution_url": "https://example.com/attribution",
    }
    explicit_record = MODULE.make_record(explicit_sample, {2: explicit_sample}, tmp_path)
    assert explicit_record["attribution_name"] == "Explicit attribution"
    assert explicit_record["attribution_url"] == "https://example.com/attribution"
