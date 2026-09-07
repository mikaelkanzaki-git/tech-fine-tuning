from __future__ import annotations

import json
from pathlib import Path

import pytest

from tech_fine_tuning.errors import SftCurationValidationError
from tech_fine_tuning.integrations.dataset.jsonl import file_sha256, write_json, write_jsonl
from tech_fine_tuning.services.sft_curation_service import curate_sft_dataset


def _example(record_id: str, question: str, answer: str) -> dict[str, object]:
    return {
        "messages": [
            {"role": "system", "content": "Safe medical assistant."},
            {"role": "user", "content": question},
            {"role": "assistant", "content": answer},
        ],
        "metadata": {
            "record_id": record_id,
            "source": {"relative_path": f"collection/{record_id}.xml"},
        },
    }


def _write_sft(root: Path) -> Path:
    source = root / "sft"
    records = {
        "train": [
            _example(
                "train-hpo",
                "What are the symptoms ?",
                (
                    "What are the symptoms? Fever and cough. - The Human Phenotype "
                    "Ontology (HPO) has collected information on how often a sign or "
                    "symptom occurs in a condition."
                ),
            ),
            _example(
                "train-resource",
                "What are the treatments?",
                (
                    "What are the treatments? These resources address the diagnosis "
                    "or management of the condition."
                ),
            ),
        ],
        "validation": [_example("validation-1", "Validation question?", "Answer.")],
        "test": [_example("test-1", "Test question?", "Answer.")],
    }
    hashes: dict[str, str] = {}
    for split, examples in records.items():
        path = source / f"{split}.jsonl"
        write_jsonl(path, examples)
        hashes[split] = file_sha256(path)
    write_json(
        source / "manifest.json",
        {
            "schema_version": "1.0",
            "format": {"type": "conversational_messages"},
            "summary": {
                "examples": 4,
                "examples_by_split": {
                    split: len(examples) for split, examples in records.items()
                },
            },
            "output_files_sha256": hashes,
        },
    )
    return source


def test_curation_removes_non_instructional_text_with_traceability(tmp_path: Path) -> None:
    source = _write_sft(tmp_path)

    outcome = curate_sft_dataset(source, tmp_path / "curated")

    assert outcome.summary == {
        "input_examples": 4,
        "output_examples": 3,
        "excluded_examples": 1,
        "examples_by_split": {"train": 1, "validation": 1, "test": 1},
        "transformations": {
            "hpo_boilerplate_removed": 1,
            "leading_question_removed": 2,
        },
        "exclusions": {"resource_navigation_answer": 1},
    }
    train = json.loads((outcome.output_path / "train.jsonl").read_text(encoding="utf-8"))
    assert train["messages"][2]["content"] == "Fever and cough."
    assert train["metadata"]["curation"] == {
        "profile": "medquad_conservative_v1",
        "transformations": [
            "leading_question_removed",
            "hpo_boilerplate_removed",
        ],
    }
    excluded = json.loads(outcome.excluded_path.read_text(encoding="utf-8"))
    assert excluded == {
        "reasons": ["resource_navigation_answer"],
        "record_id": "train-resource",
        "split": "train",
    }
    manifest = json.loads(outcome.manifest_path.read_text(encoding="utf-8"))
    assert manifest["validation"]["cross_split_normalized_question_overlap"] == 0
    assert manifest["curation"]["medical_text_policy"] == (
        "remove_only_no_medical_facts_synthesized"
    )


def test_curation_rejects_cross_split_questions(tmp_path: Path) -> None:
    source = _write_sft(tmp_path)
    validation_path = source / "validation.jsonl"
    write_jsonl(
        validation_path,
        [_example("validation-1", " what are the symptoms ? ", "Another answer.")],
    )
    manifest = json.loads((source / "manifest.json").read_text(encoding="utf-8"))
    manifest["output_files_sha256"]["validation"] = file_sha256(validation_path)
    write_json(source / "manifest.json", manifest)

    with pytest.raises(SftCurationValidationError, match="perguntas normalizadas"):
        curate_sft_dataset(source, tmp_path / "curated")


def test_curation_rejects_tampered_source_and_nonempty_output(tmp_path: Path) -> None:
    source = _write_sft(tmp_path)
    with (source / "train.jsonl").open("a", encoding="utf-8") as stream:
        stream.write("\n")
    with pytest.raises(SftCurationValidationError, match="hash do split"):
        curate_sft_dataset(source, tmp_path / "curated")

    clean_source = _write_sft(tmp_path / "clean")
    output = tmp_path / "occupied"
    output.mkdir()
    (output / "keep.txt").write_text("keep", encoding="utf-8")
    with pytest.raises(SftCurationValidationError, match="não está vazio"):
        curate_sft_dataset(clean_source, output)
