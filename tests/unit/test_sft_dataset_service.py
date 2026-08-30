from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any
from uuid import NAMESPACE_URL, uuid5

import pytest

from tech_fine_tuning.errors import SftPreparationError
from tech_fine_tuning.integrations.dataset.jsonl import write_json, write_jsonl
from tech_fine_tuning.services.sft_dataset_service import prepare_sft_dataset


def canonical_record(
    index: int,
    *,
    document_index: int | None = None,
    pii_status: str = "not_detected",
) -> dict[str, Any]:
    document_number = document_index if document_index is not None else index
    question = f"Question {index}?"
    answer = f"Answer {index}."
    return {
        "schema_version": "1.1",
        "record_id": str(uuid5(NAMESPACE_URL, f"record-{index}")),
        "document_id": str(uuid5(NAMESPACE_URL, f"document-{document_number}")),
        "content_sha256": hashlib.sha256(f"{question}|{answer}".encode()).hexdigest(),
        "question": question,
        "answer": answer,
        "source": {
            "dataset": "MedQuAD",
            "relative_path": f"collection/{document_number}.xml",
            "upstream_revision": "a" * 40,
            "url": f"https://example.test/{document_number}",
            "license": "CC-BY-4.0",
        },
        "curation": {
            "status": "accepted",
            "pii_status": pii_status,
            "pii_types": ["email_address"] if pii_status == "redacted" else [],
        },
    }


def create_source(
    root: Path,
    *,
    records_by_split: dict[str, list[dict[str, Any]]] | None = None,
) -> Path:
    source = root / "dataset"
    records = records_by_split or {
        "train": [canonical_record(1), canonical_record(2, pii_status="redacted")],
        "validation": [canonical_record(3)],
        "test": [canonical_record(4)],
    }
    for split, split_records in records.items():
        write_jsonl(source / f"{split}.jsonl", split_records)
    write_json(
        source / "manifest.json",
        {
            "canonical_schema_version": "1.1",
            "dataset": {
                "name": "MedQuAD",
                "upstream_repository": "https://github.com/abachaa/MedQuAD.git",
                "upstream_revision": "a" * 40,
            },
            "summary": {
                "records_by_split": {
                    split: len(split_records) for split, split_records in records.items()
                }
            },
        },
    )
    return source


def test_preparation_generates_model_neutral_messages_and_manifest(tmp_path: Path) -> None:
    source = create_source(tmp_path)
    output = tmp_path / "sft"

    prepared = prepare_sft_dataset(source, output, system_prompt="  Safe medical assistant.  ")

    train_lines = (output / "train.jsonl").read_text(encoding="utf-8").splitlines()
    first = json.loads(train_lines[0])
    assert [message["role"] for message in first["messages"]] == [
        "system",
        "user",
        "assistant",
    ]
    assert first["messages"][0]["content"] == "Safe medical assistant."
    assert first["messages"][1]["content"] == "Question 1?"
    assert first["messages"][2]["content"] == "Answer 1."
    assert first["metadata"]["record_id"] == canonical_record(1)["record_id"]
    assert first["metadata"]["source"]["relative_path"] == "collection/1.xml"
    assert prepared.manifest["summary"] == {
        "examples": 4,
        "examples_by_split": {"train": 2, "validation": 1, "test": 1},
        "redacted_examples_by_split": {"train": 1, "validation": 0, "test": 0},
    }
    assert prepared.manifest["format"]["model_chat_template_applied"] is False
    assert prepared.manifest["format"]["metadata_is_training_text"] is False
    assert prepared.manifest["validation"] == {
        "cross_split_document_id_overlap": 0,
        "cross_split_record_id_overlap": 0,
        "cross_split_content_sha256_overlap": 0,
    }
    assert len(prepared.manifest["output_files_sha256"]["train"]) == 64
    assert (output / "manifest.json").exists()


def test_preparation_rejects_unsupported_schema(tmp_path: Path) -> None:
    records = {
        "train": [canonical_record(1)],
        "validation": [canonical_record(2)],
        "test": [canonical_record(3)],
    }
    records["train"][0]["schema_version"] = "1.0"
    source = create_source(tmp_path, records_by_split=records)

    with pytest.raises(SftPreparationError, match="não suportado"):
        prepare_sft_dataset(source, tmp_path / "sft", system_prompt="System")


def test_preparation_rejects_unresolved_pii(tmp_path: Path) -> None:
    records = {
        "train": [canonical_record(1, pii_status="not_evaluated")],
        "validation": [canonical_record(2)],
        "test": [canonical_record(3)],
    }
    source = create_source(tmp_path, records_by_split=records)

    with pytest.raises(SftPreparationError, match="PII não resolvida"):
        prepare_sft_dataset(source, tmp_path / "sft", system_prompt="System")


def test_preparation_rejects_cross_split_document_leakage(tmp_path: Path) -> None:
    records = {
        "train": [canonical_record(1, document_index=99)],
        "validation": [canonical_record(2)],
        "test": [canonical_record(3, document_index=99)],
    }
    source = create_source(tmp_path, records_by_split=records)

    with pytest.raises(SftPreparationError, match="Vazamento detectado"):
        prepare_sft_dataset(source, tmp_path / "sft", system_prompt="System")


def test_preparation_rejects_manifest_count_mismatch(tmp_path: Path) -> None:
    source = create_source(tmp_path)
    manifest_path = source / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["summary"]["records_by_split"]["train"] = 99
    write_json(manifest_path, manifest)

    with pytest.raises(SftPreparationError, match="manifesto declara 99"):
        prepare_sft_dataset(source, tmp_path / "sft", system_prompt="System")


def test_preparation_rejects_invalid_paths_and_empty_prompt(tmp_path: Path) -> None:
    with pytest.raises(SftPreparationError, match="não encontrado"):
        prepare_sft_dataset(tmp_path / "missing", tmp_path / "sft", system_prompt="System")

    source = create_source(tmp_path)
    with pytest.raises(SftPreparationError, match="diretórios diferentes"):
        prepare_sft_dataset(source, source, system_prompt="System")
    with pytest.raises(SftPreparationError, match="não pode ser vazia"):
        prepare_sft_dataset(source, tmp_path / "sft", system_prompt=" ")


def test_preparation_rejects_manifest_schema_mismatch(tmp_path: Path) -> None:
    source = create_source(tmp_path)
    manifest_path = source / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["canonical_schema_version"] = "1.0"
    write_json(manifest_path, manifest)

    with pytest.raises(SftPreparationError, match="Manifesto usa schema"):
        prepare_sft_dataset(source, tmp_path / "sft", system_prompt="System")
