from __future__ import annotations

import json
from pathlib import Path

import pytest

from tech_fine_tuning.errors import SftAuditValidationError
from tech_fine_tuning.integrations.dataset.jsonl import file_sha256, write_json, write_jsonl
from tech_fine_tuning.services.sft_audit_service import audit_sft_dataset


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
                "train-1",
                "Shared question?",
                (
                    "Shared question? Answer. The Human Phenotype Ontology (HPO) has "
                    "collected information on how often a sign or symptom occurs."
                ),
            ),
            _example(
                "train-2",
                "Duplicate question?",
                "Repeated sentence with enough characters. " * 3,
            ),
            _example("train-3", "Duplicate question?", "Different answer."),
        ],
        "validation": [
            _example(
                "validation-1",
                " shared   QUESTION? ",
                "These resources address the diagnosis or management of a condition.",
            )
        ],
        "test": [_example("test-1", "Long question?", "x" * 5000)],
    }
    hashes: dict[str, str] = {}
    for split, examples in records.items():
        split_path = source / f"{split}.jsonl"
        write_jsonl(split_path, examples)
        hashes[split] = file_sha256(split_path)
    write_json(
        source / "manifest.json",
        {
            "schema_version": "1.0",
            "summary": {
                "examples": 5,
                "examples_by_split": {
                    split: len(examples) for split, examples in records.items()
                },
            },
            "output_files_sha256": hashes,
        },
    )
    return source


def test_audit_reports_quality_and_cross_split_risks(tmp_path: Path) -> None:
    source = _write_sft(tmp_path)

    outcome = audit_sft_dataset(
        source_path=source,
        output_path=tmp_path / "audit",
        max_sequence_length=1000,
    )

    summary = outcome.summary
    assert summary["examples"] == 5
    assert summary["cross_split_normalized_question_overlap"]["train_validation"] == 1
    assert summary["issue_counts"]["hpo_boilerplate"] == 1
    assert summary["issue_counts"]["resource_navigation_answer"] == 1
    assert summary["issue_counts"]["estimated_context_overflow"] == 1
    assert summary["issue_counts"]["repeated_sentence"] == 1
    assert summary["issue_counts"]["duplicate_question_within_split"] == 2
    issue_records = [
        json.loads(line)
        for line in outcome.issues_path.read_text(encoding="utf-8").splitlines()
    ]
    train_one = next(record for record in issue_records if record["record_id"] == "train-1")
    assert "answer_repeats_question" in train_one["issues"]
    assert "question_overlap_across_splits" in train_one["issues"]
    assert "# Auditoria do dataset SFT" in outcome.report_path.read_text(encoding="utf-8")
    manifest = json.loads(outcome.manifest_path.read_text(encoding="utf-8"))
    assert manifest["status"] == "completed"
    assert manifest["outputs_sha256"]["summary"] == file_sha256(outcome.summary_path)


def test_audit_rejects_tampered_split(tmp_path: Path) -> None:
    source = _write_sft(tmp_path)
    with (source / "train.jsonl").open("a", encoding="utf-8") as stream:
        stream.write("\n")

    with pytest.raises(SftAuditValidationError, match="hash do split"):
        audit_sft_dataset(
            source_path=source,
            output_path=tmp_path / "audit",
            max_sequence_length=2048,
        )


def test_audit_rejects_invalid_context_limit_and_nonempty_output(tmp_path: Path) -> None:
    source = _write_sft(tmp_path)
    with pytest.raises(SftAuditValidationError, match="deve ser positivo"):
        audit_sft_dataset(
            source_path=source,
            output_path=tmp_path / "audit",
            max_sequence_length=0,
        )

    output = tmp_path / "audit"
    output.mkdir()
    (output / "keep.txt").write_text("keep", encoding="utf-8")
    with pytest.raises(SftAuditValidationError, match="não está vazia"):
        audit_sft_dataset(
            source_path=source,
            output_path=output,
            max_sequence_length=2048,
        )


def test_audit_rejects_invalid_message_contract(tmp_path: Path) -> None:
    source = _write_sft(tmp_path)
    records = [
        _example("train-1", "Question?", "Answer."),
        _example("train-2", "Question 2?", "Answer 2."),
        _example("train-3", "Question 3?", "Answer 3."),
    ]
    records[0]["messages"] = []
    write_jsonl(source / "train.jsonl", records)
    manifest = json.loads((source / "manifest.json").read_text(encoding="utf-8"))
    manifest["output_files_sha256"]["train"] = file_sha256(source / "train.jsonl")
    write_json(source / "manifest.json", manifest)

    with pytest.raises(SftAuditValidationError, match="messages deve conter três"):
        audit_sft_dataset(
            source_path=source,
            output_path=tmp_path / "audit",
            max_sequence_length=2048,
        )
