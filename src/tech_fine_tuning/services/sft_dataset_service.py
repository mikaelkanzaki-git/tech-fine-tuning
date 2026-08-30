"""Derivação auditável do dataset canônico para exemplos conversacionais de SFT."""

from __future__ import annotations

import hashlib
import re
from collections.abc import Mapping
from datetime import UTC, datetime
from pathlib import Path
from typing import Any
from uuid import UUID

from tech_fine_tuning.errors import SftPreparationError
from tech_fine_tuning.integrations.dataset.jsonl import (
    file_sha256,
    read_json,
    read_jsonl,
    write_json,
    write_jsonl,
)
from tech_fine_tuning.models.dataset import (
    CanonicalTrainingRecord,
    PreparedSftDataset,
    SftExample,
    SftMessage,
    SplitName,
)

CANONICAL_SCHEMA_VERSION = "1.1"
SFT_DATASET_SCHEMA_VERSION = "1.0"
SPLITS: tuple[SplitName, ...] = ("train", "validation", "test")
_APPROVED_PII_STATUSES = {"not_detected", "redacted"}
_SHA256 = re.compile(r"^[0-9a-f]{64}$")


def _required_mapping(
    data: Mapping[str, Any],
    field: str,
    *,
    context: str,
) -> Mapping[str, Any]:
    value = data.get(field)
    if not isinstance(value, dict):
        raise SftPreparationError(f"{context}: campo {field!r} deve ser um objeto.")
    return value


def _required_string(data: Mapping[str, Any], field: str, *, context: str) -> str:
    value = data.get(field)
    if not isinstance(value, str) or not value.strip():
        raise SftPreparationError(f"{context}: campo {field!r} deve ser uma string não vazia.")
    return value


def _optional_string(data: Mapping[str, Any], field: str, *, context: str) -> str | None:
    value = data.get(field)
    if value is not None and not isinstance(value, str):
        raise SftPreparationError(f"{context}: campo {field!r} deve ser string ou null.")
    return value


def _validate_uuid(value: str, *, field: str, context: str) -> None:
    try:
        UUID(value)
    except ValueError as error:
        raise SftPreparationError(f"{context}: campo {field!r} não é um UUID válido.") from error


def _parse_canonical_record(
    data: Mapping[str, Any],
    *,
    context: str,
) -> CanonicalTrainingRecord:
    schema_version = _required_string(data, "schema_version", context=context)
    if schema_version != CANONICAL_SCHEMA_VERSION:
        raise SftPreparationError(
            f"{context}: schema canônico {schema_version!r} não suportado; "
            f"esperado {CANONICAL_SCHEMA_VERSION!r}."
        )

    record_id = _required_string(data, "record_id", context=context)
    document_id = _required_string(data, "document_id", context=context)
    content_sha256 = _required_string(data, "content_sha256", context=context)
    _validate_uuid(record_id, field="record_id", context=context)
    _validate_uuid(document_id, field="document_id", context=context)
    if not _SHA256.fullmatch(content_sha256):
        raise SftPreparationError(f"{context}: content_sha256 inválido.")

    curation = _required_mapping(data, "curation", context=context)
    if _required_string(curation, "status", context=context) != "accepted":
        raise SftPreparationError(f"{context}: somente registros aceitos podem entrar no SFT.")
    pii_status = _required_string(curation, "pii_status", context=context)
    if pii_status not in _APPROVED_PII_STATUSES:
        raise SftPreparationError(
            f"{context}: PII não resolvida; status recebido: {pii_status!r}."
        )
    pii_types = curation.get("pii_types")
    if not isinstance(pii_types, list) or any(not isinstance(item, str) for item in pii_types):
        raise SftPreparationError(f"{context}: campo 'pii_types' deve ser uma lista de strings.")
    if pii_status == "redacted" and not pii_types:
        raise SftPreparationError(f"{context}: status redacted requer ao menos um tipo de PII.")
    if pii_status == "not_detected" and pii_types:
        raise SftPreparationError(f"{context}: not_detected não aceita tipos de PII.")

    source = _required_mapping(data, "source", context=context)
    return CanonicalTrainingRecord(
        schema_version=schema_version,
        record_id=record_id,
        document_id=document_id,
        content_sha256=content_sha256,
        question=_required_string(data, "question", context=context),
        answer=_required_string(data, "answer", context=context),
        pii_status=pii_status,
        source_dataset=_required_string(source, "dataset", context=context),
        source_relative_path=_required_string(source, "relative_path", context=context),
        source_upstream_revision=_optional_string(
            source,
            "upstream_revision",
            context=context,
        ),
        source_url=_optional_string(source, "url", context=context),
        source_license=_required_string(source, "license", context=context),
    )


def _build_example(record: CanonicalTrainingRecord, system_prompt: str) -> SftExample:
    return SftExample(
        messages=(
            SftMessage(role="system", content=system_prompt),
            SftMessage(role="user", content=record.question),
            SftMessage(role="assistant", content=record.answer),
        ),
        canonical_record=record,
    )


def _overlap_count(values: Mapping[SplitName, set[str]]) -> int:
    overlap: set[str] = set()
    for index, left_name in enumerate(SPLITS):
        for right_name in SPLITS[index + 1 :]:
            overlap.update(values[left_name] & values[right_name])
    return len(overlap)


def _expected_split_counts(source_manifest: Mapping[str, Any]) -> Mapping[str, Any]:
    summary = _required_mapping(source_manifest, "summary", context="manifest canônico")
    return _required_mapping(summary, "records_by_split", context="manifest canônico")


def prepare_sft_dataset(
    source_root: Path,
    output_root: Path,
    *,
    system_prompt: str,
) -> PreparedSftDataset:
    """Valida os splits canônicos e produz mensagens sem chat template específico."""

    source_root = source_root.expanduser().resolve()
    output_root = output_root.expanduser().resolve()
    if source_root == output_root:
        raise SftPreparationError("Origem canônica e saída SFT devem ser diretórios diferentes.")
    if not source_root.exists() or not source_root.is_dir():
        raise SftPreparationError(f"Dataset canônico não encontrado: {source_root}")
    system_prompt = system_prompt.strip()
    if not system_prompt:
        raise SftPreparationError("A instrução de sistema não pode ser vazia.")

    source_manifest_path = source_root / "manifest.json"
    source_manifest = read_json(source_manifest_path)
    source_schema_version = _required_string(
        source_manifest,
        "canonical_schema_version",
        context="manifest canônico",
    )
    if source_schema_version != CANONICAL_SCHEMA_VERSION:
        raise SftPreparationError(
            f"Manifesto usa schema canônico {source_schema_version!r}; "
            f"esperado {CANONICAL_SCHEMA_VERSION!r}."
        )
    expected_counts = _expected_split_counts(source_manifest)

    records_by_split: dict[SplitName, tuple[CanonicalTrainingRecord, ...]] = {}
    input_hashes: dict[SplitName, str] = {}
    for split in SPLITS:
        input_path = source_root / f"{split}.jsonl"
        raw_records = read_jsonl(input_path)
        expected_count = expected_counts.get(split)
        if not isinstance(expected_count, int) or expected_count != len(raw_records):
            raise SftPreparationError(
                f"Split {split!r} possui {len(raw_records)} registros, mas o manifesto "
                f"declara {expected_count!r}."
            )
        records_by_split[split] = tuple(
            _parse_canonical_record(record, context=f"{input_path}, linha {line_number}")
            for line_number, record in enumerate(raw_records, start=1)
        )
        input_hashes[split] = file_sha256(input_path)

    identity_sets = {
        "document_id": {
            split: {record.document_id for record in records}
            for split, records in records_by_split.items()
        },
        "record_id": {
            split: {record.record_id for record in records}
            for split, records in records_by_split.items()
        },
        "content_sha256": {
            split: {record.content_sha256 for record in records}
            for split, records in records_by_split.items()
        },
    }
    validation = {
        f"cross_split_{identity}_overlap": _overlap_count(values)
        for identity, values in identity_sets.items()
    }
    if any(validation.values()):
        raise SftPreparationError(f"Vazamento detectado nos splits canônicos: {validation}")

    output_hashes: dict[SplitName, str] = {}
    examples_by_split: dict[str, int] = {}
    redacted_by_split: dict[str, int] = {}
    for split, records in records_by_split.items():
        examples = tuple(_build_example(record, system_prompt) for record in records)
        output_path = output_root / f"{split}.jsonl"
        write_jsonl(output_path, (example.as_dict() for example in examples))
        output_hashes[split] = file_sha256(output_path)
        examples_by_split[split] = len(examples)
        redacted_by_split[split] = sum(
            record.pii_status == "redacted" for record in records
        )

    dataset_source = _required_mapping(source_manifest, "dataset", context="manifest canônico")
    manifest: dict[str, Any] = {
        "schema_version": SFT_DATASET_SCHEMA_VERSION,
        "generated_at": datetime.now(UTC).isoformat(),
        "source": {
            "canonical_schema_version": source_schema_version,
            "canonical_manifest_sha256": file_sha256(source_manifest_path),
            "dataset": dataset_source.get("name"),
            "upstream_repository": dataset_source.get("upstream_repository"),
            "upstream_revision": dataset_source.get("upstream_revision"),
            "input_files_sha256": input_hashes,
        },
        "format": {
            "type": "conversational_messages",
            "message_roles": ["system", "user", "assistant"],
            "model_chat_template_applied": False,
            "metadata_is_training_text": False,
            "system_prompt": system_prompt,
            "system_prompt_sha256": hashlib.sha256(system_prompt.encode("utf-8")).hexdigest(),
        },
        "summary": {
            "examples": sum(examples_by_split.values()),
            "examples_by_split": examples_by_split,
            "redacted_examples_by_split": redacted_by_split,
        },
        "output_files_sha256": output_hashes,
        "validation": validation,
    }
    write_json(output_root / "manifest.json", manifest)
    return PreparedSftDataset(manifest=manifest)
