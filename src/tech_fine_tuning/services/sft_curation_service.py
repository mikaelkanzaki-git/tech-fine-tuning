"""Curadoria conservadora e reproduzível do dataset conversacional."""

from __future__ import annotations

import copy
from collections import Counter
from collections.abc import Mapping
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, cast

from tech_fine_tuning.errors import (
    SftCurationExecutionError,
    SftCurationValidationError,
)
from tech_fine_tuning.integrations.dataset.jsonl import (
    file_sha256,
    read_json,
    read_jsonl,
    write_json,
    write_jsonl,
)
from tech_fine_tuning.models.curation import SftCurationOutcome
from tech_fine_tuning.models.dataset import SplitName
from tech_fine_tuning.services.text_normalization import normalize_text

SPLITS: tuple[SplitName, ...] = ("train", "validation", "test")
CURATION_PROFILE = "medquad_conservative_v1"
HPO_BOILERPLATE_START = (
    "the human phenotype ontology (hpo) has collected information on how often "
    "a sign or symptom occurs"
)
RESOURCE_NAVIGATION_PREFIX = "these resources address the diagnosis or management"


def _mapping(value: Any, *, context: str) -> Mapping[str, Any]:
    if not isinstance(value, dict):
        raise SftCurationValidationError(f"{context} deve ser um objeto.")
    return value


def _string(value: Any, *, context: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise SftCurationValidationError(f"{context} deve ser uma string não vazia.")
    return value.strip()


def _validate_example(data: Mapping[str, Any], *, split: SplitName, line: int) -> None:
    context = f"{split}.jsonl, linha {line}"
    messages = data.get("messages")
    if not isinstance(messages, list) or len(messages) != 3:
        raise SftCurationValidationError(f"{context}: messages deve conter três mensagens.")
    for index, role in enumerate(("system", "user", "assistant")):
        message = _mapping(messages[index], context=f"{context}.messages[{index}]")
        if message.get("role") != role:
            raise SftCurationValidationError(
                f"{context}.messages[{index}].role deve ser {role!r}."
            )
        _string(message.get("content"), context=f"{context}.messages[{index}].content")
    metadata = _mapping(data.get("metadata"), context=f"{context}.metadata")
    _string(metadata.get("record_id"), context=f"{context}.metadata.record_id")


def _validate_source(
    source_path: Path,
) -> tuple[dict[str, Any], dict[SplitName, tuple[dict[str, Any], ...]]]:
    if not source_path.exists() or not source_path.is_dir():
        raise SftCurationValidationError(f"Dataset SFT não encontrado: {source_path}")
    manifest = read_json(source_path / "manifest.json")
    if manifest.get("schema_version") != "1.0":
        raise SftCurationValidationError("O manifesto SFT deve usar schema_version '1.0'.")
    summary = _mapping(manifest.get("summary"), context="manifest.summary")
    expected_counts = _mapping(
        summary.get("examples_by_split"), context="manifest.summary.examples_by_split"
    )
    expected_hashes = _mapping(
        manifest.get("output_files_sha256"), context="manifest.output_files_sha256"
    )
    records_by_split: dict[SplitName, tuple[dict[str, Any], ...]] = {}
    for split in SPLITS:
        split_path = source_path / f"{split}.jsonl"
        expected_hash = expected_hashes.get(split)
        if not isinstance(expected_hash, str) or file_sha256(split_path) != expected_hash:
            raise SftCurationValidationError(
                f"O hash do split {split!r} não corresponde ao manifesto."
            )
        records = read_jsonl(split_path)
        if expected_counts.get(split) != len(records):
            raise SftCurationValidationError(
                f"A contagem do split {split!r} não corresponde ao manifesto."
            )
        for line, record in enumerate(records, start=1):
            _validate_example(record, split=split, line=line)
        records_by_split[split] = records
    return manifest, records_by_split


def _remove_leading_question(question: str, answer: str) -> tuple[str, bool]:
    normalized_question = normalize_text(question)
    minimum = max(1, len(question) - 16)
    maximum = min(len(answer), len(question) + 32)
    for end in range(minimum, maximum + 1):
        if normalize_text(answer[:end]) != normalized_question:
            continue
        following = answer[end : end + 1]
        if following and following not in " \t\r\n:-":
            continue
        remaining = answer[end:].lstrip(" \t\r\n:-")
        return (remaining, True) if remaining else (answer, False)
    return answer, False


def _remove_hpo_boilerplate(answer: str) -> tuple[str, bool]:
    marker_index = answer.casefold().find(HPO_BOILERPLATE_START)
    if marker_index < 0:
        return answer, False
    remaining = answer[:marker_index].rstrip(" \t\r\n-")
    return remaining, True


def _curate_record(
    raw_record: Mapping[str, Any],
    *,
    split: SplitName,
) -> tuple[dict[str, Any] | None, dict[str, Any] | None, tuple[str, ...]]:
    record = copy.deepcopy(dict(raw_record))
    messages = cast(list[dict[str, Any]], record["messages"])
    question = cast(str, messages[1]["content"]).strip()
    answer = cast(str, messages[2]["content"]).strip()
    metadata = cast(dict[str, Any], record["metadata"])
    record_id = cast(str, metadata["record_id"])
    transformations: list[str] = []

    answer, question_removed = _remove_leading_question(question, answer)
    if question_removed:
        transformations.append("leading_question_removed")
    if normalize_text(answer).startswith(RESOURCE_NAVIGATION_PREFIX):
        return None, {
            "record_id": record_id,
            "split": split,
            "reasons": ["resource_navigation_answer"],
        }, tuple(transformations)
    answer, hpo_removed = _remove_hpo_boilerplate(answer)
    if hpo_removed:
        transformations.append("hpo_boilerplate_removed")
    if not answer.strip():
        return None, {
            "record_id": record_id,
            "split": split,
            "reasons": ["empty_after_transform"],
        }, tuple(transformations)

    messages[2]["content"] = answer.strip()
    metadata["curation"] = {
        "profile": CURATION_PROFILE,
        "transformations": transformations,
    }
    return record, None, tuple(transformations)


def _cross_split_question_overlap(
    curated_by_split: Mapping[SplitName, tuple[dict[str, Any], ...]],
) -> int:
    question_sets: dict[SplitName, set[str]] = {}
    for split, records in curated_by_split.items():
        question_sets[split] = {
            normalize_text(cast(str, cast(list[dict[str, Any]], record["messages"])[1]["content"]))
            for record in records
        }
    return len(
        (question_sets["train"] & question_sets["validation"])
        | (question_sets["train"] & question_sets["test"])
        | (question_sets["validation"] & question_sets["test"])
    )


def curate_sft_dataset(source_path: Path, output_path: Path) -> SftCurationOutcome:
    """Remove respostas não instrutivas e boilerplate sem sintetizar fatos médicos."""

    source_path = source_path.expanduser().resolve()
    output_path = output_path.expanduser().resolve()
    if source_path == output_path:
        raise SftCurationValidationError("Origem e saída da curadoria devem ser diferentes.")
    if output_path.exists() and any(output_path.iterdir()):
        raise SftCurationValidationError(f"O diretório de saída não está vazio: {output_path}")
    source_manifest, records_by_split = _validate_source(source_path)

    curated_by_split: dict[SplitName, tuple[dict[str, Any], ...]] = {}
    excluded: list[dict[str, Any]] = []
    transformation_counts: Counter[str] = Counter()
    exclusion_counts: Counter[str] = Counter()
    for split, records in records_by_split.items():
        curated: list[dict[str, Any]] = []
        for record in records:
            curated_record, excluded_record, transformations = _curate_record(
                record, split=split
            )
            transformation_counts.update(transformations)
            if excluded_record is not None:
                excluded.append(excluded_record)
                exclusion_counts.update(cast(list[str], excluded_record["reasons"]))
            elif curated_record is not None:
                curated.append(curated_record)
        curated_by_split[split] = tuple(curated)

    overlap = _cross_split_question_overlap(curated_by_split)
    if overlap:
        raise SftCurationValidationError(
            f"A origem ainda contém {overlap} perguntas normalizadas em mais de um split."
        )

    try:
        output_hashes: dict[str, str] = {}
        for split, records in curated_by_split.items():
            split_path = output_path / f"{split}.jsonl"
            write_jsonl(split_path, records)
            output_hashes[split] = file_sha256(split_path)
        excluded_path = output_path / "excluded.jsonl"
        write_jsonl(excluded_path, excluded)
        summary = {
            "input_examples": sum(len(records) for records in records_by_split.values()),
            "output_examples": sum(len(records) for records in curated_by_split.values()),
            "excluded_examples": len(excluded),
            "examples_by_split": {
                split: len(records) for split, records in curated_by_split.items()
            },
            "transformations": dict(sorted(transformation_counts.items())),
            "exclusions": dict(sorted(exclusion_counts.items())),
        }
        manifest = {
            "schema_version": "1.0",
            "generated_at": datetime.now(UTC).isoformat(),
            "source": {
                "sft_manifest_sha256": file_sha256(source_path / "manifest.json"),
                "input_files_sha256": {
                    split: file_sha256(source_path / f"{split}.jsonl") for split in SPLITS
                },
            },
            "format": source_manifest.get("format"),
            "curation": {
                "profile": CURATION_PROFILE,
                "medical_text_policy": "remove_only_no_medical_facts_synthesized",
                "rules": [
                    "remove_exact_leading_question",
                    "exclude_resource_navigation_answer",
                    "remove_hpo_explanatory_boilerplate",
                ],
                "excluded_file": "excluded.jsonl",
                "excluded_file_sha256": file_sha256(excluded_path),
            },
            "summary": {
                "examples": summary["output_examples"],
                "examples_by_split": summary["examples_by_split"],
                "curation": summary,
            },
            "output_files_sha256": output_hashes,
            "validation": {"cross_split_normalized_question_overlap": overlap},
        }
        manifest_path = output_path / "manifest.json"
        write_json(manifest_path, manifest)
    except OSError as error:
        raise SftCurationExecutionError(
            f"Não foi possível gravar o dataset curado em {output_path}: {error}"
        ) from error

    return SftCurationOutcome(
        output_path=output_path,
        manifest_path=manifest_path,
        excluded_path=excluded_path,
        summary=summary,
    )
