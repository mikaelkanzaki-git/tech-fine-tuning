"""Auditoria reproduzível de qualidade e isolamento dos splits SFT."""

from __future__ import annotations

import math
import re
from collections import Counter, defaultdict
from collections.abc import Mapping
from datetime import UTC, datetime
from itertools import combinations
from pathlib import Path
from typing import Any, cast

from tech_fine_tuning.errors import SftAuditExecutionError, SftAuditValidationError
from tech_fine_tuning.integrations.dataset.jsonl import (
    file_sha256,
    read_json,
    read_jsonl,
    write_json,
    write_jsonl,
)
from tech_fine_tuning.models.audit import SftAuditExample, SftAuditOutcome
from tech_fine_tuning.models.dataset import SplitName
from tech_fine_tuning.services.text_normalization import normalize_text

SPLITS: tuple[SplitName, ...] = ("train", "validation", "test")
HPO_BOILERPLATE = "has collected information on how often a sign or symptom occurs"
RESOURCE_NAVIGATION_PREFIX = "these resources address the diagnosis or management"
_SENTENCE_BOUNDARY = re.compile(r"(?<=[.!?])\s+")


def _mapping(value: Any, *, context: str) -> Mapping[str, Any]:
    if not isinstance(value, dict):
        raise SftAuditValidationError(f"{context} deve ser um objeto.")
    return value


def _string(value: Any, *, context: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise SftAuditValidationError(f"{context} deve ser uma string não vazia.")
    return value.strip()


def _maximum_sentence_repetitions(answer: str) -> int:
    sentences = [
        normalize_text(sentence)
        for sentence in _SENTENCE_BOUNDARY.split(answer)
        if len(normalize_text(sentence)) >= 30
    ]
    return max(Counter(sentences).values(), default=1)


def _parse_example(data: Mapping[str, Any], *, split: SplitName, line: int) -> SftAuditExample:
    context = f"{split}.jsonl, linha {line}"
    messages = data.get("messages")
    if not isinstance(messages, list) or len(messages) != 3:
        raise SftAuditValidationError(f"{context}: messages deve conter três mensagens.")
    expected_roles = ("system", "user", "assistant")
    contents: list[str] = []
    for index, expected_role in enumerate(expected_roles):
        message = _mapping(messages[index], context=f"{context}.messages[{index}]")
        if message.get("role") != expected_role:
            raise SftAuditValidationError(
                f"{context}.messages[{index}].role deve ser {expected_role!r}."
            )
        contents.append(
            _string(message.get("content"), context=f"{context}.messages[{index}].content")
        )
    system_prompt, question, answer = contents
    metadata = _mapping(data.get("metadata"), context=f"{context}.metadata")
    source = _mapping(metadata.get("source"), context=f"{context}.metadata.source")
    normalized_question = normalize_text(question)
    normalized_answer = normalize_text(answer)
    return SftAuditExample(
        split=split,
        record_id=_string(metadata.get("record_id"), context=f"{context}.metadata.record_id"),
        question=question,
        normalized_question=normalized_question,
        normalized_answer=normalized_answer,
        source_relative_path=_string(
            source.get("relative_path"), context=f"{context}.metadata.source.relative_path"
        ),
        answer_chars=len(answer),
        estimated_total_tokens=math.ceil(
            (len(system_prompt) + len(question) + len(answer)) / 4
        ),
        has_hpo_boilerplate=HPO_BOILERPLATE in normalized_answer,
        is_resource_navigation=normalized_answer.startswith(RESOURCE_NAVIGATION_PREFIX),
        answer_repeats_question=normalized_answer.startswith(normalized_question),
        maximum_sentence_repetitions=_maximum_sentence_repetitions(answer),
    )


def _validate_source(source_path: Path) -> tuple[dict[str, Any], dict[SplitName, Path]]:
    if not source_path.exists() or not source_path.is_dir():
        raise SftAuditValidationError(f"Dataset SFT não encontrado: {source_path}")
    manifest_path = source_path / "manifest.json"
    manifest = read_json(manifest_path)
    if manifest.get("schema_version") != "1.0":
        raise SftAuditValidationError("O manifesto SFT deve usar schema_version '1.0'.")
    summary = _mapping(manifest.get("summary"), context="manifest.summary")
    expected_counts = _mapping(
        summary.get("examples_by_split"), context="manifest.summary.examples_by_split"
    )
    expected_hashes = _mapping(
        manifest.get("output_files_sha256"), context="manifest.output_files_sha256"
    )
    split_paths: dict[SplitName, Path] = {}
    for split in SPLITS:
        split_path = source_path / f"{split}.jsonl"
        expected_hash = expected_hashes.get(split)
        if not isinstance(expected_hash, str) or file_sha256(split_path) != expected_hash:
            raise SftAuditValidationError(
                f"O hash do split {split!r} não corresponde ao manifesto."
            )
        expected_count = expected_counts.get(split)
        if isinstance(expected_count, bool) or not isinstance(expected_count, int):
            raise SftAuditValidationError(
                f"manifest.summary.examples_by_split.{split} deve ser um inteiro."
            )
        split_paths[split] = split_path
    return manifest, split_paths


def _read_examples(
    split_paths: Mapping[SplitName, Path], manifest: Mapping[str, Any]
) -> dict[SplitName, tuple[SftAuditExample, ...]]:
    expected_counts = cast(
        Mapping[str, Any],
        cast(Mapping[str, Any], manifest["summary"])["examples_by_split"],
    )
    result: dict[SplitName, tuple[SftAuditExample, ...]] = {}
    seen_record_ids: set[str] = set()
    for split in SPLITS:
        raw_records = read_jsonl(split_paths[split])
        if len(raw_records) != expected_counts[split]:
            raise SftAuditValidationError(
                f"Split {split!r} possui {len(raw_records)} registros; "
                f"o manifesto declara {expected_counts[split]}."
            )
        examples = tuple(
            _parse_example(record, split=split, line=line)
            for line, record in enumerate(raw_records, start=1)
        )
        for example in examples:
            if example.record_id in seen_record_ids:
                raise SftAuditValidationError(
                    f"record_id aparece em mais de um exemplo: {example.record_id}."
                )
            seen_record_ids.add(example.record_id)
        result[split] = examples
    return result


def _percentile(values: list[int], percentile: float) -> int:
    ordered = sorted(values)
    return ordered[math.floor((len(ordered) - 1) * percentile)]


def _build_audit(
    examples_by_split: Mapping[SplitName, tuple[SftAuditExample, ...]],
    *,
    max_sequence_length: int,
) -> tuple[dict[str, Any], tuple[dict[str, Any], ...]]:
    issues: dict[tuple[SplitName, str], set[str]] = defaultdict(set)
    question_groups: dict[SplitName, dict[str, list[SftAuditExample]]] = {}
    answer_groups: dict[SplitName, dict[str, list[SftAuditExample]]] = {}

    for split, examples in examples_by_split.items():
        by_question: dict[str, list[SftAuditExample]] = defaultdict(list)
        by_answer: dict[str, list[SftAuditExample]] = defaultdict(list)
        for example in examples:
            key = (split, example.record_id)
            by_question[example.normalized_question].append(example)
            by_answer[example.normalized_answer].append(example)
            if example.has_hpo_boilerplate:
                issues[key].add("hpo_boilerplate")
            if example.is_resource_navigation:
                issues[key].add("resource_navigation_answer")
            if example.answer_repeats_question:
                issues[key].add("answer_repeats_question")
            if example.maximum_sentence_repetitions >= 3:
                issues[key].add("repeated_sentence")
            if example.estimated_total_tokens > max_sequence_length:
                issues[key].add("estimated_context_overflow")
        for grouped in by_question.values():
            if len(grouped) > 1:
                for example in grouped:
                    issues[(split, example.record_id)].add("duplicate_question_within_split")
        for grouped in by_answer.values():
            if len(grouped) > 1:
                for example in grouped:
                    issues[(split, example.record_id)].add("duplicate_answer_within_split")
        question_groups[split] = by_question
        answer_groups[split] = by_answer

    cross_split_overlaps: dict[str, int] = {}
    for left, right in combinations(SPLITS, 2):
        overlap = set(question_groups[left]) & set(question_groups[right])
        cross_split_overlaps[f"{left}_{right}"] = len(overlap)
        for question in overlap:
            for example in (*question_groups[left][question], *question_groups[right][question]):
                issues[(example.split, example.record_id)].add("question_overlap_across_splits")

    issue_records: list[dict[str, Any]] = []
    split_summaries: dict[str, Any] = {}
    total_issue_counts: Counter[str] = Counter()
    for split in SPLITS:
        examples = examples_by_split[split]
        issue_counts: Counter[str] = Counter()
        for example in examples:
            example_issues = sorted(issues[(split, example.record_id)])
            issue_counts.update(example_issues)
            if example_issues:
                issue_records.append(
                    {
                        "split": split,
                        "record_id": example.record_id,
                        "question": example.question,
                        "source_relative_path": example.source_relative_path,
                        "answer_chars": example.answer_chars,
                        "estimated_total_tokens": example.estimated_total_tokens,
                        "maximum_sentence_repetitions": example.maximum_sentence_repetitions,
                        "issues": example_issues,
                    }
                )
        total_issue_counts.update(issue_counts)
        answer_lengths = [example.answer_chars for example in examples]
        split_summaries[split] = {
            "examples": len(examples),
            "records_with_issues": sum(
                bool(issues[(split, example.record_id)]) for example in examples
            ),
            "answer_chars": {
                "p50": _percentile(answer_lengths, 0.50),
                "p95": _percentile(answer_lengths, 0.95),
                "p99": _percentile(answer_lengths, 0.99),
                "maximum": max(answer_lengths),
            },
            "issue_counts": dict(sorted(issue_counts.items())),
        }

    all_examples = [example for values in examples_by_split.values() for example in values]
    summary = {
        "schema_version": "1.0",
        "examples": len(all_examples),
        "max_sequence_length": max_sequence_length,
        "token_estimate": "ceil(total_message_characters / 4)",
        "records_with_issues": len(issue_records),
        "issue_counts": dict(sorted(total_issue_counts.items())),
        "cross_split_normalized_question_overlap": cross_split_overlaps,
        "splits": split_summaries,
    }
    return summary, tuple(issue_records)


def _report_markdown(summary: Mapping[str, Any]) -> str:
    splits = cast(Mapping[str, Mapping[str, Any]], summary["splits"])
    overlaps = cast(Mapping[str, int], summary["cross_split_normalized_question_overlap"])
    issue_counts = cast(Mapping[str, int], summary["issue_counts"])
    lines = [
        "# Auditoria do dataset SFT",
        "",
        f"Exemplos analisados: {summary['examples']}.",
        f"Registros com ao menos um alerta: {summary['records_with_issues']}.",
        "",
        "## Comprimento e alertas por split",
        "",
        "| Split | Exemplos | Com alerta | P50 caracteres | P95 | P99 | Máximo |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for split in SPLITS:
        values = splits[split]
        lengths = cast(Mapping[str, int], values["answer_chars"])
        lines.append(
            f"| {split} | {values['examples']} | {values['records_with_issues']} | "
            f"{lengths['p50']} | {lengths['p95']} | {lengths['p99']} | "
            f"{lengths['maximum']} |"
        )
    lines.extend(["", "## Alertas", ""])
    for issue, count in issue_counts.items():
        lines.append(f"- `{issue}`: {count}")
    lines.extend(
        [
            "",
            "## Sobreposição de perguntas entre splits",
            "",
            f"- `train` x `validation`: {overlaps['train_validation']}",
            f"- `train` x `test`: {overlaps['train_test']}",
            f"- `validation` x `test`: {overlaps['validation_test']}",
            "",
            "A estimativa de tokens usa quatro caracteres por token e serve apenas como triagem. "
            "A validação exata deve usar o tokenizer fixado na configuração do modelo.",
            "",
        ]
    )
    return "\n".join(lines)


def audit_sft_dataset(
    *, source_path: Path, output_path: Path, max_sequence_length: int
) -> SftAuditOutcome:
    """Inspeciona o SFT sem alterar exemplos ou splits."""

    if max_sequence_length <= 0:
        raise SftAuditValidationError("max_sequence_length deve ser positivo.")
    source_path = source_path.expanduser().resolve()
    output_path = output_path.expanduser().resolve()
    if source_path == output_path:
        raise SftAuditValidationError("Origem SFT e saída da auditoria devem ser diferentes.")
    if output_path.exists() and any(output_path.iterdir()):
        raise SftAuditValidationError(f"A saída da auditoria não está vazia: {output_path}")
    manifest, split_paths = _validate_source(source_path)
    examples_by_split = _read_examples(split_paths, manifest)
    summary, issue_records = _build_audit(
        examples_by_split, max_sequence_length=max_sequence_length
    )

    summary_path = output_path / "audit-summary.json"
    issues_path = output_path / "issues.jsonl"
    report_path = output_path / "report.md"
    audit_manifest_path = output_path / "audit-manifest.json"
    try:
        output_path.mkdir(parents=True, exist_ok=True)
        write_json(summary_path, summary)
        write_jsonl(issues_path, issue_records)
        report_path.write_text(_report_markdown(summary), encoding="utf-8", newline="\n")
        inputs_sha256: dict[str, str] = {
            "sft_manifest": file_sha256(source_path / "manifest.json"),
            **{str(split): file_sha256(path) for split, path in split_paths.items()},
        }
        audit_manifest = {
            "schema_version": "1.0",
            "status": "completed",
            "generated_at": datetime.now(UTC).isoformat(),
            "inputs_sha256": inputs_sha256,
            "outputs_sha256": {
                "summary": file_sha256(summary_path),
                "issues": file_sha256(issues_path),
                "report": file_sha256(report_path),
            },
        }
        write_json(audit_manifest_path, audit_manifest)
    except (OSError, UnicodeError) as error:
        raise SftAuditExecutionError(f"Não foi possível persistir a auditoria: {error}") from error
    return SftAuditOutcome(
        output_path=output_path,
        summary_path=summary_path,
        issues_path=issues_path,
        report_path=report_path,
        manifest_path=audit_manifest_path,
        summary=summary,
    )
